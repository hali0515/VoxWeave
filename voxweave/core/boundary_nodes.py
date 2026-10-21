"""The lattice node sets: barriers, candidate nodes, relief and sentence ends.

Everything that decides which atom edges may or must be cuts. It depends on the
atom layer in ``boundary_atoms`` and never on the lattice builder, so
``boundary_lattice`` can import it and re-export every name here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .boundary_atoms import (
    BARRIER_UNCERTAINTY_MS,
    RELIEF_TRIGGER_FACTOR,
    AtomLayer,
    LatticeAtom,
    _real,
    _reindex,
    span_max,
    span_min,
    unit_edge_nodes,
)
from .layout import (
    _join,
    _line_budget_width,
    _no_spaces,
    _token_char_count,
    _vis_width,
)
from .segdoc import DisplayProfile, SourceUnit
from .smart_split import (
    _display_chars,
    _segment_sentences,
    _snap_sentence_breaks,
    _tokens,
)


# --------------------------------------------------------------- C4 barriers


@dataclass(frozen=True)
class HardBarrier:
    """A cut point no partition may cross: a document end or a robust silence."""

    node: int
    unit_id: int
    kind: str
    gap_ms: float | None


def build_barriers(
    layer: AtomLayer, profile: DisplayProfile
) -> tuple[HardBarrier, ...]:
    """The two document ends plus every inter-atom gap that is robustly silent.

    "Robustly" means the gap clears ``vad_skip_ms`` by more than the measurement
    uncertainty and *both* of its bounds are real -- a gap next to a ghost unit
    is unmeasured, not long.

    A barrier is a cut like any other, so it is restricted to the same unit-edge
    node space (see :func:`unit_edge_nodes`): an interval boundary interior to a
    source unit would hand two intervals overlapping ownership of it. Today the
    restriction never fires on its own -- atoms of one unit share that unit's
    span, so the gap between them measures zero -- but that is a property of the
    atom builder rather than of this rule, and the topology must not depend on it.

    DECLARED POLICY DELTA: this ignores v1's phrase gate and its sticky-token
    suppression, so a no-space language can get a barrier mid-phrase where v1
    would have crossed. That is a deliberate v2 choice, recorded in the artifact
    rather than inherited as truth.
    """
    threshold = profile.vad_skip_ms + BARRIER_UNCERTAINTY_MS
    barriers = [
        HardBarrier(node=0, unit_id=layer.unit_bound(0), kind="document", gap_ms=None)
    ]
    representable = set(unit_edge_nodes(layer.atoms))
    for node in range(1, len(layer.atoms)):
        if node not in representable:
            continue
        left_end = layer.atoms[node - 1].end
        right_start = layer.atoms[node].start
        if left_end is None or right_start is None:
            continue
        if not (_real(left_end) and _real(right_start)):
            continue
        gap_ms = max(0.0, (right_start - left_end) * 1000.0)
        if gap_ms >= threshold:
            barriers.append(
                HardBarrier(
                    node=node,
                    unit_id=layer.unit_bound(node),
                    kind="robust-silence",
                    gap_ms=gap_ms,
                )
            )
    last = len(layer.atoms)
    if last != 0:
        barriers.append(
            HardBarrier(
                node=last, unit_id=layer.unit_bound(last), kind="document", gap_ms=None
            )
        )
    seen: dict[int, HardBarrier] = {}
    for barrier in barriers:
        seen.setdefault(barrier.node, barrier)
    return tuple(seen[node] for node in sorted(seen))


@dataclass(frozen=True)
class HardInterval:
    """One barrier-to-barrier region, solved independently of every other."""

    index: int
    node_start: int
    node_end: int
    unit_start: int
    unit_end: int
    left: HardBarrier
    right: HardBarrier

    @property
    def atom_count(self) -> int:
        return self.node_end - self.node_start


def build_intervals(
    layer: AtomLayer, barriers: Sequence[HardBarrier]
) -> tuple[HardInterval, ...]:
    """Consecutive barrier pairs; empty regions are dropped, the rest tile."""
    out: list[HardInterval] = []
    for left, right in zip(barriers, barriers[1:]):
        if left.node == right.node:
            continue
        out.append(
            HardInterval(
                index=len(out),
                node_start=left.node,
                node_end=right.node,
                unit_start=layer.unit_bound(left.node),
                unit_end=layer.unit_bound(right.node),
                left=left,
                right=right,
            )
        )
    return tuple(out)


# ------------------------------------------------------------ C1 node space


def candidate_nodes(atoms: Sequence[LatticeAtom], lang: str) -> tuple[int, ...]:
    """Atom edges a cue boundary may sit on.

    Space-delimited languages may break between any two words. No-space
    languages may break only at a phrase start, plus wherever an overlong
    embedded Latin run was forced to expose its internal whitespace -- that is a
    permission the packer needs, not a mandate to use it.

    Both readings are then intersected with :func:`unit_edge_nodes`, because a
    linguistically attractive break interior to a source unit is still one the
    partition cannot express. Where the aligner emitted one unit per atom the
    intersection removes nothing; where it emitted coarser units it removes
    exactly the boundaries that had no unit-level evidence behind them.
    """
    count = len(atoms)
    nodes = {0, count}
    if not _no_spaces(lang):
        nodes.update(range(count))
    else:
        for index, atom in enumerate(atoms):
            if atom.phrase_start or atom.forced_boundary:
                nodes.add(index)
    nodes &= set(unit_edge_nodes(atoms))
    nodes |= {0, count}
    return tuple(sorted(nodes))


def split_candidate_at_unit(
    atoms: Sequence[LatticeAtom],
    unit_id: int,
    units: Sequence[SourceUnit],
    lang: str,
) -> tuple[LatticeAtom, ...]:
    """Expose one source-unit boundary that an atom is currently hiding.

    This is the relief seam: an atom that owns several units has internal
    boundaries the packer cannot see, and this is the only sanctioned way to
    make one of them a candidate. Splitting is by footprint, so the pieces'
    surfaces and spans come from the units themselves rather than from a
    re-derived cursor into the atom's own text.

    Penalty transport is conservative on purpose: the newly exposed edge carries
    no linguistic evidence at all (zero on both sides of it), while the parent's
    start-side penalties stay with the left piece and its end-side penalty with
    the right. Inventing a penalty for a boundary no analyzer ever scored would
    be worse than scoring it neutral.
    """
    target = next(
        (i for i, a in enumerate(atoms) if a.unit_start < unit_id < a.unit_end), None
    )
    if target is None:
        raise ValueError(
            f"unit {unit_id} is not strictly inside any atom footprint: "
            "the boundary is already exposed or out of range"
        )
    parent = atoms[target]
    left = _piece_from_units(
        parent,
        parent.unit_start,
        unit_id,
        units,
        lang,
        end_pen=0,
        start_pen=parent.start_pen,
        boundary_pen=parent.boundary_pen,
        phrase_start=parent.phrase_start,
        forced_boundary=False,
        members=parent.members,
    )
    right = _piece_from_units(
        parent,
        unit_id,
        parent.unit_end,
        units,
        lang,
        end_pen=parent.end_pen,
        start_pen=0,
        boundary_pen=0,
        phrase_start=True,
        forced_boundary=parent.forced_boundary,
        members=(),
    )
    return _reindex([*atoms[:target], left, right, *atoms[target + 1 :]])


def _piece_from_units(
    parent: LatticeAtom,
    lo: int,
    hi: int,
    units: Sequence[SourceUnit],
    lang: str,
    *,
    end_pen: int,
    start_pen: int,
    boundary_pen: int,
    phrase_start: bool,
    forced_boundary: bool,
    members: tuple[int, ...],
) -> LatticeAtom:
    owned = [units[i] for i in range(lo, hi) if 0 <= i < len(units)]
    text = _join([unit.surface for unit in owned], lang)
    return LatticeAtom(
        index=parent.index,
        text=text,
        start=span_min([unit.start for unit in owned]),
        end=span_max([unit.end for unit in owned]),
        unit_start=lo,
        unit_end=hi,
        end_pen=end_pen,
        start_pen=start_pen,
        boundary_pen=boundary_pen,
        phrase_start=phrase_start,
        forced_boundary=forced_boundary,
        display="".join(_display_chars([text])),
        members=members,
    )


# --------------------------------------------------------------- C16 relief


def relief_trigger(
    atoms: Sequence[LatticeAtom], *, max_line_length: int, max_lines: int
) -> bool:
    """v1's trigger verbatim, unit quirk included.

    The left side is a non-space **character** count and the right side is
    native **cells** times lines. Those are different units, and on a wide-glyph
    language they differ by a factor of two -- but reproducing the quirk is the
    point: this decides when the valve opens, and a "fixed" threshold would open
    it somewhere v1 never did.
    """
    load = sum(_token_char_count(atom.text) for atom in atoms)
    return load > round(RELIEF_TRIGGER_FACTOR * max_line_length * max_lines)


def relief_nodes(
    atoms: Sequence[LatticeAtom], *, max_line_length: int, max_lines: int
) -> tuple[int, ...]:
    """Unit edges nearest each budget multiple, ties resolved toward the left.

    The valve exists for text with no legal break point of its own; these edges
    are invented, so they are placed by the only defensible rule available -- as
    close as possible to where a full cue would have ended -- and never at the
    interval's own ends, which are already barriers.

    Invented is not the same as arbitrary: the candidates are still drawn from
    :func:`unit_edge_nodes`, since a relief cut interior to a source unit would
    break the ownership the valve is trying to rescue. An interval with no
    interior unit edge at all gets no relief, and the caller's path-existence
    check turns that into a typed ``relief-insufficient`` fallback rather than an
    unrepresentable partition.
    """
    budget = max_line_length * max_lines
    if budget <= 0 or len(atoms) < 2:
        return ()
    legal = [node for node in unit_edge_nodes(atoms) if 0 < node < len(atoms)]
    if not legal:
        return ()
    cumulative = [0]
    for atom in atoms:
        cumulative.append(cumulative[-1] + _token_char_count(atom.text))
    total = cumulative[-1]
    out: set[int] = set()
    target = budget
    while target < total:
        best: int | None = None
        best_distance: int | None = None
        for node in legal:
            distance = abs(cumulative[node] - target)
            if best_distance is None or distance < best_distance:
                best, best_distance = node, distance
        if best is not None:
            out.add(best)
        target += budget
    return tuple(sorted(out))


# ------------------------------------------------------- granularity preflight


@dataclass(frozen=True)
class GranularityCheck:
    """Whether a node space can express *any* legal partition of its interval."""

    required_cuts: int
    available_cuts: int

    @property
    def collapsed(self) -> bool:
        return self.available_cuts < self.required_cuts


def granularity_check(
    atoms: Sequence[LatticeAtom], nodes: Sequence[int], profile: DisplayProfile
) -> GranularityCheck:
    """Count the interior cuts a legal partition needs against the ones on offer.

    ``candidate_nodes`` intersects the linguistic break set with the source-unit
    edges, because a boundary interior to a source unit is one the partition
    cannot express. On a word-level stream that removes nothing. On a stream
    whose ``word_data`` is coarser than a phrase -- the legacy sentence-level
    granularity this repo documents as valid -- it can remove *everything*: the
    phrase starts a no-space segmenter finds sit inside the units, so the node
    space collapses to ``{0, N}`` and no cue short enough to fit the layout
    budget can be expressed at all.

    That is a structural fact about the input, not a search that failed, and it
    is worth its own answer: the required cut count comes from a greedy chunking
    at the loosest capacity a cue could ever have (``max_lines`` whole lines,
    and the separator a line break replaces never charged: the first
    ``max_lines - 1`` joins of a chunk are free), with every atom at the fewest
    cells any cue can show of it -- an atom that shows nothing takes neither
    width nor a join. That width is the atom's text normalized on its own, the
    same context-free projection :func:`build_atom_layer` gives
    ``LatticeAtom.display``: on the stream joined without spaces the ``.`` of
    ``de.`` before ``5`` survives ``[.,](?!\\d)``, while a cue joins a space
    there (or ends) and strips it. Every other strip rule is context-free, so the
    text alone strips at least what any cue does. The count is therefore a
    strict lower bound, and a ``collapsed`` verdict cannot be wrong about there
    being no legal path.

    Being a lower bound, it is deliberately one-sided. A coarse stream that has
    *enough* candidate boundaries but none of them in a usable place still fails,
    and keeps the honest ``no-path``/``relief-insufficient`` reason -- this
    counts boundaries, it does not place them.

    P5 resolves this class by splitting *below* the source unit; P4 measures the
    boundary decision on the units it is given and refuses to invent evidence,
    so a collapsed interval takes the typed ``coarse-granularity`` fallback.
    """
    count = len(atoms)
    capacity = profile.max_lines * _line_budget_width(
        profile.max_line_length, profile.language
    )
    available = sum(1 for node in nodes if 0 < node < count)
    if count == 0 or capacity <= 0:
        return GranularityCheck(required_cuts=0, available_cuts=available)
    separator = 0 if _no_spaces(profile.language) else 1
    chunks = 1
    width = 0
    pieces = 0
    for atom in atoms:
        piece = _vis_width(_display_chars([atom.text])[0])
        if not piece:
            continue
        extra = separator if pieces >= profile.max_lines else 0
        if pieces and width + extra + piece > capacity:
            chunks += 1
            width = piece
            pieces = 1
        else:
            width += extra + piece
            pieces += 1
    return GranularityCheck(required_cuts=chunks - 1, available_cuts=available)


# --------------------------------------------------------------- sentences


@dataclass(frozen=True)
class SentenceEnds:
    nodes: frozenset[int]
    missed: int


def sentence_end_nodes(layer: AtomLayer) -> SentenceEnds:
    """Sentence-segmenter cuts mapped onto atom edges, misses included.

    The engine's own mapping silently drops a sentence cut that does not land on
    an atom edge. That is the right behaviour for the engine and the wrong one
    for a measurement, so the count is recorded here: a document with many
    misses is one whose sentence-crossing feature is under-informed, and a
    reader deserves to know that rather than infer it.

    A cut that lands on an atom edge interior to a source unit counts as a miss
    for the same reason: no partition can end a cue there, so charging every path
    alike for crossing it would describe the stream as better informed than it is.
    """
    lang = layer.lang
    text = layer.text
    atoms = layer.atoms
    sentences = _snap_sentence_breaks(text, _segment_sentences(text, lang), lang)
    if len(sentences) < 2:
        return SentenceEnds(frozenset(), 0)
    no_space = _no_spaces(lang)
    atom_ends: dict[int, int] = {}
    consumed = 0
    for index, atom in enumerate(atoms, 1):
        consumed += _token_char_count(atom.text) if no_space else 1
        atom_ends[consumed] = index
    representable = set(unit_edge_nodes(atoms))
    wanted: set[int] = set()
    missed = 0
    consumed = 0
    for sentence in sentences[:-1]:
        tokens = _tokens(sentence, lang)
        consumed += (
            sum(_token_char_count(token) for token in tokens)
            if no_space
            else len(tokens)
        )
        index = atom_ends.get(consumed)
        if index is not None and 0 < index < len(atoms) and index in representable:
            wanted.add(index)
        else:
            missed += 1
    return SentenceEnds(frozenset(wanted), missed)
