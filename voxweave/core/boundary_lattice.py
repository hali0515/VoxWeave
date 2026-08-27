"""The hard-legal boundary lattice: what a cue is *allowed* to be.

Everything a partition optimizer may consider is enumerated here, and nothing
that is merely *preferred* is. The split matters because it is what makes the
search bounded and what keeps policy out of legality: a cost model that could
also veto an edge would be able to hide an infeasibility as a very large number.

Three properties carry the boundedness argument, and each is pinned by a test
rather than argued:

* **positive display progress.** Atoms whose display projection is empty are
  coalesced into the nearest preceding visible atom before any edge is built, so
  every remaining atom is worth at least one half-width cell. A legal cue
  therefore spans at most ``band_atoms(profile)`` atoms and the monotone early
  break in the edge scan provably fires -- without coalescing, a run of
  punctuation would let the scan walk an unbounded number of zero-width atoms.
* **the all-invisible branch is total.** An interval with no visible atom at all
  is not "skipped": it still obeys the duration cap, source-unit relief and the
  held-chain whitelist before it may call itself a defined result. It just does
  so with a forced edge chain rather than a search.
* **infeasibility is typed, never silent.** Every way an interval can fail to
  produce a legal path has a name in :data:`INFEASIBLE_REASONS`, and the caller
  adopts the v1 partition for that region instead of the lattice inventing one.

**Scope: the node space is only as fine as the source units.** A cue boundary
must be a source-unit edge, so a document whose ``word_data`` is coarser than a
phrase -- the legacy sentence-level granularity this repo treats as valid -- can
have no expressible partition at all, no matter what the cost model prefers. That
is detected structurally before any search (:func:`granularity_check`) and typed
:data:`COARSE_GRANULARITY`, so it is never mistaken for a solver that could not
find a path it had. **The P5 resolution is splitting below the source unit**;
P4 measures the boundary decision on the units it is handed and refuses to invent
sub-unit evidence, so a collapsed interval falls back to v1 by design.

Every helper this module borrows from ``smart_split``/``layout`` is used
verbatim, including the quirks: the relief trigger compares a non-space *char*
count against a *cell* budget, and the display projection is taken on the joined
stream so the context-sensitive punctuation rule sees the same context the width
oracle will.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any

from .canonical_text import (
    CanonicalWork,
    band_scan_lower_bound_exceeded,
    canonical_legal,
    canonical_text,
)
from .layout import (
    _PUNCT_TO_SPACE_RE,
    _UNIT_GLYPHS,
    _is_ascii_run_char,
    _join,
    _line_budget_width,
    _no_spaces,
    _token_char_count,
    _vis_width,
    strip_punct_for_subtitles,
)
from .partition_check import Waiver
from .schema import Unit
from .segdoc import DisplayProfile, SegDocument, SourceUnit
from .smart_split import (
    _attach_end_penalties,
    _build_atoms,
    _display_chars,
    _phrase_boundary_atoms,
    _segment_sentences,
    _snap_sentence_breaks,
    _tokens,
)
from .timing import HELD_WORD_MAX_GAP_S

#: Half of the robust-silence band. A barrier needs the configured skip
#: threshold *plus* this margin on top, so a gap has to clear the threshold by
#: more than the measurement uncertainty before the topology is allowed to
#: depend on it. It does not make barriers immune to a probe of the same size --
#: it relocates the cliff, and the perturbation runner probes the new location.
BARRIER_UNCERTAINTY_MS: float = 50.0

#: Duration-legality tolerance, inherited verbatim from the v1 hard-edge builder
#: so an edge this module certifies is one v1 would also have accepted.
CAP_EPS_S: float = 1e-9

#: v1's ``FORCE_BREAK_FACTOR``: how far a boundary-less run may exceed the line
#: budget before the relief valve is allowed to invent break points.
RELIEF_TRIGGER_FACTOR: float = 1.5

#: The held-word chain tolerance, taken from the timing pass rather than
#: restated, so the evidence test and the pass that consumes it cannot drift.
HELD_CHAIN_MAX_GAP_S: float = HELD_WORD_MAX_GAP_S

#: Locality cell for the perturbation lanes, in source units (one legacy
#: locality window per side). Declared here; consumed by the harness.
INFLUENCE_RADIUS_UNITS: int = 96

BARRIER_KINDS: tuple[str, ...] = ("document", "robust-silence")

#: The one infeasibility that is a statement about the INPUT rather than about
#: the search: the source units are coarser than a cue, so the expressible node
#: space cannot tile the interval however the solver is tuned. Named so a reader
#: (and the harness) can tell it apart from a genuine optimizer failure.
COARSE_GRANULARITY: str = "coarse-granularity"

INFEASIBLE_REASONS: tuple[str, ...] = (
    COARSE_GRANULARITY,
    "duration-unwaivable",
    "no-path",
    "relief-insufficient",
    "span-preflight",
)

SPAN_VIOLATION_REASONS: tuple[str, ...] = (
    "bool-bound",
    "ends-non-monotone",
    "non-finite",
    "reversed",
    "starts-non-monotone",
)

PROFILE_VIOLATION_REASONS: tuple[str, ...] = ("not-positive", "negative", "too-small")


# --------------------------------------------------------------------- spans


def span_min(values: Iterable[float | None]) -> float | None:
    """Smallest non-``None`` value, or ``None`` when nothing is timed.

    The ``None``-propagating semantics of the engine's own span helpers: a
    missing bound is missing evidence, not a zero.
    """
    known = [v for v in values if v is not None]
    return min(known) if known else None


def span_max(values: Iterable[float | None]) -> float | None:
    """Largest non-``None`` value, or ``None`` when nothing is timed."""
    known = [v for v in values if v is not None]
    return max(known) if known else None


def _real(value: Any) -> bool:
    """A finite, non-bool number. ``None`` is missing evidence, not a failure."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


# ---------------------------------------------------------------- preflights


@dataclass(frozen=True)
class SpanViolation:
    """One source unit whose recorded span cannot be interpreted at all."""

    unit_index: int
    unit_id: str
    reason: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "reason": self.reason,
            "unit_id": self.unit_id,
            "unit_index": self.unit_index,
        }


def preflight_units(units: Sequence[SourceUnit]) -> tuple[SpanViolation, ...]:
    """Reject unit spans no downstream rule can give a meaning to.

    ``None`` is deliberately **tolerated**: a ghost unit carries no timing and
    the span-resolution rules already answer for it. What is refused is a bound
    that claims to be a measurement and is not -- a bool, a non-number, NaN or an
    infinity -- plus a reversed span and a stream whose known starts or known
    ends walk backwards. ``None`` bounds are skipped without breaking the
    monotonicity chain.

    A violation marks the enclosing hard interval infeasible; it never aborts
    the document, because one unreadable row must not throw away a whole file.
    """
    out: list[SpanViolation] = []
    last_start: float | None = None
    last_end: float | None = None
    for index, unit in enumerate(units):
        bad = False
        for name, value in (("start", unit.start), ("end", unit.end)):
            if value is None:
                continue
            if isinstance(value, bool):
                out.append(
                    SpanViolation(index, unit.id, "bool-bound", f"{name}={value!r}")
                )
                bad = True
            elif not _real(value):
                out.append(
                    SpanViolation(index, unit.id, "non-finite", f"{name}={value!r}")
                )
                bad = True
        if bad:
            continue
        start, end = unit.start, unit.end
        if start is not None and end is not None and end < start:
            out.append(
                SpanViolation(index, unit.id, "reversed", f"end {end} < start {start}")
            )
            continue
        if start is not None:
            if last_start is not None and start < last_start:
                out.append(
                    SpanViolation(
                        index,
                        unit.id,
                        "starts-non-monotone",
                        f"start {start} < previous start {last_start}",
                    )
                )
            last_start = start
        if end is not None:
            if last_end is not None and end < last_end:
                out.append(
                    SpanViolation(
                        index,
                        unit.id,
                        "ends-non-monotone",
                        f"end {end} < previous end {last_end}",
                    )
                )
            last_end = end
    out.sort(key=lambda v: (v.unit_index, v.reason))
    return tuple(out)


@dataclass(frozen=True)
class ProfileViolation:
    """One resolved display knob the shadow refuses to interpret."""

    key: str
    value: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "reason": self.reason, "value": self.value}


def preflight_profile(profile: DisplayProfile) -> tuple[ProfileViolation, ...]:
    """Refuse a profile whose knobs have no defined meaning.

    Zero **disables** the duration cap, which is what the timing pass' own truth
    test means by it; a negative cap is refused outright rather than
    reinterpreted, so the shadow never quietly disagrees with cleanup about what
    a negative cap does. A non-empty result aborts the measurement for that
    document -- an invalid measurement, never a silent fallback.
    """
    out: list[ProfileViolation] = []
    for key in ("clause_ms", "offline_ms", "vad_skip_ms"):
        value = float(getattr(profile, key))
        if value <= 0:
            out.append(ProfileViolation(key, value, "not-positive"))
    if profile.max_cue_s < 0:
        out.append(ProfileViolation("max_cue_s", float(profile.max_cue_s), "negative"))
    if profile.max_line_length < 1:
        out.append(
            ProfileViolation(
                "max_line_length", float(profile.max_line_length), "too-small"
            )
        )
    if profile.max_lines < 1:
        out.append(ProfileViolation("max_lines", float(profile.max_lines), "too-small"))
    return tuple(out)


# --------------------------------------------------------------- atom layer


@dataclass(frozen=True)
class LatticeAtom:
    """One non-breakable packing unit, with its source-unit footprint.

    ``text`` is the pre-strip surface (what the cue will store), ``display`` the
    projection the width oracle will actually measure -- empty means the atom
    renders to nothing. The footprint ``[unit_start, unit_end)`` is the only
    granularity-safe way to slice the unit stream back, so every ownership
    question is answered by it rather than by re-deriving a character cursor.
    """

    index: int
    text: str
    start: float | None
    end: float | None
    unit_start: int
    unit_end: int
    end_pen: int
    start_pen: int
    boundary_pen: int
    phrase_start: bool
    forced_boundary: bool
    display: str
    members: tuple[int, ...]

    @property
    def visible(self) -> bool:
        return bool(self.display)


@dataclass(frozen=True)
class AtomLayer:
    """The document's whole atom stream, in the engine's own construction."""

    lang: str
    text: str
    atoms: tuple[LatticeAtom, ...]
    unit_count: int

    def unit_bound(self, node: int) -> int:
        """The source-unit id an atom-edge node cuts at."""
        if node >= len(self.atoms):
            return self.unit_count
        return self.atoms[max(node, 0)].unit_start


def build_atom_layer(document: SegDocument) -> AtomLayer:
    """Build the atom stream with v1's construction and nothing of its own.

    The word_data handed to ``_build_atoms`` is the document's units in their
    recorded surface/span form, and the phrase boundaries and end penalties come
    from the same helpers the engine calls, in the same order. ``document.text``
    is required: re-joining the surfaces here would re-implement the no-space
    language rule and could disagree with the join that actually ran.
    """
    if document.text is None:
        raise ValueError(
            "SegDocument.text is required: the shadow reads the joined stream the "
            "engine consumed and never re-derives it from unit surfaces"
        )
    lang = document.profile.language
    text = document.text
    word_data: list[Unit] = [
        {"text": unit.surface, "start": unit.start, "end": unit.end}
        for unit in document.units
    ]
    raw = _build_atoms(
        text,
        word_data,
        lang,
        max_atom_width=_line_budget_width(document.profile.max_line_length, lang),
    )
    boundary = _phrase_boundary_atoms(raw, text, lang) if _no_spaces(lang) else None
    _attach_end_penalties(raw, boundary, lang)
    displays = _display_chars([atom["text"] for atom in raw])
    atoms = tuple(
        LatticeAtom(
            index=index,
            text=atom["text"],
            start=atom.get("start"),
            end=atom.get("end"),
            unit_start=int(atom["_unit_start"]),
            unit_end=int(atom["_unit_end"]),
            end_pen=int(atom.get("end_pen", 0)),
            start_pen=int(atom.get("start_pen", 0)),
            boundary_pen=int(atom.get("boundary_pen", 0)),
            phrase_start=True if boundary is None else index in boundary,
            forced_boundary=bool(atom.get("forced_boundary", False)),
            display=displays[index],
            members=(index,),
        )
        for index, atom in enumerate(raw)
    )
    return AtomLayer(lang=lang, text=text, atoms=atoms, unit_count=len(document.units))


def band_atoms(profile: DisplayProfile) -> int:
    """The proven hard band on a legal edge's atom span, in coalesced atoms.

    Every coalesced atom is worth at least one half-width cell, so a cue that
    fits ``max_lines`` lines of the resolved budget cannot contain more than this
    many of them; the ``+1`` is the overflow-discovery atom.
    """
    return (
        profile.max_lines
        * _line_budget_width(profile.max_line_length, profile.language)
        + 1
    )


# --------------------------------------------------------------- coalescing


@dataclass(frozen=True)
class CoalesceResult:
    atoms: tuple[LatticeAtom, ...]
    coalesced_atoms: int
    all_invisible: bool


def _reindex(atoms: Sequence[LatticeAtom]) -> tuple[LatticeAtom, ...]:
    """Renumber a sequence's ``index`` fields to its own positions."""
    return tuple(
        atom if atom.index == position else _replace(atom, index=position)
        for position, atom in enumerate(atoms)
    )


def _replace(atom: LatticeAtom, **over: Any) -> LatticeAtom:
    fields = {
        "index": atom.index,
        "text": atom.text,
        "start": atom.start,
        "end": atom.end,
        "unit_start": atom.unit_start,
        "unit_end": atom.unit_end,
        "end_pen": atom.end_pen,
        "start_pen": atom.start_pen,
        "boundary_pen": atom.boundary_pen,
        "phrase_start": atom.phrase_start,
        "forced_boundary": atom.forced_boundary,
        "display": atom.display,
        "members": atom.members,
    }
    fields.update(over)
    return LatticeAtom(**fields)


def coalesce_zero_display(
    atoms: Sequence[LatticeAtom], *, lang: str = "en"
) -> CoalesceResult:
    """Fold every invisible atom into a visible carrier (canonical, in order).

    An invisible atom joins the nearest **preceding** visible atom; a leading
    invisible run, which has no predecessor, joins the **first** visible atom.
    Ownership merges exactly by footprint and spans by ``None``-propagating
    min/max, so nothing is lost and nothing is invented; the carrier keeps its
    own penalties, because they describe a break at a boundary that still
    exists. The merged surface is the *language join* of its members rather than
    a bare concatenation, so the interval's text still reconstructs byte for
    byte and a trailing full stop is still visible to the punctuation-affinity
    feature.

    A boundary between two invisible atoms is meaningless for subtitles, which
    is the policy justification; the mechanical consequence is that every
    surviving atom has positive display width, which is what bounds the edge
    scan. When *every* atom is invisible nothing is folded and the caller takes
    the all-invisible branch instead.

    ``lang`` selects that join and defaults to the space-delimited one, so a
    positional caller keeps the shape it had before the parameter existed.
    """
    if not atoms:
        return CoalesceResult((), 0, False)
    first_visible = next((i for i, atom in enumerate(atoms) if atom.visible), None)
    if first_visible is None:
        return CoalesceResult(tuple(atoms), 0, True)

    groups: list[list[LatticeAtom]] = [list(atoms[: first_visible + 1])]
    carriers: list[LatticeAtom] = [atoms[first_visible]]
    for atom in atoms[first_visible + 1 :]:
        if atom.visible:
            groups.append([atom])
            carriers.append(atom)
        else:
            groups[-1].append(atom)

    folded = sum(len(group) - 1 for group in groups)
    return CoalesceResult(
        atoms=tuple(
            _merge_group(group, carrier, position, lang)
            for position, (group, carrier) in enumerate(zip(groups, carriers))
        ),
        coalesced_atoms=folded,
        all_invisible=False,
    )


def _merge_group(
    group: Sequence[LatticeAtom], carrier: LatticeAtom, position: int, lang: str
) -> LatticeAtom:
    if len(group) == 1:
        return _replace(group[0], index=position)
    members: list[int] = []
    for atom in group:
        members.extend(atom.members)
    return LatticeAtom(
        index=position,
        text=_join([atom.text for atom in group], lang),
        start=span_min([atom.start for atom in group]),
        end=span_max([atom.end for atom in group]),
        unit_start=group[0].unit_start,
        unit_end=group[-1].unit_end,
        end_pen=carrier.end_pen,
        start_pen=carrier.start_pen,
        boundary_pen=carrier.boundary_pen,
        phrase_start=carrier.phrase_start,
        forced_boundary=carrier.forced_boundary,
        display=carrier.display,
        members=tuple(sorted(members)),
    )


# ------------------------------------------------------ C1 unit-edge node space


def unit_edge_nodes(atoms: Sequence[LatticeAtom]) -> tuple[int, ...]:
    """The atom edges that are also source-unit edges -- the whole node space.

    C1 puts the node space in *unit* ids, and every conservation argument
    downstream restates that as a fact about atoms ("every cut is an atom edge
    and therefore a source-unit edge"). It is not one. ``_build_atoms``
    subdivides a source unit freely: one atom per CJK glyph however many glyphs
    the aligner packed into a unit, one atom per word however many words a
    coarse-grained ``word_data`` entry carries. Word-level ja/zh evidence and the
    legacy sentence-level granularity therefore both produce atom edges that sit
    *inside* a unit, and they are the common case rather than the exotic one.

    Cutting there does not merely give a worse boundary, it gives an
    unrepresentable one. The partition is reported in unit ids, so both sides of
    such a cut collapse onto the same id: the emitted cut list stops being
    strictly increasing, the cue stream splits a unit its own partition calls
    whole, the owned-unit text no longer matches the cue text, and -- since every
    atom of a subdivided unit inherits that unit's full span -- the two cues
    overlap in time. Restricting the node space here is what makes the footprint
    the single ownership authority the rest of the design assumes it already is.

    The two ends are always edges: a node space that could not close the interval
    would have no legal path at all.
    """
    count = len(atoms)
    nodes = {0, count}
    for index in range(1, count):
        if atoms[index].unit_start >= atoms[index - 1].unit_end:
            nodes.add(index)
    return tuple(sorted(nodes))


def _latest_unit_edge(legal: Sequence[int], upper: int, lower: int) -> int | None:
    """The largest unit edge in ``(lower, upper]``, or None when there is none."""
    position = bisect_right(legal, upper) - 1
    if position < 0:
        return None
    node = legal[position]
    return node if node > lower else None


def exclusively_owned(atoms: Sequence[LatticeAtom], index: int) -> bool:
    """Does this atom own every unit in its footprint, sharing none of them?

    The relief seam mints its pieces from the *unit* stream, which is only sound
    when the atom it is splitting owns those units outright. It does not always:
    a source unit subdivided into several atoms leaves each of them claiming the
    same unit id, and coalescing can then hand the trailing piece a footprint
    that starts on a unit its left neighbour already carries. Re-minting such a
    piece from ``units[lo:hi]`` resurrects the whole unit surface next to the
    sibling that still holds part of it, duplicating text that conservation then
    correctly reports as missing from the other side.

    So relief asks this first. An atom that shares a unit is left alone and the
    duration ladder continues to the held-chain test and the typed fallback --
    the outcomes that exist for "no legal cut is available here".
    """
    atom = atoms[index]
    if index > 0 and atoms[index - 1].unit_end > atom.unit_start:
        return False
    if index + 1 < len(atoms) and atoms[index + 1].unit_start < atom.unit_end:
        return False
    return True


# --------------------------------------------------------------- C4 barriers


@dataclass(frozen=True)
class HardBarrier:
    """A cut point no partition may cross: a document end or a robust silence."""

    node: int
    unit_id: int
    kind: str
    gap_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_ms": self.gap_ms,
            "kind": self.kind,
            "node": self.node,
            "unit_id": self.unit_id,
        }


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_count": self.atom_count,
            "barrier_left": self.left.kind,
            "barrier_right": self.right.kind,
            "index": self.index,
            "node_range": [self.node_start, self.node_end],
            "unit_range": [self.unit_start, self.unit_end],
        }


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


# --------------------------------------------------------- C6 packing oracle


@dataclass(frozen=True)
class PackMeasure:
    """The batch layout answer for one atom prefix."""

    fits: bool
    lines: int
    line_widths: tuple[int, ...]
    text: str

    @property
    def balance(self) -> float:
        if self.lines != 2:
            return 0.0
        return float(abs(self.line_widths[0] - self.line_widths[1]))


class _PackState:
    """Everything about a prefix measurement that the flush can still change."""

    __slots__ = (
        "started",
        "space_pending",
        "token",
        "token_ascii",
        "gap",
        "held",
        "held_gap",
        "finished",
        "cur_width",
        "cur_open",
    )

    def __init__(self) -> None:
        self.started = False
        self.space_pending = False
        self.token = ""
        self.token_ascii = False
        self.gap = ""
        self.held: str | None = None
        self.held_gap = ""
        self.finished: list[int] = []
        self.cur_width = 0
        self.cur_open = False

    def clone(self) -> _PackState:
        other = _PackState()
        other.started = self.started
        other.space_pending = self.space_pending
        other.token = self.token
        other.token_ascii = self.token_ascii
        other.gap = self.gap
        other.held = self.held
        other.held_gap = self.held_gap
        other.finished = list(self.finished)
        other.cur_width = self.cur_width
        other.cur_open = self.cur_open
        return other


class IncrementalPacker:
    """Prefix-extension oracle over the normalized display stream.

    The input projection is ``strip_punct_for_subtitles(_join(texts, lang))``,
    and the naive way to get it per prefix is to redo all four passes every
    time. The production lattice uses this fold only for spaced languages, where
    it is differential-pinned to the batch packing oracle. No-space admission is
    governed by cached :func:`canonical_text` projections instead: kinsoku can
    move a closing glyph across a normalized gap, so no bounded streaming state
    is allowed to approximate that decision.

    Each pass is a left-to-right fold with bounded lookahead, so the whole chain
    streams:

    * punctuation stripping needs exactly **one character** of lookahead -- the
      rule that keeps the dot of ``3.75`` is ``[.,](?!\\d)``, and every match is a
      single character, so the production regex can be asked about one character
      at a time by matching it against ``char + next_char``;
    * whitespace collapsing and trimming need only a "space pending" flag;
    * tokenization is local, and greedy first-fit line packing never re-flows a
      line once a later token has been placed.

    What cannot be committed early is held: the last raw character (awaiting its
    lookahead), a partial token, and -- for no-space languages -- the last whole
    token, because a following unit glyph merges *into* it. A measurement
    therefore flushes a **copy** of that bounded tail, which is what makes the
    reported answer the batch answer for the current prefix rather than for the
    prefix plus whatever comes next.

    One honest caveat: the *decision* (fits, line count, widths) is O(1)
    amortised per atom, but materializing ``PackMeasure.text`` is linear in the
    prefix. That is not a shortcoming of the fold -- the caller needs that exact
    string for the edge it is about to build, so the cost is inherent to the
    output rather than to the measurement.
    """

    def __init__(self, lang: str, max_line_length: int, max_lines: int) -> None:
        self._lang = lang
        self._no_spaces = _no_spaces(lang)
        self._sep = "" if self._no_spaces else " "
        self._sep_width = _vis_width(self._sep)
        self._budget = _line_budget_width(max_line_length, lang)
        self._max_lines = max_lines
        self._steps = 0
        self.reset()

    def reset(self) -> None:
        """Start a fresh prefix. The step counter deliberately survives."""
        self._first = True
        self._hold: str | None = None
        self._norm: list[str] = []
        self._state = _PackState()

    @property
    def steps(self) -> int:
        """Cumulative extension steps -- the work counter, not a prefix length."""
        return self._steps

    def extend(self, atom_text: str) -> PackMeasure:
        """Append one atom and return the batch measure for the whole prefix."""
        self._steps += 1
        chunk = atom_text if self._first else self._sep + atom_text
        self._first = False
        for char in chunk:
            if self._hold is not None:
                self._norm.append(
                    self._feed(self._state, self._resolve(self._hold, char))
                )
            self._hold = char
        return self._measure()

    # -- stage A: punctuation, resolved one character at a time ---------------

    @staticmethod
    def _resolve(char: str, nxt: str | None) -> str:
        """The character the strip pass keeps, given its single lookahead.

        Asked of the production regex directly: every alternative it can match is
        exactly one character wide, so a match anchored at position 0 of
        ``char + lookahead`` is precisely the substitution the batch pass would
        make at that offset.
        """
        return " " if _PUNCT_TO_SPACE_RE.match(char + (nxt or "")) else char

    # -- stages B and C: whitespace, tokenization ----------------------------

    def _feed(self, state: _PackState, char: str) -> str:
        """Push one stripped character through collapsing and tokenization."""
        if char.isspace():
            if state.started:
                state.space_pending = True
            return ""
        out = ""
        if state.space_pending:
            state.space_pending = False
            out += " "
            self._feed_token(state, " ")
        state.started = True
        self._feed_token(state, char)
        return out + char

    def _feed_token(self, state: _PackState, char: str) -> None:
        if not self._no_spaces:
            if char == " ":
                self._close_token(state)
            else:
                state.token += char
            return
        if char == " ":
            self._close_token(state)
            # The strip pass emits one normalized space.  Keep it as the gap
            # before the next token so packing charges it when both tokens stay
            # on one line and drops it when the line breaks at that boundary.
            state.gap = char
            return
        if _is_ascii_run_char(char):
            if state.token and state.token_ascii:
                state.token += char
            else:
                self._close_token(state)
                state.token = char
                state.token_ascii = True
            return
        self._close_token(state)
        self._emit_token(state, char)

    def _close_token(self, state: _PackState) -> None:
        if state.token:
            self._emit_token(state, state.token)
        state.token = ""
        state.token_ascii = False

    def _emit_token(self, state: _PackState, token: str) -> None:
        if not self._no_spaces:
            self._commit_token(state, token)
            return
        gap_before = state.gap
        state.gap = ""
        if state.held is not None:
            if not gap_before and token in _UNIT_GLYPHS and state.held[-1:].isdigit():
                state.held += token
                return
            self._commit_token(state, state.held, gap_before=state.held_gap)
        state.held = token
        state.held_gap = gap_before

    # -- stage D: greedy first-fit packing -----------------------------------

    def _commit_token(
        self, state: _PackState, token: str, *, gap_before: str | None = None
    ) -> None:
        width = _vis_width(token)
        separator_width = (
            self._sep_width if gap_before is None else _vis_width(gap_before)
        )
        extra = separator_width if state.cur_open else 0
        if state.cur_open and state.cur_width + width + extra > self._budget:
            state.finished.append(state.cur_width)
            state.cur_width = width
        else:
            state.cur_width += width + extra
        state.cur_open = True

    def _measure(self) -> PackMeasure:
        state = self._state.clone()
        tail = ""
        if self._hold is not None:
            tail = self._feed(state, self._resolve(self._hold, None))
        self._close_token(state)
        if state.held is not None:
            self._commit_token(state, state.held, gap_before=state.held_gap)
            state.held = None
            state.held_gap = ""
        widths = list(state.finished)
        if state.cur_open:
            widths.append(state.cur_width)
        if not widths:
            # An empty projection is one zero-width line, matching the batch
            # call's behaviour on the empty string.
            widths = [0]
        return PackMeasure(
            fits=len(widths) <= self._max_lines
            and all(width <= self._budget for width in widths),
            lines=len(widths),
            line_widths=tuple(widths),
            text="".join(self._norm) + tail,
        )


def _canonical_pack_measure(
    atoms: Sequence[LatticeAtom],
    start: int,
    end: int,
    profile: DisplayProfile,
    work: CanonicalWork,
) -> PackMeasure:
    """Direct no-space admission facts for one cached lattice span.

    The cache key is the lattice's own ``(start, end)`` pair. Legality and the
    recorded line facts come only from ``FinalText``; the unwrapped display text
    remains the direct batch strip used by the cost model before finalization.
    """
    chunk = atoms[start:end]
    lang = profile.language
    raw = _join([atom.text for atom in chunk], lang)
    final = work.cached(
        (start, end),
        lambda: canonical_text(
            [
                {"text": atom.text, "start": atom.start, "end": atom.end}
                for atom in chunk
            ],
            fallback_text=raw,
            lang=lang,
            profile=profile,
            expected_footprint=raw,
            work=work,
        ),
    )
    return PackMeasure(
        fits=canonical_legal(final, profile),
        lines=len(final.lines),
        line_widths=final.cell_widths,
        text=strip_punct_for_subtitles(raw),
    )


# ------------------------------------------------------------------- edges


@dataclass(frozen=True)
class Edge:
    """One hard-legal candidate cue, spanning ``[start_node, end_node)``."""

    start_node: int
    end_node: int
    text: str
    display_text: str
    lines: int
    line_widths: tuple[int, ...]
    span_start: float | None
    span_end: float | None
    waiver: Waiver | None
    # W3 candidate cache.  Typed as Any here so the hard-lattice module does
    # not import the policy module at import time; build_document_lattice fills
    # it through one local import after hard admission has finished.
    evidence_span: Any | None = None
    lyric: bool = False

    @property
    def vis_width(self) -> int:
        return _vis_width(self.display_text)

    @property
    def balance(self) -> float:
        if self.lines != 2:
            return 0.0
        return float(abs(self.line_widths[0] - self.line_widths[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_text": self.display_text,
            "end_node": self.end_node,
            "evidence_span": None
            if self.evidence_span is None
            else self.evidence_span.to_dict(),
            "line_widths": list(self.line_widths),
            "lines": self.lines,
            "lyric": self.lyric,
            "span": [self.span_start, self.span_end],
            "start_node": self.start_node,
            "waiver": None if self.waiver is None else self.waiver.to_dict(),
        }


@dataclass(frozen=True)
class Infeasible:
    """A typed reason one interval produced no legal partition at all."""

    reason: str
    detail: str
    unit_range: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "reason": self.reason,
            "unit_range": list(self.unit_range),
        }


def held_chain_continuous(
    units: Sequence[SourceUnit],
    unit_start: int,
    unit_end: int,
    *,
    max_gap_s: float = HELD_CHAIN_MAX_GAP_S,
) -> bool:
    """Is every timed unit in the range part of one continuous sounding chain?

    This is the timing pass' own held-word predicate, not a restatement of it:
    walk the fully timed units in start order and follow the chain while each
    next start is within ``max_gap_s`` of the previous end. Zero timed units
    returns ``False`` -- absence of evidence is not evidence -- and exactly one
    returns ``True``, since a single span is trivially continuous.
    """
    timed = sorted(
        (unit.start, unit.end)
        for unit in units[max(unit_start, 0) : max(unit_end, 0)]
        if unit.start is not None and unit.end is not None
    )
    if not timed:
        return False
    for (_prev_start, prev_end), (next_start, _next_end) in zip(timed, timed[1:]):
        if next_start - prev_end > max_gap_s:
            return False
    return True


def greedy_cap_partition(
    atoms: Sequence[LatticeAtom], max_cue_s: float
) -> tuple[int, ...]:
    """Deterministic linear cap partition at source-unit edges.

    Accumulate the running chunk's raw span and cut as late as legally possible
    at or before the first atom whose inclusion would push it over the cap. Cuts
    are drawn from :func:`unit_edge_nodes`, so ownership and conservation hold
    exactly; where every atom is its own unit that is the atom edge itself and
    the walk is the plain greedy one. When the offending atom sits inside a
    source unit the chunk keeps accumulating instead of cutting somewhere it
    cannot express -- the over-cap chunk is then handed to the relief, held-chain
    and typed-fallback ladder that exists for exactly this case. A chunk whose
    span cannot be resolved never triggers a cut: there is nothing to be over.
    """
    if max_cue_s <= 0:
        return ()
    legal = unit_edge_nodes(atoms)
    cuts: list[int] = []
    chunk_start = 0
    low: float | None = None
    high: float | None = None
    for index, atom in enumerate(atoms):
        merged_low = span_min([low, atom.start])
        merged_high = span_max([high, atom.end])
        over = (
            index > chunk_start
            and merged_low is not None
            and merged_high is not None
            and merged_high - merged_low > max_cue_s + CAP_EPS_S
        )
        cut = _latest_unit_edge(legal, index, chunk_start) if over else None
        if cut is not None:
            cuts.append(cut)
            chunk_start = cut
            low = span_min([a.start for a in atoms[cut : index + 1]])
            high = span_max([a.end for a in atoms[cut : index + 1]])
        else:
            low, high = merged_low, merged_high
    return tuple(cuts)


@dataclass(frozen=True)
class CapResolution:
    """The outcome of making one atom run obey the duration cap."""

    atoms: tuple[LatticeAtom, ...]
    cuts: tuple[int, ...]
    waivers: tuple[Waiver, ...]
    relief_injections: int
    infeasible: Infeasible | None


def _chunk_bounds(cuts: Sequence[int], count: int) -> tuple[tuple[int, int], ...]:
    edges = (0, *cuts, count)
    return tuple((edges[i], edges[i + 1]) for i in range(len(edges) - 1))


def _over_cap(atoms: Sequence[LatticeAtom], lo: int, hi: int, max_cue_s: float) -> bool:
    if max_cue_s <= 0:
        return False
    low = span_min([atom.start for atom in atoms[lo:hi]])
    high = span_max([atom.end for atom in atoms[lo:hi]])
    if low is None or high is None:
        return False
    return high - low > max_cue_s + CAP_EPS_S


def resolve_cap_partition(
    atoms: Sequence[LatticeAtom],
    units: Sequence[SourceUnit],
    *,
    max_cue_s: float,
    unit_offset: int = 0,
    lang: str = "en",
    allow_relief: bool = True,
) -> CapResolution:
    """Make an atom run cap-legal: relief first, then a waiver, then a fallback.

    The ordering is the whole point. An atom that owns several source units is
    hiding boundaries the cap partition could have used, so relief runs *before*
    any exemption is considered -- a cue is only exempted from the cap once no
    legal cut exists at all. Only then does the held-chain evidence test decide
    whether the over-long cue is a word still sounding (waivable, with recorded
    provenance) or an unexplained overrun.

    That leaves ``duration-unwaivable`` unreachable for a single-unit footprint
    with a timed span, since one timed span is trivially a continuous chain. The
    typed terminal stays in the contract as a defensive one: its code path is
    asserted by contradicting the evidence test directly, not by a document that
    can actually be built.

    ``lang`` is a deviation from the reviewed signature: relief mints piece
    surfaces by joining unit surfaces, which is language-dependent. It defaults
    to the space-delimited join so existing positional callers are unaffected.

    ``allow_relief=False`` is for the mixed branch, whose atoms have already been
    through :func:`_relieve_over_cap_atoms` (the same split, under the same
    exclusivity test) and then re-coalesced. Splitting again here would mint
    pieces the caller's atom indices cannot address, so the returned ``cuts``
    would no longer point at anything it can inject.
    """
    current = list(atoms)
    injections = 0
    guard = (len(units) + len(current) + 2) if allow_relief else 0
    while guard > 0:
        guard -= 1
        cuts = greedy_cap_partition(current, max_cue_s)
        target: int | None = None
        for lo, hi in _chunk_bounds(cuts, len(current)):
            if hi - lo != 1 or not _over_cap(current, lo, hi, max_cue_s):
                continue
            atom = current[lo]
            # The exclusivity test is redundant *here* and deliberately kept. A
            # single-atom chunk has a legal cut on both sides by construction, and
            # a legal cut is a unit edge, so such an atom cannot be sharing a unit
            # with either neighbour -- no mutation of this line is observable
            # today. It stays so the two relief call sites cannot drift apart if
            # the cap partition is ever allowed to cut somewhere else again.
            if atom.unit_end - atom.unit_start > 1 and exclusively_owned(current, lo):
                target = lo
                break
        if target is None:
            break
        atom = current[target]
        for unit_id in range(atom.unit_start + 1, atom.unit_end):
            current = list(split_candidate_at_unit(current, unit_id, units, lang))
            injections += 1

    cuts = greedy_cap_partition(current, max_cue_s)
    waivers: list[Waiver] = []
    for chunk_index, (lo, hi) in enumerate(_chunk_bounds(cuts, len(current))):
        if not _over_cap(current, lo, hi, max_cue_s):
            continue
        unit_start = current[lo].unit_start
        unit_end = current[hi - 1].unit_end
        low = span_min([atom.start for atom in current[lo:hi]])
        high = span_max([atom.end for atom in current[lo:hi]])
        if held_chain_continuous(
            units, unit_start - unit_offset, unit_end - unit_offset
        ):
            waivers.append(
                Waiver(
                    kind="held-chain-duration",
                    cue_index=chunk_index,
                    unit_ids=tuple(range(unit_start, unit_end)),
                    span=(low, high),
                    cap=max_cue_s,
                    detail="a word is still sounding past the cap",
                )
            )
            continue
        return CapResolution(
            atoms=tuple(current),
            cuts=tuple(cuts),
            waivers=tuple(waivers),
            relief_injections=injections,
            infeasible=Infeasible(
                reason="duration-unwaivable",
                detail=(
                    f"units [{unit_start}, {unit_end}) span {low} to {high} past the "
                    f"cap {max_cue_s} with no continuous held chain"
                ),
                unit_range=(unit_start, unit_end),
            ),
        )
    return CapResolution(
        atoms=tuple(current),
        cuts=tuple(cuts),
        waivers=tuple(waivers),
        relief_injections=injections,
        infeasible=None,
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
    every atom at least one half-width cell), so it is a strict lower bound and
    a ``collapsed`` verdict cannot be wrong about there being no legal path.

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
    for atom in atoms:
        piece = _vis_width(atom.display)
        extra = separator if width else 0
        if width and width + extra + piece > capacity:
            chunks += 1
            width = piece
        else:
            width += extra + piece
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


# -------------------------------------------------------------- the lattice


@dataclass(frozen=True)
class IntervalLattice:
    """Every legal cue and every legal boundary inside one hard interval."""

    interval: HardInterval
    atoms: tuple[LatticeAtom, ...]
    nodes: tuple[int, ...]
    edges: tuple[Edge, ...]
    edges_from: Mapping[int, tuple[Edge, ...]]
    coalesced_atoms: int
    all_invisible: bool
    relief_injections: int
    waivers: tuple[Waiver, ...]
    infeasible: Infeasible | None
    packer_steps: int
    #: Raw character visits charged by cached no-space FinalText projections.
    canonical_chars: int
    #: Candidate boundaries the duration ladder had to expose because the run
    #: was over the cap and splittable at a source-unit edge the linguistic node
    #: space had hidden. Counted apart from ``relief_injections`` so an artifact
    #: reader can tell a layout rescue (C16) from a duration one (AD6-1).
    cap_relief_nodes: int = 0

    def unit_bound(self, node: int) -> int:
        """The source-unit id a node cuts at, closed against the interval.

        Interior nodes read their atom's footprint, but the two ends come from
        the *interval*, never from an atom. That is not a stylistic choice: atom
        footprints do not always tile the unit stream (the reconciliation can
        leave a trailing unit unowned), so deriving both ends from atoms would
        silently lose units on punctuation-heavy streams.
        """
        if node <= 0:
            return self.interval.unit_start
        if node >= len(self.atoms):
            return self.interval.unit_end
        return self.atoms[node].unit_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_invisible": self.all_invisible,
            "atom_count": len(self.atoms),
            "candidate_count": len(self.nodes),
            "canonical_chars": self.canonical_chars,
            "cap_relief_nodes": self.cap_relief_nodes,
            "coalesced_atoms": self.coalesced_atoms,
            "edge_count": len(self.edges),
            "infeasible": None
            if self.infeasible is None
            else self.infeasible.to_dict(),
            "packer_steps": self.packer_steps,
            "relief_injections": self.relief_injections,
            "waivers": [waiver.to_dict() for waiver in self.waivers],
        }


def _edges_from(edges: Sequence[Edge]) -> dict[int, tuple[Edge, ...]]:
    grouped: dict[int, list[Edge]] = {}
    for edge in edges:
        grouped.setdefault(edge.start_node, []).append(edge)
    return {
        node: tuple(sorted(items, key=lambda e: e.end_node))
        for node, items in sorted(grouped.items())
    }


def _reachable(edges_from: Mapping[int, tuple[Edge, ...]], target: int) -> bool:
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for edge in edges_from.get(node, ()):
            if edge.end_node not in seen:
                seen.add(edge.end_node)
                stack.append(edge.end_node)
    return target in seen


def _relieve_over_cap_atoms(
    atoms: Sequence[LatticeAtom],
    units: Sequence[SourceUnit],
    max_cue_s: float,
    lang: str,
) -> tuple[tuple[LatticeAtom, ...], int]:
    """Expose the internal unit edges of any atom that is over the cap alone.

    Running this before the edge scan keeps the ordering the duration contract
    demands (relief, then waiver, then fallback) without making the scan itself
    order-dependent.
    """
    current = list(atoms)
    injections = 0
    guard = len(units) + len(current) + 2
    while guard > 0:
        guard -= 1
        target = next(
            (
                index
                for index, atom in enumerate(current)
                if atom.unit_end - atom.unit_start > 1
                and exclusively_owned(current, index)
                and _over_cap(current, index, index + 1, max_cue_s)
            ),
            None,
        )
        if target is None:
            break
        atom = current[target]
        for unit_id in range(atom.unit_start + 1, atom.unit_end):
            current = list(split_candidate_at_unit(current, unit_id, units, lang))
            injections += 1
    return tuple(current), injections


@dataclass(frozen=True)
class MixedEdges:
    """One scan of the mixed branch: its edges, exemptions, and what it needs."""

    edges: tuple[Edge, ...]
    waivers: tuple[Waiver, ...]
    infeasible: Infeasible | None
    packer_steps: int
    cap_nodes: tuple[int, ...] = ()


def _build_mixed_edges(
    atoms: Sequence[LatticeAtom],
    nodes: Sequence[int],
    profile: DisplayProfile,
    units: Sequence[SourceUnit],
    *,
    work: CanonicalWork | None = None,
) -> MixedEdges:
    """Scan every start node through the provably bounded candidate band.

    Spaced-language packing and aligned duration are monotone in the end node.
    Canonical no-space legality is not: kinsoku can repair a prefix when a later
    closing glyph arrives, so that branch stops only at the LAW's monotone
    stripped-cell lower bound. The duration blocker case (even the shortest
    candidate cue is over the cap) is the only place an exemption can enter,
    and it goes through the same resolution the all-invisible branch uses.

    That resolution answers three different things and they must not be
    conflated. It can say *the run splits* (a cap-legal partition exists at
    source-unit edges this node space happened to hide), *the run is exempt* (one
    word still sounding, held-chain evidence), or *the run is unwaivable*. Only
    the third is a terminal. Reading an empty waiver list as the terminal --
    which is what a bare ``if resolution.waivers: ... else: infeasible`` does --
    turns a SUCCESSFUL split into ``duration-unwaivable``, skips the held-chain
    test AD6-1 orders before it, and short-circuits the C16 relief valve, since
    the valve only runs where no ``infeasible`` was set. Split points are
    returned as ``cap_nodes`` for the caller to inject, exactly as C16 injects
    relief nodes.
    """
    lang = profile.language
    no_spaces = _no_spaces(lang)
    max_cue_s = profile.max_cue_s
    node_set = set(nodes)
    packer = (
        None
        if no_spaces
        else IncrementalPacker(lang, profile.max_line_length, profile.max_lines)
    )
    canonical_work = work if work is not None else CanonicalWork()
    edges: list[Edge] = []
    waivers: list[Waiver] = []
    cap_nodes: set[int] = set()
    infeasible: Infeasible | None = None
    count = len(atoms)
    for start in nodes:
        if start >= count:
            continue
        if packer is not None:
            packer.reset()
        surfaces: list[str] = []
        starts: list[float | None] = []
        ends: list[float | None] = []
        emitted = False
        for index in range(start, count):
            atom = atoms[index]
            surfaces.append(atom.text)
            starts.append(atom.start)
            ends.append(atom.end)
            end_node = index + 1
            if no_spaces:
                joined = _join(surfaces, lang)
                if band_scan_lower_bound_exceeded(joined, profile):
                    break
                if end_node not in node_set:
                    # This atom edge is not a candidate boundary. Keep scanning
                    # the raw source span, but do not spend canonical work on a
                    # projection the lattice could never admit.
                    continue
                measure = _canonical_pack_measure(
                    atoms, start, end_node, profile, canonical_work
                )
            else:
                assert packer is not None
                measure = packer.extend(atom.text)
            if not measure.fits:
                if no_spaces:
                    # Canonical wrapping is not monotone: a later kinsoku glyph
                    # can repair this prefix. Only the stripped-cell lower bound
                    # above is strong enough to terminate a no-space scan.
                    continue
                break
            low = span_min(starts)
            high = span_max(ends)
            over = (
                max_cue_s > 0
                and low is not None
                and high is not None
                and high - low > max_cue_s + CAP_EPS_S
            )
            if end_node in node_set and low is not None and high is not None:
                waiver: Waiver | None = None
                if over:
                    if emitted:
                        break
                    resolution = resolve_cap_partition(
                        atoms[start:end_node],
                        units,
                        max_cue_s=max_cue_s,
                        lang=lang,
                        allow_relief=False,
                    )
                    exposed = {start + cut for cut in resolution.cuts} - node_set
                    if exposed:
                        # The run IS splittable; this particular edge simply is
                        # not legal. Expose the cuts and let the caller re-scan.
                        cap_nodes |= exposed
                        break
                    if resolution.waivers:
                        waiver = resolution.waivers[0]
                        waivers.append(waiver)
                    elif resolution.infeasible is not None:
                        infeasible = resolution.infeasible
                        break
                    else:
                        # No split, no evidence, no typed terminal: the run has
                        # no legal edge from here, which the reachability check
                        # types honestly as no-path/relief-insufficient. Claiming
                        # ``duration-unwaivable`` would name a terminal the
                        # duration ladder never reached.
                        break
                edges.append(
                    Edge(
                        start_node=start,
                        end_node=end_node,
                        text=_join(surfaces, lang),
                        display_text=measure.text,
                        lines=measure.lines,
                        line_widths=measure.line_widths,
                        span_start=low,
                        span_end=high,
                        waiver=waiver,
                    )
                )
                emitted = True
            if over:
                break
        if infeasible is not None:
            break
    edges.sort(key=lambda e: (e.start_node, e.end_node))
    return MixedEdges(
        edges=tuple(edges),
        waivers=tuple(waivers),
        infeasible=infeasible,
        packer_steps=0 if packer is None else packer.steps,
        cap_nodes=tuple(sorted(cap_nodes)),
    )


def _forced_chain(
    atoms: Sequence[LatticeAtom],
    cuts: Sequence[int],
    waivers: Sequence[Waiver],
    lang: str,
) -> tuple[Edge, ...]:
    """The all-invisible branch's answer: one edge per cap-legal chunk.

    Emitting a chain rather than a free lattice is what makes this branch a
    *defined result* instead of a cheap one: the solver has exactly one path to
    find, its work is constant, and no layout search runs at all because an empty
    display projection fits any budget by construction.
    """
    by_chunk = {waiver.cue_index: waiver for waiver in waivers}
    edges: list[Edge] = []
    for chunk_index, (lo, hi) in enumerate(_chunk_bounds(cuts, len(atoms))):
        text = _join([atom.text for atom in atoms[lo:hi]], lang)
        display = strip_punct_for_subtitles(text)
        edges.append(
            Edge(
                start_node=lo,
                end_node=hi,
                text=text,
                display_text=display,
                lines=1,
                line_widths=(_vis_width(display),),
                span_start=span_min([atom.start for atom in atoms[lo:hi]]),
                span_end=span_max([atom.end for atom in atoms[lo:hi]]),
                waiver=by_chunk.get(chunk_index),
            )
        )
    return tuple(edges)


def build_interval_lattice(
    interval: HardInterval,
    layer: AtomLayer,
    profile: DisplayProfile,
    *,
    units: Sequence[SourceUnit],
    span_violations: Sequence[SpanViolation] = (),
) -> IntervalLattice:
    """Build one interval's legal cue set, or say precisely why there is none."""
    lang = profile.language
    blocking = [
        violation
        for violation in span_violations
        if interval.unit_start <= violation.unit_index < interval.unit_end
    ]
    if blocking:
        return IntervalLattice(
            interval=interval,
            atoms=(),
            nodes=(),
            edges=(),
            edges_from={},
            coalesced_atoms=0,
            all_invisible=False,
            relief_injections=0,
            waivers=(),
            infeasible=Infeasible(
                reason="span-preflight",
                detail="; ".join(
                    f"{v.unit_id}: {v.reason} ({v.detail})" for v in blocking
                ),
                unit_range=(interval.unit_start, interval.unit_end),
            ),
            packer_steps=0,
            canonical_chars=0,
        )

    raw = _reindex(layer.atoms[interval.node_start : interval.node_end])
    raw = tuple(
        _replace(atom, members=(position,)) for position, atom in enumerate(raw)
    )
    coalesced = coalesce_zero_display(raw, lang=lang)

    if coalesced.all_invisible:
        resolution = resolve_cap_partition(
            coalesced.atoms,
            units,
            max_cue_s=profile.max_cue_s,
            lang=lang,
        )
        atoms = resolution.atoms
        edges = (
            ()
            if resolution.infeasible is not None
            else _forced_chain(atoms, resolution.cuts, resolution.waivers, lang)
        )
        return IntervalLattice(
            interval=interval,
            atoms=atoms,
            nodes=(0, *resolution.cuts, len(atoms)),
            edges=edges,
            edges_from=_edges_from(edges),
            coalesced_atoms=coalesced.coalesced_atoms,
            all_invisible=True,
            relief_injections=resolution.relief_injections,
            waivers=resolution.waivers,
            infeasible=resolution.infeasible,
            packer_steps=0,
            canonical_chars=0,
        )

    atoms = coalesced.atoms
    coalesced_atoms = coalesced.coalesced_atoms
    relief_injections = 0
    if profile.max_cue_s > 0:
        relieved, relief_injections = _relieve_over_cap_atoms(
            atoms, units, profile.max_cue_s, lang
        )
        if relief_injections:
            # AD3-1, again: relief mints pieces by footprint and a piece that
            # owns only punctuation renders to nothing. An invisible atom inside
            # a MIXED interval breaks the very positive-display-progress
            # invariant the band and the early break are proved from, and admits
            # candidate edges whose whole display is empty. Re-coalescing
            # restores it. Where that folds the split straight back, relief was
            # not actually available on this atom, and the duration ladder falls
            # through to the held-chain waiver -- which is the AD6-1 ordering.
            recoalesced = coalesce_zero_display(relieved, lang=lang)
            atoms = recoalesced.atoms
            coalesced_atoms += recoalesced.coalesced_atoms
        else:
            atoms = relieved
    nodes = candidate_nodes(atoms, lang)

    granularity = granularity_check(atoms, nodes, profile)
    if granularity.collapsed:
        widened = nodes
        if relief_trigger(
            atoms,
            max_line_length=profile.max_line_length,
            max_lines=profile.max_lines,
        ):
            widened = tuple(
                sorted(
                    set(nodes)
                    | set(
                        relief_nodes(
                            atoms,
                            max_line_length=profile.max_line_length,
                            max_lines=profile.max_lines,
                        )
                    )
                )
            )
        if granularity_check(atoms, widened, profile).collapsed:
            return IntervalLattice(
                interval=interval,
                atoms=atoms,
                nodes=nodes,
                edges=(),
                edges_from={},
                coalesced_atoms=coalesced_atoms,
                all_invisible=False,
                relief_injections=relief_injections,
                waivers=(),
                infeasible=Infeasible(
                    reason=COARSE_GRANULARITY,
                    detail=(
                        f"{granularity.available_cuts} interior candidate "
                        f"boundaries where a legal partition needs at least "
                        f"{granularity.required_cuts}: the source units are "
                        "coarser than a cue"
                    ),
                    unit_range=(interval.unit_start, interval.unit_end),
                ),
                packer_steps=0,
                canonical_chars=0,
                cap_relief_nodes=0,
            )

    canonical_work = CanonicalWork()
    scan = _build_mixed_edges(atoms, nodes, profile, units, work=canonical_work)
    steps = scan.packer_steps
    cap_relief_nodes = 0
    guard = len(atoms) + 2
    while scan.cap_nodes and guard > 0:
        guard -= 1
        fresh = tuple(node for node in scan.cap_nodes if node not in set(nodes))
        if not fresh:
            break
        nodes = tuple(sorted(set(nodes) | set(fresh)))
        cap_relief_nodes += len(fresh)
        scan = _build_mixed_edges(atoms, nodes, profile, units, work=canonical_work)
        steps += scan.packer_steps
    edges, waivers, infeasible = scan.edges, scan.waivers, scan.infeasible
    edges_from = _edges_from(edges)

    if infeasible is None and not _reachable(edges_from, len(atoms)):
        if relief_trigger(
            atoms,
            max_line_length=profile.max_line_length,
            max_lines=profile.max_lines,
        ):
            injected = relief_nodes(
                atoms,
                max_line_length=profile.max_line_length,
                max_lines=profile.max_lines,
            )
            nodes = tuple(sorted(set(nodes) | set(injected)))
            scan = _build_mixed_edges(atoms, nodes, profile, units, work=canonical_work)
            edges, waivers, infeasible = scan.edges, scan.waivers, scan.infeasible
            steps += scan.packer_steps
            edges_from = _edges_from(edges)
            relief_injections += len(injected)
            if infeasible is None and not _reachable(edges_from, len(atoms)):
                infeasible = Infeasible(
                    reason="relief-insufficient",
                    detail=f"{len(injected)} injected nodes left no legal path",
                    unit_range=(interval.unit_start, interval.unit_end),
                )
        else:
            infeasible = Infeasible(
                reason="no-path",
                detail="no legal cue chain covers the interval",
                unit_range=(interval.unit_start, interval.unit_end),
            )

    return IntervalLattice(
        interval=interval,
        atoms=atoms,
        nodes=nodes,
        edges=edges,
        edges_from=edges_from,
        coalesced_atoms=coalesced_atoms,
        all_invisible=False,
        relief_injections=relief_injections,
        waivers=waivers,
        infeasible=infeasible,
        packer_steps=steps,
        canonical_chars=canonical_work.canonical_chars,
        cap_relief_nodes=cap_relief_nodes,
    )


@dataclass(frozen=True)
class DocumentLattice:
    """Every interval of one document, plus the evidence the build produced."""

    layer: AtomLayer
    barriers: tuple[HardBarrier, ...]
    intervals: tuple[HardInterval, ...]
    lattices: tuple[IntervalLattice, ...]
    span_violations: tuple[SpanViolation, ...]
    sentence_ends: SentenceEnds


def _cache_candidate_evidence(
    lattice: IntervalLattice, document: SegDocument
) -> IntervalLattice:
    """Attach W3 EvidenceSpan/lyric facts without changing edge admission.

    The edge set is already closed when this runs.  Thus singing evidence can
    neither admit nor reject a candidate; it only supplies the stable cache the
    cost and selected materializer share.  Missing display endpoints use the
    same deterministic prior-end chain as materialization: the latest finite
    source end before the candidate, or zero at the document front.  This keeps
    fully and partially untimed candidates typed without inventing an acoustic
    anchor.
    """
    from .speaker_evidence import lyric_for_evidence, make_evidence_span

    prior_end: list[float] = [0.0]
    latest = 0.0
    for unit in document.units:
        if (
            unit.end is not None
            and not isinstance(unit.end, bool)
            and math.isfinite(unit.end)
        ):
            latest = float(unit.end)
        prior_end.append(latest)

    decorated: list[Edge] = []
    for edge in lattice.edges:
        low = lattice.unit_bound(edge.start_node)
        high = lattice.unit_bound(edge.end_node)
        if low < high:
            fallback = prior_end[low]
            input_start = (
                float(edge.span_start)
                if edge.span_start is not None and math.isfinite(edge.span_start)
                else fallback
            )
            input_end = (
                float(edge.span_end)
                if edge.span_end is not None and math.isfinite(edge.span_end)
                else input_start
            )
            span = make_evidence_span(
                document.units,
                (low, high),
                input_start=input_start,
                input_end=input_end,
            )
            decorated.append(
                dataclass_replace(
                    edge,
                    evidence_span=span,
                    lyric=lyric_for_evidence(span, document.sing_spans),
                )
            )
        else:
            decorated.append(edge)
    edges = tuple(decorated)
    return dataclass_replace(lattice, edges=edges, edges_from=_edges_from(edges))


def build_document_lattice(
    document: SegDocument, *, cache_speaker_evidence: bool = False
) -> DocumentLattice:
    """Preflight, atom layer, barriers, intervals, per-interval lattices.

    Raises nothing. A profile violation is the caller's business (the shadow
    entry point refuses the whole measurement before reaching here), and a span
    violation only marks the interval that contains it.
    """
    span_violations = preflight_units(document.units)
    layer = build_atom_layer(document)
    barriers = build_barriers(layer, document.profile)
    intervals = build_intervals(layer, barriers)
    raw_lattices = tuple(
        build_interval_lattice(
            interval,
            layer,
            document.profile,
            units=document.units,
            span_violations=span_violations,
        )
        for interval in intervals
    )
    lattices = (
        tuple(_cache_candidate_evidence(item, document) for item in raw_lattices)
        if cache_speaker_evidence
        else raw_lattices
    )
    return DocumentLattice(
        layer=layer,
        barriers=barriers,
        intervals=intervals,
        lattices=lattices,
        span_violations=span_violations,
        sentence_ends=sentence_end_nodes(layer),
    )
