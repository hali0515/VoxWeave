"""The lattice data model: knobs, typed reasons, preflights, atoms and edges.

The leaf vocabulary of the boundary lattice. It imports no other ``boundary_*``
module, so the node-set and packing modules and the lattice builder can all
depend on it without a cycle. ``boundary_lattice`` re-exports every name here.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .layout import _join, _line_budget_width, _vis_width
from .partition_check import Waiver
from .segdoc import DisplayProfile, SourceUnit
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

#: The infeasibility of an interval that WOULD have a legal path if its
#: untimed words carried timing: every cue able to cover them is timeless, and
#: a timeless cue is never admitted in a mixed interval. The v1 fallback it
#: takes is the declared ``v2-untimed-chunk-fallback`` policy delta.
UNTIMED_RUN: str = "untimed-run"

INFEASIBLE_REASONS: tuple[str, ...] = (
    COARSE_GRANULARITY,
    "duration-unwaivable",
    "no-path",
    "relief-insufficient",
    "span-preflight",
    UNTIMED_RUN,
)

SPAN_VIOLATION_REASONS: tuple[str, ...] = (
    "bool-bound",
    "ends-non-monotone",
    "non-finite",
    "reversed",
    "starts-non-monotone",
)


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
        # A non-finite value is written as its name ("nan", "inf"): the
        # artifact stays valid JSON and still says what was refused.
        value: float | str = (
            self.value if math.isfinite(self.value) else str(self.value)
        )
        return {"key": self.key, "reason": self.reason, "value": value}


#: Every float knob of :class:`DisplayProfile`. NaN compares false against any
#: bound, so without a finiteness check a NaN cap reads as "no cap" and a NaN
#: threshold crashes deep inside the solver.
_PROFILE_FLOAT_KNOBS: tuple[str, ...] = (
    "clause_ms",
    "vad_skip_ms",
    "offline_ms",
    "min_cue_s",
    "max_cue_s",
    "glue_gap_s",
    "cps",
    "lag_out_s",
    "shot_snap_s",
)


def preflight_profile(profile: DisplayProfile) -> tuple[ProfileViolation, ...]:
    """Refuse a profile whose knobs have no defined meaning.

    Zero **disables** the duration cap, which is what the timing pass' own truth
    test means by it; a negative cap is refused outright rather than
    reinterpreted, so the shadow never quietly disagrees with cleanup about what
    a negative cap does. A NaN or infinite knob is refused too. A non-empty
    result aborts the measurement for that document -- an invalid measurement,
    never a silent fallback.
    """
    out: list[ProfileViolation] = []
    non_finite = {
        key
        for key in _PROFILE_FLOAT_KNOBS
        if not math.isfinite(float(getattr(profile, key)))
    }
    for key in _PROFILE_FLOAT_KNOBS:
        if key in non_finite:
            out.append(
                ProfileViolation(key, float(getattr(profile, key)), "non-finite")
            )
    for key in ("clause_ms", "offline_ms", "vad_skip_ms"):
        value = float(getattr(profile, key))
        if key not in non_finite and value <= 0:
            out.append(ProfileViolation(key, value, "not-positive"))
    if "max_cue_s" not in non_finite and profile.max_cue_s < 0:
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

    ``text`` is the pre-strip surface (what the cue will store), ``display`` what
    every cue holding the atom still shows of it, whatever follows -- empty means
    some cue renders the atom to nothing. The footprint ``[unit_start, unit_end)``
    is the only granularity-safe way to slice the unit stream back, so every
    ownership question is answered by it rather than by re-deriving a character
    cursor.
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
    input_start: float | None = None
    input_end: float | None = None
    evidence_unavailable_reason: str | None = None

    @property
    def vis_width(self) -> int:
        return _vis_width(self.display_text)

    @property
    def balance(self) -> float:
        if self.lines != 2:
            return 0.0
        return float(abs(self.line_widths[0] - self.line_widths[1]))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "display_text": self.display_text,
            "end_node": self.end_node,
            "line_widths": list(self.line_widths),
            "lines": self.lines,
            "span": [self.span_start, self.span_end],
            "start_node": self.start_node,
            "waiver": None if self.waiver is None else self.waiver.to_dict(),
        }
        if self.evidence_span is not None:
            result.update(
                {
                    "evidence_span": self.evidence_span.to_dict(),
                    "input_span": [self.input_start, self.input_end],
                    "lyric": self.lyric,
                }
            )
        elif self.evidence_unavailable_reason is not None:
            result.update(
                {
                    "evidence_span_refusal": self.evidence_unavailable_reason,
                    "input_span": [self.input_start, self.input_end],
                    "lyric": False,
                }
            )
        return result


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
