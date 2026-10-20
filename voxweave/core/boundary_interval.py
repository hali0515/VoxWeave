"""One interval's committed answer: select a path or adopt v1, then cues.

Materializing a chosen edge chain into cues, the typed fallback that adopts
v1's own cues for an infeasible interval, and :func:`optimize_interval`, which
ties the DP, the margin policy and the partition validator together for one
interval. Nothing here knows about the document driver: the cost tables and the
cost context arrive as arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .boundary_cost import CostContext
from .boundary_lattice import (
    Edge,
    HardInterval,
    IntervalLattice,
    LatticeAtom,
    _resolve_candidate_evidence,
    _resolve_edge_input_bounds,
)
from .boundary_select import (
    CostTables,
    Selection,
    V1Partition,
    _margin_summary,
    _pinned_neighbour_margins,
    _select,
    solve_interval,
)
from .layout import _join
from .partition_check import (
    PartitionCheckResult,
    Waiver,
    check_partition,
    owned_unit_ids,
)
from .schema import Cue, Unit
from .segdoc import SourceUnit
from .speaker_evidence import EvidenceSpan, SpeakerPricingSummary
from .subunit import speech_span_units


def _resolve_edge_for_previous(
    lattice: IntervalLattice,
    edge: Edge,
    ctx: CostContext,
    *,
    previous_end: float,
) -> Edge:
    """Resolve/cache an edge against one actual predecessor-state value."""
    input_start, input_end = _resolve_edge_input_bounds(
        edge,
        previous_end=previous_end,
    )
    if (
        edge.input_start == input_start
        and edge.input_end == input_end
        and (
            isinstance(edge.evidence_span, EvidenceSpan)
            or edge.evidence_unavailable_reason is not None
        )
    ):
        return edge
    return _resolve_candidate_evidence(
        edge,
        lattice,
        units=ctx.units,
        sing_spans=ctx.sing_spans,
        previous_end=previous_end,
    )


# ---------------------------------------------------------- materialization


def materialize_cues(
    edges: Sequence[Edge],
    atoms: Sequence[LatticeAtom],
    lang: str,
    *,
    fallback_start: float = 0.0,
    units: Sequence[SourceUnit] | None = None,
) -> tuple[Cue, ...]:
    """Turn a chosen edge chain into cues with the engine's own cue shapes.

    ``word_data`` is *atom* level and carries ``text``, exactly as the v1
    materializer emits it: handing back raw per-character entries instead would
    make every later reader fall through to a character cursor and report a total
    diff for reasons that have nothing to do with boundaries.

    One divergence is unavoidable and deliberate. v1 lays an untimed chunk over
    the stretch between the nearest real bounds inside its parent cue (and short
    made-up cues borrow display time from their siblings), and a v2 partition has
    no parent cue, so the fallback here is the previous cue's end (or
    ``fallback_start`` at the front). The acoustic anchors take no fallback at all in either engine:
    invented display time must never be laundered into the evidence layer.  If
    ``units`` contains derived provenance, each cue's exact start/end is taken
    only from its first/last owned aligner unit; an all-aligner stream retains
    the legacy edge fold byte-for-byte.
    """
    cues: list[Cue] = []
    # Absolute byte freeze for the ordinary stream: when every source unit is
    # still aligner-provenance, retain the original edge span fold verbatim.
    # Only a mixed-provenance shadow stream takes the W2 sibling fold.
    provenance_aware = bool(units) and any(
        unit.provenance != "aligner" for unit in units or ()
    )
    previous_end = float(fallback_start)
    for edge in edges:
        chunk = list(atoms[edge.start_node : edge.end_node])
        speech_start = edge.span_start
        speech_end = edge.span_end
        if provenance_aware and units is not None and chunk:
            unit_start = chunk[0].unit_start
            unit_end = chunk[-1].unit_end
            speech_start, speech_end = speech_span_units(units[unit_start:unit_end])
        # Derived sub-unit spans remain the display/cap time authority even when
        # provenance correctly withholds an acoustic anchor.  Conflating these
        # channels would collapse every all-refined cue to zero duration.
        if (edge.input_start is None) != (edge.input_end is None):
            raise ValueError("candidate input-bound cache is only partially populated")
        if edge.input_start is not None and edge.input_end is not None:
            start = float(edge.input_start)
            end = float(edge.input_end)
        else:
            start, end = _resolve_edge_input_bounds(
                edge,
                previous_end=previous_end,
            )
        word_data: list[Unit] = [
            {"text": atom.text, "start": atom.start, "end": atom.end} for atom in chunk
        ]
        cue: Cue = {
            "text": _join([atom.text for atom in chunk], lang),
            "start": start,
            "end": end,
            "word_data": word_data,
            "speech_start": speech_start,
            "speech_end": speech_end,
        }
        if edge.lyric:
            cue["lyric"] = True
        cues.append(cue)
        previous_end = end
    return tuple(cues)


@dataclass(frozen=True)
class AdoptedV1:
    """The typed fallback: v1's own cues stand in for an infeasible interval."""

    unit_range: tuple[int, int]
    fallback_expansion_units: tuple[int, int] | None
    cues: tuple[Cue, ...]
    reason: str
    cuts: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuts": list(self.cuts),
            "fallback_expansion_units": None
            if self.fallback_expansion_units is None
            else list(self.fallback_expansion_units),
            "reason": self.reason,
            "unit_range": list(self.unit_range),
        }


def _adopt_v1(
    interval: HardInterval,
    reason: str,
    v1: V1Partition | None,
    unit_count: int,
) -> AdoptedV1:
    """Adopt the smallest set of COMPLETE v1 cues that covers the interval.

    Complete cues, never a slice of one: half a v1 cue is neither engine's
    answer. When the covering cues reach past the interval the adopted region
    expands with them and the expansion is recorded, so a reader can see that the
    fallback owns more than the interval it was asked about.
    """
    span = (interval.unit_start, interval.unit_end)
    if v1 is None or not v1.cues:
        return AdoptedV1(
            unit_range=span, fallback_expansion_units=None, cues=(), reason=reason
        )
    bounds = owned_unit_ids(v1.cuts, unit_count)
    picked = [
        index
        for index, (low, high) in enumerate(bounds)
        if low < interval.unit_end and high > interval.unit_start
    ]
    if not picked:
        return AdoptedV1(
            unit_range=span, fallback_expansion_units=None, cues=(), reason=reason
        )
    low = bounds[picked[0]][0]
    high = bounds[picked[-1]][1]
    covered = (low, high)
    return AdoptedV1(
        unit_range=covered,
        fallback_expansion_units=None if covered == span else covered,
        cues=tuple(v1.cues[index] for index in picked if index < len(v1.cues)),
        reason=reason,
        cuts=tuple(bounds[index][1] for index in picked[:-1]),
    )


#: Why a solvable interval carries v1 cues: a neighbouring fallback's complete
#: v1 cues reach into it.
ADOPTION_ABSORBED: str = "absorbed-by-fallback"


def _adoption_plan(
    lattices: Sequence[IntervalLattice],
    v1: V1Partition | None,
    unit_count: int,
) -> dict[int, AdoptedV1]:
    """Every fallback interval's adoption, grown until the adoptions tile.

    A covering v1 cue can straddle a barrier, so a fallback's complete cues can
    reach into the neighbouring interval, before or after it. Keeping that
    neighbour's own cues as well would own the straddled units twice, and the
    document pass would report the duplicate against the neighbour -- as a false
    exit-driving v2 violation when the neighbour was optimized. So a fallback
    absorbs every interval its adoption touches, repeating until the adopted
    span starts and ends on both an interval boundary and a v1 cue boundary.
    Inside that span each member carries the v1 cues that start in it, so the
    members' ranges still tile the stream in interval order; a member whose
    units all sit inside an earlier member's cue carries an empty range. An
    adoption that reaches no neighbour is exactly :func:`_adopt_v1`'s.
    """
    if v1 is None or not v1.cues:
        return {}
    bounds = owned_unit_ids(v1.cuts, unit_count)
    spans = [(item.interval.unit_start, item.interval.unit_end) for item in lattices]
    plan: dict[int, AdoptedV1] = {}
    for seed, seed_lattice in enumerate(lattices):
        infeasible = seed_lattice.infeasible
        if infeasible is None or seed in plan:
            continue
        low, high = spans[seed]
        first = last = seed
        while True:
            picked = [
                index for index, (a, b) in enumerate(bounds) if a < high and b > low
            ]
            grown_low = min([low, *(bounds[index][0] for index in picked)])
            grown_high = max([high, *(bounds[index][1] for index in picked)])
            touched = [
                position
                for position, (start, end) in enumerate(spans)
                if start < grown_high and end > grown_low
            ]
            grown_first = min([first, *touched])
            grown_last = max([last, *touched])
            grown_low = min(grown_low, spans[grown_first][0])
            grown_high = max(grown_high, spans[grown_last][1])
            if (grown_low, grown_high, grown_first, grown_last) == (
                low,
                high,
                first,
                last,
            ):
                break
            low, high, first, last = grown_low, grown_high, grown_first, grown_last
        if first == last:
            plan[seed] = _adopt_v1(
                seed_lattice.interval, infeasible.reason, v1, unit_count
            )
            continue
        members = range(first, last + 1)
        owned: dict[int, list[int]] = {position: [] for position in members}
        for index in picked:
            owner = first
            for position in members:
                if spans[position][0] <= bounds[index][0]:
                    owner = position
            owned[owner].append(index)
        cursor = low
        for position in members:
            indices = owned[position]
            covered = (
                (bounds[indices[0]][0], bounds[indices[-1]][1])
                if indices
                else (cursor, cursor)
            )
            cursor = covered[1]
            terminal = lattices[position].infeasible
            plan[position] = AdoptedV1(
                unit_range=covered,
                fallback_expansion_units=None
                if covered == spans[position]
                else covered,
                cues=tuple(v1.cues[index] for index in indices if index < len(v1.cues)),
                reason=ADOPTION_ABSORBED if terminal is None else terminal.reason,
                cuts=tuple(bounds[index][1] for index in indices[:-1]),
            )
    return plan


# ------------------------------------------------------- interval solutions


@dataclass(frozen=True)
class IntervalSolution:
    """One interval's committed answer plus everything a reader must audit it."""

    interval: HardInterval
    lattice: IntervalLattice
    selection: Selection | None
    adopted: AdoptedV1 | None
    cues: tuple[Cue, ...]
    partition_units: tuple[int, ...]
    validator_raw: PartitionCheckResult
    dp_relaxations: int
    packer_steps: int
    waivers: tuple[Waiver, ...] = ()
    speaker_pricing: SpeakerPricingSummary | None = None
    decision_margins: tuple[float, ...] = ()

    @property
    def optimized(self) -> bool:
        return self.adopted is None

    @property
    def unit_range(self) -> tuple[int, int]:
        if self.adopted is not None:
            return self.adopted.unit_range
        return (self.interval.unit_start, self.interval.unit_end)

    @property
    def coarse_caused(self) -> bool:
        """Whether shared-unit atoms caused this interval's infeasibility.

        ``coarse-granularity`` is only one possible terminal for that input
        class.  A locally collapsed stream can honestly end as ``no-path`` or
        ``relief-insufficient`` too, so the cause is the shared footprint plus
        any typed infeasibility, never the terminal's spelling.
        """
        atoms = self.lattice.atoms
        shared = any(
            left.unit_end > right.unit_start for left, right in zip(atoms, atoms[1:])
        )
        return self.lattice.infeasible is not None and shared

    def to_dict(self) -> dict[str, Any]:
        selection = self.selection
        infeasible = self.lattice.infeasible
        return {
            "adopted_v1": self.adopted is not None,
            "all_invisible": self.lattice.all_invisible,
            "atom_count": len(self.lattice.atoms),
            "barrier_left": self.interval.left.kind,
            "barrier_right": self.interval.right.kind,
            "candidate_count": len(self.lattice.nodes),
            "canonical_chars": self.lattice.canonical_chars,
            "cap_relief_nodes": self.lattice.cap_relief_nodes,
            "coalesced_atoms": self.lattice.coalesced_atoms,
            "coarse_caused": self.coarse_caused,
            "dp_relaxations": self.dp_relaxations,
            "edge_count": len(self.lattice.edges),
            "fallback_expansion_units": None
            if self.adopted is None or self.adopted.fallback_expansion_units is None
            else list(self.adopted.fallback_expansion_units),
            "infeasible": None if infeasible is None else infeasible.to_dict(),
            "interval_index": self.interval.index,
            "margin_summary": _margin_summary(
                self.decision_margins,
                selected_cut_count=0
                if selection is None
                else len(selection.policy_selected.cuts),
            ),
            "node_range": [self.interval.node_start, self.interval.node_end],
            "packer_steps": self.packer_steps,
            "policy_selected": None
            if selection is None
            else selection.policy_selected.to_dict(),
            "raw_optimum": None
            if selection is None
            else selection.raw_optimum.to_dict(),
            "relief_injections": self.lattice.relief_injections,
            "selected_is_v1": None
            if selection is None or not selection.v1_supplied
            else selection.selected_is_v1,
            "unit_range": list(self.unit_range),
            "v1_cost_under_v2": None
            if selection is None or selection.v1_cost_under_v2 is None
            else selection.v1_cost_under_v2.to_dict(),
            "v1_illegality": None if selection is None else selection.v1_illegality,
            "v1_path_legal": None
            if selection is None or not selection.v1_supplied
            else selection.v1_path_legal,
            "v2_partition": list(self.partition_units),
            "validator_raw": self.validator_raw.to_dict(),
            "waivers": [waiver.to_dict() for waiver in self.waivers],
        }


def _path_edges(lattice: IntervalLattice, cuts: Sequence[int]) -> tuple[Edge, ...]:
    index = {(edge.start_node, edge.end_node): edge for edge in lattice.edges}
    nodes = (0, *cuts, len(lattice.atoms))
    return tuple(index[(left, right)] for left, right in zip(nodes, nodes[1:]))


def _resolve_selected_path(
    lattice: IntervalLattice,
    tables: CostTables,
    cuts: Sequence[int],
) -> tuple[IntervalLattice, tuple[Edge, ...]]:
    """Resolve the selected chain and install those exact facts before sealing."""
    selected = _path_edges(lattice, cuts)
    ctx = tables.speaker_context
    if ctx is None:
        return lattice, selected

    previous_end = tables.fallback_start
    resolved: list[Edge] = []
    for edge in selected:
        item = _resolve_edge_for_previous(
            lattice,
            edge,
            ctx,
            previous_end=previous_end,
        )
        if item.input_end is None:
            raise ValueError("selected speaker edge has no resolved input end")
        resolved.append(item)
        previous_end = float(item.input_end)

    replacements = {(edge.start_node, edge.end_node): edge for edge in resolved}
    edges = tuple(
        replacements.get((edge.start_node, edge.end_node), edge)
        for edge in lattice.edges
    )
    # One pass over the edges (not one per node): same keys in lattice.nodes
    # order, same per-node edge order as the edge tuple.
    nodes = set(lattice.nodes)
    grouped: dict[int, list[Edge]] = {}
    for edge in edges:
        if edge.start_node in nodes:
            grouped.setdefault(edge.start_node, []).append(edge)
    edges_from: dict[int, tuple[Edge, ...]] = {
        node: tuple(grouped[node]) for node in lattice.nodes if node in grouped
    }
    sealed_lattice = replace(lattice, edges=edges, edges_from=edges_from)
    return sealed_lattice, tuple(resolved)


def optimize_interval(
    lattice: IntervalLattice,
    tables: CostTables,
    ctx: CostContext,
    *,
    units: Sequence[SourceUnit],
    v1: V1Partition | None = None,
    fallback_start: float = 0.0,
    adopted: AdoptedV1 | None = None,
) -> IntervalSolution:
    """Solve one interval, or adopt v1's cues for it and say why.

    ``ctx`` is a deviation from the reviewed signature: the validator needs the
    resolved display profile and materialization needs the language, and both
    already ride on the cost context rather than being threaded a second time.
    ``adopted`` is the document's adoption plan for this interval (see
    :func:`_adoption_plan`); without one an infeasible interval adopts alone.
    """
    interval = lattice.interval
    profile = ctx.profile
    lang = profile.language

    if adopted is None and lattice.infeasible is not None:
        adopted = _adopt_v1(interval, lattice.infeasible.reason, v1, len(units))
    if adopted is not None:
        low, high = adopted.unit_range
        return IntervalSolution(
            interval=interval,
            lattice=lattice,
            selection=None,
            adopted=adopted,
            cues=adopted.cues,
            partition_units=adopted.cuts,
            # An empty range with no cue has nothing to conserve: its units
            # belong to a cue an earlier member of the same fallback adopted.
            validator_raw=PartitionCheckResult(
                origin="v1",
                stage="raw",
                violations=(),
                waivers=(),
                cue_count=0,
                unit_count=0,
            )
            if low == high and not adopted.cues
            else check_partition(
                [cut - low for cut in adopted.cuts],
                adopted.cues,
                units=units[low:high],
                profile=profile,
                # Units no adopted v1 cue covers are v2's failure to cover them.
                origin="v1" if adopted.cues else "v2",
                stage="raw",
            ),
            dp_relaxations=0,
            packer_steps=lattice.packer_steps,
            speaker_pricing=tables.speaker_pricing,
        )

    dp = solve_interval(lattice, tables)
    selection = _select(lattice, tables, dp, v1)
    decision_margins = _pinned_neighbour_margins(
        lattice, tables, selection.policy_selected
    )
    chosen = selection.policy_selected
    selected_lattice, edges = _resolve_selected_path(lattice, tables, chosen.cuts)
    cues = materialize_cues(
        edges,
        selected_lattice.atoms,
        lang,
        fallback_start=fallback_start,
        units=units,
    )
    waivers = tuple(
        replace(edge.waiver, cue_index=index)
        for index, edge in enumerate(edges)
        if edge.waiver is not None
    )
    partition_units = tuple(lattice.unit_bound(node) for node in chosen.cuts)
    low, high = interval.unit_start, interval.unit_end
    return IntervalSolution(
        interval=interval,
        lattice=selected_lattice,
        selection=selection,
        adopted=None,
        cues=cues,
        partition_units=partition_units,
        validator_raw=check_partition(
            [cut - low for cut in partition_units],
            cues,
            units=units[low:high],
            profile=profile,
            origin="v2",
            stage="raw",
            waivers={waiver.cue_index: waiver for waiver in waivers},
        ),
        dp_relaxations=dp.relaxations,
        packer_steps=lattice.packer_steps,
        waivers=waivers,
        speaker_pricing=tables.speaker_pricing,
        decision_margins=decision_margins,
    )
