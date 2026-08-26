"""The exact whole-interval solver, its selection policy, and the artifact.

Nothing about the search is approximate. With the tile machinery deleted, each
hard interval is solved by one forward shortest-path pass over its whole legal
edge set, so margins, the runner-up and the selection policy are *whole-interval*
quantities rather than per-window ones that could disagree at a seam. That is
only affordable because legality is self-bounding: coalescing gives every atom
positive display width, so a legal cue spans at most ``band_atoms(profile)``
atoms and the edge scan's early break provably fires.

Three properties make the exactness claim testable rather than merely asserted:

* **one cost table.** Every edge and every candidate cut is priced exactly once,
  and the DP, :func:`score_path` and the v1 reference all read from that table.
  A brute-force enumeration compared against a solver that priced its own paths
  would only prove the solver agrees with itself.
* **a local tie-break.** At equal quantized total the DP prefers the smaller
  predecessor node, which is O(1) per comparison and carries no path tuples. The
  induced canonical path is characterised globally as: among all optima minimise
  the last cut, then the one before it, and so on -- and that is what the
  brute-force test compares against.
* **two counters, separately.** DP relaxations count ``(node, outgoing edge)``
  pairs, not 2-best ranks; packer extensions are counted by the lattice. Both are
  asserted against the resolved band by the tests rather than by production
  asserts, because a work bound that fires in production is a crash, not a proof.

The selection policy is deliberately conservative: v1's partition wins whenever
it is a legal path here and within :data:`POLICY_MARGIN` of the optimum. The
shadow exists to find where v2 is *convincingly* better, and a margin is what
separates that from a rounding error.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .boundary_cost import (
    POLICY_NAME,
    POLICY_VERSION,
    CostBreakdown,
    CostContext,
    cut_cost,
    edge_cost,
    pause_knees,
    quantize,
    sum_breakdowns,
)
from .boundary_lattice import (
    CAP_EPS_S,
    INFLUENCE_RADIUS_UNITS,
    AtomLayer,
    DocumentLattice,
    Edge,
    HardInterval,
    IncrementalPacker,
    IntervalLattice,
    LatticeAtom,
    ProfileViolation,
    build_barriers,
    build_document_lattice,
    held_chain_continuous,
    preflight_profile,
    span_max,
    span_min,
)
from .layout import _join
from .partition_check import (
    PartitionCheckResult,
    Waiver,
    check_partition,
    owned_unit_ids,
)
from .schema import Cue, Unit
from .segdoc import DisplayProfile, SegDocument, SourceUnit
from .timing_preview import DisplayTimingPreview, LegacyCleanupPreview

__all__ = [
    "ENGINE_V2",
    "POLICY_DELTAS",
    "POLICY_MARGIN",
    "POLICY_NAME",
    "POLICY_VERSION",
    "SCHEMA_VERSION",
    "AdoptedV1",
    "CostTables",
    "DPResult",
    "DocumentSolution",
    "IntervalSolution",
    "PathResult",
    "Selection",
    "V1Partition",
    "V1Reference",
    "build_cost_context",
    "build_cost_tables",
    "materialize_cues",
    "optimize_document",
    "optimize_interval",
    "score_path",
    "score_v1_global",
    "shadow_artifact",
    "solve_interval",
]

#: The engine name every shadow artifact is stamped with.
ENGINE_V2: str = "boundary-optimizer-v2"

#: Artifact schema version. Bumped when a reader would have to change.
SCHEMA_VERSION: int = 1

#: How much worse than the raw optimum a legal v1 path may be and still be
#: selected. A margin rather than an equality test because the point of the
#: shadow lane is to migrate only where v2 is *convincingly* better, not
#: wherever it is better by a rounding error.
POLICY_MARGIN: float = 1.0

#: Places where v2 knowingly does something v1 does not. Recorded on every
#: artifact so a reader never has to discover a divergence by diffing output.
POLICY_DELTAS: tuple[str, ...] = (
    "barrier-ignores-at-boundary",
    "missing-pause-evidence-1.5",
    "v2-untimed-chunk-fallback",
)


# ---------------------------------------------------------------- cost tables


@dataclass(frozen=True)
class CostTables:
    """Every edge and cut price of one interval, computed exactly once."""

    edges: Mapping[tuple[int, int], CostBreakdown]
    cuts: Mapping[int, CostBreakdown]


def build_cost_context(
    document: SegDocument,
    lattice: DocumentLattice,
    *,
    preview: DisplayTimingPreview | None = None,
    v1: V1Partition | None = None,
) -> CostContext:
    """Bundle everything a cost term needs that is not the edge or the cut.

    The preview defaults to the mirror of today's cleanup pass; P5 hands in the
    finalizer's own preview and nothing else here changes.
    """
    return CostContext(
        profile=document.profile,
        preview=LegacyCleanupPreview() if preview is None else preview,
        speech_spans=document.vad_speech,
        shot_changes=document.shot_changes,
        sentence_nodes=lattice.sentence_ends.nodes,
        v1_cut_units=frozenset() if v1 is None else frozenset(v1.cuts),
        layer=lattice.layer,
    )


def _document_nodes(lattice: IntervalLattice, layer: AtomLayer) -> tuple[int, ...]:
    """Interval-local node -> document atom-stream node.

    Two coordinate systems meet here and neither survives the other's
    transformations: sentence ends are recorded in document atom-stream nodes on
    the *raw* stream, while an interval's nodes are post-coalescing and
    post-relief. Atom membership cannot bridge them -- a relief split mints a
    piece with no member provenance at all -- so the bridge is the one coordinate
    both sides agree on, the source-unit id. Unit bounds are non-decreasing along
    the atom stream, so the lookup is a bisect rather than a scan.
    """
    interval = lattice.interval
    width = interval.node_end - interval.node_start
    bounds = [
        layer.unit_bound(node)
        for node in range(interval.node_start, interval.node_end + 1)
    ]
    out: list[int] = []
    for node in range(len(lattice.atoms) + 1):
        if node <= 0:
            out.append(interval.node_start)
        elif node >= len(lattice.atoms):
            out.append(interval.node_end)
        else:
            position = bisect.bisect_left(bounds, lattice.unit_bound(node))
            out.append(interval.node_start + min(position, width))
    return tuple(out)


def build_cost_tables(lattice: IntervalLattice, ctx: CostContext) -> CostTables:
    """Price every legal edge and every candidate cut of one interval.

    The DP, an independent path scorer and the v1 reference all read from here,
    so the solver cannot disagree with a path someone else scored -- which is
    what makes the brute-force equality test meaningful rather than circular.
    """
    profile = ctx.profile
    atoms = lattice.atoms
    document_nodes = _document_nodes(lattice, ctx.layer)
    sentence_nodes = ctx.sentence_nodes

    edges: dict[tuple[int, int], CostBreakdown] = {}
    for edge in lattice.edges:
        left = document_nodes[edge.start_node]
        right = document_nodes[edge.end_node]
        edges[(edge.start_node, edge.end_node)] = edge_cost(
            edge,
            atoms,
            profile=profile,
            preview=ctx.preview,
            next_start=ctx.next_start_after(right),
            sentence_cross_count=sum(
                1 for node in sentence_nodes if left < node < right
            ),
        )

    cuts: dict[int, CostBreakdown] = {}
    for node in lattice.nodes:
        if not 0 < node < len(atoms):
            continue
        cuts[node] = cut_cost(
            atoms[node - 1],
            atoms[node],
            unit_id=lattice.unit_bound(node),
            profile=profile,
            speech_spans=ctx.speech_spans,
            shot_changes=ctx.shot_changes,
            v1_cut_units=ctx.v1_cut_units,
        )
    return CostTables(edges=edges, cuts=cuts)


# ------------------------------------------------------------------- paths


@dataclass(frozen=True)
class PathResult:
    """One scored node path: its cuts, its total, and how the total was made."""

    cuts: tuple[int, ...]
    total: float
    edge_breakdowns: tuple[CostBreakdown, ...]
    cut_breakdowns: tuple[CostBreakdown, ...]
    breakdown: CostBreakdown
    unit_cuts: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Node ids in Python, source-unit ids in the artifact.

        ``cuts`` is the unit-space projection because every persisted coordinate
        is a source-unit id; the node tuple is kept alongside it so a reader
        debugging the solver can still see what the DP actually chose.
        """
        return {
            "breakdown": self.breakdown.to_dict(),
            "cuts": list(self.unit_cuts),
            "node_cuts": list(self.cuts),
            "total": self.total,
        }


def _assemble_path(
    lattice: IntervalLattice, tables: CostTables, cuts: Sequence[int]
) -> PathResult:
    count = len(lattice.atoms)
    nodes = (0, *cuts, count)
    total = 0.0
    edge_parts: list[CostBreakdown] = []
    cut_parts: list[CostBreakdown] = []
    for left, right in zip(nodes, nodes[1:]):
        edge = tables.edges.get((left, right))
        if edge is None:
            raise ValueError(f"edge({left}, {right}): no legal cue spans these atoms")
        edge_parts.append(edge)
        total = quantize(total + edge.total)
        if right != count:
            cut = tables.cuts.get(right)
            if cut is None:
                raise ValueError(f"node {right} is not a candidate boundary")
            cut_parts.append(cut)
            total = quantize(total + cut.total)
    return PathResult(
        cuts=tuple(cuts),
        total=total,
        edge_breakdowns=tuple(edge_parts),
        cut_breakdowns=tuple(cut_parts),
        breakdown=sum_breakdowns([*edge_parts, *cut_parts]),
        unit_cuts=tuple(lattice.unit_bound(node) for node in cuts),
    )


def score_path(
    lattice: IntervalLattice, tables: CostTables, cuts: Sequence[int]
) -> PathResult:
    """Score an arbitrary node path over one interval's lattice.

    The accumulation order is fixed and shared with the DP -- edge, then the cut
    that opened the next cue -- so an independently scored path and the solver's
    own total agree bit for bit rather than to a tolerance. A path whose
    consecutive pair is not a legal edge raises, which is how the v1 reference
    learns that v1's partition is not expressible here.
    """
    return _assemble_path(lattice, tables, cuts)


@dataclass(frozen=True)
class DPResult:
    """The optimum, its best path-distinct alternative, and the work spent."""

    best: PathResult
    runner_up: PathResult | None
    relaxations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best.to_dict(),
            "relaxations": self.relaxations,
            "runner_up": None if self.runner_up is None else self.runner_up.to_dict(),
        }


def solve_interval(lattice: IntervalLattice, tables: CostTables) -> DPResult:
    """Exact forward DP over one whole hard interval, keeping two best paths.

    Nodes are processed in ascending order and every edge runs forward, so by the
    time a node is finalized all of its incoming candidates have been offered:
    one pass suffices and no queue is needed. Each node keeps its two best
    ``(total, predecessor, predecessor rank)`` candidates, ordered by total then
    by the smaller predecessor -- the local tie-break, which induces the path
    that minimises the last cut, then the one before it, and so on. Two
    candidates with distinct ``(predecessor, rank)`` are distinct paths by
    induction, so the runner-up is genuinely path-distinct rather than merely
    differently priced.
    """
    count = len(lattice.atoms)
    ranked: dict[int, list[tuple[float, int, int]]] = {0: [(0.0, -1, -1)]}
    pending: dict[int, list[tuple[float, int, int]]] = {}
    relaxations = 0

    for node in lattice.nodes:
        if node != 0:
            pool = pending.get(node, [])
            pool.sort()
            ranked[node] = pool[:2]
        entries = ranked.get(node) or []
        if not entries:
            continue
        for edge in lattice.edges_from.get(node, ()):
            relaxations += 1
            step = tables.edges[(edge.start_node, edge.end_node)].total
            interior = edge.end_node != count
            cut = tables.cuts[edge.end_node].total if interior else 0.0
            for rank, (total, _pred, _pred_rank) in enumerate(entries):
                value = quantize(total + step)
                if interior:
                    value = quantize(value + cut)
                pending.setdefault(edge.end_node, []).append((value, node, rank))

    final = ranked.get(count) or []
    if not final:
        raise ValueError(
            f"interval {lattice.interval.index}: no legal path reaches node {count}"
        )

    def rebuild(rank: int) -> tuple[int, ...]:
        node, current = count, rank
        cuts: list[int] = []
        while node != 0:
            _total, pred, pred_rank = ranked[node][current]
            if pred < 0:
                break
            if pred != 0:
                cuts.append(pred)
            node, current = pred, pred_rank
        return tuple(reversed(cuts))

    best = _assemble_path(lattice, tables, rebuild(0))
    runner_up = _assemble_path(lattice, tables, rebuild(1)) if len(final) > 1 else None
    return DPResult(best=best, runner_up=runner_up, relaxations=relaxations)


# ---------------------------------------------------------------- selection


@dataclass(frozen=True)
class Selection:
    """What the optimizer found versus what the policy is willing to ship."""

    raw_optimum: PathResult
    policy_selected: PathResult
    selected_is_v1: bool
    margin: float | None
    low_margin: bool
    v1_path_legal: bool
    v1_illegality: str | None
    v1_cost_under_v2: CostBreakdown | None


def _v1_local_cuts(
    lattice: IntervalLattice, v1: V1Partition
) -> tuple[tuple[int, ...] | None, str | None]:
    """Project v1's unit-space cuts onto this interval's node space.

    A v1 cut that falls strictly inside an atom has no node to land on, which is
    itself a finding rather than an error: v1 committed a boundary this lattice
    cannot express, and the reason is recorded instead of being rounded away.
    """
    interval = lattice.interval
    wanted = sorted(
        cut for cut in v1.cuts if interval.unit_start < cut < interval.unit_end
    )
    by_unit: dict[int, int] = {}
    for node in range(1, len(lattice.atoms)):
        by_unit.setdefault(lattice.unit_bound(node), node)
    cuts: list[int] = []
    for unit in wanted:
        node = by_unit.get(unit)
        if node is None:
            return None, f"unit {unit} is not an atom edge in this interval"
        cuts.append(node)
    return tuple(cuts), None


def _select(
    lattice: IntervalLattice,
    tables: CostTables,
    dp: DPResult,
    v1: V1Partition | None,
) -> Selection:
    raw = dp.best
    margin = None if dp.runner_up is None else quantize(dp.runner_up.total - raw.total)

    v1_path: PathResult | None = None
    illegality: str | None = None
    if v1 is not None:
        cuts, reason = _v1_local_cuts(lattice, v1)
        if cuts is None:
            illegality = reason
        else:
            try:
                v1_path = score_path(lattice, tables, cuts)
            except ValueError as exc:
                illegality = str(exc)

    within = v1_path is not None and v1_path.total <= raw.total + POLICY_MARGIN
    return Selection(
        raw_optimum=raw,
        policy_selected=v1_path if (within and v1_path is not None) else raw,
        selected_is_v1=within,
        margin=margin,
        low_margin=(margin is not None and margin < POLICY_MARGIN) or within,
        v1_path_legal=v1_path is not None,
        v1_illegality=illegality,
        v1_cost_under_v2=None if v1_path is None else v1_path.breakdown,
    )


# ---------------------------------------------------------- materialization


def materialize_cues(
    edges: Sequence[Edge],
    atoms: Sequence[LatticeAtom],
    lang: str,
    *,
    fallback_start: float = 0.0,
) -> tuple[Cue, ...]:
    """Turn a chosen edge chain into cues with the engine's own cue shapes.

    ``word_data`` is *atom* level and carries ``text``, exactly as the v1
    materializer emits it: handing back raw per-character entries instead would
    make every later reader fall through to a character cursor and report a total
    diff for reasons that have nothing to do with boundaries.

    One divergence is unavoidable and deliberate. v1 falls back to the *parent
    cue's* bounds for an untimed chunk, and a v2 partition has no parent cue, so
    the fallback here is the previous cue's end (or ``fallback_start`` at the
    front). The acoustic anchors take no fallback at all in either engine:
    invented display time must never be laundered into the evidence layer.
    """
    cues: list[Cue] = []
    previous_end = float(fallback_start)
    for edge in edges:
        chunk = list(atoms[edge.start_node : edge.end_node])
        speech_start = edge.span_start
        speech_end = edge.span_end
        start = previous_end if speech_start is None else float(speech_start)
        end = start if speech_end is None else float(speech_end)
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
        cues.append(cue)
        previous_end = end
    return tuple(cues)


# ------------------------------------------------------------- v1 reference


@dataclass(frozen=True)
class V1Partition:
    """v1's committed answer: interior cuts in unit-id space, plus its cues.

    ``cues`` is optional because most of what the shadow asks of v1 needs only
    the cut set; the typed fallback is the one consumer that cannot work without
    the cue dicts, and it reports an empty adoption rather than inventing them.
    """

    cuts: tuple[int, ...]
    cues: tuple[Cue, ...] = ()


@dataclass(frozen=True)
class V1Reference:
    """v1's whole-document partition, priced under v2's policy."""

    global_cost: CostBreakdown
    hard_disagreements: tuple[dict[str, Any], ...]
    cut_units: frozenset[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cut_units": sorted(self.cut_units),
            "global_cost": self.global_cost.to_dict(),
            "hard_disagreements": list(self.hard_disagreements),
        }


def score_v1_global(
    v1: V1Partition,
    layer: AtomLayer,
    ctx: CostContext,
    *,
    units: Sequence[SourceUnit] = (),
) -> V1Reference:
    """Price v1's actual partition under v2's policy, legality notwithstanding.

    Cut and edge costs are defined on arbitrary node pairs, so v1's answer is
    scored directly rather than being forced through the lattice first -- which
    matters because the interesting v1 partitions are exactly the ones the
    lattice would refuse. Those refusals are recorded separately as typed
    disagreements instead of being priced as infinities.
    """
    profile = ctx.profile
    lang = profile.language
    atoms = layer.atoms
    count = len(atoms)
    bounds = [layer.unit_bound(node) for node in range(count + 1)]

    nodes: list[int] = []
    for cut in sorted(set(v1.cuts)):
        if not 0 < cut < layer.unit_count:
            continue
        position = bisect.bisect_left(bounds, cut)
        if 0 < position < count:
            nodes.append(position)
    chain = (0, *sorted(set(nodes)), count)

    barriers = {
        barrier.node
        for barrier in build_barriers(layer, profile)
        if barrier.kind == "robust-silence"
    }
    packer = IncrementalPacker(lang, profile.max_line_length, profile.max_lines)
    parts: list[CostBreakdown] = []
    disagreements: list[dict[str, Any]] = []

    for cue_index, (left, right) in enumerate(zip(chain, chain[1:])):
        if left >= right:
            continue
        chunk = atoms[left:right]
        packer.reset()
        measure = None
        for atom in chunk:
            measure = packer.extend(atom.text)
        if measure is None:
            continue
        low = span_min([atom.start for atom in chunk])
        high = span_max([atom.end for atom in chunk])
        edge = Edge(
            start_node=left,
            end_node=right,
            text=_join([atom.text for atom in chunk], lang),
            display_text=measure.text,
            lines=measure.lines,
            line_widths=measure.line_widths,
            span_start=low,
            span_end=high,
            waiver=None,
        )
        unit_range = (layer.unit_bound(left), layer.unit_bound(right))
        parts.append(
            edge_cost(
                edge,
                atoms,
                profile=profile,
                preview=ctx.preview,
                next_start=ctx.next_start_after(right),
                sentence_cross_count=sum(
                    1 for node in ctx.sentence_nodes if left < node < right
                ),
            )
        )
        if left != 0:
            parts.append(
                cut_cost(
                    atoms[left - 1],
                    atoms[left],
                    unit_id=layer.unit_bound(left),
                    profile=profile,
                    speech_spans=ctx.speech_spans,
                    shot_changes=ctx.shot_changes,
                    v1_cut_units=ctx.v1_cut_units,
                )
            )
        if not measure.fits:
            disagreements.append(
                {
                    "cue_index": cue_index,
                    "detail": f"{measure.lines} lines of widths "
                    f"{list(measure.line_widths)} exceed the budget",
                    "kind": "over-budget",
                    "unit_range": list(unit_range),
                }
            )
        crossed = sorted(node for node in barriers if left < node < right)
        if crossed:
            disagreements.append(
                {
                    "cue_index": cue_index,
                    "detail": f"crosses robust-silence barriers at nodes {crossed}",
                    "kind": "crosses-barrier",
                    "unit_range": list(unit_range),
                }
            )
        if (
            profile.max_cue_s > 0
            and low is not None
            and high is not None
            and high - low > profile.max_cue_s + CAP_EPS_S
            and not held_chain_continuous(units, unit_range[0], unit_range[1])
        ):
            disagreements.append(
                {
                    "cue_index": cue_index,
                    "detail": f"span {high - low} exceeds the cap "
                    f"{profile.max_cue_s} with no continuous held chain",
                    "kind": "over-cap-unwaived",
                    "unit_range": list(unit_range),
                }
            )

    return V1Reference(
        global_cost=sum_breakdowns(parts),
        hard_disagreements=tuple(disagreements),
        cut_units=frozenset(v1.cuts),
    )


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

    @property
    def optimized(self) -> bool:
        return self.adopted is None

    @property
    def unit_range(self) -> tuple[int, int]:
        if self.adopted is not None:
            return self.adopted.unit_range
        return (self.interval.unit_start, self.interval.unit_end)

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
            "coalesced_atoms": self.lattice.coalesced_atoms,
            "dp_relaxations": self.dp_relaxations,
            "edge_count": len(self.lattice.edges),
            "fallback_expansion_units": None
            if self.adopted is None or self.adopted.fallback_expansion_units is None
            else list(self.adopted.fallback_expansion_units),
            "infeasible": None if infeasible is None else infeasible.to_dict(),
            "interval_index": self.interval.index,
            "low_margin": None if selection is None else selection.low_margin,
            "margin": None if selection is None else selection.margin,
            "node_range": [self.interval.node_start, self.interval.node_end],
            "packer_steps": self.packer_steps,
            "policy_selected": None
            if selection is None
            else selection.policy_selected.to_dict(),
            "raw_optimum": None
            if selection is None
            else selection.raw_optimum.to_dict(),
            "relief_injections": self.lattice.relief_injections,
            "runner_up_total": None
            if selection is None or selection.margin is None
            else quantize(selection.raw_optimum.total + selection.margin),
            "selected_is_v1": None if selection is None else selection.selected_is_v1,
            "unit_range": list(self.unit_range),
            "v1_cost_under_v2": None
            if selection is None or selection.v1_cost_under_v2 is None
            else selection.v1_cost_under_v2.to_dict(),
            "v1_illegality": None if selection is None else selection.v1_illegality,
            "v1_path_legal": None if selection is None else selection.v1_path_legal,
            "v2_partition": list(self.partition_units),
            "validator_raw": self.validator_raw.to_dict(),
            "waivers": [waiver.to_dict() for waiver in self.waivers],
        }


def _path_edges(lattice: IntervalLattice, cuts: Sequence[int]) -> tuple[Edge, ...]:
    index = {(edge.start_node, edge.end_node): edge for edge in lattice.edges}
    nodes = (0, *cuts, len(lattice.atoms))
    return tuple(index[(left, right)] for left, right in zip(nodes, nodes[1:]))


def optimize_interval(
    lattice: IntervalLattice,
    tables: CostTables,
    ctx: CostContext,
    *,
    units: Sequence[SourceUnit],
    v1: V1Partition | None = None,
    fallback_start: float = 0.0,
) -> IntervalSolution:
    """Solve one interval, or adopt v1's cues for it and say why.

    ``ctx`` is a deviation from the reviewed signature: the validator needs the
    resolved display profile and materialization needs the language, and both
    already ride on the cost context rather than being threaded a second time.
    """
    interval = lattice.interval
    profile = ctx.profile
    lang = profile.language

    if lattice.infeasible is not None:
        adopted = _adopt_v1(interval, lattice.infeasible.reason, v1, len(units))
        low, high = adopted.unit_range
        return IntervalSolution(
            interval=interval,
            lattice=lattice,
            selection=None,
            adopted=adopted,
            cues=adopted.cues,
            partition_units=adopted.cuts,
            validator_raw=check_partition(
                [cut - low for cut in adopted.cuts],
                adopted.cues,
                units=units[low:high],
                profile=profile,
                origin="v1",
                stage="raw",
            ),
            dp_relaxations=0,
            packer_steps=lattice.packer_steps,
        )

    dp = solve_interval(lattice, tables)
    selection = _select(lattice, tables, dp, v1)
    chosen = selection.policy_selected
    edges = _path_edges(lattice, chosen.cuts)
    cues = materialize_cues(edges, lattice.atoms, lang, fallback_start=fallback_start)
    waivers = tuple(
        replace(edge.waiver, cue_index=index)
        for index, edge in enumerate(edges)
        if edge.waiver is not None
    )
    partition_units = tuple(lattice.unit_bound(node) for node in chosen.cuts)
    low, high = interval.unit_start, interval.unit_end
    return IntervalSolution(
        interval=interval,
        lattice=lattice,
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
    )


# ------------------------------------------------------- document solutions


@dataclass(frozen=True)
class DocumentSolution:
    """Every interval's answer for one document, plus the shadow artifact."""

    document: SegDocument
    lattice: DocumentLattice | None
    ctx: CostContext | None
    solutions: tuple[IntervalSolution, ...]
    v1_reference: V1Reference | None
    invalid_profile: tuple[ProfileViolation, ...]
    artifact: dict[str, Any]


def _profile_dict(profile: DisplayProfile) -> dict[str, Any]:
    return {
        "clause_ms": profile.clause_ms,
        "cps": profile.cps,
        "glue_gap_s": profile.glue_gap_s,
        "lag_out_s": profile.lag_out_s,
        "language": profile.language,
        "max_cue_s": profile.max_cue_s,
        "max_line_length": profile.max_line_length,
        "max_lines": profile.max_lines,
        "min_cue_s": profile.min_cue_s,
        "offline_ms": profile.offline_ms,
        "shot_snap_s": profile.shot_snap_s,
        "vad_skip_ms": profile.vad_skip_ms,
    }


def _invalid_artifact(violations: Sequence[ProfileViolation]) -> dict[str, Any]:
    """An invalid measurement, stated as one -- never a degraded measurement.

    The block is deliberately minimal: a reader that finds ``invalid_profile``
    must not be able to accidentally read totals or intervals that describe a run
    which never legitimately happened.
    """
    return {
        "engine_v2": ENGINE_V2,
        "invalid_profile": [violation.to_dict() for violation in violations],
        "kind": "segmentation-shadow",
        "policy_version": POLICY_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def _document_partition(
    solutions: Sequence[IntervalSolution], unit_count: int
) -> tuple[int, ...]:
    """The whole document's interior cuts, from the per-interval answers."""
    cuts: list[int] = []
    for solution in solutions:
        low = solution.unit_range[0]
        if 0 < low:
            cuts.append(low)
        cuts.extend(solution.partition_units)
    return tuple(sorted({cut for cut in cuts if 0 < cut < unit_count}))


def _document_waivers(solutions: Sequence[IntervalSolution]) -> dict[int, Waiver]:
    """Every interval's exemptions, re-stamped into document cue indices.

    A waiver's ``cue_index`` is interval-local, and the document-level pass reads
    the concatenated cue stream, so handing the ledger over unchanged would point
    each exemption at the wrong cue -- and omitting it entirely (the shape this
    replaced) is worse still: the document pass then re-reports an exemption the
    interval pass granted, as an *unwaived* and therefore exit-driving violation.
    An artifact whose ``validator.raw`` and ``totals.hard_violations`` disagree
    about the same cue tells a Wave B reader two different things about whether
    the run failed, and which one it believes is an accident of which field it
    happened to read.
    """
    stamped: dict[int, Waiver] = {}
    offset = 0
    for solution in solutions:
        for waiver in solution.waivers:
            index = offset + waiver.cue_index
            stamped[index] = replace(waiver, cue_index=index)
        offset += len(solution.cues)
    return stamped


def _artifact(
    document: SegDocument,
    lattice: DocumentLattice,
    solutions: Sequence[IntervalSolution],
    v1_reference: V1Reference | None,
    document_check: PartitionCheckResult,
) -> dict[str, Any]:
    unit_count = len(document.units)
    optimized = [solution for solution in solutions if solution.optimized]
    optimized_units = sum(
        solution.interval.unit_end - solution.interval.unit_start
        for solution in optimized
    )
    waivers: list[dict[str, Any]] = []
    for solution in solutions:
        waivers.extend(waiver.to_dict() for waiver in solution.waivers)
    return {
        "barrier_flips": None,
        "engine_v2": ENGINE_V2,
        "influence_cell": {"radius_units": INFLUENCE_RADIUS_UNITS},
        "intervals": [solution.to_dict() for solution in solutions],
        "kind": "segmentation-shadow",
        "language": document.language,
        "lanes": None,
        "pause_knees": {
            state: list(values)
            for state, values in sorted(pause_knees(document.profile).items())
        },
        "perturbation": None,
        "policy_deltas": list(POLICY_DELTAS),
        "policy_name": POLICY_NAME,
        "policy_version": POLICY_VERSION,
        "production_degraded": [],
        "profile": _profile_dict(document.profile),
        "quality": None,
        "schema_version": SCHEMA_VERSION,
        "shadow_degraded": [],
        "totals": {
            "all_invisible_intervals": sum(
                1 for solution in solutions if solution.lattice.all_invisible
            ),
            "atom_count": len(lattice.layer.atoms),
            "barrier_count": len(lattice.barriers),
            "coalesced_atoms": sum(
                solution.lattice.coalesced_atoms for solution in solutions
            ),
            "dp_relaxations": sum(solution.dp_relaxations for solution in solutions),
            "fallback_intervals": len(solutions) - len(optimized),
            "hard_violations": sum(
                len(solution.validator_raw.exit_driving) for solution in solutions
            ),
            "interval_count": len(solutions),
            "optimized_intervals": len(optimized),
            "optimized_unit_ratio": 1.0
            if unit_count == 0
            else optimized_units / unit_count,
            "packer_steps": sum(solution.packer_steps for solution in solutions),
            "relief_injections": sum(
                solution.lattice.relief_injections for solution in solutions
            ),
            "sentence_ends_missed": lattice.sentence_ends.missed,
            "unit_count": unit_count,
            "waivers": waivers,
        },
        "v1": None if v1_reference is None else v1_reference.to_dict(),
        "validator": {
            "core": None,
            "legacy_overlay": None,
            "raw": document_check.to_dict(),
        },
    }


def optimize_document(
    document: SegDocument,
    *,
    v1: V1Partition | None = None,
    preview: DisplayTimingPreview | None = None,
) -> DocumentSolution:
    """Solve every hard interval of one document and assemble its artifact.

    The profile preflight runs first and is fatal for the document: a knob with
    no defined meaning makes the measurement invalid, and reporting an invalid
    measurement as a degraded one is how a shadow lane starts lying.
    """
    invalid = preflight_profile(document.profile)
    if invalid:
        return DocumentSolution(
            document=document,
            lattice=None,
            ctx=None,
            solutions=(),
            v1_reference=None,
            invalid_profile=invalid,
            artifact=_invalid_artifact(invalid),
        )

    lattice = build_document_lattice(document)
    ctx = build_cost_context(document, lattice, preview=preview, v1=v1)

    solutions: list[IntervalSolution] = []
    fallback_start = 0.0
    for interval_lattice in lattice.lattices:
        tables = build_cost_tables(interval_lattice, ctx)
        solution = optimize_interval(
            interval_lattice,
            tables,
            ctx,
            units=document.units,
            v1=v1,
            fallback_start=fallback_start,
        )
        solutions.append(solution)
        if solution.cues:
            fallback_start = float(solution.cues[-1]["end"])

    v1_reference = (
        None
        if v1 is None
        else score_v1_global(v1, lattice.layer, ctx, units=document.units)
    )
    cues = tuple(cue for solution in solutions for cue in solution.cues)
    document_check = check_partition(
        _document_partition(solutions, len(document.units)),
        cues,
        units=document.units,
        profile=document.profile,
        origin="v2" if all(solution.optimized for solution in solutions) else "v1",
        stage="raw",
        waivers=_document_waivers(solutions),
    )
    return DocumentSolution(
        document=document,
        lattice=lattice,
        ctx=ctx,
        solutions=tuple(solutions),
        v1_reference=v1_reference,
        invalid_profile=(),
        artifact=_artifact(document, lattice, solutions, v1_reference, document_check),
    )


def shadow_artifact(
    document: SegDocument,
    *,
    v1: V1Partition | None = None,
    preview: DisplayTimingPreview | None = None,
) -> dict[str, Any]:
    """The one call the Wave B hook makes."""
    return optimize_document(document, v1=v1, preview=preview).artifact
