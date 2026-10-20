"""Path choice over one interval's cost table.

The value objects of a scored path, the exact forward DP, the order-fixed path
scoring shared with it, and the margin policy that decides whether v1's
partition or the raw optimum ships. Nothing here knows about documents: every
function takes one interval's lattice and its cost table.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .boundary_cost import CostBreakdown, CostContext, quantize, sum_breakdowns
from .boundary_lattice import IntervalLattice
from .schema import Cue
from .speaker_evidence import SpeakerPricingSummary


#: How much worse than the raw optimum a legal v1 path may be and still be
#: selected. A margin rather than an equality test because the point of the
#: shadow lane is to migrate only where v2 is *convincingly* better, not
#: wherever it is better by a rounding error.
POLICY_MARGIN: float = 1.0


@dataclass(frozen=True)
class CostTables:
    """Every edge and cut price of one interval, computed exactly once."""

    edges: Mapping[tuple[int, int], CostBreakdown]
    cuts: Mapping[int, CostBreakdown]
    speaker_pricing: SpeakerPricingSummary | None = None
    speaker_context: CostContext | None = None
    fallback_start: float = 0.0
    document_nodes: tuple[int, ...] = ()


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

        ``breakdown`` is a ``sum_breakdowns`` aggregate, and aggregation is lossy
        in exactly the place AD-3 cares about: a categorical feature has no sum,
        so ``vad_state`` -- the field that says what kind of evidence a boundary
        rested on -- vanishes from it entirely, and ``gap_ms_raw``/
        ``effective_ms``/``overlap_fraction``/``ramp_ms`` survive only as
        interval totals nobody can re-weight a single decision from. AD-3
        requires the features "per cut candidate", so the raw halves are emitted
        per cut and per edge alongside the aggregate. Weighted terms are NOT
        repeated here: those are the policy's opinion and the aggregate already
        carries them under a stated ``policy_version``.
        """
        return {
            "breakdown": self.breakdown.to_dict(),
            "cut_features": [
                {"unit": unit, **{k: part.features[k] for k in sorted(part.features)}}
                for unit, part in zip(self.unit_cuts, self.cut_breakdowns)
            ],
            "cuts": list(self.unit_cuts),
            "edge_features": [
                {k: part.features[k] for k in sorted(part.features)}
                for part in self.edge_breakdowns
            ],
            "node_cuts": list(self.cuts),
            "total": self.total,
        }


def _assemble_path(
    lattice: IntervalLattice, tables: CostTables, cuts: Sequence[int]
) -> PathResult:
    count = len(lattice.atoms)
    nodes = (0, *cuts, count)
    edge_index = {(edge.start_node, edge.end_node) for edge in lattice.edges}
    total = 0.0
    edge_parts: list[CostBreakdown] = []
    cut_parts: list[CostBreakdown] = []
    for left, right in zip(nodes, nodes[1:]):
        key = (left, right)
        edge = tables.edges.get(key)
        if edge is None or key not in edge_index:
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
    """The optimum and the work spent finding it."""

    best: PathResult
    relaxations: int


def solve_interval(lattice: IntervalLattice, tables: CostTables) -> DPResult:
    """Exact forward DP over one whole hard interval.

    Nodes are processed in ascending order and every edge runs forward, so by the
    time a node is finalized all of its incoming candidates have been offered:
    one pass suffices and no queue is needed. Each node keeps its best
    ``(total, predecessor)``, ordered by total then by the smaller predecessor --
    the local tie-break, which induces the path that minimises the last cut, then
    the one before it, and so on.
    """
    count = len(lattice.atoms)
    best: dict[int, tuple[float, int]] = {0: (0.0, -1)}
    pending: dict[int, list[tuple[float, int]]] = {}
    relaxations = 0

    for node in lattice.nodes:
        if node != 0:
            pool = pending.get(node)
            if not pool:
                continue
            best[node] = min(pool)
        entry = best.get(node)
        if entry is None:
            continue
        total = entry[0]
        for edge in lattice.edges_from.get(node, ()):
            relaxations += 1
            step = tables.edges[(edge.start_node, edge.end_node)].total
            interior = edge.end_node != count
            value = quantize(total + step)
            if interior:
                value = quantize(value + tables.cuts[edge.end_node].total)
            pending.setdefault(edge.end_node, []).append((value, node))

    if count not in best:
        raise ValueError(
            f"interval {lattice.interval.index}: no legal path reaches node {count}"
        )
    cuts: list[int] = []
    node = count
    while node != 0:
        pred = best[node][1]
        if pred < 0:
            break
        if pred != 0:
            cuts.append(pred)
        node = pred
    return DPResult(
        best=_assemble_path(lattice, tables, tuple(reversed(cuts))),
        relaxations=relaxations,
    )


# ---------------------------------------------------------------- selection


@dataclass(frozen=True)
class Selection:
    """What the optimizer found versus what the policy is willing to ship.

    ``v1_supplied`` separates "v1 chose nothing here" from "there is no v1
    reference at all". Without it every v1 field reads as a measurement -- legal:
    no, cost: none, selected: no -- when in truth nothing was measured, and a
    reader cannot tell the two apart from the artifact.
    """

    raw_optimum: PathResult
    policy_selected: PathResult
    selected_is_v1: bool
    v1_path_legal: bool
    v1_illegality: str | None
    v1_cost_under_v2: CostBreakdown | None
    v1_supplied: bool = True


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
        v1_path_legal=v1_path is not None,
        v1_illegality=illegality,
        v1_cost_under_v2=None if v1_path is None else v1_path.breakdown,
        v1_supplied=v1 is not None,
    )


def _pinned_neighbour_margins(
    lattice: IntervalLattice,
    tables: CostTables,
    selected: PathResult,
) -> tuple[float, ...]:
    """Best single-cut relocation delta with adjacent selected cuts pinned.

    A relocated path shares the selected path's accumulation up to the cut
    before the moved one, so the fold restarts from that prefix total and only
    replays the changed edges and the unchanged tail. Totals are therefore the
    same floats :func:`score_path` would give, without rescoring the whole path
    (or rebuilding its edge index) once per candidate node.
    """
    if not selected.cuts:
        return ()
    count = len(lattice.atoms)
    chain = (0, *selected.cuts, count)
    # score_path's accumulation order: an edge, then the cut that opens the
    # next cue. Edge ``j`` sits at step ``2j`` and the cut after it at ``2j+1``.
    steps: list[float] = []
    for left, right in zip(chain, chain[1:]):
        steps.append(tables.edges[(left, right)].total)
        if right != count:
            steps.append(tables.cuts[right].total)
    prefix = [0.0]
    for step in steps:
        prefix.append(quantize(prefix[-1] + step))
    nodes = sorted(lattice.nodes)
    margins: list[float] = []
    for index, cut in enumerate(selected.cuts):
        previous_node, next_node = chain[index], chain[index + 2]
        head = prefix[2 * index]
        tail = steps[2 * index + 3 :]
        alternatives: list[float] = []
        low = bisect.bisect_right(nodes, previous_node)
        high = bisect.bisect_left(nodes, next_node)
        for candidate in nodes[low:high]:
            if candidate == cut:
                continue
            left = tables.edges.get((previous_node, candidate))
            right = tables.edges.get((candidate, next_node))
            opened = tables.cuts.get(candidate)
            if left is None or right is None or opened is None:
                continue
            total = head
            for step in (left.total, opened.total, right.total, *tail):
                total = quantize(total + step)
            alternatives.append(quantize(total - selected.total))
        if alternatives:
            margins.append(min(alternatives))
    return tuple(margins)


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _margin_summary(
    margins: Sequence[float], *, selected_cut_count: int
) -> dict[str, Any] | None:
    """Artifact-only pinned-neighbour relocation summary (nearest-rank)."""
    if selected_cut_count == 0:
        return None
    values = tuple(margins)
    return {
        "count": len(values),
        "exact_ties": sum(value == 0.0 for value in values),
        "min": min(values) if values else None,
        "p05": _nearest_rank(values, 0.05),
        "p50": _nearest_rank(values, 0.50),
    }


@dataclass(frozen=True)
class V1Partition:
    """v1's committed answer: interior cuts in unit-id space, plus its cues.

    ``cues`` is optional because most of what the shadow asks of v1 needs only
    the cut set; the typed fallback is the one consumer that cannot work without
    the cue dicts, and it reports an empty adoption rather than inventing them.
    """

    cuts: tuple[int, ...]
    cues: tuple[Cue, ...] = ()
