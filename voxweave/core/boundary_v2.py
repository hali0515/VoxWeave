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
* **work counters, separately.** DP relaxations count ``(node, outgoing edge)``
  pairs, not 2-best ranks; spaced packer extensions and no-space canonical
  character visits are counted independently by the lattice. Each is asserted
  against its resolved bound by tests rather than production asserts, because a
  work bound that fires in production is a crash, not a proof.

The selection policy is deliberately conservative: v1's partition wins whenever
it is a legal path here and within :data:`POLICY_MARGIN` of the optimum. The
shadow exists to find where v2 is *convincingly* better, and a margin is what
separates that from a rounding error.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from .boundary_cost import (
    POLICY_NAME,
    POLICY_VERSION,
    VAD_STATES,
    CostBreakdown,
    CostContext,
    cut_cost,
    edge_cost,
    make_breakdown,
    pause_knees,
    quantize,
    sum_breakdowns,
)
from .boundary_lattice import (
    CAP_EPS_S,
    COARSE_GRANULARITY,
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
    _cache_candidate_evidence,
    _canonical_pack_measure,
    _resolve_edge_input_bounds,
    held_chain_continuous,
    preflight_profile,
    span_max,
    span_min,
)
from .canonical_text import CanonicalWork
from .layout import _join, _no_spaces
from .partition_check import (
    Origin,
    PartitionCheckResult,
    Waiver,
    check_partition,
    owned_unit_ids,
)
from .policy_delta import delta_registry_data
from .schema import Cue, Unit
from .segdoc import DisplayProfile, SegDocument, SourceUnit
from .subunit import (
    RefineResult,
    empty_refine_result,
    require_issued_refinement,
    speech_span_units,
)
from .speaker_evidence import (
    EVIDENCE_SPAN_REFUSAL_REVERSED,
    W_SPEAKER_INTERIOR,
    EvidenceSpan,
    SpeakerPricingSummary,
    UnitSpeakers,
    evidence_span_from_cue,
    lyric_for_evidence,
    make_evidence_span,
    named_multi_cues_unannotated,
    speaker_edge_cost,
    speaker_evidence,
    summarize_speaker_prices,
    try_make_evidence_span,
)
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
    "selected_evidence_spans",
    "shadow_artifact",
    "solve_interval",
]

#: The engine name every shadow artifact is stamped with.
ENGINE_V2: str = "boundary-optimizer-v2"

#: Standalone optimizer artifact schema.  This module cannot materialize W4's
#: finalizer/authority/lane blocks, so it must never claim the live schema 2
#: contract.  The live shadow assembler validates the completed payload and
#: stamps version 2 at that later admission boundary.
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

LEGACY_POLICY_NAME: str = "experimental_policy_1"
LEGACY_POLICY_VERSION: int = 1
SPEAKER_POLICY_DELTAS: tuple[str, ...] = (*POLICY_DELTAS, "PD-SPK")


# ---------------------------------------------------------------- cost tables


@dataclass(frozen=True)
class CostTables:
    """Every edge and cut price of one interval, computed exactly once."""

    edges: Mapping[tuple[int, int], CostBreakdown]
    cuts: Mapping[int, CostBreakdown]
    speaker_pricing: SpeakerPricingSummary | None = None
    base_edges: Mapping[tuple[int, int], CostBreakdown] | None = None
    speaker_context: CostContext | None = None
    fallback_start: float = 0.0
    predecessor_stateful: bool = False
    speaker_pricing_refused: bool = False
    document_nodes: tuple[int, ...] = ()


@dataclass
class _OptimizationReuse:
    """One document's immutable topology plus reusable unit-node maps.

    The two speaker pricing rows differ only in their weight.  They therefore
    share this raw lattice and the interval-to-document node maps; selected-edge
    evidence remains row-local and is installed on a replaced lattice below.
    """

    document: SegDocument
    lattice: DocumentLattice
    document_nodes: dict[int, tuple[int, ...]]


def _optimization_reuse(
    document: SegDocument, *, canonical_spaced: bool = False
) -> _OptimizationReuse:
    """Open a reuse scope for solves over the exact same document object."""
    return _OptimizationReuse(
        document=document,
        lattice=build_document_lattice(
            document,
            cache_speaker_evidence=False,
            canonical_spaced=canonical_spaced,
        ),
        document_nodes={},
    )


def build_cost_context(
    document: SegDocument,
    lattice: DocumentLattice,
    *,
    preview: DisplayTimingPreview | None = None,
    v1: V1Partition | None = None,
    speakers: UnitSpeakers | None = None,
    speaker_weight: float | None = None,
) -> CostContext:
    """Bundle everything a cost term needs that is not the edge or the cut.

    The preview defaults to the mirror of today's cleanup pass; P5 hands in the
    finalizer's own preview and nothing else here changes.
    """
    resolved_speakers = speakers
    if resolved_speakers is not None and tuple(
        resolved_speakers.refined_units
    ) != tuple(document.units):
        raise ValueError(
            "speaker evidence does not describe this document's unit stream"
        )
    resolved_weight = (
        0.0
        if resolved_speakers is None
        else W_SPEAKER_INTERIOR
        if speaker_weight is None
        else speaker_weight
    )
    if (
        isinstance(resolved_weight, bool)
        or not isinstance(resolved_weight, (int, float))
        or not math.isfinite(resolved_weight)
        or resolved_weight < 0
    ):
        raise ValueError("speaker weight must be finite and non-negative")
    return CostContext(
        profile=document.profile,
        preview=LegacyCleanupPreview() if preview is None else preview,
        speech_spans=document.vad_speech,
        shot_changes=document.shot_changes,
        sentence_nodes=lattice.sentence_ends.nodes,
        v1_cut_units=frozenset() if v1 is None else frozenset(v1.cuts),
        layer=lattice.layer,
        units=tuple(document.units),
        unit_speakers=()
        if resolved_speakers is None
        else resolved_speakers.unit_speakers,
        speaker_evidence=resolved_speakers,
        sing_spans=document.sing_spans,
        speaker_weight=resolved_weight,
    )


def _with_speaker_cost(base: CostBreakdown, speaker: CostBreakdown) -> CostBreakdown:
    """Add the disjoint W3 feature/term block without lossy aggregation."""
    duplicate_features = set(base.features) & set(speaker.features)
    duplicate_terms = set(base.weighted_terms) & set(speaker.weighted_terms)
    if duplicate_features or duplicate_terms:
        raise ValueError(
            "speaker cost collided with existing cost keys: "
            f"features={sorted(duplicate_features)}, terms={sorted(duplicate_terms)}"
        )
    return make_breakdown(
        {**base.features, **speaker.features},
        {**base.weighted_terms, **speaker.weighted_terms},
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


def _base_edge_cost(
    lattice: IntervalLattice,
    edge: Edge,
    ctx: CostContext,
    document_nodes: Sequence[int],
) -> CostBreakdown:
    """Price one edge from the same phase-1 facts materialization will use."""
    left = document_nodes[edge.start_node]
    right = document_nodes[edge.end_node]
    common = {
        "profile": ctx.profile,
        "preview": ctx.preview,
        "next_start": ctx.next_start_after(right),
        "sentence_cross_count": sum(
            1 for node in ctx.sentence_nodes if left < node < right
        ),
    }
    # The staged policy-1 solve remains the exact legacy experiment. Explicit
    # P5 rows always carry UnitSpeakers (including an absent track), and consume
    # the finalizer preview's input/acoustic split below.
    if not isinstance(ctx.speaker_evidence, UnitSpeakers):
        return edge_cost(edge, lattice.atoms, **common)

    low = lattice.unit_bound(edge.start_node)
    high = lattice.unit_bound(edge.end_node)
    speech_start, speech_end = speech_span_units(ctx.units[low:high])
    owned_footprint = _join(
        [unit.surface for unit in ctx.units[low:high]], ctx.profile.language
    )
    return edge_cost(
        edge,
        lattice.atoms,
        **common,
        input_start=edge.input_start,
        input_end=edge.input_end,
        speech_start=speech_start,
        speech_end=speech_end,
        expected_footprint=owned_footprint,
    )


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

    unit_range = (
        lattice.unit_bound(edge.start_node),
        lattice.unit_bound(edge.end_node),
    )
    evidence_span = try_make_evidence_span(
        ctx.units,
        unit_range,
        input_start=input_start,
        input_end=input_end,
    )
    if evidence_span is None:
        return replace(
            edge,
            evidence_span=None,
            lyric=False,
            input_start=input_start,
            input_end=input_end,
            evidence_deferred=False,
            evidence_unavailable_reason=EVIDENCE_SPAN_REFUSAL_REVERSED,
        )
    return replace(
        edge,
        evidence_span=evidence_span,
        lyric=lyric_for_evidence(evidence_span, ctx.sing_spans),
        input_start=input_start,
        input_end=input_end,
        evidence_deferred=False,
        evidence_unavailable_reason=None,
    )


def _speaker_price(
    lattice: IntervalLattice,
    edge: Edge,
    ctx: CostContext,
    base: CostBreakdown,
    document_nodes: Sequence[int],
    *,
    previous_end: float,
) -> tuple[Edge, CostBreakdown]:
    """Resolve one predecessor-dependent edge and compose its speaker term."""
    if not isinstance(ctx.speaker_evidence, UnitSpeakers):
        return edge, base
    resolved = _resolve_edge_for_previous(
        lattice,
        edge,
        ctx,
        previous_end=previous_end,
    )
    if resolved.evidence_unavailable_reason is not None:
        return resolved, _base_edge_cost(lattice, resolved, ctx, document_nodes)
    if not isinstance(resolved.evidence_span, EvidenceSpan):
        raise ValueError("resolved speaker edge has no EvidenceSpan")
    unit_range = (
        lattice.unit_bound(edge.start_node),
        lattice.unit_bound(edge.end_node),
    )
    speaker = speaker_edge_cost(
        ctx.speaker_evidence,
        unit_range,
        evidence_span=resolved.evidence_span,
        sing_spans=ctx.sing_spans,
        weight=ctx.speaker_weight,
        suppressed_lyric=resolved.lyric,
    )
    # A fabricated start can inherit the selected predecessor's end. That value
    # is a phase-1 input, so both the speaker term and the base preview price must
    # be recomputed for the resolved edge rather than retaining a representative
    # table entry produced with ``fallback_start``.
    resolved_base = _base_edge_cost(lattice, resolved, ctx, document_nodes)
    return resolved, _with_speaker_cost(resolved_base, speaker)


def build_cost_tables(
    lattice: IntervalLattice,
    ctx: CostContext,
    *,
    fallback_start: float = 0.0,
    document_nodes: Sequence[int] | None = None,
) -> CostTables:
    """Price every legal edge and every candidate cut of one interval.

    The DP, an independent path scorer and the v1 reference all read from here,
    so the solver cannot disagree with a path someone else scored -- which is
    what makes the brute-force equality test meaningful rather than circular.
    """
    profile = ctx.profile
    atoms = lattice.atoms
    edges: dict[tuple[int, int], CostBreakdown] = {}
    base_edges: dict[tuple[int, int], CostBreakdown] = {}
    speaker_parts: list[CostBreakdown] = []
    predecessor_stateful = False
    speaker_pricing_refused = False
    resolved_document_nodes = (
        _document_nodes(lattice, ctx.layer)
        if document_nodes is None
        else tuple(document_nodes)
    )
    if len(resolved_document_nodes) != len(lattice.atoms) + 1:
        raise ValueError("document-node map does not match interval topology")
    for edge in lattice.edges:
        priced_edge = edge
        if isinstance(ctx.speaker_evidence, UnitSpeakers) and edge.evidence_deferred:
            priced_edge = _resolve_edge_for_previous(
                lattice, edge, ctx, previous_end=fallback_start
            )
        base = _base_edge_cost(lattice, priced_edge, ctx, resolved_document_nodes)
        key = (edge.start_node, edge.end_node)
        base_edges[key] = base
        if isinstance(ctx.speaker_evidence, UnitSpeakers):
            unit_range = (
                lattice.unit_bound(edge.start_node),
                lattice.unit_bound(edge.end_node),
            )
            evidence_span = edge.evidence_span
            if not isinstance(evidence_span, EvidenceSpan):
                if edge.evidence_unavailable_reason is not None:
                    speaker_pricing_refused = True
                    edges[key] = base
                    continue
                if not edge.evidence_deferred:
                    raise ValueError(
                        "speaker pricing requires cached candidate EvidenceSpan values"
                    )
                predecessor_stateful = True
                representative = _resolve_edge_for_previous(
                    lattice,
                    edge,
                    ctx,
                    previous_end=fallback_start,
                )
                evidence_span = representative.evidence_span
                if representative.evidence_unavailable_reason is not None:
                    speaker_pricing_refused = True
                    edges[key] = base
                    continue
                if not isinstance(evidence_span, EvidenceSpan):
                    raise ValueError("deferred candidate did not resolve EvidenceSpan")
                lyric = representative.lyric
            else:
                lyric = edge.lyric
            speaker = speaker_edge_cost(
                ctx.speaker_evidence,
                unit_range,
                evidence_span=evidence_span,
                sing_spans=ctx.sing_spans,
                weight=ctx.speaker_weight,
                suppressed_lyric=lyric,
            )
            speaker_parts.append(speaker)
            edges[key] = _with_speaker_cost(base, speaker)
        else:
            edges[key] = base

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
    return CostTables(
        edges=edges,
        cuts=cuts,
        speaker_pricing=(
            summarize_speaker_prices(speaker_parts)
            if isinstance(ctx.speaker_evidence, UnitSpeakers)
            else None
        ),
        base_edges=base_edges,
        speaker_context=(
            ctx if isinstance(ctx.speaker_evidence, UnitSpeakers) else None
        ),
        fallback_start=float(fallback_start),
        predecessor_stateful=predecessor_stateful,
        speaker_pricing_refused=speaker_pricing_refused,
        document_nodes=resolved_document_nodes,
    )


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
    edge_index = {(edge.start_node, edge.end_node): edge for edge in lattice.edges}
    total = 0.0
    previous_end = tables.fallback_start
    edge_parts: list[CostBreakdown] = []
    cut_parts: list[CostBreakdown] = []
    for left, right in zip(nodes, nodes[1:]):
        key = (left, right)
        edge = tables.edges.get(key)
        candidate = edge_index.get(key)
        if edge is None or candidate is None:
            raise ValueError(f"edge({left}, {right}): no legal cue spans these atoms")
        if tables.speaker_context is not None and tables.predecessor_stateful:
            if tables.base_edges is None:
                raise ValueError("speaker cost table has no base-edge authority")
            candidate, edge = _speaker_price(
                lattice,
                candidate,
                tables.speaker_context,
                tables.base_edges[key],
                tables.document_nodes,
                previous_end=previous_end,
            )
            if candidate.input_end is None:
                raise ValueError("resolved speaker edge has no input end")
            previous_end = float(candidate.input_end)
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
    if tables.predecessor_stateful:
        return _solve_interval_with_predecessor_state(lattice, tables)

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


def _state_path_key(item: tuple[float, tuple[int, ...]]) -> tuple[Any, ...]:
    """Whole-path order matching the ordinary DP's recursive local tie-break."""
    total, cuts = item
    return (total, *reversed(cuts))


def _top_two_state_paths(
    candidates: Sequence[tuple[float, tuple[int, ...]]],
) -> list[tuple[float, tuple[int, ...]]]:
    by_path: dict[tuple[int, ...], float] = {}
    for total, cuts in candidates:
        prior = by_path.get(cuts)
        if prior is None or total < prior:
            by_path[cuts] = total
    ranked = [(total, cuts) for cuts, total in by_path.items()]
    ranked.sort(key=_state_path_key)
    return ranked[:2]


def _solve_interval_with_predecessor_state(
    lattice: IntervalLattice,
    tables: CostTables,
) -> DPResult:
    """Exact two-best DP when an edge start inherits its selected predecessor.

    The state is the preceding resolved edge end.  Keeping two paths per
    ``(node, end)`` is sufficient: every continuation from that state has the
    same future costs, so a third path can never become the global runner-up.
    """
    ctx = tables.speaker_context
    bases = tables.base_edges
    if ctx is None or bases is None:
        raise ValueError("predecessor-state solve requires speaker cost authority")

    count = len(lattice.atoms)
    ranked: dict[int, dict[float, list[tuple[float, tuple[int, ...]]]]] = {
        0: {tables.fallback_start: [(0.0, ())]}
    }
    pending: dict[int, dict[float, list[tuple[float, tuple[int, ...]]]]] = {}
    relaxations = 0

    for node in lattice.nodes:
        if node != 0:
            states = pending.get(node, {})
            ranked[node] = {
                previous_end: _top_two_state_paths(candidates)
                for previous_end, candidates in states.items()
            }
        states = ranked.get(node, {})
        if not states:
            continue
        for previous_end, entries in states.items():
            for edge in lattice.edges_from.get(node, ()):
                relaxations += 1
                key = (edge.start_node, edge.end_node)
                resolved, step = _speaker_price(
                    lattice,
                    edge,
                    ctx,
                    bases[key],
                    tables.document_nodes,
                    previous_end=previous_end,
                )
                if resolved.input_end is None:
                    raise ValueError("resolved speaker edge has no input end")
                next_end = float(resolved.input_end)
                interior = edge.end_node != count
                cut = tables.cuts[edge.end_node].total if interior else 0.0
                bucket = pending.setdefault(edge.end_node, {}).setdefault(next_end, [])
                for total, cuts in entries:
                    value = quantize(total + step.total)
                    next_cuts = cuts
                    if interior:
                        value = quantize(value + cut)
                        next_cuts = (*cuts, edge.end_node)
                    bucket.append((value, next_cuts))

    final_candidates = [
        candidate
        for candidates in ranked.get(count, {}).values()
        for candidate in candidates
    ]
    final = _top_two_state_paths(final_candidates)
    if not final:
        raise ValueError(
            f"interval {lattice.interval.index}: no legal path reaches node {count}"
        )
    best = _assemble_path(lattice, tables, final[0][1])
    runner_up = _assemble_path(lattice, tables, final[1][1]) if len(final) > 1 else None
    return DPResult(best=best, runner_up=runner_up, relaxations=relaxations)


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
    """Best single-cut relocation delta with adjacent selected cuts pinned."""
    if not selected.cuts:
        return ()
    edge_keys = {(edge.start_node, edge.end_node) for edge in lattice.edges}
    chain = (0, *selected.cuts, len(lattice.atoms))
    margins: list[float] = []
    for index, cut in enumerate(selected.cuts):
        previous_node, next_node = chain[index], chain[index + 2]
        alternatives: list[float] = []
        for candidate in lattice.nodes:
            if (
                candidate == cut
                or not previous_node < candidate < next_node
                or (previous_node, candidate) not in edge_keys
                or (candidate, next_node) not in edge_keys
            ):
                continue
            relocated = list(selected.cuts)
            relocated[index] = candidate
            try:
                result = score_path(lattice, tables, relocated)
            except ValueError:
                continue
            alternatives.append(quantize(result.total - selected.total))
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

    One divergence is unavoidable and deliberate. v1 falls back to the *parent
    cue's* bounds for an untimed chunk, and a v2 partition has no parent cue, so
    the fallback here is the previous cue's end (or ``fallback_start`` at the
    front). The acoustic anchors take no fallback at all in either engine:
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
    """v1's whole-document partition, priced under v2's policy.

    ``rounded_cuts`` is kept apart from ``hard_disagreements``: a cut that did
    not land on an atom edge is not v1 breaking a hard rule, it is *this
    reference* pricing a partition slightly different from the one v1 committed.
    Silently rounding it and saying nothing would make the C9 number look like a
    measurement of v1's answer when it is a measurement of the nearest
    expressible one.
    """

    global_cost: CostBreakdown
    hard_disagreements: tuple[dict[str, Any], ...]
    cut_units: frozenset[int]
    rounded_cuts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cut_units": sorted(self.cut_units),
            "global_cost": self.global_cost.to_dict(),
            "hard_disagreements": list(self.hard_disagreements),
            "rounded_cuts": list(self.rounded_cuts),
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
    rounded: list[dict[str, Any]] = []
    for cut in sorted(set(v1.cuts)):
        if not 0 < cut < layer.unit_count:
            continue
        position = bisect.bisect_left(bounds, cut)
        if 0 < position < count:
            if bounds[position] != cut:
                rounded.append(
                    {"landed_unit": bounds[position], "node": position, "unit": cut}
                )
            nodes.append(position)
    chain = (0, *sorted(set(nodes)), count)

    barriers = {
        barrier.node
        for barrier in build_barriers(layer, profile)
        if barrier.kind == "robust-silence"
    }
    packer = (
        None
        if _no_spaces(lang)
        else IncrementalPacker(lang, profile.max_line_length, profile.max_lines)
    )
    canonical_work = CanonicalWork()
    parts: list[CostBreakdown] = []
    disagreements: list[dict[str, Any]] = []

    for cue_index, (left, right) in enumerate(zip(chain, chain[1:])):
        if left >= right:
            continue
        chunk = atoms[left:right]
        if packer is None:
            measure = _canonical_pack_measure(
                atoms, left, right, profile, canonical_work
            )
        else:
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
        base = edge_cost(
            edge,
            atoms,
            profile=profile,
            preview=ctx.preview,
            next_start=ctx.next_start_after(right),
            sentence_cross_count=sum(
                1 for node in ctx.sentence_nodes if left < node < right
            ),
        )
        if isinstance(ctx.speaker_evidence, UnitSpeakers):
            if cue_index < len(v1.cues):
                evidence_span = evidence_span_from_cue(v1.cues[cue_index])
            elif (
                low is not None
                and high is not None
                and isinstance(low, (int, float))
                and isinstance(high, (int, float))
                and math.isfinite(low)
                and math.isfinite(high)
                and low <= high
            ):
                evidence_span = make_evidence_span(
                    ctx.units,
                    unit_range,
                    input_start=float(low),
                    input_end=float(high),
                )
            else:
                evidence_span = EvidenceSpan(0.0, 0.0, "fabricated", "fabricated")
            speaker = speaker_edge_cost(
                ctx.speaker_evidence,
                unit_range,
                evidence_span=evidence_span,
                sing_spans=ctx.sing_spans,
                weight=ctx.speaker_weight,
            )
            parts.append(_with_speaker_cost(base, speaker))
        else:
            parts.append(base)
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
        rounded_cuts=tuple(rounded),
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
    edges_from: dict[int, tuple[Edge, ...]] = {}
    for node in lattice.nodes:
        outgoing = tuple(edge for edge in edges if edge.start_node == node)
        if outgoing:
            edges_from[node] = outgoing
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
    subunit_split: RefineResult
    artifact: dict[str, Any]
    speaker_evidence: UnitSpeakers | None = None


def selected_evidence_spans(solution: DocumentSolution) -> tuple[EvidenceSpan, ...]:
    """Return the selected row's cached H/lyric evidence basis in cue order.

    Optimized intervals carry the exact objects priced by the solver. A typed
    v1 adoption has no candidate edge, so its retained cue is converted through
    the same v1 EvidenceSpan constructor frozen by section 6.4.
    """
    spans: list[EvidenceSpan] = []
    for interval in solution.solutions:
        if interval.selection is not None:
            for edge in _path_edges(
                interval.lattice, interval.selection.policy_selected.cuts
            ):
                if not isinstance(edge.evidence_span, EvidenceSpan):
                    raise ValueError(
                        "selected row does not carry cached EvidenceSpan authority"
                    )
                spans.append(edge.evidence_span)
        elif interval.adopted is not None:
            spans.extend(evidence_span_from_cue(cue) for cue in interval.adopted.cues)
    return tuple(spans)


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


def _invalid_artifact(
    violations: Sequence[ProfileViolation], *, speaker_enabled: bool
) -> dict[str, Any]:
    """An invalid measurement, stated as one -- never a degraded measurement.

    The block is deliberately minimal: a reader that finds ``invalid_profile``
    must not be able to accidentally read totals or intervals that describe a run
    which never legitimately happened.
    """
    return {
        "engine_v2": ENGINE_V2,
        "invalid_profile": [violation.to_dict() for violation in violations],
        "kind": "segmentation-shadow",
        "policy_version": POLICY_VERSION if speaker_enabled else LEGACY_POLICY_VERSION,
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
            # ``replace`` on the cue index alone, deliberately: every other field
            # is the provenance AD-4 requires (unit ids, span, cap, reason), and
            # a re-stamp that dropped it would leave a ledger whose exemptions
            # cannot be re-derived -- exactly the "taken on trust" shape
            # ``Waiver``'s own docstring exists to refuse.
            stamped[index] = replace(waiver, cue_index=index)
        offset += len(solution.cues)
    return stamped


def _document_origins(solutions: Sequence[IntervalSolution]) -> dict[int, str]:
    """Which engine produced each cue of the concatenated document stream.

    AD3-3 types attribution per violation, and a document that fell back on one
    interval still ran v2 everywhere else. Typing the whole stream from
    ``all(optimized)`` -- the shape this replaced -- reported a genuine v2
    ``speech-truncated-start`` in a fully optimized interval as v1 damage, i.e.
    as non-exit-driving, purely because some *other* interval had adopted v1.
    """
    origins: dict[int, str] = {}
    offset = 0
    for solution in solutions:
        engine = "v2" if solution.optimized else "v1"
        for index in range(len(solution.cues)):
            origins[offset + index] = engine
        offset += len(solution.cues)
    return origins


def _vad_state_totals(solutions: Sequence[IntervalSolution]) -> dict[str, Any]:
    """Spec v3's ``vad_state`` artifact block: the evidence behind every cut made.

    ``sum_breakdowns`` drops categorical features, so the interval aggregates say
    nothing at all about which pause regime priced a boundary. This counts the
    selected cuts by state, which is the one aggregate that *is* meaningful for a
    categorical, and it is computed from the same breakdowns the DP read rather
    than from a second pass over the document.
    """
    counts = {state: 0 for state in VAD_STATES}
    unknown = 0
    for solution in solutions:
        if solution.selection is None:
            continue
        for part in solution.selection.policy_selected.cut_breakdowns:
            state = part.features.get("vad_state")
            if isinstance(state, str) and state in counts:
                counts[state] += 1
            else:
                unknown += 1
    return {"selected_cuts_by_state": counts, "unclassified_cuts": unknown}


def _speaker_pricing_totals(
    solutions: Sequence[IntervalSolution],
) -> SpeakerPricingSummary:
    summaries = [
        solution.speaker_pricing
        for solution in solutions
        if solution.speaker_pricing is not None
    ]
    return SpeakerPricingSummary(
        priced_edges=sum(item.priced_edges for item in summaries),
        speaker_changes_in_cue_raw=sum(
            item.speaker_changes_in_cue_raw for item in summaries
        ),
        suppressed_lyric_edges=sum(item.suppressed_lyric_edges for item in summaries),
        two_speaker_edges=sum(item.two_speaker_edges for item in summaries),
        turn_states={
            state: sum(item.turn_states.get(state, 0) for item in summaries)
            for state in ("absent", "overlap", "multi", "single", "unattributed")
        },
    )


def _resolve_speaker_evidence(
    document: SegDocument,
    split: RefineResult,
    supplied: UnitSpeakers | None,
) -> UnitSpeakers:
    """Return evidence whose optimizer-space projection is exactly ``split``.

    A non-identity refinement has destroyed the production-parent records that
    attribution must read.  If a speaker track exists, only the caller that
    still owns those parents can construct the projection, so silently
    classifying the proportional child times is forbidden.  When the track is
    absent there is no acoustic fact to reconstruct; a ghost parent stream is
    sufficient to produce the explicit all-``none`` projection while retaining
    the complete origin tuple for audit.
    """
    identity = split.refined_parent_count == 0 and split.origin == tuple(
        range(len(document.units))
    )
    if supplied is not None:
        if tuple(supplied.refined_units) != tuple(document.units):
            raise ValueError(
                "parent-projected speaker evidence does not describe this unit stream"
            )
        if supplied.origin != split.origin:
            raise ValueError(
                "parent-projected speaker evidence does not use this origin tuple"
            )
        if (
            supplied.parent_units != split.parent_units
            or supplied.language != split.parent_language
        ):
            raise ValueError(
                "parent-projected speaker evidence disagrees with the production "
                "parent payload"
            )
        if not supplied.matches_document_track(document):
            raise ValueError(
                "parent-projected speaker evidence disagrees with the document track"
            )
        return supplied
    if identity:
        return speaker_evidence(document)
    if document.speaker_turns is not None:
        raise ValueError(
            "a refined speaker track requires explicit parent-projected speaker evidence"
        )

    parent_document = SegDocument(
        language=split.parent_language,
        units=list(split.parent_units),
        profile=document.profile,
        vad_speech=None,
        shot_changes=None,
        sing_spans=None,
        speaker_turns=None,
        manifest={},
        text=None,
    )
    return speaker_evidence(
        parent_document,
        refined_units=document.units,
        origin=split.origin,
    )


def _artifact(
    document: SegDocument,
    lattice: DocumentLattice,
    solutions: Sequence[IntervalSolution],
    v1_reference: V1Reference | None,
    document_check: PartitionCheckResult,
    subunit_split: RefineResult,
    speakers: UnitSpeakers | None,
    speaker_weight: float,
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
    interval_hard = sum(
        len(solution.validator_raw.exit_driving) for solution in solutions
    )
    coarse_caused_intervals = sum(solution.coarse_caused for solution in solutions)
    speaker_pricing = (
        _speaker_pricing_totals(solutions) if speakers is not None else None
    )
    named_multi = 0
    if speakers is not None:
        materialized = tuple(solution for solution in solutions if solution.cues)
        cues = tuple(cue for solution in materialized for cue in solution.cues)
        cue_ranges = tuple(
            pair
            for solution in materialized
            for pair in zip(
                (solution.unit_range[0], *solution.partition_units),
                (*solution.partition_units, solution.unit_range[1]),
            )
        )
        if cues and len(cue_ranges) != len(cues):
            raise ValueError(
                "selected cue ownership does not match materialized cue count"
            )
        complete_ownership = (
            bool(cue_ranges)
            and cue_ranges[0][0] == 0
            and cue_ranges[-1][1] == unit_count
            and all(
                left[1] == right[0] for left, right in zip(cue_ranges, cue_ranges[1:])
            )
        )
        if cues and complete_ownership:
            named_multi = named_multi_cues_unannotated(
                cue_ranges, speakers.unit_speakers
            )
    artifact: dict[str, Any] = {
        "coverage": {
            "coarse_caused_intervals": coarse_caused_intervals,
            "dual_form_unmeasured": speakers is not None
            and document.profile.max_lines >= 2
            and speaker_pricing is not None
            and speaker_pricing.two_speaker_edges > 0,
            "named_multi_cues_unannotated": named_multi,
        },
        "engine_v2": ENGINE_V2,
        # W1 supplied the finalizer and W3 supplies speaker evidence; W4 wires
        # their row-specific blocks.  ``None`` means that stage did not run for
        # this standalone optimizer artifact, not a reserved metric value.
        "finalizer": None,
        "influence_cell": {"radius_units": INFLUENCE_RADIUS_UNITS},
        "intervals": [solution.to_dict() for solution in solutions],
        "kind": "segmentation-shadow",
        "language": document.language,
        "lanes": None,
        "margin_summary": _margin_summary(
            [margin for solution in solutions for margin in solution.decision_margins],
            selected_cut_count=sum(
                0
                if solution.selection is None
                else len(solution.selection.policy_selected.cuts)
                for solution in solutions
            ),
        ),
        "pause_knees": {
            state: list(values)
            for state, values in sorted(pause_knees(document.profile).items())
        },
        "policy_deltas": list(
            SPEAKER_POLICY_DELTAS if speakers is not None else POLICY_DELTAS
        ),
        "policy_name": POLICY_NAME if speakers is not None else LEGACY_POLICY_NAME,
        "policy_version": (
            POLICY_VERSION if speakers is not None else LEGACY_POLICY_VERSION
        ),
        "production_degraded": [],
        "profile": _profile_dict(document.profile),
        # AD3-5: filled by the hook's caller once the outer capture has closed.
        # Reserved here so the artifact's key order does not depend on when the
        # value arrives.
        "providers": {},
        "schema_version": SCHEMA_VERSION,
        "shadow_degraded": [],
        "speaker_evidence": None
        if speakers is None
        else speakers.to_dict(
            pricing=speaker_pricing,
            speaker_weight=speaker_weight,
        ),
        "subunit_split": subunit_split.to_dict(),
        "vad_state": _vad_state_totals(solutions),
        "totals": {
            "all_invisible_intervals": sum(
                1 for solution in solutions if solution.lattice.all_invisible
            ),
            "atom_count": len(lattice.layer.atoms),
            "barrier_count": len(lattice.barriers),
            "cap_relief_nodes": sum(
                solution.lattice.cap_relief_nodes for solution in solutions
            ),
            "coalesced_atoms": sum(
                solution.lattice.coalesced_atoms for solution in solutions
            ),
            "canonical_chars": sum(
                solution.lattice.canonical_chars for solution in solutions
            ),
            "coarse_granularity_intervals": sum(
                1
                for solution in solutions
                if solution.lattice.infeasible is not None
                and solution.lattice.infeasible.reason == COARSE_GRANULARITY
            ),
            "coarse_caused_intervals": coarse_caused_intervals,
            "dp_relaxations": sum(solution.dp_relaxations for solution in solutions),
            "fallback_intervals": len(solutions) - len(optimized),
            "hard_violations": interval_hard,
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
            # The two counters the module docstring says are cross-checked, and
            # the answer stated rather than left for a reader to recompute: the
            # document pass sees cross-interval predicates the per-interval
            # passes cannot, so it may report MORE, but a document pass reporting
            # fewer than the intervals did means an exemption or an attribution
            # drifted between the two.
            "interval_hard_violations": interval_hard,
            "interval_document_agree": len(document_check.exit_driving)
            >= interval_hard,
            "legacy_overlay": None,
            "raw": document_check.to_dict(),
        },
    }
    if speakers is not None:
        artifact["delta_registry"] = delta_registry_data()
    return artifact


def optimize_document(
    document: SegDocument,
    *,
    v1: V1Partition | None = None,
    preview: DisplayTimingPreview | None = None,
    subunit_split: RefineResult | None = None,
    speakers: UnitSpeakers | None = None,
    speaker_weight: float | None = None,
    _reuse: _OptimizationReuse | None = None,
) -> DocumentSolution:
    """Solve every hard interval of one document and assemble its artifact.

    ``subunit_split`` is the audited result that produced ``document``; omitting
    it declares the identity mapping.  The profile preflight is fatal for the
    document: a knob with no defined meaning makes the measurement invalid, and
    reporting an invalid measurement as a degraded one is how a shadow lane
    starts lying.  W3 remains staged behind an explicit ``speaker_weight`` (or
    supplied parent projection): ``None`` preserves the pre-W4 live call, while
    W4 can request the full and counterfactual rows with ``3.0`` and ``0.0``.
    """
    if subunit_split is None:
        if any(unit.provenance.startswith("subunit-") for unit in document.units):
            raise ValueError(
                "a derived sub-unit stream requires its audited subunit_split result"
            )
        split = empty_refine_result(document.units, language=document.language)
    else:
        split = subunit_split
    require_issued_refinement(split)
    if tuple(document.units) != split.units or len(split.origin) != len(document.units):
        raise ValueError("subunit_split does not describe this document's unit stream")

    speaker_enabled = speakers is not None or speaker_weight is not None
    invalid = preflight_profile(document.profile)
    if invalid:
        return DocumentSolution(
            document=document,
            lattice=None,
            ctx=None,
            solutions=(),
            v1_reference=None,
            invalid_profile=invalid,
            subunit_split=split,
            artifact=_invalid_artifact(invalid, speaker_enabled=speaker_enabled),
        )

    resolved_weight = W_SPEAKER_INTERIOR if speaker_weight is None else speaker_weight
    resolved_speakers = (
        _resolve_speaker_evidence(document, split, speakers)
        if speaker_enabled
        else None
    )
    # The first fabricated bound of each interval depends on the preceding
    # selected interval's delivered input end.  Build admission first, then
    # resolve each interval only when that predecessor is known.
    if _reuse is not None and _reuse.document is not document:
        raise ValueError("optimization reuse belongs to a different document")
    reuse = (
        _optimization_reuse(document, canonical_spaced=speaker_enabled)
        if _reuse is None
        else _reuse
    )
    lattice = reuse.lattice
    ctx = build_cost_context(
        document,
        lattice,
        preview=preview,
        v1=v1,
        speakers=resolved_speakers,
        speaker_weight=resolved_weight,
    )

    solutions: list[IntervalSolution] = []
    resolved_lattices: list[IntervalLattice] = []
    fallback_start = 0.0
    for raw_interval_lattice in lattice.lattices:
        interval_lattice = (
            _cache_candidate_evidence(
                raw_interval_lattice,
                document,
                fallback_start=fallback_start,
            )
            if speaker_enabled
            else raw_interval_lattice
        )
        cached_nodes = reuse.document_nodes.get(raw_interval_lattice.interval.index)
        tables = build_cost_tables(
            interval_lattice,
            ctx,
            fallback_start=fallback_start,
            document_nodes=cached_nodes,
        )
        if cached_nodes is None:
            reuse.document_nodes[raw_interval_lattice.interval.index] = (
                tables.document_nodes
            )
        solution = optimize_interval(
            interval_lattice,
            tables,
            ctx,
            units=document.units,
            v1=v1,
            fallback_start=fallback_start,
        )
        solutions.append(solution)
        resolved_lattices.append(solution.lattice)
        if solution.cues:
            fallback_start = float(solution.cues[-1]["end"])

    lattice = replace(lattice, lattices=tuple(resolved_lattices))

    v1_reference = (
        None
        if v1 is None
        else score_v1_global(v1, lattice.layer, ctx, units=document.units)
    )
    cues = tuple(cue for solution in solutions for cue in solution.cues)
    origins = _document_origins(solutions)
    document_check = check_partition(
        _document_partition(solutions, len(document.units)),
        cues,
        units=document.units,
        profile=document.profile,
        # The whole-partition row belongs to no cue, so it keeps the document's
        # own character; per-cue rows are attributed to the interval that
        # produced them.
        origin="v2" if all(solution.optimized for solution in solutions) else "v1",
        stage="raw",
        waivers=_document_waivers(solutions),
        origins=cast("Mapping[int, Origin]", origins),
    )
    return DocumentSolution(
        document=document,
        lattice=lattice,
        ctx=ctx,
        solutions=tuple(solutions),
        v1_reference=v1_reference,
        invalid_profile=(),
        subunit_split=split,
        artifact=_artifact(
            document,
            lattice,
            solutions,
            v1_reference,
            document_check,
            split,
            resolved_speakers,
            resolved_weight,
        ),
        speaker_evidence=resolved_speakers,
    )


def shadow_artifact(
    document: SegDocument,
    *,
    v1: V1Partition | None = None,
    preview: DisplayTimingPreview | None = None,
    subunit_split: RefineResult | None = None,
    speakers: UnitSpeakers | None = None,
    speaker_weight: float | None = None,
) -> dict[str, Any]:
    """The one call the Wave B hook makes."""
    return optimize_document(
        document,
        v1=v1,
        preview=preview,
        subunit_split=subunit_split,
        speakers=speakers,
        speaker_weight=speaker_weight,
    ).artifact
