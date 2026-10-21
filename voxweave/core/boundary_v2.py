"""The exact whole-interval solver, its selection policy, and the artifact.

Nothing about the search is approximate. With the tile machinery deleted, each
hard interval is solved by one forward shortest-path pass over its whole legal
edge set, so margins and the selection policy are *whole-interval* quantities
rather than per-window ones that could disagree at a seam. That is
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
  pairs; spaced packer extensions and no-space canonical
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
    sum_breakdowns,
)
from .boundary_lattice import (
    CAP_EPS_S,
    COARSE_GRANULARITY,
    INFLUENCE_RADIUS_UNITS,
    AtomLayer,
    DocumentLattice,
    Edge,
    IncrementalPacker,
    IntervalLattice,
    LatticeAtom,
    PackMeasure,
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
from .boundary_select import (
    POLICY_MARGIN as POLICY_MARGIN,
    CostTables as CostTables,
    DPResult as DPResult,
    PathResult as PathResult,
    Selection as Selection,
    V1Partition as V1Partition,
    _assemble_path as _assemble_path,
    _margin_summary as _margin_summary,
    _nearest_rank as _nearest_rank,
    _pinned_neighbour_margins as _pinned_neighbour_margins,
    _select as _select,
    _v1_local_cuts as _v1_local_cuts,
    score_path as score_path,
    solve_interval as solve_interval,
)
from .boundary_interval import (
    ADOPTION_ABSORBED as ADOPTION_ABSORBED,
    AdoptedV1 as AdoptedV1,
    IntervalSolution as IntervalSolution,
    _adopt_v1 as _adopt_v1,
    _adoption_plan as _adoption_plan,
    _path_edges as _path_edges,
    _resolve_edge_for_previous as _resolve_edge_for_previous,
    _resolve_selected_path as _resolve_selected_path,
    materialize_cues as materialize_cues,
    optimize_interval as optimize_interval,
)
from .canonical_text import CanonicalWork
from .layout import _join, _no_spaces
from .partition_check import (
    Origin,
    PartitionCheckResult,
    Waiver,
    check_partition,
)
from .policy_delta import delta_registry_data
from .segdoc import DisplayProfile, SegDocument, SourceUnit
from .subunit import (
    RefineResult,
    empty_refine_result,
    require_issued_refinement,
    speech_span_units,
)
from .speaker_evidence import (
    W_SPEAKER_INTERIOR,
    EvidenceSpan,
    SpeakerPricingSummary,
    UnitSpeakers,
    evidence_span_from_cue,
    make_evidence_span,
    named_multi_cues_unannotated,
    speaker_edge_cost,
    speaker_evidence,
    summarize_speaker_prices,
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

    The preview defaults to experimental_policy_1's frozen preview (extensions
    clamp at the next cue's start); the mirror of today's cleanup pass is
    ``LegacyCleanupPreview(two_frame_extension_floor=True)``. P5 hands in the
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


def _sentence_cross_count(ctx: CostContext, left: int, right: int) -> int:
    """How many document sentence-end nodes lie strictly inside ``(left, right)``."""
    nodes = ctx.sorted_sentence_nodes
    return max(0, bisect.bisect_left(nodes, right) - bisect.bisect_right(nodes, left))


def _phase1_unit_facts(
    ctx: CostContext, low: int, high: int
) -> tuple[float | None, float | None, str]:
    """The acoustic anchors and owned footprint phase 1 reads for ``[low, high)``."""
    speech_start, speech_end = speech_span_units(ctx.units[low:high])
    footprint = _join(
        [unit.surface for unit in ctx.units[low:high]], ctx.profile.language
    )
    return speech_start, speech_end, footprint


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
        "sentence_cross_count": _sentence_cross_count(ctx, left, right),
    }
    # The staged policy-1 solve remains the exact legacy experiment. Explicit
    # P5 rows always carry UnitSpeakers (including an absent track), and consume
    # the finalizer preview's input/acoustic split below.
    if not isinstance(ctx.speaker_evidence, UnitSpeakers):
        return edge_cost(edge, lattice.atoms, **common)

    speech_start, speech_end, owned_footprint = _phase1_unit_facts(
        ctx, lattice.unit_bound(edge.start_node), lattice.unit_bound(edge.end_node)
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
    speaker_parts: list[CostBreakdown] = []
    resolved_document_nodes = (
        _document_nodes(lattice, ctx.layer)
        if document_nodes is None
        else tuple(document_nodes)
    )
    if len(resolved_document_nodes) != len(lattice.atoms) + 1:
        raise ValueError("document-node map does not match interval topology")
    for edge in lattice.edges:
        base = _base_edge_cost(lattice, edge, ctx, resolved_document_nodes)
        key = (edge.start_node, edge.end_node)
        if isinstance(ctx.speaker_evidence, UnitSpeakers):
            unit_range = (
                lattice.unit_bound(edge.start_node),
                lattice.unit_bound(edge.end_node),
            )
            evidence_span = edge.evidence_span
            if not isinstance(evidence_span, EvidenceSpan):
                if edge.evidence_unavailable_reason is not None:
                    edges[key] = base
                    continue
                raise ValueError(
                    "speaker pricing requires cached candidate EvidenceSpan values"
                )
            speaker = speaker_edge_cost(
                ctx.speaker_evidence,
                unit_range,
                evidence_span=evidence_span,
                sing_spans=ctx.sing_spans,
                weight=ctx.speaker_weight,
                suppressed_lyric=edge.lyric,
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
        speaker_context=(
            ctx if isinstance(ctx.speaker_evidence, UnitSpeakers) else None
        ),
        fallback_start=float(fallback_start),
        document_nodes=resolved_document_nodes,
    )


# ------------------------------------------------------------- v1 reference


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


# Lands every v1 cut on an atom node. Returns the node chain, the cuts that did
# not land on an atom edge, and the v1 cue index for each landed span.
def _v1_landed_chain(
    v1: V1Partition,
    layer: AtomLayer,
    bounds: Sequence[int],
    count: int,
) -> tuple[tuple[int, ...], list[dict[str, Any]], dict[tuple[int, int], int]]:
    nodes: list[int] = []
    rounded: list[dict[str, Any]] = []
    # The node every v1 cue boundary landed on, in v1 cue order. Rounding can
    # collapse several v1 cuts onto one node (or onto a document end), so a
    # chain cue is matched to the v1 cue whose landed span it is -- never to the
    # v1 cue that merely shares its position in the shortened chain.
    landed: list[int] = [0]
    for cut in sorted(set(v1.cuts)):
        if not 0 < cut < layer.unit_count:
            landed.append(0 if cut <= 0 else count)
            continue
        position = bisect.bisect_left(bounds, cut)
        if bounds[position] != cut:
            # Includes a cut inside the last atom, which rounds up onto the
            # document end and so vanishes from the chain: still a finding.
            rounded.append(
                {"landed_unit": bounds[position], "node": position, "unit": cut}
            )
        landed.append(position)
        if 0 < position < count:
            nodes.append(position)
    landed.append(count)
    chain = (0, *sorted(set(nodes)), count)
    v1_cue_for_span = {
        span: index
        for index, span in enumerate(zip(landed, landed[1:]))
        if span[0] < span[1]
    }
    return chain, rounded, v1_cue_for_span


# Measures one chain cue and builds its edge; None when the prefix packer never
# produced a measure (an empty span).
def _v1_chain_edge(
    atoms: Sequence[LatticeAtom],
    left: int,
    right: int,
    packer: IncrementalPacker | None,
    profile: DisplayProfile,
    canonical_work: CanonicalWork,
    lang: str,
) -> tuple[Edge, PackMeasure, float | None, float | None] | None:
    chunk = atoms[left:right]
    if packer is None:
        measure = _canonical_pack_measure(atoms, left, right, profile, canonical_work)
    else:
        packer.reset()
        measure = None
        for atom in chunk:
            measure = packer.extend(atom.text)
        if measure is None:
            return None
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
    return edge, measure, low, high


# The acoustic-or-input evidence span a v1 chain cue is priced with.
def _v1_evidence_span(
    ctx: CostContext,
    v1: V1Partition,
    v1_cue_for_span: Mapping[tuple[int, int], int],
    left: int,
    right: int,
    low: float | None,
    high: float | None,
    unit_range: tuple[int, int],
) -> EvidenceSpan:
    v1_index = v1_cue_for_span.get((left, right))
    if v1_index is not None and v1_index < len(v1.cues):
        evidence_span = evidence_span_from_cue(v1.cues[v1_index])
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
    return evidence_span


# The typed hard disagreements for one chain cue, in the fixed order
# over-budget, crosses-barrier, over-cap-unwaived.
def _v1_hard_disagreements(
    cue_index: int,
    measure: PackMeasure,
    barriers: set[int],
    left: int,
    right: int,
    low: float | None,
    high: float | None,
    unit_range: tuple[int, int],
    profile: DisplayProfile,
    units: Sequence[SourceUnit],
) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
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
    return disagreements


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

    With a speaker track (the P5 rows) v1 is judged and priced under the rules
    the speaker lattice and its tables use: canonical ``FinalText`` admission in
    every language, and the phase-1 input bounds, acoustic anchors and owned
    footprint in the edge price. Scoring v1 under the policy-1 packer and span
    defaults instead would compare two partitions under two different policies.
    """
    profile = ctx.profile
    lang = profile.language
    atoms = layer.atoms
    count = len(atoms)
    bounds = [layer.unit_bound(node) for node in range(count + 1)]
    speaker_rows = isinstance(ctx.speaker_evidence, UnitSpeakers)

    chain, rounded, v1_cue_for_span = _v1_landed_chain(v1, layer, bounds, count)

    barriers = {
        barrier.node
        for barrier in build_barriers(layer, profile)
        if barrier.kind == "robust-silence"
    }
    packer = (
        None
        if _no_spaces(lang) or speaker_rows
        else IncrementalPacker(lang, profile.max_line_length, profile.max_lines)
    )
    canonical_work = CanonicalWork()
    parts: list[CostBreakdown] = []
    disagreements: list[dict[str, Any]] = []
    previous_end = 0.0

    for cue_index, (left, right) in enumerate(zip(chain, chain[1:])):
        if left >= right:
            continue
        chain_edge = _v1_chain_edge(
            atoms, left, right, packer, profile, canonical_work, lang
        )
        if chain_edge is None:
            continue
        edge, measure, low, high = chain_edge
        unit_range = (layer.unit_bound(left), layer.unit_bound(right))
        common = {
            "profile": profile,
            "preview": ctx.preview,
            "next_start": ctx.next_start_after(right),
            "sentence_cross_count": _sentence_cross_count(ctx, left, right),
        }
        if speaker_rows:
            input_start, input_end = _resolve_edge_input_bounds(
                edge, previous_end=previous_end
            )
            previous_end = input_end
            speech_start, speech_end, footprint = _phase1_unit_facts(
                ctx, unit_range[0], unit_range[1]
            )
            base = edge_cost(
                edge,
                atoms,
                **common,
                input_start=input_start,
                input_end=input_end,
                speech_start=speech_start,
                speech_end=speech_end,
                expected_footprint=footprint,
            )
        else:
            base = edge_cost(edge, atoms, **common)
        if isinstance(ctx.speaker_evidence, UnitSpeakers):
            evidence_span = _v1_evidence_span(
                ctx, v1, v1_cue_for_span, left, right, low, high, unit_range
            )
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
        disagreements.extend(
            _v1_hard_disagreements(
                cue_index,
                measure,
                barriers,
                left,
                right,
                low,
                high,
                unit_range,
                profile,
                units,
            )
        )

    return V1Reference(
        global_cost=sum_breakdowns(parts),
        hard_disagreements=tuple(disagreements),
        cut_units=frozenset(v1.cuts),
        rounded_cuts=tuple(rounded),
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


def _whole_partition_origin(
    solutions: Sequence[IntervalSolution], unit_count: int
) -> Origin:
    """Who a document-level tiling failure belongs to.

    The whole-partition row names no cue, so it is attributed by what broke the
    tiling. v2 owns the plan: interval ranges that must tile the stream in order,
    the cues of every optimized interval, and a cue for every unit a fallback was
    handed. Only adopted v1 cues that do not tile their own range are v1's, and
    that interval's own check already types them so. Everything else, including
    a stream that tiles and so emits no row, is v2's.
    """
    v1_defect = False
    cursor = 0
    for solution in solutions:
        low, high = solution.unit_range
        if low != cursor:
            return "v2"
        cursor = high
        expected = 0 if low == high else len(solution.partition_units) + 1
        if len(solution.cues) == expected:
            continue
        if solution.adopted is not None and solution.adopted.cues:
            v1_defect = True
        else:
            return "v2"
    if cursor != unit_count:
        return "v2"
    return "v1" if v1_defect else "v2"


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


# Count the materialized cues that carry more than one named speaker; the
# ownership check raises at the same point the inline block did.
def _named_multi_cues(
    solutions: Sequence[IntervalSolution],
    speakers: UnitSpeakers | None,
    unit_count: int,
) -> int:
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
    return named_multi


# The ``totals`` block of the artifact; the caller keeps its key position.
def _artifact_totals(
    lattice: DocumentLattice,
    solutions: Sequence[IntervalSolution],
    optimized: Sequence[IntervalSolution],
    optimized_units: int,
    unit_count: int,
    interval_hard: int,
    coarse_caused_intervals: int,
    waivers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
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
    }


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
    named_multi = _named_multi_cues(solutions, speakers, unit_count)
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
        "totals": _artifact_totals(
            lattice,
            solutions,
            optimized,
            optimized_units,
            unit_count,
            interval_hard,
            coarse_caused_intervals,
            waivers,
        ),
        "v1": None if v1_reference is None else v1_reference.to_dict(),
        "validator": {
            "core": None,
            # The two counters are cross-checked here, and
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
    adoptions = _adoption_plan(lattice.lattices, v1, len(document.units))
    for position, raw_interval_lattice in enumerate(lattice.lattices):
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
            adopted=adoptions.get(position),
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
        # Per-cue rows are attributed to the interval that produced them; the
        # whole-partition row to whatever broke the tiling.
        origin=_whole_partition_origin(solutions, len(document.units)),
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
    """Artifact of :func:`optimize_document` — a test/calibration convenience.

    The shadow hook (``shadow_v2``) calls :func:`optimize_document` directly.
    """
    return optimize_document(
        document,
        v1=v1,
        preview=preview,
        subunit_split=subunit_split,
        speakers=speakers,
        speaker_weight=speaker_weight,
    ).artifact
