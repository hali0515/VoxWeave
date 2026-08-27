"""W3 integration pins that stop short of the W4 live-lane boundary."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from voxweave.core.authority import AuthorityLedger, SealBroken
from voxweave.core.boundary_v2 import (
    build_cost_context,
    build_cost_tables,
    optimize_document,
)
from voxweave.core.boundary_lattice import build_document_lattice
from voxweave.core.finalizer import (
    phase1_from_optimizer_selection,
    register_optimizer_selection,
)
from voxweave.core.segdoc import DisplayProfile, SegDocument, SourceUnit
from voxweave.core.speaker_evidence import (
    SPEAKER_EDGE_RUN_MIN_S,
    SPEAKER_MIN_RUN_S,
    SPEAKER_MULTI_MIN_FRAC,
    SPEAKER_UNIT_COVER_FRAC,
    W_SPEAKER_INTERIOR,
    speaker_evidence,
)
from voxweave.core.subunit import refine_document


def profile(*, max_line_length: int = 42, max_lines: int = 2) -> DisplayProfile:
    return DisplayProfile(
        language="en",
        max_line_length=max_line_length,
        max_lines=max_lines,
        clause_ms=400.0,
        vad_skip_ms=250.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=7.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=11 / 24,
    )


def source(
    index: int,
    text: str,
    start: float | None,
    end: float | None,
    *,
    provenance: str = "aligner",
) -> SourceUnit:
    return SourceUnit(
        id=f"u{index}",
        surface=text,
        start=start,
        end=end,
        provenance=provenance,
    )


def activation_document(*, singing: bool = False, max_lines: int = 2) -> SegDocument:
    # The 250 ms gap is below the 300 ms robust-silence barrier but past the
    # pause ramp.  Hand-derived path totals (other features are zero):
    #   one edge, speaker on  = cue_base 2 + speaker 3 = 5
    #   split                  = 2 + 2 + pause 0.027273 = 4.027273
    #   one edge, speaker off = cue_base 2
    # Hence on cuts at unit 1 and off keeps the change inside one cue.
    units = [source(0, "alpha", 0.0, 1.0), source(1, "bravo", 1.25, 2.25)]
    return SegDocument(
        language="en",
        units=units,
        profile=profile(max_lines=max_lines),
        vad_speech=[],
        shot_changes=None,
        sing_spans=[(0.0, 1.125)] if singing else None,
        speaker_turns=[(0.0, 1.0, "A"), (1.25, 2.25, "B")],
        manifest={},
        text="alpha bravo",
    )


def test_policy_v2_speaker_term_activates_selection_in_expression_direction():
    doc = activation_document()
    pristine = copy.deepcopy(doc)
    on = optimize_document(doc, speaker_weight=W_SPEAKER_INTERIOR)
    off = optimize_document(doc, speaker_weight=0.0)

    assert on.artifact["policy_version"] == 2
    assert on.artifact["policy_name"] == "experimental_policy_2"
    assert on.solutions[0].partition_units == (1,)
    assert off.solutions[0].partition_units == ()
    on_selection = on.solutions[0].selection
    off_selection = off.solutions[0].selection
    assert on_selection is not None and off_selection is not None
    assert on_selection.raw_optimum.total == pytest.approx(4.027273, abs=1e-12)
    assert off_selection.raw_optimum.total == 2.0

    assert on.lattice is not None and on.ctx is not None
    interval = on.lattice.lattices[0]
    tables = build_cost_tables(interval, on.ctx)
    assert tables.edges[(0, 2)].total == 5.0
    assert (
        tables.edges[(0, 1)].total + tables.cuts[1].total + tables.edges[(1, 2)].total
    ) == pytest.approx(4.027273, abs=1e-12)
    assert doc == pristine


def test_cost_context_carries_projected_units_and_singing_evidence():
    doc = activation_document(singing=True)
    lattice = build_document_lattice(doc, cache_speaker_evidence=True)
    speakers = speaker_evidence(doc)
    ctx = build_cost_context(doc, lattice, speakers=speakers)
    assert ctx.unit_speakers == speakers.unit_speakers
    assert ctx.speaker_evidence is speakers
    assert ctx.sing_spans == doc.sing_spans
    assert ctx.speaker_weight == W_SPEAKER_INTERIOR


def test_candidate_evidence_span_is_cached_and_lyric_suppression_matches_flag():
    doc = activation_document(singing=True)
    speakers = speaker_evidence(doc)
    lattice = build_document_lattice(doc, cache_speaker_evidence=True)
    ctx = build_cost_context(doc, lattice, speakers=speakers)
    interval = lattice.lattices[0]
    tables = build_cost_tables(interval, ctx)

    for edge in interval.edges:
        part = tables.edges[(edge.start_node, edge.end_node)]
        assert edge.evidence_span is not None
        assert bool(part.features["suppressed_lyric"]) is edge.lyric

    # The selected full-span edge is lyric-suppressed, so the on row becomes
    # bit-identical to the speaker-off choice and materialization carries True.
    result = optimize_document(doc, speaker_weight=W_SPEAKER_INTERIOR)
    assert result.solutions[0].partition_units == ()
    assert result.solutions[0].cues[0]["lyric"] is True
    selected = result.solutions[0].selection
    assert selected is not None
    assert (
        selected.policy_selected.edge_breakdowns[0].features["suppressed_lyric"] is True
    )


def test_partially_untimed_candidate_caches_independent_endpoint_kinds():
    doc = SegDocument(
        language="en",
        units=[
            source(0, "alpha", None, 0.5),
            source(1, "bravo", 0.5, 1.0),
        ],
        profile=profile(),
        vad_speech=None,
        shot_changes=None,
        sing_spans=[(0.5, 0.75)],
        speaker_turns=[],
        manifest={},
        text="alpha bravo",
    )
    lattice = build_document_lattice(doc, cache_speaker_evidence=True)
    full = next(
        edge
        for edge in lattice.lattices[0].edges
        if edge.start_node == 0 and edge.end_node == len(lattice.lattices[0].atoms)
    )
    assert full.evidence_span.start == 0.5
    assert full.evidence_span.end == 1.0
    assert (full.evidence_span.start_kind, full.evidence_span.end_kind) == (
        "fabricated",
        "exact",
    )
    assert full.lyric is True


def test_pricing_consumes_the_cached_lyric_classification_once(monkeypatch):
    doc = activation_document(singing=True)
    lattice = build_document_lattice(doc, cache_speaker_evidence=True)
    ctx = build_cost_context(doc, lattice, speakers=speaker_evidence(doc))

    def recomputation_is_a_bug(*_args, **_kwargs):
        raise AssertionError("cached candidate lyric was recomputed")

    monkeypatch.setattr(
        "voxweave.core.speaker_evidence.lyric_for_evidence",
        recomputation_is_a_bug,
    )
    tables = build_cost_tables(lattice.lattices[0], ctx)
    assert any(
        part.features["suppressed_lyric"] is True for part in tables.edges.values()
    )


def test_speaker_pricing_refuses_a_lattice_without_candidate_evidence_cache():
    doc = activation_document(singing=True)
    lattice = build_document_lattice(doc)
    ctx = build_cost_context(doc, lattice, speakers=speaker_evidence(doc))
    with pytest.raises(ValueError, match="cached candidate EvidenceSpan"):
        build_cost_tables(lattice.lattices[0], ctx)


def test_cached_lyric_survives_sealed_authority_rematerialization():
    result = optimize_document(
        activation_document(singing=True), speaker_weight=W_SPEAKER_INTERIOR
    )
    ledger = AuthorityLedger()
    authority = register_optimizer_selection(result, ledger=ledger)
    stream = phase1_from_optimizer_selection(
        authority,
        ledger=ledger,
        row_id="delivery_finalizer/v2",
        evaluation_id="speaker-lyric",
    )
    assert len(stream.cues) == 1
    assert stream.cues[0].lyric is True


def test_cached_lyric_is_inside_the_optimizer_selection_seal():
    result = optimize_document(
        activation_document(singing=True), speaker_weight=W_SPEAKER_INTERIOR
    )
    ledger = AuthorityLedger()
    authority = register_optimizer_selection(result, ledger=ledger)
    first = authority.edges[0]
    forged = replace(
        authority,
        edges=(replace(first, lyric=not first.lyric), *authority.edges[1:]),
    )
    with pytest.raises(SealBroken):
        phase1_from_optimizer_selection(
            forged,
            ledger=ledger,
            row_id="delivery_finalizer/v2",
            evaluation_id="speaker-lyric-seal-negative",
        )
    assert ledger.events == ()


def test_standalone_artifact_populates_speaker_block_and_w3_coverage():
    on = optimize_document(activation_document(), speaker_weight=W_SPEAKER_INTERIOR)
    off = optimize_document(activation_document(), speaker_weight=0.0)
    block = on.artifact["speaker_evidence"]

    assert block["attribution"] == "parent-projected"
    assert block["constants"] == {
        "speaker_edge_run_min_s": SPEAKER_EDGE_RUN_MIN_S,
        "speaker_edge_silence_s": 0.3,
        "speaker_min_run_s": SPEAKER_MIN_RUN_S,
        "speaker_multi_min_frac": SPEAKER_MULTI_MIN_FRAC,
        "speaker_unit_cover_frac": SPEAKER_UNIT_COVER_FRAC,
        "speaker_weight_interior": W_SPEAKER_INTERIOR,
    }
    assert block["conditioning"]["transitions_before"] == 1
    assert block["conditioning"]["transitions_after"] == 1
    assert block["pricing"]["priced_edges"] == 3
    assert block["pricing"]["two_speaker_edges"] == 1
    # Two lines were available and the full-span two-speaker candidate really
    # was priced, even though the on row selected two single-speaker cues.
    assert on.artifact["coverage"]["dual_form_unmeasured"] is True
    assert on.artifact["coverage"]["named_multi_cues_unannotated"] == 0
    assert off.artifact["coverage"]["named_multi_cues_unannotated"] == 1
    # W4 owns final delivered boundaries, so the standalone block states that
    # measurement buckets are not yet attached instead of measuring raw cues as
    # if they were finalizer output.
    assert block["measurement"] is None
    assert on.artifact["schema_version"] == 1


def test_single_line_profile_does_not_claim_dual_form_coverage():
    result = optimize_document(
        activation_document(max_lines=1), speaker_weight=W_SPEAKER_INTERIOR
    )
    assert result.artifact["speaker_evidence"]["pricing"]["two_speaker_edges"] >= 1
    assert result.artifact["coverage"]["dual_form_unmeasured"] is False


def test_refined_track_requires_explicit_parent_projection():
    parent = SegDocument(
        language="en",
        units=[source(0, "alpha bravo", 0.0, 2.0)],
        profile=profile(max_line_length=5),
        vad_speech=None,
        shot_changes=None,
        sing_spans=None,
        speaker_turns=[(0.0, 1.0, "A"), (1.0, 2.0, "B")],
        manifest={},
        text="alpha bravo",
    )
    refined, split = refine_document(parent)
    assert split.origin == (0, 0)

    with pytest.raises(ValueError, match="parent-projected speaker evidence"):
        optimize_document(
            refined,
            subunit_split=split,
            speaker_weight=W_SPEAKER_INTERIOR,
        )

    projected = speaker_evidence(parent, refined_units=split.units, origin=split.origin)
    result = optimize_document(refined, subunit_split=split, speakers=projected)
    assert result.speaker_evidence is projected
    assert result.artifact["speaker_evidence"]["attribution"] == "parent-projected"
    assert tuple(item.kind for item in projected.unit_speakers) == (
        "ambiguous",
        "ambiguous",
    )


def test_supplied_projection_must_describe_the_same_turn_track():
    doc = activation_document()
    projected = speaker_evidence(doc)
    doc.speaker_turns = [(0.0, 1.0, "A"), (1.25, 2.25, "C")]
    with pytest.raises(ValueError, match="document track"):
        optimize_document(doc, speakers=projected)


def test_no_turn_refined_documents_keep_existing_w2_callers_valid():
    parent = SegDocument(
        language="en",
        units=[source(0, "alpha bravo", 0.0, 2.0)],
        profile=profile(max_line_length=5),
        vad_speech=None,
        shot_changes=None,
        sing_spans=None,
        speaker_turns=None,
        manifest={},
        text="alpha bravo",
    )
    refined, split = refine_document(parent)
    result = optimize_document(refined, subunit_split=split)
    assert result.artifact["speaker_evidence"] is None
    assert result.speaker_evidence is None

    enabled = optimize_document(
        refined,
        subunit_split=split,
        speaker_weight=W_SPEAKER_INTERIOR,
    )
    assert enabled.artifact["speaker_evidence"]["turn_track_present"] is False
    assert enabled.speaker_evidence is not None
    assert enabled.speaker_evidence.origin == split.origin
