import dataclasses
import json

import pytest


def _frozen_object(value):
    from voxweave.align_snapshot import FrozenObject, freeze_json

    frozen = freeze_json(value)
    assert isinstance(frozen, FrozenObject)
    return frozen


def _frozen_array(value):
    from voxweave.align_snapshot import FrozenArray, freeze_json

    frozen = freeze_json(value)
    assert isinstance(frozen, FrozenArray)
    return frozen


def _frozen_value(value):
    from voxweave.align_snapshot import freeze_json

    return freeze_json(value)


def _profile():
    from voxweave.core.segdoc import DisplayProfile

    return DisplayProfile(
        language="en",
        max_line_length=42,
        max_lines=2,
        clause_ms=400.0,
        vad_skip_ms=1000.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=7.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=0.458,
    )


def _document():
    from voxweave.core.segdoc import SegDocument, SourceUnit

    return SegDocument(
        language="en",
        units=[
            SourceUnit("u0", "hello", 0.0, 0.4),
            SourceUnit("u1", "world", 0.5, 1.0),
        ],
        profile=_profile(),
        vad_speech=[(0.0, 1.0)],
        shot_changes=[0.45],
        sing_spans=[(0.0, 0.4)],
        speaker_turns=[(0.0, 1.1, "S0")],
        manifest={},
        text="hello world",
    )


def _issued(tmp_path, *, timestamps=True):
    from voxweave.align_context import issue_segmentation_context
    from voxweave.align_snapshot import RawJSONCarrier
    from voxweave.segmentation_adapter import (
        SegmentationCarriers,
        SegmentationDelivery,
        SegmentationDeliveryCue,
        SegmentationProjectionInputs,
        issue_legacy_segmentation,
    )

    manifest = _frozen_object(
        {
            "manifest_version": 1,
            "engine": "legacy-v1",
            "voxweave": "test",
            "python": "test",
            "language": "en",
            "profile": {"max_line_length": 42, "max_lines": 2},
            "env": {"gap_adaptive": False, "vad_emission_mask": False},
            "providers": {"layout": "test"},
            "degraded": [],
        }
    )
    units = (
        _frozen_object({"text": "hello", "start": 0.0, "end": 0.4}),
        _frozen_object({"text": "world", "start": 0.5, "end": 1.0}),
    )
    cues = (
        SegmentationDeliveryCue(
            (0, 1),
            "hello",
            0.0,
            0.4,
            (units[0],),
            0.0,
            0.4,
            True,
            ("S0",),
        ),
        SegmentationDeliveryCue(
            (1, 2),
            "world",
            0.5,
            1.0,
            (units[1],),
            0.5,
            1.0,
            None,
            ("S0",),
        ),
    )
    carriers = SegmentationCarriers(
        vad_speech=(_frozen_array([0.0, 1.0]),),
        shot_changes=(_frozen_value(0.45),),
        sing_spans=(_frozen_array([0.0, 0.4]),),
        speaker_turns=RawJSONCarrier(
            True, _frozen_array([[0.0, 1.1, "S0"]])
        ),
        voiceprint_capture="capture_abcdefghijklmnopqrstuvwxyz234567",
        voiceprint_media="a" * 64,
    )
    stable = _frozen_object({"case": "segmentation", "timestamps": timestamps})
    context = issue_segmentation_context(
        stable_fields=stable,
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        effective_iso="en",
    )
    delivery = SegmentationDelivery(
        context.context_content_digest,
        "legacy-v1",
        "en",
        cues,
        units,
        carriers,
        manifest,
    )
    projection_inputs = SegmentationProjectionInputs(
        timestamps=timestamps,
        speaker_names=(("S0", "Alice"),),
    )
    issued = issue_legacy_segmentation(
        context,
        delivery=delivery,
        projection_inputs=projection_inputs,
        document=_document(),
    )
    return context, issued


def _result(tmp_path, *, shadow_enabled=False, semantic_selector_enabled=False, timestamps=True):
    from voxweave.segmentation_adapter import run_locked_segmentation_adapter

    context, issued = _issued(tmp_path, timestamps=timestamps)
    result = run_locked_segmentation_adapter(
        context,
        issued,
        shadow_enabled=shadow_enabled,
        semantic_selector_enabled=semantic_selector_enabled,
    )
    return context, result


def test_composite_segmentation_encoder_preserves_exact_legacy_projection(tmp_path):
    from voxweave.align_context import role_vector
    from voxweave.candidate_encoder import CandidateNotRequested, EncodedCandidate
    from voxweave.segmentation_candidates import encode_segmentation_candidates

    context, result = _result(tmp_path)
    candidates = encode_segmentation_candidates(context, result)
    assert tuple(family for family, _outcome in candidates.outcomes) == (
        "legacy-v1",
        "boundary-v2",
    )
    legacy = candidates.outcome_for("legacy-v1")
    assert isinstance(legacy, EncodedCandidate)
    assert candidates.outcome_for("boundary-v2") == CandidateNotRequested()
    assert legacy.vtt_bytes == (
        b"WEBVTT\n\n00:00:00.000 --> 00:00:00.400\n"
        b"<v Alice>\xe2\x99\xaa hello \xe2\x99\xaa</v>\n\n"
        b"00:00:00.500 --> 00:00:01.000\n<v Alice>world</v>\n"
    )
    value = json.loads(legacy.main_json_bytes)
    assert tuple(value) == (
        "language",
        "segments",
        "word_segments",
        "vad_speech",
        "shot_changes",
        "sing_spans",
        "speaker_turns",
        "voiceprint_capture",
        "voiceprint_media",
        "segmentation",
    )
    assert tuple(value["segments"][0]) == (
        "text",
        "start",
        "end",
        "word_data",
        "lyric",
    )
    assert "speech_start" not in value["segments"][0]
    assert "speaker_ids" not in value["segments"][0]
    assert value["segmentation"]["engine"] == "legacy-v1"
    assert not legacy.main_json_bytes.endswith(b"\n")
    assert role_vector(context) == ("C", "C", "L")


def test_encoder_role_is_single_use_and_candidate_set_remains_issuer_only(tmp_path):
    from voxweave.candidate_encoder import CandidateSet
    from voxweave.segmentation_candidates import encode_segmentation_candidates

    with pytest.raises(TypeError):
        CandidateSet()  # type: ignore[call-arg]
    context, result = _result(tmp_path)
    encode_segmentation_candidates(context, result)
    with pytest.raises(Exception) as error:
        encode_segmentation_candidates(context, result)
    assert getattr(error.value, "detail_code", None) == "context-consumed"


def test_shadow_builds_real_p5_boundary_delivery_without_selecting_it(tmp_path):
    from voxweave.candidate_encoder import EncodedCandidate
    from voxweave.segmentation_candidates import (
        encode_segmentation_candidates,
        select_segmentation_candidate,
    )

    context, result = _result(tmp_path, shadow_enabled=True)
    assert result.v2_status.kind == "valid"
    assert result.v2 is not None
    assert result.v2.engine_family == "boundary-v2"
    assert result.v2.top_level_word_segments == result.legacy.top_level_word_segments
    candidates = encode_segmentation_candidates(context, result)
    assert isinstance(candidates.outcome_for("boundary-v2"), EncodedCandidate)
    selected = select_segmentation_candidate(context, candidates)
    assert selected.engine_family == "legacy-v1"


def test_rat6_semantic_mode_is_typed_nonselected_failure_with_no_fallback(tmp_path):
    from voxweave.candidate_encoder import CandidateFailure, SelectedCandidateError
    from voxweave.segmentation_candidates import (
        _issue_simulated_boundary_row,
        encode_segmentation_candidates,
        select_qualified_segmentation_candidate,
        select_segmentation_candidate,
    )

    context, result = _result(
        tmp_path,
        shadow_enabled=True,
        semantic_selector_enabled=True,
    )
    assert result.v2_status.kind == "invalid"
    assert result.v2_status.failure is not None
    assert result.v2_status.failure.detail_code == "semantic-selector-unmodelled"
    candidates = encode_segmentation_candidates(context, result)
    boundary = candidates.outcome_for("boundary-v2")
    assert isinstance(boundary, CandidateFailure)
    assert select_segmentation_candidate(context, candidates).engine_family == "legacy-v1"
    qualification = _issue_simulated_boundary_row("rat6-no-fallback")
    with pytest.raises(SelectedCandidateError) as error:
        select_qualified_segmentation_candidate(
            context, candidates, qualification=qualification
        )
    assert error.value.failure.detail_code == "selected-candidate-missing"


def test_simulated_row_qualification_selects_and_verifies_real_boundary_bytes(tmp_path):
    from voxweave.engine_registry import LANGUAGE_ENGINE_FAMILY
    from voxweave.segmentation_candidates import (
        _issue_simulated_boundary_row,
        encode_segmentation_candidates,
        select_qualified_segmentation_candidate,
        verify_selected_segmentation_projection,
    )

    context, result = _result(tmp_path, shadow_enabled=True)
    candidates = encode_segmentation_candidates(context, result)
    qualification = _issue_simulated_boundary_row("projection")
    selected = select_qualified_segmentation_candidate(
        context, candidates, qualification=qualification
    )
    verified = verify_selected_segmentation_projection(context, result, selected)
    assert verified.engine_family == "boundary-v2"
    assert json.loads(verified.main_json_bytes)["segmentation"]["engine"] == (
        "boundary-optimizer-v2"
    )
    assert set(LANGUAGE_ENGINE_FAMILY.values()) == {"legacy-v1"}
    with pytest.raises(Exception):
        select_qualified_segmentation_candidate(
            context, candidates, qualification=qualification
        )


def test_independent_segmentation_projection_rejects_byte_corruption(tmp_path):
    from voxweave.candidate_encoder import SelectedRenderError
    from voxweave.segmentation_candidates import (
        encode_segmentation_candidates,
        select_segmentation_candidate,
        verify_selected_segmentation_projection,
    )

    context, result = _result(tmp_path)
    selected = select_segmentation_candidate(
        context, encode_segmentation_candidates(context, result)
    )
    verify_selected_segmentation_projection(context, result, selected)
    corrupted = dataclasses.replace(
        selected, main_json_bytes=selected.main_json_bytes + b"\n"
    )
    with pytest.raises(SelectedRenderError) as error:
        verify_selected_segmentation_projection(context, result, corrupted)
    assert error.value.failure.detail_code == "derived-hash"


def test_selected_sdh_dialogue_uses_intrinsic_delivery_times_when_vtt_is_plain(tmp_path):
    from voxweave.segmentation_candidates import (
        encode_segmentation_candidates,
        project_selected_sdh_dialogue,
        select_segmentation_candidate,
        verify_selected_segmentation_projection,
    )

    context, result = _result(tmp_path, timestamps=False)
    selected = select_segmentation_candidate(
        context, encode_segmentation_candidates(context, result)
    )
    verified = verify_selected_segmentation_projection(context, result, selected)
    assert b"-->" not in verified.vtt_bytes
    dialogue = project_selected_sdh_dialogue(context, result, verified)
    assert dialogue == (
        {"text": "hello", "start": 0.0, "end": 0.4, "lyric": True},
        {"text": "world", "start": 0.5, "end": 1.0},
    )
    assert all("speaker_ids" not in cue for cue in dialogue)
