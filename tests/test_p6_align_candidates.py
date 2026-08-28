import dataclasses
import json

import pytest


def _evaluated(tmp_path, *, shadow_requested=False):
    from voxweave.align_acquisition import (
        _bind_fresh_adapter_payload,
        _fresh_alignment_call_observer,
        _fresh_producer_core_inputs,
        begin_fresh_alignment,
        seal_fresh_alignment,
    )
    from voxweave.align_adapter import (
        AlignDelivery,
        AlignDeliveryCue,
        AlignProjectionInputs,
        PersistedAlignUnit,
        SourceBlockDecoration,
        _adapter_semantic_observation,
        issue_align_evaluated_result,
        run_locked_align_adapter,
    )
    from voxweave.align_context import issue_align_context
    from voxweave.align_evidence_core import build_evidence_core
    from voxweave.align_inputs import (
        LegacyAlignPolicy,
        resolve_align_profile,
        resolve_finalize_evidence,
        validate_v2_policy,
    )
    from voxweave.align_snapshot import (
        FrozenObject,
        StrictInputStatus,
        freeze_json,
    )

    stable_fields = freeze_json({"case": "candidate"})
    assert isinstance(stable_fields, FrozenObject)
    context = issue_align_context(
        stable_fields=stable_fields,
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )
    session = begin_fresh_alignment(
        context,
        alignment_texts=("hello",),
        source_indices=(0,),
        language="en",
        prepared_audio_sample_count=48_000,
    )
    raw_units = ({"text": "hello", "start": 1.0, "end": 2.0},)
    _fresh_alignment_call_observer(session)(raw_units, raw_units, (0,), 0.0)
    acquisition = seal_fresh_alignment(session)

    unit = PersistedAlignUnit("hello", 1.0, 2.0)
    cue = AlignDeliveryCue(
        source_index=0,
        text="hello",
        start=1.0,
        end=2.0,
        lyric=True,
        unit_ids=("r0",),
        word_data=(unit,),
        speech_start=1.0,
        speech_end=2.0,
    )
    delivery = AlignDelivery(
        context.context_content_digest,
        acquisition.receipt_digest,
        "legacy-v1",
        "ctc-full",
        (cue,),
        (unit,),
    )
    segmentation = freeze_json({"manifest_version": 1})
    assert isinstance(segmentation, FrozenObject)
    inputs = AlignProjectionInputs(
        language="en",
        source_blocks=(SourceBlockDecoration(0, "Alice", None),),
        vad_speech=(),
        shot_changes=(1.5,),
        sing_spans=((1.0, 2.0),),
        speaker_turns=((0.0, 3.0, "SPEAKER_00"),),
        voiceprint_capture=None,
        voiceprint_media=None,
        segmentation=segmentation,
    )
    policy = LegacyAlignPolicy(0.0, 0.0, 0.0)
    profile_manifest = {
        "manifest_version": 1,
        "engine": "legacy-v1",
        "language": "en",
        "profile": {
            "max_line_length": 42,
            "max_lines": 2,
            "clause_ms": 400,
            "vad_skip_ms": 1000,
            "offline_ms": 700,
            "min_cue_s": 0,
            "max_cue_s": 0,
            "glue_gap_s": 0,
            "cps": 0,
            "lag_out_s": 0,
            "shot_snap_s": 0,
        },
    }
    profile = resolve_align_profile(profile_manifest, effective_iso="en")
    evidence_resolution = resolve_finalize_evidence(
        shot_changes=(1.5,), sing_spans=((1.0, 2.0),)
    )
    strict_status = StrictInputStatus("valid", None)
    policy_status = validate_v2_policy(policy)
    _bind_fresh_adapter_payload(
        context,
        acquisition,
        legacy_delivery=delivery,
        projection_inputs=inputs,
        strict_input_status=strict_status,
        v2_policy_status=policy_status,
        profile_resolution=profile,
        evidence_resolution=evidence_resolution,
    )
    adapter_result = run_locked_align_adapter(
        context, acquisition, shadow_enabled=shadow_requested
    )
    evidence_core = build_evidence_core(
        _fresh_producer_core_inputs(
            context,
            acquisition,
            strict_input_status=strict_status,
            v2_policy_status=policy_status,
            profile_status=profile.status,
            evidence_status=evidence_resolution.status,
        )
    )
    comparison = None
    if adapter_result.v2_status.kind == "valid":
        from voxweave.core.align_compare import compare_semantic_deltas

        assert adapter_result.v2 is not None
        assert profile.profile is not None
        observation = _adapter_semantic_observation(context, adapter_result)
        assert observation is not None
        comparison = compare_semantic_deltas(
            route_kind=context.route_kind,
            physical_calls=evidence_core.physical_calls,
            authority_blocks=evidence_core.blocks,
            legacy=adapter_result.legacy,
            v2=adapter_result.v2,
            semantic_observation=observation,
            profile=profile.profile,
            evidence=evidence_resolution,
        )
    result = issue_align_evaluated_result(
        context,
        adapter_result,
        evidence_core=evidence_core,
        comparison=comparison,
    )
    return context, result


def test_composite_encoder_returns_fixed_family_order_and_exact_legacy_bytes(tmp_path):
    from voxweave.candidate_encoder import (
        CandidateNotRequested,
        EncodedCandidate,
        encode_align_candidates,
    )

    context, result = _evaluated(tmp_path)
    candidates = encode_align_candidates(context, result)
    assert tuple(family for family, _outcome in candidates.outcomes) == (
        "legacy-v1",
        "boundary-v2",
    )
    legacy = candidates.outcome_for("legacy-v1")
    assert isinstance(legacy, EncodedCandidate)
    assert candidates.outcome_for("boundary-v2") == CandidateNotRequested()
    assert legacy.vtt_bytes == (
        b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n"
        b"<v Alice>\xe2\x99\xaa hello \xe2\x99\xaa</v>\n"
    )
    data = json.loads(legacy.main_json_bytes)
    assert tuple(data) == (
        "language",
        "segments",
        "word_segments",
        "vad_speech",
        "shot_changes",
        "sing_spans",
        "speaker_turns",
        "segmentation",
    )
    assert data["segments"] == [
        {"text": "hello", "start": 1.0, "end": 2.0, "lyric": True}
    ]
    assert data["word_segments"] == [{"text": "hello", "start": 1.0, "end": 2.0}]
    assert not legacy.main_json_bytes.endswith(b"\n")


def test_candidate_set_is_issuer_only_and_encoder_role_is_single_use(tmp_path):
    from voxweave.candidate_encoder import CandidateSet, encode_align_candidates

    with pytest.raises(TypeError):
        CandidateSet()  # type: ignore[call-arg]
    context, result = _evaluated(tmp_path)
    encode_align_candidates(context, result)
    with pytest.raises(Exception) as error:
        encode_align_candidates(context, result)
    assert getattr(error.value, "detail_code", None) == "context-consumed"


def test_ratified_shadow_candidate_does_not_change_selected_legacy(tmp_path):
    from voxweave.candidate_encoder import (
        EncodedCandidate,
        encode_align_candidates,
        select_align_candidate,
    )

    context, result = _evaluated(tmp_path, shadow_requested=True)
    candidates = encode_align_candidates(context, result)
    assert isinstance(candidates.outcome_for("legacy-v1"), EncodedCandidate)
    boundary = candidates.outcome_for("boundary-v2")
    assert isinstance(boundary, EncodedCandidate)
    selected = select_align_candidate(context, candidates)
    assert selected.engine_family == "legacy-v1"


def test_selected_failure_has_no_fallback(tmp_path, monkeypatch):
    import voxweave.candidate_encoder as encoder

    context, result = _evaluated(tmp_path, shadow_requested=True)

    def fail_legacy(*args, **kwargs):
        raise UnicodeError("injected legacy renderer failure")

    monkeypatch.setattr(encoder, "project_align_delivery", fail_legacy)
    candidates = encoder.encode_align_candidates(context, result)
    assert isinstance(candidates.outcome_for("legacy-v1"), encoder.CandidateFailure)
    with pytest.raises(encoder.SelectedCandidateError) as error:
        encoder.select_align_candidate(context, candidates)
    assert error.value.failure.detail_code == "selected-candidate-missing"


def test_vtt_encode_failure_is_classified_before_main_json_encode(
    tmp_path, monkeypatch
):
    from voxweave import candidate_encoder
    from voxweave import realign

    context, result = _evaluated(tmp_path, shadow_requested=True)
    monkeypatch.setattr(
        realign,
        "render_cues",
        lambda _rows: (_ for _ in ()).throw(UnicodeError("vtt encode failed exactly")),
    )

    candidates = candidate_encoder.encode_align_candidates(context, result)
    legacy = candidates.outcome_for("legacy-v1")

    assert isinstance(legacy, candidate_encoder.CandidateFailure)
    assert legacy.failure.kind == "preencode-failed"
    assert legacy.failure.phase == "encoder"
    assert legacy.failure.detail_code == "vtt-encode"


def test_independent_projection_rejects_candidate_byte_corruption(tmp_path):
    from voxweave.candidate_encoder import (
        SelectedRenderError,
        encode_align_candidates,
        select_align_candidate,
        verify_selected_align_projection,
    )

    context, result = _evaluated(tmp_path)
    selected = select_align_candidate(context, encode_align_candidates(context, result))
    verified = verify_selected_align_projection(context, result, selected)
    assert verified.vtt_sha256 == selected.vtt_sha256
    corrupted = dataclasses.replace(selected, vtt_bytes=selected.vtt_bytes + b"x")
    with pytest.raises(SelectedRenderError) as error:
        verify_selected_align_projection(context, result, corrupted)
    assert error.value.failure.detail_code == "derived-hash"
