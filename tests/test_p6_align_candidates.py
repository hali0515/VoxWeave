import dataclasses
import json

import pytest


def _context(tmp_path):
    from voxweave.align_context import consume_context_role, issue_align_context
    from voxweave.align_snapshot import FrozenObject, freeze_json

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
    consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
    return context


def _core(context):
    from voxweave.align_acquisition import capture_strict_units, transform_strict_units
    from voxweave.align_distribution import (
        AuthorityBlock,
        AuthorityCallInput,
        RouteClaim,
        RouteExpectation,
        build_authority_distribution,
    )
    from voxweave.align_evidence_core import project_evidence_core

    blocks = (AuthorityBlock(0, "hello"),)
    capture = capture_strict_units(
        ({"text": "hello", "start": 1.0, "end": 2.0},),
        call_index=0,
        raw_unit_ids=("r0",),
    )
    transform = transform_strict_units(
        capture, physical_origin_seconds=0.0, identity=True
    )
    distribution = build_authority_distribution(
        blocks=blocks,
        delivery_route=(RouteExpectation(0, 0, "call", 0),),
        calls=(
            AuthorityCallInput(0, (0,), (0, 1), ("r0",), ("hello",), "valid", None),
        ),
        skipped_blocks=(),
        route_claims=(RouteClaim("call", 0, 0, 0),),
        iso="en",
    )
    return project_evidence_core(
        context_content_digest=context.context_content_digest,
        blocks=blocks,
        captures=(capture,),
        transforms=(transform,),
        distribution=distribution,
        seed_status="valid",
        seed_reasons=(),
    )


def _evaluated(tmp_path, *, shadow_requested=False):
    from voxweave.align_adapter import (
        AlignDelivery,
        AlignDeliveryCue,
        AlignProjectionInputs,
        PersistedAlignUnit,
        SourceBlockDecoration,
        issue_legacy_align_evaluated_result,
    )
    from voxweave.align_snapshot import FrozenObject, freeze_json

    context = _context(tmp_path)
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
    evidence_core = _core(context)
    delivery = AlignDelivery(
        context.context_content_digest,
        evidence_core.core_digest,
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
    result = issue_legacy_align_evaluated_result(
        context,
        delivery=delivery,
        projection_inputs=inputs,
        evidence_core=evidence_core,
        shadow_requested=shadow_requested,
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


def test_shadow_typed_failure_survives_without_affecting_selected_legacy(tmp_path):
    from voxweave.candidate_encoder import (
        CandidateFailure,
        EncodedCandidate,
        encode_align_candidates,
        select_align_candidate,
    )

    context, result = _evaluated(tmp_path, shadow_requested=True)
    candidates = encode_align_candidates(context, result)
    assert isinstance(candidates.outcome_for("legacy-v1"), EncodedCandidate)
    boundary = candidates.outcome_for("boundary-v2")
    assert isinstance(boundary, CandidateFailure)
    assert boundary.failure.detail_code == "w1-root-event"
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
