import json

import pytest


def _verified(tmp_path):
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
        issue_align_evaluated_result,
        run_locked_align_adapter,
    )
    from voxweave.candidate_encoder import (
        encode_align_candidates,
        select_align_candidate,
        verify_selected_align_projection,
    )
    from voxweave.align_evidence_core import build_evidence_core
    from voxweave.align_inputs import (
        LegacyAlignPolicy,
        resolve_align_profile,
        resolve_finalize_evidence,
        validate_v2_policy,
    )
    from voxweave.align_orchestration import issue_public_align_context
    from voxweave.align_snapshot import StrictInputStatus
    from voxweave.episode_transaction import FileGeneration

    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"physical-audio")
    blocks = ({"source_index": 0, "text": "hello", "alignment_text": "hello"},)
    policy = LegacyAlignPolicy(0.0, 0.0, 0.0)
    context = issue_public_align_context(
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        prepared_audio_path=prepared,
        expected_vtt=FileGeneration(True, b"WEBVTT\n"),
        expected_json=FileGeneration(True, b"{}"),
        expected_vtt_sha256=None,
        media_fingerprint="a" * 64,
        effective_iso="en",
        route_kind="ctc-full",
        blocks=blocks,
        prepared_audio_sha256="b" * 64,
        legacy_policy=policy,
        stored_language="en",
        segmentation=None,
        strict_shot_changes=None,
        strict_sing_spans=None,
    )
    session = begin_fresh_alignment(
        context,
        alignment_texts=("hello",),
        source_indices=(0,),
        language="en",
        prepared_audio_sample_count=16_000,
    )
    _fresh_alignment_call_observer(session)(
        ({"text": "hello", "start": 1.0, "end": 2.0},), None, (0,), 0.0
    )
    acquisition = seal_fresh_alignment(session)
    unit = PersistedAlignUnit("hello", 1.0, 2.0)
    delivery = AlignDelivery(
        context.context_content_digest,
        acquisition.receipt_digest,
        "legacy-v1",
        "ctc-full",
        (AlignDeliveryCue(0, "hello", 1.0, 2.0, None, ("r0",), (unit,), 1.0, 2.0),),
        (unit,),
    )
    strict = StrictInputStatus("valid", None)
    policy_status = validate_v2_policy(policy)
    profile = resolve_align_profile(None, effective_iso="en", stored_iso="en")
    evidence = resolve_finalize_evidence(shot_changes=None, sing_spans=None)
    _bind_fresh_adapter_payload(
        context,
        acquisition,
        legacy_delivery=delivery,
        projection_inputs=AlignProjectionInputs(
            "en",
            (SourceBlockDecoration(0, None, None),),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        strict_input_status=strict,
        v2_policy_status=policy_status,
        profile_resolution=profile,
        evidence_resolution=evidence,
    )
    adapter = run_locked_align_adapter(context, acquisition, shadow_enabled=False)
    core = build_evidence_core(
        _fresh_producer_core_inputs(
            context,
            acquisition,
            strict_input_status=strict,
            v2_policy_status=policy_status,
            profile_status=profile.status,
            evidence_status=evidence.status,
        )
    )
    result = issue_align_evaluated_result(
        context, adapter, evidence_core=core, comparison=None
    )
    candidate = select_align_candidate(
        context, encode_align_candidates(context, result)
    )
    verified = verify_selected_align_projection(context, result, candidate)
    return (
        context,
        result,
        verified,
        acquisition,
        strict,
        policy_status,
        profile.status,
        evidence.status,
    )


def test_final_evidence_binds_only_after_verified_primary_hashes(tmp_path):
    from voxweave.align_context import role_vector
    from voxweave.align_evidence import bind_align_evidence, encode_align_evidence

    context, result, verified, acquisition, strict, policy, profile, evidence_status = (
        _verified(tmp_path)
    )
    evidence = bind_align_evidence(
        context,
        result.evidence_core,
        acquisition=acquisition,
        strict_input_status=strict,
        v2_policy_status=policy,
        profile_status=profile,
        evidence_status=evidence_status,
        engine_family=verified.engine_family,
        vtt_sha256=verified.vtt_sha256,
        main_json_sha256=verified.main_json_sha256,
    )
    assert evidence.selected_outputs.engine_family == "legacy-v1"
    assert evidence.selected_outputs.vtt_sha256 == verified.vtt_sha256
    assert evidence.durable_authority is True
    encoded = encode_align_evidence(evidence)
    parsed = json.loads(encoded)
    assert tuple(parsed)[-1] == "selected_outputs"
    assert parsed["selected_outputs"]["vtt_sha256"] == verified.vtt_sha256
    assert encoded.endswith(b"\n")
    assert role_vector(context) == ("C", "C", "C", "C", "L")


def test_evidence_bind_rejects_unverified_hash_substitution(tmp_path):
    from voxweave.align_evidence import EvidenceBindingError, bind_align_evidence

    context, result, verified, acquisition, strict, policy, profile, evidence_status = (
        _verified(tmp_path)
    )
    with pytest.raises(EvidenceBindingError) as error:
        bind_align_evidence(
            context,
            result.evidence_core,
            acquisition=acquisition,
            strict_input_status=strict,
            v2_policy_status=policy,
            profile_status=profile,
            evidence_status=evidence_status,
            engine_family=verified.engine_family,
            vtt_sha256="0" * 64,
            main_json_sha256=verified.main_json_sha256,
        )
    assert error.value.failure.detail_code == "selected-hash-link"


def test_rat2_exposes_only_the_path_bound_durable_verifier():
    import inspect

    import voxweave.align_evidence as module

    assert not hasattr(module, "verify_durable_align_evidence")
    assert tuple(inspect.signature(module.verify_align_evidence).parameters) == (
        "vtt_path",
        "explicit_media_path",
        "corpus_root",
    )
