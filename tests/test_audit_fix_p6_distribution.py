"""Evidence-core regressions for the P6 authority distribution replay."""

from __future__ import annotations

import contextlib

import pytest


def _core_facts(tmp_path, *, observed_calls, limits=None):
    """Seal one ctc-full acquisition and return (producer, reference) inputs.

    ``observed_calls`` is a sequence of ``(units, source_positions, origin)``
    tuples replayed through the issuer's physical-call observer; an empty
    sequence seals an acquisition whose every block is skipped.  ``limits``
    optionally runs the whole acquisition under a lowered test-only profile.
    """
    from voxweave import align_distribution
    from voxweave.align_acquisition import (
        _bind_fresh_adapter_payload,
        _fresh_alignment_call_observer,
        _fresh_producer_core_inputs,
        _fresh_record,
        _fresh_reference_core_inputs,
        begin_fresh_alignment,
        seal_fresh_alignment,
    )
    from voxweave.align_adapter import (
        AlignDelivery,
        AlignDeliveryCue,
        AlignProjectionInputs,
        PersistedAlignUnit,
        SourceBlockDecoration,
    )
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
    policy = LegacyAlignPolicy(0.0, 0.0, 0.0)
    scope: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    if limits is not None:
        token = align_distribution._issue_test_authority_limit_qualification(
            "audit-fix-evidence-core", *limits
        )
        scope = align_distribution._with_test_authority_limit_qualification(token)
    with scope:
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
            blocks=({"source_index": 0, "text": "word", "alignment_text": "word"},),
            prepared_audio_sha256="b" * 64,
            legacy_policy=policy,
            stored_language="en",
            segmentation=None,
            strict_shot_changes=None,
            strict_sing_spans=None,
        )
    session = begin_fresh_alignment(
        context,
        alignment_texts=("word",),
        source_indices=(0,),
        language="en",
        prepared_audio_sample_count=16_000,
        backend_model_config_facts={"model": "ctc", "revision": "test"},
        route_input_facts={"route": "ctc-full", "sources": [0]},
    )
    observe = _fresh_alignment_call_observer(session)
    for units, positions, origin in observed_calls:
        observe(units, None, positions, origin)
    acquisition = seal_fresh_alignment(session)
    (legacy_units,) = _fresh_record(context, acquisition).legacy_block_units
    # The delivery must restate the sealed legacy slice exactly, raw scalar
    # types included (an identity slice keeps an int bound as an int).
    word_data = tuple(
        PersistedAlignUnit(unit["text"], unit["start"], unit["end"])
        for unit in legacy_units
    )
    strict = StrictInputStatus("valid", None)
    policy_status = validate_v2_policy(policy)
    profile = resolve_align_profile(None, effective_iso="en", stored_iso="en")
    evidence = resolve_finalize_evidence(shot_changes=None, sing_spans=None)
    _bind_fresh_adapter_payload(
        context,
        acquisition,
        legacy_delivery=AlignDelivery(
            context.context_content_digest,
            acquisition.receipt_digest,
            "legacy-v1",
            "ctc-full",
            (AlignDeliveryCue(0, "word", 0.2, 0.8, None, (), word_data, None, None),),
            word_data,
        ),
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
    producer = _fresh_producer_core_inputs(
        context,
        acquisition,
        strict_input_status=strict,
        v2_policy_status=policy_status,
        profile_status=profile.status,
        evidence_status=evidence.status,
    )
    return producer, _fresh_reference_core_inputs(context, acquisition)


def _assert_ald6_passes(producer, reference):
    from voxweave.align_evidence_core import (
        build_evidence_core,
        evaluate_ald6,
        project_evidence_core,
    )

    producer_core = build_evidence_core(producer)
    reference_core = project_evidence_core(reference)
    assert evaluate_ald6(producer_core, reference_core).passed is True
    return producer_core


def test_reference_transform_reports_capture_failure_before_geometry(tmp_path):
    # An int end is a strict raw-node defect; a nonfinite legacy origin is a
    # sample-geometry defect.  The producer reports the capture failure first,
    # and the reference must agree instead of aborting the align.
    producer, reference = _core_facts(
        tmp_path,
        observed_calls=(
            (({"text": "word", "start": 0.2, "end": 1},), (0,), float("nan")),
        ),
    )
    call = reference.reference_calls[0]
    assert call.geometry_failure is not None
    assert reference.captures[0].status == "invalid"
    assert reference.transforms[0].failure == reference.captures[0].failure

    core = _assert_ald6_passes(producer, reference)
    failure = core.physical_calls[0].strict_failure
    assert failure is not None
    assert (failure.stage, failure.detail_code) == ("strict-capture", "strict-raw-node")


def test_reference_replays_zero_call_receipt_under_test_only_profile(tmp_path):
    from voxweave.align_distribution import CallWorkLimits, JobWorkLimits

    call_limits = CallWorkLimits(10, 20, 10, 100)
    producer, reference = _core_facts(
        tmp_path,
        observed_calls=(),
        limits=(call_limits, JobWorkLimits(2, 20, 40, 20, 200)),
    )
    work = reference.distribution.work
    assert work.limit_profile_kind == "test-only"
    assert work.calls == ()
    assert work.status == "not-run-skip-invalid"
    assert reference.authority_profile.call == call_limits

    core = _assert_ald6_passes(producer, reference)
    assert core.authority_reasons == ("partial-empty-ownership",)


def test_reference_rejects_zero_call_profile_digest_that_matches_no_limits(tmp_path):
    import dataclasses

    from voxweave.align_distribution import CallWorkLimits, JobWorkLimits
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        project_evidence_core,
    )

    _producer, reference = _core_facts(
        tmp_path,
        observed_calls=(),
        limits=(CallWorkLimits(10, 20, 10, 100), JobWorkLimits(2, 20, 40, 20, 200)),
    )
    distribution = reference.distribution
    corrupt = dataclasses.replace(
        distribution,
        work=dataclasses.replace(distribution.work, limit_profile_digest="f" * 64),
    )
    with pytest.raises(EvidenceCoreProjectionError, match="allocator"):
        project_evidence_core(dataclasses.replace(reference, distribution=corrupt))
