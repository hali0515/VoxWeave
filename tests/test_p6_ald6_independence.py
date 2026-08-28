from __future__ import annotations

import dataclasses
import ast
import inspect

import pytest


def _issued_facts(tmp_path):
    from voxweave.align_acquisition import (
        _bind_fresh_adapter_payload,
        _fresh_alignment_call_observer,
        _fresh_producer_core_inputs,
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
    _fresh_alignment_call_observer(session)(
        ({"text": "word", "start": 0.2, "end": 0.8},),
        None,
        (0,),
        0.0,
    )
    acquisition = seal_fresh_alignment(session)
    unit = PersistedAlignUnit("word", 0.2, 0.8)
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
            (
                AlignDeliveryCue(
                    0,
                    "word",
                    0.2,
                    0.8,
                    None,
                    ("r0",),
                    (unit,),
                    0.2,
                    0.8,
                ),
            ),
            (unit,),
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
    reference = _fresh_reference_core_inputs(context, acquisition)
    return producer, reference


@pytest.mark.parametrize(
    "field",
    (
        "legacy_slice_digest",
        "legacy_absolute_digest",
        "backend_model_config_digest",
        "route_input_digest",
    ),
)
def test_ald6_reference_rejects_alternate_legacy_model_and_route_digests(
    tmp_path,
    field,
):
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        project_evidence_core,
    )

    _producer, reference = _issued_facts(tmp_path)
    corrupt_call = dataclasses.replace(
        reference.claimed_physical_calls[0], **{field: "d" * 64}
    )
    corrupted = dataclasses.replace(reference, claimed_physical_calls=(corrupt_call,))
    with pytest.raises(EvidenceCoreProjectionError, match="digest|cross-link"):
        project_evidence_core(corrupted)


@pytest.mark.parametrize(
    "family",
    ("physical", "legacy", "topology", "profile", "admission", "anchor", "digest"),
)
def test_ald6_rejects_issuer_time_field_family_corruption(tmp_path, family):
    from voxweave.align_evidence_core import (
        build_evidence_core,
        evaluate_ald6,
        project_evidence_core,
    )
    from voxweave.align_inputs import ProfileStatus
    from voxweave.align_snapshot import StrictInputStatus

    producer, reference = _issued_facts(tmp_path)
    if family == "physical":
        call = dataclasses.replace(
            producer.physical_calls[0], physical_origin_seconds=0.125
        )
        producer = dataclasses.replace(producer, physical_calls=(call,))
    elif family == "legacy":
        receipt = dataclasses.replace(
            producer.legacy_receipts[0],
            final_cursor=producer.legacy_receipts[0].final_cursor + 1,
        )
        producer = dataclasses.replace(producer, legacy_receipts=(receipt,))
    elif family == "topology":
        claim = dataclasses.replace(
            producer.distribution.work.route_claims[0], source_index=7
        )
        work = dataclasses.replace(producer.distribution.work, route_claims=(claim,))
        producer = dataclasses.replace(
            producer,
            distribution=dataclasses.replace(producer.distribution, work=work),
        )
    elif family == "profile":
        producer = dataclasses.replace(
            producer,
            profile_status=ProfileStatus("invalid", "stored-profile", "profile-shape"),
        )
    elif family == "admission":
        producer = dataclasses.replace(
            producer,
            strict_input_status=StrictInputStatus("invalid", "sibling-json-nonfinite"),
        )
    elif family == "anchor":
        transform = producer.transforms[0]
        assert transform.units is not None
        unit = dataclasses.replace(transform.units[0], start=0.3)
        producer = dataclasses.replace(
            producer,
            transforms=(dataclasses.replace(transform, units=(unit,)),),
        )
    else:
        call = dataclasses.replace(
            producer.physical_calls[0], route_input_digest="d" * 64
        )
        producer = dataclasses.replace(producer, physical_calls=(call,))

    producer_core = build_evidence_core(producer)
    reference_core = project_evidence_core(reference)
    outcome = evaluate_ald6(producer_core, reference_core)
    assert outcome.triggered is True
    assert outcome.passed is False


def test_ald6_producer_and_reference_have_disjoint_local_projector_closures():
    import voxweave.align_evidence_core as module

    tree = ast.parse(inspect.getsource(module))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def local_closure(entry: str) -> set[str]:
        pending = [entry]
        visited: set[str] = set()
        while pending:
            name = pending.pop()
            node = functions[name]
            for call in (
                member for member in ast.walk(node) if isinstance(member, ast.Call)
            ):
                if not isinstance(call.func, ast.Name):
                    continue
                called = call.func.id
                if called in functions and called not in visited:
                    visited.add(called)
                    pending.append(called)
        visited.discard(entry)
        return visited

    assert "_assemble_evidence_core" not in functions
    producer = local_closure("build_evidence_core")
    reference = local_closure("project_evidence_core")
    assert producer
    assert reference
    assert producer.isdisjoint(reference)
    assert all(name.startswith("_p_") for name in producer)
    assert all(name.startswith("_r_") for name in reference)
