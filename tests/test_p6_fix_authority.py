"""RED gates for the P6 acquisition-authority fix and ratified RAT-1/RAT-5."""

from __future__ import annotations

import ast
import dataclasses
import inspect

import pytest


def _evidence_facts():
    from voxweave.align_acquisition import capture_strict_units, transform_strict_units
    from voxweave.align_distribution import (
        AuthorityBlock,
        AuthorityCallInput,
        RouteClaim,
        RouteExpectation,
        build_authority_distribution,
    )

    blocks = (AuthorityBlock(0, "word"),)
    capture = capture_strict_units(
        ({"text": "word", "start": 0.2, "end": 0.8},),
        call_index=0,
        raw_unit_ids=("r0",),
    )
    transform = transform_strict_units(
        capture,
        physical_origin_seconds=100.0,
        identity=False,
    )
    distribution = build_authority_distribution(
        blocks=blocks,
        delivery_route=(RouteExpectation(0, 0, "call", 0),),
        calls=(
            AuthorityCallInput(
                0,
                (0,),
                (0, 1),
                ("r0",),
                ("word",),
                "valid",
                None,
            ),
        ),
        skipped_blocks=(),
        route_claims=(RouteClaim("call", 0, 0, 0),),
        iso="en",
    )
    return blocks, capture, transform, distribution


def _project_core():
    from voxweave.align_evidence_core import project_evidence_core

    blocks, capture, transform, distribution = _evidence_facts()
    core = project_evidence_core(
        context_content_digest="a" * 64,
        blocks=blocks,
        captures=(capture,),
        transforms=(transform,),
        distribution=distribution,
        seed_status="valid",
        seed_reasons=(),
    )
    return core, (blocks, capture, transform, distribution)


def _call_count(module: object, function_name: str) -> int:
    tree = ast.parse(inspect.getsource(module))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == function_name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == function_name
        )
    )


def test_ald6_rejects_the_same_evidence_core_object():
    from voxweave.align_evidence_core import evaluate_ald6

    core, _facts = _project_core()
    with pytest.raises((TypeError, ValueError), match="independent|distinct|same"):
        evaluate_ald6(core, core)


def test_evidence_reference_replays_and_rejects_corrupt_allocator_receipts():
    from voxweave.align_distribution import WorkCounters
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        project_evidence_core,
    )

    _core, (blocks, capture, transform, distribution) = _project_core()
    corrupt_counters = WorkCounters(123_456, 234_567, 345_678, 456_789)
    row = distribution.work.calls[0]
    corrupt_row = dataclasses.replace(
        row,
        allocator=dataclasses.replace(row.allocator, counters=corrupt_counters),
    )
    corrupt_work = dataclasses.replace(
        distribution.work,
        calls=(corrupt_row,),
        totals=corrupt_counters,
        limit_profile_digest="f" * 64,
    )
    corrupt_distribution = dataclasses.replace(distribution, work=corrupt_work)

    with pytest.raises(
        EvidenceCoreProjectionError,
        match="allocator|counter|profile|receipt|replay",
    ):
        project_evidence_core(
            context_content_digest="a" * 64,
            blocks=blocks,
            captures=(capture,),
            transforms=(transform,),
            distribution=corrupt_distribution,
            seed_status="valid",
            seed_reasons=(),
        )


def test_call_batches_are_private_to_the_acquisition_issuer():
    from voxweave import align_orchestration, pipeline

    public_names = set(getattr(align_orchestration, "__all__", ()))
    assert not hasattr(align_orchestration, "RawAlignmentCall")
    assert not hasattr(align_orchestration, "capture_raw_alignment_call")
    assert "RawAlignmentCall" not in public_names
    assert "capture_raw_alignment_call" not in public_names
    assert (
        "raw_calls"
        not in inspect.signature(align_orchestration.build_align_selection).parameters
    )
    pipeline_source = inspect.getsource(pipeline)
    assert "RawAlignmentCall" not in pipeline_source
    assert "capture_raw_alignment_call" not in pipeline_source


def test_issued_acquisition_types_and_physical_receipt_are_closed_and_complete():
    from voxweave import align_acquisition

    required = {
        "FreshAlignmentIssuer",
        "IssuedFreshAlignment",
        "VerifiedFreshAlignment",
        "PhysicalCallReceipt",
    }
    assert required <= set(vars(align_acquisition))
    assert "FreshAlignmentIssuer" not in set(getattr(align_acquisition, "__all__", ()))

    receipt_type = align_acquisition.PhysicalCallReceipt
    assert dataclasses.is_dataclass(receipt_type)
    assert tuple(field.name for field in dataclasses.fields(receipt_type)) == (
        "call_index",
        "source_block_indices",
        "audio_sample_start",
        "audio_sample_end",
        "sample_rate",
        "physical_origin_seconds",
        "legacy_origin_seconds",
        "legacy_origin_kind",
        "authority_origin_seconds",
        "backend_model_config_digest",
        "route_input_digest",
        "strict_unit_status",
        "strict_failure",
        "raw_units_digest",
        "normalized_relative_digest",
        "legacy_slice_digest",
        "legacy_absolute_digest",
        "authority_transform_status",
        "authority_absolute_digest",
        "raw_unit_ids",
    )
    for issuer_only in (
        align_acquisition.IssuedFreshAlignment,
        align_acquisition.VerifiedFreshAlignment,
    ):
        with pytest.raises(TypeError):
            issuer_only()

    stable_identity_sources = inspect.getsource(align_acquisition)
    stable_identity_sources += inspect.getsource(
        __import__("voxweave.align_orchestration", fromlist=["*"])
    )
    assert "sibling:" in stable_identity_sources
    assert "explicit:" in stable_identity_sources


def test_hostile_surplus_is_touched_only_after_legacy_selection_closes():
    from voxweave import align_acquisition, align_orchestration
    from voxweave.align_acquisition import capture_strict_units
    from voxweave.align_distribution import legacy_distribute_before_shift

    class HostileSurplus:
        deepcopy_calls = 0

        def __deepcopy__(self, _memo):
            type(self).deepcopy_calls += 1
            raise AssertionError("hostile surplus was recursively traversed")

    hostile = HostileSurplus()
    raw = (
        {"text": "one", "start": 0.0, "end": 1.0},
        hostile,
    )
    legacy = legacy_distribute_before_shift(
        raw,
        texts=("one",),
        iso="en",
        origin=2.0,
        identity=False,
        raw_unit_ids=("r0", "r1"),
    )
    assert legacy.block_units[0][0]["start"] == 2.0
    assert legacy.receipt.leftover_unit_ids == ("r1",)
    assert HostileSurplus.deepcopy_calls == 0

    strict = capture_strict_units(raw, call_index=0, raw_unit_ids=("r0", "r1"))
    assert strict.status == "invalid"
    assert strict.failure is not None
    assert strict.failure.call_unit_index == 1
    assert HostileSurplus.deepcopy_calls == 0

    acquisition_source = inspect.getsource(align_acquisition)
    orchestration_source = inspect.getsource(align_orchestration)
    assert "legacy_distribute_before_shift(" in acquisition_source
    assert "copy.deepcopy(list(raw_units))" not in orchestration_source


def test_ctc_production_seam_preserves_pre_and_post_interpolation_values():
    from voxweave import align_ctc
    from voxweave.align_acquisition import capture_strict_units
    from voxweave.align_common import interp_missing

    original = [
        {"text": "a", "start": 0.0, "end": 0.0},
        {"text": "b", "start": 2.0, "end": 3.0},
    ]
    interpolated = interp_missing(original)
    capture = capture_strict_units(
        interpolated,
        original_units=original,
        call_index=0,
        raw_unit_ids=("r0", "r1"),
    )
    assert capture.status == "valid"
    assert capture.units is not None
    assert capture.units[0].provenance == "align-interpolated"
    assert capture.units[0].original_relative_start == 0.0
    assert capture.units[0].original_relative_end == 0.0
    assert capture.units[0].relative_start == 2.0
    assert capture.units[0].relative_end == 2.0
    assert capture.units[1].provenance == "aligner"

    source = inspect.getsource(align_ctc._ctc_align_logp)
    interpolation = source.index("interp_missing")
    pre_interpolation = source[:interpolation]
    assert any(
        marker in pre_interpolation
        for marker in (
            "original_units",
            "pre_interpolation",
            "before_interpolation",
            "raw_before_interp",
        )
    )


def test_qwen_mismatch_is_typed_and_rat5_semantics_are_live():
    from voxweave import align_acquisition, pipeline
    from voxweave.align_acquisition import qwen_sample_geometry
    from voxweave.core import align_compare
    from voxweave.p6_ratifications import QWEN_SELECTED_V2_ENABLED

    geometry = qwen_sample_geometry(
        nominal_start=0.10001,
        nominal_end=1.0,
        sample_rate=16_000,
        sample_count=32_000,
    )
    assert geometry.physical_origin_seconds == int(0.10001 * 16_000) / 16_000
    assert geometry.legacy_origin_seconds == 0.10001
    assert geometry.authority_origin_seconds == geometry.physical_origin_seconds
    assert geometry.legacy_origin_seconds != geometry.authority_origin_seconds

    acquisition_source = inspect.getsource(align_acquisition)
    assert "physical-origin-mismatch" in acquisition_source
    align_blocks_source = inspect.getsource(pipeline._align_blocks)
    if "def observe_sample_geometry" in align_blocks_source:
        observer_source = align_blocks_source.split(
            "def observe_sample_geometry", maxsplit=1
        )[1].split("cwav = slice_wav", maxsplit=1)[0]
        assert "except Exception" not in observer_source
        assert "physical_origin = cs" not in align_blocks_source

    assert QWEN_SELECTED_V2_ENABLED is True
    assert align_compare.semantic_comparison_available() is True
    comparison_source = inspect.getsource(align_compare)
    assert "physical_origin_seconds" in comparison_source
    assert "legacy_origin_seconds" in comparison_source
    assert "qwen-crop" in comparison_source


def test_seal_mismatch_lifecycle_is_called_by_the_production_issuer(tmp_path):
    from voxweave import align_acquisition
    from voxweave.align_acquisition import (
        AcquisitionAdmissionLedger,
        FreshSealBroken,
        raise_distribution_seal_mismatch,
    )
    from voxweave.align_context import (
        consume_context_role,
        issue_align_context,
        retire_live_context_roles,
        role_vector,
    )
    from voxweave.align_snapshot import freeze_json

    context = issue_align_context(
        stable_fields=freeze_json({"input": "seal-mismatch"}),
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )
    consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
    ledger = AcquisitionAdmissionLedger()
    order: list[tuple[str, int]] = []

    with pytest.raises(FreshSealBroken) as error:
        raise_distribution_seal_mismatch(
            context,
            terminal_call_index=0,
            ledger=ledger,
            dispose=lambda: order.append(("dispose", len(ledger.events))),
        )
    assert order == [("dispose", 1)]
    assert len(ledger.events) == 1
    assert ledger.events[0].terminal == "acquisition-failed"
    assert ledger.events[0].payload == (
        "AO-13",
        "fresh-seal-broken",
        "authority-distribution",
        "distribution-seal",
        0,
        (),
    )
    assert error.value.failure.kind == "fresh-seal-broken"
    retire_live_context_roles(context)
    assert role_vector(context) == ("C", "R", "R", "R", "R")

    assert _call_count(align_acquisition, "raise_distribution_seal_mismatch") == 1


def test_ao15_adapter_finishes_before_ao16_evidence_and_ald6():
    from voxweave import align_orchestration

    assert align_orchestration.ALIGN_AO_ORDER == tuple(
        f"AO-{index:02d}" for index in range(1, 26)
    )
    source = inspect.getsource(align_orchestration)
    adapter = source.find("run_locked_align_adapter(")
    evidence = source.find("project_evidence_core(")
    ald6 = source.find("evaluate_ald6(")
    assert adapter >= 0
    assert adapter < evidence < ald6
    assert source.count("project_evidence_core(") == 1
    assert source.count("evaluate_ald6(") == 1


def test_rat1_adds_only_the_fresh_authority_factory_and_adapter_result():
    from voxweave import align_acquisition, align_adapter
    from voxweave.core import authority, finalizer

    assert authority.AUTHORITY_KINDS == (
        "fresh-alignment",
        "optimizer-selection",
        "v1-capture",
    )
    factory = finalizer.phase1_from_fresh_alignment
    assert tuple(inspect.signature(factory).parameters) == (
        "verified",
        "profile",
        "ledger",
        "row_id",
        "evaluation_id",
    )
    assert hasattr(align_adapter, "AlignAdapterResult")
    adapter_fields = tuple(
        field.name for field in dataclasses.fields(align_adapter.AlignAdapterResult)
    )
    assert adapter_fields[:5] == (
        "context_content_digest",
        "receipt_digest",
        "legacy",
        "v2",
        "v2_status",
    )
    assert "evidence_core" not in adapter_fields
    assert "comparison" not in adapter_fields
    with pytest.raises(TypeError):
        align_adapter.AlignAdapterResult()

    assert hasattr(align_adapter, "run_locked_align_adapter")
    adapter_parameters = tuple(
        inspect.signature(align_adapter.run_locked_align_adapter).parameters
    )
    assert adapter_parameters == (
        "context",
        "acquisition",
        "shadow_enabled",
    )
    assert align_acquisition.VerifiedFreshAlignment is not None


def test_authority_profile_is_context_bound_and_rejects_loose_forged_input():
    from voxweave import align_acquisition, align_distribution

    assert (
        "profile"
        not in inspect.signature(
            align_distribution.build_authority_distribution
        ).parameters
    )
    acquisition_source = inspect.getsource(align_acquisition)
    assert "build_authority_distribution(" not in acquisition_source or (
        "profile="
        not in acquisition_source.split("build_authority_distribution(", maxsplit=1)[
            1
        ].split(")", maxsplit=1)[0]
    )

    forged = align_distribution.AuthorityLimitProfile(
        "production",
        align_distribution.CallWorkLimits(
            9_000_000,
            9_000_000,
            9_000_000,
            9_000_000,
        ),
        align_distribution.JobWorkLimits(
            9_000,
            9_000_000,
            9_000_000,
            9_000_000,
            9_000_000,
        ),
        "f" * 64,
    )
    with pytest.raises(
        align_distribution.AuthorityLimitProfileError,
        match="production authority limits|profile",
    ):
        align_distribution.validate_authority_limit_profile(forged)


def test_original_source_indices_survive_delivery_permutation_end_to_end(tmp_path):
    from voxweave.align_acquisition import (
        _bind_fresh_adapter_payload,
        _fresh_alignment_call_observer,
        _fresh_core_inputs,
        _fresh_record,
        _fresh_seed,
        begin_fresh_alignment,
        seal_fresh_alignment,
    )
    from voxweave.align_adapter import (
        AlignDelivery,
        AlignDeliveryCue,
        AlignProjectionInputs,
        PersistedAlignUnit,
        SourceBlockDecoration,
        _adapter_record,
        run_locked_align_adapter,
    )
    from voxweave.align_context import _align_context_stable_fields
    from voxweave.align_evidence_core import build_evidence_core
    from voxweave.align_inputs import (
        LegacyAlignPolicy,
        resolve_align_profile,
        resolve_finalize_evidence,
        validate_v2_policy,
    )
    from voxweave.align_orchestration import issue_public_align_context
    from voxweave.align_snapshot import StrictInputStatus, thaw_json
    from voxweave.episode_transaction import FileGeneration

    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"physical-audio")
    blocks = (
        {"source_index": 7, "text": "first", "alignment_text": "first"},
        {"source_index": 2, "text": "second", "alignment_text": "second"},
    )
    context = issue_public_align_context(
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.wav",
        prepared_audio_path=prepared,
        expected_vtt=FileGeneration(True, b"WEBVTT\n"),
        expected_json=FileGeneration(True, b"{}"),
        expected_vtt_sha256=None,
        media_fingerprint="m" * 64,
        effective_iso="en",
        route_kind="ctc-full",
        blocks=blocks,
        prepared_audio_sha256="a" * 64,
        legacy_policy=LegacyAlignPolicy(0.0, 0.0, 0.0),
        stored_language="en",
        segmentation=None,
        strict_shot_changes=None,
        strict_sing_spans=None,
    )
    stable = thaw_json(_align_context_stable_fields(context))
    assert tuple(row["source_index"] for row in stable["blocks"]) == (7, 2)

    session = begin_fresh_alignment(
        context,
        alignment_texts=("first", "second"),
        source_indices=(7, 2),
        language="en",
        prepared_audio_sample_count=32_000,
    )
    _fresh_alignment_call_observer(session)(
        (
            {"text": "first", "start": 0.0, "end": 1.0},
            {"text": "second", "start": 1.0, "end": 2.0},
        ),
        None,
        (0, 1),
        0.0,
    )
    acquisition = seal_fresh_alignment(session)
    record = _fresh_record(context, acquisition)
    seed = _fresh_seed(context, acquisition)
    assert acquisition.physical_calls[0].source_block_indices == (7, 2)
    assert acquisition.distribution.owner_source_indices == (7, 2)
    assert acquisition.distribution.work.calls[0].source_block_indices == (7, 2)
    assert record.legacy_receipts[0].owner_source_indices == (7, 2)
    assert tuple(block.source_index for block in seed.blocks) == (7, 2)

    first = PersistedAlignUnit("first", 0.0, 1.0)
    second = PersistedAlignUnit("second", 1.0, 2.0)
    legacy = AlignDelivery(
        context.context_content_digest,
        acquisition.receipt_digest,
        "legacy-v1",
        "ctc-full",
        (
            AlignDeliveryCue(7, "first", 0.0, 1.0, None, ("r0",), (first,), 0.0, 1.0),
            AlignDeliveryCue(2, "second", 1.0, 2.0, None, ("r1",), (second,), 1.0, 2.0),
        ),
        (first, second),
    )
    projection = AlignProjectionInputs(
        "en",
        (
            SourceBlockDecoration(7, None, None),
            SourceBlockDecoration(2, None, None),
        ),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    _bind_fresh_adapter_payload(
        context,
        acquisition,
        legacy_delivery=legacy,
        projection_inputs=projection,
        strict_input_status=StrictInputStatus("valid", None),
        v2_policy_status=validate_v2_policy(LegacyAlignPolicy(0.0, 0.0, 0.0)),
        profile_resolution=resolve_align_profile(None, effective_iso="en"),
        evidence_resolution=resolve_finalize_evidence(
            shot_changes=None, sing_spans=None
        ),
    )
    adapter = run_locked_align_adapter(context, acquisition, shadow_enabled=True)
    assert tuple(cue.source_index for cue in adapter.legacy.cues) == (7, 2)
    assert adapter.v2 is not None
    assert tuple(cue.source_index for cue in adapter.v2.cues) == (7, 2)
    assert tuple(
        item.source_index
        for item in _adapter_record(context, adapter).projection_inputs.source_blocks
    ) == (7, 2)

    core_inputs = _fresh_core_inputs(context, acquisition)
    core = build_evidence_core(
        context_content_digest=context.context_content_digest,
        blocks=core_inputs[0],
        captures=core_inputs[1],
        transforms=core_inputs[2],
        distribution=core_inputs[3],
        seed_status=core_inputs[4],
        seed_reasons=core_inputs[5],
        physical_calls=core_inputs[6],
        receipt_digest=acquisition.receipt_digest,
        language="en",
    )
    assert tuple(block.source_index for block in core.blocks) == (7, 2)
    assert core.physical_calls[0].source_block_indices == (7, 2)
