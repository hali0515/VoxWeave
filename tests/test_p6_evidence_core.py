import dataclasses
import inspect

import pytest


def _facts():
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
        capture, physical_origin_seconds=100.0, identity=False
    )
    distribution = build_authority_distribution(
        blocks=blocks,
        delivery_route=(RouteExpectation(0, 0, "call", 0),),
        calls=(AuthorityCallInput(0, (0,), (0, 1), ("r0",), ("word",), "valid", None),),
        skipped_blocks=(),
        route_claims=(RouteClaim("call", 0, 0, 0),),
        iso="en",
    )
    return blocks, capture, transform, distribution


def test_evidence_core_independently_recomputes_surface_reasons_and_digest():
    from voxweave.align_evidence_core import project_evidence_core

    blocks, capture, transform, distribution = _facts()
    core = project_evidence_core(
        context_content_digest="a" * 64,
        blocks=blocks,
        captures=(capture,),
        transforms=(transform,),
        distribution=distribution,
        seed_status="valid",
        seed_reasons=(),
    )
    assert core.schema_version == 8
    assert core.context_content_digest == "a" * 64
    assert core.raw_unit_count == 1
    assert core.authority_reasons == ()
    assert core.call_surface_chars == (8,)
    assert core.blocks[0].authority_unit_ids == ("r0",)
    assert core.blocks[0].speech_start == 100.2
    assert core.blocks[0].speech_end == 100.8
    assert len(core.core_digest) == 64


def test_evidence_core_rejects_producer_reason_or_surface_scalar_corruption():
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        project_evidence_core,
    )

    blocks, capture, transform, distribution = _facts()
    bad_reasons = dataclasses.replace(distribution, reasons=("route-owner-mismatch",))
    with pytest.raises(EvidenceCoreProjectionError):
        project_evidence_core(
            context_content_digest="a" * 64,
            blocks=blocks,
            captures=(capture,),
            transforms=(transform,),
            distribution=bad_reasons,
            seed_status="valid",
            seed_reasons=(),
        )

    bad_row = dataclasses.replace(distribution.work.calls[0], surface_chars=7)
    bad_work = dataclasses.replace(distribution.work, calls=(bad_row,))
    bad_distribution = dataclasses.replace(distribution, work=bad_work)
    with pytest.raises(EvidenceCoreProjectionError):
        project_evidence_core(
            context_content_digest="a" * 64,
            blocks=blocks,
            captures=(capture,),
            transforms=(transform,),
            distribution=bad_distribution,
            seed_status="valid",
            seed_reasons=(),
        )


def test_ald6_is_always_triggered_and_exact():
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        evaluate_ald6,
        project_evidence_core,
    )

    blocks, capture, transform, distribution = _facts()
    core = project_evidence_core(
        context_content_digest="a" * 64,
        blocks=blocks,
        captures=(capture,),
        transforms=(transform,),
        distribution=distribution,
        seed_status="valid",
        seed_reasons=(),
    )
    with pytest.raises(EvidenceCoreProjectionError, match="independent"):
        evaluate_ald6(core, core)
    independent = dataclasses.replace(core)
    assert independent is not core
    assert evaluate_ald6(core, independent).triggered is True
    assert evaluate_ald6(core, independent).passed is True
    changed = dataclasses.replace(core, context_content_digest="b" * 64)
    outcome = evaluate_ald6(changed, core)
    assert outcome.triggered is True and outcome.passed is False


def test_evidence_core_has_no_w1_semantic_producer_or_renderer_imports():
    import voxweave.align_evidence_core as module

    source = inspect.getsource(module)
    for forbidden in (
        "core.finalizer",
        "core.align_compare",
        "core.align_seed",
        "align_adapter",
        "candidate_encoder",
        "pipeline",
        "align_shadow",
    ):
        assert forbidden not in source
