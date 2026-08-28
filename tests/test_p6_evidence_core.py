import dataclasses
import inspect

import pytest


def test_evidence_core_independently_recomputes_surface_reasons_and_digest(tmp_path):
    from tests.test_p6_ald6_independence import _issued_facts
    from voxweave.align_evidence_core import project_evidence_core

    _producer, reference = _issued_facts(tmp_path)
    core = project_evidence_core(reference)
    assert core.schema_version == 8
    assert core.context_content_digest == reference.claimed_context_content_digest
    assert core.raw_unit_count == 1
    assert core.authority_reasons == ()
    assert core.call_surface_chars == (8,)
    assert core.blocks[0].authority_unit_ids == ("r0",)
    assert core.blocks[0].speech_start == 0.2
    assert core.blocks[0].speech_end == 0.8
    assert len(core.core_digest) == 64


def test_evidence_core_rejects_producer_reason_or_surface_scalar_corruption(tmp_path):
    from tests.test_p6_ald6_independence import _issued_facts
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        project_evidence_core,
    )

    _producer, reference = _issued_facts(tmp_path)
    distribution = reference.distribution
    bad_reasons = dataclasses.replace(distribution, reasons=("route-owner-mismatch",))
    with pytest.raises(EvidenceCoreProjectionError):
        project_evidence_core(dataclasses.replace(reference, distribution=bad_reasons))

    bad_row = dataclasses.replace(distribution.work.calls[0], surface_chars=7)
    bad_work = dataclasses.replace(distribution.work, calls=(bad_row,))
    bad_distribution = dataclasses.replace(distribution, work=bad_work)
    with pytest.raises(EvidenceCoreProjectionError):
        project_evidence_core(
            dataclasses.replace(reference, distribution=bad_distribution)
        )


def test_ald6_is_always_triggered_and_exact(tmp_path):
    from tests.test_p6_ald6_independence import _issued_facts
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        evaluate_ald6,
        project_evidence_core,
    )

    _producer, reference = _issued_facts(tmp_path)
    core = project_evidence_core(reference)
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
