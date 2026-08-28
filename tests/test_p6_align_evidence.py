import json

import pytest

from tests.test_p6_align_candidates import _evaluated


def _verified(tmp_path):
    from voxweave.candidate_encoder import (
        encode_align_candidates,
        select_align_candidate,
        verify_selected_align_projection,
    )

    context, result = _evaluated(tmp_path)
    candidate = select_align_candidate(
        context, encode_align_candidates(context, result)
    )
    return context, result, verify_selected_align_projection(context, result, candidate)


def test_final_evidence_binds_only_after_verified_primary_hashes(tmp_path):
    from voxweave.align_context import role_vector
    from voxweave.align_evidence import bind_align_evidence, encode_align_evidence

    context, result, verified = _verified(tmp_path)
    evidence = bind_align_evidence(
        context,
        result.evidence_core,
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

    context, result, verified = _verified(tmp_path)
    with pytest.raises(EvidenceBindingError) as error:
        bind_align_evidence(
            context,
            result.evidence_core,
            engine_family=verified.engine_family,
            vtt_sha256="0" * 64,
            main_json_sha256=verified.main_json_sha256,
        )
    assert error.value.failure.detail_code == "evidence-binding"


def test_rat2_exposes_only_the_path_bound_durable_verifier():
    import inspect

    import voxweave.align_evidence as module

    assert not hasattr(module, "verify_durable_align_evidence")
    assert tuple(inspect.signature(module.verify_align_evidence).parameters) == (
        "vtt_path",
        "explicit_media_path",
        "corpus_root",
    )
