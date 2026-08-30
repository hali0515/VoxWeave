from __future__ import annotations

import json


def test_rat2_public_evidence_uses_the_closed_v10_schema(tmp_path, monkeypatch):
    from tests.test_p6_episode_transactions import _stub_public_shadow_align
    from voxweave import artifacts, pipeline

    media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)

    assert pipeline.align(vtt_path) == vtt_path
    evidence = json.loads(
        artifacts.claim_paths(media).align_evidence(vtt_path).read_bytes()
    )

    assert tuple(evidence) == (
        "schema_version",
        "kind",
        "context_content_digest",
        "receipt_digest",
        "language",
        "route",
        "source_facts",
        "input_history",
        "route_plan",
        "physical_calls",
        "legacy_distribution",
        "authority_distribution",
        "blocks",
        "raw_unit_count",
        "strict_input_status",
        "seed_status",
        "v2_policy_status",
        "profile_status",
        "evidence_status",
        "v2_admission_status",
        "selected_outputs",
    )
    assert tuple(evidence["input_history"]) == (
        "vtt_present",
        "vtt_size",
        "vtt_sha256",
        "sibling_json_present",
        "sibling_json_size",
        "sibling_json_sha256",
        "block_content_sha256",
        "registry_family",
        "media_fingerprint",
        "media_logical_id",
        "media_display_name",
        "prepared_audio_size",
        "prepared_audio_sha256",
        "profile_input_sha256",
        "evidence_carriers_sha256",
        "authority_limit_profile_kind",
        "authority_limit_profile_digest",
        "legacy_policy_binary64",
        "target_logical_id",
        "expected_vtt_sha256",
    )
    assert evidence["input_history"]["media_logical_id"] == "sibling:.wav"
    assert tuple(evidence["source_facts"]) == (
        "backend_model_config",
        "route_input",
    )
    assert tuple(evidence["source_facts"]["backend_model_config"]) == (
        "route",
        "language",
        "backend",
        "model",
        "sample_rate",
    )
    assert tuple(evidence["source_facts"]["route_input"]) == (
        "route",
        "language",
        "blocks",
        "crops",
    )
    assert tuple(evidence["route_plan"]) == ("digest", "entries")
    assert tuple(evidence["physical_calls"][0]) == (
        "call_index",
        "source_block_indices",
        "sample_start",
        "sample_end",
        "sample_rate",
        "physical_origin_seconds",
        "legacy_origin_seconds",
        "legacy_origin_kind",
        "authority_origin_seconds",
        "backend_model_config_sha256",
        "route_input_sha256",
        "strict_unit_status",
        "strict_failure",
        "raw_units_sha256",
        "relative_units_sha256",
        "legacy_retained_units",
        "legacy_slice_sha256",
        "legacy_absolute_sha256",
        "authority_transform_status",
        "authority_absolute_sha256",
        "raw_unit_ids",
    )
    assert tuple(evidence["legacy_distribution"]) == ("digest", "calls")
    assert tuple(evidence["physical_calls"][0]["legacy_retained_units"][0][0]) == (
        "text",
        "start",
        "end",
    )
    assert tuple(evidence["authority_distribution"])[0:3] == (
        "status",
        "digest",
        "owner_source_indices",
    )
    assert "legacy_unit_ids" in evidence["blocks"][0]
    assert evidence["v2_admission_status"] == "valid"


def test_rat2_verifier_rejects_uncovered_internal_mutation(tmp_path, monkeypatch):
    from tests.test_p6_episode_transactions import _stub_public_shadow_align
    from voxweave import align_evidence, artifacts, pipeline

    media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    assert pipeline.align(vtt_path) == vtt_path

    path = artifacts.claim_paths(media).align_evidence(vtt_path)
    value = json.loads(path.read_bytes())
    value["route_plan"]["entries"][0]["source_index"] += 1
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    verification = align_evidence.verify_align_evidence(vtt_path)
    assert verification.integrity is False
    assert verification.w1_usable is False
    assert verification.detail_code == "evidence-schema"
