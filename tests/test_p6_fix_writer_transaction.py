import dataclasses
import hashlib
import json
import math

import pytest

from voxweave import pipeline


def _transaction_files(tmp_path):
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    machine_path = tmp_path / "episode.voiceprints.json"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")
    machine_path.write_bytes(b"old machine")
    return json_path, vtt_path, machine_path


def _commit_with_machine_artifact(
    *,
    json_path,
    vtt_path,
    machine_path,
):
    from voxweave import episode_transaction

    return episode_transaction.commit_primary_outputs(
        command="process",
        episode_path=vtt_path,
        json_path=json_path,
        vtt_path=vtt_path,
        expected_json=episode_transaction.capture_file_generation(json_path),
        expected_vtt=episode_transaction.capture_file_generation(vtt_path),
        main_json_bytes=b"new json",
        vtt_bytes=b"new vtt",
        machine_artifact=episode_transaction.MachineArtifactPublication(
            machine_path, b"new machine"
        ),
    )


def test_rat2_public_align_publishes_and_verifies_evidence_last(tmp_path, monkeypatch):
    from tests.test_p6_episode_transactions import _stub_public_shadow_align
    from voxweave import align_evidence, episode_transaction

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    evidence_path = tmp_path / "episode.align-evidence.json"
    evidence_path.write_bytes(b"stale evidence")
    replacements = []
    real_replace = episode_transaction._replace_stage

    def observed_replace(stage):
        replacements.append(stage.target)
        real_replace(stage)

    monkeypatch.setattr(episode_transaction, "_replace_stage", observed_replace)

    assert pipeline.align(vtt_path) == vtt_path
    assert replacements[-3:] == [json_path, vtt_path, evidence_path]
    encoded = evidence_path.read_bytes()
    assert encoded.endswith(b"\n")
    evidence = json.loads(encoded)
    assert evidence["schema_version"] == 8
    assert evidence["kind"] == "fresh-alignment"
    assert (
        evidence["selected_outputs"]["vtt_sha256"]
        == hashlib.sha256(vtt_path.read_bytes()).hexdigest()
    )
    assert (
        evidence["selected_outputs"]["json_sha256"]
        == hashlib.sha256(json_path.read_bytes()).hexdigest()
    )

    verified = align_evidence.verify_align_evidence(vtt_path)
    assert verified.integrity is True
    assert verified.w1_usable is True


@pytest.mark.parametrize(
    "field",
    (
        "legacy_slice_sha256",
        "legacy_absolute_sha256",
        "backend_model_config_sha256",
        "route_input_sha256",
    ),
)
def test_rat2_durable_verifier_recomputes_physical_call_claims(
    field, tmp_path, monkeypatch
):
    from tests.test_p6_episode_transactions import _stub_public_shadow_align
    from voxweave import align_evidence

    _media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    assert pipeline.align(vtt_path) == vtt_path
    evidence_path = tmp_path / "episode.align-evidence.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence["physical_calls"][0][field] = "d" * 64
    evidence["receipt_digest"] = align_evidence._receipt_digest(evidence)
    evidence_path.write_bytes(
        (
            json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    )

    verified = align_evidence.verify_align_evidence(vtt_path)

    assert verified.integrity is False
    assert verified.w1_usable is False
    assert verified.detail_code == "evidence-schema"


def test_rat2_default_source_facts_downgrade_production_w1_audit(
    tmp_path, monkeypatch
):
    from tests.test_p6_episode_transactions import _stub_public_shadow_align
    from voxweave import align_evidence

    _media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    assert pipeline.align(vtt_path) == vtt_path
    evidence_path = tmp_path / "episode.align-evidence.json"
    evidence = json.loads(evidence_path.read_bytes())
    model_facts = {
        "kind": "default",
        "route": evidence["route"],
        "language": evidence["language"],
    }
    route_facts = {
        "kind": "default",
        "context_content_digest": evidence["context_content_digest"],
        "route": evidence["route"],
    }
    evidence["source_facts"] = {
        "backend_model_config": model_facts,
        "route_input": route_facts,
    }
    model_digest = align_evidence._durable_fact_digest(model_facts)
    route_digest = align_evidence._durable_fact_digest(route_facts)
    for call in evidence["physical_calls"]:
        call["backend_model_config_sha256"] = model_digest
        call["route_input_sha256"] = route_digest
    evidence["receipt_digest"] = align_evidence._receipt_digest(evidence)
    evidence_path.write_bytes(
        (
            json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    )

    verified = align_evidence.verify_align_evidence(vtt_path)

    assert verified.integrity is True
    assert verified.w1_usable is False
    assert verified.detail_code is None


def test_rat2_legacy_absolute_digest_is_reprojected_from_relative_retained_units(
    tmp_path, monkeypatch
):
    from tests.test_p6_episode_transactions import _stub_public_shadow_align
    from voxweave.align_snapshot import freeze_json, frozen_json_digest

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    source = json.loads(json_path.read_bytes())
    for unit in source["word_segments"]:
        unit["start"] += 0.5
        unit["end"] += 0.5
    json_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    assert pipeline.align(vtt_path) == vtt_path
    evidence = json.loads((tmp_path / "episode.align-evidence.json").read_bytes())
    call = evidence["physical_calls"][0]
    assert call["legacy_origin_kind"] == "nominal-route"
    assert call["legacy_origin_seconds"] != 0.0
    relative = call["legacy_retained_units"]
    absolute = [
        [
            {
                "text": unit["text"],
                "start": unit["start"] + call["legacy_origin_seconds"],
                "end": unit["end"] + call["legacy_origin_seconds"],
            }
            for unit in owner
        ]
        for owner in relative
    ]
    assert frozen_json_digest(freeze_json(absolute)) == call["legacy_absolute_sha256"]
    assert frozen_json_digest(freeze_json(relative)) != call["legacy_absolute_sha256"]


@pytest.mark.parametrize(
    (
        "profile_kind",
        "admission",
        "owner_ranges",
        "raw_unit_count",
        "expected",
    ),
    (
        ("production", "valid", (("r0", "r1"), ("r2",)), 3, True),
        ("test-only", "valid", (("r0", "r1"), ("r2",)), 3, False),
        ("production", "invalid", (("r0", "r1"), ("r2",)), 3, False),
        ("production", "valid", None, 3, False),
        ("production", "valid", (("r0", "r1"), ()), 2, False),
        ("production", "valid", (("r0", "r1"), ("r1", "r2")), 3, False),
        ("production", "valid", (("r0",), ("r1",)), 3, False),
        ("production", "valid", (("r0", "r1"), ("r2",)), 2, False),
    ),
)
def test_rat2_w1_usability_is_the_exact_section_9_3_conjunction(
    profile_kind,
    admission,
    owner_ranges,
    raw_unit_count,
    expected,
):
    from voxweave.align_evidence import _w1_usable_audit

    evidence = {
        "input_history": {"authority_limit_profile_kind": profile_kind},
        "authority_distribution": {
            "owner_unit_ids": owner_ranges,
            "work": {"limit_profile_kind": profile_kind},
        },
        "v2_admission_status": admission,
        "raw_unit_count": raw_unit_count,
    }

    assert _w1_usable_audit(evidence) is expected


def test_rat3_segmentation_writer_preserves_present_null(tmp_path):
    from tests.test_p6_segmentation_candidates import _issued
    from voxweave.align_snapshot import RawJSONCarrier, freeze_json
    from voxweave.reference_projector import reference_segmentation_projection
    from voxweave.segmentation_adapter import SegmentationProjectionInputs
    from voxweave.segmentation_projector import project_segmentation_delivery

    _context, issued = _issued(tmp_path)
    carriers = dataclasses.replace(
        issued.delivery.carriers,
        speaker_turns=RawJSONCarrier(True, freeze_json(None)),
    )
    delivery = dataclasses.replace(issued.delivery, carriers=carriers)
    inputs = SegmentationProjectionInputs(
        timestamps=True,
        speaker_names=(("S0", "Alice"),),
    )

    producer = project_segmentation_delivery(delivery, inputs, strict=False)
    reference = reference_segmentation_projection(delivery, inputs, strict=False)
    assert producer.main_json_bytes == reference.main_json_bytes
    value = json.loads(producer.main_json_bytes)
    assert "speaker_turns" in value
    assert value["speaker_turns"] is None


def test_rat3_segmentation_writer_preserves_reversed_and_nonfinite_turns(tmp_path):
    from tests.test_p6_segmentation_candidates import _issued
    from voxweave.align_snapshot import RawJSONCarrier, freeze_json
    from voxweave.reference_projector import reference_segmentation_projection
    from voxweave.segmentation_adapter import SegmentationProjectionInputs
    from voxweave.segmentation_projector import project_segmentation_delivery

    raw_turns = [
        [2.0, 1.5, "reversed"],
        [float("inf"), float("nan"), "nonfinite"],
        [-0.0, 0.0, "signed-zero"],
    ]
    _context, issued = _issued(tmp_path)
    carriers = dataclasses.replace(
        issued.delivery.carriers,
        speaker_turns=RawJSONCarrier(True, freeze_json(raw_turns)),
    )
    delivery = dataclasses.replace(issued.delivery, carriers=carriers)
    inputs = SegmentationProjectionInputs(timestamps=True, speaker_names=())

    producer = project_segmentation_delivery(delivery, inputs, strict=False)
    reference = reference_segmentation_projection(delivery, inputs, strict=False)
    assert producer.main_json_bytes == reference.main_json_bytes
    turns = json.loads(producer.main_json_bytes)["speaker_turns"]
    assert turns[0] == [2.0, 1.5, "reversed"]
    assert math.isinf(turns[1][0])
    assert math.isnan(turns[1][1])
    assert math.copysign(1.0, turns[2][0]) == -1.0


def test_rat3_segmentation_writer_preserves_nested_duplicate_key_order(tmp_path):
    from tests.test_p6_segmentation_candidates import _issued
    from voxweave.align_snapshot import decode_sibling_json_snapshot
    from voxweave.reference_projector import reference_segmentation_projection
    from voxweave.segmentation_adapter import SegmentationProjectionInputs
    from voxweave.segmentation_projector import project_segmentation_delivery

    snapshot = decode_sibling_json_snapshot(
        "episode.json",
        b'{"speaker_turns":{"outer":{"dup":1,"dup":2,"tail":"x"},"after":3}}',
    )
    _context, issued = _issued(tmp_path)
    carriers = dataclasses.replace(
        issued.delivery.carriers,
        speaker_turns=snapshot.carrier("speaker_turns"),
    )
    delivery = dataclasses.replace(issued.delivery, carriers=carriers)
    inputs = SegmentationProjectionInputs(timestamps=True, speaker_names=())

    producer = project_segmentation_delivery(delivery, inputs, strict=False)
    reference = reference_segmentation_projection(delivery, inputs, strict=False)
    exact_turns_bytes = (
        b'  "speaker_turns": {\n'
        b'    "outer": {\n'
        b'      "dup": 1,\n'
        b'      "dup": 2,\n'
        b'      "tail": "x"\n'
        b"    },\n"
        b'    "after": 3\n'
        b"  },\n"
    )
    assert exact_turns_bytes in producer.main_json_bytes
    assert exact_turns_bytes in reference.main_json_bytes
    assert producer.main_json_bytes == reference.main_json_bytes


def test_rat3_align_writer_consumes_raw_carrier_and_preserves_null(tmp_path):
    from tests.test_p6_align_candidates import _evaluated
    from voxweave.align_adapter import AlignProjectionInputs, SourceBlockDecoration
    from voxweave.align_projector import project_align_delivery
    from voxweave.align_snapshot import RawJSONCarrier, freeze_json
    from voxweave.reference_projector import reference_align_projection

    _context, result = _evaluated(tmp_path)
    inputs = AlignProjectionInputs(
        language="en",
        source_blocks=(SourceBlockDecoration(0, "Alice", None),),
        vad_speech=(),
        shot_changes=None,
        sing_spans=None,
        speaker_turns=RawJSONCarrier(True, freeze_json(None)),  # type: ignore[arg-type]
        voiceprint_capture=None,
        voiceprint_media=None,
        segmentation=None,
    )

    producer = project_align_delivery(result.legacy, inputs, strict=False)
    reference = reference_align_projection(result.legacy, inputs, strict=False)
    assert producer.main_json_bytes == reference.main_json_bytes
    value = json.loads(producer.main_json_bytes)
    assert "speaker_turns" in value
    assert value["speaker_turns"] is None


def test_rat7_split_mapping_change_stale_aborts_before_mutation(tmp_path, monkeypatch):
    from tests.test_p6_episode_transactions import _units
    from voxweave import episode_transaction

    json_path = tmp_path / "episode.json"
    original_json = json.dumps(
        {
            "language": "en",
            "word_segments": _units(),
            "speaker_turns": [[0.0, 1.0, "S0"]],
        }
    ).encode()
    json_path.write_bytes(original_json)
    vtt_path = tmp_path / "episode.vtt"
    mapping_path = tmp_path / "episode.speakers.json"
    mapping_path.write_text('{"version":1,"speakers":{"S0":"Alice"}}', encoding="utf-8")
    real_segment = pipeline.segment_document

    def change_mapping(**kwargs):
        result = real_segment(**kwargs)
        mapping_path.write_text(
            '{"version":1,"speakers":{"S0":"Bob"}}', encoding="utf-8"
        )
        return result

    monkeypatch.setattr(pipeline, "segment_document", change_mapping)

    with pytest.raises(episode_transaction.InputStaleError) as caught:
        pipeline.split(json_path)
    assert caught.value.failure.kind == "input-stale"
    assert caught.value.failure.phase == "recheck"
    assert caught.value.failure.detail_code == "speaker-mapping-generation"
    assert caught.value.landed == ()
    assert json_path.read_bytes() == original_json
    assert not vtt_path.exists()
    assert json.loads(mapping_path.read_bytes())["speakers"]["S0"] == "Bob"


def test_machine_artifact_failure_details_are_registered_by_amendment_one():
    from voxweave.align_failures import CanonicalFailure, OUTCOME_DETAILS

    assert "machine-artifact-stage" in OUTCOME_DETAILS["stage-failed"]
    assert "machine-artifact-replace" in OUTCOME_DETAILS["commit-failed"]
    assert (
        CanonicalFailure("stage-failed", "stage", "machine-artifact-stage").detail_code
        == "machine-artifact-stage"
    )
    assert (
        CanonicalFailure(
            "commit-failed", "commit", "machine-artifact-replace"
        ).detail_code
        == "machine-artifact-replace"
    )


def test_machine_artifact_stage_failure_is_canonical_and_mutates_nothing(
    tmp_path, monkeypatch
):
    from voxweave import episode_transaction

    json_path, vtt_path, machine_path = _transaction_files(tmp_path)
    real_stage = episode_transaction._stage_bytes

    def fail_machine_stage(target, value):
        if target == machine_path:
            raise OSError("machine stage disk full")
        return real_stage(target, value)

    monkeypatch.setattr(episode_transaction, "_stage_bytes", fail_machine_stage)

    with pytest.raises(episode_transaction.TransactionOperationError) as caught:
        _commit_with_machine_artifact(
            json_path=json_path,
            vtt_path=vtt_path,
            machine_path=machine_path,
        )
    assert caught.value.failure.kind == "stage-failed"
    assert caught.value.failure.phase == "stage"
    assert caught.value.failure.detail_code == "machine-artifact-stage"
    assert caught.value.landed == ()
    assert caught.value.machine_landed == ()
    assert caught.value.leftovers == ()
    assert json_path.read_bytes() == b"old json"
    assert vtt_path.read_bytes() == b"old vtt"
    assert machine_path.read_bytes() == b"old machine"
    assert not tuple(tmp_path.glob("*.part*"))


def test_machine_artifact_replace_failure_is_canonical_after_primaries_land(
    tmp_path, monkeypatch
):
    from voxweave import episode_transaction

    json_path, vtt_path, machine_path = _transaction_files(tmp_path)
    real_replace = episode_transaction._replace_stage

    def fail_machine_replace(stage):
        if stage.target == machine_path:
            raise OSError("machine replace failed")
        return real_replace(stage)

    monkeypatch.setattr(episode_transaction, "_replace_stage", fail_machine_replace)

    with pytest.raises(episode_transaction.TransactionOperationError) as caught:
        _commit_with_machine_artifact(
            json_path=json_path,
            vtt_path=vtt_path,
            machine_path=machine_path,
        )
    assert caught.value.failure.kind == "commit-failed"
    assert caught.value.failure.phase == "commit"
    assert caught.value.failure.detail_code == "machine-artifact-replace"
    assert caught.value.landed == (json_path, vtt_path)
    assert caught.value.machine_landed == ()
    assert caught.value.leftovers == ()
    assert json_path.read_bytes() == b"new json"
    assert vtt_path.read_bytes() == b"new vtt"
    assert machine_path.read_bytes() == b"old machine"
    assert not tuple(tmp_path.glob("*.part*"))
