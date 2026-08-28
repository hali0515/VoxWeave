"""Whole-program regressions for the P5 close fix wave."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.test_calib_shadow import calib, cc
from tests.test_shadow_hook import _case_speakers, _segment
from voxweave import backend, pipeline


def _live_speaker_artifact(monkeypatch) -> dict[str, Any]:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(_case_speakers()).shadow
    assert isinstance(artifact, dict) and artifact["schema_version"] == 2
    return artifact


def _mutate_overlap(artifact: dict[str, Any]) -> None:
    cues = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v2"]["cues"]
    assert len(cues) >= 2
    cues[0]["end"] = float(cues[1]["start"]) + 0.01


def _mutate_partition(artifact: dict[str, Any]) -> None:
    row = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v2"]
    assert row["partition"] and row["partition"][0] > 1
    row["partition"][0] -= 1


def _mutate_trace_and_movement(artifact: dict[str, Any]) -> None:
    block = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v2"]["finalizer"]
    assert block["trace"]["legs"]
    assert block["movement"]
    block["trace"] = {}
    block["movement"] = []


SCHEMA_VALUE_MUTATIONS = (
    pytest.param(_mutate_overlap, "overlap", id="delivered-overlap"),
    pytest.param(_mutate_partition, "partition", id="partition-detached"),
    pytest.param(_mutate_trace_and_movement, "trace", id="trace-movement-deleted"),
)


@pytest.mark.parametrize(("mutate", "needle"), SCHEMA_VALUE_MUTATIONS)
def test_schema_two_rejects_post_construction_value_mutations(
    monkeypatch, mutate, needle
) -> None:
    from voxweave.core.shadow_schema import validate_shadow_v2_payload

    artifact = copy.deepcopy(_live_speaker_artifact(monkeypatch))
    mutate(artifact)

    errors = validate_shadow_v2_payload(artifact)
    assert errors
    assert any(needle in error.lower() for error in errors), errors

    measurement_errors = calib.shadow_measurement_errors(
        cast(Any, SimpleNamespace(id="schema-value-mutation")), artifact, {}
    )
    assert measurement_errors
    assert calib.shadow_exit_code([], [], [], [], measurement_errors) == cc.EXIT_INVALID


@pytest.mark.parametrize(("mutate", "_needle"), SCHEMA_VALUE_MUTATIONS)
def test_live_post_assembly_admission_rejects_value_mutations(
    monkeypatch, mutate, _needle
) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    real = pipeline._shadow_v2_artifact

    def corrupted(*args, **kwargs):
        artifact = real(*args, **kwargs)
        mutate(artifact)
        return artifact

    monkeypatch.setattr(pipeline, "_shadow_v2_artifact", corrupted)
    artifact = _segment(_case_speakers()).shadow
    assert isinstance(artifact, dict)
    assert artifact["schema_version"] == 1
    assert artifact["kind"] == "segmentation-shadow-error"
    assert "invalid live shadow schema 2" in artifact["error"]["detail"]


@pytest.mark.parametrize(("mutate", "_needle"), SCHEMA_VALUE_MUTATIONS)
def test_shadow_cli_exits_two_for_post_construction_value_mutations(
    monkeypatch, mutate, _needle
) -> None:
    real = pipeline._shadow_v2_artifact

    def corrupted(*args, **kwargs):
        artifact = real(*args, **kwargs)
        mutate(artifact)
        return artifact

    monkeypatch.setattr(pipeline, "_shadow_v2_artifact", corrupted)
    argv = [
        "shadow",
        "--corpus",
        str(calib.DEFAULT_CORPUS),
        "--case",
        "en-01",
        "--no-ablation",
        "--check",
    ]
    with pytest.raises(SystemExit) as raised:
        cc.run_cli(lambda: calib.main(argv))
    assert raised.value.code == cc.EXIT_INVALID


TURN_FIXTURES = (
    pytest.param([[2.0, 1.0, "A"]], id="reversed"),
    pytest.param([[3.0, 3.0, "B"]], id="point"),
    pytest.param(
        [[float("inf"), float("nan"), "C"]],
        id="non-finite",
    ),
)


def _split_replay_bytes(
    root: Path, monkeypatch, turns: list[list[Any]], *, shadow: bool
) -> bytes:
    root.mkdir()
    json_path = root / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "en",
                "segments": [],
                "speaker_turns": turns,
                "word_segments": [{"text": "word", "start": 0.0, "end": 0.5}],
            }
        ),
        encoding="utf-8",
    )

    def detached_segment(**kwargs):
        units = [dict(unit) for unit in kwargs["word_segments"]]
        return SimpleNamespace(
            language="en",
            units=units,
            cues=[
                {
                    "text": "word",
                    "start": 0.0,
                    "end": 0.5,
                    "word_data": units,
                }
            ],
            manifest={},
        )

    monkeypatch.setattr(pipeline, "segment_document", detached_segment)
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1" if shadow else "0")
    pipeline.split(json_path)
    return json_path.read_bytes()


def _align_replay_bytes(
    root: Path, monkeypatch, turns: list[list[Any]], *, shadow: bool
) -> bytes:
    root.mkdir()
    media = root / "episode.wav"
    media.write_bytes(b"audio")
    json_path = root / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "zh",
                "segments": [],
                "speaker_turns": turns,
                "word_segments": [{"text": "a", "start": 0.0, "end": 0.5}],
            }
        ),
        encoding="utf-8",
    )
    vtt = root / "episode.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:00.500\na\n", encoding="utf-8")
    prepared = root / "prepared.wav"
    prepared.write_bytes(b"prepared")
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: prepared
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda *_args, **_kwargs: [{"text": "a", "start": 0.0, "end": 0.5}],
    )
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1" if shadow else "0")
    pipeline.align(vtt, separate=False)
    return json_path.read_bytes()


@pytest.mark.parametrize("turns", TURN_FIXTURES)
@pytest.mark.parametrize("replay", [_split_replay_bytes, _align_replay_bytes])
def test_replay_preserves_pre_p5_speaker_turn_bytes_with_shadow_off_and_on(
    tmp_path, monkeypatch, turns, replay
) -> None:
    off = replay(tmp_path / "off", monkeypatch, turns, shadow=False)
    on = replay(tmp_path / "on", monkeypatch, turns, shadow=True)
    assert on == off
    persisted = json.loads(off)["speaker_turns"]
    assert len(persisted) == len(turns)
    for actual, expected in zip(persisted, turns, strict=True):
        assert actual[2] == expected[2]
        for actual_bound, expected_bound in zip(actual[:2], expected[:2], strict=True):
            if isinstance(expected_bound, float) and math.isnan(expected_bound):
                assert math.isnan(actual_bound)
            else:
                assert actual_bound == expected_bound


def test_document_node_mapping_is_not_rebuilt_per_edge(monkeypatch) -> None:
    from tests.test_boundary_v2 import document, timed
    from voxweave.core import boundary_v2

    doc = document(timed(["word"] * 120, dur=0.01, gap=0.0))
    calls = 0
    real = boundary_v2._document_nodes

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(boundary_v2, "_document_nodes", counted)
    reuse = boundary_v2._optimization_reuse(doc)
    first = boundary_v2.optimize_document(doc, _reuse=reuse)
    second = boundary_v2.optimize_document(doc, _reuse=reuse)
    assert first.lattice is not None
    assert second.artifact == first.artifact
    assert calls <= len(first.lattice.lattices)


def test_document_node_mapping_scale_ratio_is_subquadratic() -> None:
    from tests.test_boundary_v2 import document, timed
    from voxweave.core.boundary_v2 import optimize_document

    elapsed: list[float] = []
    for size in (250, 500, 1000):
        doc = document(timed(["word"] * size, dur=0.005, gap=0.0))
        started = time.perf_counter()
        optimize_document(doc)
        elapsed.append(time.perf_counter() - started)
    # Use the full 250 -> 1,000 curve so one loaded-CI sample cannot dominate.
    # The removed per-edge map measured about 13x across these points; a linear
    # seam has ample room below this deliberately generous ceiling.
    ratios = (
        elapsed[1] / max(elapsed[0], 1e-9),
        elapsed[2] / max(elapsed[1], 1e-9),
    )
    assert ratios[0] * ratios[1] < 8.0, (elapsed, ratios)


def test_n14_rejects_impossible_work_and_organic_fd9(monkeypatch) -> None:
    artifact = copy.deepcopy(_live_speaker_artifact(monkeypatch))
    artifact["totals"]["canonical_chars"] = 10**18
    row = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v2"]
    report = {
        "cue_index": 0,
        "evidence": {"scans": 4},
        "kind": "stutter-not-proven-fixed-within-4-scans",
    }
    row["finalizer"]["entries"].append(report)
    row["finalizer"]["refusals"].append(copy.deepcopy(report))
    row["finalizer"]["deltas_fired"].append("FD-9")

    evidence = calib.n14_artifact_evidence(
        artifact, case_id="mutated", corpus="tracked"
    )
    assert evidence["failures"]
    assert evidence["unknown"] == []
    assert calib.n14_exit_code([evidence]) == cc.EXIT_GATE_FAILED


def test_n14_cli_rejects_live_impossible_work_and_organic_fd9(
    tmp_path, monkeypatch
) -> None:
    real = calib.shadow_artifact_of

    def mutated(case, result):
        artifact = real(case, result)
        artifact["totals"]["canonical_chars"] = 10**18
        row = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v2"]
        report = {
            "cue_index": 0,
            "evidence": {"scans": 4},
            "kind": "stutter-not-proven-fixed-within-4-scans",
        }
        row["finalizer"]["entries"].append(report)
        row["finalizer"]["refusals"].append(copy.deepcopy(report))
        row["finalizer"]["deltas_fired"].append("FD-9")
        return artifact

    monkeypatch.setattr(calib, "shadow_artifact_of", mutated)
    output = tmp_path / "mutated-shadow.json"
    code = calib.main(
        [
            "shadow",
            "--corpus",
            str(calib.DEFAULT_CORPUS),
            "--case",
            "en-01",
            "--json-out",
            str(output),
            "--no-ablation",
            "--check",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == cc.EXIT_GATE_FAILED
    assert any("canonical_chars" in item for item in report["n14"]["failures"])
    assert any("organic FD-9" in item for item in report["n14"]["failures"])


def test_n14_missing_oracle_evidence_is_invalid() -> None:
    assert calib.n14_exit_code([]) == cc.EXIT_INVALID
    missing = {
        "case": "missing",
        "corpus": "tracked",
        "failures": [],
        "oracle": {"checked": 0},
        "unknown": ["no oracle candidates"],
    }
    assert calib.n14_exit_code([missing]) == cc.EXIT_INVALID


def test_authorized_deferral_allowlist_is_exact_and_unknown_stops_are_invalid() -> None:
    expected = {
        "N20/en-03/v1",
        "N3b-expressed-rate",
        "PD-SUBUNIT/coarse-width-ja",
        "PD-SUBUNIT/coarse-both-ja",
        "PD-SUBUNIT/coarse-per-char-yue",
        "PD-SUBUNIT/coarse-mixed-en",
    }
    assert calib.load_authorized_deferrals() == expected
    authorized, unknown = calib.adjudicate_deferrals(
        [{"id": item, "detail": f"blocked: {item}"} for item in sorted(expected)]
    )
    assert {item["id"] for item in authorized} == expected
    assert all(item["status"] == "authorized_deferred" for item in authorized)
    assert unknown == []
    assert calib.shadow_exit_code([], [], [], [], [], unknown) == cc.EXIT_OK

    _authorized, unknown = calib.adjudicate_deferrals(
        [{"id": "unexpected-stop", "detail": "not authorized"}]
    )
    assert unknown
    assert calib.shadow_exit_code([], [], [], [], [], unknown) == cc.EXIT_INVALID


def test_shadow_normalization_is_detached_from_production_turns(monkeypatch) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    raw = [(2.0, 1.0, "A"), (3.0, 3.0, "B"), (math.inf, 4.0, "C")]
    assert pipeline._turns_in(raw) == raw
