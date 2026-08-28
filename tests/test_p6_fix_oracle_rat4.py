from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_ROOT = REPO_ROOT / "calibration" / "p6-oracle"
ORACLE_MANIFEST = ORACLE_ROOT / "manifest.json"
ORACLE_SCHEMA = REPO_ROOT / "calibration" / "schemas" / "p6-oracle.schema.json"
ORACLE_RUNNER = REPO_ROOT / "scripts" / "p6_oracle.py"

REFERENCE_SETS = (
    "6e6033f",
    "post-p11",
    "p6-lexical",
    "selected-v2-align",
    "selected-v2-segmentation",
)

NAMED_GATES = {
    "G-CONTEXT",
    "G-TIME-ORDER",
    "G-QWEN-ORIGIN",
    "G-QWEN-WINDOW",
    "G-PROVENANCE",
    "G-ORDER",
    "G-LEGACY-DISTRIBUTION",
    "G-ALIGN-AO",
    "G-AUTHORITY-DISTRIBUTION",
    "G-FROZEN-JSON",
    "G-POLICY",
    "G-EMPTY/FOOTPRINT",
    "G-MEDIA-TRUST",
    "G-PROCESS-SOURCE",
    "G-SPEAKER-MAP-P11",
    "G-SPEAKER-MAP-CAS",
    "G-COMMAND-ORDER",
    "G-SDH-SELECTED",
}

AO_PHASES = tuple(f"AO-{index:02d}" for index in range(1, 26))


class _PhysicalCallReached(RuntimeError):
    pass


def _vtt(*cues: tuple[str, str, str]) -> bytes:
    rows = ["WEBVTT", ""]
    for start, end, text in cues:
        rows.extend((f"{start} --> {end}", text, ""))
    return "\n".join(rows).encode("utf-8")


def _write_episode(
    tmp_path: Path,
    *,
    language: str,
    cues: tuple[tuple[str, str, str], ...],
) -> tuple[Path, Path, Path]:
    vtt_path = tmp_path / "episode.vtt"
    json_path = tmp_path / "episode.json"
    media_path = tmp_path / "episode.wav"
    vtt_path.write_bytes(_vtt(*cues))
    json_path.write_text(
        json.dumps({"language": language}, ensure_ascii=False), encoding="utf-8"
    )
    media_path.write_bytes(b"synthetic-media")
    return vtt_path, json_path, media_path


def _route_full_pass(
    monkeypatch: pytest.MonkeyPatch,
    *,
    route: str,
    prepared_path: Path,
    callback: Any,
) -> None:
    from voxweave import backend, config, pipeline

    monkeypatch.setattr(
        pipeline,
        "_prepare_16k_for_align",
        lambda *_args, **_kwargs: prepared_path,
    )
    monkeypatch.setattr(backend, "uses_mms", lambda _iso: route == "mms")
    monkeypatch.setattr(
        config,
        "align_model_for",
        lambda _iso: None if route == "mms" else "synthetic-ctc",
    )
    target = "align_blocks_full_mms" if route == "mms" else "align_blocks_full_ctc"
    monkeypatch.setattr(backend, target, callback)


def _load_manifest() -> dict[str, Any]:
    assert ORACLE_MANIFEST.is_file(), "the checked P6 oracle manifest is missing"
    value = json.loads(ORACLE_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _oracle_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORACLE_RUNNER), *arguments],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def test_p6_oracle_checked_layout_is_substantive():
    required_files = (ORACLE_MANIFEST, ORACLE_SCHEMA, ORACLE_RUNNER)
    missing_files = [
        str(path.relative_to(REPO_ROOT))
        for path in required_files
        if not path.is_file()
    ]
    assert not missing_files, f"missing P6 oracle files: {missing_files}"

    required_trees = (
        ORACLE_ROOT / "inputs",
        ORACLE_ROOT / "media",
        ORACLE_ROOT / "backend-receipts",
        *(ORACLE_ROOT / "expected" / name for name in REFERENCE_SETS),
    )
    missing_trees = [
        str(path.relative_to(REPO_ROOT))
        for path in required_trees
        if not path.is_dir() or not any(item.is_file() for item in path.rglob("*"))
    ]
    assert not missing_trees, f"missing or empty P6 oracle trees: {missing_trees}"


def test_p6_oracle_manifest_has_closed_schema_references_and_gate_inventory():
    import jsonschema

    manifest = _load_manifest()
    assert ORACLE_SCHEMA.is_file(), "the checked P6 oracle schema is missing"
    schema = json.loads(ORACLE_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(manifest)

    references = manifest["reference_commits"]
    assert references["historical"] == "6e6033fa3930b263133f02c1332ae4d79a490f8b"
    assert references["post_p11"] == "b6d3b76dd518f943d922dc31cde227745892933d"

    cases = manifest["cases"]
    assert cases
    assert {case["reference_set"] for case in cases} == set(REFERENCE_SETS)
    assert {"RAT-1", "RAT-2", "RAT-3", "RAT-4", "RAT-5", "RAT-7"} <= {
        delta for case in cases for delta in case["ratified_deltas"]
    }

    gates = manifest["gates"]
    assert NAMED_GATES <= set(gates)
    align_ao = gates["G-ALIGN-AO"]
    assert align_ao["phase_order"] == list(AO_PHASES)
    assert set(align_ao["runtime_routes"]) == {"ctc-full", "mms-full", "qwen-crop"}
    assert set(align_ao["selected_families"]) == {"legacy-v1", "boundary-v2"}
    assert {"all-skip", "ao15-uncontained", "ao15-isolated-then-ao16"} <= set(
        align_ao["failure_cases"]
    )


def test_p6_oracle_runner_has_exact_exit_contract_and_read_only_compare(tmp_path):
    assert ORACLE_RUNNER.is_file(), "the detached P6 oracle runner is missing"

    invalid_manifest = tmp_path / "invalid.json"
    invalid_manifest.write_text("{}\n", encoding="utf-8")
    invalid = _oracle_command("validate", "--manifest", str(invalid_manifest))
    assert invalid.returncode == 2, (invalid.stdout, invalid.stderr)

    validated = _oracle_command("validate", "--manifest", str(ORACLE_MANIFEST))
    assert validated.returncode == 0, (validated.stdout, validated.stderr)

    before = {
        path.relative_to(ORACLE_ROOT): (path.stat().st_size, path.read_bytes())
        for path in ORACLE_ROOT.rglob("*")
        if path.is_file()
    }
    compared = _oracle_command("compare", "--manifest", str(ORACLE_MANIFEST), "--check")
    assert compared.returncode == 0, (compared.stdout, compared.stderr)
    after = {
        path.relative_to(ORACLE_ROOT): (path.stat().st_size, path.read_bytes())
        for path in ORACLE_ROOT.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_p6_oracle_compare_writes_a_deterministic_closed_json_report(tmp_path):
    manifest = _load_manifest()
    first_report = tmp_path / "first" / "p6-oracle-report.json"
    second_report = tmp_path / "second" / "p6-oracle-report.json"

    for report_path in (first_report, second_report):
        compared = _oracle_command(
            "compare",
            "--manifest",
            str(ORACLE_MANIFEST),
            "--check",
            "--json-out",
            str(report_path),
        )
        assert compared.returncode == 0, (compared.stdout, compared.stderr)

    assert first_report.read_bytes() == second_report.read_bytes()
    assert json.loads(first_report.read_bytes()) == {
        "artifact_count": sum(
            len(case["expected_paths"]) for case in manifest["cases"]
        ),
        "case_count": len(manifest["cases"]),
        "command": "compare",
        "failure_count": 0,
        "failures": [],
        "manifest_sha256": hashlib.sha256(ORACLE_MANIFEST.read_bytes()).hexdigest(),
        "runner_version": manifest["runner_version"],
        "schema_version": manifest["schema_version"],
        "status": "match",
    }


def test_p6_source_registry_and_dependency_gate_is_checked():
    assert ORACLE_RUNNER.is_file(), "the detached P6 oracle runner is missing"
    result = _oracle_command(
        "source-gates", "--manifest", str(ORACLE_MANIFEST), "--check"
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_align_has_one_exact_auditable_ao_phase_order():
    from voxweave import align_orchestration

    assert align_orchestration.ALIGN_AO_PHASE_ORDER == AO_PHASES
    source = inspect.getsource(align_orchestration)
    tree = ast.parse(source)
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "ALIGN_AO_PHASE_ORDER"
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        )
    ]
    assert len(definitions) == 1


@pytest.mark.parametrize(
    ("route", "language"),
    (("ctc", "en"), ("mms", "ja")),
)
def test_public_direct_full_pass_uses_lexical_vtt_order(
    route: str,
    language: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from voxweave import pipeline

    vtt_path, _json_path, media_path = _write_episode(
        tmp_path,
        language=language,
        cues=(
            ("00:01:40.000", "00:01:41.000", "FIRST"),
            ("00:00:00.000", "00:00:01.000", "SECOND"),
        ),
    )
    seen: dict[str, Any] = {}

    def stop_at_physical_call(
        _wav: Path,
        texts: list[str],
        _iso: str,
        *args: Any,
        bounds: Any = None,
        **_kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        seen["texts"] = texts
        seen["bounds"] = bounds
        raise _PhysicalCallReached

    _route_full_pass(
        monkeypatch,
        route=route,
        prepared_path=media_path,
        callback=stop_at_physical_call,
    )
    with pytest.raises(_PhysicalCallReached):
        pipeline.align(vtt_path, media_path=media_path, separate=False)

    assert seen == {
        "texts": ["FIRST", "SECOND"],
        "bounds": [(100.0, 101.0), (0.0, 1.0)],
    }


def _configure_real_over_budget_mms(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepared_path: Path,
    seconds: int,
) -> None:
    from voxweave import align_common, align_mms, backend, config, pipeline

    sample_rate = align_mms.MMS_SR
    monkeypatch.setattr(
        pipeline,
        "_prepare_16k_for_align",
        lambda *_args, **_kwargs: prepared_path,
    )
    monkeypatch.setattr(backend, "uses_mms", lambda _iso: True)
    monkeypatch.setattr(config, "align_model_for", lambda _iso: None)
    monkeypatch.setattr(
        align_mms,
        "_read_wav_16k",
        lambda _path: np.zeros(seconds * sample_rate, dtype=np.float32),
    )
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    monkeypatch.setattr(align_mms, "_empty_cache", lambda: None)


def _assert_dp_refusal(error: BaseException, detail_code: str) -> None:
    failure = getattr(error, "failure", None)
    assert failure is not None
    assert failure.kind == "dp-route-hints-invalid"
    assert failure.phase == "route-plan"
    assert failure.detail_code == detail_code


def test_public_over_budget_nonchronological_hints_refuse_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from voxweave import align_mms, pipeline

    vtt_path, _json_path, media_path = _write_episode(
        tmp_path,
        language="ja",
        cues=(
            ("00:01:40.000", "00:01:41.000", "A"),
            ("00:00:00.000", "00:00:01.000", "B"),
        ),
    )
    _configure_real_over_budget_mms(monkeypatch, prepared_path=media_path, seconds=120)
    monkeypatch.setattr(
        align_mms,
        "_mms_emit_units",
        lambda *_args, **_kwargs: pytest.fail(
            "unsafe over-budget hints reached model work"
        ),
    )

    with pytest.raises(RuntimeError) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    _assert_dp_refusal(caught.value, "hint-nonmonotone")


@pytest.mark.parametrize(
    ("plans", "detail_code"),
    (
        ([], "plan-nontiling"),
        (
            [
                {"lo": 0, "hi": 1, "start": 0.0, "end": 20.0},
                {"lo": 0, "hi": 2, "start": 20.0, "end": 40.0},
            ],
            "plan-nontiling",
        ),
        (
            [{"lo": 0, "hi": 2, "start": 0.0, "end": math.inf}],
            "crop-geometry",
        ),
        (
            [{"lo": 0, "hi": 2, "start": 0.0, "end": 40.0}],
            "crop-over-budget",
        ),
    ),
)
def test_public_over_budget_rejects_unsafe_planner_output_before_model_work(
    plans: list[dict[str, Any]],
    detail_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from voxweave import align_mms, chunking, pipeline

    vtt_path, _json_path, media_path = _write_episode(
        tmp_path,
        language="ja",
        cues=(
            ("00:00:00.000", "00:00:08.000", "A"),
            ("00:00:22.000", "00:00:39.000", "B"),
        ),
    )
    _configure_real_over_budget_mms(monkeypatch, prepared_path=media_path, seconds=40)
    monkeypatch.setattr(chunking, "plan_dp_chunks", lambda *_args, **_kwargs: plans)
    monkeypatch.setattr(
        align_mms,
        "_mms_emit_units",
        lambda *_args, **_kwargs: pytest.fail("unsafe plan reached model work"),
    )

    with pytest.raises(RuntimeError) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    _assert_dp_refusal(caught.value, detail_code)
