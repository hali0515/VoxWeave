from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import jsonschema
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "calibration" / "align-shadow"
CASE_ROOT = CORPUS_ROOT / "cases"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
BASELINE_PATH = CORPUS_ROOT / "baseline.json"
MANIFEST_SCHEMA_PATH = (
    REPO_ROOT / "calibration" / "schemas" / "align-shadow-manifest.schema.json"
)
ARTIFACT_SCHEMA_PATH = (
    REPO_ROOT / "calibration" / "schemas" / "align-shadow-artifact.schema.json"
)
REPORT_SCHEMA_PATH = (
    REPO_ROOT / "calibration" / "schemas" / "align-shadow-report.schema.json"
)
RUNNER_PATH = REPO_ROOT / "scripts" / "calib_align_shadow.py"

MANIFEST_KEYS = (
    "schema_version",
    "artifact_schema_version",
    "registry_sha256",
    "environment",
    "cases",
)
CASE_KEYS = (
    "id",
    "route",
    "effective_iso",
    "argv",
    "env",
    "inputs",
    "backend_receipt",
    "authority_limit_profile",
    "expected",
)
RICH_KEYS = (
    "schema_version",
    "artifact_kind",
    "status",
    "failure",
    "input",
    "fresh",
    "legacy",
    "v2",
    "comparison",
    "selected",
)
SEMANTIC_KEYS = (
    "semantic_root_lineage",
    "phase1_seed",
    "delivered",
    "report",
    "trace",
)
VALIDATOR_KEYS = (
    "partition_result",
    "trace_problems",
    "stability_problems",
)
COMPARISON_KEYS = (
    "registry_sha256",
    "active_classes",
    "primitive_field_diffs",
    "violations",
)
NORMAL_VARIANTS = {
    (False, "none"),
    (False, "collector"),
    (True, "none"),
    (True, "collector"),
    (True, "throwing"),
}
WATCHED_IMPORTS = (
    "voxweave.align_evidence_core",
    "voxweave.align_delta_registry",
    "voxweave.core.finalizer",
    "voxweave.core.align_compare",
    "voxweave.align_shadow",
    "voxweave.align_shadow_minimal",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_layout() -> tuple[Path, ...]:
    return (
        MANIFEST_PATH,
        BASELINE_PATH,
        MANIFEST_SCHEMA_PATH,
        ARTIFACT_SCHEMA_PATH,
        REPORT_SCHEMA_PATH,
        RUNNER_PATH,
    )


def _assert_layout() -> None:
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in _required_layout()
        if not path.is_file()
    ]
    if not CASE_ROOT.is_dir() or not any(
        path.is_file() for path in CASE_ROOT.rglob("*")
    ):
        missing.append("calibration/align-shadow/cases/")
    assert not missing, f"missing align-shadow calibration layout: {missing}"


def _run_runner(
    *arguments: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER_PATH), *arguments],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=None if env is None else dict(env),
    )


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _load_runner_module() -> ModuleType:
    _assert_layout()
    spec = importlib.util.spec_from_file_location("calib_align_shadow_red", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _paths(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, member in value.items():
            if key == "path" and isinstance(member, str):
                yield member
            yield from _paths(member)
    elif isinstance(value, list):
        for member in value:
            yield from _paths(member)


def _replace_first_path(value: object, replacement: str) -> bool:
    if isinstance(value, dict):
        for key, member in value.items():
            if key == "path" and isinstance(member, str):
                value[key] = replacement
                return True
            if _replace_first_path(member, replacement):
                return True
    elif isinstance(value, list):
        for member in value:
            if _replace_first_path(member, replacement):
                return True
    return False


def _artifact_from_run(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    encoded = run["artifact_bytes_b64"]
    if encoded is None:
        return None
    assert isinstance(encoded, str)
    raw = base64.b64decode(encoded, validate=True)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    value = json.loads(raw)
    assert isinstance(value, Mapping)
    assert hashlib.sha256(raw).hexdigest() == run["artifact_sha256"]
    return value


@pytest.fixture(scope="module")
def align_shadow_report(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Any]:
    _assert_layout()
    output = tmp_path_factory.mktemp("align-shadow-report") / "report.json"
    result = _run_runner(
        "report",
        "--manifest",
        str(MANIFEST_PATH),
        "--json-out",
        str(output),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    report = _load_json(output)
    jsonschema.Draft202012Validator(_load_json(REPORT_SCHEMA_PATH)).validate(report)
    assert isinstance(report, Mapping)
    return report


def test_align_shadow_checked_layout_is_complete_and_nonempty():
    _assert_layout()
    assert any(path.is_file() for path in CASE_ROOT.rglob("*"))


def test_align_shadow_manifest_baseline_and_schemas_are_closed():
    _assert_layout()
    manifest = _load_json(MANIFEST_PATH)
    manifest_schema = _load_json(MANIFEST_SCHEMA_PATH)
    artifact_schema = _load_json(ARTIFACT_SCHEMA_PATH)
    report_schema = _load_json(REPORT_SCHEMA_PATH)
    for schema in (manifest_schema, artifact_schema, report_schema):
        jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)

    assert tuple(manifest) == MANIFEST_KEYS
    assert manifest["artifact_schema_version"] == 2
    assert len(manifest["registry_sha256"]) == 64
    cases = manifest["cases"]
    assert cases and [case["id"] for case in cases] == sorted(
        case["id"] for case in cases
    )
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert tuple(case) == CASE_KEYS
        assert case["route"] in {"ctc-full", "mms-full", "qwen-crop"}
        assert "VOXWEAVE_SEG_V2_SHADOW" in case["env"]
        expected = case["expected"]
        assert {
            "fully_admitted",
            "selected",
            "artifact_kind",
            "failure",
            "detail_code",
            "alds",
            "artifact_sha256",
        } <= set(expected)
        assert expected["selected"]["engine_family"] in {
            "legacy-v1",
            "boundary-v2",
        }
        assert expected["artifact_kind"] in {"rich", "minimal-failure", "none"}
        for raw_path in _paths(case):
            path = Path(raw_path)
            assert raw_path and not path.is_absolute() and ".." not in path.parts

    baseline = _load_json(BASELINE_PATH)
    assert {
        "schema_version",
        "manifest_sha256",
        "manifest_schema_sha256",
        "artifact_schema_sha256",
        "report_schema_sha256",
        "registry_sha256",
        "cases",
    } == set(baseline)
    assert baseline["manifest_sha256"] == _sha256(MANIFEST_PATH)
    assert baseline["manifest_schema_sha256"] == _sha256(MANIFEST_SCHEMA_PATH)
    assert baseline["artifact_schema_sha256"] == _sha256(ARTIFACT_SCHEMA_PATH)
    assert baseline["report_schema_sha256"] == _sha256(REPORT_SCHEMA_PATH)
    assert baseline["registry_sha256"] == manifest["registry_sha256"]
    assert list(baseline["cases"]) == [case["id"] for case in cases]


def test_align_shadow_runner_calls_real_public_align_and_has_no_rerecord_path():
    _assert_layout()
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    public_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "align"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pipeline"
    ]
    assert public_calls, "the corpus runner must invoke real pipeline.align"
    for forbidden in (
        "run_locked_align_adapter",
        "build_align_selection",
        "_align_blocks",
        "record-baseline",
        "--update",
        "--write-baseline",
        "--baseline-out",
    ):
        assert forbidden not in source


def test_align_shadow_runner_exact_zero_one_two_and_check_only_contract(tmp_path):
    _assert_layout()
    before = _tree_snapshot(CORPUS_ROOT)
    report_path = tmp_path / "report.json"
    reported = _run_runner(
        "report",
        "--manifest",
        str(MANIFEST_PATH),
        "--json-out",
        str(report_path),
    )
    assert reported.returncode == 0, (reported.stdout, reported.stderr)
    report = _load_json(report_path)
    assert tuple(report)[-4:] == (
        "case_count",
        "valid_count",
        "infrastructure_invalid_count",
        "correctness_failure_count",
    )
    assert report["case_count"] == len(report["cases"])
    assert report["valid_count"] == report["case_count"]
    assert report["infrastructure_invalid_count"] == 0
    assert report["correctness_failure_count"] == 0

    checked = _run_runner(
        "check",
        "--manifest",
        str(MANIFEST_PATH),
        "--baseline",
        str(BASELINE_PATH),
        "--json-out",
        str(tmp_path / "checked.json"),
    )
    assert checked.returncode == 0, (checked.stdout, checked.stderr)
    assert _tree_snapshot(CORPUS_ROOT) == before

    bad_baseline = copy.deepcopy(_load_json(BASELINE_PATH))
    first_case = next(iter(bad_baseline["cases"].values()))
    first_case["artifact_sha256"] = "0" * 64
    bad_baseline_path = tmp_path / "mismatch-baseline.json"
    bad_baseline_path.write_text(
        json.dumps(bad_baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mismatch = _run_runner(
        "check",
        "--manifest",
        str(MANIFEST_PATH),
        "--baseline",
        str(bad_baseline_path),
        "--json-out",
        str(tmp_path / "mismatch.json"),
    )
    assert mismatch.returncode == 1, (mismatch.stdout, mismatch.stderr)

    invalid_manifest = tmp_path / "invalid-manifest.json"
    invalid_manifest.write_text("{}\n", encoding="utf-8")
    invalid = _run_runner(
        "report",
        "--manifest",
        str(invalid_manifest),
        "--json-out",
        str(tmp_path / "invalid.json"),
    )
    assert invalid.returncode == 2, (invalid.stdout, invalid.stderr)

    traversal_manifest = copy.deepcopy(_load_json(MANIFEST_PATH))
    assert _replace_first_path(traversal_manifest["cases"][0], "../escape")
    traversal_path = tmp_path / "traversal-manifest.json"
    traversal_path.write_text(
        json.dumps(traversal_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    traversal = _run_runner(
        "report",
        "--manifest",
        str(traversal_path),
        "--json-out",
        str(tmp_path / "traversal.json"),
    )
    assert traversal.returncode == 2, (traversal.stdout, traversal.stderr)


def test_align_shadow_report_has_exact_real_pipeline_matrix(
    align_shadow_report: Mapping[str, Any],
):
    cases = align_shadow_report["cases"]
    assert [case["id"] for case in cases] == sorted(case["id"] for case in cases)
    fully_admitted = [case for case in cases if case["fully_admitted"] is True]
    assert {case["engine_family"] for case in fully_admitted} == {
        "legacy-v1",
        "boundary-v2",
    }
    for case in fully_admitted:
        runs = case["runs"]
        assert {
            (run["shadow_enabled"], run["observer"]) for run in runs
        } == NORMAL_VARIANTS
        stable = {
            (
                run["normal_return"],
                json.dumps(run["selected"], sort_keys=True),
                json.dumps(run["p11_decisions"], sort_keys=True),
                run["evidence_bytes_b64"],
            )
            for run in runs
        }
        assert len(stable) == 1
        for run in runs:
            assert run["selected"]["engine_family"] == case["engine_family"]
            assert set(run["imports"]) == set(WATCHED_IMPORTS)
            artifact = _artifact_from_run(run)
            has_collector = run["observer"] in {"collector", "throwing"}
            if run["shadow_enabled"] and has_collector:
                assert run["artifact_kind"] == "rich"
                assert artifact is not None
            else:
                assert run["artifact_kind"] == "none"
                assert artifact is None


def test_align_shadow_report_pins_fresh_interpreter_import_truth(
    align_shadow_report: Mapping[str, Any],
):
    for case in align_shadow_report["cases"]:
        if case["fully_admitted"] is not True:
            continue
        family = case["engine_family"]
        for run in case["runs"]:
            imports = run["imports"]
            shadow = run["shadow_enabled"]
            observer = run["observer"]
            assert imports["voxweave.align_evidence_core"] is True
            assert imports["voxweave.align_delta_registry"] is True
            semantic_loaded = family == "boundary-v2" or shadow
            assert imports["voxweave.core.finalizer"] is semantic_loaded
            assert imports["voxweave.core.align_compare"] is semantic_loaded
            rich_loaded = shadow and observer in {"collector", "throwing"}
            assert imports["voxweave.align_shadow"] is rich_loaded
            assert imports["voxweave.align_shadow_minimal"] is False


def test_valid_rich_artifacts_retain_complete_atomic_semantic_facts(
    align_shadow_report: Mapping[str, Any],
):
    schema = _load_json(ARTIFACT_SCHEMA_PATH)
    seen_families: set[str] = set()
    for case in align_shadow_report["cases"]:
        if case["fully_admitted"] is not True:
            continue
        collector = next(
            run
            for run in case["runs"]
            if run["shadow_enabled"] is True and run["observer"] == "collector"
        )
        artifact = _artifact_from_run(collector)
        assert artifact is not None
        jsonschema.Draft202012Validator(schema).validate(artifact)
        assert tuple(artifact) == RICH_KEYS
        assert artifact["schema_version"] == 2
        assert artifact["artifact_kind"] == "rich"
        assert artifact["status"] == "valid"
        assert artifact["failure"] is None
        assert tuple(artifact["v2"]) == ("semantic", "validators")
        assert tuple(artifact["v2"]["semantic"]) == SEMANTIC_KEYS
        assert tuple(artifact["v2"]["validators"]) == VALIDATOR_KEYS
        assert tuple(artifact["comparison"]) == ("result",)
        assert tuple(artifact["comparison"]["result"]) == COMPARISON_KEYS
        assert artifact["v2"]["validators"]["trace_problems"] == []
        assert artifact["v2"]["validators"]["stability_problems"] == []
        assert artifact["comparison"]["result"]["violations"] == []
        seen_families.add(artifact["selected"]["engine_family"])
    assert seen_families == {"legacy-v1", "boundary-v2"}


def test_typed_invalid_observation_is_a_successful_corpus_outcome(
    align_shadow_report: Mapping[str, Any],
):
    invalid_cases = [
        case
        for case in align_shadow_report["cases"]
        if case["expected_failure"] is not None
    ]
    assert invalid_cases
    for case in invalid_cases:
        run = next(
            row
            for row in case["runs"]
            if row["shadow_enabled"] is True and row["observer"] == "collector"
        )
        artifact = _artifact_from_run(run)
        assert artifact is not None
        assert artifact["status"] == "invalid"
        assert artifact["failure"]["kind"] == case["expected_failure"]
        assert artifact["failure"]["detail_code"] == case["expected_detail_code"]
        assert run["alds"] == case["expected_alds"]
        assert run["outcome"] == "valid"


@pytest.mark.parametrize(
    "seam",
    ("freeze", "encode", "schema"),
)
def test_each_internal_rich_construction_failure_yields_one_minimal_callback(
    seam: str, tmp_path: Path
):
    runner = _load_runner_module()
    output = tmp_path / f"{seam}.json"
    result = runner.main(
        [
            "report",
            "--manifest",
            str(MANIFEST_PATH),
            "--json-out",
            str(output),
        ],
        _injections=frozenset({f"rich-{seam}"}),
    )
    assert result == 0
    report = _load_json(output)
    injected = [
        run
        for case in report["cases"]
        for run in case["runs"]
        if run.get("injection") == f"rich-{seam}"
    ]
    assert injected
    for run in injected:
        assert run["observer_call_count"] == 1
        assert run["artifact_kind"] == "minimal-failure"
        artifact = _artifact_from_run(run)
        assert artifact is not None
        assert artifact["failure"]["detail_code"] == "rich-artifact-construction"


def test_minimal_unavailable_waives_callback_and_makes_harness_exit_two(tmp_path):
    runner = _load_runner_module()
    output = tmp_path / "minimal-unavailable.json"
    result = runner.main(
        [
            "report",
            "--manifest",
            str(MANIFEST_PATH),
            "--json-out",
            str(output),
        ],
        _injections=frozenset({"rich-freeze", "minimal-construction"}),
    )
    assert result == 2
    report = _load_json(output)
    unavailable = [
        run
        for case in report["cases"]
        for run in case["runs"]
        if run.get("outcome") == "infrastructure-invalid"
    ]
    assert unavailable
    assert all(run["observer_call_count"] == 0 for run in unavailable)
    assert all(run["artifact_kind"] == "unavailable" for run in unavailable)
    assert report["infrastructure_invalid_count"] == len(unavailable)


def test_minimal_canonical_serializer_is_independent_for_both_selected_families():
    from voxweave.align_shadow_minimal import (
        build_minimal_align_shadow_failure_artifact,
    )

    for family in ("legacy-v1", "boundary-v2"):
        artifact = build_minimal_align_shadow_failure_artifact(
            context_content_digest="a" * 64,
            receipt_digest="b" * 64,
            engine_family=family,
            vtt_sha256="c" * 64,
            json_sha256="d" * 64,
            evidence_sha256="e" * 64,
            prior_failure=None,
        )
        expected = {
            "schema_version": 2,
            "artifact_kind": "minimal-failure",
            "status": "invalid",
            "failure": {
                "kind": "shadow-internal-error",
                "phase": "rich-artifact",
                "detail_code": "rich-artifact-construction",
                "secondary": [],
            },
            "context_content_digest": "a" * 64,
            "receipt_digest": "b" * 64,
            "selected": {
                "engine_family": family,
                "vtt_sha256": "c" * 64,
                "json_sha256": "d" * 64,
                "evidence_sha256": "e" * 64,
            },
        }
        expected_bytes = (
            json.dumps(expected, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        assert artifact.to_canonical_bytes() == expected_bytes


def test_align_shadow_report_is_relocation_stable(tmp_path):
    _assert_layout()
    reports: list[Mapping[str, Any]] = []
    for label in ("unrelated-a", "unrelated-b"):
        temp_root = tmp_path / label
        temp_root.mkdir()
        output = tmp_path / f"{label}.json"
        run_env = dict(os.environ)
        run_env["TMPDIR"] = str(temp_root)
        result = _run_runner(
            "report",
            "--manifest",
            str(MANIFEST_PATH),
            "--json-out",
            str(output),
            env=run_env,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        raw = output.read_bytes()
        assert str(temp_root).encode() not in raw
        report = json.loads(raw)
        assert isinstance(report, Mapping)
        reports.append(report)
    assert reports[0]["cases"] == reports[1]["cases"]
    assert {
        key: reports[0][key]
        for key in (
            "case_count",
            "valid_count",
            "infrastructure_invalid_count",
            "correctness_failure_count",
        )
    } == {
        key: reports[1][key]
        for key in (
            "case_count",
            "valid_count",
            "infrastructure_invalid_count",
            "correctness_failure_count",
        )
    }


def test_evidence_core_projector_and_ald6_have_one_dependency_safe_owner():
    owner = REPO_ROOT / "voxweave" / "align_evidence_core.py"
    definitions: dict[str, list[Path]] = {
        "EvidenceCore": [],
        "project_evidence_core": [],
        "evaluate_ald6": [],
    }
    for path in sorted((REPO_ROOT / "voxweave").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in definitions
            ):
                definitions[node.name].append(path)
    assert definitions == {name: [owner] for name in definitions}

    tree = ast.parse(owner.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden = {
        "voxweave.core.finalizer",
        "voxweave.core.align_compare",
        "voxweave.align_projector",
        "voxweave.reference_projector",
        "voxweave.candidate_encoder",
        "voxweave.episode_transaction",
        "voxweave.pipeline",
        "voxweave.align_shadow",
        "voxweave.align_shadow_minimal",
    }
    assert imports.isdisjoint(forbidden)
