from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_MANIFEST = REPO_ROOT / "calibration" / "p6-oracle" / "manifest.json"
ORACLE_RUNNER = REPO_ROOT / "scripts" / "p6_oracle.py"
PUBLIC_CASE_IDS = (
    "historical-selected-legacy",
    "post-p11-speaker-turns",
    "rat4-lexical-direct",
    "selected-v2-align",
    "selected-v2-segmentation",
    "combined-ratified-align",
)


def _load_oracle_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "p6_oracle_public_command_runner", ORACLE_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("LC_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("VOXWEAVE_CONFIG", raising=False)


def test_every_oracle_case_executes_its_recorded_public_command_in_a_clean_root(
    monkeypatch: pytest.MonkeyPatch,
):
    _pinned_environment(monkeypatch)
    oracle = _load_oracle_runner()
    manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
    oracle_root = ORACLE_MANIFEST.parent
    source_roots: set[Path] = set()
    episode_roots: set[Path] = set()

    for case in manifest["cases"]:
        result = oracle._execute_public_case(
            case,
            manifest_path=ORACLE_MANIFEST,
        )
        assert result.case_id == case["id"]
        assert result.command == case["command"]
        assert result.artifacts == oracle._project_case(case, oracle_root)
        assert result.source_root not in source_roots
        assert result.episode_root not in episode_roots
        assert not result.source_root.exists()
        assert not result.episode_root.exists()
        source_roots.add(result.source_root)
        episode_roots.add(result.episode_root)

    assert tuple(case["id"] for case in manifest["cases"]) == PUBLIC_CASE_IDS


def test_align_public_cases_expose_production_owned_runtime_phase_traces(
    monkeypatch: pytest.MonkeyPatch,
):
    _pinned_environment(monkeypatch)
    oracle = _load_oracle_runner()
    manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
    align_cases = [case for case in manifest["cases"] if case["command"] == "align"]

    for case in align_cases:
        result = oracle._execute_public_case(case, manifest_path=ORACLE_MANIFEST)
        trace = result.runtime_trace
        assert trace is not None
        assert trace["route_kind"] == case["route"]
        assert trace["engine_family"] in {"legacy-v1", "boundary-v2"}
        events = trace["events"]
        assert events
        assert [event["ordinal"] for event in events] == list(range(len(events)))
        assert all(event["phase"].startswith("AO-") for event in events)
        assert oracle._runtime_ao_failures(case, trace) == []


@pytest.mark.parametrize("command", ("compare", "source-gates"))
def test_live_ao_order_mutation_forces_gate_exit_one_without_changing_declarations(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _pinned_environment(monkeypatch)
    oracle = _load_oracle_runner()
    declaration = REPO_ROOT / "voxweave" / "align_orchestration.py"
    before = hashlib.sha256(declaration.read_bytes()).hexdigest()
    arguments = [command, "--manifest", str(ORACLE_MANIFEST), "--check"]

    with oracle._runtime_mutation_for_test("ao10-after-ao11"):
        assert oracle.main(arguments) == 1

    assert hashlib.sha256(declaration.read_bytes()).hexdigest() == before


def test_manifest_public_runtime_contract_is_closed_and_non_declarative():
    manifest = json.loads(ORACLE_MANIFEST.read_bytes())
    assert tuple(case["id"] for case in manifest["cases"]) == PUBLIC_CASE_IDS
    for case in manifest["cases"]:
        runtime = case["public_runtime"]
        assert set(runtime) == {
            "fixture",
            "expected_family",
            "expected_trace",
        }
        assert runtime["fixture"].startswith("inputs/")
        assert runtime["expected_family"] in {"legacy-v1", "boundary-v2"}
        if case["command"] == "align":
            assert runtime["expected_trace"]
        else:
            assert runtime["expected_trace"] is None

