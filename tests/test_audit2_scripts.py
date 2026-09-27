"""Regression tests for the scripts/ tooling fixes: the detached P6 oracle, its
release refresh, and the song-score diagnostic.

These load each script by path (``scripts/`` is not a package) and exercise the
exit-code contract and messages directly; none of them runs a public command.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
ORACLE_MANIFEST = REPO_ROOT / "calibration" / "p6-oracle" / "manifest.json"


def _load(name: str, *, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.path.pop(0) == str(SCRIPTS)
    return module


@pytest.fixture(scope="module")
def oracle() -> Any:
    return _load("p6_oracle", module_name="p6_oracle_audit2_runner")


@pytest.fixture(scope="module")
def refresh() -> Any:
    return _load(
        "p6_oracle_release_refresh", module_name="p6_oracle_audit2_release_refresh"
    )


# --------------------------------------------------------------------------- #
# p6_oracle.py
# --------------------------------------------------------------------------- #


def test_oracle_crash_is_invalid_not_mismatch(
    oracle: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def crash(_path: Path) -> Any:
        raise KeyError("execution")

    monkeypatch.setattr(oracle, "_load_checked_manifest", crash)

    code = oracle.main(["compare", "--manifest", str(ORACLE_MANIFEST), "--check"])

    assert code == oracle.EXIT_INVALID == 2
    err = capsys.readouterr().err
    assert "invalid: internal oracle error: KeyError" in err
    assert "Traceback" in err


def test_evidence_that_cannot_run_is_invalid(
    oracle: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("no such interpreter")

    monkeypatch.setattr(oracle.subprocess, "run", refuse)
    manifest = json.loads(ORACLE_MANIFEST.read_bytes())

    with pytest.raises(oracle.OracleInvalid, match="could not run"):
        oracle._execute_test_evidence(manifest)


def test_environment_mismatch_names_recorded_and_observed(oracle: Any) -> None:
    execution = dict(json.loads(ORACLE_MANIFEST.read_bytes())["execution"])
    execution["interpreter"] = "CPython 0.0.1"

    with pytest.raises(oracle.OracleInvalid) as excinfo:
        oracle._validate_execution(execution)

    message = str(excinfo.value)
    assert "recorded 'CPython 0.0.1'" in message
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in message


def test_check_flag_is_optional_and_documented(
    oracle: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    parsed = oracle._parser().parse_args(["source-gates", "--manifest", "m.json"])
    assert parsed.check is False

    with pytest.raises(SystemExit):
        oracle._parser().parse_args(["compare", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "never rewrites its corpus" in out
    assert "comparison report" in out


def test_delivery_input_must_be_unique(oracle: Any, tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for folder in ("a", "b"):
        (tmp_path / folder / "delivery.json").write_text("{}", encoding="utf-8")
    case = {
        "id": "dup",
        "input_files": [{"path": "a/delivery.json"}, {"path": "b/delivery.json"}],
    }

    with pytest.raises(oracle.OracleInvalid, match="one input named delivery.json"):
        oracle._load_delivery(case, tmp_path, basename="delivery.json")


def test_unknown_runtime_scenario_reports_no_validator(oracle: Any) -> None:
    events = [
        {"ordinal": 0, "phase": "AO-01", "activity": "a", "state": "started"},
        {"ordinal": 1, "phase": "AO-01", "activity": "a", "state": "completed"},
    ]
    result = oracle.RuntimeScenarioResult(
        evidence_verification=None,
        outcome={"success": True},
        runtime_trace={
            "schema_version": 1,
            "route_kind": "ctc-full",
            "engine_family": "legacy-v1",
            "events": events,
        },
        scenario_id="not-a-scenario",
    )
    scenario = {
        "id": "not-a-scenario",
        "route": "ctc-full",
        "expected_family": "legacy-v1",
        "expect_failure": False,
    }

    failures = oracle._runtime_scenario_failures(scenario, result)

    assert any("scenario has no validator" in line for line in failures)


def test_runtime_scenarios_refuse_disagreeing_case_environments(oracle: Any) -> None:
    manifest = {
        "cases": [
            {"environment": {"LANG": "zh_CN.UTF-8", "TZ": None}},
            {"environment": {"LANG": "C.UTF-8", "TZ": None}},
        ]
    }

    with pytest.raises(oracle.OracleInvalid, match="one environment shared"):
        oracle._scenario_environment(manifest)


# --------------------------------------------------------------------------- #
# p6_oracle_release_refresh.py
# --------------------------------------------------------------------------- #


def _pin_versions(refresh: Any, monkeypatch: pytest.MonkeyPatch, lock: str) -> None:
    version = json.loads(ORACLE_MANIFEST.read_bytes())["execution"]["package_version"]
    environment = refresh.oracle_environment
    monkeypatch.setattr(environment, "project_package_version", lambda _root: version)
    monkeypatch.setattr(environment, "locked_package_version", lambda _root: version)
    monkeypatch.setattr(environment, "installed_package_version", lambda: version)
    monkeypatch.setattr(environment, "sha256_file", lambda _path: lock)


def test_release_refresh_accepts_a_dependency_only_lock_change(
    refresh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = ORACLE_MANIFEST.read_bytes()
    before = json.loads(raw)["execution"]
    _pin_versions(refresh, monkeypatch, "b" * 64)

    _candidate_raw, candidate = refresh._candidate_manifest(raw)

    after = candidate["execution"]
    assert {key for key in before if before[key] != after[key]} == {
        "dependency_lock_sha256",
        "container_digest",
    }
    assert after["dependency_lock_sha256"] == "b" * 64


def test_release_refresh_with_nothing_to_change_says_so(
    refresh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = ORACLE_MANIFEST.read_bytes()
    recorded = json.loads(raw)["execution"]["dependency_lock_sha256"]
    _pin_versions(refresh, monkeypatch, recorded)

    with pytest.raises(refresh.RefreshInvalid, match="nothing to refresh"):
        refresh._candidate_manifest(raw)


# --------------------------------------------------------------------------- #
# song_scores.py
# --------------------------------------------------------------------------- #


def test_song_scores_falls_back_to_the_mix_on_an_unreadable_cache_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    song_scores = _load("song_scores", module_name="song_scores_audit2")

    def broken(_media: Path) -> Any:
        raise song_scores.artifacts.ArtifactCollisionError("claim belongs to x.mkv")

    monkeypatch.setattr(song_scores.artifacts, "inspect_paths", broken)

    assert song_scores._vocals_source(tmp_path / "episode.mkv") is None
    assert "ignoring the artifact cache: claim belongs to x.mkv" in (
        capsys.readouterr().out
    )
