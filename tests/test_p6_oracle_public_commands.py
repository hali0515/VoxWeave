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
PUBLIC_ARTIFACTS = (
    ("historical-selected-legacy", "vtt"),
    ("post-p11-speaker-turns", "main-json"),
    ("rat4-lexical-direct", "route-evidence"),
    ("selected-v2-align", "vtt"),
    ("selected-v2-align", "main-json"),
    ("selected-v2-align", "align-evidence"),
    ("selected-v2-segmentation", "vtt"),
    ("selected-v2-segmentation", "main-json"),
    ("combined-ratified-align", "vtt"),
    ("combined-ratified-align", "main-json"),
    ("combined-ratified-align", "align-evidence"),
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


@pytest.fixture(scope="module")
def public_authority_run() -> dict[str, Any]:
    monkeypatch = pytest.MonkeyPatch()
    _pinned_environment(monkeypatch)
    try:
        oracle = _load_oracle_runner()
        manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
        oracle_root = ORACLE_MANIFEST.parent
        results = {}
        projections = {}
        for case in manifest["cases"]:
            results[case["id"]] = oracle._execute_public_case(
                case,
                manifest_path=ORACLE_MANIFEST,
            )
            projections[case["id"]] = oracle._project_case(case, oracle_root)
        return {
            "manifest": manifest,
            "projections": projections,
            "results": results,
        }
    finally:
        monkeypatch.undo()


def test_every_oracle_case_executes_its_recorded_public_command_in_a_clean_root(
    public_authority_run: dict[str, Any],
):
    manifest = public_authority_run["manifest"]
    results = public_authority_run["results"]
    source_roots: set[Path] = set()
    episode_roots: set[Path] = set()

    for case in manifest["cases"]:
        result = results[case["id"]]
        assert result.case_id == case["id"]
        assert result.command == case["command"]
        assert set(result.artifacts) == {
            output["artifact"] for output in case["expected_paths"]
        }
        if case["command"] == "align" and case["reference_set"] != "6e6033f":
            assert result.evidence_verification["detail_code"] is None
            assert result.evidence_verification["integrity"] is True
            assert (
                result.evidence_verification["w1_usable"]
                is case["public_runtime"]["expected_w1_usable"]
            )
        else:
            assert result.evidence_verification is None
        assert result.source_root not in source_roots
        assert result.episode_root not in episode_roots
        assert not result.source_root.exists()
        assert not result.episode_root.exists()
        source_roots.add(result.source_root)
        episode_roots.add(result.episode_root)

    assert tuple(case["id"] for case in manifest["cases"]) == PUBLIC_CASE_IDS


@pytest.mark.parametrize(
    ("case_id", "artifact"),
    PUBLIC_ARTIFACTS,
    ids=(f"{case_id}-{artifact}" for case_id, artifact in PUBLIC_ARTIFACTS),
)
def test_public_command_artifact_matches_the_standalone_projector_authority(
    case_id: str,
    artifact: str,
    public_authority_run: dict[str, Any],
):
    result = public_authority_run["results"][case_id]
    projection = public_authority_run["projections"][case_id]

    assert result.artifacts[artifact] == projection[artifact]


def test_public_command_environment_rejects_ambient_feature_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    _pinned_environment(monkeypatch)
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    oracle = _load_oracle_runner()
    manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
    case = next(
        case for case in manifest["cases"] if case["id"] == "selected-v2-segmentation"
    )

    result = oracle._execute_public_case(case, manifest_path=ORACLE_MANIFEST)
    assert result.artifacts == oracle._project_case(case, ORACLE_MANIFEST.parent)


def test_align_public_cases_expose_production_owned_runtime_phase_traces(
    monkeypatch: pytest.MonkeyPatch,
):
    _pinned_environment(monkeypatch)
    oracle = _load_oracle_runner()
    manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
    align_cases = [
        case
        for case in manifest["cases"]
        if case["command"] == "align" and case["reference_set"] != "6e6033f"
    ]

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


@pytest.mark.parametrize("mutation", ("missing-ao22", "reordered-ao10"))
def test_mms_scenario_validator_rejects_incomplete_or_reordered_live_trace(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _pinned_environment(monkeypatch)
    oracle = _load_oracle_runner()
    manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
    scenario = next(
        scenario
        for scenario in manifest["gates"]["G-ALIGN-AO"]["runtime_scenarios"]
        if scenario["id"] == "mms-legacy-happy"
    )
    result = oracle._execute_runtime_scenario(
        scenario,
        manifest=manifest,
        manifest_path=ORACLE_MANIFEST,
    )
    events = [dict(event) for event in result.runtime_trace["events"]]
    if mutation == "missing-ao22":
        events = [event for event in events if event["phase"] != "AO-22"]
    else:
        group = [event for event in events if event["activity"] == "group-block-spans"]
        empty = [
            event
            for event in events
            if event["activity"] == "common-all-empty-decision"
        ]
        events = [
            event
            for event in events
            if event["activity"]
            not in {"group-block-spans", "common-all-empty-decision"}
        ]
        insertion = next(
            index for index, event in enumerate(events) if event["phase"] == "AO-11"
        )
        events[insertion:insertion] = empty + group
    for ordinal, event in enumerate(events):
        event["ordinal"] = ordinal
    mutated_trace = dict(result.runtime_trace)
    mutated_trace["events"] = events
    mutated = oracle.RuntimeScenarioResult(
        evidence_verification=result.evidence_verification,
        outcome=result.outcome,
        runtime_trace=mutated_trace,
        scenario_id=result.scenario_id,
    )

    assert oracle._runtime_scenario_failures(scenario, mutated)


@pytest.mark.parametrize("forbidden_phase", ("AO-22", "AO-23", "AO-25"))
def test_uncontained_ao15_validator_rejects_all_late_work(
    forbidden_phase: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _pinned_environment(monkeypatch)
    oracle = _load_oracle_runner()
    manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
    scenario = next(
        scenario
        for scenario in manifest["gates"]["G-ALIGN-AO"]["runtime_scenarios"]
        if scenario["id"] == "ao15-uncontained-boundary"
    )
    result = oracle._execute_runtime_scenario(
        scenario,
        manifest=manifest,
        manifest_path=ORACLE_MANIFEST,
    )
    events = [dict(event) for event in result.runtime_trace["events"]]
    insertion = (
        len(events)
        if forbidden_phase == "AO-25"
        else next(
            index for index, event in enumerate(events) if event["phase"] == "AO-24"
        )
    )
    events[insertion:insertion] = [
        {
            "ordinal": -1,
            "phase": forbidden_phase,
            "activity": "mutant-late-work",
            "state": state,
        }
        for state in ("started", "failed")
    ]
    for ordinal, event in enumerate(events):
        event["ordinal"] = ordinal
    mutated_trace = dict(result.runtime_trace)
    mutated_trace["events"] = events
    mutated = oracle.RuntimeScenarioResult(
        evidence_verification=result.evidence_verification,
        outcome=result.outcome,
        runtime_trace=mutated_trace,
        scenario_id=result.scenario_id,
    )

    assert oracle._runtime_scenario_failures(scenario, mutated)


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
    oracle_phases = tuple(f"AO-{index:02d}" for index in range(1, 26))
    assert tuple(case["id"] for case in manifest["cases"]) == PUBLIC_CASE_IDS
    for case in manifest["cases"]:
        runtime = case["public_runtime"]
        assert set(runtime) == {
            "fixture",
            "expected_family",
            "expected_trace",
            "expected_w1_usable",
        }
        assert runtime["fixture"].startswith("inputs/")
        assert runtime["expected_family"] in {"legacy-v1", "boundary-v2"}
        if case["command"] == "align" and case["reference_set"] != "6e6033f":
            assert runtime["expected_trace"]
            assert type(runtime["expected_w1_usable"]) is bool
            required_phases = (
                oracle_phases
                if runtime["expected_family"] == "boundary-v2"
                else tuple(
                    phase for phase in oracle_phases if phase not in {"AO-15", "AO-17"}
                )
            )
            assert (
                tuple(event["phase"] for event in runtime["expected_trace"])
                == required_phases
            )
        else:
            assert runtime["expected_trace"] is None
            assert runtime["expected_w1_usable"] is None
