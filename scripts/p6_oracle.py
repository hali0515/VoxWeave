#!/usr/bin/env python3
"""Detached, read-only P6 oracle validator and comparator.

Exit codes are exact: 0 means valid and matching, 1 means a valid comparison or
source gate failed, and 2 means the manifest, environment, reference, or tooling
is invalid.  This module imports no voxweave production module or serializer.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator, NoReturn, cast

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_calib_common() -> Any:
    cached = sys.modules.get("calib_common")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "calib_common", SCRIPTS_DIR / "calib_common.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise RuntimeError(f"cannot load {SCRIPTS_DIR / 'calib_common.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["calib_common"] = module
    spec.loader.exec_module(module)
    return module


cc = _load_calib_common()

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_INVALID = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "calibration" / "schemas" / "p6-oracle.schema.json"
EXPECTED_REFERENCE_SETS = {
    "6e6033f",
    "post-p11",
    "p6-lexical",
    "selected-v2-align",
    "selected-v2-segmentation",
}
EXPECTED_RATIFIED_DELTAS = {"RAT-1", "RAT-2", "RAT-3", "RAT-4", "RAT-5", "RAT-7"}
EXPECTED_GATES = {
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
AO10_HELPER_CHAIN = (
    "group-block-spans",
    "common-all-empty-decision",
    "fill-insert-blocks",
    "enforce-min-duration",
    "rescue-tiny-cues",
    "clamp-spans",
    "seal-selected-legacy-result",
)
EXPECTED_RUNTIME_SCENARIOS = {
    "mms-legacy-happy": {
        "route": "mms-full",
        "expected_family": "legacy-v1",
        "shadow_requested": False,
        "injections": [],
        "expect_failure": False,
    },
    "qwen-all-skip-legacy": {
        "route": "qwen-crop",
        "expected_family": "legacy-v1",
        "shadow_requested": False,
        "injections": [],
        "expect_failure": True,
    },
    "qwen-all-skip-boundary": {
        "route": "qwen-crop",
        "expected_family": "boundary-v2",
        "shadow_requested": False,
        "injections": [],
        "expect_failure": True,
    },
    "ao15-uncontained-boundary": {
        "route": "ctc-full",
        "expected_family": "boundary-v2",
        "shadow_requested": False,
        "injections": ["ao15-w1"],
        "expect_failure": True,
    },
    "ao15-isolated-then-ao16-legacy": {
        "route": "ctc-full",
        "expected_family": "legacy-v1",
        "shadow_requested": True,
        "injections": ["ao15-w1"],
        "expect_failure": False,
    },
    "paired-ao15-ao16-boundary": {
        "route": "ctc-full",
        "expected_family": "boundary-v2",
        "shadow_requested": False,
        "injections": ["ao15-w1", "ao16-core"],
        "expect_failure": True,
    },
    "paired-ao15-ao16-legacy": {
        "route": "ctc-full",
        "expected_family": "legacy-v1",
        "shadow_requested": True,
        "injections": ["ao15-w1", "ao16-core"],
        "expect_failure": True,
    },
}
REFERENCE_COMMITS = {
    "historical": "6e6033fa3930b263133f02c1332ae4d79a490f8b",
    "post_p11": "b6d3b76dd518f943d922dc31cde227745892933d",
}
REGISTRY_PATHS = {
    "align-delta-registry": REPO_ROOT / "voxweave" / "align_delta_registry.py",
    "align-failure-registry": REPO_ROOT / "voxweave" / "align_failures.py",
    "engine-registry": REPO_ROOT / "voxweave" / "engine_registry.py",
    "p6-oracle-schema": SCHEMA_PATH,
}

_RUNTIME_MUTATION: ContextVar[str | None] = ContextVar(
    "p6_oracle_runtime_mutation",
    default=None,
)


class PublicCaseResult:
    """Bytes and production observations from one isolated public command."""

    __slots__ = (
        "artifacts",
        "case_id",
        "command",
        "episode_root",
        "evidence_verification",
        "runtime_trace",
        "source_root",
    )

    def __init__(
        self,
        *,
        artifacts: Mapping[str, bytes],
        case_id: str,
        command: str,
        episode_root: Path,
        evidence_verification: Mapping[str, Any] | None,
        runtime_trace: Mapping[str, Any] | None,
        source_root: Path,
    ) -> None:
        self.artifacts = dict(artifacts)
        self.case_id = case_id
        self.command = command
        self.episode_root = episode_root
        self.evidence_verification = evidence_verification
        self.runtime_trace = runtime_trace
        self.source_root = source_root


class RuntimeScenarioResult:
    __slots__ = ("evidence_verification", "outcome", "runtime_trace", "scenario_id")

    def __init__(
        self,
        *,
        evidence_verification: Mapping[str, Any] | None,
        outcome: Mapping[str, Any],
        runtime_trace: Mapping[str, Any],
        scenario_id: str,
    ) -> None:
        self.evidence_verification = evidence_verification
        self.outcome = outcome
        self.runtime_trace = runtime_trace
        self.scenario_id = scenario_id


@contextmanager
def _runtime_mutation_for_test(name: str) -> Iterator[None]:
    """Select a copied-source mutation; the checked-in tree is never changed."""

    if name != "ao10-after-ao11":
        raise ValueError(f"unknown P6 runtime mutation {name!r}")
    token: Token[str | None] = _RUNTIME_MUTATION.set(name)
    try:
        yield
    finally:
        _RUNTIME_MUTATION.reset(token)


class OracleInvalid(Exception):
    """The run has no standing to compare outputs or source gates."""


def _invalid(message: str) -> NoReturn:
    raise OracleInvalid(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        _invalid(f"cannot read {path}: {exc}")
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, member in pairs:
            if key in value:
                raise ValueError(f"duplicate object member {key!r}")
            value[key] = member
        return value

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=closed_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _invalid(f"cannot decode JSON {path}: {exc}")
    return value


def _resolve_under(base: Path, value: object, *, label: str) -> Path:
    if type(value) is not str or not value:
        _invalid(f"{label} is not a nonempty relative path")
    candidate = (base / value).resolve()
    root = base.resolve()
    if not candidate.is_relative_to(root):
        _invalid(f"{label} escapes {root}")
    return candidate


def _schema_validate(manifest: object) -> Mapping[str, Any]:
    try:
        import jsonschema
    except ImportError as exc:
        _invalid(f"jsonschema is unavailable: {exc}")
    jsonschema_api = cast(Any, jsonschema)
    schema = _read_json(SCHEMA_PATH)
    if not isinstance(schema, Mapping):
        _invalid(f"schema is not an object: {SCHEMA_PATH}")
    try:
        jsonschema_api.Draft202012Validator.check_schema(schema)
        jsonschema_api.Draft202012Validator(schema).validate(manifest)
    except jsonschema_api.exceptions.SchemaError as exc:
        _invalid(f"invalid P6 oracle schema: {exc.message}")
    except jsonschema_api.exceptions.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        _invalid(f"invalid P6 oracle manifest at {location}: {exc.message}")
    assert isinstance(manifest, Mapping)
    return manifest


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_file_fact(root: Path, fact: Mapping[str, Any], *, label: str) -> None:
    path = _resolve_under(root, fact["path"], label=f"{label}.path")
    present = fact["present"]
    if present is False:
        if path.exists():
            _invalid(f"{label} records an absent path that exists: {path}")
        return
    if not path.is_file():
        _invalid(f"{label} is missing: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        _invalid(f"cannot stat {path}: {exc}")
    if size != fact["size"]:
        _invalid(f"{label} size differs: {path}")
    if _sha256_file(path) != fact["sha256"]:
        _invalid(f"{label} digest differs: {path}")


def _validate_reference_commits(references: Mapping[str, Any]) -> None:
    for name, expected in REFERENCE_COMMITS.items():
        if references.get(name) != expected:
            _invalid(f"reference commit {name} is not the governing exact commit")
    for name, commit in references.items():
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=REPO_ROOT,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            _invalid(f"reference commit {name} is unavailable: {commit}")


def _validate_execution(execution: Mapping[str, Any]) -> None:
    expected_interpreter = (
        f"{platform.python_implementation()} "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    expected_platform = f"{platform.system()} {platform.machine()}"
    if execution["interpreter"] != expected_interpreter:
        _invalid("interpreter does not match the recorded oracle environment")
    if execution["platform"] != expected_platform:
        _invalid("platform does not match the recorded oracle environment")
    lock_path = REPO_ROOT / "uv.lock"
    lock_digest = _sha256_file(lock_path)
    if lock_digest != execution["dependency_lock_sha256"]:
        _invalid("dependency lock digest differs from the oracle environment")
    locale = os.environ.get("LC_ALL") or os.environ.get("LANG") or "unset"
    timezone = os.environ.get("TZ") or "unset"
    hash_seed = os.environ.get("PYTHONHASHSEED") or "unset"
    if execution["locale"] != locale:
        _invalid("locale does not match the recorded oracle environment")
    if execution["timezone"] != timezone:
        _invalid("timezone does not match the recorded oracle environment")
    if execution["hash_seed"] != hash_seed:
        _invalid("hash seed does not match the recorded oracle environment")
    environment_digest = _canonical_digest(
        {
            "dependency_lock_sha256": lock_digest,
            "hash_seed": hash_seed,
            "interpreter": expected_interpreter,
            "locale": locale,
            "platform": expected_platform,
            "timezone": timezone,
            "toolchain": "detached-environment-v1",
        }
    )
    if execution["container_digest"] != environment_digest:
        _invalid(
            "detached container/toolchain digest differs from the oracle environment"
        )


def _validate_registry_digests(digests: Mapping[str, Any]) -> None:
    if set(digests) != set(REGISTRY_PATHS):
        _invalid("registry digest inventory is incomplete or contains unknown entries")
    for name, path in REGISTRY_PATHS.items():
        if _sha256_file(path) != digests[name]:
            _invalid(f"registry/schema digest differs: {name}")


def _validate_environment(environment: Mapping[str, Any], *, case_id: str) -> None:
    for name, expected in environment.items():
        if os.environ.get(name) != expected:
            _invalid(f"case {case_id} environment differs for {name}")


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        _invalid(f"{label} is not closed: {observed} != {sorted(expected)}")
    return cast(Mapping[str, Any], value)


def _validate_public_fixture(
    case: Mapping[str, Any],
    *,
    oracle_root: Path,
) -> None:
    runtime = cast(Mapping[str, Any], case["public_runtime"])
    fixture_path = _resolve_under(
        oracle_root,
        runtime["fixture"],
        label=f"{case['id']}.public_runtime.fixture",
    )
    matching_facts = [
        fact for fact in case["input_files"] if fact["path"] == runtime["fixture"]
    ]
    if len(matching_facts) != 1:
        _invalid(
            f"case {case['id']} public fixture must have one immutable input-file fact"
        )
    fixture = _read_json(fixture_path)
    command = case["command"]
    route = case["route"]
    common = {"target", "language", "route", "json"}
    if command == "align":
        expected = common | {"vtt", "media"}
        expected.add("qwen_call_units" if route == "qwen-crop" else "block_units")
    elif command == "correct":
        expected = common | {"vtt"}
    elif command == "split":
        expected = common | {"speaker_mapping"}
    else:
        _invalid(f"case {case['id']} has no public fixture law for {command}")
    root = _require_exact_keys(fixture, expected, label=f"case {case['id']} fixture")
    if root["target"] != case["logical_target"] or root["route"] != route:
        _invalid(f"case {case['id']} public fixture identity differs from manifest")
    for name in ("target", "language", "route", "json"):
        if type(root[name]) is not str or not root[name]:
            _invalid(f"case {case['id']} fixture {name} is not a nonempty string")
    if command in {"align", "correct"} and type(root["vtt"]) is not str:
        _invalid(f"case {case['id']} fixture VTT is not text")
    if command == "split" and type(root["speaker_mapping"]) is not str:
        _invalid(f"case {case['id']} fixture speaker mapping is not text")
    if command == "align":
        media = _require_exact_keys(
            root["media"],
            {"sample_rate", "sample_count"},
            label=f"case {case['id']} fixture media",
        )
        if any(type(media[name]) is not int or media[name] <= 0 for name in media):
            _invalid(f"case {case['id']} fixture media geometry is invalid")
        receipt_name = "qwen_call_units" if route == "qwen-crop" else "block_units"
        receipts = root[receipt_name]
        if not isinstance(receipts, list) or not receipts:
            _invalid(f"case {case['id']} fixture has no physical receipt units")
        for call_index, units in enumerate(receipts):
            if not isinstance(units, list) or not units:
                _invalid(f"case {case['id']} fixture call {call_index} has no units")
            for unit_index, unit in enumerate(units):
                row = _require_exact_keys(
                    unit,
                    {"text", "start", "end"},
                    label=(
                        f"case {case['id']} fixture call {call_index} unit {unit_index}"
                    ),
                )
                if type(row["text"]) is not str or not all(
                    type(row[name]) in {int, float} for name in ("start", "end")
                ):
                    _invalid(f"case {case['id']} fixture unit has invalid scalars")


def _validate_manifest_semantics(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> None:
    oracle_root = manifest_path.resolve().parent
    if oracle_root != (REPO_ROOT / "calibration" / "p6-oracle").resolve():
        _invalid("the checked oracle manifest must live in calibration/p6-oracle")

    _validate_reference_commits(manifest["reference_commits"])
    _validate_execution(manifest["execution"])
    _validate_registry_digests(manifest["registry_digests"])

    cases = manifest["cases"]
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        _invalid("oracle case IDs are not unique")
    case_id_set = set(case_ids)
    reference_sets = {case["reference_set"] for case in cases}
    if reference_sets != EXPECTED_REFERENCE_SETS:
        _invalid("oracle cases do not cover the exact reference-set inventory")
    deltas = {delta for case in cases for delta in case["ratified_deltas"]}
    if not EXPECTED_RATIFIED_DELTAS.issubset(deltas):
        _invalid("oracle cases do not cover every landed ratified delta")
    selected_align = [
        case
        for case in cases
        if case["projector"] in {"selected-v2-align", "combined-ratified-align"}
    ]
    if {case["projector"] for case in selected_align} != {
        "selected-v2-align",
        "combined-ratified-align",
    }:
        _invalid("selected-v2 align and combined full-golden cases are both required")
    for case in selected_align:
        artifacts = {output["artifact"] for output in case["expected_paths"]}
        if artifacts != {"vtt", "main-json", "align-evidence"}:
            _invalid(f"case {case['id']} lacks its complete align artifact set")
        sizes = {
            output["artifact"]: output["size"] for output in case["expected_paths"]
        }
        if (
            sizes["vtt"] < 50
            or sizes["main-json"] < 500
            or sizes["align-evidence"] < 5000
        ):
            _invalid(f"case {case['id']} has a non-substantive selected-v2 golden")
    combined = next(
        case
        for case in selected_align
        if case["projector"] == "combined-ratified-align"
    )
    if len(combined["ratified_deltas"]) < 4:
        _invalid("combined align golden does not bind multiple ratified deltas")
    segmentation_cases = [
        case for case in cases if case["projector"] == "selected-v2-segmentation"
    ]
    if len(segmentation_cases) != 1:
        _invalid("exactly one full selected-v2 segmentation case is required")
    segmentation = segmentation_cases[0]
    if {output["artifact"] for output in segmentation["expected_paths"]} != {
        "vtt",
        "main-json",
    }:
        _invalid("selected-v2 segmentation lacks its complete primary artifact set")
    segmentation_sizes = {
        output["artifact"]: output["size"] for output in segmentation["expected_paths"]
    }
    if segmentation_sizes["vtt"] < 50 or segmentation_sizes["main-json"] < 500:
        _invalid("selected-v2 segmentation golden is not substantive")

    gates = manifest["gates"]
    if set(gates) != EXPECTED_GATES:
        _invalid("named gate inventory is incomplete or contains unknown gates")
    for name, gate in gates.items():
        unknown = set(gate["case_ids"]) - case_id_set
        if unknown:
            _invalid(f"gate {name} references unknown cases: {sorted(unknown)}")
    align_gate = gates["G-ALIGN-AO"]
    if tuple(align_gate.get("phase_order", ())) != AO_PHASES:
        _invalid("G-ALIGN-AO does not record the exact AO-01 through AO-25 order")
    if set(align_gate.get("runtime_routes", ())) != {
        "ctc-full",
        "mms-full",
        "qwen-crop",
    }:
        _invalid("G-ALIGN-AO runtime-route inventory is incomplete")
    if set(align_gate.get("selected_families", ())) != {
        "legacy-v1",
        "boundary-v2",
    }:
        _invalid("G-ALIGN-AO selected-family inventory is incomplete")
    scenarios = align_gate.get("runtime_scenarios", ())
    if not isinstance(scenarios, list):
        _invalid("G-ALIGN-AO runtime scenario inventory is not an array")
    scenario_ids = [scenario["id"] for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)) or set(scenario_ids) != set(
        EXPECTED_RUNTIME_SCENARIOS
    ):
        _invalid("G-ALIGN-AO runtime scenario inventory is incomplete")
    for scenario in scenarios:
        scenario_id = scenario["id"]
        observed_law = {
            name: scenario[name]
            for name in (
                "route",
                "expected_family",
                "shadow_requested",
                "injections",
                "expect_failure",
            )
        }
        if observed_law != EXPECTED_RUNTIME_SCENARIOS[scenario_id]:
            _invalid(f"G-ALIGN-AO scenario law differs: {scenario_id}")
        _validate_file_fact(
            oracle_root,
            scenario["fixture"],
            label=f"G-ALIGN-AO.{scenario_id}.fixture",
        )
        scenario_fixture_path = _resolve_under(
            oracle_root,
            scenario["fixture"]["path"],
            label=f"G-ALIGN-AO.{scenario_id}.fixture.path",
        )
        scenario_fixture = _read_json(scenario_fixture_path)
        if (
            not isinstance(scenario_fixture, Mapping)
            or scenario_fixture.get("route") != scenario["route"]
        ):
            _invalid(f"G-ALIGN-AO scenario fixture route differs: {scenario_id}")

    for name, matrix in manifest["matrices"].items():
        vector_ids = [vector["id"] for vector in matrix["vectors"]]
        if len(vector_ids) != len(set(vector_ids)):
            _invalid(f"matrix {name} has duplicate vector IDs")

    for case in cases:
        case_id = case["id"]
        _validate_environment(case["environment"], case_id=case_id)
        runtime = cast(Mapping[str, Any], case["public_runtime"])
        expected_trace = runtime["expected_trace"]
        expected_w1_usable = runtime["expected_w1_usable"]
        current_align = (
            case["command"] == "align" and case["reference_set"] != "6e6033f"
        )
        if current_align != (expected_trace is not None):
            _invalid(f"case {case_id} has inconsistent public runtime trace law")
        if current_align != (type(expected_w1_usable) is bool):
            _invalid(f"case {case_id} has inconsistent W1 audit expectation")
        if current_align:
            required_phases = (
                AO_PHASES
                if runtime["expected_family"] == "boundary-v2"
                else tuple(
                    phase for phase in AO_PHASES if phase not in {"AO-15", "AO-17"}
                )
            )
            if tuple(event["phase"] for event in expected_trace) != required_phases:
                _invalid(
                    f"case {case_id} public trace does not pin every applicable phase"
                )
        _validate_public_fixture(case, oracle_root=oracle_root)
        if (case["logical_media"] is None) != (case["route"] == "media-free"):
            _invalid(f"case {case_id} has inconsistent media-free route facts")
        for index, fact in enumerate(case["input_files"]):
            _validate_file_fact(oracle_root, fact, label=f"{case_id}.input[{index}]")
        _validate_file_fact(
            oracle_root, case["backend_receipt"], label=f"{case_id}.backend"
        )
        optimizer = case["optimizer_receipt"]
        if optimizer is not None:
            _validate_file_fact(oracle_root, optimizer, label=f"{case_id}.optimizer")
        profile = case["authority_limit_profile"]
        if _canonical_digest(profile["values"]) != profile["digest"]:
            _invalid(f"case {case_id} authority limit-profile digest differs")
        if (profile["kind"] == "test-qualified") != (
            profile["test_case_id"] is not None
        ):
            _invalid(f"case {case_id} qualification metadata is inconsistent")
        output_sets = {
            "detached": case["expected_paths"],
            "public-command": runtime["expected_paths"],
        }
        detached_artifacts = {output["artifact"] for output in output_sets["detached"]}
        public_artifacts = {
            output["artifact"] for output in output_sets["public-command"]
        }
        if detached_artifacts != public_artifacts:
            _invalid(f"case {case_id} public and detached artifact sets differ")
        for authority, outputs in output_sets.items():
            artifacts = [output["artifact"] for output in outputs]
            if len(artifacts) != len(set(artifacts)):
                _invalid(f"case {case_id} declares a duplicate {authority} artifact")
            for index, output in enumerate(outputs):
                expected = _resolve_under(
                    oracle_root,
                    output["expected_path"],
                    label=f"{case_id}.{authority}.expected[{index}]",
                )
                if not expected.is_file():
                    _invalid(f"expected oracle output is missing: {expected}")
                if expected.stat().st_size != output["size"]:
                    _invalid(f"expected oracle output size differs: {expected}")
                if _sha256_file(expected) != output["sha256"]:
                    _invalid(f"expected oracle output digest differs: {expected}")


def _load_checked_manifest(path: Path) -> Mapping[str, Any]:
    manifest = _schema_validate(_read_json(path))
    _validate_manifest_semantics(manifest, manifest_path=path)
    return manifest


def _tree_files(root: Path) -> set[Path]:
    try:
        return {path.resolve() for path in root.rglob("*") if path.is_file()}
    except OSError as exc:
        _invalid(f"cannot inventory {root}: {exc}")


_VTT_TIME = re.compile(
    r"^(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2})\.(?P<sms>\d{3})"
    r" --> "
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2})\.(?P<ems>\d{3})$"
)


def _vtt_cues(value: bytes) -> list[tuple[float, str, str, str]]:
    try:
        lines = value.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        _invalid(f"detached VTT projector cannot decode its input: {exc}")
    if not lines or lines[0] != "WEBVTT":
        _invalid("detached VTT projector requires a WEBVTT input")
    cues: list[tuple[float, str, str, str]] = []
    index = 1
    while index < len(lines):
        if not lines[index]:
            index += 1
            continue
        match = _VTT_TIME.fullmatch(lines[index])
        if match is None or index + 1 >= len(lines):
            _invalid("detached VTT projector found a malformed cue")
        start = (
            int(match["sh"]) * 3600
            + int(match["sm"]) * 60
            + int(match["ss"])
            + int(match["sms"]) / 1000
        )
        cues.append((start, lines[index], lines[index + 1], match.group(0)))
        index += 2
    if not cues:
        _invalid("detached VTT projector found no cues")
    return cues


def _render_vtt(cues: Sequence[tuple[float, str, str, str]]) -> bytes:
    rows = ["WEBVTT", ""]
    for _start, timing, text, _raw in cues:
        rows.extend((timing, text, ""))
    return "\n".join(rows).encode("utf-8")


def _compact_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _primary_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        _invalid(f"detached primary projector cannot encode its delivery: {exc}")


def _evidence_json(value: object) -> bytes:
    return _primary_json(value)


def _timestamp(seconds: object) -> str:
    if type(seconds) not in (int, float):
        _invalid("detached selected delivery has a nonnumeric timestamp")
    value = float(cast(int | float, seconds))
    if not (value >= 0.0 and value < float("inf")):
        _invalid("detached selected delivery has an invalid timestamp")
    milliseconds = round(value * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _closed_mapping(
    value: object, expected_keys: Sequence[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid(f"{label} is not an object")
    if tuple(value) != tuple(expected_keys):
        _invalid(f"{label} does not have the exact closed key order")
    return value


def _closed_sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _invalid(f"{label} is not an array")
    return value


def _load_delivery(
    case: Mapping[str, Any], oracle_root: Path, *, basename: str
) -> Mapping[str, Any]:
    path = next(
        (
            _resolve_under(oracle_root, fact["path"], label=f"{case['id']}.input")
            for fact in case["input_files"]
            if Path(fact["path"]).name == basename
        ),
        None,
    )
    if path is None:
        _invalid(f"case {case['id']} does not have one input named {basename}")
    value = _read_json(path)
    if not isinstance(value, Mapping):
        _invalid(f"detached delivery {basename} is not an object")
    return value


def _voice_text(text: str, decoration: Mapping[str, Any]) -> str:
    speaker = decoration["speaker"]
    speakers = decoration["speakers"]
    if speaker is not None:
        if type(speaker) is not str or speakers is not None:
            _invalid("selected align cue has invalid speaker decoration")
        return f"<v {speaker}>{text}</v>"
    if speakers is None:
        return text
    rows = _closed_sequence(speakers, label="selected align line speakers")
    lines = text.split("\n")
    if len(rows) != len(lines):
        _invalid("selected align line-speaker cardinality differs from cue lines")
    rendered: list[str] = []
    for line, row in zip(lines, rows, strict=True):
        if not isinstance(row, list) or len(row) != 2:
            _invalid("selected align line-speaker row is malformed")
        name, bound_text = row
        if bound_text != line or (name is not None and type(name) is not str):
            _invalid("selected align line-speaker row is not content-bound")
        rendered.append(line if name is None else f"<v {name}>{line}</v>")
    return "\n".join(rendered)


def _align_primaries(document: Mapping[str, Any]) -> tuple[bytes, bytes]:
    top = _closed_mapping(
        document,
        ("delivery", "projection_inputs", "evidence_core"),
        label="selected align input",
    )
    delivery = _closed_mapping(
        top["delivery"],
        (
            "context_content_digest",
            "receipt_digest",
            "engine_family",
            "route_kind",
            "cues",
            "word_segments",
        ),
        label="selected align delivery",
    )
    inputs = _closed_mapping(
        top["projection_inputs"],
        (
            "language",
            "source_blocks",
            "vad_speech",
            "shot_changes",
            "sing_spans",
            "speaker_turns",
            "voiceprint_capture",
            "voiceprint_media",
            "segmentation",
        ),
        label="selected align projection inputs",
    )
    if delivery["engine_family"] != "boundary-v2":
        _invalid("selected align delivery is not boundary-v2")
    source_blocks = _closed_sequence(
        inputs["source_blocks"], label="selected align source blocks"
    )
    decorations: dict[int, Mapping[str, Any]] = {}
    for row in source_blocks:
        decoration = _closed_mapping(
            row,
            ("source_index", "speaker", "speakers"),
            label="selected align source decoration",
        )
        source_index = decoration["source_index"]
        if type(source_index) is not int or source_index in decorations:
            _invalid("selected align source decorations are not uniquely indexed")
        decorations[source_index] = decoration

    cues = _closed_sequence(delivery["cues"], label="selected align cues")
    if not cues:
        _invalid("selected align delivery has no cues")
    vtt_rows = ["WEBVTT", ""]
    segments: list[dict[str, Any]] = []
    observed_indices: list[int] = []
    for row in cues:
        cue = _closed_mapping(
            row,
            (
                "source_index",
                "text",
                "start",
                "end",
                "lyric",
                "unit_ids",
                "word_data",
                "speech_start",
                "speech_end",
            ),
            label="selected align cue",
        )
        source_index = cue["source_index"]
        text = cue["text"]
        if type(source_index) is not int or type(text) is not str:
            _invalid("selected align cue identity is malformed")
        if source_index not in decorations or source_index in observed_indices:
            _invalid(
                "selected align cue source order is not a unique delivery projection"
            )
        observed_indices.append(source_index)
        start, end = cue["start"], cue["end"]
        if type(start) not in (int, float) or type(end) not in (int, float):
            _invalid("selected align cue bounds are not numeric")
        if float(start) > float(end):
            _invalid("selected align cue bounds are reversed")
        display = f"♪ {text} ♪" if cue["lyric"] is True else text
        display = _voice_text(display, decorations[source_index])
        vtt_rows.extend((f"{_timestamp(start)} --> {_timestamp(end)}", display, ""))
        segment: dict[str, Any] = {"text": text, "start": start, "end": end}
        if cue["lyric"] is True:
            segment["lyric"] = True
        segments.append(segment)
    if set(observed_indices) != set(decorations):
        _invalid("selected align source decorations do not exactly cover delivery cues")

    units = _closed_sequence(
        delivery["word_segments"], label="selected align word segments"
    )
    main: dict[str, Any] = {
        "language": inputs["language"],
        "segments": segments,
        "word_segments": [
            {
                "text": _closed_mapping(
                    unit,
                    ("text", "start", "end"),
                    label="selected align persisted unit",
                )["text"],
                "start": unit["start"],
                "end": unit["end"],
            }
            for unit in units
        ],
    }
    for name in ("vad_speech", "shot_changes", "sing_spans"):
        if inputs[name] is not None:
            main[name] = inputs[name]
    turns = _closed_mapping(
        inputs["speaker_turns"],
        ("present", "value"),
        label="selected align speaker_turns carrier",
    )
    if turns["present"] is True:
        main["speaker_turns"] = turns["value"]
    elif turns["present"] is not False or turns["value"] is not None:
        _invalid("selected align absent speaker_turns carrier has a value")
    capture, media = inputs["voiceprint_capture"], inputs["voiceprint_media"]
    if (capture is None) != (media is None):
        _invalid("selected align voiceprint carriers are not a pair")
    if capture is not None:
        main["voiceprint_capture"] = capture
        main["voiceprint_media"] = media
    if inputs["segmentation"] is not None:
        main["segmentation"] = inputs["segmentation"]
    return ("\n".join(vtt_rows).rstrip() + "\n").encode("utf-8"), _primary_json(main)


def _align_evidence(
    document: Mapping[str, Any], *, vtt_bytes: bytes, main_json_bytes: bytes
) -> bytes:
    delivery = document["delivery"]
    core = _closed_mapping(
        document["evidence_core"],
        (
            "schema_version",
            "context_content_digest",
            "receipt_digest",
            "language",
            "route",
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
        ),
        label="selected align evidence core",
    )
    if (
        core["schema_version"] != 8
        or core["context_content_digest"] != delivery["context_content_digest"]
        or core["receipt_digest"] != delivery["receipt_digest"]
        or core["route"] != delivery["route_kind"]
        or core["language"] != document["projection_inputs"]["language"]
        or core["v2_admission_status"] != "valid"
    ):
        _invalid("selected align evidence core does not cross-link its delivery")
    calls = _closed_sequence(core["physical_calls"], label="physical calls")
    if not calls:
        _invalid("selected align evidence has no physical call")
    for row in calls:
        call = _closed_mapping(
            row,
            (
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
                "legacy_slice_sha256",
                "legacy_absolute_sha256",
                "authority_transform_status",
                "authority_absolute_sha256",
                "raw_unit_ids",
            ),
            label="selected align physical call",
        )
        sample_start = call["sample_start"]
        sample_end = call["sample_end"]
        sample_rate = call["sample_rate"]
        if (
            type(sample_start) is not int
            or type(sample_end) is not int
            or type(sample_rate) is not int
            or sample_rate <= 0
            or not 0 <= sample_start < sample_end
            or call["physical_origin_seconds"] != sample_start / sample_rate
            or call["authority_origin_seconds"] != sample_start / sample_rate
        ):
            _invalid("selected align physical call has unsafe sample geometry")
        origin_kind = call["legacy_origin_kind"]
        if origin_kind == "identity" and (
            sample_start != 0 or call["legacy_origin_seconds"] != 0.0
        ):
            _invalid("selected align identity call is not exact zero")
        if origin_kind == "sample-origin" and (
            call["legacy_origin_seconds"] != sample_start / sample_rate
        ):
            _invalid("selected align sample-origin call does not match geometry")
        if origin_kind == "nominal-route" and core["route"] != "qwen-crop":
            _invalid("selected align nominal origin appears outside Qwen")
    value = {
        "schema_version": core["schema_version"],
        "kind": "fresh-alignment",
        **{name: member for name, member in core.items() if name != "schema_version"},
        "selected_outputs": {
            "engine_family": delivery["engine_family"],
            "vtt_present": True,
            "vtt_sha256": _sha256_bytes(vtt_bytes),
            "json_present": True,
            "json_sha256": _sha256_bytes(main_json_bytes),
        },
    }
    return _evidence_json(value)


def _project_selected_align(
    case: Mapping[str, Any], oracle_root: Path, *, basename: str
) -> dict[str, bytes]:
    document = _load_delivery(case, oracle_root, basename=basename)
    vtt_bytes, main_json_bytes = _align_primaries(document)
    return {
        "vtt": vtt_bytes,
        "main-json": main_json_bytes,
        "align-evidence": _align_evidence(
            document, vtt_bytes=vtt_bytes, main_json_bytes=main_json_bytes
        ),
    }


def _project_selected_segmentation(
    case: Mapping[str, Any], oracle_root: Path
) -> dict[str, bytes]:
    document = _closed_mapping(
        _load_delivery(
            case, oracle_root, basename="selected-v2-segmentation-delivery.json"
        ),
        ("delivery", "projection_inputs"),
        label="selected segmentation input",
    )
    delivery = _closed_mapping(
        document["delivery"],
        (
            "context_content_digest",
            "engine_family",
            "language",
            "terminal",
            "cues",
            "top_level_word_segments",
            "carriers",
            "manifest",
        ),
        label="selected segmentation delivery",
    )
    projection = _closed_mapping(
        document["projection_inputs"],
        ("timestamps", "speaker_names"),
        label="selected segmentation projection inputs",
    )
    if delivery["engine_family"] != "boundary-v2":
        _invalid("selected segmentation delivery is not boundary-v2")
    names_rows = _closed_sequence(
        projection["speaker_names"], label="selected segmentation speaker names"
    )
    names: dict[str, str] = {}
    for row in names_rows:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(type(member) is str for member in row)
            or row[0] in names
        ):
            _invalid("selected segmentation speaker mapping is malformed")
        names[row[0]] = row[1]
    vtt_rows = ["WEBVTT", ""]
    segments: list[dict[str, Any]] = []
    cues = _closed_sequence(delivery["cues"], label="selected segmentation cues")
    previous_high = 0
    for row in cues:
        cue = _closed_mapping(
            row,
            (
                "unit_range",
                "text",
                "start",
                "end",
                "word_data",
                "speech_start",
                "speech_end",
                "lyric",
                "speaker_ids",
            ),
            label="selected segmentation cue",
        )
        unit_range = cue["unit_range"]
        if (
            not isinstance(unit_range, list)
            or len(unit_range) != 2
            or type(unit_range[0]) is not int
            or type(unit_range[1]) is not int
            or unit_range[0] != previous_high
            or unit_range[1] <= unit_range[0]
        ):
            _invalid("selected segmentation unit ranges are not a contiguous tiling")
        previous_high = unit_range[1]
        text = cue["text"]
        if type(text) is not str:
            _invalid("selected segmentation cue text is malformed")
        display = f"♪ {text} ♪" if cue["lyric"] is True else text
        speaker_ids = cue["speaker_ids"]
        if speaker_ids:
            ids = _closed_sequence(
                speaker_ids, label="selected segmentation speaker IDs"
            )
            resolved = [names.get(speaker_id) for speaker_id in ids]
            if len(resolved) == 1 and resolved[0] is not None:
                display = f"<v {resolved[0]}>{display}</v>"
            elif len(resolved) > 1:
                lines = display.split("\n")
                if len(lines) != len(resolved):
                    _invalid("selected segmentation line speakers do not match cue")
                display = "\n".join(
                    line if name is None else f"<v {name}>{line}</v>"
                    for line, name in zip(lines, resolved, strict=True)
                )
        if projection["timestamps"] is True:
            vtt_rows.append(f"{_timestamp(cue['start'])} --> {_timestamp(cue['end'])}")
        elif projection["timestamps"] is not False:
            _invalid("selected segmentation timestamps flag is not boolean")
        vtt_rows.extend((display, ""))
        segment: dict[str, Any] = {
            "text": text,
            "start": cue["start"],
            "end": cue["end"],
            "word_data": cue["word_data"],
        }
        if cue["lyric"] is not None:
            segment["lyric"] = cue["lyric"]
        segments.append(segment)
    top_units = _closed_sequence(
        delivery["top_level_word_segments"],
        label="selected segmentation top-level units",
    )
    if previous_high != len(top_units):
        _invalid("selected segmentation cues do not cover top-level units")
    carriers = _closed_mapping(
        delivery["carriers"],
        (
            "vad_speech",
            "shot_changes",
            "sing_spans",
            "speaker_turns",
            "voiceprint_capture",
            "voiceprint_media",
        ),
        label="selected segmentation carriers",
    )
    main: dict[str, Any] = {
        "language": delivery["language"],
        "segments": segments,
        "word_segments": top_units,
        "vad_speech": carriers["vad_speech"],
    }
    for name in ("shot_changes", "sing_spans"):
        if carriers[name] is not None:
            main[name] = carriers[name]
    turns = _closed_mapping(
        carriers["speaker_turns"],
        ("present", "value"),
        label="selected segmentation speaker_turns",
    )
    if turns["present"] is True:
        main["speaker_turns"] = turns["value"]
    elif turns["present"] is not False or turns["value"] is not None:
        _invalid("selected segmentation absent speaker_turns carrier has a value")
    capture, media = carriers["voiceprint_capture"], carriers["voiceprint_media"]
    if (capture is None) != (media is None):
        _invalid("selected segmentation voiceprint carriers are not a pair")
    if capture is not None:
        main["voiceprint_capture"] = capture
        main["voiceprint_media"] = media
    main["segmentation"] = delivery["manifest"]
    return {
        "vtt": ("\n".join(vtt_rows).rstrip() + "\n").encode("utf-8"),
        "main-json": _primary_json(main),
    }


def _case_input(case: Mapping[str, Any], oracle_root: Path, *, basename: str) -> bytes:
    matches = [
        _resolve_under(oracle_root, fact["path"], label=f"{case['id']}.input")
        for fact in case["input_files"]
        if Path(fact["path"]).name == basename
    ]
    if len(matches) != 1:
        _invalid(f"case {case['id']} does not have one input named {basename}")
    try:
        return matches[0].read_bytes()
    except OSError as exc:
        _invalid(f"cannot read detached projector input {matches[0]}: {exc}")


def _project_case(case: Mapping[str, Any], oracle_root: Path) -> dict[str, bytes]:
    """Project one candidate wholly inside this detached stdlib-only runner."""

    projector = case["projector"]
    if projector == "historical-vtt-time-order":
        cues = _vtt_cues(_case_input(case, oracle_root, basename="lexical-poison.vtt"))
        return {"vtt": _render_vtt(sorted(cues, key=lambda cue: cue[0]))}
    if projector == "post-p11-speaker-turns":
        raw = _case_input(case, oracle_root, basename="lexical-poison.json")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _invalid(f"post-P11 detached projector input is invalid: {exc}")
        return {
            "main-json": _compact_json(
                {
                    "language": document["language"],
                    "speaker_turns": document["speaker_turns"],
                }
            )
        }
    if projector == "p6-lexical-call":
        cues = _vtt_cues(_case_input(case, oracle_root, basename="lexical-poison.vtt"))
        bounds: list[list[float]] = []
        texts: list[str] = []
        for _start, _timing, text, raw_timing in cues:
            match = _VTT_TIME.fullmatch(raw_timing)
            assert match is not None
            start = (
                int(match["sh"]) * 3600
                + int(match["sm"]) * 60
                + int(match["ss"])
                + int(match["sms"]) / 1000
            )
            end = (
                int(match["eh"]) * 3600
                + int(match["em"]) * 60
                + int(match["es"])
                + int(match["ems"]) / 1000
            )
            bounds.append([start, end])
            texts.append(text)
        return {
            "route-evidence": _compact_json(
                {"bounds": bounds, "route": case["route"], "texts": texts}
            )
        }
    if projector == "selected-v2-align":
        return _project_selected_align(
            case, oracle_root, basename="selected-v2-align-delivery.json"
        )
    if projector == "combined-ratified-align":
        return _project_selected_align(
            case, oracle_root, basename="combined-ratified-align-delivery.json"
        )
    if projector == "selected-v2-segmentation":
        return _project_selected_segmentation(case, oracle_root)
    _invalid(f"unknown detached projector: {projector}")


def _apply_runtime_mutation(source_root: Path, name: str | None) -> None:
    if name is None:
        return
    if name != "ao10-after-ao11":  # pragma: no cover - guarded by context manager
        _invalid(f"unknown copied-source runtime mutation: {name}")
    path = source_root / "voxweave" / "pipeline.py"
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _invalid(f"cannot read copied pipeline for mutation: {exc}")
    seal = "        acquisition = seal_fresh_alignment(fresh_session)\n"
    anchor = '        with align_runtime_activity("AO-10", "group-block-spans"):\n'
    if source.count(seal) != 1 or source.count(anchor) != 1:
        _invalid("copied pipeline does not expose the reviewed AO-10/AO-11 seam")
    source = source.replace(seal, "", 1).replace(anchor, seal + anchor, 1)
    try:
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(source)
    except OSError as exc:
        _invalid(f"cannot apply copied-source runtime mutation: {exc}")


def _copy_public_source(source_root: Path, *, historical: bool) -> None:
    if not historical:
        try:
            shutil.copytree(REPO_ROOT / "voxweave", source_root / "voxweave")
        except OSError as exc:
            _invalid(f"cannot copy current public-command source: {exc}")
        return
    try:
        archived = subprocess.run(
            ["git", "archive", REFERENCE_COMMITS["historical"], "voxweave"],
            cwd=REPO_ROOT,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        _invalid(f"cannot archive historical public-command source: {exc}")
    if archived.returncode != 0:
        _invalid("historical public-command source archive is unavailable")
    source_root.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
            archive.extractall(source_root, filter="data")
    except (OSError, tarfile.TarError) as exc:
        _invalid(f"cannot extract historical public-command source: {exc}")


def _public_artifact_path(
    artifact: str,
    *,
    episode_root: Path,
    route_evidence_path: Path,
) -> Path:
    paths = {
        "vtt": episode_root / "episode.vtt",
        "main-json": episode_root / "episode.json",
        "align-evidence": episode_root / "episode.align-evidence.json",
        "route-evidence": route_evidence_path,
    }
    try:
        return paths[artifact]
    except KeyError:
        _invalid(f"unknown public artifact: {artifact}")


def _public_worker_environment(
    recorded: Mapping[str, str | None], *, source_root: Path
) -> dict[str, str]:
    """Build the closed environment used by isolated public-command workers."""

    environment = {"PATH": os.environ.get("PATH", os.defpath)}
    for name, value in recorded.items():
        if value is not None:
            environment[name] = value
    environment["PYTHONPATH"] = str(source_root)
    return environment


def _execute_public_case(
    case: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> PublicCaseResult:
    """Run one recorded CLI command against copied production in a clean root."""

    oracle_root = manifest_path.resolve().parent
    runtime = cast(Mapping[str, Any], case["public_runtime"])
    fixture_path = _resolve_under(
        oracle_root,
        runtime["fixture"],
        label=f"{case['id']}.public_runtime.fixture",
    )
    fixture = _read_json(fixture_path)
    if not isinstance(fixture, Mapping):
        _invalid(f"case {case['id']} public runtime fixture is not an object")

    historical = case["reference_set"] == "6e6033f"
    with tempfile.TemporaryDirectory(prefix=f"p6-oracle-{case['id']}-") as raw_root:
        isolated_root = Path(raw_root)
        source_root = isolated_root / "source"
        episode_root = isolated_root / "episode"
        worker_path = source_root / "p6_oracle_public.py"
        request_path = isolated_root / "request.json"
        trace_path = isolated_root / "trace.json"
        verification_path = isolated_root / "evidence-verification.json"
        outcome_path = isolated_root / "outcome.json"
        route_evidence_path = isolated_root / "route-evidence.json"
        try:
            _copy_public_source(source_root, historical=historical)
            shutil.copy2(SCRIPTS_DIR / "p6_oracle_public.py", worker_path)
        except OSError as exc:
            _invalid(f"cannot construct isolated public-command source: {exc}")
        if not historical:
            _apply_runtime_mutation(source_root, _RUNTIME_MUTATION.get())
        request = {
            "arguments": list(case["arguments"]),
            "case_id": case["id"],
            "command": case["command"],
            "expected_family": runtime["expected_family"],
            "fixture": fixture,
            "historical": historical,
            "shadow_requested": case["id"] == "selected-v2-segmentation",
        }
        try:
            cc.write_json(request_path, request)
        except OSError as exc:
            _invalid(f"cannot write isolated public-command request: {exc}")
        environment = _public_worker_environment(
            case["environment"], source_root=source_root
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker_path),
                    "--request",
                    str(request_path),
                    "--episode-root",
                    str(episode_root),
                    "--trace-out",
                    str(trace_path),
                    "--evidence-verification-out",
                    str(verification_path),
                    "--outcome-out",
                    str(outcome_path),
                    "--route-evidence-out",
                    str(route_evidence_path),
                ],
                cwd=isolated_root,
                check=False,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _invalid(f"public command {case['id']} could not execute: {exc}")
        if completed.returncode != 0:
            excerpt = completed.stdout[-4000:].strip()
            _invalid(
                f"public command {case['id']} exited {completed.returncode}: {excerpt}"
            )

        runtime_trace = _read_json(trace_path)
        if runtime_trace is not None and not isinstance(runtime_trace, Mapping):
            _invalid(f"public command {case['id']} emitted a nonobject runtime trace")
        evidence_verification = _read_json(verification_path)
        if evidence_verification is not None and not isinstance(
            evidence_verification, Mapping
        ):
            _invalid(
                f"public command {case['id']} emitted invalid evidence verification"
            )
        outcome = _read_json(outcome_path)
        if outcome != {"exception_class": None, "success": True}:
            _invalid(
                f"public command {case['id']} did not report successful completion"
            )
        artifacts: dict[str, bytes] = {}
        for output in runtime["expected_paths"]:
            artifact = output["artifact"]
            path = _public_artifact_path(
                artifact,
                episode_root=episode_root,
                route_evidence_path=route_evidence_path,
            )
            try:
                artifacts[artifact] = path.read_bytes()
            except OSError as exc:
                _invalid(
                    f"public command {case['id']} did not produce {artifact}: {exc}"
                )
        return PublicCaseResult(
            artifacts=artifacts,
            case_id=case["id"],
            command=case["command"],
            episode_root=episode_root,
            evidence_verification=cast(
                Mapping[str, Any] | None,
                evidence_verification,
            ),
            runtime_trace=cast(Mapping[str, Any] | None, runtime_trace),
            source_root=source_root,
        )


def _execute_runtime_scenario(
    scenario: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> RuntimeScenarioResult:
    oracle_root = manifest_path.resolve().parent
    scenario_id = scenario["id"]
    fixture_path = _resolve_under(
        oracle_root,
        scenario["fixture"]["path"],
        label=f"G-ALIGN-AO.{scenario_id}.fixture.path",
    )
    fixture = _read_json(fixture_path)
    if not isinstance(fixture, Mapping):
        _invalid(f"G-ALIGN-AO scenario fixture is not an object: {scenario_id}")
    with tempfile.TemporaryDirectory(prefix=f"p6-oracle-{scenario_id}-") as raw_root:
        isolated_root = Path(raw_root)
        source_root = isolated_root / "source"
        episode_root = isolated_root / "episode"
        worker_path = source_root / "p6_oracle_public.py"
        request_path = isolated_root / "request.json"
        trace_path = isolated_root / "trace.json"
        verification_path = isolated_root / "evidence-verification.json"
        outcome_path = isolated_root / "outcome.json"
        route_evidence_path = isolated_root / "route-evidence.json"
        try:
            shutil.copytree(REPO_ROOT / "voxweave", source_root / "voxweave")
            shutil.copy2(SCRIPTS_DIR / "p6_oracle_public.py", worker_path)
        except OSError as exc:
            _invalid(f"cannot construct G-ALIGN-AO scenario source: {exc}")
        _apply_runtime_mutation(source_root, _RUNTIME_MUTATION.get())
        request = {
            "arguments": ["align", "episode.vtt", "--media", "episode.wav"],
            "case_id": scenario_id,
            "command": "align",
            "expect_failure": scenario["expect_failure"],
            "expected_family": scenario["expected_family"],
            "fixture": fixture,
            "injections": list(scenario["injections"]),
            "shadow_requested": scenario["shadow_requested"],
        }
        try:
            cc.write_json(request_path, request)
        except OSError as exc:
            _invalid(f"cannot write G-ALIGN-AO scenario request: {exc}")
        environment = _public_worker_environment(
            manifest["cases"][0]["environment"], source_root=source_root
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker_path),
                    "--request",
                    str(request_path),
                    "--episode-root",
                    str(episode_root),
                    "--trace-out",
                    str(trace_path),
                    "--evidence-verification-out",
                    str(verification_path),
                    "--outcome-out",
                    str(outcome_path),
                    "--route-evidence-out",
                    str(route_evidence_path),
                ],
                cwd=isolated_root,
                check=False,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _invalid(f"G-ALIGN-AO scenario {scenario_id} could not execute: {exc}")
        if completed.returncode != 0:
            excerpt = completed.stdout[-4000:].strip()
            _invalid(
                f"G-ALIGN-AO scenario {scenario_id} exited "
                f"{completed.returncode}: {excerpt}"
            )
        trace = _read_json(trace_path)
        outcome = _read_json(outcome_path)
        verification = _read_json(verification_path)
        if not isinstance(trace, Mapping) or not isinstance(outcome, Mapping):
            _invalid(f"G-ALIGN-AO scenario {scenario_id} emitted invalid records")
        if verification is not None and not isinstance(verification, Mapping):
            _invalid(f"G-ALIGN-AO scenario {scenario_id} emitted invalid verification")
        return RuntimeScenarioResult(
            evidence_verification=cast(Mapping[str, Any] | None, verification),
            outcome=cast(Mapping[str, Any], outcome),
            runtime_trace=cast(Mapping[str, Any], trace),
            scenario_id=scenario_id,
        )


def _runtime_ao_failures(
    case: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> list[str]:
    prefix = f"public-command/{case['id']}/G-ALIGN-AO"
    failures: list[str] = []
    if set(trace) != {"schema_version", "route_kind", "engine_family", "events"}:
        return [f"{prefix}: runtime trace is not closed"]
    runtime = cast(Mapping[str, Any], case["public_runtime"])
    if trace["schema_version"] != 1:
        failures.append(f"{prefix}: runtime trace schema differs")
    if trace["route_kind"] != case["route"]:
        failures.append(f"{prefix}: route identity differs")
    if trace["engine_family"] != runtime["expected_family"]:
        failures.append(f"{prefix}: selected family identity differs")
    events = trace["events"]
    if not isinstance(events, list) or not events:
        return failures + [f"{prefix}: runtime event stream is empty"]
    stack: list[tuple[str, str]] = []
    starts: list[tuple[str, str]] = []
    completed: list[dict[str, str]] = []
    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping) or set(raw_event) != {
            "ordinal",
            "phase",
            "activity",
            "state",
        }:
            failures.append(f"{prefix}: event {index} is not closed")
            continue
        phase = raw_event["phase"]
        activity = raw_event["activity"]
        state = raw_event["state"]
        if raw_event["ordinal"] != index:
            failures.append(f"{prefix}: event ordinal {index} differs")
        if phase not in AO_PHASES or type(activity) is not str or not activity:
            failures.append(f"{prefix}: event {index} identity is invalid")
            continue
        key = (phase, activity)
        if state == "started":
            if starts and AO_PHASES.index(phase) < AO_PHASES.index(starts[-1][0]):
                failures.append(f"{prefix}: live start order regressed at {phase}")
            starts.append(key)
            stack.append(key)
        elif state in {"completed", "failed"}:
            if not stack or stack[-1] != key:
                failures.append(f"{prefix}: event lifecycle is not properly nested")
            else:
                stack.pop()
            if state == "completed":
                completed.append({"phase": phase, "activity": activity})
        else:
            failures.append(f"{prefix}: event {index} has unknown state")
    if stack:
        failures.append(f"{prefix}: runtime event lifecycle is incomplete")
    expected = runtime["expected_trace"]
    first_completed: list[dict[str, str]] = []
    completed_phases: set[str] = set()
    for event in completed:
        if event["phase"] in completed_phases:
            continue
        completed_phases.add(event["phase"])
        first_completed.append(event)
    if first_completed != expected:
        failures.append(f"{prefix}: first completed phase activities differ")
    ao10 = [activity for phase, activity in starts if phase == "AO-10"]
    if tuple(ao10) != AO10_HELPER_CHAIN:
        failures.append(f"{prefix}: AO-10 helper chain differs")
    try:
        ao10_end = starts.index(("AO-10", "seal-selected-legacy-result"))
        ao11 = starts.index(("AO-11", "strict-capture"))
    except ValueError:
        failures.append(f"{prefix}: AO-10/AO-11 live activities are incomplete")
    else:
        if ao10_end >= ao11:
            failures.append(f"{prefix}: AO-11 began before AO-10 completed")
    return sorted(set(failures))


def _runtime_scenario_failures(
    scenario: Mapping[str, Any],
    result: RuntimeScenarioResult,
) -> list[str]:
    scenario_id = scenario["id"]
    prefix = f"public-command/G-ALIGN-AO/{scenario_id}"
    trace = result.runtime_trace
    failures: list[str] = []
    if set(trace) != {"schema_version", "route_kind", "engine_family", "events"}:
        return [f"{prefix}: runtime trace is not closed"]
    if trace["schema_version"] != 1:
        failures.append(f"{prefix}: trace schema differs")
    if trace["route_kind"] != scenario["route"]:
        failures.append(f"{prefix}: route identity differs")
    if trace["engine_family"] != scenario["expected_family"]:
        failures.append(f"{prefix}: family identity differs")
    expected_success = not scenario["expect_failure"]
    if result.outcome.get("success") is not expected_success:
        failures.append(f"{prefix}: command outcome differs")
    events = trace["events"]
    if not isinstance(events, list) or not events:
        return failures + [f"{prefix}: event stream is empty"]
    stack: list[tuple[str, str]] = []
    starts: list[tuple[str, str]] = []
    states: set[tuple[str, str, str]] = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != {
            "ordinal",
            "phase",
            "activity",
            "state",
        }:
            failures.append(f"{prefix}: event {index} is not closed")
            continue
        phase, activity, state = event["phase"], event["activity"], event["state"]
        if event["ordinal"] != index:
            failures.append(f"{prefix}: event ordinal {index} differs")
        if phase not in AO_PHASES or type(activity) is not str or not activity:
            failures.append(f"{prefix}: event {index} identity is invalid")
            continue
        key = (phase, activity)
        states.add((phase, activity, state))
        if state == "started":
            if starts and AO_PHASES.index(phase) < AO_PHASES.index(starts[-1][0]):
                failures.append(f"{prefix}: live start order regressed at {phase}")
            starts.append(key)
            stack.append(key)
        elif state in {"completed", "failed"}:
            if not stack or stack[-1] != key:
                failures.append(f"{prefix}: lifecycle is not properly nested")
            else:
                stack.pop()
        else:
            failures.append(f"{prefix}: event {index} state is invalid")
    if stack:
        failures.append(f"{prefix}: lifecycle is incomplete")

    first_started_phases = tuple(dict.fromkeys(phase for phase, _activity in starts))
    expected_started_phases = {
        "mms-legacy-happy": AO_PHASES[:14] + ("AO-16",) + AO_PHASES[17:],
        "qwen-all-skip-legacy": AO_PHASES[:10] + ("AO-24",),
        "qwen-all-skip-boundary": AO_PHASES[:10] + ("AO-24",),
        "ao15-uncontained-boundary": AO_PHASES[:15] + ("AO-24",),
        "paired-ao15-ao16-boundary": AO_PHASES[:15] + ("AO-24",),
        "ao15-isolated-then-ao16-legacy": (AO_PHASES[:16] + AO_PHASES[17:]),
        "paired-ao15-ao16-legacy": AO_PHASES[:16] + ("AO-24",),
    }[scenario_id]
    if first_started_phases != expected_started_phases:
        failures.append(f"{prefix}: first live phase sequence differs")

    def require(phase: str, activity: str, state: str) -> None:
        if (phase, activity, state) not in states:
            failures.append(f"{prefix}: missing {phase}/{activity}/{state}")

    def reject_phases(*phases: str) -> None:
        observed = sorted({phase for phase, _activity in starts} & set(phases))
        if observed:
            failures.append(f"{prefix}: forbidden live phases started: {observed}")

    if scenario_id == "mms-legacy-happy":
        ao10 = tuple(activity for phase, activity in starts if phase == "AO-10")
        if ao10 != AO10_HELPER_CHAIN:
            failures.append(f"{prefix}: AO-10 helper chain differs")
        try:
            ao10_end = starts.index(("AO-10", "seal-selected-legacy-result"))
            ao11 = starts.index(("AO-11", "strict-capture"))
        except ValueError:
            failures.append(f"{prefix}: AO-10/AO-11 live activities are incomplete")
        else:
            if ao10_end >= ao11:
                failures.append(f"{prefix}: AO-11 began before AO-10 completed")
        require("AO-16", "mandatory-evidence-core-and-ald6", "completed")
        require("AO-25", "artifact-and-observer-dispatch", "completed")
        if any(phase == "AO-15" for phase, _activity in starts):
            failures.append(f"{prefix}: unrequested AO-15 started")
        verification = result.evidence_verification
        if (
            verification is None
            or verification.get("detail_code") is not None
            or verification.get("integrity") is not True
            or verification.get("w1_usable") is not True
        ):
            failures.append(f"{prefix}: successful evidence integrity failed")
    elif scenario_id.startswith("qwen-all-skip-"):
        require("AO-10", "group-block-spans", "completed")
        require("AO-10", "common-all-empty-decision", "failed")
        reject_phases(*AO_PHASES[10:23], "AO-25")
    elif scenario_id in {
        "ao15-uncontained-boundary",
        "paired-ao15-ao16-boundary",
    }:
        require("AO-15", "fresh-w1-finalizer-and-validation", "failed")
        reject_phases(
            "AO-16",
            "AO-17",
            "AO-18",
            "AO-19",
            "AO-20",
            "AO-21",
            "AO-22",
            "AO-23",
            "AO-25",
        )
    elif scenario_id == "ao15-isolated-then-ao16-legacy":
        require("AO-15", "fresh-w1-finalizer-and-validation", "completed")
        require("AO-16", "mandatory-evidence-core-and-ald6", "completed")
        reject_phases("AO-17")
        verification = result.evidence_verification
        if (
            verification is None
            or verification.get("detail_code") is not None
            or verification.get("integrity") is not True
            or verification.get("w1_usable") is not True
        ):
            failures.append(f"{prefix}: isolated-path evidence integrity failed")
    elif scenario_id == "paired-ao15-ao16-legacy":
        require("AO-15", "fresh-w1-finalizer-and-validation", "completed")
        require("AO-16", "mandatory-evidence-core-and-ald6", "failed")
        reject_phases(
            "AO-17", "AO-18", "AO-19", "AO-20", "AO-21", "AO-22", "AO-23", "AO-25"
        )
    else:  # pragma: no cover - closed at manifest validation
        failures.append(f"{prefix}: scenario has no validator")
    return sorted(set(failures))


def _compare(manifest: Mapping[str, Any], *, manifest_path: Path) -> list[str]:
    oracle_root = manifest_path.resolve().parent
    expected_declared: set[Path] = set()
    mismatches: list[str] = []
    for case in manifest["cases"]:
        candidates = _project_case(case, oracle_root)
        public = _execute_public_case(case, manifest_path=manifest_path)
        runtime = cast(Mapping[str, Any], case["public_runtime"])
        declared_artifacts = {output["artifact"] for output in case["expected_paths"]}
        public_declared_artifacts = {
            output["artifact"] for output in runtime["expected_paths"]
        }
        if set(candidates) != declared_artifacts:
            mismatches.append(
                f"detached-authority/{case['id']}: artifact set "
                f"{sorted(candidates)} != {sorted(declared_artifacts)}"
            )
        if set(public.artifacts) != public_declared_artifacts:
            mismatches.append(
                f"public-command/{case['id']}: artifact set "
                f"{sorted(public.artifacts)} != {sorted(public_declared_artifacts)}"
            )
        if case["command"] == "align" and case["reference_set"] != "6e6033f":
            verification = public.evidence_verification
            if (
                verification is None
                or verification.get("detail_code") is not None
                or verification.get("integrity") is not True
                or verification.get("w1_usable") is not runtime["expected_w1_usable"]
            ):
                mismatches.append(
                    f"public-command/{case['id']}/align-evidence: "
                    "production integrity verification failed"
                )
            if public.runtime_trace is None:
                mismatches.append(
                    f"public-command/{case['id']}/G-ALIGN-AO: trace is absent"
                )
            else:
                mismatches.extend(_runtime_ao_failures(case, public.runtime_trace))
        for output in case["expected_paths"]:
            artifact = output["artifact"]
            expected = _resolve_under(
                oracle_root, output["expected_path"], label="expected output"
            )
            expected_declared.add(expected)
            try:
                expected_bytes = expected.read_bytes()
            except OSError as exc:
                _invalid(f"cannot compare oracle output: {exc}")
            candidate_bytes = candidates.get(artifact)
            if candidate_bytes is None:
                continue
            if candidate_bytes != expected_bytes:
                authority = (
                    "detached-evidence"
                    if artifact == "align-evidence"
                    else "detached-projector"
                )
                mismatches.append(
                    f"{authority}/{case['id']}/{artifact}: byte mismatch != "
                    f"{expected.relative_to(oracle_root)}"
                )
            if len(candidate_bytes) != output["size"]:
                mismatches.append(
                    f"detached-projector/{case['id']}/{artifact}: size mismatch"
                )
            if _sha256_bytes(candidate_bytes) != output["sha256"]:
                mismatches.append(
                    f"detached-projector/{case['id']}/{artifact}: digest mismatch"
                )
        for output in runtime["expected_paths"]:
            artifact = output["artifact"]
            expected = _resolve_under(
                oracle_root,
                output["expected_path"],
                label="public-command expected output",
            )
            expected_declared.add(expected)
            try:
                expected_bytes = expected.read_bytes()
            except OSError as exc:
                _invalid(f"cannot compare public-command oracle output: {exc}")
            public_bytes = public.artifacts.get(artifact)
            if public_bytes is None:
                continue
            if public_bytes != expected_bytes:
                mismatches.append(
                    f"public-command/{case['id']}/{artifact}: byte mismatch != "
                    f"{expected.relative_to(oracle_root)}"
                )
            if len(public_bytes) != output["size"]:
                mismatches.append(
                    f"public-command/{case['id']}/{artifact}: size mismatch"
                )
            if _sha256_bytes(public_bytes) != output["sha256"]:
                mismatches.append(
                    f"public-command/{case['id']}/{artifact}: digest mismatch"
                )

    expected_tree = _tree_files(oracle_root / "expected")
    for extra in sorted(expected_tree - expected_declared):
        _invalid(f"undeclared expected path: {extra.relative_to(oracle_root)}")
    for missing in sorted(expected_declared - expected_tree):
        _invalid(
            f"declared expected path is absent: {missing.relative_to(oracle_root)}"
        )
    return sorted(set(mismatches))


def _comparison_report(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    failures: Sequence[str],
) -> Mapping[str, Any]:
    return {
        "artifact_count": sum(
            len(case["expected_paths"]) for case in manifest["cases"]
        ),
        "authority_artifact_counts": {
            "detached": sum(len(case["expected_paths"]) for case in manifest["cases"]),
            "public-command": sum(
                len(case["public_runtime"]["expected_paths"])
                for case in manifest["cases"]
            ),
        },
        "case_count": len(manifest["cases"]),
        "command": "compare",
        "failure_count": len(failures),
        "failures": list(failures),
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "runner_version": manifest["runner_version"],
        "schema_version": manifest["schema_version"],
        "status": "mismatch" if failures else "match",
    }


def _write_comparison_report(
    path: Path,
    report: Mapping[str, Any],
    *,
    oracle_root: Path,
) -> None:
    resolved = path.resolve()
    if resolved == oracle_root or oracle_root in resolved.parents:
        _invalid(
            "comparison report cannot be written inside the immutable oracle corpus"
        )
    try:
        cc.write_json(resolved, report)
    except OSError as exc:
        _invalid(f"cannot write comparison report: {exc}")


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets: Sequence[ast.expr]
    if isinstance(node, ast.Assign):
        targets = node.targets
    else:
        targets = (node.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _check_ao_source() -> list[str]:
    path = REPO_ROOT / "voxweave" / "align_orchestration.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        _invalid(f"cannot parse align orchestration source: {exc}")
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and "ALIGN_AO_PHASE_ORDER" in _assignment_names(node)
    ]
    if len(definitions) != 1:
        return ["G-ALIGN-AO requires exactly one ALIGN_AO_PHASE_ORDER definition"]
    node = definitions[0]
    if node.value is None:
        return ["G-ALIGN-AO phase order is not a static literal"]
    try:
        value = ast.literal_eval(node.value)
    except (TypeError, ValueError):
        return ["G-ALIGN-AO phase order is not a static literal"]
    if tuple(value) != AO_PHASES:
        return ["G-ALIGN-AO phase order differs from AO-01 through AO-25"]
    return []


def _test_evidence_entries(manifest: Mapping[str, Any]) -> set[str]:
    evidence = {
        item
        for matrix in manifest["matrices"].values()
        for vector in matrix["vectors"]
        for item in vector["evidence"]
    }
    evidence.update(injection["evidence"] for injection in manifest["injections"])
    evidence.update(
        item
        for details in manifest["failure_registry_coverage"].values()
        for row in details.values()
        for item in row["evidence"]
        if item.startswith("tests/")
    )
    return evidence


def _check_test_evidence(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    parsed: dict[Path, set[str]] = {}
    for reference in sorted(_test_evidence_entries(manifest)):
        path_text, function_name = reference.split("::", maxsplit=1)
        path = _resolve_under(REPO_ROOT, path_text, label="test evidence path")
        names = parsed.get(path)
        if names is None:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                _invalid(f"cannot parse test evidence {path}: {exc}")
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            parsed[path] = names
        if function_name not in names:
            failures.append(f"missing declared test evidence: {reference}")
    return failures


def _execute_test_evidence(manifest: Mapping[str, Any]) -> list[str]:
    """Run every declared vector/injection node, deduplicated, as one real gate."""

    references = sorted(_test_evidence_entries(manifest))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", *references],
            cwd=REPO_ROOT,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"declared executable evidence could not run: {exc}"]
    if result.returncode == 0:
        return []
    tail = "\n".join(result.stdout.splitlines()[-20:])
    return [
        "declared executable evidence failed with "
        f"pytest exit {result.returncode}:\n{tail}"
    ]


def _failure_registry() -> Mapping[str, tuple[str, ...]]:
    path = REPO_ROOT / "voxweave" / "align_failures.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        _invalid(f"cannot parse failure registry: {exc}")
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and "_OUTCOME_DETAILS" in _assignment_names(node)
    ]
    if len(definitions) != 1:
        _invalid(
            "failure registry does not have one static _OUTCOME_DETAILS definition"
        )
    expression = definitions[0].value
    if expression is None:
        _invalid("failure registry is not a static literal")
    try:
        value = ast.literal_eval(expression)
    except (TypeError, ValueError):
        _invalid("failure registry is not a static literal")
    if not isinstance(value, Mapping):
        _invalid("failure registry literal is not an object")
    return value


def _production_string_locations() -> Mapping[str, set[Path]]:
    """Index exact production string literals outside the registry definition."""

    locations: dict[str, set[Path]] = {}
    for path in sorted((REPO_ROOT / "voxweave").rglob("*.py")):
        if "vendor" in path.parts or path.name == "align_failures.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            _invalid(f"cannot parse failure-coverage source {path}: {exc}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is str:
                locations.setdefault(node.value, set()).add(path)
    return locations


def _check_failure_registry_coverage(manifest: Mapping[str, Any]) -> list[str]:
    """Require one proved disposition for every closed failure-registry detail."""

    registry = _failure_registry()
    coverage = manifest["failure_registry_coverage"]
    failures: list[str] = []
    if tuple(coverage) != tuple(registry):
        missing = sorted(set(registry) - set(coverage))
        extra = sorted(set(coverage) - set(registry))
        failures.append(
            f"failure coverage kind inventory differs; missing={missing}, extra={extra}"
        )
    locations = _production_string_locations()
    source_cache: dict[Path, set[str]] = {}
    test_cache: dict[tuple[Path, str], set[str]] = {}
    for kind, registered_details in registry.items():
        rows = coverage.get(kind)
        if not isinstance(rows, Mapping):
            continue
        if tuple(rows) != tuple(registered_details):
            missing = sorted(set(registered_details) - set(rows))
            extra = sorted(set(rows) - set(registered_details))
            failures.append(
                f"failure coverage detail inventory differs for {kind}; "
                f"missing={missing}, extra={extra}"
            )
        for detail in registered_details:
            row = rows.get(detail)
            if not isinstance(row, Mapping):
                continue
            label = f"{kind}/{detail}"
            evidence = row["evidence"]
            if row["status"] == "structural-reserve":
                if evidence:
                    failures.append(
                        f"structural reserve {label} declares live evidence"
                    )
                if detail in locations:
                    rendered = sorted(
                        str(path.relative_to(REPO_ROOT)) for path in locations[detail]
                    )
                    failures.append(
                        f"structural reserve {label} is present in production: {rendered}"
                    )
                continue
            if not evidence:
                failures.append(f"reachable registry row has no evidence: {label}")
                continue
            exact_evidence = False
            for reference in evidence:
                if reference.startswith("tests/"):
                    path_text, function_name = reference.split("::", maxsplit=1)
                    path = _resolve_under(
                        REPO_ROOT,
                        path_text,
                        label=f"failure coverage {label}",
                    )
                    cache_key = (path, function_name)
                    values = test_cache.get(cache_key)
                    if values is None:
                        try:
                            tree = ast.parse(path.read_text(encoding="utf-8"))
                        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                            _invalid(f"cannot parse failure evidence {path}: {exc}")
                        values = {
                            node.value
                            for function in ast.walk(tree)
                            if isinstance(
                                function, (ast.FunctionDef, ast.AsyncFunctionDef)
                            )
                            and function.name == function_name
                            for node in ast.walk(function)
                            if isinstance(node, ast.Constant)
                            and type(node.value) is str
                        }
                        test_cache[cache_key] = values
                    exact_evidence = exact_evidence or detail in values
                    continue
                if not reference.startswith("source:"):
                    failures.append(
                        f"unknown failure evidence for {label}: {reference}"
                    )
                    continue
                path = _resolve_under(
                    REPO_ROOT,
                    reference.removeprefix("source:"),
                    label=f"failure coverage {label}",
                )
                values = source_cache.get(path)
                if values is None:
                    try:
                        tree = ast.parse(path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                        _invalid(f"cannot parse failure evidence {path}: {exc}")
                    values = {
                        node.value
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Constant) and type(node.value) is str
                    }
                    source_cache[path] = values
                if detail not in values:
                    failures.append(
                        f"source evidence does not name {label}: "
                        f"{path.relative_to(REPO_ROOT)}"
                    )
                else:
                    exact_evidence = True
            if not exact_evidence:
                failures.append(
                    f"reachable registry row has no exact evidence: {label}"
                )
    return failures


def _check_injection_registry(manifest: Mapping[str, Any]) -> list[str]:
    registry = _failure_registry()
    failures: list[str] = []
    ids = [injection["id"] for injection in manifest["injections"]]
    if len(ids) != len(set(ids)):
        failures.append("injection IDs are not unique")
    required_amendment = {"machine-artifact-stage", "machine-artifact-replace"}
    if not required_amendment.issubset(ids):
        failures.append("amendment-1 machine-artifact injections are incomplete")
    if {injection["section"] for injection in manifest["injections"]} != {
        "14.5",
        "14.6",
    }:
        failures.append("injections do not cover both sections 14.5 and 14.6")
    for injection in manifest["injections"]:
        kind = injection["expected_kind"]
        detail = injection["expected_detail"]
        if (kind is None) != (detail is None):
            failures.append(f"{injection['id']}: partial canonical failure pair")
        elif kind is not None and detail not in registry.get(kind, ()):
            failures.append(f"{injection['id']}: unregistered {kind}/{detail}")
    return failures


def _literal_argument(node: ast.Call, position: int, keyword: str) -> str | None:
    value: ast.expr | None = node.args[position] if len(node.args) > position else None
    if value is None:
        value = next(
            (item.value for item in node.keywords if item.arg == keyword), None
        )
    return (
        value.value
        if isinstance(value, ast.Constant) and type(value.value) is str
        else None
    )


def _check_literal_source_terminals() -> list[str]:
    """Prove every statically named production terminal has a closed registry row."""

    registry = _failure_registry()
    failures: list[str] = []
    for path in sorted((REPO_ROOT / "voxweave").rglob("*.py")):
        if "vendor" in path.parts or path.name == "align_failures.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            _invalid(f"cannot parse source-terminal input {path}: {exc}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if name not in {"CanonicalFailure", "SecondaryFailure"}:
                continue
            kind = _literal_argument(node, 0, "kind")
            detail = _literal_argument(node, 2, "detail_code")
            if (
                kind is not None
                and detail is not None
                and detail not in registry.get(kind, ())
            ):
                relative = path.relative_to(REPO_ROOT)
                failures.append(
                    f"unregistered source terminal {kind}/{detail} at {relative}:{node.lineno}"
                )
    return failures


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        _invalid(f"cannot parse dependency-gate source {path}: {exc}")
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _check_dependencies() -> list[str]:
    rules = {
        "voxweave/align_evidence_core.py": {
            "voxweave.core.finalizer",
            "voxweave.core.align_compare",
            "voxweave.align_projector",
            "voxweave.segmentation_projector",
            "voxweave.candidate_encoder",
            "voxweave.pipeline",
        },
        "voxweave/core/align_compare.py": {
            "voxweave.align_evidence_core",
            "voxweave.align_projector",
            "voxweave.segmentation_projector",
            "voxweave.candidate_encoder",
            "voxweave.pipeline",
        },
        "voxweave/reference_projector.py": {
            "voxweave.align_projector",
            "voxweave.segmentation_projector",
            "voxweave.candidate_encoder",
            "voxweave.episode_transaction",
            "voxweave.pipeline",
        },
        "voxweave/episode_transaction.py": {
            "voxweave.backend",
            "voxweave.candidate_encoder",
            "voxweave.pipeline",
        },
        "voxweave/align_dp_safety.py": {
            "voxweave.backend",
            "voxweave.pipeline",
        },
    }
    failures: list[str] = []
    for relative, forbidden in rules.items():
        observed = _imports(REPO_ROOT / relative)
        collisions = sorted(observed & forbidden)
        if collisions:
            failures.append(f"dependency violation in {relative}: {collisions}")
    return failures


def _public_runtime_gate(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> list[str]:
    failures: list[str] = []
    for case in manifest["cases"]:
        if case["command"] != "align" or case["reference_set"] == "6e6033f":
            continue
        result = _execute_public_case(case, manifest_path=manifest_path)
        if result.runtime_trace is None:
            failures.append(f"public-command/{case['id']}/G-ALIGN-AO: trace is absent")
            continue
        failures.extend(_runtime_ao_failures(case, result.runtime_trace))
    scenarios = manifest["gates"]["G-ALIGN-AO"]["runtime_scenarios"]
    for scenario in scenarios:
        result = _execute_runtime_scenario(
            scenario,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        failures.extend(_runtime_scenario_failures(scenario, result))
    return failures


def _source_gates(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
) -> list[str]:
    failures = _check_ao_source()
    failures.extend(
        _public_runtime_gate(
            manifest,
            manifest_path=(
                REPO_ROOT / "calibration" / "p6-oracle" / "manifest.json"
                if manifest_path is None
                else manifest_path
            ),
        )
    )
    failures.extend(_check_test_evidence(manifest))
    failures.extend(_execute_test_evidence(manifest))
    failures.extend(_check_injection_registry(manifest))
    failures.extend(_check_failure_registry_coverage(manifest))
    failures.extend(_check_literal_source_terminals())
    failures.extend(_check_dependencies())
    for check in manifest["source_checks"]:
        path = _resolve_under(REPO_ROOT, check["path"], label=f"{check['id']}.path")
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _invalid(f"cannot read source gate input {path}: {exc}")
        for token in check.get("must_contain", ()):
            if token not in source:
                failures.append(f"{check['id']}: required token is absent: {token!r}")
        for token in check.get("must_not_contain", ()):
            if token in source:
                failures.append(f"{check['id']}: forbidden token is present: {token!r}")
        cursor = -1
        for token in check.get("ordered", ()):
            location = source.find(token, cursor + 1)
            if location < 0:
                failures.append(
                    f"{check['id']}: ordered token is absent or out of order: {token!r}"
                )
                break
            cursor = location
        for token, expected_count in check.get("exact_count", {}).items():
            observed_count = source.count(token)
            if observed_count != expected_count:
                failures.append(
                    f"{check['id']}: {token!r} count {observed_count} != {expected_count}"
                )
    return sorted(set(failures))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "compare", "source-gates"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name != "validate":
            command.add_argument("--check", action="store_true", required=True)
        if name == "compare":
            command.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest_path = arguments.manifest.resolve()
        manifest = _load_checked_manifest(manifest_path)
        if arguments.command == "validate":
            return EXIT_OK
        if arguments.command == "compare":
            failures = _compare(manifest, manifest_path=manifest_path)
            if arguments.json_out is not None:
                _write_comparison_report(
                    arguments.json_out,
                    _comparison_report(
                        manifest,
                        manifest_path=manifest_path,
                        failures=failures,
                    ),
                    oracle_root=manifest_path.parent,
                )
        else:
            failures = _source_gates(manifest, manifest_path=manifest_path)
    except OracleInvalid as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return EXIT_INVALID
    if failures:
        for failure in failures:
            print(f"mismatch: {failure}", file=sys.stderr)
        return EXIT_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
