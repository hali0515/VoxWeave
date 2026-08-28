#!/usr/bin/env python3
"""Check the tracked P6 align-shadow corpus through the public align API."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import locale
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_PATH = (
    REPO_ROOT / "calibration" / "schemas" / "align-shadow-manifest.schema.json"
)
ARTIFACT_SCHEMA_PATH = (
    REPO_ROOT / "calibration" / "schemas" / "align-shadow-artifact.schema.json"
)
REPORT_SCHEMA_PATH = (
    REPO_ROOT / "calibration" / "schemas" / "align-shadow-report.schema.json"
)
WATCHED_IMPORTS = (
    "voxweave.align_evidence_core",
    "voxweave.align_delta_registry",
    "voxweave.core.finalizer",
    "voxweave.core.align_compare",
    "voxweave.align_shadow",
    "voxweave.align_shadow_minimal",
)
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
EXPECTED_KEYS = (
    "fully_admitted",
    "selected",
    "artifact_kind",
    "failure",
    "detail_code",
    "alds",
    "artifact_sha256",
)
NORMAL_VARIANTS = (
    (False, "none"),
    (False, "collector"),
    (True, "none"),
    (True, "collector"),
    (True, "throwing"),
)
ALLOWED_INJECTIONS = frozenset(
    {
        "rich-freeze",
        "rich-encode",
        "rich-schema",
        "minimal-construction",
        "post-finalize-partition",
    }
)


class HarnessInvalid(RuntimeError):
    """The harness cannot make a trustworthy measurement."""


def _reject_constant(token: str) -> None:
    raise HarnessInvalid(f"nonfinite JSON token {token}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessInvalid(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_bytes(),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except HarnessInvalid:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessInvalid("JSON input is unavailable or invalid") from exc


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_bytes(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _closed_mapping(
    value: object,
    keys: tuple[str, ...],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != keys:
        raise HarnessInvalid(f"{label} is not a closed ordered object")
    return value


def _safe_path(root: Path, raw: object) -> Path:
    if type(raw) is not str or not raw:
        raise HarnessInvalid("case path is empty or non-string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise HarnessInvalid("case path is absolute or traverses its root")
    base = root.resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HarnessInvalid("case path escapes its root") from exc
    return candidate


def _validate_file_fact(root: Path, value: object) -> Mapping[str, Any]:
    fact = _closed_mapping(value, ("path", "size", "sha256"), label="file fact")
    path = _safe_path(root, fact["path"])
    if not path.is_file():
        raise HarnessInvalid("declared corpus file is missing")
    raw = path.read_bytes()
    if (
        type(fact["size"]) is not int
        or fact["size"] != len(raw)
        or not _is_sha256(fact["sha256"])
        or fact["sha256"] != _sha256_bytes(raw)
    ):
        raise HarnessInvalid("declared corpus file fact does not match")
    return fact


def _environment_projection() -> dict[str, str | None]:
    return {
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "locale": locale.getencoding(),
        "timezone": os.environ.get("TZ"),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def _validate_selected(value: object) -> Mapping[str, Any]:
    selected = _closed_mapping(
        value,
        ("engine_family", "vtt", "json", "evidence"),
        label="selected output",
    )
    if selected["engine_family"] not in ("legacy-v1", "boundary-v2"):
        raise HarnessInvalid("selected engine family is invalid")
    for key in ("vtt", "json", "evidence"):
        member = _closed_mapping(
            selected[key], ("path", "sha256"), label=f"selected {key}"
        )
        raw_path = member["path"]
        if (
            type(raw_path) is not str
            or not raw_path
            or Path(raw_path).is_absolute()
            or ".." in Path(raw_path).parts
            or not _is_sha256(member["sha256"])
        ):
            raise HarnessInvalid(f"selected {key} declaration is invalid")
    return selected


def _validate_profile(value: object) -> Mapping[str, Any]:
    profile = _closed_mapping(
        value,
        ("kind", "values", "digest", "test_case_id"),
        label="authority limit profile",
    )
    if profile["kind"] != "production" or profile["test_case_id"] is not None:
        raise HarnessInvalid("corpus authority profile is not production")
    if not isinstance(profile["values"], Mapping) or not _is_sha256(profile["digest"]):
        raise HarnessInvalid("corpus authority profile is malformed")
    from voxweave.align_distribution import production_authority_limit_profile

    issued = production_authority_limit_profile()
    expected_values = {
        "call": {
            "state_limit": issued.call.state_limit,
            "edge_limit": issued.call.edge_limit,
            "interval_limit": issued.call.interval_limit,
            "normalize_char_limit": issued.call.normalize_char_limit,
        },
        "job": {
            "call_limit": issued.job.call_limit,
            "state_limit": issued.job.state_limit,
            "edge_limit": issued.job.edge_limit,
            "interval_limit": issued.job.interval_limit,
            "normalize_char_limit": issued.job.normalize_char_limit,
        },
    }
    if (
        profile["values"] != expected_values
        or profile["digest"] != issued.profile_digest
    ):
        raise HarnessInvalid("corpus authority profile does not match runtime")
    return profile


def _validate_manifest(path: Path) -> Mapping[str, Any]:
    manifest = _closed_mapping(_load_json(path), MANIFEST_KEYS, label="manifest")
    if manifest["schema_version"] != 1 or manifest["artifact_schema_version"] != 2:
        raise HarnessInvalid("manifest version is unsupported")
    if not _is_sha256(manifest["registry_sha256"]):
        raise HarnessInvalid("manifest registry digest is invalid")
    from voxweave.engine_registry import REGISTRY_SHA256

    if manifest["registry_sha256"] != REGISTRY_SHA256:
        raise HarnessInvalid("manifest registry digest does not match runtime")
    environment = _closed_mapping(
        manifest["environment"],
        ("python_major_minor", "platform", "locale", "timezone", "hash_seed"),
        label="environment",
    )
    if dict(environment) != _environment_projection():
        raise HarnessInvalid("runtime environment does not match the manifest")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise HarnessInvalid("manifest cases are missing")
    root = Path(path).resolve().parent
    identifiers: list[str] = []
    for raw_case in cases:
        case = _closed_mapping(raw_case, CASE_KEYS, label="case")
        case_id = case["id"]
        if type(case_id) is not str or not case_id:
            raise HarnessInvalid("case id is invalid")
        identifiers.append(case_id)
        if case["route"] not in ("ctc-full", "mms-full", "qwen-crop"):
            raise HarnessInvalid("case route is invalid")
        if type(case["effective_iso"]) is not str or not case["effective_iso"]:
            raise HarnessInvalid("case effective ISO is invalid")
        argv = case["argv"]
        if not isinstance(argv, list) or not all(type(item) is str for item in argv):
            raise HarnessInvalid("case argv is invalid")
        env = case["env"]
        if (
            not isinstance(env, Mapping)
            or "VOXWEAVE_SEG_V2_SHADOW" not in env
            or not all(
                type(key) is str and (value is None or type(value) is str)
                for key, value in env.items()
            )
        ):
            raise HarnessInvalid("case environment is invalid")
        inputs = case["inputs"]
        if not isinstance(inputs, list) or not inputs:
            raise HarnessInvalid("case inputs are missing")
        for fact in inputs:
            _validate_file_fact(root, fact)
        _validate_file_fact(root, case["backend_receipt"])
        _validate_profile(case["authority_limit_profile"])
        expected = _closed_mapping(case["expected"], EXPECTED_KEYS, label="expected")
        if type(expected["fully_admitted"]) is not bool:
            raise HarnessInvalid("expected admission is invalid")
        _validate_selected(expected["selected"])
        if expected["artifact_kind"] not in ("rich", "minimal-failure", "none"):
            raise HarnessInvalid("expected artifact kind is invalid")
        if expected["failure"] is not None and type(expected["failure"]) is not str:
            raise HarnessInvalid("expected failure kind is invalid")
        if (
            expected["detail_code"] is not None
            and type(expected["detail_code"]) is not str
        ):
            raise HarnessInvalid("expected failure detail is invalid")
        from voxweave.align_failures import OUTCOME_DETAILS

        if (expected["failure"] is None) != (expected["detail_code"] is None) or (
            expected["failure"] is not None
            and expected["detail_code"]
            not in OUTCOME_DETAILS.get(expected["failure"], ())
        ):
            raise HarnessInvalid("expected failure pair is not registered")
        if (
            not isinstance(expected["alds"], list)
            or not all(
                type(delta) is str
                and delta in tuple(f"ALD-{index}" for index in range(7))
                for delta in expected["alds"]
            )
            or len(set(expected["alds"])) != len(expected["alds"])
        ):
            raise HarnessInvalid("expected ALD set is invalid")
        if expected["artifact_sha256"] is not None and not _is_sha256(
            expected["artifact_sha256"]
        ):
            raise HarnessInvalid("expected artifact digest is invalid")
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise HarnessInvalid("manifest case ids are not unique and sorted")
    return manifest


def _manifest_case(manifest: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    for case in manifest["cases"]:
        if case["id"] == case_id:
            return case
    raise HarnessInvalid("worker case id is absent from the manifest")


def _apply_environment(env: Mapping[str, Any]) -> None:
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _copy_case_inputs(
    manifest_path: Path,
    case: Mapping[str, Any],
    episode_root: Path,
) -> dict[str, Path]:
    corpus_root = manifest_path.resolve().parent
    copied: dict[str, Path] = {}
    for fact in case["inputs"]:
        source = _safe_path(corpus_root, fact["path"])
        target = episode_root / source.name
        if source.name in copied:
            raise HarnessInvalid("case input basenames are not unique")
        shutil.copyfile(source, target)
        copied[source.name] = target
    return copied


def _load_receipt(manifest_path: Path, case: Mapping[str, Any]) -> Mapping[str, Any]:
    root = manifest_path.resolve().parent
    fact = case["backend_receipt"]
    value = _load_json(_safe_path(root, fact["path"]))
    receipt = _closed_mapping(
        value, ("schema_version", "route", "calls"), label="backend receipt"
    )
    if (
        receipt["schema_version"] != 1
        or receipt["route"] != case["route"]
        or not isinstance(receipt["calls"], list)
        or len(receipt["calls"]) != 1
    ):
        raise HarnessInvalid("backend receipt header is invalid")
    return receipt


def _raise_injected(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("align shadow calibration injection")


class _ExitDrivingPartitionResult:
    exit_driving = (object(),)


def _post_finalize_partition_cutoff(
    *_args: object, **_kwargs: object
) -> _ExitDrivingPartitionResult:
    return _ExitDrivingPartitionResult()


def _install_injections(injections: frozenset[str]) -> None:
    if not injections:
        return
    unknown = injections - ALLOWED_INJECTIONS
    if unknown:
        raise HarnessInvalid("unknown align shadow injection")
    rich_seams = injections & {
        "rich-freeze",
        "rich-encode",
        "rich-schema",
    }
    if rich_seams:
        from voxweave import align_shadow

        if "rich-freeze" in rich_seams:
            align_shadow._immutable = _raise_injected
        if "rich-encode" in rich_seams:
            align_shadow._canonical_bytes = _raise_injected
        if "rich-schema" in rich_seams:
            align_shadow._validate_rich_artifact = _raise_injected
    if "minimal-construction" in injections:
        from voxweave import align_shadow_minimal

        align_shadow_minimal.build_minimal_align_shadow_failure_artifact = (
            _raise_injected
        )
    if "post-finalize-partition" in injections:
        from voxweave.core import partition_check

        partition_check.check_partition = _post_finalize_partition_cutoff


def _watched_imports() -> dict[str, bool]:
    return {name: name in sys.modules for name in WATCHED_IMPORTS}


def _empty_run(
    *,
    shadow_enabled: bool,
    observer_kind: str,
    injection: str | None,
) -> dict[str, Any]:
    return {
        "shadow_enabled": shadow_enabled,
        "observer": observer_kind,
        "injection": injection,
        "normal_return": None,
        "selected": None,
        "p11_decisions": None,
        "evidence_bytes_b64": None,
        "artifact_kind": "unavailable",
        "artifact_bytes_b64": None,
        "artifact_sha256": None,
        "imports": _watched_imports(),
        "observer_call_count": 0,
        "alds": [],
        "outcome": "infrastructure-invalid",
    }


def _selected_projection(
    *,
    vtt_path: Path,
    json_path: Path,
    evidence_path: Path,
    evidence_bytes: bytes,
) -> dict[str, Any]:
    evidence_value = json.loads(evidence_bytes)
    selected = evidence_value["selected_outputs"]
    return {
        "engine_family": selected["engine_family"],
        "vtt": {"path": vtt_path.name, "sha256": _sha256_path(vtt_path)},
        "json": {"path": json_path.name, "sha256": _sha256_path(json_path)},
        "evidence": {
            "path": evidence_path.name,
            "sha256": _sha256_bytes(evidence_bytes),
        },
    }


def _artifact_alds(raw: bytes | None) -> list[str]:
    active: list[str] = []
    if raw is not None:
        value = json.loads(raw)
        result = value.get("comparison", {}).get("result")
        if isinstance(result, Mapping):
            raw_active = result.get("active_classes")
            if isinstance(raw_active, list):
                active.extend(
                    item
                    for item in raw_active
                    if type(item) is str and item.startswith("ALD-")
                )
    active.append("ALD-6")
    return [f"ALD-{index}" for index in range(7) if f"ALD-{index}" in active]


def _worker_run(
    *,
    manifest_path: Path,
    case: Mapping[str, Any],
    shadow_enabled: bool,
    observer_kind: str,
    injections: frozenset[str],
) -> dict[str, Any]:
    injection = next(
        (name for name in sorted(injections) if name.startswith("rich-")), None
    )
    if case["expected"]["failure"] is not None:
        injection = None
    run = _empty_run(
        shadow_enabled=shadow_enabled,
        observer_kind=observer_kind,
        injection=injection,
    )
    _apply_environment(case["env"])
    if shadow_enabled:
        os.environ["VOXWEAVE_SEG_V2_SHADOW"] = "1"
    else:
        os.environ.pop("VOXWEAVE_SEG_V2_SHADOW", None)

    with tempfile.TemporaryDirectory(prefix="voxweave-align-shadow-") as temporary:
        episode_root = Path(temporary)
        copied = _copy_case_inputs(manifest_path, case, episode_root)
        argv = case["argv"]
        if argv != ["episode.vtt", "--media", "episode.wav"]:
            raise HarnessInvalid("case argv is outside the calibrated public call")
        vtt_path = copied[argv[0]]
        media_path = copied[argv[2]]
        json_path = copied["episode.json"]
        evidence_path = episode_root / "episode.align-evidence.json"
        receipt = _load_receipt(manifest_path, case)
        prepared_path = episode_root / "prepared.wav"
        with wave.open(str(prepared_path), "wb") as prepared:
            prepared.setnchannels(1)
            prepared.setsampwidth(2)
            prepared.setframerate(16_000)
            prepared.writeframes(b"\x00\x00" * 32_000)

        from voxweave import backend, config, engine_registry, pipeline
        from voxweave.progress import Reporter

        family = case["expected"]["selected"]["engine_family"]
        registry = dict(engine_registry.LANGUAGE_ENGINE_FAMILY)
        registry[case["effective_iso"]] = family
        engine_registry.LANGUAGE_ENGINE_FAMILY = MappingProxyType(registry)
        pipeline._prepare_16k_for_align = lambda *_args, **_kwargs: prepared_path
        backend.uses_mms = lambda _iso: False
        config.align_model_for = lambda _iso: "align-shadow-synthetic-ctc"
        backend.release = lambda: None

        call_cursor = 0

        def fake_full_ctc(
            _wav: Path,
            texts: list[str],
            iso: str,
            _model_name: str,
            *args: object,
            bounds: object = None,
            speech_spans: object = None,
            _raw_call_observer: Any = None,
            **kwargs: object,
        ) -> list[list[dict[str, Any]]]:
            del args, bounds, speech_spans, kwargs
            nonlocal call_cursor
            calls = receipt["calls"]
            if call_cursor >= len(calls):
                raise HarnessInvalid("physical backend exceeded its receipt")
            call = calls[call_cursor]
            call_cursor += 1
            if iso != case["effective_iso"] or len(texts) != len(call["block_units"]):
                raise HarnessInvalid("physical backend invocation mismatched receipt")
            flattened = [unit for block in call["block_units"] for unit in block]
            if _raw_call_observer is not None:
                _raw_call_observer(
                    copy.deepcopy(flattened),
                    copy.deepcopy(flattened),
                    tuple(call["source_indices"]),
                    float(call["physical_origin_seconds"]),
                )
            return copy.deepcopy(call["block_units"])

        backend.align_blocks_full_ctc = fake_full_ctc
        _install_injections(injections)

        artifact_bytes: bytes | None = None
        artifact_kind = "none"
        observer_call_count = 0

        def collect(artifact: object) -> None:
            nonlocal artifact_bytes, artifact_kind, observer_call_count
            observer_call_count += 1
            encoded = artifact.to_canonical_bytes()  # type: ignore[attr-defined]
            if artifact_bytes is not None:
                raise HarnessInvalid("observer was called more than once")
            artifact_bytes = encoded
            artifact_kind = str(artifact.artifact_kind)  # type: ignore[attr-defined]
            if observer_kind == "throwing":
                raise RuntimeError("align shadow calibration throwing observer")

        callback = None if observer_kind == "none" else collect
        returned = pipeline.align(
            vtt_path,
            media_path=media_path,
            separate=False,
            reporter=Reporter(),
            _shadow_observer=callback,
        )
        if call_cursor != len(receipt["calls"]):
            raise HarnessInvalid("physical backend receipt was not exhausted")
        evidence_bytes = evidence_path.read_bytes()
        selected_projection = _selected_projection(
            vtt_path=vtt_path,
            json_path=json_path,
            evidence_path=evidence_path,
            evidence_bytes=evidence_bytes,
        )
        selected_json = json.loads(json_path.read_bytes())
        p11_decisions = {
            "speaker_turns_present": "speaker_turns" in selected_json,
            "voiceprint_capture_present": "voiceprint_capture" in selected_json,
            "voiceprint_media_present": "voiceprint_media" in selected_json,
        }

        expected_callback = shadow_enabled and observer_kind != "none"
        rich_injected = any(name.startswith("rich-") for name in injections)
        minimal_injected = "minimal-construction" in injections
        if expected_callback and observer_call_count == 0:
            artifact_kind = "unavailable"
        if minimal_injected and rich_injected and observer_call_count == 0:
            outcome = "infrastructure-invalid"
        elif expected_callback and observer_call_count != 1:
            outcome = "infrastructure-invalid"
        elif not expected_callback and observer_call_count != 0:
            outcome = "infrastructure-invalid"
        elif rich_injected and artifact_kind != "minimal-failure":
            outcome = "correctness-failure"
        elif not rich_injected and expected_callback and artifact_kind != "rich":
            outcome = "correctness-failure"
        elif not expected_callback and artifact_kind != "none":
            outcome = "correctness-failure"
        else:
            outcome = "valid"
        run.update(
            {
                "normal_return": returned.name,
                "selected": selected_projection,
                "p11_decisions": p11_decisions,
                "evidence_bytes_b64": base64.b64encode(evidence_bytes).decode("ascii"),
                "artifact_kind": artifact_kind,
                "artifact_bytes_b64": (
                    None
                    if artifact_bytes is None
                    else base64.b64encode(artifact_bytes).decode("ascii")
                ),
                "artifact_sha256": (
                    None if artifact_bytes is None else _sha256_bytes(artifact_bytes)
                ),
                "imports": _watched_imports(),
                "observer_call_count": observer_call_count,
                "alds": _artifact_alds(artifact_bytes),
                "outcome": outcome,
            }
        )
        return run


def _worker_main(arguments: argparse.Namespace) -> int:
    shadow_enabled = arguments.shadow == "1"
    injections = frozenset(arguments.injection)
    injection = next(
        (name for name in sorted(injections) if name.startswith("rich-")), None
    )
    try:
        manifest = _validate_manifest(arguments.manifest)
        case = _manifest_case(manifest, arguments.case_id)
        run = _worker_run(
            manifest_path=arguments.manifest,
            case=case,
            shadow_enabled=shadow_enabled,
            observer_kind=arguments.observer,
            injections=injections,
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        run = _empty_run(
            shadow_enabled=shadow_enabled,
            observer_kind=arguments.observer,
            injection=injection,
        )
    _write_json(arguments.result_out, run)
    return 0 if run["outcome"] != "infrastructure-invalid" else 2


def _variant_rows(injections: frozenset[str]) -> tuple[tuple[bool, str], ...]:
    if not injections:
        return NORMAL_VARIANTS
    return ((True, "collector"),)


def _worker_command(
    *,
    manifest_path: Path,
    case_id: str,
    shadow_enabled: bool,
    observer_kind: str,
    injections: frozenset[str],
    result_path: Path,
    env: Mapping[str, Any],
) -> dict[str, Any]:
    child_env = dict(os.environ)
    for key, value in env.items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--manifest",
        str(manifest_path),
        "--case-id",
        case_id,
        "--shadow",
        "1" if shadow_enabled else "0",
        "--observer",
        observer_kind,
        "--result-out",
        str(result_path),
    ]
    for injection in sorted(injections):
        command.extend(("--injection", injection))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=child_env,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if not result_path.is_file():
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        return _empty_run(
            shadow_enabled=shadow_enabled,
            observer_kind=observer_kind,
            injection=next(
                (name for name in sorted(injections) if name.startswith("rich-")),
                None,
            ),
        )
    run = _load_json(result_path)
    if not isinstance(run, dict):
        raise HarnessInvalid("worker result is not an object")
    if completed.returncode not in (0, 2):
        run["outcome"] = "infrastructure-invalid"
        run["artifact_kind"] = "unavailable"
    if run["outcome"] == "infrastructure-invalid" and completed.stderr:
        sys.stderr.write(completed.stderr)
    return run


def _artifact_value(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    encoded = run["artifact_bytes_b64"]
    if encoded is None:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
        value = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HarnessInvalid("worker artifact is not canonical JSON") from exc
    if (
        not isinstance(value, Mapping)
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or run["artifact_sha256"] != _sha256_bytes(raw)
    ):
        raise HarnessInvalid("worker artifact bytes are malformed")
    return value


def _compare_normal_run(run: dict[str, Any], expected: Mapping[str, Any]) -> None:
    if run["outcome"] == "infrastructure-invalid":
        return
    mismatch = run["selected"] != expected["selected"]
    expected_callback = run["shadow_enabled"] and run["observer"] != "none"
    expected_kind = expected["artifact_kind"] if expected_callback else "none"
    mismatch = mismatch or run["artifact_kind"] != expected_kind
    artifact = _artifact_value(run)
    if expected_callback:
        if artifact is None:
            mismatch = True
        else:
            failure = artifact.get("failure")
            actual_kind = None if failure is None else failure.get("kind")
            actual_detail = None if failure is None else failure.get("detail_code")
            mismatch = mismatch or actual_kind != expected["failure"]
            mismatch = mismatch or actual_detail != expected["detail_code"]
            mismatch = mismatch or run["alds"] != expected["alds"]
            mismatch = mismatch or run["artifact_sha256"] != expected["artifact_sha256"]
    elif artifact is not None:
        mismatch = True
    if run["normal_return"] != expected["selected"]["vtt"]["path"]:
        mismatch = True
    if mismatch:
        run["outcome"] = "correctness-failure"


def _compare_injected_run(run: dict[str, Any], injections: frozenset[str]) -> None:
    if run["outcome"] == "infrastructure-invalid":
        return
    rich_injected = any(name.startswith("rich-") for name in injections)
    if not rich_injected:
        run["outcome"] = "correctness-failure"
        return
    artifact = _artifact_value(run)
    failure = None if artifact is None else artifact.get("failure")
    secondary = failure.get("secondary", []) if isinstance(failure, Mapping) else []
    construction_described = (
        isinstance(failure, Mapping)
        and failure.get("detail_code") == "rich-artifact-construction"
    ) or (
        isinstance(secondary, list)
        and bool(secondary)
        and isinstance(secondary[-1], Mapping)
        and secondary[-1].get("detail_code") == "rich-artifact-construction"
    )
    if (
        run["observer_call_count"] != 1
        or run["artifact_kind"] != "minimal-failure"
        or artifact is None
        or not construction_described
    ):
        run["outcome"] = "correctness-failure"


def _build_report(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    injections: frozenset[str],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="voxweave-align-shadow-results-") as raw:
        result_root = Path(raw)
        for case_index, case in enumerate(manifest["cases"]):
            runs: list[dict[str, Any]] = []
            for variant_index, (shadow_enabled, observer_kind) in enumerate(
                _variant_rows(injections)
            ):
                run = _worker_command(
                    manifest_path=manifest_path,
                    case_id=case["id"],
                    shadow_enabled=shadow_enabled,
                    observer_kind=observer_kind,
                    injections=injections,
                    result_path=result_root
                    / f"{case_index:04d}-{variant_index:02d}.json",
                    env=case["env"],
                )
                if injections:
                    _compare_injected_run(run, injections)
                else:
                    _compare_normal_run(run, case["expected"])
                runs.append(run)
            actual_family = next(
                (
                    run["selected"]["engine_family"]
                    for run in runs
                    if isinstance(run.get("selected"), Mapping)
                ),
                case["expected"]["selected"]["engine_family"],
            )
            cases.append(
                {
                    "id": case["id"],
                    "engine_family": actual_family,
                    "fully_admitted": case["expected"]["fully_admitted"],
                    "expected_failure": case["expected"]["failure"],
                    "expected_detail_code": case["expected"]["detail_code"],
                    "expected_alds": case["expected"]["alds"],
                    "runs": runs,
                }
            )
    invalid_runs = [
        run
        for case in cases
        for run in case["runs"]
        if run["outcome"] == "infrastructure-invalid"
    ]
    failed_runs = [
        run
        for case in cases
        for run in case["runs"]
        if run["outcome"] == "correctness-failure"
    ]
    valid_cases = sum(
        all(run["outcome"] == "valid" for run in case["runs"]) for case in cases
    )
    return {
        "schema_version": 1,
        "manifest_sha256": _sha256_path(manifest_path),
        "registry_sha256": manifest["registry_sha256"],
        "cases": cases,
        "case_count": len(cases),
        "valid_count": valid_cases,
        "infrastructure_invalid_count": len(invalid_runs),
        "correctness_failure_count": len(failed_runs),
    }


def _baseline_cases(report: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for case in report["cases"]:
        collector = next(
            run
            for run in case["runs"]
            if run["shadow_enabled"] is True and run["observer"] == "collector"
        )
        projected[case["id"]] = {
            "selected": collector["selected"],
            "artifact_kind": collector["artifact_kind"],
            "failure": case["expected_failure"],
            "detail_code": case["expected_detail_code"],
            "alds": collector["alds"],
            "artifact_sha256": collector["artifact_sha256"],
        }
    return projected


def _check_baseline(
    baseline_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> bool:
    baseline = _closed_mapping(
        _load_json(baseline_path),
        (
            "schema_version",
            "manifest_sha256",
            "manifest_schema_sha256",
            "artifact_schema_sha256",
            "report_schema_sha256",
            "registry_sha256",
            "cases",
        ),
        label="baseline",
    )
    expected_header = {
        "schema_version": 1,
        "manifest_sha256": _sha256_path(manifest_path),
        "manifest_schema_sha256": _sha256_path(MANIFEST_SCHEMA_PATH),
        "artifact_schema_sha256": _sha256_path(ARTIFACT_SCHEMA_PATH),
        "report_schema_sha256": _sha256_path(REPORT_SCHEMA_PATH),
        "registry_sha256": manifest["registry_sha256"],
    }
    return all(baseline[key] == value for key, value in expected_header.items()) and (
        baseline["cases"] == _baseline_cases(report)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("report", "check"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--json-out", type=Path, required=True)
        if command == "check":
            child.add_argument("--baseline", type=Path, required=True)
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--case-id", required=True)
    worker.add_argument("--shadow", choices=("0", "1"), required=True)
    worker.add_argument(
        "--observer", choices=("none", "collector", "throwing"), required=True
    )
    worker.add_argument("--result-out", type=Path, required=True)
    worker.add_argument("--injection", action="append", default=[])
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    _injections: frozenset[str] = frozenset(),
) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "_worker":
        return _worker_main(arguments)
    if _injections - ALLOWED_INJECTIONS:
        return 2
    try:
        manifest_path = arguments.manifest.resolve()
        manifest = _validate_manifest(manifest_path)
        report = _build_report(manifest_path, manifest, _injections)
        _write_json(arguments.json_out, report)
        if report["infrastructure_invalid_count"]:
            return 2
        if report["correctness_failure_count"]:
            return 1
        if arguments.command == "check" and not _check_baseline(
            arguments.baseline.resolve(), manifest_path, manifest, report
        ):
            return 1
        return 0
    except HarnessInvalid as exc:
        print(f"align-shadow harness invalid: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        print(
            f"align-shadow harness unavailable: {type(exc).__name__}", file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
