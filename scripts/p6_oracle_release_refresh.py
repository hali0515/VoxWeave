#!/usr/bin/env python3
"""Refresh only the P6 oracle execution triplet for one package release."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn, Sequence

import p6_oracle_environment as oracle_environment


EXIT_OK = 0
EXIT_INVALID = 2
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
MANIFEST_PATH = REPO_ROOT / "calibration" / "p6-oracle" / "manifest.json"
EXPECTED_ROOT = REPO_ROOT / "calibration" / "p6-oracle" / "expected"
EXECUTION_FIELDS = frozenset(
    {"package_version", "dependency_lock_sha256", "container_digest"}
)


class RefreshInvalid(Exception):
    """The release refresh cannot make its one authorized manifest edit."""


def _invalid(message: str) -> NoReturn:
    raise RefreshInvalid(message)


def _load_oracle_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "p6_oracle_release_refresh_runner",
        SCRIPTS_DIR / "p6_oracle.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        _invalid("cannot load the P6 oracle runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(arguments: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        _invalid(f"git is unavailable: {exc}")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        _invalid(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _head_bytes(path: Path) -> bytes:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        _invalid(f"release refresh path escapes the repository: {path}")
    return _git(["show", f"HEAD:{relative}"])


def _assert_expected_authorities() -> dict[str, bytes]:
    prefix = EXPECTED_ROOT.relative_to(REPO_ROOT).as_posix()
    tracked = {
        row.decode("utf-8")
        for row in _git(
            ["ls-tree", "-r", "--name-only", "-z", "HEAD", "--", prefix]
        ).split(b"\0")
        if row
    }
    try:
        current_paths = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in EXPECTED_ROOT.rglob("*")
            if path.is_file()
        }
    except OSError as exc:
        _invalid(f"cannot inventory expected artifacts: {exc}")
    if current_paths != tracked:
        _invalid("expected artifact inventory differs from HEAD")
    snapshot: dict[str, bytes] = {}
    for relative in sorted(tracked):
        path = REPO_ROOT / relative
        try:
            value = path.read_bytes()
        except OSError as exc:
            _invalid(f"cannot read expected artifact {relative}: {exc}")
        if value != _git(["show", f"HEAD:{relative}"]):
            _invalid(f"expected artifact differs from HEAD: {relative}")
        snapshot[relative] = value
    return snapshot


def _assert_clean_authorities(manifest_bytes: bytes) -> dict[str, bytes]:
    if manifest_bytes != _head_bytes(MANIFEST_PATH):
        _invalid("manifest differs from HEAD before release refresh")
    return _assert_expected_authorities()


def _replace_execution_strings(raw: bytes, replacements: dict[str, str]) -> bytes:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _invalid(f"manifest is not UTF-8: {exc}")
    start_marker = '  "execution": {\n'
    end_marker = '\n  },\n  "registry_digests":'
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        _invalid("manifest does not expose one closed execution block")
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    block = text[start:end]
    for name, value in replacements.items():
        pattern = re.compile(
            rf'^(    "{re.escape(name)}": )"(?:[^"\\]|\\.)*"(,?)$',
            re.MULTILINE,
        )
        block, count = pattern.subn(
            lambda match: (
                match.group(1) + json.dumps(value, ensure_ascii=False) + match.group(2)
            ),
            block,
        )
        if count != 1:
            _invalid(f"execution field {name} is absent or duplicated")
    return (text[:start] + block + text[end:]).encode("utf-8")


def _read_manifest(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _invalid(f"manifest is invalid JSON: {exc}")
    if not isinstance(value, dict) or not isinstance(value.get("execution"), dict):
        _invalid("manifest has no execution object")
    return value


def _candidate_manifest(raw: bytes) -> tuple[bytes, dict[str, Any]]:
    current = _read_manifest(raw)
    execution = current["execution"]
    try:
        project_version = oracle_environment.project_package_version(REPO_ROOT)
        lock_version = oracle_environment.locked_package_version(REPO_ROOT)
        installed_version = oracle_environment.installed_package_version()
        lock_digest = oracle_environment.sha256_file(REPO_ROOT / "uv.lock")
    except oracle_environment.ExecutionEnvironmentError as exc:
        _invalid(str(exc))
    if not project_version == lock_version == installed_version:
        _invalid("pyproject, editable lock row, and installed package versions differ")
    container_digest = oracle_environment.container_digest(
        dependency_lock_sha256=lock_digest,
        hash_seed=execution["hash_seed"],
        interpreter=execution["interpreter"],
        locale=execution["locale"],
        package_version=project_version,
        platform=execution["platform"],
        timezone=execution["timezone"],
    )
    replacements = {
        "package_version": project_version,
        "dependency_lock_sha256": lock_digest,
        "container_digest": container_digest,
    }
    candidate_raw = _replace_execution_strings(raw, replacements)
    candidate = _read_manifest(candidate_raw)
    if {key: value for key, value in current.items() if key != "execution"} != {
        key: value for key, value in candidate.items() if key != "execution"
    }:
        _invalid("release refresh would alter non-execution manifest content")
    changed = {
        key
        for key in set(execution) | set(candidate["execution"])
        if execution.get(key) != candidate["execution"].get(key)
    }
    if changed != EXECUTION_FIELDS:
        _invalid(
            "release refresh requires exactly package version, lock digest, and "
            f"container digest changes; observed {sorted(changed)}"
        )
    return candidate_raw, candidate


def _preflight(candidate: dict[str, Any]) -> None:
    oracle = _load_oracle_runner()
    try:
        checked = oracle._schema_validate(candidate)
        oracle._validate_manifest_semantics(checked, manifest_path=MANIFEST_PATH)
        failures = oracle._compare(checked, manifest_path=MANIFEST_PATH)
    except oracle.OracleInvalid as exc:
        _invalid(f"candidate execution record is invalid: {exc}")
    if failures:
        _invalid(
            "candidate execution record would require oracle artifact changes: "
            + failures[0]
        )


def _atomic_write(path: Path, value: bytes) -> None:
    if path.resolve() != MANIFEST_PATH.resolve():
        _invalid("release refresh may write only the oracle manifest")
    mode = path.stat().st_mode
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(mode))
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        _invalid(f"cannot atomically write manifest: {exc}")
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _refresh() -> None:
    try:
        original = MANIFEST_PATH.read_bytes()
    except OSError as exc:
        _invalid(f"cannot read manifest: {exc}")
    expected_snapshot = _assert_clean_authorities(original)
    candidate_raw, candidate = _candidate_manifest(original)
    _preflight(candidate)
    try:
        current = MANIFEST_PATH.read_bytes()
    except OSError as exc:
        _invalid(f"cannot reread manifest after release preflight: {exc}")
    if _assert_clean_authorities(current) != expected_snapshot:
        _invalid("expected artifacts changed during release preflight")
    _atomic_write(MANIFEST_PATH, candidate_raw)
    try:
        if MANIFEST_PATH.read_bytes() != candidate_raw:
            _invalid("manifest differs from the release candidate after commit")
        if _assert_expected_authorities() != expected_snapshot:
            _invalid("expected artifacts changed during release commit")
    except (OSError, RefreshInvalid) as exc:
        _atomic_write(MANIFEST_PATH, original)
        if isinstance(exc, OSError):
            _invalid(f"cannot verify the committed manifest: {exc}")
        raise


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        _refresh()
    except (RefreshInvalid, KeyError, TypeError) as exc:
        print(f"P6 ORACLE RELEASE REFRESH INVALID: {exc}", file=sys.stderr)
        return EXIT_INVALID
    print("P6 ORACLE RELEASE REFRESH PASS: execution triplet updated")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
