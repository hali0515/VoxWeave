"""Closed execution-environment facts shared by the P6 oracle release path."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tomllib
from pathlib import Path
from typing import Any


PACKAGE_DISTRIBUTION = "voxweave"
PACKAGE_VERSION_TEMPLATE = "__P6_ORACLE_EXECUTION_PACKAGE_VERSION__"
TOOLCHAIN_ID = "detached-environment-v2"


class ExecutionEnvironmentError(ValueError):
    """The project, lock, or installed distribution is not one exact release."""


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ExecutionEnvironmentError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):  # pragma: no cover - tomllib root contract
        raise ExecutionEnvironmentError(f"{path.name} is not a TOML table")
    return value


def project_package_version(repo_root: Path) -> str:
    project = _read_toml(repo_root / "pyproject.toml").get("project")
    if not isinstance(project, dict):
        raise ExecutionEnvironmentError("pyproject.toml has no project table")
    version = project.get("version")
    if type(version) is not str or not version:
        raise ExecutionEnvironmentError("pyproject.toml has no package version")
    return version


def locked_package_version(repo_root: Path) -> str:
    packages = _read_toml(repo_root / "uv.lock").get("package")
    if not isinstance(packages, list):
        raise ExecutionEnvironmentError("uv.lock has no package rows")
    matches = []
    for package in packages:
        if not isinstance(package, dict) or package.get("name") != PACKAGE_DISTRIBUTION:
            continue
        source = package.get("source")
        if isinstance(source, dict) and source.get("editable") == ".":
            matches.append(package)
    if len(matches) != 1:
        raise ExecutionEnvironmentError(
            "uv.lock must contain one editable voxweave package row"
        )
    version = matches[0].get("version")
    if type(version) is not str or not version:
        raise ExecutionEnvironmentError("uv.lock editable voxweave version is invalid")
    return version


def installed_package_version() -> str:
    try:
        version = importlib.metadata.version(PACKAGE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ExecutionEnvironmentError(
            "the installed voxweave distribution is unavailable"
        ) from exc
    if not version:
        raise ExecutionEnvironmentError("the installed voxweave version is empty")
    return version


def installed_metadata_path() -> Path:
    try:
        distribution = importlib.metadata.distribution(PACKAGE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ExecutionEnvironmentError(
            "the installed voxweave distribution metadata is unavailable"
        ) from exc
    raw_path = getattr(distribution, "_path", None)
    if not isinstance(raw_path, Path):
        raise ExecutionEnvironmentError(
            "the installed voxweave metadata has no concrete path"
        )
    path = raw_path.resolve()
    if not path.is_dir():
        raise ExecutionEnvironmentError(
            "the installed voxweave metadata path is not a directory"
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ExecutionEnvironmentError(f"cannot hash {path.name}: {exc}") from exc
    return digest.hexdigest()


def container_digest(
    *,
    dependency_lock_sha256: str,
    hash_seed: str,
    interpreter: str,
    locale: str,
    package_version: str,
    platform: str,
    timezone: str,
) -> str:
    value = {
        "dependency_lock_sha256": dependency_lock_sha256,
        "hash_seed": hash_seed,
        "interpreter": interpreter,
        "locale": locale,
        "package_version": package_version,
        "platform": platform,
        "timezone": timezone,
        "toolchain": TOOLCHAIN_ID,
    }
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
