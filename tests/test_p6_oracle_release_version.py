from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_ROOT = REPO_ROOT / "calibration" / "p6-oracle"
ORACLE_MANIFEST = ORACLE_ROOT / "manifest.json"
ORACLE_RUNNER = REPO_ROOT / "scripts" / "p6_oracle.py"
PUBLIC_WORKER = REPO_ROOT / "scripts" / "p6_oracle_public.py"
RELEASE_REFRESH = REPO_ROOT / "scripts" / "p6_oracle_release_refresh.py"
PACKAGE_VERSION_TEMPLATE = "__P6_ORACLE_EXECUTION_PACKAGE_VERSION__"


def _load_oracle_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "p6_oracle_release_version_runner", ORACLE_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_public_worker() -> Any:
    spec = importlib.util.spec_from_file_location(
        "p6_oracle_release_version_worker", PUBLIC_WORKER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_release_refresh(path: Path = RELEASE_REFRESH) -> Any:
    spec = importlib.util.spec_from_file_location(
        "p6_oracle_release_version_refresh", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    scripts_dir = str(path.parent)
    sys.path.insert(0, scripts_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.path.pop(0) == scripts_dir
    return module


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _git(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = _run(["git", *arguments], cwd=cwd)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _copy_repository(tmp_path: Path) -> Path:
    destination = tmp_path / "release-repository"

    def ignore(_path: str, names: list[str]) -> set[str]:
        excluded = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "build",
            "REPORT.md",
            "TASK.md",
        }
        return set(names) & excluded

    shutil.copytree(REPO_ROOT, destination, ignore=ignore)
    _git(["init", "-q"], cwd=destination)
    _git(["config", "user.email", "p6-oracle@example.invalid"], cwd=destination)
    _git(["config", "user.name", "P6 Oracle Test"], cwd=destination)
    common = Path(_git(["rev-parse", "--git-common-dir"], cwd=REPO_ROOT).stdout.strip())
    if not common.is_absolute():
        common = (REPO_ROOT / common).resolve()
    alternates = destination / ".git" / "objects" / "info" / "alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_text(str(common / "objects") + "\n", encoding="utf-8")
    _git(["add", "-A"], cwd=destination)
    _git(["commit", "-qm", "test baseline"], cwd=destination)
    return destination


def _simulate_version_bump(repository: Path) -> str:
    pyproject = repository / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8")
    current_version = tomllib.loads(pyproject_text)["project"]["version"]
    parsed = Version(current_version)
    major = parsed.release[0]
    minor = parsed.release[1] if len(parsed.release) > 1 else 0
    epoch = f"{parsed.epoch}!" if parsed.epoch else ""
    simulated_version = f"{epoch}{major}.{minor + 1}.0"
    assert Version(simulated_version) > parsed
    old_project = f'version = "{current_version}"'
    assert pyproject_text.count(old_project) == 1
    pyproject.write_text(
        pyproject_text.replace(old_project, f'version = "{simulated_version}"', 1),
        encoding="utf-8",
    )

    lock_path = repository / "uv.lock"
    lock_text = lock_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(\[\[package\]\]\nname = "voxweave"\nversion = ")'
        r'[^"\n]+'
        r'("\nsource = \{ editable = "\." \})'
    )
    updated, count = pattern.subn(rf"\g<1>{simulated_version}\g<2>", lock_text)
    assert count == 1
    lock_path.write_text(updated, encoding="utf-8")
    return simulated_version


def _fake_distribution(repository: Path, *, package_version: str) -> Path:
    site = repository / ".release-test-site"
    metadata = site / f"voxweave-{package_version}.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: voxweave\nVersion: {package_version}\n",
        encoding="utf-8",
    )
    return site


def _oracle_environment(repository: Path, metadata_site: Path) -> dict[str, str]:
    return {
        "LANG": "zh_CN.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": os.pathsep.join((str(metadata_site), str(repository))),
    }


def _expected_snapshot(repository: Path) -> dict[str, bytes]:
    root = repository / "calibration" / "p6-oracle" / "expected"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_package_version_is_one_execution_recorded_projection_value():
    manifest = json.loads(ORACLE_MANIFEST.read_bytes())
    recorded = manifest["execution"]["package_version"]
    assert recorded == importlib.metadata.version("voxweave")

    delivery_path = ORACLE_ROOT / "inputs" / "selected-v2-segmentation-delivery.json"
    delivery = json.loads(delivery_path.read_bytes())
    assert "voxweave" not in delivery["delivery"]["manifest"]
    assert recorded.encode("utf-8") not in delivery_path.read_bytes()

    expected_path = (
        ORACLE_ROOT / "expected" / "selected-v2-segmentation" / "episode.json"
    )
    expected = expected_path.read_text(encoding="utf-8")
    assert expected.count(PACKAGE_VERSION_TEMPLATE) == 1
    assert recorded not in expected

    case = next(
        row for row in manifest["cases"] if row["id"] == "selected-v2-segmentation"
    )
    output = next(
        row for row in case["expected_paths"] if row["artifact"] == "main-json"
    )
    assert output["parameterization"] == "execution.package_version"


def test_versioned_expected_bytes_are_both_emitted_by_the_standalone_projector(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    oracle = _load_oracle_runner()
    manifest = oracle._load_checked_manifest(ORACLE_MANIFEST)
    case = next(
        row for row in manifest["cases"] if row["id"] == "selected-v2-segmentation"
    )
    live = oracle._project_case(
        case,
        ORACLE_ROOT,
        package_version=manifest["execution"]["package_version"],
    )
    template = oracle._project_case(
        case,
        ORACLE_ROOT,
        package_version=PACKAGE_VERSION_TEMPLATE,
    )
    assert (
        template["main-json"]
        == (
            ORACLE_ROOT / "expected" / "selected-v2-segmentation" / "episode.json"
        ).read_bytes()
    )
    live_document = json.loads(live["main-json"])
    template_document = json.loads(template["main-json"])
    assert (
        live_document["segmentation"]["voxweave"]
        == manifest["execution"]["package_version"]
    )
    assert template_document["segmentation"]["voxweave"] == PACKAGE_VERSION_TEMPLATE
    live_document["segmentation"]["voxweave"] = PACKAGE_VERSION_TEMPLATE
    assert live_document == template_document
    assert (
        oracle._package_version_projection_failure(
            template["main-json"],
            live["main-json"],
            package_version=manifest["execution"]["package_version"],
        )
        is None
    )
    assert "beyond" in oracle._package_version_projection_failure(
        template["main-json"],
        live["main-json"] + b" ",
        package_version=manifest["execution"]["package_version"],
    )


def test_public_worker_rejects_a_recorded_version_other_than_live_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _load_public_worker()
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda _name: "7.8.9")

    worker._verify_package_version("7.8.9")
    with pytest.raises(RuntimeError, match="distribution version differs"):
        worker._verify_package_version("7.8.8")


def test_release_refresh_writer_refuses_an_expected_artifact_target():
    refresh = _load_release_refresh()
    expected_path = (
        ORACLE_ROOT / "expected" / "selected-v2-segmentation" / "episode.json"
    )
    before = expected_path.read_bytes()

    with pytest.raises(refresh.RefreshInvalid, match="only the oracle manifest"):
        refresh._atomic_write(expected_path, b"mutated expected bytes")

    assert expected_path.read_bytes() == before


def test_container_digest_independently_binds_the_recorded_package_version():
    execution = json.loads(ORACLE_MANIFEST.read_bytes())["execution"]
    digest_facts = {
        "dependency_lock_sha256": execution["dependency_lock_sha256"],
        "hash_seed": execution["hash_seed"],
        "interpreter": execution["interpreter"],
        "locale": execution["locale"],
        "package_version": execution["package_version"],
        "platform": execution["platform"],
        "timezone": execution["timezone"],
        "toolchain": "detached-environment-v2",
    }
    encoded = json.dumps(
        digest_facts,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == execution["container_digest"]


def test_release_refresh_does_not_overwrite_a_precommit_manifest_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository = _copy_repository(tmp_path)
    simulated_version = _simulate_version_bump(repository)
    manifest_path = repository / "calibration" / "p6-oracle" / "manifest.json"
    refresh = _load_release_refresh(
        repository / "scripts" / "p6_oracle_release_refresh.py"
    )
    monkeypatch.setattr(
        refresh.oracle_environment,
        "installed_package_version",
        lambda: simulated_version,
    )

    def mutate_during_preflight(_candidate: dict[str, Any]) -> None:
        raw = manifest_path.read_text(encoding="utf-8")
        assert raw.count('"runner_version": 1') == 1
        manifest_path.write_text(
            raw.replace('"runner_version": 1', '"runner_version": 2', 1),
            encoding="utf-8",
        )

    monkeypatch.setattr(refresh, "_preflight", mutate_during_preflight)
    with pytest.raises(refresh.RefreshInvalid, match="manifest differs from HEAD"):
        refresh._refresh()

    assert json.loads(manifest_path.read_bytes())["runner_version"] == 2


@pytest.mark.parametrize("dirty_kind", ("expected", "non-execution-manifest"))
def test_release_refresh_refuses_non_execution_or_expected_drift(
    dirty_kind: str,
    tmp_path: Path,
):
    repository = _copy_repository(tmp_path)
    simulated_version = _simulate_version_bump(repository)
    metadata_site = _fake_distribution(
        repository,
        package_version=simulated_version,
    )
    environment = _oracle_environment(repository, metadata_site)
    manifest_path = repository / "calibration" / "p6-oracle" / "manifest.json"
    if dirty_kind == "expected":
        expected_path = (
            repository
            / "calibration"
            / "p6-oracle"
            / "expected"
            / "selected-v2-segmentation"
            / "episode.json"
        )
        expected_path.write_bytes(expected_path.read_bytes() + b" ")
    else:
        text = manifest_path.read_text(encoding="utf-8")
        assert text.count('"runner_version": 1') == 1
        manifest_path.write_text(
            text.replace('"runner_version": 1', '"runner_version": 2', 1),
            encoding="utf-8",
        )
    before_manifest = manifest_path.read_bytes()
    before_expected = _expected_snapshot(repository)

    result = _run(
        [sys.executable, str(repository / "scripts" / RELEASE_REFRESH.name)],
        cwd=repository,
        environment=environment,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    expected_message = (
        "expected artifact differs from HEAD"
        if dirty_kind == "expected"
        else "manifest differs from HEAD before release refresh"
    )
    assert expected_message in result.stderr
    assert manifest_path.read_bytes() == before_manifest
    assert _expected_snapshot(repository) == before_expected


def test_simulated_release_refresh_passes_all_three_oracle_gates(tmp_path: Path):
    repository = _copy_repository(tmp_path)
    manifest_path = repository / "calibration" / "p6-oracle" / "manifest.json"
    before = json.loads(manifest_path.read_bytes())
    before_expected = _expected_snapshot(repository)
    simulated_version = _simulate_version_bump(repository)
    metadata_site = _fake_distribution(
        repository,
        package_version=simulated_version,
    )
    environment = _oracle_environment(repository, metadata_site)
    runner = repository / "scripts" / "p6_oracle.py"

    bypass = _run(
        [sys.executable, str(runner), "validate", "--manifest", str(manifest_path)],
        cwd=repository,
        environment=environment,
    )
    assert bypass.returncode == 2, bypass.stdout + bypass.stderr

    refresh = _run(
        [sys.executable, str(repository / "scripts" / RELEASE_REFRESH.name)],
        cwd=repository,
        environment=environment,
        timeout=300,
    )
    assert refresh.returncode == 0, refresh.stdout + refresh.stderr
    after = json.loads(manifest_path.read_bytes())
    assert {key: value for key, value in after.items() if key != "execution"} == {
        key: value for key, value in before.items() if key != "execution"
    }
    assert {
        key
        for key in before["execution"]
        if before["execution"][key] != after["execution"][key]
    } == {"package_version", "dependency_lock_sha256", "container_digest"}
    assert after["execution"]["package_version"] == simulated_version
    assert (
        after["execution"]["dependency_lock_sha256"]
        == hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()
    )
    assert _expected_snapshot(repository) == before_expected

    commands = (
        ["validate", "--manifest", str(manifest_path)],
        ["compare", "--manifest", str(manifest_path), "--check"],
        ["source-gates", "--manifest", str(manifest_path), "--check"],
    )
    for arguments in commands:
        result = _run(
            [sys.executable, str(runner), *arguments],
            cwd=repository,
            environment=environment,
            timeout=420,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert _expected_snapshot(repository) == before_expected
