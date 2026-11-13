"""Tests for contributor portability of host-pinned P6 oracle tests."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _pytest.outcomes import Skipped

CONFTEST_PATH = Path(__file__).resolve().with_name("conftest.py")


@pytest.fixture
def conftest_mod(pytestconfig: pytest.Config) -> Any:
    for plugin in pytestconfig.pluginmanager.get_plugins():
        plugin_file = getattr(plugin, "__file__", None)
        if plugin_file and Path(plugin_file).resolve() == CONFTEST_PATH:
            return plugin
    raise AssertionError(f"loaded conftest plugin not found for {CONFTEST_PATH}")


def test_strict_oracle_env_var_is_preserved_by_process_isolation(
    conftest_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "VOXWEAVE_STRICT_ORACLE" in conftest_mod._OPT_IN_TEST_ENV
    monkeypatch.setenv("VOXWEAVE_STRICT_ORACLE", "1")
    monkeypatch.setenv("VOXWEAVE_UNRELATED_LEAK", "1")
    monkeypatch.setenv("TERM", "dumb")

    conftest_mod._isolate_process_environment()

    assert conftest_mod.os.environ.get("VOXWEAVE_STRICT_ORACLE") == "1"
    assert "VOXWEAVE_UNRELATED_LEAK" not in conftest_mod.os.environ
    assert "TERM" not in conftest_mod.os.environ


def test_every_host_pinned_oracle_test_entry_exists_in_its_module(
    conftest_mod: Any,
) -> None:
    for rel_path, names in conftest_mod._HOST_PINNED_ORACLE_TESTS.items():
        path = conftest_mod._REPO_ROOT / rel_path
        assert path.is_file(), f"missing test module {rel_path}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = sorted(names - defined)
        assert not missing, f"{rel_path} no longer defines {missing}"


def test_oracle_host_mismatch_reason_detects_interpreter_platform_and_shallow_clone(
    conftest_mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        (
            "{\n"
            '  "execution": {\n'
            '    "interpreter": "CPython 3.13.12",\n'
            '    "platform": "Linux x86_64"\n'
            "  },\n"
            '  "reference_commits": {\n'
            '    "historical": "6e6033fa3930b263133f02c1332ae4d79a490f8b"\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    conftest_mod._load_oracle_pin.cache_clear()
    monkeypatch.setattr(
        conftest_mod.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(
        conftest_mod.sys, "version_info", SimpleNamespace(major=3, minor=13, micro=12)
    )
    monkeypatch.setattr(conftest_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(conftest_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(conftest_mod, "_git_commit_reachable", lambda _r, _c: True)

    assert (
        conftest_mod.oracle_host_mismatch_reason(
            manifest_path=manifest, repo_root=tmp_path
        )
        is None
    )

    monkeypatch.setattr(
        conftest_mod.sys, "version_info", SimpleNamespace(major=3, minor=12, micro=9)
    )
    reason_py = conftest_mod.oracle_host_mismatch_reason(
        manifest_path=manifest, repo_root=tmp_path
    )
    assert reason_py is not None and "CPython 3.12.9" in reason_py

    monkeypatch.setattr(
        conftest_mod.sys, "version_info", SimpleNamespace(major=3, minor=13, micro=12)
    )
    monkeypatch.setattr(conftest_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(conftest_mod.platform, "machine", lambda: "arm64")
    reason_os = conftest_mod.oracle_host_mismatch_reason(
        manifest_path=manifest, repo_root=tmp_path
    )
    assert reason_os is not None and "Darwin arm64" in reason_os

    monkeypatch.setattr(conftest_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(conftest_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(conftest_mod, "_git_commit_reachable", lambda _r, _c: False)
    reason_git = conftest_mod.oracle_host_mismatch_reason(
        manifest_path=manifest, repo_root=tmp_path
    )
    assert reason_git is not None and "shallow git clone" in reason_git


def test_pytest_runtest_setup_skips_unless_strict_oracle_is_enabled(
    conftest_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = (
        conftest_mod._REPO_ROOT / "tests" / "test_p6_oracle_public_commands.py"
    )
    pinned_item = SimpleNamespace(
        path=target_path,
        originalname="test_public_command_artifact_matches_the_standalone_projector_authority",
        name="test_public_command_artifact_matches_the_standalone_projector_authority[selected-v2-align-vtt]",
    )
    unpinned_item = SimpleNamespace(
        path=target_path,
        originalname="test_manifest_public_runtime_contract_is_closed_and_non_declarative",
        name="test_manifest_public_runtime_contract_is_closed_and_non_declarative",
    )
    monkeypatch.setattr(
        conftest_mod,
        "oracle_host_mismatch_reason",
        lambda: "platform is 'Darwin arm64', oracle pins 'Linux x86_64'",
    )

    monkeypatch.delenv("VOXWEAVE_STRICT_ORACLE", raising=False)
    with pytest.raises(Skipped, match="VOXWEAVE_STRICT_ORACLE=1"):
        conftest_mod.pytest_runtest_setup(pinned_item)

    # Non-host-pinned tests in the same module still run everywhere.
    conftest_mod.pytest_runtest_setup(unpinned_item)

    # Setting VOXWEAVE_STRICT_ORACLE=1 disables the skip so CI enforces the pin.
    monkeypatch.setenv("VOXWEAVE_STRICT_ORACLE", "1")
    conftest_mod.pytest_runtest_setup(pinned_item)
