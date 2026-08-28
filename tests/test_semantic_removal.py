"""RAT-6 option B removes the public semantic-split compatibility mode."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from click.testing import CliRunner

from voxweave import config, semantic_breaks
from voxweave.cli import cli, cmd_transcribe


def _option_names(command) -> set[str]:
    return {
        option
        for parameter in command.params
        for option in (*parameter.opts, *parameter.secondary_opts)
    }


def test_bundled_worker_module_and_lock_remain_absent():
    assert importlib.util.find_spec("voxweave.semantic_worker") is None
    package_dir = Path(semantic_breaks.__file__).parent
    assert not (package_dir / "semantic_worker.py").exists()
    assert not (package_dir / "semantic_worker.py.lock").exists()
    assert not hasattr(semantic_breaks, "LocalTransformersSelector")


@pytest.mark.parametrize(
    "command",
    [cmd_transcribe, cli.commands["split"]],
    ids=["transcribe", "split"],
)
def test_public_commands_expose_no_semantic_mode(command):
    names = _option_names(command)
    assert "--semantic-split" not in names
    assert "--no-semantic-split" not in names
    assert "--semantic-model" not in names


def test_public_help_contains_no_semantic_mode():
    runner = CliRunner()
    root_help = runner.invoke(cli, ["--help"])
    split_help = runner.invoke(cli, ["split", "--help"])
    assert root_help.exit_code == 0
    assert split_help.exit_code == 0
    assert "semantic-split" not in root_help.output
    assert "semantic-model" not in root_help.output
    assert "semantic-split" not in split_help.output
    assert "semantic-model" not in split_help.output


def test_public_config_exposes_no_semantic_mode():
    assert "semantic" not in config._KNOWN_KEYS
    assert not hasattr(config, "DEFAULT_SEMANTIC_MODEL")
    assert not hasattr(config, "conf_semantic_model")
