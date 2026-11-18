"""Tests for ``models``, ``config``, ``completions``, ``--quiet``, and ``--json`` CLI modes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from voxweave import cli as cli_module
from voxweave import config, export, llm_commands, mux, pipeline, runtime, ui
from voxweave.paths import swap_ext


def test_completions_generates_valid_scripts_for_all_shells(tmp_path, monkeypatch):
    conf = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    runner = CliRunner()

    for shell, marker in (
        ("bash", "_VOXWEAVE_COMPLETE=bash_complete"),
        ("zsh", "_VOXWEAVE_COMPLETE=zsh_complete"),
        ("fish", "_VOXWEAVE_COMPLETE=fish_complete"),
    ):
        res = runner.invoke(cli_module.cli, ["completions", shell])
        assert res.exit_code == 0, res.output
        assert marker in res.stdout
    assert not conf.exists()


def test_config_path_init_and_show_redacts_secrets(tmp_path, monkeypatch):
    conf = tmp_path / "voxweave.conf"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(cache_dir))
    secret = "hf_secret_should_never_be_printed"
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", secret)

    runner = CliRunner()
    res_path = runner.invoke(cli_module.cli, ["config", "path"])
    assert res_path.exit_code == 0
    assert res_path.stdout.strip() == str(conf)
    assert not conf.exists()

    res_init = runner.invoke(cli_module.cli, ["config", "init"])
    assert res_init.exit_code == 0
    assert conf.exists()

    # Refuses overwriting without --force, succeeds with --force.
    res_dup = runner.invoke(cli_module.cli, ["config", "init"])
    assert res_dup.exit_code == 1
    res_force = runner.invoke(cli_module.cli, ["config", "init", "--force"])
    assert res_force.exit_code == 0

    res_show = runner.invoke(cli_module.cli, ["config", "show", "--json"])
    assert res_show.exit_code == 0, res_show.output
    assert secret not in res_show.stdout
    payload = json.loads(res_show.stdout)
    assert payload["config_path"] == str(conf)
    assert payload["exists"] is True
    by_key = {row["key"]: row for row in payload["settings"]}
    assert by_key["hf_token"] == {
        "key": "hf_token",
        "value": "<redacted>",
        "source": "env VOXWEAVE_HF_TOKEN",
    }

    res_table = runner.invoke(cli_module.cli, ["config", "show"], terminal_width=140)
    assert res_table.exit_code == 0
    assert "VoxWeave Effective Configuration" in res_table.stdout
    assert secret not in res_table.stdout


def test_models_path_list_pull_and_clean(tmp_path, monkeypatch):
    cache_dir = tmp_path / "voxweave-cache"
    conf = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(cache_dir))
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    monkeypatch.setattr(runtime, "_use_mlx", lambda: False)

    runner = CliRunner()
    res_path = runner.invoke(cli_module.cli, ["models", "path", "--json"])
    assert res_path.exit_code == 0
    assert json.loads(res_path.stdout)["root"] == str(cache_dir)
    assert not conf.exists()

    # Fake _hf_download and _hf_snapshot writing real files into cache_dir.
    def _fake_download(repo: str, filename: str, cache_dir: str | None = None) -> str:
        assert cache_dir is not None
        target_dir = Path(cache_dir) / f"models--{repo.replace('/', '--')}" / "blobs"
        target_dir.mkdir(parents=True, exist_ok=True)
        blob = target_dir / filename
        blob.write_bytes(b"x" * 1024)
        return str(blob)

    def _fake_snapshot(repo: str, cache_dir: str) -> str:
        target_dir = Path(cache_dir) / f"models--{repo.replace('/', '--')}" / "blobs"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "weights.bin").write_bytes(b"y" * 2048)
        return str(target_dir)

    monkeypatch.setattr(runtime, "_hf_download", _fake_download)
    monkeypatch.setattr(runtime, "_hf_snapshot", _fake_snapshot)

    res_pull = runner.invoke(
        cli_module.cli,
        ["-q", "models", "pull", "--default", "--language", "ja", "--json"],
    )
    assert res_pull.exit_code == 0, res_pull.output
    pulled = json.loads(res_pull.stdout)["pulled"]
    assert [p["key"] for p in pulled] == [
        "separator",
        "panns",
        "qwen3-asr-0.6b",
        "mms-300m",
    ]

    res_list = runner.invoke(cli_module.cli, ["models", "list", "--json"])
    assert res_list.exit_code == 0
    listed = {m["key"]: m for m in json.loads(res_list.stdout)["models"]}
    assert listed["separator"]["cached"] is True
    assert listed["separator"]["bytes"] == 1024
    assert listed["qwen3-asr-1.7b"]["cached"] is False

    # Dry-run clean leaves files on disk; real clean with --yes removes them.
    res_dry = runner.invoke(
        cli_module.cli, ["models", "clean", "separator", "--dry-run", "--json"]
    )
    assert res_dry.exit_code == 0
    assert json.loads(res_dry.stdout)["dry_run"] is True
    assert Path(listed["separator"]["path"]).exists()

    res_clean = runner.invoke(
        cli_module.cli, ["models", "clean", "separator", "--yes", "--json"]
    )
    assert res_clean.exit_code == 0
    assert not Path(listed["separator"]["path"]).exists()


def test_quiet_and_verbose_conflict_and_quiet_suppresses_stderr_banners(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ensure_default_config", lambda: None)
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"media")
    vtt = swap_ext(media, ".vtt")
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "process", Mock(return_value=vtt))
    runner = CliRunner()

    conflict = runner.invoke(cli_module.cli, ["-q", "-v", str(media)])
    assert conflict.exit_code == 2
    assert "not both" in conflict.output

    try:
        quiet_res = runner.invoke(cli_module.cli, ["-q", str(media)])
        assert quiet_res.exit_code == 0, quiet_res.output
        assert quiet_res.stdout.strip() == str(vtt)
        assert quiet_res.stderr == ""
    finally:
        ui.install_logging(verbose=False, quiet=False)


@pytest.mark.parametrize(
    ("args_fn", "patch_target", "return_fn", "expected_cmd"),
    [
        (
            lambda m, v, j: [str(m), "--json"],
            (pipeline, "process"),
            lambda _m, v, _j: v,
            "transcribe",
        ),
        (
            lambda _m, v, j: ["render", str(j), "--json"],
            (pipeline, "split"),
            lambda _m, v, _j: v,
            "render",
        ),
        (
            lambda _m, v, _j: ["align", str(v), "--json"],
            (pipeline, "align"),
            lambda _m, v, _j: v,
            "align",
        ),
        (
            lambda _m, v, _j: ["translate", str(v), "--target", "zh", "--json"],
            (llm_commands, "translate"),
            lambda _m, v, _j: v.with_name("ep.zh.vtt"),
            "translate",
        ),
        (
            lambda _m, v, _j: ["export", str(v), "-f", "srt", "--json"],
            (export, "export_subtitles"),
            lambda _m, v, _j: [v.with_name("ep.srt")],
            "export",
        ),
        (
            lambda _m, v, _j: ["pack", str(v), "--json"],
            (mux, "pack"),
            lambda _m, v, _j: v.with_name("ep.pack.mkv"),
            "pack",
        ),
        (
            lambda _m, v, _j: ["burn", str(v), "--json"],
            (mux, "burn"),
            lambda _m, v, _j: v.with_name("ep.burn.mp4"),
            "burn",
        ),
        (
            lambda _m, v, _j: ["correct", str(v), "--json"],
            (llm_commands, "correct"),
            lambda _m, v, _j: {
                "out": v.with_name("ep.asrfix.vtt"),
                "audit": None,
                "applied": [{"i": 1, "orig": "a", "fixed": "b", "reason": "typo"}],
                "rejected": [],
                "n_cues": 5,
                "aligned": False,
            },
            "correct",
        ),
    ],
)
def test_json_output_mode_across_all_processing_commands(
    tmp_path, monkeypatch, args_fn, patch_target, return_fn, expected_cmd
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "ensure_default_config", lambda: None)

    media = tmp_path / "ep.mkv"
    media.write_bytes(b"media")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n", encoding="utf-8")
    js = tmp_path / "ep.json"
    js.write_text("{}", encoding="utf-8")

    mod, attr = patch_target
    monkeypatch.setattr(mod, attr, Mock(return_value=return_fn(media, vtt, js)))

    runner = CliRunner()
    try:
        res = runner.invoke(cli_module.cli, ["-q", *args_fn(media, vtt, js)])
        assert res.exit_code == 0, res.output
        data = json.loads(res.stdout)
        assert data["command"] == expected_cmd
    finally:
        ui.install_logging(verbose=False, quiet=False)
