"""Tests for ``voxweave doctor`` diagnostics and clean ``Ctrl+C`` handling."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess

import onnxruntime as ort
from click.testing import CliRunner

from voxweave import cli as cli_module
from voxweave import cli_doctor, config, pipeline, runtime


def _reset_conf_cache() -> None:
    config._WARNED.clear()


def _mock_asr_specs(monkeypatch, present: set[str], missing: set[str] = set()) -> None:
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, *args, **kwargs):
        if name in present:
            return importlib.machinery.ModuleSpec(name, loader=None)
        if name in missing:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(cli_doctor.importlib.util, "find_spec", _fake_find_spec)


def test_doctor_json_passes_and_skips_first_run_config_creation(tmp_path, monkeypatch):
    conf_file = tmp_path / "voxweave.conf"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf_file))
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(cache_dir))
    monkeypatch.delenv("VOXWEAVE_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    _reset_conf_cache()

    monkeypatch.setattr(
        cli_doctor.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "ffprobe"} else None,
    )
    monkeypatch.setattr(
        cli_doctor.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=0,
            stdout="ffmpeg version 7.1\n V..... hevc_nvenc           NVIDIA NVENC hevc\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(runtime, "get_device", lambda: "cpu")
    monkeypatch.setattr(runtime, "_use_mlx", lambda: False)
    _mock_asr_specs(monkeypatch, {"qwen_asr", "faster_whisper"})

    runner = CliRunner()
    res = runner.invoke(cli_module.cli, ["doctor", "--json"])
    assert res.exit_code == 0, res.output
    assert not conf_file.exists(), (
        "doctor must stay read-only and never create voxweave.conf"
    )

    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    names = [item["name"] for item in payload["checks"]]
    assert names == [
        "platform",
        "ffmpeg",
        "compute",
        "asr_backend",
        "onnxruntime",
        "config",
        "hf_auth",
        "cache",
    ]
    by_name = {item["name"]: item for item in payload["checks"]}
    assert by_name["ffmpeg"]["status"] == "ok"
    assert "hevc_nvenc" in by_name["ffmpeg"]["summary"]
    assert by_name["compute"]["status"] == "warn"


def test_doctor_human_table_renders_warnings_and_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    _reset_conf_cache()
    monkeypatch.setattr(
        cli_doctor.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "ffprobe"} else None,
    )
    monkeypatch.setattr(
        cli_doctor.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=0, stdout="ffmpeg version 7.1\n", stderr=""
        ),
    )
    monkeypatch.setattr(runtime, "get_device", lambda: "cuda")
    monkeypatch.setattr(runtime, "_use_mlx", lambda: False)
    _mock_asr_specs(monkeypatch, {"qwen_asr", "faster_whisper"})
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )

    runner = CliRunner()
    res = runner.invoke(cli_module.cli, ["doctor"], terminal_width=140)
    assert res.exit_code == 0, res.output
    assert "VoxWeave Environment Check" in res.stdout
    assert "CUDAExecutionProvider missing" in res.stdout
    assert "--overrides" in res.stdout


def test_doctor_asr_backend_checks_mlx_when_mlx_active(monkeypatch):
    monkeypatch.setattr(runtime, "_use_mlx", lambda: True)
    _mock_asr_specs(monkeypatch, {"mlx_audio", "mlx_whisper"})
    assert cli_doctor.check_asr_backend().status == "ok"

    _mock_asr_specs(monkeypatch, set(), missing={"mlx_audio", "mlx_whisper"})
    missing = cli_doctor.check_asr_backend()
    assert missing.status == "error"
    assert "mlx-audio" in missing.summary


def test_doctor_exits_one_when_ffmpeg_missing_or_config_invalid(tmp_path, monkeypatch):
    conf_file = tmp_path / "voxweave.conf"
    conf_file.write_text("not = [valid toml\n", encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf_file))
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    _reset_conf_cache()
    monkeypatch.setattr(cli_doctor.shutil, "which", lambda _name: None)

    runner = CliRunner()
    res = runner.invoke(cli_module.cli, ["doctor", "--json"])
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    by_name = {item["name"]: item for item in payload["checks"]}
    assert by_name["ffmpeg"]["status"] == "error"
    assert "ffmpeg" in by_name["ffmpeg"]["summary"]
    assert by_name["config"]["status"] == "error"
    assert "invalid TOML" in by_name["config"]["summary"]


def test_doctor_config_warns_on_unknown_keys(tmp_path, monkeypatch):
    conf_file = tmp_path / "voxweave.conf"
    conf_file.write_text(
        "mystery_top = 1\n\n[defaults]\nseparate = true\nunknown_toggle = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf_file))
    _reset_conf_cache()

    result = cli_doctor.check_config()
    assert result.status == "warn"
    assert "mystery_top" in result.summary
    assert "[defaults].unknown_toggle" in result.summary


def test_doctor_hf_auth_never_leaks_secret_and_errors_when_diarize_default_unauthenticated(
    tmp_path, monkeypatch
):
    conf_file = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf_file))
    secret = "hf_super_secret_token_value_12345"
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", secret)
    _reset_conf_cache()

    authed = cli_doctor.check_hf_auth()
    assert authed.status == "ok"
    assert "VOXWEAVE_HF_TOKEN" in authed.summary
    assert secret not in authed.summary
    assert secret not in authed.hint

    monkeypatch.delenv("VOXWEAVE_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    conf_file.write_text("[defaults]\ndiarize = true\n", encoding="utf-8")
    _reset_conf_cache()

    missing = cli_doctor.check_hf_auth()
    assert missing.status == "error"
    assert "[defaults].diarize = true" in missing.summary


def test_keyboard_interrupt_in_pipeline_exits_130_without_traceback(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    monkeypatch.setattr(config, "ensure_default_config", lambda: None)

    def _raise_sigint(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "process", _raise_sigint)

    runner = CliRunner()
    res = runner.invoke(cli_module.cli, [str(media)])
    assert res.exit_code == 130
    assert "Interrupted." in res.stderr
    assert "Traceback" not in res.output
    assert "Traceback" not in res.stderr


def test_keyboard_interrupt_in_read_only_command_exits_130(monkeypatch):
    def _raise_sigint():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_doctor, "run_all_checks", _raise_sigint)

    runner = CliRunner()
    res = runner.invoke(cli_module.cli, ["doctor"])
    assert res.exit_code == 130
    assert "Interrupted." in res.stderr
    assert "Traceback" not in res.output
