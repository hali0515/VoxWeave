"""Delivery commands share workflow progress without writing diagnostics to stdout."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from rich.console import Console

from voxweave import cli as cli_mod
from voxweave import mux, speakers, speakerserve, ui
from voxweave.progress import Reporter

VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello there\n"
PROGRESS = (
    "frame=24\nout_time=00:00:01.000000\nprogress=continue\n"
    "frame=48\nout_time=00:00:02.000000\nprogress=end\n"
)


@pytest.fixture
def delivery_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "install_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        ui, "console", Console(stderr=True, force_terminal=False, color_system=None)
    )
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    subtitle = tmp_path / "episode.vtt"
    subtitle.write_text(VTT, encoding="utf-8")
    monkeypatch.setattr(
        mux,
        "probe_streams",
        lambda path: [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "nb_frames": "48",
            }
        ],
    )
    return media, subtitle


def test_export_numbered_step_and_output_paths(delivery_inputs):
    _, subtitle = delivery_inputs
    result = CliRunner().invoke(
        cli_mod.cli, ["export", str(subtitle), "--to", "srt", "--to", "ass"]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        str(subtitle.with_suffix(".srt")),
        str(subtitle.with_suffix(".ass")),
    ]
    assert "[1/1] export subtitles" in result.stderr
    assert "Export done" in result.stderr
    assert "\x1b[" not in result.stderr


def test_export_failure_uses_error_panel(delivery_inputs):
    _, subtitle = delivery_inputs
    result = CliRunner().invoke(cli_mod.cli, ["export", str(subtitle), "--to", "vtt"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "already .vtt" in result.stderr
    assert "Traceback" not in result.stderr
    assert subtitle.read_text(encoding="utf-8") == VTT


def test_pack_numbered_steps_and_only_output_path(delivery_inputs, monkeypatch):
    media, subtitle = delivery_inputs

    def fake_ffmpeg(cmd, *, capture):
        assert capture is True
        Path(cmd[-1]).write_bytes(b"packed")

    monkeypatch.setattr(mux, "_run_ffmpeg", fake_ffmpeg)
    result = CliRunner().invoke(cli_mod.cli, ["pack", str(subtitle)])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(media.with_name("episode.pack.mkv"))
    assert "[1/2] check inputs" in result.stderr
    assert "[2/2] pack subtitles" in result.stderr
    assert "Pack done" in result.stderr
    assert media.read_bytes() == b"media"


class _EncodingProcess:
    def __init__(self, *, stdout=PROGRESS, returncode=0):
        self.stdout = io.StringIO(stdout)
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.mark.parametrize("native_ass", [False, True])
def test_burn_steps_follow_input_format(delivery_inputs, monkeypatch, native_ass):
    media, subtitle = delivery_inputs
    if native_ass:
        subtitle = subtitle.with_suffix(".ass")
        subtitle.write_text("[Script Info]\n", encoding="utf-8")
    monkeypatch.setattr(mux, "pick_encoder", lambda codec, *, force: "libx264")

    def fake_popen(cmd, **kwargs):
        assert "-stats" not in cmd
        assert "-nostats" in cmd
        assert cmd[cmd.index("-progress") + 1] == "pipe:1"
        assert kwargs["stdin"] is mux.subprocess.DEVNULL
        assert kwargs["stdout"] is mux.subprocess.PIPE
        assert kwargs["stderr"].writable()
        Path(cmd[-1]).write_bytes(b"encoded")
        return _EncodingProcess()

    monkeypatch.setattr(mux.subprocess, "Popen", fake_popen)
    result = CliRunner().invoke(cli_mod.cli, ["burn", str(subtitle)])
    assert result.exit_code == 0, result.output
    total = 3 if native_ass else 4
    assert f"[1/{total}] check inputs" in result.stderr
    assert f"[2/{total}] select encoder" in result.stderr
    assert f"[{total}/{total}] encode video" in result.stderr
    assert "Burn done" in result.stderr
    assert ("prepare subtitles" in result.stderr) is not native_ass
    assert result.stdout.strip() == str(media.with_suffix(".mp4"))
    assert "frame=" not in result.stdout + result.stderr
    assert media.read_bytes() == b"media"


class _EncodingReporter(Reporter):
    def __init__(self):
        self.tasks = []
        self.statuses = []
        self.completed = 0

    def task(self, label, total):
        self.tasks.append((label, total))

    def advance(self, n=1):
        self.completed += n

    def status(self, label):
        self.statuses.append(label)


@pytest.mark.parametrize("total_frames", [48, None])
def test_ffmpeg_progress_uses_measured_frames_and_time(monkeypatch, total_frames):
    monkeypatch.setattr(
        mux.subprocess, "Popen", lambda *args, **kwargs: _EncodingProcess()
    )
    reporter = _EncodingReporter()
    mux._run_ffmpeg_progress(
        ["ffmpeg", "-stats", "output.mp4"], reporter, total_frames=total_frames
    )
    if total_frames is None:
        assert reporter.tasks == []
        assert reporter.statuses == [
            "encoded 24 frames, 00:00:01",
            "encoded 48 frames, 00:00:02",
        ]
    else:
        assert reporter.tasks == [("encoding frames", 48)]
        assert reporter.completed == 48


def test_ffmpeg_progress_failure_retains_early_error_and_tail(monkeypatch):
    def fake_popen(cmd, **kwargs):
        kwargs["stderr"].write(
            "ERROR: encoder initialization failed\n" + "detail\n" * 12
        )
        kwargs["stderr"].flush()
        return _EncodingProcess(stdout="", returncode=7)

    monkeypatch.setattr(mux.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="exit 7") as caught:
        mux._run_ffmpeg_progress(
            ["ffmpeg", "output.mp4"], Reporter(), total_frames=None
        )
    assert "ERROR: encoder initialization failed" in str(caught.value)
    assert str(caught.value).count("detail") == 8


def test_burn_failure_preserves_existing_output(delivery_inputs, monkeypatch):
    media, subtitle = delivery_inputs
    output = media.with_suffix(".mp4")
    output.write_bytes(b"previous output")
    monkeypatch.setattr(mux, "pick_encoder", lambda codec, *, force: "libx264")

    def fake_popen(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"partial encoding")
        kwargs["stderr"].write("ERROR: encoding failed\n")
        return _EncodingProcess(stdout="", returncode=1)

    monkeypatch.setattr(mux.subprocess, "Popen", fake_popen)
    result = CliRunner().invoke(cli_mod.cli, ["burn", str(subtitle)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "ERROR: encoding failed" in result.stderr
    assert "Burn done" not in result.stderr
    assert output.read_bytes() == b"previous output"
    assert not list(output.parent.glob("*.part.mp4"))


def test_ffmpeg_progress_interrupt_stops_encoder(monkeypatch):
    class InterruptedStream:
        closed = False

        def __iter__(self):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    class InterruptedProcess:
        stdout = InterruptedStream()
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            assert self.terminated
            return -15

    process = InterruptedProcess()
    monkeypatch.setattr(mux.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(KeyboardInterrupt):
        mux._run_ffmpeg_progress(
            ["ffmpeg", "output.mp4"], Reporter(), total_frames=None
        )
    assert process.terminated
    assert process.stdout.closed


def test_speaker_service_url_and_session_updates_use_separate_streams(
    delivery_inputs, monkeypatch
):
    media, subtitle = delivery_inputs
    audition = SimpleNamespace(
        page="<html></html>",
        media_path=media,
        mapping_path=media.parent / "speakers.json",
        sibling_json_path=subtitle.with_suffix(".json"),
        speaker_ids=("SPEAKER_00",),
        pristine_mapping_generation=None,
    )
    monkeypatch.setattr(speakers, "create_speaker_audition", lambda path: audition)

    def fake_serve(**kwargs):
        report = kwargs["report"]
        report("http://127.0.0.1:41533/")
        report("Saved speakers.json")
        report("Next: voxweave render episode.json")
        return "http://127.0.0.1:41533/"

    monkeypatch.setattr(speakerserve, "serve", fake_serve)
    result = CliRunner().invoke(cli_mod.cli, ["speakers", str(media)])
    assert result.exit_code == 0, result.output
    assert result.stdout == "http://127.0.0.1:41533/\n"
    assert "Saved speakers.json" in result.stderr
    assert "Next: voxweave render episode.json" in result.stderr
    assert "[1/" not in result.stderr
