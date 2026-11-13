"""Tests for multi-file ``transcribe`` batching, ``--skip-existing``, and ``--no-clobber``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from voxweave import backend, chunking, config, pipeline, songdet, transcription
from voxweave import cli as cli_module
from voxweave.paths import swap_ext
from voxweave.progress import Reporter


class _SilentReporter(Reporter):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def batch_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "ensure_default_config", lambda: None)
    monkeypatch.setattr(cli_module, "RichReporter", _SilentReporter)
    monkeypatch.setattr(cli_module, "summary_panel", Mock())

    ep1 = tmp_path / "ep01.mkv"
    ep2 = tmp_path / "ep02.mkv"
    ep3 = tmp_path / "ep03.mkv"
    for p in (ep1, ep2, ep3):
        p.write_bytes(b"media")

    def _fake_process(media: Path, **_kwargs) -> Path:
        out = swap_ext(media, ".vtt")
        out.write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nok\n", encoding="utf-8"
        )
        return out

    proc_mock = Mock(side_effect=_fake_process)
    monkeypatch.setattr(pipeline, "process", proc_mock)
    return CliRunner(), (ep1, ep2, ep3), proc_mock


def test_shorthand_and_explicit_multi_file_batch_process_in_order(batch_env):
    runner, (ep1, ep2, ep3), proc_mock = batch_env

    res = runner.invoke(cli_module.cli, [str(ep1), str(ep2), str(ep3)])
    assert res.exit_code == 0, res.output
    assert [c.args[0] for c in proc_mock.call_args_list] == [ep1, ep2, ep3]
    assert res.stdout.splitlines() == [
        str(swap_ext(ep1, ".vtt")),
        str(swap_ext(ep2, ".vtt")),
        str(swap_ext(ep3, ".vtt")),
    ]
    assert "Batch [1/3] ep01.mkv" in res.stderr
    assert "Batch [3/3] ep03.mkv" in res.stderr


def test_skip_existing_skips_completed_episodes_and_echoes_all_vtts(batch_env):
    runner, (ep1, ep2, ep3), proc_mock = batch_env
    existing_vtt = swap_ext(ep1, ".vtt")
    existing_vtt.write_text("WEBVTT\n\nexisting\n", encoding="utf-8")

    res = runner.invoke(
        cli_module.cli, ["--skip-existing", str(ep1), str(ep2), str(ep3)]
    )
    assert res.exit_code == 0, res.output
    assert [c.args[0] for c in proc_mock.call_args_list] == [ep2, ep3]
    assert existing_vtt.read_text(encoding="utf-8") == "WEBVTT\n\nexisting\n"
    assert res.stdout.splitlines() == [
        str(existing_vtt),
        str(swap_ext(ep2, ".vtt")),
        str(swap_ext(ep3, ".vtt")),
    ]
    assert "Skipping ep01.mkv" in res.stderr


def test_no_clobber_refuses_before_processing_any_file(batch_env):
    runner, (ep1, ep2, _ep3), proc_mock = batch_env
    # Only the SECOND file already has a .vtt; preflight must still refuse before touching ep1.
    swap_ext(ep2, ".vtt").write_text("WEBVTT\n", encoding="utf-8")

    res = runner.invoke(cli_module.cli, ["--no-clobber", str(ep1), str(ep2)])
    assert res.exit_code == 1
    proc_mock.assert_not_called()
    assert not swap_ext(ep1, ".vtt").exists()
    assert "--no-clobber" in res.stderr
    assert "ep02.vtt" in res.stderr


def test_no_clobber_refuses_duplicate_stem_collision(batch_env, tmp_path):
    runner, (ep1, _ep2, _ep3), proc_mock = batch_env
    dup = tmp_path / "ep01.mp4"
    dup.write_bytes(b"media")

    res = runner.invoke(cli_module.cli, ["--no-clobber", str(ep1), str(dup)])
    assert res.exit_code == 1
    proc_mock.assert_not_called()
    assert "same output" in res.stderr


def test_skip_existing_and_no_clobber_are_mutually_exclusive(batch_env):
    runner, (ep1, _ep2, _ep3), proc_mock = batch_env
    res = runner.invoke(cli_module.cli, ["--skip-existing", "--no-clobber", str(ep1)])
    assert res.exit_code == 2
    assert "not both" in res.output
    proc_mock.assert_not_called()


def test_non_media_in_second_position_rejected_before_running_first(
    batch_env, tmp_path
):
    runner, (ep1, _ep2, _ep3), proc_mock = batch_env
    bad = tmp_path / "notes.vtt"
    bad.write_text("WEBVTT\n", encoding="utf-8")

    res = runner.invoke(cli_module.cli, [str(ep1), str(bad)])
    assert res.exit_code == 2
    assert "align" in res.output.lower()
    proc_mock.assert_not_called()


def test_batch_session_retains_models_between_items_and_releases_on_exit(
    tmp_path, monkeypatch
):
    backend_rel = Mock()
    vad_rel = Mock()
    panns_rel = Mock()
    monkeypatch.setattr(backend, "release", backend_rel)
    monkeypatch.setattr(chunking, "release_silero_vad", vad_rel)
    monkeypatch.setattr(songdet, "release_model", panns_rel)

    t1 = tmp_path / "tmp1.wav"
    t1.write_bytes(b"pcm")
    run = transcription._TranscribeRun(
        media_path=tmp_path / "ep01.mkv",
        dbg=transcription._debug_sink(
            tmp_path / "ep01.mkv", debug=False, debug_root=None
        ),
        rep=Reporter(),
        tmp=[t1],
    )

    with transcription.batch_session(retain_models=True):
        transcription._release_and_clean(
            run, panns_handoff=False, keep_models=transcription._RETAIN_MODELS.get()
        )
        # Temp files deleted immediately, models retained during batch.
        assert not t1.exists()
        backend_rel.assert_not_called()
        vad_rel.assert_not_called()

    # Models unconditionally freed on batch exit.
    backend_rel.assert_called_once()
    vad_rel.assert_called_once()
