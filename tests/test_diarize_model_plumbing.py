from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from voxweave import backend, chunking, diarize, pipeline, songdet
from voxweave.cli import cli


LEGACY_MODEL = "pyannote/speaker-diarization-3.1"
COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.delenv("VOXWEAVE_DIARIZE_MODEL", raising=False)


def _write_model_config(tmp_path: Path, value: str) -> None:
    (tmp_path / "voxweave.conf").write_text(
        f'[diarize]\nmodel = "{value}"\n', encoding="utf-8"
    )


def _invoke_cli(tmp_path: Path, *arguments: str):
    media = tmp_path / "episode.wav"
    media.write_bytes(b"media")
    output = tmp_path / "episode.vtt"
    output.write_text("WEBVTT\n", encoding="utf-8")
    with patch("voxweave.pipeline.process", return_value=output) as process:
        result = CliRunner().invoke(cli, [*arguments, "--diarize", str(media)])
    assert result.exit_code == 0, result.output
    return process


def test_cli_diarize_model_precedence_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_model_config(tmp_path, "community-1")

    default_from_config = _invoke_cli(tmp_path)
    assert default_from_config.call_args.kwargs["diarize_model"] == COMMUNITY_MODEL

    monkeypatch.setenv("VOXWEAVE_DIARIZE_MODEL", "example/environment-model")
    environment = _invoke_cli(tmp_path)
    assert environment.call_args.kwargs["diarize_model"] == "example/environment-model"

    explicit = _invoke_cli(tmp_path, "--diarize-model", "3.1")
    assert explicit.call_args.kwargs["diarize_model"] == LEGACY_MODEL

    monkeypatch.delenv("VOXWEAVE_DIARIZE_MODEL")
    (tmp_path / "voxweave.conf").unlink()
    builtin = _invoke_cli(tmp_path)
    assert builtin.call_args.kwargs["diarize_model"] == LEGACY_MODEL


def test_process_forwards_diarize_model_to_transcribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.wav"
    media.write_bytes(b"media")
    captured: dict[str, object] = {}

    def fake_transcribe(_source: Path, **kwargs: object):
        captured.update(kwargs)
        units = [{"text": "hello", "start": 0.0, "end": 1.0}]
        return "en", units, [(0.0, 1.0)], [], [], None

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)

    pipeline.process(
        media,
        diarize=True,
        diarize_model=COMMUNITY_MODEL,
        shot_snap=False,
    )

    assert captured["diarize_model"] == COMMUNITY_MODEL


def test_transcribe_forwards_diarize_model_to_diarize_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "episode.wav"
    media.write_bytes(b"media")
    wav = tmp_path / "speech.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(
        pipeline,
        "vad_speech_segments",
        lambda *_args, **_kwargs: [{"start": 0.0, "end": 1.0}],
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(backend, "chunk_pass_count", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(
        backend,
        "transcribe_chunks",
        lambda *_args, **_kwargs: [
            ("English", "hello", [{"text": "hello", "start": 0.0, "end": 1.0}])
        ],
    )
    monkeypatch.setattr(backend, "release", lambda: None)
    monkeypatch.setattr(chunking, "release_silero_vad", lambda: None)
    monkeypatch.setattr(songdet, "release_model", lambda: None)
    requested: list[str | None] = []

    def fake_diarize(_wav: Path, **kwargs: object) -> diarize.DiarizationResult:
        requested.append(kwargs.get("model"))
        return diarize.DiarizationResult(turns=[], centroids=None, provenance={})

    monkeypatch.setattr(diarize, "diarize_turns", fake_diarize)
    monkeypatch.setattr(diarize, "release", lambda: None)

    pipeline.transcribe(
        media,
        separate=False,
        diarize=True,
        diarize_model=COMMUNITY_MODEL,
    )

    assert requested == [COMMUNITY_MODEL]
