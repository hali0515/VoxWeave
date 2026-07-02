# tests/test_diarize_smoothing.py
# Fix 4: raw pyannote turns are noisy (16-31% <0.5s, overlap-track fragments fully
# contained in another speaker's turn). diarize_turns smooths the list before
# persisting: merge consecutive same-speaker turns across sub-0.35s gaps, drop
# sub-0.2s turns fully contained in another speaker's turn. Thresholds are module
# constants overridable by env. Plus --min-speakers/--max-speakers plumbing
# (CLI -> pipeline.process -> transcribe -> diarize_turns).

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from voxweave import diarize, pipeline
from voxweave.cli import cli
from voxweave.diarize import _smooth_turns


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))


# --- _smooth_turns ----------------------------------------------------------


def test_merge_same_speaker_across_small_gap():
    assert _smooth_turns([(0.0, 1.0, "A"), (1.2, 2.0, "A")]) == [(0.0, 2.0, "A")]


def test_drop_short_turn_contained_in_other_speaker():
    assert _smooth_turns([(0.0, 2.0, "A"), (0.5, 0.55, "B")]) == [(0.0, 2.0, "A")]


def test_standalone_short_interjection_preserved():
    turns = [(0.0, 1.0, "A"), (1.5, 1.8, "B"), (2.0, 3.0, "A")]
    assert _smooth_turns(turns) == turns


def test_clean_input_unchanged():
    turns = [(0.0, 1.0, "A"), (2.0, 3.0, "B"), (4.0, 5.0, "A")]
    assert _smooth_turns(turns) == turns


def test_merge_gap_env_override(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_DIARIZE_MERGE_GAP_S", "0.1")
    # gap 0.2 now exceeds the 0.1 threshold -> not merged
    assert _smooth_turns([(0.0, 1.0, "A"), (1.2, 2.0, "A")]) == [
        (0.0, 1.0, "A"),
        (1.2, 2.0, "A"),
    ]


def test_drop_contained_env_override(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_DIARIZE_DROP_CONTAINED_S", "0.01")
    # 0.05s turn now above the 0.01 drop threshold -> kept
    assert _smooth_turns([(0.0, 2.0, "A"), (0.5, 0.55, "B")]) == [
        (0.0, 2.0, "A"),
        (0.5, 0.55, "B"),
    ]


# --- diarize_turns applies smoothing ----------------------------------------


class _FakeSeg:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for seg, name, label in self._tracks:
            yield (seg, name, label) if yield_label else (seg, name)


class _FakePipeline:
    def __init__(self, tracks):
        self._tracks = tracks
        self.calls = []

    def __call__(self, file, **kwargs):
        self.calls.append((file, kwargs))
        return _FakeAnnotation(self._tracks)


def _wav(tmp_path):
    p = tmp_path / "clip.wav"
    sf.write(str(p), (np.random.randn(16000) * 0.01).astype("float32"), 16000)
    return p


def test_diarize_turns_smooths_output(monkeypatch, tmp_path):
    # two same-speaker turns 0.2s apart -> merged into one after smoothing
    tracks = [
        (_FakeSeg(0.0, 1.0), "a", "SPEAKER_00"),
        (_FakeSeg(1.2, 2.0), "b", "SPEAKER_00"),
    ]
    fake = _FakePipeline(tracks)
    monkeypatch.setattr(diarize, "_get_pipeline", lambda token: fake)
    turns = diarize.diarize_turns(_wav(tmp_path), token="hf_test")
    assert turns == [(0.0, 2.0, "SPEAKER_00")]


def test_diarize_turns_forwards_min_max_speakers(monkeypatch, tmp_path):
    fake = _FakePipeline([(_FakeSeg(0.0, 1.0), "a", "SPEAKER_00")])
    monkeypatch.setattr(diarize, "_get_pipeline", lambda token: fake)
    diarize.diarize_turns(
        _wav(tmp_path), token="hf_test", min_speakers=2, max_speakers=3
    )
    _, kwargs = fake.calls[0]
    assert kwargs.get("min_speakers") == 2
    assert kwargs.get("max_speakers") == 3


# --- min/max speaker plumbing -----------------------------------------------


def _media(tmp_path):
    m = tmp_path / "a.wav"
    m.write_bytes(b"x")
    out = tmp_path / "a.vtt"
    out.write_text("WEBVTT\n", encoding="utf-8")
    return m, out


def test_cli_process_passes_min_max_speakers(tmp_path, monkeypatch):
    from unittest.mock import patch

    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(
            cli, ["--min-speakers", "2", "--max-speakers", "3", str(media)]
        )
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["min_speakers"] == 2
    assert m.call_args.kwargs["max_speakers"] == 3


def test_cli_process_speakers_default_none(tmp_path):
    from unittest.mock import patch

    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["min_speakers"] is None
    assert m.call_args.kwargs["max_speakers"] is None


def test_process_forwards_min_max_speakers_to_transcribe(tmp_path, monkeypatch):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    captured = {}

    def fake_transcribe(path, **kw):
        captured.update(kw)
        units = [{"text": "hi", "start": 0.0, "end": 0.5}]
        return "en", units, [(0.0, 0.5)], [], []

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    pipeline.process(
        media, diarize=True, min_speakers=2, max_speakers=3, shot_snap=False
    )
    assert captured["min_speakers"] == 2
    assert captured["max_speakers"] == 3
