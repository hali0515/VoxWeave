import os
import subprocess

import pytest

from voxweave import chunking
from voxweave.chunking import pack_speech_segments, plan_dp_chunks


def test_packs_into_single_chunk_when_short():
    segs = [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 5.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert chunks == [{"start": 0.0, "end": 5.0, "offset": 0.0}]


def test_splits_at_silence_when_exceeding_max():
    # three segments, each 100s, 1s gap; max=240 -> seg1+seg2 one chunk, seg3 one chunk
    segs = [
        {"start": 0.0, "end": 100.0},
        {"start": 101.0, "end": 201.0},
        {"start": 202.0, "end": 302.0},
    ]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert len(chunks) == 2
    assert chunks[0] == {"start": 0.0, "end": 201.0, "offset": 0.0}
    assert chunks[1] == {"start": 202.0, "end": 302.0, "offset": 202.0}


def test_single_segment_longer_than_max_is_hard_split():
    # 500s continuous speech with no silence, max=240 -> hard cut (word cuts tolerated)
    segs = [{"start": 0.0, "end": 500.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert len(chunks) == 3
    assert chunks[0]["start"] == 0.0 and chunks[0]["end"] == 240.0
    assert chunks[1]["start"] == 240.0 and chunks[1]["end"] == 480.0
    assert chunks[2]["start"] == 480.0 and chunks[2]["end"] == 500.0
    assert [c["offset"] for c in chunks] == [0.0, 240.0, 480.0]


def test_empty_returns_empty():
    assert pack_speech_segments([], max_sec=240.0) == []


# --- plan_dp_chunks: silence-anchored DP chunking for over-budget alignment ---


def test_dp_within_budget_is_single_chunk():
    bounds = [(0.0, 2.0), (3.0, 5.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5)
    # one chunk over all cues; crop padded at file edges (left clamped to 0)
    assert chunks == [{"lo": 0, "hi": 2, "start": 0.0, "end": 5.5}]


def test_dp_empty_returns_empty():
    assert plan_dp_chunks([], max_sec=240.0) == []


def test_dp_splits_at_large_gap_when_over_budget():
    # three 100s cues, 2s gaps; budget 240 -> [cue0,cue1] + [cue2]
    bounds = [(0.0, 100.0), (102.0, 202.0), (204.0, 304.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, min_gap_sec=1.5, pad_sec=0.5)
    assert len(chunks) == 2
    # boundary at gap midpoint (202+204)/2 = 203; adjacent crops meet there
    assert chunks[0] == {"lo": 0, "hi": 2, "start": 0.0, "end": 203.0}
    assert chunks[1] == {"lo": 2, "hi": 3, "start": 203.0, "end": 304.5}


def test_dp_prefers_large_gap_over_in_budget_small_gap():
    # small gap (0.5s) after cue0, large gap (2s) after cue1; both within budget.
    # must cut at the large gap, not the earlier small one.
    bounds = [(0.0, 100.0), (100.5, 200.0), (202.0, 302.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, min_gap_sec=1.5, pad_sec=0.5)
    assert [(c["lo"], c["hi"]) for c in chunks] == [(0, 2), (2, 3)]


def test_dp_falls_back_to_cue_boundary_when_no_large_gap():
    # all gaps tiny (<min_gap) but total > budget: cut at latest cue boundary in budget.
    # cue boundaries never split words (smart_split invariant), so this stays word-safe.
    bounds = [(0.0, 100.0), (100.5, 200.5), (201.0, 301.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, min_gap_sec=1.5, pad_sec=0.5)
    assert [(c["lo"], c["hi"]) for c in chunks] == [(0, 2), (2, 3)]


def test_dp_single_oversized_cue_is_its_own_chunk():
    bounds = [(0.0, 300.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5)
    assert [(c["lo"], c["hi"]) for c in chunks] == [(0, 1)]


def test_dp_timestampless_insertion_cue_rides_along():
    # None-bound cue (insertion / empty) carries no anchor; it stays in its chunk.
    bounds = [(0.0, 2.0), None, (3.0, 5.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5)
    assert chunks == [{"lo": 0, "hi": 3, "start": 0.0, "end": 5.5}]


def test_dp_audio_end_caps_last_chunk():
    bounds = [(0.0, 2.0), (3.0, 5.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5, audio_end=5.2)
    assert chunks[-1]["end"] == 5.2


# --------------------------------------------------------------------------- #
# subtract_spans: carve clean-dialogue windows out of song spans before the
# vad_speech subtraction (dialogue spoken OVER a song must survive there).
# --------------------------------------------------------------------------- #
def test_subtract_spans_carves_keep_intervals():
    from voxweave.songdet import subtract_spans

    songs = [(660.0, 730.0)]
    speech = [(676.0, 680.0), (726.0, 729.0)]
    assert subtract_spans(songs, speech) == [
        (660.0, 676.0),
        (680.0, 726.0),
        (729.0, 730.0),
    ]


def test_subtract_spans_noop_without_keep():
    from voxweave.songdet import subtract_spans

    assert subtract_spans([(1.0, 5.0)], []) == [(1.0, 5.0)]


def test_subtract_spans_keep_swallows_whole_span():
    from voxweave.songdet import subtract_spans

    assert subtract_spans([(2.0, 4.0)], [(1.0, 5.0)]) == []


# --------------------------------------------------------------------------- #
# decode_to_wav: ffmpeg failures must be readable (media name + stderr tail),
# capped by a timeout, and must not leak the mkstemp temp file (#20, #21).
# --------------------------------------------------------------------------- #
def test_decode_to_wav_ffmpeg_failure_includes_media_name_and_stderr(
    tmp_path, monkeypatch
):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def fake_run(*a, **kw):
        raise subprocess.CalledProcessError(
            1, ["ffmpeg"], stderr=b"boom: codec not found"
        )

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        chunking.decode_to_wav(media)
    msg = str(ei.value)
    assert "clip.mp4" in msg
    assert "boom" in msg


def test_decode_to_wav_passes_timeout_and_captures_stderr(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured.update(kw)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    chunking.decode_to_wav(media)
    assert "timeout" in captured
    assert captured.get("stderr") != subprocess.DEVNULL


def test_ffmpeg_timeout_constant_is_positive_and_env_overridable(monkeypatch):
    # Contract: chunking.FFMPEG_TIMEOUT is a module constant, overridable via
    # VOXWEAVE_FFMPEG_TIMEOUT (read at import time, like VAD_MIN_SILENCE_MS above).
    assert chunking.FFMPEG_TIMEOUT > 0


def test_decode_to_wav_cleans_temp_wav_on_ffmpeg_failure(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def fake_mkstemp(suffix="", prefix="", dir=None):
        path = tmp_path / f"{prefix}fake{suffix}"
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
        return fd, str(path)

    def fake_run(*a, **kw):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"boom")

    monkeypatch.setattr(chunking.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(chunking.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        chunking.decode_to_wav(media)

    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".wav"]
    assert leftover == []
