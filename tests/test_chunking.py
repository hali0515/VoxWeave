import os
import shutil
import subprocess

import numpy as np
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


def test_hard_cut_remainder_merges_with_following_segment():
    # 240.04s of speech leaves a 40ms remainder after the hard cut; it stays the open
    # block, so the next segment merges into it instead of it being a sliver chunk.
    segs = [{"start": 0.0, "end": 240.04}, {"start": 241.0, "end": 250.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert chunks == [
        {"start": 0.0, "end": 240.0, "offset": 0.0},
        {"start": 240.0, "end": 250.0, "offset": 240.0},
    ]


def test_hard_cut_remainder_still_closes_when_next_segment_does_not_fit():
    # remainder 240-300 + next segment up to 500 would span 260s > 240: close at silence
    segs = [{"start": 0.0, "end": 300.0}, {"start": 301.0, "end": 500.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert [(c["start"], c["end"]) for c in chunks] == [
        (0.0, 240.0),
        (240.0, 300.0),
        (301.0, 500.0),
    ]


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


def test_decode_command_pins_the_channel_count(tmp_path):
    media = tmp_path / "clip.mkv"
    out = tmp_path / "out.wav"

    mono = chunking.decode_command(media, out, audio_filter="loudnorm")
    assert mono == [
        "ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(media),
        "-af", "loudnorm",
        "-ac", "1",
        "-ar", "16000", "-f", "wav", str(out),
    ]  # fmt: skip

    # Full band for separation: a 5.1 / 7.1 source is downmixed to stereo
    # before it can reach the stereo-only Roformer.
    fullband = chunking.decode_command(media, out, sample_rate=44100, mono=False)
    assert fullband == [
        "ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(media),
        "-af", "aformat=channel_layouts=mono|stereo",
        "-ar", "44100", "-f", "wav", str(out),
    ]  # fmt: skip
    assert chunking.STEREO_CAP_FILTER == "aformat=channel_layouts=mono|stereo"

    filtered = chunking.decode_command(
        media, out, sample_rate=44100, mono=False, audio_filter="loudnorm"
    )
    assert filtered[filtered.index("-af") + 1] == (
        "loudnorm,aformat=channel_layouts=mono|stereo"
    )
    assert "-ac" not in filtered


def test_decode_to_wav_runs_the_decode_command(tmp_path, monkeypatch):
    media = tmp_path / "clip.mkv"
    media.write_bytes(b"x")
    commands: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    out = chunking.decode_to_wav(media, sample_rate=44100, mono=False)
    try:
        assert commands == [
            chunking.decode_command(media, out, sample_rate=44100, mono=False)
        ]
    finally:
        out.unlink(missing_ok=True)


_FFMPEG_TOOLS = shutil.which("ffmpeg") is not None and shutil.which("ffprobe")


@pytest.mark.skipif(not _FFMPEG_TOOLS, reason="needs ffmpeg and ffprobe")
@pytest.mark.parametrize(
    ("layout", "codec", "channels"),
    [("5.1", "eac3", 2), ("7.1", "pcm_s16le", 2), ("stereo", "pcm_s16le", 2),
     ("mono", "pcm_s16le", 1)],
)  # fmt: skip
def test_real_ffmpeg_caps_the_fullband_decode_at_stereo(
    tmp_path, layout, codec, channels
):
    import soundfile as sf

    source = tmp_path / f"tone.{'mka' if codec == 'eac3' else 'wav'}"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", f"aevalsrc=0.2*sin(2*PI*440*t):c={layout}:s=48000:d=0.3",
            "-c:a", codec, str(source),
        ],
        check=True, stdin=subprocess.DEVNULL, timeout=60,
    )  # fmt: skip
    probed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels", "-of", "csv=p=0", str(source),
        ],
        check=True, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert int(probed.stdout.strip()) == {"5.1": 6, "7.1": 8}.get(layout, channels)

    wav = chunking.decode_to_wav(source, sample_rate=44100, mono=False)
    try:
        data, rate = sf.read(str(wav), always_2d=True)
    finally:
        wav.unlink(missing_ok=True)

    assert rate == 44100
    assert data.shape[1] == channels
    # Mono stays at unity (no -3 dB upmix); the downmix keeps the tone audible.
    assert np.abs(data).max() > 0.15


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


def test_decode_to_wav_missing_ffmpeg_is_friendly_and_cleans_temp(
    tmp_path, monkeypatch
):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def fake_mkstemp(suffix="", prefix="", dir=None):
        path = tmp_path / f"{prefix}fake{suffix}"
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
        return fd, str(path)

    def fake_run(*a, **kw):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(chunking.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(chunking.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg not found.*PATH") as ei:
        chunking.decode_to_wav(media)
    assert isinstance(ei.value.__cause__, FileNotFoundError)
    assert [p for p in tmp_path.iterdir() if p.suffix == ".wav"] == []


def test_decode_to_wav_timeout_names_the_knob_and_cleans_temp(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def fake_mkstemp(suffix="", prefix="", dir=None):
        path = tmp_path / f"{prefix}fake{suffix}"
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
        return fd, str(path)

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"], stderr=b"size=  1kB")

    monkeypatch.setattr(chunking.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    monkeypatch.setattr(chunking, "FFMPEG_TIMEOUT", 12.0)

    with pytest.raises(RuntimeError) as ei:
        chunking.decode_to_wav(media)
    msg = str(ei.value)
    assert "timed out after 12s" in msg and "VOXWEAVE_FFMPEG_TIMEOUT" in msg
    assert "clip.mp4" in msg
    assert [p for p in tmp_path.iterdir() if p.suffix == ".wav"] == []


def test_decode_to_wav_error_keeps_only_the_stderr_tail(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    stderr = "\n".join(f"line {i}" for i in range(40)).encode()

    def fake_run(*a, **kw):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr=stderr)

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        chunking.decode_to_wav(media)
    msg = str(ei.value)
    assert "line 39" in msg and "line 32" in msg
    assert "line 31" not in msg


def test_vad_rejects_wrong_sample_rate_with_value_error(tmp_path, monkeypatch):
    import soundfile as sf

    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(800, dtype=np.float32), 8000)
    monkeypatch.setattr(chunking, "_get_silero_vad", lambda: object())
    pytest.importorskip("silero_vad")
    with pytest.raises(ValueError, match="expected 16000 Hz"):
        chunking.vad_speech_segments(wav)


def test_malformed_env_knob_warns_and_keeps_default(monkeypatch, caplog):
    monkeypatch.setenv("VOXWEAVE_VAD_MIN_SILENCE_MS", "300ms")
    monkeypatch.setenv("VOXWEAVE_FFMPEG_TIMEOUT", "1h")
    with caplog.at_level("WARNING", logger="voxweave"):
        assert chunking._env_int("VOXWEAVE_VAD_MIN_SILENCE_MS", 300) == 300
        assert chunking._env_float("VOXWEAVE_FFMPEG_TIMEOUT", 3600.0) == 3600.0
    text = caplog.text
    assert "VOXWEAVE_VAD_MIN_SILENCE_MS" in text and "VOXWEAVE_FFMPEG_TIMEOUT" in text
    monkeypatch.setenv("VOXWEAVE_VAD_MIN_SILENCE_MS", " 150 ")
    assert chunking._env_int("VOXWEAVE_VAD_MIN_SILENCE_MS", 300) == 150


def test_malformed_env_knobs_do_not_break_import():
    # a typo'd knob used to raise at import time, killing every command (even --help)
    import sys

    env = {
        **os.environ,
        "VOXWEAVE_VAD_MIN_SILENCE_MS": "abc",
        "VOXWEAVE_FFMPEG_TIMEOUT": "1h",
        "VOXWEAVE_QWEN_MAX_NEW_TOKENS": "1k",
        "VOXWEAVE_ASR_BATCH_MIN_CPS": "x",
        "VOXWEAVE_ASR_BATCH_MIN_CHECK_SEC": "2s",
    }
    code = (
        "from voxweave import backend, chunking\n"
        "print(chunking.VAD_MIN_SILENCE_MS, chunking.FFMPEG_TIMEOUT,"
        " backend.QWEN_MAX_NEW_TOKENS, backend.ASR_BATCH_MIN_CPS,"
        " backend.ASR_BATCH_MIN_CHECK_SEC)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["300", "3600.0", "1024", "0.5", "2.0"]
    for name in ("VOXWEAVE_VAD_MIN_SILENCE_MS", "VOXWEAVE_QWEN_MAX_NEW_TOKENS"):
        assert name in proc.stderr
