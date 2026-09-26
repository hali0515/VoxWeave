"""Regression tests for the pipeline.py audit fixes (see each test's comment)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from voxweave import backend, chunking, config, pipeline, songdet
from voxweave.progress import Reporter

UNITS = [
    {"text": "hello", "start": 0.0, "end": 0.5},
    {"text": "world", "start": 0.5, "end": 1.0},
]


def _stub_transcribe_stages(tmp_path, monkeypatch, *, vad):
    """Replace every heavy transcribe stage; return the decode_to_wav call log."""
    decoded: list[tuple[Path, dict]] = []

    def fake_decode(source, **kwargs):
        out = tmp_path / f"decoded-{len(decoded)}.wav"
        out.write_bytes(b"audio")
        decoded.append((Path(source), kwargs))
        return out

    def fake_slice(_wav, _start, _end, **_kwargs):
        out = tmp_path / f"chunk-{len(list(tmp_path.glob('chunk-*')))}.wav"
        out.write_bytes(b"chunk")
        return out

    def transcribe_chunks(chunks, _language, **kwargs):
        for index in range(len(chunks)):
            kwargs["on_done"](index)
        return [("en", "hello world", [dict(u) for u in UNITS]) for _ in chunks]

    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
    monkeypatch.setattr(pipeline, "slice_wav", fake_slice)
    monkeypatch.setattr(pipeline, "vad_speech_segments", vad)
    monkeypatch.setattr(backend, "chunk_pass_count", lambda _model: 1)
    monkeypatch.setattr(backend, "transcribe_chunks", transcribe_chunks)
    monkeypatch.setattr(backend, "release", lambda: None)
    monkeypatch.setattr(chunking, "release_silero_vad", lambda: None)
    monkeypatch.setattr(songdet, "release_model", lambda: None)
    return decoded


def _vad_by_threshold(calls):
    """Separated-vocals VAD (default threshold) and original-audio VAD differ."""

    def vad(wav, **kwargs):
        calls.append((Path(wav), kwargs))
        if kwargs.get("threshold") == pipeline.SNAP_VAD_THRESHOLD:
            return [{"start": 0.0, "end": 1.2}]  # original mix keeps a back-channel
        return [{"start": 0.0, "end": 1.0}]

    return vad


@pytest.mark.parametrize("cache_hit", [False, True])
def test_original_audio_vad_reference_survives_a_vocals_cache_hit(
    tmp_path, monkeypatch, cache_hit
):
    # A vocals-cache hit has no full-band stem; the sensitive original-audio VAD
    # reference must still be computed (from the source media), or a re-run's
    # vad_speech / snapping silently differs from the first run's.
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    cache = tmp_path / "cache" / "vocals.32k.flac"
    cache.parent.mkdir()
    cache.write_bytes(b"flac")
    vad_calls: list[tuple[Path, dict]] = []
    decoded = _stub_transcribe_stages(
        tmp_path, monkeypatch, vad=_vad_by_threshold(vad_calls)
    )
    fullband = tmp_path / "fullband.wav"
    parts = [fullband] + [tmp_path / n for n in ("voc.wav", "16k.wav", "32k.wav")]
    for part in parts:
        part.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "_vocals_cache_fresh", lambda *_a: cache_hit)
    monkeypatch.setattr(pipeline, "_separate_to_16k_32k", lambda *_a, **_k: parts)
    monkeypatch.setattr(pipeline, "_encode_flac", lambda *_a: None)

    result = pipeline.transcribe(media, cache_vocals=cache)

    reference_source = media if cache_hit else fullband
    reference = [
        (i, source) for i, (source, kw) in enumerate(decoded) if kw == {}
    ]  # the default 16k mono decode, same on both paths
    assert [source for _i, source in reference] == [reference_source]
    reference_wav = tmp_path / f"decoded-{reference[0][0]}.wav"
    assert (reference_wav, {"threshold": pipeline.SNAP_VAD_THRESHOLD}) in vad_calls
    assert result[2] == [(0.0, 1.2)]  # vad_spans come from the original audio
    assert not reference_wav.exists()  # registered for cleanup


def test_transcribe_release_failure_neither_masks_the_error_nor_skips_cleanup(
    tmp_path, monkeypatch, caplog
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    decoded = _stub_transcribe_stages(tmp_path, monkeypatch, vad=lambda *_a, **_k: [])
    released: list[str] = []

    def cuda_fault():
        released.append("backend")
        raise RuntimeError("CUDA error: device-side assert")

    monkeypatch.setattr(backend, "release", cuda_fault)
    monkeypatch.setattr(
        chunking, "release_silero_vad", lambda: released.append("silero")
    )
    monkeypatch.setattr(songdet, "release_model", lambda: released.append("panns"))

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        with pytest.raises(RuntimeError, match="no speech detected"):
            pipeline.transcribe(media, separate=False)

    # post-detection PANNs release, then every finally release despite the fault
    assert released == ["panns", "backend", "silero", "panns"]
    assert decoded and not any(
        (tmp_path / f"decoded-{i}.wav").exists() for i in range(len(decoded))
    )
    assert "CUDA error" in caplog.text


def test_separation_cleans_its_temps_on_keyboard_interrupt(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    made: list[Path] = []

    def fake_decode(_source, **_kwargs):
        out = tmp_path / f"full-{len(made)}.wav"
        out.write_bytes(b"x" * 1024)
        made.append(out)
        return out

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
    monkeypatch.setattr(backend, "separate_vocals", interrupted)

    with pytest.raises(KeyboardInterrupt):
        pipeline._separate_to_16k_32k(media, reporter=Reporter(), normalize=False)

    assert made and not any(path.exists() for path in made)


# --- vocals cache: a missing/hung ffprobe is not an unreadable cache ---------


def _paths(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"m")
    cache = pipeline.cache_vocals_path(media)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"c")
    return media, cache


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError(2, "No such file", "ffprobe"), "ffprobe not found on PATH"),
        (subprocess.TimeoutExpired(["ffprobe"], 60), "ffprobe timed out"),
    ],
)
def test_ffprobe_unavailable_is_reported_as_such(
    tmp_path, monkeypatch, caplog, error, message
):
    media, cache = _paths(tmp_path)

    def run(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert pipeline._vocals_cache_fresh(cache, media) is False  # re-separate

    assert message in caplog.text
    assert "cannot validate the vocals cache" in caplog.text
    assert "unreadable" not in caplog.text


def test_ffprobe_failure_on_the_file_is_still_unreadable(tmp_path, monkeypatch, caplog):
    media, cache = _paths(tmp_path)
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 1, "", "Invalid data"),
    )
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert pipeline._vocals_cache_fresh(cache, media) is False

    assert "vocals cache unreadable" in caplog.text


# --- import-time env knobs ---------------------------------------------------


@pytest.mark.parametrize(
    ("helper", "default"), [(config._env_int, 100), (config._env_float, 0.25)]
)
def test_env_helpers_warn_naming_the_variable_and_fall_back(
    monkeypatch, caplog, helper, default
):
    monkeypatch.setenv("VOXWEAVE_AUDIT_KNOB", "O.3")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert helper("VOXWEAVE_AUDIT_KNOB", default) == default
    assert "VOXWEAVE_AUDIT_KNOB" in caplog.text

    caplog.clear()
    monkeypatch.setenv("VOXWEAVE_AUDIT_KNOB", "  ")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert helper("VOXWEAVE_AUDIT_KNOB", default) == default
    assert caplog.text == ""  # blank means unset, not malformed

    monkeypatch.setenv("VOXWEAVE_AUDIT_KNOB", "7")
    got = helper("VOXWEAVE_AUDIT_KNOB", default)
    assert got == 7 and type(got) is type(default)


def test_malformed_pipeline_knob_does_not_break_import():
    # Every command imports pipeline (even --help): a typo must warn, not crash.
    env = dict(os.environ)
    env.update(
        VOXWEAVE_MAX_CHUNK_SEC="12O",
        VOXWEAVE_SONG_FINE_SILENCE_MS="100ms",
        VOXWEAVE_CACHE_DUR_TOL_SEC="half",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from voxweave import pipeline as p;"
            "print(p.MAX_CHUNK_SEC, p.SONG_FINE_SILENCE_MS, p.CACHE_DUR_TOL_SEC)",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["120.0", "100", "0.5"]
    for name in (
        "VOXWEAVE_MAX_CHUNK_SEC",
        "VOXWEAVE_SONG_FINE_SILENCE_MS",
        "VOXWEAVE_CACHE_DUR_TOL_SEC",
    ):
        assert name in proc.stderr


# --- split / replay ------------------------------------------------------------


def test_language_repair_reads_word_keyed_units():
    # schema.Unit allows the surface under ``word``: a stale ``en`` label over a
    # coarse Han paragraph must be repaired exactly as the ``text`` form is.
    text = "这是足够长且脚本特征明确的中文正文内容我们今天继续测试"
    as_text = [{"text": text, "start": 1.0, "end": 9.0}]
    as_word = [{"word": text, "start": 1.0, "end": 9.0}]

    iso_text, units_text = pipeline._reconcile_word_segment_language("en", as_text)
    iso_word, units_word = pipeline._reconcile_word_segment_language("en", as_word)

    assert iso_text == iso_word == "zh"
    assert units_word == units_text
    assert "".join(u["text"] for u in units_word) == text
    assert len(units_word) == len(text)
    assert as_word == [{"word": text, "start": 1.0, "end": 9.0}]  # input untouched


@pytest.mark.parametrize(
    "word_segments",
    [
        ["hello", "world"],
        [{"text": "hello", "start": 0.0, "end": 0.5}, 3],
        {"text": "hello"},
    ],
)
def test_split_rejects_malformed_word_segments_with_regenerate_hint(
    tmp_path, word_segments
):
    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps({"language": "en", "segments": [], "word_segments": word_segments}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="word_segments") as caught:
        pipeline.split(json_path)

    assert "re-run `voxweave transcribe <media>` to regenerate it" in str(caught.value)
    assert caught.value.failure.detail_code == "sibling-top-level-shape"


def test_corrupt_sibling_names_the_real_command(tmp_path):
    json_path = tmp_path / "episode.json"
    json_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError) as caught:
        pipeline.split(json_path)

    message = str(caught.value)
    assert "voxweave transcribe <media>" in message
    assert "process" not in message


# --- per-cue (Qwen) alignment progress ----------------------------------------


class _EventReporter(Reporter):
    def __init__(self, events: list) -> None:
        self.events = events

    def task(self, label: str, total: int) -> None:
        self.events.append(("task", label, total))

    def advance(self, n: int = 1) -> None:
        self.events.append(("advance", n))


def test_per_cue_alignment_advances_as_each_cue_lands(tmp_path, monkeypatch):
    events: list = []
    chunks: list[Path] = []

    def fake_slice(_wav, start, _end, **kwargs):
        kwargs["_sample_geometry_observer"](int(start * 16000), 0, 16000, 1)
        out = tmp_path / f"cue-{len(chunks)}.wav"
        out.write_bytes(b"x")
        return out

    def align_text(_wav, text, _iso):
        events.append(("align", text))
        return [{"text": text, "start": 0.0, "end": 0.1}]

    monkeypatch.setattr(pipeline, "slice_wav", fake_slice)
    monkeypatch.setattr(backend, "align_text", align_text)
    blocks = [
        {"text": "一", "start": 0.0, "end": 1.0},
        {"text": "二", "start": None, "end": None},  # insertion block: no crop
        {"text": "三", "start": 2.0, "end": 3.0},
    ]

    pipeline._align_blocks(
        tmp_path / "prepared.wav",
        blocks,
        "zh",
        mms=False,
        ctc_model=None,
        crops=[(0.0, 1.0), None, (2.0, 3.0)],
        reporter=_EventReporter(events),
        tmp_chunks=chunks,
    )

    assert events == [
        ("task", "per-cue alignment", 3),
        ("advance", 1),  # the skipped insertion block, at preparation
        ("align", "一"),
        ("advance", 1),
        ("align", "三"),
        ("advance", 1),
    ]


# --- correct --apply with re-alignment ---------------------------------------


def _vtt(tmp_path: Path) -> Path:
    path = tmp_path / "episode.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8"
    )
    return path


def _fix_hello(monkeypatch, calls: list | None = None):
    def fixes(payload, **_kwargs):
        if calls is not None:
            calls.append(payload)
        return [{"i": 0, "orig": "hello", "fixed": "hallo", "reason": "typo"}]

    monkeypatch.setattr(pipeline.asrfix_mod, "correct_cues", fixes)


def test_correct_rejects_missing_explicit_media_before_the_llm_call(
    tmp_path, monkeypatch
):
    vtt = _vtt(tmp_path)
    original = vtt.read_bytes()
    calls: list = []
    _fix_hello(monkeypatch, calls)

    with pytest.raises(FileNotFoundError, match="source media for episode.vtt") as e:
        pipeline.correct(
            vtt, apply=True, align_after=True, media_path=tmp_path / "gone.mkv"
        )

    assert e.value.failure.detail_code == "media-not-found"
    assert calls == []  # no LLM round trip wasted
    assert vtt.read_bytes() == original  # nothing committed


def test_correct_keeps_the_diff_when_realignment_fails_after_commit(
    tmp_path, monkeypatch, caplog
):
    vtt = _vtt(tmp_path)
    _fix_hello(monkeypatch)

    def no_media(*_args, **_kwargs):
        raise pipeline._media_not_found_error(vtt)

    monkeypatch.setattr(pipeline, "align", no_media)
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        with pytest.raises(FileNotFoundError, match="source media"):
            pipeline.correct(vtt, apply=True, align_after=True)

    assert "hallo" in vtt.read_text(encoding="utf-8")  # committed before align
    assert "'hello' -> 'hallo'" in caplog.text
    assert f"voxweave align {vtt}" in caplog.text
