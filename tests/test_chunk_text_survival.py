"""Chunk-level text survival: a failed alignment must never delete the transcript.

Alignment is a timing refinement, not a source of truth for words. When the aligner
blows up on one chunk the ASR text is still the best transcript available, so the
chunk degrades to evenly spread word timing instead of vanishing from the output.
"""

import math
from itertools import pairwise

import numpy as np
import pytest
import soundfile as sf

from voxweave import backend

SR = 16000


@pytest.fixture(autouse=True)
def _force_torch_backend(monkeypatch):
    """Pin the torch backend so engine dispatch is deterministic on any host."""
    monkeypatch.setattr(backend, "_use_mlx", lambda: False)


def _write_wav(path, seconds: float, sr: int = SR):
    """Deterministic mono wav of the requested duration (real file: duration is readable)."""
    n = round(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = (0.3 * np.sin(2.0 * math.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(path), sig, sr)
    return path


def _assert_monotone_cover(units, duration):
    """Units tile [0, duration] end to end without overlap or gaps."""
    assert units, "fallback units must not be empty for non-empty text"
    assert units[0]["start"] == pytest.approx(0.0, abs=1e-6)
    assert units[-1]["end"] == pytest.approx(duration, abs=1e-6)
    for u in units:
        assert u["end"] >= u["start"]
    for a, b in pairwise(units):
        assert b["start"] == pytest.approx(a["end"], abs=1e-6)


# --- _fallback_units: evenly distributed timing for a chunk with no alignment ---


def test_fallback_units_spaced_language_splits_on_whitespace():
    units = backend._fallback_units("hello world", "en", 10.0)
    assert [u["text"] for u in units] == ["hello", "world"]
    _assert_monotone_cover(units, 10.0)
    assert units[0]["end"] == pytest.approx(5.0, abs=1e-6)


def test_fallback_units_spaced_language_collapses_runs_of_whitespace():
    units = backend._fallback_units("  a   b\tc \n", "en", 3.0)
    assert [u["text"] for u in units] == ["a", "b", "c"]
    _assert_monotone_cover(units, 3.0)


def test_fallback_units_no_space_language_is_per_character():
    units = backend._fallback_units("你好世界", "zh", 4.0)
    assert [u["text"] for u in units] == ["你", "好", "世", "界"]
    _assert_monotone_cover(units, 4.0)


def test_fallback_units_no_space_language_drops_spaces():
    # "per non-space character": interior whitespace is layout, not a unit
    units = backend._fallback_units("こん にちは", "ja", 5.0)
    assert [u["text"] for u in units] == ["こ", "ん", "に", "ち", "は"]
    _assert_monotone_cover(units, 5.0)


def test_fallback_units_empty_text_is_empty():
    assert backend._fallback_units("", "en", 10.0) == []
    assert backend._fallback_units("   \n\t ", "ja", 10.0) == []


def test_fallback_units_single_token_spans_whole_chunk():
    units = backend._fallback_units("hello", "en", 7.5)
    assert [u["text"] for u in units] == ["hello"]
    _assert_monotone_cover(units, 7.5)


# --- _align_chunk_safe: aligner failure keeps the words, loses only the timing ---


def test_align_chunk_safe_returns_fallback_units_when_aligner_raises(
    monkeypatch, tmp_path, caplog
):
    import logging

    wav = _write_wav(tmp_path / "c0.wav", 2.0)

    def _boom(w, text, alang):
        raise RuntimeError("CUDA error: device-side assert")

    monkeypatch.setattr(backend, "align_text", _boom)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        units = backend._align_chunk_safe(wav, "hello there world", "en", 0, 3)

    # the transcript survives: every ASR word is still present, in order
    assert [u["text"] for u in units] == ["hello", "there", "world"]
    _assert_monotone_cover(units, 2.0)
    # containment is still reported, not silently swallowed
    assert "alignment failed on chunk 1/3" in caplog.text


def test_align_chunk_safe_empty_text_stays_empty(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path / "c0.wav", 2.0)

    def _boom(w, text, alang):
        raise RuntimeError("boom")

    monkeypatch.setattr(backend, "align_text", _boom)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    assert backend._align_chunk_safe(wav, "   ", "en", 0, 1) == []


def test_align_chunk_safe_passes_through_on_success(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path / "c0.wav", 2.0)
    real = [{"text": "ok", "start": 0.25, "end": 0.75}]
    monkeypatch.setattr(backend, "align_text", lambda w, t, a: real)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    assert backend._align_chunk_safe(wav, "ok", "en", 0, 1) == real


# --- transcribe_chunks: text conservation across the whole chunk stream ---


def test_transcribe_chunks_conserves_text_of_align_failed_chunk(monkeypatch, tmp_path):
    # one chunk's alignment failing must not delete its words from the unit stream
    def _align(w, text, alang):
        if w.name == "c1.wav":
            raise RuntimeError("boom")
        return [{"text": text, "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("English", "alpha beta", "en")
    )
    monkeypatch.setattr(backend, "align_text", _align)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)

    wavs = [_write_wav(tmp_path / f"c{i}.wav", 2.0) for i in range(3)]
    out = backend.transcribe_chunks(wavs, None, asr_model="qwen3-asr-1.7b")

    assert len(out) == 3
    # the failed chunk keeps both its text and a usable unit stream
    assert out[1][1] == "alpha beta"
    assert [u["text"] for u in out[1][2]] == ["alpha", "beta"]
    _assert_monotone_cover(out[1][2], 2.0)
    # its neighbours are untouched by the failure
    assert out[0][2] == [{"text": "alpha beta", "start": 0.0, "end": 1.0}]
    assert out[2][2] == [{"text": "alpha beta", "start": 0.0, "end": 1.0}]


def test_transcribe_chunks_conserves_text_when_every_alignment_fails(
    monkeypatch, tmp_path
):
    # alignment dying everywhere is not an ASR failure: no raise, no lost words
    def _boom(w, text, alang):
        raise RuntimeError("aligner gone")

    monkeypatch.setattr(
        backend, "_asr_only", lambda e, w, lang, m, c: ("Japanese", "こんにちは", "ja")
    )
    monkeypatch.setattr(backend, "align_text", _boom)
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)

    wavs = [_write_wav(tmp_path / f"c{i}.wav", 2.0) for i in range(2)]
    out = backend.transcribe_chunks(wavs, None, asr_model="qwen3-asr-1.7b")

    for _, text, units in out:
        assert text == "こんにちは"
        assert "".join(u["text"] for u in units) == "こんにちは"
        _assert_monotone_cover(units, 2.0)
