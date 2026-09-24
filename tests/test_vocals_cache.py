"""Vocals cache duration freshness: a replaced/trimmed source must invalidate the cache."""

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from voxweave import pipeline, vocalscache


def _durations(mapping):
    """Patch _probe_duration to look paths up by name in mapping (None = unprobeable)."""
    return patch(
        "voxweave.pipeline._probe_duration",
        side_effect=lambda p: mapping.get(p.name),
    )


def _paths(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"m")
    cache = pipeline.cache_vocals_path(media)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"c")
    return media, cache


def test_fresh_when_durations_match(tmp_path):
    media, cache = _paths(tmp_path)
    with _durations({media.name: 1420.0, cache.name: 1420.2}):
        assert pipeline._vocals_cache_fresh(cache, media)


def test_stale_when_durations_diverge(tmp_path):
    media, cache = _paths(tmp_path)
    with _durations({media.name: 1360.0, cache.name: 1420.0}):
        assert not pipeline._vocals_cache_fresh(cache, media)


def test_stale_when_cache_unreadable(tmp_path):
    media, cache = _paths(tmp_path)
    with _durations({media.name: 1420.0, cache.name: None}):
        assert not pipeline._vocals_cache_fresh(cache, media)


def test_fresh_when_media_unprobeable(tmp_path):
    media, cache = _paths(tmp_path)
    with _durations({media.name: None, cache.name: 1420.0}):
        assert pipeline._vocals_cache_fresh(cache, media)


def test_prepare_align_reuses_fresh_cache(tmp_path):
    media, cache = _paths(tmp_path)
    wav = tmp_path / "out.wav"
    tmp: list = []
    with (
        _durations({media.name: 100.0, cache.name: 100.0}),
        patch("voxweave.pipeline.decode_to_wav", return_value=wav) as dec,
        patch("voxweave.pipeline._separate_to_16k_32k") as sep,
    ):
        got = pipeline._prepare_16k_for_align(
            media, separate=True, normalize=False, reporter=pipeline.Reporter(), tmp=tmp
        )
    assert got == wav
    dec.assert_called_once()
    sep.assert_not_called()


@pytest.mark.parametrize("normalize", [True, False])
def test_first_run_16k_decode_matches_a_cache_hit(tmp_path, normalize, monkeypatch):
    # A cache hit decodes vocals.32k.flac (the 32k mono vocals, losslessly) to 16k;
    # a first run must decode the same 32k vocals with the same options, or the two
    # runs feed ASR/diarization differently normalized audio.
    media, cache = _paths(tmp_path)
    with (
        _durations({media.name: 100.0, cache.name: 100.0}),
        patch("voxweave.pipeline.decode_to_wav", return_value=tmp_path / "o") as dec,
    ):
        pipeline._prepare_16k_for_align(
            media,
            separate=True,
            normalize=normalize,
            reporter=pipeline.Reporter(),
            tmp=[],
        )
    ((hit_source,), hit_kwargs) = dec.call_args
    assert hit_source == cache

    decoded: list[tuple[Path, dict]] = []

    def fake_decode(source, **kwargs):
        out = tmp_path / f"d{len(decoded)}.wav"
        decoded.append((Path(source), kwargs))
        return out

    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
    monkeypatch.setattr(
        pipeline.backend, "separate_vocals", lambda *_a, **_k: tmp_path / "v.wav"
    )
    _full, _vocals, wav, voc32 = pipeline._separate_to_16k_32k(
        media, reporter=pipeline.Reporter(), normalize=normalize
    )
    first_source, first_kwargs = decoded[-1]
    assert wav == tmp_path / f"d{len(decoded) - 1}.wav"
    assert first_source == voc32  # the file the cache stores
    assert first_kwargs == hit_kwargs


class _StopAfterAudio(Exception):
    """Raised by the first step after audio preparation, to end transcribe early."""


@pytest.mark.parametrize("normalize", [True, False])
def test_transcribe_rerun_decodes_the_cache_like_its_first_run(
    tmp_path, monkeypatch, normalize
):
    # transcribe's own cache-hit decode (not only align's) must feed ASR and
    # diarization what the first run fed them: the 16k decode of the very 32k
    # vocals the first run stored, with the same options.
    media, cache = _paths(tmp_path)
    cache.unlink()
    decoded: list[tuple[Path, dict]] = []
    encoded: list[Path] = []

    def fake_decode(source, **kwargs):
        out = tmp_path / f"d{len(decoded)}.wav"
        out.write_bytes(b"x")
        decoded.append((Path(source), kwargs))
        return out

    def fake_encode(src_wav, dst_flac):
        encoded.append(Path(src_wav))
        Path(dst_flac).write_bytes(b"flac")

    def stop(*_args, **_kwargs):
        raise _StopAfterAudio

    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
    monkeypatch.setattr(pipeline, "_encode_flac", fake_encode)
    monkeypatch.setattr(pipeline, "vad_speech_segments", stop)
    monkeypatch.setattr(
        pipeline.backend, "separate_vocals", lambda *_a, **_k: tmp_path / "v.wav"
    )

    with pytest.raises(_StopAfterAudio):
        pipeline.transcribe(media, normalize=normalize, cache_vocals=cache)
    first_source, first_kwargs = decoded[-1]
    assert encoded == [first_source]  # the cache stores what the 16k came from

    decoded.clear()
    monkeypatch.setattr(pipeline, "_vocals_cache_fresh", lambda *_a: True)
    with pytest.raises(_StopAfterAudio):
        pipeline.transcribe(media, normalize=normalize, cache_vocals=cache)
    assert decoded == [(cache, first_kwargs)]


_FFMPEG_TOOLS = shutil.which("ffmpeg") is not None and shutil.which("ffprobe")


@pytest.mark.skipif(not _FFMPEG_TOOLS, reason="needs ffmpeg and ffprobe")
@pytest.mark.parametrize("normalize", [True, False])
def test_real_cache_round_trip_gives_the_first_run_samples(tmp_path, normalize):
    # The parity above also rests on vocals.32k.flac being a lossless copy of
    # the first run's 32k wav: with real ffmpeg, the 16k input decoded from the
    # stored flac must equal the first run's sample for sample.
    rng = np.random.default_rng(7)
    seconds, rate = 6.0, 44100
    t = np.arange(int(seconds * rate)) / rate
    voice = 0.3 * np.sin(2 * np.pi * 220.0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * t))
    stereo = np.stack([voice, 0.8 * voice], axis=1) + 0.01 * rng.normal(
        size=(t.size, 2)
    )
    vocals = tmp_path / "vocals.flac"  # the separator writes 16-bit FLAC
    sf.write(vocals, stereo.astype(np.float32), rate, subtype="PCM_16")

    voc32 = pipeline.decode_to_wav(vocals, sample_rate=pipeline.SONGDET_SR)
    stored = tmp_path / "cache" / "vocals.32k.flac"
    first = hit = None
    try:
        pipeline._encode_flac(voc32, stored)
        first = pipeline._vocals_to_16k(voc32, normalize=normalize)
        hit = pipeline._vocals_to_16k(stored, normalize=normalize)
        first_samples, first_rate = sf.read(first, dtype="int16")
        hit_samples, hit_rate = sf.read(hit, dtype="int16")
    finally:
        for path in (voc32, first, hit):
            if path is not None:
                Path(path).unlink(missing_ok=True)
    assert first_rate == hit_rate == 16000
    assert first_samples.ndim == 1 and first_samples.size > 5 * 16000
    assert np.array_equal(first_samples, hit_samples)


def test_prepare_align_reseparates_and_overwrites_stale_cache(tmp_path):
    media, cache = _paths(tmp_path)
    parts = tuple(tmp_path / n for n in ("full.wav", "voc.flac", "16k.wav", "32k.wav"))
    tmp: list = []
    with (
        _durations({media.name: 90.0, cache.name: 100.0}),
        patch("voxweave.pipeline._separate_to_16k_32k", return_value=parts) as sep,
        patch("voxweave.pipeline._encode_flac") as enc,
    ):
        got = pipeline._prepare_16k_for_align(
            media, separate=True, normalize=False, reporter=pipeline.Reporter(), tmp=tmp
        )
    assert got == parts[2]
    sep.assert_called_once()
    enc.assert_called_once_with(parts[3], cache)  # stale cache overwritten in place


def test_prepare_align_skips_stale_legacy_cache(tmp_path):
    media, _ = _paths(tmp_path)
    pipeline.cache_vocals_path(media).unlink()  # only the legacy 16k cache remains
    legacy = pipeline.cache_16k_path(media)
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_bytes(b"l")
    parts = tuple(tmp_path / n for n in ("full.wav", "voc.flac", "16k.wav", "32k.wav"))
    with (
        _durations({media.name: 90.0, legacy.name: 100.0}),
        patch("voxweave.pipeline._separate_to_16k_32k", return_value=parts) as sep,
        patch("voxweave.pipeline._encode_flac"),
    ):
        got = pipeline._prepare_16k_for_align(
            media, separate=True, normalize=False, reporter=pipeline.Reporter(), tmp=[]
        )
    assert got == parts[2]
    sep.assert_called_once()


def test_fresh_vocals_miss_writes_into_the_adjacent_cache_claim(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"m")
    managed = pipeline.cache_vocals_path(media)
    assert managed == tmp_path / "cache" / "ep" / "vocals.32k.flac"
    parts = tuple(
        tmp_path / name for name in ("full.wav", "voc.wav", "16k.wav", "32k.wav")
    )
    for part in parts:
        part.write_bytes(b"audio")

    def encode(_source, destination):
        destination.write_bytes(b"flac")

    with (
        patch("voxweave.pipeline._separate_to_16k_32k", return_value=parts),
        patch("voxweave.pipeline._encode_flac", side_effect=encode),
    ):
        assert (
            pipeline._prepare_16k_for_align(
                media,
                separate=True,
                normalize=False,
                reporter=pipeline.Reporter(),
                tmp=[],
            )
            == parts[2]
        )

    assert managed.read_bytes() == b"flac"
    assert Path(f"{managed}.lock").is_file()
    assert (tmp_path / "cache" / "ep").is_dir()


def test_existing_adjacent_32k_cache_remains_the_read_writeback_lane(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"m")
    legacy = tmp_path / "cache/ep.vocals.32k.flac"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy flac bytes")
    companion = Path(f"{legacy}.meta.json")
    companion.write_bytes(b"legacy companion bytes")

    assert pipeline.cache_vocals_path(media) == legacy
    assert legacy.read_bytes() == b"legacy flac bytes"
    assert companion.read_bytes() == b"legacy companion bytes"


@pytest.mark.parametrize("lane", ["managed", "legacy"])
@pytest.mark.parametrize("node_kind", ["symlink", "fifo"])
def test_vocals_cache_lock_rejects_nonregular_nodes(tmp_path, lane, node_kind):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"media")
    if lane == "legacy":
        cache = tmp_path / "cache/ep.vocals.32k.flac"
        cache.parent.mkdir()
        cache.write_bytes(b"legacy")
    else:
        cache = pipeline.cache_vocals_path(media)
    lock = Path(f"{cache.resolve()}.lock")
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    original_mode = victim.stat().st_mode
    if node_kind == "symlink":
        lock.symlink_to(victim)
    else:
        os.mkfifo(lock)

    with pytest.raises(OSError):
        with vocalscache.cache_lock(cache):
            pass

    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert victim.stat().st_mode == original_mode
