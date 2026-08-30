"""Vocals cache duration freshness: a replaced/trimmed source must invalidate the cache."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

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
    legacy.parent.mkdir()
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


def test_fresh_vocals_miss_never_creates_adjacent_cache_directory(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"m")
    managed = pipeline.cache_vocals_path(media)
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
    assert not (tmp_path / "cache").exists()


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
