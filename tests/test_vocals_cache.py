"""Vocals cache duration freshness: a replaced/trimmed source must invalidate the cache."""

from unittest.mock import patch

from voxweave import pipeline


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
    cache.parent.mkdir(parents=True)
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
