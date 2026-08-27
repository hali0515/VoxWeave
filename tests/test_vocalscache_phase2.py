"""Strict cache companion, full-FLAC identity, lock, and torn-pair gates."""

import hashlib
import multiprocessing
from pathlib import Path

import pytest

from voxweave import voicebase, vocalscache


def _separator():
    return {
        "repo": "audio/separator",
        "file": "model.ckpt",
        "checkpoint": "blob-123",
        "config_sha256": "c" * 64,
    }


def _cache(tmp_path, payload=b"fLaC fake vocals"):
    path = tmp_path / "episode.vocals.32k.flac"
    path.write_bytes(payload)
    return path


def _lock_worker(path, connection):
    connection.send("started")
    with vocalscache.cache_write_window(path) as handle:
        connection.send(
            (
                "acquired",
                str(handle.cache_path),
                str(handle.companion_path),
                str(handle.lock_path),
            )
        )
        connection.recv()
    connection.close()


def test_build_write_load_and_validate_fake_flac_pair(tmp_path):
    cache = _cache(tmp_path)
    media = "a" * 64
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint=media,
        separator=_separator(),
    )
    assert companion == {
        "version": 1,
        "media_fingerprint": media,
        "separator": _separator(),
        "cache_size": cache.stat().st_size,
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    path = vocalscache.cache_companion_path(cache)
    vocalscache.write_cache_companion(path, companion)
    raw, validated = vocalscache.load_cache_companion(path)
    assert raw == companion
    assert validated.cache_size == cache.stat().st_size
    assert (
        vocalscache.validate_cache_pair(
            raw,
            cache,
            media_fingerprint=media,
            separator=_separator(),
        )
        == validated
    )


def test_empty_cache_has_exact_zero_size_and_full_hash(tmp_path):
    cache = _cache(tmp_path, b"")
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    assert companion["cache_size"] == 0
    assert companion["cache_sha256"] == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_companion_version_is_exact_integer_one(tmp_path, version):
    companion = vocalscache.build_cache_companion(
        _cache(tmp_path),
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    companion["version"] = version
    with pytest.raises(vocalscache.CacheCompanionError, match="integer 1"):
        vocalscache.validate_cache_companion(companion)


@pytest.mark.parametrize("cache_size", [True, 1.0, "1", -1])
def test_cache_size_is_exact_non_bool_nonnegative_int(tmp_path, cache_size):
    companion = vocalscache.build_cache_companion(
        _cache(tmp_path),
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    companion["cache_size"] = cache_size
    with pytest.raises(vocalscache.CacheCompanionError, match="cache_size"):
        vocalscache.validate_cache_companion(companion)


def test_duplicate_keys_are_rejected_by_bounded_loader(tmp_path):
    path = tmp_path / "cache.meta.json"
    path.write_text(
        '{"version":1,"version":1,"media_fingerprint":"' + "a" * 64 + '"}',
        encoding="utf-8",
    )
    with pytest.raises(voicebase.DuplicateKeyError, match="version"):
        vocalscache.load_cache_companion(path)


def test_companion_cap_is_checked_before_parse(tmp_path, monkeypatch):
    path = tmp_path / "cache.meta.json"
    path.write_bytes(b"{" + b"x" * voicebase.CACHE_COMPANION_MAX_BYTES)

    def must_not_parse(*_args, **_kwargs):
        raise AssertionError("oversized companion reached JSON parser")

    monkeypatch.setattr(voicebase.json, "loads", must_not_parse)
    with pytest.raises(voicebase.Phase2DataError, match="exceeds"):
        vocalscache.load_cache_companion(path)


def test_all_sha_fields_require_lowercase_64_hex(tmp_path):
    companion = vocalscache.build_cache_companion(
        _cache(tmp_path),
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    for field in ("media_fingerprint", "cache_sha256"):
        bad = dict(companion)
        bad[field] = "A" * 64
        with pytest.raises(vocalscache.CacheCompanionError):
            vocalscache.validate_cache_companion(bad)
    bad = dict(companion)
    bad["separator"] = {**_separator(), "config_sha256": "A" * 64}
    with pytest.raises(vocalscache.CacheCompanionError):
        vocalscache.validate_cache_companion(bad)


def test_separator_strings_reach_real_512_byte_maximum(tmp_path):
    separator = {
        "repo": "r" * 512,
        "file": "f" * 512,
        "checkpoint": "p" * 512,
        "config_sha256": "c" * 64,
    }
    cache = _cache(tmp_path)
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint="a" * 64,
        separator=separator,
    )
    path = vocalscache.cache_companion_path(cache)
    vocalscache.write_cache_companion(path, companion)
    assert path.stat().st_size <= voicebase.CACHE_COMPANION_MAX_BYTES
    assert vocalscache.load_cache_companion(path)[0] == companion


def test_separator_config_alias_canonicalizes_and_disagreement_refuses():
    alias = {**_separator()}
    alias["config"] = alias.pop("config_sha256")
    validated = vocalscache.validate_separator_identity(alias)
    assert validated.as_mapping() == _separator()
    alias["config_sha256"] = "d" * 64
    with pytest.raises(vocalscache.CacheCompanionError, match="disagree"):
        vocalscache.validate_separator_identity(alias)


def test_unknown_top_level_companion_fields_are_forward_compatible(tmp_path):
    companion = vocalscache.build_cache_companion(
        _cache(tmp_path),
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    companion["future"] = {"safe": True}
    vocalscache.validate_cache_companion(companion)


def test_same_size_torn_pair_is_detected_by_full_flac_hash(tmp_path):
    cache = _cache(tmp_path, b"A" * 4096)
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    cache.write_bytes(b"B" * 4096)
    assert not vocalscache.cache_pair_valid(
        companion,
        cache,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    with pytest.raises(vocalscache.CacheCompanionMismatch, match="SHA-256"):
        vocalscache.validate_cache_pair(
            companion,
            cache,
            media_fingerprint="a" * 64,
            separator=_separator(),
        )


def test_full_hash_catches_middle_only_mutation(tmp_path):
    payload = bytearray(b"A" * (3 * 1024 * 1024))
    cache = _cache(tmp_path, payload)
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    payload[1024 * 1024 + 100] = ord("B")
    cache.write_bytes(payload)
    assert not vocalscache.cache_pair_valid(
        companion,
        cache,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )


def test_cache_size_media_and_separator_mismatches_each_refuse(tmp_path):
    cache = _cache(tmp_path)
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    cache.write_bytes(cache.read_bytes() + b"x")
    with pytest.raises(vocalscache.CacheCompanionMismatch, match="size"):
        vocalscache.validate_cache_pair(
            companion,
            cache,
            media_fingerprint="a" * 64,
            separator=_separator(),
        )
    cache.write_bytes(cache.read_bytes()[:-1])
    with pytest.raises(vocalscache.CacheCompanionMismatch, match="media"):
        vocalscache.validate_cache_pair(
            companion,
            cache,
            media_fingerprint="b" * 64,
            separator=_separator(),
        )
    changed_separator = {**_separator(), "checkpoint": "other-blob"}
    with pytest.raises(vocalscache.CacheCompanionMismatch, match="separator"):
        vocalscache.validate_cache_pair(
            companion,
            cache,
            media_fingerprint="a" * 64,
            separator=changed_separator,
        )


def test_missing_or_nonregular_cache_is_not_a_valid_pair(tmp_path):
    missing = tmp_path / "missing.flac"
    companion = {
        "version": 1,
        "media_fingerprint": "a" * 64,
        "separator": _separator(),
        "cache_size": 0,
        "cache_sha256": hashlib.sha256(b"").hexdigest(),
    }
    assert not vocalscache.cache_pair_valid(
        companion,
        missing,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    directory = tmp_path / "directory.flac"
    directory.mkdir()
    assert not vocalscache.cache_pair_valid(
        companion,
        directory,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )


def test_realpath_aliases_share_cache_companion_and_lock_paths(tmp_path):
    real = _cache(tmp_path)
    alias = tmp_path / "alias.flac"
    alias.symlink_to(real)
    assert vocalscache.canonical_cache_path(alias) == real
    assert vocalscache.cache_lock_path(alias) == Path(f"{real}.lock")
    assert vocalscache.cache_companion_path(alias) == Path(f"{real}.meta.json")


def test_alias_writer_blocks_while_reader_holds_lock_through_decode(tmp_path):
    real = _cache(tmp_path)
    alias = tmp_path / "alias.flac"
    alias.symlink_to(real)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_lock_worker, args=(alias, child))
    try:
        with vocalscache.cache_lock(real):
            process.start()
            assert parent.recv() == "started"
            assert not parent.poll(0.2)
            # This read stands in for decode_to_wav: the lock remains held after
            # companion validation and through the consumer's final open/read.
            assert real.read_bytes().startswith(b"fLaC")
        assert parent.poll(2)
        assert parent.recv() == (
            "acquired",
            str(real),
            f"{real}.meta.json",
            f"{real}.lock",
        )
        parent.send("release")
        process.join(2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_delete_first_capture_off_overwrite_leaves_companion_absent(tmp_path):
    cache = _cache(tmp_path, b"A")
    companion_path = vocalscache.cache_companion_path(cache)
    companion_path.write_text("old claim", encoding="utf-8")
    with vocalscache.cache_write_window(cache) as handle:
        assert not handle.companion_path.exists()
        handle.cache_path.write_bytes(b"B")
    assert cache.read_bytes() == b"B"
    assert not companion_path.exists()


def test_delete_first_writer_failure_leaves_detectable_miss(tmp_path):
    cache = _cache(tmp_path, b"A")
    companion_path = vocalscache.cache_companion_path(cache)
    companion_path.write_text("old claim", encoding="utf-8")
    with pytest.raises(RuntimeError, match="encoder failed"):
        with vocalscache.cache_write_window(cache) as handle:
            assert not handle.companion_path.exists()
            handle.cache_path.write_bytes(b"partial")
            raise RuntimeError("encoder failed")
    assert cache.read_bytes() == b"partial"
    assert not companion_path.exists()


def test_capture_writer_publishes_only_after_finished_cache(tmp_path):
    cache = _cache(tmp_path, b"A")
    media = "a" * 64
    with vocalscache.cache_write_window(cache) as handle:
        assert not handle.companion_path.exists()
        handle.cache_path.write_bytes(b"finished fake FLAC")
        companion = vocalscache.publish_cache_companion(
            handle.cache_path,
            media_fingerprint=media,
            separator=_separator(),
        )
        assert handle.companion_path.exists()
    assert vocalscache.cache_pair_valid(
        companion,
        cache,
        media_fingerprint=media,
        separator=_separator(),
    )


def test_explicit_delete_first_helper_is_idempotent(tmp_path):
    cache = _cache(tmp_path)
    companion = vocalscache.cache_companion_path(cache)
    companion.write_text("old", encoding="utf-8")
    assert vocalscache.delete_cache_companion_first(cache) == companion
    assert not companion.exists()
    vocalscache.delete_cache_companion_first(cache)


def test_companion_writer_preflight_preserves_existing_target(tmp_path):
    cache = _cache(tmp_path)
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint="a" * 64,
        separator=_separator(),
    )
    companion["future"] = "x" * voicebase.CACHE_COMPANION_MAX_BYTES
    path = vocalscache.cache_companion_path(cache)
    path.write_text("old", encoding="utf-8")
    with pytest.raises(voicebase.Phase2DataError, match="encoded JSON exceeds"):
        vocalscache.write_cache_companion(path, companion)
    assert path.read_text(encoding="utf-8") == "old"
