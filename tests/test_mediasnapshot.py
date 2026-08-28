"""Immutable snapshot, secure janitor, fallback, and fault-injection gates."""

import errno
import os
import stat
import time
from pathlib import Path

import pytest

from voxweave import mediasnapshot, voicebase


def _unsupported_clone(_source_fd, _destination_fd):
    raise OSError(errno.EOPNOTSUPP, "clone unsupported")


def _force_copy(monkeypatch):
    monkeypatch.setattr(mediasnapshot, "_clone_reflink", _unsupported_clone)


def _write_all(fd, payload):
    offset = 0
    while offset < len(payload):
        offset += os.pwrite(fd, payload[offset:], offset)


def test_snapshot_is_private_verified_suffix_preserving_and_exception_safe(tmp_path):
    cache = tmp_path / "cache"
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"A" * 4096)
    expected = voicebase.media_fingerprint(source)

    with mediasnapshot.MediaSnapshot(source, cache_root=cache) as snapshot:
        private_path = snapshot.path
        assert private_path != source
        assert private_path.parent == cache / "snapshots"
        assert private_path.suffix == ".mkv"
        assert private_path.read_bytes() == source.read_bytes()
        assert snapshot.fingerprint == expected
        assert snapshot.size == 4096
        assert snapshot.copy_method in {"reflink", "copy"}
        assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(private_path.parent.stat().st_mode) == 0o700

    assert not private_path.exists()
    assert list((cache / "snapshots").iterdir()) == []
    with pytest.raises(mediasnapshot.SnapshotUnavailable, match="not active"):
        _ = snapshot.path


def test_snapshot_is_single_use(tmp_path):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A")
    holder = mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache")
    with holder:
        pass
    with pytest.raises(mediasnapshot.SnapshotUnavailable, match="single-use"):
        with holder:
            pass


def test_same_inode_otrunc_a_b_a_cannot_change_snapshot_bytes(tmp_path, monkeypatch):
    _force_copy(monkeypatch)
    source = tmp_path / "episode.mp4"
    original = b"A" * 8192
    source.write_bytes(original)
    with mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache") as snapshot:
        stable_path = snapshot.path
        with open(source, "r+b") as live:
            live.truncate(0)
            live.write(b"B" * len(original))
            live.flush()
            os.fsync(live.fileno())
        assert stable_path.read_bytes() == original
        with open(source, "r+b") as live:
            live.truncate(0)
            live.write(original)
            live.flush()
            os.fsync(live.fileno())
        assert stable_path.read_bytes() == original
        assert snapshot.fingerprint == voicebase.media_fingerprint(stable_path)


def test_pathname_a_b_a_replacement_cannot_change_snapshot_bytes(tmp_path, monkeypatch):
    _force_copy(monkeypatch)
    source = tmp_path / "episode.mp4"
    original_path = tmp_path / "episode-a.mp4"
    original = b"A" * 4096
    source.write_bytes(original)
    with mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache") as snapshot:
        source.rename(original_path)
        source.write_bytes(b"B" * len(original))
        assert snapshot.path.read_bytes() == original
        source.unlink()
        original_path.rename(source)
        assert snapshot.path.read_bytes() == original


def test_reflink_unsupported_uses_verified_fd_copy(tmp_path, monkeypatch):
    _force_copy(monkeypatch)
    original_copy = mediasnapshot._copy_sequential
    calls = []

    def recording_copy(source_fd, destination_fd, size):
        assert isinstance(source_fd, int)
        assert isinstance(destination_fd, int)
        calls.append(size)
        original_copy(source_fd, destination_fd, size)

    monkeypatch.setattr(mediasnapshot, "_copy_sequential", recording_copy)
    source = tmp_path / "episode.flac"
    source.write_bytes(b"voice" * 1000)
    with mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache") as snapshot:
        assert snapshot.copy_method == "copy"
        assert snapshot.path.read_bytes() == source.read_bytes()
    assert calls == [5000]


def test_verified_reflink_path_does_not_run_copy_fallback(tmp_path, monkeypatch):
    def fake_clone(source_fd, destination_fd):
        size = os.fstat(source_fd).st_size
        os.ftruncate(destination_fd, 0)
        _write_all(destination_fd, os.pread(source_fd, size, 0))
        os.ftruncate(destination_fd, size)

    def forbidden_copy(*_args):
        raise AssertionError("verified clone unexpectedly fell back")

    monkeypatch.setattr(mediasnapshot, "_clone_reflink", fake_clone)
    monkeypatch.setattr(mediasnapshot, "_copy_sequential", forbidden_copy)
    source = tmp_path / "episode.wav"
    source.write_bytes(b"pcm" * 100)
    with mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache") as snapshot:
        assert snapshot.copy_method == "reflink"
        assert snapshot.path.read_bytes() == source.read_bytes()


def test_enospc_partial_copy_is_cleaned_and_typed(tmp_path, monkeypatch):
    _force_copy(monkeypatch)

    def partial_then_full(_source_fd, destination_fd, _size):
        os.pwrite(destination_fd, b"partial", 0)
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(mediasnapshot, "_copy_sequential", partial_then_full)
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A" * 1024)
    cache = tmp_path / "cache"
    with pytest.raises(mediasnapshot.SnapshotUnavailable, match="copy failed") as error:
        with mediasnapshot.MediaSnapshot(source, cache_root=cache):
            pass
    assert error.value.failure.detail_code == "reflink-and-copy-failed"
    assert list((cache / "snapshots").iterdir()) == []


def test_interrupt_during_partial_copy_cleans_residue(tmp_path, monkeypatch):
    _force_copy(monkeypatch)

    def interrupted(_source_fd, destination_fd, _size):
        os.pwrite(destination_fd, b"partial", 0)
        raise KeyboardInterrupt

    monkeypatch.setattr(mediasnapshot, "_copy_sequential", interrupted)
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A" * 1024)
    cache = tmp_path / "cache"
    with pytest.raises(KeyboardInterrupt):
        with mediasnapshot.MediaSnapshot(source, cache_root=cache):
            pass
    assert list((cache / "snapshots").iterdir()) == []


def test_source_permission_failure_is_typed_and_creates_no_partial(
    tmp_path, monkeypatch
):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A")
    cache = tmp_path / "cache"

    def denied(_path):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(mediasnapshot, "_open_source", denied)
    with pytest.raises(mediasnapshot.SnapshotUnavailable, match="cannot create"):
        with mediasnapshot.MediaSnapshot(source, cache_root=cache):
            pass
    assert list((cache / "snapshots").iterdir()) == []


def test_destination_mode_failure_cleans_o_excl_file(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A")
    cache = tmp_path / "cache"

    def denied(_descriptor, _mode):
        raise PermissionError(errno.EPERM, "fchmod denied")

    monkeypatch.setattr(mediasnapshot.os, "fchmod", denied)
    with pytest.raises(mediasnapshot.SnapshotUnavailable, match="cannot create"):
        with mediasnapshot.MediaSnapshot(source, cache_root=cache):
            pass
    assert list((cache / "snapshots").iterdir()) == []


def test_non_regular_source_fails_closed(tmp_path):
    source = tmp_path / "media-dir"
    source.mkdir()
    cache = tmp_path / "cache"
    with pytest.raises(mediasnapshot.SnapshotUnavailable, match="not a regular"):
        with mediasnapshot.MediaSnapshot(source, cache_root=cache):
            pass
    assert list((cache / "snapshots").iterdir()) == []


def test_symlinked_snapshots_directory_is_refused_not_followed(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    (cache / "snapshots").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A")
    with pytest.raises(mediasnapshot.SnapshotUnavailable, match="private directory"):
        with mediasnapshot.MediaSnapshot(source, cache_root=cache):
            pass
    assert list(outside.iterdir()) == []


def test_context_body_exception_always_deletes_snapshot(tmp_path):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A")
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="consumer failed"):
        with mediasnapshot.MediaSnapshot(source, cache_root=cache) as snapshot:
            path = snapshot.path
            raise RuntimeError("consumer failed")
    assert not path.exists()


def test_torn_copy_with_different_sampled_identity_is_rejected(tmp_path, monkeypatch):
    _force_copy(monkeypatch)
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A" * 4096)
    cache = tmp_path / "cache"

    def torn_copy(_source_fd, destination_fd, size):
        os.ftruncate(destination_fd, 0)
        _write_all(destination_fd, b"B" * size)
        os.ftruncate(destination_fd, size)

    monkeypatch.setattr(mediasnapshot, "_copy_sequential", torn_copy)
    with pytest.raises(
        mediasnapshot.SnapshotUnavailable, match="sampled identity"
    ) as error:
        with mediasnapshot.MediaSnapshot(source, cache_root=cache):
            pass
    assert error.value.failure.detail_code == "snapshot-verification-failed"
    assert list((cache / "snapshots").iterdir()) == []


def test_free_space_preflight_is_advisory_actual_copy_is_authoritative(
    tmp_path, monkeypatch
):
    _force_copy(monkeypatch)
    monkeypatch.setattr(mediasnapshot, "_available_bytes", lambda _directory: 0)
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A" * 4096)
    with mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache") as snapshot:
        assert snapshot.copy_method == "copy"
        assert snapshot.free_space_sufficient is False
        assert snapshot.path.read_bytes() == source.read_bytes()


def test_unavailable_free_space_probe_does_not_replace_copy_result(
    tmp_path, monkeypatch
):
    _force_copy(monkeypatch)

    def unavailable(_directory):
        raise OSError(errno.EIO, "probe failed")

    monkeypatch.setattr(mediasnapshot, "_available_bytes", unavailable)
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A" * 100)
    with mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache") as snapshot:
        assert snapshot.free_space_sufficient is None
        assert snapshot.path.read_bytes() == source.read_bytes()


def test_active_snapshot_older_than_hour_is_lease_protected(tmp_path):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A" * 100)
    cache = tmp_path / "cache"
    now = time.time()
    with mediasnapshot.MediaSnapshot(source, cache_root=cache) as snapshot:
        os.utime(snapshot.path, (now - 7200, now - 7200))
        removed = mediasnapshot.cleanup_stale_snapshots(
            snapshot.path.parent,
            now=now,
        )
        assert removed == ()
        assert snapshot.path.exists()


def test_janitor_removes_only_old_regular_unleased_owned_names(tmp_path):
    directory = tmp_path / "snapshots"
    directory.mkdir()
    now = time.time()
    old = directory / ("snapshot-" + "1" * 32 + ".mp4")
    old.write_bytes(b"residue")
    young = directory / ("snapshot-" + "2" * 32 + ".mp4")
    young.write_bytes(b"young")
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    symlink = directory / ("snapshot-" + "3" * 32 + ".mp4")
    symlink.symlink_to(outside)
    nonregular = directory / ("snapshot-" + "4" * 32 + ".mp4")
    nonregular.mkdir()
    unrelated = directory / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    os.utime(old, (now - 7200, now - 7200))
    os.utime(young, (now - 30, now - 30))

    removed = mediasnapshot.cleanup_stale_snapshots(directory, now=now)
    assert removed == (old,)
    assert not old.exists()
    assert young.exists()
    assert symlink.is_symlink()
    assert nonregular.is_dir()
    assert unrelated.exists()
    assert outside.read_bytes() == b"keep"


def test_janitor_cleanup_failures_are_best_effort(tmp_path, monkeypatch):
    directory = tmp_path / "snapshots"
    directory.mkdir()
    now = time.time()
    residue = directory / ("snapshot-" + "1" * 32 + ".mp4")
    residue.write_bytes(b"old")
    os.utime(residue, (now - 7200, now - 7200))
    original_unlink = Path.unlink

    def refuse_residue(path, *args, **kwargs):
        if path == residue:
            raise PermissionError(errno.EPERM, "cannot unlink")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_residue)
    assert mediasnapshot.cleanup_stale_snapshots(directory, now=now) == ()
    assert residue.exists()


def test_new_snapshot_runs_crash_residue_janitor(tmp_path):
    cache = tmp_path / "cache"
    directory = cache / "snapshots"
    directory.mkdir(parents=True)
    residue = directory / ("snapshot-" + "1" * 32 + ".mkv")
    residue.write_bytes(b"old")
    old = time.time() - 7200
    os.utime(residue, (old, old))
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"new")
    with mediasnapshot.MediaSnapshot(source, cache_root=cache):
        assert not residue.exists()


def test_o_excl_name_collision_retries_unpredictable_name(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    directory = cache / "snapshots"
    directory.mkdir(parents=True)
    occupied = directory / ("snapshot-" + "1" * 32 + ".mp4")
    occupied.write_bytes(b"do not replace")
    tokens = iter(["1" * 32, "2" * 32])
    monkeypatch.setattr(mediasnapshot.secrets, "token_hex", lambda _count: next(tokens))
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"A")
    with mediasnapshot.MediaSnapshot(source, cache_root=cache) as snapshot:
        assert snapshot.path.name == "snapshot-" + "2" * 32 + ".mp4"
        assert occupied.read_bytes() == b"do not replace"


def test_unsafe_or_missing_source_suffix_uses_demuxable_fallback(tmp_path):
    source = tmp_path / "episode"
    source.write_bytes(b"A")
    with mediasnapshot.MediaSnapshot(source, cache_root=tmp_path / "cache") as snapshot:
        assert snapshot.path.suffix == ".media"


def test_source_symlink_is_snapshotted_by_fd_without_path_fallback(
    tmp_path, monkeypatch
):
    _force_copy(monkeypatch)
    real = tmp_path / "real.mp4"
    real.write_bytes(b"A" * 512)
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(real)
    original_copy = mediasnapshot._copy_sequential
    seen = []

    def fd_only(source_fd, destination_fd, size):
        seen.append((source_fd, destination_fd))
        assert isinstance(source_fd, int) and isinstance(destination_fd, int)
        original_copy(source_fd, destination_fd, size)

    monkeypatch.setattr(mediasnapshot, "_copy_sequential", fd_only)
    with mediasnapshot.MediaSnapshot(alias, cache_root=tmp_path / "cache") as snapshot:
        assert snapshot.path != alias
        assert snapshot.path.read_bytes() == real.read_bytes()
    assert len(seen) == 1
