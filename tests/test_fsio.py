# tests/test_fsio.py
# Atomic file writes: replaceable artifacts use os.replace, while protected new
# user sidecars publish completed content without overwriting concurrent files.

import errno
import fcntl
import os
import stat
import time

import pytest

from voxweave import fsio


@pytest.fixture
def umask():
    """Set the process umask for one test and restore it afterwards."""
    previous = os.umask(0o022)
    os.umask(previous)

    def set_mask(mask):
        os.umask(mask)

    yield set_mask
    os.umask(previous)


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_atomic_write_text_writes_content(tmp_path):
    dst = tmp_path / "out.vtt"
    fsio.atomic_write_text(dst, "WEBVTT\n\nhello\n")
    assert dst.read_text(encoding="utf-8") == "WEBVTT\n\nhello\n"


def test_atomic_write_text_overwrites_existing(tmp_path):
    dst = tmp_path / "out.vtt"
    dst.write_text("old", encoding="utf-8")
    fsio.atomic_write_text(dst, "new")
    assert dst.read_text(encoding="utf-8") == "new"


def test_atomic_write_text_leaves_no_temp_residue(tmp_path):
    dst = tmp_path / "out.json"
    fsio.atomic_write_text(dst, "{}")
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_atomic_write_text_can_select_fallback_at_replace_edge(tmp_path):
    dst = tmp_path / "out.json"
    dst.write_text("old", encoding="utf-8")
    checked = []

    def select_fallback():
        assert dst.read_text(encoding="utf-8") == "old"
        checked.append(True)
        return "unbound"

    fsio.atomic_write_text(
        dst,
        "bound",
        before_replace=select_fallback,
    )

    assert checked == [True]
    assert dst.read_text(encoding="utf-8") == "unbound"


def test_atomic_write_text_new_creates_without_temp_residue(tmp_path):
    dst = tmp_path / "mapping.json"
    installed = fsio.atomic_write_text_new(dst, '{"version": 1}')
    assert dst.read_text(encoding="utf-8") == '{"version": 1}'
    assert list(tmp_path.iterdir()) == [dst]
    metadata = dst.stat()
    assert installed == (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def test_atomic_write_text_new_checks_authority_at_install_edge(tmp_path):
    dst = tmp_path / "mapping.json"
    checked = []

    def check_authority():
        assert not dst.exists()
        checked.append(True)

    fsio.atomic_write_text_new(
        dst,
        '{"version": 1}',
        before_install=check_authority,
    )

    assert checked == [True]
    assert dst.read_text(encoding="utf-8") == '{"version": 1}'
    assert list(tmp_path.iterdir()) == [dst]


def test_atomic_write_text_new_failed_install_check_publishes_nothing(tmp_path):
    dst = tmp_path / "mapping.json"

    def reject():
        raise RuntimeError("authority changed")

    with pytest.raises(RuntimeError, match="authority changed"):
        fsio.atomic_write_text_new(
            dst,
            '{"version": 1}',
            before_install=reject,
        )

    assert list(tmp_path.iterdir()) == []


def test_atomic_write_text_new_refuses_existing_file(tmp_path):
    dst = tmp_path / "mapping.json"
    dst.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fsio.atomic_write_text_new(dst, "replacement")
    assert dst.read_text(encoding="utf-8") == "user data"
    assert list(tmp_path.iterdir()) == [dst]


def test_atomic_write_text_new_prefers_content_atomic_hard_link(tmp_path, monkeypatch):
    dst = tmp_path / "mapping.json"

    def unexpected_replace(*_args, **_kwargs):
        raise AssertionError("hard-link capable filesystems must not use the fallback")

    monkeypatch.setattr(fsio.os, "replace", unexpected_replace)
    fsio.atomic_write_text_new(dst, '{"version": 1}')

    assert dst.read_text(encoding="utf-8") == '{"version": 1}'
    assert list(tmp_path.iterdir()) == [dst]


@pytest.mark.parametrize(
    "link_errno",
    [errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS, errno.EXDEV],
)
def test_atomic_write_text_new_claims_then_replaces_when_links_unavailable(
    tmp_path, monkeypatch, link_errno
):
    dst = tmp_path / "mapping.json"
    payload = '{"version": 1}'
    real_replace = fsio.os.replace
    replacements = []

    def unavailable(*_args, **_kwargs):
        raise OSError(link_errno, "hard links unsupported")

    def replace_owned_claim(src, target):
        assert target == dst
        assert dst.exists() and dst.stat().st_size == 0
        assert src.read_text(encoding="utf-8") == payload
        replacements.append((src, target))
        real_replace(src, target)

    monkeypatch.setattr(fsio.os, "link", unavailable)
    monkeypatch.setattr(fsio.os, "replace", replace_owned_claim)
    fsio.atomic_write_text_new(dst, payload)

    assert dst.read_text(encoding="utf-8") == payload
    assert len(replacements) == 1
    assert list(tmp_path.iterdir()) == [dst]


def test_atomic_write_text_new_fallback_still_refuses_existing_file(
    tmp_path, monkeypatch
):
    dst = tmp_path / "mapping.json"
    dst.write_text("user data", encoding="utf-8")

    def unavailable(*_args, **_kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    monkeypatch.setattr(fsio.os, "link", unavailable)
    with pytest.raises(FileExistsError):
        fsio.atomic_write_text_new(dst, "replacement")

    assert dst.read_text(encoding="utf-8") == "user data"
    assert list(tmp_path.iterdir()) == [dst]


def test_atomic_write_text_new_fallback_rechecks_before_claim(tmp_path, monkeypatch):
    dst = tmp_path / "mapping.json"
    checks = []

    def unavailable(*_args, **_kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    def reject_second_install_attempt():
        assert not dst.exists()
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("authority changed before claim")

    monkeypatch.setattr(fsio.os, "link", unavailable)
    with pytest.raises(RuntimeError, match="authority changed before claim"):
        fsio.atomic_write_text_new(
            dst,
            '{"version": 1}',
            before_install=reject_second_install_attempt,
        )

    assert checks == [True, True]
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_text_new_does_not_publish_incomplete_content(
    tmp_path, monkeypatch
):
    dst = tmp_path / "mapping.json"

    def fail_fsync(_fd):
        assert not dst.exists()
        raise OSError("simulated disk failure")

    monkeypatch.setattr(fsio.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated disk failure"):
        fsio.atomic_write_text_new(dst, '{"version": 1}')

    assert list(tmp_path.iterdir()) == []


def test_atomic_path_failure_preserves_existing_dst(tmp_path):
    dst = tmp_path / "out.mkv"
    dst.write_bytes(b"good output from a previous run")
    with pytest.raises(RuntimeError):
        with fsio.atomic_path(dst) as tmp:
            tmp.write_bytes(b"half-writ")
            raise RuntimeError("ffmpeg died")
    assert dst.read_bytes() == b"good output from a previous run"
    assert list(tmp_path.iterdir()) == [dst]  # temp cleaned up


def test_atomic_path_failure_leaves_nothing_when_dst_missing(tmp_path):
    dst = tmp_path / "out.mp4"
    with pytest.raises(ValueError):
        with fsio.atomic_path(dst):
            raise ValueError("boom")
    assert list(tmp_path.iterdir()) == []


def test_atomic_path_success_moves_temp_to_dst(tmp_path):
    dst = tmp_path / "out.flac"
    with fsio.atomic_path(dst) as tmp:
        assert tmp.parent == dst.parent  # same fs so os.replace is atomic
        assert tmp != dst
        tmp.write_bytes(b"data")
    assert dst.read_bytes() == b"data"
    assert list(tmp_path.iterdir()) == [dst]


def test_atomic_path_temp_keeps_dst_suffix(tmp_path):
    # ffmpeg picks its muxer from the output extension, so the temp file the
    # command actually writes must end with the real suffix.
    with fsio.atomic_path(tmp_path / "out.mp4") as tmp:
        assert tmp.suffix == ".mp4"
        tmp.write_bytes(b"x")


def test_atomic_path_cleans_temp_on_keyboard_interrupt(tmp_path):
    dst = tmp_path / "out.vtt"
    dst.write_text("keep me", encoding="utf-8")
    with pytest.raises(KeyboardInterrupt):
        with fsio.atomic_path(dst) as tmp:
            tmp.write_text("partial", encoding="utf-8")
            raise KeyboardInterrupt
    assert dst.read_text(encoding="utf-8") == "keep me"
    assert list(tmp_path.iterdir()) == [dst]


# -- permissions: deliverables honour the umask, private data stays 0600 ----


@pytest.mark.parametrize(("mask", "expected"), [(0o022, 0o644), (0o027, 0o640)])
def test_new_deliverable_honours_the_umask(tmp_path, umask, mask, expected):
    umask(mask)
    text = tmp_path / "ep.vtt"
    media = tmp_path / "ep.burn.mp4"
    fsio.atomic_write_text(text, "WEBVTT\n")
    with fsio.atomic_path(media) as tmp:
        tmp.write_bytes(b"encoded")
    assert _mode(text) == expected
    assert _mode(media) == expected


def test_overwritten_deliverable_keeps_its_mode(tmp_path, umask):
    umask(0o022)
    dst = tmp_path / "ep.srt"
    dst.write_text("old", encoding="utf-8")
    os.chmod(dst, 0o664)
    fsio.atomic_write_text(dst, "new")
    assert dst.read_text(encoding="utf-8") == "new"
    assert _mode(dst) == 0o664


def test_private_writes_are_0600_new_or_replaced(tmp_path, umask):
    umask(0o022)
    new = tmp_path / "progress.json"
    replaced = tmp_path / "identities.json"
    replaced.write_text("old", encoding="utf-8")
    os.chmod(replaced, 0o644)
    fsio.atomic_write_text(new, "{}", private=True)
    with fsio.atomic_path(replaced, private=True) as tmp:
        tmp.write_text("new", encoding="utf-8")
    assert _mode(new) == 0o600
    assert _mode(replaced) == 0o600


def test_atomic_write_text_new_mode_follows_privacy(tmp_path, umask):
    umask(0o022)
    public = tmp_path / "public.json"
    private = tmp_path / "private.json"
    fsio.atomic_write_text_new(public, "{}")
    fsio.atomic_write_text_new(private, "{}", private=True)
    assert _mode(public) == 0o644
    assert _mode(private) == 0o600


def test_refused_chmod_still_lands_the_file(tmp_path, monkeypatch):
    # Some NAS exports refuse chmod; the deliverable must still be written.
    def refuse(*_args, **_kwargs):
        raise PermissionError(errno.EPERM, "chmod refused")

    monkeypatch.setattr(fsio.os, "chmod", refuse)
    dst = tmp_path / "ep.vtt"
    fsio.atomic_write_text(dst, "WEBVTT\n")
    assert dst.read_text(encoding="utf-8") == "WEBVTT\n"
    assert list(tmp_path.iterdir()) == [dst]


# -- temp names and crash residue ------------------------------------------


@pytest.mark.parametrize("stem", ["a" * 250, "字" * 83])
def test_longest_valid_names_still_get_a_temp_file(tmp_path, stem):
    dst = tmp_path / f"{stem}.vtt"
    assert len(os.fsencode(dst.name)) <= 255
    fsio.atomic_write_text(dst, "WEBVTT\n")
    with fsio.atomic_path(dst) as tmp:
        assert len(os.fsencode(tmp.name)) <= 255
        assert tmp.name.startswith(".") and tmp.name.endswith(".part.vtt")
        tmp.write_text("WEBVTT\n\nnew\n", encoding="utf-8")
    assert dst.read_text(encoding="utf-8") == "WEBVTT\n\nnew\n"
    assert list(tmp_path.iterdir()) == [dst]


def _age(path, seconds):
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_stale_temp_files_of_the_destination_are_swept(tmp_path):
    dst = tmp_path / "ep.json"
    stale = tmp_path / ".ep.abcd_123.part.json"
    live = tmp_path / ".ep.efgh4567.part.json"
    other = tmp_path / ".ep.zh.abcd_123.part.json"
    for path in (stale, live, other):
        path.write_bytes(b"partial")
    _age(stale, fsio.STALE_TEMP_SECONDS + 60)
    _age(live, fsio.STALE_TEMP_SECONDS - 60)
    _age(other, fsio.STALE_TEMP_SECONDS + 60)

    fsio.atomic_write_text(dst, "{}")

    assert not stale.exists()
    assert live.exists()  # modified within a day: an unlocked stage may own it
    assert other.exists()  # another destination's, and not a media temp file
    assert dst.read_text(encoding="utf-8") == "{}"


def test_killed_media_writers_temp_is_reclaimed_by_any_write_in_the_directory(
    tmp_path,
):
    # SIGKILL leaves a multi-GB .part behind; nothing holds its lock any more
    idle = fsio.ABANDONED_MEDIA_TEMP_SECONDS + 60
    killed_burn = tmp_path / ".ep.burn.abcd_123.part.mp4"
    killed_pack = tmp_path / ".ep.zh.ijkl_890.part.MKV"
    recent = tmp_path / ".ep.burn.efgh4567.part.mp4"
    text_stage = tmp_path / ".ep.mnop1234.part.vtt"
    for path in (killed_burn, killed_pack, recent, text_stage):
        path.write_bytes(b"partial")
    for path in (killed_burn, killed_pack, text_stage):
        _age(path, idle)
    _age(recent, 30)

    fsio.atomic_write_text(tmp_path / "ep.zh.vtt", "WEBVTT\n")

    assert not killed_burn.exists()
    assert not killed_pack.exists()
    assert recent.exists()  # may be an older version's live, unlocked writer
    assert text_stage.exists()  # text stages are not locked: day-long rule only


def test_media_temp_held_by_a_live_writer_is_never_reclaimed(tmp_path):
    held = tmp_path / ".ep.burn.abcd_123.part.mp4"
    held.write_bytes(b"encoding")
    _age(held, fsio.STALE_TEMP_SECONDS + 60)
    fd = os.open(held, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another writer's lock
        fsio.atomic_write_text(tmp_path / "ep.burn.mp4", "x")
        assert held.exists()
    finally:
        os.close(fd)


def test_atomic_path_locks_its_temp_until_it_lands(tmp_path):
    dst = tmp_path / "ep.burn.mp4"
    with fsio.atomic_path(dst) as tmp:
        tmp.write_bytes(b"encoded")
        probe = os.open(tmp, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # this process's own live temp file is never swept, however idle
            _age(tmp, fsio.STALE_TEMP_SECONDS + 60)
            fsio.sweep_stale_temps(dst)
            assert tmp.exists()
        finally:
            os.close(probe)
    probe = os.open(dst, os.O_RDWR)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released on landing
    finally:
        os.close(probe)
    assert fsio._LIVE_TEMPS == set()


def test_media_write_reclaims_a_temp_that_went_idle_during_it(tmp_path):
    # a re-run started right after the kill: the old temp was still recent
    # when this run began, and is reclaimed once this run's output lands
    dst = tmp_path / "ep.burn.mp4"
    killed = tmp_path / ".ep.burn.abcd_123.part.mp4"
    with fsio.atomic_path(dst) as tmp:
        killed.write_bytes(b"partial")
        _age(killed, fsio.ABANDONED_MEDIA_TEMP_SECONDS + 60)
        tmp.write_bytes(b"encoded")
    assert not killed.exists()
    assert dst.read_bytes() == b"encoded"


def test_without_locks_only_the_destinations_day_old_temp_goes(tmp_path, monkeypatch):
    def no_locks(*_args):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(fsio.fcntl, "flock", no_locks)
    dst = tmp_path / "ep.burn.mp4"
    own = tmp_path / ".ep.burn.abcd_123.part.mp4"
    other = tmp_path / ".ep.pack.efgh4567.part.mkv"
    for path in (own, other):
        path.write_bytes(b"partial")
        _age(path, fsio.STALE_TEMP_SECONDS + 60)
    with fsio.atomic_path(dst) as tmp:
        tmp.write_bytes(b"encoded")
    assert not own.exists()
    assert other.exists()  # cannot tell whether its writer is alive
