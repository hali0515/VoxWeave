# tests/test_fsio.py
# Atomic file writes: replaceable artifacts use os.replace, while protected new
# user sidecars publish completed content without overwriting concurrent files.

import errno

import pytest

from voxweave import fsio


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


def test_atomic_write_text_new_creates_without_temp_residue(tmp_path):
    dst = tmp_path / "mapping.json"
    fsio.atomic_write_text_new(dst, '{"version": 1}')
    assert dst.read_text(encoding="utf-8") == '{"version": 1}'
    assert list(tmp_path.iterdir()) == [dst]


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


@pytest.mark.parametrize("link_errno", [errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV])
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
