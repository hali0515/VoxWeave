# tests/test_fsio.py
# Atomic file writes: content lands only via os.replace, so an interrupted
# write can never leave a truncated file at the destination path.

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
