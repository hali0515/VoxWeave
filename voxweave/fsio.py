"""Atomic file-write helpers.

Every artifact writer (VTT/JSON siblings, translated subtitles, mux/burn
outputs, the vocals cache) must go through these. Replaceable outputs land via
a same-directory temp file and ``os.replace``; protected user sidecars use an
exclusive first write so a concurrent creator can never be overwritten.
"""

from __future__ import annotations

import errno
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def atomic_path(dst: Path) -> Iterator[Path]:
    """Yield a temp path next to ``dst``; on clean exit rename it onto ``dst``,
    on any exception delete it and leave ``dst`` untouched.

    The temp file keeps ``dst``'s suffix (ffmpeg picks its muxer from the
    output extension) and lives in the same directory (same filesystem, so the
    ``os.replace`` is atomic).
    """
    dst = Path(dst)
    fd, name = tempfile.mkstemp(
        dir=dst.parent, prefix=f".{dst.stem}.", suffix=f".part{dst.suffix}"
    )
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        os.replace(tmp, dst)
    except BaseException:  # KeyboardInterrupt included: never leave a .part file
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(dst: Path, text: str, *, encoding: str = "utf-8") -> None:
    """``Path.write_text`` with atomic-replace semantics (fsynced before rename)."""
    with atomic_path(dst) as tmp:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_text_new(dst: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically create a text file, raising ``FileExistsError`` if it exists.

    Prefer installing a completed, fsynced temp through an atomic hard link. On
    filesystems without hard links, atomically claim ``dst`` with ``O_EXCL`` and
    replace that claim with the completed temp. The fallback has a tiny crash
    window where an empty claim can remain, but never overwrites another writer.
    """
    dst = Path(dst)
    fd, name = tempfile.mkstemp(
        dir=dst.parent, prefix=f".{dst.stem}.", suffix=f".part{dst.suffix}"
    )
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, dst)
        except OSError as exc:
            if isinstance(exc, FileExistsError) or exc.errno not in {
                errno.EPERM,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            claim_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            os.close(claim_fd)
            try:
                os.replace(tmp, dst)
            except BaseException:
                dst.unlink(missing_ok=True)
                raise
    finally:
        tmp.unlink(missing_ok=True)
