"""Atomic file-write helpers.

Every artifact writer (VTT/JSON siblings, translated subtitles, mux/burn
outputs, the vocals cache) must go through these. Replaceable outputs land via
a same-directory temp file and ``os.replace``; protected user sidecars use an
exclusive first write so a concurrent creator can never be overwritten.
"""

from __future__ import annotations

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

    Exclusive creation closes the race with an earlier existence check without
    requiring hard-link support (media libraries commonly live on FAT/FUSE/SMB
    filesystems).  A failed write removes the newly-created partial file.
    """
    dst = Path(dst)
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
