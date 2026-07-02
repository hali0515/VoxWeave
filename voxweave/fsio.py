"""Atomic file-write helpers.

Every artifact writer (VTT/JSON siblings, translated subtitles, mux/burn
outputs, the vocals cache) must go through these so an interrupted run —
Ctrl-C, OOM kill, full disk — can never leave a truncated file at a path a
later run (or the user) trusts. The write lands in a temp file in the same
directory and reaches the destination only via ``os.replace``.
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
