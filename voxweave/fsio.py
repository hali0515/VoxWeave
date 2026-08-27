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
from collections.abc import Callable
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


def atomic_write_text(
    dst: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    before_replace: Callable[[], str | None] | None = None,
) -> None:
    """Write fsynced text and atomically replace ``dst``.

    ``before_replace`` runs after the requested bytes are durable in the temp
    file and immediately before the rename. Returning text substitutes a
    fsynced fallback; returning ``None`` keeps the prepared bytes. This lets a
    transaction make its last authority decision at the actual replace edge.
    """

    def write_temp(path: Path, content: str) -> None:
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

    with atomic_path(dst) as tmp:
        write_temp(tmp, text)
        if before_replace is not None:
            fallback = before_replace()
            if fallback is not None:
                write_temp(tmp, fallback)


def atomic_write_text_new(
    dst: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    before_install: Callable[[], None] | None = None,
) -> None:
    """Atomically create a text file, raising ``FileExistsError`` if it exists.

    Prefer installing a completed, fsynced temp through an atomic hard link. On
    filesystems without hard links, atomically claim ``dst`` with ``O_EXCL`` and
    replace that claim with the completed temp. The fallback has a tiny crash
    window where an empty claim can remain, but never overwrites another writer.

    ``before_install`` runs after the requested bytes are durable and adjacent
    to each protected install attempt. A fallback from hard links to ``O_EXCL``
    therefore invokes it again before claiming ``dst``. Raising leaves ``dst``
    absent and removes the prepared temp.
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
        if before_install is not None:
            before_install()
        try:
            os.link(tmp, dst)
        except OSError as exc:
            if isinstance(exc, FileExistsError) or exc.errno not in {
                errno.EPERM,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            if before_install is not None:
                before_install()
            claim_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            os.close(claim_fd)
            try:
                os.replace(tmp, dst)
            except BaseException:
                dst.unlink(missing_ok=True)
                raise
    finally:
        tmp.unlink(missing_ok=True)
