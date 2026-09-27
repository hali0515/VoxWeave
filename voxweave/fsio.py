"""Atomic file-write helpers.

Every artifact writer (VTT/JSON siblings, translated subtitles, mux/burn
outputs, the vocals cache) must go through these. Replaceable outputs land via
a same-directory temp file and ``os.replace``; protected user sidecars use an
exclusive first write so a concurrent creator can never be overwritten.

Permissions: by default a written file is a user deliverable, so a new file
gets ``0o666`` less the process umask (as if it had been opened directly) and
a replaced file keeps the mode it had. Writers of private data (voice
library, ``cache/<stem>/`` artifacts, progress sidecars, speaker mappings)
pass ``private=True`` and always get ``0o600``. Temp files are ``0o600``
until they land either way.
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

FileGeneration = tuple[int, int, int, int]

# A temp file untouched for this long belongs to a killed writer: live writers
# finish text in well under a second and ffmpeg updates its output constantly.
STALE_TEMP_SECONDS = 24 * 60 * 60

# pack/burn temp files can hold several GB, so they are reclaimed much sooner,
# whichever output they were for. Only atomic_path writes these suffixes, and
# it holds an exclusive flock on its temp file for as long as the writer
# lives: one that nobody holds and nobody has written for this long is a
# killed writer's. (The idle time also covers mounts whose locks other
# machines cannot see, since ffmpeg writes its output constantly.)
MEDIA_TEMP_SUFFIXES = frozenset({".mkv", ".mp4", ".webm", ".m4v", ".mov"})
ABANDONED_MEDIA_TEMP_SECONDS = 5 * 60

_NAME_MAX = 255  # bytes per file name on every filesystem voxweave targets
_TEMP_RANDOM_LENGTH = 8  # tempfile's random name part
_TEMP_RANDOM = f"[a-z0-9_]{{{_TEMP_RANDOM_LENGTH}}}"
_MEDIA_TEMP = re.compile(
    r"\..*\."
    + _TEMP_RANDOM
    + r"\.part(?i:"
    + "|".join(re.escape(suffix) for suffix in sorted(MEDIA_TEMP_SUFFIXES))
    + ")"
)
_UMASK_LOCK = threading.Lock()

# (st_dev, st_ino) of the temp files this process is writing through
# atomic_path. A sweep skips them without probing their lock: NFS emulates
# flock with byte-range locks, so a probe from the writer's own process is
# not a reliable liveness test there.
_LIVE_TEMPS: set[tuple[int, int]] = set()
_LIVE_TEMPS_LOCK = threading.Lock()


def _process_umask() -> int:
    """The process umask. Linux reports it in /proc; elsewhere it can only be
    read by setting it, so it is briefly set to a stricter value (a file
    created by another thread in that window is more private, never less)."""
    try:
        with open("/proc/self/status", encoding="ascii", errors="replace") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    with _UMASK_LOCK:
        mask = os.umask(0o077)
        os.umask(mask)
    return mask


def deliverable_mode(dst: Path) -> int:
    """Permission bits for a user deliverable about to land on ``dst``.

    An existing regular file keeps its own bits, so a user's chmod survives a
    re-run; a new file gets ``0o666`` less the process umask.
    """
    try:
        existing = os.stat(dst)
    except OSError:
        existing = None
    if existing is not None and stat.S_ISREG(existing.st_mode):
        return stat.S_IMODE(existing.st_mode) & 0o777
    return 0o666 & ~_process_umask()


def _name_max(directory: Path) -> int:
    try:
        return int(os.pathconf(directory, "PC_NAME_MAX"))
    except (AttributeError, OSError, ValueError):
        return _NAME_MAX


def temp_affixes(dst: Path) -> tuple[str, str]:
    """``mkstemp`` prefix and suffix for a temp file next to ``dst``.

    The name is ``.<stem>.<random>.part<suffix>``: hidden, same directory
    (same filesystem, so the final ``os.replace`` is atomic) and ending in
    ``dst``'s suffix (ffmpeg picks its muxer from the output extension). The
    stem is shortened when the full name would exceed the filesystem's name
    limit, so a valid ``dst`` name always has a valid temp name.
    """
    dst = Path(dst)
    suffix = f".part{dst.suffix}"
    budget = (
        _name_max(dst.parent)
        - len(os.fsencode(suffix))
        - _TEMP_RANDOM_LENGTH
        - 2  # the leading dot and the dot after the stem
    )
    stem = dst.stem
    while stem and len(os.fsencode(stem)) > budget:
        stem = stem[:-1]
    return f".{stem}.", suffix


def _temp_lock_held(fd: int) -> bool | None:
    """Whether a live writer holds the temp file's lock (``None``: this
    filesystem cannot tell)."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return None
    return False


def _remove_abandoned(
    entry: os.DirEntry[str],
    *,
    now: float,
    free_after: float | None,
    unknown_after: float | None,
) -> None:
    """Unlink one temp file once idle long enough and not held by a writer.

    ``free_after`` applies when its lock is provably free, ``unknown_after``
    when the filesystem cannot tell; ``None`` never removes it in that state.
    """
    info = entry.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        return
    with _LIVE_TEMPS_LOCK:
        if (info.st_dev, info.st_ino) in _LIVE_TEMPS:
            return
    idle = now - info.st_mtime
    ages = [age for age in (free_after, unknown_after) if age is not None]
    if not ages or idle <= min(ages):
        return
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(entry.path, flags | getattr(os, "O_NONBLOCK", 0))
    try:
        held = _temp_lock_held(fd)
        if held:
            return
        limit = unknown_after if held is None else free_after
        if limit is None or idle <= limit:
            return
        if os.fstat(fd).st_ino == info.st_ino:
            os.unlink(entry.path)
    finally:
        os.close(fd)


def sweep_stale_temps(dst: Path, *, max_age: float = STALE_TEMP_SECONDS) -> None:
    """Delete temp files that killed writers left next to ``dst``.

    ``dst``'s own temp files go once nobody has modified them for ``max_age``
    seconds. pack/burn temp files of any output in the directory go once
    idle for ``ABANDONED_MEDIA_TEMP_SECONDS`` with their lock provably free
    (see ``MEDIA_TEMP_SUFFIXES``). Only names this module generates are
    considered, and a temp file whose writer holds its lock is never touched.
    Best effort: errors are ignored.
    """
    dst = Path(dst)
    prefix, suffix = temp_affixes(dst)
    own = re.compile(re.escape(prefix) + _TEMP_RANDOM + re.escape(suffix))
    now = time.time()
    try:
        entries = os.scandir(dst.parent)
    except OSError:
        return
    with entries:
        for entry in entries:
            is_own = own.fullmatch(entry.name) is not None
            is_media = _MEDIA_TEMP.fullmatch(entry.name) is not None
            if not (is_own or is_media):
                continue
            try:
                _remove_abandoned(
                    entry,
                    now=now,
                    free_after=ABANDONED_MEDIA_TEMP_SECONDS if is_media else max_age,
                    unknown_after=max_age if is_own else None,
                )
            except OSError:
                continue


def _hold_temp(fd: int) -> tuple[int, int]:
    """Mark a temp file as this process's live write: lock it (best effort)
    and register it; see ``MEDIA_TEMP_SUFFIXES`` and ``_LIVE_TEMPS``."""
    info = os.fstat(fd)
    key = (info.st_dev, info.st_ino)
    with _LIVE_TEMPS_LOCK:
        _LIVE_TEMPS.add(key)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        pass  # no locks on this mount: only the idle-time rules protect it
    return key


def _release_temp(fd: int, key: tuple[int, int] | None) -> None:
    if key is not None:
        with _LIVE_TEMPS_LOCK:
            _LIVE_TEMPS.discard(key)
    os.close(fd)


def make_temp(dst: Path) -> tuple[int, Path]:
    """Create a ``0o600`` temp file next to ``dst``; return its fd and path.

    Abandoned temp files in the directory are swept first (see
    ``sweep_stale_temps``).
    """
    dst = Path(dst)
    sweep_stale_temps(dst)
    prefix, suffix = temp_affixes(dst)
    fd, name = tempfile.mkstemp(dir=dst.parent, prefix=prefix, suffix=suffix)
    return fd, Path(name)


def fsync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable (best effort: some filesystems
    cannot fsync a directory)."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _set_mode(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # some network mounts refuse chmod; the file still lands


def install_temp(tmp: Path, dst: Path, *, private: bool = False) -> None:
    """Give a finished temp file its final mode and rename it onto ``dst``."""
    tmp, dst = Path(tmp), Path(dst)
    if not private:
        _set_mode(tmp, deliverable_mode(dst))
    os.replace(tmp, dst)
    fsync_directory(dst.parent)


@contextmanager
def atomic_path(dst: Path, *, private: bool = False) -> Iterator[Path]:
    """Yield a temp path next to ``dst``; on clean exit rename it onto ``dst``,
    on any exception delete it and leave ``dst`` untouched.

    The temp file keeps ``dst``'s suffix (ffmpeg picks its muxer from the
    output extension) and lives in the same directory (same filesystem, so the
    ``os.replace`` is atomic). It stays locked while the writer lives, so a
    sweep can tell it from a killed writer's (see ``sweep_stale_temps``); a
    media output sweeps again once it lands, reclaiming a killed run's temp
    file that went idle meanwhile. See the module docstring for ``private``.
    """
    dst = Path(dst)
    fd, tmp = make_temp(dst)
    key: tuple[int, int] | None = None
    try:
        key = _hold_temp(fd)
        yield tmp
        install_temp(tmp, dst, private=private)
    except BaseException:  # KeyboardInterrupt included: never leave a .part file
        tmp.unlink(missing_ok=True)
        raise
    finally:
        _release_temp(fd, key)
    if dst.suffix.lower() in MEDIA_TEMP_SUFFIXES:
        sweep_stale_temps(dst)


def atomic_write_text(
    dst: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    before_replace: Callable[[], str | None] | None = None,
    private: bool = False,
) -> None:
    """Write fsynced text and atomically replace ``dst``.

    ``before_replace`` runs after the requested bytes are durable in the temp
    file and immediately before the rename. Returning text substitutes a
    fsynced fallback; returning ``None`` keeps the prepared bytes. This lets a
    transaction make its last authority decision at the actual replace edge.
    See the module docstring for ``private``.
    """

    def write_temp(path: Path, content: str) -> None:
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

    with atomic_path(dst, private=private) as tmp:
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
    private: bool = False,
) -> FileGeneration:
    """Atomically create a text file, raising ``FileExistsError`` if it exists.

    Prefer installing a completed, fsynced temp through an atomic hard link. On
    filesystems without hard links, atomically claim ``dst`` with ``O_EXCL`` and
    replace that claim with the completed temp. The fallback has a tiny crash
    window where an empty claim can remain, but never overwrites another writer.

    ``before_install`` runs after the requested bytes are durable and adjacent
    to each protected install attempt. A fallback from hard links to ``O_EXCL``
    therefore invokes it again before claiming ``dst``. Raising leaves ``dst``
    absent and removes the prepared temp. The return value identifies the exact
    file generation installed, captured from the durable temp before publication.
    A new file gets ``0o666`` less the umask, or ``0o600`` when ``private``.
    """
    dst = Path(dst)
    fd, tmp = make_temp(dst)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            if not private:
                try:
                    os.fchmod(f.fileno(), 0o666 & ~_process_umask())
                except OSError:
                    pass  # some network mounts refuse chmod
            os.fsync(f.fileno())
            metadata = os.fstat(f.fileno())
            installed_generation = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
        if before_install is not None:
            before_install()
        try:
            os.link(tmp, dst)
        except OSError as exc:
            if isinstance(exc, FileExistsError) or exc.errno not in {
                errno.EPERM,
                errno.EOPNOTSUPP,
                errno.ENOTSUP,  # macOS: distinct from EOPNOTSUPP
                errno.ENOSYS,  # FUSE filesystems without link()
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
        fsync_directory(dst.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return installed_generation
