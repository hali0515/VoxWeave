"""Immutable private media snapshots for phase-2 byte authority."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import logging
import os
import re
import secrets
import shutil
import stat
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Literal

from voxweave.voicebase import Phase2DataError, media_fingerprint_from_fd

SNAPSHOT_MAX_AGE_SECONDS = 60 * 60
SNAPSHOT_NAME_RE = re.compile(r"^snapshot-[0-9a-f]{32}\.[A-Za-z0-9]{1,16}$")
COPY_CHUNK_BYTES = 1024 * 1024
CopyMethod = Literal["reflink", "copy"]

_FICLONE = 0x40049409
_COPYFILE_DATA = 1 << 3
_COPYFILE_CLONE_FORCE = 1 << 25

log = logging.getLogger("voxweave")


class SnapshotUnavailable(RuntimeError):
    """A private, verified snapshot could not be created or retained."""


def default_cache_root() -> Path:
    return Path(
        os.environ.get(
            "VOXWEAVE_CACHE_ROOT",
            str(Path.home() / ".cache" / "voxweave"),
        )
    ).expanduser()


def snapshots_directory(cache_root: Path | None = None) -> Path:
    return (
        default_cache_root() if cache_root is None else Path(cache_root)
    ) / "snapshots"


def _ensure_snapshot_directory(cache_root: Path | None) -> Path:
    directory = snapshots_directory(cache_root)
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = os.lstat(directory)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SnapshotUnavailable(
                f"snapshot directory is not a private directory: {directory}"
            )
        os.chmod(directory, 0o700)
    except SnapshotUnavailable:
        raise
    except OSError as exc:
        raise SnapshotUnavailable(
            f"cannot create private snapshot directory {directory}: {exc}"
        ) from exc
    return directory


def _safe_suffix(source: Path) -> str:
    suffix = source.suffix
    if re.fullmatch(r"\.[A-Za-z0-9]{1,16}", suffix):
        return suffix
    return ".media"


def _create_destination(directory: Path, suffix: str) -> tuple[Path, int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(1024):
        path = directory / f"snapshot-{secrets.token_hex(16)}{suffix}"
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        os.fchmod(descriptor, 0o600)
        return path, descriptor
    raise SnapshotUnavailable("could not allocate an unpredictable snapshot name")


def _open_source(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise SnapshotUnavailable(f"media source is not a regular file: {path}")
    try:
        os.pread(descriptor, 0, 0)
    except OSError as exc:
        os.close(descriptor)
        raise SnapshotUnavailable(f"media source is not seekable: {path}") from exc
    return descriptor


def _darwin_reflink(source_fd: int, destination_fd: int) -> None:
    """Use fcopyfile's forced clone flag so both endpoints remain fd-bound."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        fcopyfile = libc.fcopyfile
    except AttributeError as exc:
        raise OSError(errno.EOPNOTSUPP, "fcopyfile is unavailable") from exc
    fcopyfile.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    fcopyfile.restype = ctypes.c_int
    result = fcopyfile(
        source_fd,
        destination_fd,
        None,
        _COPYFILE_DATA | _COPYFILE_CLONE_FORCE,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _clone_reflink(source_fd: int, destination_fd: int) -> None:
    """Attempt an fd-to-fd COW clone, raising OSError when unsupported."""
    if sys.platform.startswith("linux"):
        fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        return
    if sys.platform == "darwin":
        _darwin_reflink(source_fd, destination_fd)
        return
    raise OSError(errno.EOPNOTSUPP, "reflink cloning is unsupported")


def _write_all_at(fd: int, payload: bytes, offset: int) -> None:
    written = 0
    while written < len(payload):
        count = os.pwrite(fd, payload[written:], offset + written)
        if count <= 0:
            raise OSError(errno.EIO, "snapshot write made no progress")
        written += count


def _copy_sequential(source_fd: int, destination_fd: int, size: int) -> None:
    """Copy exactly the preflight size without ever reopening the live path."""
    os.ftruncate(destination_fd, 0)
    offset = 0
    while offset < size:
        length = min(COPY_CHUNK_BYTES, size - offset)
        chunk = os.pread(source_fd, length, offset)
        if not chunk:
            raise OSError(errno.EIO, "media ended during snapshot copy")
        _write_all_at(destination_fd, chunk, offset)
        offset += len(chunk)
    os.ftruncate(destination_fd, size)
    os.fsync(destination_fd)


def _available_bytes(directory: Path) -> int:
    return shutil.disk_usage(directory).free


def _verify_private_copy(
    source_fd: int,
    destination_fd: int,
    expected_size: int,
) -> str:
    source_stat = os.fstat(source_fd)
    destination_stat = os.fstat(destination_fd)
    if source_stat.st_size != expected_size:
        raise SnapshotUnavailable("media size changed while creating its snapshot")
    if destination_stat.st_size != expected_size:
        raise SnapshotUnavailable("snapshot size does not match its media source")
    try:
        source_fingerprint = media_fingerprint_from_fd(
            source_fd,
            size=expected_size,
        )
        destination_fingerprint = media_fingerprint_from_fd(
            destination_fd,
            size=expected_size,
        )
    except (OSError, Phase2DataError) as exc:
        raise SnapshotUnavailable(f"cannot verify media snapshot: {exc}") from exc
    if source_fingerprint != destination_fingerprint:
        raise SnapshotUnavailable(
            "snapshot sampled identity differs from its media source"
        )
    return destination_fingerprint


def _same_open_file(path: Path, descriptor: int) -> bool:
    try:
        path_stat = os.lstat(path)
        fd_stat = os.fstat(descriptor)
    except (FileNotFoundError, OSError):
        return False
    return (
        stat.S_ISREG(path_stat.st_mode)
        and path_stat.st_dev == fd_stat.st_dev
        and path_stat.st_ino == fd_stat.st_ino
    )


def _unlink_if_owned(path: Path, descriptor: int) -> None:
    if _same_open_file(path, descriptor):
        path.unlink()


def cleanup_stale_snapshots(
    directory: Path,
    *,
    now: float | None = None,
    max_age_seconds: float = SNAPSHOT_MAX_AGE_SECONDS,
) -> tuple[Path, ...]:
    """Remove only old, regular, unlocked files in the owned namespace."""
    current_time = time.time() if now is None else now
    removed: list[Path] = []
    try:
        candidates = tuple(Path(directory).iterdir())
    except FileNotFoundError:
        return ()
    for candidate in candidates:
        if SNAPSHOT_NAME_RE.fullmatch(candidate.name) is None:
            continue
        try:
            path_stat = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            continue
        if current_time - path_stat.st_mtime <= max_age_seconds:
            continue
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError:
            continue
        try:
            if not _same_open_file(candidate, descriptor):
                continue
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if _same_open_file(candidate, descriptor):
                candidate.unlink()
                removed.append(candidate)
        finally:
            os.close(descriptor)
    return tuple(removed)


class MediaSnapshot:
    """Exception-safe context manager owning one immutable private media file."""

    def __init__(
        self,
        source: Path,
        *,
        cache_root: Path | None = None,
        janitor_age_seconds: float = SNAPSHOT_MAX_AGE_SECONDS,
    ) -> None:
        self.source = Path(source)
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.janitor_age_seconds = janitor_age_seconds
        self._snapshot_path: Path | None = None
        self._snapshot_fd: int | None = None
        self._fingerprint: str | None = None
        self._size: int | None = None
        self.copy_method: CopyMethod | None = None
        self.free_space_sufficient: bool | None = None
        self._entered = False
        self._used = False

    @property
    def path(self) -> Path:
        if self._snapshot_path is None or not self._entered:
            raise SnapshotUnavailable("media snapshot is not active")
        return self._snapshot_path

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None or not self._entered:
            raise SnapshotUnavailable("media snapshot is not active")
        return self._fingerprint

    @property
    def size(self) -> int:
        if self._size is None or not self._entered:
            raise SnapshotUnavailable("media snapshot is not active")
        return self._size

    def __enter__(self) -> MediaSnapshot:
        if self._used:
            raise SnapshotUnavailable("MediaSnapshot instances are single-use")
        self._used = True
        directory = _ensure_snapshot_directory(self.cache_root)
        cleanup_stale_snapshots(
            directory,
            max_age_seconds=self.janitor_age_seconds,
        )
        source_fd: int | None = None
        try:
            source_fd = _open_source(self.source)
            source_stat = os.fstat(source_fd)
            self._size = source_stat.st_size
            destination, destination_fd = _create_destination(
                directory,
                _safe_suffix(self.source),
            )
            self._snapshot_path = destination
            self._snapshot_fd = destination_fd
            fcntl.flock(destination_fd, fcntl.LOCK_SH)

            try:
                _clone_reflink(source_fd, destination_fd)
                os.fchmod(destination_fd, 0o600)
                os.fsync(destination_fd)
                self._fingerprint = _verify_private_copy(
                    source_fd,
                    destination_fd,
                    self._size,
                )
                self.copy_method = "reflink"
            except (OSError, SnapshotUnavailable, Phase2DataError) as clone_error:
                os.ftruncate(destination_fd, 0)
                try:
                    available = _available_bytes(directory)
                except OSError as advisory_error:
                    available = None
                    log.warning(
                        "snapshot free-space advisory unavailable: %s",
                        advisory_error,
                    )
                self.free_space_sufficient = (
                    None if available is None else available >= self._size
                )
                if self.free_space_sufficient is False:
                    log.warning(
                        "snapshot free-space advisory: %s bytes available for %s bytes",
                        available,
                        self._size,
                    )
                try:
                    _copy_sequential(source_fd, destination_fd, self._size)
                    os.fchmod(destination_fd, 0o600)
                    self._fingerprint = _verify_private_copy(
                        source_fd,
                        destination_fd,
                        self._size,
                    )
                except (OSError, SnapshotUnavailable, Phase2DataError) as copy_error:
                    raise SnapshotUnavailable(
                        "cannot create a verified private media snapshot: "
                        f"clone failed ({clone_error}); copy failed ({copy_error})"
                    ) from copy_error
                self.copy_method = "copy"
            self._entered = True
            return self
        except SnapshotUnavailable:
            self._cleanup()
            raise
        except OSError as exc:
            self._cleanup()
            raise SnapshotUnavailable(
                f"cannot create private media snapshot for {self.source}: {exc}"
            ) from exc
        except BaseException:
            self._cleanup()
            raise
        finally:
            if source_fd is not None:
                os.close(source_fd)

    def _cleanup(self) -> None:
        descriptor = self._snapshot_fd
        path = self._snapshot_path
        try:
            if descriptor is not None and path is not None:
                _unlink_if_owned(path, descriptor)
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            self._snapshot_fd = None
            self._snapshot_path = None
            self._entered = False

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        self._cleanup()
        return False


__all__ = [
    "COPY_CHUNK_BYTES",
    "MediaSnapshot",
    "SNAPSHOT_MAX_AGE_SECONDS",
    "SNAPSHOT_NAME_RE",
    "SnapshotUnavailable",
    "cleanup_stale_snapshots",
    "default_cache_root",
    "snapshots_directory",
]
