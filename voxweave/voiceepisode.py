"""Per-episode transaction lock shared by phase-2 writers and replayers."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from voxweave import artifacts

_MEDIA_SUFFIXES = (
    ".mkv",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".ts",
    ".m4v",
    ".flac",
    ".wav",
    ".m4a",
    ".mp3",
    ".aac",
    ".opus",
    ".ogg",
)
_DERIVED_SUBTITLE_TAGS = frozenset({"asrfix", "sdh"})


@dataclass(frozen=True)
class EpisodeLockHandle:
    episode_path: Path
    lock_path: Path


def episode_base_path(path: Path) -> Path:
    value = Path(path)
    stem = value.name.rsplit(".", 1)[0] if "." in value.name else value.name
    return value.with_name(stem)


def episode_lock_path(path: Path) -> Path:
    """Return the cache-owned lock shared by a media episode and its siblings."""
    return _lock_path_for_owner(_episode_owner(Path(path)))


def _lock_path_for_owner(owner: Path) -> Path:
    return artifacts.claim_paths(owner).episode_lock


def _artifact_lock_paths(owner: Path) -> tuple[Path, tuple[Path, ...]]:
    """Return every same-directory/same-stem claim lock in stable order."""
    claimed = artifacts.claim_paths(owner)
    locks: list[Path] = []
    for source in artifacts.claimed_sources(claimed.source.parent, claimed.source.stem):
        paths = artifacts.inspect_paths(source)
        if paths is not None:
            locks.append(paths.episode_lock)
    locks.append(claimed.episode_lock)
    return claimed.episode_lock, tuple(sorted(dict.fromkeys(locks), key=str))


def _legacy_lock_path(owner: Path) -> Path:
    """Return the historical adjacent lock without following its final node."""
    episode = episode_base_path(owner)
    parent = Path(os.path.realpath(os.fspath(episode.parent)))
    return parent / f"{episode.name}.episode.lock"


def _open_lock(path: Path, *, create: bool) -> int:
    """Open one lock without following or blocking on hostile filesystem nodes."""
    flags = (
        os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if create:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"episode lock is not a regular file: {path}")
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _episode_owner(path: Path) -> Path:
    """Resolve a subtitle/JSON reference to sibling media when one is present."""
    value = Path(path)
    if value.suffix.lower() in _MEDIA_SUFFIXES:
        return value
    parent = value.parent if str(value.parent) else Path(".")
    try:
        entries = tuple(parent.iterdir())
    except OSError:
        return value
    order = {suffix: index for index, suffix in enumerate(_MEDIA_SUFFIXES)}
    from voxweave.mux import detect_subtitle_language

    stem = episode_base_path(value).name
    normalized_parent = Path(os.path.realpath(os.fspath(parent)))
    while True:
        matches = sorted(
            (
                order[candidate.suffix.lower()],
                str(candidate),
                candidate,
            )
            for candidate in entries
            if candidate.is_file()
            and candidate.suffix.lower() in order
            and episode_base_path(candidate).name == stem
        )
        if matches:
            return matches[0][2]
        recorded = sorted(
            (
                order[source.suffix.lower()],
                str(source),
                source,
            )
            for source in artifacts.claimed_sources(normalized_parent, stem)
            if source.suffix.lower() in order
        )
        if recorded:
            return recorded[0][2]
        if "." not in stem:
            break
        tag = stem.rsplit(".", 1)[1].casefold()
        tagged = parent / stem
        if (
            tag not in _DERIVED_SUBTITLE_TAGS
            and detect_subtitle_language(tagged) is None
        ):
            break
        stem = stem.rsplit(".", 1)[0]
    return value


@contextmanager
def episode_lock(path: Path) -> Iterator[EpisodeLockHandle]:
    """Take the canonical persistent exclusive lock for one episode stem."""
    owner = _episode_owner(Path(path))
    episode = episode_base_path(owner)
    domain_lock = artifacts.episode_domain_lock_path(owner)
    lock = domain_lock
    descriptors: list[int] = []
    try:
        legacy = _legacy_lock_path(owner)
        try:
            legacy.lstat()
        except FileNotFoundError:
            pass
        else:
            descriptors.append(_open_lock(legacy, create=False))
        descriptors.append(_open_lock(domain_lock, create=True))
        for descriptor in descriptors:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            lock, artifact_locks = _artifact_lock_paths(owner)
        except artifacts.ArtifactMarkerError:
            # A selected adjacent legacy lane must not be blocked by unrelated
            # cache corruption. Any later cache access still validates and
            # fails closed at its actual resolver boundary.
            artifact_locks = ()
        for lock_path in artifact_locks:
            descriptor = _open_lock(lock_path, create=True)
            descriptors.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield EpisodeLockHandle(episode_path=episode, lock_path=lock)
    finally:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


__all__ = [
    "EpisodeLockHandle",
    "episode_base_path",
    "episode_lock",
    "episode_lock_path",
]
