"""Per-episode transaction lock shared by phase-2 writers and replayers."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpisodeLockHandle:
    episode_path: Path
    lock_path: Path


def episode_base_path(path: Path) -> Path:
    value = Path(path)
    stem = value.name.rsplit(".", 1)[0] if "." in value.name else value.name
    return value.with_name(stem)


def episode_lock_path(path: Path) -> Path:
    lock = episode_base_path(path).with_name(
        f"{episode_base_path(path).name}.episode.lock"
    )
    return Path(os.path.realpath(os.fspath(lock)))


@contextmanager
def episode_lock(path: Path) -> Iterator[EpisodeLockHandle]:
    """Take the canonical persistent exclusive lock for one episode stem."""
    episode = episode_base_path(path)
    lock = episode_lock_path(path)
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield EpisodeLockHandle(episode_path=episode, lock_path=lock)
    finally:
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
