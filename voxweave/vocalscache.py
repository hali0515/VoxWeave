"""Strict companion metadata and locking helpers for the vocals FLAC cache."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from voxweave import fsio
from voxweave.align_failures import CanonicalFailure
from voxweave.voicebase import (
    CACHE_COMPANION_MAX_BYTES,
    MAX_PROVENANCE_STRING_BYTES,
    Phase2DataError,
    canonical_path,
    load_json_object,
    require_exact_int,
    require_mapping,
    require_sha256,
    require_string,
    require_version_one,
    write_json_object,
)

HASH_CHUNK_BYTES = 1024 * 1024


class CacheCompanionError(Phase2DataError):
    """A vocals-cache companion is malformed."""


class CacheCompanionMismatch(CacheCompanionError):
    """A well-formed companion does not describe the requested cache pair."""


def _classify_cache_failure(
    exc: BaseException,
    *,
    kind: str,
    detail_code: str,
) -> None:
    if not isinstance(getattr(exc, "failure", None), CanonicalFailure):
        exc.failure = CanonicalFailure(  # type: ignore[attr-defined]
            kind,
            "vocals-cache",
            detail_code,
        )


def classify_cache_decode_failure(exc: BaseException) -> None:
    """Attach the cache-decode terminal while preserving the public exception."""
    _classify_cache_failure(
        exc,
        kind="cache-operation-failed",
        detail_code="cache-decode",
    )


@dataclass(frozen=True)
class SeparatorIdentity:
    repo: str
    file: str
    checkpoint: str
    config_sha256: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "file": self.file,
            "checkpoint": self.checkpoint,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class ValidatedCacheCompanion:
    media_fingerprint: str
    separator: SeparatorIdentity
    cache_size: int
    cache_sha256: str


@dataclass(frozen=True)
class CacheLockHandle:
    cache_path: Path
    companion_path: Path
    lock_path: Path


def canonical_cache_path(path: Path) -> Path:
    """Cache-side name for voicebase's shared realpath normalization."""
    return canonical_path(path)


def cache_companion_path(cache_path: Path) -> Path:
    return Path(f"{canonical_cache_path(cache_path)}.meta.json")


def cache_lock_path(cache_path: Path) -> Path:
    return Path(f"{canonical_cache_path(cache_path)}.lock")


def validate_separator_identity(value: object) -> SeparatorIdentity:
    try:
        separator = require_mapping(value, "separator")
        repo = require_string(
            separator.get("repo"),
            "separator.repo",
            max_bytes=MAX_PROVENANCE_STRING_BYTES,
        )
        filename = require_string(
            separator.get("file"),
            "separator.file",
            max_bytes=MAX_PROVENANCE_STRING_BYTES,
        )
        checkpoint = require_string(
            separator.get("checkpoint"),
            "separator.checkpoint",
            max_bytes=MAX_PROVENANCE_STRING_BYTES,
        )
        config_sha256 = separator.get(
            "config_sha256",
            separator.get("config"),
        )
        if (
            "config_sha256" in separator
            and "config" in separator
            and separator["config_sha256"] != separator["config"]
        ):
            raise CacheCompanionError(
                "separator config and config_sha256 identities disagree"
            )
        config_hash = require_sha256(
            config_sha256,
            "separator.config_sha256",
        )
    except CacheCompanionError:
        raise
    except Phase2DataError as exc:
        raise CacheCompanionError(str(exc)) from exc
    return SeparatorIdentity(
        repo=repo,
        file=filename,
        checkpoint=checkpoint,
        config_sha256=config_hash,
    )


def validate_cache_companion(value: object) -> ValidatedCacheCompanion:
    """Validate a decoded v1 companion, accepting future top-level fields."""
    try:
        root = require_mapping(value, "cache companion")
        require_version_one(root.get("version"))
        media_fingerprint = require_sha256(
            root.get("media_fingerprint"),
            "media_fingerprint",
        )
        separator = validate_separator_identity(root.get("separator"))
        cache_size = require_exact_int(
            root.get("cache_size"),
            "cache_size",
            minimum=0,
        )
        cache_hash = require_sha256(root.get("cache_sha256"), "cache_sha256")
    except CacheCompanionError as exc:
        _classify_cache_failure(
            exc,
            kind="cache-companion-invalid",
            detail_code="companion-schema",
        )
        raise
    except Phase2DataError as exc:
        failure = CacheCompanionError(str(exc))
        _classify_cache_failure(
            failure,
            kind="cache-companion-invalid",
            detail_code="companion-schema",
        )
        raise failure from exc
    return ValidatedCacheCompanion(
        media_fingerprint=media_fingerprint,
        separator=separator,
        cache_size=cache_size,
        cache_sha256=cache_hash,
    )


def file_size_sha256(path: Path) -> tuple[int, str]:
    """Return exact size and full SHA-256 from one regular-file descriptor."""
    descriptor = os.open(
        canonical_cache_path(path),
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CacheCompanionError(f"cache is not a regular file: {path}")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(HASH_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise CacheCompanionMismatch("cache ended while hashing")
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
        ):
            raise CacheCompanionMismatch("cache changed while hashing")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def build_cache_companion(
    cache_path: Path,
    *,
    media_fingerprint: str,
    separator: Mapping[str, object] | SeparatorIdentity,
) -> dict[str, object]:
    """Build metadata from the exact finished FLAC bytes."""
    media_hash = require_sha256(media_fingerprint, "media_fingerprint")
    separator_identity = (
        separator
        if isinstance(separator, SeparatorIdentity)
        else validate_separator_identity(separator)
    )
    cache_size, cache_hash = file_size_sha256(cache_path)
    companion: dict[str, object] = {
        "version": 1,
        "media_fingerprint": media_hash,
        "separator": separator_identity.as_mapping(),
        "cache_size": cache_size,
        "cache_sha256": cache_hash,
    }
    validate_cache_companion(companion)
    return companion


def load_cache_companion(
    path: Path,
) -> tuple[dict[str, object], ValidatedCacheCompanion]:
    try:
        raw = load_json_object(path, max_bytes=CACHE_COMPANION_MAX_BYTES)
    except (OSError, Phase2DataError) as exc:
        classify_cache_decode_failure(exc)
        raise
    return raw, validate_cache_companion(raw)


def validate_cache_pair(
    companion: Mapping[str, object],
    cache_path: Path,
    *,
    media_fingerprint: str,
    separator: Mapping[str, object] | SeparatorIdentity,
) -> ValidatedCacheCompanion:
    """Require companion schema, full FLAC identity, media, and separator."""
    validated = validate_cache_companion(companion)
    expected_media = require_sha256(media_fingerprint, "expected media_fingerprint")
    expected_separator = (
        separator
        if isinstance(separator, SeparatorIdentity)
        else validate_separator_identity(separator)
    )
    if validated.media_fingerprint != expected_media:
        failure = CacheCompanionMismatch("cache companion media fingerprint differs")
        _classify_cache_failure(
            failure,
            kind="cache-companion-invalid",
            detail_code="companion-media",
        )
        raise failure
    if validated.separator != expected_separator:
        failure = CacheCompanionMismatch("cache companion separator identity differs")
        _classify_cache_failure(
            failure,
            kind="cache-companion-invalid",
            detail_code="companion-media",
        )
        raise failure
    try:
        actual_size, actual_hash = file_size_sha256(cache_path)
    except CacheCompanionError as exc:
        _classify_cache_failure(
            exc,
            kind="cache-companion-invalid",
            detail_code="companion-hash",
        )
        raise
    except OSError as exc:
        failure = CacheCompanionMismatch(f"cannot read vocals cache: {exc}")
        _classify_cache_failure(
            failure,
            kind="cache-companion-invalid",
            detail_code="companion-hash",
        )
        raise failure from exc
    if validated.cache_size != actual_size:
        failure = CacheCompanionMismatch("cache companion size differs from FLAC")
        _classify_cache_failure(
            failure,
            kind="cache-companion-invalid",
            detail_code="companion-size",
        )
        raise failure
    if validated.cache_sha256 != actual_hash:
        failure = CacheCompanionMismatch("cache companion SHA-256 differs from FLAC")
        _classify_cache_failure(
            failure,
            kind="cache-companion-invalid",
            detail_code="companion-hash",
        )
        raise failure
    return validated


def cache_pair_valid(
    companion: Mapping[str, object],
    cache_path: Path,
    *,
    media_fingerprint: str,
    separator: Mapping[str, object] | SeparatorIdentity,
) -> bool:
    try:
        validate_cache_pair(
            companion,
            cache_path,
            media_fingerprint=media_fingerprint,
            separator=separator,
        )
    except (CacheCompanionError, OSError):
        return False
    return True


def write_cache_companion(path: Path, value: Mapping[str, object]) -> None:
    validate_cache_companion(value)
    try:
        write_json_object(path, value, max_bytes=CACHE_COMPANION_MAX_BYTES)
    except OSError as exc:
        _classify_cache_failure(
            exc,
            kind="cache-operation-failed",
            detail_code="companion-replace",
        )
        raise


def publish_cache_companion(
    cache_path: Path,
    *,
    media_fingerprint: str,
    separator: Mapping[str, object] | SeparatorIdentity,
    companion_path: Path | None = None,
) -> dict[str, object]:
    """Hash the landed cache and atomically publish its fresh companion."""
    companion = build_cache_companion(
        cache_path,
        media_fingerprint=media_fingerprint,
        separator=separator,
    )
    destination = (
        cache_companion_path(cache_path)
        if companion_path is None
        else Path(companion_path)
    )
    write_cache_companion(destination, companion)
    return companion


def delete_cache_companion_first(
    cache_path: Path,
    *,
    companion_path: Path | None = None,
) -> Path:
    """Delete the old claim before any writer can replace the FLAC."""
    destination = (
        cache_companion_path(cache_path)
        if companion_path is None
        else Path(companion_path)
    )
    try:
        destination.unlink(missing_ok=True)
    except OSError as exc:
        _classify_cache_failure(
            exc,
            kind="cache-operation-failed",
            detail_code="companion-unlink",
        )
        raise
    return destination


@contextmanager
def cache_lock(cache_path: Path) -> Iterator[CacheLockHandle]:
    """Serialize every cache reader/writer through the canonical realpath lock."""
    descriptor: int | None = None
    try:
        resolved = canonical_cache_path(cache_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        lock = cache_lock_path(resolved)
        descriptor = os.open(
            lock,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            0o600,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"vocals cache lock is not a regular file: {lock}")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _classify_cache_failure(
            exc,
            kind="cache-lock-failed",
            detail_code="cache-lock-acquire",
        )
        raise
    assert descriptor is not None
    try:
        yield CacheLockHandle(
            cache_path=resolved,
            companion_path=cache_companion_path(resolved),
            lock_path=lock,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def cache_write_window(cache_path: Path) -> Iterator[CacheLockHandle]:
    """Take the cache lock and enforce delete-companion-before-write ordering."""
    with cache_lock(cache_path) as handle:
        delete_cache_companion_first(
            handle.cache_path,
            companion_path=handle.companion_path,
        )
        yield handle


@contextmanager
def cache_publish_path(cache_path: Path) -> Iterator[Path]:
    """Stage and atomically replace one cache file with exact edge terminals."""
    destination = canonical_cache_path(cache_path)
    staged = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # fsio owns the staging convention (same-directory ".<stem>.*.part<suffix>"
        # temp file, os.replace on clean exit, unlink on any failure); only the
        # cache-specific terminal classification stays here.
        with fsio.atomic_path(destination) as temporary:
            staged = True
            try:
                yield temporary
            except BaseException as exc:
                _classify_cache_failure(
                    exc,
                    kind="cache-operation-failed",
                    detail_code="cache-stage",
                )
                raise
    except BaseException as exc:
        # Classification is first-write-wins, so a body failure keeps the
        # "cache-stage" terminal attached above; only staging (no temp file yet)
        # and the replace edge reach this untouched.
        _classify_cache_failure(
            exc,
            kind="cache-operation-failed",
            detail_code="cache-replace" if staged else "cache-stage",
        )
        raise


__all__ = [
    "CacheCompanionError",
    "CacheCompanionMismatch",
    "CacheLockHandle",
    "HASH_CHUNK_BYTES",
    "SeparatorIdentity",
    "ValidatedCacheCompanion",
    "build_cache_companion",
    "cache_companion_path",
    "cache_lock",
    "cache_lock_path",
    "cache_pair_valid",
    "cache_publish_path",
    "cache_write_window",
    "canonical_cache_path",
    "classify_cache_decode_failure",
    "delete_cache_companion_first",
    "file_size_sha256",
    "load_cache_companion",
    "publish_cache_companion",
    "validate_cache_companion",
    "validate_cache_pair",
    "validate_separator_identity",
    "write_cache_companion",
]
