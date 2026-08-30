"""Per-source cache locations for machine-made episode artifacts.

The editable transcript and subtitle deliverables remain beside the media.  All
other generated episode state lives in one owner-only cache claim, while an
existing adjacent sidecar remains the read and write-back lane for compatibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

from voxweave import fsio
from voxweave.mediasnapshot import default_cache_root

_MARKER_MAX_BYTES: Final = 65_536


class ArtifactMarkerError(RuntimeError):
    """A cache claim has an unreadable or non-canonical source marker."""


class ArtifactCollisionError(ArtifactMarkerError):
    """The prescribed collision fallback belongs to another source."""


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Machine-artifact paths in one source claim."""

    source: Path
    directory: Path
    marker: Path
    speaker_mapping: Path
    speaker_suggest: Path
    voiceprints: Path
    episode_lock: Path
    vocals_cache: Path

    def translation_progress(self, subtitle: Path, target: str) -> Path:
        """Return an input-specific translation progress path."""
        encoded = quote(target, safe="-_.")
        return self.directory / f"{_stem(Path(subtitle))}.{encoded}.progress.json"

    def align_evidence(self, subtitle: Path) -> Path:
        """Return an input-specific durable alignment-evidence path."""
        return self.directory / f"{_stem(Path(subtitle))}.align-evidence.json"

    def asrfix_audit(self, subtitle: Path) -> Path:
        """Return an input-specific ASR-correction audit path."""
        return self.directory / f"{_stem(Path(subtitle))}.asrfix.json"

    @property
    def debug(self) -> Path:
        """Return the root for the cohesive optional debug bundle."""
        return self.directory / "debug"


def artifacts_root() -> Path:
    """Return the artifact root, honoring ``VOXWEAVE_CACHE_ROOT`` at call time."""
    return default_cache_root() / "artifacts"


def _ensure_private_directory(directory: Path) -> None:
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactMarkerError(
                f"artifact directory is not a private directory: {directory}"
            )
        os.chmod(directory, 0o700)
    except ArtifactMarkerError:
        raise
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot create artifact directory {directory}: {exc}"
        ) from exc


def _absolute(path: Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.realpath(os.path.abspath(os.fspath(expanded))))


def _stem(path: Path) -> str:
    return path.name[: -len(path.suffix)] if path.suffix else path.name


def _swap_ext(path: Path, suffix: str) -> Path:
    return path.with_name(f"{_stem(path)}{suffix}")


def _claim_digest(source: Path) -> str:
    return hashlib.sha1(str(source).encode(), usedforsecurity=False).hexdigest()[:8]


def _marker_text(source: Path) -> str:
    return (
        json.dumps(
            {"version": 1, "source": str(source)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate marker member {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite marker value {value}")


def _read_regular_bytes(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("marker is not a regular file")
        if metadata.st_size > _MARKER_MAX_BYTES:
            raise ValueError("marker exceeds its byte limit")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size > _MARKER_MAX_BYTES
            or metadata.st_dev != opened.st_dev
            or metadata.st_ino != opened.st_ino
            or metadata.st_size != opened.st_size
        ):
            raise ValueError("opened marker is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = opened.st_size + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        closed = os.fstat(descriptor)
        final_metadata = path.lstat()
        if (
            opened.st_dev != closed.st_dev
            or opened.st_ino != closed.st_ino
            or opened.st_size != closed.st_size
            or stat.S_ISLNK(final_metadata.st_mode)
            or not stat.S_ISREG(final_metadata.st_mode)
            or opened.st_dev != final_metadata.st_dev
            or opened.st_ino != final_metadata.st_ino
            or opened.st_size != final_metadata.st_size
            or len(encoded) != opened.st_size
        ):
            raise ValueError("marker changed while reading")
        return encoded
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_marker(marker: Path) -> Path:
    try:
        encoded = _read_regular_bytes(marker)
        raw = json.loads(
            encoded.decode(),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactMarkerError(
            f"invalid artifact source marker {marker}: {exc}"
        ) from exc
    if (
        type(raw) is not dict
        or set(raw) != {"version", "source"}
        or type(raw.get("version")) is not int
        or raw["version"] != 1
        or type(raw.get("source")) is not str
    ):
        raise ArtifactMarkerError(f"invalid artifact source marker schema: {marker}")
    source = Path(raw["source"])
    if not source.is_absolute() or source != _absolute(source):
        raise ArtifactMarkerError(f"non-normalized source in artifact marker: {marker}")
    if encoded != _marker_text(source).encode():
        raise ArtifactMarkerError(f"non-canonical artifact source marker: {marker}")
    return source


def _directory_owner(directory: Path) -> Path | None:
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot inspect artifact claim {directory}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactMarkerError(
            f"artifact claim is not a private directory: {directory}"
        )
    marker = directory / "source.json"
    try:
        marker.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot inspect artifact marker {marker}: {exc}"
        ) from exc
    return _read_marker(marker)


def _paths(source: Path, directory: Path) -> ArtifactPaths:
    return ArtifactPaths(
        source=source,
        directory=directory,
        marker=directory / "source.json",
        speaker_mapping=directory / "speakers.json",
        speaker_suggest=directory / "speakers.suggest.json",
        voiceprints=directory / "voiceprints.json",
        episode_lock=directory / f"{source.stem}.episode.lock",
        vocals_cache=directory / "vocals.32k.flac",
    )


def _claim_directory(source: Path, directory: Path) -> bool:
    try:
        _ensure_private_directory(directory)
    except ArtifactMarkerError:
        raise
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot create artifact claim {directory}: {exc}"
        ) from exc
    try:
        fsio.atomic_write_text_new(directory / "source.json", _marker_text(source))
    except FileExistsError:
        return _read_marker(directory / "source.json") == source
    return True


def claim_paths(source: Path) -> ArtifactPaths:
    """Claim the deterministic cache directory for one media/source path."""
    absolute = _absolute(Path(source))
    root = artifacts_root()
    _ensure_private_directory(root)
    primary = root / absolute.stem
    if _claim_directory(absolute, primary):
        return _paths(absolute, primary)
    fallback = artifacts_root() / f"{absolute.stem}--{_claim_digest(absolute)}"
    if _claim_directory(absolute, fallback):
        return _paths(absolute, fallback)
    owner = _read_marker(fallback / "source.json")
    raise ArtifactCollisionError(
        f"artifact fallback {fallback} belongs to {owner}, not {absolute}"
    )


def inspect_paths(source: Path) -> ArtifactPaths | None:
    """Inspect an existing matching claim without creating cache state."""
    absolute = _absolute(Path(source))
    root = artifacts_root()
    try:
        root.lstat()
    except FileNotFoundError:
        return None
    _ensure_private_directory(root)
    primary = root / absolute.stem
    owner = _directory_owner(primary)
    if owner == absolute:
        return _paths(absolute, primary)
    fallback = artifacts_root() / f"{absolute.stem}--{_claim_digest(absolute)}"
    fallback_owner = _directory_owner(fallback)
    if fallback_owner is None:
        return None
    if fallback_owner == absolute:
        return _paths(absolute, fallback)
    raise ArtifactCollisionError(
        f"artifact fallback {fallback} belongs to {fallback_owner}, not {absolute}"
    )


def claimed_sources(stem: str) -> tuple[Path, ...]:
    """Return canonical sources recorded by the closed claim set for one stem."""
    root = artifacts_root()
    try:
        root.lstat()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot inspect artifact root {root}: {exc}"
        ) from exc
    _ensure_private_directory(root)
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot inspect artifact root {root}: {exc}"
        ) from exc
    prefix = f"{stem}--"
    sources: list[Path] = []
    for entry in sorted(entries, key=lambda path: path.name):
        fallback_tail = (
            entry.name[len(prefix) :] if entry.name.startswith(prefix) else ""
        )
        if entry.name != stem and not (
            len(fallback_tail) == 8
            and all(character in "0123456789abcdef" for character in fallback_tail)
        ):
            continue
        owner = _directory_owner(entry)
        if owner is not None and owner.stem == stem:
            sources.append(owner)
    return tuple(dict.fromkeys(sources))


def episode_domain_lock_path(source: Path) -> Path:
    """Return a stable lock without consulting an unselected cache marker."""
    absolute = _absolute(Path(source))
    root = artifacts_root()
    _ensure_private_directory(root)
    lock_root = root / absolute.stem
    _ensure_private_directory(lock_root)
    domain = absolute.parent / absolute.stem
    digest = hashlib.sha256(str(domain).encode()).hexdigest()
    return lock_root / f".episode-domain-{digest}.lock"


def path_present(path: Path) -> bool:
    """Return false only when a filesystem node is truly absent."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    return True


def legacy_path(source: Path, suffix: str) -> Path:
    """Return a historical media-adjacent machine-sidecar path."""
    return _swap_ext(Path(source), suffix)


def speaker_mapping_path(source: Path, reference: Path | None = None) -> Path:
    if reference is not None:
        exact = _swap_ext(Path(reference), ".speakers.json")
        if path_present(exact):
            return exact
    legacy = legacy_path(source, ".speakers.json")
    return legacy if path_present(legacy) else claim_paths(source).speaker_mapping


def inspect_speaker_mapping_path(
    source: Path,
    reference: Path | None = None,
) -> Path:
    """Resolve a mapping for reading without claiming an empty cache directory."""
    if reference is not None:
        exact = _swap_ext(Path(reference), ".speakers.json")
        if path_present(exact):
            return exact
    legacy = legacy_path(source, ".speakers.json")
    if path_present(legacy):
        return legacy
    paths = inspect_paths(source)
    return legacy if paths is None else paths.speaker_mapping


def speaker_suggest_path(source: Path) -> Path:
    legacy = legacy_path(source, ".speakers.suggest.json")
    return legacy if path_present(legacy) else claim_paths(source).speaker_suggest


def voiceprints_path(source: Path) -> Path:
    legacy = legacy_path(source, ".voiceprints.json")
    return legacy if path_present(legacy) else claim_paths(source).voiceprints


def translation_progress_path(source: Path, subtitle: Path, target: str) -> Path:
    legacy = _swap_ext(Path(subtitle), f".{target}.progress.json")
    return (
        legacy
        if path_present(legacy)
        else claim_paths(source).translation_progress(subtitle, target)
    )


def align_evidence_path(source: Path, subtitle: Path) -> Path:
    legacy = _swap_ext(Path(subtitle), ".align-evidence.json")
    return (
        legacy if path_present(legacy) else claim_paths(source).align_evidence(subtitle)
    )


def asrfix_audit_path(source: Path, subtitle: Path) -> Path:
    legacy = _swap_ext(Path(subtitle), ".asrfix.json")
    return (
        legacy if path_present(legacy) else claim_paths(source).asrfix_audit(subtitle)
    )


def fixed_candidates(source: Path, suffix: str, attribute: str) -> tuple[Path, ...]:
    """Return deterministic legacy and cache paths for invalidation or purge."""
    legacy = legacy_path(source, suffix)
    cached = getattr(claim_paths(source), attribute)
    return tuple(dict.fromkeys((legacy, cached)))


def align_evidence_candidates(source: Path, subtitle: Path) -> tuple[Path, ...]:
    legacy = _swap_ext(Path(subtitle), ".align-evidence.json")
    cached = claim_paths(source).align_evidence(subtitle)
    return tuple(dict.fromkeys((legacy, cached)))


def translation_progress_candidates(
    source: Path,
    subtitle: Path,
    target: str,
) -> tuple[Path, ...]:
    legacy = _swap_ext(Path(subtitle), f".{target}.progress.json")
    try:
        paths = inspect_paths(source)
    except ArtifactMarkerError:
        if path_present(legacy):
            return (legacy,)
        raise
    cached = None if paths is None else paths.translation_progress(subtitle, target)
    return tuple(dict.fromkeys(path for path in (legacy, cached) if path is not None))


__all__ = [
    "ArtifactCollisionError",
    "ArtifactMarkerError",
    "ArtifactPaths",
    "align_evidence_path",
    "align_evidence_candidates",
    "artifacts_root",
    "asrfix_audit_path",
    "claim_paths",
    "claimed_sources",
    "episode_domain_lock_path",
    "fixed_candidates",
    "inspect_paths",
    "inspect_speaker_mapping_path",
    "legacy_path",
    "path_present",
    "speaker_mapping_path",
    "speaker_suggest_path",
    "translation_progress_path",
    "translation_progress_candidates",
    "voiceprints_path",
]
