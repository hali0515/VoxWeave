"""Per-source cache locations for machine-made episode artifacts.

The editable transcript and subtitle deliverables remain beside the media.  All
other generated episode state lives in one owner-only claim directory under the
media directory's own ``cache/`` root, so artifacts travel with the media when
the directory moves.  An existing adjacent sidecar remains the read and
write-back lane for compatibility.  Claim markers record only the source file
name (never an absolute path) so a relocated media directory keeps its claims.

Names derived from a stem (the claim directory itself, the episode lock and the
per-subtitle progress, evidence and audit files) keep their plain form whenever
it fits the filesystem's name limit.  Only a name that would not fit is
shortened, deterministically, by :func:`fitted_name`; an entry is found under
either form (:func:`_entry`), so the form chosen when it was written stands.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

from voxweave import fsio
from voxweave.paths import name_bytes, name_fits, name_limit, swap_ext

_MARKER_MAX_BYTES: Final = 65_536
_CACHE_DIR_NAME: Final = "cache"
# Shortened names end in "--" + this many hex digits of sha1(stem), then the suffix.
_STEM_DIGEST_CHARS: Final = 8


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
    speaker_split_undo: Path
    episode_lock: Path
    vocals_cache: Path

    def translation_progress(self, subtitle: Path, target: str) -> Path:
        """Return an input-specific translation progress path."""
        encoded = quote(target, safe="-_.")
        return _entry(
            self.directory, _stem(Path(subtitle)), f".{encoded}.progress.json"
        )

    def align_evidence(self, subtitle: Path) -> Path:
        """Return an input-specific durable alignment-evidence path."""
        return _entry(self.directory, _stem(Path(subtitle)), ".align-evidence.json")

    def asrfix_audit(self, subtitle: Path) -> Path:
        """Return an input-specific ASR-correction audit path."""
        return _entry(self.directory, _stem(Path(subtitle)), ".asrfix.json")

    @property
    def debug(self) -> Path:
        """Return the root for the cohesive optional debug bundle."""
        return self.directory / "debug"


def artifacts_root(source: Path) -> Path:
    """Return the media-adjacent artifact root for one source path."""
    return _absolute(Path(source)).parent / _CACHE_DIR_NAME


def _ensure_private_directory(directory: Path) -> None:
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactMarkerError(
                f"artifact directory is not a private directory: {directory}"
            )
    except ArtifactMarkerError:
        raise
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot create artifact directory {directory}: {exc}"
        ) from exc
    try:
        os.chmod(directory, 0o700)
    except OSError:
        # Network mounts may not honor chmod; privacy tightening is best-effort
        # there, while symlink/type checks above stay strict.
        pass


def _inspect_private_directory(directory: Path) -> bool:
    """Validate an existing artifact directory without changing its permissions."""
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot inspect artifact directory {directory}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactMarkerError(
            f"artifact directory is not a private directory: {directory}"
        )
    return True


def _absolute(path: Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.realpath(os.path.abspath(os.fspath(expanded))))


def _stem(path: Path) -> str:
    return swap_ext(path, "").name


def _claim_digest(source: Path) -> str:
    return hashlib.sha1(source.name.encode(), usedforsecurity=False).hexdigest()[:8]


def _stem_digest(stem: str) -> str:
    digest = hashlib.sha1(os.fsencode(stem), usedforsecurity=False).hexdigest()
    return digest[:_STEM_DIGEST_CHARS]


def fitted_name(directory: Path, stem: str, suffix: str = "") -> str:
    """Name a new cache entry ``stem + suffix`` inside ``directory``.

    The plain name is kept whenever it fits the filesystem's limit
    (``PC_NAME_MAX`` in UTF-8 bytes, 255 when unknown). Only a name that would
    not fit becomes ``<stem cut to fit>--<sha1(stem)[:8]><suffix>``: the suffix
    stays whole and the stem loses whole characters from its end, so the result
    is deterministic and never ends in a broken multi-byte character.
    """
    name = f"{stem}{suffix}"
    limit = name_limit(directory)
    if name_bytes(name) <= limit:
        return name
    tail = f"--{_stem_digest(stem)}{suffix}"
    budget = limit - name_bytes(tail)
    prefix = stem
    while prefix and name_bytes(prefix) > budget:
        prefix = prefix[:-1]
    return f"{prefix}{tail}"


def _is_shortened_form(name: str, stem: str, suffix: str) -> bool:
    """Whether ``name`` is :func:`fitted_name`'s shortened form of ``stem + suffix``.

    The length of the kept stem prefix depends on the name limit it was made
    under, so any prefix of the stem is accepted.
    """
    tail = f"--{_stem_digest(stem)}{suffix}"
    return name.endswith(tail) and stem.startswith(name[: -len(tail)])


def _entry(directory: Path, stem: str, suffix: str = "") -> Path:
    """Locate the cache entry ``stem + suffix`` inside ``directory``.

    An entry that already exists is used under whichever form it was written:
    the plain name, the shortened one for this filesystem, or a shortened one
    made under another name limit (a directory moved between filesystems).
    Otherwise the entry gets :func:`fitted_name`.
    """
    plain = directory / f"{stem}{suffix}"
    try:
        if path_present(plain):
            return plain
        fitted = directory / fitted_name(directory, stem, suffix)
        if fitted != plain and path_present(fitted):
            return fitted
        try:
            names = sorted(entry.name for entry in os.scandir(directory))
        except (FileNotFoundError, NotADirectoryError):
            return fitted
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot inspect artifact directory {directory}: {exc}"
        ) from exc
    for name in names:
        if _is_shortened_form(name, stem, suffix):
            return directory / name
    return fitted


def _marker_text(source_name: str) -> str:
    return (
        json.dumps(
            {"version": 2, "source": source_name},
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


def _read_marker(marker: Path) -> str:
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
        or raw["version"] != 2
        or type(raw.get("source")) is not str
    ):
        raise ArtifactMarkerError(f"invalid artifact source marker schema: {marker}")
    name = raw["source"]
    if (
        not name
        or name in (os.curdir, os.pardir)
        or "/" in name
        or (os.sep != "/" and os.sep in name)
        or (os.altsep is not None and os.altsep in name)
        or name != Path(name).name
    ):
        raise ArtifactMarkerError(f"non-name source in artifact marker: {marker}")
    if encoded != _marker_text(name).encode():
        raise ArtifactMarkerError(f"non-canonical artifact source marker: {marker}")
    return name


def _directory_owner(directory: Path) -> str | None:
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
        speaker_split_undo=directory / "speaker-split.undo.json",
        episode_lock=_entry(directory, source.stem, ".episode.lock"),
        vocals_cache=directory / "vocals.32k.flac",
    )


def _claim_directory(source: Path, directory: Path) -> bool:
    # Raises ArtifactMarkerError for every OSError itself.
    _ensure_private_directory(directory)
    try:
        fsio.atomic_write_text_new(
            directory / "source.json", _marker_text(source.name), private=True
        )
    except FileExistsError:
        return _read_marker(directory / "source.json") == source.name
    return True


def claim_paths(source: Path) -> ArtifactPaths:
    """Claim the deterministic cache directory for one media/source path."""
    absolute = _absolute(Path(source))
    root = artifacts_root(absolute)
    _ensure_private_directory(root)
    primary = _entry(root, absolute.stem)
    if _claim_directory(absolute, primary):
        return _paths(absolute, primary)
    fallback = _entry(root, absolute.stem, f"--{_claim_digest(absolute)}")
    if _claim_directory(absolute, fallback):
        return _paths(absolute, fallback)
    owner = _read_marker(fallback / "source.json")
    raise ArtifactCollisionError(
        f"artifact fallback {fallback} belongs to {owner}, not {absolute}"
    )


def inspect_paths(source: Path) -> ArtifactPaths | None:
    """Inspect an existing matching claim without creating cache state."""
    absolute = _absolute(Path(source))
    root = artifacts_root(absolute)
    if not _inspect_private_directory(root):
        return None
    primary = _entry(root, absolute.stem)
    owner = _directory_owner(primary)
    if owner == absolute.name:
        return _paths(absolute, primary)
    fallback = _entry(root, absolute.stem, f"--{_claim_digest(absolute)}")
    fallback_owner = _directory_owner(fallback)
    if fallback_owner is None:
        return None
    if fallback_owner == absolute.name:
        return _paths(absolute, fallback)
    raise ArtifactCollisionError(
        f"artifact fallback {fallback} belongs to {fallback_owner}, not {absolute}"
    )


def _is_claim_name(name: str, stem: str) -> bool:
    """Whether ``name`` is a claim directory for ``stem``, in either form.

    That is the primary claim (``<stem>``) or a collision fallback
    (``<stem>--<8 hex>``), each plain or shortened by :func:`fitted_name`.
    """
    if name == stem or _is_shortened_form(name, stem, ""):
        return True
    fallback_tail = name[-10:]
    return (
        len(fallback_tail) == 10
        and fallback_tail.startswith("--")
        and all(character in "0123456789abcdef" for character in fallback_tail[2:])
        and (
            name == f"{stem}{fallback_tail}"
            or _is_shortened_form(name, stem, fallback_tail)
        )
    )


def claimed_sources(directory: Path, stem: str) -> tuple[Path, ...]:
    """Return sources recorded by one media directory's closed claim set."""
    parent = _absolute(Path(directory))
    root = parent / _CACHE_DIR_NAME
    if not _inspect_private_directory(root):
        return ()
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise ArtifactMarkerError(
            f"cannot inspect artifact root {root}: {exc}"
        ) from exc
    sources: list[Path] = []
    for entry in sorted(entries, key=lambda path: path.name):
        if not _is_claim_name(entry.name, stem):
            continue
        owner = _directory_owner(entry)
        if owner is not None and Path(owner).stem == stem:
            sources.append(parent / owner)
    return tuple(dict.fromkeys(sources))


def episode_domain_lock_path(source: Path) -> Path:
    """Return a stable lock without consulting an unselected cache marker."""
    absolute = _absolute(Path(source))
    root = artifacts_root(absolute)
    _ensure_private_directory(root)
    lock_root = _entry(root, absolute.stem)
    _ensure_private_directory(lock_root)
    # The media directory itself scopes the domain, and same-stem siblings
    # (episode.mp4 + episode.mp3) share sibling files, so they share one lock.
    return lock_root / ".episode-domain.lock"


def path_present(path: Path) -> bool:
    """Return false only when a filesystem node is truly absent."""
    try:
        Path(path).lstat()
    except OSError as exc:
        # A name too long to exist (ENAMETOOLONG) is absent too: a long media stem
        # plus a legacy sidecar suffix can exceed NAME_MAX.
        if exc.errno in (errno.ENOENT, errno.ENAMETOOLONG):
            return False
        raise
    return True


def legacy_path(source: Path, suffix: str) -> Path:
    """Return a historical media-adjacent machine-sidecar path."""
    return swap_ext(Path(source), suffix)


def speaker_mapping_path(source: Path, reference: Path | None = None) -> Path:
    if reference is not None:
        exact = swap_ext(Path(reference), ".speakers.json")
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
        exact = swap_ext(Path(reference), ".speakers.json")
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


def speaker_split_undo_path(source: Path) -> Path:
    """Return the cache-owned, single-level speaker-split undo snapshot."""
    return claim_paths(source).speaker_split_undo


def translation_progress_path(source: Path, subtitle: Path, target: str) -> Path:
    legacy = swap_ext(Path(subtitle), f".{target}.progress.json")
    return (
        legacy
        if path_present(legacy)
        else claim_paths(source).translation_progress(subtitle, target)
    )


def align_evidence_path(source: Path, subtitle: Path) -> Path:
    legacy = swap_ext(Path(subtitle), ".align-evidence.json")
    return (
        legacy if path_present(legacy) else claim_paths(source).align_evidence(subtitle)
    )


def asrfix_audit_path(source: Path, subtitle: Path) -> Path:
    legacy = swap_ext(Path(subtitle), ".asrfix.json")
    return (
        legacy if path_present(legacy) else claim_paths(source).asrfix_audit(subtitle)
    )


def _candidates(legacy: Path, cached: Path | None) -> tuple[Path, ...]:
    # A legacy name too long for its filesystem cannot exist (a long media stem
    # plus a sidecar suffix), so there is nothing there to invalidate or purge.
    kept = (legacy,) if name_fits(legacy) else ()
    return tuple(dict.fromkeys((*kept, *(() if cached is None else (cached,)))))


def fixed_candidates(source: Path, suffix: str, attribute: str) -> tuple[Path, ...]:
    """Return deterministic legacy and cache paths for invalidation or purge."""
    legacy = legacy_path(source, suffix)
    cached = getattr(claim_paths(source), attribute)
    return _candidates(legacy, cached)


def align_evidence_candidates(source: Path, subtitle: Path) -> tuple[Path, ...]:
    legacy = swap_ext(Path(subtitle), ".align-evidence.json")
    cached = claim_paths(source).align_evidence(subtitle)
    return _candidates(legacy, cached)


def translation_progress_candidates(
    source: Path,
    subtitle: Path,
    target: str,
) -> tuple[Path, ...]:
    legacy = swap_ext(Path(subtitle), f".{target}.progress.json")
    try:
        paths = inspect_paths(source)
    except ArtifactMarkerError:
        if path_present(legacy):
            return (legacy,)
        raise
    cached = None if paths is None else paths.translation_progress(subtitle, target)
    return _candidates(legacy, cached)


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
    "fitted_name",
    "fixed_candidates",
    "inspect_paths",
    "inspect_speaker_mapping_path",
    "legacy_path",
    "path_present",
    "speaker_mapping_path",
    "speaker_split_undo_path",
    "speaker_suggest_path",
    "translation_progress_path",
    "translation_progress_candidates",
    "voiceprints_path",
]
