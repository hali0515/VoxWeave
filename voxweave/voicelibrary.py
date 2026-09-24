"""Global voice library: named voices shared across media folders and machines.

Layout (plain JSON, never SQLite: the library may live on an NFS share, where
SQLite's locking and WAL are unsafe)::

    <voices_dir>/
      identities.json               identities shared by every embedding space
      spaces/<model>-<fp12>.json    the vectors of one embedding space
      history.jsonl                 audit log; never holds vectors or names
      .library.lock                 one library-wide flock

An identity id is a random ``v`` + 12 hex key; display names are attributes,
never keys (two "Alex" may be two people). Identities are shared by every
space, so a rename applies everywhere, while vectors live per space: a space
file is named by the embedding model and the first 12 hex digits of the
compatibility fingerprint, and freezes the provenance that defines it. An
unresolved fingerprint cannot be stored (the phase-2 compatibility law).

Every exemplar records the scope (show or folder) it was enrolled under and a
pointer to its source media, so a future re-embedding can find the media
again; audio is never stored.

Writers hold the exclusive lock, re-read every file they change, apply a pure
transition, then replace each changed file atomically, checking just before
each rename that its bytes are still the ones they read, and append the
history row last. Readers hold the shared lock. The check is a read followed
by a rename, not an atomic compare-and-swap: it catches most writers that
raced past a lock that does nothing (a ``nolock`` NFS mount), but two such
writers can both pass it, so correctness relies on the lock. What such a race
(or a manual edit) can leave behind, vectors of an identity that no longer
exists, is skipped by every reader and deleted by the next writer.

``identities.json`` lists every space file, written before a new space file
is created, so a reader that must see every space (``forget``) does not
depend on a directory listing, which an NFS client may serve from a stale
cache for a minute. Nothing relies on inode identity or hard links, both of
which are unreliable on network filesystems; a rename within one directory
is atomic on the server.
"""

from __future__ import annotations

import copy
import dataclasses
import errno
import fcntl
import json
import logging
import os
import re
import secrets
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from voxweave import config, fsio
from voxweave.voicebase import (
    IDENTITY_ID_RE,
    MAX_PROVENANCE_STRING_BYTES,
    MAX_SIDECAR_LABEL_BYTES,
    Phase2DataError,
    canonical_json_bytes,
    canonical_json_digest,
    encode_json_bytes,
    require_capture_id,
    require_exact_int,
    require_exemplar_id,
    require_identity_id,
    require_mapping,
    require_sha256,
    require_string,
    require_utc_timestamp,
    require_version_one,
    strict_json_object_loads,
    utc_timestamp,
    validate_provenance,
    validate_vector,
)
from voxweave.voiceembed import LANE_DECOUPLED
from voxweave.voicematch import (
    CompatibilityError,
    CompatibilityUnknown,
    build_compatibility_fingerprint,
)
from voxweave.voicestore import (
    MAX_ALIASES,
    MAX_EXEMPLARS,
    MAX_NAME_BYTES,
    CrossIndexRefusal,
    EnrollmentRefusal,
    ExemplarKey,
    canonical_store_path,
    load_voice_store,
    normalize_episode,
    normalize_speaker_key,
    plan_indexed_enrollment,
    store_lock_path,
    validate_voice_store,
)

log = logging.getLogger("voxweave")

ENV_VOICES_DIR = "VOXWEAVE_VOICES_DIR"
IDENTITIES_NAME = "identities.json"
SPACES_DIRNAME = "spaces"
HISTORY_NAME = "history.jsonl"
LOCK_NAME = ".library.lock"
# The pre-library per-directory store, still read (never written) for matching.
LEGACY_STORE_NAME = "voxweave.voices.json"

IDENTITIES_MAX_BYTES = 16 * 1024 * 1024
SPACE_MAX_BYTES = 64 * 1024 * 1024
MAX_LIBRARY_IDENTITIES = 10_000
# Tombstones of forgotten ids; past the cap the oldest are dropped.
MAX_FORGOTTEN = 100_000
MAX_SPACES = 1024
MAX_SCOPES = 256
MAX_MEDIA_PATH_BYTES = 4096
MAX_SLUG_CHARS = 48
DEFAULT_SCOPE = "default"
HISTORY_ACTIONS = frozenset(
    {
        "create",
        "enroll",
        "replace",
        "evict",
        "rename",
        "forget",
        "import",
        "split",
        "orphan",
        "rescope",
    }
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SPACE_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{12}$")

FIRST_WRITE_NOTICE = (
    "voice library created at %s: it stores voice biometrics of the people you "
    "name; `voxweave voices forget ID` removes one person, deleting the "
    "directory removes everyone"
)


class VoiceLibraryError(Phase2DataError):
    """The voice library violates its schema or a relational invariant."""


class LibraryConflict(RuntimeError):
    """A library file changed underneath a writer that held the library lock."""


class UnknownIdentity(LookupError):
    """No identity has the requested id."""


# --------------------------------------------------------------------------
# Location
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryLocation:
    """The resolved library directory and the precedence layer that chose it.

    ``default`` marks the built-in location, the only one whose missing
    parent directories a writer creates: a configured location may sit on a
    network share, and creating its parents under an unmounted mount point
    would start a second, local library that the share later hides.
    """

    root: Path
    source: str
    default: bool = False


def default_voices_dir(environ: Mapping[str, str] | None = None) -> LibraryLocation:
    """``$XDG_DATA_HOME/voxweave/voices``, else ``~/.local/share/voxweave/voices``.

    A data directory rather than the cache: the library is curated knowledge
    that must outlive the media it was learned from. The same rule applies on
    every platform (like the fixed ``~/.config`` and ``~/.cache`` roots); a
    relative ``XDG_DATA_HOME`` is invalid per the XDG spec and ignored.
    """
    values = os.environ if environ is None else environ
    xdg = values.get("XDG_DATA_HOME", "").strip()
    if xdg and Path(xdg).is_absolute():
        return LibraryLocation(
            Path(xdg) / "voxweave" / "voices",
            "built-in default ($XDG_DATA_HOME)",
            default=True,
        )
    return LibraryLocation(
        Path.home() / ".local" / "share" / "voxweave" / "voices",
        "built-in default",
        default=True,
    )


def resolve_voices_dir(cli_value: Path | str | None = None) -> LibraryLocation:
    """Resolve the library directory: CLI > env > config > built-in default."""
    if cli_value is not None and str(cli_value).strip():
        return LibraryLocation(
            Path(str(cli_value).strip()).expanduser().absolute(), "--voices-dir"
        )
    env = os.environ.get(ENV_VOICES_DIR)
    if env is not None and env.strip():
        return LibraryLocation(
            Path(env.strip()).expanduser().absolute(), f"environment {ENV_VOICES_DIR}"
        )
    conf = config.conf_voices_dir()
    if conf is not None:
        return LibraryLocation(conf, f"config [voices].dir ({config.config_path()})")
    return default_voices_dir()


@dataclass(frozen=True)
class LibraryPaths:
    root: Path

    @property
    def identities(self) -> Path:
        return self.root / IDENTITIES_NAME

    @property
    def spaces_dir(self) -> Path:
        return self.root / SPACES_DIRNAME

    @property
    def history(self) -> Path:
        return self.root / HISTORY_NAME

    @property
    def lock(self) -> Path:
        return self.root / LOCK_NAME

    def space(self, name: str) -> Path:
        return self.spaces_dir / f"{name}.json"


def legacy_store_path(media: Path) -> Path:
    """Where a pre-library per-directory store for ``media`` would be."""
    return Path(os.path.abspath(media)).parent / LEGACY_STORE_NAME


def read_legacy_store(path: Path) -> tuple[Path, dict[str, object]]:
    """Read and validate a per-show store without needing write access to it.

    The store's shared lock is taken when its lock file can be opened for
    writing (created if missing, as voicestore does); in a read-only place (a
    snapshot, a read-only mount or copy) an existing lock file is opened
    read-only, and without one the store is read unlocked, which is safe
    because nothing can write there either. Returns the resolved path.
    """
    resolved = canonical_store_path(path)
    lock_path = store_lock_path(path)
    descriptor: int | None
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
        try:
            descriptor = os.open(lock_path, os.O_RDONLY)
        except OSError:
            descriptor = None
    try:
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
        store, _validated = load_voice_store(resolved)
        return resolved, store
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


# --------------------------------------------------------------------------
# Scopes and embedding spaces
# --------------------------------------------------------------------------


def normalize_scope(raw: object) -> str:
    """Scopes use the speaker-name normalization: NFC(sanitize), no casefold."""
    return normalize_speaker_key(raw, field="scope")


# Folder names that many shows share: a season, disc, part or year number, or
# a bonus-material folder. They only name a scope together with their parent.
_NUMBERED_FOLDER_RE = re.compile(
    r"(?:(?:season|series|staffel|saison|temporada|stagione|seizoen|s|disc|disk|"
    r"cd|dvd|bd|part|pt|vol|volume|book|cour)[\s._-]*)?\d+",
    re.IGNORECASE,
)
_BONUS_FOLDER_RE = re.compile(
    r"specials?|extras?|bonus|featurettes?|ovas?|sp", re.IGNORECASE
)
# A CJK season or part number: DI (U+7B2C), a number in ASCII, full-width or
# CJK digits, then season, period or part (U+5B63, U+671F, U+90E8).
_CJK_SEASON_RE = re.compile(
    "\u7b2c\\s*[0-9\uff10-\uff19"
    "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+"
    "\\s*[\u5b63\u671f\u90e8]"
)


def is_generic_folder_name(name: str) -> bool:
    """Whether a folder name (``Season 1``, ``S02``, ``Disc 1``, ``2023``,
    ``Specials``, a CJK season number) is shared by many shows."""
    text = name.strip()
    return any(
        pattern.fullmatch(text) is not None
        for pattern in (_NUMBERED_FOLDER_RE, _BONUS_FOLDER_RE, _CJK_SEASON_RE)
    )


def episode_scope(media: Path, show: str | None = None) -> str:
    """``--show`` when given, else the name of the media's directory.

    A generic directory name (``Season 1``, ``Disc 2``, ``Specials``) is
    qualified with its parent's (``Frieren / Season 1``): a bare ``Season 1``
    would put every show's first season in one scope, where a typed name
    alone links identities and episode labels collide.
    """
    if show is not None:
        return normalize_scope(show)
    folder = Path(os.path.abspath(media)).parent
    name = folder.name
    if is_generic_folder_name(name) and folder.parent.name:
        try:
            return normalize_scope(f"{folder.parent.name} / {name}")
        except Phase2DataError:
            pass  # too long together: fall back to the folder's own name
    try:
        return normalize_scope(name)
    except Phase2DataError:
        return DEFAULT_SCOPE


def space_model_slug(provenance: Mapping[str, object]) -> str:
    """The ``<model>`` part of a space file name (legacy lane -> ``pyannote``)."""
    if "embedding_lane" not in provenance:
        return "pyannote"
    lane = provenance.get("embedding_lane")
    if lane != LANE_DECOUPLED:
        raise VoiceLibraryError(f"unknown embedding lane {lane!r}")
    model = require_string(
        provenance.get("embedding_model"),
        "embedding_model",
        max_bytes=MAX_PROVENANCE_STRING_BYTES,
    )
    slug = _SLUG_RE.sub("-", model.lower()).strip("-")[:MAX_SLUG_CHARS].strip("-")
    return slug or "model"


def space_identity(provenance: Mapping[str, object]) -> tuple[str, str]:
    """Return ``(space file stem, full compatibility fingerprint)``."""
    try:
        result = build_compatibility_fingerprint(provenance)
    except CompatibilityError as exc:
        raise VoiceLibraryError(f"embedding provenance is malformed: {exc}") from exc
    if isinstance(result, CompatibilityUnknown):
        fields = ", ".join(result.unresolved_fields)
        raise VoiceLibraryError(
            f"the embedding space is unresolved ({fields}); its voices cannot be "
            "stored in or matched against the voice library"
        )
    return f"{space_model_slug(provenance)}-{result.value[:12]}", result.value


# --------------------------------------------------------------------------
# Documents and validation
# --------------------------------------------------------------------------


def empty_identities() -> dict[str, object]:
    return {
        "version": 1,
        "revision": 0,
        "identities": {},
        "spaces": [],
        "forgotten": [],
    }


def listed_spaces(identities: Mapping[str, object]) -> list[str]:
    """The space names ``identities.json`` lists (absent in early libraries)."""
    return list(cast(list[str], identities.get("spaces", [])))


def forgotten_ids(identities: Mapping[str, object]) -> list[str]:
    """Ids removed by ``voices forget``, oldest first: id-only tombstones.

    A forgotten id is never offered, imported or adopted again, although a
    per-folder store from an earlier version may still hold its vectors.
    """
    return list(cast(list[str], identities.get("forgotten", [])))


def _list_space(identities: dict[str, object], space_name: str) -> bool:
    """List ``space_name`` in ``identities`` (in place); report whether it changed."""
    listed = listed_spaces(identities)
    if space_name in listed:
        return False
    identities["spaces"] = sorted([*listed, space_name])
    return True


def new_space(provenance: Mapping[str, object]) -> dict[str, object]:
    _name, fingerprint = space_identity(provenance)
    return {
        "version": 1,
        "revision": 0,
        "space": {
            "fingerprint": fingerprint,
            "provenance": copy.deepcopy(dict(provenance)),
        },
        "exemplars": {},
    }


def _as_library_error(exc: Phase2DataError, source: str) -> VoiceLibraryError:
    if isinstance(exc, VoiceLibraryError):
        return exc
    return VoiceLibraryError(f"{source}: {exc}")


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise VoiceLibraryError(f"{field} must be an array")
    return cast(list[object], value)


def validate_identities(value: object) -> Mapping[str, Mapping[str, object]]:
    """Validate ``identities.json``; return its identities mapping."""
    try:
        root = require_mapping(value, "identities document")
        require_version_one(root.get("version"))
        require_exact_int(root.get("revision"), "revision", minimum=0)
        identities = require_mapping(root.get("identities"), "identities")
        if len(identities) > MAX_LIBRARY_IDENTITIES:
            raise VoiceLibraryError(
                f"identities may contain at most {MAX_LIBRARY_IDENTITIES} entries"
            )
        for identity_id, raw in identities.items():
            require_identity_id(identity_id)
            prefix = f"identities.{identity_id}"
            identity = require_mapping(raw, prefix)
            display = require_string(
                identity.get("display_name"),
                f"{prefix}.display_name",
                max_bytes=MAX_NAME_BYTES,
            )
            normalize_speaker_key(display, field=f"{prefix}.display_name")
            aliases = _require_list(identity.get("aliases"), f"{prefix}.aliases")
            if len(aliases) > MAX_ALIASES:
                raise VoiceLibraryError(f"{prefix}.aliases exceeds {MAX_ALIASES}")
            for position, alias in enumerate(aliases):
                normalize_speaker_key(alias, field=f"{prefix}.aliases[{position}]")
            scopes = _require_list(identity.get("scopes"), f"{prefix}.scopes")
            if len(scopes) > MAX_SCOPES:
                raise VoiceLibraryError(f"{prefix}.scopes exceeds {MAX_SCOPES}")
            for position, scope in enumerate(scopes):
                if normalize_scope(scope) != scope:
                    raise VoiceLibraryError(
                        f"{prefix}.scopes[{position}] is not a normalized scope"
                    )
            if len(set(cast(list[str], scopes))) != len(scopes):
                raise VoiceLibraryError(f"{prefix}.scopes contains duplicates")
            require_utc_timestamp(identity.get("created"), f"{prefix}.created")
            require_utc_timestamp(identity.get("updated"), f"{prefix}.updated")
        if "spaces" in root:
            spaces = _require_list(root.get("spaces"), "spaces")
            if len(spaces) > MAX_SPACES:
                raise VoiceLibraryError(f"spaces may list at most {MAX_SPACES} files")
            for position, name in enumerate(spaces):
                if not isinstance(name, str) or _SPACE_NAME_RE.fullmatch(name) is None:
                    raise VoiceLibraryError(f"spaces[{position}] is not a space name")
            if len(set(cast(list[str], spaces))) != len(spaces):
                raise VoiceLibraryError("spaces contains duplicates")
        if "forgotten" in root:
            forgotten = _require_list(root.get("forgotten"), "forgotten")
            if len(forgotten) > MAX_FORGOTTEN:
                raise VoiceLibraryError(
                    f"forgotten may list at most {MAX_FORGOTTEN} ids"
                )
            for position, identity_id in enumerate(forgotten):
                require_identity_id(identity_id, f"forgotten[{position}]")
                if identity_id in identities:
                    raise VoiceLibraryError(
                        f"forgotten identity {identity_id} is still in identities"
                    )
            if len(set(cast(list[str], forgotten))) != len(forgotten):
                raise VoiceLibraryError("forgotten contains duplicates")
    except Phase2DataError as exc:
        raise _as_library_error(exc, IDENTITIES_NAME) from exc
    return cast(Mapping[str, Mapping[str, object]], identities)


@dataclass(frozen=True)
class ValidatedSpace:
    name: str
    fingerprint: str
    revision: int
    embedding_dim: int
    provenance: Mapping[str, object]
    exemplars: Mapping[str, Sequence[Mapping[str, object]]]


def _validate_source(value: object, field: str) -> None:
    source = require_mapping(value, field)
    media_path = source.get("media_path")
    if media_path is not None:
        require_string(
            media_path, f"{field}.media_path", max_bytes=MAX_MEDIA_PATH_BYTES
        )
    require_sha256(source.get("media_fingerprint"), f"{field}.media_fingerprint")
    require_capture_id(source.get("capture_id"), f"{field}.capture_id")
    turns_digest = source.get("turns_digest")
    if turns_digest is not None:
        require_sha256(turns_digest, f"{field}.turns_digest")
    label = source.get("speaker_label")
    if label is not None:
        require_string(
            label, f"{field}.speaker_label", max_bytes=MAX_SIDECAR_LABEL_BYTES
        )
    normalize_episode(source.get("episode"))


def validate_space(value: object, *, name: str) -> ValidatedSpace:
    """Validate one ``spaces/<name>.json`` document, including its file name."""
    try:
        root = require_mapping(value, "space document")
        require_version_one(root.get("version"))
        revision = require_exact_int(root.get("revision"), "revision", minimum=0)
        space = require_mapping(root.get("space"), "space")
        fingerprint = require_sha256(space.get("fingerprint"), "space.fingerprint")
        provenance, dim = validate_provenance(space.get("provenance"))
        expected_name, expected_fingerprint = space_identity(provenance)
        if expected_fingerprint != fingerprint:
            raise VoiceLibraryError(
                "space.fingerprint does not match its frozen provenance"
            )
        if expected_name != name:
            raise VoiceLibraryError(
                f"space file {name}.json holds space {expected_name}"
            )
        exemplars = require_mapping(root.get("exemplars"), "exemplars")
        seen_ids: set[str] = set()
        for identity_id, raw_list in exemplars.items():
            require_identity_id(identity_id, f"exemplars.{identity_id}")
            items = _require_list(raw_list, f"exemplars.{identity_id}")
            if not items:
                raise VoiceLibraryError(f"exemplars.{identity_id} must not be empty")
            if len(items) > MAX_EXEMPLARS:
                raise VoiceLibraryError(
                    f"exemplars.{identity_id} exceeds {MAX_EXEMPLARS}"
                )
            captures: set[str] = set()
            media: set[str] = set()
            episodes: set[str] = set()
            for index, raw in enumerate(items):
                field_name = f"exemplars.{identity_id}[{index}]"
                exemplar = require_mapping(raw, field_name)
                exemplar_id = require_exemplar_id(
                    exemplar.get("id"), f"{field_name}.id"
                )
                if exemplar_id in seen_ids:
                    raise VoiceLibraryError(f"duplicate exemplar id {exemplar_id}")
                seen_ids.add(exemplar_id)
                validate_vector(
                    exemplar.get("vector"), dim=dim, field=f"{field_name}.vector"
                )
                scope = exemplar.get("scope")
                if normalize_scope(scope) != scope:
                    raise VoiceLibraryError(
                        f"{field_name}.scope is not a normalized scope"
                    )
                _validate_source(exemplar.get("source"), f"{field_name}.source")
                require_utc_timestamp(exemplar.get("added"), f"{field_name}.added")
                source = cast(Mapping[str, object], exemplar["source"])
                capture = cast(str, source["capture_id"])
                media_hash = cast(str, source["media_fingerprint"])
                episode = scoped_episode(
                    cast(str, scope), normalize_episode(source["episode"])
                )
                if capture in captures or media_hash in media or episode in episodes:
                    raise VoiceLibraryError(
                        f"exemplars.{identity_id} repeats a capture, source media "
                        "or episode of one scope"
                    )
                captures.add(capture)
                media.add(media_hash)
                episodes.add(episode)
    except Phase2DataError as exc:
        raise _as_library_error(exc, f"{SPACES_DIRNAME}/{name}.json") from exc
    return ValidatedSpace(
        name=name,
        fingerprint=fingerprint,
        revision=revision,
        embedding_dim=dim,
        provenance=provenance,
        exemplars=cast(Mapping[str, Sequence[Mapping[str, object]]], exemplars),
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass
class LibraryState:
    """Validated library documents plus the exact bytes they were read from.

    ``spaces`` holds only the spaces that were requested and exist
    (``all_spaces`` says whether every space file was loaded); ``observed``
    maps every file that was looked at to its bytes (None when absent), the
    compare-and-swap baseline of a later :func:`commit`.

    ``orphans`` counts, per space, the exemplars of identities that do not
    exist, which were dropped from ``spaces`` on read (the next commit
    deletes them from disk); ``missing_spaces`` names spaces that
    ``identities.json`` lists but that could not be found.
    """

    paths: LibraryPaths
    identities: dict[str, object]
    spaces: dict[str, dict[str, object]]
    observed: dict[Path, bytes | None] = field(default_factory=dict)
    all_spaces: bool = False
    orphans: dict[str, dict[str, int]] = field(default_factory=dict)
    missing_spaces: tuple[str, ...] = ()

    @property
    def identity_map(self) -> dict[str, dict[str, object]]:
        return cast(dict[str, dict[str, object]], self.identities["identities"])

    @property
    def forgotten(self) -> frozenset[str]:
        return frozenset(forgotten_ids(self.identities))

    def space_exemplars(self, name: str) -> dict[str, list[dict[str, object]]]:
        space = self.spaces.get(name)
        if space is None:
            return {}
        return cast(dict[str, list[dict[str, object]]], space["exemplars"])


def list_space_names(paths: LibraryPaths) -> list[str]:
    """Names of the space files present (temp files and strays ignored)."""
    try:
        entries = sorted(os.listdir(paths.spaces_dir))
    except FileNotFoundError:
        return []
    names: list[str] = []
    for entry in entries:
        if entry.startswith(".") or not entry.endswith(".json"):
            continue
        stem = entry[: -len(".json")]
        if _SPACE_NAME_RE.fullmatch(stem) is None:
            log.warning("ignoring unexpected file %s in the voice library", entry)
            continue
        names.append(stem)
    return names


def _read_bounded(path: Path, max_bytes: int) -> bytes | None:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return None
    if size > max_bytes:
        raise VoiceLibraryError(f"{path} exceeds the {max_bytes}-byte limit")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    if len(raw) > max_bytes:
        raise VoiceLibraryError(f"{path} exceeds the {max_bytes}-byte limit")
    return raw


def _decode(raw: bytes, *, source: str, max_bytes: int) -> dict[str, object]:
    try:
        return strict_json_object_loads(raw, max_bytes=max_bytes, source=source)
    except Phase2DataError as exc:
        raise _as_library_error(exc, source) from exc


def validate_relations(state: LibraryState) -> None:
    """Every loaded space's exemplars belong to an existing identity."""
    identities = state.identity_map
    for name, space in state.spaces.items():
        for identity_id in cast(Mapping[str, object], space["exemplars"]):
            if identity_id not in identities:
                raise VoiceLibraryError(
                    f"{SPACES_DIRNAME}/{name}.json holds exemplars of unknown "
                    f"identity {identity_id}"
                )


def _drop_orphans(
    identities: Mapping[str, object], spaces: Mapping[str, dict[str, object]]
) -> dict[str, dict[str, int]]:
    """Remove, in place, the exemplars of identities that do not exist.

    They can only be left by an interrupted or unlocked writer (or a manual
    edit); refusing to read the library over them would also block the
    ``forget`` that removes them, so readers skip them and writers delete them.
    """
    known = cast(Mapping[str, object], identities["identities"])
    dropped: dict[str, dict[str, int]] = {}
    for name, document in spaces.items():
        exemplars = cast(dict[str, list[object]], document["exemplars"])
        for identity_id in sorted(set(exemplars) - set(known)):
            dropped.setdefault(name, {})[identity_id] = len(exemplars.pop(identity_id))
        if name in dropped:
            log.warning(
                "voice library space %s holds voice samples of %d unknown "
                "identities (left by an interrupted or unlocked writer); they are "
                "ignored, and the next change to the library deletes them",
                name,
                len(dropped[name]),
            )
    return dropped


def read_state(
    root: Path,
    *,
    spaces: Iterable[str] | None = None,
    include: Iterable[str] = (),
) -> LibraryState:
    """Read and validate the library; ``spaces=None`` loads every space.

    Every space means every space ``identities.json`` lists, every space file
    found in the directory, and ``include`` (spaces a writer may create).
    Call it while holding :func:`library_lock`. A missing library reads as an
    empty one.
    """
    paths = LibraryPaths(Path(root))
    observed: dict[Path, bytes | None] = {}
    raw = _read_bounded(paths.identities, IDENTITIES_MAX_BYTES)
    observed[paths.identities] = raw
    identities = (
        empty_identities()
        if raw is None
        else _decode(raw, source=IDENTITIES_NAME, max_bytes=IDENTITIES_MAX_BYTES)
    )
    validate_identities(identities)
    listed = listed_spaces(identities)
    if spaces is None:
        names = sorted({*listed, *list_space_names(paths), *include})
    else:
        names = list(dict.fromkeys([*spaces, *include]))
    loaded: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for name in names:
        path = paths.space(name)
        raw_space = _read_bounded(path, SPACE_MAX_BYTES)
        observed[path] = raw_space
        if raw_space is None:
            if name in listed:
                missing.append(name)
            continue
        document = _decode(
            raw_space, source=f"{SPACES_DIRNAME}/{name}.json", max_bytes=SPACE_MAX_BYTES
        )
        validate_space(document, name=name)
        loaded[name] = document
    if missing:
        log.warning(
            "voice library %s lists space file(s) that cannot be found: %s",
            paths.root,
            ", ".join(missing),
        )
    orphans = _drop_orphans(identities, loaded)
    state = LibraryState(
        paths,
        identities,
        loaded,
        observed,
        spaces is None,
        orphans=orphans,
        missing_spaces=tuple(missing),
    )
    validate_relations(state)
    return state


# --------------------------------------------------------------------------
# Locking and committing
# --------------------------------------------------------------------------


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        # Some NAS exports refuse chmod (root squash, ACL-only shares).
        pass


def _ensure_private_dir(path: Path, *, parents: bool = False) -> None:
    """Create ``path``; only a directory created here gets 0o700.

    A pre-existing directory keeps its mode, so a NAS share prepared for
    several users or machines is never narrowed behind the owner's back.
    Missing parents are created only with ``parents`` (the built-in
    location); otherwise they are refused, since a missing parent of a
    configured library usually is a network share that is not mounted.
    """
    if path.is_dir():
        return
    try:
        path.mkdir(parents=parents, exist_ok=True, mode=0o700)
    except FileNotFoundError as exc:
        raise VoiceLibraryError(
            f"cannot create the voice library {path}: its parent directory "
            f"{path.parent} does not exist (is a network share not mounted?); "
            "create the parent directory first if this location is intended"
        ) from exc
    _chmod_best_effort(path, 0o700)


@contextmanager
def library_lock(
    root: Path, *, exclusive: bool, create_parents: bool = False
) -> Iterator[bool]:
    """Hold the library-wide flock; yield whether a lock is actually held.

    ``exclusive`` creates the library directory (and, with
    ``create_parents``, its missing parents) and its 0o600 lock file. A
    shared lock never creates the directory: a missing library, or a
    read-only one that never had a lock file, is read without a lock (no
    locking writer can have written it). flock on NFSv4 (and on NFSv3 with
    NLM) is emulated with byte-range locks; a ``nolock`` mount makes it
    local-only (see :func:`commit` for what that still catches).
    """
    paths = LibraryPaths(Path(root))
    if exclusive:
        _ensure_private_dir(paths.root, parents=create_parents)
        descriptor = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
    else:
        if not paths.root.is_dir():
            yield False
            return
        try:
            descriptor = os.open(paths.lock, os.O_RDONLY)
        except FileNotFoundError:
            try:
                descriptor = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
            except OSError:
                descriptor = None
        if descriptor is None:
            yield False
            return
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield True
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class LibraryChange:
    """New documents for the files a transition changed, plus history rows.

    ``identities_first`` orders the writes so that every crash point leaves
    a valid library: additions write identities before spaces (an identity
    without vectors is valid), removals write spaces before identities
    (vectors of a missing identity would not be).
    """

    identities: dict[str, object] | None = None
    spaces: Mapping[str, dict[str, object]] = field(default_factory=dict)
    history: Sequence[Mapping[str, object]] = ()
    identities_first: bool = True
    # Strip the labels (scopes, episodes) of this identity's history rows.
    redact_identity: str | None = None

    @property
    def empty(self) -> bool:
        return self.identities is None and not self.spaces


def _reject_vectors(value: object, where: str = "history row") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"vector", "vectors"}:
                raise VoiceLibraryError(f"{where} must never hold a vector")
            _reject_vectors(nested, where)
    elif isinstance(value, list):
        for nested in value:
            _reject_vectors(nested, where)


def history_row(action: str, at: str, **fields: object) -> dict[str, object]:
    if action not in HISTORY_ACTIONS:
        raise VoiceLibraryError(f"unknown history action {action!r}")
    row: dict[str, object] = {"at": require_utc_timestamp(at, "at"), "action": action}
    row.update(fields)
    _reject_vectors(row)
    return row


def _write_cas(path: Path, payload: str, *, expected: bytes | None) -> None:
    def still_expected() -> None:
        try:
            current: bytes | None = path.read_bytes()
        except FileNotFoundError:
            current = None
        if current != expected:
            raise LibraryConflict(
                f"{path} changed while this command held the library lock "
                "(is locking enabled on this mount?); re-run the command"
            )

    fsio.atomic_write_text(path, payload, before_replace=still_expected)


# fsio.atomic_path names its temp files ``.<stem>.<random>.part<suffix>``.
_TEMP_FILE_RE = re.compile(r"^\..+\.part\.jsonl?$")


def _sweep_temp_files(paths: LibraryPaths) -> None:
    """Delete temp files that a killed writer left in the library.

    Only called under the exclusive lock, where no live writer can own one.
    A temp file of a space holds every vector of that space, so leaving it
    would let a forgotten person's voice outlive ``voices forget``.
    """
    for directory in (paths.root, paths.spaces_dir):
        try:
            entries = os.listdir(directory)
        except FileNotFoundError:
            continue
        for entry in entries:
            if _TEMP_FILE_RE.fullmatch(entry) is None:
                continue
            try:
                (directory / entry).unlink()
            except FileNotFoundError:
                pass
            else:
                log.info("removed a stale temp file %s from the voice library", entry)


def _append_history(paths: LibraryPaths, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    descriptor = os.open(paths.history, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# History fields that can carry what a person is called: folder and show
# names (scopes) and media stems (episode labels).
_REDACTED_HISTORY_FIELDS = frozenset({"scope", "old_scope", "episode"})


def read_history(paths: LibraryPaths) -> list[dict[str, object]]:
    """The parsed history rows; unreadable lines are skipped."""
    try:
        raw = paths.history.read_bytes()
    except FileNotFoundError:
        return []
    rows: list[dict[str, object]] = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(cast(dict[str, object], row))
    return rows


def _rewrite_history_redacted(
    paths: LibraryPaths, identity_id: str, rows: Sequence[Mapping[str, object]]
) -> None:
    """Replace the history with ``identity_id``'s labels stripped, plus ``rows``.

    Under the exclusive lock no other writer appends meanwhile. Lines that do
    not parse are kept byte for byte.
    """
    try:
        raw = paths.history.read_bytes()
    except FileNotFoundError:
        raw = b""
    kept: list[bytes] = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, ValueError):
            kept.append(line)
            continue
        if isinstance(row, dict) and row.get("identity") == identity_id:
            row = {
                key: value
                for key, value in row.items()
                if key not in _REDACTED_HISTORY_FIELDS
            }
            kept.append(canonical_json_bytes(row))
        else:
            kept.append(line)
    kept.extend(canonical_json_bytes(row) for row in rows)
    payload = b"".join(line + b"\n" for line in kept)
    with fsio.atomic_path(paths.history) as temporary:
        with open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def commit(state: LibraryState, change: LibraryChange) -> None:
    """Write ``change`` under the exclusive lock ``state`` was read with.

    Each changed file is replaced only if its bytes still equal the ones
    ``state`` observed; history is appended after every data file is in
    place. Every document is validated and encoded before the first file is
    replaced, so a refusal (a size limit included) changes nothing. The
    first write into an empty library logs a biometrics notice.
    """
    if change.empty:
        return
    paths = state.paths
    healed_spaces: dict[str, dict[str, object]] = {}
    healed_history: list[dict[str, object]] = []
    for name, dropped in sorted(state.orphans.items()):
        # Delete what read_state skipped: vectors of identities that do not
        # exist (a space the change rewrites anyway was read without them).
        healed_history.append(
            history_row(
                "orphan",
                utc_timestamp(),
                space=name,
                identities=sorted(dropped),
                exemplars=sum(dropped.values()),
            )
        )
        if name not in change.spaces:
            document = copy.deepcopy(state.spaces[name])
            document["revision"] = cast(int, document["revision"]) + 1
            healed_spaces[name] = document
    if healed_spaces or healed_history:
        change = dataclasses.replace(
            change,
            spaces={**change.spaces, **healed_spaces},
            history=(*healed_history, *change.history),
        )
    for row in change.history:
        _reject_vectors(row)
    identities = change.identities or state.identities
    validate_identities(identities)
    merged_spaces = dict(state.spaces)
    for name, document in change.spaces.items():
        validate_space(document, name=name)
        merged_spaces[name] = document
    validate_relations(LibraryState(paths, identities, merged_spaces, state.observed))

    writes: list[tuple[Path, str]] = []
    if change.identities is not None:
        writes.append(
            (
                paths.identities,
                encode_json_bytes(
                    change.identities, max_bytes=IDENTITIES_MAX_BYTES
                ).decode("utf-8"),
            )
        )
    space_writes = [
        (
            paths.space(name),
            encode_json_bytes(document, max_bytes=SPACE_MAX_BYTES).decode("utf-8"),
        )
        for name, document in sorted(change.spaces.items())
    ]
    writes = writes + space_writes if change.identities_first else space_writes + writes
    for path, _payload in writes:
        if path not in state.observed:
            raise VoiceLibraryError(f"{path} was not read before being written")

    first_write = (
        state.observed.get(paths.identities) is None and not paths.history.exists()
    )
    _sweep_temp_files(paths)
    if space_writes:
        _ensure_private_dir(paths.spaces_dir)
    for path, payload in writes:
        _write_cas(path, payload, expected=state.observed[path])
    try:
        if change.redact_identity is not None:
            _rewrite_history_redacted(paths, change.redact_identity, change.history)
        else:
            _append_history(paths, change.history)
    except OSError as exc:
        log.error(
            "voice library change committed, but its history row could not be "
            "appended to %s: %s",
            paths.history,
            exc,
        )
    if first_write:
        log.warning(FIRST_WRITE_NOTICE, paths.root)


# --------------------------------------------------------------------------
# Transitions (pure: they never touch the filesystem)
# --------------------------------------------------------------------------


def _default_identity_id() -> str:
    return f"v{secrets.token_hex(6)}"


def _default_exemplar_id() -> str:
    return f"x{secrets.token_hex(4)}"


def _mint(
    factory: Callable[[], str],
    used: set[str],
    validator: Callable[[object], str],
    kind: str,
) -> str:
    for _attempt in range(1024):
        candidate = validator(factory())
        if candidate not in used:
            return candidate
    raise EnrollmentRefusal(f"could not mint a unique {kind} id")


def _event_time(at: str | datetime | None) -> str:
    if isinstance(at, str):
        return require_utc_timestamp(at, "event timestamp")
    return utc_timestamp(at)


def scoped_episode(scope: str, episode: str) -> str:
    """The episode index key of the library: an episode label within a scope.

    Episode labels default to the media stem, and "Episode 01" exists in every
    season folder; the per-show store's episode index therefore applies within
    one scope, never across the whole library.
    """
    return f"{episode} ({scope})"


def _exemplar_keys(exemplars: Sequence[Mapping[str, object]]) -> list[ExemplarKey]:
    keys: list[ExemplarKey] = []
    for exemplar in exemplars:
        source = cast(Mapping[str, object], exemplar["source"])
        keys.append(
            ExemplarKey(
                id=cast(str, exemplar["id"]),
                capture_id=cast(str, source["capture_id"]),
                media_fingerprint=cast(str, source["media_fingerprint"]),
                episode=scoped_episode(
                    cast(str, exemplar["scope"]), normalize_episode(source["episode"])
                ),
                vector=cast(list[int | float], exemplar["vector"]),
                added=cast(str, exemplar["added"]),
            )
        )
    return keys


def _used_exemplar_ids(space: Mapping[str, object]) -> set[str]:
    return {
        cast(str, exemplar["id"])
        for items in cast(
            Mapping[str, list[Mapping[str, object]]], space["exemplars"]
        ).values()
        for exemplar in items
    }


def require_same_space(state: LibraryState, name: str, fingerprint: str) -> None:
    """Refuse a space file whose full fingerprint differs from ``fingerprint``.

    File names carry only 12 hex digits of the fingerprint; the frozen full
    value decides whether two captures really share an embedding space.
    """
    space = state.spaces.get(name)
    if space is None:
        return
    stored = cast(Mapping[str, object], space["space"])["fingerprint"]
    if stored != fingerprint:
        raise VoiceLibraryError(
            f"space file {name}.json belongs to another embedding space "
            "(12-digit fingerprint collision); it cannot be shared"
        )


def _space_for(
    state: LibraryState, provenance: Mapping[str, object]
) -> tuple[str, dict[str, object]]:
    name, fingerprint = space_identity(provenance)
    if state.paths.space(name) not in state.observed:
        raise VoiceLibraryError(f"space {name} was not read before the transition")
    require_same_space(state, name, fingerprint)
    existing = state.spaces.get(name)
    document = (
        copy.deepcopy(existing) if existing is not None else new_space(provenance)
    )
    validate_space(document, name=name)
    return name, document


def _add_scopes(
    state: LibraryState,
    identity_id: str,
    identity: dict[str, object],
    new_scopes: Sequence[str],
    *,
    space_name: str,
    space_exemplars: Mapping[str, Sequence[Mapping[str, object]]],
    event_at: str,
) -> int:
    """Add ``new_scopes`` to ``identity`` (in place); return how many were new.

    Scopes accumulate as voices are enrolled in new folders while only
    MAX_EXEMPLARS samples per space stay, so past MAX_SCOPES the oldest
    scopes that no retained sample (in any space) carries are dropped
    rather than refusing the enrollment. That needs every space loaded.
    """
    scopes = cast(list[str], identity["scopes"])
    missing = [scope for scope in new_scopes if scope not in scopes]
    if not missing:
        return 0
    merged = [*scopes, *missing]
    excess = len(merged) - MAX_SCOPES
    if excess > 0:
        if not state.all_spaces:
            raise VoiceLibraryError(
                f"identity {identity_id} has {MAX_SCOPES} scopes; pruning them "
                "must read every embedding space first"
            )
        keep = set(new_scopes)
        for name in state.spaces:
            items = (
                space_exemplars.get(identity_id, [])
                if name == space_name
                else state.space_exemplars(name).get(identity_id, [])
            )
            keep.update(cast(str, item["scope"]) for item in items)
        if space_name not in state.spaces:
            keep.update(
                cast(str, item["scope"])
                for item in space_exemplars.get(identity_id, [])
            )
        pruned: list[str] = []
        for scope in merged:  # oldest first
            if excess > 0 and scope not in keep:
                excess -= 1
                continue
            pruned.append(scope)
        if excess > 0:
            raise VoiceLibraryError(
                f"identity {identity_id} cannot hold more than {MAX_SCOPES} scopes"
            )
        merged = pruned
    identity["scopes"] = merged
    identity["updated"] = event_at
    return len(missing)


@dataclass(frozen=True)
class LibraryPlan:
    """:class:`voicestore.IndexedPlan` plus ``rescope``: the same capture,
    unchanged, moved to another scope (``target`` indexes it)."""

    outcome: Literal["enroll", "replace", "noop", "rescope"]
    target: int | None = None
    evict: int | None = None


def plan_library_enrollment(
    current: Sequence[Mapping[str, object]],
    *,
    capture_id: str,
    media_fingerprint: str,
    scope: str,
    episode: str,
    vector: Sequence[int | float],
    turns_digest: str | None,
    replace_episode: bool,
) -> LibraryPlan:
    """Decide what one incoming voice does to one identity's exemplars.

    The per-show relation (voicestore.plan_indexed_enrollment, with the
    episode index scoped) decides, except for two changes it cannot tell
    apart from corruption, both about a capture or media the identity
    already holds under the same episode label:

    - another scope (the folder was renamed, or ``--show`` changed): the
      unchanged capture moves to the new scope (``rescope``, no flag
      needed); a new capture of the same media replaces it with
      ``replace_episode``;
    - the same capture with a new vector and new speaker turns (a speaker
      split rewrote the centroids but keeps the capture id): a replacement
      that ``replace_episode`` authorizes.
    """
    keys = _exemplar_keys(current)
    scoped = scoped_episode(scope, episode)
    capture_hit = next(
        (i for i, key in enumerate(keys) if key.capture_id == capture_id), None
    )
    media_hit = next(
        (i for i, key in enumerate(keys) if key.media_fingerprint == media_fingerprint),
        None,
    )
    hit = capture_hit if capture_hit is not None else media_hit
    if hit is not None:
        stored = current[hit]
        stored_scope = cast(str, stored["scope"])
        stored_source = cast(Mapping[str, object], stored["source"])
        vector_changed = capture_hit is not None and list(keys[hit].vector) != list(
            vector
        )
        if stored_scope != scope or vector_changed:
            episode_hit = next(
                (i for i, key in enumerate(keys) if key.episode == scoped), None
            )
            hits = {i for i in (capture_hit, media_hit, episode_hit) if i is not None}
            if len(hits) > 1:
                raise CrossIndexRefusal(
                    "enrollment evidence resolves different exemplars across "
                    "indexes: " + ", ".join(sorted(keys[i].id for i in hits))
                )
            what = (
                f"capture {capture_id}" if capture_hit is not None else "source media"
            )
            if keys[hit].media_fingerprint != media_fingerprint:
                raise EnrollmentRefusal(
                    f"capture integrity failure for {capture_id}: media differs"
                )
            stored_label = normalize_episode(stored_source["episode"])
            if stored_label != episode:
                raise EnrollmentRefusal(
                    f"{what} is already enrolled as {stored_label!r} "
                    f"in scope {stored_scope!r}"
                )
            if capture_hit is not None and not vector_changed:
                return LibraryPlan("rescope", target=hit)
            if vector_changed and (
                turns_digest is None
                or stored_source.get("turns_digest") == turns_digest
            ):
                # The same speaker turns cannot yield another centroid.
                raise EnrollmentRefusal(
                    f"capture integrity failure for {capture_id}: vector differs"
                )
            if not replace_episode:
                if vector_changed:
                    raise EnrollmentRefusal(
                        f"the voices of capture {capture_id} changed since it was "
                        "enrolled (a speaker was split); use --replace to update it"
                    )
                raise EnrollmentRefusal(
                    f"source media for {episode!r} is already enrolled in scope "
                    f"{stored_scope!r}; use --replace to move it to {scope!r}"
                )
            return LibraryPlan("replace", target=hit)
    plan = plan_indexed_enrollment(
        keys,
        capture_id=capture_id,
        media_fingerprint=media_fingerprint,
        episode=scoped,
        vector=vector,
        replace_episode=replace_episode,
    )
    return LibraryPlan(plan.outcome, target=plan.target, evict=plan.evict)


@dataclass(frozen=True)
class EpisodeSource:
    """Where one enrolled voice came from (a pointer, never audio)."""

    media_path: str | None
    media_fingerprint: str
    capture_id: str
    turns_digest: str | None
    episode: str


@dataclass(frozen=True)
class EnrollEntry:
    """One human-named speaker to enroll; ``identity_id`` None creates one."""

    identity_id: str | None
    raw_name: str
    speaker_label: str | None
    vector: Sequence[int | float]


@dataclass(frozen=True)
class EnrollOutcome:
    identity_id: str
    outcome: Literal["enroll", "replace", "noop", "rescope"]
    exemplar_id: str
    evicted_exemplar_id: str | None = None
    created: bool = False


def enroll_entries(
    state: LibraryState,
    *,
    provenance: Mapping[str, object],
    scope: str,
    source: EpisodeSource,
    entries: Sequence[EnrollEntry],
    replace_episode: bool = False,
    at: str | datetime | None = None,
    identity_id_factory: Callable[[], str] = _default_identity_id,
    exemplar_id_factory: Callable[[], str] = _default_exemplar_id,
) -> tuple[LibraryChange, tuple[EnrollOutcome, ...]]:
    """Enroll one episode's named speakers into the space of ``provenance``.

    Each identity keeps the per-show store's relation within one space (see
    :func:`plan_library_enrollment`): the same capture is a no-op (or moves
    to a new scope), the same media or episode needs ``replace_episode``, and
    at most MAX_EXEMPLARS stay per identity, the oldest displaced. An
    identity gains ``scope`` only when one of its exemplars actually changes.
    """
    if type(replace_episode) is not bool:
        raise EnrollmentRefusal("replace_episode must be a boolean")
    event_at = _event_time(at)
    scope = normalize_scope(scope)
    episode = normalize_episode(source.episode)
    capture = require_capture_id(source.capture_id)
    media_hash = require_sha256(source.media_fingerprint, "media_fingerprint")
    turns_digest = (
        None
        if source.turns_digest is None
        else require_sha256(source.turns_digest, "turns_digest")
    )
    media_path = (
        None
        if source.media_path is None
        else require_string(
            source.media_path, "media_path", max_bytes=MAX_MEDIA_PATH_BYTES
        )
    )
    space_name, space = _space_for(state, provenance)
    dim = validate_space(space, name=space_name).embedding_dim
    identities_document = copy.deepcopy(state.identities)
    identities = cast(dict[str, dict[str, object]], identities_document["identities"])
    exemplars_by_identity = cast(dict[str, list[dict[str, object]]], space["exemplars"])
    used_exemplars = _used_exemplar_ids(space)
    history: list[dict[str, object]] = []
    outcomes: list[EnrollOutcome] = []
    targets: set[str] = set()
    identities_changed = False
    space_changed = False

    for entry in entries:
        name = require_string(entry.raw_name, "speaker name", max_bytes=MAX_NAME_BYTES)
        normalize_speaker_key(name)
        vector = list(validate_vector(entry.vector, dim=dim, field="incoming vector"))
        if entry.identity_id is None:
            identity_id = _mint(
                identity_id_factory, set(identities), require_identity_id, "identity"
            )
        else:
            identity_id = require_identity_id(entry.identity_id)
            if identity_id in state.forgotten:
                raise EnrollmentRefusal(
                    f"identity {identity_id} was forgotten (`voxweave voices "
                    "forget`) and cannot be enrolled again"
                )
        if identity_id in targets:
            raise EnrollmentRefusal(
                f"two speakers of one episode resolve to identity {identity_id}"
            )
        targets.add(identity_id)
        current = list(exemplars_by_identity.get(identity_id, []))
        plan = plan_library_enrollment(
            current,
            capture_id=capture,
            media_fingerprint=media_hash,
            scope=scope,
            episode=episode,
            vector=vector,
            turns_digest=turns_digest,
            replace_episode=replace_episode,
        )
        if plan.outcome == "noop":
            assert plan.target is not None
            outcomes.append(
                EnrollOutcome(
                    identity_id, "noop", cast(str, current[plan.target]["id"])
                )
            )
            continue

        created = identity_id not in identities
        if created:
            identities[identity_id] = {
                "display_name": name,
                "aliases": [],
                "scopes": [scope],
                "created": event_at,
                "updated": event_at,
            }
            history.append(
                history_row("create", event_at, identity=identity_id, scope=scope)
            )
            identities_changed = True
        elif _add_scopes(
            state,
            identity_id,
            identities[identity_id],
            [scope],
            space_name=space_name,
            space_exemplars=exemplars_by_identity,
            event_at=event_at,
        ):
            identities_changed = True

        if plan.outcome == "rescope":
            # The same voice sample, now filed under the new scope; it keeps
            # its id, vector and age.
            assert plan.target is not None
            moved = copy.deepcopy(current[plan.target])
            old_scope = moved["scope"]
            moved["scope"] = scope
            moved_source = cast(dict[str, object], moved["source"])
            if media_path is not None:
                moved_source["media_path"] = media_path
            moved_source["speaker_label"] = entry.speaker_label
            current[plan.target] = moved
            exemplars_by_identity[identity_id] = current
            space_changed = True
            history.append(
                history_row(
                    "rescope",
                    event_at,
                    identity=identity_id,
                    space=space_name,
                    exemplar=moved["id"],
                    old_scope=old_scope,
                    scope=scope,
                    episode=episode,
                )
            )
            outcomes.append(
                EnrollOutcome(identity_id, "rescope", cast(str, moved["id"]))
            )
            continue

        exemplar_id = _mint(
            exemplar_id_factory, used_exemplars, require_exemplar_id, "exemplar"
        )
        used_exemplars.add(exemplar_id)
        exemplar: dict[str, object] = {
            "id": exemplar_id,
            "vector": vector,
            "scope": scope,
            "source": {
                "media_path": media_path,
                "media_fingerprint": media_hash,
                "capture_id": capture,
                "turns_digest": turns_digest,
                "speaker_label": entry.speaker_label,
                "episode": episode,
            },
            "added": event_at,
        }
        evicted: str | None = None
        if plan.outcome == "replace":
            assert plan.target is not None
            old_id = cast(str, current[plan.target]["id"])
            current[plan.target] = exemplar
            history.append(
                history_row(
                    "replace",
                    event_at,
                    identity=identity_id,
                    space=space_name,
                    old_exemplar=old_id,
                    new_exemplar=exemplar_id,
                    scope=scope,
                    episode=episode,
                )
            )
        else:
            if plan.evict is not None:
                oldest = current.pop(plan.evict)
                evicted = cast(str, oldest["id"])
                history.append(
                    history_row(
                        "evict",
                        event_at,
                        identity=identity_id,
                        space=space_name,
                        exemplar=evicted,
                        episode=cast(Mapping[str, object], oldest["source"])["episode"],
                    )
                )
            current.append(exemplar)
            history.append(
                history_row(
                    "enroll",
                    event_at,
                    identity=identity_id,
                    space=space_name,
                    exemplar=exemplar_id,
                    scope=scope,
                    episode=episode,
                )
            )
        exemplars_by_identity[identity_id] = current
        space_changed = True
        outcomes.append(
            EnrollOutcome(
                identity_id, plan.outcome, exemplar_id, evicted, created=created
            )
        )

    if space_changed and _list_space(identities_document, space_name):
        identities_changed = True
    if identities_changed:
        identities_document["revision"] = cast(int, identities_document["revision"]) + 1
    if space_changed:
        space["revision"] = cast(int, space["revision"]) + 1
    change = LibraryChange(
        identities=identities_document if identities_changed else None,
        spaces={space_name: space} if space_changed else {},
        history=tuple(history),
    )
    return change, tuple(outcomes)


def _identity_or_raise(state: LibraryState, identity_id: str) -> dict[str, object]:
    identity = state.identity_map.get(identity_id)
    if identity is None:
        raise UnknownIdentity(f"no voice library identity {identity_id}")
    return identity


def rename_identity(
    state: LibraryState,
    identity_id: str,
    new_name: str,
    *,
    at: str | datetime | None = None,
) -> LibraryChange:
    """Rename one identity; the name applies in every embedding space."""
    _identity_or_raise(state, identity_id)
    name = require_string(new_name, "new name", max_bytes=MAX_NAME_BYTES)
    normalize_speaker_key(name, field="new name")
    document = copy.deepcopy(state.identities)
    identity = cast(dict[str, dict[str, object]], document["identities"])[identity_id]
    if identity["display_name"] == name:
        return LibraryChange()
    event_at = _event_time(at)
    identity["display_name"] = name
    identity["updated"] = event_at
    document["revision"] = cast(int, document["revision"]) + 1
    return LibraryChange(
        identities=document,
        history=(history_row("rename", event_at, identity=identity_id),),
    )


def forget_identity(
    state: LibraryState,
    identity_id: str,
    *,
    at: str | datetime | None = None,
) -> tuple[LibraryChange, dict[str, int]]:
    """Remove an identity from identities.json and from every space.

    ``state`` must have been read with every space loaded. The id stays as a
    tombstone in ``forgotten``, so neither a per-folder store from an
    earlier version nor a review page generated before the forget can bring
    it back. The history row records the id and per-space counts only, and
    the id's earlier rows lose their scopes and episode labels, which are
    often a person's name (a folder of home recordings, a media stem).
    """
    _identity_or_raise(state, identity_id)
    if not state.all_spaces:
        raise VoiceLibraryError("forget must read every embedding space first")
    if state.missing_spaces:
        # Forgetting without them could leave this person's vectors behind.
        raise VoiceLibraryError(
            "identities.json lists space file(s) that cannot be found: "
            + ", ".join(state.missing_spaces)
            + ". Another machine may have just created them on a network share "
            "whose directory cache is stale: re-run in a minute. If a file was "
            'deleted on purpose, remove its name from the "spaces" list in '
            "identities.json"
        )
    event_at = _event_time(at)
    identities = copy.deepcopy(state.identities)
    del cast(dict[str, object], identities["identities"])[identity_id]
    identities["forgotten"] = [*forgotten_ids(identities), identity_id][-MAX_FORGOTTEN:]
    identities["revision"] = cast(int, identities["revision"]) + 1
    spaces: dict[str, dict[str, object]] = {}
    removed: dict[str, int] = {}
    for name, space in sorted(state.spaces.items()):
        exemplars = cast(Mapping[str, list[object]], space["exemplars"])
        if identity_id not in exemplars:
            continue
        document = copy.deepcopy(space)
        items = cast(dict[str, list[object]], document["exemplars"]).pop(identity_id)
        document["revision"] = cast(int, document["revision"]) + 1
        spaces[name] = document
        removed[name] = len(items)
    change = LibraryChange(
        identities=identities,
        spaces=spaces,
        history=(
            history_row("forget", event_at, identity=identity_id, exemplars=removed),
        ),
        identities_first=False,
        redact_identity=identity_id,
    )
    return change, removed


def legacy_store_candidates(state: LibraryState, identity_id: str) -> list[Path]:
    """Per-folder stores that may still hold ``identity_id``'s vectors.

    The stores ``voices import`` read (its history rows) and the
    ``voxweave.voices.json`` beside every media file one of the identity's
    samples came from. ``forget`` does not edit them (they belong to earlier
    versions and are only read); it names the ones that still hold the id.
    """
    candidates: dict[str, Path] = {}
    for row in read_history(state.paths):
        source = row.get("source")
        if row.get("action") == "import" and isinstance(source, str):
            candidates.setdefault(source, Path(source))
    for name in sorted(state.spaces):
        for item in state.space_exemplars(name).get(identity_id, []):
            media_path = cast(Mapping[str, object], item["source"]).get("media_path")
            if isinstance(media_path, str):
                path = legacy_store_path(Path(media_path))
                candidates.setdefault(str(path), path)
    return list(candidates.values())


def legacy_stores_holding(
    paths: Iterable[Path], identity_id: str
) -> list[tuple[Path, int]]:
    """The stores among ``paths`` that hold ``identity_id``, with its sample count."""
    holding: list[tuple[Path, int]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            resolved, store = read_legacy_store(path)
        except (OSError, Phase2DataError) as exc:
            log.warning("could not check per-folder store %s: %s", path, exc)
            continue
        identity = cast(Mapping[str, object], store["identities"]).get(identity_id)
        if identity is not None:
            exemplars = cast(Mapping[str, object], identity)["exemplars"]
            holding.append((resolved, len(cast(list[object], exemplars))))
    return holding


def _older_than_retained(
    keys: Sequence[ExemplarKey], added: str, exemplar_id: str
) -> bool:
    """Whether an exemplar stamped ``added`` would itself be the one displaced.

    At the cap the oldest exemplar (by ``added``, then id) gives way. One
    older than every retained exemplar would be displaced at once, so it
    counts as already superseded instead of evicting a newer one: merging
    old voice samples keeps the newest MAX_EXEMPLARS of the union.
    """
    if len(keys) < MAX_EXEMPLARS:
        return False
    return (added, exemplar_id) < min((key.added, key.id) for key in keys)


@dataclass(frozen=True)
class ImportSummary:
    space: str
    scope: str
    identities_created: int
    exemplars_added: int
    exemplars_present: int
    refused: tuple[str, ...] = ()
    # Older than every sample an identity at the cap already keeps.
    exemplars_superseded: int = 0
    # Identities of the store that `voices forget` removed: never re-imported.
    forgotten: tuple[str, ...] = ()
    # Every scope the store's identities now carry, and how many were new.
    scopes: tuple[str, ...] = ()
    scopes_added: int = 0


def default_import_scopes(store_path: Path, show: str) -> tuple[str, ...]:
    """The scopes under which a per-show store already served suggestions.

    A ``voxweave.voices.json`` found beside the media served tier 1 both to
    episodes of its folder (without ``--show``) and under its show, so its
    identities keep both scopes, the folder's first (the scope its voice
    samples are filed under). A store with another name was only ever used
    with ``--voices`` and ``--show``: its show.
    """
    show_scope = normalize_scope(show)
    if Path(store_path).name != LEGACY_STORE_NAME:
        return (show_scope,)
    return tuple(dict.fromkeys([episode_scope(Path(store_path)), show_scope]))


def import_store(
    state: LibraryState,
    store: Mapping[str, object],
    *,
    scope: str,
    source_label: str,
    also_scopes: Sequence[str] = (),
    at: str | datetime | None = None,
    exemplar_id_factory: Callable[[], str] = _default_exemplar_id,
) -> tuple[LibraryChange, ImportSummary]:
    """Merge a pre-library per-show store into the library, idempotently.

    Identity and exemplar ids are kept, so importing the same store again
    finds every capture already present and changes nothing; an identity the
    library already has (imported before, then maybe renamed) keeps its
    library names. Exemplars go into the space of the store's frozen
    provenance, filed under ``scope``, and follow the same per-identity
    relation and cap as enrollment; one that contradicts the library is
    reported and skipped. Newest first, so an identity at the cap keeps the
    newest samples of the union: an older legacy sample is superseded, never
    swapped in for a newer library one (which the next import would then
    swap back). Every identity of the store gains ``scope`` and
    ``also_scopes``, also on a re-import (the way to add a scope).
    """
    validated = validate_voice_store(store)
    scope = normalize_scope(scope)
    import_scopes = list(
        dict.fromkeys([scope, *(normalize_scope(extra) for extra in also_scopes)])
    )
    scopes_added = 0
    event_at = _event_time(at)
    provenance = cast(Mapping[str, object], store["provenance"])
    space_name, space = _space_for(state, provenance)
    identities_document = copy.deepcopy(state.identities)
    identities = cast(dict[str, dict[str, object]], identities_document["identities"])
    exemplars_by_identity = cast(dict[str, list[dict[str, object]]], space["exemplars"])
    used_exemplars = _used_exemplar_ids(space)
    history: list[dict[str, object]] = []
    refused: list[str] = []
    created_count = added_count = present_count = superseded_count = 0
    identities_changed = space_changed = False
    tombstones = state.forgotten
    skipped_forgotten = sorted(set(validated.identities) & tombstones)

    for identity_id in sorted(set(validated.identities) - tombstones):
        legacy = cast(Mapping[str, object], validated.identities[identity_id])
        current = list(exemplars_by_identity.get(identity_id, []))
        added_here = 0
        legacy_exemplars = sorted(
            cast(list[Mapping[str, object]], legacy["exemplars"]),
            key=lambda item: (cast(str, item["added"]), cast(str, item["id"])),
            reverse=True,
        )
        for legacy_exemplar in legacy_exemplars:
            episode = normalize_episode(legacy_exemplar["episode"])
            vector = cast(list[int | float], legacy_exemplar["vector"])
            keys = _exemplar_keys(current)
            if any(
                key.capture_id == legacy_exemplar["capture_id"]
                and list(key.vector) == list(vector)
                and key.media_fingerprint == legacy_exemplar["media_fingerprint"]
                for key in keys
            ):
                # Imported before, possibly under another --scope.
                present_count += 1
                continue
            if _older_than_retained(
                keys,
                cast(str, legacy_exemplar["added"]),
                cast(str, legacy_exemplar["id"]),
            ):
                superseded_count += 1
                continue
            try:
                plan = plan_indexed_enrollment(
                    keys,
                    capture_id=cast(str, legacy_exemplar["capture_id"]),
                    media_fingerprint=cast(str, legacy_exemplar["media_fingerprint"]),
                    episode=scoped_episode(scope, episode),
                    vector=vector,
                    replace_episode=False,
                )
            except EnrollmentRefusal as exc:
                refused.append(f"{identity_id}/{legacy_exemplar['id']}: {exc}")
                continue
            if plan.outcome == "noop":
                present_count += 1
                continue
            legacy_id = cast(str, legacy_exemplar["id"])
            exemplar_id = (
                legacy_id
                if legacy_id not in used_exemplars
                else _mint(
                    exemplar_id_factory, used_exemplars, require_exemplar_id, "exemplar"
                )
            )
            used_exemplars.add(exemplar_id)
            if plan.evict is not None:
                oldest = current.pop(plan.evict)
                history.append(
                    history_row(
                        "evict",
                        event_at,
                        identity=identity_id,
                        space=space_name,
                        exemplar=cast(str, oldest["id"]),
                        episode=cast(Mapping[str, object], oldest["source"])["episode"],
                    )
                )
            current.append(
                {
                    "id": exemplar_id,
                    "vector": list(vector),
                    "scope": scope,
                    "source": {
                        "media_path": None,
                        "media_fingerprint": legacy_exemplar["media_fingerprint"],
                        "capture_id": legacy_exemplar["capture_id"],
                        "turns_digest": None,
                        "speaker_label": None,
                        "episode": episode,
                    },
                    "added": legacy_exemplar["added"],
                }
            )
            added_here += 1

        if added_here:
            exemplars_by_identity[identity_id] = current
            added_count += added_here
            space_changed = True
        if identity_id not in identities:
            identities[identity_id] = {
                "display_name": legacy["display_name"],
                "aliases": list(cast(list[str], legacy["aliases"])),
                "scopes": list(import_scopes),
                "created": event_at,
                "updated": event_at,
            }
            created_count += 1
            identities_changed = True
        else:
            added_scopes = _add_scopes(
                state,
                identity_id,
                identities[identity_id],
                import_scopes,
                space_name=space_name,
                space_exemplars=exemplars_by_identity,
                event_at=event_at,
            )
            if added_scopes:
                identities_changed = True
                scopes_added += added_scopes

    summary = ImportSummary(
        space=space_name,
        scope=scope,
        identities_created=created_count,
        exemplars_added=added_count,
        exemplars_present=present_count,
        refused=tuple(refused),
        exemplars_superseded=superseded_count,
        forgotten=tuple(skipped_forgotten),
        scopes=tuple(import_scopes),
        scopes_added=scopes_added,
    )
    if space_changed and _list_space(identities_document, space_name):
        identities_changed = True
    if not (identities_changed or space_changed):
        return LibraryChange(), summary
    if identities_changed:
        identities_document["revision"] = cast(int, identities_document["revision"]) + 1
    if space_changed:
        space["revision"] = cast(int, space["revision"]) + 1
    history.append(
        history_row(
            "import",
            event_at,
            source=source_label,
            scopes=list(import_scopes),
            space=space_name,
            identities_created=created_count,
            exemplars_added=added_count,
            scopes_added=scopes_added,
        )
    )
    change = LibraryChange(
        identities=identities_document if identities_changed else None,
        spaces={space_name: space} if space_changed else {},
        history=tuple(history),
    )
    return change, summary


# --------------------------------------------------------------------------
# Lookup and matching views
# --------------------------------------------------------------------------


def _names_of(identity: Mapping[str, object]) -> set[str]:
    return {
        normalize_speaker_key(name)
        for name in [identity["display_name"], *cast(list[object], identity["aliases"])]
    }


def identities_named(
    state: LibraryState, raw_name: str, *, scope: str | None = None
) -> list[str]:
    """Ids whose display name or alias normalizes to ``raw_name``'s key."""
    key = normalize_speaker_key(raw_name)
    scope_key = None if scope is None else normalize_scope(scope)
    return sorted(
        identity_id
        for identity_id, identity in state.identity_map.items()
        if key in _names_of(identity)
        and (scope_key is None or scope_key in cast(list[str], identity["scopes"]))
    )


def find_identities(state: LibraryState, query: str) -> list[str]:
    """Resolve an identity id, or every identity carrying ``query`` as a name."""
    text = query.strip()
    if IDENTITY_ID_RE.fullmatch(text) and text in state.identity_map:
        return [text]
    try:
        return identities_named(state, query)
    except Phase2DataError:
        return []


def identity_summary(state: LibraryState, identity_id: str) -> dict[str, object]:
    """A vector-free description of one identity."""
    identity = _identity_or_raise(state, identity_id)
    counts = {
        name: len(state.space_exemplars(name)[identity_id])
        for name in sorted(state.spaces)
        if identity_id in state.space_exemplars(name)
    }
    return {
        "id": identity_id,
        "display_name": identity["display_name"],
        "aliases": list(cast(list[str], identity["aliases"])),
        "scopes": list(cast(list[str], identity["scopes"])),
        "exemplars": counts,
        "created": identity["created"],
        "updated": identity["updated"],
    }


def list_identities(
    state: LibraryState, *, scope: str | None = None
) -> list[dict[str, object]]:
    scope_key = None if scope is None else normalize_scope(scope)
    rows = [
        identity_summary(state, identity_id)
        for identity_id, identity in state.identity_map.items()
        if scope_key is None or scope_key in cast(list[str], identity["scopes"])
    ]
    return sorted(
        rows, key=lambda row: (cast(str, row["display_name"]).casefold(), row["id"])
    )


def describe_identity(state: LibraryState, identity_id: str) -> dict[str, object]:
    """``identity_summary`` plus each exemplar's source pointer (no vectors)."""
    summary = identity_summary(state, identity_id)
    spaces: dict[str, object] = {}
    for name in sorted(state.spaces):
        items = state.space_exemplars(name).get(identity_id)
        if not items:
            continue
        provenance = cast(
            Mapping[str, object],
            cast(Mapping[str, object], state.spaces[name]["space"])["provenance"],
        )
        spaces[name] = {
            "embedding_model": provenance.get("embedding_model"),
            "exemplars": [
                {
                    "id": item["id"],
                    "scope": item["scope"],
                    "episode": cast(Mapping[str, object], item["source"])["episode"],
                    "media_path": cast(Mapping[str, object], item["source"])[
                        "media_path"
                    ],
                    "added": item["added"],
                }
                for item in items
            ],
        }
    summary["spaces"] = spaces
    return summary


@dataclass(frozen=True)
class MatchPools:
    """One embedding space's identities, split into the two matching tiers."""

    in_scope: dict[str, dict[str, object]]
    other_scopes: dict[str, dict[str, object]]
    scopes: dict[str, tuple[str, ...]]


def matching_pools(state: LibraryState, space_name: str, scope: str) -> MatchPools:
    """Split a space's identities by whether they carry ``scope``."""
    scope_key = normalize_scope(scope)
    in_scope: dict[str, dict[str, object]] = {}
    other: dict[str, dict[str, object]] = {}
    scopes: dict[str, tuple[str, ...]] = {}
    for identity_id, items in state.space_exemplars(space_name).items():
        identity = state.identity_map[identity_id]
        identity_scopes = tuple(cast(list[str], identity["scopes"]))
        entry: dict[str, object] = {
            "display_name": identity["display_name"],
            "origin": "library",
            "exemplars": [
                {"id": item["id"], "vector": item["vector"]} for item in items
            ],
        }
        (in_scope if scope_key in identity_scopes else other)[identity_id] = entry
        scopes[identity_id] = identity_scopes
    return MatchPools(in_scope=in_scope, other_scopes=other, scopes=scopes)


def add_legacy_store(
    pools: MatchPools,
    store: Mapping[str, object],
    *,
    state: LibraryState,
    space_name: str,
    in_scope: bool,
) -> tuple[MatchPools, bool]:
    """Add a read-only pre-library store's identities to ``pools``.

    Identities the library already holds are skipped (the library copy wins),
    and so are identities forgotten in this library. Returns the new pools
    and whether the store holds anything the library lacks: an identity it
    does not have, or a capture that is not in this space yet and that an
    import would keep (the cue to suggest ``voxweave voices import``); a
    capture older than every sample of an identity at the cap is
    superseded, not missing.
    """
    validated = validate_voice_store(store)
    show = normalize_scope(validated.show)
    library_exemplars = state.space_exemplars(space_name)
    target = dict(pools.in_scope if in_scope else pools.other_scopes)
    scopes = dict(pools.scopes)
    unimported = False
    tombstones = state.forgotten
    for identity_id, raw in validated.identities.items():
        if identity_id in tombstones:
            continue  # forgotten in this library: never offered again
        identity = cast(Mapping[str, object], raw)
        exemplars = cast(list[Mapping[str, object]], identity["exemplars"])
        if identity_id in state.identity_map:
            keys = _exemplar_keys(library_exemplars.get(identity_id, []))
            present = {key.capture_id for key in keys}
            if any(
                item["capture_id"] not in present
                and not _older_than_retained(
                    keys, cast(str, item["added"]), cast(str, item["id"])
                )
                for item in exemplars
            ):
                unimported = True
            continue
        unimported = True
        if not exemplars:
            continue
        target[identity_id] = {
            "display_name": identity["display_name"],
            "origin": "legacy",
            "exemplars": [
                {"id": item["id"], "vector": item["vector"]} for item in exemplars
            ],
        }
        scopes[identity_id] = (show,)
    if in_scope:
        return MatchPools(target, dict(pools.other_scopes), scopes), unimported
    return MatchPools(dict(pools.in_scope), target, scopes), unimported


def match_input_digest(
    state: LibraryState,
    space_name: str,
    legacy_store: Mapping[str, object] | None = None,
) -> str:
    """Digest of every document a library match read (for the suggest record)."""
    return canonical_json_digest(
        {
            "identities": state.identities,
            "space": state.spaces.get(space_name),
            "legacy": legacy_store,
        }
    )


def space_revision(state: LibraryState, space_name: str) -> int:
    space = state.spaces.get(space_name)
    return 0 if space is None else cast(int, space["revision"])


__all__ = [
    "DEFAULT_SCOPE",
    "ENV_VOICES_DIR",
    "EnrollEntry",
    "EnrollOutcome",
    "EpisodeSource",
    "FIRST_WRITE_NOTICE",
    "HISTORY_ACTIONS",
    "ImportSummary",
    "LEGACY_STORE_NAME",
    "LibraryChange",
    "LibraryConflict",
    "LibraryLocation",
    "LibraryPaths",
    "LibraryPlan",
    "LibraryState",
    "MatchPools",
    "UnknownIdentity",
    "ValidatedSpace",
    "VoiceLibraryError",
    "add_legacy_store",
    "commit",
    "default_import_scopes",
    "default_voices_dir",
    "describe_identity",
    "empty_identities",
    "enroll_entries",
    "episode_scope",
    "find_identities",
    "forget_identity",
    "forgotten_ids",
    "history_row",
    "identities_named",
    "identity_summary",
    "import_store",
    "is_generic_folder_name",
    "legacy_store_candidates",
    "legacy_store_path",
    "legacy_stores_holding",
    "library_lock",
    "list_identities",
    "list_space_names",
    "listed_spaces",
    "match_input_digest",
    "matching_pools",
    "new_space",
    "normalize_scope",
    "plan_library_enrollment",
    "read_history",
    "read_legacy_store",
    "read_state",
    "rename_identity",
    "require_same_space",
    "resolve_voices_dir",
    "scoped_episode",
    "space_identity",
    "space_model_slug",
    "space_revision",
    "validate_identities",
    "validate_relations",
    "validate_space",
]
