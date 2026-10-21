"""Voice library documents: locations, validation, state and change types."""

from __future__ import annotations

import copy
import errno
import fcntl
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from voxweave import config
from voxweave.voicebase import (
    MAX_PROVENANCE_STRING_BYTES,
    MAX_SIDECAR_LABEL_BYTES,
    Phase2DataError,
    require_capture_id,
    require_exact_int,
    require_exemplar_id,
    require_identity_id,
    require_mapping,
    require_sha256,
    require_string,
    require_utc_timestamp,
    require_version_one,
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
    canonical_store_path,
    load_voice_store,
    normalize_episode,
    normalize_speaker_key,
    store_lock_path,
)


ENV_VOICES_DIR = "VOXWEAVE_VOICES_DIR"


IDENTITIES_NAME = "identities.json"


SPACES_DIRNAME = "spaces"


HISTORY_NAME = "history.jsonl"


LOCK_NAME = ".library.lock"


# The pre-library per-directory store, still read (never written) for matching.
LEGACY_STORE_NAME = "voxweave.voices.json"


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
        "orphan",
        "rescope",
    }
)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


_SPACE_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{12}$")


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
    exist, which were dropped from ``spaces`` on read; ``orphan_exemplars``
    holds them. The next commit deletes only those of forgotten (tombstoned)
    ids and carries the rest through unchanged: an ``identities.json``
    rolled back to an older copy must not cost the newer voices their
    samples. ``missing_spaces`` names spaces that ``identities.json`` lists
    but that could not be found.
    """

    paths: LibraryPaths
    identities: dict[str, object]
    spaces: dict[str, dict[str, object]]
    observed: dict[Path, bytes | None] = field(default_factory=dict)
    all_spaces: bool = False
    orphans: dict[str, dict[str, int]] = field(default_factory=dict)
    missing_spaces: tuple[str, ...] = ()
    orphan_exemplars: dict[str, dict[str, list[object]]] = field(default_factory=dict)

    @property
    def identity_map(self) -> dict[str, dict[str, object]]:
        return cast(dict[str, dict[str, object]], self.identities["identities"])

    @property
    def forgotten(self) -> frozenset[str]:
        return frozenset(forgotten_ids(self.identities))

    def kept_orphans(self, name: str) -> dict[str, list[object]]:
        """Orphan exemplars of ``name`` a writer keeps: ids not forgotten."""
        tombstones = self.forgotten
        return {
            identity_id: items
            for identity_id, items in self.orphan_exemplars.get(name, {}).items()
            if identity_id not in tombstones
        }

    def orphan_ids(self, name: str) -> tuple[set[str], set[str]]:
        """Identity and exemplar ids held by the orphans of space ``name``."""
        held = self.orphan_exemplars.get(name, {})
        return set(held), {
            cast(str, cast(Mapping[str, object], item)["id"])
            for items in held.values()
            for item in items
        }

    def space_exemplars(self, name: str) -> dict[str, list[dict[str, object]]]:
        space = self.spaces.get(name)
        if space is None:
            return {}
        return cast(dict[str, list[dict[str, object]]], space["exemplars"])


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


# --------------------------------------------------------------------------
# Transitions (pure: they never touch the filesystem)
# --------------------------------------------------------------------------
def scoped_episode(scope: str, episode: str) -> str:
    """The episode index key of the library: an episode label within a scope.

    Episode labels default to the media stem, and "Episode 01" exists in every
    season folder; the per-show store's episode index therefore applies within
    one scope, never across the whole library.
    """
    return f"{episode} ({scope})"
