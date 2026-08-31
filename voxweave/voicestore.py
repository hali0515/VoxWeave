"""Validated voice-store state and the pure indexed enrollment transition."""

from __future__ import annotations

import copy
import fcntl
import os
import secrets
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from voxweave.voicebase import (
    VOICES_STORE_MAX_BYTES,
    Phase2DataError,
    canonical_json_digest,
    canonical_path,
    load_json_object,
    require_capture_id,
    require_exact_int,
    require_exemplar_id,
    require_identity_id,
    require_mapping,
    require_sha256,
    require_string,
    require_utc_timestamp,
    require_version_one,
    utc_timestamp,
    validate_provenance,
    validate_vector,
    write_json_object,
)

MAX_IDENTITIES = 64
MAX_EXEMPLARS = 5
MAX_ALIASES = 8
MAX_LOG_ROWS = 1024
MAX_NAME_BYTES = 256
MAX_EPISODE_BYTES = 128


class VoiceStoreError(Phase2DataError):
    """The voices store violates a schema or relational invariant."""


class EnrollmentRefusal(VoiceStoreError):
    """An indexed enrollment transition is explicitly forbidden."""


class CrossIndexRefusal(EnrollmentRefusal):
    """Enrollment indexes resolve the incoming evidence to different slots."""


@dataclass(frozen=True)
class ValidatedVoiceStore:
    """Small validated view used by matching and enrollment."""

    show: str
    revision: int
    embedding_dim: int
    identities: Mapping[str, object]


@dataclass(frozen=True)
class StoreLockHandle:
    """Canonical paths yielded while one persistent store lock is held."""

    store_path: Path
    lock_path: Path


@dataclass(frozen=True)
class EnrollmentResult:
    """A new in-memory store plus the exact transition that produced it."""

    store: dict[str, object]
    outcome: Literal["enroll", "replace", "noop"]
    identity_id: str
    exemplar_id: str
    evicted_exemplar_id: str | None = None


def _raise(message: str) -> None:
    raise VoiceStoreError(message)


def normalize_speaker_key(
    raw: object,
    *,
    field: str = "speaker name",
    max_bytes: int = MAX_NAME_BYTES,
) -> str:
    """Apply exactly NFC(sanitize_speaker_name(raw)), never casefold/NFKC."""
    value = require_string(raw, field, max_bytes=max_bytes)
    # Lazy import avoids a cycle when the future integration wave imports this
    # module from voxweave.speakers.
    from voxweave.speakers import sanitize_speaker_name

    normalized = unicodedata.normalize("NFC", sanitize_speaker_name(value))
    require_string(normalized, f"normalized {field}", max_bytes=max_bytes)
    return normalized


def normalize_show(raw: object) -> str:
    return normalize_speaker_key(raw, field="show")


def normalize_episode(raw: object) -> str:
    return normalize_speaker_key(
        raw,
        field="episode",
        max_bytes=MAX_EPISODE_BYTES,
    )


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        _raise(f"{field} must be an array")
    return cast(list[object], value)


def _validate_log_row(row: object, index: int) -> None:
    entry = require_mapping(row, f"log[{index}]")
    require_utc_timestamp(entry.get("at"), f"log[{index}].at")
    action = entry.get("action")
    if action not in {"enroll", "replace", "evict"}:
        _raise(f"log[{index}].action must be enroll, replace, or evict")
    require_identity_id(entry.get("identity"), f"log[{index}].identity")
    require_string(
        entry.get("episode"),
        f"log[{index}].episode",
        max_bytes=MAX_EPISODE_BYTES,
    )
    if action == "replace":
        require_exemplar_id(
            entry.get("old_exemplar"),
            f"log[{index}].old_exemplar",
        )
        require_exemplar_id(
            entry.get("new_exemplar"),
            f"log[{index}].new_exemplar",
        )
        require_capture_id(entry.get("old_capture"), f"log[{index}].old_capture")
        require_capture_id(entry.get("new_capture"), f"log[{index}].new_capture")
    else:
        require_exemplar_id(entry.get("exemplar"), f"log[{index}].exemplar")
        if "capture" in entry:
            require_capture_id(entry.get("capture"), f"log[{index}].capture")


def validate_voice_store(value: object) -> ValidatedVoiceStore:
    """Validate the v1 store including all bounded relational indexes."""
    try:
        root = require_mapping(value, "voices store")
        require_version_one(root.get("version"))
        show = require_string(root.get("show"), "show", max_bytes=MAX_NAME_BYTES)
        normalize_show(show)
        revision = require_exact_int(root.get("revision"), "revision", minimum=0)
        _provenance, dim = validate_provenance(root.get("provenance"))
        identities = require_mapping(root.get("identities"), "identities")
        if len(identities) > MAX_IDENTITIES:
            _raise(f"identities may contain at most {MAX_IDENTITIES} entries")

        active_exemplar_ids: set[str] = set()
        namespace_owners: dict[str, set[str]] = {}
        for identity_id, raw_identity in identities.items():
            require_identity_id(identity_id)
            identity = require_mapping(raw_identity, f"identities.{identity_id}")
            display_name = require_string(
                identity.get("display_name"),
                f"identities.{identity_id}.display_name",
                max_bytes=MAX_NAME_BYTES,
            )
            aliases = _require_list(
                identity.get("aliases"), f"identities.{identity_id}.aliases"
            )
            if len(aliases) > MAX_ALIASES:
                _raise(f"identities.{identity_id}.aliases exceeds {MAX_ALIASES}")
            for position, raw_name in enumerate([display_name, *aliases]):
                name = normalize_speaker_key(
                    raw_name,
                    field=f"identities.{identity_id}.names[{position}]",
                )
                namespace_owners.setdefault(name, set()).add(identity_id)

            exemplars = _require_list(
                identity.get("exemplars"),
                f"identities.{identity_id}.exemplars",
            )
            if len(exemplars) > MAX_EXEMPLARS:
                _raise(f"identities.{identity_id}.exemplars exceeds {MAX_EXEMPLARS}")
            captures: set[str] = set()
            media: set[str] = set()
            episodes: set[str] = set()
            for exemplar_index, raw_exemplar in enumerate(exemplars):
                field = f"identities.{identity_id}.exemplars[{exemplar_index}]"
                exemplar = require_mapping(raw_exemplar, field)
                exemplar_id = require_exemplar_id(exemplar.get("id"), f"{field}.id")
                if exemplar_id in active_exemplar_ids:
                    _raise(f"duplicate active exemplar id {exemplar_id}")
                active_exemplar_ids.add(exemplar_id)
                capture = require_capture_id(
                    exemplar.get("capture_id"), f"{field}.capture_id"
                )
                source_media = require_sha256(
                    exemplar.get("media_fingerprint"),
                    f"{field}.media_fingerprint",
                )
                episode = normalize_episode(exemplar.get("episode"))
                require_utc_timestamp(exemplar.get("added"), f"{field}.added")
                validate_vector(
                    exemplar.get("vector"), dim=dim, field=f"{field}.vector"
                )
                if capture in captures:
                    _raise(f"duplicate capture id {capture} in identity {identity_id}")
                if source_media in media:
                    _raise(
                        f"duplicate source media {source_media} in identity {identity_id}"
                    )
                if episode in episodes:
                    _raise(f"duplicate episode {episode!r} in identity {identity_id}")
                captures.add(capture)
                media.add(source_media)
                episodes.add(episode)

        for owners in namespace_owners.values():
            if len(owners) > 1:
                owner_list = ", ".join(sorted(owners))
                _raise(f"normalized name namespace collision: {owner_list}")

        log = _require_list(root.get("log"), "log")
        if len(log) > MAX_LOG_ROWS:
            _raise(f"log may contain at most {MAX_LOG_ROWS} rows")
        for index, row in enumerate(log):
            _validate_log_row(row, index)
    except VoiceStoreError:
        raise
    except Phase2DataError as exc:
        raise VoiceStoreError(str(exc)) from exc

    return ValidatedVoiceStore(
        show=show,
        revision=revision,
        embedding_dim=dim,
        identities=identities,
    )


def new_voice_store(show: str, provenance: Mapping[str, object]) -> dict[str, object]:
    """Build an empty, validated store; creation remains an integration choice."""
    store: dict[str, object] = {
        "version": 1,
        "show": show,
        "revision": 0,
        "provenance": copy.deepcopy(dict(provenance)),
        "identities": {},
        "log": [],
    }
    validate_voice_store(store)
    return store


def voice_store_digest(value: Mapping[str, object]) -> str:
    validate_voice_store(value)
    return canonical_json_digest(value)


def load_voice_store(path: Path) -> tuple[dict[str, object], ValidatedVoiceStore]:
    raw = load_json_object(path, max_bytes=VOICES_STORE_MAX_BYTES)
    return raw, validate_voice_store(raw)


def canonical_store_path(path: Path) -> Path:
    """Store-side name for voicebase's shared realpath normalization."""
    return canonical_path(path)


def store_lock_path(path: Path) -> Path:
    return Path(f"{canonical_store_path(path)}.lock")


@contextmanager
def store_lock(path: Path, *, exclusive: bool) -> Iterator[StoreLockHandle]:
    """Take the canonical persistent flock and yield the one resolved target."""
    resolved = canonical_store_path(path)
    lock = Path(f"{resolved}.lock")
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, mode)
        yield StoreLockHandle(store_path=resolved, lock_path=lock)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def exclusive_store_lock(path: Path) -> AbstractContextManager[StoreLockHandle]:
    return store_lock(path, exclusive=True)


def shared_store_lock(path: Path) -> AbstractContextManager[StoreLockHandle]:
    return store_lock(path, exclusive=False)


def write_voice_store(path: Path, value: Mapping[str, object]) -> None:
    """Validate, preflight, and replace the real store target atomically."""
    validate_voice_store(value)
    write_json_object(
        canonical_store_path(path),
        value,
        max_bytes=VOICES_STORE_MAX_BYTES,
    )


def _identity_owners(
    identities: Mapping[str, object], normalized_name: str
) -> set[str]:
    owners: set[str] = set()
    for identity_id, raw_identity in identities.items():
        identity = require_mapping(raw_identity, f"identities.{identity_id}")
        display = cast(str, identity["display_name"])
        aliases = cast(list[object], identity["aliases"])
        for raw_name in [display, *aliases]:
            if normalize_speaker_key(raw_name) == normalized_name:
                owners.add(identity_id)
    return owners


def resolve_identity_id(store: Mapping[str, object], raw_name: str) -> str | None:
    """Resolve the joint display/alias namespace, or return None for creation."""
    validated = validate_voice_store(store)
    key = normalize_speaker_key(raw_name)
    owners = _identity_owners(validated.identities, key)
    if len(owners) > 1:
        raise EnrollmentRefusal(
            "speaker name is owned by multiple identities: " + ", ".join(sorted(owners))
        )
    return next(iter(owners), None)


def _default_identity_id() -> str:
    return f"v{secrets.token_hex(6)}"


def _default_exemplar_id() -> str:
    return f"x{secrets.token_hex(4)}"


def _mint_unique_id(
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
        return require_utc_timestamp(at, "enrollment timestamp")
    return utc_timestamp(at)


def _active_exemplar_ids(identities: Mapping[str, object]) -> set[str]:
    return {
        cast(str, exemplar["id"])
        for raw_identity in identities.values()
        for exemplar in cast(
            list[dict[str, object]], cast(dict[str, object], raw_identity)["exemplars"]
        )
    }


def _append_log(log: list[object], row: dict[str, object]) -> None:
    log.append(row)
    if len(log) > MAX_LOG_ROWS:
        del log[: len(log) - MAX_LOG_ROWS]


def enroll_exemplar(
    store: Mapping[str, object],
    *,
    raw_name: str,
    capture_id: str,
    media_fingerprint: str,
    episode: str,
    vector: object,
    replace_episode: bool = False,
    at: str | datetime | None = None,
    identity_id_factory: Callable[[], str] = _default_identity_id,
    exemplar_id_factory: Callable[[], str] = _default_exemplar_id,
) -> EnrollmentResult:
    """Apply the final capture→source-media→episode enrollment relation.

    The input mapping is never mutated.  Refusals therefore leave both its
    revision and its audit log unchanged; a true no-op returns an equal copy.
    """
    if type(replace_episode) is not bool:
        raise EnrollmentRefusal("replace_episode must be a boolean")
    validated = validate_voice_store(store)
    normalized_name = normalize_speaker_key(raw_name)
    incoming_capture = require_capture_id(capture_id)
    incoming_media = require_sha256(media_fingerprint, "media_fingerprint")
    incoming_episode = normalize_episode(episode)
    incoming_vector = list(
        validate_vector(vector, dim=validated.embedding_dim, field="incoming vector")
    )
    event_at = _event_time(at)

    result_store = cast(dict[str, object], copy.deepcopy(dict(store)))
    identities = cast(dict[str, object], result_store["identities"])
    owners = _identity_owners(identities, normalized_name)
    if len(owners) > 1:
        raise EnrollmentRefusal(
            "speaker name is owned by multiple identities: " + ", ".join(sorted(owners))
        )

    used_identity_ids = set(identities)
    identity_id = next(iter(owners), None)
    if identity_id is None:
        identity_id = _mint_unique_id(
            identity_id_factory,
            used_identity_ids,
            require_identity_id,
            "identity",
        )
        identities[identity_id] = {
            "display_name": raw_name,
            "aliases": [],
            "exemplars": [],
        }

    identity = cast(dict[str, object], identities[identity_id])
    exemplars = cast(list[dict[str, object]], identity["exemplars"])
    by_capture = {cast(str, item["capture_id"]): item for item in exemplars}
    by_media = {cast(str, item["media_fingerprint"]): item for item in exemplars}
    by_episode = {normalize_episode(item["episode"]): item for item in exemplars}
    capture_hit = by_capture.get(incoming_capture)
    media_hit = by_media.get(incoming_media)
    episode_hit = by_episode.get(incoming_episode)
    indexed_hits = [
        hit for hit in (capture_hit, media_hit, episode_hit) if hit is not None
    ]
    hit_ids = {cast(str, hit["id"]) for hit in indexed_hits}
    if len(hit_ids) > 1:
        raise CrossIndexRefusal(
            "enrollment evidence resolves different exemplars across indexes: "
            + ", ".join(sorted(hit_ids))
        )

    if capture_hit is not None:
        stored_vector = cast(list[object], capture_hit["vector"])
        stored_media = cast(str, capture_hit["media_fingerprint"])
        stored_episode = normalize_episode(capture_hit["episode"])
        if stored_vector != incoming_vector or stored_media != incoming_media:
            raise EnrollmentRefusal(
                f"capture integrity failure for {incoming_capture}: vector or media differs"
            )
        if stored_episode != incoming_episode:
            raise EnrollmentRefusal(
                f"capture {incoming_capture} is already enrolled as {stored_episode!r}"
            )
        return EnrollmentResult(
            store=result_store,
            outcome="noop",
            identity_id=identity_id,
            exemplar_id=cast(str, capture_hit["id"]),
        )

    replacement: dict[str, object] | None = None
    if media_hit is not None:
        stored_episode = normalize_episode(media_hit["episode"])
        if stored_episode != incoming_episode:
            raise EnrollmentRefusal(
                f"same source media is already enrolled as {stored_episode!r}"
            )
        if not replace_episode:
            raise EnrollmentRefusal(
                f"source media for {incoming_episode!r} is already enrolled; "
                "use --replace-episode"
            )
        replacement = media_hit
    elif episode_hit is not None:
        if not replace_episode:
            old_capture = cast(str, episode_hit["capture_id"])
            raise EnrollmentRefusal(
                f"episode {incoming_episode!r} already has capture {old_capture}; "
                "use --replace-episode"
            )
        replacement = episode_hit

    used_exemplar_ids = _active_exemplar_ids(identities)
    new_exemplar_id = _mint_unique_id(
        exemplar_id_factory,
        used_exemplar_ids,
        require_exemplar_id,
        "exemplar",
    )
    new_exemplar: dict[str, object] = {
        "id": new_exemplar_id,
        "vector": incoming_vector,
        "episode": incoming_episode,
        "capture_id": incoming_capture,
        "media_fingerprint": incoming_media,
        "added": event_at,
    }
    log = cast(list[object], result_store["log"])
    evicted_id: str | None = None

    if replacement is not None:
        old_id = cast(str, replacement["id"])
        old_capture = cast(str, replacement["capture_id"])
        replace_index = next(
            index for index, exemplar in enumerate(exemplars) if exemplar is replacement
        )
        exemplars[replace_index] = new_exemplar
        _append_log(
            log,
            {
                "at": event_at,
                "action": "replace",
                "identity": identity_id,
                "old_exemplar": old_id,
                "new_exemplar": new_exemplar_id,
                "episode": incoming_episode,
                "old_capture": old_capture,
                "new_capture": incoming_capture,
            },
        )
        outcome: Literal["enroll", "replace"] = "replace"
    else:
        if len(exemplars) >= MAX_EXEMPLARS:
            oldest = min(
                exemplars,
                key=lambda item: (cast(str, item["added"]), cast(str, item["id"])),
            )
            exemplars.remove(oldest)
            evicted_id = cast(str, oldest["id"])
            _append_log(
                log,
                {
                    "at": event_at,
                    "action": "evict",
                    "identity": identity_id,
                    "exemplar": evicted_id,
                    "episode": cast(str, oldest["episode"]),
                },
            )
        exemplars.append(new_exemplar)
        _append_log(
            log,
            {
                "at": event_at,
                "action": "enroll",
                "identity": identity_id,
                "exemplar": new_exemplar_id,
                "episode": incoming_episode,
            },
        )
        outcome = "enroll"

    result_store["revision"] = validated.revision + 1
    validate_voice_store(result_store)
    return EnrollmentResult(
        store=result_store,
        outcome=outcome,
        identity_id=identity_id,
        exemplar_id=new_exemplar_id,
        evicted_exemplar_id=evicted_id,
    )


__all__ = [
    "CrossIndexRefusal",
    "EnrollmentRefusal",
    "EnrollmentResult",
    "MAX_ALIASES",
    "MAX_EPISODE_BYTES",
    "MAX_EXEMPLARS",
    "MAX_IDENTITIES",
    "MAX_LOG_ROWS",
    "MAX_NAME_BYTES",
    "StoreLockHandle",
    "ValidatedVoiceStore",
    "VoiceStoreError",
    "canonical_store_path",
    "enroll_exemplar",
    "exclusive_store_lock",
    "load_voice_store",
    "new_voice_store",
    "normalize_episode",
    "normalize_show",
    "normalize_speaker_key",
    "resolve_identity_id",
    "shared_store_lock",
    "store_lock",
    "store_lock_path",
    "validate_voice_store",
    "voice_store_digest",
    "write_voice_store",
]
