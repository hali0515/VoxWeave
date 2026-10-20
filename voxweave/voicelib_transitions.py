"""Voice library transitions: pure functions from a state to a change."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from voxweave.voicebase import (
    IDENTITY_ID_RE,
    Phase2DataError,
    canonical_json_digest,
    require_capture_id,
    require_exemplar_id,
    require_identity_id,
    require_sha256,
    require_string,
    validate_vector,
)
from voxweave.voicestore import (
    CrossIndexRefusal,
    EnrollmentRefusal,
    ExemplarKey,
    MAX_EXEMPLARS,
    MAX_NAME_BYTES,
    event_time,
    mint_unique_id,
    new_exemplar_id,
    new_identity_id,
    normalize_episode,
    normalize_speaker_key,
    plan_indexed_enrollment,
    validate_voice_store,
)
from voxweave.voicelib_schema import (
    _list_space,
    episode_scope,
    forgotten_ids,
    history_row,
    LEGACY_STORE_NAME,
    LibraryChange,
    LibraryState,
    MAX_FORGOTTEN,
    MAX_MEDIA_PATH_BYTES,
    MAX_SCOPES,
    new_space,
    normalize_scope,
    scoped_episode,
    space_identity,
    UnknownIdentity,
    validate_space,
    VoiceLibraryError,
)


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
    identity_id_factory: Callable[[], str] = new_identity_id,
    exemplar_id_factory: Callable[[], str] = new_exemplar_id,
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
    event_at = event_time(at, "event timestamp")
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
    orphan_identities, orphan_exemplars = state.orphan_ids(space_name)
    used_exemplars = _used_exemplar_ids(space) | orphan_exemplars
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
            identity_id = mint_unique_id(
                identity_id_factory,
                set(identities) | orphan_identities,
                require_identity_id,
                "identity",
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

        exemplar_id = mint_unique_id(
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
    event_at = event_time(at, "event timestamp")
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
    event_at = event_time(at, "event timestamp")
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
    exemplar_id_factory: Callable[[], str] = new_exemplar_id,
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
    event_at = event_time(at, "event timestamp")
    provenance = cast(Mapping[str, object], store["provenance"])
    space_name, space = _space_for(state, provenance)
    identities_document = copy.deepcopy(state.identities)
    identities = cast(dict[str, dict[str, object]], identities_document["identities"])
    exemplars_by_identity = cast(dict[str, list[dict[str, object]]], space["exemplars"])
    used_exemplars = _used_exemplar_ids(space) | state.orphan_ids(space_name)[1]
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
                # replace_episode only tells a replacement apart: an import
                # never replaces a library sample, it reports the conflict.
                plan = plan_indexed_enrollment(
                    keys,
                    capture_id=cast(str, legacy_exemplar["capture_id"]),
                    media_fingerprint=cast(str, legacy_exemplar["media_fingerprint"]),
                    episode=scoped_episode(scope, episode),
                    vector=vector,
                    replace_episode=True,
                )
            except EnrollmentRefusal as exc:
                refused.append(f"{identity_id}/{legacy_exemplar['id']}: {exc}")
                continue
            if plan.outcome == "noop":
                present_count += 1
                continue
            if plan.outcome == "replace":
                refused.append(
                    f"{identity_id}/{legacy_exemplar['id']}: conflicts with a voice "
                    f"sample already in the library (episode {episode!r}, scope "
                    f"{scope!r}); not imported"
                )
                continue
            legacy_id = cast(str, legacy_exemplar["id"])
            exemplar_id = (
                legacy_id
                if legacy_id not in used_exemplars
                else mint_unique_id(
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
    store_path: Path | None = None,
) -> tuple[MatchPools, bool]:
    """Add a read-only pre-library store's identities to ``pools``.

    Identities the library already holds are skipped (the library copy wins),
    and so are identities forgotten in this library. Returns the new pools
    and whether ``voxweave voices import`` of the store would add anything
    (the cue to suggest it): an identity or a voice sample the library lacks
    and the import keeps. That is a dry run of :func:`import_store` under the
    scopes the import defaults to for ``store_path`` (the store's show without
    it), so samples the import refuses (they conflict with the library) or
    supersedes (older than every sample of an identity at the cap) do not
    count, and the hint cannot outlive a successful import.
    """
    validated = validate_voice_store(store)
    show = normalize_scope(validated.show)
    target = dict(pools.in_scope if in_scope else pools.other_scopes)
    scopes = dict(pools.scopes)
    tombstones = state.forgotten
    for identity_id, raw in validated.identities.items():
        if identity_id in tombstones:
            continue  # forgotten in this library: never offered again
        identity = cast(Mapping[str, object], raw)
        exemplars = cast(list[Mapping[str, object]], identity["exemplars"])
        if identity_id in state.identity_map or not exemplars:
            continue
        target[identity_id] = {
            "display_name": identity["display_name"],
            "origin": "legacy",
            "exemplars": [
                {"id": item["id"], "vector": item["vector"]} for item in exemplars
            ],
        }
        scopes[identity_id] = (show,)
    import_scopes = (
        (show,)
        if store_path is None
        else default_import_scopes(store_path, validated.show)
    )
    try:
        _change, summary = import_store(
            state,
            store,
            scope=import_scopes[0],
            also_scopes=import_scopes[1:],
            source_label="(dry run)",
        )
    except Phase2DataError:
        # The import would stop on the same error: nothing to suggest.
        unimported = False
    else:
        unimported = bool(summary.identities_created or summary.exemplars_added)
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
