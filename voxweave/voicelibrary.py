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
import fcntl
import json
import logging
import os
import re
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from voxweave import fsio
from voxweave.voicebase import (
    Phase2DataError,
    canonical_json_bytes,
    encode_json_bytes,
    strict_json_object_loads,
    utc_timestamp,
)
from voxweave.voicelib_schema import (
    _as_library_error as _as_library_error,
    _BONUS_FOLDER_RE as _BONUS_FOLDER_RE,
    _CJK_SEASON_RE as _CJK_SEASON_RE,
    _list_space as _list_space,
    _NUMBERED_FOLDER_RE as _NUMBERED_FOLDER_RE,
    _reject_vectors as _reject_vectors,
    _require_list as _require_list,
    _SLUG_RE as _SLUG_RE,
    _SPACE_NAME_RE as _SPACE_NAME_RE,
    _validate_source as _validate_source,
    DEFAULT_SCOPE as DEFAULT_SCOPE,
    default_voices_dir as default_voices_dir,
    empty_identities as empty_identities,
    ENV_VOICES_DIR as ENV_VOICES_DIR,
    episode_scope as episode_scope,
    forgotten_ids as forgotten_ids,
    HISTORY_ACTIONS as HISTORY_ACTIONS,
    HISTORY_NAME as HISTORY_NAME,
    history_row as history_row,
    IDENTITIES_NAME as IDENTITIES_NAME,
    is_generic_folder_name as is_generic_folder_name,
    LEGACY_STORE_NAME as LEGACY_STORE_NAME,
    legacy_store_path as legacy_store_path,
    LibraryChange as LibraryChange,
    LibraryConflict as LibraryConflict,
    LibraryLocation as LibraryLocation,
    LibraryPaths as LibraryPaths,
    LibraryState as LibraryState,
    listed_spaces as listed_spaces,
    LOCK_NAME as LOCK_NAME,
    MAX_FORGOTTEN as MAX_FORGOTTEN,
    MAX_LIBRARY_IDENTITIES as MAX_LIBRARY_IDENTITIES,
    MAX_MEDIA_PATH_BYTES as MAX_MEDIA_PATH_BYTES,
    MAX_SCOPES as MAX_SCOPES,
    MAX_SLUG_CHARS as MAX_SLUG_CHARS,
    MAX_SPACES as MAX_SPACES,
    new_space as new_space,
    normalize_scope as normalize_scope,
    read_legacy_store as read_legacy_store,
    resolve_voices_dir as resolve_voices_dir,
    scoped_episode as scoped_episode,
    space_identity as space_identity,
    space_model_slug as space_model_slug,
    SPACES_DIRNAME as SPACES_DIRNAME,
    UnknownIdentity as UnknownIdentity,
    validate_identities as validate_identities,
    validate_relations as validate_relations,
    validate_space as validate_space,
    ValidatedSpace as ValidatedSpace,
    VoiceLibraryError as VoiceLibraryError,
)
from voxweave.voicelib_transitions import (
    _add_scopes as _add_scopes,
    _exemplar_keys as _exemplar_keys,
    _identity_or_raise as _identity_or_raise,
    _names_of as _names_of,
    _older_than_retained as _older_than_retained,
    _space_for as _space_for,
    _used_exemplar_ids as _used_exemplar_ids,
    add_legacy_store as add_legacy_store,
    default_import_scopes as default_import_scopes,
    describe_identity as describe_identity,
    enroll_entries as enroll_entries,
    EnrollEntry as EnrollEntry,
    EnrollOutcome as EnrollOutcome,
    EpisodeSource as EpisodeSource,
    find_identities as find_identities,
    forget_identity as forget_identity,
    identities_named as identities_named,
    identity_summary as identity_summary,
    import_store as import_store,
    ImportSummary as ImportSummary,
    LibraryPlan as LibraryPlan,
    list_identities as list_identities,
    match_input_digest as match_input_digest,
    matching_pools as matching_pools,
    MatchPools as MatchPools,
    plan_library_enrollment as plan_library_enrollment,
    rename_identity as rename_identity,
    require_same_space as require_same_space,
    space_revision as space_revision,
)


log = logging.getLogger("voxweave")


IDENTITIES_MAX_BYTES = 16 * 1024 * 1024


SPACE_MAX_BYTES = 64 * 1024 * 1024


FIRST_WRITE_NOTICE = (
    "voice library created at %s: it stores voice biometrics of the people you "
    "name; `voxweave voices forget ID` removes one person, deleting the "
    "directory removes everyone"
)


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


def _drop_orphans(
    identities: Mapping[str, object], spaces: Mapping[str, dict[str, object]]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, list[object]]]]:
    """Remove, in place, the exemplars of identities that do not exist.

    Returns their per-space counts and the removed exemplars. They are left
    by a forget racing a writer on a mount without working locks, by a
    manual edit, or by an ``identities.json`` rolled back to an older copy
    (a partial backup restore, a sync conflict). Refusing to read the
    library over them would also block the ``forget`` that removes them, so
    readers skip them. Writers delete only those of forgotten (tombstoned)
    ids; the others are kept until ``identities.json`` is restored (see
    :func:`commit`). A missing ``identities.json`` is refused by
    :func:`read_state` instead (see :func:`_refuse_missing_identities`).
    """
    known = cast(Mapping[str, object], identities["identities"])
    tombstones = set(forgotten_ids(identities))
    dropped: dict[str, dict[str, int]] = {}
    removed: dict[str, dict[str, list[object]]] = {}
    for name, document in spaces.items():
        exemplars = cast(dict[str, list[object]], document["exemplars"])
        for identity_id in sorted(set(exemplars) - set(known)):
            items = exemplars.pop(identity_id)
            dropped.setdefault(name, {})[identity_id] = len(items)
            removed.setdefault(name, {})[identity_id] = items
        if name in dropped:
            kept = {i: n for i, n in dropped[name].items() if i not in tombstones}
            if kept:
                log.warning(
                    "voice library space %s holds %d voice sample(s) of %d "
                    "identities that %s does not list (it may have been "
                    "restored from an older copy, or edited by hand); they are "
                    "ignored but kept. Restore %s to use them again; to delete "
                    "them, restore it and run `voxweave voices forget` on those "
                    "ids, or remove them from %s/%s.json by hand",
                    name,
                    sum(kept.values()),
                    len(kept),
                    IDENTITIES_NAME,
                    IDENTITIES_NAME,
                    SPACES_DIRNAME,
                    name,
                )
            if len(kept) < len(dropped[name]):
                log.warning(
                    "voice library space %s holds %d voice sample(s) of "
                    "forgotten identities; they are ignored, and the next "
                    "command that changes the library deletes them",
                    name,
                    sum(n for i, n in dropped[name].items() if i in tombstones),
                )
    return dropped, removed


def _refuse_missing_identities(
    paths: LibraryPaths,
    loaded: Mapping[str, dict[str, object]],
    unread: Iterable[str],
) -> None:
    """Refuse a library whose ``identities.json`` is gone but whose samples are not.

    Read as an empty library, every voice sample would be an orphan, and the
    next enrollment or import would delete them all. ``loaded`` holds the
    spaces already read; the space files named in ``unread`` are read here.
    """
    samples = 0
    for name in sorted({*loaded, *unread}):
        document = loaded.get(name)
        if document is None:
            raw = _read_bounded(paths.space(name), SPACE_MAX_BYTES)
            if raw is None:
                continue
            document = _decode(
                raw, source=f"{SPACES_DIRNAME}/{name}.json", max_bytes=SPACE_MAX_BYTES
            )
            validate_space(document, name=name)
        exemplars = cast(Mapping[str, Sequence[object]], document["exemplars"])
        samples += sum(len(items) for items in exemplars.values())
    if samples:
        raise VoiceLibraryError(
            f"voice library {paths.root}: {IDENTITIES_NAME} is missing, but "
            f"{SPACES_DIRNAME}/ holds {samples} voice sample(s); without it they "
            f"belong to no one and the next change would delete them. Restore "
            f"{IDENTITIES_NAME} (from a backup, or wherever it was moved) to use "
            "the library; nothing was deleted"
        )


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
    empty one; a missing ``identities.json`` beside space files that still
    hold voice samples is refused.
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
    if raw is None:
        _refuse_missing_identities(
            paths, loaded, () if spaces is None else list_space_names(paths)
        )
    if missing:
        log.warning(
            "voice library %s lists space file(s) that cannot be found: %s",
            paths.root,
            ", ".join(missing),
        )
    orphans, orphan_exemplars = _drop_orphans(identities, loaded)
    state = LibraryState(
        paths,
        identities,
        loaded,
        observed,
        spaces is None,
        orphans=orphans,
        missing_spaces=tuple(missing),
        orphan_exemplars=orphan_exemplars,
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

    A library belongs to one account: every file in it (the lock included) is
    created 0o600, and libraries shared between user accounts are not
    supported (several machines can share one through the same account). A
    pre-existing directory keeps the mode its owner gave it; the files inside
    are private either way. Missing parents are created only with
    ``parents`` (the built-in location); otherwise they are refused, since a
    missing parent of a configured library usually is a network share that is
    not mounted.
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

    fsio.atomic_write_text(path, payload, before_replace=still_expected, private=True)


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
    with fsio.atomic_path(paths.history, private=True) as temporary:
        with open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def _heal_orphans(state: LibraryState, change: LibraryChange) -> LibraryChange:
    # Phase 1 of commit: returns ``change`` itself, or a copy that also
    # deletes the orphans of forgotten ids; the heal history rows go first.
    healed_spaces: dict[str, dict[str, object]] = {}
    healed_history: list[dict[str, object]] = []
    tombstones = state.forgotten
    for name, all_dropped in sorted(state.orphans.items()):
        # Delete the orphans of forgotten ids (a space the change rewrites
        # anyway was read without them). Those of other ids are carried
        # through below: identities.json may be an older copy.
        dropped = {i: n for i, n in all_dropped.items() if i in tombstones}
        if not dropped:
            continue
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
    return change


def _validate_change(state: LibraryState, change: LibraryChange) -> None:
    # Phase 2 of commit: validates the healed change before orphans are carried.
    for row in change.history:
        _reject_vectors(row)
    identities = change.identities or state.identities
    validate_identities(identities)
    merged_spaces = dict(state.spaces)
    for name, document in change.spaces.items():
        validate_space(document, name=name)
        merged_spaces[name] = document
    validate_relations(
        LibraryState(state.paths, identities, merged_spaces, state.observed)
    )


def _carry_kept_orphans(state: LibraryState, change: LibraryChange) -> LibraryChange:
    # Phase 3 of commit: rewritten spaces keep the orphans of ids that
    # identities.json does not list (it may be an older copy).
    carried: dict[str, dict[str, object]] = {}
    for name, document in change.spaces.items():
        kept = state.kept_orphans(name)
        if not kept:
            continue
        written = copy.deepcopy(document)
        exemplars = cast(dict[str, list[object]], written["exemplars"])
        clash = sorted(set(kept) & set(exemplars))
        if clash:
            raise VoiceLibraryError(
                f"{SPACES_DIRNAME}/{name}.json holds "
                f"{sum(len(items) for items in kept.values())} voice sample(s) "
                f"of identities {IDENTITIES_NAME} does not list, and this "
                f"change would add samples under the same id ({', '.join(clash)}). "
                f"Restore {IDENTITIES_NAME} (it may be an older copy); nothing "
                "was changed"
            )
        exemplars.update(copy.deepcopy(kept))
        validate_space(written, name=name)
        carried[name] = written
    if carried:
        change = dataclasses.replace(change, spaces={**change.spaces, **carried})
    return change


def _encode_writes(
    state: LibraryState, change: LibraryChange
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    # Phase 4 of commit: encodes every document (size limits raise here) and
    # returns (writes in order, space writes) before any file is replaced.
    paths = state.paths
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
    return writes, space_writes


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
    change = _heal_orphans(state, change)
    _validate_change(state, change)
    change = _carry_kept_orphans(state, change)
    writes, space_writes = _encode_writes(state, change)

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
