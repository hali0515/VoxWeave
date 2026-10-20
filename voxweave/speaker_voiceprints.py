"""Exact-byte reads of the sibling, the voiceprints sidecar and the voices store.

Also holds the matching of one episode's voiceprints against a per-show store or
against the voice library.
"""

from __future__ import annotations

import logging
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from voxweave import artifacts, voicelibrary
from voxweave.voicebase import (
    VOICES_STORE_MAX_BYTES,
    VOICEPRINTS_MAX_BYTES,
    Phase2DataError,
    require_capture_id,
    require_sha256,
    strict_json_object_loads,
    validate_voiceprints_mapping,
    voiceprints_digest,
)
from voxweave.voicematch import (
    MatchThresholds,
    SpeakerMatch,
    ThresholdError,
    build_compatibility_fingerprint,
    build_library_suggest_record,
    build_suggest_record,
    compatibility_equal,
    describe_compatibility_mismatch,
    embedding_space_label,
    match_speakers,
    match_tiers,
    parse_global_suggest,
    parse_thresholds,
)
from voxweave.voicestore import (
    canonical_store_path,
    normalize_show,
    validate_voice_store,
)

log = logging.getLogger("voxweave")


def _read_exact_object(path: Path, max_bytes: int) -> tuple[bytes, dict[str, object]]:
    raw = Path(path).read_bytes()
    return raw, strict_json_object_loads(
        raw,
        max_bytes=max_bytes,
        source=Path(path).name,
    )


def _read_sibling_exact(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = Path(path).read_bytes()
    return raw, strict_json_object_loads(
        raw,
        max_bytes=max(1, len(raw)),
        source=Path(path).name,
    )


def _declared_voiceprint_pair(
    sibling: Mapping[str, object],
) -> tuple[str, str] | None:
    capture_present = "voiceprint_capture" in sibling
    media_present = "voiceprint_media" in sibling
    if not capture_present and not media_present:
        return None
    if not capture_present or not media_present:
        raise Phase2DataError("sibling voiceprint keys are not a complete pair")
    return (
        require_capture_id(
            sibling.get("voiceprint_capture"), "sibling.voiceprint_capture"
        ),
        require_sha256(sibling.get("voiceprint_media"), "sibling.voiceprint_media"),
    )


def _resolved_voices_path(media: Path, voices: Path) -> Path:
    """The canonical path of an explicit ``--voices`` per-show store."""
    raw = Path(voices)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    path = canonical_store_path(raw)
    media_parent = Path(os.path.realpath(os.fspath(media.parent)))
    if path.parent == media_parent and path.name.startswith(f"{media.stem}."):
        raise RuntimeError(
            f"voices store {path.name} is inside the {media.stem}.* episode namespace"
        )
    return path


def _load_generation_store(
    media: Path,
    *,
    voices: Path,
    show: str | None,
) -> tuple[Path, bytes, dict[str, object]]:
    """Read an explicit ``--voices`` store for matching; refuse an unusable one.

    (A per-folder ``voxweave.voices.json`` found beside the media is read by
    the voice-library path instead, see :func:`_read_legacy_store`.)
    """
    path = _resolved_voices_path(media, voices)
    if not path.exists():
        raise FileNotFoundError(f"explicit voices store not found: {path}")
    try:
        raw, store = _read_exact_object(path, VOICES_STORE_MAX_BYTES)
        validated = validate_voice_store(store)
    except (OSError, Phase2DataError) as exc:
        raise RuntimeError(f"explicit voices store {path} is unusable: {exc}") from exc
    if show is not None and normalize_show(show) != normalize_show(validated.show):
        raise RuntimeError(
            f"--show {show!r} does not match voices store show {validated.show!r}"
        )
    return path, raw, store


def _load_voiceprints_exact(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = Path(path).read_bytes()
    sidecar = _voiceprints_from_bytes(raw, source=Path(path).name)
    return raw, sidecar


def _voiceprints_from_bytes(raw: bytes, *, source: str) -> dict[str, object]:
    sidecar = strict_json_object_loads(
        raw,
        max_bytes=VOICEPRINTS_MAX_BYTES,
        source=source,
    )
    validate_voiceprints_mapping(sidecar)
    return sidecar


def _matching_record(
    sidecar: Mapping[str, object],
    store_path: Path,
    store: Mapping[str, object],
) -> tuple[dict[str, SpeakerMatch], dict[str, object], MatchThresholds] | None:
    sidecar_provenance = cast(Mapping[str, object], sidecar["provenance"])
    store_provenance = cast(Mapping[str, object], store["provenance"])
    left = build_compatibility_fingerprint(sidecar_provenance)
    right = build_compatibility_fingerprint(store_provenance)
    if not compatibility_equal(left, right):
        # One line per invocation, not per speaker: the whole store is out.
        log.warning(
            "matching against voices store %s skipped: %s",
            store_path,
            describe_compatibility_mismatch(
                sidecar_provenance,
                store_provenance,
                episode=left,
                store=right,
            ),
        )
        return None
    if sidecar_provenance.get("torch_version") != store_provenance.get("torch_version"):
        log.warning("voice evidence was produced by a different torch version")
    try:
        # Both sides share one embedding space here, so the sidecar's
        # provenance picks that space's default thresholds.
        thresholds = parse_thresholds(provenance=sidecar_provenance)
    except ThresholdError as exc:
        log.warning("voice matching thresholds are invalid; matching skipped: %s", exc)
        return None
    centroids = cast(Mapping[str, object], sidecar["speakers"])
    matches = match_speakers(centroids, store, thresholds)
    record = build_suggest_record(
        matches,
        capture_id=cast(str, sidecar["capture_id"]),
        voiceprints_content_digest=voiceprints_digest(sidecar),
        compatibility=left,
        thresholds=thresholds,
        store_path=store_path,
        store=store,
    )
    return matches, record, thresholds


_WARNED_LEGACY_STORES: set[Path] = set()


def _warn_legacy_store_once(path: Path) -> None:
    if path in _WARNED_LEGACY_STORES:
        return
    _WARNED_LEGACY_STORES.add(path)
    log.warning(
        "found a per-folder voices store %s; it is still read for suggestions, "
        "but new voices go to the voice library. Run `voxweave voices import %s` "
        "to merge it",
        path,
        shlex.quote(str(path)),
    )


def _read_legacy_store(
    path: Path, sidecar_provenance: Mapping[str, object]
) -> dict[str, object] | None:
    """Read a pre-library per-folder store read-only, if it can serve matching."""
    if not artifacts.path_present(path):
        return None
    try:
        # Read-only is enough (a media folder on a read-only share still counts).
        _resolved, store = voicelibrary.read_legacy_store(path)
    except (OSError, Phase2DataError) as exc:
        log.warning(
            "per-folder voices store %s is unusable; ignoring it: %s", path, exc
        )
        return None
    store_provenance = cast(Mapping[str, object], store["provenance"])
    left = build_compatibility_fingerprint(sidecar_provenance)
    right = build_compatibility_fingerprint(store_provenance)
    if not compatibility_equal(left, right):
        log.warning(
            "matching against voices store %s skipped: %s",
            path,
            describe_compatibility_mismatch(
                sidecar_provenance, store_provenance, episode=left, store=right
            ),
        )
        return None
    return store


def _library_matching(
    sidecar: Mapping[str, object],
    *,
    media: Path,
    location: voicelibrary.LibraryLocation,
    scope: str,
    folder_scope: bool,
) -> tuple[dict[str, SpeakerMatch], dict[str, object]] | None:
    """Two-tier suggestions from the voice library (plus a legacy folder store).

    Tier 1 holds identities enrolled under ``scope``; tier 2 every other
    scope, above the stricter global bar. A pre-library ``voxweave.voices.json``
    beside the media is read, never written: its identities join tier 1 when
    the scope is this folder (or equals the store's show), else tier 2, and
    identities the library already holds are left to the library copy.
    """
    root = location.root
    provenance = cast(Mapping[str, object], sidecar["provenance"])
    try:
        space_name, fingerprint = voicelibrary.space_identity(provenance)
    except Phase2DataError as exc:
        log.warning("voice library matching skipped: %s", exc)
        return None
    if not location.default and not root.is_dir():
        # A missing default library is simply empty; a configured one that
        # is missing is usually a network share that is not mounted.
        log.warning(
            "voice library %s (set by %s) does not exist yet, or its network "
            "share is not mounted; suggestions come only from a per-folder store",
            root,
            location.source,
        )
    try:
        with voicelibrary.library_lock(root, exclusive=False):
            state = voicelibrary.read_state(root, spaces=[space_name])
            voicelibrary.require_same_space(state, space_name, fingerprint)
            other_spaces = [
                name
                for name in voicelibrary.list_space_names(
                    voicelibrary.LibraryPaths(root)
                )
                if name != space_name
            ]
    except (OSError, Phase2DataError) as exc:
        log.warning(
            "voice library %s is unusable; library matching skipped: %s", root, exc
        )
        state = voicelibrary.LibraryState(
            voicelibrary.LibraryPaths(root), voicelibrary.empty_identities(), {}
        )
        other_spaces = []
    if space_name not in state.spaces and other_spaces:
        log.warning(
            "voice library %s holds no voices captured with %s (it has: %s); "
            "enroll this episode to start that space",
            root,
            embedding_space_label(provenance),
            ", ".join(other_spaces),
        )

    pools = voicelibrary.matching_pools(state, space_name, scope)
    legacy_path = canonical_store_path(voicelibrary.legacy_store_path(media))
    legacy_store = _read_legacy_store(legacy_path, provenance)
    if legacy_store is not None:
        legacy_show = voicelibrary.normalize_scope(legacy_store["show"])
        pools, unimported = voicelibrary.add_legacy_store(
            pools,
            legacy_store,
            state=state,
            space_name=space_name,
            in_scope=folder_scope or legacy_show == scope,
            store_path=legacy_path,
        )
        if unimported:
            _warn_legacy_store_once(legacy_path)
    if not pools.in_scope and not pools.other_scopes:
        return None
    try:
        thresholds = parse_thresholds(provenance=provenance)
        global_suggest = parse_global_suggest(
            provenance=provenance, suggest=thresholds.suggest
        )
    except ThresholdError as exc:
        log.warning("voice matching thresholds are invalid; matching skipped: %s", exc)
        return None
    matches = match_tiers(
        cast(Mapping[str, object], sidecar["speakers"]),
        in_scope=pools.in_scope,
        other_scopes=pools.other_scopes,
        scopes=pools.scopes,
        embedding_dim=cast(int, provenance["embedding_dim"]),
        thresholds=thresholds,
        global_suggest=global_suggest,
    )
    record = build_library_suggest_record(
        matches,
        capture_id=cast(str, sidecar["capture_id"]),
        voiceprints_content_digest=voiceprints_digest(sidecar),
        compatibility=build_compatibility_fingerprint(provenance),
        thresholds=thresholds,
        global_suggest=global_suggest,
        library_path=root,
        scope=scope,
        revision=voicelibrary.space_revision(state, space_name),
        content_digest=voicelibrary.match_input_digest(state, space_name, legacy_store),
    )
    return matches, record
