"""Speaker-name mapping, audition snippets, and display metadata helpers.

Phase 1 deliberately keeps names out of the transcription sibling JSON.  The
per-media speaker mapping (or a legacy adjacent ``.speakers.json`` sidecar) maps
diarizer ids to display names, while VTT voice tags are parsed into transient
cue metadata before any text-processing stage.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from voxweave import artifacts, fsio, sidecars, voicelibrary
from voxweave.chunking import FFMPEG_TIMEOUT
from voxweave.core.overlay import spans_in, turns_in
from voxweave.mediasnapshot import MediaSnapshot, SnapshotUnavailable
from voxweave.paths import swap_ext
from voxweave.voicebase import (
    VOICES_STORE_MAX_BYTES,
    Phase2DataError,
    ValidatedVoiceprints,
    media_fingerprint,
    sanitize_speaker_name,
    strict_turn_projection,
    strict_json_object_loads,
    validate_voiceprint_conjunction,
    voiceprints_digest,
)
from voxweave.voiceepisode import episode_lock
from voxweave.voicematch import (
    CompatibilityError,
    SpeakerMatch,
    build_compatibility_fingerprint,
    compatibility_equal,
    delete_suggest,
    describe_compatibility_mismatch,
    load_suggest,
    write_suggest,
)
from voxweave.voicestore import (
    MAX_EXEMPLARS,
    EnrollmentRefusal,
    enroll_exemplar,
    exclusive_store_lock,
    new_voice_store,
    normalize_episode,
    normalize_show,
    normalize_speaker_key,
    resolve_identity_id,
    shared_store_lock,
    validate_voice_store,
    write_voice_store,
)

# The span arithmetic and clip selection live in voxweave.speaker_snippets. These are
# value copies kept for existing importers (voxweave.speakerserve reads them from
# this module at call time).
from voxweave.speaker_snippets import (
    MAX_SNIPPET_S as MAX_SNIPPET_S,
    MAX_SNIPPETS_PER_SPEAKER as MAX_SNIPPETS_PER_SPEAKER,
    MIN_SNIPPET_GAP_S as MIN_SNIPPET_GAP_S,
    MIN_SNIPPET_S as MIN_SNIPPET_S,
    Span as Span,
    Turn as Turn,
    _intersect_spans as _intersect_spans,
    _merge_spans as _merge_spans,
    _pick_spread as _pick_spread,
    _subtract_spans as _subtract_spans,
    _valid_spans as _valid_spans,
    _window as _window,
    select_snippets as select_snippets,
)

# The speaker-name decoding and voice-tag helpers live in voxweave.speaker_labels.
# These are value copies kept for existing importers: a test that patches one of
# these helpers must patch the module whose code looks the name up. Production
# importers deliberately keep importing from this module.
from voxweave.speaker_labels import (
    MAPPING_VERSION as MAPPING_VERSION,
    SpeakerLine as SpeakerLine,
    _VOICE_SPAN_RE as _VOICE_SPAN_RE,
    _VOICE_TAG_RE as _VOICE_TAG_RE,
    _VOICE_WRAP_RE as _VOICE_WRAP_RE,
    _mapping_entries as _mapping_entries,
    _mapping_entries_bytes as _mapping_entries_bytes,
    _normalized_name as _normalized_name,
    _voice_name as _voice_name,
    load_speaker_display_names as load_speaker_display_names,
    load_speaker_mapping_bytes as load_speaker_mapping_bytes,
    named_speaker_ids as named_speaker_ids,
    sanitize_ass_speaker_name as sanitize_ass_speaker_name,
    speaker_layout as speaker_layout,
    speaker_metadata as speaker_metadata,
    strip_srt_speaker_prefixes as strip_srt_speaker_prefixes,
    strip_voice_tags as strip_voice_tags,
    voice_tag_text as voice_tag_text,
    voice_text_for_block as voice_text_for_block,
    voice_text_for_ids as voice_text_for_ids,
)

# The audition page renderer lives in voxweave.speaker_audition_page. These are
# value copies kept for existing importers; create_speaker_audition still looks
# _render_audition_html up in this module.
from voxweave.speaker_audition_page import (
    _render_audition_html as _render_audition_html,
    _suggestion_button as _suggestion_button,
    _tiered_suggestions as _tiered_suggestions,
)

# The exact-byte reads and the store and library matching live in
# voxweave.speaker_voiceprints. These are value copies kept for existing importers.
# _WARNED_LEGACY_STORES is the one set object of that module. create_speaker_audition
# looks _matching_record and _load_generation_store up here, so a test that patches
# one of them patches this module.
from voxweave.speaker_voiceprints import (
    _WARNED_LEGACY_STORES as _WARNED_LEGACY_STORES,
    _declared_voiceprint_pair as _declared_voiceprint_pair,
    _library_matching as _library_matching,
    _load_generation_store as _load_generation_store,
    _load_voiceprints_exact as _load_voiceprints_exact,
    _matching_record as _matching_record,
    _read_exact_object as _read_exact_object,
    _read_legacy_store as _read_legacy_store,
    _read_sibling_exact as _read_sibling_exact,
    _resolved_voices_path as _resolved_voices_path,
    _voiceprints_from_bytes as _voiceprints_from_bytes,
    _warn_legacy_store_once as _warn_legacy_store_once,
)

log = logging.getLogger("voxweave")


@dataclass(frozen=True, slots=True)
class SpeakerAudition:
    """One in-memory audition page and its authoritative episode paths."""

    page: str
    media_path: Path
    sibling_json_path: Path
    mapping_path: Path
    speaker_ids: tuple[str, ...]
    pristine_mapping_generation: fsio.FileGeneration | None = None


def build_clip_command(
    media: Path, start: float, end: float, output: Path
) -> list[str]:
    """Build one 16 kHz mono MP3 clip using ffmpeg's default audio stream.

    The transcription decoder also leaves stream choice to ffmpeg; keeping the
    same selection rule is essential for multi-audio media.
    """
    duration = float(end) - float(start)
    if duration <= 0:
        raise ValueError("speaker snippet end must be after start")
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start):.3f}",
        "-i",
        str(media),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "32k",
        str(output),
    ]


def run_clip_command(cmd: list[str]) -> None:
    """Execute a clip command with the repository's non-interactive contract."""
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFMPEG_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found; install ffmpeg and make sure it is on your PATH"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg speaker clip timed out after {FFMPEG_TIMEOUT:g}s"
        ) from exc
    if proc.returncode != 0:
        detail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}){suffix}")


def extract_clip(media: Path, start: float, end: float, output: Path) -> None:
    """Extract one clip atomically; command construction remains separately testable."""
    with fsio.atomic_path(output, private=True) as tmp:
        run_clip_command(build_clip_command(media, start, end, tmp))


def _delete_stale_suggest_for_refusal(
    media: Path,
    *,
    sibling_path: Path,
    sibling_bytes: bytes | None,
) -> None:
    with episode_lock(media):
        if artifacts.path_present(sidecars.speakers_mapping_path(media)):
            return
        if sibling_bytes is None:
            # Initial sibling read failures have no exact observation to
            # compare. The episode lock and absent completion marker exclude a
            # cooperating successful generator, so any suggestion is stale.
            delete_suggest(sidecars.speakers_suggest_path(media))
            return
        try:
            unchanged = sibling_path.read_bytes() == sibling_bytes
        except OSError:
            return
        if unchanged:
            delete_suggest(sidecars.speakers_suggest_path(media))


def _publish_audition(
    media: Path,
    *,
    mapping_path: Path,
    skeleton: Mapping[str, object],
    suggest_record: Mapping[str, object] | None,
    before_mapping_install: Callable[[], None] | None = None,
) -> fsio.FileGeneration | None:
    """Publish state and identify the exact skeleton generation this call installed."""
    if suggest_record is None:
        legacy_mapping = artifacts.legacy_path(media, ".speakers.json")
        legacy_suggest = artifacts.legacy_path(media, ".speakers.suggest.json")
        try:
            cached = artifacts.inspect_paths(media)
        except artifacts.ArtifactMarkerError:
            if mapping_path != legacy_mapping:
                raise
            cached = None
        suggest_paths = tuple(
            dict.fromkeys(
                path
                for path in (
                    legacy_suggest,
                    None if cached is None else cached.speaker_suggest,
                )
                if path is not None
            )
        )
    else:
        suggest_paths = (sidecars.speakers_suggest_path(media),)

    def delete_suggestions() -> None:
        for path in suggest_paths:
            delete_suggest(path)

    def check_authority_before_install() -> None:
        assert before_mapping_install is not None
        try:
            before_mapping_install()
        except BaseException:
            # The callback runs before this invocation can install the
            # protected mapping. These replaceable outputs therefore remain
            # ours to remove even if another actor created a mapping meanwhile.
            delete_suggestions()
            raise

    try:
        if suggest_record is None:
            delete_suggestions()
        else:
            write_suggest(suggest_paths[0], suggest_record)
        if mapping_path.exists():
            if before_mapping_install is not None:
                check_authority_before_install()
            return None
        try:
            installed_generation = fsio.atomic_write_text_new(
                mapping_path,
                json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n",
                before_install=(
                    check_authority_before_install
                    if before_mapping_install is not None
                    else None
                ),
                private=True,
            )
        except FileExistsError:
            # A concurrent editor or generator won the protected install.
            # Its mapping is authoritative user data and remains untouched.
            return None
        return installed_generation
    except BaseException:
        # These outputs are wholly regenerable and owned by this invocation.
        # The protected mapping either does not exist or was created atomically
        # as the final successful edge.
        if not mapping_path.exists():
            delete_suggestions()
        raise


def create_speaker_audition(
    media: Path,
    *,
    voices: Path | None = None,
    voices_dir: Path | None = None,
    show: str | None = None,
    no_match: bool = False,
) -> SpeakerAudition:
    """Build an in-memory audition and install a mapping skeleton if absent.

    Suggestions come from the voice library (``voices_dir``, else its
    configured location) under the episode's scope (``show``, else the media
    folder's name), unless ``voices`` names an explicit per-show store, which
    keeps its pre-library behavior.
    """
    if voices is not None and voices_dir is not None:
        raise ValueError("use either a voices store or a voice library, not both")
    media = Path(media)
    if not media.is_file():
        raise FileNotFoundError(f"media file not found: {media}")
    quoted_media = shlex.quote(str(media))
    manual_hint = f"use `voxweave speakers serve {quoted_media} --manual`"
    json_path = swap_ext(media, ".json")
    sibling_bytes: bytes | None = None
    try:
        if not json_path.exists():
            raise FileNotFoundError(
                f"sibling transcript {json_path.name} not found; run voxweave {quoted_media} --diarize first"
            )
        try:
            sibling_bytes = json_path.read_bytes()
            data = json.loads(sibling_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid sibling JSON in {json_path.name}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"invalid sibling JSON in {json_path.name}: expected object"
            )
    except BaseException:
        _delete_stale_suggest_for_refusal(
            media,
            sibling_path=json_path,
            sibling_bytes=sibling_bytes,
        )
        raise
    assert sibling_bytes is not None

    declared = "voiceprint_capture" in data or "voiceprint_media" in data
    pair: tuple[str, str] | None = None
    sidecar_bytes: bytes | None = None
    sidecar: dict[str, object] | None = None
    if declared and not no_match:
        try:
            data = strict_json_object_loads(
                sibling_bytes,
                max_bytes=max(1, len(sibling_bytes)),
                source=json_path.name,
            )
            pair = _declared_voiceprint_pair(data)
            sidecar_path = sidecars.voiceprints_path(media)
            sidecar_bytes, sidecar = _load_voiceprints_exact(sidecar_path)
        except (OSError, Phase2DataError) as exc:
            _delete_stale_suggest_for_refusal(
                media,
                sibling_path=json_path,
                sibling_bytes=sibling_bytes,
            )
            if isinstance(exc, FileNotFoundError) and pair is not None:
                raise RuntimeError(
                    "voiceprint evidence is declared but not usable: this "
                    "episode's voiceprints were removed (voxweave speakers purge); "
                    f"{manual_hint} to review names without suggestions"
                ) from exc
            raise RuntimeError(
                "voiceprint evidence is declared but not usable; rerun "
                f"--diarize --voiceprints or {manual_hint}: {exc}"
            ) from exc

    library_location: voicelibrary.LibraryLocation | None = None
    library_scope: str | None = None
    try:
        if no_match or voices is None:
            store_stage = None
        else:
            store_stage = _load_generation_store(media, voices=voices, show=show)
        if not no_match and voices is None:
            library_location = voicelibrary.resolve_voices_dir(voices_dir)
            library_scope = voicelibrary.episode_scope(media, show)
    except (OSError, RuntimeError, Phase2DataError):
        _delete_stale_suggest_for_refusal(
            media,
            sibling_path=json_path,
            sibling_bytes=sibling_bytes,
        )
        raise

    turns = turns_in(data.get("speaker_turns"))
    if not turns:
        _delete_stale_suggest_for_refusal(
            media,
            sibling_path=json_path,
            sibling_bytes=sibling_bytes,
        )
        raise RuntimeError(
            f"{json_path.name} has no speaker_turns; run voxweave {quoted_media} --diarize first"
        )
    vad_speech = spans_in(data.get("vad_speech"))
    sing_spans = spans_in(data.get("sing_spans"))
    picks = select_snippets(turns, vad_speech, sing_spans)

    def generate_from(
        source: Path, snapshot_fingerprint: str | None
    ) -> SpeakerAudition:
        if pair is not None:
            assert sidecar is not None
            assert snapshot_fingerprint is not None
            try:
                validate_voiceprint_conjunction(sidecar, data, snapshot_fingerprint)
            except Phase2DataError as exc:
                _delete_stale_suggest_for_refusal(
                    media,
                    sibling_path=json_path,
                    sibling_bytes=sibling_bytes,
                )
                raise RuntimeError(
                    "voiceprint evidence does not bind this episode; rerun "
                    f"--diarize --voiceprints or {manual_hint}: {exc}"
                ) from exc

        embedded: dict[str, list[tuple[Span, str]]] = {label: [] for label in picks}
        with tempfile.TemporaryDirectory(prefix="voxweave_speakers_") as tmp_dir:
            root = Path(tmp_dir)
            for speaker_index, (speaker_id, spans) in enumerate(picks.items()):
                for clip_index, (start, end) in enumerate(spans):
                    clip_path = root / f"speaker-{speaker_index}-{clip_index}.mp3"
                    extract_clip(source, start, end, clip_path)
                    encoded = base64.b64encode(clip_path.read_bytes()).decode("ascii")
                    embedded[speaker_id].append(
                        ((start, end), f"data:audio/mpeg;base64,{encoded}")
                    )

        skeleton = {
            "version": MAPPING_VERSION,
            "speakers": {speaker_id: "" for speaker_id in picks},
        }

        def publish_locked(store_handle=None) -> SpeakerAudition:
            active_mapping_path = sidecars.speakers_mapping_path(media)
            current_data = data
            current_sidecar = sidecar
            current_sibling_bytes = json_path.read_bytes()
            if current_sibling_bytes != sibling_bytes:
                raise RuntimeError("input changed during speaker generation; re-run")
            if pair is not None:
                assert sidecar_bytes is not None
                assert snapshot_fingerprint is not None
                current_data = strict_json_object_loads(
                    current_sibling_bytes,
                    max_bytes=max(1, len(current_sibling_bytes)),
                    source=json_path.name,
                )
                current_sidecar_bytes = sidecars.voiceprints_path(media).read_bytes()
                if current_sidecar_bytes != sidecar_bytes:
                    raise RuntimeError(
                        "input changed during speaker generation; re-run"
                    )
                current_sidecar = _voiceprints_from_bytes(
                    current_sidecar_bytes,
                    source=sidecars.voiceprints_path(media).name,
                )
                validate_voiceprint_conjunction(
                    current_sidecar,
                    current_data,
                    snapshot_fingerprint,
                )

            matches: dict[str, SpeakerMatch] | None = None
            suggest_record: dict[str, object] | None = None
            if store_handle is not None:
                assert pair is not None
                assert current_sidecar is not None
                assert store_stage is not None
                store_path, store_bytes, _staged_store = store_stage
                current_bytes, current_store = _read_exact_object(
                    store_handle.store_path,
                    VOICES_STORE_MAX_BYTES,
                )
                validate_voice_store(current_store)
                if current_bytes != store_bytes:
                    raise RuntimeError(
                        "voices store changed during speaker generation; re-run"
                    )
                try:
                    matched = _matching_record(
                        current_sidecar,
                        store_path,
                        current_store,
                    )
                except (CompatibilityError, Phase2DataError) as exc:
                    log.warning("voice matching skipped: %s", exc)
                    matched = None
                if matched is not None:
                    matches, suggest_record, _thresholds = matched
            elif library_location is not None and pair is not None:
                assert current_sidecar is not None
                assert library_scope is not None
                try:
                    library_matched = _library_matching(
                        current_sidecar,
                        media=media,
                        location=library_location,
                        scope=library_scope,
                        folder_scope=show is None,
                    )
                except (CompatibilityError, Phase2DataError) as exc:
                    log.warning("voice matching skipped: %s", exc)
                    library_matched = None
                if library_matched is not None:
                    matches, suggest_record = library_matched

            def recheck_media_before_mapping_install() -> None:
                assert snapshot_fingerprint is not None
                try:
                    live_fingerprint = media_fingerprint(media)
                except OSError as exc:
                    raise RuntimeError(
                        "media could not be rechecked during speaker generation; re-run"
                    ) from exc
                if live_fingerprint != snapshot_fingerprint:
                    raise RuntimeError(
                        "media changed during speaker generation; re-run"
                    )

            pristine_mapping_generation = _publish_audition(
                media,
                mapping_path=active_mapping_path,
                skeleton=skeleton,
                suggest_record=suggest_record,
                before_mapping_install=(
                    recheck_media_before_mapping_install if pair is not None else None
                ),
            )
            page = _render_audition_html(
                media.name,
                active_mapping_path.name,
                embedded,
                matches,
                scope=library_scope if store_handle is None else None,
            )
            return SpeakerAudition(
                page=page,
                media_path=media,
                sibling_json_path=json_path,
                mapping_path=active_mapping_path,
                speaker_ids=tuple(picks),
                pristine_mapping_generation=pristine_mapping_generation,
            )

        with episode_lock(media):
            if pair is not None and store_stage is not None:
                store_path, _store_bytes, _store = store_stage
                with shared_store_lock(store_path) as lock_handle:
                    return publish_locked(lock_handle)
            return publish_locked()

    try:
        if pair is not None:
            snapshots = ExitStack()
            try:
                snapshot = snapshots.enter_context(MediaSnapshot(media))
            except SnapshotUnavailable as exc:
                snapshots.close()
                raise RuntimeError(
                    "cannot validate declared voiceprints without a media snapshot: "
                    f"{exc}"
                ) from exc
            else:
                with snapshots:
                    output = generate_from(snapshot.path, snapshot.fingerprint)
        else:
            output = generate_from(media, None)
    except BaseException:
        # A failed attempt must not leave an older machine claim behind. The
        # helper rechecks the exact staged sibling and mapping absence under the
        # episode lock, so it cannot erase a cooperating concurrent winner.
        _delete_stale_suggest_for_refusal(
            media,
            sibling_path=json_path,
            sibling_bytes=sibling_bytes,
        )
        raise
    log.info("prepared speaker audition for %s", output.mapping_path.name)
    return output


@dataclass(frozen=True)
class _StagedEnrollment:
    """The episode evidence paths and the bytes read before any lock."""

    json_path: Path
    sidecar_path: Path
    mapping_path: Path
    sibling_bytes: bytes = b""
    sidecar_bytes: bytes = b""
    mapping_bytes: bytes = b""


@dataclass(frozen=True)
class _EnrollmentEvidence:
    """The authoritative in-lock observation of one episode's evidence."""

    sibling: dict[str, object]
    sidecar: dict[str, object]
    voiceprints: ValidatedVoiceprints
    mapping_bytes: bytes


def _enrollment_paths(media: Path) -> _StagedEnrollment:
    if not media.is_file():
        raise FileNotFoundError(f"media file not found: {media}")
    staged = _StagedEnrollment(
        json_path=swap_ext(media, ".json"),
        sidecar_path=sidecars.voiceprints_path(media),
        mapping_path=sidecars.speakers_mapping_path(media),
    )
    if not staged.mapping_path.exists():
        raise FileNotFoundError(
            f"speaker mapping {staged.mapping_path.name} not found; "
            "generate and review it first"
        )
    return staged


def _stage_enrollment(paths: _StagedEnrollment) -> _StagedEnrollment:
    try:
        sibling_bytes, sibling = _read_sibling_exact(paths.json_path)
        pair = _declared_voiceprint_pair(sibling)
        if pair is None:
            raise Phase2DataError("sibling does not declare voiceprint evidence")
        sidecar_bytes, _sidecar = _load_voiceprints_exact(paths.sidecar_path)
        mapping_bytes = paths.mapping_path.read_bytes()
    except (OSError, Phase2DataError) as exc:
        raise RuntimeError(f"cannot stage enrollment evidence: {exc}") from exc
    return _StagedEnrollment(
        json_path=paths.json_path,
        sidecar_path=paths.sidecar_path,
        mapping_path=paths.mapping_path,
        sibling_bytes=sibling_bytes,
        sidecar_bytes=sidecar_bytes,
        mapping_bytes=mapping_bytes,
    )


def _recheck_enrollment(
    media: Path, staged: _StagedEnrollment, snapshot_fingerprint: str
) -> _EnrollmentEvidence:
    """Re-read the staged evidence under the locks and bind it to the media."""
    current_sibling_bytes = staged.json_path.read_bytes()
    if current_sibling_bytes != staged.sibling_bytes:
        raise RuntimeError("input changed during enrollment; re-run")
    current_sidecar_bytes = staged.sidecar_path.read_bytes()
    if current_sidecar_bytes != staged.sidecar_bytes:
        raise RuntimeError("input changed during enrollment; re-run")
    current_mapping_bytes = staged.mapping_path.read_bytes()
    if current_mapping_bytes != staged.mapping_bytes:
        raise RuntimeError("input changed during enrollment; re-run")

    current_sibling = strict_json_object_loads(
        current_sibling_bytes,
        max_bytes=max(1, len(current_sibling_bytes)),
        source=staged.json_path.name,
    )
    current_sidecar = _voiceprints_from_bytes(
        current_sidecar_bytes,
        source=staged.sidecar_path.name,
    )
    validated_sidecar = validate_voiceprint_conjunction(
        current_sidecar,
        current_sibling,
        snapshot_fingerprint,
    )
    if media_fingerprint(media) != snapshot_fingerprint:
        raise RuntimeError("media changed during enrollment; re-run")
    return _EnrollmentEvidence(
        sibling=current_sibling,
        sidecar=current_sidecar,
        voiceprints=validated_sidecar,
        mapping_bytes=current_mapping_bytes,
    )


def _named_speakers(
    evidence: _EnrollmentEvidence, mapping_path: Path
) -> dict[str, str]:
    mapping = _mapping_entries_bytes(evidence.mapping_bytes, source=mapping_path.name)
    named = {
        str(local_id): raw_name
        for local_id, raw_name in mapping.items()
        if isinstance(raw_name, str) and raw_name.strip()
    }
    if not named:
        raise EnrollmentRefusal("speaker mapping has no human-entered names to enroll")
    for raw_name in named.values():
        normalize_speaker_key(raw_name)
    return named


def _speaker_durations(sibling: Mapping[str, object]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for start, end, local_id in strict_turn_projection(sibling.get("speaker_turns")):
        durations[local_id] = durations.get(local_id, 0.0) + end - start
    return durations


_Candidate = tuple[str, str, tuple[int | float, ...], float]


def _group_winners(
    groups: Mapping[tuple[str, str], list[_Candidate]],
) -> list[tuple[str, str, tuple[int | float, ...]]]:
    """Keep the longest-speaking local id of each over-split identity."""
    selected: list[tuple[str, str, tuple[int | float, ...]]] = []
    for candidates in groups.values():
        winner = min(candidates, key=lambda item: (-item[3], item[0]))
        selected.append((winner[0], winner[1], winner[2]))
        for skipped in sorted(
            local_id for local_id, *_rest in candidates if local_id != winner[0]
        ):
            log.info(
                "skipping over-split local speaker %s in favor of %s",
                skipped,
                winner[0],
            )
    return selected


def enroll_speaker_voices(
    media: Path,
    *,
    voices: Path | None = None,
    voices_dir: Path | None = None,
    show: str | None = None,
    episode: str | None = None,
    replace_episode: bool = False,
) -> Path:
    """Enroll human-named, bound episode centroids.

    The voices go to the global voice library (``voices_dir``, else its
    configured location) under the episode's scope, which is ``show`` or else
    the media folder's name (see voicelibrary.episode_scope); the library
    directory is returned. An explicit ``voices`` per-show store keeps its
    pre-library behavior instead and its path is returned.
    """
    if voices is not None and voices_dir is not None:
        raise ValueError("use either a voices store or a voice library, not both")
    media = Path(media)
    if voices is None:
        return _enroll_into_library(
            media,
            voices_dir=voices_dir,
            show=show,
            episode=episode,
            replace_episode=replace_episode,
        )
    return _enroll_into_store(
        media,
        voices=voices,
        show=show,
        episode=episode,
        replace_episode=replace_episode,
    )


def _enroll_into_store(
    media: Path,
    *,
    voices: Path,
    show: str | None,
    episode: str | None,
    replace_episode: bool,
) -> Path:
    """Enroll into one explicit per-show store (the pre-library behavior)."""
    paths = _enrollment_paths(media)
    store_path = _resolved_voices_path(media, voices)
    create_store = not store_path.exists()
    if create_store and show is None:
        raise RuntimeError(
            "creating a voices store requires explicit --voices PATH and --show NAME"
        )
    if create_store:
        store_path.parent.mkdir(parents=True, exist_ok=True)

    staged = _stage_enrollment(paths)
    try:
        snapshot_context = MediaSnapshot(media)
        snapshot = snapshot_context.__enter__()
    except SnapshotUnavailable as exc:
        raise RuntimeError(f"cannot snapshot media for enrollment: {exc}") from exc
    try:
        with episode_lock(media):
            with exclusive_store_lock(store_path) as lock_handle:
                evidence = _recheck_enrollment(media, staged, snapshot.fingerprint)
                current_sidecar = evidence.sidecar
                validated_sidecar = evidence.voiceprints

                if lock_handle.store_path.exists():
                    _store_bytes, store = _read_exact_object(
                        lock_handle.store_path,
                        VOICES_STORE_MAX_BYTES,
                    )
                    validated_store = validate_voice_store(store)
                    if show is not None and normalize_show(show) != normalize_show(
                        validated_store.show
                    ):
                        raise EnrollmentRefusal(
                            f"--show {show!r} does not match the voices store"
                        )
                else:
                    assert show is not None
                    store = new_voice_store(
                        show,
                        cast(Mapping[str, object], current_sidecar["provenance"]),
                    )

                episode_provenance = cast(
                    Mapping[str, object], current_sidecar["provenance"]
                )
                store_provenance = cast(Mapping[str, object], store["provenance"])
                side_compat = build_compatibility_fingerprint(episode_provenance)
                store_compat = build_compatibility_fingerprint(store_provenance)
                if not compatibility_equal(side_compat, store_compat):
                    raise EnrollmentRefusal(
                        "enrollment refused: "
                        + describe_compatibility_mismatch(
                            episode_provenance,
                            store_provenance,
                            episode=side_compat,
                            store=store_compat,
                        )
                    )

                named = _named_speakers(evidence, paths.mapping_path)
                durations = _speaker_durations(evidence.sibling)
                groups: dict[tuple[str, str], list[_Candidate]] = {}
                for local_id, raw_name in named.items():
                    vector = validated_sidecar.speakers.get(local_id)
                    if vector is None:
                        log.info(
                            "skipping named local speaker %s without a usable centroid",
                            local_id,
                        )
                        continue
                    identity_id = resolve_identity_id(store, raw_name)
                    group_key = (
                        ("identity", identity_id)
                        if identity_id is not None
                        else ("name", normalize_speaker_key(raw_name))
                    )
                    groups.setdefault(group_key, []).append(
                        (local_id, raw_name, vector, durations.get(local_id, 0.0))
                    )
                if not groups:
                    raise EnrollmentRefusal("nothing to enroll")
                selected = _group_winners(groups)

                episode_key = normalize_episode(episode or media.stem)
                working = store
                base_revision = cast(int, store["revision"])
                mutations = 0
                noops = 0
                evicted = 0
                for _local_id, raw_name, vector in selected:
                    result = enroll_exemplar(
                        working,
                        raw_name=raw_name,
                        capture_id=validated_sidecar.capture_id,
                        media_fingerprint=validated_sidecar.media_fingerprint,
                        episode=episode_key,
                        vector=vector,
                        replace_episode=replace_episode,
                    )
                    working = result.store
                    if result.outcome == "noop":
                        noops += 1
                    else:
                        mutations += 1
                    evicted += result.evicted_exemplar_id is not None
                if mutations:
                    working["revision"] = base_revision + 1
                    validate_voice_store(working)
                    write_voice_store(lock_handle.store_path, working)
                    log.info(
                        "enrolled %d voice exemplar(s) into %s", mutations, store_path
                    )
                    _log_evictions(evicted)
                elif noops:
                    log.info("already enrolled; voices store unchanged")
                else:  # defensive: selected is nonempty and every transition is total
                    raise EnrollmentRefusal("nothing to enroll")
                return lock_handle.store_path
    finally:
        snapshot_context.__exit__(None, None, None)


def _log_evictions(count: int) -> None:
    """Say when enrollment displaced stored samples (the per-voice cap)."""
    if count:
        log.info(
            "%d voice(s) already had %d samples; the oldest sample of each was "
            "replaced",
            count,
            MAX_EXEMPLARS,
        )


def _accepted_suggestions(
    media: Path, sidecar: Mapping[str, object]
) -> dict[str, list[tuple[str, str, int, str | None]]]:
    """Per local speaker, the ``(identity, display name, tier, origin)``
    suggestions the review page offered for this exact capture.

    Only a record bound to the current capture and voiceprints is trusted; any
    other record is stale and ignored (enrollment then falls back to names).
    """
    path = sidecars.speakers_suggest_path(media)
    try:
        record = load_suggest(path)
    except FileNotFoundError:
        return {}
    except (OSError, Phase2DataError) as exc:
        log.info("ignoring an unusable suggestion record %s: %s", path, exc)
        return {}
    if record.get("capture_id") != sidecar.get("capture_id") or record.get(
        "voiceprints_digest"
    ) != voiceprints_digest(sidecar):
        return {}
    offered: dict[str, list[tuple[str, str, int, str | None]]] = {}
    for local_id, raw_match in cast(Mapping[str, object], record["speakers"]).items():
        match = cast(Mapping[str, object], raw_match)
        rows = offered.setdefault(local_id, [])
        tiers: list[tuple[int, Mapping[str, object]]] = [(1, match)]
        if "secondary" in match:
            tiers.append((2, cast(Mapping[str, object], match["secondary"])))
        for tier, tier_match in tiers:
            for raw in cast(list[Mapping[str, object]], tier_match["candidates"]):
                rows.append(
                    (
                        cast(str, raw["identity"]),
                        cast(str, raw["display_name"]),
                        tier,
                        cast("str | None", raw.get("origin")),
                    )
                )
    return offered


def _library_identity_for(
    state: voicelibrary.LibraryState,
    raw_name: str,
    *,
    scope: str,
    offered: Sequence[tuple[str, str, int, str | None]],
) -> str | None:
    """Resolve the library identity a reviewed name refers to, or None to create.

    Display names are not keys, so a name alone never links scopes: the
    reviewer links one only by using a suggestion, whose name is what they
    entered (tier 1 preferred). Otherwise the name resolves among the
    identities of this scope, and a name no identity of this scope carries
    becomes a new identity even if another scope has an identity of that
    name. A suggested id the library does not hold yet is adopted only when
    it came from a per-folder store not imported yet, so a later import
    merges into it; a suggestion of an identity that has been forgotten, or
    that left the library since the page was made, is refused.
    """
    key = normalize_speaker_key(raw_name)
    accepted = [
        (tier, identity_id, origin)
        for identity_id, display_name, tier, origin in offered
        if normalize_speaker_key(display_name) == key
    ]
    if accepted:
        best = min(tier for tier, _identity, _origin in accepted)
        chosen = {
            identity: origin for tier, identity, origin in accepted if tier == best
        }
        ids = sorted(chosen)
        if len(ids) > 1:
            raise EnrollmentRefusal(
                f"speaker name {raw_name!r} matches several suggested identities "
                f"({', '.join(ids)}); rename one with `voxweave voices rename`"
            )
        [identity_id] = ids
        if identity_id in state.identity_map:
            return identity_id
        if identity_id in state.forgotten:
            raise EnrollmentRefusal(
                f"speaker name {raw_name!r} was taken from a suggestion of "
                f"identity {identity_id}, which has been forgotten since "
                "(`voxweave voices forget`); run `voxweave speakers serve` again "
                "and review this speaker"
            )
        if chosen[identity_id] == "library":
            raise EnrollmentRefusal(
                f"speaker name {raw_name!r} was taken from a suggestion of "
                f"identity {identity_id}, which is no longer in the voice "
                "library; run `voxweave speakers serve` again and review this "
                "speaker"
            )
        return identity_id
    in_scope = voicelibrary.identities_named(state, raw_name, scope=scope)
    if len(in_scope) > 1:
        raise EnrollmentRefusal(
            f"speaker name {raw_name!r} belongs to several identities in scope "
            f"{scope!r} ({', '.join(in_scope)}); rename one with "
            "`voxweave voices rename`"
        )
    return in_scope[0] if in_scope else None


def _enroll_into_library(
    media: Path,
    *,
    voices_dir: Path | None,
    show: str | None,
    episode: str | None,
    replace_episode: bool,
) -> Path:
    """Enroll into the global voice library under the episode's scope."""
    paths = _enrollment_paths(media)
    location = voicelibrary.resolve_voices_dir(voices_dir)
    root = location.root
    scope = voicelibrary.episode_scope(media, show)
    episode_key = normalize_episode(episode or media.stem)
    staged = _stage_enrollment(paths)
    try:
        snapshot_context = MediaSnapshot(media)
        snapshot = snapshot_context.__enter__()
    except SnapshotUnavailable as exc:
        raise RuntimeError(f"cannot snapshot media for enrollment: {exc}") from exc
    try:
        with episode_lock(media):
            with voicelibrary.library_lock(
                root, exclusive=True, create_parents=location.default
            ):
                evidence = _recheck_enrollment(media, staged, snapshot.fingerprint)
                provenance = cast(Mapping[str, object], evidence.sidecar["provenance"])
                try:
                    space_name, _fingerprint = voicelibrary.space_identity(provenance)
                except voicelibrary.VoiceLibraryError as exc:
                    raise EnrollmentRefusal(f"enrollment refused: {exc}") from exc
                state = voicelibrary.read_state(root, include=[space_name])
                named = _named_speakers(evidence, paths.mapping_path)
                offered = _accepted_suggestions(media, evidence.sidecar)
                durations = _speaker_durations(evidence.sibling)
                groups: dict[tuple[str, str], list[_Candidate]] = {}
                for local_id, raw_name in named.items():
                    vector = evidence.voiceprints.speakers.get(local_id)
                    if vector is None:
                        log.info(
                            "skipping named local speaker %s without a usable centroid",
                            local_id,
                        )
                        continue
                    identity_id = _library_identity_for(
                        state,
                        raw_name,
                        scope=scope,
                        offered=offered.get(local_id, ()),
                    )
                    group_key = (
                        ("identity", identity_id)
                        if identity_id is not None
                        else ("name", normalize_speaker_key(raw_name))
                    )
                    groups.setdefault(group_key, []).append(
                        (local_id, raw_name, vector, durations.get(local_id, 0.0))
                    )
                if not groups:
                    raise EnrollmentRefusal("nothing to enroll")
                identity_of = {
                    local_id: key[1] if key[0] == "identity" else None
                    for key, candidates in groups.items()
                    for local_id, *_rest in candidates
                }
                entries = [
                    voicelibrary.EnrollEntry(
                        identity_id=identity_of[local_id],
                        raw_name=raw_name,
                        speaker_label=local_id,
                        vector=vector,
                    )
                    for local_id, raw_name, vector in _group_winners(groups)
                ]
                change, outcomes = voicelibrary.enroll_entries(
                    state,
                    provenance=provenance,
                    scope=scope,
                    source=voicelibrary.EpisodeSource(
                        media_path=os.path.abspath(media),
                        media_fingerprint=evidence.voiceprints.media_fingerprint,
                        capture_id=evidence.voiceprints.capture_id,
                        turns_digest=evidence.voiceprints.turns_digest,
                        episode=episode_key,
                    ),
                    entries=entries,
                    replace_episode=replace_episode,
                )
                voicelibrary.commit(state, change)
                mutations = sum(outcome.outcome != "noop" for outcome in outcomes)
                if mutations:
                    log.info(
                        "enrolled %d voice exemplar(s) into the voice library %s "
                        "(scope %r)",
                        mutations,
                        root,
                        scope,
                    )
                    _log_evictions(
                        sum(o.evicted_exemplar_id is not None for o in outcomes)
                    )
                else:
                    log.info("already enrolled; voice library unchanged")
                return root
    finally:
        snapshot_context.__exit__(None, None, None)


def purge_voiceprints(media: Path) -> tuple[Path, ...]:
    """Delete the complete per-episode biometric/derived artifact set."""
    media = Path(media)
    removed: list[Path] = []
    with episode_lock(media):
        targets = (
            *artifacts.fixed_candidates(
                media,
                ".voiceprints.json",
                "voiceprints",
            ),
            *artifacts.fixed_candidates(
                media,
                ".speakers.suggest.json",
                "speaker_suggest",
            ),
            artifacts.speaker_split_undo_path(media),
            sidecars.speakers_html_path(media),
        )
        for target in targets:
            if artifacts.path_present(target):
                target.unlink()
                removed.append(target)
                log.info("deleted %s", target)
    return tuple(removed)


__all__ = [
    "FFMPEG_TIMEOUT",
    "MAPPING_VERSION",
    "MAX_SNIPPET_S",
    "MIN_SNIPPET_GAP_S",
    "MIN_SNIPPET_S",
    "SpeakerAudition",
    "build_clip_command",
    "create_speaker_audition",
    "enroll_speaker_voices",
    "extract_clip",
    "load_speaker_display_names",
    "load_speaker_mapping_bytes",
    "purge_voiceprints",
    "run_clip_command",
    "sanitize_ass_speaker_name",
    "sanitize_speaker_name",
    "select_snippets",
    "speaker_layout",
    "speaker_metadata",
    "strip_srt_speaker_prefixes",
    "strip_voice_tags",
    "voice_tag_text",
    "voice_text_for_block",
    "voice_text_for_ids",
]
