from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from voxweave import (
    artifacts,
    backend,
    chunking,
    config,
    episode_transaction,
    realign,
    songdet,
)
from voxweave.chunking import (
    decode_to_wav as decode_to_wav,
    pack_speech_segments as pack_speech_segments,
    silence_gaps as silence_gaps,
    slice_wav,
    vad_speech_segments as vad_speech_segments,
)
from voxweave.align_failures import CanonicalFailure, SecondaryFailure
from voxweave.align_runtime import (
    align_runtime_activity,
    bind_align_runtime_identity,
)
from voxweave.core.overlay import (
    LYRIC_MIN_OVERLAP as LYRIC_MIN_OVERLAP,
    lyric_display_text as lyric_display_text,
    mark_lyric_cues as mark_lyric_cues,
    spans_in,
    turns_in,
)

# The v2 shadow lane lives in ``voxweave.core.shadow_v2``; its flag and lane
# names are re-exported here for the readers that always imported them from
# ``pipeline`` (calibration harness, tests). The lane module imports nothing at
# module scope beyond what the v1 engine already loads.
from voxweave.core.shadow_v2 import (
    SEG_V2_SHADOW_ENV as SEG_V2_SHADOW_ENV,
    SHADOW_LANE_CORE as SHADOW_LANE_CORE,
    SHADOW_LANE_DELIVERY_LEGACY as SHADOW_LANE_DELIVERY_LEGACY,
    shadow_enabled,
    SHADOW_LANE_FINALIZER as SHADOW_LANE_FINALIZER,
    SHADOW_LANE_LEGACY_DISPLAY as SHADOW_LANE_LEGACY_DISPLAY,
)
from voxweave.debug import DebugSink as DebugSink, FileDebugSink as FileDebugSink
from voxweave.lang import (
    is_supported as is_supported,
    require_aligner_language,
    to_asr_iso,
    to_iso_or,
)
from voxweave.mediasnapshot import MediaSnapshot, SnapshotUnavailable

# Path, sidecar, vocals-cache and segmentation helpers moved to their own
# modules; the public names stay importable from here.
from voxweave.paths import MEDIA_EXTS as MEDIA_EXTS
from voxweave.paths import find_subtitle_media, require_output_name, swap_ext
from voxweave.progress import Reporter
from voxweave.segmentation import (
    SEGMENTATION_ENGINE as SEGMENTATION_ENGINE,
    SEGMENTATION_MANIFEST_VERSION as SEGMENTATION_MANIFEST_VERSION,
    SegmentationResult as SegmentationResult,
    resolve_segmentation_manifest,
    segment_document,
)
from voxweave.sidecars import (
    artifact_owner,
    inspect_speakers_mapping_path as inspect_speakers_mapping_path,
    voiceprints_path as voiceprints_path,
)
from voxweave.sidecars import (
    speakers_mapping_path as speakers_mapping_path,
    speakers_suggest_path as speakers_suggest_path,
)
from voxweave.songdet import (
    detect_song_spans as detect_song_spans,
    excise_spans_from_segments as excise_spans_from_segments,
    expand_spans_to_voiced_blocks as expand_spans_to_voiced_blocks,
    filter_short_spans as filter_short_spans,
    group_segments_by_spans as group_segments_by_spans,
    rescue_speech_segments as rescue_speech_segments,
    subtract_spans as subtract_spans,
)
from voxweave.subformats import require_subtitle
from voxweave.timestamps import shift_units
from voxweave.transcript_language import (
    _reconcile_word_segment_language as _reconcile_word_segment_language,
    _select_transcript_language as _select_transcript_language,
)
from voxweave.vocals import (
    ASR_LOUDNORM as ASR_LOUDNORM,
    CACHE_DIRNAME as CACHE_DIRNAME,
    CACHE_DUR_TOL_SEC as CACHE_DUR_TOL_SEC,
    SONGDET_SR as SONGDET_SR,
    VocalsCacheMiss,
    acquire_16k,
    cache_vocals_path,
)
from voxweave.vocals import cache_16k_path as cache_16k_path
from voxweave.voicebase import (
    Phase2DataError,
    media_fingerprint,
    require_capture_id,
    require_sha256,
    strict_json_object_loads,
)
from voxweave.voiceprint_capture import (
    VoiceprintCapture as VoiceprintCapture,
    _decoupled_voiceprint_capture as _decoupled_voiceprint_capture,
    _prefetch_voiceprint_models as _prefetch_voiceprint_models,
)
from voxweave.transcription import (
    MAX_CHUNK_SEC as MAX_CHUNK_SEC,
    MIN_SONG_SKIP_SEC as MIN_SONG_SKIP_SEC,
    SNAP_VAD_THRESHOLD as SNAP_VAD_THRESHOLD,
    SONG_FINE_SILENCE_MS as SONG_FINE_SILENCE_MS,
    plan_song_skip as plan_song_skip,
    transcribe as transcribe,
)
from voxweave.process_publication import (
    _ProcessPublication as _ProcessPublication,
    _ProcessSource,
    _voiceprint_capture_from_generation as _voiceprint_capture_from_generation,
    _warn_stale_speaker_names as _warn_stale_speaker_names,
    _voiceprints_document as _voiceprints_document,
    _finish_process_from_units as _finish_process_from_units,
    _write_sdh_sidecar as _write_sdh_sidecar,
)
from voxweave.failure_annotations import (
    _attach_canonical_failure as _attach_canonical_failure,
    _append_secondary_terminal as _append_secondary_terminal,
    _record_disposal_failure as _record_disposal_failure,
)
from voxweave.align_route import (
    AlignRoute as AlignRoute,
    resolve_align_route as resolve_align_route,
    _align_blocks_from_snapshot,
    _align_route_facts,
    _plan_qwen_crops,
    _prepared_audio_geometry,
    _stable_fact_digest,
)

if TYPE_CHECKING:
    from voxweave.shotdet import ShotDetectionJob

log = logging.getLogger("voxweave")

_VOICEPRINT_NOTICE_LOCK = threading.Lock()
_voiceprint_notice_logged = False


def _log_voiceprint_notice_once() -> None:
    global _voiceprint_notice_logged
    with _VOICEPRINT_NOTICE_LOCK:
        if _voiceprint_notice_logged:
            return
        log.warning(
            "voiceprint capture enabled: a sensitive voice-biometric sidecar may be written"
        )
        _voiceprint_notice_logged = True


# Align-stage cue duration floor. Default 0 (disabled): enforce_min_duration only
# resolves overlaps without padding, so short back-channels keep their real ~0.6s.
# Set VOXWEAVE_MIN_CUE_SEC=0.8 to re-enable padding. Distinct from VOXWEAVE_SEG_MIN_CUE_SEC.
MIN_CUE_SEC = config._env_float("VOXWEAVE_MIN_CUE_SEC", 0.0)
# Flash-cue rescue (orthogonal to MIN_CUE_SEC): genuine flash cues (so/あ at 0.1-0.2s)
# are extended to TINY_CUE_TARGET, allowed to overlap only the immediately following cue.
# VOXWEAVE_TINY_CUE_SEC=0 disables.
TINY_CUE_SEC = config._env_float("VOXWEAVE_TINY_CUE_SEC", 0.2)
TINY_CUE_TARGET = config._env_float("VOXWEAVE_TINY_CUE_TARGET", 0.5)


def _attach_json_decode_failure(exc: BaseException) -> None:
    cause = exc.__cause__
    if isinstance(cause, UnicodeDecodeError):
        detail = "sibling-json-encoding"
    elif isinstance(cause, json.JSONDecodeError):
        detail = "sibling-json-syntax"
    else:
        detail = "sibling-top-level-shape"
    _attach_canonical_failure(
        exc,
        kind="align-input-decode-invalid",
        phase="decode",
        detail_code=detail,
    )


def _attach_vtt_decode_failure(exc: BaseException, vtt_name: str) -> None:
    # Encoding failures reach here raw (a BOM-declared encoding that does not
    # decode) or as the subtitle decoder's "cannot determine text encoding" refusal.
    if (
        isinstance(exc, UnicodeDecodeError)
        or isinstance(exc.__cause__, UnicodeDecodeError)
        or str(exc).startswith(f"{vtt_name}: cannot determine text encoding")
    ):
        detail = "vtt-encoding"
    elif str(exc).startswith("no cues in "):
        detail = "vtt-no-cues"
    else:
        detail = "vtt-format-mismatch"
    _attach_canonical_failure(
        exc,
        kind="align-input-decode-invalid",
        phase="decode",
        detail_code=detail,
    )


def _panns_release_failure() -> CanonicalFailure:
    return CanonicalFailure("model-release-failed", "dispose", "panns-release")


def _annotate_panns_release_primary(
    exc: BaseException, publication: _ProcessPublication
) -> None:
    """Attach the terminal without replacing the release exception itself."""
    for name, value in (
        ("failure", _panns_release_failure()),
        ("landed", publication.landed),
        ("auxiliary_landed", publication.auxiliary_landed),
    ):
        try:
            setattr(exc, name, value)
        except Exception:
            pass


def _append_panns_release_secondary(
    primary: BaseException, release: BaseException
) -> None:
    """Retain an earlier exception and append the closed late-release terminal."""
    secondary = SecondaryFailure("model-release-failed", "dispose", "panns-release")
    failure = getattr(primary, "failure", None)
    if isinstance(failure, CanonicalFailure):
        try:
            setattr(
                primary,
                "failure",
                CanonicalFailure(
                    failure.kind,
                    failure.phase,
                    failure.detail_code,
                    failure.secondary + (secondary,),
                ),
            )
        except Exception:
            pass
    try:
        current = tuple(getattr(primary, "secondary_failures", ()))
        setattr(primary, "secondary_failures", current + (secondary,))
        setattr(primary, "panns_release_exception", release)
    except Exception:
        pass


def require_vtt(path: Path) -> Path:
    """Reject non-VTT inputs for align/correct; return the path unchanged.

    Both commands write VTT back (align overwrites the input in place, correct's
    ``--apply`` does too), so running them on ``.srt``/``.ass`` would corrupt the
    file with VTT content. translate/pack/burn/export accept the other formats
    via :func:`voxweave.subformats.require_subtitle`.
    """
    return require_subtitle(path, exts=(".vtt",))


def _load_sibling_json_bytes(
    json_path: Path,
    raw: bytes,
    *,
    require: str | None = None,
) -> dict:
    """Decode the exact sibling bytes staged by an optimistic transaction."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        detail = (
            f"{e.msg} at line {e.lineno}"
            if isinstance(e, json.JSONDecodeError)
            else "invalid UTF-8"
        )
        raise RuntimeError(
            f"{json_path.name} is corrupt JSON ({detail});"
            " re-run `voxweave transcribe <media>` to regenerate it"
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(
            f"{json_path.name}: expected a JSON object, got {type(data).__name__};"
            " re-run `voxweave transcribe <media>` to regenerate it"
        )
    if require is not None and require not in data:
        raise RuntimeError(
            f"{json_path.name} has no {require!r} key;"
            " re-run `voxweave transcribe <media>` to regenerate it"
        )
    return data


def _replay_voiceprint_pair(
    data: Mapping[str, object],
    raw: bytes,
    *,
    source: str,
) -> tuple[str, str] | None:
    """Return an exact grammar-valid replay pair, warning and dropping otherwise."""
    if "voiceprint_capture" not in data and "voiceprint_media" not in data:
        return None
    try:
        strict = strict_json_object_loads(
            raw,
            max_bytes=max(1, len(raw)),
            source=source,
        )
        capture = require_capture_id(
            strict.get("voiceprint_capture"),
            "voiceprint_capture",
        )
        media = require_sha256(strict.get("voiceprint_media"), "voiceprint_media")
    except Phase2DataError as exc:
        log.warning("%s: dropping invalid voiceprint replay pair: %s", source, exc)
        return None
    return capture, media


def process(
    media_path: Path,
    lang_override: str | None = None,
    separate: bool = True,
    reporter: Reporter | None = None,
    debug: bool = False,
    normalize: bool = False,
    skip_songs: bool = False,
    keep_lyrics: bool = False,
    sdh: bool = False,
    diarize: bool = False,
    voiceprints: bool = False,
    word_segments: tuple[str, list[dict]] | None = None,
    asr_model: str | None = None,
    context: str | None = None,
    timestamps: bool = True,
    shot_snap: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    diarize_model: str | None = None,
    voiceprint_model: str | None = None,
    speaker_clustering: str | None = None,
) -> Path:
    """Full pipeline: transcribe -> smart_split -> write siblings. Return the .vtt path.

    Pass ``word_segments`` to skip transcription (tests / special cases); that path
    also skips shot detection (no media decode in unit tests). ``keep_lyrics``
    transcribes detected songs instead of excising them and flags the sung cues
    (rendered with a music-note wrap; spans persist to JSON for ``split`` replay).
    ``sdh`` additionally writes a ``<stem>.sdh.vtt`` sidecar with PANNs-detected
    non-speech event tags merged into the dialogue (main VTT/JSON untouched).
    ``diarize`` runs pyannote speaker diarization and formats multi-speaker cues
    (dual-speaker hyphens / speaker-boundary splits; turns persist to JSON);
    ``speaker_clustering`` (``"pyannote"``/``"voiceprint"``, ``None`` = configured)
    picks how its turns are grouped into speakers.
    """
    media_path = Path(media_path)
    if lang_override and lang_override.strip():
        # A --language typo fails here, not after separation and VAD.
        to_asr_iso(lang_override)
    # The deliverables keep their names beside the media: refuse one that cannot
    # exist before any audio work, not at the final write.
    writes_sdh = sdh and word_segments is None
    for suffix in (".json", ".vtt", *((".sdh.vtt",) if writes_sdh else ())):
        require_output_name(swap_ext(media_path, suffix))
    rep = reporter or Reporter()
    steps = ["inspect source"]
    if word_segments is None:
        steps.append("prepare audio")
        if separate and (skip_songs or keep_lyrics):
            steps.append("detect songs")
        steps.extend(("find speech", "transcribe and align"))
        if diarize:
            steps.append("identify speakers")
        if shot_snap:
            steps.append("detect shot changes")
    steps.extend(("layout subtitles", "write outputs"))
    if sdh and word_segments is None:
        steps.append("create SDH sidecar")
    rep.plan(steps)
    rep.step("inspect source")
    expected_json: episode_transaction.FileGeneration | None = None
    expected_vtt: episode_transaction.FileGeneration | None = None
    expected_media: str | None = None
    source_mode: episode_transaction.ProcessSourceMode | None = None
    debug_root = artifacts.claim_paths(media_path).debug if debug else None

    def run_with_source(
        source_path: Path,
        snapshot_fingerprint: str | None,
        capture_enabled: bool,
    ) -> Path:
        assert expected_json is not None
        assert expected_vtt is not None
        assert source_mode is not None
        # Built at call time: expected_json, expected_vtt, expected_media and
        # source_mode are assigned after this closure is defined.
        source = _ProcessSource(
            path=source_path,
            snapshot_fingerprint=snapshot_fingerprint,
            expected_json=expected_json,
            expected_vtt=expected_vtt,
            expected_media_fingerprint=expected_media,
            mode=source_mode,
        )

        # Called by _process_from_source after the shot detection job starts;
        # ``transcribe`` and ``cache_vocals_path`` are looked up when it runs.
        def transcribe_source() -> tuple[
            str,
            list[dict],
            list[tuple[float, float]],
            list[tuple[float, float]],
            list[tuple[float, float, str]],
            VoiceprintCapture | None,
        ]:
            return transcribe(
                source_path,
                lang_override=lang_override,
                separate=separate,
                skip_songs=skip_songs,
                keep_lyrics=keep_lyrics,
                diarize=diarize,
                diarize_model=diarize_model,
                voiceprints=capture_enabled,
                voiceprint_model=voiceprint_model,
                speaker_clustering=speaker_clustering,
                normalize=normalize,
                reporter=rep,
                debug=debug,
                debug_root=debug_root,
                cache_vocals=cache_vocals_path(media_path),
                source_fingerprint=snapshot_fingerprint,
                asr_model=asr_model,
                context=context,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                # SDH reuses the successfully returned deferred PANNs owner.
                release_panns=not sdh,
            )

        return _process_from_source(
            media_path,
            source=source,
            lang_override=lang_override,
            reporter=rep,
            sdh=sdh,
            word_segments=word_segments,
            timestamps=timestamps,
            shot_snap=shot_snap,
            transcribe_source=transcribe_source,
        )

    if voiceprints and (not diarize or word_segments is not None):
        raise ValueError("voiceprint capture requires a fresh diarization run")
    if diarize and word_segments is None:
        from voxweave import diarize as diarize_mod

        # Refuse a diarization that cannot succeed (a gated model without a
        # token, impossible speaker bounds) now, not after separation and ASR.
        diarize_mod.preflight(
            diarize_model, min_speakers=min_speakers, max_speakers=max_speakers
        )
    capture_ready = False
    if voiceprints:
        from voxweave import voiceembed

        # Fail on a bad --voiceprint-model / env / conf value before any audio
        # work; the per-language routing itself waits for the detected language.
        voiceembed.resolve_voiceprint_choice(voiceprint_model)
        _log_voiceprint_notice_once()
        capture_ready = _prefetch_voiceprint_models(
            voiceprint_model, lang_override, rep
        )
    try:
        expected_json = episode_transaction.capture_file_generation(
            swap_ext(media_path, ".json")
        )
    except OSError as exc:
        _attach_canonical_failure(
            exc,
            kind="subtitle-snapshot-failed",
            phase="snapshot",
            detail_code="sibling-read",
        )
        raise
    try:
        expected_vtt = episode_transaction.capture_file_generation(
            swap_ext(media_path, ".vtt")
        )
    except OSError as exc:
        _attach_canonical_failure(
            exc,
            kind="subtitle-snapshot-failed",
            phase="snapshot",
            detail_code="vtt-read",
        )
        raise
    source_mode = "injected-words" if word_segments is not None else "transcribed-media"
    if source_mode == "injected-words":
        expected_media = None
    else:
        try:
            expected_media = media_fingerprint(media_path)
        except OSError as exc:
            _attach_canonical_failure(
                exc,
                kind="media-identity-invalid",
                phase="media",
                detail_code=(
                    "media-not-found"
                    if isinstance(exc, FileNotFoundError)
                    else "media-fingerprint"
                ),
            )
            raise
    if capture_ready:
        snapshots = ExitStack()
        try:
            snapshot = snapshots.enter_context(MediaSnapshot(media_path))
        except SnapshotUnavailable as exc:
            snapshots.close()
            log.warning(
                "voiceprint capture unavailable; continuing without capture: %s",
                exc,
            )
        else:
            with snapshots:
                try:
                    return run_with_source(
                        snapshot.path,
                        snapshot.fingerprint,
                        True,
                    )
                except chunking.NoAudioStreamError as exc:
                    # The decode saw the private snapshot copy; name the user's file.
                    raise chunking.NoAudioStreamError(media_path.name) from exc
    return run_with_source(media_path, None, False)


def _process_from_source(
    media_path: Path,
    *,
    source: _ProcessSource,
    lang_override: str | None = None,
    reporter: Reporter | None = None,
    sdh: bool = False,
    word_segments: tuple[str, list[dict]] | None = None,
    timestamps: bool = True,
    shot_snap: bool = True,
    transcribe_source: Callable[
        [],
        tuple[
            str,
            list[dict],
            list[tuple[float, float]],
            list[tuple[float, float]],
            list[tuple[float, float, str]],
            VoiceprintCapture | None,
        ],
    ],
) -> Path:
    """Execute one process run against the selected immutable/live source."""
    rep = reporter or Reporter()
    panns_handoff_owned = False
    vad_speech: list[tuple[float, float]] | None = None
    shot_changes: list[float] | None = None
    sing_spans: list[tuple[float, float]] | None = None
    speaker_turns: list[tuple[float, float, str]] | None = None
    _voiceprint_capture: VoiceprintCapture | None = None
    # A started-but-uncollected shot detection pass; reaped on any failure below.
    shot_job: ShotDetectionJob | None = None
    try:
        if word_segments is not None:
            iso, units = word_segments
            iso, units = _reconcile_word_segment_language(
                iso, units, override=lang_override
            )
        else:
            if shot_snap:
                from voxweave import shotdet

                # CPU-only ffmpeg pass: start it now so it overlaps the GPU stages
                # and collect it at the "detect shot changes" step afterwards.
                shot_job = shotdet.ShotDetectionJob().start(source.path)
            (
                iso,
                units,
                vad_speech,
                sing_spans,
                speaker_turns,
                _voiceprint_capture,
            ) = transcribe_source()
            panns_handoff_owned = sdh
            sing_spans = sing_spans or None
            speaker_turns = speaker_turns or None
            if shot_job is not None:
                # The step timer measures the join only: ffmpeg itself has been
                # running since before transcribe(), so this is normally ~0s.
                rep.step("detect shot changes")
                rep.stage("collect shot changes")
                shot_changes = shot_job.result()
                shot_job = None

        publication = _finish_process_from_units(
            media_path,
            rep=rep,
            iso=iso,
            units=units,
            vad_speech=vad_speech,
            shot_changes=shot_changes,
            sing_spans=sing_spans,
            speaker_turns=speaker_turns,
            timestamps=timestamps,
            capture=_voiceprint_capture,
            source=source,
            sdh_enabled=sdh,
        )
    except BaseException as primary:
        if shot_job is not None:
            try:
                shot_job.cancel()
            except BaseException as cancel_error:
                log.warning(
                    "shot detection cancel failed after earlier process failure: %r",
                    cancel_error,
                )
        if panns_handoff_owned:
            try:
                songdet.release_model()
            except BaseException as release_error:
                log.warning(
                    "PANNs release failed after earlier process failure: %r",
                    release_error,
                )
                _append_panns_release_secondary(primary, release_error)
        raise
    if panns_handoff_owned:
        try:
            songdet.release_model()
        except BaseException as release_error:
            _annotate_panns_release_primary(release_error, publication)
            raise
    return publication.path


def _check_layout_overrides(iso: str, overrides: Mapping[str, Any]) -> None:
    """Reject line limits the layout cannot honour before they replace the VTT."""
    from voxweave.core.layout import min_max_line_length

    max_lines = overrides.get("max_lines")
    if max_lines is not None and max_lines < 1:
        raise ValueError(f"max_lines must be at least 1, got {max_lines}")
    max_line_length = overrides.get("max_line_length")
    floor = min_max_line_length(iso)
    if max_line_length is not None and max_line_length < floor:
        raise ValueError(
            f"max_line_length {max_line_length} is below the minimum of {floor} for"
            f" {iso!r}: shorter lines split ordinary words apart"
        )


class VttEditsKeptError(RuntimeError):
    """``split(keep_vtt_edits=True)`` found cues the JSON would not reproduce."""


def _cue_content(vtt_path: Path, raw: bytes) -> list[tuple[Any, Any, str, bool]] | None:
    """Cue times, text and lyric flag of a VTT, without speaker names (None = unreadable)."""
    from voxweave.subformats import load_subtitle_blocks_bytes

    try:
        blocks = load_subtitle_blocks_bytes(vtt_path, raw)
    except (RuntimeError, ValueError, UnicodeError):
        return None
    return [
        (block.get("start"), block.get("end"), block["text"], bool(block.get("lyric")))
        for block in blocks
    ]


@dataclass(frozen=True)
class _SplitSource:
    # Everything ``split`` reads before it lays anything out: the exact sibling
    # JSON bytes, the replay carriers decoded from them, and the speaker mapping.
    json_path: Path
    owner: Path
    input_bytes: bytes
    voiceprint_pair: tuple[str, str] | None
    pair_declared: bool
    iso: str
    units: list[dict]
    speech_spans: list[tuple[float, float]] | None
    shot_changes: list[float] | None
    sing_spans: list[tuple[float, float]] | None
    speaker_turns: list[tuple[float, float, str]] | None
    annotate_speakers: bool
    mapping_path: Path
    mapping_generation: episode_transaction.SpeakerMappingGeneration


@dataclass(frozen=True)
class _SplitCommit:
    # What ``split`` publishes: the verified selection and the generations the
    # commit must still find on disk.
    vtt_path: Path
    selection: Any
    expected_json: episode_transaction.FileGeneration
    expected_vtt: episode_transaction.FileGeneration | None
    cleanup: tuple[episode_transaction.ArtifactCleanup, ...]


def _read_split_source(
    json_path: Path, layout_overrides: Mapping[str, Any]
) -> _SplitSource:
    # Read stage of split: snapshot the sibling JSON, decode the replay carriers
    # and capture the speaker mapping. Nothing is laid out or written here.
    owner = artifact_owner(json_path)
    try:
        input_bytes = json_path.read_bytes()
    except OSError as exc:
        _attach_canonical_failure(
            exc,
            kind="subtitle-snapshot-failed",
            phase="snapshot",
            detail_code="sibling-read",
        )
        raise
    try:
        data = _load_sibling_json_bytes(
            json_path,
            input_bytes,
            require="word_segments",
        )
        word_segments = data["word_segments"]
        if not isinstance(word_segments, list) or not all(
            isinstance(unit, Mapping) for unit in word_segments
        ):
            raise RuntimeError(
                f"{json_path.name}: 'word_segments' must be a list of objects;"
                " re-run `voxweave transcribe <media>` to regenerate it"
            )
    except RuntimeError as exc:
        _attach_json_decode_failure(exc)
        raise
    voiceprint_pair = _replay_voiceprint_pair(
        data,
        input_bytes,
        source=json_path.name,
    )
    # Label what produced the document being replayed before touching it: split
    # re-segments, so it regenerates the manifest rather than preserving one.
    log.debug("replaying %s (%s)", json_path.name, resolve_segmentation_manifest(data))
    units = data["word_segments"]
    iso, units = _reconcile_word_segment_language(data.get("language", "en"), units)
    _check_layout_overrides(iso, layout_overrides)
    speech_spans = spans_in(data.get("vad_speech"))
    shot_changes = [float(t) for t in data.get("shot_changes") or []] or None
    sing_spans = spans_in(data.get("sing_spans"))
    speaker_turns = turns_in(data.get("speaker_turns"))
    mapping_path = artifacts.speaker_mapping_path(
        owner,
        reference=json_path,
    )
    mapping_generation = episode_transaction.capture_speaker_mapping(
        mapping_path,
        known_ids={label for _start, _end, label in speaker_turns or ()},
        warn=lambda message: log.warning("%s", message),
    )
    speaker_names = dict(mapping_generation.names)
    return _SplitSource(
        json_path=json_path,
        owner=owner,
        input_bytes=input_bytes,
        voiceprint_pair=voiceprint_pair,
        pair_declared="voiceprint_capture" in data or "voiceprint_media" in data,
        iso=iso,
        units=units,
        speech_spans=speech_spans,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        annotate_speakers=bool(speaker_names),
        mapping_path=mapping_path,
        mapping_generation=mapping_generation,
    )


def _plan_split_commit(
    source: _SplitSource,
    segmented: SegmentationResult,
    *,
    timestamps: bool,
    keep_vtt_edits: bool,
) -> _SplitCommit:
    # Select stage of split: bind the laid-out cues to the generations read
    # earlier and build the verified selection with its cleanup list.
    from voxweave import segmentation_orchestration
    from voxweave.align_snapshot import decode_sibling_json_snapshot

    json_path = source.json_path
    if segmented.document is None or segmented.manifest is None:
        raise RuntimeError("segmentation result lacks its production authority")
    expected_json = episode_transaction.FileGeneration(True, source.input_bytes)
    vtt_path = swap_ext(json_path, ".vtt")
    expected_vtt: episode_transaction.FileGeneration | None = None
    if keep_vtt_edits:
        try:
            expected_vtt = episode_transaction.FileGeneration(
                True, vtt_path.read_bytes()
            )
        except FileNotFoundError:
            expected_vtt = episode_transaction.FileGeneration(False, None)
    lexical_snapshot = decode_sibling_json_snapshot(json_path.name, source.input_bytes)
    selection = segmentation_orchestration.build_segmentation_selection(
        command="split",
        target_path=swap_ext(json_path, ".vtt"),
        sibling_path=json_path,
        language=source.iso,
        cues=segmented.cues,
        top_level_units=segmented.units,
        document=segmented.document,
        manifest=segmented.manifest,
        vad_speech=source.speech_spans,
        shot_changes=source.shot_changes,
        sing_spans=source.sing_spans,
        speaker_turns=lexical_snapshot.carrier("speaker_turns"),
        voiceprint_pair=source.voiceprint_pair,
        timestamps=timestamps,
        speaker_names=source.mapping_generation.names,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        source_mode=None,
        mapping_generation=source.mapping_generation,
        mapping_path=source.mapping_path,
        shadow_enabled=shadow_enabled(),
    )
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if source.pair_declared and source.voiceprint_pair is None:
        cleanup.extend(
            episode_transaction.speaker_artifact_cleanups(source.owner, json_path)
        )
    cleanup.extend(
        episode_transaction.evidence_cleanups(
            swap_ext(json_path, ".vtt"), media=source.owner
        )
    )
    return _SplitCommit(
        vtt_path, selection, expected_json, expected_vtt, tuple(cleanup)
    )


def _commit_split(source: _SplitSource, plan: _SplitCommit) -> None:
    # Write stage of split: refuse to replace hand edits when asked to keep them,
    # commit, and retire the selection however the commit ends.
    from voxweave import segmentation_orchestration

    json_path = source.json_path
    vtt_path = plan.vtt_path
    selection = plan.selection
    expected_vtt = plan.expected_vtt
    try:
        if expected_vtt is not None and expected_vtt.bytes_value is not None:
            old = _cue_content(vtt_path, expected_vtt.bytes_value)
            if old is None or old != _cue_content(
                vtt_path, selection.verified.vtt_bytes
            ):
                raise VttEditsKeptError(
                    f"{vtt_path.name} was not overwritten: its cue text or timing"
                    f" differs from what {json_path.name} produces (hand edits, align"
                    " output, or other render options); run"
                    f" 'voxweave render {json_path}' to replace it anyway"
                )
        episode_transaction.commit_primary_outputs(
            command="split",
            episode_path=source.owner,
            json_path=json_path,
            vtt_path=vtt_path,
            expected_json=plan.expected_json,
            expected_vtt=expected_vtt,
            main_json_bytes=selection.verified.main_json_bytes,
            vtt_bytes=selection.verified.vtt_bytes,
            cleanup_paths=plan.cleanup,
            context=selection.context,
            speaker_mapping_path=source.mapping_path,
            expected_speaker_mapping=source.mapping_generation,
        )
    finally:
        segmentation_orchestration.retire_segmentation_selection(selection)


def split(
    json_path: Path,
    timestamps: bool = True,
    *,
    reporter: Reporter | None = None,
    keep_vtt_edits: bool = False,
    **smart_split_kwargs,
) -> Path:
    """Re-run smart_split from persisted word_segments.

    Reuses ``vad_speech`` from the sibling JSON for gap splitting; falls back to gap-only
    mode if absent. ``timestamps`` behaves as in :func:`process`.

    ``keep_vtt_edits`` writes only when the existing VTT is missing or differs from
    the new one in speaker names alone; otherwise it raises :class:`VttEditsKeptError`
    and writes nothing. A VTT changed after that check is not overwritten either. Use
    it when the caller did not ask to replace the VTT (render after a speakers Save):
    the VTT may hold hand edits the JSON does not have yet.
    """
    # Accept the .vtt sibling too: `voxweave split foo.vtt` should not feed
    # WEBVTT bytes to json.loads.
    json_path = swap_ext(Path(json_path), ".json")
    rep = reporter or Reporter()
    rep.plan(("read subtitle data", "layout subtitles", "write outputs"))
    rep.step("read subtitle data")
    source = _read_split_source(json_path, smart_split_kwargs)
    rep.step("layout subtitles")
    segmented = segment_document(
        language=source.iso,
        word_segments=source.units,
        vad_speech=source.speech_spans,
        shot_changes=source.shot_changes,
        sing_spans=source.sing_spans,
        speaker_turns=source.speaker_turns,
        smart_split_kwargs=smart_split_kwargs,
        annotate_speakers=source.annotate_speakers,
    )
    plan = _plan_split_commit(
        source, segmented, timestamps=timestamps, keep_vtt_edits=keep_vtt_edits
    )
    rep.step("write outputs")
    _commit_split(source, plan)
    log.info("rendered %s → %d cues", plan.vtt_path.name, len(segmented.cues))
    return plan.vtt_path


def _prepare_16k_for_align(
    media: Path,
    *,
    separate: bool,
    normalize: bool,
    reporter: Reporter,
    tmp: list[Path],
    cache_media: Path | None = None,
    source_fingerprint: str | None = None,
    hit_only: bool = False,
) -> Path:
    """Prepare 16k vocals for align; append temp paths to tmp. Return the 16k wav path.

    ``media`` is the byte authority used for a cache miss. ``cache_media`` owns
    the persistent cache namespace and defaults to the same path. When
    ``source_fingerprint`` is present, only a fully validated v1 companion can
    establish a hit; legacy duration-only caches remain unbound-only behavior.
    ``hit_only`` raises :class:`voxweave.vocals.VocalsCacheMiss` rather than
    reading ``media`` when the cache cannot serve the request.
    """
    media = Path(media)
    cache_owner = Path(cache_media) if cache_media is not None else media
    bound = source_fingerprint is not None
    cache: Path | None = None
    separator_identity: dict[str, object] | None = None
    if separate:
        cache = cache_vocals_path(cache_owner)
        separator_identity = backend.separator_identity() if bound else None
    return acquire_16k(
        media,
        separate=separate,
        normalize=normalize,
        reporter=reporter,
        tmp=tmp,
        cache=cache,
        owner=cache_owner,
        fingerprint=source_fingerprint,
        separator=separator_identity,
        legacy_16k=True,
        purpose="align source",
        want_voc32=False,
        hit_only=hit_only,
    ).wav


@dataclass(frozen=True)
class _SelectedLegacyAlignResult:
    block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    spans: tuple[tuple[float, float], ...]
    all_units: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _PreparedQwenCall:
    source_index: int
    wav_path: Path
    text: str
    nominal_start: float
    nominal_end: float
    sample_geometry: tuple[int, int, int, int] | None


def _retain_qwen_owner_slice(raw_units: Sequence[dict]) -> list[dict]:
    """Assign one physical Qwen result to its single historical cue owner."""

    return list(raw_units)


def _seal_selected_legacy_align_result(
    block_units: Sequence[Sequence[Mapping[str, Any]]],
    spans: Sequence[tuple[float, float]],
    all_units: Sequence[Mapping[str, Any]],
) -> _SelectedLegacyAlignResult:
    """Seal the exact selected-legacy projection before strict AO work begins."""

    def seal_unit(unit: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "text": unit["text"],
            "start": unit["start"],
            "end": unit["end"],
        }

    return _SelectedLegacyAlignResult(
        tuple(tuple(seal_unit(unit) for unit in owner) for owner in block_units),
        tuple((start, end) for start, end in spans),
        tuple(seal_unit(unit) for unit in all_units),
    )


def _align_blocks(
    wav: Path,
    blocks: list[dict],
    iso: str,
    *,
    mms: bool,
    ctc_model: str | None,
    crops: list[tuple[float, float] | None],
    reporter: Reporter,
    tmp_chunks: list[Path],
    speech_spans: list[tuple[float, float]] | None = None,
    raw_call_observer: Callable[..., None] | None = None,
    qwen_invoker: Callable[..., Sequence[Any]] | None = None,
    backend_invoker: Callable[[Callable[[], Any]], Any] | None = None,
    physical_preparation_invoker: Callable[[Callable[[], Any]], Any] | None = None,
    legacy_distribution_invoker: Callable[[Callable[[], Any]], Any] | None = None,
    legacy_shift_invoker: Callable[[Callable[[], Any]], Any] | None = None,
) -> list[list[dict]]:
    """Route blocks to the configured aligner and return per-block units.

    Three paths — these ARE the hard-constraint full-pass routing; do NOT collapse them:
    - ja MMS: one full-file pass (``align_blocks_full_mms``).
    - en wav2vec2 CTC: one full-file windowed-emission pass (``align_blocks_full_ctc``).
    - zh·yue (no CTC config): per-cue tight-crop Qwen — each cue gets its own audio slice so
      error is contained within the sentence and inter-sentence pauses are preserved.

    Per-cue slices are appended to ``tmp_chunks`` for the caller's ``finally`` to clean up.
    """
    if (legacy_distribution_invoker is None) != (legacy_shift_invoker is None):
        raise ValueError("legacy projection phase invokers must be supplied together")

    # MMS/CTC full passes: cue (start,end) bounds are used ONLY as silence anchors to split
    # movie-length audio into memory-sized chunks when it overflows the single-pass DP
    # budget — NOT to crop/route per cue. These routes are routing-free because the input
    # VTT timestamps are exactly what may be wrong (the reason to re-align); the global DP
    # self-locates every word. None for cues without timestamps. (The Qwen per-cue route
    # below crops each cue from ``crops`` instead.)
    bounds = [
        (b["start"], b["end"])
        if b["start"] is not None and b["end"] is not None
        else None
        for b in blocks
    ]
    if mms:
        reporter.task("full-file alignment (MMS)", 1)
        units = backend.align_blocks_full_mms(
            wav,
            [b.get("alignment_text", b["text"]) for b in blocks],
            iso,
            bounds=bounds,
            _raw_call_observer=raw_call_observer,
            _backend_invoker=backend_invoker,
            _preparation_invoker=physical_preparation_invoker,
            _legacy_distribution_invoker=legacy_distribution_invoker,
            _legacy_shift_invoker=legacy_shift_invoker,
        )
        reporter.advance(1)
        return units
    if ctc_model:  # en wav2vec2: windowed emission + single global DP (routing-free)
        reporter.task("full-file alignment (CTC)", 1)
        units = backend.align_blocks_full_ctc(
            wav,
            [b.get("alignment_text", b["text"]) for b in blocks],
            iso,
            ctc_model,
            bounds=bounds,
            speech_spans=speech_spans,
            _raw_call_observer=raw_call_observer,
            _backend_invoker=backend_invoker,
            _preparation_invoker=physical_preparation_invoker,
            _legacy_distribution_invoker=legacy_distribution_invoker,
            _legacy_shift_invoker=legacy_shift_invoker,
        )
        reporter.advance(1)
        return units
    if os.environ.get("VOXWEAVE_VAD_EMISSION_MASK", "").strip() == "1":
        # The VAD emission mask is wired into the wav2vec2 CTC pass only.
        log.warning(
            "--vad-mask / VOXWEAVE_VAD_EMISSION_MASK has no effect on Qwen alignment"
            " (language %s); only the wav2vec2 CTC aligner applies it",
            iso,
        )
    reporter.task("per-cue alignment", len(blocks))
    block_units: list[list[dict]] = [[] for _ in blocks]

    def prepare_qwen_calls() -> list[_PreparedQwenCall]:
        prepared: list[_PreparedQwenCall] = []
        for i, crop in enumerate(crops):
            text = realign.join_block_texts(
                [blocks[i].get("alignment_text", blocks[i]["text"])], iso
            )
            if crop is None or not text:  # insertion block or empty: skip
                reporter.advance(1)
                continue
            cs, ce = crop
            observed_geometry: tuple[int, int, int, int] | None = None

            def observe_sample_geometry(
                sample_start: int,
                sample_end: int,
                sample_rate: int,
                sample_count: int,
            ) -> None:
                nonlocal observed_geometry
                if observed_geometry is not None:
                    raise RuntimeError("Qwen slice reported sample geometry twice")
                observed_geometry = (
                    sample_start,
                    sample_end,
                    sample_rate,
                    sample_count,
                )

            cwav = slice_wav(
                wav,
                cs,
                ce,
                _sample_geometry_observer=observe_sample_geometry,
                _canonical_qwen_failures=True,
            )
            tmp_chunks.append(cwav)
            prepared.append(
                _PreparedQwenCall(
                    source_index=i,
                    wav_path=cwav,
                    text=text,
                    nominal_start=float(cs),
                    nominal_end=float(ce),
                    sample_geometry=observed_geometry,
                )
            )
        return prepared

    prepared_qwen = (
        prepare_qwen_calls()
        if physical_preparation_invoker is None
        else cast(
            list[_PreparedQwenCall],
            physical_preparation_invoker(prepare_qwen_calls),
        )
    )
    pending_qwen: list[tuple[int, list[dict], float]] = []
    for prepared in prepared_qwen:
        geometry = prepared.sample_geometry
        if qwen_invoker is None:
            raw_units = backend.align_text(prepared.wav_path, prepared.text, iso)
        else:
            raw_units = cast(
                list[dict[str, Any]],
                qwen_invoker(
                    lambda: backend.align_text(
                        prepared.wav_path,
                        prepared.text,
                        iso,
                    ),
                    prepared.source_index,
                    prepared.nominal_start,
                    prepared.nominal_end,
                    audio_sample_start=None if geometry is None else geometry[0],
                    audio_sample_end=None if geometry is None else geometry[1],
                    sample_rate=None if geometry is None else geometry[2],
                    sample_count=None if geometry is None else geometry[3],
                ),
            )

        pending_qwen.append((prepared.source_index, raw_units, prepared.nominal_start))
        # One tick per aligned cue as it lands (skipped cues ticked at preparation),
        # so the bar moves with the work instead of jumping from 0/N at the end.
        reporter.advance(1)

    if not pending_qwen:
        with align_runtime_activity("AO-07", "no-physical-calls"):
            pass

    def retain_owners() -> list[tuple[int, list[dict], float]]:
        return [
            (index, _retain_qwen_owner_slice(raw_units), origin)
            for index, raw_units, origin in pending_qwen
        ]

    retained_owners = (
        retain_owners()
        if legacy_distribution_invoker is None
        else cast(
            list[tuple[int, list[dict], float]],
            legacy_distribution_invoker(retain_owners),
        )
    )

    def shift_owners() -> list[list[dict]]:
        for index, owner_units, origin in retained_owners:
            block_units[index] = shift_units(owner_units, origin)
        return block_units

    block_units = (
        shift_owners()
        if legacy_shift_invoker is None
        else cast(list[list[dict]], legacy_shift_invoker(shift_owners))
    )
    return block_units


def _notify_align_shadow_observer(
    observer: Callable[[object], object],
    *,
    selection: Any,
    input_summary: Mapping[str, Any],
    prepared_audio_sha256: str,
) -> None:
    """Build rich/minimal observation after disposal; never change production."""
    from voxweave.align_evidence import encode_align_evidence

    evidence_sha256 = hashlib.sha256(
        encode_align_evidence(selection.evidence)
    ).hexdigest()
    try:
        from voxweave import align_shadow

        artifact = align_shadow.build_rich_align_shadow_artifact(
            selection=selection,
            input_summary=input_summary,
            prepared_audio_sha256=prepared_audio_sha256,
        )
    except Exception as rich_error:
        log.warning("rich align shadow construction failed: %s", rich_error)
        try:
            from voxweave import align_shadow_minimal

            artifact = align_shadow_minimal.build_minimal_align_shadow_failure_artifact(
                context_content_digest=selection.context.context_content_digest,
                receipt_digest=selection.result.receipt_digest,
                engine_family=selection.verified.engine_family,
                vtt_sha256=selection.verified.vtt_sha256,
                json_sha256=selection.verified.main_json_sha256,
                evidence_sha256=evidence_sha256,
                prior_failure=(
                    selection.observation_failure or selection.result.v2_status.failure
                ),
            )
        except Exception as minimal_error:
            failure = CanonicalFailure(
                "shadow-artifact-unavailable",
                "minimal-artifact",
                "minimal-artifact-construction",
            )
            log.warning(
                "align shadow artifact unavailable: %s",
                minimal_error,
                extra={"failure": failure},
            )
            return
    try:
        observer(artifact)
    except Exception as exc:
        failure = CanonicalFailure(
            "observer-failed",
            "observer",
            "observer-callback",
        )
        log.warning(
            "align shadow observer failed: %s",
            exc,
            extra={"failure": failure},
        )


def _media_not_found_error(vtt_path: Path) -> FileNotFoundError:
    """The classified error :func:`align` raises when the source media is missing."""
    exc = FileNotFoundError(
        f"source media for {vtt_path.name} not found (expected sibling with same stem); "
        f"align needs the original file to re-align, or specify --media"
    )
    _attach_canonical_failure(
        exc,
        kind="media-identity-invalid",
        phase="media",
        detail_code="media-not-found",
    )
    return exc


@dataclass(frozen=True)
class _CachedMediaIdentity:
    """Selected-media identity for align when a bound vocals cache serves the audio.

    Stands where a :class:`MediaSnapshot` would: the media bytes are never read,
    so only the fingerprint the cache companion was validated against is needed.
    """

    fingerprint: str


def _acquire_align_audio(
    media: Path,
    *,
    separate: bool,
    normalize: bool,
    bind_voiceprint: bool,
    media_input_fingerprint: str,
    rep: Reporter,
    tmp: list[Path],
    snapshots: ExitStack,
) -> tuple[Path, MediaSnapshot | _CachedMediaIdentity | None]:
    # Prepare-audio stage of align: a vocals-cache hit probe, the private media
    # snapshot fallback, then the AO-04 acquisition. The caller owns ``snapshots``
    # and closes it; this stage only enters contexts into it.
    selected_snapshot: MediaSnapshot | _CachedMediaIdentity | None = None
    acquisition_media = media
    cached_wav: Path | None = None
    if bind_voiceprint and separate:
        # A bound vocals-cache hit decodes the cache and never reads the
        # media, so it needs no private copy of the whole file: the cache
        # companion binds the vocals to this fingerprint, and the commit
        # rechecks the media against it.
        try:
            cached_wav = _prepare_16k_for_align(
                media,
                separate=separate,
                normalize=normalize,
                reporter=rep,
                tmp=tmp,
                cache_media=media,
                source_fingerprint=media_input_fingerprint,
                hit_only=True,
            )
        except VocalsCacheMiss:
            pass
        else:
            selected_snapshot = _CachedMediaIdentity(media_input_fingerprint)
    if bind_voiceprint and cached_wav is None:
        try:
            selected_snapshot = snapshots.enter_context(MediaSnapshot(media))
        except SnapshotUnavailable as exc:
            log.warning(
                "voiceprint binding will be omitted during align: "
                "selected media snapshot unavailable: %s",
                exc,
            )
        else:
            acquisition_media = selected_snapshot.path
    with align_runtime_activity("AO-04", "prepared-audio-and-cache"):
        wav = (
            cached_wav
            if cached_wav is not None
            else _prepare_16k_for_align(
                acquisition_media,
                separate=separate,
                normalize=normalize,
                reporter=rep,
                tmp=tmp,
                cache_media=media,
                source_fingerprint=(
                    selected_snapshot.fingerprint
                    if selected_snapshot is not None
                    else None
                ),
            )
        )
    return wav, selected_snapshot


def align(
    vtt_path: Path,
    *,
    media_path: Path | None = None,
    separate: bool = True,
    normalize: bool = False,
    lang_override: str | None = None,
    reporter: Reporter | None = None,
    _shadow_observer: Callable[[object], object] | None = None,
    _expected_vtt_sha256: str | None = None,
) -> Path:
    """Re-align edited VTT text against original audio; overwrite VTT and update JSON.

    The language's configured aligner (:func:`voxweave.config.align_model_for`)
    picks one of three routes:

    - MMS full pass (default for ja): one global forced alignment of all cue text
      over the whole audio; cue timestamps serve only as silence anchors when the
      audio overflows the single-pass budget.
    - CTC full pass (wav2vec2, default for en): the same routing-free global
      alignment over windowed emissions (optionally soft-masked by the persisted
      ``vad_speech`` under ``--vad-mask``).
    - Qwen per-cue crop (every language without an MMS/CTC aligner, e.g. zh/yue):
      each block is routed to its audio window (via word_segments or VTT
      timestamps), tightly cropped and aligned on its own.

    Insertion blocks are then interpolated and timing is written. ASR is not re-run;
    smart_split is not touched. All models run in-process (no network calls).
    """
    vtt_path = require_vtt(Path(vtt_path))  # align overwrites the input as VTT
    require_output_name(swap_ext(vtt_path, ".json"))  # refuse before the model work
    explicit_media_requested = media_path is not None
    rep = reporter or Reporter()
    rep.plan(("read subtitles", "prepare audio", "align subtitles", "write outputs"))
    rep.step("read subtitles")
    json_path = swap_ext(vtt_path, ".json")
    with align_runtime_activity("AO-01", "vtt-generation-snapshot"):
        try:
            vtt_input_bytes = vtt_path.read_bytes()
        except OSError as exc:
            _attach_canonical_failure(
                exc,
                kind="subtitle-snapshot-failed",
                phase="snapshot",
                detail_code="vtt-read",
            )
            raise
        if (
            _expected_vtt_sha256 is not None
            and hashlib.sha256(vtt_input_bytes).hexdigest() != _expected_vtt_sha256
        ):
            raise episode_transaction.InputStaleError(
                "vtt-generation", "input changed before alignment; re-run"
            )
    with align_runtime_activity("AO-01", "sibling-generation-snapshot"):
        try:
            expected_json = episode_transaction.capture_file_generation(json_path)
        except OSError as exc:
            _attach_canonical_failure(
                exc,
                kind="subtitle-snapshot-failed",
                phase="snapshot",
                detail_code="sibling-read",
            )
            raise
    json_input_bytes = expected_json.bytes_value
    from voxweave.align_snapshot import decode_sibling_json_snapshot

    with align_runtime_activity("AO-02", "sibling-decode-and-raw-carriers"):
        try:
            sibling_snapshot = decode_sibling_json_snapshot(
                json_path.name,
                json_input_bytes,
            )
        except RuntimeError as exc:
            _attach_json_decode_failure(exc)
            raise
        data = sibling_snapshot.thaw_legacy()
        pair_declared = "voiceprint_capture" in data or "voiceprint_media" in data
        voiceprint_pair = (
            _replay_voiceprint_pair(
                data,
                json_input_bytes,
                source=json_path.name,
            )
            if json_input_bytes is not None
            else None
        )
    # align re-times an existing cue stream; it never re-segments, so it only
    # labels (and later preserves) whatever produced that stream.
    log.debug("re-timing %s (%s)", vtt_path.name, resolve_segmentation_manifest(data))
    word_segments = data.get("word_segments", [])

    if lang_override:
        iso = require_aligner_language(lang_override)
    else:
        # A detected language outside the aligner's set falls back to English;
        # only an explicit override has to name a supported one.
        iso = to_iso_or(data.get("language") or "english", "en")

    from voxweave.align_snapshot import decode_align_snapshot

    with align_runtime_activity("AO-02", "subtitle-decode"):
        try:
            input_snapshot = decode_align_snapshot(
                vtt_path.name,
                vtt_input_bytes,
                json_input_bytes,
                effective_iso=iso,
                sibling_snapshot=sibling_snapshot,
            )
        except (RuntimeError, UnicodeDecodeError) as exc:
            _attach_vtt_decode_failure(exc, vtt_path.name)
            raise

    with align_runtime_activity("AO-03", "selected-media-identity"):
        media = Path(media_path) if media_path else find_subtitle_media(vtt_path)
        if media is None or not media.exists():
            raise _media_not_found_error(vtt_path)
        try:
            media_input_fingerprint = media_fingerprint(media)
        except OSError as exc:
            _attach_canonical_failure(
                exc,
                kind="media-identity-invalid",
                phase="media",
                detail_code="media-fingerprint",
            )
            raise

    # Full-file single-pass alignment (whisperx fork align_ctc) for both MMS (ja) and wav2vec2
    # CTC (en): concatenate all cue text, run one global monotone forced-align over the whole
    # audio, slice units back per cue by char/word count. The global path self-locates every
    # token (blank / <star> absorbs silence + song spans), immune to per-cue cropping drift
    # (observed: wrong coarse crop displaced エルダドワーフ by 11s; crammed en "blocks" into dead
    # air). Needs no has_ts/route/crop. ja MMS emission is windowed inside ctc-forced-aligner;
    # en wav2vec2 emission is windowed in align_blocks_full_ctc (full-file xlsr is O(T^2) -> OOM
    # at 23min). zh·yue have no CTC config -> per-cue tight-crop Qwen (routing+crop below). Do
    # NOT revert ja to per-cue MMS: repeated small ONNX calls corrupt the heap (~180-226 cues).
    from voxweave.config import align_model_for

    with align_runtime_activity("AO-03", "route-family-plan"):
        mms = backend.uses_mms(iso)
        ctc_model = None if mms else align_model_for(iso)
        align_route = resolve_align_route(mms, ctc_model)
        full_pass = align_route.full_pass
        route_kind = align_route.kind
    delivery_order = (
        tuple(range(len(input_snapshot.blocks)))
        if full_pass
        else input_snapshot.qwen_delivery_order
    )
    blocks = _align_blocks_from_snapshot(input_snapshot, delivery_order)
    crops: list[
        tuple[float, float] | None
    ] = []  # set + looped only on the per-cue (zh·yue) path
    if not full_pass:
        crops = _plan_qwen_crops(blocks, word_segments, json_path)

    tmp: list[Path] = []
    tmp_chunks: list[Path] = []
    snapshots = ExitStack()
    selected_snapshot: MediaSnapshot | _CachedMediaIdentity | None = None
    align_context = None
    completed_selection = None
    observation_input: Mapping[str, Any] | None = None
    prepared_audio_sha256: str | None = None
    aligned_cue_count = 0
    aligned_unit_count = 0
    production_failure: BaseException | None = None
    try:
        try:
            rep.step("prepare audio")
            wav, selected_snapshot = _acquire_align_audio(
                media,
                separate=separate,
                normalize=normalize,
                bind_voiceprint=voiceprint_pair is not None,
                media_input_fingerprint=media_input_fingerprint,
                rep=rep,
                tmp=tmp,
                snapshots=snapshots,
            )
            from voxweave import align_orchestration

            with align_runtime_activity("AO-04", "prepared-audio-digest"):
                prepared_audio_sha256 = align_orchestration.file_sha256(wav)
            from voxweave.align_inputs import LegacyAlignPolicy

            legacy_policy = LegacyAlignPolicy(
                MIN_CUE_SEC,
                TINY_CUE_SEC,
                TINY_CUE_TARGET,
            )
            stored_manifest_value = data.get("segmentation")
            strict_shot_changes = data.get("shot_changes")
            strict_sing_spans = data.get("sing_spans")
            with align_runtime_activity("AO-05", "context-and-limit-profile-issuance"):
                align_context = align_orchestration.issue_public_align_context(
                    target_path=vtt_path,
                    sibling_path=json_path,
                    media_path=media,
                    prepared_audio_path=wav,
                    expected_vtt=episode_transaction.FileGeneration(
                        True, vtt_input_bytes
                    ),
                    expected_json=expected_json,
                    expected_vtt_sha256=_expected_vtt_sha256,
                    media_fingerprint=media_input_fingerprint,
                    effective_iso=iso,
                    route_kind=route_kind,
                    blocks=blocks,
                    prepared_audio_sha256=prepared_audio_sha256,
                    legacy_policy=legacy_policy,
                    stored_language=data.get("language"),
                    segmentation=stored_manifest_value,
                    strict_shot_changes=strict_shot_changes,
                    strict_sing_spans=strict_sing_spans,
                    explicit_media=explicit_media_requested,
                    block_content_sha256=input_snapshot.block_content_sha256,
                )
            bind_align_runtime_identity(
                route_kind=align_context.route_kind,
                engine_family=align_context.engine_family,
            )
            observation_input = {
                "context_content_digest": align_context.context_content_digest,
                "vtt_sha256": hashlib.sha256(vtt_input_bytes).hexdigest(),
                "sibling_present": json_input_bytes is not None,
                "sibling_sha256": (
                    None
                    if json_input_bytes is None
                    else hashlib.sha256(json_input_bytes).hexdigest()
                ),
                "media_fingerprint": media_input_fingerprint,
                "media_logical_id": align_orchestration._media_logical_identity(
                    media,
                    explicit_media=explicit_media_requested,
                ),
                "effective_iso": iso,
                "route": route_kind,
                "block_count": len(blocks),
                "block_content_sha256": input_snapshot.block_content_sha256,
                "profile_source": (
                    "manifest-absent"
                    if not isinstance(stored_manifest_value, Mapping)
                    else "stored-or-default"
                ),
            }
            from voxweave.align_acquisition import (
                _fresh_alignment_backend_invoker,
                _fresh_alignment_call_observer,
                _fresh_alignment_qwen_invoker,
                begin_fresh_alignment,
                seal_fresh_alignment,
            )

            prepared_sample_rate, prepared_sample_count = _prepared_audio_geometry(wav)

            model_facts = {
                "route": route_kind,
                "language": iso,
                "backend": align_route.backend_label(lambda: backend._use_mlx()),
                "model": align_route.model_label(backend.ALIGNER_MODEL),
                "sample_rate": prepared_sample_rate,
            }
            route_facts = _align_route_facts(route_kind, iso, blocks, crops)

            with align_runtime_activity("AO-05", "acquisition-authorization"):
                fresh_session = begin_fresh_alignment(
                    align_context,
                    alignment_texts=tuple(
                        str(block.get("alignment_text", block["text"]))
                        for block in blocks
                    ),
                    source_texts=tuple(str(block["text"]) for block in blocks),
                    source_indices=tuple(
                        int(block["source_index"]) for block in blocks
                    ),
                    language=iso,
                    prepared_audio_sample_count=prepared_sample_count,
                    sample_rate=prepared_sample_rate,
                    backend_model_config_digest=_stable_fact_digest(model_facts),
                    route_input_digest=_stable_fact_digest(route_facts),
                    backend_model_config_facts=model_facts,
                    route_input_facts=route_facts,
                )
            capture_raw_call = _fresh_alignment_call_observer(fresh_session)
            invoke_backend_call = _fresh_alignment_backend_invoker(fresh_session)
            invoke_qwen_call = _fresh_alignment_qwen_invoker(fresh_session)

            def invoke_physical_preparation(operation: Callable[[], Any]) -> Any:
                with align_runtime_activity("AO-06", "physical-call-preparation"):
                    return operation()

            def invoke_legacy_distribution(operation: Callable[[], Any]) -> Any:
                with align_runtime_activity("AO-08", "legacy-owner-slice"):
                    return operation()

            def invoke_legacy_shift(operation: Callable[[], Any]) -> Any:
                with align_runtime_activity("AO-09", "legacy-time-transform"):
                    return operation()

            rep.step("align subtitles")
            block_units = _align_blocks(
                wav,
                blocks,
                iso,
                mms=mms,
                ctc_model=ctc_model,
                crops=crops,
                reporter=rep,
                tmp_chunks=tmp_chunks,
                # vad_speech persisted by transcribe (same media timeline): lets the CTC
                # full pass mask non-speech emissions; absent/empty -> no masking
                speech_spans=spans_in(data.get("vad_speech")),
                raw_call_observer=capture_raw_call,
                qwen_invoker=invoke_qwen_call,
                backend_invoker=invoke_backend_call,
                physical_preparation_invoker=invoke_physical_preparation,
                legacy_distribution_invoker=invoke_legacy_distribution,
                legacy_shift_invoker=invoke_legacy_shift,
            )

            # position_units_with_vad is not needed here (unlike the transcribe path): on
            # the Qwen per-cue route the tight crop already stops a cue's last word from
            # drifting into inter-sentence silence, and the MMS/CTC full passes absorb
            # silence in their blank tokens.
            with align_runtime_activity("AO-10", "group-block-spans"):
                final, all_units = realign.group_block_spans(block_units)
            with align_runtime_activity("AO-10", "common-all-empty-decision"):
                if not all_units:
                    exc = RuntimeError(f"no aligned units for {media.name}")
                    _attach_canonical_failure(
                        exc,
                        kind="no-aligned-units",
                        phase="fresh-acquisition",
                        detail_code="all-block-units-empty",
                    )
                    raise exc
            # Preserve the exact historical helper chain.  The selected result is sealed
            # before seal_fresh_alignment may begin AO-11 strict recursive capture.
            with align_runtime_activity("AO-10", "fill-insert-blocks"):
                filled = realign.fill_insert_blocks(final)
            with align_runtime_activity("AO-10", "enforce-min-duration"):
                duration_enforced = realign.enforce_min_duration(
                    filled,
                    min_dur=MIN_CUE_SEC,
                )
            with align_runtime_activity("AO-10", "rescue-tiny-cues"):
                rescued = realign.rescue_tiny_cues(
                    duration_enforced,
                    trig=TINY_CUE_SEC,
                    target=TINY_CUE_TARGET,
                )
            with align_runtime_activity("AO-10", "clamp-spans"):
                spans_filled = realign.clamp_spans(rescued)
            with align_runtime_activity("AO-10", "seal-selected-legacy-result"):
                selected_legacy = _seal_selected_legacy_align_result(
                    block_units,
                    spans_filled,
                    all_units,
                )
            acquisition = seal_fresh_alignment(fresh_session)

            # Preserve vad_speech / shot_changes from the original JSON (computed by
            # transcribe from the original media; align does not recompute them).
            keep_vad = spans_in(data.get("vad_speech"))
            keep_shots = [float(t) for t in data.get("shot_changes") or []] or None
            keep_sing = spans_in(data.get("sing_spans"))
            keep_turns = sibling_snapshot.carrier("speaker_turns")
            # align never re-segments, so the segmentation manifest is preserved
            # verbatim (and stays absent when the document never had one).
            stored_manifest = data.get("segmentation")
            keep_manifest = (
                stored_manifest if isinstance(stored_manifest, Mapping) else None
            )
            preserve_pair = bool(
                voiceprint_pair is not None
                and selected_snapshot is not None
                and selected_snapshot.fingerprint == voiceprint_pair[1]
            )
            if (
                voiceprint_pair is not None
                and selected_snapshot is not None
                and not preserve_pair
            ):
                log.warning(
                    "voiceprint binding omitted during align: "
                    "selected media does not match the sibling binding"
                )
            assert align_context is not None
            selection = align_orchestration.build_align_selection(
                context=align_context,
                acquisition=acquisition,
                blocks=blocks,
                block_units=selected_legacy.block_units,
                spans=selected_legacy.spans,
                all_units=selected_legacy.all_units,
                language=iso,
                vad_speech=keep_vad,
                shot_changes=keep_shots,
                sing_spans=keep_sing,
                speaker_turns=keep_turns,
                voiceprint_pair=voiceprint_pair if preserve_pair else None,
                manifest=keep_manifest,
                shadow_requested=shadow_enabled(),
                strict_input_status=sibling_snapshot.strict_input_status,
                legacy_policy=legacy_policy,
                stored_language=data.get("language"),
                strict_shot_changes=strict_shot_changes,
                strict_sing_spans=strict_sing_spans,
            )
            if observation_input is not None:
                observation_input = {
                    **observation_input,
                    "profile_source": selection.profile_status.source,
                }
            cleanup: list[episode_transaction.ArtifactCleanup] = []
            if pair_declared and not preserve_pair:
                cleanup.extend(
                    episode_transaction.speaker_artifact_cleanups(media, vtt_path)
                )

            rep.step("write outputs")
            rep.stage("write VTT + JSON")
            from voxweave.align_evidence import encode_align_evidence

            with align_runtime_activity("AO-22", "selected-evidence-preencode"):
                try:
                    evidence_bytes = encode_align_evidence(selection.evidence)
                except BaseException as exc:
                    _attach_canonical_failure(
                        exc,
                        kind="preencode-failed",
                        phase="preencode",
                        detail_code="evidence-encode",
                    )
                    raise
            evidence_path = artifacts.align_evidence_path(media, vtt_path)
            cleanup.extend(
                episode_transaction.evidence_cleanups(
                    vtt_path, media=media, keep=evidence_path
                )
            )
            evidence_artifact = episode_transaction.EvidencePublication(
                evidence_path,
                evidence_bytes,
            )
            episode_transaction.commit_primary_outputs(
                command="align",
                episode_path=media,
                json_path=json_path,
                vtt_path=vtt_path,
                expected_json=expected_json,
                expected_vtt=episode_transaction.FileGeneration(True, vtt_input_bytes),
                main_json_bytes=selection.verified.main_json_bytes,
                vtt_bytes=selection.verified.vtt_bytes,
                cleanup_paths=tuple(cleanup),
                context=selection.context,
                media_path=media,
                expected_media_fingerprint=media_input_fingerprint,
                expected_voiceprint_media_fingerprint=(
                    voiceprint_pair[1]
                    if voiceprint_pair is not None and selected_snapshot is not None
                    else None
                ),
                expected_pair_decision=(
                    preserve_pair
                    if voiceprint_pair is not None and selected_snapshot is not None
                    else None
                ),
                evidence_artifact=evidence_artifact,
            )
            completed_selection = selection
            aligned_cue_count = len(blocks)
            aligned_unit_count = len(selected_legacy.all_units)
        except BaseException as exc:
            production_failure = exc
            raise
        finally:
            # Release aligner singleton VRAM (separation self-releases earlier).
            disposal_failure: BaseException | None = None
            with align_runtime_activity("AO-24", "media-snapshot-disposal"):
                try:
                    snapshots.close()
                except BaseException as exc:
                    disposal_failure = _record_disposal_failure(
                        production_failure,
                        disposal_failure,
                        exc,
                        detail_code="media-snapshot-residue",
                    )
            with align_runtime_activity("AO-24", "backend-and-audio-temp-disposal"):
                # Guarded: a release that raises (e.g. torch.cuda.empty_cache after a CUDA
                # fault) must neither mask the original error nor skip the temp cleanup.
                try:
                    backend.release()
                except Exception as exc:
                    log.warning("releasing the alignment models failed: %r", exc)
                for p in tmp:
                    try:
                        p.unlink(missing_ok=True)
                    except BaseException as exc:
                        disposal_failure = _record_disposal_failure(
                            production_failure,
                            disposal_failure,
                            exc,
                            detail_code="audio-temp-residue",
                        )
                for c in tmp_chunks:
                    try:
                        c.unlink(missing_ok=True)
                    except BaseException as exc:
                        disposal_failure = _record_disposal_failure(
                            production_failure,
                            disposal_failure,
                            exc,
                            detail_code="audio-temp-residue",
                        )
            with align_runtime_activity("AO-24", "selection-role-retirement"):
                if align_context is not None:
                    from voxweave import align_orchestration

                    align_orchestration.retire_align_selection(align_context)
            if production_failure is None and disposal_failure is not None:
                raise disposal_failure
        with align_runtime_activity("AO-25", "artifact-and-observer-dispatch"):
            if (
                shadow_enabled()
                and _shadow_observer is not None
                and completed_selection is not None
                and observation_input is not None
                and prepared_audio_sha256 is not None
            ):
                _notify_align_shadow_observer(
                    _shadow_observer,
                    selection=completed_selection,
                    input_summary=observation_input,
                    prepared_audio_sha256=prepared_audio_sha256,
                )
    finally:
        # Only after AO-25: the shadow observer still reads the registries.
        if align_context is not None:
            from voxweave import align_orchestration

            align_orchestration.release_align_selection(align_context)
    log.info(
        "aligned %s → %d cues, %d units",
        vtt_path.name,
        aligned_cue_count,
        aligned_unit_count,
    )
    return vtt_path
