from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from voxweave import (
    artifacts,
    backend,
    chunking,
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
from voxweave.align_blocks import AlignBlocksRequest, align_blocks as run_align_blocks
from voxweave.align_run import (
    MIN_CUE_SEC as MIN_CUE_SEC,
    TINY_CUE_SEC as TINY_CUE_SEC,
    TINY_CUE_TARGET as TINY_CUE_TARGET,
    AlignInputs,
    AlignRun,
    CachedMediaIdentity,
    begin_fresh_acquisition,
    build_align_selection,
    commit_align_outputs,
    dispose_align_run,
    invoke_legacy_distribution,
    invoke_legacy_shift,
    invoke_physical_preparation,
    select_legacy_alignment,
)
from voxweave.align_route import (
    AlignRoute as AlignRoute,
    resolve_align_route as resolve_align_route,
    _align_blocks_from_snapshot,
    _plan_qwen_crops,
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


def _retain_qwen_owner_slice(raw_units: Sequence[dict]) -> list[dict]:
    """Assign one physical Qwen result to its single historical cue owner."""

    return list(raw_units)


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

    :mod:`voxweave.align_blocks` holds one aligner per route (ja MMS full pass,
    en CTC full pass, zh·yue per-cue Qwen); do NOT collapse them. The Qwen route's
    audio slicing, time shifting and owner slicing are looked up on this module
    when the call starts.

    Per-cue slices are appended to ``tmp_chunks`` for the caller's ``finally`` to clean up.
    """
    return run_align_blocks(
        resolve_align_route(mms, ctc_model).kind,
        AlignBlocksRequest(
            wav=wav,
            blocks=blocks,
            iso=iso,
            ctc_model=ctc_model,
            crops=crops,
            reporter=reporter,
            tmp_chunks=tmp_chunks,
            speech_spans=speech_spans,
            raw_call_observer=raw_call_observer,
            qwen_invoker=qwen_invoker,
            backend_invoker=backend_invoker,
            physical_preparation_invoker=physical_preparation_invoker,
            legacy_distribution_invoker=legacy_distribution_invoker,
            legacy_shift_invoker=legacy_shift_invoker,
            slice_audio=slice_wav,
            shift_units=shift_units,
            retain_owner_slice=_retain_qwen_owner_slice,
        ),
    )


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
) -> tuple[Path, MediaSnapshot | CachedMediaIdentity | None]:
    # Prepare-audio stage of align: a vocals-cache hit probe, the private media
    # snapshot fallback, then the AO-04 acquisition. The caller owns ``snapshots``
    # and closes it; this stage only enters contexts into it.
    selected_snapshot: MediaSnapshot | CachedMediaIdentity | None = None
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
            selected_snapshot = CachedMediaIdentity(media_input_fingerprint)
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


def _select_align_media(vtt_path: Path, media_path: Path | None) -> tuple[Path, str]:
    # AO-03: the media align re-times against (explicit or the VTT's sibling) and
    # its fingerprint.
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
    return media, media_input_fingerprint


def _plan_align_route(iso: str) -> tuple[bool, str | None, AlignRoute]:
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
    return mms, ctc_model, align_route


def _read_align_inputs(
    vtt_path: Path,
    *,
    media_path: Path | None,
    lang_override: str | None,
    expected_vtt_sha256: str | None,
) -> AlignInputs:
    # Read stage of align (AO-01 to AO-03): snapshot the VTT and the sibling JSON,
    # decode both, select the media and plan the aligner route.
    explicit_media_requested = media_path is not None
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
            expected_vtt_sha256 is not None
            and hashlib.sha256(vtt_input_bytes).hexdigest() != expected_vtt_sha256
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

    media, media_input_fingerprint = _select_align_media(vtt_path, media_path)
    mms, ctc_model, align_route = _plan_align_route(iso)
    full_pass = align_route.full_pass
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
    return AlignInputs(
        vtt_path=vtt_path,
        json_path=json_path,
        vtt_input_bytes=vtt_input_bytes,
        expected_json=expected_json,
        json_input_bytes=json_input_bytes,
        sibling_snapshot=sibling_snapshot,
        data=data,
        pair_declared=pair_declared,
        voiceprint_pair=voiceprint_pair,
        iso=iso,
        input_snapshot=input_snapshot,
        media=media,
        media_input_fingerprint=media_input_fingerprint,
        explicit_media=explicit_media_requested,
        mms=mms,
        ctc_model=ctc_model,
        route=align_route,
        blocks=blocks,
        crops=crops,
    )


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
    rep = reporter or Reporter()
    rep.plan(("read subtitles", "prepare audio", "align subtitles", "write outputs"))
    rep.step("read subtitles")
    inputs = _read_align_inputs(
        vtt_path,
        media_path=media_path,
        lang_override=lang_override,
        expected_vtt_sha256=_expected_vtt_sha256,
    )
    run = AlignRun()
    production_failure: BaseException | None = None
    try:
        try:
            rep.step("prepare audio")
            binding = begin_fresh_acquisition(
                inputs,
                run,
                acquire_audio=_acquire_align_audio,
                rep=rep,
                separate=separate,
                normalize=normalize,
                expected_vtt_sha256=_expected_vtt_sha256,
            )
            fresh_session = binding.fresh_session
            from voxweave.align_acquisition import seal_fresh_alignment

            rep.step("align subtitles")
            block_units = _align_blocks(
                binding.wav,
                inputs.blocks,
                inputs.iso,
                mms=inputs.mms,
                ctc_model=inputs.ctc_model,
                crops=inputs.crops,
                reporter=rep,
                tmp_chunks=run.tmp_chunks,
                # vad_speech persisted by transcribe (same media timeline): lets the CTC
                # full pass mask non-speech emissions; absent/empty -> no masking
                speech_spans=spans_in(inputs.data.get("vad_speech")),
                raw_call_observer=binding.capture_raw_call,
                qwen_invoker=binding.invoke_qwen_call,
                backend_invoker=binding.invoke_backend_call,
                physical_preparation_invoker=invoke_physical_preparation,
                legacy_distribution_invoker=invoke_legacy_distribution,
                legacy_shift_invoker=invoke_legacy_shift,
            )

            # AO-10 and the seal below stay in this body: scripts/p6_oracle.py
            # rewrites this seam in a copy of this file, so it must not move.
            # position_units_with_vad is not needed here (unlike the transcribe path): on
            # the Qwen per-cue route the tight crop already stops a cue's last word from
            # drifting into inter-sentence silence, and the MMS/CTC full passes absorb
            # silence in their blank tokens.
            with align_runtime_activity("AO-10", "group-block-spans"):
                final, all_units = realign.group_block_spans(block_units)
            selected_legacy = select_legacy_alignment(
                block_units, final, all_units, inputs.media
            )
            acquisition = seal_fresh_alignment(fresh_session)

            selection, preserve_pair, cleanup = build_align_selection(
                inputs, run, binding, acquisition, selected_legacy
            )
            commit_align_outputs(
                inputs, run, selection, preserve_pair, cleanup, rep=rep
            )
            run.completed_selection = selection
            run.aligned_cue_count = len(inputs.blocks)
            run.aligned_unit_count = len(selected_legacy.all_units)
        except BaseException as exc:
            production_failure = exc
            raise
        finally:
            dispose_align_run(run, production_failure)
        with align_runtime_activity("AO-25", "artifact-and-observer-dispatch"):
            if (
                shadow_enabled()
                and _shadow_observer is not None
                and run.completed_selection is not None
                and run.observation_input is not None
                and run.prepared_audio_sha256 is not None
            ):
                _notify_align_shadow_observer(
                    _shadow_observer,
                    selection=run.completed_selection,
                    input_summary=run.observation_input,
                    prepared_audio_sha256=run.prepared_audio_sha256,
                )
    finally:
        # Only after AO-25: the shadow observer still reads the registries.
        if run.align_context is not None:
            from voxweave import align_orchestration

            align_orchestration.release_align_selection(run.align_context)
    log.info(
        "aligned %s → %d cues, %d units",
        vtt_path.name,
        run.aligned_cue_count,
        run.aligned_unit_count,
    )
    return vtt_path
