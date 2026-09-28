from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from collections import Counter
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
from voxweave import sdh as sdh_mod
from voxweave.chunking import (
    decode_to_wav,
    pack_speech_segments,
    silence_gaps,
    slice_wav,
    vad_speech_segments,
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
    SHADOW_LANE_FINALIZER as SHADOW_LANE_FINALIZER,
    SHADOW_LANE_LEGACY_DISPLAY as SHADOW_LANE_LEGACY_DISPLAY,
)
from voxweave.debug import DebugSink, FileDebugSink
from voxweave.lang import (
    is_supported,
    reconcile_detected_language,
    to_iso_or,
    transcript_content_weight,
)
from voxweave.mediasnapshot import MediaSnapshot, SnapshotUnavailable

# Path, sidecar, vocals-cache and segmentation helpers moved to their own
# modules; the public names stay importable from here.
from voxweave.paths import MEDIA_EXTS as MEDIA_EXTS
from voxweave.paths import find_subtitle_media, swap_ext
from voxweave.progress import Reporter, progress_bridge
from voxweave.segmentation import (
    SEGMENTATION_ENGINE as SEGMENTATION_ENGINE,
    SEGMENTATION_MANIFEST_VERSION as SEGMENTATION_MANIFEST_VERSION,
    SegmentationResult as SegmentationResult,
    resolve_segmentation_manifest,
    segment_document,
)
from voxweave.sidecars import (
    align_evidence_candidates,
    artifact_owner,
    inspect_speakers_mapping_path,
    speaker_suggest_candidates,
    speakers_html_path,
    voiceprints_candidates,
    voiceprints_path,
)
from voxweave.sidecars import (
    speakers_mapping_path as speakers_mapping_path,
    speakers_suggest_path as speakers_suggest_path,
)
from voxweave.songdet import (
    detect_song_spans,
    excise_spans_from_segments,
    expand_spans_to_voiced_blocks,
    filter_short_spans,
    group_segments_by_spans,
    rescue_speech_segments,
    subtract_spans,
)
from voxweave.speakers import named_speaker_ids
from voxweave.subformats import require_subtitle
from voxweave.timestamps import shift_units
from voxweave.vocals import (
    ASR_LOUDNORM as ASR_LOUDNORM,
    CACHE_DIRNAME as CACHE_DIRNAME,
    CACHE_DUR_TOL_SEC as CACHE_DUR_TOL_SEC,
    SONGDET_SR,
    acquire_16k,
    cache_vocals_path,
)
from voxweave.vocals import cache_16k_path as cache_16k_path
from voxweave.voicebase import (
    Phase2DataError,
    VOICEPRINTS_MAX_BYTES,
    canonical_turns_digest,
    encode_json_bytes,
    media_fingerprint,
    mint_capture_id,
    require_capture_id,
    require_sha256,
    strict_json_object_loads,
    utc_timestamp,
    validate_voiceprints_mapping,
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


# ≤120s: long chunks occasionally trigger ASR repetition loops (stuck token ->
# zero-duration wall). Do NOT raise this to pack more; the risk and blast radius grow.
MAX_CHUNK_SEC = config._env_float("VOXWEAVE_MAX_CHUNK_SEC", 120.0)
# Spans shorter than this after expansion are kept as dialogue, not skipped.
# Real OP/ED runs 30-90s; short instrumental BGM scattered through speech would hurt ASR
# if dropped (env VOXWEAVE_MIN_SONG_SKIP_SEC).
MIN_SONG_SKIP_SEC = config._env_float("VOXWEAVE_MIN_SONG_SKIP_SEC", 8.0)
# Sensitive VAD threshold for snapping zero-duration units to original (pre-separation)
# audio. Silero default 0.5 misses back-channels (はい/ええ) attenuated by vocal separation;
# 0.25 catches them. Used only for snap positioning, not for chunk boundary decisions.
SNAP_VAD_THRESHOLD = config._env_float("VOXWEAVE_SNAP_VAD_THRESHOLD", 0.25)
# Fine VAD pass for song excision: a small min-silence (vs the 300ms chunking default)
# surfaces brief intra-segment pauses, so excision cut points land in real silence and
# never bisect a dialogue word. Only runs when song spans were detected.
SONG_FINE_SILENCE_MS = config._env_int("VOXWEAVE_SONG_FINE_SILENCE_MS", 100)
# Align-stage cue duration floor. Default 0 (disabled): enforce_min_duration only
# resolves overlaps without padding, so short back-channels keep their real ~0.6s.
# Set VOXWEAVE_MIN_CUE_SEC=0.8 to re-enable padding. Distinct from VOXWEAVE_SEG_MIN_CUE_SEC.
MIN_CUE_SEC = config._env_float("VOXWEAVE_MIN_CUE_SEC", 0.0)
# Flash-cue rescue (orthogonal to MIN_CUE_SEC): genuine flash cues (so/あ at 0.1-0.2s)
# are extended to TINY_CUE_TARGET, allowed to overlap only the immediately following cue.
# VOXWEAVE_TINY_CUE_SEC=0 disables.
TINY_CUE_SEC = config._env_float("VOXWEAVE_TINY_CUE_SEC", 0.2)
TINY_CUE_TARGET = config._env_float("VOXWEAVE_TINY_CUE_TARGET", 0.5)


@dataclass(frozen=True)
class VoiceprintCapture:
    """Embedding evidence kept alive from diarization through episode commit."""

    centroids: dict[str, list[float]]
    provenance: dict[str, object]
    turns: list[tuple[float, float, str]]


def _prefetch_voiceprint_models(
    voiceprint_model: str | None,
    lang_override: str | None,
    reporter: Reporter,
) -> bool:
    """Fetch and verify the voiceprint checkpoint(s) before any audio work.

    The capture itself only runs after separation, ASR and diarization; a
    missing or stalled download discovered there would waste the whole run.
    ``auto`` needs both embedders unless ``--language`` already fixes the language
    (resolved to ISO exactly as :func:`transcribe` resolves it). ``False``
    (after a warning) means voiceprints are off for this run; the subtitles are
    still produced. An invalid model choice raises ``ValueError``.
    """
    from voxweave import voiceembed

    forced = lang_override.strip() if lang_override else ""
    language_iso = to_iso_or(forced, "en") if forced else None
    reporter.stage("voiceprint models")
    try:
        voiceembed.prefetch_checkpoints(voiceprint_model, language_iso)
    except voiceembed.VoiceEmbeddingError as exc:
        log.warning(
            "voiceprint models unavailable; continuing without voiceprint capture: %s",
            exc,
        )
        return False
    return True


def _decoupled_voiceprint_capture(
    wav: Path,
    diarization: Any,
    spec: Any,
    *,
    reporter: Reporter,
) -> VoiceprintCapture | None:
    """Voiceprints from a dedicated embedder over the audio pyannote heard.

    ``wav`` is the exact 16 kHz mono file the diarizer was fed, so the centroids
    describe the signal the provenance ``audio`` block records. The embedder is
    released before returning. A failure to fetch or run it drops the capture
    with a warning (the subtitles are the primary output); it never falls back
    to another embedding space.
    """
    from voxweave import voiceembed

    if not diarization.turns:
        return None
    reporter.stage(f"speaker voiceprints ({spec.name})")
    try:
        captured = voiceembed.capture_voiceprints(
            wav,
            diarization.turns,
            spec,
            diarization.provenance,
        )
    except (voiceembed.VoiceEmbeddingError, OSError) as exc:
        log.warning(
            "voiceprint capture unavailable; continuing without capture: %s", exc
        )
        return None
    finally:
        voiceembed.release()
    if captured is None:
        return None
    centroids, provenance = captured
    return VoiceprintCapture(
        centroids=centroids,
        provenance=provenance,
        turns=diarization.turns,
    )


@dataclass(frozen=True)
class _ProcessPublication:
    path: Path
    landed: tuple[Path, ...]
    auxiliary_landed: tuple[Path, ...] = ()


def _attach_canonical_failure(
    exc: BaseException,
    *,
    kind: str,
    phase: str,
    detail_code: str,
) -> None:
    """Classify an unchanged public exception without replacing its boundary."""
    try:
        if not isinstance(getattr(exc, "failure", None), CanonicalFailure):
            setattr(exc, "failure", CanonicalFailure(kind, phase, detail_code))
    except Exception:
        pass


def _append_secondary_terminal(
    primary: BaseException,
    secondary_exception: BaseException,
    *,
    kind: str,
    phase: str,
    detail_code: str,
) -> None:
    """Append one later closed terminal without replacing the first exception."""
    secondary = SecondaryFailure(kind, phase, detail_code)
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
        exceptions = tuple(getattr(primary, "secondary_exceptions", ()))
        setattr(
            primary,
            "secondary_exceptions",
            exceptions + (secondary_exception,),
        )
    except Exception:
        pass


def _record_disposal_failure(
    production_failure: BaseException | None,
    disposal_failure: BaseException | None,
    exc: BaseException,
    *,
    detail_code: str,
) -> BaseException:
    primary = production_failure or disposal_failure
    if primary is None:
        _attach_canonical_failure(
            exc,
            kind="snapshot-dispose-failed",
            phase="dispose",
            detail_code=detail_code,
        )
        return exc
    _append_secondary_terminal(
        primary,
        exc,
        kind="snapshot-dispose-failed",
        phase="dispose",
        detail_code=detail_code,
    )
    return disposal_failure or exc


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


def _select_transcript_language(
    results: Sequence[tuple[str | None, str, Sequence[dict]]],
    override: str | None = None,
) -> str:
    """Select the file language from pre-alignment transcript content.

    Unit counts cannot be used here: a wrong aligner is liable to collapse Han
    prose into a few units (or fragment another script), making the original
    classification error self-reinforcing.  Normalize equivalent labels and
    weight each reconciled label by its transcript's alphanumeric content.
    """
    if override and override.strip():
        return override.strip()

    weights: Counter[str] = Counter()
    for detected, text, _units in results:
        if not text.strip():
            continue
        effective = reconcile_detected_language(detected, text)
        if not effective:
            continue
        key = to_iso_or(effective, None) or effective.strip().casefold()
        mass = transcript_content_weight(text)
        if mass:
            weights[key] += mass
    return weights.most_common(1)[0][0] if weights else "english"


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


def plan_song_skip(
    song_spans: list[tuple[float, float]],
    sing_spans: list[tuple[float, float]],
    segs: list[dict],
    *,
    speech_spans: list[tuple[float, float]] | None = None,
    silences: list[tuple[float, float]] | None = None,
    min_skip_sec: float,
    max_chunk_sec: float,
) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], list[dict], list[dict]
]:
    """Pure song-skip decision chain: expand -> filter -> excise -> group -> pack.

    Returns (expanded_spans, final_spans, kept_segs, chunks). No side effects, no GPU
    calls -- shared with scenario replay tests.

    Two song scales, two treatments:
    - Long singing spans (>= min_skip_sec) anchor OP/ED sequences: they absorb their whole
      voiced block (rap verses PANNs hears as Speech ride along), clean dialogue is trimmed
      from the block edges (``protect=speech_spans``), and instrumental-only spans still
      shorter than min_skip_sec after expansion are kept as content (Cecilia guard).
    - Short singing spans (< min_skip_sec — a hummed bar inside a dialogue block) must NOT
      absorb their block and must not be discarded by the length filter either: they go
      straight to excision.

    Excision replaces whole-segment dropping for everything: song intervals are cut OUT of
    the VAD segments (cut points snapped into real silences), so a segment mixing
    "speech, brief pause, humming, speech" keeps its dialogue and loses only the
    song + flanking silence.

    After expansion (and before excision), PANNs clean-dialogue spans rescue waveform-VAD
    misses: a >=3s stretch silero left uncovered but PANNs scored as Speech joins ``segs``
    (see :func:`songdet.rescue_speech_segments`), so the cold-open dialogue silero
    under-scores still reaches ASR. Rescue is deliberately AFTER expansion — rescue
    segments widen chunk coverage only and must not reshape voiced blocks or the edge trim.
    """
    long_sing = [sp for sp in sing_spans if sp[1] - sp[0] >= min_skip_sec]
    expanded = expand_spans_to_voiced_blocks(
        segs, song_spans, expandable=long_sing, protect=speech_spans
    )
    # Rescue AFTER expansion, before excision: rescue segments must only widen chunk/ASR
    # coverage — they must not reshape voiced blocks. A rescue segment can fill the one >3s
    # gap separating a dialogue block from a song block, gluing them into one block whose
    # edge trim then stops at the first non-clean segment far from the song (observed:
    # +131s over-excision). Songs are still cut out of rescued segments by excision below.
    if speech_spans:
        rescued = rescue_speech_segments(speech_spans, segs)
        if rescued:
            log.info(
                "speech rescue: %d PANNs-only segment(s) silero missed: %s",
                len(rescued),
                [(round(s["start"], 1), round(s["end"], 1)) for s in rescued],
            )
            segs = sorted(segs + rescued, key=lambda s: s["start"])
            if silences:
                # Fine-VAD calls the whole rescued region silence (silero under-scored
                # it — that is why it needed rescuing), so excision snapping could pull
                # a cut up to SNAP_SEC into rescued dialogue. Remove rescued intervals
                # from the snap targets; genuine silences elsewhere still snap.
                silences = subtract_spans(
                    list(silences), [(r["start"], r["end"]) for r in rescued]
                )
    final_long = filter_short_spans(expanded, min_sec=min_skip_sec)
    short_sing = [
        (a, b)
        for a, b in sing_spans
        if b - a < min_skip_sec
        and not any(max(a, fa) < min(b, fb) for fa, fb in final_long)
    ]
    to_cut = sorted(final_long + short_sing)
    if not to_cut:
        return expanded, [], segs, pack_speech_segments(segs, max_sec=max_chunk_sec)
    kept, final = excise_spans_from_segments(segs, to_cut, silences=silences)
    chunks: list[dict] = []
    for group in group_segments_by_spans(kept, final):
        chunks.extend(pack_speech_segments(group, max_sec=max_chunk_sec))
    return expanded, final, kept, chunks


def transcribe(
    media_path: Path,
    *,
    lang_override: str | None = None,
    separate: bool = True,
    skip_songs: bool = False,
    keep_lyrics: bool = False,
    diarize: bool = False,
    diarize_model: str | None = None,
    voiceprints: bool = False,
    voiceprint_model: str | None = None,
    normalize: bool = False,
    reporter: Reporter | None = None,
    debug: bool = False,
    debug_root: Path | None = None,
    cache_vocals: Path | None = None,
    source_fingerprint: str | None = None,
    asr_model: str | None = None,
    context: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    release_panns: bool = True,
    speaker_clustering: str | None = None,
) -> tuple[
    str,
    list[dict],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float, str]],
    VoiceprintCapture | None,
]:
    """Run separation -> song skip -> VAD chunking -> ASR -> alignment.

    Returns ``(iso_language, word_segments, vad_spans, sing_spans, speaker_turns,
    voiceprint_capture)``. The original five positions remain unchanged.
    vad_spans are the original-audio speech intervals, persisted to JSON for gap
    splitting. ``keep_lyrics`` runs song detection but skips excision: sung regions go
    through ASR/alignment like dialogue, and the detected singing spans come back so
    :func:`process` can flag lyric cues (empty unless keep_lyrics). ``diarize`` runs
    pyannote on the separated-vocals wav and returns the speaker turns (empty unless
    set). With ``voiceprints`` the per-speaker centroids come from the embedder
    ``voiceprint_model`` resolves to for the detected language (see
    :func:`voxweave.voiceembed.resolve_voiceprint_model`); only its ``pyannote``
    legacy lane reads the diarization pipeline's own embeddings; its checkpoint(s)
    are fetched and verified before any audio work, and if that fails voiceprints
    are off for the run (warning) while everything else proceeds.
    ``speaker_clustering`` picks how diarization groups its turns into speakers
    (:func:`voxweave.config.resolve_diarize_clustering`; resolved and validated
    before any audio work). All models run
    in-process (weights are fetched once into the voxweave cache). smart_split and
    file writing are handled by :func:`process`.

    ``release_panns=False`` keeps the PANNs singleton resident on return: the caller
    has a second detection pass queued (the ``--sdh`` sidecar tags the ORIGINAL mix
    with the same model) and would otherwise pay a reload. That caller owns the
    release. Every other singleton this function loads is released before it returns.
    """
    media_path = Path(media_path)
    rep = reporter or Reporter()
    if diarize:
        # Fail on a bad env/conf value now, not after separation and ASR.
        speaker_clustering = config.resolve_diarize_clustering(speaker_clustering)
    dbg: DebugSink = DebugSink()
    if debug:
        if debug_root is None:
            debug_root = artifacts.claim_paths(media_path).debug
        dbg = FileDebugSink(root=debug_root)
    tmp: list[
        Path
    ] = []  # intermediate files (fullband/vocals/16k/32k wav), deleted at end
    tmp_chunks: list[Path] = []
    # Set only on a successful return with release_panns=False: a caller can inherit
    # the loaded model, but never as the fallout of a failed run.
    panns_handoff = False
    try:
        rep.step("prepare audio")
        if voiceprints:
            # Idempotent after process()'s own prefetch: a cached checkpoint is
            # only re-verified. A failure turns voiceprints off for this run.
            voiceprints = _prefetch_voiceprint_models(
                voiceprint_model, lang_override, rep
            )
        if separate and voiceprints:
            require_sha256(source_fingerprint, "capture media fingerprint")

        def dump_separation(fullband: Path, vocals: Path) -> None:
            dbg.audio("00_fullband_44k.wav", fullband)
            dbg.audio("01_vocals.flac", vocals)
            log.info("separated vocals (local Roformer)")

        acquired = acquire_16k(
            media_path,
            separate=separate,
            normalize=normalize,
            reporter=rep,
            tmp=tmp,
            cache=cache_vocals,
            # A capture binds the cache to the snapshot it separated; otherwise
            # the duration check against the source decides.
            fingerprint=(source_fingerprint or "") if voiceprints else None,
            purpose="capture",
            on_separated=dump_separation,
        )
        wav = acquired.wav
        fullband = acquired.fullband
        voc32 = acquired.voc32  # 32k mono vocals: PANNs input + cache source
        separator_identity = acquired.separator
        dbg.audio("02_speech_16k.wav", wav)

        # Song detection must run on clean separated vocals; BGM causes speech/music confusion.
        song_spans: list[tuple[float, float]] = []
        sing_spans: list[tuple[float, float]] = []  # subset triggering block expansion
        speech_spans: list[tuple[float, float]] = []  # trimmed from song core edges
        if skip_songs or keep_lyrics:
            if not separate or voc32 is None:
                # --no-separate + skip-songs is valid (clean input); skip detection silently.
                log.debug(
                    "song detection requires separated vocals; skipping with --no-separate"
                )
            else:
                try:
                    rep.step("detect songs")
                    rep.stage("song detection (PANNs)")
                    song_spans, sing_spans, speech_spans = detect_song_spans(
                        voc32, progress=progress_bridge(rep, "song detection (PANNs)")
                    )
                    if song_spans:
                        log.info(
                            "song spans: %s",
                            [(round(a, 1), round(b, 1)) for a, b in song_spans],
                        )
                except ModuleNotFoundError as e:
                    # panns-inference (or one of its deps) is missing from a broken
                    # environment: it is a core dependency, so reinstalling fixes it.
                    log.warning(
                        "song detection unavailable (missing module: %s) -- continuing "
                        "without song skip; panns-inference is a core dependency, so "
                        "reinstall voxweave, or pass --no-skip-songs (and drop "
                        "--keep-lyrics) to silence this",
                        e,
                    )
        if release_panns:
            # Last PANNs consumer of this job is done (an --sdh run defers this to
            # its own pass). Drop the ~300MB before the ASR/aligner weights load, so
            # the two never share the card. No-op when detection never ran.
            songdet.release_model()

        rep.step("find speech")
        rep.stage("VAD chunking")
        segs = vad_speech_segments(wav)
        # PANNs speech rescue, computed once against the raw silero segs (same inputs as
        # plan_song_skip's internal rescue, so the two lists are identical on the song
        # branch). Used to (a) widen chunk coverage on the keep-lyrics / no-song branches,
        # which never reach plan_song_skip, and (b) union into the persisted vad_speech
        # below — without (b) the timing reference calls the rescued dialogue silence and
        # a later `align --vad-mask` would evict exactly the words the rescue recovered.
        rescued = rescue_speech_segments(speech_spans, segs) if speech_spans else []
        rescued_spans = [(r["start"], r["end"]) for r in rescued]
        if keep_lyrics:
            # Keep-lyrics: sung regions stay in the chunk stream and get ASR'd like
            # dialogue; only the singing spans (human vocals) are kept for cue marking.
            if sing_spans:
                log.info(
                    "keeping %d singing span(s) as lyrics: %s",
                    len(sing_spans),
                    [(round(a, 1), round(b, 1)) for a, b in sing_spans],
                )
            song_spans = []  # disable excision + VAD-reference exclusion below
            if rescued:
                log.info(
                    "speech rescue: %d PANNs-only segment(s) silero missed: %s",
                    len(rescued),
                    rescued_spans,
                )
                segs = sorted(segs + rescued, key=lambda s: s["start"])
            chunks = pack_speech_segments(segs, max_sec=MAX_CHUNK_SEC)
        elif song_spans:
            # Fine VAD (small min-silence) exposes brief intra-segment pauses; excision
            # snaps its cut points into these so dialogue words are never bisected.
            fine = vad_speech_segments(wav, min_silence_ms=SONG_FINE_SILENCE_MS)
            silences = silence_gaps(fine)
            # Decision chain lives in plan_song_skip (pure, shared with scenario tests).
            before = sum(s["end"] - s["start"] for s in segs)
            expanded, song_spans, segs, chunks = plan_song_skip(
                song_spans,
                sing_spans,
                segs,
                speech_spans=speech_spans,
                silences=silences,
                min_skip_sec=MIN_SONG_SKIP_SEC,
                max_chunk_sec=MAX_CHUNK_SEC,
            )
            log.info(
                "song spans (expanded): %s",
                [(round(a, 1), round(b, 1)) for a, b in expanded],
            )
            short = [
                (round(a, 1), round(b, 1))
                for a, b in song_spans
                if (b - a) < MIN_SONG_SKIP_SEC
            ]
            if short:
                log.info(
                    "short singing spans excised in-segment (<%.0fs): %s",
                    MIN_SONG_SKIP_SEC,
                    short,
                )
            after = sum(s["end"] - s["start"] for s in segs)
            log.info("excised %.1fs of speech-segment time as song", before - after)
        else:
            # skip_songs on but nothing detected: rescue still applies (silero misses are
            # orthogonal to song presence — a recap episode has no OP yet can lose a cold open).
            if rescued:
                log.info(
                    "speech rescue: %d PANNs-only segment(s) silero missed: %s",
                    len(rescued),
                    rescued_spans,
                )
                segs = sorted(segs + rescued, key=lambda s: s["start"])
            chunks = pack_speech_segments(segs, max_sec=MAX_CHUNK_SEC)
        if not chunks:
            raise RuntimeError(f"no speech detected in {media_path.name}")

        rep.step("transcribe and align")
        rep.stage("load ASR/alignment models")
        # Slice all chunk waveforms upfront so dual-pass (full ASR -> release -> full
        # alignment) can shave VRAM peak.
        cwavs: list[Path] = []
        for ch in chunks:
            cwav = slice_wav(wav, ch["start"], ch["end"])
            tmp_chunks.append(cwav)
            cwavs.append(cwav)
        strategy = config.conf_load_strategy()
        rep.chunks(len(chunks) * backend.chunk_pass_count(asr_model))
        # full_wav + bounds let CTC/MMS languages run ONE full-file alignment pass over
        # the whole audio (chunk windows as DP silence anchors) instead of N per-chunk
        # calls; Qwen-aligned languages (zh/yue) keep per-chunk inside transcribe_chunks.
        results = backend.transcribe_chunks(
            cwavs,
            lang_override,
            asr_model=asr_model,
            context=context,
            on_done=lambda _i: rep.chunk_done(),
            strategy=strategy,
            full_wav=wav,
            bounds=[(ch["start"], ch["end"]) for ch in chunks],
            # post-excise speech segments on the separated wav: the CTC full pass
            # soft-masks emissions outside these so words cannot park in music/silence
            speech_spans=[(s["start"], s["end"]) for s in segs],
            # final excised song intervals: muted in the full-pass waveform so mid-file
            # songs (which survive the envelope crop) cannot host smeared sentence
            # fragments. Empty under --keep-lyrics (songs stay transcribed -> unmuted).
            song_spans=song_spans or None,
        )
        # reinject_punct runs after language resolution (tokenization must match iso),
        # so punctuation cannot be reinjected per-chunk.
        chunk_pairs: list[tuple[str, list[dict]]] = []
        detected: list[str] = []  # per-chunk detected language (debug meta only)
        for idx, (ch, cwav, (det_lang, text, units)) in enumerate(
            zip(chunks, cwavs, results)
        ):
            if not text.strip():
                log.warning("empty ASR for chunk @%.1fs, skipping", ch["start"])
                dbg.chunk(
                    idx,
                    wav=cwav,
                    start=ch["start"],
                    end=ch["end"],
                    text=text,
                    lang=det_lang,
                    units=None,
                )
                continue
            if det_lang:
                detected.append(det_lang)
            dbg.chunk(
                idx,
                wav=cwav,
                start=ch["start"],
                end=ch["end"],
                text=text,
                lang=det_lang,
                units=units,
            )
            chunk_pairs.append((text, shift_units(units, ch["offset"])))

        if not chunk_pairs:
            raise RuntimeError(f"no aligned units for {media_path.name}")

        # Transcript-content weighting lets long dialogue dominate without
        # trusting the aligner whose language choice we are validating.
        lang_name = _select_transcript_language(results, lang_override)
        if not is_supported(lang_name):
            log.warning(
                "language %r is not officially supported; subtitle line breaking may"
                " be poor — pass --language to override detection",
                lang_name,
            )
        iso = to_iso_or(lang_name, "en")

        # Aligner strips punctuation; reinject_punct reattaches it by time so smart_split
        # can use it for sentence breaking and space insertion.
        all_units: list[dict] = []
        for txt, u in chunk_pairs:
            all_units.extend(realign.reinject_punct(txt, u, iso))
        if not all_units:
            raise RuntimeError(f"no aligned units for {media_path.name}")
        # Zero-duration snap: the aligner collapses short words after a pause (e.g. はい)
        # to zero duration. We snap them into the actual speech region using VAD.
        # Vocal separation attenuates secondary-speaker back-channels, so separated-vocals
        # VAD misses them. We run VAD on the ORIGINAL audio (retains attenuated speech) as
        # the timing reference, excluding song spans to avoid snapping onto singing.
        # vad_spans are persisted to .json (vad_speech) for reuse by split.
        # SNAP_VAD_THRESHOLD (0.25) catches attenuated back-channels. The original is the
        # full-band stem on a fresh separation, or the source media itself on a vocals-cache
        # hit (no stem exists then) -- both decoded to the same 16k mono, so a re-run gets
        # the same reference as the first run. --no-separate reuses the chunking VAD
        # (silero default 0.5): the decoded 16k input already IS the original audio.
        if separate:
            orig16k = decode_to_wav(fullband if fullband is not None else media_path)
            tmp.append(orig16k)
            orig_segs = vad_speech_segments(orig16k, threshold=SNAP_VAD_THRESHOLD)
            if song_spans:
                # Subtract only the truly sung/instrumental parts: clean-dialogue
                # windows inside expanded song spans (dialogue spoken OVER the
                # song) must survive in vad_speech, otherwise snapping and the
                # emission mask forbid those words' true location and the aligner
                # smears them across the song (observed on movie dialogue-over-
                # montage: a 15s exchange stretched over 65s).
                orig_segs, _ = excise_spans_from_segments(
                    orig_segs, subtract_spans(song_spans, speech_spans)
                )
            vad_spans = [(s["start"], s["end"]) for s in orig_segs]
        else:
            vad_spans = [(s["start"], s["end"]) for s in segs]
        if rescued_spans:
            # Rescued regions are PANNs-confirmed speech BOTH silero passes under-score
            # (the separated pass by premise; the original-mix pass credibly too — same
            # theatrical delivery plus BGM). Union them into the timing reference, minus
            # any excised song overlap, or downstream consumers (position_units_with_vad,
            # smart_split gaps, a later `align --vad-mask`) treat the rescued dialogue as
            # silence — the mask would evict exactly the words the rescue recovered.
            add = (
                subtract_spans(rescued_spans, sorted(song_spans))
                if song_spans
                else rescued_spans
            )
            merged = sorted(vad_spans + [(a, b) for a, b in add if b > a])
            vad_spans = []
            for a, b in merged:
                if vad_spans and a <= vad_spans[-1][1]:
                    vad_spans[-1] = (vad_spans[-1][0], max(vad_spans[-1][1], b))
                else:
                    vad_spans.append((a, b))
        # Qwen aligner has no CTC blank token, so word durations bleed into silence.
        # position_units_with_vad carves true gaps, giving smart_split an accurate signal.
        # Routed through the sink so --debug also records the pre/post snapshots and the
        # zero-duration repair accounting; the no-op sink calls the pass unchanged.
        all_units = dbg.position_units(all_units, vad_spans, language=lang_name)
        dbg.meta(
            {
                "media": str(media_path),
                "separate": separate,
                "skip_songs": skip_songs,
                "song_spans": song_spans,
                "language": iso,
                "detected": detected,
                "chunks": len(chunks),
                "units": len(all_units),
                # Wall-clock seconds per step entered so far (the open step counts up
                # to now); later steps (speaker identification, layout) are not in it.
                "timings": dict(rep.timings()),
            }
        )
        speaker_turns: list[tuple[float, float, str]] = []
        voiceprint_capture: VoiceprintCapture | None = None
        if diarize:
            from voxweave import diarize as diarize_mod
            from voxweave import voiceembed

            rep.step("identify speakers")
            voiceprint_target: (
                voiceembed.EmbedderSpec | voiceembed.LegacyLane | None
            ) = (
                voiceembed.resolve_voiceprint_model(voiceprint_model, iso)
                if voiceprints
                else None
            )
            # Voiceprint clustering embeds with ReDimNet2; when the capture below
            # uses the same model, keep it resident across the two stages. The
            # capture releases it, and so does the finally below on any path
            # that never reaches the capture.
            reuse_clustering_embedder = (
                speaker_clustering == config.DIARIZE_CLUSTERING_VOICEPRINT
                and isinstance(voiceprint_target, voiceembed.EmbedderSpec)
                and voiceprint_target.name == voiceembed.REDIMNET2_B6.name
            )
            rep.stage("speaker diarization (pyannote)")
            try:
                try:
                    diarization = diarize_mod.diarize_turns(
                        wav,
                        model=diarize_model,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        clustering=speaker_clustering,
                        release_embedder=not reuse_clustering_embedder,
                        # Only the legacy lane reads the pipeline's own embeddings;
                        # a dedicated embedder computes its voiceprints below.
                        want_embeddings=voiceprint_target is voiceembed.LEGACY,
                        audio_profile={
                            "separated": separate,
                            "normalized": normalize,
                            "sample_rate": 16000,
                            **(
                                {
                                    "separator": {
                                        **(
                                            separator_identity
                                            or {
                                                "repo": backend.SEPARATOR_REPO,
                                                "file": backend.SEPARATOR_REPO_FILE,
                                                "checkpoint": "unresolved",
                                                "config_sha256": "unresolved",
                                            }
                                        ),
                                    }
                                }
                                if separate
                                else {}
                            ),
                        },
                    )
                finally:
                    diarize_mod.release()
                speaker_turns = diarization.turns
                if isinstance(voiceprint_target, voiceembed.EmbedderSpec):
                    voiceprint_capture = _decoupled_voiceprint_capture(
                        wav,
                        diarization,
                        voiceprint_target,
                        reporter=rep,
                    )
                elif voiceprint_target is voiceembed.LEGACY and diarization.centroids:
                    voiceprint_capture = VoiceprintCapture(
                        centroids=diarization.centroids,
                        provenance=diarization.provenance,
                        turns=diarization.turns,
                    )
            finally:
                if reuse_clustering_embedder:
                    voiceembed.release()  # idempotent after the capture's own
        panns_handoff = not release_panns
        return (
            iso,
            all_units,
            vad_spans,
            sing_spans if keep_lyrics else [],
            speaker_turns,
            voiceprint_capture,
        )
    finally:
        # Each release is guarded on its own: a release that raises (e.g.
        # torch.cuda.empty_cache after a CUDA fault) must neither mask the original
        # error nor skip the remaining releases and the temp-file cleanup below.
        # Release ASR/alignment singleton VRAM (separation self-releases earlier).
        try:
            backend.release()
        except Exception as e:
            log.warning("releasing the ASR/alignment models failed: %r", e)
        # No VAD pass can follow: every vad_speech_segments call lives above.
        try:
            chunking.release_silero_vad()
        except Exception as e:
            log.warning("releasing the VAD model failed: %r", e)
        if not panns_handoff:
            # Safety net: an exception before the post-detection release (or before
            # the sdh caller can take over) would otherwise strand PANNs on the card.
            # Idempotent when the release above already ran.
            try:
                songdet.release_model()
            except Exception as e:
                log.warning("releasing the song-detection model failed: %r", e)
        for p in tmp:
            p.unlink(missing_ok=True)
        for c in tmp_chunks:
            c.unlink(missing_ok=True)


def _reconcile_word_segment_language(
    language: str | None,
    units: list[dict],
    *,
    override: str | None = None,
) -> tuple[str, list[dict]]:
    """Repair a strong persisted language/tokenization mismatch before splitting.

    Older outputs can say ``en`` even when their transcript is overwhelmingly
    Han.  Those files also carry the damage from the English aligner: a whole
    Chinese paragraph may be stored as one 10--20 second ``word``.  Merely
    changing the smart-split language cannot recover from that coarse timing.

    Reconstruct the text using the *stored* unit contract, reconcile its script,
    and only for a strong spaced -> no-space correction rebuild per-character
    timings through :func:`realign.reinject_punct`.  That helper retains
    punctuation and spaces inside Latin runs (``GPT Red``) while distributing
    each coarse unit's existing span; no ASR/alignment model is called.
    """
    if not units:
        raise RuntimeError("no word segments to split")
    # schema.Unit carries the surface under ``text`` OR ``word``; read both.
    from voxweave.core.smart_split import _unit_text

    original_units = units
    stored_iso = to_iso_or(language, "en")
    pieces = [_unit_text(unit) for unit in units]
    meaningful = [piece for piece in pieces if piece.strip()]
    if not meaningful:
        return stored_iso, units

    # A correctly reinjected no-space stream has exactly one visible character
    # per unit; spaces inside Latin phrases ride on the preceding unit (``T ``).
    # If only the persisted label is stale, joining such a stream with the
    # label's English separator would invent ``G P T``.  Recognize the
    # representation itself and preserve its exact whitespace instead.
    char_grained = len(meaningful) == len(units) and all(
        sum(not ch.isspace() for ch in piece) == 1 for piece in meaningful
    )
    if char_grained:
        text = "".join(pieces)
    else:
        stored_sep = "" if stored_iso in realign.NO_SPACE_LANGS else " "
        text = stored_sep.join(meaningful)

    effective = reconcile_detected_language(language, text, override=override)
    effective_iso = to_iso_or(effective, stored_iso)
    if effective_iso == stored_iso:
        return stored_iso, units

    stored_no_space = stored_iso in realign.NO_SPACE_LANGS
    effective_no_space = effective_iso in realign.NO_SPACE_LANGS
    if stored_no_space == effective_no_space:
        log.warning(
            "word-segment language/script mismatch: %s -> %s; "
            "reusing compatible timings",
            stored_iso,
            effective_iso,
        )
        return effective_iso, units

    # A char-grained stream already satisfies every no-space target's unit
    # contract.  Relabel it directly instead of destructively reinjecting.
    if char_grained and effective_no_space:
        log.warning(
            "word-segment language/script mismatch: %s -> %s; "
            "reusing existing character timings",
            stored_iso,
            effective_iso,
        )
        return effective_iso, units

    def _timings(seq: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]] | None:
        rows: list[tuple[float, float]] = []
        for unit in seq:
            try:
                start = float(unit["start"])
                end = float(unit["end"])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                return None
            rows.append((start, end))
        return rows or None

    source_times = _timings(units)
    if source_times is None:
        log.warning(
            "cannot repair word-segment language %s -> %s: malformed source timings",
            stored_iso,
            effective_iso,
        )
        return stored_iso, original_units

    # reinject_punct reads ``text`` only: hand it ``word``-keyed units under that key.
    text_units = [
        unit
        if unit.get("text") == _unit_text(unit)
        else {**unit, "text": _unit_text(unit)}
        for unit in units
    ]
    try:
        rebuilt = realign.reinject_punct(text, text_units, effective_iso)
    except Exception as exc:  # noqa: BLE001 -- persisted input may be arbitrarily malformed
        log.warning(
            "cannot repair word-segment language %s -> %s: %s",
            stored_iso,
            effective_iso,
            exc,
        )
        return stored_iso, original_units

    rebuilt_times = _timings(rebuilt)
    source_monotone = all(
        a[0] <= b[0] and a[1] <= b[1] for a, b in zip(source_times, source_times[1:])
    )
    rebuilt_monotone = rebuilt_times is not None and all(
        a[0] <= b[0] and a[1] <= b[1] for a, b in zip(rebuilt_times, rebuilt_times[1:])
    )
    if effective_no_space:
        round_trip_ok = "".join(_unit_text(u) for u in rebuilt) == text
    else:
        rebuilt_text = " ".join(_unit_text(u) for u in rebuilt)
        round_trip_ok = " ".join(rebuilt_text.split()) == " ".join(text.split())
    envelope_ok = rebuilt_times is not None and all(
        math.isclose(a, b, abs_tol=1e-6)
        for a, b in zip(
            (min(s for s, _e in source_times), max(e for _s, e in source_times)),
            (
                min(s for s, _e in rebuilt_times),
                max(e for _s, e in rebuilt_times),
            ),
        )
    )
    if (
        not rebuilt
        or not round_trip_ok
        or not envelope_ok
        or (source_monotone and not rebuilt_monotone)
    ):
        log.warning(
            "cannot repair word-segment language %s -> %s without changing "
            "content/timing; keeping persisted representation",
            stored_iso,
            effective_iso,
        )
        return stored_iso, original_units

    noun = "character" if effective_no_space else "word"
    log.warning(
        "word-segment language/script mismatch: %s -> %s; rebuilding %s timings",
        stored_iso,
        effective_iso,
        noun,
    )
    return effective_iso, rebuilt


def _voiceprint_capture_from_generation(
    generation: episode_transaction.FileGeneration,
) -> str | None:
    if generation.bytes_value is None:
        return None
    try:
        value = json.loads(generation.bytes_value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(value, dict) and isinstance(value.get("voiceprint_capture"), str):
        return value["voiceprint_capture"]
    return None


def _warn_stale_speaker_names(
    media_path: Path,
    previous_json: episode_transaction.FileGeneration,
    speaker_turns: Sequence[tuple[float, float, str]] | None,
) -> None:
    """Warn when saved speaker names were given to turns this run replaced.

    The speaker mapping keys names by ``SPEAKER_NN`` id alone. A re-run whose
    turns differ (another ``--speaker-clustering`` or ``--diarize-model``, a
    voiceprint stage that fell back to pyannote, new audio) can hand an id to
    another voice, and its saved name follows the id into rendered subtitles,
    the speakers page and ``speakers enroll`` (which would store that voice
    under the name in the voice library). Names whose id kept exactly the
    same turns are not reported. Never raises: this is advice, not a gate.
    """
    if not speaker_turns or previous_json.bytes_value is None:
        return
    try:
        previous = json.loads(previous_json.bytes_value.decode("utf-8"))
        raw_old = previous.get("speaker_turns") if isinstance(previous, dict) else None
        mapping = inspect_speakers_mapping_path(media_path)
        if not raw_old or not artifacts.path_present(mapping):
            return
        named = named_speaker_ids(mapping)
        old = [(float(s), float(e), str(label)) for s, e, label in raw_old]
    except (OSError, RuntimeError, UnicodeError, ValueError, TypeError):
        return

    def spans_by_id(turns: Sequence[tuple[float, float, str]]) -> dict[str, list]:
        spans: dict[str, list] = {}
        for start, end, label in turns:
            spans.setdefault(label, []).append((float(start), float(end)))
        return {label: sorted(values) for label, values in spans.items()}

    before, after = spans_by_id(old), spans_by_id(speaker_turns)
    stale = sorted(i for i in named if before.get(i) != after.get(i))
    if stale:
        log.warning(
            "%s names %s, but this run's speaker turns for %s differ from those "
            "the names were given to (changing --speaker-clustering or "
            "--diarize-model renumbers speakers), so a name may now label another "
            "voice; review the names with `voxweave speakers %s` before "
            "`voxweave speakers enroll`",
            mapping.name,
            ", ".join(stale),
            "that id" if len(stale) == 1 else "those ids",
            media_path.name,
        )


def _voiceprints_document(
    media_path: Path,
    capture: VoiceprintCapture,
    *,
    capture_id: str,
    source_fingerprint: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "capture_id": capture_id,
        "provenance": capture.provenance,
        "binding": {
            "turns_digest": canonical_turns_digest(capture.turns),
            "media_fingerprint": source_fingerprint,
            "media_stem": media_path.stem,
            "created": utc_timestamp(),
        },
        "speakers": capture.centroids,
    }
    validate_voiceprints_mapping(value)
    return value


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
        return _process_from_source(
            media_path,
            source_path=source_path,
            snapshot_fingerprint=snapshot_fingerprint,
            expected_json=expected_json,
            expected_vtt=expected_vtt,
            expected_media_fingerprint=expected_media,
            source_mode=source_mode,
            lang_override=lang_override,
            separate=separate,
            reporter=rep,
            debug=debug,
            debug_root=debug_root,
            normalize=normalize,
            skip_songs=skip_songs,
            keep_lyrics=keep_lyrics,
            sdh=sdh,
            diarize=diarize,
            diarize_model=diarize_model,
            voiceprints=capture_enabled,
            voiceprint_model=voiceprint_model,
            speaker_clustering=speaker_clustering,
            word_segments=word_segments,
            asr_model=asr_model,
            context=context,
            timestamps=timestamps,
            shot_snap=shot_snap,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
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
                return run_with_source(
                    snapshot.path,
                    snapshot.fingerprint,
                    True,
                )
    return run_with_source(media_path, None, False)


def _process_from_source(
    media_path: Path,
    *,
    source_path: Path,
    snapshot_fingerprint: str | None,
    expected_json: episode_transaction.FileGeneration,
    expected_vtt: episode_transaction.FileGeneration,
    expected_media_fingerprint: str | None,
    source_mode: episode_transaction.ProcessSourceMode,
    lang_override: str | None = None,
    separate: bool = True,
    reporter: Reporter | None = None,
    debug: bool = False,
    debug_root: Path | None = None,
    normalize: bool = False,
    skip_songs: bool = False,
    keep_lyrics: bool = False,
    sdh: bool = False,
    diarize: bool = False,
    diarize_model: str | None = None,
    voiceprints: bool = False,
    voiceprint_model: str | None = None,
    speaker_clustering: str | None = None,
    word_segments: tuple[str, list[dict]] | None = None,
    asr_model: str | None = None,
    context: str | None = None,
    timestamps: bool = True,
    shot_snap: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
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
                shot_job = shotdet.ShotDetectionJob().start(source_path)
            (
                iso,
                units,
                vad_speech,
                sing_spans,
                speaker_turns,
                _voiceprint_capture,
            ) = transcribe(
                source_path,
                lang_override=lang_override,
                separate=separate,
                skip_songs=skip_songs,
                keep_lyrics=keep_lyrics,
                diarize=diarize,
                diarize_model=diarize_model,
                voiceprints=voiceprints,
                voiceprint_model=voiceprint_model,
                speaker_clustering=speaker_clustering,
                normalize=normalize,
                reporter=reporter,
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
            snapshot_fingerprint=snapshot_fingerprint,
            expected_json=expected_json,
            expected_vtt=expected_vtt,
            expected_media_fingerprint=expected_media_fingerprint,
            source_mode=source_mode,
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


def _finish_process_from_units(
    media_path: Path,
    *,
    rep: Reporter,
    iso: str,
    units: list[dict],
    vad_speech: list[tuple[float, float]] | None,
    shot_changes: list[float] | None,
    sing_spans: list[tuple[float, float]] | None,
    speaker_turns: list[tuple[float, float, str]] | None,
    timestamps: bool,
    capture: VoiceprintCapture | None,
    snapshot_fingerprint: str | None,
    expected_json: episode_transaction.FileGeneration,
    expected_vtt: episode_transaction.FileGeneration,
    expected_media_fingerprint: str | None,
    source_mode: episode_transaction.ProcessSourceMode,
    sdh_enabled: bool,
) -> _ProcessPublication:
    """Finish segmentation and publication after source ownership is sealed."""
    rep.step("layout subtitles")
    rep.stage("smart_split layout")
    segmented = segment_document(
        language=iso,
        word_segments=units,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
    )
    units, cues = segmented.units, segmented.cues

    rep.step("write outputs")
    rep.stage("write siblings")
    from voxweave import segmentation_orchestration

    if segmented.document is None or segmented.manifest is None:
        raise RuntimeError("segmentation result lacks its production authority")
    if capture is not None and capture.turns is not speaker_turns:
        raise RuntimeError("voiceprint capture turns diverged from sibling turns")

    voiceprint_pair: tuple[str, str] | None = None
    machine_artifact: episode_transaction.MachineArtifactPublication | None = None
    if capture is not None and snapshot_fingerprint is not None:
        capture_id = mint_capture_id(
            current=_voiceprint_capture_from_generation(expected_json)
        )
        try:
            sidecar = _voiceprints_document(
                media_path,
                capture,
                capture_id=capture_id,
                source_fingerprint=snapshot_fingerprint,
            )
            sidecar_bytes = encode_json_bytes(sidecar, max_bytes=VOICEPRINTS_MAX_BYTES)
        except Phase2DataError as exc:
            log.warning("voiceprint capture dropped: %s", exc)
        else:
            voiceprint_pair = (capture_id, snapshot_fingerprint)
            machine_artifact = episode_transaction.MachineArtifactPublication(
                voiceprints_path(media_path), sidecar_bytes
            )

    selection = segmentation_orchestration.build_segmentation_selection(
        command="process",
        target_path=swap_ext(media_path, ".vtt"),
        sibling_path=swap_ext(media_path, ".json"),
        language=iso,
        cues=cues,
        top_level_units=units,
        document=segmented.document,
        manifest=segmented.manifest,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=segmentation_orchestration.semantic_speaker_turns_carrier(
            speaker_turns
        ),
        voiceprint_pair=voiceprint_pair,
        timestamps=timestamps,
        speaker_names=(),
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        source_mode=source_mode,
        mapping_generation=None,
        mapping_path=None,
        shadow_enabled=os.environ.get(SEG_V2_SHADOW_ENV, "").strip() == "1",
    )
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if machine_artifact is None:
        cleanup.extend(
            episode_transaction.ArtifactCleanup(path, "voiceprints-unlink")
            for path in voiceprints_candidates(media_path)
        )
    cleanup.extend(
        episode_transaction.ArtifactCleanup(path, "suggest-unlink")
        for path in speaker_suggest_candidates(media_path)
    )
    cleanup.extend(
        (
            episode_transaction.ArtifactCleanup(
                speakers_html_path(media_path), "html-unlink"
            ),
        )
    )
    cleanup.extend(
        episode_transaction.ArtifactCleanup(path, "evidence-unlink")
        for path in align_evidence_candidates(
            swap_ext(media_path, ".vtt"), media=media_path
        )
    )
    try:
        receipt = episode_transaction.commit_primary_outputs(
            command="process",
            episode_path=media_path,
            json_path=swap_ext(media_path, ".json"),
            vtt_path=swap_ext(media_path, ".vtt"),
            expected_json=expected_json,
            expected_vtt=expected_vtt,
            main_json_bytes=selection.verified.main_json_bytes,
            vtt_bytes=selection.verified.vtt_bytes,
            cleanup_paths=tuple(cleanup),
            context=selection.context,
            media_path=(media_path if expected_media_fingerprint is not None else None),
            expected_media_fingerprint=expected_media_fingerprint,
            machine_artifact=machine_artifact,
        )
    finally:
        segmentation_orchestration.retire_segmentation_selection(selection)
    vtt_out = swap_ext(media_path, ".vtt")
    landed = receipt.landed
    selected_sdh_cues: Sequence[Mapping[str, Any]] = selection.sdh_dialogue
    if machine_artifact is not None:
        log.info("wrote voice-biometric sidecar %s", machine_artifact.path.name)
    log.info("wrote %s + .json (%d cues, lang=%s)", vtt_out.name, len(cues), iso)
    _warn_stale_speaker_names(media_path, expected_json, speaker_turns)
    auxiliary_landed: tuple[Path, ...] = ()
    if sdh_enabled and source_mode == "transcribed-media":
        rep.step("create SDH sidecar")
        committed_json = episode_transaction.capture_file_generation(
            swap_ext(media_path, ".json")
        )
        committed_vtt = episode_transaction.capture_file_generation(vtt_out)
        try:
            sidecar = _write_sdh_sidecar(
                media_path,
                selected_sdh_cues,
                rep,
                expected_json_generation=committed_json,
                expected_vtt_generation=committed_vtt,
                expected_media_fingerprint=expected_media_fingerprint,
            )
            if sidecar is not None:
                auxiliary_landed = (sidecar,)
        except Exception as exc:
            log.warning("SDH sidecar failed (non-fatal): %r", exc)
    return _ProcessPublication(vtt_out, landed, auxiliary_landed)


def _write_sdh_sidecar(
    media_path: Path,
    cues: Sequence[Mapping[str, Any]],
    rep: Reporter,
    *,
    expected_json_generation: episode_transaction.FileGeneration | None = None,
    expected_vtt_generation: episode_transaction.FileGeneration | None = None,
    expected_media_fingerprint: str | None = None,
) -> Path | None:
    """Detect non-speech events on the ORIGINAL mix (effects are stripped from the
    separated-vocals stem) and write ``<stem>.sdh.vtt`` (dialogue + event tags).
    Returns None when panns-inference is missing (warned, non-fatal)."""
    rep.stage("SDH event detection (PANNs)")
    wav32 = decode_to_wav(media_path, sample_rate=SONGDET_SR)
    try:
        events = sdh_mod.detect_events(
            wav32, progress=progress_bridge(rep, "SDH event detection (PANNs)")
        )
    except ModuleNotFoundError as e:
        log.warning(
            "SDH detection requires panns-inference (not installed: %s); skipping sidecar",
            e,
        )
        return None
    finally:
        wav32.unlink(missing_ok=True)
    events = sdh_mod.fit_events_to_gaps(events, cues)
    path = swap_ext(media_path, ".sdh.vtt")
    json_path = swap_ext(media_path, ".json")
    vtt_path = swap_ext(media_path, ".vtt")
    expected_json = (
        expected_json_generation
        or episode_transaction.capture_file_generation(json_path)
    )
    expected_vtt = (
        expected_vtt_generation or episode_transaction.capture_file_generation(vtt_path)
    )
    expected_media = expected_media_fingerprint or media_fingerprint(media_path)
    landed = episode_transaction.commit_auxiliary_sdh(
        episode_path=media_path,
        sidecar_path=path,
        sidecar_bytes=sdh_mod.render_sdh_vtt(cues, events).encode("utf-8"),
        json_path=json_path,
        expected_json=expected_json,
        vtt_path=vtt_path,
        expected_vtt=expected_vtt,
        media_path=media_path,
        expected_media_fingerprint=expected_media,
    )
    if not landed:
        log.warning("stale SDH sidecar discarded; primaries or media changed")
        return None
    log.info("wrote %s (%d event tag(s))", path.name, len(events))
    return path


def split(
    json_path: Path,
    timestamps: bool = True,
    *,
    reporter: Reporter | None = None,
    **smart_split_kwargs,
) -> Path:
    """Re-run smart_split from persisted word_segments.

    Reuses ``vad_speech`` from the sibling JSON for gap splitting; falls back to gap-only
    mode if absent. ``timestamps`` behaves as in :func:`process`.
    """
    # Accept the .vtt sibling too: `voxweave split foo.vtt` should not feed
    # WEBVTT bytes to json.loads.
    json_path = swap_ext(Path(json_path), ".json")
    rep = reporter or Reporter()
    rep.plan(("read subtitle data", "layout subtitles", "write outputs"))
    rep.step("read subtitle data")
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
    rep.step("layout subtitles")
    segmented = segment_document(
        language=iso,
        word_segments=units,
        vad_speech=speech_spans,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        smart_split_kwargs=smart_split_kwargs,
        annotate_speakers=bool(speaker_names),
    )
    units, cues = segmented.units, segmented.cues
    from voxweave import segmentation_orchestration
    from voxweave.align_snapshot import decode_sibling_json_snapshot

    if segmented.document is None or segmented.manifest is None:
        raise RuntimeError("segmentation result lacks its production authority")
    expected_json = episode_transaction.FileGeneration(True, input_bytes)
    lexical_snapshot = decode_sibling_json_snapshot(json_path.name, input_bytes)
    selection = segmentation_orchestration.build_segmentation_selection(
        command="split",
        target_path=swap_ext(json_path, ".vtt"),
        sibling_path=json_path,
        language=iso,
        cues=cues,
        top_level_units=units,
        document=segmented.document,
        manifest=segmented.manifest,
        vad_speech=speech_spans,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=lexical_snapshot.carrier("speaker_turns"),
        voiceprint_pair=voiceprint_pair,
        timestamps=timestamps,
        speaker_names=mapping_generation.names,
        expected_json=expected_json,
        expected_vtt=None,
        source_mode=None,
        mapping_generation=mapping_generation,
        mapping_path=mapping_path,
        shadow_enabled=os.environ.get(SEG_V2_SHADOW_ENV, "").strip() == "1",
    )
    pair_declared = "voiceprint_capture" in data or "voiceprint_media" in data
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if pair_declared and voiceprint_pair is None:
        cleanup.extend(
            episode_transaction.ArtifactCleanup(path, "voiceprints-unlink")
            for path in voiceprints_candidates(owner)
        )
        cleanup.extend(
            episode_transaction.ArtifactCleanup(path, "suggest-unlink")
            for path in speaker_suggest_candidates(owner)
        )
        cleanup.append(
            episode_transaction.ArtifactCleanup(
                speakers_html_path(json_path), "html-unlink"
            )
        )
    cleanup.extend(
        episode_transaction.ArtifactCleanup(path, "evidence-unlink")
        for path in align_evidence_candidates(
            swap_ext(json_path, ".vtt"),
            media=owner,
        )
    )
    rep.step("write outputs")
    try:
        episode_transaction.commit_primary_outputs(
            command="split",
            episode_path=owner,
            json_path=json_path,
            vtt_path=swap_ext(json_path, ".vtt"),
            expected_json=expected_json,
            expected_vtt=None,
            main_json_bytes=selection.verified.main_json_bytes,
            vtt_bytes=selection.verified.vtt_bytes,
            cleanup_paths=tuple(cleanup),
            context=selection.context,
            speaker_mapping_path=mapping_path,
            expected_speaker_mapping=mapping_generation,
        )
    finally:
        segmentation_orchestration.retire_segmentation_selection(selection)
    vtt_out = swap_ext(json_path, ".vtt")
    log.info("rendered %s → %d cues", vtt_out.name, len(cues))
    return vtt_out


def _prepare_16k_for_align(
    media: Path,
    *,
    separate: bool,
    normalize: bool,
    reporter: Reporter,
    tmp: list[Path],
    cache_media: Path | None = None,
    source_fingerprint: str | None = None,
) -> Path:
    """Prepare 16k vocals for align; append temp paths to tmp. Return the 16k wav path.

    ``media`` is the byte authority used for a cache miss. ``cache_media`` owns
    the persistent cache namespace and defaults to the same path. When
    ``source_fingerprint`` is present, only a fully validated v1 companion can
    establish a hit; legacy duration-only caches remain unbound-only behavior.
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

    lang_name = lang_override or data.get("language") or "english"
    iso = to_iso_or(lang_name, "en")

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
        full_pass = mms or bool(ctc_model)
        route_kind = "mms-full" if mms else "ctc-full" if ctc_model else "qwen-crop"
    delivery_order = (
        tuple(range(len(input_snapshot.blocks)))
        if full_pass
        else input_snapshot.qwen_delivery_order
    )
    bounds_by_source = {
        bound.source_index: bound for bound in input_snapshot.route_bounds
    }
    blocks: list[dict[str, Any]] = []
    for source_index in delivery_order:
        content = input_snapshot.blocks[source_index]
        bound = bounds_by_source[source_index]
        blocks.append(
            {
                "text": content.text,
                "alignment_text": content.alignment_text,
                "start": bound.start,
                "end": bound.end,
                "lyric": content.lyric,
                "speaker": content.speaker,
                "speakers": (
                    None if content.speakers is None else list(content.speakers)
                ),
                "source_index": content.source_index,
            }
        )
    crops: list[
        tuple[float, float] | None
    ] = []  # set + looped only on the per-cue (zh·yue) path
    if not full_pass:
        has_ts = all(b["start"] is not None and b["end"] is not None for b in blocks)
        if not has_ts and not word_segments:
            untimed = sum(1 for b in blocks if b["start"] is None or b["end"] is None)
            if untimed == len(blocks):
                message = (
                    f"{json_path.name} has no word_segments and VTT has no timestamps; "
                    f"cannot route audio windows"
                )
            else:
                message = (
                    f"{json_path.name} has no word_segments and {untimed} of "
                    f"{len(blocks)} VTT cues have no timestamps; cannot route audio "
                    f"windows (give every cue a timing line, or restore the sibling "
                    f"JSON's word_segments)"
                )
            exc = RuntimeError(message)
            _attach_canonical_failure(
                exc,
                kind="qwen-route-invalid",
                phase="route-plan",
                detail_code="no-route-source",
            )
            raise exc
        with align_runtime_activity("AO-03", "qwen-route-plan"):
            try:
                spans = realign.route_blocks(blocks, word_segments)
                crops = realign.crop_blocks(spans)
            except (IndexError, KeyError) as exc:
                _attach_canonical_failure(
                    exc,
                    kind="qwen-window-operation-failed",
                    phase="route-plan",
                    detail_code="route-bound-access",
                )
                raise
            except (ArithmeticError, TypeError, ValueError) as exc:
                _attach_canonical_failure(
                    exc,
                    kind="qwen-window-operation-failed",
                    phase="route-plan",
                    detail_code="route-bound-arithmetic",
                )
                raise
        if all(c is None for c in crops):
            exc = RuntimeError(
                "routing failed: no alignable blocks (text completely mismatches word_segments?)"
            )
            _attach_canonical_failure(
                exc,
                kind="qwen-route-invalid",
                phase="route-plan",
                detail_code="all-crops-none",
            )
            raise exc

    tmp: list[Path] = []
    tmp_chunks: list[Path] = []
    snapshots = ExitStack()
    selected_snapshot = None
    align_context = None
    completed_selection = None
    observation_input: Mapping[str, Any] | None = None
    prepared_audio_sha256: str | None = None
    aligned_cue_count = 0
    aligned_unit_count = 0
    acquisition_media = media
    production_failure: BaseException | None = None
    try:
        if voiceprint_pair is not None:
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
        try:
            rep.step("prepare audio")
            with align_runtime_activity("AO-04", "prepared-audio-and-cache"):
                wav = _prepare_16k_for_align(
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

            try:
                import soundfile as sf

                prepared_info = sf.info(str(wav))
                prepared_sample_rate = int(prepared_info.samplerate)
                prepared_sample_count = int(prepared_info.frames)
            except Exception:  # noqa: BLE001 - unavailable geometry is sealed as invalid
                prepared_sample_rate = 16_000
                prepared_sample_count = 0

            model_facts = {
                "route": route_kind,
                "language": iso,
                "backend": (
                    "mms"
                    if mms
                    else "ctc"
                    if ctc_model
                    else "mlx-qwen"
                    if backend._use_mlx()
                    else "qwen-asr"
                ),
                "model": (
                    "mms" if mms else ctc_model if ctc_model else backend.ALIGNER_MODEL
                ),
                "sample_rate": prepared_sample_rate,
            }
            route_facts = {
                "route": route_kind,
                "language": iso,
                "blocks": [
                    {
                        "source_index": block["source_index"],
                        "alignment_text": block["alignment_text"],
                        "start": (
                            None
                            if block["start"] is None
                            else float(block["start"]).hex()
                        ),
                        "end": (
                            None if block["end"] is None else float(block["end"]).hex()
                        ),
                    }
                    for block in blocks
                ],
                "crops": [
                    None
                    if crop is None
                    else [float(crop[0]).hex(), float(crop[1]).hex()]
                    for crop in crops
                ],
            }

            def stable_fact_digest(value: object) -> str:
                encoded = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                return hashlib.sha256(encoded).hexdigest()

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
                    backend_model_config_digest=stable_fact_digest(model_facts),
                    route_input_digest=stable_fact_digest(route_facts),
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
                shadow_requested=os.environ.get(SEG_V2_SHADOW_ENV, "").strip() == "1",
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
                    episode_transaction.ArtifactCleanup(path, "voiceprints-unlink")
                    for path in voiceprints_candidates(media)
                )
                cleanup.extend(
                    episode_transaction.ArtifactCleanup(path, "suggest-unlink")
                    for path in speaker_suggest_candidates(media)
                )
                cleanup.append(
                    episode_transaction.ArtifactCleanup(
                        speakers_html_path(vtt_path), "html-unlink"
                    )
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
                episode_transaction.ArtifactCleanup(path, "evidence-unlink")
                for path in align_evidence_candidates(vtt_path, media=media)
                if path != evidence_path
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
                os.environ.get(SEG_V2_SHADOW_ENV, "").strip() == "1"
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
