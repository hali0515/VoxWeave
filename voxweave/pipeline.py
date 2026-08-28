from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import subprocess
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from voxweave import asrfix as asrfix_mod
from voxweave import backend, chunking, episode_transaction, fsio, realign, songdet
from voxweave import translate as translate_mod
from voxweave.chunking import (
    decode_to_wav,
    pack_speech_segments,
    silence_gaps,
    slice_wav,
    vad_speech_segments,
)
from voxweave.align_failures import CanonicalFailure, SecondaryFailure
from voxweave.core.providers import degradation_capture, provider_snapshot
from voxweave.core.schema import Cue, Unit
from voxweave.core.segdoc import (
    THRESHOLD_KEYS,
    DisplayProfile,
    SegDocument,
    build_seg_document,
)
from voxweave.debug import DebugSink, FileDebugSink
from voxweave.lang import (
    is_supported,
    reconcile_detected_language,
    to_iso_or,
    transcript_content_weight,
)
from voxweave.mediasnapshot import MediaSnapshot, SnapshotUnavailable
from voxweave.progress import Reporter
from voxweave.songdet import (
    detect_song_spans,
    excise_spans_from_segments,
    expand_spans_to_voiced_blocks,
    filter_short_spans,
    group_segments_by_spans,
    rescue_speech_segments,
    subtract_spans,
)
from voxweave.speakers import voice_text_for_ids
from voxweave.timestamps import shift_units
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
from voxweave.vocalscache import (
    cache_lock,
    cache_write_window,
    load_cache_companion,
    publish_cache_companion,
    validate_cache_pair,
)

if TYPE_CHECKING:  # the v2 shadow is import-free unless its flag is on
    from voxweave.core.boundary_v2 import DocumentSolution
    from voxweave.core.partition_check import Origin, Stage

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
MAX_CHUNK_SEC = float(os.environ.get("VOXWEAVE_MAX_CHUNK_SEC", "120"))
# Spans shorter than this after expansion are kept as dialogue, not skipped.
# Real OP/ED runs 30-90s; short instrumental BGM scattered through speech would hurt ASR
# if dropped (env VOXWEAVE_MIN_SONG_SKIP_SEC).
MIN_SONG_SKIP_SEC = float(os.environ.get("VOXWEAVE_MIN_SONG_SKIP_SEC", "8"))
# Loudness normalization applied only to the 16k VAD/ASR path; 44.1k separation path is untouched.
ASR_LOUDNORM = os.environ.get("VOXWEAVE_LOUDNORM", "loudnorm=I=-16:TP=-1.5:LRA=11")
# PANNs Cnn14 is trained at 32k.
SONGDET_SR = 32000
# Sensitive VAD threshold for snapping zero-duration units to original (pre-separation)
# audio. Silero default 0.5 misses back-channels (はい/ええ) attenuated by vocal separation;
# 0.25 catches them. Used only for snap positioning, not for chunk boundary decisions.
SNAP_VAD_THRESHOLD = float(os.environ.get("VOXWEAVE_SNAP_VAD_THRESHOLD", "0.25"))
# Fine VAD pass for song excision: a small min-silence (vs the 300ms chunking default)
# surfaces brief intra-segment pauses, so excision cut points land in real silence and
# never bisect a dialogue word. Only runs when song spans were detected.
SONG_FINE_SILENCE_MS = int(os.environ.get("VOXWEAVE_SONG_FINE_SILENCE_MS", "100"))
# Align-stage cue duration floor. Default 0 (disabled): enforce_min_duration only
# resolves overlaps without padding, so short back-channels keep their real ~0.6s.
# Set VOXWEAVE_MIN_CUE_SEC=0.8 to re-enable padding. Distinct from VOXWEAVE_SEG_MIN_CUE_SEC.
MIN_CUE_SEC = float(os.environ.get("VOXWEAVE_MIN_CUE_SEC", "0"))
# Flash-cue rescue (orthogonal to MIN_CUE_SEC): genuine flash cues (so/あ at 0.1-0.2s)
# are extended to TINY_CUE_TARGET, allowed to overlap only the immediately following cue.
# VOXWEAVE_TINY_CUE_SEC=0 disables.
TINY_CUE_SEC = float(os.environ.get("VOXWEAVE_TINY_CUE_SEC", "0.2"))
TINY_CUE_TARGET = float(os.environ.get("VOXWEAVE_TINY_CUE_TARGET", "0.5"))


# Vocals cache: <media_dir>/cache/<stem>.vocals.32k.flac (32k mono, no BGM).
# Shared by process and align; PANNs eats it directly, ASR/alignment downsample to 16k.
# Legacy <stem>.16k.flac caches are still accepted by align for backward compatibility.
# A cache hit also requires the durations to match (_vocals_cache_fresh): a replaced or
# trimmed source silently invalidates the old separation, so it is re-run and overwritten.
CACHE_DIRNAME = "cache"
# Max |cache - media| duration drift still treated as the same source. Covers
# container-vs-stream duration jitter; real source edits move duration by seconds.
CACHE_DUR_TOL_SEC = float(os.environ.get("VOXWEAVE_CACHE_DUR_TOL_SEC", "0.5"))
# Extensions tried when locating the source media by stem (align only receives the VTT).
MEDIA_EXTS = (
    ".mkv",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".ts",
    ".m4v",
    ".flac",
    ".wav",
    ".m4a",
    ".mp3",
    ".aac",
    ".opus",
    ".ogg",
)


@dataclass(frozen=True)
class VoiceprintCapture:
    """Embedding evidence kept alive from diarization through episode commit."""

    centroids: dict[str, list[float]]
    provenance: dict[str, object]
    turns: list[tuple[float, float, str]]


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


def _attach_vtt_decode_failure(exc: BaseException) -> None:
    cause = exc.__cause__
    if isinstance(cause, UnicodeDecodeError):
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


def _attach_semantic_configuration_failure(exc: BaseException) -> None:
    if type(exc).__name__ == "SemanticBackendUnavailable":
        _attach_canonical_failure(
            exc,
            kind="semantic-backend-unavailable",
            phase="semantic-config",
            detail_code="endpoint-not-configured",
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


def _release_semantic_engine(engine: Any | None) -> None:
    """Best-effort optional-model cleanup; never replace deterministic output."""

    if engine is None:
        return
    try:
        engine.release()
    except Exception as exc:  # noqa: BLE001 - cleanup must not overturn baseline cues
        log.warning("semantic splitter cleanup failed: %s", exc)


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


def _progress_bridge(rep: Reporter, label: str):
    """Convert the ``(done, total)`` callback from backend/songdet into a Reporter task bar.

    Keeps backend/songdet free of any rich dependency.
    """
    started = {"v": False}

    def cb(done: int, total: int) -> None:
        if not started["v"]:
            rep.task(label, total)
            started["v"] = True
        rep.advance(1)

    return cb


def cache_vocals_path(media_path: Path) -> Path:
    """Return the canonical vocals cache path: <media_dir>/cache/<stem>.vocals.32k.flac."""
    media_path = Path(media_path)
    return media_path.parent / CACHE_DIRNAME / f"{media_path.stem}.vocals.32k.flac"


def cache_16k_path(media_path: Path) -> Path:
    """Return the legacy 16k vocals cache path: <media_dir>/cache/<stem>.16k.flac (read-only backward compat)."""
    media_path = Path(media_path)
    return media_path.parent / CACHE_DIRNAME / f"{media_path.stem}.16k.flac"


def _probe_duration(path: Path) -> float | None:
    """Media duration in seconds via ffprobe, or None if unreadable."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _vocals_cache_fresh(cache: Path, media: Path) -> bool:
    """True if the cached vocals still match the source media duration.

    An unreadable cache is stale (truncated/corrupt flac must not be trusted);
    an unprobeable media keeps the cache hit — decoding will surface the real
    error later, and burning a separation pass on a maybe-valid cache helps nobody.
    """
    cache_dur = _probe_duration(cache)
    if cache_dur is None:
        log.warning("vocals cache unreadable, re-separating: %s", cache)
        return False
    media_dur = _probe_duration(media)
    if media_dur is None:
        return True
    if abs(cache_dur - media_dur) <= CACHE_DUR_TOL_SEC:
        return True
    log.warning(
        "vocals cache stale (cache %.2fs vs media %.2fs), re-separating: %s",
        cache_dur,
        media_dur,
        cache,
    )
    return False


def _encode_flac(src_wav: Path, dst_flac: Path) -> None:
    """Encode wav to flac for caching (lossless); caller treats failure as non-fatal.

    Atomic: an interrupted encode must not leave a truncated flac at the cache
    path — every later run would treat it as a cache hit and fail obscurely.
    """
    dst_flac.parent.mkdir(parents=True, exist_ok=True)
    with fsio.atomic_path(dst_flac) as tmp:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(src_wav), "-c:a", "flac", str(tmp)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def swap_ext(path: Path, new_ext: str) -> Path:
    """Replace the trailing extension of path with new_ext (include leading dot; "" removes it).

    Do NOT use ``Path.with_suffix`` for sibling paths: filenames with interior dots
    (e.g. YouTube titles containing ``...``) cause with_suffix to misidentify the first
    interior dot as the suffix, silently truncating the name. This function only replaces
    ``path.suffix``, leaving interior dots untouched.
    """
    if path.suffix:
        return path.with_name(path.name[: -len(path.suffix)] + new_ext)
    return path.with_name(path.name + new_ext)


def require_vtt(path: Path) -> Path:
    """Reject non-VTT inputs for align/correct; return the path unchanged.

    Both commands write VTT back (align overwrites the input in place, correct's
    ``--apply`` does too), so running them on ``.srt``/``.ass`` would corrupt the
    file with VTT content. translate/pack/burn/export accept the other formats
    via :func:`voxweave.subformats.require_subtitle`.
    """
    from voxweave.subformats import require_subtitle

    return require_subtitle(path, exts=(".vtt",))


def _find_sibling_media(ref: Path) -> Path | None:
    """Find the source media alongside ref by stem, matching extensions case-insensitively.

    Returns the best candidate by ``MEDIA_EXTS`` order (``.mkv`` before ``.mp4`` ...);
    when more than one media sibling exists the ambiguity is logged and the selection
    stays deterministic. ``None`` when no sibling is found.
    """
    ref = Path(ref)
    base = swap_ext(ref, "").name  # stem without the reference extension (dot-safe)
    order = {ext: i for i, ext in enumerate(MEDIA_EXTS)}
    matches: list[tuple[int, Path]] = []
    parent = ref.parent if str(ref.parent) else Path(".")
    if parent.exists():
        for p in parent.iterdir():
            if p.is_file() and swap_ext(p, "").name == base:
                rank = order.get(p.suffix.lower())
                if rank is not None:
                    matches.append((rank, p))
    if not matches:
        return None
    matches.sort(key=lambda m: (m[0], str(m[1])))
    if len(matches) > 1:
        log.warning(
            "multiple sibling media files for %s: %s; using %s",
            ref.name,
            [m[1].name for m in matches],
            matches[0][1].name,
        )
    return matches[0][1]


@overload
def _separate_to_16k_32k(
    media: Path,
    *,
    reporter: Reporter,
    normalize: bool,
    return_separator_identity: Literal[False] = False,
) -> tuple[Path, Path, Path, Path]: ...


@overload
def _separate_to_16k_32k(
    media: Path,
    *,
    reporter: Reporter,
    normalize: bool,
    return_separator_identity: Literal[True],
) -> tuple[Path, Path, Path, Path, dict[str, object]]: ...


def _separate_to_16k_32k(
    media: Path,
    *,
    reporter: Reporter,
    normalize: bool,
    return_separator_identity: bool = False,
) -> tuple[Path, Path, Path, Path] | tuple[Path, Path, Path, Path, dict[str, object]]:
    """Decode full-band 44.1k stereo -> Roformer separate -> resample, returning
    ``(fullband, vocals, wav_16k, voc32_32k)`` and, when requested, the
    load-bound separator identity as a fifth item.

    The full-band 44.1k stereo feed is a hard constraint (Roformer is trained at 44.1k);
    downsampling to 16k/32k happens only after separation. Callers own temp bookkeeping,
    debug dumps, and caching of the returned paths.

    On a clean return the caller registers the paths in its own ``tmp`` list (cleaned in its
    ``finally``). Since that registration only runs after this returns, the helper self-cleans
    its partial outputs if a later step raises — otherwise an OOM/ffmpeg failure mid-separation
    would orphan the already-decoded temp files.
    """
    af = ASR_LOUDNORM if normalize else None
    created: list[Path] = []
    try:
        reporter.stage("decode fullband 44.1k")
        fullband = decode_to_wav(media, sample_rate=44100, mono=False)
        created.append(fullband)
        reporter.stage("vocal separation (Roformer)")
        if return_separator_identity:
            vocals, separator_identity = backend.separate_vocals(
                fullband,
                progress=_progress_bridge(reporter, "vocal separation (Roformer)"),
                return_identity=True,
            )
        else:
            vocals = backend.separate_vocals(
                fullband,
                progress=_progress_bridge(reporter, "vocal separation (Roformer)"),
            )
        created.append(vocals)
        reporter.stage("resample 16k")
        wav = decode_to_wav(vocals, audio_filter=af)
        created.append(wav)
        voc32 = decode_to_wav(
            vocals, sample_rate=SONGDET_SR
        )  # 32k mono: PANNs + cache source
        if return_separator_identity:
            return fullband, vocals, wav, voc32, separator_identity
        return fullband, vocals, wav, voc32
    except Exception:
        for p in created:
            p.unlink(missing_ok=True)
        raise


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
            " re-run transcribe/process to regenerate it"
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(
            f"{json_path.name}: expected a JSON object, got {type(data).__name__};"
            " re-run transcribe/process to regenerate it"
        )
    if require is not None and require not in data:
        raise RuntimeError(
            f"{json_path.name} has no {require!r} key;"
            " re-run transcribe/process to regenerate it"
        )
    return data


def _load_sibling_json(json_path: Path, *, require: str | None = None) -> dict:
    """Load a sibling ``.json`` with readable failures: a corrupt file or a
    missing required key names the file and points at regeneration instead of
    surfacing a bare JSONDecodeError/KeyError deep in the stack."""
    json_path = Path(json_path)
    return _load_sibling_json_bytes(
        json_path,
        json_path.read_bytes(),
        require=require,
    )


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


def resolve_segmentation_manifest(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """The segmentation manifest of a loaded sibling document.

    Every sibling written before P3 carries no ``segmentation`` key at all, and
    that absence is itself the label: such a document was produced by the legacy
    v1 engine, so it resolves to ``{"engine": "legacy-v1", "inferred": True}``
    rather than to nothing. ``inferred`` distinguishes the deduction from a
    recorded manifest, which is returned exactly as stored (a non-mapping value
    is not one and falls back to the inference).

    The engine name here is a literal on purpose and must NOT become
    :data:`SEGMENTATION_ENGINE`: that constant tracks whichever engine this build
    runs, while a manifest-less file was written by v1 no matter what this build
    does now.
    """
    found = data.get("segmentation")
    if isinstance(found, Mapping):
        return found
    return {"engine": "legacy-v1", "inferred": True}


def _load_cues(vtt_path: Path) -> list[dict]:
    """Parse subtitle cue blocks by extension (VTT/SRT/ASS/SSA); raise if the
    file has no cues. Shared guard for align/translate/correct."""
    from voxweave.subformats import load_subtitle_blocks

    return load_subtitle_blocks(Path(vtt_path))


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
    voiceprints: bool = False,
    normalize: bool = False,
    reporter: Reporter | None = None,
    debug: bool = False,
    cache_vocals: Path | None = None,
    source_fingerprint: str | None = None,
    debug_stem: str | None = None,
    asr_model: str | None = None,
    context: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    release_panns: bool = True,
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
    set). All models run in-process (no network calls). smart_split and file writing
    are handled by :func:`process`.

    ``release_panns=False`` keeps the PANNs singleton resident on return: the caller
    has a second detection pass queued (the ``--sdh`` sidecar tags the ORIGINAL mix
    with the same model) and would otherwise pay a reload. That caller owns the
    release. Every other singleton this function loads is released before it returns.
    """
    media_path = Path(media_path)
    rep = reporter or Reporter()
    dbg: DebugSink = (
        FileDebugSink(debug_stem or media_path.stem) if debug else DebugSink()
    )
    af = ASR_LOUDNORM if normalize else None
    tmp: list[
        Path
    ] = []  # intermediate files (fullband/vocals/16k/32k wav), deleted at end
    tmp_chunks: list[Path] = []
    # Set only on a successful return with release_panns=False: a caller can inherit
    # the loaded model, but never as the fallout of a failed run.
    panns_handoff = False
    try:
        vocals: Path | None = None
        fullband: Path | None = None
        voc32: Path | None = None  # 32k mono vocals: PANNs input + cache source
        separator_identity: dict[str, object] | None = None
        if separate and voiceprints:
            require_sha256(source_fingerprint, "capture media fingerprint")
        if separate:
            cache_hit = False
            if cache_vocals is not None:
                with cache_lock(Path(cache_vocals)) as cache_handle:
                    cache_path = cache_handle.cache_path
                    if cache_path.exists():
                        if voiceprints:
                            try:
                                companion, _validated = load_cache_companion(
                                    cache_handle.companion_path
                                )
                                current_separator = backend.separator_identity()
                                validated = validate_cache_pair(
                                    companion,
                                    cache_path,
                                    media_fingerprint=source_fingerprint or "",
                                    separator=current_separator,
                                )
                                separator_identity = validated.separator.as_mapping()
                                cache_hit = True
                            except (OSError, Phase2DataError):
                                log.info(
                                    "vocals cache is not bound to this capture; re-separating: %s",
                                    cache_path,
                                )
                        else:
                            cache_hit = _vocals_cache_fresh(cache_path, media_path)
                    if cache_hit:
                        # Keep the cache lock through decoder completion. A validated
                        # hash followed by an unlocked open is not a stable read.
                        rep.stage("vocals cache (32k)")
                        log.info("reuse cached vocals %s", cache_path)
                        voc32 = cache_path
                        wav = decode_to_wav(
                            voc32, audio_filter=af
                        )  # 32k flac -> 16k mono
            if cache_hit:
                # Cache hit: skip Roformer; PANNs eats 32k directly, ASR downsamples to 16k.
                pass
            else:
                if voiceprints:
                    (
                        fullband,
                        vocals,
                        wav,
                        voc32,
                        separator_identity,
                    ) = _separate_to_16k_32k(
                        media_path,
                        reporter=rep,
                        normalize=normalize,
                        return_separator_identity=True,
                    )
                else:
                    fullband, vocals, wav, voc32 = _separate_to_16k_32k(
                        media_path, reporter=rep, normalize=normalize
                    )
                tmp.append(fullband)
                dbg.audio("00_fullband_44k.wav", fullband)
                tmp.append(vocals)
                dbg.audio("01_vocals.flac", vocals)
                tmp.append(voc32)
                log.info("separated vocals (local Roformer)")
                if cache_vocals is not None:
                    try:
                        with cache_write_window(Path(cache_vocals)) as cache_handle:
                            _encode_flac(voc32, cache_handle.cache_path)
                            if voiceprints:
                                publish_cache_companion(
                                    cache_handle.cache_path,
                                    media_fingerprint=source_fingerprint or "",
                                    separator=separator_identity or {},
                                    companion_path=cache_handle.companion_path,
                                )
                        log.info("cached vocals 32k → %s", cache_vocals)
                    except (
                        OSError,
                        subprocess.CalledProcessError,
                        Phase2DataError,
                    ) as e:
                        log.warning("cache vocals failed (non-fatal): %r", e)
        else:
            rep.stage("decode 16k")
            wav = decode_to_wav(media_path, audio_filter=af)
        tmp.append(wav)
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
                    rep.stage("song detection (PANNs)")
                    song_spans, sing_spans, speech_spans = detect_song_spans(
                        voc32, progress=_progress_bridge(rep, "song detection (PANNs)")
                    )
                    if song_spans:
                        log.info(
                            "song spans: %s",
                            [(round(a, 1), round(b, 1)) for a, b in song_spans],
                        )
                except ModuleNotFoundError as e:
                    # panns-inference not installed; continue without song skip.
                    # Install voxweave[songdet] or pass --no-skip-songs to suppress.
                    log.warning(
                        "song detection requires panns-inference (not installed: %s) -- "
                        "continuing without song skip; install voxweave[songdet] or pass --no-skip-songs",
                        e,
                    )
        if release_panns:
            # Last PANNs consumer of this job is done (an --sdh run defers this to
            # its own pass). Drop the ~300MB before the ASR/aligner weights load, so
            # the two never share the card. No-op when detection never ran.
            songdet.release_model()

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

        rep.stage("load ASR/alignment models")
        # Slice all chunk waveforms upfront so dual-pass (full ASR -> release -> full
        # alignment) can shave VRAM peak.
        cwavs: list[Path] = []
        for ch in chunks:
            cwav = slice_wav(wav, ch["start"], ch["end"])
            tmp_chunks.append(cwav)
            cwavs.append(cwav)
        from voxweave.config import conf_load_strategy

        strategy = conf_load_strategy()
        rep.chunks(len(chunks) * backend.chunk_pass_count(asr_model, strategy))
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
                    raw=text,
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
                raw=text,
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
                "language %r not in aligner set; smart_split may misbehave", lang_name
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
        # SNAP_VAD_THRESHOLD (0.25) catches attenuated back-channels; --no-separate uses
        # silero default (0.5) since the original audio is not available separately.
        if separate and fullband is not None:
            orig16k = decode_to_wav(fullband)
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
            }
        )
        speaker_turns: list[tuple[float, float, str]] = []
        voiceprint_capture: VoiceprintCapture | None = None
        if diarize:
            from voxweave import diarize as diarize_mod

            rep.stage("speaker diarization (pyannote)")
            try:
                diarization = diarize_mod.diarize_turns(
                    wav,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    want_embeddings=voiceprints,
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
            if voiceprints and diarization.centroids:
                voiceprint_capture = VoiceprintCapture(
                    centroids=diarization.centroids,
                    provenance=diarization.provenance,
                    turns=diarization.turns,
                )
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
        # Release ASR/alignment singleton VRAM (separation self-releases earlier).
        backend.release()
        # No VAD pass can follow: every vad_speech_segments call lives above.
        chunking.release_silero_vad()
        if not panns_handoff:
            # Safety net: an exception before the post-detection release (or before
            # the sdh caller can take over) would otherwise strand PANNs on the card.
            # Idempotent when the release above already ran.
            songdet.release_model()
        for p in tmp:
            p.unlink(missing_ok=True)
        for c in tmp_chunks:
            c.unlink(missing_ok=True)


def _spans_in(raw: Any) -> list[tuple[float, float]] | None:
    """Parse a persisted ``vad_speech`` array (``[[start, end], ...]``) to float tuples.

    Malformed entries (wrong arity, non-numeric bounds) are skipped with a warning
    rather than crashing the whole re-split. None if absent/empty or nothing survives.
    """
    if not raw:
        return None
    out: list[tuple[float, float]] = []
    for entry in raw:
        try:
            s, e = entry
            out.append((float(s), float(e)))
        except (TypeError, ValueError):
            log.warning("skipping malformed vad_speech entry: %r", entry)
    return out or None


def _turns_in(raw: Any) -> list[tuple[float, float, str]] | None:
    """Parse persisted ``speaker_turns`` (``[[start, end, label], ...]``).

    This is the byte-preserving production replay seam: numeric bounds are only
    coerced to ``float``.  Reversed, point and non-finite legacy values survive
    exactly as they did before P5; the shadow's speaker-evidence consumer owns
    normalization on its detached document copy after the feature flag.

    Malformed entries (wrong arity or non-numeric bounds) are skipped with a
    warning. None if absent/empty or nothing survives.
    """
    if not raw:
        return None
    out: list[tuple[float, float, str]] = []
    for entry in raw:
        try:
            s, e, lb = entry
            out.append((float(s), float(e), str(lb)))
        except (TypeError, ValueError):
            log.warning("skipping malformed speaker_turns entry: %r", entry)
    return out or None


def _maybe_adaptive_thresholds(th: dict, units: list[dict]) -> dict:
    """Scale clause/offline gap thresholds to this file's gap distribution.

    EXPERIMENTAL, default off: opt in via VOXWEAVE_GAP_ADAPTIVE=1. Replaces the
    static clause_ms (and offline_ms at the same clause:offline ratio) with a
    per-file estimate from the inter-unit gap distribution; vad_skip_ms is
    untouched. Validate against scripts/calib_segmentation.py before trusting.
    """
    if os.environ.get("VOXWEAVE_GAP_ADAPTIVE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return th
    from voxweave.core.gap_split import adaptive_clause_ms

    gaps_ms: list[float] = []
    for prev, nxt in zip(units, units[1:]):
        pe, ns = prev.get("end"), nxt.get("start")
        if pe is not None and ns is not None:
            gaps_ms.append((float(ns) - float(pe)) * 1000.0)
    clause = adaptive_clause_ms(gaps_ms)
    if clause is None:
        return th
    ratio = th["offline_ms"] / th["clause_ms"] if th.get("clause_ms") else 1.75
    out = dict(th)
    out["clause_ms"] = clause
    out["offline_ms"] = round(clause * ratio)
    log.info(
        "adaptive gap thresholds: clause %dms offline %dms (static %s/%s)",
        out["clause_ms"],
        out["offline_ms"],
        th.get("clause_ms"),
        th.get("offline_ms"),
    )
    return out


def _resnap_shots(
    cues: list[Cue], shot_changes: list[float] | None, thresholds: dict
) -> list[Cue]:
    """Re-apply shot snapping after speaker formatting rewrote the cue stream.

    smart_split snaps boundaries to shot changes as its last timing step, but
    speaker formatting splits cues at speaker turns and runs another timing
    cleanup, which moves those boundaries again -- so a formatted cue can end up
    flashing across a cut that the first snap had cleared. Snapping once more
    with the same cuts and the same duration cap restores the invariant;
    ``_snap_to_shots`` leaves boundaries that already sit in a landing zone
    untouched, so the extra pass is a no-op when formatting changed nothing.
    """
    if not shot_changes:
        return cues
    from voxweave.core.smart_split import SplitThresholds
    from voxweave.core.timing import _snap_to_shots

    th = SplitThresholds.from_mapping(thresholds)
    return _snap_to_shots(
        cues, sorted(shot_changes), snap_s=th.shot_snap_s, max_cue_s=th.max_cue_s
    )


# A cue is a lyric when at least this fraction of its span overlaps detected singing.
LYRIC_MIN_OVERLAP = 0.5


def mark_lyric_cues(
    cues: Sequence[Cue], sing_spans: list[tuple[float, float]] | None
) -> None:
    """Flag cues whose span mostly overlaps detected singing (``lyric=True``).

    The stored cue text stays clean; display layers (`_write_siblings` VTT rows,
    SRT/ASS export) wrap flagged cues with music notes per the Netflix lyric
    convention. Runs in place after smart_split so flags ride the final cues.
    """
    if not sing_spans:
        return
    for c in cues:
        start, end = c.get("start"), c.get("end")
        if start is None or end is None or end - start <= 0:
            continue
        overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in sing_spans)
        if overlap / (end - start) >= LYRIC_MIN_OVERLAP:
            c["lyric"] = True


def lyric_display_text(cue: Mapping[str, Any]) -> str:
    """Cue display text: lyric cues get the Netflix music-note wrap (note + space
    at the start and end of the subtitle), others pass through unchanged."""
    text = str(cue["text"])
    return f"♪ {text} ♪" if cue.get("lyric") else text


def _sibling_json_data(
    *,
    language: str,
    segments: Sequence[Mapping[str, Any]],
    units: list[dict],
    vad_speech: list[tuple[float, float]] | None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
    voiceprint_capture: str | None = None,
    voiceprint_media: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical sibling JSON document without touching the filesystem."""
    data: dict[str, Any] = {
        "language": language,
        "segments": segments,
        "word_segments": units,
    }
    if vad_speech is not None:
        data["vad_speech"] = [[float(s), float(e)] for s, e in vad_speech]
    if shot_changes is not None:
        data["shot_changes"] = [float(t) for t in shot_changes]
    if sing_spans is not None:
        data["sing_spans"] = [[float(s), float(e)] for s, e in sing_spans]
    if speaker_turns is not None:
        data["speaker_turns"] = [
            [float(s), float(e), str(lb)] for s, e, lb in speaker_turns
        ]
    if (voiceprint_capture is None) != (voiceprint_media is None):
        raise Phase2DataError("voiceprint sibling keys must be present as a pair")
    if voiceprint_capture is not None and voiceprint_media is not None:
        data["voiceprint_capture"] = require_capture_id(voiceprint_capture)
        data["voiceprint_media"] = require_sha256(voiceprint_media, "voiceprint_media")
    if manifest is not None:
        data["segmentation"] = dict(manifest)
    return data


def _encode_sibling_json_bytes(
    *,
    language: str,
    segments: Sequence[Mapping[str, Any]],
    units: list[dict],
    vad_speech: list[tuple[float, float]] | None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
    voiceprint_capture: str | None = None,
    voiceprint_media: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> bytes:
    """Encode the final sibling JSON bytes for a staged publication."""
    data = _sibling_json_data(
        language=language,
        segments=segments,
        units=units,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        voiceprint_capture=voiceprint_capture,
        voiceprint_media=voiceprint_media,
        manifest=manifest,
    )
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _dump_sibling_json(
    json_path: Path,
    *,
    language: str,
    segments: Sequence[Mapping[str, Any]],
    units: list[dict],
    vad_speech: list[tuple[float, float]] | None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
    voiceprint_capture: str | None = None,
    voiceprint_media: str | None = None,
    final_voiceprint_check: Callable[[], bool] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Write the sibling JSON document (language + segments + word_segments + optional
    vad_speech / shot_changes / sing_spans / speaker_turns / segmentation).

    ``vad_speech=None`` omits the key; a list (even empty) writes it coerced to
    ``[[float, float], ...]``. ``shot_changes`` behaves the same (written only when not
    None, so ``split`` can replay shot snapping without re-decoding the video), as do
    ``sing_spans`` (lyric re-flagging without PANNs) and ``speaker_turns`` (speaker
    re-formatting without pyannote). Single source of truth for the sibling-JSON shape
    shared by process and align.

    ``segments[].word_data`` entries carry their atom surface under ``text``
    alongside the span (``smart_split._chunk_to_cue``) — a reader has no other way
    to tell that stream's granularity. Replay reads ``word_segments``, not
    ``segments``, so older files stay loadable.

    ``manifest`` (the ``SegmentationManifest``) is written last, after
    ``speaker_turns``, so the top-level key order of every pre-P3 document is
    untouched and byte-diff tooling can strip exactly one trailing key. Absent
    means the file predates the manifest — legacy-v1 by definition, see
    :func:`resolve_segmentation_manifest`.
    """
    data = _sibling_json_data(
        language=language,
        segments=segments,
        units=units,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        voiceprint_capture=voiceprint_capture,
        voiceprint_media=voiceprint_media,
        manifest=manifest,
    )
    text = json.dumps(data, ensure_ascii=False, indent=2)
    fallback_selector: Callable[[], str | None] | None = None
    if final_voiceprint_check is not None:
        if voiceprint_capture is None or voiceprint_media is None:
            raise Phase2DataError("a final voiceprint check requires a complete pair")
        unbound = dict(data)
        del unbound["voiceprint_capture"]
        del unbound["voiceprint_media"]
        unbound_text = json.dumps(unbound, ensure_ascii=False, indent=2)

        def select_unbound() -> str | None:
            return None if final_voiceprint_check() else unbound_text

        fallback_selector = select_unbound

    fsio.atomic_write_text(
        json_path,
        text,
        before_replace=fallback_selector,
    )


# Cue keys that exist only in memory: raw acoustic anchors captured at cue
# construction plus speaker ids used while rendering a named VTT. Nothing in
# legacy-v1 reads them and the sibling JSON predates them, so the writer drops
# all three.
_UNPERSISTED_CUE_KEYS = ("speech_start", "speech_end", "speaker_ids")


def _persistable_cue(cue: Mapping[str, Any]) -> dict[str, Any]:
    """A cue as the sibling JSON stores it: everything except the raw anchors.

    A drop-list rather than a whitelist, so every other key a cue carries today
    (``lyric``, any ``word_data`` shape) and anything a later pass adds still
    ships unchanged -- the persisted bytes only move when a key is dropped here.
    """
    return {k: v for k, v in cue.items() if k not in _UNPERSISTED_CUE_KEYS}


def _write_siblings(
    src: Path,
    cues: Sequence[Mapping[str, Any]],
    units: list[dict],
    lang: str,
    vad_speech: list[tuple[float, float]] | None = None,
    timestamps: bool = True,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
    voiceprint_capture: str | None = None,
    voiceprint_media: str | None = None,
    final_voiceprint_check: Callable[[], bool] | None = None,
    manifest: Mapping[str, Any] | None = None,
    speaker_names: Mapping[str, str] | None = None,
) -> Path:
    """Write sibling .json (ground truth) and .vtt alongside src; return the .vtt path.

    ``timestamps=True`` writes a timing line before each cue (word-level precision); cues
    missing start/end fall back to plain text. ``timestamps=False`` writes a plain-text
    edit draft for human editing before re-running ``align``. Both formats are accepted by
    ``realign.parse_vtt_blocks``. Lyric-flagged cues render with the music-note wrap in
    the VTT only; the JSON keeps clean text + the flag. Uses ``swap_ext`` (not
    ``with_suffix``) to preserve interior dots in filenames.

    The persisted cues are projected through :func:`_persistable_cue`, which drops
    the in-memory-only acoustic anchors and speaker ids; ``manifest`` is
    forwarded to the JSON writer, which appends it as the last top-level key.
    """
    _dump_sibling_json(
        swap_ext(src, ".json"),
        language=lang,
        segments=[_persistable_cue(c) for c in cues],
        units=units,
        vad_speech=vad_speech or [],
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        voiceprint_capture=voiceprint_capture,
        voiceprint_media=voiceprint_media,
        final_voiceprint_check=final_voiceprint_check,
        manifest=manifest,
    )
    rows = []
    for c in cues:
        text = lyric_display_text(c)
        if speaker_names:
            text = voice_text_for_ids(text, c.get("speaker_ids"), speaker_names)
        rows.append(
            (
                c.get("start") if timestamps else None,
                c.get("end") if timestamps else None,
                text,
            )
        )
    vtt_path = swap_ext(src, ".vtt")
    fsio.atomic_write_text(vtt_path, realign.render_cues(rows))
    return vtt_path


def _units_to_seg(units: list[dict], iso: str) -> dict:
    """Flatten word_segments into a single segment dict for smart_split.

    Units already carry punctuation from reinject_punct. No-space languages join without
    separator; smart_split uses punctuation for sentence breaking and converts it to spaces.
    Surfaces are read through the tolerant accessor: units legally carry their text under
    ``text`` or ``word`` (see ``schema.Unit``), and replayed sibling JSONs use either.
    """
    from voxweave.core.smart_split import _unit_text

    sep = "" if iso in realign.NO_SPACE_LANGS else " "
    words = [
        {"word": _unit_text(u), "start": u["start"], "end": u["end"]} for u in units
    ]
    return {
        "start": units[0]["start"],
        "end": units[-1]["end"],
        "text": sep.join(_unit_text(u) for u in units),
        "words": words,
    }


#: Shape version of the sibling JSON's ``segmentation`` block. Bump when a field
#: changes meaning; a reader that does not know the version must not guess.
SEGMENTATION_MANIFEST_VERSION = 1
#: The segmentation engine this build runs. P3 records the pre-strangler one so a
#: later document can be told apart from every file written before the manifest.
SEGMENTATION_ENGINE = "legacy-v1"


def _voxweave_version() -> str:
    """Installed voxweave version, or ``"unknown"`` running from a source tree."""
    try:
        return importlib.metadata.version("voxweave")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class SegmentationResult:
    """Output of :func:`segment_document`: the cue stream plus what produced it.

    ``units`` is the (copied) unit stream after punctuation snapping -- the same
    stream the callers persist as ``word_segments``, so a replay writes back what
    it actually split. ``thresholds_used`` is the effective gap/duration mapping
    after the optional adaptive pass, and ``diagnostics`` records which optional
    passes ran (all values deterministic, so two identical inputs compare equal).

    ``manifest`` is the ``SegmentationManifest`` the callers persist as the
    sibling JSON's ``segmentation`` key, and ``document`` is the
    :class:`~voxweave.core.segdoc.SegDocument` holding that same manifest object
    -- minted before the engine runs, so it describes the inputs rather than
    summarizing the output. Both are additive and default to ``None`` so
    existing constructors keep working; in legacy-v1 the engine consumes
    neither.

    ``shadow`` is the BoundaryOptimizer v2 measurement artifact, present only
    when :data:`SEG_V2_SHADOW_ENV` is on. It is *returned* rather than written
    because ``segment_document`` is pinned pure; persisting it is a caller's or
    the harness's job. Nothing in the shipped output depends on it -- a run with
    the flag on and a run with it off produce byte-identical siblings.
    """

    cues: list[Cue]
    language: str
    units: list[dict]
    thresholds_used: dict
    diagnostics: dict[str, Any]
    manifest: dict[str, Any] | None = None
    document: SegDocument | None = None
    shadow: dict[str, Any] | None = None


def _copied_spans(
    spans: Sequence[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """Copy a span sequence to plain float tuples; empty/absent -> ``None``.

    Mirrors :func:`_spans_in`: "no spans recorded" and "empty array" are the same
    thing for every consumer of persisted spans.
    """
    return [(float(s), float(e)) for s, e in spans] if spans else None


def _copied_turns(
    turns: Sequence[tuple[float, float, str]] | None,
) -> list[tuple[float, float, str]] | None:
    """Copy speaker turns to plain tuples; empty/absent -> ``None`` (see :func:`_turns_in`)."""
    return (
        [(float(s), float(e), str(label)) for s, e, label in turns] if turns else None
    )


# --------------------------------------------------------- v2 shadow lane

#: Opt-in for the BoundaryOptimizer v2 shadow measurement. Parsed with the
#: exact-``"1"`` test the manifest already uses for the VAD mask: a paired
#: ``--no-`` flag writes the literal ``"0"``, which is truthy as a *string*, so
#: ``bool()`` would latch the shadow on for the run that turned it off.
#:
#: Deliberately environment-only, and deliberately absent from the manifest. The
#: manifest is the record of what produced the *shipped* output and a shadow run
#: ships v1, so mentioning it there would move persisted bytes for a lane that
#: changes nothing. A ``[defaults]`` conf key is worse still: inner keys are
#: never validated, so a typo is silent and a latched value could leave a user
#: running a measurement build for months.
SEG_V2_SHADOW_ENV = "VOXWEAVE_SEG_V2_SHADOW"

#: P5's lane names. Core and the renamed legacy delivery proxy retain the P4
#: evidence; the finalizer row matrix is gated, and the legacy-display isolation
#: comparator supplies N3a/N11 without speaker overlay or resnapping.
SHADOW_LANE_CORE = "core_partition_pre_overlay"
SHADOW_LANE_DELIVERY_LEGACY = "delivery_v1_legacy"
SHADOW_LANE_FINALIZER = "delivery_finalizer"
SHADOW_LANE_LEGACY_DISPLAY = "legacy_display"
# Compatibility name for downstream readers that imported the P4 constant.
SHADOW_LANE_DELIVERY = SHADOW_LANE_DELIVERY_LEGACY


def _shadow_surface_partition(
    units: Sequence[Any], cues: Sequence[Cue]
) -> tuple[tuple[int, ...] | None, str]:
    """Project cue boundaries by stored surfaces, never by a character cursor."""
    from voxweave.core.smart_split import _surface_ranges

    if not cues:
        return (), "empty"
    word_data = [entry for cue in cues for entry in cue.get("word_data") or ()]
    ranges = _surface_ranges([unit.surface for unit in units], word_data)
    if ranges is None or len(ranges) != len(units):
        return None, "surface-reconciliation-failed"
    boundaries = {
        ranges[index - 1][1]: index
        for index in range(1, len(ranges))
        if ranges[index - 1][1] == ranges[index][0]
    }
    cursor = 0
    cuts: list[int] = []
    for cue in cues[:-1]:
        cursor += len(cue.get("word_data") or ())
        cut = boundaries.get(cursor)
        if cut is None:
            return None, f"surface-boundary-unresolved-at-entry-{cursor}"
        cuts.append(cut)
    cursor += len(cues[-1].get("word_data") or ())
    if cursor != len(word_data):
        return None, f"surface-stream-ends-at-entry-{cursor}-of-{len(word_data)}"
    if any(left >= right for left, right in zip(cuts, cuts[1:])):
        return None, "surface-cuts-non-monotone"
    return tuple(cuts), "surface-footprint"


def _shadow_v1_partition(
    parent: SegDocument,
    origin: Sequence[int],
    cues: Sequence[Cue],
) -> tuple[tuple[int, ...] | None, str]:
    """Resolve v1 structurally, translating parent coordinates through origin."""
    import bisect

    parent_cuts, parent_mode = _shadow_surface_partition(parent.units, cues)
    if parent_cuts is not None:
        translated = tuple(bisect.bisect_left(origin, cut) for cut in parent_cuts)
        return translated, f"{parent_mode}-parent-through-origin"
    return None, f"parent:{parent_mode};origin-translation-unavailable"


def _shadow_cue_rows(
    cues: Sequence[Cue], partition: Sequence[int] | None, unit_count: int
) -> list[dict[str, Any]]:
    """The artifact projection of one cue stream: display facts plus ownership."""
    from voxweave.core.partition_check import owned_unit_ids

    bounds = (
        owned_unit_ids(partition, unit_count)
        if partition is not None and len(partition) + 1 == len(cues)
        else None
    )
    rows: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        text = str(cue["text"])
        rows.append(
            {
                "end": cue.get("end"),
                "index": index,
                "lines": len(text.split("\n")),
                "lyric": bool(cue.get("lyric", False)),
                "speaker_ids": list(cue.get("speaker_ids") or ()),
                "speech_end": cue.get("speech_end"),
                "speech_start": cue.get("speech_start"),
                "start": cue.get("start"),
                "text": text,
                "unit_range": None if bounds is None else list(bounds[index]),
            }
        )
    return rows


def _restamp_by_footprint(
    waivers: Mapping[int, Any], partition: Sequence[int] | None, unit_count: int
) -> dict[int, Any]:
    """Re-point an interval-minted waiver ledger at another stage's cue indices.

    A waiver's cue index is only meaningful against the stream it was minted for,
    and the later stages re-time, re-wrap and (in the overlay lane) split and
    merge cues -- so index identity is not a contract. Source-unit ownership is:
    the exemption names the units it covers, and every stage still owns those
    units in exactly one cue. Handing the checker no ledger at all -- the shape
    this replaced -- made it re-report an exemption the solver had granted as an
    *unwaived*, exit-driving violation, which is the failure mode
    ``_document_waivers`` exists to prevent, reproduced one level up.
    """
    from dataclasses import replace

    from voxweave.core.partition_check import owned_unit_ids

    if partition is None or not waivers:
        return {}
    bounds = owned_unit_ids(partition, unit_count)
    out: dict[int, Any] = {}
    for waiver in waivers.values():
        if not waiver.unit_ids:
            continue
        low, high = min(waiver.unit_ids), max(waiver.unit_ids) + 1
        for index, (start, end) in enumerate(bounds):
            if start <= low and high <= end:
                out[index] = replace(waiver, cue_index=index)
                break
    return out


def _origins_by_footprint(
    fallback_ranges: Sequence[Sequence[int]],
    partition: Sequence[int] | None,
    unit_count: int,
) -> dict[int, Origin]:
    """Which engine produced each cue of a stage's stream, by unit ownership.

    AD3-3 attributes a violation to the engine that produced the violating cue.
    A document with one adopted-v1 interval is not a v1 document, so typing the
    whole stage "v2" blames v2 for v1's damage and typing it "v1" excuses v2's.
    """
    from voxweave.core.partition_check import owned_unit_ids

    if partition is None or not fallback_ranges:
        return {}
    bounds = owned_unit_ids(partition, unit_count)
    return {
        index: "v1"
        for index, (start, end) in enumerate(bounds)
        if any(start < high and end > low for low, high in fallback_ranges)
    }


def _shadow_stream_block(
    cues: Sequence[Cue],
    partition: Sequence[int] | None,
    projection: str,
    *,
    document: SegDocument,
    origin: Origin,
    stage: Stage,
    waivers: Mapping[int, Any] | None = None,
    origins: Mapping[int, Origin] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One engine's stream at one lane: rows, its partition, and its validator."""
    from voxweave.core.partition_check import check_partition

    validator = (
        None
        if partition is None
        else check_partition(
            partition,
            cues,
            units=document.units,
            profile=document.profile,
            origin=origin,
            stage=stage,
            waivers=waivers,
            origins=origins,
        ).to_dict()
    )
    block: dict[str, Any] = {
        "cue_count": len(cues),
        "cues": _shadow_cue_rows(cues, partition, len(document.units)),
        "partition": None if partition is None else list(partition),
        "projection": projection,
        "validator": validator,
    }
    block.update(extra or {})
    return block


def _shadow_lane_block(
    lane: str, stage: Stage, v1: Mapping[str, Any], v2: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair the two engines at one lane and state where they disagree."""
    agreement: dict[str, Any] | None = None
    if v1["partition"] is not None and v2["partition"] is not None:
        left = set(v1["partition"])
        right = set(v2["partition"])
        agreement = {
            "identical_cuts": len(left & right),
            "v1_cut_count": len(left),
            "v1_only": sorted(left - right),
            "v2_cut_count": len(right),
            "v2_only": sorted(right - left),
        }
    return {
        "agreement": agreement,
        "lane": lane,
        "stage": stage,
        "v1": dict(v1),
        "v2": dict(v2),
    }


def _shadow_core_cues(
    solution: DocumentSolution, document: SegDocument, thresholds: Mapping[str, Any]
) -> list[Cue]:
    """v2's raw materialization, finished the way v1 finishes its own stream.

    Only the passes that are *not* boundary decisions are replayed: the timing
    cleanup, the shot snap and the text finalization. The merge/glue/repair
    passes are boundary decisions and are precisely what v2 replaces, so
    replaying them here would grade v2 on v1's repairs.
    """
    from voxweave.core.layout import (
        _line_budget_width,
        _merge_stutters,
        strip_punct_for_subtitles,
        wrap_cue_text,
    )
    from voxweave.core.smart_split import SplitThresholds
    from voxweave.core.timing import _cleanup_cues, _snap_to_shots

    profile = document.profile
    lang = profile.language
    th = SplitThresholds.from_mapping(dict(thresholds))
    cues: list[Cue] = [
        copy.deepcopy(cue) for item in solution.solutions for cue in item.cues
    ]
    cues = _cleanup_cues(
        cues,
        min_cue_s=th.min_cue_s,
        max_cue_s=th.max_cue_s,
        cps=th.cps,
        lag_out_s=th.lag_out_s,
    )
    if document.shot_changes:
        cues = _snap_to_shots(
            cues,
            sorted(document.shot_changes),
            snap_s=th.shot_snap_s,
            max_cue_s=th.max_cue_s,
        )
    width = _line_budget_width(profile.max_line_length, lang)
    for cue in cues:
        cue["text"] = wrap_cue_text(
            _merge_stutters(strip_punct_for_subtitles(cue["text"])),
            lang,
            profile.max_lines,
            max_line_length=width,
        )
    return cues


def _shadow_overlay_cues(
    cues: Sequence[Cue], document: SegDocument, thresholds: Mapping[str, Any]
) -> list[Cue]:
    """The legacy overlays applied to a copy of a stream, for the delivery lane.

    Every input the overlays touch is copied first: production runs these same
    overlays on the real cue stream immediately after the hook returns, so a
    formatter that mutated its ``turns`` list here would change shipped bytes.
    """
    out: list[Cue] = [copy.deepcopy(cue) for cue in cues]
    mark_lyric_cues(out, _copied_spans(document.sing_spans))
    turns = _copied_turns(document.speaker_turns)
    if turns:
        from voxweave.diarize import apply_speaker_format

        out = apply_speaker_format(
            out,
            turns,
            document.language,
            thresholds=dict(thresholds),
            max_line_length=document.profile.max_line_length,
            max_lines=document.profile.max_lines,
        )
        out = _resnap_shots(
            out, list(document.shot_changes or ()) or None, dict(thresholds)
        )
    return out


def _shadow_boundary_times(cues: Sequence[Cue]) -> tuple[float, ...]:
    """Delivered interior boundaries in the one frozen transition coordinate."""
    from voxweave.core.boundary_cost import transition_time

    values: list[float] = []
    for left, right in zip(cues, cues[1:]):
        value = transition_time(left.get("end"), right.get("start"))
        if value is not None:
            values.append(float(value))
    return tuple(values)


def _shadow_movement_distribution(movement: Sequence[Any]) -> dict[str, Any]:
    """Compact absolute phase-2 movement distribution, nearest-rank."""
    import math

    def summary(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)

        def rank(percentile: float) -> float | None:
            if not ordered:
                return None
            return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

        return {
            "count": len(ordered),
            "max": max(ordered) if ordered else None,
            "p50": rank(0.50),
            "p90": rank(0.90),
        }

    return {
        side: summary(
            [abs(float(item.delta)) for item in movement if item.boundary.side == side]
        )
        for side in ("start", "end")
    }


def _shadow_source_units(units: Sequence[Any]) -> list[dict[str, Any]]:
    """Closed source-unit authority used by schema-side partition replay."""
    return [
        {
            "confidence": unit.confidence,
            "end": unit.end,
            "id": unit.id,
            "provenance": unit.provenance,
            "start": unit.start,
            "surface": unit.surface,
        }
        for unit in units
    ]


def _shadow_finalizer_verification(
    stream: Any, evidence: Any, policy: Any
) -> dict[str, Any]:
    """Digest-bound canonical inputs for schema-side finalizer replay."""
    from voxweave.core.finalizer import _stream_payload

    seed = _stream_payload(
        stream.cues,
        stream.profile,
        stream.row_id,
        stream.evaluation_id,
    )
    return {
        "authority_id": stream.seed_id,
        "authority_kind": stream.authority_kind,
        "evidence": {
            "shots": list(evidence.shots),
            "sing_spans": [list(span) for span in evidence.sing_spans],
        },
        "policy": {
            "grid": policy.grid,
            "min_gap": policy.min_gap,
            "overlap_policy": policy.overlap_policy,
        },
        "seed_digest": stream.capability.seal.digest,
        "seed_payload": seed,
    }


def _shadow_finalizer_row(
    result: Any,
    stream: Any,
    partition: Sequence[int] | None,
    *,
    document: SegDocument,
    origin: Any,
    evidence: Any,
    policy: Any,
) -> tuple[dict[str, Any], list[Cue]]:
    """Verify and serialize one finalizer root without trusting its trace."""
    from voxweave.core.partition_check import check_partition
    from voxweave.core.trace_validator import replay_trace, stability_check

    cues = [copy.deepcopy(cue) for cue in result.cues]
    delivered = tuple((float(cue["start"]), float(cue["end"])) for cue in cues)
    trace_errors = replay_trace(
        result.trace,
        stream.cues,
        profile=document.profile,
        evidence=evidence,
        policy=policy,
        delivered=delivered,
    )
    stability_errors = stability_check(
        delivered,
        stream.cues,
        profile=document.profile,
        evidence=evidence,
        policy=policy,
        terminal=result.trace.terminal,
    )
    validator = None
    partition_cardinality_ok = partition is not None and len(partition) + 1 == len(cues)
    if result.valid and partition is not None and len(partition) + 1 == len(cues):
        validator = check_partition(
            partition,
            cues,
            units=document.units,
            profile=document.profile,
            origin=origin,
            stage="finalizer",
            reports=result.report.entries,
            waivers={waiver.cue_index: waiver for waiver in result.report.waivers},
        ).to_dict()
    finalizer = {
        **result.report.to_dict(),
        "movement_distribution": _shadow_movement_distribution(result.report.movement),
        "refusals": [entry.to_dict() for entry in result.report.entries],
        "stability_errors": list(stability_errors),
        "trace": result.trace.to_dict(),
        "trace_errors": list(trace_errors),
        "valid": bool(result.valid),
    }
    return (
        {
            "cue_count": len(cues),
            "cues": _shadow_cue_rows(cues, partition, len(document.units)),
            "finalizer": finalizer,
            "partition": None if partition is None else list(partition),
            "projection": (
                "solver-partition"
                if partition_cardinality_ok
                else "cue/range-cardinality-mismatch"
                if partition is not None
                else "unresolved"
            ),
            "validator": validator,
            "verification": _shadow_finalizer_verification(stream, evidence, policy),
        },
        cues,
    )


def _shadow_fallback_rechecks(
    stream: Any,
    footprints: Sequence[str],
    *,
    row_id: str,
) -> list[dict[str, Any]]:
    """Show whether W1's missing factory footprint is the sole fallback cause."""
    from voxweave.core.canonical_text import canonical_text

    checks: list[dict[str, Any]] = []
    if len(footprints) != len(stream.cues):
        return [
            {
                "cue_index": None,
                "reason": "footprint-cardinality-mismatch",
                "row": row_id,
                "with_owned_footprint": None,
            }
        ]
    for cue, footprint in zip(stream.cues, footprints):
        fallback = next(
            (
                report
                for report in cue.reports
                if report.kind == "canonical-text-fallback"
            ),
            None,
        )
        if fallback is None:
            continue
        replayed = canonical_text(
            cue.word_data,
            fallback_text=cue.text,
            lang=stream.profile.language,
            profile=stream.profile,
            expected_footprint=footprint,
        )
        checks.append(
            {
                "cue_index": cue.index,
                "reason": fallback.evidence.get("reason"),
                "row": row_id,
                "with_owned_footprint": replayed.source,
                "with_owned_footprint_reason": replayed.fallback_reason,
            }
        )
    return checks


def _shadow_stamp_comparator_deltas(
    finalizer_row: Mapping[str, Any], comparator_row: Mapping[str, Any]
) -> None:
    """Join W4's upstream FD-2 producer fact into the finalizer row report."""
    finalizer = finalizer_row.get("finalizer")
    if not isinstance(finalizer, dict):
        return
    evidence_flags = {
        tuple(row["unit_range"]): bool(row.get("lyric"))
        for row in finalizer_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    legacy_flags = {
        tuple(row["unit_range"]): bool(row.get("lyric"))
        for row in comparator_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    fired: set[str] = set(finalizer.get("deltas_fired") or ())
    if any(
        evidence_flags[unit_range] != legacy_flags[unit_range]
        for unit_range in evidence_flags.keys() & legacy_flags.keys()
    ):
        fired.add("FD-2")
    finalizer["deltas_fired"] = sorted(fired)


def _shadow_diff_classification(
    finalizer_row: Mapping[str, Any],
    comparator_row: Mapping[str, Any],
    *,
    stream: Any | None = None,
    seed_cues: Sequence[Cue] | None = None,
    sing_spans: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    """N11: independently recompute per-cue triggers and allowed relations."""
    from voxweave.core.speaker_evidence import (
        evidence_span_from_cue,
        lyric_for_evidence,
    )
    from voxweave.core.timing import LINGER_CAP_S, TWO_FRAME_S

    finalizer = finalizer_row["finalizer"]
    producer_fired = set(finalizer.get("deltas_fired") or ())
    trace_clean = not finalizer.get("trace_errors") and not finalizer.get(
        "stability_errors"
    )
    permitted = {
        "text": {"FD-9"},
        "start": {"FD-4"},
        "end": {"FD-1", "FD-3", "FD-4", "FD-6", "FD-8"},
        "lyric": {"FD-2"},
    }
    facts: dict[int, dict[str, Any]] = {}
    independent_fired: set[str] = set()
    if (
        stream is not None
        and seed_cues is not None
        and len(stream.cues) == len(seed_cues)
    ):
        extends = (
            stream.profile.min_cue_s > 0
            or stream.profile.lag_out_s > 0
            or stream.profile.cps > 0
        )
        trace_legs = finalizer.get("trace", {}).get("legs") or ()
        for index, (phase1, seed) in enumerate(zip(stream.cues, seed_cues)):
            per_field = {field: set() for field in permitted}
            if phase1.reading_chars != phase1.raw_reading_chars:
                per_field["end"].add("FD-1")
                independent_fired.add("FD-1")
            if phase1.speech_end is None and extends:
                per_field["end"].add("FD-8")
                independent_fired.add("FD-8")
            if any(
                report.kind == "stutter-not-proven-fixed-within-4-scans"
                for report in phase1.reports
            ):
                per_field["text"].add("FD-9")
                independent_fired.add("FD-9")
            if index + 1 < len(seed_cues):
                next_seed = seed_cues[index + 1]
                if float(seed["end"]) > float(next_seed["start"]):
                    per_field["end"].add("FD-3")
                    independent_fired.add("FD-3")
                if float(next_seed["start"]) - float(seed["end"]) < TWO_FRAME_S:
                    per_field["end"].add("FD-6")
                    independent_fired.add("FD-6")
            target_legs = [
                leg
                for leg in trace_legs
                if int(leg["target"]["cue_index"]) == index
                and leg["target"]["side"] in ("start", "end")
            ]
            for leg in target_legs:
                if leg["rule_id"] in ("chain", "shot-in", "shot-out") or str(
                    leg["rule_id"]
                ).startswith("ladder-"):
                    per_field[str(leg["target"]["side"])].add("FD-4")
                    independent_fired.add("FD-4")

            evidence_lyric = lyric_for_evidence(
                evidence_span_from_cue(seed), sing_spans
            )
            start, end = float(seed["start"]), float(seed["end"])
            duration = end - start
            overlap = sum(
                max(0.0, min(end, high) - max(start, low)) for low, high in sing_spans
            )
            legacy_lyric = duration > 0 and overlap / duration >= LYRIC_MIN_OVERLAP
            if evidence_lyric != legacy_lyric:
                per_field["lyric"].add("FD-2")
                independent_fired.add("FD-2")

            wanted_end = float(seed["end"])
            if phase1.speech_end is not None:
                if stream.profile.min_cue_s > 0:
                    wanted_end = max(
                        wanted_end,
                        float(seed["start"]) + stream.profile.min_cue_s,
                    )
                if stream.profile.lag_out_s > 0:
                    wanted_end = max(
                        wanted_end,
                        float(phase1.speech_end) + stream.profile.lag_out_s,
                    )
                if stream.profile.cps > 0:
                    needed = sum(not char.isspace() for char in phase1.text) / (
                        stream.profile.cps
                    )
                    wanted_end = max(
                        wanted_end,
                        min(
                            float(seed["start"]) + needed,
                            float(phase1.speech_end) + LINGER_CAP_S,
                        ),
                    )
            facts[index] = {
                "evidence_lyric": evidence_lyric,
                "expected_phase1_end": wanted_end,
                "legacy_lyric": legacy_lyric,
                "target_legs": target_legs,
                "triggers": {field: sorted(ids) for field, ids in per_field.items()},
            }
        # FD-7 is derived from the phase-1 stream, immutable seed, delivered
        # state, and trace terminal -- never from either serialized report
        # channel.  ``entries``/``refusals`` are producer output and are checked
        # for exact equality by the shared schema validator; reading either here
        # would let a serializer mutation erase both sides of the N11 check.
        has_report = any(cue.reports for cue in stream.cues)
        has_report = has_report or any(
            float(left["end"]) > float(right["start"])
            for left, right in zip(seed_cues, seed_cues[1:])
        )
        delivered = finalizer_row.get("cues") or ()
        if len(delivered) == len(stream.cues):
            for index, (row, cue) in enumerate(zip(delivered, stream.cues)):
                start, end = float(row["start"]), float(row["end"])
                if (
                    stream.profile.min_cue_s > 0
                    and end - start < stream.profile.min_cue_s - 1e-9
                ):
                    has_report = True
                if index + 1 >= len(delivered):
                    continue
                next_start = float(delivered[index + 1]["start"])
                if (
                    next_start - end < TWO_FRAME_S - 1e-9
                    and cue.speech_end is not None
                    and next_start - TWO_FRAME_S < cue.speech_end <= next_start
                ):
                    has_report = True
        trace = finalizer.get("trace") or {}
        cycle = trace.get("cycle")
        if isinstance(cycle, Mapping):
            has_report = has_report or any(
                len(set(row.get("values") or ())) > 1
                for row in cycle.get("per_boundary_values") or ()
                if isinstance(row, Mapping)
            )
        if trace.get("terminal") == "budget-exhausted":
            has_report = True
        if has_report:
            independent_fired.add("FD-7")

    left = {
        tuple(row["unit_range"]): row
        for row in finalizer_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    right = {
        tuple(row["unit_range"]): row
        for row in comparator_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    changed: list[dict[str, Any]] = []
    unclassified = 0
    relation_failures = 0
    alignment_error = set(left) != set(right)
    movement = {
        (
            int(item["boundary"]["cue_index"]),
            str(item["boundary"]["side"]),
        ): item
        for item in finalizer.get("movement") or ()
    }
    for unit_range in sorted(set(left) & set(right)):
        before, after = right[unit_range], left[unit_range]
        index = int(after["index"])
        fact = facts.get(index)
        for field in ("text", "start", "end", "lyric"):
            if before.get(field) == after.get(field):
                continue
            eligible = (
                sorted(permitted[field] & producer_fired)
                if fact is None
                else list(fact["triggers"][field])
            )
            relation_ok = False
            if eligible and fact is None:
                relation_ok = field == "lyric" or trace_clean
            elif fact is not None and field == "lyric" and "FD-2" in eligible:
                relation_ok = (
                    bool(after.get("lyric")) == fact["evidence_lyric"]
                    and bool(before.get("lyric")) == fact["legacy_lyric"]
                )
            elif (
                fact is not None
                and stream is not None
                and field == "text"
                and "FD-9" in eligible
            ):
                relation_ok = str(after.get("text")) == stream.cues[index].text
            elif fact is not None and field in ("start", "end"):
                move = movement.get((index, field))
                delivered_matches = move is not None and float(
                    move["delivered"]
                ) == float(after[field])
                targeted = any(
                    leg["target"]["side"] == field for leg in fact["target_legs"]
                )
                if field == "start":
                    relation_ok = delivered_matches and trace_clean and targeted
                else:
                    phase1_matches = move is not None and float(
                        move["phase1"]
                    ) == float(fact["expected_phase1_end"])
                    relation_ok = delivered_matches and any(
                        (
                            trigger == "FD-1"
                            and phase1_matches
                            and (not targeted or trace_clean)
                        )
                        or (
                            trigger in {"FD-3", "FD-4", "FD-6"}
                            and targeted
                            and trace_clean
                        )
                        or (
                            trigger == "FD-8"
                            and phase1_matches
                            and (not targeted or trace_clean)
                        )
                        for trigger in eligible
                    )
            if not eligible:
                unclassified += 1
            elif not relation_ok:
                relation_failures += 1
            changed.append(
                {
                    "allowed_relation": relation_ok,
                    "field": field,
                    "from": before.get(field),
                    "trigger_ids": eligible,
                    "to": after.get(field),
                    "unit_range": list(unit_range),
                }
            )
    return {
        "alignment_error": alignment_error,
        "changed_fields": changed,
        "independent_fired": sorted(independent_fired),
        "producer_fired": sorted(producer_fired),
        "relation_failures": relation_failures,
        "trigger_mismatches": sorted(independent_fired ^ producer_fired),
        "unclassified_field_diff": unclassified,
    }


def _shadow_v2_artifact(
    document: SegDocument, v1_cues: Sequence[Cue], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the complete P5 row matrix and assemble one schema-2 artifact."""
    from voxweave.core.authority import (
        AuthorityKind,
        AuthorityLedger,
        check_roots,
        digest_payload,
        lineage_tuples,
    )
    from voxweave.core.boundary_v2 import (
        V1Partition,
        _document_partition,
        _document_waivers,
        _optimization_reuse,
        optimize_document,
        selected_evidence_spans,
    )
    from voxweave.core.finalizer import (
        FinalizeEvidence,
        FinalizePolicy,
        FinalizerPreview,
        capture_v1_reference,
        finalize,
        phase1_cue,
        phase1_from_optimizer_selection,
        phase1_from_v1_capture,
        register_optimizer_selection,
    )
    from voxweave.core.layout import _join
    from voxweave.core.partition_check import owned_unit_ids
    from voxweave.core.speaker_evidence import (
        W_SPEAKER_INTERIOR,
        annotate_speaker_ids,
        evidence_span_from_cue,
        lyric_for_evidence,
        measure_speaker_events,
        named_multi_cues_unannotated,
        project_speaker_evidence,
        speaker_evidence,
    )
    from voxweave.core.subunit import empty_refine_result, refine_document
    from voxweave.core.timing_preview import CueCandidate, CuePreview

    class AuditedFinalizerPreview:
        """N7: compare every consumed preview with phase 1 itself, exactly."""

        def __init__(self, delegate: FinalizerPreview) -> None:
            self.delegate = delegate
            self.scored_edges = 0
            self.checked_edges = 0
            self.uncheckable_edges = 0
            self.mismatches: list[dict[str, Any]] = []
            self.selected_rows: dict[str, dict[str, Any]] = {}

        @staticmethod
        def _facts(value: CuePreview) -> dict[str, Any]:
            return {
                "display_end": value.display_end,
                "display_start": value.display_start,
                "final_text": value.final_text,
                "line_count": value.line_count,
                "reading_chars": value.reading_chars,
                "refusals": [row.to_dict() for row in value.refusals],
                "waivers": [row.to_dict() for row in value.waivers],
            }

        def preview_cue(self, candidate: CueCandidate) -> CuePreview:
            consumed = self.delegate.preview_cue(candidate)
            edge_index = self.scored_edges
            self.scored_edges += 1
            if candidate.start is None or candidate.end is None:
                self.uncheckable_edges += 1
                return consumed
            seed: Cue = {
                "text": candidate.text,
                "start": candidate.start,
                "end": candidate.end,
                "word_data": list(candidate.word_data),
                "speech_start": candidate.speech_start,
                "speech_end": candidate.speech_end,
            }
            phase1 = phase1_cue(
                seed,
                profile=candidate.profile,
                index=0,
                expected_footprint=candidate.expected_footprint,
            )
            expected = CuePreview(
                display_start=phase1.start,
                display_end=phase1.end,
                final_text=phase1.text,
                line_count=len(phase1.lines),
                reading_chars=phase1.reading_chars,
                waivers=(),
                refusals=phase1.reports,
            )
            self.checked_edges += 1
            if consumed != expected:
                self.mismatches.append(
                    {
                        "consumed": self._facts(consumed),
                        "edge_index": edge_index,
                        "phase1": self._facts(expected),
                    }
                )
            return consumed

        def preview_display_span(
            self,
            start: float,
            end: float,
            next_start: float | None,
            *,
            text: str,
            word_data: Sequence[Unit],
            min_cue_s: float,
            max_cue_s: float,
            cps: float = 0.0,
            lag_out_s: float = 0.0,
        ) -> float:
            return self.delegate.preview_display_span(
                start,
                end,
                next_start,
                text=text,
                word_data=word_data,
                min_cue_s=min_cue_s,
                max_cue_s=max_cue_s,
                cps=cps,
                lag_out_s=lag_out_s,
            )

        def check_selected(self, row_id: str, solution: Any, stream: Any) -> None:
            """Bridge scored selected-edge facts to the factory's actual seed."""
            edge_facts = [
                part.features
                for interval in solution.solutions
                if interval.selection is not None
                for part in interval.selection.policy_selected.edge_breakdowns
            ]
            mismatches: list[dict[str, Any]] = []
            if len(edge_facts) != len(stream.cues):
                mismatches.append(
                    {
                        "cue_count": len(stream.cues),
                        "edge_count": len(edge_facts),
                        "reason": "cardinality",
                    }
                )
            for index, (facts, cue) in enumerate(zip(edge_facts, stream.cues)):
                consumed = {
                    "display_end": facts.get("preview_display_end"),
                    "display_start": facts.get("preview_display_start"),
                    "final_text": facts.get("preview_final_text"),
                    "line_count": facts.get("preview_line_count"),
                    "reading_chars": facts.get("preview_reading_chars"),
                    "refusal_count": facts.get("preview_refusal_count"),
                }
                phase1 = {
                    "display_end": cue.end,
                    "display_start": cue.start,
                    "final_text": cue.text,
                    "line_count": len(cue.lines),
                    "reading_chars": cue.reading_chars,
                    "refusal_count": len(cue.reports),
                }
                if consumed != phase1:
                    mismatches.append(
                        {
                            "consumed": consumed,
                            "cue_index": index,
                            "phase1": phase1,
                            "reason": "facts",
                        }
                    )
            self.selected_rows[row_id] = {
                "cue_count": len(stream.cues),
                "edge_count": len(edge_facts),
                "mismatches": mismatches,
            }

        def to_dict(self) -> dict[str, Any]:
            return {
                "checked_edges": self.checked_edges,
                "mismatches": list(self.mismatches),
                "scored_edges": self.scored_edges,
                "selected_rows": copy.deepcopy(self.selected_rows),
                "uncheckable_edges": self.uncheckable_edges,
            }

    # Capture the committed v1 bytes before any legacy overlay. The finalizer
    # input is a separate evidence-stamped copy; the delivery tripwire retains
    # this raw reference byte for byte.
    reference: list[Cue] = [copy.deepcopy(cue) for cue in v1_cues]
    parent_speakers = speaker_evidence(document)
    finalizer_reference = [copy.deepcopy(cue) for cue in reference]
    for cue in finalizer_reference:
        lyric = lyric_for_evidence(evidence_span_from_cue(cue), document.sing_spans)
        if lyric:
            cue["lyric"] = True
        else:
            cue.pop("lyric", None)
    ledger = AuthorityLedger()
    capture = capture_v1_reference(finalizer_reference, ledger=ledger)

    # Refinement is the first v2 topology operation and acts on a detached copy.
    shadow_document, split = refine_document(document)
    projected_speakers = project_speaker_evidence(
        parent_speakers, refined_units=split.units, origin=split.origin
    )
    v1_partition, v1_projection = _shadow_v1_partition(
        document, split.origin, reference
    )
    v1_reference_input = (
        None
        if v1_partition is None
        else V1Partition(cuts=v1_partition, cues=tuple(reference))
    )
    preview = AuditedFinalizerPreview(FinalizerPreview(shadow_document.profile))
    pricing_reuse = _optimization_reuse(shadow_document, canonical_spaced=True)
    solution = optimize_document(
        shadow_document,
        v1=v1_reference_input,
        preview=preview,
        subunit_split=split,
        speakers=projected_speakers,
        speaker_weight=W_SPEAKER_INTERIOR,
        _reuse=pricing_reuse,
    )
    speaker_off = optimize_document(
        shadow_document,
        v1=v1_reference_input,
        preview=preview,
        subunit_split=split,
        speakers=projected_speakers,
        speaker_weight=0.0,
        _reuse=pricing_reuse,
    )
    optimizer_artifact_bytes = json.dumps(
        solution.artifact, sort_keys=True, separators=(",", ":")
    )
    artifact = solution.artifact
    artifact["units"] = _shadow_source_units(shadow_document.units)
    artifact["preview_fidelity"] = preview.to_dict()
    artifact["v1_projection"] = {
        "cut_count": None if v1_partition is None else len(v1_partition),
        "mode": v1_projection,
        "unprojected": v1_partition is None,
    }
    if solution.invalid_profile:
        return {
            "diagnostic": artifact,
            "error": {
                "detail": "optimizer profile preflight failed",
                "type": "IncompleteShadowArtifact",
            },
            "kind": "segmentation-shadow-incomplete",
            "schema_version": 1,
        }

    unit_count = len(shadow_document.units)
    raw_partition = _document_partition(solution.solutions, unit_count)
    off_partition = _document_partition(speaker_off.solutions, unit_count)
    solver_waivers = _document_waivers(solution.solutions)
    fallback_ranges = [
        list(item.unit_range) for item in solution.solutions if not item.optimized
    ]

    raw_cues = [copy.deepcopy(cue) for item in solution.solutions for cue in item.cues]
    raw_v2 = _shadow_stream_block(
        raw_cues,
        raw_partition,
        "solver-partition",
        document=shadow_document,
        origin="v2",
        stage="raw",
        waivers=_restamp_by_footprint(solver_waivers, raw_partition, unit_count),
        origins=_origins_by_footprint(fallback_ranges, raw_partition, unit_count),
    )
    artifact["raw"] = raw_v2

    def owned_footprints(partition: Sequence[int] | None) -> list[str]:
        if partition is None:
            return []
        return [
            _join(
                [unit.surface for unit in shadow_document.units[low:high]],
                shadow_document.language,
            )
            for low, high in owned_unit_ids(partition, unit_count)
        ]

    core_cues = _shadow_core_cues(solution, shadow_document, thresholds)
    core_projection, core_projection_mode = _shadow_surface_partition(
        shadow_document.units, core_cues
    )
    core_v2 = _shadow_stream_block(
        core_cues,
        raw_partition,
        "solver-partition",
        document=shadow_document,
        origin="v2",
        stage="core",
        waivers=_restamp_by_footprint(solver_waivers, raw_partition, unit_count),
        origins=_origins_by_footprint(fallback_ranges, raw_partition, unit_count),
        extra={
            "projection_cross_check": {
                "agrees": core_projection == raw_partition,
                "mode": core_projection_mode,
            }
        },
    )
    core_v1 = _shadow_stream_block(
        reference,
        v1_partition,
        v1_projection,
        document=shadow_document,
        origin="v1",
        stage="core",
    )

    delivery_v2_cues = _shadow_overlay_cues(core_cues, shadow_document, thresholds)
    delivery_v2_partition, delivery_v2_mode = _shadow_surface_partition(
        shadow_document.units, delivery_v2_cues
    )
    delivery_v1_cues = _shadow_overlay_cues(reference, shadow_document, thresholds)
    delivery_v1_partition, delivery_v1_mode = _shadow_surface_partition(
        shadow_document.units, delivery_v1_cues
    )
    delivery_v2 = _shadow_stream_block(
        delivery_v2_cues,
        delivery_v2_partition,
        delivery_v2_mode,
        document=shadow_document,
        origin="v2",
        stage="legacy-overlay",
        waivers=_restamp_by_footprint(
            solver_waivers, delivery_v2_partition, unit_count
        ),
        origins=_origins_by_footprint(
            fallback_ranges, delivery_v2_partition, unit_count
        ),
    )
    delivery_v1 = _shadow_stream_block(
        delivery_v1_cues,
        delivery_v1_partition,
        delivery_v1_mode,
        document=shadow_document,
        origin="v1",
        stage="legacy-overlay",
    )

    # Adjacent typed fallbacks adopt COMPLETE v1 cues, so two of them can expand
    # onto the same cue and the raw-stage document validator then sees that cue
    # twice. That is a reporting artifact of the fallback contract, not a
    # conservation result, so it is flagged where a reader meets it rather than
    # left to be mistaken for evidence. It cannot arise on the public corpus,
    # where the C13 gate forbids fallbacks outright.
    overlapping = any(
        left[1] > right[0] for left, right in zip(fallback_ranges, fallback_ranges[1:])
    )
    # Every row gets one typed upstream authority and one finalizer root.
    evaluation_id = (
        "p5:"
        + digest_payload(
            {
                "language": shadow_document.language,
                "profile": artifact["profile"],
                "units": [unit.surface for unit in shadow_document.units],
            }
        )[:20]
    )
    finalizer_evidence = FinalizeEvidence(
        shots=tuple(shadow_document.shot_changes or ()),
        sing_spans=tuple(shadow_document.sing_spans or ()),
    )
    finalizer_policy = FinalizePolicy()
    unavailable = {
        "v2": [
            item.interval.index for item in solution.solutions if not item.optimized
        ],
        "v2-speaker-off": [
            item.interval.index for item in speaker_off.solutions if not item.optimized
        ],
    }
    if any(unavailable.values()):
        # The frozen optimizer factory correctly refuses an adopted-v1 interval:
        # it has no optimizer selection to seal. Preserve the useful core/legacy
        # diagnostics and state the unmaterialized rows instead of letting that
        # typed precondition collapse the whole fail-open artifact to ``error``.
        v1_stream = phase1_from_v1_capture(
            capture,
            profile=shadow_document.profile,
            ledger=ledger,
            row_id=f"{SHADOW_LANE_FINALIZER}/v1",
            evaluation_id=evaluation_id,
        )
        v1_finalized = finalize(
            v1_stream,
            profile=shadow_document.profile,
            evidence=finalizer_evidence,
            policy=finalizer_policy,
        )
        v1_row, _v1_final_cues = _shadow_finalizer_row(
            v1_finalized,
            v1_stream,
            v1_partition,
            document=shadow_document,
            origin="v1",
            evidence=finalizer_evidence,
            policy=finalizer_policy,
        )
        comparator_cues = [copy.deepcopy(cue) for cue in capture.cues]
        for cue in comparator_cues:
            cue.pop("lyric", None)
        mark_lyric_cues(comparator_cues, _copied_spans(shadow_document.sing_spans))
        comparator_row = _shadow_stream_block(
            comparator_cues,
            v1_partition,
            v1_projection,
            document=shadow_document,
            origin="v1",
            stage="core",
        )
        actual_fallbacks = sum(not item.optimized for item in solution.solutions)
        optimized_units = sum(
            item.interval.unit_end - item.interval.unit_start
            for item in solution.solutions
            if item.optimized
        )
        artifact["coverage"] = {
            **artifact["coverage"],
            "coarse_granularity_intervals": sum(
                item.lattice.infeasible is not None
                and item.lattice.infeasible.reason == "coarse-granularity"
                for item in solution.solutions
            ),
            "fallback_intervals": actual_fallbacks,
            "fallback_ranges_overlap": overlapping,
            "fallback_unit_ranges": fallback_ranges,
            "optimized_intervals": len(solution.solutions) - actual_fallbacks,
            "optimized_unit_ratio": (
                1.0 if unit_count == 0 else optimized_units / unit_count
            ),
            "raw_conservation_trustworthy": not overlapping,
            "unit_count": unit_count,
            "v1_unprojected": v1_partition is None,
        }
        artifact["validator"]["raw"] = raw_v2["validator"]
        artifact["validator"]["core"] = core_v2["validator"]
        artifact["validator"]["legacy_overlay"] = delivery_v2["validator"]
        artifact["validator"]["raw_duplicate_v1_cues"] = overlapping
        artifact["validator"]["finalizer"] = None
        artifact["lanes"] = {
            SHADOW_LANE_CORE: _shadow_lane_block(
                SHADOW_LANE_CORE, "core", core_v1, core_v2
            ),
            SHADOW_LANE_DELIVERY_LEGACY: _shadow_lane_block(
                SHADOW_LANE_DELIVERY_LEGACY,
                "legacy-overlay",
                delivery_v1,
                delivery_v2,
            ),
            SHADOW_LANE_FINALIZER: {
                "lane": SHADOW_LANE_FINALIZER,
                "rows": {
                    "v1": v1_row,
                    "v2": {
                        "materialized": False,
                        "reason": "adopted-v1-has-no-optimizer-authority",
                    },
                    "v2-speaker-off": {
                        "materialized": False,
                        "reason": "adopted-v1-has-no-optimizer-authority",
                    },
                },
                "stage": "finalizer",
            },
            SHADOW_LANE_LEGACY_DISPLAY: {
                "lane": SHADOW_LANE_LEGACY_DISPLAY,
                "rows": {"v1": comparator_row},
                "stage": "legacy-display",
            },
        }
        expected: dict[str, AuthorityKind] = {
            f"{SHADOW_LANE_FINALIZER}/v1": "v1-capture"
        }
        artifact["authorities"] = {
            "events": [event.to_dict() for event in ledger.events],
            "expected": expected,
            "lineage": [list(record) for record in lineage_tuples(ledger)],
            "violations": list(check_roots(ledger, expected=expected)),
        }
        _shadow_stamp_comparator_deltas(v1_row, comparator_row)
        artifact["diff_classification"] = _shadow_diff_classification(
            v1_row,
            comparator_row,
            stream=v1_stream,
            seed_cues=capture.cues,
            sing_spans=tuple(shadow_document.sing_spans or ()),
        )
        artifact["canonical_fallback_rechecks"] = _shadow_fallback_rechecks(
            v1_stream,
            owned_footprints(v1_partition),
            row_id="v1",
        )
        artifact["finalizer"] = None
        artifact["invalid_optimizer_rows"] = unavailable
        artifact["refiner_comparison"] = {
            "materialized": False,
            "reason": "optimizer-selection-unavailable",
            "refined_parent_count": split.refined_parent_count,
            "status": "unmaterialized",
        }
        artifact["speaker_evidence"]["measurement_refusal"] = (
            "optimizer-selection-unavailable"
        )
        artifact["preview_fidelity"] = preview.to_dict()
        return {
            "diagnostic": artifact,
            "error": {
                "detail": "optimizer selection authority unavailable for one or more rows",
                "type": "IncompleteShadowArtifact",
            },
            "kind": "segmentation-shadow-incomplete",
            "schema_version": 1,
        }

    v1_stream = phase1_from_v1_capture(
        capture,
        profile=shadow_document.profile,
        ledger=ledger,
        row_id=f"{SHADOW_LANE_FINALIZER}/v1",
        evaluation_id=evaluation_id,
    )
    on_authority = register_optimizer_selection(solution, ledger=ledger)
    on_stream = phase1_from_optimizer_selection(
        on_authority,
        ledger=ledger,
        row_id=f"{SHADOW_LANE_FINALIZER}/v2",
        evaluation_id=evaluation_id,
    )
    off_authority = register_optimizer_selection(speaker_off, ledger=ledger)
    off_stream = phase1_from_optimizer_selection(
        off_authority,
        ledger=ledger,
        row_id=f"{SHADOW_LANE_FINALIZER}/v2-speaker-off",
        evaluation_id=evaluation_id,
    )
    preview.check_selected("v2", solution, on_stream)
    preview.check_selected("v2-speaker-off", speaker_off, off_stream)
    canonical_fallback_rechecks = [
        *_shadow_fallback_rechecks(
            v1_stream,
            owned_footprints(v1_partition),
            row_id="v1",
        ),
        *_shadow_fallback_rechecks(
            on_stream,
            owned_footprints(raw_partition),
            row_id="v2",
        ),
        *_shadow_fallback_rechecks(
            off_stream,
            owned_footprints(off_partition),
            row_id="v2-speaker-off",
        ),
    ]
    v1_finalized = finalize(
        v1_stream,
        profile=shadow_document.profile,
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    on_finalized = finalize(
        on_stream,
        profile=shadow_document.profile,
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    off_finalized = finalize(
        off_stream,
        profile=shadow_document.profile,
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    v1_row, _v1_final_cues = _shadow_finalizer_row(
        v1_finalized,
        v1_stream,
        v1_partition,
        document=shadow_document,
        origin="v1",
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    on_row, on_final_cues = _shadow_finalizer_row(
        on_finalized,
        on_stream,
        raw_partition,
        document=shadow_document,
        origin="v2",
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    off_row, off_final_cues = _shadow_finalizer_row(
        off_finalized,
        off_stream,
        off_partition,
        document=shadow_document,
        origin="v2",
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )

    # Speaker measurement uses one selected evidence basis for both independent
    # global boundary matching runs. Budget-invalid rows short-circuit here.
    measurement_refusal: str | None = None
    row_cardinality_ok = len(raw_partition) + 1 == len(on_final_cues) and len(
        off_partition
    ) + 1 == len(off_final_cues)
    if on_finalized.valid and off_finalized.valid and not row_cardinality_ok:
        measurement_refusal = "cue/range-cardinality-mismatch"
    elif on_finalized.valid and off_finalized.valid:
        try:
            evidence_spans = selected_evidence_spans(solution)
        except ValueError as exc:
            measurement_refusal = str(exc)
        else:
            on_measurement = measure_speaker_events(
                projected_speakers,
                evidence_spans=evidence_spans,
                delivered_boundaries=_shadow_boundary_times(on_final_cues),
                off_boundaries=_shadow_boundary_times(off_final_cues),
            )
            off_measurement = measure_speaker_events(
                projected_speakers,
                evidence_spans=evidence_spans,
                delivered_boundaries=_shadow_boundary_times(off_final_cues),
            )
            for name, measured in (
                ("v2", on_measurement),
                ("v2-speaker-off", off_measurement),
            ):
                if (
                    sum(measured.buckets.values())
                    != measured.raw_in_speech_turn_changes
                ):
                    raise ValueError(f"{name} speaker bucket conservation failed")
            artifact["speaker_evidence"]["measurement"] = on_measurement.to_dict()
            artifact["speaker_evidence"]["off_row_measurement"] = (
                off_measurement.to_dict()
            )
            off_row["speaker_measurement"] = off_measurement.to_dict()
            ranges = owned_unit_ids(raw_partition, unit_count)
            annotate_speaker_ids(
                on_final_cues, ranges, projected_speakers.unit_speakers
            )
            projected_named_multi = named_multi_cues_unannotated(
                ranges, projected_speakers.unit_speakers
            )
            if projected_named_multi != int(
                artifact["coverage"]["named_multi_cues_unannotated"]
            ):
                raise ValueError(
                    "speaker projection counter disagrees with selected ownership"
                )
            artifact["speaker_evidence"]["projection"] = {
                "cue_count": len(on_final_cues),
                "named_multi_cues_unannotated": projected_named_multi,
                "range_count": len(ranges),
                "status": "verified",
            }
            on_row["cues"] = _shadow_cue_rows(on_final_cues, raw_partition, unit_count)
    artifact["speaker_evidence"]["measurement_refusal"] = measurement_refusal

    # Comparator: the exact captured input, with only legacy display-span lyric
    # classification applied (set and clear), no speaker overlay or resnap.
    comparator_cues = [copy.deepcopy(cue) for cue in capture.cues]
    for cue in comparator_cues:
        cue.pop("lyric", None)
    mark_lyric_cues(comparator_cues, _copied_spans(shadow_document.sing_spans))
    comparator_row = _shadow_stream_block(
        comparator_cues,
        v1_partition,
        v1_projection,
        document=shadow_document,
        origin="v1",
        stage="core",
    )
    _shadow_stamp_comparator_deltas(v1_row, comparator_row)
    diff_classification = _shadow_diff_classification(
        v1_row,
        comparator_row,
        stream=v1_stream,
        seed_cues=capture.cues,
        sing_spans=tuple(shadow_document.sing_spans or ()),
    )

    # Refiner bypass is an exact identity gate on tracked rows and a typed
    # diagnostic on genuinely refined rows. Only a fallback-free bypass can own
    # an optimizer authority/finalizer root under the frozen W1 API.
    parent_v1_cuts, parent_v1_mode = _shadow_surface_partition(
        document.units, reference
    )
    parent_v1 = (
        None
        if parent_v1_cuts is None
        else V1Partition(parent_v1_cuts, tuple(reference))
    )
    refiner_off = optimize_document(
        document,
        v1=parent_v1,
        preview=preview,
        subunit_split=empty_refine_result(document.units, language=document.language),
        speakers=parent_speakers,
        speaker_weight=W_SPEAKER_INTERIOR,
    )
    if split.refined_parent_count == 0:
        refiner_comparison: dict[str, Any] = {
            "byte_identical": optimizer_artifact_bytes
            == json.dumps(refiner_off.artifact, sort_keys=True, separators=(",", ":")),
            "refined_parent_count": 0,
            "status": "tracked-identity",
        }
    else:
        off_fallbacks = sum(not item.optimized for item in refiner_off.solutions)
        refiner_off_partition = _document_partition(
            refiner_off.solutions, len(document.units)
        )
        coarse_ranges = [
            list(item.unit_range)
            for item in refiner_off.solutions
            if item.coarse_caused
        ]
        mapped_on: set[int] = set()
        internal_on: list[int] = []
        for cut in raw_partition:
            left_parent = split.origin[cut - 1]
            right_parent = split.origin[cut]
            if left_parent == right_parent:
                internal_on.append(left_parent)
            else:
                mapped_on.add(right_parent)
        external_diff = sorted(mapped_on ^ set(refiner_off_partition))

        def covered(parent: int) -> bool:
            return any(low <= parent < high for low, high in coarse_ranges)

        diffs_confined = all(covered(parent) for parent in internal_on) and all(
            covered(max(0, cut - 1)) or covered(cut) for cut in external_diff
        )
        refiner_comparison = {
            "byte_identical": False,
            "coarse_caused_intervals": refiner_off.artifact["coverage"][
                "coarse_caused_intervals"
            ],
            "coarse_caused_unit_ranges": coarse_ranges,
            "diffs_confined_to_coarse_caused": diffs_confined,
            "external_parent_cut_diff": external_diff,
            "fallback_intervals": off_fallbacks,
            "internal_refinement_cut_parents": sorted(internal_on),
            "materialized": off_fallbacks == 0,
            "off_partition": list(refiner_off_partition),
            "on_parent_edge_partition": sorted(mapped_on),
            "parent_v1_projection": parent_v1_mode,
            "refined_parent_count": split.refined_parent_count,
            "status": "refined-counterfactual",
        }

    rows: dict[str, Any] = {
        "v1": v1_row,
        "v2": on_row,
        "v2-speaker-off": off_row,
    }
    expected: dict[str, AuthorityKind] = {
        f"{SHADOW_LANE_FINALIZER}/v1": "v1-capture",
        f"{SHADOW_LANE_FINALIZER}/v2": "optimizer-selection",
        f"{SHADOW_LANE_FINALIZER}/v2-speaker-off": "optimizer-selection",
    }
    if split.refined_parent_count and refiner_comparison["materialized"]:
        refiner_partition = _document_partition(
            refiner_off.solutions, len(document.units)
        )
        refiner_authority = register_optimizer_selection(refiner_off, ledger=ledger)
        refiner_stream = phase1_from_optimizer_selection(
            refiner_authority,
            ledger=ledger,
            row_id=f"{SHADOW_LANE_FINALIZER}/refiner-off",
            evaluation_id=evaluation_id,
        )
        preview.check_selected("refiner-off", refiner_off, refiner_stream)
        refiner_finalized = finalize(
            refiner_stream,
            profile=document.profile,
            evidence=FinalizeEvidence(
                shots=tuple(document.shot_changes or ()),
                sing_spans=tuple(document.sing_spans or ()),
            ),
            policy=finalizer_policy,
        )
        refiner_row, _refiner_cues = _shadow_finalizer_row(
            refiner_finalized,
            refiner_stream,
            refiner_partition,
            document=document,
            origin="v2",
            evidence=FinalizeEvidence(
                shots=tuple(document.shot_changes or ()),
                sing_spans=tuple(document.sing_spans or ()),
            ),
            policy=finalizer_policy,
        )
        rows["refiner-off"] = refiner_row
        expected[f"{SHADOW_LANE_FINALIZER}/refiner-off"] = "optimizer-selection"
        canonical_fallback_rechecks.extend(
            _shadow_fallback_rechecks(
                refiner_stream,
                owned_footprints(refiner_partition),
                row_id="refiner-off",
            )
        )

    totals = artifact["totals"]
    artifact["coverage"] = {
        **artifact["coverage"],
        "coarse_granularity_intervals": totals["coarse_granularity_intervals"],
        "fallback_intervals": totals["fallback_intervals"],
        "fallback_ranges_overlap": overlapping,
        "fallback_unit_ranges": fallback_ranges,
        "optimized_intervals": totals["optimized_intervals"],
        "optimized_unit_ratio": totals["optimized_unit_ratio"],
        "raw_conservation_trustworthy": not overlapping,
        "unit_count": totals["unit_count"],
        "v1_unprojected": v1_partition is None,
    }
    artifact["validator"]["raw"] = raw_v2["validator"]
    artifact["validator"]["core"] = core_v2["validator"]
    artifact["validator"]["legacy_overlay"] = delivery_v2["validator"]
    artifact["validator"]["raw_duplicate_v1_cues"] = overlapping
    artifact["validator"]["finalizer"] = on_row["validator"]
    artifact["finalizer"] = on_row["finalizer"]
    artifact["lanes"] = {
        SHADOW_LANE_CORE: _shadow_lane_block(
            SHADOW_LANE_CORE, "core", core_v1, core_v2
        ),
        SHADOW_LANE_DELIVERY_LEGACY: _shadow_lane_block(
            SHADOW_LANE_DELIVERY_LEGACY,
            "legacy-overlay",
            delivery_v1,
            delivery_v2,
        ),
        SHADOW_LANE_FINALIZER: {
            "lane": SHADOW_LANE_FINALIZER,
            "rows": rows,
            "stage": "finalizer",
        },
        SHADOW_LANE_LEGACY_DISPLAY: {
            "lane": SHADOW_LANE_LEGACY_DISPLAY,
            "rows": {"v1": comparator_row},
            "stage": "legacy-display",
        },
    }
    artifact["authorities"] = {
        "events": [event.to_dict() for event in ledger.events],
        "expected": expected,
        "lineage": [list(record) for record in lineage_tuples(ledger)],
        "violations": list(check_roots(ledger, expected=expected)),
    }
    artifact["diff_classification"] = diff_classification
    artifact["canonical_fallback_rechecks"] = canonical_fallback_rechecks
    artifact["preview_fidelity"] = preview.to_dict()
    artifact["refiner_comparison"] = refiner_comparison
    artifact["invalid_finalizer_rows"] = [
        name for name, row in rows.items() if not row["finalizer"]["valid"]
    ]
    if v1_partition is None or (
        split.refined_parent_count
        and refiner_comparison.get("materialized") is not True
    ):
        reason = (
            "v1 source partition could not be projected"
            if v1_partition is None
            else "refiner-off optimizer selection authority unavailable"
        )
        return {
            "diagnostic": artifact,
            "error": {"detail": reason, "type": "IncompleteShadowArtifact"},
            "kind": "segmentation-shadow-incomplete",
            "schema_version": 1,
        }
    # The optimizer artifact remains schema 1 until this exact completed
    # payload passes the one shared live/harness contract.  Validation ignores
    # only the version field for this pre-admission pass; every other top-level
    # key, lane/row, evidence block, and cross-block cardinality is live.
    from voxweave.core.shadow_schema import (
        LIVE_SHADOW_SCHEMA_VERSION,
        assert_shadow_v2_payload,
    )

    if artifact.get("schema_version") != 1:
        raise ValueError("live shadow admission requires a schema-1 optimizer payload")
    assert_shadow_v2_payload(artifact, require_version=False)
    artifact["schema_version"] = LIVE_SHADOW_SCHEMA_VERSION
    return artifact


def _maybe_shadow_v2(
    document: SegDocument, cues: Sequence[Cue], *, thresholds: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Measure BoundaryOptimizer v2 beside the shipped v1 answer, or do nothing.

    The flag is read FIRST and the optimizer is imported only after it passes, so
    an off run costs one environment read and a branch and never pulls a v2
    module into the process at all.

    The shadow opens its OWN nested degradation capture. ``note_degraded``
    aggregates by ``(slot, reason)`` and bumps a count on repeats, so a shadow
    that re-tokenizes inside production's capture would change the persisted
    manifest -- breaking the very contract this lane exists to respect. Nesting
    restores the outer capture on exit, so production's ledger never sees a
    shadow event. The capture is also ``quiet``: the one-shot warning log is
    process-global, and a measurement that re-runs the same providers would
    otherwise win that latch and silence the line the shipping run owed its
    operator.

    Nothing here may fail the run: a measurement that can crash the pipeline is
    worse than no measurement, so an unexpected error is recorded as a typed
    ``error`` block and the shipped cues are returned untouched.
    """
    if os.environ.get(SEG_V2_SHADOW_ENV, "").strip() != "1":
        return None
    with degradation_capture(quiet=True) as shadow_degraded:
        try:
            artifact = _shadow_v2_artifact(document, cues, thresholds)
            # AD4-4: the shadow's own ledger, collected at hook time and
            # deliberately kept out of the persisted manifest. Copied off the
            # live list so a later capture cannot append to published evidence.
            artifact["shadow_degraded"] = list(shadow_degraded)
            if artifact.get("schema_version") == 2:
                from voxweave.core.shadow_schema import assert_shadow_v2_payload

                assert_shadow_v2_payload(artifact)
        except Exception as exc:  # noqa: BLE001 - a measurement never fails the run
            log.warning("v2 shadow lane failed; shipped output is unaffected (%s)", exc)
            artifact = {
                "error": {"detail": str(exc), "type": type(exc).__name__},
                "kind": "segmentation-shadow-error",
                "schema_version": 1,
                "shadow_degraded": list(shadow_degraded),
            }
    return artifact


def segment_document(
    *,
    language: str,
    word_segments: Sequence[Mapping[str, Any]],
    vad_speech: Sequence[tuple[float, float]] | None = (),
    shot_changes: Sequence[float] | None = (),
    sing_spans: Sequence[tuple[float, float]] | None = (),
    speaker_turns: Sequence[tuple[float, float, str]] | None = (),
    thresholds: Mapping[str, Any] | None = None,
    semantic_engine: Any | None = None,
    semantic_model: str | None = None,
    smart_split_kwargs: Mapping[str, Any] | None = None,
    annotate_speakers: bool = False,
) -> SegmentationResult:
    """Turn aligned word segments into the final cue stream. Pure and deterministic.

    This is the single segmentation orchestration shared by :func:`process` (the
    post-ASR half), :func:`split` (sibling-JSON replay) and offline calibration
    replay -- nobody re-implements the adapter logic around ``smart_split``.
    The pass order is exactly what production runs:

    1. snap sentence-break punctuation onto word boundaries (zh only),
    2. flatten the units into one segment,
    3. resolve the effective thresholds (optional adaptive gap scaling),
    4. record the manifest and mint the :class:`SegDocument` (see below),
    5. ``smart_split_segments`` (content breaks + timing cleanup + shot snap),
    6. lyric marking from ``sing_spans``,
    7. speaker formatting from ``speaker_turns`` (which re-runs timing cleanup),
    8. re-snap to ``shot_changes`` because step 7 moved boundaries again.

    Step 4 sits *before* the engine on purpose: the document is the single
    authority describing what this segmentation runs on, so a later engine takes
    it as input instead of being reverse-engineered from its own output. Only
    ``degraded`` cannot be known that early; the manifest reserves the key at
    build time and the ledger is written into it once the capture closes, so the
    persisted block is byte-identical either way.

    With :data:`SEG_V2_SHADOW_ENV` on, the v2 optimizer is measured between steps
    5 and 6 and its artifact is returned on ``result.shadow``. It ships nothing:
    the cue stream, the units and the persisted manifest are byte-identical to a
    run with the flag off.

    No filesystem writes, no model loads, no ASR: an already-constructed
    ``semantic_engine`` may be passed in (its owner creates and releases it), but
    nothing here downloads or instantiates one. Every input sequence is copied
    before use, so callers can reuse their own lists afterwards.

    ``vad_speech`` distinguishes absent (``None``/empty -> single-gap-threshold
    degradation in ``gap_split``) from real spans; ``shot_changes``,
    ``sing_spans`` and ``speaker_turns`` treat absent and empty alike.
    ``thresholds`` defaults to ``config.gap_thresholds(language)``.
    ``smart_split_kwargs`` forwards layout overrides (``max_line_length``,
    ``max_lines``, ...) to ``smart_split_segments``; ``max_line_length`` also
    reaches the speaker formatter so both measure the same budget. The layout
    pair is resolved here rather than inside the engine, so the manifest records
    the values that actually ran instead of re-deriving them from a second copy
    of the defaulting rule.
    """
    from voxweave.config import gap_thresholds
    from voxweave.core.layout import default_max_line_length, default_max_lines
    from voxweave.core.smart_split import SplitThresholds, smart_split_segments

    iso = language
    units: list[dict] = [copy.deepcopy(dict(u)) for u in word_segments]
    speech_spans = _copied_spans(vad_speech)
    cuts = [float(t) for t in shot_changes] if shot_changes else None
    sings = _copied_spans(sing_spans)
    turns = _copied_turns(speaker_turns)
    extra: dict[str, Any] = dict(smart_split_kwargs or {})
    # Resolve the layout pair once and hand the resolved values to the engine:
    # passing them explicitly is what the engine would have defaulted to anyway,
    # so output is unchanged, and the manifest/profile can then quote what ran.
    max_line_length = extra.get("max_line_length")
    if max_line_length is None:
        max_line_length = default_max_line_length(iso)
    max_lines = extra.get("max_lines")
    if max_lines is None:
        max_lines = default_max_lines(iso)
    extra["max_line_length"] = max_line_length
    extra["max_lines"] = max_lines

    # zh: Qwen punctuation can drift up to one character; snap to jieba word boundary
    # to prevent smart_split from splitting mid-word (e.g. 数据|中心 instead of 数据中心).
    snapped = realign.snap_break_punct(units, iso)
    # Stranded word tails (aligner parked a word-final char across dead air) are
    # repaired only on the stream cue formation sees; ``result.units`` keeps the
    # raw aligner timings, so persisted siblings stay alignment evidence and
    # every replay re-derives the repair.
    from voxweave.core.unit_repair import repair_stranded_tails

    repaired = repair_stranded_tails(snapped, iso, speech_spans)
    seg = _units_to_seg(repaired, iso)
    base = dict(thresholds) if thresholds is not None else gap_thresholds(iso)
    effective = _maybe_adaptive_thresholds(base, snapped)
    # The nine threshold values the engine really ran on: ``effective`` is the
    # caller's mapping, which ``smart_split_segments`` normalizes through
    # ``SplitThresholds.from_mapping`` (partial mappings fill dataclass defaults).
    # Quoting that same normalization keeps the profile honest for a partial
    # mapping AND keeps the recorder strict -- ``DisplayProfile.from_resolved``
    # still raises on a missing key, it is just never handed an incomplete one.
    resolved_th = SplitThresholds.from_mapping(effective)
    profile_thresholds = {key: getattr(resolved_th, key) for key in THRESHOLD_KEYS}
    # Mirror the ONLY consumer, ``align_ctc.align_blocks_full_ctc``, which masks
    # iff the value is exactly "1". ``--no-vad-mask`` writes the literal "0",
    # which is truthy as a string, so ``bool()`` would record masking as ON for
    # the run that explicitly turned it off.
    vad_mask_on = os.environ.get("VOXWEAVE_VAD_EMISSION_MASK", "").strip() == "1"
    manifest: dict[str, Any] = {
        "manifest_version": SEGMENTATION_MANIFEST_VERSION,
        "engine": SEGMENTATION_ENGINE,
        "voxweave": _voxweave_version(),
        "python": platform.python_version(),
        "language": iso,
        # Verbatim: the profile's whole value is saying what ran, so no clamp and
        # no renormalization (the tree carries two disagreeing default sets).
        "profile": {
            "max_line_length": max_line_length,
            "max_lines": max_lines,
            **profile_thresholds,
        },
        "env": {
            # True only when the adaptive pass actually replaced values, not
            # merely because the opt-in env var was set: the pass can hand back a
            # fresh dict whose estimate happens to equal the static one.
            "gap_adaptive": effective != base,
            "vad_emission_mask": vad_mask_on,
        },
        "providers": provider_snapshot(iso),
        # Placeholder in its final position: the ledger only exists once the run
        # is over, but the key is inserted here so the persisted key order does
        # not depend on when the value arrives.
        "degraded": [],
    }
    # The document is the single authority for this segmentation, so it is minted
    # before the engine runs, not reconstructed from its output. It holds the
    # manifest by reference, which is what lets ``degraded`` be filled in below
    # without the document and the sibling JSON drifting apart.
    document = build_seg_document(
        language=iso,
        units=repaired,
        profile=DisplayProfile.from_resolved(
            iso,
            profile_thresholds,
            max_line_length=max_line_length,
            max_lines=max_lines,
        ),
        manifest=manifest,
        vad_speech=speech_spans,
        shot_changes=cuts,
        sing_spans=sings,
        speaker_turns=turns,
        text=seg["text"],
    )
    # Everything the language providers touch runs inside the capture, so the
    # manifest can say which fallbacks actually fired on this document. Nothing
    # above reaches a provider (``_units_to_seg`` joins surfaces, the threshold
    # passes read config/env, and ``provider_snapshot`` only *reports* identity),
    # so narrowing the window to the engine run drops no event.
    with degradation_capture() as degraded:
        cues = smart_split_segments(
            [seg],
            lang=iso,
            speech_spans=speech_spans,
            thresholds=effective,
            shot_changes=cuts,
            semantic_engine=semantic_engine,
            semantic_model=semantic_model,
            **extra,
        )
        # Immediately after the v1 answer and before any overlay: the shadow
        # sees exactly the stream v1 produced from exactly the inputs v1 read.
        shadow = _maybe_shadow_v2(document, cues, thresholds=effective)
        mark_lyric_cues(cues, sings)
        split_cue_count = len(cues)
        if turns:
            from voxweave.diarize import apply_speaker_format

            # Same thresholds AND line budget as smart_split so speaker splits get the
            # same timing polish and wrap width the deterministic layout just used.
            cues = apply_speaker_format(
                cues,
                turns,
                iso,
                thresholds=effective,
                max_line_length=max_line_length,
                max_lines=max_lines,
                annotate_speakers=annotate_speakers,
            )
            # ... and its cleanup can push a boundary back across a cut, so snap again.
            cues = _resnap_shots(cues, cuts, effective)
    # Fill the reserved key in place: assigning an existing key never reorders a
    # dict, so the persisted ``segmentation`` block is byte-identical to the one
    # built after the run.
    manifest["degraded"] = degraded
    if shadow is not None:
        # AD3-5/AD4-4: two origin-typed ledgers and never one merged ``degraded``.
        # ``production_degraded`` can only be copied here, once the outer capture
        # has closed and the manifest's reserved key holds the real list;
        # ``shadow_degraded`` was collected by the hook's own nested capture.
        shadow["production_degraded"] = copy.deepcopy(manifest["degraded"])
        # AD3-5's other half. Without it an artifact reader cannot tell which
        # language providers the measured run actually resolved -- and a run
        # whose ja POS tagger fell back to the character table is measuring a
        # different boundary decision than one whose tagger loaded.
        shadow["providers"] = copy.deepcopy(manifest["providers"])
    diagnostics: dict[str, Any] = {
        "unit_count": len(snapped),
        "punct_snapped": snapped is not units,
        "adaptive_thresholds": effective is not base,
        "speech_span_count": len(speech_spans or ()),
        "shot_change_count": len(cuts or ()),
        "sing_span_count": len(sings or ()),
        "speaker_turn_count": len(turns or ()),
        "semantic_engine": semantic_engine is not None,
        "split_cue_count": split_cue_count,
        "lyric_cue_count": sum(1 for c in cues if c.get("lyric")),
        "speaker_formatted": bool(turns),
        "shot_resnapped": bool(turns and cuts),
        "cue_count": len(cues),
        "shadow_v2": shadow is not None,
    }
    return SegmentationResult(
        cues=cues,
        language=iso,
        units=snapped,
        thresholds_used=effective,
        diagnostics=diagnostics,
        manifest=manifest,
        document=document,
        shadow=shadow,
    )


def _make_semantic_engine(enabled: bool) -> Any | None:
    """Build the optional semantic break engine, or ``None`` when it is off.

    Construction only resolves the configured backend; no model is loaded and no
    request is sent.  A missing backend is therefore a configuration error, not a
    runtime degradation, and is raised to the caller so ``--semantic-split``
    fails before any model or audio work instead of after a full transcription.
    The caller owns creation and release (``_release_semantic_engine``); the
    deterministic layout stays the source of truth either way.
    """
    if not enabled:
        return None
    from voxweave.semantic_breaks import SemanticBreakEngine

    return SemanticBreakEngine()


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

    original_units = units
    stored_iso = to_iso_or(language, "en")
    pieces = [str(unit.get("text") or "") for unit in units]
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

    try:
        rebuilt = realign.reinject_punct(text, units, effective_iso)
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
        round_trip_ok = "".join(str(u.get("text") or "") for u in rebuilt) == text
    else:
        rebuilt_text = " ".join(str(u.get("text") or "") for u in rebuilt)
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


def voiceprints_path(path: Path) -> Path:
    return swap_ext(Path(path), ".voiceprints.json")


def speakers_suggest_path(path: Path) -> Path:
    return swap_ext(Path(path), ".speakers.suggest.json")


def speakers_html_path(path: Path) -> Path:
    return swap_ext(Path(path), ".speakers.html")


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
    semantic_split: bool = False,
    semantic_model: str | None = None,
) -> Path:
    """Full pipeline: transcribe -> smart_split -> write siblings. Return the .vtt path.

    Pass ``word_segments`` to skip transcription (tests / special cases); that path
    also skips shot detection (no media decode in unit tests). ``keep_lyrics``
    transcribes detected songs instead of excising them and flags the sung cues
    (rendered with a music-note wrap; spans persist to JSON for ``split`` replay).
    ``sdh`` additionally writes a ``<stem>.sdh.vtt`` sidecar with PANNs-detected
    non-speech event tags merged into the dialogue (main VTT/JSON untouched).
    ``diarize`` runs pyannote speaker diarization and formats multi-speaker cues
    (dual-speaker hyphens / speaker-boundary splits; turns persist to JSON).
    ``semantic_split`` optionally lets a small model choose among legal cue
    boundaries.  The deterministic splitter remains the source of truth and
    automatic fallback; model output can never replace text or timestamps.
    """
    media_path = Path(media_path)
    rep = reporter or Reporter()
    try:
        semantic_engine = _make_semantic_engine(semantic_split)
    except Exception as exc:
        _attach_semantic_configuration_failure(exc)
        raise
    semantic_handed_off = False
    expected_json: episode_transaction.FileGeneration | None = None
    expected_vtt: episode_transaction.FileGeneration | None = None
    expected_media: str | None = None
    source_mode: episode_transaction.ProcessSourceMode | None = None

    def run_with_source(
        source_path: Path,
        snapshot_fingerprint: str | None,
        capture_enabled: bool,
    ) -> Path:
        nonlocal semantic_handed_off
        assert expected_json is not None
        assert expected_vtt is not None
        assert source_mode is not None
        semantic_handed_off = True
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
            normalize=normalize,
            skip_songs=skip_songs,
            keep_lyrics=keep_lyrics,
            sdh=sdh,
            diarize=diarize,
            voiceprints=capture_enabled,
            word_segments=word_segments,
            asr_model=asr_model,
            context=context,
            timestamps=timestamps,
            shot_snap=shot_snap,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            semantic_engine=semantic_engine,
            semantic_model=semantic_model,
        )

    try:
        if voiceprints and (not diarize or word_segments is not None):
            raise ValueError("voiceprint capture requires a fresh diarization run")
        if voiceprints:
            _log_voiceprint_notice_once()
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
        source_mode = (
            "injected-words" if word_segments is not None else "transcribed-media"
        )
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
        if voiceprints:
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
    finally:
        if not semantic_handed_off:
            _release_semantic_engine(semantic_engine)


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
    semantic_engine: Any | None = None,
    semantic_model: str | None = None,
) -> Path:
    """Execute one process run against the selected immutable/live source."""
    rep = reporter or Reporter()
    semantic_owned = True
    panns_handoff_owned = False
    vad_speech: list[tuple[float, float]] | None = None
    shot_changes: list[float] | None = None
    sing_spans: list[tuple[float, float]] | None = None
    speaker_turns: list[tuple[float, float, str]] | None = None
    _voiceprint_capture: VoiceprintCapture | None = None
    try:
        if word_segments is not None:
            iso, units = word_segments
            iso, units = _reconcile_word_segment_language(
                iso, units, override=lang_override
            )
        else:
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
                voiceprints=voiceprints,
                normalize=normalize,
                reporter=reporter,
                debug=debug,
                cache_vocals=cache_vocals_path(media_path),
                source_fingerprint=snapshot_fingerprint,
                debug_stem=media_path.stem,
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
            if shot_snap:
                from voxweave import shotdet

                rep.stage("shot detection")
                shot_changes = shotdet.detect_shot_changes(source_path)

        semantic_owned = False
        publication = _finish_process_from_units(
            media_path,
            rep=rep,
            semantic_engine=semantic_engine,
            semantic_model=semantic_model,
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
        if semantic_owned:
            _release_semantic_engine(semantic_engine)
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
    semantic_engine: Any | None,
    semantic_model: str | None,
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
    rep.stage(
        "semantic subtitle boundaries" if semantic_engine else "smart_split layout"
    )
    try:
        segmented = segment_document(
            language=iso,
            word_segments=units,
            vad_speech=vad_speech,
            shot_changes=shot_changes,
            sing_spans=sing_spans,
            speaker_turns=speaker_turns,
            semantic_engine=semantic_engine,
            semantic_model=semantic_model,
        )
    finally:
        _release_semantic_engine(semantic_engine)
    units, cues = segmented.units, segmented.cues

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
        shadow_enabled=os.environ.get(SEG_V2_SHADOW_ENV, "").strip() == "1",
        semantic_selector_enabled=semantic_engine is not None,
    )
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if machine_artifact is None:
        cleanup.append(
            episode_transaction.ArtifactCleanup(
                voiceprints_path(media_path), "voiceprints-unlink"
            )
        )
    cleanup.extend(
        (
            episode_transaction.ArtifactCleanup(
                speakers_suggest_path(media_path), "suggest-unlink"
            ),
            episode_transaction.ArtifactCleanup(
                speakers_html_path(media_path), "html-unlink"
            ),
            episode_transaction.ArtifactCleanup(
                swap_ext(media_path, ".align-evidence.json"), "evidence-unlink"
            ),
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
    auxiliary_landed: tuple[Path, ...] = ()
    if sdh_enabled and source_mode == "transcribed-media":
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
    from voxweave import sdh as sdh_mod

    rep.stage("SDH event detection (PANNs)")
    wav32 = decode_to_wav(media_path, sample_rate=SONGDET_SR)
    try:
        events = sdh_mod.detect_events(
            wav32, progress=_progress_bridge(rep, "SDH event detection (PANNs)")
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
    semantic_split: bool = False,
    semantic_model: str | None = None,
    **smart_split_kwargs,
) -> Path:
    """Re-run smart_split from persisted word_segments.

    Reuses ``vad_speech`` from the sibling JSON for gap splitting; falls back to gap-only
    mode if absent. ``timestamps`` behaves as in :func:`process`.  This remains model-free
    by default; ``semantic_split=True`` opts into the configured endpoint selector
    and fails immediately when none is configured.
    """
    # Accept the .vtt sibling too: `voxweave split foo.vtt` should not feed
    # WEBVTT bytes to json.loads.
    json_path = swap_ext(Path(json_path), ".json")
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
    speech_spans = _spans_in(data.get("vad_speech"))
    shot_changes = [float(t) for t in data.get("shot_changes") or []] or None
    sing_spans = _spans_in(data.get("sing_spans"))
    speaker_turns = _turns_in(data.get("speaker_turns"))
    mapping_path = swap_ext(json_path, ".speakers.json")
    mapping_generation = episode_transaction.capture_speaker_mapping(
        mapping_path,
        known_ids={label for _start, _end, label in speaker_turns or ()},
        warn=lambda message: log.warning("%s", message),
    )
    speaker_names = dict(mapping_generation.names)
    try:
        semantic_engine = _make_semantic_engine(semantic_split)
    except Exception as exc:
        _attach_semantic_configuration_failure(exc)
        raise
    try:
        segmented = segment_document(
            language=iso,
            word_segments=units,
            vad_speech=speech_spans,
            shot_changes=shot_changes,
            sing_spans=sing_spans,
            speaker_turns=speaker_turns,
            semantic_engine=semantic_engine,
            semantic_model=semantic_model,
            smart_split_kwargs=smart_split_kwargs,
            annotate_speakers=bool(speaker_names),
        )
    finally:
        _release_semantic_engine(semantic_engine)
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
        shadow_enabled=os.environ.get(SEG_V2_SHADOW_ENV, "").strip() == "1",
        semantic_selector_enabled=semantic_engine is not None,
    )
    pair_declared = "voiceprint_capture" in data or "voiceprint_media" in data
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if pair_declared and voiceprint_pair is None:
        cleanup.extend(
            (
                episode_transaction.ArtifactCleanup(
                    voiceprints_path(json_path), "voiceprints-unlink"
                ),
                episode_transaction.ArtifactCleanup(
                    speakers_suggest_path(json_path), "suggest-unlink"
                ),
                episode_transaction.ArtifactCleanup(
                    speakers_html_path(json_path), "html-unlink"
                ),
            )
        )
    cleanup.append(
        episode_transaction.ArtifactCleanup(
            swap_ext(json_path, ".align-evidence.json"), "evidence-unlink"
        )
    )
    try:
        episode_transaction.commit_primary_outputs(
            command="split",
            episode_path=json_path,
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
    log.info("re-split %s → %d cues", vtt_out.name, len(cues))
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
    af = ASR_LOUDNORM if normalize else None
    if separate:
        cache = cache_vocals_path(cache_owner)
        separator_identity = backend.separator_identity() if bound else None
        with cache_lock(cache) as cache_handle:
            cache_hit = False
            if cache_handle.cache_path.exists():
                if bound:
                    try:
                        companion, _validated = load_cache_companion(
                            cache_handle.companion_path
                        )
                        validate_cache_pair(
                            companion,
                            cache_handle.cache_path,
                            media_fingerprint=source_fingerprint or "",
                            separator=separator_identity or {},
                        )
                        cache_hit = True
                    except (OSError, Phase2DataError):
                        log.info(
                            "vocals cache is not bound to this align source; "
                            "re-separating: %s",
                            cache_handle.cache_path,
                        )
                else:
                    cache_hit = _vocals_cache_fresh(
                        cache_handle.cache_path,
                        cache_owner,
                    )
            if cache_hit:
                reporter.stage("vocals cache (32k)")
                log.info("reuse cached vocals %s", cache_handle.cache_path)
                wav = decode_to_wav(
                    cache_handle.cache_path,
                    audio_filter=af,
                )  # 32k flac -> 16k
                tmp.append(wav)
                return wav
        if not bound:
            legacy = cache_16k_path(cache_owner)
            with cache_lock(legacy) as legacy_handle:
                if legacy_handle.cache_path.exists() and _vocals_cache_fresh(
                    legacy_handle.cache_path,
                    cache_owner,
                ):
                    reporter.stage("vocals cache (16k legacy)")
                    log.info("reuse legacy 16k vocals %s", legacy_handle.cache_path)
                    wav = decode_to_wav(legacy_handle.cache_path, audio_filter=af)
                    tmp.append(wav)
                    return wav
        fullband, vocals, wav, voc32 = _separate_to_16k_32k(
            media, reporter=reporter, normalize=normalize
        )
        tmp.extend((fullband, vocals, wav, voc32))
        try:
            with cache_write_window(cache) as cache_handle:
                _encode_flac(voc32, cache_handle.cache_path)
            log.info("cached vocals 32k → %s", cache)
        except (OSError, subprocess.CalledProcessError) as e:
            log.warning("cache vocals failed (non-fatal): %r", e)
        return wav
    reporter.stage("decode 16k")
    wav = decode_to_wav(media, audio_filter=af)
    tmp.append(wav)
    return wav


def _write_align_json(
    json_path: Path,
    blocks: list[dict],
    spans: list[tuple[float, float]],
    units: list[dict],
    lang: str,
    vad_speech: list[tuple[float, float]] | None = None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
    voiceprint_capture: str | None = None,
    voiceprint_media: str | None = None,
    final_voiceprint_check: Callable[[], bool] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Update the sibling JSON with new alignment timing. Passes vad_speech,
    shot_changes, sing_spans, and speaker_turns through so split and subsequent
    align runs can reuse them without recomputing. Lyric flags survive on the
    re-timed segments.

    ``manifest`` is likewise a pass-through: align re-times an existing cue
    stream and never re-segments, so it preserves whatever segmentation manifest
    the document already carried and invents none when there is none.

    A stored ``segmentation`` value that is not a mapping (hand-edited file, or a
    shape from some future writer) is treated as ABSENT on the way in -- the
    align call site passes ``None`` -- and therefore does not survive the
    rewrite. That is deliberate and matches the read side:
    :func:`resolve_segmentation_manifest` also treats a non-mapping value as
    absent, so read and write agree that "not a mapping" means "no manifest".
    Preserving one verbatim is impossible anyway, since :func:`_dump_sibling_json`
    copies the manifest with ``dict()``.
    """
    segments = [
        {"text": b["text"], "start": a, "end": e}
        | ({"lyric": True} if b.get("lyric") else {})
        for b, (a, e) in zip(blocks, spans)
    ]
    _dump_sibling_json(
        json_path,
        language=lang,
        segments=segments,
        units=units,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        voiceprint_capture=voiceprint_capture,
        voiceprint_media=voiceprint_media,
        final_voiceprint_check=final_voiceprint_check,
        manifest=manifest,
    )


def _encode_align_json_bytes(
    blocks: list[dict],
    spans: list[tuple[float, float]],
    units: list[dict],
    lang: str,
    vad_speech: list[tuple[float, float]] | None = None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
    voiceprint_capture: str | None = None,
    voiceprint_media: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> bytes:
    """Encode align's final main JSON candidate without publishing it."""
    segments = [
        {"text": block["text"], "start": start, "end": end}
        | ({"lyric": True} if block.get("lyric") else {})
        for block, (start, end) in zip(blocks, spans)
    ]
    return _encode_sibling_json_bytes(
        language=lang,
        segments=segments,
        units=units,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        voiceprint_capture=voiceprint_capture,
        voiceprint_media=voiceprint_media,
        manifest=manifest,
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
) -> list[list[dict]]:
    """Route blocks to the configured aligner and return per-block units.

    Three paths — these ARE the hard-constraint full-pass routing; do NOT collapse them:
    - ja MMS: one full-file pass (``align_blocks_full_mms``).
    - en wav2vec2 CTC: one full-file windowed-emission pass (``align_blocks_full_ctc``).
    - zh·yue (no CTC config): per-cue tight-crop Qwen — each cue gets its own audio slice so
      error is contained within the sentence and inter-sentence pauses are preserved.

    Per-cue slices are appended to ``tmp_chunks`` for the caller's ``finally`` to clean up.
    """
    # cue (start,end) bounds are used ONLY as silence anchors to split movie-length audio
    # into memory-sized chunks when it overflows the single-pass DP budget — NOT to crop/route
    # per cue. align is routing-free because the input VTT timestamps are exactly what may be
    # wrong (the reason to re-align); the global DP self-locates every word. None for cues
    # without timestamps. See memory voxweave-alignment-timing.
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
        )
        reporter.advance(1)
        return units
    reporter.task("per-cue alignment", len(blocks))
    block_units: list[list[dict]] = [[] for _ in blocks]
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
        geometry = observed_geometry
        if qwen_invoker is None:
            raw_units = backend.align_text(cwav, text, iso)
        else:
            raw_units = cast(
                list[dict[str, Any]],
                qwen_invoker(
                    lambda: backend.align_text(cwav, text, iso),
                    i,
                    float(cs),
                    float(ce),
                    audio_sample_start=None if geometry is None else geometry[0],
                    audio_sample_end=None if geometry is None else geometry[1],
                    sample_rate=None if geometry is None else geometry[2],
                    sample_count=None if geometry is None else geometry[3],
                ),
            )
        block_units[i] = shift_units(raw_units, cs)
        reporter.advance(1)
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
                prior_failure=selection.result.v2_status.failure,
            )
        except Exception as minimal_error:
            log.warning("align shadow artifact unavailable: %s", minimal_error)
            return
    try:
        observer(artifact)
    except Exception as exc:
        log.warning("align shadow observer failed: %s", exc)


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

    Routes each block to its audio window (via word_segments or VTT timestamps), slices
    and aligns locally, interpolates insertion blocks, then writes timing. ASR is not
    re-run; smart_split is not touched. All models run in-process (no network calls).
    """
    vtt_path = require_vtt(Path(vtt_path))  # align overwrites the input as VTT
    explicit_media_requested = media_path is not None
    rep = reporter or Reporter()
    json_path = swap_ext(vtt_path, ".json")
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

    try:
        input_snapshot = decode_align_snapshot(
            vtt_path.name,
            vtt_input_bytes,
            json_input_bytes,
            effective_iso=iso,
            sibling_snapshot=sibling_snapshot,
        )
    except RuntimeError as exc:
        _attach_vtt_decode_failure(exc)
        raise

    media = Path(media_path) if media_path else _find_sibling_media(vtt_path)
    if media is None or not media.exists():
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
        raise exc
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

    mms = backend.uses_mms(iso)
    ctc_model = None if mms else align_model_for(iso)
    full_pass = mms or bool(ctc_model)
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
            exc = RuntimeError(
                f"{json_path.name} has no word_segments and VTT has no timestamps; "
                f"cannot route audio windows"
            )
            _attach_canonical_failure(
                exc,
                kind="qwen-route-invalid",
                phase="route-plan",
                detail_code="no-route-source",
            )
            raise exc
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
        wav = _prepare_16k_for_align(
            acquisition_media,
            separate=separate,
            normalize=normalize,
            reporter=rep,
            tmp=tmp,
            cache_media=media,
            source_fingerprint=(
                selected_snapshot.fingerprint if selected_snapshot is not None else None
            ),
        )
        from voxweave import align_orchestration

        route_kind = "mms-full" if mms else "ctc-full" if ctc_model else "qwen-crop"
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
        align_context = align_orchestration.issue_public_align_context(
            target_path=vtt_path,
            sibling_path=json_path,
            media_path=media,
            prepared_audio_path=wav,
            expected_vtt=episode_transaction.FileGeneration(True, vtt_input_bytes),
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
            "media_logical_id": (
                f"explicit:{media.name}"
                if explicit_media_requested
                else f"sibling:{media.suffix.lower()}"
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
                    "start": block["start"],
                    "end": block["end"],
                }
                for block in blocks
            ],
            "crops": [
                None if crop is None else [float(crop[0]).hex(), float(crop[1]).hex()]
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

        fresh_session = begin_fresh_alignment(
            align_context,
            alignment_texts=tuple(
                str(block.get("alignment_text", block["text"])) for block in blocks
            ),
            source_indices=tuple(int(block["source_index"]) for block in blocks),
            language=iso,
            prepared_audio_sample_count=prepared_sample_count,
            sample_rate=prepared_sample_rate,
            backend_model_config_digest=stable_fact_digest(model_facts),
            route_input_digest=stable_fact_digest(route_facts),
        )
        capture_raw_call = _fresh_alignment_call_observer(fresh_session)
        invoke_qwen_call = _fresh_alignment_qwen_invoker(fresh_session)

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
            speech_spans=_spans_in(data.get("vad_speech")),
            raw_call_observer=capture_raw_call,
            qwen_invoker=invoke_qwen_call,
        )

        # Tight cropping eliminates "last word drifts into inter-sentence silence", so
        # position_units_with_vad is not needed here (unlike the transcribe path).
        final, all_units = realign.group_block_spans(block_units)
        if not all_units:
            exc = RuntimeError(f"no aligned units for {media.name}")
            _attach_canonical_failure(
                exc,
                kind="no-aligned-units",
                phase="fresh-acquisition",
                detail_code="all-block-units-empty",
            )
            raise exc
        acquisition = seal_fresh_alignment(fresh_session)
        # fill_insert -> enforce_min_duration -> rescue_tiny_cues (extend flash cues like
        # so/あ, overlap allowed with next-neighbor only) -> clamp.
        spans_filled = realign.clamp_spans(
            realign.rescue_tiny_cues(
                realign.enforce_min_duration(
                    realign.fill_insert_blocks(final), min_dur=MIN_CUE_SEC
                ),
                trig=TINY_CUE_SEC,
                target=TINY_CUE_TARGET,
            )
        )

        # Preserve vad_speech / shot_changes from the original JSON (computed by
        # transcribe from the original media; align does not recompute them).
        keep_vad = _spans_in(data.get("vad_speech"))
        keep_shots = [float(t) for t in data.get("shot_changes") or []] or None
        keep_sing = _spans_in(data.get("sing_spans"))
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
            block_units=block_units,
            spans=spans_filled,
            all_units=all_units,
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
                (
                    episode_transaction.ArtifactCleanup(
                        voiceprints_path(vtt_path), "voiceprints-unlink"
                    ),
                    episode_transaction.ArtifactCleanup(
                        speakers_suggest_path(vtt_path), "suggest-unlink"
                    ),
                    episode_transaction.ArtifactCleanup(
                        speakers_html_path(vtt_path), "html-unlink"
                    ),
                )
            )

        rep.stage("write VTT + JSON")
        from voxweave.align_evidence import encode_align_evidence

        evidence_artifact = episode_transaction.EvidencePublication(
            swap_ext(vtt_path, ".align-evidence.json"),
            encode_align_evidence(selection.evidence),
        )
        episode_transaction.commit_primary_outputs(
            command="align",
            episode_path=vtt_path,
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
            evidence_artifact=evidence_artifact,
        )
        completed_selection = selection
        aligned_cue_count = len(blocks)
        aligned_unit_count = len(all_units)
    finally:
        # Release aligner singleton VRAM (separation self-releases earlier).
        snapshots.close()
        backend.release()
        for p in tmp:
            p.unlink(missing_ok=True)
        for c in tmp_chunks:
            c.unlink(missing_ok=True)
        if align_context is not None:
            from voxweave import align_orchestration

            align_orchestration.retire_align_selection(align_context)
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
    log.info(
        "aligned %s → %d cues, %d units",
        vtt_path.name,
        aligned_cue_count,
        aligned_unit_count,
    )
    return vtt_path


def translate(
    vtt_path: Path,
    *,
    to: str = "zh",
    context: str | None = None,
    glossary: dict[str, str] | str | None = None,
    model: str = translate_mod.TRANSLATE_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    reporter: Reporter | None = None,
) -> Path:
    """Translate subtitle cues via OpenAI; write <stem>.<to>.<ext> (source untouched).

    Accepts VTT/SRT/ASS/SSA; the output mirrors the input format (SSA is written
    back as ASS). Missing translations are retried once; any remaining are
    back-filled with source text. Output cue count always equals input cue count.
    """
    from voxweave.subformats import require_subtitle

    vtt_path = require_subtitle(Path(vtt_path))
    ext = ".ass" if vtt_path.suffix.lower() == ".ssa" else vtt_path.suffix.lower()
    rep = reporter or Reporter()
    blocks = _load_cues(vtt_path)
    if any(b.get("start") is None for b in blocks):
        if ext != ".vtt":
            raise ValueError(
                f"{vtt_path.name} has cues without timestamps; cannot render {ext}"
            )
        log.warning(
            "%s has no timestamps; translated output will be plain-text blocks (run align first)",
            vtt_path.name,
        )

    payload = translate_mod.build_payload(blocks)
    # Progress sidecar: completed windows survive a mid-run failure (network,
    # rate limit), so rerunning the same command resumes instead of restarting.
    progress_path = swap_ext(vtt_path, f".{to}.progress.json")
    tx_kwargs: dict[str, Any] = dict(
        to=to,
        model=model,
        context=context,
        glossary=glossary,
        base_url=base_url,
        api_key=api_key,
        progress_path=progress_path,
        progress_sig=translate_mod.payload_signature(payload),
    )
    rep.stage(f"translate {len(payload)} cues -> {to}")
    try:
        trans = translate_mod.translate_cues(payload, **tx_kwargs, reporter=rep)

        missing = translate_mod.validate_and_fill(blocks, trans)
        if missing:
            rep.stage(f"retry translate {len(missing)} cues")
            retry_payload = [payload[i] for i in missing]
            # Continuity tail: hand the retry window the already-translated cues that
            # precede the first gap so the model keeps register/terminology consistent.
            tail = [
                (payload[j]["t"], trans[j])
                for j in range(missing[0])
                if trans.get(j, "").strip()
            ][-translate_mod.CONTEXT_TAIL :]
            trans.update(
                translate_mod.translate_cues(
                    retry_payload, **tx_kwargs, tail=tail, reporter=rep
                )
            )
            still = translate_mod.validate_and_fill(blocks, trans)
            if still:
                log.warning(
                    "%d cues still untranslated, back-filling with source text: %s",
                    len(still),
                    still,
                )
    except Exception:
        if progress_path.exists():
            log.warning(
                "translation interrupted; progress saved to %s -- rerun the same "
                "command to resume",
                progress_path.name,
            )
        raise

    rep.stage(f"write translated {ext.lstrip('.').upper()}")
    rows = translate_mod.translated_rows(
        blocks,
        trans,
        to_iso=to_iso_or(to, None),
        voice_tags=ext == ".vtt",
    )
    if ext == ".vtt":
        content = realign.render_cues(rows)
    else:
        from voxweave.export import render_ass, render_srt

        timed = [
            (float(s), float(e), t)
            for s, e, t in rows
            if s is not None and e is not None
        ]
        timed_blocks = [
            translate_mod._speaker_block_for_rendered(block, text)
            for block, (start, end, text) in zip(blocks, rows)
            if start is not None and end is not None
        ]
        content = (
            render_srt(timed, blocks=timed_blocks)
            if ext == ".srt"
            else render_ass(timed, blocks=timed_blocks)
        )
    out_path = swap_ext(vtt_path, f".{to}{ext}")
    fsio.atomic_write_text(out_path, content)
    progress_path.unlink(missing_ok=True)  # translation landed; sidecar done
    log.info("wrote %s (%d cues → %s)", out_path.name, len(blocks), to)
    return out_path


def correct(
    vtt_path: Path,
    *,
    glossary: dict[str, str] | str | None = None,
    model: str = asrfix_mod.FIX_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    apply: bool = False,
    align_after: bool = False,
    media_path: Path | None = None,
    separate: bool = True,
    normalize: bool = False,
    lang_override: str | None = None,
    reporter: Reporter | None = None,
) -> dict[str, Any]:
    """LLM ASR correction (run before align): send VTT to the LLM for a conservative diff.

    Default (review): writes sidecar ``<stem>.asrfix.vtt`` + audit ``<stem>.asrfix.json``,
    source untouched. ``apply``: overwrites the original VTT in place and writes **no audit
    json** (the diff is shown in the summary). When ``align_after`` and a real change was
    applied, immediately re-runs :func:`align` to refresh timestamps (text edits change
    word counts) and update the sibling ``<stem>.json``.

    Returns ``{out, audit, applied, rejected, n_cues, applied_in_place, aligned}``.
    """
    vtt_path = require_vtt(Path(vtt_path))  # --apply overwrites the input as VTT
    rep = reporter or Reporter()
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
    from voxweave.subformats import load_subtitle_blocks_bytes

    blocks = load_subtitle_blocks_bytes(vtt_path, vtt_input_bytes)

    payload = asrfix_mod.build_payload(blocks)
    rep.stage(f"LLM correction {len(payload)} cues (model={model})")
    fixes = asrfix_mod.correct_cues(
        payload, model=model, glossary=glossary, base_url=base_url, api_key=api_key
    )
    new_texts, applied, rejected = asrfix_mod.apply_fixes(blocks, fixes)
    rendered = asrfix_mod.render_vtt(blocks, new_texts)

    audit_path: Path | None = None
    if apply:
        # in-place edit: overwrite the original, no sidecar json (diff lives in the summary)
        rep.stage("overwrite VTT in place")
        rendered_bytes = rendered.encode("utf-8")
        episode_transaction.commit_correction(
            vtt_path=vtt_path,
            expected_vtt=episode_transaction.FileGeneration(True, vtt_input_bytes),
            rendered_vtt_bytes=rendered_bytes,
            evidence_path=swap_ext(vtt_path, ".align-evidence.json"),
        )
        out_path = vtt_path
    else:
        rep.stage("write sidecar VTT + audit json")
        out_path = swap_ext(vtt_path, ".asrfix.vtt")
        fsio.atomic_write_text(out_path, rendered)
        audit_path = swap_ext(vtt_path, ".asrfix.json")
        try:
            fsio.atomic_write_text(
                audit_path,
                json.dumps(
                    {"applied": applied, "rejected": rejected},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception:
            # Sidecar VTT + audit JSON are a pair; if the audit write fails, unlink
            # the VTT so no orphaned half-pair is left behind (source stays untouched).
            out_path.unlink(missing_ok=True)
            raise
    log.info(
        "asrfix %s: %d applied / %d rejected → %s",
        vtt_path.name,
        len(applied),
        len(rejected),
        out_path.name,
    )

    # apply means "change the file for real" -> refresh timing right away (only worth it
    # if something actually changed; an empty diff leaves the VTT identical).
    aligned = False
    if apply and align_after and applied:
        align(
            out_path,
            media_path=media_path,
            separate=separate,
            normalize=normalize,
            lang_override=lang_override,
            reporter=rep,
            _expected_vtt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        )
        aligned = True

    return {
        "out": out_path,
        "audit": audit_path,
        "applied": applied,
        "rejected": rejected,
        "n_cues": len(blocks),
        "applied_in_place": apply,
        "aligned": aligned,
    }
