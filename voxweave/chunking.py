from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import soundfile as sf

from voxweave import config
from voxweave.align_failures import CanonicalFailure

log = logging.getLogger("voxweave")


def _env_int(name: str, default: int) -> int:
    """Import-time int knob via the tolerant ``config._env_int``: a malformed value falls
    back to ``default`` instead of raising (which would break every CLI command, even
    ``--help``), with a warning naming the variable."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            int(raw)
        except ValueError:
            log.warning("ignoring %s=%r (not an integer); using %s", name, raw, default)
    return config._env_int(name, default)


def _env_float(name: str, default: float) -> float:
    """Float counterpart of :func:`_env_int` (``config._env_float``, warning on a typo)."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            float(raw)
        except ValueError:
            log.warning("ignoring %s=%r (not a number); using %s", name, raw, default)
    return config._env_float(name, default)


SAMPLE_RATE = 16000
# Raised from silero default 100ms to 300ms: 200ms chops natural mid-sentence pauses.
VAD_MIN_SILENCE_MS = _env_int("VOXWEAVE_VAD_MIN_SILENCE_MS", 300)
# Wall-clock cap for a single ffmpeg decode; overridable via VOXWEAVE_FFMPEG_TIMEOUT.
FFMPEG_TIMEOUT = _env_float("VOXWEAVE_FFMPEG_TIMEOUT", 3600.0)
# Lines of ffmpeg stderr kept in a decode error (the tail holds the actual failure).
_FFMPEG_STDERR_LINES = 8
# Non-mono decodes (the separator's full-band input) are capped at stereo. For
# 3+ channels the negotiated conversion is ffmpeg's standard downmix, sample for
# sample what `-ac 2` produces (centre and surrounds folded into left/right).
# Unlike `-ac 2`, a mono source stays mono instead of being upmixed at -3 dB:
# separate_vocals duplicates it at unity, exactly as before this cap existed.
# Stereo passes through untouched.
STEREO_CAP_FILTER = "aformat=channel_layouts=mono|stereo"


def pack_speech_segments(segments: list[dict], max_sec: float) -> list[dict]:
    """Bin-pack silero speech segments [{start,end}] into chunks of <= max_sec, cut at silence boundaries.

    Returns [{start, end, offset}] (offset == start, for timestamp shifting).
    Single segments longer than max_sec are hard-cut into max_sec slices (no silence to snap
    to; word cuts tolerated); the remainder stays the open block, so following segments can
    still merge into it instead of it becoming a sliver chunk of its own.
    """
    if not segments:
        return []
    chunks: list[dict] = []

    def emit(start: float, end: float) -> None:
        chunks.append({"start": start, "end": end, "offset": start})

    def open_block(start: float, end: float) -> tuple[float, float]:
        # Emits the full max_sec slices of an overlong segment; returns the open remainder.
        t = start
        while end - t > max_sec:
            emit(t, t + max_sec)
            t += max_sec
        return t, end

    cur_start, cur_end = open_block(segments[0]["start"], segments[0]["end"])
    for seg in segments[1:]:
        if seg["end"] - cur_start <= max_sec:
            cur_end = seg[
                "end"
            ]  # still within budget; merge into current chunk (including intervening silence)
        else:
            emit(cur_start, cur_end)  # close at silence boundary
            cur_start, cur_end = open_block(seg["start"], seg["end"])
    emit(cur_start, cur_end)
    return chunks


def plan_dp_chunks(
    bounds: Sequence[tuple[float, float] | None],
    *,
    max_sec: float,
    min_gap_sec: float = 1.5,
    pad_sec: float = 0.5,
    audio_end: float | None = None,
) -> list[dict]:
    """Partition cues into DP chunks split at silence anchors under a duration budget.

    The full-file CTC forced-align DP is O(T*L); movie-length audio overflows it. This splits
    the cue list into contiguous runs whose audio span stays within ``max_sec``, cutting only at
    cue boundaries (which never bisect a word — smart_split invariant) and PREFERRING boundaries
    backed by an inter-cue gap >= ``min_gap_sec`` (real silence, so the crop window has room for
    the per-chunk ``<star>`` edges to absorb lead-in/out). Each chunk re-runs the routing-free
    global DP over its own crop, so within-chunk drift-immunity is preserved; boundaries land in
    silence, so no word crosses them.

    ``bounds[i]`` = ``(start, end)`` of cue i, or ``None`` for a timestamp-less cue (insertion /
    empty) — it carries no anchor and just rides along in its chunk. Returns
    ``[{lo, hi, start, end}]`` where ``lo:hi`` is the cue index slice (``hi`` exclusive) and
    ``start``/``end`` is the audio crop window: adjacent chunks meet at the gap midpoint, file
    edges are padded by ``pad_sec`` (left clamped to 0, right capped at ``audio_end`` if given).

    The budget applies to that crop window, not just the cue span, since the crop is what the
    DP runs over (align_dp_safety.validate_over_budget_plans refuses a crop over ``max_sec``).
    The final chunk extends to ``audio_end`` when that still fits, else stops ``pad_sec`` past
    its last cue. A single cue (plus its crop edges) longer than ``max_sec`` cannot fit
    whatever the split, and is still emitted as its own over-budget chunk.
    """
    n = len(bounds)
    if n == 0:
        return []

    def _start(i: int) -> float | None:
        b = bounds[i]
        return b[0] if b is not None else None

    def _end(i: int) -> float | None:
        b = bounds[i]
        return b[1] if b is not None else None

    def _first_start(i: int) -> float | None:
        for k in range(i, n):
            if (s := _start(k)) is not None:
                return s
        return None

    def _last_end(i: int) -> float | None:
        for k in range(n - 1, i - 1, -1):
            if (e := _end(k)) is not None:
                return e
        return None

    def _gap(c: int) -> float | None:
        e, s = _end(c), _start(c + 1)
        return None if e is None or s is None else s - e

    def _split_time(c: int) -> float:
        # midpoint of the gap after cue c; if next cue has no timestamp, pad past cue c's end
        e, s = _end(c), _start(c + 1)
        if e is None:  # cut cue has no end (pathological): fall back to next known end
            e = next((_end(k) for k in range(c, n) if _end(k) is not None), None)
        if e is None:
            return _last_end(0) or 0.0
        return e + pad_sec if s is None else (e + s) / 2.0

    def _crop_start(lo: int) -> float:
        # left edge of the crop of the chunk starting at cue lo (what the budget measures from)
        if lo > 0:
            return _split_time(lo - 1)
        return max(0.0, (_first_start(lo) or 0.0) - pad_sec)

    def _tail_end(last: float) -> float:
        # tightest right edge of a final chunk: pad_sec past its last cue, capped at audio_end
        return last + pad_sec if audio_end is None else min(audio_end, last + pad_sec)

    cuts: list[int] = []  # split AFTER cue index c
    i = 0
    while True:
        cstart = _first_start(i)
        rem_end = _last_end(i)
        if cstart is None or rem_end is None:
            break
        crop_start = _crop_start(i)
        if _tail_end(rem_end) - crop_start <= max_sec:
            break  # remaining cues fit in one final chunk
        last_any: int | None = None
        last_gap: int | None = None
        for k in range(i, n - 1):
            ek = _end(k)
            if ek is None:
                continue
            if ek - crop_start > max_sec:
                break
            if _split_time(k) - crop_start > max_sec:
                continue  # the crop would run to the gap midpoint after cue k
            last_any = k
            g = _gap(k)
            if g is not None and g >= min_gap_sec:
                last_gap = k
        chosen = last_gap if last_gap is not None else last_any
        if chosen is None:  # leading cue alone exceeds budget; no anchor to do better
            chosen = i
        if chosen >= n - 1:  # cut would fall after the last cue: rest is one chunk
            break
        cuts.append(chosen)
        i = chosen + 1

    edges = [0, *[c + 1 for c in cuts], n]
    chunks: list[dict] = []
    for lo, hi in zip(edges, edges[1:]):
        start = _crop_start(lo)
        if hi < n:
            end = _split_time(hi - 1)
        else:
            end = (
                audio_end if audio_end is not None else (_last_end(lo) or 0.0) + pad_sec
            )
            last = _last_end(lo)
            if last is not None and end - start > max_sec:
                # trailing audio past the last cue would blow the budget: stop pad_sec after it
                end = _tail_end(last)
        chunks.append({"lo": lo, "hi": hi, "start": start, "end": end})
    return chunks


def decode_command(
    media_path: Path,
    out: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
    audio_filter: str | None = None,
) -> list[str]:
    """The ffmpeg argv :func:`decode_to_wav` runs.

    ``mono`` forces one channel (``-ac 1``). Otherwise the output is capped at
    stereo (:data:`STEREO_CAP_FILTER`): left alone, ffmpeg keeps the source
    layout, so a 5.1 / 7.1 track (most TV and film rips) would reach the
    stereo-only separator with 6 or 8 channels.
    """
    filters = [audio_filter] if audio_filter else []
    channels: list[str] = ["-ac", "1"] if mono else []
    if not mono:
        filters.append(STEREO_CAP_FILTER)
    af = ["-af", ",".join(filters)] if filters else []
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        str(media_path),
        *af,
        *channels,
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(out),
    ]


def decode_to_wav(
    media_path: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
    audio_filter: str | None = None,
) -> Path:
    """Decode media to a temp WAV via ffmpeg; caller is responsible for deletion.

    Default: 16k mono for VAD/ASR. For separation, use sample_rate=44100, mono=False
    (full-band, at most stereo: multichannel sources are downmixed, see
    :func:`decode_command`). ``audio_filter`` inserts an ``-af`` stage (e.g. loudnorm).
    """
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxweave_")
    os.close(fd)
    out = Path(path)
    try:
        subprocess.run(
            decode_command(
                media_path,
                out,
                sample_rate=sample_rate,
                mono=mono,
                audio_filter=audio_filter,
            ),
            check=True,
            timeout=FFMPEG_TIMEOUT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except BaseException as e:
        out.unlink(missing_ok=True)  # never leak the mkstemp wav, whatever went wrong
        if isinstance(e, subprocess.TimeoutExpired):
            raise RuntimeError(
                f"ffmpeg timed out after {FFMPEG_TIMEOUT:g}s decoding {media_path.name}; "
                f"raise VOXWEAVE_FFMPEG_TIMEOUT for very long media: {_stderr_tail(e.stderr)}"
            ) from e
        if isinstance(e, subprocess.CalledProcessError):
            raise RuntimeError(
                f"ffmpeg failed to decode {media_path.name}: {_stderr_tail(e.stderr)}"
            ) from e
        if isinstance(e, OSError):  # FileNotFoundError: no ffmpeg executable on PATH
            raise RuntimeError(
                "ffmpeg not found; install ffmpeg and make sure it is on your PATH"
            ) from e
        raise
    return out


def _stderr_tail(err: bytes | str | None) -> str:
    """Last few lines of captured ffmpeg stderr (where the actual error is), for messages."""
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    lines = (err or "").strip().splitlines()[-_FFMPEG_STDERR_LINES:]
    return "\n".join(lines) or "(no stderr)"


_silero_model = (
    None  # silero VAD singleton — loaded once per process (see _get_silero_vad)
)


def _get_silero_vad():
    """Return the process-wide silero VAD model, loading it on first use.

    A pipeline run calls the VAD several times (chunking pass, fine pass for song
    excision, snap passes); reloading the model each time costs a needless
    torch.hub-style init per call. Single-threaded by design — no locking.
    """
    global _silero_model
    if _silero_model is None:
        from silero_vad import load_silero_vad

        _silero_model = load_silero_vad()
    return _silero_model


def release_silero_vad() -> None:
    """Drop the cached silero VAD model; the next VAD call reloads it."""
    global _silero_model
    _silero_model = None


def vad_speech_segments(
    wav_path: Path, *, threshold: float = 0.5, min_silence_ms: int | None = None
) -> list[dict]:
    """silero VAD → speech segments [{start, end}] in seconds.

    threshold=0.5 is the silero default, used for chunking. Lowering to ~0.25 catches
    weakly voiced speech (e.g. secondary speaker attenuated by separation) but increases
    false positives on loud BGM, so only lower in specific scenarios.

    min_silence_ms defaults to VAD_MIN_SILENCE_MS (300ms, anti-overchop for chunking).
    Pass a smaller value (e.g. 100ms) for a fine pass that surfaces brief intra-segment
    silences — used by song excision to snap cut points into real silence.
    """
    import torch
    from silero_vad import get_speech_timestamps

    model = _get_silero_vad()
    # soundfile bypasses torchaudio>=2.9's torchcodec requirement
    data, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"expected {SAMPLE_RATE} Hz wav, got {sr!r} Hz — run decode_to_wav first"
        )
    wav = torch.from_numpy(data)
    return get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        threshold=threshold,
        min_silence_duration_ms=(
            VAD_MIN_SILENCE_MS if min_silence_ms is None else min_silence_ms
        ),
        speech_pad_ms=100,
    )


def silence_gaps(
    segments: list[dict], *, audio_end: float | None = None
) -> list[tuple[float, float]]:
    """Complement of speech segments: silence intervals [(start, end)] including file edges.

    Feed with a fine VAD pass (small min_silence_ms) to expose brief pauses; song excision
    snaps its cut points into these so dialogue words are never bisected. Pure logic.
    """
    gaps: list[tuple[float, float]] = []
    prev = 0.0
    for seg in segments:
        if seg["start"] > prev:
            gaps.append((prev, seg["start"]))
        prev = max(prev, seg["end"])
    if audio_end is not None and audio_end > prev:
        gaps.append((prev, audio_end))
    return gaps


def slice_wav(
    wav_path: Path,
    start: float,
    end: float,
    *,
    _sample_geometry_observer: Callable[[int, int, int, int], None] | None = None,
    _canonical_qwen_failures: bool = False,
) -> Path:
    """Slice the [start,end] segment from a 16k wav, write to a temp wav, return path (caller deletes).

    Reads only the requested frame range (seek + read) rather than decoding the whole file:
    an episode is sliced once per chunk, so a full read would re-materialize the entire
    waveform tens of times (a feature-length separated wav is multiple GB as float32).
    Frame arithmetic and clamping mirror the previous full-read-then-slice exactly, so the
    output is sample-identical.
    """

    def classify(exc: BaseException, detail_code: str) -> None:
        if not _canonical_qwen_failures:
            return
        try:
            if not isinstance(getattr(exc, "failure", None), CanonicalFailure):
                setattr(
                    exc,
                    "failure",
                    CanonicalFailure(
                        "qwen-window-operation-failed",
                        "qwen-window",
                        detail_code,
                    ),
                )
        except Exception:
            pass

    try:
        f = sf.SoundFile(str(wav_path))
    except Exception as exc:
        classify(exc, "sample-open")
        raise
    try:
        sr = f.samplerate
        frames = len(f)
        try:
            a = min(max(0, int(start * sr)), frames)
        except (ArithmeticError, TypeError, ValueError) as exc:
            classify(exc, "sample-start-index")
            raise
        try:
            b = min(frames, int(end * sr))
        except (ArithmeticError, TypeError, ValueError) as exc:
            classify(exc, "sample-end-index")
            raise
        if _sample_geometry_observer is not None:
            _sample_geometry_observer(a, b, sr, frames)
        try:
            f.seek(a)
            data = f.read(max(0, b - a), dtype="float32")
        except Exception as exc:
            classify(exc, "sample-seek-read")
            raise
    finally:
        f.close()
    try:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxweave_chunk_")
        os.close(fd)
    except Exception as exc:
        classify(exc, "sample-temp-create")
        raise
    out = Path(path)
    try:
        sf.write(str(out), data, sr)
    except Exception as exc:
        classify(exc, "sample-write")
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return out
