from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf

SAMPLE_RATE = 16000
# Raised from silero default 100ms to 300ms: 200ms chops natural mid-sentence pauses.
VAD_MIN_SILENCE_MS = int(os.environ.get("VOXWEAVE_VAD_MIN_SILENCE_MS", "300"))


def pack_speech_segments(segments: list[dict], max_sec: float) -> list[dict]:
    """Bin-pack silero speech segments [{start,end}] into chunks of <= max_sec, cut at silence boundaries.

    Returns [{start, end, offset}] (offset == start, for timestamp shifting).
    Single segments longer than max_sec are hard-cut (no silence to snap to; word cuts tolerated).
    """
    if not segments:
        return []
    chunks: list[dict] = []

    def emit(start: float, end: float) -> None:
        chunks.append({"start": start, "end": end, "offset": start})

    def open_block(start, end):
        # Returns (None, None) after hard-cutting an overlong segment into slices.
        if end - start > max_sec:
            t = start
            while end - t > max_sec:
                emit(t, t + max_sec)
                t += max_sec
            emit(t, end)
            return None, None
        return start, end

    cur_start, cur_end = open_block(segments[0]["start"], segments[0]["end"])
    for seg in segments[1:]:
        if cur_start is None:
            cur_start, cur_end = open_block(seg["start"], seg["end"])
        elif seg["end"] - cur_start <= max_sec:
            cur_end = seg[
                "end"
            ]  # still within budget; merge into current chunk (including intervening silence)
        else:
            emit(cur_start, cur_end)  # type: ignore[arg-type]  # close at silence boundary
            cur_start, cur_end = open_block(seg["start"], seg["end"])
    if cur_start is not None:
        emit(cur_start, cur_end)  # type: ignore[arg-type]  # cur_end is always assigned together with cur_start
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

    cuts: list[int] = []  # split AFTER cue index c
    i = 0
    while True:
        cstart = _first_start(i)
        rem_end = _last_end(i)
        if cstart is None or rem_end is None or rem_end - cstart <= max_sec:
            break  # remaining cues fit in one final chunk
        last_any: int | None = None
        last_gap: int | None = None
        for k in range(i, n - 1):
            ek = _end(k)
            if ek is None:
                continue
            if ek - cstart > max_sec:
                break
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
        start = (
            _split_time(lo - 1)
            if lo > 0
            else max(0.0, (_first_start(lo) or 0.0) - pad_sec)
        )
        if hi < n:
            end = _split_time(hi - 1)
        else:
            end = (
                audio_end if audio_end is not None else (_last_end(lo) or 0.0) + pad_sec
            )
        chunks.append({"lo": lo, "hi": hi, "start": start, "end": end})
    return chunks


def decode_to_wav(
    media_path: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
    audio_filter: str | None = None,
) -> Path:
    """Decode media to a temp WAV via ffmpeg; caller is responsible for deletion.

    Default: 16k mono for VAD/ASR. For separation, use sample_rate=44100, mono=False
    (full-band stereo). ``audio_filter`` inserts an ``-af`` stage (e.g. loudnorm).
    """
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxweave_")
    os.close(fd)
    out = Path(path)
    ac = ["-ac", "1"] if mono else []
    af = ["-af", audio_filter] if audio_filter else []
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(media_path),
            *af,
            *ac,
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def vad_speech_segments(wav_path: Path, *, threshold: float = 0.5) -> list[dict]:
    """silero VAD → speech segments [{start, end}] in seconds.

    threshold=0.5 is the silero default, used for chunking. Lowering to ~0.25 catches
    weakly voiced speech (e.g. secondary speaker attenuated by separation) but increases
    false positives on loud BGM, so only lower in specific scenarios.
    """
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    # soundfile bypasses torchaudio>=2.9's torchcodec requirement
    data, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == SAMPLE_RATE, (
        f"expected {SAMPLE_RATE} Hz wav, got {sr!r} Hz — run decode_to_wav first"
    )
    wav = torch.from_numpy(data)
    return get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        threshold=threshold,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=100,
    )


def slice_wav(wav_path: Path, start: float, end: float) -> Path:
    """Slice the [start,end] segment from a 16k wav, write to a temp wav, return path (caller deletes)."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    a = max(0, int(start * sr))
    b = min(len(data), int(end * sr))
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxweave_chunk_")
    os.close(fd)
    out = Path(path)
    sf.write(str(out), data[a:b], sr)
    return out
