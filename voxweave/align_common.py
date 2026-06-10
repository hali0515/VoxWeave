"""Logic shared by the CTC (wav2vec2) and MMS forced-alignment paths.

Pure unit post-processing (punctuation strip, span interpolation, per-block
distribution), audio loading, VAD emission masking and the silence-anchored
DP-chunking driver that both full-pass aligners run under. Nothing here loads
a model; ``align_ctc`` / ``align_mms`` own their singletons.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from voxweave import config

log = logging.getLogger("voxweave")

_CTC_STRIDE = 320  # wav2vec2 @16k downsamples 320x -> 50fps (20ms/frame)
# Single global forced-align DP is O(T*L); cap audio length so it stays in memory. Movies past
# this are auto-split at silence anchors (plan_dp_chunks) when cue timestamps are available.
# Sourced via config (env VOXWEAVE_CTC_MAX_DP_FRAMES > conf ctc_max_dp_frames > 90000≈30min).
CTC_MAX_DP_FRAMES = config.conf_ctc_max_dp_frames()
# Per-chunk DP budget as a fraction of CTC_MAX_DP_FRAMES: leaves headroom so an off-by-a-bit
# silence anchor never pushes a chunk's O(T*L) trellis past the memory cap.
CTC_DP_CHUNK_FRAC = float(os.environ.get("VOXWEAVE_CTC_DP_CHUNK_FRAC", "0.8"))


def _strip_trailing_punct(word: str) -> str:
    """Strip trailing punctuation (timing still covers the full word). Returns original if all punctuation."""
    i = len(word)
    while i > 0 and not word[i - 1].isalnum():
        i -= 1
    return word[:i] or word


def interp_missing(units: list[dict]) -> list[dict]:
    """Fill zero-length spans (end<=start) by linear interpolation from neighboring valid spans. Never drops a unit.

    Last-resort fallback; OOV chars already get spans via wildcard so this rarely triggers.
    Two anchors -> linear by index; one side only -> ffill/bfill; no anchors -> as-is. Pure function.
    """
    out = [dict(u) for u in units]
    valid = [i for i, u in enumerate(out) if u["end"] > u["start"]]
    if not valid:
        return out
    for i, u in enumerate(out):
        if u["end"] > u["start"]:
            continue
        prev = max((j for j in valid if j < i), default=None)
        nxt = min((j for j in valid if j > i), default=None)
        if prev is not None and nxt is not None:
            lo, hi = out[prev]["end"], max(out[nxt]["start"], out[prev]["end"])
            t = lo + (hi - lo) * (i - prev) / (nxt - prev)
        elif prev is not None:
            t = out[prev]["end"]
        elif nxt is not None:
            t = out[nxt]["start"]
        else:  # unreachable: valid is non-empty, so at least one side exists
            continue
        u["start"] = u["end"] = round(t, 3)
    return out


def _load_mono(wav_path: Path, target_sr: int, *, as_numpy: bool = False):
    """Read audio as mono float32 at target_sr. Returns a torch tensor (default) or numpy (as_numpy).

    Shared by the CTC aligners (torch tensor at al.sr) and MMS (_read_wav_16k, numpy at 16k);
    keeps the read+downmix+conditional-resample sequence in one place.
    """
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    data, fsr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # mono 1D (numpy)
    if as_numpy:
        if fsr != target_sr:
            mono = AF.resample(torch.from_numpy(mono), fsr, target_sr).numpy()
        return mono
    wav = torch.from_numpy(mono)
    if fsr != target_sr:
        wav = AF.resample(wav, fsr, target_sr)
    return wav


def _distribute_units(flat: list[dict], texts: list[str], iso: str) -> list[list[dict]]:
    """Slice full-audio flat units back to per-block lists.

    No-space langs: alnum character count per block (one unit per char, punctuation skipped).
    Spaced langs: word count (len(split())). Consistent with whisperx reformat_vtt. Pure logic.
    """
    from voxweave.realign import NO_SPACE_LANGS

    nospace = iso in NO_SPACE_LANGS
    out: list[list[dict]] = []
    cur = 0
    for t in texts:
        t = (t or "").strip()
        n = sum(1 for c in t if c.isalnum()) if nospace else len(t.split())
        out.append(flat[cur : cur + n] if n else [])
        cur += n
    return out


# Non-speech emission masking (stable-ts analogue): outside VAD speech spans, non-blank
# log-probs are penalized so the global DP cannot park words inside music/silence — the
# residual "Pattern A" tail slide on sparse-dialogue movies. Soft penalty (not -inf): a
# VAD false negative can still be overridden by strong acoustic evidence. Spans are
# dilated so VAD edge jitter never clips a word onset/coda. The star wildcard column is
# built AFTER masking (max over non-blank), so it inherits the penalty and cannot claim
# silence either; blanks fill the gaps.
#
# OPT-IN (VOXWEAVE_VAD_EMISSION_MASK=1), default off: A/B on real media showed the harm
# case — transcribed vocalizations that VAD misses (humming "Pa-pa-pa") are real sounds
# at real positions, and masking relocates them seconds away. The mechanism only wins
# when out-of-VAD words are alignment ERRORS (the sparse-dialogue movie profile); enable
# it for that profile, and validate against ground truth before trusting a new corpus.
_VAD_MASK_PENALTY = 4.0
_VAD_MASK_DILATE_S = 0.12


def _mask_emissions_outside_speech(logp, speech_spans, total_samples, sr, blank):
    """Penalize non-blank columns of [T,V] log-probs in frames outside speech spans.

    ``speech_spans`` are seconds on the same timeline as the waveform that produced
    ``logp`` (caller shifts chunk offsets). Empty/None spans return ``logp`` unchanged —
    no VAD information must never mean "everything is silence"."""
    import torch

    if not speech_spans:
        return logp
    t_frames = logp.shape[0]
    spf = total_samples / t_frames / sr  # seconds per frame (~0.02)
    keep = torch.zeros(t_frames, dtype=torch.bool)
    for s, e in speech_spans:
        a = max(0, int((s - _VAD_MASK_DILATE_S) / spf))
        b = min(t_frames, int((e + _VAD_MASK_DILATE_S) / spf) + 1)
        if b > a:
            keep[a:b] = True
    if bool(keep.all()):
        return logp
    pen_row = torch.full((logp.shape[1],), _VAD_MASK_PENALTY, device=logp.device)
    pen_row[blank] = 0.0
    masked = logp.clone()
    idx = (~keep).to(logp.device)
    masked[idx] = masked[idx] - pen_row
    return masked


def _dp_chunked_pass(
    wav,
    sr: int,
    norm: list[str],
    bounds: Sequence[tuple[float, float] | None] | None,
    pass_fn,
    label: str,
) -> list[list[dict]]:
    """Run `pass_fn(wav_slice, texts, offset_s) -> per-block units` under the global-DP memory budget.

    Shared by the wav2vec2 and MMS full-pass aligners: both end in a single forced-align
    trellis that is O(T*L) and overflows on movie-length audio. Within budget the whole wav
    goes through one pass. Past CTC_MAX_DP_FRAMES, cue `bounds` (per-cue (start,end), aligned
    with `norm`) are used as silence anchors to split the audio (plan_dp_chunks): each chunk
    re-runs the full pass over its own crop (within-chunk routing-free, so drift-immunity
    holds) and units are offset back to absolute time. Boundaries land in inter-cue silence,
    so no word crosses them. Without bounds an over-budget file is rejected (raise
    VOXWEAVE_CTC_MAX_DP_FRAMES to force a single pass). `wav` may be a torch tensor or a
    numpy array; only 1D slicing and shape[-1] are used.
    """
    from voxweave.chunking import plan_dp_chunks
    from voxweave.timestamps import shift_units

    frames = wav.shape[-1] / _CTC_STRIDE
    if frames <= CTC_MAX_DP_FRAMES:
        return pass_fn(wav, norm, 0.0)

    if not bounds or len(bounds) != len(norm):
        raise RuntimeError(
            f"audio ~{frames / 3000:.0f}min exceeds single-pass CTC DP budget "
            f"(~{CTC_MAX_DP_FRAMES / 3000:.0f}min) and cue timestamps are unavailable for "
            f"silence-anchored DP-chunking (raise VOXWEAVE_CTC_MAX_DP_FRAMES to override)"
        )

    total_sec = wav.shape[-1] / sr
    budget_sec = CTC_MAX_DP_FRAMES * _CTC_STRIDE / sr * CTC_DP_CHUNK_FRAC
    plans = plan_dp_chunks(bounds, max_sec=budget_sec, audio_end=total_sec)
    log.info(
        "%s DP-chunking %.0fmin audio into %d silence-anchored chunks (budget ~%.0fmin)",
        label,
        total_sec / 60,
        len(plans),
        budget_sec / 60,
    )
    out: list[list[dict]] = []
    for p in plans:
        a = max(0, int(p["start"] * sr))
        b = min(wav.shape[-1], int(p["end"] * sr))
        offset = a / sr
        sub = pass_fn(wav[a:b], norm[p["lo"] : p["hi"]], offset)
        out.extend(shift_units(u, offset) for u in sub)
    return out
