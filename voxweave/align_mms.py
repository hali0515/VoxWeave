"""MMS-300m forced alignment via ctc-forced-aligner (ONNX) — default for ja/CJK.

MMS + uroman romanization gives zero OOV, fixing wav2vec2-xlsr's
OOV-wildcard-without-anchor on rare kanji (cue-tail collapse / drift). Same
model as whisperx --align_backend ctc. Must run full-audio rather than per-cue:
the ONNX/cython path accumulates heap corruption on repeated small calls
(crashes at ~180-226 calls). Movie-length audio is DP-chunked at silence
anchors via ``align_common._dp_chunked_pass``.
"""

from __future__ import annotations

import logging
import os
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

from voxweave import config
from voxweave.align_common import (
    _distribute_units,
    _dp_chunked_pass,
    _load_mono,
    _strip_trailing_punct,
    mute_spans_in_wav,
)
from voxweave.runtime import _empty_cache, _hf_download, _require, get_device

log = logging.getLogger("voxweave")

# ONNX singleton, cleared on release().
CtcMms = namedtuple("CtcMms", "session tokenizer")
_mms = None  # CtcMms
# config [align] aliases that route to align_text_mms instead of wav2vec2
_MMS_NAMES = {"mms", "mms_fa", "ctc", "ctc-forced-aligner", "mms-300m"}
MMS_SR = 16000  # ctc-forced-aligner requires fixed 16k
# VOXWEAVE_MMS_MODEL (explicit local path) wins if it exists; otherwise pulled from HF -> config.ALIGN_CACHE.
MMS_MODEL = os.path.expanduser(os.environ.get("VOXWEAVE_MMS_MODEL", ""))
MMS_REPO = os.environ.get("VOXWEAVE_MMS_REPO", "deskpai/ctc_forced_aligner")
MMS_REPO_FILE = os.environ.get(
    "VOXWEAVE_MMS_REPO_FILE", "04ac86b67129634da93aea76e0147ef3.onnx"
)


def _resolve_mms_onnx() -> str:
    """Return MMS-300m onnx path: explicit VOXWEAVE_MMS_MODEL wins; otherwise auto-downloads from HF -> config.ALIGN_CACHE."""
    if MMS_MODEL and os.path.exists(MMS_MODEL):
        return MMS_MODEL
    return _hf_download(MMS_REPO, MMS_REPO_FILE, cache_dir=config.ALIGN_CACHE)


def _mms_providers(dev: str) -> list[str]:
    """ONNX execution providers for the MMS aligner, by resolved device.

    CUDA build: GPU provider first (onnxruntime-gpu — the CPU onnxruntime is dropped on Linux via
    [tool.uv] so it can't shadow it), CPU fallback. Everything else — including macOS/MPS — runs on
    CPU. CoreML is deliberately NOT used on macOS: its EP initializes a Metal context, and per-chunk
    we already run MLX ASR on Metal in the same process, then MMS alignment; two independent Metal
    backends coexisting hard-segfaults (signal 11 at the first MMS run). Plain CPU onnxruntime never
    touches Metal, so it coexists with MLX safely — at the cost of CPU-bound alignment (a chunk's
    MMS pass is still only a few seconds on Apple Silicon).
    """
    if dev.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_mms_aligner() -> CtcMms:
    """Lazy-load MMS-300m ONNX session + tokenizer singleton, cleared on release().

    Same model as whisperx --align_backend ctc. Execution provider from the resolved device
    (see _mms_providers): CUDA on the [cuda] build, CPU everywhere else (incl. macOS/MPS).
    """
    global _mms
    if _mms is None:
        try:
            import onnxruntime as ort
            from ctc_forced_aligner import Tokenizer
        except ModuleNotFoundError as e:
            raise _require(e.name or "ctc_forced_aligner") from e
        onnx_path = _resolve_mms_onnx()
        providers = _mms_providers(get_device())
        so = ort.SessionOptions()
        so.log_severity_level = 3  # suppress CUDA Memcpy WARNINGs (advisory only)
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        _mms = CtcMms(sess, Tokenizer())
        log.info("loaded MMS CTC aligner (mms-300m) providers=%s", sess.get_providers())
    return _mms


def _is_mms_name(model_name: str | None) -> bool:
    """True if an [align] alias routes to MMS (align_text_mms) instead of wav2vec2."""
    return bool(model_name) and model_name.strip().lower() in _MMS_NAMES


def uses_mms(iso: str) -> bool:
    """True if the configured aligner for this iso routes to MMS (mms/ctc alias)."""
    return _is_mms_name(config.align_model_for(iso))


def _read_wav_16k(wav_path: Path):
    """Read audio as 16k mono float32 numpy array (MMS_SR=16k required by ctc-forced-aligner)."""
    return _load_mono(wav_path, MMS_SR, as_numpy=True)


def _mms_emit_units(wav, text: str, iso: str) -> list[dict]:
    """16k numpy waveform + text -> flat units [{text,start,end}], times relative to wav start.

    MMS-300m + uroman -> zero OOV, fixing xlsr's wildcard-without-anchor on rare kanji (cue-tail
    collapse / drift). No-space langs: per-char units with punctuation filtered out (strips-punctuation
    invariant). Spaced langs: word-level units with trailing punctuation stripped.
    """
    import numpy as np
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        postprocess_results,
        preprocess_text,
    )

    from voxweave.lang import to_iso3
    from voxweave.realign import NO_SPACE_LANGS

    text = (text or "").strip()
    if not any(c.isalnum() for c in text):
        return []  # pure punctuation (e.g. "..."): uroman produces nothing, get_spans would assert
    if wav.shape[0] < 400:  # generate_emissions minimum input length
        wav = np.pad(wav, (0, 400 - wav.shape[0]))
    mm = _get_mms_aligner()
    iso3 = to_iso3(iso)
    emissions, stride = generate_emissions(
        mm.session, wav.astype(np.float32), batch_size=config.conf_batch("mms")
    )
    toks_starred, text_starred = preprocess_text(text, romanize=True, language=iso3)
    segments, scores, blank = get_alignments(emissions, toks_starred, mm.tokenizer)
    spans = get_spans(toks_starred, segments, blank)
    word_ts = postprocess_results(text_starred, spans, stride, scores)

    nospace = iso in NO_SPACE_LANGS
    units: list[dict] = []
    for w in word_ts:
        t = (w.get("text") or "").strip()
        if nospace:
            if not any(
                c.isalnum() for c in t
            ):  # drop space/punctuation units (strip-punctuation invariant)
                continue
        else:
            t = _strip_trailing_punct(t)
            if not t:
                continue
        units.append(
            {
                "text": t,
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
            }
        )
    return units


def align_text_mms(wav_path: Path, text: str, iso: str) -> list[dict]:
    """Per-chunk MMS alignment -> flat units (0-based). chunk <=120s, single generate_emissions call.

    DO NOT call with per-cue clips: repeated small ONNX inputs accumulate heap corruption and crash
    (munmap/glibc malloc assert; empirically at ~180-226 calls; crash point varies = cumulative,
    not input-specific). The align subcommand uses align_blocks_full_mms (single full-audio pass) instead.
    Exceptions propagate; align_text catches and falls back to Qwen.
    """
    units = _mms_emit_units(_read_wav_16k(wav_path), text, iso)
    _empty_cache()
    return units


def align_blocks_full_mms(
    wav_path: Path,
    texts: list[str],
    iso: str,
    bounds: Sequence[tuple[float, float] | None] | None = None,
    crop_to_envelope: bool = False,
    mute_spans: Sequence[tuple[float, float]] | None = None,
) -> list[list[dict]]:
    """Full-audio single-pass MMS alignment (equivalent to whisperx align_ctc).

    Concatenates all block texts, runs one generate_emissions pass over the entire audio
    (<star> between segments absorbs untranscribed sections like skipped songs), then slices
    flat units back to each block by alnum character count.

    Must be full-audio rather than per-cue: ctc-forced-aligner's ONNX/cython path accumulates
    heap corruption on repeated small calls (crashes at ~180-226 calls with munmap_chunk / glibc
    malloc assert; crash point varies = cumulative, not input-specific). whisperx avoids this by
    running full-audio in one pass; we do the same.
    Units are absolute timestamps relative to the full wav.

    generate_emissions windows the encoder internally, but get_alignments still builds one
    full-length O(T*L) trellis — the same movie-length wall as the wav2vec2 path. Past
    CTC_MAX_DP_FRAMES (MMS-300m is a wav2vec2 backbone: same 320x downsample at 16k, so the
    budget applies unchanged) the audio is DP-chunked at silence anchors via cue `bounds`
    (see _dp_chunked_pass). The handful of resulting large ONNX calls stays far below the
    ~180-call heap-corruption regime.
    """
    wav = _read_wav_16k(wav_path)
    if mute_spans:
        # excised song intervals: no transcript belongs there by construction, so zeroing
        # them only removes the acoustic bait that smears neighbouring sentences (mid-file
        # songs survive the envelope crop). Transcribe-path only (fresh spans of this run).
        wav = mute_spans_in_wav(wav, MMS_SR, mute_spans)
    norm = [(t or "").strip() for t in texts]

    # offset_s is part of the _dp_chunked_pass pass_fn contract (used by the CTC
    # path's emission masking); MMS does not mask yet, pending ja truth validation.
    def _pass(w, sub: list[str], offset_s: float = 0.0) -> list[list[dict]]:
        full = " ".join(t for t in sub if t)
        flat = _mms_emit_units(w, full, iso)
        _empty_cache()
        return _distribute_units(flat, sub, iso)

    return _dp_chunked_pass(
        wav, MMS_SR, norm, bounds, _pass, "MMS", crop_to_envelope=crop_to_envelope
    )


def release_mms() -> None:
    """Drop the MMS ONNX singleton (backend.release() calls this)."""
    global _mms
    _mms = None
