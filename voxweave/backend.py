from __future__ import annotations

import inspect
import logging
import os
import re
import tempfile
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path

from voxweave import config
from voxweave.align_common import interp_missing as interp_missing  # re-export
from voxweave.align_ctc import align_blocks_full_ctc, align_text_ctc, release_ctc
from voxweave.align_mms import (
    _is_mms_name,
    align_blocks_full_mms,
    align_text_mms,
    release_mms,
    uses_mms as uses_mms,  # re-export (pipeline-facing)
)
from voxweave.runtime import (
    _MISSING_WHISPER,
    _empty_cache,
    _hf_download,
    _hf_snapshot,
    _load_yaml,
    _model_dtype,
    _require,
    _use_mlx,
    get_device,
)

log = logging.getLogger("voxweave")

# Heavy deps (torch/qwen_asr/roformer) are lazy-imported so importing voxweave doesn't pull in torch.
# Dynamic loading: separation and ASR/alignment are loaded in separate phases so
# peak VRAM = max(the two), not sum.
ASR_MODEL = os.environ.get("VOXWEAVE_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ALIGNER_MODEL = os.environ.get(
    "VOXWEAVE_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"
)
# --model short name -> HF repo id; case-insensitive, tolerates missing org prefix
_ASR_ALIASES = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
}


def resolve_asr_model(name: str | None) -> str:
    """--model value -> HF repo id. Empty -> default; contains '/' -> pass-through; otherwise check alias table, fall back to prepending 'Qwen/'."""
    if not name or not name.strip():
        return ASR_MODEL
    v = name.strip()
    if "/" in v:
        return v
    return _ASR_ALIASES.get(v.lower(), f"Qwen/{v}")


# If --model matches one of these size strings -> whisper-hybrid path (whisper text + Qwen3-ForcedAligner units);
# otherwise -> qwen path.
_WHISPER_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
}
_WHISPER_ALIASES = {"whisper": "large-v3-turbo", "turbo": "large-v3-turbo"}
# Fusion aliases: whisper produces accurate text + Qwen provides punctuation positions, merged on a shared timeline.
# Sub-models resolve via config.conf_fusion_whisper/qwen (env > conf > default).
# Whisper defaults to large-v3-turbo; punctuation path uses 1.7B (0.6B doesn't emit punctuation).
_FUSION_ALIASES = {"fusion", "fuse", "hybrid", "hybrid+"}


def _select_engine(name: str | None) -> tuple[str, str]:
    """--model value -> (engine, resolved model id).

    Empty -> ('qwen', ASR_MODEL). Fusion aliases -> ('fusion', ''). Whisper size/distil- prefix -> ('whisper', size).
    Otherwise -> ('qwen', repo id).
    """
    if not name or not name.strip():
        return "qwen", ASR_MODEL
    key = name.strip().lower()
    if key in _FUSION_ALIASES:
        return "fusion", ""
    if key in _WHISPER_ALIASES:
        return "whisper", _WHISPER_ALIASES[key]
    if key in _WHISPER_MODELS or key.startswith("distil-"):
        return "whisper", key
    return "qwen", resolve_asr_model(name)


# Separator ckpt + companion yaml default to ~/.cache/voxweave/; model class frozen in voxweave.vendor.
MODEL_DIR = Path(
    os.environ.get("VOXWEAVE_MODEL_DIR", Path.home() / ".cache" / "voxweave")
)
# One-time migration from pre-rename ~/.cache/qsub; if it fails weights re-download from HF.
_LEGACY_MODEL_DIR = Path.home() / ".cache" / "qsub"
if (
    not os.environ.get("VOXWEAVE_MODEL_DIR")
    and not MODEL_DIR.exists()
    and _LEGACY_MODEL_DIR.exists()
):
    try:
        _LEGACY_MODEL_DIR.rename(MODEL_DIR)
        log.info("migrated model cache %s -> %s", _LEGACY_MODEL_DIR, MODEL_DIR)
    except OSError as e:
        log.warning("could not migrate model cache %s (%r)", _LEGACY_MODEL_DIR, e)
SEPARATOR_CKPT = os.environ.get(
    "VOXWEAVE_SEPARATOR_CKPT", str(MODEL_DIR / "vocals_mel_band_roformer.ckpt")
)
SEPARATOR_CONFIG = os.environ.get(
    "VOXWEAVE_SEPARATOR_CONFIG", str(MODEL_DIR / "vocals_mel_band_roformer.yaml")
)
# Auto-download from HF if weights missing; downloaded to HF cache so subsequent runs hit cache.
SEPARATOR_REPO = os.environ.get(
    "VOXWEAVE_SEPARATOR_REPO", "KimberleyJSN/melbandroformer"
)
SEPARATOR_REPO_FILE = os.environ.get(
    "VOXWEAVE_SEPARATOR_REPO_FILE", "MelBandRoformer.ckpt"
)
# Companion yaml is bundled (matches frozen vendor architecture); used as fallback for SEPARATOR_CONFIG.
_BUNDLED_SEPARATOR_CONFIG = (
    Path(__file__).parent / "vendor" / "vocals_mel_band_roformer.yaml"
)


# Empty -> float16 on cuda, int8 on cpu/mps (ctranslate2/faster-whisper is CUDA-or-CPU only).
WHISPER_COMPUTE = os.environ.get("VOXWEAVE_WHISPER_COMPUTE", "")
# A 120-second dense transcript can legitimately exceed qwen-asr's 512-token
# constructor default.  Raising only the ceiling does not change ordinary greedy
# decodes (generation still stops at EOS), but avoids silent tail truncation.
QWEN_MAX_NEW_TOKENS = int(os.environ.get("VOXWEAVE_QWEN_MAX_NEW_TOKENS", "1024"))

# ASR/alignment process-level singletons; call release() at end of episode.
# Separator is not kept resident (self-loads, self-releases).
_asr = None  # qwen_asr.Qwen3ASRModel
_asr_id = None  # currently loaded ASR repo id (reloaded on --model change)
# Standalone aligner for the align command (no ASR needed, so we skip the full Qwen3ASRModel stack).
_aligner = None  # qwen_asr.Qwen3ForcedAligner
_whisper = None  # faster_whisper.WhisperModel
_whisper_id = None  # currently loaded whisper size string


# ───────────────────────────── vocal separation (Mel-Band Roformer, self-load/self-release) ──────────────


def _strip_state_dict(sd: dict) -> dict:
    """Strip Lightning state_dict wrapper and 'model.' prefix."""
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if sd and all(k.startswith("model.") for k in sd):
        sd = {k[len("model.") :]: v for k, v in sd.items()}
    return sd


def _resolve_separator_files() -> tuple[Path, Path]:
    """Locate separator ckpt + yaml. Auto-downloads ckpt from HF if missing; falls back to bundled yaml if yaml is absent."""
    ckpt = Path(SEPARATOR_CKPT)
    if not ckpt.exists():
        log.info(
            "separator model missing, downloading from HF %s/%s (-> HF cache, subsequent runs will hit cache) ...",
            SEPARATOR_REPO,
            SEPARATOR_REPO_FILE,
        )
        try:
            ckpt = Path(
                _hf_download(
                    SEPARATOR_REPO, SEPARATOR_REPO_FILE, cache_dir=config.AUDIO_CACHE
                )
            )
        except RuntimeError:
            raise  # missing-dep errors are already friendly, re-raise as-is
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"separator model auto-download failed ({SEPARATOR_REPO}/{SEPARATOR_REPO_FILE}): {e!r} -- "
                f"check network / HF_TOKEN, or manually place weights at {SEPARATOR_CKPT} "
                f"(VOXWEAVE_SEPARATOR_CKPT|CONFIG / VOXWEAVE_SEPARATOR_REPO are configurable), "
                f"or use --no-separate"
            ) from e
    conf = Path(SEPARATOR_CONFIG)
    if not conf.exists():
        conf = _BUNDLED_SEPARATOR_CONFIG
    return ckpt, conf


def _load_separator():
    """Instantiate MelBandRoformer (not cached; caller del's it after separation to free VRAM).

    Uses the frozen copy from voxweave.vendor: the latest PyPI bs-roformer has diverged in
    architecture and can no longer load community ckpts. Extra yaml keys are filtered against
    __init__ signature so unknown fields don't crash.
    """
    try:
        import torch

        from voxweave.vendor.mel_band_roformer import MelBandRoformer
    except ModuleNotFoundError as e:
        raise _require(e.name or "torch") from e

    ckpt, conf = _resolve_separator_files()
    cfg = _load_yaml(conf)
    model_cfg = dict(cfg["model"])
    sig = inspect.signature(MelBandRoformer.__init__)
    kwargs = {k: v for k, v in model_cfg.items() if k in sig.parameters}
    dropped = set(model_cfg) - set(kwargs)
    if dropped:
        log.debug(
            "separator config: ignoring unrecognized MelBandRoformer keys %s",
            sorted(dropped),
        )
    model = MelBandRoformer(**kwargs)
    # weights_only=True: disables arbitrary-object deserialization (prevents pickle RCE)
    sd = _strip_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.load_state_dict(sd)
    dev = get_device()
    model.to(dev).eval()
    log.info("loaded separator ckpt=%s on %s", ckpt, dev)
    return model, cfg


def _demix(model, mix, cfg, progress=None, batch=None):
    """Chunked overlap-add inference: mix [ch, t] float32 -> vocals [ch, t].

    Hann window + >=2x overlap satisfies COLA; tail normalized by window sum. num_stems=1
    output may or may not have a stem dimension -- both shapes are handled.
    Windows are stacked `batch` at a time into one forward (default conf [batch].separate;
    1 unless configured -- batch=1 already saturates 8 GB-class GPUs, see config._BATCH_DEFAULTS).
    progress(done, total) called after each window if provided.
    """
    import torch

    audio = cfg.get("audio", {})
    inf = cfg.get("inference", {})
    chunk = int(audio.get("chunk_size", 131584))
    overlap = int(inf.get("num_overlap", 4))
    step = max(1, chunk // overlap)
    ch, total = mix.shape

    window = torch.hann_window(chunk)
    result = torch.zeros(ch, total)
    weight = torch.zeros(total)
    dev = next(model.parameters()).device
    starts = list(range(0, total, step))
    nwin = len(starts)
    bs = batch if batch is not None else config.conf_batch("separate")
    with torch.no_grad():
        for i in range(0, nwin, bs):
            grp = starts[i : i + bs]
            segs, lens = [], []
            for start in grp:
                seg = mix[:, start : start + chunk]
                n = seg.shape[1]
                lens.append(n)
                if n < chunk:  # pad final segment to full chunk size
                    seg = torch.nn.functional.pad(seg, (0, chunk - n))
                segs.append(seg)
            out = model(torch.stack(segs).to(dev))  # [B, (stems,) ch, t]
            if out.dim() == 4:  # [B, stems, ch, t] -> take first (vocals) stem
                out = out[:, 0]
            out = out.float().cpu()  # [B, ch, chunk]
            for j, (start, n) in enumerate(zip(grp, lens)):
                w = window[:n]
                result[:, start : start + n] += out[j, :, :n] * w
                weight[start : start + n] += w
                if progress is not None:
                    progress(i + j + 1, nwin)
    return result / weight.clamp_min(1e-8)


def separate_vocals(audio_path: Path, *, progress=None) -> Path:
    """Separate vocals locally; returns a FLAC temp file (caller deletes).

    Input must be full-band 44.1k stereo -- Roformer was trained at 44.1k; feeding 16k degrades quality badly.
    Model is loaded and freed within this function so it doesn't co-occupy VRAM with ASR/alignment.
    progress(done, total) called per demix window if provided.
    """
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ModuleNotFoundError as e:
        raise _require(e.name or "torch") from e

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)  # [t, ch]
    mix = torch.from_numpy(data.T.copy())  # [ch, t]
    if mix.shape[0] == 1:  # mono -> duplicate to stereo for the stereo model
        mix = mix.repeat(2, 1)

    model, cfg = _load_separator()
    try:
        vocals = _demix(model, mix, cfg, progress=progress)  # [ch, t]
    finally:
        del model
        _empty_cache()

    fd, dst = tempfile.mkstemp(suffix=".flac", prefix="voxweave_vocals_")
    os.close(fd)
    out = Path(dst)
    sf.write(str(out), np.asarray(vocals.T), sr, format="FLAC")  # [t, ch]
    return out


# ───────────────────────────── ASR + forced alignment (singletons, released via release()) ─────────────────


def _get_asr(asr_model: str | None = None):
    """Lazy-load Qwen3ASRModel singleton (ASR-only, no forced aligner attached).

    Timestamps come from align_text; on ja/en paths the Qwen aligner is never loaded, saving VRAM.
    dtype kwarg (not torch_dtype) reflects transformers 4.57.6 API.
    Reloads if the model repo changes.
    """
    if (
        _use_mlx()
    ):  # Apple Silicon: serve ASR from the native MLX Qwen3-ASR (see backend_mlx)
        from voxweave import backend_mlx

        return backend_mlx.get_asr(resolve_asr_model(asr_model))
    global _asr, _asr_id
    mid = resolve_asr_model(asr_model)
    if _asr is not None and _asr_id != mid:  # model changed -> release old one
        release()
    if _asr is None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as e:
            raise _require(e.name or "qwen_asr") from e

        dev = get_device()
        log.info("loading ASR=%s (text-only) on %s", mid, dev)
        # Use snapshot so model + processor both land in config.ASR_CACHE (see _hf_snapshot docstring).
        local = _hf_snapshot(mid, config.ASR_CACHE)
        load_kwargs = {"dtype": _model_dtype(dev), "device_map": dev}
        if "max_new_tokens" in inspect.signature(
            Qwen3ASRModel.from_pretrained
        ).parameters:
            load_kwargs["max_new_tokens"] = QWEN_MAX_NEW_TOKENS
        _asr = Qwen3ASRModel.from_pretrained(local, **load_kwargs)
        _asr_id = mid
        log.info("ASR ready")
    return _asr


def _resolve_align_lang(lang: str | None, source: str) -> str:
    """Return lang if it's in the aligner's supported set, else fall back to 'en' with a warning. source is for log wording only."""
    from voxweave.lang import is_supported

    if lang and is_supported(lang):
        return lang
    if lang:
        log.warning(
            "%s lang %r not in aligner's 11 supported languages, falling back to en for alignment",
            source,
            lang,
        )
    return "en"


# Qwen3-ASR context framing: the system-prompt slot treats a FRAMED term list as
# biasing metadata, while a bare list triggers "list dictation mode" and regresses
# WER below the empty baseline (TypeWhisper/typewhisper-mac#321, 184-run sweep:
# 87.5% -> 28.1% WER on dense technical audio; only "Technical terms:" /
# "Vocabulary:" / "Proper nouns:" framings survived). Prose context (sentence
# punctuation present) is genuine background knowledge and passes through, as
# does input that already carries one of the known-good framings.
_CONTEXT_FRAMINGS = ("technical terms:", "vocabulary:", "proper nouns:")
_CONTEXT_TERM_SPLIT_RE = re.compile(r"[,，\n]+")
_ASCII_SENTENCE_RE = re.compile(r"[.!?](?:\s+|$)")


def _looks_like_prose_context(text: str) -> bool:
    """Distinguish background prose from a term/version list conservatively."""
    if any(ch in "。！？" for ch in text):
        return True
    words = text.split()
    if _ASCII_SENTENCE_RE.search(text) and len(words) >= 4:
        return True
    return "," not in text and "，" not in text and "\n" not in text and len(words) >= 8


def _context_terms(text: str) -> list[str]:
    return [term.strip() for term in _CONTEXT_TERM_SPLIT_RE.split(text) if term.strip()]


def format_qwen_context(context: str | None) -> str | None:
    """Frame a bare term list as ``Proper nouns: <terms>.`` for the Qwen system slot.

    Already-framed lists and prose pass through unchanged; None/blank stays None.
    Whisper's ``initial_prompt`` is a different mechanism (transcript-prefix
    conditioning) and must NOT receive this framing.
    """
    s = (context or "").strip()
    if not s:
        return None
    low = s.lower()
    if any(low.startswith(p) for p in _CONTEXT_FRAMINGS):
        return s
    if _looks_like_prose_context(s):
        return s
    return f"Proper nouns: {', '.join(_context_terms(s))}."


def whisper_hotwords(context: str | None) -> str | None:
    """Extract an explicit term list for faster-whisper's hotword bias.

    Background prose remains only an ``initial_prompt``.  Bare lists and the
    same framed lists accepted by :func:`format_qwen_context` additionally use
    faster-whisper's purpose-built hotword channel.  No inferred vocabulary is
    injected here: every term comes from the user's context.
    """
    s = (context or "").strip()
    if not s:
        return None
    for line in s.splitlines():
        low = line.strip().lower()
        for framing in _CONTEXT_FRAMINGS:
            if low.startswith(framing):
                payload = line.strip()[len(framing) :].strip().rstrip(".")
                terms = _context_terms(payload)
                return ", ".join(terms) or None
    if _looks_like_prose_context(s):
        return None
    return ", ".join(_context_terms(s)) or None


_ASR_LOOP_END_PUNCT = set(".!?。！？")
_ASR_LOOP_MIN_CONTENT = 12
_ASR_LOOP_MIN_REPEATS = 4


def stabilize_asr_text(text: str) -> str:
    """Strip edges and collapse only high-confidence generation loops at EOF.

    ASR failure loops characteristically repeat the same long, punctuated span
    until the token limit.  Restricting cleanup to four or more *exact* terminal
    copies with at least 12 alphanumeric characters leaves stutters, emphasis,
    short refrains, and non-terminal repetition untouched.
    """
    from voxweave.lang import transcript_content_weight

    clean = (text or "").strip()
    n = len(clean)
    best = clean
    best_removed = 0
    for width in range(1, n // _ASR_LOOP_MIN_REPEATS + 1):
        unit = clean[n - width :]
        if not unit or unit[0].isspace() or unit[-1] not in _ASR_LOOP_END_PUNCT:
            continue
        if transcript_content_weight(unit) < _ASR_LOOP_MIN_CONTENT:
            continue
        end = n
        starts: list[int] = []
        while end >= width and clean[end - width : end] == unit:
            starts.append(end - width)
            end -= width
            while end > 0 and clean[end - 1].isspace():
                end -= 1
        if len(starts) < _ASR_LOOP_MIN_REPEATS:
            continue
        candidate = (clean[: starts[-1]] + unit).rstrip()
        removed = n - len(candidate)
        if removed > best_removed:
            best = candidate
            best_removed = removed
    return best


def _asr_only(
    engine: str,
    wav_path: Path,
    language: str | None,
    model_id: str,
    context: str | None,
) -> tuple[str | None, str, str]:
    """Transcribe only: return (effective language, punctuated text, align_lang).

    First pass of the two-pass peak strategy: alignment deferred to pass two after ASR is released.
    Explicit language always wins; auto-detected labels are reconciled with the
    transcript script before choosing the aligner.
    align_lang pre-computed here; falls back to 'en' for empty text (skipped in pass two anyway).
    """
    from voxweave.lang import reconcile_detected_language, to_iso_or

    if engine == "whisper":
        model = _get_whisper(model_id)
        lang_iso = to_iso_or(language, None)
        if (
            lang_iso == "yue"
        ):  # whisper has no Cantonese code; alignment still uses yue downstream
            lang_iso = "zh"
        segments, info = model.transcribe(
            str(wav_path),
            language=lang_iso,
            initial_prompt=context or None,
            hotwords=whisper_hotwords(context),
            condition_on_previous_text=False,  # prevents repetition hallucination
            vad_filter=False,  # VAD chunking already done upstream
            word_timestamps=False,  # hybrid uses Qwen for timestamps, not whisper
        )
        raw_text = "".join(s.text for s in segments)  # segments is a generator
        raw_det = info.language
        src = "whisper"
    else:  # qwen
        model = _get_asr(model_id)
        kwargs: dict = {"language": language or None, "return_time_stamps": False}
        if context:  # omit kwarg entirely when empty to preserve legacy behavior
            kwargs["context"] = format_qwen_context(context)
        r = model.transcribe(str(wav_path), **kwargs)[0]
        raw_det = r.language or None
        raw_text = r.text
        src = "ASR"
    text = stabilize_asr_text(raw_text)
    if len(text) < len(raw_text.strip()):
        log.warning(
            "%s removed a repeated generation tail (%d -> %d characters)",
            src,
            len(raw_text.strip()),
            len(text),
        )
    det = reconcile_detected_language(raw_det, text, override=language)
    if not language and raw_det and det and det.casefold() != raw_det.strip().casefold():
        log.info(
            "%s language %r reconciled to %r from transcript script",
            src,
            raw_det,
            det,
        )
    align_lang = _resolve_align_lang(det, src) if text.strip() else "en"
    return det, text, align_lang


def _transcribe_qwen_align(
    wav_path: Path,
    language: str | None,
    model_id: str,
    context: str | None,
) -> tuple[str | None, str, list[dict]]:
    """Qwen3-ASR single chunk: ASR then forced alignment via align_text.

    ja/en -> CTC (blank absorbs silence, prevents Qwen NAR from drifting weak tokens);
    zh/yue -> align_text falls back to Qwen3ForcedAligner internally.
    Full-episode batches use transcribe_chunks two-pass strategy to avoid co-resident ASR+aligner.
    """
    det, text, align_lang = _asr_only("qwen", wav_path, language, model_id, context)
    if not text.strip():
        _empty_cache()
        return det, "", []
    units = align_text(wav_path, text, align_lang)
    _empty_cache()  # alignment matrix grows with audio length; reclaim after each chunk to prevent VRAM creep
    return det, text, units


def _transcribe_whisper_align(
    wav_path: Path,
    language: str | None,
    model_id: str,
    context: str | None,
) -> tuple[str | None, str, list[dict]]:
    """Whisper text + Qwen3-ForcedAligner units, single chunk. Same (lang, text, units) contract as qwen path."""
    det, text, align_lang = _asr_only("whisper", wav_path, language, model_id, context)
    if not text.strip():
        return det, "", []
    units = align_text(wav_path, text, align_lang)
    return det, text, units


_FUSION_PUNCT = set("。、！？，,.!?")
_FUSION_MIN_CONTENT_AGREEMENT = 0.50


def _asr_content_agreement(left: str, right: str) -> float:
    """Case-insensitive alphanumeric agreement between two ASR hypotheses."""
    a = [ch.casefold() for ch in left if ch.isalnum()]
    b = [ch.casefold() for ch in right if ch.isalnum()]
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _fuse_chunk(
    w_res: tuple[str | None, str, list[dict]],
    q_res: tuple[str | None, str, list[dict]],
    language: str | None,
) -> tuple[str | None, str, list[dict]]:
    """Merge one chunk: whisper text + Qwen punctuation -> (lang, punctuated text, units).

    If whisper is empty, returns Qwen result entirely to avoid losing content.
    Units come from whisper; punctuation inserted by timestamp from Qwen. Pure logic, no GPU.
    """
    det_w, text_w, units_w = w_res
    if not text_w.strip():
        return q_res
    from voxweave.lang import to_iso_or
    from voxweave.realign import NO_SPACE_LANGS, fuse_punct_into_text, reinject_punct

    det_q, text_q, units_q = q_res
    # Punctuation transfer is an enhancement, never a reason to destroy a good
    # Whisper hypothesis.  With no Qwen punctuation/alignment, or when the two
    # decoders substantially disagree, there is no trustworthy content anchor;
    # retain Whisper's own punctuation verbatim.
    if (
        not text_q.strip()
        or not units_q
        or not any(ch in _FUSION_PUNCT for ch in text_q)
        or _asr_content_agreement(text_w, text_q) < _FUSION_MIN_CONTENT_AGREEMENT
    ):
        return det_w or det_q, text_w, units_w
    cand = language or det_w or det_q or "en"
    iso = to_iso_or(cand, "en")
    qwen_punct = reinject_punct(
        text_q, units_q, iso
    )  # Qwen units carry punctuation positions
    # Spaced langs: strip whisper's sparse punctuation and use Qwen's (word-level transplant is stable).
    # No-space langs: keep whisper's own punctuation (char-level time transplant drifts; see fuse_punct_into_text).
    fused = fuse_punct_into_text(
        text_w, units_w, qwen_punct, strip_existing=iso not in NO_SPACE_LANGS
    )
    return det_w or det_q, fused, units_w


def _transcribe_fusion(
    wav_path: Path,
    language: str | None,
    context: str | None,
) -> tuple[str | None, str, list[dict]]:
    """Dual-ASR fusion, single chunk (used by transcribe_align and tests): whisper + Qwen co-resident -> merge.

    Full-episode batches use transcribe_chunks three-pass strategy to avoid both models in VRAM simultaneously.
    """
    qid = resolve_asr_model(config.conf_fusion_qwen())
    w_res = _transcribe_whisper_align(
        wav_path, language, config.conf_fusion_whisper(), context
    )
    if not w_res[1].strip():  # whisper empty -> use Qwen only
        return _transcribe_qwen_align(wav_path, language, qid, context)
    q_res = _transcribe_qwen_align(wav_path, language, qid, context)
    return _fuse_chunk(w_res, q_res, language)


def chunk_pass_count(asr_model: str | None, strategy: str = "peak") -> int:
    """Passes per chunk for transcribe_chunks: fusion=3, others=2.

    The pass structure is identical for both load strategies; "sum" merely keeps the
    singletons resident between passes (peak VRAM = sum of models) instead of releasing
    them (peak = max). The strategy parameter is kept for call-site compatibility.
    """
    del strategy
    return 3 if _select_engine(asr_model)[0] == "fusion" else 2


def _weighted_align_lang(asr_out: list[tuple[str | None, str, str]]) -> str | None:
    """File-level alignment language: chunk align_langs weighted by alnum text length.

    Full-file alignment runs ONE pass for the whole file, so it needs one language
    (per-chunk languages only steered the per-chunk path). Weighting by text mass
    mirrors the pipeline's transcript-content vote, so the two levels agree even
    when a bad aligner would have produced a misleading number of units.
    """
    from collections import Counter
    from voxweave.lang import to_iso_or, transcript_content_weight

    weight: Counter[str] = Counter()
    for _, text, align_lang in asr_out:
        n = transcript_content_weight(text)
        if n:
            key = to_iso_or(align_lang, None) or align_lang
            weight[key] += n
    return weight.most_common(1)[0][0] if weight else None


def _full_pass_units(
    full_wav: Path | None,
    bounds: Sequence[tuple[float, float]] | None,
    texts: list[str],
    align_lang: str | None,
    speech_spans: list[tuple[float, float]] | None = None,
    song_spans: list[tuple[float, float]] | None = None,
) -> list[list[dict]] | None:
    """Full-file alignment for CTC/MMS languages; None -> caller aligns per chunk.

    One global pass over the whole audio replaces N per-chunk calls: chunk-edge words
    self-locate on the global monotone path, and the MMS ONNX call count drops from one
    per chunk (a movie is ~80 chunks, brushing the ~180-small-call heap-corruption
    regime) to a handful of DP chunks. Returned units are shifted back to chunk-relative
    times, preserving the transcribe_chunks contract.

    Qwen-aligned languages (no CTC config) return None and stay per-chunk: the NAR
    aligner is capped at 180s input (qwen_asr MAX_FORCE_ALIGN_INPUT_SECONDS) and has no
    windowed-emission + global-DP decomposition to scale past it — its timestamps are
    regressed then monotonized by an LIS pass, with no CTC blank to absorb long spans.
    Any full-pass failure also returns None (per-chunk fallback), mirroring align_text's
    CTC->Qwen fallback.
    """
    from voxweave.lang import is_supported, to_iso

    if full_wav is None or bounds is None or len(bounds) != len(texts):
        return None
    if not align_lang or not is_supported(align_lang):
        return None
    iso = to_iso(align_lang)
    model_name = config.align_model_for(iso)
    if not model_name:
        return None
    from voxweave.timestamps import shift_units

    try:
        # crop_to_envelope=True and mute_spans are safe ONLY here: bounds are fresh VAD
        # chunk windows and song_spans the intervals this run's song detection excised —
        # both computed on this very audio, no external timestamp trusted. The align
        # subcommand passes input-VTT bounds and must keep the defaults (routing-free,
        # unmuted). Muting kills mid-file smear the envelope crop cannot reach: an
        # excised song has no transcript by construction, so zeroing its samples only
        # removes acoustic bait for neighbouring sentences.
        if _is_mms_name(model_name):
            blocks = align_blocks_full_mms(
                full_wav,
                texts,
                iso,
                bounds=list(bounds),
                crop_to_envelope=True,
                mute_spans=song_spans,
            )
        else:
            blocks = align_blocks_full_ctc(
                full_wav,
                texts,
                iso,
                model_name,
                bounds=list(bounds),
                speech_spans=speech_spans,
                crop_to_envelope=True,
                mute_spans=song_spans,
            )
    except Exception as e:  # noqa: BLE001 -- any failure falls back to per-chunk alignment
        log.warning(
            "full-file alignment failed (%s: %s), falling back to per-chunk alignment",
            type(e).__name__,
            e,
        )
        # an OOM'd full pass leaves fragmented VRAM; reclaim it or the
        # fallback path can cascade into OOM as well
        _empty_cache()
        return None
    return [shift_units(u, -b[0]) for u, b in zip(blocks, bounds)]


def _asr_chunk_safe(
    engine: str,
    wav: Path,
    language: str | None,
    model_id: str,
    context: str | None,
    idx: int,
    total: int,
    failures: list[Exception],
) -> tuple[str | None, str, str]:
    """One chunk's ASR with failure containment: an exception degrades to empty
    text (the same path as genuine silence downstream) instead of killing the
    run, so hours of prior chunks are not thrown away."""
    try:
        return _asr_only(engine, wav, language, model_id, context)
    except Exception as e:  # noqa: BLE001 -- one bad chunk must not kill the run
        failures.append(e)
        log.warning(
            "ASR failed on chunk %d/%d (%s: %s); continuing with empty text",
            idx + 1,
            total,
            type(e).__name__,
            e,
        )
        _empty_cache()
        return (None, "", "")


def _align_chunk_safe(
    wav: Path, text: str, align_lang: str, idx: int, total: int
) -> list[dict]:
    """One chunk's per-chunk alignment with failure containment: the transcript
    text survives, only its word timing is lost."""
    try:
        return align_text(wav, text, align_lang)
    except Exception as e:  # noqa: BLE001 -- one bad chunk must not kill the run
        log.warning(
            "alignment failed on chunk %d/%d (%s: %s); keeping text without word timing",
            idx + 1,
            total,
            type(e).__name__,
            e,
        )
        _empty_cache()
        return []


def _raise_if_all_failed(failures: list[Exception], total: int) -> None:
    """Every chunk erroring is a broken run, not a silent empty transcript."""
    if total and len(failures) >= total:
        e = failures[-1]
        raise RuntimeError(
            f"ASR failed on all {total} chunks (last error: {type(e).__name__}: {e})"
        )


def transcribe_chunks(
    wav_paths: list[Path],
    language: str | None,
    asr_model: str | None = None,
    context: str | None = None,
    on_done=None,
    strategy: str = "peak",
    full_wav: Path | None = None,
    bounds: list[tuple[float, float]] | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
    song_spans: list[tuple[float, float]] | None = None,
) -> list[tuple[str | None, str, list[dict]]]:
    """Transcribe a list of chunks -> [(lang, text, units)] matching the transcribe_align contract.

    Pass structure is fixed: all-chunks ASR pass(es), then one alignment pass (fusion:
    whisper ASR -> Qwen ASR -> align + merge). strategy controls model residency between
    passes (config.conf_load_strategy):
    - "peak" (default): singletons released between passes; peak VRAM = max(models).
    - "sum": singletons stay resident across passes; peak = sum(models); saves the
      release/reload overhead on high-VRAM cards.

    ``full_wav`` + ``bounds`` (absolute chunk windows on full_wav) enable ONE full-file
    alignment pass for CTC/MMS file-level languages (see _full_pass_units); Qwen-aligned
    languages and callers that omit them keep per-chunk alignment. on_done(i) is called
    per completed pass; total = N * chunk_pass_count(). Aligner kept alive until release().
    """
    counter = [0]

    def _tick() -> None:
        if on_done:
            on_done(counter[0])
        counter[0] += 1

    release = strategy != "sum"  # sum keeps singletons co-resident between passes
    engine, mid = _select_engine(asr_model)
    if engine == "fusion":
        qid = resolve_asr_model(config.conf_fusion_qwen())
        fusion_whisper = config.conf_fusion_whisper()
        n = len(wav_paths)
        # pass A: whisper ASR all chunks
        w_fail: list[Exception] = []
        w_asr: list[tuple[str | None, str, str]] = []
        for i, w in enumerate(wav_paths):
            w_asr.append(
                _asr_chunk_safe(
                    "whisper", w, language, fusion_whisper, context, i, n, w_fail
                )
            )
            _tick()
        if release:
            _release_whisper()
        # pass B: Qwen ASR all chunks
        q_fail: list[Exception] = []
        q_asr: list[tuple[str | None, str, str]] = []
        for i, w in enumerate(wav_paths):
            q_asr.append(
                _asr_chunk_safe("qwen", w, language, qid, context, i, n, q_fail)
            )
            _tick()
        if release:
            _release_qwen_asr()
        # both engines failing everywhere = broken run; one engine surviving
        # anywhere still fuses into usable output
        if len(w_fail) >= n:
            _raise_if_all_failed(q_fail, n)
        # pass C: align both texts (whisper units carry the timing; Qwen units only
        # position punctuation), full-file where the language allows, then merge
        full_w = _full_pass_units(
            full_wav,
            bounds,
            [t for _, t, _ in w_asr],
            _weighted_align_lang(w_asr),
            speech_spans=speech_spans,
            song_spans=song_spans,
        )
        full_q = _full_pass_units(
            full_wav,
            bounds,
            [t for _, t, _ in q_asr],
            _weighted_align_lang(q_asr),
            speech_spans=speech_spans,
            song_spans=song_spans,
        )
        out: list[tuple[str | None, str, list[dict]]] = []
        for i, (w, (dw, tw, aw), (dq, tq, aq)) in enumerate(
            zip(wav_paths, w_asr, q_asr)
        ):
            uw = (
                (
                    full_w[i]
                    if full_w is not None
                    else _align_chunk_safe(w, tw, aw, i, n)
                )
                if tw.strip()
                else []
            )
            uq = (
                (
                    full_q[i]
                    if full_q is not None
                    else _align_chunk_safe(w, tq, aq, i, n)
                )
                if tq.strip()
                else []
            )
            out.append(_fuse_chunk((dw, tw, uw), (dq, tq, uq), language))
            _empty_cache()
            _tick()
        return out
    # qwen / whisper: ASR pass -> alignment pass (full-file where the language allows)
    n = len(wav_paths)
    failures: list[Exception] = []
    asr_out: list[tuple[str | None, str, str]] = []  # (det_lang, text, align_lang)
    for i, w in enumerate(wav_paths):
        asr_out.append(
            _asr_chunk_safe(engine, w, language, mid, context, i, n, failures)
        )
        _tick()
    if release:
        _release_whisper() if engine == "whisper" else _release_qwen_asr()
    _raise_if_all_failed(failures, n)
    full_units = _full_pass_units(
        full_wav,
        bounds,
        [t for _, t, _ in asr_out],
        _weighted_align_lang(asr_out),
        speech_spans=speech_spans,
        song_spans=song_spans,
    )
    out2: list[tuple[str | None, str, list[dict]]] = []
    for i, (w, (det, text, align_lang)) in enumerate(zip(wav_paths, asr_out)):
        if not text.strip():
            units: list[dict] = []
        elif full_units is not None:
            units = full_units[i]
        else:
            units = _align_chunk_safe(w, text, align_lang, i, n)
        out2.append((det, text, units))
        _empty_cache()  # alignment matrix grows with length; reclaim after each chunk
        _tick()
    return out2


def transcribe_align(
    wav_path: Path,
    language: str | None,
    asr_model: str | None = None,
    context: str | None = None,
) -> tuple[str | None, str, list[dict]]:
    """Local ASR + forced alignment -> (detected language | None, punctuated text, units).

    Engine selected from asr_model: whisper size -> hybrid; fusion alias -> dual-ASR; else -> Qwen.
    All engines return the same contract; pipeline is engine-agnostic.
    """
    engine, model_id = _select_engine(asr_model)
    if engine == "fusion":
        return _transcribe_fusion(wav_path, language, context)
    if engine == "whisper":
        return _transcribe_whisper_align(wav_path, language, model_id, context)
    return _transcribe_qwen_align(wav_path, language, model_id, context)


# ───────────────────────────── forced alignment (align command, independent singleton) ────────────────────


def _get_aligner():
    """Lazy-load Qwen3ForcedAligner singleton for the align command (reused across windows).

    Loads only the aligner, not the full Qwen3ASRModel stack, since the align command already
    has text and only needs alignment. Saves the ASR half of VRAM.
    """
    global _aligner
    if _aligner is None:
        try:
            from qwen_asr import Qwen3ForcedAligner
        except ModuleNotFoundError as e:
            raise _require(e.name or "qwen_asr") from e

        dev = get_device()
        log.info("loading forced aligner=%s on %s", ALIGNER_MODEL, dev)
        # Use snapshot so model + processor both land in config.ALIGN_CACHE (see _hf_snapshot docstring).
        local = _hf_snapshot(ALIGNER_MODEL, config.ALIGN_CACHE)
        _aligner = Qwen3ForcedAligner.from_pretrained(
            local, dtype=_model_dtype(dev), device_map=dev
        )
        log.info("forced aligner ready")
    return _aligner


def _parse_whisper_device() -> tuple[str, int]:
    """Resolved device ('cuda:0'/'cuda'/'mps'/'cpu') -> faster-whisper's (device, device_index).
    ctranslate2 has no Metal backend, so mps (and anything non-cuda) maps to CPU."""
    dev = get_device().strip()
    if dev.startswith("cuda"):
        idx = int(dev.split(":", 1)[1]) if ":" in dev else 0
        return "cuda", idx
    return "cpu", 0


def _get_whisper(model_id: str):
    """Lazy-load the whisper engine singleton (reloads if size changes).

    Apple Silicon: mlx-whisper (Metal); ctranslate2/faster-whisper has no Metal backend, so the
    MLX port supplies the hybrid/fusion engines' text. Else: faster-whisper (CUDA fp16 / CPU int8).
    """
    if _use_mlx():
        from voxweave import backend_mlx

        return backend_mlx.get_whisper(model_id)
    global _whisper, _whisper_id
    if _whisper is not None and _whisper_id != model_id:
        release()
    if _whisper is None:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as e:
            raise _require(e.name or "faster_whisper", _MISSING_WHISPER) from e

        device, index = _parse_whisper_device()
        compute = WHISPER_COMPUTE or ("float16" if device == "cuda" else "int8")
        log.info("loading whisper=%s on %s:%d (%s)", model_id, device, index, compute)
        _whisper = WhisperModel(
            model_id,
            device=device,
            device_index=index,
            compute_type=compute,
            download_root=config.ASR_CACHE,
        )
        _whisper_id = model_id
        log.info("whisper ready")
    return _whisper


# --------------------------------------------------------------------------- #
# wav2vec2 CTC forced alignment (WhisperX-equivalent; English default, per-language selection in voxweave.config)
# --------------------------------------------------------------------------- #


def align_text(wav_path: Path, text: str, language: str) -> list[dict]:
    """Forced alignment -> units [{text,start,end}]. language accepts ISO or full name.

    Dispatch: if config.align_model_for(iso) is set, use CTC (mms/ctc alias -> align_text_mms;
    torchaudio bundle / HF wav2vec2 id -> align_text_ctc). CTC failure falls back to the
    Qwen3ForcedAligner. Languages with no CTC config go directly to Qwen.

    The CTC aligners run on every backend (wav2vec2 is torch -> MPS-capable; MMS is onnxruntime ->
    CUDA on Linux, CPU on macOS — CoreML is avoided as its Metal context segfaults alongside MLX,
    see _mms_providers), so English keeps its WhisperX-grade wav2vec2 alignment even on Apple
    Silicon. Only the Qwen fallback differs: on the MLX backend the torch qwen-asr aligner is
    absent, so the native MLX Qwen3-ForcedAligner serves it instead.
    """
    from voxweave.lang import is_supported, to_aligner_name, to_iso

    if is_supported(language):
        iso = to_iso(language)
        model_name = config.align_model_for(iso)
        if model_name:
            try:
                if _is_mms_name(model_name):
                    return align_text_mms(wav_path, text, iso)
                return align_text_ctc(wav_path, text, iso, model_name)
            except Exception as e:  # noqa: BLE001 -- any CTC failure falls back to Qwen
                log.warning(
                    "CTC alignment failed (%s: %s), falling back to Qwen alignment",
                    type(e).__name__,
                    e,
                )
                _empty_cache()  # reclaim CTC debris before loading the Qwen aligner

    if (
        _use_mlx()
    ):  # Apple Silicon: torch qwen-asr aligner is absent -> use the MLX Qwen aligner
        from voxweave import backend_mlx

        return backend_mlx.align(wav_path, text, language)

    aligner = _get_aligner()
    results = aligner.align(str(wav_path), text, to_aligner_name(language))
    units = [
        {"text": it.text, "start": float(it.start_time), "end": float(it.end_time)}
        for it in results[0]
    ]
    _empty_cache()
    return units


def _release_whisper() -> None:
    """Release the whisper engine singleton. Called between fusion passes to reduce peak VRAM."""
    global _whisper, _whisper_id
    _whisper = None
    _whisper_id = None
    if _use_mlx():
        from voxweave import backend_mlx

        backend_mlx.release_whisper()
    _empty_cache()


def _release_qwen_asr() -> None:
    """Release Qwen3-ASR singleton. Called between fusion passes; CTC aligner released separately."""
    global _asr, _asr_id
    _asr = None
    _asr_id = None
    if _use_mlx():
        from voxweave import backend_mlx

        backend_mlx.release_asr()
    _empty_cache()


def release() -> None:
    """Release all ASR/alignment singletons. Call after end of transcription or alignment episode."""
    global _aligner
    _release_qwen_asr()
    _release_whisper()
    _aligner = None
    release_ctc()
    release_mms()
    if _use_mlx():
        from voxweave import backend_mlx

        backend_mlx.release()
    _empty_cache()
