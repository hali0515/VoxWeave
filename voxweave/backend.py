from __future__ import annotations

import inspect
import logging
import os
import tempfile
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

from voxweave import config

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
# Compute device, resolved lazily on first use so importing voxweave never pulls in torch.
# VOXWEAVE_DEVICE overrides ('cuda:0' / 'mps' / 'cpu'); otherwise autodetect cuda -> mps -> cpu.
_DEVICE: str | None = None


def get_device() -> str:
    """Resolve the compute device once and cache it. Env override wins; else cuda > mps > cpu."""
    global _DEVICE
    if _DEVICE is None:
        env = os.environ.get("VOXWEAVE_DEVICE", "").strip()
        if env:
            _DEVICE = env
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    _DEVICE = "cuda:0"
                elif (
                    getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                ):
                    _DEVICE = "mps"
                else:
                    _DEVICE = "cpu"
            except Exception:  # noqa: BLE001 -- torch absent/broken -> CPU is the safe default
                _DEVICE = "cpu"
    return _DEVICE


def _model_dtype(device: str):
    """Per-device load dtype: bfloat16 on CUDA, float16 on MPS (bf16 is only partially supported
    on Metal), float32 on CPU."""
    import torch

    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _use_mlx() -> bool:
    """True if the MLX (Apple Silicon) backend should serve ASR + forced alignment.

    VOXWEAVE_BACKEND=mlx|torch overrides; otherwise auto-on when the resolved device is mps.
    See voxweave.backend_mlx for the adapters; separation/song-skip always stay on torch.
    """
    env = os.environ.get("VOXWEAVE_BACKEND", "").strip().lower()
    if env == "mlx":
        return True
    if env == "torch":
        return False
    return get_device() == "mps"


# Empty -> float16 on cuda, int8 on cpu/mps (ctranslate2/faster-whisper is CUDA-or-CPU only).
WHISPER_COMPUTE = os.environ.get("VOXWEAVE_WHISPER_COMPUTE", "")

# ASR/alignment process-level singletons; call release() at end of episode.
# Separator is not kept resident (self-loads, self-releases).
_asr = None  # qwen_asr.Qwen3ASRModel
_asr_id = None  # currently loaded ASR repo id (reloaded on --model change)
# Standalone aligner for the align command (no ASR needed, so we skip the full Qwen3ASRModel stack).
_aligner = None  # qwen_asr.Qwen3ForcedAligner
_whisper = None  # faster_whisper.WhisperModel
_whisper_id = None  # currently loaded whisper size string
# wav2vec2 CTC aligner; cached by iso. blank/sep_id/invocab come from the model, not hardcoded.
# proc is only set on the HF path (required for z-score normalization).
CtcAligner = namedtuple("CtcAligner", "kind model sr blank sep_id invocab proc")
_ctc = None  # CtcAligner
_ctc_lang = None  # iso of the loaded CTC singleton (reloaded on language change)

# MMS-300m (ctc-forced-aligner + uroman romanization -> zero OOV): default for ja/CJK.
# Replaces wav2vec2-xlsr whose OOV-wildcard on rare kanji caused cue-tail collapse / drift.
# Same model as whisperx --align_backend ctc. ONNX singleton, cleared on release().
CtcMms = namedtuple("CtcMms", "session tokenizer")
_mms = None  # CtcMms
# config [align] aliases that route to align_text_mms instead of wav2vec2
_MMS_NAMES = {"mms", "mms_fa", "ctc", "ctc-forced-aligner", "mms-300m"}
MMS_SR = 16000  # ctc-forced-aligner requires fixed 16k
MMS_BATCH = int(os.environ.get("VOXWEAVE_MMS_BATCH", "4"))
# VOXWEAVE_MMS_MODEL (explicit local path) wins if it exists; otherwise pulled from HF -> config.ALIGN_CACHE.
MMS_MODEL = os.path.expanduser(os.environ.get("VOXWEAVE_MMS_MODEL", ""))
MMS_REPO = os.environ.get("VOXWEAVE_MMS_REPO", "deskpai/ctc_forced_aligner")
MMS_REPO_FILE = os.environ.get(
    "VOXWEAVE_MMS_REPO_FILE", "04ac86b67129634da93aea76e0147ef3.onnx"
)
# wav2vec2 CTC windowed emission (mirrors ctc-forced-aligner generate_emissions): encode the
# waveform in CTC_EMIT_WINDOW_S windows with CTC_EMIT_CONTEXT_S overlap each side (so edge frames
# stay well-attended), drop the context frames, concatenate -> bounds the encoder's O(T^2)
# self-attention so the full-file CTC pass survives long audio (full-file xlsr OOMs at 23min).
CTC_EMIT_WINDOW_S = float(os.environ.get("VOXWEAVE_CTC_WINDOW_S", "30"))
CTC_EMIT_CONTEXT_S = float(os.environ.get("VOXWEAVE_CTC_CONTEXT_S", "2"))
_CTC_STRIDE = 320  # wav2vec2 @16k downsamples 320x -> 50fps (20ms/frame)
# Single global forced-align DP is O(T*L); cap audio length so it stays in memory. Movies past
# this are auto-split at silence anchors (plan_dp_chunks) when cue timestamps are available.
# Sourced via config (env VOXWEAVE_CTC_MAX_DP_FRAMES > conf ctc_max_dp_frames > 90000≈30min).
CTC_MAX_DP_FRAMES = config.conf_ctc_max_dp_frames()
# Per-chunk DP budget as a fraction of CTC_MAX_DP_FRAMES: leaves headroom so an off-by-a-bit
# silence anchor never pushes a chunk's O(T*L) trellis past the memory cap.
CTC_DP_CHUNK_FRAC = float(os.environ.get("VOXWEAVE_CTC_DP_CHUNK_FRAC", "0.8"))

_MISSING_HINT = (
    "Local model loading requires the voxweave[cuda] or voxweave[mps] install "
    "(qwen-asr + einops/rotary-embedding-torch/... + torch). "
    "Install: `make install` (NVIDIA/Linux, cu128 wheel) or `make install VARIANT=mps` "
    "(Apple Silicon/macOS). Missing: {mod}"
)


_MISSING_WHISPER = (
    "faster-whisper engine requires the voxweave[cuda] install (faster-whisper + qwen-asr aligner; "
    "CUDA/Linux only — ctranslate2 has no Metal/MPS backend). On Apple Silicon the whisper engine "
    "runs via mlx-whisper instead (voxweave[mps]); this torch path is only reached with "
    "VOXWEAVE_BACKEND=torch. Install: `make install` or `uv pip install -e '.[cuda]'`. Missing: {mod}"
)


def _require(mod: str, hint: str = _MISSING_HINT) -> RuntimeError:
    """Build a missing-dependency error. Pass _MISSING_WHISPER for the whisper path."""
    return RuntimeError(hint.format(mod=mod))


def _load_yaml(path: Path) -> dict:
    """SafeLoader + !!python/tuple support (used by window sizes in MSST-style configs)."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )
    return yaml.load(path.read_text(), Loader=_Loader)


def _empty_cache() -> None:
    """Best-effort VRAM reclaim (optimization only; failure is harmless)."""
    try:
        import torch

        dev = get_device()
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        elif dev == "mps":
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ───────────────────────────── vocal separation (Mel-Band Roformer, self-load/self-release) ──────────────


def _strip_state_dict(sd: dict) -> dict:
    """Strip Lightning state_dict wrapper and 'model.' prefix."""
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if sd and all(k.startswith("model.") for k in sd):
        sd = {k[len("model.") :]: v for k, v in sd.items()}
    return sd


def _hf_download(repo: str, filename: str, cache_dir: str | None = None) -> str:
    """Download a single file from HF, return local path. cache_dir routes into a voxweave-owned subdir; None keeps HF default."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as e:
        raise _require(e.name or "huggingface_hub") from e
    return hf_hub_download(repo, filename, cache_dir=cache_dir)


def _hf_snapshot(repo: str, cache_dir: str) -> str:
    """Download a full repo snapshot into cache_dir, return the local snapshot dir.

    qwen_asr's from_pretrained loads the processor without forwarding cache_dir
    (AutoProcessor.from_pretrained(path) with no kwargs), so passing cache_dir alone would leak
    processor files to the default HF hub. Pointing from_pretrained at a local snapshot path
    ensures model + processor both land under cache_dir.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as e:
        raise _require(e.name or "huggingface_hub") from e
    return snapshot_download(repo, cache_dir=cache_dir)


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


def _demix(model, mix, cfg, progress=None):
    """Chunked overlap-add inference: mix [ch, t] float32 -> vocals [ch, t].

    Hann window + >=2x overlap satisfies COLA; tail normalized by window sum. num_stems=1
    output may or may not have a stem dimension -- both shapes are handled.
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
    with torch.no_grad():
        for k, start in enumerate(starts):
            seg = mix[:, start : start + chunk]
            n = seg.shape[1]
            if n < chunk:  # pad final segment to full chunk size
                seg = torch.nn.functional.pad(seg, (0, chunk - n))
            out = model(seg.unsqueeze(0).to(dev))  # [1, (stems,) ch, t]
            if out.dim() == 4:  # [1, stems, ch, t] -> take first (vocals) stem
                out = out[:, 0]
            out = out[0].float().cpu()  # [ch, chunk]
            w = window[:n]
            result[:, start : start + n] += out[:, :n] * w
            weight[start : start + n] += w
            if progress is not None:
                progress(k + 1, nwin)
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
        _asr = Qwen3ASRModel.from_pretrained(
            local,
            dtype=_model_dtype(dev),
            device_map=dev,
        )
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


def _asr_only(
    engine: str,
    wav_path: Path,
    language: str | None,
    model_id: str,
    context: str | None,
) -> tuple[str | None, str, str]:
    """Transcribe only (no alignment): returns (detected language | None, punctuated text, align_lang).

    First pass of the two-pass peak strategy: alignment deferred to pass two after ASR is released.
    align_lang pre-computed here; falls back to 'en' for empty text (skipped in pass two anyway).
    """
    if engine == "whisper":
        from voxweave.lang import to_iso_or

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
            condition_on_previous_text=False,  # prevents repetition hallucination
            vad_filter=False,  # VAD chunking already done upstream
            word_timestamps=False,  # hybrid uses Qwen for timestamps, not whisper
        )
        text = "".join(s.text for s in segments)  # segments is a generator
        det = info.language
        src = "whisper"
    else:  # qwen
        model = _get_asr(model_id)
        kwargs: dict = {"language": language or None, "return_time_stamps": False}
        if context:  # omit kwarg entirely when empty to preserve legacy behavior
            kwargs["context"] = context
        r = model.transcribe(str(wav_path), **kwargs)[0]
        det = (r.language or "").split(",")[
            0
        ].strip() or None  # "Chinese,English" -> take first
        text = r.text
        src = "ASR"
    align_lang = _resolve_align_lang(language or det, src) if text.strip() else "en"
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
    mirrors the pipeline's unit-count vote (unit count ~ alnum chars / words), so the
    two levels agree on mixed-language files.
    """
    from collections import Counter

    weight: Counter[str] = Counter()
    for _, text, align_lang in asr_out:
        n = sum(1 for c in text if c.isalnum())
        if n:
            weight[align_lang] += n
    return weight.most_common(1)[0][0] if weight else None


def _full_pass_units(
    full_wav: Path | None,
    bounds: Sequence[tuple[float, float]] | None,
    texts: list[str],
    align_lang: str | None,
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
        if _is_mms_name(model_name):
            blocks = align_blocks_full_mms(full_wav, texts, iso, bounds=list(bounds))
        else:
            blocks = align_blocks_full_ctc(
                full_wav, texts, iso, model_name, bounds=list(bounds)
            )
    except Exception as e:  # noqa: BLE001 -- any failure falls back to per-chunk alignment
        log.warning(
            "full-file alignment failed (%s: %s), falling back to per-chunk alignment",
            type(e).__name__,
            e,
        )
        return None
    return [shift_units(u, -b[0]) for u, b in zip(blocks, bounds)]


def transcribe_chunks(
    wav_paths: list[Path],
    language: str | None,
    asr_model: str | None = None,
    context: str | None = None,
    on_done=None,
    strategy: str = "peak",
    full_wav: Path | None = None,
    bounds: list[tuple[float, float]] | None = None,
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
        # pass A: whisper ASR all chunks
        w_asr: list[tuple[str | None, str, str]] = []
        for w in wav_paths:
            w_asr.append(_asr_only("whisper", w, language, fusion_whisper, context))
            _tick()
        if release:
            _release_whisper()
        # pass B: Qwen ASR all chunks
        q_asr: list[tuple[str | None, str, str]] = []
        for w in wav_paths:
            q_asr.append(_asr_only("qwen", w, language, qid, context))
            _tick()
        if release:
            _release_qwen_asr()
        # pass C: align both texts (whisper units carry the timing; Qwen units only
        # position punctuation), full-file where the language allows, then merge
        full_w = _full_pass_units(
            full_wav, bounds, [t for _, t, _ in w_asr], _weighted_align_lang(w_asr)
        )
        full_q = _full_pass_units(
            full_wav, bounds, [t for _, t, _ in q_asr], _weighted_align_lang(q_asr)
        )
        out: list[tuple[str | None, str, list[dict]]] = []
        for i, (w, (dw, tw, aw), (dq, tq, aq)) in enumerate(
            zip(wav_paths, w_asr, q_asr)
        ):
            uw = (
                (full_w[i] if full_w is not None else align_text(w, tw, aw))
                if tw.strip()
                else []
            )
            uq = (
                (full_q[i] if full_q is not None else align_text(w, tq, aq))
                if tq.strip()
                else []
            )
            out.append(_fuse_chunk((dw, tw, uw), (dq, tq, uq), language))
            _empty_cache()
            _tick()
        return out
    # qwen / whisper: ASR pass -> alignment pass (full-file where the language allows)
    asr_out: list[tuple[str | None, str, str]] = []  # (det_lang, text, align_lang)
    for w in wav_paths:
        asr_out.append(_asr_only(engine, w, language, mid, context))
        _tick()
    if release:
        _release_whisper() if engine == "whisper" else _release_qwen_asr()
    full_units = _full_pass_units(
        full_wav, bounds, [t for _, t, _ in asr_out], _weighted_align_lang(asr_out)
    )
    out2: list[tuple[str | None, str, list[dict]]] = []
    for i, (w, (det, text, align_lang)) in enumerate(zip(wav_paths, asr_out)):
        if not text.strip():
            units: list[dict] = []
        elif full_units is not None:
            units = full_units[i]
        else:
            units = align_text(w, text, align_lang)
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
        else:
            t = out[nxt]["start"]
        u["start"] = u["end"] = round(t, 3)
    return out


def _ctc_words_from_spans(
    spans, meta: list[int], words: list[str], ratio: float
) -> list[dict]:
    """Token-level spans + word idx -> word-level units [{text,start,end}].

    Groups tokens by word idx (separator meta<0 skipped); start/end = frame range * ratio (-> seconds).
    Pure logic; spans just need .start/.end attributes.
    """
    groups: dict[int, list] = {}  # dict insertion order == word order (Py3.7+)
    for span, m in zip(spans, meta):
        if m < 0:
            continue
        groups.setdefault(m, []).append(span)
    units: list[dict] = []
    for widx, sps in groups.items():
        start = min(s.start for s in sps) * ratio
        end = max(s.end for s in sps) * ratio
        units.append(
            {
                "text": _strip_trailing_punct(words[widx]),
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
    return units


def _get_ctc_aligner(iso: str, model_name: str):
    """Lazy-load wav2vec2 CTC aligner singleton, cached by iso. Reloads on language change.

    model_name in torchaudio.pipelines -> torchaudio bundle; otherwise -> HF Wav2Vec2ForCTC
    (auto-downloads to config.ALIGN_CACHE). Returns a CtcAligner namedtuple.
    """
    global _ctc, _ctc_lang
    if _ctc is not None and _ctc_lang != iso:
        _ctc = None
        _ctc_lang = None
        _empty_cache()
    if _ctc is None:
        import torchaudio

        dev = get_device()
        if (
            model_name in torchaudio.pipelines.__all__
        ):  # torchaudio bundle (English large)
            bundle = torchaudio.pipelines.__dict__[model_name]
            model = bundle.get_model().to(dev).eval()
            labels = bundle.get_labels()
            sep_id = labels.index("|") if "|" in labels else -1
            invocab = {c: i for i, c in enumerate(labels) if i not in (0, sep_id)}
            _ctc = CtcAligner(
                "torchaudio", model, bundle.sample_rate, 0, sep_id, invocab, None
            )
        else:  # HF Wav2Vec2ForCTC (English LV60K-self, Japanese xlsr etc.): blank=pad_id, invocab=full vocab
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            # Auto-downloads to config.ALIGN_CACHE on first run; cache hit on subsequent runs.
            proc = Wav2Vec2Processor.from_pretrained(
                model_name, cache_dir=config.ALIGN_CACHE
            )
            model = (
                Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=config.ALIGN_CACHE)
                .to(dev)
                .eval()
            )
            vocab = proc.tokenizer.get_vocab()  # {char: id}
            _ctc = CtcAligner(
                "hf",
                model,
                proc.feature_extractor.sampling_rate,
                proc.tokenizer.pad_token_id,
                vocab.get("|", -1),
                vocab,
                proc,
            )
        _ctc_lang = iso
        log.info(
            "loaded CTC aligner lang=%s model=%s kind=%s on %s",
            iso,
            model_name,
            _ctc.kind,
            dev,
        )
    return _ctc


def _ctc_logp(al, wav):
    """Single forward pass: 1D waveform tensor @ al.sr -> [T,V] log-probs (softmax of raw logits).

    HF models need the processor's z-score normalization (skipping it shifts argmax); torchaudio
    bundles take the raw waveform. Must log_softmax the raw logits before forced_align.
    """
    import torch

    if wav.shape[-1] < 400:  # wav2vec2 conv minimum input length, prevents crash
        wav = torch.nn.functional.pad(wav, (0, 400 - wav.shape[-1]))
    with torch.inference_mode():
        if al.kind == "hf":  # HF: processor z-score normalization required
            inp = al.proc(
                wav.cpu().numpy(), sampling_rate=al.sr, return_tensors="pt"
            ).input_values
            logits = al.model(inp.to(get_device())).logits[0]
        else:  # torchaudio bundle: raw wav, returns (emissions, lengths)
            emis, _ = al.model(wav.unsqueeze(0).to(get_device()))
            logits = emis[0]
        return torch.log_softmax(logits, dim=-1)  # [T,V]


def _ctc_emit_full(al, wav):
    """Long waveform -> seamless [T,V] log-probs via windowed forward passes.

    Mirrors ctc-forced-aligner generate_emissions for wav2vec2: encode in CTC_EMIT_WINDOW_S
    windows padded by CTC_EMIT_CONTEXT_S of overlap each side (edge frames stay well-attended),
    drop the context frames, concatenate. The kept interior of each window tiles the file
    gap-free; the encoder never sees more than window+2*context, bounding O(T^2) self-attention.
    """
    import torch

    sr = al.sr
    win = int(CTC_EMIT_WINDOW_S * sr)
    ctx = int(CTC_EMIT_CONTEXT_S * sr)
    n = wav.shape[-1]
    if n <= win + 2 * ctx:
        return _ctc_logp(al, wav)  # short enough for a single pass
    parts = []
    pos = 0
    while pos < n:
        a = max(0, pos - ctx)
        b = min(n, pos + win + ctx)
        lp = _ctc_logp(al, wav[a:b])  # [t,V]
        stride = (b - a) / lp.shape[0]  # samples/frame for this window (~320)
        end = min(pos + win, n)
        lo = round((pos - a) / stride)  # drop prepended left-context frames
        hi = lp.shape[0] - (round((b - end) / stride) if end < n else 0)
        lo = max(0, min(lo, lp.shape[0]))
        hi = max(lo, min(hi, lp.shape[0]))
        parts.append(lp[lo:hi])
        pos = end
    return torch.cat(parts, dim=0)


def _ctc_align_logp(al, logp, toks, meta, words, nospace, total_samples):
    """[T,V] log-probs + tokens -> word/char units. Shared by per-cue and full-file CTC.

    Appends a wildcard column for OOV tokens (WhisperX technique), runs forced_align + merge,
    maps frames to seconds via total_samples/T/sr. No-space langs get a last-resort span fill.
    """
    import torch
    import torchaudio.functional as AF

    toks = list(toks)
    if any(
        t is None for t in toks
    ):  # OOV wildcard: max non-blank score per frame column
        cols = [i for i in range(logp.shape[1]) if i != al.blank]
        star = logp[:, cols].max(dim=1).values
        logp = torch.cat([logp, star.unsqueeze(1)], dim=1)
        star_id = logp.shape[1] - 1
        toks = [star_id if t is None else t for t in toks]
    # torchaudio.forced_align has no MPS kernel, so on Apple Silicon run the (cheap) DP on CPU —
    # emissions stay on MPS for the forward. CUDA is left untouched: forced_align runs on the GPU
    # exactly as before (the DP is deterministic, so CPU/CUDA give identical alignments).
    if logp.device.type == "mps":
        logp = logp.detach().to("cpu")
    targets = torch.tensor([toks], dtype=torch.int32, device=logp.device)
    aligned, scores = AF.forced_align(
        logp.unsqueeze(0).contiguous(), targets, blank=al.blank
    )
    spans = AF.merge_tokens(aligned[0], scores[0], blank=al.blank)
    ratio = total_samples / logp.shape[0] / al.sr
    units = _ctc_words_from_spans(spans, meta, words, ratio)
    if nospace:  # last-resort span fill; never drops a character
        units = interp_missing(units)
    return units


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


def _ctc_build_tokens(norm: list[str], nospace: bool, al):
    """Build the <star>-interleaved token stream for full-pass CTC over cue texts `norm`.

    Token stream: <star> word0 <star> word1 <star> ... wordN <star>. A wildcard star sits at
    EVERY word boundary (not just cue boundaries) and both edges. The star is a None token
    (-> wildcard column in _ctc_align_logp) at meta=-1 (-> skipped in word grouping AND in
    _distribute_units, which counts real words/chars). Because a star sits between every pair of
    words regardless of how cues are grouped, the global monotone path absorbs ANY inter-word gap
    -- intra-cue music/silence included -- instead of cramming the later word forward (the
    failure: "...these <2-3s gap> blocks" placed blocks right after these, ignoring the pause).
    Returns (toks, meta, words); words are flattened in cue order for _distribute_units.
    """
    toks: list[int | None] = []
    meta: list[int] = []
    words: list[str] = []

    def _star() -> None:
        toks.append(None)
        meta.append(-1)

    _star()
    for t in norm:
        if not t:
            continue
        for it in list(t) if nospace else t.split():
            if nospace and not it.isalnum():
                continue
            widx = len(words)
            # no case-fold for no-space vocabs (xlsr-ja has uppercase A/C/P only); upper otherwise
            toks.extend(al.invocab.get(c if nospace else c.upper()) for c in it)
            meta.extend(widx for _ in it)
            words.append(it)
            _star()  # wildcard after every word absorbs the inter-word gap
    return toks, meta, words


def _ctc_full_pass(
    al, wav, norm: list[str], nospace: bool, iso: str
) -> list[list[dict]]:
    """One windowed-emission + global forced_align over `wav` for cue texts `norm`.

    Times are relative to the start of `wav` (caller offsets when `wav` is a chunk). Returns
    per-cue units in `norm` order; empty/wordless cues get []. The single DP is O(T*L).
    """
    toks, meta, words = _ctc_build_tokens(norm, nospace, al)
    if not words:
        return [[] for _ in norm]
    logp = _ctc_emit_full(al, wav)
    units = _ctc_align_logp(al, logp, toks, meta, words, nospace, wav.shape[-1])
    return _distribute_units(units, norm, iso)


def _dp_chunked_pass(
    wav,
    sr: int,
    norm: list[str],
    bounds: Sequence[tuple[float, float] | None] | None,
    pass_fn,
    label: str,
) -> list[list[dict]]:
    """Run `pass_fn(wav_slice, texts) -> per-block units` under the global-DP memory budget.

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
        return pass_fn(wav, norm)

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
        sub = pass_fn(wav[a:b], norm[p["lo"] : p["hi"]])
        offset = a / sr
        out.extend(shift_units(u, offset) for u in sub)
    return out


def align_blocks_full_ctc(
    wav_path: Path,
    texts: list[str],
    iso: str,
    model_name: str,
    bounds: Sequence[tuple[float, float] | None] | None = None,
) -> list[list[dict]]:
    """Full-audio single-pass wav2vec2 CTC alignment (en analogue of align_blocks_full_mms).

    Runs ONE windowed-emission + global forced_align over the whole audio, then slices flat
    units back to each block by word/char count. The global monotone CTC path self-locates every
    word, immune to the per-cue cropping drift that crammed words into dead air (en "blocks"
    displaced into a 2.6s silence); inter-cue stars absorb untranscribed gaps (music/silence
    between cues) so the path never stretches a real word across a gap. Units are absolute
    timestamps relative to the full wav. Movie-length audio is DP-chunked at silence anchors
    via cue `bounds` (see _dp_chunked_pass).
    """
    from voxweave.realign import NO_SPACE_LANGS

    al = _get_ctc_aligner(iso, model_name)
    nospace = iso in NO_SPACE_LANGS
    norm = [(t or "").strip() for t in texts]
    wav = _load_mono(wav_path, al.sr)

    def _pass(w, sub: list[str]) -> list[list[dict]]:
        out = _ctc_full_pass(al, w, sub, nospace, iso)
        _empty_cache()
        return out

    return _dp_chunked_pass(wav, al.sr, norm, bounds, _pass, "CTC")


def align_text_ctc(wav_path: Path, text: str, iso: str, model_name: str) -> list[dict]:
    """wav2vec2 CTC forced alignment: blank absorbs silence giving tight boundaries.

    Spaced langs -> word-level units. No-space langs (NO_SPACE_LANGS) -> per-char units
    (punctuation skipped; OOV kanji use wildcard; missing spans filled by interp_missing).
    Same star-interleaved full pass as the align subcommand (_ctc_full_pass with a single
    text): a wildcard at every word boundary absorbs intra-chunk gaps (pauses the ASR did
    not transcribe) instead of cramming the next word forward, and the windowed emission
    bounds the encoder's O(T^2) self-attention on long chunks.
    Exceptions propagate; align_text catches and falls back to Qwen.
    """
    from voxweave.realign import NO_SPACE_LANGS

    text = (text or "").strip()
    if not text:
        return []
    al = _get_ctc_aligner(iso, model_name)
    nospace = iso in NO_SPACE_LANGS
    wav = _load_mono(wav_path, al.sr)
    units = _ctc_full_pass(al, wav, [text], nospace, iso)[0]
    _empty_cache()
    return units


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
        mm.session, wav.astype(np.float32), batch_size=MMS_BATCH
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
    norm = [(t or "").strip() for t in texts]

    def _pass(w, sub: list[str]) -> list[list[dict]]:
        full = " ".join(t for t in sub if t)
        flat = _mms_emit_units(w, full, iso)
        _empty_cache()
        return _distribute_units(flat, sub, iso)

    return _dp_chunked_pass(wav, MMS_SR, norm, bounds, _pass, "MMS")


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
    global _aligner, _ctc, _ctc_lang, _mms
    _release_qwen_asr()
    _release_whisper()
    _aligner = None
    _ctc = None
    _ctc_lang = None
    _mms = None
    if _use_mlx():
        from voxweave import backend_mlx

        backend_mlx.release()
    _empty_cache()
