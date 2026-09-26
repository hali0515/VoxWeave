from __future__ import annotations

import hashlib
import inspect
import logging
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, BinaryIO, Literal, overload

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
from voxweave.core.langsets import LANGUAGES_WITHOUT_SPACES
from voxweave.runtime import (
    _MISSING_WHISPER,
    _empty_cache,
    _hf_download,
    _hf_snapshot,
    _load_yaml as _load_yaml,  # re-export for compatibility/tests
    _model_dtype,
    _parse_yaml,
    _require,
    _use_mlx,
    get_device,
)

log = logging.getLogger("voxweave")

# Heavy deps (torch/qwen_asr/roformer) are lazy-imported so importing voxweave doesn't pull in torch.
# Dynamic loading: separation and ASR/alignment are loaded in separate phases so
# peak VRAM = max(the two), not sum.
ASR_MODEL = os.environ.get("VOXWEAVE_ASR_MODEL", config.DEFAULT_ASR_MODEL)
ALIGNER_MODEL = os.environ.get(
    "VOXWEAVE_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"
)
# --asr-model short name -> HF repo id; case-insensitive, tolerates missing org prefix
_ASR_ALIASES = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
}


def resolve_asr_model(name: str | None) -> str:
    """--asr-model value -> HF repo id. Empty -> default; contains '/' -> pass-through; otherwise check alias table, fall back to prepending 'Qwen/'."""
    if not name or not name.strip():
        return ASR_MODEL
    v = name.strip()
    if "/" in v:
        return v
    return _ASR_ALIASES.get(v.lower(), f"Qwen/{v}")


# If --asr-model matches one of these size strings -> whisper-hybrid path (whisper text; units from
# align_text: by default wav2vec2 CTC for en, MMS for ja, Qwen3-ForcedAligner otherwise); else -> qwen path.
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
# Whisper defaults to config.DEFAULT_FUSION_WHISPER (large-v3); punctuation path uses 1.7B (0.6B emits no punctuation).
_FUSION_ALIASES = {"fusion", "fuse", "hybrid", "hybrid+"}


def _select_engine(name: str | None) -> tuple[str, str]:
    """--asr-model value -> (engine, resolved model id).

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


# Empty -> float16 on cuda, int8 on cpu/mps (ctranslate2/faster-whisper is CUDA-or-CPU only).
WHISPER_COMPUTE = os.environ.get("VOXWEAVE_WHISPER_COMPUTE", "")
# A 120-second dense transcript can legitimately exceed qwen-asr's 512-token
# constructor default.  Raising only the ceiling does not change ordinary greedy
# decodes (generation still stops at EOS), but avoids silent tail truncation.
QWEN_MAX_NEW_TOKENS = _env_int("VOXWEAVE_QWEN_MAX_NEW_TOKENS", 1024)
# qwen-asr #207 guard for the batched ASR pass (_asr_pass): a mixed-length batch can
# corrupt its shorter item to a lone "!", so a batched result with fewer alphanumeric
# characters than this many per second of chunk audio (empty text included) is re-run
# alone through the legacy per-chunk call. Speech in a VAD chunk yields several per
# second in every supported script, so 0.5 leaves a wide margin; chunks shorter than
# the check floor are exempt (a cough or a breath legitimately transcribes to nothing).
ASR_BATCH_MIN_CPS = _env_float("VOXWEAVE_ASR_BATCH_MIN_CPS", 0.5)
ASR_BATCH_MIN_CHECK_SEC = _env_float("VOXWEAVE_ASR_BATCH_MIN_CHECK_SEC", 2.0)

# ASR/alignment process-level singletons; call release() at end of episode.
# Separator is not kept resident (self-loads, self-releases).
_asr = None  # qwen_asr.Qwen3ASRModel
_asr_id = None  # currently loaded ASR repo id (reloaded on --asr-model change)
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


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _resolve_separator_files() -> tuple[Path, Path]:
    """Locate separator ckpt + yaml. Auto-downloads ckpt from HF if missing; falls back to bundled yaml if yaml is absent."""
    ckpt = Path(SEPARATOR_CKPT)
    if not ckpt.exists():
        log.info(
            "separator model missing, downloading from HF %s/%s (-> HF cache, subsequent runs will hit cache) ...",
            SEPARATOR_REPO,
            SEPARATOR_REPO_FILE,
        )
        fallback = (
            f"manually place weights at {SEPARATOR_CKPT} "
            f"(VOXWEAVE_SEPARATOR_CKPT|CONFIG / VOXWEAVE_SEPARATOR_REPO are configurable), "
            f"or use --no-separate"
        )
        try:
            ckpt = Path(
                _hf_download(
                    SEPARATOR_REPO, SEPARATOR_REPO_FILE, cache_dir=config.AUDIO_CACHE
                )
            )
        except RuntimeError as e:
            if isinstance(e.__cause__, ModuleNotFoundError):
                raise  # missing-dep errors are already friendly, re-raise as-is
            # _hf_download wraps every download failure in a RuntimeError (repo +
            # HF_TOKEN hint); add the separator-specific ways out
            raise RuntimeError(f"{e} -- or {fallback}") from e
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"separator model auto-download failed ({SEPARATOR_REPO}/{SEPARATOR_REPO_FILE}): {e!r} -- "
                f"check network / HF_TOKEN, or {fallback}"
            ) from e
    conf = Path(SEPARATOR_CONFIG)
    if not conf.exists():
        conf = _BUNDLED_SEPARATOR_CONFIG
    return ckpt, conf


def _effective_autocast(mode: str, device: str) -> str:
    """The autocast mode that will really wrap the forward on ``device``.

    Autocast here is CUDA-only (see :func:`_autocast_context`), so on CPU/MPS
    every mode degrades to the fp32 path and the effective mode is ``"off"``.
    The identity, the probe and the forward all resolve through this one rule,
    so the recorded numerics are the numerics that ran: a vocals cache sitting
    next to the media and shared between a CUDA host and a CPU/MPS host that
    are both configured ``bf16`` would otherwise label fp32 stems ``"bf16"``
    and reuse them across the two.
    """
    if mode not in config.SEP_AUTOCAST_MODES:
        raise ValueError(
            f"unknown separator autocast mode {mode!r} "
            f"(expected one of {'/'.join(config.SEP_AUTOCAST_MODES)})"
        )
    # get_device() strings ("cuda:0"/"cpu"/"mps") and str(torch.device) ("cuda")
    # both answer this the same way, and the model is moved to get_device(), so
    # the load edge and the forward cannot disagree about the device.
    return mode if device.startswith("cuda") else "off"


def separator_identity() -> dict[str, object]:
    """Describe the separator this run would use, for cache validation.

    Hashes the currently configured bytes and records the numerics that would
    wrap the forward on this host: ``autocast`` is part of the identity because
    bf16/fp16 change the stems, so a cache produced under one mode must not be
    reused under another (see ``vocalscache.SeparatorIdentity``). The mode is
    the effective one, not the configured one -- a host that cannot run
    autocast reports (and matches) the fp32 path it would actually take.
    """
    checkpoint, config_path = _resolve_separator_files()
    resolved_checkpoint = checkpoint.resolve(strict=True)
    resolved_config = config_path.resolve(strict=True)
    return {
        "repo": SEPARATOR_REPO,
        "file": SEPARATOR_REPO_FILE,
        "checkpoint": _sha256_file(resolved_checkpoint),
        "config_sha256": _sha256_file(resolved_config),
        "autocast": _effective_autocast(config.conf_separate_autocast(), get_device()),
    }


_FALSE_ENV_VALUES = frozenset({"0", "false", "off"})


def _tf32_enabled() -> bool:
    """TF32 matmuls are on by default; VOXWEAVE_TF32=0/false/off opts out. Read per call."""
    return (
        os.environ.get("VOXWEAVE_TF32", "").strip().casefold() not in _FALSE_ENV_VALUES
    )


def _load_separator(autocast: str | None = None):
    """Instantiate MelBandRoformer and return its load-bound content identity.

    ``autocast`` is the mode the caller will wrap the forward with; its effective
    value on this device (:func:`_effective_autocast`) is stamped into the
    identity, so the identity always describes both the bytes and the numerics of
    the vocals about to be produced. ``None`` resolves it from the config
    (env > conf > "off"), mirroring ``_demix``; :func:`separate_vocals` resolves
    it once and hands the same value to both.

    The model is not cached; the caller deletes it after separation to free VRAM.
    Uses the frozen copy from voxweave.vendor: the latest PyPI bs-roformer has
    diverged in architecture and can no longer load community ckpts. Extra yaml
    keys are filtered against __init__ signature so unknown fields don't crash.
    """
    try:
        import torch

        from voxweave.vendor.mel_band_roformer import MelBandRoformer
    except ModuleNotFoundError as e:
        raise _require(e.name or "torch") from e

    # Resolved before the checkpoint load so an unknown mode fails fast, and reduced
    # to what this device will really run: get_device() is cached, so the dev the
    # model is moved to below is the same string this was resolved against.
    mode = _effective_autocast(
        autocast if autocast is not None else config.conf_separate_autocast(),
        get_device(),
    )
    ckpt, conf = _resolve_separator_files()
    resolved_checkpoint = Path(ckpt).resolve(strict=True)
    resolved_config = Path(conf).resolve(strict=True)
    config_bytes = resolved_config.read_bytes()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    cfg = _parse_yaml(config_bytes.decode("utf-8"))
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
    # Keep one descriptor across hash/load/re-hash. Atomic pathname replacement
    # cannot redirect torch.load, while in-place mutation is detected before any
    # state is installed. weights_only=True prevents pickle RCE.
    with resolved_checkpoint.open("rb") as checkpoint_stream:
        checkpoint_digest = _sha256_stream(checkpoint_stream)
        checkpoint_stream.seek(0)
        loaded = torch.load(
            checkpoint_stream,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_stream.seek(0)
        if _sha256_stream(checkpoint_stream) != checkpoint_digest:
            raise RuntimeError("separator checkpoint changed while loading")
    sd = _strip_state_dict(loaded)
    model.load_state_dict(sd)
    dev = get_device()
    if dev.startswith("cuda") and _tf32_enabled():
        # The roformer is attention-heavy and separation dominates wall clock; TF32 matmuls
        # on Ampere+ cut that substantially and the stem difference is below the noise floor
        # of the downstream 16k ASR. VOXWEAVE_TF32=0 restores strict fp32 matmuls.
        torch.set_float32_matmul_precision("high")
    model.to(dev).eval()
    log.info("loaded separator ckpt=%s on %s", resolved_checkpoint, dev)
    separator_identity: dict[str, object] = {
        "repo": SEPARATOR_REPO,
        "file": SEPARATOR_REPO_FILE,
        "checkpoint": checkpoint_digest,
        "config_sha256": config_digest,
        # Already device-effective (see above), so this is the mode the forward
        # will really run under, not merely the configured one.
        "autocast": mode,
    }
    return model, cfg, separator_identity


def _autocast_context(dev, mode: str):
    """Context manager for the separator forward under autocast ``mode``.

    ``"off"`` returns a ``nullcontext``. ``"bf16"`` / ``"fp16"`` return a
    ``torch.autocast`` bound to CUDA when ``dev`` is a CUDA device; on any other
    device autocast is not attempted (logged once at debug level) and the fp32
    path runs unchanged. The CUDA gate and the mode check are
    :func:`_effective_autocast`'s, the same rule the recorded identity uses.
    Only the model forward is meant to run inside it -- the overlap-add
    accumulation in :func:`_demix` stays fp32 either way.
    """
    effective = _effective_autocast(mode, str(dev))
    if effective == "off":
        if mode != "off":
            log.debug("separator autocast=%s ignored on %s (CUDA only)", mode, dev)
        return nullcontext()
    import torch

    dtype = torch.bfloat16 if effective == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _demix(model, mix, cfg, progress=None, batch=None, autocast=None):
    """Chunked overlap-add inference: mix [ch, t] float32 -> vocals [ch, t].

    Hann window + >=2x overlap satisfies COLA; tail normalized by window sum. num_stems=1
    output may or may not have a stem dimension -- both shapes are handled.
    Windows are stacked `batch` at a time into one forward (default conf [batch].separate;
    1 unless configured -- batch=1 already saturates 8 GB-class GPUs, see config._BATCH_DEFAULTS).
    `autocast` ("off" | "bf16" | "fp16"; default conf [separate].autocast, "off" unless
    configured) wraps only the model forward, and only on CUDA -- see _autocast_context.
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
    mode = autocast if autocast is not None else config.conf_separate_autocast()
    forward_ctx = _autocast_context(dev, mode)
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
            with forward_ctx:
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


@overload
def separate_vocals(
    audio_path: Path,
    *,
    progress=None,
    return_identity: Literal[False] = False,
) -> Path: ...


@overload
def separate_vocals(
    audio_path: Path,
    *,
    progress=None,
    return_identity: Literal[True],
) -> tuple[Path, dict[str, object]]: ...


def separate_vocals(
    audio_path: Path,
    *,
    progress=None,
    return_identity: bool = False,
) -> Path | tuple[Path, dict[str, object]]:
    """Separate vocals locally; return a FLAC temp file (caller deletes).

    Input must be full-band 44.1k stereo -- Roformer was trained at 44.1k; feeding 16k degrades quality badly.
    Model is loaded and freed within this function so it doesn't co-occupy VRAM with ASR/alignment.
    progress(done, total) called per demix window if provided.
    ``return_identity=True`` also returns the identity produced at the model load
    edge, for capture provenance and vocals-cache validation. The autocast mode is
    resolved exactly once here and handed to both the identity and the forward,
    and both reduce it to its effective value the same way, so the recorded
    numerics can never disagree with the numerics that ran.
    """
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ModuleNotFoundError as e:
        raise _require(e.name or "torch") from e

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)  # [t, ch]
    if data.shape[1] > 2:
        # The Roformer is a stereo model and has no downmix of its own; refuse
        # before loading it. decode_to_wav(mono=False) already caps at stereo, so
        # this only trips on a caller that bypassed it.
        raise ValueError(
            f"vocal separation needs mono or stereo audio, but {Path(audio_path).name} "
            f"has {data.shape[1]} channels; decode it with "
            "chunking.decode_to_wav(mono=False), which downmixes to stereo"
        )
    mix = torch.from_numpy(data.T.copy())  # [ch, t]
    if mix.shape[0] == 1:  # mono -> duplicate to stereo for the stereo model
        mix = mix.repeat(2, 1)

    autocast = config.conf_separate_autocast()
    model, cfg, separator_identity = _load_separator(autocast=autocast)
    try:
        # vocals: [ch, t]
        vocals = _demix(model, mix, cfg, progress=progress, autocast=autocast)
    finally:
        del model
        _empty_cache()

    fd, dst = tempfile.mkstemp(suffix=".flac", prefix="voxweave_vocals_")
    os.close(fd)
    out = Path(dst)
    try:
        sf.write(str(out), np.asarray(vocals.T), sr, format="FLAC")  # [t, ch]
    except BaseException:
        # a failed write (disk full, ^C) must not leak the temp file
        out.unlink(missing_ok=True)
        raise
    if return_identity:
        return out, separator_identity
    return out


# ───────────────────────────── ASR + forced alignment (singletons, released via release()) ─────────────────


def _get_asr(asr_model: str | None = None):
    """Lazy-load Qwen3ASRModel singleton (ASR-only, no forced aligner attached).

    Timestamps come from align_text; on ja/en paths the Qwen aligner is never loaded, saving VRAM.
    dtype kwarg (not torch_dtype) reflects transformers 4.57.6 API.
    Reloads if the model repo changes.
    qwen_asr's max_inference_batch_size mirrors config ``[batch].asr`` so it never
    sub-splits the chunk groups _asr_pass hands it.
    """
    if (
        _use_mlx()
    ):  # Apple Silicon: serve ASR from the native MLX Qwen3-ASR (see backend_mlx)
        from voxweave import backend_mlx

        return backend_mlx.get_asr(resolve_asr_model(asr_model))
    global _asr, _asr_id
    mid = resolve_asr_model(asr_model)
    batch = config.conf_batch("asr")
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
        params = inspect.signature(Qwen3ASRModel.from_pretrained).parameters
        if "max_new_tokens" in params:
            load_kwargs["max_new_tokens"] = QWEN_MAX_NEW_TOKENS
        if "max_inference_batch_size" in params:
            load_kwargs["max_inference_batch_size"] = batch
        _asr = Qwen3ASRModel.from_pretrained(local, **load_kwargs)
        _asr_id = mid
        log.info("ASR ready")
    if hasattr(_asr, "max_inference_batch_size"):  # already loaded / older signature
        _asr.max_inference_batch_size = batch
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


def _engine_language(engine: str, language: str | None) -> str | None:
    """--language value -> the spelling ``engine``'s ASR call takes; None/blank -> None (auto-detect).

    Qwen3-ASR (torch and the MLX adapter) gets the capitalized English name: qwen_asr
    validates exactly that form, so a raw ISO code failed every chunk ("Unsupported
    language: Ja"), and mlx-audio matches the same names case-insensitively. Whisper gets
    the ISO code. Either spelling is accepted for every engine; an unknown value raises
    ValueError naming the supported set (transcribe_chunks/transcribe_align check this
    before any model loads, so a typo cannot fail every chunk after a whole ASR pass).
    """
    if not language or not language.strip():
        return None
    from voxweave.lang import to_asr_iso, to_asr_name

    if engine == "whisper":
        iso = to_asr_iso(language)
        # whisper before large-v3 has no Cantonese token (v3 added "yue"); zh is safe on
        # every size, and alignment still uses yue downstream
        return "zh" if iso == "yue" else iso
    return to_asr_name(language)


def _qwen_asr_kwargs(language: str | None, context: str | None) -> dict:
    """Keyword arguments for one Qwen3ASRModel.transcribe call (single path or batch).

    ASR-only (timestamps come from align_text); the context kwarg is omitted entirely
    when empty to preserve legacy behavior (older qwen-asr lacks the parameter).
    """
    kwargs: dict = {
        "language": _engine_language("qwen", language),
        "return_time_stamps": False,
    }
    if context:
        kwargs["context"] = format_qwen_context(context)
    return kwargs


def _asr_postprocess(
    raw_det: str | None,
    raw_text: str,
    language: str | None,
    src: str,
) -> tuple[str | None, str, str]:
    """One chunk's raw engine output -> (effective language, text, align_lang).

    Shared by the per-chunk (_asr_only) and batched (_asr_pass) paths so both yield
    identical results: terminal generation loops are collapsed (stabilize_asr_text),
    the detected label is reconciled with the transcript script (an explicit
    ``language`` always wins) and align_lang is pre-computed, falling back to 'en'
    for empty text (skipped by the alignment pass anyway). ``src`` is for log
    wording only.
    """
    from voxweave.lang import reconcile_detected_language

    text = stabilize_asr_text(raw_text)
    if len(text) < len(raw_text.strip()):
        log.warning(
            "%s removed a repeated generation tail (%d -> %d characters)",
            src,
            len(raw_text.strip()),
            len(text),
        )
    det = reconcile_detected_language(raw_det, text, override=language)
    if (
        not language
        and raw_det
        and det
        and det.casefold() != raw_det.strip().casefold()
    ):
        log.info(
            "%s language %r reconciled to %r from transcript script",
            src,
            raw_det,
            det,
        )
    align_lang = _resolve_align_lang(det, src) if text.strip() else "en"
    return det, text, align_lang


def _asr_only(
    engine: str,
    wav_path: Path,
    language: str | None,
    model_id: str,
    context: str | None,
) -> tuple[str | None, str, str]:
    """Transcribe only: return (effective language, punctuated text, align_lang).

    First pass of the two-pass peak strategy: alignment deferred to pass two after ASR is released.
    The raw engine output goes through _asr_postprocess (language reconciliation,
    align_lang), the same step the batched Qwen pass applies per result.
    """
    if engine == "whisper":
        model = _get_whisper(model_id)
        segments, info = model.transcribe(
            str(wav_path),
            language=_engine_language("whisper", language),
            initial_prompt=context or None,
            hotwords=whisper_hotwords(context),
            condition_on_previous_text=False,  # prevents repetition hallucination
            vad_filter=False,  # VAD chunking already done upstream
            word_timestamps=False,  # timestamps come from align_text, not whisper
        )
        raw_text = "".join(s.text for s in segments)  # segments is a generator
        raw_det = info.language
        src = "whisper"
    else:  # qwen
        model = _get_asr(model_id)
        r = model.transcribe(str(wav_path), **_qwen_asr_kwargs(language, context))[0]
        raw_det = r.language or None
        raw_text = r.text
        src = "ASR"
    return _asr_postprocess(raw_det, raw_text, language, src)


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
    """Whisper text + align_text units, single chunk. Same (lang, text, units) contract as qwen path."""
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
        text_w, qwen_punct, strip_existing=iso not in NO_SPACE_LANGS
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


def chunk_pass_count(asr_model: str | None) -> int:
    """Passes per chunk for transcribe_chunks: fusion=3, others=2.

    The pass structure is identical for both load strategies; "sum" merely keeps the
    singletons resident between passes (peak VRAM = sum of models) instead of releasing
    them (peak = max), so the count never depends on the load strategy.
    """
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
    run, so hours of prior chunks are not thrown away. This is the per-chunk unit
    of _asr_pass: batch size 1, whisper and MLX go through it directly; a batched
    Qwen group whose call raised is redone chunk by chunk here, as is a single
    chunk whose batched result failed the qwen-asr #207 content floor."""
    try:
        return _asr_only(engine, wav, language, model_id, context)
    except Exception as e:  # noqa: BLE001 -- one bad chunk must not kill the run
        return _asr_failed(e, idx, total, failures)


def _asr_failed(
    e: Exception, idx: int, total: int, failures: list[Exception]
) -> tuple[None, str, str]:
    """Record one chunk's ASR failure and hand back the empty-text placeholder."""
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


def _wav_duration(wav: Path) -> float:
    """Duration of a wav in seconds from its header; 0.0 if it cannot be read."""
    import soundfile as sf

    try:
        info = sf.info(str(wav))
        return float(info.frames) / float(info.samplerate)
    except Exception as e:  # noqa: BLE001 -- header unreadable is not worth killing a chunk over
        log.debug("could not read duration of %s (%s: %s)", wav, type(e).__name__, e)
        return 0.0


def _fallback_units(text: str, lang: str, duration: float) -> list[dict]:
    """Units [{text,start,end}] tiling [0, duration] evenly, for text that could not be aligned.

    Alignment refines timing; it is not the source of the transcript. When it fails, the ASR
    words are still the best output available, so they degrade to a uniform tiling of the
    chunk rather than disappearing. Tokenization matches what the real aligners emit:
    whitespace-separated words for spaced languages, single characters otherwise.
    """
    from voxweave.lang import to_iso_or

    if to_iso_or(lang, "") in LANGUAGES_WITHOUT_SPACES:
        tokens = [ch for ch in text if not ch.isspace()]
    else:
        tokens = text.split()
    if not tokens:
        return []
    step = duration / len(tokens)
    units = [
        {"text": t, "start": i * step, "end": (i + 1) * step}
        for i, t in enumerate(tokens)
    ]
    units[-1]["end"] = duration  # exact end, free of accumulated float drift
    return units


def _align_chunk_safe(
    wav: Path, text: str, align_lang: str, idx: int, total: int
) -> list[dict]:
    """One chunk's per-chunk alignment with failure containment: the transcript
    text survives, only its word timing is lost (evenly spread over the chunk)."""
    try:
        return align_text(wav, text, align_lang)
    except Exception as e:  # noqa: BLE001 -- one bad chunk must not kill the run
        log.warning(
            "alignment failed on chunk %d/%d (%s: %s); keeping text with evenly spread timing",
            idx + 1,
            total,
            type(e).__name__,
            e,
        )
        _empty_cache()
        return _fallback_units(text, align_lang, _wav_duration(wav))


def _raise_if_all_failed(failures: list[Exception], total: int) -> None:
    """Every chunk erroring is a broken run, not a silent empty transcript."""
    if total and len(failures) >= total:
        e = failures[-1]
        raise RuntimeError(
            f"ASR failed on all {total} chunks (last error: {type(e).__name__}: {e})"
        )


def _asr_batch_size(engine: str) -> int:
    """Chunks per ASR call in _asr_pass; 1 = the legacy per-chunk call.

    Only the torch qwen_asr backend takes a list of audio paths. Whisper and the
    MLX Qwen adapter (backend_mlx._MlxAsr.transcribe) take one path per call, so
    they stay per-chunk whatever ``[batch].asr`` says.
    """
    if engine != "qwen" or _use_mlx():
        return 1
    return config.conf_batch("asr")


def _batched_asr_suspect(text: str, duration: float) -> bool:
    """True when a batched result is implausibly short for a non-silent chunk.

    qwen-asr #207: a mixed-length batch can corrupt its shorter item to a lone "!".
    Alphanumeric content below ASR_BATCH_MIN_CPS per second of audio (empty text
    included) is not accepted. Chunks shorter than ASR_BATCH_MIN_CHECK_SEC, or whose
    duration is unknown (0.0), are exempt: little or no text is legitimate there.
    """
    from voxweave.lang import transcript_content_weight

    if duration < ASR_BATCH_MIN_CHECK_SEC:
        return False
    return transcript_content_weight(text) < ASR_BATCH_MIN_CPS * duration


def _asr_pass(
    engine: str,
    wav_paths: list[Path],
    language: str | None,
    model_id: str,
    context: str | None,
    failures: list[Exception],
    tick: Callable[[], None],
) -> list[tuple[str | None, str, str]]:
    """All-chunks ASR pass -> [(det_lang, text, align_lang)] in input order.

    ``tick`` fires once per chunk after its result is in (per-chunk progress total
    unchanged; on the batched path ticks follow duration order, not chunk order).
    Batch size 1 (the default), a group of one, whisper and MLX make the legacy
    single-path call through _asr_chunk_safe. On the torch Qwen backend with
    ``[batch].asr`` > 1 the chunks are grouped by duration into batches of that
    size and every group is ONE model.transcribe(list) call. Batching is opt-in
    because its output is not byte-identical to batch 1 (bf16 batched kernels
    drift ~1.5% CER; numbers in transcribe_chunks). Duration sorting bought no
    speed in the A/B but keeps each group's lengths close, which minimises the
    qwen-asr #207 exposure (a mixed-length batch can corrupt its shorter item to
    a lone "!"); against what slips through, a batched result that is empty or
    below the ASR_BATCH_MIN_CPS content floor for a chunk of at least
    ASR_BATCH_MIN_CHECK_SEC (_batched_asr_suspect) is not accepted: that chunk
    alone is re-run through the legacy per-chunk call. A group whose batched call
    raises (or returns the wrong count) is redone chunk by chunk through
    _asr_chunk_safe, so one poisoned chunk degrades alone and its failure is
    recorded exactly as on the per-chunk path.
    """
    n = len(wav_paths)
    out: list[tuple[str | None, str, str]] = [(None, "", "")] * n

    def _per_chunk(indices: Sequence[int]) -> None:
        for i in indices:
            out[i] = _asr_chunk_safe(
                engine, wav_paths[i], language, model_id, context, i, n, failures
            )
            tick()

    batch = _asr_batch_size(engine)
    if batch <= 1:
        _per_chunk(range(n))
        return out
    durations = [_wav_duration(p) for p in wav_paths]
    order = sorted(range(n), key=durations.__getitem__)
    kwargs = _qwen_asr_kwargs(language, context)
    for start in range(0, n, batch):
        group = order[start : start + batch]
        if len(group) == 1:  # nothing to pad against: legacy single-path call
            _per_chunk(group)
            continue
        try:
            # MLX is excluded by _asr_batch_size, so this is the torch Qwen3ASRModel,
            # whose transcribe() decodes a list of paths as one padded batch
            model: Any = _get_asr(model_id)
            results = model.transcribe([str(wav_paths[i]) for i in group], **kwargs)
            if len(results) != len(group):
                raise RuntimeError(
                    f"batched ASR returned {len(results)} results for {len(group)} chunks"
                )
        except Exception as e:  # noqa: BLE001 -- redo the group per chunk so one bad chunk degrades alone
            log.warning(
                "batched ASR failed on chunks %s of %d (%s: %s); retrying them one by one",
                ", ".join(str(i + 1) for i in sorted(group)),
                n,
                type(e).__name__,
                e,
            )
            _empty_cache()
            _per_chunk(sorted(group))
            continue
        for i, r in zip(group, results):
            raw_text = r.text or ""
            if _batched_asr_suspect(raw_text, durations[i]):
                log.warning(
                    "batched ASR returned %d alphanumeric characters for chunk %d/%d "
                    "(%.1fs of audio, floor %.0f); re-running it alone (qwen-asr #207)",
                    sum(ch.isalnum() for ch in raw_text),
                    i + 1,
                    n,
                    durations[i],
                    ASR_BATCH_MIN_CPS * durations[i],
                )
                _per_chunk([i])
                continue
            try:
                out[i] = _asr_postprocess(r.language or None, raw_text, language, "ASR")
            except Exception as e:  # noqa: BLE001 -- same containment as _asr_chunk_safe
                out[i] = _asr_failed(e, i, n, failures)
            tick()
    return out


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
    whisper ASR -> Qwen ASR -> align + merge). Each ASR pass is _asr_pass: with config
    ``[batch].asr`` > 1 (opt-in; default 1) the torch Qwen engine decodes chunks in
    duration-sorted groups of that size, one model.transcribe(list) call per group,
    results restored to input order. Measured on an RTX PRO 4000 (Qwen3-ASR-1.7B,
    greedy, 24-min episode): batch 4 = 1.34x faster at 6.4 GiB peak, batch 8 = 1.49x
    at 8.9 GiB, but batched transcripts drift ~1.5% CER from batch 1 (bf16 batched
    kernels; scattered small edits, no chunk lost), and qwen-asr #207 reports
    mixed-length batches corrupting the shorter item, which _asr_pass guards with a
    per-chunk re-run. Whisper, MLX and batch size 1 transcribe one chunk per call.
    strategy controls model residency between passes (config.conf_load_strategy):
    - "peak" (default): singletons released between passes; peak VRAM = max(models).
    - "sum": singletons stay resident across passes; peak = sum(models); saves the
      release/reload overhead on high-VRAM cards.

    ``full_wav`` + ``bounds`` (absolute chunk windows on full_wav) enable ONE full-file
    alignment pass for CTC/MMS file-level languages (see _full_pass_units); Qwen-aligned
    languages and callers that omit them keep per-chunk alignment. on_done(i) is called
    per chunk per completed pass (a batched chunk ticks once its group has returned,
    so batched ticks follow duration order, not chunk order); total = N *
    chunk_pass_count(). Aligner kept alive until release().
    """
    counter = [0]

    def _tick() -> None:
        if on_done:
            on_done(counter[0])
        counter[0] += 1

    release_between_passes = strategy != "sum"  # sum keeps singletons co-resident
    engine, mid = _select_engine(asr_model)
    # Fail fast on an unknown --language, before any model loads: inside the ASR pass
    # it would fail every chunk and only surface after the whole pass (whisper and Qwen
    # accept the same set, so this one check covers both fusion passes).
    _engine_language(engine, language)
    if engine == "fusion":
        qid = resolve_asr_model(config.conf_fusion_qwen())
        fusion_whisper = config.conf_fusion_whisper()
        n = len(wav_paths)
        # pass A: whisper ASR all chunks (always per-chunk: whisper takes one path)
        w_fail: list[Exception] = []
        w_asr = _asr_pass(
            "whisper", wav_paths, language, fusion_whisper, context, w_fail, _tick
        )
        if release_between_passes:
            _release_whisper()
        # pass B: Qwen ASR all chunks (batched on the torch backend)
        q_fail: list[Exception] = []
        q_asr = _asr_pass("qwen", wav_paths, language, qid, context, q_fail, _tick)
        if release_between_passes:
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
    # (det_lang, text, align_lang) per chunk, in input order
    asr_out = _asr_pass(engine, wav_paths, language, mid, context, failures, _tick)
    if release_between_passes:
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
    # an unknown --language fails here, before any model loads
    _engine_language(engine, language)
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
