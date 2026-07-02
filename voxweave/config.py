"""User configuration ``~/.config/voxweave.conf`` (TOML).

Precedence: CLI options > environment variables > config file > built-in defaults.
Pure stdlib (``tomllib`` 3.11+), no torch dependency. Values are string names;
the backend resolves them to torchaudio bundles / HF ids / Qwen models.
On first CLI run, :func:`ensure_default_config` writes a commented template.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

log = logging.getLogger("voxweave")

# Built-in defaults (mirrors backend.ASR_MODEL / FUSION_*).
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
# Dual-ASR fusion (--hybrid): whisper supplies text, Qwen supplies punctuation.
# 0.6B emits no punctuation, so fusion must use 1.7B.
DEFAULT_FUSION_WHISPER = "large-v3"
DEFAULT_FUSION_QWEN = "Qwen/Qwen3-ASR-1.7B"
# Per-language aligner defaults. Unlisted languages fall back to Qwen3-ForcedAligner.
#
# en: facebook/wav2vec2-large-960h-lv60-self loaded via HF (same LV60K-self weights as
#     the torchaudio bundle), per-cue crop alignment (<=120s). Torchaudio bundle name
#     also accepted for back-compat.
#
# ja: "mms" = MMS-300m ONNX + uroman, full-file single pass (align_blocks_full_mms,
#     equivalent to whisperx fork align_ctc). Full-file is required: per-cue cropping
#     causes coarse routing errors that drift timestamps — empirically misplaced
#     エルダドワーフ by 11s. xlsr full-file is O(T²) so 23 min → ~300 GB OOM; only
#     MMS (ctc-forced-aligner internal windowing) can handle full-file safely.
#     Note: MMS needs onnxruntime-gpu; CPU ort has the same package name and silently
#     drops CUDAExecutionProvider — see pyproject [tool.uv] override-dependencies.
DEFAULT_ALIGN_MODELS = {"en": "facebook/wav2vec2-large-960h-lv60-self", "ja": "mms"}

# Model cache layout: all weights go under VOXWEAVE_CACHE_ROOT (~/.cache/voxweave),
# split by role (asr / align / audio). Each subdir is a self-contained HF hub tree
# passed as cache_dir/download_root= to every model download.
# Weights do NOT share with ~/.cache/huggingface/hub — a model already pulled by
# `hf download` will re-download here on first run. This isolation is intentional:
# it keeps the voxweave weight set self-contained for packaging/migration/deletion.
CACHE_ROOT = Path(
    os.environ.get("VOXWEAVE_CACHE_ROOT", str(Path.home() / ".cache" / "voxweave"))
).expanduser()
ASR_CACHE = str(CACHE_ROOT / "asr")  # Qwen ASR, faster-whisper
ALIGN_CACHE = str(CACHE_ROOT / "align")  # Qwen aligner, en wav2vec2, ja MMS onnx
AUDIO_CACHE = str(CACHE_ROOT / "audio")  # separator roformer, songdet PANNs

_TEMPLATE = """\
# voxweave configuration  (~/.config/voxweave.conf)
# Precedence: CLI options > environment variables > this file > built-in defaults.
# Remove a line to revert to the built-in default.

# Default ASR model (= --model / env VOXWEAVE_ASR_MODEL); short name qwen3-asr-1.7B or full HF id.
# Special value "hybrid" (= CLI --hybrid) -> dual-ASR fusion (whisper text quality + Qwen punctuation).
# asr_model = "Qwen/Qwen3-ASR-0.6B"

# CTC forced-align single-pass DP frame budget (= env VOXWEAVE_CTC_MAX_DP_FRAMES). Long audio
# (movies) whose emission exceeds this is auto-split at silence anchors before the O(T*L) DP.
# Default 90000 (~30min at 50fps). Bigger = fewer/larger chunks = more accurate on long
# sparse-dialogue audio (the global DP keeps more context), at higher GPU memory; e.g. 150000
# (~40min chunks) measurably tightens movie alignment on a 24 GB card. Lower it on small cards.
# ctc_max_dp_frames = 90000

# Model load strategy (= env VOXWEAVE_LOAD_STRATEGY):
#   peak (default) = serial peak-shaving: all-chunk ASR -> release -> all-chunk align;
#                    ASR and aligner never co-reside; peak VRAM = max(each model); works on 8 GB cards.
#   sum            = concurrent co-residence: per-chunk ASR+align in one pass;
#                    peak VRAM = sum(models); saves two swap round-trips on large-VRAM cards.
# load_strategy = "peak"

# dual-ASR fusion sub-models (= CLI --hybrid; env VOXWEAVE_FUSION_WHISPER / VOXWEAVE_FUSION_QWEN).
# whisper supplies accurate text, Qwen supplies punctuation positions (merged on a shared timeline).
#   whisper = faster-whisper size: large-v3 (highest quality) | large-v3-turbo (~5x faster, default).
#   qwen    = punctuation model; must emit punctuation so 1.7B (not 0.6B).
[fusion]
# whisper = "large-v3-turbo"
# qwen = "Qwen/Qwen3-ASR-1.7B"

# Inference batch sizes: windows per GPU forward pass (= env VOXWEAVE_SEP_BATCH /
# VOXWEAVE_CTC_BATCH / VOXWEAVE_MMS_BATCH). On an 8 GB-class card batch=1 already
# saturates compute (measured: no speedup at 2/4, just +~0.8 GiB VRAM per extra
# separation window) -- only worth raising on much wider GPUs, and only after measuring.
[batch]
# separate = 1   # vocal separation (MelBandRoformer) 8s windows
# ctc = 1        # wav2vec2 CTC emission 30s windows (en aligner)
# mms = 4        # MMS-300m emission batch (ja aligner, ctc-forced-aligner generate_emissions)

# Per-language forced-alignment models; unlisted languages use Qwen3-ForcedAligner (built-in default).
# Values: "mms" (MMS-300m + uroman, full-file single pass; bundled in core) |
#         HF wav2vec2 id (downloaded to the voxweave align cache ~/.cache/voxweave/align) | torchaudio bundle name (-> torch.hub cache).
# Set to "" to explicitly fall back to Qwen.  "mms" uses the full-file pass (immune to per-cue drift);
# all other values use per-cue crop alignment.
[align]
en = "facebook/wav2vec2-large-960h-lv60-self"   # English: large wav2vec2 CTC per-cue, HF path -> ~/.cache/voxweave/align (same LV60K-self weights as torchaudio WAV2VEC2_ASR_LARGE_LV60K_960H bundle)
ja = "mms"                             # Japanese: MMS-300m + uroman full-file single pass (= whisperx fork align_ctc; gold standard); requires onnxruntime-gpu for CUDA (bundled in core)
# ja = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"   # Alternative: xlsr character-level CTC per-cue (full-file single pass is O(T^2) -> OOM)
# zh = "mms"   # Chinese can also use MMS full-file pass; default is Qwen (native CJK character-level)
"""


def config_path() -> Path:
    """Config file path: ``VOXWEAVE_CONFIG`` env if set, else ``~/.config/voxweave.conf``."""
    env = os.environ.get("VOXWEAVE_CONFIG")
    return Path(env) if env else Path.home() / ".config" / "voxweave.conf"


# Recognized top-level keys; anything else in the file is a typo/stale setting.
_KNOWN_KEYS = frozenset(
    {
        "asr_model",
        "ctc_max_dp_frames",
        "load_strategy",
        "hf_token",
        "fusion",
        "batch",
        "align",
    }
)


def _load() -> dict:
    """Parse the config TOML. Missing or malformed file returns {} (no crash).
    Unknown top-level keys are warned about but do not stop known keys loading."""
    p = config_path()
    if not p.exists():
        return {}
    try:
        with p.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        log.warning("config %s read failed (%r), treating as empty", p, e)
        return {}
    for key in data:
        if key not in _KNOWN_KEYS:
            log.warning("unknown config key %r in %s (ignored)", key, p)
    return data


# Legacy config path from when the tool was named "qsub"; auto-migrated on first run.
_LEGACY_CONFIG = Path.home() / ".config" / "qsub.conf"


def ensure_default_config() -> None:
    """Write the default template on first run; no-op if the file already exists.

    If ``~/.config/qsub.conf`` exists (pre-rename legacy config) and
    ``VOXWEAVE_CONFIG`` is not set, migrate it in place instead of writing a fresh
    template.
    """
    p = config_path()
    if p.exists():
        return
    if not os.environ.get("VOXWEAVE_CONFIG") and _LEGACY_CONFIG.exists():
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            _LEGACY_CONFIG.rename(p)
            log.info("migrated legacy config %s -> %s", _LEGACY_CONFIG, p)
            return
        except OSError as e:
            log.warning(
                "could not migrate legacy config %s (%r), writing default",
                _LEGACY_CONFIG,
                e,
            )
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_TEMPLATE, encoding="utf-8")
        log.info("created default config %s", p)
    except OSError as e:
        log.warning("could not create default config %s (%r), ignoring", p, e)


def _nonempty_str(v: object) -> str | None:
    """Return v if it is a non-blank string, else None (config values may be missing/blank/non-str)."""
    return v if isinstance(v, str) and v.strip() else None


def conf_asr_model() -> str | None:
    """ASR model from config; None if unset (caller falls back to env/built-in)."""
    v = _load().get("asr_model")
    if v is not None and not isinstance(v, str):
        log.warning(
            "config key %r has wrong type (expected string), ignoring", "asr_model"
        )
        return None
    return _nonempty_str(v)


def _conf_fusion(key: str) -> str | None:
    """``[fusion].<key>`` from config, or None if absent/empty."""
    fusion = _load().get("fusion")
    if isinstance(fusion, dict):
        return _nonempty_str(fusion.get(key))
    return None


def conf_fusion_whisper() -> str:
    """Fusion whisper sub-model (faster-whisper size string).
    Precedence: env VOXWEAVE_FUSION_WHISPER > conf [fusion].whisper > default."""
    v = os.environ.get("VOXWEAVE_FUSION_WHISPER") or _conf_fusion("whisper")
    return _nonempty_str(v) or DEFAULT_FUSION_WHISPER


def conf_fusion_qwen() -> str:
    """Fusion Qwen punctuation sub-model (must be 1.7B — 0.6B emits no punctuation).
    Precedence: env VOXWEAVE_FUSION_QWEN > conf [fusion].qwen > default."""
    v = os.environ.get("VOXWEAVE_FUSION_QWEN") or _conf_fusion("qwen")
    return _nonempty_str(v) or DEFAULT_FUSION_QWEN


def conf_hf_token() -> str | None:
    """Hugging Face token for gated checkpoints (pyannote diarization).
    Precedence: env VOXWEAVE_HF_TOKEN > HF_TOKEN > HUGGING_FACE_HUB_TOKEN >
    conf ``hf_token``; None when nowhere set."""
    for key in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = _nonempty_str(os.environ.get(key))
        if v:
            return v
    return _nonempty_str(_load().get("hf_token"))


_LOAD_STRATEGIES = ("peak", "sum")


def conf_load_strategy() -> str:
    """Model load strategy.

    - ``"peak"`` (default): serial peak-shaving — ASR and aligner never co-reside;
      peak VRAM = max(each model). Works on 8 GB cards.
    - ``"sum"``: concurrent co-residence — per-chunk ASR+align; peak VRAM = sum(models).
      Saves swap overhead on large-VRAM cards.

    Precedence: env VOXWEAVE_LOAD_STRATEGY > conf load_strategy > "peak". Invalid
    values fall back to "peak".
    """
    v = os.environ.get("VOXWEAVE_LOAD_STRATEGY") or _load().get("load_strategy")
    v = v.strip().lower() if isinstance(v, str) else ""
    return v if v in _LOAD_STRATEGIES else "peak"


# CTC forced-align single-pass DP frame budget. Audio whose emission exceeds this is split at
# silence anchors (chunking.plan_dp_chunks) before the O(T*L) DP; ~30min default (50fps*60*30).
# Bigger = fewer/larger chunks = more accurate on long sparse-dialogue audio (the global DP gains
# context), at higher GPU memory. Lower on small cards.
_CTC_MAX_DP_FRAMES_DEFAULT = 90000


def conf_ctc_max_dp_frames() -> int:
    """Max emission frames for one CTC forced-align DP before silence-anchored chunking kicks in.

    Precedence: env VOXWEAVE_CTC_MAX_DP_FRAMES > conf ``ctc_max_dp_frames`` > 90000 (~30min).
    Non-integer values (env or file) are ignored and fall through to the next source.
    """
    env = os.environ.get("VOXWEAVE_CTC_MAX_DP_FRAMES")
    if env is not None and env.strip():
        try:
            return int(env)
        except ValueError:
            pass
    v = _load().get("ctc_max_dp_frames")
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if v is not None:
        log.warning(
            "config key %r has wrong type (expected integer), using default",
            "ctc_max_dp_frames",
        )
    return _CTC_MAX_DP_FRAMES_DEFAULT


# Inference batch sizes (windows per GPU forward). Defaults = 1: measured on an RTX 4070
# Laptop (8 GB), separation batch=1 already saturates compute (steady-state latency scales
# linearly with batch; same for the wav2vec2 CTC emission), so batching only costs VRAM
# (~+0.8 GiB per extra separation window). The knob exists for much wider GPUs, where
# per-window kernels may underfill the SMs — measure before raising. mms=4 is the
# ctc-forced-aligner upstream default (ONNX path, pre-existing behavior).
_BATCH_DEFAULTS = {"separate": 1, "ctc": 1, "mms": 4}
_BATCH_ENV = {
    "separate": "VOXWEAVE_SEP_BATCH",
    "ctc": "VOXWEAVE_CTC_BATCH",
    "mms": "VOXWEAVE_MMS_BATCH",  # pre-[batch] env name, kept for back-compat
}


def conf_batch(key: str) -> int:
    """Inference batch size for stage ``key`` ("separate" | "ctc" | "mms"), min 1.

    Precedence: env _BATCH_ENV[key] > conf ``[batch].<key>`` > _BATCH_DEFAULTS.
    Non-integer values (env or file) are ignored and fall through to the next source.
    """
    env = os.environ.get(_BATCH_ENV[key])
    if env is not None and env.strip():
        try:
            return max(1, int(env))
        except ValueError:
            pass
    batch = _load().get("batch")
    if isinstance(batch, dict):
        v = batch.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            return max(1, v)
    return _BATCH_DEFAULTS[key]


def align_model_for(iso: str) -> str | None:
    """Forced-alignment model for ``iso``. None → use Qwen default.

    Config ``[align]`` takes precedence (set to "" to explicitly revert to Qwen);
    falls back to DEFAULT_ALIGN_MODELS, then None.
    """
    align = _load().get("align")
    if isinstance(align, dict) and iso in align:
        v = align[iso]
        if isinstance(v, str):
            return _nonempty_str(v)  # "" = explicit disable (fall back to Qwen)
        log.warning(
            "config [align].%s has wrong type (expected string), using default", iso
        )
    return DEFAULT_ALIGN_MODELS.get(iso)


# Gap-aware segmentation thresholds (env > built-in).
_JA_GAP_MULT = 1.4  # ja inter-sentence gaps run larger; scale clause_ms and offline_ms
_GAP_DEFAULTS = {"clause_ms": 400, "vad_skip_ms": 1000, "offline_ms": 700}
_MIN_CUE_DEFAULT = 0.5
_MAX_CUE_DEFAULT = 7.0
_MIN_CUE_CEIL = 5.0 / 6.0  # Netflix floor: never require longer than 5/6s
_GLUE_GAP_DEFAULT_MS = 300  # lone-word flicker cue glues back if gap < this (0=off); < clause_ms so real pauses never merge
# Reading-speed linger targets (non-space chars/sec): a cue whose natural span is
# shorter than chars/cps extends into the following gap (capped in smart_split).
# These are linger targets for flash cues, not display-rate enforcement — verbatim
# text cannot be slowed below the speech rate.
_CPS_DEFAULTS = {"ja": 7.0, "zh": 9.0, "ko": 9.0}
_CPS_LATIN_DEFAULT = 17.0  # ~Netflix 20 cps incl. spaces, measured without spaces
_LAG_OUT_DEFAULT_MS = 250  # flat tail pad after speech ends (0=off)
_SHOT_SNAP_DEFAULT_MS = (
    458  # shot-change pairing window: 11 frames @24fps, the outermost Netflix
    # adjustment zone; boundaries past it are left alone (0=off)
)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v is not None and v.strip() else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v is not None and v.strip() else default
    except ValueError:
        return default


def gap_thresholds(iso: str) -> dict[str, int | float]:
    """Gap/duration thresholds for ``iso``. ja gets _JA_GAP_MULT on clause/offline.
    ``min_cue_s`` is clamped to <=5/6s (Netflix floor)."""
    mult = _JA_GAP_MULT if iso == "ja" else 1.0
    clause = _env_int(
        "VOXWEAVE_GAP_CLAUSE_MS", round(_GAP_DEFAULTS["clause_ms"] * mult)
    )
    skip = _env_int("VOXWEAVE_GAP_VAD_SKIP_MS", _GAP_DEFAULTS["vad_skip_ms"])
    offline = _env_int(
        "VOXWEAVE_GAP_OFFLINE_MS", round(_GAP_DEFAULTS["offline_ms"] * mult)
    )
    min_cue = min(
        _env_float("VOXWEAVE_SEG_MIN_CUE_SEC", _MIN_CUE_DEFAULT), _MIN_CUE_CEIL
    )  # distinct from pipeline's VOXWEAVE_MIN_CUE_SEC (align-stage floor); orthogonal knobs
    max_cue = _env_float(
        "VOXWEAVE_MAX_CUE_SEC", _MAX_CUE_DEFAULT
    )  # intentionally unclamped
    glue_gap = _env_int("VOXWEAVE_GLUE_GAP_MS", _GLUE_GAP_DEFAULT_MS) / 1000.0
    cps = _env_float("VOXWEAVE_CPS", _CPS_DEFAULTS.get(iso, _CPS_LATIN_DEFAULT))
    lag_out = _env_int("VOXWEAVE_LAG_OUT_MS", _LAG_OUT_DEFAULT_MS) / 1000.0
    shot_snap = _env_int("VOXWEAVE_SHOT_SNAP_MS", _SHOT_SNAP_DEFAULT_MS) / 1000.0
    return {
        "clause_ms": clause,
        "vad_skip_ms": skip,
        "offline_ms": offline,
        "min_cue_s": min_cue,
        "max_cue_s": max_cue,
        "glue_gap_s": glue_gap,
        "cps": cps,
        "lag_out_s": lag_out,
        "shot_snap_s": shot_snap,
    }
