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

# Built-in defaults (backend.ASR_MODEL falls back to DEFAULT_ASR_MODEL).
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
# Dual-ASR fusion (--hybrid): whisper supplies text, Qwen supplies punctuation.
# 0.6B emits no punctuation, so fusion must use 1.7B.
DEFAULT_FUSION_WHISPER = "large-v3"
DEFAULT_FUSION_QWEN = "Qwen/Qwen3-ASR-1.7B"
# LLM behind `translate` / `correct` (any OpenAI-compatible chat-completions endpoint).
# Always a concrete model name, never a "-latest" alias: OpenAI retired
# gpt-5.3-chat-latest and both commands were broken out of the box until this
# constant replaced it. "auto" resolves to the endpoint's only served model at run
# time, for self-hosted servers (vLLM) whose served name drifts between restarts.
DEFAULT_LLM_MODEL = "gpt-6-luna"
DEFAULT_LLM_API_KEY_ENV = "OPENAI_API_KEY"
LLM_MODEL_AUTO = "auto"
# `translate` windowing for a self-hosted server: several bounded windows in flight
# instead of one whole-episode request. A thinking model burns ~1k reasoning tokens
# per request and a single 800-cue request is both slow and a single point of
# failure (one aborted stream loses everything). concurrency = 1 restores the
# whole-episode single request (best cross-window continuity, the OpenAI path).
DEFAULT_TRANSLATE_CONCURRENCY = 8
DEFAULT_TRANSLATE_WINDOW_CUES = 100
# community-1 is the default (better multi-speaker separation in practice);
# the 3.1 pipeline stays selectable via the "3.1" alias and keeps its own
# LEGACY constant because the pyannote-4 fail-closed guard for an unverified
# 3.1 plan is keyed on this id (PLDA suppression itself is keyed on plan shape).
# Both pipelines are separately gated on Hugging Face; every module that needs
# either id binds these constants instead of repeating the string, so flipping
# the default cannot leave a value-coincidence literal behind.
COMMUNITY_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
LEGACY_DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_DIARIZE_MODEL = COMMUNITY_DIARIZE_MODEL
DIARIZE_MODEL_ALIASES = {
    "3.1": LEGACY_DIARIZE_MODEL,
    "community-1": COMMUNITY_DIARIZE_MODEL,
}
# Who-is-who grouping of the diarizer's turns: "pyannote" keeps the pipeline's own
# clustering; "voiceprint" regroups the raw turns with ReDimNet2 voiceprints and drops
# turns nobody can be attributed to (voxweave.speakercluster). The built-in default is
# this one constant: flipping it is a one-line change once the evaluation gates pass.
DIARIZE_CLUSTERING_VOICEPRINT = "voiceprint"
DIARIZE_CLUSTERING_PYANNOTE = "pyannote"
DIARIZE_CLUSTERING_CHOICES = (
    DIARIZE_CLUSTERING_VOICEPRINT,
    DIARIZE_CLUSTERING_PYANNOTE,
)
DEFAULT_DIARIZE_CLUSTERING = DIARIZE_CLUSTERING_PYANNOTE
DIARIZE_CLUSTERING_ENV = "VOXWEAVE_DIARIZE_CLUSTERING"
# Per-language aligner defaults. Unlisted languages fall back to Qwen3-ForcedAligner.
#
# Every CTC aligner (en wav2vec2, ja "mms", any HF wav2vec2 id or torchaudio bundle)
# runs one full-file pass: windowed emission, then a global DP that is split at silence
# anchors once it exceeds ctc_max_dp_frames (backend._full_pass_units). Only languages
# without an entry here (Qwen3-ForcedAligner) align per chunk.
#
# en: facebook/wav2vec2-large-960h-lv60-self loaded via HF (same LV60K-self weights as
#     the torchaudio bundle). Torchaudio bundle name also accepted for back-compat.
#
# ja: "mms" = MMS-300m ONNX + uroman (align_blocks_full_mms, equivalent to whisperx
#     fork align_ctc). Full-file is required: per-cue cropping causes coarse routing
#     errors that drift timestamps — empirically misplaced エルダドワーフ by 11s.
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

# Default ASR model (= --asr-model / env VOXWEAVE_ASR_MODEL); short name qwen3-asr-1.7B or full HF id.
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
#   sum            = same passes as peak, but the ASR model(s) stay resident while the
#                    aligner loads; peak VRAM = sum(models); saves two swap round-trips on
#                    large-VRAM cards.
# load_strategy = "peak"

# LLM used by `translate` and `correct`: any OpenAI-compatible chat-completions endpoint.
#   model       = model name (= --model; env VOXWEAVE_TRANSLATE_MODEL / VOXWEAVE_FIX_MODEL
#                 override it per command). "auto" = the endpoint's only served model,
#                 probed at run time (self-hosted vLLM whose served name changes on restart).
#   base_url    = endpoint URL (= --base-url / env OPENAI_BASE_URL); unset = api.openai.com.
#   api_key_env = env var holding the API key (= --api-key-env). "" declares the endpoint
#                 keyless (local vLLM); a placeholder key is sent instead.
#   reasoning_effort = translate's optional reasoning effort (= --reasoning-effort /
#                 VOXWEAVE_TRANSLATE_REASONING_EFFORT). Values depend on the served model.
#                 Unset or "default" leaves the endpoint's default unchanged.
#   concurrency = translate windows in flight at once (= --concurrency /
#                 VOXWEAVE_TRANSLATE_CONCURRENCY; default 8, min 1). 1 = one whole-episode
#                 request with translated-tail continuity (best consistency, OpenAI-style);
#                 >1 = bounded windows translated in parallel with source-text context
#                 (throughput on a self-hosted vLLM; use glossary/context for consistency).
#   window_cues = cues per window when concurrency > 1 (= --window /
#                 VOXWEAVE_TRANSLATE_WINDOW_CUES; default 100, min 1).
[llm]
# model = "gpt-6-luna"
# base_url = "http://127.0.0.1:8000/v1"
# api_key_env = "OPENAI_API_KEY"
# reasoning_effort = "low"
# concurrency = 8
# window_cues = 100

# dual-ASR fusion sub-models (= CLI --hybrid; env VOXWEAVE_FUSION_WHISPER / VOXWEAVE_FUSION_QWEN).
# whisper supplies accurate text, Qwen supplies punctuation positions (merged on a shared timeline).
#   whisper = faster-whisper size: large-v3 (highest quality, default) | large-v3-turbo (~5x faster).
#   qwen    = punctuation model; must emit punctuation so 1.7B (not 0.6B).
[fusion]
# whisper = "large-v3"
# qwen = "Qwen/Qwen3-ASR-1.7B"

# Inference batch sizes: windows per GPU forward pass (= env VOXWEAVE_SEP_BATCH /
# VOXWEAVE_CTC_BATCH / VOXWEAVE_MMS_BATCH / VOXWEAVE_ASR_BATCH). On an 8 GB-class card
# batch=1 already saturates compute for separation and the CTC emission (measured: no
# speedup at 2/4, just +~0.8 GiB VRAM per extra separation window) -- only worth raising
# on much wider GPUs, and only after measuring.
[batch]
# separate = 1   # vocal separation (MelBandRoformer) 8s windows
# ctc = 1        # wav2vec2 CTC emission 30s windows (en aligner)
# mms = 4        # MMS-300m emission batch (ja aligner, ctc-forced-aligner generate_emissions)
# asr = 1        # Qwen3-ASR chunks per decode call (torch only; whisper/MLX per-chunk). Measured on
#                # RTX PRO 4000, 1.7B greedy: 4 -> 1.34x faster / 6.4 GiB peak, 8 -> 1.49x / 8.9 GiB,
#                # but transcripts drift ~1.5% CER vs batch 1 (bf16 batched kernels); qwen-asr #207:
#                # mixed-length batches can corrupt the shorter item (guarded by a per-chunk re-run).

# Vocal separation (MelBandRoformer) numerics (= env VOXWEAVE_SEP_AUTOCAST). autocast wraps
# only the model forward; the overlap-add accumulation always stays fp32.
#   off  (default) = fp32 forward (TF32 matmuls on Ampere+ unless VOXWEAVE_TF32=0); the
#                    reference output, byte-identical run to run.
#   bf16 | fp16    = mixed-precision forward: higher separation throughput and lower VRAM,
#                    at the cost of small waveform differences in the stem. Measured on an
#                    RTX PRO 4000 (23.8 min episode): bf16 1.35x faster, peak VRAM
#                    1.69 -> 1.57 GiB, stem SNR 52 dB vs fp32, but the downstream
#                    transcript drifts ~2.3% CER -- so it stays off until measured on
#                    your own content.
# CUDA only: on CPU / MPS the setting is ignored and the fp32 path runs.
# The effective mode is part of the separator identity, so changing it re-separates
# instead of reusing a vocals cache produced under the previous numerics -- and a CPU /
# MPS host records the fp32 path it really ran, not the mode it was configured with.
[separate]
# autocast = "off"

# Speaker diarization pipeline. The default is "community-1" (better multi-speaker
# separation); accept its model-card conditions on Hugging Face first. Use "3.1" to
# stay on the older pipeline your existing gated access already covers, or provide
# any full Hugging Face pipeline ID. (= --diarize-model / VOXWEAVE_DIARIZE_MODEL)
#
# clustering decides who is who once the pipeline has found the speaker turns
# (= --speaker-clustering / VOXWEAVE_DIARIZE_CLUSTERING; default "@DIARIZE_CLUSTERING@"):
#   pyannote   = the pipeline's own clustering
#   voiceprint = regroup the turns with ReDimNet2 voiceprints (weights non-commercial,
#                CC BY-NC-SA 4.0; 51 MB download on first use); turns nobody can be
#                attributed to are dropped and their words stay with the speaker around them
[diarize]
# model = "community-1"
# clustering = "@DIARIZE_CLUSTERING@"

# Voiceprint embedder (= --voiceprint-model / VOXWEAVE_VOICEPRINT_MODEL); only used with
# --voiceprints. pyannote still finds the speaker turns; the per-speaker voiceprints come
# from a separate speaker-embedding model:
#   auto (default) = anime-va for Japanese, redimnet2 for every other language
#   redimnet2      = ReDimNet2-B6 (vb2+vox2+cnc2, large-margin); weights are
#                    non-commercial (CC BY-NC-SA 4.0, VoxBlink2 training data)
#   anime-va       = anime voice-actor ECAPA-TDNN (Japanese only)
#   pyannote       = legacy: the diarization pipeline's own embeddings; keeps matching
#                    voice stores built before the dedicated embedders
# Voice stores are per embedding space: a store built by one embedder never matches
# another. Changing [diarize].model no longer orphans a store built by redimnet2/anime-va.
[voiceprint]
# model = "auto"

# Voice library (= --voices-dir / VOXWEAVE_VOICES_DIR): where `speakers enroll` stores the
# named voices and `speakers serve` looks them up, across every media folder. Default:
# $XDG_DATA_HOME/voxweave/voices, else ~/.local/share/voxweave/voices. It holds voice
# biometrics of the people you name; `voxweave voices forget ID` removes one person.
# It may point at a NAS path shared by several machines (plain JSON files plus one flock;
# needs an NFSv4 or lock-enabled mount, not `nolock`). A relative path is relative to
# this file's directory.
[voices]
# dir = "/mnt/nas/voxweave/voices"

# Default on/off for the boolean pipeline flags. Explicit CLI flags always win
# (e.g. separate = false here, --separate on the command line for one run).
[defaults]
# separate = true      # vocal separation before ASR/alignment (--separate/--no-separate)
# skip_songs = true    # PANNs music detection + skip before ASR (--skip-songs/--no-skip-songs)
# normalize = false    # loudnorm on the 16k input (--normalize/--no-normalize)
# diarize = false      # pyannote speaker diarization; gated-model HF token required (--diarize/--no-diarize)
# voiceprints = false   # opt-in biometric sidecar capture; requires diarize (--voiceprints/--no-voiceprints)
# timestamps = true    # cue timing lines in the VTT (--timestamps/--no-timestamps)
# shot_snap = true     # snap cue boundaries onto shot changes (--shot-snap/--no-shot-snap)
# vad_mask = false     # suppress CTC emissions outside speech spans (--vad-mask/--no-vad-mask)

# Per-language forced-alignment models; unlisted languages use Qwen3-ForcedAligner (built-in default).
# Values: "mms" (MMS-300m + uroman, full-file single pass; bundled in core) |
#         HF wav2vec2 id (downloaded to the voxweave align cache ~/.cache/voxweave/align) | torchaudio bundle name (-> torch.hub cache).
# Set to "" to explicitly fall back to Qwen. Every CTC value runs one full-file pass (DP split
# at silence anchors above ctc_max_dp_frames); only Qwen-aligned languages align per chunk.
[align]
# en = "facebook/wav2vec2-large-960h-lv60-self"   # English (built-in default): large wav2vec2 CTC, HF path -> ~/.cache/voxweave/align (same LV60K-self weights as torchaudio WAV2VEC2_ASR_LARGE_LV60K_960H bundle)
# ja = "mms"                             # Japanese (built-in default): MMS-300m + uroman (= whisperx fork align_ctc; gold standard); requires onnxruntime-gpu for CUDA (bundled in core)
# ja = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"   # Alternative: xlsr character-level CTC
# zh = "mms"   # Chinese can also use MMS full-file pass; default is Qwen (native CJK character-level)
""".replace("@DIARIZE_CLUSTERING@", DEFAULT_DIARIZE_CLUSTERING)


def config_path() -> Path:
    """Config file path: ``VOXWEAVE_CONFIG`` env if set, else ``~/.config/voxweave.conf``."""
    env = (os.environ.get("VOXWEAVE_CONFIG") or "").strip()
    return Path(env).expanduser() if env else Path.home() / ".config" / "voxweave.conf"


# Recognized top-level keys; anything else in the file is a typo/stale setting.
_KNOWN_KEYS = frozenset(
    {
        "asr_model",
        "ctc_max_dp_frames",
        "load_strategy",
        "hf_token",
        "fusion",
        "batch",
        "separate",
        "diarize",
        "voiceprint",
        "voices",
        "align",
        "defaults",
        "llm",
    }
)


# Recognized keys inside each table; [align] is keyed by language code instead.
# A typo here would otherwise silently revert the setting (e.g. ``[llm] base-url``
# would send subtitles to api.openai.com instead of the local endpoint).
_KNOWN_SECTION_KEYS = {
    "llm": frozenset(
        {
            "model",
            "base_url",
            "api_key_env",
            "reasoning_effort",
            "concurrency",
            "window_cues",
        }
    ),
    "fusion": frozenset({"whisper", "qwen"}),
    "batch": frozenset({"separate", "ctc", "mms", "asr"}),
    "separate": frozenset({"autocast"}),
    "diarize": frozenset({"model", "clustering"}),
    "voiceprint": frozenset({"model"}),
    "voices": frozenset({"dir"}),
    "defaults": frozenset(
        {
            "separate",
            "skip_songs",
            "normalize",
            "diarize",
            "voiceprints",
            "timestamps",
            "shot_snap",
            "vad_mask",
        }
    ),
}


def _load() -> dict:
    """Parse the config TOML. Missing or malformed file returns {} (no crash).
    Unknown top-level keys, and unknown keys inside a known table, are warned
    about but do not stop known keys loading."""
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
            continue
        known = _KNOWN_SECTION_KEYS.get(key)
        section = data[key]
        if known is not None and isinstance(section, dict):
            for inner in section:
                if inner not in known:
                    log.warning(
                        "unknown config key %r in [%s] of %s (ignored)", inner, key, p
                    )
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
    """Return v stripped if it is a non-blank string, else None (config values may be missing/blank/non-str)."""
    return v.strip() if isinstance(v, str) and v.strip() else None


def conf_asr_model() -> str | None:
    """ASR model from config; None if unset. The CLI consults it after --asr-model /
    VOXWEAVE_ASR_MODEL and before the built-in default."""
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


def _conf_llm(key: str) -> str | None:
    """``[llm].<key>`` as written (may be ""), or None if absent / not a string."""
    llm = _load().get("llm")
    if not isinstance(llm, dict) or key not in llm:
        return None
    v = llm[key]
    if not isinstance(v, str):
        log.warning(
            "config key %r has wrong type (expected string), ignoring", f"[llm].{key}"
        )
        return None
    return v


def resolve_llm_model(cli_value: str | None, *, task_envvar: str) -> str:
    """Model for an LLM command. Precedence: CLI value > ``task_envvar``
    (VOXWEAVE_TRANSLATE_MODEL / VOXWEAVE_FIX_MODEL) > conf ``[llm].model`` >
    :data:`DEFAULT_LLM_MODEL`. May return :data:`LLM_MODEL_AUTO`, which the
    caller resolves against the endpoint's model list."""
    return (
        _nonempty_str(cli_value)
        or _nonempty_str(os.environ.get(task_envvar))
        or _nonempty_str(_conf_llm("model"))
        or DEFAULT_LLM_MODEL
    )


def resolve_llm_base_url(cli_value: str | None) -> str | None:
    """Endpoint URL for the LLM commands. Precedence: CLI value > env
    ``OPENAI_BASE_URL`` > conf ``[llm].base_url``; None means the OpenAI default."""
    return (
        _nonempty_str(cli_value)
        or _nonempty_str(os.environ.get("OPENAI_BASE_URL"))
        or _nonempty_str(_conf_llm("base_url"))
    )


def resolve_llm_api_key_env(cli_value: str | None) -> str:
    """Name of the env var holding the LLM API key. Precedence: CLI value > conf
    ``[llm].api_key_env`` > :data:`DEFAULT_LLM_API_KEY_ENV`. An empty string
    (allowed from either source) declares the endpoint keyless."""
    if cli_value is not None:
        return cli_value.strip()
    conf = _conf_llm("api_key_env")
    if conf is not None:
        return conf.strip()
    return DEFAULT_LLM_API_KEY_ENV


def resolve_llm_reasoning_effort(cli_value: str | None) -> str | None:
    """Translate effort: CLI > environment > config; absent leaves server defaults.

    Keep the explicit ``default`` sentinel until the request is built, so it can
    override a configured effort without a downstream resolver restoring it.
    Supported effort names belong to the endpoint/model, not to this client.
    """
    value = (
        _nonempty_str(cli_value)
        or _nonempty_str(os.environ.get("VOXWEAVE_TRANSLATE_REASONING_EFFORT"))
        or _nonempty_str(_conf_llm("reasoning_effort"))
    )
    return value.strip() if value is not None else None


def _positive_int(value: object, source: str) -> int | None:
    """``value`` as an int >= 1, or None (with a warning) when it is missing or invalid.

    Accepts ints and digit strings (env vars arrive as text); bools, floats,
    zero and negatives are rejected so a typo falls through to the next source.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            parsed = None
    else:
        parsed = None
    if parsed is None or parsed < 1:
        log.warning("%s must be an integer >= 1 (got %r); ignoring it", source, value)
        return None
    return parsed


def _resolve_llm_positive_int(
    cli_value: object, *, envvar: str, key: str, cli_flag: str, default: int
) -> int:
    """Shared CLI > env > conf ``[llm].<key>`` > default resolution for the two
    translate windowing knobs; every invalid source warns and falls through.

    ``cli_flag`` is the option a library caller should be pointed at; it is not
    derivable from ``key`` (``window_cues`` is spelled ``--window``)."""
    resolved = _positive_int(cli_value, cli_flag)
    if resolved is not None:
        return resolved
    env = os.environ.get(envvar)
    if env is not None and env.strip():
        resolved = _positive_int(env, f"environment {envvar}")
        if resolved is not None:
            return resolved
    llm = _load().get("llm")
    if isinstance(llm, dict) and key in llm:
        resolved = _positive_int(llm[key], f"config [llm].{key}")
        if resolved is not None:
            return resolved
    return default


def resolve_llm_concurrency(cli_value: object = None) -> int:
    """Translate windows in flight at once. Precedence: CLI value >
    ``VOXWEAVE_TRANSLATE_CONCURRENCY`` > conf ``[llm].concurrency`` >
    :data:`DEFAULT_TRANSLATE_CONCURRENCY`; always >= 1 (1 = the sequential
    whole-episode planner)."""
    return _resolve_llm_positive_int(
        cli_value,
        envvar="VOXWEAVE_TRANSLATE_CONCURRENCY",
        key="concurrency",
        cli_flag="--concurrency",
        default=DEFAULT_TRANSLATE_CONCURRENCY,
    )


def resolve_llm_window_cues(cli_value: object = None) -> int:
    """Cues per translate window when running concurrently. Precedence: CLI value >
    ``VOXWEAVE_TRANSLATE_WINDOW_CUES`` > conf ``[llm].window_cues`` >
    :data:`DEFAULT_TRANSLATE_WINDOW_CUES`; always >= 1."""
    return _resolve_llm_positive_int(
        cli_value,
        envvar="VOXWEAVE_TRANSLATE_WINDOW_CUES",
        key="window_cues",
        cli_flag="--window",
        default=DEFAULT_TRANSLATE_WINDOW_CUES,
    )


def conf_hf_token() -> str | None:
    """Hugging Face token for gated checkpoints (pyannote diarization).

    Precedence: env VOXWEAVE_HF_TOKEN > HF_TOKEN > HUGGING_FACE_HUB_TOKEN >
    conf ``hf_token`` > huggingface_hub stored token; None when nowhere set.

    The lowest-precedence source is the token written by ``hf auth login``
    (``~/.cache/huggingface/token``), read via ``huggingface_hub.get_token``.
    The import is lazy and any failure (missing package, IO error) is treated
    as "no token".
    """
    for key in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = _nonempty_str(os.environ.get(key))
        if v:
            return v
    v = _nonempty_str(_load().get("hf_token"))
    if v:
        return v
    try:
        import huggingface_hub

        return _nonempty_str(huggingface_hub.get_token())
    except Exception:
        return None


def _conf_diarize_model() -> str | None:
    """Return ``[diarize].model`` when it is a non-blank string."""
    section = _load().get("diarize")
    if section is None:
        return None
    if not isinstance(section, dict):
        log.warning(
            "config key %r has wrong type (expected table), ignoring", "diarize"
        )
        return None
    raw = section.get("model")
    if raw is not None and not isinstance(raw, str):
        log.warning("config [diarize].model has wrong type (expected string), ignoring")
        return None
    return _nonempty_str(raw)


def resolve_diarize_model(cli_value: str | None = None) -> str:
    """Resolve and normalize the diarization pipeline ID.

    Precedence is explicit CLI value, ``VOXWEAVE_DIARIZE_MODEL``,
    ``[diarize].model``, then the community-1 default. Known short names
    map to their full Hugging Face IDs; every other non-blank value passes
    through unchanged.
    """
    selected = _nonempty_str(cli_value)
    if selected is None:
        selected = _nonempty_str(os.environ.get("VOXWEAVE_DIARIZE_MODEL"))
    if selected is None:
        selected = _conf_diarize_model()
    if selected is None:
        selected = DEFAULT_DIARIZE_MODEL
    selected = selected.strip()
    return DIARIZE_MODEL_ALIASES.get(selected.lower(), selected)


def _conf_diarize_clustering() -> str | None:
    """Return ``[diarize].clustering`` when it is a non-blank string."""
    section = _load().get("diarize")
    if section is None:
        return None
    if not isinstance(section, dict):
        log.warning(
            "config key %r has wrong type (expected table), ignoring", "diarize"
        )
        return None
    raw = section.get("clustering")
    if raw is not None and not isinstance(raw, str):
        log.warning(
            "config [diarize].clustering has wrong type (expected string), ignoring"
        )
        return None
    return _nonempty_str(raw)


def normalize_diarize_clustering(
    value: str, *, source: str = "--speaker-clustering"
) -> str:
    """Map a user value to ``"voiceprint"`` or ``"pyannote"`` (case-insensitive).

    An unknown value raises ``ValueError`` naming its source instead of silently
    running the other clustering.
    """
    key = value.strip().casefold()
    if key in DIARIZE_CLUSTERING_CHOICES:
        return key
    raise ValueError(
        f"{source} has unknown speaker clustering {value!r}; "
        f"choose one of {', '.join(DIARIZE_CLUSTERING_CHOICES)}"
    )


def resolve_diarize_clustering(cli_value: str | None = None) -> str:
    """Resolve how diarization groups its turns into speakers.

    Precedence is explicit CLI value, ``VOXWEAVE_DIARIZE_CLUSTERING``,
    ``[diarize].clustering``, then :data:`DEFAULT_DIARIZE_CLUSTERING`. Blank
    values fall through to the next source; an unknown value raises
    ``ValueError`` naming its source (a config value of the wrong type is warned
    about and ignored, like ``[diarize].model``).
    """
    selected = _nonempty_str(cli_value)
    if selected is not None:
        return normalize_diarize_clustering(selected)
    env = _nonempty_str(os.environ.get(DIARIZE_CLUSTERING_ENV))
    if env is not None:
        return normalize_diarize_clustering(
            env, source=f"environment {DIARIZE_CLUSTERING_ENV}"
        )
    conf = _conf_diarize_clustering()
    if conf is not None:
        return normalize_diarize_clustering(conf, source="config [diarize].clustering")
    return DEFAULT_DIARIZE_CLUSTERING


def conf_voiceprint_model() -> str | None:
    """Return ``[voiceprint].model`` when it is a non-blank string.

    The value is resolved (aliases, per-language ``auto`` routing, validation) by
    :func:`voxweave.voiceembed.resolve_voiceprint_choice`; this only reads it.
    """
    section = _load().get("voiceprint")
    if section is None:
        return None
    if not isinstance(section, dict):
        log.warning(
            "config key %r has wrong type (expected table), ignoring", "voiceprint"
        )
        return None
    raw = section.get("model")
    if raw is not None and not isinstance(raw, str):
        log.warning(
            "config [voiceprint].model has wrong type (expected string), ignoring"
        )
        return None
    return _nonempty_str(raw)


def conf_voices_dir() -> Path | None:
    """Return ``[voices].dir`` as an absolute path, or None when unset.

    ``~`` is expanded, and a relative value is taken relative to the config
    file's directory (not the working directory, which changes per command).
    Precedence against ``--voices-dir`` / ``VOXWEAVE_VOICES_DIR`` is resolved
    by :func:`voxweave.voicelibrary.resolve_voices_dir`.
    """
    section = _load().get("voices")
    if section is None:
        return None
    if not isinstance(section, dict):
        log.warning("config key %r has wrong type (expected table), ignoring", "voices")
        return None
    raw = section.get("dir")
    if raw is not None and not isinstance(raw, str):
        log.warning("config [voices].dir has wrong type (expected string), ignoring")
        return None
    value = _nonempty_str(raw)
    if value is None:
        return None
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = config_path().expanduser().absolute().parent / path
    return path


_LOAD_STRATEGIES = ("peak", "sum")


def conf_load_strategy() -> str:
    """Model load strategy.

    - ``"peak"`` (default): serial peak-shaving — ASR and aligner never co-reside;
      peak VRAM = max(each model). Works on 8 GB cards.
    - ``"sum"``: same passes as peak, but the ASR model(s) stay resident while the
      aligner loads; peak VRAM = sum(models). Saves swap overhead on large-VRAM cards.

    Precedence: env VOXWEAVE_LOAD_STRATEGY > conf load_strategy > "peak". Invalid
    values are warned about and fall back to "peak".
    """
    raw = os.environ.get("VOXWEAVE_LOAD_STRATEGY") or _load().get("load_strategy")
    v = raw.strip().lower() if isinstance(raw, str) else ""
    if (raw is not None and raw != "") and v not in _LOAD_STRATEGIES:
        log.warning(
            "load_strategy must be one of %s (got %r); using 'peak'",
            ", ".join(sorted(_LOAD_STRATEGIES)),
            raw,
        )
    return v if v in _LOAD_STRATEGIES else "peak"


# CTC forced-align single-pass DP frame budget. Audio whose emission exceeds this is split at
# silence anchors (chunking.plan_dp_chunks) before the O(T*L) DP; ~30min default (50fps*60*30).
# Bigger = fewer/larger chunks = more accurate on long sparse-dialogue audio (the global DP gains
# context), at higher GPU memory. Lower on small cards.
_CTC_MAX_DP_FRAMES_DEFAULT = 90000


def conf_ctc_max_dp_frames() -> int:
    """Max emission frames for one CTC forced-align DP before silence-anchored chunking kicks in.

    Precedence: env VOXWEAVE_CTC_MAX_DP_FRAMES > conf ``ctc_max_dp_frames`` > 90000 (~30min).
    Values that are not integers >= 1 (env or file) are warned about and fall through
    to the next source.
    """
    env = os.environ.get("VOXWEAVE_CTC_MAX_DP_FRAMES")
    if env is not None and env.strip():
        parsed = _positive_int(env, "environment VOXWEAVE_CTC_MAX_DP_FRAMES")
        if parsed is not None:
            return parsed
    v = _load().get("ctc_max_dp_frames")
    if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
        return v
    if v is not None:
        log.warning(
            "config key %r must be an integer >= 1 (got %r), using default",
            "ctc_max_dp_frames",
            v,
        )
    return _CTC_MAX_DP_FRAMES_DEFAULT


# Inference batch sizes (windows per GPU forward). Defaults = 1: measured on an RTX 4070
# Laptop (8 GB), separation batch=1 already saturates compute (steady-state latency scales
# linearly with batch; same for the wav2vec2 CTC emission), so batching only costs VRAM
# (~+0.8 GiB per extra separation window). The knob exists for much wider GPUs, where
# per-window kernels may underfill the SMs — measure before raising. mms=4 is the
# ctc-forced-aligner upstream default (ONNX path, pre-existing behavior). asr=1 keeps the
# legacy per-chunk Qwen3-ASR call; batching (backend._asr_pass) is opt-in: it is faster
# (RTX PRO 4000, 1.7B greedy: 4 -> 1.34x, 8 -> 1.49x) but its transcripts drift ~1.5% CER
# from batch 1 and there is no truth ruler yet to judge the sign of that change.
_BATCH_DEFAULTS = {"separate": 1, "ctc": 1, "mms": 4, "asr": 1}
_BATCH_ENV = {
    "separate": "VOXWEAVE_SEP_BATCH",
    "ctc": "VOXWEAVE_CTC_BATCH",
    "mms": "VOXWEAVE_MMS_BATCH",  # pre-[batch] env name, kept for back-compat
    "asr": "VOXWEAVE_ASR_BATCH",
}


def conf_batch(key: str) -> int:
    """Inference batch size for stage ``key`` ("separate" | "ctc" | "mms" | "asr"), min 1.

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


# Autocast for the MelBandRoformer separation forward. "off" is the byte-identical
# reference path (fp32 with TF32 matmuls, see backend._load_separator); bf16/fp16 trade
# tiny stem differences for throughput. The default stays off until an A/B decides it.
SEP_AUTOCAST_MODES = ("off", "bf16", "fp16")
SEP_AUTOCAST_DEFAULT = "off"
SEP_AUTOCAST_ENV = "VOXWEAVE_SEP_AUTOCAST"


def conf_separate_autocast() -> str:
    """Separation forward autocast mode: ``"off"`` (default) | ``"bf16"`` | ``"fp16"``.

    Precedence: env VOXWEAVE_SEP_AUTOCAST > conf ``[separate].autocast`` > "off".
    Values are case-insensitive. An invalid value (unknown mode or wrong type) from
    the winning source is warned about once and falls back to "off" -- it does not
    fall through to the next source, so a typo never silently enables autocast. A
    non-table ``separate`` key (a scalar instead of a ``[separate]`` section) is
    warned about the same way.
    """
    env = os.environ.get(SEP_AUTOCAST_ENV)
    if env is not None and env.strip():
        raw: object = env
        source = f"env {SEP_AUTOCAST_ENV}"
    else:
        separate = _load().get("separate")
        if separate is not None and not isinstance(separate, dict):
            log.warning(
                "config key %r has wrong type (expected table), ignoring", "separate"
            )
            return SEP_AUTOCAST_DEFAULT
        raw = separate.get("autocast") if isinstance(separate, dict) else None
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return SEP_AUTOCAST_DEFAULT
        source = "config [separate].autocast"
    if isinstance(raw, str):
        mode = raw.strip().casefold()
        if mode in SEP_AUTOCAST_MODES:
            return mode
    log.warning(
        "%s has invalid value %r (expected one of %s), using %r",
        source,
        raw,
        "/".join(SEP_AUTOCAST_MODES),
        SEP_AUTOCAST_DEFAULT,
    )
    return SEP_AUTOCAST_DEFAULT


def conf_default_flag(key: str, builtin: bool) -> bool:
    """Default for the boolean pipeline flag ``key`` from ``[defaults]``.

    Precedence is resolved by the CLI: an explicit CLI flag wins; otherwise this
    value applies. Non-boolean values are warned about and fall back to
    ``builtin``.
    """
    return conf_default_flag_source(key, builtin)[0]


def conf_default_flag_source(key: str, builtin: bool) -> tuple[bool, str]:
    """Resolve a config boolean and report its source for cross-flag errors."""
    defaults = _load().get("defaults")
    if isinstance(defaults, dict) and key in defaults:
        v = defaults[key]
        if isinstance(v, bool):
            return v, f"config [defaults].{key}"
        log.warning(
            "config [defaults].%s has wrong type (expected boolean), using default",
            key,
        )
    return builtin, "built-in default"


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
_CPS_DEFAULTS = {"ja": 7.0, "zh": 9.0, "yue": 9.0, "ko": 9.0}
_CPS_LATIN_DEFAULT = 17.0  # ~Netflix 20 cps incl. spaces, measured without spaces
_LAG_OUT_DEFAULT_MS = 250  # flat tail pad after speech ends (0=off)
_SHOT_SNAP_DEFAULT_MS = (
    458  # shot-change pairing window: 11 frames @24fps, the outermost Netflix
    # adjustment zone; boundaries past it are left alone (0=off)
)


def _env_int(name: str, default: int) -> int:
    """Integer env knob; unset/blank -> ``default``. A malformed value also falls back
    to ``default`` (never raises: import-time knobs would break every command, even
    ``--help``), with a warning naming the variable."""
    v = os.environ.get(name)
    try:
        return int(v) if v is not None and v.strip() else default
    except ValueError:
        log.warning("ignoring %s=%r (not an integer); using %s", name, v, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Float env knob; same fallback-with-warning contract as :func:`_env_int`."""
    v = os.environ.get(name)
    try:
        return float(v) if v is not None and v.strip() else default
    except ValueError:
        log.warning("ignoring %s=%r (not a number); using %s", name, v, default)
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
