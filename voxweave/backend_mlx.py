"""Apple-Silicon (MLX) backend for the local Qwen3 pipeline.

On macOS / MPS the PyTorch Qwen3-ASR + Qwen3-ForcedAligner are served instead by their native
MLX ports from `mlx-audio` (https://github.com/Blaizzy/mlx-audio): purpose-built Metal kernels +
4/8-bit quantization, faster and lower-memory than running the torch models through the MPS
backend. The two models cover the same 11 languages as the torch aligner, so on this backend ALL
alignment (incl. ja/CJK and en, which the torch path routes to MMS/wav2vec2 CTC) goes through the
MLX Qwen3-ForcedAligner — onnxruntime has no Metal provider, so the ONNX MMS aligner could never
run on the GPU here anyway.

Vocal separation (MelBandRoformer) and PANNs song-skip have no MLX port and stay on torch-MPS;
this module only owns ASR + forced alignment. Selection lives in voxweave.backend._use_mlx().

These adapters mirror the tiny slice of the qwen-asr / Qwen3ForcedAligner API that backend.py
calls, so the call sites in backend.py stay backend-agnostic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from voxweave import config

log = logging.getLogger("voxweave")

# torch HF repo id -> mlx-community quantized repo id. Substring fallback handles custom ids.
_MLX_ASR_REPOS = {
    "Qwen/Qwen3-ASR-0.6B": "mlx-community/Qwen3-ASR-0.6B-8bit",
    "Qwen/Qwen3-ASR-1.7B": "mlx-community/Qwen3-ASR-1.7B-8bit",
}
_DEFAULT_MLX_ASR = "mlx-community/Qwen3-ASR-0.6B-8bit"
MLX_ALIGNER_REPO = os.environ.get(
    "VOXWEAVE_MLX_ALIGNER_REPO", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
)

_MISSING_MLX = (
    "The MLX backend requires the voxweave[mps] install (mlx-audio + mlx). "
    "Install: `make install VARIANT=mps` (Apple Silicon/macOS only). "
    "Force the torch backend instead with VOXWEAVE_BACKEND=torch. Missing: {mod}"
)

# whisper size string -> mlx-community converted repo. The hybrid/fusion engines (--model large-v3,
# --hybrid) need whisper text, but faster-whisper's ctranslate2 has no Metal backend; mlx-whisper is
# the native Metal port. Sizes outside this table fall back to the generic mlx-community naming
# `whisper-<size>-mlx` (covers tiny/base/small/medium/large-v3); turbo/distil deviate, so list them.
_MLX_WHISPER_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "distil-large-v2": "mlx-community/distil-whisper-large-v2",
}

# Process-level singletons; released by release()/release_asr()/release_whisper() at end of episode
# (mirrors backend.py).
_asr = None  # _MlxAsr adapter
_asr_repo = None  # currently loaded MLX ASR repo id (reloaded on --model change)
_aligner = None  # mlx_audio forced-aligner model
_whisper = None  # _MlxWhisper adapter
_whisper_id = None  # currently loaded whisper size string (reloaded on --model change)


def _require(mod: str) -> RuntimeError:
    return RuntimeError(_MISSING_MLX.format(mod=mod))


def _load(repo: str, cache_dir: str):
    """Download repo into cache_dir (VoxWeave's own cache, matching the torch _hf_snapshot path)
    then load from the local snapshot — mlx_audio.stt.load accepts a local dir, so it won't re-fetch
    from the hub. Keeps MLX weights under ~/.cache/voxweave alongside the separator/PANNs weights."""
    try:
        from huggingface_hub import snapshot_download
        from mlx_audio.stt import load  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as e:
        raise _require(e.name or "mlx_audio") from e
    local = snapshot_download(repo, cache_dir=cache_dir)
    return load(local)


def _snapshot(repo: str, cache_dir: str) -> str:
    """Download repo into cache_dir and return the local snapshot dir (no mlx_audio.stt.load).
    Used for the whisper path, which loads via mlx_whisper.transcribe(path_or_hf_repo=local_dir)."""
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as e:
        raise _require(e.name or "huggingface_hub") from e
    return snapshot_download(repo, cache_dir=cache_dir)


def _clear_cache() -> None:
    """Best-effort MLX Metal cache reclaim (optimization only; failure is harmless)."""
    try:
        import mlx.core as mx  # pyright: ignore[reportMissingImports]

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


def _mlx_asr_repo(model_id: str | None) -> str:
    """Map a torch ASR repo id (or --model value) to the mlx-community quantized repo.

    Size follows --model / VOXWEAVE_ASR_MODEL: 'Qwen/Qwen3-ASR-1.7B' (or any id containing '1.7')
    -> the 1.7B quant, else the 0.6B quant. VOXWEAVE_MLX_ASR_REPO hard-overrides everything (e.g.
    to pin a 4-bit quant or a non-standard repo), regardless of --model.
    """
    override = os.environ.get("VOXWEAVE_MLX_ASR_REPO", "").strip()
    if override:
        return override
    if model_id and model_id in _MLX_ASR_REPOS:
        return _MLX_ASR_REPOS[model_id]
    if model_id and "1.7" in model_id:
        return "mlx-community/Qwen3-ASR-1.7B-8bit"
    return _DEFAULT_MLX_ASR


class _AsrResult:
    """Mirror of qwen_asr's transcribe() result: only .language + .text are read by backend.py."""

    __slots__ = ("language", "text")

    def __init__(self, language: str | None, text: str):
        self.language = language
        self.text = text


class _MlxAsr:
    """Adapter exposing qwen-asr's `.transcribe(path, language=, return_time_stamps=, context=)`
    over mlx-audio's Qwen3-ASR `.generate()`. context maps to the model's system_prompt (best-effort
    proper-noun biasing; not identical to qwen-asr's native context= field)."""

    def __init__(self, model):
        self._m = model

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None = None,
        return_time_stamps: bool = False,  # noqa: ARG002 -- qwen-asr arg name; MLX is text-only here
        context: str | None = None,
    ) -> list[_AsrResult]:
        from voxweave.lang import to_aligner_name

        lang = to_aligner_name(language) if language and language.strip() else None
        out = self._m.generate(
            str(wav_path), language=lang, system_prompt=context or None
        )
        lang_field = getattr(out, "language", None)
        if isinstance(lang_field, (list, tuple)):
            det = next((x for x in lang_field if x), None)
        else:
            det = lang_field or None
        return [_AsrResult(det, out.text)]


def get_asr(model_id: str | None = None):
    """Lazy-load the MLX Qwen3-ASR singleton, reloading if the requested repo changes."""
    global _asr, _asr_repo
    repo = _mlx_asr_repo(model_id)
    if _asr is not None and _asr_repo != repo:
        release_asr()
    if _asr is None:
        log.info("loading MLX ASR=%s", repo)
        _asr = _MlxAsr(_load(repo, config.ASR_CACHE))
        _asr_repo = repo
        log.info("MLX ASR ready")
    return _asr


def _mlx_whisper_repo(size: str) -> str:
    """Map a faster-whisper size string to the mlx-community converted repo.

    VOXWEAVE_MLX_WHISPER_REPO hard-overrides (pin a specific quant/repo). Known sizes resolve via
    the table; everything else uses the generic `mlx-community/whisper-<size>-mlx` convention.
    """
    override = os.environ.get("VOXWEAVE_MLX_WHISPER_REPO", "").strip()
    if override:
        return override
    if size in _MLX_WHISPER_REPOS:
        return _MLX_WHISPER_REPOS[size]
    return f"mlx-community/whisper-{size}-mlx"


class _WhisperSegment:
    """Mirror of faster-whisper's Segment: backend._asr_only reads only `.text`."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


class _WhisperInfo:
    """Mirror of faster-whisper's TranscriptionInfo: backend._asr_only reads only `.language`."""

    __slots__ = ("language",)

    def __init__(self, language: str | None):
        self.language = language


class _MlxWhisper:
    """Adapter exposing faster-whisper's `WhisperModel.transcribe(path, language=, initial_prompt=,
    condition_on_previous_text=, vad_filter=, word_timestamps=) -> (segments, info)` over
    mlx_whisper.transcribe (an openai-whisper decode-loop port, so the kwargs are name-compatible).
    backend._asr_only joins segment `.text` and reads `info.language`."""

    def __init__(self, local_path: str):
        self._path = local_path

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        condition_on_previous_text: bool = False,
        vad_filter: bool = False,  # noqa: ARG002 -- faster-whisper arg; mlx_whisper has none (VAD upstream)
        word_timestamps: bool = False,
    ) -> tuple[list[_WhisperSegment], _WhisperInfo]:
        try:
            import mlx_whisper  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as e:
            raise _require(e.name or "mlx_whisper") from e
        res = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=self._path,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
        )
        segs = [_WhisperSegment(s.get("text", "")) for s in res.get("segments", [])]
        if not segs and res.get(
            "text"
        ):  # no per-segment breakdown -> single full-text segment
            segs = [_WhisperSegment(res["text"])]
        return segs, _WhisperInfo(res.get("language"))


def get_whisper(model_id: str):
    """Lazy-load the MLX whisper adapter singleton, reloading if the requested size changes.

    Mirrors backend._get_whisper: snapshot-downloads the converted repo into VoxWeave's ASR cache,
    then hands the local dir to mlx_whisper.transcribe (which lru-caches the loaded weights)."""
    global _whisper, _whisper_id
    if _whisper is not None and _whisper_id != model_id:
        release_whisper()
    if _whisper is None:
        repo = _mlx_whisper_repo(model_id)
        log.info("loading MLX whisper=%s", repo)
        _whisper = _MlxWhisper(_snapshot(repo, config.ASR_CACHE))
        _whisper_id = model_id
        log.info("MLX whisper ready")
    return _whisper


def release_whisper() -> None:
    """Drop the MLX whisper singleton (called between fusion passes to cut peak memory)."""
    global _whisper, _whisper_id
    _whisper = None
    _whisper_id = None
    _clear_cache()


def _get_aligner():
    """Lazy-load the MLX Qwen3-ForcedAligner singleton."""
    global _aligner
    if _aligner is None:
        log.info("loading MLX forced aligner=%s", MLX_ALIGNER_REPO)
        _aligner = _load(MLX_ALIGNER_REPO, config.ALIGN_CACHE)
        log.info("MLX forced aligner ready")
    return _aligner


def align(wav_path: Path, text: str, language: str) -> list[dict]:
    """Forced alignment via MLX Qwen3-ForcedAligner -> units [{text,start,end}].

    Covers all 11 languages (incl. ja/CJK/en), so this fully replaces the torch MMS/wav2vec2 CTC
    path on the MLX backend. language accepts ISO or full name.
    """
    from voxweave.lang import to_aligner_name

    model = _get_aligner()
    res = model.generate(
        audio=str(wav_path), text=text, language=to_aligner_name(language)
    )
    if isinstance(res, list):  # batch API returns a list; we align a single pair
        res = res[0]
    items = getattr(res, "items", None)
    if items is None:
        items = list(res)  # ForcedAlignResult is also directly iterable over its items
    units = [
        {
            "text": it.text,
            "start": float(it.start_time),
            "end": float(it.end_time),
        }
        for it in items
    ]
    _clear_cache()
    return units


def release_asr() -> None:
    """Drop the MLX ASR singleton (called between transcribe_chunks passes to cut peak memory)."""
    global _asr, _asr_repo
    _asr = None
    _asr_repo = None
    _clear_cache()


def release() -> None:
    """Drop all MLX singletons. Safe no-op when the MLX backend was never used."""
    global _aligner
    release_asr()
    release_whisper()
    _aligner = None
    _clear_cache()
