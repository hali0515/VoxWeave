"""Process-level runtime infrastructure shared by the model backends.

Device resolution, dtype policy, VRAM reclaim, dependency-error messages and
HF download helpers. No model logic lives here — ``backend`` (separation/ASR),
``align_ctc`` and ``align_mms`` all build on this module, which keeps the
import direction acyclic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("voxweave")

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
