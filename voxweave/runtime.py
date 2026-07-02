"""Process-level runtime infrastructure shared by the model backends.

Device resolution, dtype policy, VRAM reclaim, dependency-error messages and
HF download helpers. No model logic lives here — ``backend`` (separation/ASR),
``align_ctc`` and ``align_mms`` all build on this module, which keeps the
import direction acyclic.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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


def _hf_error(repo: str, err: Exception) -> RuntimeError:
    """Wrap a raw huggingface_hub download failure with an actionable auth/network hint."""
    return RuntimeError(
        f"failed to download {repo!r} from Hugging Face: {err}. "
        "Check network access; gated repos need HF_TOKEN (or VOXWEAVE_HF_TOKEN) set."
    )


def _load_yaml(path: Path) -> dict:
    """SafeLoader + !!python/tuple support (used by window sizes in MSST-style configs)."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),  # pyright: ignore[reportArgumentType]
    )
    return yaml.load(path.read_text(), Loader=_Loader)


# Reporter receiving byte progress for HF downloads, installed by the CLI for the lifetime of its
# RichReporter (see ui.RichReporter.__enter__). None -> huggingface_hub renders its own tqdm bars
# (library callers, no rich Live to fight with).
_dl_reporter = None


def set_download_reporter(reporter) -> None:
    """Install (or clear, with None) the progress.Reporter that receives HF download byte progress."""
    global _dl_reporter
    _dl_reporter = reporter


@contextmanager
def _bridged_bars(label: str):
    """Silence huggingface_hub's tqdm bars and forward their byte counts to the active Reporter.

    Raw hub tqdm fights the rich Live region: rich's FileProxy flushes every tqdm refresh as a
    separate permanent line, and the xet backend only reports per completed ~xorb, so on slow
    links the bar sits at 0% for minutes and then several lines burst out at once -- users read
    that as a hang. Bridging into the Reporter row keeps spinner/elapsed animating between
    bursts and renders one in-place updating line.

    Yields a tqdm-compatible class to pass as ``tqdm_class=`` where the hub API supports it
    (hub >= 1.x forwards it down to the byte bars; 0.36 accepts it only on snapshot_download's
    outer bar). For 0.36's byte bars -- constructed from the module-global ``tqdm`` inside
    ``_get_progress_bar_context`` -- the module global is patched for the duration. Yields
    ``None`` (and touches nothing) when no Reporter is installed or hub internals moved.
    """
    rep = _dl_reporter
    if rep is None:
        yield None
        return
    try:
        # importlib, not `import huggingface_hub.utils.tqdm`: the utils package __init__
        # rebinds its `tqdm` attribute to the class, shadowing the submodule.
        hub_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")

        base = hub_tqdm_mod.tqdm
    except Exception:  # noqa: BLE001 -- internals moved; fall back to hub's own bars
        yield None
        return

    lock = threading.Lock()
    bars: dict[int, tuple[int, int]] = {}  # id(bar) -> (done_bytes, total_bytes)

    def _report() -> None:
        # rep.download inside the lock: keeps deliveries monotonic when xet/snapshot worker
        # threads land updates concurrently (sum computed and shipped atomically).
        with lock:
            done = sum(d for d, _ in bars.values())
            known = [t for _, t in bars.values() if t > 0]
            total = sum(known) if bars and len(known) == len(bars) else None
            rep.download(label, done, total)

    class _Bridge(base):  # type: ignore[misc,valid-type]
        """Never-rendering tqdm that aggregates byte updates across files into the Reporter."""

        def __init__(self, *args, **kwargs):
            self._vox_bytes = kwargs.get("unit") == "B"
            self._vox_total = int(kwargs.get("total") or 0)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            if self._vox_bytes:
                with lock:
                    bars[id(self)] = (int(kwargs.get("initial") or 0), self._vox_total)
                _report()

        def update(self, n: float | None = 1):
            if self._vox_bytes and n:
                with lock:
                    done, _ = bars[id(self)]
                    # re-read total each update: snapshot aggregators assign .total after construction
                    total = int(getattr(self, "total", None) or self._vox_total)
                    bars[id(self)] = (done + int(n), total)
                _report()
            return super().update(n)

    setattr(hub_tqdm_mod, "tqdm", _Bridge)
    try:
        yield _Bridge
    finally:
        setattr(hub_tqdm_mod, "tqdm", base)


def _hf_download(repo: str, filename: str, cache_dir: str | None = None) -> str:
    """Download a single file from HF, return local path. cache_dir routes into a voxweave-owned subdir; None keeps HF default."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as e:
        raise _require(e.name or "huggingface_hub") from e
    with _bridged_bars(filename) as bridge:
        kwargs = {}
        if (
            bridge is not None
            and "tqdm_class" in inspect.signature(hf_hub_download).parameters
        ):
            kwargs["tqdm_class"] = bridge
        try:
            return hf_hub_download(repo, filename, cache_dir=cache_dir, **kwargs)
        except RuntimeError:
            raise
        except Exception as e:
            raise _hf_error(repo, e) from e


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
    with _bridged_bars(repo) as bridge:
        kwargs: dict[str, Any] = {} if bridge is None else {"tqdm_class": bridge}
        try:
            return snapshot_download(repo, cache_dir=cache_dir, **kwargs)
        except RuntimeError:
            raise
        except Exception as e:
            raise _hf_error(repo, e) from e
