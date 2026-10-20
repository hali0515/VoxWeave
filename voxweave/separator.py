"""Vocal separation (Mel-Band Roformer): model resolution, identity, demix, separate_vocals.

Heavy deps (torch, numpy, soundfile, the frozen voxweave.vendor model) are imported lazily
inside the functions that need them, so importing this module does not pull in torch.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import BinaryIO, Literal, overload

from voxweave import config
from voxweave.runtime import (
    _empty_cache,
    _hf_download,
    _parse_yaml,
    _require,
    get_device,
)

log = logging.getLogger("voxweave")

# Where an explicitly placed separator ckpt + companion yaml are looked for: VOXWEAVE_MODEL_DIR,
# else the model cache root (VOXWEAVE_CACHE_ROOT, default ~/.cache/voxweave). The model class is
# frozen in voxweave.vendor.
MODEL_DIR = Path(os.environ.get("VOXWEAVE_MODEL_DIR") or config.CACHE_ROOT).expanduser()
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
    """TF32 matmuls for the separator forward are on by default; VOXWEAVE_TF32=0/false/off
    opts out. Read per call."""
    return (
        os.environ.get("VOXWEAVE_TF32", "").strip().casefold() not in _FALSE_ENV_VALUES
    )


@contextmanager
def _separation_matmul_precision(dev: str) -> Iterator[None]:
    """TF32 matmuls for the separator forward only; the previous policy is restored on exit.

    The roformer is attention-heavy and separation dominates wall clock; TF32 matmuls on
    Ampere+ cut that substantially and the stem difference is below the noise floor of the
    downstream 16k ASR. VOXWEAVE_TF32=0 keeps strict fp32 matmuls, and off CUDA nothing
    changes. The matmul policy is process-wide in torch, so it must not outlive the
    separation: left on, every later fp32 CUDA stage (PANNs song/SDH scoring, the wav2vec2
    CTC aligner) would run with TF32 matmuls only on runs that separated, and a
    vocals-cache hit would not reproduce the first run's numbers.
    """
    if not (dev.startswith("cuda") and _tf32_enabled()):
        yield
        return
    import torch

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


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
        with _separation_matmul_precision(get_device()):
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
