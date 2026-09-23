"""Decoupled speaker-embedding models for voiceprints.

pyannote finds the speaker turns. The per-speaker centroids persisted in
``<stem>.voiceprints.json``, matched against ``voxweave.voices.json`` and
recomputed by the speakers page's "Split this speaker" come from a dedicated
speaker-embedding model chosen per language instead of the diarization
pipeline's own embedding head:

- ``redimnet2`` (default for every language but Japanese): ReDimNet2-B6 trained
  on VoxBlink2 + VoxCeleb2 + CN-Celeb2 with large-margin fine-tuning.
- ``anime-va`` (default for Japanese): an anime voice-actor ECAPA-TDNN.
- ``pyannote``: the legacy lane -- the diarization pipeline's own embeddings,
  kept so an existing legacy voice store can still be matched.

Every checkpoint is pinned by size and SHA-256 and verified before it is
deserialized; the weights live under ``config.AUDIO_CACHE`` and nothing here
touches the GPU or the network at import time. A voiceprint run fetches and
verifies what it needs up front (:func:`prefetch_checkpoints`) instead of at
the capture step.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voxweave import config, fsio, runtime
from voxweave.voicebase import (
    MAX_EMBEDDING_DIM,
    MIN_EMBEDDING_DIM,
    Phase2DataError,
    validate_vector,
)

log = logging.getLogger("voxweave")

SAMPLE_RATE = 16_000
ENV_MODEL = "VOXWEAVE_VOICEPRINT_MODEL"

# Provenance values of the decoupled lane (voicematch fingerprints them).
LANE_DECOUPLED = "decoupled"
CENTROID_RECIPE = "centroid-v1"

# centroid-v1 recipe. Kept together because the recipe id above names exactly
# this combination; changing any of them is a new recipe id, never an edit.
#
# A turn (after trimming the stretches another speaker overlaps) counts as a
# clean voiceprint segment from this length on -- the same floor the split
# feature has always used for whole turns (turnembed re-exports it).
MIN_TURN_SECONDS = 2.0
# A speaker with no clean segment at MIN_TURN_SECONDS still gets a centroid
# from its longest pieces down to this length (padded up to the model's
# min_seconds); below it the speaker simply has no voiceprint.
FALLBACK_SEGMENT_SECONDS = 0.5
# Cost cap: only the longest segments of a speaker are embedded.
MAX_SEGMENTS_PER_SPEAKER = 20
# Long segments are embedded as equal windows no longer than this and the
# window vectors are averaged (weighted by window length), so a 60 s monologue
# cannot dominate memory or drown the pooling statistics.
WINDOW_SECONDS = 10.0

# Checkpoint downloads: urlopen's timeout bounds every connect and every read,
# so a stalled network fails within this many seconds instead of hanging the
# run (a slow but live transfer never trips it).
DOWNLOAD_TIMEOUT_SECONDS = 30.0
# Read size for streaming downloads and checkpoint hashing.
_IO_CHUNK_BYTES = 1 << 20


class VoiceEmbeddingError(RuntimeError):
    """A voiceprint embedder could not be resolved, loaded or run."""


@dataclass(frozen=True)
class EmbedderSpec:
    """One pinned speaker-embedding checkpoint and how to run it.

    ``name`` is the stable id recorded as ``embedding_model`` in voiceprint
    provenance; together with ``sha256`` it defines an embedding space.
    ``suggest``/``margin`` are the matching defaults for that space. ``size``
    is the pinned checkpoint's exact byte count: a file of any other size is
    refused before a single byte of it is read.
    """

    name: str
    languages: tuple[str, ...] | None
    embedding_dim: int
    sample_rate: int
    min_seconds: float
    suggest: float
    margin: float
    sha256: str
    size: int
    filename: str
    checkpoint_env: str
    loader: Callable[[object], tuple[Any, int]]
    url: str | None = None
    hf_repo: str | None = None
    hf_revision: str | None = None
    cache_subdir: str | None = None

    @property
    def source(self) -> str:
        """Human-readable download source."""
        if self.url is not None:
            return self.url
        return f"https://huggingface.co/{self.hf_repo} ({self.filename} @ {self.hf_revision})"


@dataclass(frozen=True)
class LegacyLane:
    """Sentinel for the legacy lane: voiceprints from the diarization pipeline."""

    name: str = "pyannote"


LEGACY = LegacyLane()


def _load_redimnet2(checkpoint: object) -> tuple[Any, int]:
    from voxweave import voiceembed_models

    return voiceembed_models.build_redimnet2(checkpoint)


def _load_anime_va(checkpoint: object) -> tuple[Any, int]:
    from voxweave import voiceembed_models

    return voiceembed_models.build_anime_va(checkpoint)


# ReDimNet2-B6 trained on VoxBlink2 + VoxCeleb2 + CN-Celeb2, large-margin
# fine-tuned.
REDIMNET2_B6 = EmbedderSpec(
    name="redimnet2-b6-vb2-vox2-cnc2-lm",
    languages=None,
    embedding_dim=192,
    sample_rate=SAMPLE_RATE,
    min_seconds=1.0,
    suggest=0.45,  # provisional: calibrate with scripts/calibrate_voiceprints.py
    margin=0.05,  # provisional: calibrate with scripts/calibrate_voiceprints.py
    sha256="287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868",
    size=51_152_991,
    filename="b6-vb2+vox2+cnc2_v0-lm.pt",
    checkpoint_env="VOXWEAVE_REDIMNET2_CKPT",
    loader=_load_redimnet2,
    # The release tag is mutable upstream; the SHA-256 pin above is the identity.
    url=(
        "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/"
        "b6-vb2%2Bvox2%2Bcnc2_v0-lm.pt"
    ),
    cache_subdir="redimnet2",
)

# Anime voice-actor ECAPA-TDNN (GroupNorm), Japanese only.
ANIME_VA = EmbedderSpec(
    name="anime-va-ecapa-gn",
    languages=("ja",),
    embedding_dim=192,
    sample_rate=SAMPLE_RATE,
    min_seconds=1.0,
    # The model card warns that same-speaker cosines run lower than other
    # models', hence the lower bar.
    suggest=0.35,  # provisional: calibrate with scripts/calibrate_voiceprints.py
    margin=0.05,  # provisional: calibrate with scripts/calibrate_voiceprints.py
    sha256="41d5ad6b5c758a03e46ab53388f42394f40bf115aa9d9df25d4adff6e21072ef",
    size=83_136_305,
    filename="embedding_model.pth",
    checkpoint_env="VOXWEAVE_ANIME_VA_CKPT",
    loader=_load_anime_va,
    hf_repo="litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm",
    hf_revision="1677c9702cca7aca7dc5a74b3c76f3c8b05969b7",
)

EMBEDDERS: dict[str, EmbedderSpec] = {
    spec.name: spec for spec in (REDIMNET2_B6, ANIME_VA)
}
AUTO = "auto"
# Short names a user types; full spec names are accepted too.
ALIASES: dict[str, str] = {
    AUTO: AUTO,
    "redimnet2": REDIMNET2_B6.name,
    "anime-va": ANIME_VA.name,
    "pyannote": LEGACY.name,
}
CHOICES: tuple[str, ...] = tuple(ALIASES)


def normalize_voiceprint_choice(
    value: str, *, source: str = "--voiceprint-model"
) -> str:
    """Map a user value to ``auto``, ``pyannote`` or a registered embedder name."""
    key = value.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    if key in EMBEDDERS:
        return key
    raise ValueError(
        f"{source} has unknown voiceprint model {value!r}; "
        f"choose one of {', '.join(CHOICES)}"
    )


def resolve_voiceprint_choice(cli_value: str | None = None) -> str:
    """Resolve the configured voiceprint choice before the language is known.

    Precedence: CLI value, ``VOXWEAVE_VOICEPRINT_MODEL``, ``[voiceprint].model``,
    then ``auto``. An unknown value raises ``ValueError`` naming its source
    instead of silently falling back to another embedding space.
    """
    if cli_value is not None and cli_value.strip():
        return normalize_voiceprint_choice(cli_value)
    env = os.environ.get(ENV_MODEL)
    if env is not None and env.strip():
        return normalize_voiceprint_choice(env, source=f"environment {ENV_MODEL}")
    conf = config.conf_voiceprint_model()
    if conf is not None:
        return normalize_voiceprint_choice(conf, source="config [voiceprint].model")
    return AUTO


def resolve_voiceprint_model(
    cli_value: str | None,
    language_iso: str | None,
) -> EmbedderSpec | LegacyLane:
    """Pick the voiceprint embedder for one run.

    ``auto`` routes Japanese to the anime voice-actor model and every other
    language to ReDimNet2; ``pyannote`` returns :data:`LEGACY`.
    """
    choice = resolve_voiceprint_choice(cli_value)
    language = (language_iso or "").strip().lower()
    if choice == AUTO:
        return ANIME_VA if language == "ja" else REDIMNET2_B6
    if choice == LEGACY.name:
        return LEGACY
    spec = EMBEDDERS[choice]
    if spec.languages is not None and language not in spec.languages:
        log.warning(
            "voiceprint model %s is trained for %s only; this episode is %s",
            spec.name,
            "/".join(spec.languages),
            language or "of unknown language",
        )
    return spec


def spec_by_name(name: object) -> EmbedderSpec | None:
    """Registered embedder recorded as ``embedding_model`` in provenance, if any."""
    return EMBEDDERS.get(name) if isinstance(name, str) else None


# --------------------------------------------------------------------------
# Checkpoint acquisition
# --------------------------------------------------------------------------


def _open_url(url: str, timeout: float) -> Any:
    """Open ``url`` for streaming (the network boundary tests replace)."""
    request = urllib.request.Request(url, headers={"User-Agent": "voxweave"})
    return urllib.request.urlopen(request, timeout=timeout)


def _download_url(spec: EmbedderSpec, target: Path) -> None:
    """Stream ``spec.url`` into ``target``, hashing the bytes as they arrive.

    Every connect and read is bounded by DOWNLOAD_TIMEOUT_SECONDS. The bytes go
    to a temp file next to ``target`` that is renamed onto it only once their
    size and SHA-256 match the pin, so an interrupted, stalled or tampered
    download never leaves a partial or foreign file in the cache.
    """
    if spec.url is None:
        raise VoiceEmbeddingError(f"voiceprint model {spec.name} has no URL")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    received = 0
    with fsio.atomic_path(target) as partial:
        with (
            _open_url(spec.url, DOWNLOAD_TIMEOUT_SECONDS) as response,
            open(partial, "wb") as sink,
        ):
            while True:
                chunk = response.read(_IO_CHUNK_BYTES)
                if not chunk:
                    break
                received += len(chunk)
                if received > spec.size:
                    raise VoiceEmbeddingError(
                        f"the server sent more than the pinned {spec.size} bytes"
                    )
                digest.update(chunk)
                sink.write(chunk)
        if received != spec.size:
            raise VoiceEmbeddingError(
                f"the download stopped after {received} of {spec.size} bytes"
            )
        if digest.hexdigest() != spec.sha256:
            raise VoiceEmbeddingError(
                f"the download has SHA-256 {digest.hexdigest()}, but "
                f"{spec.name} is pinned to {spec.sha256}"
            )


def _download_hf(repo: str, filename: str, revision: str | None) -> Path:
    # huggingface_hub bounds its own requests (HF_HUB_DOWNLOAD_TIMEOUT /
    # HF_HUB_ETAG_TIMEOUT) and resolves a cached pinned commit without the
    # network.
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo,
            filename,
            revision=revision,
            cache_dir=config.AUDIO_CACHE,
        )
    )


def checkpoint_path(spec: EmbedderSpec) -> Path:
    """Local checkpoint for ``spec``: the env override, the cache, or a download.

    The env override (``spec.checkpoint_env``) names an explicit local file and
    skips the network entirely; it must still be the pinned checkpoint.
    """
    override = os.environ.get(spec.checkpoint_env, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise VoiceEmbeddingError(
                f"{spec.checkpoint_env}={override!r} is not a file"
            )
        return path
    manual = (
        f"download it from {spec.source} and point {spec.checkpoint_env} at the "
        "file, or check the network"
    )
    if spec.url is not None:
        target = Path(config.AUDIO_CACHE) / (spec.cache_subdir or spec.name)
        target = target / spec.filename
        if target.is_file():
            return target
        log.info("downloading voiceprint model %s -> %s", spec.name, target)
        try:
            _download_url(spec, target)
        except Exception as exc:  # noqa: BLE001 -- network/size/hash failures, one message
            raise VoiceEmbeddingError(
                f"could not download voiceprint model {spec.name}: {exc}; {manual}"
            ) from exc
        return target
    if spec.hf_repo is None:
        raise VoiceEmbeddingError(f"voiceprint model {spec.name} has no source")
    try:
        return _download_hf(spec.hf_repo, spec.filename, spec.hf_revision)
    except Exception as exc:  # noqa: BLE001 -- hub/network failures, one message
        raise VoiceEmbeddingError(
            f"could not download voiceprint model {spec.name}: {exc}; {manual}"
        ) from exc


def _read_pinned(spec: EmbedderSpec, path: Path, *, keep: bool) -> bytearray | None:
    """Stream ``path`` through SHA-256 and prove it is the pinned checkpoint.

    The size is checked against the pin before anything is read, so a wrong
    (possibly huge) file is refused up front; at most ``spec.size`` bytes are
    ever buffered. With ``keep`` the verified bytes are returned.
    """
    repair = (
        f"delete the file to re-download it, or point {spec.checkpoint_env} at "
        "the pinned checkpoint"
    )
    digest = hashlib.sha256()
    payload: bytearray | None = None
    offset = 0
    try:
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size != spec.size:
                raise VoiceEmbeddingError(
                    f"voiceprint checkpoint {path} is {size} bytes, but "
                    f"{spec.name} is pinned to a {spec.size}-byte file; {repair}"
                )
            if keep:
                payload = bytearray(spec.size)
            while True:
                chunk = handle.read(_IO_CHUNK_BYTES)
                if not chunk:
                    break
                end = offset + len(chunk)
                if end > spec.size:
                    raise VoiceEmbeddingError(
                        f"voiceprint checkpoint {path} grew while it was read"
                    )
                digest.update(chunk)
                if payload is not None:
                    payload[offset:end] = chunk
                offset = end
    except (MemoryError, OSError) as exc:
        raise VoiceEmbeddingError(
            f"could not read voiceprint checkpoint {path}: {exc}"
        ) from exc
    if offset != spec.size:
        raise VoiceEmbeddingError(
            f"voiceprint checkpoint {path} shrank while it was read"
        )
    if digest.hexdigest() != spec.sha256:
        raise VoiceEmbeddingError(
            f"voiceprint checkpoint {path} has SHA-256 {digest.hexdigest()}, but "
            f"{spec.name} is pinned to {spec.sha256}; {repair}"
        )
    return payload


def verify_checkpoint(spec: EmbedderSpec, path: Path) -> None:
    """Prove ``path`` is the pinned checkpoint without keeping its bytes."""
    _read_pinned(spec, Path(path), keep=False)


def read_verified_checkpoint(spec: EmbedderSpec, path: Path) -> bytearray:
    """Read ``path`` once and prove its bytes are the pinned checkpoint."""
    payload = _read_pinned(spec, Path(path), keep=True)
    assert payload is not None
    return payload


def prefetch_specs(
    cli_value: str | None, language_iso: str | None
) -> tuple[EmbedderSpec, ...]:
    """The embedders a voiceprint run may need, before its language is detected.

    ``auto`` needs both routes unless the language is already fixed (``--lang``);
    an explicit embedder needs only itself; the legacy lane needs none.
    """
    choice = resolve_voiceprint_choice(cli_value)
    if choice == LEGACY.name:
        return ()
    if choice != AUTO:
        return (EMBEDDERS[choice],)
    language = (language_iso or "").strip().lower()
    if language:
        return (ANIME_VA if language == "ja" else REDIMNET2_B6,)
    return (REDIMNET2_B6, ANIME_VA)


def prefetch_checkpoints(
    cli_value: str | None, language_iso: str | None
) -> tuple[EmbedderSpec, ...]:
    """Download (when missing) and verify every checkpoint the run may need.

    Meant for the start of a run: a missing network then fails in seconds,
    before minutes of separation and ASR, instead of at the capture step. A
    cached checkpoint is only re-hashed. Raises :class:`VoiceEmbeddingError`;
    an invalid model choice raises ``ValueError``.
    """
    specs = prefetch_specs(cli_value, language_iso)
    for spec in specs:
        verify_checkpoint(spec, checkpoint_path(spec))
    return specs


# --------------------------------------------------------------------------
# Resident model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedEmbedder:
    """A constructed network bound to the checkpoint bytes it was built from."""

    spec: EmbedderSpec
    checkpoint_sha256: str
    network: Any
    device: Any

    def embed_samples(self, samples: np.ndarray) -> np.ndarray:
        """One forward pass over one mono segment -> unit vector (float64)."""
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        tensor = tensor.reshape(1, -1).to(self.device)
        with torch.inference_mode():
            output = self.network(tensor)
        row = np.asarray(output.detach().float().cpu().numpy(), dtype=np.float64)
        row = row.reshape(-1)
        if row.size != self.spec.embedding_dim:
            raise VoiceEmbeddingError(
                f"{self.spec.name} returned a {row.size}-dim vector, "
                f"expected {self.spec.embedding_dim}"
            )
        return unit_vector(row, what=f"{self.spec.name} embedding")


_resident: LoadedEmbedder | None = None
# The speakers server runs requests on threads and the pipeline is single
# threaded; one lock serializes construction and forward passes on the shared
# module either way.
_lock = threading.RLock()


def _construct(spec: EmbedderSpec) -> LoadedEmbedder:
    path = checkpoint_path(spec)
    payload = read_verified_checkpoint(spec, path)
    try:
        import torch
    except ImportError as exc:
        raise VoiceEmbeddingError("voiceprint embedders require torch") from exc
    try:
        # weights_only=True loads both pinned checkpoints (plain tensors plus a
        # primitive model_config dict). There is deliberately no
        # weights_only=False fallback: the SHA-256 pin fixes the exact bytes,
        # so a fallback could only ever run on bytes other than the ones that
        # were verified to load safely.
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        network, declared_dim = spec.loader(state)
    except Exception as exc:  # noqa: BLE001 -- unpickling/layout errors, one message
        raise VoiceEmbeddingError(
            f"could not load voiceprint model {spec.name} from {path}: {exc}"
        ) from exc
    if declared_dim != spec.embedding_dim:
        raise VoiceEmbeddingError(
            f"{spec.name} declares {declared_dim}-dim embeddings, "
            f"registry expects {spec.embedding_dim}"
        )
    if not MIN_EMBEDDING_DIM <= declared_dim <= MAX_EMBEDDING_DIM:
        raise VoiceEmbeddingError(
            f"{spec.name} embedding dimension {declared_dim} is unsupported"
        )
    device = torch.device(runtime.get_device())
    try:
        network = network.to(device).eval()
    except Exception as exc:  # noqa: BLE001
        raise VoiceEmbeddingError(
            f"could not move voiceprint model {spec.name} to {device}: {exc}"
        ) from exc
    log.info("loaded voiceprint model %s on %s", spec.name, device)
    return LoadedEmbedder(
        spec=spec,
        # read_verified_checkpoint proved these exact bytes hash to the pin.
        checkpoint_sha256=spec.sha256,
        network=network,
        device=device,
    )


def embedder_lock() -> threading.RLock:
    """The (re-entrant) lock serializing the resident embedder.

    Hold it across ``get_embedder`` + inference when the loaded checkpoint must
    be attested for exactly the vectors it produced.
    """
    return _lock


def get_embedder(spec: EmbedderSpec) -> LoadedEmbedder:
    """Process-singleton embedder keyed by spec name + pinned checkpoint."""
    global _resident
    with _lock:
        resident = _resident
        if (
            resident is not None
            and resident.spec.name == spec.name
            and resident.checkpoint_sha256 == spec.sha256
        ):
            return resident
        _release_locked()
        _resident = _construct(spec)
        return _resident


def release() -> None:
    """Drop the resident embedder and free its VRAM (mirrors backend.release)."""
    with _lock:
        _release_locked()


def _release_locked() -> None:
    global _resident
    had_model = _resident is not None
    _resident = None
    if not had_model:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


# --------------------------------------------------------------------------
# Audio + embedding
# --------------------------------------------------------------------------


def read_mono_16k(wav_path: Path) -> np.ndarray:
    """Decode a wav to the float32 16 kHz mono samples pyannote was given."""
    import soundfile as sf

    try:
        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise VoiceEmbeddingError(f"could not read audio {wav_path}: {exc}") from exc
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise VoiceEmbeddingError(f"audio is empty: {wav_path}")
    mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
    if int(sample_rate) != SAMPLE_RATE:
        import torch
        import torchaudio.functional as audio_functional

        mono = (
            audio_functional.resample(
                torch.from_numpy(np.ascontiguousarray(mono)),
                int(sample_rate),
                SAMPLE_RATE,
            )
            .cpu()
            .numpy()
        )
    return np.ascontiguousarray(mono, dtype=np.float32)


def unit_vector(value: object, *, what: str = "vector") -> np.ndarray:
    """L2-normalize a finite, non-zero 1-D vector (float64)."""
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VoiceEmbeddingError(f"{what} is not numeric") from exc
    if vector.size == 0 or not np.isfinite(vector).all():
        raise VoiceEmbeddingError(f"{what} is empty or non-finite")
    norm = math.sqrt(math.fsum(float(item) * float(item) for item in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise VoiceEmbeddingError(f"{what} is zero or non-finite")
    return vector / norm


def _span_samples(
    samples: np.ndarray, start: float, end: float, *, index: int
) -> tuple[int, int]:
    if not (math.isfinite(start) and math.isfinite(end)) or not 0 <= start < end:
        raise VoiceEmbeddingError(f"segment {index} must satisfy 0 <= start < end")
    first = max(0, math.floor(start * SAMPLE_RATE))
    last = min(len(samples), math.ceil(end * SAMPLE_RATE))
    if first >= len(samples) or last <= first:
        raise VoiceEmbeddingError(f"segment {index} falls outside the audio")
    return first, last


def window_bounds(first: int, last: int) -> list[tuple[int, int]]:
    """Split ``[first, last)`` into equal windows no longer than WINDOW_SECONDS."""
    length = last - first
    limit = max(1, round(WINDOW_SECONDS * SAMPLE_RATE))
    count = max(1, math.ceil(length / limit))
    edges = [first + (length * step) // count for step in range(count + 1)]
    return [(edges[k], edges[k + 1]) for k in range(count) if edges[k + 1] > edges[k]]


def _padded(segment: np.ndarray, minimum: int) -> np.ndarray:
    """Repeat a short segment up to ``minimum`` samples.

    Neither embedder masks its statistics pooling, so zero padding would pool
    silence into the voice statistics; cyclic repetition only reuses speech.
    """
    if len(segment) >= minimum:
        return segment
    return np.resize(segment, minimum)


def embed_segments(
    waveform: Any,
    spans: Sequence[tuple[float, float]],
    spec: EmbedderSpec,
) -> np.ndarray:
    """Embed each ``(start, end)`` span of 16 kHz mono audio -> ``[N, D]``.

    Rows are L2-normalized. Each span is its own forward pass (window by
    window for long spans): ReDimNet2's attentive pooling has no padding mask,
    so mixed lengths are never batched together.
    """
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise VoiceEmbeddingError("audio is empty")
    minimum = max(1, math.ceil(spec.min_seconds * SAMPLE_RATE))
    rows: list[np.ndarray] = []
    with _lock:
        embedder = get_embedder(spec)
        for index, (start, end) in enumerate(spans):
            first, last = _span_samples(samples, float(start), float(end), index=index)
            windows = window_bounds(first, last)
            vectors = []
            weights = []
            for low, high in windows:
                try:
                    vectors.append(
                        embedder.embed_samples(_padded(samples[low:high], minimum))
                    )
                except VoiceEmbeddingError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- inference errors (OOM, ...)
                    raise VoiceEmbeddingError(
                        f"{spec.name} inference failed for segment {index}: {exc}"
                    ) from exc
                weights.append(float(high - low))
            rows.append(weighted_unit_mean(vectors, weights))
    if not rows:
        return np.zeros((0, spec.embedding_dim), dtype=np.float64)
    return np.stack(rows)


def weighted_unit_mean(
    vectors: Sequence[Sequence[float]] | np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    """Weighted mean of unit vectors, re-normalized to unit length."""
    matrix = np.asarray(vectors, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or weight.shape != (matrix.shape[0],)
        or not np.isfinite(weight).all()
        or (weight <= 0.0).any()
    ):
        raise VoiceEmbeddingError("weighted centroid inputs are malformed")
    rows = np.stack([unit_vector(row) for row in matrix])
    return unit_vector((rows * weight[:, None]).sum(axis=0) / weight.sum())


# --------------------------------------------------------------------------
# centroid-v1 recipe
# --------------------------------------------------------------------------

Turn = tuple[float, float, str]


def _subtract(
    start: float, end: float, blockers: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for block_start, block_end in blockers:
        if block_end <= start or block_start >= end:
            continue
        next_pieces: list[tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if block_end <= piece_start or block_start >= piece_end:
                next_pieces.append((piece_start, piece_end))
                continue
            if block_start > piece_start:
                next_pieces.append((piece_start, block_start))
            if block_end < piece_end:
                next_pieces.append((block_end, piece_end))
        pieces = next_pieces
    return pieces


def centroid_segments(turns: Sequence[Turn], label: str) -> list[tuple[float, float]]:
    """The centroid-v1 segments of one speaker, in chronological order.

    Each of the speaker's turns loses every stretch another speaker's turn
    overlaps. Pieces of at least MIN_TURN_SECONDS qualify; with none, the
    pieces of at least FALLBACK_SEGMENT_SECONDS do. The longest
    MAX_SEGMENTS_PER_SPEAKER survive (earlier first on ties). An empty result
    means the speaker gets no voiceprint.
    """
    others = [
        (float(start), float(end))
        for start, end, other in turns
        if other != label and end > start
    ]
    pieces: list[tuple[float, float]] = []
    for start, end, owner in turns:
        if owner == label and end > start:
            pieces.extend(_subtract(float(start), float(end), others))
    qualifying = [p for p in pieces if p[1] - p[0] >= MIN_TURN_SECONDS]
    if not qualifying:
        qualifying = [p for p in pieces if p[1] - p[0] >= FALLBACK_SEGMENT_SECONDS]
    ranked = sorted(qualifying, key=lambda p: (-(p[1] - p[0]), p[0], p[1]))
    return sorted(ranked[:MAX_SEGMENTS_PER_SPEAKER])


def speaker_centroids(
    waveform: Any,
    turns: Sequence[Turn],
    spec: EmbedderSpec,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """centroid-v1 voiceprints for ``labels`` (default: every turn label).

    A centroid is the duration-weighted mean of the unit vectors of the
    speaker's centroid segments, re-normalized. Speakers without a segment, or
    whose centroid fails the shared vector law, are left out.
    """
    wanted = sorted({label for _s, _e, label in turns} if labels is None else labels)
    centroids: dict[str, list[float]] = {}
    for label in wanted:
        segments = centroid_segments(turns, label)
        if not segments:
            log.debug("speaker %s has no voiceprint segment", label)
            continue
        vectors = embed_segments(waveform, segments, spec)
        centroid = weighted_unit_mean(vectors, [end - start for start, end in segments])
        values = [float(value) for value in centroid]
        try:
            validate_vector(values, dim=spec.embedding_dim, field=f"speakers.{label}")
        except Phase2DataError as exc:
            log.warning("dropping voiceprint for speaker %s: %s", label, exc)
            continue
        centroids[label] = values
    return centroids


def decoupled_provenance(
    base: Mapping[str, object],
    spec: EmbedderSpec,
    *,
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Voiceprint provenance of the decoupled lane.

    The diarization fields of ``base`` (pipeline id, config digest, pyannote and
    torch versions, audio profile) stay for information; the embedding fields
    now describe the dedicated embedder, which alone defines compatibility.
    """
    provenance = dict(base)
    provenance.update(
        {
            "embedding_lane": LANE_DECOUPLED,
            "embedding_model": spec.name,
            "embedding_checkpoint": checkpoint_sha256,
            "embedding_dim": spec.embedding_dim,
            "embedding_recipe": CENTROID_RECIPE,
        }
    )
    return provenance


def capture_voiceprints(
    wav_path: Path,
    turns: Sequence[Turn],
    spec: EmbedderSpec,
    base_provenance: Mapping[str, object],
) -> tuple[dict[str, list[float]], dict[str, object]] | None:
    """Centroids + provenance for one diarized episode, or ``None`` if empty.

    ``wav_path`` must be the exact 16 kHz mono audio the diarizer saw, so the
    voiceprints describe the same (separated / normalized) signal its
    provenance ``audio`` block records.
    """
    if not turns:
        return None
    waveform = read_mono_16k(Path(wav_path))
    centroids = speaker_centroids(waveform, turns, spec)
    if not centroids:
        return None
    with _lock:
        checkpoint_sha256 = get_embedder(spec).checkpoint_sha256
    return centroids, decoupled_provenance(
        base_provenance, spec, checkpoint_sha256=checkpoint_sha256
    )


__all__ = [
    "ALIASES",
    "ANIME_VA",
    "AUTO",
    "CENTROID_RECIPE",
    "CHOICES",
    "DOWNLOAD_TIMEOUT_SECONDS",
    "EMBEDDERS",
    "ENV_MODEL",
    "EmbedderSpec",
    "FALLBACK_SEGMENT_SECONDS",
    "LANE_DECOUPLED",
    "LEGACY",
    "LegacyLane",
    "LoadedEmbedder",
    "MAX_SEGMENTS_PER_SPEAKER",
    "MIN_TURN_SECONDS",
    "REDIMNET2_B6",
    "VoiceEmbeddingError",
    "WINDOW_SECONDS",
    "capture_voiceprints",
    "centroid_segments",
    "checkpoint_path",
    "decoupled_provenance",
    "embed_segments",
    "embedder_lock",
    "get_embedder",
    "normalize_voiceprint_choice",
    "prefetch_checkpoints",
    "prefetch_specs",
    "read_mono_16k",
    "read_verified_checkpoint",
    "release",
    "resolve_voiceprint_choice",
    "resolve_voiceprint_model",
    "speaker_centroids",
    "spec_by_name",
    "unit_vector",
    "verify_checkpoint",
    "weighted_unit_mean",
    "window_bounds",
]
