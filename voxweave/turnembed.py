"""Turn-level speaker embeddings and deterministic two-way clustering."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import operator
import os
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voxweave import config, runtime
from voxweave.voicebase import MAX_EMBEDDING_DIM, MIN_EMBEDDING_DIM

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_CHECKPOINT_FILE = "pytorch_model.bin"
SAMPLE_RATE = 16_000
MIN_TURN_SECONDS = 2.0
MAX_LLOYD_ITERATIONS = 32
EIGENGAP_ABSOLUTE_TOLERANCE = 1e-12
# Keep a healthy margin above float64/BLAS round-off: below this relative gap,
# tiny platform-level covariance changes can select a different principal axis.
EIGENGAP_RELATIVE_TOLERANCE = 1e-9


class TurnEmbeddingError(RuntimeError):
    """Turn audio could not be converted into usable speaker embeddings."""


class UnsplittableSpeakerError(TurnEmbeddingError):
    """The supplied turns do not contain evidence for two distinct clusters."""


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Exact embedding model and checkpoint bytes bound during construction."""

    model: str
    checkpoint_sha256: str
    pyannote_version: str


class AttestedTurnEmbeddings(dict[int, list[float]]):
    """Dictionary-compatible embeddings bound to their resident model identity."""

    __slots__ = ("__identity",)

    def __init__(
        self,
        values: Mapping[int, list[float]],
        *,
        identity: EmbeddingIdentity,
    ) -> None:
        super().__init__(values)
        self.__identity = identity

    @property
    def identity(self) -> EmbeddingIdentity:
        """Frozen identity of the inference instance that produced these rows."""
        return self.__identity


@dataclass(frozen=True, slots=True)
class _AudioMetadata:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 0
    encoding: str = ""


_inference: Any | None = None
_inference_identity: EmbeddingIdentity | None = None
# SpeakerHTTPServer dispatches requests on threads. The model loader and the
# underlying torch module are both process singletons, so serialize construction
# and forward calls rather than racing the same GPU module from two /split posts.
_inference_lock = threading.RLock()


def _configure_pyannote_import() -> Path:
    """Set the private cache and restore torchaudio APIs pyannote 3.x imports."""
    os.environ.setdefault(
        "PYANNOTE_CACHE",
        os.fspath(Path(config.AUDIO_CACHE) / "pyannote"),
    )
    cache_dir = Path(os.environ["PYANNOTE_CACHE"]).expanduser()
    import torchaudio

    module: Any = torchaudio
    if not hasattr(module, "AudioMetaData"):
        module.AudioMetaData = _AudioMetadata
    if not hasattr(module, "info"):

        def info(filepath: object, *_args: object, **_kwargs: object) -> _AudioMetadata:
            import soundfile as sf

            metadata = sf.info(str(filepath))
            return _AudioMetadata(
                sample_rate=int(metadata.samplerate),
                num_frames=int(metadata.frames),
                num_channels=int(metadata.channels),
            )

        module.info = info
    if not hasattr(module, "list_audio_backends"):
        module.list_audio_backends = lambda: ["soundfile"]
    return cache_dir


def _download_checkpoint(cache_dir: Path, token: str) -> Path:
    """Resolve the exact PyTorch checkpoint through the current HF token API."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise TurnEmbeddingError(
            "speaker splitting requires the huggingface-hub dependency"
        ) from exc
    try:
        checkpoint = hf_hub_download(
            repo_id=EMBEDDING_MODEL,
            filename=EMBEDDING_CHECKPOINT_FILE,
            cache_dir=cache_dir,
            token=token,
        )
    except Exception as exc:
        raise TurnEmbeddingError(
            f"could not download speaker embedding model {EMBEDDING_MODEL}: {exc}"
        ) from exc
    return Path(checkpoint)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pyannote_version() -> str:
    try:
        return importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError:
        return "unresolved"


def _construct_bound_checkpoint(
    checkpoint: Path,
    construct: Callable[[Path], Any],
) -> tuple[Any, str]:
    """Construct from one resolved file and refuse a concurrent byte change."""
    try:
        resolved = Path(checkpoint).resolve(strict=True)
        before = _sha256_file(resolved)
    except OSError as exc:
        raise TurnEmbeddingError(
            "speaker embedding checkpoint could not be bound before construction"
        ) from exc
    value = construct(resolved)
    try:
        after = _sha256_file(resolved)
    except OSError as exc:
        raise TurnEmbeddingError(
            "speaker embedding checkpoint changed during construction"
        ) from exc
    if before != after:
        raise TurnEmbeddingError(
            "speaker embedding checkpoint changed during construction"
        )
    return value, before


def _load_inference() -> tuple[Any, EmbeddingIdentity]:
    """Load the production embedding family lazily on the best torch device."""
    token = config.conf_hf_token()
    if not token:
        raise TurnEmbeddingError(
            "speaker splitting needs the Hugging Face token used for diarization; "
            "set VOXWEAVE_HF_TOKEN / HF_TOKEN or run `hf auth login`"
        )
    cache_dir = _configure_pyannote_import()
    checkpoint = _download_checkpoint(cache_dir, token)
    try:
        import torch
        from pyannote.audio import Model  # pyright: ignore[reportMissingImports]
        from pyannote.audio.core.task import (  # pyright: ignore[reportMissingImports]
            Problem,
            Resolution,
            Specifications,
        )
        from pyannote.audio.pipelines.speaker_verification import (  # pyright: ignore[reportMissingImports]
            PretrainedSpeakerEmbedding,
        )
        from torch.torch_version import TorchVersion
    except (ImportError, AttributeError) as exc:
        raise TurnEmbeddingError(
            "speaker splitting could not import the pyannote embedding runtime"
        ) from exc

    device = torch.device(runtime.get_device())
    safe_globals = getattr(torch.serialization, "safe_globals", None)

    def construct(bound_checkpoint: Path) -> Any:
        if safe_globals is None:
            return Model.from_pretrained(
                bound_checkpoint,
                map_location=device,
                strict=False,
            )
        with safe_globals([TorchVersion, Specifications, Problem, Resolution]):
            return Model.from_pretrained(
                bound_checkpoint,
                map_location=device,
                strict=False,
            )

    try:
        model, checkpoint_sha256 = _construct_bound_checkpoint(
            checkpoint,
            construct,
        )
    except TurnEmbeddingError:
        raise
    except Exception as exc:
        raise TurnEmbeddingError(
            f"could not load speaker embedding model {EMBEDDING_MODEL}: {exc}"
        ) from exc
    if model is None:
        raise TurnEmbeddingError(
            f"could not load speaker embedding model {EMBEDDING_MODEL}"
        )
    try:
        inference = PretrainedSpeakerEmbedding(model, device=device)
    except Exception as exc:
        raise TurnEmbeddingError(
            f"could not initialize speaker embedding model {EMBEDDING_MODEL}: {exc}"
        ) from exc
    try:
        sample_rate = operator.index(inference.sample_rate)
    except (AttributeError, TypeError) as exc:
        raise TurnEmbeddingError(
            "speaker embedding model did not declare an integer sample rate"
        ) from exc
    if sample_rate != SAMPLE_RATE:
        raise TurnEmbeddingError(
            f"speaker embedding model requires {sample_rate} Hz audio, expected {SAMPLE_RATE} Hz"
        )
    return inference, EmbeddingIdentity(
        model=EMBEDDING_MODEL,
        checkpoint_sha256=checkpoint_sha256,
        pyannote_version=_pyannote_version(),
    )


def _get_inference() -> Any:
    global _inference, _inference_identity
    with _inference_lock:
        if _inference is None:
            loaded, identity = _load_inference()
            _inference_identity = identity
            _inference = loaded
        elif _inference_identity is None:
            raise TurnEmbeddingError(
                "speaker embedding inference has no checkpoint identity"
            )
        return _inference


def _turn_bounds(turn: object, index: int) -> tuple[float, float]:
    if not isinstance(turn, (list, tuple)) or len(turn) < 2:
        raise TurnEmbeddingError(f"turn {index} must contain start and end bounds")
    start, end = turn[0], turn[1]
    if type(start) not in (int, float) or type(end) not in (int, float):
        raise TurnEmbeddingError(f"turn {index} bounds must be numbers")
    start_f, end_f = float(start), float(end)
    if not math.isfinite(start_f) or not math.isfinite(end_f):
        raise TurnEmbeddingError(f"turn {index} bounds must be finite")
    if not 0 <= start_f < end_f:
        raise TurnEmbeddingError(f"turn {index} must satisfy 0 <= start < end")
    return start_f, end_f


def _read_mono_16k(wav_path: Path) -> np.ndarray:
    import soundfile as sf

    try:
        samples, sample_rate = sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise TurnEmbeddingError(
            f"could not read turn audio {wav_path}: {exc}"
        ) from exc
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise TurnEmbeddingError(f"turn audio is empty: {wav_path}")
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


def _normalized_vector(value: object, *, field: str) -> list[float]:
    candidate: Any = value
    raw = candidate.tolist() if hasattr(candidate, "tolist") else candidate
    try:
        vector = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TurnEmbeddingError(f"{field} is not a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise TurnEmbeddingError(f"{field} is zero or non-finite")
    values = [float(item) for item in vector]
    norm = math.sqrt(math.fsum(item * item for item in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise TurnEmbeddingError(f"{field} is zero or non-finite")
    return [item / norm for item in values]


def _embedding_row(value: object, *, index: int) -> list[float]:
    candidate: Any = value
    raw = candidate.tolist() if hasattr(candidate, "tolist") else candidate
    try:
        rows = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TurnEmbeddingError(
            f"turn embedding {index} is not a numeric matrix"
        ) from exc
    if rows.ndim != 2 or rows.shape[0] != 1:
        raise TurnEmbeddingError(
            f"turn embedding {index} must contain exactly one model row"
        )
    return _normalized_vector(rows[0], field=f"turn embedding {index}")


def _minimum_samples(inference: object) -> int:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"std\(\): degrees of freedom is <= 0",
            )
            minimum = operator.index(getattr(inference, "min_num_samples"))
    except (AttributeError, TypeError) as exc:
        raise TurnEmbeddingError(
            "speaker embedding model did not declare an integer minimum input length"
        ) from exc
    if minimum <= 0:
        raise TurnEmbeddingError(
            "speaker embedding model declared an invalid minimum input length"
        )
    return max(minimum, round(MIN_TURN_SECONDS * SAMPLE_RATE))


def turn_embeddings(
    wav_path: Path,
    turns: Sequence[object],
) -> AttestedTurnEmbeddings:
    """Return one normalized embedding for each turn index.

    Audio is decoded to 16 kHz mono in memory. Turns shorter than two seconds
    or the model's safe lower bound are right-padded with silence. Imports and
    model construction stay lazy so CPU-only tests can replace the inference
    boundary without importing pyannote.
    """
    if not turns:
        with _inference_lock:
            _get_inference()
            identity = _inference_identity
            if identity is None:
                raise TurnEmbeddingError(
                    "speaker embedding inference has no checkpoint identity"
                )
            return AttestedTurnEmbeddings({}, identity=identity)
    waveform = _read_mono_16k(Path(wav_path))
    if waveform.size == 0:
        raise TurnEmbeddingError(f"turn audio is empty: {wav_path}")
    import torch

    result: dict[int, list[float]] = {}
    dimension: int | None = None
    with _inference_lock:
        inference = _get_inference()
        identity = _inference_identity
        if identity is None:
            raise TurnEmbeddingError(
                "speaker embedding inference has no checkpoint identity"
            )
        minimum_samples = _minimum_samples(inference)
        for index, turn in enumerate(turns):
            start, end = _turn_bounds(turn, index)
            first = max(0, math.floor(start * SAMPLE_RATE))
            last = min(len(waveform), math.ceil(end * SAMPLE_RATE))
            if first >= len(waveform) or last <= first:
                raise TurnEmbeddingError(f"turn {index} falls outside the audio")
            segment = waveform[first:last]
            if len(segment) < minimum_samples:
                segment = np.pad(segment, (0, minimum_samples - len(segment)))
            tensor = torch.from_numpy(np.ascontiguousarray(segment)).reshape(1, 1, -1)
            try:
                embedded = inference(tensor)
            except Exception as exc:
                raise TurnEmbeddingError(
                    f"speaker embedding inference failed for turn {index}: {exc}"
                ) from exc
            vector = _embedding_row(embedded, index=index)
            if not MIN_EMBEDDING_DIM <= len(vector) <= MAX_EMBEDDING_DIM:
                raise TurnEmbeddingError(
                    f"turn embedding {index} has unsupported dimension {len(vector)}"
                )
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise TurnEmbeddingError("turn embedding model returned ragged vectors")
            result[index] = vector
        return AttestedTurnEmbeddings(result, identity=identity)


def normalized_centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Return the unit-normalized arithmetic mean of compatible vectors."""
    if not vectors:
        raise UnsplittableSpeakerError("each proposed group must contain a turn")
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TurnEmbeddingError("centroid inputs must be numeric vectors") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise TurnEmbeddingError("centroid inputs must be finite, equal-length vectors")
    return _normalized_vector(matrix.mean(axis=0), field="speaker centroid")


def bisect_embeddings(
    embeddings: Mapping[int, Sequence[float]],
) -> dict[int, str]:
    """Bisect normalized turn vectors with deterministic principal-axis 2-means."""
    if len(embeddings) < 2:
        raise UnsplittableSpeakerError("a speaker needs at least two turns to split")
    raw_keys = list(embeddings)
    if any(type(index) is not int or index < 0 for index in raw_keys):
        raise TurnEmbeddingError("embedding indexes must be non-negative integers")
    keys = sorted(raw_keys)
    try:
        matrix = np.asarray([embeddings[index] for index in keys], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TurnEmbeddingError("turn embeddings must be numeric vectors") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise TurnEmbeddingError("turn embeddings must be finite, equal-length vectors")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise TurnEmbeddingError("turn embeddings must be non-zero vectors")
    matrix = matrix / norms[:, None]
    centered = matrix - matrix.mean(axis=0)
    covariance = centered.T @ centered / len(keys)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    leading = float(eigenvalues[-1])
    if not math.isfinite(leading) or leading <= EIGENGAP_ABSOLUTE_TOLERANCE:
        raise UnsplittableSpeakerError(
            "these turns have indistinguishable voice embeddings"
        )
    if len(eigenvalues) > 1:
        runner_up = float(eigenvalues[-2])
        if not math.isfinite(runner_up):
            raise UnsplittableSpeakerError(
                "these turns have an ambiguous principal voice axis"
            )
        eigengap = leading - runner_up
        eigengap_tolerance = EIGENGAP_ABSOLUTE_TOLERANCE + (
            EIGENGAP_RELATIVE_TOLERANCE * max(abs(leading), abs(runner_up))
        )
        if eigengap <= eigengap_tolerance:
            raise UnsplittableSpeakerError(
                "these turns have an ambiguous principal voice axis"
            )
    axis = eigenvectors[:, -1]
    anchor = int(np.argmax(np.abs(axis)))
    if axis[anchor] < 0:
        axis = -axis
    scores = centered @ axis
    labels = (scores >= 0.0).astype(np.int8)
    if len(set(labels.tolist())) != 2:
        raise UnsplittableSpeakerError(
            "these turns do not form two distinct voice groups"
        )

    for _iteration in range(MAX_LLOYD_ITERATIONS):
        centroids = np.stack(
            [matrix[labels == cluster].mean(axis=0) for cluster in (0, 1)]
        )
        distances = np.sum((matrix[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        updated = labels.copy()
        updated[distances[:, 0] < distances[:, 1]] = 0
        updated[distances[:, 1] < distances[:, 0]] = 1
        if len(set(updated.tolist())) != 2:
            raise UnsplittableSpeakerError(
                "these turns do not form two stable voice groups"
            )
        if np.array_equal(updated, labels):
            break
        labels = updated

    # Group A retains the original id. Canonicalize the otherwise arbitrary
    # cluster names by putting the earliest persisted turn in A.
    if labels[0] == 1:
        labels = 1 - labels
    return {
        index: ("A" if int(label) == 0 else "B")
        for index, label in zip(keys, labels, strict=True)
    }


__all__ = [
    "AttestedTurnEmbeddings",
    "EmbeddingIdentity",
    "TurnEmbeddingError",
    "UnsplittableSpeakerError",
    "bisect_embeddings",
    "normalized_centroid",
    "turn_embeddings",
]
