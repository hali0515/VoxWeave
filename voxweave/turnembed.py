"""Turn-level speaker embeddings and deterministic two-way clustering."""

from __future__ import annotations

import hashlib
import io
import importlib.metadata
import math
import operator
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, overload

import numpy as np

from voxweave import config, runtime
from voxweave.diarize import _snapshot_commit
from voxweave.diarize import _canonical_embedding_source as _canonical_source
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


class AttestedTurnRequest(Sequence[object]):
    """Turn sequence carrying the exact voiceprint authority it must reproduce."""

    __slots__ = ("__identity", "__turns")

    def __init__(
        self,
        turns: Sequence[object],
        *,
        identity: EmbeddingIdentity,
    ) -> None:
        self.__turns = tuple(turns)
        self.__identity = identity

    @property
    def identity(self) -> EmbeddingIdentity:
        """Frozen identity the provider must verify before inference."""
        return self.__identity

    def __len__(self) -> int:
        return len(self.__turns)

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[object, ...]: ...

    def __getitem__(self, index: int | slice) -> object | tuple[object, ...]:
        return self.__turns[index]


@dataclass(frozen=True, slots=True)
class _EmbeddingAuthority:
    checkpoint: str
    revision: str | None
    subfolder: str | None


_inference: Any | None = None
_inference_identity: EmbeddingIdentity | None = None
# SpeakerHTTPServer dispatches requests on threads. The model loader and the
# underlying torch module are both process singletons, so serialize construction
# and forward calls rather than racing the same GPU module from two /split posts.
_inference_lock = threading.RLock()


def _canonical_embedding_source(authority: _EmbeddingAuthority) -> str:
    """Dataclass-shaped view of the shared ``checkpoint[@rev][#subfolder=]`` grammar."""
    return _canonical_source(
        authority.checkpoint,
        revision=authority.revision,
        subfolder=authority.subfolder,
    )


def _parse_embedding_source(source: str) -> _EmbeddingAuthority:
    """Parse the canonical source format persisted in voiceprint provenance."""
    if not source or source == "unresolved" or source.count("#") > 1:
        raise TurnEmbeddingError("speaker embedding model authority is invalid")
    checkpoint_and_revision, separator, fragment = source.partition("#")
    subfolder: str | None = None
    if separator:
        prefix = "subfolder="
        if not fragment.startswith(prefix) or not fragment.removeprefix(prefix):
            raise TurnEmbeddingError("speaker embedding model authority is invalid")
        subfolder = fragment.removeprefix(prefix)
        subfolder_path = PurePosixPath(subfolder)
        if subfolder_path.is_absolute() or ".." in subfolder_path.parts:
            raise TurnEmbeddingError("speaker embedding model authority is invalid")

    configured = Path(checkpoint_and_revision).expanduser()
    if configured.exists():
        checkpoint = checkpoint_and_revision
        revision = None
    elif "@" in checkpoint_and_revision:
        checkpoint, revision = checkpoint_and_revision.rsplit("@", 1)
        if not checkpoint or not revision:
            raise TurnEmbeddingError("speaker embedding model authority is invalid")
    else:
        checkpoint = checkpoint_and_revision
        revision = None
    authority = _EmbeddingAuthority(checkpoint, revision, subfolder)
    if _canonical_embedding_source(authority) != source:
        raise TurnEmbeddingError("speaker embedding model authority is not canonical")
    return authority


def _download_checkpoint(
    authority: _EmbeddingAuthority,
    token: str | None,
) -> Path:
    """Resolve one exact authority inside VoxWeave's private audio cache."""
    configured = Path(authority.checkpoint).expanduser()
    if configured.is_file():
        if authority.revision is not None or authority.subfolder is not None:
            raise TurnEmbeddingError(
                "local speaker embedding files cannot use revision or subfolder"
            )
        return configured
    if configured.is_dir():
        if authority.revision is not None:
            raise TurnEmbeddingError(
                "local speaker embedding directories cannot use a revision"
            )
        checkpoint = configured
        if authority.subfolder is not None:
            checkpoint = checkpoint / authority.subfolder
        checkpoint = checkpoint / EMBEDDING_CHECKPOINT_FILE
        if not checkpoint.is_file():
            raise TurnEmbeddingError(
                "local speaker embedding checkpoint does not exist"
            )
        return checkpoint
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise TurnEmbeddingError(
            "speaker splitting requires the huggingface-hub dependency"
        ) from exc
    try:
        checkpoint = hf_hub_download(
            authority.checkpoint,
            EMBEDDING_CHECKPOINT_FILE,
            revision=authority.revision,
            subfolder=authority.subfolder,
            token=token,
            cache_dir=config.AUDIO_CACHE,
        )
    except Exception as exc:
        raise TurnEmbeddingError(
            "could not download speaker embedding model "
            f"{_canonical_embedding_source(authority)}: {exc}"
        ) from exc
    return Path(checkpoint)


def _embedding_source(
    authority: _EmbeddingAuthority,
    cached_path: Path,
) -> str:
    commit = _snapshot_commit(
        cached_path,
        authority.checkpoint,
        filename=EMBEDDING_CHECKPOINT_FILE,
        subfolder=authority.subfolder,
    )
    if commit is None:
        return _canonical_embedding_source(authority)
    return _canonical_embedding_source(
        _EmbeddingAuthority(
            authority.checkpoint,
            commit,
            authority.subfolder,
        )
    )


def _pyannote_version() -> str:
    try:
        return importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError:
        return "unresolved"


def _construct_bound_checkpoint(
    checkpoint: Path,
    construct: Callable[[io.BytesIO], Any],
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, str]:
    """Hash and construct from the same frozen checkpoint bytes."""
    try:
        resolved = Path(checkpoint).resolve(strict=True)
        payload = resolved.read_bytes()
    except (MemoryError, OSError) as exc:
        raise TurnEmbeddingError(
            "speaker embedding checkpoint could not be bound before construction"
        ) from exc
    before = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and before != expected_sha256:
        raise TurnEmbeddingError(
            "speaker embedding checkpoint does not match requested identity"
        )
    checkpoint_buffer = io.BytesIO(payload)
    value = construct(checkpoint_buffer)
    after = hashlib.sha256(checkpoint_buffer.getvalue()).hexdigest()
    if before != after:
        raise TurnEmbeddingError(
            "speaker embedding checkpoint changed during construction"
        )
    return value, before


def _load_inference(
    expected_identity: EmbeddingIdentity | None = None,
) -> tuple[Any, EmbeddingIdentity]:
    """Load the production embedding family lazily on the best torch device."""
    if expected_identity is None:
        authority = _EmbeddingAuthority(EMBEDDING_MODEL, None, None)
    else:
        if (
            len(expected_identity.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_identity.checkpoint_sha256
            )
            or not expected_identity.pyannote_version
            or expected_identity.pyannote_version == "unresolved"
        ):
            raise TurnEmbeddingError("speaker embedding identity is invalid")
        authority = _parse_embedding_source(expected_identity.model)

    pyannote_version = _pyannote_version()
    if (
        expected_identity is not None
        and pyannote_version != expected_identity.pyannote_version
    ):
        raise TurnEmbeddingError(
            "installed pyannote.audio version does not match requested identity"
        )
    token = config.conf_hf_token()
    if (
        not token
        and not Path(authority.checkpoint).expanduser().exists()
        and authority.checkpoint in {EMBEDDING_MODEL, config.COMMUNITY_DIARIZE_MODEL}
    ):
        raise TurnEmbeddingError(
            "speaker splitting needs the Hugging Face token used for diarization; "
            "set VOXWEAVE_HF_TOKEN / HF_TOKEN or run `hf auth login`"
        )
    checkpoint = _download_checkpoint(authority, token)
    embedding_source = _embedding_source(authority, checkpoint)
    if expected_identity is not None and embedding_source != expected_identity.model:
        raise TurnEmbeddingError(
            "resolved speaker embedding model does not match requested identity"
        )
    try:
        import torch
        from pyannote.audio import Model  # pyright: ignore[reportMissingImports]
        from pyannote.audio.pipelines.speaker_verification import (  # pyright: ignore[reportMissingImports]
            PretrainedSpeakerEmbedding,
        )
    except (ImportError, AttributeError) as exc:
        raise TurnEmbeddingError(
            "speaker splitting could not import the pyannote embedding runtime"
        ) from exc

    device = torch.device(runtime.get_device())

    def construct(bound_checkpoint: io.BytesIO) -> Any:
        return Model.from_pretrained(
            bound_checkpoint,
            map_location=device,
            strict=True,
            token=token,
            cache_dir=config.AUDIO_CACHE,
        )

    try:
        model, checkpoint_sha256 = _construct_bound_checkpoint(
            checkpoint,
            construct,
            expected_sha256=(
                expected_identity.checkpoint_sha256
                if expected_identity is not None
                else None
            ),
        )
    except TurnEmbeddingError:
        raise
    except Exception as exc:
        raise TurnEmbeddingError(
            f"could not load speaker embedding model {embedding_source}: {exc}"
        ) from exc
    if model is None:
        raise TurnEmbeddingError(
            f"could not load speaker embedding model {embedding_source}"
        )
    try:
        inference = PretrainedSpeakerEmbedding(
            model,
            device=device,
            token=token,
            cache_dir=config.AUDIO_CACHE,
        )
    except Exception as exc:
        raise TurnEmbeddingError(
            f"could not initialize speaker embedding model {embedding_source}: {exc}"
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
    identity = EmbeddingIdentity(
        model=embedding_source,
        checkpoint_sha256=checkpoint_sha256,
        pyannote_version=pyannote_version,
    )
    if expected_identity is not None and identity != expected_identity:
        raise TurnEmbeddingError(
            "loaded speaker embedding does not match requested identity"
        )
    return inference, identity


def _resident_matches(expected_identity: EmbeddingIdentity | None) -> bool:
    identity = _inference_identity
    if identity is None:
        return False
    if expected_identity is not None:
        return identity == expected_identity
    try:
        authority = _parse_embedding_source(identity.model)
    except TurnEmbeddingError:
        return False
    return authority.checkpoint == EMBEDDING_MODEL and authority.subfolder is None


def _get_inference(
    expected_identity: EmbeddingIdentity | None = None,
) -> Any:
    global _inference, _inference_identity
    with _inference_lock:
        if _inference is not None and _inference_identity is None:
            raise TurnEmbeddingError(
                "speaker embedding inference has no checkpoint identity"
            )
        if _inference is None or not _resident_matches(expected_identity):
            loaded, identity = _load_inference(expected_identity)
            _inference_identity = identity
            _inference = loaded
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
    expected_identity = (
        turns.identity if isinstance(turns, AttestedTurnRequest) else None
    )
    if not turns:
        with _inference_lock:
            _get_inference(expected_identity)
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
        inference = _get_inference(expected_identity)
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
    "AttestedTurnRequest",
    "EmbeddingIdentity",
    "TurnEmbeddingError",
    "UnsplittableSpeakerError",
    "bisect_embeddings",
    "normalized_centroid",
    "turn_embeddings",
]
