"""Speaker diarization (pyannote) + Netflix speaker-aware cue formatting.

Detection runs once per file on the separated-vocals 16k wav (clean speech beats
the BGM mix for speaker embeddings) and persists ``speaker_turns`` to the
sibling JSON. Formatting is a pure post-pass over smart_split's cues: each
cue's atoms get a speaker by time overlap with the turns; a cue containing two
speakers becomes a Netflix dual-speaker event (one line per speaker, leading
hyphen, no space) when the language allows two lines and both halves fit one
line, otherwise the cue splits at the speaker boundaries. ``split`` replays
formatting from the persisted turns without re-running pyannote.

The default pipeline remains ``pyannote/speaker-diarization-3.1`` for existing
gated-model access. ``community-1`` and arbitrary Hugging Face pipeline IDs are
selectable per invocation. Both bundled aliases run through pyannote.audio 4.x.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import logging
import math
import os
import threading
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, cast

import yaml

from voxweave import config
from voxweave.core.schema import Cue
from voxweave.voicebase import MAX_EMBEDDING_DIM, MIN_EMBEDDING_DIM

if TYPE_CHECKING:
    from voxweave.core.smart_split import SplitThresholds

log = logging.getLogger("voxweave")

DIARIZE_MODEL = config.DEFAULT_DIARIZE_MODEL
COMMUNITY_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_CHECKPOINT_FILE = "pytorch_model.bin"

# Atom-level speaker assignment needs at least this much absolute overlap with a
# turn (seconds); below it the atom inherits its neighbors (guards 20ms grazes).
MIN_ATOM_OVERLAP_S = 0.05

# A speaker run whose total atom duration is below this is a candidate for
# absorption into an adjacent run (a word is never cut into two speaker cues).
# Whether a sub-floor run is actually absorbed is position-aware (see
# _absorb_tiny_runs): mid-phrase thrash goes, real edge utterances stay.
MIN_RUN_S = 0.2

# A sub-MIN_RUN_S run that is *not* a same-speaker (A-B-A) sandwich -- i.e. one at
# a cue edge or between two different speakers (A-B-C) -- is a real short
# utterance and is kept when at least this long; only shorter fragments are
# pyannote noise. Keeps e.g. a 160ms trailing '你好' as its own speaker cue while
# still dropping an 80ms mid-phrase fragment.
EDGE_RUN_MIN_S = 0.12

# A speaker boundary is also a cue boundary, so the left piece's tail is scored
# with the Level-2 (UniDic POS) end penalty smart_split uses for its own breaks.
# 2 = the tail cannot end an utterance at all (格助詞 with no head, 連体詞 with
# nothing to modify); splitting there is only acceptable when the cue has no
# cleaner edge to offer. ja only -- no validated equivalent grading exists for zh,
# and the Level-1 char table over-fires (準体の, adverbial に).
BAD_TAIL_PENALTY = 2

# Inter-run silence a bad-tail merge may span. A reply that starts after a beat is
# a genuine second speaker, whatever the left tail looks like -- swallowing it is
# the 2026-07-03 EDGE_RUN_MIN_S regression class. Only a boundary the diarizer
# placed inside continuous speech can plausibly be a mis-cut phrase, so the merge
# is confined to gaps below the shortest inter-speaker pause worth trusting
# (same order as MIN_RUN_S: shorter than any real turn-taking beat).
BAD_TAIL_MAX_GAP_S = 0.2

# Turn-list smoothing (raw pyannote turns are noisy: 16-31% run <0.5s, and
# overlap-track fragments sit fully inside another speaker's turn). Module
# constants, overridable via env.
DIARIZE_MERGE_GAP_S = (
    0.35  # merge consecutive same-speaker turns across a gap below this
)
DIARIZE_DROP_CONTAINED_S = (
    0.2  # drop turns shorter than this fully inside another speaker's turn
)

Turn = tuple[float, float, str]


@dataclass(frozen=True)
class DiarizationResult:
    """Diarization turns plus optional, label-keyed speaker centroids."""

    turns: list[Turn]
    centroids: dict[str, list[float]] | None
    provenance: dict[str, object]


@dataclass(frozen=True)
class _EmbeddingCheckpointBinding:
    """Exact checkpoint path and content identity fixed before construction."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class _EmbeddingLoadAuthority:
    """Embedding source passed to the loader and its pre-load authority."""

    loader_value: object
    provenance_value: str
    binding: _EmbeddingCheckpointBinding
    loaded_path: Path


@dataclass(frozen=True)
class _PipelineLoadPlan:
    """Immutable pipeline config plus the embedding authority it carries."""

    checkpoint: str | Path | dict[str, Any]
    revision: str | None
    authority: _EmbeddingLoadAuthority | None
    outer_config_sha256: str


class EmbeddingCheckpointChangedError(RuntimeError):
    """The configured local embedding checkpoint changed during construction."""


_EMBEDDING_BINDING_ATTR = "_voxweave_embedding_checkpoint_binding"
_EMBEDDING_MODEL_ATTR = "_voxweave_embedding_model"
_OUTER_CONFIG_ATTR = "_voxweave_outer_config_sha256"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_model_revision(model: str) -> tuple[str, str | None]:
    """Split Hugging Face ``repo@revision`` without rewriting local paths."""
    if Path(model).exists() or "@" not in model:
        return model, None
    model_id, revision = model.rsplit("@", 1)
    if not model_id or not revision:
        return model, None
    return model_id, revision


def _snapshot_commit(
    cached_path: Path,
    model_id: str,
    *,
    filename: str = EMBEDDING_CHECKPOINT_FILE,
    subfolder: str | None = None,
) -> str | None:
    """Extract a commit from a validated HF snapshot path, including subfolders."""
    absolute = cached_path.absolute()
    expected_relative = Path(subfolder) / filename if subfolder else Path(filename)
    for snapshot in absolute.parents:
        if snapshot.parent.name != "snapshots":
            continue
        repo_cache = snapshot.parent.parent
        expected_repo_cache = f"models--{model_id.replace('/', '--')}"
        if repo_cache.name != expected_repo_cache:
            return None
        try:
            relative = absolute.relative_to(snapshot)
        except ValueError:
            return None
        if relative != expected_relative:
            return None
        commit = snapshot.name
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            return None
        return commit
    return None


def _canonical_embedding_source(
    checkpoint: str,
    *,
    revision: str | None,
    subfolder: str | None,
) -> str:
    source = checkpoint if revision is None else f"{checkpoint}@{revision}"
    if subfolder:
        source = f"{source}#subfolder={subfolder}"
    return source


def _pipeline_config_path(model: str, token: str | None = None) -> Path | None:
    """Download or resolve the exact pipeline config inside VoxWeave's cache."""
    configured = Path(model)
    if configured.is_dir():
        configured = configured / "config.yaml"
    try:
        return configured.resolve(strict=True)
    except OSError:
        pass
    model_id, revision = _split_model_revision(model)
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            model_id,
            "config.yaml",
            revision=revision,
            token=token,
            cache_dir=config.AUDIO_CACHE,
        )
    except (ImportError, OSError, ValueError):
        return None
    return Path(cached).absolute()


def _outer_config_identity(
    model: str = DIARIZE_MODEL,
    token: str | None = None,
) -> str:
    """Hash the exact local/cached pyannote pipeline config."""
    try:
        config_path = _pipeline_config_path(model, token)
        return "unresolved" if config_path is None else _sha256_file(config_path)
    except (OSError, ValueError):
        return "unresolved"


def _embedding_load_authority(
    raw_path: object,
    *,
    parent_model: str | None = None,
    parent_revision: str | None = None,
    token: str | None = None,
) -> _EmbeddingLoadAuthority | None:
    """Bind one local or Hub embedding checkpoint before construction."""
    original_mapping = dict(raw_path) if isinstance(raw_path, Mapping) else None
    checkpoint: object = raw_path
    revision: object = None
    subfolder: object = None
    if isinstance(raw_path, Mapping):
        checkpoint = raw_path.get("checkpoint")
        revision = raw_path.get("revision")
        subfolder = raw_path.get("subfolder")
    elif isinstance(raw_path, str) and raw_path.startswith("$model/"):
        checkpoint = parent_model
        revision = parent_revision
        subfolder = raw_path.removeprefix("$model/")

    if not isinstance(checkpoint, (str, os.PathLike)):
        return None
    if revision is not None and not isinstance(revision, str):
        return None
    if subfolder is not None and not isinstance(subfolder, str):
        return None
    raw_checkpoint = os.fspath(checkpoint)
    configured = Path(raw_checkpoint)
    local_checkpoint: Path | None = None

    def mapping_loader_value(
        resolved_checkpoint: str,
        *,
        resolved_revision: str | None,
        resolved_subfolder: str | None,
    ) -> dict[str, object]:
        value = dict(original_mapping or {})
        value["checkpoint"] = resolved_checkpoint
        if resolved_revision is not None:
            value["revision"] = resolved_revision
        elif original_mapping is None:
            value.pop("revision", None)
        if resolved_subfolder is not None:
            value["subfolder"] = resolved_subfolder
        elif original_mapping is None:
            value.pop("subfolder", None)
        if original_mapping is None:
            # pyannote's string branch calls Model.from_pretrained(strict=False).
            value["strict"] = False
        value["token"] = token
        value["cache_dir"] = config.AUDIO_CACHE
        return value

    if configured.is_file() and not subfolder:
        local_checkpoint = configured
        resolved_configured = os.fspath(configured.resolve(strict=True))
        loader_value: object = (
            resolved_configured
            if original_mapping is None
            else mapping_loader_value(
                resolved_configured,
                resolved_revision=cast(str | None, revision),
                resolved_subfolder=None,
            )
        )
    elif configured.is_dir():
        local_checkpoint = configured
        if subfolder:
            local_checkpoint = local_checkpoint / subfolder
        local_checkpoint = local_checkpoint / EMBEDDING_CHECKPOINT_FILE
        loader_value = mapping_loader_value(
            os.fspath(configured.resolve(strict=True)),
            resolved_revision=cast(str | None, revision),
            resolved_subfolder=cast(str | None, subfolder),
        )
    else:
        model_id, inline_revision = _split_model_revision(raw_checkpoint)
        selected_revision = cast(str | None, revision) or inline_revision
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            cached = hf_hub_download(
                model_id,
                EMBEDDING_CHECKPOINT_FILE,
                subfolder=cast(str | None, subfolder),
                revision=selected_revision,
                token=token,
                cache_dir=config.AUDIO_CACHE,
            )
        except EntryNotFoundError:
            return None
        except (ImportError, OSError, ValueError):
            return None
        cached_path = Path(cached).absolute()
        commit = _snapshot_commit(
            cached_path,
            model_id,
            subfolder=cast(str | None, subfolder),
        )
        pinned_revision = commit or selected_revision
        local_checkpoint = cached_path
        loader_value = mapping_loader_value(
            model_id,
            resolved_revision=pinned_revision,
            resolved_subfolder=cast(str | None, subfolder),
        )
        raw_checkpoint = model_id
        revision = pinned_revision

    try:
        resolved = local_checkpoint.resolve(strict=True)
        binding = _EmbeddingCheckpointBinding(
            path=resolved,
            sha256=_sha256_file(resolved),
        )
    except OSError:
        return None
    return _EmbeddingLoadAuthority(
        loader_value=loader_value,
        provenance_value=_canonical_embedding_source(
            raw_checkpoint,
            revision=cast(str | None, revision),
            subfolder=cast(str | None, subfolder),
        ),
        binding=binding,
        loaded_path=resolved,
    )


def _expand_model_references(
    value: object,
    *,
    model_id: str,
    revision: str | None,
    token: str | None,
    parent_subfolder: str | None = None,
) -> None:
    """Expand pyannote 4.x ``$model/subfolder`` references with stable coordinates."""

    def expand(reference: str) -> dict[str, object]:
        subfolder = reference.removeprefix("$model/")
        if "@" in subfolder:
            subfolder, child_revision = subfolder.rsplit("@", 1)
        else:
            child_revision = revision
        if parent_subfolder:
            subfolder = f"{parent_subfolder.rstrip('/')}/{subfolder.lstrip('/')}"
        return {
            "checkpoint": model_id,
            **({"revision": child_revision} if child_revision is not None else {}),
            "subfolder": subfolder,
            "token": token,
            "cache_dir": config.AUDIO_CACHE,
        }

    if isinstance(value, dict):
        for key, child in tuple(value.items()):
            if isinstance(child, str) and child.startswith("$model/"):
                value[key] = expand(child)
            else:
                _expand_model_references(
                    child,
                    model_id=model_id,
                    revision=revision,
                    token=token,
                    parent_subfolder=parent_subfolder,
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str) and child.startswith("$model/"):
                value[index] = expand(child)
            else:
                _expand_model_references(
                    child,
                    model_id=model_id,
                    revision=revision,
                    token=token,
                    parent_subfolder=parent_subfolder,
                )


def _prepare_pipeline_load(
    model: str = DIARIZE_MODEL,
    token: str | None = None,
) -> _PipelineLoadPlan:
    """Pin outer config and embedding coordinates before pipeline construction."""
    model_id, requested_revision = _split_model_revision(model)
    config_path = _pipeline_config_path(model, token)
    if config_path is None:
        return _PipelineLoadPlan(
            checkpoint=model_id,
            revision=requested_revision,
            authority=None,
            outer_config_sha256="unresolved",
        )
    try:
        config_bytes = config_path.read_bytes()
        outer_sha256 = hashlib.sha256(config_bytes).hexdigest()
        document: Any = yaml.safe_load(config_bytes.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return _PipelineLoadPlan(
            checkpoint=model_id,
            revision=requested_revision,
            authority=None,
            outer_config_sha256="unresolved",
        )
    if not isinstance(document, dict):
        return _PipelineLoadPlan(
            checkpoint=model_id,
            revision=requested_revision,
            authority=None,
            outer_config_sha256=outer_sha256,
        )

    try:
        configured_model = Path(model).resolve(strict=True)
    except OSError:
        configured_model = None
    if configured_model is None:
        snapshot_revision = _snapshot_commit(
            config_path,
            model_id,
            filename="config.yaml",
        )
        pinned_revision = snapshot_revision or requested_revision
        reference_model = model_id
    else:
        pinned_revision = None
        reference_model = os.fspath(
            configured_model if configured_model.is_dir() else configured_model.parent
        )
    _expand_model_references(
        document,
        model_id=reference_model,
        revision=pinned_revision,
        token=token,
    )
    authority = None
    pipeline_config = document.get("pipeline")
    if isinstance(pipeline_config, dict):
        params = pipeline_config.get("params")
        if isinstance(params, dict):
            authority = _embedding_load_authority(
                params.get("embedding"),
                parent_model=reference_model,
                parent_revision=pinned_revision,
                token=token,
            )
            if authority is not None:
                params["embedding"] = authority.loader_value
    return _PipelineLoadPlan(
        checkpoint=document,
        revision=None,
        authority=authority,
        outer_config_sha256=outer_sha256,
    )


def _store_embedding_checkpoint(
    pipeline: object,
    binding: _EmbeddingCheckpointBinding | None,
    model: str | None = None,
    outer_config_sha256: str | None = None,
) -> None:
    """Carry a pre-construction binding onto the resident pipeline once."""
    try:
        if model is not None:
            setattr(pipeline, _EMBEDDING_MODEL_ATTR, model)
        setattr(pipeline, _EMBEDDING_BINDING_ATTR, binding)
        if outer_config_sha256 is not None:
            setattr(pipeline, _OUTER_CONFIG_ATTR, outer_config_sha256)
    except (AttributeError, TypeError):
        log.warning("could not bind embedding checkpoint provenance to pipeline")


def _checkpoint_identity(pipeline: object) -> str:
    """Return only the checkpoint identity stored during construction."""
    binding = getattr(pipeline, _EMBEDDING_BINDING_ATTR, None)
    if isinstance(binding, _EmbeddingCheckpointBinding):
        return binding.sha256
    return "unresolved"


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unresolved"


def _build_provenance(
    pipeline: object,
    *,
    model: str = DIARIZE_MODEL,
    embedding_dim: int | str,
    audio_profile: Mapping[str, object] | None,
    torch_version: str,
) -> dict[str, object]:
    embedding_model = getattr(pipeline, _EMBEDDING_MODEL_ATTR, None)
    if not isinstance(embedding_model, str) or not embedding_model:
        raw_embedding = getattr(pipeline, "embedding", None)
        if isinstance(raw_embedding, str) and raw_embedding:
            embedding_model = raw_embedding
        elif isinstance(raw_embedding, Mapping):
            checkpoint = raw_embedding.get("checkpoint")
            revision = raw_embedding.get("revision")
            subfolder = raw_embedding.get("subfolder")
            if (
                isinstance(checkpoint, str)
                and checkpoint
                and (revision is None or isinstance(revision, str))
                and (subfolder is None or isinstance(subfolder, str))
            ):
                embedding_model = _canonical_embedding_source(
                    checkpoint,
                    revision=cast(str | None, revision),
                    subfolder=cast(str | None, subfolder),
                )
            else:
                embedding_model = "unresolved"
        else:
            embedding_model = "unresolved"
    audio: dict[str, object] = (
        dict(audio_profile)
        if audio_profile is not None
        else {"separated": False, "normalized": False, "sample_rate": 16000}
    )
    outer_config_sha256 = getattr(pipeline, _OUTER_CONFIG_ATTR, None)
    if not isinstance(outer_config_sha256, str) or not outer_config_sha256:
        outer_config_sha256 = _outer_config_identity(model)
    return {
        "diarization_model": model,
        "outer_config_sha256": outer_config_sha256,
        "embedding_model": embedding_model,
        "embedding_checkpoint": _checkpoint_identity(pipeline),
        "embedding_dim": embedding_dim,
        "audio": audio,
        "pyannote_version": _package_version("pyannote.audio"),
        "torch_version": torch_version,
    }


def _normalized_centroids(
    annotation: object,
    embeddings: object,
) -> tuple[dict[str, list[float]] | None, int | str]:
    """Map row i through labels()[i], dropping unusable padded rows."""
    labels_method: Any = getattr(annotation, "labels", None)
    if not callable(labels_method) or embeddings is None:
        log.warning("diarization pipeline did not return label-keyed embeddings")
        return None, "unresolved"
    try:
        labels = [str(label) for label in cast(Any, labels_method)()]
        rows = list(cast(Any, embeddings))
    except (TypeError, ValueError):
        log.warning("diarization pipeline returned unusable embeddings")
        return None, "unresolved"
    if len(labels) != len(set(labels)):
        log.warning("diarization pipeline returned duplicate speaker labels")
        return None, "unresolved"
    dimension: int | None = None
    centroids: dict[str, list[float]] = {}
    for index, label in enumerate(labels):
        if index >= len(rows):
            log.debug("dropping embedding-missing speaker %s", label)
            continue
        raw_row = rows[index]
        try:
            converted = raw_row.tolist() if hasattr(raw_row, "tolist") else raw_row
            values = [float(value) for value in converted]
        except (TypeError, ValueError, OverflowError):
            log.debug("dropping malformed centroid for speaker %s", label)
            continue
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            log.warning("diarization pipeline returned ragged speaker embeddings")
            return None, "unresolved"
        norm = math.sqrt(math.fsum(value * value for value in values))
        if (
            not math.isfinite(norm)
            or norm <= 0.0
            or any(not math.isfinite(value) for value in values)
        ):
            log.debug("dropping zero/non-finite centroid for speaker %s", label)
            continue
        centroids[label] = [value / norm for value in values]
    if dimension is None:
        return None, "unresolved"
    if not MIN_EMBEDDING_DIM <= dimension <= MAX_EMBEDDING_DIM:
        log.warning("diarization pipeline returned unsupported embedding dimension")
        return None, "unresolved"
    return centroids, dimension


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v is not None and v.strip() else default
    except ValueError:
        return default


_pipeline = None  # pyannote Pipeline singleton -- lazy-loaded, released after use
_pipeline_model: str | None = None
_pipeline_lock = threading.RLock()


def _call_pipeline_from_pretrained(
    pipeline_cls,
    token: str | None,
    checkpoint_path: str | Path | dict[str, Any],
    *,
    revision: str | None = None,
):
    """Call the pyannote 4.x loader with its explicit revision and private cache."""
    return pipeline_cls.from_pretrained(
        checkpoint_path,
        revision=revision,
        token=token,
        cache_dir=config.AUDIO_CACHE,
    )


def _is_legacy_agglomerative_plan(model: str, plan: _PipelineLoadPlan) -> bool:
    model_id, _revision = _split_model_revision(model)
    if model_id != config.DEFAULT_DIARIZE_MODEL or not isinstance(
        plan.checkpoint, dict
    ):
        return False
    pipeline_config = plan.checkpoint.get("pipeline")
    if not isinstance(pipeline_config, Mapping):
        return False
    if pipeline_config.get("name") != "pyannote.audio.pipelines.SpeakerDiarization":
        return False
    params = pipeline_config.get("params")
    return (
        isinstance(params, Mapping)
        and params.get("clustering") == "AgglomerativeClustering"
        and "plda" not in params
    )


@contextmanager
def _without_unused_legacy_plda(
    model: str,
    plan: _PipelineLoadPlan,
) -> Iterator[None]:
    """Prevent pyannote 4 from fetching c1's unused PLDA for the 3.1 pipeline."""
    model_id, _revision = _split_model_revision(model)
    if model_id != config.DEFAULT_DIARIZE_MODEL:
        yield
        return
    if not _is_legacy_agglomerative_plan(model, plan):
        raise RuntimeError(
            "speaker-diarization-3.1 under pyannote.audio 4 requires a verified "
            "AgglomerativeClustering pipeline config"
        )
    speaker_diarization = cast(
        Any,
        importlib.import_module("pyannote.audio.pipelines.speaker_diarization"),
    )

    original = speaker_diarization.get_plda
    speaker_diarization.get_plda = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        speaker_diarization.get_plda = original


def _load_pipeline(
    pipeline_cls,
    token: str | None,
    model: str = DIARIZE_MODEL,
):
    """Construct a pipeline with loader-authoritative embedding provenance."""
    with _pipeline_lock:
        return _load_pipeline_locked(pipeline_cls, token, model)


def _load_pipeline_locked(
    pipeline_cls,
    token: str | None,
    model: str,
):
    plan = _prepare_pipeline_load(model, token)
    with _without_unused_legacy_plda(model, plan):
        pl = _call_pipeline_from_pretrained(
            pipeline_cls,
            token,
            plan.checkpoint,
            revision=plan.revision,
        )
    if pl is None:
        return None

    authority = plan.authority
    if authority is not None:
        try:
            after = _sha256_file(authority.loaded_path)
        except OSError as exc:
            raise EmbeddingCheckpointChangedError(
                "embedding checkpoint changed during pipeline construction"
            ) from exc
        if after != authority.binding.sha256:
            raise EmbeddingCheckpointChangedError(
                "embedding checkpoint changed during pipeline construction"
            )
    _store_embedding_checkpoint(
        pl,
        authority.binding if authority is not None else None,
        authority.provenance_value if authority is not None else None,
        plan.outer_config_sha256,
    )
    return pl


def _model_card_url(model: str) -> str:
    model_id, _revision = _split_model_revision(model)
    return f"https://hf.co/{model_id}"


def _gated_model_error(model: str) -> RuntimeError:
    return RuntimeError(
        f"could not load {model}: accept the model-card conditions at "
        f"{_model_card_url(model)}, then use `hf auth login` or set "
        "VOXWEAVE_HF_TOKEN / HF_TOKEN (or hf_token in "
        "~/.config/voxweave.conf)"
    )


def _get_pipeline(token: str | None, model: str = DIARIZE_MODEL):
    with _pipeline_lock:
        return _get_pipeline_locked(token, model)


def _get_pipeline_locked(token: str | None, model: str):
    global _pipeline, _pipeline_model
    if _pipeline is not None and _pipeline_model != model:
        release()
    if _pipeline is None:
        try:
            import torch

            from pyannote.audio import (  # pyright: ignore[reportMissingImports]
                Pipeline,
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "diarization requires the default pyannote.audio dependency; "
                "reinstall voxweave to repair the environment"
            ) from e
        try:
            pl = _load_pipeline(Pipeline, token, model)
        except Exception as exc:
            try:
                from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                gated = isinstance(exc, GatedRepoError) or (
                    isinstance(exc, HfHubHTTPError) and status_code == 403
                )
            except ImportError:
                gated = False
            if gated:
                raise _gated_model_error(model) from exc
            raise
        if pl is None:
            raise _gated_model_error(model)
        if torch.cuda.is_available():
            pl.to(torch.device("cuda"))
        _pipeline = pl
        _pipeline_model = model
        log.info("loaded diarization pipeline %s", model)
    return _pipeline


def release() -> None:
    """Drop the pipeline singleton and free its VRAM (mirrors backend.release)."""
    with _pipeline_lock:
        _release_locked()


def _release_locked() -> None:
    global _pipeline, _pipeline_model
    had_pipeline = _pipeline is not None
    _pipeline = None
    _pipeline_model = None
    if not had_pipeline:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def diarize_turns(
    wav_path: Path,
    *,
    token: str | None = None,
    model: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    want_embeddings: bool = False,
    audio_profile: Mapping[str, object] | None = None,
) -> DiarizationResult:
    """Run diarization and optionally return normalized, label-keyed centroids."""
    with _pipeline_lock:
        return _diarize_turns_locked(
            wav_path,
            token=token,
            model=model,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            want_embeddings=want_embeddings,
            audio_profile=audio_profile,
        )


def _diarize_turns_locked(
    wav_path: Path,
    *,
    token: str | None,
    model: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    want_embeddings: bool,
    audio_profile: Mapping[str, object] | None,
) -> DiarizationResult:
    resolved_model = config.resolve_diarize_model(model)
    token = token or config.conf_hf_token()
    if not token and resolved_model in {
        config.DEFAULT_DIARIZE_MODEL,
        COMMUNITY_DIARIZE_MODEL,
    }:
        raise _gated_model_error(resolved_model)
    import soundfile as sf
    import torch

    # pyannote's reproducibility guard force-disables TF32 process-wide and warns
    # whenever it finds it enabled (pyannote-audio#1370) -- which it always does,
    # because torch enables cudnn TF32 by default and the separator deliberately
    # opts into TF32 matmuls. Pre-comply for the diarization span so the guard
    # stays silent, then restore the process policy so pyannote cannot turn the
    # separator's TF32 off behind our back.
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        pl = _get_pipeline(token, resolved_model)
        kwargs: dict[str, int] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        # Feed a decoded waveform dict rather than a path: pyannote's file-path branch
        # goes through torchaudio.info/load, which torchaudio 2.11 broke. This dict
        # form is a first-class pyannote input and sidesteps its runtime audio I/O.
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (T, C)
        wav = torch.from_numpy(data).T.contiguous()  # (C, T)
        if wav.shape[0] > 1:  # defensive stereo downmix -> mono (1, T)
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(torch.float32)
        with warnings.catch_warnings():
            # pyannote's stat pooling hits std() on single-frame reductions for
            # very short speaker segments; the result is guarded internally and
            # the warning is pure noise on every episode.
            warnings.filterwarnings(
                "ignore", message=r"std\(\): degrees of freedom is <= 0"
            )
            warnings.filterwarnings(
                "ignore", message=r"TensorFloat-32 \(TF32\) has been disabled"
            )
            audio_input = {"waveform": wav, "sample_rate": int(sr)}
            version = _package_version("pyannote.audio")
            try:
                pyannote_major = int(version.split(".", 1)[0])
            except ValueError:
                pyannote_major = 4
            if pyannote_major >= 4:
                raw_result = pl(audio_input, **kwargs)
            else:
                raw_result = pl(
                    audio_input,
                    return_embeddings=want_embeddings,
                    **kwargs,
                )
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    output_annotation = getattr(raw_result, "speaker_diarization", None)
    if output_annotation is not None:
        annotation = output_annotation
        embeddings = (
            getattr(raw_result, "speaker_embeddings", None) if want_embeddings else None
        )
        if want_embeddings:
            centroids, embedding_dim = _normalized_centroids(annotation, embeddings)
        else:
            centroids = None
            embedding_dim = "unresolved"
    elif want_embeddings and isinstance(raw_result, tuple) and len(raw_result) == 2:
        annotation, embeddings = raw_result
        centroids, embedding_dim = _normalized_centroids(annotation, embeddings)
    else:
        annotation = raw_result
        centroids = None
        embedding_dim = "unresolved"
        if want_embeddings:
            log.warning("diarization pipeline cannot provide speaker embeddings")
    annotation_view = cast(Any, annotation)
    turns = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation_view.itertracks(yield_label=True)
    ]
    turns = _smooth_turns(turns)
    if centroids is not None:
        persisted_labels = {label for _start, _end, label in turns}
        centroids = {
            label: vector
            for label, vector in centroids.items()
            if label in persisted_labels
        }
        if not centroids:
            centroids = None
    log.info(
        "diarization: %d turn(s), %d speaker(s)",
        len(turns),
        len({lb for _, _, lb in turns}),
    )
    provenance = _build_provenance(
        pl,
        model=resolved_model,
        embedding_dim=embedding_dim,
        audio_profile=audio_profile,
        torch_version=str(torch.__version__),
    )
    return DiarizationResult(
        turns=turns,
        centroids=centroids,
        provenance=provenance,
    )


def _smooth_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Smooth raw pyannote turns before persisting (pure, order-preserving).

    Two passes, both robust to noisy input:
    - drop turns shorter than ``VOXWEAVE_DIARIZE_DROP_CONTAINED_S`` that are fully
      contained inside a *different* speaker's turn (overlap-track fragments);
      standalone short interjections (not contained) are spared.
    - merge consecutive same-speaker turns separated by a gap below
      ``VOXWEAVE_DIARIZE_MERGE_GAP_S`` (a single speaker split by a micro-pause).

    Containment is tested against the original turn set, so dropping never depends
    on merge order. Clean input is returned unchanged.
    """
    if not turns:
        return []
    ordered = sorted(turns, key=lambda t: (t[0], t[1], t[2]))
    drop_s = _env_float("VOXWEAVE_DIARIZE_DROP_CONTAINED_S", DIARIZE_DROP_CONTAINED_S)
    merge_gap = _env_float("VOXWEAVE_DIARIZE_MERGE_GAP_S", DIARIZE_MERGE_GAP_S)
    kept: list[Turn] = []
    for i, (a, b, lb) in enumerate(ordered):
        if b - a < drop_s and any(
            olb != lb and oa <= a and b <= ob
            for j, (oa, ob, olb) in enumerate(ordered)
            if j != i
        ):
            continue
        kept.append((a, b, lb))
    merged: list[Turn] = []
    for a, b, lb in kept:
        if merged and merged[-1][2] == lb and a - merged[-1][1] < merge_gap:
            pa, pb, plb = merged[-1]
            merged[-1] = (pa, max(pb, b), plb)
        else:
            merged.append((a, b, lb))
    return merged


def _span_speaker(
    start: float | None, end: float | None, turns: Sequence[Turn]
) -> str | None:
    """Dominant speaker for a time span by accumulated overlap (whisperX pattern)."""
    if start is None or end is None or end <= start:
        return None
    overlap: dict[str, float] = {}
    for a, b, label in turns:
        if b <= start:
            continue
        if a >= end:
            break  # turns are sorted by start
        ov = min(end, b) - max(start, a)
        if ov > 0:
            overlap[label] = overlap.get(label, 0.0) + ov
    if not overlap:
        return None
    label, best = max(overlap.items(), key=lambda kv: kv[1])
    return label if best >= MIN_ATOM_OVERLAP_S else None


def _run_dur(atoms: list[dict]) -> float:
    """Total spoken duration of a run: sum of its atoms' positive durations."""
    total = 0.0
    for a in atoms:
        s, e = a.get("start"), a.get("end")
        if s is not None and e is not None and e > s:
            total += e - s
    return total


def _coalesce_runs(
    runs: list[tuple[str, list[dict]]],
) -> list[tuple[str, list[dict]]]:
    """Merge adjacent same-speaker runs (their atoms concatenate)."""
    out: list[tuple[str, list[dict]]] = []
    for lb, ats in runs:
        if out and out[-1][0] == lb:
            out[-1][1].extend(ats)
        else:
            out.append((lb, list(ats)))
    return out


def _absorbable(i: int, durs: list[float], runs: list[tuple[str, list[dict]]]) -> bool:
    """Whether run ``i`` (already known sub-``MIN_RUN_S``) should be absorbed.

    Position-aware policy:
    - A run sandwiched between two runs of the *same* speaker (A-B-A) is label
      thrash and is always absorbed, regardless of ``EDGE_RUN_MIN_S``.
    - A run at a cue edge, or between two *different* speakers (A-B-C), is a real
      short utterance: kept when >= ``EDGE_RUN_MIN_S``, absorbed only below it.
    """
    has_left = i > 0
    has_right = i + 1 < len(runs)
    sandwiched_same = has_left and has_right and runs[i - 1][0] == runs[i + 1][0]
    return sandwiched_same or durs[i] < EDGE_RUN_MIN_S


def _absorb_tiny_runs(
    runs: list[tuple[str, list[dict]]],
) -> list[tuple[str, list[dict]]]:
    """Fold absorbable sub-``MIN_RUN_S`` runs into their longer neighbor.

    Collapses A-B-A label thrash to a single run while keeping real short edge or
    A-B-C utterances (see ``_absorbable``). Repeats to a fixpoint: absorbing one
    run and re-coalescing can turn a surviving run into a new same-speaker
    sandwich, which the next pass then absorbs."""
    runs = [(lb, list(ats)) for lb, ats in runs]
    while len(runs) > 1:
        durs = [_run_dur(ats) for _, ats in runs]
        tiny = [
            (durs[i], i)
            for i in range(len(runs))
            if durs[i] < MIN_RUN_S and _absorbable(i, durs, runs)
        ]
        if not tiny:
            break
        _, i = min(tiny)  # shortest absorbable run first
        left = durs[i - 1] if i > 0 else -1.0
        right = durs[i + 1] if i + 1 < len(runs) else -1.0
        if left < 0 and right < 0:
            break
        if right > left:  # merge into the following (longer) run: prepend atoms
            runs[i + 1] = (runs[i + 1][0], runs[i][1] + runs[i + 1][1])
        else:  # merge into the preceding run: append atoms
            runs[i - 1] = (runs[i - 1][0], runs[i - 1][1] + runs[i][1])
        del runs[i]
        runs = _coalesce_runs(runs)
    return runs


def _snap_runs_to_phrases(
    runs: list[tuple[str, list[dict]]], lang: str
) -> list[tuple[str, list[dict]]]:
    """Snap run boundaries onto legal token edges so a run never cuts mid-word.

    No-space langs only (space-delimited atoms are already whole words). Each
    jieba/BudouX phrase is reassigned to its dominant speaker (by atom duration),
    which is exactly a boundary snap: a phrase spanning a mid-word label flip
    goes wholly to one speaker instead of splitting. Reuses the phrase-boundary
    machinery smart_split uses (``_phrase_boundary_atoms`` -> ``phrase_atoms``).
    """
    from voxweave.core.layout import _no_spaces

    if not _no_spaces(lang) or len(runs) < 2:
        return runs
    from voxweave.core.smart_split import _phrase_boundary_atoms

    flat = [a for _, ats in runs for a in ats]
    labels = [lb for lb, ats in runs for _ in ats]
    text = "".join(a["text"] for a in flat)
    boundaries = _phrase_boundary_atoms([{"text": a["text"]} for a in flat], text, lang)
    edges = sorted(boundaries | {0, len(flat)})
    new_labels = list(labels)
    for s, e in zip(edges, edges[1:]):
        weight: dict[str, float] = {}
        for k in range(s, e):
            weight[labels[k]] = weight.get(labels[k], 0.0) + _run_dur([flat[k]])
        if not weight:
            continue
        first = labels[s]
        best = max(weight, key=lambda lb: (weight[lb], lb == first))
        for k in range(s, e):
            new_labels[k] = best
    out: list[tuple[str, list[dict]]] = []
    for a, lb in zip(flat, new_labels):
        if out and out[-1][0] == lb:
            out[-1][1].append(a)
        else:
            out.append((lb, [a]))
    return out


def _run_gap(left: list[dict], right: list[dict]) -> float:
    """Silence between the end of ``left`` and the start of ``right``.

    ``inf`` when either side has no usable timestamp: an unmeasurable gap must
    never be read as "contiguous" by a caller that merges on contiguity.
    """
    end = left[-1].get("end") if left else None
    start = right[0].get("start") if right else None
    if end is None or start is None:
        return math.inf
    # +epsilon keeps BAD_TAIL_MAX_GAP_S the exclusive bound its comment documents:
    # 1.4 - 1.2 is 0.19999999999999996 in binary floating point, so without it a
    # nominally 0.2s gap slips under the ceiling and merges.
    return max(0.0, float(start) - float(end)) + 1e-9


def _ja_edge_penalties(flat: list[dict], edges: Sequence[int]) -> dict[int, int] | None:
    """Level-2 (UniDic POS) end penalty for each internal atom edge, or ``None``.

    Wired exactly like ``smart_split._attach_end_penalties``: the POS map is keyed
    by the non-space char offset of each token's LAST char, so the edge before
    atom ``e`` is scored at the cumulative offset of atom ``e - 1``.

    ``None`` means no Level-2 source (fugashi absent, or ``VOXWEAVE_JA_POS=0``).
    Unlike line breaking, this pass has no safe Level-1 fallback: the char table
    scores every trailing の/に 2, including 準体の (そうな|の) and the adverbial
    copula (そんな|に), and acting on those merges away a real speaker. So the
    caller must leave the boundary alone rather than degrade. Offsets the tagger
    does not end a token on (BudouX/MeCab disagreement) score 0 for the same
    reason.
    """
    from voxweave.core.kinsoku import ja_pos_end_penalties
    from voxweave.core.layout import _token_char_count

    pos = ja_pos_end_penalties("".join(a["text"] for a in flat), bound_tails_only=True)
    if pos is None:
        return None
    offsets: list[int] = []
    total = 0
    for a in flat:
        total += _token_char_count(a["text"])
        offsets.append(total)
    return {e: pos.get(offsets[e - 1] - 1, 0) for e in edges if 0 < e < len(flat)}


def _merge_bad_tail_runs(
    runs: list[tuple[str, list[dict]]], lang: str
) -> list[tuple[str, list[dict]]]:
    """Merge a speaker boundary that would strand a bound word on the left cue.

    ja only: the gate needs the Level-2 POS signal to tell a dangling 格助詞 from
    a legal clause end, and no validated equivalent grading exists for zh.

    Fires only when all three hold: the boundary's tail scores
    ``BAD_TAIL_PENALTY``, no other internal phrase edge of the cue would score
    below it, and the two runs are separated by less than ``BAD_TAIL_MAX_GAP_S``
    of silence. A cleaner edge means the boundary stays where the speaker signal
    put it (relocating it is a separate concern); an audible gap means the second
    run is a real reply, not a mis-cut phrase. The merged run keeps the longer
    speaker's label, the same duration comparison ``_absorb_tiny_runs`` uses.
    Duration alone never triggers this pass, so the tiny-run policy above is
    untouched.

    Repeats to a fixpoint: merging can join two runs of one speaker, and the
    re-coalesced neighbours form a boundary that was not there before.
    """
    if lang != "ja" or len(runs) < 2:
        return runs
    from voxweave.core.smart_split import _phrase_boundary_atoms

    runs = [(lb, list(ats)) for lb, ats in runs]
    while len(runs) > 1:
        flat = [a for _, ats in runs for a in ats]
        text = "".join(a["text"] for a in flat)
        edges = sorted(
            _phrase_boundary_atoms([{"text": a["text"]} for a in flat], text, lang)
            | {0, len(flat)}
        )
        pen = _ja_edge_penalties(flat, edges)
        if pen is None:
            break
        target = None
        cut = 0
        for i in range(len(runs) - 1):
            cut += len(runs[i][1])
            # An unknown cut is not a phrase edge at all -- treat it as clean and
            # leave it to _snap_runs_to_phrases rather than merging on a guess.
            if pen.get(cut, 0) < BAD_TAIL_PENALTY:
                continue
            if any(p < BAD_TAIL_PENALTY for e, p in pen.items() if e != cut):
                continue
            if _run_gap(runs[i][1], runs[i + 1][1]) >= BAD_TAIL_MAX_GAP_S:
                continue
            target = i
            break
        if target is None:
            break
        left, right = runs[target], runs[target + 1]
        label = right[0] if _run_dur(right[1]) > _run_dur(left[1]) else left[0]
        runs[target] = (label, left[1] + right[1])
        del runs[target + 1]
        runs = _coalesce_runs(runs)
    return runs


def _speaker_runs(
    atoms: list[dict], turns: Sequence[Turn], lang: str
) -> list[tuple[str, list[dict]]]:
    """Group a cue's atoms into consecutive same-speaker runs.

    Atoms without a confident speaker (no span / no overlap) inherit the current
    run; leading unassigned atoms join the first labeled run. Raw runs are then
    de-noised: sub-``MIN_RUN_S`` label thrash is absorbed into the longer
    neighbor, and surviving boundaries snap to jieba/BudouX phrase edges so a
    lexeme is never split across two speaker cues. Finally (ja only) a boundary
    that survives the snap but would leave the left cue ending on a bound word --
    with no cleaner edge available and no audible gap -- is merged away.
    """
    runs: list[tuple[str, list[dict]]] = []
    pending: list[dict] = []  # unassigned atoms before the first labeled one
    for atom in atoms:
        spk = _span_speaker(atom.get("start"), atom.get("end"), turns)
        if spk is None:
            (runs[-1][1] if runs else pending).append(atom)
            continue
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(atom)
        else:
            runs.append((spk, [atom]))
            if pending:
                runs[-1][1][:0] = pending
                pending = []
    if pending:  # no atom got a speaker at all
        return []
    if len(runs) <= 1:
        return runs
    runs = _absorb_tiny_runs(runs)
    runs = _snap_runs_to_phrases(runs, lang)
    return _merge_bad_tail_runs(runs, lang)


def _slice_text_by_runs(text: str, runs: list[tuple[str, list[dict]]]) -> list[str]:
    """Slice the cue's display text into one piece per run.

    Atoms cover exactly the text's non-space characters in order, so each run
    consumes its atoms' character count from the original string (interior
    spacing preserved, boundaries trimmed).
    """
    from voxweave.core.layout import _token_char_count

    pieces: list[str] = []
    i = 0
    for _, atoms in runs:
        need = sum(_token_char_count(a["text"]) for a in atoms)
        j = i
        seen = 0
        while j < len(text) and seen < need:
            if not text[j].isspace():
                seen += 1
            j += 1
        pieces.append(text[i:j].strip())
        i = j
    if i < len(text) and pieces:  # trailing slack (whitespace) sticks to the last piece
        pieces[-1] = (pieces[-1] + text[i:]).strip()
    return pieces


def _run_span(
    atoms: list[dict], fallback_start: float, fallback_end: float
) -> tuple[float, float]:
    starts = [s for a in atoms if (s := a.get("start")) is not None]
    ends = [e for a in atoms if (e := a.get("end")) is not None]
    return (
        float(min(starts)) if starts else fallback_start,
        float(max(ends)) if ends else fallback_end,
    )


def _run_speech_span(atoms: list[dict]) -> tuple[float | None, float | None]:
    """Acoustic anchor of one speaker run: its timed atoms, or nothing.

    Deliberately not :func:`_run_span` -- that falls back to the parent cue's
    DISPLAY bounds, and a run with no timed atom must stay anchorless rather
    than inherit a lag-out pad or a shot lead-in as if it were speech.
    """
    starts = [s for a in atoms if (s := a.get("start")) is not None]
    ends = [e for a in atoms if (e := a.get("end")) is not None]
    return (
        float(min(starts)) if starts else None,
        float(max(ends)) if ends else None,
    )


def format_speaker_cues(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    annotate_speakers: bool = False,
) -> list[Cue]:
    """Speaker-aware post-pass over smart_split's cues (pure, replayable).

    Single-speaker cues pass through. A two-speaker cue becomes one Netflix
    dual-speaker event (``-line\\n-line``, hyphen without space, one speaker per
    line) when the language renders two lines and both halves fit one line;
    otherwise — and for 3+ speakers — the cue splits at the speaker boundaries
    with word-accurate timing. Lyric cues pass through untouched (the music-note
    wrap owns that display).

    ``max_line_length`` and ``max_lines`` are the same per-cue budgets that
    smart_split used for this file (``None`` = the language defaults), so the
    dual gate and the piece re-wrap render for the configured player instead of
    the built-in layout profile.  ``annotate_speakers`` adds transient diarizer
    ids for the display layer; the sibling JSON writer always removes them.
    """
    if not turns:
        return cues
    from voxweave.core.layout import (
        _line_budget_width,
        _vis_width,
        default_max_line_length,
        default_max_lines,
        wrap_cue_text,
    )
    from voxweave.core.smart_split import _build_atoms

    effective_max_line_length = (
        default_max_line_length(lang) if max_line_length is None else max_line_length
    )
    effective_max_lines = default_max_lines(lang) if max_lines is None else max_lines
    # Half-width cells -- the unit _vis_width and wrap_cue_text measure in.
    budget = _line_budget_width(effective_max_line_length, lang)
    dual_ok = effective_max_lines >= 2
    out: list[Cue] = []
    for cue in cues:
        word_data = cue.get("word_data") or []
        if cue.get("lyric") or not word_data:
            if annotate_speakers:
                tagged = cast(Cue, dict(cue))
                label = _span_speaker(cue.get("start"), cue.get("end"), turns)
                if label is not None:
                    tagged["speaker_ids"] = [label]
                out.append(tagged)
            else:
                out.append(cue)
            continue
        atoms = _build_atoms(
            cue["text"],
            cast(list, word_data),
            lang,
            max_atom_width=budget,
        )
        runs = _speaker_runs(atoms, turns, lang)
        if len(runs) <= 1:
            if annotate_speakers and runs:
                tagged = cast(Cue, dict(cue))
                tagged["speaker_ids"] = [runs[0][0]]
                out.append(tagged)
            else:
                out.append(cue)
            continue
        pieces = _slice_text_by_runs(cue["text"], runs)
        # Collapse each piece to one logical line before the dual-budget test:
        # smart_split may have soft-wrapped the cue, so a piece can carry an
        # interior "\n" that _vis_width would (wrongly) count as width 1, letting
        # a really-3-line dual event slip past the guard.
        one_line = [" ".join(p.split()) for p in pieces]
        if (
            len(runs) == 2
            and dual_ok
            and all(_vis_width(f"-{t}") <= budget for t in one_line)
        ):
            # Netflix dual-speaker event: one line per speaker, both within budget.
            dual = cast(Cue, dict(cue))
            dual["text"] = f"-{one_line[0]}\n-{one_line[1]}"
            if annotate_speakers:
                dual["speaker_ids"] = [label for label, _atoms in runs]
            out.append(dual)
            continue
        wd_cursor = 0
        for index, ((_label, atoms_run), piece) in enumerate(zip(runs, pieces)):
            # Slice by each run's recorded word_data footprint, not by atom
            # count: a repacked cue stores one entry per atom while a
            # first-generation one stores one per character. Entries the display
            # dropped (punctuation) sit between footprints and go to the run that
            # follows them; the last run takes the remainder, so a trailing
            # dropped entry is kept rather than lost from the stream.
            if index == len(runs) - 1:
                wd_end = len(word_data)
            elif atoms_run:
                wd_end = min(
                    max(wd_cursor, int(atoms_run[-1]["_unit_end"])), len(word_data)
                )
            else:
                wd_end = wd_cursor
            start, end = _run_span(atoms_run, cue["start"], cue["end"])
            part = cast(Cue, dict(cue))
            # Re-wrap the piece for its language (the same layout machinery
            # smart_split uses): an en split re-flows to <=2 clean lines, a zh/ja
            # piece stays one line and never carries a stale "\n".
            part["text"] = wrap_cue_text(
                piece,
                lang,
                effective_max_lines,
                max_line_length=budget,
            )
            part["start"] = start
            part["end"] = end
            part["speech_start"], part["speech_end"] = _run_speech_span(atoms_run)
            part["word_data"] = list(word_data[wd_cursor:wd_end])
            if annotate_speakers:
                part["speaker_ids"] = [_label]
            wd_cursor = wd_end
            out.append(part)
    return out


def _ordered_speaker_format(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    annotate_speakers: bool = False,
) -> list[Cue]:
    """format_speaker_cues + re-sort + overlap trim (splits can abut)."""
    out = format_speaker_cues(
        cues,
        turns,
        lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        annotate_speakers=annotate_speakers,
    )
    out.sort(key=lambda c: (c["start"], c["end"]))
    for prev, nxt in zip(out, out[1:]):
        prev["end"] = min(prev["end"], nxt["start"])
    return out


def apply_speaker_format(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    thresholds: dict | SplitThresholds | None = None,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    annotate_speakers: bool = False,
) -> list[Cue]:
    """Public entry: no-op without turns, otherwise format + keep cue order sane.

    When ``thresholds`` is given (the same gap thresholds smart_split used for this
    file), the formatted cues run through ``timing._cleanup_cues`` so speaker
    splits/dashes get the same timing polish as ordinary cues (short pieces extend
    into the following gap, sub-0.5s gaps chain) and never render as sub-flash
    cues. ``_cleanup_cues`` is timing-only and never merges content, so distinct
    speakers stay separate cues, and it is idempotent for cues carrying timed
    ``word_data``, so this second pass cannot stack another pad onto an already
    padded cue. ``thresholds=None`` keeps the pre-polish behavior for
    replay/back-compat callers. ``max_line_length`` and ``max_lines`` mirror
    smart_split's per-cue budgets (see :func:`format_speaker_cues`).
    """
    if not turns:
        return cues
    out = _ordered_speaker_format(
        cues,
        turns,
        lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        annotate_speakers=annotate_speakers,
    )
    if thresholds is not None:
        from voxweave.core.smart_split import SplitThresholds
        from voxweave.core.timing import _cleanup_cues

        th = (
            SplitThresholds.from_mapping(thresholds)
            if isinstance(thresholds, dict)
            else thresholds
        )
        out = _cleanup_cues(
            out,
            min_cue_s=th.min_cue_s,
            max_cue_s=th.max_cue_s,
            cps=th.cps,
            lag_out_s=th.lag_out_s,
        )
    return out
