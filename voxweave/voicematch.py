"""Deterministic phase-2 voice matching and suggestion records."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

from voxweave import config
from voxweave.voicebase import (
    MAX_PROVENANCE_STRING_BYTES,
    MAX_SIDECAR_LABEL_BYTES,
    SUGGEST_MAX_BYTES,
    Phase2DataError,
    canonical_json_digest,
    encode_json_bytes,
    load_json_object,
    require_capture_id,
    require_dimension,
    require_exact_int,
    require_exemplar_id,
    require_identity_id,
    require_mapping,
    require_sha256,
    require_string,
    require_utc_timestamp,
    require_version_one,
    utc_timestamp,
    validate_vector,
    write_json_object,
)
from voxweave.voiceembed import ALIASES, AUTO, LANE_DECOUPLED, spec_by_name
from voxweave.voicestore import (
    MAX_NAME_BYTES,
    validate_voice_store,
    voice_store_digest,
)

ENV_ACCEPT = "VOXWEAVE_VOICES_ACCEPT"
ENV_SUGGEST = "VOXWEAVE_VOICES_SUGGEST"
ENV_MARGIN = "VOXWEAVE_VOICES_MARGIN"
ENV_GLOBAL_SUGGEST = "VOXWEAVE_VOICES_GLOBAL_SUGGEST"

DEFAULT_ACCEPT = "off"
# Defaults of the legacy pyannote lane; decoupled embedders carry their own
# (voiceembed registry, see threshold_defaults).
DEFAULT_SUGGEST = 0.45
DEFAULT_MARGIN = 0.05
# Tier 2 of the global voice library: identities enrolled under other scopes
# (other shows, other folders) are suggested only above a stricter bar.
# Open-set false suggestions grow with library size: if one impostor identity
# clears the bar with probability FAR, a library holding N impostors yields at
# least one false suggestion with probability 1 - (1 - FAR)^N (a FAR of 0.1%
# over 1000 identities is already 63%). Tier 1 (the episode's own scope) stays
# small, so it keeps the plain suggest threshold.
DEFAULT_GLOBAL_SUGGEST = 0.60  # provisional: calibrate
MAX_CANDIDATES = 5

Decision = Literal["prefill", "suggest", "collision", "none"]


class ThresholdError(Phase2DataError):
    """Resolved matching thresholds are malformed or internally inconsistent."""


class CompatibilityError(Phase2DataError):
    """Strict compatibility provenance is malformed."""


@dataclass(frozen=True)
class CompatibilityFingerprint:
    value: str

    def __post_init__(self) -> None:
        require_sha256(self.value, "compatibility fingerprint")


@dataclass(frozen=True, eq=False)
class CompatibilityUnknown:
    """Typed unknown result; no unknown value establishes compatibility."""

    unresolved_fields: tuple[str, ...]

    def __eq__(self, _other: object) -> bool:
        return False


CompatibilityResult: TypeAlias = CompatibilityFingerprint | CompatibilityUnknown


@dataclass(frozen=True)
class MatchThresholds:
    accept: float | None
    suggest: float
    margin: float

    def as_mapping(self) -> dict[str, object]:
        return {
            "accept": "off" if self.accept is None else self.accept,
            "suggest": self.suggest,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class MatchCandidate:
    identity_id: str
    display_name: str
    similarity: float
    exemplar_id: str
    # Library scopes the identity was enrolled under; None for a per-show store.
    scopes: tuple[str, ...] | None = None

    def as_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "identity": self.identity_id,
            "display_name": self.display_name,
            "similarity": self.similarity,
            "exemplar": self.exemplar_id,
        }
        if self.scopes is not None:
            mapping["scopes"] = list(self.scopes)
        return mapping


@dataclass(frozen=True)
class SpeakerMatch:
    candidates: tuple[MatchCandidate, ...]
    truncated: int
    decision: Decision
    top_identity_id: str | None
    top_similarity: float | None
    # Tier 2 of a library match (other scopes, stricter bar, never prefilled);
    # None for a per-show store, which has a single tier.
    secondary: SpeakerMatch | None = None

    def as_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "candidates": [candidate.as_mapping() for candidate in self.candidates],
            "truncated": self.truncated,
            "decision": self.decision,
        }
        if self.secondary is not None:
            mapping["secondary"] = self.secondary.as_mapping()
        return mapping


def _finite_float(raw: object, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise ThresholdError(f"{field} must be a finite number")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ThresholdError(f"{field} must be a finite number") from exc
    if not math.isfinite(value):
        raise ThresholdError(f"{field} must be a finite number")
    return value


def validate_thresholds(thresholds: MatchThresholds) -> MatchThresholds:
    suggest = _finite_float(thresholds.suggest, ENV_SUGGEST)
    margin = _finite_float(thresholds.margin, ENV_MARGIN)
    accept = (
        None
        if thresholds.accept is None
        else _finite_float(thresholds.accept, ENV_ACCEPT)
    )
    if not -1.0 <= suggest <= 1.0:
        raise ThresholdError(f"{ENV_SUGGEST} must be between -1 and 1")
    if accept is not None and not suggest <= accept <= 1.0:
        raise ThresholdError("thresholds must satisfy -1 <= suggest <= accept <= 1")
    if margin < 0.0:
        raise ThresholdError(f"{ENV_MARGIN} must be nonnegative")
    return MatchThresholds(accept=accept, suggest=suggest, margin=margin)


def threshold_defaults(
    provenance: Mapping[str, object] | None = None,
) -> tuple[float, float]:
    """Built-in ``(suggest, margin)`` for the embedding space ``provenance`` names.

    Cosine scales differ between embedders, so each registered decoupled
    embedder carries its own defaults; the legacy pyannote lane (and any
    embedder this version does not know) keeps :data:`DEFAULT_SUGGEST` /
    :data:`DEFAULT_MARGIN`.
    """
    if provenance is not None and provenance.get("embedding_lane") == LANE_DECOUPLED:
        spec = spec_by_name(provenance.get("embedding_model"))
        if spec is not None:
            return spec.suggest, spec.margin
    return DEFAULT_SUGGEST, DEFAULT_MARGIN


def global_suggest_default(provenance: Mapping[str, object] | None = None) -> float:
    """Built-in tier-2 (other scopes) suggest bar for ``provenance``'s space."""
    if provenance is not None and provenance.get("embedding_lane") == LANE_DECOUPLED:
        spec = spec_by_name(provenance.get("embedding_model"))
        if spec is not None:
            return spec.global_suggest
    return DEFAULT_GLOBAL_SUGGEST


def parse_global_suggest(
    env: Mapping[str, str] | None = None,
    *,
    provenance: Mapping[str, object] | None = None,
    suggest: float,
) -> float:
    """Resolve the tier-2 suggest bar: ``VOXWEAVE_VOICES_GLOBAL_SUGGEST`` or the
    embedding space's default.

    Tier 2 is never looser than tier 1: the result is at least ``suggest``, so
    lowering only ``VOXWEAVE_VOICES_SUGGEST`` cannot widen cross-scope matching.
    """
    values = os.environ if env is None else env
    raw = values.get(ENV_GLOBAL_SUGGEST)
    if raw is None or not raw.strip():
        value = global_suggest_default(provenance)
    else:
        value = _finite_float(raw, ENV_GLOBAL_SUGGEST)
    if not -1.0 <= value <= 1.0:
        raise ThresholdError(f"{ENV_GLOBAL_SUGGEST} must be between -1 and 1")
    return max(value, _finite_float(suggest, ENV_SUGGEST))


def parse_thresholds(
    env: Mapping[str, str] | None = None,
    *,
    provenance: Mapping[str, object] | None = None,
) -> MatchThresholds:
    """Resolve the frozen environment policy without invalid-value defaults.

    ``VOXWEAVE_VOICES_*`` always win; unset ones fall back to the defaults of the
    embedding space ``provenance`` describes (see :func:`threshold_defaults`).
    """
    values = os.environ if env is None else env
    default_suggest, default_margin = threshold_defaults(provenance)
    raw_accept = values.get(ENV_ACCEPT, DEFAULT_ACCEPT)
    if raw_accept.strip().lower() == "off":
        accept: float | None = None
    else:
        accept = _finite_float(raw_accept, ENV_ACCEPT)
    suggest = _finite_float(values.get(ENV_SUGGEST, str(default_suggest)), ENV_SUGGEST)
    margin = _finite_float(values.get(ENV_MARGIN, str(default_margin)), ENV_MARGIN)
    return validate_thresholds(
        MatchThresholds(accept=accept, suggest=suggest, margin=margin)
    )


def _strict_string(
    value: object,
    field: str,
    unresolved: list[str],
) -> str:
    text = require_string(
        value,
        field,
        max_bytes=MAX_PROVENANCE_STRING_BYTES,
    )
    if text == "unresolved":
        unresolved.append(field)
    return text


def _strict_sha(
    value: object,
    field: str,
    unresolved: list[str],
) -> str:
    if value == "unresolved":
        unresolved.append(field)
        return "unresolved"
    return require_sha256(value, field)


def _strict_int(
    value: object,
    field: str,
    unresolved: list[str],
    *,
    minimum: int,
    maximum: int | None = None,
) -> int | str:
    if value == "unresolved":
        unresolved.append(field)
        return "unresolved"
    return require_exact_int(
        value,
        field,
        minimum=minimum,
        maximum=maximum,
    )


def _strict_audio(
    provenance: Mapping[str, object],
    unresolved: list[str],
) -> dict[str, object]:
    """The audio profile both lanes hash: what signal the embedder heard."""
    audio = require_mapping(provenance.get("audio"), "audio")
    separated = audio.get("separated")
    normalized = audio.get("normalized")
    if type(separated) is not bool or type(normalized) is not bool:
        raise CompatibilityError("audio separated/normalized must be booleans")
    sample_rate = _strict_int(
        audio.get("sample_rate"),
        "audio.sample_rate",
        unresolved,
        minimum=1,
    )
    strict_audio: dict[str, object] = {
        "separated": separated,
        "normalized": normalized,
        "sample_rate": sample_rate,
    }
    if separated:
        separator = require_mapping(audio.get("separator"), "audio.separator")
        config_value = separator.get("config_sha256", separator.get("config"))
        strict_audio["separator"] = {
            "repo": _strict_string(
                separator.get("repo"),
                "audio.separator.repo",
                unresolved,
            ),
            "file": _strict_string(
                separator.get("file"),
                "audio.separator.file",
                unresolved,
            ),
            "checkpoint": _strict_string(
                separator.get("checkpoint"),
                "audio.separator.checkpoint",
                unresolved,
            ),
            "config_sha256": _strict_sha(
                config_value,
                "audio.separator.config_sha256",
                unresolved,
            ),
        }
    return strict_audio


# Decoupled-lane fields that define the embedding space, in reporting order.
DECOUPLED_FIELDS: tuple[str, ...] = (
    "embedding_model",
    "embedding_checkpoint",
    "embedding_dim",
    "embedding_recipe",
)


def _decoupled_fingerprint(provenance: Mapping[str, object]) -> CompatibilityResult:
    """Hash only what defines a dedicated embedder's space.

    The diarization pipeline, its config digest and the pyannote version are
    deliberately excluded: they choose the turns, not the vector space, so
    switching the diarizer must not orphan a voice store.
    """
    unresolved: list[str] = []
    try:
        for field in DECOUPLED_FIELDS:
            if provenance.get(field) is None:
                unresolved.append(field)
        model = provenance.get("embedding_model")
        if model is not None:
            _strict_string(model, "embedding_model", unresolved)
        checkpoint = provenance.get("embedding_checkpoint")
        if checkpoint is not None:
            _strict_sha(checkpoint, "embedding_checkpoint", unresolved)
        raw_dim = provenance.get("embedding_dim")
        if raw_dim == "unresolved":
            unresolved.append("embedding_dim")
        elif raw_dim is not None:
            require_dimension(raw_dim)
        recipe = provenance.get("embedding_recipe")
        if recipe is not None:
            _strict_string(recipe, "embedding_recipe", unresolved)
        strict_audio: dict[str, object] | None = None
        if provenance.get("audio") is None:
            unresolved.append("audio")
        else:
            strict_audio = _strict_audio(provenance, unresolved)
    except CompatibilityError:
        raise
    except Phase2DataError as exc:
        raise CompatibilityError(str(exc)) from exc
    if unresolved:
        return CompatibilityUnknown(tuple(sorted(set(unresolved))))
    strict = {
        "embedding_lane": LANE_DECOUPLED,
        "embedding_model": model,
        "embedding_checkpoint": checkpoint,
        "embedding_dim": raw_dim,
        "embedding_recipe": recipe,
        "audio": strict_audio,
    }
    return CompatibilityFingerprint(canonical_json_digest(strict))


def build_compatibility_fingerprint(
    provenance: Mapping[str, object],
) -> CompatibilityResult:
    """Hash resolved embedding-space provenance, excluding descriptive torch.

    Provenance without ``embedding_lane`` is the legacy pyannote lane and keeps
    its original digest byte for byte; ``embedding_lane: "decoupled"`` hashes
    only the dedicated embedder's space (see :func:`_decoupled_fingerprint`).
    Any other lane value is unknown and never compatible.
    """
    if "embedding_lane" in provenance:
        lane = provenance.get("embedding_lane")
        if lane == LANE_DECOUPLED:
            return _decoupled_fingerprint(provenance)
        return CompatibilityUnknown(("embedding_lane",))
    unresolved: list[str] = []
    try:
        outer_model = _strict_string(
            provenance.get("diarization_model"),
            "diarization_model",
            unresolved,
        )
        outer_config = _strict_sha(
            provenance.get("outer_config_sha256"),
            "outer_config_sha256",
            unresolved,
        )
        embedding_model = _strict_string(
            provenance.get("embedding_model"),
            "embedding_model",
            unresolved,
        )
        embedding_checkpoint = _strict_string(
            provenance.get("embedding_checkpoint"),
            "embedding_checkpoint",
            unresolved,
        )
        raw_dim = provenance.get("embedding_dim")
        if raw_dim == "unresolved":
            unresolved.append("embedding_dim")
            embedding_dim: int | str = "unresolved"
        else:
            embedding_dim = require_dimension(raw_dim)
        pyannote_version = _strict_string(
            provenance.get("pyannote_version"),
            "pyannote_version",
            unresolved,
        )
        require_string(
            provenance.get("torch_version"),
            "torch_version",
            max_bytes=MAX_PROVENANCE_STRING_BYTES,
        )
        strict_audio = _strict_audio(provenance, unresolved)
    except CompatibilityError:
        raise
    except Phase2DataError as exc:
        raise CompatibilityError(str(exc)) from exc

    strict = {
        "diarization_model": outer_model,
        "outer_config_sha256": outer_config,
        "embedding_model": embedding_model,
        "embedding_checkpoint": embedding_checkpoint,
        "embedding_dim": embedding_dim,
        "audio": strict_audio,
        "pyannote_version": pyannote_version,
    }
    if unresolved:
        return CompatibilityUnknown(tuple(sorted(set(unresolved))))
    return CompatibilityFingerprint(canonical_json_digest(strict))


def compatibility_equal(left: CompatibilityResult, right: CompatibilityResult) -> bool:
    """Return true only for two resolved, byte-equal compatibility hashes."""
    return (
        isinstance(left, CompatibilityFingerprint)
        and isinstance(right, CompatibilityFingerprint)
        and left.value == right.value
    )


def require_known_compatibility(result: CompatibilityResult) -> str:
    if isinstance(result, CompatibilityUnknown):
        fields = ", ".join(result.unresolved_fields)
        raise CompatibilityError(f"compatibility is unknown: {fields}")
    return result.value


# Provenance fields worth naming to a human when two embedding spaces differ.
# outer_config_sha256 is deliberately absent: it is a derived digest that moves
# with the pipeline, so printing two hashes adds noise instead of a reason.
_REPORTED_PROVENANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("diarization_model", "diarization model"),
    ("embedding_model", "embedding model"),
    ("embedding_checkpoint", "embedding checkpoint"),
    ("embedding_dim", "embedding dimension"),
    ("pyannote_version", "pyannote version"),
)

LEGACY_DIARIZE_HINT = (
    "run transcription and speakers with --diarize-model 3.1 "
    '(or [diarize].model = "3.1") to keep matching against this store, '
    "or re-enroll it under the new model"
)


def _readable_provenance_value(value: object) -> str:
    """Render one provenance scalar for a human-facing diagnostic."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.strip() or "unrecorded"
    if isinstance(value, int | float):
        return str(value)
    return "unrecorded"


def _audio_summary(value: object) -> str:
    if not isinstance(value, Mapping):
        return "unrecorded"
    parts = [
        f"separated {_readable_provenance_value(value.get('separated'))}",
        f"normalized {_readable_provenance_value(value.get('normalized'))}",
        f"sample rate {_readable_provenance_value(value.get('sample_rate'))}",
    ]
    separator = value.get("separator")
    if isinstance(separator, Mapping):
        checkpoint = _readable_provenance_value(separator.get("checkpoint"))
        parts.append(f"separator {checkpoint}")
    return ", ".join(parts)


_DECOUPLED_REPORTED_FIELDS: tuple[tuple[str, str], ...] = (
    ("embedding_checkpoint", "embedding checkpoint"),
    ("embedding_dim", "embedding dimension"),
    ("embedding_recipe", "centroid recipe"),
)

LEGACY_VOICEPRINT_HINT = (
    "run transcription with --voiceprint-model pyannote "
    '(or [voiceprint].model = "pyannote") to keep matching against this legacy '
    "store, or re-enroll it with the new embedder"
)


def _is_decoupled(provenance: Mapping[str, object]) -> bool:
    return provenance.get("embedding_lane") == LANE_DECOUPLED


def embedding_space_label(provenance: Mapping[str, object]) -> str:
    """Name the embedding space a provenance record describes, for humans."""
    model = _readable_provenance_value(provenance.get("embedding_model"))
    if _is_decoupled(provenance):
        return f"{model} embeddings"
    if "embedding_lane" in provenance:
        lane = _readable_provenance_value(provenance.get("embedding_lane"))
        return f"an unknown embedding lane ({lane})"
    return f"pyannote embeddings ({model})"


def _audio_difference(
    episode_provenance: Mapping[str, object],
    store_provenance: Mapping[str, object],
) -> list[str]:
    stored_audio = store_provenance.get("audio")
    current_audio = episode_provenance.get("audio")
    if stored_audio == current_audio:
        return []
    return [
        f"audio profile: store {_audio_summary(stored_audio)}, "
        f"this run {_audio_summary(current_audio)}"
    ]


def _space_differences(
    episode_provenance: Mapping[str, object],
    store_provenance: Mapping[str, object],
) -> list[str] | None:
    """Differences once either side is outside the legacy lane, else ``None``."""
    episode_lane = episode_provenance.get("embedding_lane")
    store_lane = store_provenance.get("embedding_lane")
    if episode_lane is None and store_lane is None:
        return None
    space = (
        f"store was built with {embedding_space_label(store_provenance)}; "
        f"this run uses {embedding_space_label(episode_provenance)}"
    )
    if episode_lane != store_lane:
        return [space]
    differences: list[str] = []
    if store_provenance.get("embedding_model") != episode_provenance.get(
        "embedding_model"
    ):
        differences.append(space)
    for field, label in _DECOUPLED_REPORTED_FIELDS:
        stored = store_provenance.get(field)
        current = episode_provenance.get(field)
        if stored == current:
            continue
        differences.append(
            f"{label}: store {_readable_provenance_value(stored)}, "
            f"this run {_readable_provenance_value(current)}"
        )
    return differences + _audio_difference(episode_provenance, store_provenance)


def _provenance_differences(
    episode_provenance: Mapping[str, object],
    store_provenance: Mapping[str, object],
) -> list[str]:
    space = _space_differences(episode_provenance, store_provenance)
    if space is not None:
        return space
    differences: list[str] = []
    for field, label in _REPORTED_PROVENANCE_FIELDS:
        stored = store_provenance.get(field)
        current = episode_provenance.get(field)
        if stored == current:
            continue
        differences.append(
            f"{label}: store {_readable_provenance_value(stored)}, "
            f"this run {_readable_provenance_value(current)}"
        )
    return differences + _audio_difference(episode_provenance, store_provenance)


def _describe_unresolved(
    *,
    episode: CompatibilityResult,
    store: CompatibilityResult,
) -> str | None:
    parts: list[str] = []
    if isinstance(store, CompatibilityUnknown):
        fields = ", ".join(store.unresolved_fields)
        parts.append(f"the store records no resolved {fields}")
    if isinstance(episode, CompatibilityUnknown):
        fields = ", ".join(episode.unresolved_fields)
        parts.append(f"this run records no resolved {fields}")
    if not parts:
        return None
    return "compatibility is unresolved; " + "; ".join(parts)


def legacy_diarize_recovery_hint(
    episode_provenance: Mapping[str, object],
    store_provenance: Mapping[str, object],
) -> str | None:
    """Return the way back when only the diarizer default flip separates them.

    Only the legacy lane depends on the diarizer: a store in a dedicated
    embedder's space never needs the old pipeline back.
    """
    if (
        "embedding_lane" not in store_provenance
        and store_provenance.get("diarization_model") == config.LEGACY_DIARIZE_MODEL
        and episode_provenance.get("diarization_model") == config.DEFAULT_DIARIZE_MODEL
    ):
        return LEGACY_DIARIZE_HINT
    return None


def _voiceprint_alias(model: object) -> str:
    """The ``--voiceprint-model`` value that selects a recorded embedder."""
    for alias, name in ALIASES.items():
        if name == model and alias != AUTO:
            return alias
    return _readable_provenance_value(model)


def embedding_space_recovery_hint(
    episode_provenance: Mapping[str, object],
    store_provenance: Mapping[str, object],
) -> str | None:
    """Return the way back when the voiceprint embedder separates them."""
    episode_lane = episode_provenance.get("embedding_lane")
    store_lane = store_provenance.get("embedding_lane")
    if store_lane is None and episode_lane == LANE_DECOUPLED:
        hint = LEGACY_VOICEPRINT_HINT
        diarize_hint = legacy_diarize_recovery_hint(
            episode_provenance, store_provenance
        )
        if diarize_hint is not None:
            hint = (
                f"{hint}; the store also predates the diarizer default: {diarize_hint}"
            )
        return hint
    if store_lane == LANE_DECOUPLED and store_provenance.get(
        "embedding_model"
    ) != episode_provenance.get("embedding_model"):
        alias = _voiceprint_alias(store_provenance.get("embedding_model"))
        return (
            f"run transcription with --voiceprint-model {alias} to keep matching "
            "against this store, or re-enroll it with the new embedder"
        )
    return None


def describe_compatibility_mismatch(
    episode_provenance: Mapping[str, object],
    store_provenance: Mapping[str, object],
    *,
    episode: CompatibilityResult,
    store: CompatibilityResult,
) -> str:
    """Say in plain words why a voices store cannot serve this episode.

    Reporting only: callers pass the fingerprints they already computed, and
    nothing here feeds the matching decision. Unresolved provenance keeps its
    own wording because no model swap explains it.
    """
    unresolved = _describe_unresolved(episode=episode, store=store)
    if unresolved is not None:
        return unresolved
    differences = _provenance_differences(episode_provenance, store_provenance)
    if not differences:
        differences = ["the recorded pipeline configuration differs"]
    detail = "compatibility differs; " + "; ".join(differences)
    hint = embedding_space_recovery_hint(
        episode_provenance, store_provenance
    ) or legacy_diarize_recovery_hint(episode_provenance, store_provenance)
    return detail if hint is None else f"{detail}. {hint}"


def _best_identity_candidates(
    centroid: tuple[int | float, ...],
    identities: Mapping[str, object],
    scopes: Mapping[str, tuple[str, ...]] | None = None,
) -> list[MatchCandidate]:
    """Score every identity by its best exemplar, in deterministic order.

    ``identities`` maps ids to ``{"display_name", "exemplars": [{"id",
    "vector"}]}``: the per-show store's identities or one library tier.
    """
    scores: list[MatchCandidate] = []
    centroid_floats = tuple(float(value) for value in centroid)
    for identity_id, raw_identity in identities.items():
        identity = cast(Mapping[str, object], raw_identity)
        display_name = cast(str, identity["display_name"])
        exemplars = cast(list[Mapping[str, object]], identity["exemplars"])
        best: tuple[float, str] | None = None
        for exemplar in exemplars:
            exemplar_id = cast(str, exemplar["id"])
            vector = cast(list[int | float], exemplar["vector"])
            similarity = math.fsum(
                left * float(right)
                for left, right in zip(centroid_floats, vector, strict=True)
            )
            choice = (similarity, exemplar_id)
            if (
                best is None
                or similarity > best[0]
                or (similarity == best[0] and exemplar_id < best[1])
            ):
                best = choice
        if best is not None:
            scores.append(
                MatchCandidate(
                    identity_id=identity_id,
                    display_name=display_name,
                    similarity=best[0],
                    exemplar_id=best[1],
                    scopes=None if scopes is None else scopes.get(identity_id, ()),
                )
            )
    return sorted(scores, key=lambda item: (-item.similarity, item.identity_id))


def _validated_centroids(
    centroids: Mapping[str, object], embedding_dim: int
) -> dict[str, tuple[int | float, ...]]:
    if len(centroids) > 64:
        raise Phase2DataError("centroids may contain at most 64 speakers")
    validated: dict[str, tuple[int | float, ...]] = {}
    for local_id in sorted(centroids):
        require_string(
            local_id,
            "local speaker id",
            max_bytes=MAX_SIDECAR_LABEL_BYTES,
        )
        validated[local_id] = validate_vector(
            centroids[local_id],
            dim=embedding_dim,
            field=f"centroids.{local_id}",
        )
    return validated


def _match_pool(
    centroids: Mapping[str, tuple[int | float, ...]],
    identities: Mapping[str, object],
    *,
    suggest: float,
    margin: float,
    accept: float | None,
    scopes: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, SpeakerMatch]:
    """Max-dot scoring, deterministic order, margin, and collision law for
    one pool of identities (validated centroids and identities)."""
    provisional: dict[str, SpeakerMatch] = {}
    eligible_top_owners: dict[str, list[str]] = {}

    for local_id in sorted(centroids):
        all_scores = _best_identity_candidates(centroids[local_id], identities, scopes)
        qualifying = [
            candidate for candidate in all_scores if candidate.similarity >= suggest
        ]
        kept = tuple(qualifying[:MAX_CANDIDATES])
        truncated = max(0, len(qualifying) - len(kept))
        if not all_scores or all_scores[0].similarity < suggest:
            decision: Decision = "none"
            top_id: str | None = None
            top_similarity: float | None = None
        else:
            top = all_scores[0]
            top_id = top.identity_id
            top_similarity = top.similarity
            eligible_top_owners.setdefault(top_id, []).append(local_id)
            margin_ok = len(all_scores) == 1 or (
                top.similarity - all_scores[1].similarity >= margin
            )
            if accept is not None and top.similarity >= accept and margin_ok:
                decision = "prefill"
            else:
                decision = "suggest"
        provisional[local_id] = SpeakerMatch(
            candidates=kept,
            truncated=truncated,
            decision=decision,
            top_identity_id=top_id,
            top_similarity=top_similarity,
        )

    collisions = {
        local_id
        for owners in eligible_top_owners.values()
        if len(owners) > 1
        for local_id in owners
    }
    for local_id in collisions:
        provisional[local_id] = replace(provisional[local_id], decision="collision")
    return provisional


def match_speakers(
    centroids: Mapping[str, object],
    store: Mapping[str, object],
    thresholds: MatchThresholds,
) -> dict[str, SpeakerMatch]:
    """Apply max-dot scoring, deterministic order, margin, and collision law."""
    validated_store = validate_voice_store(store)
    resolved_thresholds = validate_thresholds(thresholds)
    validated = _validated_centroids(centroids, validated_store.embedding_dim)
    return _match_pool(
        validated,
        validated_store.identities,
        suggest=resolved_thresholds.suggest,
        margin=resolved_thresholds.margin,
        accept=resolved_thresholds.accept,
    )


def match_tiers(
    centroids: Mapping[str, object],
    *,
    in_scope: Mapping[str, object],
    other_scopes: Mapping[str, object],
    scopes: Mapping[str, tuple[str, ...]],
    embedding_dim: int,
    thresholds: MatchThresholds,
    global_suggest: float,
) -> dict[str, SpeakerMatch]:
    """Two-tier suggest-only matching against a voice library.

    Tier 1 (``in_scope``: identities enrolled under the episode's scope) keeps
    the plain thresholds, including an operator's accept/prefill. Tier 2
    (``other_scopes``) uses the stricter ``global_suggest`` bar, is never
    prefilled, and carries each identity's scopes so the review page can say
    where the voice was heard. Collision law applies within each tier only.
    Both pools must already be validated library identities of one embedding
    space with ``embedding_dim``-long vectors.
    """
    resolved = validate_thresholds(thresholds)
    bar = _finite_float(global_suggest, ENV_GLOBAL_SUGGEST)
    if not resolved.suggest <= bar <= 1.0:
        raise ThresholdError("thresholds must satisfy suggest <= global suggest <= 1")
    validated = _validated_centroids(centroids, embedding_dim)
    primary = _match_pool(
        validated,
        in_scope,
        suggest=resolved.suggest,
        margin=resolved.margin,
        accept=resolved.accept,
        scopes=scopes,
    )
    secondary = _match_pool(
        validated,
        other_scopes,
        suggest=bar,
        margin=resolved.margin,
        accept=None,
        scopes=scopes,
    )
    return {
        local_id: replace(primary[local_id], secondary=secondary[local_id])
        for local_id in primary
    }


def _validate_record_thresholds(value: object) -> None:
    thresholds = require_mapping(value, "thresholds")
    raw_accept = thresholds.get("accept")
    if raw_accept == "off":
        accept = None
    else:
        accept = _exact_record_float(raw_accept, "thresholds.accept")
    resolved = validate_thresholds(
        MatchThresholds(
            accept=accept,
            suggest=_exact_record_float(
                thresholds.get("suggest"), "thresholds.suggest"
            ),
            margin=_exact_record_float(thresholds.get("margin"), "thresholds.margin"),
        )
    )
    if "global_suggest" in thresholds:
        bar = _exact_record_float(
            thresholds.get("global_suggest"), "thresholds.global_suggest"
        )
        if not resolved.suggest <= bar <= 1.0:
            raise Phase2DataError(
                "thresholds.global_suggest must lie between suggest and 1"
            )


def _exact_record_float(raw: object, field: str) -> float:
    if type(raw) not in (int, float):
        raise Phase2DataError(f"{field} must be a non-bool number")
    try:
        value = float(cast(int | float, raw))
    except OverflowError as exc:
        raise Phase2DataError(f"{field} exceeds finite range") from exc
    if not math.isfinite(value):
        raise Phase2DataError(f"{field} must be finite")
    return value


_TIER_ONE_DECISIONS = frozenset({"prefill", "suggest", "collision", "none"})
# Tier 2 is suggest-only by construction: it is never prefilled.
_TIER_TWO_DECISIONS = frozenset({"suggest", "collision", "none"})
MAX_RECORD_SCOPES = 256


def _validate_record_match(
    match: Mapping[str, object], field: str, *, decisions: frozenset[str]
) -> None:
    candidates = match.get("candidates")
    if not isinstance(candidates, list):
        raise Phase2DataError(f"{field}.candidates must be an array")
    if len(candidates) > MAX_CANDIDATES:
        raise Phase2DataError(
            f"{field}.candidates may contain at most {MAX_CANDIDATES}"
        )
    ordering: list[tuple[float, str]] = []
    for index, raw_candidate in enumerate(candidates):
        candidate_field = f"{field}.candidates[{index}]"
        candidate = require_mapping(raw_candidate, candidate_field)
        identity_id = require_identity_id(
            candidate.get("identity"), f"{candidate_field}.identity"
        )
        require_string(
            candidate.get("display_name"),
            f"{candidate_field}.display_name",
            max_bytes=MAX_NAME_BYTES,
        )
        similarity = _exact_record_float(
            candidate.get("similarity"), f"{candidate_field}.similarity"
        )
        require_exemplar_id(candidate.get("exemplar"), f"{candidate_field}.exemplar")
        if "scopes" in candidate:
            scopes = candidate.get("scopes")
            if not isinstance(scopes, list) or len(scopes) > MAX_RECORD_SCOPES:
                raise Phase2DataError(
                    f"{candidate_field}.scopes must be an array of at most "
                    f"{MAX_RECORD_SCOPES} names"
                )
            for position, scope in enumerate(scopes):
                require_string(
                    scope,
                    f"{candidate_field}.scopes[{position}]",
                    max_bytes=MAX_NAME_BYTES,
                )
        ordering.append((-similarity, identity_id))
    if ordering != sorted(ordering):
        raise Phase2DataError(f"{field}.candidates are not deterministically ordered")
    require_exact_int(match.get("truncated"), f"{field}.truncated", minimum=0)
    if match.get("decision") not in decisions:
        raise Phase2DataError(f"{field}.decision is invalid")


def validate_suggest_record(value: object) -> None:
    """Validate a reproducible v1 suggestion record and its stable ordering.

    A voice-library record additionally carries each speaker's tier-2
    ``secondary`` match, candidate ``scopes`` and ``thresholds.global_suggest``.
    """
    root = require_mapping(value, "suggest record")
    require_version_one(root.get("version"))
    require_utc_timestamp(root.get("generated"), "generated")
    require_capture_id(root.get("capture_id"))
    require_sha256(root.get("voiceprints_digest"), "voiceprints_digest")
    require_sha256(root.get("compat_fingerprint"), "compat_fingerprint")
    voices = require_mapping(root.get("voices"), "voices")
    require_string(voices.get("path"), "voices.path", max_bytes=4096)
    require_string(voices.get("show"), "voices.show", max_bytes=MAX_NAME_BYTES)
    require_exact_int(voices.get("revision"), "voices.revision", minimum=0)
    require_sha256(voices.get("content_digest"), "voices.content_digest")
    _validate_record_thresholds(root.get("thresholds"))

    speakers = require_mapping(root.get("speakers"), "speakers")
    if len(speakers) > 64:
        raise Phase2DataError("suggest record may contain at most 64 speakers")
    for local_id, raw_match in speakers.items():
        require_string(local_id, "local speaker id", max_bytes=MAX_SIDECAR_LABEL_BYTES)
        field = f"speakers.{local_id}"
        match = require_mapping(raw_match, field)
        _validate_record_match(match, field, decisions=_TIER_ONE_DECISIONS)
        if "secondary" in match:
            secondary = require_mapping(match.get("secondary"), f"{field}.secondary")
            if "secondary" in secondary:
                raise Phase2DataError(f"{field}.secondary cannot nest another tier")
            _validate_record_match(
                secondary, f"{field}.secondary", decisions=_TIER_TWO_DECISIONS
            )


def _suggest_record(
    matches: Mapping[str, SpeakerMatch],
    *,
    capture_id: str,
    voiceprints_content_digest: str,
    compatibility: CompatibilityResult | str,
    thresholds: Mapping[str, object],
    voices: Mapping[str, object],
    generated: str | None,
) -> dict[str, object]:
    if isinstance(compatibility, str):
        compatibility_hash = require_sha256(compatibility, "compatibility fingerprint")
    else:
        compatibility_hash = require_known_compatibility(compatibility)
    record: dict[str, object] = {
        "version": 1,
        "generated": generated or utc_timestamp(),
        "capture_id": require_capture_id(capture_id),
        "voiceprints_digest": require_sha256(
            voiceprints_content_digest,
            "voiceprints_digest",
        ),
        "voices": dict(voices),
        "compat_fingerprint": compatibility_hash,
        "thresholds": dict(thresholds),
        "speakers": {
            local_id: matches[local_id].as_mapping() for local_id in sorted(matches)
        },
    }
    validate_suggest_record(record)
    return record


def build_suggest_record(
    matches: Mapping[str, SpeakerMatch],
    *,
    capture_id: str,
    voiceprints_content_digest: str,
    compatibility: CompatibilityResult | str,
    thresholds: MatchThresholds,
    store_path: Path,
    store: Mapping[str, object],
    generated: str | None = None,
) -> dict[str, object]:
    """Build the full provenance record for one reproducible match run."""
    validated_store = validate_voice_store(store)
    resolved_thresholds = validate_thresholds(thresholds)
    return _suggest_record(
        matches,
        capture_id=capture_id,
        voiceprints_content_digest=voiceprints_content_digest,
        compatibility=compatibility,
        thresholds=resolved_thresholds.as_mapping(),
        voices={
            "path": require_string(
                os.fspath(store_path),
                "voices.path",
                max_bytes=4096,
            ),
            "show": validated_store.show,
            "revision": validated_store.revision,
            "content_digest": voice_store_digest(store),
        },
        generated=generated,
    )


def build_library_suggest_record(
    matches: Mapping[str, SpeakerMatch],
    *,
    capture_id: str,
    voiceprints_content_digest: str,
    compatibility: CompatibilityResult | str,
    thresholds: MatchThresholds,
    global_suggest: float,
    library_path: Path,
    scope: str,
    revision: int,
    content_digest: str,
    generated: str | None = None,
) -> dict[str, object]:
    """Build the record of one two-tier match run against a voice library.

    ``voices.show`` holds the episode's scope, ``revision`` the embedding
    space's revision and ``content_digest`` the digest of every library input
    the match read, so the record stays reproducible like a per-show one.
    """
    resolved_thresholds = validate_thresholds(thresholds)
    return _suggest_record(
        matches,
        capture_id=capture_id,
        voiceprints_content_digest=voiceprints_content_digest,
        compatibility=compatibility,
        thresholds={
            **resolved_thresholds.as_mapping(),
            "global_suggest": _finite_float(global_suggest, ENV_GLOBAL_SUGGEST),
        },
        voices={
            "path": require_string(
                os.fspath(library_path),
                "voices.path",
                max_bytes=4096,
            ),
            "show": require_string(scope, "voices.show", max_bytes=MAX_NAME_BYTES),
            "revision": require_exact_int(revision, "voices.revision", minimum=0),
            "content_digest": require_sha256(content_digest, "voices.content_digest"),
        },
        generated=generated,
    )


def suggest_bytes(value: Mapping[str, object]) -> bytes:
    validate_suggest_record(value)
    return encode_json_bytes(value, max_bytes=SUGGEST_MAX_BYTES)


def load_suggest(path: Path) -> dict[str, object]:
    raw = load_json_object(path, max_bytes=SUGGEST_MAX_BYTES)
    validate_suggest_record(raw)
    return raw


def write_suggest(path: Path, value: Mapping[str, object]) -> None:
    validate_suggest_record(value)
    write_json_object(path, value, max_bytes=SUGGEST_MAX_BYTES)


def delete_suggest(path: Path) -> None:
    Path(path).unlink(missing_ok=True)


__all__ = [
    "CompatibilityError",
    "CompatibilityFingerprint",
    "CompatibilityResult",
    "CompatibilityUnknown",
    "DECOUPLED_FIELDS",
    "DEFAULT_ACCEPT",
    "DEFAULT_GLOBAL_SUGGEST",
    "DEFAULT_MARGIN",
    "DEFAULT_SUGGEST",
    "ENV_ACCEPT",
    "ENV_GLOBAL_SUGGEST",
    "ENV_MARGIN",
    "ENV_SUGGEST",
    "LEGACY_DIARIZE_HINT",
    "LEGACY_VOICEPRINT_HINT",
    "MAX_CANDIDATES",
    "MatchCandidate",
    "MatchThresholds",
    "SpeakerMatch",
    "ThresholdError",
    "build_compatibility_fingerprint",
    "build_library_suggest_record",
    "build_suggest_record",
    "compatibility_equal",
    "delete_suggest",
    "describe_compatibility_mismatch",
    "embedding_space_label",
    "embedding_space_recovery_hint",
    "global_suggest_default",
    "legacy_diarize_recovery_hint",
    "load_suggest",
    "match_speakers",
    "match_tiers",
    "parse_global_suggest",
    "parse_thresholds",
    "require_known_compatibility",
    "suggest_bytes",
    "threshold_defaults",
    "validate_suggest_record",
    "validate_thresholds",
    "write_suggest",
]
