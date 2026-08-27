"""Deterministic phase-2 voice matching and suggestion records."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

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
from voxweave.voicestore import (
    MAX_NAME_BYTES,
    ValidatedVoiceStore,
    validate_voice_store,
    voice_store_digest,
)

ENV_ACCEPT = "VOXWEAVE_VOICES_ACCEPT"
ENV_SUGGEST = "VOXWEAVE_VOICES_SUGGEST"
ENV_MARGIN = "VOXWEAVE_VOICES_MARGIN"

DEFAULT_ACCEPT = "off"
DEFAULT_SUGGEST = 0.45
DEFAULT_MARGIN = 0.05
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

    def as_mapping(self) -> dict[str, object]:
        return {
            "identity": self.identity_id,
            "display_name": self.display_name,
            "similarity": self.similarity,
            "exemplar": self.exemplar_id,
        }


@dataclass(frozen=True)
class SpeakerMatch:
    candidates: tuple[MatchCandidate, ...]
    truncated: int
    decision: Decision
    top_identity_id: str | None
    top_similarity: float | None

    def as_mapping(self) -> dict[str, object]:
        return {
            "candidates": [candidate.as_mapping() for candidate in self.candidates],
            "truncated": self.truncated,
            "decision": self.decision,
        }


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


def parse_thresholds(env: Mapping[str, str] | None = None) -> MatchThresholds:
    """Resolve the frozen environment policy without invalid-value defaults."""
    values = os.environ if env is None else env
    raw_accept = values.get(ENV_ACCEPT, DEFAULT_ACCEPT)
    if raw_accept.strip().lower() == "off":
        accept: float | None = None
    else:
        accept = _finite_float(raw_accept, ENV_ACCEPT)
    suggest = _finite_float(values.get(ENV_SUGGEST, str(DEFAULT_SUGGEST)), ENV_SUGGEST)
    margin = _finite_float(values.get(ENV_MARGIN, str(DEFAULT_MARGIN)), ENV_MARGIN)
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


def build_compatibility_fingerprint(
    provenance: Mapping[str, object],
) -> CompatibilityResult:
    """Hash resolved embedding-space provenance, excluding descriptive torch."""
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


def _best_identity_candidates(
    centroid: tuple[int | float, ...],
    store: ValidatedVoiceStore,
) -> list[MatchCandidate]:
    scores: list[MatchCandidate] = []
    centroid_floats = tuple(float(value) for value in centroid)
    for identity_id, raw_identity in store.identities.items():
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
                )
            )
    return sorted(scores, key=lambda item: (-item.similarity, item.identity_id))


def match_speakers(
    centroids: Mapping[str, object],
    store: Mapping[str, object],
    thresholds: MatchThresholds,
) -> dict[str, SpeakerMatch]:
    """Apply max-dot scoring, deterministic order, margin, and collision law."""
    validated_store = validate_voice_store(store)
    resolved_thresholds = validate_thresholds(thresholds)
    if len(centroids) > 64:
        raise Phase2DataError("centroids may contain at most 64 speakers")
    provisional: dict[str, SpeakerMatch] = {}
    eligible_top_owners: dict[str, list[str]] = {}

    for local_id in sorted(centroids):
        require_string(
            local_id,
            "local speaker id",
            max_bytes=MAX_SIDECAR_LABEL_BYTES,
        )
        centroid = validate_vector(
            centroids[local_id],
            dim=validated_store.embedding_dim,
            field=f"centroids.{local_id}",
        )
        all_scores = _best_identity_candidates(centroid, validated_store)
        qualifying = [
            candidate
            for candidate in all_scores
            if candidate.similarity >= resolved_thresholds.suggest
        ]
        kept = tuple(qualifying[:MAX_CANDIDATES])
        truncated = max(0, len(qualifying) - len(kept))
        if not all_scores or all_scores[0].similarity < resolved_thresholds.suggest:
            decision: Decision = "none"
            top_id: str | None = None
            top_similarity: float | None = None
        else:
            top = all_scores[0]
            top_id = top.identity_id
            top_similarity = top.similarity
            eligible_top_owners.setdefault(top_id, []).append(local_id)
            margin_ok = len(all_scores) == 1 or (
                top.similarity - all_scores[1].similarity >= resolved_thresholds.margin
            )
            if (
                resolved_thresholds.accept is not None
                and top.similarity >= resolved_thresholds.accept
                and margin_ok
            ):
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
        match = provisional[local_id]
        provisional[local_id] = SpeakerMatch(
            candidates=match.candidates,
            truncated=match.truncated,
            decision="collision",
            top_identity_id=match.top_identity_id,
            top_similarity=match.top_similarity,
        )
    return provisional


def _validate_record_thresholds(value: object) -> None:
    thresholds = require_mapping(value, "thresholds")
    raw_accept = thresholds.get("accept")
    if raw_accept == "off":
        accept = None
    else:
        accept = _exact_record_float(raw_accept, "thresholds.accept")
    validate_thresholds(
        MatchThresholds(
            accept=accept,
            suggest=_exact_record_float(
                thresholds.get("suggest"), "thresholds.suggest"
            ),
            margin=_exact_record_float(thresholds.get("margin"), "thresholds.margin"),
        )
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


def validate_suggest_record(value: object) -> None:
    """Validate a reproducible v1 suggestion record and its stable ordering."""
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
        match = require_mapping(raw_match, f"speakers.{local_id}")
        candidates = match.get("candidates")
        if not isinstance(candidates, list):
            raise Phase2DataError(f"speakers.{local_id}.candidates must be an array")
        if len(candidates) > MAX_CANDIDATES:
            raise Phase2DataError(
                f"speakers.{local_id}.candidates may contain at most {MAX_CANDIDATES}"
            )
        ordering: list[tuple[float, str]] = []
        for index, raw_candidate in enumerate(candidates):
            field = f"speakers.{local_id}.candidates[{index}]"
            candidate = require_mapping(raw_candidate, field)
            identity_id = require_identity_id(
                candidate.get("identity"), f"{field}.identity"
            )
            require_string(
                candidate.get("display_name"),
                f"{field}.display_name",
                max_bytes=MAX_NAME_BYTES,
            )
            similarity = _exact_record_float(
                candidate.get("similarity"), f"{field}.similarity"
            )
            require_exemplar_id(candidate.get("exemplar"), f"{field}.exemplar")
            ordering.append((-similarity, identity_id))
        if ordering != sorted(ordering):
            raise Phase2DataError(
                f"speakers.{local_id}.candidates are not deterministically ordered"
            )
        require_exact_int(
            match.get("truncated"),
            f"speakers.{local_id}.truncated",
            minimum=0,
        )
        if match.get("decision") not in {
            "prefill",
            "suggest",
            "collision",
            "none",
        }:
            raise Phase2DataError(f"speakers.{local_id}.decision is invalid")


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
        "voices": {
            "path": require_string(
                os.fspath(store_path),
                "voices.path",
                max_bytes=4096,
            ),
            "show": validated_store.show,
            "revision": validated_store.revision,
            "content_digest": voice_store_digest(store),
        },
        "compat_fingerprint": compatibility_hash,
        "thresholds": resolved_thresholds.as_mapping(),
        "speakers": {
            local_id: matches[local_id].as_mapping() for local_id in sorted(matches)
        },
    }
    validate_suggest_record(record)
    return record


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
    "DEFAULT_ACCEPT",
    "DEFAULT_MARGIN",
    "DEFAULT_SUGGEST",
    "ENV_ACCEPT",
    "ENV_MARGIN",
    "ENV_SUGGEST",
    "MAX_CANDIDATES",
    "MatchCandidate",
    "MatchThresholds",
    "SpeakerMatch",
    "ThresholdError",
    "build_compatibility_fingerprint",
    "build_suggest_record",
    "compatibility_equal",
    "delete_suggest",
    "load_suggest",
    "match_speakers",
    "parse_thresholds",
    "require_known_compatibility",
    "suggest_bytes",
    "validate_suggest_record",
    "validate_thresholds",
    "write_suggest",
]
