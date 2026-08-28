"""Rich immutable observation for a completed selected align transaction."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from voxweave.align_evidence import encode_align_evidence
from voxweave.align_failures import CanonicalFailure


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _immutable(member) for key, member in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(member) for member in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return [_json_value(member) for member in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class RichAlignShadowArtifact:
    schema_version: Literal[2]
    artifact_kind: Literal["rich"]
    status: Literal["valid", "invalid"]
    failure: CanonicalFailure | None
    input: Mapping[str, Any]
    fresh: Mapping[str, Any]
    legacy: Mapping[str, Any]
    v2: Mapping[str, Any]
    comparison: Mapping[str, Any]
    selected: Mapping[str, str]

    def to_canonical_bytes(self) -> bytes:
        value = {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "input": self.input,
            "fresh": self.fresh,
            "legacy": self.legacy,
            "v2": self.v2,
            "comparison": self.comparison,
            "selected": self.selected,
        }
        return _canonical_bytes(value)


def _normalized_delivery(cues: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_index": cue.source_index,
            "text": cue.text,
            "start": cue.start,
            "end": cue.end,
            "lyric": cue.lyric,
            "unit_ids": list(cue.unit_ids),
        }
        for cue in cues
    ]


def build_rich_align_shadow_artifact(
    *,
    selection: Any,
    input_summary: Mapping[str, Any],
    prepared_audio_sha256: str,
) -> RichAlignShadowArtifact:
    """Construct and canonicalize the complete post-commit rich observation."""
    result = selection.result
    verified = selection.verified
    core = result.evidence_core
    failure = result.v2_status.failure
    status: Literal["valid", "invalid"] = "invalid" if failure is not None else "valid"
    distribution_value = dataclasses.asdict(selection.distribution)
    evidence_sha256 = hashlib.sha256(
        encode_align_evidence(selection.evidence)
    ).hexdigest()
    legacy_delivery = _normalized_delivery(result.legacy.cues)
    artifact = RichAlignShadowArtifact(
        2,
        "rich",
        status,
        failure,
        _immutable(dict(input_summary)),
        _immutable(
            {
                "receipt_digest": result.receipt_digest,
                "prepared_audio_sha256": prepared_audio_sha256,
                "physical_call_count": len(core.physical_calls),
                "raw_unit_count": core.raw_unit_count,
                "legacy_distribution_digest": _digest(legacy_delivery),
                "authority_distribution_digest": _digest(distribution_value),
                "authority_distribution_status": core.authority_status,
                "seed_status": core.seed_status,
                "strict_input_status": (selection.strict_input_status.kind),
                "v2_policy_status": selection.v2_policy_status.kind,
                "profile_status": selection.profile_status.kind,
                "evidence_status": selection.evidence_status.kind,
                "v2_admission_status": (
                    "valid"
                    if result.v2_status.kind == "valid"
                    else result.v2_status.kind
                ),
            }
        ),
        _immutable(
            {
                "normalized_delivery": legacy_delivery,
                "vtt_sha256": hashlib.sha256(selection.verified.vtt_bytes).hexdigest(),
                "json_sha256": hashlib.sha256(
                    selection.verified.main_json_bytes
                ).hexdigest(),
            }
        ),
        _immutable({"semantic": None, "validators": None}),
        _immutable({"result": None}),
        _immutable(
            {
                "engine_family": verified.engine_family,
                "vtt_sha256": verified.vtt_sha256,
                "json_sha256": verified.main_json_sha256,
                "evidence_sha256": evidence_sha256,
            }
        ),
    )
    if status == "valid" and failure is not None:
        raise ValueError("valid rich shadow artifact carries a failure")
    if status == "invalid" and failure is None:
        raise ValueError("invalid rich shadow artifact lacks a failure")
    artifact.to_canonical_bytes()
    return artifact


__all__ = ["RichAlignShadowArtifact", "build_rich_align_shadow_artifact"]
