"""Independent stdlib-only fallback for failed rich align observation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from voxweave.align_failures import CanonicalFailure, SecondaryFailure


@dataclass(frozen=True)
class MinimalAlignShadowFailureArtifact:
    schema_version: Literal[2]
    artifact_kind: Literal["minimal-failure"]
    status: Literal["invalid"]
    failure: CanonicalFailure
    context_content_digest: str
    receipt_digest: str | None
    selected: Mapping[str, str]

    def to_canonical_bytes(self) -> bytes:
        if (
            self.schema_version != 2
            or self.artifact_kind != "minimal-failure"
            or self.status != "invalid"
            or not isinstance(self.failure, CanonicalFailure)
        ):
            raise ValueError("minimal shadow discriminator or failure is invalid")
        _sha256(self.context_content_digest)
        if self.receipt_digest is not None:
            _sha256(self.receipt_digest)
        if not isinstance(self.selected, Mapping) or tuple(self.selected) != (
            "engine_family",
            "vtt_sha256",
            "json_sha256",
            "evidence_sha256",
        ):
            raise ValueError("minimal shadow selected projection is not closed")
        if self.selected["engine_family"] not in ("legacy-v1", "boundary-v2"):
            raise ValueError("minimal shadow selected family is invalid")
        for key in ("vtt_sha256", "json_sha256", "evidence_sha256"):
            _sha256(self.selected[key])
        value = {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "failure": self.failure.to_dict(),
            "context_content_digest": self.context_content_digest,
            "receipt_digest": self.receipt_digest,
            "selected": dict(self.selected),
        }
        return (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("minimal shadow hash is not lowercase SHA-256")
    return value


def build_minimal_align_shadow_failure_artifact(
    *,
    context_content_digest: str,
    receipt_digest: str | None,
    engine_family: str,
    vtt_sha256: str,
    json_sha256: str,
    evidence_sha256: str,
    prior_failure: CanonicalFailure | None,
) -> MinimalAlignShadowFailureArtifact:
    """Build the closed scalar fallback without importing rich observation code."""
    if engine_family not in ("legacy-v1", "boundary-v2"):
        raise ValueError("minimal shadow selected family is invalid")
    if prior_failure is not None and not isinstance(prior_failure, CanonicalFailure):
        raise TypeError("minimal shadow prior failure is not canonical")
    secondary = SecondaryFailure(
        "shadow-internal-error",
        "rich-artifact",
        "rich-artifact-construction",
    )
    failure = (
        CanonicalFailure(
            "shadow-internal-error",
            "rich-artifact",
            "rich-artifact-construction",
        )
        if prior_failure is None
        else CanonicalFailure(
            prior_failure.kind,
            prior_failure.phase,
            prior_failure.detail_code,
            prior_failure.secondary + (secondary,),
        )
    )
    selected = MappingProxyType(
        {
            "engine_family": str(engine_family),
            "vtt_sha256": _sha256(vtt_sha256),
            "json_sha256": _sha256(json_sha256),
            "evidence_sha256": _sha256(evidence_sha256),
        }
    )
    artifact = MinimalAlignShadowFailureArtifact(
        2,
        "minimal-failure",
        "invalid",
        failure,
        _sha256(context_content_digest),
        None if receipt_digest is None else _sha256(receipt_digest),
        selected,
    )
    artifact.to_canonical_bytes()
    return artifact


__all__ = [
    "MinimalAlignShadowFailureArtifact",
    "build_minimal_align_shadow_failure_artifact",
]
