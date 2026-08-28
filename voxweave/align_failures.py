"""Closed P6 failure and seed-reason vocabularies.

These values classify evidence and harness outcomes; unchanged public
operations still propagate their historical exception class and message.  The
registry is dependency-light so every later P6 layer can validate a canonical
failure without importing pipeline, renderers, or finalizer code.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

OUTCOME_KIND_ORDER = (
    "subtitle-snapshot-failed",
    "align-input-decode-invalid",
    "v2-input-invalid",
    "carrier-extraction-failed",
    "media-identity-invalid",
    "media-snapshot-unavailable",
    "semantic-backend-unavailable",
    "qwen-route-invalid",
    "qwen-window-operation-failed",
    "dp-route-hints-invalid",
    "cache-lock-failed",
    "cache-companion-invalid",
    "cache-operation-failed",
    "context-authority-invalid",
    "legacy-time-transform-failed",
    "fresh-backend-output-invalid",
    "fresh-time-transform-invalid",
    "fresh-distribution-invalid",
    "fresh-reconciliation-invalid",
    "fresh-seed-invalid",
    "v2-policy-invalid",
    "fresh-authority-invalid",
    "fresh-seal-broken",
    "profile-invalid",
    "evidence-invalid",
    "segmentation-v2-invalid",
    "finalizer-budget-exhausted",
    "finalizer-output-invalid",
    "finalizer-partition-failed",
    "finalizer-trace-failed",
    "finalizer-stability-failed",
    "align-delta-invalid",
    "no-aligned-units",
    "selected-render-invalid",
    "final-evidence-invalid",
    "shadow-internal-error",
    "shadow-artifact-unavailable",
    "preencode-failed",
    "stage-failed",
    "episode-lock-failed",
    "input-stale",
    "media-stale",
    "commit-failed",
    "artifact-cleanup-failed",
    "model-release-failed",
    "observer-failed",
    "snapshot-dispose-failed",
)

_OUTCOME_DETAILS = {
    "subtitle-snapshot-failed": ("vtt-read", "sibling-read"),
    "align-input-decode-invalid": (
        "vtt-encoding",
        "vtt-format-mismatch",
        "vtt-no-cues",
        "sibling-json-encoding",
        "sibling-json-syntax",
        "sibling-top-level-shape",
    ),
    "v2-input-invalid": (
        "sibling-json-duplicate-key",
        "sibling-json-nonfinite",
        "strict-carrier-domain",
    ),
    "carrier-extraction-failed": (
        "unsupported-selected-json-node",
        "non-string-selected-object-key",
        "selected-json-cycle",
    ),
    "media-identity-invalid": (
        "media-not-found",
        "media-fingerprint",
        "media-logical-id",
    ),
    "media-snapshot-unavailable": (
        "reflink-and-copy-failed",
        "snapshot-verification-failed",
    ),
    "semantic-backend-unavailable": ("endpoint-not-configured",),
    "qwen-route-invalid": ("no-route-source", "all-crops-none"),
    "qwen-window-operation-failed": (
        "route-bound-access",
        "route-bound-arithmetic",
        "sample-start-index",
        "sample-end-index",
        "sample-open",
        "sample-seek-read",
        "sample-temp-create",
        "sample-write",
    ),
    "dp-route-hints-invalid": (
        "hint-shape",
        "hint-nonfinite",
        "hint-nonmonotone",
        "plan-nontiling",
        "crop-geometry",
        "crop-over-budget",
    ),
    "cache-lock-failed": ("cache-lock-acquire",),
    "cache-companion-invalid": (
        "companion-schema",
        "companion-media",
        "companion-size",
        "companion-hash",
    ),
    "cache-operation-failed": (
        "cache-decode",
        "cache-stage",
        "cache-replace",
        "companion-unlink",
        "companion-replace",
    ),
    "context-authority-invalid": (
        "context-unissued",
        "context-role",
        "context-binding",
        "context-consumed",
        "context-unused-role",
        "expected-vtt-generation",
        "allocator-limit-profile",
    ),
    "legacy-time-transform-failed": (
        "retained-unit-text",
        "retained-unit-start",
        "retained-unit-end",
        "retained-unit-operand",
    ),
    "fresh-backend-output-invalid": (
        "backend-call-shape",
        "backend-raised",
        "relative-normalization",
    ),
    "fresh-time-transform-invalid": (
        "strict-raw-node",
        "sample-geometry",
        "physical-origin-mismatch",
        "authority-recompute",
        "surplus-transform",
    ),
    "fresh-distribution-invalid": (
        "route-owner-mismatch",
        "partial-empty-ownership",
        "punctuation-only-block",
        "allocation-no-tiling",
        "allocation-ambiguous",
        "allocation-budget",
    ),
    "fresh-reconciliation-invalid": ("footprint-reconciliation",),
    "fresh-seed-invalid": ("seed-admission",),
    "v2-policy-invalid": ("nonfinite-policy", "negative-policy"),
    "fresh-authority-invalid": (
        "acquisition-unissued",
        "receipt-context",
        "root-transfer",
        "w1-root-event",
    ),
    "fresh-seal-broken": (
        "context-seal",
        "raw-seal",
        "relative-seal",
        "legacy-slice-seal",
        "authority-seal",
        "distribution-seal",
        "phase1-seal",
    ),
    "profile-invalid": (
        "profile-shape",
        "profile-language",
        "profile-domain",
        "resolved-default-domain",
    ),
    "evidence-invalid": ("shot-shape", "sing-shape", "evidence-domain"),
    "segmentation-v2-invalid": (
        "semantic-selector-unmodelled",
        "named-multi-unattributed",
        "p7-prerequisite-open",
        "delivery-unit-range",
    ),
    "finalizer-budget-exhausted": ("sweep-budget",),
    "finalizer-output-invalid": (
        "terminal-validity",
        "cue-schema",
        "footprint-fallback",
        "phase1-fidelity",
    ),
    "finalizer-partition-failed": ("finalizer-partition",),
    "finalizer-trace-failed": ("trace-replay",),
    "finalizer-stability-failed": ("stability-check",),
    "align-delta-invalid": (
        "registry-digest",
        "trigger-set",
        "primitive-relation",
    ),
    "no-aligned-units": ("all-block-units-empty",),
    "selected-render-invalid": (
        "selected-candidate-missing",
        "cue-source-map",
        "unit-coverage",
        "vtt-projection",
        "json-projection",
        "derived-hash",
        "candidate-family-manifest",
    ),
    "final-evidence-invalid": (
        "evidence-binding",
        "selected-hash-link",
        "closed-schema",
        "independent-projection",
    ),
    "shadow-internal-error": (
        "profile-stage",
        "evidence-stage",
        "w1-stage",
        "comparator-stage",
        "renderer-stage",
        "rich-artifact-construction",
    ),
    "shadow-artifact-unavailable": ("minimal-artifact-construction",),
    "preencode-failed": (
        "main-json-encode",
        "vtt-encode",
        "evidence-encode",
    ),
    "stage-failed": (
        "main-json-stage",
        "vtt-stage",
        "evidence-stage",
        "machine-artifact-stage",
    ),
    "episode-lock-failed": ("episode-lock-acquire",),
    "input-stale": (
        "vtt-generation",
        "sibling-generation",
        "speaker-mapping-generation",
        "correct-generation",
        "process-output-generation",
    ),
    "media-stale": ("media-generation", "pair-decision"),
    "commit-failed": (
        "main-json-replace",
        "vtt-replace",
        "evidence-replace",
        "machine-artifact-replace",
    ),
    "artifact-cleanup-failed": (
        "voiceprints-unlink",
        "suggest-unlink",
        "html-unlink",
        "evidence-unlink",
    ),
    "model-release-failed": ("panns-release",),
    "observer-failed": ("observer-callback",),
    "snapshot-dispose-failed": (
        "media-snapshot-residue",
        "audio-temp-residue",
        "stage-residue",
    ),
}

if tuple(_OUTCOME_DETAILS) != OUTCOME_KIND_ORDER:  # pragma: no cover - import invariant
    raise RuntimeError(
        "P6 outcome detail registry order does not match its kind registry"
    )

OUTCOME_DETAILS = MappingProxyType(_OUTCOME_DETAILS)

AUTHORITY_REASON_ORDER = (
    "partial-empty-ownership",
    "punctuation-only-block",
    "authority-transform-invalid",
    "route-owner-mismatch",
    "allocation-no-tiling",
    "allocation-ambiguous",
    "allocation-budget-exhausted",
)
SEED_REASON_ORDER = AUTHORITY_REASON_ORDER + (
    "absolute-bound-invalid",
    "absolute-order-invalid",
    "display-seed-invalid",
    "footprint-reconciliation",
)

RATIFICATION_DORMANT_DETAILS: tuple[tuple[str, str, str], ...] = ()


class FailureRegistryError(ValueError):
    """A canonical failure named an unregistered parent/detail pair."""


def _validate(kind: str, detail_code: str) -> None:
    details = OUTCOME_DETAILS.get(kind)
    if details is None or detail_code not in details:
        raise FailureRegistryError(
            f"unregistered P6 failure pair {kind!r}/{detail_code!r}"
        )


@dataclass(frozen=True)
class SecondaryFailure:
    kind: str
    phase: str
    detail_code: str

    def __post_init__(self) -> None:
        _validate(self.kind, self.detail_code)
        if not self.phase:
            raise FailureRegistryError("failure phase must be nonempty")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "phase": self.phase,
            "detail_code": self.detail_code,
        }


@dataclass(frozen=True)
class CanonicalFailure:
    kind: str
    phase: str
    detail_code: str
    secondary: tuple[SecondaryFailure, ...] = ()

    def __post_init__(self) -> None:
        _validate(self.kind, self.detail_code)
        if not self.phase:
            raise FailureRegistryError("failure phase must be nonempty")
        if not isinstance(self.secondary, tuple) or not all(
            isinstance(item, SecondaryFailure) for item in self.secondary
        ):
            raise FailureRegistryError("secondary failures must be a flat tuple")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "phase": self.phase,
            "detail_code": self.detail_code,
            "secondary": [item.to_dict() for item in self.secondary],
        }


def is_detail_dormant(kind: str, detail_code: str) -> bool:
    return any(
        dormant_kind == kind and dormant_detail == detail_code
        for dormant_kind, dormant_detail, _decision in RATIFICATION_DORMANT_DETAILS
    )
