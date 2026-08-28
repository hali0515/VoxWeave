"""Strict P6 policy, profile, and finalizer-evidence input projections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from voxweave.config import gap_thresholds
from voxweave.align_failures import CanonicalFailure
from voxweave.core.boundary_lattice import preflight_profile
from voxweave.core.layout import default_max_line_length, default_max_lines
from voxweave.core.segdoc import THRESHOLD_KEYS, DisplayProfile
from voxweave.core.smart_split import SplitThresholds
from voxweave.engine_registry import canonical_registry_iso


@dataclass(frozen=True)
class LegacyAlignPolicy:
    min_cue_sec: float
    tiny_cue_sec: float
    tiny_cue_target: float


@dataclass(frozen=True)
class V2PolicyStatus:
    kind: Literal["valid", "invalid"]
    detail_code: Literal["nonfinite-policy", "negative-policy"] | None


ProfileSource = Literal[
    "language-override",
    "unsupported-manifest",
    "profile-absent",
    "stored-profile",
    "manifest-absent",
]


@dataclass(frozen=True)
class ProfileStatus:
    kind: Literal["valid", "invalid"]
    source: ProfileSource
    detail_code: (
        Literal[
            "profile-shape",
            "profile-language",
            "profile-domain",
            "resolved-default-domain",
        ]
        | None
    )


@dataclass(frozen=True)
class ProfileResolution:
    profile: DisplayProfile | None
    status: ProfileStatus


@dataclass(frozen=True)
class EvidenceStatus:
    kind: Literal["valid", "invalid"]
    detail_code: Literal["shot-shape", "sing-shape", "evidence-domain"] | None

    @property
    def failure(self) -> CanonicalFailure | None:
        if self.kind != "invalid" or self.detail_code is None:
            return None
        return CanonicalFailure(
            "evidence-invalid", "finalizer-evidence", self.detail_code
        )


@dataclass(frozen=True)
class FinalizerEvidenceResolution:
    shots: tuple[float, ...] | None
    sing_spans: tuple[tuple[float, float], ...] | None
    status: EvidenceStatus


def validate_v2_policy(policy: LegacyAlignPolicy) -> V2PolicyStatus:
    if not isinstance(policy, LegacyAlignPolicy):
        raise TypeError("policy must be a LegacyAlignPolicy")
    values = (policy.min_cue_sec, policy.tiny_cue_sec, policy.tiny_cue_target)
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        return V2PolicyStatus("invalid", "nonfinite-policy")
    if any(value < 0.0 for value in values):
        return V2PolicyStatus("invalid", "negative-policy")
    return V2PolicyStatus("valid", None)


def _defaults(iso: str, source: ProfileSource) -> ProfileResolution:
    thresholds = SplitThresholds.from_mapping(gap_thresholds(iso))
    values = {key: getattr(thresholds, key) for key in THRESHOLD_KEYS}
    profile = DisplayProfile.from_resolved(
        iso,
        values,
        max_line_length=default_max_line_length(iso),
        max_lines=default_max_lines(iso),
    )
    finite = all(
        not isinstance(getattr(profile, key), bool)
        and math.isfinite(float(getattr(profile, key)))
        for key in THRESHOLD_KEYS
    )
    if not finite or preflight_profile(profile):
        return ProfileResolution(
            None, ProfileStatus("invalid", source, "resolved-default-domain")
        )
    return ProfileResolution(profile, ProfileStatus("valid", source, None))


def _invalid(source: ProfileSource, detail: str) -> ProfileResolution:
    allowed = {
        "profile-shape",
        "profile-language",
        "profile-domain",
        "resolved-default-domain",
    }
    if detail not in allowed:  # pragma: no cover - internal exhaustiveness
        raise ValueError("unknown profile detail")
    return ProfileResolution(None, ProfileStatus("invalid", source, detail))  # type: ignore[arg-type]


def resolve_align_profile(
    segmentation: Mapping[str, Any] | None,
    *,
    effective_iso: str,
    stored_iso: str | None = None,
) -> ProfileResolution:
    """Apply the closed §7.1 source order without mutating the carrier."""
    iso = canonical_registry_iso(effective_iso)
    if iso is None:
        return _invalid("manifest-absent", "profile-language")
    stored = canonical_registry_iso(stored_iso) if stored_iso is not None else None
    if stored is not None and stored != iso:
        return _defaults(iso, "language-override")
    if not isinstance(segmentation, Mapping):
        return _defaults(iso, "manifest-absent")
    if (
        type(segmentation.get("manifest_version")) is not int
        or segmentation.get("manifest_version") != 1
        or segmentation.get("engine") not in ("legacy-v1", "boundary-optimizer-v2")
    ):
        return _defaults(iso, "unsupported-manifest")
    if "profile" not in segmentation:
        return _defaults(iso, "profile-absent")
    source: ProfileSource = "stored-profile"
    language = canonical_registry_iso(segmentation.get("language"))
    if language != iso:
        return _invalid(source, "profile-language")
    raw = segmentation["profile"]
    if not isinstance(raw, Mapping):
        return _invalid(source, "profile-shape")
    expected = ("max_line_length", "max_lines", *THRESHOLD_KEYS)
    if tuple(raw) != expected and set(raw) != set(expected):
        return _invalid(source, "profile-shape")
    if len(raw) != len(expected):
        return _invalid(source, "profile-shape")
    max_line_length = raw.get("max_line_length")
    max_lines = raw.get("max_lines")
    if (
        type(max_line_length) is not int
        or type(max_lines) is not int
        or max_line_length <= 0
        or max_lines <= 0
    ):
        return _invalid(source, "profile-domain")
    thresholds: dict[str, float | int] = {}
    for key in THRESHOLD_KEYS:
        value = raw.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return _invalid(source, "profile-domain")
        thresholds[key] = value
    profile = DisplayProfile.from_resolved(
        iso,
        thresholds,
        max_line_length=max_line_length,
        max_lines=max_lines,
    )
    if preflight_profile(profile):
        return _invalid(source, "profile-domain")
    return ProfileResolution(profile, ProfileStatus("valid", source, None))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        projected = float(value)
    except (OverflowError, ValueError):
        return None
    return projected


def _invalid_evidence(
    detail_code: Literal["shot-shape", "sing-shape", "evidence-domain"],
) -> FinalizerEvidenceResolution:
    return FinalizerEvidenceResolution(
        None,
        None,
        EvidenceStatus("invalid", detail_code),
    )


def resolve_finalize_evidence(
    *,
    shot_changes: object,
    sing_spans: object,
) -> FinalizerEvidenceResolution:
    """Strictly project only sorted shots and sing spans."""
    shots: list[float] = []
    spans: list[tuple[float, float]] = []
    if shot_changes is not None:
        if isinstance(shot_changes, (str, bytes)) or not isinstance(
            shot_changes, Sequence
        ):
            return _invalid_evidence("shot-shape")
        try:
            shot_values = tuple(shot_changes)
        except (TypeError, ValueError):
            return _invalid_evidence("shot-shape")
        for raw in shot_values:
            value = _number(raw)
            if value is None:
                return _invalid_evidence("shot-shape")
            if not math.isfinite(value) or value < 0.0:
                return _invalid_evidence("evidence-domain")
            shots.append(value)
    if sing_spans is not None:
        if isinstance(sing_spans, (str, bytes)) or not isinstance(sing_spans, Sequence):
            return _invalid_evidence("sing-shape")
        try:
            span_values = tuple(sing_spans)
        except (TypeError, ValueError):
            return _invalid_evidence("sing-shape")
        for raw in span_values:
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                return _invalid_evidence("sing-shape")
            try:
                if len(raw) != 2:
                    return _invalid_evidence("sing-shape")
                start, end = _number(raw[0]), _number(raw[1])
            except (IndexError, TypeError, ValueError):
                return _invalid_evidence("sing-shape")
            if start is None or end is None:
                return _invalid_evidence("sing-shape")
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0.0
                or end < start
            ):
                return _invalid_evidence("evidence-domain")
            spans.append((start, end))
    return FinalizerEvidenceResolution(
        tuple(sorted(shots)),
        tuple(sorted(spans)),
        EvidenceStatus("valid", None),
    )


__all__ = [
    "EvidenceStatus",
    "FinalizerEvidenceResolution",
    "LegacyAlignPolicy",
    "ProfileResolution",
    "ProfileSource",
    "ProfileStatus",
    "V2PolicyStatus",
    "resolve_align_profile",
    "resolve_finalize_evidence",
    "validate_v2_policy",
]
