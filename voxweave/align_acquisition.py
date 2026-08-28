"""Strict fresh-alignment capture, physical transforms, and private terminals.

This module sits after the selected-legacy helper chain.  Backend result nodes
are observed opaquely before distribution, then captured all-or-none into the
closed P6 value domain.  Absolute authority is derived from the physical audio
origin; it never borrows the selected legacy route origin.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from voxweave.align_context import (
    IssuedAlignContext,
    _private_context_subject,
    role_vector,
)
from voxweave.align_distribution import StrictFailureLocator
from voxweave.align_failures import CanonicalFailure, SecondaryFailure
from voxweave.align_snapshot import (
    FROZEN_NULL,
    FrozenArray,
    FrozenJSON,
    freeze_json,
    frozen_json_digest,
)


Provenance = Literal["aligner", "align-interpolated"]


@dataclass(frozen=True)
class StrictCapturedUnit:
    unit_id: str
    call_index: int
    call_unit_index: int
    surface: str
    relative_start: float | None
    relative_end: float | None
    provenance: Provenance
    original_relative_start: float | None
    original_relative_end: float | None
    raw: FrozenJSON


@dataclass(frozen=True)
class StrictCaptureResult:
    call_index: int
    status: Literal["valid", "invalid"]
    units: tuple[StrictCapturedUnit, ...] | None
    raw_units_digest: str | None
    normalized_relative_digest: str | None
    failure: StrictFailureLocator | None
    observed_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class FreshUnit:
    unit_id: str
    call_index: int
    call_unit_index: int
    surface: str
    relative_start: float | None
    relative_end: float | None
    physical_origin_seconds: float
    start: float | None
    end: float | None
    provenance: Provenance
    original_relative_start: float | None
    original_relative_end: float | None
    raw: FrozenJSON


@dataclass(frozen=True)
class AuthorityTransformResult:
    call_index: int
    status: Literal["valid", "invalid"]
    capture: StrictCaptureResult
    units: tuple[FreshUnit, ...] | None
    authority_absolute_digest: str | None
    failure: StrictFailureLocator | None


@dataclass(frozen=True)
class QwenSampleGeometry:
    sample_start: int
    sample_end: int
    sample_rate: int
    sample_count: int
    physical_origin_seconds: float
    legacy_origin_seconds: float
    legacy_origin_kind: Literal["nominal-route"]
    authority_origin_seconds: float


class SampleGeometryError(ValueError):
    """Qwen route bounds cannot name one valid physical call."""

    def __init__(self, detail_code: str, message: str):
        super().__init__(message)
        self.detail_code = detail_code


def _exact_index(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be an exact nonnegative integer")
    return value


def _exact_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) is not float:
        raise TypeError("relative bounds must be exact floats or null")
    return value


def _same_optional_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.hex() == right.hex()


def _strict_node(
    node: Any,
    *,
    original: Any | None,
    unit_id: str,
    call_index: int,
    call_unit_index: int,
) -> StrictCapturedUnit:
    if not isinstance(node, Mapping):
        raise TypeError("raw alignment unit must be an object")
    if "text" not in node or "start" not in node or "end" not in node:
        raise TypeError("raw alignment unit is missing a required member")
    surface = node["text"]
    if type(surface) is not str:
        raise TypeError("raw alignment unit text must be an exact string")
    relative_start = _exact_optional_float(node["start"])
    relative_end = _exact_optional_float(node["end"])
    raw = freeze_json(node)

    if original is None:
        original_start = relative_start
        original_end = relative_end
        provenance: Provenance = "aligner"
    else:
        if not isinstance(original, Mapping):
            raise TypeError("original alignment unit must be an object")
        if "start" not in original or "end" not in original:
            raise TypeError("original alignment unit is missing a bound")
        original_start = _exact_optional_float(original["start"])
        original_end = _exact_optional_float(original["end"])
        provenance = (
            "aligner"
            if _same_optional_float(relative_start, original_start)
            and _same_optional_float(relative_end, original_end)
            else "align-interpolated"
        )
    return StrictCapturedUnit(
        unit_id,
        call_index,
        call_unit_index,
        surface,
        relative_start,
        relative_end,
        provenance,
        original_start,
        original_end,
        raw,
    )


def _optional_json_float(value: float | None) -> FrozenJSON:
    return FROZEN_NULL if value is None else freeze_json(value)


def _relative_unit_value(unit: StrictCapturedUnit) -> FrozenArray:
    """The displayed §5.3 fields, in their displayed order."""
    return FrozenArray(
        (
            freeze_json(unit.unit_id),
            freeze_json(unit.call_index),
            freeze_json(unit.call_unit_index),
            freeze_json(unit.surface),
            _optional_json_float(unit.relative_start),
            _optional_json_float(unit.relative_end),
            freeze_json(unit.provenance),
            _optional_json_float(unit.original_relative_start),
            _optional_json_float(unit.original_relative_end),
            unit.raw,
        )
    )


def capture_strict_units(
    raw_units: Sequence[Any],
    *,
    call_index: int,
    raw_unit_ids: Sequence[str],
    original_units: Sequence[Any] | None = None,
) -> StrictCaptureResult:
    """Capture every outer result node or expose no typed member at all."""
    call = _exact_index(call_index, name="call_index")
    try:
        raw_count = len(raw_units)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise TypeError("raw_units must be a sized sequence") from exc
    observed_ids = tuple(raw_unit_ids)
    if len(observed_ids) != raw_count or any(
        type(unit_id) is not str or not unit_id for unit_id in observed_ids
    ):
        raise ValueError("raw_unit_ids must assign one exact string to every raw node")
    if original_units is not None and len(original_units) != raw_count:
        raise ValueError("original_units must match the current raw result length")

    captured: list[StrictCapturedUnit] = []
    raw_values: list[FrozenJSON] = []
    for index in range(raw_count):
        try:
            unit = _strict_node(
                raw_units[index],
                original=original_units[index] if original_units is not None else None,
                unit_id=observed_ids[index],
                call_index=call,
                call_unit_index=index,
            )
        except Exception:  # noqa: BLE001 - all raw-domain defects share one locator
            return StrictCaptureResult(
                call,
                "invalid",
                None,
                None,
                None,
                StrictFailureLocator("strict-capture", index, "strict-raw-node"),
                observed_ids,
            )
        captured.append(unit)
        raw_values.append(unit.raw)
    units = tuple(captured)
    return StrictCaptureResult(
        call,
        "valid",
        units,
        frozen_json_digest(FrozenArray(tuple(raw_values))),
        frozen_json_digest(FrozenArray(tuple(_relative_unit_value(u) for u in units))),
        None,
        observed_ids,
    )


def _absolute_unit_value(unit: FreshUnit) -> FrozenArray:
    return FrozenArray(
        (
            freeze_json(unit.unit_id),
            freeze_json(unit.call_index),
            freeze_json(unit.call_unit_index),
            freeze_json(unit.surface),
            _optional_json_float(unit.relative_start),
            _optional_json_float(unit.relative_end),
            freeze_json(unit.physical_origin_seconds),
            _optional_json_float(unit.start),
            _optional_json_float(unit.end),
            freeze_json(unit.provenance),
            _optional_json_float(unit.original_relative_start),
            _optional_json_float(unit.original_relative_end),
            unit.raw,
        )
    )


def transform_strict_units(
    capture: StrictCaptureResult,
    *,
    physical_origin_seconds: float,
    identity: bool,
    retained_unit_count: int | None = None,
) -> AuthorityTransformResult:
    """Project complete relative capture through the sealed physical origin."""
    if not isinstance(capture, StrictCaptureResult):
        raise TypeError("capture must be a StrictCaptureResult")
    if type(identity) is not bool:
        raise TypeError("identity must be an exact bool")
    if type(physical_origin_seconds) is not float or not math.isfinite(
        physical_origin_seconds
    ):
        raise SampleGeometryError(
            "sample-geometry", "physical origin must be one finite exact float"
        )
    if retained_unit_count is None:
        retained_count = len(capture.observed_unit_ids)
    else:
        retained_count = _exact_index(retained_unit_count, name="retained_unit_count")
        if retained_count > len(capture.observed_unit_ids):
            raise ValueError("retained_unit_count exceeds the captured result")
    if capture.status != "valid" or capture.units is None:
        return AuthorityTransformResult(
            capture.call_index, "invalid", capture, None, None, capture.failure
        )

    transformed: list[FreshUnit] = []
    for index, unit in enumerate(capture.units):
        relative_start = unit.relative_start
        relative_end = unit.relative_end
        valid = (
            relative_start is not None
            and relative_end is not None
            and math.isfinite(relative_start)
            and math.isfinite(relative_end)
            and relative_start <= relative_end
        )
        if not valid:
            detail = (
                "authority-recompute" if index < retained_count else "surplus-transform"
            )
            return AuthorityTransformResult(
                capture.call_index,
                "invalid",
                capture,
                None,
                None,
                StrictFailureLocator("authority-transform", index, detail),
            )
        assert relative_start is not None
        assert relative_end is not None
        if identity:
            start = relative_start
            end = relative_end
        else:
            start = relative_start + physical_origin_seconds
            end = relative_end + physical_origin_seconds
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            detail = (
                "authority-recompute" if index < retained_count else "surplus-transform"
            )
            return AuthorityTransformResult(
                capture.call_index,
                "invalid",
                capture,
                None,
                None,
                StrictFailureLocator("authority-transform", index, detail),
            )
        transformed.append(
            FreshUnit(
                unit.unit_id,
                unit.call_index,
                unit.call_unit_index,
                unit.surface,
                relative_start,
                relative_end,
                physical_origin_seconds,
                start,
                end,
                unit.provenance,
                unit.original_relative_start,
                unit.original_relative_end,
                unit.raw,
            )
        )
    units = tuple(transformed)
    return AuthorityTransformResult(
        capture.call_index,
        "valid",
        capture,
        units,
        frozen_json_digest(FrozenArray(tuple(_absolute_unit_value(u) for u in units))),
        None,
    )


def qwen_sample_geometry(
    *,
    nominal_start: float,
    nominal_end: float,
    sample_rate: int,
    sample_count: int,
) -> QwenSampleGeometry:
    """Reproduce HEAD's Qwen slice arithmetic and retain both time origins."""
    if type(nominal_start) is not float or type(nominal_end) is not float:
        raise SampleGeometryError(
            "sample-geometry", "Qwen nominal bounds must be exact floats"
        )
    if not math.isfinite(nominal_start) or not math.isfinite(nominal_end):
        raise SampleGeometryError(
            "sample-geometry", "Qwen nominal bounds must be finite"
        )
    if type(sample_rate) is not int or sample_rate <= 0:
        raise SampleGeometryError(
            "sample-geometry", "sample rate must be a positive exact integer"
        )
    if type(sample_count) is not int or sample_count < 0:
        raise SampleGeometryError(
            "sample-geometry", "sample count must be an exact nonnegative integer"
        )
    try:
        sample_start = min(max(0, int(nominal_start * sample_rate)), sample_count)
        sample_end = min(sample_count, int(nominal_end * sample_rate))
    except (
        OverflowError,
        ValueError,
    ) as exc:  # pragma: no cover - finite checked above
        raise SampleGeometryError(
            "sample-geometry", "invalid Qwen sample bounds"
        ) from exc
    if sample_end < sample_start or sample_end < 0:
        raise SampleGeometryError(
            "sample-geometry", "Qwen sample interval is reversed or negative"
        )
    physical_origin = sample_start / sample_rate
    return QwenSampleGeometry(
        sample_start,
        sample_end,
        sample_rate,
        sample_count,
        physical_origin,
        nominal_start,
        "nominal-route",
        physical_origin,
    )


@dataclass(frozen=True)
class AcquisitionAdmissionEvent:
    terminal: Literal["acquisition-failed"]
    subject: tuple[str, str]
    payload: tuple[str, str, str, str, int, tuple[()]]


@dataclass
class AcquisitionAdmissionLedger:
    events: list[AcquisitionAdmissionEvent] = field(default_factory=list)
    _terminal_contexts: set[int] = field(default_factory=set, init=False, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )


class FreshSealBroken(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


def _distribution_seal_failure(
    secondary: tuple[SecondaryFailure, ...] = (),
) -> CanonicalFailure:
    return CanonicalFailure(
        "fresh-seal-broken",
        "authority-distribution",
        "distribution-seal",
        secondary,
    )


def raise_distribution_seal_mismatch(
    context: IssuedAlignContext,
    *,
    terminal_call_index: int,
    ledger: AcquisitionAdmissionLedger,
    dispose: Callable[[], None],
) -> None:
    """Append the sole private AO-13 terminal, dispose, and always raise."""
    call_index = _exact_index(terminal_call_index, name="terminal_call_index")
    if not isinstance(ledger, AcquisitionAdmissionLedger):
        raise TypeError("ledger must be an AcquisitionAdmissionLedger")
    if not callable(dispose):
        raise TypeError("dispose must be callable")
    context_key = id(context)
    with ledger._lock:
        if context_key in ledger._terminal_contexts:
            raise FreshSealBroken(_distribution_seal_failure())
        vector = role_vector(context)
        if not vector or vector[0] != "C":
            raise FreshSealBroken(
                CanonicalFailure(
                    "fresh-authority-invalid",
                    "authority-distribution",
                    "acquisition-unissued",
                )
            )
        subject = _private_context_subject(context)
        event = AcquisitionAdmissionEvent(
            "acquisition-failed",
            subject,
            (
                "AO-13",
                "fresh-seal-broken",
                "authority-distribution",
                "distribution-seal",
                call_index,
                (),
            ),
        )
        ledger.events.append(event)
        ledger._terminal_contexts.add(context_key)

    secondary: tuple[SecondaryFailure, ...] = ()
    try:
        dispose()
    except Exception:  # noqa: BLE001 - cleanup is a canonical secondary terminal
        secondary = (
            SecondaryFailure("snapshot-dispose-failed", "dispose", "stage-residue"),
        )
    raise FreshSealBroken(_distribution_seal_failure(secondary))


__all__ = [
    "AcquisitionAdmissionEvent",
    "AcquisitionAdmissionLedger",
    "AuthorityTransformResult",
    "FreshSealBroken",
    "FreshUnit",
    "QwenSampleGeometry",
    "SampleGeometryError",
    "StrictCaptureResult",
    "StrictCapturedUnit",
    "capture_strict_units",
    "qwen_sample_geometry",
    "raise_distribution_seal_mismatch",
    "transform_strict_units",
]
