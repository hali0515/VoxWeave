"""Strict fresh-alignment capture, physical transforms, and private terminals.

This module sits after the selected-legacy helper chain.  Backend result nodes
are observed opaquely before distribution, then captured all-or-none into the
closed P6 value domain.  Absolute authority is derived from the physical audio
origin; it never borrows the selected legacy route origin.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn

from voxweave.align_context import (
    IssuedAlignContext,
    _align_context_authority_profile,
    _align_context_stable_fields,
    _private_context_subject,
    consume_context_role,
    role_vector,
)
from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityCallInput,
    AuthorityDistributionReceipt,
    AuthorityLimitProfile,
    AuthoritySkippedBlockInput,
    LegacyCallDistributionReceipt,
    RouteClaim,
    RouteExpectation,
    StrictFailureLocator,
    _build_context_authority_distribution,
    legacy_distribute_before_shift,
)
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


@dataclass(frozen=True)
class PhysicalCallReceipt:
    call_index: int
    source_block_indices: tuple[int, ...]
    audio_sample_start: int
    audio_sample_end: int
    sample_rate: int
    physical_origin_seconds: float
    legacy_origin_seconds: float
    legacy_origin_kind: Literal["identity", "sample-origin", "nominal-route"]
    authority_origin_seconds: float
    backend_model_config_digest: str
    route_input_digest: str
    strict_unit_status: Literal["valid", "invalid"]
    strict_failure: StrictFailureLocator | None
    raw_units_digest: str | None
    normalized_relative_digest: str | None
    legacy_slice_digest: str
    legacy_absolute_digest: str | None
    authority_transform_status: Literal["valid", "invalid"]
    authority_absolute_digest: str | None
    raw_unit_ids: tuple[str, ...]


class SampleGeometryError(ValueError):
    """Qwen route bounds cannot name one valid physical call."""

    def __init__(self, detail_code: str, message: str):
        super().__init__(message)
        self.detail_code = detail_code


def _attach_canonical_failure(
    exc: BaseException,
    *,
    kind: str,
    phase: str,
    detail_code: str,
) -> None:
    """Classify an unchanged historical exception boundary."""
    try:
        if not isinstance(getattr(exc, "failure", None), CanonicalFailure):
            setattr(exc, "failure", CanonicalFailure(kind, phase, detail_code))
    except Exception:
        pass


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
        classified = TypeError("raw_units must be a sized sequence")
        _attach_canonical_failure(
            classified,
            kind="fresh-backend-output-invalid",
            phase="backend-output",
            detail_code="backend-call-shape",
        )
        raise classified from exc
    observed_ids = tuple(raw_unit_ids)
    if len(observed_ids) != raw_count or any(
        type(unit_id) is not str or not unit_id for unit_id in observed_ids
    ):
        raise ValueError("raw_unit_ids must assign one exact string to every raw node")
    if original_units is not None and len(original_units) != raw_count:
        exc = ValueError("original_units must match the current raw result length")
        _attach_canonical_failure(
            exc,
            kind="fresh-backend-output-invalid",
            phase="relative-normalization",
            detail_code="relative-normalization",
        )
        raise exc

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


@dataclass(frozen=True, init=False)
class IssuedFreshAlignment:
    context_content_digest: str
    receipt_digest: str
    physical_calls: tuple[PhysicalCallReceipt, ...]
    distribution: AuthorityDistributionReceipt
    seed_status: Literal["valid", "invalid"]
    seed_reasons: tuple[str, ...]
    _binding: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("IssuedFreshAlignment is issuer-only")


@dataclass(frozen=True, init=False)
class VerifiedFreshAlignment:
    context_content_digest: str
    receipt_digest: str
    _binding: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("VerifiedFreshAlignment is issuer-only")


@dataclass(frozen=True, init=False)
class FreshAlignmentSession:
    context_content_digest: str
    _binding: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("FreshAlignmentSession is issuer-only")


@dataclass
class _ObservedPhysicalCall:
    call_index: int
    source_block_indices: tuple[int, ...]
    post_units: Sequence[Any]
    original_units: Sequence[Any] | None
    raw_unit_ids: tuple[str, ...]
    raw_node_range: tuple[int, int]
    audio_sample_start: int
    audio_sample_end: int
    sample_rate: int
    physical_origin_seconds: float
    legacy_origin_seconds: float
    legacy_origin_kind: Literal["identity", "sample-origin", "nominal-route"]
    authority_origin_seconds: float
    backend_model_config_digest: str
    route_input_digest: str
    geometry_failure: StrictFailureLocator | None


@dataclass(frozen=True)
class _AdapterPayload:
    legacy_delivery: object
    projection_inputs: object
    strict_input_status: object
    v2_policy_status: object
    profile_resolution: object
    evidence_resolution: object


@dataclass(frozen=True)
class _FreshSeals:
    context: str
    raw: str
    relative: str
    legacy_slice: str
    authority: str
    distribution: str
    phase1: str


@dataclass
class _FreshRecord:
    context: IssuedAlignContext
    issued: IssuedFreshAlignment
    public_snapshot: tuple[object, ...]
    blocks: tuple[AuthorityBlock, ...]
    captures: tuple[StrictCaptureResult, ...]
    transforms: tuple[AuthorityTransformResult, ...]
    distribution: AuthorityDistributionReceipt
    seed: object
    physical_calls: tuple[PhysicalCallReceipt, ...]
    legacy_receipts: tuple[LegacyCallDistributionReceipt, ...]
    legacy_block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    adapter_payload: _AdapterPayload | None = None
    verified: VerifiedFreshAlignment | None = None
    transfer_terminal: Literal["live", "consumed", "retired"] = "live"
    seals: _FreshSeals | None = None


@dataclass
class _VerifiedRecord:
    context: IssuedAlignContext
    acquisition: IssuedFreshAlignment
    verified: VerifiedFreshAlignment
    seed_snapshot: object
    profile_snapshot: object
    sing_spans_snapshot: tuple[tuple[float, float], ...]
    binding: str


@dataclass
class _SessionRecord:
    session: FreshAlignmentSession
    issuer: FreshAlignmentIssuer
    binding: str
    sealed: bool = False


_FRESH: dict[int, _FreshRecord] = {}
_VERIFIED_FRESH: dict[int, _VerifiedRecord] = {}
_FRESH_SESSIONS: dict[int, _SessionRecord] = {}
_FRESH_LOCK = threading.RLock()
_ISSUER_TOKEN = object()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _default_digest(label: str, value: str) -> str:
    return hashlib.sha256(f"{label}\0{value}".encode()).hexdigest()


def _stable_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _stable_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_stable_value(member) for member in value]
    if isinstance(value, list):
        return [_stable_value(member) for member in value]
    if isinstance(value, Mapping):
        return {str(key): _stable_value(member) for key, member in value.items()}
    return value


def _stable_digest(value: Any) -> str:
    return frozen_json_digest(freeze_json(_stable_value(value)))


def _physical_projection(
    calls: tuple[PhysicalCallReceipt, ...], names: tuple[str, ...]
) -> tuple[tuple[object, ...], ...]:
    return tuple(tuple(getattr(call, name) for name in names) for call in calls)


def _context_seal_digest(record: _FreshRecord) -> str:
    stable = _align_context_stable_fields(record.context)
    context = record.context
    return _stable_digest(
        (
            context.context_content_digest,
            context.context_binding_digest,
            context.engine_family,
            context.effective_iso,
            context.route_kind,
            context._issuance_nonce,
            record.issued.context_content_digest,
            stable,
        )
    )


def _raw_seal_digest(record: _FreshRecord) -> str:
    captures = tuple(
        (
            capture.call_index,
            capture.status,
            capture.raw_units_digest,
            capture.failure,
            capture.observed_unit_ids,
            None
            if capture.units is None
            else tuple((unit.unit_id, unit.raw) for unit in capture.units),
        )
        for capture in record.captures
    )
    names = (
        "call_index",
        "strict_unit_status",
        "strict_failure",
        "raw_units_digest",
        "raw_unit_ids",
    )
    return _stable_digest(
        (
            captures,
            _physical_projection(record.physical_calls, names),
            _physical_projection(record.issued.physical_calls, names),
        )
    )


def _relative_seal_digest(record: _FreshRecord) -> str:
    captures = tuple(
        (
            capture.call_index,
            capture.status,
            capture.normalized_relative_digest,
            None
            if capture.units is None
            else tuple(
                (
                    unit.unit_id,
                    unit.call_index,
                    unit.call_unit_index,
                    unit.surface,
                    unit.relative_start,
                    unit.relative_end,
                    unit.provenance,
                    unit.original_relative_start,
                    unit.original_relative_end,
                )
                for unit in capture.units
            ),
        )
        for capture in record.captures
    )
    names = ("call_index", "normalized_relative_digest")
    return _stable_digest(
        (
            captures,
            _physical_projection(record.physical_calls, names),
            _physical_projection(record.issued.physical_calls, names),
        )
    )


def _legacy_slice_seal_digest(record: _FreshRecord) -> str:
    names = ("call_index", "legacy_slice_digest", "legacy_absolute_digest")
    return _stable_digest(
        (
            record.legacy_receipts,
            record.legacy_block_units,
            _physical_projection(record.physical_calls, names),
            _physical_projection(record.issued.physical_calls, names),
        )
    )


def _authority_seal_digest(record: _FreshRecord) -> str:
    transforms = tuple(
        (
            transform.call_index,
            transform.status,
            transform.authority_absolute_digest,
            transform.failure,
            None
            if transform.units is None
            else tuple(
                (
                    unit.unit_id,
                    unit.call_index,
                    unit.call_unit_index,
                    unit.surface,
                    unit.relative_start,
                    unit.relative_end,
                    unit.physical_origin_seconds,
                    unit.start,
                    unit.end,
                    unit.provenance,
                    unit.original_relative_start,
                    unit.original_relative_end,
                )
                for unit in transform.units
            ),
        )
        for transform in record.transforms
    )
    names = (
        "call_index",
        "source_block_indices",
        "audio_sample_start",
        "audio_sample_end",
        "sample_rate",
        "physical_origin_seconds",
        "legacy_origin_seconds",
        "legacy_origin_kind",
        "authority_origin_seconds",
        "backend_model_config_digest",
        "route_input_digest",
        "authority_transform_status",
        "authority_absolute_digest",
    )
    return _stable_digest(
        (
            transforms,
            _physical_projection(record.physical_calls, names),
            _physical_projection(record.issued.physical_calls, names),
        )
    )


def _distribution_seal_digest(record: _FreshRecord) -> str:
    return _stable_digest((record.distribution, record.issued.distribution))


def _phase1_seal_digest(record: _FreshRecord) -> str:
    return _stable_digest(
        (
            record.seed,
            record.issued.seed_status,
            record.issued.seed_reasons,
        )
    )


def _fresh_seals(record: _FreshRecord) -> _FreshSeals:
    return _FreshSeals(
        _context_seal_digest(record),
        _raw_seal_digest(record),
        _relative_seal_digest(record),
        _legacy_slice_seal_digest(record),
        _authority_seal_digest(record),
        _distribution_seal_digest(record),
        _phase1_seal_digest(record),
    )


_SEAL_PHASES = {
    "context-seal": "context",
    "raw-seal": "strict-capture",
    "relative-seal": "relative-normalization",
    "legacy-slice-seal": "legacy-distribution",
    "authority-seal": "authority-transform",
    "distribution-seal": "authority-distribution",
    "phase1-seal": "w1-admission",
}


def _raise_component_seal(detail_code: str) -> NoReturn:
    raise FreshSealBroken(
        CanonicalFailure(
            "fresh-seal-broken",
            _SEAL_PHASES[detail_code],
            detail_code,
        )
    )


def _verify_fresh_seals(record: _FreshRecord) -> None:
    expected = record.seals
    if expected is None:
        _raise_component_seal("context-seal")
    checks = (
        ("context-seal", expected.context, _context_seal_digest),
        ("raw-seal", expected.raw, _raw_seal_digest),
        ("relative-seal", expected.relative, _relative_seal_digest),
        ("legacy-slice-seal", expected.legacy_slice, _legacy_slice_seal_digest),
        ("authority-seal", expected.authority, _authority_seal_digest),
        ("distribution-seal", expected.distribution, _distribution_seal_digest),
        ("phase1-seal", expected.phase1, _phase1_seal_digest),
    )
    for detail_code, sealed, projector in checks:
        try:
            current = projector(record)
        except Exception:
            _raise_component_seal(detail_code)
        if current != sealed:
            _raise_component_seal(detail_code)


def _invalid_capture(
    call_index: int,
    raw_unit_ids: tuple[str, ...],
    *,
    index: int = 0,
) -> StrictCaptureResult:
    return StrictCaptureResult(
        call_index,
        "invalid",
        None,
        None,
        None,
        StrictFailureLocator("strict-capture", index, "strict-raw-node"),
        raw_unit_ids,
    )


class FreshAlignmentIssuer:
    """Private AO-05 through AO-13 owner of live physical-call observations."""

    def __init__(
        self,
        token: object,
        context: IssuedAlignContext,
        *,
        alignment_texts: Sequence[str],
        source_indices: Sequence[int],
        language: str,
        prepared_audio_sample_count: int,
        sample_rate: int,
        backend_model_config_digest: str | None,
        route_input_digest: str | None,
        ledger: AcquisitionAdmissionLedger | None,
        verifier_cut_mutator: Callable[[tuple[int, ...]], tuple[int, ...]] | None,
    ) -> None:
        if token is not _ISSUER_TOKEN:
            raise TypeError("FreshAlignmentIssuer is private")
        if language != context.effective_iso:
            raise ValueError("fresh acquisition language is cross-context")
        if (
            type(prepared_audio_sample_count) is not int
            or prepared_audio_sample_count < 0
        ):
            raise ValueError("prepared audio sample count must be nonnegative")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample rate must be a positive exact integer")
        texts = tuple(alignment_texts)
        original_sources = tuple(source_indices)
        if (
            len(original_sources) != len(texts)
            or len(set(original_sources)) != len(original_sources)
            or any(type(index) is not int or index < 0 for index in original_sources)
        ):
            raise ValueError(
                "fresh acquisition source indices must be unique nonnegative integers"
            )
        consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
        self.context = context
        self.blocks = tuple(
            AuthorityBlock(source_index, str(text))
            for source_index, text in zip(original_sources, texts, strict=True)
        )
        self.language = language
        self.prepared_audio_sample_count = prepared_audio_sample_count
        self.sample_rate = sample_rate
        self.backend_model_config_digest = (
            backend_model_config_digest
            if _is_sha256(backend_model_config_digest)
            else _default_digest("backend-model", context.route_kind)
        )
        self.route_input_digest = (
            route_input_digest
            if _is_sha256(route_input_digest)
            else _default_digest("route-input", context.context_content_digest)
        )
        self.ledger = ledger or AcquisitionAdmissionLedger()
        self.verifier_cut_mutator = verifier_cut_mutator
        self.observed: list[_ObservedPhysicalCall] = []
        self.raw_cursor = 0
        self.sealed = False

    def _observe_physical_call(
        self,
        post_units: Sequence[Any],
        original_units: Sequence[Any] | None,
        source_indices: Sequence[int],
        legacy_origin_seconds: float,
        *,
        audio_sample_start: int | None = None,
        audio_sample_end: int | None = None,
        sample_rate: int | None = None,
        sample_count: int | None = None,
        nominal_end_seconds: float | None = None,
        backend_model_config_digest: str | None = None,
        route_input_digest: str | None = None,
    ) -> None:
        """AO-07 callback: retain identity/length only; recurse only at sealing."""
        if self.sealed:
            raise RuntimeError("fresh acquisition issuer is already sealed")
        try:
            raw_count = len(post_units)
        except Exception as exc:
            classified = TypeError("backend result must be one sized sequence")
            _attach_canonical_failure(
                classified,
                kind="fresh-backend-output-invalid",
                phase="backend-output",
                detail_code="backend-call-shape",
            )
            raise classified from exc
        if original_units is not None:
            try:
                len(original_units)
            except Exception as exc:
                classified = TypeError("original backend result must be sized")
                _attach_canonical_failure(
                    classified,
                    kind="fresh-backend-output-invalid",
                    phase="backend-output",
                    detail_code="backend-call-shape",
                )
                raise classified from exc
        source_positions = tuple(source_indices)
        if (
            not source_positions
            or any(
                type(position) is not int or not 0 <= position < len(self.blocks)
                for position in source_positions
            )
            or len(set(source_positions)) != len(source_positions)
        ):
            raise ValueError("physical call delivery positions are invalid")
        sources = tuple(
            self.blocks[position].source_index for position in source_positions
        )
        call_index = len(self.observed)
        raw_start = self.raw_cursor
        raw_ids = tuple(
            f"r{index}" for index in range(raw_start, raw_start + raw_count)
        )
        self.raw_cursor += raw_count

        rate = self.sample_rate if sample_rate is None else sample_rate
        count = (
            self.prepared_audio_sample_count if sample_count is None else sample_count
        )
        geometry_failure: StrictFailureLocator | None = None
        if type(legacy_origin_seconds) is not float or not math.isfinite(
            legacy_origin_seconds
        ):
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            legacy_origin = 0.0
        else:
            legacy_origin = legacy_origin_seconds
        if type(rate) is not int or rate <= 0 or type(count) is not int or count < 0:
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            rate = self.sample_rate
            count = self.prepared_audio_sample_count

        geometry_supplied = (
            audio_sample_start is not None
            and audio_sample_end is not None
            and sample_rate is not None
            and sample_count is not None
        )
        if geometry_supplied:
            start = audio_sample_start
            end = audio_sample_end
        elif self.context.route_kind != "qwen-crop" and legacy_origin == 0.0:
            start = 0
            end = count
        else:
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            start = 0
            end = 0
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= count
        ):
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            start = 0
            end = 0
        physical_origin = start / rate
        if self.context.route_kind == "qwen-crop":
            legacy_kind: Literal["identity", "sample-origin", "nominal-route"] = (
                "nominal-route"
            )
            if nominal_end_seconds is None:
                geometry_failure = StrictFailureLocator(
                    "sample-geometry", None, "sample-geometry"
                )
            else:
                try:
                    expected = qwen_sample_geometry(
                        nominal_start=legacy_origin,
                        nominal_end=nominal_end_seconds,
                        sample_rate=rate,
                        sample_count=count,
                    )
                except SampleGeometryError:
                    geometry_failure = StrictFailureLocator(
                        "sample-geometry", None, "sample-geometry"
                    )
                else:
                    if expected.sample_start != start or expected.sample_end != end:
                        geometry_failure = StrictFailureLocator(
                            "sample-geometry", None, "physical-origin-mismatch"
                        )
        elif start == 0 and legacy_origin == 0.0:
            legacy_kind = "identity"
        else:
            legacy_kind = "sample-origin"
            if physical_origin.hex() != legacy_origin.hex():
                geometry_failure = StrictFailureLocator(
                    "sample-geometry", None, "physical-origin-mismatch"
                )

        model_digest = backend_model_config_digest or self.backend_model_config_digest
        input_digest = route_input_digest or self.route_input_digest
        if not _is_sha256(model_digest) or not _is_sha256(input_digest):
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            model_digest = self.backend_model_config_digest
            input_digest = self.route_input_digest
        assert isinstance(model_digest, str)
        assert isinstance(input_digest, str)
        self.observed.append(
            _ObservedPhysicalCall(
                call_index,
                sources,
                post_units,
                original_units,
                raw_ids,
                (raw_start, raw_start + raw_count),
                start,
                end,
                rate,
                physical_origin,
                legacy_origin,
                legacy_kind,
                physical_origin,
                model_digest,
                input_digest,
                geometry_failure,
            )
        )

    def _invoke_backend_call(self, backend_call: Callable[[], Any]) -> Any:
        """Preserve a backend exception while attaching its closed classification."""
        if not callable(backend_call):
            raise TypeError("backend call must be callable")
        try:
            return backend_call()
        except Exception as exc:
            _attach_canonical_failure(
                exc,
                kind="fresh-backend-output-invalid",
                phase="backend-call",
                detail_code="backend-raised",
            )
            raise

    def _invoke_qwen_physical_call(
        self,
        backend_call: Callable[[], Sequence[Any]],
        source_position: int,
        legacy_origin_seconds: float,
        nominal_end_seconds: float,
        *,
        audio_sample_start: int | None,
        audio_sample_end: int | None,
        sample_rate: int | None,
        sample_count: int | None,
    ) -> Sequence[Any]:
        """Own one Qwen backend attempt and its inseparable AO-07 observation."""
        if self.context.route_kind != "qwen-crop":
            raise ValueError("Qwen invocation is unavailable for this route")
        if not callable(backend_call):
            raise TypeError("Qwen backend call must be callable")
        raw_units = self._invoke_backend_call(backend_call)
        self._observe_physical_call(
            raw_units,
            None,
            (source_position,),
            legacy_origin_seconds,
            audio_sample_start=audio_sample_start,
            audio_sample_end=audio_sample_end,
            sample_rate=sample_rate,
            sample_count=sample_count,
            nominal_end_seconds=nominal_end_seconds,
        )
        return raw_units

    def _dispose(self) -> None:
        self.observed.clear()


def begin_fresh_alignment(
    context: IssuedAlignContext,
    *,
    alignment_texts: Sequence[str],
    source_indices: Sequence[int],
    language: str,
    prepared_audio_sample_count: int,
    sample_rate: int = 16_000,
    backend_model_config_digest: str | None = None,
    route_input_digest: str | None = None,
    ledger: AcquisitionAdmissionLedger | None = None,
    _verifier_cut_mutator: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
) -> FreshAlignmentSession:
    """Issue an opaque session; the private backend seam owns its observations."""
    issuer = FreshAlignmentIssuer(
        _ISSUER_TOKEN,
        context,
        alignment_texts=alignment_texts,
        source_indices=source_indices,
        language=language,
        prepared_audio_sample_count=prepared_audio_sample_count,
        sample_rate=sample_rate,
        backend_model_config_digest=backend_model_config_digest,
        route_input_digest=route_input_digest,
        ledger=ledger,
        verifier_cut_mutator=_verifier_cut_mutator,
    )
    session = object.__new__(FreshAlignmentSession)
    binding = secrets.token_hex(32)
    object.__setattr__(
        session, "context_content_digest", context.context_content_digest
    )
    object.__setattr__(session, "_binding", binding)
    with _FRESH_LOCK:
        _FRESH_SESSIONS[id(session)] = _SessionRecord(session, issuer, binding)
    return session


def _fresh_session_record(session: FreshAlignmentSession) -> _SessionRecord:
    with _FRESH_LOCK:
        record = _FRESH_SESSIONS.get(id(session))
        if (
            record is None
            or record.session is not session
            or session._binding != record.binding
            or session.context_content_digest
            != record.issuer.context.context_content_digest
            or record.sealed
        ):
            raise ValueError(
                "fresh alignment session is unissued, changed, or consumed"
            )
        return record


def _fresh_alignment_call_observer(
    session: FreshAlignmentSession,
) -> Callable[..., None]:
    """Return the issuer-bound callback used only inside backend physical seams."""
    issuer = _fresh_session_record(session).issuer
    return issuer._observe_physical_call


def _fresh_alignment_qwen_invoker(
    session: FreshAlignmentSession,
) -> Callable[..., Sequence[Any]]:
    """Return the private issuer-owned Qwen physical invocation seam."""
    issuer = _fresh_session_record(session).issuer
    return issuer._invoke_qwen_physical_call


def _fresh_alignment_backend_invoker(
    session: FreshAlignmentSession,
) -> Callable[[Callable[[], Any]], Any]:
    """Return the issuer-owned wrapper for one configured backend attempt."""
    issuer = _fresh_session_record(session).issuer
    return issuer._invoke_backend_call


def _call_capture(
    issuer: FreshAlignmentIssuer,
    call: _ObservedPhysicalCall,
) -> tuple[
    StrictCaptureResult,
    AuthorityTransformResult,
    AuthorityCallInput,
    PhysicalCallReceipt,
    LegacyCallDistributionReceipt,
    tuple[tuple[Mapping[str, Any], ...], ...],
]:
    blocks_by_source = {block.source_index: block for block in issuer.blocks}
    texts = tuple(
        blocks_by_source[index].alignment_text for index in call.source_block_indices
    )
    try:
        legacy = legacy_distribute_before_shift(
            call.post_units,
            texts=texts,
            iso=issuer.language,
            origin=call.legacy_origin_seconds,
            identity=call.legacy_origin_kind == "identity",
            raw_unit_ids=call.raw_unit_ids,
            source_indices=call.source_block_indices,
        )
    except Exception as exc:
        missing_key = exc.args[0] if isinstance(exc, KeyError) and exc.args else None
        retained_field = missing_key if type(missing_key) is str else ""
        detail_code = {
            "text": "retained-unit-text",
            "start": "retained-unit-start",
            "end": "retained-unit-end",
        }.get(retained_field, "retained-unit-operand")
        _attach_canonical_failure(
            exc,
            kind="legacy-time-transform-failed",
            phase="legacy-time-transform",
            detail_code=detail_code,
        )
        raise
    try:
        capture = capture_strict_units(
            call.post_units,
            original_units=call.original_units,
            call_index=call.call_index,
            raw_unit_ids=call.raw_unit_ids,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(getattr(exc, "failure", None), CanonicalFailure):
            raise
        capture = _invalid_capture(call.call_index, call.raw_unit_ids)
    retained_count = len(legacy.receipt.consumed_prefix_unit_ids)
    if call.geometry_failure is not None:
        transform = AuthorityTransformResult(
            call.call_index,
            "invalid",
            capture,
            None,
            None,
            call.geometry_failure,
        )
    else:
        transform = transform_strict_units(
            capture,
            physical_origin_seconds=call.authority_origin_seconds,
            identity=call.legacy_origin_kind == "identity",
            retained_unit_count=retained_count,
        )
    if capture.status != "valid":
        preflight_status: Literal["valid", "capture-invalid", "transform-invalid"] = (
            "capture-invalid"
        )
        surfaces = None
        failure = capture.failure
    elif transform.status != "valid":
        preflight_status = "transform-invalid"
        surfaces = tuple(unit.surface for unit in capture.units or ())
        failure = transform.failure
    else:
        preflight_status = "valid"
        surfaces = tuple(unit.surface for unit in capture.units or ())
        failure = None
    call_input = AuthorityCallInput(
        call.call_index,
        call.source_block_indices,
        call.raw_node_range,
        call.raw_unit_ids,
        surfaces,
        preflight_status,
        failure,
    )
    legacy_slice_digest = _stable_digest(legacy.receipt)
    try:
        legacy_absolute_digest = _stable_digest(legacy.block_units)
    except Exception:
        legacy_absolute_digest = None
    physical = PhysicalCallReceipt(
        call.call_index,
        call.source_block_indices,
        call.audio_sample_start,
        call.audio_sample_end,
        call.sample_rate,
        call.physical_origin_seconds,
        call.legacy_origin_seconds,
        call.legacy_origin_kind,
        call.authority_origin_seconds,
        call.backend_model_config_digest,
        call.route_input_digest,
        capture.status,
        transform.failure,
        capture.raw_units_digest,
        capture.normalized_relative_digest,
        legacy_slice_digest,
        legacy_absolute_digest,
        transform.status,
        transform.authority_absolute_digest,
        call.raw_unit_ids,
    )
    return (
        capture,
        transform,
        call_input,
        physical,
        legacy.receipt,
        tuple(tuple(unit for unit in owner) for owner in legacy.block_units),
    )


def _route_inputs(
    issuer: FreshAlignmentIssuer,
    calls: tuple[AuthorityCallInput, ...],
) -> tuple[
    tuple[RouteExpectation, ...],
    tuple[AuthoritySkippedBlockInput, ...],
    tuple[RouteClaim, ...],
]:
    owners = {
        source_index: call.call_index
        for call in calls
        for source_index in call.source_block_indices
    }
    route: list[RouteExpectation] = []
    skipped: list[AuthoritySkippedBlockInput] = []
    claims: list[RouteClaim] = []
    for delivery_index, block in enumerate(issuer.blocks):
        owner_index = owners.get(block.source_index)
        if owner_index is not None:
            route.append(
                RouteExpectation(
                    delivery_index, block.source_index, "call", owner_index
                )
            )
            claims.append(
                RouteClaim("call", owner_index, delivery_index, block.source_index)
            )
            continue
        stripped = block.alignment_text.strip()
        skip_index = len(skipped)
        skipped.append(
            AuthoritySkippedBlockInput(
                delivery_index,
                block.source_index,
                "empty-alignment-text" if not stripped else "missing-crop",
                "empty"
                if not block.alignment_text
                else "whitespace"
                if not stripped
                else "nonempty",
            )
        )
        route.append(
            RouteExpectation(delivery_index, block.source_index, "skip", skip_index)
        )
        claims.append(
            RouteClaim("skip", skip_index, delivery_index, block.source_index)
        )
    return tuple(route), tuple(skipped), tuple(claims)


def _issued_public_snapshot(issued: IssuedFreshAlignment) -> tuple[object, ...]:
    return (
        issued.context_content_digest,
        issued.receipt_digest,
        issued.physical_calls,
        issued.distribution,
        issued.seed_status,
        issued.seed_reasons,
        issued._binding,
    )


def seal_fresh_alignment(session: FreshAlignmentSession) -> IssuedFreshAlignment:
    """Run AO-11 through AO-13 and atomically issue one sealed acquisition."""
    if not isinstance(session, FreshAlignmentSession):
        raise TypeError("fresh alignment session is invalid")
    session_record = _fresh_session_record(session)
    session_record.sealed = True
    issuer = session_record.issuer
    issuer.sealed = True
    rows = tuple(_call_capture(issuer, call) for call in issuer.observed)
    captures = tuple(row[0] for row in rows)
    transforms = tuple(row[1] for row in rows)
    call_inputs = tuple(row[2] for row in rows)
    physical_calls = tuple(row[3] for row in rows)
    legacy_receipts = tuple(row[4] for row in rows)
    legacy_by_source: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for call, row in zip(issuer.observed, rows, strict=True):
        for source_index, owner in zip(call.source_block_indices, row[5], strict=True):
            legacy_by_source[source_index] = owner
    legacy_block_units = tuple(
        legacy_by_source.get(block.source_index, ()) for block in issuer.blocks
    )
    route, skipped, claims = _route_inputs(issuer, call_inputs)
    profile = _align_context_authority_profile(issuer.context)
    if not isinstance(profile, AuthorityLimitProfile):
        raise TypeError("context authority profile has the wrong type")
    distribution = _build_context_authority_distribution(
        blocks=issuer.blocks,
        delivery_route=route,
        calls=call_inputs,
        skipped_blocks=skipped,
        route_claims=claims,
        iso=issuer.language,
        _limits=profile,
        _verifier_cut_mutator=issuer.verifier_cut_mutator,
    )
    if distribution.work.status == "seal-mismatch":
        terminal = distribution.work.terminal_call_index
        raise_distribution_seal_mismatch(
            issuer.context,
            terminal_call_index=0 if terminal is None else terminal,
            ledger=issuer.ledger,
            dispose=issuer._dispose,
        )
    from voxweave.core.align_seed import build_align_seed

    fresh_units = tuple(
        unit
        for transform in transforms
        for unit in (transform.units if transform.units is not None else ())
    )
    seed = build_align_seed(
        blocks=issuer.blocks,
        units=fresh_units,
        distribution=distribution,
        iso=issuer.language,
    )
    receipt_digest = _stable_digest(
        {
            "context_content_digest": issuer.context.context_content_digest,
            "physical_calls": physical_calls,
            "legacy_distribution": legacy_receipts,
            "authority_distribution": distribution,
            "seed_status": seed.status,
            "seed_reasons": seed.reasons,
        }
    )
    issued = object.__new__(IssuedFreshAlignment)
    object.__setattr__(
        issued, "context_content_digest", issuer.context.context_content_digest
    )
    object.__setattr__(issued, "receipt_digest", receipt_digest)
    object.__setattr__(issued, "physical_calls", physical_calls)
    object.__setattr__(issued, "distribution", distribution)
    object.__setattr__(issued, "seed_status", seed.status)
    object.__setattr__(issued, "seed_reasons", seed.reasons)
    object.__setattr__(issued, "_binding", secrets.token_hex(32))
    record = _FreshRecord(
        issuer.context,
        issued,
        (),
        issuer.blocks,
        copy.deepcopy(captures),
        copy.deepcopy(transforms),
        copy.deepcopy(distribution),
        copy.deepcopy(seed),
        copy.deepcopy(physical_calls),
        copy.deepcopy(legacy_receipts),
        copy.deepcopy(legacy_block_units),
    )
    record.public_snapshot = _issued_public_snapshot(issued)
    record.seals = _fresh_seals(record)
    with _FRESH_LOCK:
        _FRESH[id(issued)] = record
    issuer._dispose()
    return issued


def _fresh_record(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> _FreshRecord:
    with _FRESH_LOCK:
        record = _FRESH.get(id(acquisition))
        if (
            record is None
            or record.context is not context
            or record.issued is not acquisition
        ):
            raise ValueError("fresh alignment is unissued, changed, or cross-context")
        _verify_fresh_seals(record)
        if (
            acquisition.context_content_digest != context.context_content_digest
            or _issued_public_snapshot(acquisition) != record.public_snapshot
        ):
            raise ValueError("fresh alignment is unissued, changed, or cross-context")
        return record


def _bind_fresh_adapter_payload(
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
    *,
    legacy_delivery: object,
    projection_inputs: object,
    strict_input_status: object,
    v2_policy_status: object,
    profile_resolution: object,
    evidence_resolution: object,
) -> None:
    record = _fresh_record(context, acquisition)
    if record.adapter_payload is not None:
        raise ValueError("fresh alignment adapter payload is already bound")
    cues = getattr(legacy_delivery, "cues", None)
    top_units = getattr(legacy_delivery, "word_segments", None)
    if not isinstance(cues, tuple) or not isinstance(top_units, tuple):
        raise ValueError("legacy delivery lacks its immutable AO-10 unit projection")

    def scalar_fact(value: Any) -> tuple[str, Any]:
        return ("float", value.hex()) if type(value) is float else ("value", value)

    def expected_facts(
        units: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                scalar_fact(unit["text"]),
                scalar_fact(unit["start"]),
                scalar_fact(unit["end"]),
            )
            for unit in units
        )

    def delivered_facts(units: Sequence[object]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                scalar_fact(getattr(unit, "text", None)),
                scalar_fact(getattr(unit, "start", None)),
                scalar_fact(getattr(unit, "end", None)),
            )
            for unit in units
        )

    if tuple(getattr(cue, "source_index", None) for cue in cues) != tuple(
        block.source_index for block in record.blocks
    ):
        raise ValueError("legacy delivery source order changed after acquisition")
    for cue, expected_units in zip(cues, record.legacy_block_units, strict=True):
        if delivered_facts(getattr(cue, "word_data", ())) != expected_facts(
            expected_units
        ):
            raise ValueError(
                "legacy delivery block units do not match the sealed slice"
            )
    expected_top = tuple(
        fact for owner in record.legacy_block_units for fact in expected_facts(owner)
    )
    if delivered_facts(top_units) != expected_top:
        raise ValueError(
            "legacy delivery top-level units do not match the sealed slice"
        )
    record.adapter_payload = copy.deepcopy(
        _AdapterPayload(
            legacy_delivery,
            projection_inputs,
            strict_input_status,
            v2_policy_status,
            profile_resolution,
            evidence_resolution,
        )
    )


def _fresh_adapter_payload(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> _AdapterPayload:
    record = _fresh_record(context, acquisition)
    if record.adapter_payload is None:
        raise ValueError("fresh alignment lacks its sealed adapter payload")
    return copy.deepcopy(record.adapter_payload)


def _verify_fresh_alignment(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> VerifiedFreshAlignment | None:
    record = _fresh_record(context, acquisition)
    seed = record.seed
    if (
        acquisition.distribution.status != "valid"
        or getattr(seed, "status", None) != "valid"
        or any(transform.status != "valid" for transform in record.transforms)
    ):
        record.transfer_terminal = "retired"
        return None
    if record.verified is not None or record.transfer_terminal != "live":
        raise ValueError("fresh alignment transfer was already issued or retired")
    if record.adapter_payload is None:
        raise ValueError("fresh alignment transfer lacks adapter admission")
    profile = getattr(record.adapter_payload.profile_resolution, "profile", None)
    if profile is None:
        record.transfer_terminal = "retired"
        return None
    evidence = record.adapter_payload.evidence_resolution
    sing_spans = getattr(evidence, "sing_spans", None)
    if sing_spans is None:
        admitted_sing_spans: tuple[tuple[float, float], ...] = ()
    else:
        admitted_sing_spans = tuple(tuple(span) for span in sing_spans)
    verified = object.__new__(VerifiedFreshAlignment)
    object.__setattr__(
        verified, "context_content_digest", context.context_content_digest
    )
    object.__setattr__(verified, "receipt_digest", acquisition.receipt_digest)
    object.__setattr__(verified, "_binding", secrets.token_hex(32))
    record.verified = verified
    with _FRESH_LOCK:
        _VERIFIED_FRESH[id(verified)] = _VerifiedRecord(
            context,
            acquisition,
            verified,
            copy.deepcopy(seed),
            copy.deepcopy(profile),
            copy.deepcopy(admitted_sing_spans),
            verified._binding,
        )
    return verified


def _consume_verified_fresh_alignment(
    verified: VerifiedFreshAlignment,
    *,
    profile: object,
) -> tuple[object, str, tuple[tuple[float, float], ...]]:
    with _FRESH_LOCK:
        verified_record = _VERIFIED_FRESH.get(id(verified))
        if (
            verified_record is None
            or verified_record.verified is not verified
            or verified._binding != verified_record.binding
            or verified.context_content_digest
            != verified_record.context.context_content_digest
            or verified.receipt_digest != verified_record.acquisition.receipt_digest
            or profile != verified_record.profile_snapshot
        ):
            raise ValueError("verified fresh alignment is unissued or changed")
        record = _fresh_record(verified_record.context, verified_record.acquisition)
        if record.verified is not verified or record.transfer_terminal != "live":
            raise ValueError("fresh alignment transfer is not live")
        record.transfer_terminal = "consumed"
        return (
            copy.deepcopy(verified_record.seed_snapshot),
            verified.receipt_digest,
            copy.deepcopy(verified_record.sing_spans_snapshot),
        )


def _retire_fresh_transfer(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> None:
    record = _fresh_record(context, acquisition)
    if record.transfer_terminal == "live":
        record.transfer_terminal = "retired"


def _fresh_core_inputs(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> tuple[
    tuple[AuthorityBlock, ...],
    tuple[StrictCaptureResult, ...],
    tuple[AuthorityTransformResult, ...],
    AuthorityDistributionReceipt,
    str,
    tuple[str, ...],
    tuple[PhysicalCallReceipt, ...],
]:
    record = _fresh_record(context, acquisition)
    seed = record.seed
    return (
        copy.deepcopy(record.blocks),
        copy.deepcopy(record.captures),
        copy.deepcopy(record.transforms),
        copy.deepcopy(record.distribution),
        getattr(seed, "status"),
        tuple(getattr(seed, "reasons")),
        copy.deepcopy(record.physical_calls),
    )


def _fresh_evidence_inputs(
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
) -> tuple[
    tuple[AuthorityBlock, ...],
    tuple[StrictCaptureResult, ...],
    tuple[AuthorityTransformResult, ...],
    tuple[LegacyCallDistributionReceipt, ...],
    tuple[tuple[Mapping[str, Any], ...], ...],
]:
    """Return a separate thaw of facts needed by the AO-21 evidence binder."""
    record = _fresh_record(context, acquisition)
    return (
        copy.deepcopy(record.blocks),
        copy.deepcopy(record.captures),
        copy.deepcopy(record.transforms),
        copy.deepcopy(record.legacy_receipts),
        copy.deepcopy(record.legacy_block_units),
    )


def _fresh_seed(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> object:
    return copy.deepcopy(_fresh_record(context, acquisition).seed)


__all__ = [
    "AcquisitionAdmissionEvent",
    "AcquisitionAdmissionLedger",
    "AuthorityTransformResult",
    "FreshSealBroken",
    "FreshAlignmentSession",
    "FreshUnit",
    "IssuedFreshAlignment",
    "PhysicalCallReceipt",
    "QwenSampleGeometry",
    "SampleGeometryError",
    "StrictCaptureResult",
    "StrictCapturedUnit",
    "VerifiedFreshAlignment",
    "begin_fresh_alignment",
    "capture_strict_units",
    "qwen_sample_geometry",
    "raise_distribution_seal_mismatch",
    "seal_fresh_alignment",
    "transform_strict_units",
]
