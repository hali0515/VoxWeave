"""Fresh-alignment admission ledger, record types and seal digests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, TYPE_CHECKING

from voxweave.align_context import IssuedAlignContext, _align_context_stable_fields
from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityDistributionReceipt,
    AuthorityLimitProfile,
    LegacyCallDistributionReceipt,
    StrictFailureLocator,
)
from voxweave.align_failures import CanonicalFailure, SecondaryFailure
from voxweave.align_snapshot import (
    FrozenJSON,
    FrozenObject,
    freeze_json,
    frozen_json_digest,
)
from voxweave.align_strict_capture import (
    PhysicalCallReceipt,
    AuthorityTransformResult,
    StrictCaptureResult,
)

if TYPE_CHECKING:
    from voxweave.align_acquisition import FreshAlignmentIssuer


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
        super().__init__(
            f"{failure.kind}/{failure.phase}/{failure.detail_code}: an internal"
            " alignment record changed after it was sealed. This is a VoxWeave bug,"
            " not a problem with your files; please report it with this message"
        )
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
    sample_count: int
    physical_origin_seconds: float
    legacy_origin_seconds: float
    legacy_origin_kind: Literal["identity", "sample-origin", "nominal-route"]
    authority_origin_seconds: float
    backend_model_config_digest: str
    route_input_digest: str
    geometry_failure: StrictFailureLocator | None


@dataclass(frozen=True)
class ReferencePhysicalCallFacts:
    """Issuer-time call facts retained independently from its public receipt."""

    call_index: int
    source_block_indices: tuple[int, ...]
    raw_nodes: tuple[Any, ...]
    original_nodes: tuple[Any, ...] | None
    raw_unit_ids: tuple[str, ...]
    raw_node_range: tuple[int, int]
    audio_sample_start: int
    audio_sample_end: int
    sample_rate: int
    sample_count: int
    physical_origin_seconds: float
    legacy_origin_seconds: float
    legacy_origin_kind: Literal["identity", "sample-origin", "nominal-route"]
    authority_origin_seconds: float
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
    legacy_relative_block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    reference_calls: tuple[ReferencePhysicalCallFacts, ...]
    backend_model_config_facts: FrozenJSON
    route_input_facts: FrozenJSON
    adapter_payload: _AdapterPayload | None = None
    verified: VerifiedFreshAlignment | None = None
    transfer_terminal: Literal["live", "consumed", "retired"] = "live"
    seals: _FreshSeals | None = None
    # Per seal detail code, the exact objects its digest last matched.
    verified_seal_inputs: dict[str, tuple[object, ...]] = field(default_factory=dict)


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


@dataclass(frozen=True)
class EvidenceCoreProducerInputs:
    """One producer-side thaw of every pre-selected-output EvidenceCore fact."""

    context_content_digest: str
    receipt_digest: str
    engine_family: str
    language: str
    route_kind: str
    stable_fields: FrozenObject
    authority_profile: AuthorityLimitProfile
    blocks: tuple[AuthorityBlock, ...]
    captures: tuple[StrictCaptureResult, ...]
    transforms: tuple[AuthorityTransformResult, ...]
    distribution: AuthorityDistributionReceipt
    seed_status: Literal["valid", "invalid"]
    seed_reasons: tuple[str, ...]
    physical_calls: tuple[PhysicalCallReceipt, ...]
    legacy_receipts: tuple[LegacyCallDistributionReceipt, ...]
    legacy_block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    legacy_relative_block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    backend_model_config_facts: FrozenJSON
    route_input_facts: FrozenJSON
    strict_input_status: object
    v2_policy_status: object
    profile_status: object
    evidence_status: object


@dataclass(frozen=True)
class EvidenceCoreReferenceInputs:
    """Independent issuer/context thaws used only by projector C."""

    claimed_context_content_digest: str
    claimed_receipt_digest: str
    engine_family: str
    language: str
    route_kind: str
    stable_fields: FrozenObject
    authority_profile: AuthorityLimitProfile
    blocks: tuple[AuthorityBlock, ...]
    captures: tuple[StrictCaptureResult, ...]
    transforms: tuple[AuthorityTransformResult, ...]
    distribution: AuthorityDistributionReceipt
    seed_status: Literal["valid", "invalid"]
    seed_reasons: tuple[str, ...]
    claimed_physical_calls: tuple[PhysicalCallReceipt, ...]
    legacy_receipts: tuple[LegacyCallDistributionReceipt, ...]
    legacy_block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    legacy_relative_block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    reference_calls: tuple[ReferencePhysicalCallFacts, ...]
    backend_model_config_facts: FrozenJSON
    route_input_facts: FrozenJSON
    strict_input_status: object
    v2_policy_status: object
    profile_status: object
    evidence_status: object


def _stable_fact_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_digest_facts(
    facts: object | None,
    claimed_digest: str | None,
    *,
    label: str,
    default_facts: object,
) -> tuple[FrozenJSON, str]:
    if facts is None:
        if claimed_digest is not None:
            raise ValueError(f"{label} facts are required with an explicit digest")
        value = default_facts
    else:
        value = facts
    try:
        digest = _stable_fact_digest(value)
        frozen = freeze_json(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} facts are outside the stable JSON domain") from exc
    if claimed_digest is not None and claimed_digest != digest:
        raise ValueError(f"{label} digest does not match its stable facts")
    return frozen, digest


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


# Each seal is split into an inputs projection and a digest over exactly those
# inputs.  The projection returns the sealed objects themselves (record fields,
# issued-receipt fields and the context's scalar attributes), so the digest can
# read nothing the projection does not name and the verifier can memoize on
# the identity of that tuple (see ``_verify_fresh_seals``).
def _context_seal_inputs(record: _FreshRecord) -> tuple[object, ...]:
    stable = _align_context_stable_fields(record.context)
    context = record.context
    return (
        context.context_content_digest,
        context.context_binding_digest,
        context.engine_family,
        context.effective_iso,
        context.route_kind,
        context._issuance_nonce,
        record.issued.context_content_digest,
        stable,
    )


def _context_seal_digest(*facts: object) -> str:
    return _stable_digest(facts)


def _capture_seal_inputs(record: _FreshRecord) -> tuple[object, ...]:
    return (record.captures, record.physical_calls, record.issued.physical_calls)


def _raw_seal_digest(
    record_captures: tuple[StrictCaptureResult, ...],
    physical_calls: tuple[PhysicalCallReceipt, ...],
    issued_physical_calls: tuple[PhysicalCallReceipt, ...],
) -> str:
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
        for capture in record_captures
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
            _physical_projection(physical_calls, names),
            _physical_projection(issued_physical_calls, names),
        )
    )


def _relative_seal_digest(
    record_captures: tuple[StrictCaptureResult, ...],
    physical_calls: tuple[PhysicalCallReceipt, ...],
    issued_physical_calls: tuple[PhysicalCallReceipt, ...],
) -> str:
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
        for capture in record_captures
    )
    names = ("call_index", "normalized_relative_digest")
    return _stable_digest(
        (
            captures,
            _physical_projection(physical_calls, names),
            _physical_projection(issued_physical_calls, names),
        )
    )


def _legacy_slice_seal_inputs(record: _FreshRecord) -> tuple[object, ...]:
    return (
        record.legacy_receipts,
        record.legacy_block_units,
        record.legacy_relative_block_units,
        record.physical_calls,
        record.issued.physical_calls,
    )


def _legacy_slice_seal_digest(
    legacy_receipts: tuple[LegacyCallDistributionReceipt, ...],
    legacy_block_units: tuple[tuple[Mapping[str, Any], ...], ...],
    legacy_relative_block_units: tuple[tuple[Mapping[str, Any], ...], ...],
    physical_calls: tuple[PhysicalCallReceipt, ...],
    issued_physical_calls: tuple[PhysicalCallReceipt, ...],
) -> str:
    names = ("call_index", "legacy_slice_digest", "legacy_absolute_digest")
    return _stable_digest(
        (
            legacy_receipts,
            legacy_block_units,
            legacy_relative_block_units,
            _physical_projection(physical_calls, names),
            _physical_projection(issued_physical_calls, names),
        )
    )


def _authority_seal_inputs(record: _FreshRecord) -> tuple[object, ...]:
    return (
        record.transforms,
        record.physical_calls,
        record.issued.physical_calls,
        record.backend_model_config_facts,
        record.route_input_facts,
        record.reference_calls,
    )


def _authority_seal_digest(
    record_transforms: tuple[AuthorityTransformResult, ...],
    physical_calls: tuple[PhysicalCallReceipt, ...],
    issued_physical_calls: tuple[PhysicalCallReceipt, ...],
    backend_model_config_facts: FrozenJSON,
    route_input_facts: FrozenJSON,
    reference_calls: tuple[ReferencePhysicalCallFacts, ...],
) -> str:
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
        for transform in record_transforms
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
            _physical_projection(physical_calls, names),
            _physical_projection(issued_physical_calls, names),
            backend_model_config_facts,
            route_input_facts,
            tuple(
                (
                    call.call_index,
                    call.source_block_indices,
                    call.raw_unit_ids,
                    call.raw_node_range,
                    call.audio_sample_start,
                    call.audio_sample_end,
                    call.sample_rate,
                    call.sample_count,
                    call.physical_origin_seconds,
                    call.legacy_origin_seconds,
                    call.legacy_origin_kind,
                    call.authority_origin_seconds,
                    call.geometry_failure,
                )
                for call in reference_calls
            ),
        )
    )


def _distribution_seal_inputs(record: _FreshRecord) -> tuple[object, ...]:
    return (record.distribution, record.issued.distribution)


def _distribution_seal_digest(
    distribution: AuthorityDistributionReceipt,
    issued_distribution: AuthorityDistributionReceipt,
) -> str:
    return _stable_digest((distribution, issued_distribution))


def _phase1_seal_inputs(record: _FreshRecord) -> tuple[object, ...]:
    return (record.seed, record.issued.seed_status, record.issued.seed_reasons)


def _phase1_seal_digest(
    seed: object, seed_status: str, seed_reasons: tuple[str, ...]
) -> str:
    return _stable_digest((seed, seed_status, seed_reasons))


# (detail code, ``_FreshSeals`` field, inputs projection, digest over the
# projected inputs), in verification order.
_SEALS: tuple[
    tuple[str, str, Callable[[_FreshRecord], tuple[object, ...]], Callable[..., str]],
    ...,
] = (
    ("context-seal", "context", _context_seal_inputs, _context_seal_digest),
    ("raw-seal", "raw", _capture_seal_inputs, _raw_seal_digest),
    ("relative-seal", "relative", _capture_seal_inputs, _relative_seal_digest),
    (
        "legacy-slice-seal",
        "legacy_slice",
        _legacy_slice_seal_inputs,
        _legacy_slice_seal_digest,
    ),
    ("authority-seal", "authority", _authority_seal_inputs, _authority_seal_digest),
    (
        "distribution-seal",
        "distribution",
        _distribution_seal_inputs,
        _distribution_seal_digest,
    ),
    ("phase1-seal", "phase1", _phase1_seal_inputs, _phase1_seal_digest),
)


def _seal_fresh_record(record: _FreshRecord) -> None:
    """Digest every component seal once and remember the objects it covered."""
    digests: dict[str, str] = {}
    covered: dict[str, tuple[object, ...]] = {}
    for detail_code, name, inputs_of, digest_of in _SEALS:
        inputs = inputs_of(record)
        digests[name] = digest_of(*inputs)
        covered[detail_code] = inputs
    record.seals = _FreshSeals(**digests)
    record.verified_seal_inputs = covered


def _same_objects(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return len(left) == len(right) and all(
        member is other for member, other in zip(left, right)
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
    """Re-digest every component seal whose covered objects were swapped.

    A seal whose inputs projection still returns the very objects that were
    last digested is not recomputed: re-digesting them on every
    ``_fresh_record`` access cost seven full digests per access, the phase-1
    seed alone seconds on a long full-pass episode.  Reassigning a record
    field (a ``dataclasses.replace`` swap included) or an attribute of the
    context or the issued receipt changes that identity, so the seal is
    digested again and must still match.

    Not caught: in-place mutation inside an already verified object, such as
    item assignment in the plain unit mappings of the legacy block units or
    the reference call nodes, or an ``object.__setattr__`` bypass on a nested
    frozen dataclass.  Every record field is a private deep copy taken at
    seal time and every thaw hands out deep copies, so nothing outside this
    module holds a reference into those objects, and the issued receipt's
    fields are frozen dataclasses and tuples.  Code that reaches into
    ``_FRESH`` can rewrite ``seals`` as easily, so the seals never covered it.
    """
    expected = record.seals
    if expected is None:
        _raise_component_seal("context-seal")
    for detail_code, name, inputs_of, digest_of in _SEALS:
        try:
            inputs = inputs_of(record)
            covered = record.verified_seal_inputs.get(detail_code)
            if covered is not None and _same_objects(covered, inputs):
                continue
            current = digest_of(*inputs)
        except Exception:
            _raise_component_seal(detail_code)
        if current != getattr(expected, name):
            _raise_component_seal(detail_code)
        record.verified_seal_inputs[detail_code] = inputs
