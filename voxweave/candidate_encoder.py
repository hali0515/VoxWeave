"""One-use composite align encoding and pure selected-family lookup."""

from __future__ import annotations

import hashlib
import secrets
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from voxweave.align_adapter import (
    AlignDelivery,
    AlignEvaluatedResult,
    AlignProjectionInputs,
    _evaluated_record,
)
from voxweave.align_context import (
    IssuedAlignContext,
    IssuedContext,
    consume_context_role,
)
from voxweave.align_failures import CanonicalFailure
from voxweave.align_projector import project_align_delivery
from voxweave.engine_registry import EngineFamily
from voxweave.reference_projector import reference_align_projection


@dataclass(frozen=True)
class CandidateNotRequested:
    reason: Literal["not-requested"] = "not-requested"


@dataclass(frozen=True)
class CandidateFailure:
    failure: CanonicalFailure


@dataclass(frozen=True)
class EncodedCandidate:
    context_content_digest: str
    engine_family: EngineFamily
    delivery_digest: str
    vtt_bytes: bytes
    main_json_bytes: bytes
    vtt_sha256: str
    main_json_sha256: str


CandidateOutcome: TypeAlias = (
    CandidateNotRequested | CandidateFailure | EncodedCandidate
)


@dataclass(frozen=True, init=False)
class CandidateSet:
    context_content_digest: str
    outcomes: tuple[tuple[EngineFamily, CandidateOutcome], ...]
    _binding: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("CandidateSet is issuer-only")

    def outcome_for(self, family: EngineFamily) -> CandidateOutcome:
        matches = tuple(outcome for name, outcome in self.outcomes if name == family)
        if len(matches) != 1:
            raise ValueError("candidate family outcome is missing or duplicated")
        return matches[0]


@dataclass(frozen=True)
class VerifiedEncodedCandidate:
    context_content_digest: str
    engine_family: EngineFamily
    delivery_digest: str
    vtt_bytes: bytes
    main_json_bytes: bytes
    vtt_sha256: str
    main_json_sha256: str


@dataclass(frozen=True)
class _CandidateRecord:
    context: IssuedContext
    result: object
    candidate_set: CandidateSet
    outcomes: tuple[tuple[EngineFamily, CandidateOutcome], ...]
    outcome_snapshot: tuple[tuple[EngineFamily, CandidateOutcome], ...]
    binding: str


@dataclass(frozen=True)
class _EncodedRecord:
    context: IssuedAlignContext
    result: AlignEvaluatedResult
    candidate: EncodedCandidate
    delivery: AlignDelivery
    projection_inputs: AlignProjectionInputs


@dataclass(frozen=True)
class _VerifiedRecord:
    context: IssuedAlignContext
    result: AlignEvaluatedResult
    evidence_core: object
    candidate: VerifiedEncodedCandidate


_SETS: dict[int, _CandidateRecord] = {}
_ENCODED: dict[int, _EncodedRecord] = {}
_VERIFIED: dict[tuple[int, int, EngineFamily, str, str], _VerifiedRecord] = {}
_LOCK = threading.RLock()


class SelectedCandidateError(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


class SelectedRenderError(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encode_family(
    context: IssuedAlignContext,
    result: AlignEvaluatedResult,
    family: EngineFamily,
    delivery: AlignDelivery,
    projection_inputs: AlignProjectionInputs,
) -> EncodedCandidate:
    projected = project_align_delivery(
        delivery,
        projection_inputs,
        strict=family == "boundary-v2",
    )
    candidate = EncodedCandidate(
        context.context_content_digest,
        family,
        delivery.receipt_digest,
        projected.vtt_bytes,
        projected.main_json_bytes,
        _digest(projected.vtt_bytes),
        _digest(projected.main_json_bytes),
    )
    with _LOCK:
        _ENCODED[id(candidate)] = _EncodedRecord(
            context,
            result,
            candidate,
            delivery,
            projection_inputs,
        )
    return candidate


def _encoder_failure() -> CandidateFailure:
    return CandidateFailure(
        CanonicalFailure("preencode-failed", "encoder", "main-json-encode")
    )


def encode_align_candidates(
    context: IssuedAlignContext,
    result: AlignEvaluatedResult,
) -> CandidateSet:
    """Attempt both align families once and return their fixed-order outcomes."""
    record = _evaluated_record(context, result)
    consume_context_role(context, "encoder", consumer="encode_align_candidates")

    outcomes: list[tuple[EngineFamily, CandidateOutcome]] = []
    try:
        legacy: CandidateOutcome = _encode_family(
            context,
            result,
            "legacy-v1",
            result.legacy,
            record.projection_inputs,
        )
    except Exception:
        legacy = _encoder_failure()
    outcomes.append(("legacy-v1", legacy))

    if result.v2_status.kind == "not-requested":
        boundary: CandidateOutcome = CandidateNotRequested()
    elif result.v2_status.kind == "invalid":
        if result.v2_status.failure is None:
            raise ValueError("invalid v2 status lacks its canonical failure")
        boundary = CandidateFailure(result.v2_status.failure)
    elif result.v2 is None:
        boundary = CandidateFailure(
            CanonicalFailure("preencode-failed", "encoder", "main-json-encode")
        )
    else:
        try:
            boundary = _encode_family(
                context,
                result,
                "boundary-v2",
                result.v2,
                record.projection_inputs,
            )
        except Exception:
            boundary = _encoder_failure()
    outcomes.append(("boundary-v2", boundary))

    return _issue_candidate_set(context, result, tuple(outcomes))


def _issue_candidate_set(
    context: IssuedContext,
    result: object,
    outcomes: tuple[tuple[EngineFamily, CandidateOutcome], ...],
) -> CandidateSet:
    if tuple(family for family, _outcome in outcomes) != (
        "legacy-v1",
        "boundary-v2",
    ):
        raise ValueError("candidate outcomes are not in fixed family order")
    candidate_set = object.__new__(CandidateSet)
    object.__setattr__(
        candidate_set, "context_content_digest", context.context_content_digest
    )
    object.__setattr__(candidate_set, "outcomes", outcomes)
    object.__setattr__(candidate_set, "_binding", secrets.token_hex(32))
    with _LOCK:
        _SETS[id(candidate_set)] = _CandidateRecord(
            context,
            result,
            candidate_set,
            candidate_set.outcomes,
            deepcopy(candidate_set.outcomes),
            candidate_set._binding,
        )
    return candidate_set


def _candidate_set_record(
    context: IssuedContext, candidate_set: CandidateSet
) -> _CandidateRecord:
    with _LOCK:
        record = _SETS.get(id(candidate_set))
        if (
            record is None
            or record.candidate_set is not candidate_set
            or record.context is not context
            or candidate_set.context_content_digest != context.context_content_digest
            or candidate_set.outcomes is not record.outcomes
            or candidate_set.outcomes != record.outcome_snapshot
            or candidate_set._binding != record.binding
        ):
            raise ValueError("candidate set is unissued or cross-context")
        return record


def select_align_candidate(
    context: IssuedAlignContext, candidate_set: CandidateSet
) -> EncodedCandidate:
    """Select the registry family without rendering or fallback."""
    return _select_candidate(context, candidate_set, context.engine_family)


def _select_candidate(
    context: IssuedContext,
    candidate_set: CandidateSet,
    family: EngineFamily,
) -> EncodedCandidate:
    _candidate_set_record(context, candidate_set)
    outcome = candidate_set.outcome_for(family)
    if not isinstance(outcome, EncodedCandidate):
        raise SelectedCandidateError(
            CanonicalFailure(
                "selected-render-invalid",
                "renderer",
                "selected-candidate-missing",
            )
        )
    return outcome


def _render_failure() -> SelectedRenderError:
    return SelectedRenderError(
        CanonicalFailure("selected-render-invalid", "renderer", "derived-hash")
    )


def verify_selected_align_projection(
    context: IssuedAlignContext,
    result: AlignEvaluatedResult,
    candidate: EncodedCandidate,
) -> VerifiedEncodedCandidate:
    """Independently reproduce selected primary bytes and bind their hashes."""
    _evaluated_record(context, result)
    with _LOCK:
        encoded = _ENCODED.get(id(candidate))
    if (
        encoded is None
        or encoded.candidate is not candidate
        or encoded.context is not context
        or encoded.result is not result
        or candidate.engine_family != context.engine_family
        or candidate.context_content_digest != context.context_content_digest
    ):
        raise _render_failure()
    try:
        reference = reference_align_projection(
            encoded.delivery,
            encoded.projection_inputs,
            strict=candidate.engine_family == "boundary-v2",
        )
    except Exception as exc:
        raise _render_failure() from exc
    reference_vtt_hash = _digest(reference.vtt_bytes)
    reference_json_hash = _digest(reference.main_json_bytes)
    if (
        candidate.vtt_bytes != reference.vtt_bytes
        or candidate.main_json_bytes != reference.main_json_bytes
        or candidate.delivery_digest != encoded.delivery.receipt_digest
        or candidate.vtt_sha256 != reference_vtt_hash
        or candidate.main_json_sha256 != reference_json_hash
        or _digest(candidate.vtt_bytes) != candidate.vtt_sha256
        or _digest(candidate.main_json_bytes) != candidate.main_json_sha256
    ):
        raise _render_failure()
    verified = VerifiedEncodedCandidate(
        candidate.context_content_digest,
        candidate.engine_family,
        candidate.delivery_digest,
        candidate.vtt_bytes,
        candidate.main_json_bytes,
        candidate.vtt_sha256,
        candidate.main_json_sha256,
    )
    key = (
        id(context),
        id(result.evidence_core),
        verified.engine_family,
        verified.vtt_sha256,
        verified.main_json_sha256,
    )
    with _LOCK:
        _VERIFIED[key] = _VerifiedRecord(
            context,
            result,
            result.evidence_core,
            verified,
        )
    return verified


def _verified_hash_binding(
    context: IssuedAlignContext,
    evidence_core: object,
    family: EngineFamily,
    vtt_sha256: str,
    main_json_sha256: str,
) -> bool:
    key = (id(context), id(evidence_core), family, vtt_sha256, main_json_sha256)
    with _LOCK:
        record = _VERIFIED.get(key)
        return (
            record is not None
            and record.context is context
            and record.evidence_core is evidence_core
            and record.candidate.vtt_sha256 == vtt_sha256
            and record.candidate.main_json_sha256 == main_json_sha256
        )


__all__ = [
    "CandidateFailure",
    "CandidateNotRequested",
    "CandidateOutcome",
    "CandidateSet",
    "EncodedCandidate",
    "SelectedCandidateError",
    "SelectedRenderError",
    "VerifiedEncodedCandidate",
    "encode_align_candidates",
    "select_align_candidate",
    "verify_selected_align_projection",
]
