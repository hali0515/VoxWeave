"""Composite segmentation encoding, selection, verification, and SDH view."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from typing import Literal

from voxweave.align_context import IssuedSegmentationContext, consume_context_role
from voxweave.align_failures import CanonicalFailure
from voxweave.align_snapshot import thaw_json
from voxweave.candidate_encoder import (
    CandidateFailure,
    CandidateNotRequested,
    CandidateOutcome,
    CandidateSet,
    EncodedCandidate,
    SelectedRenderError,
    VerifiedEncodedCandidate,
    _issue_candidate_set,
    _select_candidate,
)
from voxweave.engine_registry import EngineFamily, MANIFEST_ENGINE_BY_FAMILY
from voxweave.reference_projector import reference_segmentation_projection
from voxweave.segmentation_adapter import (
    SegmentationAdapterResult,
    SegmentationDelivery,
    SegmentationProjectionInputs,
    _adapter_record,
    segmentation_delivery_digest,
)
from voxweave.segmentation_projector import (
    SegmentationProjectionEncodeError,
    project_segmentation_delivery,
)


@dataclass(frozen=True)
class _EncodedRecord:
    context: IssuedSegmentationContext
    result: SegmentationAdapterResult
    candidate: EncodedCandidate
    delivery: SegmentationDelivery
    projection_inputs: SegmentationProjectionInputs


@dataclass(frozen=True)
class _VerifiedRecord:
    context: IssuedSegmentationContext
    result: SegmentationAdapterResult
    verified: VerifiedEncodedCandidate
    delivery: SegmentationDelivery


@dataclass(frozen=True, init=False)
class _SimulatedBoundaryRow:
    case_id: str
    nonce: str = field(repr=False, compare=False)
    scope: tuple[int, int | None] = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("simulated row qualification is issuer-only")


@dataclass
class _QualificationRecord:
    token: _SimulatedBoundaryRow
    consumed: bool = False


_ENCODED: dict[int, _EncodedRecord] = {}
_VERIFIED: dict[int, _VerifiedRecord] = {}
_QUALIFIED_SELECTED: dict[int, tuple[IssuedSegmentationContext, object]] = {}
_QUALIFICATIONS: dict[str, _QualificationRecord] = {}
_LOCK = threading.RLock()


def _scope() -> tuple[int, int | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), None if task is None else id(task)


def _issue_simulated_boundary_row(case_id: str) -> _SimulatedBoundaryRow:
    """Issue one task-local qualification for the P6 mechanical row test."""
    if type(case_id) is not str or not case_id:
        raise ValueError("simulated row qualification needs a case id")
    token = object.__new__(_SimulatedBoundaryRow)
    object.__setattr__(token, "case_id", case_id)
    object.__setattr__(token, "nonce", secrets.token_hex(32))
    object.__setattr__(token, "scope", _scope())
    with _LOCK:
        _QUALIFICATIONS[token.nonce] = _QualificationRecord(token)
    return token


def _consume_qualification(token: _SimulatedBoundaryRow) -> None:
    with _LOCK:
        record = _QUALIFICATIONS.get(getattr(token, "nonce", ""))
        if (
            record is None
            or record.token is not token
            or record.consumed
            or token.scope != _scope()
        ):
            raise ValueError("simulated row qualification is invalid or consumed")
        record.consumed = True


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_engine(delivery: SegmentationDelivery) -> object:
    manifest = thaw_json(delivery.manifest)
    return manifest.get("engine") if isinstance(manifest, dict) else None


def _encode_family(
    context: IssuedSegmentationContext,
    result: SegmentationAdapterResult,
    delivery: SegmentationDelivery,
    projection_inputs: SegmentationProjectionInputs,
) -> EncodedCandidate:
    family = delivery.engine_family
    if (
        delivery.context_content_digest != context.context_content_digest
        or _manifest_engine(delivery) != MANIFEST_ENGINE_BY_FAMILY[family]
    ):
        raise ValueError("candidate family and manifest engine disagree")
    projection = project_segmentation_delivery(
        delivery,
        projection_inputs,
        strict=family == "boundary-v2",
    )
    candidate = EncodedCandidate(
        context.context_content_digest,
        family,
        segmentation_delivery_digest(delivery),
        projection.vtt_bytes,
        projection.main_json_bytes,
        _digest(projection.vtt_bytes),
        _digest(projection.main_json_bytes),
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


def _encoder_failure(
    detail_code: Literal["main-json-encode", "vtt-encode"] = "main-json-encode",
) -> CandidateFailure:
    return CandidateFailure(
        CanonicalFailure("preencode-failed", "encoder", detail_code)
    )


def _renderer_stage_failure() -> CandidateFailure:
    return CandidateFailure(
        CanonicalFailure(
            "shadow-internal-error",
            "renderer-stage",
            "renderer-stage",
        )
    )


def encode_segmentation_candidates(
    context: IssuedSegmentationContext,
    result: SegmentationAdapterResult,
) -> CandidateSet:
    """Consume one encoder role and attempt both typed deliveries in order."""
    record = _adapter_record(context, result)
    consume_context_role(
        context,
        "encoder",
        consumer="encode_segmentation_candidates",
    )
    outcomes: list[tuple[EngineFamily, CandidateOutcome]] = []
    try:
        legacy: CandidateOutcome = _encode_family(
            context,
            result,
            result.legacy,
            record.projection_inputs,
        )
    except SegmentationProjectionEncodeError as exc:
        legacy = _encoder_failure(exc.detail_code)
    except Exception:
        legacy = _encoder_failure()
    outcomes.append(("legacy-v1", legacy))
    if result.v2_status.kind == "not-requested":
        boundary: CandidateOutcome = CandidateNotRequested()
    elif result.v2_status.kind == "invalid":
        if result.v2_status.failure is None:
            raise ValueError("invalid boundary status lacks its failure")
        boundary = CandidateFailure(result.v2_status.failure)
    elif result.v2 is None:
        boundary = _encoder_failure()
    else:
        try:
            boundary = _encode_family(
                context,
                result,
                result.v2,
                record.projection_inputs,
            )
        except SegmentationProjectionEncodeError as exc:
            boundary = _encoder_failure(exc.detail_code)
        except Exception:
            boundary = _renderer_stage_failure()
    outcomes.append(("boundary-v2", boundary))
    return _issue_candidate_set(context, result, tuple(outcomes))


def select_segmentation_candidate(
    context: IssuedSegmentationContext,
    candidate_set: CandidateSet,
) -> EncodedCandidate:
    """Select the production registry row without rendering or fallback."""
    return _select_candidate(context, candidate_set, context.engine_family)


def select_qualified_segmentation_candidate(
    context: IssuedSegmentationContext,
    candidate_set: CandidateSet,
    *,
    qualification: _SimulatedBoundaryRow,
) -> EncodedCandidate:
    """Select boundary only for the one explicitly scoped mechanical test."""
    _consume_qualification(qualification)
    candidate = _select_candidate(context, candidate_set, "boundary-v2")
    with _LOCK:
        _QUALIFIED_SELECTED[id(candidate)] = (context, candidate_set)
    return candidate


def _render_failure() -> SelectedRenderError:
    return SelectedRenderError(
        CanonicalFailure("selected-render-invalid", "renderer", "derived-hash")
    )


def verify_selected_segmentation_projection(
    context: IssuedSegmentationContext,
    result: SegmentationAdapterResult,
    candidate: EncodedCandidate,
) -> VerifiedEncodedCandidate:
    """Independently reproduce one selected delivery and bind its exact bytes."""
    _adapter_record(context, result)
    with _LOCK:
        encoded = _ENCODED.get(id(candidate))
        qualified = _QUALIFIED_SELECTED.get(id(candidate))
    selected_family_ok = candidate.engine_family == context.engine_family or (
        candidate.engine_family == "boundary-v2"
        and qualified is not None
        and qualified[0] is context
    )
    if (
        encoded is None
        or encoded.context is not context
        or encoded.result is not result
        or encoded.candidate is not candidate
        or not selected_family_ok
        or candidate.context_content_digest != context.context_content_digest
        or candidate.delivery_digest != segmentation_delivery_digest(encoded.delivery)
    ):
        raise _render_failure()
    try:
        reference = reference_segmentation_projection(
            encoded.delivery,
            encoded.projection_inputs,
            strict=candidate.engine_family == "boundary-v2",
        )
    except Exception as exc:
        raise _render_failure() from exc
    if (
        candidate.vtt_bytes != reference.vtt_bytes
        or candidate.main_json_bytes != reference.main_json_bytes
        or candidate.vtt_sha256 != _digest(reference.vtt_bytes)
        or candidate.main_json_sha256 != _digest(reference.main_json_bytes)
        or candidate.vtt_sha256 != _digest(candidate.vtt_bytes)
        or candidate.main_json_sha256 != _digest(candidate.main_json_bytes)
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
    with _LOCK:
        _VERIFIED[id(verified)] = _VerifiedRecord(
            context,
            result,
            verified,
            encoded.delivery,
        )
    return verified


def project_selected_sdh_dialogue(
    context: IssuedSegmentationContext,
    result: SegmentationAdapterResult,
    verified: VerifiedEncodedCandidate,
) -> tuple[dict[str, object], ...]:
    """Return the selected delivery's intrinsic dialogue view for SDH fitting."""
    _adapter_record(context, result)
    with _LOCK:
        record = _VERIFIED.get(id(verified))
    if (
        record is None
        or record.context is not context
        or record.result is not result
        or record.verified is not verified
        or verified.delivery_digest != segmentation_delivery_digest(record.delivery)
    ):
        raise _render_failure()
    return tuple(
        {
            "text": cue.text,
            "start": cue.start,
            "end": cue.end,
            **({"lyric": True} if cue.lyric is True else {}),
        }
        for cue in record.delivery.cues
    )


__all__ = [
    "encode_segmentation_candidates",
    "project_selected_sdh_dialogue",
    "select_qualified_segmentation_candidate",
    "select_segmentation_candidate",
    "verify_selected_segmentation_projection",
]
