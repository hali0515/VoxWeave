"""Immutable align delivery values and the pending-RAT evaluation issuer."""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from voxweave.align_context import (
    IssuedAlignContext,
    consume_context_role,
    role_vector,
)
from voxweave.align_evidence_core import EvidenceCore
from voxweave.align_failures import CanonicalFailure
from voxweave.align_snapshot import FrozenObject
from voxweave.engine_registry import EngineFamily


@dataclass(frozen=True)
class PersistedAlignUnit:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignDeliveryCue:
    source_index: int
    text: str
    start: float
    end: float
    lyric: bool | None
    unit_ids: tuple[str, ...]
    word_data: tuple[PersistedAlignUnit, ...]
    speech_start: float | None
    speech_end: float | None


@dataclass(frozen=True)
class AlignDelivery:
    context_content_digest: str
    receipt_digest: str
    engine_family: EngineFamily
    route_kind: Literal["ctc-full", "mms-full", "qwen-crop"]
    cues: tuple[AlignDeliveryCue, ...]
    word_segments: tuple[PersistedAlignUnit, ...]


@dataclass(frozen=True)
class SourceBlockDecoration:
    source_index: int
    speaker: str | None
    speakers: tuple[tuple[str | None, str], ...] | None


@dataclass(frozen=True)
class AlignProjectionInputs:
    language: str
    source_blocks: tuple[SourceBlockDecoration, ...]
    vad_speech: tuple[tuple[float, float], ...] | None
    shot_changes: tuple[float, ...] | None
    sing_spans: tuple[tuple[float, float], ...] | None
    speaker_turns: tuple[tuple[float, float, str], ...] | None
    voiceprint_capture: str | None
    voiceprint_media: str | None
    segmentation: FrozenObject | None


@dataclass(frozen=True)
class V2Status:
    kind: Literal["not-requested", "valid", "invalid"]
    failure: CanonicalFailure | None

    def __post_init__(self) -> None:
        if (self.kind == "invalid") != (self.failure is not None):
            raise ValueError("v2 status failure does not match its kind")


@dataclass(frozen=True, init=False)
class AlignEvaluatedResult:
    context_content_digest: str
    receipt_digest: str
    legacy: AlignDelivery
    v2: AlignDelivery | None
    v2_status: V2Status
    evidence_core: EvidenceCore
    comparison: None
    _issuance_nonce: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("AlignEvaluatedResult is issuer-only")


@dataclass(frozen=True)
class _EvaluatedRecord:
    context: IssuedAlignContext
    result: AlignEvaluatedResult
    projection_inputs: AlignProjectionInputs
    delivery: AlignDelivery
    delivery_snapshot: AlignDelivery
    projection_snapshot: AlignProjectionInputs
    evidence_core: EvidenceCore
    evidence_snapshot: EvidenceCore
    status_snapshot: V2Status
    nonce: str


_EVALUATED: dict[int, _EvaluatedRecord] = {}
_LOCK = threading.RLock()


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def issue_legacy_align_evaluated_result(
    context: IssuedAlignContext,
    *,
    delivery: AlignDelivery,
    projection_inputs: AlignProjectionInputs,
    evidence_core: EvidenceCore,
    shadow_requested: bool,
    v2_failure: CanonicalFailure | None = None,
) -> AlignEvaluatedResult:
    """Issue the honest no-W1 evaluated result while RAT-1 remains pending."""
    if type(shadow_requested) is not bool:
        raise TypeError("shadow_requested must be an exact bool")
    if not role_vector(context) or role_vector(context)[0] != "C":
        raise ValueError("fresh acquisition role was not consumed")
    if (
        delivery.context_content_digest != context.context_content_digest
        or delivery.receipt_digest != evidence_core.core_digest
        or not _sha256(delivery.receipt_digest)
        or delivery.engine_family != "legacy-v1"
        or delivery.route_kind != context.route_kind
        or evidence_core.context_content_digest != context.context_content_digest
        or projection_inputs.language != context.effective_iso
    ):
        raise ValueError("align evaluated-result binding is invalid")
    source_indices = tuple(cue.source_index for cue in delivery.cues)
    decoration_indices = tuple(
        item.source_index for item in projection_inputs.source_blocks
    )
    if (
        len(set(source_indices)) != len(source_indices)
        or decoration_indices != source_indices
    ):
        raise ValueError("align source decoration order is invalid")
    if (projection_inputs.voiceprint_capture is None) != (
        projection_inputs.voiceprint_media is None
    ):
        raise ValueError("voiceprint carriers must be a pair")
    consume_context_role(context, "adapter", consumer="run_locked_align_adapter")
    status = (
        V2Status(
            "invalid",
            v2_failure
            or CanonicalFailure(
                "fresh-authority-invalid", "w1-admission", "w1-root-event"
            ),
        )
        if shadow_requested
        else V2Status("not-requested", None)
    )
    import secrets

    result = object.__new__(AlignEvaluatedResult)
    object.__setattr__(result, "context_content_digest", context.context_content_digest)
    object.__setattr__(result, "receipt_digest", delivery.receipt_digest)
    object.__setattr__(result, "legacy", delivery)
    object.__setattr__(result, "v2", None)
    object.__setattr__(result, "v2_status", status)
    object.__setattr__(result, "evidence_core", evidence_core)
    object.__setattr__(result, "comparison", None)
    object.__setattr__(result, "_issuance_nonce", secrets.token_hex(32))
    with _LOCK:
        _EVALUATED[id(result)] = _EvaluatedRecord(
            context,
            result,
            projection_inputs,
            delivery,
            deepcopy(delivery),
            deepcopy(projection_inputs),
            evidence_core,
            deepcopy(evidence_core),
            deepcopy(result.v2_status),
            result._issuance_nonce,
        )
    return result


def _evaluated_record(
    context: IssuedAlignContext, result: AlignEvaluatedResult
) -> _EvaluatedRecord:
    with _LOCK:
        record = _EVALUATED.get(id(result))
        if (
            record is None
            or record.result is not result
            or record.context is not context
            or record.delivery is not result.legacy
            or record.evidence_core is not result.evidence_core
        ):
            raise ValueError("align evaluated result is unissued or cross-context")
        if (
            result.context_content_digest != context.context_content_digest
            or result.receipt_digest != record.delivery.receipt_digest
            or result.legacy != record.delivery_snapshot
            or result.v2 is not None
            or result.v2_status != record.status_snapshot
            or result.comparison is not None
            or result.evidence_core != record.evidence_snapshot
            or record.projection_inputs != record.projection_snapshot
            or result._issuance_nonce != record.nonce
        ):
            raise ValueError("align evaluated result stable binding changed")
        return record


__all__ = [
    "AlignDelivery",
    "AlignDeliveryCue",
    "AlignEvaluatedResult",
    "AlignProjectionInputs",
    "PersistedAlignUnit",
    "SourceBlockDecoration",
    "V2Status",
    "issue_legacy_align_evaluated_result",
]
