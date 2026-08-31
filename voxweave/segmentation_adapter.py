"""Context-bound legacy and P5-authority segmentation deliveries."""

from __future__ import annotations

import copy
import secrets
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from voxweave.align_adapter import V2Status
from voxweave.align_context import IssuedSegmentationContext, consume_context_role
from voxweave.align_failures import CanonicalFailure
from voxweave.align_snapshot import (
    FrozenArray,
    FrozenJSON,
    FrozenObject,
    RawJSONCarrier,
    freeze_json,
    frozen_json_digest,
    thaw_json,
)
from voxweave.core.schema import Cue
from voxweave.engine_registry import EngineFamily, MANIFEST_ENGINE_BY_FAMILY

if TYPE_CHECKING:
    from voxweave.core.segdoc import SegDocument


@dataclass(frozen=True)
class SegmentationDeliveryCue:
    unit_range: tuple[int, int]
    text: str
    start: float
    end: float
    word_data: tuple[FrozenObject, ...]
    speech_start: float | None
    speech_end: float | None
    lyric: bool | None
    speaker_ids: tuple[str, ...] | None


@dataclass(frozen=True)
class SegmentationCarriers:
    vad_speech: tuple[FrozenArray, ...]
    shot_changes: tuple[FrozenJSON, ...] | None
    sing_spans: tuple[FrozenArray, ...] | None
    speaker_turns: RawJSONCarrier
    voiceprint_capture: str | None
    voiceprint_media: str | None


@dataclass(frozen=True)
class SegmentationDelivery:
    context_content_digest: str
    engine_family: EngineFamily
    language: str
    cues: tuple[SegmentationDeliveryCue, ...]
    top_level_word_segments: tuple[FrozenObject, ...]
    carriers: SegmentationCarriers
    manifest: FrozenObject


@dataclass(frozen=True)
class SegmentationProjectionInputs:
    timestamps: bool
    speaker_names: tuple[tuple[str, str], ...]


@dataclass(frozen=True, init=False)
class IssuedLegacySegmentation:
    context_content_digest: str
    delivery: SegmentationDelivery
    provider_ledger: FrozenJSON
    manifest_digest: str
    _binding: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("IssuedLegacySegmentation is issuer-only")


@dataclass(frozen=True, init=False)
class SegmentationAdapterResult:
    context_content_digest: str
    legacy: SegmentationDelivery
    v2: SegmentationDelivery | None
    v2_status: V2Status
    _binding: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("SegmentationAdapterResult is issuer-only")


@dataclass(frozen=True)
class _LegacyRecord:
    context: IssuedSegmentationContext
    issued: IssuedLegacySegmentation
    delivery: SegmentationDelivery
    delivery_snapshot: SegmentationDelivery
    provider_ledger_snapshot: FrozenJSON
    projection_inputs: SegmentationProjectionInputs
    projection_snapshot: SegmentationProjectionInputs
    document: SegDocument
    context_content_digest: str
    manifest_digest: str
    binding: str


@dataclass(frozen=True)
class _AdapterRecord:
    context: IssuedSegmentationContext
    issued: IssuedLegacySegmentation
    result: SegmentationAdapterResult
    snapshot: SegmentationAdapterResult
    projection_inputs: SegmentationProjectionInputs
    legacy: SegmentationDelivery
    boundary: SegmentationDelivery | None
    binding: str


_LEGACY: dict[int, _LegacyRecord] = {}
_ADAPTER: dict[int, _AdapterRecord] = {}
_LOCK = threading.RLock()


class SegmentationProductionError(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


def _object(value: object) -> FrozenObject:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenObject):
        raise TypeError("segmentation value is not an object")
    return frozen


def _manifest_value(delivery: SegmentationDelivery) -> dict[str, object]:
    value = thaw_json(delivery.manifest)
    if not isinstance(value, dict):
        raise ValueError("segmentation manifest is not an object")
    return value


def _delivery_value(delivery: SegmentationDelivery) -> dict[str, object]:
    return {
        "context_content_digest": delivery.context_content_digest,
        "engine_family": delivery.engine_family,
        "language": delivery.language,
        "cues": [
            {
                "unit_range": list(cue.unit_range),
                "text": cue.text,
                "start": cue.start,
                "end": cue.end,
                "word_data": [thaw_json(unit) for unit in cue.word_data],
                "speech_start": cue.speech_start,
                "speech_end": cue.speech_end,
                "lyric": cue.lyric,
                "speaker_ids": None
                if cue.speaker_ids is None
                else list(cue.speaker_ids),
            }
            for cue in delivery.cues
        ],
        "top_level_word_segments": [
            thaw_json(unit) for unit in delivery.top_level_word_segments
        ],
        "carriers": {
            "vad_speech": [thaw_json(row) for row in delivery.carriers.vad_speech],
            "shot_changes": None
            if delivery.carriers.shot_changes is None
            else [thaw_json(row) for row in delivery.carriers.shot_changes],
            "sing_spans": None
            if delivery.carriers.sing_spans is None
            else [thaw_json(row) for row in delivery.carriers.sing_spans],
            "speaker_turns": {
                "present": delivery.carriers.speaker_turns.present,
                "value": None
                if delivery.carriers.speaker_turns.value is None
                else thaw_json(delivery.carriers.speaker_turns.value),
            },
            "voiceprint_capture": delivery.carriers.voiceprint_capture,
            "voiceprint_media": delivery.carriers.voiceprint_media,
        },
        "manifest": _manifest_value(delivery),
    }


def segmentation_delivery_digest(delivery: SegmentationDelivery) -> str:
    return frozen_json_digest(freeze_json(_delivery_value(delivery)))


def _validate_ranges(delivery: SegmentationDelivery, unit_count: int) -> None:
    ranges = tuple(cue.unit_range for cue in delivery.cues)
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != unit_count:
        raise ValueError("segmentation delivery does not cover its unit stream")
    if any(low < 0 or high <= low or high > unit_count for low, high in ranges) or any(
        left[1] != right[0] for left, right in zip(ranges, ranges[1:])
    ):
        raise ValueError("segmentation delivery ranges are not a contiguous tiling")


def _validate_delivery(
    context: IssuedSegmentationContext,
    delivery: SegmentationDelivery,
    projection_inputs: SegmentationProjectionInputs,
    document: SegDocument,
) -> None:
    manifest = _manifest_value(delivery)
    if (
        delivery.context_content_digest != context.context_content_digest
        or delivery.engine_family != "legacy-v1"
        or delivery.language != context.effective_iso
        or document.language != context.effective_iso
        or manifest.get("engine") != MANIFEST_ENGINE_BY_FAMILY["legacy-v1"]
        or type(projection_inputs.timestamps) is not bool
        or len(dict(projection_inputs.speaker_names))
        != len(projection_inputs.speaker_names)
    ):
        raise ValueError("legacy segmentation binding is invalid")
    if (delivery.carriers.voiceprint_capture is None) != (
        delivery.carriers.voiceprint_media is None
    ):
        raise ValueError("segmentation voiceprint carriers must be a pair")
    if len(delivery.top_level_word_segments) != len(document.units):
        raise ValueError("segmentation top-level stream cardinality changed")
    for frozen, unit in zip(delivery.top_level_word_segments, document.units):
        value = thaw_json(frozen)
        # The persisted top-level stream is the snapped source evidence. The
        # SegDocument intentionally carries the separately repaired timing view
        # consumed by cue formation, so only cardinality and surface identity
        # are common authority here.
        if (
            not isinstance(value, dict)
            or value.get("text", value.get("word", "")) != unit.surface
        ):
            raise ValueError("segmentation top-level surface stream changed")
    # The historical delivery may contain proportional coarse-unit cues whose
    # word-data footprint cannot name a source-unit partition. Only a v2
    # delivery claims the increasing contiguous unit-range authority.
    if delivery.engine_family == "boundary-v2":
        _validate_ranges(delivery, len(document.units))


def issue_legacy_segmentation(
    context: IssuedSegmentationContext,
    *,
    delivery: SegmentationDelivery,
    projection_inputs: SegmentationProjectionInputs,
    document: SegDocument,
) -> IssuedLegacySegmentation:
    """Seal the already-produced historical delivery and its v2 source document."""
    _validate_delivery(context, delivery, projection_inputs, document)
    sealed_document = copy.deepcopy(document)
    sealed_document.manifest = copy.deepcopy(_manifest_value(delivery))
    provider_ledger = freeze_json(
        {
            "providers": sealed_document.manifest.get("providers", {}),
            "degraded": sealed_document.manifest.get("degraded", []),
        }
    )
    issued = object.__new__(IssuedLegacySegmentation)
    object.__setattr__(issued, "context_content_digest", context.context_content_digest)
    object.__setattr__(issued, "delivery", delivery)
    object.__setattr__(issued, "provider_ledger", provider_ledger)
    object.__setattr__(issued, "manifest_digest", frozen_json_digest(delivery.manifest))
    object.__setattr__(issued, "_binding", secrets.token_hex(32))
    with _LOCK:
        _LEGACY[id(issued)] = _LegacyRecord(
            context,
            issued,
            delivery,
            copy.deepcopy(delivery),
            copy.deepcopy(provider_ledger),
            projection_inputs,
            copy.deepcopy(projection_inputs),
            sealed_document,
            issued.context_content_digest,
            issued.manifest_digest,
            issued._binding,
        )
    return issued


def _legacy_record(
    context: IssuedSegmentationContext,
    issued: IssuedLegacySegmentation,
) -> _LegacyRecord:
    with _LOCK:
        record = _LEGACY.get(id(issued))
        if (
            record is None
            or record.context is not context
            or record.issued is not issued
            or issued.delivery is not record.delivery
            or issued.delivery != record.delivery_snapshot
            or issued.provider_ledger != record.provider_ledger_snapshot
            or issued.context_content_digest != record.context_content_digest
            or issued.context_content_digest != context.context_content_digest
            or issued.manifest_digest != record.manifest_digest
            or issued.manifest_digest != frozen_json_digest(issued.delivery.manifest)
            or issued._binding != record.binding
            or record.projection_inputs != record.projection_snapshot
        ):
            raise ValueError(
                "legacy segmentation is unissued, changed, or cross-context"
            )
        return record


def _failure(kind: str, phase: str, detail: str) -> SegmentationProductionError:
    return SegmentationProductionError(CanonicalFailure(kind, phase, detail))


def _freeze_cue(cue: Cue, unit_range: tuple[int, int]) -> SegmentationDeliveryCue:
    word_data: list[FrozenObject] = []
    for unit in cue.get("word_data", ()):
        word_data.append(_object(copy.deepcopy(dict(unit))))
    speaker_ids_value = cue.get("speaker_ids")
    speaker_ids = (
        None
        if speaker_ids_value is None
        else tuple(str(value) for value in speaker_ids_value)
    )
    speech_start = cue.get("speech_start")
    speech_end = cue.get("speech_end")
    return SegmentationDeliveryCue(
        unit_range,
        str(cue["text"]),
        float(cue["start"]),
        float(cue["end"]),
        tuple(word_data),
        None if speech_start is None else float(speech_start),
        None if speech_end is None else float(speech_end),
        True if cue.get("lyric") is True else None,
        speaker_ids,
    )


def _build_boundary_delivery(record: _LegacyRecord) -> SegmentationDelivery:
    from voxweave.core.authority import AuthorityLedger, check_roots
    from voxweave.core.boundary_v2 import optimize_document
    from voxweave.core.finalizer import (
        FinalizeEvidence,
        FinalizePolicy,
        FinalizerPreview,
        finalize,
        phase1_from_optimizer_selection,
        register_optimizer_selection,
    )
    from voxweave.core.partition_check import check_partition, owned_unit_ids
    from voxweave.core.policy_delta import DELTA_REGISTRY
    from voxweave.core.providers import degradation_capture, provider_snapshot
    from voxweave.core.speaker_evidence import (
        W_SPEAKER_INTERIOR,
        annotate_speaker_ids,
        named_multi_cues_unannotated,
        project_speaker_evidence,
        speaker_evidence,
    )
    from voxweave.core.subunit import refine_document
    from voxweave.core.trace_validator import replay_trace, stability_check

    document = copy.deepcopy(record.document)
    parent_speakers = speaker_evidence(document)
    with degradation_capture(quiet=True) as degraded:
        refined, refinement = refine_document(document)
        projected_speakers = project_speaker_evidence(
            parent_speakers,
            refined_units=refinement.units,
            origin=refinement.origin,
        )
        solution = optimize_document(
            refined,
            preview=FinalizerPreview(refined.profile),
            subunit_split=refinement,
            speakers=projected_speakers,
            speaker_weight=W_SPEAKER_INTERIOR,
        )
        if solution.invalid_profile:
            raise _failure("profile-invalid", "segmentation-adapter", "profile-domain")
        ledger = AuthorityLedger()
        try:
            authority = register_optimizer_selection(solution, ledger=ledger)
        except ValueError as exc:
            raise _failure(
                "segmentation-v2-invalid",
                "segmentation-adapter",
                "delivery-unit-range",
            ) from exc
        stream = phase1_from_optimizer_selection(
            authority,
            ledger=ledger,
            row_id="segmentation-production/v2",
            evaluation_id=record.context.context_content_digest,
        )
        evidence = FinalizeEvidence(
            shots=tuple(refined.shot_changes or ()),
            sing_spans=tuple(refined.sing_spans or ()),
        )
        policy = FinalizePolicy()
        result = finalize(
            stream,
            profile=refined.profile,
            evidence=evidence,
            policy=policy,
        )
    roots = check_roots(
        ledger,
        expected={"segmentation-production/v2": "optimizer-selection"},
    )
    if roots:
        raise _failure(
            "finalizer-output-invalid", "segmentation-adapter", "phase1-fidelity"
        )
    if not result.valid:
        raise _failure(
            "finalizer-budget-exhausted", "segmentation-adapter", "sweep-budget"
        )
    cues = [cast("Cue", copy.deepcopy(dict(cue))) for cue in result.cues]
    delivered = tuple((float(cue["start"]), float(cue["end"])) for cue in cues)
    trace_errors = replay_trace(
        result.trace,
        stream.cues,
        profile=refined.profile,
        evidence=evidence,
        policy=policy,
        delivered=delivered,
    )
    if trace_errors:
        raise _failure("finalizer-trace-failed", "segmentation-adapter", "trace-replay")
    stability_errors = stability_check(
        delivered,
        stream.cues,
        profile=refined.profile,
        evidence=evidence,
        policy=policy,
        terminal=result.trace.terminal,
    )
    if stability_errors:
        raise _failure(
            "finalizer-stability-failed", "segmentation-adapter", "stability-check"
        )
    partition = authority.partition
    ranges = owned_unit_ids(partition, len(refined.units))
    if len(ranges) != len(cues):
        raise _failure(
            "segmentation-v2-invalid",
            "segmentation-adapter",
            "delivery-unit-range",
        )
    partition_result = check_partition(
        partition,
        cues,
        units=refined.units,
        profile=refined.profile,
        origin="v2",
        stage="finalizer",
        reports=result.report.entries,
        waivers={waiver.cue_index: waiver for waiver in result.report.waivers},
    )
    if partition_result.exit_driving:
        raise _failure(
            "finalizer-partition-failed",
            "segmentation-adapter",
            "finalizer-partition",
        )
    delta_ids = {record.id for record in DELTA_REGISTRY}
    if any(delta not in delta_ids for delta in result.report.deltas_fired):
        raise _failure("align-delta-invalid", "segmentation-adapter", "trigger-set")
    annotate_speaker_ids(cues, ranges, projected_speakers.unit_speakers)
    if named_multi_cues_unannotated(ranges, projected_speakers.unit_speakers):
        raise _failure(
            "segmentation-v2-invalid",
            "segmentation-adapter",
            "named-multi-unattributed",
        )
    manifest = copy.deepcopy(_manifest_value(record.issued.delivery))
    manifest["engine"] = MANIFEST_ENGINE_BY_FAMILY["boundary-v2"]
    manifest["providers"] = provider_snapshot(refined.language)
    manifest["degraded"] = copy.deepcopy(degraded)
    return SegmentationDelivery(
        record.context.context_content_digest,
        "boundary-v2",
        refined.language,
        tuple(_freeze_cue(cue, unit_range) for cue, unit_range in zip(cues, ranges)),
        record.issued.delivery.top_level_word_segments,
        record.issued.delivery.carriers,
        _object(manifest),
    )


def run_locked_segmentation_adapter(
    context: IssuedSegmentationContext,
    issued: IssuedLegacySegmentation,
    *,
    shadow_enabled: bool,
) -> SegmentationAdapterResult:
    """Consume one adapter role and produce immutable legacy/v2 delivery status."""
    if type(shadow_enabled) is not bool:
        raise TypeError("segmentation adapter switch must be an exact bool")
    record = _legacy_record(context, issued)
    consume_context_role(
        context,
        "adapter",
        consumer="run_locked_segmentation_adapter",
    )
    boundary: SegmentationDelivery | None = None
    if not shadow_enabled:
        status = V2Status("not-requested", None)
    else:
        try:
            boundary = _build_boundary_delivery(record)
        except SegmentationProductionError as exc:
            status = V2Status("invalid", exc.failure)
        except Exception:
            status = V2Status(
                "invalid",
                CanonicalFailure(
                    "shadow-internal-error",
                    "segmentation-adapter",
                    "w1-stage",
                ),
            )
        else:
            status = V2Status("valid", None)
    result = object.__new__(SegmentationAdapterResult)
    object.__setattr__(result, "context_content_digest", context.context_content_digest)
    object.__setattr__(result, "legacy", issued.delivery)
    object.__setattr__(result, "v2", boundary)
    object.__setattr__(result, "v2_status", status)
    object.__setattr__(result, "_binding", secrets.token_hex(32))
    with _LOCK:
        _ADAPTER[id(result)] = _AdapterRecord(
            context,
            issued,
            result,
            copy.deepcopy(result),
            record.projection_inputs,
            result.legacy,
            result.v2,
            result._binding,
        )
    return result


def _adapter_record(
    context: IssuedSegmentationContext,
    result: SegmentationAdapterResult,
) -> _AdapterRecord:
    with _LOCK:
        record = _ADAPTER.get(id(result))
        if (
            record is None
            or record.context is not context
            or record.result is not result
            or result != record.snapshot
            or result.legacy is not record.legacy
            or result.v2 is not record.boundary
            or result._binding != record.binding
            or result.context_content_digest != context.context_content_digest
        ):
            raise ValueError("segmentation adapter result is unissued or changed")
        _legacy_record(context, record.issued)
        return record


__all__ = [
    "IssuedLegacySegmentation",
    "SegmentationAdapterResult",
    "SegmentationCarriers",
    "SegmentationDelivery",
    "SegmentationDeliveryCue",
    "SegmentationProductionError",
    "SegmentationProjectionInputs",
    "issue_legacy_segmentation",
    "run_locked_segmentation_adapter",
    "segmentation_delivery_digest",
]
