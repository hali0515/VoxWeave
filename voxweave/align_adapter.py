"""Context-bound AO-14/AO-15 align adapter and evaluated-result issuers."""

from __future__ import annotations

import math
import threading
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal

from voxweave.align_context import (
    IssuedAlignContext,
    consume_context_role,
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
    speaker_turns: object | None
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
class AlignAdapterResult:
    context_content_digest: str
    receipt_digest: str
    legacy: AlignDelivery
    v2: AlignDelivery | None
    v2_status: V2Status
    _binding: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("AlignAdapterResult is issuer-only")


@dataclass(frozen=True, init=False)
class AlignEvaluatedResult:
    context_content_digest: str
    receipt_digest: str
    legacy: AlignDelivery
    v2: AlignDelivery | None
    v2_status: V2Status
    evidence_core: EvidenceCore
    comparison: object | None
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
    v2_snapshot: AlignDelivery | None
    status_snapshot: V2Status
    comparison_snapshot: object | None
    semantic_observation: _SemanticObservation | None
    nonce: str


@dataclass(frozen=True)
class _AdapterRecord:
    context: IssuedAlignContext
    acquisition: object
    result: AlignAdapterResult
    projection_inputs: AlignProjectionInputs
    result_snapshot: AlignAdapterResult
    semantic_observation: _SemanticObservation | None


@dataclass(frozen=True)
class _SemanticObservation:
    semantic_root_lineage: object
    phase1_seed: object
    delivered: object
    report: object
    trace: object
    partition_result: object | None
    trace_problems: object | None
    stability_problems: object | None


class AlignAdapterError(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


_EVALUATED: dict[int, _EvaluatedRecord] = {}
_ADAPTERS: dict[int, _AdapterRecord] = {}
_LOCK = threading.RLock()


def _pre_w1_failure(context: IssuedAlignContext, acquisition: object, payload: object):
    from voxweave.align_acquisition import IssuedFreshAlignment, _fresh_seed
    from voxweave.align_distribution import project_authority_failure
    from voxweave.align_inputs import (
        FinalizerEvidenceResolution,
        ProfileResolution,
        V2PolicyStatus,
    )
    from voxweave.align_snapshot import StrictInputStatus

    if not isinstance(acquisition, IssuedFreshAlignment):
        return CanonicalFailure(
            "fresh-authority-invalid", "adapter-binding", "acquisition-unissued"
        )
    strict = getattr(payload, "strict_input_status", None)
    policy = getattr(payload, "v2_policy_status", None)
    profile = getattr(payload, "profile_resolution", None)
    evidence = getattr(payload, "evidence_resolution", None)
    unexpected_profile_failure = getattr(profile, "unexpected_failure", None)
    if isinstance(unexpected_profile_failure, CanonicalFailure):
        return unexpected_profile_failure
    if not isinstance(strict, StrictInputStatus):
        return CanonicalFailure(
            "v2-input-invalid", "strict-input", "strict-carrier-domain"
        )
    if strict.kind == "invalid":
        return CanonicalFailure(
            "v2-input-invalid",
            "strict-input",
            strict.detail_code or "strict-carrier-domain",
        )
    authority_failure = project_authority_failure(acquisition.distribution)
    if authority_failure is not None:
        return authority_failure
    seed = _fresh_seed(context, acquisition)
    seed_status = getattr(seed, "status", None)
    seed_reasons = tuple(getattr(seed, "reasons", ()))
    if "footprint-reconciliation" in seed_reasons:
        return CanonicalFailure(
            "fresh-reconciliation-invalid",
            "seed-reconciliation",
            "footprint-reconciliation",
        )
    if seed_status != "valid":
        return CanonicalFailure(
            "fresh-seed-invalid", "seed-admission", "seed-admission"
        )
    if not isinstance(policy, V2PolicyStatus):
        return CanonicalFailure("v2-policy-invalid", "v2-policy", "nonfinite-policy")
    if policy.kind == "invalid":
        return CanonicalFailure(
            "v2-policy-invalid",
            "v2-policy",
            policy.detail_code or "nonfinite-policy",
        )
    if not isinstance(profile, ProfileResolution) or profile.status.kind == "invalid":
        detail = (
            "profile-shape"
            if not isinstance(profile, ProfileResolution)
            else profile.status.detail_code or "profile-shape"
        )
        return CanonicalFailure("profile-invalid", "display-profile", detail)
    if (
        not isinstance(evidence, FinalizerEvidenceResolution)
        or evidence.status.kind == "invalid"
    ):
        detail = (
            "evidence-domain"
            if not isinstance(evidence, FinalizerEvidenceResolution)
            else evidence.status.detail_code or "evidence-domain"
        )
        return CanonicalFailure("evidence-invalid", "finalizer-evidence", detail)
    return None


def _adapter_result(
    context: IssuedAlignContext,
    acquisition: object,
    *,
    legacy: AlignDelivery,
    v2: AlignDelivery | None,
    status: V2Status,
    projection_inputs: AlignProjectionInputs,
    semantic_observation: _SemanticObservation | None = None,
) -> AlignAdapterResult:
    from voxweave.align_acquisition import IssuedFreshAlignment

    if not isinstance(acquisition, IssuedFreshAlignment):
        raise ValueError("align adapter acquisition is unissued")
    result = object.__new__(AlignAdapterResult)
    object.__setattr__(result, "context_content_digest", context.context_content_digest)
    object.__setattr__(result, "receipt_digest", acquisition.receipt_digest)
    object.__setattr__(result, "legacy", legacy)
    object.__setattr__(result, "v2", v2)
    object.__setattr__(result, "v2_status", status)
    object.__setattr__(result, "_binding", __import__("secrets").token_hex(32))
    with _LOCK:
        _ADAPTERS[id(result)] = _AdapterRecord(
            context,
            acquisition,
            result,
            projection_inputs,
            deepcopy(result),
            deepcopy(semantic_observation),
        )
    return result


def _adapter_record(
    context: IssuedAlignContext, result: AlignAdapterResult
) -> _AdapterRecord:
    with _LOCK:
        record = _ADAPTERS.get(id(result))
        if (
            record is None
            or record.context is not context
            or record.result is not result
            or result != record.result_snapshot
            or result.context_content_digest != context.context_content_digest
        ):
            raise ValueError(
                "align adapter result is unissued, changed, or cross-context"
            )
        return record


def _adapter_semantic_observation(
    context: IssuedAlignContext, result: AlignAdapterResult
) -> _SemanticObservation | None:
    """Return the private atomic AO-15 observation for AO-17 or shadow use."""
    return deepcopy(_adapter_record(context, result).semantic_observation)


def _w1_delivery(
    context: IssuedAlignContext,
    acquisition: object,
    payload: object,
    *,
    _observation_sink: list[_SemanticObservation] | None = None,
) -> tuple[AlignDelivery, _SemanticObservation]:
    from typing import cast

    from voxweave.align_acquisition import (
        IssuedFreshAlignment,
        _fresh_seed,
        _verify_fresh_alignment,
    )
    from voxweave.core.authority import AuthorityLedger, check_roots, lineage_tuples
    from voxweave.core.finalizer import (
        FinalizeEvidence,
        FinalizePolicy,
        FinalizerPreview,
        finalize,
        phase1_from_fresh_alignment,
    )
    from voxweave.core.partition_check import check_partition
    from voxweave.core.schema import Cue
    from voxweave.core.segdoc import SourceUnit
    from voxweave.core.trace_validator import replay_trace, stability_check
    from voxweave.core.timing_preview import CueCandidate

    if not isinstance(acquisition, IssuedFreshAlignment):
        raise AlignAdapterError(
            CanonicalFailure(
                "fresh-authority-invalid", "w1-admission", "acquisition-unissued"
            )
        )
    verified = _verify_fresh_alignment(context, acquisition)
    if verified is None:
        raise AlignAdapterError(
            CanonicalFailure("fresh-authority-invalid", "w1-admission", "root-transfer")
        )
    profile_resolution = getattr(payload, "profile_resolution")
    evidence_resolution = getattr(payload, "evidence_resolution")
    profile = profile_resolution.profile
    if profile is None:
        raise AlignAdapterError(
            CanonicalFailure("profile-invalid", "display-profile", "profile-shape")
        )
    ledger = AuthorityLedger()
    stream = phase1_from_fresh_alignment(
        verified,
        profile=profile,
        ledger=ledger,
        row_id="align/delivery-finalizer/v2",
        evaluation_id=context.context_content_digest,
    )
    seed = _fresh_seed(context, acquisition)
    seed_blocks = getattr(seed, "blocks", None)
    seed_cues = tuple(getattr(stream, "cues", ()))
    if seed_blocks is None or len(seed_blocks) != len(seed_cues):
        raise AlignAdapterError(
            CanonicalFailure(
                "finalizer-output-invalid", "w1-finalizer", "phase1-fidelity"
            )
        )
    preview = FinalizerPreview()

    def same_bound(left: float | None, right: float | None) -> bool:
        if left is None or right is None:
            return left is right
        return left.hex() == right.hex()

    for block, phase1 in zip(seed_blocks, seed_cues, strict=True):
        candidate = CueCandidate(
            block.display_start,
            block.display_end,
            None,
            block.text,
            tuple(
                {
                    "text": unit.surface,
                    "start": unit.start,
                    "end": unit.end,
                }
                for unit in block.units
            ),
            block.speech_start,
            block.speech_end,
            profile,
            block.footprint,
        )
        promised = preview.preview_cue(candidate)
        if (
            promised.final_text != phase1.text
            or promised.line_count != len(phase1.lines)
            or promised.reading_chars != phase1.reading_chars
            or not same_bound(promised.display_start, phase1.start)
            or not same_bound(promised.display_end, phase1.end)
        ):
            raise AlignAdapterError(
                CanonicalFailure(
                    "finalizer-output-invalid", "w1-finalizer", "phase1-fidelity"
                )
            )
    finalizer_evidence = FinalizeEvidence(
        shots=tuple(evidence_resolution.shots or ()),
        sing_spans=tuple(evidence_resolution.sing_spans or ()),
    )
    policy = FinalizePolicy()
    semantic_root_lineage = lineage_tuples(ledger)
    finalized = finalize(
        stream,
        profile=profile,
        evidence=finalizer_evidence,
        policy=policy,
    )
    semantic_observation = _SemanticObservation(
        semantic_root_lineage,
        deepcopy(stream.cues),
        deepcopy(finalized.cues),
        deepcopy(finalized.report),
        deepcopy(finalized.trace),
        None,
        None,
        None,
    )

    def retain_observation() -> None:
        if _observation_sink is not None:
            _observation_sink[:] = [semantic_observation]

    retain_observation()
    root_errors = check_roots(
        ledger,
        expected={"align/delivery-finalizer/v2": "fresh-alignment"},
    )
    if root_errors:
        raise AlignAdapterError(
            CanonicalFailure("fresh-authority-invalid", "w1-admission", "w1-root-event")
        )
    terminal = getattr(finalized.report, "terminal", None)
    trace_terminal = getattr(finalized.trace, "terminal", None)
    if terminal != trace_terminal:
        raise AlignAdapterError(
            CanonicalFailure(
                "finalizer-output-invalid", "w1-finalizer", "terminal-validity"
            )
        )
    if terminal == "budget-exhausted":
        if finalized.valid is not False:
            raise AlignAdapterError(
                CanonicalFailure(
                    "finalizer-output-invalid",
                    "w1-finalizer",
                    "terminal-validity",
                )
            )
        raise AlignAdapterError(
            CanonicalFailure(
                "finalizer-budget-exhausted", "w1-finalizer", "sweep-budget"
            )
        )
    if terminal not in ("fixed-point", "cycle-adoption") or finalized.valid is not True:
        raise AlignAdapterError(
            CanonicalFailure(
                "finalizer-output-invalid", "w1-finalizer", "terminal-validity"
            )
        )
    if any(
        getattr(report, "kind", None) == "canonical-text-fallback"
        for cue in seed_cues
        for report in getattr(cue, "reports", ())
    ) or any(
        getattr(cue, "unit_range", None) != block.unit_range
        for block, cue in zip(seed_blocks, seed_cues, strict=True)
    ):
        raise AlignAdapterError(
            CanonicalFailure(
                "finalizer-output-invalid", "w1-finalizer", "footprint-fallback"
            )
        )
    ordered_units = tuple(getattr(seed, "ordered_units", ()))
    if seed_blocks is None or len(seed_blocks) != len(finalized.cues):
        raise AlignAdapterError(
            CanonicalFailure("finalizer-output-invalid", "w1-finalizer", "cue-schema")
        )
    for cue in finalized.cues:
        if (
            not isinstance(cue, dict)
            or type(cue.get("text")) is not str
            or type(cue.get("start")) is not float
            or type(cue.get("end")) is not float
            or not isinstance(cue.get("word_data"), list)
            or not math.isfinite(cue["start"])
            or not math.isfinite(cue["end"])
            or cue["start"] < 0.0
            or cue["start"] > cue["end"]
            or ("lyric" in cue and type(cue["lyric"]) is not bool)
        ):
            raise AlignAdapterError(
                CanonicalFailure(
                    "finalizer-output-invalid", "w1-finalizer", "cue-schema"
                )
            )
    delivered = tuple((cue["start"], cue["end"]) for cue in finalized.cues)
    sources = tuple(
        SourceUnit(
            unit.unit_id,
            unit.surface,
            unit.start,
            unit.end,
            unit.provenance,
            None,
        )
        for unit in ordered_units
    )
    partition = tuple(block.unit_range[1] for block in seed_blocks[:-1])
    waivers = {waiver.cue_index: waiver for waiver in finalized.report.waivers}
    partition_result = check_partition(
        partition,
        cast("tuple[Cue, ...]", finalized.cues),
        units=sources,
        profile=profile,
        origin="v2",
        stage="finalizer",
        reports=finalized.report.entries,
        waivers=waivers,
    )
    semantic_observation = replace(
        semantic_observation,
        partition_result=deepcopy(partition_result),
    )
    retain_observation()
    if partition_result.exit_driving:
        raise AlignAdapterError(
            CanonicalFailure(
                "finalizer-partition-failed",
                "finalizer-partition",
                "finalizer-partition",
            )
        )
    trace_errors = replay_trace(
        finalized.trace,
        stream.cues,
        profile=profile,
        evidence=finalizer_evidence,
        policy=policy,
        delivered=delivered,
    )
    semantic_observation = replace(
        semantic_observation,
        trace_problems=deepcopy(trace_errors),
    )
    retain_observation()
    if trace_errors:
        raise AlignAdapterError(
            CanonicalFailure("finalizer-trace-failed", "trace-replay", "trace-replay")
        )
    stability_errors = stability_check(
        delivered,
        stream.cues,
        profile=profile,
        evidence=finalizer_evidence,
        policy=policy,
        terminal=finalized.trace.terminal,
    )
    semantic_observation = replace(
        semantic_observation,
        stability_problems=deepcopy(stability_errors),
    )
    retain_observation()
    if stability_errors:
        raise AlignAdapterError(
            CanonicalFailure(
                "finalizer-stability-failed",
                "stability-check",
                "stability-check",
            )
        )
    delivery_cues = tuple(
        AlignDeliveryCue(
            block.source_index,
            str(cue["text"]),
            float(cue["start"]),
            float(cue["end"]),
            True if cue.get("lyric") is True else None,
            block.owner_unit_ids,
            tuple(
                PersistedAlignUnit(unit.surface, float(unit.start), float(unit.end))
                for unit in block.units
                if unit.start is not None and unit.end is not None
            ),
            block.speech_start,
            block.speech_end,
        )
        for block, cue in zip(seed_blocks, finalized.cues, strict=True)
    )
    delivery = AlignDelivery(
        context.context_content_digest,
        acquisition.receipt_digest,
        "boundary-v2",
        context.route_kind,
        delivery_cues,
        tuple(
            PersistedAlignUnit(unit.surface, float(unit.start), float(unit.end))
            for unit in ordered_units
            if unit.start is not None and unit.end is not None
        ),
    )
    return delivery, semantic_observation


def run_locked_align_adapter(
    context: IssuedAlignContext,
    acquisition: object,
    *,
    shadow_enabled: bool,
) -> AlignAdapterResult:
    """Complete AO-14 and the applicable AO-15 W1 attempt exactly once."""
    from voxweave.align_acquisition import (
        IssuedFreshAlignment,
        _fresh_adapter_payload,
        _fresh_record,
        _retire_fresh_transfer,
    )

    if type(shadow_enabled) is not bool:
        raise TypeError("shadow_enabled must be an exact bool")
    if not isinstance(acquisition, IssuedFreshAlignment):
        raise AlignAdapterError(
            CanonicalFailure(
                "fresh-authority-invalid", "adapter-binding", "acquisition-unissued"
            )
        )
    _fresh_record(context, acquisition)
    payload = _fresh_adapter_payload(context, acquisition)
    legacy = payload.legacy_delivery
    projection_inputs = payload.projection_inputs
    if not isinstance(legacy, AlignDelivery) or not isinstance(
        projection_inputs, AlignProjectionInputs
    ):
        raise AlignAdapterError(
            CanonicalFailure(
                "fresh-authority-invalid", "adapter-binding", "receipt-context"
            )
        )
    if (
        legacy.context_content_digest != context.context_content_digest
        or legacy.receipt_digest != acquisition.receipt_digest
        or legacy.engine_family != "legacy-v1"
        or legacy.route_kind != context.route_kind
        or projection_inputs.language != context.effective_iso
    ):
        raise AlignAdapterError(
            CanonicalFailure(
                "fresh-authority-invalid", "adapter-binding", "receipt-context"
            )
        )
    consume_context_role(context, "adapter", consumer="run_locked_align_adapter")
    requested = shadow_enabled or context.engine_family == "boundary-v2"
    if not requested:
        _retire_fresh_transfer(context, acquisition)
        return _adapter_result(
            context,
            acquisition,
            legacy=legacy,
            v2=None,
            status=V2Status("not-requested", None),
            projection_inputs=projection_inputs,
        )
    failure = _pre_w1_failure(context, acquisition, payload)
    if failure is not None:
        _retire_fresh_transfer(context, acquisition)
        return _adapter_result(
            context,
            acquisition,
            legacy=legacy,
            v2=None,
            status=V2Status("invalid", failure),
            projection_inputs=projection_inputs,
        )
    retained_observation: list[_SemanticObservation] = []
    try:
        v2, semantic_observation = _w1_delivery(
            context,
            acquisition,
            payload,
            _observation_sink=retained_observation,
        )
    except AlignAdapterError as exc:
        _retire_fresh_transfer(context, acquisition)
        if context.engine_family == "boundary-v2":
            raise
        return _adapter_result(
            context,
            acquisition,
            legacy=legacy,
            v2=None,
            status=V2Status("invalid", exc.failure),
            projection_inputs=projection_inputs,
            semantic_observation=(
                retained_observation[0] if retained_observation else None
            ),
        )
    except Exception as exc:
        _retire_fresh_transfer(context, acquisition)
        failure = CanonicalFailure("shadow-internal-error", "w1-stage", "w1-stage")
        if context.engine_family == "boundary-v2":
            raise AlignAdapterError(failure) from exc
        return _adapter_result(
            context,
            acquisition,
            legacy=legacy,
            v2=None,
            status=V2Status("invalid", failure),
            projection_inputs=projection_inputs,
            semantic_observation=(
                retained_observation[0] if retained_observation else None
            ),
        )
    _fresh_record(context, acquisition)
    return _adapter_result(
        context,
        acquisition,
        legacy=legacy,
        v2=v2,
        status=V2Status("valid", None),
        projection_inputs=projection_inputs,
        semantic_observation=semantic_observation,
    )


def issue_align_evaluated_result(
    context: IssuedAlignContext,
    adapter_result: AlignAdapterResult,
    *,
    evidence_core: EvidenceCore,
    comparison: object | None,
) -> AlignEvaluatedResult:
    """Fold genuine AO-15, AO-16, and nullable AO-17 facts at AO-18."""
    record = _adapter_record(context, adapter_result)
    if not isinstance(evidence_core, EvidenceCore):
        raise TypeError("evidence_core must be an EvidenceCore")
    evidence_receipt = getattr(
        evidence_core, "receipt_digest", evidence_core.core_digest
    )
    if (
        evidence_core.context_content_digest != context.context_content_digest
        or adapter_result.receipt_digest != evidence_receipt
    ):
        raise ValueError("evaluated evidence is not bound to the adapter receipt")

    status = adapter_result.v2_status
    if isinstance(comparison, CanonicalFailure):
        if (
            status.kind != "valid"
            or comparison.kind != "shadow-internal-error"
            or comparison.phase != "comparator-stage"
            or comparison.detail_code != "comparator-stage"
        ):
            raise ValueError("unexpected comparator terminal is not closed")
        status = V2Status("invalid", comparison)
        comparison = None
    elif status.kind == "valid":
        from voxweave.align_delta_registry import ALIGN_DELTA_REGISTRY_SHA256
        from voxweave.core.align_compare import AlignComparison

        if adapter_result.v2 is None or not isinstance(comparison, AlignComparison):
            raise ValueError("valid v2 delivery requires semantic comparison")
        if comparison.registry_sha256 != ALIGN_DELTA_REGISTRY_SHA256:
            status = V2Status(
                "invalid",
                CanonicalFailure(
                    "align-delta-invalid",
                    "semantic-comparison",
                    "registry-digest",
                ),
            )
        elif comparison.violations:
            status = V2Status(
                "invalid",
                CanonicalFailure(
                    "align-delta-invalid",
                    "semantic-comparison",
                    "primitive-relation",
                ),
            )
    elif comparison is not None:
        raise ValueError("semantic comparison is forbidden without valid v2 delivery")

    if context.engine_family == "boundary-v2" and status.kind != "valid":
        failure = status.failure or CanonicalFailure(
            "fresh-authority-invalid", "v2-admission", "w1-root-event"
        )
        raise AlignAdapterError(failure)

    import secrets

    result = object.__new__(AlignEvaluatedResult)
    object.__setattr__(result, "context_content_digest", context.context_content_digest)
    object.__setattr__(result, "receipt_digest", adapter_result.receipt_digest)
    object.__setattr__(result, "legacy", adapter_result.legacy)
    object.__setattr__(result, "v2", adapter_result.v2)
    object.__setattr__(result, "v2_status", status)
    object.__setattr__(result, "evidence_core", evidence_core)
    object.__setattr__(result, "comparison", comparison)
    object.__setattr__(result, "_issuance_nonce", secrets.token_hex(32))
    with _LOCK:
        _EVALUATED[id(result)] = _EvaluatedRecord(
            context,
            result,
            record.projection_inputs,
            adapter_result.legacy,
            deepcopy(adapter_result.legacy),
            deepcopy(record.projection_inputs),
            evidence_core,
            deepcopy(evidence_core),
            deepcopy(adapter_result.v2),
            deepcopy(status),
            deepcopy(comparison),
            deepcopy(record.semantic_observation),
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
            or result.v2 != record.v2_snapshot
            or result.v2_status != record.status_snapshot
            or result.comparison != record.comparison_snapshot
            or result.evidence_core != record.evidence_snapshot
            or record.projection_inputs != record.projection_snapshot
            or result._issuance_nonce != record.nonce
        ):
            raise ValueError("align evaluated result stable binding changed")
        return record


def _align_semantic_observation(
    context: IssuedAlignContext, result: AlignEvaluatedResult
) -> _SemanticObservation | None:
    """Return the complete private post-AO-15 group for rich observation only."""
    return deepcopy(_evaluated_record(context, result).semantic_observation)


__all__ = [
    "AlignAdapterError",
    "AlignAdapterResult",
    "AlignDelivery",
    "AlignDeliveryCue",
    "AlignEvaluatedResult",
    "AlignProjectionInputs",
    "PersistedAlignUnit",
    "SourceBlockDecoration",
    "V2Status",
    "issue_align_evaluated_result",
    "run_locked_align_adapter",
]
