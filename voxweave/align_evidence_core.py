"""Independent mandatory P6 acquisition-core projection and ALD-6 gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voxweave.align_acquisition import (
    AuthorityTransformResult,
    FreshUnit,
    StrictCaptureResult,
)
from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityDistributionReceipt,
    AuthorityJobWorkReceipt,
    StrictFailureLocator,
)
from voxweave.align_failures import AUTHORITY_REASON_ORDER, SEED_REASON_ORDER
from voxweave.align_snapshot import freeze_json, frozen_json_digest
from voxweave.core.segdoc import SourceUnit
from voxweave.core.subunit import speech_span_units


class EvidenceCoreProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceCoreWord:
    unit_id: str
    call_index: int
    call_unit_index: int
    text: str
    relative_start: float | None
    relative_end: float | None
    physical_origin_seconds: float
    start: float | None
    end: float | None
    provenance: str
    original_relative_start: float | None
    original_relative_end: float | None


@dataclass(frozen=True)
class EvidenceCoreBlock:
    source_index: int
    authority_unit_ids: tuple[str, ...] | None
    word_data: tuple[EvidenceCoreWord, ...] | None
    speech_start: float | None
    speech_end: float | None


@dataclass(frozen=True)
class EvidenceCorePhysicalCall:
    call_index: int
    strict_unit_status: Literal["valid", "invalid"]
    strict_failure: StrictFailureLocator | None
    raw_units_sha256: str | None
    relative_units_sha256: str | None
    authority_transform_status: Literal["valid", "invalid"]
    authority_absolute_sha256: str | None
    raw_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCore:
    schema_version: Literal[8]
    context_content_digest: str
    physical_calls: tuple[EvidenceCorePhysicalCall, ...]
    authority_status: Literal["valid", "invalid"]
    authority_reasons: tuple[str, ...]
    authority_work: AuthorityJobWorkReceipt
    call_surface_chars: tuple[int | None, ...]
    blocks: tuple[EvidenceCoreBlock, ...]
    raw_unit_count: int
    seed_status: Literal["valid", "invalid"]
    seed_reasons: tuple[str, ...]
    core_digest: str


@dataclass(frozen=True)
class ALD6Outcome:
    delta_id: Literal["ALD-6"]
    triggered: Literal[True]
    passed: bool


def _authority_reasons(work: AuthorityJobWorkReceipt) -> tuple[str, ...]:
    if work.status == "seal-mismatch":
        raise EvidenceCoreProjectionError("seal mismatch cannot reach EvidenceCore")
    present: set[str] = set()
    if work.route_status == "invalid":
        if work.route_mismatch is None:
            raise EvidenceCoreProjectionError("invalid route lacks mismatch")
        present.add("route-owner-mismatch")
    elif work.route_mismatch is not None:
        raise EvidenceCoreProjectionError("valid route carries mismatch")
    if work.skipped_blocks:
        present.add("partial-empty-ownership")
    if any(row.strict_failure is not None for row in work.calls):
        present.add("authority-transform-invalid")
    details = {
        "partial-empty-ownership",
        "punctuation-only-block",
        "allocation-no-tiling",
        "allocation-ambiguous",
    }
    for row in work.calls:
        detail = row.allocator.terminal_detail_code
        if detail in details:
            present.add(detail)
    if work.status == "budget-exhausted":
        present.add("allocation-budget-exhausted")
    return tuple(reason for reason in AUTHORITY_REASON_ORDER if reason in present)


def _word(unit: FreshUnit) -> EvidenceCoreWord:
    return EvidenceCoreWord(
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


def _anchors(units: tuple[FreshUnit, ...]) -> tuple[float | None, float | None]:
    source = tuple(
        SourceUnit(
            unit.unit_id,
            unit.surface,
            unit.start,
            unit.end,
            unit.provenance,
            None,
        )
        for unit in units
    )
    return speech_span_units(source)


def _digest_projection(
    *,
    context_content_digest: str,
    calls: tuple[EvidenceCorePhysicalCall, ...],
    authority_status: str,
    authority_reasons: tuple[str, ...],
    work: AuthorityJobWorkReceipt,
    surface_chars: tuple[int | None, ...],
    blocks: tuple[EvidenceCoreBlock, ...],
    raw_unit_count: int,
    seed_status: str,
    seed_reasons: tuple[str, ...],
) -> str:
    value = {
        "schema_version": 8,
        "context_content_digest": context_content_digest,
        "physical_calls": [
            {
                "call_index": call.call_index,
                "strict_unit_status": call.strict_unit_status,
                "strict_failure": None
                if call.strict_failure is None
                else {
                    "stage": call.strict_failure.stage,
                    "call_unit_index": call.strict_failure.call_unit_index,
                    "detail_code": call.strict_failure.detail_code,
                },
                "raw_units_sha256": call.raw_units_sha256,
                "relative_units_sha256": call.relative_units_sha256,
                "authority_transform_status": call.authority_transform_status,
                "authority_absolute_sha256": call.authority_absolute_sha256,
                "raw_unit_ids": list(call.raw_unit_ids),
            }
            for call in calls
        ],
        "authority_status": authority_status,
        "authority_reasons": list(authority_reasons),
        "work_status": work.status,
        "route_claims": [
            [
                claim.owner_kind,
                claim.owner_index,
                claim.delivery_index,
                claim.source_index,
            ]
            for claim in work.route_claims
        ],
        "work_totals": [
            work.totals.states,
            work.totals.edges,
            work.totals.intervals,
            work.totals.normalize_chars,
        ],
        "call_surface_chars": list(surface_chars),
        "blocks": [
            {
                "source_index": block.source_index,
                "authority_unit_ids": None
                if block.authority_unit_ids is None
                else list(block.authority_unit_ids),
                "word_data": None
                if block.word_data is None
                else [
                    [
                        word.unit_id,
                        word.call_index,
                        word.call_unit_index,
                        word.text,
                        word.relative_start,
                        word.relative_end,
                        word.physical_origin_seconds,
                        word.start,
                        word.end,
                        word.provenance,
                        word.original_relative_start,
                        word.original_relative_end,
                    ]
                    for word in block.word_data
                ],
                "speech_start": block.speech_start,
                "speech_end": block.speech_end,
            }
            for block in blocks
        ],
        "raw_unit_count": raw_unit_count,
        "seed_status": seed_status,
        "seed_reasons": list(seed_reasons),
    }
    return frozen_json_digest(freeze_json(value))


def project_evidence_core(
    *,
    context_content_digest: str,
    blocks: tuple[AuthorityBlock, ...],
    captures: tuple[StrictCaptureResult, ...],
    transforms: tuple[AuthorityTransformResult, ...],
    distribution: AuthorityDistributionReceipt,
    seed_status: Literal["valid", "invalid"],
    seed_reasons: tuple[str, ...],
) -> EvidenceCore:
    """Independently reconstruct the mandatory acquisition primitive core."""
    if len(context_content_digest) != 64 or any(
        character not in "0123456789abcdef" for character in context_content_digest
    ):
        raise EvidenceCoreProjectionError("context digest is not lowercase SHA-256")
    if len(captures) != len(transforms) or len(captures) != len(
        distribution.work.calls
    ):
        raise EvidenceCoreProjectionError("physical call cardinality mismatch")
    if (
        seed_status not in ("valid", "invalid")
        or tuple(reason for reason in SEED_REASON_ORDER if reason in set(seed_reasons))
        != seed_reasons
    ):
        raise EvidenceCoreProjectionError("seed status or reason order is invalid")

    block_by_source = {block.source_index: block for block in blocks}
    surface_chars: list[int | None] = []
    physical_calls: list[EvidenceCorePhysicalCall] = []
    fresh_by_id: dict[str, FreshUnit] = {}
    for call_index, (capture, transform, row) in enumerate(
        zip(captures, transforms, distribution.work.calls)
    ):
        if (
            capture.call_index != call_index
            or transform.call_index != call_index
            or row.call_index != call_index
            or transform.capture != capture
        ):
            raise EvidenceCoreProjectionError("physical call cross-link mismatch")
        expected_surface = (
            None
            if capture.units is None
            else sum(
                len(block_by_source[source].alignment_text)
                for source in row.source_block_indices
            )
            + sum(len(unit.surface) for unit in capture.units)
        )
        if row.surface_chars != expected_surface:
            raise EvidenceCoreProjectionError("surface_chars projection mismatch")
        surface_chars.append(expected_surface)
        if transform.units is not None:
            for unit in transform.units:
                if unit.unit_id in fresh_by_id:
                    raise EvidenceCoreProjectionError("duplicate global fresh unit id")
                fresh_by_id[unit.unit_id] = unit
        physical_calls.append(
            EvidenceCorePhysicalCall(
                call_index,
                capture.status,
                transform.failure,
                capture.raw_units_digest,
                capture.normalized_relative_digest,
                transform.status,
                transform.authority_absolute_digest,
                capture.observed_unit_ids,
            )
        )

    authority_reasons = _authority_reasons(distribution.work)
    if distribution.reasons != authority_reasons:
        raise EvidenceCoreProjectionError("authority reason projection mismatch")
    evidence_blocks: list[EvidenceCoreBlock] = []
    if distribution.status == "valid":
        if distribution.owners is None:
            raise EvidenceCoreProjectionError("valid distribution lacks owners")
        for source_index, owner_ids in zip(
            distribution.owner_source_indices, distribution.owners
        ):
            try:
                owned = tuple(fresh_by_id[unit_id] for unit_id in owner_ids)
            except KeyError as exc:
                raise EvidenceCoreProjectionError(
                    "authority owner lacks transformed unit"
                ) from exc
            speech_start, speech_end = _anchors(owned)
            evidence_blocks.append(
                EvidenceCoreBlock(
                    source_index,
                    owner_ids,
                    tuple(_word(unit) for unit in owned),
                    speech_start,
                    speech_end,
                )
            )
    else:
        evidence_blocks.extend(
            EvidenceCoreBlock(source, None, None, None, None)
            for source in distribution.owner_source_indices
        )
    raw_unit_count = sum(len(capture.observed_unit_ids) for capture in captures)
    physical_tuple = tuple(physical_calls)
    block_tuple = tuple(evidence_blocks)
    digest = _digest_projection(
        context_content_digest=context_content_digest,
        calls=physical_tuple,
        authority_status=distribution.status,
        authority_reasons=authority_reasons,
        work=distribution.work,
        surface_chars=tuple(surface_chars),
        blocks=block_tuple,
        raw_unit_count=raw_unit_count,
        seed_status=seed_status,
        seed_reasons=seed_reasons,
    )
    return EvidenceCore(
        8,
        context_content_digest,
        physical_tuple,
        distribution.status,
        authority_reasons,
        distribution.work,
        tuple(surface_chars),
        block_tuple,
        raw_unit_count,
        seed_status,
        seed_reasons,
        digest,
    )


def evaluate_ald6(producer: EvidenceCore, reference: EvidenceCore) -> ALD6Outcome:
    if not isinstance(producer, EvidenceCore) or not isinstance(
        reference, EvidenceCore
    ):
        raise TypeError("ALD-6 operands must be EvidenceCore values")
    return ALD6Outcome("ALD-6", True, producer == reference)


__all__ = [
    "ALD6Outcome",
    "EvidenceCore",
    "EvidenceCoreBlock",
    "EvidenceCorePhysicalCall",
    "EvidenceCoreProjectionError",
    "EvidenceCoreWord",
    "evaluate_ald6",
    "project_evidence_core",
]
