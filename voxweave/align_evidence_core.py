"""Independent mandatory P6 acquisition-core projection and ALD-6 gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voxweave.align_delta_registry import ALIGN_DELTA_REGISTRY
from voxweave.align_acquisition import (
    AuthorityTransformResult,
    FreshUnit,
    PhysicalCallReceipt,
    StrictCaptureResult,
)
from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityCallInput,
    AuthorityDistributionReceipt,
    AuthorityJobWorkReceipt,
    AuthoritySkippedBlockInput,
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
    source_block_indices: tuple[int, ...]
    sample_start: int
    sample_end: int
    sample_rate: int
    physical_origin_seconds: float
    legacy_origin_seconds: float
    legacy_origin_kind: Literal["identity", "sample-origin", "nominal-route"]
    authority_origin_seconds: float
    backend_model_config_sha256: str
    route_input_sha256: str
    strict_unit_status: Literal["valid", "invalid"]
    strict_failure: StrictFailureLocator | None
    raw_units_sha256: str | None
    relative_units_sha256: str | None
    legacy_slice_sha256: str
    legacy_absolute_sha256: str | None
    authority_transform_status: Literal["valid", "invalid"]
    authority_absolute_sha256: str | None
    raw_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCore:
    schema_version: Literal[8]
    context_content_digest: str
    receipt_digest: str
    physical_calls: tuple[EvidenceCorePhysicalCall, ...]
    authority_status: Literal["valid", "invalid"]
    authority_reasons: tuple[str, ...]
    authority_work: AuthorityJobWorkReceipt
    call_surface_chars: tuple[int | None, ...]
    blocks: tuple[EvidenceCoreBlock, ...]
    raw_unit_count: int
    seed_status: Literal["valid", "invalid"]
    seed_reasons: tuple[str, ...]

    @property
    def core_digest(self) -> str:
        """Compatibility spelling for the acquisition receipt digest."""
        return self.receipt_digest


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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _same_float(left: object, right: object) -> bool:
    return type(left) is float and type(right) is float and left.hex() == right.hex()


def _validate_physical_receipt(receipt: PhysicalCallReceipt) -> None:
    if (
        type(receipt.audio_sample_start) is not int
        or type(receipt.audio_sample_end) is not int
        or type(receipt.sample_rate) is not int
        or receipt.sample_rate <= 0
        or not 0 <= receipt.audio_sample_start <= receipt.audio_sample_end
    ):
        raise EvidenceCoreProjectionError("physical sample geometry is invalid")
    quotient = receipt.audio_sample_start / receipt.sample_rate
    if not _same_float(receipt.physical_origin_seconds, quotient) or not _same_float(
        receipt.authority_origin_seconds, quotient
    ):
        raise EvidenceCoreProjectionError("physical origin receipt mismatch")
    if receipt.legacy_origin_kind == "identity":
        if receipt.audio_sample_start != 0 or not _same_float(
            receipt.legacy_origin_seconds, 0.0
        ):
            raise EvidenceCoreProjectionError("identity origin receipt mismatch")
    elif receipt.legacy_origin_kind == "sample-origin":
        if not _same_float(receipt.legacy_origin_seconds, quotient):
            raise EvidenceCoreProjectionError("sample origin receipt mismatch")
    elif receipt.legacy_origin_kind != "nominal-route":
        raise EvidenceCoreProjectionError("physical origin kind is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            receipt.backend_model_config_digest,
            receipt.route_input_digest,
            receipt.legacy_slice_digest,
        )
    ):
        raise EvidenceCoreProjectionError("physical receipt digest is invalid")
    if receipt.legacy_absolute_digest is not None and not _is_sha256(
        receipt.legacy_absolute_digest
    ):
        raise EvidenceCoreProjectionError("legacy absolute digest is invalid")
    if receipt.strict_unit_status == "valid":
        if not _is_sha256(receipt.raw_units_digest) or not _is_sha256(
            receipt.normalized_relative_digest
        ):
            raise EvidenceCoreProjectionError("strict physical digests are unavailable")
    elif receipt.strict_unit_status == "invalid":
        if (
            receipt.raw_units_digest is not None
            or receipt.normalized_relative_digest is not None
        ):
            raise EvidenceCoreProjectionError("invalid strict call carries digests")
    else:
        raise EvidenceCoreProjectionError("strict physical status is invalid")
    if receipt.authority_transform_status == "valid":
        if receipt.strict_failure is not None or not _is_sha256(
            receipt.authority_absolute_digest
        ):
            raise EvidenceCoreProjectionError("valid authority transform is incomplete")
    elif receipt.authority_transform_status == "invalid":
        if (
            receipt.strict_failure is None
            or receipt.authority_absolute_digest is not None
        ):
            raise EvidenceCoreProjectionError(
                "invalid authority transform is incomplete"
            )
    else:
        raise EvidenceCoreProjectionError("authority transform status is invalid")


def _replay_distribution(
    *,
    blocks: tuple[AuthorityBlock, ...],
    captures: tuple[StrictCaptureResult, ...],
    transforms: tuple[AuthorityTransformResult, ...],
    physical_calls: tuple[PhysicalCallReceipt, ...] | None,
    distribution: AuthorityDistributionReceipt,
    iso: str,
) -> None:
    """Rebuild the receipt from sealed primitive facts, never recorded counters."""
    from voxweave.align_distribution_reference import (
        DistributionReferenceError,
        replay_authority_distribution,
    )

    work = distribution.work
    if len(captures) != len(work.calls) or len(transforms) != len(work.calls):
        raise EvidenceCoreProjectionError(
            "allocator physical-call cardinality mismatch"
        )
    call_inputs: list[AuthorityCallInput] = []
    raw_cursor = 0
    for index, (capture, transform, row) in enumerate(
        zip(captures, transforms, work.calls, strict=True)
    ):
        physical = None if physical_calls is None else physical_calls[index]
        if physical is not None:
            _validate_physical_receipt(physical)
        sources = (
            row.source_block_indices
            if physical is None
            else physical.source_block_indices
        )
        if physical is not None and (
            physical.call_index != index
            or physical.raw_unit_ids != capture.observed_unit_ids
            or physical.strict_unit_status != capture.status
            or physical.strict_failure != transform.failure
            or physical.raw_units_digest != capture.raw_units_digest
            or physical.normalized_relative_digest != capture.normalized_relative_digest
            or physical.authority_transform_status != transform.status
            or physical.authority_absolute_digest != transform.authority_absolute_digest
            or physical.source_block_indices != row.source_block_indices
        ):
            raise EvidenceCoreProjectionError("physical receipt cross-link mismatch")
        if capture.status != "valid":
            preflight: Literal["valid", "capture-invalid", "transform-invalid"] = (
                "capture-invalid"
            )
            surfaces = None
            failure = capture.failure
        elif transform.status != "valid":
            preflight = "transform-invalid"
            surfaces = tuple(unit.surface for unit in capture.units or ())
            failure = transform.failure
        else:
            preflight = "valid"
            surfaces = tuple(unit.surface for unit in capture.units or ())
            failure = None
        call_inputs.append(
            AuthorityCallInput(
                index,
                sources,
                (raw_cursor, raw_cursor + len(capture.observed_unit_ids)),
                capture.observed_unit_ids,
                surfaces,
                preflight,
                failure,
            )
        )
        raw_cursor += len(capture.observed_unit_ids)
    call_owner = {
        source for call in call_inputs for source in call.source_block_indices
    }
    skipped_rows: list[AuthoritySkippedBlockInput] = []
    for delivery_index, block in enumerate(blocks):
        if block.source_index in call_owner:
            continue
        stripped = block.alignment_text.strip()
        skipped_rows.append(
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
    try:
        replay_authority_distribution(
            blocks=blocks,
            calls=tuple(call_inputs),
            skipped=tuple(skipped_rows),
            receipt=distribution,
            iso=iso,
        )
    except DistributionReferenceError as exc:
        raise EvidenceCoreProjectionError("allocator receipt replay failed") from exc


def _synthetic_physical_call(
    capture: StrictCaptureResult,
    transform: AuthorityTransformResult,
    row: object,
) -> PhysicalCallReceipt:
    source_indices = tuple(getattr(row, "source_block_indices"))
    digest = frozen_json_digest(freeze_json(list(capture.observed_unit_ids)))
    origin = transform.units[0].physical_origin_seconds if transform.units else 0.0
    return PhysicalCallReceipt(
        capture.call_index,
        source_indices,
        0,
        0,
        1,
        origin,
        origin,
        "identity" if origin == 0.0 else "sample-origin",
        origin,
        digest,
        digest,
        capture.status,
        transform.failure,
        capture.raw_units_digest,
        capture.normalized_relative_digest,
        digest,
        digest,
        transform.status,
        transform.authority_absolute_digest,
        capture.observed_unit_ids,
    )


def _evidence_call(receipt: PhysicalCallReceipt) -> EvidenceCorePhysicalCall:
    return EvidenceCorePhysicalCall(
        receipt.call_index,
        receipt.source_block_indices,
        receipt.audio_sample_start,
        receipt.audio_sample_end,
        receipt.sample_rate,
        receipt.physical_origin_seconds,
        receipt.legacy_origin_seconds,
        receipt.legacy_origin_kind,
        receipt.authority_origin_seconds,
        receipt.backend_model_config_digest,
        receipt.route_input_digest,
        receipt.strict_unit_status,
        receipt.strict_failure,
        receipt.raw_units_digest,
        receipt.normalized_relative_digest,
        receipt.legacy_slice_digest,
        receipt.legacy_absolute_digest,
        receipt.authority_transform_status,
        receipt.authority_absolute_digest,
        receipt.raw_unit_ids,
    )


def _digest_projection(
    *,
    context_content_digest: str,
    receipt_digest: str | None,
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
        "receipt_digest": receipt_digest,
        "physical_calls": [
            {
                "call_index": call.call_index,
                "source_block_indices": list(call.source_block_indices),
                "sample_start": call.sample_start,
                "sample_end": call.sample_end,
                "sample_rate": call.sample_rate,
                "physical_origin_seconds": call.physical_origin_seconds,
                "legacy_origin_seconds": call.legacy_origin_seconds,
                "legacy_origin_kind": call.legacy_origin_kind,
                "authority_origin_seconds": call.authority_origin_seconds,
                "backend_model_config_sha256": call.backend_model_config_sha256,
                "route_input_sha256": call.route_input_sha256,
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
                "legacy_slice_sha256": call.legacy_slice_sha256,
                "legacy_absolute_sha256": call.legacy_absolute_sha256,
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


def _assemble_evidence_core(
    *,
    context_content_digest: str,
    blocks: tuple[AuthorityBlock, ...],
    captures: tuple[StrictCaptureResult, ...],
    transforms: tuple[AuthorityTransformResult, ...],
    distribution: AuthorityDistributionReceipt,
    seed_status: Literal["valid", "invalid"],
    seed_reasons: tuple[str, ...],
    physical_calls: tuple[PhysicalCallReceipt, ...] | None,
    receipt_digest: str | None,
    language: str,
    replay_distribution: bool,
) -> EvidenceCore:
    """Construct one immutable core from a single thaw of sealed facts."""
    if len(context_content_digest) != 64 or any(
        character not in "0123456789abcdef" for character in context_content_digest
    ):
        raise EvidenceCoreProjectionError("context digest is not lowercase SHA-256")
    if len(captures) != len(transforms) or len(captures) != len(
        distribution.work.calls
    ):
        raise EvidenceCoreProjectionError("physical call cardinality mismatch")
    if physical_calls is not None and len(physical_calls) != len(captures):
        raise EvidenceCoreProjectionError("physical receipt cardinality mismatch")
    if (
        seed_status not in ("valid", "invalid")
        or tuple(reason for reason in SEED_REASON_ORDER if reason in set(seed_reasons))
        != seed_reasons
    ):
        raise EvidenceCoreProjectionError("seed status or reason order is invalid")
    if receipt_digest is not None and not _is_sha256(receipt_digest):
        raise EvidenceCoreProjectionError("receipt digest is not lowercase SHA-256")
    if type(language) is not str or not language:
        raise EvidenceCoreProjectionError("evidence language is invalid")
    if replay_distribution:
        _replay_distribution(
            blocks=blocks,
            captures=captures,
            transforms=transforms,
            physical_calls=physical_calls,
            distribution=distribution,
            iso=language,
        )

    block_by_source = {block.source_index: block for block in blocks}
    surface_chars: list[int | None] = []
    evidence_calls: list[EvidenceCorePhysicalCall] = []
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
        receipt = (
            _synthetic_physical_call(capture, transform, row)
            if physical_calls is None
            else physical_calls[call_index]
        )
        evidence_calls.append(_evidence_call(receipt))

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
    physical_tuple = tuple(evidence_calls)
    block_tuple = tuple(evidence_blocks)
    digest = _digest_projection(
        context_content_digest=context_content_digest,
        receipt_digest=receipt_digest,
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
    stable_receipt = digest if receipt_digest is None else receipt_digest
    return EvidenceCore(
        8,
        context_content_digest,
        stable_receipt,
        physical_tuple,
        distribution.status,
        authority_reasons,
        distribution.work,
        tuple(surface_chars),
        block_tuple,
        raw_unit_count,
        seed_status,
        seed_reasons,
    )


def build_evidence_core(
    *,
    context_content_digest: str,
    blocks: tuple[AuthorityBlock, ...],
    captures: tuple[StrictCaptureResult, ...],
    transforms: tuple[AuthorityTransformResult, ...],
    distribution: AuthorityDistributionReceipt,
    seed_status: Literal["valid", "invalid"],
    seed_reasons: tuple[str, ...],
    physical_calls: tuple[PhysicalCallReceipt, ...] | None = None,
    receipt_digest: str | None = None,
    language: str = "en",
) -> EvidenceCore:
    """Build the producer core without consulting the reference replay."""
    return _assemble_evidence_core(
        context_content_digest=context_content_digest,
        blocks=blocks,
        captures=captures,
        transforms=transforms,
        distribution=distribution,
        seed_status=seed_status,
        seed_reasons=seed_reasons,
        physical_calls=physical_calls,
        receipt_digest=receipt_digest,
        language=language,
        replay_distribution=False,
    )


def project_evidence_core(
    *,
    context_content_digest: str,
    blocks: tuple[AuthorityBlock, ...],
    captures: tuple[StrictCaptureResult, ...],
    transforms: tuple[AuthorityTransformResult, ...],
    distribution: AuthorityDistributionReceipt,
    seed_status: Literal["valid", "invalid"],
    seed_reasons: tuple[str, ...],
    physical_calls: tuple[PhysicalCallReceipt, ...] | None = None,
    receipt_digest: str | None = None,
    language: str = "en",
) -> EvidenceCore:
    """Independently replay and reconstruct the mandatory acquisition core."""
    return _assemble_evidence_core(
        context_content_digest=context_content_digest,
        blocks=blocks,
        captures=captures,
        transforms=transforms,
        distribution=distribution,
        seed_status=seed_status,
        seed_reasons=seed_reasons,
        physical_calls=physical_calls,
        receipt_digest=receipt_digest,
        language=language,
        replay_distribution=True,
    )


def evaluate_ald6(producer: EvidenceCore, reference: EvidenceCore) -> ALD6Outcome:
    definition = ALIGN_DELTA_REGISTRY.get("ALD-6")
    if definition is None or definition.phase != "mandatory-core":
        raise EvidenceCoreProjectionError(
            "mandatory ALD-6 registry entry is unavailable"
        )
    if not isinstance(producer, EvidenceCore) or not isinstance(
        reference, EvidenceCore
    ):
        raise TypeError("ALD-6 operands must be EvidenceCore values")
    if producer is reference:
        raise EvidenceCoreProjectionError(
            "ALD-6 requires independent distinct EvidenceCore operands"
        )
    return ALD6Outcome("ALD-6", True, producer == reference)


__all__ = [
    "ALD6Outcome",
    "EvidenceCore",
    "EvidenceCoreBlock",
    "EvidenceCorePhysicalCall",
    "EvidenceCoreProjectionError",
    "EvidenceCoreWord",
    "build_evidence_core",
    "evaluate_ald6",
    "project_evidence_core",
]
