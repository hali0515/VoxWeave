"""Independent mandatory P6 acquisition-core projection and ALD-6 gate.

The producer and reference implementations in this module intentionally do not
share field projectors.  The producer serializes its issued receipt view; the
reference reconstructs the same closed pre-selected-output value from separate
context, raw-call, legacy, and adapter thaws.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from voxweave.align_acquisition import (
    AuthorityTransformResult,
    EvidenceCoreProducerInputs,
    EvidenceCoreReferenceInputs,
    FreshUnit,
    PhysicalCallReceipt,
    ReferencePhysicalCallFacts,
    StrictCaptureResult,
    StrictCapturedUnit,
)
from voxweave.align_delta_registry import ALIGN_DELTA_REGISTRY
from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityCallInput,
    AuthorityJobWorkReceipt,
    AuthoritySkippedBlockInput,
    LegacyCallDistributionReceipt,
    StrictFailureLocator,
)
from voxweave.align_failures import AUTHORITY_REASON_ORDER, SEED_REASON_ORDER
from voxweave.align_snapshot import (
    FROZEN_NULL,
    FrozenArray,
    FrozenObject,
    FrozenString,
    freeze_json,
    frozen_json_digest,
    thaw_json,
)
from voxweave.core.langsets import LANGUAGES_WITHOUT_SPACES
from voxweave.core.segdoc import SourceUnit
from voxweave.core.subunit import speech_span_units
from voxweave.engine_registry import engine_family_for


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
    legacy_unit_ids: tuple[str, ...]
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
    language: str
    route: str
    input_history: FrozenObject
    route_plan: FrozenObject
    physical_calls: tuple[EvidenceCorePhysicalCall, ...]
    legacy_distribution: FrozenObject
    authority_distribution: FrozenObject
    blocks: tuple[EvidenceCoreBlock, ...]
    raw_unit_count: int
    strict_input_status: FrozenObject
    seed_status: Literal["valid", "invalid"]
    seed_reasons: tuple[str, ...]
    v2_policy_status: FrozenObject
    profile_status: FrozenObject
    evidence_status: FrozenObject
    v2_admission_status: Literal["valid", "invalid"]
    authority_status: Literal["valid", "invalid"]
    authority_reasons: tuple[str, ...]
    authority_work: AuthorityJobWorkReceipt
    call_surface_chars: tuple[int | None, ...]
    _projection: FrozenObject = field(repr=False, compare=True)

    @property
    def core_digest(self) -> str:
        return self.receipt_digest


@dataclass(frozen=True)
class ALD6Outcome:
    delta_id: Literal["ALD-6"]
    triggered: Literal[True]
    passed: bool


# ---------------------------------------------------------------------------
# Producer projector.  Nothing in this section is called by the reference.


def _p_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _p_stable_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            member.name: _p_stable_value(getattr(value, member.name))
            for member in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_p_stable_value(member) for member in value]
    if isinstance(value, list):
        return [_p_stable_value(member) for member in value]
    if isinstance(value, Mapping):
        return {str(key): _p_stable_value(member) for key, member in value.items()}
    return value


def _p_digest(label: str, value: Any) -> str:
    return frozen_json_digest(freeze_json([label, _p_stable_value(value)]))


def _p_object(value: Mapping[str, Any], label: str) -> FrozenObject:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenObject):
        raise EvidenceCoreProjectionError(f"producer {label} is not an object")
    return frozen


def _p_failure(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "stage": getattr(value, "stage"),
        "call_unit_index": getattr(value, "call_unit_index"),
        "detail_code": getattr(value, "detail_code"),
    }


def _p_lane(value: object) -> dict[str, Any]:
    counters = getattr(value, "counters")
    return {
        "status": getattr(value, "status"),
        "states": counters.states,
        "edges": counters.edges,
        "intervals": counters.intervals,
        "normalize_chars": counters.normalize_chars,
        "terminal_detail_code": getattr(value, "terminal_detail_code"),
    }


def _p_denied(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "lane": getattr(value, "lane"),
        "event_ordinal": getattr(value, "event_ordinal"),
        "event_kind": getattr(value, "event_kind"),
        "subject": list(getattr(value, "subject")),
        "counters": [
            {
                "counter": counter.counter,
                "amount": counter.amount,
                "scopes": list(counter.scopes),
            }
            for counter in getattr(value, "counters")
        ],
    }


def _p_work(work: AuthorityJobWorkReceipt) -> dict[str, Any]:
    mismatch = work.route_mismatch
    return {
        "status": work.status,
        "route_status": work.route_status,
        "route_mismatch": None
        if mismatch is None
        else {
            "kind": mismatch.kind,
            "observation_index": mismatch.observation_index,
            "expected_delivery_index": mismatch.expected_delivery_index,
            "observed_delivery_index": mismatch.observed_delivery_index,
        },
        "route_claims": [
            {
                "owner_kind": claim.owner_kind,
                "owner_index": claim.owner_index,
                "delivery_index": claim.delivery_index,
                "source_index": claim.source_index,
            }
            for claim in work.route_claims
        ],
        "declared_delivery_block_count": work.declared_delivery_block_count,
        "declared_call_count": work.declared_call_count,
        "declared_skip_count": work.declared_skip_count,
        "declared_raw_node_count": work.declared_raw_node_count,
        "charged_call_count": work.charged_call_count,
        "limit_profile_kind": work.limit_profile_kind,
        "limit_profile_digest": work.limit_profile_digest,
        "limits": {
            "call_limit": work.limits.call_limit,
            "state_limit": work.limits.state_limit,
            "edge_limit": work.limits.edge_limit,
            "interval_limit": work.limits.interval_limit,
            "normalize_char_limit": work.limits.normalize_char_limit,
        },
        "totals": {
            "states": work.totals.states,
            "edges": work.totals.edges,
            "intervals": work.totals.intervals,
            "normalize_chars": work.totals.normalize_chars,
        },
        "terminal_call_index": work.terminal_call_index,
        "denied_charge": _p_denied(work.denied_charge),
        "calls": [
            {
                "call_index": row.call_index,
                "route_claim_positions": list(row.route_claim_positions),
                "source_block_indices": list(row.source_block_indices),
                "raw_node_range": list(row.raw_node_range),
                "block_count": row.block_count,
                "raw_node_count": row.raw_node_count,
                "typed_unit_count": row.typed_unit_count,
                "surface_chars": row.surface_chars,
                "strict_preflight_status": row.strict_preflight_status,
                "strict_failure": _p_failure(row.strict_failure),
                "limits": {
                    "state_limit": row.limits.state_limit,
                    "edge_limit": row.limits.edge_limit,
                    "interval_limit": row.limits.interval_limit,
                    "normalize_char_limit": row.limits.normalize_char_limit,
                },
                "allocator": _p_lane(row.allocator),
                "verifier": None if row.verifier is None else _p_lane(row.verifier),
            }
            for row in work.calls
        ],
        "skipped_blocks": [
            {
                "route_claim_positions": list(row.route_claim_positions),
                "delivery_index": row.delivery_index,
                "source_index": row.source_index,
                "route_skip_reason": row.route_skip_reason,
                "source_text_kind": row.source_text_kind,
                "detail_code": row.detail_code,
                "work_status": row.work_status,
                "states": row.counters.states,
                "edges": row.counters.edges,
                "intervals": row.counters.intervals,
                "normalize_chars": row.counters.normalize_chars,
            }
            for row in work.skipped_blocks
        ],
    }


def _p_history(facts: EvidenceCoreProducerInputs) -> dict[str, Any]:
    stable = thaw_json(facts.stable_fields)
    if not isinstance(stable, dict):
        raise EvidenceCoreProjectionError("producer stable context is not an object")
    vtt = stable.get("vtt_generation")
    sibling = stable.get("sibling_generation")
    adapter = stable.get("adapter_inputs")
    if not isinstance(vtt, Mapping):
        vtt = {}
    if not isinstance(sibling, Mapping):
        sibling = {}
    if not isinstance(adapter, Mapping):
        adapter = {}
    policy = adapter.get("legacy_policy")
    if not isinstance(policy, Mapping):
        policy = {}
    try:
        policy_bits = {
            name: struct.pack(">d", policy.get(name, 0.0)).hex()
            for name in ("min_cue_sec", "tiny_cue_sec", "tiny_cue_target")
        }
    except (TypeError, ValueError, struct.error) as exc:
        raise EvidenceCoreProjectionError("producer legacy policy is invalid") from exc
    return {
        "vtt_present": vtt.get("present"),
        "vtt_size": vtt.get("size"),
        "vtt_sha256": vtt.get("sha256"),
        "sibling_json_present": sibling.get("present"),
        "sibling_json_size": sibling.get("size"),
        "sibling_json_sha256": sibling.get("sha256"),
        "block_content_sha256": stable.get("block_content_sha256"),
        "registry_family": facts.engine_family,
        "media_fingerprint": stable.get("media_fingerprint"),
        "media_logical_id": stable.get("media_logical_id"),
        "media_display_name": stable.get("media_display_name"),
        "prepared_audio_size": stable.get("prepared_audio_size"),
        "prepared_audio_sha256": stable.get("prepared_audio_sha256"),
        "profile_input_sha256": stable.get("profile_input_sha256"),
        "evidence_carriers_sha256": stable.get("evidence_carriers_sha256"),
        "authority_limit_profile_kind": facts.authority_profile.kind,
        "authority_limit_profile_digest": facts.authority_profile.profile_digest,
        "legacy_policy_binary64": policy_bits,
        "target_logical_id": stable.get("target_logical_id"),
        "expected_vtt_sha256": stable.get("expected_vtt_sha256"),
    }


def _p_source_facts(facts: EvidenceCoreProducerInputs) -> dict[str, Any]:
    model = thaw_json(facts.backend_model_config_facts)
    route = thaw_json(facts.route_input_facts)
    if not isinstance(model, dict) or not isinstance(route, dict):
        raise EvidenceCoreProjectionError("producer durable source facts are invalid")
    return {"backend_model_config": model, "route_input": route}


def _p_call(receipt: PhysicalCallReceipt) -> EvidenceCorePhysicalCall:
    for digest in (
        receipt.backend_model_config_digest,
        receipt.route_input_digest,
        receipt.legacy_slice_digest,
    ):
        if not _p_sha(digest):
            raise EvidenceCoreProjectionError("producer physical digest is invalid")
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


def _p_route(
    facts: EvidenceCoreProducerInputs,
    calls: tuple[EvidenceCorePhysicalCall, ...],
) -> dict[str, Any]:
    stable = thaw_json(facts.stable_fields)
    blocks = stable.get("blocks", []) if isinstance(stable, dict) else []
    calls_by_source = {
        source: call.call_index
        for call in calls
        for source in call.source_block_indices
    }
    skips = {row.source_index: row for row in facts.distribution.work.skipped_blocks}
    by_index = {call.call_index: call for call in calls}
    entries: list[dict[str, Any]] = []
    for delivery_index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise EvidenceCoreProjectionError("producer route block is invalid")
        source = block.get("source_index")
        call_index = calls_by_source.get(source) if type(source) is int else None
        if facts.route_kind in ("ctc-full", "mms-full"):
            action, skip_reason = "full-pass-member", None
        elif call_index is not None:
            action, skip_reason = "qwen-call", None
        else:
            action = "qwen-skip"
            skip = skips.get(source) if type(source) is int else None
            skip_reason = None if skip is None else skip.route_skip_reason
        route_start = block.get("start")
        route_end = block.get("end")
        if facts.route_kind == "qwen-crop" and call_index is not None:
            physical = by_index[call_index]
            route_start = physical.legacy_origin_seconds
            if type(route_end) is not float:
                route_end = physical.sample_end / physical.sample_rate
        entries.append(
            {
                "delivery_index": delivery_index,
                "source_index": source,
                "route_start": route_start,
                "route_end": route_end,
                "action": action,
                "call_index": call_index,
                "skip_reason": skip_reason,
            }
        )
    return {"digest": _p_digest("p6-route-plan-v1", entries), "entries": entries}


def _p_legacy(receipts: tuple[LegacyCallDistributionReceipt, ...]) -> dict[str, Any]:
    calls = [
        {
            "call_index": index,
            "owner_source_indices": list(row.owner_source_indices),
            "expected_counts": list(row.expected_counts),
            "requested_ranges": [list(member) for member in row.requested_ranges],
            "realized_ranges": [list(member) for member in row.realized_ranges],
            "owner_unit_ids": [list(member) for member in row.owner_unit_ids],
            "final_cursor": row.final_cursor,
            "consumed_prefix_unit_ids": list(row.consumed_prefix_unit_ids),
            "shortage_source_indices": list(row.shortage_source_indices),
            "leftover_unit_ids": list(row.leftover_unit_ids),
        }
        for index, row in enumerate(receipts)
    ]
    return {
        "digest": _p_digest("p6-legacy-distribution-v1", calls),
        "calls": calls,
    }


def _p_authority(facts: EvidenceCoreProducerInputs) -> dict[str, Any]:
    distribution = facts.distribution
    valid = distribution.status == "valid"
    body = {
        "status": distribution.status,
        "owner_source_indices": list(distribution.owner_source_indices)
        if valid
        else None,
        "expected_counts": list(distribution.expected_counts or ()) if valid else None,
        "owner_unit_ids": [list(owner) for owner in distribution.owners or ()]
        if valid
        else None,
        "consumed_count": distribution.consumed_count,
        "leftover_unit_ids": list(distribution.leftovers),
        "reasons": list(distribution.reasons),
        "work": _p_work(distribution.work),
    }
    return {
        "status": distribution.status,
        "digest": _p_digest("p6-authority-distribution-v1", body),
        **{key: value for key, value in body.items() if key != "status"},
    }


def _p_word(unit: FreshUnit) -> EvidenceCoreWord:
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


def _p_blocks(facts: EvidenceCoreProducerInputs) -> tuple[EvidenceCoreBlock, ...]:
    legacy: dict[int, tuple[str, ...]] = {}
    for receipt in facts.legacy_receipts:
        for source, ids in zip(
            receipt.owner_source_indices, receipt.owner_unit_ids, strict=True
        ):
            legacy[source] = ids
    fresh = {
        unit.unit_id: unit
        for transform in facts.transforms
        for unit in (transform.units or ())
    }
    out: list[EvidenceCoreBlock] = []
    if facts.distribution.status == "valid" and facts.distribution.owners is not None:
        for source, ids in zip(
            facts.distribution.owner_source_indices,
            facts.distribution.owners,
            strict=True,
        ):
            try:
                owned = tuple(fresh[unit_id] for unit_id in ids)
            except KeyError as exc:
                raise EvidenceCoreProjectionError(
                    "producer owner lacks transformed unit"
                ) from exc
            source_units = tuple(
                SourceUnit(
                    unit.unit_id,
                    unit.surface,
                    unit.start,
                    unit.end,
                    unit.provenance,
                    None,
                )
                for unit in owned
            )
            start, end = speech_span_units(source_units)
            out.append(
                EvidenceCoreBlock(
                    source,
                    legacy.get(source, ()),
                    ids,
                    tuple(_p_word(unit) for unit in owned),
                    start,
                    end,
                )
            )
    else:
        out.extend(
            EvidenceCoreBlock(
                block.source_index,
                legacy.get(block.source_index, ()),
                None,
                None,
                None,
                None,
            )
            for block in facts.blocks
        )
    return tuple(out)


def _p_block_values(blocks: tuple[EvidenceCoreBlock, ...]) -> list[dict[str, Any]]:
    return [
        {
            "source_index": block.source_index,
            "legacy_unit_ids": list(block.legacy_unit_ids),
            "authority_unit_ids": None
            if block.authority_unit_ids is None
            else list(block.authority_unit_ids),
            "word_data": None
            if block.word_data is None
            else [
                {
                    "unit_id": word.unit_id,
                    "call_index": word.call_index,
                    "call_unit_index": word.call_unit_index,
                    "text": word.text,
                    "relative_start": word.relative_start,
                    "relative_end": word.relative_end,
                    "physical_origin_seconds": word.physical_origin_seconds,
                    "start": word.start,
                    "end": word.end,
                    "provenance": word.provenance,
                    "original_relative_start": word.original_relative_start,
                    "original_relative_end": word.original_relative_end,
                }
                for word in block.word_data
            ],
            "speech_start": block.speech_start,
            "speech_end": block.speech_end,
        }
        for block in blocks
    ]


def _p_physical_values(
    calls: tuple[EvidenceCorePhysicalCall, ...],
    facts: EvidenceCoreProducerInputs,
) -> list[dict[str, Any]]:
    legacy_by_source = {
        block.source_index: units
        for block, units in zip(
            facts.blocks, facts.legacy_relative_block_units, strict=True
        )
    }
    return [
        {
            "call_index": row.call_index,
            "source_block_indices": list(row.source_block_indices),
            "sample_start": row.sample_start,
            "sample_end": row.sample_end,
            "sample_rate": row.sample_rate,
            "physical_origin_seconds": row.physical_origin_seconds,
            "legacy_origin_seconds": row.legacy_origin_seconds,
            "legacy_origin_kind": row.legacy_origin_kind,
            "authority_origin_seconds": row.authority_origin_seconds,
            "backend_model_config_sha256": row.backend_model_config_sha256,
            "route_input_sha256": row.route_input_sha256,
            "strict_unit_status": row.strict_unit_status,
            "strict_failure": _p_failure(row.strict_failure),
            "raw_units_sha256": row.raw_units_sha256,
            "relative_units_sha256": row.relative_units_sha256,
            "legacy_retained_units": [
                _p_stable_value(legacy_by_source.get(source, ()))
                for source in row.source_block_indices
            ],
            "legacy_slice_sha256": row.legacy_slice_sha256,
            "legacy_absolute_sha256": row.legacy_absolute_sha256,
            "authority_transform_status": row.authority_transform_status,
            "authority_absolute_sha256": row.authority_absolute_sha256,
            "raw_unit_ids": list(row.raw_unit_ids),
        }
        for row in calls
    ]


def _p_status(value: object) -> dict[str, Any]:
    return {
        "kind": getattr(value, "kind"),
        "detail_code": getattr(value, "detail_code"),
    }


def _p_profile(value: object) -> dict[str, Any]:
    return {
        "kind": getattr(value, "kind"),
        "source": getattr(value, "source"),
        "detail_code": getattr(value, "detail_code"),
    }


def build_evidence_core(facts: EvidenceCoreProducerInputs) -> EvidenceCore:
    """Project the producer's complete §9 value without reference helpers."""
    if not isinstance(facts, EvidenceCoreProducerInputs):
        raise TypeError("producer EvidenceCore inputs have the wrong type")
    if not _p_sha(facts.context_content_digest) or not _p_sha(facts.receipt_digest):
        raise EvidenceCoreProjectionError("producer context/receipt digest is invalid")
    if (
        facts.seed_status not in ("valid", "invalid")
        or tuple(
            reason for reason in SEED_REASON_ORDER if reason in set(facts.seed_reasons)
        )
        != facts.seed_reasons
    ):
        raise EvidenceCoreProjectionError("producer seed status is invalid")
    calls = tuple(_p_call(receipt) for receipt in facts.physical_calls)
    blocks = _p_blocks(facts)
    history_value = _p_history(facts)
    route_value = _p_route(facts, calls)
    legacy_value = _p_legacy(facts.legacy_receipts)
    authority_value = _p_authority(facts)
    strict_value = _p_status(facts.strict_input_status)
    policy_value = _p_status(facts.v2_policy_status)
    profile_value = _p_profile(facts.profile_status)
    evidence_value = _p_status(facts.evidence_status)
    admission_valid = (
        strict_value["kind"] == "valid"
        and all(call.authority_transform_status == "valid" for call in calls)
        and facts.distribution.status == "valid"
        and facts.seed_status == "valid"
        and policy_value["kind"] == "valid"
        and profile_value["kind"] == "valid"
        and evidence_value["kind"] == "valid"
    )
    projection_value = {
        "schema_version": 8,
        "kind": "fresh-alignment",
        "context_content_digest": facts.context_content_digest,
        "receipt_digest": facts.receipt_digest,
        "language": facts.language,
        "route": facts.route_kind,
        "source_facts": _p_source_facts(facts),
        "input_history": history_value,
        "route_plan": route_value,
        "physical_calls": _p_physical_values(calls, facts),
        "legacy_distribution": legacy_value,
        "authority_distribution": authority_value,
        "blocks": _p_block_values(blocks),
        "raw_unit_count": sum(
            len(capture.observed_unit_ids) for capture in facts.captures
        ),
        "strict_input_status": strict_value,
        "seed_status": {"kind": facts.seed_status, "reasons": list(facts.seed_reasons)},
        "v2_policy_status": policy_value,
        "profile_status": profile_value,
        "evidence_status": evidence_value,
        "v2_admission_status": "valid" if admission_valid else "invalid",
    }
    projection = _p_object(projection_value, "EvidenceCore")
    return EvidenceCore(
        8,
        facts.context_content_digest,
        facts.receipt_digest,
        facts.language,
        facts.route_kind,
        _p_object(history_value, "input history"),
        _p_object(route_value, "route plan"),
        calls,
        _p_object(legacy_value, "legacy distribution"),
        _p_object(authority_value, "authority distribution"),
        blocks,
        projection_value["raw_unit_count"],
        _p_object(strict_value, "strict status"),
        facts.seed_status,
        facts.seed_reasons,
        _p_object(policy_value, "policy status"),
        _p_object(profile_value, "profile status"),
        _p_object(evidence_value, "evidence status"),
        "valid" if admission_valid else "invalid",
        facts.distribution.status,
        facts.distribution.reasons,
        facts.distribution.work,
        tuple(row.surface_chars for row in facts.distribution.work.calls),
        projection,
    )


# ---------------------------------------------------------------------------
# Reference projector C.  Every field helper below is a separate implementation;
# none calls the producer section above.


def _r_sha(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    for character in value:
        if character not in "0123456789abcdef":
            return False
    return True


def _r_stable_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        projected: dict[str, Any] = {}
        for member in dataclasses.fields(value):
            projected[member.name] = _r_stable_value(getattr(value, member.name))
        return projected
    if isinstance(value, (tuple, list)):
        return [_r_stable_value(member) for member in value]
    if isinstance(value, Mapping):
        return {str(key): _r_stable_value(value[key]) for key in value}
    return value


def _r_stable_digest(value: Any) -> str:
    return frozen_json_digest(freeze_json(_r_stable_value(value)))


def _r_labeled_digest(label: str, value: Any) -> str:
    return frozen_json_digest(freeze_json([label, _r_stable_value(value)]))


def _r_fact_digest(frozen: object) -> str:
    value = thaw_json(frozen)  # type: ignore[arg-type]
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidenceCoreProjectionError(
            "reference model/route facts are invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _r_object(value: Mapping[str, Any], label: str) -> FrozenObject:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenObject):
        raise EvidenceCoreProjectionError(f"reference {label} is not an object")
    return frozen


def _r_failure(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    stage = getattr(value, "stage")
    index = getattr(value, "call_unit_index")
    detail = getattr(value, "detail_code")
    return {"stage": stage, "call_unit_index": index, "detail_code": detail}


def _r_lane(value: object) -> dict[str, Any]:
    counters = getattr(value, "counters")
    status = getattr(value, "status")
    detail = getattr(value, "terminal_detail_code")
    return {
        "status": status,
        "states": counters.states,
        "edges": counters.edges,
        "intervals": counters.intervals,
        "normalize_chars": counters.normalize_chars,
        "terminal_detail_code": detail,
    }


def _r_denied(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    counters: list[dict[str, Any]] = []
    for counter in getattr(value, "counters"):
        counters.append(
            {
                "counter": counter.counter,
                "amount": counter.amount,
                "scopes": [scope for scope in counter.scopes],
            }
        )
    return {
        "lane": getattr(value, "lane"),
        "event_ordinal": getattr(value, "event_ordinal"),
        "event_kind": getattr(value, "event_kind"),
        "subject": [member for member in getattr(value, "subject")],
        "counters": counters,
    }


def _r_work(work: AuthorityJobWorkReceipt) -> dict[str, Any]:
    mismatch_value: dict[str, Any] | None = None
    if work.route_mismatch is not None:
        mismatch_value = {
            "kind": work.route_mismatch.kind,
            "observation_index": work.route_mismatch.observation_index,
            "expected_delivery_index": work.route_mismatch.expected_delivery_index,
            "observed_delivery_index": work.route_mismatch.observed_delivery_index,
        }
    claims: list[dict[str, Any]] = []
    for claim in work.route_claims:
        claims.append(
            {
                "owner_kind": claim.owner_kind,
                "owner_index": claim.owner_index,
                "delivery_index": claim.delivery_index,
                "source_index": claim.source_index,
            }
        )
    calls: list[dict[str, Any]] = []
    for row in work.calls:
        calls.append(
            {
                "call_index": row.call_index,
                "route_claim_positions": [value for value in row.route_claim_positions],
                "source_block_indices": [value for value in row.source_block_indices],
                "raw_node_range": [row.raw_node_range[0], row.raw_node_range[1]],
                "block_count": row.block_count,
                "raw_node_count": row.raw_node_count,
                "typed_unit_count": row.typed_unit_count,
                "surface_chars": row.surface_chars,
                "strict_preflight_status": row.strict_preflight_status,
                "strict_failure": _r_failure(row.strict_failure),
                "limits": {
                    "state_limit": row.limits.state_limit,
                    "edge_limit": row.limits.edge_limit,
                    "interval_limit": row.limits.interval_limit,
                    "normalize_char_limit": row.limits.normalize_char_limit,
                },
                "allocator": _r_lane(row.allocator),
                "verifier": None if row.verifier is None else _r_lane(row.verifier),
            }
        )
    skipped: list[dict[str, Any]] = []
    for row in work.skipped_blocks:
        skipped.append(
            {
                "route_claim_positions": [value for value in row.route_claim_positions],
                "delivery_index": row.delivery_index,
                "source_index": row.source_index,
                "route_skip_reason": row.route_skip_reason,
                "source_text_kind": row.source_text_kind,
                "detail_code": row.detail_code,
                "work_status": row.work_status,
                "states": row.counters.states,
                "edges": row.counters.edges,
                "intervals": row.counters.intervals,
                "normalize_chars": row.counters.normalize_chars,
            }
        )
    return {
        "status": work.status,
        "route_status": work.route_status,
        "route_mismatch": mismatch_value,
        "route_claims": claims,
        "declared_delivery_block_count": work.declared_delivery_block_count,
        "declared_call_count": work.declared_call_count,
        "declared_skip_count": work.declared_skip_count,
        "declared_raw_node_count": work.declared_raw_node_count,
        "charged_call_count": work.charged_call_count,
        "limit_profile_kind": work.limit_profile_kind,
        "limit_profile_digest": work.limit_profile_digest,
        "limits": {
            "call_limit": work.limits.call_limit,
            "state_limit": work.limits.state_limit,
            "edge_limit": work.limits.edge_limit,
            "interval_limit": work.limits.interval_limit,
            "normalize_char_limit": work.limits.normalize_char_limit,
        },
        "totals": {
            "states": work.totals.states,
            "edges": work.totals.edges,
            "intervals": work.totals.intervals,
            "normalize_chars": work.totals.normalize_chars,
        },
        "terminal_call_index": work.terminal_call_index,
        "denied_charge": _r_denied(work.denied_charge),
        "calls": calls,
        "skipped_blocks": skipped,
    }


def _r_context_digest(facts: EvidenceCoreReferenceInputs) -> str:
    family = engine_family_for(facts.language)
    if family != facts.engine_family:
        raise EvidenceCoreProjectionError("reference registry family is inconsistent")
    return frozen_json_digest(
        FrozenArray(
            (
                FrozenString("align-context-v2"),
                facts.stable_fields,
                FrozenString(facts.authority_profile.kind),
                FrozenString(facts.authority_profile.profile_digest),
                FrozenString(facts.language),
                FrozenString(facts.route_kind),
                FrozenString(family),
            )
        )
    )


def _r_history(facts: EvidenceCoreReferenceInputs) -> dict[str, Any]:
    stable = thaw_json(facts.stable_fields)
    if not isinstance(stable, dict):
        raise EvidenceCoreProjectionError("reference stable context is not an object")
    vtt = stable.get("vtt_generation")
    sibling = stable.get("sibling_generation")
    adapter = stable.get("adapter_inputs")
    if not isinstance(vtt, Mapping):
        vtt = {}
    if not isinstance(sibling, Mapping):
        sibling = {}
    if not isinstance(adapter, Mapping):
        adapter = {}
    policy = adapter.get("legacy_policy")
    if not isinstance(policy, Mapping):
        policy = {}
    policy_bits: dict[str, str] = {}
    try:
        for name in ("min_cue_sec", "tiny_cue_sec", "tiny_cue_target"):
            policy_bits[name] = struct.pack(">d", policy.get(name, 0.0)).hex()
    except (TypeError, ValueError, struct.error) as exc:
        raise EvidenceCoreProjectionError("reference legacy policy is invalid") from exc
    return {
        "vtt_present": vtt.get("present"),
        "vtt_size": vtt.get("size"),
        "vtt_sha256": vtt.get("sha256"),
        "sibling_json_present": sibling.get("present"),
        "sibling_json_size": sibling.get("size"),
        "sibling_json_sha256": sibling.get("sha256"),
        "block_content_sha256": stable.get("block_content_sha256"),
        "registry_family": engine_family_for(facts.language),
        "media_fingerprint": stable.get("media_fingerprint"),
        "media_logical_id": stable.get("media_logical_id"),
        "media_display_name": stable.get("media_display_name"),
        "prepared_audio_size": stable.get("prepared_audio_size"),
        "prepared_audio_sha256": stable.get("prepared_audio_sha256"),
        "profile_input_sha256": stable.get("profile_input_sha256"),
        "evidence_carriers_sha256": stable.get("evidence_carriers_sha256"),
        "authority_limit_profile_kind": facts.authority_profile.kind,
        "authority_limit_profile_digest": facts.authority_profile.profile_digest,
        "legacy_policy_binary64": policy_bits,
        "target_logical_id": stable.get("target_logical_id"),
        "expected_vtt_sha256": stable.get("expected_vtt_sha256"),
    }


def _r_source_facts(facts: EvidenceCoreReferenceInputs) -> dict[str, Any]:
    model = thaw_json(facts.backend_model_config_facts)
    route = thaw_json(facts.route_input_facts)
    if not isinstance(model, dict) or not isinstance(route, dict):
        raise EvidenceCoreProjectionError("reference durable source facts are invalid")
    return {"backend_model_config": model, "route_input": route}


def _r_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if type(value) is not float:
        raise TypeError("reference relative bound is not an exact float or null")
    return value


def _r_capture(call: ReferencePhysicalCallFacts) -> StrictCaptureResult:
    if len(call.raw_nodes) != len(call.raw_unit_ids):
        raise EvidenceCoreProjectionError("reference raw ID cardinality mismatch")
    if call.original_nodes is not None and len(call.original_nodes) != len(
        call.raw_nodes
    ):
        raise EvidenceCoreProjectionError(
            "reference original-node cardinality mismatch"
        )
    units: list[StrictCapturedUnit] = []
    raw_values: list[object] = []
    for index, node in enumerate(call.raw_nodes):
        try:
            if not isinstance(node, Mapping):
                raise TypeError
            if "text" not in node or "start" not in node or "end" not in node:
                raise TypeError
            surface = node["text"]
            if type(surface) is not str:
                raise TypeError
            start = _r_optional_float(node["start"])
            end = _r_optional_float(node["end"])
            raw = freeze_json(node)
            if call.original_nodes is None:
                original_start = start
                original_end = end
                provenance: Literal["aligner", "align-interpolated"] = "aligner"
            else:
                original = call.original_nodes[index]
                if not isinstance(original, Mapping):
                    raise TypeError
                if "start" not in original or "end" not in original:
                    raise TypeError
                original_start = _r_optional_float(original["start"])
                original_end = _r_optional_float(original["end"])
                same_start = (
                    start is original_start
                    if start is None or original_start is None
                    else start.hex() == original_start.hex()
                )
                same_end = (
                    end is original_end
                    if end is None or original_end is None
                    else end.hex() == original_end.hex()
                )
                provenance = (
                    "aligner" if same_start and same_end else "align-interpolated"
                )
            unit = StrictCapturedUnit(
                call.raw_unit_ids[index],
                call.call_index,
                index,
                surface,
                start,
                end,
                provenance,
                original_start,
                original_end,
                raw,
            )
        except Exception:
            return StrictCaptureResult(
                call.call_index,
                "invalid",
                None,
                None,
                None,
                StrictFailureLocator("strict-capture", index, "strict-raw-node"),
                call.raw_unit_ids,
            )
        units.append(unit)
        raw_values.append(raw)
    raw_digest = frozen_json_digest(FrozenArray(tuple(raw_values)))
    relative_rows = []
    for unit in units:
        relative_rows.append(
            FrozenArray(
                (
                    freeze_json(unit.unit_id),
                    freeze_json(unit.call_index),
                    freeze_json(unit.call_unit_index),
                    freeze_json(unit.surface),
                    FROZEN_NULL
                    if unit.relative_start is None
                    else freeze_json(unit.relative_start),
                    FROZEN_NULL
                    if unit.relative_end is None
                    else freeze_json(unit.relative_end),
                    freeze_json(unit.provenance),
                    FROZEN_NULL
                    if unit.original_relative_start is None
                    else freeze_json(unit.original_relative_start),
                    FROZEN_NULL
                    if unit.original_relative_end is None
                    else freeze_json(unit.original_relative_end),
                    unit.raw,
                )
            )
        )
    return StrictCaptureResult(
        call.call_index,
        "valid",
        tuple(units),
        raw_digest,
        frozen_json_digest(FrozenArray(tuple(relative_rows))),
        None,
        call.raw_unit_ids,
    )


def _r_legacy_count(text: str, language: str) -> int:
    stripped = (text or "").strip()
    if language in LANGUAGES_WITHOUT_SPACES:
        return sum(1 for character in stripped if character.isalnum())
    return len(stripped.split())


def _r_legacy_call(
    call: ReferencePhysicalCallFacts,
    blocks: Mapping[int, AuthorityBlock],
    language: str,
) -> tuple[
    LegacyCallDistributionReceipt,
    tuple[tuple[Mapping[str, Any], ...], ...],
    tuple[tuple[Mapping[str, Any], ...], ...],
]:
    expected: list[int] = []
    requested: list[tuple[int, int]] = []
    realized: list[tuple[int, int]] = []
    owner_ids: list[tuple[str, ...]] = []
    owners: list[tuple[Mapping[str, Any], ...]] = []
    relative_owners: list[tuple[Mapping[str, Any], ...]] = []
    shortages: list[int] = []
    cursor = 0
    raw_count = len(call.raw_nodes)
    for source in call.source_block_indices:
        if source not in blocks:
            raise EvidenceCoreProjectionError("reference legacy owner is unknown")
        count = _r_legacy_count(blocks[source].alignment_text, language)
        lower, upper = cursor, cursor + count
        low_clamp, high_clamp = min(lower, raw_count), min(upper, raw_count)
        expected.append(count)
        requested.append((lower, upper))
        realized.append((low_clamp, high_clamp))
        owner_ids.append(call.raw_unit_ids[lower:upper])
        if high_clamp - low_clamp < count:
            shortages.append(source)
        projected: list[Mapping[str, Any]] = []
        relative_projected: list[Mapping[str, Any]] = []
        for node in call.raw_nodes[lower:upper]:
            if not isinstance(node, Mapping):
                raise EvidenceCoreProjectionError(
                    "reference retained legacy unit is invalid"
                )
            try:
                text = node["text"]
                start = node["start"]
                end = node["end"]
                relative_projected.append({"text": text, "start": start, "end": end})
                if call.legacy_origin_kind != "identity":
                    start = start + call.legacy_origin_seconds
                    end = end + call.legacy_origin_seconds
                projected.append({"text": text, "start": start, "end": end})
            except (KeyError, TypeError, ValueError) as exc:
                raise EvidenceCoreProjectionError(
                    "reference retained legacy projection failed"
                ) from exc
        owners.append(tuple(projected))
        relative_owners.append(tuple(relative_projected))
        cursor = upper
    consumed = min(cursor, raw_count)
    receipt = LegacyCallDistributionReceipt(
        tuple(call.source_block_indices),
        tuple(expected),
        tuple(requested),
        tuple(realized),
        tuple(owner_ids),
        cursor,
        call.raw_unit_ids[:consumed],
        tuple(shortages),
        call.raw_unit_ids[consumed:],
    )
    return receipt, tuple(owners), tuple(relative_owners)


def _r_transform(
    call: ReferencePhysicalCallFacts,
    capture: StrictCaptureResult,
    retained_count: int,
) -> AuthorityTransformResult:
    if call.geometry_failure is not None:
        return AuthorityTransformResult(
            call.call_index,
            "invalid",
            capture,
            None,
            None,
            call.geometry_failure,
        )
    if capture.status != "valid" or capture.units is None:
        return AuthorityTransformResult(
            call.call_index,
            "invalid",
            capture,
            None,
            None,
            capture.failure,
        )
    transformed: list[FreshUnit] = []
    for index, unit in enumerate(capture.units):
        start = unit.relative_start
        end = unit.relative_end
        valid = (
            start is not None
            and end is not None
            and math.isfinite(start)
            and math.isfinite(end)
            and start <= end
        )
        if not valid:
            detail = (
                "authority-recompute" if index < retained_count else "surplus-transform"
            )
            return AuthorityTransformResult(
                call.call_index,
                "invalid",
                capture,
                None,
                None,
                StrictFailureLocator("authority-transform", index, detail),
            )
        assert start is not None and end is not None
        absolute_start = (
            start
            if call.legacy_origin_kind == "identity"
            else start + call.physical_origin_seconds
        )
        absolute_end = (
            end
            if call.legacy_origin_kind == "identity"
            else end + call.physical_origin_seconds
        )
        if (
            not math.isfinite(absolute_start)
            or not math.isfinite(absolute_end)
            or absolute_start > absolute_end
        ):
            detail = (
                "authority-recompute" if index < retained_count else "surplus-transform"
            )
            return AuthorityTransformResult(
                call.call_index,
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
                start,
                end,
                call.physical_origin_seconds,
                absolute_start,
                absolute_end,
                unit.provenance,
                unit.original_relative_start,
                unit.original_relative_end,
                unit.raw,
            )
        )
    absolute_rows = []
    for unit in transformed:
        absolute_rows.append(
            FrozenArray(
                (
                    freeze_json(unit.unit_id),
                    freeze_json(unit.call_index),
                    freeze_json(unit.call_unit_index),
                    freeze_json(unit.surface),
                    FROZEN_NULL
                    if unit.relative_start is None
                    else freeze_json(unit.relative_start),
                    FROZEN_NULL
                    if unit.relative_end is None
                    else freeze_json(unit.relative_end),
                    freeze_json(unit.physical_origin_seconds),
                    FROZEN_NULL if unit.start is None else freeze_json(unit.start),
                    FROZEN_NULL if unit.end is None else freeze_json(unit.end),
                    freeze_json(unit.provenance),
                    FROZEN_NULL
                    if unit.original_relative_start is None
                    else freeze_json(unit.original_relative_start),
                    FROZEN_NULL
                    if unit.original_relative_end is None
                    else freeze_json(unit.original_relative_end),
                    unit.raw,
                )
            )
        )
    return AuthorityTransformResult(
        call.call_index,
        "valid",
        capture,
        tuple(transformed),
        frozen_json_digest(FrozenArray(tuple(absolute_rows))),
        None,
    )


def _r_receipt_equal(left: PhysicalCallReceipt, right: PhysicalCallReceipt) -> bool:
    for member in dataclasses.fields(PhysicalCallReceipt):
        a = getattr(left, member.name)
        b = getattr(right, member.name)
        if type(a) is float and type(b) is float:
            if a.hex() != b.hex():
                return False
        elif a != b:
            return False
    return True


def _r_rebuild_calls(
    facts: EvidenceCoreReferenceInputs,
) -> tuple[
    tuple[EvidenceCorePhysicalCall, ...],
    tuple[StrictCaptureResult, ...],
    tuple[AuthorityTransformResult, ...],
    tuple[LegacyCallDistributionReceipt, ...],
    tuple[tuple[Mapping[str, Any], ...], ...],
    tuple[tuple[Mapping[str, Any], ...], ...],
    tuple[PhysicalCallReceipt, ...],
]:
    count = len(facts.reference_calls)
    if not (
        count
        == len(facts.captures)
        == len(facts.transforms)
        == len(facts.legacy_receipts)
        == len(facts.claimed_physical_calls)
    ):
        raise EvidenceCoreProjectionError("reference physical cardinality mismatch")
    model_digest = _r_fact_digest(facts.backend_model_config_facts)
    route_digest = _r_fact_digest(facts.route_input_facts)
    blocks_by_source = {block.source_index: block for block in facts.blocks}
    global_legacy = {
        block.source_index: owner
        for block, owner in zip(facts.blocks, facts.legacy_block_units, strict=True)
    }
    evidence_calls: list[EvidenceCorePhysicalCall] = []
    captures: list[StrictCaptureResult] = []
    transforms: list[AuthorityTransformResult] = []
    receipts: list[LegacyCallDistributionReceipt] = []
    expected_physical: list[PhysicalCallReceipt] = []
    ordered_legacy: dict[int, tuple[Mapping[str, Any], ...]] = {}
    ordered_relative_legacy: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for index, raw_call in enumerate(facts.reference_calls):
        if raw_call.call_index != index:
            raise EvidenceCoreProjectionError("reference call order is invalid")
        if (
            type(raw_call.sample_rate) is not int
            or raw_call.sample_rate <= 0
            or type(raw_call.sample_count) is not int
            or raw_call.sample_count < 0
            or not 0
            <= raw_call.audio_sample_start
            <= raw_call.audio_sample_end
            <= raw_call.sample_count
        ):
            raise EvidenceCoreProjectionError("reference physical geometry is invalid")
        quotient = raw_call.audio_sample_start / raw_call.sample_rate
        if (
            raw_call.physical_origin_seconds.hex() != quotient.hex()
            or raw_call.authority_origin_seconds.hex() != quotient.hex()
        ):
            raise EvidenceCoreProjectionError("reference physical origin is invalid")
        if raw_call.legacy_origin_kind == "identity":
            if (
                raw_call.audio_sample_start != 0
                or raw_call.legacy_origin_seconds.hex() != (0.0).hex()
            ):
                raise EvidenceCoreProjectionError(
                    "reference identity origin is invalid"
                )
        elif raw_call.legacy_origin_kind == "sample-origin":
            if raw_call.legacy_origin_seconds.hex() != quotient.hex():
                raise EvidenceCoreProjectionError("reference sample origin is invalid")
        elif raw_call.legacy_origin_kind != "nominal-route":
            raise EvidenceCoreProjectionError("reference origin kind is invalid")
        capture = _r_capture(raw_call)
        legacy_receipt, call_legacy, call_relative_legacy = _r_legacy_call(
            raw_call, blocks_by_source, facts.language
        )
        retained_count = len(legacy_receipt.consumed_prefix_unit_ids)
        transform = _r_transform(raw_call, capture, retained_count)
        if capture != facts.captures[index]:
            raise EvidenceCoreProjectionError(
                "reference strict snapshot cross-link mismatch"
            )
        if transform != facts.transforms[index]:
            raise EvidenceCoreProjectionError(
                "reference authority snapshot cross-link mismatch"
            )
        if legacy_receipt != facts.legacy_receipts[index]:
            raise EvidenceCoreProjectionError(
                "reference legacy receipt cross-link mismatch"
            )
        for source, owner in zip(
            raw_call.source_block_indices, call_legacy, strict=True
        ):
            if global_legacy.get(source) != owner:
                raise EvidenceCoreProjectionError(
                    "reference legacy block cross-link mismatch"
                )
            ordered_legacy[source] = owner
        for source, owner in zip(
            raw_call.source_block_indices, call_relative_legacy, strict=True
        ):
            ordered_relative_legacy[source] = owner
        legacy_slice_digest = _r_stable_digest(legacy_receipt)
        legacy_absolute_digest = _r_stable_digest(call_legacy)
        physical = PhysicalCallReceipt(
            call_index=raw_call.call_index,
            source_block_indices=raw_call.source_block_indices,
            audio_sample_start=raw_call.audio_sample_start,
            audio_sample_end=raw_call.audio_sample_end,
            sample_rate=raw_call.sample_rate,
            physical_origin_seconds=raw_call.physical_origin_seconds,
            legacy_origin_seconds=raw_call.legacy_origin_seconds,
            legacy_origin_kind=raw_call.legacy_origin_kind,
            authority_origin_seconds=raw_call.authority_origin_seconds,
            backend_model_config_digest=model_digest,
            route_input_digest=route_digest,
            strict_unit_status=capture.status,
            strict_failure=transform.failure,
            raw_units_digest=capture.raw_units_digest,
            normalized_relative_digest=capture.normalized_relative_digest,
            legacy_slice_digest=legacy_slice_digest,
            legacy_absolute_digest=legacy_absolute_digest,
            authority_transform_status=transform.status,
            authority_absolute_digest=transform.authority_absolute_digest,
            raw_unit_ids=raw_call.raw_unit_ids,
        )
        if not _r_receipt_equal(physical, facts.claimed_physical_calls[index]):
            raise EvidenceCoreProjectionError(
                "reference physical digest/cross-link mismatch"
            )
        evidence_calls.append(
            EvidenceCorePhysicalCall(
                physical.call_index,
                physical.source_block_indices,
                physical.audio_sample_start,
                physical.audio_sample_end,
                physical.sample_rate,
                physical.physical_origin_seconds,
                physical.legacy_origin_seconds,
                physical.legacy_origin_kind,
                physical.authority_origin_seconds,
                physical.backend_model_config_digest,
                physical.route_input_digest,
                physical.strict_unit_status,
                physical.strict_failure,
                physical.raw_units_digest,
                physical.normalized_relative_digest,
                physical.legacy_slice_digest,
                physical.legacy_absolute_digest,
                physical.authority_transform_status,
                physical.authority_absolute_digest,
                physical.raw_unit_ids,
            )
        )
        captures.append(capture)
        transforms.append(transform)
        receipts.append(legacy_receipt)
        expected_physical.append(physical)
    legacy_blocks = tuple(
        ordered_legacy.get(block.source_index, ()) for block in facts.blocks
    )
    if legacy_blocks != facts.legacy_block_units:
        raise EvidenceCoreProjectionError("reference legacy block inventory mismatch")
    relative_legacy_blocks = tuple(
        ordered_relative_legacy.get(block.source_index, ()) for block in facts.blocks
    )
    if relative_legacy_blocks != facts.legacy_relative_block_units:
        raise EvidenceCoreProjectionError(
            "reference relative legacy block inventory mismatch"
        )
    return (
        tuple(evidence_calls),
        tuple(captures),
        tuple(transforms),
        tuple(receipts),
        legacy_blocks,
        relative_legacy_blocks,
        tuple(expected_physical),
    )


def _r_replay_distribution(
    facts: EvidenceCoreReferenceInputs,
    captures: tuple[StrictCaptureResult, ...],
    transforms: tuple[AuthorityTransformResult, ...],
) -> tuple[str, ...]:
    from voxweave.align_distribution_reference import (
        DistributionReferenceError,
        replay_authority_distribution,
    )

    calls: list[AuthorityCallInput] = []
    for raw, capture, transform in zip(
        facts.reference_calls, captures, transforms, strict=True
    ):
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
        calls.append(
            AuthorityCallInput(
                raw.call_index,
                raw.source_block_indices,
                raw.raw_node_range,
                raw.raw_unit_ids,
                surfaces,
                preflight,
                failure,
            )
        )
    called_sources = {source for call in calls for source in call.source_block_indices}
    skipped: list[AuthoritySkippedBlockInput] = []
    for delivery_index, block in enumerate(facts.blocks):
        if block.source_index in called_sources:
            continue
        stripped = block.alignment_text.strip()
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
    try:
        replay_authority_distribution(
            blocks=facts.blocks,
            calls=tuple(calls),
            skipped=tuple(skipped),
            receipt=facts.distribution,
            iso=facts.language,
        )
    except DistributionReferenceError as exc:
        raise EvidenceCoreProjectionError(
            "reference allocator receipt replay failed"
        ) from exc
    work = facts.distribution.work
    if work.status == "seal-mismatch":
        raise EvidenceCoreProjectionError("seal mismatch cannot reach EvidenceCore")
    present: set[str] = set()
    if work.route_status == "invalid":
        if work.route_mismatch is None:
            raise EvidenceCoreProjectionError("reference route mismatch is absent")
        present.add("route-owner-mismatch")
    elif work.route_mismatch is not None:
        raise EvidenceCoreProjectionError("reference valid route carries mismatch")
    if work.skipped_blocks:
        present.add("partial-empty-ownership")
    if any(row.strict_failure is not None for row in work.calls):
        present.add("authority-transform-invalid")
    for row in work.calls:
        if row.allocator.terminal_detail_code in {
            "partial-empty-ownership",
            "punctuation-only-block",
            "allocation-no-tiling",
            "allocation-ambiguous",
        }:
            present.add(row.allocator.terminal_detail_code)
    if work.status == "budget-exhausted":
        present.add("allocation-budget-exhausted")
    reasons = tuple(reason for reason in AUTHORITY_REASON_ORDER if reason in present)
    if reasons != facts.distribution.reasons:
        raise EvidenceCoreProjectionError("reference authority reasons disagree")
    return reasons


def _r_route(
    facts: EvidenceCoreReferenceInputs,
    calls: tuple[EvidenceCorePhysicalCall, ...],
) -> dict[str, Any]:
    stable = thaw_json(facts.stable_fields)
    block_values = stable.get("blocks", []) if isinstance(stable, dict) else []
    source_calls: dict[int, int] = {}
    for call in calls:
        for source in call.source_block_indices:
            if source in source_calls:
                raise EvidenceCoreProjectionError("reference duplicate route owner")
            source_calls[source] = call.call_index
    skipped = {row.source_index: row for row in facts.distribution.work.skipped_blocks}
    by_index = {call.call_index: call for call in calls}
    entries: list[dict[str, Any]] = []
    for delivery_index, block in enumerate(block_values):
        if not isinstance(block, Mapping):
            raise EvidenceCoreProjectionError("reference route block is invalid")
        source = block.get("source_index")
        call_index = source_calls.get(source) if type(source) is int else None
        if facts.route_kind == "ctc-full" or facts.route_kind == "mms-full":
            action = "full-pass-member"
            skip_reason = None
        elif call_index is not None:
            action = "qwen-call"
            skip_reason = None
        else:
            action = "qwen-skip"
            row = skipped.get(source) if type(source) is int else None
            skip_reason = None if row is None else row.route_skip_reason
        route_start = block.get("start")
        route_end = block.get("end")
        if facts.route_kind == "qwen-crop" and call_index is not None:
            call = by_index[call_index]
            route_start = call.legacy_origin_seconds
            if type(route_end) is not float:
                route_end = call.sample_end / call.sample_rate
        entries.append(
            {
                "delivery_index": delivery_index,
                "source_index": source,
                "route_start": route_start,
                "route_end": route_end,
                "action": action,
                "call_index": call_index,
                "skip_reason": skip_reason,
            }
        )
    return {
        "digest": _r_labeled_digest("p6-route-plan-v1", entries),
        "entries": entries,
    }


def _r_legacy(receipts: tuple[LegacyCallDistributionReceipt, ...]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        calls.append(
            {
                "call_index": index,
                "owner_source_indices": [
                    value for value in receipt.owner_source_indices
                ],
                "expected_counts": [value for value in receipt.expected_counts],
                "requested_ranges": [
                    [lower, upper] for lower, upper in receipt.requested_ranges
                ],
                "realized_ranges": [
                    [lower, upper] for lower, upper in receipt.realized_ranges
                ],
                "owner_unit_ids": [list(ids) for ids in receipt.owner_unit_ids],
                "final_cursor": receipt.final_cursor,
                "consumed_prefix_unit_ids": list(receipt.consumed_prefix_unit_ids),
                "shortage_source_indices": list(receipt.shortage_source_indices),
                "leftover_unit_ids": list(receipt.leftover_unit_ids),
            }
        )
    return {
        "digest": _r_labeled_digest("p6-legacy-distribution-v1", calls),
        "calls": calls,
    }


def _r_authority(
    facts: EvidenceCoreReferenceInputs, reasons: tuple[str, ...]
) -> dict[str, Any]:
    distribution = facts.distribution
    valid = distribution.status == "valid"
    body: dict[str, Any] = {
        "status": distribution.status,
        "owner_source_indices": list(distribution.owner_source_indices)
        if valid
        else None,
        "expected_counts": list(distribution.expected_counts or ()) if valid else None,
        "owner_unit_ids": [list(owner) for owner in distribution.owners or ()]
        if valid
        else None,
        "consumed_count": distribution.consumed_count,
        "leftover_unit_ids": list(distribution.leftovers),
        "reasons": [reason for reason in reasons],
        "work": _r_work(distribution.work),
    }
    result: dict[str, Any] = {
        "status": distribution.status,
        "digest": _r_labeled_digest("p6-authority-distribution-v1", body),
    }
    for key, value in body.items():
        if key != "status":
            result[key] = value
    return result


def _r_word(unit: FreshUnit) -> EvidenceCoreWord:
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


def _r_blocks(
    facts: EvidenceCoreReferenceInputs,
    transforms: tuple[AuthorityTransformResult, ...],
    receipts: tuple[LegacyCallDistributionReceipt, ...],
) -> tuple[EvidenceCoreBlock, ...]:
    legacy: dict[int, tuple[str, ...]] = {}
    for receipt in receipts:
        for source, ids in zip(
            receipt.owner_source_indices, receipt.owner_unit_ids, strict=True
        ):
            if source in legacy:
                raise EvidenceCoreProjectionError("reference duplicate legacy owner")
            legacy[source] = tuple(ids)
    fresh: dict[str, FreshUnit] = {}
    for transform in transforms:
        for unit in transform.units or ():
            if unit.unit_id in fresh:
                raise EvidenceCoreProjectionError("reference duplicate fresh unit")
            fresh[unit.unit_id] = unit
    out: list[EvidenceCoreBlock] = []
    if facts.distribution.status == "valid":
        if facts.distribution.owners is None:
            raise EvidenceCoreProjectionError("reference valid authority lacks owners")
        for source, ids in zip(
            facts.distribution.owner_source_indices,
            facts.distribution.owners,
            strict=True,
        ):
            try:
                owned = tuple(fresh[unit_id] for unit_id in ids)
            except KeyError as exc:
                raise EvidenceCoreProjectionError(
                    "reference owner lacks transformed unit"
                ) from exc
            if not owned:
                start = None
                end = None
            else:
                first, last = owned[0], owned[-1]
                start = (
                    first.start
                    if first.provenance == "aligner"
                    and first.start is not None
                    and math.isfinite(first.start)
                    else None
                )
                end = (
                    last.end
                    if last.provenance == "aligner"
                    and last.end is not None
                    and math.isfinite(last.end)
                    else None
                )
            out.append(
                EvidenceCoreBlock(
                    source,
                    legacy.get(source, ()),
                    tuple(ids),
                    tuple(_r_word(unit) for unit in owned),
                    start,
                    end,
                )
            )
    else:
        for block in facts.blocks:
            out.append(
                EvidenceCoreBlock(
                    block.source_index,
                    legacy.get(block.source_index, ()),
                    None,
                    None,
                    None,
                    None,
                )
            )
    return tuple(out)


def _r_block_values(blocks: tuple[EvidenceCoreBlock, ...]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for block in blocks:
        words = None
        if block.word_data is not None:
            words = []
            for word in block.word_data:
                words.append(
                    {
                        "unit_id": word.unit_id,
                        "call_index": word.call_index,
                        "call_unit_index": word.call_unit_index,
                        "text": word.text,
                        "relative_start": word.relative_start,
                        "relative_end": word.relative_end,
                        "physical_origin_seconds": word.physical_origin_seconds,
                        "start": word.start,
                        "end": word.end,
                        "provenance": word.provenance,
                        "original_relative_start": word.original_relative_start,
                        "original_relative_end": word.original_relative_end,
                    }
                )
        values.append(
            {
                "source_index": block.source_index,
                "legacy_unit_ids": list(block.legacy_unit_ids),
                "authority_unit_ids": None
                if block.authority_unit_ids is None
                else list(block.authority_unit_ids),
                "word_data": words,
                "speech_start": block.speech_start,
                "speech_end": block.speech_end,
            }
        )
    return values


def _r_physical_values(
    calls: tuple[EvidenceCorePhysicalCall, ...],
    blocks: tuple[EvidenceCoreBlock, ...],
    legacy_blocks: tuple[tuple[Mapping[str, Any], ...], ...],
) -> list[dict[str, Any]]:
    legacy_by_source = {
        block.source_index: units
        for block, units in zip(blocks, legacy_blocks, strict=True)
    }
    values: list[dict[str, Any]] = []
    for row in calls:
        values.append(
            {
                "call_index": row.call_index,
                "source_block_indices": list(row.source_block_indices),
                "sample_start": row.sample_start,
                "sample_end": row.sample_end,
                "sample_rate": row.sample_rate,
                "physical_origin_seconds": row.physical_origin_seconds,
                "legacy_origin_seconds": row.legacy_origin_seconds,
                "legacy_origin_kind": row.legacy_origin_kind,
                "authority_origin_seconds": row.authority_origin_seconds,
                "backend_model_config_sha256": row.backend_model_config_sha256,
                "route_input_sha256": row.route_input_sha256,
                "strict_unit_status": row.strict_unit_status,
                "strict_failure": _r_failure(row.strict_failure),
                "raw_units_sha256": row.raw_units_sha256,
                "relative_units_sha256": row.relative_units_sha256,
                "legacy_retained_units": [
                    _r_stable_value(legacy_by_source.get(source, ()))
                    for source in row.source_block_indices
                ],
                "legacy_slice_sha256": row.legacy_slice_sha256,
                "legacy_absolute_sha256": row.legacy_absolute_sha256,
                "authority_transform_status": row.authority_transform_status,
                "authority_absolute_sha256": row.authority_absolute_sha256,
                "raw_unit_ids": list(row.raw_unit_ids),
            }
        )
    return values


def _r_status(value: object) -> dict[str, Any]:
    kind = getattr(value, "kind")
    detail = getattr(value, "detail_code")
    return {"kind": kind, "detail_code": detail}


def _r_profile(value: object) -> dict[str, Any]:
    kind = getattr(value, "kind")
    source = getattr(value, "source")
    detail = getattr(value, "detail_code")
    return {"kind": kind, "source": source, "detail_code": detail}


def project_evidence_core(facts: EvidenceCoreReferenceInputs) -> EvidenceCore:
    """Independently reconstruct every §9 field before selected outputs exist."""
    if not isinstance(facts, EvidenceCoreReferenceInputs):
        raise TypeError("reference EvidenceCore inputs have the wrong type")
    context_digest = _r_context_digest(facts)
    if context_digest != facts.claimed_context_content_digest:
        raise EvidenceCoreProjectionError(
            "reference context digest cross-link mismatch"
        )
    if facts.seed_status not in ("valid", "invalid"):
        raise EvidenceCoreProjectionError("reference seed status is invalid")
    ordered_reasons = tuple(
        reason for reason in SEED_REASON_ORDER if reason in set(facts.seed_reasons)
    )
    if ordered_reasons != facts.seed_reasons:
        raise EvidenceCoreProjectionError("reference seed reason order is invalid")
    (
        calls,
        captures,
        transforms,
        legacy_receipts,
        legacy_blocks,
        relative_legacy_blocks,
        physical_receipts,
    ) = _r_rebuild_calls(facts)
    authority_reasons = _r_replay_distribution(facts, captures, transforms)
    receipt_digest = _r_stable_digest(
        {
            "context_content_digest": context_digest,
            "physical_calls": physical_receipts,
            "legacy_distribution": legacy_receipts,
            "authority_distribution": facts.distribution,
            "seed_status": facts.seed_status,
            "seed_reasons": facts.seed_reasons,
        }
    )
    if receipt_digest != facts.claimed_receipt_digest:
        raise EvidenceCoreProjectionError(
            "reference receipt digest cross-link mismatch"
        )
    history_value = _r_history(facts)
    route_value = _r_route(facts, calls)
    legacy_value = _r_legacy(legacy_receipts)
    authority_value = _r_authority(facts, authority_reasons)
    blocks = _r_blocks(facts, transforms, legacy_receipts)
    strict_value = _r_status(facts.strict_input_status)
    policy_value = _r_status(facts.v2_policy_status)
    profile_value = _r_profile(facts.profile_status)
    evidence_value = _r_status(facts.evidence_status)
    admission_valid = (
        strict_value["kind"] == "valid"
        and all(call.authority_transform_status == "valid" for call in calls)
        and facts.distribution.status == "valid"
        and facts.seed_status == "valid"
        and policy_value["kind"] == "valid"
        and profile_value["kind"] == "valid"
        and evidence_value["kind"] == "valid"
    )
    raw_count = sum(len(call.raw_unit_ids) for call in facts.reference_calls)
    projection_value = {
        "schema_version": 8,
        "kind": "fresh-alignment",
        "context_content_digest": context_digest,
        "receipt_digest": receipt_digest,
        "language": facts.language,
        "route": facts.route_kind,
        "source_facts": _r_source_facts(facts),
        "input_history": history_value,
        "route_plan": route_value,
        "physical_calls": _r_physical_values(calls, blocks, relative_legacy_blocks),
        "legacy_distribution": legacy_value,
        "authority_distribution": authority_value,
        "blocks": _r_block_values(blocks),
        "raw_unit_count": raw_count,
        "strict_input_status": strict_value,
        "seed_status": {"kind": facts.seed_status, "reasons": list(facts.seed_reasons)},
        "v2_policy_status": policy_value,
        "profile_status": profile_value,
        "evidence_status": evidence_value,
        "v2_admission_status": "valid" if admission_valid else "invalid",
    }
    projection = _r_object(projection_value, "EvidenceCore")
    return EvidenceCore(
        8,
        context_digest,
        receipt_digest,
        facts.language,
        facts.route_kind,
        _r_object(history_value, "input history"),
        _r_object(route_value, "route plan"),
        calls,
        _r_object(legacy_value, "legacy distribution"),
        _r_object(authority_value, "authority distribution"),
        blocks,
        raw_count,
        _r_object(strict_value, "strict status"),
        facts.seed_status,
        facts.seed_reasons,
        _r_object(policy_value, "policy status"),
        _r_object(profile_value, "profile status"),
        _r_object(evidence_value, "evidence status"),
        "valid" if admission_valid else "invalid",
        facts.distribution.status,
        authority_reasons,
        facts.distribution.work,
        tuple(row.surface_chars for row in facts.distribution.work.calls),
        projection,
    )


def evidence_core_value(core: EvidenceCore) -> dict[str, Any]:
    """Thaw one already-ALD-6-approved pre-selected-output projection."""
    if not isinstance(core, EvidenceCore):
        raise TypeError("core must be an EvidenceCore")
    value = thaw_json(core._projection)
    if not isinstance(value, dict):
        raise EvidenceCoreProjectionError("EvidenceCore projection is not an object")
    return value


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
    "evidence_core_value",
    "project_evidence_core",
]
