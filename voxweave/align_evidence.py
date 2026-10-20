"""Closed RAT-2 evidence binding and independent durable verification.

The binder receives only already-issued context/acquisition values and the
producer EvidenceCore, which AO-16 has already checked against the independent
reference projection. The path verifier deliberately does not consult any
in-memory issuer registry.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from voxweave import artifacts
from voxweave.align_acquisition import IssuedFreshAlignment
from voxweave.align_context import (
    IssuedAlignContext,
    consume_context_role,
)
from voxweave.align_evidence_core import (
    EvidenceCore,
    evidence_core_value,
)
from voxweave.align_evidence_schema import (
    _TOP_LEVEL_KEYS as _TOP_LEVEL_KEYS,
    _SOURCE_FACT_KEYS as _SOURCE_FACT_KEYS,
    _MODEL_FACT_KEYS as _MODEL_FACT_KEYS,
    _DEFAULT_MODEL_FACT_KEYS as _DEFAULT_MODEL_FACT_KEYS,
    _ROUTE_FACT_KEYS as _ROUTE_FACT_KEYS,
    _DEFAULT_ROUTE_FACT_KEYS as _DEFAULT_ROUTE_FACT_KEYS,
    _ROUTE_FACT_BLOCK_KEYS as _ROUTE_FACT_BLOCK_KEYS,
    _INPUT_HISTORY_KEYS as _INPUT_HISTORY_KEYS,
    _POLICY_KEYS as _POLICY_KEYS,
    _ROUTE_PLAN_KEYS as _ROUTE_PLAN_KEYS,
    _ROUTE_ENTRY_KEYS as _ROUTE_ENTRY_KEYS,
    _PHYSICAL_CALL_KEYS as _PHYSICAL_CALL_KEYS,
    _STRICT_FAILURE_KEYS as _STRICT_FAILURE_KEYS,
    _LEGACY_DISTRIBUTION_KEYS as _LEGACY_DISTRIBUTION_KEYS,
    _LEGACY_CALL_KEYS as _LEGACY_CALL_KEYS,
    _AUTHORITY_DISTRIBUTION_KEYS as _AUTHORITY_DISTRIBUTION_KEYS,
    _WORK_KEYS as _WORK_KEYS,
    _ROUTE_MISMATCH_KEYS as _ROUTE_MISMATCH_KEYS,
    _ROUTE_CLAIM_KEYS as _ROUTE_CLAIM_KEYS,
    _JOB_LIMIT_KEYS as _JOB_LIMIT_KEYS,
    _CALL_LIMIT_KEYS as _CALL_LIMIT_KEYS,
    _COUNTER_KEYS as _COUNTER_KEYS,
    _DENIED_CHARGE_KEYS as _DENIED_CHARGE_KEYS,
    _DENIED_COUNTER_KEYS as _DENIED_COUNTER_KEYS,
    _WORK_CALL_KEYS as _WORK_CALL_KEYS,
    _LANE_KEYS as _LANE_KEYS,
    _SKIP_KEYS as _SKIP_KEYS,
    _BLOCK_KEYS as _BLOCK_KEYS,
    _WORD_KEYS as _WORD_KEYS,
    _STRICT_STATUS_KEYS as _STRICT_STATUS_KEYS,
    _SEED_STATUS_KEYS as _SEED_STATUS_KEYS,
    _PROFILE_STATUS_KEYS as _PROFILE_STATUS_KEYS,
    _SELECTED_OUTPUT_KEYS as _SELECTED_OUTPUT_KEYS,
    _WORK_STATUSES as _WORK_STATUSES,
    _LANE_STATUSES as _LANE_STATUSES,
    _LANE_DETAILS as _LANE_DETAILS,
    _PROFILE_SOURCES as _PROFILE_SOURCES,
    _is_sha256 as _is_sha256,
    _same_float as _same_float,
    _same_optional_float as _same_optional_float,
    _digest_projection as _digest_projection,
    _DuplicateEvidenceKey as _DuplicateEvidenceKey,
    _EvidenceInvalid as _EvidenceInvalid,
    _invalid as _invalid,
    _object_without_duplicates as _object_without_duplicates,
    _reject_nonfinite as _reject_nonfinite,
    _closed_mapping as _closed_mapping,
    _list as _list,
    _exact_int as _exact_int,
    _nullable_int as _nullable_int,
    _finite_float as _finite_float,
    _nullable_float as _nullable_float,
    _sha_or_none as _sha_or_none,
    _int_array as _int_array,
    _string_array as _string_array,
    _pair as _pair,
    _validate_generation as _validate_generation,
    _validate_history as _validate_history,
    _durable_fact_digest as _durable_fact_digest,
    _fact_hex_float as _fact_hex_float,
    _validate_source_facts as _validate_source_facts,
    _validate_route_plan as _validate_route_plan,
    _validate_strict_failure as _validate_strict_failure,
    _validate_physical_calls as _validate_physical_calls,
    _crosslink_source_facts as _crosslink_source_facts,
    _validate_legacy as _validate_legacy,
    _validate_limits as _validate_limits,
    _project_route_mismatch as _project_route_mismatch,
    _validate_lane as _validate_lane,
    _validate_denied_charge as _validate_denied_charge,
    _work_reason_projection as _work_reason_projection,
)
from voxweave.align_failures import (
    SEED_REASON_ORDER,
    CanonicalFailure,
)
from voxweave.align_inputs import EvidenceStatus, ProfileStatus, V2PolicyStatus
from voxweave.align_snapshot import (
    StrictInputStatus,
    freeze_json,
    frozen_json_digest,
)
from voxweave.candidate_encoder import _verified_hash_binding
from voxweave.engine_registry import EngineFamily
from voxweave.paths import find_sibling_media, find_subtitle_media, swap_ext
from voxweave.voicebase import media_fingerprint


@dataclass(frozen=True)
class SelectedOutputs:
    engine_family: EngineFamily
    vtt_present: Literal[True]
    vtt_sha256: str
    json_present: Literal[True]
    json_sha256: str


@dataclass(frozen=True, init=False)
class FinalAlignEvidence:
    core: EvidenceCore
    selected_outputs: SelectedOutputs
    durable_authority: Literal[True]
    _projection: Mapping[str, Any] = field(repr=False, compare=True)

    def __init__(self) -> None:
        raise TypeError("FinalAlignEvidence is binder-only")


@dataclass(frozen=True)
class AlignEvidenceVerification:
    integrity: bool
    w1_usable: bool
    detail_code: str | None


@dataclass(frozen=True)
class _EvidenceRecord:
    evidence: FinalAlignEvidence
    context: IssuedAlignContext
    snapshot: FinalAlignEvidence


_EVIDENCE: dict[int, _EvidenceRecord] = {}
_LOCK = threading.RLock()


class EvidenceBindingError(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(
            f"{failure.kind}/{failure.phase}/{failure.detail_code}: align stopped"
            " because its alignment evidence failed an internal consistency check;"
            " no subtitle file was written. This is a VoxWeave bug, not a problem"
            " with your files; please report it with this message"
        )
        self.failure = failure


def _binding_failure(detail: str = "evidence-binding") -> EvidenceBindingError:
    return EvidenceBindingError(
        CanonicalFailure("final-evidence-invalid", "evidence-bind", detail)
    )


def _release_evidence(context: IssuedAlignContext) -> None:
    """Forget the evidence bound for ``context``.

    See :func:`voxweave.align_orchestration.release_align_selection`.
    """
    with _LOCK:
        for key in [key for key, row in _EVIDENCE.items() if row.context is context]:
            del _EVIDENCE[key]


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            member.name: _json_value(getattr(value, member.name))
            for member in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_json_value(member) for member in value]
    if isinstance(value, list):
        return [_json_value(member) for member in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(member) for key, member in value.items()}
    return value


def _core_value(
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
    evidence_core: EvidenceCore,
    *,
    strict_input_status: StrictInputStatus,
    v2_policy_status: V2PolicyStatus,
    profile_status: ProfileStatus,
    evidence_status: EvidenceStatus,
) -> dict[str, Any]:
    projection = evidence_core_value(evidence_core)
    strict_value = {
        "kind": strict_input_status.kind,
        "detail_code": strict_input_status.detail_code,
    }
    policy_value = {
        "kind": v2_policy_status.kind,
        "detail_code": v2_policy_status.detail_code,
    }
    profile_value = {
        "kind": profile_status.kind,
        "source": profile_status.source,
        "detail_code": profile_status.detail_code,
    }
    evidence_value = {
        "kind": evidence_status.kind,
        "detail_code": evidence_status.detail_code,
    }
    if (
        projection.get("schema_version") != 8
        or projection.get("kind") != "fresh-alignment"
        or projection.get("context_content_digest") != context.context_content_digest
        or projection.get("receipt_digest") != acquisition.receipt_digest
        or projection.get("language") != context.effective_iso
        or projection.get("route") != context.route_kind
        or projection.get("raw_unit_count") != evidence_core.raw_unit_count
        or projection.get("strict_input_status") != strict_value
        or projection.get("v2_policy_status") != policy_value
        or projection.get("profile_status") != profile_value
        or projection.get("evidence_status") != evidence_value
        or projection.get("seed_status")
        != {
            "kind": evidence_core.seed_status,
            "reasons": list(evidence_core.seed_reasons),
        }
        or evidence_core.receipt_digest != acquisition.receipt_digest
        or len(evidence_core.physical_calls) != len(acquisition.physical_calls)
        or evidence_core.authority_status != acquisition.distribution.status
        or evidence_core.authority_work != acquisition.distribution.work
        or evidence_core.seed_status != acquisition.seed_status
        or evidence_core.seed_reasons != acquisition.seed_reasons
    ):
        raise _binding_failure("independent-projection")
    return projection


def bind_align_evidence(
    context: IssuedAlignContext,
    evidence_core: EvidenceCore,
    *,
    acquisition: IssuedFreshAlignment,
    strict_input_status: StrictInputStatus,
    v2_policy_status: V2PolicyStatus,
    profile_status: ProfileStatus,
    evidence_status: EvidenceStatus,
    engine_family: EngineFamily,
    vtt_sha256: str,
    main_json_sha256: str,
) -> FinalAlignEvidence:
    """Bind a complete closed core only after independent primary verification."""
    if (
        not isinstance(context, IssuedAlignContext)
        or not isinstance(acquisition, IssuedFreshAlignment)
        or not isinstance(evidence_core, EvidenceCore)
        or not isinstance(strict_input_status, StrictInputStatus)
        or not isinstance(v2_policy_status, V2PolicyStatus)
        or not isinstance(profile_status, ProfileStatus)
        or not isinstance(evidence_status, EvidenceStatus)
        or evidence_core.context_content_digest != context.context_content_digest
        or acquisition.context_content_digest != context.context_content_digest
        or engine_family != context.engine_family
        or not _is_sha256(vtt_sha256)
        or not _is_sha256(main_json_sha256)
    ):
        raise _binding_failure()
    if not _verified_hash_binding(
        context,
        evidence_core,
        engine_family,
        vtt_sha256,
        main_json_sha256,
    ):
        raise _binding_failure("selected-hash-link")
    consume_context_role(context, "evidence-bind", consumer="bind_align_evidence")
    try:
        projection = _core_value(
            context,
            acquisition,
            evidence_core,
            strict_input_status=strict_input_status,
            v2_policy_status=v2_policy_status,
            profile_status=profile_status,
            evidence_status=evidence_status,
        )
        selected = SelectedOutputs(
            engine_family, True, vtt_sha256, True, main_json_sha256
        )
        _validate_evidence_value(
            {**projection, "selected_outputs": _json_value(selected)}
        )
    except EvidenceBindingError:
        raise
    except Exception as exc:
        raise _binding_failure("closed-schema") from exc
    evidence = object.__new__(FinalAlignEvidence)
    object.__setattr__(evidence, "core", evidence_core)
    object.__setattr__(evidence, "selected_outputs", selected)
    object.__setattr__(evidence, "durable_authority", True)
    object.__setattr__(evidence, "_projection", copy.deepcopy(projection))
    with _LOCK:
        _EVIDENCE[id(evidence)] = _EvidenceRecord(
            evidence, context, copy.deepcopy(evidence)
        )
    return evidence


def _final_value(evidence: FinalAlignEvidence) -> dict[str, Any]:
    return {
        **copy.deepcopy(dict(evidence._projection)),
        "selected_outputs": _json_value(evidence.selected_outputs),
    }


def encode_align_evidence(evidence: FinalAlignEvidence) -> bytes:
    """Encode one genuine bound record in canonical UTF-8/LF form."""
    with _LOCK:
        record = _EVIDENCE.get(id(evidence))
    if (
        record is None
        or record.evidence is not evidence
        or evidence != record.snapshot
        or evidence.durable_authority is not True
    ):
        raise _binding_failure()
    value = _final_value(evidence)
    try:
        _validate_evidence_value(value)
        return (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except EvidenceBindingError:
        raise
    except Exception as exc:
        raise _binding_failure("closed-schema") from exc


def _validate_work(
    value: object,
    route_plan: Mapping[str, Any],
    physical_calls: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
) -> dict[str, Any]:
    work = _closed_mapping(value, _WORK_KEYS, "authority work")
    if work["status"] not in _WORK_STATUSES:
        _invalid("authority work status is invalid")
    if work["route_status"] not in ("valid", "invalid"):
        _invalid("authority route status is invalid")
    mismatch = work["route_mismatch"]
    if mismatch is not None:
        mismatch = _closed_mapping(mismatch, _ROUTE_MISMATCH_KEYS, "route mismatch")
        if mismatch["kind"] not in (
            "gap",
            "overlap",
            "unexpected-index",
            "reorder",
            "owner-crosslink",
        ):
            _invalid("route mismatch kind is invalid")
        for name in _ROUTE_MISMATCH_KEYS[1:]:
            _nullable_int(mismatch[name], f"route mismatch {name}")
    claims = _list(work["route_claims"], "route claims")
    for member in claims:
        claim = _closed_mapping(member, _ROUTE_CLAIM_KEYS, "route claim")
        if claim["owner_kind"] not in ("call", "skip"):
            _invalid("route claim owner kind is invalid")
        for name in _ROUTE_CLAIM_KEYS[1:]:
            _exact_int(claim[name], f"route claim {name}")
    calls = _list(work["calls"], "authority work calls")
    skips = _list(work["skipped_blocks"], "authority skipped blocks")
    if len(calls) != len(physical_calls):
        _invalid("work/physical call cardinality mismatch")
    raw_cursor = 0
    lane_totals = {name: 0 for name in _COUNTER_KEYS}
    for call_index, (member, physical) in enumerate(
        zip(calls, physical_calls, strict=True)
    ):
        row = _closed_mapping(member, _WORK_CALL_KEYS, "authority work call")
        if row["call_index"] != call_index:
            _invalid("authority work call indexes are not contiguous")
        positions = _int_array(
            row["route_claim_positions"], "call claim positions", unique=True
        )
        for position in positions:
            if (
                position >= len(claims)
                or claims[position]["owner_kind"] != "call"
                or claims[position]["owner_index"] != call_index
            ):
                _invalid("call claim positions do not cross-link")
        sources = _int_array(row["source_block_indices"], "work call sources")
        if sources != physical["source_block_indices"]:
            _invalid("work call sources disagree with physical receipt")
        raw_range = _pair(row["raw_node_range"], "raw node range")
        if raw_range != (
            raw_cursor,
            raw_cursor + len(physical["raw_unit_ids"]),
        ):
            _invalid("raw node ranges do not tile global raw units")
        raw_cursor = raw_range[1]
        if row["block_count"] != len(sources) or row["raw_node_count"] != len(
            physical["raw_unit_ids"]
        ):
            _invalid("work call declared counts disagree")
        typed = row["typed_unit_count"]
        surface = row["surface_chars"]
        strict = row["strict_preflight_status"]
        failure = _validate_strict_failure(row["strict_failure"], "work strict failure")
        if failure != physical["strict_failure"]:
            _invalid("physical/work strict locators disagree")
        if strict == "capture-invalid":
            if (
                typed is not None
                or surface is not None
                or failure is None
                or failure["stage"] != "strict-capture"
            ):
                _invalid("capture-invalid work row has invalid nullability")
        elif strict in ("valid", "transform-invalid"):
            _exact_int(typed, "typed unit count")
            _exact_int(surface, "surface chars")
            if typed != len(physical["raw_unit_ids"]):
                _invalid("typed unit count mismatch")
            if (strict == "valid") != (failure is None):
                _invalid("work strict preflight/failure mismatch")
        else:
            _invalid("strict preflight status is invalid")
        allocator = _validate_lane(row["allocator"], "allocator lane")
        verifier = (
            None
            if row["verifier"] is None
            else _validate_lane(row["verifier"], "verifier lane")
        )
        for name in _COUNTER_KEYS:
            lane_totals[name] += allocator[name]
            if verifier is not None:
                lane_totals[name] += verifier[name]
    for skip_index, member in enumerate(skips):
        row = _closed_mapping(member, _SKIP_KEYS, "authority skipped block")
        positions = _int_array(
            row["route_claim_positions"], "skip claim positions", unique=True
        )
        for position in positions:
            if (
                position >= len(claims)
                or claims[position]["owner_kind"] != "skip"
                or claims[position]["owner_index"] != skip_index
            ):
                _invalid("skip claim positions do not cross-link")
        _exact_int(row["delivery_index"], "skip delivery index")
        _exact_int(row["source_index"], "skip source index")
        if (
            row["route_skip_reason"] not in ("missing-crop", "empty-alignment-text")
            or row["source_text_kind"] not in ("empty", "whitespace", "nonempty")
            or row["detail_code"] != "partial-empty-ownership"
            or row["work_status"] != "not-run"
        ):
            _invalid("skipped block shape is invalid")
        if any(_exact_int(row[name], f"skip {name}") for name in _COUNTER_KEYS):
            _invalid("skipped block carries work")
    expected_mismatch = _project_route_mismatch(
        claims, route_plan["entries"], calls, skips
    )
    if mismatch != expected_mismatch or (work["route_status"] == "valid") != (
        mismatch is None
    ):
        _invalid("route mismatch projection disagrees with claims")
    declared = {
        "declared_delivery_block_count": len(route_plan["entries"]),
        "declared_call_count": len(calls),
        "declared_skip_count": len(skips),
        "declared_raw_node_count": raw_cursor,
    }
    for name, expected in declared.items():
        if work[name] != expected:
            _invalid(f"authority work {name} mismatch")
    _exact_int(work["charged_call_count"], "charged call count")
    if work["charged_call_count"] > len(calls):
        _invalid("charged call count exceeds declared calls")
    if (
        work["limit_profile_kind"] != history["authority_limit_profile_kind"]
        or work["limit_profile_digest"] != history["authority_limit_profile_digest"]
    ):
        _invalid("work/input authority profile mismatch")
    job_limits = _closed_mapping(work["limits"], _JOB_LIMIT_KEYS, "job limits")
    _validate_limits(
        work["limit_profile_kind"],
        work["limit_profile_digest"],
        job_limits,
        calls,
    )
    totals = _closed_mapping(work["totals"], _COUNTER_KEYS, "work totals")
    for name in _COUNTER_KEYS:
        if totals[name] != lane_totals[name]:
            _invalid("work totals do not equal lane sums")
    terminal = _nullable_int(work["terminal_call_index"], "terminal call index")
    if terminal is not None and terminal >= len(calls):
        _invalid("terminal call index is outside calls")
    denied = _validate_denied_charge(work["denied_charge"])
    if (work["status"] == "budget-exhausted") != (denied is not None):
        _invalid("budget status/denied charge mismatch")
    if work["status"].startswith("not-run-"):
        if (
            work["charged_call_count"] != 0
            or any(totals.values())
            or denied is not None
        ):
            _invalid("not-run authority work carries charges")
        if any(row["verifier"] is not None for row in calls):
            _invalid("not-run authority work carries a verifier")
    return work


def _validate_authority(
    value: object,
    route_plan: Mapping[str, Any],
    physical_calls: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    raw_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[int, list[str]] | None]:
    authority = _closed_mapping(
        value, _AUTHORITY_DISTRIBUTION_KEYS, "authority_distribution"
    )
    if authority["status"] not in ("valid", "invalid"):
        _invalid("authority distribution status is invalid")
    work = _validate_work(authority["work"], route_plan, physical_calls, history)
    reasons = _string_array(authority["reasons"], "authority reasons", unique=True)
    if reasons != _work_reason_projection(work):
        _invalid("authority reason membership/order mismatch")
    owners_by_source: dict[int, list[str]] | None
    if authority["status"] == "valid":
        sources = _int_array(
            authority["owner_source_indices"],
            "authority owner sources",
            unique=True,
        )
        counts = _int_array(authority["expected_counts"], "authority expected counts")
        owner_rows = _list(authority["owner_unit_ids"], "authority owners")
        if not (len(sources) == len(counts) == len(owner_rows)):
            _invalid("authority owner vectors disagree")
        owners_by_source = {}
        flattened: list[str] = []
        for source, count, owner_value in zip(sources, counts, owner_rows, strict=True):
            owner = _string_array(owner_value, "authority owner IDs", unique=True)
            if not owner or count != len(owner):
                _invalid("authority owner is empty or count disagrees")
            owners_by_source[source] = owner
            flattened.extend(owner)
        if flattened != list(raw_ids) or len(flattened) != len(set(flattened)):
            _invalid("authority owners do not tile raw units in order")
        if (
            authority["consumed_count"] != len(raw_ids)
            or authority["leftover_unit_ids"] != []
            or reasons
            or work["status"] != "complete"
        ):
            _invalid("valid authority terminal fields disagree")
    else:
        owners_by_source = None
        if (
            authority["owner_source_indices"] is not None
            or authority["expected_counts"] is not None
            or authority["owner_unit_ids"] is not None
        ):
            _invalid("invalid authority carries owner/count arrays")
        if (
            authority["consumed_count"] != 0
            or authority["leftover_unit_ids"] != list(raw_ids)
            or not reasons
        ):
            _invalid("invalid authority terminal fields disagree")
    digest_input = {
        key: authority[key] for key in _AUTHORITY_DISTRIBUTION_KEYS if key != "digest"
    }
    if not _is_sha256(authority["digest"]) or authority["digest"] != _digest_projection(
        "p6-authority-distribution-v1", digest_input
    ):
        _invalid("authority distribution digest mismatch")
    return authority, owners_by_source


def _validate_blocks(
    value: object,
    route_by_source: Mapping[int, Mapping[str, Any]],
    legacy_by_source: Mapping[int, list[str]],
    authority_by_source: Mapping[int, list[str]] | None,
    physical_calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocks = _list(value, "evidence blocks")
    if len(blocks) != len(route_by_source):
        _invalid("block/route cardinality mismatch")
    physical_by_index = {row["call_index"]: row for row in physical_calls}
    observed_sources: list[int] = []
    for member in blocks:
        block = _closed_mapping(member, _BLOCK_KEYS, "evidence block")
        source = _exact_int(block["source_index"], "block source index")
        observed_sources.append(source)
        if block["legacy_unit_ids"] != legacy_by_source.get(source, []):
            _invalid("block legacy IDs disagree with legacy distribution")
        if authority_by_source is None:
            if (
                block["authority_unit_ids"] is not None
                or block["word_data"] is not None
                or block["speech_start"] is not None
                or block["speech_end"] is not None
            ):
                _invalid("invalid authority block carries authority data")
            continue
        owner = authority_by_source.get(source)
        if owner is None or block["authority_unit_ids"] != owner:
            _invalid("block authority IDs disagree with distribution")
        words = _list(block["word_data"], "block word_data")
        if len(words) != len(owner):
            _invalid("block word_data cardinality mismatch")
        for unit_id, member_word in zip(owner, words, strict=True):
            word = _closed_mapping(member_word, _WORD_KEYS, "evidence word")
            if word["unit_id"] != unit_id or type(word["text"]) is not str:
                _invalid("evidence word identity/text mismatch")
            call_index = _exact_int(word["call_index"], "word call index")
            call_unit_index = _exact_int(
                word["call_unit_index"], "word call unit index"
            )
            call = physical_by_index.get(call_index)
            if (
                call is None
                or call_unit_index >= len(call["raw_unit_ids"])
                or call["raw_unit_ids"][call_unit_index] != unit_id
            ):
                _invalid("evidence word does not cross-link raw unit")
            origin = _finite_float(
                word["physical_origin_seconds"], "word physical origin"
            )
            if not _same_float(origin, call["physical_origin_seconds"]):
                _invalid("evidence word physical origin mismatch")
            relative_start = _nullable_float(
                word["relative_start"], "word relative start"
            )
            relative_end = _nullable_float(word["relative_end"], "word relative end")
            start = _nullable_float(word["start"], "word absolute start")
            end = _nullable_float(word["end"], "word absolute end")
            if (
                relative_start is None
                or relative_end is None
                or start is None
                or end is None
            ):
                _invalid("admitted authority word lacks complete bounds")
            if (
                relative_start > relative_end
                or start > end
                or not _same_float(start, relative_start + origin)
                or not _same_float(end, relative_end + origin)
            ):
                _invalid("evidence word time transform mismatch")
            if word["provenance"] not in ("aligner", "align-interpolated"):
                _invalid("evidence word provenance is invalid")
            _nullable_float(word["original_relative_start"], "original relative start")
            _nullable_float(word["original_relative_end"], "original relative end")
        first, last = words[0], words[-1]
        expected_start = first["start"] if first["provenance"] == "aligner" else None
        expected_end = last["end"] if last["provenance"] == "aligner" else None
        if not _same_optional_float(
            block["speech_start"], expected_start
        ) or not _same_optional_float(block["speech_end"], expected_end):
            _invalid("block endpoint anchor projection mismatch")
    if observed_sources != [
        entry["source_index"] for entry in route_by_source.values()
    ]:
        _invalid("blocks are not in sealed delivery order")
    return blocks


def _validate_statuses(
    value: Mapping[str, Any], authority_valid: bool, transform_valid: bool
) -> None:
    strict = _closed_mapping(
        value["strict_input_status"], _STRICT_STATUS_KEYS, "strict input status"
    )
    if strict["kind"] == "valid":
        if strict["detail_code"] is not None:
            _invalid("valid strict input carries a detail")
    elif strict["kind"] == "invalid":
        if strict["detail_code"] not in (
            "sibling-json-duplicate-key",
            "sibling-json-nonfinite",
        ):
            _invalid("strict input detail is invalid")
    else:
        _invalid("strict input status kind is invalid")
    seed = _closed_mapping(value["seed_status"], _SEED_STATUS_KEYS, "seed status")
    reasons = _string_array(seed["reasons"], "seed reasons", unique=True)
    if reasons != [reason for reason in SEED_REASON_ORDER if reason in reasons]:
        _invalid("seed reasons are not registry ordered")
    if (seed["kind"] == "valid") != (not reasons):
        _invalid("seed status/reasons disagree")
    policy = _closed_mapping(
        value["v2_policy_status"], _STRICT_STATUS_KEYS, "v2 policy status"
    )
    if policy["kind"] == "valid":
        if policy["detail_code"] is not None:
            _invalid("valid policy carries a detail")
    elif policy["kind"] == "invalid":
        if policy["detail_code"] not in ("nonfinite-policy", "negative-policy"):
            _invalid("policy detail is invalid")
    else:
        _invalid("policy status is invalid")
    profile = _closed_mapping(
        value["profile_status"], _PROFILE_STATUS_KEYS, "profile status"
    )
    if profile["source"] not in _PROFILE_SOURCES:
        _invalid("profile source is invalid")
    if profile["kind"] == "valid":
        if profile["detail_code"] is not None:
            _invalid("valid profile carries a detail")
    elif profile["kind"] == "invalid":
        if profile["detail_code"] not in (
            "profile-shape",
            "profile-language",
            "profile-domain",
            "resolved-default-domain",
        ):
            _invalid("profile detail is invalid")
    else:
        _invalid("profile status is invalid")
    evidence = _closed_mapping(
        value["evidence_status"], _STRICT_STATUS_KEYS, "evidence status"
    )
    if evidence["kind"] == "valid":
        if evidence["detail_code"] is not None:
            _invalid("valid evidence status carries a detail")
    elif evidence["kind"] == "invalid":
        if evidence["detail_code"] != "evidence-domain":
            _invalid("evidence status detail is invalid")
    else:
        _invalid("evidence status is invalid")
    admission = all(
        (
            strict["kind"] == "valid",
            transform_valid,
            authority_valid,
            seed["kind"] == "valid",
            policy["kind"] == "valid",
            profile["kind"] == "valid",
            evidence["kind"] == "valid",
        )
    )
    if value["v2_admission_status"] != ("valid" if admission else "invalid"):
        _invalid("v2 admission status does not equal its conjunctive inputs")


def _lane_receipt_value(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": value["status"],
        "counters": {name: value[name] for name in _COUNTER_KEYS},
        "terminal_detail_code": value["terminal_detail_code"],
    }


def _work_receipt_value(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": value["status"],
        "route_status": value["route_status"],
        "route_mismatch": value["route_mismatch"],
        "route_claims": value["route_claims"],
        "declared_delivery_block_count": value["declared_delivery_block_count"],
        "declared_call_count": value["declared_call_count"],
        "declared_skip_count": value["declared_skip_count"],
        "declared_raw_node_count": value["declared_raw_node_count"],
        "charged_call_count": value["charged_call_count"],
        "limit_profile_kind": value["limit_profile_kind"],
        "limit_profile_digest": value["limit_profile_digest"],
        "limits": value["limits"],
        "totals": value["totals"],
        "terminal_call_index": value["terminal_call_index"],
        "denied_charge": value["denied_charge"],
        "calls": [
            {
                "call_index": row["call_index"],
                "route_claim_positions": row["route_claim_positions"],
                "source_block_indices": row["source_block_indices"],
                "raw_node_range": row["raw_node_range"],
                "block_count": row["block_count"],
                "raw_node_count": row["raw_node_count"],
                "typed_unit_count": row["typed_unit_count"],
                "surface_chars": row["surface_chars"],
                "strict_preflight_status": row["strict_preflight_status"],
                "strict_failure": row["strict_failure"],
                "limits": row["limits"],
                "allocator": _lane_receipt_value(row["allocator"]),
                "verifier": (
                    None
                    if row["verifier"] is None
                    else _lane_receipt_value(row["verifier"])
                ),
            }
            for row in value["calls"]
        ],
        "skipped_blocks": [
            {
                "route_claim_positions": row["route_claim_positions"],
                "delivery_index": row["delivery_index"],
                "source_index": row["source_index"],
                "route_skip_reason": row["route_skip_reason"],
                "source_text_kind": row["source_text_kind"],
                "detail_code": row["detail_code"],
                "work_status": row["work_status"],
                "counters": {name: row[name] for name in _COUNTER_KEYS},
            }
            for row in value["skipped_blocks"]
        ],
    }


def _receipt_digest(value: Mapping[str, Any]) -> str:
    physical = [
        {
            "call_index": row["call_index"],
            "source_block_indices": row["source_block_indices"],
            "audio_sample_start": row["sample_start"],
            "audio_sample_end": row["sample_end"],
            "sample_rate": row["sample_rate"],
            "physical_origin_seconds": row["physical_origin_seconds"],
            "legacy_origin_seconds": row["legacy_origin_seconds"],
            "legacy_origin_kind": row["legacy_origin_kind"],
            "authority_origin_seconds": row["authority_origin_seconds"],
            "backend_model_config_digest": row["backend_model_config_sha256"],
            "route_input_digest": row["route_input_sha256"],
            "strict_unit_status": row["strict_unit_status"],
            "strict_failure": row["strict_failure"],
            "raw_units_digest": row["raw_units_sha256"],
            "normalized_relative_digest": row["relative_units_sha256"],
            "legacy_slice_digest": row["legacy_slice_sha256"],
            "legacy_absolute_digest": row["legacy_absolute_sha256"],
            "authority_transform_status": row["authority_transform_status"],
            "authority_absolute_digest": row["authority_absolute_sha256"],
            "raw_unit_ids": row["raw_unit_ids"],
        }
        for row in value["physical_calls"]
    ]
    legacy = [
        {key: row[key] for key in _LEGACY_CALL_KEYS if key != "call_index"}
        for row in value["legacy_distribution"]["calls"]
    ]
    authority = value["authority_distribution"]
    owner_sources = authority["owner_source_indices"]
    if owner_sources is None:
        owner_sources = [
            entry["source_index"] for entry in value["route_plan"]["entries"]
        ]
    distribution = {
        "status": authority["status"],
        "owner_source_indices": owner_sources,
        "owners": authority["owner_unit_ids"],
        "expected_counts": authority["expected_counts"],
        "reasons": authority["reasons"],
        "consumed_count": authority["consumed_count"],
        "leftovers": authority["leftover_unit_ids"],
        "work": _work_receipt_value(authority["work"]),
    }
    projection = {
        "context_content_digest": value["context_content_digest"],
        "physical_calls": physical,
        "legacy_distribution": legacy,
        "authority_distribution": distribution,
        "seed_status": value["seed_status"]["kind"],
        "seed_reasons": value["seed_status"]["reasons"],
    }
    return frozen_json_digest(freeze_json(projection))


def _validate_evidence_value(value: object) -> dict[str, Any]:
    root = _closed_mapping(value, _TOP_LEVEL_KEYS, "align evidence")
    if root["schema_version"] != 8 or root["kind"] != "fresh-alignment":
        _invalid("align evidence discriminator is invalid")
    if not _is_sha256(root["context_content_digest"]) or not _is_sha256(
        root["receipt_digest"]
    ):
        _invalid("align evidence root digests are invalid")
    if type(root["language"]) is not str or not root["language"]:
        _invalid("align evidence language is invalid")
    if root["route"] not in ("ctc-full", "mms-full", "qwen-crop"):
        _invalid("align evidence route is invalid")
    _source_facts, model_digest, route_digest = _validate_source_facts(
        root["source_facts"],
        context_digest=root["context_content_digest"],
        language=root["language"],
        route=root["route"],
    )
    history = _validate_history(root["input_history"])
    route_plan, route_by_source = _validate_route_plan(
        root["route_plan"], root["route"]
    )
    physical_calls = _validate_physical_calls(
        root["physical_calls"],
        root["route"],
        route_by_source,
        model_digest,
        route_digest,
    )
    _crosslink_source_facts(_source_facts, route_plan, physical_calls)
    raw_ids = [unit_id for call in physical_calls for unit_id in call["raw_unit_ids"]]
    if root["raw_unit_count"] != len(raw_ids):
        _invalid("raw unit count mismatch")
    _legacy, legacy_by_source = _validate_legacy(
        root["legacy_distribution"], physical_calls, root["route"]
    )
    authority, authority_by_source = _validate_authority(
        root["authority_distribution"],
        route_plan,
        physical_calls,
        history,
        raw_ids,
    )
    _validate_blocks(
        root["blocks"],
        route_by_source,
        legacy_by_source,
        authority_by_source,
        physical_calls,
    )
    selected = _closed_mapping(
        root["selected_outputs"], _SELECTED_OUTPUT_KEYS, "selected outputs"
    )
    if (
        selected["engine_family"] not in ("legacy-v1", "boundary-v2")
        or selected["vtt_present"] is not True
        or selected["json_present"] is not True
        or not _is_sha256(selected["vtt_sha256"])
        or not _is_sha256(selected["json_sha256"])
    ):
        _invalid("selected output binding is invalid")
    if history["registry_family"] != selected["engine_family"]:
        _invalid("context registry family and selected output disagree")
    transform_valid = all(
        call["authority_transform_status"] == "valid" for call in physical_calls
    )
    _validate_statuses(root, authority["status"] == "valid", transform_valid)
    if root["receipt_digest"] != _receipt_digest(root):
        _invalid("receipt digest does not cover the durable primitive projection")
    return root


def _normalized_basename(path: Path) -> str:
    return unicodedata.normalize("NFC", Path(path).name)


def _media_integrity(
    value: Mapping[str, Any],
    vtt_path: Path,
    explicit_media_path: Path | None,
    corpus_root: Path | None,
) -> bool:
    history = value["input_history"]
    logical_id = history["media_logical_id"]
    expected = history["media_fingerprint"]
    media: Path
    if logical_id.startswith("explicit:"):
        if explicit_media_path is None or corpus_root is not None:
            return False
        media = Path(explicit_media_path)
        if _normalized_basename(media) != logical_id.removeprefix("explicit:"):
            return False
    elif logical_id.startswith("corpus:"):
        if corpus_root is None or explicit_media_path is not None:
            return False
        locator = logical_id.removeprefix("corpus:")
        pure = PurePosixPath(locator)
        if (
            pure.is_absolute()
            or not locator
            or "\\" in locator
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            return False
        root = Path(corpus_root).resolve()
        media = root.joinpath(*pure.parts).resolve()
        try:
            media.relative_to(root)
        except ValueError:
            return False
    elif logical_id.startswith("sibling:"):
        if explicit_media_path is not None or corpus_root is not None:
            return False
        suffix = logical_id.removeprefix("sibling:")
        if (
            not suffix.startswith(".")
            or suffix != suffix.lower()
            or "/" in suffix
            or "\\" in suffix
        ):
            return False
        try:
            resolved = find_sibling_media(vtt_path)
        except OSError:
            return False
        if resolved is None or resolved.suffix.lower() != suffix:
            return False
        media = resolved
    else:
        return False
    try:
        return media.is_file() and media_fingerprint(media) == expected
    except OSError:
        return False


def _w1_usable_audit(root: Mapping[str, Any]) -> bool:
    """Return the unsigned W1 usability conjunction: production limit profile,
    valid V2 admission, full (non-default) source facts, and owner ranges that
    cover every raw unit exactly once."""
    try:
        history = root["input_history"]
        source_facts = root["source_facts"]
        model_facts = source_facts["backend_model_config"]
        route_facts = source_facts["route_input"]
        authority = root["authority_distribution"]
        work = authority["work"]
        owner_ranges = authority["owner_unit_ids"]
        raw_unit_count = root["raw_unit_count"]
    except (KeyError, TypeError):
        return False
    if (
        not isinstance(history, Mapping)
        or not isinstance(source_facts, Mapping)
        or not isinstance(model_facts, Mapping)
        or not isinstance(route_facts, Mapping)
        or tuple(model_facts) != _MODEL_FACT_KEYS
        or tuple(route_facts) != _ROUTE_FACT_KEYS
        or not isinstance(authority, Mapping)
        or not isinstance(work, Mapping)
        or history.get("authority_limit_profile_kind") != "production"
        or work.get("limit_profile_kind") != "production"
        or root.get("v2_admission_status") != "valid"
        or type(raw_unit_count) is not int
        or raw_unit_count < 0
        or not isinstance(owner_ranges, (list, tuple))
        or not owner_ranges
    ):
        return False
    flattened: list[str] = []
    for owner_range in owner_ranges:
        if not isinstance(owner_range, (list, tuple)) or not owner_range:
            return False
        if any(type(unit_id) is not str for unit_id in owner_range):
            return False
        flattened.extend(owner_range)
    return len(flattened) == raw_unit_count and len(set(flattened)) == raw_unit_count


def verify_align_evidence(
    vtt_path: Path,
    *,
    explicit_media_path: Path | None = None,
    corpus_root: Path | None = None,
) -> AlignEvidenceVerification:
    """Verify canonical sidecar bytes and live selected-primary/media links."""
    target = Path(vtt_path)
    legacy_evidence = swap_ext(target, ".align-evidence.json")
    artifact_media = (
        Path(explicit_media_path) if explicit_media_path is not None else None
    )
    if artifact_media is None:
        try:
            artifact_media = find_subtitle_media(target)
        except OSError:
            artifact_media = None
    if artifacts.path_present(legacy_evidence):
        evidence_path = legacy_evidence
    else:
        cached = artifacts.inspect_paths(artifact_media or target)
        evidence_path = (
            legacy_evidence if cached is None else cached.align_evidence(target)
        )
    json_path = swap_ext(target, ".json")
    try:
        encoded = evidence_path.read_bytes()
    except OSError:
        return AlignEvidenceVerification(False, False, "evidence-read")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite,
        )
        canonical = (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        if encoded != canonical:
            _invalid("evidence bytes are not canonical")
        root = _validate_evidence_value(value)
        selected = root["selected_outputs"]
    except (KeyError, TypeError, UnicodeError, ValueError):
        return AlignEvidenceVerification(False, False, "evidence-schema")
    try:
        vtt_bytes = target.read_bytes()
        json_bytes = json_path.read_bytes()
    except OSError:
        return AlignEvidenceVerification(False, False, "selected-output-read")
    if (
        hashlib.sha256(vtt_bytes).hexdigest() != selected["vtt_sha256"]
        or hashlib.sha256(json_bytes).hexdigest() != selected["json_sha256"]
    ):
        return AlignEvidenceVerification(False, False, "selected-output-hash")
    if not _media_integrity(root, target, explicit_media_path, corpus_root):
        return AlignEvidenceVerification(False, False, "media-identity")
    # This audit result never remints or replaces a fresh W1 capability.
    return AlignEvidenceVerification(True, _w1_usable_audit(root), None)


__all__ = [
    "AlignEvidenceVerification",
    "EvidenceBindingError",
    "FinalAlignEvidence",
    "SelectedOutputs",
    "bind_align_evidence",
    "encode_align_evidence",
    "verify_align_evidence",
]
