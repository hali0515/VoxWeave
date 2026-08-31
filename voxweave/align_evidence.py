"""Closed RAT-2 evidence binding and independent durable verification.

The producer receives only already-issued context/acquisition values and the
independently projected EvidenceCore. The path verifier deliberately does not
consult any in-memory issuer registry.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NoReturn

from voxweave import artifacts
from voxweave.align_acquisition import IssuedFreshAlignment
from voxweave.align_context import (
    IssuedAlignContext,
    consume_context_role,
)
from voxweave.align_distribution import (
    AUTH_ALLOC_EDGE_LIMIT,
    AUTH_ALLOC_INTERVAL_LIMIT,
    AUTH_ALLOC_JOB_CALL_LIMIT,
    AUTH_ALLOC_JOB_EDGE_LIMIT,
    AUTH_ALLOC_JOB_INTERVAL_LIMIT,
    AUTH_ALLOC_JOB_NORMALIZE_CHAR_LIMIT,
    AUTH_ALLOC_JOB_STATE_LIMIT,
    AUTH_ALLOC_NORMALIZE_CHAR_LIMIT,
    AUTH_ALLOC_STATE_LIMIT,
)
from voxweave.align_evidence_core import (
    EvidenceCore,
    evidence_core_value,
)
from voxweave.align_failures import (
    AUTHORITY_REASON_ORDER,
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
from voxweave.voicebase import media_fingerprint


_TOP_LEVEL_KEYS = (
    "schema_version",
    "kind",
    "context_content_digest",
    "receipt_digest",
    "language",
    "route",
    "source_facts",
    "input_history",
    "route_plan",
    "physical_calls",
    "legacy_distribution",
    "authority_distribution",
    "blocks",
    "raw_unit_count",
    "strict_input_status",
    "seed_status",
    "v2_policy_status",
    "profile_status",
    "evidence_status",
    "v2_admission_status",
    "selected_outputs",
)
_SOURCE_FACT_KEYS = ("backend_model_config", "route_input")
_MODEL_FACT_KEYS = ("route", "language", "backend", "model", "sample_rate")
_DEFAULT_MODEL_FACT_KEYS = ("kind", "route", "language")
_ROUTE_FACT_KEYS = ("route", "language", "blocks", "crops")
_DEFAULT_ROUTE_FACT_KEYS = ("kind", "context_content_digest", "route")
_ROUTE_FACT_BLOCK_KEYS = (
    "source_index",
    "alignment_text",
    "start",
    "end",
)
_INPUT_HISTORY_KEYS = (
    "vtt_present",
    "vtt_size",
    "vtt_sha256",
    "sibling_json_present",
    "sibling_json_size",
    "sibling_json_sha256",
    "block_content_sha256",
    "registry_family",
    "media_fingerprint",
    "media_logical_id",
    "media_display_name",
    "prepared_audio_size",
    "prepared_audio_sha256",
    "profile_input_sha256",
    "evidence_carriers_sha256",
    "authority_limit_profile_kind",
    "authority_limit_profile_digest",
    "legacy_policy_binary64",
    "target_logical_id",
    "expected_vtt_sha256",
)
_POLICY_KEYS = ("min_cue_sec", "tiny_cue_sec", "tiny_cue_target")
_ROUTE_PLAN_KEYS = ("digest", "entries")
_ROUTE_ENTRY_KEYS = (
    "delivery_index",
    "source_index",
    "route_start",
    "route_end",
    "action",
    "call_index",
    "skip_reason",
)
_PHYSICAL_CALL_KEYS = (
    "call_index",
    "source_block_indices",
    "sample_start",
    "sample_end",
    "sample_rate",
    "physical_origin_seconds",
    "legacy_origin_seconds",
    "legacy_origin_kind",
    "authority_origin_seconds",
    "backend_model_config_sha256",
    "route_input_sha256",
    "strict_unit_status",
    "strict_failure",
    "raw_units_sha256",
    "relative_units_sha256",
    "legacy_retained_units",
    "legacy_slice_sha256",
    "legacy_absolute_sha256",
    "authority_transform_status",
    "authority_absolute_sha256",
    "raw_unit_ids",
)
_STRICT_FAILURE_KEYS = ("stage", "call_unit_index", "detail_code")
_LEGACY_DISTRIBUTION_KEYS = ("digest", "calls")
_LEGACY_CALL_KEYS = (
    "call_index",
    "owner_source_indices",
    "expected_counts",
    "requested_ranges",
    "realized_ranges",
    "owner_unit_ids",
    "final_cursor",
    "consumed_prefix_unit_ids",
    "shortage_source_indices",
    "leftover_unit_ids",
)
_AUTHORITY_DISTRIBUTION_KEYS = (
    "status",
    "digest",
    "owner_source_indices",
    "expected_counts",
    "owner_unit_ids",
    "consumed_count",
    "leftover_unit_ids",
    "reasons",
    "work",
)
_WORK_KEYS = (
    "status",
    "route_status",
    "route_mismatch",
    "route_claims",
    "declared_delivery_block_count",
    "declared_call_count",
    "declared_skip_count",
    "declared_raw_node_count",
    "charged_call_count",
    "limit_profile_kind",
    "limit_profile_digest",
    "limits",
    "totals",
    "terminal_call_index",
    "denied_charge",
    "calls",
    "skipped_blocks",
)
_ROUTE_MISMATCH_KEYS = (
    "kind",
    "observation_index",
    "expected_delivery_index",
    "observed_delivery_index",
)
_ROUTE_CLAIM_KEYS = (
    "owner_kind",
    "owner_index",
    "delivery_index",
    "source_index",
)
_JOB_LIMIT_KEYS = (
    "call_limit",
    "state_limit",
    "edge_limit",
    "interval_limit",
    "normalize_char_limit",
)
_CALL_LIMIT_KEYS = (
    "state_limit",
    "edge_limit",
    "interval_limit",
    "normalize_char_limit",
)
_COUNTER_KEYS = ("states", "edges", "intervals", "normalize_chars")
_DENIED_CHARGE_KEYS = (
    "lane",
    "event_ordinal",
    "event_kind",
    "subject",
    "counters",
)
_DENIED_COUNTER_KEYS = ("counter", "amount", "scopes")
_WORK_CALL_KEYS = (
    "call_index",
    "route_claim_positions",
    "source_block_indices",
    "raw_node_range",
    "block_count",
    "raw_node_count",
    "typed_unit_count",
    "surface_chars",
    "strict_preflight_status",
    "strict_failure",
    "limits",
    "allocator",
    "verifier",
)
_LANE_KEYS = (
    "status",
    "states",
    "edges",
    "intervals",
    "normalize_chars",
    "terminal_detail_code",
)
_SKIP_KEYS = (
    "route_claim_positions",
    "delivery_index",
    "source_index",
    "route_skip_reason",
    "source_text_kind",
    "detail_code",
    "work_status",
    "states",
    "edges",
    "intervals",
    "normalize_chars",
)
_BLOCK_KEYS = (
    "source_index",
    "legacy_unit_ids",
    "authority_unit_ids",
    "word_data",
    "speech_start",
    "speech_end",
)
_WORD_KEYS = (
    "unit_id",
    "call_index",
    "call_unit_index",
    "text",
    "relative_start",
    "relative_end",
    "physical_origin_seconds",
    "start",
    "end",
    "provenance",
    "original_relative_start",
    "original_relative_end",
)
_STRICT_STATUS_KEYS = ("kind", "detail_code")
_SEED_STATUS_KEYS = ("kind", "reasons")
_PROFILE_STATUS_KEYS = ("kind", "source", "detail_code")
_SELECTED_OUTPUT_KEYS = (
    "engine_family",
    "vtt_present",
    "vtt_sha256",
    "json_present",
    "json_sha256",
)
_WORK_STATUSES = {
    "complete",
    "invalid",
    "budget-exhausted",
    "not-run-route-invalid",
    "not-run-skip-invalid",
    "not-run-strict-unavailable",
}
_LANE_STATUSES = {
    "not-run",
    "not-run-prior-terminal",
    "complete",
    "invalid",
    "budget-exhausted",
}
_LANE_DETAILS = {
    "partial-empty-ownership",
    "punctuation-only-block",
    "allocation-no-tiling",
    "allocation-ambiguous",
    "allocation-budget",
}
_PROFILE_SOURCES = {
    "language-override",
    "unsupported-manifest",
    "profile-absent",
    "stored-profile",
    "manifest-absent",
}


@dataclass(frozen=True)
class SelectedOutputs:
    engine_family: EngineFamily
    vtt_present: Literal[True]
    vtt_sha256: str
    json_present: Literal[True]
    json_sha256: str

    @property
    def main_json_sha256(self) -> str:
        return self.json_sha256


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
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


def _binding_failure(detail: str = "evidence-binding") -> EvidenceBindingError:
    return EvidenceBindingError(
        CanonicalFailure("final-evidence-invalid", "evidence-bind", detail)
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _same_float(left: object, right: object) -> bool:
    return type(left) is float and type(right) is float and left.hex() == right.hex()


def _same_optional_float(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    return _same_float(left, right)


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


def _digest_projection(label: str, value: Any) -> str:
    return frozen_json_digest(freeze_json([label, value]))


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


class _DuplicateEvidenceKey(ValueError):
    pass


class _EvidenceInvalid(ValueError):
    pass


def _invalid(message: str) -> NoReturn:
    raise _EvidenceInvalid(message)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise _DuplicateEvidenceKey(key)
        value[key] = member
    return value


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"nonfinite JSON token {token}")


def _closed_mapping(value: object, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or tuple(value) != keys:
        _invalid(f"{label} keys are not closed and ordered")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        _invalid(f"{label} is not an array")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _invalid(f"{label} is not an exact integer")
    return value


def _nullable_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _exact_int(value, label)


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _invalid(f"{label} is not one finite exact float")
    return value


def _nullable_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label)


def _sha_or_none(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not _is_sha256(value):
        _invalid(f"{label} is not lowercase SHA-256")
    assert isinstance(value, str)
    return value


def _int_array(value: object, label: str, *, unique: bool = False) -> list[int]:
    members = _list(value, label)
    projected = [_exact_int(member, label) for member in members]
    if unique and len(projected) != len(set(projected)):
        _invalid(f"{label} contains a duplicate")
    return projected


def _string_array(value: object, label: str, *, unique: bool = False) -> list[str]:
    members = _list(value, label)
    if any(type(member) is not str or not member for member in members):
        _invalid(f"{label} contains a non-string")
    projected = list(members)
    if unique and len(projected) != len(set(projected)):
        _invalid(f"{label} contains a duplicate")
    return projected


def _pair(value: object, label: str) -> tuple[int, int]:
    members = _int_array(value, label)
    if len(members) != 2 or members[1] < members[0]:
        _invalid(f"{label} is not an ordered pair")
    return members[0], members[1]


def _validate_generation(
    present: object, size: object, digest: object, label: str
) -> None:
    if present is True:
        _exact_int(size, f"{label} size")
        if not _is_sha256(digest):
            _invalid(f"{label} digest is invalid")
    elif present is False:
        if size is not None or digest is not None:
            _invalid(f"absent {label} carries size/hash")
    else:
        _invalid(f"{label} presence is not an exact bool")


def _validate_history(value: object) -> dict[str, Any]:
    history = _closed_mapping(value, _INPUT_HISTORY_KEYS, "input_history")
    _validate_generation(
        history["vtt_present"],
        history["vtt_size"],
        history["vtt_sha256"],
        "VTT input",
    )
    if history["vtt_present"] is not True:
        _invalid("align evidence requires one VTT input")
    _validate_generation(
        history["sibling_json_present"],
        history["sibling_json_size"],
        history["sibling_json_sha256"],
        "sibling JSON input",
    )
    for name in (
        "block_content_sha256",
        "media_fingerprint",
        "prepared_audio_sha256",
        "profile_input_sha256",
        "evidence_carriers_sha256",
        "authority_limit_profile_digest",
    ):
        if not _is_sha256(history[name]):
            _invalid(f"input_history.{name} is invalid")
    if history["registry_family"] not in ("legacy-v1", "boundary-v2"):
        _invalid("input registry family is invalid")
    for name in ("media_logical_id", "media_display_name", "target_logical_id"):
        if type(history[name]) is not str or not history[name]:
            _invalid(f"input_history.{name} is invalid")
    _exact_int(history["prepared_audio_size"], "prepared audio size")
    if history["authority_limit_profile_kind"] not in ("production", "test-only"):
        _invalid("authority limit profile kind is invalid")
    policy = _closed_mapping(
        history["legacy_policy_binary64"], _POLICY_KEYS, "legacy policy"
    )
    for member in policy.values():
        if (
            type(member) is not str
            or len(member) != 16
            or any(character not in "0123456789abcdef" for character in member)
        ):
            _invalid("legacy policy member is not exact binary64 hex")
    _sha_or_none(history["expected_vtt_sha256"], "expected VTT digest")
    return history


def _durable_fact_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise _EvidenceInvalid("durable source facts are outside stable JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _fact_hex_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    if type(value) is not str:
        _invalid(f"{label} is not canonical binary64 text")
    try:
        decoded = float.fromhex(value)
    except ValueError as exc:
        raise _EvidenceInvalid(f"{label} is not canonical binary64 text") from exc
    if not math.isfinite(decoded) or decoded.hex() != value:
        _invalid(f"{label} is not canonical finite binary64 text")
    return decoded


def _validate_source_facts(
    value: object,
    *,
    context_digest: str,
    language: str,
    route: str,
) -> tuple[dict[str, Any], str, str]:
    facts = _closed_mapping(value, _SOURCE_FACT_KEYS, "source_facts")
    model = facts["backend_model_config"]
    if not isinstance(model, dict):
        _invalid("backend model/config source facts are not an object")
    if tuple(model) == _MODEL_FACT_KEYS:
        if model["route"] != route or model["language"] != language:
            _invalid("backend model/config facts disagree with evidence identity")
        if (
            type(model["backend"]) is not str
            or not model["backend"]
            or type(model["model"]) is not str
            or not model["model"]
        ):
            _invalid("backend model/config identity is invalid")
        _exact_int(model["sample_rate"], "model/config sample rate", minimum=1)
    elif tuple(model) == _DEFAULT_MODEL_FACT_KEYS:
        if (
            model["kind"] != "default"
            or model["route"] != route
            or model["language"] != language
        ):
            _invalid("default model/config facts disagree with evidence identity")
    else:
        _invalid("backend model/config source facts are not closed")

    route_facts = facts["route_input"]
    if not isinstance(route_facts, dict):
        _invalid("route input source facts are not an object")
    if tuple(route_facts) == _ROUTE_FACT_KEYS:
        if route_facts["route"] != route or route_facts["language"] != language:
            _invalid("route input facts disagree with evidence identity")
        blocks = _list(route_facts["blocks"], "route fact blocks")
        observed_sources: list[int] = []
        for member in blocks:
            block = _closed_mapping(member, _ROUTE_FACT_BLOCK_KEYS, "route fact block")
            observed_sources.append(
                _exact_int(block["source_index"], "route fact source index")
            )
            if type(block["alignment_text"]) is not str:
                _invalid("route fact alignment text is not an exact string")
            _fact_hex_float(block["start"], "route fact block start")
            _fact_hex_float(block["end"], "route fact block end")
        if len(observed_sources) != len(set(observed_sources)):
            _invalid("route fact source index is duplicated")
        crops = _list(route_facts["crops"], "route fact crops")
        if route in ("ctc-full", "mms-full"):
            if crops:
                _invalid("full-pass route facts carry crop windows")
        elif len(crops) != len(blocks):
            _invalid("Qwen route fact crop/block cardinality differs")
        for member in crops:
            if member is None:
                continue
            pair = _list(member, "route fact crop")
            if len(pair) != 2:
                _invalid("route fact crop is not one pair")
            start = _fact_hex_float(pair[0], "route fact crop start")
            end = _fact_hex_float(pair[1], "route fact crop end")
            assert start is not None and end is not None
            if start > end:
                _invalid("route fact crop is reversed")
    elif tuple(route_facts) == _DEFAULT_ROUTE_FACT_KEYS:
        if (
            route_facts["kind"] != "default"
            or route_facts["context_content_digest"] != context_digest
            or route_facts["route"] != route
        ):
            _invalid("default route facts disagree with evidence identity")
    else:
        _invalid("route input source facts are not closed")
    return (
        facts,
        _durable_fact_digest(model),
        _durable_fact_digest(route_facts),
    )


def _validate_route_plan(
    value: object, route: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    plan = _closed_mapping(value, _ROUTE_PLAN_KEYS, "route_plan")
    if not _is_sha256(plan["digest"]):
        _invalid("route plan digest is invalid")
    entries = _list(plan["entries"], "route entries")
    by_source: dict[int, dict[str, Any]] = {}
    for delivery_index, member in enumerate(entries):
        row = _closed_mapping(member, _ROUTE_ENTRY_KEYS, "route entry")
        if row["delivery_index"] != delivery_index:
            _invalid("route delivery indexes are not contiguous")
        source = _exact_int(row["source_index"], "route source index")
        if source in by_source:
            _invalid("route source index is duplicated")
        by_source[source] = row
        _nullable_float(row["route_start"], "route start")
        _nullable_float(row["route_end"], "route end")
        action = row["action"]
        if route in ("ctc-full", "mms-full"):
            if action != "full-pass-member" or row["skip_reason"] is not None:
                _invalid("full-pass route action is invalid")
            _exact_int(row["call_index"], "full-pass call index")
        elif action == "qwen-call":
            _exact_int(row["call_index"], "Qwen call index")
            if row["skip_reason"] is not None:
                _invalid("Qwen call carries a skip reason")
        elif action == "qwen-skip":
            if row["call_index"] is not None or row["skip_reason"] not in (
                "missing-crop",
                "empty-alignment-text",
            ):
                _invalid("Qwen skip projection is invalid")
        else:
            _invalid("route action is invalid")
    if plan["digest"] != _digest_projection("p6-route-plan-v1", entries):
        _invalid("route plan digest mismatch")
    return plan, by_source


def _validate_strict_failure(value: object, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    failure = _closed_mapping(value, _STRICT_FAILURE_KEYS, label)
    if failure["stage"] not in (
        "strict-capture",
        "sample-geometry",
        "authority-transform",
    ):
        _invalid(f"{label} stage is invalid")
    _nullable_int(failure["call_unit_index"], f"{label} unit index")
    if failure["detail_code"] not in {
        "strict-raw-node",
        "sample-geometry",
        "physical-origin-mismatch",
        "authority-recompute",
        "surplus-transform",
    }:
        _invalid(f"{label} detail is invalid")
    if (failure["stage"] == "sample-geometry") != (failure["call_unit_index"] is None):
        _invalid(f"{label} locator nullability is invalid")
    return failure


def _validate_physical_calls(
    value: object,
    route: str,
    route_by_source: Mapping[int, Mapping[str, Any]],
    model_digest: str,
    route_digest: str,
) -> list[dict[str, Any]]:
    calls = _list(value, "physical_calls")
    global_ids: set[str] = set()
    for call_index, member in enumerate(calls):
        row = _closed_mapping(member, _PHYSICAL_CALL_KEYS, "physical call")
        if row["call_index"] != call_index:
            _invalid("physical call indexes are not contiguous")
        sources = _int_array(row["source_block_indices"], "physical call sources")
        if not sources:
            _invalid("physical call owns no source block")
        for source in sources:
            entry = route_by_source.get(source)
            if entry is None or entry["call_index"] != call_index:
                _invalid("physical call and route plan disagree")
        sample_start = _exact_int(row["sample_start"], "sample start")
        sample_end = _exact_int(row["sample_end"], "sample end")
        sample_rate = _exact_int(row["sample_rate"], "sample rate", minimum=1)
        if sample_end < sample_start:
            _invalid("physical sample interval is reversed")
        quotient = sample_start / sample_rate
        physical = _finite_float(row["physical_origin_seconds"], "physical origin")
        authority = _finite_float(row["authority_origin_seconds"], "authority origin")
        legacy = _finite_float(row["legacy_origin_seconds"], "legacy origin")
        if not _same_float(physical, quotient) or not _same_float(authority, quotient):
            _invalid("physical origin does not match sample geometry")
        kind = row["legacy_origin_kind"]
        if kind == "identity":
            if sample_start != 0 or not _same_float(legacy, 0.0):
                _invalid("identity origin is not exact zero")
        elif kind == "sample-origin":
            if not _same_float(legacy, quotient):
                _invalid("sample origin does not match sample geometry")
        elif kind == "nominal-route":
            if route != "qwen-crop" or len(sources) != 1:
                _invalid("nominal-route origin is outside one Qwen call")
            if not _same_optional_float(
                legacy, route_by_source[sources[0]]["route_start"]
            ):
                _invalid("Qwen legacy origin does not match nominal route")
        else:
            _invalid("legacy origin kind is invalid")
        if row["backend_model_config_sha256"] != model_digest:
            _invalid("physical call model/config digest does not match source facts")
        if row["route_input_sha256"] != route_digest:
            _invalid("physical call route digest does not match source facts")
        failure = _validate_strict_failure(row["strict_failure"], "strict failure")
        raw_digest = _sha_or_none(row["raw_units_sha256"], "raw units digest")
        relative_digest = _sha_or_none(
            row["relative_units_sha256"], "relative units digest"
        )
        if row["strict_unit_status"] == "valid":
            if raw_digest is None or relative_digest is None:
                _invalid("valid strict call lacks complete digests")
        elif row["strict_unit_status"] == "invalid":
            if raw_digest is not None or relative_digest is not None:
                _invalid("invalid strict call carries strict digests")
            if failure is None or failure["stage"] != "strict-capture":
                _invalid("invalid strict call lacks strict-capture locator")
        else:
            _invalid("strict unit status is invalid")
        _sha_or_none(row["legacy_slice_sha256"], "legacy slice digest")
        relative_legacy_groups = _list(
            row["legacy_retained_units"], "legacy retained unit groups"
        )
        if len(relative_legacy_groups) != len(sources):
            _invalid("legacy retained groups disagree with physical sources")
        absolute_legacy_groups: list[list[dict[str, Any]]] = []
        for owner_index, owner_value in enumerate(relative_legacy_groups):
            owner = _list(owner_value, "legacy retained unit group")
            absolute_owner: list[dict[str, Any]] = []
            for unit_index, unit_value in enumerate(owner):
                unit = _closed_mapping(
                    unit_value,
                    ("text", "start", "end"),
                    "legacy retained unit",
                )
                if type(unit["text"]) is not str:
                    _invalid("legacy retained unit text is not an exact string")
                start = unit["start"]
                end = unit["end"]
                if (
                    type(start) not in (int, float)
                    or type(end) not in (int, float)
                    or not math.isfinite(start)
                    or not math.isfinite(end)
                ):
                    _invalid("legacy retained unit bounds are not finite numbers")
                if start > end:
                    _invalid(
                        f"legacy retained unit {owner_index}:{unit_index} is reversed"
                    )
                if kind == "identity":
                    absolute_start, absolute_end = start, end
                else:
                    absolute_start = start + legacy
                    absolute_end = end + legacy
                if (
                    type(absolute_start) not in (int, float)
                    or type(absolute_end) not in (int, float)
                    or not math.isfinite(absolute_start)
                    or not math.isfinite(absolute_end)
                    or absolute_start > absolute_end
                ):
                    _invalid("legacy absolute unit projection is invalid")
                absolute_owner.append(
                    {
                        "text": unit["text"],
                        "start": absolute_start,
                        "end": absolute_end,
                    }
                )
            absolute_legacy_groups.append(absolute_owner)
        legacy_absolute = _sha_or_none(
            row["legacy_absolute_sha256"], "legacy absolute digest"
        )
        recomputed_absolute = frozen_json_digest(freeze_json(absolute_legacy_groups))
        if legacy_absolute != recomputed_absolute:
            _invalid("legacy absolute digest does not match retained projections")
        authority_digest = _sha_or_none(
            row["authority_absolute_sha256"], "authority absolute digest"
        )
        if row["authority_transform_status"] == "valid":
            if (
                failure is not None
                or authority_digest is None
                or row["strict_unit_status"] != "valid"
            ):
                _invalid("valid authority transform is incomplete")
        elif row["authority_transform_status"] == "invalid":
            if failure is None or authority_digest is not None:
                _invalid("invalid authority transform is incomplete")
        else:
            _invalid("authority transform status is invalid")
        raw_ids = _string_array(row["raw_unit_ids"], "raw unit IDs", unique=True)
        if global_ids.intersection(raw_ids):
            _invalid("global raw unit ID is duplicated")
        global_ids.update(raw_ids)
    return calls


def _crosslink_source_facts(
    facts: Mapping[str, Any],
    route_plan: Mapping[str, Any],
    physical_calls: Sequence[Mapping[str, Any]],
) -> None:
    model = facts["backend_model_config"]
    if tuple(model) == _MODEL_FACT_KEYS:
        sample_rate = model["sample_rate"]
        if any(call["sample_rate"] != sample_rate for call in physical_calls):
            _invalid("model/config sample rate disagrees with physical calls")

    route_facts = facts["route_input"]
    if tuple(route_facts) != _ROUTE_FACT_KEYS:
        return
    blocks = route_facts["blocks"]
    crops = route_facts["crops"]
    entries = route_plan["entries"]
    if len(blocks) != len(entries):
        _invalid("route source facts disagree with route-plan cardinality")
    for index, (block, entry) in enumerate(zip(blocks, entries, strict=True)):
        if block["source_index"] != entry["source_index"]:
            _invalid("route source facts disagree with delivery source order")
        block_start = _fact_hex_float(block["start"], "route fact block start")
        block_end = _fact_hex_float(block["end"], "route fact block end")
        if route_facts["route"] in ("ctc-full", "mms-full"):
            if not _same_optional_float(block_start, entry["route_start"]):
                _invalid("full-pass route start disagrees with source facts")
            if not _same_optional_float(block_end, entry["route_end"]):
                _invalid("full-pass route end disagrees with source facts")
            continue
        crop = crops[index]
        if entry["action"] == "qwen-call":
            if not isinstance(crop, list):
                _invalid("Qwen call lacks its durable crop facts")
            crop_start = _fact_hex_float(crop[0], "route fact crop start")
            if not _same_optional_float(crop_start, entry["route_start"]):
                _invalid("Qwen route start disagrees with crop facts")
        elif crop is not None:
            _invalid("Qwen skip carries durable crop facts")
        elif not _same_optional_float(block_start, entry["route_start"]):
            _invalid("Qwen skipped route start disagrees with block facts")
        if block_end is not None and not _same_optional_float(
            block_end, entry["route_end"]
        ):
            _invalid("Qwen route end disagrees with block facts")


def _validate_legacy(
    value: object,
    physical_calls: Sequence[Mapping[str, Any]],
    route: str,
) -> tuple[dict[str, Any], dict[int, list[str]]]:
    legacy = _closed_mapping(value, _LEGACY_DISTRIBUTION_KEYS, "legacy_distribution")
    calls = _list(legacy["calls"], "legacy calls")
    if len(calls) != len(physical_calls):
        _invalid("legacy/physical call cardinality mismatch")
    by_source: dict[int, list[str]] = {}
    for call_index, (member, physical) in enumerate(
        zip(calls, physical_calls, strict=True)
    ):
        row = _closed_mapping(member, _LEGACY_CALL_KEYS, "legacy call")
        if row["call_index"] != call_index:
            _invalid("legacy call indexes are not contiguous")
        sources = _int_array(row["owner_source_indices"], "legacy owner sources")
        if sources != physical["source_block_indices"]:
            _invalid("legacy owners disagree with physical call")
        expected = _int_array(row["expected_counts"], "legacy expected counts")
        requested = _list(row["requested_ranges"], "legacy requested ranges")
        realized = _list(row["realized_ranges"], "legacy realized ranges")
        owners = _list(row["owner_unit_ids"], "legacy owner unit IDs")
        if not (
            len(sources)
            == len(expected)
            == len(requested)
            == len(realized)
            == len(owners)
        ):
            _invalid("legacy slice vectors disagree in length")
        raw_ids = physical["raw_unit_ids"]
        if route == "qwen-crop" and (len(sources) != 1 or expected != [len(raw_ids)]):
            _invalid("Qwen legacy receipt does not retain its complete raw result")
        cursor = 0
        shortages: list[int] = []
        for source, count, requested_pair, realized_pair, owner_value in zip(
            sources, expected, requested, realized, owners, strict=True
        ):
            request = _pair(requested_pair, "legacy requested range")
            actual = _pair(realized_pair, "legacy realized range")
            if request != (cursor, cursor + count):
                _invalid("legacy requested range is not the count cursor")
            clamped = (min(cursor, len(raw_ids)), min(cursor + count, len(raw_ids)))
            if actual != clamped:
                _invalid("legacy realized range is not Python slice clamping")
            owner_ids = _string_array(owner_value, "legacy owner IDs", unique=True)
            if owner_ids != raw_ids[actual[0] : actual[1]]:
                _invalid("legacy owner IDs do not match retained raw slice")
            if source in by_source:
                _invalid("legacy source is owned twice")
            by_source[source] = owner_ids
            if len(owner_ids) < count:
                shortages.append(source)
            cursor += count
        if row["final_cursor"] != cursor:
            _invalid("legacy final cursor mismatch")
        prefix = raw_ids[: min(cursor, len(raw_ids))]
        leftovers = raw_ids[min(cursor, len(raw_ids)) :]
        if (
            _string_array(
                row["consumed_prefix_unit_ids"], "legacy consumed prefix", unique=True
            )
            != prefix
        ):
            _invalid("legacy consumed prefix mismatch")
        if _int_array(row["shortage_source_indices"], "legacy shortages") != shortages:
            _invalid("legacy shortage projection mismatch")
        if (
            _string_array(row["leftover_unit_ids"], "legacy leftovers", unique=True)
            != leftovers
        ):
            _invalid("legacy leftover projection mismatch")
        retained_groups = physical["legacy_retained_units"]
        if any(
            len(group) != len(owner)
            for group, owner in zip(retained_groups, owners, strict=True)
        ):
            _invalid("legacy retained unit cardinality disagrees with owner IDs")
        slice_value = {
            key: row[key] for key in _LEGACY_CALL_KEYS if key != "call_index"
        }
        recomputed_slice = frozen_json_digest(freeze_json(slice_value))
        if physical["legacy_slice_sha256"] != recomputed_slice:
            _invalid("legacy slice digest does not match durable receipt")
    if not _is_sha256(legacy["digest"]) or legacy["digest"] != _digest_projection(
        "p6-legacy-distribution-v1", calls
    ):
        _invalid("legacy distribution digest mismatch")
    return legacy, by_source


def _validate_limits(
    kind: str,
    profile_digest: object,
    job: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
) -> None:
    production_job = (
        AUTH_ALLOC_JOB_CALL_LIMIT,
        AUTH_ALLOC_JOB_STATE_LIMIT,
        AUTH_ALLOC_JOB_EDGE_LIMIT,
        AUTH_ALLOC_JOB_INTERVAL_LIMIT,
        AUTH_ALLOC_JOB_NORMALIZE_CHAR_LIMIT,
    )
    production_call = (
        AUTH_ALLOC_STATE_LIMIT,
        AUTH_ALLOC_EDGE_LIMIT,
        AUTH_ALLOC_INTERVAL_LIMIT,
        AUTH_ALLOC_NORMALIZE_CHAR_LIMIT,
    )
    job_values = tuple(job[name] for name in _JOB_LIMIT_KEYS)
    if any(type(member) is not int or member <= 0 for member in job_values):
        _invalid("job authority limits are invalid")
    call_values: tuple[Any, ...] | None = None
    for call in calls:
        limits = _closed_mapping(call["limits"], _CALL_LIMIT_KEYS, "call limits")
        observed = tuple(limits[name] for name in _CALL_LIMIT_KEYS)
        if any(type(member) is not int or member <= 0 for member in observed):
            _invalid("call authority limits are invalid")
        if call_values is None:
            call_values = observed
        elif call_values != observed:
            _invalid("call authority limits disagree")
    if call_values is None:
        call_values = production_call if kind == "production" else None
    if call_values is None:
        _invalid("test-only profile lacks one physical call")
    if kind == "production":
        if job_values != production_job or call_values != production_call:
            _invalid("production authority limit values changed")
    elif kind == "test-only":
        if any(
            observed > maximum
            for observed, maximum in zip(job_values, production_job, strict=True)
        ) or any(
            observed > maximum
            for observed, maximum in zip(call_values, production_call, strict=True)
        ):
            _invalid("test-only authority limit increased")
        if job_values == production_job and call_values == production_call:
            _invalid("test-only authority profile did not decrease")
    else:
        _invalid("authority profile kind is invalid")
    from voxweave.align_distribution import (
        AuthorityLimitProfile,
        CallWorkLimits,
        JobWorkLimits,
        _profile_digest,
    )

    profile = AuthorityLimitProfile(
        kind,
        CallWorkLimits(*call_values),
        JobWorkLimits(*job_values),
        str(profile_digest),
    )
    if profile.profile_digest != _profile_digest(
        profile.kind, profile.call, profile.job
    ):
        _invalid("authority profile digest mismatch")


def _project_route_mismatch(
    claims: Sequence[Mapping[str, Any]],
    route_entries: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    skips: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    count = len(route_entries)
    observed = [claim["delivery_index"] for claim in claims]
    present = {index for index in observed if 0 <= index < count}
    for expected in range(count):
        if expected not in present:
            return {
                "kind": "gap",
                "observation_index": None,
                "expected_delivery_index": expected,
                "observed_delivery_index": None,
            }
    for duplicated in range(count):
        positions = [
            position for position, index in enumerate(observed) if index == duplicated
        ]
        if len(positions) > 1:
            position = positions[1]
            return {
                "kind": "overlap",
                "observation_index": position,
                "expected_delivery_index": position if position < count else None,
                "observed_delivery_index": observed[position],
            }
    for position, index in enumerate(observed):
        if index < 0 or index >= count:
            return {
                "kind": "unexpected-index",
                "observation_index": position,
                "expected_delivery_index": position if position < count else None,
                "observed_delivery_index": index,
            }
    if observed != list(range(count)):
        position = next(
            position
            for position, (left, right) in enumerate(zip(observed, range(count)))
            if left != right
        )
        return {
            "kind": "reorder",
            "observation_index": position,
            "expected_delivery_index": position,
            "observed_delivery_index": observed[position],
        }
    for position, claim in enumerate(claims):
        route = route_entries[position]
        owner_kind = claim["owner_kind"]
        owner_index = claim["owner_index"]
        exists = (owner_kind == "call" and owner_index < len(calls)) or (
            owner_kind == "skip" and owner_index < len(skips)
        )
        expected_kind = "skip" if route["action"] == "qwen-skip" else "call"
        expected_owner = (
            route["delivery_index"] if expected_kind == "skip" else route["call_index"]
        )
        if (
            not exists
            or claim["source_index"] != route["source_index"]
            or owner_kind != expected_kind
            or owner_index != expected_owner
        ):
            return {
                "kind": "owner-crosslink",
                "observation_index": position,
                "expected_delivery_index": position,
                "observed_delivery_index": claim["delivery_index"],
            }
    return None


def _validate_lane(value: object, label: str) -> dict[str, Any]:
    lane = _closed_mapping(value, _LANE_KEYS, label)
    if lane["status"] not in _LANE_STATUSES:
        _invalid(f"{label} status is invalid")
    for name in _COUNTER_KEYS:
        _exact_int(lane[name], f"{label} {name}")
    detail = lane["terminal_detail_code"]
    if detail is not None and detail not in _LANE_DETAILS:
        _invalid(f"{label} terminal detail is invalid")
    if lane["status"] in ("complete", "not-run", "not-run-prior-terminal"):
        if detail is not None:
            _invalid(f"{label} nonterminal status carries a detail")
    elif detail is None:
        _invalid(f"{label} terminal status lacks a detail")
    if lane["status"].startswith("not-run") and any(
        lane[name] for name in _COUNTER_KEYS
    ):
        _invalid(f"{label} not-run lane carries work")
    return lane


def _validate_denied_charge(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    denied = _closed_mapping(value, _DENIED_CHARGE_KEYS, "denied charge")
    if denied["lane"] not in ("job", "allocator", "verifier"):
        _invalid("denied charge lane is invalid")
    _exact_int(denied["event_ordinal"], "denied charge ordinal")
    arity = {
        "call-start": 1,
        "block-normalize": 1,
        "state-insert": 2,
        "edge-test": 3,
        "interval-normalize": 2,
    }.get(denied["event_kind"])
    subject = _int_array(denied["subject"], "denied charge subject")
    if arity is None or len(subject) != arity:
        _invalid("denied charge event subject is invalid")
    counters = _list(denied["counters"], "denied counters")
    if not counters:
        _invalid("denied charge has no denied counter")
    order = {name: index for index, name in enumerate(("calls", *_COUNTER_KEYS))}
    observed: list[int] = []
    for member in counters:
        row = _closed_mapping(member, _DENIED_COUNTER_KEYS, "denied counter")
        if row["counter"] not in order:
            _invalid("denied counter name is invalid")
        observed.append(order[row["counter"]])
        _exact_int(row["amount"], "denied counter amount", minimum=1)
        scopes = _list(row["scopes"], "denied counter scopes")
        if not scopes or any(scope not in ("job", "call") for scope in scopes):
            _invalid("denied counter scopes are invalid")
        if scopes != [scope for scope in ("job", "call") if scope in scopes]:
            _invalid("denied counter scopes are not canonical")
    if observed != sorted(set(observed)):
        _invalid("denied counters are not unique/canonical")
    return denied


def _work_reason_projection(work: Mapping[str, Any]) -> list[str]:
    present: set[str] = set()
    if work["route_status"] == "invalid":
        present.add("route-owner-mismatch")
    if work["skipped_blocks"]:
        present.add("partial-empty-ownership")
    if any(row["strict_failure"] is not None for row in work["calls"]):
        present.add("authority-transform-invalid")
    for row in work["calls"]:
        detail = row["allocator"]["terminal_detail_code"]
        if detail in (
            "partial-empty-ownership",
            "punctuation-only-block",
            "allocation-no-tiling",
            "allocation-ambiguous",
        ):
            present.add(detail)
    if work["status"] == "budget-exhausted":
        present.add("allocation-budget-exhausted")
    return [reason for reason in AUTHORITY_REASON_ORDER if reason in present]


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


def _swap_ext(path: Path, new_ext: str) -> Path:
    target = Path(path)
    if target.suffix:
        return target.with_name(target.name[: -len(target.suffix)] + new_ext)
    return target.with_name(target.name + new_ext)


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
            from voxweave.pipeline import _find_sibling_media

            resolved = _find_sibling_media(vtt_path)
        except (ImportError, OSError):
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
    """Return the unsigned section 9.3 usability audit conjunction."""
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
    legacy_evidence = _swap_ext(target, ".align-evidence.json")
    artifact_media = (
        Path(explicit_media_path) if explicit_media_path is not None else None
    )
    if artifact_media is None:
        try:
            from voxweave.pipeline import _find_subtitle_media

            artifact_media = _find_subtitle_media(target)
        except (ImportError, OSError):
            artifact_media = None
    if artifacts.path_present(legacy_evidence):
        evidence_path = legacy_evidence
    else:
        cached = artifacts.inspect_paths(artifact_media or target)
        evidence_path = (
            legacy_evidence if cached is None else cached.align_evidence(target)
        )
    json_path = _swap_ext(target, ".json")
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
