"""Closed key tables and field validators for durable align evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

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
from voxweave.align_failures import (
    AUTHORITY_REASON_ORDER,
)
from voxweave.align_snapshot import (
    freeze_json,
    frozen_json_digest,
)


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


def _digest_projection(label: str, value: Any) -> str:
    return frozen_json_digest(freeze_json([label, value]))


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
                # No start <= end check: the legacy lane records the aligner's units
                # verbatim, and an aligner may return a reversed one. The strict
                # authority lane is what judges it (and invalidates authority).
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
    skip_ordinal = 0
    for position, claim in enumerate(claims):
        route = route_entries[position]
        owner_kind = claim["owner_kind"]
        owner_index = claim["owner_index"]
        exists = (owner_kind == "call" and owner_index < len(calls)) or (
            owner_kind == "skip" and owner_index < len(skips)
        )
        expected_kind = "skip" if route["action"] == "qwen-skip" else "call"
        if expected_kind == "skip":
            # A skip is owned by its ordinal among the plan's qwen-skip entries.
            expected_owner = skip_ordinal
            skip_ordinal += 1
        else:
            expected_owner = route["call_index"]
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
