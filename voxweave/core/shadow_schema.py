"""Closed admission contract for a complete live P5 shadow artifact.

The optimizer's standalone artifact is deliberately schema 1: it cannot carry
the finalizer row matrix, authority roots, or the W4 calibration evidence.  A
payload may claim schema 2 only after the live assembler has populated every
block below and this validator has accepted the cross-block relationships.

This module contains no production hook imports.  The hook imports it lazily
after ``VOXWEAVE_SEG_V2_SHADOW`` is enabled, preserving the flag-off import pin.
The calibration harness imports the same function and therefore cannot drift to
a weaker interpretation of the live contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .shadow_lanes import LANE_CORE, LANE_DISPLAY, LANE_FINALIZER, LANE_LEGACY
from .shadow_schema_fields import (
    _BUCKET_KEYS as _BUCKET_KEYS,
    _CHECK_KEYS as _CHECK_KEYS,
    _MARGIN_KEYS as _MARGIN_KEYS,
    _MEASUREMENT_KEYS as _MEASUREMENT_KEYS,
    _UNIT_KEYS as _UNIT_KEYS,
    _closed as _closed,
    _display_profile as _display_profile,
    _finite_number as _finite_number,
    _list as _list,
    _mapping as _mapping,
    _margin_summary_block as _margin_summary_block,
    _nonnegative_int as _nonnegative_int,
    _report_rows as _report_rows,
    _source_units as _source_units,
    _speaker_measurement as _speaker_measurement,
    _string_list as _string_list,
    _validator_block as _validator_block,
    _waiver_rows as _waiver_rows,
)
from .shadow_schema_finalizer import (
    _BOUNDARY_KEYS as _BOUNDARY_KEYS,
    _CYCLE_KEYS as _CYCLE_KEYS,
    _CYCLE_VALUES_KEYS as _CYCLE_VALUES_KEYS,
    _EVIDENCE_KEYS as _EVIDENCE_KEYS,
    _FINALIZER_KEYS as _FINALIZER_KEYS,
    _MOVEMENT_DISTRIBUTION_KEYS as _MOVEMENT_DISTRIBUTION_KEYS,
    _MOVEMENT_KEYS as _MOVEMENT_KEYS,
    _MOVEMENT_SUMMARY_KEYS as _MOVEMENT_SUMMARY_KEYS,
    _POLICY_KEYS as _POLICY_KEYS,
    _READ_KEYS as _READ_KEYS,
    _SEED_PAYLOAD_KEYS as _SEED_PAYLOAD_KEYS,
    _TRACE_KEYS as _TRACE_KEYS,
    _TRACE_LEG_KEYS as _TRACE_LEG_KEYS,
    _VERIFICATION_KEYS as _VERIFICATION_KEYS,
    _boundary_ref as _boundary_ref,
    _finalizer as _finalizer,
    _finalizer_inputs as _finalizer_inputs,
    _finalizer_verification as _finalizer_verification,
    _movement_distribution as _movement_distribution,
    _stream_state as _stream_state,
    _trace as _trace,
)

LIVE_SHADOW_SCHEMA_VERSION = 2

TOP_LEVEL_KEYS = frozenset(
    {
        "authorities",
        "canonical_fallback_rechecks",
        "coverage",
        "delta_registry",
        "diff_classification",
        "engine_v2",
        "finalizer",
        "influence_cell",
        "intervals",
        "invalid_finalizer_rows",
        "kind",
        "lanes",
        "language",
        "margin_summary",
        "pause_knees",
        "policy_deltas",
        "policy_name",
        "policy_version",
        "preview_fidelity",
        "production_degraded",
        "profile",
        "providers",
        "raw",
        "refiner_comparison",
        "schema_version",
        "shadow_degraded",
        "speaker_evidence",
        "subunit_split",
        "totals",
        "units",
        "v1",
        "v1_projection",
        "vad_state",
        "validator",
    }
)

_COVERAGE_KEYS = frozenset(
    {
        "coarse_caused_intervals",
        "coarse_granularity_intervals",
        "dual_form_unmeasured",
        "fallback_intervals",
        "fallback_ranges_overlap",
        "fallback_unit_ranges",
        "named_multi_cues_unannotated",
        "optimized_intervals",
        "optimized_unit_ratio",
        "raw_conservation_trustworthy",
        "unit_count",
        "v1_unprojected",
    }
)
_TOTAL_KEYS = frozenset(
    {
        "all_invisible_intervals",
        "atom_count",
        "barrier_count",
        "canonical_chars",
        "cap_relief_nodes",
        "coalesced_atoms",
        "coarse_caused_intervals",
        "coarse_granularity_intervals",
        "dp_relaxations",
        "fallback_intervals",
        "hard_violations",
        "interval_count",
        "optimized_intervals",
        "optimized_unit_ratio",
        "packer_steps",
        "relief_injections",
        "sentence_ends_missed",
        "unit_count",
        "waivers",
    }
)
_VALIDATOR_KEYS = frozenset(
    {
        "core",
        "finalizer",
        "interval_document_agree",
        "interval_hard_violations",
        "legacy_overlay",
        "raw",
        "raw_duplicate_v1_cues",
    }
)
_STREAM_KEYS = frozenset({"cue_count", "cues", "partition", "projection", "validator"})
_CUE_KEYS = frozenset(
    {
        "end",
        "index",
        "lines",
        "lyric",
        "speaker_ids",
        "speech_end",
        "speech_start",
        "start",
        "text",
        "unit_range",
    }
)
_FINALIZER_ROW_KEYS = _STREAM_KEYS | {"finalizer", "verification"}
_SPEAKER_KEYS = frozenset(
    {
        "attribution",
        "conditioning",
        "constants",
        "measurement",
        "measurement_refusal",
        "off_row_measurement",
        "parent_count",
        "pricing",
        "projection",
        "raw_turn_change_count",
        "refined_unit_count",
        "turn_track_present",
    }
)
_DIFF_KEYS = frozenset(
    {
        "alignment_error",
        "changed_fields",
        "independent_fired",
        "producer_fired",
        "relation_failures",
        "trigger_mismatches",
        "unclassified_field_diff",
    }
)
_DIFF_ROW_KEYS = frozenset(
    {"allowed_relation", "field", "from", "to", "trigger_ids", "unit_range"}
)
_PREVIEW_KEYS = frozenset(
    {
        "checked_edges",
        "mismatches",
        "scored_edges",
        "selected_rows",
        "uncheckable_edges",
    }
)
_PREVIEW_ROW_KEYS = frozenset({"cue_count", "edge_count", "mismatches"})
_AUTHORITY_KEYS = frozenset({"events", "expected", "lineage", "violations"})
_SUBUNIT_KEYS = frozenset(
    {"degraded", "evidence", "minted", "origin", "refined_parent_count"}
)
_SUBUNIT_EVIDENCE_KEYS = frozenset({"per-char", "phrase", "punct", "whitespace"})
_RECHECK_KEYS = frozenset(
    {
        "cue_index",
        "reason",
        "row",
        "with_owned_footprint",
        "with_owned_footprint_reason",
    }
)


def _cue_rows(
    value: Any, *, unit_count: int | None, path: str, errors: list[str]
) -> list[Any] | None:
    cues = _list(value, path, errors)
    if cues is None:
        return None
    ranges: list[tuple[int, int]] = []
    for index, raw in enumerate(cues):
        row_path = f"{path}[{index}]"
        row = _mapping(raw, row_path, errors)
        if row is None:
            continue
        _closed(row, _CUE_KEYS, row_path, errors)
        if row.get("index") != index:
            errors.append(f"{row_path}.index: expected {index}")
        unit_range = row.get("unit_range")
        if (
            not isinstance(unit_range, list)
            or len(unit_range) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in unit_range
            )
        ):
            errors.append(f"{row_path}.unit_range: expected two integer bounds")
        else:
            low, high = unit_range
            if not 0 <= low < high or (unit_count is not None and high > unit_count):
                errors.append(f"{row_path}.unit_range: invalid bounds {unit_range!r}")
            else:
                ranges.append((low, high))
        start = _finite_number(row.get("start"), f"{row_path}.start", errors)
        end = _finite_number(row.get("end"), f"{row_path}.end", errors)
        if start is not None and end is not None and end < start:
            errors.append(f"{row_path}: end precedes start")
        for key in ("speech_start", "speech_end"):
            if row.get(key) is not None:
                _finite_number(row.get(key), f"{row_path}.{key}", errors)
        if not isinstance(row.get("text"), str):
            errors.append(f"{row_path}.text: expected string")
        _nonnegative_int(row.get("lines"), f"{row_path}.lines", errors)
        if not isinstance(row.get("lyric"), bool):
            errors.append(f"{row_path}.lyric: expected boolean")
        speakers = row.get("speaker_ids")
        if not isinstance(speakers, list):
            errors.append(f"{row_path}.speaker_ids: expected list")
        elif any(not isinstance(speaker, str) for speaker in speakers):
            errors.append(f"{row_path}.speaker_ids: expected strings")
    if ranges:
        if ranges[0][0] != 0:
            errors.append(f"{path}: ownership does not start at unit zero")
        if any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
            errors.append(f"{path}: ownership ranges are not contiguous")
        if unit_count is not None and ranges[-1][1] != unit_count:
            errors.append(f"{path}: ownership does not cover {unit_count} units")
    elif unit_count not in (None, 0):
        errors.append(f"{path}: non-empty document has no cue ownership rows")
    for index, (left, right) in enumerate(zip(cues, cues[1:])):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        left_end = left.get("end")
        right_start = right.get("start")
        if (
            isinstance(left_end, (int, float))
            and not isinstance(left_end, bool)
            and math.isfinite(float(left_end))
            and isinstance(right_start, (int, float))
            and not isinstance(right_start, bool)
            and math.isfinite(float(right_start))
            and float(right_start) < float(left_end) - 1e-6
        ):
            errors.append(
                f"{path}: overlap between cues {index} and {index + 1} "
                f"({left_end} > {right_start})"
            )
    return cues


def _recompute_partition_row(
    row: Mapping[str, Any],
    *,
    units: Sequence[Any] | None,
    profile: Any | None,
    path: str,
    errors: list[str],
    validator_stage: str,
    validator_origins: frozenset[str],
    finalizer: bool,
) -> None:
    """Rebuild one partition check from serialized values, never producer aliases."""
    from .partition_check import check_partition

    cues = row.get("cues")
    partition = row.get("partition")
    validator = row.get("validator")
    if (
        units is None
        or profile is None
        or not isinstance(cues, list)
        or not isinstance(partition, list)
        or not isinstance(validator, Mapping)
    ):
        return
    if any(isinstance(item, bool) or not isinstance(item, int) for item in partition):
        return
    unit_count = len(units)
    if any(not 0 < item < unit_count for item in partition):
        errors.append(f"{path}.partition: cuts must lie inside (0, {unit_count})")
    if any(left >= right for left, right in zip(partition, partition[1:])):
        errors.append(f"{path}.partition: cuts must be strictly increasing")

    expected_cuts: list[int] = []
    row_cues: list[dict[str, Any]] = []
    for index, raw in enumerate(cues):
        if not isinstance(raw, Mapping):
            return
        unit_range = raw.get("unit_range")
        if (
            not isinstance(unit_range, list)
            or len(unit_range) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in unit_range
            )
        ):
            return
        if index + 1 < len(cues):
            expected_cuts.append(unit_range[1])
        row_cues.append(
            {
                "end": raw.get("end"),
                "lyric": raw.get("lyric"),
                "speech_end": raw.get("speech_end"),
                "speech_start": raw.get("speech_start"),
                "start": raw.get("start"),
                "text": raw.get("text"),
            }
        )
    if partition != expected_cuts:
        errors.append(
            f"{path}.partition: expected cue unit-range boundaries "
            f"{expected_cuts!r}, got {partition!r}"
        )

    origin = validator.get("origin")
    if origin not in validator_origins or validator_stage not in {
        "raw",
        "core",
        "legacy-overlay",
        "finalizer",
    }:
        return
    waivers_value: Any = validator.get("waivers")
    reports: list[Any] = []
    if finalizer:
        finalizer_block = row.get("finalizer")
        if not isinstance(finalizer_block, Mapping):
            return
        waivers_value = finalizer_block.get("waivers")
        decoded_reports = _report_rows(
            finalizer_block.get("entries"), f"{path}.finalizer.entries", errors
        )
        if decoded_reports is None:
            return
        reports = decoded_reports
    waivers = _waiver_rows(waivers_value, f"{path}.replay_waivers", errors)
    if waivers is None:
        return
    recomputed = check_partition(
        partition,
        cast("Sequence[Any]", row_cues),
        units=cast("Sequence[Any]", units),
        profile=profile,
        origin=cast("Any", origin),
        stage=cast("Any", validator_stage),
        waivers={waiver.cue_index: waiver for waiver in waivers},
        reports=reports,
    ).to_dict()
    if recomputed != validator:
        errors.append(
            f"{path}.validator: serialized check_partition result is stale; "
            f"recomputed {recomputed!r}"
        )


def _stream_row(
    value: Any,
    *,
    unit_count: int | None,
    units: Sequence[Any] | None,
    profile: Any | None,
    path: str,
    errors: list[str],
    validator_stage: str,
    validator_origins: frozenset[str],
    finalizer: bool = False,
    speaker_measurement: bool = False,
    projection_cross_check: bool = False,
    finalizer_row_id: str | None = None,
) -> Mapping[str, Any] | None:
    row = _mapping(value, path, errors)
    if row is None:
        return None
    expected = _FINALIZER_ROW_KEYS if finalizer else _STREAM_KEYS
    if speaker_measurement:
        expected = expected | {"speaker_measurement"}
    if projection_cross_check:
        expected = expected | {"projection_cross_check"}
    _closed(row, frozenset(expected), path, errors)
    cue_count = _nonnegative_int(row.get("cue_count"), f"{path}.cue_count", errors)
    cues = _cue_rows(
        row.get("cues"), unit_count=unit_count, path=f"{path}.cues", errors=errors
    )
    partition = _list(row.get("partition"), f"{path}.partition", errors)
    if cues is not None and cue_count is not None and len(cues) != cue_count:
        errors.append(f"{path}: cue_count {cue_count} != {len(cues)} serialized cues")
    if partition is not None:
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in partition
        ):
            errors.append(f"{path}.partition: cuts must be integers")
        if cue_count is not None and len(partition) != max(0, cue_count - 1):
            errors.append(f"{path}: partition/cue cardinality mismatch")
    if not isinstance(row.get("projection"), str):
        errors.append(f"{path}.projection: expected string")
    _validator_block(
        row.get("validator"),
        f"{path}.validator",
        errors,
        expected_stage=validator_stage,
        permitted_origins=validator_origins,
        expected_cue_count=cue_count,
        expected_unit_count=unit_count,
    )
    if projection_cross_check:
        cross = _mapping(
            row.get("projection_cross_check"),
            f"{path}.projection_cross_check",
            errors,
        )
        if cross is not None:
            _closed(
                cross,
                frozenset({"agrees", "mode"}),
                f"{path}.projection_cross_check",
                errors,
            )
            if cross.get("agrees") is not True:
                errors.append(f"{path}.projection_cross_check.agrees: expected true")
    if finalizer:
        seed = _finalizer_verification(
            row.get("verification"),
            row,
            path=f"{path}.verification",
            errors=errors,
            expected_row_id=finalizer_row_id,
            profile=profile,
        )
        _finalizer(
            row.get("finalizer"),
            f"{path}.finalizer",
            errors,
            row=row,
            seed=seed,
            verification=row.get("verification"),
            profile=profile,
        )
    if speaker_measurement:
        _speaker_measurement(
            row.get("speaker_measurement"), f"{path}.speaker_measurement", errors
        )
    _recompute_partition_row(
        row,
        units=units,
        profile=profile,
        path=path,
        errors=errors,
        validator_stage=validator_stage,
        validator_origins=validator_origins,
        finalizer=finalizer,
    )
    return row


def _lanes(
    value: Any,
    *,
    unit_count: int | None,
    units: Sequence[Any] | None,
    profile: Any | None,
    errors: list[str],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    lanes = _mapping(value, "lanes", errors)
    if lanes is None:
        return None, None
    _closed(
        lanes,
        frozenset({LANE_CORE, LANE_LEGACY, LANE_FINALIZER, LANE_DISPLAY}),
        "lanes",
        errors,
    )
    for lane_id, stage in ((LANE_CORE, "core"), (LANE_LEGACY, "legacy-overlay")):
        lane = _mapping(lanes.get(lane_id), f"lanes.{lane_id}", errors)
        if lane is None:
            continue
        _closed(
            lane,
            frozenset({"agreement", "lane", "stage", "v1", "v2"}),
            f"lanes.{lane_id}",
            errors,
        )
        if lane.get("lane") != lane_id or lane.get("stage") != stage:
            errors.append(f"lanes.{lane_id}: lane/stage identity mismatch")
        _mapping(lane.get("agreement"), f"lanes.{lane_id}.agreement", errors)
        _stream_row(
            lane.get("v1"),
            unit_count=unit_count,
            units=units,
            profile=profile,
            path=f"lanes.{lane_id}.v1",
            errors=errors,
            validator_stage=stage,
            validator_origins=frozenset({"v1"}),
        )
        _stream_row(
            lane.get("v2"),
            unit_count=unit_count,
            units=units,
            profile=profile,
            path=f"lanes.{lane_id}.v2",
            errors=errors,
            validator_stage=stage,
            validator_origins=frozenset({"v2"}),
            projection_cross_check=lane_id == LANE_CORE,
        )

    final_lane = _mapping(lanes.get(LANE_FINALIZER), f"lanes.{LANE_FINALIZER}", errors)
    final_rows: Mapping[str, Any] | None = None
    if final_lane is not None:
        _closed(
            final_lane,
            frozenset({"lane", "rows", "stage"}),
            f"lanes.{LANE_FINALIZER}",
            errors,
        )
        if (
            final_lane.get("lane") != LANE_FINALIZER
            or final_lane.get("stage") != "finalizer"
        ):
            errors.append(f"lanes.{LANE_FINALIZER}: lane/stage identity mismatch")
        final_rows = _mapping(
            final_lane.get("rows"), f"lanes.{LANE_FINALIZER}.rows", errors
        )
        if final_rows is not None:
            required = {"v1", "v2", "v2-speaker-off"}
            if not required <= set(final_rows) or set(final_rows) - (
                required | {"refiner-off"}
            ):
                errors.append(
                    f"lanes.{LANE_FINALIZER}.rows: expected v1, v2, v2-speaker-off and optional refiner-off"
                )
            for row_id in sorted(set(final_rows) & (required | {"refiner-off"})):
                _stream_row(
                    final_rows[row_id],
                    unit_count=unit_count,
                    units=units,
                    profile=profile,
                    path=f"lanes.{LANE_FINALIZER}.rows.{row_id}",
                    errors=errors,
                    validator_stage="finalizer",
                    validator_origins=frozenset({"v1" if row_id == "v1" else "v2"}),
                    finalizer=True,
                    speaker_measurement=row_id == "v2-speaker-off",
                    finalizer_row_id=row_id,
                )

    display = _mapping(lanes.get(LANE_DISPLAY), f"lanes.{LANE_DISPLAY}", errors)
    if display is not None:
        _closed(
            display,
            frozenset({"lane", "rows", "stage"}),
            f"lanes.{LANE_DISPLAY}",
            errors,
        )
        if (
            display.get("lane") != LANE_DISPLAY
            or display.get("stage") != "legacy-display"
        ):
            errors.append(f"lanes.{LANE_DISPLAY}: lane/stage identity mismatch")
        rows = _mapping(display.get("rows"), f"lanes.{LANE_DISPLAY}.rows", errors)
        if rows is not None:
            _closed(rows, frozenset({"v1"}), f"lanes.{LANE_DISPLAY}.rows", errors)
            _stream_row(
                rows.get("v1"),
                unit_count=unit_count,
                units=units,
                profile=profile,
                path=f"lanes.{LANE_DISPLAY}.rows.v1",
                errors=errors,
                validator_stage="core",
                validator_origins=frozenset({"v1"}),
            )
    return lanes, final_rows


def _lane_row(
    lanes: Mapping[str, Any] | None, lane_id: str, row_id: str
) -> Mapping[str, Any] | None:
    """Return one already-validated row without inventing a fallback slot."""
    if lanes is None:
        return None
    lane = lanes.get(lane_id)
    if not isinstance(lane, Mapping):
        return None
    if "rows" in lane:
        rows = lane.get("rows")
        if not isinstance(rows, Mapping):
            return None
        row = rows.get(row_id)
    else:
        row = lane.get(row_id)
    return row if isinstance(row, Mapping) else None


def _declared_cue_count(row: Mapping[str, Any] | None) -> int | None:
    if row is None:
        return None
    value = row.get("cue_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _fallback_rechecks(
    artifact: Mapping[str, Any], final_rows: Mapping[str, Any] | None, errors: list[str]
) -> None:
    checks = _list(
        artifact.get("canonical_fallback_rechecks"),
        "canonical_fallback_rechecks",
        errors,
    )
    if checks is None or final_rows is None:
        return
    expected: list[tuple[str, int, Any]] = []
    for row_id, raw_row in final_rows.items():
        if not isinstance(raw_row, Mapping):
            continue
        finalizer = raw_row.get("finalizer")
        if not isinstance(finalizer, Mapping):
            continue
        for entry in finalizer.get("entries") or ():
            if (
                isinstance(entry, Mapping)
                and entry.get("kind") == "canonical-text-fallback"
            ):
                cue_index = entry.get("cue_index")
                evidence = entry.get("evidence")
                if (
                    isinstance(cue_index, bool)
                    or not isinstance(cue_index, int)
                    or not isinstance(evidence, Mapping)
                    or not isinstance(evidence.get("reason"), str)
                ):
                    continue
                expected.append(
                    (
                        str(row_id),
                        cue_index,
                        evidence.get("reason"),
                    )
                )
    actual: list[tuple[str, int, Any]] = []
    for index, raw in enumerate(checks):
        path = f"canonical_fallback_rechecks[{index}]"
        check = _mapping(raw, path, errors)
        if check is None:
            continue
        _closed(check, _RECHECK_KEYS, path, errors)
        cue_index = check.get("cue_index")
        if isinstance(cue_index, bool) or not isinstance(cue_index, int):
            errors.append(f"{path}.cue_index: expected integer")
            continue
        row_id = check.get("row")
        reason = check.get("reason")
        if not isinstance(row_id, str) or row_id not in final_rows:
            errors.append(f"{path}.row: expected materialized finalizer row")
            continue
        if not isinstance(reason, str):
            errors.append(f"{path}.reason: expected string")
            continue
        footprint = check.get("with_owned_footprint")
        footprint_reason = check.get("with_owned_footprint_reason")
        if not isinstance(footprint, str):
            errors.append(f"{path}.with_owned_footprint: expected source string")
        if footprint_reason is not None and not isinstance(footprint_reason, str):
            errors.append(
                f"{path}.with_owned_footprint_reason: expected string or null"
            )
        actual.append((row_id, cue_index, reason))
    if sorted(expected) != sorted(actual):
        errors.append(
            "canonical_fallback_rechecks: canonical fallback entries and independent rechecks do not match"
        )


# Section 9: authority roots, events and lineage against the finalizer rows.
def _authorities_block(
    artifact: Mapping[str, Any], final_rows: Mapping[str, Any] | None, errors: list[str]
) -> None:
    authorities = _mapping(artifact.get("authorities"), "authorities", errors)
    if authorities is not None:
        _closed(authorities, _AUTHORITY_KEYS, "authorities", errors)
        expected = _mapping(authorities.get("expected"), "authorities.expected", errors)
        events = _list(authorities.get("events"), "authorities.events", errors)
        lineage = _list(authorities.get("lineage"), "authorities.lineage", errors)
        violations = _list(
            authorities.get("violations"), "authorities.violations", errors
        )
        if violations:
            errors.append("authorities.violations: expected empty list")
        if expected is not None and final_rows is not None:
            wanted = {f"{LANE_FINALIZER}/{row_id}" for row_id in final_rows}
            expected_rows = [row_id for row_id in expected if isinstance(row_id, str)]
            if len(expected_rows) != len(expected):
                errors.append("authorities.expected: row ids must be strings")
            if set(expected) != wanted:
                errors.append(
                    "authorities.expected: does not name every finalizer row exactly once"
                )
            for row_id, authority_kind in expected.items():
                wanted_kind = (
                    "v1-capture"
                    if row_id == f"{LANE_FINALIZER}/v1"
                    else "optimizer-selection"
                )
                if authority_kind != wanted_kind:
                    errors.append(
                        f"authorities.expected.{row_id}: expected {wanted_kind!r}"
                    )
            if events is not None and len(events) != len(expected):
                errors.append(
                    "authorities.events: cardinality disagrees with expected roots"
                )
            if lineage is not None and len(lineage) != len(expected):
                errors.append(
                    "authorities.lineage: cardinality disagrees with expected roots"
                )
            if events is not None:
                event_rows: list[str] = []
                event_keys = frozenset(
                    {
                        "authority_id",
                        "authority_kind",
                        "call_id",
                        "evaluation_id",
                        "input_kind",
                        "input_seed_id",
                        "parent_finalize_call_id",
                        "row_id",
                    }
                )
                for index, raw in enumerate(events):
                    path = f"authorities.events[{index}]"
                    event = _mapping(raw, path, errors)
                    if event is None:
                        continue
                    _closed(event, event_keys, path, errors)
                    row_id = event.get("row_id")
                    if not isinstance(row_id, str):
                        errors.append(f"{path}.row_id: expected string")
                        continue
                    event_rows.append(row_id)
                    if event.get("authority_kind") != expected.get(row_id):
                        errors.append(
                            f"{path}: authority kind disagrees with root matrix"
                        )
                    if event.get("input_kind") != "phase1":
                        errors.append(f"{path}.input_kind: expected 'phase1'")
                    if event.get("parent_finalize_call_id") is not None:
                        errors.append(
                            f"{path}: finalizer root is parented by a finalize call"
                        )
                    short_row_id = row_id.removeprefix(f"{LANE_FINALIZER}/")
                    final_row = final_rows.get(short_row_id)
                    verification = (
                        final_row.get("verification")
                        if isinstance(final_row, Mapping)
                        else None
                    )
                    if isinstance(verification, Mapping):
                        if event.get("call_id") != verification.get("authority_id"):
                            errors.append(
                                f"{path}.call_id: disagrees with {row_id} verification"
                            )
                        if event.get("authority_kind") != verification.get(
                            "authority_kind"
                        ):
                            errors.append(
                                f"{path}.authority_kind: disagrees with {row_id} "
                                "verification"
                            )
                        seed_payload = verification.get("seed_payload")
                        if isinstance(seed_payload, Mapping):
                            if seed_payload.get("row_id") != row_id:
                                errors.append(
                                    f"{path}.row_id: disagrees with sealed seed row"
                                )
                            if seed_payload.get("evaluation_id") != event.get(
                                "evaluation_id"
                            ):
                                errors.append(
                                    f"{path}.evaluation_id: disagrees with sealed seed"
                                )
                if sorted(event_rows) != sorted(expected_rows):
                    errors.append("authorities.events: row roots are not one-to-one")
            if lineage is not None:
                lineage_rows: list[str] = []
                for index, raw in enumerate(lineage):
                    path = f"authorities.lineage[{index}]"
                    if not isinstance(raw, list) or len(raw) != 6:
                        errors.append(f"{path}: expected six-field lineage record")
                        continue
                    if not isinstance(raw[1], str):
                        errors.append(f"{path}[1]: expected row id string")
                        continue
                    lineage_rows.append(raw[1])
                    if raw[4] != "phase1" or raw[5] is not None:
                        errors.append(
                            f"{path}: lineage is not an unparented phase-1 root"
                        )
                if sorted(lineage_rows) != sorted(expected_rows):
                    errors.append("authorities.lineage: row roots are not one-to-one")


# Section 10: preview fidelity counters and selected-row coverage.
def _preview_fidelity_block(
    artifact: Mapping[str, Any], final_rows: Mapping[str, Any] | None, errors: list[str]
) -> None:
    preview = _mapping(artifact.get("preview_fidelity"), "preview_fidelity", errors)
    if preview is not None:
        _closed(preview, _PREVIEW_KEYS, "preview_fidelity", errors)
        scored = _nonnegative_int(
            preview.get("scored_edges"), "preview_fidelity.scored_edges", errors
        )
        checked = _nonnegative_int(
            preview.get("checked_edges"), "preview_fidelity.checked_edges", errors
        )
        uncheckable = _nonnegative_int(
            preview.get("uncheckable_edges"),
            "preview_fidelity.uncheckable_edges",
            errors,
        )
        mismatches = _list(
            preview.get("mismatches"), "preview_fidelity.mismatches", errors
        )
        selected = _mapping(
            preview.get("selected_rows"), "preview_fidelity.selected_rows", errors
        )
        if scored is not None and (scored == 0 or checked != scored):
            errors.append(
                "preview_fidelity: scored/checked edge coverage is incomplete"
            )
        if uncheckable:
            errors.append("preview_fidelity.uncheckable_edges: expected zero")
        if mismatches:
            errors.append("preview_fidelity.mismatches: expected empty list")
        if selected is not None:
            wanted = {"v2", "v2-speaker-off"}
            if final_rows is not None and "refiner-off" in final_rows:
                wanted.add("refiner-off")
            if set(selected) != wanted:
                errors.append("preview_fidelity.selected_rows: row set is incomplete")
            for row_id, raw in selected.items():
                row = _mapping(raw, f"preview_fidelity.selected_rows.{row_id}", errors)
                if row is None:
                    continue
                _closed(
                    row,
                    _PREVIEW_ROW_KEYS,
                    f"preview_fidelity.selected_rows.{row_id}",
                    errors,
                )
                if row.get("edge_count") != row.get("cue_count") or row.get(
                    "mismatches"
                ):
                    errors.append(
                        f"preview_fidelity.selected_rows.{row_id}: incomplete or mismatched"
                    )


# Section 11: speaker evidence measurements and projection.
def _speaker_evidence_block(
    artifact: Mapping[str, Any],
    final_rows: Mapping[str, Any] | None,
    unit_count: int | None,
    errors: list[str],
) -> None:
    speaker = _mapping(artifact.get("speaker_evidence"), "speaker_evidence", errors)
    if speaker is not None:
        _closed(speaker, _SPEAKER_KEYS, "speaker_evidence", errors)
        _speaker_measurement(
            speaker.get("measurement"), "speaker_evidence.measurement", errors
        )
        _speaker_measurement(
            speaker.get("off_row_measurement"),
            "speaker_evidence.off_row_measurement",
            errors,
        )
        if speaker.get("measurement_refusal") is not None:
            errors.append("speaker_evidence.measurement_refusal: expected null")
        projection = _mapping(
            speaker.get("projection"), "speaker_evidence.projection", errors
        )
        if projection is not None:
            _closed(
                projection,
                frozenset(
                    {
                        "cue_count",
                        "named_multi_cues_unannotated",
                        "range_count",
                        "status",
                    }
                ),
                "speaker_evidence.projection",
                errors,
            )
            if projection.get("status") != "verified":
                errors.append("speaker_evidence.projection.status: expected 'verified'")
        if final_rows is not None:
            off = final_rows.get("v2-speaker-off")
            if isinstance(off, Mapping) and off.get(
                "speaker_measurement"
            ) != speaker.get("off_row_measurement"):
                errors.append(
                    "speaker_evidence.off_row_measurement: disagrees with row copy"
                )
        if unit_count is not None and speaker.get("refined_unit_count") != unit_count:
            errors.append(
                "speaker_evidence.refined_unit_count: disagrees with coverage"
            )


# Section 12: sub-unit split evidence; the mapping is returned for section 14.
def _subunit_split_block(
    artifact: Mapping[str, Any], unit_count: int | None, errors: list[str]
) -> Mapping[str, Any] | None:
    split = _mapping(artifact.get("subunit_split"), "subunit_split", errors)
    if split is not None:
        _closed(split, _SUBUNIT_KEYS, "subunit_split", errors)
        evidence = _mapping(split.get("evidence"), "subunit_split.evidence", errors)
        if evidence is not None:
            _closed(evidence, _SUBUNIT_EVIDENCE_KEYS, "subunit_split.evidence", errors)
            for key in _SUBUNIT_EVIDENCE_KEYS:
                _nonnegative_int(
                    evidence.get(key), f"subunit_split.evidence.{key}", errors
                )
        origin = _list(split.get("origin"), "subunit_split.origin", errors)
        if origin is not None and unit_count is not None and len(origin) != unit_count:
            errors.append(
                "subunit_split.origin: cardinality disagrees with coverage.unit_count"
            )
        _nonnegative_int(split.get("minted"), "subunit_split.minted", errors)
        _nonnegative_int(
            split.get("refined_parent_count"),
            "subunit_split.refined_parent_count",
            errors,
        )
        _list(split.get("degraded"), "subunit_split.degraded", errors)
    return split


# Section 13: margin summary and per-interval margin evidence.
def _intervals_block(
    artifact: Mapping[str, Any], intervals: list[Any] | None, errors: list[str]
) -> None:
    margin = artifact.get("margin_summary")
    margin_count = (
        _margin_summary_block(margin, "margin_summary", errors)
        if margin is not None
        else None
    )
    selected_cut_count = 0
    interval_margin_count = 0
    if intervals is not None:
        for index, raw in enumerate(intervals):
            path = f"intervals[{index}]"
            interval = _mapping(raw, path, errors)
            if interval is None:
                continue
            forbidden = {"low_margin", "margin", "runner_up_total"} & set(interval)
            if forbidden:
                errors.append(
                    f"{path}: legacy margin keys remain: {', '.join(sorted(forbidden))}"
                )
            if "margin_summary" not in interval:
                errors.append(f"{path}: missing margin_summary")
            selected = interval.get("policy_selected")
            if isinstance(selected, Mapping) and isinstance(selected.get("cuts"), list):
                selected_cut_count += len(selected["cuts"])
            interval_margin = interval.get("margin_summary")
            if interval_margin is not None:
                count = _margin_summary_block(
                    interval_margin, f"{path}.margin_summary", errors
                )
                if count is not None:
                    interval_margin_count += count
            elif (
                isinstance(selected, Mapping)
                and isinstance(selected.get("cuts"), list)
                and selected["cuts"]
            ):
                errors.append(f"{path}.margin_summary: absent despite selected cuts")
    if selected_cut_count == 0 and margin is not None:
        errors.append("margin_summary: expected null when there are no selected cuts")
    if selected_cut_count > 0 and not isinstance(margin, Mapping):
        errors.append("margin_summary: absent despite selected cuts")
    elif margin_count is not None and margin_count != interval_margin_count:
        errors.append("margin_summary.count: disagrees with interval margin evidence")


# Section 13b: diff classification and trigger agreement.
def _diff_classification_block(artifact: Mapping[str, Any], errors: list[str]) -> None:
    diff = _mapping(artifact.get("diff_classification"), "diff_classification", errors)
    if diff is not None:
        _closed(diff, _DIFF_KEYS, "diff_classification", errors)
        changed = _list(
            diff.get("changed_fields"), "diff_classification.changed_fields", errors
        )
        independent = _string_list(
            diff.get("independent_fired"),
            "diff_classification.independent_fired",
            errors,
        )
        producer = _string_list(
            diff.get("producer_fired"), "diff_classification.producer_fired", errors
        )
        mismatches = _string_list(
            diff.get("trigger_mismatches"),
            "diff_classification.trigger_mismatches",
            errors,
        )
        _nonnegative_int(
            diff.get("relation_failures"),
            "diff_classification.relation_failures",
            errors,
        )
        _nonnegative_int(
            diff.get("unclassified_field_diff"),
            "diff_classification.unclassified_field_diff",
            errors,
        )
        if not isinstance(diff.get("alignment_error"), bool):
            errors.append("diff_classification.alignment_error: expected boolean")
        if changed is not None:
            for index, raw in enumerate(changed):
                row = _mapping(
                    raw, f"diff_classification.changed_fields[{index}]", errors
                )
                if row is not None:
                    _closed(
                        row,
                        _DIFF_ROW_KEYS,
                        f"diff_classification.changed_fields[{index}]",
                        errors,
                    )
                    _string_list(
                        row.get("trigger_ids"),
                        f"diff_classification.changed_fields[{index}].trigger_ids",
                        errors,
                    )
        if independent is not None and producer is not None and mismatches is not None:
            if sorted(set(independent) ^ set(producer)) != sorted(mismatches):
                errors.append(
                    "diff_classification.trigger_mismatches: not the symmetric difference"
                )


# Section 13c: delta registry against the frozen registry (lazy import).
def _delta_registry_block(artifact: Mapping[str, Any], errors: list[str]) -> None:
    registry = _list(artifact.get("delta_registry"), "delta_registry", errors)
    if registry is not None:
        from .policy_delta import delta_registry_data

        if registry != delta_registry_data():
            errors.append("delta_registry: differs from the frozen registry")


# Section 14: refiner comparison against the sub-unit split and finalizer rows.
def _refiner_comparison_block(
    artifact: Mapping[str, Any],
    split: Mapping[str, Any] | None,
    final_rows: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    comparison = _mapping(
        artifact.get("refiner_comparison"), "refiner_comparison", errors
    )
    if comparison is not None:
        status = comparison.get("status")
        refined = split.get("refined_parent_count") if split is not None else None
        if comparison.get("refined_parent_count") != refined:
            errors.append(
                "refiner_comparison.refined_parent_count: disagrees with subunit_split"
            )
        if status == "tracked-identity":
            _closed(
                comparison,
                frozenset({"byte_identical", "refined_parent_count", "status"}),
                "refiner_comparison",
                errors,
            )
            if comparison.get("byte_identical") is not True or refined != 0:
                errors.append(
                    "refiner_comparison: tracked identity is not byte-identical"
                )
            if final_rows is not None and "refiner-off" in final_rows:
                errors.append(
                    "refiner_comparison: identity row must not materialize refiner-off"
                )
        elif status == "refined-counterfactual":
            expected_keys = frozenset(
                {
                    "byte_identical",
                    "coarse_caused_intervals",
                    "coarse_caused_unit_ranges",
                    "diffs_confined_to_coarse_caused",
                    "external_parent_cut_diff",
                    "fallback_intervals",
                    "internal_refinement_cut_parents",
                    "materialized",
                    "off_partition",
                    "on_parent_edge_partition",
                    "parent_v1_projection",
                    "refined_parent_count",
                    "status",
                }
            )
            _closed(comparison, expected_keys, "refiner_comparison", errors)
            if comparison.get("materialized") is not True:
                errors.append("refiner_comparison.materialized: schema 2 requires true")
            if final_rows is not None and "refiner-off" not in final_rows:
                errors.append(
                    "refiner_comparison: materialized counterfactual row is absent"
                )
        else:
            errors.append(f"refiner_comparison.status: unknown value {status!r}")


# Section 15: v1 projection.
def _v1_projection_block(artifact: Mapping[str, Any], errors: list[str]) -> None:
    projection = _mapping(artifact.get("v1_projection"), "v1_projection", errors)
    if projection is not None:
        _closed(
            projection,
            frozenset({"cut_count", "mode", "unprojected"}),
            "v1_projection",
            errors,
        )
        if projection.get("unprojected") is not False:
            errors.append("v1_projection.unprojected: expected false")


# Section 16: residual closed-shape and identity checks (lazy import).
def _residual_block(artifact: Mapping[str, Any], errors: list[str]) -> None:
    for key in (
        "finalizer",
        "influence_cell",
        "pause_knees",
        "profile",
        "providers",
        "vad_state",
    ):
        _mapping(artifact.get(key), key, errors)
    for key in (
        "invalid_finalizer_rows",
        "policy_deltas",
        "production_degraded",
        "shadow_degraded",
    ):
        _list(artifact.get(key), key, errors)
    if artifact.get("engine_v2") != "boundary-optimizer-v2":
        errors.append("engine_v2: unexpected engine identity")
    if artifact.get("policy_name") != "experimental_policy_2":
        errors.append("policy_name: live schema 2 requires experimental_policy_2")
    if artifact.get("policy_version") != 2:
        errors.append("policy_version: live schema 2 requires version 2")
    if not isinstance(artifact.get("language"), str):
        errors.append("language: expected string")
    from .boundary_v2 import SPEAKER_POLICY_DELTAS

    if artifact.get("policy_deltas") != list(SPEAKER_POLICY_DELTAS):
        errors.append("policy_deltas: differs from the live policy-2 declaration")


# Section 2a: the coverage block; returns the declared unit count.
def _coverage_block(
    coverage: Mapping[str, Any] | None, errors: list[str]
) -> int | None:
    unit_count: int | None = None
    if coverage is not None:
        _closed(coverage, _COVERAGE_KEYS, "coverage", errors)
        unit_count = _nonnegative_int(
            coverage.get("unit_count"), "coverage.unit_count", errors
        )
        for key in (
            "coarse_caused_intervals",
            "coarse_granularity_intervals",
            "fallback_intervals",
            "named_multi_cues_unannotated",
            "optimized_intervals",
        ):
            _nonnegative_int(coverage.get(key), f"coverage.{key}", errors)
        if not isinstance(coverage.get("dual_form_unmeasured"), bool):
            errors.append("coverage.dual_form_unmeasured: expected boolean")
        if coverage.get("fallback_intervals") != 0:
            errors.append("coverage.fallback_intervals: schema 2 requires zero")
        if coverage.get("optimized_unit_ratio") != 1.0:
            errors.append("coverage.optimized_unit_ratio: schema 2 requires 1.0")
        if coverage.get("fallback_ranges_overlap") is not False:
            errors.append("coverage.fallback_ranges_overlap: expected false")
        if coverage.get("raw_conservation_trustworthy") is not True:
            errors.append("coverage.raw_conservation_trustworthy: expected true")
        if coverage.get("v1_unprojected") is not False:
            errors.append("coverage.v1_unprojected: expected false")
        _list(
            coverage.get("fallback_unit_ranges"),
            "coverage.fallback_unit_ranges",
            errors,
        )
    return unit_count


# Section 2b: the totals block against coverage and the raw serialized intervals.
def _totals_block(
    totals: Mapping[str, Any] | None,
    coverage: Mapping[str, Any] | None,
    artifact: Mapping[str, Any],
    errors: list[str],
) -> None:
    if totals is not None:
        _closed(totals, _TOTAL_KEYS, "totals", errors)
        for key in _TOTAL_KEYS - {"optimized_unit_ratio", "waivers"}:
            _nonnegative_int(totals.get(key), f"totals.{key}", errors)
        _list(totals.get("waivers"), "totals.waivers", errors)
        if coverage is not None:
            for key in (
                "coarse_caused_intervals",
                "coarse_granularity_intervals",
                "fallback_intervals",
                "optimized_intervals",
                "optimized_unit_ratio",
                "unit_count",
            ):
                if totals.get(key) != coverage.get(key):
                    errors.append(f"totals.{key}: disagrees with coverage.{key}")
        intervals = artifact.get("intervals")
        if isinstance(intervals, list) and totals.get("interval_count") != len(
            intervals
        ):
            errors.append("totals.interval_count: disagrees with serialized intervals")


# Sections 3 and 4: source units against the declared count, and the display profile.
def _units_and_profile(
    artifact: Mapping[str, Any], unit_count: int | None, errors: list[str]
) -> tuple[list[Any] | None, Any | None]:
    source_units = _source_units(artifact.get("units"), "units", errors)
    if (
        source_units is not None
        and unit_count is not None
        and len(source_units) != unit_count
    ):
        errors.append(
            f"units: expected coverage.unit_count {unit_count}, got {len(source_units)}"
        )
    profile_value = _display_profile(artifact.get("profile"), "profile", errors)
    return source_units, profile_value


# Section 5b: the validator summary fields (the per-stage blocks come later).
def _validator_summary_block(
    validator: Mapping[str, Any] | None, errors: list[str]
) -> None:
    if validator is not None:
        _closed(validator, _VALIDATOR_KEYS, "validator", errors)
        if not isinstance(validator.get("interval_document_agree"), bool):
            errors.append("validator.interval_document_agree: expected boolean")
        _nonnegative_int(
            validator.get("interval_hard_violations"),
            "validator.interval_hard_violations",
            errors,
        )
        if not isinstance(validator.get("raw_duplicate_v1_cues"), bool):
            errors.append("validator.raw_duplicate_v1_cues: expected boolean")


# Section 7: validator stage blocks, then their agreement with the rows.
def _validator_alias_block(
    validator: Mapping[str, Any] | None,
    raw_row: Mapping[str, Any] | None,
    core_v2: Mapping[str, Any] | None,
    legacy_v2: Mapping[str, Any] | None,
    final_v2: Mapping[str, Any] | None,
    unit_count: int | None,
    errors: list[str],
) -> None:
    if validator is not None:
        for key, stage, row in (
            ("raw", "raw", raw_row),
            ("core", "core", core_v2),
            ("legacy_overlay", "legacy-overlay", legacy_v2),
            ("finalizer", "finalizer", final_v2),
        ):
            _validator_block(
                validator.get(key),
                f"validator.{key}",
                errors,
                expected_stage=stage,
                permitted_origins=frozenset({"v2"}),
                expected_cue_count=_declared_cue_count(row),
                expected_unit_count=unit_count,
            )
        for key, row, row_path in (
            ("raw", raw_row, "raw"),
            ("core", core_v2, f"lanes.{LANE_CORE}.v2"),
            ("legacy_overlay", legacy_v2, f"lanes.{LANE_LEGACY}.v2"),
            ("finalizer", final_v2, f"lanes.{LANE_FINALIZER}.rows.v2"),
        ):
            if row is not None and validator.get(key) != row.get("validator"):
                errors.append(f"validator.{key}: disagrees with {row_path}.validator")


# Section 8: the finalizer alias and the invalid finalizer row list.
def _finalizer_alias_block(
    artifact: Mapping[str, Any], final_rows: Mapping[str, Any] | None, errors: list[str]
) -> None:
    if final_rows is not None:
        v2 = final_rows.get("v2")
        if isinstance(v2, Mapping):
            if artifact.get("finalizer") != v2.get("finalizer"):
                errors.append("finalizer: disagrees with delivery_finalizer/v2")
        invalid = [
            row_id
            for row_id, row in final_rows.items()
            if not isinstance(row, Mapping)
            or not isinstance(row.get("finalizer"), Mapping)
            or row["finalizer"].get("valid") is not True
        ]
        if artifact.get("invalid_finalizer_rows") != invalid:
            errors.append("invalid_finalizer_rows: disagrees with materialized rows")


def validate_shadow_v2_payload(
    artifact: Mapping[str, Any], *, require_version: bool = True
) -> list[str]:
    """Return every structural reason ``artifact`` cannot claim live schema 2."""
    errors: list[str] = []
    _closed(artifact, TOP_LEVEL_KEYS, "artifact", errors)
    if require_version and artifact.get("schema_version") != LIVE_SHADOW_SCHEMA_VERSION:
        errors.append(
            f"schema_version: expected {LIVE_SHADOW_SCHEMA_VERSION}, got {artifact.get('schema_version')!r}"
        )
    if artifact.get("kind") != "segmentation-shadow":
        errors.append("kind: expected 'segmentation-shadow'")

    coverage = _mapping(artifact.get("coverage"), "coverage", errors)
    totals = _mapping(artifact.get("totals"), "totals", errors)
    unit_count = _coverage_block(coverage, errors)
    _totals_block(totals, coverage, artifact, errors)
    source_units, profile_value = _units_and_profile(artifact, unit_count, errors)

    intervals = _list(artifact.get("intervals"), "intervals", errors)
    validator = _mapping(artifact.get("validator"), "validator", errors)
    _validator_summary_block(validator, errors)

    raw_row = _stream_row(
        artifact.get("raw"),
        unit_count=unit_count,
        units=source_units,
        profile=profile_value,
        path="raw",
        errors=errors,
        validator_stage="raw",
        validator_origins=frozenset({"v2"}),
    )
    lanes, final_rows = _lanes(
        artifact.get("lanes"),
        unit_count=unit_count,
        units=source_units,
        profile=profile_value,
        errors=errors,
    )
    core_v2 = _lane_row(lanes, LANE_CORE, "v2")
    legacy_v2 = _lane_row(lanes, LANE_LEGACY, "v2")
    final_v2 = _lane_row(lanes, LANE_FINALIZER, "v2")
    _validator_alias_block(
        validator, raw_row, core_v2, legacy_v2, final_v2, unit_count, errors
    )
    _finalizer_alias_block(artifact, final_rows, errors)

    _authorities_block(artifact, final_rows, errors)
    _preview_fidelity_block(artifact, final_rows, errors)
    _speaker_evidence_block(artifact, final_rows, unit_count, errors)
    split = _subunit_split_block(artifact, unit_count, errors)
    _intervals_block(artifact, intervals, errors)
    _diff_classification_block(artifact, errors)
    _delta_registry_block(artifact, errors)
    _refiner_comparison_block(artifact, split, final_rows, errors)
    _v1_projection_block(artifact, errors)
    _residual_block(artifact, errors)
    _fallback_rechecks(artifact, final_rows, errors)
    return errors
