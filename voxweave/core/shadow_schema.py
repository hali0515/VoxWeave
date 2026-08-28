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

LIVE_SHADOW_SCHEMA_VERSION = 2

LANE_CORE = "core_partition_pre_overlay"
LANE_LEGACY = "delivery_v1_legacy"
LANE_FINALIZER = "delivery_finalizer"
LANE_DISPLAY = "legacy_display"

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
_CHECK_KEYS = frozenset(
    {
        "cue_count",
        "exit_driving",
        "origin",
        "stage",
        "unit_count",
        "unwaived",
        "violations",
        "waivers",
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
_FINALIZER_KEYS = frozenset(
    {
        "deltas_fired",
        "entries",
        "max_start_movement_s",
        "max_sweeps_observed",
        "movement",
        "movement_distribution",
        "refusals",
        "schedule_canonicality",
        "stability_errors",
        "terminal",
        "trace",
        "trace_errors",
        "valid",
        "waivers",
    }
)
_FINALIZER_ROW_KEYS = _STREAM_KEYS | {"finalizer", "verification"}
_UNIT_KEYS = frozenset({"confidence", "end", "id", "provenance", "start", "surface"})
_VERIFICATION_KEYS = frozenset(
    {
        "authority_id",
        "authority_kind",
        "evidence",
        "policy",
        "seed_digest",
        "seed_payload",
    }
)
_SEED_PAYLOAD_KEYS = frozenset({"cues", "evaluation_id", "profile", "row_id"})
_EVIDENCE_KEYS = frozenset({"shots", "sing_spans"})
_POLICY_KEYS = frozenset({"grid", "min_gap", "overlap_policy"})
_TRACE_KEYS = frozenset(
    {"cycle", "legs", "schedule_canonicality", "sweeps", "terminal"}
)
_TRACE_LEG_KEYS = frozenset(
    {"cue_index", "from", "reads", "rule_id", "slot", "sweep", "target", "to"}
)
_BOUNDARY_KEYS = frozenset({"cue_index", "side"})
_READ_KEYS = frozenset({"boundary", "value"})
_CYCLE_KEYS = frozenset({"adopted", "members", "per_boundary_values"})
_CYCLE_VALUES_KEYS = frozenset({"boundary", "values"})
_MOVEMENT_KEYS = frozenset({"boundary", "delivered", "delta", "phase1"})
_MOVEMENT_DISTRIBUTION_KEYS = frozenset({"start", "end"})
_MOVEMENT_SUMMARY_KEYS = frozenset({"count", "max", "p50", "p90"})
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
_MEASUREMENT_KEYS = frozenset(
    {
        "buckets",
        "expressed_rate",
        "expressible_hit_rate",
        "matches",
        "raw_in_speech_turn_changes",
        "speaker_attributable_expressed_cuts",
    }
)
_BUCKET_KEYS = frozenset(
    {
        "expressed",
        "policy_filtered",
        "survived_expressible_but_missed",
        "unattributed_loss",
        "unexpressible",
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
_MARGIN_KEYS = frozenset({"count", "exact_ties", "min", "p05", "p50"})
_RECHECK_KEYS = frozenset(
    {
        "cue_index",
        "reason",
        "row",
        "with_owned_footprint",
        "with_owned_footprint_reason",
    }
)


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected mapping, got {type(value).__name__}")
        return None
    return value


def _list(value: Any, path: str, errors: list[str]) -> list[Any] | None:
    if not isinstance(value, list):
        errors.append(f"{path}: expected list, got {type(value).__name__}")
        return None
    return value


def _string_list(value: Any, path: str, errors: list[str]) -> list[str] | None:
    rows = _list(value, path, errors)
    if rows is None:
        return None
    if any(not isinstance(row, str) for row in rows):
        errors.append(f"{path}: expected strings")
        return None
    return rows


def _closed(
    block: Mapping[str, Any], expected: frozenset[str], path: str, errors: list[str]
) -> None:
    actual = set(block)
    missing = sorted(expected - actual)
    extra = sorted(
        actual - expected, key=lambda item: (type(item).__name__, repr(item))
    )
    if missing:
        errors.append(f"{path}: missing keys {', '.join(missing)}")
    if extra:
        errors.append(f"{path}: unknown keys {', '.join(str(item) for item in extra)}")


def _nonnegative_int(value: Any, path: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{path}: expected non-negative integer")
        return None
    return value


def _finite_number(value: Any, path: str, errors: list[str]) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        errors.append(f"{path}: expected finite number")
        return None
    return float(value)


def _source_units(value: Any, path: str, errors: list[str]) -> list[Any] | None:
    """Decode the closed unit authority used by independent row validation."""
    from .segdoc import SourceUnit

    rows = _list(value, path, errors)
    if rows is None:
        return None
    units: list[SourceUnit] = []
    for index, raw in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _mapping(raw, row_path, errors)
        if row is None:
            continue
        _closed(row, _UNIT_KEYS, row_path, errors)
        unit_id = row.get("id")
        surface = row.get("surface")
        provenance = row.get("provenance")
        if unit_id != f"u{index}":
            errors.append(f"{row_path}.id: expected positional id 'u{index}'")
        if not isinstance(surface, str):
            errors.append(f"{row_path}.surface: expected string")
        if not isinstance(provenance, str):
            errors.append(f"{row_path}.provenance: expected string")
        bounds: dict[str, float | None] = {}
        for key in ("start", "end", "confidence"):
            raw_value = row.get(key)
            if raw_value is None:
                bounds[key] = None
            else:
                bounds[key] = _finite_number(raw_value, f"{row_path}.{key}", errors)
        if (
            unit_id == f"u{index}"
            and isinstance(surface, str)
            and isinstance(provenance, str)
            and all(
                row.get(key) is None or bounds[key] is not None
                for key in ("start", "end", "confidence")
            )
        ):
            units.append(
                SourceUnit(
                    cast(str, unit_id),
                    surface,
                    bounds["start"],
                    bounds["end"],
                    provenance,
                    bounds["confidence"],
                )
            )
    return units if len(units) == len(rows) else None


def _display_profile(value: Any, path: str, errors: list[str]) -> Any | None:
    from .segdoc import THRESHOLD_KEYS, DisplayProfile

    block = _mapping(value, path, errors)
    if block is None:
        return None
    expected = frozenset({"language", "max_line_length", "max_lines", *THRESHOLD_KEYS})
    _closed(block, expected, path, errors)
    language = block.get("language")
    if not isinstance(language, str):
        errors.append(f"{path}.language: expected string")
    integer_values: dict[str, int] = {}
    for key in ("max_line_length", "max_lines"):
        raw = block.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            errors.append(f"{path}.{key}: expected positive integer")
        else:
            integer_values[key] = raw
    numeric_values: dict[str, float] = {}
    for key in THRESHOLD_KEYS:
        number = _finite_number(block.get(key), f"{path}.{key}", errors)
        if number is not None:
            numeric_values[key] = number
    if (
        isinstance(language, str)
        and len(integer_values) == 2
        and len(numeric_values) == len(THRESHOLD_KEYS)
    ):
        return DisplayProfile(
            language=language,
            max_line_length=integer_values["max_line_length"],
            max_lines=integer_values["max_lines"],
            **numeric_values,
        )
    return None


def _waiver_rows(value: Any, path: str, errors: list[str]) -> list[Any] | None:
    from .partition_check import Waiver

    rows = _list(value, path, errors)
    if rows is None:
        return None
    out: list[Waiver] = []
    expected = frozenset({"cap", "cue_index", "detail", "kind", "span", "unit_ids"})
    for index, raw in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _mapping(raw, row_path, errors)
        if row is None:
            continue
        _closed(row, expected, row_path, errors)
        cue_index = _nonnegative_int(
            row.get("cue_index"), f"{row_path}.cue_index", errors
        )
        cap = _finite_number(row.get("cap"), f"{row_path}.cap", errors)
        kind, detail = row.get("kind"), row.get("detail")
        if not isinstance(kind, str):
            errors.append(f"{row_path}.kind: expected string")
        if not isinstance(detail, str):
            errors.append(f"{row_path}.detail: expected string")
        span = _list(row.get("span"), f"{row_path}.span", errors)
        decoded_span: list[float | None] = []
        if span is not None:
            if len(span) != 2:
                errors.append(f"{row_path}.span: expected two bounds")
            else:
                for slot, item in enumerate(span):
                    decoded_span.append(
                        None
                        if item is None
                        else _finite_number(item, f"{row_path}.span[{slot}]", errors)
                    )
        unit_ids = _list(row.get("unit_ids"), f"{row_path}.unit_ids", errors)
        decoded_ids: list[int] = []
        if unit_ids is not None:
            for slot, item in enumerate(unit_ids):
                decoded = _nonnegative_int(item, f"{row_path}.unit_ids[{slot}]", errors)
                if decoded is not None:
                    decoded_ids.append(decoded)
        if (
            cue_index is not None
            and cap is not None
            and isinstance(kind, str)
            and isinstance(detail, str)
            and len(decoded_span) == 2
            and all(item is None or isinstance(item, float) for item in decoded_span)
            and unit_ids is not None
            and len(decoded_ids) == len(unit_ids)
        ):
            out.append(
                Waiver(
                    kind,
                    cue_index,
                    tuple(decoded_ids),
                    cast("tuple[float | None, float | None]", tuple(decoded_span)),
                    cap,
                    detail,
                )
            )
    return out if len(out) == len(rows) else None


def _report_rows(value: Any, path: str, errors: list[str]) -> list[Any] | None:
    from .partition_check import ReportTag

    rows = _list(value, path, errors)
    if rows is None:
        return None
    out: list[ReportTag] = []
    for index, raw in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _mapping(raw, row_path, errors)
        if row is None:
            continue
        _closed(row, frozenset({"cue_index", "evidence", "kind"}), row_path, errors)
        cue_index = row.get("cue_index")
        if cue_index is not None and (
            isinstance(cue_index, bool) or not isinstance(cue_index, int)
        ):
            errors.append(f"{row_path}.cue_index: expected integer or null")
        kind = row.get("kind")
        evidence = _mapping(row.get("evidence"), f"{row_path}.evidence", errors)
        if not isinstance(kind, str):
            errors.append(f"{row_path}.kind: expected string")
        if (
            (cue_index is None or isinstance(cue_index, int))
            and not isinstance(cue_index, bool)
            and isinstance(kind, str)
            and evidence is not None
        ):
            out.append(ReportTag(kind, cue_index, dict(evidence)))
    return out if len(out) == len(rows) else None


def _margin_summary_block(value: Any, path: str, errors: list[str]) -> int | None:
    block = _mapping(value, path, errors)
    if block is None:
        return None
    _closed(block, _MARGIN_KEYS, path, errors)
    count = _nonnegative_int(block.get("count"), f"{path}.count", errors)
    ties = _nonnegative_int(block.get("exact_ties"), f"{path}.exact_ties", errors)
    if count is not None and ties is not None and ties > count:
        errors.append(f"{path}.exact_ties: exceeds count")
    quantiles: list[float] = []
    for key in ("min", "p05", "p50"):
        raw = block.get(key)
        if raw is None:
            if count:
                errors.append(f"{path}.{key}: absent despite non-zero count")
            continue
        number = _finite_number(raw, f"{path}.{key}", errors)
        if number is not None:
            quantiles.append(number)
    if count == 0 and any(block.get(key) is not None for key in ("min", "p05", "p50")):
        errors.append(f"{path}: zero count must carry null margin values")
    if len(quantiles) == 3 and quantiles != sorted(quantiles):
        errors.append(f"{path}: expected min <= p05 <= p50")
    return count


def _validator_block(
    value: Any,
    path: str,
    errors: list[str],
    *,
    expected_stage: str,
    permitted_origins: frozenset[str],
    expected_cue_count: int | None,
    expected_unit_count: int | None,
) -> Mapping[str, Any] | None:
    block = _mapping(value, path, errors)
    if block is None:
        return None
    _closed(block, _CHECK_KEYS, path, errors)
    cue_count = _nonnegative_int(block.get("cue_count"), f"{path}.cue_count", errors)
    unit_count = _nonnegative_int(block.get("unit_count"), f"{path}.unit_count", errors)
    for key in ("exit_driving", "unwaived"):
        _nonnegative_int(block.get(key), f"{path}.{key}", errors)
    _list(block.get("violations"), f"{path}.violations", errors)
    _list(block.get("waivers"), f"{path}.waivers", errors)
    origin = block.get("origin")
    if not isinstance(origin, str):
        errors.append(f"{path}.origin: expected string")
    elif origin not in permitted_origins:
        expected = ", ".join(repr(item) for item in sorted(permitted_origins))
        errors.append(f"{path}.origin: expected one of {expected}, got {origin!r}")
    stage = block.get("stage")
    if not isinstance(stage, str):
        errors.append(f"{path}.stage: expected string")
    elif stage != expected_stage:
        errors.append(f"{path}.stage: expected stage {expected_stage!r}, got {stage!r}")
    if expected_cue_count is not None and cue_count != expected_cue_count:
        errors.append(
            f"{path}.cue_count: expected row cue_count {expected_cue_count}, got {cue_count!r}"
        )
    if expected_unit_count is not None and unit_count != expected_unit_count:
        errors.append(
            f"{path}.unit_count: expected coverage.unit_count "
            f"{expected_unit_count}, got {unit_count!r}"
        )
    return block


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


def _finalizer_verification(
    value: Any,
    row: Mapping[str, Any],
    *,
    path: str,
    errors: list[str],
    expected_row_id: str | None,
    profile: Any | None,
) -> tuple[Any, ...] | None:
    """Decode and authenticate the phase-1 seed carried for trace replay."""
    from .authority import AUTHORITY_KINDS, digest_payload
    from .finalizer import Phase1Cue, _profile_payload

    block = _mapping(value, path, errors)
    if block is None:
        return None
    _closed(block, _VERIFICATION_KEYS, path, errors)
    authority_id = block.get("authority_id")
    authority_kind = block.get("authority_kind")
    digest = block.get("seed_digest")
    if not isinstance(authority_id, str) or not authority_id:
        errors.append(f"{path}.authority_id: expected non-empty string")
    if authority_kind not in AUTHORITY_KINDS:
        errors.append(f"{path}.authority_kind: outside closed authority vocabulary")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        errors.append(f"{path}.seed_digest: expected lowercase sha256")

    payload = _mapping(block.get("seed_payload"), f"{path}.seed_payload", errors)
    if payload is None:
        return None
    _closed(payload, _SEED_PAYLOAD_KEYS, f"{path}.seed_payload", errors)
    if isinstance(digest, str) and digest_payload(payload) != digest:
        errors.append(f"{path}.seed_digest: does not bind seed_payload")
    wanted_row = (
        None if expected_row_id is None else f"{LANE_FINALIZER}/{expected_row_id}"
    )
    if wanted_row is not None and payload.get("row_id") != wanted_row:
        errors.append(
            f"{path}.seed_payload.row_id: expected {wanted_row!r}, "
            f"got {payload.get('row_id')!r}"
        )
    if not isinstance(payload.get("evaluation_id"), str):
        errors.append(f"{path}.seed_payload.evaluation_id: expected string")
    if profile is not None and payload.get("profile") != _profile_payload(profile):
        errors.append(f"{path}.seed_payload.profile: disagrees with artifact profile")

    raw_cues = _list(payload.get("cues"), f"{path}.seed_payload.cues", errors)
    if raw_cues is None:
        return None
    cues: list[Phase1Cue] = []
    for position, raw in enumerate(raw_cues):
        cue_path = f"{path}.seed_payload.cues[{position}]"
        if not isinstance(raw, list) or len(raw) != 12:
            errors.append(f"{cue_path}: expected twelve-field phase-1 seed")
            continue
        index = _nonnegative_int(raw[0], f"{cue_path}[0]", errors)
        numeric: dict[int, float] = {}
        for slot in (1, 2, 3, 4):
            number = _finite_number(raw[slot], f"{cue_path}[{slot}]", errors)
            if number is not None:
                numeric[slot] = number
        optional: dict[int, float | None] = {}
        for slot in (5, 6):
            optional[slot] = (
                None
                if raw[slot] is None
                else _finite_number(raw[slot], f"{cue_path}[{slot}]", errors)
            )
        text = raw[7]
        reading = _nonnegative_int(raw[8], f"{cue_path}[8]", errors)
        lyric = raw[9]
        if not isinstance(text, str):
            errors.append(f"{cue_path}[7]: expected text string")
        if lyric is not None and not isinstance(lyric, bool):
            errors.append(f"{cue_path}[9]: expected boolean or null")
        unit_range_raw = raw[10]
        unit_range: tuple[int, int] | None = None
        if unit_range_raw is not None:
            if (
                not isinstance(unit_range_raw, list)
                or len(unit_range_raw) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in unit_range_raw
                )
            ):
                errors.append(f"{cue_path}[10]: expected two integer bounds or null")
            else:
                unit_range = (unit_range_raw[0], unit_range_raw[1])
        unit_rows = _list(raw[11], f"{cue_path}[11]", errors)
        word_data: list[dict[str, Any]] = []
        if unit_rows is not None:
            for unit_index, unit_raw in enumerate(unit_rows):
                unit_path = f"{cue_path}[11][{unit_index}]"
                if not isinstance(unit_raw, list) or len(unit_raw) != 3:
                    errors.append(f"{unit_path}: expected text/start/end triple")
                    continue
                if not isinstance(unit_raw[0], str):
                    errors.append(f"{unit_path}[0]: expected string")
                    continue
                decoded: list[float | None] = []
                for slot in (1, 2):
                    decoded.append(
                        None
                        if unit_raw[slot] is None
                        else _finite_number(
                            unit_raw[slot], f"{unit_path}[{slot}]", errors
                        )
                    )
                if all(
                    unit_raw[slot] is None or decoded[slot - 1] is not None
                    for slot in (1, 2)
                ):
                    word_data.append(
                        {
                            "text": unit_raw[0],
                            "start": decoded[0],
                            "end": decoded[1],
                        }
                    )
        if (
            index == position
            and len(numeric) == 4
            and all(raw[slot] is None or optional[slot] is not None for slot in (5, 6))
            and isinstance(text, str)
            and reading is not None
            and (lyric is None or isinstance(lyric, bool))
            and unit_rows is not None
            and len(word_data) == len(unit_rows)
        ):
            cues.append(
                Phase1Cue(
                    index=position,
                    start=numeric[1],
                    end=numeric[2],
                    seed_start=numeric[3],
                    seed_end=numeric[4],
                    speech_start=optional[5],
                    speech_end=optional[6],
                    text=text,
                    lines=(text,),
                    cell_widths=(),
                    reading_chars=reading,
                    raw_reading_chars=reading,
                    word_data=cast("Any", word_data),
                    unit_range=unit_range,
                    lyric=lyric,
                    reports=(),
                )
            )
    if len(cues) != len(raw_cues):
        return None
    if len(cues) != row.get("cue_count"):
        errors.append(
            f"{path}.seed_payload.cues: seed count disagrees with delivered row"
        )
    return tuple(cues)


def _boundary_ref(value: Any, path: str, errors: list[str]) -> Any | None:
    from .finalizer import BoundaryRef

    block = _mapping(value, path, errors)
    if block is None:
        return None
    _closed(block, _BOUNDARY_KEYS, path, errors)
    cue_index = _nonnegative_int(block.get("cue_index"), f"{path}.cue_index", errors)
    side = block.get("side")
    if side not in {"start", "end"}:
        errors.append(f"{path}.side: expected 'start' or 'end'")
    if cue_index is None or side not in {"start", "end"}:
        return None
    return BoundaryRef(cue_index, cast("Any", side))


def _stream_state(value: Any, path: str, errors: list[str]) -> Any | None:
    rows = _list(value, path, errors)
    if rows is None:
        return None
    out: list[tuple[float, float]] = []
    for index, raw in enumerate(rows):
        row_path = f"{path}[{index}]"
        if not isinstance(raw, list) or len(raw) != 2:
            errors.append(f"{row_path}: expected start/end pair")
            continue
        start = _finite_number(raw[0], f"{row_path}[0]", errors)
        end = _finite_number(raw[1], f"{row_path}[1]", errors)
        if start is not None and end is not None:
            out.append((start, end))
    return tuple(out) if len(out) == len(rows) else None


def _trace(value: Any, path: str, errors: list[str]) -> Any | None:
    from .finalizer import CycleEvidence, NeighbourRead, Trace, TraceLeg

    block = _mapping(value, path, errors)
    if block is None:
        return None
    _closed(block, _TRACE_KEYS, path, errors)
    terminal = block.get("terminal")
    if terminal not in {"budget-exhausted", "cycle-adoption", "fixed-point"}:
        errors.append(f"{path}.terminal: outside closed terminal vocabulary")
    sweeps = _nonnegative_int(block.get("sweeps"), f"{path}.sweeps", errors)
    if block.get("schedule_canonicality") != "unverified":
        errors.append(f"{path}.schedule_canonicality: expected 'unverified'")
    legs_raw = _list(block.get("legs"), f"{path}.legs", errors)
    legs: list[TraceLeg] = []
    if legs_raw is not None:
        for index, raw in enumerate(legs_raw):
            leg_path = f"{path}.legs[{index}]"
            leg = _mapping(raw, leg_path, errors)
            if leg is None:
                continue
            _closed(leg, _TRACE_LEG_KEYS, leg_path, errors)
            cue_index = _nonnegative_int(
                leg.get("cue_index"), f"{leg_path}.cue_index", errors
            )
            slot = _nonnegative_int(leg.get("slot"), f"{leg_path}.slot", errors)
            sweep = _nonnegative_int(leg.get("sweep"), f"{leg_path}.sweep", errors)
            from_value = _finite_number(leg.get("from"), f"{leg_path}.from", errors)
            to_value = _finite_number(leg.get("to"), f"{leg_path}.to", errors)
            rule_id = leg.get("rule_id")
            if not isinstance(rule_id, str):
                errors.append(f"{leg_path}.rule_id: expected string")
            target = _boundary_ref(leg.get("target"), f"{leg_path}.target", errors)
            reads_raw = _list(leg.get("reads"), f"{leg_path}.reads", errors)
            reads: list[NeighbourRead] = []
            if reads_raw is not None:
                for read_index, read_raw in enumerate(reads_raw):
                    read_path = f"{leg_path}.reads[{read_index}]"
                    read = _mapping(read_raw, read_path, errors)
                    if read is None:
                        continue
                    _closed(read, _READ_KEYS, read_path, errors)
                    boundary = _boundary_ref(
                        read.get("boundary"), f"{read_path}.boundary", errors
                    )
                    read_value = (
                        None
                        if read.get("value") is None
                        else _finite_number(
                            read.get("value"), f"{read_path}.value", errors
                        )
                    )
                    if boundary is not None and (
                        read.get("value") is None or read_value is not None
                    ):
                        reads.append(NeighbourRead(boundary, read_value))
            if (
                cue_index is not None
                and slot is not None
                and sweep is not None
                and from_value is not None
                and to_value is not None
                and isinstance(rule_id, str)
                and target is not None
                and reads_raw is not None
                and len(reads) == len(reads_raw)
            ):
                legs.append(
                    TraceLeg(
                        rule_id,
                        sweep,
                        cue_index,
                        slot,
                        target,
                        from_value,
                        to_value,
                        tuple(reads),
                    )
                )

    cycle_value = block.get("cycle")
    cycle = None
    cycle_valid = cycle_value is None
    if cycle_value is not None:
        cycle_block = _mapping(cycle_value, f"{path}.cycle", errors)
        if cycle_block is not None:
            _closed(cycle_block, _CYCLE_KEYS, f"{path}.cycle", errors)
            adopted = _stream_state(
                cycle_block.get("adopted"), f"{path}.cycle.adopted", errors
            )
            members_raw = _list(
                cycle_block.get("members"), f"{path}.cycle.members", errors
            )
            members: list[Any] = []
            if members_raw is not None:
                for index, member in enumerate(members_raw):
                    decoded = _stream_state(
                        member, f"{path}.cycle.members[{index}]", errors
                    )
                    if decoded is not None:
                        members.append(decoded)
            values_raw = _list(
                cycle_block.get("per_boundary_values"),
                f"{path}.cycle.per_boundary_values",
                errors,
            )
            values: list[tuple[Any, tuple[float, ...]]] = []
            if values_raw is not None:
                for index, raw in enumerate(values_raw):
                    values_path = f"{path}.cycle.per_boundary_values[{index}]"
                    values_block = _mapping(raw, values_path, errors)
                    if values_block is None:
                        continue
                    _closed(values_block, _CYCLE_VALUES_KEYS, values_path, errors)
                    boundary = _boundary_ref(
                        values_block.get("boundary"),
                        f"{values_path}.boundary",
                        errors,
                    )
                    numbers_raw = _list(
                        values_block.get("values"), f"{values_path}.values", errors
                    )
                    numbers: list[float] = []
                    if numbers_raw is not None:
                        for number_index, number_raw in enumerate(numbers_raw):
                            number = _finite_number(
                                number_raw,
                                f"{values_path}.values[{number_index}]",
                                errors,
                            )
                            if number is not None:
                                numbers.append(number)
                    if (
                        boundary is not None
                        and numbers_raw is not None
                        and len(numbers) == len(numbers_raw)
                    ):
                        values.append((boundary, tuple(numbers)))
            if (
                adopted is not None
                and members_raw is not None
                and len(members) == len(members_raw)
                and values_raw is not None
                and len(values) == len(values_raw)
            ):
                cycle = CycleEvidence(tuple(members), tuple(values), adopted)
                cycle_valid = True
    if (
        terminal in {"budget-exhausted", "cycle-adoption", "fixed-point"}
        and sweeps is not None
        and legs_raw is not None
        and len(legs) == len(legs_raw)
        and cycle_valid
    ):
        return Trace(
            tuple(legs),
            cast("Any", terminal),
            cycle,
            sweeps,
        )
    return None


def _finalizer_inputs(
    verification: Any, path: str, errors: list[str]
) -> tuple[Any, Any] | None:
    from .finalizer import FinalizeEvidence, FinalizePolicy

    block = _mapping(verification, path, errors)
    if block is None:
        return None
    evidence_block = _mapping(block.get("evidence"), f"{path}.evidence", errors)
    policy_block = _mapping(block.get("policy"), f"{path}.policy", errors)
    if evidence_block is None or policy_block is None:
        return None
    _closed(evidence_block, _EVIDENCE_KEYS, f"{path}.evidence", errors)
    _closed(policy_block, _POLICY_KEYS, f"{path}.policy", errors)
    shots_raw = _list(evidence_block.get("shots"), f"{path}.evidence.shots", errors)
    shots: list[float] = []
    if shots_raw is not None:
        for index, raw in enumerate(shots_raw):
            value = _finite_number(raw, f"{path}.evidence.shots[{index}]", errors)
            if value is not None:
                shots.append(value)
    spans_raw = _list(
        evidence_block.get("sing_spans"), f"{path}.evidence.sing_spans", errors
    )
    spans: list[tuple[float, float]] = []
    if spans_raw is not None:
        for index, raw in enumerate(spans_raw):
            span_path = f"{path}.evidence.sing_spans[{index}]"
            if not isinstance(raw, list) or len(raw) != 2:
                errors.append(f"{span_path}: expected start/end pair")
                continue
            start = _finite_number(raw[0], f"{span_path}[0]", errors)
            end = _finite_number(raw[1], f"{span_path}[1]", errors)
            if start is not None and end is not None:
                spans.append((start, end))
    if policy_block != {
        "grid": None,
        "min_gap": "two-frame",
        "overlap_policy": "reject",
    }:
        errors.append(f"{path}.policy: outside frozen P5 policy")
    if (
        shots_raw is None
        or len(shots) != len(shots_raw)
        or spans_raw is None
        or len(spans) != len(spans_raw)
    ):
        return None
    return (
        FinalizeEvidence(tuple(shots), tuple(spans)),
        FinalizePolicy(),
    )


def _movement_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summary(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)

        def rank(percentile: float) -> float | None:
            if not ordered:
                return None
            return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

        return {
            "count": len(ordered),
            "max": max(ordered) if ordered else None,
            "p50": rank(0.50),
            "p90": rank(0.90),
        }

    return {
        side: summary(
            [
                abs(float(row["delta"]))
                for row in rows
                if isinstance(row.get("boundary"), Mapping)
                and row["boundary"].get("side") == side
            ]
        )
        for side in ("start", "end")
    }


def _finalizer(
    value: Any,
    path: str,
    errors: list[str],
    *,
    row: Mapping[str, Any],
    seed: tuple[Any, ...] | None,
    verification: Any,
    profile: Any | None,
) -> Mapping[str, Any] | None:
    block = _mapping(value, path, errors)
    if block is None:
        return None
    _closed(block, _FINALIZER_KEYS, path, errors)
    entries = _list(block.get("entries"), f"{path}.entries", errors)
    refusals = _list(block.get("refusals"), f"{path}.refusals", errors)
    if entries is not None and refusals is not None and entries != refusals:
        errors.append(f"{path}: entries/refusals report channels differ")
    if entries is not None:
        from .finalizer import REPORT_KINDS

        for index, raw in enumerate(entries):
            entry_path = f"{path}.entries[{index}]"
            entry = _mapping(raw, entry_path, errors)
            if entry is None:
                continue
            _closed(
                entry,
                frozenset({"cue_index", "evidence", "kind"}),
                entry_path,
                errors,
            )
            cue_index = entry.get("cue_index")
            if cue_index is not None and (
                isinstance(cue_index, bool) or not isinstance(cue_index, int)
            ):
                errors.append(f"{entry_path}.cue_index: expected integer or null")
            if entry.get("kind") not in REPORT_KINDS:
                errors.append(f"{entry_path}.kind: outside closed report vocabulary")
            evidence = _mapping(entry.get("evidence"), f"{entry_path}.evidence", errors)
            if (
                entry.get("kind") == "canonical-text-fallback"
                and evidence is not None
                and not isinstance(evidence.get("reason"), str)
            ):
                errors.append(
                    f"{entry_path}.evidence.reason: expected canonical fallback reason"
                )
    _string_list(block.get("deltas_fired"), f"{path}.deltas_fired", errors)
    movement_rows = _list(block.get("movement"), f"{path}.movement", errors)
    stability_rows = _string_list(
        block.get("stability_errors"), f"{path}.stability_errors", errors
    )
    trace_error_rows = _string_list(
        block.get("trace_errors"), f"{path}.trace_errors", errors
    )
    _waiver_rows(block.get("waivers"), f"{path}.waivers", errors)
    if block.get("valid") is not True:
        errors.append(f"{path}.valid: schema 2 requires a valid finalizer row")
    if block.get("stability_errors"):
        errors.append(f"{path}.stability_errors: expected empty list")
    if block.get("trace_errors"):
        errors.append(f"{path}.trace_errors: expected empty list")
    if block.get("schedule_canonicality") != "unverified":
        errors.append(f"{path}.schedule_canonicality: expected 'unverified'")
    if block.get("terminal") not in {
        "budget-exhausted",
        "cycle-adoption",
        "fixed-point",
    }:
        errors.append(f"{path}.terminal: outside closed terminal vocabulary")
    max_start = _finite_number(
        block.get("max_start_movement_s"),
        f"{path}.max_start_movement_s",
        errors,
    )
    max_sweeps = _nonnegative_int(
        block.get("max_sweeps_observed"),
        f"{path}.max_sweeps_observed",
        errors,
    )
    trace = _trace(block.get("trace"), f"{path}.trace", errors)
    distribution = _mapping(
        block.get("movement_distribution"), f"{path}.movement_distribution", errors
    )
    if distribution is not None:
        _closed(
            distribution,
            _MOVEMENT_DISTRIBUTION_KEYS,
            f"{path}.movement_distribution",
            errors,
        )
        for side in ("start", "end"):
            summary = _mapping(
                distribution.get(side), f"{path}.movement_distribution.{side}", errors
            )
            if summary is None:
                continue
            _closed(
                summary,
                _MOVEMENT_SUMMARY_KEYS,
                f"{path}.movement_distribution.{side}",
                errors,
            )
            _nonnegative_int(
                summary.get("count"),
                f"{path}.movement_distribution.{side}.count",
                errors,
            )
            for name in ("max", "p50", "p90"):
                if summary.get(name) is not None:
                    _finite_number(
                        summary.get(name),
                        f"{path}.movement_distribution.{side}.{name}",
                        errors,
                    )

    inputs = _finalizer_inputs(verification, f"{path}.verification", errors)
    row_cues = row.get("cues")
    if (
        seed is not None
        and profile is not None
        and inputs is not None
        and trace is not None
        and isinstance(row_cues, list)
    ):
        from .trace_validator import replay_trace, stability_check

        evidence, policy = inputs
        delivered_rows: list[tuple[float, float]] = []
        for index, cue in enumerate(row_cues):
            if not isinstance(cue, Mapping):
                break
            start = _finite_number(
                cue.get("start"), f"{path}.row.cues[{index}].start", errors
            )
            end = _finite_number(
                cue.get("end"), f"{path}.row.cues[{index}].end", errors
            )
            if start is None or end is None:
                break
            delivered_rows.append((start, end))
        if len(delivered_rows) == len(row_cues) == len(seed):
            delivered = tuple(delivered_rows)
            recomputed_trace_errors = list(
                replay_trace(
                    trace,
                    seed,
                    profile=profile,
                    evidence=evidence,
                    policy=policy,
                    delivered=delivered,
                )
            )
            if (
                trace_error_rows is not None
                and trace_error_rows != recomputed_trace_errors
            ):
                errors.append(
                    f"{path}.trace_errors: stale trace replay; recomputed "
                    f"{recomputed_trace_errors!r}"
                )
            recomputed_stability_errors = list(
                stability_check(
                    delivered,
                    seed,
                    profile=profile,
                    evidence=evidence,
                    policy=policy,
                    terminal=trace.terminal,
                )
            )
            if (
                stability_rows is not None
                and stability_rows != recomputed_stability_errors
            ):
                errors.append(
                    f"{path}.stability_errors: stale stability replay; recomputed "
                    f"{recomputed_stability_errors!r}"
                )

            expected_movement: list[dict[str, Any]] = []
            for index, (phase1, delivered_pair) in enumerate(zip(seed, delivered)):
                for side, slot in (("start", 0), ("end", 1)):
                    phase1_value = phase1.start if slot == 0 else phase1.end
                    delivered_value = delivered_pair[slot]
                    expected_movement.append(
                        {
                            "boundary": {"cue_index": index, "side": side},
                            "delivered": delivered_value,
                            "delta": delivered_value - phase1_value,
                            "phase1": phase1_value,
                        }
                    )
            decoded_movement: list[Mapping[str, Any]] = []
            if movement_rows is not None:
                for index, raw in enumerate(movement_rows):
                    movement_path = f"{path}.movement[{index}]"
                    movement = _mapping(raw, movement_path, errors)
                    if movement is None:
                        continue
                    _closed(movement, _MOVEMENT_KEYS, movement_path, errors)
                    _boundary_ref(
                        movement.get("boundary"), f"{movement_path}.boundary", errors
                    )
                    for key in ("delivered", "delta", "phase1"):
                        _finite_number(
                            movement.get(key), f"{movement_path}.{key}", errors
                        )
                    decoded_movement.append(movement)
                if movement_rows != expected_movement:
                    errors.append(
                        f"{path}.movement: ledger is detached from seed/delivered row"
                    )
            if distribution is not None and decoded_movement:
                recomputed_distribution = _movement_distribution(decoded_movement)
                if distribution != recomputed_distribution:
                    errors.append(
                        f"{path}.movement_distribution: stale; recomputed "
                        f"{recomputed_distribution!r}"
                    )
            expected_max_start = max(
                (
                    abs(delivered_pair[0] - phase1.start)
                    for phase1, delivered_pair in zip(seed, delivered)
                ),
                default=0.0,
            )
            if max_start is not None and max_start != expected_max_start:
                errors.append(
                    f"{path}.max_start_movement_s: expected {expected_max_start}, "
                    f"got {max_start}"
                )
            if max_sweeps is not None and max_sweeps != trace.sweeps:
                errors.append(
                    f"{path}.max_sweeps_observed: expected trace sweeps "
                    f"{trace.sweeps}, got {max_sweeps}"
                )
            if block.get("terminal") != trace.terminal:
                errors.append(f"{path}.terminal: disagrees with trace terminal")
    return block


def _speaker_measurement(value: Any, path: str, errors: list[str]) -> None:
    block = _mapping(value, path, errors)
    if block is None:
        return
    _closed(block, _MEASUREMENT_KEYS, path, errors)
    buckets = _mapping(block.get("buckets"), f"{path}.buckets", errors)
    raw = _nonnegative_int(
        block.get("raw_in_speech_turn_changes"),
        f"{path}.raw_in_speech_turn_changes",
        errors,
    )
    if buckets is not None:
        _closed(buckets, _BUCKET_KEYS, f"{path}.buckets", errors)
        values = [
            _nonnegative_int(buckets.get(key), f"{path}.buckets.{key}", errors)
            for key in _BUCKET_KEYS
        ]
        if raw is not None and all(value is not None for value in values):
            if sum(value for value in values if value is not None) != raw:
                errors.append(f"{path}: speaker buckets do not conserve raw events")
    _list(block.get("matches"), f"{path}.matches", errors)


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

    intervals = _list(artifact.get("intervals"), "intervals", errors)
    validator = _mapping(artifact.get("validator"), "validator", errors)
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

    registry = _list(artifact.get("delta_registry"), "delta_registry", errors)
    if registry is not None:
        from .policy_delta import delta_registry_data

        if registry != delta_registry_data():
            errors.append("delta_registry: differs from the frozen registry")

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
    _fallback_rechecks(artifact, final_rows, errors)
    return errors


def assert_shadow_v2_payload(
    artifact: Mapping[str, Any], *, require_version: bool = True
) -> None:
    """Raise one compact error when a payload cannot claim live schema 2."""
    errors = validate_shadow_v2_payload(artifact, require_version=require_version)
    if errors:
        raise ValueError("invalid live shadow schema 2: " + "; ".join(errors))
