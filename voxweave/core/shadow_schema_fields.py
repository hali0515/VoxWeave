"""Field and sub-block readers for the closed schema-2 shadow validator.

This leaf layer holds the primitive readers, the IR decoders and the three
shared sub-block checks that every section checker of ``shadow_schema`` calls.
It imports nothing from ``shadow_schema``.  Like ``shadow_schema`` it is
reachable only through the lazily imported validator, so the flag-off import
pin holds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast


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
_UNIT_KEYS = frozenset({"confidence", "end", "id", "provenance", "start", "surface"})
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
_MARGIN_KEYS = frozenset({"count", "exact_ties", "min", "p05", "p50"})


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
