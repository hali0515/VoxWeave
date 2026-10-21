"""Finalizer row decoding and replay for the closed schema-2 shadow validator.

This module decodes the finalizer row and its verification block into the
finalizer dataclasses (seed, trace, inputs) and re-runs ``replay_trace`` and
``stability_check`` on them.  It reads fields only through the leaf readers of
``shadow_schema_fields`` and imports nothing from ``shadow_schema``.  Like
``shadow_schema`` it is reachable only through the lazily imported validator,
so the flag-off import pin holds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .shadow_lanes import LANE_FINALIZER
from .shadow_schema_fields import (
    _closed,
    _finite_number,
    _list,
    _mapping,
    _nonnegative_int,
    _string_list,
    _waiver_rows,
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


# Trace legs: closed key set per leg, then the neighbour reads of each leg.
def _trace_legs(legs_raw: list[Any] | None, path: str, errors: list[str]) -> list[Any]:
    from .finalizer import NeighbourRead, TraceLeg

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
    return legs


# Trace cycle evidence: adopted state and members, absent cycle is valid.
def _trace_cycle(cycle_value: Any, path: str, errors: list[str]) -> tuple[Any, bool]:
    from .finalizer import CycleEvidence

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
    return cycle, cycle_valid


def _trace(value: Any, path: str, errors: list[str]) -> Any | None:
    from .finalizer import Trace

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
    legs = _trace_legs(legs_raw, path, errors)

    cycle_value = block.get("cycle")
    cycle, cycle_valid = _trace_cycle(cycle_value, path, errors)
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


# Report entries: closed key set, cue index, kind vocabulary and fallback reason.
def _finalizer_entries(entries: list[Any], path: str, errors: list[str]) -> None:
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


# Shape of the movement_distribution block: closed keys per side and summary.
def _finalizer_distribution_shape(
    distribution: Mapping[str, Any], path: str, errors: list[str]
) -> None:
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


# Delivered cue times, then the trace and stability replays. Returns the
# delivered times when the row, the seed and the delivered cues have the same
# length, otherwise None.
def _finalizer_replay(
    path: str,
    errors: list[str],
    *,
    trace: Any,
    seed: tuple[Any, ...],
    profile: Any,
    inputs: tuple[Any, Any],
    row_cues: list[Any],
    trace_error_rows: list[str] | None,
    stability_rows: list[str] | None,
) -> tuple[tuple[float, float], ...] | None:
    from .trace_validator import replay_trace, stability_check

    evidence, policy = inputs
    delivered_rows: list[tuple[float, float]] = []
    for index, cue in enumerate(row_cues):
        if not isinstance(cue, Mapping):
            break
        start = _finite_number(
            cue.get("start"), f"{path}.row.cues[{index}].start", errors
        )
        end = _finite_number(cue.get("end"), f"{path}.row.cues[{index}].end", errors)
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
        if trace_error_rows is not None and trace_error_rows != recomputed_trace_errors:
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
        if stability_rows is not None and stability_rows != recomputed_stability_errors:
            errors.append(
                f"{path}.stability_errors: stale stability replay; recomputed "
                f"{recomputed_stability_errors!r}"
            )
        return delivered
    return None


# Movement ledger against seed and delivered cues, then the derived maxima and
# the terminal. The terminal is read from block last.
def _finalizer_movement(
    block: Mapping[str, Any],
    path: str,
    errors: list[str],
    *,
    seed: tuple[Any, ...],
    delivered: tuple[tuple[float, float], ...],
    trace: Any,
    movement_rows: list[Any] | None,
    distribution: Mapping[str, Any] | None,
    max_start: float | None,
    max_sweeps: int | None,
) -> None:
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
            _boundary_ref(movement.get("boundary"), f"{movement_path}.boundary", errors)
            for key in ("delivered", "delta", "phase1"):
                _finite_number(movement.get(key), f"{movement_path}.{key}", errors)
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
        _finalizer_entries(entries, path, errors)
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
        _finalizer_distribution_shape(distribution, path, errors)

    inputs = _finalizer_inputs(verification, f"{path}.verification", errors)
    row_cues = row.get("cues")
    if (
        seed is not None
        and profile is not None
        and inputs is not None
        and trace is not None
        and isinstance(row_cues, list)
    ):
        delivered = _finalizer_replay(
            path,
            errors,
            trace=trace,
            seed=seed,
            profile=profile,
            inputs=inputs,
            row_cues=row_cues,
            trace_error_rows=trace_error_rows,
            stability_rows=stability_rows,
        )
        if delivered is not None:
            _finalizer_movement(
                block,
                path,
                errors,
                seed=seed,
                delivered=delivered,
                trace=trace,
                movement_rows=movement_rows,
                distribution=distribution,
                max_start=max_start,
                max_sweeps=max_sweeps,
            )
    return block
