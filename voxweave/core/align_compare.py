"""Lazy, producer-independent P6 semantic primitive comparison (ALD-0--5)."""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass
from typing import Literal, cast

from voxweave.align_delta_registry import ALIGN_DELTA_REGISTRY_SHA256


SemanticDeltaID = Literal["ALD-0", "ALD-1", "ALD-2", "ALD-3", "ALD-4", "ALD-5"]

_TWO_FRAME = 2.0 / 24.0
_CHAIN_MAX_GAP = 0.5
_LINGER_CAP = 1.0
_HELD_WORD_MAX_GAP = 1.0
_FRAME = 1.0 / 24.0
_SHOT_LANDING = 12 * _FRAME
_TIMING_EPS = 1e-9
_PARTITION_EPS = 1e-6
_SWEEP_BUDGET = 10_000
_NO_SPACE_LANGUAGES = {"zh", "yue", "ja", "th", "lo", "my"}
_PUNCTUATION = re.compile(r"[.,](?!\d)|[;!?:。；！？：﹒﹔﹕﹖﹗，、﹐﹑]")


@dataclass(frozen=True)
class AlignDeltaOutcome:
    delta_id: SemanticDeltaID
    triggered: bool
    passed: bool


@dataclass(frozen=True)
class AlignComparison:
    registry_sha256: str
    active_classes: tuple[SemanticDeltaID, ...]
    outcomes: tuple[AlignDeltaOutcome, ...]
    primitive_field_diffs: tuple[str, ...]
    violations: tuple[SemanticDeltaID, ...]


class SemanticComparisonUnavailable(RuntimeError):
    pass


def semantic_comparison_available() -> bool:
    return True


def _same_float(left: object, right: object) -> bool:
    return type(left) is float and type(right) is float and left.hex() == right.hex()


def _same_optional_float(left: object, right: object) -> bool:
    return left is None and right is None or _same_float(left, right)


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _ordered_cues(value: object) -> tuple[object, ...]:
    cues = getattr(value, "cues", None)
    if not isinstance(cues, tuple):
        raise TypeError("semantic delivery cues must be an immutable tuple")
    return cues


def _display_normal_form(text: object) -> str | None:
    if type(text) is not str:
        return None
    stripped = _PUNCTUATION.sub("", text.replace("\n", " "))
    return "".join(
        character
        for character in stripped
        if not character.isspace() and character != "-"
    )


def _visual_width(text: str) -> int:
    return sum(
        1 if character.isascii() or character.isspace() else 2 for character in text
    )


def _word_text(word: object) -> str | None:
    value = getattr(word, "text", None)
    return value if type(value) is str else None


def _word_bound(word: object, side: str) -> float | None:
    value = getattr(word, side, None)
    return value if value is None or type(value) is float else None


def _exact_anchor_pair(
    words: tuple[object, ...],
) -> tuple[float | None, float | None]:
    if not words:
        return None, None
    first = words[0]
    last = words[-1]
    first_start = _word_bound(first, "start")
    last_end = _word_bound(last, "end")
    speech_start = (
        first_start
        if getattr(first, "provenance", None) == "aligner"
        and _finite_float(first_start)
        else None
    )
    speech_end = (
        last_end
        if getattr(last, "provenance", None) == "aligner" and _finite_float(last_end)
        else None
    )
    return speech_start, speech_end


def _fill_missing_spans(
    spans: tuple[tuple[float, float] | None, ...],
) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float] | None] = list(spans)
    index = 0
    while index < len(out):
        if out[index] is not None:
            index += 1
            continue
        stop = index
        while stop < len(out) and out[stop] is None:
            stop += 1
        previous = out[index - 1] if index else None
        following = spans[stop] if stop < len(spans) else None
        count = stop - index
        if (
            previous is not None
            and following is not None
            and following[0] - previous[1] <= 2.0
            and following[0] > previous[1]
        ):
            step = (following[0] - previous[1]) / (count + 1)
            for offset in range(count):
                start = previous[1] + step * (offset + 1)
                out[index + offset] = start, start + step
        elif previous is not None:
            start = previous[1]
            for offset in range(count):
                out[index + offset] = (
                    start + offset * 2.0,
                    start + (offset + 1) * 2.0,
                )
        elif following is not None:
            start = max(0.0, following[0] - count * 2.0)
            for offset in range(count):
                out[index + offset] = (
                    start + offset * 2.0,
                    start + (offset + 1) * 2.0,
                )
        else:
            for offset in range(count):
                out[index + offset] = offset * 2.0, (offset + 1) * 2.0
        index = stop
    return tuple(value if value is not None else (0.0, 2.0) for value in out)


def _display_seeds(
    anchors: tuple[tuple[float | None, float | None], ...],
) -> tuple[tuple[float, float], ...]:
    complete = tuple(
        (start, end) if start is not None and end is not None else None
        for start, end in anchors
    )
    fabricated = _fill_missing_spans(complete)
    result: list[tuple[float, float]] = []
    for (speech_start, speech_end), (fallback_start, fallback_end) in zip(
        anchors, fabricated, strict=True
    ):
        if speech_start is not None and speech_end is not None:
            result.append(
                (
                    speech_start,
                    speech_start + 0.05 if speech_start == speech_end else speech_end,
                )
            )
        elif speech_start is not None:
            result.append((speech_start, max(fallback_end, speech_start + 0.05)))
        elif speech_end is not None:
            result.append(
                (max(0.0, min(fallback_start, speech_end - 0.05)), speech_end)
            )
        else:
            result.append((fallback_start, fallback_end))
    return tuple(result)


def _duration_desire(cue: object, profile: object) -> float | None:
    start = getattr(cue, "seed_start", None)
    seed_end = getattr(cue, "seed_end", None)
    speech_end = getattr(cue, "speech_end", None)
    reading_chars = getattr(cue, "reading_chars", None)
    if (
        not _finite_float(start)
        or not _finite_float(seed_end)
        or speech_end is not None
        and not _finite_float(speech_end)
        or type(reading_chars) is not int
        or reading_chars < 0
    ):
        return None
    want = seed_end
    if speech_end is None:
        return want
    min_cue = getattr(profile, "min_cue_s", None)
    lag_out = getattr(profile, "lag_out_s", None)
    cps = getattr(profile, "cps", None)
    if not all(_finite_float(value) for value in (min_cue, lag_out, cps)):
        return None
    start = cast(float, start)
    seed_end = cast(float, seed_end)
    speech_end = cast(float, speech_end)
    min_cue = cast(float, min_cue)
    lag_out = cast(float, lag_out)
    cps = cast(float, cps)
    reading_chars = cast(int, reading_chars)
    want = seed_end
    if min_cue > 0:
        want = max(want, start + min_cue)
    if lag_out > 0:
        want = max(want, speech_end + lag_out)
    if cps > 0:
        need = reading_chars / cps
        want = max(want, min(start + need, speech_end + _LINGER_CAP))
    return want


def _nearest(shots: tuple[float, ...], value: float, window: float) -> float | None:
    position = bisect.bisect_left(shots, value)
    best: float | None = None
    for candidate in shots[max(position - 1, 0) : position + 1]:
        distance = abs(candidate - value)
        if distance <= window and (best is None or distance < abs(best - value)):
            best = candidate
    return best


def _guarded_end(want: float, seed_end: float, next_start: float | None) -> float:
    if want <= seed_end:
        return seed_end
    if next_start is None:
        return want
    if next_start - seed_end <= _TWO_FRAME:
        return seed_end
    return min(want, next_start)


def _chain_end(end: float, next_start: float | None) -> float:
    if next_start is None:
        return end
    gap = next_start - end
    if 0 <= gap < _CHAIN_MAX_GAP and gap > _TWO_FRAME:
        return next_start - _TWO_FRAME
    return end


def _timed_word_data(cue: object) -> tuple[tuple[float, float], ...]:
    rows: list[tuple[float, float]] = []
    word_data = getattr(cue, "word_data", None)
    if not isinstance(word_data, list):
        return ()
    for word in word_data:
        if not isinstance(word, dict):
            continue
        start = word.get("start")
        end = word.get("end")
        if _finite_float(start) and _finite_float(end):
            rows.append((cast(float, start), cast(float, end)))
    return tuple(sorted(rows))


def _cap_end(
    start: float,
    end: float,
    cue: object,
    next_start: float | None,
    max_cue: float,
) -> float:
    if max_cue == 0.0 or end - start <= max_cue:
        return end
    cap = start + max_cue
    timed = _timed_word_data(cue)
    if not timed or max(value[1] for value in timed) <= cap:
        return cap
    held_end = timed[0][1]
    for previous, following in zip(timed, timed[1:]):
        if following[0] - previous[1] > _HELD_WORD_MAX_GAP:
            break
        held_end = following[1]
    target = held_end if next_start is None else min(held_end, next_start)
    return max(cap, target)


def _within_cap(start: float, end: float, cap: float | None) -> bool:
    return cap is None or end - start <= cap + _TIMING_EPS


def _shot_in(
    start: float,
    end: float,
    *,
    speech_start: float | None,
    previous_end: float | None,
    shots: tuple[float, ...],
    snap: float,
    cap: float | None,
) -> float:
    cut = _nearest(shots, start, snap)
    if cut is None or abs(cut - start) <= _TIMING_EPS:
        return start
    offset = start - cut
    if -7 * _FRAME - _TIMING_EPS <= offset < 0:
        candidate = cut
    elif offset < 0:
        candidate = cut - _SHOT_LANDING
    elif offset <= 9 * _FRAME + _TIMING_EPS:
        candidate = cut
    else:
        candidate = cut + _SHOT_LANDING
    if previous_end is not None:
        candidate = max(candidate, previous_end + _TWO_FRAME)
    truncates = (
        candidate > start
        and speech_start is not None
        and candidate > speech_start + _TIMING_EPS
    )
    if (
        candidate < end - _TWO_FRAME
        and (candidate <= start or candidate - start <= _SHOT_LANDING)
        and _within_cap(candidate, end, cap)
        and not truncates
    ):
        return candidate
    return start


def _shot_out(
    start: float,
    end: float,
    *,
    speech_end: float | None,
    next_start: float | None,
    shots: tuple[float, ...],
    snap: float,
    cap: float | None,
) -> float:
    cut = _nearest(shots, end, snap)
    if cut is None:
        return end
    anchor = end if speech_end is None else speech_end
    offset = end - cut
    target = (
        cut - _TWO_FRAME if offset <= 5 * _FRAME + _TIMING_EPS else cut + _SHOT_LANDING
    )
    applied = False
    if target > end + _TIMING_EPS:
        if (next_start is None or target <= next_start - _TWO_FRAME) and _within_cap(
            start, target, cap
        ):
            end = target
            applied = True
    elif target < end - _TIMING_EPS and target >= anchor and target > start:
        end = target
        applied = True
    if not applied and 0 < offset <= 5 * _FRAME + _TIMING_EPS:
        target = cut + _SHOT_LANDING
        if (
            target > end
            and (next_start is None or target <= next_start - _TWO_FRAME)
            and _within_cap(start, target, cap)
        ):
            end = target
    return end


def _ladder(
    previous_end: float, previous_speech_end: float | None, next_start: float
) -> tuple[int, float] | None:
    if next_start - previous_end >= _TWO_FRAME - _PARTITION_EPS:
        return None
    if previous_speech_end is None or previous_speech_end <= next_start - _TWO_FRAME:
        return 1, next_start - _TWO_FRAME
    if previous_speech_end <= next_start:
        return 2, min(previous_end, previous_speech_end)
    return 3, min(previous_end, previous_speech_end)


LegDescriptor = tuple[
    str,
    int,
    int,
    int,
    tuple[int, str],
    float,
    float,
    tuple[tuple[int, str, float | None], ...],
]


def _sweep(
    state: tuple[tuple[float, float], ...],
    cues: tuple[object, ...],
    profile: object,
    evidence: object,
    sweep: int,
) -> tuple[tuple[tuple[float, float], ...], tuple[LegDescriptor, ...]]:
    starts = [pair[0] for pair in state]
    ends = [pair[1] for pair in state]
    shots_raw = getattr(evidence, "shots", ())
    if not isinstance(shots_raw, tuple) or not all(
        _finite_float(value) for value in shots_raw
    ):
        return state, ()
    shots = tuple(sorted(shots_raw))
    snap = getattr(profile, "shot_snap_s", None)
    max_cue = getattr(profile, "max_cue_s", None)
    if not _finite_float(snap) or not _finite_float(max_cue):
        return state, ()
    cap = max_cue if max_cue else None
    legs: list[LegDescriptor] = []

    def emit(
        rule: str,
        slot: int,
        cue_index: int,
        side: str,
        before: float,
        after: float,
        reads: tuple[tuple[int, str, float | None], ...],
        *,
        target_index: int | None = None,
    ) -> None:
        if before == after:
            return
        legs.append(
            (
                rule,
                sweep,
                cue_index,
                slot,
                (cue_index if target_index is None else target_index, side),
                before,
                after,
                reads,
            )
        )

    for index, cue in enumerate(cues):
        next_start = starts[index + 1] if index + 1 < len(cues) else None
        next_read = () if next_start is None else ((index + 1, "start", next_start),)
        start_read = ((index, "start", starts[index]),)
        desire = _duration_desire(cue, profile)
        if desire is None:
            return state, ()
        seed_end = getattr(cue, "seed_end", None)
        if not _finite_float(seed_end):
            return state, ()
        seed_end = cast(float, seed_end)
        snap = cast(float, snap)
        max_cue = cast(float, max_cue)
        changed = _guarded_end(desire, seed_end, next_start)
        emit(
            "duration-desire",
            1,
            index,
            "end",
            ends[index],
            changed,
            start_read + next_read,
        )
        ends[index] = changed
        changed = _chain_end(ends[index], next_start)
        emit("chain", 2, index, "end", ends[index], changed, next_read)
        ends[index] = changed
        changed = _cap_end(starts[index], ends[index], cue, next_start, max_cue)
        emit(
            "cap",
            3,
            index,
            "end",
            ends[index],
            changed,
            start_read + next_read,
        )
        ends[index] = changed
        if snap > 0 and shots:
            previous_end = ends[index - 1] if index else None
            previous_read = (
                () if previous_end is None else ((index - 1, "end", previous_end),)
            )
            changed_start = _shot_in(
                starts[index],
                ends[index],
                speech_start=getattr(cue, "speech_start", None),
                previous_end=previous_end,
                shots=shots,
                snap=snap,
                cap=cap,
            )
            emit(
                "shot-in",
                4,
                index,
                "start",
                starts[index],
                changed_start,
                previous_read + ((index, "end", ends[index]),),
            )
            starts[index] = changed_start
            changed_end = _shot_out(
                starts[index],
                ends[index],
                speech_end=getattr(cue, "speech_end", None),
                next_start=next_start,
                shots=shots,
                snap=snap,
                cap=cap,
            )
            emit(
                "shot-out",
                5,
                index,
                "end",
                ends[index],
                changed_end,
                ((index, "start", starts[index]),) + next_read,
            )
            ends[index] = changed_end
        if index:
            ladder = _ladder(
                ends[index - 1],
                getattr(cues[index - 1], "speech_end", None),
                starts[index],
            )
            if ladder is not None:
                branch, changed = ladder
                emit(
                    f"ladder-{branch}",
                    6,
                    index,
                    "end",
                    ends[index - 1],
                    changed,
                    ((index, "start", starts[index]),),
                    target_index=index - 1,
                )
                ends[index - 1] = changed
    return tuple(zip(starts, ends, strict=True)), tuple(legs)


def _observed_leg(value: object) -> LegDescriptor | None:
    target = getattr(value, "target", None)
    target_index = getattr(target, "cue_index", None)
    target_side = getattr(target, "side", None)
    raw_reads = getattr(value, "reads", None)
    if (
        type(getattr(value, "rule_id", None)) is not str
        or type(getattr(value, "sweep", None)) is not int
        or type(getattr(value, "cue_index", None)) is not int
        or type(getattr(value, "slot", None)) is not int
        or type(target_index) is not int
        or target_side not in ("start", "end")
        or not _finite_float(getattr(value, "from_value", None))
        or not _finite_float(getattr(value, "to_value", None))
        or not isinstance(raw_reads, tuple)
    ):
        return None
    reads: list[tuple[int, str, float | None]] = []
    for read in raw_reads:
        boundary = getattr(read, "boundary", None)
        index = getattr(boundary, "cue_index", None)
        side = getattr(boundary, "side", None)
        observed = getattr(read, "value", None)
        if (
            type(index) is not int
            or side not in ("start", "end")
            or observed is not None
            and not _finite_float(observed)
        ):
            return None
        reads.append((index, side, observed))
    return (
        getattr(value, "rule_id"),
        getattr(value, "sweep"),
        getattr(value, "cue_index"),
        getattr(value, "slot"),
        (target_index, target_side),
        getattr(value, "from_value"),
        getattr(value, "to_value"),
        tuple(reads),
    )


def _reference_trace(
    phase1: tuple[object, ...], profile: object, evidence: object
) -> tuple[
    str,
    int,
    tuple[LegDescriptor, ...],
    tuple[tuple[float, float], ...],
]:
    raw_state = tuple(
        (getattr(cue, "start", None), getattr(cue, "end", None)) for cue in phase1
    )
    if not all(_finite_float(start) and _finite_float(end) for start, end in raw_state):
        raise SemanticComparisonUnavailable("phase-1 state is not finite")
    state = cast(tuple[tuple[float, float], ...], raw_state)
    order = [state]
    seen = {state}
    legs: list[LegDescriptor] = []
    for sweep in range(1, _SWEEP_BUDGET + 1):
        moved, sweep_legs = _sweep(state, phase1, profile, evidence, sweep)
        legs.extend(sweep_legs)
        if moved == state:
            return "fixed-point", sweep, tuple(legs), state
        if moved in seen:
            first = order.index(moved)
            members = tuple(order[first:])
            adopted = min(
                members,
                key=lambda item: tuple(number for pair in item for number in pair),
            )
            return "cycle-adoption", sweep, tuple(legs), adopted
        seen.add(moved)
        order.append(moved)
        state = moved
    return "budget-exhausted", _SWEEP_BUDGET, tuple(legs), state


def _trace_relation(
    observation: object,
    phase1: tuple[object, ...],
    profile: object,
    evidence: object,
    v2_cues: tuple[object, ...],
) -> tuple[bool, tuple[tuple[float, float], ...]]:
    trace = getattr(observation, "trace", None)
    raw_legs = getattr(trace, "legs", None)
    if not isinstance(raw_legs, tuple):
        return False, ()
    observed_legs = tuple(_observed_leg(leg) for leg in raw_legs)
    if any(leg is None for leg in observed_legs):
        return False, ()
    terminal, sweeps, expected_legs, delivered = _reference_trace(
        phase1, profile, evidence
    )
    observed_delivery_raw = getattr(observation, "delivered", None)
    observed_delivery = None
    if isinstance(observed_delivery_raw, tuple):
        if all(
            isinstance(row, tuple)
            and len(row) == 2
            and _finite_float(row[0])
            and _finite_float(row[1])
            for row in observed_delivery_raw
        ):
            observed_delivery = observed_delivery_raw
        elif all(isinstance(row, dict) for row in observed_delivery_raw):
            projected = tuple(
                (row.get("start"), row.get("end")) for row in observed_delivery_raw
            )
            if all(
                _finite_float(start) and _finite_float(end) for start, end in projected
            ):
                observed_delivery = projected
        elif all(
            _finite_float(getattr(row, "start", None))
            and _finite_float(getattr(row, "end", None))
            for row in observed_delivery_raw
        ):
            observed_delivery = tuple(
                (getattr(row, "start"), getattr(row, "end"))
                for row in observed_delivery_raw
            )
    v2_delivery = tuple(
        (getattr(cue, "start", None), getattr(cue, "end", None)) for cue in v2_cues
    )
    report = getattr(observation, "report", None)
    valid = (
        getattr(trace, "terminal", None) == terminal
        and getattr(trace, "sweeps", None) == sweeps
        and observed_legs == expected_legs
        and observed_delivery == delivered
        and v2_delivery == delivered
        and getattr(report, "terminal", None) == terminal
    )
    return valid, delivered


def _phase1_relation(
    authority_blocks: tuple[object, ...],
    phase1: tuple[object, ...],
    profile: object,
) -> tuple[bool, tuple[tuple[float | None, float | None], ...]]:
    if len(authority_blocks) != len(phase1):
        return False, ()
    anchors = tuple(
        _exact_anchor_pair(tuple(getattr(block, "word_data", None) or ()))
        for block in authority_blocks
    )
    seeds = _display_seeds(anchors)
    language = getattr(profile, "language", None)
    if type(language) is not str:
        return False, anchors
    valid = True
    cursor = 0
    for index, (block, cue, anchor, seed) in enumerate(
        zip(authority_blocks, phase1, anchors, seeds, strict=True)
    ):
        words = tuple(getattr(block, "word_data", None) or ())
        surfaces = tuple(_word_text(word) for word in words)
        if not surfaces or any(surface is None for surface in surfaces):
            valid = False
            continue
        footprint = ("" if language in _NO_SPACE_LANGUAGES else " ").join(
            surface for surface in surfaces if surface is not None
        )
        text = getattr(cue, "text", None)
        text_value = text if type(text) is str else ""
        lines = getattr(cue, "lines", None)
        widths = getattr(cue, "cell_widths", None)
        reading_chars = getattr(cue, "reading_chars", None)
        desired = _duration_desire(cue, profile)
        unit_range = cursor, cursor + len(words)
        cursor = unit_range[1]
        valid = valid and (
            type(getattr(block, "source_index", None)) is int
            and getattr(cue, "index", None) == index
            and _same_float(getattr(cue, "seed_start", None), seed[0])
            and _same_float(getattr(cue, "seed_end", None), seed[1])
            and _same_float(getattr(cue, "start", None), seed[0])
            and desired is not None
            and _same_float(getattr(cue, "end", None), desired)
            and _same_optional_float(getattr(cue, "speech_start", None), anchor[0])
            and _same_optional_float(getattr(cue, "speech_end", None), anchor[1])
            and _display_normal_form(text) == _display_normal_form(footprint)
            and type(text) is str
            and isinstance(lines, tuple)
            and all(type(line) is str for line in lines)
            and text_value == "\n".join(lines)
            and isinstance(widths, tuple)
            and widths == tuple(_visual_width(line) for line in lines)
            and type(reading_chars) is int
            and reading_chars
            == sum(1 for character in text_value if not character.isspace())
            and getattr(cue, "unit_range", None) == unit_range
        )
    return valid, anchors


def _lyric_for_span(
    start: float,
    end: float,
    sing_spans: tuple[tuple[float, float], ...],
) -> bool:
    duration = end - start
    if duration <= 0.0 or not sing_spans:
        return False
    overlap = 0.0
    for raw_start, raw_end in sing_spans:
        if not _finite_float(raw_start) or not _finite_float(raw_end):
            continue
        overlap += max(0.0, min(end, raw_end) - max(start, raw_start))
    return overlap / duration >= 0.5


def _expected_lyrics(
    authority_blocks: tuple[object, ...],
    seeds: tuple[tuple[float, float], ...],
    evidence: object,
) -> tuple[bool, ...] | None:
    sing_spans = getattr(evidence, "sing_spans", None)
    if not isinstance(sing_spans, tuple):
        return None
    expected: list[bool] = []
    for block, seed in zip(authority_blocks, seeds, strict=True):
        words = tuple(getattr(block, "word_data", None) or ())
        if not words:
            return None
        start_anchor, end_anchor = _exact_anchor_pair(words)
        start = seed[0] if start_anchor is None else start_anchor
        end = seed[1] if end_anchor is None else end_anchor
        expected.append(_lyric_for_span(start, end, sing_spans))
    return tuple(expected)


def _qwen_origin_relation(
    physical_calls: tuple[object, ...], authority_blocks: tuple[object, ...]
) -> tuple[bool, bool]:
    triggered = any(
        not _same_float(
            getattr(call, "legacy_origin_seconds", None),
            getattr(call, "physical_origin_seconds", None),
        )
        for call in physical_calls
    )
    calls = {getattr(call, "call_index", None): call for call in physical_calls}
    passed = bool(physical_calls)
    for call in physical_calls:
        sample_start = getattr(call, "sample_start", None)
        sample_end = getattr(call, "sample_end", None)
        sample_rate = getattr(call, "sample_rate", None)
        physical = getattr(call, "physical_origin_seconds", None)
        passed = passed and (
            type(sample_start) is int
            and type(sample_end) is int
            and type(sample_rate) is int
            and sample_start >= 0
            and sample_end >= sample_start
            and sample_rate > 0
            and _same_float(physical, sample_start / sample_rate)
            and _same_float(getattr(call, "authority_origin_seconds", None), physical)
            and getattr(call, "legacy_origin_kind", None) == "nominal-route"
        )
    for block in authority_blocks:
        for word in tuple(getattr(block, "word_data", None) or ()):
            call = calls.get(getattr(word, "call_index", None))
            if call is None:
                passed = False
                continue
            origin = getattr(call, "physical_origin_seconds", None)
            relative_start = getattr(word, "relative_start", None)
            relative_end = getattr(word, "relative_end", None)
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if relative_start is None:
                passed = passed and start is None
            else:
                passed = (
                    passed
                    and _finite_float(origin)
                    and _same_float(start, relative_start + origin)
                )
            if relative_end is None:
                passed = passed and end is None
            else:
                passed = (
                    passed
                    and _finite_float(origin)
                    and _same_float(end, relative_end + origin)
                )
    return triggered, passed


def _common_observation_relation(observation: object) -> bool:
    lineage = getattr(observation, "semantic_root_lineage", None)
    partition = getattr(observation, "partition_result", None)
    exit_driving = getattr(partition, "exit_driving", None)
    return (
        isinstance(lineage, tuple)
        and any(
            isinstance(row, tuple)
            and len(row) == 6
            and row[1] == "align/delivery-finalizer/v2"
            and row[4] == "phase1"
            and row[5] is None
            for row in lineage
        )
        and (exit_driving is False or exit_driving == ())
        and getattr(observation, "trace_problems", None) == ()
        and getattr(observation, "stability_problems", None) == ()
    )


def compare_semantic_deltas(
    *,
    route_kind: str,
    physical_calls: tuple[object, ...],
    authority_blocks: tuple[object, ...],
    legacy: object,
    v2: object,
    semantic_observation: object,
    profile: object,
    evidence: object,
) -> AlignComparison:
    """Derive O from sealed primitives and validate every active semantic class."""
    legacy_cues = _ordered_cues(legacy)
    v2_cues = _ordered_cues(v2)
    phase1 = getattr(semantic_observation, "phase1_seed", None)
    if not isinstance(phase1, tuple):
        raise SemanticComparisonUnavailable("semantic phase-1 group is unavailable")
    same_shape = (
        len(authority_blocks) == len(legacy_cues) == len(v2_cues) == len(phase1)
    )
    phase1_valid, anchors = _phase1_relation(authority_blocks, phase1, profile)
    seeds = _display_seeds(anchors) if anchors else ()
    trace_valid, _delivered = _trace_relation(
        semantic_observation, phase1, profile, evidence, v2_cues
    )
    common_valid = (
        same_shape
        and phase1_valid
        and _common_observation_relation(semantic_observation)
    )

    if route_kind == "qwen-crop":
        ald0_triggered, ald0_relation = _qwen_origin_relation(
            physical_calls, authority_blocks
        )
    else:
        ald0_triggered = False
        ald0_relation = all(
            _same_float(
                getattr(call, "legacy_origin_seconds", None),
                getattr(call, "physical_origin_seconds", None),
            )
            for call in physical_calls
        )

    ald1_triggered = not same_shape or any(
        getattr(seed, "text", None) != getattr(old, "text", None)
        for seed, old in zip(phase1, legacy_cues)
    )
    ald1_relation = common_valid and all(
        getattr(new, "text", None) == getattr(seed, "text", None)
        for seed, new in zip(phase1, v2_cues, strict=True)
    )

    ald2_triggered = not same_shape or any(
        not _same_float(getattr(seed, "end", None), getattr(old, "end", None))
        for seed, old in zip(phase1, legacy_cues)
    )
    ald2_relation = common_valid and trace_valid

    ald3_triggered = not same_shape or any(
        start is None or end is None or start == end for start, end in anchors
    )
    ald3_relation = common_valid and trace_valid

    raw_legs = getattr(getattr(semantic_observation, "trace", None), "legs", None)
    ald4_triggered = not isinstance(raw_legs, tuple) or bool(raw_legs)
    ald4_relation = common_valid and trace_valid

    expected_lyrics = _expected_lyrics(authority_blocks, seeds, evidence)
    ald5_triggered = (
        expected_lyrics is None
        or not same_shape
        or any(
            (True if expected else None) != getattr(old, "lyric", None)
            for expected, old in zip(expected_lyrics, legacy_cues)
        )
    )
    ald5_relation = (
        common_valid
        and expected_lyrics is not None
        and all(
            getattr(seed, "lyric", None) == (True if expected else None)
            and getattr(new, "lyric", None) == (True if expected else None)
            for expected, seed, new in zip(
                expected_lyrics, phase1, v2_cues, strict=True
            )
        )
    )

    outcomes = (
        AlignDeltaOutcome("ALD-0", ald0_triggered, ald0_relation),
        AlignDeltaOutcome("ALD-1", ald1_triggered, ald1_relation),
        AlignDeltaOutcome("ALD-2", ald2_triggered, ald2_relation),
        AlignDeltaOutcome("ALD-3", ald3_triggered, ald3_relation),
        AlignDeltaOutcome("ALD-4", ald4_triggered, ald4_relation),
        AlignDeltaOutcome("ALD-5", ald5_triggered, ald5_relation),
    )
    active = cast(
        tuple[SemanticDeltaID, ...],
        tuple(outcome.delta_id for outcome in outcomes if outcome.triggered),
    )
    fields: list[str] = []
    for field, active_now in (
        ("authority-time", ald0_triggered),
        ("text", ald1_triggered),
        ("start", ald3_triggered or ald4_triggered),
        ("end", ald2_triggered or ald3_triggered or ald4_triggered),
        ("lyric", ald5_triggered),
    ):
        if active_now:
            fields.append(field)
    violations = cast(
        tuple[SemanticDeltaID, ...],
        tuple(
            outcome.delta_id
            for outcome in outcomes
            if outcome.triggered and not outcome.passed
        ),
    )
    return AlignComparison(
        ALIGN_DELTA_REGISTRY_SHA256,
        active,
        outcomes,
        tuple(fields),
        violations,
    )


__all__ = [
    "AlignComparison",
    "AlignDeltaOutcome",
    "SemanticComparisonUnavailable",
    "compare_semantic_deltas",
    "semantic_comparison_available",
]
