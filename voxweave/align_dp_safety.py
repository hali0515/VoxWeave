"""Fail-closed validation for over-budget CTC/MMS route hints and plans.

The silence-anchor planner is deliberately kept separate from this module.  These
checks neither sort route hints nor repair planner output: unsafe data is classified
and refused before a backend can construct its forced-alignment trellis.  The one
projection offered here, :func:`route_hint_envelope`, only widens validated
overlapping cues to a monotone cover for the planner; it never reorders them, and
:func:`validate_widened_plans` refuses a plan on that cover that splits a cue.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, NoReturn, cast

from voxweave.align_failures import CanonicalFailure

DpRouteDetail = Literal[
    "hint-shape",
    "hint-nonfinite",
    "hint-nonmonotone",
    "plan-nontiling",
    "crop-geometry",
    "crop-over-budget",
]


class DpRouteHintsInvalid(RuntimeError):
    """Canonical RAT-4 refusal for an unsafe over-budget full-pass route."""

    def __init__(self, detail_code: DpRouteDetail, reason: str) -> None:
        super().__init__(
            f"DP budget route planning refused unsafe hints ({detail_code}): {reason}"
        )
        self.failure = CanonicalFailure(
            "dp-route-hints-invalid", "route-plan", detail_code
        )


def _refuse(detail_code: DpRouteDetail, reason: str) -> NoReturn:
    raise DpRouteHintsInvalid(detail_code, reason)


def _is_exact_number(value: object) -> bool:
    return type(value) in (int, float)


def _require_positive_finite(value: object, *, name: str) -> float:
    if not _is_exact_number(value):
        _refuse("hint-shape", f"{name} is not an exact numeric value")
    number = float(cast(int | float, value))
    if not math.isfinite(number) or number <= 0.0:
        _refuse("hint-nonfinite", f"{name} is not finite and positive")
    return number


def _clock(seconds: float) -> str:
    """``HH:MM:SS.mmm`` for a validated finite, non-negative time."""

    ms = round(seconds * 1000)
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    whole, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{ms:03d}"


def _untimed_remedy(
    sample_rate: object, max_dp_frames: object, frame_stride: object
) -> str:
    """What to do when long audio arrives without any cue timestamp."""

    budget = ""
    values = (sample_rate, max_dp_frames, frame_stride)
    if all(_is_exact_number(value) for value in values):
        rate, frames, stride = (float(cast(int | float, value)) for value in values)
        if rate > 0.0:
            seconds = frames * stride / rate
            if math.isfinite(seconds) and seconds > 0.0:
                budget = f" (about {seconds / 60:.0f} min)"
    return (
        "audio longer than the single-pass alignment budget"
        f"{budget} is aligned in pieces split at cue timestamps. Align a VTT that"
        " keeps its cue timing lines (`voxweave process` writes them unless"
        " --no-timestamps is given), or raise VOXWEAVE_CTC_MAX_DP_FRAMES (config"
        " key ctc_max_dp_frames) to align the whole file in one pass, which needs"
        " more memory"
    )


def validate_over_budget_hints(
    bounds: Sequence[tuple[float, float] | None] | None,
    *,
    block_count: int,
    audio_end: float,
    sample_rate: int,
    max_dp_frames: int | float,
    frame_stride: int | float,
    chunk_fraction: float,
) -> None:
    """Validate lexical route hints and the configured physical DP budget.

    ``None`` entries are permitted for untimed blocks, but every present entry is
    an exact two-number pair.  Hints stay in lexical source order; no timestamp-key
    sorting is performed here or by the caller.  Overlapping and nested cues (the
    align finalizer's flash-cue rescue writes the latter) are accepted: the caller
    plans on their :func:`route_hint_envelope` and refuses a plan that still splits
    inside one (:func:`validate_widened_plans`).  A cue that lies entirely before
    an earlier cue is out of time order and is refused, naming both cues.
    """

    if type(block_count) is not int or block_count <= 0:
        _refuse("hint-shape", "block count is not a positive exact integer")
    if bounds is None or isinstance(bounds, (str, bytes)):
        _refuse(
            "hint-shape",
            "no cue timestamps were supplied; "
            + _untimed_remedy(sample_rate, max_dp_frames, frame_stride),
        )
    try:
        bound_count = len(bounds)
    except Exception:
        _refuse("hint-shape", "route bound vector is not sized")
    if bound_count != block_count:
        _refuse("hint-shape", "route bound vector does not match block count")

    # Latest start among the earlier cues; a cue must not end at or before it.
    latest_start: float | None = None
    latest_index = -1
    known = 0
    for index in range(bound_count):
        cue = index + 1
        try:
            pair = bounds[index]
        except Exception:
            _refuse("hint-shape", f"cue {cue} route bound is not indexable")
        if pair is None:
            continue
        if type(pair) not in (tuple, list) or len(pair) != 2:
            _refuse("hint-shape", f"cue {cue} route bound is not an exact pair")
        start, end = pair
        if not _is_exact_number(start) or not _is_exact_number(end):
            _refuse("hint-shape", f"cue {cue} route bound is not exact numeric data")
        start_value = float(start)
        end_value = float(end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            _refuse("hint-nonfinite", f"cue {cue} has a nonfinite timestamp")
        if start_value < 0.0 or end_value < start_value:
            _refuse(
                "hint-nonmonotone",
                f"cue {cue} has a negative start or ends before it starts",
            )
        if latest_start is not None and start_value < latest_start:
            if end_value <= latest_start:
                _refuse(
                    "hint-nonmonotone",
                    f"cue {cue} ({_clock(start_value)} --> {_clock(end_value)})"
                    f" comes after cue {latest_index + 1} in the file but ends"
                    f" at or before cue {latest_index + 1} starts"
                    f" ({_clock(latest_start)}); put the cues in time order",
                )
        else:
            latest_start = start_value
            latest_index = index
        known += 1

    if known == 0:
        _refuse(
            "hint-shape",
            "no cue has a timestamp; "
            + _untimed_remedy(sample_rate, max_dp_frames, frame_stride),
        )

    _require_positive_finite(audio_end, name="audio_end")
    _require_positive_finite(sample_rate, name="sample_rate")
    _require_positive_finite(max_dp_frames, name="max_dp_frames")
    _require_positive_finite(frame_stride, name="frame_stride")
    _require_positive_finite(chunk_fraction, name="chunk_fraction")


def route_hint_envelope(
    bounds: Sequence[tuple[float, float] | None],
) -> Sequence[tuple[float, float] | None]:
    """Widen validated route hints to their monotone envelope, in lexical order.

    Each known cue becomes ``(earliest start of it and every later cue, latest end
    of it and every earlier cue)``, so both starts and ends are nondecreasing and
    every cue still lies inside its own entry.  A positive gap of this envelope is
    silence for every cue, not just the two neighbours, which is where the planner
    prefers to cut.  When no such gap fits the budget the planner still cuts inside
    an overlap, so a plan made on widened hints must pass
    :func:`validate_widened_plans`.  ``None`` entries stay ``None``; hints that are
    already monotone are returned unchanged (the same object).
    """

    count = len(bounds)
    widened = False
    ends: list[float] = [0.0] * count
    latest_end: float | None = None
    for index in range(count):
        pair = bounds[index]
        if pair is None:
            continue
        end = pair[1]
        if latest_end is not None and end < latest_end:
            end = latest_end
            widened = True
        ends[index] = latest_end = end
    starts: list[float] = [0.0] * count
    earliest_start: float | None = None
    for index in reversed(range(count)):
        pair = bounds[index]
        if pair is None:
            continue
        start = pair[0]
        if earliest_start is not None and earliest_start < start:
            start = earliest_start
            widened = True
        starts[index] = earliest_start = start
    if not widened:
        return bounds
    return [
        None if bounds[index] is None else (starts[index], ends[index])
        for index in range(count)
    ]


def validate_over_budget_plans(
    plans: Any,
    *,
    block_count: int,
    audio_end: float,
    sample_count: int,
    sample_rate: int,
    max_dp_frames: int | float,
    frame_stride: int | float,
    chunk_fraction: float,
) -> None:
    """Independently verify the planner's index tiling and physical crop budget."""

    if type(plans) not in (list, tuple) or not plans:
        _refuse("plan-nontiling", "planner returned no ordered partition")

    cursor = 0
    for index, plan in enumerate(plans):
        if type(plan) is not dict:
            _refuse("plan-nontiling", f"plan {index} is not an exact object")
        lo = plan.get("lo")
        hi = plan.get("hi")
        if type(lo) is not int or type(hi) is not int:
            _refuse("plan-nontiling", f"plan {index} has non-integer ownership")
        if lo != cursor or hi <= lo or hi > block_count:
            _refuse("plan-nontiling", f"plan {index} does not extend the exact tiling")
        cursor = hi
    if cursor != block_count:
        _refuse("plan-nontiling", "planner output does not cover every block once")

    audio_end_value = _require_positive_finite(audio_end, name="audio_end")
    sample_rate_value = _require_positive_finite(sample_rate, name="sample_rate")
    max_frames_value = _require_positive_finite(max_dp_frames, name="max_dp_frames")
    stride_value = _require_positive_finite(frame_stride, name="frame_stride")
    fraction_value = _require_positive_finite(chunk_fraction, name="chunk_fraction")
    if type(sample_count) is not int or sample_count <= 0:
        _refuse("crop-geometry", "prepared audio has no positive exact sample count")

    previous_sample_end: int | None = None
    for index, plan in enumerate(plans):
        start = plan.get("start")
        end = plan.get("end")
        if not _is_exact_number(start) or not _is_exact_number(end):
            _refuse("crop-geometry", f"plan {index} crop is not exact numeric data")
        start_value = float(start)
        end_value = float(end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            _refuse("crop-geometry", f"plan {index} crop is nonfinite")
        if start_value < 0.0 or end_value <= start_value or end_value > audio_end_value:
            _refuse("crop-geometry", f"plan {index} crop is outside prepared audio")

        # This is the exact clamp used by the physical slicing loop.  Validation is
        # deliberately performed on the clamped integer samples, not idealized seconds.
        sample_start = max(0, int(start_value * sample_rate_value))
        sample_end = min(sample_count, int(end_value * sample_rate_value))
        if sample_start >= sample_end:
            _refuse("crop-geometry", f"plan {index} clamps to an empty crop")
        if previous_sample_end is not None and sample_start < previous_sample_end:
            _refuse("crop-geometry", f"plan {index} crop overlaps its predecessor")
        previous_sample_end = sample_end

        crop_frames = (sample_end - sample_start) / stride_value
        if crop_frames > max_frames_value * fraction_value:
            first, last = plan["lo"] + 1, plan["hi"]
            cues = f"cue {first}" if first == last else f"cues {first}-{last}"
            _refuse(
                "crop-over-budget",
                f"plan {index} ({cues}) exceeds the physical DP budget",
            )


def _cue_span(hints: Sequence[tuple[float, float] | None], index: int) -> str:
    pair = cast(tuple[float, float], hints[index])
    return f"cue {index + 1} ({_clock(float(pair[0]))} --> {_clock(float(pair[1]))})"


def _refuse_split(
    hints: Sequence[tuple[float, float] | None],
    boundary: int,
    time: float,
    cropped: int,
) -> NoReturn:
    """Refuse the split before 0-based cue ``boundary`` that crops cue ``cropped``."""

    # The latest-ending cue before the split and the earliest-starting one after it
    # are the pair whose overlap left no silence to split in.
    left: int | None = None
    right: int | None = None
    for index in range(len(hints)):
        pair = hints[index]
        if pair is None:
            continue
        if index < boundary:
            if left is None or pair[1] >= cast(tuple[float, float], hints[left])[1]:
                left = index
        elif right is None or pair[0] < cast(tuple[float, float], hints[right])[0]:
            right = index
    split = (
        f"splitting this long audio into alignment pieces between cues {boundary}"
        f" and {boundary + 1} (at {_clock(time)})"
    )
    if left is not None and right is not None:
        left_pair = cast(tuple[float, float], hints[left])
        right_pair = cast(tuple[float, float], hints[right])
        if right_pair[0] < left_pair[1]:
            first, second = sorted((left + 1, right + 1))
            _refuse(
                "hint-nonmonotone",
                f"{_cue_span(hints, right)} starts before {_cue_span(hints, left)}"
                f" ends, so {split} would cut a cue off from its own audio; check"
                f" the timestamps of cues {first} and {second}",
            )
    _refuse(
        "hint-nonmonotone",
        f"{split} would cut {_cue_span(hints, cropped)} off from part of its own"
        " audio; check the timestamps around it",
    )


def validate_widened_plans(
    plans: Sequence[dict[str, Any]],
    *,
    hints: Sequence[tuple[float, float] | None],
    envelope: Sequence[tuple[float, float] | None],
) -> None:
    """Refuse a plan made on widened hints that cuts a cue off from its own audio.

    Run after :func:`validate_over_budget_plans`.  A cue whose timestamps overlap
    cues far from it can leave no silence where a split has to fall; the planner
    then splits inside the overlap and would align whole cues against audio that
    does not contain them.  Every known cue must lie inside its own plan's crop
    (the last crop may stop at the end of the audio).  Hints that needed no
    widening keep their previous behaviour and are not checked here.
    """

    if envelope is hints:
        return
    last = len(plans) - 1
    for index, plan in enumerate(plans):
        lo, hi = plan["lo"], plan["hi"]
        crop_start, crop_end = float(plan["start"]), float(plan["end"])
        for cue in range(lo, hi):
            pair = hints[cue]
            if pair is None:
                continue
            if index > 0 and pair[0] < crop_start:
                _refuse_split(hints, lo, crop_start, cue)
            if index < last and pair[1] > crop_end:
                _refuse_split(hints, hi, crop_end, cue)


__all__ = [
    "DpRouteHintsInvalid",
    "route_hint_envelope",
    "validate_over_budget_hints",
    "validate_over_budget_plans",
    "validate_widened_plans",
]
