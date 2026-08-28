"""Fail-closed validation for over-budget CTC/MMS route hints and plans.

The silence-anchor planner is deliberately kept separate from this module.  These
checks neither sort route hints nor repair planner output: unsafe data is classified
and refused before a backend can construct its forced-alignment trellis.
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
    an exact two-number pair.  Known starts and ends remain in lexical source order;
    no timestamp-key sorting is performed here or by the caller.
    """

    if type(block_count) is not int or block_count <= 0:
        _refuse("hint-shape", "block count is not a positive exact integer")
    if bounds is None or isinstance(bounds, (str, bytes)):
        _refuse("hint-shape", "route bound vector is unavailable")
    try:
        bound_count = len(bounds)
    except Exception:
        _refuse("hint-shape", "route bound vector is not sized")
    if bound_count != block_count:
        _refuse("hint-shape", "route bound vector does not match block count")

    previous_start: float | None = None
    previous_end: float | None = None
    known = 0
    for index in range(bound_count):
        try:
            pair = bounds[index]
        except Exception:
            _refuse("hint-shape", f"route bound {index} is not indexable")
        if pair is None:
            continue
        if type(pair) not in (tuple, list) or len(pair) != 2:
            _refuse("hint-shape", f"route bound {index} is not an exact pair")
        start, end = pair
        if not _is_exact_number(start) or not _is_exact_number(end):
            _refuse("hint-shape", f"route bound {index} is not exact numeric data")
        start_value = float(start)
        end_value = float(end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            _refuse("hint-nonfinite", f"route bound {index} is nonfinite")
        if start_value < 0.0 or end_value < start_value:
            _refuse("hint-nonmonotone", f"route bound {index} has unsafe geometry")
        if previous_start is not None and start_value < previous_start:
            _refuse("hint-nonmonotone", "known starts are not nondecreasing")
        if previous_end is not None and end_value < previous_end:
            _refuse("hint-nonmonotone", "known ends are not nondecreasing")
        previous_start = start_value
        previous_end = end_value
        known += 1

    if known == 0:
        _refuse("hint-shape", "route bounds have no usable first start and last end")

    _require_positive_finite(audio_end, name="audio_end")
    _require_positive_finite(sample_rate, name="sample_rate")
    _require_positive_finite(max_dp_frames, name="max_dp_frames")
    _require_positive_finite(frame_stride, name="frame_stride")
    _require_positive_finite(chunk_fraction, name="chunk_fraction")


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
            _refuse("crop-over-budget", f"plan {index} exceeds the physical DP budget")


__all__ = [
    "DpRouteHintsInvalid",
    "validate_over_budget_hints",
    "validate_over_budget_plans",
]
