"""plan_dp_chunks budgets the crop it emits, not just the cue span.

The over-budget full-file DP runs over each plan's ``[start, end]`` crop, and
align_dp_safety.validate_over_budget_plans refuses any crop longer than the budget.
The crop reaches past the cues (back to the previous gap midpoint / the left pad,
forward to the next gap midpoint / ``audio_end``), so measuring only the cue span
produced plans the validator refused for exactly the long media the planner exists for.
"""

import random

import pytest

from voxweave.align_dp_safety import DpRouteHintsInvalid, validate_over_budget_plans
from voxweave.chunking import plan_dp_chunks

SR = 16000
STRIDE = 320
FRAC = 0.8


def _budget_sec(max_frames: int) -> float:
    # the exact expression align_common._prepare_dp_calls hands the planner
    return max_frames * STRIDE / SR * FRAC


def _validate(plans, bounds, max_frames: int, audio_end: float) -> None:
    validate_over_budget_plans(
        plans,
        block_count=len(bounds),
        audio_end=audio_end,
        sample_count=int(audio_end * SR),
        sample_rate=SR,
        max_dp_frames=max_frames,
        frame_stride=STRIDE,
        chunk_fraction=FRAC,
    )


def _single_cue_fits(bounds, budget: float, audio_end: float, pad: float) -> bool:
    """False when some cue's tightest possible crop (cut right before and after it) is
    already over budget: no partition can satisfy the validator then."""
    idx = [j for j, b in enumerate(bounds) if b is not None]
    for p, j in enumerate(idx):
        s, e = bounds[j]
        if p == 0:
            left = max(0.0, s - pad)
        else:
            prev = idx[p - 1]
            left = (bounds[prev][1] + s) / 2 if prev == j - 1 else bounds[prev][1] + pad
        if p == len(idx) - 1:
            right = min(audio_end, e + pad)
        else:
            nxt = idx[p + 1]
            right = (e + bounds[nxt][0]) / 2 if nxt == j + 1 else e + pad
        if right - left > budget:
            return False
    return True


def test_long_silences_between_cues_stay_within_budget():
    # every 10th gap is 30s of silence: the crop runs to its midpoint, 15s past the cue
    bounds, t = [], 10.0
    for i in range(200):
        bounds.append((t, t + 4.0))
        t += 4.0 + (30.0 if i % 10 == 9 else 2.0)
    audio_end = t + 5.0
    plans = plan_dp_chunks(bounds, max_sec=_budget_sec(12000), audio_end=audio_end)
    assert len(plans) > 1
    assert all(p["end"] - p["start"] <= _budget_sec(12000) for p in plans)
    _validate(plans, bounds, 12000, audio_end)


def test_trailing_audio_is_capped_to_the_last_cue_when_it_would_overflow():
    # 5 minutes of credits after the last cue: the final crop may not run to audio_end
    bounds = [(10.0 + i * 6, 14.0 + i * 6) for i in range(100)]
    audio_end = bounds[-1][1] + 300.0
    budget = _budget_sec(12000)
    plans = plan_dp_chunks(bounds, max_sec=budget, audio_end=audio_end)
    _validate(plans, bounds, 12000, audio_end)
    assert plans[-1]["end"] == bounds[-1][1] + 0.5  # pad_sec past the last cue


def test_final_chunk_keeps_audio_end_when_it_fits():
    bounds = [(10.0 + i * 6, 14.0 + i * 6) for i in range(100)]
    audio_end = bounds[-1][1] + 3.0
    plans = plan_dp_chunks(bounds, max_sec=_budget_sec(12000), audio_end=audio_end)
    _validate(plans, bounds, 12000, audio_end)
    assert plans[-1]["end"] == audio_end


def test_single_chunk_output_unchanged_when_the_crop_fits():
    bounds = [(3.0, 5.0), None, (6.0, 9.0)]
    assert plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5, audio_end=12.0) == [
        {"lo": 0, "hi": 3, "start": 2.5, "end": 12.0}
    ]


@pytest.mark.parametrize("seed", range(8))
def test_every_feasible_plan_passes_the_dp_validator(seed):
    rng = random.Random(seed)
    checked = 0
    for _ in range(300):
        max_frames = rng.choice([3000, 6000, 12000])
        budget = _budget_sec(max_frames)
        t = rng.uniform(0.0, 30.0)
        bounds: list[tuple[float, float] | None] = []
        for _ in range(rng.randint(1, 150)):
            if rng.random() < 0.08:
                bounds.append(None)  # untimed insertion cue rides along
                continue
            dur = rng.uniform(0.2, 10.0)
            bounds.append((t, t + dur))
            t += dur + rng.choice([0.0, 0.05, 0.3, 1.0, 2.0, 5.0, 15.0, 40.0])
        known = [b for b in bounds if b is not None]
        if not known:
            continue
        audio_end = known[-1][1] + rng.choice([0.01, 0.3, 1.0, 30.0, 300.0])
        plans = plan_dp_chunks(bounds, max_sec=budget, audio_end=audio_end)
        # the index tiling always holds, feasible or not
        assert [p["lo"] for p in plans] == [0, *[p["hi"] for p in plans[:-1]]]
        assert plans[-1]["hi"] == len(bounds)
        if not _single_cue_fits(bounds, budget, audio_end, 0.5):
            continue  # no partition at cue boundaries can meet the budget
        try:
            _validate(plans, bounds, max_frames, audio_end)
        except DpRouteHintsInvalid as exc:  # pragma: no cover - failure detail
            pytest.fail(f"{exc}: budget={budget} audio_end={audio_end} plans={plans}")
        checked += 1
    assert checked > 200
