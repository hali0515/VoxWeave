"""Characterization tests pinning the scoring helpers to the original inline code."""

from __future__ import annotations

import pytest

from voxweave.core.smart_split import micro_fragment_damage, pause_damage


def _ref_pause(left_end, right_start):
    gap = 0
    if isinstance(left_end, (int, float)) and isinstance(right_start, (int, float)):
        pause_ms = max(0.0, (float(right_start) - float(left_end)) * 1000)
        if pause_ms < 40:
            gap = 3
        elif pause_ms < 120:
            gap = 2
        elif pause_ms < 220:
            gap = 1
    return gap


def _ref_micro(left_width, right_width):
    return (
        8
        if min(left_width, right_width) <= 1
        else 4
        if min(left_width, right_width) <= 2
        else 0
    )


def _ref_micro_small(left_width, right_width):
    small = min(left_width, right_width)
    return 8 if small <= 1 else 4 if small <= 2 else 0


BOUNDARY_MS = [0, 39, 40, 41, 119, 120, 121, 219, 220, 221]


@pytest.mark.parametrize("ms", BOUNDARY_MS)
@pytest.mark.parametrize("start", [0, 1.5, 12.25])
def test_pause_damage_matches_inline_ladder(ms, start):
    end = start + ms / 1000
    assert pause_damage(start, end) == _ref_pause(start, end)
    assert type(pause_damage(start, end)) is int


@pytest.mark.parametrize("ms", BOUNDARY_MS)
def test_pause_damage_overlap_clamped(ms):
    assert pause_damage(5.0 + ms / 1000, 5.0) == _ref_pause(5.0 + ms / 1000, 5.0)
    assert pause_damage(5.0 + ms / 1000, 5.0) == 3


@pytest.mark.parametrize(
    ("left", "right"),
    [(None, 1.0), (1.0, None), (None, None), ("1", 2.0), (1.0, "2")],
)
def test_pause_damage_untimed_is_zero(left, right):
    assert pause_damage(left, right) == _ref_pause(left, right) == 0


@pytest.mark.parametrize("left", [0, 1, 2, 3, 10])
@pytest.mark.parametrize("right", [0, 1, 2, 3, 10])
def test_micro_fragment_damage_matches_inline(left, right):
    got = micro_fragment_damage(left, right)
    assert got == _ref_micro(left, right) == _ref_micro_small(left, right)
    assert type(got) is int
