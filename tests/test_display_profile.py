# tests/test_display_profile.py
# wrap_cue_text reads DEFAULT_MAX_LINE_LENGTH (42 half-width cells) directly for
# its no-wrap gate, its two-line balance search and the sticky-end slide, so a
# caller cannot render for a narrower (or wider) player. These tests pin the
# max_line_length parameter: an explicit budget must be honoured by every stage,
# and None must reproduce the current default layout exactly.
import pytest

from voxweave.core.layout import (
    DEFAULT_MAX_LINE_LENGTH,
    _join_line,
    _two_line_break,
    _two_line_break_index,
    _vis_width,
    _wrap_units,
    wrap_cue_text,
)

# Current (default-profile) output, captured from the 42-column renderer.
DEFAULT_LAYOUTS = [
    (
        "the quick brown fox jumps over the lazy dog near the river bank",
        "en",
        2,
        "the quick brown fox jumps\nover the lazy dog near the river bank",
    ),
    (
        "she opened the heavy wooden door and stepped into the cold morning air outside",
        "en",
        2,
        "she opened the heavy wooden door and\nstepped into the cold morning air outside",
    ),
    ("we should go to the store now", "en", 2, "we should go to the store now"),
    (
        (
            "the committee will announce the final results tomorrow morning"
            " after the vote is counted and verified"
        ),
        "en",
        3,
        (
            "the committee will announce\nthe final results tomorrow morning"
            "\nafter the vote is counted and verified"
        ),
    ),
    (
        "私たちは明日の朝早くから新しいプロジェクトの準備を始めることになりました",
        "ja",
        2,
        "私たちは明日の朝早くから新しいプロジェ\nクトの準備を始めることになりました",
    ),
]


def _lines(rendered):
    return rendered.split("\n")


@pytest.mark.parametrize(("text", "lang", "max_lines", "expected"), DEFAULT_LAYOUTS)
def test_none_preserves_default_layout(text, lang, max_lines, expected):
    # Regression lock: the new parameter must not move the default renderer.
    assert wrap_cue_text(text, lang, max_lines, max_line_length=None) == expected
    assert wrap_cue_text(text, lang, max_lines, max_line_length=None) == wrap_cue_text(
        text, lang, max_lines
    )


def test_narrow_profile_wraps_within_budget():
    # ~26 half-width cells: one line under the 42-column default, two lines that
    # each fit when the caller asks for a 20-column player.
    text = "we should go to the store"
    assert _vis_width(text) < DEFAULT_MAX_LINE_LENGTH
    rendered = wrap_cue_text(text, "en", 2, max_line_length=20)
    lines = _lines(rendered)
    assert len(lines) == 2, rendered
    assert all(_vis_width(line) <= 20 for line in lines), rendered
    assert " ".join(rendered.split()) == text


def test_two_line_break_follows_profile():
    # Fits 42 (so the default renderer emits one line); under a 20-column budget
    # the balance search must find a different break instead of the 42-cell one.
    text = "we should go to the store now"
    default = wrap_cue_text(text, "en", 2)
    assert "\n" not in default
    narrow = wrap_cue_text(text, "en", 2, max_line_length=20)
    assert narrow != default
    lines = _lines(narrow)
    assert len(lines) == 2, narrow
    assert all(_vis_width(line) <= 20 for line in lines), narrow


def test_narrow_profile_applies_to_greedy_wrap():
    # 3+ lines take the greedy balance path plus the sticky-end slide; both must
    # respect the passed budget rather than the 42-cell constant.
    text = "we can go there now and then come back"
    rendered = wrap_cue_text(text, "en", 3, max_line_length=16)
    lines = _lines(rendered)
    assert len(lines) <= 3, rendered
    assert all(_vis_width(line) <= 16 for line in lines), rendered
    assert " ".join(rendered.split()) == text


def test_wide_profile_skips_wrapping():
    # The no-wrap gate must use the passed budget too: a cue that wraps at 42
    # stays on one line for a player with an 80-column budget.
    text, lang, max_lines, default = DEFAULT_LAYOUTS[1][:4]
    assert "\n" in default
    rendered = wrap_cue_text(text, lang, max_lines, max_line_length=80)
    assert rendered == text


# ------------------------------------------- the two-line chooser's own widths


@pytest.mark.parametrize(
    "text,lang,budget",
    [
        ("voice assistant and it's training for the custom", "en", 42),
        ("alpha bravo charlie delta echo foxtrot golf hotel", "en", 24),
        ("a bb ccc dddd eeeee ffffff", "en", 12),
        ("これはテストですこんにちは世界今日はいい天気", "ja", 20),
        ("你好世界这是一个测试今天天气很好我们一起走吧", "zh", 18),
    ],
)
def test_two_line_break_widths_match_the_joined_slices(text, lang, budget):
    """The prefix-sum rewrite must be the same function, not merely similar.

    ``_two_line_break`` reports the chosen line widths so the boundary optimizer
    can price the lines the renderer will deliver; those widths are computed from
    prefix sums instead of re-joining a slice per candidate, which is what makes
    the scan linear. ``_vis_width`` is a per-character sum, so the two readings
    are the same integer -- pinned here rather than assumed.
    """
    units = _wrap_units(text, lang)
    chosen = _two_line_break(units, lang, budget)
    assert (None if chosen is None else chosen[0]) == _two_line_break_index(
        units, lang, budget
    )
    if chosen is None:
        return
    index, top, bottom = chosen
    assert top == _vis_width(_join_line(units[:index]))
    assert bottom == _vis_width(_join_line(units[index:]))
