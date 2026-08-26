# tests/test_timing_preview.py
"""``LegacyCleanupPreview`` must equal what ``_cleanup_cues`` actually produces.

Every test here builds a real cue stream, runs the real timing pass over it, and
compares the pass's resulting duration against the preview -- exactly, not to a
tolerance, because the preview mirrors the pass statement by statement. A
discrepancy is a mirror bug: fix ``timing_preview``, never this file and never
``timing``.
"""

import random

import pytest

from voxweave.core.timing import (
    CHAIN_MAX_GAP_S,
    HELD_WORD_MAX_GAP_S,
    LINGER_CAP_S,
    TWO_FRAME_S,
    _cleanup_cues,
)
from voxweave.core.timing_preview import DisplayTimingPreview, LegacyCleanupPreview

PREVIEW = LegacyCleanupPreview()

# The evidence-map probe profile (semantic-salvage.md 7.3).
PROFILE = {"min_cue_s": 0.5, "max_cue_s": 7.0, "cps": 17.0, "lag_out_s": 0.25}


def _cue(text, start, end, words=None):
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": list(words or []),
    }


def _words(*spans):
    """Timed units from (start, end) pairs; ``None`` bounds pass through."""
    return [{"text": "w", "start": s, "end": e} for s, e in spans]


def _check(cues, **thresholds):
    """Run the real pass and the preview over the same stream; return durations.

    Asserts equality for every cue in the stream, so a 3-cue case exercises the
    middle cue against its real neighbours.
    """
    out = _cleanup_cues([dict(c) for c in cues], **thresholds)
    actual = [o["end"] - c["start"] for o, c in zip(out, cues)]
    predicted = [
        PREVIEW.preview_display_span(
            c["start"],
            c["end"],
            cues[i + 1]["start"] if i + 1 < len(cues) else None,
            text=c["text"],
            word_data=c["word_data"],
            **thresholds,
        )
        for i, c in enumerate(cues)
    ]
    assert predicted == actual
    return actual


def test_implements_the_protocol():
    preview: DisplayTimingPreview = LegacyCleanupPreview()
    assert isinstance(preview, DisplayTimingPreview)


# --- no next / far next -------------------------------------------------------


def test_no_next_cue():
    # evidence case A: spoken 0.3, lag-out wins over the tiny reading need
    got = _check([_cue("hello", 1.0, 1.3, _words((1.0, 1.3)))], **PROFILE)
    assert got[0] == pytest.approx(0.55)


def test_far_next_cue():
    # evidence case A2: a next cue 20s away cannot constrain anything
    got = _check(
        [_cue("hello", 1.0, 1.3, _words((1.0, 1.3))), _cue("b", 21.0, 21.4)],
        **PROFILE,
    )
    assert got[0] == pytest.approx(0.55)


def test_long_text_reading_linger_capped():
    # evidence case B: need = 21/17 = 1.235 beats the lag-out pad, far next
    got = _check(
        [_cue("x" * 21, 1.0, 1.7, _words((1.0, 1.7))), _cue("b", 21.0, 21.4)],
        **PROFILE,
    )
    assert got[0] == pytest.approx(1.235294117647059)


def test_linger_cap_binds_when_reading_need_is_huge():
    # need = 100/17 = 5.88s would reach 6.88s; the cap holds it to end + 1.0
    got = _check([_cue("x" * 100, 1.0, 1.4, _words((1.0, 1.4)))], **PROFILE)
    assert got[0] == pytest.approx(1.4 + LINGER_CAP_S - 1.0)


# --- chaining band ------------------------------------------------------------


def test_chaining_band_wide_gap():
    # evidence case C: gap 0.40 -> chained to next - 2 frames
    got = _check(
        [_cue("a", 1.0, 2.2), _cue("b", 2.6, 3.0)],
        **PROFILE,
    )
    assert got[0] == pytest.approx(2.6 - TWO_FRAME_S - 1.0)


def test_chaining_band_narrow_gap():
    # evidence case C2: gap 0.30 -> the lag-out extension eats it down to 0.05,
    # which is already under the 2-frame floor, so chaining does not fire
    got = _check([_cue("a", 1.0, 2.2), _cue("b", 2.5, 3.0)], **PROFILE)
    assert got[0] == pytest.approx(1.45)


def test_gap_just_below_chain_max_still_chains():
    _check(
        [_cue("a", 1.0, 2.2), _cue("b", 2.2 + CHAIN_MAX_GAP_S - 0.01, 3.5)],
        **PROFILE,
    )


def test_gap_at_chain_max_stays_visible():
    _check([_cue("a", 1.0, 2.2), _cue("b", 2.2 + CHAIN_MAX_GAP_S, 3.5)], **PROFILE)


def test_extension_clamps_to_next_start():
    # evidence case D: gap 0.10 -> the extension wants 2.45 and clamps to 2.30
    # (cleanup's ``min(want, nxt_start)``, not next - 2 frames)
    got = _check([_cue("a", 1.0, 2.2), _cue("b", 2.3, 3.0)], **PROFILE)
    assert got[0] == pytest.approx(1.3)


# --- two-frame band -----------------------------------------------------------


def test_two_frame_band_blocks_extension():
    # evidence case E: gap 0.05 <= 2 frames -> no extension, no chaining
    got = _check([_cue("a", 1.0, 2.2), _cue("b", 2.25, 3.0)], **PROFILE)
    assert got[0] == pytest.approx(1.2)


def test_gap_at_the_two_frame_boundary():
    # 2.2 + TWO_FRAME_S - 2.2 is not exactly TWO_FRAME_S in binary floats, so this
    # gap lands on whichever side the pass's own comparison puts it. No anchor:
    # the point is that the mirror copies the comparison instead of rounding it.
    _check([_cue("a", 1.0, 2.2), _cue("b", 2.2 + TWO_FRAME_S, 3.0)], **PROFILE)


def test_overlapping_next_start_leaves_end_alone():
    # gap < 0 is left to the caller by the pass; the mirror must agree
    _check([_cue("a", 1.0, 2.2), _cue("b", 2.0, 3.0)], **PROFILE)


# --- cps on / off -------------------------------------------------------------


def test_cps_off():
    # evidence case F
    got = _check(
        [_cue("hello", 1.0, 1.3, _words((1.0, 1.3)))],
        min_cue_s=0.5,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.25,
    )
    assert got[0] == pytest.approx(0.55)


def test_cps_on_drives_the_extension():
    # need = 17/17 = 1.0s from the cue start, still inside the linger cap
    got = _check(
        [_cue("x" * 17, 1.0, 1.3, _words((1.0, 1.3)))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=17.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(1.0)


def test_cps_with_empty_text_is_a_noop():
    _check(
        [_cue("", 1.0, 1.3, _words((1.0, 1.3)))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=17.0,
        lag_out_s=0.0,
    )


# --- min_cue floors -----------------------------------------------------------


def test_min_cue_floor_alone():
    # evidence case G
    got = _check(
        [_cue("hello", 1.0, 1.3, _words((1.0, 1.3)))],
        min_cue_s=2.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(2.0)


def test_min_cue_floor_clamped_by_next_start():
    got = _check(
        [_cue("a", 1.0, 1.3), _cue("b", 2.0, 2.5)],
        min_cue_s=2.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(1.0)  # min(want, nxt_start), not next - 2 frames


def test_min_cue_zero_keeps_short_cue():
    got = _check(
        [_cue("a", 1.0, 1.2)],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(0.2)


# --- lag-out anchor: timed vs untimed word_data -------------------------------


def test_lag_out_untimed_word_data_pads_from_display_end():
    got = _check(
        [_cue("a", 1.0, 2.0)],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.25,
    )
    assert got[0] == pytest.approx(1.25)


def test_lag_out_timed_word_data_pads_from_speech_end():
    # display end already sits past speech_end + pad -> no growth (idempotence)
    got = _check(
        [_cue("a", 1.0, 2.0, _words((1.0, 1.5)))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.25,
    )
    assert got[0] == pytest.approx(1.0)


def test_lag_out_half_timed_word_data():
    # a unit missing its end contributes nothing to the anchor
    got = _check(
        [_cue("a", 1.0, 2.0, _words((1.0, None), (1.2, 1.9)))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.25,
    )
    assert got[0] == pytest.approx(1.15)


# --- max_cue_s cap ------------------------------------------------------------


def test_cap_without_held_chain_untimed():
    got = _check(
        [_cue("a", 1.0, 9.0)],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(7.0)


def test_cap_without_held_chain_words_stop_before_cap():
    got = _check(
        [_cue("a", 1.0, 9.0, _words((1.0, 6.0), (6.1, 7.5)))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(7.0)


def test_held_word_waiver_keeps_cue_past_cap():
    # evidence case H: continuous words run to 7.45s past a 7.0s cap
    spans = [(0.0 + i * 0.5, 0.45 + i * 0.5) for i in range(15)]
    got = _check(
        [_cue("a", 0.0, 7.45, _words(*spans))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(7.45)


def test_held_word_waiver_clamped_by_next_start():
    spans = [(0.0 + i * 0.5, 0.45 + i * 0.5) for i in range(15)]
    got = _check(
        [_cue("a", 0.0, 7.45, _words(*spans)), _cue("b", 7.2, 8.0)],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(7.2)


def test_held_word_broken_chain_lets_the_cap_win():
    # a stray syllable past a silence wider than HELD_WORD_MAX_GAP_S
    got = _check(
        [
            _cue(
                "a",
                0.0,
                7.45,
                _words(
                    (0.0, 3.0), (3.05, 6.0), (6.0 + HELD_WORD_MAX_GAP_S + 0.3, 7.45)
                ),
            )
        ],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(7.0)


def test_held_chain_walks_words_in_start_order():
    # word_data out of order: the pass sorts before walking, so must the mirror
    spans = [(0.0 + i * 0.5, 0.45 + i * 0.5) for i in range(15)]
    got = _check(
        [_cue("a", 0.0, 7.45, _words(*reversed(spans)))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(7.45)


def test_held_chain_sorts_by_start_not_by_end():
    # a long word overlapping a short one: ordering by start ends the walk on the
    # short word (cap wins), ordering by end would end it on the long one (7.45)
    got = _check(
        [_cue("a", 0.0, 7.45, _words((0.0, 7.45), (6.0, 6.2)))],
        min_cue_s=0.0,
        max_cue_s=7.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(7.0)


def test_max_cue_zero_disables_the_cap():
    got = _check(
        [_cue("a", 1.0, 12.0)],
        min_cue_s=0.0,
        max_cue_s=0.0,
        cps=0.0,
        lag_out_s=0.0,
    )
    assert got[0] == pytest.approx(11.0)


# --- multi-cue streams --------------------------------------------------------


def test_three_cue_stream_middle_cue_matches():
    cues = [
        _cue("first", 1.0, 1.4, _words((1.0, 1.4))),
        _cue("middle", 1.6, 2.1, _words((1.6, 2.1))),
        _cue("last", 2.4, 3.9, _words((2.4, 3.9))),
    ]
    got = _check(cues, **PROFILE)
    assert got[1] == pytest.approx(0.75)  # lag-out from speech_end 2.1, clamped


def test_dense_stream_every_cue_matches():
    cues = [
        _cue(
            f"cue {i}",
            1.0 + i * 0.9,
            1.0 + i * 0.9 + 0.55,
            _words((1.0 + i * 0.9, 1.0 + i * 0.9 + 0.55)),
        )
        for i in range(8)
    ]
    _check(cues, **PROFILE)


# --- randomized sweep ---------------------------------------------------------

_GAPS = [
    -0.2,
    0.0,
    0.02,
    TWO_FRAME_S,
    TWO_FRAME_S + 1e-6,
    0.12,
    0.3,
    0.49,
    CHAIN_MAX_GAP_S,
    0.8,
    2.0,
    20.0,
]
_MIN_CUE = [0.0, 0.5, 1.0, 2.0]
_MAX_CUE = [0.0, 2.0, 7.0]
_CPS = [0.0, 9.0, 17.0, 30.0]
_LAG_OUT = [0.0, 0.15, 0.25, 0.8]


def _random_words(rng, start, end):
    kind = rng.choice(("none", "empty", "flat", "chain", "held", "ragged"))
    if kind == "none":
        return []
    if kind == "empty":
        return _words((None, None))
    if kind == "flat":
        return _words((start, end))
    if kind == "chain":
        n = rng.randint(2, 6)
        step = (end - start) / n
        return _words(*[(start + i * step, start + (i + 1) * step) for i in range(n)])
    if kind == "held":
        # words continuing past the display end, sometimes across a wide silence
        n = rng.randint(2, 8)
        gap = rng.choice((0.0, 0.4, HELD_WORD_MAX_GAP_S, HELD_WORD_MAX_GAP_S + 0.5))
        out, cursor = [], start
        for _ in range(n):
            out.append((cursor, cursor + 0.5))
            cursor += 0.5 + gap
        return _words(*out)
    ragged = [(start, None), (None, end), (start + 0.1, start + 0.2)]
    rng.shuffle(ragged)
    return _words(*ragged)


@pytest.mark.parametrize("case", range(200))
def test_random_sweep_matches_cleanup(case):
    rng = random.Random(f"timing-preview:{case}")
    n = rng.randint(1, 3)
    cues, cursor = [], round(rng.uniform(0.0, 5.0), 3)
    for _ in range(n):
        dur = rng.choice((0.12, 0.3, 0.9, 1.6, 3.4, 7.4, 9.0))
        text = "x" * rng.randint(0, 40)
        if rng.random() < 0.3:
            text = " ".join(["word"] * rng.randint(1, 8))
        cues.append(
            _cue(text, cursor, cursor + dur, _random_words(rng, cursor, cursor + dur))
        )
        cursor = cursor + dur + rng.choice(_GAPS)
    _check(
        cues,
        min_cue_s=rng.choice(_MIN_CUE),
        max_cue_s=rng.choice(_MAX_CUE),
        cps=rng.choice(_CPS),
        lag_out_s=rng.choice(_LAG_OUT),
    )
