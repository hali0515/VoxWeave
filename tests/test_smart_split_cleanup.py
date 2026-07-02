# tests/test_smart_split_cleanup.py
import pytest

from voxweave.core.timing import HELD_WORD_MAX_GAP_S, TWO_FRAME_S, _cleanup_cues


def test_min_duration_extends_into_following_gap():
    cues = [
        {"text": "はい", "start": 1.0, "end": 1.2, "word_data": []},
        {"text": "次", "start": 3.0, "end": 3.5, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.5, max_cue_s=7.0)
    assert out[0]["end"] - out[0]["start"] >= 0.5 - 1e-9
    assert out[0]["end"] <= out[1]["start"]  # no overlap


def test_min_duration_zero_keeps_real_short():
    cues = [
        {"text": "はい", "start": 1.0, "end": 1.2, "word_data": []},
        {"text": "次", "start": 3.0, "end": 3.5, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == 1.2  # not padded


def test_chain_small_gap_to_two_frames():
    # adjacent gap 0.2s (3-11 frame dead zone) -> chained down to ~2 frames (0.083s)
    cues = [
        {"text": "a", "start": 1.0, "end": 1.4, "word_data": []},
        {"text": "b", "start": 1.6, "end": 2.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0)
    assert out[1]["start"] - out[0]["end"] <= 0.084 + 1e-6


def test_big_real_gap_kept_visible():
    cues = [
        {"text": "a", "start": 1.0, "end": 1.4, "word_data": []},
        {"text": "b", "start": 3.0, "end": 3.4, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0)
    assert (
        out[1]["start"] - out[0]["end"] > 1.0
    )  # >=1s real pause: visible gap preserved


def test_min_duration_last_cue_extends_freely():
    cues = [{"text": "fin", "start": 5.0, "end": 5.1, "word_data": []}]
    out = _cleanup_cues(cues, min_cue_s=0.5, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(5.5)  # no successor -> extend freely to want


def test_lag_out_pads_into_gap():
    cues = [
        {"text": "hello there", "start": 0.0, "end": 2.0, "word_data": []},
        {"text": "next", "start": 4.0, "end": 5.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0, lag_out_s=0.25)
    assert out[0]["end"] == pytest.approx(2.25)


def test_lag_out_capped_by_next_start():
    cues = [
        {"text": "hello", "start": 0.0, "end": 2.0, "word_data": []},
        {"text": "next", "start": 2.1, "end": 3.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0, lag_out_s=0.25)
    assert out[0]["end"] <= out[1]["start"] + 1e-9


def test_cps_linger_extends_flash_cue_with_cap():
    # 20 chars at cps 10 want 2.0s; natural span 0.8s; gap available -> extends,
    # but never more than LINGER_CAP_S (1.0) past speech end.
    cues = [
        {"text": "a" * 20, "start": 0.0, "end": 0.8, "word_data": []},
        {"text": "next", "start": 5.0, "end": 6.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0, cps=10.0)
    assert out[0]["end"] == pytest.approx(1.8)


def test_cps_no_change_when_reading_time_already_met():
    # slow speech: natural duration exceeds chars/cps -> timing untouched
    cues = [
        {"text": "ab", "start": 0.0, "end": 3.0, "word_data": []},
        {"text": "next", "start": 6.0, "end": 7.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0, cps=10.0)
    assert out[0]["end"] == pytest.approx(3.0)


def test_cps_and_lag_defaults_off_keep_timing():
    cues = [
        {"text": "hello", "start": 0.0, "end": 2.0, "word_data": []},
        {"text": "next", "start": 4.0, "end": 5.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(2.0)


def test_cleanup_does_not_exceed_max_cue():
    # 6.9s cue + 0.3s gap -> chaining would extend to ~7.12s; must be clamped back to 7.0 by max_cue_s
    cues = [
        {"text": "a", "start": 0.0, "end": 6.9, "word_data": []},
        {"text": "b", "start": 7.2, "end": 8.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.5, max_cue_s=7.0)
    assert out[0]["end"] - out[0]["start"] <= 7.0 + 1e-9


# --- Defect B: max_cue_s clamp must not truncate a still-sounding word -------


def _held_cue(next_start: float):
    # 'pearl My pearl' held/sung: last word_data end (11.7) is 4.7s past
    # start+max_cue_s (7.0). Clamping to 7.0 would drop the subtitle 4.7s before
    # the vocal ends.
    return [
        {
            "text": "pearl My pearl",
            "start": 0.0,
            "end": 11.7,
            "word_data": [
                {"word": "pearl", "start": 0.0, "end": 1.0},
                {"word": "My", "start": 1.0, "end": 2.0},
                {"word": "pearl", "start": 2.0, "end": 11.7},
            ],
        },
        {"text": "next", "start": next_start, "end": next_start + 1.0, "word_data": []},
    ]


def test_max_cue_extends_to_last_word_when_held():
    # (a) last word ends 4.7s past the cap, next cue far away -> end extends all
    # the way to the last word end (speech must not vanish mid-utterance).
    out = _cleanup_cues(_held_cue(next_start=30.0), min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(11.7)


def test_max_cue_extension_capped_by_next_start():
    # (b) same held cue but the next cue starts 1s after the cap -> extend only up
    # to next.start (never overlap the following cue).
    out = _cleanup_cues(_held_cue(next_start=8.0), min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(8.0)


def test_max_cue_clamps_when_words_end_before_cap():
    # (c) word_data ends before the cap -> ordinary clamp to start+max_cue_s (a
    # long-linger cue whose own words already stopped is still capped).
    cues = [
        {
            "text": "hello",
            "start": 0.0,
            "end": 10.0,
            "word_data": [{"word": "hello", "start": 0.0, "end": 5.0}],
        },
        {"text": "next", "start": 30.0, "end": 31.0, "word_data": []},
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(7.0)


# --- P2-1: _cleanup_cues must be idempotent (diarize runs it twice) ----------


def _dense_chained_stream():
    # Fully-dense stream whose first cleanup pass produces a *chained two-frame*
    # gap (A->B) plus a sub-frame gap (B->C), with the last cue clamped by
    # max_cue_s. Every source of non-idempotency here is the gap interaction, so
    # after the fix cleanup(cleanup(x)) == cleanup(x) exactly.
    return [
        {
            "text": "a",
            "start": 0.0,
            "end": 1.0,
            "word_data": [{"start": 0.0, "end": 1.0}],
        },
        {
            "text": "b",
            "start": 1.45,
            "end": 2.0,
            "word_data": [{"start": 1.45, "end": 2.0}],
        },
        {
            "text": "c",
            "start": 2.3,
            "end": 9.5,
            "word_data": [{"start": 2.3, "end": 9.5}],
        },
    ]


def test_cleanup_is_idempotent_over_chained_gaps():
    # (a) IDEMPOTENCY PROPERTY: a chained two-frame gap survives a second pass.
    kw = dict(min_cue_s=0.0, max_cue_s=7.0, lag_out_s=0.25)
    once = _cleanup_cues(_dense_chained_stream(), **kw)
    # precondition: the first pass really does mint a chained two-frame gap.
    assert once[1]["start"] - once[0]["end"] == pytest.approx(TWO_FRAME_S)
    twice = _cleanup_cues(once, **kw)
    assert len(twice) == len(once)
    for a, b in zip(once, twice):
        assert a["text"] == b["text"]
        assert a["start"] == pytest.approx(b["start"])
        assert a["end"] == pytest.approx(b["end"])


def test_two_frame_gap_in_input_survives_cleanup():
    # (b) a two-frame gap present in the input is left intact by lag-out.
    cues = [
        {
            "text": "a",
            "start": 0.0,
            "end": 1.0,
            "word_data": [{"start": 0.0, "end": 1.0}],
        },
        {
            "text": "b",
            "start": 1.0 + TWO_FRAME_S,
            "end": 2.0,
            "word_data": [{"start": 1.0 + TWO_FRAME_S, "end": 2.0}],
        },
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0, lag_out_s=0.25)
    assert out[1]["start"] - out[0]["end"] == pytest.approx(TWO_FRAME_S)


def test_gap_above_two_frames_still_lags_out():
    # (c) a gap wider than the 2-frame floor still gets the normal flat lag-out
    # pad (existing behavior must not regress).
    cues = [
        {
            "text": "a",
            "start": 0.0,
            "end": 2.0,
            "word_data": [{"start": 0.0, "end": 2.0}],
        },
        {
            "text": "b",
            "start": 4.0,
            "end": 5.0,
            "word_data": [{"start": 4.0, "end": 5.0}],
        },
    ]
    out = _cleanup_cues(cues, min_cue_s=0.0, max_cue_s=7.0, lag_out_s=0.25)
    assert out[0]["end"] == pytest.approx(2.25)


# --- P2-2: held-word max_cue escape must not cross internal dead air ----------


def test_held_word_stops_before_dead_air_gap():
    # (a) ja shape: continuous words end ~2s past the cap, then a 3.7s silent
    # gap, then one isolated 80ms syllable. The escape must stop at the last word
    # before the gap (9.0) and NOT extend across dead air to the stray syllable.
    cue = {
        "text": "abcd",
        "start": 0.0,
        "end": 12.78,
        "word_data": [
            {"start": 0.0, "end": 3.0},
            {"start": 3.0, "end": 6.0},
            {"start": 6.0, "end": 9.0},  # cap (7.0) + ~2s, last continuous word
            {"start": 12.7, "end": 12.78},  # stray syllable after 3.7s dead air
        ],
    }
    out = _cleanup_cues([cue], min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(9.0)


def test_held_word_extends_across_small_sustain_gaps():
    # (b) en 'pearl My pearl' shape: continuous sustain with a 0.84s internal gap
    # (< HELD_WORD_MAX_GAP_S) ending 4.7s past the cap -> still extends to the
    # last word end (regression guard for today's earlier held-word fix).
    assert HELD_WORD_MAX_GAP_S > 0.84  # sustain-with-breath must pass the gate
    cue = {
        "text": "pearl My pearl",
        "start": 0.0,
        "end": 11.7,
        "word_data": [
            {"word": "pearl", "start": 0.0, "end": 1.0},
            {"word": "My", "start": 1.0, "end": 2.0},
            {"word": "pearl", "start": 2.84, "end": 11.7},
        ],
    }
    out = _cleanup_cues([cue], min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(11.7)


def test_held_word_words_before_cap_plain_clamp():
    # (c) all words end before the cap -> exact old clamp to start+max_cue_s.
    cue = {
        "text": "hello",
        "start": 0.0,
        "end": 10.0,
        "word_data": [{"word": "hello", "start": 0.0, "end": 5.0}],
    }
    out = _cleanup_cues([cue], min_cue_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(7.0)
