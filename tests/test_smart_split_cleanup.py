# tests/test_smart_split_cleanup.py
import pytest

from voxweave.core.timing import _cleanup_cues


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
