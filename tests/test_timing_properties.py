# tests/test_timing_properties.py
"""Composition properties of the terminal timing passes.

Three audit probes are pinned here:

- PROBE C: ``_cleanup_cues`` is not idempotent. Its lag-out pad is measured from
  the *display* end, so every extra pass adds another ``lag_out_s`` to any cue
  with room after it (sparse streams, and the final cue which extends freely).
  The diarize path runs cleanup a second time, so its output shipped the extra
  pad. API PIN 2: the extension anchors on the cue's speech end
  (``max(w["end"]) over word_data``) with ``want = max(current_end, speech_end +
  lag_out_s)``, so a second application is a no-op for cues with word_data while
  an already-longer display end is preserved (max rule). Cues without word_data
  keep the legacy behavior.
- in-time cap: ``_snap_to_shots`` guards both out-time branches with
  ``max_cue_s`` but not the in-time lead-in, so a cue sitting exactly at the cap
  is pulled back onto a cut and ends up over the cap. API PIN 3: reject that
  move (keep the original start).
- PROBE D: a second cleanup pushes a shot-snapped end back over its cut. With an
  idempotent cleanup and the pipeline re-snap (API PIN 10) the terminal
  composition ``cleanup -> snap`` must be a fixed point.
"""

import copy
import itertools

import pytest

from voxweave.core.timing import (
    _FRAME_S,
    TWO_FRAME_S,
    _cleanup_cues,
    _snap_to_shots,
)

MIN_CUE_S = 0.5
MAX_CUE_S = 7.0
LAG_OUT_S = 0.25  # config default (_LAG_OUT_DEFAULT_MS)
SNAP_S = 0.458  # config default (11 frames @ 24fps)

CLEANUP_KW = {
    "min_cue_s": MIN_CUE_S,
    "max_cue_s": MAX_CUE_S,
    "cps": 0.0,
    "lag_out_s": LAG_OUT_S,
}


def _cue(text, start, end, speech_end=None):
    """A cue whose word_data covers its span (speech_end defaults to the end)."""
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [
            {"start": start, "end": end if speech_end is None else speech_end}
        ],
    }


def _snapshot(cues):
    """Comparable view of a cue stream (text + rounded span + word spans)."""
    return [
        (
            c["text"],
            round(c["start"], 9),
            round(c["end"], 9),
            tuple(
                (round(w["start"], 9), round(w["end"], 9))
                for w in c.get("word_data") or []
            ),
        )
        for c in cues
    ]


def _cleanup(cues, **overrides):
    kw = {**CLEANUP_KW, **overrides}
    return _cleanup_cues(copy.deepcopy(cues), **kw)


# --------------------------------------------------------------------------- #
# PROBE C: _cleanup_cues must be idempotent for cues carrying word_data
# --------------------------------------------------------------------------- #


def test_cleanup_idempotent_on_sparse_stream():
    """PROBE C: two cues with word_data and a 2.0s gap -- the second cleanup pass
    must reproduce the first exactly (today each pass adds another lag-out pad)."""
    cues = [_cue("a", 0.0, 2.0), _cue("b", 4.0, 5.0)]
    once = _cleanup(cues)
    # precondition: the first pass really does apply the tail pad
    assert once[0]["end"] == pytest.approx(2.25)
    twice = _cleanup(once)
    assert _snapshot(twice) == _snapshot(once)


def test_cleanup_idempotent_on_final_cue():
    """PROBE C: the final cue extends freely (no successor), so it is the purest
    accumulation case -- a second pass must not pad it again."""
    cues = [_cue("fin", 10.0, 11.0)]
    once = _cleanup(cues)
    assert once[0]["end"] == pytest.approx(11.25)
    twice = _cleanup(once)
    assert _snapshot(twice) == _snapshot(once)


@pytest.mark.parametrize("passes", [2, 3, 4])
def test_cleanup_final_cue_pad_does_not_accumulate(passes):
    """PROBE C: repeated application of cleanup on a sparse tail converges after
    the first pass instead of growing by lag_out_s every time."""
    cues = [_cue("fin", 10.0, 11.0)]
    once = _cleanup(cues)
    repeated = copy.deepcopy(cues)
    for _ in range(passes):
        repeated = _cleanup_cues(repeated, **CLEANUP_KW)
    assert _snapshot(repeated) == _snapshot(once)
    assert repeated[0]["end"] == pytest.approx(11.25)


def test_second_cleanup_is_a_noop_for_the_diarize_double_application():
    """PROBE C: the diarize path runs cleanup once in segmentation and again after
    speaker formatting. On a stream mixing a chained pair, a lone cue and a sparse
    tail, the second application must change nothing."""
    cues = [
        _cue("x", 0.0, 1.0),
        _cue("y", 1.45, 2.0),
        _cue("z", 4.0, 6.0),
        _cue("w", 20.0, 21.0),
    ]
    once = _cleanup(cues)
    # precondition: the first pass mints a chained two-frame gap and pads the rest
    assert once[1]["start"] - once[0]["end"] == pytest.approx(TWO_FRAME_S)
    assert once[1]["end"] == pytest.approx(2.25)
    assert once[2]["end"] == pytest.approx(6.25)
    assert once[3]["end"] == pytest.approx(21.25)
    twice = _cleanup(once)
    assert _snapshot(twice) == _snapshot(once)


def test_cleanup_keeps_display_end_already_past_the_lag_out_target():
    """PROBE C: the max rule -- a cue whose display end was already extended past
    speech_end + lag_out_s (CPS linger, an earlier snap) keeps that longer end and
    is neither shrunk back to the anchor nor padded again."""
    cues = [
        {
            "text": "lingered",
            "start": 0.0,
            "end": 5.0,
            "word_data": [{"start": 0.0, "end": 2.0}],
        },
        _cue("next", 20.0, 21.0),
    ]
    once = _cleanup(cues, min_cue_s=0.0)
    assert once[0]["end"] == pytest.approx(5.0)
    twice = _cleanup(once, min_cue_s=0.0)
    assert twice[0]["end"] == pytest.approx(5.0)


def test_cue_without_word_data_keeps_legacy_lag_out():
    """PROBE C: with no word_data there is no speech end to anchor on, so the flat
    display-end pad stays exactly as before."""
    cues = [
        {"text": "hello", "start": 0.0, "end": 2.0, "word_data": []},
        {"text": "next", "start": 8.0, "end": 9.0, "word_data": []},
    ]
    once = _cleanup(cues, min_cue_s=0.0)
    assert once[0]["end"] == pytest.approx(2.25)


# --------------------------------------------------------------------------- #
# in-time cap: the lead-in snap must respect max_cue_s like the out-times do
# --------------------------------------------------------------------------- #

_CUT_BEFORE = 10.0 - 7 * _FRAME_S  # 7 frames before the cue start (zone 1-9 after)


def test_in_time_snap_rejected_when_it_would_exceed_max_cue():
    """in-time cap: a cue already at max_cue_s must keep its start rather than be
    pulled back onto a cut 7 frames earlier (that move inflates it past the cap,
    which both out-time branches already refuse)."""
    cues = [_cue("long", 10.0, 10.0 + MAX_CUE_S)]
    out = _snap_to_shots(cues, [_CUT_BEFORE], snap_s=SNAP_S, max_cue_s=MAX_CUE_S)
    assert out[0]["start"] == pytest.approx(10.0)
    assert out[0]["end"] - out[0]["start"] <= MAX_CUE_S + 1e-9


def test_short_cue_still_gets_the_lead_in_snap():
    """in-time cap: same geometry with a short cue -- the normal lead-in still
    applies, so the cap guard must not veto every backward move."""
    cues = [_cue("short", 10.0, 12.0)]
    out = _snap_to_shots(cues, [_CUT_BEFORE], snap_s=SNAP_S, max_cue_s=MAX_CUE_S)
    assert out[0]["start"] == pytest.approx(_CUT_BEFORE)


def test_in_time_snap_allowed_when_result_lands_exactly_on_the_cap():
    """in-time cap: a lead-in that brings the duration to exactly max_cue_s is
    legal -- the guard rejects only what exceeds the cap."""
    cues = [_cue("exact", 10.0, 10.0 + MAX_CUE_S - 7 * _FRAME_S)]
    out = _snap_to_shots(cues, [_CUT_BEFORE], snap_s=SNAP_S, max_cue_s=MAX_CUE_S)
    assert out[0]["start"] == pytest.approx(_CUT_BEFORE)
    assert out[0]["end"] - out[0]["start"] == pytest.approx(MAX_CUE_S)


# --------------------------------------------------------------------------- #
# PROBE D: cleanup -> snap must be a terminal fixed point
# --------------------------------------------------------------------------- #

_SHOTS = [11.1]


def _probe_d_stream():
    # "a" ends just before a cut, so snapping parks its end on cut - 2 frames.
    # "b" is a sparse tail far from any cut: nothing re-snaps it, so any cleanup
    # accumulation survives into the terminal output.
    return [_cue("a", 8.0, 11.0), _cue("b", 20.0, 21.0)]


def _terminal_rounds(cues, passes):
    out = copy.deepcopy(cues)
    for _ in range(passes):
        out = _cleanup_cues(out, **CLEANUP_KW)
        out = _snap_to_shots(out, _SHOTS, snap_s=SNAP_S, max_cue_s=MAX_CUE_S)
    return out


@pytest.mark.parametrize("passes", [2, 3, 4])
def test_cleanup_snap_composition_is_a_fixed_point(passes):
    """PROBE D: re-running the terminal pair (as the diarize path does, with the
    re-snap of API PIN 10) must reproduce the single-pass result exactly."""
    once = _terminal_rounds(_probe_d_stream(), 1)
    # precondition: the single pass really does snap the first cue onto the cut
    assert once[0]["end"] == pytest.approx(_SHOTS[0] - TWO_FRAME_S)
    assert once[1]["end"] == pytest.approx(21.25)
    again = _terminal_rounds(_probe_d_stream(), passes)
    assert _snapshot(again) == _snapshot(once)


@pytest.mark.parametrize("passes", [1, 2, 4])
def test_snapped_end_never_crosses_its_cut(passes):
    """PROBE D: however many times the terminal pair runs, a cue snapped to die on
    a cut stays on the safe side of it and the stream stays non-overlapping."""
    out = _terminal_rounds(_probe_d_stream(), passes)
    assert out[0]["end"] == pytest.approx(_SHOTS[0] - TWO_FRAME_S)
    assert out[0]["end"] <= _SHOTS[0] + 1e-9
    for cur, nxt in itertools.pairwise(out):
        assert cur["end"] <= nxt["start"] + 1e-9
