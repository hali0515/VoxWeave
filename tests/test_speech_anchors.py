# tests/test_speech_anchors.py
"""P3 PIN 4 -- ``speech_start`` / ``speech_end``: immutable acoustic anchors.

A cue's ``start``/``end`` are *display* time: ``_cleanup_cues`` pads them,
``_snap_to_shots`` moves them onto cuts, diarize trims them. Once moved, the
acoustic span they came from is unrecoverable -- ``_snap_to_shots`` has no raw
anchor for ``start`` at all. P3 captures the raw span at cue construction and
never lets a display pass touch it again:

* construction sites stamp the raw unit/atom span (fabricated timing stamps
  ``None`` -- invented time is not acoustic evidence),
* content-level folds (micro-merge, glue, bound-particle repair) recompute it
  from the material they folded,
* display passes must neither read nor write it,
* the sibling writer projects it out, so legacy-v1 JSON bytes do not move.

Nothing in legacy-v1 reads these fields; they arm the P5 finalizer.
"""

import json

import pytest

from voxweave import diarize, pipeline
from voxweave.core.smart_split import (
    _chunk_to_cue,
    _repair_bound_particle_cues,
    _split_without_timings,
    split_at_sentence_end,
)
from voxweave.core.timing import (
    _cleanup_cues,
    _glue_short_cues,
    _merge_micro_cues,
    _snap_to_shots,
    combine_speech,
)

EN_UNITS = [
    {"text": "Where", "start": 0.0, "end": 0.4},
    {"text": "did", "start": 0.5, "end": 0.8},
    {"text": "you", "start": 0.9, "end": 1.2},
    {"text": "go", "start": 1.4, "end": 2.0},
    {"text": "Nowhere", "start": 2.4, "end": 3.0},
    {"text": "special", "start": 3.1, "end": 3.6},
]


def _speech(cue):
    return (cue.get("speech_start"), cue.get("speech_end"))


def _atom_cue(surfaces, t0, step=0.3):
    """A cue whose word_data is one entry per packed atom (_chunk_to_cue shape)."""
    wd = []
    t = t0
    for surface in surfaces:
        wd.append({"text": surface, "start": round(t, 3), "end": round(t + step, 3)})
        t = round(t + step, 3)
    return {
        "text": "".join(surfaces),
        "start": wd[0]["start"],
        "end": wd[-1]["end"],
        "word_data": wd,
        "speech_start": wd[0]["start"],
        "speech_end": wd[-1]["end"],
    }


# --- combine_speech ----------------------------------------------------------


def test_combine_speech_takes_min_start_and_max_end():
    a = {"speech_start": 1.0, "speech_end": 2.0}
    b = {"speech_start": 3.0, "speech_end": 4.0}
    assert combine_speech(a, b) == (1.0, 4.0)
    assert combine_speech(b, a) == (1.0, 4.0)


def test_combine_speech_skips_none_sides():
    timed = {"speech_start": 1.0, "speech_end": 2.0}
    blank = {"speech_start": None, "speech_end": None}
    assert combine_speech(timed, blank) == (1.0, 2.0)
    assert combine_speech(blank, timed) == (1.0, 2.0)


def test_combine_speech_is_none_when_neither_side_has_an_anchor():
    blank = {"speech_start": None, "speech_end": None}
    assert combine_speech(blank, blank) == (None, None)
    assert combine_speech({}, {}) == (None, None)


def test_combine_speech_never_falls_back_to_display_bounds():
    """A missing anchor stays missing -- display time must not be laundered in."""
    a = {"start": 0.0, "end": 9.0, "speech_start": None, "speech_end": None}
    b = {"start": 0.0, "end": 9.0}
    assert combine_speech(a, b) == (None, None)


# --- construction sites ------------------------------------------------------


def test_split_at_sentence_end_stamps_the_raw_clause_span():
    word_data = [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "there.", "start": 0.5, "end": 0.9},
        {"word": "How", "start": 2.0, "end": 2.3},
        {"word": "are", "start": 2.4, "end": 2.6},
        {"word": "you", "start": 2.7, "end": 3.0},
    ]
    cues = split_at_sentence_end("Hello there. How are you", word_data, "en", 42, 2)
    assert [c["text"] for c in cues] == ["Hello there.", "How are you"]
    assert _speech(cues[0]) == (0.0, 0.9) == (cues[0]["start"], cues[0]["end"])
    assert _speech(cues[1]) == (2.0, 3.0) == (cues[1]["start"], cues[1]["end"])


def test_split_at_sentence_end_fabricated_timing_carries_no_anchor():
    """The prev_end / word-count fallback invents time; it is not evidence."""
    word_data = [{"word": "Hello"}, {"word": "there."}, {"word": "How"}]
    cues = split_at_sentence_end("Hello there. How", word_data, "en", 42, 2)
    assert [(c["start"], c["end"]) for c in cues] == [(0.0, 1.0), (1.0, 2.0)]
    assert all(_speech(c) == (None, None) for c in cues)


def test_chunk_to_cue_stamps_the_raw_chunk_span():
    parent = {"text": "x", "start": 5.0, "end": 6.0, "word_data": []}
    chunk = [
        {"text": "one", "start": 1.0, "end": 1.5},
        {"text": "two", "start": 1.6, "end": 2.2},
    ]
    cue = _chunk_to_cue(chunk, parent, "en")
    assert (cue["start"], cue["end"]) == (1.0, 2.2)
    assert _speech(cue) == (1.0, 2.2)


def test_chunk_to_cue_uses_no_parent_fallback_for_the_anchor():
    """Display start/end fall back to the parent cue; the anchor must not --
    that would launder a display pad into the raw layer."""
    parent = {"text": "x", "start": 5.0, "end": 6.0, "word_data": []}
    chunk = [
        {"text": "one", "start": None, "end": None},
        {"text": "two", "start": None, "end": None},
    ]
    cue = _chunk_to_cue(chunk, parent, "en")
    assert (cue["start"], cue["end"]) == (5.0, 6.0)
    assert _speech(cue) == (None, None)


def test_split_without_timings_carries_no_anchor():
    cue = {
        "text": "alpha beta gamma delta epsilon zeta",
        "start": 0.0,
        "end": 6.0,
        "word_data": [],
    }
    out = _split_without_timings(dict(cue), 12, 1, "en")
    assert len(out) == 3
    assert all(_speech(c) == (None, None) for c in out)


# --- display passes never touch the anchors ----------------------------------


def test_cleanup_extends_the_display_end_and_leaves_speech_end_untouched():
    cue = {
        "text": "hi",
        "start": 0.0,
        "end": 1.0,
        "word_data": [{"text": "hi", "start": 0.0, "end": 1.0}],
        "speech_start": 0.0,
        "speech_end": 1.0,
    }
    out = _cleanup_cues([cue], min_cue_s=0.5, max_cue_s=7.0, cps=0.0, lag_out_s=0.25)
    assert out[0]["end"] == pytest.approx(1.25)  # lag-out pad applied
    assert _speech(out[0]) == (0.0, 1.0)


def test_cleanup_min_duration_floor_leaves_the_anchor_untouched():
    cue = {
        "text": "a",
        "start": 0.0,
        "end": 0.1,
        "word_data": [{"text": "a", "start": 0.0, "end": 0.1}],
        "speech_start": 0.0,
        "speech_end": 0.1,
    }
    out = _cleanup_cues([cue], min_cue_s=2.0, max_cue_s=7.0, cps=0.0, lag_out_s=0.0)
    assert out[0]["end"] == pytest.approx(2.0)
    assert _speech(out[0]) == (0.0, 0.1)


def test_shot_snap_moves_display_bounds_only():
    cue = {
        "text": "hi",
        "start": 1.0,
        "end": 2.0,
        "word_data": [{"text": "hi", "start": 1.0, "end": 2.0}],
        "speech_start": 1.0,
        "speech_end": 2.0,
    }
    out = _snap_to_shots([cue], [0.95], snap_s=0.458, max_cue_s=7.0)
    assert _speech(out[0]) == (1.0, 2.0)
    assert out[0]["start"] != 1.0 or out[0]["end"] != 2.0


# --- content folds recompute the anchors -------------------------------------


def test_merge_micro_cues_folds_the_anchors():
    cues = [
        {
            "text": "そう",
            "start": 0.0,
            "end": 0.5,
            "word_data": [{"text": "そう", "start": 0.0, "end": 0.5}],
            "speech_start": 0.0,
            "speech_end": 0.5,
        },
        {
            "text": "だね",
            "start": 0.6,
            "end": 1.0,
            "word_data": [{"text": "だね", "start": 0.6, "end": 1.0}],
            "speech_start": 0.6,
            "speech_end": 1.0,
        },
    ]
    out = _merge_micro_cues(
        cues,
        "ja",
        max_gap_s=0.3,
        max_line_length=18,
        max_cue_s=7.0,
        min_cue_s=0.0,
        max_lines=1,
    )
    assert len(out) == 1 and out[0]["text"] == "そうだね"
    assert _speech(out[0]) == (0.0, 1.0)


def test_glue_short_cues_folds_the_anchors_forward():
    cues = [
        {
            "text": "え",
            "start": 0.0,
            "end": 0.2,
            "word_data": [{"text": "え", "start": 0.0, "end": 0.2}],
            "speech_start": 0.0,
            "speech_end": 0.2,
        },
        {
            "text": "こんにちは",
            "start": 0.25,
            "end": 1.0,
            "word_data": [{"text": "こんにちは", "start": 0.25, "end": 1.0}],
            "speech_start": 0.25,
            "speech_end": 1.0,
        },
    ]
    out = _glue_short_cues(
        cues, "ja", max_gap_s=0.3, max_line_length=18, max_lines=1, max_cue_s=7.0
    )
    assert len(out) == 1 and out[0]["text"] == "えこんにちは"
    assert _speech(out[0]) == (0.0, 1.0)


def test_glue_short_cues_folds_the_anchors_backward():
    cues = [
        {
            "text": "こんにちは",
            "start": 0.0,
            "end": 1.0,
            "word_data": [{"text": "こんにちは", "start": 0.0, "end": 1.0}],
            "speech_start": 0.0,
            "speech_end": 1.0,
        },
        {
            "text": "ね",
            "start": 1.05,
            "end": 1.2,
            "word_data": [{"text": "ね", "start": 1.05, "end": 1.2}],
            "speech_start": 1.05,
            "speech_end": 1.2,
        },
    ]
    out = _glue_short_cues(
        cues, "ja", max_gap_s=0.3, max_line_length=18, max_lines=1, max_cue_s=7.0
    )
    assert len(out) == 1 and out[0]["text"] == "こんにちはね"
    assert _speech(out[0]) == (0.0, 1.2)


# --- fold sites never launder display padding into the raw layer -------------
#
# The fold tests above build cues whose display bounds already EQUAL their raw
# span, so substituting ``cue["start"], cue["end"]`` for ``combine_speech`` there
# is invisible. Every fixture below is deliberately display-padded (a lag-out
# tail, a shot lead-in) so that substitution is visible, and each also has an
# anchorless variant so ``None`` has to survive the fold.


def _padded(text, display, raw, *, word_data=None):
    """A cue whose visible bounds were moved by a display pass.

    ``display`` is what a viewer sees, ``raw`` is the acoustic span (``None``
    for a cue built from fabricated timing).
    """
    start, end = display
    speech_start, speech_end = raw
    if word_data is None:
        word_data = (
            [{"text": text, "start": speech_start, "end": speech_end}]
            if speech_start is not None
            else [{"text": text}]
        )
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": word_data,
        "speech_start": speech_start,
        "speech_end": speech_end,
    }


MERGE_KWARGS = dict(
    max_gap_s=0.3, max_line_length=18, max_cue_s=7.0, min_cue_s=0.0, max_lines=1
)
GLUE_KWARGS = dict(max_gap_s=0.3, max_line_length=18, max_lines=1, max_cue_s=7.0)


def test_merge_micro_cues_folds_raw_bounds_not_the_padded_display_ones():
    cues = [
        _padded("そう", display=(0.0, 0.55), raw=(0.2, 0.5)),
        _padded("だね", display=(0.6, 1.03), raw=(0.6, 1.0)),
    ]
    out = _merge_micro_cues(cues, "ja", **MERGE_KWARGS)
    assert len(out) == 1 and out[0]["text"] == "そうだね"
    # display keeps the pads on both edges ...
    assert (out[0]["start"], out[0]["end"]) == (0.0, 1.03)
    # ... and the anchor is the raw envelope, which is strictly inside them
    assert _speech(out[0]) == (0.2, 1.0)


def test_merge_micro_cues_keeps_an_anchorless_side_anchorless():
    cues = [
        _padded("そう", display=(0.0, 0.55), raw=(None, None)),
        _padded("だね", display=(0.6, 1.03), raw=(0.6, 1.0)),
    ]
    out = _merge_micro_cues(cues, "ja", **MERGE_KWARGS)
    assert len(out) == 1
    assert _speech(out[0]) == (0.6, 1.0)

    both_blank = [
        _padded("そう", display=(0.0, 0.55), raw=(None, None)),
        _padded("だね", display=(0.6, 1.03), raw=(None, None)),
    ]
    out = _merge_micro_cues(both_blank, "ja", **MERGE_KWARGS)
    assert len(out) == 1
    assert _speech(out[0]) == (None, None)


def test_glue_forward_folds_raw_bounds_not_the_padded_display_ones():
    cues = [
        _padded("え", display=(0.0, 0.22), raw=(0.05, 0.2)),
        _padded("こんにちは", display=(0.25, 1.03), raw=(0.25, 1.0)),
    ]
    out = _glue_short_cues(cues, "ja", **GLUE_KWARGS)
    assert len(out) == 1 and out[0]["text"] == "えこんにちは"
    assert (out[0]["start"], out[0]["end"]) == (0.0, 1.03)
    assert _speech(out[0]) == (0.05, 1.0)


def test_glue_forward_keeps_an_anchorless_side_anchorless():
    cues = [
        _padded("え", display=(0.0, 0.22), raw=(None, None)),
        _padded("こんにちは", display=(0.25, 1.03), raw=(None, None)),
    ]
    out = _glue_short_cues(cues, "ja", **GLUE_KWARGS)
    assert len(out) == 1
    assert _speech(out[0]) == (None, None)


def test_glue_backward_folds_raw_bounds_not_the_padded_display_ones():
    cues = [
        _padded("こんにちは", display=(0.0, 1.03), raw=(0.1, 1.0)),
        _padded("ね", display=(1.05, 1.2), raw=(1.05, 1.15)),
    ]
    out = _glue_short_cues(cues, "ja", **GLUE_KWARGS)
    assert len(out) == 1 and out[0]["text"] == "こんにちはね"
    assert (out[0]["start"], out[0]["end"]) == (0.0, 1.2)
    assert _speech(out[0]) == (0.1, 1.15)


def test_glue_backward_keeps_an_anchorless_side_anchorless():
    cues = [
        _padded("こんにちは", display=(0.0, 1.03), raw=(None, None)),
        _padded("ね", display=(1.05, 1.2), raw=(None, None)),
    ]
    out = _glue_short_cues(cues, "ja", **GLUE_KWARGS)
    assert len(out) == 1
    assert _speech(out[0]) == (None, None)


REPAIR_KWARGS = dict(
    lang="zh", max_line_length=8, max_lines=1, max_cue_s=7.0, connected_gap_s=0.4
)


def _zh_units(surfaces, t0, step=0.3):
    out = []
    t = t0
    for surface in surfaces:
        out.append({"text": surface, "start": round(t, 3), "end": round(t + step, 3)})
        t = round(t + step, 3)
    return out


def _untimed_units(surfaces):
    return [{"text": s, "start": None, "end": None} for s in surfaces]


def test_repair_merge_branch_folds_raw_bounds_not_the_padded_display_ones():
    left = _padded(
        "这个的", display=(0.0, 3.0), raw=(0.2, 1.1), word_data=_zh_units("这个的", 0.2)
    )
    right = _padded(
        "照片", display=(1.2, 3.0), raw=(1.2, 1.8), word_data=_zh_units("照片", 1.2)
    )
    out = _repair_bound_particle_cues([left, right], **REPAIR_KWARGS)
    assert [c["text"] for c in out] == ["这个的照片"]
    assert (out[0]["start"], out[0]["end"]) == (0.0, 3.0)  # display pads survive
    assert _speech(out[0]) == (0.2, 1.8)  # the anchor does not inherit them


def test_repair_merge_branch_keeps_an_anchorless_side_anchorless():
    left = _padded(
        "这个的", display=(0.0, 3.0), raw=(0.2, 1.1), word_data=_zh_units("这个的", 0.2)
    )
    right = _padded(
        "照片",
        display=(1.2, 3.0),
        raw=(None, None),
        word_data=[{"text": "照"}, {"text": "片"}],
    )
    out = _repair_bound_particle_cues([left, right], **REPAIR_KWARGS)
    assert [c["text"] for c in out] == ["这个的照片"]
    assert _speech(out[0]) == (0.2, 1.1)


def test_repair_repartition_branch_never_inherits_the_display_bounds():
    """The moved material decides the anchor; the visible bounds do not.

    The right piece here keeps the parent's padded display end (9.0) while its
    atoms stop at 1.2 -- exactly the gap a display fallback would paper over.
    """
    left = _padded(
        "这个照片的",
        display=(0.0, 1.5),
        raw=(0.0, 1.2),
        word_data=_zh_units("这个照片", 0.0) + _untimed_units("的"),
    )
    right = _padded(
        "很好看",
        display=(1.4, 9.0),
        raw=(None, None),
        word_data=_untimed_units("很好看"),
    )
    out = _repair_bound_particle_cues([left, right], **REPAIR_KWARGS)
    assert [c["text"] for c in out] == ["这个", "照片的很好看"]
    assert (out[1]["start"], out[1]["end"]) == (0.6, 9.0)
    assert _speech(out[0]) == (0.0, 0.6)
    assert _speech(out[1]) == (0.6, 1.2)


def test_repair_repartition_branch_propagates_none_for_an_untimed_side():
    """A piece whose atoms are all untimed carries no anchor at all, even though
    its display start/end fall back to the parent cue's bounds."""
    left = _padded(
        "这个照片的",
        display=(0.0, 1.5),
        raw=(0.0, 0.6),
        word_data=_zh_units("这个", 0.0) + _untimed_units("照片的"),
    )
    right = _padded(
        "很好看啊",
        display=(0.9, 3.0),
        raw=(None, None),
        word_data=_untimed_units("很好看啊"),
    )
    out = _repair_bound_particle_cues([left, right], **REPAIR_KWARGS)
    assert [c["text"] for c in out] == ["这个", "照片的很好看啊"]
    # the visible span still uses the inherited fallback ...
    assert (out[1]["start"], out[1]["end"]) == (0.9, 3.0)
    # ... while the anchor stays empty
    assert _speech(out[1]) == (None, None)
    assert _speech(out[0]) == (0.0, 0.6)


def test_repair_merge_branch_folds_the_anchors():
    left = _atom_cue(["这", "个", "的"], 0.0)
    right = _atom_cue(["照", "片"], 0.9)
    out = _repair_bound_particle_cues([dict(left), dict(right)], **REPAIR_KWARGS)
    assert [c["text"] for c in out] == ["这个的照片"]
    assert _speech(out[0]) == (0.0, 1.5)


def test_repair_repartition_branch_recomputes_the_anchors_from_raw_atoms():
    left = _atom_cue(["这", "个", "GPT-4"], 0.0)
    right = _atom_cue(["的", "照", "片", "很", "好"], 0.9)
    out = _repair_bound_particle_cues([dict(left), dict(right)], **REPAIR_KWARGS)
    assert [c["text"] for c in out] == ["这个GPT-4的照片", "很好"]
    assert _speech(out[0]) == (0.0, 1.8)
    assert _speech(out[1]) == (1.8, 2.4)


# --- diarize -----------------------------------------------------------------


def _en_cue():
    """A cue built the way production builds one, then display-extended."""
    atoms = [
        {"text": "One", "start": 0.0, "end": 0.5},
        {"text": "two", "start": 1.0, "end": 1.5},
        {"text": "three", "start": 2.0, "end": 2.5},
    ]
    parent = {"text": "x", "start": 0.0, "end": 2.5, "word_data": []}
    cue = _chunk_to_cue(atoms, parent, "en")
    cue["end"] = 9.0  # a display pad a cleanup/shot-snap pass would have added
    return cue


def test_diarize_dual_branch_carries_the_anchors():
    cue = _en_cue()
    out = diarize.format_speaker_cues([cue], [(0.0, 1.7, "A"), (1.9, 2.7, "B")], "en")
    assert len(out) == 1 and out[0]["text"].count("\n") == 1
    assert out[0]["end"] == 9.0  # display pad preserved
    assert _speech(out[0]) == (0.0, 2.5)  # raw span, not the pad


def test_diarize_split_pieces_stamp_from_the_run_atoms():
    cue = _en_cue()
    out = diarize.format_speaker_cues(
        [cue], [(0.0, 0.7, "A"), (0.9, 1.7, "B"), (1.9, 2.7, "C")], "en"
    )
    assert [c["text"] for c in out] == ["One", "two", "three"]
    assert [_speech(c) for c in out] == [(0.0, 0.5), (1.0, 1.5), (2.0, 2.5)]


def test_diarize_split_piece_without_timed_atoms_gets_none_not_display_bounds(
    monkeypatch,
):
    """``_run_span`` falls back to the cue's DISPLAY bounds for the piece's
    start/end. The anchor must NOT inherit that fallback."""
    cue = _en_cue()
    runs = [
        (
            "A",
            [
                {
                    "text": "One",
                    "start": 0.0,
                    "end": 0.5,
                    "_unit_start": 0,
                    "_unit_end": 1,
                }
            ],
        ),
        (
            "B",
            [
                {
                    "text": "two",
                    "start": None,
                    "end": None,
                    "_unit_start": 1,
                    "_unit_end": 2,
                }
            ],
        ),
        (
            "C",
            [
                {
                    "text": "three",
                    "start": 2.0,
                    "end": 2.5,
                    "_unit_start": 2,
                    "_unit_end": 3,
                }
            ],
        ),
    ]
    monkeypatch.setattr(diarize, "_speaker_runs", lambda atoms, turns, lang: runs)
    out = diarize.format_speaker_cues([cue], [(0.0, 9.0, "A")], "en")

    assert [c["text"] for c in out] == ["One", "two", "three"]
    # the untimed run keeps today's display fallback for its visible span ...
    assert (out[1]["start"], out[1]["end"]) == (0.0, 9.0)
    # ... but carries no acoustic anchor at all
    assert _speech(out[1]) == (None, None)
    assert _speech(out[0]) == (0.0, 0.5)
    assert _speech(out[2]) == (2.0, 2.5)


# --- writer projection -------------------------------------------------------


def test_write_siblings_projects_the_anchors_out(tmp_path):
    src = tmp_path / "clip.mkv"
    cues = [
        {
            "text": "hi",
            "start": 0.0,
            "end": 1.25,
            "word_data": [{"text": "hi", "start": 0.0, "end": 1.0}],
            "lyric": True,
            "speech_start": 0.0,
            "speech_end": 1.0,
        }
    ]
    units = [{"text": "hi", "start": 0.0, "end": 1.0}]
    pipeline._write_siblings(src, cues, units, "en")

    data = json.loads((tmp_path / "clip.json").read_text(encoding="utf-8"))
    assert data["segments"] == [
        {
            "text": "hi",
            "start": 0.0,
            "end": 1.25,
            "word_data": [{"text": "hi", "start": 0.0, "end": 1.0}],
            "lyric": True,
        }
    ]
    # the in-memory cue is untouched by the projection
    assert _speech(cues[0]) == (0.0, 1.0)


def test_write_siblings_drops_only_the_anchor_keys(tmp_path):
    """Drop-list, not whitelist: any other key a cue carries still ships."""
    src = tmp_path / "clip.mkv"
    cues = [
        {
            "text": "hi",
            "start": 0.0,
            "end": 1.0,
            "word_data": [],
            "speech_start": 0.0,
            "speech_end": 1.0,
            "future_key": {"nested": 1},
        }
    ]
    pipeline._write_siblings(src, cues, [], "en")
    data = json.loads((tmp_path / "clip.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["future_key"] == {"nested": 1}
    assert "speech_start" not in data["segments"][0]
    assert "speech_end" not in data["segments"][0]


# --- end to end --------------------------------------------------------------


def test_segment_document_cues_all_carry_the_anchor_keys():
    result = pipeline.segment_document(
        language="en", word_segments=[dict(u) for u in EN_UNITS]
    )
    assert result.cues
    for cue in result.cues:
        assert "speech_start" in cue
        assert "speech_end" in cue


def test_split_writes_no_speech_keys_into_the_sibling(tmp_path):
    path = tmp_path / "ep.json"
    path.write_text(
        json.dumps({"language": "en", "word_segments": [dict(u) for u in EN_UNITS]}),
        encoding="utf-8",
    )
    pipeline.split(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["segments"]
    for segment in data["segments"]:
        assert "speech_start" not in segment
        assert "speech_end" not in segment
