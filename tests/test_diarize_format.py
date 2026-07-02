# tests/test_diarize_format.py
# Speaker-formatting defects fixed after the GPU E2E audit (pure post-pass, no
# pyannote/GPU):
#   Fix 1 - dash cues must render <=2 lines (normalize pre-wrapped pieces before the
#           dual-budget test; re-wrap split pieces per language).
#   Fix 2 - split/dash pieces must go through timing polish (_cleanup_cues) so they
#           are not sub-flash cues; the no-thresholds call stays byte-compatible.
#   Fix 3 - speaker runs must not cut mid-word (absorb <0.2s label thrash, snap
#           surviving boundaries to jieba/BudouX phrase edges).
from voxweave.core.layout import _vis_width
from voxweave.diarize import (
    _speaker_runs,
    apply_speaker_format,
    format_speaker_cues,
)


def _cue(text, start, end, words):
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [{"start": s, "end": e} for s, e in words],
    }


def _atoms(items):
    return [{"text": t, "start": s, "end": e} for t, s, e in items]


def _lines(text):
    return text.split("\n")


def _max_lines(cues):
    return max(c["text"].count("\n") + 1 for c in cues)


# --- Fix 1: dash cues render <=2 lines ---------------------------------------

# Two speakers over 9 words; the cue text arrives pre-wrapped (contains "\n")
# because smart_split wrapped it before diarization ran.
EN_DASH_TURNS = [(0.0, 4.0, "SPEAKER_00"), (4.5, 6.0, "SPEAKER_01")]


def _en_dash_cue():
    return _cue(
        "And yours is mine\nand mine is yours Correct",
        0.5,
        5.5,
        [
            (0.5, 0.8),
            (0.9, 1.2),
            (1.3, 1.5),
            (1.6, 1.9),
            (2.0, 2.3),
            (2.4, 2.7),
            (2.8, 3.0),
            (3.1, 3.5),
            (5.0, 5.5),  # "Correct" -> SPEAKER_01
        ],
    )


def test_dash_cue_normalizes_prewrapped_text_to_two_lines():
    out = format_speaker_cues([_en_dash_cue()], EN_DASH_TURNS, "en")
    assert len(out) == 1
    text = out[0]["text"]
    # exactly one newline: two visible lines, not three
    assert text.count("\n") == 1
    lines = _lines(text)
    # both lines start with a bare hyphen (no space) and fit one 42-col line
    for ln in lines:
        assert ln.startswith("-") and not ln.startswith("- ")
        assert _vis_width(ln) <= 42


def test_no_output_cue_renders_more_than_two_lines():
    # Speaker A's line is 43 cols: once "-" is added it no longer fits one line, so
    # the dual event (which would be 3 lines) must NOT be emitted. The cue falls
    # through to a split, and the pre-wrapped "\n" piece re-wraps to <=2 clean lines.
    cue = _cue(
        "the quick brown fox jumps over\nthe lazy dog runs away",
        0.0,
        6.0,
        [
            (0.0, 0.3),
            (0.4, 0.7),
            (0.8, 1.1),
            (1.2, 1.5),
            (1.6, 1.9),
            (2.0, 2.3),
            (2.4, 2.7),
            (2.8, 3.1),
            (3.2, 3.5),  # "dog" -> last SPEAKER_00 word
            (4.6, 4.9),  # "runs" -> SPEAKER_01
            (5.0, 5.4),  # "away"
        ],
    )
    turns = [(0.0, 4.0, "SPEAKER_00"), (4.5, 6.5, "SPEAKER_01")]
    out = format_speaker_cues([cue], turns, "en")
    assert len(out) >= 2  # dual rejected -> split branch taken
    assert _max_lines(out) <= 2


def test_ja_split_piece_has_no_newline():
    # ja is single-line: a two-speaker ja cue splits (no dash pairing). The cue
    # text carries a stale "\n" (position 6, inside speaker B's run); each emitted
    # piece must be re-wrapped so it contains zero "\n".
    cue = _cue(
        "これはテスト\nですね",
        0.0,
        1.8,
        [
            (0.0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
            (1.0, 1.2),
            (1.2, 1.4),
            (1.4, 1.6),
            (1.6, 1.8),
        ],
    )
    turns = [(0.0, 0.6, "SPEAKER_00"), (0.6, 1.8, "SPEAKER_01")]
    out = format_speaker_cues([cue], turns, "ja")
    assert len(out) >= 2  # it actually split
    for c in out:
        assert "\n" not in c["text"]


# --- Fix 2: split/dash pieces go through timing polish -----------------------

TH = {"min_cue_s": 0.5, "max_cue_s": 7.0, "cps": 0.0, "lag_out_s": 0.0}
ZH_TURNS = [(0.0, 0.4, "SPEAKER_00"), (0.4, 4.0, "SPEAKER_01")]


def _zh_split_cue():
    # "你好朋友" -> "你好" (SPEAKER_00) / "朋友" (SPEAKER_01)
    return _cue(
        "你好朋友",
        0.0,
        0.7,
        [(0.0, 0.2), (0.2, 0.4), (0.4, 0.55), (0.55, 0.7)],
    )


def test_short_split_piece_extended_into_following_gap():
    cue2 = _cue("在", 2.0, 2.1, [(2.0, 2.1)])
    out = apply_speaker_format([_zh_split_cue(), cue2], ZH_TURNS, "zh", thresholds=TH)
    peng = next(c for c in out if c["text"] == "朋友")
    zai = next(c for c in out if c["text"] == "在")
    assert peng["end"] - peng["start"] >= TH["min_cue_s"] - 1e-9
    assert peng["end"] <= zai["start"] + 1e-9


def test_distinct_speaker_abutting_pieces_not_merged():
    out = apply_speaker_format([_zh_split_cue()], ZH_TURNS, "zh", thresholds=TH)
    assert [c["text"] for c in out] == ["你好", "朋友"]


def test_no_thresholds_call_preserves_timing():
    cue2 = _cue("在", 2.0, 2.1, [(2.0, 2.1)])
    out = apply_speaker_format([_zh_split_cue(), cue2], ZH_TURNS, "zh")
    peng = next(c for c in out if c["text"] == "朋友")
    # no polish: the piece keeps its raw 0.4-0.7 span (no min-dur extension)
    assert peng["end"] == 0.7


# --- Fix 3: speaker runs never cut mid-word ----------------------------------


def test_label_thrash_inside_one_word_collapses_to_single_run():
    # A-B-A over the 3 chars of one jieba word "大碴子" (30-80ms turns).
    atoms = _atoms([("大", 0.0, 0.08), ("碴", 0.08, 0.16), ("子", 0.16, 0.24)])
    turns = [(0.0, 0.08, "A"), (0.08, 0.16, "B"), (0.16, 0.24, "A")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert len(runs) == 1


def test_label_flip_inside_jieba_word_snaps_to_word_edge():
    # "大碴子" is one jieba lexeme; a mid-word speaker flip must snap to the word
    # edge so the word is never split across cues.
    atoms = _atoms([("大", 0.0, 0.3), ("碴", 0.3, 0.65), ("子", 0.65, 0.9)])
    turns = [(0.0, 0.3, "A"), (0.3, 1.0, "B")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert len(runs) == 1
    assert "".join(a["text"] for _, ats in runs for a in ats) == "大碴子"


def test_genuine_second_speaker_run_still_splits():
    # Two distinct jieba words, each a >0.5s speaker run: must stay two runs.
    atoms = _atoms(
        [("你", 0.0, 0.3), ("好", 0.3, 0.6), ("朋", 0.6, 0.9), ("友", 0.9, 1.2)]
    )
    turns = [(0.0, 0.6, "A"), (0.6, 1.2, "B")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert [("".join(a["text"] for a in ats)) for _, ats in runs] == [
        "你好",
        "朋友",
    ]
