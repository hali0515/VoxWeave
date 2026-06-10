# tests/test_smart_split_glue.py
from voxweave import config
from voxweave.core.smart_split import (
    GLUE_MAX_GAP_S,
    _glue_short_cues,
    smart_split_segments,
)


def _cue(text, start, end):
    return {"text": text, "start": start, "end": end, "word_data": []}


def test_short_word_with_tiny_gap_glues_back():
    # "that" is a lone word 0.1s after the previous cue (no real pause) -> glued
    cues = [_cue("I think", 1.0, 2.0), _cue("that", 2.1, 2.25)]
    out = _glue_short_cues(cues, "en", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 1
    assert out[0]["text"] == "I think that"
    assert out[0]["end"] == 2.25  # end extended to the fragment's end


def test_real_pause_above_threshold_not_glued():
    # 0.45s gap > 0.3s -> a real micro-pause, kept separate
    cues = [_cue("I think", 1.0, 2.0), _cue("that", 2.45, 2.6)]
    out = _glue_short_cues(cues, "en", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 2


def test_multiword_cue_not_a_fragment():
    # tiny gap but the second cue is not a lone word -> not a flicker fragment
    cues = [_cue("I think", 1.0, 2.0), _cue("that we go", 2.1, 2.8)]
    out = _glue_short_cues(cues, "en", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 2


def test_cjk_single_particle_glues_no_space():
    # ja lone particle joins with no separator
    cues = [_cue("そう", 1.0, 1.6), _cue("ね", 1.65, 1.75)]
    out = _glue_short_cues(cues, "ja", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 1
    assert out[0]["text"] == "そうね"
    assert out[0]["end"] == 1.75


def test_cjk_long_tail_not_fragment():
    # 3+ CJK chars is not a flicker fragment even with a tiny gap; neither side glues
    cues = [_cue("わかった", 1.0, 1.6), _cue("ですね", 1.65, 2.0)]
    out = _glue_short_cues(cues, "ja", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 2


def test_disabled_when_threshold_zero():
    cues = [_cue("I think", 1.0, 2.0), _cue("that", 2.1, 2.25)]
    out = _glue_short_cues(cues, "en", max_gap_s=0.0)
    assert len(out) == 2  # disabled -> untouched


def test_chained_fragments_all_glue():
    # "I" then "that", both tiny gaps -> fold into the head cue
    cues = [
        _cue("well", 1.0, 1.4),
        _cue("I", 1.45, 1.55),
        _cue("guess", 1.6, 1.9),
    ]
    out = _glue_short_cues(cues, "en", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 1
    assert out[0]["text"] == "well I guess"
    assert out[0]["end"] == 1.9


def test_leading_fragment_glues_forward():
    # first cue has no predecessor; tiny gap ahead -> glues forward into the next cue
    cues = [_cue("I", 1.0, 1.1), _cue("really think so", 1.15, 2.0)]
    out = _glue_short_cues(cues, "en", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 1
    assert out[0]["text"] == "I really think so"
    assert out[0]["start"] == 1.0  # next cue's start pulled back to the fragment


def test_filler_leads_next_line_glues_forward():
    # え: real pause behind (1.1s), tiny gap ahead (0.083s) -> forward, not stranded
    cues = [
        _cue("ヨハンもそうさね", 8.0, 9.0),
        _cue("え", 10.1, 10.857),
        _cue("開国祭では結構", 10.94, 12.0),
    ]
    out = _glue_short_cues(cues, "ja", max_gap_s=GLUE_MAX_GAP_S)
    assert [c["text"] for c in out] == ["ヨハンもそうさね", "え開国祭では結構"]
    assert out[1]["start"] == 10.1


def test_nearer_side_wins_backward():
    # ん: gap_back 0.0, gap_fwd 0.083 -> backward (smaller) wins
    cues = [
        _cue("報酬で得た自分のお金", 8.0, 9.04),
        _cue("ん", 9.04, 9.857),
        _cue("次の話", 9.94, 11.0),
    ]
    out = _glue_short_cues(cues, "ja", max_gap_s=GLUE_MAX_GAP_S)
    assert out[0]["text"] == "報酬で得た自分のお金ん"
    assert [c["text"] for c in out] == ["報酬で得た自分のお金ん", "次の話"]


def test_isolated_fragment_real_pause_both_sides_kept():
    # real pauses on both sides -> genuine standalone utterance, not merged
    cues = [_cue("そう", 1.0, 1.6), _cue("ん", 2.5, 3.0), _cue("次", 4.0, 4.5)]
    out = _glue_short_cues(cues, "ja", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out) == 3


def test_word_data_is_concatenated():
    cues = [
        {
            "text": "I think",
            "start": 1.0,
            "end": 2.0,
            "word_data": [{"start": 1.0, "end": 2.0}],
        },
        {
            "text": "that",
            "start": 2.1,
            "end": 2.25,
            "word_data": [{"start": 2.1, "end": 2.25}],
        },
    ]
    out = _glue_short_cues(cues, "en", max_gap_s=GLUE_MAX_GAP_S)
    assert len(out[0]["word_data"]) == 2
    assert out[0]["word_data"][-1]["end"] == 2.25


def test_config_exposes_glue_gap_key():
    th = config.gap_thresholds("en")
    assert th["glue_gap_s"] == GLUE_MAX_GAP_S  # 0.3s default, env VOXWEAVE_GLUE_GAP_MS


def test_wired_through_smart_split_segments():
    # sentence-end split makes "yes" its own cue; tiny 0.05s gap -> glued back end-to-end
    segments = [
        {
            "text": "okay so. yes",
            "words": [
                {"start": 0.0, "end": 0.4},
                {"start": 0.4, "end": 0.7},
                {"start": 0.75, "end": 0.95},
            ],
        }
    ]
    cues = smart_split_segments(segments, "en", thresholds=config.gap_thresholds("en"))
    assert len(cues) == 1
    assert cues[0]["text"] == "okay so yes"  # period stripped, fragment glued
    # production thresholds carry lag_out/cps lingering, so the end may extend past
    # speech end (0.95) but never run earlier than it
    assert cues[0]["end"] >= 0.95
