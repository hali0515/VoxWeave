# tests/test_smart_split_micro.py
# _merge_micro_cues: adjacent cues with sub-glue gaps merge when the result still
# fits one display line and the duration cap — micro-sentence chains (そう。だね。)
# become one readable cue instead of a flicker sequence. Real pauses, over-line
# merges, and over-duration merges must all stay separate.
#
# Defect A (forced-alignment collapse): a cluster of abutting sub-frame cues
# (degenerate uniform word timestamps) must fold into one legible cue even when
# the joined text overruns the 1-line budget — the flicker is unreadable
# otherwise. The escape must NEVER fire on real cues.
import pytest

from voxweave.core.smart_split import smart_split_segments
from voxweave.core.timing import DEGENERATE_CUE_S, _cleanup_cues, _merge_micro_cues

TH = {
    "clause_ms": 400,
    "vad_skip_ms": 1000,
    "offline_ms": 700,
    "min_cue_s": 0.0,
    "max_cue_s": 7.0,
}


def _seg(words, lang):
    text = ("" if lang in {"zh", "ja"} else " ").join(w["word"] for w in words)
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "words": words,
    }


def _ja_words(spec):
    """spec: list of (char, start, end)."""
    return [{"word": c, "start": s, "end": e} for c, s, e in spec]


def test_micro_sentences_merge_when_contiguous():
    # そう。だね。 with a 0.1s gap -> one cue (period becomes the separator space)
    words = _ja_words(
        [
            ("そ", 0.0, 0.2),
            ("う", 0.2, 0.4),
            ("。", 0.4, 0.4),
            ("だ", 0.5, 0.7),
            ("ね", 0.7, 0.9),
            ("。", 0.9, 0.9),
        ]
    )
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=None, thresholds=TH
    )
    assert len(cues) == 1, [c["text"] for c in cues]
    assert cues[0]["text"] == "そう だね"
    assert cues[0]["start"] == 0.0 and cues[0]["end"] == 0.9


def test_micro_sentences_stay_apart_across_real_pause():
    # same chain but a 0.6s pause between sentences -> two cues
    words = _ja_words(
        [
            ("そ", 0.0, 0.2),
            ("う", 0.2, 0.4),
            ("。", 0.4, 0.4),
            ("だ", 1.0, 1.2),
            ("ね", 1.2, 1.4),
            ("。", 1.4, 1.4),
        ]
    )
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=None, thresholds=TH
    )
    assert len(cues) == 2, [c["text"] for c in cues]


def test_micro_merge_respects_single_line_budget():
    # two 12-char sentences: merged 24 chars > 18-char single line -> stay separate
    a = "あいうえおかきくけこさし"
    b = "たちつてとなにぬねのはひ"
    spec = []
    t = 0.0
    for ch in a:
        spec.append((ch, t, t + 0.1))
        t += 0.1
    spec.append(("。", t, t))
    t += 0.1  # 0.1s gap
    for ch in b:
        spec.append((ch, t, t + 0.1))
        t += 0.1
    spec.append(("。", t, t))
    cues = smart_split_segments(
        [_seg(_ja_words(spec), "ja")], "ja", speech_spans=None, thresholds=TH
    )
    assert len(cues) == 2, [c["text"] for c in cues]


def test_micro_merge_respects_duration_cap():
    # slow speech: merged span would exceed max_cue_s -> stay separate
    words = _ja_words(
        [
            ("そ", 0.0, 2.0),
            ("う", 2.0, 4.0),
            ("。", 4.0, 4.0),
            ("だ", 4.2, 6.0),
            ("ね", 6.0, 7.5),
            ("。", 7.5, 7.5),
        ]
    )
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=None, thresholds=TH
    )
    assert len(cues) == 2, [c["text"] for c in cues]


def test_micro_merge_english_pair():
    words = [
        {"word": "Yeah.", "start": 0.0, "end": 0.3},
        {"word": "Right.", "start": 0.4, "end": 0.7},
    ]
    cues = smart_split_segments(
        [_seg(words, "en")], "en", speech_spans=None, thresholds=TH
    )
    assert len(cues) == 1
    assert cues[0]["text"] == "Yeah Right"


# --- Defect A: degenerate micro-cluster collapse escape ---------------------


def _cue(text, start, end):
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [{"word": text, "start": start, "end": end}],
    }


# Real E2E anime shape: window 1208.900-1208.963 packed 6 cues into 63ms because
# ASR text overran its aligned span (uniform 2-12ms word timestamps). Joined text
# (26 chars) blows the 18-char ja single-line budget, so the ordinary budget gate
# declines the merge.
_DEGENERATE_SPANS = [
    ("反転する", 1208.900, 1208.903),  # 4 chars, 3ms
    ("うまいこと", 1208.910, 1208.915),  # 5 chars, 5ms
    ("がするね", 1208.920, 1208.927),  # 4 chars, 7ms
    ("だろうな", 1208.933, 1208.938),  # 4 chars, 5ms
    ("そうかな", 1208.945, 1208.951),  # 4 chars, 6ms
    ("なるほどね", 1208.957, 1208.963),  # 5 chars, 6ms
]


def test_degenerate_cluster_collapses_to_one_cue():
    # (a) 6 abutting sub-floor cues totaling 63ms -> a single merged cue with all
    # text preserved, spanning the full available run (merge does not extend).
    cues = [_cue(t, s, e) for t, s, e in _DEGENERATE_SPANS]
    # over-budget guard: the ordinary path must be declining this merge
    assert sum(len(t) for t, _, _ in _DEGENERATE_SPANS) > 18
    out = _merge_micro_cues(
        cues, "ja", max_gap_s=0.3, max_line_length=18, max_cue_s=7.0
    )
    assert len(out) == 1, [c["text"] for c in out]
    text = out[0]["text"].replace("\n", "")
    for t, _, _ in _DEGENERATE_SPANS:
        assert t in text  # every fragment preserved
    assert out[0]["start"] == pytest.approx(1208.900)
    assert out[0]["end"] == pytest.approx(1208.963)
    assert out[0]["end"] - out[0]["start"] == pytest.approx(0.063)


def test_degenerate_cluster_reaches_floor_after_cleanup():
    # (b) after the merge + cleanup passes (pipeline order), the collapsed cue is
    # extended into the following gap to at least the degenerate floor: no output
    # cue remains a sub-frame flicker.
    cues = [_cue(t, s, e) for t, s, e in _DEGENERATE_SPANS]
    cues.append(_cue("次の台詞です", 1215.0, 1216.5))  # real cue, far away
    merged = _merge_micro_cues(
        cues, "ja", max_gap_s=0.3, max_line_length=18, max_cue_s=7.0, min_cue_s=0.5
    )
    out = _cleanup_cues(merged, min_cue_s=0.5, max_cue_s=7.0)
    assert all(c["end"] - c["start"] >= DEGENERATE_CUE_S - 1e-9 for c in out), [
        c["end"] - c["start"] for c in out
    ]


def test_run_below_min_cue_merges_over_budget():
    # a run of abutting cues each above the degenerate floor (0.15s) but which
    # collectively cannot reach min_cue_s, with over-budget joined text: the
    # run-length escape folds them so cleanup can extend the single cue.
    cues = [
        _cue("長いテキストのA", 0.0, 0.15),  # 8 chars
        _cue("長いテキストのB", 0.155, 0.30),
        _cue("長いテキストのC", 0.305, 0.45),
    ]
    out = _merge_micro_cues(
        cues, "ja", max_gap_s=0.3, max_line_length=18, max_cue_s=7.0, min_cue_s=0.5
    )
    assert len(out) == 1, [c["text"] for c in out]
    assert out[0]["end"] - out[0]["start"] == pytest.approx(0.45)


def test_micro_merge_leaves_normal_cues_untouched():
    # (c) two 1.5s cues abutting (0.1s gap) with over-budget joined text: the
    # ordinary path declines (over one line) and the escape must not fire — both
    # are far above the degenerate floor and their run far exceeds min_cue_s.
    cues = [
        _cue("これはとても長い文章です", 0.0, 1.5),  # 12 chars
        _cue("そしてこれも長い文章だよね", 1.6, 3.1),  # 13 chars
    ]
    out = _merge_micro_cues(
        cues, "ja", max_gap_s=0.3, max_line_length=18, max_cue_s=7.0, min_cue_s=0.5
    )
    assert len(out) == 2, [c["text"] for c in out]


def test_micro_merge_leaves_interjection_with_real_gap_untouched():
    # (d) a legit 0.3s interjection with a real (>= max_gap_s) pause on both sides
    # is content, not a collapse artifact: never merged.
    cues = [
        _cue("はい", 0.0, 1.0),
        _cue("えっ", 1.5, 1.8),  # 0.3s interjection, 0.5s gap before/after
        _cue("そうだね", 2.3, 3.3),
    ]
    out = _merge_micro_cues(
        cues, "ja", max_gap_s=0.3, max_line_length=18, max_cue_s=7.0, min_cue_s=0.5
    )
    assert len(out) == 3, [c["text"] for c in out]
