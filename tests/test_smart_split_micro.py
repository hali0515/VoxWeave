# tests/test_smart_split_micro.py
# _merge_micro_cues: adjacent cues with sub-glue gaps merge when the result still
# fits one display line and the duration cap — micro-sentence chains (そう。だね。)
# become one readable cue instead of a flicker sequence. Real pauses, over-line
# merges, and over-duration merges must all stay separate.
from voxweave.core.smart_split import smart_split_segments

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
