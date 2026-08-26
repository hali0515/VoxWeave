# tests/test_diarize.py
# Speaker-aware cue formatting (pure post-pass, no pyannote/GPU): atoms get a
# speaker by overlap with persisted turns; two-speaker cues become Netflix
# dual-speaker events (-line per speaker, hyphen without space), 3+ speakers or
# over-budget halves split the cue at speaker boundaries with word timing;
# split replays formatting from JSON speaker_turns.
import json

from voxweave import pipeline
from voxweave.diarize import (
    _slice_text_by_runs,
    _span_speaker,
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


TURNS = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]


def test_span_speaker_picks_dominant_overlap():
    assert _span_speaker(0.5, 1.0, TURNS) == "SPEAKER_00"
    assert _span_speaker(1.8, 2.8, TURNS) == "SPEAKER_01"  # 0.8s vs 0.2s
    assert _span_speaker(10.0, 11.0, TURNS) is None


def test_single_speaker_cue_untouched():
    cue = _cue("hello there", 0.0, 1.0, [(0.0, 0.4), (0.5, 1.0)])
    out = format_speaker_cues([cue], TURNS, "en")
    assert out == [cue]


def test_two_speaker_cue_becomes_dual_dash_event():
    cue = _cue(
        "are you coming in a minute",
        0.5,
        3.5,
        [(0.5, 0.8), (0.9, 1.2), (1.3, 1.6), (2.4, 2.7), (2.8, 3.1), (3.2, 3.5)],
    )
    out = format_speaker_cues([cue], TURNS, "en")
    assert len(out) == 1
    assert out[0]["text"] == "-are you coming\n-in a minute"  # hyphen, no space
    assert out[0]["start"] == 0.5 and out[0]["end"] == 3.5


def test_two_speaker_cue_splits_for_single_line_lang():
    # zh renders one line per cue: dash pairing is off, the cue splits instead
    cue = _cue(
        "你来吗 马上就来",
        0.5,
        3.5,
        [(s, s + 0.2) for s in (0.5, 0.7, 0.9, 2.4, 2.6, 2.8, 3.0)],
    )
    out = format_speaker_cues([cue], TURNS, "zh")
    assert [c["text"] for c in out] == ["你来吗", "马上就来"]
    assert out[0]["end"] <= out[1]["start"]
    assert len(out[0]["word_data"]) == 3 and len(out[1]["word_data"]) == 4


def test_lyric_cue_passes_through():
    cue = _cue("la la", 0.5, 3.5, [(0.5, 1.0), (2.5, 3.0)])
    cue["lyric"] = True
    out = format_speaker_cues([cue], TURNS, "en")
    assert out == [cue]


def test_slice_text_preserves_interior_spacing():
    runs = [
        ("A", [{"text": "好"}, {"text": "我们"}]),
        ("B", [{"text": "走吧"}]),
    ]
    assert _slice_text_by_runs("好 我们 走吧", runs) == ["好 我们", "走吧"]


def test_split_replays_speaker_turns(tmp_path):
    units = [
        {"text": "are", "start": 0.5, "end": 0.7},
        {"text": "you", "start": 0.8, "end": 1.0},
        {"text": "coming", "start": 1.1, "end": 1.4},
        {"text": "in", "start": 2.4, "end": 2.5},
        {"text": "a", "start": 2.6, "end": 2.7},
        {"text": "minute", "start": 2.8, "end": 3.1},
    ]
    json_path = tmp_path / "clip.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "en",
                "word_segments": units,
                "segments": [],
                "vad_speech": [],
                "speaker_turns": [[0.0, 2.0, "SPEAKER_00"], [2.0, 4.0, "SPEAKER_01"]],
            }
        ),
        encoding="utf-8",
    )
    vtt_out = pipeline.split(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["speaker_turns"] == [[0.0, 2.0, "SPEAKER_00"], [2.0, 4.0, "SPEAKER_01"]]
    # persisted cue word_data carries each atom's surface: nothing else states
    # the stream's granularity to a later reader
    assert sorted(data["segments"][0]["word_data"][0]) == ["end", "start", "text"]
    vtt = vtt_out.read_text(encoding="utf-8")
    # either one dual-speaker event or a split at the speaker boundary
    assert ("-are you coming" in vtt) or (
        "are you coming" in vtt and "in a minute" in vtt
    )


def test_apply_speaker_format_noop_without_turns():
    cue = _cue("hello", 0.0, 1.0, [(0.0, 1.0)])
    assert apply_speaker_format([cue], None, "en") == [cue]
    assert apply_speaker_format([cue], [], "en") == [cue]


def test_speaker_split_survives_embedded_latin_atom():
    """A repacked cue must be re-read at atom granularity, not per character.

    ``_chunk_to_cue`` emits one word_data entry per packed atom, so an embedded
    Latin run (``GPT``) is one entry covering three characters. Walking that
    stream with a character cursor shifted every later atom onto the wrong
    timestamps: ``GPT`` swallowed ``世界``'s span and the tail atoms got
    ``start=end=None``, which made the speaker boundary land mid-phrase.
    """
    from voxweave.core.smart_split import split_long_cues_with_word_timings

    text = "你好GPT世界真棒学习"
    word_data = [
        {"word": ch, "start": i * 1.0, "end": i * 1.0 + 0.9}
        for i, ch in enumerate(text)
    ]
    packed = split_long_cues_with_word_timings(
        [{"text": text, "start": 0.0, "end": 10.9, "word_data": word_data}],
        max_line_length=18,
        max_lines=1,
        min_duration=0.0,
        desired_wps=4.0,
        lang="zh",
    )
    assert len(packed) == 1  # the fixture must reach the formatter as one cue

    turns = [(0.0, 6.95, "SPEAKER_00"), (6.95, 11.0, "SPEAKER_01")]
    out = format_speaker_cues(packed, turns, "zh")

    assert [c["text"] for c in out] == ["你好GPT世界", "真棒学习"]
    assert out[1]["start"] == 7.0  # 真's own start, not a shifted neighbour's
    assert sum(len(c["word_data"]) for c in out) == len(packed[0]["word_data"])
    assert all(
        w["start"] is not None and w["end"] is not None
        for c in out
        for w in c["word_data"]
    )


def _char_seg(text, step=0.4):
    """One aligner unit per character, the shape ``smart_split_segments`` consumes."""
    words = [
        {
            "word": ch,
            "start": round(i * step, 3),
            "end": round(i * step + step * 0.8, 3),
        }
        for i, ch in enumerate(text)
    ]
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "words": words,
    }


def test_speaker_split_after_real_smart_split_with_punctuation():
    """The production path hands ``format_speaker_cues`` a rendered-vs-frozen pair.

    ``smart_split_segments`` finalizes cue text with ``strip_punct_for_subtitles``
    *after* word_data is packed, so a punctuated CJK cue reaches the formatter
    with entries the text no longer shows (here the comma). Reconciling the two
    sides is what keeps every entry with its own atom.
    """
    from voxweave.core.smart_split import smart_split_segments

    cues = smart_split_segments(
        [_char_seg("上涨,是3.75元了")], "zh", max_line_length=18, thresholds=None
    )
    assert len(cues) == 1
    assert cues[0]["text"] == "上涨 是3.75元了"  # comma stripped, decimal kept
    assert [w["text"] for w in cues[0]["word_data"]] == [
        "上",
        "涨",
        ",",
        "是",
        "3.75",
        "元",
        "了",
    ]

    out = format_speaker_cues(
        cues, [(0.0, 1.4, "SPEAKER_00"), (1.4, 20.0, "SPEAKER_01")], "zh"
    )
    assert [c["text"] for c in out] == ["上涨 是", "3.75元了"]
    assert [[w["text"] for w in c["word_data"]] for c in out] == [
        ["上", "涨", ",", "是"],
        ["3.75", "元", "了"],
    ]
    assert out[1]["start"] == cues[0]["word_data"][4]["start"]  # 3.75's own start


def test_trailing_dropped_punctuation_entry_stays_in_the_last_piece():
    """The last run takes the remainder, so a trailing stripped entry is kept.

    ``友``'s footprint ends before the sentence-final ``。`` entry: without the
    last-run rule that entry would be dropped from the emitted cue and from the
    sibling JSON, silently shortening the stream.
    """
    from voxweave.core.smart_split import smart_split_segments

    cues = smart_split_segments(
        [_char_seg("你好,朋友。")], "zh", max_line_length=18, thresholds=None
    )
    assert len(cues) == 1
    assert cues[0]["text"] == "你好 朋友"
    assert [w["text"] for w in cues[0]["word_data"]][-1] == "。"

    out = format_speaker_cues(
        cues, [(0.0, 1.0, "SPEAKER_00"), (1.0, 20.0, "SPEAKER_01")], "zh"
    )
    assert [c["text"] for c in out] == ["你好", "朋友"]
    assert [w["text"] for w in out[-1]["word_data"]] == [",", "朋", "友", "。"]
    assert sum(len(c["word_data"]) for c in out) == len(cues[0]["word_data"])


def test_char_level_word_data_against_rendered_text_survives_split():
    """The ``--semantic-split`` shape: char-level entries against a rendered text.

    ``_semantic_materialize`` keeps the raw per-character units, so the formatter
    sees a stripped comma entry next to a ``3.75`` the text still shows. The
    stripping rule is context-sensitive, so the stream has to be normalized
    joined -- read entry by entry, the ``.`` looks like a sentence period and
    every atom after it collects the wrong span.
    """
    source = "这个,价格是3.75元"
    cue = {
        "text": "这个 价格是3.75元",
        "start": 0.0,
        "end": 4.3,
        "word_data": [
            {"word": ch, "start": round(i * 0.4, 3), "end": round(i * 0.4 + 0.3, 3)}
            for i, ch in enumerate(source)
        ],
    }
    out = format_speaker_cues(
        [cue], [(0.0, 1.8, "SPEAKER_00"), (1.8, 20.0, "SPEAKER_01")], "zh"
    )
    assert [c["text"] for c in out] == ["这个 价格", "是3.75元"]
    assert sum(len(c["word_data"]) for c in out) == len(source)
    assert out[1]["start"] == cue["word_data"][5]["start"]  # 是's own start


def test_split_replays_coarse_legacy_word_segments(tmp_path):
    """Legacy word_segments hold whole clauses, not characters (see pipeline docs).

    Those cues reach the formatter wider than their word_data (an unlocatable
    clause yields an empty one, which ``_glue_short_cues`` then merges into a
    timed neighbour). That state has to degrade to the old None-filled slicing:
    aborting here throws away ASR, alignment and diarization that already ran.
    """
    json_path = tmp_path / "clip.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": [
                    {"text": "今天我们来看一个问题。", "start": 0.0, "end": 3.0},
                    {"text": "数据中心的上涨。", "start": 3.2, "end": 6.0},
                    {"text": "他说好的很厉害。", "start": 6.3, "end": 9.0},
                ],
                "segments": [],
                "vad_speech": [[0.0, 9.5]],
                "speaker_turns": [
                    [0.0, 3.1, "S0"],
                    [3.1, 6.2, "S1"],
                    [6.2, 9.5, "S0"],
                ],
            }
        ),
        encoding="utf-8",
    )
    vtt = pipeline.split(json_path).read_text(encoding="utf-8")
    assert "今天我们" in vtt and "他说好的很厉害" in vtt
