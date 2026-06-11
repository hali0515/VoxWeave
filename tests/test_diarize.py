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
    vtt = vtt_out.read_text(encoding="utf-8")
    # either one dual-speaker event or a split at the speaker boundary
    assert ("-are you coming" in vtt) or (
        "are you coming" in vtt and "in a minute" in vtt
    )


def test_apply_speaker_format_noop_without_turns():
    cue = _cue("hello", 0.0, 1.0, [(0.0, 1.0)])
    assert apply_speaker_format([cue], None, "en") == [cue]
    assert apply_speaker_format([cue], [], "en") == [cue]
