# tests/test_keep_lyrics.py
# Keep-lyrics mode: cues mostly overlapping detected singing spans get the
# lyric flag; the VTT renders them wrapped in music notes while the JSON keeps
# clean text + the flag; align/translate round-trip the wrap via parse/render;
# split replays the marking from persisted sing_spans.
import json

from voxweave import pipeline, realign
from voxweave.export import render_ass
from voxweave.pipeline import lyric_display_text, mark_lyric_cues


def _cue(start, end, text="la la la"):
    return {"text": text, "start": start, "end": end, "word_data": []}


def test_mark_lyric_cues_by_overlap():
    cues = [_cue(0.0, 2.0), _cue(10.0, 12.0)]
    mark_lyric_cues(cues, [(0.5, 3.0)])  # 75% of cue 1, 0% of cue 2
    assert cues[0].get("lyric") is True
    assert "lyric" not in cues[1]


def test_mark_lyric_cues_below_threshold_untouched():
    cues = [_cue(0.0, 4.0)]
    mark_lyric_cues(cues, [(0.0, 1.0)])  # 25% overlap < 50%
    assert "lyric" not in cues[0]


def test_lyric_display_text_wraps():
    assert lyric_display_text({"text": "hello", "lyric": True}) == "♪ hello ♪"
    assert lyric_display_text({"text": "hello"}) == "hello"


def test_parse_vtt_blocks_strips_and_flags_lyric_wrap():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n♪ sung line ♪\n\nplain text\n"
    blocks = realign.parse_vtt_blocks(vtt)
    assert blocks[0]["text"] == "sung line" and blocks[0].get("lyric") is True
    assert blocks[1]["text"] == "plain text" and "lyric" not in blocks[1]


def test_render_vtt_restores_lyric_wrap():
    blocks = [{"text": "sung line", "lyric": True}, {"text": "plain"}]
    out = realign.render_vtt(blocks, [(1.0, 2.0), (3.0, 4.0)])
    assert "♪ sung line ♪" in out
    assert "♪ plain" not in out


def test_ass_export_italicizes_lyric_cues():
    ass = render_ass([(0.0, 1.0, "♪ sung line ♪")])
    assert "{\\i1}♪ sung line ♪{\\i0}" in ass


def test_split_replays_sing_spans(tmp_path):
    units = [
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.6, "end": 1.0},
        {"text": "lala", "start": 10.0, "end": 10.5},
        {"text": "lala", "start": 10.6, "end": 11.0},
    ]
    json_path = tmp_path / "clip.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "en",
                "word_segments": units,
                "segments": [],
                "vad_speech": [],
                "sing_spans": [[9.5, 11.5]],
            }
        ),
        encoding="utf-8",
    )
    vtt_out = pipeline.split(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["sing_spans"] == [[9.5, 11.5]]  # round-trips
    lyric_segs = [s for s in data["segments"] if s.get("lyric")]
    assert lyric_segs and all("♪" not in s["text"] for s in lyric_segs)
    vtt = vtt_out.read_text(encoding="utf-8")
    assert "♪" in vtt  # display wrap lives in the VTT only
