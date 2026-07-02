# tests/test_subformats.py
# Extension-dispatched subtitle loading: SRT rides the VTT parser, ASS/SSA get
# a dedicated Events parser (Format-aware field order, override stripping,
# italic mapping, lyric flag), unknown extensions are rejected.
from pathlib import Path

import pytest

from voxweave.subformats import (
    load_subtitle_blocks,
    parse_ass_blocks,
    require_subtitle,
)

ASS_DOC = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,72

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:00.50,Default,,0,0,0,,skip me
Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,hello\\Nworld
Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,{\\pos(10,20)}styled {\\i1}part{\\i0} here
Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,out of order
"""


def test_parse_ass_dialogue_and_ordering():
    blocks = parse_ass_blocks(ASS_DOC)
    assert [b["text"] for b in blocks] == [
        "hello\nworld",
        "out of order",  # sorted by start time
        "styled <i>part</i> here",  # overrides dropped, italics mapped
    ]
    assert blocks[0]["start"] == 1.0 and blocks[0]["end"] == 2.5


def test_parse_ass_full_line_italic_lyric():
    doc = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}♪ la la ♪{\\i0}\n"
    )
    blocks = parse_ass_blocks(doc)
    assert blocks == [{"text": "la la", "start": 1.0, "end": 2.0, "lyric": True}]


def test_parse_ass_without_format_line_uses_default_order():
    doc = "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,plain\n"
    blocks = parse_ass_blocks(doc)
    assert blocks == [{"text": "plain", "start": 1.0, "end": 2.0}]


def test_parse_ass_text_with_commas_survives():
    doc = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,one, two, three\n"
    )
    assert parse_ass_blocks(doc)[0]["text"] == "one, two, three"


def test_load_srt_via_vtt_parser(tmp_path):
    srt = tmp_path / "ep.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n"
        "2\n00:00:03,500 --> 00:00:04,000\nworld\n",
        encoding="utf-8",
    )
    blocks = load_subtitle_blocks(srt)
    assert [b["text"] for b in blocks] == ["hello", "world"]
    assert blocks[1]["start"] == 3.5


def test_load_ssa_uses_ass_parser(tmp_path):
    ssa = tmp_path / "ep.ssa"
    ssa.write_text(
        "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,hi\n",
        encoding="utf-8",
    )
    assert load_subtitle_blocks(ssa)[0]["text"] == "hi"


def test_load_rejects_unknown_extension(tmp_path):
    txt = tmp_path / "ep.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported subtitle format"):
        load_subtitle_blocks(txt)


def test_load_empty_file_raises(tmp_path):
    ass = tmp_path / "ep.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no cues"):
        load_subtitle_blocks(ass)


def test_require_subtitle_custom_exts():
    with pytest.raises(ValueError, match="expected vtt"):
        require_subtitle(Path("ep.srt"), exts=(".vtt",))


# --- content sniffing (extension vs actual format) ---------------------------


def test_load_rejects_ass_content_renamed_vtt(tmp_path):
    # ASS renamed .vtt used to parse [Script Info] lines into garbage cues and,
    # via align --apply, could overwrite the file with bogus VTT
    p = tmp_path / "ep.vtt"
    p.write_text(ASS_DOC, encoding="utf-8")
    with pytest.raises(RuntimeError, match="ASS/SSA"):
        load_subtitle_blocks(p)


def test_load_rejects_ass_content_renamed_srt(tmp_path):
    p = tmp_path / "ep.srt"
    p.write_text(ASS_DOC, encoding="utf-8")
    with pytest.raises(RuntimeError, match="ASS/SSA"):
        load_subtitle_blocks(p)


def test_load_rejects_vtt_content_renamed_ass(tmp_path):
    p = tmp_path / "ep.ass"
    p.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="WebVTT"):
        load_subtitle_blocks(p)


def test_load_ass_starting_at_events_still_loads(tmp_path):
    # headerless-but-valid ASS (no [Script Info]) must not trip the sniffer
    p = tmp_path / "ep.ass"
    p.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,plain\n",
        encoding="utf-8",
    )
    assert load_subtitle_blocks(p) == [{"text": "plain", "start": 1.0, "end": 2.0}]


def test_load_vtt_content_in_srt_is_tolerated(tmp_path):
    # .vtt and .srt share a parser; a WEBVTT doc named .srt still loads fine
    p = tmp_path / "ep.srt"
    p.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8")
    assert load_subtitle_blocks(p)[0]["text"] == "hello"


# --- encoding tolerance ------------------------------------------------------

SRT_DOC = "1\n00:00:00,000 --> 00:00:01,000\ncafé naïve\n"
VTT_DOC = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n"


def test_load_vtt_with_utf8_bom(tmp_path):
    p = tmp_path / "ep.vtt"
    p.write_bytes(b"\xef\xbb\xbf" + VTT_DOC.encode("utf-8"))
    blocks = load_subtitle_blocks(p)
    # BOM must not turn the WEBVTT header into a phantom text cue
    assert [b["text"] for b in blocks] == ["hello"]
    assert blocks[0]["start"] == 0.0


def test_load_srt_utf16(tmp_path):
    p = tmp_path / "ep.srt"
    p.write_bytes(SRT_DOC.encode("utf-16"))  # BOM included by the codec
    blocks = load_subtitle_blocks(p)
    assert [b["text"] for b in blocks] == ["café naïve"]


def test_load_srt_utf16_be(tmp_path):
    p = tmp_path / "ep.srt"
    p.write_bytes(b"\xfe\xff" + SRT_DOC.encode("utf-16-be"))
    blocks = load_subtitle_blocks(p)
    assert [b["text"] for b in blocks] == ["café naïve"]


def test_load_srt_gbk_fallback(tmp_path):
    p = tmp_path / "ep.srt"
    doc = "1\n00:00:00,000 --> 00:00:01,000\n你好世界\n"
    p.write_bytes(doc.encode("gbk"))
    blocks = load_subtitle_blocks(p)
    assert [b["text"] for b in blocks] == ["你好世界"]


def test_load_srt_undecodable_raises_readable_error(tmp_path):
    p = tmp_path / "ep.srt"
    # invalid in utf-8, gb18030 and cp1252 alike
    p.write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\n\xff\x81\x00\x80\xff\n")
    with pytest.raises(RuntimeError, match="encoding"):
        load_subtitle_blocks(p)


def test_parse_vtt_blocks_tolerates_leading_bom():
    from voxweave.realign import parse_vtt_blocks

    blocks = parse_vtt_blocks("﻿" + VTT_DOC)
    assert [b["text"] for b in blocks] == ["hello"]
