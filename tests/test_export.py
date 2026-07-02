# tests/test_export.py
# SRT/ASS export from the sibling VTT: timestamps re-rendered per format,
# <i> tags pass through to SRT and become {\i1}/{\i0} in ASS, plain-text
# edit drafts (no timestamps) are rejected.
import pytest

from voxweave.export import (
    _ass_ts,
    _srt_ts,
    export_subtitles,
    render_ass,
    render_srt,
)

ROWS = [
    (0.0, 1.25, "Hello there"),
    (3661.5, 3662.0, "line one\nline two"),
]


def test_srt_timestamp_format():
    assert _srt_ts(0.0) == "00:00:00,000"
    assert _srt_ts(3661.5) == "01:01:01,500"


def test_ass_timestamp_format():
    assert _ass_ts(0.0) == "0:00:00.00"
    assert _ass_ts(3661.5) == "1:01:01.50"


def test_render_srt_numbered_cues():
    srt = render_srt(ROWS)
    assert "1\n00:00:00,000 --> 00:00:01,250\nHello there" in srt
    assert "2\n01:01:01,500 --> 01:01:02,000\nline one\nline two" in srt


def test_render_ass_events_and_linebreaks():
    ass = render_ass(ROWS)
    assert "[V4+ Styles]" in ass and "Style: Default," in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.25,Default,,0,0,0,,Hello there" in ass
    assert "line one\\Nline two" in ass


def test_ass_italics_and_brace_neutralization():
    ass = render_ass([(0.0, 1.0, "<i>sung line</i> {raw}")])
    assert "{\\i1}sung line{\\i0}" in ass
    assert "{raw}" not in ass  # braces would open an override block


def test_srt_keeps_italic_tags():
    srt = render_srt([(0.0, 1.0, "<i>sung line</i>")])
    assert "<i>sung line</i>" in srt


def test_export_writes_siblings(tmp_path):
    vtt = tmp_path / "ep.01.vtt"  # interior dot: must not truncate the stem
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n", encoding="utf-8")
    paths = export_subtitles(vtt, ("srt", "ass", "srt"))
    assert [p.name for p in paths] == ["ep.01.srt", "ep.01.ass"]  # deduped
    assert (tmp_path / "ep.01.srt").read_text(encoding="utf-8").startswith("1\n")


def test_export_rejects_plain_text_draft(tmp_path):
    vtt = tmp_path / "draft.vtt"
    vtt.write_text("WEBVTT\n\njust text no timing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="align"):
        export_subtitles(vtt, ("srt",))


def test_export_rejects_unknown_format(tmp_path):
    vtt = tmp_path / "x.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        export_subtitles(vtt, ("sub",))


def test_export_rejects_same_format_as_source(tmp_path):
    srt = tmp_path / "x.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already .srt"):
        export_subtitles(srt, ("srt",))


def test_export_srt_input_to_vtt_and_ass(tmp_path):
    srt = tmp_path / "ep.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,500\nhello\nworld\n", encoding="utf-8")
    paths = export_subtitles(srt, ("vtt", "ass"))
    assert [p.name for p in paths] == ["ep.vtt", "ep.ass"]
    vtt = (tmp_path / "ep.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:02.500\nhello\nworld" in vtt
    ass = (tmp_path / "ep.ass").read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,hello\\Nworld" in ass


def test_export_ass_input_to_srt(tmp_path):
    ass = tmp_path / "ep.ass"
    ass.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,hi there\n",
        encoding="utf-8",
    )
    paths = export_subtitles(ass, ("srt",))
    assert [p.name for p in paths] == ["ep.srt"]
    srt = (tmp_path / "ep.srt").read_text(encoding="utf-8")
    assert "1\n00:00:01,000 --> 00:00:02,000\nhi there" in srt


def test_export_restores_lyric_wrap(tmp_path):
    # keep-lyrics VTTs store the music-note wrap as a flag; export must render it
    # (and ASS then italicizes the line).
    vtt = tmp_path / "ep.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n♪ la la ♪\n", encoding="utf-8"
    )
    export_subtitles(vtt, ("srt", "ass"))
    srt = (tmp_path / "ep.srt").read_text(encoding="utf-8")
    assert "♪ la la ♪" in srt
    ass = (tmp_path / "ep.ass").read_text(encoding="utf-8")
    assert "{\\i1}♪ la la ♪{\\i0}" in ass
