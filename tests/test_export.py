# tests/test_export.py
# SRT/ASS export from the sibling VTT: timestamps re-rendered per format,
# <i> tags pass through to SRT and become {\i1}/{\i0} in ASS, plain-text
# edit drafts (no timestamps) are rejected.
import pytest

from voxweave.export import (
    _ass_ts,
    _srt_ts,
    ass_header,
    export_subtitles,
    render_ass,
    render_srt,
    render_vtt_rows,
)
from voxweave.subformats import parse_ass_blocks

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
    with pytest.raises(ValueError, match="already .srt") as info:
        export_subtitles(srt, ("srt",))
    # Names the visible option with a format that differs from the source.
    assert "pick another --format (e.g. -f vtt)" in str(info.value)
    assert "--to" not in str(info.value)


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


def test_export_sidecarless_sdh_srt_preserves_literal_speaker_labels(tmp_path):
    srt = tmp_path / "sdh.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nMAN: Get down!\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nMAN: Now!\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nWOMAN: I can't.\n\n"
        "4\n00:00:07,000 --> 00:00:08,000\nWOMAN: Help me.\n",
        encoding="utf-8",
    )

    export_subtitles(srt, ("vtt", "ass"))
    vtt = (tmp_path / "sdh.vtt").read_text(encoding="utf-8")
    ass = (tmp_path / "sdh.ass").read_text(encoding="utf-8")

    assert "\nMAN: Get down!\n" in vtt
    assert "\nWOMAN: Help me.\n" in vtt
    assert "<v " not in vtt
    assert "Default,,0,0,0,,MAN: Get down!" in ass
    assert "Default,,0,0,0,,WOMAN: Help me." in ass


def test_export_srt_ignores_corrupt_speaker_sidecar_once(tmp_path, caplog):
    srt = tmp_path / "ep.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nAoi: Hello\n", encoding="utf-8")
    (tmp_path / "ep.speakers.json").write_text("", encoding="utf-8")

    with caplog.at_level("WARNING", logger="voxweave"):
        export_subtitles(srt, ("vtt",))

    rendered = (tmp_path / "ep.vtt").read_text(encoding="utf-8")
    assert "\nAoi: Hello\n" in rendered
    assert (
        sum(
            "ignoring unreadable speaker mapping" in record.message
            for record in caplog.records
        )
        == 1
    )


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


def test_export_does_not_promote_foreign_ass_actor_names(tmp_path):
    ass = tmp_path / "fansub.ass"
    ass.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,sign,0,0,0,,Chapter One\n"
        "Dialogue: 0,0:00:04.00,0:00:06.00,Default,TS note: fix later,0,0,0,,Hello there\n"
        "Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,Plain cue\n",
        encoding="utf-8",
    )

    export_subtitles(ass, ("srt", "vtt"))
    srt = (tmp_path / "fansub.srt").read_text(encoding="utf-8")
    vtt = (tmp_path / "fansub.vtt").read_text(encoding="utf-8")

    assert "\nChapter One\n" in srt and "\nHello there\n" in srt
    assert "sign: Chapter One" not in srt and "TS note" not in srt
    assert "<v " not in vtt
    parsed = parse_ass_blocks(ass.read_text(encoding="utf-8"))
    assert ["Chapter One", "Hello there", "Plain cue"] == [
        block["text"] for block in parsed
    ]
    assert all("speaker" not in block for block in parsed)


def test_ass_header_strips_commas_from_font_name():
    # a comma in the font name would shift every field after Fontname in the
    # "Style:" line (ASS is comma-delimited), corrupting the style entirely.
    default_line = next(
        line for line in ass_header().splitlines() if line.startswith("Style:")
    )
    default_commas = default_line.count(",")
    h = ass_header(font="Weird, Font")
    style_line = next(line for line in h.splitlines() if line.startswith("Style:"))
    assert "Weird Font" in style_line
    assert style_line.count(",") == default_commas


def test_ass_keeps_srt_position_tag_as_override():
    # {\an8} is the de facto SRT "raise to the top" tag; ASS must receive it as
    # a real override, never as literal "(\an8)" text
    ass = render_ass([(0.0, 1.0, "{\\an8}Top line {raw}")])
    assert "Default,,0,0,0,,{\\an8}Top line (raw)" in ass
    assert "(\\an8)" not in ass
    # legacy SSA numbering (\a6 = top center) is an ASS override as well
    assert "Default,,0,0,0,,{\\a6}Top" in render_ass([(0.0, 1.0, "{\\a6}Top")])
    # only the first position tag counts (as in libass); the tag moves to the front
    ass = render_ass([(0.0, 1.0, "<i>Hi</i>{\\an8} there{\\an2}")])
    assert "Default,,0,0,0,,{\\an8}{\\i1}Hi{\\i0} there" in ass


def test_srt_and_vtt_never_show_position_tag_as_text():
    rows = [(0.0, 1.0, "{\\an8}Top line")]
    # SRT players honor the tag, so it stays (at the start of the cue)
    assert "\n{\\an8}Top line\n" in render_srt(rows)
    # the VTT writer has no cue settings: the tag is dropped, the text kept
    vtt = render_vtt_rows(rows)
    assert "\\an8" not in vtt and "\nTop line\n" in vtt


def test_export_srt_position_tag_to_ass_and_vtt(tmp_path):
    srt = tmp_path / "ep.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}Sign on the wall\n", encoding="utf-8"
    )
    export_subtitles(srt, ("ass", "vtt"))
    ass = (tmp_path / "ep.ass").read_text(encoding="utf-8")
    assert "Default,,0,0,0,,{\\an8}Sign on the wall" in ass
    vtt = (tmp_path / "ep.vtt").read_text(encoding="utf-8")
    assert "an8" not in vtt and "Sign on the wall" in vtt


def test_export_rejects_partly_timed_vtt(tmp_path):
    vtt = tmp_path / "ep.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nfirst\n\njust text\n\n"
        "00:00:02.000 --> 00:00:03.000\nthird\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as info:
        export_subtitles(vtt, ("srt", "ass"))
    msg = str(info.value)
    assert "ep.vtt has cues without timestamps (cue 2" in msg
    assert "voxweave align ep.vtt" in msg
    assert not (tmp_path / "ep.srt").exists() and not (tmp_path / "ep.ass").exists()


def test_export_partly_timed_error_lists_only_the_first_few(tmp_path):
    vtt = tmp_path / "ep.vtt"
    cues = ["00:00:00.000 --> 00:00:01.000\ntimed"] + [f"loose {n}" for n in range(8)]
    vtt.write_text("WEBVTT\n\n" + "\n\n".join(cues) + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match=r"\(cue 2, 3, 4, 5, 6, \.\.\. 8 in all; first: 'loose 0'\)"
    ):
        export_subtitles(vtt, ("srt",))


def test_export_partly_timed_srt_names_the_fix(tmp_path):
    # a blank line inside an SRT cue body splits off an untimed fragment; align
    # takes VTT only, so the fix is the timing line itself
    srt = tmp_path / "ep.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nfirst\n\nstray second line\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nsecond\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as info:
        export_subtitles(srt, ("vtt",))
    msg = str(info.value)
    assert "(cue 2; first: 'stray second line')" in msg
    assert "add their timing lines" in msg and "align" not in msg


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
