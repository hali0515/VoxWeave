from voxweave import realign


def _texts(blocks):
    return [b["text"] for b in blocks]


# --------------------------------------------------------------------------- #
# parse_vtt_blocks: WEBVTT header / NOTE / STYLE / REGION per the WebVTT grammar
# --------------------------------------------------------------------------- #
def test_plain_draft_keeps_dialogue_starting_with_keywords():
    # `render --no-timestamps` draft: cue text that merely begins with a keyword-like
    # word is dialogue, not a NOTE/STYLE/REGION block.
    lines = [
        "Noted, sir.",
        "Notes are on the desk.",
        "note to self",
        "Styles change.",
        "Stylish shoes!",
        "Regional office.",
        "Region two is closed.",
        "Webvtt files are text.",
        "Hello.",
    ]
    vtt = realign.render_cues([(None, None, t) for t in lines])
    assert _texts(realign.parse_vtt_blocks(vtt)) == lines


def test_timed_cues_starting_with_note_are_kept():
    vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\nNote to self: buy milk.\n\n"
        "NOTE\n00:00:03.000 --> 00:00:04.000\nNOTE THIS DOWN\n\n"
        "Styles\n00:00:05.000 --> 00:00:06.000\nRegional office.\n"
    )
    blocks = realign.parse_vtt_blocks(vtt)
    assert _texts(blocks) == [
        "Note to self: buy milk.",
        "NOTE THIS DOWN",  # "NOTE" first line is a cue id: the block has a timing line
        "Regional office.",
    ]
    assert [(b["start"], b["end"]) for b in blocks] == [
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
    ]


def test_real_vtt_header_note_style_region_still_skipped():
    vtt = (
        "WEBVTT - Episode 1\nKind: captions\nLanguage: en\n\n"
        "NOTE\nmulti-line\ncomment\n\n"
        "NOTE\tcomment after a tab\n\n"
        "STYLE\n::cue { color: yellow }\n\n"
        "REGION\nid:fred\nwidth:40%\n\n"
        "REGION id:bill\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello\n\n"
        "NOTE trailing comment\n\n"
        "00:00:03.000 --> 00:00:04.000\nworld\n"
    )
    blocks = realign.parse_vtt_blocks(vtt)
    assert _texts(blocks) == ["hello", "world"]
    assert blocks[0]["start"] == 1.0 and blocks[1]["start"] == 3.0


def test_webvtt_header_is_case_insensitive_first_block_only():
    assert _texts(realign.parse_vtt_blocks("webvtt\n\nhi\n")) == ["hi"]
    assert _texts(realign.parse_vtt_blocks("﻿\n\nWEBVTT\n\nhi\n")) == ["hi"]
    # Not the first block: dialogue, not a header.
    assert _texts(realign.parse_vtt_blocks("WEBVTT\n\nhi\n\nWebVTT rocks.\n")) == [
        "hi",
        "WebVTT rocks.",
    ]
    # Headerless draft: a first cue merely starting with the letters is kept.
    assert _texts(realign.parse_vtt_blocks("WEBVTTish\n\nhi\n")) == ["WEBVTTish", "hi"]


def test_lowercase_keyword_lines_are_not_comments():
    # Only the exact uppercase keyword opens a NOTE/STYLE/REGION block.
    vtt = "WEBVTT\n\nnote the time\n\nStyle over substance\n\nREGIONAL news\n"
    assert _texts(realign.parse_vtt_blocks(vtt)) == [
        "note the time",
        "Style over substance",
        "REGIONAL news",
    ]


# --------------------------------------------------------------------------- #
# fuse_punct_into_text: numbers survive strip_existing
# --------------------------------------------------------------------------- #
def _words(s):
    return [{"text": w, "start": 0.0, "end": 0.0} for w in s.split()]


def test_strip_existing_keeps_number_separators_without_qwen_punct():
    text = "it rose 14.2 percent to 1,000 units, fine."
    assert (
        realign.fuse_punct_into_text(text, [])
        == "it rose 14.2 percent to 1,000 units fine"
    )


def test_strip_existing_keeps_number_separators_in_rebuild():
    # Qwen spells the number differently → whisper's own separators must survive the
    # rebuild instead of collapsing 14.2 → 142 / 1,000 → 1000.
    text = "it rose 14.2 percent to 1,000 units"
    qwen = _words("it rose fourteen point two percent to 1000 units.")
    assert (
        realign.fuse_punct_into_text(text, qwen)
        == "it rose 14.2 percent to 1,000 units."
    )


def test_strip_existing_number_not_doubled_by_matching_qwen_separator():
    text = "it rose 14.2 percent to 1,000 units"
    for qwen in (
        "it rose 14.2 percent to 1,000 units.",
        "it rose 14,2 percent to 1.000 units.",
    ):
        assert (
            realign.fuse_punct_into_text(text, _words(qwen))
            == "it rose 14.2 percent to 1,000 units."
        )


def test_strip_existing_still_strips_sentence_period_after_number():
    text = "it costs 14. Then we left"
    qwen = _words("it costs 14 then we left.")
    assert realign.fuse_punct_into_text(text, qwen) == "it costs 14 Then we left."


# --------------------------------------------------------------------------- #
# fuse_punct_into_text: no double punctuation for no-space languages
# --------------------------------------------------------------------------- #
def _chars(s):
    return [{"text": c, "start": 0.0, "end": 0.0} for c in s]


def test_keep_existing_skips_qwen_punct_where_whisper_has_one():
    text = "畑です。次"
    out = realign.fuse_punct_into_text(text, _chars("畑です。次"), strip_existing=False)
    assert out == "畑です。次"


def test_keep_existing_prefers_whisper_mark_and_adds_missing_ones():
    # whisper 、 wins over Qwen 。 at the same boundary; Qwen's trailing 。 is still added.
    text = "畑です、次"
    out = realign.fuse_punct_into_text(
        text, _chars("畑です。次。"), strip_existing=False
    )
    assert out == "畑です、次。"


def test_keep_existing_no_double_trailing_punct():
    text = "畑次。"
    out = realign.fuse_punct_into_text(text, _chars("畑次。"), strip_existing=False)
    assert out == "畑次。"
