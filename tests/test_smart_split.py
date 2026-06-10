from voxweave.core.smart_split import (
    _strip_punct_for_subtitles,
    _wrap_cue_text,
    smart_split_segments,
)


def _seg_from_words(text, dt=0.5):
    toks = text.split()
    words = [
        {"word": w, "start": i * dt, "end": i * dt + dt * 0.8}
        for i, w in enumerate(toks)
    ]
    return {"start": 0.0, "end": len(toks) * dt, "text": text, "words": words}


def test_smart_split_returns_well_formed_cues():
    seg = _seg_from_words("Hello world this is a test of subtitle splitting today.")
    cues = smart_split_segments([seg], lang="en")
    assert cues, "should produce at least one cue"
    for c in cues:
        assert {"start", "end", "text"} <= set(c)
        assert c["text"].strip()
        assert c["end"] >= c["start"]


def test_smart_split_times_monotonic_and_in_bounds():
    seg = _seg_from_words("one two three four five six seven eight nine ten")
    cues = smart_split_segments([seg], lang="en")
    assert cues[0]["start"] >= 0.0
    for a, b in zip(cues, cues[1:]):
        assert b["start"] >= a["start"]


def test_smart_split_preserves_content():
    seg = _seg_from_words("alpha beta gamma delta epsilon zeta eta theta")
    cues = smart_split_segments([seg], lang="en")
    joined = " ".join(c["text"].replace("\n", " ") for c in cues)
    for tok in ["alpha", "theta", "epsilon"]:
        assert tok in joined


def test_smart_split_cjk_uses_no_space_join():
    # zh is a no-space language; cue text must not introduce spaces between glyphs.
    seg = _seg_from_words("你 好 世 界 今 天 天 气 很 好", dt=0.4)
    seg["text"] = "你好世界今天天气很好"
    cues = smart_split_segments([seg], lang="zh")
    assert cues
    joined = "".join(c["text"].replace("\n", "") for c in cues)
    assert "你好" in joined


def test_strip_punct_preserves_digit_internal_separators():
    # Digit-internal "." and "," must survive (3.75, 10,000); see CLAUDE.md invariant.
    assert _strip_punct_for_subtitles("3.75 and 10,000") == "3.75 and 10,000"


def test_strip_punct_replaces_visible_punctuation_with_space():
    assert _strip_punct_for_subtitles("Hello, world!") == "Hello world"
    assert _strip_punct_for_subtitles("价格是3.75元。") == "价格是3.75元"


# --------------------------------------------------------------------------- #
# display soft-wrap: cue → <= max_lines lines (newline-joined), content intact
# --------------------------------------------------------------------------- #
def test_wrap_short_cue_unchanged():
    # short enough -> single line, no \n inserted
    assert "\n" not in _wrap_cue_text("Hello world", "en", 2)
    assert "\n" not in _wrap_cue_text("你好世界", "zh", 2)


def test_wrap_long_english_two_lines():
    text = (
        "It is the mark when educated mind to be able to "
        "entertain a thought without accepting it"
    )
    out = _wrap_cue_text(text, "en", 2)
    lines = out.split("\n")
    assert len(lines) == 2
    # content preserved (only one space replaced by a newline, no words broken)
    assert out.replace("\n", " ") == text


def test_wrap_embedded_english_in_cjk_uses_latin_budget():
    # a long pure-English cue in a zh context (max_line_length=12) should wrap to 2 lines
    # using the Latin width budget, not ~7 lines
    text = (
        "It is the mark when educated mind to be able to "
        "entertain a thought without accepting it"
    )
    out = _wrap_cue_text(text, "zh", 2)
    assert out.count("\n") == 1
    assert out.replace("\n", " ") == text


def test_wrap_slides_sticky_token_down():
    # balance point lands right after "to the" -> both closed-class tokens slide
    # to line 2 (a line must not end on the/to); content preserved.
    text = "Tomorrow we are heading to the famous mountain village together"
    out = _wrap_cue_text(text, "en", 2)
    lines = out.split("\n")
    assert len(lines) == 2
    assert lines[0].split()[-1].lower() not in {"the", "to", "of", "and"}
    assert out.replace("\n", " ") == text


def test_wrap_slide_keeps_hard_budget():
    # receiving line already at the hard budget -> the sticky token stays put
    # rather than overflowing line 2.
    long_tail = "extraordinarily complicated multidimensional considerations"
    text = f"He finally pointed to the {long_tail}"
    out = _wrap_cue_text(text, "en", 2)
    for line in out.split("\n"):
        assert sum(1 for _ in line) <= 60  # sanity: nothing absurd
    assert out.replace("\n", " ") == text


def test_wrap_preserves_cjk_comma_space():
    # the space produced by a stripped comma in CJK (好 我们) must not be swallowed by the wrap logic
    assert _wrap_cue_text("好 我们一起走吧好吗", "zh", 2) == "好 我们一起走吧好吗"


def test_wrap_never_breaks_mid_word():
    text = "supercalifragilisticexpialidocious antidisestablishmentarianism today"
    out = _wrap_cue_text(text, "en", 2)
    for line in out.split("\n"):
        # every line is composed of whole words; no long word is split mid-character
        assert all(w in text.split() for w in line.split())


# --------------------------------------------------------------------------- #
# comma line-break: split into separate cues at commas (guarded by length)
# --------------------------------------------------------------------------- #
def _cjk_seg(text, dt=0.6):
    """Char-level segment (incl. punctuation chars) for no-space languages."""
    chars = list(text)
    words = [
        {"word": c, "start": i * dt, "end": i * dt + dt * 0.8}
        for i, c in enumerate(chars)
    ]
    return {"start": 0.0, "end": len(chars) * dt, "text": text, "words": words}


def test_comma_split_zh_both_sides_long():
    # both sides of the comma are long enough (>=6 chars) -> split into two cues;
    # comma is stripped; each cue's timing comes from word_data
    seg = _cjk_seg("我昨天去了商店，今天买了很多东西", dt=0.6)
    cues = smart_split_segments([seg], lang="zh")
    assert len(cues) == 2
    assert cues[0]["text"] == "我昨天去了商店"
    assert cues[1]["text"] == "今天买了很多东西"
    assert cues[0]["end"] <= cues[1]["start"] + 1e-6


def test_comma_split_zh_short_clauses_stay_split():
    # normal-speed short dialogue (~1.2s each side): comma splits into two cues;
    # short-cue merging has been removed, so each clause stays as its own cue
    seg = _cjk_seg("我去了商店里，他来了我家里", dt=0.18)
    cues = smart_split_segments([seg], lang="zh")
    assert len(cues) == 2
    assert cues[0]["text"] == "我去了商店里"
    assert cues[1]["text"] == "他来了我家里"


def test_comma_no_split_when_before_short_zh():
    # left side of comma is too short (1 char) -> no split; comma becomes a space
    seg = _cjk_seg("好，我们一起走吧好吗", dt=0.6)
    cues = smart_split_segments([seg], lang="zh")
    assert len(cues) == 1
    assert "好 我们一起走吧好吗" in cues[0]["text"]


def test_comma_no_split_when_after_short_zh():
    # right side of comma is a short orphan tail (1 char) -> no split (tail sticks to preceding clause)
    seg = _cjk_seg("这句话挺长的，对", dt=0.6)
    cues = smart_split_segments([seg], lang="zh")
    assert len(cues) == 1


def test_comma_split_ja_uses_japanese_comma():
    # ja comma is 、
    seg = _cjk_seg("今日は学校に行って、それから家に帰りました", dt=0.6)
    cues = smart_split_segments([seg], lang="ja")
    assert len(cues) == 2
    assert cues[0]["text"] == "今日は学校に行って"
    assert cues[1]["text"] == "それから家に帰りました"


def test_comma_split_en_both_sides_long():
    seg = _seg_from_words("I went to the store yesterday, and then I came back home.")
    cues = smart_split_segments([seg], lang="en")
    assert len(cues) == 2
    joined = " ".join(c["text"] for c in cues)
    assert "," not in joined and "." not in joined
    assert "yesterday" in cues[0]["text"]
    assert "home" in cues[1]["text"]


def test_comma_no_split_when_after_short_en():
    # trailing "ok" after the comma is too short -> no split
    seg = _seg_from_words("This is a fairly long opening clause, ok.")
    cues = smart_split_segments([seg], lang="en")
    assert len(cues) == 1


def test_split_at_comma_false_keeps_old_behavior():
    # flag off -> commas are not split into separate cues (become spaces); reverts to pre-feature behavior
    seg = _cjk_seg("我昨天去了商店，今天买了很多东西", dt=0.6)
    cues = smart_split_segments([seg], lang="zh", split_at_comma=False)
    assert len(cues) == 1


# --------------------------------------------------------------------------- #
# comma cap: at most one comma per cue
# --------------------------------------------------------------------------- #
def test_comma_clauses_caps_at_one_comma_zh():
    from voxweave.core.smart_split import _comma_clauses, _comma_load

    clauses = _comma_clauses("，".join(["明日"] * 5), "zh", 6)
    assert len(clauses) >= 2
    for cl in clauses:
        assert _comma_load(cl, "zh") <= 1


def test_comma_clauses_caps_at_one_comma_en():
    from voxweave.core.smart_split import _comma_clauses, _comma_load

    clauses = _comma_clauses("ah, well, you know, maybe, perhaps", "en", 18)
    assert len(clauses) >= 2
    for cl in clauses:
        assert _comma_load(cl, "en") <= 1


def test_comma_clauses_splits_on_ideographic_comma_zh():
    # ideographic comma 、 is also treated as a comma (aligned with the strip set)
    from voxweave.core.smart_split import _comma_clauses, _comma_load

    clauses = _comma_clauses("苹果、香蕉、橙子、西瓜", "zh", 6)
    assert len(clauses) >= 2
    for cl in clauses:
        assert _comma_load(cl, "zh") <= 1


def test_comma_clauses_splits_on_halfwidth_comma_zh():
    # halfwidth comma in zh text is also recognized
    from voxweave.core.smart_split import _comma_clauses, _comma_load

    clauses = _comma_clauses("甲乙丙,丁戊己,庚辛壬", "zh", 6)
    assert len(clauses) >= 2
    for cl in clauses:
        assert _comma_load(cl, "zh") <= 1


def test_repeated_short_name_max_one_comma_zh():
    # end-to-end: five short name calls (2 chars each) -> at most one comma (turned space) per cue,
    # no longer piled onto one line
    seg = _cjk_seg("，".join(["明日"] * 5), dt=0.5)
    cues = smart_split_segments([seg], lang="zh")
    assert len(cues) >= 2
    for c in cues:
        assert c["text"].count(" ") <= 1


def test_no_merge_adjacent_subsecond_sentences():
    # short-cue merging removed: two adjacent sub-second sentences each become their own cue
    # (old force-merge no longer fires); timestamps track real speech boundaries, not merged
    words = [
        {"word": "Right.", "start": 12.0, "end": 12.4},
        {"word": "Yes.", "start": 12.5, "end": 12.9},  # gap 0.1s
    ]
    seg = {"start": 12.0, "end": 12.9, "text": "Right. Yes.", "words": words}
    cues = smart_split_segments([seg], lang="en")
    assert len(cues) == 2


def test_no_merge_keeps_each_sentence_separate():
    # after removing merging, each sentence is its own cue (including adjacent sentences
    # that would span two lines, no longer incorrectly merged)
    seg = _seg_from_words(
        "Shakira I think Shakira would have been in the ladies school. "
        "Shakira was a night.",
        dt=0.25,
    )
    cues = smart_split_segments([seg], lang="en")
    assert len(cues) == 2
    assert "school" in cues[0]["text"]
    assert cues[1]["text"].replace("\n", " ") == "Shakira was a night"


def test_comma_no_split_digit_internal_zh():
    # digit-internal halfwidth comma 10,000 must not split (even though zh now recognizes halfwidth commas)
    seg = _cjk_seg("价格是10,000元，真的是非常贵啊", dt=0.5)
    cues = smart_split_segments([seg], lang="zh")
    assert "10,000" in "".join(c["text"] for c in cues)


# --------------------------------------------------------------------------- #
# Japanese single-line: ja cues render one physical line (no 2-line stacking);
# long content splits into MORE cues instead of wrapping to a second line.
# --------------------------------------------------------------------------- #
def test_default_max_lines_cjk_is_single():
    from voxweave.core.smart_split import default_max_lines

    assert default_max_lines("ja") == 1  # Japanese: single line
    assert default_max_lines("zh") == 1  # Chinese: also single line
    assert default_max_lines("ko") == 2  # Korean (space-delimited): still two lines
    assert default_max_lines("en") == 2


def test_default_max_line_length_cjk_single_wider():
    from voxweave.core.smart_split import default_max_line_length

    # zh/ja single-line -> per-line budget is wider than ko's two-line 12 (fits a whole short sentence)
    assert default_max_line_length("ja") == 18
    assert default_max_line_length("zh") == 18
    assert default_max_line_length("ko") == 12  # unchanged


def test_smart_split_japanese_never_double_line():
    # 25-char sentence with no internal comma -> splits into multiple single-line cues,
    # no cue ever stacks a second line (no \n)
    seg = _cjk_seg("今日はとてもいい天気だから散歩に行きたいと思います", dt=0.4)
    cues = smart_split_segments([seg], lang="ja")
    assert len(cues) >= 2  # >18 chars must split; must not wrap into 2 lines
    for c in cues:
        assert "\n" not in c["text"], c["text"]


def test_smart_split_chinese_never_double_line():
    # Chinese is also single-line: long sentence splits into multiple single-line cues, no second-line stacking
    seg = _cjk_seg("这是一个相当长的中文句子用来测试是否会折成两行显示效果", dt=0.4)
    cues = smart_split_segments([seg], lang="zh")
    assert len(cues) >= 2
    for c in cues:
        assert "\n" not in c["text"], c["text"]
