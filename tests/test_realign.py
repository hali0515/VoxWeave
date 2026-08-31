import pytest

from voxweave import realign


# --------------------------------------------------------------------------- #
# parse_vtt_blocks
# --------------------------------------------------------------------------- #
def test_parse_text_only_blocks():
    blocks = realign.parse_vtt_blocks("WEBVTT\n\nhello\n\nworld\n")
    assert [b["text"] for b in blocks] == ["hello", "world"]
    assert all(b["start"] is None and b["end"] is None for b in blocks)


def test_parse_timestamped_blocks():
    vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.500\nhello\n\n"
        "00:00:03.000 --> 00:00:04.000\nworld\n"
    )
    blocks = realign.parse_vtt_blocks(vtt)
    assert blocks[0]["text"] == "hello"
    assert blocks[0]["start"] == 1.0 and blocks[0]["end"] == 2.5
    assert blocks[1]["start"] == 3.0


def test_parse_skips_note_and_keeps_multiline_cue():
    vtt = "WEBVTT\n\nNOTE 这是注释\n\nline1\nline2\n"
    blocks = realign.parse_vtt_blocks(vtt)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "line1\nline2"


def test_parse_drops_cue_id_line():
    vtt = "WEBVTT\n\n7\n00:00:01.000 --> 00:00:02.000\nhi\n"
    blocks = realign.parse_vtt_blocks(vtt)
    assert blocks[0]["text"] == "hi"
    assert blocks[0]["start"] == 1.0


def test_parse_handles_mm_ss_timestamp():
    blocks = realign.parse_vtt_blocks("WEBVTT\n\n01:02.500 --> 01:03.000\nx\n")
    assert blocks[0]["start"] == 62.5


# --------------------------------------------------------------------------- #
# char_owner_map / spans
# --------------------------------------------------------------------------- #
def test_char_owner_map_identical():
    sets = realign.char_owner_map(["你好", "世界"], ["你", "好", "世", "界"])
    assert sets == [{0, 1}, {2, 3}]


def test_char_owner_map_typo_keeps_anchor():
    # old ASR heard 好 as 耗; user corrected back to 好 → still anchors both units
    sets = realign.char_owner_map(["你好"], ["你", "耗"])
    assert sets == [{0, 1}]


def test_char_owner_map_line_merge():
    # editor merged two lines into one: single block covers all units
    sets = realign.char_owner_map(["你好世界"], ["你", "好", "世", "界"])
    assert sets == [{0, 1, 2, 3}]


def test_char_owner_map_pure_insert_empty():
    # inserted block between matching blocks: surrounding blocks anchor to equal regions,
    # middle block has no match → empty set
    sets = realign.char_owner_map(
        ["你好", "凭空乱入", "世界"], ["你", "好", "世", "界"]
    )
    assert sets[0] == {0, 1}
    assert sets[1] == set()  # pure insertion, no anchor
    assert sets[2] == {2, 3}


def test_spans_from_sets():
    units = [
        {"text": "a", "start": 0.0, "end": 0.5},
        {"text": "b", "start": 0.5, "end": 1.0},
        {"text": "c", "start": 2.0, "end": 2.5},
    ]
    spans = realign.spans_from_sets([{0, 1}, set(), {2}], units)
    assert spans == [(0.0, 1.0), None, (2.0, 2.5)]


# --------------------------------------------------------------------------- #
# route_blocks
# --------------------------------------------------------------------------- #
def _ws():
    return [
        {"text": "你", "start": 0.0, "end": 0.5},
        {"text": "好", "start": 0.5, "end": 1.0},
        {"text": "世", "start": 2.0, "end": 2.5},
        {"text": "界", "start": 2.5, "end": 3.0},
    ]


def test_route_uses_block_timestamps_when_present():
    blocks = [
        {"text": "x", "start": 5.0, "end": 6.0},
        {"text": "y", "start": 7.0, "end": 8.0},
    ]
    assert realign.route_blocks(blocks, _ws()) == [(5.0, 6.0), (7.0, 8.0)]


def test_route_falls_back_to_difflib():
    blocks = [
        {"text": "你好", "start": None, "end": None},
        {"text": "世界", "start": None, "end": None},
    ]
    assert realign.route_blocks(blocks, _ws()) == [(0.0, 1.0), (2.0, 3.0)]


# --------------------------------------------------------------------------- #
# crop_blocks (per-sentence cropping, WhisperX equivalent)
# --------------------------------------------------------------------------- #
def test_crop_tight_window():
    # WhisperX equivalent: each sentence cropped to [start-pad, end+pad], not extended to next
    # sentence → CTC clamped to the cue's own acoustic range, no room for the last word to drift
    # into inter-sentence silence (the old approach, which extended to next sentence start,
    # required snap/carve to fix).
    crops = realign.crop_blocks([(1.0, 2.0), (3.0, 4.0)], pad=0.3)
    assert crops[0] == (0.7, 2.3)
    assert crops[1] == (2.7, 4.3)


def test_crop_window_independent_of_neighbor_distance():
    # tight crop only looks at own span: neighbor distance (song hole / back-to-back) does not
    # change the window → naturally avoids crossing holes and consuming the next sentence
    crops = realign.crop_blocks([(0.0, 1.0), (50.0, 51.0)], pad=0.3)
    assert crops[0] == (0.0, 1.3)
    assert crops[1] == (49.7, 51.3)


def test_crop_skips_none_and_clamps_start():
    crops = realign.crop_blocks([(0.0, 1.0), None, (2.0, 3.0)], pad=0.3)
    assert crops[1] is None  # insertion block not cropped (neighbor interpolation)
    assert crops[0] == (0.0, 1.3)  # start-pad floored at 0; right bound = own end+pad
    assert crops[2] == (1.7, 3.3)


def test_crop_degenerate_reversed_span():
    # rough span reversed/zero-width → degenerate guard: right bound at least start + 0.1s
    crops = realign.crop_blocks([(5.0, 4.0)], pad=0.3)
    cs, ce = crops[0]
    assert ce > cs


# --------------------------------------------------------------------------- #
# join_block_texts
# --------------------------------------------------------------------------- #
def test_join_no_space_lang():
    assert realign.join_block_texts(["你好", "世界"], "zh") == "你好世界"


def test_join_spaced_lang_and_fold_wrap():
    assert realign.join_block_texts(["a\nb", "c"], "en") == "a b c"


# --------------------------------------------------------------------------- #
# fill_insert_blocks (neighbor interpolation, no hole crossing)
# --------------------------------------------------------------------------- #
def test_fill_interpolates_between_neighbors():
    out = realign.fill_insert_blocks([(0.0, 1.0), None, (3.0, 4.0)], gap_sec=2.0)
    assert out[1] == (2.0, 3.0)


def test_fill_does_not_cross_gap():
    out = realign.fill_insert_blocks(
        [(0.0, 1.0), None, (50.0, 51.0)], gap_sec=2.0, default_dur=2.0
    )
    # 50s hole: no crossing; anchors to prev side with default_dur
    assert out[1] == (1.0, 3.0)


def test_fill_prev_only():
    out = realign.fill_insert_blocks([(0.0, 1.0), None], default_dur=2.0)
    assert out[1] == (1.0, 3.0)


def test_fill_next_only():
    out = realign.fill_insert_blocks([None, (10.0, 11.0)], default_dur=2.0)
    assert out[0] == (8.0, 10.0)


# --------------------------------------------------------------------------- #
# clamp / fmt / render
# --------------------------------------------------------------------------- #
def test_clamp_enforces_min_dur():
    assert realign.clamp_spans([(1.0, 1.0)], min_dur=0.05) == [(1.0, 1.05)]


def test_fmt_ts():
    assert realign.fmt_ts(3661.5) == "01:01:01.500"
    assert realign.fmt_ts(0) == "00:00:00.000"


def test_fmt_ts_carries_rounding_across_field_boundaries():
    # Sub-millisecond values below a minute/hour boundary must carry instead of
    # formatting an invalid 60-second field.
    assert realign.fmt_ts(59.9996) == "00:01:00.000"
    assert realign.fmt_ts(3599.9999) == "01:00:00.000"
    # Control: a value that rounds down keeps its field.
    assert realign.fmt_ts(59.9994) == "00:00:59.999"


def test_render_vtt():
    out = realign.render_vtt([{"text": "hi"}], [(1.0, 2.0)])
    assert out == "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n"


# --------------------------------------------------------------------------- #
# fuse_punct_into_text (dual-ASR fusion: whisper raw text + Qwen punctuation by content alignment)
# --------------------------------------------------------------------------- #
def test_fuse_punct_inserts_into_gap_ja():
    # qwen per-char punctuated output (reinject ja): 。 immediately follows です → content
    # alignment inserts it after whisper's です
    text = "畑です次"
    qwen = [
        {"text": "畑", "start": 1.0, "end": 1.2},
        {"text": "で", "start": 1.2, "end": 1.4},
        {"text": "す", "start": 1.4, "end": 1.6},
        {"text": "。", "start": 1.6, "end": 1.6},
        {"text": "次", "start": 2.5, "end": 2.7},
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "畑です。次"


def test_fuse_punct_trailing_attaches_to_end():
    text = "畑次"
    qwen = [
        {"text": "畑", "start": 1.0, "end": 1.2},
        {"text": "次", "start": 2.5, "end": 2.7},
        {"text": "。", "start": 2.7, "end": 2.7},
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "畑次。"


def test_fuse_punct_spaced_word_units():
    # en: punctuation as standalone unit (with content anchors on both sides) → content
    # alignment inserts it after whisper's hello
    text = "hello world"
    qwen = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": ".", "start": 1.1, "end": 1.1},
        {"text": "world", "start": 2.0, "end": 3.0},
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "hello. world"


def test_fuse_punct_content_anchored_not_time_ja():
    # Key regression for content alignment (not time-based): Qwen's 。 immediately follows
    # どの, but どの collides with whisper's 造 on the time axis (OOV drift). Time-based
    # approach inserts 。 after 造 (番酒造。り); content approach inserts after whisper's どの.
    text = "どの番酒造りが"  # whisper transcript (accurate)
    qwen = [  # Qwen per-char: どの。番酒造りが (。immediately follows どの)
        {"text": "ど", "start": 9.0, "end": 9.1},
        {"text": "の", "start": 9.1, "end": 9.2},
        {"text": "。", "start": 9.2, "end": 9.2},
        {"text": "番", "start": 9.3, "end": 9.4},
        {"text": "酒", "start": 9.4, "end": 9.5},
        {"text": "造", "start": 9.5, "end": 9.6},
        {"text": "り", "start": 9.6, "end": 9.7},
        {"text": "が", "start": 9.7, "end": 9.8},
    ]
    out = realign.fuse_punct_into_text(text, qwen, strip_existing=False)
    assert (
        out == "どの。番酒造りが"
    )  # 。 lands after whisper's どの, not inside 番酒造り
    assert "造。り" not in out


def test_fuse_punct_replace_region_falls_back_to_agreed_boundary():
    # Real-world case: Qwen mishears 番酒造り as サチ作り (replace region), and the Qwen
    # punctuation falls inside the misheard region. Only equal-block chars are trusted as anchors
    # → punctuation falls back to the nearest equal char (の from どの), not inside whisper's
    # 番酒造り.
    text = "どの番酒造りが"  # whisper (accurate)
    qwen = [  # Qwen: どのサ。チ作りが (。inside the misheard region サチ作り)
        {"text": "ど", "start": 0.0, "end": 0.1},
        {"text": "の", "start": 0.1, "end": 0.2},
        {"text": "サ", "start": 0.2, "end": 0.3},
        {"text": "。", "start": 0.3, "end": 0.3},
        {"text": "チ", "start": 0.3, "end": 0.4},
        {"text": "作", "start": 0.4, "end": 0.5},
        {"text": "り", "start": 0.5, "end": 0.6},
        {"text": "が", "start": 0.6, "end": 0.7},
    ]
    out = realign.fuse_punct_into_text(text, qwen, strip_existing=False)
    # 。 falls back to after どの (equal-region boundary); whisper 番酒造り is not split
    assert out == "どの。番酒造りが"
    assert "番。" not in out and "造。り" not in out


def test_fuse_punct_drops_punct_in_allreplace_song():
    # OP lyrics: entire segment is a replace region (no content agreement), no equal prefix →
    # all Qwen punctuation is dropped; no per-char cue explosion.
    text = "のんびり農業して"  # whisper lyrics
    qwen = [  # Qwen mishears the entire segment + every char has punctuation
        #       (would cause per-char cue explosion if inserted unconditionally)
        {"text": ch, "start": i * 0.1, "end": i * 0.1 + 0.05}
        for i, ch in enumerate("ホゲ。フガ。ピヨ。")
    ]
    out = realign.fuse_punct_into_text(text, qwen, strip_existing=False)
    assert (
        out == "のんびり農業して"
    )  # no equal anchors → all punctuation dropped, whisper text unchanged


def test_fuse_punct_glued_word_units_en():
    # Key regression: reinject_punct for spaced languages produces **glued punctuation**
    # ("hello." not standalone "."). Old fuse only accepted standalone punct units → 100%
    # miss for English punctuation. Must be able to extract trailing punct from word units.
    text = "hello world"  # whisper transcript (no punctuation)
    qwen = [  # real reinject_punct format: punctuation glued to word
        {"text": "hello.", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 2.0, "end": 3.0},
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "hello. world"


def test_fuse_punct_glued_sentence_period_en():
    # Real case (BGM clip): Qwen "BGM. It's not" period must land after whisper's BGM so
    # that smart_split breaks at the sentence end (old fuse dropped the period → "BGM it's not"
    # ended up as a single cue).
    text = "BGM it's not"
    qwen = [
        {"text": "BGM.", "start": 0.0, "end": 0.5},
        {"text": "It's", "start": 1.0, "end": 1.2},
        {"text": "not", "start": 1.2, "end": 1.4},
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "BGM. it's not"


def test_fuse_punct_glued_multiple_clauses_en():
    # Multiple delimiters: comma and period must both be preserved, each after its corresponding word.
    text = "okay it's loud I edit it"
    qwen = [
        {"text": "Okay,", "start": 0.0, "end": 0.4},
        {"text": "it's", "start": 0.5, "end": 0.7},
        {"text": "loud.", "start": 0.7, "end": 1.0},
        {"text": "I", "start": 1.5, "end": 1.6},
        {"text": "edit", "start": 1.6, "end": 1.9},
        {"text": "it", "start": 1.9, "end": 2.1},
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "okay, it's loud. I edit it"


def test_fuse_qwen_punct_authoritative_strips_whisper_own():
    # Qwen punctuation is authoritative: whisper's own delimiters are stripped, replaced by
    # Qwen's (fixes "double punctuation" over-splitting in fusion — whisper had 43 + Qwen 90
    # = 120 total commas, excess commas caused comma-split to shred the final sentences).
    text = "hello, world"  # whisper has its own comma
    qwen = [
        {"text": "hello", "start": 0.0, "end": 1.0},  # Qwen has no punctuation here
        {"text": "world.", "start": 2.0, "end": 3.0},  # Qwen period after world
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "hello world."


def test_fuse_keeps_nonpunct_symbols_when_stripping():
    # Only _FUSE_PUNCT is stripped; ・ / % / hyphens and other non-delimiter symbols are preserved.
    text = "100% rock-solid"
    qwen = [
        {"text": "100", "start": 0.0, "end": 0.5},
        {"text": "rock-solid.", "start": 0.6, "end": 1.2},
    ]
    assert realign.fuse_punct_into_text(text, qwen) == "100% rock-solid."


def test_fuse_strip_existing_gates_whisper_punct():
    # Spaced languages strip_existing=True (strip whisper's own delimiters, use Qwen's);
    # no-space languages strip_existing=False (keep whisper's own punctuation — char-level
    # time mapping would drift and split words if stripped).
    text = "畑です。次"  # whisper has its own period
    # CJK path: retain whisper period
    assert realign.fuse_punct_into_text(text, [], strip_existing=False) == "畑です。次"
    # spaced-language path (default): strip whisper period (no Qwen punct to replace it → bare text)
    assert realign.fuse_punct_into_text(text, [], strip_existing=True) == "畑です次"


def test_fuse_punct_preserves_nakaguro_in_raw_text():
    # Key: text is preserved char-by-char; name separator ・ must not be dropped (fixes
    # "rebuilding from units drops non-alnum"); ・ is not a sentence boundary so no punct inserted.
    text = "ラスティス・ムーン"  # ・ is not alnum; not present in units
    assert realign.fuse_punct_into_text(text, []) == "ラスティス・ムーン"


def test_fuse_punct_no_punct_returns_text_verbatim():
    text = "私は"
    assert realign.fuse_punct_into_text(text, []) == "私は"


# --------------------------------------------------------------------------- #
# explode_units (fix for cross-cue overlap bug)
# --------------------------------------------------------------------------- #
def test_explode_units_splits_time_evenly():
    out = realign.explode_units([{"text": "好柯", "start": 5.96, "end": 6.12}])
    assert [u["text"] for u in out] == ["好", "柯"]
    assert out[0]["start"] == 5.96 and abs(out[0]["end"] - 6.04) < 1e-9
    assert abs(out[1]["start"] - 6.04) < 1e-9 and out[1]["end"] == 6.12


def test_explode_units_skips_punct_and_empty():
    # punctuation is not alnum; no pseudo-unit is produced; pure-punct units are dropped entirely
    out = realign.explode_units(
        [
            {"text": "a,", "start": 0.0, "end": 1.0},
            {"text": "。", "start": 1.0, "end": 2.0},
        ]
    )
    assert [u["text"] for u in out] == ["a"]


def test_explode_units_fixes_cross_cue_overlap():
    # Aligner groups last char of "听好" and first char of "柯蕾" into a single unit '好柯'
    # → using unit index sets directly causes overlap
    wtexts = ["听好", "柯蕾"]
    units = [
        {"text": "听", "start": 5.48, "end": 5.80},
        {"text": "好柯", "start": 5.96, "end": 6.12},
        {"text": "蕾", "start": 7.00, "end": 7.40},
    ]
    # old path (unit-level): both cues claim '好柯' → overlap
    old = realign.spans_from_sets(
        realign.char_owner_map(wtexts, [u["text"] for u in units]), units
    )
    assert (
        old[1][0] < old[0][1]
    )  # prove old path actually overlaps (cue1.start < cue0.end)
    # new path (single-char pseudo-units): cues abut, no overlap
    punits = realign.explode_units(units)
    new = realign.spans_from_sets(
        realign.char_owner_map(wtexts, [p["text"] for p in punits]), punits
    )
    assert new[1][0] >= new[0][1]  # cue1.start >= cue0.end


# --------------------------------------------------------------------------- #
# reinject_punct (re-attach punctuation to aligned units)
# --------------------------------------------------------------------------- #
def test_reinject_punct_nospace_japanese():
    units = [
        {"text": "あい", "start": 0.0, "end": 1.0},
        {"text": "う", "start": 1.0, "end": 1.5},
    ]
    out = realign.reinject_punct("あ。い、う", units, "ja")
    # one item per non-whitespace character (punctuation also gets an item)
    assert [u["text"] for u in out] == ["あ", "。", "い", "、", "う"]
    # "".join restores asr_text → satisfies smart_split char-level count contract
    assert "".join(u["text"] for u in out) == "あ。い、う"
    # alnum chars get real timestamps; punctuation gets thin zero-width slice; monotone
    assert out[0]["start"] == 0.0
    starts = [u["start"] for u in out]
    assert starts == sorted(starts)


def test_reinject_punct_nospace_keeps_latin_word_spaces():
    # Embedded English in no-space language (zh/ja) text: spaces between Latin words must be
    # preserved, otherwise process joining with "" would collapse "It is" into "Itis".
    # Spaces are appended to the preceding non-whitespace unit's text, not as new units
    # (maintains "one unit = one non-whitespace character" char-level count contract).
    units = [
        {"text": "看", "start": 0.0, "end": 0.3},
        {"text": "It", "start": 0.5, "end": 0.8},
        {"text": "is", "start": 0.9, "end": 1.1},
        {"text": "哦", "start": 1.3, "end": 1.6},
    ]
    out = realign.reinject_punct("看 It is 哦", units, "zh")
    # process joins with "" preserving spaces (smart_split._tokens treats "It is" as a Latin run)
    assert "".join(u["text"] for u in out) == "看 It is 哦"
    # one unit = one non-whitespace character (看 I t i s 哦 = 6); spaces don't inflate word_data
    assert len(out) == 6
    starts = [u["start"] for u in out]
    assert starts == sorted(starts)


def test_reinject_punct_space_english_keeps_words_and_punct():
    units = [
        {"text": "Listen", "start": 0.0, "end": 0.5},
        {"text": "up", "start": 0.5, "end": 0.8},
        {"text": "Clay", "start": 1.0, "end": 1.4},
    ]
    out = realign.reinject_punct("Listen up, Clay.", units, "en")
    # one item per whitespace-delimited token; punctuation glued to the token
    assert [u["text"] for u in out] == ["Listen", "up,", "Clay."]
    # " ".join restores asr_text → satisfies smart_split token-level count contract
    assert " ".join(u["text"] for u in out) == "Listen up, Clay."


def test_reinject_punct_splits_em_dash():
    # 1.7B emits em-dash: "today—a" swallows the pause into one word_segment → should split
    # into "today—" + "a" with real per-char timestamps (today ends ~2.0, a starts ~5.0 →
    # 3s gap exposed for gap-based segmentation)
    units = [
        {"text": "today", "start": 1.0, "end": 2.0},
        {"text": "a", "start": 5.0, "end": 5.3},
    ]
    out = realign.reinject_punct("today—a", units, "en")
    assert [u["text"] for u in out] == ["today—", "a"]
    assert out[0]["end"] == pytest.approx(2.0)  # left sub-token ends at today's end
    assert out[1]["start"] == pytest.approx(
        5.0
    )  # right sub-token starts at a → 3s gap visible


def test_reinject_punct_keeps_leading_and_trailing_dash():
    # leading (dialogue "—a") / trailing ("today—") dashes are not split; single hyphen in
    # compound words is not split either
    units = [
        {"text": "well", "start": 0.0, "end": 0.4},
        {"text": "known", "start": 0.4, "end": 0.8},
        {"text": "a", "start": 1.0, "end": 1.2},
    ]
    out = realign.reinject_punct("well-known —a", units, "en")
    assert [u["text"] for u in out] == [
        "well-known",
        "—a",
    ]  # single '-' not split; leading '—' not split


def test_reinject_punct_splits_double_hyphen_and_ellipsis():
    units = [
        {"text": "wait", "start": 0.0, "end": 0.5},
        {"text": "what", "start": 2.0, "end": 2.4},
    ]
    out = realign.reinject_punct("wait--what", units, "en")
    assert [u["text"] for u in out] == ["wait--", "what"]
    out2 = realign.reinject_punct("wait…what", units, "en")
    assert [u["text"] for u in out2] == ["wait…", "what"]


def test_reinject_punct_fallback_no_alnum():
    units = [{"text": "x", "start": 0.0, "end": 1.0}]
    assert realign.reinject_punct("。、！", units, "ja") == units
    assert realign.reinject_punct("", units, "en") == units


def test_reinject_then_smart_split_unglues_and_strips_punct():
    # end-to-end: reinject punctuation → smart_split → output has spaces, no punctuation, split by sentence
    smart_split = pytest.importorskip("voxweave.core.smart_split")
    units = [
        {"text": "Listen", "start": 0.0, "end": 0.5},
        {"text": "up", "start": 0.6, "end": 0.9},
        {"text": "Clay", "start": 1.0, "end": 1.5},
        {"text": "stay", "start": 2.0, "end": 2.4},
        {"text": "sharp", "start": 2.5, "end": 3.0},
    ]
    pu = realign.reinject_punct("Listen up, Clay. Stay sharp.", units, "en")
    seg = {
        "start": pu[0]["start"],
        "end": pu[-1]["end"],
        "text": " ".join(u["text"] for u in pu),
        "words": [
            {"word": u["text"], "start": u["start"], "end": u["end"]} for u in pu
        ],
    }
    cues = smart_split.smart_split_segments([seg], lang="en")
    joined = " ".join(c["text"] for c in cues)
    assert "Listenup" not in joined  # English words not fused together
    assert "Listen up" in joined  # inter-word space present
    assert (
        "," not in joined and "." not in joined
    )  # punctuation converted to space / dropped


# --------------------------------------------------------------------------- #
# enforce_min_duration (fix for flash-by short cues)
# --------------------------------------------------------------------------- #
def test_min_dur_extends_into_gap():
    # はい 80ms, large gap after → extend to min_dur, start unchanged (second sentence is long, unaffected)
    out = realign.enforce_min_duration([(11.96, 12.04), (14.44, 16.0)], min_dur=0.8)
    assert out[0][0] == 11.96 and out[0][1] == pytest.approx(
        12.76
    )  # start unchanged, extended 0.8s
    assert out[1] == (14.44, 16.0)


def test_min_dur_pushes_glued_next():
    # クレイ 150ms immediately adjacent to ダンジョン → push ダンジョン start to borrow time, no overlap
    out = realign.enforce_min_duration(
        [(7.00, 7.15), (7.15, 8.44), (10.0, 11.0)], min_dur=0.8
    )
    assert out[0] == (7.00, 7.80)  # クレイ extended to 0.8s
    assert out[1][0] == 7.80  # ダンジョン start pushed forward
    assert out[1][0] >= out[0][1] - 1e-9  # no longer overlapping


def test_min_dur_leaves_long_cues():
    out = realign.enforce_min_duration([(0.0, 2.0), (3.0, 5.0)], min_dur=0.8)
    assert out == [(0.0, 2.0), (3.0, 5.0)]


def test_min_dur_removes_existing_overlap():
    # input already overlapping (previous cue end past next cue start) → truncate prev, eliminate overlap
    out = realign.enforce_min_duration([(0.0, 3.0), (1.0, 4.0)], min_dur=0.8)
    assert out[0][1] <= out[1][0] + 1e-9  # no longer overlapping
    assert out[1] == (1.0, 4.0)


def test_min_dur_capped_when_unpushable():
    # next sentence too short to be pushed → cap at next sentence start (no overlap, no forced extension)
    out = realign.enforce_min_duration([(0.0, 0.1), (0.2, 0.3)], min_dur=0.8)
    assert out[0][1] <= out[1][0] + 1e-9


def test_min_dur_extends_last_cue():
    out = realign.enforce_min_duration([(0.0, 0.1)], min_dur=0.8)
    assert out[0] == (0.0, 0.8)


# --------------------------------------------------------------------------- #
# rescue_tiny_cues (flash-display extension, allows side-by-side with at most 1 neighbor)
# --------------------------------------------------------------------------- #
def test_tiny_cue_extends_into_gap():
    # so 100ms (<0.2 trig), large gap after → extend to target, no overlap (start unchanged)
    out = realign.rescue_tiny_cues([(10.0, 10.1), (12.0, 13.0)], trig=0.2, target=0.5)
    assert out[0][0] == 10.0 and out[0][1] == pytest.approx(10.5)
    assert out[1] == (12.0, 13.0)


def test_tiny_cue_overlaps_one_neighbor():
    # so 100ms immediately adjacent to next cue (no gap) → extends to target, shown side-by-side
    # with next cue, but disappears before the cue after that
    out = realign.rescue_tiny_cues(
        [(10.0, 10.1), (10.3, 11.8), (11.9, 12.5)], trig=0.2, target=0.5
    )
    assert out[0][0] == 10.0 and out[0][1] == pytest.approx(10.5)
    assert out[0][1] > out[1][0]  # side-by-side with next cue (10.5 > 10.3)
    assert out[0][1] < 11.9  # disappears before cue-after-next appears
    assert out[1] == (10.3, 11.8) and out[2] == (11.9, 12.5)  # long cues unchanged


def test_tiny_cue_overlap_capped_by_next_next():
    # consecutive flash cues (so そう) densely packed → each extended to target but capped at
    # cue-after-next start to prevent 3+ simultaneous lines
    out = realign.rescue_tiny_cues(
        [(10.0, 10.1), (10.2, 10.3), (10.4, 11.4)], trig=0.2, target=0.5
    )
    assert out[0][0] == 10.0 and out[0][1] == pytest.approx(
        10.4
    )  # capped at cue2.start
    assert out[1][0] == 10.2 and out[1][1] == pytest.approx(
        10.7
    )  # last flash cue extended to target
    assert out[2] == (10.4, 11.4)  # long cue unchanged


def test_tiny_cue_leaves_normal_cues():
    # dur >= trig: never touched
    out = realign.rescue_tiny_cues([(0.0, 0.5), (1.0, 1.5)], trig=0.2, target=0.5)
    assert out == [(0.0, 0.5), (1.0, 1.5)]


def test_tiny_cue_disabled_when_trig_zero():
    # trig<=0 → no-op (even 50ms flash cues are not touched)
    out = realign.rescue_tiny_cues([(0.0, 0.05), (0.1, 0.2)], trig=0.0, target=0.5)
    assert out == [(0.0, 0.05), (0.1, 0.2)]


def test_tiny_cue_extends_last_cue():
    out = realign.rescue_tiny_cues([(0.0, 0.1)], trig=0.2, target=0.5)
    assert out[0] == (0.0, pytest.approx(0.5))


# --------------------------------------------------------------------------- #
# snap_zero_duration_units (relocate zero-duration units to actual gap speech; fix aligner collapse)
# --------------------------------------------------------------------------- #
def test_snap_relocates_zero_dur_unit_to_gap_speech():
    # はい collapses to zero-duration pinned at な end (12.06); actual speech is in gap 13.0-13.6 → relocated
    units = [
        {"text": "な", "start": 11.9, "end": 12.06},
        {"text": "はい", "start": 12.06, "end": 12.06},
        {"text": "例えば", "start": 14.46, "end": 15.1},
    ]
    vad = [(11.0, 12.06), (13.0, 13.6), (14.4, 15.2)]
    out = realign.snap_zero_duration_units(units, vad)
    assert out[1]["start"] == 13.0 and out[1]["end"] == 13.6
    assert out[0] == units[0] and out[2] == units[2]  # neighbors unchanged


def test_snap_keeps_subtoken_with_no_orphan_segment():
    # だ is a sub-token of 例えばだ, collapsed to 例えば end (15.1); no isolated speech segment
    # in the gap → leave unchanged
    units = [
        {"text": "例えば", "start": 14.46, "end": 15.1},
        {"text": "だ", "start": 15.1, "end": 15.1},
        {"text": "次", "start": 17.84, "end": 18.0},
    ]
    vad = [
        (14.4, 15.3),
        (17.8, 18.1),
    ]  # two segments belong to 例えば / 次 respectively; no isolated segment
    out = realign.snap_zero_duration_units(units, vad)
    assert out[1] == units[1]  # だ unchanged


def test_snap_spreads_charsplit_punct_run_to_gap_speech():
    # Real form: はい split by reinject into 。/は/い/。 run of zeros pinned at な end (12.12);
    # entire run spread evenly to gap speech 13.0-13.6 → は/い land in actual utterance interval
    # (reproduces Dungeon ep1)
    units = [
        {"text": "な", "start": 11.88, "end": 12.12},
        {"text": "。", "start": 12.12, "end": 12.12},
        {"text": "は", "start": 12.12, "end": 12.12},
        {"text": "い", "start": 12.12, "end": 12.12},
        {"text": "。", "start": 12.12, "end": 12.12},
        {"text": "例", "start": 14.44, "end": 14.63},
    ]
    vad = [(11.0, 12.12), (13.0, 13.6), (14.4, 15.2)]
    out = realign.snap_zero_duration_units(units, vad)
    assert (
        13.0 <= out[2]["start"] < out[3]["start"] <= 13.6
    )  # は < い, both within 13.0-13.6
    assert out[0] == units[0] and out[5] == units[5]  # non-zero neighbors unchanged


def test_snap_leaves_long_repetition_run_alone():
    # repetition-collapse wall (run > max_run): not touched even if gap has a speech segment
    units = [{"text": "だ", "start": 5.0, "end": 5.0} for _ in range(20)]
    out = realign.snap_zero_duration_units(units, [(6.0, 7.0)], max_run=8)
    assert out == units


def test_snap_noop_without_vad():
    units = [{"text": "はい", "start": 1.0, "end": 1.0}]
    assert realign.snap_zero_duration_units(units, []) == units


# --------------------------------------------------------------------------- #
# snap_silence_stranded_units (CTC point timestamps drift into adjacent silence → pull to nearest speech edge; fix positional misalignment)
# --------------------------------------------------------------------------- #
def test_snap_stranded_pulls_run_back_to_speech_offset():
    # 行くよ (CTC 20ms point timestamps) drifted 0.39s past VAD offset 55.70; です is in VAD, unchanged.
    # Expected: 行くよ run pulled back (run end <= 55.70), not past prev word end 55.31, monotone.
    units = [
        {"text": "で", "start": 54.99, "end": 55.01},
        {"text": "す", "start": 55.29, "end": 55.31},
        {"text": "。", "start": 55.31, "end": 55.31},
        {"text": "行", "start": 56.09, "end": 56.11},
        {"text": "く", "start": 56.25, "end": 56.27},
        {"text": "よ", "start": 56.37, "end": 56.39},
        {"text": "。", "start": 56.39, "end": 56.39},
    ]
    out = realign.snap_silence_stranded_units(units, [(51.90, 55.70)], tol=0.5)
    assert (
        out[0] == units[0] and out[1] == units[1] and out[2] == units[2]
    )  # in VAD, unchanged
    assert out[3]["start"] >= 55.31 - 1e-9  # not past prev word end
    assert out[5]["end"] <= 55.70 + 1e-9  # last char pulled back to speech offset
    assert out[3]["start"] < out[4]["start"] < out[5]["start"]  # monotone
    assert out[6]["start"] == out[5]["end"]  # trailing punct attached to content end


def test_snap_stranded_pushes_leading_run_to_speech_onset():
    # のんびり農業 first 6 chars drifted before VAD onset 209.90; し starts inside VAD, bounds run right.
    # Expected: first 6 chars pushed to >= 209.90, not past し start 210.04, monotone.
    units = [
        {"text": "の", "start": 209.06, "end": 209.08},
        {"text": "ん", "start": 209.24, "end": 209.26},
        {"text": "び", "start": 209.34, "end": 209.36},
        {"text": "り", "start": 209.46, "end": 209.48},
        {"text": "農", "start": 209.62, "end": 209.64},
        {"text": "業", "start": 209.84, "end": 209.86},
        {"text": "し", "start": 210.04, "end": 210.06},
        {"text": "て", "start": 210.08, "end": 210.10},
    ]
    out = realign.snap_silence_stranded_units(units, [(209.90, 210.80)], tol=0.5)
    assert out[0]["start"] >= 209.90 - 1e-9  # first char pushed to speech onset
    assert out[5]["end"] <= 210.04 + 1e-9  # not past し start
    assert out[0]["start"] < out[3]["start"] < out[5]["start"]  # monotone
    assert out[6] == units[6] and out[7] == units[7]  # し/て inside VAD, unchanged


def test_snap_stranded_leaves_isolated_run_in_long_silence():
    # dropped audio: run in long silence, nearest speech edge > tol → leave it (user decision: dropped audio is acceptable)
    units = [
        {"text": "メ", "start": 505.2, "end": 505.22},
        {"text": "ー", "start": 505.4, "end": 505.42},
    ]
    out = realign.snap_silence_stranded_units(
        units, [(503.0, 504.0), (507.0, 508.0)], tol=0.5
    )
    assert out == units


def test_snap_stranded_leaves_in_speech_units():
    # units that normally overlap speech (have VAD overlap) → not triggered, unchanged
    units = [
        {"text": "あ", "start": 1.0, "end": 1.3},
        {"text": "い", "start": 1.3, "end": 1.6},
    ]
    assert realign.snap_silence_stranded_units(units, [(0.5, 2.0)], tol=0.5) == units


def test_snap_stranded_noop_without_vad():
    units = [{"text": "x", "start": 1.0, "end": 1.02}]
    assert realign.snap_silence_stranded_units(units, []) == units


# --------------------------------------------------------------------------- #
# carve_units_over_silence (trim leading/trailing silence from units; geometric CTC blank equivalent)
# --------------------------------------------------------------------------- #
def test_carve_trims_trailing_silence_balloon():
    # "Oh" inflated to 2.98→5.54 spanning [4.2,6.0] silence; actual speech only [3.3,4.2] → carved to [3.3,4.2]
    units = [{"text": "Oh", "start": 2.98, "end": 5.54}]
    vad = [(3.3, 4.2), (6.0, 8.9)]
    out = realign.carve_units_over_silence(units, vad)
    assert out[0]["start"] == 3.3 and out[0]["end"] == 4.2
    assert out[0]["text"] == "Oh"


def test_carve_trims_leading_silence_only():
    # start lands in silence (5.0→5.5 no speech), speech 5.5→6.0; tail fits exactly → only leading trim
    units = [{"text": "あ", "start": 5.0, "end": 6.0}]
    out = realign.carve_units_over_silence(units, [(5.5, 6.0)])
    assert out[0]["start"] == 5.5 and out[0]["end"] == 6.0


def test_carve_leaves_continuous_long_vowel():
    # long vowel そう〜 fully voiced (VAD covers entire span) → not trimmed (key safety property
    # distinguishing this from outlier-statistics methods)
    units = [{"text": "そう", "start": 10.0, "end": 11.8}]
    out = realign.carve_units_over_silence(units, [(9.5, 12.0)])
    assert out[0] == units[0]


def test_carve_ignores_small_overhang():
    # leading/trailing silence < min_overhang (0.2) → not trimmed (debounce)
    units = [{"text": "x", "start": 1.0, "end": 2.0}]
    out = realign.carve_units_over_silence(units, [(1.15, 1.85)])
    assert out[0] == units[0]


def test_carve_leaves_unit_fully_in_silence():
    # unit fully in silence (no speech overlap) → left unchanged; handed to snap_zero_duration_units
    units = [{"text": "はい", "start": 4.5, "end": 5.0}]
    out = realign.carve_units_over_silence(units, [(1.0, 2.0), (6.0, 7.0)])
    assert out[0] == units[0]


def test_carve_keeps_internal_silence():
    # speech-silence-speech within a single unit: only outer edges trimmed; internal silence retained (v1 does not cut internally)
    units = [{"text": "word", "start": 1.0, "end": 5.0}]
    out = realign.carve_units_over_silence(units, [(1.5, 2.0), (4.0, 4.5)])
    assert out[0]["start"] == 1.5 and out[0]["end"] == 4.5


def test_carve_leaves_zero_duration_unit():
    # zero-duration unit (snap's domain): not touched
    units = [{"text": "は", "start": 5.0, "end": 5.0}]
    out = realign.carve_units_over_silence(units, [(6.0, 7.0)])
    assert out[0] == units[0]


def test_carve_only_shrinks_never_expands():
    # only shrinks inward → adjacent units don't reverse or overlap; new interval always inside old
    units = [
        {"text": "a", "start": 1.0, "end": 3.0},
        {"text": "b", "start": 3.0, "end": 5.0},
    ]
    out = realign.carve_units_over_silence(units, [(2.5, 3.0), (3.0, 3.5)])
    assert out[0]["end"] <= out[1]["start"]  # no overlap
    for o, u in zip(out, units):
        assert (
            u["start"] <= o["start"] <= o["end"] <= u["end"]
        )  # only shrinks, never expands


def test_carve_noop_without_vad():
    units = [{"text": "x", "start": 1.0, "end": 3.0}]
    assert realign.carve_units_over_silence(units, []) == units


# --------------------------------------------------------------------------- #
# position_units_with_vad (shared snap + carve pipeline; prevents drift across transcribe/align paths)
# --------------------------------------------------------------------------- #
def test_position_units_with_vad_snaps_then_carves():
    # single pass: zero-duration はい relocated to gap speech (snap) + Oh inflated past 14.5 trimmed (carve)
    units = [
        {"text": "な", "start": 11.9, "end": 12.06},
        {
            "text": "はい",
            "start": 12.06,
            "end": 12.06,
        },  # zero-duration → snap to 13.0-13.6
        {
            "text": "Oh",
            "start": 13.8,
            "end": 16.0,
        },  # inflated past 14.5 silence → carve to 14.5
    ]
    vad = [(11.0, 12.06), (13.0, 13.6), (14.0, 14.5)]
    out = realign.position_units_with_vad(units, vad)
    assert out[1]["start"] == 13.0 and out[1]["end"] == 13.6  # snapped
    assert out[2]["start"] == 13.8 and out[2]["end"] == 14.5  # trailing silence carved


def test_position_units_with_vad_noop_without_vad():
    units = [{"text": "は", "start": 1.0, "end": 1.0}]
    assert realign.position_units_with_vad(units, []) == units


# --------------------------------------------------------------------------- #
# group_block_spans (align path: per-block raw units → block span; WhisperX equivalent, no VAD post-processing)
# --------------------------------------------------------------------------- #
def test_group_block_spans_first_to_last():
    # block span = (first word start, last word end), trusting raw CTC timestamps from cropped window
    block_units = [
        [
            {"text": "a", "start": 1.0, "end": 1.4},
            {"text": "b", "start": 1.4, "end": 1.9},
        ],
        [{"text": "c", "start": 3.0, "end": 3.4}],
    ]
    spans, flat = realign.group_block_spans(block_units)
    assert spans[0] == (1.0, 1.9)
    assert spans[1] == (3.0, 3.4)
    assert len(flat) == 3


def test_group_block_spans_preserves_none_for_empty_block():
    # insertion / empty-text block (no units) → None span (fill_insert_blocks handles interpolation)
    block_units = [
        [{"text": "a", "start": 1.0, "end": 1.4}],
        [],
        [{"text": "b", "start": 3.0, "end": 3.4}],
    ]
    spans, _ = realign.group_block_spans(block_units)
    assert spans[1] is None
    assert spans[0] == (1.0, 1.4) and spans[2] == (3.0, 3.4)


def test_group_block_spans_trusts_raw_no_vad_snap():
    # WhisperX equivalent: no relocation/trimming (tight crop leaves no drift space); zero-duration
    # passed through to clamp_spans for floor
    block_units = [[{"text": "ー", "start": 45.0, "end": 45.0}]]
    spans, _ = realign.group_block_spans(block_units)
    assert spans[0] == (45.0, 45.0)
