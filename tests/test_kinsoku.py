# tests/test_kinsoku.py
import pytest

from voxweave.core.kinsoku import apply_kinsoku, line_end_penalty


def test_no_line_starts_with_small_kana():
    # っ in "って" must not start a line -> should be pulled back to the previous line
    lines = ["これは", "っという"]
    out = apply_kinsoku(lines)
    assert not out[1].startswith("っ")


def test_no_line_starts_with_punct():
    lines = ["こんにちは", "。さようなら"]
    out = apply_kinsoku(lines)
    assert not out[1].startswith("。")


def test_no_line_ends_with_open_bracket():
    lines = ["彼は「", "本当」と言った"]
    out = apply_kinsoku(lines)
    assert not out[0].endswith("「")


def test_single_line_unchanged():
    assert apply_kinsoku(["普通の一行"]) == ["普通の一行"]


def test_zh_punct_not_line_start():
    # Chinese leading-edge prohibition (kinsoku also handles zh): 。 must not start a line -> pulled back
    lines = ["他说", "。好的"]
    out = apply_kinsoku(lines)
    assert not out[1].startswith("。")


def test_no_empty_line_when_drained():
    # entire line consists of leading-edge-prohibited chars -> all pulled to previous line; result must have no empty lines
    out = apply_kinsoku(["あ", "っ"])
    assert "" not in out
    assert "".join(out) == "あっ"


@pytest.mark.parametrize("ch", list("のをにへ"))
def test_line_end_penalty_high_case_particles(ch):
    # case/adnominal particle at line end = heavy penalty 2 (splits noun phrase like 大樹の|村)
    assert line_end_penalty("大樹" + ch) == 2


@pytest.mark.parametrize("ch", list("とまでより"))
def test_line_end_penalty_med_binding(ch):
    assert line_end_penalty("彼" + ch) == 1


@pytest.mark.parametrize(
    "text", ["大樹の村", "なった", "晴れ", "GPT-4", "hello", "", "  "]
)
def test_line_end_penalty_clean(text):
    # noun/verb/Latin/empty at line end = 0 (no binding, no suppression). Space-delimited languages are naturally 0 (no regression)
    assert line_end_penalty(text) == 0


def test_line_end_penalty_ignores_trailing_space():
    assert line_end_penalty("大樹の  ") == 2


@pytest.mark.parametrize("tok", ["the", "The", "of", "and", "to", "with", "his", "was"])
def test_line_end_penalty_en_forbidden(tok):
    # en closed-class tokens must not end a line/cue (the | store)
    assert line_end_penalty(tok, "en") == 2


@pytest.mark.parametrize("tok", ["store", "went", "I", "yesterday", "GPT-4"])
def test_line_end_penalty_en_content(tok):
    assert line_end_penalty(tok, "en") == 0


def test_line_end_penalty_en_strips_punct():
    # trailing punctuation on the token must not hide the match
    assert line_end_penalty("the,", "en") == 2


def test_line_end_penalty_en_needs_lang():
    # without lang, en tokens stay 0 (back-compat for legacy callers)
    assert line_end_penalty("the") == 0


@pytest.mark.parametrize("w", ["的", "地", "得", "把", "被", "和", "或", "而"])
def test_line_end_penalty_zh_high(w):
    assert line_end_penalty(w, "zh") == 2


@pytest.mark.parametrize("w", ["在", "对", "很", "给"])
def test_line_end_penalty_zh_med(w):
    assert line_end_penalty(w, "zh") == 1


@pytest.mark.parametrize("w", ["目的", "村庄", "数据中心", "大树"])
def test_line_end_penalty_zh_whole_word(w):
    # whole-word semantics: 目的 ends with 的 but is a noun — no penalty
    assert line_end_penalty(w, "zh") == 0


def test_line_end_penalty_ja_chars_active_without_lang():
    # kana particle check stays active regardless of lang (cannot false-positive elsewhere)
    assert line_end_penalty("大樹の", "ja") == 2


# --------------------------------------------------------------------------- #
# Level 2: UniDic POS penalties (fugashi + unidic-lite)
# --------------------------------------------------------------------------- #


def test_pos_distinguishes_case_no_from_nominalizer_no():
    pytest.importorskip("fugashi")
    from voxweave.core.kinsoku import ja_pos_end_penalties

    pens = ja_pos_end_penalties("大樹の村")
    assert pens is not None and pens[2] == 2  # 連体の: binds forward
    pens = ja_pos_end_penalties("走るのが好きだ")
    # 準体の (offset 2) is a legal break the char table had to penalize
    assert pens is not None and pens[2] == 0


def test_pos_reaches_classes_surface_tables_cannot():
    pytest.importorskip("fugashi")
    from voxweave.core.kinsoku import ja_pos_end_penalties

    pens = ja_pos_end_penalties("この村まで")
    assert pens is not None
    assert pens[1] == 2  # この 連体詞
    assert pens[4] == 1  # まで 副助詞
    pens = ja_pos_end_penalties("お名前は")
    assert pens is not None
    assert pens[0] == 2  # お 接頭辞
    assert pens[3] == 0  # は 係助詞


def test_pos_env_kill_switch(monkeypatch):
    pytest.importorskip("fugashi")
    from voxweave.core.kinsoku import _load_ja_tagger, ja_pos_end_penalties

    monkeypatch.setenv("VOXWEAVE_JA_POS", "0")
    _load_ja_tagger.cache_clear()
    try:
        assert ja_pos_end_penalties("大樹の村") is None
    finally:
        _load_ja_tagger.cache_clear()


def test_attach_end_penalties_uses_pos_for_ja():
    pytest.importorskip("fugashi")
    from voxweave.core.smart_split import _attach_end_penalties

    atoms = [{"text": c} for c in "走るのが好きだ"]
    _attach_end_penalties(atoms, {0, 3}, "ja")  # 走るの | が好きだ
    assert atoms[2]["end_pen"] == 0  # POS: 準体の is a legal break
    atoms2 = [{"text": c} for c in "大樹の村"]
    _attach_end_penalties(atoms2, {0, 3}, "ja")
    assert atoms2[2]["end_pen"] == 2  # POS: 格助詞の binds forward
