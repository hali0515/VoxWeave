# tests/test_breakpoints.py
from voxweave.core.breakpoints import legal_break_index, phrase_atoms
from voxweave.core.smart_split import _fit_split_clause


def test_slides_off_article():
    # must not break after "the" (article|noun); target=2 lands after "the" -> slides to a legal point
    toks = ["I", "saw", "the", "red", "car", "today"]
    # target index = break between toks[:i] | toks[i:]; i=3 would leave left side ending with "the" -> illegal
    assert legal_break_index(toks, "en", 3) != 3


def test_keeps_legal_midpoint():
    toks = ["I", "left", "but", "she", "stayed", "home"]
    # i=3: left side "...left but" -- "but" is a conjunction at the end, also an illegal left token -> should avoid
    i = legal_break_index(toks, "en", 3)
    assert toks[i - 1].lower() not in {"the", "a", "an", "but", "and"}


def test_non_en_returns_target_unchanged():
    assert legal_break_index(["a", "b", "c"], "ja", 1) == 1


def test_target_clamped_in_range():
    toks = ["one", "two"]
    i = legal_break_index(toks, "en", 5)
    assert 1 <= i <= len(toks)


def test_fit_split_avoids_article_break():
    # 6 tokens, no terminal/conjunction -> falls through to even-split; old mid=3 left
    # "I gave the" ending with "the" (forbidden); the new guard slides away from it.
    # max_line_length=12 forces overflow beyond 1 line to trigger the fallback.
    clause = "I gave the enormous shiny present"
    parts = _fit_split_clause(clause, max_line_length=12, max_lines=1, lang="en")
    for p in parts[:-1]:
        assert p.split()[-1].strip(".,").lower() not in {"the", "a", "an", "to", "of"}


def test_phrase_atoms_en_is_words():
    assert phrase_atoms("hello world foo", "en") == ["hello", "world", "foo"]


def test_phrase_atoms_ja_grouping_or_fallback():
    out = phrase_atoms("今日は天気です", "ja")
    assert (
        "".join(out) == "今日は天気です"
    )  # byte-preserving (regardless of grouping or per-char fallback)
    assert all(o for o in out)


def test_phrase_atoms_zh_fallback_when_no_segmenter(monkeypatch):
    import voxweave.core.breakpoints as B

    monkeypatch.setattr(B, "_load_jieba", lambda: None)  # simulate jieba absent
    monkeypatch.setattr(B, "_load_parser", lambda lang: None)  # simulate budoux absent
    out = phrase_atoms("今天是晴天", "zh")
    assert out == ["今", "天", "是", "晴", "天"]  # per-char fallback


def test_phrase_atoms_zh_uses_jieba():
    import pytest

    pytest.importorskip("jieba")
    # when jieba is installed, zh must use real segmentation: 数据中心 / 每年 as whole words
    # (BudouX would either glue or over-split these)
    out = phrase_atoms("数据中心业务每年增长", "zh")
    assert "数据中心" in out and "每年" in out
    assert "".join(out) == "数据中心业务每年增长"  # byte-preserving


def test_phrase_atoms_ja_real_grouping():
    import pytest

    pytest.importorskip("budoux")
    # when budoux is installed, must use real grouping (not per-char fallback): atom count < char count, byte-preserving
    out = phrase_atoms("今日は天気です", "ja")
    assert "".join(out) == "今日は天気です"
    assert (
        1 < len(out) < 7
    )  # 7 chars -> grouped into multiple phrase nodes, not per-char


def test_no_space_sets_in_sync():
    # Both now alias the canonical core.langsets.LANGUAGES_WITHOUT_SPACES; this guards the re-exports.
    from voxweave.core.breakpoints import _NO_SPACE
    from voxweave.core.smart_split import LANGUAGES_WITHOUT_SPACES

    assert _NO_SPACE == LANGUAGES_WITHOUT_SPACES
    assert "yue" in _NO_SPACE
