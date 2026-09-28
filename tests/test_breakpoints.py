# tests/test_breakpoints.py
from voxweave.core.breakpoints import phrase_atoms


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
