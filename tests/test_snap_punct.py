"""snap_break_punct: snap zh sentence-boundary punctuation onto jieba word boundaries
(fixes smart_split splitting a word at a punctuation position)."""

import pytest

from voxweave.realign import snap_break_punct


def _units(text: str) -> list[dict]:
    """One unit per character (no-space language per-char contract); timestamps increase with position. Punctuation is also its own unit."""
    out: list[dict] = []
    for i, c in enumerate(text):
        out.append(
            {"text": c, "start": round(i * 0.1, 3), "end": round(i * 0.1 + 0.05, 3)}
        )
    return out


def _text(units: list[dict]) -> str:
    return "".join(u["text"] for u in units)


def _fake_starts(boundaries: set[int]):
    """Fake segmenter: inject a word-start offset set directly (bypasses real jieba/budoux; tests pure logic)."""
    return lambda cstr, lang: set(boundaries)


def test_period_snaps_to_word_boundary(monkeypatch):
    import voxweave.core.breakpoints as B

    # original "同比增长29%。数据中心" was output by Qwen as "29%数。据中心" (。 landed inside the 数据中心 word)
    units = _units("同比增长29%数。据中心")
    # content string "同比增长29%数据中心" (len 11): 数据中心 starts at offset 7
    monkeypatch.setattr(B, "word_starts", _fake_starts({4, 7, 9}))
    out = snap_break_punct(units, "zh")
    # 。 should be moved before 数: "同比增长29%。数据中心"
    assert _text(out) == "同比增长29%。数据中心"


def test_comma_snaps_forward(monkeypatch):
    import voxweave.core.breakpoints as B

    # "同比增长92，%依然" should become "同比增长92%，依然" (，shifts one position past %)
    units = _units("同比增长92，%依然")
    # content "同比增长92%依然" (len 9): 依然 starts at offset 7 (% is at 6)
    monkeypatch.setattr(B, "word_starts", _fake_starts({4, 7}))
    out = snap_break_punct(units, "zh")
    assert _text(out) == "同比增长92%，依然"


def test_no_move_when_already_on_boundary(monkeypatch):
    import voxweave.core.breakpoints as B

    units = _units("营收816亿美元，同比上涨")
    # content "营收816亿美元同比上涨": ，follows 元 (content position 8), 同 starts at offset 8 -> already on boundary, no move
    monkeypatch.setattr(B, "word_starts", _fake_starts({8}))
    out = snap_break_punct(units, "zh")
    assert _text(out) == "营收816亿美元，同比上涨"


def test_respects_max_shift(monkeypatch):
    import voxweave.core.breakpoints as B

    units = _units("同比增长29%数。据中心")
    # nearest word start is more than max_shift away from punctuation (3 > 2) -> no move
    monkeypatch.setattr(B, "word_starts", _fake_starts({3}))
    out = snap_break_punct(units, "zh")
    assert _text(out) == "同比增长29%数。据中心"  # unchanged


def test_noop_unsupported_lang():
    units = _units("hello. world")
    assert snap_break_punct(units, "en") == units  # space-delimited languages: no snap


def test_noop_ja():
    # ja does not use snap (BudouX too weak for this; ja punctuation misplacement is fixed by fuse_punct_into_text content alignment)
    units = _units("番酒造。りがわしらの仕事だ")
    assert snap_break_punct(units, "ja") == units


def test_noop_when_jieba_absent(monkeypatch):
    import voxweave.core.breakpoints as B

    monkeypatch.setattr(
        B, "word_starts", lambda cstr, lang: None
    )  # simulate jieba absent
    units = _units("同比增长29%数。据中心")
    out = snap_break_punct(units, "zh")
    assert _text(out) == "同比增长29%数。据中心"  # returned unchanged


def test_punct_time_monotonic(monkeypatch):
    import voxweave.core.breakpoints as B

    units = _units("同比增长29%数。据中心")
    monkeypatch.setattr(B, "word_starts", _fake_starts({4, 7, 9}))
    out = snap_break_punct(units, "zh")
    ts = [u["start"] for u in out]
    assert ts == sorted(
        ts
    )  # timestamps remain monotonically non-decreasing after reorder


def test_real_jieba_fixes_all_failing_cases():
    pytest.importorskip("jieba")
    cases = [
        ("同比增长29%数。据中心业务", "同比增长29%。数据中心业务"),
        ("核心位置值。得注意的是", "核心位置。值得注意的是"),
        ("的是本，季度面向", "的是，本季度面向"),
        (
            "份额从95%暴跌至零而中国",
            "份额从95%暴跌至零而中国",
        ),  # 而 is a conjunction; jieba segments it as a single char; no strict move expectation
    ]
    for src, _ in cases[:3]:  # first 3 have clear expected word boundaries
        out = snap_break_punct(_units(src), "zh")
        # assert that keywords are no longer split by punctuation
        joined = _text(out)
        for kw in ("数据中心", "值得", "本季度"):
            if kw[0] in src:
                # the word must not contain a sentence-boundary punctuation mark inside it
                import re

                assert not re.search(kw[0] + "[。！？；，、]" + kw[1], joined), (
                    f"{kw} still split: {joined!r}"
                )


def _word_units(text: str) -> list[dict]:
    """Same per-char stream, but with the surface under the ``word`` key —
    the other legal Unit shape (schema.Unit; corpus captures mirror it)."""
    out: list[dict] = []
    for i, c in enumerate(text):
        out.append(
            {"word": c, "start": round(i * 0.1, 3), "end": round(i * 0.1 + 0.05, 3)}
        )
    return out


def test_word_key_units_do_not_crash_and_snap_identically(monkeypatch):
    import voxweave.core.breakpoints as B

    monkeypatch.setattr(B, "word_starts", _fake_starts({4, 7, 9}))
    text_out = snap_break_punct(_units("同比增长29%数。据中心"), "zh")
    word_out = snap_break_punct(_word_units("同比增长29%数。据中心"), "zh")
    joined = "".join(u.get("text") or u.get("word") or "" for u in word_out)
    assert joined == _text(text_out) == "同比增长29%。数据中心"
    spans = [(u["start"], u["end"]) for u in word_out]
    assert spans == [(u["start"], u["end"]) for u in text_out]


def test_mixed_key_units_survive(monkeypatch):
    import voxweave.core.breakpoints as B

    units = _units("同比增长29%数。据中心")
    units[3] = {
        "word": units[3]["text"],
        "start": units[3]["start"],
        "end": units[3]["end"],
    }
    monkeypatch.setattr(B, "word_starts", _fake_starts({4, 7, 9}))
    out = snap_break_punct(units, "zh")
    assert (
        "".join(u.get("text") or u.get("word") or "" for u in out)
        == "同比增长29%。数据中心"
    )
