# tests/test_cursor_recovery.py
# _anchor_cursor only scans +/-8 units around the blind cursor. A longer upstream
# desync (a ghost run of units, or a clause whose units were lost) falls outside
# that window, _anchor_cursor reports ok=False, and split_at_sentence_end then
# keeps slicing with the stale cursor -- so every later cue in the segment carries
# another sentence's timing. These tests pin the recovery contract: the sentence
# that cannot be anchored may degrade to proportional timing, but the sentences
# after it must be paired exactly again.
import logging

import pytest

from voxweave.core.smart_split import split_at_sentence_end

STEP = 0.5
DUR = 0.4
CJK_STEP = 0.2
CJK_DUR = 0.1


def _units(tokens, step=STEP, dur=DUR):
    """Uniformly timed word_data for a token stream (ghost units included)."""
    return [
        {"word": tok, "start": i * step, "end": i * step + dur}
        for i, tok in enumerate(tokens)
    ]


def _expected_tokens(text, lang):
    if lang in {"ja", "zh", "yue"}:
        return [c for c in text if not c.isspace()]
    return text.split()


def _assert_paired(cues, lang):
    """Each cue's word_data must be the units of exactly the tokens in its text."""
    for cue in cues:
        words = [(w.get("word") or "").strip() for w in cue["word_data"]]
        assert words == _expected_tokens(cue["text"], lang), (
            f"cue text/word_data desync: text={cue['text']!r} words={words!r}"
        )


def _messages(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


S1 = "alpha beta gamma delta."
S2 = "xa xb xc xd xe xf xg xh xi xj xk xl."
S3 = "rho sigma tau upsilon."
S4 = "phi chi psi omega."
TEXT = f"{S1} {S2} {S3} {S4}"
GHOSTS = [f"g{i}" for i in range(10)]  # 10 > _anchor_cursor's max_shift of 8


def test_ghost_run_beyond_window_resyncs_forward(caplog):
    # A ghost run of 10 units sits between sentence 1 and sentence 2, so sentence
    # 2's real units are 10 ahead of the cursor -- past the +/-8 rescan. The whole
    # sentence is still present in word_data, so a forward re-anchor scan over the
    # remaining window must find it and every cue must pair exactly.
    stream = S1.split() + GHOSTS + S2.split() + S3.split() + S4.split()
    with caplog.at_level(logging.WARNING):
        cues = split_at_sentence_end(TEXT, _units(stream), "en", 42, 2)
    assert [c["text"].strip() for c in cues] == [S1, S2, S3, S4]
    _assert_paired(cues, "en")
    assert cues[1]["start"] == pytest.approx(stream.index("xa") * STEP)
    assert cues[2]["start"] == pytest.approx(stream.index("rho") * STEP)
    messages = _messages(caplog)
    assert any("desync" in m for m in messages), messages
    assert any("xa xb" in m for m in messages), (
        f"the warning must name the resynced sentence: {messages!r}"
    )


def test_unanchorable_sentence_does_not_desync_followers(caplog):
    # The ghost run is spliced into the middle of sentence 2, so that sentence
    # matches nowhere in word_data. Its own timing may degrade to a proportional
    # fill, but sentences 3 and 4 must be re-anchored on content instead of
    # inheriting the 10-unit shift.
    s2 = S2.split()
    stream = S1.split() + s2[:5] + GHOSTS + s2[5:] + S3.split() + S4.split()
    word_data = _units(stream)
    with caplog.at_level(logging.WARNING):
        cues = split_at_sentence_end(TEXT, word_data, "en", 42, 2)
    assert [c["text"].strip() for c in cues] == [S1, S2, S3, S4]

    # Downstream sentences are paired again and carry their own units' timing.
    _assert_paired([cues[0], cues[2], cues[3]], "en")
    assert cues[2]["start"] == pytest.approx(stream.index("rho") * STEP)
    assert cues[3]["start"] == pytest.approx(stream.index("phi") * STEP)

    # The unrecoverable sentence keeps monotone timing inside its own window.
    bad = cues[1]
    assert bad["start"] < bad["end"]
    assert bad["start"] >= cues[0]["end"] - 1e-9
    assert bad["end"] <= cues[2]["start"] + 1e-9
    assert bad["end"] <= word_data[-1]["end"] + 1e-9

    messages = _messages(caplog)
    assert any("desync" in m for m in messages), messages
    assert any("xa xb" in m for m in messages), (
        f"the warning must name the failing sentence: {messages!r}"
    )


JA_S1 = "今日は晴れです。"
JA_S2 = "明日は雨が降ります。"
JA_S3 = "傘を持って行きます。"
JA_TEXT = JA_S1 + JA_S2 + JA_S3
JA_GHOSTS = ["ん"] * 10


def test_no_space_ghost_run_beyond_window_resyncs():
    # Same failure on the char-level CJK cursor: a 10-unit ghost run pushes every
    # later sentence past the local rescan window.
    stream = list(JA_S1) + JA_GHOSTS + list(JA_S2) + list(JA_S3)
    cues = split_at_sentence_end(
        JA_TEXT, _units(stream, CJK_STEP, CJK_DUR), "ja", 18, 1
    )
    assert [c["text"].strip() for c in cues] == [JA_S1, JA_S2, JA_S3]
    _assert_paired(cues, "ja")
    assert cues[1]["start"] == pytest.approx(stream.index("明") * CJK_STEP)
    assert cues[2]["start"] == pytest.approx(stream.index("傘") * CJK_STEP)


def test_clean_stream_needs_no_recovery(caplog):
    # Control: recovery must not fire on a well-formed stream.
    with caplog.at_level(logging.WARNING):
        cues = split_at_sentence_end(TEXT, _units(TEXT.split()), "en", 42, 2)
    assert [c["text"].strip() for c in cues] == [S1, S2, S3, S4]
    _assert_paired(cues, "en")
    assert cues[0]["start"] == pytest.approx(0.0)
    assert not _messages(caplog)
