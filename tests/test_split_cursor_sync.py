# tests/test_split_cursor_sync.py
# split_at_sentence_end zips clause tokens back onto word_data by index. A sentence
# boundary falling inside a whitespace token (e.g. an ASR token transcribing laughter
# as "哈哈哈哈哈！哇。") inflates the token count and silently shifts every later cue's
# word timing for the rest of the file. These tests pin the pairing contract:
# each cue's word_data must be the units of exactly the tokens in its text.
import logging

from voxweave import pipeline
from voxweave.core.smart_split import split_at_sentence_end


def _word_data(text, step=0.5, dur=0.4):
    return [
        {"word": tok, "start": i * step, "end": i * step + dur}
        for i, tok in enumerate(text.split())
    ]


def _assert_paired(cues):
    for cue in cues:
        words = [w["word"] for w in cue["word_data"]]
        assert words == cue["text"].split(), (
            f"cue text/word_data desync: text={cue['text']!r} words={words!r}"
        )


def test_sentence_boundary_inside_token_keeps_pairing():
    # pysbd splits the single token 哈哈哈哈哈！哇。 at the fullwidth ！ -> one unit
    # counted as two tokens -> every cue after it shifts by one word (the PHM bug).
    text = "so funny 哈哈哈哈哈！哇。 and now we keep talking"
    cues = split_at_sentence_end(text, _word_data(text), "en", 42, 2)
    _assert_paired(cues)


def test_clean_sentence_boundaries_unchanged():
    text = "alpha beta gamma. delta epsilon zeta."
    cues = split_at_sentence_end(text, _word_data(text), "en", 42, 2)
    assert [c["text"].strip() for c in cues] == [
        "alpha beta gamma.",
        "delta epsilon zeta.",
    ]
    _assert_paired(cues)


def test_comma_inside_quoted_token_keeps_pairing():
    # A comma followed by a closing quote (so,") is inside one whitespace token;
    # comma clause-splitting there divides the token and desyncs the index zip
    # (PHM tail: 'say "I told you so," even though you were right.').
    text = (
        'At least I never have to hear you say "I told you so,"'
        " even though you were right."
    )
    cues = split_at_sentence_end(text, _word_data(text), "en", 42, 2)
    _assert_paired(cues)


def test_token_final_comma_still_splits():
    # Normal comma at a token boundary keeps splitting into separate cues.
    text = "after we finished the entire harvest celebration, everyone went back home to rest."
    cues = split_at_sentence_end(text, _word_data(text), "en", 42, 2)
    assert len(cues) == 2
    assert cues[0]["text"].strip().endswith("celebration,")
    _assert_paired(cues)


def test_extra_unit_resyncs_following_clause(caplog):
    # word_data has a ghost unit the text does not (unknown upstream desync):
    # the next clause must re-anchor on content instead of inheriting the shift.
    text = "alpha beta gamma. delta epsilon zeta."
    word_data = _word_data(text)
    ghost = {"word": "ghost", "start": 1.45, "end": 1.49}
    word_data.insert(3, ghost)
    with caplog.at_level(logging.WARNING):
        cues = split_at_sentence_end(text, word_data, "en", 42, 2)
    second = cues[1]
    assert [w["word"] for w in second["word_data"]] == ["delta", "epsilon", "zeta."]
    assert second["start"] == 1.5
    assert any("desync" in r.message for r in caplog.records)


def test_unrecoverable_desync_warns_and_degrades(caplog):
    # A unit is missing mid-clause: no shift can restore pairing. Keep legacy
    # slicing (timing may be off for that clause) but never crash, and warn.
    text = "alpha beta gamma delta epsilon"
    word_data = _word_data(text)
    del word_data[2]  # "gamma" unit lost upstream
    with caplog.at_level(logging.WARNING):
        cues = split_at_sentence_end(text, word_data, "en", 42, 2)
    assert cues and cues[0]["text"].split()[0] == "alpha"
    assert any("desync" in r.message for r in caplog.records)


def test_pipeline_split_accepts_vtt_path(tmp_path):
    # `voxweave split foo.vtt` should resolve the sibling JSON instead of
    # feeding WEBVTT bytes to json.loads.
    units = _word_data("hello there. general kenobi.")
    json_path = tmp_path / "clip.json"
    json_path.write_text(
        pipeline.json.dumps(
            {
                "language": "en",
                "word_segments": [
                    {"text": u["word"], "start": u["start"], "end": u["end"]}
                    for u in units
                ],
                "segments": [],
                "vad_speech": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "clip.vtt").write_text("WEBVTT\n\nhello there\n", encoding="utf-8")
    out = pipeline.split(tmp_path / "clip.vtt")
    assert out == tmp_path / "clip.vtt"
    assert "hello there" in out.read_text(encoding="utf-8")
