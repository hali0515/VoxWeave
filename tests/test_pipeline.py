"""Regression tests for pipeline path derivation and sibling file writing.

Key concern: filenames with interior dots (e.g. ``...`` in YouTube titles) must not be
silently truncated by ``Path.with_suffix`` when deriving .vtt/.json sibling paths.
"""

import logging
from pathlib import Path

import pytest

from voxweave import backend, pipeline

# Real filename that triggered the bug: title has interior ``...``
DOTTED = (
    "Fuwawa一次播放所有音樂來轟炸鄰居...氣到Kronii受不了【Hololive 中文】 [s6r36ux1d4Q]"
)


def test_swap_ext_preserves_mid_name_dots():
    m = Path(f"{DOTTED}.webm")
    assert pipeline.swap_ext(m, ".vtt").name == f"{DOTTED}.vtt"
    assert pipeline.swap_ext(m, ".json").name == f"{DOTTED}.json"
    # chained (translate output): .vtt -> .zh.vtt also must not truncate
    v = pipeline.swap_ext(m, ".vtt")
    assert pipeline.swap_ext(v, ".zh.vtt").name == f"{DOTTED}.zh.vtt"


def test_swap_ext_no_suffix_appends():
    assert pipeline.swap_ext(Path("README"), ".vtt").name == "README.vtt"


def test_process_dotted_filename_writes_full_siblings(tmp_path):
    media = (
        tmp_path / f"{DOTTED}.webm"
    )  # does not need to exist: word_segments bypasses transcription
    units = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0},
    ]
    out = pipeline.process(media, word_segments=("en", units))
    # returned .vtt path must be the full name, not truncated at the first ``...``
    assert out == tmp_path / f"{DOTTED}.vtt"
    assert out.exists()
    assert (tmp_path / f"{DOTTED}.json").exists()
    # truncated name must not exist
    assert not (tmp_path / "Fuwawa一次播放所有音樂來轟炸鄰居...vtt").exists()


def test_process_vtt_has_timestamps_by_default(tmp_path):
    # default process output includes timestamps: cues carry word-level start/end -> timing line written
    media = tmp_path / "ep.mkv"
    units = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0},
    ]
    out = pipeline.process(media, word_segments=("en", units))
    body = out.read_text(encoding="utf-8")
    assert "-->" in body
    assert "00:00:00.000 -->" in body
    # re-parse as timestamped blocks (align path compat): all blocks carry start/end
    blocks = pipeline.realign.parse_vtt_blocks(body)
    assert blocks and all(b["start"] is not None for b in blocks)


def test_process_no_timestamps_strips(tmp_path):
    # --no-timestamps: plain-text edit draft, no timing lines (edit text/breaks then re-run align)
    media = tmp_path / "ep.mkv"
    units = [{"text": "hi", "start": 0.0, "end": 1.0}]
    out = pipeline.process(media, word_segments=("en", units), timestamps=False)
    body = out.read_text(encoding="utf-8")
    assert "-->" not in body
    assert "hi" in body


def test_write_siblings_drops_ts_line_when_cue_time_missing(tmp_path):
    # defensive: cue missing start/end (rare) -> falls back to plain text, does not crash (fmt_ts rejects None)
    cues = [
        {"text": "a", "start": None, "end": None},
        {"text": "b", "start": 0.0, "end": 1.0},
    ]
    out = pipeline._write_siblings(tmp_path / "x.mkv", cues, [], "en")
    body = out.read_text(encoding="utf-8")
    assert "00:00:00.000 --> 00:00:01.000" in body  # second cue has timing
    # first cue (a) has no timing line: line before "a" must be blank, not "-->"
    lines = body.splitlines()
    assert "a" in lines
    assert lines[lines.index("a") - 1] == ""


def test_find_sibling_media_matches_dotted_name(tmp_path):
    media = tmp_path / f"{DOTTED}.webm"
    media.write_bytes(b"x")
    vtt = tmp_path / f"{DOTTED}.vtt"
    assert pipeline._find_sibling_media(vtt) == media


def test_split_corrupt_sibling_json_raises_readable_error(tmp_path):
    j = tmp_path / "ep.json"
    j.write_text('{"word_segments": [truncated', encoding="utf-8")
    with pytest.raises(RuntimeError, match="ep.json.*corrupt"):
        pipeline.split(j)


def test_split_missing_word_segments_raises_readable_error(tmp_path):
    j = tmp_path / "ep.json"
    j.write_text('{"language": "en"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="word_segments"):
        pipeline.split(j)


def test_align_corrupt_sibling_json_raises_readable_error(tmp_path):
    # a half-written .json next to the .vtt must fail with a message naming the
    # file, not a bare JSONDecodeError deep in the stack
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8")
    (tmp_path / "ep.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ep.json.*corrupt"):
        pipeline.align(vtt)


def test_separate_self_cleans_partial_temps_on_failure(tmp_path, monkeypatch):
    # Regression: _separate_to_16k_32k must unlink already-decoded temps if a later step raises.
    # Callers register the returned paths in their `tmp` cleanup list only AFTER a clean return,
    # so an OOM/ffmpeg failure mid-separation would otherwise orphan the fullband temp file.
    created: list[Path] = []

    def fake_decode(media, **kw):
        p = tmp_path / f"f{len(created)}.wav"
        p.write_bytes(b"x")
        created.append(p)
        return p

    def boom(fullband, **kw):
        raise RuntimeError("separation OOM")

    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
    monkeypatch.setattr(backend, "separate_vocals", boom)

    with pytest.raises(RuntimeError):
        pipeline._separate_to_16k_32k(
            tmp_path / "m.mkv", reporter=pipeline.Reporter(), normalize=False
        )
    # fullband was decoded before separation failed -> helper must have cleaned it up
    assert created and not created[0].exists()


# --- #18: _spans_in / _turns_in must skip malformed persisted entries instead of crashing ---


def test_spans_in_skips_malformed_entries_and_warns(caplog):
    # [2] has wrong arity (missing end); [3, "x"] has a non-numeric end.
    # Both must be skipped (not raise) while the well-formed entry survives.
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = pipeline._spans_in([[0, 1], [2], [3, "x"]])
    assert result == [(0.0, 1.0)]
    assert any("malformed" in r.getMessage().lower() for r in caplog.records)


def test_spans_in_all_malformed_returns_none(caplog):
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = pipeline._spans_in([[1], ["a", "b"]])
    assert result is None


def test_turns_in_skips_malformed_entries_and_warns(caplog):
    # [2] has wrong arity (missing end + label); the well-formed entry survives.
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = pipeline._turns_in([[0, 1, "A"], [2]])
    assert result == [(0.0, 1.0, "A")]
    assert any("malformed" in r.getMessage().lower() for r in caplog.records)


def test_turns_in_all_malformed_returns_none(caplog):
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = pipeline._turns_in([[1], ["a", "b", "c", "d"]])
    assert result is None


# --- #23: SDH sidecar failure must not lose the already-written main VTT ---


def test_process_sdh_sidecar_failure_does_not_lose_main_vtt(
    tmp_path, monkeypatch, caplog
):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"x")
    units = [{"text": "hello", "start": 0.0, "end": 1.0}]

    def fake_transcribe(*a, **kw):
        return ("en", units, None, [], [])

    def boom_sdh(*a, **kw):
        raise RuntimeError("PANNs exploded")

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "_write_sdh_sidecar", boom_sdh)

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        out = pipeline.process(media, sdh=True, shot_snap=False)

    assert out == tmp_path / "ep.vtt"
    assert out.exists()
    assert any("sdh" in r.getMessage().lower() for r in caplog.records)


# --- #26: sibling media lookup must be case-insensitive and warn on ambiguous matches ---


def test_find_sibling_media_case_insensitive_extension(tmp_path):
    media = tmp_path / "ep.MP4"
    media.write_bytes(b"x")
    vtt = tmp_path / "ep.vtt"
    assert pipeline._find_sibling_media(vtt) == media


def test_find_sibling_media_multiple_candidates_warns_and_is_deterministic(
    tmp_path, caplog
):
    # Both ep.mkv and ep.mp4 exist; MEDIA_EXTS lists ".mkv" before ".mp4", so the
    # first-by-order candidate must win, and the ambiguity must be logged.
    mkv = tmp_path / "ep.mkv"
    mkv.write_bytes(b"x")
    mp4 = tmp_path / "ep.mp4"
    mp4.write_bytes(b"x")
    vtt = tmp_path / "ep.vtt"
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        found = pipeline._find_sibling_media(vtt)
    assert found == mkv
    assert any("multiple" in r.getMessage().lower() for r in caplog.records)
