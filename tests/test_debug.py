import json

import pytest

from voxweave.debug import DebugSink, FileDebugSink


def test_noop_sink_writes_nothing(tmp_path):
    sink = DebugSink()
    assert sink.enabled is False
    sink.audio("x.wav", tmp_path / "nope.wav")
    sink.chunk(
        0,
        wav=tmp_path / "n.wav",
        start=0.0,
        end=1.0,
        raw="r",
        text="t",
        lang="en",
        units=None,
    )
    sink.meta({"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_file_sink_writes_artifacts(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFfake")
    sink = FileDebugSink("clip", base=tmp_path / "debug")
    sink.audio("02_speech_16k.wav", src)
    sink.chunk(
        3,
        wav=src,
        start=1.5,
        end=4.25,
        raw="language English<asr_text>hi",
        text="hi",
        lang="English",
        units=[{"text": "hi", "start": 1.5, "end": 4.0}],
    )
    # chunk skipped due to empty ASR: units=None, but wav/raw/text/lang are still saved
    sink.chunk(4, wav=src, start=4.25, end=5.0, raw="", text="", lang=None, units=None)
    sink.meta({"separate": True, "cues": 7})

    root = tmp_path / "debug" / "clip"
    ch = root / "chunks"
    assert (root / "02_speech_16k.wav").read_bytes() == b"RIFFfake"
    assert (ch / "003_1.5-4.2.wav").exists()
    assert (ch / "003_1.5-4.2.raw.txt").read_text() == "language English<asr_text>hi"
    assert (ch / "003_1.5-4.2.text.txt").read_text() == "hi"
    assert (ch / "003_1.5-4.2.lang.txt").read_text() == "English"
    assert json.loads((ch / "003_1.5-4.2.units.json").read_text())[0]["text"] == "hi"
    assert (ch / "004_4.2-5.0.wav").exists()
    assert (ch / "004_4.2-5.0.raw.txt").read_text() == ""
    assert not (ch / "004_4.2-5.0.units.json").exists()
    assert json.loads((root / "meta.json").read_text())["cues"] == 7


def test_file_sink_has_no_implicit_adjacent_debug_root():
    with pytest.raises(ValueError, match="exactly one of base or root is required"):
        FileDebugSink("clip")
