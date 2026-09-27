import json

from voxweave.debug import HEALTH_FILE, DebugSink, FileDebugSink


def test_noop_sink_writes_nothing(tmp_path):
    sink = DebugSink()
    sink.audio("x.wav", tmp_path / "nope.wav")
    sink.chunk(
        0,
        wav=tmp_path / "n.wav",
        start=0.0,
        end=1.0,
        text="t",
        lang="en",
        units=None,
    )
    sink.meta({"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_file_sink_writes_artifacts(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFfake")
    root = tmp_path / "debug" / "clip"
    sink = FileDebugSink(root)
    sink.audio("02_speech_16k.wav", src)
    sink.chunk(
        3,
        wav=src,
        start=1.5,
        end=4.25,
        text="hi",
        lang="English",
        units=[{"text": "hi", "start": 1.5, "end": 4.0}],
    )
    # chunk skipped due to empty ASR: units=None, but wav/text/lang are still saved
    sink.chunk(4, wav=src, start=4.25, end=5.0, text="", lang=None, units=None)
    sink.meta({"separate": True, "cues": 7})

    ch = root / "chunks"
    assert (root / "02_speech_16k.wav").read_bytes() == b"RIFFfake"
    assert (ch / "003_1.5-4.2.wav").exists()
    assert (ch / "003_1.5-4.2.text.txt").read_text() == "hi"
    assert (ch / "003_1.5-4.2.lang.txt").read_text() == "English"
    assert json.loads((ch / "003_1.5-4.2.units.json").read_text())[0]["text"] == "hi"
    assert (ch / "004_4.2-5.0.wav").exists()
    assert (ch / "004_4.2-5.0.text.txt").read_text() == ""
    assert not (ch / "004_4.2-5.0.units.json").exists()
    # The chunk text is saved once; no duplicate "raw" copy of the same string.
    assert not list(ch.glob("*.raw.txt"))
    assert json.loads((root / "meta.json").read_text())["cues"] == 7


def test_file_sink_replaces_the_previous_run_bundle(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFfake")
    root = tmp_path / "debug"
    first = FileDebugSink(root)
    first.audio("00_fullband_44k.wav", src)
    first.chunk(0, wav=src, start=0.0, end=9.0, text="old", lang="en", units=[])
    first.position_units([], [], language="English")
    assert (root / HEALTH_FILE).exists()

    second = FileDebugSink(root)
    second.chunk(0, wav=src, start=0.0, end=4.0, text="new", lang="en", units=[])

    # Nothing from the first run survives beside the second run's files.
    assert sorted(p.name for p in (root / "chunks").iterdir()) == [
        "000_0.0-4.0.lang.txt",
        "000_0.0-4.0.text.txt",
        "000_0.0-4.0.units.json",
        "000_0.0-4.0.wav",
    ]
    assert not (root / "00_fullband_44k.wav").exists()
    assert not (root / HEALTH_FILE).exists()
