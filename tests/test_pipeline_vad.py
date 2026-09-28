# tests/test_pipeline_vad.py
import json
from pathlib import Path


def test_split_uses_persisted_vad(tmp_path: Path, monkeypatch):
    calls = {}
    import voxweave.pipeline as P

    def fake_split(segs, lang, **kw):
        calls.update(kw)
        return [{"text": "a", "start": 0.0, "end": 1.0}]

    # split() lazy-imports smart_split_segments, so patch it at the SOURCE module
    # (patching voxweave.pipeline.smart_split_segments would not intercept it):
    monkeypatch.setattr("voxweave.core.smart_split.smart_split_segments", fake_split)
    j = tmp_path / "ep.json"
    j.write_text(
        json.dumps(
            {
                "language": "ja",
                "word_segments": [{"text": "あ", "start": 0.0, "end": 0.1}],
                "vad_speech": [[0.0, 0.2]],
            }
        ),
        encoding="utf-8",
    )
    P.split(j)
    assert calls["speech_spans"] == [(0.0, 0.2)]
    assert "thresholds" in calls


def test_split_without_vad_degrades_to_none(tmp_path: Path, monkeypatch):
    calls = {}
    import voxweave.pipeline as P

    def fake_split(segs, lang, **kw):
        calls.update(kw)
        return [{"text": "a", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr("voxweave.core.smart_split.smart_split_segments", fake_split)
    j = tmp_path / "old.json"  # legacy JSON predating vad_speech persistence
    j.write_text(
        json.dumps(
            {
                "language": "en",
                "word_segments": [{"text": "a", "start": 0.0, "end": 0.1}],
            }
        ),
        encoding="utf-8",
    )
    P.split(j)
    assert calls["speech_spans"] is None
    assert "thresholds" in calls
