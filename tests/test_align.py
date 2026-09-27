import json
from unittest.mock import patch

import pytest

from voxweave import pipeline


def _setup(tmp_path, vtt_text):
    stem = tmp_path / "ep"
    media = stem.with_suffix(".wav")
    media.write_bytes(b"x")
    ws = [
        {"text": "你", "start": 0.0, "end": 0.5},
        {"text": "好", "start": 0.5, "end": 1.0},
        {"text": "世", "start": 2.0, "end": 2.5},
        {"text": "界", "start": 2.5, "end": 3.0},
    ]
    stem.with_suffix(".json").write_text(
        json.dumps({"language": "zh", "word_segments": ws}, ensure_ascii=False),
        encoding="utf-8",
    )
    vtt = stem.with_suffix(".vtt")
    vtt.write_text(vtt_text, encoding="utf-8")
    return media, vtt, stem.with_suffix(".json")


def _fake_align(cwav, text, lang):
    # one unit per character, timestamps increase with character position (relative to slice)
    return [
        {"text": c, "start": float(i), "end": float(i) + 0.5}
        for i, c in enumerate(text)
    ]


def _run_align(media, vtt, vad=None):
    with (
        patch("voxweave.pipeline._prepare_16k_for_align", return_value=media),
        patch("voxweave.pipeline.slice_wav", return_value=media),
        patch("voxweave.backend.align_text", side_effect=_fake_align),
        # align no longer does VAD post-processing (per-sentence crop CTC trusts raw units);
        # patch only prevents accidental real VAD calls
        patch("voxweave.pipeline.vad_speech_segments", return_value=vad or []),
    ):
        return pipeline.align(vtt)


def test_align_overwrites_vtt_with_timestamps(tmp_path):
    media, vtt, json_path = _setup(tmp_path, "WEBVTT\n\n你好\n\n世界\n")
    out = _run_align(media, vtt)
    assert out == vtt
    content = vtt.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert content.count("-->") == 2
    assert "你好" in content and "世界" in content


def test_align_updates_json(tmp_path):
    media, vtt, json_path = _setup(tmp_path, "WEBVTT\n\n你好\n\n世界\n")
    _run_align(media, vtt)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["language"] == "zh"
    assert len(data["segments"]) == 2
    assert [s["text"] for s in data["segments"]] == ["你好", "世界"]
    assert len(data["word_segments"]) == 4  # newly aligned units


def test_align_timestamps_monotonic(tmp_path):
    media, vtt, _ = _setup(tmp_path, "WEBVTT\n\n你好\n\n世界\n")
    _run_align(media, vtt)
    data = json.loads((vtt.with_suffix(".json")).read_text(encoding="utf-8"))
    segs = data["segments"]
    for s in segs:
        assert s["end"] > s["start"]
    assert (
        segs[0]["end"] <= segs[1]["start"] + 1e-6
        or segs[1]["start"] >= segs[0]["start"]
    )


def test_align_missing_media_raises(tmp_path):
    stem = tmp_path / "noep"
    stem.with_suffix(".json").write_text(
        json.dumps({"language": "zh", "word_segments": []}), encoding="utf-8"
    )
    vtt = stem.with_suffix(".vtt")
    vtt.write_text("WEBVTT\n\n你好\n", encoding="utf-8")
    try:
        pipeline.align(vtt)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_align_persists_vad_speech(tmp_path):
    # VAD segments persisted by transcribe are passed through into align's output JSON →
    # split can reuse them offline for gap-based segmentation
    media, vtt, json_path = _setup(tmp_path, "WEBVTT\n\n你好\n\n世界\n")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["vad_speech"] = [[0.0, 1.0], [2.0, 3.0]]
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _run_align(media, vtt)
    out = json.loads(json_path.read_text(encoding="utf-8"))
    assert out["vad_speech"] == [[0.0, 1.0], [2.0, 3.0]]


def test_align_trusts_raw_units_no_vad_carve(tmp_path):
    # WhisperX equivalent: align trusts raw CTC timestamps from cropped window; **no VAD snap/carve**
    # (tight crop leaves no drift space). Even if JSON has persisted VAD, align does not use it to
    # trim words — raw last-word timestamp is preserved as-is.
    media, vtt, json_path = _setup(tmp_path, "WEBVTT\n\n你好\n")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["vad_speech"] = [[0.0, 1.0]]  # VAD only extends to 1.0
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _fake_raw(cwav, text, lang):
        return [
            {"text": "你", "start": 0.0, "end": 0.5},
            {"text": "好", "start": 0.5, "end": 8.0},  # raw end at 8.0
        ]

    with (
        patch("voxweave.pipeline._prepare_16k_for_align", return_value=media),
        patch("voxweave.pipeline.slice_wav", return_value=media),
        patch("voxweave.pipeline.realign.crop_blocks", return_value=[(0.0, 10.0)]),
        patch("voxweave.backend.align_text", side_effect=_fake_raw),
    ):
        pipeline.align(vtt)

    out = json.loads(json_path.read_text(encoding="utf-8"))
    ws = out["word_segments"]
    assert (
        ws[1]["text"] == "好" and ws[1]["end"] == 8.0
    )  # not trimmed; raw value preserved
    assert out["vad_speech"] == [
        [0.0, 1.0]
    ]  # persisted VAD still passed through (for split reuse); just not used for carving


def test_align_no_routing_anchor_raises(tmp_path):
    # no word_segments and VTT has no timestamps → routing impossible
    media, vtt, json_path = _setup(tmp_path, "WEBVTT\n\n你好\n")
    json_path.write_text(
        json.dumps({"language": "zh", "word_segments": []}), encoding="utf-8"
    )
    try:
        pipeline.align(vtt)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_align_partly_untimed_vtt_names_the_untimed_cues(tmp_path):
    # no word_segments and only some cues timed → the error counts the untimed ones
    media, vtt, json_path = _setup(
        tmp_path, "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n你好\n\n世界\n"
    )
    json_path.write_text(
        json.dumps({"language": "zh", "word_segments": []}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError) as caught:
        pipeline.align(vtt)
    assert str(caught.value) == (
        "ep.json has no word_segments and 1 of 2 VTT cues have no timestamps; "
        "cannot route audio windows (give every cue a timing line, or restore "
        "the sibling JSON's word_segments)"
    )
    assert caught.value.failure.kind == "qwen-route-invalid"
    assert caught.value.failure.detail_code == "no-route-source"
