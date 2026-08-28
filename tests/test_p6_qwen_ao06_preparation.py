from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest


def _write_two_cue_episode(tmp_path: Path) -> tuple[Path, Path]:
    media_path = tmp_path / "episode.wav"
    with wave.open(str(media_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 48_000)
    (tmp_path / "episode.json").write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": [
                    {"text": "你", "start": 0.0, "end": 1.0},
                    {"text": "好", "start": 1.0, "end": 2.0},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "你\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "好\n",
        encoding="utf-8",
    )
    return vtt_path, media_path


def test_qwen_prepares_every_window_before_starting_any_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import backend, config, pipeline, realign
    from voxweave.align_acquisition import qwen_sample_geometry
    from voxweave.align_runtime import capture_align_runtime_trace

    vtt_path, media_path = _write_two_cue_episode(tmp_path)
    first_window = tmp_path / "first-qwen-window.wav"
    slice_count = 0
    backend_count = 0

    monkeypatch.setattr(
        pipeline,
        "_prepare_16k_for_align",
        lambda *_args, **_kwargs: media_path,
    )
    monkeypatch.setattr(backend, "uses_mms", lambda _iso: False)
    monkeypatch.setattr(config, "align_model_for", lambda _iso: None)
    monkeypatch.setattr(realign, "crop_blocks", lambda _spans: [(0.0, 1.0), (1.0, 2.0)])
    monkeypatch.setattr(backend, "release", lambda: None)

    def staged_slice(
        _wav: Path,
        start: float,
        end: float,
        *,
        _sample_geometry_observer=None,
        **_kwargs: Any,
    ) -> Path:
        nonlocal slice_count
        slice_count += 1
        if slice_count == 2:
            raise OSError("second Qwen window failed")
        geometry = qwen_sample_geometry(
            nominal_start=float(start),
            nominal_end=float(end),
            sample_rate=16_000,
            sample_count=48_000,
        )
        if _sample_geometry_observer is not None:
            _sample_geometry_observer(
                geometry.sample_start,
                geometry.sample_end,
                geometry.sample_rate,
                geometry.sample_count,
            )
        first_window.write_bytes(media_path.read_bytes())
        return first_window

    def count_backend(
        _wav: Path,
        text: str,
        _iso: str,
    ) -> list[dict[str, Any]]:
        nonlocal backend_count
        backend_count += 1
        return [{"text": text, "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(pipeline, "slice_wav", staged_slice)
    monkeypatch.setattr(backend, "align_text", count_backend)

    with capture_align_runtime_trace() as capture:
        with pytest.raises(OSError, match="^second Qwen window failed$"):
            pipeline.align(vtt_path, media_path=media_path, separate=False)
    trace = capture.snapshot()

    assert slice_count == 2
    assert backend_count == 0
    assert [
        (event.phase, event.activity)
        for event in trace.events
        if event.state == "failed"
    ] == [("AO-06", "physical-call-preparation")]
    assert (
        sum(
            event.phase == "AO-06"
            and event.activity == "physical-call-preparation"
            and event.state == "started"
            for event in trace.events
        )
        == 1
    )
    assert not [
        event
        for event in trace.events
        if event.phase == "AO-06" and event.state == "completed"
    ]
    assert not [
        event
        for event in trace.events
        if 7 <= int(event.phase.removeprefix("AO-")) <= 23 or event.phase == "AO-25"
    ]
    assert any(
        event.phase == "AO-24"
        and event.activity == "backend-and-audio-temp-disposal"
        and event.state == "completed"
        for event in trace.events
    )
    assert not first_window.exists()
