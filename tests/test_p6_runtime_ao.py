from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest


class _DisplayHelperFailure(RuntimeError):
    pass


class _StrictCaptureFailure(RuntimeError):
    pass


def _write_public_align_episode(
    tmp_path: Path,
    *,
    route: str,
) -> tuple[Path, Path]:
    language, text = {
        "ctc-full": ("en", "hello"),
        "mms-full": ("ja", "あ"),
        "qwen-crop": ("zh", "你"),
    }[route]
    media_path = tmp_path / "episode.wav"
    with wave.open(str(media_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 32_000)
    (tmp_path / "episode.json").write_text(
        json.dumps(
            {
                "language": language,
                "word_segments": [
                    {"text": text, "start": 0.0, "end": 1.0},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text(
        f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{text}\n",
        encoding="utf-8",
    )
    return vtt_path, media_path


def _stub_public_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    route: str,
    media_path: Path,
) -> None:
    from voxweave import backend, config, pipeline
    from voxweave.align_acquisition import qwen_sample_geometry

    monkeypatch.setattr(
        pipeline,
        "_prepare_16k_for_align",
        lambda *_args, **_kwargs: media_path,
    )
    monkeypatch.setattr(backend, "release", lambda: None)
    monkeypatch.setattr(backend, "uses_mms", lambda _iso: route == "mms-full")
    monkeypatch.setattr(
        config,
        "align_model_for",
        lambda _iso: "synthetic-ctc" if route == "ctc-full" else None,
    )

    if route in {"ctc-full", "mms-full"}:
        target = (
            "align_blocks_full_ctc"
            if route == "ctc-full"
            else "align_blocks_full_mms"
        )

        def fake_full_pass(
            _wav: Path,
            texts: list[str],
            _iso: str,
            *_args: Any,
            _raw_call_observer=None,
            _backend_invoker=None,
            **_kwargs: Any,
        ) -> list[list[dict[str, Any]]]:
            raw = [
                {"text": texts[0], "start": 0.0, "end": 1.0},
            ]
            if _backend_invoker is not None:
                raw = _backend_invoker(lambda: raw)
            if _raw_call_observer is not None:
                _raw_call_observer(
                    raw,
                    None,
                    (0,),
                    0.0,
                    audio_sample_start=0,
                    audio_sample_end=32_000,
                    sample_rate=16_000,
                    sample_count=32_000,
                    nominal_end_seconds=None,
                )
            return [raw]

        monkeypatch.setattr(backend, target, fake_full_pass)
        return

    def fake_slice(
        _wav: Path,
        start: float,
        end: float,
        *,
        _sample_geometry_observer=None,
        **_kwargs: Any,
    ) -> Path:
        geometry = qwen_sample_geometry(
            nominal_start=float(start),
            nominal_end=float(end),
            sample_rate=16_000,
            sample_count=32_000,
        )
        if _sample_geometry_observer is not None:
            _sample_geometry_observer(
                geometry.sample_start,
                geometry.sample_end,
                geometry.sample_rate,
                geometry.sample_count,
            )
        crop = tmp_path / "owned-qwen-crop.wav"
        crop.write_bytes(media_path.read_bytes())
        return crop

    monkeypatch.setattr(pipeline, "slice_wav", fake_slice)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [
            {"text": text, "start": 0.0, "end": 1.0},
        ],
    )


def _completed(trace: object) -> list[tuple[str, str]]:
    return [
        (event.phase, event.activity)
        for event in trace.events  # type: ignore[attr-defined]
        if event.state == "completed"
    ]


def _activity_count(
    trace: object,
    phase: str,
    activity: str,
    state: str,
) -> int:
    return sum(
        event.phase == phase
        and event.activity == activity
        and event.state == state
        for event in trace.events  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("route", ("ctc-full", "mms-full", "qwen-crop"))
def test_public_align_runtime_trace_records_real_route_and_ao10_before_ao11(
    route: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import pipeline
    from voxweave.align_runtime import capture_align_runtime_trace

    vtt_path, media_path = _write_public_align_episode(tmp_path, route=route)
    _stub_public_route(
        monkeypatch,
        tmp_path,
        route=route,
        media_path=media_path,
    )

    with capture_align_runtime_trace() as capture:
        assert (
            pipeline.align(vtt_path, media_path=media_path, separate=False) == vtt_path
        )
    trace = capture.snapshot()

    assert trace.route_kind == route
    assert trace.engine_family == "legacy-v1"
    record = trace.as_record()
    assert record["schema_version"] == 1
    assert [event["ordinal"] for event in record["events"]] == list(
        range(len(record["events"]))
    )
    completed = _completed(trace)
    ao10 = (
        ("AO-10", "group-block-spans"),
        ("AO-10", "common-all-empty-decision"),
        ("AO-10", "fill-insert-blocks"),
        ("AO-10", "enforce-min-duration"),
        ("AO-10", "rescue-tiny-cues"),
        ("AO-10", "clamp-spans"),
        ("AO-10", "seal-selected-legacy-result"),
    )
    assert tuple(row for row in completed if row[0] == "AO-10") == ao10
    assert completed.index(("AO-10", "seal-selected-legacy-result")) < completed.index(
        ("AO-11", "strict-capture")
    )


@pytest.mark.parametrize("route", ("ctc-full", "mms-full", "qwen-crop"))
def test_public_align_paired_helper_and_strict_failure_stops_before_strict_activity(
    route: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_acquisition, pipeline, realign
    from voxweave.align_runtime import capture_align_runtime_trace

    vtt_path, media_path = _write_public_align_episode(tmp_path, route=route)
    _stub_public_route(
        monkeypatch,
        tmp_path,
        route=route,
        media_path=media_path,
    )

    def fail_helper(*_args: Any, **_kwargs: Any) -> Any:
        raise _DisplayHelperFailure("injected display helper failure")

    def fail_strict(*_args: Any, **_kwargs: Any) -> Any:
        raise _StrictCaptureFailure("injected strict capture failure")

    monkeypatch.setattr(realign, "fill_insert_blocks", fail_helper)
    monkeypatch.setattr(align_acquisition, "capture_strict_units", fail_strict)

    with capture_align_runtime_trace() as capture:
        with pytest.raises(_DisplayHelperFailure, match="display helper failure"):
            pipeline.align(vtt_path, media_path=media_path, separate=False)
    trace = capture.snapshot()

    assert _activity_count(trace, "AO-10", "fill-insert-blocks", "started") == 1
    assert _activity_count(trace, "AO-10", "fill-insert-blocks", "failed") == 1
    assert _activity_count(trace, "AO-11", "strict-capture", "started") == 0
    assert _activity_count(trace, "AO-11", "strict-capture", "failed") == 0


@pytest.mark.parametrize("route", ("ctc-full", "mms-full", "qwen-crop"))
def test_public_align_strict_failure_runs_each_ao10_helper_exactly_once_first(
    route: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_acquisition, pipeline
    from voxweave.align_runtime import capture_align_runtime_trace

    vtt_path, media_path = _write_public_align_episode(tmp_path, route=route)
    _stub_public_route(
        monkeypatch,
        tmp_path,
        route=route,
        media_path=media_path,
    )

    def fail_strict(*_args: Any, **_kwargs: Any) -> Any:
        raise _StrictCaptureFailure("injected strict capture failure")

    monkeypatch.setattr(align_acquisition, "capture_strict_units", fail_strict)

    with capture_align_runtime_trace() as capture:
        with pytest.raises(_StrictCaptureFailure, match="strict capture failure"):
            pipeline.align(vtt_path, media_path=media_path, separate=False)
    trace = capture.snapshot()

    for activity in (
        "group-block-spans",
        "common-all-empty-decision",
        "fill-insert-blocks",
        "enforce-min-duration",
        "rescue-tiny-cues",
        "clamp-spans",
        "seal-selected-legacy-result",
    ):
        assert _activity_count(trace, "AO-10", activity, "completed") == 1
    assert _activity_count(trace, "AO-11", "strict-capture", "started") == 1
    assert _activity_count(trace, "AO-11", "strict-capture", "failed") == 1
