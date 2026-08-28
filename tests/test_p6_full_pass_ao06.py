from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class _FullPassPreparationFailure(RuntimeError):
    pass


def _phase_invoker(phase: str, activity: str) -> Callable[[Callable[[], Any]], Any]:
    from voxweave.align_runtime import align_runtime_activity

    def invoke(operation: Callable[[], Any]) -> Any:
        with align_runtime_activity(phase, activity):
            return operation()

    return invoke


def _failed(trace: object) -> list[tuple[str, str]]:
    return [
        (event.phase, event.activity)
        for event in trace.events  # type: ignore[attr-defined]
        if event.state == "failed"
    ]


@pytest.mark.parametrize("route", ("ctc", "mms"))
def test_full_pass_audio_load_failure_is_owned_by_ao06(
    route: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_ctc, align_mms
    from voxweave.align_runtime import capture_align_runtime_trace

    backend_calls = 0

    def fail_load(*_args: Any, **_kwargs: Any) -> Any:
        raise _FullPassPreparationFailure("injected full-pass audio load failure")

    def invoke_backend(operation: Callable[[], Any]) -> Any:
        nonlocal backend_calls
        backend_calls += 1
        return _phase_invoker("AO-07", "backend-call")(operation)

    wav_path = tmp_path / "prepared.wav"
    wav_path.write_bytes(b"unused")
    preparation_invoker = _phase_invoker("AO-06", "physical-call-preparation")
    if route == "ctc":
        monkeypatch.setattr(align_ctc, "_load_mono", fail_load)

        def run() -> Any:
            return align_ctc.align_blocks_full_ctc(
                wav_path,
                ["word"],
                "en",
                "synthetic-ctc",
                _preparation_invoker=preparation_invoker,
                _backend_invoker=invoke_backend,
            )

    else:
        monkeypatch.setattr(align_mms, "_read_wav_16k", fail_load)

        def run() -> Any:
            return align_mms.align_blocks_full_mms(
                wav_path,
                ["あ"],
                "ja",
                _preparation_invoker=preparation_invoker,
                _backend_invoker=invoke_backend,
            )

    with capture_align_runtime_trace() as capture:
        with pytest.raises(
            _FullPassPreparationFailure,
            match="full-pass audio load failure",
        ):
            run()
    trace = capture.snapshot()

    assert _failed(trace) == [("AO-06", "physical-call-preparation")]
    assert backend_calls == 0
    assert not [
        event
        for event in trace.events
        if event.state == "started" and int(event.phase.removeprefix("AO-")) >= 7
    ]


@pytest.mark.parametrize("route", ("ctc", "mms"))
def test_full_pass_dp_planner_failure_is_owned_by_ao06_before_backend(
    route: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_common, align_ctc, align_mms, chunking
    from voxweave.align_runtime import capture_align_runtime_trace

    sample_rate = 16_000
    waveform = np.zeros(40 * sample_rate, dtype=np.float32)
    bounds = ((0.0, 8.0), (10.0, 18.0), (22.0, 30.0), (32.0, 39.0))
    backend_calls = 0

    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1_250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    monkeypatch.setattr(
        chunking,
        "plan_dp_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _FullPassPreparationFailure("injected full-pass DP planner failure")
        ),
    )

    def invoke_backend(operation: Callable[[], Any]) -> Any:
        nonlocal backend_calls
        backend_calls += 1
        return _phase_invoker("AO-07", "backend-call")(operation)

    wav_path = tmp_path / "prepared.wav"
    wav_path.write_bytes(b"unused")
    preparation_invoker = _phase_invoker("AO-06", "physical-call-preparation")
    if route == "ctc":
        monkeypatch.setattr(align_ctc, "_load_mono", lambda *_args: waveform)

        def run() -> Any:
            return align_ctc.align_blocks_full_ctc(
                wav_path,
                ["A", "B", "C", "D"],
                "en",
                "synthetic-ctc",
                bounds=bounds,
                _preparation_invoker=preparation_invoker,
                _backend_invoker=invoke_backend,
            )

    else:
        monkeypatch.setattr(align_mms, "_read_wav_16k", lambda *_args: waveform)

        def run() -> Any:
            return align_mms.align_blocks_full_mms(
                wav_path,
                ["A", "B", "C", "D"],
                "ja",
                bounds=bounds,
                _preparation_invoker=preparation_invoker,
                _backend_invoker=invoke_backend,
            )

    with capture_align_runtime_trace() as capture:
        with pytest.raises(
            _FullPassPreparationFailure,
            match="full-pass DP planner failure",
        ):
            run()
    trace = capture.snapshot()

    assert _failed(trace) == [("AO-06", "physical-call-preparation")]
    assert backend_calls == 0
    assert not [
        event
        for event in trace.events
        if event.state == "started" and int(event.phase.removeprefix("AO-")) >= 7
    ]
