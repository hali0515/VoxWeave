"""Regressions for align route handling, env float knobs, and sibling snapshots."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _write_public_align_episode(
    tmp_path: Path, *, route: str = "qwen-crop"
) -> tuple[Path, Path]:
    vtt_path = tmp_path / "episode.vtt"
    media_path = tmp_path / "episode.wav"
    media_path.write_bytes(b"synthetic-media")
    if route == "qwen-crop":
        vtt_path.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n你\n", encoding="utf-8"
        )
        (tmp_path / "episode.json").write_text(
            json.dumps(
                {
                    "language": "zh",
                    "word_segments": [{"text": "你", "start": 0.0, "end": 1.0}],
                }
            ),
            encoding="utf-8",
        )
    else:
        vtt_path.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8"
        )
        (tmp_path / "episode.json").write_text(
            json.dumps(
                {
                    "language": "en",
                    "word_segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
                }
            ),
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

    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media_path
    )
    if route == "qwen-crop":
        monkeypatch.setattr(backend, "uses_mms", lambda _iso: False)
        monkeypatch.setattr(config, "align_model_for", lambda _iso: None)

        def fake_slice(_wav: Path, _cs: float, _ce: float, **_kw: Any) -> Path:
            out = tmp_path / "crop.wav"
            out.write_bytes(b"crop")
            return out

        monkeypatch.setattr(pipeline, "slice_wav", fake_slice)
    else:
        monkeypatch.setattr(backend, "uses_mms", lambda _iso: False)
        monkeypatch.setattr(config, "align_model_for", lambda _iso: "synthetic-ctc")
        monkeypatch.setattr(
            backend,
            "align_blocks_full_ctc",
            lambda *_a, **_kw: [[{"text": "hello", "start": 0.1, "end": 0.9}]],
        )


def test_reversed_aligner_unit_does_not_abort_align(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit the aligner returns with end < start is legacy data, not a fatal error."""
    from voxweave import backend, pipeline

    vtt_path, media_path = _write_public_align_episode(tmp_path, route="qwen-crop")
    _stub_public_route(monkeypatch, tmp_path, route="qwen-crop", media_path=media_path)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [{"text": text, "start": 0.8, "end": 0.2}],
    )

    assert pipeline.align(vtt_path, media_path=media_path, separate=False) == vtt_path
    assert "你" in vtt_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (None, 30.0),
        ("12.5", 12.5),
        ("abc", 30.0),
        ("0", 30.0),
        ("-3", 30.0),
        ("nan", 30.0),
        ("inf", 30.0),
    ),
)
def test_env_float_knobs_fall_back_instead_of_breaking(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    from voxweave.align_common import _env_float_in_range

    if raw is None:
        monkeypatch.delenv("VOXWEAVE_TEST_WINDOW_S", raising=False)
    else:
        monkeypatch.setenv("VOXWEAVE_TEST_WINDOW_S", raw)
    assert _env_float_in_range("VOXWEAVE_TEST_WINDOW_S", 30.0, low=1.0) == expected


def test_env_fraction_knob_rejects_zero_and_values_above_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave.align_common import _env_float_in_range

    for raw, expected in (("0", 0.8), ("1.5", 0.8), ("1", 1.0), ("0.5", 0.5)):
        monkeypatch.setenv("VOXWEAVE_TEST_FRAC", raw)
        assert (
            _env_float_in_range(
                "VOXWEAVE_TEST_FRAC", 0.8, low=0.0, high=1.0, low_inclusive=False
            )
            == expected
        )


def test_ctc_transcribe_path_refuses_unsafe_plan_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CTC pass plans and checks before model work."""
    from voxweave import align_common, align_ctc, chunking
    from voxweave.align_dp_safety import DpRouteHintsInvalid

    waveform = np.zeros(40 * align_ctc.CTC_AUDIO_SR, dtype=np.float32)
    monkeypatch.setattr(align_ctc, "_load_mono", lambda *_args, **_kwargs: waveform)
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(
        chunking,
        "plan_dp_chunks",
        lambda *_args, **_kwargs: [
            {"lo": 0, "hi": 1, "start": 0.0, "end": 18.0},
            {"lo": 1, "hi": 2, "start": 17.0, "end": 35.0},
        ],
    )
    monkeypatch.setattr(
        align_ctc,
        "_get_ctc_aligner",
        lambda *_args, **_kwargs: pytest.fail("unsafe plan reached model loading"),
    )
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with pytest.raises(DpRouteHintsInvalid) as caught:
        align_ctc.align_blocks_full_ctc(
            wav_path,
            ["A", "B"],
            "en",
            "synthetic",
            bounds=((0.0, 8.0), (22.0, 39.0)),
        )
    assert caught.value.failure.detail_code == "crop-geometry"


def test_ctc_transcribe_path_refuses_a_non_16k_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from voxweave import align_ctc

    waveform = np.zeros(align_ctc.CTC_AUDIO_SR, dtype=np.float32)
    monkeypatch.setattr(align_ctc, "_load_mono", lambda *_args, **_kwargs: waveform)
    monkeypatch.setattr(
        align_ctc, "_get_ctc_aligner", lambda *_args: SimpleNamespace(sr=8000)
    )
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with pytest.raises(RuntimeError, match="sample rate differs"):
        align_ctc.align_blocks_full_ctc(wav_path, ["A"], "en", "synthetic")


def test_vad_mask_on_mms_alignment_is_reported_as_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from voxweave import align_mms

    waveform = np.zeros(align_mms.MMS_SR, dtype=np.float32)
    monkeypatch.setattr(align_mms, "_read_wav_16k", lambda _path: waveform)
    monkeypatch.setattr(align_mms, "_mms_emit_units", lambda *_args: [])
    monkeypatch.setattr(align_mms, "_empty_cache", lambda: None)
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        align_mms.align_blocks_full_mms(wav_path, ["あ"], "ja")
    assert any("no effect on MMS" in record.message for record in caplog.records)

    caplog.clear()
    monkeypatch.delenv("VOXWEAVE_VAD_EMISSION_MASK")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        align_mms.align_blocks_full_mms(wav_path, ["あ"], "ja")
    assert not any("no effect on MMS" in record.message for record in caplog.records)


def test_vad_mask_on_the_qwen_align_route_is_reported_as_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from voxweave import backend, pipeline

    vtt_path, media_path = _write_public_align_episode(tmp_path, route="qwen-crop")
    _stub_public_route(monkeypatch, tmp_path, route="qwen-crop", media_path=media_path)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [{"text": text, "start": 0.2, "end": 0.8}],
    )
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    warnings = [r for r in caplog.records if "no effect on Qwen" in r.message]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("mask", "language", "warnings"),
    (("1", "Chinese", 1), ("1", "English", 0), ("", "Chinese", 0)),
)
def test_transcribe_reports_vad_mask_ignored_once_per_qwen_aligned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mask: str,
    language: str,
    warnings: int,
) -> None:
    from voxweave import backend

    monkeypatch.setattr(backend, "_use_mlx", lambda: False)
    monkeypatch.setattr(
        backend, "_asr_only", lambda *_args: (language, "hello there", language)
    )
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _lang: [{"text": text, "start": 0.0, "end": 0.5}],
    )
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", mask)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for wav in wavs:
        wav.write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        backend.transcribe_chunks(wavs, None, asr_model="qwen3-asr-1.7b")
    found = [r for r in caplog.records if "no effect on Qwen" in r.message]
    assert len(found) == warnings


def test_public_align_reports_a_legacy_render_failure_with_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxweave import pipeline, publication, realign

    vtt_path, media_path = _write_public_align_episode(tmp_path, route="ctc-full")
    _stub_public_route(monkeypatch, tmp_path, route="ctc-full", media_path=media_path)
    before = vtt_path.read_bytes()

    def fail_render(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("render target vanished")

    monkeypatch.setattr(realign, "render_cues", fail_render)
    with pytest.raises(
        publication.PublicationEncodeError, match="render target vanished"
    ):
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert vtt_path.read_bytes() == before


def test_sibling_snapshot_digest_is_computed_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_snapshot

    calls: list[object] = []
    original = align_snapshot._sibling_digest

    def counting(**kwargs: Any) -> str:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(align_snapshot, "_sibling_digest", counting)
    snapshot = align_snapshot.decode_sibling_json_snapshot(
        "episode.json", b'{"word_segments": []}'
    )
    assert calls == []
    assert snapshot.digest == snapshot.digest
    assert len(calls) == 2


def test_frozen_json_primitives_do_not_import_the_subtitle_stack() -> None:
    probe = (
        "import sys, voxweave.align_snapshot; "
        "print(sorted(m for m in ('voxweave.realign', 'voxweave.subformats',"
        " 'voxweave.speakers') if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "[]"
