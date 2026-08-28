from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest


EXPECTED_VTT = ("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n你！\n").encode("utf-8")

EXPECTED_JSON = json.dumps(
    {
        "language": "zh",
        "segments": [{"text": "你！", "start": 0.0, "end": 1.0}],
        "word_segments": [
            {"text": "你", "start": 0.0, "end": 0.5},
            {"text": "！", "start": 0.5, "end": 1.0},
        ],
    },
    ensure_ascii=False,
    indent=2,
).encode("utf-8")


def _write_qwen_episode(tmp_path: Path) -> tuple[Path, Path, Path]:
    media_path = tmp_path / "episode.wav"
    with wave.open(str(media_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 32_000)
    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": [{"text": "你！", "start": 0.0, "end": 1.0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_bytes(EXPECTED_VTT)
    return media_path, json_path, vtt_path


def _stub_qwen_punctuation_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    monkeypatch.setattr(backend, "uses_mms", lambda _iso: False)
    monkeypatch.setattr(config, "align_model_for", lambda _iso: None)

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
        crop = tmp_path / "qwen-punctuation.wav"
        crop.write_bytes(media_path.read_bytes())
        return crop

    monkeypatch.setattr(pipeline, "slice_wav", fake_slice)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, _text, _iso: [
            {"text": "你", "start": 0.0, "end": 0.5},
            {"text": "！", "start": 0.5, "end": 1.0},
        ],
    )


def test_public_legacy_qwen_retains_separately_timed_punctuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_evidence, pipeline

    media_path, json_path, vtt_path = _write_qwen_episode(tmp_path)
    _stub_qwen_punctuation_result(monkeypatch, tmp_path, media_path)

    assert pipeline.align(vtt_path, media_path=media_path, separate=False) == vtt_path
    assert vtt_path.read_bytes() == EXPECTED_VTT
    assert json_path.read_bytes() == EXPECTED_JSON

    evidence = json.loads(
        (tmp_path / "episode.align-evidence.json").read_text(encoding="utf-8")
    )
    call = evidence["legacy_distribution"]["calls"][0]
    raw_ids = evidence["physical_calls"][0]["raw_unit_ids"]
    assert call["owner_source_indices"] == [0]
    assert call["expected_counts"] == [2]
    assert call["requested_ranges"] == [[0, 2]]
    assert call["realized_ranges"] == [[0, 2]]
    assert call["owner_unit_ids"] == [raw_ids]
    assert call["final_cursor"] == 2
    assert call["consumed_prefix_unit_ids"] == raw_ids
    assert call["shortage_source_indices"] == []
    assert call["leftover_unit_ids"] == []
    assert evidence["physical_calls"][0]["legacy_retained_units"] == [
        [
            {"text": "你", "start": 0.0, "end": 0.5},
            {"text": "！", "start": 0.5, "end": 1.0},
        ]
    ]
    verified = align_evidence.verify_align_evidence(
        vtt_path,
        explicit_media_path=media_path,
    )
    assert (verified.integrity, verified.w1_usable, verified.detail_code) == (
        True,
        True,
        None,
    )
