"""Contract tests for ``scripts/calibrate_voiceprints.py`` (no models, no audio I/O)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    """Import a module from ``scripts/`` by path (it is not an installed package)."""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calib = _load_script("calibrate_voiceprints")


def test_error_rates_for_perfectly_separated_scores():
    rates = calib.error_rates([0.8, 0.9, 0.7], [0.1, 0.2, 0.3])

    assert rates["eer"] == pytest.approx(0.0)
    assert 0.3 < rates["eer_threshold"] <= 0.7
    assert rates["far1_threshold"] == pytest.approx(0.7)
    assert rates["frr_at_far1"] == pytest.approx(0.0)


def test_error_rates_interpolate_the_crossing():
    # Accept = score >= t. At t=0.5: FAR 1/2, FRR 1/2 exactly.
    rates = calib.error_rates([0.4, 0.6], [0.3, 0.5])

    assert rates["eer"] == pytest.approx(0.5)
    assert rates["far1_threshold"] == pytest.approx(0.6)
    assert rates["frr_at_far1"] == pytest.approx(0.5)


def test_error_rates_without_both_populations_are_undefined():
    assert calib.error_rates([], [0.1])["eer"] is None
    assert calib.error_rates([0.9], [])["far1_threshold"] is None


def test_calibration_segments_trim_overlaps_and_keep_time_order():
    turns = [
        (0.0, 3.0, "A"),
        (2.5, 6.0, "B"),
        (6.0, 9.0, "A"),
        (9.0, 9.5, "A"),
    ]

    # A's first turn loses 2.5-3.0 to B; the 0.5 s tail turn is too short.
    assert calib.calibration_segments(turns, "A", max_segments=40) == [
        (0.0, 2.5),
        (6.0, 9.0),
    ]
    assert calib.calibration_segments(turns, "A", max_segments=1) == [(6.0, 9.0)]
    assert calib.calibration_segments(turns, "B", max_segments=40) == [(3.0, 6.0)]


def test_episode_scores_split_even_and_odd_segments():
    spans = [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0), (9.0, 11.0)]
    a_vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    b_vectors = np.array([[1.0, 0.0]], dtype=np.float64)

    targets, non_targets = calib.episode_scores(
        {"A": (spans, a_vectors), "B": ([(0.0, 2.0)], b_vectors)}
    )

    # even = segments 0, 2 -> [1, 0]; odd = segments 1, 3 -> [0, 1].
    assert targets == [{"speaker": "A", "score": pytest.approx(0.0), "segments": 4}]
    assert non_targets == [{"speakers": ["A", "B"], "score": pytest.approx(2**-0.5)}]


def test_calibrate_reports_every_model_and_releases_it():
    episodes = [
        calib.Episode(
            media=Path("ep.mkv"),
            turns=[
                (0.0, 2.0, "A"),
                (2.0, 4.0, "B"),
                (4.0, 6.0, "A"),
                (6.0, 8.0, "B"),
            ],
            audio_source=Path("ep.mkv"),
            separated=False,
        )
    ]
    waveform = np.zeros(8 * 16000, dtype=np.float32)
    released = []

    def embed(_waveform, spans):
        # Speaker A lives on axis 0, speaker B on axis 1.
        return np.array(
            [[1.0, 0.0] if start in (0.0, 4.0) else [0.0, 1.0] for start, _end in spans]
        )

    embedder = calib.Embedder(
        name="fake", embed=embed, release=lambda: released.append(True)
    )

    models = calib.calibrate(episodes, [waveform], [embedder], max_segments=40)

    result = models["fake"]
    assert result["target"]["count"] == 2
    assert result["target"]["mean"] == pytest.approx(1.0)
    assert result["non_target"]["count"] == 1
    assert result["non_target"]["mean"] == pytest.approx(0.0)
    assert result["eer"] == pytest.approx(0.0)
    assert released == [True]


def test_main_writes_the_json_report(tmp_path, monkeypatch, capsys):
    media = tmp_path / "ep.mkv"
    episode = calib.Episode(
        media=media,
        turns=[(0.0, 2.0, "A"), (2.0, 4.0, "B"), (4.0, 6.0, "A")],
        audio_source=media,
        separated=False,
    )
    monkeypatch.setattr(
        calib,
        "build_embedder",
        lambda name: calib.Embedder(
            name=name,
            embed=lambda _w, spans: np.ones((len(spans), 2)),
            release=lambda: None,
        ),
    )
    monkeypatch.setattr(calib, "load_episode", lambda _media, **_kw: episode)
    monkeypatch.setattr(
        calib, "decode_episode", lambda _episode, **_kw: np.zeros(96_000)
    )
    out = tmp_path / "report.json"

    code = calib.main([str(media), "--models", "redimnet2", "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["recipe"] == "centroid-v1"
    assert set(report["models"]) == {"redimnet2"}
    captured = capsys.readouterr()
    assert "no separated-vocals cache" in captured.err
    assert "redimnet2" in captured.out


def test_unknown_model_is_an_input_error(tmp_path, capsys):
    code = calib.main([str(tmp_path / "ep.mkv"), "--models", "ecapa"])

    assert code == 2
    assert "unknown voiceprint model" in capsys.readouterr().err
