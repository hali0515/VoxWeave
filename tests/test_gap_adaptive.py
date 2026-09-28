# tests/test_gap_adaptive.py
# Adaptive gap thresholds (experimental, env-gated): clause_ms scales to the
# file's inter-unit gap distribution; offline follows at the same ratio.
import pytest

from voxweave import segmentation
from voxweave.core.gap_split import adaptive_clause_ms


def test_fast_talker_clamps_low():
    # dense dialogue: p90 articulation gap ~120ms -> clamped to the 300 floor
    gaps = [100.0] * 90 + [150.0] * 10
    assert adaptive_clause_ms(gaps) == 300


def test_slow_narration_raises_threshold():
    # slow narration: p90 ~600ms -> clause ~690 (margin 1.15), under the 800 cap
    gaps = [300.0] * 50 + [600.0] * 50
    got = adaptive_clause_ms(gaps)
    assert got is not None and 600 < got <= 800


def test_too_few_samples_returns_none():
    assert adaptive_clause_ms([100.0] * 10) is None


def test_out_of_band_gaps_ignored():
    # >=2s gaps are real pauses, not articulation; they must not inflate p90
    gaps = [100.0] * 60 + [5000.0] * 60
    assert adaptive_clause_ms(gaps) == 300


def _units(n, gap_s):
    out = []
    t = 0.0
    for _ in range(n):
        out.append({"text": "x", "start": t, "end": t + 0.2})
        t += 0.2 + gap_s
    return out


def test_pipeline_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_GAP_ADAPTIVE", raising=False)
    th = {"clause_ms": 400, "offline_ms": 700}
    assert segmentation._maybe_adaptive_thresholds(th, _units(100, 0.6)) is th


def test_pipeline_gate_on_rescales(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_GAP_ADAPTIVE", "1")
    th = {"clause_ms": 400, "vad_skip_ms": 1000, "offline_ms": 700}
    out = segmentation._maybe_adaptive_thresholds(th, _units(100, 0.6))
    assert out["clause_ms"] == pytest.approx(round(600 * 1.15), abs=2)
    # offline keeps the static clause:offline ratio
    assert out["offline_ms"] == pytest.approx(out["clause_ms"] * 1.75, abs=2)
    assert out["vad_skip_ms"] == 1000  # untouched


def test_pipeline_gate_on_but_sparse_keeps_static(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_GAP_ADAPTIVE", "1")
    th = {"clause_ms": 400, "offline_ms": 700}
    assert segmentation._maybe_adaptive_thresholds(th, _units(10, 0.6)) is th
