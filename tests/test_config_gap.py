# tests/test_config_gap.py
import importlib
from voxweave import config


def _reload():
    return importlib.reload(config)


def test_defaults_en(monkeypatch):
    for k in (
        "VOXWEAVE_GAP_CLAUSE_MS",
        "VOXWEAVE_GAP_VAD_SKIP_MS",
        "VOXWEAVE_GAP_OFFLINE_MS",
        "VOXWEAVE_SEG_MIN_CUE_SEC",
        "VOXWEAVE_MAX_CUE_SEC",
    ):
        monkeypatch.delenv(k, raising=False)
    c = _reload().gap_thresholds("en")
    assert c["clause_ms"] == 400 and c["vad_skip_ms"] == 1000
    assert c["offline_ms"] == 700 and c["min_cue_s"] == 0.5 and c["max_cue_s"] == 7.0


def test_ja_multiplier(monkeypatch):
    for k in ("VOXWEAVE_GAP_CLAUSE_MS", "VOXWEAVE_GAP_OFFLINE_MS"):
        monkeypatch.delenv(k, raising=False)
    c = _reload().gap_thresholds("ja")
    assert c["clause_ms"] == 560 and c["offline_ms"] == 980  # x1.4


def test_yue_uses_chinese_reading_speed(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_CPS", raising=False)
    assert _reload().gap_thresholds("yue")["cps"] == 9.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_GAP_CLAUSE_MS", "300")
    monkeypatch.setenv("VOXWEAVE_SEG_MIN_CUE_SEC", "0")
    c = _reload().gap_thresholds("en")
    assert c["clause_ms"] == 300 and c["min_cue_s"] == 0.0


def test_min_cue_clamped_to_5_6(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_SEG_MIN_CUE_SEC", "2.0")
    c = _reload().gap_thresholds("en")
    assert abs(c["min_cue_s"] - 5 / 6) < 1e-9  # clamp ceiling
