"""Separator download hints and temp-file hygiene (no real model loading)."""

import numpy as np
import pytest

from voxweave import backend, runtime


@pytest.fixture
def _missing_separator(monkeypatch):
    monkeypatch.setattr(backend, "SEPARATOR_CKPT", "/nonexistent/x.ckpt")
    monkeypatch.setattr(backend, "SEPARATOR_CONFIG", "/nonexistent/x.yaml")


def test_separator_download_failure_through_hf_download_keeps_the_hint(
    monkeypatch, _missing_separator
):
    # runtime._hf_download turns every download failure into a RuntimeError; the
    # separator's own ways out (--no-separate / manual weights) must survive that
    def _dl(repo, filename, cache_dir=None):
        try:
            raise OSError("no network")
        except OSError as e:
            raise runtime._hf_error(repo, e) from e

    monkeypatch.setattr(backend, "_hf_download", _dl)
    with pytest.raises(RuntimeError) as failure:
        backend._resolve_separator_files()
    msg = str(failure.value)
    assert "HF_TOKEN" in msg  # runtime's own hint kept
    assert "--no-separate" in msg and "/nonexistent/x.ckpt" in msg
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert isinstance(failure.value.__cause__.__cause__, OSError)


def test_separator_missing_dependency_error_is_reraised_as_is(
    monkeypatch, _missing_separator
):
    missing = runtime._require("huggingface_hub")

    def _dl(repo, filename, cache_dir=None):
        try:
            raise ModuleNotFoundError("No module named 'huggingface_hub'")
        except ModuleNotFoundError as e:
            raise missing from e

    monkeypatch.setattr(backend, "_hf_download", _dl)
    with pytest.raises(RuntimeError) as failure:
        backend._resolve_separator_files()
    assert failure.value is missing


def test_separate_vocals_removes_temp_flac_when_the_write_fails(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import soundfile as sf

    made: list = []

    def fake_mkstemp(suffix="", prefix="", dir=None):
        import os

        path = tmp_path / f"{prefix}fake{suffix}"
        made.append(path)
        return os.open(str(path), os.O_CREAT | os.O_RDWR), str(path)

    def failing_write(*_a, **_k):
        raise OSError("No space left on device")

    monkeypatch.setattr(
        sf, "read", lambda *a, **k: (np.zeros((64, 2), dtype=np.float32), 44100)
    )
    monkeypatch.setattr(sf, "write", failing_write)
    monkeypatch.setattr(backend.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(backend.config, "conf_separate_autocast", lambda: "off")
    monkeypatch.setattr(
        backend, "_load_separator", lambda autocast=None: (object(), {}, {})
    )
    monkeypatch.setattr(backend, "_demix", lambda *a, **k: torch.zeros(2, 64))
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)

    with pytest.raises(OSError, match="No space left"):
        backend.separate_vocals(tmp_path / "in.wav")
    assert made and not any(p.exists() for p in made)
