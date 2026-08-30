# tests/test_diarize_compat.py
# pyannote.audio 4.x compatibility with no network / GPU / real pyannote.
# Covers retirement of the 3.4-era torchaudio shim and diarize_turns feeding a
# decoded waveform dictionary (bypassing torchcodec) with defensive stereo downmix.
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from voxweave import config, diarize


@pytest.fixture(autouse=True)
def _isolated_model_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.delenv("VOXWEAVE_DIARIZE_MODEL", raising=False)


class _FakeSeg:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, tracks) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        for seg, name, label in self._tracks:
            yield (seg, name, label) if yield_label else (seg, name)

    def labels(self):
        return list(dict.fromkeys(label for _seg, _name, label in self._tracks))


class _CapturePipeline:
    """Callable pipeline stub recording the exact input pyannote is handed."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, file, **kwargs):
        self.calls.append((file, kwargs))
        return _FakeAnnotation([(_FakeSeg(0.0, 1.0), "A", "SPEAKER_00")])


def test_pyannote34_torchaudio_shim_is_retired():
    assert not hasattr(diarize, "_ensure_torchaudio_compat")


# --- diarize_turns waveform-dict input --------------------------------------


def test_diarize_turns_feeds_waveform_dict(monkeypatch, tmp_path):
    wav_path = tmp_path / "clip.wav"
    sig = (np.random.randn(16000) * 0.01).astype("float32")
    sf.write(str(wav_path), sig, 16000, subtype="FLOAT")

    fake = _CapturePipeline()
    monkeypatch.setattr(diarize, "_get_pipeline", lambda _token, _model: fake)

    result = diarize.diarize_turns(wav_path, token="hf_test")
    assert result.turns == [(0.0, 1.0, "SPEAKER_00")]

    file_arg, _ = fake.calls[0]
    assert isinstance(file_arg, dict)
    assert set(file_arg) >= {"waveform", "sample_rate"}
    wf = file_arg["waveform"]
    assert isinstance(wf, torch.Tensor)
    assert wf.dtype == torch.float32
    assert wf.ndim == 2
    assert wf.shape[0] == 1
    assert wf.shape[1] == 16000
    assert file_arg["sample_rate"] == 16000


def test_diarize_turns_downmixes_stereo(monkeypatch, tmp_path):
    wav_path = tmp_path / "stereo.wav"
    left = np.full(8000, 0.2, dtype="float32")
    right = np.full(8000, 0.4, dtype="float32")
    sf.write(str(wav_path), np.stack([left, right], axis=1), 16000, subtype="FLOAT")

    fake = _CapturePipeline()
    monkeypatch.setattr(diarize, "_get_pipeline", lambda _token, _model: fake)

    diarize.diarize_turns(wav_path, token="hf_test")
    wf = fake.calls[0][0]["waveform"]
    assert wf.shape == (1, 8000)
    assert torch.allclose(wf, torch.full((1, 8000), 0.3), atol=1e-6)


def test_diarize_turns_no_token_mentions_hf_auth_login(monkeypatch):
    for key in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "conf_hf_token", lambda: None)

    def _boom(_token, _model):
        raise AssertionError("pipeline must not load without a token")

    monkeypatch.setattr(diarize, "_get_pipeline", _boom)
    with pytest.raises(RuntimeError) as ei:
        diarize.diarize_turns(Path("nope.wav"), model="3.1")
    assert "hf auth login" in str(ei.value)
