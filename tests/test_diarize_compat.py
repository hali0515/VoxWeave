# tests/test_diarize_compat.py
# torch/torchaudio 2.11 compatibility for pyannote.audio 3.4 diarization, with no
# network / GPU / real pyannote. Covers: the torchaudio symbol shims
# (AudioMetaData / info / list_audio_backends) applied only when missing,
# idempotent, and never clobbering an older torchaudio that still ships them; and
# diarize_turns feeding pyannote a {"waveform", "sample_rate"} dict (bypassing
# pyannote's runtime torchaudio I/O) with defensive stereo downmix.
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from voxweave import config, diarize


def _fake_torchaudio(**attrs) -> types.ModuleType:
    """A stand-in torchaudio module (``load`` present, like 2.11) plus overrides."""
    m = types.ModuleType("torchaudio")
    m.load = lambda *a, **k: None  # torchaudio 2.11 still ships load
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


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


class _CapturePipeline:
    """Callable pipeline stub recording the exact input pyannote is handed."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, file, **kwargs):
        self.calls.append((file, kwargs))
        return _FakeAnnotation([(_FakeSeg(0.0, 1.0), "A", "SPEAKER_00")])


# --- torchaudio shims -------------------------------------------------------


def test_ensure_compat_adds_missing_symbols(monkeypatch):
    fake = _fake_torchaudio()
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    diarize._ensure_torchaudio_compat()
    assert hasattr(fake, "AudioMetaData")
    assert hasattr(fake, "info")
    assert hasattr(fake, "list_audio_backends")
    assert fake.list_audio_backends() == ["soundfile"]
    meta = fake.AudioMetaData(sample_rate=16000, num_frames=32000, num_channels=1)
    assert meta.sample_rate == 16000
    assert meta.num_frames == 32000
    assert meta.num_channels == 1
    assert meta.bits_per_sample == 0
    assert meta.encoding == ""


def test_ensure_compat_info_reads_via_soundfile(monkeypatch):
    fake_ta = _fake_torchaudio()
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)

    class _Info:
        samplerate = 22050
        frames = 44100
        channels = 2

    fake_sf = types.ModuleType("soundfile")
    fake_sf.info = lambda path: _Info()
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    diarize._ensure_torchaudio_compat()
    meta = fake_ta.info("whatever.wav")
    assert (meta.sample_rate, meta.num_frames, meta.num_channels) == (22050, 44100, 2)


def test_ensure_compat_is_idempotent(monkeypatch):
    fake = _fake_torchaudio()
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    diarize._ensure_torchaudio_compat()
    amd, info, backends = fake.AudioMetaData, fake.info, fake.list_audio_backends
    diarize._ensure_torchaudio_compat()  # second call must not rebuild the shims
    assert fake.AudioMetaData is amd
    assert fake.info is info
    assert fake.list_audio_backends is backends


def test_ensure_compat_leaves_existing_symbols(monkeypatch):
    sentinel_amd = object()

    def sentinel_info(*a, **k):
        return "real"

    def sentinel_backends():
        return ["ffmpeg", "soundfile"]

    fake = _fake_torchaudio(
        AudioMetaData=sentinel_amd,
        info=sentinel_info,
        list_audio_backends=sentinel_backends,
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    diarize._ensure_torchaudio_compat()
    assert fake.AudioMetaData is sentinel_amd
    assert fake.info is sentinel_info
    assert fake.list_audio_backends is sentinel_backends


# --- diarize_turns waveform-dict input --------------------------------------


def test_diarize_turns_feeds_waveform_dict(monkeypatch, tmp_path):
    wav_path = tmp_path / "clip.wav"
    sig = (np.random.randn(16000) * 0.01).astype("float32")
    sf.write(str(wav_path), sig, 16000, subtype="FLOAT")

    fake = _CapturePipeline()
    monkeypatch.setattr(diarize, "_get_pipeline", lambda token: fake)

    turns = diarize.diarize_turns(wav_path, token="hf_test")
    assert turns == [(0.0, 1.0, "SPEAKER_00")]

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
    monkeypatch.setattr(diarize, "_get_pipeline", lambda token: fake)

    diarize.diarize_turns(wav_path, token="hf_test")
    wf = fake.calls[0][0]["waveform"]
    assert wf.shape == (1, 8000)
    assert torch.allclose(wf, torch.full((1, 8000), 0.3), atol=1e-6)


def test_diarize_turns_no_token_mentions_hf_auth_login(monkeypatch):
    for key in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "conf_hf_token", lambda: None)

    def _boom(token):
        raise AssertionError("pipeline must not load without a token")

    monkeypatch.setattr(diarize, "_get_pipeline", _boom)
    with pytest.raises(RuntimeError) as ei:
        diarize.diarize_turns(Path("nope.wav"))
    assert "hf auth login" in str(ei.value)
