"""Audio I/O and model-residency contracts for chunking/songdet/separator loading.

Three separate costs are pinned here:
- slice_wav must not decode the whole file to hand back a few seconds of it, while
  staying sample-identical to the full-read-then-slice reference.
- the silero VAD model must be loaded once per process, with an explicit release hook.
- PANNs must have the same explicit release hook so the pipeline can free its VRAM.
Plus the separator's TF32 opt-out, which is pure flag plumbing and needs no real model.
"""

import math
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from voxweave import backend, chunking, songdet

SR = 16000


def _write_wav(path, seconds: float, sr: int = SR):
    """Deterministic, non-repeating mono signal: any slice offset error is visible."""
    n = round(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = (
        0.5 * np.sin(2.0 * math.pi * 440.0 * t)
        + 0.25 * np.sin(2.0 * math.pi * 97.0 * t)
        + 0.1 * np.sin(2.0 * math.pi * 3.0 * t)
    ).astype(np.float32)
    sf.write(str(path), sig, sr)
    return path


def _reference_slice(wav_path, start: float, end: float):
    """Full-read-then-slice: the behaviour slice_wav must reproduce exactly."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    a = max(0, int(start * sr))
    b = min(len(data), int(end * sr))
    return data[a:b]


# --- slice_wav: seek-based read, sample-identical output ---


@pytest.mark.parametrize(
    "start,end",
    [
        (0.0, 1.0),
        (0.5, 2.25),
        (1.0, 3.0),
        (2.9, 5.0),  # end past EOF clamps to the last frame
        (0.0, 0.0),  # degenerate window
        (1.23456, 2.34567),  # non-frame-aligned boundaries
    ],
)
def test_slice_wav_is_sample_identical_to_full_read(tmp_path, start, end):
    src = _write_wav(tmp_path / "src.wav", 3.0)
    ref = _reference_slice(src, start, end)

    out = chunking.slice_wav(src, start, end)
    got, sr = sf.read(str(out), dtype="float32")
    out.unlink(missing_ok=True)

    assert sr == SR
    assert got.dtype == np.float32
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_slice_wav_does_not_read_the_whole_file(tmp_path, monkeypatch):
    # Contract: slice_wav seeks to the requested frame range. A full sf.read of a
    # 2-hour separated wav costs ~7 GB of RAM per chunk and is the reason this exists.
    src = _write_wav(tmp_path / "src.wav", 3.0)
    ref = _reference_slice(src, 1.0, 2.0)

    def _no_full_read(*a, **kw):
        raise AssertionError("slice_wav must not full-read the source wav")

    monkeypatch.setattr(chunking.sf, "read", _no_full_read)
    out = chunking.slice_wav(src, 1.0, 2.0)
    monkeypatch.undo()

    got, _ = sf.read(str(out), dtype="float32")
    out.unlink(missing_ok=True)
    assert np.array_equal(got, ref)


# --- silero VAD singleton: loaded once, released on demand ---


@pytest.fixture
def _released_silero():
    """Drop any cached silero model around the test so ordering cannot leak state."""
    release = getattr(chunking, "release_silero_vad", None)
    if release:
        release()
    yield
    release = getattr(chunking, "release_silero_vad", None)
    if release:
        release()


def _fake_silero(monkeypatch):
    """Inject a silero_vad stand-in; returns the list of load calls."""
    loads: list[object] = []

    def _load_silero_vad(*a, **kw):
        model = object()
        loads.append(model)
        return model

    def _get_speech_timestamps(wav, model, **kw):
        assert model in loads
        return [{"start": 0.0, "end": 0.5}]

    mod = types.ModuleType("silero_vad")
    mod.load_silero_vad = _load_silero_vad  # type: ignore[attr-defined]
    mod.get_speech_timestamps = _get_speech_timestamps  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", mod)
    return loads


def test_vad_reuses_one_silero_model_across_calls(
    tmp_path, monkeypatch, _released_silero
):
    loads = _fake_silero(monkeypatch)
    wav = _write_wav(tmp_path / "a.wav", 0.5)

    first = chunking.vad_speech_segments(wav)
    second = chunking.vad_speech_segments(wav)

    assert first == second == [{"start": 0.0, "end": 0.5}]
    assert len(loads) == 1, "silero must be loaded once per process, not per call"


def test_release_silero_vad_forces_a_reload(tmp_path, monkeypatch, _released_silero):
    loads = _fake_silero(monkeypatch)
    wav = _write_wav(tmp_path / "a.wav", 0.5)

    chunking.vad_speech_segments(wav)
    chunking.vad_speech_segments(wav)
    assert len(loads) == 1

    chunking.release_silero_vad()
    chunking.vad_speech_segments(wav)
    assert len(loads) == 2


def test_release_silero_vad_is_safe_when_nothing_is_loaded(_released_silero):
    chunking.release_silero_vad()
    chunking.release_silero_vad()


# --- PANNs singleton: explicit release so the pipeline can reclaim VRAM ---


def _fake_panns(monkeypatch, *, cuda: bool):
    """Inject panns_inference + torch stand-ins; returns (instances, empty_cache_calls)."""
    instances: list[object] = []
    empties: list[int] = []

    class _AudioTagging:
        def __init__(self, checkpoint_path=None, device=None):
            self.checkpoint_path = checkpoint_path
            self.device = device
            instances.append(self)

    panns = types.ModuleType("panns_inference")
    panns.AudioTagging = _AudioTagging  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "panns_inference", panns)

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda,
        empty_cache=lambda: empties.append(1),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    monkeypatch.setattr(songdet, "_ensure_panns_labels", lambda: None)
    monkeypatch.setattr(songdet, "_resolve_panns_ckpt", lambda: "/fake/Cnn14.pth")
    monkeypatch.setattr(songdet, "_model", None)
    return instances, empties


def test_songdet_release_model_clears_singleton_and_forces_reload(monkeypatch):
    instances, _ = _fake_panns(monkeypatch, cuda=False)

    first = songdet._get_model()
    assert songdet._get_model() is first
    assert len(instances) == 1

    songdet.release_model()
    assert songdet._model is None

    second = songdet._get_model()
    assert second is not first
    assert len(instances) == 2


def test_songdet_release_model_empties_cuda_cache(monkeypatch):
    _, empties = _fake_panns(monkeypatch, cuda=True)
    songdet._get_model()
    songdet.release_model()
    assert empties, "release_model must free the CUDA allocator cache when on cuda"


def test_songdet_release_model_is_safe_when_nothing_is_loaded(monkeypatch):
    _fake_panns(monkeypatch, cuda=False)
    songdet.release_model()
    songdet.release_model()
    assert songdet._model is None


# --- separator TF32: on by default on cuda, opt-out via VOXWEAVE_TF32 ---


def _fake_separator_env(monkeypatch, tmp_path, *, device: str):
    """Patch _load_separator's model/runtime dependencies around real temp files."""
    precisions: list[str] = []

    torch_stub = types.ModuleType("torch")
    torch_stub.load = lambda *a, **kw: {}  # type: ignore[attr-defined]
    torch_stub.set_float32_matmul_precision = precisions.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    class _FakeRoformer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, sd):
            return None

        def to(self, dev):
            self.device = dev
            return self

        def eval(self):
            return self

    roformer = types.ModuleType("voxweave.vendor.mel_band_roformer")
    roformer.MelBandRoformer = _FakeRoformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voxweave.vendor.mel_band_roformer", roformer)

    checkpoint = tmp_path / "separator.ckpt"
    checkpoint.write_bytes(b"stable separator checkpoint")
    separator_config = tmp_path / "separator.yaml"
    separator_config.write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        backend,
        "_resolve_separator_files",
        lambda: (checkpoint, separator_config),
    )
    monkeypatch.setattr(backend, "get_device", lambda: device)
    return precisions, checkpoint, torch_stub


def test_load_separator_enables_tf32_on_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv("VOXWEAVE_TF32", raising=False)
    precisions, _checkpoint, _torch_stub = _fake_separator_env(
        monkeypatch, tmp_path, device="cuda"
    )
    backend._load_separator()
    assert precisions == ["high"]


@pytest.mark.parametrize("value", ["0", "false", "off", "FALSE", "Off"])
def test_load_separator_tf32_opt_out(monkeypatch, tmp_path, value):
    # env is read at call time, not import time, so operators can flip it per run
    monkeypatch.setenv("VOXWEAVE_TF32", value)
    precisions, _checkpoint, _torch_stub = _fake_separator_env(
        monkeypatch, tmp_path, device="cuda"
    )
    backend._load_separator()
    assert precisions == []


def test_load_separator_leaves_tf32_alone_off_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv("VOXWEAVE_TF32", raising=False)
    precisions, _checkpoint, _torch_stub = _fake_separator_env(
        monkeypatch, tmp_path, device="cpu"
    )
    backend._load_separator()
    assert precisions == []


def test_load_separator_refuses_checkpoint_mutation_during_load(monkeypatch, tmp_path):
    _precisions, checkpoint, torch_stub = _fake_separator_env(
        monkeypatch, tmp_path, device="cpu"
    )

    def mutate_while_loading(stream, **_kwargs):
        checkpoint.write_bytes(b"replacement separator checkpoint")
        stream.seek(0)
        stream.read()
        return {}

    torch_stub.load = mutate_while_loading  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="checkpoint changed while loading"):
        backend._load_separator()
