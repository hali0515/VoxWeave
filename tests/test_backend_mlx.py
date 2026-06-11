"""MLX (Apple Silicon) backend: selector, repo mapping, and API adapters (no real model loading)."""

import sys
import types

import pytest

from voxweave import backend, backend_mlx, runtime


# ───────────────────────────── backend selector ──────────────────────────────


def test_use_mlx_env_overrides_win(monkeypatch):
    monkeypatch.setattr(runtime, "_DEVICE", "cuda:0")  # device says cuda...
    monkeypatch.setenv("VOXWEAVE_BACKEND", "mlx")  # ...but env forces mlx
    assert backend._use_mlx() is True
    monkeypatch.setenv("VOXWEAVE_BACKEND", "torch")
    monkeypatch.setattr(runtime, "_DEVICE", "mps")  # device says mps...
    assert backend._use_mlx() is False  # ...but env forces torch


def test_use_mlx_auto_on_for_mps(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_BACKEND", raising=False)
    monkeypatch.setattr(runtime, "_DEVICE", "mps")
    assert backend._use_mlx() is True
    monkeypatch.setattr(runtime, "_DEVICE", "cuda:0")
    assert backend._use_mlx() is False
    monkeypatch.setattr(runtime, "_DEVICE", "cpu")
    assert backend._use_mlx() is False


# ───────────────────────────── repo mapping ──────────────────────────────


def test_mlx_asr_repo_mapping(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_MLX_ASR_REPO", raising=False)
    assert (
        backend_mlx._mlx_asr_repo("Qwen/Qwen3-ASR-0.6B")
        == "mlx-community/Qwen3-ASR-0.6B-8bit"
    )
    assert (
        backend_mlx._mlx_asr_repo("Qwen/Qwen3-ASR-1.7B")
        == "mlx-community/Qwen3-ASR-1.7B-8bit"
    )
    # unknown id with a 1.7 hint -> 1.7B; otherwise default 0.6B
    assert "1.7B" in backend_mlx._mlx_asr_repo("custom/whatever-1.7b")
    assert backend_mlx._mlx_asr_repo("custom/tiny") == backend_mlx._DEFAULT_MLX_ASR
    assert backend_mlx._mlx_asr_repo(None) == backend_mlx._DEFAULT_MLX_ASR


def test_mlx_asr_repo_env_override_wins(monkeypatch):
    # VOXWEAVE_MLX_ASR_REPO hard-overrides the size mapping (e.g. to pin a 4-bit quant)
    monkeypatch.setenv("VOXWEAVE_MLX_ASR_REPO", "mlx-community/Qwen3-ASR-1.7B-4bit")
    assert (
        backend_mlx._mlx_asr_repo("Qwen/Qwen3-ASR-0.6B")
        == "mlx-community/Qwen3-ASR-1.7B-4bit"
    )


# ───────────────────────────── ASR adapter ──────────────────────────────


class _FakeSTTOutput:
    def __init__(self, text, language):
        self.text = text
        self.language = language


class _FakeAsrModel:
    """Records generate() kwargs and returns a canned STTOutput."""

    def __init__(self, text="hello", language=None):
        self._text = text
        self._language = language
        self.calls = []

    def generate(self, audio, *, language=None, system_prompt=None, **kw):
        self.calls.append(
            {"audio": audio, "language": language, "system_prompt": system_prompt}
        )
        return _FakeSTTOutput(self._text, self._language)


def test_mlx_asr_maps_language_and_context():
    fake = _FakeAsrModel(text="こんにちは", language=["japanese"])
    asr = backend_mlx._MlxAsr(fake)
    out = asr.transcribe(
        "/tmp/a.wav", language="ja", return_time_stamps=False, context="固有名詞"
    )
    assert fake.calls[0]["language"] == "japanese"  # ISO -> Qwen full name
    # context -> system_prompt, bare term auto-framed (see backend.format_qwen_context)
    assert fake.calls[0]["system_prompt"] == "Proper nouns: 固有名詞."
    assert out[0].text == "こんにちは"
    assert out[0].language == "japanese"  # first non-empty of the per-segment list


def test_mlx_asr_none_language_autodetect():
    fake = _FakeAsrModel(text="hi", language="english")
    asr = backend_mlx._MlxAsr(fake)
    out = asr.transcribe("/tmp/a.wav", language=None, context=None)
    assert fake.calls[0]["language"] is None  # None -> auto-detect
    assert fake.calls[0]["system_prompt"] is None
    assert out[0].language == "english"


# ───────────────────────────── aligner adapter ──────────────────────────────


class _FakeItem:
    def __init__(self, text, start, end):
        self.text = text
        self.start_time = start
        self.end_time = end


class _FakeAlignResult:
    def __init__(self, items):
        self.items = items


class _FakeAligner:
    def __init__(self, items):
        self._items = items
        self.calls = []

    def generate(self, *, audio, text, language):
        self.calls.append({"audio": audio, "text": text, "language": language})
        return _FakeAlignResult(self._items)


def test_mlx_align_converts_items_to_units(monkeypatch):
    fake = _FakeAligner([_FakeItem("a", 0.0, 0.5), _FakeItem("b", 0.5, 1.0)])
    monkeypatch.setattr(backend_mlx, "_aligner", fake)
    units = backend_mlx.align("/tmp/a.wav", "a b", "en")
    assert fake.calls[0]["language"] == "english"  # ISO -> full name
    assert units == [
        {"text": "a", "start": 0.0, "end": 0.5},
        {"text": "b", "start": 0.5, "end": 1.0},
    ]


def test_mlx_align_handles_batch_list_and_iterable(monkeypatch):
    # generate returns a list (batch API) of results that are directly iterable over items
    class _IterResult(list):
        pass

    class _BatchAligner:
        def generate(self, *, audio, text, language):
            return [_IterResult([_FakeItem("x", 1.0, 2.0)])]

    monkeypatch.setattr(backend_mlx, "_aligner", _BatchAligner())
    units = backend_mlx.align("/tmp/a.wav", "x", "ja")
    assert units == [{"text": "x", "start": 1.0, "end": 2.0}]


# ───────────────────────────── dispatch from backend.py ──────────────────────────────


def test_align_text_dispatches_to_mlx(monkeypatch):
    # MLX is the Qwen *fallback*: only reached when no CTC model is configured for the language
    # (CTC-configured langs like en->wav2vec2 stay on CTC even on the MLX backend).
    monkeypatch.setattr(backend, "_use_mlx", lambda: True)
    monkeypatch.setattr(backend.config, "align_model_for", lambda iso: None)
    called = {}

    def _fake_align(wav, text, lang):
        called["args"] = (str(wav), text, lang)
        return [{"text": "z", "start": 0.0, "end": 0.1}]

    monkeypatch.setattr(backend_mlx, "align", _fake_align)
    units = backend.align_text("/tmp/a.wav", "z", "zh")
    assert called["args"] == ("/tmp/a.wav", "z", "zh")
    assert units == [{"text": "z", "start": 0.0, "end": 0.1}]


def test_align_text_ctc_lang_stays_on_ctc_under_mlx(monkeypatch):
    # en has a CTC model configured -> wav2vec2 CTC even on the MLX backend; MLX is NOT called.
    monkeypatch.setattr(backend, "_use_mlx", lambda: True)
    monkeypatch.setattr(
        backend.config, "align_model_for", lambda iso: "WAV2VEC2_ASR_LARGE_LV60K_960H"
    )

    def _fake_ctc(wav, text, iso, model):
        return [{"text": "w", "start": 0.0, "end": 0.2}]

    def _boom_mlx(*a, **k):
        raise AssertionError("MLX must not be used when a CTC model is configured")

    monkeypatch.setattr(backend, "align_text_ctc", _fake_ctc)
    monkeypatch.setattr(backend_mlx, "align", _boom_mlx)
    units = backend.align_text("/tmp/a.wav", "w", "en")
    assert units == [{"text": "w", "start": 0.0, "end": 0.2}]


def test_get_asr_dispatches_to_mlx(monkeypatch):
    monkeypatch.setattr(backend, "_use_mlx", lambda: True)
    sentinel = object()
    seen = {}

    def _fake_get_asr(mid):
        seen["mid"] = mid
        return sentinel

    monkeypatch.setattr(backend_mlx, "get_asr", _fake_get_asr)
    assert backend._get_asr("qwen3-asr-1.7b") is sentinel
    assert seen["mid"] == "Qwen/Qwen3-ASR-1.7B"  # resolved before dispatch


# ───────────────────────────── whisper (mlx-whisper) ──────────────────────────────


def test_mlx_whisper_repo_mapping(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_MLX_WHISPER_REPO", raising=False)
    assert (
        backend_mlx._mlx_whisper_repo("large-v3")
        == "mlx-community/whisper-large-v3-mlx"
    )
    assert (
        backend_mlx._mlx_whisper_repo("large-v3-turbo")
        == "mlx-community/whisper-large-v3-turbo"
    )
    assert (
        backend_mlx._mlx_whisper_repo("distil-large-v3")
        == "mlx-community/distil-whisper-large-v3"
    )
    # unknown size -> generic mlx-community convention
    assert backend_mlx._mlx_whisper_repo("small") == "mlx-community/whisper-small-mlx"


def test_mlx_whisper_repo_env_override_wins(monkeypatch):
    monkeypatch.setenv(
        "VOXWEAVE_MLX_WHISPER_REPO", "mlx-community/whisper-large-v3-8bit"
    )
    assert (
        backend_mlx._mlx_whisper_repo("large-v3")
        == "mlx-community/whisper-large-v3-8bit"
    )


def test_mlx_whisper_adapter_mirrors_faster_whisper(monkeypatch):
    # mlx_whisper.transcribe -> (segments, info) contract that backend._asr_only consumes
    captured = {}
    fake_mod = types.ModuleType("mlx_whisper")

    def _transcribe(audio, **kw):
        captured.update(kw)
        captured["audio"] = audio
        return {
            "text": "ignored when segments present",
            "segments": [{"text": "hello "}, {"text": "world"}],
            "language": "en",
        }

    fake_mod.transcribe = _transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)

    w = backend_mlx._MlxWhisper("/local/snapshot")
    segs, info = w.transcribe(
        "/tmp/a.wav",
        language="en",
        initial_prompt="ctx",
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=False,
    )
    assert "".join(s.text for s in segs) == "hello world"
    assert info.language == "en"
    assert captured["path_or_hf_repo"] == "/local/snapshot"
    assert captured["language"] == "en"
    assert captured["initial_prompt"] == "ctx"
    assert "vad_filter" not in captured  # faster-whisper-only arg dropped


def test_mlx_whisper_adapter_falls_back_to_full_text(monkeypatch):
    # no per-segment breakdown -> single segment from the top-level text
    fake_mod = types.ModuleType("mlx_whisper")
    fake_mod.transcribe = lambda audio, **kw: {
        "text": "solo",
        "segments": [],
        "language": "ja",
    }
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mod)
    segs, info = backend_mlx._MlxWhisper("/p").transcribe("/tmp/a.wav")
    assert [s.text for s in segs] == ["solo"]
    assert info.language == "ja"


def test_get_whisper_dispatches_to_mlx(monkeypatch):
    monkeypatch.setattr(backend, "_use_mlx", lambda: True)
    sentinel = object()
    seen = {}

    def _fake_get_whisper(mid):
        seen["mid"] = mid
        return sentinel

    monkeypatch.setattr(backend_mlx, "get_whisper", _fake_get_whisper)
    assert backend._get_whisper("large-v3") is sentinel
    assert seen["mid"] == "large-v3"


def test_mlx_get_whisper_reloads_on_size_change(monkeypatch):
    monkeypatch.setattr(backend_mlx, "_whisper", None)
    monkeypatch.setattr(backend_mlx, "_whisper_id", None)
    monkeypatch.setattr(
        backend_mlx, "_snapshot", lambda repo, cache: f"/snap/{repo.split('/')[-1]}"
    )
    a = backend_mlx.get_whisper("large-v3")
    assert backend_mlx._whisper_id == "large-v3"
    b = backend_mlx.get_whisper("large-v3")  # same size -> same singleton
    assert b is a
    backend_mlx.get_whisper("large-v3-turbo")  # size change -> reload
    assert backend_mlx._whisper_id == "large-v3-turbo"
    backend_mlx.release_whisper()
    assert backend_mlx._whisper is None and backend_mlx._whisper_id is None


# ───────────────────────────── missing-dependency error ──────────────────────────────


def test_load_missing_mlx_audio_raises_friendly(monkeypatch):
    # simulate mlx_audio not installed -> friendly RuntimeError pointing to voxweave[mps]
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "mlx_audio" or name.startswith("mlx_audio."):
            raise ModuleNotFoundError("No module named 'mlx_audio'", name="mlx_audio")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    with pytest.raises(RuntimeError, match=r"voxweave\[mps\]"):
        backend_mlx._load("mlx-community/Qwen3-ASR-0.6B-8bit", "/tmp/cache")


def test_release_is_noop_without_mlx(monkeypatch):
    # release()/release_asr() must not import mlx when the backend was never used
    monkeypatch.setattr(backend_mlx, "_asr", None)
    monkeypatch.setattr(backend_mlx, "_aligner", None)
    # ensure mlx.core import would fail loudly if attempted
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    backend_mlx.release()  # should not raise
    assert backend_mlx._asr is None and backend_mlx._aligner is None
