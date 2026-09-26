"""--language normalization for the ASR engines (no real model loading).

qwen_asr validates the capitalized English name ("Japanese"): the documented ISO
form (``--language ja``) used to reach it raw, fail every chunk with "Unsupported
language: Ja", and kill the run only after the whole ASR pass.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from voxweave import backend, lang


@pytest.fixture(autouse=True)
def _force_torch_backend(monkeypatch):
    monkeypatch.setattr(backend, "_use_mlx", lambda: False)


@pytest.mark.parametrize(
    ("given", "qwen", "whisper"),
    [
        ("ja", "Japanese", "ja"),
        ("Japanese", "Japanese", "ja"),
        ("japanese", "Japanese", "ja"),
        (" JA ", "Japanese", "ja"),
        ("en-US", "English", "en"),
        ("zh_CN", "Chinese", "zh"),
        ("yue", "Cantonese", "zh"),  # whisper < large-v3 has no Cantonese token
        ("Cantonese", "Cantonese", "zh"),
        # Qwen3-ASR recognizes more languages than the aligner's 11
        ("ar", "Arabic", "ar"),
        ("Arabic", "Arabic", "ar"),
        ("Filipino", "Filipino", "tl"),
    ],
)
def test_engine_language_maps_iso_codes_and_names(given, qwen, whisper):
    assert backend._engine_language("qwen", given) == qwen
    assert backend._engine_language("whisper", given) == whisper


@pytest.mark.parametrize("given", [None, "", "   "])
def test_engine_language_blank_means_auto_detect(given):
    assert backend._engine_language("qwen", given) is None
    assert backend._engine_language("whisper", given) is None


@pytest.mark.parametrize("engine", ["qwen", "whisper", "fusion"])
def test_engine_language_unknown_value_names_value_and_supported_set(engine):
    with pytest.raises(ValueError) as failure:
        backend._engine_language(engine, "klingon")
    msg = str(failure.value)
    assert "'klingon'" in msg
    assert "ja (Japanese)" in msg and "en (English)" in msg


def test_qwen_kwargs_carry_the_capitalized_name():
    assert backend._qwen_asr_kwargs("ja", None) == {
        "language": "Japanese",
        "return_time_stamps": False,
    }
    assert backend._qwen_asr_kwargs(None, None)["language"] is None


def _qwen_supported_languages() -> list[str]:
    """qwen_asr's SUPPORTED_LANGUAGES, read from source (importing qwen_asr pulls in
    transformers/torch for several seconds)."""
    spec = importlib.util.find_spec("qwen_asr")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("qwen-asr not installed")
    utils = Path(next(iter(spec.submodule_search_locations))) / "inference" / "utils.py"
    for node in ast.parse(utils.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(t, ast.Name) and t.id == "SUPPORTED_LANGUAGES"
                for t in targets
            ):
                assert node.value is not None
                return list(ast.literal_eval(node.value))
    pytest.skip("qwen_asr.inference.utils has no SUPPORTED_LANGUAGES")


def test_asr_language_table_mirrors_qwen_asr():
    supported = _qwen_supported_languages()
    # every accepted value becomes a name validate_language takes verbatim
    assert sorted(lang.to_asr_name(iso) for iso in lang._ASR_ISO_TO_NAME) == sorted(
        supported
    )


def _stub_chunks(tmp_path, n: int) -> list[Path]:
    wavs = [tmp_path / f"c{i}.wav" for i in range(n)]
    for w in wavs:
        w.write_bytes(b"x")
    return wavs


class _AsrRes:
    def __init__(self, text):
        self.text, self.language, self.time_stamps = text, None, None


def test_transcribe_chunks_iso_language_reaches_qwen_as_its_name(monkeypatch, tmp_path):
    calls: list = []

    class _Model:
        def transcribe(self, audio, **kw):
            calls.append(kw)
            return [_AsrRes("はい")]

    monkeypatch.setenv("VOXWEAVE_ASR_BATCH", "1")
    monkeypatch.setattr(backend, "_get_asr", lambda m=None: _Model())
    aligned: list = []
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda w, t, a: aligned.append(a) or [{"text": t, "start": 0.0, "end": 1.0}],
    )
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)

    out = backend.transcribe_chunks(
        _stub_chunks(tmp_path, 2), "ja", asr_model="qwen3-asr-1.7b"
    )

    assert [kw["language"] for kw in calls] == ["Japanese", "Japanese"]
    assert [lng for lng, _t, _u in out] == ["ja", "ja"]  # override kept as given
    assert aligned == ["ja", "ja"]


def test_transcribe_chunks_iso_language_reaches_whisper_as_iso(monkeypatch, tmp_path):
    seen: list = []

    class _Info:
        language = "ja"

    class _Seg:
        text = "はい"

    class _Model:
        def transcribe(self, path, **kw):
            seen.append(kw["language"])
            return iter([_Seg()]), _Info()

    monkeypatch.setattr(backend, "_get_whisper", lambda mid: _Model())
    monkeypatch.setattr(
        backend, "align_text", lambda w, t, a: [{"text": t, "start": 0.0, "end": 1.0}]
    )
    monkeypatch.setattr(backend, "_release_whisper", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)

    backend.transcribe_chunks(
        _stub_chunks(tmp_path, 2), "Japanese", asr_model="large-v3"
    )
    assert seen == ["ja", "ja"]


@pytest.mark.parametrize("asr_model", ["qwen3-asr-1.7b", "large-v3", "fusion"])
def test_unknown_language_fails_before_any_model_loads(
    monkeypatch, tmp_path, asr_model
):
    def _no_load(*_a, **_k):
        raise AssertionError("no model may load for an invalid --language")

    monkeypatch.setattr(backend, "_get_asr", _no_load)
    monkeypatch.setattr(backend, "_get_whisper", _no_load)
    wavs = _stub_chunks(tmp_path, 3)
    with pytest.raises(ValueError, match="unsupported language 'xx'"):
        backend.transcribe_chunks(wavs, "xx", asr_model=asr_model)
    with pytest.raises(ValueError, match="unsupported language 'xx'"):
        backend.transcribe_align(wavs[0], "xx", asr_model=asr_model)
