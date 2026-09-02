from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voxweave import diarize


_COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
_LEGACY_MODEL = "pyannote/speaker-diarization-3.1"
_COMMUNITY_PLDA = {"checkpoint": _COMMUNITY_MODEL, "subfolder": "plda"}
_SPEAKER_MODULE = "pyannote.audio.pipelines.speaker_diarization"


def _write_pipeline_config(
    path: Path,
    *,
    clustering: str,
    embedding: str,
    community_children: bool = False,
) -> Path:
    children = ""
    if community_children:
        children = "    segmentation: $model/segmentation\n    plda: $model/plda\n"
    path.write_text(
        "pipeline:\n"
        "  name: pyannote.audio.pipelines.SpeakerDiarization\n"
        "  params:\n"
        f"    clustering: {clustering}\n"
        f"    embedding: {embedding}\n"
        f"{children}"
        "params:\n"
        "  clustering:\n"
        "    threshold: 0.6\n",
        encoding="utf-8",
    )
    return path


def _install_speaker_module(
    monkeypatch: pytest.MonkeyPatch,
    get_plda: Any,
) -> types.ModuleType:
    pyannote = types.ModuleType("pyannote")
    pyannote.__path__ = []
    audio = types.ModuleType("pyannote.audio")
    audio.__path__ = []
    pipelines = types.ModuleType("pyannote.audio.pipelines")
    pipelines.__path__ = []
    module = types.ModuleType(_SPEAKER_MODULE)
    module.get_plda = get_plda
    pipelines.speaker_diarization = module
    audio.pipelines = pipelines
    pyannote.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, _SPEAKER_MODULE, module)
    return module


def _direct_construct(pipeline_cls, token: str, checkpoint, *, revision=None):
    return pipeline_cls.from_pretrained(
        checkpoint,
        revision=revision,
        token=token,
        cache_dir=diarize.config.AUDIO_CACHE,
    )


class _PLDAProbePipeline:
    plda_results: list[object] = []

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        speaker_module = importlib.import_module(_SPEAKER_MODULE)
        cls.plda_results.append(
            speaker_module.get_plda(
                dict(_COMMUNITY_PLDA),
                token=kwargs.get("token"),
                cache_dir=kwargs.get("cache_dir"),
            )
        )
        return SimpleNamespace()


def test_pyannote4_loader_splits_revision_and_uses_v4_keywords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "loaded"

    cache_dir = tmp_path / "audio-cache"
    monkeypatch.setattr(diarize.config, "AUDIO_CACHE", str(cache_dir))
    monkeypatch.setattr(
        diarize, "_pipeline_config_path", lambda *_args, **_kwargs: None
    )

    result = diarize._load_pipeline(
        Pipeline,
        "hf_test_token",
        model="acme/diarization@release-7",
    )

    assert result is not None
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("acme/diarization",)
    assert kwargs["revision"] == "release-7"
    assert kwargs["token"] == "hf_test_token"
    assert Path(str(kwargs["cache_dir"])) == cache_dir
    assert "use_auth_token" not in kwargs


def test_legacy_31_suppresses_only_its_unused_community_plda_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_pipeline_config(
        tmp_path / "config.yaml",
        clustering="AgglomerativeClustering",
        embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
    )
    forbidden_calls: list[object] = []

    def forbidden_get_plda(value, **_kwargs):
        forbidden_calls.append(value)
        raise AssertionError("legacy 3.1 attempted to load Community-1 PLDA")

    speaker_module = _install_speaker_module(monkeypatch, forbidden_get_plda)
    _PLDAProbePipeline.plda_results = []
    monkeypatch.setattr(
        diarize, "_pipeline_config_path", lambda *_args, **_kwargs: config_path
    )
    monkeypatch.setattr(
        diarize, "_embedding_load_authority", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", _direct_construct)

    selected = f"{_LEGACY_MODEL}@legacy-revision"
    loaded = diarize._load_pipeline(
        _PLDAProbePipeline,
        "hf_legacy",
        model=selected,
    )

    assert loaded is not None
    assert forbidden_calls == []
    assert _PLDAProbePipeline.plda_results == [None]
    assert speaker_module.get_plda is forbidden_get_plda


def test_legacy_31_refuses_plda_suppression_without_agglomerative_precondition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_pipeline_config(
        tmp_path / "config.yaml",
        clustering="VBxClustering",
        embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
    )
    calls: list[object] = []

    def forbidden_get_plda(value, **_kwargs):
        calls.append(value)
        raise AssertionError("unsafe 3.1 plan reached Community-1 PLDA")

    speaker_module = _install_speaker_module(monkeypatch, forbidden_get_plda)
    monkeypatch.setattr(
        diarize, "_pipeline_config_path", lambda *_args, **_kwargs: config_path
    )
    monkeypatch.setattr(
        diarize, "_embedding_load_authority", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", _direct_construct)

    with pytest.raises(RuntimeError, match="3\\.1.*Agglomerative"):
        diarize._load_pipeline(
            _PLDAProbePipeline,
            "hf_legacy",
            model=_LEGACY_MODEL,
        )

    assert calls == []
    assert speaker_module.get_plda is forbidden_get_plda


def test_community_plan_keeps_remote_model_context_for_structured_subfolders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_pipeline_config(
        tmp_path / "config.yaml",
        clustering="VBxClustering",
        embedding="$model/embedding",
        community_children=True,
    )
    calls: list[tuple[object, str | None]] = []

    def construct(_pipeline_cls, _token, checkpoint, *, revision=None):
        calls.append((checkpoint, revision))
        return SimpleNamespace()

    monkeypatch.setattr(
        diarize, "_pipeline_config_path", lambda *_args, **_kwargs: config_path
    )
    monkeypatch.setattr(
        diarize, "_embedding_load_authority", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", construct)

    selected = f"{_COMMUNITY_MODEL}@community-revision"
    loaded = diarize._load_pipeline(object, "hf_community", model=selected)

    assert loaded is not None
    assert len(calls) == 1
    checkpoint, revision = calls[0]
    assert revision is None
    assert isinstance(checkpoint, dict)
    params = checkpoint["pipeline"]["params"]
    expected_common = {
        "checkpoint": _COMMUNITY_MODEL,
        "revision": "community-revision",
        "token": "hf_community",
        "cache_dir": diarize.config.AUDIO_CACHE,
    }
    assert params["segmentation"] == {**expected_common, "subfolder": "segmentation"}
    assert params["embedding"] == {**expected_common, "subfolder": "embedding"}
    assert params["plda"] == {**expected_common, "subfolder": "plda"}
    config_text = config_path.read_text(encoding="utf-8")
    assert "$model/segmentation" in config_text
    assert "$model/embedding" in config_text
    assert "$model/plda" in config_text


def test_community_nested_embedding_has_cache_bound_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "b" * 40
    cache_dir = tmp_path / "audio-cache"
    checkpoint = (
        cache_dir
        / "models--pyannote--speaker-diarization-community-1"
        / "snapshots"
        / revision
        / "embedding"
        / "pytorch_model.bin"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint_bytes = b"community-1 nested embedding checkpoint"
    checkpoint.write_bytes(checkpoint_bytes)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def locate(*args, **kwargs):
        calls.append((args, kwargs))
        return str(checkpoint)

    import huggingface_hub

    monkeypatch.setattr(diarize.config, "AUDIO_CACHE", str(cache_dir))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", locate)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", locate)

    authority = diarize._embedding_load_authority(
        {
            "checkpoint": _COMMUNITY_MODEL,
            "revision": revision,
            "subfolder": "embedding",
            "token": "hf_secret_must_not_enter_provenance",
            "cache_dir": str(cache_dir),
        }
    )

    assert authority is not None
    assert authority.binding.path == checkpoint.resolve()
    assert authority.binding.sha256 == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert _COMMUNITY_MODEL in authority.provenance_value
    assert revision in authority.provenance_value
    assert "embedding" in authority.provenance_value
    assert "hf_secret_must_not_enter_provenance" not in authority.provenance_value
    assert str(cache_dir) not in authority.provenance_value
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == _COMMUNITY_MODEL
    assert args[1] in {"pytorch_model.bin", "embedding/pytorch_model.bin"}
    assert kwargs.get("revision") == revision
    assert kwargs.get("subfolder") == "embedding" or args[1].startswith("embedding/")
    assert Path(str(kwargs["cache_dir"])) == cache_dir
