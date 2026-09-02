from __future__ import annotations

import hashlib
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from voxweave import config, diarize


COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
SPEAKER_MODULE = "pyannote.audio.pipelines.speaker_diarization"


class _MutatingStream:
    """Swap the backing file after this reader has consumed its stable view."""

    def __init__(self, stream: Any, mutate) -> None:
        self._stream = stream
        self._mutate = mutate

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        result = self._stream.__exit__(exc_type, exc_value, traceback)
        self._mutate()
        return result

    def read(self, *args: object, **kwargs: object):
        return self._stream.read(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def test_outer_config_digest_and_document_share_one_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = (
        b'{"marker":"initial","pipeline":{"name":"example.Pipeline",'
        b'"params":{"clustering":"CustomClustering"}}}'
    )
    replacement = (
        b'{"marker":"replacement","pipeline":{"name":"example.Pipeline",'
        b'"params":{"clustering":"CustomClustering"}}}'
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(initial)
    real_open = Path.open
    observed_opens: list[str] = []

    def replace_after_first_read() -> None:
        with real_open(config_path, "wb") as stream:
            stream.write(replacement)

    def controlled_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        stream = real_open(path, mode, *args, **kwargs)
        if path != config_path:
            return stream
        observed_opens.append(mode)
        if len(observed_opens) == 1:
            return _MutatingStream(stream, replace_after_first_read)
        return stream

    monkeypatch.setattr(Path, "open", controlled_open)
    monkeypatch.setattr(
        diarize,
        "_pipeline_config_path",
        lambda _model, _token: config_path,
    )
    monkeypatch.setattr(
        diarize,
        "_embedding_load_authority",
        lambda *_args, **_kwargs: None,
    )

    plan = diarize._prepare_pipeline_load("example/custom-pipeline", "hf_test")

    assert observed_opens == ["rb"]
    assert plan.outer_config_sha256 == hashlib.sha256(initial).hexdigest()
    assert isinstance(plan.checkpoint, dict)
    assert plan.checkpoint["marker"] == "initial"
    with real_open(config_path, "rb") as stream:
        assert stream.read() == replacement


def _install_fake_pyannote(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_class: type,
    speaker_module: types.ModuleType,
) -> None:
    pyannote_package = types.ModuleType("pyannote")
    pyannote_package.__path__ = []
    audio_module = types.ModuleType("pyannote.audio")
    audio_module.__path__ = []
    pipelines_module = types.ModuleType("pyannote.audio.pipelines")
    pipelines_module.__path__ = []
    audio_module.Pipeline = pipeline_class
    audio_module.pipelines = pipelines_module
    pipelines_module.speaker_diarization = speaker_module
    pyannote_package.audio = audio_module
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_package)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", pipelines_module)
    monkeypatch.setitem(sys.modules, SPEAKER_MODULE, speaker_module)


def test_legacy_and_community_construction_serialize_plda_and_singleton_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_started = threading.Event()
    community_started = threading.Event()
    allow_legacy_finish = threading.Event()
    events: list[str] = []
    event_lock = threading.Lock()
    community_plda: list[object] = []
    errors: list[BaseException] = []
    results: dict[str, object] = {}
    original_plda = object()

    def record(event: str) -> None:
        with event_lock:
            events.append(event)

    speaker_module = types.ModuleType(SPEAKER_MODULE)
    speaker_module.get_plda = lambda *_args, **_kwargs: original_plda
    legacy_pipeline = SimpleNamespace(model=config.LEGACY_DIARIZE_MODEL)
    community_pipeline = SimpleNamespace(model=COMMUNITY_MODEL)

    class Pipeline:
        @classmethod
        def from_pretrained(cls, checkpoint: object, **_kwargs: object) -> object:
            assert isinstance(checkpoint, dict)
            params = checkpoint["pipeline"]["params"]
            if params["clustering"] == "AgglomerativeClustering":
                record("legacy:start")
                legacy_started.set()
                assert speaker_module.get_plda("unused-legacy-plda") is None
                assert allow_legacy_finish.wait(timeout=5.0)
                record("legacy:end")
                return legacy_pipeline

            record("community:start")
            community_plda.append(
                speaker_module.get_plda({"checkpoint": COMMUNITY_MODEL})
            )
            community_started.set()
            record("community:end")
            return community_pipeline

    _install_fake_pyannote(monkeypatch, Pipeline, speaker_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diarize, "_pipeline", None)
    monkeypatch.setattr(diarize, "_pipeline_model", None)

    def prepare(model: str, _token: str | None) -> diarize._PipelineLoadPlan:
        clustering = (
            "AgglomerativeClustering"
            if model == config.LEGACY_DIARIZE_MODEL
            else "VBxClustering"
        )
        params: dict[str, object] = {"clustering": clustering}
        if model == COMMUNITY_MODEL:
            params["plda"] = {"checkpoint": COMMUNITY_MODEL, "subfolder": "plda"}
        return diarize._PipelineLoadPlan(
            checkpoint={
                "pipeline": {
                    "name": "pyannote.audio.pipelines.SpeakerDiarization",
                    "params": params,
                }
            },
            revision=None,
            authority=None,
            outer_config_sha256="a" * 64,
        )

    monkeypatch.setattr(diarize, "_prepare_pipeline_load", prepare)
    real_release = diarize.release

    def tracked_release() -> None:
        record("release")
        real_release()

    monkeypatch.setattr(diarize, "release", tracked_release)

    def load(name: str, model: str) -> None:
        try:
            results[name] = diarize._get_pipeline("hf_test", model)
        except BaseException as exc:
            errors.append(exc)

    legacy_thread = threading.Thread(
        target=load,
        args=("legacy", config.LEGACY_DIARIZE_MODEL),
        daemon=True,
    )
    community_thread = threading.Thread(
        target=load,
        args=("community", COMMUNITY_MODEL),
        daemon=True,
    )
    legacy_thread.start()
    assert legacy_started.wait(timeout=2.0)
    community_thread.start()
    overlapped = community_started.wait(timeout=0.25)
    allow_legacy_finish.set()
    legacy_thread.join(timeout=5.0)
    community_thread.join(timeout=5.0)

    assert not legacy_thread.is_alive()
    assert not community_thread.is_alive()
    assert errors == []
    assert overlapped is False
    assert events == [
        "legacy:start",
        "legacy:end",
        "release",
        "community:start",
        "community:end",
    ]
    assert community_plda == [original_plda]
    assert results == {
        "legacy": legacy_pipeline,
        "community": community_pipeline,
    }
    assert diarize._pipeline is community_pipeline
    assert diarize._pipeline_model == COMMUNITY_MODEL
