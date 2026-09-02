from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import soundfile as sf
import torch
from huggingface_hub.errors import HfHubHTTPError

from voxweave import config, diarize


LEGACY_MODEL = "pyannote/speaker-diarization-3.1"
COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"


@pytest.fixture(autouse=True)
def _isolated_model_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.delenv("VOXWEAVE_DIARIZE_MODEL", raising=False)
    monkeypatch.setattr(diarize, "_pipeline", None)
    if hasattr(diarize, "_pipeline_model"):
        monkeypatch.setattr(diarize, "_pipeline_model", None)


def _write_config(tmp_path: Path, text: str) -> None:
    (tmp_path / "voxweave.conf").write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3.1", LEGACY_MODEL),
        ("community-1", COMMUNITY_MODEL),
        (LEGACY_MODEL, LEGACY_MODEL),
        (COMMUNITY_MODEL, COMMUNITY_MODEL),
        ("example/custom-diarizer", "example/custom-diarizer"),
    ],
)
def test_diarize_model_aliases_and_passthrough(value: str, expected: str) -> None:
    assert config.resolve_diarize_model(value) == expected


def test_diarize_model_default_is_community_1() -> None:
    assert config.resolve_diarize_model() == COMMUNITY_MODEL


def test_diarize_model_config_is_used_when_env_and_cli_are_absent(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, '[diarize]\nmodel = "example/conf-diarizer"\n')

    assert config.resolve_diarize_model() == "example/conf-diarizer"


def test_diarize_model_env_beats_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, '[diarize]\nmodel = "community-1"\n')
    monkeypatch.setenv("VOXWEAVE_DIARIZE_MODEL", "3.1")

    assert config.resolve_diarize_model() == LEGACY_MODEL


def test_diarize_model_cli_beats_env_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, '[diarize]\nmodel = "3.1"\n')
    monkeypatch.setenv("VOXWEAVE_DIARIZE_MODEL", "example/env-model")

    assert config.resolve_diarize_model("community-1") == COMMUNITY_MODEL


def test_blank_env_falls_through_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, '[diarize]\nmodel = "example/conf-diarizer"\n')
    monkeypatch.setenv("VOXWEAVE_DIARIZE_MODEL", "   ")

    assert config.resolve_diarize_model() == "example/conf-diarizer"


class _Segment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _Annotation:
    def __init__(
        self,
        tracks: list[tuple[_Segment, str, str]],
        *,
        labels: list[str] | None = None,
    ) -> None:
        self._tracks = tracks
        self._labels = labels

    def itertracks(self, *, yield_label: bool = False):
        for segment, track, label in self._tracks:
            yield ((segment, track, label) if yield_label else (segment, track))

    def labels(self) -> list[str]:
        if self._labels is not None:
            return self._labels
        return list(dict.fromkeys(label for _segment, _track, label in self._tracks))


class _CapturePipeline:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, audio: object, **kwargs: object) -> object:
        self.calls.append((audio, kwargs))
        return self.result


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000, subtype="FLOAT")
    return path


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _CapturePipeline,
) -> list[tuple[str, str]]:
    requested: list[tuple[str, str]] = []

    def get_pipeline(token: str, model: str):
        requested.append((token, model))
        return pipeline

    monkeypatch.setattr(diarize, "_get_pipeline", get_pipeline)
    return requested


def test_pyannote4_output_is_adapted_without_return_embeddings_kw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation = _Annotation(
        [
            (_Segment(0.0, 1.0), "track-a", "SPEAKER_A"),
            (_Segment(1.0, 2.0), "track-b", "SPEAKER_B"),
        ],
        labels=["SPEAKER_B", "SPEAKER_A"],
    )
    exclusive = _Annotation([(_Segment(0.0, 2.0), "exclusive", "EXCLUSIVE_ONLY")])
    output = SimpleNamespace(
        speaker_diarization=annotation,
        exclusive_speaker_diarization=exclusive,
        speaker_embeddings=np.array(
            [
                [3.0, 4.0, *([0.0] * 14)],
                [0.0, 0.0, 5.0, *([0.0] * 13)],
            ],
            dtype=np.float32,
        ),
    )
    pipeline = _CapturePipeline(output)
    requested = _patch_pipeline(monkeypatch, pipeline)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "4.0.7")

    result = diarize.diarize_turns(
        _wav(tmp_path),
        token="hf_test",
        model=COMMUNITY_MODEL,
        min_speakers=2,
        max_speakers=4,
        want_embeddings=True,
    )

    assert requested == [("hf_test", COMMUNITY_MODEL)]
    assert result.turns == [
        (0.0, 1.0, "SPEAKER_A"),
        (1.0, 2.0, "SPEAKER_B"),
    ]
    assert all(label != "EXCLUSIVE_ONLY" for _start, _end, label in result.turns)
    assert (result.centroids or {})["SPEAKER_B"][:2] == pytest.approx([0.6, 0.8])
    assert (result.centroids or {})["SPEAKER_A"][2] == pytest.approx(1.0)
    _audio, kwargs = pipeline.calls[0]
    assert "return_embeddings" not in kwargs
    assert kwargs["min_speakers"] == 2
    assert kwargs["max_speakers"] == 4


def test_pyannote4_embeddings_are_not_exposed_without_voiceprint_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation = _Annotation([(_Segment(0.0, 1.0), "track", "SPEAKER_00")])
    output = SimpleNamespace(
        speaker_diarization=annotation,
        exclusive_speaker_diarization=annotation,
        speaker_embeddings=np.array([[1.0, *([0.0] * 15)]], dtype=np.float32),
    )
    pipeline = _CapturePipeline(output)
    _patch_pipeline(monkeypatch, pipeline)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "4.0.7")

    result = diarize.diarize_turns(
        _wav(tmp_path),
        token="hf_test",
        model=COMMUNITY_MODEL,
        want_embeddings=False,
    )

    assert result.centroids is None
    assert "return_embeddings" not in pipeline.calls[0][1]


def test_legacy_annotation_output_is_still_adapted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation = _Annotation([(_Segment(0.0, 1.0), "track", "SPEAKER_00")])
    pipeline = _CapturePipeline(annotation)
    _patch_pipeline(monkeypatch, pipeline)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "3.4.0")

    result = diarize.diarize_turns(_wav(tmp_path), token="hf_test", model=LEGACY_MODEL)

    assert result.turns == [(0.0, 1.0, "SPEAKER_00")]
    assert result.centroids is None


def test_legacy_annotation_embedding_tuple_is_still_adapted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation = _Annotation([(_Segment(0.0, 1.0), "track", "SPEAKER_00")])
    pipeline = _CapturePipeline(
        (annotation, np.array([[1.0, *([0.0] * 15)]], dtype=np.float32))
    )
    _patch_pipeline(monkeypatch, pipeline)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "3.4.0")

    result = diarize.diarize_turns(
        _wav(tmp_path),
        token="hf_test",
        model=LEGACY_MODEL,
        want_embeddings=True,
    )

    assert pipeline.calls[0][1]["return_embeddings"] is True
    assert set(result.centroids or {}) == {"SPEAKER_00"}


class _LoadedPipeline:
    def to(self, _device: object) -> None:
        return None


class _PipelineLoader:
    calls: list[tuple[object, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, checkpoint: object, **kwargs: object) -> _LoadedPipeline:
        cls.calls.append((checkpoint, kwargs))
        return _LoadedPipeline()


def _install_pyannote_module(
    monkeypatch: pytest.MonkeyPatch, pipeline_class: type
) -> None:
    pyannote_package = types.ModuleType("pyannote")
    audio_module = types.ModuleType("pyannote.audio")
    audio_module.Pipeline = pipeline_class  # type: ignore[attr-defined]
    pyannote_package.audio = audio_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_package)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)


def test_pipeline_cache_is_keyed_by_model_and_pyannote4_uses_token_and_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _PipelineLoader.calls = []
    _install_pyannote_module(monkeypatch, _PipelineLoader)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "4.0.7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    second_model = "example/second-diarizer"

    def direct_plan(model: str, _token: str):
        checkpoint, revision = diarize._split_model_revision(model)
        return diarize._PipelineLoadPlan(
            checkpoint=checkpoint,
            revision=revision,
            authority=None,
            outer_config_sha256="unresolved",
        )

    monkeypatch.setattr(diarize, "_prepare_pipeline_load", direct_plan)

    first = diarize._get_pipeline("hf_secret", COMMUNITY_MODEL)
    repeated = diarize._get_pipeline("hf_secret", COMMUNITY_MODEL)
    second = diarize._get_pipeline("hf_secret", second_model)

    assert first is repeated
    assert second is not first
    assert [checkpoint for checkpoint, _kwargs in _PipelineLoader.calls] == [
        COMMUNITY_MODEL,
        second_model,
    ]
    for _checkpoint, kwargs in _PipelineLoader.calls:
        assert kwargs["token"] == "hf_secret"
        assert kwargs["cache_dir"] == config.AUDIO_CACHE
        assert "use_auth_token" not in kwargs


class _ForbiddenPipeline:
    @classmethod
    def from_pretrained(cls, _checkpoint: object, **_kwargs: object) -> object:
        response = requests.Response()
        response.status_code = 403
        response.url = f"https://hf.co/{COMMUNITY_MODEL}"
        raise HfHubHTTPError("403 Forbidden", response=response)


def test_community1_403_names_the_exact_model_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pyannote_module(monkeypatch, _ForbiddenPipeline)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "4.0.7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as caught:
        diarize._get_pipeline("hf_secret", COMMUNITY_MODEL)

    message = str(caught.value)
    assert COMMUNITY_MODEL in message
    assert f"https://hf.co/{COMMUNITY_MODEL}" in message
    assert "accept" in message.lower()


def test_provenance_records_resolved_model_and_redacts_loader_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_cache = "/private/host/cache/that/must/not/be/persisted"
    pipeline = SimpleNamespace(
        embedding={
            "checkpoint": COMMUNITY_MODEL,
            "subfolder": "embedding",
            "token": "hf_secret_must_not_leak",
            "cache_dir": private_cache,
        }
    )
    monkeypatch.setattr(
        diarize, "_outer_config_identity", lambda *_args, **_kwargs: "a" * 64
    )
    monkeypatch.setattr(diarize, "_checkpoint_identity", lambda _pipeline: "b" * 64)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "4.0.7")

    provenance = diarize._build_provenance(
        pipeline,
        model=COMMUNITY_MODEL,
        embedding_dim=256,
        audio_profile={
            "separated": False,
            "normalized": False,
            "sample_rate": 16000,
        },
        torch_version="2.11.0",
    )

    assert provenance["diarization_model"] == COMMUNITY_MODEL
    assert provenance["pyannote_version"] == "4.0.7"
    embedding_model = provenance["embedding_model"]
    assert isinstance(embedding_model, str)
    assert embedding_model != "unresolved"
    assert COMMUNITY_MODEL in embedding_model
    assert "embedding" in embedding_model
    encoded = json.dumps(provenance, sort_keys=True)
    assert "hf_secret_must_not_leak" not in encoded
    assert private_cache not in encoded
