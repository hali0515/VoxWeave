# tests/test_diarize_gate_paths.py
# Gate/guard/hint paths of the diarization loader: which model ids are known to be
# gated, which Hub refusals must surface as a model-card error, when pyannote 4's
# unused PLDA fetch is suppressed, and what the CLI hint says about all of it.

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voxweave import config, diarize, turnembed, ui


_COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
_LEGACY_MODEL = "pyannote/speaker-diarization-3.1"
_SPEAKER_MODULE = "pyannote.audio.pipelines.speaker_diarization"


def _no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "conf_hf_token", lambda: None)


def _forbid_pipeline_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("gated model must be refused before any pipeline load")

    monkeypatch.setattr(diarize, "_load_pipeline", _boom)
    monkeypatch.setattr(diarize, "_get_pipeline", _boom)


def _install_fake_pyannote(
    monkeypatch: pytest.MonkeyPatch,
    get_plda: Any,
    pipeline_cls: Any = None,
) -> types.ModuleType:
    """Install a minimal pyannote package tree so nothing real is imported."""
    pyannote = types.ModuleType("pyannote")
    pyannote.__path__ = []  # type: ignore[attr-defined]
    audio = types.ModuleType("pyannote.audio")
    audio.__path__ = []  # type: ignore[attr-defined]
    if pipeline_cls is not None:
        audio.Pipeline = pipeline_cls  # type: ignore[attr-defined]
    pipelines = types.ModuleType("pyannote.audio.pipelines")
    pipelines.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType(_SPEAKER_MODULE)
    module.get_plda = get_plda  # type: ignore[attr-defined]
    pipelines.speaker_diarization = module  # type: ignore[attr-defined]
    audio.pipelines = pipelines  # type: ignore[attr-defined]
    pyannote.audio = audio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, _SPEAKER_MODULE, module)
    return module


def _write_pipeline_config(path: Path, *, clustering: str, plda: bool = False) -> Path:
    path.write_text(
        "pipeline:\n"
        "  name: pyannote.audio.pipelines.SpeakerDiarization\n"
        "  params:\n"
        f"    clustering: {clustering}\n"
        "    embedding: pyannote/wespeaker-voxceleb-resnet34-LM\n"
        + ("    plda: pyannote/speaker-diarization-community-1\n" if plda else "")
        + "params:\n"
        "  clustering:\n"
        "    threshold: 0.6\n",
        encoding="utf-8",
    )
    return path


def _plan(*, clustering: str, plda: bool = False) -> diarize._PipelineLoadPlan:
    params: dict[str, object] = {
        "clustering": clustering,
        "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
    }
    if plda:
        params["plda"] = {"checkpoint": _COMMUNITY_MODEL, "subfolder": "plda"}
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


def _direct_construct(pipeline_cls, token, checkpoint, *, revision=None):
    return pipeline_cls.from_pretrained(
        checkpoint,
        revision=revision,
        token=token,
        cache_dir=diarize.config.AUDIO_CACHE,
    )


class _PLDAProbe:
    """Records what get_plda returns at construction time."""

    def __init__(self) -> None:
        self.sentinel = object()
        self.seen: list[object] = []

    def get_plda(self, *_args: object, **_kwargs: object) -> object:
        return self.sentinel

    def pipeline_cls(self) -> Any:
        probe = self

        class Pipeline:
            @classmethod
            def from_pretrained(cls, _checkpoint, **_kwargs):
                import importlib

                module = importlib.import_module(_SPEAKER_MODULE)
                probe.seen.append(module.get_plda({"checkpoint": _COMMUNITY_MODEL}))
                return SimpleNamespace()

        return Pipeline


# --- R1: the gated model set is derived from config, not re-typed -------------


def _source_without_module_docstring(module: Any) -> str:
    source = Path(module.__file__).read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source), clean=False)
    return source if docstring is None else source.replace(docstring, "")


def test_community_constant_is_the_config_constant() -> None:
    # (identity, not just equality -- but config is reloaded by other tests, so
    # the reload-stable form of the claim is: nobody re-types the id in code)
    assert diarize.COMMUNITY_DIARIZE_MODEL == config.COMMUNITY_DIARIZE_MODEL
    assert config.DEFAULT_DIARIZE_MODEL == config.COMMUNITY_DIARIZE_MODEL
    assert config.DIARIZE_MODEL_ALIASES["community-1"] is config.COMMUNITY_DIARIZE_MODEL
    for module in (diarize, turnembed):
        assert _COMMUNITY_MODEL not in _source_without_module_docstring(module)


def test_gated_model_set_covers_both_bundled_pipelines() -> None:
    assert config.DEFAULT_DIARIZE_MODEL in diarize.GATED_DIARIZE_MODELS
    assert config.LEGACY_DIARIZE_MODEL in diarize.GATED_DIARIZE_MODELS
    assert set(diarize.GATED_DIARIZE_MODELS) == set(
        config.DIARIZE_MODEL_ALIASES.values()
    )


# --- R2: the no-token pre-check sees through a pinned revision ---------------


@pytest.mark.parametrize("model_id", [_LEGACY_MODEL, _COMMUNITY_MODEL])
def test_pinned_revision_of_gated_model_is_refused_without_token(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    _no_token(monkeypatch)
    _forbid_pipeline_load(monkeypatch)

    with pytest.raises(RuntimeError) as ei:
        diarize.diarize_turns(Path("nope.wav"), model=f"{model_id}@{'a' * 40}")

    assert "model-card" in str(ei.value)


def test_bare_gated_model_is_still_refused_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_token(monkeypatch)
    _forbid_pipeline_load(monkeypatch)

    with pytest.raises(RuntimeError) as ei:
        diarize.diarize_turns(Path("nope.wav"), model="community-1")

    assert "model-card" in str(ei.value)


# --- R3: gated Hub refusals are not swallowed as "no config" -----------------


def _gated_repo_error() -> Exception:
    from huggingface_hub.errors import GatedRepoError

    return GatedRepoError("Access to model X is restricted and you are not in it")


def _http_error(status: int) -> Exception:
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    response = requests.Response()
    response.status_code = status
    return HfHubHTTPError(f"{status} Client Error", response)


def test_pipeline_config_path_propagates_gated_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    def boom(*_args: object, **_kwargs: object) -> str:
        raise _gated_repo_error()

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)

    from huggingface_hub.errors import GatedRepoError

    with pytest.raises(GatedRepoError):
        diarize._pipeline_config_path(_COMMUNITY_MODEL, "hf_token")


@pytest.mark.parametrize("status", [401, 403])
def test_pipeline_config_path_propagates_unauthorized_http_error(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    import huggingface_hub

    def boom(*_args: object, **_kwargs: object) -> str:
        raise _http_error(status)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)

    from huggingface_hub.errors import HfHubHTTPError

    with pytest.raises(HfHubHTTPError):
        diarize._pipeline_config_path(_COMMUNITY_MODEL, "hf_token")


def test_pipeline_config_path_still_swallows_offline_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    def boom(*_args: object, **_kwargs: object) -> str:
        raise OSError("offline and no cached config.yaml")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)

    assert diarize._pipeline_config_path(_COMMUNITY_MODEL, "hf_token") is None


@pytest.mark.parametrize("model", [_COMMUNITY_MODEL, _LEGACY_MODEL])
def test_gated_config_download_surfaces_model_card_error(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    import huggingface_hub

    def boom(*_args: object, **_kwargs: object) -> str:
        raise _gated_repo_error()

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("gated repo must not reach pipeline construction")

    _install_fake_pyannote(monkeypatch, lambda *_a, **_k: None, Pipeline)
    monkeypatch.setattr(diarize, "_pipeline", None)
    monkeypatch.setattr(diarize, "_pipeline_model", None)

    with pytest.raises(RuntimeError) as ei:
        diarize._get_pipeline("hf_token_without_gate", model)

    message = str(ei.value)
    assert "could not load" in message
    assert "model-card" in message
    assert "Agglomerative" not in message


# --- R4: the community default names the 3.1 escape hatch -------------------


def test_gated_error_for_default_points_at_the_31_escape_hatch() -> None:
    message = str(diarize._gated_model_error(config.DEFAULT_DIARIZE_MODEL))
    assert "accept the model-card conditions" in message
    assert "--diarize-model 3.1" in message


def test_gated_error_for_legacy_has_no_escape_hatch() -> None:
    message = str(diarize._gated_model_error(config.LEGACY_DIARIZE_MODEL))
    assert "accept the model-card conditions" in message
    assert "--diarize-model 3.1" not in message


# --- R5: PLDA suppression is keyed on checkpoint shape, not model id ---------


def test_local_directory_with_31_shape_suppresses_plda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_dir = tmp_path / "diarization-3.1"
    pipeline_dir.mkdir()
    _write_pipeline_config(
        pipeline_dir / "config.yaml", clustering="AgglomerativeClustering"
    )
    probe = _PLDAProbe()
    original_get_plda = probe.get_plda
    module = _install_fake_pyannote(monkeypatch, original_get_plda)
    monkeypatch.setattr(diarize, "_embedding_load_authority", lambda *_a, **_k: None)
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", _direct_construct)

    loaded = diarize._load_pipeline(
        probe.pipeline_cls(), "hf_local", model=str(pipeline_dir)
    )

    assert loaded is not None
    assert probe.seen == [None]
    assert module.get_plda is original_get_plda


def test_shape_predicate_reads_a_local_pipeline_directory(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "mirror"
    pipeline_dir.mkdir()
    _write_pipeline_config(
        pipeline_dir / "config.yaml", clustering="AgglomerativeClustering"
    )
    assert diarize._is_agglomerative_plda_free_plan(str(pipeline_dir)) is True

    community_dir = tmp_path / "community"
    community_dir.mkdir()
    _write_pipeline_config(
        community_dir / "config.yaml", clustering="VBxClustering", plda=True
    )
    assert diarize._is_agglomerative_plda_free_plan(str(community_dir)) is False
    assert diarize._is_agglomerative_plda_free_plan("acme/not-a-path") is False


def test_mirror_id_with_31_shape_suppresses_plda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _PLDAProbe()
    _install_fake_pyannote(monkeypatch, probe.get_plda)
    monkeypatch.setattr(
        diarize,
        "_prepare_pipeline_load",
        lambda *_a, **_k: _plan(clustering="AgglomerativeClustering"),
    )
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", _direct_construct)

    loaded = diarize._load_pipeline(
        probe.pipeline_cls(),
        "hf_mirror",
        model="acme/speaker-diarization-3.1-mirror",
    )

    assert loaded is not None
    assert probe.seen == [None]


def test_community_shaped_plan_keeps_its_plda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _PLDAProbe()
    _install_fake_pyannote(monkeypatch, probe.get_plda)
    monkeypatch.setattr(
        diarize,
        "_prepare_pipeline_load",
        lambda *_a, **_k: _plan(clustering="VBxClustering", plda=True),
    )
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", _direct_construct)

    loaded = diarize._load_pipeline(
        probe.pipeline_cls(), "hf_community", model=_COMMUNITY_MODEL
    )

    assert loaded is not None
    assert probe.seen == [probe.sentinel]


def test_legacy_id_with_unverified_shape_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _PLDAProbe()
    _install_fake_pyannote(monkeypatch, probe.get_plda)
    monkeypatch.setattr(
        diarize,
        "_prepare_pipeline_load",
        lambda *_a, **_k: _plan(clustering="VBxClustering"),
    )
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", _direct_construct)

    with pytest.raises(RuntimeError, match="3\\.1.*Agglomerative"):
        diarize._load_pipeline(probe.pipeline_cls(), "hf_legacy", model=_LEGACY_MODEL)

    assert probe.seen == []


def test_legacy_id_with_verified_shape_suppresses_plda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _PLDAProbe()
    _install_fake_pyannote(monkeypatch, probe.get_plda)
    monkeypatch.setattr(
        diarize,
        "_prepare_pipeline_load",
        lambda *_a, **_k: _plan(clustering="AgglomerativeClustering"),
    )
    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", _direct_construct)

    loaded = diarize._load_pipeline(
        probe.pipeline_cls(),
        "hf_legacy",
        model=f"{_LEGACY_MODEL}@legacy-revision",
    )

    assert loaded is not None
    assert probe.seen == [None]


# --- R6: the CLI hint distinguishes a gate failure from an empty pipeline ----


def test_hint_for_gated_model_error_points_at_the_gate() -> None:
    hint = ui._hint_for(diarize._gated_model_error(config.DEFAULT_DIARIZE_MODEL))
    assert "hf auth login" in hint
    assert "--diarize-model 3.1" in hint


def test_hint_for_non_default_gate_failure_has_no_escape_hatch() -> None:
    # The way out only exists when the community-1 gate is the one refusing:
    # a failing 3.1 or custom id still gets the gate hint, minus the 3.1 clause.
    for model in (config.LEGACY_DIARIZE_MODEL, "acme/custom-diarizer"):
        hint = ui._hint_for(diarize._gated_model_error(model))
        assert "hf auth login" in hint
        assert "--diarize-model 3.1" not in hint


def test_hint_for_pipeline_abort_is_unchanged() -> None:
    assert "no speech detected" in ui._hint_for(
        RuntimeError("no speech detected in ep01.mkv")
    )
    assert "no speech detected" in ui._hint_for(
        RuntimeError("no aligned units for ep01.mkv")
    )


def test_hint_for_unrelated_runtime_error_is_empty() -> None:
    assert ui._hint_for(RuntimeError("ffprobe failed for ep01.mkv: bad data")) == ""


def test_hint_for_cuda_oom_still_wins_over_runtime_error() -> None:
    hint = ui._hint_for(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert "VOXWEAVE_MAX_CHUNK_SEC" in hint
