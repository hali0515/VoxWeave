"""The bundled local semantic worker is removed; only an endpoint can serve it.

The 0.12 release only deprecated the bundled FP8 worker.  Measurement then put
its pairwise preference signal at the FP8 quantization floor, so a second
Torch/CUDA runtime bought no subtitle-quality gain and the worker is now gone.
What remains is a user-run OpenAI-compatible server plus the deterministic
splitter, and a missing server is a configuration error: it must abort at engine
construction rather than after a full transcription.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from voxweave import pipeline, semantic_breaks, ui
from voxweave.cli import cli, cmd_transcribe
from voxweave.semantic_breaks import (
    MISSING_SEMANTIC_BACKEND_MESSAGE,
    BoundaryTask,
    OpenAICompatibleSelector,
    SemanticBackendUnavailable,
    SemanticBreakEngine,
)


@pytest.fixture(autouse=True)
def _unconfigured_endpoint(monkeypatch):
    """Tests that want the endpoint set it explicitly; the default is "absent"."""

    monkeypatch.delenv("VOXWEAVE_SEMANTIC_BASE_URL", raising=False)


def _task():
    return BoundaryTask(
        atoms=("欢迎", "收看", "今天", "的", "节目"),
        candidate_indices=(1, 2, 3, 4),
        language="zh",
        fallback_indices=(2,),
        required_indices=(),
        pauses_ms={2: 260, 4: 410},
        target_chars=None,
        max_segment_chars=None,
    )


def test_bundled_worker_module_and_lock_are_gone():
    assert importlib.util.find_spec("voxweave.semantic_worker") is None
    package_dir = Path(semantic_breaks.__file__).parent
    assert not (package_dir / "semantic_worker.py").exists()
    assert not (package_dir / "semantic_worker.py.lock").exists()
    assert not hasattr(semantic_breaks, "LocalTransformersSelector")


def test_missing_endpoint_fails_fast_with_an_actionable_message():
    with pytest.raises(SemanticBackendUnavailable) as excinfo:
        SemanticBreakEngine()

    message = str(excinfo.value)
    assert message == MISSING_SEMANTIC_BACKEND_MESSAGE
    # Says what happened, what to do, and how to opt out.
    assert "removed" in message
    assert "0.12" in message
    assert "VOXWEAVE_SEMANTIC_BASE_URL" in message
    assert "--semantic-split" in message
    assert "deterministic" in message


def test_blank_base_url_is_still_an_unconfigured_backend(monkeypatch):
    monkeypatch.setenv("VOXWEAVE_SEMANTIC_BASE_URL", "   ")

    with pytest.raises(SemanticBackendUnavailable):
        SemanticBreakEngine()


def test_module_import_and_release_never_build_a_backend():
    """No import-time engine: a missing endpoint must not break unrelated imports."""

    # Reaching this line already proves the import itself resolved no backend.
    semantic_breaks.release_semantic_model()
    assert semantic_breaks._DEFAULT_ENGINE is None


def test_configured_endpoint_selects_the_openai_backend(monkeypatch, caplog):
    monkeypatch.setenv("VOXWEAVE_SEMANTIC_BASE_URL", "http://127.0.0.1:8000/v1")

    with caplog.at_level("WARNING", logger="voxweave"):
        engine = SemanticBreakEngine()

    assert isinstance(engine.selector, OpenAICompatibleSelector)
    assert [r for r in caplog.records if "deprecat" in r.getMessage()] == []


def test_engine_still_selects_boundaries_against_a_stubbed_endpoint():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            payload = json.loads(kwargs["messages"][-1]["content"])
            results = [
                {"id": item["id"], "breaks": [item["candidate_indices"][0]]}
                for item in payload["tasks"]
            ]
            message = SimpleNamespace(content=json.dumps({"results": results}))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    engine = SemanticBreakEngine(OpenAICompatibleSelector(client=client))

    (decision,) = engine.choose([_task()])

    assert decision.source == "model"
    assert decision.break_indices == (1,)
    assert len(calls) == 1


def test_process_refuses_before_any_audio_or_model_work(monkeypatch, tmp_path):
    def never(*_args, **_kwargs):
        raise AssertionError("no transcription may start without a semantic backend")

    monkeypatch.setattr(pipeline, "transcribe", never)

    with pytest.raises(SemanticBackendUnavailable):
        pipeline.process(tmp_path / "absent.mkv", semantic_split=True)


def test_semantic_split_off_needs_no_backend():
    """The deterministic default must stay reachable with no endpoint at all."""

    assert pipeline._make_semantic_engine(False) is None


def test_error_panel_hint_points_at_the_endpoint():
    hint = ui._hint_for(SemanticBackendUnavailable(MISSING_SEMANTIC_BACKEND_MESSAGE))

    assert "VOXWEAVE_SEMANTIC_BASE_URL" in hint
    assert "--semantic-split" in hint
    # The generic RuntimeError hint would blame the audio instead.
    assert "no speech detected" not in hint


@pytest.mark.parametrize(
    "command",
    [cmd_transcribe, cli.commands["split"]],
    ids=["transcribe", "split"],
)
def test_semantic_split_help_points_at_the_endpoint(command):
    help_text = next(p.help or "" for p in command.params if p.name == "semantic_split")

    assert "VOXWEAVE_SEMANTIC_BASE_URL" in help_text
    assert "OpenAI-compatible" in help_text
    assert "deprecated" not in help_text
    assert "bundled" not in help_text
