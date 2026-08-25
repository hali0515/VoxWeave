"""The bundled local semantic worker is deprecated; only that path may warn.

The audit found no measured subtitle-quality gain for the bundled FP8 worker, so
it is scheduled for removal.  This release only warns: behavior is unchanged, a
user-managed OpenAI-compatible endpoint stays first class, and the deterministic
splitter (the default) must never see the notice.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from voxweave import semantic_breaks
from voxweave.cli import cli, cmd_transcribe
from voxweave.semantic_breaks import (
    LOCAL_BACKEND_DEPRECATION_MESSAGE,
    OpenAICompatibleSelector,
    SemanticBreakEngine,
)


class StubLocalSelector:
    """Inert stand-in: the real selector must not be constructed under test."""

    instances: ClassVar[list[StubLocalSelector]] = []

    def __init__(self, *, timeout: float = 0.0, **_kwargs: object) -> None:
        self.timeout = timeout
        StubLocalSelector.instances.append(self)

    def select(self, model_id, messages, *, max_new_tokens):  # pragma: no cover
        raise AssertionError("deprecation tests never reach the worker protocol")

    def release(self) -> None:
        return None


@pytest.fixture
def local_selector(monkeypatch):
    """Route the default selector to the stub and re-arm the once-per-process flag."""

    StubLocalSelector.instances = []
    monkeypatch.setattr(semantic_breaks, "LocalTransformersSelector", StubLocalSelector)
    monkeypatch.setattr(semantic_breaks, "_local_backend_deprecation_warned", False)
    return StubLocalSelector


def _deprecation_records(caplog):
    return [r for r in caplog.records if "deprecated" in r.getMessage()]


def test_local_backend_warns_once_per_process(local_selector, monkeypatch, caplog):
    monkeypatch.delenv("VOXWEAVE_SEMANTIC_BASE_URL", raising=False)

    with caplog.at_level("WARNING", logger="voxweave"):
        engines = [SemanticBreakEngine() for _ in range(3)]

    records = _deprecation_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert message == LOCAL_BACKEND_DEPRECATION_MESSAGE
    assert "will be removed in a future release" in message
    assert "VOXWEAVE_SEMANTIC_BASE_URL" in message
    assert "deterministic" in message
    # Warning only: every engine still gets the local selector it asked for.
    assert len(local_selector.instances) == 3
    assert all(isinstance(e.selector, StubLocalSelector) for e in engines)


def test_blank_base_url_still_takes_the_deprecated_local_path(
    local_selector, monkeypatch, caplog
):
    monkeypatch.setenv("VOXWEAVE_SEMANTIC_BASE_URL", "   ")

    with caplog.at_level("WARNING", logger="voxweave"):
        engine = SemanticBreakEngine()

    assert isinstance(engine.selector, StubLocalSelector)
    assert len(_deprecation_records(caplog)) == 1


def test_endpoint_backend_is_warning_free(local_selector, monkeypatch, caplog):
    monkeypatch.setenv("VOXWEAVE_SEMANTIC_BASE_URL", "http://127.0.0.1:8000/v1")

    with caplog.at_level("WARNING", logger="voxweave"):
        engine = SemanticBreakEngine()

    assert isinstance(engine.selector, OpenAICompatibleSelector)
    assert _deprecation_records(caplog) == []
    assert local_selector.instances == []


def test_explicit_selector_never_warns(local_selector, monkeypatch, caplog):
    monkeypatch.delenv("VOXWEAVE_SEMANTIC_BASE_URL", raising=False)

    with caplog.at_level("WARNING", logger="voxweave"):
        engine = SemanticBreakEngine(selector=OpenAICompatibleSelector())

    assert isinstance(engine.selector, OpenAICompatibleSelector)
    assert _deprecation_records(caplog) == []
    assert local_selector.instances == []


@pytest.mark.parametrize(
    "command",
    [cmd_transcribe, cli.commands["split"]],
    ids=["transcribe", "split"],
)
def test_semantic_split_help_points_at_the_endpoint(command):
    help_text = next(p.help or "" for p in command.params if p.name == "semantic_split")
    assert "local bundled model deprecated" in help_text
    assert "VOXWEAVE_SEMANTIC_BASE_URL" in help_text
