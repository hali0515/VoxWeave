# tests/test_ui.py
# Error-panel hints: exception type -> actionable troubleshooting line.

from voxweave import ui
import pytest


def test_hint_for_file_not_found():
    assert "ffmpeg" in ui._hint_for(FileNotFoundError("x"))


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "--api-key-env"),
        (403, "[llm].api_key_env"),
        (404, "--model"),
        (400, "--reasoning-effort default"),
        (422, "--reasoning-effort default"),
        (500, "endpoint"),
        (None, "endpoint"),
    ],
)
def test_hint_for_openai_errors(status, expected):
    # any exception from the openai package gets an API-focused hint, without
    # importing openai here (the class is faked with the right __module__)
    exc_cls = type(
        "APIError", (Exception,), {"__module__": "openai", "status_code": status}
    )
    hint = ui._hint_for(exc_cls("request failed"))
    assert expected in hint


def test_hint_for_unknown_is_empty():
    assert ui._hint_for(ValueError("x")) == ""


def test_hint_for_cuda_oom():
    # torch's real message is "CUDA out of memory. Tried to allocate 2.00 GiB ..."
    # but we must not import torch here -- a plain RuntimeError with matching text suffices.
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    hint = ui._hint_for(exc)
    assert "VOXWEAVE_MAX_CHUNK_SEC" in hint
    assert "--model" in hint


def test_hint_for_cuda_oom_is_case_insensitive():
    exc = RuntimeError("Out Of Memory while allocating tensor")
    hint = ui._hint_for(exc)
    assert "VOXWEAVE_MAX_CHUNK_SEC" in hint
