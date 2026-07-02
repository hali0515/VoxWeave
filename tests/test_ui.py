# tests/test_ui.py
# Error-panel hints: exception type -> actionable troubleshooting line.

from voxweave import ui


def test_hint_for_file_not_found():
    assert "ffmpeg" in ui._hint_for(FileNotFoundError("x"))


def test_hint_for_openai_errors():
    # any exception from the openai package gets an API-focused hint, without
    # importing openai here (the class is faked with the right __module__)
    exc_cls = type("AuthenticationError", (Exception,), {"__module__": "openai"})
    hint = ui._hint_for(exc_cls("401"))
    assert "OPENAI_API_KEY" in hint


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
