# tests/test_ui.py
# Error-panel hints: exception type -> actionable troubleshooting line.

import io
import json

import pytest
from rich.console import Console

from voxweave import ui


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_hint_for_missing_ffmpeg_names_the_tool(tool):
    exc = FileNotFoundError(2, "No such file or directory", tool)
    assert ui._hint_for(exc) == f"{tool} is not on PATH: install ffmpeg."


def test_hint_for_missing_sibling_json_suggests_transcribing():
    exc = FileNotFoundError(2, "No such file or directory", "episode.json")
    assert "transcribe the media first" in ui._hint_for(exc)


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("x"),
        FileNotFoundError(2, "No such file or directory", "episode.mkv"),
    ],
)
def test_hint_for_other_missing_files_does_not_blame_ffmpeg(exc):
    assert ui._hint_for(exc) == "File not found."


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


def test_hint_for_partial_translation_points_at_resume_and_allow_partial():
    from voxweave.translate import PartialTranslationError

    hint = ui._hint_for(PartialTranslationError([3, 4], 10))
    assert "Rerun the same command" in hint
    assert "--allow-partial" in hint


def test_hint_for_incomplete_response_points_at_the_server():
    from voxweave.translate import IncompleteResponse

    hint = ui._hint_for(IncompleteResponse("length", "partial text"))
    assert "finish_reason" in hint
    assert "json_object" in hint


def test_hint_for_cuda_oom():
    # torch's real message is "CUDA out of memory. Tried to allocate 2.00 GiB ..."
    # but we must not import torch here -- a plain RuntimeError with matching text suffices.
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    hint = ui._hint_for(exc)
    assert "VOXWEAVE_MAX_CHUNK_SEC" in hint
    assert "--asr-model" in hint


def test_hint_for_cuda_oom_is_case_insensitive():
    exc = RuntimeError("Out Of Memory while allocating tensor")
    hint = ui._hint_for(exc)
    assert "VOXWEAVE_MAX_CHUNK_SEC" in hint


def _capture(monkeypatch):
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=200, color_system=None)
    monkeypatch.setattr(ui, "console", console)
    return output


def test_error_panel_names_the_nearest_builtin_class(monkeypatch):
    output = _capture(monkeypatch)

    class Phase2DataError(ValueError):
        pass

    ui.error_panel(Phase2DataError("bad store"))
    ui.error_panel(json.JSONDecodeError("Expecting value", "x", 0))
    ui.error_panel(KeyError("k"))
    text = output.getvalue()
    assert "Phase2DataError" not in text
    assert "JSONDecodeError" not in text
    assert "ValueError: bad store" in text
    assert "ValueError: Expecting value" in text
    assert "KeyError: 'k'" in text


def test_summary_panel_reports_separation_off_without_naming_a_flag(
    monkeypatch, tmp_path
):
    output = _capture(monkeypatch)
    ui.summary_panel(tmp_path / "episode.vtt", separated=False)
    assert "sep  : off" in output.getvalue()
    assert "--no-separate" not in output.getvalue()


def _correct_result(tmp_path, **overrides):
    res = {
        "out": tmp_path / "[draft] ep.asrfix.vtt",
        "audit": tmp_path / "audit.json",
        "applied": [],
        "rejected": [],
        "n_cues": 3,
        "applied_in_place": False,
        "aligned": False,
    }
    res.update(overrides)
    return res


def test_correct_summary_keeps_names_and_llm_text_literal(monkeypatch, tmp_path):
    output = _capture(monkeypatch)
    applied = [
        {"i": 1, "orig": "[/laughs] hi", "fixed": "[bold]Hi[/bold]", "reason": "[x]"}
    ]
    ui.correct_summary_panel(_correct_result(tmp_path, applied=applied))
    text = output.getvalue()
    assert "[/laughs] hi" in text
    assert "[bold]Hi[/bold]" in text
    assert "[draft] ep.asrfix.vtt" in text


def test_correct_summary_sidecar_hint_explains_accept_and_apply(monkeypatch, tmp_path):
    output = _capture(monkeypatch)
    ui.correct_summary_panel(_correct_result(tmp_path))
    text = " ".join(output.getvalue().split())
    assert "Next: review [draft] ep.asrfix.vtt" in text
    assert "replace the VTT with it and run voxweave align" in text
    assert "rerun with --apply" in text
    assert "re-aligns automatically" in text


def test_correct_summary_apply_without_changes_says_timing_unchanged(
    monkeypatch, tmp_path
):
    output = _capture(monkeypatch)
    ui.correct_summary_panel(
        _correct_result(
            tmp_path, out=tmp_path / "ep.vtt", audit=None, applied_in_place=True
        )
    )
    text = output.getvalue()
    assert "No corrections applied; timestamps unchanged." in text
    assert "voxweave align" not in text


def test_correct_summary_apply_no_align_suggests_align(monkeypatch, tmp_path):
    output = _capture(monkeypatch)
    applied = [{"i": i, "orig": "a", "fixed": "b", "reason": ""} for i in range(21)]
    ui.correct_summary_panel(
        _correct_result(
            tmp_path,
            out=tmp_path / "ep.vtt",
            audit=None,
            applied=applied,
            applied_in_place=True,
        )
    )
    text = output.getvalue()
    assert "Next: run voxweave align" in text
    assert "... and 1 more" in text
    assert "audit JSON" not in text
