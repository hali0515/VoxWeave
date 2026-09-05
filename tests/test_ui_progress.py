"""The same truthful workflow and terminal treatment serves every command."""

import io

import pytest
from rich.cells import cell_len
from rich.console import Console

from voxweave import ui
from voxweave.progress import Reporter


def _reporter(monkeypatch, *, width=80, terminal=True):
    output = io.StringIO()
    console = Console(
        file=output, force_terminal=terminal, width=width, color_system=None
    )
    monkeypatch.setattr(ui, "console", console)
    return ui.RichReporter(), console, output


def _frame(rep, console, output):
    output.seek(0)
    output.truncate()
    console.print(rep._progress.get_renderable())
    return output.getvalue()


def test_base_reporter_workflow_methods_are_optional():
    rep = Reporter()
    rep.plan(["read", "translate", "write"])
    rep.step("read")
    rep.status("reading subtitle")


@pytest.mark.parametrize("width", [35, 50, 80, 120])
def test_workflow_and_subtask_are_legible_at_terminal_widths(monkeypatch, width):
    rep, console, output = _reporter(monkeypatch, width=width)
    rep.plan(["Read subtitles", "Translate", "Write subtitles"])
    rep.step("Translate")
    rep.task("subtitle cues", 100)
    rep.advance(40)
    frame = _frame(rep, console, output)
    assert "[2/3] Translate" in frame
    assert "40/100" in frame
    assert "⣿" in frame and "⣀" in frame
    assert all(cell_len(line) <= width for line in frame.splitlines())


def test_downloads_and_retries_do_not_inflate_workflow(monkeypatch):
    rep, console, output = _reporter(monkeypatch)
    rep.plan(["Read", "Translate", "Write"])
    rep.step("Translate")
    rep.download("model", 100, 200)
    frame = _frame(rep, console, output)
    assert "[2/3] Translate" in frame and "100 bytes/200 bytes" in frame
    rep.stage("retry translation")
    frame = _frame(rep, console, output)
    assert "[2/3] Translate" in frame
    assert "200 bytes" not in frame and "0/?" not in frame
    assert "%" not in frame
    rep.step("Write")
    assert "[3/3] Write" in _frame(rep, console, output)


def test_indeterminate_task_has_no_invented_total(monkeypatch):
    rep, console, output = _reporter(monkeypatch)
    rep.plan(["Prepare", "Encode"])
    rep.step("Encode")
    rep.stage("waiting for encoder")
    task = rep._progress.tasks[0]
    assert task.total is None
    assert ui._MofNIfKnown().render(task).plain == ""
    frame = _frame(rep, console, output)
    assert "[2/2] Encode" in frame
    assert "%" not in frame and "0/?" not in frame


def test_status_keeps_elapsed_and_countable_progress(monkeypatch):
    rep, _, _ = _reporter(monkeypatch)
    rep.task("encoding", 200)
    rep.advance(60)
    task = rep._progress.tasks[0]
    start = task.start_time
    rep.status("encoded 60 frames, 00:00:02")
    assert rep._progress.tasks[0] is task
    assert task.start_time == start and task.completed == 60
    assert task.description == "encoded 60 frames, 00:00:02"


def test_non_tty_prints_steps_once_and_throttles_detail(monkeypatch):
    rep, _, output = _reporter(monkeypatch, terminal=False)
    now = [100.0]
    monkeypatch.setattr(ui.time, "monotonic", lambda: now[0])
    rep.plan(["Read", "Encode"])
    rep.step("Read")
    rep.step("Read")
    rep.step("Encode")
    rep.status("encoded 1 frame")
    rep.status("encoded 2 frames")
    now[0] += 30
    rep.status("encoded 100 frames")
    assert output.getvalue().splitlines() == [
        "[1/2] Read",
        "[2/2] Encode",
        "encoded 1 frame",
        "encoded 100 frames",
    ]
    assert rep._progress.disable is True
    assert "\x1b" not in output.getvalue()


def test_ascii_terminal_uses_segmented_fallback(monkeypatch):
    rep, console, _ = _reporter(monkeypatch)
    rep.task("encoding", 10)
    rep.advance(5)
    column = ui._DotBarColumn(console)
    column.ascii_only = True
    assert column.render(rep._progress.tasks[0]).plain == "[##########----------]"


def test_complete_bar_and_elapsed_follow_shared_status_palette(monkeypatch):
    rep, console, _ = _reporter(monkeypatch)
    rep.task("encoding", 10)
    rep.advance(10)
    task = rep._progress.tasks[0]
    bar = ui._DotBarColumn(console).render(task)
    assert "⣀" not in bar.plain
    assert any(span.style == ui._SUCCESS for span in bar.spans)
    assert ui._ElapsedColumn().render(task).style == ui._MUTED


def test_stream_retry_does_not_display_more_than_the_total(monkeypatch):
    rep, _, _ = _reporter(monkeypatch)
    rep.task("subtitle cues", 2)
    rep.advance(3)
    assert ui._MofNIfKnown().render(rep._progress.tasks[0]).plain == "2/2"


def test_summary_and_errors_preserve_literal_bracketed_values(monkeypatch):
    _, _, output = _reporter(monkeypatch, terminal=False)
    ui.success_panel("Export done", ["/tmp/[draft].vtt"])
    ui.error_panel(ValueError("invalid [llm].reasoning_effort"))
    assert "/tmp/[draft].vtt" in output.getvalue()
    assert "[llm].reasoning_effort" in output.getvalue()
