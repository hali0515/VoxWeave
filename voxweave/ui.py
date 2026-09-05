from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich import filesize
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text

from voxweave.progress import Reporter

# Logs/errors to stderr; result paths to stdout so `voxweave x | ...` pipelines work cleanly.
console = Console(stderr=True)

# One status palette for every command, in the terminal's own ANSI theme.
_ACTIVE = "cyan"
_SUCCESS = "green"
_WARNING = "yellow"
_ERROR = "red"
_MUTED = "dim"


class _DotBarColumn(ProgressColumn):
    """Compose-style segmented progress; a moving pulse never implies a percent."""

    def __init__(self, target_console: Console) -> None:
        super().__init__()
        self.console = target_console
        try:
            "⣿⣀⣤".encode(target_console.encoding)
            self.ascii_only = target_console.legacy_windows
        except (LookupError, UnicodeEncodeError):
            self.ascii_only = True

    def _width(self) -> int:
        return 20 if self.console.width >= 70 else 12 if self.console.width >= 45 else 6

    def get_table_column(self) -> Column:
        return Column(width=self._width() + 2, no_wrap=True)

    def render(self, task: Task) -> Text:
        width = self._width()
        full, empty = ("#", "-") if self.ascii_only else ("⣿", "⣀")
        bar = Text()
        bar.append("[", style=_MUTED)
        if task.total is None:
            position = int((task.elapsed or 0) * 6) % (width + 3) - 3
            for index in range(width):
                active = position <= index < position + 3
                bar.append(
                    full if active else empty, style=_ACTIVE if active else _MUTED
                )
        else:
            fraction = (
                1.0
                if task.total == 0
                else max(0.0, min(1.0, task.completed / task.total))
            )
            cells = fraction * width
            completed = int(cells)
            bar.append(full * completed, style=_SUCCESS if fraction == 1 else _ACTIVE)
            if completed < width:
                partial = int((cells - completed) * 8)
                glyph = (
                    (">" if partial else empty)
                    if self.ascii_only
                    else "⣀⡀⣀⣄⣤⣦⣶⣷"[partial]
                )
                bar.append(glyph, style=_ACTIVE if partial else _MUTED)
                bar.append(empty * (width - completed - 1), style=_MUTED)
        bar.append("]", style=_MUTED)
        return bar


class _WorkflowProgress(Progress):
    """Keep workflow position separate from the currently measured subtask."""

    heading: Text | None = None

    def get_renderable(self):
        tasks = super().get_renderable()
        return Group(self.heading, tasks) if self.heading is not None else tasks


class _ElapsedColumn(TimeElapsedColumn):
    def render(self, task: Task) -> Text:
        return Text(super().render(task).plain, style=_MUTED)


def install_logging(*, verbose: bool = False) -> None:
    """Attach root logger to rich, sharing the console with the progress bar."""
    import warnings

    # VOXWEAVE_OFFLINE=1 -> fully offline once everything is cached: hf_hub/transformers skip the
    # per-file HEAD revalidation + optional-file probing (chat_template / safetensors PR / etc.) they
    # do in online mode even on a cache hit. Must be set before huggingface_hub/transformers import
    # (read at import time); install_logging runs at startup, before the lazy backend imports.
    if os.environ.get("VOXWEAVE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # TRANSFORMERS_VERBOSITY must be set before the first import; setLevel after import
    # is overridden by transformers itself. Suppresses per-chunk pad_token_id notices.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", message=".*window was not provided.*")
    warnings.filterwarnings("ignore", message=".*sdp_kernel.*")
    # Wav2Vec2ForCTC emits a gradient_checkpointing deprecation on load — irrelevant for inference.
    warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
    # pyannote.audio warns at import when torchcodec cannot load (tool venvs lack
    # the CUDA NPP library it dlopens). Harmless here: diarization is always fed a
    # pre-decoded in-memory waveform dict, never pyannote's built-in decoding.
    warnings.filterwarnings(
        "ignore", message="(?s).*torchcodec is not installed correctly.*"
    )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
        force=True,
    )
    # transformers logs "Setting pad_token_id..." on every .transcribe() call via logging, not
    # warnings — suppress by setting ERROR (retains real errors, drops per-chunk noise).
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # Third-party HTTP clients log every request at INFO ("HTTP Request: GET ... 200 OK"): the
    # huggingface_hub cache revalidation + optional-file probing floods the console on each run.
    # Drop them to WARNING so only genuine problems surface (set VOXWEAVE_OFFLINE=1 to skip the
    # requests entirely once cached).
    for _noisy in (
        "httpx",
        "httpcore",
        "httpx2",
        "httpcore2",
        "huggingface_hub",
        "urllib3",
        "filelock",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


class _MofNIfKnown(ProgressColumn):
    """Renders ``x/N`` when total is known; renders nothing when total is unknown (avoids misleading ``0/?``).

    Byte tasks (``unit="B"`` field, used for model downloads) render human-readable sizes
    (``268.4 MB/913.6 MB``), and show the running count even while total is unknown.
    """

    def render(self, task: Task) -> Text:
        if task.fields.get("unit") == "B":
            done = filesize.decimal(int(task.completed))
            if task.total is None:
                return Text(done, style=_ACTIVE)
            return Text(f"{done}/{filesize.decimal(int(task.total))}", style=_ACTIVE)
        if task.total is None:
            return Text("")
        completed = min(task.completed, task.total)
        return Text(f"{int(completed)}/{int(task.total)}", style=_ACTIVE)


class RichReporter(Reporter):
    """Rich progress: a single morphing task row sharing the console with logging.

    Context manager usage::

        with RichReporter() as rep:
            process(..., reporter=rep)

    Each :meth:`stage` / :meth:`task` call replaces the active row (elapsed resets per stage):
    - :meth:`plan` / :meth:`step`: stable workflow position, separate from subtasks.
    - :meth:`stage` (indeterminate): dot pulse + spinner + elapsed, no ``x/?`` counter.
    - :meth:`task` + :meth:`advance` (countable): real progress bar + ``x/N`` + elapsed.

    Animations are disabled when not connected to a terminal.
    """

    def __init__(self) -> None:
        self._progress = _WorkflowProgress(
            SpinnerColumn(style=_ACTIVE, finished_text=Text("done", style=_SUCCESS)),
            TextColumn(
                "{task.description}",
                markup=False,
                table_column=Column(ratio=1, no_wrap=True, overflow="ellipsis"),
            ),
            _DotBarColumn(console),
            _MofNIfKnown(),
            _ElapsedColumn(),
            console=console,
            expand=True,
            transient=True,
            disable=not console.is_terminal,
        )
        self._task_id: TaskID | None = None
        self._dl_label: str | None = None
        self._steps: tuple[str, ...] = ()
        self._step_index: int | None = None
        self._last_status_at: float | None = None

    def __enter__(self) -> RichReporter:
        self._progress.start()
        self._task_id = self._progress.add_task("starting", total=None)
        # Route HF model-download byte progress into this row while the UI is live;
        # hub's own tqdm bars are silenced for the duration (they fight the Live region).
        from voxweave.runtime import set_download_reporter

        set_download_reporter(self)
        return self

    def __exit__(self, *exc: object) -> None:
        from voxweave.runtime import set_download_reporter

        set_download_reporter(None)
        self._progress.stop()

    def _switch(self, label: str, total: int | None, **fields: Any) -> None:
        # remove+add rather than reset: rich treats total=None in update as "no change",
        # so the previous stage's total bleeds through. A fresh add_task properly resets
        # to total=None (BarColumn pulse) and restarts elapsed time for this stage.
        self._dl_label = None
        self._last_status_at = None
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(label, total=total, **fields)

    def plan(self, steps: Sequence[str]) -> None:
        self._steps = tuple(steps)
        self._step_index = None
        self._progress.heading = None

    def step(self, label: str) -> None:
        if label not in self._steps:
            # A library may call a pipeline helper without an outer CLI plan.
            self.stage(label)
            return
        index = self._steps.index(label)
        if index == self._step_index:
            return
        self._step_index = index
        heading = Text(f"[{index + 1}/{len(self._steps)}] ", style=_ACTIVE)
        heading.append(label, style="bold")
        self._progress.heading = heading
        self.stage("processing")
        if not console.is_terminal:
            # Logs remain useful under redirection, without animation/ANSI or
            # thousands of per-cue updates. Result paths still go to stdout.
            console.print(heading)

    def stage(self, label: str) -> None:
        self._switch(label, None)

    def status(self, label: str) -> None:
        if self._task_id is None:
            self._switch(label, None)
        else:
            self._progress.update(self._task_id, description=label)
        if not console.is_terminal:
            now = time.monotonic()
            if self._last_status_at is None or now - self._last_status_at >= 30:
                console.print(Text(label, style=_MUTED))
                self._last_status_at = now

    def task(self, label: str, total: int) -> None:
        self._switch(label, total)

    def advance(self, n: int = 1) -> None:
        if self._task_id is not None:
            self._progress.advance(self._task_id, n)

    def download(self, label: str, done: int, total: int | None) -> None:
        # Absolute update-in-place (not remove+add): keeps elapsed running and lets total
        # grow as snapshot downloads discover more files. The row animates at the Live
        # refresh rate even when xet delivers bytes in minutes-apart bursts.
        if self._dl_label != label:
            self._switch(f"downloading {label}", total, unit="B")
            self._dl_label = label
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=done, total=total)


# Pipeline aborts that leave nothing to write (pipeline.transcribe/split).
_PIPELINE_ABORT_MARKERS = (
    "no speech",
    "no aligned",
    "no alignment",
    "no word segments",
)


def _hint_for(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "File not found, or ffmpeg is not on PATH."
    if type(exc).__module__.startswith("openai"):
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            return (
                "The endpoint rejected authentication. Check --api-key-env / "
                "[llm].api_key_env and its key; an empty value declares a keyless endpoint."
            )
        if status == 404:
            return (
                "Check --base-url and --model against the endpoint's /v1/models list."
            )
        if status in {400, 422}:
            return (
                "The endpoint rejected the request. Check its model and supported "
                "parameters; translate --reasoning-effort default uses the server default."
            )
        return "LLM API error: check the configured endpoint and network access."
    if "out of memory" in str(exc).lower():
        return (
            "GPU out of memory: lower VOXWEAVE_MAX_CHUNK_SEC or pick a smaller --model."
        )
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        if "could not load" in message and "model-card" in message:
            hint = (
                "Accept the model-card conditions on Hugging Face and authenticate "
                "(hf auth login, or VOXWEAVE_HF_TOKEN / HF_TOKEN)."
            )
            # The escape hatch only applies when the default (community-1) gate
            # is the one refusing; a failing 3.1 or custom id has no such way out.
            if "speaker-diarization-community-1" in message:
                hint += " If you only have the 3.1 gate, run with --diarize-model 3.1."
            return hint
        # Only the aborts this line actually describes; every other RuntimeError
        # carries its own message and a wrong hint is worse than none.
        if any(marker in message for marker in _PIPELINE_ABORT_MARKERS):
            return "Pipeline aborted (no speech detected or no alignment result)."
    return ""


def error_panel(exc: Exception) -> None:
    """Render exception as a red panel with a troubleshooting hint."""
    body = Text(f"{type(exc).__name__}: ", style=_ERROR)
    body.append(str(exc), style="default")
    hint = _hint_for(exc)
    if hint:
        body.append(f"\n\n{hint}", style=_MUTED)
    console.print(Panel(body, title="Error", border_style=_ERROR))


def success_panel(title: str, lines: Sequence[str]) -> None:
    """Shared completion treatment; paths and provider text are always literal."""
    console.print(
        Panel(
            Text("\n".join(lines)),
            title=Text(title, style=_SUCCESS),
            border_style=_SUCCESS,
        )
    )


def summary_panel(
    vtt_path: Path,
    *,
    separated: bool,
    debug_dir: Path | None = None,
    normalized: bool = False,
) -> None:
    """Print transcription success panel: paths, language, cue count, and flags."""
    from voxweave.pipeline import swap_ext

    vtt = Path(vtt_path)
    json_path = swap_ext(vtt, ".json")  # sibling derivation: never Path.with_suffix
    lines = [f"VTT  : {vtt}", f"JSON : {json_path}"]
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        lines.append(f"lang : {data.get('language', '?')}")
        lines.append(f"cues : {len(data.get('segments', []))}")
    except (OSError, ValueError):
        pass
    lines.append(f"sep  : {'on' if separated else 'off (--no-separate)'}")
    if normalized:
        lines.append("vol  : loudnorm applied")
    if debug_dir is not None:
        lines.append(f"debug: {debug_dir}/ (intermediate artifacts saved)")
    success_panel("Done", lines)


def translate_summary_panel(out_path: Path, *, to: str) -> None:
    """Translation success summary: translated VTT path + target language; original VTT/JSON untouched."""
    lines = [
        f"out  : {Path(out_path)}",
        f"lang : {to}",
        "original files unchanged (VTT/JSON preserved)",
    ]
    success_panel("Translation done", lines)


def correct_summary_panel(res: dict) -> None:
    """Print correction summary: applied/rejected counts, diff preview (up to 20), and next-step hint."""
    applied = res.get("applied", [])
    rejected = res.get("rejected", [])
    out = Path(res["out"])
    in_place = res.get("applied_in_place", False)
    aligned = res.get("aligned", False)
    overwritten = (
        "  (original VTT overwritten + re-aligned)"
        if aligned
        else "  (original VTT overwritten)"
    )
    head = [
        f"out   : {out}"
        + (overwritten if in_place else "  (sidecar, original VTT unchanged)"),
    ]
    if res.get("audit"):
        head.append(f"audit : {Path(res['audit'])}")
    head.append(
        f"stats : {len(applied)} applied / {len(rejected)} rejected / {res.get('n_cues', 0)} cues"
    )
    success_panel("Correction done", head)

    if applied:
        t = Table(title="Applied revisions (diff)", show_lines=False, expand=True)
        t.add_column("#", justify="right", style=_MUTED, no_wrap=True)
        t.add_column("Original", style=_ERROR)
        t.add_column("Fixed", style=_SUCCESS)
        t.add_column("Reason", style=_MUTED)
        for f in applied[:20]:
            t.add_row(
                str(f.get("i")),
                f.get("orig", ""),
                f.get("fixed", ""),
                f.get("reason", ""),
            )
        console.print(t)
        if len(applied) > 20:
            console.print(
                f"[dim]... and {len(applied) - 20} more; see audit JSON for full list[/]"
            )

    if rejected:
        reasons: dict[str, int] = {}
        for r in rejected:
            reasons[r.get("_why", "?")] = reasons.get(r.get("_why", "?"), 0) + 1
        why = "  ".join(f"{k}x{v}" for k, v in reasons.items())
        message = Text(f"Rejected {len(rejected)} (safety gate)", style=_WARNING)
        message.append(f": {why}", style="default")
        console.print(message)

    if aligned:
        nxt = "[green]Done[/]: corrections applied and timestamps re-aligned in place."
    elif in_place:
        nxt = "Next: run [bold]voxweave align[/] to reassign timestamps (text changed, timestamps need refresh)"
    else:
        nxt = f"Next: review [bold]{out.name}[/] -> [bold]voxweave correct --apply[/] to overwrite original VTT -> [bold]voxweave align[/]"
    console.print(nxt)
