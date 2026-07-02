from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from rich import filesize
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from voxweave.progress import Reporter

# Logs/errors to stderr; result paths to stdout so `voxweave x | ...` pipelines work cleanly.
console = Console(stderr=True)


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
    for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
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
                return Text(done, style="cyan")
            return Text(f"{done}/{filesize.decimal(int(task.total))}", style="cyan")
        if task.total is None:
            return Text("")
        return Text(f"{int(task.completed)}/{int(task.total)}", style="cyan")


class RichReporter(Reporter):
    """Rich progress: a single morphing task row sharing the console with logging.

    Context manager usage::

        with RichReporter() as rep:
            process(..., reporter=rep)

    Each :meth:`stage` / :meth:`task` call replaces the active row (elapsed resets per stage):
    - :meth:`stage` (indeterminate): pulse bar + spinner + elapsed, no ``x/?`` counter.
    - :meth:`task` + :meth:`advance` (countable): real progress bar + ``x/N`` + elapsed.

    Animations are disabled when not connected to a terminal.
    """

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            _MofNIfKnown(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            disable=not console.is_terminal,
        )
        self._task_id: TaskID | None = None
        self._dl_label: str | None = None

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
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(label, total=total, **fields)

    def stage(self, label: str) -> None:
        self._switch(label, None)

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


def _hint_for(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "File not found, or ffmpeg is not on PATH."
    if type(exc).__module__.startswith("openai"):
        return "OpenAI API error: check OPENAI_API_KEY and network access."
    if "out of memory" in str(exc).lower():
        return (
            "GPU out of memory: lower VOXWEAVE_MAX_CHUNK_SEC or pick a smaller --model."
        )
    if isinstance(exc, RuntimeError):
        return "Pipeline aborted (no speech detected or no alignment result)."
    return ""


def error_panel(exc: Exception) -> None:
    """Render exception as a red panel with a troubleshooting hint."""
    body = f"[red]{type(exc).__name__}[/]: {exc}"
    hint = _hint_for(exc)
    if hint:
        body += f"\n\n[dim]{hint}[/]"
    console.print(Panel(body, title="Error", border_style="red"))


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
    console.print(Panel("\n".join(lines), title="[green]Done[/]", border_style="green"))


def translate_summary_panel(out_path: Path, *, to: str) -> None:
    """Translation success summary: translated VTT path + target language; original VTT/JSON untouched."""
    lines = [
        f"out  : {Path(out_path)}",
        f"lang : {to}",
        "original files unchanged (VTT/JSON preserved)",
    ]
    console.print(
        Panel(
            "\n".join(lines), title="[green]Translation done[/]", border_style="green"
        )
    )


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
    console.print(
        Panel("\n".join(head), title="[green]Correction done[/]", border_style="green")
    )

    if applied:
        t = Table(title="Applied revisions (diff)", show_lines=False, expand=True)
        t.add_column("#", justify="right", style="dim", no_wrap=True)
        t.add_column("Original", style="red")
        t.add_column("Fixed", style="green")
        t.add_column("Reason", style="dim")
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
        console.print(f"[yellow]Rejected {len(rejected)} (safety gate)[/]: {why}")

    if aligned:
        nxt = "[green]Done[/]: corrections applied and timestamps re-aligned in place."
    elif in_place:
        nxt = "Next: run [bold]voxweave align[/] to reassign timestamps (text changed, timestamps need refresh)"
    else:
        nxt = f"Next: review [bold]{out.name}[/] -> [bold]voxweave correct --apply[/] to overwrite original VTT -> [bold]voxweave align[/]"
    console.print(nxt)
