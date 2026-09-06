from __future__ import annotations

import time
from collections.abc import Sequence


class Reporter:
    """Base class for pipeline progress callbacks.

    All methods are no-ops by default, with no rich dependency; the CLI injects
    ``RichReporter`` to render progress. The pipeline only depends on this interface,
    decoupling it from any specific renderer (library callers can omit it entirely).

    Two progress modes:
    - :meth:`stage` for indeterminate stages (decode / load model / VAD / write) -- total unknown, spinner only;
    - :meth:`task` + :meth:`advance` for countable stages (demix windows / song-skip batches / per-chunk ASR) --
      total known, renders a real ``x/N`` progress bar.

    ``chunks`` / ``chunk_done`` are semantic aliases for ``task`` / ``advance`` (legacy API, used for per-chunk ASR).

    The one piece of state the base class keeps is wall-clock accounting per step
    (:meth:`step` starts a clock, :meth:`timings` reads it) so "where did the time go"
    is answerable from any run, with or without a renderer. It produces no output.
    """

    # Timing state is created lazily: subclasses that skip ``super().__init__()``
    # (recording reporters, adapters) still get correct accounting.
    _timing_open: tuple[str, float] | None = None
    _timing_totals: dict[str, float] | None = None

    def plan(self, steps: Sequence[str]) -> None:
        """Declare the actual ordered workflow, before beginning its first step.

        Optional work belongs in the plan only when enabled. Downloads, retries,
        and other subtasks do not add steps or change the denominator.
        """

    def step(self, label: str) -> None:
        """Enter a named step from the declared plan; stage/task remain subtasks.

        Closes the clock of the previous step and starts one for ``label``.
        Overrides must call ``super().step(label)`` to keep :meth:`timings` accurate.
        """
        now = time.monotonic()
        self._close_step(now)
        self._timing_open = (label, now)

    def finish(self) -> None:
        """Close the open step's clock; safe to call with no step open or repeatedly."""
        self._close_step(time.monotonic())

    def timings(self) -> dict[str, float]:
        """Wall-clock seconds per step label, in first-entry order.

        A label entered more than once accumulates. The step still open counts up
        to now, so a snapshot taken mid-run covers the work done so far.
        """
        totals = dict(self._timing_totals or {})
        if self._timing_open is not None:
            label, started = self._timing_open
            totals[label] = totals.get(label, 0.0) + max(
                0.0, time.monotonic() - started
            )
        return totals

    def _close_step(self, now: float) -> None:
        if self._timing_open is None:
            return
        label, started = self._timing_open
        if self._timing_totals is None:
            self._timing_totals = {}
        self._timing_totals[label] = self._timing_totals.get(label, 0.0) + max(
            0.0, now - started
        )
        self._timing_open = None

    def stage(self, label: str) -> None:
        """Enter an indeterminate stage (decode / load model / VAD / re-layout / write)."""

    def status(self, label: str) -> None:
        """Update the current task's detail without resetting progress or elapsed time."""

    def task(self, label: str, total: int) -> None:
        """Start a countable stage with a known total (renders a real ``x/N`` progress bar)."""

    def advance(self, n: int = 1) -> None:
        """Advance the current countable stage by n steps."""

    def download(self, label: str, done: int, total: int | None) -> None:
        """Report cumulative byte progress for a model download.

        Called repeatedly with the running byte count (``done``) and the expected size
        (``total``; ``None`` while unknown). Unlike :meth:`task`/:meth:`advance` this is
        absolute, not incremental -- xet/parallel downloads deliver from worker threads
        in bursts, and absolute counts stay correct regardless of delivery order.
        """

    def chunks(self, total: int) -> None:
        """Signal that the number of chunks to process is known; begin per-chunk progress (alias for ``task``)."""
        self.task("per-chunk ASR+align", total)

    def chunk_done(self) -> None:
        """Signal that one chunk (ASR + alignment) is complete (alias for ``advance``)."""
        self.advance(1)
