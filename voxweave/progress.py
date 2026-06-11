from __future__ import annotations


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
    """

    def stage(self, label: str) -> None:
        """Enter an indeterminate stage (decode / load model / VAD / re-layout / write)."""

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
