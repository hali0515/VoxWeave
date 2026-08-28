"""Invocation-scoped runtime evidence for the public align AO schedule.

The recorder is deliberately inert unless a caller explicitly captures one public
invocation.  Production operation sites, rather than a declared phase tuple, emit
the events.  This makes the resulting trace useful to an independent oracle while
keeping observation unable to select an engine or alter command behaviour.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Literal


AlignRuntimeState = Literal["started", "completed", "failed"]


@dataclass(frozen=True)
class AlignRuntimeEvent:
    ordinal: int
    phase: str
    activity: str
    state: AlignRuntimeState

    def as_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "phase": self.phase,
            "activity": self.activity,
            "state": self.state,
        }


@dataclass(frozen=True)
class AlignRuntimeTrace:
    route_kind: str | None
    engine_family: str | None
    events: tuple[AlignRuntimeEvent, ...]

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "route_kind": self.route_kind,
            "engine_family": self.engine_family,
            "events": [event.as_record() for event in self.events],
        }


class _TraceState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.route_kind: str | None = None
        self.engine_family: str | None = None
        self.events: list[AlignRuntimeEvent] = []
        self.closed = False

    def bind_identity(self, route_kind: str, engine_family: str) -> None:
        if not route_kind or not engine_family:
            raise ValueError("align runtime identity values must be nonempty")
        with self.lock:
            if self.closed:
                raise RuntimeError("align runtime capture is closed")
            if self.route_kind is not None and self.route_kind != route_kind:
                raise RuntimeError("align runtime route identity changed")
            if self.engine_family is not None and self.engine_family != engine_family:
                raise RuntimeError("align runtime engine identity changed")
            self.route_kind = route_kind
            self.engine_family = engine_family

    def append(self, phase: str, activity: str, state: AlignRuntimeState) -> None:
        if (
            len(phase) != 5
            or not phase.startswith("AO-")
            or not phase[3:].isdigit()
            or not 1 <= int(phase[3:]) <= 25
        ):
            raise ValueError("align runtime phase must be AO-01 through AO-25")
        if not activity:
            raise ValueError("align runtime activity must be nonempty")
        with self.lock:
            if self.closed:
                raise RuntimeError("align runtime capture is closed")
            self.events.append(
                AlignRuntimeEvent(len(self.events), phase, activity, state)
            )

    def snapshot(self) -> AlignRuntimeTrace:
        with self.lock:
            return AlignRuntimeTrace(
                self.route_kind,
                self.engine_family,
                tuple(self.events),
            )


class AlignRuntimeTraceCapture:
    """Read-only handle retained after its capture context detaches."""

    def __init__(self, state: _TraceState) -> None:
        self._state = state

    def snapshot(self) -> AlignRuntimeTrace:
        return self._state.snapshot()


_ACTIVE_TRACE: ContextVar[_TraceState | None] = ContextVar(
    "voxweave_align_runtime_trace",
    default=None,
)


@contextmanager
def capture_align_runtime_trace() -> Iterator[AlignRuntimeTraceCapture]:
    """Capture actual AO activities for one public align invocation.

    The handle remains readable if the invocation raises.  Nested capture in the
    same context is rejected so one trace cannot ambiguously contain two owners.
    """

    if _ACTIVE_TRACE.get() is not None:
        raise RuntimeError("align runtime capture is already active")
    state = _TraceState()
    token: Token[_TraceState | None] = _ACTIVE_TRACE.set(state)
    capture = AlignRuntimeTraceCapture(state)
    try:
        yield capture
    finally:
        _ACTIVE_TRACE.reset(token)
        with state.lock:
            state.closed = True


def bind_align_runtime_identity(*, route_kind: str, engine_family: str) -> None:
    """Bind genuine issued-context identity to an active trace, if any."""

    state = _ACTIVE_TRACE.get()
    if state is not None:
        state.bind_identity(route_kind, engine_family)


@contextmanager
def align_runtime_activity(phase: str, activity: str) -> Iterator[None]:
    """Record the exact lifecycle of one production operation when observed."""

    state = _ACTIVE_TRACE.get()
    if state is None:
        yield
        return
    state.append(phase, activity, "started")
    try:
        yield
    except BaseException:
        state.append(phase, activity, "failed")
        raise
    else:
        state.append(phase, activity, "completed")


__all__ = [
    "AlignRuntimeEvent",
    "AlignRuntimeTrace",
    "AlignRuntimeTraceCapture",
    "align_runtime_activity",
    "bind_align_runtime_identity",
    "capture_align_runtime_trace",
]
