"""Reporter interface contract + pipeline progress bridge (no rich / no models)."""

from voxweave import pipeline, progress
from voxweave.progress import Reporter


class _Clock:
    """Deterministic stand-in for ``time.monotonic``."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _RecordingReporter(Reporter):
    """Records task/advance calls to verify countable-stage bridging."""

    def __init__(self) -> None:
        self.tasks: list[tuple[str, int]] = []
        self.advances = 0

    def task(self, label: str, total: int) -> None:
        self.tasks.append((label, total))

    def advance(self, n: int = 1) -> None:
        self.advances += n


def test_base_reporter_methods_are_noops():
    # base class is all no-ops, no rich dependency; library callers can pass a bare Reporter
    rep = Reporter()
    rep.stage("x")
    rep.task("y", 5)
    rep.advance()
    rep.chunks(3)
    rep.chunk_done()  # must not raise


def test_chunks_chunk_done_delegate_to_task_advance():
    rep = _RecordingReporter()
    rep.chunks(4)
    rep.chunk_done()
    rep.chunk_done()
    assert rep.tasks == [("per-chunk ASR+align", 4)]
    assert rep.advances == 2


def test_progress_bridge_starts_task_once_then_advances():
    # backend/songdet sequential callbacks (done=1,2,3, total=3) -> first call creates task, subsequent calls advance(1)
    rep = _RecordingReporter()
    cb = pipeline._progress_bridge(rep, "人声分离 (Roformer)")
    cb(1, 3)
    cb(2, 3)
    cb(3, 3)
    assert rep.tasks == [("人声分离 (Roformer)", 3)]  # task created only once
    assert rep.advances == 3  # +1 per window, reaches 3/3


def test_base_reporter_accumulates_wall_time_per_step(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(progress.time, "monotonic", clock)
    rep = Reporter()
    assert rep.timings() == {}
    rep.finish()  # nothing open: no-op
    assert rep.timings() == {}

    rep.step("prepare audio")
    clock.now += 12.5
    rep.step("transcribe and align")
    clock.now += 30.0
    rep.step("prepare audio")  # re-entering a label accumulates onto it
    clock.now += 2.5
    rep.finish()

    assert rep.timings() == {"prepare audio": 15.0, "transcribe and align": 30.0}
    assert list(rep.timings()) == ["prepare audio", "transcribe and align"]
    clock.now += 100.0
    rep.finish()  # closing twice adds nothing
    assert rep.timings() == {"prepare audio": 15.0, "transcribe and align": 30.0}


def test_timings_snapshot_counts_the_open_step_without_double_counting(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(progress.time, "monotonic", clock)
    rep = Reporter()
    rep.step("read")
    clock.now += 3.0
    assert rep.timings() == {"read": 3.0}
    clock.now += 2.0
    assert rep.timings() == {"read": 5.0}
    rep.step("write")
    clock.now += 1.0
    assert rep.timings() == {"read": 5.0, "write": 1.0}


def test_subclass_without_super_init_still_answers_timings():
    rep = _RecordingReporter()  # never calls super().__init__()
    assert rep.timings() == {}
    rep.finish()
    assert rep.timings() == {}


def test_nested_reporter_keeps_the_clock_on_the_enclosing_step(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(progress.time, "monotonic", clock)
    parent = Reporter()
    nested = pipeline._NestedReporter(parent)
    parent.step("check and refresh timing")
    clock.now += 4.0
    nested.plan(["read subtitles", "align subtitles"])
    nested.step("read subtitles")
    clock.now += 6.0
    nested.step("align subtitles")
    clock.now += 10.0
    assert nested.timings() == {"check and refresh timing": 20.0}

    nested.finish()
    clock.now += 50.0
    assert parent.timings() == {"check and refresh timing": 20.0}
    assert nested.timings() == parent.timings()
