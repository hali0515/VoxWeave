"""HF download progress bridge: hub tqdm bars silenced + byte counts forwarded to the Reporter."""

import importlib

import pytest

from voxweave import runtime
from voxweave.progress import Reporter

hub_tqdm_mod = pytest.importorskip("huggingface_hub.utils.tqdm")


class _RecordingReporter(Reporter):
    """Records download() deliveries (cumulative byte counts)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []

    def download(self, label: str, done: int, total: int | None) -> None:
        self.calls.append((label, done, total))


@pytest.fixture
def reporter():
    rep = _RecordingReporter()
    runtime.set_download_reporter(rep)
    yield rep
    runtime.set_download_reporter(None)


def _make_bar(cls, **kw):
    # mimic hub's _get_progress_bar_context construction kwargs
    defaults = dict(unit="B", unit_scale=True, total=None, initial=0, desc="f")
    defaults.update(kw)
    return cls(**defaults)


def test_no_reporter_yields_none_and_leaves_hub_untouched():
    runtime.set_download_reporter(None)
    original = hub_tqdm_mod.tqdm
    with runtime._bridged_bars("x") as bridge:
        assert bridge is None
        assert hub_tqdm_mod.tqdm is original


def test_bridge_forwards_cumulative_bytes(reporter):
    with runtime._bridged_bars("model.ckpt") as bridge:
        bar = _make_bar(bridge, total=100)
        bar.update(30)
        bar.update(70)
    assert reporter.calls == [
        ("model.ckpt", 0, 100),
        ("model.ckpt", 30, 100),
        ("model.ckpt", 100, 100),
    ]


def test_bridge_never_renders(reporter):
    with runtime._bridged_bars("model.ckpt") as bridge:
        bar = _make_bar(bridge, total=100, disable=False)  # hub may pass disable=False
        assert bar.disable  # forced off: the Reporter row is the only display
        bar.update(10)


def test_bridge_aggregates_across_files(reporter):
    # snapshot downloads run several per-file bars concurrently; the row shows the sum
    with runtime._bridged_bars("repo") as bridge:
        a = _make_bar(bridge, total=60, desc="a")
        b = _make_bar(bridge, total=40, desc="b")
        a.update(10)
        b.update(20)
    assert reporter.calls[-1] == ("repo", 30, 100)


def test_bridge_total_unknown_until_all_bars_know_it(reporter):
    with runtime._bridged_bars("repo") as bridge:
        bar = _make_bar(bridge, total=None)
        bar.update(10)
        assert reporter.calls[-1] == ("repo", 10, None)


def test_non_byte_bars_are_silenced_but_not_reported(reporter):
    # snapshot's outer "Fetching N files" bar has unit="it" -- must not pollute byte progress
    with runtime._bridged_bars("repo") as bridge:
        bar = bridge(total=8, desc="Fetching 8 files", unit="it")
        bar.update(1)
        assert bar.disable
    assert reporter.calls == []


def test_module_global_restored_after_context(reporter):
    original = hub_tqdm_mod.tqdm
    with runtime._bridged_bars("x") as bridge:
        assert hub_tqdm_mod.tqdm is bridge
        assert issubclass(bridge, original)
    assert hub_tqdm_mod.tqdm is original


def test_hub_create_progress_bar_uses_patched_class(reporter):
    # the path 0.36's byte bars take: _get_progress_bar_context -> module-global tqdm
    tqdm_module = importlib.import_module("huggingface_hub.utils.tqdm")
    if not hasattr(tqdm_module, "_get_progress_bar_context"):
        pytest.skip("hub version without _get_progress_bar_context")
    import logging

    with runtime._bridged_bars("model.ckpt"):
        cm = tqdm_module._get_progress_bar_context(
            desc="model.ckpt",
            log_level=logging.INFO,
            total=50,
            name="huggingface_hub.http",
        )
        with cm as bar:
            bar.update(50)
    assert reporter.calls[-1] == ("model.ckpt", 50, 50)


def test_base_reporter_download_is_noop():
    Reporter().download("x", 1, None)  # must not raise
