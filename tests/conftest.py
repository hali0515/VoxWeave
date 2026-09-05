import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_voxweave_cache(tmp_path: Path) -> Iterator[None]:
    """Keep cache-owned episode artifacts inside each test's temporary root."""
    name = "VOXWEAVE_CACHE_ROOT"
    previous = os.environ.get(name)
    os.environ[name] = str(tmp_path / ".voxweave-cache")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.fixture(autouse=True)
def _isolate_voxweave_config(tmp_path: Path) -> Iterator[None]:
    """Never let a test read the developer's real ~/.config/voxweave.conf.

    Points VOXWEAVE_CONFIG at a file that does not exist, so config._load() sees
    an empty config unless a test writes its own (the conf_at fixtures do).
    """
    name = "VOXWEAVE_CONFIG"
    previous = os.environ.get(name)
    os.environ[name] = str(tmp_path / "voxweave.conf")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.fixture(autouse=True)
def _per_chunk_asr_unless_opted_in() -> Iterator[None]:
    """Pin the Qwen ASR batch to 1 unless a test sets VOXWEAVE_ASR_BATCH itself.

    Tests stub the per-chunk seam (backend._asr_only) and expect transcribe_chunks
    to call it once per chunk; the batched pass bypasses that seam and would reach
    the real model loader. Batching tests opt in with monkeypatch.setenv; the
    config tests clear the variable to check the built-in default.
    """
    name = "VOXWEAVE_ASR_BATCH"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
