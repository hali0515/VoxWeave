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
