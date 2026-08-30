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
