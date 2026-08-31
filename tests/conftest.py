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


def pytest_runtest_setup(item):
    """Debug-branch sentinel: report the first test that sees a poisoned hub."""
    import faulthandler
    import sys
    import threading

    utils = sys.modules.get("huggingface_hub.utils")
    if utils is None or hasattr(utils, "HfFolder"):
        return
    log = Path("/tmp/hub_sentinel.log")
    if log.exists():
        return
    spec = getattr(utils, "__spec__", None)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(
            "=== POISONED huggingface_hub.utils detected before %s ===\n"
            "id=%s file=%s initializing=%s\n"
            "dict keys (first 60): %s\n"
            "hub-modules: %s\n"
            "threads: %s\n"
            % (
                item.nodeid,
                id(utils),
                getattr(utils, "__file__", None),
                getattr(spec, "_initializing", None),
                sorted(list(vars(utils)))[:60],
                sorted(m for m in sys.modules if m.startswith("huggingface_hub")),
                [t.name for t in threading.enumerate()],
            )
        )
        faulthandler.dump_traceback(all_threads=True, file=fh)
        fh.write("=== end sentinel ===\n")
