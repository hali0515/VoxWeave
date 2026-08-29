"""The package must report the installed distribution version, never a hardcoded one."""

import importlib.metadata

import voxweave


def test_dunder_version_matches_installed_distribution() -> None:
    assert voxweave.__version__ == importlib.metadata.version("voxweave")
