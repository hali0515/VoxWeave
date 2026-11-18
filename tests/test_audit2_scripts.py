"""Regression tests for scripts/ tooling fixes (song_scores.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str, *, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.path.pop(0) == str(SCRIPTS)
    return module


def test_song_scores_falls_back_to_the_mix_on_an_unreadable_cache_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    song_scores = _load("song_scores", module_name="song_scores_audit2")

    def broken(_media: Path) -> Any:
        raise song_scores.artifacts.ArtifactCollisionError("claim belongs to x.mkv")

    monkeypatch.setattr(song_scores.artifacts, "inspect_paths", broken)

    assert song_scores._vocals_source(tmp_path / "episode.mkv") is None
    assert "ignoring the artifact cache: claim belongs to x.mkv" in (
        capsys.readouterr().out
    )
