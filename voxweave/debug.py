from __future__ import annotations

import json
import shutil
from pathlib import Path


class DebugSink:
    """No-op base for intermediate artifact persistence.

    The pipeline calls these methods unconditionally; ``FileDebugSink`` overrides them
    when ``debug=True``. Keeps the pipeline free of ``if debug`` checks.
    """

    enabled = False
    root: Path | None = None

    def audio(self, name: str, path: Path) -> None:
        """Save a track-level audio artifact."""

    def chunk(
        self,
        idx: int,
        *,
        wav: Path,
        start: float,
        end: float,
        raw: str,
        text: str,
        lang: str | None,
        units: list[dict] | None,
    ) -> None:
        """Save a VAD chunk with its raw ASR output and alignment units."""

    def meta(self, data: dict) -> None:
        """Save track-level metadata."""


class FileDebugSink(DebugSink):
    """Write intermediate artifacts to ``debug/<stem>/``.

    Saves raw ASR output per chunk (including markers, useful for spotting hallucinations
    and repetitions). Skipped chunks (``units=None``) are also saved for pinpointing problems.
    """

    enabled = True
    root: Path  # always set in __init__ (the base class None is for the no-op sink)

    def __init__(self, stem: str, base: Path | None = None) -> None:
        self.root = (base or Path("debug")) / stem
        self.chunks_dir = self.root / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def audio(self, name: str, path: Path) -> None:
        shutil.copy(path, self.root / name)

    def chunk(
        self,
        idx: int,
        *,
        wav: Path,
        start: float,
        end: float,
        raw: str,
        text: str,
        lang: str | None,
        units: list[dict] | None,
    ) -> None:
        tag = f"{idx:03d}_{start:.1f}-{end:.1f}"
        shutil.copy(wav, self.chunks_dir / f"{tag}.wav")
        (self.chunks_dir / f"{tag}.raw.txt").write_text(raw, encoding="utf-8")
        (self.chunks_dir / f"{tag}.text.txt").write_text(text, encoding="utf-8")
        (self.chunks_dir / f"{tag}.lang.txt").write_text(lang or "", encoding="utf-8")
        if units is not None:
            (self.chunks_dir / f"{tag}.units.json").write_text(
                json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def meta(self, data: dict) -> None:
        (self.root / "meta.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
