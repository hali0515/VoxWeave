from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from voxweave import realign
from voxweave.lang import to_iso_or

# Language bucket for units whose language never resolved (chunk-level detection is
# allowed to return None); kept out of the ISO namespace on purpose.
UNKNOWN_LANGUAGE = "unknown"
# Aggregate snapshot names, ordered after the 00-02 audio artifacts.
PRE_POSITION_STAGE = "03-pre-position"
POST_POSITION_STAGE = "04-post-position"
HEALTH_FILE = "alignment-health.json"
HEALTH_SCHEMA_VERSION = 1


def _rate(bad: int, eligible: int) -> dict[str, Any]:
    """A rate that keeps its numerator and denominator.

    ``value`` is ``None`` (not ``0.0``) when nothing was eligible, so "no lexical unit
    to judge" cannot masquerade as a perfect score. Same shape as the calibration
    tooling's ratio blocks, so a report can micro-aggregate several runs by summing
    ``bad`` and ``eligible`` instead of averaging rates.
    """
    return {
        "bad": bad,
        "eligible": eligible,
        "value": bad / eligible if eligible else None,
    }


def _exact_zero_lexical(units: list[dict]) -> int:
    """Lexical units the aligner emitted with ``start == end`` (QwenLM/Qwen3-ASR#197)."""
    return sum(
        1
        for u in units
        if realign.is_lexical_unit(u)
        and float(u["end"]) - float(u["start"]) <= realign.EXACT_ZERO_EPS
    )


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

    def units(self, stage: str, units: list[dict]) -> None:
        """Save a named track-level unit snapshot (whole file, absolute timestamps)."""

    def meta(self, data: dict) -> None:
        """Save track-level metadata."""

    def position_units(
        self,
        units: list[dict],
        vad: list[tuple[float, float]],
        *,
        language: str | None = None,
    ) -> list[dict]:
        """Run the VAD positioning pass; the no-op sink adds no instrumentation.

        Owned by the sink rather than open-coded in the pipeline so that the pre/post
        snapshots and the zero-duration accounting stay on the ``if debug`` free path.
        The return value is identical in both sinks.
        """
        return realign.position_units_with_vad(units, vad)


class FileDebugSink(DebugSink):
    """Write intermediate artifacts to ``debug/<stem>/``.

    Saves raw ASR output per chunk (including markers, useful for spotting hallucinations
    and repetitions). Skipped chunks (``units=None``) are also saved for pinpointing problems.

    Also keeps a run-level alignment health record (``alignment-health.json``): how many
    zero-duration units the aligner emitted, how many the positioning pass repaired, and
    what is left over. Rewritten after every positioning pass, so it is complete whenever
    the run reached that point (and needs no flush hook if the run dies later).
    """

    enabled = True
    root: Path  # always set in __init__ (the base class None is for the no-op sink)

    def __init__(self, stem: str, base: Path | None = None) -> None:
        self.root = (base or Path("debug")) / stem
        self.chunks_dir = self.root / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._health: dict[str, realign.ZeroDurationDiagnostics] = {}
        # (chunk language, exact-zero lexical units) for chunks that produced units.
        self._chunk_zeros: list[tuple[str | None, int]] = []

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
            # Chunk-level census: the "how many chunks are affected at all" denominator
            # is only visible here, since the positioning pass sees one flat unit list.
            self._chunk_zeros.append(
                (to_iso_or(lang, None), _exact_zero_lexical(units))
            )

    def units(self, stage: str, units: list[dict]) -> None:
        rows = [{"unit_id": f"{i:06d}", **u} for i, u in enumerate(units)]
        (self.root / f"{stage}.units.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def meta(self, data: dict) -> None:
        (self.root / "meta.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def position_units(
        self,
        units: list[dict],
        vad: list[tuple[float, float]],
        *,
        language: str | None = None,
    ) -> list[dict]:
        """Snapshot the pass input and output and accumulate its health counters.

        ``unit_id`` in both snapshots is the flat index: the three positioning steps are
        index-preserving, so the same id addresses the same unit before and after (chunk
        indices are gone by this point -- units arrive already shifted and concatenated).
        """
        key = to_iso_or(language, None) or UNKNOWN_LANGUAGE
        diagnostics = self._health.setdefault(key, realign.ZeroDurationDiagnostics())
        self.units(PRE_POSITION_STAGE, units)
        out = realign.position_units_with_vad(units, vad, diagnostics=diagnostics)
        self.units(POST_POSITION_STAGE, out)
        (self.root / HEALTH_FILE).write_text(
            json.dumps(self.health_report(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out

    def health_report(self) -> dict[str, Any]:
        """Build the alignment health record: counters and rates grouped by language."""
        fallback = self._fallback_language()
        chunk_zeros: dict[str, list[int]] = {}
        for lang, zeros in self._chunk_zeros:
            chunk_zeros.setdefault(lang or fallback, []).append(zeros)

        languages: dict[str, Any] = {}
        for key in sorted(set(self._health) | set(chunk_zeros)):
            diagnostics = self._health.get(key) or realign.ZeroDurationDiagnostics()
            per_chunk = chunk_zeros.get(key, [])
            affected = sum(1 for z in per_chunk if z)
            languages[key] = {
                "counters": diagnostics.to_dict(),
                "chunks": {
                    "with_units": len(per_chunk),
                    "with_exact_zero": affected,
                    "exact_zero_units": sum(per_chunk),
                },
                "rates": {
                    # Raw rates are upstream input health, not voxweave output quality:
                    # trend signal only. The residual rates are the ones that must be 0.
                    "raw_exact_zero_unit_rate": _rate(
                        diagnostics.raw_exact_zero, diagnostics.lexical_units
                    ),
                    "raw_collapse_candidate_rate": _rate(
                        diagnostics.raw_collapse_candidates, diagnostics.lexical_units
                    ),
                    "repaired_exact_zero_rate": _rate(
                        diagnostics.repaired_exact_zero, diagnostics.raw_exact_zero
                    ),
                    "repaired_candidate_rate": _rate(
                        diagnostics.repaired_collapse_candidates,
                        diagnostics.raw_collapse_candidates,
                    ),
                    "residual_exact_zero_unit_rate": _rate(
                        diagnostics.residual_exact_zero, diagnostics.lexical_units
                    ),
                    "residual_collapse_candidate_rate": _rate(
                        diagnostics.residual_collapse_candidates,
                        diagnostics.lexical_units,
                    ),
                    "chunk_affected_exact_zero_rate": _rate(affected, len(per_chunk)),
                },
            }

        totals = {
            "units_seen": sum(d.units_seen for d in self._health.values()),
            "lexical_units": sum(d.lexical_units for d in self._health.values()),
            "raw_exact_zero": sum(d.raw_exact_zero for d in self._health.values()),
            "repaired_exact_zero": sum(
                d.repaired_exact_zero for d in self._health.values()
            ),
            "residual_exact_zero": sum(
                d.residual_exact_zero for d in self._health.values()
            ),
            "chunks_with_units": len(self._chunk_zeros),
            "chunks_with_exact_zero": sum(1 for _, z in self._chunk_zeros if z),
            "accounting_balanced": all(
                d.accounting_balanced for d in self._health.values()
            ),
        }
        return {
            "schema_version": HEALTH_SCHEMA_VERSION,
            "kind": "alignment-health",
            "stages": {"pre": PRE_POSITION_STAGE, "post": POST_POSITION_STAGE},
            "totals": totals,
            "languages": languages,
        }

    def _fallback_language(self) -> str:
        """Language bucket for chunks whose own detection returned nothing.

        Per-chunk detection is allowed to fail while the file-level language (the one the
        positioning pass ran under) is known; when exactly one such language exists, the
        undetected chunks belong to it. Otherwise they stay in their own bucket rather
        than being attributed by guesswork.
        """
        known = sorted(set(self._health) - {UNKNOWN_LANGUAGE})
        return known[0] if len(known) == 1 else UNKNOWN_LANGUAGE
