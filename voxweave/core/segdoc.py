"""Immutable segmentation IR: the record of what one segmentation actually ran.

P3 introduces this record *around* the legacy v1 engine without feeding it: the
engine keeps consuming the raw dicts, while :class:`SegDocument` is the
authoritative parallel account that P4 strangles the engine onto.

Everything here is a *recorder*. It applies no defaults, clamps nothing and
normalizes nothing, because its whole value is being able to say what ran rather
than what should have run: the tree carries two disagreeing definitions of the
same thresholds (``config.gap_thresholds`` vs the ``SplitThresholds`` dataclass
defaults, ``0.458`` vs ``11/24`` for the shot-snap window), so a profile that
"fixed up" a value would describe a run that never happened.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The nine gap/timing knobs ``pipeline.segment_document`` resolves into
#: ``thresholds_used`` and hands to the engine. Ordered as declared on
#: :class:`DisplayProfile`.
THRESHOLD_KEYS: tuple[str, ...] = (
    "clause_ms",
    "vad_skip_ms",
    "offline_ms",
    "min_cue_s",
    "max_cue_s",
    "glue_gap_s",
    "cps",
    "lag_out_s",
    "shot_snap_s",
)


def normalize_speaker_turn_bounds(start: float, end: float) -> tuple[float, float]:
    """Normalize a persisted turn to a total, forward interval.

    Point turns are retained because adjacent records on either side can encode
    two distinct, coincident label changes.  A reversed persisted record is
    collapsed at its declared start instead of being allowed to create negative
    overlap or crash a shadow-only consumer.  Both the persisted parser and W3
    evidence use this function, so normalization cannot diverge by lane.
    """
    normalized_start = float(start)
    normalized_end = float(end)
    if not math.isfinite(normalized_start) or not math.isfinite(normalized_end):
        raise ValueError("speaker turn bounds must be finite")
    return normalized_start, max(normalized_start, normalized_end)


@dataclass(frozen=True)
class SourceUnit:
    """One aligned token of the ingested stream, with a positional identity.

    ``id`` is ``u{index}`` at ingest, so replaying the same ``word_segments``
    mints the same ids -- the ids are never persisted (the sibling JSON stays
    byte-identical), they exist so passes downstream of ingest can refer to a
    unit instead of re-deriving a character cursor.

    ``surface`` is the granularity-blind view ``smart_split._unit_text`` takes
    (``text`` wins, ``word`` is the ASR-side fallback, absent is ``""``); spans
    are whatever the aligner recorded, including ``None`` for ghost units.

    ``provenance`` and ``confidence`` are RESERVED for P5: they are minted with
    their defaults today and nothing reads them, so the field set is already the
    one a later per-unit evidence pass needs and that pass does not have to
    migrate the IR a second time. Additive with defaults, so every existing
    positional construction and equality comparison is unchanged.
    """

    id: str
    surface: str
    start: float | None
    end: float | None
    provenance: str = "aligner"
    confidence: float | None = None


@dataclass(frozen=True)
class DisplayProfile:
    """The layout and timing knobs one segmentation ran with, verbatim.

    ``max_line_length`` is counted in native cells (the CJK presets are 18/12,
    latin is 42), matching what the engine measures.
    """

    language: str
    max_line_length: int
    max_lines: int
    clause_ms: float
    vad_skip_ms: float
    offline_ms: float
    min_cue_s: float
    max_cue_s: float
    glue_gap_s: float
    cps: float
    lag_out_s: float
    shot_snap_s: float

    @classmethod
    def from_resolved(
        cls,
        language: str,
        thresholds: Mapping[str, Any],
        *,
        max_line_length: int,
        max_lines: int,
    ) -> DisplayProfile:
        """Record already-resolved values; ``float`` coercion is the only change.

        ``thresholds`` is the caller's ``thresholds_used`` mapping, which always
        carries all nine keys; a missing one means the caller resolved something
        else and raises :class:`KeyError` rather than being papered over with a
        default that did not run. Extra keys are ignored, and the layout pair is
        stored exactly as handed in.
        """
        values = {key: float(thresholds[key]) for key in THRESHOLD_KEYS}
        return cls(
            language=language,
            max_line_length=max_line_length,
            max_lines=max_lines,
            **values,
        )


@dataclass
class SegDocument:
    """One segmentation's inputs: units, resolved profile, evidence, manifest.

    The evidence arrays are the same objects ``pipeline.segment_document``
    already copied for the engine (``None`` when absent), and ``manifest`` is the
    dict the pipeline built and persists -- held by reference, so the document
    and the sibling JSON can never disagree about what ran. Because the manifest
    is held by reference, the pipeline can mint the document *before* the engine
    runs and still have it carry the degradation ledger the run fills in.

    ``text`` is the exact joined surface stream the v1 engine consumed
    (``pipeline._units_to_seg``'s ``text``), recorded rather than re-derived: a
    consumer that re-joined ``units`` itself would have to re-implement the
    no-space-language rule and could disagree with what actually ran. ``None``
    means the builder was not handed one.
    """

    language: str
    units: list[SourceUnit]
    profile: DisplayProfile
    vad_speech: list[tuple[float, float]] | None
    shot_changes: list[float] | None
    sing_spans: list[tuple[float, float]] | None
    speaker_turns: list[tuple[float, float, str]] | None
    manifest: dict
    text: str | None = None


def _coerce_span(value: Any) -> float | None:
    """A unit bound as a float, keeping ``None`` (ghost unit) distinguishable."""
    return None if value is None else float(value)


def build_seg_document(
    *,
    language: str,
    units: Sequence[Mapping[str, Any]],
    profile: DisplayProfile,
    manifest: dict,
    vad_speech: list[tuple[float, float]] | None = None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
    text: str | None = None,
) -> SegDocument:
    """Mint a :class:`SegDocument` from the stream the engine is about to run on.

    ``units`` is the raw ``word_segments``-shaped sequence (the repaired copy
    cue formation sees, not the persisted evidence); ids are positional, so the
    same stream always mints the same ids. ``text`` is the joined stream the
    engine consumes, passed in rather than rebuilt here. Everything else is
    recorded as given.
    """
    # Single source of truth for the surface view; imported lazily so this
    # module stays importable without pulling in the segmentation engine.
    from voxweave.core.smart_split import _unit_text

    minted = [
        SourceUnit(
            id=f"u{index}",
            surface=_unit_text(unit),
            start=_coerce_span(unit.get("start")),
            end=_coerce_span(unit.get("end")),
        )
        for index, unit in enumerate(units)
    ]
    return SegDocument(
        language=language,
        units=minted,
        profile=profile,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
        manifest=manifest,
        text=text,
    )
