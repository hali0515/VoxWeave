"""Cue overlays after the v1 engine, and the span/turn inputs they read.

``segment_document`` runs smart_split and then overlays the result: lyric flags
from the singing spans, speaker formatting, and a shot re-snap after it. The
shadow lane replays the same overlays on its comparators, and every subtitle
renderer shows a lyric cue through the same music-note wrap, so the helpers live
here rather than in either caller. The span/turn parsers and copies are the
input side of the same contract: "no spans recorded" and "empty array" mean the
same thing to every consumer of persisted spans.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from voxweave.core.schema import Cue

log = logging.getLogger("voxweave")

# A cue is a lyric when at least this fraction of its span overlaps detected singing.
LYRIC_MIN_OVERLAP = 0.5


def spans_in(raw: Any) -> list[tuple[float, float]] | None:
    """Parse a persisted ``vad_speech`` array (``[[start, end], ...]``) to float tuples.

    Malformed entries (wrong arity, non-numeric bounds) are skipped with a warning
    rather than crashing the whole re-split. None if absent/empty or nothing survives.
    """
    if not raw:
        return None
    out: list[tuple[float, float]] = []
    for entry in raw:
        try:
            s, e = entry
            out.append((float(s), float(e)))
        except (TypeError, ValueError):
            log.warning("skipping malformed vad_speech entry: %r", entry)
    return out or None


def turns_in(raw: Any) -> list[tuple[float, float, str]] | None:
    """Parse persisted ``speaker_turns`` (``[[start, end, label], ...]``).

    This is the byte-preserving production replay seam: numeric bounds are only
    coerced to ``float``.  Reversed, point and non-finite legacy values survive
    exactly as they did before P5; the shadow's speaker-evidence consumer owns
    normalization on its detached document copy after the feature flag.

    Malformed entries (wrong arity or non-numeric bounds) are skipped with a
    warning. None if absent/empty or nothing survives.
    """
    if not raw:
        return None
    out: list[tuple[float, float, str]] = []
    for entry in raw:
        try:
            s, e, lb = entry
            out.append((float(s), float(e), str(lb)))
        except (TypeError, ValueError):
            log.warning("skipping malformed speaker_turns entry: %r", entry)
    return out or None


def copied_spans(
    spans: Sequence[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """Copy a span sequence to plain float tuples; empty/absent -> ``None``.

    Mirrors :func:`spans_in`: "no spans recorded" and "empty array" are the same
    thing for every consumer of persisted spans.
    """
    return [(float(s), float(e)) for s, e in spans] if spans else None


def copied_turns(
    turns: Sequence[tuple[float, float, str]] | None,
) -> list[tuple[float, float, str]] | None:
    """Copy speaker turns to plain tuples; empty/absent -> ``None`` (see :func:`turns_in`)."""
    return (
        [(float(s), float(e), str(label)) for s, e, label in turns] if turns else None
    )


def resnap_shots(
    cues: list[Cue], shot_changes: list[float] | None, thresholds: dict
) -> list[Cue]:
    """Re-apply shot snapping after speaker formatting rewrote the cue stream.

    smart_split snaps boundaries to shot changes as its last timing step, but
    speaker formatting splits cues at speaker turns and runs another timing
    cleanup, which moves those boundaries again -- so a formatted cue can end up
    flashing across a cut that the first snap had cleared. Snapping once more
    with the same cuts and the same duration cap restores the invariant;
    ``_snap_to_shots`` leaves boundaries that already sit in a landing zone
    untouched, so the extra pass is a no-op when formatting changed nothing.
    """
    if not shot_changes:
        return cues
    from voxweave.core.smart_split import SplitThresholds
    from voxweave.core.timing import _snap_to_shots

    th = SplitThresholds.from_mapping(thresholds)
    return _snap_to_shots(
        cues, sorted(shot_changes), snap_s=th.shot_snap_s, max_cue_s=th.max_cue_s
    )


def mark_lyric_cues(
    cues: Sequence[Cue], sing_spans: list[tuple[float, float]] | None
) -> None:
    """Flag cues whose span mostly overlaps detected singing (``lyric=True``).

    The stored cue text stays clean; display layers (the VTT rows rendered by
    ``segmentation_projector`` / ``align_projector``, SRT/ASS export) wrap flagged
    cues with music notes per the Netflix lyric convention. Runs in place after
    smart_split so flags ride the final cues.
    """
    if not sing_spans:
        return
    for c in cues:
        start, end = c.get("start"), c.get("end")
        if start is None or end is None or end - start <= 0:
            continue
        overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in sing_spans)
        if overlap / (end - start) >= LYRIC_MIN_OVERLAP:
            c["lyric"] = True


def wrap_lyric(text: str, lyric: object) -> str:
    """``text`` inside the Netflix music-note wrap (note + space at the start and
    end of the subtitle) when ``lyric`` is truthy, otherwise ``text`` unchanged."""
    return f"♪ {text} ♪" if lyric else text


def lyric_display_text(cue: Mapping[str, Any]) -> str:
    """Cue display text: lyric cues get the Netflix music-note wrap (note + space
    at the start and end of the subtitle), others pass through unchanged."""
    text = str(cue["text"])
    return wrap_lyric(text, cue.get("lyric"))
