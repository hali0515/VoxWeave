from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from voxweave import realign
from voxweave.lang import (
    reconcile_detected_language,
    to_iso_or,
    transcript_content_weight,
)

log = logging.getLogger("voxweave")


def _select_transcript_language(
    results: Sequence[tuple[str | None, str, Sequence[dict]]],
    override: str | None = None,
) -> str:
    """Select the file language from pre-alignment transcript content.

    Unit counts cannot be used here: a wrong aligner is liable to collapse Han
    prose into a few units (or fragment another script), making the original
    classification error self-reinforcing.  Normalize equivalent labels and
    weight each reconciled label by its transcript's alphanumeric content.
    """
    if override and override.strip():
        return override.strip()

    weights: Counter[str] = Counter()
    for detected, text, _units in results:
        if not text.strip():
            continue
        effective = reconcile_detected_language(detected, text)
        if not effective:
            continue
        key = to_iso_or(effective, None) or effective.strip().casefold()
        mass = transcript_content_weight(text)
        if mass:
            weights[key] += mass
    return weights.most_common(1)[0][0] if weights else "english"


def _joined_unit_text(
    units: list[dict],
    pieces: list[str],
    meaningful: list[str],
    stored_iso: str,
) -> tuple[bool, str]:
    # A correctly reinjected no-space stream has exactly one visible character
    # per unit; spaces inside Latin phrases ride on the preceding unit (``T ``).
    # If only the persisted label is stale, joining such a stream with the
    # label's English separator would invent ``G P T``.  Recognize the
    # representation itself and preserve its exact whitespace instead.
    char_grained = len(meaningful) == len(units) and all(
        sum(not ch.isspace() for ch in piece) == 1 for piece in meaningful
    )
    if char_grained:
        text = "".join(pieces)
    else:
        stored_sep = "" if stored_iso in realign.NO_SPACE_LANGS else " "
        text = stored_sep.join(meaningful)
    return char_grained, text


def _unit_timings(seq: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]] | None:
    rows: list[tuple[float, float]] = []
    for unit in seq:
        try:
            start = float(unit["start"])
            end = float(unit["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            return None
        rows.append((start, end))
    return rows or None


def _rebuild_preserves_source(
    source_times: list[tuple[float, float]],
    rebuilt: list[dict],
    text: str,
    effective_no_space: bool,
) -> bool:
    # schema.Unit carries the surface under ``text`` OR ``word``; read both.
    from voxweave.core.smart_split import _unit_text

    rebuilt_times = _unit_timings(rebuilt)
    source_monotone = all(
        a[0] <= b[0] and a[1] <= b[1] for a, b in zip(source_times, source_times[1:])
    )
    rebuilt_monotone = rebuilt_times is not None and all(
        a[0] <= b[0] and a[1] <= b[1] for a, b in zip(rebuilt_times, rebuilt_times[1:])
    )
    if effective_no_space:
        round_trip_ok = "".join(_unit_text(u) for u in rebuilt) == text
    else:
        rebuilt_text = " ".join(_unit_text(u) for u in rebuilt)
        round_trip_ok = " ".join(rebuilt_text.split()) == " ".join(text.split())
    envelope_ok = rebuilt_times is not None and all(
        math.isclose(a, b, abs_tol=1e-6)
        for a, b in zip(
            (min(s for s, _e in source_times), max(e for _s, e in source_times)),
            (
                min(s for s, _e in rebuilt_times),
                max(e for _s, e in rebuilt_times),
            ),
        )
    )
    return not (
        not rebuilt
        or not round_trip_ok
        or not envelope_ok
        or (source_monotone and not rebuilt_monotone)
    )


def _reconcile_word_segment_language(
    language: str | None,
    units: list[dict],
    *,
    override: str | None = None,
) -> tuple[str, list[dict]]:
    """Repair a strong persisted language/tokenization mismatch before splitting.

    Older outputs can say ``en`` even when their transcript is overwhelmingly
    Han.  Those files also carry the damage from the English aligner: a whole
    Chinese paragraph may be stored as one 10--20 second ``word``.  Merely
    changing the smart-split language cannot recover from that coarse timing.

    Reconstruct the text using the *stored* unit contract, reconcile its script,
    and only for a strong spaced -> no-space correction rebuild per-character
    timings through :func:`realign.reinject_punct`.  That helper retains
    punctuation and spaces inside Latin runs (``GPT Red``) while distributing
    each coarse unit's existing span; no ASR/alignment model is called.
    """
    if not units:
        raise RuntimeError("no word segments to split")
    # schema.Unit carries the surface under ``text`` OR ``word``; read both.
    from voxweave.core.smart_split import _unit_text

    original_units = units
    stored_iso = to_iso_or(language, "en")
    pieces = [_unit_text(unit) for unit in units]
    meaningful = [piece for piece in pieces if piece.strip()]
    if not meaningful:
        return stored_iso, units

    char_grained, text = _joined_unit_text(units, pieces, meaningful, stored_iso)

    effective = reconcile_detected_language(language, text, override=override)
    effective_iso = to_iso_or(effective, stored_iso)
    if effective_iso == stored_iso:
        return stored_iso, units

    stored_no_space = stored_iso in realign.NO_SPACE_LANGS
    effective_no_space = effective_iso in realign.NO_SPACE_LANGS
    if stored_no_space == effective_no_space:
        log.warning(
            "word-segment language/script mismatch: %s -> %s; "
            "reusing compatible timings",
            stored_iso,
            effective_iso,
        )
        return effective_iso, units

    # A char-grained stream already satisfies every no-space target's unit
    # contract.  Relabel it directly instead of destructively reinjecting.
    if char_grained and effective_no_space:
        log.warning(
            "word-segment language/script mismatch: %s -> %s; "
            "reusing existing character timings",
            stored_iso,
            effective_iso,
        )
        return effective_iso, units

    source_times = _unit_timings(units)
    if source_times is None:
        log.warning(
            "cannot repair word-segment language %s -> %s: malformed source timings",
            stored_iso,
            effective_iso,
        )
        return stored_iso, original_units

    # reinject_punct reads ``text`` only: hand it ``word``-keyed units under that key.
    text_units = [
        unit
        if unit.get("text") == _unit_text(unit)
        else {**unit, "text": _unit_text(unit)}
        for unit in units
    ]
    try:
        rebuilt = realign.reinject_punct(text, text_units, effective_iso)
    except Exception as exc:  # noqa: BLE001 -- persisted input may be arbitrarily malformed
        log.warning(
            "cannot repair word-segment language %s -> %s: %s",
            stored_iso,
            effective_iso,
            exc,
        )
        return stored_iso, original_units

    if not _rebuild_preserves_source(source_times, rebuilt, text, effective_no_space):
        log.warning(
            "cannot repair word-segment language %s -> %s without changing "
            "content/timing; keeping persisted representation",
            stored_iso,
            effective_iso,
        )
        return stored_iso, original_units

    noun = "character" if effective_no_space else "word"
    log.warning(
        "word-segment language/script mismatch: %s -> %s; rebuilding %s timings",
        stored_iso,
        effective_iso,
        noun,
    )
    return effective_iso, rebuilt
