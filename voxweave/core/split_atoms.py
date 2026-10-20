"""Text ranges and atoms: map units to display characters and build the break atoms.

The lowest layer of the segmentation engine. It turns word-level units into
surface text, display-character ranges and phrase-boundary atoms. It imports
only leaf modules, never ``smart_split``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .breakpoints import phrase_atoms
from .layout import (
    _PUNCT_TO_SPACE_RE,
    _no_spaces,
    _token_char_count,
    _tokens,
    _vis_width,
)
from .schema import Unit

log = logging.getLogger("voxweave.core.smart_split")


def _span_start(
    items: Sequence[Mapping[str, Any]], default: float | None = None
) -> float | None:
    """First non-None ``start`` across items, else ``default``."""
    return next(
        (it.get("start") for it in items if it.get("start") is not None), default
    )


def _span_end(
    items: Sequence[Mapping[str, Any]], default: float | None = None
) -> float | None:
    """Last non-None ``end`` across items, else ``default``."""
    return next(
        (it.get("end") for it in reversed(items) if it.get("end") is not None), default
    )


def _unit_text(unit: Mapping[str, Any]) -> str:
    """Surface a word_data entry stores: ``text`` (aligner/repacked) or ``word`` (ASR)."""
    return str(unit.get("text") or unit.get("word") or "")


def _display_chars(surfaces: Sequence[str]) -> list[str]:
    """Per surface, the characters a finished cue text still shows.

    Cue text is finalized *after* word_data is frozen: ``strip_punct_for_subtitles``
    turns prose punctuation into spaces and ``_merge_stutters`` hyphenates a
    repeated ASCII word. Normalizing both sides the same way lets a stored
    surface be matched against the rendered text it came from.

    The stream is normalized *joined*, then split back per surface, because
    ``_PUNCT_TO_SPACE_RE`` is context-sensitive: ``[.,](?!\\d)`` keeps the ``.``
    of ``3.75``, but a lone ``.`` unit looked at in isolation has no following
    digit and would be dropped from one side only.
    """
    joined = "".join(surfaces)
    stripped = {
        i
        for m in _PUNCT_TO_SPACE_RE.finditer(joined)
        for i in range(m.start(), m.end())
    }
    out: list[str] = []
    at = 0
    for surface in surfaces:
        out.append(
            "".join(
                ch
                for i, ch in enumerate(surface, at)
                if i not in stripped and not ch.isspace() and ch != "-"
            )
        )
        at += len(surface)
    return out


def _surface_ranges(
    surfaces: Sequence[str], word_data: Sequence[Unit], offset: int = 0
) -> list[tuple[int, int]] | None:
    """Map each surface onto the word_data entries that spell it, else ``None``.

    Pairing is by stored surface rather than by character count, so it holds at
    either granularity: an entry may be one whole atom, several entries may
    spell one token, and an entry may cover several rendered tokens (an embedded
    Latin run the display split apart, a legacy sentence-sized ``word``).
    Entries the finished cue text no longer shows — punctuation
    ``strip_punct_for_subtitles`` removed — have no display char, so they fall
    between ranges and land with the atom that *follows* them; a trailing one
    has no follower and is left to the caller.

    ``None`` when the two sides cannot be reconciled: inventing an alignment
    would hand every later atom another atom's timestamps, so the caller degrades
    to the plain cursor instead.
    """
    total = len(word_data)
    unit_chars = _display_chars([_unit_text(u) for u in word_data])
    text_chars = _display_chars(surfaces)
    ranges: list[tuple[int, int]] = []
    ti = ei = 0
    while ti < len(surfaces):
        if not text_chars[ti]:
            # Punctuation still in the text: it owns the matching entry (this is
            # the pre-render stage, where neither side has been stripped yet).
            end = ei + 1 if ei < total and not unit_chars[ei] else ei
            ranges.append((ei + offset, end + offset))
            ei = end
            ti += 1
            continue
        while ei < total and not unit_chars[ei]:  # punctuation the cue text dropped
            ei += 1
        block_start = ei
        block_size = 0
        got_text = got_units = ""
        while not got_text or got_text != got_units:
            if len(got_text) <= len(got_units):
                if ti >= len(surfaces):
                    return None
                got_text += text_chars[ti]
                ti += 1
                block_size += 1
            else:
                if ei >= total:
                    return None
                got_units += unit_chars[ei]
                ei += 1
            if not (got_text.startswith(got_units) or got_units.startswith(got_text)):
                return None
        ranges.extend([(block_start + offset, ei + offset)] * block_size)
    return ranges


def _cursor_ranges(
    surfaces: Sequence[str],
    word_data: Sequence[Unit],
    offset: int = 0,
    *,
    one_per_surface: bool = False,
) -> list[tuple[int, int]]:
    """Per-surface footprint from a plain char cursor (one entry per non-space char).

    Space-delimited streams set ``one_per_surface`` to retain their legacy
    one-token-per-entry fallback when word_data carries no reconcilable surface.
    Tolerant on purpose: a surface past the end of the stream gets an empty
    slice, which renders as ``start=end=None``. That degraded timing is the only
    safe answer for an unreadable cue — this code runs after ASR, forced
    alignment and diarization, so refusing the cue would throw a whole file's
    work away over one bad row.
    """
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for surface in surfaces:
        n = 1 if one_per_surface else _token_char_count(surface)
        ranges.append((cursor + offset, cursor + n + offset))
        cursor += n
    return ranges


def _unit_ranges(
    surfaces: Sequence[str],
    word_data: Sequence[Unit],
    offset: int = 0,
    *,
    cursor_one_per_surface: bool = False,
) -> list[tuple[int, int]]:
    """Per-surface ``word_data`` footprint, whatever granularity the stream carries.

    Reconciliation is by the entries' own surfaces, the only granularity-agnostic
    reading: the same stream may hold one entry per non-space character (aligner
    output) or one per packed atom (a cue already materialized by
    ``_chunk_to_cue``), and the key it uses says nothing about which —
    ``realign.reinject_punct`` writes ``text`` on char-level units too. Streams
    that store no surface at all, and streams that cannot be reconciled, fall
    back to the char cursor.
    """
    if any(_unit_text(u) for u in word_data):
        ranges = _surface_ranges(surfaces, word_data, offset)
        if ranges is not None:
            return ranges
        log.debug(
            "word_data desync: %d entries do not spell %r; falling back to the "
            "char cursor (some atoms may lose their timestamps)",
            len(word_data),
            "".join(surfaces)[:60],
        )
    return _cursor_ranges(
        surfaces, word_data, offset, one_per_surface=cursor_one_per_surface
    )


def _build_atoms(
    text: str,
    word_data: list[Unit],
    lang: str,
    max_atom_width: int | None = None,
) -> list[dict]:
    """Build non-breakable atoms, each with aggregated start/end from word_data.

    Space-delimited: one displayed word per atom, reconciled against word_data.
    No-space: one atom per CJK char or Latin run (from ``_tokens``). BudouX phrase
    boundaries are applied later in the packing loop — atoms stay per-char so
    gap/duration breaks have full granularity.

    word_data comes at one of two granularities and they are *not*
    interchangeable: the first-generation stream holds one entry per non-space
    character, while a cue already materialized by ``_chunk_to_cue`` holds one
    entry per packed atom. ``_unit_ranges`` reconciles either against the text,
    so a re-read never advances a character cursor over atom entries. Every atom
    records its ``_unit_start``/``_unit_end`` footprint so callers can slice the
    source word_data without redoing that arithmetic; the footprint may run past
    the end of an unreconcilable stream, which is how an uncoverable atom ends up
    with ``start=end=None``. Never raises — see ``_cursor_ranges``.

    Reconciling by surface also narrows the aggregated span itself, on the plain
    non-diarize path too: entries the renderer dropped from the display text (a
    trailing ``。``, the ``.`` of ``e.g.``) no longer extend the neighbouring
    atom's start/end the way the old character cursor's off-by-N did. Corpus
    impact at the production layout config is 0 of 40 cases.
    """
    if not _no_spaces(lang):
        toks = text.split()
        ranges = _unit_ranges(toks, word_data, cursor_one_per_surface=True)
        atoms: list[dict] = []
        for tok, (start, end) in zip(toks, ranges):
            chunk = word_data[start:end]
            atoms.append(
                {
                    "text": tok,
                    "start": _span_start(chunk),
                    "end": _span_end(chunk),
                    "_unit_start": start,
                    "_unit_end": end,
                }
            )
        return atoms
    units = _tokens(text, lang)
    ranges = _unit_ranges(units, word_data)
    atoms: list[dict] = []
    for unit, (start, end) in zip(units, ranges):
        chunk = word_data[start:end]
        # Short embedded Latin phrases stay atomic, but a phrase wider than a
        # physical line must expose its whitespace boundaries to the packer.
        # Keep trailing spaces on each sub-atom so no-space joining reconstructs
        # the original surface, and retain real per-character timing.
        if (
            max_atom_width is not None
            and _vis_width(unit) > max_atom_width
            and any(ch.isspace() for ch in unit)
        ):
            parts = [m.group(0) for m in re.finditer(r"\S+\s*", unit)]
            part_ranges = _unit_ranges(parts, chunk, offset=start)
            for part_i, (surface, (p_start, p_end)) in enumerate(
                zip(parts, part_ranges)
            ):
                part_chunk = word_data[p_start:p_end]
                atoms.append(
                    {
                        "text": surface,
                        "start": _span_start(part_chunk),
                        "end": _span_end(part_chunk),
                        "forced_boundary": part_i > 0,
                        "_unit_start": p_start,
                        "_unit_end": p_end,
                    }
                )
            continue
        atoms.append(
            {
                "text": unit,
                "start": _span_start(chunk),
                "end": _span_end(chunk),
                "_unit_start": start,
                "_unit_end": end,
            }
        )
    return atoms


def _phrase_boundary_atoms(atoms: list[dict], text: str, lang: str) -> set[int]:
    """Atom indices that are BudouX phrase starts — the only legal length-break
    points (prevents splitting mid-phrase, e.g. です into で|す).

    Reconciles per-char/per-run atoms with BudouX phrases via a non-space char
    cursor. Returns ATOM indices (not char offsets) — a Latin run like GPT-4 is
    one atom but multiple chars. Without BudouX, phrase_atoms falls back to
    per-char, degrading to length-only breaks.
    """
    phrase_starts: set[int] = set()
    c = 0
    for ph in phrase_atoms(text, lang):
        phrase_starts.add(c)
        c += _token_char_count(ph)
    boundary: set[int] = set()
    c = 0
    for i, a in enumerate(atoms):
        if c in phrase_starts:
            boundary.add(i)
        c += _token_char_count(a["text"])
    return boundary
