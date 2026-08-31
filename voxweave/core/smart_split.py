"""Subtitle splitting with gap-aware cue segmentation.

Two stages:
1. ``split_at_sentence_end`` — PySBD (or regex fallback) sentence boundaries,
   then ``split_sentence_heuristically`` for comma/conjunction splits.
2. ``split_long_cues_with_word_timings`` — word-level greedy packing into
   cues fitting ``max_lines × max_line_length``, with gap/duration breaks.

Each sentence/comma clause is its own cue so timings track real speech
boundaries; the one exception is ``_glue_short_cues`` (see ``timing``), which
folds a lone-word flicker cue onto whichever neighbor abuts it within a
sub-0.3s gap (no real pause crossed) — forward for leading interjections,
backward for tail fragments.

This module owns the segmentation *engine*: clause/sentence splitting and the
atom packing loop. Pure text helpers and display wrapping live in ``layout``;
cue-stream timing polish (glue/merge/cleanup/shot-snap) lives in ``timing``.
"""

from __future__ import annotations

import functools
import logging
import math
import re
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, cast

from .breakpoints import legal_break_index, phrase_atoms
from .conjunctions import conjunctions_by_language, get_comma
from .gap_split import gap_qualifies
from .kinsoku import (
    line_end_penalty,
    line_start_penalty,
    zh_pos_boundary_penalties,
)
from .langsets import LANGUAGES_WITHOUT_SPACES as LANGUAGES_WITHOUT_SPACES  # re-export
from .layout import (
    WIDE_GLYPH_LANGUAGES,
    _PUNCT_TO_SPACE_RE,
    _comma_chars,
    _fits_budget,
    _join,
    _line_budget_width,
    _merge_stutters,
    _no_spaces,
    _strip_trailing_commas,
    _token_char_count,
    _tokens,
    _vis_width,
    _visual_len,
    default_max_line_length,
    default_max_lines,
    split_subtitle,
    strip_punct_for_subtitles,
    wrap_cue_text,
)
from .providers import note_degraded
from .schema import Cue, Unit
from .timing import (
    GLUE_MAX_GAP_S,
    _cleanup_cues,
    _glue_short_cues,
    _merge_micro_cues,
    _snap_to_shots,
    combine_speech,
)

log = logging.getLogger(__name__)

DEFAULT_MIN_DURATION = 3.0  # reading-speed pad for single cues
DEFAULT_DESIRED_WPS = 4.0  # target reading speed (English wps)

# Comma line-break: split into separate cues at commas, but only when both
# sides are at least this long (visual chars). Shorter clauses stay attached
# to a neighbor so we never strand a tiny fragment on its own cue.
DEFAULT_COMMA_SPLIT_MIN_LEN = 18  # latin / space-delimited
DEFAULT_COMMA_SPLIT_MIN_LEN_CJK = 6  # zh/yue/ja/ko: chars are ~2x visual width

FORCE_BREAK_FACTOR = 1.5  # boundary-less run may exceed the line budget by at most this before a forced cut

# Cursor recovery: how many later clauses may be probed for a resync point when a
# clause cannot be located in ``word_data`` at all. A desync that survives this
# many clauses is not a local glitch, and each surviving clause still re-anchors
# on its own content, so a wider search buys nothing.
RESYNC_LOOKAHEAD_CLAUSES = 8


def default_comma_split_min_len(lang: str) -> int:
    """Minimum clause length (visual chars) for a comma to become a cue boundary.
    Wide-glyph languages use a smaller value (~2x visual width per char)."""
    return (
        DEFAULT_COMMA_SPLIT_MIN_LEN_CJK
        if lang in WIDE_GLYPH_LANGUAGES
        else DEFAULT_COMMA_SPLIT_MIN_LEN
    )


def _comma_load(s: str, lang: str) -> int:
    """Count commas inside the clause (trailing comma excluded — it's the split boundary)."""
    commas = _comma_chars(lang)
    s = _strip_trailing_commas(s.strip(), lang)
    return sum(1 for c in s if c in commas)


def _split_keep_comma(sentence: str, lang: str) -> list[str]:
    """Split a sentence after each comma (comma stays on the left part).
    Commas between digits (e.g. 10,000) are NOT split points. For spaced
    languages the comma must also end its token (next char is whitespace):
    a mid-token comma (e.g. ``so,"``) would divide the token and desync the
    token-to-word_data index zip in ``split_at_sentence_end``."""
    commas = _comma_chars(lang)
    no_spaces = _no_spaces(lang)
    out: list[str] = []
    buf: list[str] = []
    n = len(sentence)
    for i, ch in enumerate(sentence):
        buf.append(ch)
        if ch in commas:
            prev = sentence[i - 1] if i > 0 else ""
            nxt = sentence[i + 1] if i + 1 < n else ""
            if prev.isdigit() and nxt.isdigit():
                continue
            if not no_spaces and nxt and not nxt.isspace():
                continue
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def _comma_clauses(sentence: str, lang: str, min_len: int) -> list[str]:
    """Group comma-delimited pieces into cue clauses.

    A clause flushes once it reaches ``min_len`` visual chars. The comma-load
    cap (<=1) prevents piling repeated short fragments (e.g. a name said several
    times) onto one line. Trailing commas are kept for downstream stripping."""
    pieces = _split_keep_comma(sentence, lang)
    clauses: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and _comma_load(buf + piece, lang) > 1:
            clauses.append(buf)
            buf = piece
        else:
            buf += piece
        if _visual_len(buf, lang) >= min_len:
            clauses.append(buf)
            buf = ""
    if buf:
        if (
            clauses
            and _visual_len(buf, lang) < min_len
            and _comma_load(clauses[-1] + buf, lang) <= 1
        ):
            clauses[-1] += buf
        else:
            clauses.append(buf)
    return clauses


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


def _snap_mid_to_phrase_boundary(
    toks: list[str], text: str, lang: str, target: int
) -> int:
    """Snap a midpoint index to the best nearby phrase boundary.

    Raw ``mid = n//2`` can land inside a phrase (e.g. splitting です into で|す).
    Among legal boundaries, prefer one whose left side does not end on a sticky
    token (の/的/...), then the one nearest ``target``. Falls back to ``target``
    when the whole clause is a single phrase.
    """
    atoms = [{"text": t} for t in toks]
    boundaries = sorted(_phrase_boundary_atoms(atoms, text, lang))
    n = len(toks)
    # Only consider boundaries 1..n-1 (index 0 = start of first phrase, not a
    # valid split point; index n = after last atom, also not valid).
    valid = [b for b in boundaries if 0 < b < n]
    if not valid:
        # degenerate: whole clause is one BudouX phrase (no internal boundary) → midpoint
        return max(1, min(target, n - 1))

    def left_pen(b: int) -> int:
        # penalty of the word ending just before the break: atoms from the last
        # phrase start below b through b-1 (whole-word semantics for zh tables)
        ws = max((x for x in boundaries if x < b), default=0)
        return line_end_penalty("".join(toks[ws:b]), lang)

    return min(valid, key=lambda b: (left_pen(b), abs(b - target)))


@functools.cache  # one pattern per language; avoids recompiling per clause
def _build_split_pattern(lang: str) -> re.Pattern:
    comma = get_comma(lang)
    extra_terminals = ";。！？" if _no_spaces(lang) else ";"
    conj = conjunctions_by_language.get(lang, set())
    terminal_class = re.escape(comma) + "".join(re.escape(c) for c in extra_terminals)
    if conj:
        conj_alt = "|".join(re.escape(c) for c in sorted(conj, key=len, reverse=True))
        if _no_spaces(lang):
            # No whitespace boundary; split right after terminal or right before conjunction
            return re.compile(rf"(?<=[{terminal_class}])|(?={conj_alt})")
        return re.compile(rf"(?<=[{terminal_class}])\s+|(?<=\s)(?=\b(?:{conj_alt})\b)")
    if _no_spaces(lang):
        return re.compile(rf"(?<=[{terminal_class}])")
    return re.compile(rf"(?<=[{terminal_class}])\s+")


def split_sentence_heuristically(
    sentence: str,
    max_line_length: int,
    max_lines: int,
    lang: str,
    split_at_comma: bool = True,
    comma_split_min_len: int | None = None,
) -> list[str]:
    if split_at_comma:
        if comma_split_min_len is None:
            comma_split_min_len = default_comma_split_min_len(lang)
        clauses = _comma_clauses(sentence, lang, comma_split_min_len)
    else:
        clauses = [sentence]
    out: list[str] = []
    for clause in clauses:
        out.extend(_fit_split_clause(clause, max_line_length, max_lines, lang))
    return [p for p in out if p]


def _repack_parts(
    parts: list[str], max_line_length: int, max_lines: int, lang: str
) -> list[str]:
    """Greedily merge adjacent terminal/conjunction parts back up to the budget.

    The split pattern marks *candidate* break points, not mandates: keeping every
    part separate shatters a long sentence into fragment cues ("and bought milk" |
    "and eggs"). Mirrors the accumulate-then-flush behavior of _comma_clauses.
    """
    sep = "" if _no_spaces(lang) else " "
    packed: list[str] = []
    for part in parts:
        if packed:
            cand = packed[-1] + sep + part
            if _fits_budget(cand, max_line_length, max_lines, lang):
                packed[-1] = cand
                continue
            balanced = _rebalance_adjacent_parts(
                packed[-1], part, max_line_length, max_lines, lang
            )
            if balanced is not None:
                packed[-1], part = balanced
        packed.append(part)
    return packed


def _visual_midpoint_index(tokens: list[str], lang: str) -> int:
    """Token boundary nearest the visual midpoint (never 0 or len(tokens))."""
    if len(tokens) < 2:
        return 1
    return min(
        range(1, len(tokens)),
        key=lambda i: abs(
            _vis_width(_join(tokens[:i], lang)) - _vis_width(_join(tokens[i:], lang))
        ),
    )


def _split_part_to_budget(
    part: str, max_line_length: int, max_lines: int, lang: str
) -> list[str]:
    """Recursively split a multi-token part until every result fits.

    A single indivisible token is deliberately returned intact: text-only
    splitting here would desynchronise it from its one aligned ``word_data``
    unit.  The timed atom stage owns the token-internal emergency fallback.
    """
    part = part.strip()
    if not part or _fits_budget(part, max_line_length, max_lines, lang):
        return [part] if part else []
    tokens = _tokens(part, lang)
    if len(tokens) < 2:
        return [part]
    target = _visual_midpoint_index(tokens, lang)
    if _no_spaces(lang):
        mid = _snap_mid_to_phrase_boundary(tokens, part, lang, target)
    else:
        mid = legal_break_index(tokens, lang, target)
    if not 0 < mid < len(tokens):
        return [part]
    left, right = _join(tokens[:mid], lang), _join(tokens[mid:], lang)
    return _split_part_to_budget(
        left, max_line_length, max_lines, lang
    ) + _split_part_to_budget(right, max_line_length, max_lines, lang)


def _rebalance_adjacent_parts(
    left: str,
    right: str,
    max_line_length: int,
    max_lines: int,
    lang: str,
) -> tuple[str, str] | None:
    """Move a legal boundary between two fitting parts to remove a short side.

    This runs only inside one sentence/clause after overlong parts have already
    been split.  It never merges the pair; both new sides must independently fit
    the display budget, preserve order, and improve visual balance.
    """
    sep = "" if _no_spaces(lang) else " "
    combined = left.rstrip() + sep + right.lstrip()
    tokens = _tokens(combined, lang)
    if len(tokens) < 2:
        return None
    if _no_spaces(lang):
        atoms = [{"text": token} for token in tokens]
        candidates = sorted(_phrase_boundary_atoms(atoms, combined, lang) - {0})
    else:
        candidates = list(range(1, len(tokens)))
    old_imbalance = abs(_vis_width(left) - _vis_width(right))
    old_tokens = _tokens(left, lang)
    old_penalty = line_end_penalty(old_tokens[-1], lang) if old_tokens else 0
    choices: list[tuple[int, int, int, str, str]] = []
    for i in candidates:
        new_left = _join(tokens[:i], lang)
        new_right = _join(tokens[i:], lang)
        if not new_left or not new_right:
            continue
        if not _fits_budget(new_left, max_line_length, max_lines, lang):
            continue
        if not _fits_budget(new_right, max_line_length, max_lines, lang):
            continue
        imbalance = abs(_vis_width(new_left) - _vis_width(new_right))
        if imbalance >= old_imbalance:
            continue
        penalty = line_end_penalty(tokens[i - 1], lang)
        if penalty > old_penalty:
            continue
        choices.append((penalty, imbalance, -_vis_width(new_left), new_left, new_right))
    if not choices:
        return None
    _penalty, _imbalance, _left_width, new_left, new_right = min(choices)
    return new_left, new_right


def _fit_split_clause(
    clause: str,
    max_line_length: int,
    max_lines: int,
    lang: str,
) -> list[str]:
    """Keep a clause whole if it fits ``max_lines``; otherwise split at
    terminals/conjunctions (repacked to the budget), then fall back to an even
    token split."""
    clause = clause.strip()
    if not clause:
        return []
    if _fits_budget(clause, max_line_length, max_lines, lang):
        return [clause]

    pattern = _build_split_pattern(lang)
    candidate_parts = [p.strip() for p in pattern.split(clause) if p and p.strip()]
    fitted_parts: list[str] = []
    for part in candidate_parts:
        fitted_parts.extend(
            _split_part_to_budget(part, max_line_length, max_lines, lang)
        )
    return _repack_parts(fitted_parts, max_line_length, max_lines, lang)


def _segment_sentences(text: str, lang: str) -> list[str]:
    """Sentence boundaries from pysbd, falling back to a terminal-punctuation regex.

    Two distinct failures land on the same fallback -- pysbd absent, and pysbd
    having no model for this language (``Segmenter(language="yue")`` raises,
    which also hits ``pt``/``ko``). Both are recorded through
    :func:`providers.note_degraded`; the returned sentences are unchanged.
    """
    try:
        import pysbd  # type: ignore

        try:
            seg = pysbd.Segmenter(language=lang, clean=False)
            return [s for s in seg.segment(text) if s and s.strip()]
        except Exception:
            note_degraded("sentences", "pysbd-language-unsupported:regex")
    except ImportError:
        note_degraded("sentences", "pysbd-missing:regex")
    return [s for s in re.split(r"(?<=[.!?。！？])\s*", text) if s and s.strip()]


def _snap_sentence_breaks(text: str, sentences: list[str], lang: str) -> list[str]:
    """Drop sentence boundaries that fall inside a whitespace-delimited token.

    ASR tokens can carry internal sentence punctuation (e.g. laughter transcribed
    as a single CJK run ``哈哈哈哈哈！哇。``). The segmenter splits there, which
    inflates the token count versus ``word_data`` — and the index zip in
    ``split_at_sentence_end`` then shifts every later cue's timing for the rest
    of the segment. Rebuild sentences as exact slices of ``text`` keeping only
    boundaries adjacent to whitespace. No-space languages pair by char count,
    which intra-token splits cannot desync.
    """
    if _no_spaces(lang) or len(sentences) < 2:
        return sentences
    cuts: list[int] = []
    pos = 0
    for sent in sentences[:-1]:
        idx = text.find(sent, pos)
        if idx < 0:
            return sentences  # segmenter rewrote content; nothing safe to snap
        pos = idx + len(sent)
        cuts.append(pos)
    pieces: list[str] = []
    last = 0
    for cut in cuts:
        if cut >= len(text) or text[cut].isspace() or text[cut - 1].isspace():
            piece = text[last:cut]
            if piece.strip():
                pieces.append(piece)
                last = cut
    tail = text[last:]
    if tail.strip():
        pieces.append(tail)
    return pieces or sentences


def _units_match(word_data: list[Unit], at: int, tokens: Sequence[str]) -> bool:
    """True when ``word_data`` spells ``tokens`` exactly starting at index ``at``.

    ``reinject_punct`` can glue a boundary space onto a unit ('开 ' at a
    CJK<->Latin seam); the cursor arithmetic is whitespace-insensitive, so
    content is compared stripped.
    """
    if at < 0 or at + len(tokens) > len(word_data):
        return False
    return all(
        (word_data[at + j].get("word") or "").strip() == tok.strip()
        for j, tok in enumerate(tokens)
    )


def _anchor_cursor(
    word_data: list[Unit],
    cursor: int,
    clause_tokens: Sequence[str],
    max_shift: int = 8,
) -> tuple[int, bool]:
    """Verify the clause's tokens match ``word_data`` at ``cursor``; search nearby on mismatch.

    Returns ``(start_index, ok)``. The index contract can still break on inputs
    we have not anticipated (ghost or lost units); rather than silently shifting
    every later cue, re-anchor on content within ``max_shift`` units, else keep
    the cursor and report ``ok=False`` — ``_locate_clause_units`` then widens the
    search to the whole remaining stream.
    """
    if _units_match(word_data, cursor, clause_tokens):
        return cursor, True
    for d in range(1, max_shift + 1):
        if _units_match(word_data, cursor + d, clause_tokens):
            return cursor + d, True
        if _units_match(word_data, cursor - d, clause_tokens):
            return cursor - d, True
    return cursor, False


def _unit_word_index(word_data: Sequence[Unit]) -> dict[str, list[int]]:
    """Ascending positions of every distinct unit surface in ``word_data``.

    A wide re-anchor scan always starts from a known first token, so probing only
    that token's positions keeps recovery near O(occurrences) instead of walking
    the whole stream once per damaged clause.
    """
    index: dict[str, list[int]] = {}
    for i, unit in enumerate(word_data):
        index.setdefault((unit.get("word") or "").strip(), []).append(i)
    return index


def _find_clause_forward(
    word_data: list[Unit],
    index: dict[str, list[int]],
    start: int,
    tokens: Sequence[str],
) -> int | None:
    """First position at or after ``start`` whose units spell ``tokens`` exactly."""
    if not tokens:
        return None
    positions = index.get(tokens[0].strip())
    if not positions:
        return None
    for at in positions[bisect_left(positions, max(0, start)) :]:
        if _units_match(word_data, at, tokens):
            return at
    return None


@dataclass(frozen=True)
class _ClausePlan:
    """One cue clause plus the ``word_data`` arithmetic it needs.

    ``anchor_tokens`` is what the clause must spell in ``word_data`` (whole words
    for spaced languages, non-space characters otherwise). It holds exactly
    ``unit_count`` entries, so the same tuple doubles as the surface list for a
    synthesized proportional fill when the clause cannot be located.
    """

    text: str
    anchor_tokens: tuple[str, ...]
    unit_count: int


def _clause_plans(
    text: str,
    lang: str,
    max_line_length: int,
    max_lines: int,
    split_at_comma: bool,
    comma_split_min_len: int | None,
    defer_length_split: bool,
) -> list[_ClausePlan]:
    """Segment ``text`` into cue clauses, each with its ``word_data`` footprint."""
    sentences = _snap_sentence_breaks(text, _segment_sentences(text, lang), lang)
    plans: list[_ClausePlan] = []
    for sent in sentences:
        if defer_length_split:
            min_len = (
                default_comma_split_min_len(lang)
                if comma_split_min_len is None
                else comma_split_min_len
            )
            clauses = _comma_clauses(sent, lang, min_len) if split_at_comma else [sent]
        else:
            clauses = split_sentence_heuristically(
                sent,
                max_line_length,
                max_lines,
                lang,
                split_at_comma,
                comma_split_min_len,
            )
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            clause_tokens = _tokens(clause, lang)
            if _no_spaces(lang):
                # word_data is char-level for CJK; advance by non-space char count.
                # Anchor on the same per-char granularity (reinject_punct emits one
                # item per non-whitespace char, so units match chars 1:1).
                anchor_tokens = tuple(c for c in clause if not c.isspace())
                unit_count = sum(_token_char_count(t) for t in clause_tokens)
            else:
                anchor_tokens = tuple(clause_tokens)
                unit_count = len(clause_tokens)
            plans.append(_ClausePlan(clause, anchor_tokens, unit_count))
    return plans


def _proportional_units(tokens: Sequence[str], window: Sequence[Unit]) -> list[Unit]:
    """Spread ``tokens`` evenly across the span ``window`` covers.

    Only used for a clause whose own units cannot be located anywhere: the cue
    still gets monotone, non-overlapping timing that fully covers its own window
    instead of borrowing a neighbouring clause's timestamps. Returns ``[]`` when
    the window carries no usable span, leaving the caller's estimate in charge.
    """
    start = _span_start(window)
    end = _span_end(window)
    if not tokens or start is None or end is None or end < start:
        return []
    first, last = float(start), float(end)
    step = (last - first) / len(tokens)
    final = len(tokens) - 1
    return [
        {
            "word": tok,
            "start": first + i * step,
            "end": last if i == final else first + (i + 1) * step,
        }
        for i, tok in enumerate(tokens)
    ]


def _resync_position(
    word_data: list[Unit],
    index: dict[str, list[int]],
    cursor: int,
    plans: Sequence[_ClausePlan],
    position: int,
) -> int | None:
    """Unit index where the next locatable clause after ``position`` begins."""
    for plan in plans[position + 1 : position + 1 + RESYNC_LOOKAHEAD_CLAUSES]:
        at = _find_clause_forward(word_data, index, cursor, plan.anchor_tokens)
        if at is not None:
            return at
    return None


def _locate_clause_units(
    word_data: list[Unit],
    index: dict[str, list[int]],
    cursor: int,
    plans: Sequence[_ClausePlan],
    position: int,
    *,
    anchored: bool,
) -> tuple[list[Unit], int, bool]:
    """Return ``(units, next_cursor, located)`` for the clause at ``position``.

    The blind index cursor is only trusted where the clause's content actually
    sits. ``_anchor_cursor`` repairs a local slip; a wider one (a ghost run, a
    clause whose units were lost upstream) is repaired by a forward content scan
    over the entire remaining stream. A clause that matches nowhere keeps its
    text but gets proportional timing across the window up to the next locatable
    clause, and the cursor resyncs on that clause's content — so the damage stops
    at the one clause instead of shifting every later cue in the segment.

    ``located`` is False only on that proportional path: its unit times are
    invented tiling, not measurements, so callers must not promote them to
    acoustic anchors.
    """
    plan = plans[position]
    if not anchored or not plan.anchor_tokens:
        return (
            word_data[cursor : cursor + plan.unit_count],
            cursor + plan.unit_count,
            True,
        )
    local, ok = _anchor_cursor(word_data, cursor, plan.anchor_tokens)
    start_at: int | None = (
        local
        if ok
        else _find_clause_forward(word_data, index, cursor, plan.anchor_tokens)
    )
    if start_at is not None:
        if start_at != cursor:
            log.warning(
                "cue/word desync at clause %d %r: cursor %d -> %d (resynced on content)",
                position,
                plan.text[:40],
                cursor,
                start_at,
            )
        end_at = start_at + plan.unit_count
        return word_data[start_at:end_at], end_at, True
    resume = _resync_position(word_data, index, cursor, plans, position)
    # No resync point in reach: fall back to the blind span so later clauses can
    # still re-anchor themselves instead of being starved of units.
    window_end = (
        min(cursor + plan.unit_count, len(word_data)) if resume is None else resume
    )
    log.warning(
        "cue/word desync at clause %d %r: units unlocatable, proportional timing "
        "over units %d-%d, cursor -> %d",
        position,
        plan.text[:40],
        cursor,
        window_end,
        window_end,
    )
    filled = _proportional_units(plan.anchor_tokens, word_data[cursor:window_end])
    return filled, window_end, False


def split_at_sentence_end(
    text: str,
    word_data: list[Unit],
    lang: str,
    max_line_length: int,
    max_lines: int,
    split_at_comma: bool = True,
    comma_split_min_len: int | None = None,
    *,
    defer_length_split: bool = False,
) -> list[Cue]:
    plans = _clause_plans(
        text,
        lang,
        max_line_length,
        max_lines,
        split_at_comma,
        comma_split_min_len,
        defer_length_split,
    )
    cues: list[Cue] = []
    cursor = 0
    # Content verification needs unit texts; legacy callers without a "word"
    # key keep the blind index cursor.
    anchored = bool(word_data) and "word" in word_data[0]
    index = _unit_word_index(word_data) if anchored else {}
    for position, plan in enumerate(plans):
        chunk_words, cursor, located = _locate_clause_units(
            word_data, index, cursor, plans, position, anchored=anchored
        )
        if chunk_words:
            start = next((w["start"] for w in chunk_words if "start" in w), None)
            end = next((w["end"] for w in reversed(chunk_words) if "end" in w), None)
        else:
            start = end = None
        # The raw clause span doubles as the acoustic anchor only when the units
        # were actually located; proportional tiling and the fabricated fallback
        # below invent time, which is not evidence.
        speech_start, speech_end = (start, end) if located else (None, None)
        if start is None or end is None:
            # No timing data: extend from previous cue end or estimate from word count
            prev_end = cues[-1]["end"] if cues else 0.0
            start = start if start is not None else prev_end
            end = (
                end
                if end is not None
                else start + max(1.0, plan.unit_count / DEFAULT_DESIRED_WPS)
            )
            speech_start = speech_end = None
        cues.append(
            {
                "text": plan.text,
                "start": start,
                "end": end,
                "word_data": chunk_words,
                "speech_start": speech_start,
                "speech_end": speech_end,
            }
        )
    return cues


@dataclass(frozen=True)
class SplitThresholds:
    """Gap-aware segmentation knobs — one typed source for field names + defaults.

    Built from ``config.gap_thresholds()``'s mapping at the ``smart_split_segments`` boundary via
    :meth:`from_mapping`. Passing ``thresholds=None`` to ``smart_split_segments`` selects the
    legacy length-break-only path (gap/duration breaks and the cleanup pass are skipped), so these
    values are only read in gap-aware mode.
    """

    clause_ms: int = 400
    vad_skip_ms: int = 1000
    offline_ms: int = 700
    min_cue_s: float = 0.5
    max_cue_s: float = 7.0
    glue_gap_s: float = GLUE_MAX_GAP_S
    # Reading-speed linger (0 = off): a cue shorter than reading_chars/cps extends
    # into the following gap, capped at LINGER_CAP_S past speech end. lag_out_s is
    # a flat tail pad applied to every cue end (0 = off). config.gap_thresholds
    # supplies per-language values; the dataclass defaults keep both off so direct
    # constructions (tests/legacy) preserve exact timing.
    cps: float = 0.0
    lag_out_s: float = 0.0
    # Shot-change pairing window (0 = off): a cue boundary within this of a
    # detected cut gets the Netflix zone treatment (see _snap_to_shots). 11
    # frames @24fps covers the outermost adjustment zone. Only consulted when
    # the caller passes shot_changes.
    shot_snap_s: float = 11.0 / 24.0

    @classmethod
    def from_mapping(cls, d: dict) -> SplitThresholds:
        """Build from a (possibly partial) mapping, ignoring unknown keys and filling defaults."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SplitContext:
    """Per-call segmentation context: language, line budget, thresholds, VAD.

    Bundles the invariants of one ``split_long_cues_with_word_timings`` call so
    they travel into the packing loop (``_pack_atoms_into_chunks`` /
    ``_classify_atom_break``) as one value instead of five parallel parameters.
    ``do_new=False`` is the legacy length-break-only path: gap/duration breaks
    are disabled and ``th`` is a never-read placeholder.
    """

    lang: str
    max_line_length: int
    max_lines: int
    th: SplitThresholds
    do_new: bool
    speech_spans: list[tuple[float, float]] | None = None


def _gap_ms(prev_end: float | None, next_start: float | None) -> float | None:
    """Inter-atom gap in ms (None when either bound is missing/non-positive)."""
    if prev_end is None or next_start is None:
        return None
    gap = (next_start - prev_end) * 1000.0
    return gap if gap > 0 else None


def _atom_end_pen(atom: dict) -> int:
    """Line-end penalty for breaking after ``atom``.

    Reads the precomputed ``end_pen`` (set by ``_attach_end_penalties``, which has
    whole-word and lang context); falls back to the bare char-table score for
    callers that pass raw atoms (unit tests, legacy paths)."""
    pen = atom.get("end_pen")
    return line_end_penalty(atom["text"]) if pen is None else pen


def _atom_start_pen(atom: dict) -> int:
    """Cue-start penalty for the segmented phrase beginning at ``atom``."""

    pen = atom.get("start_pen")
    return 0 if pen is None else int(pen)


def _atom_boundary_pen(atom: dict) -> int:
    """Optional POS modifier/head damage for the boundary before ``atom``."""

    pen = atom.get("boundary_pen")
    return 0 if pen is None else int(pen)


def _attach_end_penalties(
    atoms: list[dict], boundary: set[int] | None, lang: str
) -> None:
    """Precompute ``atom["end_pen"]`` — penalty for ending a cue/line on this atom.

    Spaced langs (``boundary is None``): the atom is a whole word; score it directly
    (en closed-class table). No-space langs: score the *word* — the atom span since
    the last phrase boundary — so zh whole-word semantics hold (目的 never matches
    的) and the ja kana check still reads the word's last char. For ja, UniDic POS
    (ja_pos_end_penalties) overrides the char table where it scores a token end,
    disambiguating 準体の from 格助詞の. Atoms a break cannot legally follow (next
    atom mid-phrase) score 0; they are never candidates.
    """
    n = len(atoms)
    for atom in atoms:
        atom["start_pen"] = 0
        atom["boundary_pen"] = 0

    starts = list(range(n)) if boundary is None else sorted(boundary | {0})
    for position, start in enumerate(starts):
        if not 0 <= start < n:
            continue
        end = starts[position + 1] if position + 1 < len(starts) else n
        phrase = _join([atom["text"] for atom in atoms[start:end]], lang)
        atoms[start]["start_pen"] = line_start_penalty(phrase, lang)

    pos_boundaries = [index for index in starts if 0 < index < n]
    for index, penalty in zh_pos_boundary_penalties(
        [atom["text"] for atom in atoms], pos_boundaries, lang
    ).items():
        atoms[index]["boundary_pen"] = penalty

    pos_map: dict[int, int] | None = None
    if lang == "ja" and boundary is not None:
        from .kinsoku import ja_pos_end_penalties

        pos_map = ja_pos_end_penalties("".join(a["text"] for a in atoms))
    word_start = 0
    char_end = 0  # cumulative non-space char count through atom k
    for k, a in enumerate(atoms):
        char_end += _token_char_count(a["text"])
        if boundary is not None and k in boundary:
            word_start = k
        if boundary is None:
            a["end_pen"] = line_end_penalty(a["text"], lang)
        elif k + 1 >= n or (k + 1) in boundary:
            pen = pos_map.get(char_end - 1) if pos_map is not None else None
            if pen is None:  # no POS source / mid-token offset -> char-table fallback
                word = "".join(x["text"] for x in atoms[word_start : k + 1])
                pen = line_end_penalty(word, lang)
            a["end_pen"] = pen
        else:
            a["end_pen"] = 0


def _best_len_break_pos(
    cur: list[dict],
    cur_bnd: list[bool],
    at_boundary_next: bool,
    next_atom: dict | None = None,
    ctx: SplitContext | None = None,
) -> int:
    """Choose a grammatical, audible split from the boundaries already seen.

    Candidates: break before the incoming atom (pos n) and any internal
    phrase-start k (0<k<n).  Hard edge damage (``的|特性`` or
    ``肉身|的``) ranks first.  Within grammatically legal choices, POS damage,
    a graded real inter-atom pause, and tiny-fragment avoidance select the
    boundary that best follows the heard phrase rather than the fullest line.
    Legacy callers without ``next_atom`` retain the old sticky-end/fullest
    behavior.  Falls back to n when no candidate exists.
    """
    n = len(cur)
    positions: list[int] = []
    if at_boundary_next:
        positions.append(n)
    for k in range(1, n):
        if cur_bnd[k]:
            positions.append(k)
    enhanced = next_atom is not None and ctx is not None
    lang = ctx.lang if ctx is not None else ""
    if not positions:
        if not enhanced:
            return n
        # Emergency only: one parser phrase has exceeded the 1.5x safety cap
        # (or the duration cap) and exposes no legal internal boundary.  Search
        # atom edges and let balance choose a midpoint instead of always cutting
        # immediately before the final character (天気で|す).
        positions = list(range(1, n + 1))

    def score(k: int) -> tuple[int, int, int, int]:
        if not enhanced:
            return (_atom_end_pen(cur[k - 1]), 0, 0, -k)
        right = cur[k] if k < n else next_atom
        if right is None:
            return (_atom_end_pen(cur[k - 1]), 0, 0, -k)
        # Surface kinsoku is a hard relation; POS is deliberately soft enough
        # that a clearly audible pause can still win over a mild noun-noun hint.
        hard_damage = 3 * (_atom_end_pen(cur[k - 1]) + _atom_start_pen(right))
        gap_damage = 0
        if enhanced:
            left_end = cur[k - 1].get("end")
            right_start = right.get("start")
            if isinstance(left_end, (int, float)) and isinstance(
                right_start, (int, float)
            ):
                pause_ms = max(0.0, (float(right_start) - float(left_end)) * 1000)
                if pause_ms < 40:
                    gap_damage = 3
                elif pause_ms < 120:
                    gap_damage = 2
                elif pause_ms < 220:
                    gap_damage = 1
        right_known = cur[k:] + ([next_atom] if next_atom is not None else [])
        left_width = _vis_width(_join([atom["text"] for atom in cur[:k]], lang))
        right_width = _vis_width(_join([atom["text"] for atom in right_known], lang))
        micro_damage = (
            8
            if min(left_width, right_width) <= 1
            else 4
            if min(left_width, right_width) <= 2
            else 0
        )
        soft_damage = _atom_boundary_pen(right) + gap_damage + micro_damage
        imbalance = abs(left_width - right_width)
        return hard_damage, soft_damage, imbalance, -k

    return min(positions, key=score)


def _classify_atom_break(
    cur: list[dict],
    atom: dict,
    *,
    at_boundary: bool,
    ctx: SplitContext,
    cur_bnd: Sequence[bool] | None = None,
) -> tuple[bool, bool, bool]:
    """Decide ``(gap_break, dur_break, len_break)`` for appending ``atom`` after ``cur``.

    Pure: reads ``cur``/``atom``/``ctx``, mutates nothing. All-False when ``cur`` is empty
    (the first atom of a chunk never breaks).
    - gap_break: a qualifying inter-atom pause, but only at a phrase boundary, and suppressed in
      the clause_ms..vad_skip_ms zone when it would strand a sticky token at line end.
    - dur_break: hard last-resort cap when the running cue would exceed ``max_cue_s`` (ignores
      word boundaries — intra-word spans over the cap are rare and an overlong cue is worse).
    - len_break: the line budget overflows AND this atom is a legal (phrase-start) break point.
    """
    if not cur:
        return False, False, False
    th = ctx.th
    prev = cur[-1]
    # Gap/len breaks require a word boundary (no-space langs): atom must be a BudouX(ja)/jieba(zh)
    # phrase start. Guards against CTC timing errors on OOV chars creating spurious intra-word gaps
    # (e.g. 酒造り: 番酒造 OOV drift makes a 2.1s gap between 造 and り, but BudouX keeps 番酒造りが
    # as one phrase, suppressing the spurious split). The dur_break cap is exempt and always cuts.
    gap_break = (
        ctx.do_new
        and at_boundary
        and gap_qualifies(
            prev.get("end"),
            atom.get("start"),
            ctx.speech_spans,
            clause_ms=th.clause_ms,
            vad_skip_ms=th.vad_skip_ms,
            offline_ms=th.offline_ms,
        )
    )
    # In the clause_ms..vad_skip_ms zone, suppress the gap-split if it would strand a sticky
    # token at line end: ja 大樹の|村, zh ...的|... , en a hesitation after "the". True
    # silence (>=vad_skip_ms) always cuts — a real pause beats line-end aesthetics.
    if gap_break and (_atom_end_pen(prev) >= 2 or _atom_start_pen(atom) >= 2):
        gms = _gap_ms(prev.get("end"), atom.get("start"))
        if gms is not None and th.clause_ms <= gms < th.vad_skip_ms:
            gap_break = False
    tentative = _join([a["text"] for a in cur + [atom]], ctx.lang)
    len_overflow = not _fits_budget(
        tentative, ctx.max_line_length, ctx.max_lines, ctx.lang
    )
    # If the incoming atom is mid-phrase, retreat to an earlier legal boundary
    # already in ``cur``.  Waiting until 1.5x overflow lets a 19-char Chinese
    # sentence escape an 18-char budget and later creates tiny tail cues.
    has_internal_boundary = bool(cur_bnd and any(cur_bnd[1:]))
    len_break = len_overflow and (at_boundary or has_internal_boundary)
    # Boundary-less overlong run (a single phrase atom exceeding the budget by
    # 1.5x, e.g. a long katakana loan chain): bail out off-boundary rather than
    # emit a mega-line — _best_len_break_pos still prefers any earlier legal
    # boundary held in the running chunk.
    if (
        len_overflow
        and not len_break
        and _token_char_count(tentative)
        > round(FORCE_BREAK_FACTOR * ctx.max_line_length * ctx.max_lines)
    ):
        len_break = True
    start0 = _span_start(cur)
    dur_break = (
        ctx.do_new
        and start0 is not None
        and atom.get("end") is not None
        and (atom["end"] - start0) > th.max_cue_s
    )
    return gap_break, dur_break, len_break


def _hard_wrap_surface(text: str, line_budget: int) -> list[str]:
    """Split one indivisible surface token into line-sized pieces.

    Normal segmentation never calls this for ordinary words.  It is the final
    fallback for an alignment atom that is itself wider than a physical line
    (wrong-language unspaced text, a huge URL, or another coarse token).  A
    nearby whitespace is preferred; otherwise a character boundary is the only
    lossless place available.
    """
    if not text:
        return []
    clean = text.strip()
    wanted = max(1, math.ceil(_vis_width(clean) / max(1, line_budget)))
    parts = [clean]
    while len(parts) < wanted or any(
        _vis_width(part) > line_budget and len(part) > 1 for part in parts
    ):
        splittable = [
            (i, _vis_width(part)) for i, part in enumerate(parts) if len(part) > 1
        ]
        if not splittable:
            break
        index = max(splittable, key=lambda item: item[1])[0]
        divided = _split_surface_mid(parts[index])
        if divided is None:
            break
        parts[index : index + 1] = list(divided)
    return parts


def _split_surface_mid(text: str) -> tuple[str, str] | None:
    """Split a surface near its visual midpoint, preferring whitespace."""
    if len(text) < 2:
        return None
    candidates = range(1, len(text))
    mid = min(
        candidates,
        key=lambda i: (
            0 if text[i - 1].isspace() or text[i].isspace() else 1,
            abs(_vis_width(text[:i]) - _vis_width(text[i:])),
        ),
    )
    left, right = text[:mid].rstrip(), text[mid:].lstrip()
    return (left, right) if left and right else None


def _surface_parts_for_limits(
    text: str,
    start: float,
    end: float,
    ctx: SplitContext,
) -> list[tuple[str, float, float]]:
    """Return display-safe pieces with proportional spans for a coarse surface.

    Duration subdivision is activated only once the atom is structurally too
    wide (or occupies over half a line while itself exceeding the cue cap).
    This keeps a genuinely held short lexical word intact while preventing a
    paragraph-sized alignment atom from bypassing both hard limits.
    """
    line_budget = _line_budget_width(ctx.max_line_length, ctx.lang)
    parts = _hard_wrap_surface(text, line_budget)
    duration = max(0.0, end - start)
    duration_parts = (
        math.ceil(duration / ctx.th.max_cue_s)
        if ctx.do_new and ctx.th.max_cue_s > 0
        else 1
    )
    coarse_timed_atom = duration_parts > 1 and _vis_width(text) > line_budget / 2
    if len(parts) == 1 and not coarse_timed_atom:
        return [(text, start, end)]
    wanted = max(len(parts), duration_parts)
    while len(parts) < wanted:
        splittable = [
            (i, _vis_width(part)) for i, part in enumerate(parts) if len(part) > 1
        ]
        if not splittable:
            break
        index = max(splittable, key=lambda item: item[1])[0]
        divided = _split_surface_mid(parts[index])
        if divided is None:
            break
        parts[index : index + 1] = list(divided)
    if len(parts) == 1:
        return [(text, start, end)]
    step = duration / len(parts) if duration > 0 else 0.0
    return [
        (
            part,
            start + i * step,
            end if i == len(parts) - 1 else start + (i + 1) * step,
        )
        for i, part in enumerate(parts)
    ]


def _split_oversized_atom(atom: dict, cue: Cue, ctx: SplitContext) -> list[dict]:
    """Subdivide one structurally coarse atom; ordinary atoms pass through."""
    start = atom.get("start")
    end = atom.get("end")
    safe_start = cue["start"] if start is None else start
    safe_end = cue["end"] if end is None else end
    pieces = _surface_parts_for_limits(atom["text"], safe_start, safe_end, ctx)
    if len(pieces) == 1:
        return [atom]
    return [
        {"text": text, "start": piece_start, "end": piece_end, "end_pen": 0}
        for text, piece_start, piece_end in pieces
    ]


def _pack_with_oversized_fallback(
    atoms: list[dict],
    *,
    boundary: set[int] | None,
    ctx: SplitContext,
    cue: Cue,
) -> list[list[dict]]:
    """Pack normal runs and emit token-internal fallback pieces standalone."""
    chunks: list[list[dict]] = []
    run: list[dict] = []
    run_indices: list[int] = []

    def flush_run() -> None:
        if not run:
            return
        local_boundary = (
            None
            if boundary is None
            else {
                local_i
                for local_i, original_i in enumerate(run_indices)
                if original_i in boundary
            }
        )
        chunks.extend(_pack_atoms_into_chunks(run, boundary=local_boundary, ctx=ctx))
        run.clear()
        run_indices.clear()

    for i, atom in enumerate(atoms):
        pieces = _split_oversized_atom(atom, cue, ctx)
        if len(pieces) == 1:
            run.append(atom)
            run_indices.append(i)
            continue
        flush_run()
        chunks.extend([[piece] for piece in pieces])
    flush_run()
    return chunks


def _pack_atoms_into_chunks(
    atoms: list[dict],
    *,
    boundary: set[int] | None,
    ctx: SplitContext,
) -> list[list[dict]]:
    """Greedily pack atoms into chunks, cutting on the first qualifying gap/dur/len break.

    Time-forced breaks (gap/dur) cut immediately; a length overflow picks the phrase-boundary
    candidate with the smallest sticky-token penalty via ``_best_len_break_pos`` (Level 1).
    """
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_bnd: list[
        bool
    ] = []  # parallel to cur: True = phrase-start (legal len-break point)
    for i, atom in enumerate(atoms):
        at_boundary = boundary is None or i in boundary
        gap_break, dur_break, len_break = _classify_atom_break(
            cur,
            atom,
            at_boundary=at_boundary,
            ctx=ctx,
            cur_bnd=cur_bnd,
        )
        if cur and gap_break:
            chunks.append(cur)
            cur, cur_bnd = [atom], [at_boundary]
        elif cur and (dur_break or len_break):
            k = _best_len_break_pos(
                cur,
                cur_bnd,
                at_boundary,
                next_atom=atom,
                ctx=ctx,
            )
            chunks.append(cur[:k])
            cur, cur_bnd = cur[k:] + [atom], cur_bnd[k:] + [at_boundary]
        else:
            cur.append(atom)
            cur_bnd.append(at_boundary)
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_to_cue(chunk: list[dict], cue: Cue, lang: str) -> Cue:
    """Materialize a packed atom chunk into a cue dict (first/last non-None span, falling back
    to the parent cue's start/end).

    The emitted word_data is *atom*-level, one entry per packed atom, so each
    entry carries its ``text``: without that surface a later reader
    (``_build_atoms``) has nothing to reconcile against and falls back to a
    character cursor, which is one entry per character.

    The acoustic anchor takes the same raw span but *without* the parent-cue
    fallback: an untimed chunk has no acoustic evidence, and inheriting the
    parent's (possibly display-padded) bound would launder that pad into the raw
    layer permanently.
    """
    # the default is the parent cue's (required, non-None) bound, so the span is always a float
    start = cast(float, _span_start(chunk, cue["start"]))
    end = cast(float, _span_end(chunk, cue["end"]))
    return {
        "text": _join([a["text"] for a in chunk], lang),
        "start": start,
        "end": end,
        "word_data": [
            {"text": a["text"], "start": a["start"], "end": a["end"]} for a in chunk
        ],
        "speech_start": _span_start(chunk, None),
        "speech_end": _span_end(chunk, None),
    }


def _repair_bound_particle_cues(
    cues: list[Cue],
    *,
    lang: str,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
    connected_gap_s: float,
) -> list[Cue]:
    """Remove connected cue edges that strand an independently tagged particle.

    This is a final safety net for boundaries inherited from separate ASR
    segments or an earlier hard layout decision.  A direct merge is preferred;
    if it would exceed width/duration, the two cues are repartitioned at a
    better phrase/POS boundary while retaining every source unit and timestamp.
    True pauses are never crossed.  Whole-token/POS scoring distinguishes the
    particles ``了/地`` from lexical words such as ``了解/地方``.  Repartitioning
    is one-directional: a candidate edge may relieve a dangling tail but is never
    allowed to leave the left cue a worse ``line_end_penalty`` tail than the edge
    it replaces (see ``candidate_score``).
    """

    if lang not in {"zh", "yue", "ja"} or len(cues) < 2 or connected_gap_s <= 0:
        return cues

    work = [cast(Cue, dict(cue)) for cue in cues]
    i = 0
    while i + 1 < len(work):
        left, right = work[i], work[i + 1]
        left_units = list(left.get("word_data") or [])
        right_units = list(right.get("word_data") or [])
        if not left_units or not right_units:
            i += 1
            continue
        speech_end = _span_end(left_units, left.get("end"))
        speech_start = _span_start(right_units, right.get("start"))
        if speech_end is None or speech_start is None:
            i += 1
            continue
        gap = float(speech_start) - float(speech_end)
        if gap < -1e-6 or gap >= connected_gap_s:
            i += 1
            continue

        left_text = left["text"].replace("\n", "").strip()
        right_text = right["text"].replace("\n", "").strip()
        combined_text = _join([left_text, right_text], lang)
        units = left_units + right_units
        atoms = _build_atoms(
            combined_text,
            units,
            lang,
            max_atom_width=_line_budget_width(max_line_length, lang),
        )
        # Repartitioning hands `units[:unit_cut]` to the left cue, so it is only
        # safe while the atoms account for every source unit exactly; a desynced
        # pair (atoms stop short, or the cursor ran past the end) keeps the
        # boundary it already has.
        covers_all = bool(atoms) and (
            atoms[0]["_unit_start"],
            atoms[-1]["_unit_end"],
        ) == (0, len(units))
        if not covers_all:
            i += 1
            continue

        # The original edge sits where the left cue's units run out. Counting
        # units (not characters) is what makes this hold for a repacked pair,
        # whose entries are whole atoms.
        original = next(
            (
                atom_index
                for atom_index, atom in enumerate(atoms)
                if atom["_unit_start"] == len(left_units)
            ),
            None,
        )
        if original is None or not 0 < original < len(atoms):
            i += 1
            continue

        boundaries = _phrase_boundary_atoms(atoms, combined_text, lang)
        boundaries.update({0, original})
        _attach_end_penalties(atoms, boundaries, lang)
        original_right = atoms[original]
        # Only repair high-confidence particle/function-word damage.  Ordinary
        # noun or clause boundaries are left to the main/model selector.
        if (
            _atom_end_pen(atoms[original - 1]) < 2
            and _atom_start_pen(original_right) < 2
        ):
            i += 1
            continue

        outer_start = float(left["start"])
        outer_end = float(right["end"])
        if outer_end - outer_start <= max_cue_s + 1e-9 and _fits_budget(
            combined_text, max_line_length, max_lines, lang
        ):
            merged_speech_start, merged_speech_end = combine_speech(left, right)
            work[i : i + 2] = [
                {
                    "text": combined_text,
                    "start": left["start"],
                    "end": right["end"],
                    "word_data": units,
                    "speech_start": merged_speech_start,
                    "speech_end": merged_speech_end,
                }
            ]
            if i:
                i -= 1
            continue

        starts = sorted(boundaries - {0, len(atoms)})
        # The tail the inherited edge already leaves: a repartition may relieve a
        # dangling one, but must never hand the left cue a worse tail than this.
        original_tail_pen = _atom_end_pen(atoms[original - 1])

        def candidate_score(k: int) -> tuple[int, int, int, int, int]:
            edge_right = atoms[k]
            # ``hard`` sums the two sides of the edge, so a bound particle sitting
            # at the left cue's END and the same particle heading the right cue
            # score an equal tie there and ``soft`` picks either side.  That lets a
            # repartition strand a 格助詞 / 的-class particle on a cue tail — the
            # >= 2 ``line_end_penalty`` position — where the original edge had a
            # clean one.  This leading component breaks the tie asymmetrically:
            # any candidate whose left tail scores WORSE than the original edge's
            # ranks strictly below it, so the pass can only relieve a dangling tail,
            # never manufacture one.  The signal is the same ``end_pen`` the rest of
            # the pass scores with (whole-word char table for zh/yue, UniDic Level-2
            # map for ja); candidates that leave the tail as good or better keep
            # competing on the old signals exactly as before.
            worsens_tail = 1 if _atom_end_pen(atoms[k - 1]) > original_tail_pen else 0
            hard = 3 * (_atom_end_pen(atoms[k - 1]) + _atom_start_pen(edge_right))
            prev_end = atoms[k - 1].get("end")
            next_start = edge_right.get("start")
            pause = 0
            if isinstance(prev_end, (int, float)) and isinstance(
                next_start, (int, float)
            ):
                pause_ms = max(0.0, (float(next_start) - float(prev_end)) * 1000)
                if pause_ms < 40:
                    pause = 3
                elif pause_ms < 120:
                    pause = 2
                elif pause_ms < 220:
                    pause = 1
            left_surface = _join([atom["text"] for atom in atoms[:k]], lang)
            right_surface = _join([atom["text"] for atom in atoms[k:]], lang)
            small = min(_vis_width(left_surface), _vis_width(right_surface))
            micro = 8 if small <= 1 else 4 if small <= 2 else 0
            soft = _atom_boundary_pen(edge_right) + pause + micro
            return (
                worsens_tail,
                hard,
                soft,
                abs(_vis_width(left_surface) - _vis_width(right_surface)),
                -k,
            )

        original_score = candidate_score(original)
        choices: list[tuple[tuple[int, int, int, int, int], int, int]] = []
        for k in starts:
            if k == original:
                continue
            left_chunk, right_chunk = atoms[:k], atoms[k:]
            left_surface = _join([atom["text"] for atom in left_chunk], lang)
            right_surface = _join([atom["text"] for atom in right_chunk], lang)
            if not _fits_budget(
                left_surface, max_line_length, max_lines, lang
            ) or not _fits_budget(right_surface, max_line_length, max_lines, lang):
                continue
            left_start = _span_start(left_chunk, left.get("start"))
            left_end = _span_end(left_chunk, left.get("end"))
            right_start = _span_start(right_chunk, right.get("start"))
            right_end = _span_end(right_chunk, right.get("end"))
            if (
                left_start is None
                or left_end is None
                or right_start is None
                or right_end is None
            ):
                continue
            if (
                float(left_end) - float(left_start) > max_cue_s + 1e-9
                or float(right_end) - float(right_start) > max_cue_s + 1e-9
            ):
                continue
            unit_cut = right_chunk[0]["_unit_start"]
            score = candidate_score(k)
            if score < original_score:
                choices.append((score, k, unit_cut))
        if not choices:
            i += 1
            continue

        _score, split_at, unit_cut = min(choices)
        new_left_atoms, new_right_atoms = atoms[:split_at], atoms[split_at:]
        # Repartitioning moves material between the two cues, so each side's
        # anchor is re-derived from the atoms it now owns -- with no display
        # fallback, unlike the visible bounds beside it.
        work[i : i + 2] = [
            {
                "text": _join([atom["text"] for atom in new_left_atoms], lang),
                "start": left["start"],
                "end": cast(float, _span_end(new_left_atoms, left.get("end"))),
                "word_data": units[:unit_cut],
                "speech_start": _span_start(new_left_atoms, None),
                "speech_end": _span_end(new_left_atoms, None),
            },
            {
                "text": _join([atom["text"] for atom in new_right_atoms], lang),
                "start": cast(float, _span_start(new_right_atoms, right.get("start"))),
                "end": right["end"],
                "word_data": units[unit_cut:],
                "speech_start": _span_start(new_right_atoms, None),
                "speech_end": _span_end(new_right_atoms, None),
            },
        ]
        i += 1
    return work


def split_long_cues_with_word_timings(
    cues: list[Cue],
    max_line_length: int,
    max_lines: int,
    min_duration: float,
    desired_wps: float,
    lang: str,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: SplitThresholds | None = None,
) -> list[Cue]:
    """Pack each cue's atoms into reading-sized cues using gap/duration/length breaks.

    ``min_duration`` / ``desired_wps`` are kept for back-compat (unused on the atom-based path).
    ``thresholds=None`` is the legacy length-break-only path: ``do_new=False`` disables the
    gap/duration breaks, so the threshold values are never read (the default instance is a
    never-read placeholder there).
    """
    do_new = thresholds is not None
    ctx = SplitContext(
        lang=lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        th=thresholds if thresholds is not None else SplitThresholds(),
        do_new=do_new,
        speech_spans=speech_spans,
    )
    new_cues: list[Cue] = []
    for cue in cues:
        word_data = list(cue.get("word_data") or [])
        if not word_data:
            new_cues.extend(
                _split_without_timings(cue, max_line_length, max_lines, lang, ctx=ctx)
            )
            continue
        atoms = _build_atoms(
            cue["text"],
            word_data,
            lang,
            max_atom_width=_line_budget_width(max_line_length, lang),
        )
        boundary = (
            _phrase_boundary_atoms(atoms, cue["text"], lang)
            if do_new and _no_spaces(lang)
            else None
        )
        if boundary is not None:
            boundary.update(
                i for i, atom in enumerate(atoms) if atom.get("forced_boundary")
            )
        _attach_end_penalties(atoms, boundary, lang)
        chunks = _pack_with_oversized_fallback(
            atoms, boundary=boundary, ctx=ctx, cue=cue
        )
        new_cues.extend(_chunk_to_cue(chunk, cue, lang) for chunk in chunks)
    return new_cues


def _split_without_timings(
    cue: Cue,
    max_line_length: int,
    max_lines: int,
    lang: str,
    *,
    ctx: SplitContext | None = None,
) -> list[Cue]:
    # One indivisible token cannot be wrapped by the normal line grouper.  Use
    # the same proportional emergency fallback as timed coarse atoms.
    if ctx is not None and len(_tokens(cue["text"], lang)) == 1:
        pieces = _surface_parts_for_limits(cue["text"], cue["start"], cue["end"], ctx)
        if len(pieces) > 1:
            return [
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "word_data": [],
                    # Proportional subdivision of an untimed cue: no acoustic
                    # evidence exists for any of these pieces.
                    "speech_start": None,
                    "speech_end": None,
                }
                for text, start, end in pieces
            ]
    formatted = split_subtitle(cue["text"], max_line_length, lang)
    lines = formatted.split("\n")
    chunks: list[list[str]] = []
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) == max_lines:
            chunks.append(buf)
            buf = []
    if buf:
        chunks.append(buf)
    if not chunks:
        return [cue]

    sep = "" if _no_spaces(lang) else " "
    total_chars = sum(len(sep.join(c)) for c in chunks) or 1
    start = cue["start"]
    duration = cue["end"] - cue["start"]
    out: list[Cue] = []
    for c in chunks:
        # Join without \n — the downstream SubtitlesWriter handles display wrapping.
        text = sep.join(c)
        proportion = len(text) / total_chars
        end = start + duration * proportion if duration > 0 else start
        out.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "word_data": [],
                "speech_start": None,
                "speech_end": None,
            }
        )
        start = end
    return out


def smart_split_segments(
    segments: list[dict[str, Any]],
    lang: str,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    min_duration: float = DEFAULT_MIN_DURATION,
    desired_wps: float = DEFAULT_DESIRED_WPS,
    split_at_comma: bool = True,
    comma_split_min_len: int | None = None,
    *,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: SplitThresholds | dict | None = None,
    shot_changes: list[float] | None = None,
) -> list[Cue]:
    """Run the full smart-split pipeline over aligned segments.

    Each segment must have ``text`` and ``words`` (with ``start``/``end``).
    Returns a flat list of cues with ``text``, ``start``, ``end``, ``word_data``.

    ``split_at_comma`` (default on) breaks at commas unless either side is
    shorter than ``comma_split_min_len`` visual chars. Each sentence/comma clause
    is its own cue, except a lone-word flicker cue with a sub-0.3s gap, which
    ``_glue_short_cues`` folds onto its nearer neighbor (forward or backward).
    """
    if max_line_length is None:
        max_line_length = default_max_line_length(lang)
    if max_lines is None:
        max_lines = default_max_lines(lang)  # zh/yue/ja -> 1, else 2
    # The packer budgets in native cells (a CJK preset counts one wide glyph per
    # cell) while the renderer measures half-width cells. Convert once so both
    # stages honour the same configured profile instead of the renderer silently
    # falling back to its built-in 42-column default.
    render_width = _line_budget_width(max_line_length, lang)
    # Accept a plain mapping (config.gap_thresholds / tests) and normalize to the typed form once.
    # th is None ⟺ legacy length-break-only mode (no gap/duration breaks, no cleanup pass).
    th = (
        SplitThresholds.from_mapping(thresholds)
        if isinstance(thresholds, dict)
        else thresholds
    )
    all_cues: list[Cue] = []
    for segment in segments:
        text = segment.get("text", "")
        words = segment.get("words", []) or []
        if not text:
            continue
        all_cues.extend(
            split_at_sentence_end(
                text,
                words,
                lang,
                max_line_length,
                max_lines,
                split_at_comma,
                comma_split_min_len,
                defer_length_split=th is not None and bool(words),
            )
        )
    cues = split_long_cues_with_word_timings(
        all_cues,
        max_line_length=max_line_length,
        max_lines=max_lines,
        min_duration=min(min_duration, 5.0 / 6.0),
        desired_wps=desired_wps,
        lang=lang,
        speech_spans=speech_spans,
        thresholds=th,
    )
    if th is not None:  # cleanup opt-in; legacy callers skip this
        cues = _repair_bound_particle_cues(
            cues,
            lang=lang,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=th.max_cue_s,
            connected_gap_s=th.clause_ms / 1000.0,
        )
        cues = _merge_micro_cues(
            cues,
            lang,
            max_gap_s=th.glue_gap_s,
            max_line_length=max_line_length,
            max_cue_s=th.max_cue_s,
            min_cue_s=th.min_cue_s,
            max_lines=max_lines,
        )
        cues = _glue_short_cues(
            cues,
            lang,
            max_gap_s=th.glue_gap_s,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=th.max_cue_s,
        )
        cues = _cleanup_cues(
            cues,
            min_cue_s=th.min_cue_s,
            max_cue_s=th.max_cue_s,
            cps=th.cps,
            lag_out_s=th.lag_out_s,
        )
        if shot_changes:
            cues = _snap_to_shots(
                cues,
                sorted(shot_changes),
                snap_s=th.shot_snap_s,
                max_cue_s=th.max_cue_s,
            )
    for cue in cues:
        cue["text"] = strip_punct_for_subtitles(cue["text"])
        if th is not None:  # stutter merging opt-in alongside gap-aware mode
            cue["text"] = _merge_stutters(cue["text"])
        # Display soft-wrap: fold over-budget cues into <=max_lines lines without
        # changing cue boundaries. Long Latin phrases inside CJK also collapse here.
        cue["text"] = wrap_cue_text(
            cue["text"], lang, max_lines, max_line_length=render_width
        )
    return cues
