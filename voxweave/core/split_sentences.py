"""Stage 1 of the segmentation engine: sentences, comma clauses and clause location.

``split_at_sentence_end`` splits a unit run at sentence ends (PySBD, or a regex
fallback), groups each sentence into comma clauses and locates every clause in
the word-level units. Stage 1 never splits for length: stage 2 does all length
breaking.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from .layout import (
    WIDE_GLYPH_LANGUAGES,
    _comma_chars,
    _no_spaces,
    _strip_trailing_commas,
    _token_char_count,
    _tokens,
    _visual_len,
)
from .providers import note_degraded
from .schema import Cue, Unit
from .split_atoms import _span_end, _span_start

log = logging.getLogger("voxweave.core.smart_split")


DEFAULT_DESIRED_WPS = 4.0  # reading speed (English wps) behind an untimed estimate

# Comma line-break: split into separate cues at commas, but only when both
# sides are at least this long (visual chars). Shorter clauses stay attached
# to a neighbor so we never strand a tiny fragment on its own cue.
DEFAULT_COMMA_SPLIT_MIN_LEN = 18  # latin / space-delimited
DEFAULT_COMMA_SPLIT_MIN_LEN_CJK = 6  # zh/yue/ja/ko: chars are ~2x visual width

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


def _segment_sentences(text: str, lang: str) -> list[str]:
    """Sentence boundaries from pysbd, falling back to a terminal-punctuation regex.

    Two distinct failures land on the same fallback -- pysbd absent, and pysbd
    having no model for this language (``Segmenter(language="yue")`` raises,
    which also hits ``pt``/``ko``). Both are recorded through
    :func:`providers.note_degraded`; the returned sentences are unchanged. A
    language pysbd has no model for is a designed fallback and is logged at
    DEBUG; pysbd failing on a language it does support keeps its warning.
    """
    try:
        import pysbd  # type: ignore

        try:
            seg = pysbd.Segmenter(language=lang, clean=False)
            return [s for s in seg.segment(text) if s and s.strip()]
        except Exception:
            codes = getattr(getattr(pysbd, "languages", None), "LANGUAGE_CODES", None)
            note_degraded(
                "sentences",
                "pysbd-language-unsupported:regex",
                expected=codes is not None and lang not in codes,
            )
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
    split_at_comma: bool,
    comma_split_min_len: int | None,
) -> list[_ClausePlan]:
    """Segment ``text`` into cue clauses, each with its ``word_data`` footprint.

    Clauses are sentences, grouped into comma clauses when ``split_at_comma``;
    none is split for length here (stage 2 owns every length break).
    """
    sentences = _snap_sentence_breaks(text, _segment_sentences(text, lang), lang)
    min_len = (
        default_comma_split_min_len(lang)
        if comma_split_min_len is None
        else comma_split_min_len
    )
    plans: list[_ClausePlan] = []
    for sent in sentences:
        clauses = _comma_clauses(sent, lang, min_len) if split_at_comma else [sent]
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
    *,
    split_at_comma: bool = True,
    comma_split_min_len: int | None = None,
) -> list[Cue]:
    """Stage 1: one cue per sentence or comma clause, timed from its units.

    The line budget plays no part here; ``split_long_cues_with_word_timings``
    breaks the clauses for length.
    """
    plans = _clause_plans(text, lang, split_at_comma, comma_split_min_len)
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
