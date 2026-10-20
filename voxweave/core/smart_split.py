"""Subtitle splitting with gap-aware cue segmentation.

Two stages:
1. ``split_at_sentence_end`` — PySBD (or regex fallback) sentence boundaries,
   each grouped into comma clauses by ``_comma_clauses``. Stage 1 never splits
   for length: stage 2 does all length breaking.
2. ``split_long_cues_with_word_timings`` — word-level greedy packing into
   cues fitting ``max_lines × max_line_length``, with gap/duration breaks
   (a clause without word timings is wrapped by ``_split_without_timings``).

A stage-1 clause boundary is a cue boundary going into stage 2 but not
necessarily in the output: ``_repair_bound_particle_cues``
(zh/yue/ja: merges or repartitions a connected edge that strands a particle),
``_merge_micro_cues`` and ``_glue_short_cues`` (see ``timing``) can fold
adjacent cues across it. Each of those folds gates on a sub-pause gap, so none
crosses a real pause.

This module owns the segmentation *engine*: clause/sentence splitting and the
atom packing loop. Pure text helpers and display wrapping live in ``layout``;
cue-stream timing polish (glue/merge/cleanup/shot-snap) lives in ``timing``.
"""

from __future__ import annotations

import logging
import math
import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

from .gap_split import gap_qualifies
from .kinsoku import (
    line_end_penalty,
    line_start_penalty,
    zh_pos_boundary_penalties,
)
from .langsets import LANGUAGES_WITHOUT_SPACES as LANGUAGES_WITHOUT_SPACES  # re-export
from .layout import (
    WIDE_GLYPH_LANGUAGES,
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
    _cleanup_cues,
    _glue_short_cues,
    _merge_micro_cues,
    _snap_to_shots,
    merge_cues,
)
from .split_atoms import (
    _build_atoms as _build_atoms,
    _cursor_ranges as _cursor_ranges,
    _display_chars as _display_chars,
    _phrase_boundary_atoms as _phrase_boundary_atoms,
    _span_end as _span_end,
    _span_start as _span_start,
    _surface_ranges as _surface_ranges,
    _unit_ranges as _unit_ranges,
    _unit_text as _unit_text,
)
from .split_types import (
    SplitContext as SplitContext,
    SplitThresholds as SplitThresholds,
)

log = logging.getLogger(__name__)

DEFAULT_DESIRED_WPS = 4.0  # reading speed (English wps) behind an untimed estimate

# Comma line-break: split into separate cues at commas, but only when both
# sides are at least this long (visual chars). Shorter clauses stay attached
# to a neighbor so we never strand a tiny fragment on its own cue.
DEFAULT_COMMA_SPLIT_MIN_LEN = 18  # latin / space-delimited
DEFAULT_COMMA_SPLIT_MIN_LEN_CJK = 6  # zh/yue/ja/ko: chars are ~2x visual width

FORCE_BREAK_FACTOR = 1.5  # boundary-less run may exceed the line budget by at most this before a forced cut

# Display floor for a cue built from text the aligner left untimed, used when the
# profile's own min_cue_s is off.
UNTIMED_MIN_DISPLAY_S = 0.5

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
    atom mid-phrase) score 0. They are normally never candidates, but
    ``_best_len_break_pos``'s emergency path (one phrase over the safety or
    duration cap with no internal boundary) makes every atom edge a candidate;
    there such an edge is priced as undamaged, so pause, fragment size and
    balance pick the cut.
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


# Pause-grade ladder shared by the break scorers: a gap shorter than each bound
# (ms) costs the matching damage, so a clearly audible pause is the cheapest cut.
PAUSE_LADDER_MS = (40, 120, 220)
PAUSE_LADDER_DAMAGE = (3, 2, 1)
# Micro-fragment penalties: a side no wider than the bound (cells) costs the
# matching damage.
MICRO_FRAGMENT_WIDTHS = (1, 2)
MICRO_FRAGMENT_DAMAGE = (8, 4)


def pause_damage(left_end: object, right_start: object) -> int:
    """Pause-ladder damage for the gap between two atoms (0 if untimed)."""
    if isinstance(left_end, (int, float)) and isinstance(right_start, (int, float)):
        pause_ms = max(0.0, (float(right_start) - float(left_end)) * 1000)
        if pause_ms < PAUSE_LADDER_MS[0]:
            return PAUSE_LADDER_DAMAGE[0]
        if pause_ms < PAUSE_LADDER_MS[1]:
            return PAUSE_LADDER_DAMAGE[1]
        if pause_ms < PAUSE_LADDER_MS[2]:
            return PAUSE_LADDER_DAMAGE[2]
    return 0


def micro_fragment_damage(left_width: int, right_width: int) -> int:
    """Penalty for a cut that leaves a very narrow side."""
    small = min(left_width, right_width)
    if small <= MICRO_FRAGMENT_WIDTHS[0]:
        return MICRO_FRAGMENT_DAMAGE[0]
    if small <= MICRO_FRAGMENT_WIDTHS[1]:
        return MICRO_FRAGMENT_DAMAGE[1]
    return 0


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
            gap_damage = pause_damage(cur[k - 1].get("end"), right.get("start"))
        right_known = cur[k:] + ([next_atom] if next_atom is not None else [])
        left_width = _vis_width(_join([atom["text"] for atom in cur[:k]], lang))
        right_width = _vis_width(_join([atom["text"] for atom in right_known], lang))
        micro_damage = micro_fragment_damage(left_width, right_width)
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
    gap_break = at_boundary and gap_qualifies(
        prev.get("end"),
        atom.get("start"),
        ctx.speech_spans,
        clause_ms=th.clause_ms,
        vad_skip_ms=th.vad_skip_ms,
        offline_ms=th.offline_ms,
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
        start0 is not None
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
    _split_widest_until(parts, wanted, line_budget)
    return parts


def _split_widest_until(
    parts: list[str], wanted: int, line_budget: int | None = None
) -> None:
    """Halve the widest multi-character part in place until ``wanted`` parts exist.

    With ``line_budget`` the loop also keeps going while any splittable part is
    still wider than that budget.  It stops early once nothing can be divided.
    """
    while len(parts) < wanted or (
        line_budget is not None
        and any(_vis_width(part) > line_budget and len(part) > 1 for part in parts)
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
        math.ceil(duration / ctx.th.max_cue_s) if ctx.th.max_cue_s > 0 else 1
    )
    coarse_timed_atom = duration_parts > 1 and _vis_width(text) > line_budget / 2
    if len(parts) == 1 and not coarse_timed_atom:
        return [(text, start, end)]
    wanted = max(len(parts), duration_parts)
    _split_widest_until(parts, wanted)
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


def _untimed_windows(atoms: list[dict], cue: Cue) -> list[tuple[float, float] | None]:
    """Display window per atom missing a real bound; ``None`` for a timed atom.

    An atom the aligner left (partly) untimed owns only the untimed stretch it
    sits in, never the whole parent span: borrowing the parent span overlaps the
    sibling cues built from the timed atoms around it. Each missing bound is
    interpolated by cumulative visible width between the nearest real bounds on
    either side, the parent cue's own start/end standing in past the first and
    last one. A real bound is kept as is, and a missing end lands exactly on the
    next missing start, so consecutive untimed atoms tile their stretch.
    """
    if all(a.get("start") is not None and a.get("end") is not None for a in atoms):
        return [None] * len(atoms)
    # atom i owns bound 2i (start, at position = width before it) and bound
    # 2i + 1 (end, at position = width through it)
    bounds: list[float | None] = []
    positions: list[int] = []
    width = 0
    for atom in atoms:
        bounds.append(atom.get("start"))
        positions.append(width)
        width += max(1, _vis_width(atom["text"]))
        bounds.append(atom.get("end"))
        positions.append(width)
    before: list[tuple[int, float]] = []
    anchor = (0, float(cue["start"]))
    for position, bound in zip(positions, bounds):
        if bound is not None:
            anchor = (position, bound)
        before.append(anchor)
    after: list[tuple[int, float]] = []
    anchor = (width, float(cue["end"]))
    for position, bound in zip(reversed(positions), reversed(bounds)):
        if bound is not None:
            anchor = (position, bound)
        after.append(anchor)
    after.reverse()
    filled: list[float] = []
    for k, bound in enumerate(bounds):
        if bound is not None:
            filled.append(bound)
            continue
        (lo_pos, lo), (hi_pos, hi) = before[k], after[k]
        hi = max(lo, hi)
        share = (positions[k] - lo_pos) / (hi_pos - lo_pos) if hi_pos > lo_pos else 0
        filled.append(lo + (hi - lo) * share)
    return [
        None
        if atom.get("start") is not None and atom.get("end") is not None
        else (filled[2 * i], max(filled[2 * i], filled[2 * i + 1]))
        for i, atom in enumerate(atoms)
    ]


def _split_oversized_atom(
    atom: dict,
    cue: Cue,
    ctx: SplitContext,
    window: tuple[float, float] | None = None,
) -> list[dict]:
    """Subdivide one structurally coarse atom; ordinary atoms pass through.

    A timed atom's pieces split its real span. An atom missing a bound is laid
    out over ``window`` (see :func:`_untimed_windows`; the parent span when
    absent), but those piece bounds are fabricated: they ride as
    ``display_start``/``display_end`` for :func:`_chunk_to_cue`, while the
    acoustic ``start``/``end`` keep only the bound the atom really had, so no
    made-up time reaches word_data or the speech anchors.
    """
    start = atom.get("start")
    end = atom.get("end")
    if start is not None and end is not None:
        pieces = _surface_parts_for_limits(atom["text"], start, end, ctx)
        if len(pieces) == 1:
            return [atom]
        return [
            {"text": text, "start": piece_start, "end": piece_end, "end_pen": 0}
            for text, piece_start, piece_end in pieces
        ]
    lo, hi = window if window is not None else (cue["start"], cue["end"])
    pieces = _surface_parts_for_limits(atom["text"], lo, hi, ctx)
    if len(pieces) == 1:
        return [atom]
    last = len(pieces) - 1
    return [
        {
            "text": text,
            "start": start if i == 0 else None,
            "end": end if i == last else None,
            "end_pen": 0,
            "display_start": piece_start,
            "display_end": piece_end,
        }
        for i, (text, piece_start, piece_end) in enumerate(pieces)
    ]


def _pack_with_oversized_fallback(
    atoms: list[dict],
    *,
    boundary: set[int] | None,
    ctx: SplitContext,
    cue: Cue,
) -> list[list[dict]]:
    """Pack normal runs and emit token-internal fallback pieces standalone.

    An atom missing a real bound also carries its untimed window as
    ``display_start``/``display_end``, so a chunk with no real time at all
    displays over its own stretch rather than the whole parent span.
    """
    chunks: list[list[dict]] = []
    run: list[dict] = []
    run_indices: list[int] = []
    windows = _untimed_windows(atoms, cue)

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
        window = windows[i]
        pieces = _split_oversized_atom(atom, cue, ctx, window)
        if len(pieces) == 1:
            if window is not None:
                atom = {**atom, "display_start": window[0], "display_end": window[1]}
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
    to the untimed display window its edge atoms carry, else to the parent cue's start/end).

    The emitted word_data is *atom*-level, one entry per packed atom, so each
    entry carries its ``text``: without that surface a later reader
    (``_build_atoms``) has nothing to reconcile against and falls back to a
    character cursor, which is one entry per character.

    The acoustic anchor takes the same raw span but *without* either fallback:
    an untimed chunk has no acoustic evidence, and inheriting the parent's
    (possibly display-padded) bound or an interpolated display window would
    launder a made-up time into the raw layer permanently.
    """
    # the defaults are floats (a display window or the parent cue's required
    # bound), so the span is always a float
    head = chunk[0] if chunk else {}
    tail = chunk[-1] if chunk else {}
    start = cast(float, _span_start(chunk, head.get("display_start", cue["start"])))
    end = cast(float, _span_end(chunk, tail.get("display_end", cue["end"])))
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


def _spread_untimed_cues(cues: list[Cue], ctx: SplitContext) -> list[Cue]:
    """Give every cue built from untimed text the display floor, borrowing
    display time from its sibling cues when its own stretch is too short.

    ``_untimed_windows`` lays untimed text over the stretch between the nearest
    real bounds. That stretch is empty when those bounds touch (timed neighbours
    back to back) or when the text trails past the last real bound of a parent
    that ends right there, and a zero-width cue is never displayed. So a cue with
    a made-up edge (``speech_start`` or ``speech_end`` is ``None``) shorter than
    the floor grows a run of consecutive siblings, one cue on each side per step,
    until the run's spare time covers what its short made-up cues lack. Spare
    time is the gaps inside the run plus whatever every other cue in it displays
    beyond the floor. The short cues then get the floor, and each gap and lender
    gives up the same fraction of its spare time, so the run keeps its order and
    its outer bounds.

    When even the whole parent falls short, its span is shared in proportion to
    need instead: the floor for a short made-up cue, and a lender's own duration
    up to the floor. Every cue with a span to share then stays visible, although
    the short ones get less than the floor.

    Display only: ``start``/``end`` move, while ``speech_start``/``speech_end``
    and ``word_data`` keep exactly what the aligner reported, so a timed sibling
    whose display time was lent out keeps its real anchors. ``cues`` are the
    siblings of one parent, in order, so a run never leaves the parent's span and
    never overlaps a cue outside it. A parent with no span of its own has nothing
    to lend. Mutates and returns ``cues``.
    """
    n = len(cues)
    floor = ctx.th.min_cue_s if ctx.th.min_cue_s > 0 else UNTIMED_MIN_DISPLAY_S
    made_up = [
        c.get("speech_start") is None or c.get("speech_end") is None for c in cues
    ]

    def duration(j: int) -> float:
        return max(0.0, cues[j]["end"] - cues[j]["start"])

    def short(j: int) -> bool:
        return made_up[j] and duration(j) < floor

    k = 0
    while k < n:
        if not short(k):
            k += 1
            continue
        lo = hi = k
        while True:
            run = range(lo, hi + 1)
            durations = [duration(j) for j in run]
            gaps = [max(0.0, cues[j + 1]["start"] - cues[j]["end"]) for j in run[:-1]]
            lacking = [short(j) for j in run]
            deficit = sum(floor - d for d, s in zip(durations, lacking) if s)
            spare = sum(gaps) + sum(
                max(0.0, d - floor) for d, s in zip(durations, lacking) if not s
            )
            if spare >= deficit or (lo == 0 and hi == n - 1):
                break
            lo, hi = max(0, lo - 1), min(n - 1, hi + 1)
        if spare >= deficit:
            take = deficit / spare
            new_durations = [
                floor if s else d - max(0.0, d - floor) * take
                for d, s in zip(durations, lacking)
            ]
            new_gaps = [g - g * take for g in gaps]
        else:
            need = [floor if s else min(d, floor) for d, s in zip(durations, lacking)]
            total_need = sum(need)
            share = (sum(durations) + sum(gaps)) / total_need if total_need else 0.0
            new_durations = [x * share for x in need]
            new_gaps = [0.0] * len(gaps)
        # lay the run out over its own outer bounds; ``fit`` is 1 unless the
        # input already overlapped, which this also resolves
        span_start = cues[lo]["start"]
        span_end = max(span_start, cues[hi]["end"])
        laid = sum(new_durations) + sum(new_gaps)
        fit = (span_end - span_start) / laid if laid > 0 else 0.0
        t = span_start
        for offset, j in enumerate(run):
            end = min(span_end, t + new_durations[offset] * fit)
            cues[j]["start"], cues[j]["end"] = t, end
            if offset < len(new_gaps):
                t = min(span_end, end + new_gaps[offset] * fit)
        k = hi + 1
    return cues


class _BoundPair(NamedTuple):
    """Atoms of two connected cues, with the edge between them located."""

    text: str
    units: list[Unit]
    atoms: list[dict]
    boundaries: set[int]
    original: int


def _analyze_bound_pair(
    left: Cue,
    right: Cue,
    *,
    lang: str,
    max_line_length: int,
    connected_gap_s: float,
) -> _BoundPair | None:
    """Atomize a connected pair, or ``None`` when its edge needs no repair.

    The guards run cheapest first: missing units, an unknown or unconnected
    gap, atoms that do not account for every source unit, an edge that cannot
    be located, and finally an edge that strands no particle or function word.
    """
    left_units = list(left.get("word_data") or [])
    right_units = list(right.get("word_data") or [])
    if not left_units or not right_units:
        return None
    speech_end = _span_end(left_units, left.get("end"))
    speech_start = _span_start(right_units, right.get("start"))
    if speech_end is None or speech_start is None:
        return None
    gap = float(speech_start) - float(speech_end)
    if gap < -1e-6 or gap >= connected_gap_s:
        return None

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
        return None

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
        return None

    boundaries = _phrase_boundary_atoms(atoms, combined_text, lang)
    boundaries.update({0, original})
    _attach_end_penalties(atoms, boundaries, lang)
    original_right = atoms[original]
    # Only repair high-confidence particle/function-word damage.  Ordinary
    # noun or clause boundaries are left to the main/model selector.
    if _atom_end_pen(atoms[original - 1]) < 2 and _atom_start_pen(original_right) < 2:
        return None
    return _BoundPair(combined_text, units, atoms, boundaries, original)


def _try_merge_bound_pair(
    left: Cue,
    right: Cue,
    pair: _BoundPair,
    *,
    lang: str,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
) -> Cue | None:
    """The two cues joined into one when the result fits width and duration."""
    outer_start = float(left["start"])
    outer_end = float(right["end"])
    if outer_end - outer_start <= max_cue_s + 1e-9 and _fits_budget(
        pair.text, max_line_length, max_lines, lang
    ):
        return cast(
            Cue,
            {
                "text": pair.text,
                "start": left["start"],
                "end": right["end"],
                **merge_cues(left, right, lang, text=pair.text),
            },
        )
    return None


def _bound_candidate_score(
    k: int,
    atoms: list[dict],
    original_tail_pen: int,
    lang: str,
) -> tuple[int, int, int, int, int]:
    """Rank a cut before atom ``k``; the smaller tuple is the better edge."""
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
    pause = pause_damage(prev_end, next_start)
    left_surface = _join([atom["text"] for atom in atoms[:k]], lang)
    right_surface = _join([atom["text"] for atom in atoms[k:]], lang)
    micro = micro_fragment_damage(_vis_width(left_surface), _vis_width(right_surface))
    soft = _atom_boundary_pen(edge_right) + pause + micro
    return (
        worsens_tail,
        hard,
        soft,
        abs(_vis_width(left_surface) - _vis_width(right_surface)),
        -k,
    )


def _best_repartition(
    left: Cue,
    right: Cue,
    pair: _BoundPair,
    *,
    lang: str,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
) -> tuple[int, int] | None:
    """``(split_at, unit_cut)`` of the best cut beating the original edge."""
    atoms = pair.atoms
    original = pair.original
    starts = sorted(pair.boundaries - {0, len(atoms)})
    # The tail the inherited edge already leaves: a repartition may relieve a
    # dangling one, but must never hand the left cue a worse tail than this.
    original_tail_pen = _atom_end_pen(atoms[original - 1])

    original_score = _bound_candidate_score(original, atoms, original_tail_pen, lang)
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
        score = _bound_candidate_score(k, atoms, original_tail_pen, lang)
        if score < original_score:
            choices.append((score, k, unit_cut))
    if not choices:
        return None

    _score, split_at, unit_cut = min(choices)
    return split_at, unit_cut


def _repartitioned_cues(
    left: Cue,
    right: Cue,
    pair: _BoundPair,
    split_at: int,
    unit_cut: int,
    lang: str,
) -> list[Cue]:
    """The pair rebuilt as two cues cut before atom ``split_at``."""
    atoms, units = pair.atoms, pair.units
    new_left_atoms, new_right_atoms = atoms[:split_at], atoms[split_at:]
    # Repartitioning moves material between the two cues, so each side's
    # anchor is re-derived from the atoms it now owns -- with no display
    # fallback, unlike the visible bounds beside it.
    return [
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
    it replaces (see ``_bound_candidate_score``).
    """

    if lang not in {"zh", "yue", "ja"} or len(cues) < 2 or connected_gap_s <= 0:
        return cues

    work = [cast(Cue, dict(cue)) for cue in cues]
    i = 0
    while i + 1 < len(work):
        left, right = work[i], work[i + 1]
        pair = _analyze_bound_pair(
            left,
            right,
            lang=lang,
            max_line_length=max_line_length,
            connected_gap_s=connected_gap_s,
        )
        if pair is None:
            i += 1
            continue

        merged = _try_merge_bound_pair(
            left,
            right,
            pair,
            lang=lang,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=max_cue_s,
        )
        if merged is not None:
            work[i : i + 2] = [merged]
            if i:
                i -= 1
            continue

        cut = _best_repartition(
            left,
            right,
            pair,
            lang=lang,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=max_cue_s,
        )
        if cut is not None:
            work[i : i + 2] = _repartitioned_cues(left, right, pair, *cut, lang)
        i += 1
    return work


def _required_thresholds(value: object, caller: str) -> SplitThresholds:
    """The typed thresholds a caller must pass; a missing value is a TypeError."""
    if isinstance(value, SplitThresholds):
        return value
    if isinstance(value, dict):
        return SplitThresholds.from_mapping(value)
    raise TypeError(
        f"{caller}() needs thresholds (a SplitThresholds or a mapping such as "
        f"config.gap_thresholds(lang)), not {type(value).__name__}"
    )


def split_long_cues_with_word_timings(
    cues: list[Cue],
    max_line_length: int,
    max_lines: int,
    lang: str,
    *,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: SplitThresholds,
) -> list[Cue]:
    """Pack each cue's atoms into reading-sized cues using gap/duration/length breaks.

    ``thresholds`` is required (``SplitThresholds()`` gives the defaults). The
    arguments after ``lang`` are keyword-only, so a caller still passing the
    removed ``min_duration``/``desired_wps`` positionally gets a ``TypeError``
    instead of having them bound to other parameters.
    """
    ctx = SplitContext(
        lang=lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        th=_required_thresholds(thresholds, "split_long_cues_with_word_timings"),
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
            if _no_spaces(lang)
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
        new_cues.extend(
            _spread_untimed_cues(
                [_chunk_to_cue(chunk, cue, lang) for chunk in chunks], ctx
            )
        )
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
    *,
    split_at_comma: bool = True,
    comma_split_min_len: int | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: SplitThresholds | dict,
    shot_changes: list[float] | None = None,
) -> list[Cue]:
    """Run the full smart-split pipeline over aligned segments.

    Each segment must have ``text`` and ``words`` (with ``start``/``end``).
    Returns a flat list of cues with ``text``, ``start``, ``end``, ``word_data``.
    Everything after ``max_lines`` is keyword-only, so a caller still passing the
    removed ``min_duration``/``desired_wps`` positionally gets a ``TypeError``
    instead of having them bound to ``split_at_comma``/``comma_split_min_len``.

    ``thresholds`` is required: a :class:`SplitThresholds`, or a mapping such as
    ``config.gap_thresholds(lang)`` (partial mappings fill the dataclass
    defaults). Omitting it, or passing ``None``, raises ``TypeError``.

    ``split_at_comma`` (default on) breaks at commas unless either side is
    shorter than ``comma_split_min_len`` visual chars. The sentence/comma
    clauses are packed by the timed atom stage and then the cue-stream folds
    run: ``_repair_bound_particle_cues``, ``_merge_micro_cues`` and
    ``_glue_short_cues`` can each merge adjacent cues across a clause boundary
    when the gap between them is below a real pause.
    """
    th = _required_thresholds(thresholds, "smart_split_segments")
    if max_line_length is None:
        max_line_length = default_max_line_length(lang)
    if max_lines is None:
        max_lines = default_max_lines(lang)  # zh/yue/ja -> 1, else 2
    # The packer budgets in native cells (a CJK preset counts one wide glyph per
    # cell) while the renderer measures half-width cells. Convert once so both
    # stages honour the same configured profile instead of the renderer silently
    # falling back to its built-in 42-column default.
    render_width = _line_budget_width(max_line_length, lang)
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
                split_at_comma=split_at_comma,
                comma_split_min_len=comma_split_min_len,
            )
        )
    cues = split_long_cues_with_word_timings(
        all_cues,
        max_line_length=max_line_length,
        max_lines=max_lines,
        lang=lang,
        speech_spans=speech_spans,
        thresholds=th,
    )
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
        cue["text"] = _merge_stutters(strip_punct_for_subtitles(cue["text"]))
        # Display soft-wrap: fold over-budget cues into <=max_lines lines without
        # changing cue boundaries. Long Latin phrases inside CJK also collapse here.
        cue["text"] = wrap_cue_text(
            cue["text"], lang, max_lines, max_line_length=render_width
        )
    return cues
