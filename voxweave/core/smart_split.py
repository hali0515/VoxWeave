"""Semantic subtitle splitting with gap-aware cue segmentation.

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
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from .breakpoints import legal_break_index, phrase_atoms
from .conjunctions import conjunctions_by_language, get_comma
from .gap_split import gap_qualifies
from .kinsoku import (
    line_end_penalty,
    line_start_penalty,
    zh_pos_boundary_penalties,
)
from .langsets import LANGUAGES_WITHOUT_SPACES as LANGUAGES_WITHOUT_SPACES  # re-export
from .schema import Cue, Unit
from .layout import (
    WIDE_GLYPH_LANGUAGES,
    _comma_chars,
    _fits_budget,
    _join,
    _line_budget_width,
    _merge_stutters,
    _no_spaces,
    _reading_chars,
    _strip_trailing_commas,
    _token_char_count,
    _tokens,
    _visual_len,
    _vis_width,
    default_max_line_length,
    default_max_lines,
    split_subtitle,
    strip_punct_for_subtitles,
    wrap_cue_text,
)
from .timing import (
    GLUE_MAX_GAP_S,
    _cleanup_cues,
    _glue_short_cues,
    _merge_micro_cues,
    _snap_to_shots,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from voxweave.semantic_breaks import SemanticBreakEngine

DEFAULT_MIN_DURATION = 3.0  # reading-speed pad for single cues
DEFAULT_DESIRED_WPS = 4.0  # target reading speed (English wps)

# Comma line-break: split into separate cues at commas, but only when both
# sides are at least this long (visual chars). Shorter clauses stay attached
# to a neighbor so we never strand a tiny fragment on its own cue.
DEFAULT_COMMA_SPLIT_MIN_LEN = 18  # latin / space-delimited
DEFAULT_COMMA_SPLIT_MIN_LEN_CJK = 6  # zh/yue/ja/ko: chars are ~2x visual width

FORCE_BREAK_FACTOR = 1.5  # boundary-less run may exceed the line budget by at most this before a forced cut

# Semantic selection is deliberately local.  Natural sentence/confirmed-pause
# barriers normally produce much smaller windows; this cap bounds malformed or
# punctuation-free transcripts so legal-edge construction remains linear in
# transcript length with a small, fixed per-window constant.
SEMANTIC_WINDOW_MAX_ATOMS = 96
SEMANTIC_QUALITY_AVG_TOLERANCE = 15.0
SEMANTIC_QUALITY_WORST_TOLERANCE = 25
SEMANTIC_PREFERRED_MIN_CUE_S = 1.0


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


def _split_keep_comma(sentence: str, lang: str) -> List[str]:
    """Split a sentence after each comma (comma stays on the left part).
    Commas between digits (e.g. 10,000) are NOT split points. For spaced
    languages the comma must also end its token (next char is whitespace):
    a mid-token comma (e.g. ``so,"``) would divide the token and desync the
    token-to-word_data index zip in ``split_at_sentence_end``."""
    commas = _comma_chars(lang)
    no_spaces = _no_spaces(lang)
    out: List[str] = []
    buf: List[str] = []
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


def _comma_clauses(sentence: str, lang: str, min_len: int) -> List[str]:
    """Group comma-delimited pieces into cue clauses.

    A clause flushes once it reaches ``min_len`` visual chars. The comma-load
    cap (<=1) prevents piling repeated short fragments (e.g. a name said several
    times) onto one line. Trailing commas are kept for downstream stripping."""
    pieces = _split_keep_comma(sentence, lang)
    clauses: List[str] = []
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


def _build_atoms(
    text: str,
    word_data: list[Unit],
    lang: str,
    max_atom_width: int | None = None,
) -> list[dict]:
    """Build non-breakable atoms, each with aggregated start/end from word_data.

    Space-delimited: one word per atom (1:1 with word_data). No-space: one atom
    per CJK char or Latin run (from ``_tokens``). BudouX phrase boundaries are
    applied later in the packing loop — atoms stay per-char so gap/duration
    breaks have full granularity. word_data is char-level; each atom consumes
    ``_token_char_count(unit)`` entries.
    """
    if not _no_spaces(lang):
        toks = text.split()
        wd = list(word_data)
        if len(wd) < len(toks):
            wd += [{}] * (len(toks) - len(wd))
        return [
            {"text": t, "start": w.get("start"), "end": w.get("end")}
            for t, w in zip(toks, wd[: len(toks)])
        ]
    units = _tokens(text, lang)
    atoms: list[dict] = []
    cursor = 0
    for unit in units:
        n = _token_char_count(unit)
        chunk = word_data[cursor : cursor + n]
        cursor += n
        # Short embedded Latin phrases stay atomic, but a phrase wider than a
        # physical line must expose its whitespace boundaries to the packer.
        # Keep trailing spaces on each sub-atom so no-space joining reconstructs
        # the original surface, and retain real per-character timing.
        if (
            max_atom_width is not None
            and _vis_width(unit) > max_atom_width
            and any(ch.isspace() for ch in unit)
        ):
            local_cursor = 0
            for part_i, match in enumerate(re.finditer(r"\S+\s*", unit)):
                surface = match.group(0)
                part_n = _token_char_count(surface)
                part_chunk = chunk[local_cursor : local_cursor + part_n]
                local_cursor += part_n
                atoms.append(
                    {
                        "text": surface,
                        "start": _span_start(part_chunk),
                        "end": _span_end(part_chunk),
                        "forced_boundary": part_i > 0,
                    }
                )
            continue
        atoms.append(
            {"text": unit, "start": _span_start(chunk), "end": _span_end(chunk)}
        )
    return atoms


def _phrase_boundary_atoms(atoms: List[dict], text: str, lang: str) -> set[int]:
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
    toks: List[str], text: str, lang: str, target: int
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


@functools.lru_cache(
    maxsize=None
)  # one pattern per language; avoids recompiling per clause
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
    comma_split_min_len: Optional[int] = None,
) -> List[str]:
    if split_at_comma:
        if comma_split_min_len is None:
            comma_split_min_len = default_comma_split_min_len(lang)
        clauses = _comma_clauses(sentence, lang, comma_split_min_len)
    else:
        clauses = [sentence]
    out: List[str] = []
    for clause in clauses:
        out.extend(_fit_split_clause(clause, max_line_length, max_lines, lang))
    return [p for p in out if p]


def _repack_parts(
    parts: List[str], max_line_length: int, max_lines: int, lang: str
) -> List[str]:
    """Greedily merge adjacent terminal/conjunction parts back up to the budget.

    The split pattern marks *candidate* break points, not mandates: keeping every
    part separate shatters a long sentence into fragment cues ("and bought milk" |
    "and eggs"). Mirrors the accumulate-then-flush behavior of _comma_clauses.
    """
    sep = "" if _no_spaces(lang) else " "
    packed: List[str] = []
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


def _visual_midpoint_index(tokens: List[str], lang: str) -> int:
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
) -> List[str]:
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
) -> List[str]:
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
    fitted_parts: List[str] = []
    for part in candidate_parts:
        fitted_parts.extend(
            _split_part_to_budget(part, max_line_length, max_lines, lang)
        )
    return _repack_parts(fitted_parts, max_line_length, max_lines, lang)


def _segment_sentences(text: str, lang: str) -> List[str]:
    try:
        import pysbd  # type: ignore

        try:
            seg = pysbd.Segmenter(language=lang, clean=False)
            return [s for s in seg.segment(text) if s and s.strip()]
        except Exception:
            pass
    except ImportError:
        pass
    return [s for s in re.split(r"(?<=[.!?。！？])\s*", text) if s and s.strip()]


def _snap_sentence_breaks(text: str, sentences: List[str], lang: str) -> List[str]:
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
    cuts: List[int] = []
    pos = 0
    for sent in sentences[:-1]:
        idx = text.find(sent, pos)
        if idx < 0:
            return sentences  # segmenter rewrote content; nothing safe to snap
        pos = idx + len(sent)
        cuts.append(pos)
    pieces: List[str] = []
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


def _anchor_cursor(
    word_data: List[Unit],
    cursor: int,
    clause_tokens: List[str],
    max_shift: int = 8,
) -> Tuple[int, bool]:
    """Verify the clause's tokens match ``word_data`` at ``cursor``; search nearby on mismatch.

    Returns ``(start_index, ok)``. The index contract can still break on inputs
    we have not anticipated (ghost or lost units); rather than silently shifting
    every later cue, re-anchor on content within ``max_shift`` units, else keep
    the cursor and report ``ok=False``.
    """

    def _matches(at: int) -> bool:
        if at < 0 or at + len(clause_tokens) > len(word_data):
            return False
        # reinject can glue a boundary space onto a unit ('开 ' at CJK<->Latin
        # seams); the cursor arithmetic is whitespace-insensitive, so compare
        # stripped content.
        return all(
            (word_data[at + j].get("word") or "").strip() == tok.strip()
            for j, tok in enumerate(clause_tokens)
        )

    if _matches(cursor):
        return cursor, True
    for d in range(1, max_shift + 1):
        if _matches(cursor + d):
            return cursor + d, True
        if _matches(cursor - d):
            return cursor - d, True
    return cursor, False


def split_at_sentence_end(
    text: str,
    word_data: List[Unit],
    lang: str,
    max_line_length: int,
    max_lines: int,
    split_at_comma: bool = True,
    comma_split_min_len: Optional[int] = None,
    *,
    defer_length_split: bool = False,
) -> List[Cue]:
    sentences = _snap_sentence_breaks(text, _segment_sentences(text, lang), lang)
    cues: List[Cue] = []
    cursor = 0
    # Content verification needs unit texts; legacy callers without a "word"
    # key keep the blind index cursor.
    anchored = bool(word_data) and "word" in word_data[0]
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
                wc = sum(_token_char_count(t) for t in clause_tokens)
                anchor_tokens = [c for c in clause if not c.isspace()]
            else:
                wc = len(clause_tokens)
                anchor_tokens = clause_tokens
            start_at = cursor
            if anchored and anchor_tokens:
                start_at, ok = _anchor_cursor(word_data, cursor, anchor_tokens)
                if start_at != cursor or not ok:
                    log.warning(
                        "cue/word desync at %r: cursor %d -> %d (%s)",
                        clause[:40],
                        cursor,
                        start_at,
                        "resynced" if ok else "unrecovered",
                    )
            chunk_words = word_data[start_at : start_at + wc]
            cursor = start_at + wc
            if chunk_words:
                start = next((w["start"] for w in chunk_words if "start" in w), None)
                end = next(
                    (w["end"] for w in reversed(chunk_words) if "end" in w), None
                )
            else:
                start = end = None
            if start is None or end is None:
                # No timing data: extend from previous cue end or estimate from word count
                prev_end = cues[-1]["end"] if cues else 0.0
                start = start if start is not None else prev_end
                end = (
                    end
                    if end is not None
                    else start + max(1.0, wc / DEFAULT_DESIRED_WPS)
                )
            cues.append(
                {
                    "text": clause,
                    "start": start,
                    "end": end,
                    "word_data": chunk_words,
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
    atoms: List[dict], boundary: set[int] | None, lang: str
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
    cur: List[dict],
    cur_bnd: List[bool],
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
    cur: List[dict],
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


def _hard_wrap_surface(text: str, line_budget: int) -> List[str]:
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
) -> List[tuple[str, float, float]]:
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


def _split_oversized_atom(atom: dict, cue: Cue, ctx: SplitContext) -> List[dict]:
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
    atoms: List[dict],
    *,
    boundary: set[int] | None,
    ctx: SplitContext,
    cue: Cue,
) -> List[List[dict]]:
    """Pack normal runs and emit token-internal fallback pieces standalone."""
    chunks: List[List[dict]] = []
    run: List[dict] = []
    run_indices: List[int] = []

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
    atoms: List[dict],
    *,
    boundary: set[int] | None,
    ctx: SplitContext,
) -> List[List[dict]]:
    """Greedily pack atoms into chunks, cutting on the first qualifying gap/dur/len break.

    Time-forced breaks (gap/dur) cut immediately; a length overflow picks the phrase-boundary
    candidate with the smallest sticky-token penalty via ``_best_len_break_pos`` (Level 1).
    """
    chunks: List[List[dict]] = []
    cur: List[dict] = []
    cur_bnd: List[
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


def _chunk_to_cue(chunk: List[dict], cue: Cue, lang: str) -> Cue:
    """Materialize a packed atom chunk into a cue dict (first/last non-None span, falling back
    to the parent cue's start/end)."""
    # the default is the parent cue's (required, non-None) bound, so the span is always a float
    start = cast(float, _span_start(chunk, cue["start"]))
    end = cast(float, _span_end(chunk, cue["end"]))
    return {
        "text": _join([a["text"] for a in chunk], lang),
        "start": start,
        "end": end,
        "word_data": [{"start": a["start"], "end": a["end"]} for a in chunk],
    }


def _repair_bound_particle_cues(
    cues: List[Cue],
    *,
    lang: str,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
    connected_gap_s: float,
) -> List[Cue]:
    """Remove connected cue edges that strand an independently tagged particle.

    This is a final safety net for boundaries inherited from separate ASR
    segments or an earlier hard layout decision.  A direct merge is preferred;
    if it would exceed width/duration, the two cues are repartitioned at a
    better phrase/POS boundary while retaining every source unit and timestamp.
    True pauses are never crossed.  Whole-token/POS scoring distinguishes the
    particles ``了/地`` from lexical words such as ``了解/地方``.
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
        expected_units = sum(_token_char_count(atom["text"]) for atom in atoms)
        if not atoms or expected_units != len(units):
            i += 1
            continue

        left_width = _token_char_count(left_text)
        cursor = 0
        original = None
        for atom_index, atom in enumerate(atoms):
            if cursor == left_width:
                original = atom_index
                break
            cursor += _token_char_count(atom["text"])
        if original is None and cursor == left_width:
            original = len(atoms)
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
            work[i : i + 2] = [
                {
                    "text": combined_text,
                    "start": left["start"],
                    "end": right["end"],
                    "word_data": units,
                }
            ]
            if i:
                i -= 1
            continue

        starts = sorted(boundaries - {0, len(atoms)})

        def candidate_score(k: int) -> tuple[int, int, int, int]:
            edge_right = atoms[k]
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
                hard,
                soft,
                abs(_vis_width(left_surface) - _vis_width(right_surface)),
                -k,
            )

        original_score = candidate_score(original)
        choices: list[tuple[tuple[int, int, int, int], int, int]] = []
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
            unit_cut = sum(_token_char_count(atom["text"]) for atom in left_chunk)
            score = candidate_score(k)
            if score < original_score:
                choices.append((score, k, unit_cut))
        if not choices:
            i += 1
            continue

        _score, split_at, unit_cut = min(choices)
        new_left_atoms, new_right_atoms = atoms[:split_at], atoms[split_at:]
        work[i : i + 2] = [
            {
                "text": _join([atom["text"] for atom in new_left_atoms], lang),
                "start": left["start"],
                "end": cast(float, _span_end(new_left_atoms, left.get("end"))),
                "word_data": units[:unit_cut],
            },
            {
                "text": _join([atom["text"] for atom in new_right_atoms], lang),
                "start": cast(float, _span_start(new_right_atoms, right.get("start"))),
                "end": right["end"],
                "word_data": units[unit_cut:],
            },
        ]
        i += 1
    return work


def split_long_cues_with_word_timings(
    cues: List[Cue],
    max_line_length: int,
    max_lines: int,
    min_duration: float,
    desired_wps: float,
    lang: str,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: Optional[SplitThresholds] = None,
) -> List[Cue]:
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
    new_cues: List[Cue] = []
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
) -> List[Cue]:
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
                }
                for text, start, end in pieces
            ]
    formatted = split_subtitle(cue["text"], max_line_length, lang)
    lines = formatted.split("\n")
    chunks: List[List[str]] = []
    buf: List[str] = []
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
    out: List[Cue] = []
    for c in chunks:
        # Join without \n — the downstream SubtitlesWriter handles display wrapping.
        text = sep.join(c)
        proportion = len(text) / total_chars
        end = start + duration * proportion if duration > 0 else start
        out.append({"text": text, "start": start, "end": end, "word_data": []})
        start = end
    return out


@dataclass
class _SemanticSplitPlan:
    """One raw aligned segment prepared for optional boundary selection.

    ``allowed_edges`` is the host-owned hard-legality graph.  Nodes are atom offsets
    0..N and an edge ``(i, j)`` means atoms ``i:j`` can be displayed as one cue
    without crossing a hard sentence/pause boundary.  The model only chooses a
    path through this graph; it never owns text or timestamps.
    """

    text: str
    atoms: list[dict]
    word_data: list[Unit]
    fallback_indices: tuple[int, ...]
    required_indices: tuple[int, ...]
    allowed_edges: frozenset[tuple[int, int]]
    edge_quality: dict[tuple[int, int], int]
    hard_after: bool
    task: Any | None


def _semantic_sentence_boundaries(text: str, atoms: list[dict], lang: str) -> set[int]:
    """Map real sentence-segmenter cuts onto atom indices without rewriting text."""

    sentences = _snap_sentence_breaks(text, _segment_sentences(text, lang), lang)
    if len(sentences) < 2:
        return set()
    atom_ends: dict[int, int] = {}
    units = 0
    for i, atom in enumerate(atoms, 1):
        units += 1 if not _no_spaces(lang) else _token_char_count(atom["text"])
        atom_ends[units] = i
    wanted: set[int] = set()
    units = 0
    for sentence in sentences[:-1]:
        tokens = _tokens(sentence, lang)
        units += (
            len(tokens)
            if not _no_spaces(lang)
            else sum(_token_char_count(token) for token in tokens)
        )
        index = atom_ends.get(units)
        if index is not None and 0 < index < len(atoms):
            wanted.add(index)
    return wanted


def _semantic_annotate_unit_ranges(
    atoms: list[dict], word_data: list[Unit], lang: str
) -> bool:
    """Attach exact source-unit slices to atoms; reject ambiguous alignments.

    The semantic path is intentionally stricter than the deterministic path.
    If atomisation cannot account for every input unit exactly, timing-aware
    selection is unsafe and the caller returns the already-computed baseline.
    """

    cursor = 0
    previous_start = -math.inf
    previous_end = -math.inf
    for atom in atoms:
        width = 1 if not _no_spaces(lang) else _token_char_count(atom["text"])
        end_cursor = cursor + width
        if width < 1 or end_cursor > len(word_data):
            return False
        atom["_unit_start"] = cursor
        atom["_unit_end"] = end_cursor
        start = atom.get("start")
        end = atom.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or end < start
            or start < previous_start
            or end < previous_end
        ):
            return False
        previous_start = float(start)
        previous_end = float(end)
        cursor = end_cursor
    return cursor == len(word_data)


def _semantic_hard_boundaries(
    text: str,
    atoms: list[dict],
    ctx: SplitContext,
) -> tuple[set[int], dict[int, int]]:
    """Return mandatory sentence/pause cuts and pause evidence for the model."""

    required = _semantic_sentence_boundaries(text, atoms, ctx.lang)
    pauses: dict[int, int] = {}
    for index in range(1, len(atoms)):
        prev_end = atoms[index - 1].get("end")
        next_start = atoms[index].get("start")
        gap = _gap_ms(prev_end, next_start)
        if gap is None:
            continue
        pauses[index] = max(0, round(gap))
        if gap_qualifies(
            prev_end,
            next_start,
            ctx.speech_spans,
            clause_ms=ctx.th.clause_ms,
            vad_skip_ms=ctx.th.vad_skip_ms,
            offline_ms=ctx.th.offline_ms,
        ):
            required.add(index)
    return required, pauses


def _semantic_edge_hard_legal(
    atoms: list[dict],
    start: int,
    end: int,
    *,
    required: set[int],
    ctx: SplitContext,
) -> bool:
    """Hard text/timeline constraints shared by strict and fallback edges."""

    if not 0 <= start < end <= len(atoms):
        return False
    if any(start < boundary < end for boundary in required):
        return False
    chunk = atoms[start:end]
    text = strip_punct_for_subtitles(_join([atom["text"] for atom in chunk], ctx.lang))
    if not _fits_budget(text, ctx.max_line_length, ctx.max_lines, ctx.lang):
        return False
    cue_start = _span_start(chunk)
    cue_end = _span_end(chunk)
    if cue_start is None or cue_end is None or cue_end < cue_start:
        return False
    return not (ctx.th.max_cue_s > 0 and cue_end - cue_start > ctx.th.max_cue_s + 1e-9)


def _semantic_available_duration(
    atoms: list[dict],
    start: int,
    end: int,
    *,
    ctx: SplitContext,
) -> float:
    """Maximum display span the existing cleanup pass can realistically offer."""

    chunk = atoms[start:end]
    cue_start = cast(float, _span_start(chunk))
    cue_end = cast(float, _span_end(chunk))
    spoken = cue_end - cue_start
    next_start = (
        atoms[end].get("start") if end < len(atoms) else atoms[-1].get("_next_start")
    )
    extension = max(
        1.0 if ctx.th.cps > 0 else 0.0,
        ctx.th.lag_out_s,
        max(0.0, ctx.th.min_cue_s - spoken),
    )
    latest_end = cue_end + extension
    if isinstance(next_start, (int, float)) and not isinstance(next_start, bool):
        latest_end = min(latest_end, float(next_start))
    possible = max(spoken, latest_end - cue_start)
    if ctx.th.max_cue_s > 0:
        possible = min(possible, ctx.th.max_cue_s)
    return max(0.0, possible)


def _semantic_edge_quality(
    atoms: list[dict],
    start: int,
    end: int,
    *,
    ctx: SplitContext,
    desired_wps: float,
) -> int:
    """Soft 0..100 timing/readability score for one otherwise legal cue.

    Fast source speech can make the configured min/CPS/WPS targets impossible
    for *every* segmentation.  Those targets must therefore rank legal paths,
    never delete them.  A score of 100 means cleanup has enough real timeline
    room for all applicable targets; lower values quantify the shortfall.
    """

    possible = _semantic_available_duration(atoms, start, end, ctx=ctx)
    text = strip_punct_for_subtitles(
        _join([atom["text"] for atom in atoms[start:end]], ctx.lang)
    )
    components: list[tuple[float, float]] = []
    if ctx.th.min_cue_s > 0:
        # The ordinary splitter's configured minimum remains a hard feasibility
        # floor.  Semantic reflow uses a more comfortable one-second soft target
        # so a punctuation-perfect but flickering cue does not beat a stable
        # time-balanced path.  Intrinsically fast speech remains legal and can
        # still fall back unchanged.
        preferred_min = max(ctx.th.min_cue_s, SEMANTIC_PREFERRED_MIN_CUE_S)
        if ctx.th.max_cue_s > 0:
            preferred_min = min(preferred_min, ctx.th.max_cue_s)
        components.append((0.30, min(1.0, possible / preferred_min)))
    if ctx.th.cps > 0:
        need = _reading_chars(text) / ctx.th.cps
        components.append((0.45, 1.0 if need <= 0 else min(1.0, possible / need)))
    if not _no_spaces(ctx.lang) and desired_wps > 0:
        need = len(atoms[start:end]) / desired_wps
        components.append((0.25, 1.0 if need <= 0 else min(1.0, possible / need)))
    if not components:
        return 100
    total_weight = sum(weight for weight, _ratio in components)
    return max(
        0,
        min(
            100,
            round(
                100 * sum(weight * ratio for weight, ratio in components) / total_weight
            ),
        ),
    )


def _semantic_path_quality(
    plan: _SemanticSplitPlan, breaks: tuple[int, ...]
) -> tuple[float, int] | None:
    """Atom-weighted mean and worst edge quality for a complete path."""

    nodes = (0, *breaks, len(plan.atoms))
    edges = tuple(zip(nodes, nodes[1:]))
    qualities: list[tuple[int, int]] = []
    for start, end in edges:
        quality = plan.edge_quality.get((start, end))
        if quality is None:
            return None
        qualities.append((quality, end - start))
    if not qualities:
        return None
    weight = sum(span for _quality, span in qualities)
    average = sum(quality * span for quality, span in qualities) / max(1, weight)
    return average, min(quality for quality, _span in qualities)


def _semantic_quality_not_significantly_worse(
    plan: _SemanticSplitPlan, breaks: tuple[int, ...]
) -> bool:
    """Allow semantic trade-offs but reject a materially worse timing path."""

    chosen = _semantic_path_quality(plan, breaks)
    fallback = _semantic_path_quality(plan, plan.fallback_indices)
    if chosen is None or fallback is None:
        return False
    chosen_avg, chosen_worst = chosen
    fallback_avg, fallback_worst = fallback
    return (
        chosen_avg + SEMANTIC_QUALITY_AVG_TOLERANCE >= fallback_avg
        and chosen_worst + SEMANTIC_QUALITY_WORST_TOLERANCE >= fallback_worst
    )


def _semantic_edges_on_complete_paths(
    edges: set[tuple[int, int]], atom_count: int
) -> set[tuple[int, int]]:
    """Prune a DAG to edges lying on at least one complete 0→N path."""

    forward = {0}
    for start, end in sorted(edges, key=lambda edge: (edge[1], edge[0])):
        if start in forward:
            forward.add(end)
    backward = {atom_count}
    for start, end in sorted(edges, reverse=True):
        if end in backward:
            backward.add(start)
    return {
        (start, end) for start, end in edges if start in forward and end in backward
    }


def _semantic_shortest_path(
    edges: set[tuple[int, int]], atom_count: int
) -> tuple[int, ...] | None:
    """Return a deterministic minimum-cue path through a forward edge graph."""

    outgoing: dict[int, list[int]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
    best: dict[int, tuple[int, ...]] = {atom_count: ()}
    for start in range(atom_count - 1, -1, -1):
        choices: list[tuple[int, ...]] = []
        for end in sorted(outgoing.get(start, ()), reverse=True):
            tail = best.get(end)
            if tail is None:
                continue
            choices.append((end, *tail) if end < atom_count else ())
        if choices:
            best[start] = min(choices, key=lambda path: (len(path), path))
    return best.get(0)


def _semantic_fallback_indices(
    atoms: list[dict],
    *,
    phrase_boundary: set[int] | None,
    required: set[int],
    ctx: SplitContext,
) -> tuple[int, ...]:
    """Mechanical atom-packer path used only as the model engine's fallback."""

    chunks = _pack_atoms_into_chunks(atoms, boundary=phrase_boundary, ctx=ctx)
    cursor = 0
    cuts: set[int] = set(required)
    for chunk in chunks[:-1]:
        cursor += len(chunk)
        cuts.add(cursor)
    return tuple(sorted(index for index in cuts if 0 < index < len(atoms)))


def _semantic_window_ranges(
    atom_count: int,
    *,
    mandatory: set[int],
    phrase_boundary: set[int] | None,
    max_atoms: int = SEMANTIC_WINDOW_MAX_ATOMS,
) -> list[tuple[int, int, bool]]:
    """Split a transcript at hard barriers and cap punctuation-free windows.

    The boolean marks a real sentence/pause barrier after the window.  Safety
    cuts bound work but are not timing barriers, so cleanup may still glue a
    tiny fragment across them when doing so is display-safe.
    """

    if max_atoms < 2:
        raise ValueError("semantic window atom cap must be at least two")
    ranges: list[tuple[int, int, bool]] = []
    phrase_points = sorted(phrase_boundary or ())
    start = 0
    for hard_end in (*sorted(mandatory), atom_count):
        if not start < hard_end <= atom_count:
            continue
        while hard_end - start > max_atoms:
            limit = start + max_atoms
            if phrase_boundary is None:
                cut = limit
            else:
                point_index = bisect_right(phrase_points, limit) - 1
                point = phrase_points[point_index] if point_index >= 0 else start
                cut = point if point > start else limit
            if cut <= start:
                cut = min(limit, hard_end)
            ranges.append((start, cut, False))
            start = cut
        ranges.append((start, hard_end, hard_end in mandatory))
        start = hard_end
    return ranges


def _semantic_hard_edges(
    atoms: list[dict], nodes: list[int], *, ctx: SplitContext
) -> set[tuple[int, int]]:
    """Build hard-legal edges with bounded forward scans.

    Windows contain no internal mandatory barriers.  Visual load and aligned
    duration are monotone as ``end`` advances, so the first overflow terminates
    that start node's scan instead of examining every later span.
    """

    edges: set[tuple[int, int]] = set()
    for pos, start in enumerate(nodes[:-1]):
        for end in nodes[pos + 1 :]:
            chunk = atoms[start:end]
            cue_start = _span_start(chunk)
            cue_end = _span_end(chunk)
            if cue_start is None or cue_end is None or cue_end < cue_start:
                break
            if ctx.th.max_cue_s > 0 and cue_end - cue_start > ctx.th.max_cue_s + 1e-9:
                break
            # Width must match the text users will actually see.  The normal
            # writer removes prose punctuation after splitting, so counting it
            # here invents overflow and can force a one-character punctuation
            # cue or split an otherwise exact-width semantic phrase.
            text = strip_punct_for_subtitles(
                _join([atom["text"] for atom in chunk], ctx.lang)
            )
            if not _fits_budget(text, ctx.max_line_length, ctx.max_lines, ctx.lang):
                break
            edges.add((start, end))
    return _semantic_edges_on_complete_paths(edges, len(atoms))


def _prepare_semantic_window(
    *,
    text: str,
    atoms: list[dict],
    word_data: list[Unit],
    phrase_boundary: set[int] | None,
    pauses: dict[int, int],
    hard_after: bool,
    ctx: SplitContext,
    desired_wps: float,
    fallback_indices: tuple[int, ...] | None = None,
) -> _SemanticSplitPlan | None:
    """Create one bounded immutable task with hard edges and soft timing scores."""

    from voxweave.semantic_breaks import BoundaryTask

    _attach_end_penalties(atoms, phrase_boundary, ctx.lang)
    exact_mechanical_fallback = fallback_indices is not None
    mechanical_fallback = (
        tuple(sorted(set(fallback_indices)))
        if fallback_indices is not None
        else _semantic_fallback_indices(
            atoms,
            phrase_boundary=phrase_boundary,
            required=set(),
            ctx=ctx,
        )
    )
    if phrase_boundary is None:
        candidates = set(range(1, len(atoms)))
    else:
        candidates = set(phrase_boundary) - {0, len(atoms)}
    candidates.update(mechanical_fallback)
    nodes = [0, *sorted(candidates), len(atoms)]
    allowed_edges = _semantic_hard_edges(atoms, nodes, ctx=ctx)
    if not allowed_edges:
        return None
    mechanical_nodes = (0, *mechanical_fallback, len(atoms))
    mechanical_edges = set(zip(mechanical_nodes, mechanical_nodes[1:]))
    if mechanical_edges <= allowed_edges:
        fallback = mechanical_fallback
    else:
        if exact_mechanical_fallback:
            # The caller supplied cuts from the already-computed deterministic
            # subtitles.  Never silently replace that reference with a second
            # greedy path: aborting this optional pass returns the exact baseline.
            return None
        fallback = _semantic_shortest_path(allowed_edges, len(atoms))
        if fallback is None:
            return None
    edge_quality = {
        edge: _semantic_edge_quality(
            atoms, edge[0], edge[1], ctx=ctx, desired_wps=desired_wps
        )
        for edge in allowed_edges
    }
    candidates.update(
        node for edge in allowed_edges for node in edge if 0 < node < len(atoms)
    )
    task = BoundaryTask(
        atoms=tuple(atom["text"] for atom in atoms),
        candidate_indices=tuple(sorted(candidates)),
        language=ctx.lang,
        fallback_indices=fallback,
        pauses_ms=tuple(
            sorted(
                (index, duration)
                for index, duration in pauses.items()
                if index in candidates
            )
        ),
        target_chars=ctx.max_line_length * ctx.max_lines,
        allowed_edges=tuple(sorted(allowed_edges)),
        edge_quality=tuple(
            (start, end, quality)
            for (start, end), quality in sorted(edge_quality.items())
        ),
    )
    return _SemanticSplitPlan(
        text=text,
        atoms=atoms,
        word_data=word_data,
        fallback_indices=fallback,
        required_indices=(),
        allowed_edges=frozenset(allowed_edges),
        edge_quality=edge_quality,
        hard_after=hard_after,
        task=task if candidates else None,
    )


def _prepare_semantic_plans(
    segment: Mapping[str, Any],
    *,
    ctx: SplitContext,
    desired_wps: float,
    fallback_unit_boundaries: set[int] | None = None,
) -> list[_SemanticSplitPlan] | None:
    """Atomise one pipeline segment, then create bounded barrier-separated tasks."""

    text = str(segment.get("text", ""))
    word_data = list(segment.get("words", []) or [])
    if not text or not word_data:
        return None
    cue_start = _span_start(word_data, segment.get("start"))
    cue_end = _span_end(word_data, segment.get("end"))
    if cue_start is None or cue_end is None:
        return None
    parent: Cue = {
        "text": text,
        "start": float(cue_start),
        "end": float(cue_end),
        "word_data": word_data,
    }
    atoms = _build_atoms(
        text,
        word_data,
        ctx.lang,
        max_atom_width=_line_budget_width(ctx.max_line_length, ctx.lang),
    )
    if not atoms or not _semantic_annotate_unit_ranges(atoms, word_data, ctx.lang):
        return None
    if any(len(_split_oversized_atom(atom, parent, ctx)) != 1 for atom in atoms):
        return None
    fallback_atom_boundaries: set[int] | None = None
    if fallback_unit_boundaries is not None:
        atom_after_unit = {
            int(atom["_unit_end"]): index for index, atom in enumerate(atoms, 1)
        }
        fallback_atom_boundaries = set()
        for unit_boundary in fallback_unit_boundaries:
            atom_boundary = atom_after_unit.get(unit_boundary)
            if atom_boundary is None:
                # A deterministic cue edge inside an indivisible atom cannot be
                # represented safely by the semantic graph.
                return None
            if 0 < atom_boundary < len(atoms):
                fallback_atom_boundaries.add(atom_boundary)
    phrase_boundary = (
        _phrase_boundary_atoms(atoms, text, ctx.lang) if _no_spaces(ctx.lang) else None
    )
    if phrase_boundary is not None:
        phrase_boundary.update(
            index for index, atom in enumerate(atoms) if atom.get("forced_boundary")
        )
    mandatory, all_pauses = _semantic_hard_boundaries(text, atoms, ctx)
    ranges = _semantic_window_ranges(
        len(atoms), mandatory=mandatory, phrase_boundary=phrase_boundary
    )
    plans: list[_SemanticSplitPlan] = []
    for start, end, hard_after in ranges:
        window_atoms = [dict(atom) for atom in atoms[start:end]]
        if end < len(atoms):
            window_atoms[-1]["_next_start"] = atoms[end].get("start")
        unit_start = window_atoms[0]["_unit_start"]
        unit_end = window_atoms[-1]["_unit_end"]
        window_words = word_data[unit_start:unit_end]
        for atom in window_atoms:
            atom["_unit_start"] -= unit_start
            atom["_unit_end"] -= unit_start
        local_boundary = (
            None
            if phrase_boundary is None
            else {index - start for index in phrase_boundary if start <= index < end}
            | {0}
        )
        local_pauses = {
            index - start: duration
            for index, duration in all_pauses.items()
            if start < index < end
        }
        window_text = _join([atom["text"] for atom in window_atoms], ctx.lang)
        plan = _prepare_semantic_window(
            text=window_text,
            atoms=window_atoms,
            word_data=window_words,
            phrase_boundary=local_boundary,
            pauses=local_pauses,
            hard_after=hard_after,
            ctx=ctx,
            desired_wps=desired_wps,
            fallback_indices=(
                None
                if fallback_atom_boundaries is None
                else tuple(
                    sorted(
                        boundary - start
                        for boundary in fallback_atom_boundaries
                        if start < boundary < end
                    )
                )
            ),
        )
        if plan is None:
            return None
        plans.append(plan)
    if [unit for plan in plans for unit in plan.word_data] != word_data:
        return None
    return plans


def _semantic_materialize(
    plan: _SemanticSplitPlan, breaks: tuple[int, ...], lang: str
) -> tuple[list[Cue], set[int]] | None:
    """Turn a validated path into cues by slicing the original aligned units."""

    if breaks != tuple(sorted(set(breaks))):
        return None
    if not set(plan.required_indices) <= set(breaks):
        return None
    nodes = (0, *breaks, len(plan.atoms))
    edges = tuple(zip(nodes, nodes[1:]))
    if any(edge not in plan.allowed_edges for edge in edges):
        return None
    cues: list[Cue] = []
    hard_after: set[int] = set()
    for start, end in edges:
        chunk = plan.atoms[start:end]
        unit_start = chunk[0]["_unit_start"]
        unit_end = chunk[-1]["_unit_end"]
        units = plan.word_data[unit_start:unit_end]
        cue_start = _span_start(chunk)
        cue_end = _span_end(chunk)
        if cue_start is None or cue_end is None:
            return None
        cues.append(
            {
                "text": _join([atom["text"] for atom in chunk], lang),
                "start": float(cue_start),
                "end": float(cue_end),
                "word_data": units,
            }
        )
        if end in plan.required_indices:
            hard_after.add(len(cues) - 1)
    if [unit for cue in cues for unit in cue["word_data"]] != plan.word_data:
        return None
    if any(
        cues[index]["start"] > cues[index]["end"]
        or (index and cues[index]["start"] < cues[index - 1]["start"])
        for index in range(len(cues))
    ):
        return None
    return cues, hard_after


def _semantic_polish_groups(
    groups: list[list[Cue]],
    *,
    lang: str,
    th: SplitThresholds,
    max_line_length: int,
    max_lines: int,
) -> list[Cue]:
    """Run the existing timing cleanup without merging across hard barriers."""

    prepared: list[list[Cue]] = []
    for group in groups:
        repaired = _repair_bound_particle_cues(
            group,
            lang=lang,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=th.max_cue_s,
            connected_gap_s=th.clause_ms / 1000.0,
        )
        merged = _merge_micro_cues(
            repaired,
            lang,
            max_gap_s=th.glue_gap_s,
            max_line_length=max_line_length,
            max_cue_s=th.max_cue_s,
            min_cue_s=th.min_cue_s,
            max_lines=max_lines,
        )
        merged = _glue_short_cues(
            merged,
            lang,
            max_gap_s=th.glue_gap_s,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=th.max_cue_s,
        )
        prepared.append(merged)
    polished: list[Cue] = []
    for index, group in enumerate(prepared):
        # A temporary next-group sentinel gives cleanup the real upper in-time
        # while keeping all text-merging passes on their own side of the barrier.
        if index + 1 < len(prepared) and prepared[index + 1]:
            sentinel = cast(Cue, dict(prepared[index + 1][0]))
            cleaned = _cleanup_cues(
                group + [sentinel],
                min_cue_s=th.min_cue_s,
                max_cue_s=th.max_cue_s,
                cps=th.cps,
                lag_out_s=th.lag_out_s,
            )[:-1]
        else:
            cleaned = _cleanup_cues(
                group,
                min_cue_s=th.min_cue_s,
                max_cue_s=th.max_cue_s,
                cps=th.cps,
                lag_out_s=th.lag_out_s,
            )
        polished.extend(cleaned)
    return polished


def _semantic_result_valid(
    cues: list[Cue],
    *,
    lang: str,
    th: SplitThresholds,
    max_line_length: int,
    max_lines: int,
) -> bool:
    """Final hard defence after timing cleanup/shot snapping.

    Min duration and reading speed are intentionally absent: they are soft
    path-quality signals and can be impossible for intrinsically fast source
    speech.  Text/layout, max duration, and monotone timing remain hard.
    """

    for index, cue in enumerate(cues):
        duration = cue["end"] - cue["start"]
        if duration < -1e-9:
            return False
        if index and cue["start"] < cues[index - 1]["start"] - 1e-9:
            return False
        display_text = strip_punct_for_subtitles(cue["text"])
        if not _fits_budget(display_text, max_line_length, max_lines, lang):
            return False
        if th.max_cue_s > 0 and duration > th.max_cue_s + 1e-9:
            return False
    return True


def smart_split_segments(
    segments: List[Dict[str, Any]],
    lang: str,
    max_line_length: Optional[int] = None,
    max_lines: Optional[int] = None,
    min_duration: float = DEFAULT_MIN_DURATION,
    desired_wps: float = DEFAULT_DESIRED_WPS,
    split_at_comma: bool = True,
    comma_split_min_len: Optional[int] = None,
    *,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: SplitThresholds | dict | None = None,
    shot_changes: list[float] | None = None,
    semantic_engine: SemanticBreakEngine | None = None,
    semantic_model: str | None = None,
) -> List[Cue]:
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
    # Accept a plain mapping (config.gap_thresholds / tests) and normalize to the typed form once.
    # th is None ⟺ legacy length-break-only mode (no gap/duration breaks, no cleanup pass).
    th = (
        SplitThresholds.from_mapping(thresholds)
        if isinstance(thresholds, dict)
        else thresholds
    )
    if semantic_engine is not None:
        # Compute the complete deterministic result first.  It is returned
        # verbatim if *any* semantic task/backend/validation step fails, making
        # the optional stage transactional across the whole transcript.
        baseline = smart_split_segments(
            segments,
            lang,
            max_line_length=max_line_length,
            max_lines=max_lines,
            min_duration=min_duration,
            desired_wps=desired_wps,
            split_at_comma=split_at_comma,
            comma_split_min_len=comma_split_min_len,
            speech_spans=speech_spans,
            thresholds=th,
            shot_changes=shot_changes,
        )
        try:
            from voxweave.semantic_breaks import DEFAULT_SEMANTIC_MODEL

            semantic_th = th if th is not None else SplitThresholds()
            semantic_ctx = SplitContext(
                lang=lang,
                max_line_length=max_line_length,
                max_lines=max_lines,
                th=semantic_th,
                do_new=True,
                speech_spans=speech_spans,
            )
            source_units = [
                unit
                for segment in segments
                if segment.get("text", "")
                for unit in list(segment.get("words", []) or [])
            ]
            baseline_unit_boundaries: set[int] = set()
            source_cursor_for_baseline = 0
            for cue_index, cue in enumerate(baseline[:-1]):
                left_units = list(cue.get("word_data", []) or [])
                right_units = list(baseline[cue_index + 1].get("word_data", []) or [])
                left_end = _span_end(left_units)
                right_start = _span_start(right_units)
                if left_end is None or right_start is None:
                    raise ValueError("deterministic cue boundary has no source timing")
                boundary = source_cursor_for_baseline
                while boundary < len(source_units):
                    unit_end = source_units[boundary].get("end")
                    if (
                        not isinstance(unit_end, (int, float))
                        or isinstance(unit_end, bool)
                        or float(unit_end) > left_end + 1e-6
                    ):
                        break
                    boundary += 1
                # Baseline display text drops prose punctuation.  Preserve any
                # zero/short punctuation units between its left text and the
                # next cue's first spoken unit so the mapped cut remains an
                # exact partition of the original aligned stream.
                while boundary < len(source_units):
                    unit = source_units[boundary]
                    unit_end = unit.get("end")
                    surface = str(unit.get("text", unit.get("word", "")))
                    if (
                        not surface
                        or strip_punct_for_subtitles(surface)
                        or not isinstance(unit_end, (int, float))
                        or isinstance(unit_end, bool)
                        or float(unit_end) > right_start + 1e-6
                    ):
                        break
                    boundary += 1
                if not source_cursor_for_baseline < boundary < len(source_units):
                    raise ValueError(
                        "deterministic cue boundaries are not monotone in source units"
                    )
                baseline_unit_boundaries.add(boundary)
                source_cursor_for_baseline = boundary

            plans: list[_SemanticSplitPlan] = []
            source_cursor = 0
            for segment in segments:
                if not segment.get("text", ""):
                    continue
                segment_units = list(segment.get("words", []) or [])
                segment_end = source_cursor + len(segment_units)
                local_fallback_boundaries = {
                    boundary - source_cursor
                    for boundary in baseline_unit_boundaries
                    if source_cursor < boundary < segment_end
                }
                segment_plans = _prepare_semantic_plans(
                    segment,
                    ctx=semantic_ctx,
                    desired_wps=desired_wps,
                    fallback_unit_boundaries=local_fallback_boundaries,
                )
                if segment_plans is None:
                    raise ValueError(
                        "segment cannot be represented by exact timed atoms"
                    )
                plans.extend(segment_plans)
                source_cursor = segment_end

            requests = [plan.task for plan in plans if plan.task is not None]
            decisions = semantic_engine.choose(
                requests,
                default_model=semantic_model or DEFAULT_SEMANTIC_MODEL,
            )
            if len(decisions) != len(requests) or any(
                getattr(decision, "source", None) != "model" for decision in decisions
            ):
                raise ValueError("semantic selector used deterministic fallback")
            decision_iter = iter(decisions)
            groups: list[list[Cue]] = [[]]
            for plan in plans:
                if plan.task is None:
                    breaks = plan.fallback_indices
                else:
                    raw_breaks = getattr(next(decision_iter), "break_indices", ())
                    if not isinstance(raw_breaks, (tuple, list)) or any(
                        isinstance(index, bool) or not isinstance(index, int)
                        for index in raw_breaks
                    ):
                        raise ValueError("semantic selector returned non-integer cuts")
                    breaks = tuple(raw_breaks)
                    if not _semantic_quality_not_significantly_worse(plan, breaks):
                        raise ValueError(
                            "semantic cuts are materially worse than fallback timing"
                        )
                materialized = _semantic_materialize(plan, breaks, lang)
                if materialized is None:
                    raise ValueError(
                        "semantic cuts failed host graph/timeline validation"
                    )
                plan_cues, hard_after = materialized
                for local_index, cue in enumerate(plan_cues):
                    groups[-1].append(cue)
                    if local_index in hard_after:
                        groups.append([])
                if plan.hard_after and groups[-1]:
                    groups.append([])
            groups = [group for group in groups if group]
            cues = _semantic_polish_groups(
                groups,
                lang=lang,
                th=semantic_th,
                max_line_length=max_line_length,
                max_lines=max_lines,
            )
            if shot_changes:
                cues = _snap_to_shots(
                    cues,
                    sorted(shot_changes),
                    snap_s=semantic_th.shot_snap_s,
                    max_cue_s=semantic_th.max_cue_s,
                )
            if not _semantic_result_valid(
                cues,
                lang=lang,
                th=semantic_th,
                max_line_length=max_line_length,
                max_lines=max_lines,
            ):
                raise ValueError("semantic cues failed final display/timing validation")
            for cue in cues:
                cue["text"] = strip_punct_for_subtitles(cue["text"])
                if th is not None:
                    cue["text"] = _merge_stutters(cue["text"])
                cue["text"] = wrap_cue_text(cue["text"], lang, max_lines)
            return cues
        except Exception as exc:  # noqa: BLE001 - optional stage is fail-safe
            log.warning(
                "semantic subtitle splitting failed; using deterministic fallback (%s)",
                exc,
            )
            return baseline

    all_cues: List[Cue] = []
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
        cue["text"] = wrap_cue_text(cue["text"], lang, max_lines)
    return cues
