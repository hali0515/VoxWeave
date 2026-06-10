"""Semantic subtitle splitting with gap-aware cue segmentation.

Two stages:
1. ``split_at_sentence_end`` — PySBD (or regex fallback) sentence boundaries,
   then ``split_sentence_heuristically`` for comma/conjunction splits.
2. ``split_long_cues_with_word_timings`` — word-level greedy packing into
   cues fitting ``max_lines × max_line_length``, with gap/duration breaks.

Each sentence/comma clause is its own cue so timings track real speech
boundaries; the one exception is ``_glue_short_cues``, which folds a lone-word
flicker cue onto whichever neighbor abuts it within a sub-0.3s gap (no real
pause crossed) — forward for leading interjections, backward for tail fragments.
"""

from __future__ import annotations

import bisect
import functools
import logging
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from .conjunctions import conjunctions_by_language, get_comma
from .breakpoints import legal_break_index, phrase_atoms
from .gap_split import gap_qualifies
from .kinsoku import line_end_penalty
from .langsets import LANGUAGES_WITHOUT_SPACES

log = logging.getLogger(__name__)

# Languages whose glyphs render at ~2x the visual width of Latin chars,
# so the per-line character budget must be roughly halved.
WIDE_GLYPH_LANGUAGES = {"zh", "ja", "ko"}

DEFAULT_MAX_LINE_LENGTH = 42  # latin / space-delimited
DEFAULT_MAX_LINE_LENGTH_CJK = 12  # ko: 2 lines × 12 chars (~2x visual width)
# zh/ja are single-line langs: one line gets a wider budget than ko's 12 so a
# short sentence fits whole; content over 18 chars splits into more cues, never wraps.
DEFAULT_MAX_LINE_LENGTH_CJK_SINGLE = 18
DEFAULT_MAX_LINES = 2
# zh/ja read best as one line per cue — stacking two lines can break mid-token
# (e.g. です -> で/す). Long utterances split into more single-line cues instead.
SINGLE_LINE_LANGS = {"zh", "ja"}
DEFAULT_MIN_DURATION = 3.0  # reading-speed pad for single cues
DEFAULT_DESIRED_WPS = 4.0  # target reading speed (English wps)

# Comma line-break: split into separate cues at commas, but only when both
# sides are at least this long (visual chars). Shorter clauses stay attached
# to a neighbor so we never strand a tiny fragment on its own cue.
DEFAULT_COMMA_SPLIT_MIN_LEN = 18  # latin / space-delimited
DEFAULT_COMMA_SPLIT_MIN_LEN_CJK = 6  # zh/ja/ko: chars are ~2x visual width


def default_max_lines(lang: str) -> int:
    """zh/ja: 1 line per cue (long utterances split into more cues). Others: 2."""
    return 1 if lang in SINGLE_LINE_LANGS else DEFAULT_MAX_LINES


def default_max_line_length(lang: str) -> int:
    """Per-language line length. CJK glyphs render ~2x Latin width, so the budget
    is halved. Single-line zh/ja gets a wider budget so short sentences fit whole."""
    if lang in SINGLE_LINE_LANGS:
        return DEFAULT_MAX_LINE_LENGTH_CJK_SINGLE
    return (
        DEFAULT_MAX_LINE_LENGTH_CJK
        if lang in WIDE_GLYPH_LANGUAGES
        else DEFAULT_MAX_LINE_LENGTH
    )


def default_comma_split_min_len(lang: str) -> int:
    """Minimum clause length (visual chars) for a comma to become a cue boundary.
    Wide-glyph languages use a smaller value (~2x visual width per char)."""
    return (
        DEFAULT_COMMA_SPLIT_MIN_LEN_CJK
        if lang in WIDE_GLYPH_LANGUAGES
        else DEFAULT_COMMA_SPLIT_MIN_LEN
    )


# Comma variants for no-space langs: treated as clause boundaries and later stripped to a space
# by _PUNCT_TO_SPACE_RE. The CJK subset is the single source for both this boundary set and that
# regex (halfwidth "," is added here but lives in the regex's digit-guarded first branch).
_CJK_PAUSE_COMMAS = (
    "，、﹐﹑"  # fullwidth, ideographic, small comma, small ideographic comma
)
_PAUSE_COMMAS_NO_SPACE = "," + _CJK_PAUSE_COMMAS  # + halfwidth comma


def _comma_chars(lang: str) -> str:
    """Comma characters treated as clause boundaries for this language."""
    return _PAUSE_COMMAS_NO_SPACE if _no_spaces(lang) else ","


def _strip_trailing_commas(s: str, lang: str) -> str:
    commas = _comma_chars(lang)
    while s and s[-1] in commas:
        s = s[:-1]
    return s


def _visual_len(s: str, lang: str) -> int:
    """Non-whitespace char count excluding trailing commas (for min-length tests)."""
    s = _strip_trailing_commas(s.strip(), lang)
    return sum(1 for c in s if not c.isspace())


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


def _no_spaces(lang: str) -> bool:
    return lang in LANGUAGES_WITHOUT_SPACES


def _join(words: List[str], lang: str) -> str:
    return "".join(words) if _no_spaces(lang) else " ".join(words)


# Unit glyphs that bind to a preceding digit: 92|% must never split across cues
# or lines. Covers halfwidth/fullwidth percent, permille, degree, temperature.
_UNIT_GLYPHS = "%％‰°℃℉"


def _tokens(text: str, lang: str) -> List[str]:
    """Tokenize for word-count alignment with ASR ``words`` entries.

    Space-delimited langs: ``text.split()``. CJK: each CJK char is one token;
    consecutive ASCII letters/digits/in-word punctuation (possibly with spaces
    between them) form one inseparable token. This keeps Latin phrases like
    ``Building in a week`` atomic inside CJK text. A unit glyph (%/℃/...)
    after a digit merges into that token so 92% stays one atom.
    """
    if not _no_spaces(lang):
        return text.split()

    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if _is_ascii_run_char(ch):
            j = i
            while j < n:
                cj = text[j]
                if _is_ascii_run_char(cj):
                    j += 1
                    continue
                if cj.isspace():
                    # Continue ASCII run only if next non-space is ASCII too.
                    k = j + 1
                    while k < n and text[k].isspace():
                        k += 1
                    if k < n and _is_ascii_run_char(text[k]):
                        j = k
                        continue
                break
            out.append(text[i:j])
            i = j
        else:
            out.append(ch)
            i += 1
    merged: List[str] = []
    for tok in out:
        if merged and tok in _UNIT_GLYPHS and merged[-1][-1].isdigit():
            merged[-1] += tok
            continue
        merged.append(tok)
    return merged


def _token_char_count(tok: str) -> int:
    """Non-whitespace chars in a token — used to advance the char-level word_data cursor."""
    return sum(1 for c in tok if not c.isspace())


def _span_start(items: list[dict], default: float | None = None) -> float | None:
    """First non-None ``start`` across items, else ``default``."""
    return next(
        (it.get("start") for it in items if it.get("start") is not None), default
    )


def _span_end(items: list[dict], default: float | None = None) -> float | None:
    """Last non-None ``end`` across items, else ``default``."""
    return next(
        (it.get("end") for it in reversed(items) if it.get("end") is not None), default
    )


def _build_atoms(text: str, word_data: list[dict], lang: str) -> list[dict]:
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


def _is_ascii_run_char(c: str) -> bool:
    """True for ASCII letters, digits, or in-word punctuation (._-).
    These chars form an inseparable Latin run inside CJK text."""
    if len(c) != 1 or ord(c) >= 128:
        return False
    return c.isalnum() or c in "._-"


def split_subtitle(text: str, max_chars: int, lang: str) -> str:
    """Soft-wrap text to lines of <= max_chars, breaking at token boundaries."""
    tokens = _tokens(text, lang)
    if not tokens:
        return text
    sep = "" if _no_spaces(lang) else " "
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for tok in tokens:
        tlen = len(tok)
        extra = len(sep) if current else 0
        if current and current_len + tlen + extra > max_chars:
            lines.append(sep.join(current))
            current = [tok]
            current_len = tlen
        else:
            current.append(tok)
            current_len += tlen + extra
    if current:
        lines.append(sep.join(current))
    return "\n".join(lines)


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


def _fits_budget(text: str, max_line_length: int, max_lines: int, lang: str) -> bool:
    """True when ``text`` soft-wraps into at most ``max_lines`` lines."""
    return split_subtitle(text, max_line_length, lang).count("\n") + 1 <= max_lines


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
        packed.append(part)
    return packed


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
    parts = [p.strip() for p in pattern.split(clause) if p and p.strip()]
    parts = _repack_parts(parts, max_line_length, max_lines, lang)

    final_parts: List[str] = []
    for part in parts:
        f_part = split_subtitle(part, max_line_length, lang)
        if f_part.count("\n") + 1 > max_lines:
            toks = _tokens(part, lang)
            target = len(toks) // 2 or 1
            if _no_spaces(lang):
                mid = _snap_mid_to_phrase_boundary(toks, part, lang, target)
            else:
                mid = legal_break_index(toks, lang, target)
            final_parts.extend([_join(toks[:mid], lang), _join(toks[mid:], lang)])
        else:
            final_parts.append(part)
    return [p for p in final_parts if p]


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
    word_data: List[Dict[str, Any]],
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
        return all(
            word_data[at + j].get("word") == tok for j, tok in enumerate(clause_tokens)
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
    word_data: List[Dict[str, Any]],
    lang: str,
    max_line_length: int,
    max_lines: int,
    split_at_comma: bool = True,
    comma_split_min_len: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sentences = _snap_sentence_breaks(text, _segment_sentences(text, lang), lang)
    cues: List[Dict[str, Any]] = []
    cursor = 0
    # Content verification needs unit texts; legacy callers without a "word"
    # key keep the blind index cursor.
    anchored = bool(word_data) and "word" in word_data[0]
    for sent in sentences:
        for clause in split_sentence_heuristically(
            sent, max_line_length, max_lines, lang, split_at_comma, comma_split_min_len
        ):
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
    cur: List[dict], cur_bnd: List[bool], at_boundary_next: bool
) -> int:
    """Split position on a length overflow (Level 1 kinsoku).

    Candidates: break before the incoming atom (pos n) and any internal
    phrase-start k (0<k<n). Pick the candidate whose left side ends with the
    smallest sticky-token penalty (``end_pen``); ties go to the fullest
    line. Falls back to n (greedy) when no candidates exist.
    """
    n = len(cur)
    cands: List[tuple[int, int]] = []  # (penalty, split_pos)
    if at_boundary_next:
        cands.append((_atom_end_pen(cur[-1]), n))
    for k in range(1, n):
        if cur_bnd[k]:
            cands.append((_atom_end_pen(cur[k - 1]), k))
    if not cands:
        return n
    return min(cands, key=lambda pk: (pk[0], -pk[1]))[1]


def _classify_atom_break(
    cur: List[dict],
    atom: dict,
    *,
    at_boundary: bool,
    do_new: bool,
    th: SplitThresholds,
    speech_spans: list[tuple[float, float]] | None,
    max_line_length: int,
    max_lines: int,
    lang: str,
) -> tuple[bool, bool, bool]:
    """Decide ``(gap_break, dur_break, len_break)`` for appending ``atom`` after ``cur``.

    Pure: reads ``cur``/``atom``/thresholds, mutates nothing. All-False when ``cur`` is empty
    (the first atom of a chunk never breaks).
    - gap_break: a qualifying inter-atom pause, but only at a phrase boundary, and suppressed in
      the clause_ms..vad_skip_ms zone when it would strand a sticky token at line end.
    - dur_break: hard last-resort cap when the running cue would exceed ``max_cue_s`` (ignores
      word boundaries — intra-word spans over the cap are rare and an overlong cue is worse).
    - len_break: the line budget overflows AND this atom is a legal (phrase-start) break point.
    """
    if not cur:
        return False, False, False
    prev = cur[-1]
    # Gap/len breaks require a word boundary (no-space langs): atom must be a BudouX(ja)/jieba(zh)
    # phrase start. Guards against CTC timing errors on OOV chars creating spurious intra-word gaps
    # (e.g. 酒造り: 番酒造 OOV drift makes a 2.1s gap between 造 and り, but BudouX keeps 番酒造りが
    # as one phrase, suppressing the spurious split). The dur_break cap is exempt and always cuts.
    gap_break = (
        do_new
        and at_boundary
        and gap_qualifies(
            prev.get("end"),
            atom.get("start"),
            speech_spans,
            clause_ms=th.clause_ms,
            vad_skip_ms=th.vad_skip_ms,
            offline_ms=th.offline_ms,
        )
    )
    # In the clause_ms..vad_skip_ms zone, suppress the gap-split if it would strand a sticky
    # token at line end: ja 大樹の|村, zh ...的|... , en a hesitation after "the". True
    # silence (>=vad_skip_ms) always cuts — a real pause beats line-end aesthetics.
    if gap_break and _atom_end_pen(prev) >= 2:
        gms = _gap_ms(prev.get("end"), atom.get("start"))
        if gms is not None and th.clause_ms <= gms < th.vad_skip_ms:
            gap_break = False
    tentative = _join([a["text"] for a in cur + [atom]], lang)
    len_overflow = (
        split_subtitle(tentative, max_line_length, lang).count("\n") + 1 > max_lines
    )
    len_break = len_overflow and at_boundary
    # Boundary-less overlong run (a single phrase atom exceeding the budget by
    # 1.5x, e.g. a long katakana loan chain): bail out off-boundary rather than
    # emit a mega-line — _best_len_break_pos still prefers any earlier legal
    # boundary held in the running chunk.
    if (
        len_overflow
        and not len_break
        and _token_char_count(tentative)
        > round(FORCE_BREAK_FACTOR * max_line_length * max_lines)
    ):
        len_break = True
    start0 = _span_start(cur)
    dur_break = (
        do_new
        and start0 is not None
        and atom.get("end") is not None
        and (atom["end"] - start0) > th.max_cue_s
    )
    return gap_break, dur_break, len_break


def _pack_atoms_into_chunks(
    atoms: List[dict],
    *,
    boundary: set[int] | None,
    do_new: bool,
    th: SplitThresholds,
    speech_spans: list[tuple[float, float]] | None,
    max_line_length: int,
    max_lines: int,
    lang: str,
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
            do_new=do_new,
            th=th,
            speech_spans=speech_spans,
            max_line_length=max_line_length,
            max_lines=max_lines,
            lang=lang,
        )
        if cur and (gap_break or dur_break):
            chunks.append(cur)
            cur, cur_bnd = [atom], [at_boundary]
        elif cur and len_break:
            k = _best_len_break_pos(cur, cur_bnd, at_boundary)
            chunks.append(cur[:k])
            cur, cur_bnd = cur[k:] + [atom], cur_bnd[k:] + [at_boundary]
        else:
            cur.append(atom)
            cur_bnd.append(at_boundary)
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_to_cue(chunk: List[dict], cue: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """Materialize a packed atom chunk into a cue dict (first/last non-None span, falling back
    to the parent cue's start/end)."""
    return {
        "text": _join([a["text"] for a in chunk], lang),
        "start": _span_start(chunk, cue["start"]),
        "end": _span_end(chunk, cue["end"]),
        "word_data": [{"start": a["start"], "end": a["end"]} for a in chunk],
    }


def split_long_cues_with_word_timings(
    cues: List[Dict[str, Any]],
    max_line_length: int,
    max_lines: int,
    min_duration: float,
    desired_wps: float,
    lang: str,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: Optional[SplitThresholds] = None,
) -> List[Dict[str, Any]]:
    """Pack each cue's atoms into reading-sized cues using gap/duration/length breaks.

    ``min_duration`` / ``desired_wps`` are kept for back-compat (unused on the atom-based path).
    ``thresholds=None`` is the legacy length-break-only path: ``do_new=False`` disables the
    gap/duration breaks, so the threshold values are never read (the default instance is a
    never-read placeholder there).
    """
    do_new = thresholds is not None
    th = thresholds if thresholds is not None else SplitThresholds()
    new_cues: List[Dict[str, Any]] = []
    for cue in cues:
        word_data = list(cue.get("word_data") or [])
        if not word_data:
            new_cues.extend(
                _split_without_timings(cue, max_line_length, max_lines, lang)
            )
            continue
        atoms = _build_atoms(cue["text"], word_data, lang)
        boundary = (
            _phrase_boundary_atoms(atoms, cue["text"], lang)
            if do_new and _no_spaces(lang)
            else None
        )
        _attach_end_penalties(atoms, boundary, lang)
        chunks = _pack_atoms_into_chunks(
            atoms,
            boundary=boundary,
            do_new=do_new,
            th=th,
            speech_spans=speech_spans,
            max_line_length=max_line_length,
            max_lines=max_lines,
            lang=lang,
        )
        new_cues.extend(_chunk_to_cue(chunk, cue, lang) for chunk in chunks)
    return new_cues


def _split_without_timings(
    cue: Dict[str, Any],
    max_line_length: int,
    max_lines: int,
    lang: str,
) -> List[Dict[str, Any]]:
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
    out: List[Dict[str, Any]] = []
    for c in chunks:
        # Join without \n — the downstream SubtitlesWriter handles display wrapping.
        text = sep.join(c)
        proportion = len(text) / total_chars
        end = start + duration * proportion if duration > 0 else start
        out.append({"text": text, "start": start, "end": end, "word_data": []})
        start = end
    return out


TWO_FRAME_S = 2.0 / 24.0  # ~0.083s Netflix min inter-cue gap
CHAIN_MAX_GAP_S = 0.5  # gaps below this are "dead zone" -> chain to 2 frames
VISIBLE_GAP_MIN_S = 1.0  # gaps >= this stay a visible pause (BBC); not enforced in code (CHAIN_MAX_GAP_S=0.5 never reaches them)
GLUE_MAX_GAP_S = 0.3  # lone-word flicker cue glues onto its nearer neighbor when that gap is below this
LINGER_CAP_S = 1.0  # CPS-driven extension never lingers more than this past speech end
FORCE_BREAK_FACTOR = 1.5  # boundary-less run may exceed the line budget by at most this before a forced cut


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
    # Shot-change snap window (0 = off): a cue boundary within this of a detected
    # cut moves onto it (see _snap_to_shots). Only consulted when the caller
    # passes shot_changes.
    shot_snap_s: float = 0.24

    @classmethod
    def from_mapping(cls, d: dict) -> SplitThresholds:
        """Build from a (possibly partial) mapping, ignoring unknown keys and filling defaults."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _is_short_fragment(text: str, lang: str) -> bool:
    """A flicker fragment worth gluing: a lone short word (spaced langs) or 1-2 CJK
    chars (no-space langs). Keeps the glue surgical — real clauses that merely abut
    a neighbor are not fragments and stay their own cue. Text size, not duration, is
    the flicker signal (a lone 「ん」 held 0.8s is still a flicker)."""
    t = text.strip()
    if not t:
        return False
    if _no_spaces(lang):
        return _visual_len(t, lang) <= 2
    return len(t.split()) == 1


def _gap_between(a: Dict[str, Any], b: Dict[str, Any]) -> float | None:
    """Inter-cue gap a->b (b.start - a.end), or None if either bound is missing."""
    ae, bs = a.get("end"), b.get("start")
    return (bs - ae) if ae is not None and bs is not None else None


def _merge_micro_cues(
    cues: List[Dict[str, Any]],
    lang: str,
    *,
    max_gap_s: float,
    max_line_length: int,
    max_cue_s: float,
) -> List[Dict[str, Any]]:
    """Merge adjacent cues separated by sub-glue gaps when the merge fits one line.

    Folds rapid micro-sentence chains (そう。だね。 / "Yeah." "Right.") into one
    readable cue instead of a flicker sequence. Safety mirrors _glue_short_cues:
    ``max_gap_s`` (0.3s) sits below ``clause_ms`` (0.4s), so a real pause is never
    crossed. A len-broken pair cannot re-merge (it would not fit one line), a
    gap-broken pair cannot either (its gap >= clause_ms), and the duration cap
    keeps a dur-broken pair apart. ``max_gap_s<=0`` disables.
    """
    if max_gap_s <= 0 or len(cues) < 2:
        return cues
    sep = "" if _no_spaces(lang) else " "
    out = [dict(cues[0])]
    for nxt in cues[1:]:
        cur = out[-1]
        gap = _gap_between(cur, nxt)
        merged_text = (cur["text"].rstrip() + sep + nxt["text"].lstrip()).strip()
        if (
            gap is not None
            and gap < max_gap_s
            and cur.get("start") is not None
            and nxt.get("end") is not None
            and nxt["end"] - cur["start"] <= max_cue_s
            and _fits_budget(merged_text, max_line_length, 1, lang)
        ):
            cur["text"] = merged_text
            cur["end"] = (
                nxt["end"] if cur.get("end") is None else max(cur["end"], nxt["end"])
            )
            cur["word_data"] = list(cur.get("word_data") or []) + list(
                nxt.get("word_data") or []
            )
            continue
        out.append(dict(nxt))
    return out


def _glue_short_cues(
    cues: List[Dict[str, Any]], lang: str, *, max_gap_s: float
) -> List[Dict[str, Any]]:
    """Glue a super-short single-word flicker cue onto whichever neighbor abuts it
    closest, when that gap is below ``max_gap_s`` — contiguous speech means a
    spurious split, not a real pause.

    Bidirectional: an interjection that *leads* the next line (え/ん with a sub-0.3s
    gap ahead but a real pause behind) glues forward; a tail fragment glues back.
    The side with the smaller gap wins (ties go backward). Safe re-introduction of
    the deleted merge_short_cues: ``max_gap_s`` (0.3s) sits below ``clause_ms``
    (0.4s), so the real-pause side (>=0.4s) is never crossed and a cue is never
    dragged over silence. ``max_gap_s<=0`` disables. Overflow is left to soft-wrap.
    """
    if max_gap_s <= 0 or len(cues) < 2:
        return cues
    sep = "" if _no_spaces(lang) else " "
    work = [dict(c) for c in cues]
    out: List[Dict[str, Any]] = []
    i, n = 0, len(work)
    while i < n:
        c = work[i]
        nxt = work[i + 1] if i + 1 < n else None
        if _is_short_fragment(c["text"], lang):
            gap_back = _gap_between(out[-1], c) if out else None
            gap_fwd = _gap_between(c, nxt) if nxt is not None else None
            back_ok = gap_back is not None and gap_back < max_gap_s
            fwd_ok = gap_fwd is not None and gap_fwd < max_gap_s
            # nearer side wins; ties go backward ("append to last cue").
            go_fwd = fwd_ok and (
                not back_ok or (gap_fwd is not None and gap_fwd < gap_back)  # type: ignore[operator]
            )
            if go_fwd and nxt is not None:  # prepend fragment into next, reprocess it
                nxt["text"] = (c["text"].rstrip() + sep + nxt["text"].lstrip()).strip()
                if c.get("start") is not None:
                    nxt["start"] = (
                        c["start"]
                        if nxt.get("start") is None
                        else min(nxt["start"], c["start"])
                    )
                nxt["word_data"] = list(c.get("word_data") or []) + list(
                    nxt.get("word_data") or []
                )
                i += 1
                continue
            if back_ok:
                prev = out[-1]
                prev["text"] = (
                    prev["text"].rstrip() + sep + c["text"].lstrip()
                ).strip()
                if c.get("end") is not None:
                    prev["end"] = (
                        c["end"]
                        if prev.get("end") is None
                        else max(prev["end"], c["end"])
                    )
                prev["word_data"] = list(prev.get("word_data") or []) + list(
                    c.get("word_data") or []
                )
                i += 1
                continue
        out.append(c)
        i += 1
    return out


def _reading_chars(text: str) -> int:
    """Non-whitespace char count — the reading-load measure for CPS lingering."""
    return sum(1 for ch in text if not ch.isspace())


def _cleanup_cues(
    cues: List[Dict[str, Any]],
    *,
    min_cue_s: float,
    max_cue_s: float,
    cps: float = 0.0,
    lag_out_s: float = 0.0,
) -> List[Dict[str, Any]]:
    """Timing-only pass — never merges content across a real pause.

    - Extends short cues into the following gap (no overlap) up to min_cue_s.
    - Reading-speed linger (cps>0): a cue displayed for less than reading_chars/cps
      extends into the gap, at most LINGER_CAP_S past speech end.
    - Tail pad (lag_out_s>0): every cue end gets a flat pad so text does not vanish
      the instant speech stops; absorbed by chaining in dense dialogue.
    - Chains sub-0.5s inter-cue gaps down to 2 frames.
    - Visible gaps (>=1s) are left untouched.
    - max_cue_s prevents any extension from re-inflating past the segmentation cap.
    """
    out = [dict(c) for c in cues]
    for i, c in enumerate(out):
        nxt_start = out[i + 1]["start"] if i + 1 < len(out) else None
        # desired duration: min-dur floor, CPS reading time (capped linger), tail pad
        dur = c["end"] - c["start"]
        desired = dur
        if min_cue_s > 0:
            desired = max(desired, min_cue_s)
        if lag_out_s > 0:
            desired = max(desired, dur + lag_out_s)
        if cps > 0:
            need = _reading_chars(c.get("text", "")) / cps
            desired = max(desired, min(need, dur + LINGER_CAP_S))
        if desired > dur:
            want = c["start"] + desired
            c["end"] = want if nxt_start is None else min(want, nxt_start)
        # chaining: close small inter-cue gaps to 2 frames
        if nxt_start is not None:
            gap = nxt_start - c["end"]
            if 0 <= gap < CHAIN_MAX_GAP_S and gap > TWO_FRAME_S:
                c["end"] = nxt_start - TWO_FRAME_S
            # overlaps (gap<0) and large gaps (>=CHAIN_MAX_GAP_S) left to caller
        # never let extension / chaining push a cue past the duration cap
        if max_cue_s and c["end"] - c["start"] > max_cue_s:
            c["end"] = c["start"] + max_cue_s
    return out


def _snap_to_shots(
    cues: List[Dict[str, Any]],
    shots: List[float],
    *,
    snap_s: float,
    max_cue_s: float,
) -> List[Dict[str, Any]]:
    """Snap cue boundaries onto nearby shot changes (runs after _cleanup_cues).

    A boundary within ``snap_s`` of a cut moves onto it, but never at speech's
    expense:

    - start: moving *earlier* to the cut is a free lead-in (bounded by the
      previous cue end + 2 frames); moving *later* (pre-cut flash removal) is
      bounded by ``snap_s`` and must stay below the cue's end.
    - end: extending to cut - 2 frames is free inside the following gap (and
      the duration cap); pulling back to cut - 2 frames must not cut speech
      (never below the last word's end).

    Cues then change exactly on the cut instead of flashing across it.
    """
    if snap_s <= 0 or not shots:
        return cues
    out = [dict(c) for c in cues]

    def nearest(t: float) -> float | None:
        i = bisect.bisect_left(shots, t)
        best: float | None = None
        for j in (i - 1, i):
            if 0 <= j < len(shots) and abs(shots[j] - t) <= snap_s:
                if best is None or abs(shots[j] - t) < abs(best - t):
                    best = shots[j]
        return best

    for i, c in enumerate(out):
        start, end = c.get("start"), c.get("end")
        if start is None or end is None:
            continue
        words = [w for w in c.get("word_data") or [] if w.get("end") is not None]
        speech_end = max((w["end"] for w in words), default=end)
        prev_end = out[i - 1].get("end") if i > 0 else None
        nxt_start = out[i + 1].get("start") if i + 1 < len(out) else None

        cut = nearest(start)
        if cut is not None and abs(cut - start) > 1e-9:
            new_start = cut
            if prev_end is not None:
                new_start = max(new_start, prev_end + TWO_FRAME_S)
            if new_start < end - TWO_FRAME_S and (
                new_start <= start or new_start - start <= snap_s
            ):
                c["start"] = new_start

        cut = nearest(end)
        if cut is not None:
            target = cut - TWO_FRAME_S
            if target > end:  # extend to die on the cut
                if (
                    (nxt_start is None or target <= nxt_start - TWO_FRAME_S)
                    and target - c["start"] <= max_cue_s
                ):
                    c["end"] = target
            elif target < end:  # pull back, never cutting speech
                if target >= speech_end and target > c["start"]:
                    c["end"] = target
    return out


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
) -> List[Dict[str, Any]]:
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
        max_lines = default_max_lines(lang)  # ja -> 1 (single line), else 2
    # Accept a plain mapping (config.gap_thresholds / tests) and normalize to the typed form once.
    # th is None ⟺ legacy length-break-only mode (no gap/duration breaks, no cleanup pass).
    th = (
        SplitThresholds.from_mapping(thresholds)
        if isinstance(thresholds, dict)
        else thresholds
    )
    all_cues: List[Dict[str, Any]] = []
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
        cues = _merge_micro_cues(
            cues,
            lang,
            max_gap_s=th.glue_gap_s,
            max_line_length=max_line_length,
            max_cue_s=th.max_cue_s,
        )
        cues = _glue_short_cues(cues, lang, max_gap_s=th.glue_gap_s)
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
        cue["text"] = _strip_punct_for_subtitles(cue["text"])
        if th is not None:  # stutter merging opt-in alongside gap-aware mode
            cue["text"] = _merge_stutters(cue["text"])
        # Display soft-wrap: fold over-budget cues into <=max_lines lines without
        # changing cue boundaries. Long Latin phrases inside CJK also collapse here.
        cue["text"] = _wrap_cue_text(cue["text"], lang, max_lines)
    return cues


# "I I" -> "I-I"; ASCII letters only (CJK and digit-containing tokens skipped).
_STUTTER_RE = re.compile(r"\b([A-Za-z]+)(\s+)(\1)\b", re.IGNORECASE)


def _merge_stutters(text: str) -> str:
    """Merge adjacent repeated ASCII words into hyphenated stutter form.

    "I I I" -> "I-I-I" (iterates to fixpoint for 3+ repetitions). Existing
    compound words like well-known are unaffected.
    """
    prev = None
    while prev != text:
        prev = text
        text = _STUTTER_RE.sub(r"\1-\3", text)
    return text


# Punctuation to replace with a space. "." and "," between digits are kept
# (e.g. 3.75, 10,000). Covers CJK fullwidth variants.
_PUNCT_TO_SPACE_RE = re.compile(
    r"[.,](?!\d)"  # latin . , only when next char is not a digit
    r"|[;!?:。；！？：﹒﹔﹕﹖﹗"
    + _CJK_PAUSE_COMMAS
    + r"]"  # halfwidth punct + CJK set (commas shared)
)
_WS_RE = re.compile(r"\s+")


def _strip_punct_for_subtitles(text: str) -> str:
    """Replace punctuation with spaces (digit-internal . and , kept), then
    collapse whitespace runs and trim."""
    cleaned = _PUNCT_TO_SPACE_RE.sub(" ", text)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _vis_width(s: str) -> int:
    """Visual width: CJK/full-width glyphs count 2; ASCII/space counts 1."""
    return sum(1 if (c.isascii() or c.isspace()) else 2 for c in s)


def _wrap_units(text: str, lang: str) -> List[Tuple[str, str]]:
    """Split a cue into ``(atom, gap_after)`` pairs for display wrapping.

    Line-breaks are legal between units only. ``gap_after`` (" " or "") preserves
    the original spacing so rejoining atoms reproduces the text exactly.

    - Space-delimited langs: each word is an atom (gap " ").
    - No-space langs: maximal ASCII runs are one atom; each CJK glyph is its own.
      Unlike ``_tokens``, this does NOT bridge ASCII runs across spaces, so long
      embedded English phrases can wrap.
    """
    if not _no_spaces(lang):
        return [(w, " ") for w in text.split()]
    units: List[Tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if _is_ascii_run_char(c):
            j = i
            while j < n and _is_ascii_run_char(text[j]):
                j += 1
            atom = text[i:j]
            i = j
        else:
            atom = c
            i += 1
        gap = ""
        if i < n and text[i].isspace():
            gap = " "
            while i < n and text[i].isspace():
                i += 1
        # a unit glyph directly after a digit joins it (92% never line-wraps apart);
        # only when no gap intervenes — rejoining must reproduce the text exactly
        if (
            units
            and atom in _UNIT_GLYPHS
            and units[-1][1] == ""
            and units[-1][0][-1].isdigit()
        ):
            prev_atom, _ = units.pop()
            atom = prev_atom + atom
        units.append((atom, gap))
    return units


def _join_line(units: List[Tuple[str, str]]) -> str:
    """Join (atom, gap) units into one line; trailing gap of last atom is dropped."""
    out: List[str] = []
    for k, (atom, gap) in enumerate(units):
        out.append(atom)
        if k < len(units) - 1:
            out.append(gap)
    return "".join(out)


def _slide_sticky_line_ends(
    groups: List[List[Tuple[str, str]]], lang: str
) -> None:
    """Slide sticky trailing tokens (line_end_penalty >= 1) down to the next line.

    The token-level counterpart of apply_kinsoku for spaced languages: a line must
    not end on a closed-class token (went to the | store). Slides while the donor
    keeps at least one token and the receiving line stays within the hard visual
    budget; bottom-heavy output is fine (pyramid shape reads better anyway).
    """
    for i in range(len(groups) - 1):
        top, bot = groups[i], groups[i + 1]
        while (
            len(top) > 1
            and line_end_penalty(top[-1][0], lang) >= 1
            and _vis_width(_join_line([top[-1], *bot])) <= DEFAULT_MAX_LINE_LENGTH
        ):
            bot.insert(0, top.pop())


def _wrap_cue_text(text: str, lang: str, max_lines: int) -> str:
    """Soft-wrap a cue into ``<=max_lines`` display lines (``\\n``-joined).

    Only changes rendered layout — cue boundaries and content are untouched.
    Wraps only when visual width exceeds ``DEFAULT_MAX_LINE_LENGTH``; short CJK
    cues with brief Latin phrases stay on one line. Lines are balanced at
    ``ceil(total/max_lines)`` to avoid stranding a fragment on the last line,
    then line ends are cleaned: kinsoku char rules for ja/zh, sticky-token
    slide for spaced languages.
    """
    units = _wrap_units(text, lang)
    if len(units) <= 1:
        return _join_line(units) if units else text
    total = _vis_width(_join_line(units))
    if total <= DEFAULT_MAX_LINE_LENGTH:  # fits on one line -> no wrap needed
        return _join_line(units)
    target = -(-total // max_lines)  # ceil div: balance across max_lines lines
    groups: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    for u in units:
        if (
            cur
            and _vis_width(_join_line(cur + [u])) > target
            and len(groups) < max_lines - 1
        ):
            groups.append(cur)
            cur = [u]
        else:
            cur.append(u)
    if cur:
        groups.append(cur)
    if not _no_spaces(lang) and len(groups) > 1:
        _slide_sticky_line_ends(groups, lang)
    lines = [_join_line(g) for g in groups]
    if lang in {"ja", "zh"} and len(lines) > 1:
        from .kinsoku import apply_kinsoku

        lines = apply_kinsoku(lines)
    return "\n".join(lines)
