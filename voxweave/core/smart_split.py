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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, cast

from .breakpoints import legal_break_index, phrase_atoms
from .conjunctions import conjunctions_by_language, get_comma
from .gap_split import gap_qualifies
from .kinsoku import line_end_penalty
from .langsets import LANGUAGES_WITHOUT_SPACES as LANGUAGES_WITHOUT_SPACES  # re-export
from .schema import Cue, Unit
from .layout import (
    WIDE_GLYPH_LANGUAGES,
    _comma_chars,
    _fits_budget,
    _join,
    _merge_stutters,
    _no_spaces,
    _strip_trailing_commas,
    _token_char_count,
    _tokens,
    _visual_len,
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

DEFAULT_MIN_DURATION = 3.0  # reading-speed pad for single cues
DEFAULT_DESIRED_WPS = 4.0  # target reading speed (English wps)

# Comma line-break: split into separate cues at commas, but only when both
# sides are at least this long (visual chars). Shorter clauses stay attached
# to a neighbor so we never strand a tiny fragment on its own cue.
DEFAULT_COMMA_SPLIT_MIN_LEN = 18  # latin / space-delimited
DEFAULT_COMMA_SPLIT_MIN_LEN_CJK = 6  # zh/ja/ko: chars are ~2x visual width

FORCE_BREAK_FACTOR = 1.5  # boundary-less run may exceed the line budget by at most this before a forced cut


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


def _build_atoms(text: str, word_data: list[Unit], lang: str) -> list[dict]:
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
    word_data: List[Unit],
    lang: str,
    max_line_length: int,
    max_lines: int,
    split_at_comma: bool = True,
    comma_split_min_len: Optional[int] = None,
) -> List[Cue]:
    sentences = _snap_sentence_breaks(text, _segment_sentences(text, lang), lang)
    cues: List[Cue] = []
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
    ctx: SplitContext,
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
    if gap_break and _atom_end_pen(prev) >= 2:
        gms = _gap_ms(prev.get("end"), atom.get("start"))
        if gms is not None and th.clause_ms <= gms < th.vad_skip_ms:
            gap_break = False
    tentative = _join([a["text"] for a in cur + [atom]], ctx.lang)
    len_overflow = (
        split_subtitle(tentative, ctx.max_line_length, ctx.lang).count("\n") + 1
        > ctx.max_lines
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
            cur, atom, at_boundary=at_boundary, ctx=ctx
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
        chunks = _pack_atoms_into_chunks(atoms, boundary=boundary, ctx=ctx)
        new_cues.extend(_chunk_to_cue(chunk, cue, lang) for chunk in chunks)
    return new_cues


def _split_without_timings(
    cue: Cue,
    max_line_length: int,
    max_lines: int,
    lang: str,
) -> List[Cue]:
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
        max_lines = default_max_lines(lang)  # ja -> 1 (single line), else 2
    # Accept a plain mapping (config.gap_thresholds / tests) and normalize to the typed form once.
    # th is None ⟺ legacy length-break-only mode (no gap/duration breaks, no cleanup pass).
    th = (
        SplitThresholds.from_mapping(thresholds)
        if isinstance(thresholds, dict)
        else thresholds
    )
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
        cue["text"] = strip_punct_for_subtitles(cue["text"])
        if th is not None:  # stutter merging opt-in alongside gap-aware mode
            cue["text"] = _merge_stutters(cue["text"])
        # Display soft-wrap: fold over-budget cues into <=max_lines lines without
        # changing cue boundaries. Long Latin phrases inside CJK also collapse here.
        cue["text"] = wrap_cue_text(cue["text"], lang, max_lines)
    return cues
