"""Text primitives and display layout for subtitle cues.

Everything here is pure text-in/text-out: language-aware tokenization and
measurement, per-language line budgets, soft-wrapping (``split_subtitle`` /
``wrap_cue_text``), punctuation stripping and stutter merging. No timing, no
cue dicts — those live in ``timing`` (cue-stream polish) and ``smart_split``
(the segmentation engine), both of which build on this module.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .kinsoku import line_end_penalty
from .langsets import LANGUAGES_WITHOUT_SPACES

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


def _no_spaces(lang: str) -> bool:
    return lang in LANGUAGES_WITHOUT_SPACES


def _join(words: List[str], lang: str) -> str:
    return "".join(words) if _no_spaces(lang) else " ".join(words)


def _is_ascii_run_char(c: str) -> bool:
    """True for ASCII letters, digits, or in-word punctuation (._-).
    These chars form an inseparable Latin run inside CJK text."""
    if len(c) != 1 or ord(c) >= 128:
        return False
    return c.isalnum() or c in "._-"


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


def _reading_chars(text: str) -> int:
    """Non-whitespace char count — the reading-load measure for CPS lingering."""
    return sum(1 for ch in text if not ch.isspace())


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


def _fits_budget(text: str, max_line_length: int, max_lines: int, lang: str) -> bool:
    """True when ``text`` soft-wraps into at most ``max_lines`` lines."""
    return split_subtitle(text, max_line_length, lang).count("\n") + 1 <= max_lines


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


def strip_punct_for_subtitles(text: str) -> str:
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


def _slide_sticky_line_ends(groups: List[List[Tuple[str, str]]], lang: str) -> None:
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


def wrap_cue_text(text: str, lang: str, max_lines: int) -> str:
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
