"""Break-point helpers: EN forbidden-break guard and BudouX/jieba phrase atoms (ja/zh)."""

from __future__ import annotations

import functools

from .langsets import LANGUAGES_WITHOUT_SPACES as _NO_SPACE

# Closed-class tokens (articles/preps/aux/conj) that must NOT end a line —
# ending here strands the token from the word it modifies.
_FORBIDDEN_LEFT = {
    # articles / determiners
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "some",
    "any",
    "no",
    "every",
    "each",
    # prepositions
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "about",
    "as",
    "than",
    # coordinating / common conjunctions (break goes BEFORE them, not after)
    "and",
    "or",
    "but",
    "nor",
    "so",
    "yet",
    "because",
    "if",
    "while",
    "when",
    # auxiliary verbs
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "am",
    "will",
    "would",
    "can",
    "could",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
}


def legal_break_index(tokens: list[str], lang: str, target: int) -> int:
    """Return a break index near ``target`` whose left side does not end on a forbidden token.

    Searches outward from target; falls back to target if none found.
    No-op for non-English (CJK handled by BudouX phrase atoms).
    """
    n = len(tokens)
    target = max(1, min(target, n))
    if lang != "en" or n <= 1:
        return target

    def ok(i: int) -> bool:
        return (
            1 <= i < n and tokens[i - 1].strip(".,!?;:").lower() not in _FORBIDDEN_LEFT
        )

    if ok(target):
        return target
    for d in range(1, n):
        for cand in (target - d, target + d):
            if ok(cand):
                return cand
    return target


# _NO_SPACE is the shared LANGUAGES_WITHOUT_SPACES (imported above), including
# yue so Cantonese follows the same character-level policy as written Chinese.


@functools.lru_cache(maxsize=8)
def _load_parser(lang: str):
    """Lazy BudouX parser singleton per lang; None if budoux is absent or has no model for this lang.

    Patchable in tests via monkeypatch.setattr — phrase_atoms looks it up through the module
    global so the replacement is honored.
    """
    try:
        import budoux  # type: ignore
    except ImportError:
        return None
    if lang == "ja":
        return budoux.load_default_japanese_parser()
    if lang == "zh":
        return budoux.load_default_simplified_chinese_parser()
    return None  # th/lo/my -> per-char fallback (no bundled model wired here)


@functools.lru_cache(maxsize=1)
def _load_jieba():
    """Lazy jieba singleton; None if jieba is absent.

    BudouX's zh model is too weak (glues 29%数据, over-splits 每年); jieba reliably
    segments 数据中心/值得/本季度/每年. Used to snap Qwen's ~1-char-off zh punctuation
    onto real word boundaries (see :func:`voxweave.realign.snap_break_punct`).
    """
    try:
        import logging

        import jieba  # type: ignore

        jieba.setLogLevel(logging.ERROR)  # silence "Building prefix dict..."
        return jieba
    except ImportError:
        return None


def word_starts(text: str, lang: str) -> set[int] | None:
    """Character offsets of word starts in ``text`` (range 1..len-1, excludes 0 and end).

    Qwen/fusion punctuation is often off by <=1 char (zh ``29%数。据``, ja ``番酒造。り``);
    this set is used to snap it to the nearest word boundary (see
    :func:`voxweave.realign.snap_break_punct`). Returns ``None`` when no segmenter is
    available (per-char fallback → no valid boundary → caller skips snap).
    """
    atoms = phrase_atoms(text, lang)
    if not any(
        len(a) > 1 for a in atoms
    ):  # per-char fallback (no segmenter) → no valid boundary
        return None
    starts: set[int] = set()
    off = 0
    for a in atoms:
        if off > 0:
            starts.add(off)
        off += len(a)
    return starts


def phrase_atoms(text: str, lang: str) -> list[str]:
    """Break ``text`` into non-breakable atoms: whitespace-split for spaced langs,
    BudouX phrases for no-space langs. Falls back to per-char when no segmenter is
    available. ``"".join(phrase_atoms(t, cjk)) == t`` (minus whitespace).

    th/lo/my are in _NO_SPACE but have no bundled BudouX model, so they always
    fall back to per-char even when budoux is installed.
    """
    if lang not in _NO_SPACE:
        return text.split()
    if (
        lang == "zh"
    ):  # jieba preferred (BudouX zh too weak); falls back to budoux/per-char
        jb = _load_jieba()
        if jb is not None:
            out = [w for w in jb.cut(text, HMM=True) if w.strip()]
            return out or [c for c in text if not c.isspace()]
    parser = _load_parser(lang)
    if parser is None:
        return [c for c in text if not c.isspace()]
    out = []
    for phrase in parser.parse(text):
        phrase = phrase.strip()
        if phrase:
            out.append(phrase)
    return out or [c for c in text if not c.isspace()]
