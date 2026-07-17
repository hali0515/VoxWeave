"""JIS X 4051 禁則処理 (line-breaking constraints) and line-end break scoring.

After lines are formed, ``apply_kinsoku`` slides breaks so no line starts with a
prohibited char (closing brackets, small kana, trailing punctuation) or ends
with a prohibited char (opening bracket/quote). Applied to ja and zh; small-kana
entries are inert in zh but the CJK punctuation rules apply to both.

``line_end_penalty`` scores how bad it is to end a line/cue on a given word
(surface tables for ja kana / zh words / en closed-class tokens); for ja,
``ja_pos_end_penalties`` upgrades the signal source to UniDic POS (fugashi)
when available, falling back to the char table otherwise.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence

# Leading-edge prohibition (行頭禁則): these chars cannot begin a line (must hang on the previous line)
LINE_START_PROHIBITED = frozenset(
    "、。，．・：；？！）｝〕〉》」』】〙〗〟"
    "’”»"  # ' " »
    "ァィゥェォッャュョヮ"
    "ぁぃぅぇぉっゃゅょゎ"
    "ーゝゞ々‐゠–〜%"
)
# Trailing-edge prohibition (行末禁則): these chars cannot end a line
LINE_END_PROHIBITED = frozenset(
    "（｛〔〈《「『【〘〖〝‘“«([{"  # ' " «
)

# Surface heuristic (no POS): ending a line on these strands the grammatical relation to
# what follows — a case/adnominal particle binds the preceding noun forward (大樹の|村
# looks broken). High-precision subset only: ambiguous particles that double as conjunctive
# particles (接続助詞) — が adversative / から reason / で connective — are deliberately
# excluded to avoid suppressing real clause breaks.
_BIND_END_HIGH = frozenset(
    "のをにへ"
)  # case/adnominal particles, almost always binds forward
_BIND_END_MED = frozenset(
    "とまでより"
)  # と parallel/quotative, まで/より range: usually binds

# zh equivalents, whole-word semantics (the caller passes the trailing *word*, so 目的/标的
# never match — only the standalone particle/preposition does). Same high-precision policy
# as ja: words with a common clause-final reading are excluded or demoted to MED.
_ZH_BIND_END_HIGH = frozenset(
    {
        "的",  # attributive 的: standalone jieba token is virtually always the particle
        "地",
        "得",  # structural particles; standalone 得 (děi "must") also binds forward
        "把",
        "被",
        "比",
        "跟",  # prepositions: object always follows
        "和",
        "与",
        "或",
        "及",
        "而",  # conjunctions: break goes before them, never after
    }
)
_ZH_BIND_END_MED = frozenset(
    {
        # prepositions with occasional verb readings (他在/他对) — mild penalty only,
        # so they bias len-break tie-breaks but never suppress a danger-zone gap split.
        "在",
        "对",
        "从",
        "向",
        "往",
        "给",
        "让",
        "由",
        # degree adverbs that modify the following word
        "很",
        "太",
        "更",
        "最",
    }
)

# A cue may not begin with an independently tokenised structural/modal particle.
# These are whole-token tables, not character prefixes: ``了解`` and ``地方``
# must remain ordinary words while a lone ``了``/``地`` is kept with its host
# phrase.  Callers therefore pass the first segmented phrase, not arbitrary cue
# text, whenever that information is available.
_ZH_BIND_START_HIGH = frozenset(
    {"的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "嘛", "啦", "呐"}
)
_JA_BIND_START_HIGH = frozenset({"の", "を", "に", "へ"})
_JA_BIND_START_MED = frozenset({"で", "が", "と"})

_ZH_ASPECT_PARTICLES = frozenset({"了", "着", "过"})
_ZH_CATEGORY_WORDS = frozenset(
    {
        "模型",
        "产品",
        "系统",
        "平台",
        "工具",
        "服务",
        "版本",
        "助手",
        "model",
        "product",
        "system",
        "platform",
        "tool",
        "service",
        "version",
        "assistant",
    }
)

_EN_TOKEN_STRIP = ".,!?;:'\"”’"


def line_end_penalty(text: str, lang: str = "") -> int:
    """Penalty for ending a line/cue on ``text`` (the trailing word or phrase).

    0 = fine, 1 = mild (likely binds forward), 2 = bad (function word/particle dangling).

    Signal source by language:
    - ja (and default): last *char* against the kana particle tables — atoms are per-char,
      and a particle is always the final char of its BudouX phrase. Always active: kana
      can't false-positive in other scripts.
    - en: whole token against breakpoints._FORBIDDEN_LEFT (articles/preps/aux/conj).
    - zh/yue: whole word against the Chinese particle/preposition tables.
    """
    s = text.rstrip()
    if not s:
        return 0
    last = s[-1]
    if last in _BIND_END_HIGH:
        return 2
    if last in _BIND_END_MED:
        return 1
    if lang == "en":
        from .breakpoints import _FORBIDDEN_LEFT

        if s.strip(_EN_TOKEN_STRIP).lower() in _FORBIDDEN_LEFT:
            return 2
    elif lang in {"zh", "yue"}:
        if s in _ZH_BIND_END_HIGH:
            return 2
        if s in _ZH_BIND_END_MED:
            return 1
    return 0


def line_start_penalty(text: str, lang: str = "") -> int:
    """Penalty for beginning a cue with the segmented phrase ``text``.

    This is the right-edge counterpart of :func:`line_end_penalty`.  Matching
    is deliberately whole-token for Chinese so lexical words such as ``了解``
    and ``地方`` are not mistaken for the independent particles ``了``/``地``.
    ``2`` means the boundary is strongly undesirable, ``1`` mildly so.
    """

    s = text.lstrip()
    if not s:
        return 0
    if s[0] in LINE_START_PROHIBITED:
        return 2
    if lang in {"zh", "yue"}:
        return 2 if s in _ZH_BIND_START_HIGH else 0
    if lang == "ja":
        if s in _JA_BIND_START_HIGH:
            return 2
        if s in _JA_BIND_START_MED:
            return 1
    return 0


def zh_pos_boundary_penalties(
    atoms: Sequence[str], candidate_indices: Sequence[int], lang: str
) -> dict[int, int]:
    """High-precision Chinese modifier/head damage at candidate boundaries.

    The score is shared by deterministic and model-assisted splitting.  Jieba
    is only a soft optional signal: absence, tokenisation disagreement, or any
    runtime error returns an empty mapping and leaves the surface rules active.
    Atom indices are converted to non-space character offsets so embedded Latin
    runs remain aligned with the subtitle atom stream.
    """

    if lang not in {"zh", "yue"}:
        return {}
    try:
        import jieba.posseg as pseg  # type: ignore
    except (ImportError, ModuleNotFoundError):
        return {}

    def width(surface: str) -> int:
        return sum(not char.isspace() for char in surface)

    text = "".join(atoms)
    previous: dict[int, tuple[str, str]] = {}
    following: dict[int, tuple[str, str]] = {}
    cursor = 0
    try:
        for token in pseg.cut(text, HMM=False):
            surface = str(token.word)
            token_width = width(surface)
            if token_width < 1:
                continue
            start, end = cursor, cursor + token_width
            following[start] = (surface, str(token.flag))
            previous[end] = (surface, str(token.flag))
            cursor = end
    except Exception:  # noqa: BLE001 - optional POS hint must stay fail-safe
        return {}

    atom_offsets = [0]
    for atom in atoms:
        atom_offsets.append(atom_offsets[-1] + width(atom))
    penalties: dict[int, int] = {}
    for boundary in candidate_indices:
        if not 0 < boundary < len(atom_offsets):
            continue
        offset = atom_offsets[boundary]
        left = previous.get(offset)
        right = following.get(offset)
        if left is None or right is None:
            continue
        left_word, left_pos = left
        right_word, right_pos = right
        penalty = 0
        if left_pos == "c":
            penalty += 4
        elif left_pos in {"m", "q", "r"}:
            penalty += 4
        elif left_pos == "p":
            penalty += 5
        elif left_pos in {"d", "f"}:
            penalty += 3
        elif left_pos in {"s", "t"} and right_pos[:1] in {"a", "v"}:
            penalty += 3
        if (
            right_pos.startswith("u")
            or right_pos == "y"
            or right_word in _ZH_BIND_START_HIGH
        ):
            penalty += 4
        left_noun = left_pos.startswith("n") or left_pos in {"eng", "vn"}
        right_noun = right_pos.startswith("n") or right_pos in {"eng", "vn"}
        if left_word == "的" and right_noun:
            penalty += 6
        elif left_word == "地" and right_pos.startswith("v"):
            penalty += 6
        elif left_word == "得" and right_pos[:1] in {"a", "d", "v"}:
            penalty += 5
        elif left_word in _ZH_ASPECT_PARTICLES and (
            right_noun or right_pos[:1] in {"a", "b", "d", "m", "q", "v"}
        ):
            penalty += 6
        if left_pos[:1] in {"a", "b"} and right_noun:
            penalty += 4
        if left_pos in {"f", "s", "t"} and right_noun:
            penalty += 4
        if left_pos == "t" and right_pos == "t":
            penalty += 4
        if left_noun and right_noun:
            penalty += 3
            if left_pos in {"eng", "vn"} or right_pos in {"eng", "vn"}:
                penalty += 3
        if left_pos == "eng" and right_pos in {"f", "s"}:
            penalty += 5
        if left_pos.startswith("v") and right_pos == "p":
            penalty += 3
        if left_pos.startswith("v") and right_pos.startswith("v"):
            penalty += 3
        elif left_pos.startswith("v") and (right_noun or right_pos in {"f", "r"}):
            penalty += 4
        if left_word.casefold() in _ZH_CATEGORY_WORDS and right_pos == "eng":
            penalty += 3
        penalties[boundary] = penalty
    return penalties


@functools.lru_cache(maxsize=1)
def _load_ja_tagger():
    """Lazy fugashi (MeCab + unidic-lite) tagger singleton.

    None when fugashi is absent or env VOXWEAVE_JA_POS=0 forces the Level-1
    char-table fallback (debug/bisection knob).
    """
    if os.environ.get("VOXWEAVE_JA_POS", "").strip() == "0":
        return None
    try:
        from fugashi import Tagger  # type: ignore

        return Tagger()
    except Exception:
        return None


def _pos_penalty(pos1: str, pos2: str) -> int:
    """UniDic POS -> line-end penalty (Level 2 of the same scorer).

    Same intent as the char tables, but disambiguated: 準体助詞の (走るの = a
    legal break) scores 0 where the char table had to penalize every の; and
    POS reaches classes a surface table cannot (連体詞 この/その, 接頭辞 お/各).
    接続助詞 (て/が/から) and 係助詞 (は/も) stay 0 — real clause breaks.
    """
    if pos1 == "助詞":
        if pos2 == "格助詞":
            return 2
        if pos2 == "副助詞":
            return 1
        return 0  # 係助詞 / 接続助詞 / 終助詞 / 準体助詞
    if pos1 in ("連体詞", "接頭辞"):
        return 2  # この|村 / お|名前: always binds forward
    return 0


def ja_pos_end_penalties(text: str) -> dict[int, int] | None:
    """Penalty by non-space char offset of each token's LAST char, or None.

    Offsets count non-space chars only, matching smart_split's atom cursor.
    Only token-end offsets are present: a break after a mid-token char is not
    scored here (callers fall back to the char table), so BudouX/MeCab boundary
    disagreements degrade to Level-1 behavior instead of guessing.
    """
    tagger = _load_ja_tagger()
    if tagger is None:
        return None
    pen: dict[int, int] = {}
    off = 0
    for word in tagger(text):
        n = sum(1 for c in word.surface if not c.isspace())
        if n == 0:
            continue
        off += n
        f = word.feature
        pen[off - 1] = _pos_penalty(
            getattr(f, "pos1", "") or "", getattr(f, "pos2", "") or ""
        )
    return pen


def apply_kinsoku(lines: list[str]) -> list[str]:
    """Nudge breaks pairwise to satisfy kinsoku constraints (oikomi/oidashi, single chars only)."""
    if len(lines) < 2:
        return list(lines)
    out = [list(line) for line in lines]
    for i in range(len(out) - 1):
        left, right = out[i], out[i + 1]
        # 行頭禁則: pull a prohibited leading char back to previous line
        while right and right[0] in LINE_START_PROHIBITED and left:
            left.append(right.pop(0))
        # 行末禁則: push a prohibited trailing char down to next line
        while left and left[-1] in LINE_END_PROHIBITED and right:
            right.insert(0, left.pop())
    return ["".join(c) for c in out if c]
