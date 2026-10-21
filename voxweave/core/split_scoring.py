"""Break-point quality for the segmentation engine.

Per-atom end and start penalties, POS penalty attachment, the pause and
micro-fragment damage ladders, the choice of a length break and the
gap/force classification of an atom boundary. The packing and bound-particle
repair passes consume these scores.
"""

from __future__ import annotations

from collections.abc import Sequence

from .gap_split import gap_qualifies
from .kinsoku import line_end_penalty, line_start_penalty, zh_pos_boundary_penalties
from .layout import _fits_budget, _join, _token_char_count, _vis_width
from .split_atoms import _span_start
from .split_types import SplitContext


FORCE_BREAK_FACTOR = 1.5  # boundary-less run may exceed the line budget by at most this before a forced cut


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
