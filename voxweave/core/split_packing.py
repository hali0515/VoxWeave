"""Stage 2 of the segmentation engine: pack atoms into cues.

Hard wrap of surfaces that cannot be split at an atom boundary, the
oversized-atom fallback, the greedy packing loop, the chunk to ``Cue``
conversion and the spread of untimed text over a time window.
"""

from __future__ import annotations

import math
from typing import cast

from .layout import _join, _line_budget_width, _vis_width
from .schema import Cue
from .split_atoms import _span_end, _span_start
from .split_scoring import _best_len_break_pos, _classify_atom_break
from .split_types import SplitContext


# Display floor for a cue built from text the aligner left untimed, used when the
# profile's own min_cue_s is off.
UNTIMED_MIN_DISPLAY_S = 0.5


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
