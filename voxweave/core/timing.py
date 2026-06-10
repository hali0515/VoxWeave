"""Timing-only polish over the final cue stream.

Every pass here preserves cue text order and never crosses a real pause:
micro-sentence merging, flicker-cue gluing, duration cleanup (min-dur floor,
CPS linger, tail pad, gap chaining) and shot-change snapping. Runs after the
segmentation engine (``smart_split``) has fixed cue contents; text helpers
come from ``layout``.
"""

from __future__ import annotations

import bisect
from typing import Any, Dict, List

from .layout import _fits_budget, _no_spaces, _reading_chars, _visual_len

TWO_FRAME_S = 2.0 / 24.0  # ~0.083s Netflix min inter-cue gap
CHAIN_MAX_GAP_S = 0.5  # gaps below this are "dead zone" -> chain to 2 frames
VISIBLE_GAP_MIN_S = 1.0  # gaps >= this stay a visible pause (BBC); not enforced in code (CHAIN_MAX_GAP_S=0.5 never reaches them)
GLUE_MAX_GAP_S = 0.3  # lone-word flicker cue glues onto its nearer neighbor when that gap is below this
LINGER_CAP_S = 1.0  # CPS-driven extension never lingers more than this past speech end


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
                    nxt_start is None or target <= nxt_start - TWO_FRAME_S
                ) and target - c["start"] <= max_cue_s:
                    c["end"] = target
            elif target < end:  # pull back, never cutting speech
                if target >= speech_end and target > c["start"]:
                    c["end"] = target
    return out
