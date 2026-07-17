"""Timing-only polish over the final cue stream.

Every pass here preserves cue text order and never crosses a real pause:
micro-sentence merging, flicker-cue gluing, duration cleanup (min-dur floor,
CPS linger, tail pad, gap chaining) and shot-change snapping. Runs after the
segmentation engine (``smart_split``) has fixed cue contents; text helpers
come from ``layout``.
"""

from __future__ import annotations

import bisect
from typing import List, cast

from .layout import _fits_budget, _no_spaces, _reading_chars, _visual_len, wrap_cue_text
from .schema import Cue

TWO_FRAME_S = 2.0 / 24.0  # ~0.083s Netflix min inter-cue gap
CHAIN_MAX_GAP_S = 0.5  # gaps below this are "dead zone" -> chain to 2 frames
VISIBLE_GAP_MIN_S = 1.0  # gaps >= this stay a visible pause (BBC); not enforced in code (CHAIN_MAX_GAP_S=0.5 never reaches them)
GLUE_MAX_GAP_S = 0.3  # lone-word flicker cue glues onto its nearer neighbor when that gap is below this
LINGER_CAP_S = 1.0  # CPS-driven extension never lingers more than this past speech end
DEGENERATE_CUE_S = (
    0.08  # a cue this short is a forced-alignment collapse artifact, not real timing
)
HELD_WORD_MAX_GAP_S = 1.0  # a held-word extension may cross word gaps up to this (sung
# sustain with breaths ~0.84s passes); a wider silence past the cap is dead air the
# extension must refuse (stops at the last word before it)
SHORT_FRAGMENT_MAX_CHARS = 12  # generous upper bound for one short lexical word


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
    # A misdetected language can send an unspaced Han paragraph through the
    # space-delimited path.  ``split()`` sees that entire paragraph as one word;
    # retain an explicit size bound so only an actually short token can glue.
    return len(t.split()) == 1 and _visual_len(t, lang) <= SHORT_FRAGMENT_MAX_CHARS


def _gap_between(a: Cue, b: Cue) -> float | None:
    """Inter-cue gap a->b (b.start - a.end), or None if either bound is missing."""
    ae, bs = a.get("end"), b.get("start")
    return (bs - ae) if ae is not None and bs is not None else None


def _merge_micro_cues(
    cues: List[Cue],
    lang: str,
    *,
    max_gap_s: float,
    max_line_length: int,
    max_cue_s: float,
    min_cue_s: float = 0.0,
    max_lines: int = 1,
) -> List[Cue]:
    """Merge adjacent cues separated by sub-glue gaps.

    Two folding rules, both gated by ``max_gap_s`` (0.3s, below ``clause_ms``
    0.4s) so a real pause is never crossed:

    - Ordinary micro-sentence chaining (そう。だね。 / "Yeah." "Right.") merges
      only when the join still fits one display line and the duration cap, so
      len-/gap-/dur-broken pairs stay apart.
    - Degenerate-collapse escape: a forced-alignment failure can pack a run of
      sub-frame cues into a few dozen ms (ASR text overrunning its aligned span
      -> uniform 2-12ms word timestamps). Such flicker is unreadable, so when
      adjacent cues are BOTH below ``DEGENERATE_CUE_S`` (~0.08s), or a run of
      abutting cues collectively spans less than ``min_cue_s``, they are folded
      regardless of the line budget and the merged text is re-wrapped with the
      layout soft-wrap so it renders legally. The escape never fires on real
      cues: a 1.5s+1.5s pair is far above the floor, and a lone 0.3s interjection
      with a real pause fails the gap gate.

    ``max_gap_s<=0`` disables the pass; ``min_cue_s<=0`` disables the run-length
    branch of the escape.
    """
    if max_gap_s <= 0 or len(cues) < 2:
        return cues
    sep = "" if _no_spaces(lang) else " "
    out = [cast(Cue, dict(cues[0]))]
    escaped = [False]  # cue was force-merged over budget -> needs a re-wrap
    for nxt in cues[1:]:
        cur = out[-1]
        gap = _gap_between(cur, nxt)
        merged_text = (cur["text"].rstrip() + sep + nxt["text"].lstrip()).strip()
        cur_start, cur_end = cur.get("start"), cur.get("end")
        nxt_start, nxt_end = nxt.get("start"), nxt.get("end")
        contiguous = gap is not None and gap < max_gap_s
        fits = (
            contiguous
            and cur_start is not None
            and nxt_end is not None
            and nxt_end - cur_start <= max_cue_s
            and _fits_budget(merged_text, max_line_length, 1, lang)
        )
        both_below_floor = (
            cur_start is not None
            and cur_end is not None
            and nxt_start is not None
            and nxt_end is not None
            and (cur_end - cur_start) < DEGENERATE_CUE_S
            and (nxt_end - nxt_start) < DEGENERATE_CUE_S
        )
        run_below_min = (
            min_cue_s > 0
            and cur_start is not None
            and nxt_end is not None
            and (nxt_end - cur_start) < min_cue_s
        )
        degenerate = contiguous and (both_below_floor or run_below_min)
        if fits or degenerate:
            cur["text"] = merged_text
            cur["end"] = nxt["end"] if cur_end is None else max(cur_end, nxt["end"])
            cur["word_data"] = list(cur.get("word_data") or []) + list(
                nxt.get("word_data") or []
            )
            if degenerate and not fits:
                escaped[-1] = True
            continue
        out.append(cast(Cue, dict(nxt)))
        escaped.append(False)
    for cue, was_escaped in zip(out, escaped):
        if was_escaped:  # over-budget forced merge -> re-wrap so it renders legally
            cue["text"] = wrap_cue_text(cue["text"], lang, max_lines)
    return out


def _glue_short_cues(
    cues: List[Cue],
    lang: str,
    *,
    max_gap_s: float,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
) -> List[Cue]:
    """Glue a super-short single-word flicker cue onto whichever neighbor abuts it
    closest, when that gap is below ``max_gap_s`` — contiguous speech means a
    spurious split, not a real pause.

    Bidirectional: an interjection that *leads* the next line (え/ん with a sub-0.3s
    gap ahead but a real pause behind) glues forward; a tail fragment glues back.
    The side with the smaller gap wins (ties go backward). Safe re-introduction of
    the deleted merge_short_cues: ``max_gap_s`` (0.3s) sits below ``clause_ms``
    (0.4s), so the real-pause side (>=0.4s) is never crossed and a cue is never
    dragged over silence. A merge must also fit the configured display budget
    and duration cap, so glue cannot undo a length- or duration-driven split.
    ``max_gap_s<=0`` disables.
    """
    if max_gap_s <= 0 or len(cues) < 2:
        return cues
    sep = "" if _no_spaces(lang) else " "
    work = [cast(Cue, dict(c)) for c in cues]
    out: List[Cue] = []
    i, n = 0, len(work)
    while i < n:
        c = work[i]
        nxt = work[i + 1] if i + 1 < n else None
        if _is_short_fragment(c["text"], lang):
            gap_back = _gap_between(out[-1], c) if out else None
            gap_fwd = _gap_between(c, nxt) if nxt is not None else None
            back_text = (
                (out[-1]["text"].rstrip() + sep + c["text"].lstrip()).strip()
                if out
                else ""
            )
            fwd_text = (
                (c["text"].rstrip() + sep + nxt["text"].lstrip()).strip()
                if nxt is not None
                else ""
            )
            back_start = out[-1].get("start") if out else None
            back_end = c.get("end")
            fwd_start = c.get("start")
            fwd_end = nxt.get("end") if nxt is not None else None
            back_ok = (
                gap_back is not None
                and gap_back < max_gap_s
                and back_start is not None
                and back_end is not None
                and back_end - back_start <= max_cue_s
                and _fits_budget(back_text, max_line_length, max_lines, lang)
            )
            fwd_ok = (
                gap_fwd is not None
                and gap_fwd < max_gap_s
                and fwd_start is not None
                and fwd_end is not None
                and fwd_end - fwd_start <= max_cue_s
                and _fits_budget(fwd_text, max_line_length, max_lines, lang)
            )
            # nearer side wins; ties go backward ("append to last cue").
            go_fwd = fwd_ok and (
                not back_ok or (gap_fwd is not None and gap_fwd < gap_back)  # type: ignore[operator]
            )
            if go_fwd and nxt is not None:  # prepend fragment into next, reprocess it
                nxt["text"] = fwd_text
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
                prev["text"] = back_text
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
    cues: List[Cue],
    *,
    min_cue_s: float,
    max_cue_s: float,
    cps: float = 0.0,
    lag_out_s: float = 0.0,
) -> List[Cue]:
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
    out = [cast(Cue, dict(c)) for c in cues]
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
            if nxt_start is None:
                c["end"] = want
            elif nxt_start - c["end"] > TWO_FRAME_S:
                c["end"] = min(want, nxt_start)
            # else: the inter-cue gap is already at/under the 2-frame floor —
            # leave it untouched. This keeps _cleanup_cues idempotent (the
            # diarize path runs it twice): without the guard a second lag-out
            # pass would extend over a chained 2-frame gap and collapse it to
            # zero, which the chaining branch then cannot restore. Gaps wider
            # than the floor still lag-out exactly as before.
        # chaining: close small inter-cue gaps to 2 frames
        if nxt_start is not None:
            gap = nxt_start - c["end"]
            if 0 <= gap < CHAIN_MAX_GAP_S and gap > TWO_FRAME_S:
                c["end"] = nxt_start - TWO_FRAME_S
            # overlaps (gap<0) and large gaps (>=CHAIN_MAX_GAP_S) left to caller
        # never let extension / chaining push a cue past the duration cap, but
        # never truncate a subtitle while its own words are still sounding: a
        # held/sung word whose word_data end runs past the cap keeps its cue up
        # to that end (capped by the next cue's start). Ordinary long-linger cues
        # whose words already stopped still clamp exactly to start+max_cue_s.
        if max_cue_s and c["end"] - c["start"] > max_cue_s:
            cap = c["start"] + max_cue_s
            timed = sorted(
                (
                    (s, e)
                    for w in c.get("word_data") or []
                    if (s := w.get("start")) is not None
                    and (e := w.get("end")) is not None
                ),
                key=lambda unit: unit[0],
            )
            last_word_end = max((e for _s, e in timed), default=None)
            if last_word_end is not None and last_word_end > cap:
                # A still-sounding word may hold the cue past the cap, but only
                # across CONTINUOUS speech. Walk the words in time order and stop
                # at the first silence gap wider than HELD_WORD_MAX_GAP_S that
                # lands at or past the cap: the extension target is the last word
                # end before that gap. This keeps a sung sustain (breaths ~0.84s)
                # visible while refusing to drag the cue across dead air to a
                # stray final syllable (real ja case: words end ~2s past the cap,
                # then a 3.7s gap, then one 80ms syllable). If the held region
                # ends before the cap, max() clamps back to plain cap behavior.
                held_end = timed[0][1]
                for (_ps, pe), (ns, ne) in zip(timed, timed[1:]):
                    if ns - pe > HELD_WORD_MAX_GAP_S and ns >= cap:
                        break
                    held_end = ne
                target = held_end
                if nxt_start is not None:
                    target = min(target, nxt_start)
                c["end"] = max(cap, target)
            else:
                c["end"] = cap
    return out


# Netflix TTSG shot-change zones, specified in frames at 24 fps. "Half a second"
# (12 frames) is the landing zone a boundary is pushed out to when it cannot sit
# on the cut itself.
_FRAME_S = 1.0 / 24.0
_SHOT_LANDING_S = 12 * _FRAME_S
_EPS = 1e-9


def _snap_to_shots(
    cues: List[Cue],
    shots: List[float],
    *,
    snap_s: float,
    max_cue_s: float,
) -> List[Cue]:
    """Adjust cue boundaries near shot changes per the Netflix TTSG zone rules
    (runs after _cleanup_cues). ``snap_s`` is the search window for pairing a
    boundary with a cut (<=0 disables snapping entirely).

    In-times (asymmetric zones, 24 fps frames):
    - 1-7 frames before the cut: move onto the cut (removes the pre-cut flash).
    - 8-11 frames before: pull out to 12 frames before (free lead-in).
    - 1-9 frames after: move back onto the cut (text appears on the cut).
    - 10-11 frames after: push out to 12 frames after.

    Out-times:
    - up to 12 frames before the cut: extend to cut - 2 frames (die on the cut).
    - 1-5 frames after: pull back to cut - 2 frames.
    - 6-11 frames after: extend out to 12 frames after.

    No move ever sacrifices speech: starts stay clear of the previous cue end
    + 2 frames and below the cue's own end; ends never pull below the last
    word's end (dialogue that crosses the cut keeps its subtitle across it,
    falling back to the 12-frames-after landing zone), never collide with the
    next cue, and respect the duration cap.
    """
    if snap_s <= 0 or not shots:
        return cues
    out = [cast(Cue, dict(c)) for c in cues]

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
        ends = [e for w in c.get("word_data") or [] if (e := w.get("end")) is not None]
        speech_end = max(ends, default=end)
        prev_end = out[i - 1].get("end") if i > 0 else None
        nxt_start = out[i + 1].get("start") if i + 1 < len(out) else None

        cut = nearest(start)
        if cut is not None and abs(cut - start) > _EPS:
            d = start - cut
            if -7 * _FRAME_S - _EPS <= d < 0:  # 1-7 frames before -> onto the cut
                new_start = cut
            elif d < 0:  # 8-11 frames before -> out to 12 frames before
                new_start = cut - _SHOT_LANDING_S
            elif d <= 9 * _FRAME_S + _EPS:  # 1-9 frames after -> back onto the cut
                new_start = cut
            else:  # 10-11 frames after -> out to 12 frames after
                new_start = cut + _SHOT_LANDING_S
            if prev_end is not None:
                new_start = max(new_start, prev_end + TWO_FRAME_S)
            # moving earlier is a free lead-in; moving later (flash removal /
            # landing-zone push) must never delay the text by over half a second
            if new_start < end - TWO_FRAME_S and (
                new_start <= start or new_start - start <= _SHOT_LANDING_S
            ):
                c["start"] = new_start

        cut = nearest(end)
        if cut is not None:
            d = end - cut
            if d <= 5 * _FRAME_S + _EPS:  # up to 12 before / 1-5 after -> die on cut
                target = cut - TWO_FRAME_S
            else:  # 6-11 frames after -> out to the 12-frames-after landing zone
                target = cut + _SHOT_LANDING_S
            applied = False
            if target > end + _EPS:  # extend into the following gap (free)
                if (
                    nxt_start is None or target <= nxt_start - TWO_FRAME_S
                ) and target - c["start"] <= max_cue_s:
                    c["end"] = target
                    applied = True
            elif target < end - _EPS:  # pull back, never cutting speech
                if target >= speech_end and target > c["start"]:
                    c["end"] = target
                    applied = True
            # speech crosses the cut so the pull-back was vetoed: the subtitle
            # legitimately crosses; land 12 frames after instead of flashing out
            # just past the cut (TTSG last resort)
            if not applied and 0 < d <= 5 * _FRAME_S + _EPS:
                target = cut + _SHOT_LANDING_S
                if (
                    target > end
                    and (nxt_start is None or target <= nxt_start - TWO_FRAME_S)
                    and target - c["start"] <= max_cue_s
                ):
                    c["end"] = target
    return out
