"""Pre-segmentation repair of aligner artifacts in the unit stream.

The MMS/CTC lane can park the tail character of a word on a much later speech
island: the aligner assigns every character somewhere, and a word-final vowel
sometimes lands seconds past the word's body, across genuine dead air (the
"stranded tail" signature). Downstream, the segmentation engine cannot break
inside a word (by design), so the stranded timestamp inflates whatever cue owns
the word — up to over-length cues whose speech ended seconds earlier.

The repair runs on a *copy* of the stream that only feeds cue formation. The
persisted ``word_segments`` keep the raw aligner output: the sibling JSON is
alignment evidence, and every replay re-derives the same repair
deterministically from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from voxweave.core.breakpoints import phrase_atoms
from voxweave.core.langsets import LANGUAGES_WITHOUT_SPACES
from voxweave.core.layout import _token_char_count

#: A word-internal gap wider than this is never phrasing — no-space languages
#: place phrase boundaries between words, and the aligner assigns voiced frames
#: monotonically, so a genuine held syllable spans *continuously*.
STRANDED_TAIL_GAP_S = 2.0

#: The gap must contain at least this much continuous VAD-confirmed silence:
#: a stranded tail sits across dead air, an ASR under-segmentation does not.
SILENT_RUN_MIN_S = 1.0


def _max_silent_run(a: float, b: float, speech: Sequence[tuple[float, float]]) -> float:
    """Longest continuous stretch of [a, b] not covered by any speech span."""
    if b <= a:
        return 0.0
    events = sorted((max(a, s), min(b, e)) for s, e in speech if e > a and s < b)
    best, cur = 0.0, a
    for s, e in events:
        if s > cur:
            best = max(best, s - cur)
        cur = max(cur, e)
    return max(best, b - cur)


def repair_stranded_tails(
    units: Sequence[dict[str, Any]],
    lang: str,
    speech_spans: Sequence[tuple[float, float]] | None,
) -> list[dict[str, Any]]:
    """Zero out the timing of word-tail units stranded across dead air.

    A unit is repaired only when every gate holds:

    1. no-space language, and the unit is *not* a phrase start (a phrase start
       after a pause is a legitimate new utterance, not a tail);
    2. the gap to the previous unit exceeds ``STRANDED_TAIL_GAP_S``;
    3. the gap contains at least ``SILENT_RUN_MIN_S`` of continuous
       VAD-confirmed silence;
    4. isolation: the VAD island hosting the unit contains no unit of a later
       phrase — a stray tail is alone in its island, while a real utterance
       onset chains into following speech (without this gate, a tail sitting
       at a genuine speech onset would drag that onset's words backward).

    A repaired unit collapses to ``start == end == previous.end``, the same
    zero-width convention the aligner's own punctuation units use. Without
    ``speech_spans`` nothing is repaired (the silence evidence is missing).
    Returns a new list; input dicts are never mutated.
    """
    if lang not in LANGUAGES_WITHOUT_SPACES or len(units) < 2 or not speech_spans:
        return list(units)

    text = "".join(str(u.get("text", u.get("word", ""))) for u in units)
    starts: set[int] = set()
    cursor = 0
    for phrase in phrase_atoms(text, lang):
        starts.add(cursor)
        cursor += _token_char_count(phrase)

    positions: list[int] = []
    cursor = 0
    for u in units:
        positions.append(cursor)
        cursor += _token_char_count(str(u.get("text", u.get("word", ""))))
    sorted_starts = sorted(starts)

    def phrase_index(char_pos: int) -> int:
        lo, hi, best = 0, len(sorted_starts) - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_starts[mid] <= char_pos:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return best

    def vad_island_of(t: float) -> tuple[float, float] | None:
        for s, e in speech_spans:
            if s <= t <= e:
                return (s, e)
        return None

    out = [dict(u) for u in units]
    for i in range(1, len(out)):
        if positions[i] in starts:
            continue
        u, prev = out[i], out[i - 1]
        us, pe = u.get("start"), prev.get("end")
        if us is None or pe is None:
            continue
        gap = float(us) - float(pe)
        if gap <= STRANDED_TAIL_GAP_S:
            continue
        if _max_silent_run(float(pe), float(us), speech_spans) < SILENT_RUN_MIN_S:
            continue
        island = vad_island_of(float(us))
        if island is not None:
            my_phrase = phrase_index(positions[i])
            hosts_later_phrase = False
            for j in range(i + 1, len(out)):
                other_start = out[j].get("start")
                if other_start is None:
                    continue
                if (
                    island[0] <= float(other_start) <= island[1]
                    and phrase_index(positions[j]) > my_phrase
                ):
                    hosts_later_phrase = True
                    break
            if hosts_later_phrase:
                continue
        u["start"] = u["end"] = float(pe)
    return out
