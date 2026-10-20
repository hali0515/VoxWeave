"""Span arithmetic and audition clip selection for diarized speakers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from voxweave.songdet import subtract_spans


MIN_SNIPPET_S = 2.0
MAX_SNIPPET_S = 6.0
MAX_SNIPPETS_PER_SPEAKER = 3
MIN_SNIPPET_GAP_S = 1.0

Span = tuple[float, float]
Turn = tuple[float, float, str]


def _valid_spans(spans: Sequence[Span] | None) -> list[Span]:
    """Return finite, positive spans in chronological order."""
    out: list[Span] = []
    for start, end in spans or ():
        a, b = float(start), float(end)
        if a < b and math.isfinite(a) and math.isfinite(b):
            out.append((a, b))
    return sorted(out)


def _merge_spans(spans: Sequence[Span] | None) -> list[Span]:
    """Union overlapping or touching spans."""
    merged: list[Span] = []
    for start, end in _valid_spans(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersect_spans(left: Sequence[Span], right: Sequence[Span]) -> list[Span]:
    """Intersection of two span unions."""
    a = _merge_spans(left)
    b = _merge_spans(right)
    out: list[Span] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract_spans(source: Sequence[Span], removed: Sequence[Span]) -> list[Span]:
    """Subtract the union of ``removed`` from the union of ``source``.

    songdet.subtract_spans owns the sweep and requires both operands already
    sorted and disjoint, so normalize each side through _merge_spans first.
    """
    return subtract_spans(_merge_spans(source), _merge_spans(removed))


def _window(span: Span, target: float | None = None) -> Span:
    """Make a 2-6 second audition window inside ``span`` near ``target``."""
    start, end = span
    length = min(MAX_SNIPPET_S, end - start)
    center = (start + end) / 2.0 if target is None else target
    clip_start = min(max(center - length / 2.0, start), end - length)
    return (clip_start, clip_start + length)


def _pick_spread(regions: Sequence[Span], limit: int) -> list[Span]:
    """Pick long clean windows near the early, middle, and late extent.

    A single 6-18 second run yields two longest-possible windows with at least
    one second between them. Otherwise the minimum gap applies only to cuts
    from the same continuous run; separately voiced runs remain eligible even
    when their natural pause is shorter.
    """
    clean = [
        span for span in _merge_spans(regions) if span[1] - span[0] >= MIN_SNIPPET_S
    ]
    if not clean or limit <= 0:
        return []
    if len(clean) == 1 and limit >= 2:
        start, end = clean[0]
        run_length = end - start
        if 6.0 <= run_length <= 18.0:
            clip_length = min(MAX_SNIPPET_S, (run_length - MIN_SNIPPET_GAP_S) / 2.0)
            return [
                (start, start + clip_length),
                (end - clip_length, end),
            ]
    extent_start, extent_end = clean[0][0], clean[-1][1]
    bin_count = min(limit, MAX_SNIPPETS_PER_SPEAKER)
    bin_width = (extent_end - extent_start) / bin_count
    bins = [
        (extent_start + bin_width * index, extent_start + bin_width * (index + 1))
        for index in range(bin_count)
    ]
    picked: list[tuple[Span, int]] = []

    def separated(candidate: Span, source_index: int) -> bool:
        for (start, end), picked_source in picked:
            if source_index != picked_source:
                continue
            if not (
                candidate[1] + MIN_SNIPPET_GAP_S <= start
                or end + MIN_SNIPPET_GAP_S <= candidate[0]
            ):
                return False
        return True

    for bin_start, bin_end in bins:
        target = (bin_start + bin_end) / 2.0
        candidates = [
            (_window(span, target), source_index)
            for source_index, span in enumerate(clean)
            if span[0] < bin_end and span[1] > bin_start
        ]
        candidates = [
            (span, source_index)
            for span, source_index in candidates
            if separated(span, source_index)
        ]
        if not candidates:
            continue
        best, source_index = max(
            candidates,
            key=lambda item: (
                item[0][1] - item[0][0],
                -abs((item[0][0] + item[0][1]) / 2.0 - target),
                -item[0][0],
            ),
        )
        picked.append((best, source_index))
        if len(picked) == limit:
            return sorted(span for span, _source in picked)

    # A narrow or empty time bin can leave another usable window unselected.
    # Fill with centered whole-span windows, retaining the same-run gap above.
    while len(picked) < limit:
        candidates = [
            (_window(span), source_index) for source_index, span in enumerate(clean)
        ]
        candidates = [
            (span, source_index)
            for span, source_index in candidates
            if separated(span, source_index)
        ]
        if not candidates:
            break
        best, source_index = max(
            candidates,
            key=lambda item: (item[0][1] - item[0][0], -item[0][0]),
        )
        picked.append((best, source_index))
    return sorted(span for span, _source in picked)


def select_snippets(
    turns: Sequence[Turn],
    vad_speech: Sequence[Span] | None,
    sing_spans: Sequence[Span] | None = None,
    *,
    max_per_speaker: int = MAX_SNIPPETS_PER_SPEAKER,
) -> dict[str, list[Span]]:
    """Select clean audition spans for every diarized speaker.

    A candidate must be inside that speaker's turn, outside every other
    speaker's turn, inside VAD speech, and outside detected singing.  Returned
    clips are always 2-6 seconds and are ordered chronologically.
    """
    labels = list(dict.fromkeys(str(label) for _, _, label in turns))
    by_speaker: dict[str, list[Span]] = {label: [] for label in labels}
    for start, end, label in turns:
        if float(end) > float(start):
            by_speaker[str(label)].append((float(start), float(end)))

    voiced = _merge_spans(vad_speech)
    singing = _merge_spans(sing_spans)
    selected: dict[str, list[Span]] = {}
    for label in labels:
        other = [
            (float(start), float(end))
            for start, end, other_label in turns
            if str(other_label) != label and float(end) > float(start)
        ]
        exclusive = _subtract_spans(by_speaker[label], other)
        clean = _subtract_spans(_intersect_spans(exclusive, voiced), singing)
        selected[label] = _pick_spread(clean, max_per_speaker)
    return selected
