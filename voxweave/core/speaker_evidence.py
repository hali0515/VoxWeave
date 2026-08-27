"""Parent-grounded speaker evidence for the P5 shadow optimizer.

Speaker turns are acoustic evidence.  Refinement times are not.  This module
therefore attributes and conditions the production parent stream first, then
projects the result through the caller's complete positional ``origin`` tuple.
It also owns the one EvidenceSpan/lyric predicate, the soft edge term, the
post-finalizer speaker-id projection, and raw-event lineage measurement.

Nothing here imports or calls the live pipeline.  W4 supplies the row wiring;
the functions in this module are deterministic standalone producers.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from .boundary_cost import CostBreakdown, make_breakdown, transition_time
from .layout import _join, _no_spaces
from .schema import Cue
from .segdoc import SegDocument, SourceUnit

__all__ = [
    "BUCKET_KINDS",
    "ENDPOINT_KINDS",
    "EXPRESS_TOL_S",
    "SPEAKER_EDGE_RUN_MIN_S",
    "SPEAKER_MIN_RUN_S",
    "SPEAKER_MULTI_MIN_FRAC",
    "SPEAKER_UNIT_COVER_FRAC",
    "TURN_STATES",
    "W_SPEAKER_INTERIOR",
    "BoundaryPoint",
    "EventBoundaryMatch",
    "EvidenceSpan",
    "LiveSpeakerEvent",
    "ParentSpeaker",
    "RawSpeakerEvent",
    "SpeakerConditioningStats",
    "SpeakerEvidenceError",
    "SpeakerMeasurement",
    "SpeakerPricingSummary",
    "SpeakerProjectionError",
    "UnitSpeaker",
    "UnitSpeakers",
    "annotate_speaker_ids",
    "evidence_span_from_cue",
    "injective_time_match",
    "lyric_for_evidence",
    "make_evidence_span",
    "measure_speaker_events",
    "named_multi_cues_unannotated",
    "speaker_edge_cost",
    "speaker_evidence",
    "summarize_speaker_prices",
]


SPEAKER_UNIT_COVER_FRAC: float = 0.5
SPEAKER_MULTI_MIN_FRAC: float = 0.25
SPEAKER_MIN_RUN_S: float = 0.2
SPEAKER_EDGE_RUN_MIN_S: float = 0.12
W_SPEAKER_INTERIOR: float = 3.0
EXPRESS_TOL_S: float = 0.5

ENDPOINT_KINDS: tuple[str, ...] = ("exact", "fabricated")
TURN_STATES: tuple[str, ...] = (
    "absent",
    "overlap",
    "multi",
    "single",
    "unattributed",
)
BUCKET_KINDS: tuple[str, ...] = (
    "expressed",
    "policy_filtered",
    "survived_expressible_but_missed",
    "unattributed_loss",
    "unexpressible",
)

EndpointKind = Literal["exact", "fabricated"]
ParentKind = Literal["multi", "none", "single"]
UnitKind = Literal["ambiguous", "multi", "none", "single"]
BucketKind = Literal[
    "expressed",
    "policy_filtered",
    "survived_expressible_but_missed",
    "unattributed_loss",
    "unexpressible",
]
Turn = tuple[float, float, str]


class SpeakerEvidenceError(ValueError):
    """Speaker evidence or ownership is not a valid deterministic input."""


class SpeakerProjectionError(ValueError):
    """Selected cue ranges do not exactly partition the attributed unit stream."""


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _positive_duration(unit: SourceUnit) -> float:
    if not _finite(unit.start) or not _finite(unit.end):
        return 0.0
    start, end = float(cast("float", unit.start)), float(cast("float", unit.end))
    return max(0.0, end - start)


@dataclass(frozen=True)
class EvidenceSpan:
    """Stable acoustic-or-input bounds used by every lyric decision."""

    start: float
    end: float
    start_kind: EndpointKind
    end_kind: EndpointKind

    def __post_init__(self) -> None:
        if self.start_kind not in ENDPOINT_KINDS or self.end_kind not in ENDPOINT_KINDS:
            raise SpeakerEvidenceError("evidence span endpoint kind is not closed")
        if not _finite(self.start) or not _finite(self.end):
            raise SpeakerEvidenceError("evidence span bounds must be finite")
        start, end = float(self.start), float(self.end)
        if end < start:
            raise SpeakerEvidenceError("evidence span is reversed")
        object.__setattr__(self, "start", 0.0 if start == 0.0 else start)
        object.__setattr__(self, "end", 0.0 if end == 0.0 else end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end,
            "end_kind": self.end_kind,
            "start": self.start,
            "start_kind": self.start_kind,
        }


def make_evidence_span(
    units: Sequence[SourceUnit],
    unit_range: tuple[int, int],
    *,
    input_start: float,
    input_end: float,
) -> EvidenceSpan:
    """Build the one per-candidate EvidenceSpan from selected unit ownership.

    Each endpoint is independent.  An aligner-provenance endpoint with a finite
    bound is exact; every other endpoint uses the phase-1 input bound.  No
    parent envelope and no interior unit can substitute for an invalid endpoint.
    """
    low, high = _checked_unit_range(unit_range, len(units))
    first, last = units[low], units[high - 1]
    exact_start = first.provenance == "aligner" and _finite(first.start)
    exact_end = last.provenance == "aligner" and _finite(last.end)
    return EvidenceSpan(
        float(cast("float", first.start)) if exact_start else input_start,
        float(cast("float", last.end)) if exact_end else input_end,
        "exact" if exact_start else "fabricated",
        "exact" if exact_end else "fabricated",
    )


def evidence_span_from_cue(cue: Mapping[str, Any]) -> EvidenceSpan:
    """Build the same typed span for a retained v1 reference cue.

    The cue's speech keys have already passed the provenance fold.  Missing or
    non-finite keys independently fall back to the input display bound.
    """
    input_start, input_end = cue.get("start"), cue.get("end")
    if not _finite(input_start) or not _finite(input_end):
        raise SpeakerEvidenceError("cue input bounds must be finite")
    speech_start, speech_end = cue.get("speech_start"), cue.get("speech_end")
    exact_start = _finite(speech_start)
    exact_end = _finite(speech_end)
    input_start_number = float(cast("float", input_start))
    input_end_number = float(cast("float", input_end))
    return EvidenceSpan(
        float(cast("float", speech_start)) if exact_start else input_start_number,
        float(cast("float", speech_end)) if exact_end else input_end_number,
        "exact" if exact_start else "fabricated",
        "exact" if exact_end else "fabricated",
    )


def lyric_for_evidence(
    evidence_span: EvidenceSpan,
    sing_spans: Sequence[tuple[float, float]] | None,
) -> bool:
    """The shared cost/delivery lyric predicate (set and clear)."""
    duration = evidence_span.end - evidence_span.start
    if duration <= 0 or not sing_spans:
        return False
    overlap = 0.0
    for raw_start, raw_end in sing_spans:
        if not _finite(raw_start) or not _finite(raw_end):
            continue
        start, end = float(raw_start), float(raw_end)
        overlap += max(
            0.0,
            min(evidence_span.end, end) - max(evidence_span.start, start),
        )
    return overlap / duration >= 0.5


@dataclass(frozen=True)
class ParentSpeaker:
    """Raw, total classification of one production parent unit."""

    kind: ParentKind
    label: str | None = None
    support: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "single":
            if not self.label or self.support != (self.label,):
                raise SpeakerEvidenceError("single parent must carry exactly its label")
        elif self.kind == "multi":
            if self.label is not None or len(self.support) < 2:
                raise SpeakerEvidenceError(
                    "multi parent must carry multi-label support"
                )
        elif self.kind == "none":
            if self.label is not None or self.support:
                raise SpeakerEvidenceError("none parent cannot carry speaker support")
        else:
            raise SpeakerEvidenceError(f"unknown parent speaker kind: {self.kind!r}")


@dataclass(frozen=True)
class UnitSpeaker:
    """Conditioned speaker state aligned to one optimizer-space unit."""

    kind: UnitKind
    label: str | None
    support: tuple[str, ...]
    parent_index: int

    def __post_init__(self) -> None:
        if isinstance(self.parent_index, bool) or self.parent_index < 0:
            raise SpeakerEvidenceError("unit speaker parent index is invalid")
        if self.kind == "single":
            if not self.label or self.support != (self.label,):
                raise SpeakerEvidenceError("single unit speaker must carry one label")
        elif self.kind in {"ambiguous", "multi"}:
            if self.label is not None or len(self.support) < 2:
                raise SpeakerEvidenceError(
                    "ambiguous unit must carry multi-label support"
                )
        elif self.kind == "none":
            if self.label is not None or self.support:
                raise SpeakerEvidenceError("none unit speaker cannot carry support")
        else:
            raise SpeakerEvidenceError(f"unknown unit speaker kind: {self.kind!r}")


@dataclass(frozen=True)
class SpeakerConditioningStats:
    units_attributed: int
    units_none: int
    filled: int
    phrase_snaps: int
    runs_absorbed: int
    transitions_before: int
    transitions_after: int
    multilabel_parents: int
    unexpressible_turn_changes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "filled": self.filled,
            "multilabel_parents": self.multilabel_parents,
            "phrase_snaps": self.phrase_snaps,
            "runs_absorbed": self.runs_absorbed,
            "transitions_after": self.transitions_after,
            "transitions_before": self.transitions_before,
            "unexpressible_turn_changes": self.unexpressible_turn_changes,
            "units_attributed": self.units_attributed,
            "units_none": self.units_none,
        }


@dataclass(frozen=True)
class RawSpeakerEvent:
    event_id: str
    index: int
    time: float
    left_label: str
    right_label: str


@dataclass(frozen=True)
class LiveSpeakerEvent:
    event_id: str
    index: int
    time: float

    def __post_init__(self) -> None:
        if not self.event_id or isinstance(self.index, bool) or self.index < 0:
            raise SpeakerEvidenceError("live speaker event identity is invalid")
        if not _finite(self.time):
            raise SpeakerEvidenceError("live speaker event time must be finite")


@dataclass(frozen=True)
class BoundaryPoint:
    boundary_id: str
    index: int
    time: float

    def __post_init__(self) -> None:
        if not self.boundary_id or isinstance(self.index, bool) or self.index < 0:
            raise SpeakerEvidenceError("boundary identity is invalid")
        if not _finite(self.time):
            raise SpeakerEvidenceError("boundary time must be finite")


@dataclass(frozen=True)
class EventBoundaryMatch:
    event_id: str
    boundary_id: str
    event_time: float
    boundary_time: float
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "boundary_time": self.boundary_time,
            "distance": self.distance,
            "event_id": self.event_id,
            "event_time": self.event_time,
        }


@dataclass(frozen=True)
class _LineageTransition:
    boundary: int
    time: float | None
    ancestry: frozenset[str] = frozenset()


@dataclass(frozen=True)
class UnitSpeakers:
    """Parent classifications, conditioned projection, and raw-event lineage."""

    language: str
    parent_units: tuple[SourceUnit, ...]
    refined_units: tuple[SourceUnit, ...]
    origin: tuple[int, ...]
    turn_track_present: bool
    turns: tuple[Turn, ...]
    parent_speakers: tuple[ParentSpeaker, ...]
    conditioned_parents: tuple[UnitSpeaker, ...]
    unit_speakers: tuple[UnitSpeaker, ...]
    edge_silence_s: float
    stats: SpeakerConditioningStats
    raw_events: tuple[RawSpeakerEvent, ...]
    in_speech_event_ids: tuple[str, ...]
    base_event_buckets: tuple[tuple[str, BucketKind], ...]
    live_events: tuple[LiveSpeakerEvent, ...]

    @property
    def labels(self) -> tuple[str | None, ...]:
        return tuple(
            item.label if item.kind == "single" else None for item in self.unit_speakers
        )

    def matches_document_track(self, document: SegDocument) -> bool:
        """Whether this projection was conditioned from the document's track."""
        return (
            self.language == document.language
            and self.turn_track_present == (document.speaker_turns is not None)
            and self.turns == _checked_turns(document.speaker_turns or ())
            and self.edge_silence_s == (document.profile.vad_skip_ms + 50.0) / 1000.0
        )

    def to_dict(
        self,
        *,
        pricing: SpeakerPricingSummary | None = None,
        measurement: SpeakerMeasurement | None = None,
        speaker_weight: float = W_SPEAKER_INTERIOR,
    ) -> dict[str, Any]:
        return {
            "attribution": "parent-projected",
            "conditioning": self.stats.to_dict(),
            "constants": {
                "speaker_edge_run_min_s": SPEAKER_EDGE_RUN_MIN_S,
                "speaker_edge_silence_s": self.edge_silence_s,
                "speaker_min_run_s": SPEAKER_MIN_RUN_S,
                "speaker_multi_min_frac": SPEAKER_MULTI_MIN_FRAC,
                "speaker_unit_cover_frac": SPEAKER_UNIT_COVER_FRAC,
                "speaker_weight_interior": speaker_weight,
            },
            "measurement": None if measurement is None else measurement.to_dict(),
            "parent_count": len(self.parent_units),
            "pricing": None if pricing is None else pricing.to_dict(),
            "raw_turn_change_count": len(self.raw_events),
            "refined_unit_count": len(self.refined_units),
            "turn_track_present": self.turn_track_present,
        }


def _checked_turns(turns: Sequence[Turn]) -> tuple[Turn, ...]:
    checked: list[tuple[float, float, str, int]] = []
    for index, entry in enumerate(turns):
        try:
            raw_start, raw_end, raw_label = entry
        except (TypeError, ValueError) as exc:
            raise SpeakerEvidenceError("speaker turn must have three fields") from exc
        if not _finite(raw_start) or not _finite(raw_end):
            raise SpeakerEvidenceError("speaker turn bounds must be finite")
        start, end = float(raw_start), float(raw_end)
        if end <= start:
            raise SpeakerEvidenceError("speaker turn duration must be positive")
        if not isinstance(raw_label, str) or not raw_label:
            raise SpeakerEvidenceError("speaker turn label must be non-empty")
        label = raw_label
        checked.append((start, end, label, index))
    checked.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
    return tuple((start, end, label) for start, end, label, _index in checked)


def _attribute_parent(unit: SourceUnit, turns: Sequence[Turn]) -> ParentSpeaker:
    duration = _positive_duration(unit)
    if duration <= 0:
        return ParentSpeaker("none")
    assert unit.start is not None and unit.end is not None
    start, end = float(unit.start), float(unit.end)
    overlap: dict[str, float] = {}
    first: dict[str, float] = {}
    for turn_start, turn_end, label in turns:
        amount = min(end, turn_end) - max(start, turn_start)
        if amount <= 0:
            continue
        overlap[label] = overlap.get(label, 0.0) + amount
        first[label] = min(first.get(label, turn_start), turn_start)
    multi = tuple(
        sorted(
            label
            for label, amount in overlap.items()
            if amount >= SPEAKER_MULTI_MIN_FRAC * duration
        )
    )
    if len(multi) >= 2:
        return ParentSpeaker("multi", support=multi)
    if not overlap:
        return ParentSpeaker("none")
    dominant = min(
        overlap,
        key=lambda label: (-overlap[label], first[label], label),
    )
    if overlap[dominant] >= SPEAKER_UNIT_COVER_FRAC * duration:
        return ParentSpeaker("single", dominant, (dominant,))
    return ParentSpeaker("none")


def _silence_break(left: SourceUnit, right: SourceUnit, threshold: float) -> bool:
    return (
        _finite(left.end)
        and _finite(right.start)
        and float(cast("float", right.start)) - float(cast("float", left.end))
        >= threshold
    )


def _regions(
    units: Sequence[SourceUnit], threshold: float
) -> tuple[tuple[int, int], ...]:
    if not units:
        return ()
    starts = [0]
    starts.extend(
        index
        for index, (left, right) in enumerate(zip(units, units[1:]), start=1)
        if _silence_break(left, right, threshold)
    )
    starts.append(len(units))
    return tuple(zip(starts, starts[1:]))


def _raw_labels(parents: Sequence[ParentSpeaker]) -> list[str | None]:
    return [item.label if item.kind == "single" else None for item in parents]


def _fill_none(
    parents: Sequence[ParentSpeaker],
    units: Sequence[SourceUnit],
    threshold: float,
) -> tuple[list[str | None], int]:
    labels = _raw_labels(parents)
    filled = 0
    for low, high in _regions(units, threshold):
        first_label = next((labels[i] for i in range(low, high) if labels[i]), None)
        if first_label is None:
            continue
        last_label: str | None = None
        for index in range(low, high):
            if labels[index] is not None:
                last_label = labels[index]
                continue
            if parents[index].kind != "none":
                continue
            chosen = first_label if last_label is None else last_label
            labels[index] = chosen
            filled += 1
    return labels, filled


def _transition_count(labels: Sequence[str | None]) -> int:
    return sum(
        left is not None and right is not None and left != right
        for left, right in zip(labels, labels[1:])
    )


def _run_duration(units: Sequence[SourceUnit], low: int, high: int) -> float:
    return sum(_positive_duration(unit) for unit in units[low:high])


def _absorb_segment(
    labels: list[str | None], units: Sequence[SourceUnit], low: int, high: int
) -> int:
    absorbed = 0
    while True:
        runs: list[tuple[str, int, int, float]] = []
        index = low
        while index < high:
            label = labels[index]
            if label is None:
                index += 1
                continue
            end = index + 1
            while end < high and labels[end] == label:
                end += 1
            runs.append((label, index, end, _run_duration(units, index, end)))
            index = end
        if len(runs) <= 1:
            return absorbed
        candidates: list[tuple[float, int]] = []
        for run_index, (_label, _start, _end, duration) in enumerate(runs):
            if duration >= SPEAKER_MIN_RUN_S:
                continue
            same_sandwich = (
                0 < run_index < len(runs) - 1
                and runs[run_index - 1][0] == runs[run_index + 1][0]
            )
            if same_sandwich or duration < SPEAKER_EDGE_RUN_MIN_S:
                candidates.append((duration, run_index))
        if not candidates:
            return absorbed
        _duration, run_index = min(candidates)
        _label, start, end, _run_duration_value = runs[run_index]
        left_duration = runs[run_index - 1][3] if run_index > 0 else -1.0
        right_duration = runs[run_index + 1][3] if run_index + 1 < len(runs) else -1.0
        if left_duration < 0 and right_duration < 0:
            return absorbed
        target = (
            runs[run_index + 1][0]
            if right_duration > left_duration
            else runs[run_index - 1][0]
        )
        labels[start:end] = [target] * (end - start)
        absorbed += 1


def _absorb_labels(
    labels: list[str | None],
    parents: Sequence[ParentSpeaker],
    units: Sequence[SourceUnit],
    threshold: float,
) -> tuple[list[str | None], int]:
    out = list(labels)
    absorbed = 0
    for region_low, region_high in _regions(units, threshold):
        segment_start = region_low
        for index in range(region_low, region_high + 1):
            at_end = index == region_high
            is_multi = not at_end and parents[index].kind == "multi"
            if not at_end and not is_multi:
                continue
            if segment_start < index:
                absorbed += _absorb_segment(out, units, segment_start, index)
            segment_start = index + 1
    return out, absorbed


def _phrase_ranges(
    units: Sequence[SourceUnit], lang: str
) -> tuple[tuple[int, int], ...]:
    """Provider phrase ranges expressed in production-parent unit indices."""
    if not units:
        return ()
    if not _no_spaces(lang):
        return tuple((index, index + 1) for index in range(len(units)))
    from .smart_split import _phrase_boundary_atoms

    atoms = [{"text": unit.surface} for unit in units]
    text = _join([unit.surface for unit in units], lang)
    starts = sorted(_phrase_boundary_atoms(atoms, text, lang) | {0, len(units)})
    return tuple(
        (left, right) for left, right in zip(starts, starts[1:]) if left < right
    )


def _split_phrase_ranges(
    ranges: Sequence[tuple[int, int]],
    units: Sequence[SourceUnit],
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    breaks = {
        index
        for index, (left, right) in enumerate(zip(units, units[1:]), start=1)
        if _silence_break(left, right, threshold)
    }
    out: list[tuple[int, int]] = []
    for low, high in ranges:
        points = [low, *sorted(point for point in breaks if low < point < high), high]
        out.extend(zip(points, points[1:]))
    return tuple(out)


def _phrase_snap(
    labels: list[str | None],
    parents: Sequence[ParentSpeaker],
    units: Sequence[SourceUnit],
    lang: str,
    threshold: float,
) -> tuple[list[str | None], int]:
    if not _no_spaces(lang):
        return list(labels), 0
    out = list(labels)
    snapped = 0
    ranges = _split_phrase_ranges(_phrase_ranges(units, lang), units, threshold)
    for low, high in ranges:
        weights: dict[str, float] = {}
        order: list[str] = []
        for index in range(low, high):
            label = labels[index]
            if label is None or parents[index].kind == "multi":
                continue
            if label not in weights:
                weights[label] = 0.0
                order.append(label)
            weights[label] += _positive_duration(units[index])
        if not weights:
            continue
        best = max(order, key=lambda label: weights[label])
        changed = False
        for index in range(low, high):
            if parents[index].kind == "multi":
                continue
            if out[index] != best:
                out[index] = best
                changed = True
        if changed:
            snapped += 1
    return out, snapped


def _conditioned_parents(
    labels: Sequence[str | None], parents: Sequence[ParentSpeaker]
) -> tuple[UnitSpeaker, ...]:
    out: list[UnitSpeaker] = []
    for index, (label, parent) in enumerate(zip(labels, parents)):
        if parent.kind == "multi":
            out.append(UnitSpeaker("multi", None, parent.support, index))
        elif label is None:
            out.append(UnitSpeaker("none", None, (), index))
        else:
            out.append(UnitSpeaker("single", label, (label,), index))
    return tuple(out)


def _checked_projection(
    parent_count: int,
    refined_units: Sequence[SourceUnit],
    origin: Sequence[int],
) -> tuple[tuple[SourceUnit, ...], tuple[int, ...]]:
    units = tuple(refined_units)
    owners = tuple(origin)
    if len(units) != len(owners):
        raise SpeakerEvidenceError("projection origin cardinality does not match units")
    if any(
        isinstance(owner, bool) or not isinstance(owner, int) or owner < 0
        for owner in owners
    ):
        raise SpeakerEvidenceError(
            "projection origin entries must be non-negative integers"
        )
    if any(left > right for left, right in zip(owners, owners[1:])):
        raise SpeakerEvidenceError("projection origin must be monotone")
    expected = tuple(range(parent_count))
    if tuple(sorted(set(owners))) != expected:
        raise SpeakerEvidenceError(
            "projection origin must be complete over every parent"
        )
    return units, owners


def _project_units(
    conditioned: Sequence[UnitSpeaker], origin: Sequence[int]
) -> tuple[UnitSpeaker, ...]:
    counts = Counter(origin)
    out: list[UnitSpeaker] = []
    for owner in origin:
        parent = conditioned[owner]
        if parent.kind == "multi":
            kind: UnitKind = "ambiguous" if counts[owner] > 1 else "multi"
            out.append(UnitSpeaker(kind, None, parent.support, owner))
        else:
            out.append(UnitSpeaker(parent.kind, parent.label, parent.support, owner))
    return tuple(out)


def _raw_events(turns: Sequence[Turn]) -> tuple[RawSpeakerEvent, ...]:
    out: list[RawSpeakerEvent] = []
    for left, right in zip(turns, turns[1:]):
        if left[2] == right[2]:
            continue
        time = transition_time(left[1], right[0])
        if time is None or not _finite(time):
            continue
        index = len(out)
        out.append(RawSpeakerEvent(f"e{index}", index, float(time), left[2], right[2]))
    return tuple(out)


def _event_in_speech(event: RawSpeakerEvent, units: Sequence[SourceUnit]) -> bool:
    return any(
        _finite(unit.start)
        and _finite(unit.end)
        and float(cast("float", unit.start))
        <= event.time
        <= float(cast("float", unit.end))
        and float(cast("float", unit.end)) > float(cast("float", unit.start))
        for unit in units
    )


def _structurally_unexpressible(
    event: RawSpeakerEvent, units: Sequence[SourceUnit]
) -> bool:
    return any(
        _finite(unit.start)
        and _finite(unit.end)
        and float(cast("float", unit.start))
        < event.time
        < float(cast("float", unit.end))
        for unit in units
    )


def _transitions(
    labels: Sequence[str | None], units: Sequence[SourceUnit]
) -> tuple[_LineageTransition, ...]:
    out: list[_LineageTransition] = []
    for boundary, (left, right) in enumerate(zip(labels, labels[1:]), start=1):
        if left is None or right is None or left == right:
            continue
        time = transition_time(units[boundary - 1].end, units[boundary].start)
        out.append(_LineageTransition(boundary, time))
    return tuple(out)


def _nearest_parent_boundary(
    event: RawSpeakerEvent, units: Sequence[SourceUnit]
) -> int | None:
    candidates: list[tuple[float, float, int]] = []
    for boundary in range(1, len(units)):
        time = transition_time(units[boundary - 1].end, units[boundary].start)
        if time is None or not _finite(time):
            continue
        distance = abs(float(time) - event.time)
        if distance <= EXPRESS_TOL_S:
            candidates.append((distance, float(time), boundary))
    return min(candidates)[2] if candidates else None


def injective_time_match(
    events: Sequence[LiveSpeakerEvent],
    boundaries: Sequence[BoundaryPoint],
) -> tuple[EventBoundaryMatch, ...]:
    """Greedy injective time-distance match with the LAW's full tie key."""
    event_ids = [event.event_id for event in events]
    boundary_ids = [boundary.boundary_id for boundary in boundaries]
    if len(set(event_ids)) != len(event_ids) or len(set(boundary_ids)) != len(
        boundary_ids
    ):
        raise SpeakerEvidenceError("match identities must be unique")
    candidates: list[
        tuple[float, float, str, float, str, LiveSpeakerEvent, BoundaryPoint]
    ] = []
    for event in events:
        for boundary in boundaries:
            distance = abs(event.time - boundary.time)
            if distance <= EXPRESS_TOL_S:
                candidates.append(
                    (
                        distance,
                        event.time,
                        event.event_id,
                        boundary.time,
                        boundary.boundary_id,
                        event,
                        boundary,
                    )
                )
    candidates.sort(key=lambda item: item[:7])
    used_events: set[str] = set()
    used_boundaries: set[str] = set()
    matches: list[EventBoundaryMatch] = []
    for distance, _et, _eid, _bt, _bid, event, boundary in candidates:
        if event.event_id in used_events or boundary.boundary_id in used_boundaries:
            continue
        used_events.add(event.event_id)
        used_boundaries.add(boundary.boundary_id)
        matches.append(
            EventBoundaryMatch(
                event_id=event.event_id,
                boundary_id=boundary.boundary_id,
                event_time=event.time,
                boundary_time=boundary.time,
                distance=distance,
            )
        )
    return tuple(matches)


def _initial_ancestry(
    events: Sequence[RawSpeakerEvent],
    transitions: Sequence[_LineageTransition],
) -> tuple[tuple[_LineageTransition, ...], frozenset[str]]:
    live = tuple(
        LiveSpeakerEvent(event.event_id, event.index, event.time) for event in events
    )
    boundaries = tuple(
        BoundaryPoint(f"t{item.boundary}", item.boundary, item.time)
        for item in transitions
        if item.time is not None and _finite(item.time)
    )
    matches = injective_time_match(live, boundaries)
    by_boundary = {int(match.boundary_id[1:]): match.event_id for match in matches}
    attached = frozenset(match.event_id for match in matches)
    return (
        tuple(
            _LineageTransition(
                item.boundary,
                item.time,
                frozenset(
                    {by_boundary[item.boundary]}
                    if item.boundary in by_boundary
                    else set()
                ),
            )
            for item in transitions
        ),
        attached,
    )


def _advance_lineage(
    previous: Sequence[_LineageTransition],
    output: Sequence[_LineageTransition],
) -> tuple[tuple[_LineageTransition, ...], frozenset[str]]:
    candidates = sorted(
        (
            abs(left.boundary - right.boundary),
            left.boundary,
            right.boundary,
            left_index,
            right_index,
        )
        for left_index, left in enumerate(previous)
        for right_index, right in enumerate(output)
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    ancestry: dict[int, frozenset[str]] = {}
    for _distance, _lb, _rb, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        ancestry[right_index] = previous[left_index].ancestry
    removed = frozenset(
        event_id
        for index, item in enumerate(previous)
        if index not in used_left
        for event_id in item.ancestry
    )
    return (
        tuple(
            _LineageTransition(
                item.boundary,
                item.time,
                ancestry.get(index, frozenset()),
            )
            for index, item in enumerate(output)
        ),
        removed,
    )


def _lineage(
    raw_events: Sequence[RawSpeakerEvent],
    in_speech_events: Sequence[RawSpeakerEvent],
    parents: Sequence[ParentSpeaker],
    parent_units: Sequence[SourceUnit],
    filled_labels: Sequence[str | None],
    absorbed_labels: Sequence[str | None],
    snapped_labels: Sequence[str | None],
) -> tuple[
    tuple[tuple[str, BucketKind], ...],
    tuple[LiveSpeakerEvent, ...],
    int,
]:
    attribution_labels = _raw_labels(parents)
    initial, attached = _initial_ancestry(
        raw_events, _transitions(attribution_labels, parent_units)
    )
    removed: set[str] = set()
    current = initial
    for labels in (filled_labels, absorbed_labels, snapped_labels):
        current, removed_now = _advance_lineage(
            current, _transitions(labels, parent_units)
        )
        removed.update(removed_now)
    final_positions: dict[str, float] = {}
    for item in current:
        if item.time is None or not _finite(item.time):
            continue
        for event_id in item.ancestry:
            final_positions[event_id] = float(item.time)

    buckets: list[tuple[str, BucketKind]] = []
    live: list[LiveSpeakerEvent] = []
    unexpressible_count = 0
    for event in in_speech_events:
        if _structurally_unexpressible(event, parent_units):
            buckets.append((event.event_id, "unexpressible"))
            unexpressible_count += 1
            continue
        boundary = _nearest_parent_boundary(event, parent_units)
        if boundary is not None:
            left = filled_labels[boundary - 1]
            right = filled_labels[boundary]
            if left is None or right is None:
                buckets.append((event.event_id, "unattributed_loss"))
                continue
        # R8-3: this is deliberately a named, literal terminal branch.  It is
        # not an ``else`` that happens to catch an initially unmatched event.
        initially_unmatched = event.event_id not in attached
        if initially_unmatched:
            buckets.append((event.event_id, "policy_filtered"))
            continue
        if event.event_id in removed or event.event_id not in final_positions:
            buckets.append((event.event_id, "policy_filtered"))
            continue
        live.append(
            LiveSpeakerEvent(
                event.event_id, event.index, final_positions[event.event_id]
            )
        )
    return tuple(buckets), tuple(live), unexpressible_count


def speaker_evidence(
    document: SegDocument,
    *,
    refined_units: Sequence[SourceUnit] | None = None,
    origin: Sequence[int] | None = None,
) -> UnitSpeakers:
    """Attribute production parents, condition them, then project through origin.

    ``document`` is always the PRE-refinement document.  Supplying one of
    ``refined_units``/``origin`` requires the other; the origin must be complete,
    monotone, and cardinality-equal to the refined stream.
    """
    parent_units = tuple(document.units)
    if (refined_units is None) != (origin is None):
        raise SpeakerEvidenceError(
            "refined units and projection origin must be supplied together"
        )
    if refined_units is None:
        projected_units, owners = _checked_projection(
            len(parent_units), parent_units, tuple(range(len(parent_units)))
        )
    else:
        assert origin is not None
        projected_units, owners = _checked_projection(
            len(parent_units), refined_units, origin
        )

    present = document.speaker_turns is not None
    turns = _checked_turns(document.speaker_turns or ())
    parents = tuple(_attribute_parent(item, turns) for item in parent_units)
    edge_silence_s = (document.profile.vad_skip_ms + 50.0) / 1000.0
    if not _finite(edge_silence_s) or edge_silence_s < 0:
        raise SpeakerEvidenceError(
            "speaker edge silence must be finite and non-negative"
        )
    before = _raw_labels(parents)
    filled, filled_count = _fill_none(parents, parent_units, edge_silence_s)
    absorbed, absorbed_count = _absorb_labels(
        filled, parents, parent_units, edge_silence_s
    )
    snapped, phrase_snaps = _phrase_snap(
        absorbed,
        parents,
        parent_units,
        document.language,
        edge_silence_s,
    )
    conditioned = _conditioned_parents(snapped, parents)
    projected = _project_units(conditioned, owners)

    raw = _raw_events(turns)
    in_speech = tuple(event for event in raw if _event_in_speech(event, parent_units))
    base_buckets, live, unexpressible_count = _lineage(
        raw,
        in_speech,
        parents,
        parent_units,
        filled,
        absorbed,
        snapped,
    )
    stats = SpeakerConditioningStats(
        units_attributed=sum(item.kind == "single" for item in parents),
        units_none=sum(item.kind == "none" for item in parents),
        filled=filled_count,
        phrase_snaps=phrase_snaps,
        runs_absorbed=absorbed_count,
        transitions_before=_transition_count(before),
        transitions_after=_transition_count(snapped),
        multilabel_parents=sum(item.kind == "multi" for item in parents),
        unexpressible_turn_changes=unexpressible_count,
    )
    return UnitSpeakers(
        language=document.language,
        parent_units=parent_units,
        refined_units=projected_units,
        origin=owners,
        turn_track_present=present,
        turns=turns,
        parent_speakers=parents,
        conditioned_parents=conditioned,
        unit_speakers=projected,
        edge_silence_s=edge_silence_s,
        stats=stats,
        raw_events=raw,
        in_speech_event_ids=tuple(event.event_id for event in in_speech),
        base_event_buckets=base_buckets,
        live_events=live,
    )


def _checked_unit_range(
    unit_range: tuple[int, int], unit_count: int
) -> tuple[int, int]:
    try:
        low, high = unit_range
    except (TypeError, ValueError) as exc:
        raise SpeakerEvidenceError("unit range must contain two bounds") from exc
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, int)
        or not isinstance(high, int)
        or not 0 <= low < high <= unit_count
    ):
        raise SpeakerEvidenceError("unit range must be a positive in-bounds interval")
    return low, high


def _raw_overlap(evidence: UnitSpeakers, span: EvidenceSpan) -> bool:
    clipped: list[tuple[float, float, str]] = []
    for start, end, label in evidence.turns:
        low, high = max(start, span.start), min(end, span.end)
        if high > low:
            clipped.append((low, high, label))
    return any(
        left[2] != right[2] and max(left[0], right[0]) < min(left[1], right[1])
        for index, left in enumerate(clipped)
        for right in clipped[index + 1 :]
    )


def speaker_edge_cost(
    evidence: UnitSpeakers,
    unit_range: tuple[int, int],
    *,
    evidence_span: EvidenceSpan,
    sing_spans: Sequence[tuple[float, float]] | None = None,
    weight: float = W_SPEAKER_INTERIOR,
    suppressed_lyric: bool | None = None,
) -> CostBreakdown:
    """Price conditioned transitions strictly inside one candidate edge."""
    low, high = _checked_unit_range(unit_range, len(evidence.unit_speakers))
    if not _finite(weight) or float(weight) < 0:
        raise SpeakerEvidenceError(
            "speaker edge weight must be finite and non-negative"
        )
    owned = evidence.unit_speakers[low:high]
    labels = [item.label if item.kind == "single" else None for item in owned]
    changes = _transition_count(labels)
    parent_ids = sorted(set(evidence.origin[low:high]))
    multi_support = any(
        evidence.parent_speakers[parent].kind == "multi" for parent in parent_ids
    )
    known = {label for label in labels if label is not None}
    supported = set(known)
    for parent in parent_ids:
        supported.update(evidence.parent_speakers[parent].support)
    if not evidence.turn_track_present:
        turn_state = "absent"
    elif _raw_overlap(evidence, evidence_span):
        turn_state = "overlap"
    elif multi_support or len(known) >= 2:
        turn_state = "multi"
    elif known:
        turn_state = "single"
    else:
        turn_state = "unattributed"
    if suppressed_lyric is not None and not isinstance(suppressed_lyric, bool):
        raise SpeakerEvidenceError("cached lyric classification must be boolean")
    suppressed = (
        lyric_for_evidence(evidence_span, sing_spans)
        if suppressed_lyric is None
        else suppressed_lyric
    )
    return make_breakdown(
        {
            "speaker_changes_in_cue_raw": float(changes),
            "suppressed_lyric": suppressed,
            "turn_state": turn_state,
            "two_speaker_raw": float(len(supported) == 2),
        },
        {
            "speaker_interior": 0.0 if suppressed else float(weight) * changes,
        },
    )


@dataclass(frozen=True)
class SpeakerPricingSummary:
    priced_edges: int
    speaker_changes_in_cue_raw: int
    suppressed_lyric_edges: int
    two_speaker_edges: int
    turn_states: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "priced_edges": self.priced_edges,
            "speaker_changes_in_cue_raw": self.speaker_changes_in_cue_raw,
            "suppressed_lyric_edges": self.suppressed_lyric_edges,
            "turn_states": {
                state: int(self.turn_states.get(state, 0)) for state in TURN_STATES
            },
            "two_speaker_edges": self.two_speaker_edges,
        }


def summarize_speaker_prices(parts: Sequence[CostBreakdown]) -> SpeakerPricingSummary:
    states = Counter(
        str(part.features["turn_state"])
        for part in parts
        if part.features.get("turn_state") in TURN_STATES
    )
    return SpeakerPricingSummary(
        priced_edges=len(parts),
        speaker_changes_in_cue_raw=sum(
            int(cast("float", part.features.get("speaker_changes_in_cue_raw", 0.0)))
            for part in parts
        ),
        suppressed_lyric_edges=sum(
            part.features.get("suppressed_lyric") is True for part in parts
        ),
        two_speaker_edges=sum(
            part.features.get("two_speaker_raw") == 1.0 for part in parts
        ),
        turn_states={state: states.get(state, 0) for state in TURN_STATES},
    )


def _checked_cue_ranges(
    cues_count: int,
    cue_unit_ranges: Sequence[tuple[int, int]],
    unit_count: int,
) -> tuple[tuple[int, int], ...]:
    ranges = tuple(cue_unit_ranges)
    if len(ranges) != cues_count:
        raise SpeakerProjectionError("cue/range cardinality differs")
    if not ranges:
        if unit_count:
            raise SpeakerProjectionError("empty ranges do not cover all units")
        return ()
    checked: list[tuple[int, int]] = []
    for item in ranges:
        try:
            low, high = item
        except (TypeError, ValueError) as exc:
            raise SpeakerProjectionError("range must contain two bounds") from exc
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, int)
            or not isinstance(high, int)
            or low >= high
        ):
            raise SpeakerProjectionError("every range must have positive width")
        checked.append((low, high))
    if checked[0][0] != 0:
        raise SpeakerProjectionError("ranges must start at 0")
    if any(left[1] != right[0] for left, right in zip(checked, checked[1:])):
        raise SpeakerProjectionError("ranges must be contiguous and non-overlapping")
    if checked[-1][1] != unit_count:
        raise SpeakerProjectionError("ranges must cover all attributed units")
    return tuple(checked)


def _speaker_id_plan(
    cue_unit_ranges: Sequence[tuple[int, int]],
    unit_speakers: Sequence[UnitSpeaker],
    *,
    cue_count: int,
) -> tuple[tuple[str | None, bool], ...]:
    ranges = _checked_cue_ranges(cue_count, cue_unit_ranges, len(unit_speakers))
    plan: list[tuple[str | None, bool]] = []
    for low, high in ranges:
        owned = unit_speakers[low:high]
        labels = {item.label for item in owned if item.kind == "single" and item.label}
        fully_single = bool(owned) and all(item.kind == "single" for item in owned)
        label = next(iter(labels)) if fully_single and len(labels) == 1 else None
        named_multi = len(labels) > 1 or any(
            item.kind in {"ambiguous", "multi"} for item in owned
        )
        plan.append((label, named_multi))
    return tuple(plan)


def annotate_speaker_ids(
    cues: Sequence[Cue],
    cue_unit_ranges: Sequence[tuple[int, int]],
    unit_speakers: Sequence[UnitSpeaker],
) -> None:
    """Project selected ownership to transient display metadata, in place only."""
    plan = _speaker_id_plan(cue_unit_ranges, unit_speakers, cue_count=len(cues))
    for cue, (label, _named_multi) in zip(cues, plan):
        cue.pop("speaker_ids", None)
        if label is not None:
            cue["speaker_ids"] = [label]


def named_multi_cues_unannotated(
    cue_unit_ranges: Sequence[tuple[int, int]],
    unit_speakers: Sequence[UnitSpeaker],
) -> int:
    """Count selected named multi/ambiguous cues the honest policy leaves unnamed."""
    return sum(
        named_multi
        for _label, named_multi in _speaker_id_plan(
            cue_unit_ranges,
            unit_speakers,
            cue_count=len(cue_unit_ranges),
        )
    )


@dataclass(frozen=True)
class SpeakerMeasurement:
    raw_in_speech_turn_changes: int
    buckets: Mapping[str, int]
    event_buckets: Mapping[str, str]
    matches: tuple[EventBoundaryMatch, ...]
    speaker_attributable_expressed_cuts: int

    @property
    def expressed_rate(self) -> float:
        if not self.raw_in_speech_turn_changes:
            return 0.0
        return self.buckets["expressed"] / self.raw_in_speech_turn_changes

    @property
    def expressible_hit_rate(self) -> float | None:
        denominator = (
            self.buckets["expressed"] + self.buckets["survived_expressible_but_missed"]
        )
        return None if denominator == 0 else self.buckets["expressed"] / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "buckets": {kind: int(self.buckets[kind]) for kind in BUCKET_KINDS},
            "expressed_rate": self.expressed_rate,
            "expressible_hit_rate": self.expressible_hit_rate,
            "matches": [match.to_dict() for match in self.matches],
            "raw_in_speech_turn_changes": self.raw_in_speech_turn_changes,
            "speaker_attributable_expressed_cuts": self.speaker_attributable_expressed_cuts,
        }


def _boundary_points(values: Sequence[float]) -> tuple[BoundaryPoint, ...]:
    return tuple(
        BoundaryPoint(f"b{index}", index, float(value))
        for index, value in enumerate(values)
    )


def measure_speaker_events(
    evidence: UnitSpeakers,
    *,
    delivered_boundaries: Sequence[float],
    off_boundaries: Sequence[float] | None = None,
) -> SpeakerMeasurement:
    """Classify every raw in-speech event into exactly one conserved bucket."""
    base = dict(evidence.base_event_buckets)
    matches = injective_time_match(
        evidence.live_events, _boundary_points(delivered_boundaries)
    )
    expressed_ids = {match.event_id for match in matches}
    event_buckets: dict[str, str] = {}
    counts = {kind: 0 for kind in BUCKET_KINDS}
    raw_by_id = {event.event_id: event for event in evidence.raw_events}
    for event_id in evidence.in_speech_event_ids:
        bucket: str
        if event_id in base:
            bucket = base[event_id]
        elif event_id in expressed_ids:
            bucket = "expressed"
        else:
            bucket = "survived_expressible_but_missed"
        event_buckets[event_id] = bucket
        counts[bucket] += 1

    attributable = 0
    if off_boundaries is not None:
        off_matches = injective_time_match(
            evidence.live_events, _boundary_points(off_boundaries)
        )
        off_ids = {match.event_id for match in off_matches}
        attributable = sum(
            event_id not in off_ids
            for event_id in expressed_ids
            if event_id in raw_by_id
        )
    raw_count = len(evidence.in_speech_event_ids)
    if sum(counts.values()) != raw_count:
        raise SpeakerEvidenceError("speaker bucket conservation failed")
    return SpeakerMeasurement(
        raw_in_speech_turn_changes=raw_count,
        buckets=counts,
        event_buckets=event_buckets,
        matches=matches,
        speaker_attributable_expressed_cuts=attributable,
    )
