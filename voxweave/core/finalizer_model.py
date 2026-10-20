"""Phase-1 model of the finalizer: closed vocabularies, value types, per-cue phase 1."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .authority import AuthorityKind, Capability
from .canonical_text import (
    CanonicalWork,
    FinalText,
    canonical_text,
    line_budget,
    over_wide_token,
)
from .layout import _reading_chars
from .partition_check import ReportTag, Waiver
from .schema import Cue, Unit
from .segdoc import DisplayProfile
from .timing import LINGER_CAP_S

# --------------------------------------------------------------- vocabularies

#: Closed report vocabulary (spec section 2.4), sorted. NOTHING outside this
#: tuple may be reported: a refusal with no vocabulary entry is a silent one.
REPORT_KINDS: tuple[str, ...] = (
    "canonical-text-fallback",
    "fabricated-time",
    "input-overlap",
    "line-capacity",
    "min-duration-short",
    "min-gap-unmet",
    "shot-unhonored",
    "solver-budget-exhausted",
    "stutter-not-proven-fixed-within-4-scans",
)

#: The only waiver kind, minted by the cap slot alone. ``fabricated-time`` is a
#: report and waives nothing -- an anchorless cue still faces every predicate.
FINALIZER_WAIVER_KINDS: tuple[str, ...] = ("held-chain-duration",)

#: Sweep rule ids in slot order. The trace and its validator index by these, so
#: the tuple is frozen: renaming a slot renames a contract.
RULE_IDS: tuple[str, ...] = (
    "duration-desire",  # slot 1
    "chain",  # slot 2
    "cap",  # slot 3
    "shot-in",  # slot 4
    "shot-out",  # slot 5
    "ladder-1",  # slot 6, branch 1
    "ladder-2",  # slot 6, branch 2
    "ladder-3",  # slot 6, branch 3
)

TERMINALS: tuple[str, ...] = ("budget-exhausted", "cycle-adoption", "fixed-point")

Terminal = Literal["budget-exhausted", "cycle-adoption", "fixed-point"]
Side = Literal["start", "end"]

#: One cue's normalized ``(start, end)`` pair per entry; the unit of identity.
StreamState = tuple[tuple[float, float], ...]


class NonFiniteTime(ValueError):
    """A cue bound or a shot time is NaN or infinite (preflight, section 2.5).

    Refused at the door rather than downstream: NaN compares false against
    everything, so it would defeat cycle detection silently instead of loudly.
    The shadow hook's catch-all turns this into the typed ``error`` block.
    """


# ------------------------------------------------------- state canonicalization


def normalize_time(value: float) -> float:
    """The one normal form a display time may take: finite, with ``-0.0 -> 0.0``.

    Signed zero is normalized because ``-0.0 == 0.0`` while their artifact bytes
    differ, so leaving it alone would make two runs "equal" and serialize
    differently.
    """
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteTime(f"{value!r} is not a finite time")
    return number + 0.0


def pack_state(state: StreamState) -> bytes:
    """Big-endian packed bytes -- IDENTITY AND HASHING ONLY.

    Never an ordering key: little-endian ordering puts ``12f`` below ``10f``,
    the opposite of numeric order, and adopting a cycle minimum by byte order
    would silently return the wrong member (pinned negative test).
    """
    flat = [value for pair in state for value in pair]
    return struct.pack(f">{len(flat)}d", *flat)


def state_key(state: StreamState) -> tuple[float, ...]:
    """The decoded numeric compare key for cycle minimality and evidence order."""
    return tuple(value for pair in state for value in pair)


# ----------------------------------------------------------------- value types


@dataclass(frozen=True)
class BoundaryRef:
    """One movable boundary: a cue index plus which side of it."""

    cue_index: int
    side: Side

    def to_dict(self) -> dict[str, Any]:
        return {"cue_index": self.cue_index, "side": self.side}


@dataclass(frozen=True)
class BoundaryMovement:
    """What phase 2 did to one boundary, recorded for every boundary.

    Phase-2 movement is UNBOUNDED-BUT-MEASURED by contract (spec section 4): the
    preview promises phase 1 and nothing else, so the honest report is the pair
    rather than a bound nobody proved.
    """

    boundary: BoundaryRef
    phase1: float
    delivered: float

    @property
    def delta(self) -> float:
        return self.delivered - self.phase1

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.to_dict(),
            "delivered": self.delivered,
            "delta": self.delta,
            "phase1": self.phase1,
        }


@dataclass(frozen=True)
class NeighbourRead:
    """A value a rule read that is not the boundary it wrote.

    Recorded so the validator can check the READ against its own evolving state
    instead of replaying the producer's claim about it -- a forged snapshot is
    the pinned negative fixture.
    """

    boundary: BoundaryRef
    value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"boundary": self.boundary.to_dict(), "value": self.value}


@dataclass(frozen=True)
class TraceLeg:
    """One rule application that moved one boundary."""

    rule_id: str
    sweep: int
    cue_index: int
    slot: int
    target: BoundaryRef
    from_value: float
    to_value: float
    reads: tuple[NeighbourRead, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_index": self.cue_index,
            "from": self.from_value,
            "reads": [read.to_dict() for read in self.reads],
            "rule_id": self.rule_id,
            "slot": self.slot,
            "sweep": self.sweep,
            "target": self.target.to_dict(),
            "to": self.to_value,
        }


@dataclass(frozen=True)
class CycleEvidence:
    """The visited cycle, its per-boundary value sets and the member adopted."""

    members: tuple[StreamState, ...]
    per_boundary_values: tuple[tuple[BoundaryRef, tuple[float, ...]], ...]
    adopted: StreamState

    def to_dict(self) -> dict[str, Any]:
        return {
            "adopted": [list(pair) for pair in self.adopted],
            "members": [[list(pair) for pair in member] for member in self.members],
            "per_boundary_values": [
                {"boundary": ref.to_dict(), "values": list(values)}
                for ref, values in self.per_boundary_values
            ],
        }


@dataclass(frozen=True)
class Trace:
    """ONE ordered, document-global trace plus its typed terminal."""

    legs: tuple[TraceLeg, ...]
    terminal: Terminal
    cycle: CycleEvidence | None
    sweeps: int
    schedule_canonicality: Literal["unverified"] = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": None if self.cycle is None else self.cycle.to_dict(),
            "legs": [leg.to_dict() for leg in self.legs],
            "schedule_canonicality": self.schedule_canonicality,
            "sweeps": self.sweeps,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class Phase1Cue:
    """One cue's phase-1 answer: the seed state and the preview's contract.

    ``seed_start``/``seed_end`` are the immutable input span. Slot 1 recomputes
    its desire against ``seed_end`` and never against the current end, so a
    sweep cannot stack an extension on the extension it granted last sweep.
    """

    index: int
    start: float
    end: float
    seed_start: float
    seed_end: float
    speech_start: float | None
    speech_end: float | None
    text: str
    lines: tuple[str, ...]
    cell_widths: tuple[int, ...]
    reading_chars: int
    raw_reading_chars: int
    word_data: list[Unit]
    unit_range: tuple[int, int] | None
    lyric: bool | None
    reports: tuple[ReportTag, ...]


@dataclass(frozen=True)
class Phase1CueStream:
    """The ONLY input ``finalize`` accepts, minted exclusively by a factory.

    The capability is the lifecycle token: ``finalize`` consumes it, so one
    sealed seed produces exactly one root and a second finalize raises instead
    of quietly minting a second answer for the same row (N8b).
    """

    cues: tuple[Phase1Cue, ...]
    profile: DisplayProfile
    seed_id: str
    row_id: str
    evaluation_id: str
    authority_kind: AuthorityKind
    capability: Capability
    input_kind: Literal["phase1"] = "phase1"


@dataclass(frozen=True)
class FinalizePolicy:
    """Reserved for P6; ``overlap_policy="reject"`` is the only P5 value."""

    overlap_policy: Literal["reject"] = "reject"
    min_gap: Literal["two-frame"] = "two-frame"
    grid: None = None


@dataclass(frozen=True)
class FinalizeEvidence:
    """Shots and sing spans ONLY -- speaker facts never enter ``finalize``.

    ``sing_spans`` is carried but unread by the P5 rules: it records the
    evidence set the run saw, and keeps the P6 seam from changing this
    signature. Lyric classification is STAMPED from the evidence span upstream
    (FD-2), never recomputed here from a display span the sweep just widened.
    """

    shots: tuple[float, ...] = ()
    sing_spans: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class FinalizeReport:
    """Every fact the run is obliged to state, derived from seed + terminal."""

    entries: tuple[ReportTag, ...]
    waivers: tuple[Waiver, ...]
    deltas_fired: tuple[str, ...]
    movement: tuple[BoundaryMovement, ...]
    max_start_movement_s: float
    max_sweeps_observed: int
    terminal: Terminal
    schedule_canonicality: Literal["unverified"] = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "deltas_fired": list(self.deltas_fired),
            "entries": [tag.to_dict() for tag in self.entries],
            "max_start_movement_s": self.max_start_movement_s,
            "max_sweeps_observed": self.max_sweeps_observed,
            "movement": [move.to_dict() for move in self.movement],
            "schedule_canonicality": self.schedule_canonicality,
            "terminal": self.terminal,
            "waivers": [waiver.to_dict() for waiver in self.waivers],
        }


@dataclass(frozen=True)
class FinalizeResult:
    """The delivered stream, its report, its trace, and whether it is a measurement."""

    cues: tuple[Cue, ...]
    report: FinalizeReport
    trace: Trace
    valid: bool


# ------------------------------------------------------------------- phase one


def _text_reports(
    final: FinalText, profile: DisplayProfile, index: int
) -> list[ReportTag]:
    """Statements C2-C5 of the phase-1 pseudocode, in emission order.

    ``line-capacity`` covers ANY over-wide delivered line, and its ``token`` is
    ``None`` exactly when no single atom is too wide -- that shape says the wrap
    ran out of lines rather than that some word cannot be broken.
    """
    tags: list[ReportTag] = []
    if final.source == "fallback":
        tags.append(
            ReportTag(
                kind="canonical-text-fallback",
                cue_index=index,
                evidence={"reason": final.fallback_reason},
            )
        )
    if not final.stutter_stable:
        tags.append(
            ReportTag(
                kind="stutter-not-proven-fixed-within-4-scans",
                cue_index=index,
                evidence={"scans": final.stutter_scans},
            )
        )
    budget = line_budget(profile)
    if len(final.lines) > profile.max_lines:
        tags.append(
            ReportTag(
                kind="line-capacity",
                cue_index=index,
                evidence={
                    "budget": budget,
                    "line_index": None,
                    "token": None,
                    "width": None,
                },
            )
        )
    for line_index, width in enumerate(final.cell_widths):
        if width > budget:
            tags.append(
                ReportTag(
                    kind="line-capacity",
                    cue_index=index,
                    evidence={
                        "budget": budget,
                        "line_index": line_index,
                        "token": over_wide_token(
                            final.lines[line_index], profile.language, budget
                        ),
                        "width": width,
                    },
                )
            )
    return tags


def _duration_desire(
    *,
    start: float,
    seed_end: float,
    speech_end: float | None,
    reading_chars: int,
    profile: DisplayProfile,
) -> float:
    """The ABSOLUTE end this cue wants (statements D1-D10), operand order frozen.

    Every ``max``/``min`` keeps ``_cleanup_cues``' associativity, so the result
    is bit-identical to the pass this ports rather than merely close. The
    anchorless branch is TOTAL (FD-8): with no speech end there is no extension
    at all, the min-duration floor included -- a floor is an extension.
    """
    want = seed_end
    if speech_end is None:
        return want
    if profile.min_cue_s > 0:
        want = max(want, start + profile.min_cue_s)
    if profile.lag_out_s > 0:
        want = max(want, speech_end + profile.lag_out_s)
    if profile.cps > 0:
        need = reading_chars / profile.cps
        want = max(want, min(start + need, speech_end + LINGER_CAP_S))
    return want


def phase1_cue(
    cue: Cue,
    *,
    profile: DisplayProfile,
    index: int,
    work: CanonicalWork | None = None,
    expected_footprint: str | None = None,
    unit_range: tuple[int, int] | None = None,
) -> Phase1Cue:
    """Phase 1 for one cue: pure, order-free, neighbour-free.

    Anchors come from the cue KEYS, never re-derived from ``word_data`` (spec
    section 2.2). The two agree for every stream either engine constructs; where
    they could disagree -- a hand-built input -- the keys are what the spec calls
    authoritative, and re-deriving would let display polish leak into evidence.
    """
    start = normalize_time(cue["start"])
    end = normalize_time(cue["end"])

    speech_start = cue.get("speech_start")
    speech_end = cue.get("speech_end")
    reports: list[ReportTag] = []
    if speech_start is None:
        reports.append(
            ReportTag(
                kind="fabricated-time", cue_index=index, evidence={"side": "start"}
            )
        )
    if speech_end is None:
        reports.append(
            ReportTag(kind="fabricated-time", cue_index=index, evidence={"side": "end"})
        )
    if speech_start is not None:
        speech_start = normalize_time(speech_start)
    if speech_end is not None:
        speech_end = normalize_time(speech_end)

    raw_text = cue["text"]
    final = canonical_text(
        cue["word_data"],
        fallback_text=raw_text,
        lang=profile.language,
        profile=profile,
        expected_footprint=expected_footprint,
        work=work,
    )
    reports.extend(_text_reports(final, profile, index))

    want = _duration_desire(
        start=start,
        seed_end=end,
        speech_end=speech_end,
        reading_chars=final.reading_chars,
        profile=profile,
    )
    if want > end:
        end = want

    return Phase1Cue(
        index=index,
        start=start,
        end=end,
        seed_start=start,
        seed_end=normalize_time(cue["end"]),
        speech_start=speech_start,
        speech_end=speech_end,
        text=final.text,
        lines=final.lines,
        cell_widths=final.cell_widths,
        reading_chars=final.reading_chars,
        raw_reading_chars=_raw_reading_chars(raw_text),
        word_data=cue["word_data"],
        unit_range=unit_range,
        lyric=cue.get("lyric"),
        reports=tuple(reports),
    )


def _raw_reading_chars(text: str) -> int:
    """The load the LEGACY preview charges, kept only to detect FD-1 firing."""
    return _reading_chars(text)


def phase1_stream(
    cues: Sequence[Cue],
    *,
    profile: DisplayProfile,
    work: CanonicalWork | None = None,
    footprints: Sequence[str | None] | None = None,
    unit_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[Phase1Cue, ...]:
    """Phase 1 over a whole stream. Pure: no seal, no ledger, no event."""
    return tuple(
        phase1_cue(
            cue,
            profile=profile,
            index=index,
            work=work,
            expected_footprint=None if footprints is None else footprints[index],
            unit_range=None if unit_ranges is None else unit_ranges[index],
        )
        for index, cue in enumerate(cues)
    )


def seed_reports(cues: Sequence[Cue]) -> tuple[ReportTag, ...]:
    """``input-overlap`` for every overlapping SEED pair, whatever resolves it.

    Derived before phase 1 and from the input bounds alone, because the fact
    being recorded is about the stream that arrived -- not about the overlaps
    phase 1 legitimately creates when a desire reaches past a neighbour.
    """
    tags: list[ReportTag] = []
    for index in range(len(cues) - 1):
        prev_end = cues[index]["end"]
        next_start = cues[index + 1]["start"]
        if prev_end > next_start:
            tags.append(
                ReportTag(
                    kind="input-overlap",
                    cue_index=index,
                    evidence={"next_start": next_start, "prev_end": prev_end},
                )
            )
    return tuple(tags)


def _unit_payload(word_data: Sequence[Unit]) -> list[list[Any]]:
    return [
        [unit.get("text", unit.get("word", "")), unit.get("start"), unit.get("end")]
        for unit in word_data
    ]


def _profile_payload(profile: DisplayProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "clause_ms": profile.clause_ms,
        "cps": profile.cps,
        "glue_gap_s": profile.glue_gap_s,
        "lag_out_s": profile.lag_out_s,
        "language": profile.language,
        "max_cue_s": profile.max_cue_s,
        "max_line_length": profile.max_line_length,
        "max_lines": profile.max_lines,
        "min_cue_s": profile.min_cue_s,
        "offline_ms": profile.offline_ms,
        "shot_snap_s": profile.shot_snap_s,
        "vad_skip_ms": profile.vad_skip_ms,
    }


def _cue_payload(cues: Sequence[Cue]) -> dict[str, Any]:
    """The projection a v1 capture is sealed over: bounds, anchors, word data."""
    return {
        "cues": [
            [
                cue.get("text", ""),
                cue.get("start"),
                cue.get("end"),
                cue.get("speech_start"),
                cue.get("speech_end"),
                cue.get("lyric"),
                _unit_payload(cue.get("word_data") or []),
            ]
            for cue in cues
        ]
    }


def _stream_payload(
    cues: Sequence[Phase1Cue],
    profile: DisplayProfile | None,
    row_id: str,
    evaluation_id: str,
) -> dict[str, Any]:
    """The projection a phase-1 stream is sealed over.

    Covers the word data too: a stream whose units were edited after issuance is
    a different seed, and a seal that could not see the edit would not be a seal.
    """
    return {
        "cues": [
            [
                cue.index,
                cue.start,
                cue.end,
                cue.seed_start,
                cue.seed_end,
                cue.speech_start,
                cue.speech_end,
                cue.text,
                cue.reading_chars,
                cue.lyric,
                list(cue.unit_range) if cue.unit_range is not None else None,
                _unit_payload(cue.word_data),
            ]
            for cue in cues
        ],
        "evaluation_id": evaluation_id,
        "profile": _profile_payload(profile),
        "row_id": row_id,
    }
