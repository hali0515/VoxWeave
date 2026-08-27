"""The TimelineFinalizer: one solver where thirteen display passes used to run.

The legacy display chain is a *sequence* of passes, each reading what the last
one wrote. That shape has three defects P5 exists to remove. It is not a
function of its input (running it twice moves cues, which is why the diarize
path has to re-run part of it and hope); its extensions are relative, so a pad
can stack on a pad; and every refusal it makes -- a snap it declined, a floor a
neighbour blocked -- is invisible, because a pass that returns a cue stream has
nowhere to say "I could not".

This module replaces that with two phases (spec sections 2.2-2.5).

**Phase 1** is pure, per-cue and order-free: from the cue's own record and the
profile it computes the canonical text and an ABSOLUTE duration desire. It reads
no neighbour, no shot, no speaker fact, and it never moves a start. Being a
function of the immutable record is what lets the same code be the optimizer's
preview (:class:`FinalizerPreview`): the number the cost model consumed is the
number the finalizer will start from, float-exact (gate N7).

**Phase 2** is one deterministic sweep ``S`` -- production's per-cue interleaved
rule order, every rule inside it -- iterated to a fixed point. Iterating rather
than running once is the point: a start that moves changes the cap target of its
own cue, and the neighbour clamps a single pass leaves behind are exactly the
run-1/run-2 divergences reviews 3 and 4 found in the legacy composite. Slot 1
recomputes the desire from the IMMUTABLE seed end, so a desire can never stack
on the extension a previous sweep granted; that immutable basis is what makes
the iteration converge instead of ratchet.

Termination is a POLICY, not a theorem (N8c): after :data:`SWEEP_BUDGET` sweeps
the run freezes, reports ``solver-budget-exhausted`` and returns
``valid=False`` -- a typed invalid measurement the harness must short-circuit,
never a quietly frozen answer. A revisited state is a cycle, and a cycle is
resolved by adopting its numerically minimal member and SAYING SO
(``shot-unhonored(reason="oscillation")``), because refusing at the seed would
make the answer depend on which member the seed happened to be.

Every veto and every refusal is a typed report (:data:`REPORT_KINDS`, closed),
and reports are facts about the seed plus the terminal state rather than a
per-sweep log -- which is what makes the report set a function of the seed and
N8a determinism cheap to hold.
"""

from __future__ import annotations

import bisect
import copy
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from .authority import (
    AuthorityKind,
    AuthorityLedger,
    Capability,
)
from .canonical_text import (
    CanonicalWork,
    FinalText,
    canonical_text,
    line_budget,
    over_wide_token,
)
from .partition_check import EPS, ReportTag, Waiver
from .schema import Cue, Unit
from .segdoc import DisplayProfile
from .timing import (
    CHAIN_MAX_GAP_S,
    HELD_WORD_MAX_GAP_S,
    LINGER_CAP_S,
    TWO_FRAME_S,
    _EPS,
    _FRAME_S,
    _SHOT_LANDING_S,
)
from .timing_preview import CueCandidate, CuePreview

if TYPE_CHECKING:  # pragma: no cover - typing only, never a runtime import
    from .boundary_lattice import Edge, LatticeAtom
    from .boundary_v2 import DocumentSolution, IntervalSolution
    from .segdoc import SegDocument, SourceUnit

__all__ = [
    "DELTA_IDS",
    "FINALIZER_WAIVER_KINDS",
    "REPORT_KINDS",
    "RULE_IDS",
    "SWEEP_BUDGET",
    "TERMINALS",
    "BoundaryMovement",
    "BoundaryRef",
    "CycleEvidence",
    "FinalizeEvidence",
    "FinalizePolicy",
    "FinalizeReport",
    "FinalizeResult",
    "FinalizerPreview",
    "NeighbourRead",
    "NonFiniteTime",
    "OptimizerSelectionAuthority",
    "Phase1Cue",
    "Phase1CueStream",
    "Trace",
    "TraceLeg",
    "V1ReferenceCapture",
    "apply_sweep",
    "capture_v1_reference",
    "finalize",
    "normalize_time",
    "pack_state",
    "phase1_cue",
    "phase1_from_optimizer_selection",
    "phase1_from_v1_capture",
    "phase1_stream",
    "register_optimizer_selection",
    "replay_trace",
    "seed_reports",
    "state_key",
]

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

#: An EXPLICIT operational policy, not a theorem (spec section 2.3). Read from
#: this module global at call time -- binding it as a default argument would let
#: a budget fixture pass while measuring the shipped value.
SWEEP_BUDGET: int = 10_000

#: Registry rows this module can fire (spec section 9).
DELTA_IDS: tuple[str, ...] = (
    "FD-1",
    "FD-2",
    "FD-3",
    "FD-4",
    "FD-6",
    "FD-7",
    "FD-8",
    "FD-9",
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
    from .layout import _reading_chars

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


# ------------------------------------------------------------------ sweep rules


def _nearest(shots: Sequence[float], value: float, snap_s: float) -> float | None:
    """``_snap_to_shots``' cut chooser: nearest cut inside the window, earlier wins.

    Single-valued by construction -- the earlier candidate is replaced only on a
    strictly smaller distance, so an exactly equidistant pair keeps the earlier.
    """
    position = bisect.bisect_left(shots, value)
    best: float | None = None
    for index in (position - 1, position):
        if 0 <= index < len(shots) and abs(shots[index] - value) <= snap_s:
            if best is None or abs(shots[index] - value) < abs(best - value):
                best = shots[index]
    return best


def _guarded_end(want: float, seed_end: float, next_start: float | None) -> float:
    """Slot 1's neighbour guard, against the SEED end (spec section 3.4).

    ``_cleanup_cues`` reads ``c["end"]`` at this statement, which in the pass it
    lives in is the untouched input end; reading the *current* end here instead
    would stack this sweep's desire on the previous sweep's grant, which is the
    ratchet review 4 measured (an anchorless cue growing 0.25 s per run).

    The ``else`` arm of the inner test is the 87fde9d gap preservation: a gap
    already at or under the two-frame floor is never extended into, because the
    chaining branch could not restore it afterwards.
    """
    if want > seed_end:
        if next_start is None:
            return want
        if next_start - seed_end > TWO_FRAME_S:
            return min(want, next_start)
        return seed_end
    return seed_end


def _chain_end(end: float, next_start: float | None) -> float:
    """Slot 2: close a sub-half-second inter-cue gap down to two frames."""
    if next_start is None:
        return end
    gap = next_start - end
    if 0 <= gap < CHAIN_MAX_GAP_S and gap > TWO_FRAME_S:
        return next_start - TWO_FRAME_S
    return end


def _cap_end(
    start: float,
    end: float,
    word_data: Sequence[Unit],
    next_start: float | None,
    max_cue_s: float,
) -> tuple[float, bool]:
    """Slot 3: the duration cap and its held-chain waiver, verbatim.

    Returns ``(end, waived)``. A word still sounding past the cap may hold the
    cue, but only across CONTINUOUS speech: the walk stops at the first silence
    wider than ``HELD_WORD_MAX_GAP_S``, so a sung sustain stays visible while a
    stray syllable across dead air does not drag the cue to itself.
    """
    if not max_cue_s or end - start <= max_cue_s:
        return end, False
    cap = start + max_cue_s
    timed = sorted(
        (
            (s, e)
            for unit in word_data
            if (s := unit.get("start")) is not None
            and (e := unit.get("end")) is not None
        ),
        key=lambda unit: unit[0],
    )
    last_word_end = max((e for _s, e in timed), default=None)
    if last_word_end is None or last_word_end <= cap:
        return cap, False
    held_end = timed[0][1]
    for (_ps, pe), (ns, ne) in zip(timed, timed[1:]):
        if ns - pe > HELD_WORD_MAX_GAP_S:
            break
        held_end = ne
    target = held_end
    if next_start is not None:
        target = min(target, next_start)
    capped = max(cap, target)
    return capped, capped > cap


def _within_cap(start: float, end: float, cap: float | None) -> bool:
    """``_snap_to_shots``' cap test; ``None`` is production's falsy-cap convention."""
    return cap is None or end - start <= cap + _EPS


def _shot_in(
    start: float,
    end: float,
    *,
    speech_start: float | None,
    prev_end: float | None,
    shots: Sequence[float],
    snap_s: float,
    cap: float | None,
) -> float:
    """Slot 4: the Netflix in-time zones, including the #24 speech-start veto.

    A delaying move that would land past the cue's own first word is skipped
    WHOLE rather than clamped to the word: an off-zone landing trades one rule
    break for another, and speech beats layout.
    """
    cut = _nearest(shots, start, snap_s)
    if cut is None or abs(cut - start) <= _EPS:
        return start
    offset = start - cut
    if -7 * _FRAME_S - _EPS <= offset < 0:
        new_start = cut
    elif offset < 0:
        new_start = cut - _SHOT_LANDING_S
    elif offset <= 9 * _FRAME_S + _EPS:
        new_start = cut
    else:
        new_start = cut + _SHOT_LANDING_S
    if prev_end is not None:
        new_start = max(new_start, prev_end + TWO_FRAME_S)
    truncates = (
        new_start > start
        and speech_start is not None
        and new_start > speech_start + _EPS
    )
    if (
        new_start < end - TWO_FRAME_S
        and (new_start <= start or new_start - start <= _SHOT_LANDING_S)
        and _within_cap(new_start, end, cap)
        and not truncates
    ):
        return new_start
    return start


def _shot_out(
    start: float,
    end: float,
    *,
    speech_end: float | None,
    next_start: float | None,
    shots: Sequence[float],
    snap_s: float,
    cap: float | None,
) -> float:
    """Slot 5: the out-time zones, the pull-back veto and its last resort.

    An anchorless cue uses its display end as the effective speech end, exactly
    as production does: with no acoustic evidence there is nothing for the veto
    to protect.
    """
    cut = _nearest(shots, end, snap_s)
    if cut is None:
        return end
    effective_speech_end = end if speech_end is None else speech_end
    offset = end - cut
    if offset <= 5 * _FRAME_S + _EPS:
        target = cut - TWO_FRAME_S
    else:
        target = cut + _SHOT_LANDING_S
    applied = False
    if target > end + _EPS:
        if (next_start is None or target <= next_start - TWO_FRAME_S) and _within_cap(
            start, target, cap
        ):
            end = target
            applied = True
    elif target < end - _EPS:
        if target >= effective_speech_end and target > start:
            end = target
            applied = True
    if not applied and 0 < offset <= 5 * _FRAME_S + _EPS:
        target = cut + _SHOT_LANDING_S
        if (
            target > end
            and (next_start is None or target <= next_start - TWO_FRAME_S)
            and _within_cap(start, target, cap)
        ):
            end = target
    return end


def _ladder(
    prev_end: float, prev_speech_end: float | None, next_start: float
) -> tuple[int, float] | None:
    """Slot 6: the overlap ladder (spec section 2.4). ``None`` = nothing to do.

    The trigger owns the WHOLE sub-two-frame band, negative gaps included, and
    is the exact negation of the validator's ``min-gap`` predicate -- solver and
    validator can therefore never disagree about the same gap, and a gap already
    chained to ``next.start - 2f`` (float-inexact by ~1e-16) does not re-fire.

    Every branch is TRIM-ONLY: the ladder shortens the left cue and never
    extends an end onto its own speech.
    """
    if next_start - prev_end >= TWO_FRAME_S - EPS:
        return None
    if prev_speech_end is None or prev_speech_end <= next_start - TWO_FRAME_S:
        return 1, next_start - TWO_FRAME_S
    if prev_speech_end <= next_start:
        return 2, min(prev_end, prev_speech_end)
    return 3, min(prev_end, prev_speech_end)


def apply_sweep(
    state: StreamState,
    cues: Sequence[Phase1Cue],
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
    policy: FinalizePolicy,
    sweep: int = 1,
) -> tuple[StreamState, tuple[TraceLeg, ...]]:
    """ONE deterministic full pass ``S``: cues ascending, all six slots per cue.

    Per-cue interleaving is not a detail. Production processes a cue's end
    before advancing to the next cue's start, so the next start reads a floor
    that is already final; a start-first/end-second global order reads a stale
    floor and lands somewhere neither run agrees with (review 3, finding 1).

    Exposed so the validator's stability check can re-run exactly one sweep on
    the delivered stream.
    """
    del policy  # P5 has one policy value; the parameter is the P6 seam.
    starts = [pair[0] for pair in state]
    ends = [pair[1] for pair in state]
    legs: list[TraceLeg] = []
    shots = tuple(sorted(evidence.shots))
    snap_s = profile.shot_snap_s
    snapping = snap_s > 0 and bool(shots)
    cap = profile.max_cue_s if profile.max_cue_s else None
    count = len(cues)

    def emit(
        rule_id: str,
        slot: int,
        cue_index: int,
        target: BoundaryRef,
        from_value: float,
        to_value: float,
        reads: tuple[NeighbourRead, ...],
    ) -> None:
        if from_value == to_value:
            return
        legs.append(
            TraceLeg(
                rule_id=rule_id,
                sweep=sweep,
                cue_index=cue_index,
                slot=slot,
                target=target,
                from_value=from_value,
                to_value=to_value,
                reads=reads,
            )
        )

    for index, cue in enumerate(cues):
        own_end = BoundaryRef(index, "end")
        next_start = starts[index + 1] if index + 1 < count else None
        next_read = (
            ()
            if next_start is None
            else (NeighbourRead(BoundaryRef(index + 1, "start"), next_start),)
        )
        own_start_read = (NeighbourRead(BoundaryRef(index, "start"), starts[index]),)

        # slot 1 -- absolute duration desire, guarded against the seed end
        want = _duration_desire(
            start=starts[index],
            seed_end=cue.seed_end,
            speech_end=cue.speech_end,
            reading_chars=cue.reading_chars,
            profile=profile,
        )
        new_end = _guarded_end(want, cue.seed_end, next_start)
        emit(
            "duration-desire",
            1,
            index,
            own_end,
            ends[index],
            new_end,
            own_start_read + next_read,
        )
        ends[index] = new_end

        # slot 2 -- chaining
        new_end = _chain_end(ends[index], next_start)
        emit("chain", 2, index, own_end, ends[index], new_end, next_read)
        ends[index] = new_end

        # slot 3 -- duration cap with its held-chain waiver
        new_end, _waived = _cap_end(
            starts[index], ends[index], cue.word_data, next_start, profile.max_cue_s
        )
        emit("cap", 3, index, own_end, ends[index], new_end, own_start_read + next_read)
        ends[index] = new_end

        if snapping:
            # slot 4 -- in-time zones
            prev_end = ends[index - 1] if index > 0 else None
            prev_read = (
                ()
                if prev_end is None
                else (NeighbourRead(BoundaryRef(index - 1, "end"), prev_end),)
            )
            new_start = _shot_in(
                starts[index],
                ends[index],
                speech_start=cue.speech_start,
                prev_end=prev_end,
                shots=shots,
                snap_s=snap_s,
                cap=cap,
            )
            emit(
                "shot-in",
                4,
                index,
                BoundaryRef(index, "start"),
                starts[index],
                new_start,
                prev_read + (NeighbourRead(own_end, ends[index]),),
            )
            starts[index] = new_start

            # slot 5 -- out-time zones
            new_end = _shot_out(
                starts[index],
                ends[index],
                speech_end=cue.speech_end,
                next_start=next_start,
                shots=shots,
                snap_s=snap_s,
                cap=cap,
            )
            emit(
                "shot-out",
                5,
                index,
                own_end,
                ends[index],
                new_end,
                (NeighbourRead(BoundaryRef(index, "start"), starts[index]),)
                + next_read,
            )
            ends[index] = new_end

        # slot 6 -- the overlap ladder, which writes the PREVIOUS cue's end
        if index > 0:
            outcome = _ladder(
                ends[index - 1], cues[index - 1].speech_end, starts[index]
            )
            if outcome is not None:
                branch, trimmed = outcome
                emit(
                    f"ladder-{branch}",
                    6,
                    index,
                    BoundaryRef(index - 1, "end"),
                    ends[index - 1],
                    trimmed,
                    (NeighbourRead(BoundaryRef(index, "start"), starts[index]),),
                )
                ends[index - 1] = trimmed

    return tuple(zip(starts, ends)), tuple(legs)


# ------------------------------------------------------------ report derivation


def _terminal_facts(
    cues: Sequence[Phase1Cue], state: StreamState, *, profile: DisplayProfile
) -> tuple[list[ReportTag], list[Waiver]]:
    """Reports and waivers that are functions of the TERMINAL state.

    Recomputed from the delivered stream rather than logged per sweep, which is
    what makes the report set a function of the seed alone. At a fixed point the
    rules are re-derivable by construction: re-applying the ladder or the cap to
    the delivered stream reproduces the same branch and the same value, so the
    ``min-gap-unmet`` evidence is the branch's own arithmetic, not a memory of
    which sweep minted it.
    """
    entries: list[ReportTag] = []
    waivers: list[Waiver] = []
    count = len(cues)
    for index, cue in enumerate(cues):
        start, end = state[index]
        if profile.min_cue_s > 0 and end - start < profile.min_cue_s - EPS:
            entries.append(
                ReportTag(
                    kind="min-duration-short",
                    cue_index=index,
                    evidence={
                        "duration": end - start,
                        "end": end,
                        "min_cue_s": profile.min_cue_s,
                        "start": start,
                    },
                )
            )
        next_start = state[index + 1][0] if index + 1 < count else None
        _capped, waived = _cap_end(
            start, end, cue.word_data, next_start, profile.max_cue_s
        )
        if waived:
            entries_range = cue.unit_range
            waivers.append(
                Waiver(
                    kind="held-chain-duration",
                    cue_index=index,
                    unit_ids=()
                    if entries_range is None
                    else tuple(range(entries_range[0], entries_range[1])),
                    span=(cue.speech_start, cue.speech_end),
                    cap=profile.max_cue_s,
                    detail="a word is still sounding past the cap",
                )
            )
    for index in range(count - 1):
        outcome = _ladder(state[index][1], cues[index].speech_end, state[index + 1][0])
        if outcome is None or outcome[0] != 2:
            continue
        speech_end = cues[index].speech_end
        next_start = state[index + 1][0]
        assert speech_end is not None  # branch 2 is unreachable without an anchor
        entries.append(
            ReportTag(
                kind="min-gap-unmet",
                cue_index=index,
                evidence={
                    "next_start": next_start,
                    "prev_end_before": state[index][1],
                    "resulting_gap": next_start - speech_end,
                    "speech_end": speech_end,
                },
            )
        )
    return entries, waivers


def _deltas_fired(
    cues: Sequence[Phase1Cue],
    *,
    profile: DisplayProfile,
    seeds: Sequence[ReportTag],
    legs: Sequence[TraceLeg],
    entries: Sequence[ReportTag],
    seed_cues: Sequence[Cue],
) -> tuple[str, ...]:
    """Which registry rows this run could have exercised (spec section 9).

    Each id is decided by its own recomputable TRIGGER, never by whether the
    delivered value happened to differ -- the classifier recomputes eligibility
    itself and cross-checks this list, so a trigger that fired must be named
    here even when the two lanes happen to agree.
    """
    fired: set[str] = set()
    extends = profile.min_cue_s > 0 or profile.lag_out_s > 0 or profile.cps > 0
    for cue in cues:
        if cue.reading_chars != cue.raw_reading_chars:
            fired.add("FD-1")
        if cue.speech_end is None and extends:
            fired.add("FD-8")
        for tag in cue.reports:
            if tag.kind == "stutter-not-proven-fixed-within-4-scans":
                fired.add("FD-9")
    if any(tag.kind == "input-overlap" for tag in seeds):
        fired.add("FD-3")
    for index in range(len(seed_cues) - 1):
        if seed_cues[index + 1]["start"] - seed_cues[index]["end"] < TWO_FRAME_S:
            fired.add("FD-6")
    if any(
        leg.rule_id in ("chain", "shot-in", "shot-out")
        or leg.rule_id.startswith("ladder-")
        for leg in legs
    ):
        fired.add("FD-4")
    if seeds or entries:
        fired.add("FD-7")
    return tuple(sorted(fired))


def _sort_reports(tags: Sequence[ReportTag]) -> tuple[ReportTag, ...]:
    """By ``(cue_index, kind)``, document-level facts first; stable within a pair."""
    return tuple(
        sorted(
            tags,
            key=lambda tag: (-1 if tag.cue_index is None else tag.cue_index, tag.kind),
        )
    )


# ------------------------------------------------------------------ the solver


def _solve(
    seed: StreamState,
    cues: Sequence[Phase1Cue],
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
    policy: FinalizePolicy,
) -> tuple[StreamState, Trace]:
    """Iterate ``S`` from the phase-1 state to a fixed point, a cycle or the budget."""
    budget = SWEEP_BUDGET
    state = seed
    order: list[StreamState] = [seed]
    seen: dict[bytes, list[StreamState]] = {pack_state(seed): [seed]}
    legs: list[TraceLeg] = []
    terminal: Terminal = "budget-exhausted"
    cycle: CycleEvidence | None = None
    sweeps = 0

    for sweep in range(1, budget + 1):
        sweeps = sweep
        moved, sweep_legs = apply_sweep(
            state, cues, profile=profile, evidence=evidence, policy=policy, sweep=sweep
        )
        legs.extend(sweep_legs)
        if moved == state:
            terminal = "fixed-point"
            break
        key = pack_state(moved)
        bucket = seen.get(key)
        # The packed bytes are an identity hash; a hit is settled by a full
        # compare so a collision can never be read as a cycle.
        if bucket is not None and any(member == moved for member in bucket):
            first = order.index(moved)
            members = tuple(order[first:])
            adopted = min(members, key=state_key)
            cycle = CycleEvidence(
                members=members,
                per_boundary_values=_cycle_values(members),
                adopted=adopted,
            )
            terminal = "cycle-adoption"
            state = adopted
            break
        seen.setdefault(key, []).append(moved)
        order.append(moved)
        state = moved
    else:
        terminal = "budget-exhausted"

    return state, Trace(legs=tuple(legs), terminal=terminal, cycle=cycle, sweeps=sweeps)


def _cycle_values(
    members: Sequence[StreamState],
) -> tuple[tuple[BoundaryRef, tuple[float, ...]], ...]:
    """Every boundary's value sequence across the cycle, in visit order."""
    values: list[tuple[BoundaryRef, tuple[float, ...]]] = []
    for index in range(len(members[0])):
        for slot, side in enumerate(("start", "end")):
            ref = BoundaryRef(index, "start" if side == "start" else "end")
            values.append((ref, tuple(member[index][slot] for member in members)))
    return tuple(values)


def _oscillation_reports(cycle: CycleEvidence) -> list[ReportTag]:
    """``shot-unhonored`` for every boundary that is not constant across the cycle."""
    tags: list[ReportTag] = []
    for ref, values in cycle.per_boundary_values:
        distinct = sorted(set(values))
        if len(distinct) < 2:
            continue
        tags.append(
            ReportTag(
                kind="shot-unhonored",
                cue_index=ref.cue_index,
                evidence={
                    "boundary": ref.side,
                    "reason": "oscillation",
                    "values": distinct,
                },
            )
        )
    return tags


def finalize(
    stream: Phase1CueStream,
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
    policy: FinalizePolicy,
) -> FinalizeResult:
    """Consume the seed's capability, solve, and deliver cues + report + trace.

    Raises :class:`~voxweave.core.authority.UnissuedAuthority` when the stream's
    seal is not one its ledger issued, :class:`SealBroken` when the sealed
    payload was mutated after issuance, :class:`CapabilityConsumed` on a second
    use, and :class:`NonFiniteTime` from the preflight.
    """
    stream.capability.consume(
        _stream_payload(
            stream.cues, stream.profile, stream.row_id, stream.evaluation_id
        )
    )
    shots = tuple(sorted(normalize_time(shot) for shot in evidence.shots))
    checked = FinalizeEvidence(shots=shots, sing_spans=evidence.sing_spans)

    cues = stream.cues
    seed = tuple((cue.start, cue.end) for cue in cues)
    delivered, trace = _solve(
        seed, cues, profile=profile, evidence=checked, policy=policy
    )

    seed_cues = tuple(_seed_cue(cue) for cue in cues)
    seeded = seed_reports(seed_cues)
    entries, waivers = _terminal_facts(cues, delivered, profile=profile)
    for cue in cues:
        entries.extend(cue.reports)
    entries.extend(seeded)
    if trace.cycle is not None:
        entries.extend(_oscillation_reports(trace.cycle))
    if trace.terminal == "budget-exhausted":
        entries.append(
            ReportTag(
                kind="solver-budget-exhausted",
                cue_index=None,
                evidence={"sweeps": trace.sweeps},
            )
        )

    movement: list[BoundaryMovement] = []
    max_start_movement = 0.0
    for index, cue in enumerate(cues):
        start, end = delivered[index]
        movement.append(BoundaryMovement(BoundaryRef(index, "start"), cue.start, start))
        movement.append(BoundaryMovement(BoundaryRef(index, "end"), cue.end, end))
        max_start_movement = max(max_start_movement, abs(start - cue.start))

    report = FinalizeReport(
        entries=_sort_reports(entries),
        waivers=tuple(waivers),
        deltas_fired=_deltas_fired(
            cues,
            profile=profile,
            seeds=seeded,
            legs=trace.legs,
            entries=entries,
            seed_cues=seed_cues,
        ),
        movement=tuple(movement),
        max_start_movement_s=max_start_movement,
        max_sweeps_observed=trace.sweeps,
        terminal=trace.terminal,
    )
    return FinalizeResult(
        cues=tuple(
            _deliver(cue, delivered[index][0], delivered[index][1])
            for index, cue in enumerate(cues)
        ),
        report=report,
        trace=trace,
        valid=trace.terminal != "budget-exhausted",
    )


def _seed_cue(cue: Phase1Cue) -> Cue:
    """The immutable input span of one phase-1 cue, as a cue dict."""
    return {
        "text": cue.text,
        "start": cue.seed_start,
        "end": cue.seed_end,
        "word_data": cue.word_data,
        "speech_start": cue.speech_start,
        "speech_end": cue.speech_end,
    }


def _deliver(cue: Phase1Cue, start: float, end: float) -> Cue:
    """One delivered cue. ``word_data`` rides through by reference, never rebuilt."""
    out: Cue = {
        "text": cue.text,
        "start": start,
        "end": end,
        "word_data": cue.word_data,
        "speech_start": cue.speech_start,
        "speech_end": cue.speech_end,
    }
    if cue.lyric is not None:
        out["lyric"] = cue.lyric
    return out


# ------------------------------------------------------------ the trace validator


def replay_trace(
    trace: Trace,
    seed: Sequence[Phase1Cue],
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
    policy: FinalizePolicy,
    delivered: StreamState,
) -> tuple[str, ...]:
    """Verify one trace (spec section 10.2). Empty tuple == pass.

    The verification itself lives in :mod:`voxweave.core.trace_validator`, which
    reimplements every rule from ``timing.py``'s own statements rather than from
    this module's ports of them. A producer that validated its own trace with its
    own helpers would prove self-consistency and nothing else; the separation is
    the point, and this function is only the name the spec froze for it.

    Imported inside the body because the validator imports this module: the
    dependency runs validator -> producer, and only the convenience alias runs
    back the other way.
    """
    from .trace_validator import replay_trace as _validate

    return _validate(
        trace,
        seed,
        profile=profile,
        evidence=evidence,
        policy=policy,
        delivered=delivered,
    )


# ----------------------------------------------------------------- the preview


def _word_data_speech(word_data: Sequence[Unit]) -> tuple[float | None, float | None]:
    """Speech bounds derived from ``word_data`` -- the scalar seam's only source."""
    starts = [s for unit in word_data if (s := unit.get("start")) is not None]
    ends = [e for unit in word_data if (e := unit.get("end")) is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)


@dataclass(frozen=True)
class FinalizerPreview:
    """Computes EXACTLY phase 1; phase 2 is excluded from the preview by contract.

    The optimizer scores an edge before any neighbour exists, so promising
    anything the sweep does would be promising a number this object cannot
    compute. What it does promise -- the canonical text, its line count, its
    reading load and the absolute duration desire -- is float-exactly what the
    finalizer will start from (gate N7).

    ``profile`` is needed only by the scalar :meth:`preview_display_span`
    compatibility method, which receives loose thresholds rather than a profile.
    """

    profile: DisplayProfile | None = None

    def preview_cue(self, candidate: CueCandidate) -> CuePreview:
        profile = candidate.profile
        if candidate.start is None or candidate.end is None:
            final = canonical_text(
                candidate.word_data,
                fallback_text=candidate.text,
                lang=profile.language,
                profile=profile,
                expected_footprint=candidate.expected_footprint,
            )
            return CuePreview(
                display_start=candidate.start,
                display_end=None,
                final_text=final.text,
                line_count=len(final.lines),
                reading_chars=final.reading_chars,
                waivers=(),
                refusals=tuple(_text_reports(final, profile, 0)),
            )
        cue: Cue = {
            "text": candidate.text,
            "start": candidate.start,
            "end": candidate.end,
            "word_data": list(candidate.word_data),
            "speech_start": candidate.speech_start,
            "speech_end": candidate.speech_end,
        }
        built = phase1_cue(
            cue,
            profile=profile,
            index=0,
            expected_footprint=candidate.expected_footprint,
        )
        return CuePreview(
            display_start=built.start,
            display_end=built.end,
            final_text=built.text,
            line_count=len(built.lines),
            reading_chars=built.reading_chars,
            waivers=(),
            refusals=built.reports,
        )

    def preview_display_span(
        self,
        start: float,
        end: float,
        next_start: float | None,
        *,
        text: str,
        word_data: Sequence[Unit],
        min_cue_s: float,
        max_cue_s: float,
        cps: float = 0.0,
        lag_out_s: float = 0.0,
    ) -> float:
        """:class:`DisplayTimingPreview` compatibility: ``preview_cue(...).available_s``.

        The scalar seam carries no cue keys, so the anchors are derived from
        ``word_data`` here -- the one place this module does that, and only
        because the caller has nothing else to hand over.
        """
        if self.profile is None:
            raise ValueError(
                "FinalizerPreview.preview_display_span needs a profile; "
                "call preview_cue with a CueCandidate instead"
            )
        speech_start, speech_end = _word_data_speech(word_data)
        candidate = CueCandidate(
            start=start,
            end=end,
            next_start=next_start,
            text=text,
            word_data=word_data,
            speech_start=speech_start,
            speech_end=speech_end,
            profile=replace(
                self.profile,
                min_cue_s=min_cue_s,
                max_cue_s=max_cue_s,
                cps=cps,
                lag_out_s=lag_out_s,
            ),
        )
        return self.preview_cue(candidate).available_s


# ---------------------------------------------------------------- the factories


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


@dataclass(frozen=True)
class V1ReferenceCapture:
    """The hook's retained deep copy, sealed AT CAPTURE TIME.

    Deep-copied because the reference stream keeps being read by the legacy
    lanes: sealing the caller's own list would seal an object that legitimately
    changes underneath, and then either the seal or the lane would be lying.
    """

    cues: tuple[Cue, ...]
    capability: Capability


def capture_v1_reference(
    cues: Sequence[Cue], *, ledger: AuthorityLedger
) -> V1ReferenceCapture:
    """Seal v1's committed cue stream as the ``v1-capture`` authority."""
    frozen = tuple(cast("Cue", copy.deepcopy(dict(cue))) for cue in cues)
    capability = ledger.issue(
        issuer="voxweave.core.finalizer.capture_v1_reference",
        kind="v1-capture",
        payload=_cue_payload(frozen),
    )
    return V1ReferenceCapture(cues=frozen, capability=capability)


@dataclass(frozen=True)
class OptimizerSelectionAuthority:
    """One row's OWN optimizer selection, sealed at registration.

    Holds the edges and atoms rather than the materialized cues, because
    ``result.cues`` is a mutable list nobody can prove was not edited between
    selection and finalization -- and "the stream descended from this row's own
    selection" is exactly the claim N8b makes.
    """

    document: SegDocument
    partition: tuple[int, ...]
    edges: tuple[Edge, ...]
    atoms: tuple[LatticeAtom, ...]
    fallback_start: float
    capability: Capability


def register_optimizer_selection(
    solution: DocumentSolution, *, ledger: AuthorityLedger
) -> OptimizerSelectionAuthority:
    """Seal a document's optimizer selection as the ``optimizer-selection`` authority.

    The document's atoms and selected edges are flattened into one chain with
    node offsets applied, which reproduces ``optimize_document``'s own
    per-interval materialization exactly: that loop chains each interval's
    ``fallback_start`` from the previous interval's last cue end, and
    ``materialize_cues`` chains the same value internally.

    An interval that adopted v1 has no optimizer selection to seal -- its cues
    are v1's own dicts -- so registering such a document is refused rather than
    laundered through this factory. The gated rows this factory serves assert
    ``adopted_v1 == 0`` (gate N4), so a refusal here is a broken precondition,
    not a supported mode.
    """
    from dataclasses import replace as _replace

    edges: list[Edge] = []
    atoms: list[LatticeAtom] = []
    for interval in solution.solutions:
        if interval.adopted is not None or interval.selection is None:
            raise ValueError(
                f"interval {interval.interval.index} adopted v1; a typed fallback "
                "carries no optimizer selection to seal"
            )
        offset = len(atoms)
        atoms.extend(interval.lattice.atoms)
        for edge in _selected_edges(interval):
            edges.append(
                _replace(
                    edge,
                    start_node=edge.start_node + offset,
                    end_node=edge.end_node + offset,
                )
            )
    cuts = _document_cuts(solution)
    fallback_start = 0.0
    capability = ledger.issue(
        issuer="voxweave.core.finalizer.register_optimizer_selection",
        kind="optimizer-selection",
        payload=_selection_payload(
            edges,
            atoms,
            cuts,
            _profile_payload(solution.document.profile),
            solution.document.units,
            fallback_start,
        ),
    )
    return OptimizerSelectionAuthority(
        document=solution.document,
        partition=cuts,
        edges=tuple(edges),
        atoms=tuple(atoms),
        fallback_start=fallback_start,
        capability=capability,
    )


def _selection_payload(
    edges: Sequence[Edge],
    atoms: Sequence[LatticeAtom],
    partition: Sequence[int],
    profile: dict[str, Any] | None,
    units: Sequence[SourceUnit],
    fallback_start: float,
) -> dict[str, Any]:
    """The projection an optimizer selection is sealed over.

    ONE definition, because the issuing side and the verifying side must digest
    the same bytes: two hand-written copies of this dict would be a seal that
    breaks the first time one of them is edited.
    """
    return {
        # These five atom fields are the complete projection read by
        # ``materialize_cues``.  Footprints are load-bearing for W2's
        # provenance-aware acoustic anchors and therefore cannot remain mutable
        # outside the same seal as the selected edge chain.
        "atoms": [
            [atom.text, atom.start, atom.end, atom.unit_start, atom.unit_end]
            for atom in atoms
        ],
        "edges": [
            [
                edge.start_node,
                edge.end_node,
                edge.span_start,
                edge.span_end,
                edge.input_start,
                edge.input_end,
                None
                if edge.evidence_span is None
                else [
                    edge.evidence_span.start,
                    edge.evidence_span.end,
                    edge.evidence_span.start_kind,
                    edge.evidence_span.end_kind,
                ],
                edge.lyric,
            ]
            for edge in edges
        ],
        "fallback_start": fallback_start,
        "partition": list(partition),
        "profile": profile,
        # W2's materializer reads endpoint provenance from the registered
        # document.  Those records therefore join the seal: otherwise a caller
        # could mutate provenance after issuance and change the phase-1 anchors
        # without breaking the capability digest.
        "units": [
            [
                unit.id,
                unit.surface,
                unit.start,
                unit.end,
                unit.provenance,
                unit.confidence,
            ]
            for unit in units
        ],
    }


def _selected_edges(interval: IntervalSolution) -> tuple[Edge, ...]:
    """The edge chain one interval committed to, from its own lattice.

    The POLICY-selected path, not the raw optimum: what a row delivered is what
    its policy shipped, and sealing the other one would seal a stream nobody ran.
    """
    from .boundary_v2 import _path_edges

    assert interval.selection is not None  # guarded by the caller
    return _path_edges(interval.lattice, interval.selection.policy_selected.cuts)


def _document_cuts(solution: DocumentSolution) -> tuple[int, ...]:
    """The document's interior cut list in source-unit space."""
    from .boundary_v2 import _document_partition

    return tuple(_document_partition(solution.solutions, len(solution.document.units)))


def _mint_stream(
    cues: tuple[Phase1Cue, ...],
    *,
    profile: DisplayProfile,
    ledger: AuthorityLedger,
    row_id: str,
    evaluation_id: str,
    kind: AuthorityKind,
    issuer: str,
    input_seed_id: str,
) -> Phase1CueStream:
    """Seal one phase-1 stream and record its root event."""
    capability = ledger.issue(
        issuer=issuer,
        kind=kind,
        payload=_stream_payload(cues, profile, row_id, evaluation_id),
    )
    from .authority import FactoryEvent

    ledger.record(
        FactoryEvent(
            evaluation_id=evaluation_id,
            row_id=row_id,
            call_id=capability.seal.authority_id,
            input_seed_id=input_seed_id,
            input_kind="phase1",
            parent_finalize_call_id=None,
            authority_kind=kind,
            authority_id=capability.seal.authority_id,
        )
    )
    return Phase1CueStream(
        cues=cues,
        profile=profile,
        seed_id=capability.seal.authority_id,
        row_id=row_id,
        evaluation_id=evaluation_id,
        authority_kind=kind,
        capability=capability,
    )


def phase1_from_v1_capture(
    capture: V1ReferenceCapture,
    *,
    profile: DisplayProfile,
    ledger: AuthorityLedger,
    row_id: str,
    evaluation_id: str,
) -> Phase1CueStream:
    """Mint the ``v1`` row's root from the hook's sealed reference capture."""
    seal = capture.capability.consume(_cue_payload(capture.cues))
    cues = phase1_stream(capture.cues, profile=profile)
    return _mint_stream(
        cues,
        profile=profile,
        ledger=ledger,
        row_id=row_id,
        evaluation_id=evaluation_id,
        kind="v1-capture",
        issuer="voxweave.core.finalizer.phase1_from_v1_capture",
        input_seed_id=seal.authority_id,
    )


def phase1_from_optimizer_selection(
    authority: OptimizerSelectionAuthority,
    *,
    ledger: AuthorityLedger,
    row_id: str,
    evaluation_id: str,
) -> Phase1CueStream:
    """Mint an optimizer row's root by REMATERIALIZING from the sealed selection.

    Never from a stored cue list: the cues are rebuilt here from the registered
    edges and atoms, so a cue dict edited after selection cannot become the seed.
    """
    from .boundary_v2 import materialize_cues

    seal = authority.capability.consume(
        _selection_payload(
            authority.edges,
            authority.atoms,
            authority.partition,
            _profile_payload(authority.document.profile),
            authority.document.units,
            authority.fallback_start,
        )
    )
    profile = authority.document.profile
    cues = materialize_cues(
        authority.edges,
        authority.atoms,
        profile.language,
        fallback_start=authority.fallback_start,
        units=authority.document.units,
    )
    bounds = (0, *authority.partition, len(authority.document.units))
    unit_ranges = [
        (bounds[index], bounds[index + 1]) for index in range(len(bounds) - 1)
    ]
    stream = phase1_stream(
        cues,
        profile=profile,
        unit_ranges=unit_ranges if len(unit_ranges) == len(cues) else None,
    )
    return _mint_stream(
        stream,
        profile=profile,
        ledger=ledger,
        row_id=row_id,
        evaluation_id=evaluation_id,
        kind="optimizer-selection",
        issuer="voxweave.core.finalizer.phase1_from_optimizer_selection",
        input_seed_id=seal.authority_id,
    )
