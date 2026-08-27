"""The finalizer's trace, checked by arithmetic that does not share its code.

A producer that emits its own trace and then verifies it with its own helpers
proves only that it is self-consistent. Review 6 built the shape that defeats
that: a defective producer moves a boundary to the wrong value and attaches a
neighbour snapshot under which the wrong value is the *right* answer. Recompute
the rule from the recorded snapshot and the leg checks out; recompute it from
the stream the document actually had, and it does not.

So this module holds a second rule engine, written from ``timing.py``'s own
statements rather than from :mod:`voxweave.core.finalizer`'s ports of them, and
a replay that:

* **reconstructs the phase-1 state itself** from each cue's immutable seed
  record instead of reading the state the producer published (a disagreement is
  a reported failure, not an assumption);
* checks every leg's ``from`` and **every neighbour it claims to have read**
  against the validator's own evolving state -- the review-6 hole;
* applies the rule with this module's arithmetic and requires equality;
* requires the replayed final state to equal the delivered stream bit-exactly,
  and -- for a fixed-point terminal -- independently requires that stream to be
  a fixed point of this module's sweep, which is what catches a rule the
  producer silently declined to run (an omitted leg replays to the same state
  and would otherwise pass);
* checks a cycle structurally: every declared transition, and minimality over
  the declared member set.

Independence is claimed for the rule arithmetic and for the state
reconstruction. It is NOT claimed for the canonical text: ``reading_chars`` is
read from the phase-1 record, because the canonical projection is a single
authority by design (spec section 3) and a second one would be a second answer
to a question that must have one. :func:`stability_check` is likewise common-mode
by construction and says so.
"""

from __future__ import annotations

import bisect
import operator
from collections.abc import Sequence
from itertools import pairwise

from .finalizer import (
    RULE_IDS,
    FinalizeEvidence,
    FinalizePolicy,
    Phase1Cue,
    StreamState,
    Terminal,
    Trace,
    apply_sweep,
    state_key,
)
from .partition_check import EPS
from .schema import Unit
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

__all__ = [
    "capped_end",
    "chained_end",
    "desired_end",
    "guarded_end",
    "ladder_branch",
    "nearest_cut",
    "reconstruct_phase1",
    "replay_trace",
    "snapped_end",
    "snapped_start",
    "stability_check",
    "sweep",
    "within_cap",
]


# ------------------------------------------------------- the second rule engine


def desired_end(
    *,
    start: float,
    seed_end: float,
    speech_end: float | None,
    reading_chars: int,
    profile: DisplayProfile,
) -> float:
    """The absolute end a cue wants (phase 1, statements D1-D10).

    Written as a candidate set because every step of the pseudocode is a
    selection between exactly-computed values: ``max`` over the whole set picks
    the same float as the nested ``max`` chain, so the shape can differ while
    the arithmetic cannot. Each candidate keeps its operand order verbatim --
    ``start + min_cue_s``, ``speech_end + lag_out_s``, ``start + need`` -- since
    those ARE arithmetic and reassociating them would move the last bit.

    The anchorless branch is total (FD-8): with no speech end there is no
    candidate at all, the min-duration floor included.
    """
    if speech_end is None:
        return seed_end
    candidates = [seed_end]
    if profile.min_cue_s > 0:
        candidates.append(start + profile.min_cue_s)
    if profile.lag_out_s > 0:
        candidates.append(speech_end + profile.lag_out_s)
    if profile.cps > 0:
        need = reading_chars / profile.cps
        candidates.append(min(start + need, speech_end + LINGER_CAP_S))
    return max(candidates)


def guarded_end(want: float, seed_end: float, next_start: float | None) -> float:
    """Slot 1's neighbour guard, expressed as its refusals (spec section 3.4).

    Three ways to keep the seed end -- the desire does not exceed it, or the
    inter-cue gap is already at or under the two-frame floor -- and one way to
    grow, clamped at the neighbour's start. The guard is measured against the
    SEED end, never the current one, so a sweep cannot stack this sweep's desire
    on the last sweep's grant.
    """
    if want <= seed_end:
        return seed_end
    if next_start is None:
        return want
    if next_start - seed_end <= TWO_FRAME_S:
        return seed_end
    return min(want, next_start)


def chained_end(end: float, next_start: float | None) -> float:
    """Slot 2: close an inter-cue gap wider than two frames but under half a second."""
    if next_start is None:
        return end
    gap = next_start - end
    if gap <= TWO_FRAME_S or gap >= CHAIN_MAX_GAP_S:
        return end
    return next_start - TWO_FRAME_S


def capped_end(
    start: float,
    end: float,
    word_data: Sequence[Unit],
    next_start: float | None,
    max_cue_s: float,
) -> tuple[float, bool]:
    """Slot 3: the duration cap and its held-chain waiver. Returns ``(end, waived)``.

    The walk stops at the first silence wider than ``HELD_WORD_MAX_GAP_S``: only
    speech CONTINUOUS with the cue's body may hold it past the cap, so a sung
    sustain stays visible while a stray syllable across dead air does not drag
    the cue to itself.
    """
    if max_cue_s == 0.0 or end - start <= max_cue_s:
        return end, False
    cap = start + max_cue_s
    spans = [
        (first, last)
        for unit in word_data
        if (first := unit.get("start")) is not None
        and (last := unit.get("end")) is not None
    ]
    spans.sort(key=operator.itemgetter(0))
    if not spans or max(last for _first, last in spans) <= cap:
        return cap, False
    held_end = spans[0][1]
    for (_before_start, before_end), (after_start, after_end) in pairwise(spans):
        if after_start - before_end > HELD_WORD_MAX_GAP_S:
            break
        held_end = after_end
    target = held_end if next_start is None else min(held_end, next_start)
    reached = max(cap, target)
    return reached, reached > cap


def within_cap(start: float, end: float, cap: float | None) -> bool:
    """The zone rules' cap test; ``None`` is production's falsy-cap convention."""
    return cap is None or end - start <= cap + _EPS


def nearest_cut(shots: Sequence[float], value: float, window: float) -> float | None:
    """The cut paired with a boundary: nearest inside ``window``, earlier on a tie.

    ``shots`` must be sorted. Only the two cuts bracketing ``value`` can win, so
    the search is the bisect neighbourhood; the earlier candidate is replaced
    only on a STRICTLY smaller distance, which is what makes an equidistant pair
    resolve to the earlier cut rather than to whichever the loop saw last.
    """
    position = bisect.bisect_left(shots, value)
    best: float | None = None
    for candidate in shots[max(position - 1, 0) : position + 1]:
        distance = abs(candidate - value)
        if distance > window:
            continue
        if best is None or distance < abs(best - value):
            best = candidate
    return best


def snapped_start(
    start: float,
    end: float,
    *,
    speech_start: float | None,
    prev_end: float | None,
    shots: Sequence[float],
    snap_s: float,
    cap: float | None,
) -> float:
    """Slot 4: the Netflix in-time zones, the previous-cue floor and the #24 veto.

    The zones split at the cut: before it, 1-7 frames land ON the cut and 8-11
    pull out to the free lead-in; after it, 1-9 frames go back onto the cut and
    10-11 push out to the landing zone. Written as the refusals that follow,
    because every one of them is a reason to keep the start where it is: the move
    would leave under two frames of cue, it would delay the text by over half a
    second, it would breach the duration cap, or -- the #24 veto -- it would land
    past the cue's own first word. That last one is skipped WHOLE rather than
    clamped to the word: an off-zone landing trades one rule break for another.
    """
    cut = nearest_cut(shots, start, snap_s)
    if cut is None or abs(cut - start) <= _EPS:
        return start
    offset = start - cut
    if offset < 0:
        landing = cut if offset >= -7 * _FRAME_S - _EPS else cut - _SHOT_LANDING_S
    else:
        landing = cut if offset <= 9 * _FRAME_S + _EPS else cut + _SHOT_LANDING_S
    if prev_end is not None:
        landing = max(landing, prev_end + TWO_FRAME_S)
    delays = landing > start
    if landing >= end - TWO_FRAME_S:
        return start
    if delays and landing - start > _SHOT_LANDING_S:
        return start
    if not within_cap(landing, end, cap):
        return start
    if delays and speech_start is not None and landing > speech_start + _EPS:
        return start
    return landing


def snapped_end(
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

    Up to 5 frames past the cut the cue dies two frames before it; further out it
    lands 12 frames after. A pull-back that would cut speech is vetoed, and the
    veto has a consequence rather than a shrug: the subtitle legitimately crosses
    the cut, so it goes to the 12-frames-after landing instead of flashing out
    just past it. An anchorless cue uses its display end as the effective speech
    end, exactly as production does -- with no acoustic evidence there is nothing
    for the veto to protect.
    """
    cut = nearest_cut(shots, end, snap_s)
    if cut is None:
        return end
    anchor = end if speech_end is None else speech_end
    offset = end - cut
    landing = (
        cut - TWO_FRAME_S if offset <= 5 * _FRAME_S + _EPS else cut + _SHOT_LANDING_S
    )
    if landing > end + _EPS:
        room = next_start is None or landing <= next_start - TWO_FRAME_S
        if room and within_cap(start, landing, cap):
            return landing
    elif landing < end - _EPS:
        if landing >= anchor and landing > start:
            return landing
    if 0 < offset <= 5 * _FRAME_S + _EPS:
        last_resort = cut + _SHOT_LANDING_S
        room = next_start is None or last_resort <= next_start - TWO_FRAME_S
        if last_resort > end and room and within_cap(start, last_resort, cap):
            return last_resort
    return end


def ladder_branch(
    prev_end: float, prev_speech_end: float | None, next_start: float
) -> tuple[int, float] | None:
    """Slot 6: the overlap ladder (spec section 2.4). ``None`` = nothing to do.

    The trigger owns the whole sub-two-frame band, negative gaps included, and is
    the exact negation of the validator's ``min-gap`` predicate: solver and
    checker can therefore never disagree about the same gap. Which branch applies
    is decided by where the left cue's speech sits relative to the band, and
    every branch is TRIM-ONLY -- the ladder shortens the left cue and never
    extends an end onto its own speech.
    """
    if next_start - prev_end >= TWO_FRAME_S - EPS:
        return None
    if prev_speech_end is not None and prev_speech_end > next_start - TWO_FRAME_S:
        branch = 2 if prev_speech_end <= next_start else 3
        return branch, min(prev_end, prev_speech_end)
    return 1, next_start - TWO_FRAME_S


# ------------------------------------------------------- reconstruction + sweep


def reconstruct_phase1(
    cues: Sequence[Phase1Cue], profile: DisplayProfile
) -> StreamState:
    """Rebuild the phase-1 state from each cue's IMMUTABLE seed record.

    Not read from ``cue.start``/``cue.end``: those are the producer's published
    answer, and a validator that starts from the answer cannot notice a phase-1
    defect. The seed span, the anchors and the canonical reading load are the
    inputs; everything else is recomputed here.
    """
    state: list[tuple[float, float]] = []
    for cue in cues:
        want = desired_end(
            start=cue.seed_start,
            seed_end=cue.seed_end,
            speech_end=cue.speech_end,
            reading_chars=cue.reading_chars,
            profile=profile,
        )
        end = want if want > cue.seed_end else cue.seed_end
        state.append((cue.seed_start, end))
    return tuple(state)


def sweep(
    state: StreamState,
    cues: Sequence[Phase1Cue],
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
) -> StreamState:
    """One full pass ``S`` in this module's arithmetic: cues ascending, six slots.

    Per-cue interleaving is load-bearing rather than incidental: a cue's end is
    settled before the next cue's start is touched, so the next start reads a
    floor that is already final. A start-first/end-second global order reads a
    stale floor and lands where neither order agrees.
    """
    starts = [pair[0] for pair in state]
    ends = [pair[1] for pair in state]
    shots = tuple(sorted(evidence.shots))
    snap_s = profile.shot_snap_s
    snapping = snap_s > 0 and bool(shots)
    cap = profile.max_cue_s if profile.max_cue_s else None
    count = len(cues)

    for index, cue in enumerate(cues):
        next_start = starts[index + 1] if index + 1 < count else None

        want = desired_end(
            start=starts[index],
            seed_end=cue.seed_end,
            speech_end=cue.speech_end,
            reading_chars=cue.reading_chars,
            profile=profile,
        )
        ends[index] = guarded_end(want, cue.seed_end, next_start)
        ends[index] = chained_end(ends[index], next_start)
        ends[index] = capped_end(
            starts[index], ends[index], cue.word_data, next_start, profile.max_cue_s
        )[0]

        if snapping:
            starts[index] = snapped_start(
                starts[index],
                ends[index],
                speech_start=cue.speech_start,
                prev_end=ends[index - 1] if index > 0 else None,
                shots=shots,
                snap_s=snap_s,
                cap=cap,
            )
            ends[index] = snapped_end(
                starts[index],
                ends[index],
                speech_end=cue.speech_end,
                next_start=next_start,
                shots=shots,
                snap_s=snap_s,
                cap=cap,
            )

        if index > 0:
            outcome = ladder_branch(
                ends[index - 1], cues[index - 1].speech_end, starts[index]
            )
            if outcome is not None:
                ends[index - 1] = outcome[1]

    return tuple(zip(starts, ends))


# ------------------------------------------------------------------- the replay


def _recompute(
    rule_id: str,
    cue_index: int,
    cues: Sequence[Phase1Cue],
    starts: Sequence[float],
    ends: Sequence[float],
    profile: DisplayProfile,
    shots: Sequence[float],
) -> float | None:
    """This module's answer for ONE rule in the validator's own current state.

    ``None`` means the rule does not apply here at all -- an unreachable slot or
    a ladder branch other than the one claimed -- which is itself a finding: a
    leg exists for a rule that had nothing to do.
    """
    count = len(cues)
    cue = cues[cue_index]
    cap = profile.max_cue_s if profile.max_cue_s else None
    next_start = starts[cue_index + 1] if cue_index + 1 < count else None

    if rule_id == "duration-desire":
        want = desired_end(
            start=starts[cue_index],
            seed_end=cue.seed_end,
            speech_end=cue.speech_end,
            reading_chars=cue.reading_chars,
            profile=profile,
        )
        return guarded_end(want, cue.seed_end, next_start)

    if rule_id == "chain":
        return chained_end(ends[cue_index], next_start)

    if rule_id == "cap":
        return capped_end(
            starts[cue_index],
            ends[cue_index],
            cue.word_data,
            next_start,
            profile.max_cue_s,
        )[0]

    if rule_id == "shot-in":
        return snapped_start(
            starts[cue_index],
            ends[cue_index],
            speech_start=cue.speech_start,
            prev_end=ends[cue_index - 1] if cue_index > 0 else None,
            shots=shots,
            snap_s=profile.shot_snap_s,
            cap=cap,
        )

    if rule_id == "shot-out":
        return snapped_end(
            starts[cue_index],
            ends[cue_index],
            speech_end=cue.speech_end,
            next_start=next_start,
            shots=shots,
            snap_s=profile.shot_snap_s,
            cap=cap,
        )

    if cue_index == 0:
        return None
    outcome = ladder_branch(
        ends[cue_index - 1], cues[cue_index - 1].speech_end, starts[cue_index]
    )
    if outcome is None or rule_id != f"ladder-{outcome[0]}":
        return None
    return outcome[1]


def _check_cycle(
    trace: Trace,
    cues: Sequence[Phase1Cue],
    replayed: StreamState,
    delivered: StreamState,
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
) -> list[str]:
    """Structural check of the declared cycle: transitions, membership, minimum.

    A cycle is a claim with three parts -- these states, in this order, each
    mapping to the next under ``S``, and the delivered stream is the numerically
    smallest of them. All three are checked, the transitions with this module's
    own sweep, because "the solver revisited a state" is exactly the sort of
    claim a defect would make about states that are not a cycle at all.
    """
    problems: list[str] = []
    cycle = trace.cycle
    if cycle is None:
        return ["cycle adoption without cycle evidence"]
    if not cycle.members:
        return ["cycle evidence declares no members"]
    if replayed not in cycle.members:
        problems.append("the replayed state is not a member of the declared cycle")
    for position, member in enumerate(cycle.members):
        successor = cycle.members[(position + 1) % len(cycle.members)]
        stepped = sweep(member, cues, profile=profile, evidence=evidence)
        if stepped != successor:
            problems.append(
                f"cycle member {position} does not step to the next member: "
                f"the sweep gives {stepped}, the evidence claims {successor}"
            )
    minimum = min(cycle.members, key=state_key)
    if cycle.adopted != minimum:
        problems.append("the adopted member is not the cycle's numeric minimum")
    if delivered != minimum:
        problems.append("the delivered state is not the cycle's numeric minimum")
    return problems


def replay_trace(
    trace: Trace,
    seed: Sequence[Phase1Cue],
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
    policy: FinalizePolicy,
    delivered: StreamState,
) -> tuple[str, ...]:
    """Verify one document-global trace (spec section 10.2). Empty tuple == pass.

    Every leg is checked three ways before it is allowed to advance the state:
    the ``from`` value must be what this replay holds, every neighbour the leg
    says it read must be what this replay holds -- the review-6 forged-snapshot
    hole -- and the rule recomputed here must give the leg's ``to``. The final
    state must then equal ``delivered`` bit-exactly.

    A fixed-point terminal is additionally required to BE a fixed point of this
    module's sweep. Leg-by-leg replay alone cannot see a rule the producer
    declined to run: an omitted leg replays to the same state the producer
    delivered, and only re-running the whole sweep notices that something should
    have moved.
    """
    del policy  # P5 has one policy value; the parameter is the P6 seam.
    problems: list[str] = []
    shots = tuple(sorted(evidence.shots))

    reconstructed = reconstruct_phase1(seed, profile)
    for index, cue in enumerate(seed):
        if reconstructed[index] != (cue.start, cue.end):
            problems.append(
                f"cue {index}: the producer published phase 1 as "
                f"{(cue.start, cue.end)}, this replay reconstructs "
                f"{reconstructed[index]}"
            )
    starts = [pair[0] for pair in reconstructed]
    ends = [pair[1] for pair in reconstructed]

    last_sweep = 0
    for position, leg in enumerate(trace.legs):
        if leg.rule_id not in RULE_IDS:
            problems.append(f"leg {position}: unknown rule {leg.rule_id!r}")
            continue
        if not 0 <= leg.cue_index < len(seed):
            problems.append(f"leg {position}: cue index {leg.cue_index} out of range")
            continue
        if not 0 <= leg.target.cue_index < len(seed):
            problems.append(
                f"leg {position}: target cue {leg.target.cue_index} out of range"
            )
            continue
        if leg.sweep < last_sweep:
            problems.append(
                f"leg {position}: sweep {leg.sweep} follows sweep {last_sweep}, "
                "so the trace is not one ordered trajectory"
            )
        last_sweep = leg.sweep

        held = (
            starts[leg.target.cue_index]
            if leg.target.side == "start"
            else ends[leg.target.cue_index]
        )
        if held != leg.from_value:
            problems.append(
                f"leg {position}: claims from={leg.from_value} but the replayed "
                f"state holds {held}"
            )
        for read in leg.reads:
            if not 0 <= read.boundary.cue_index < len(seed):
                problems.append(
                    f"leg {position}: read of cue {read.boundary.cue_index} is out "
                    "of range"
                )
                continue
            current = (
                starts[read.boundary.cue_index]
                if read.boundary.side == "start"
                else ends[read.boundary.cue_index]
            )
            if current != read.value:
                problems.append(
                    f"leg {position}: claims it read {read.value} at "
                    f"{read.boundary.side} of cue {read.boundary.cue_index}, but the "
                    f"replayed state holds {current}"
                )

        recomputed = _recompute(
            leg.rule_id, leg.cue_index, seed, starts, ends, profile, shots
        )
        if recomputed is None:
            problems.append(
                f"leg {position}: rule {leg.rule_id!r} does not apply in the "
                "replayed state"
            )
        elif recomputed != leg.to_value:
            problems.append(
                f"leg {position}: rule {leg.rule_id!r} recomputes to {recomputed}, "
                f"not {leg.to_value}"
            )

        if leg.target.side == "start":
            starts[leg.target.cue_index] = leg.to_value
        else:
            ends[leg.target.cue_index] = leg.to_value

    replayed = tuple(zip(starts, ends))
    if trace.terminal == "cycle-adoption":
        problems.extend(
            _check_cycle(
                trace,
                seed,
                replayed,
                delivered,
                profile=profile,
                evidence=evidence,
            )
        )
        return tuple(problems)

    if replayed != delivered:
        problems.append(f"replayed state {replayed} is not the delivered {delivered}")
    if trace.terminal == "fixed-point":
        stepped = sweep(delivered, seed, profile=profile, evidence=evidence)
        if stepped != delivered:
            problems.append(
                f"the delivered state is declared a fixed point but one sweep "
                f"moves it to {stepped}"
            )
    return tuple(problems)


def stability_check(
    delivered: StreamState,
    seed: Sequence[Phase1Cue],
    *,
    profile: DisplayProfile,
    evidence: FinalizeEvidence,
    policy: FinalizePolicy,
    terminal: Terminal,
) -> tuple[str, ...]:
    """Re-run ONE sweep on the delivered stream (spec section 10.3, B4 replacement).

    COMMON-MODE, deliberately and by construction: it drives the producer's own
    :func:`~voxweave.core.finalizer.apply_sweep` over the producer's own output,
    so a rule both sides implement wrongly is invisible to it. That is why it is
    a corruption check and not a correctness one -- the independent statement of
    the same property lives in :func:`replay_trace`, which steps the delivered
    stream with THIS module's sweep.

    Only a fixed-point terminal makes the claim. Under cycle adoption the
    delivered stream is a cycle member and one sweep is supposed to move it; a
    budget-exhausted run is not a measurement at all and short-circuits to the
    harness' exit 2 rather than to a failed predicate.
    """
    if terminal != "fixed-point":
        return ()
    stepped, _legs = apply_sweep(
        delivered,
        seed,
        profile=profile,
        evidence=evidence,
        policy=policy,
    )
    if stepped == delivered:
        return ()
    return (
        f"the delivered stream is not stable under one sweep: it moves to {stepped}",
    )
