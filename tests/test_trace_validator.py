# tests/test_trace_validator.py
"""The independent trace validator (P5 spec section 10.2).

Two obligations are pinned here, and they pull in opposite directions.

The validator must AGREE with the producer on every rule -- so section A is the
mirror suite: each rule function against the finalizer's own, and whole sweeps
against ``apply_sweep``, over shared targeted fixtures and a randomized corpus,
compared float-exactly rather than to a tolerance. A mirror that drifts is a
validator that fails honest traces.

The validator must not be ABLE to agree with a lie -- so sections C and D are the
negatives, led by the review-6 shape: a leg that is structurally perfect and
carries a false neighbour snapshot under which its wrong answer is the right one.
Each negative also demonstrates the defective validator that would accept it, so
the test states what the independence actually buys.
"""

import dataclasses
import random

import pytest

from voxweave.core import finalizer as fin
from voxweave.core import trace_validator as tv
from voxweave.core.authority import AuthorityLedger
from voxweave.core.segdoc import DisplayProfile

F = 1.0 / 24.0
TWO_FRAME_S = 2 * F


def profile(**over):
    """P_ZONE by default: every duration term OFF, so a zone rule can be isolated."""
    base = dict(
        language="en",
        max_line_length=42,
        max_lines=2,
        clause_ms=400.0,
        vad_skip_ms=250.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=0.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=11 * F,
    )
    base.update(over)
    return DisplayProfile(**base)


def cue(start, end, *, speech=None, words=None, text="word"):
    """A cue in the shape both engines construct (atom-level word_data)."""
    speech_start, speech_end = (start, end) if speech is None else speech
    timed = words if words is not None else [(speech_start, speech_end)]
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [
            {"text": "w", "start": span[0], "end": span[1]} for span in timed
        ],
        "speech_start": speech_start,
        "speech_end": speech_end,
    }


def run(cues, prof, *, shots=()):
    """Seal a v1 capture and finalize -- the only supported way into `finalize`."""
    ledger = AuthorityLedger()
    capture = fin.capture_v1_reference(cues, ledger=ledger)
    stream = fin.phase1_from_v1_capture(
        capture,
        profile=prof,
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    return fin.finalize(
        stream,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=tuple(shots)),
        policy=fin.FinalizePolicy(),
    )


def verify(result, cues, prof, *, shots=()):
    seed = fin.phase1_stream(cues, profile=prof)
    delivered = tuple((c["start"], c["end"]) for c in result.cues)
    return tv.replay_trace(
        result.trace,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=tuple(shots)),
        policy=fin.FinalizePolicy(),
        delivered=delivered,
    )


# =================================================== A. the mirror suite (shared)


def _random_document(rng):
    """One random document plus its profile and cuts.

    Deliberately includes the shapes the goldens isolate one at a time: seeded
    overlaps (the ladder), sub-half-second gaps (chaining), words sounding past
    the cap with and without a silence in the chain, anchorless cues (FD-8) and
    cuts sitting in every zone.
    """
    prof = profile(
        min_cue_s=rng.choice([0.0, 0.5, 1.0]),
        max_cue_s=rng.choice([0.0, 2.0, 7.0]),
        cps=rng.choice([0.0, 12.0, 17.0]),
        lag_out_s=rng.choice([0.0, 0.3]),
        shot_snap_s=rng.choice([0.0, 11 * F]),
    )
    cursor = 0.0
    cues = []
    for _ in range(rng.randint(1, 4)):
        start = max(0.0, cursor + rng.choice([-0.05, 0.0, 0.05, 0.2, 0.4, 0.9, 1.6]))
        end = start + rng.choice([0.2, 0.6, 1.1, 2.4, 8.0])
        words = [(start, start + (end - start) * 0.4)]
        if rng.random() < 0.6:
            tail_gap = rng.choice([0.05, 0.4, 1.4])
            words.append((words[-1][1] + tail_gap, end + rng.choice([0.0, 0.7])))
        anchored = rng.random() < 0.75
        speech = (
            (words[0][0], max(span[1] for span in words)) if anchored else (None, None)
        )
        cues.append(cue(start, end, speech=speech, words=words))
        cursor = end
    shots = tuple(
        sorted(
            {round(rng.uniform(0.0, cursor + 1.0), 4) for _ in range(rng.randint(0, 3))}
        )
    )
    return prof, cues, shots


@pytest.mark.parametrize("seed_value", range(60))
def test_sweep_mirrors_apply_sweep_over_random_documents(seed_value):
    """Whole sweeps, iterated: the two engines agree state for state, bit-exactly.

    Iterating rather than checking one pass matters -- a divergence in a rule
    that only fires once a neighbour has already moved is invisible to a single
    sweep, and the solver runs until nothing moves.
    """
    rng = random.Random(seed_value)
    for _ in range(4):
        prof, cues, shots = _random_document(rng)
        seed = fin.phase1_stream(cues, profile=prof)
        evidence = fin.FinalizeEvidence(shots=shots)
        state = tuple((c.start, c.end) for c in seed)
        for sweep_index in range(1, 6):
            produced, _legs = fin.apply_sweep(
                state,
                seed,
                profile=prof,
                evidence=evidence,
                policy=fin.FinalizePolicy(),
                sweep=sweep_index,
            )
            mirrored = tv.sweep(state, seed, profile=prof, evidence=evidence)
            assert mirrored == produced
            if produced == state:
                break
            state = produced


@pytest.mark.parametrize("seed_value", range(20))
def test_rule_functions_mirror_the_producer_helpers(seed_value):
    """Every rule, scalar for scalar, on inputs the sweep reaches only rarely.

    The whole-sweep mirror above cannot reach an equidistant cut pair or a
    held-word chain broken by exactly ``HELD_WORD_MAX_GAP_S``; this one aims at
    the arms directly.
    """
    rng = random.Random(1000 + seed_value)
    prof = profile(
        min_cue_s=rng.choice([0.0, 0.5, 1.0]),
        max_cue_s=rng.choice([0.0, 1.0, 7.0]),
        cps=rng.choice([0.0, 12.0, 17.0]),
        lag_out_s=rng.choice([0.0, 0.3]),
    )
    for _ in range(200):
        start = round(rng.uniform(0.0, 5.0), 3)
        end = start + round(rng.uniform(0.05, 9.0), 3)
        speech_end = None if rng.random() < 0.2 else round(rng.uniform(start, end), 3)
        speech_start = None if rng.random() < 0.2 else start
        next_start = (
            None
            if rng.random() < 0.2
            else end + rng.choice([-0.3, -0.05, 0.0, TWO_FRAME_S, 0.2, 0.6, 2.0])
        )
        chars = rng.randint(0, 60)

        want = tv.desired_end(
            start=start,
            seed_end=end,
            speech_end=speech_end,
            reading_chars=chars,
            profile=prof,
        )
        assert want == fin._duration_desire(
            start=start,
            seed_end=end,
            speech_end=speech_end,
            reading_chars=chars,
            profile=prof,
        )
        assert tv.guarded_end(want, end, next_start) == fin._guarded_end(
            want, end, next_start
        )
        assert tv.chained_end(end, next_start) == fin._chain_end(end, next_start)

        words = [
            {"text": "w", "start": start, "end": start + 0.3},
            {
                "text": "w",
                "start": start + 0.3 + rng.choice([0.1, 1.0, 1.6]),
                "end": end,
            },
        ]
        assert tv.capped_end(
            start, end, words, next_start, prof.max_cue_s
        ) == fin._cap_end(start, end, words, next_start, prof.max_cue_s)

        # Cuts placed at whole-frame offsets so every zone boundary is hit,
        # including the equidistant pair the tie rule exists for.
        cuts = tuple(
            sorted({round(start + n * F, 6) for n in rng.sample(range(-14, 15), 3)})
        )
        cap = prof.max_cue_s if prof.max_cue_s else None
        for value in (start, end):
            assert tv.nearest_cut(cuts, value, 11 * F) == fin._nearest(
                cuts, value, 11 * F
            )
        prev_end = None if rng.random() < 0.3 else start - rng.choice([0.05, 0.4])
        assert tv.snapped_start(
            start,
            end,
            speech_start=speech_start,
            prev_end=prev_end,
            shots=cuts,
            snap_s=11 * F,
            cap=cap,
        ) == fin._shot_in(
            start,
            end,
            speech_start=speech_start,
            prev_end=prev_end,
            shots=cuts,
            snap_s=11 * F,
            cap=cap,
        )
        assert tv.snapped_end(
            start,
            end,
            speech_end=speech_end,
            next_start=next_start,
            shots=cuts,
            snap_s=11 * F,
            cap=cap,
        ) == fin._shot_out(
            start,
            end,
            speech_end=speech_end,
            next_start=next_start,
            shots=cuts,
            snap_s=11 * F,
            cap=cap,
        )
        if next_start is not None:
            assert tv.ladder_branch(end, speech_end, next_start) == fin._ladder(
                end, speech_end, next_start
            )


def test_ladder_branches_mirror_at_both_thresholds():
    """The three branches and both equality boundaries, shared with N13's shapes."""
    prev_end, next_start = 1.5, 1.2
    for speech_end in (None, 1.0, 1.2 - TWO_FRAME_S, 1.15, 1.2, 1.3):
        assert tv.ladder_branch(prev_end, speech_end, next_start) == fin._ladder(
            prev_end, speech_end, next_start
        )
    # exactly two frames apart: neither engine fires
    assert tv.ladder_branch(1.0, 1.0, 1.0 + TWO_FRAME_S) is None
    assert fin._ladder(1.0, 1.0, 1.0 + TWO_FRAME_S) is None


def test_held_chain_waiver_mirrors_including_the_dead_air_refusal():
    """The cap golden's own pair: a held chain holds, a stray across dead air does not."""
    held = [
        {"text": "w", "start": 0.0, "end": 3.0},
        {"text": "w", "start": 3.2, "end": 7.6},
    ]
    stray = [
        {"text": "w", "start": 0.0, "end": 3.0},
        {"text": "w", "start": 5.5, "end": 7.6},
    ]
    for words, expected in ((held, 7.6), (stray, 7.0)):
        mirrored = tv.capped_end(0.0, 9.0, words, None, 7.0)
        assert mirrored == fin._cap_end(0.0, 9.0, words, None, 7.0)
        assert mirrored[0] == expected
    assert tv.capped_end(0.0, 9.0, held, None, 7.0)[1] is True
    assert tv.capped_end(0.0, 9.0, stray, None, 7.0)[1] is False


# ============================================= B. independent phase-1 reconstruction


@pytest.mark.parametrize("seed_value", range(20))
def test_reconstruct_phase1_matches_the_published_state(seed_value):
    """The validator's own phase-1 solve equals the producer's published one."""
    rng = random.Random(500 + seed_value)
    prof, cues, _shots = _random_document(rng)
    seed = fin.phase1_stream(cues, profile=prof)
    assert tv.reconstruct_phase1(seed, prof) == tuple((c.start, c.end) for c in seed)


def test_reconstruction_catches_a_doctored_phase1_publication():
    """A seed whose published end is not what its own record implies is reported.

    This is the check that makes "reconstructs phase-1 state itself" load
    bearing: a validator that started from ``cue.end`` would inherit whatever
    phase 1 got wrong and then confirm it.
    """
    prof = profile(min_cue_s=1.0)
    cues = [cue(0.0, 0.4, speech=(0.0, 0.4))]
    seed = fin.phase1_stream(cues, profile=prof)
    assert seed[0].end == 1.0  # the min-duration floor
    doctored = (dataclasses.replace(seed[0], end=2.5),)
    trace = fin.Trace(legs=(), terminal="fixed-point", cycle=None, sweeps=1)

    problems = tv.replay_trace(
        trace,
        doctored,
        profile=prof,
        evidence=fin.FinalizeEvidence(),
        policy=fin.FinalizePolicy(),
        delivered=((0.0, 2.5),),
    )
    assert any("reconstructs" in problem for problem in problems)


# ============================================================ C. replaying a trace


def _r6_document():
    """The review-6 fixture, adapted to a trace the finalizer can actually emit.

    R6's own numbers are `A.end = 24f`, `B.start = 29f`, a cut at `24f`: the
    zone proposes `24f`, the previous-cue floor lifts it to `26f`, and a
    defective producer delivers `28f` behind the false snapshot `prev_end = 26f`.
    Those exact numbers cannot appear in a finalizer TRACE, because slot 2 runs
    first and normalizes any gap in `(2f, 0.5)` to exactly two frames -- after
    which the floor equals `B.start` and the snap becomes a no-op. (The rule-level
    arithmetic is pinned unchanged in the test below; only the surrounding
    document is adapted.)

    So the document keeps R6's structure with a gap of exactly `CHAIN_MAX_GAP_S`
    or more: `A = [0.0, 1.0]` (its end is R6's `24f`), a cut at `1.7`, and
    `B.start = cut + 5f`, which is the 1-9-frames-after zone. The floor
    `1.0 + 2f` does not bind, so the honest answer is the cut itself.
    """
    prof = profile()
    cut = 1.7
    b_start = cut + 5 * F
    cues = [cue(0.0, 1.0), cue(b_start, 3.0)]
    return prof, cues, (cut,)


def test_honest_trace_replays_clean():
    prof, cues, shots = _r6_document()
    result = run(cues, prof, shots=shots)
    assert result.report.terminal == "fixed-point"
    assert [leg.rule_id for leg in result.trace.legs] == ["shot-in"]
    leg = result.trace.legs[0]
    assert leg.from_value == 1.9083333333333332
    assert leg.to_value == 1.7
    assert [read.value for read in leg.reads] == [1.0, 3.0]
    assert verify(result, cues, prof, shots=shots) == ()


def test_r6_rule_arithmetic_is_pinned_unchanged():
    """R6's own triple at the rule level: 24f floors to 26f, 26f floors to 28f.

    The attack needs exactly this: a snapshot one two-frame floor away from the
    value the producer wants to claim. Pinning it here keeps the numbers in the
    record even though the surrounding document has to be adapted.
    """
    honest = tv.snapped_start(
        29 * F,
        3.0,
        speech_start=None,
        prev_end=24 * F,
        shots=(24 * F,),
        snap_s=11 * F,
        cap=None,
    )
    forged = tv.snapped_start(
        29 * F,
        3.0,
        speech_start=None,
        prev_end=26 * F,
        shots=(24 * F,),
        snap_s=11 * F,
        cap=None,
    )
    assert honest == 26 * F == 1.0833333333333333
    assert forged == 28 * F == 1.1666666666666665
    assert honest == fin._shot_in(
        29 * F,
        3.0,
        speech_start=None,
        prev_end=24 * F,
        shots=(24 * F,),
        snap_s=11 * F,
        cap=None,
    )


def test_forged_neighbour_snapshot_is_rejected_r6():
    """The review-6 attack end to end, with the validator it defeats shown alongside.

    The leg is structurally perfect: right rule, right sweep and slot, right
    ``from``. Only the recorded neighbour is false -- it claims cue 0's end sat
    at `1.7` when the stream held `1.0` -- and the claimed ``to`` is exactly the
    value that false floor produces. A validator that recomputes the rule FROM
    THE SNAPSHOT therefore confirms the leg; this one recomputes from the state
    it reconstructed and does not.
    """
    prof, cues, shots = _r6_document()
    result = run(cues, prof, shots=shots)
    honest_leg = result.trace.legs[0]
    forged_prev_end = 1.7
    forged_to = forged_prev_end + TWO_FRAME_S

    # The defective validator: recompute the rule from the leg's own snapshot.
    assert (
        tv.snapped_start(
            honest_leg.from_value,
            3.0,
            speech_start=cues[1]["speech_start"],
            prev_end=forged_prev_end,
            shots=shots,
            snap_s=prof.shot_snap_s,
            cap=None,
        )
        == forged_to
    )

    forged_leg = dataclasses.replace(
        honest_leg,
        to_value=forged_to,
        reads=(
            dataclasses.replace(honest_leg.reads[0], value=forged_prev_end),
            honest_leg.reads[1],
        ),
    )
    trace = dataclasses.replace(result.trace, legs=(forged_leg,))
    seed = fin.phase1_stream(cues, profile=prof)
    problems = tv.replay_trace(
        trace,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=shots),
        policy=fin.FinalizePolicy(),
        delivered=((0.0, 1.0), (forged_to, 3.0)),
    )
    assert any(
        "read" in problem and "1.7" in problem and "1.0" in problem
        for problem in problems
    )
    assert any("recomputes to 1.7" in problem for problem in problems)


def test_forged_from_value_is_rejected():
    """A leg that lies about where the boundary started is caught before the rule."""
    prof, cues, shots = _r6_document()
    result = run(cues, prof, shots=shots)
    leg = dataclasses.replace(result.trace.legs[0], from_value=1.95)
    trace = dataclasses.replace(result.trace, legs=(leg,))
    seed = fin.phase1_stream(cues, profile=prof)
    problems = tv.replay_trace(
        trace,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=shots),
        policy=fin.FinalizePolicy(),
        delivered=tuple((c["start"], c["end"]) for c in result.cues),
    )
    assert any("claims from=1.95" in problem for problem in problems)


def test_an_omitted_rule_is_caught_by_the_fixed_point_requirement():
    """Leg-by-leg replay alone cannot see a rule the producer declined to run.

    An empty trace over a document whose first sweep DOES move something replays
    to the phase-1 state, and if the producer also delivers that state the
    per-leg checks are all vacuously satisfied. Only re-running the whole sweep
    on the delivered stream notices that a rule still had work to do -- which is
    why a fixed-point terminal is required to be an actual fixed point.
    """
    prof = profile()
    cues = [cue(0.0, 1.0), cue(1.3, 2.0)]  # gap 0.3: chaining must fire
    seed = fin.phase1_stream(cues, profile=prof)
    phase1 = tuple((c.start, c.end) for c in seed)
    trace = fin.Trace(legs=(), terminal="fixed-point", cycle=None, sweeps=1)

    problems = tv.replay_trace(
        trace,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(),
        policy=fin.FinalizePolicy(),
        delivered=phase1,
    )
    assert any("fixed point" in problem for problem in problems)

    honest = run(cues, prof)
    assert honest.cues[0]["end"] == 1.3 - TWO_FRAME_S
    assert verify(honest, cues, prof) == ()


def test_unknown_rule_and_out_of_range_targets_are_rejected():
    prof = profile()
    cues = [cue(0.0, 1.0)]
    seed = fin.phase1_stream(cues, profile=prof)
    delivered = ((0.0, 1.0),)
    ref = fin.BoundaryRef(0, "end")
    bogus = fin.TraceLeg(
        rule_id="teleport",
        sweep=1,
        cue_index=0,
        slot=1,
        target=ref,
        from_value=1.0,
        to_value=2.0,
        reads=(),
    )
    out_of_range = dataclasses.replace(bogus, rule_id="chain", cue_index=9)
    for leg, needle in ((bogus, "unknown rule"), (out_of_range, "out of range")):
        problems = tv.replay_trace(
            fin.Trace(legs=(leg,), terminal="fixed-point", cycle=None, sweeps=1),
            seed,
            profile=prof,
            evidence=fin.FinalizeEvidence(),
            policy=fin.FinalizePolicy(),
            delivered=delivered,
        )
        assert any(needle in problem for problem in problems)


def test_backwards_sweep_numbering_is_rejected():
    """One ordered trajectory, not a bag of legs: sweep numbers may not go back."""
    prof, cues, shots = _r6_document()
    result = run(cues, prof, shots=shots)
    leg = result.trace.legs[0]
    trace = dataclasses.replace(
        result.trace, legs=(dataclasses.replace(leg, sweep=3), leg)
    )
    seed = fin.phase1_stream(cues, profile=prof)
    problems = tv.replay_trace(
        trace,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=shots),
        policy=fin.FinalizePolicy(),
        delivered=tuple((c["start"], c["end"]) for c in result.cues),
    )
    assert any("ordered trajectory" in problem for problem in problems)


# ================================================== D. cycle evidence, structurally


def _cycle_run():
    """The 10f/22f oscillation golden: shots [0, 22f], snap 11f, seed 10f."""
    prof = profile()
    cues = [cue(10 * F, 5.0, speech=(None, None))]
    result = run(cues, prof, shots=(0.0, 22 * F))
    assert result.report.terminal == "cycle-adoption"
    return prof, cues, (0.0, 22 * F), result


def test_cycle_evidence_replays_clean():
    prof, cues, shots, result = _cycle_run()
    assert verify(result, cues, prof, shots=shots) == ()


def test_cycle_transitions_are_checked_with_the_validators_own_sweep():
    """A declared member that does not step to the next one is not a cycle."""
    prof, cues, shots, result = _cycle_run()
    cycle = result.trace.cycle
    assert cycle is not None
    doctored = dataclasses.replace(cycle, members=(cycle.members[0], ((9.9, 5.0),)))
    trace = dataclasses.replace(result.trace, cycle=doctored)
    seed = fin.phase1_stream(cues, profile=prof)
    problems = tv.replay_trace(
        trace,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=shots),
        policy=fin.FinalizePolicy(),
        delivered=tuple((c["start"], c["end"]) for c in result.cues),
    )
    assert any("does not step to the next member" in problem for problem in problems)


def test_adopting_a_non_minimal_member_is_rejected():
    """Adoption is the numeric minimum of the declared set, not a choice."""
    prof, cues, shots, result = _cycle_run()
    cycle = result.trace.cycle
    assert cycle is not None
    highest = max(cycle.members, key=fin.state_key)
    trace = dataclasses.replace(
        result.trace, cycle=dataclasses.replace(cycle, adopted=highest)
    )
    seed = fin.phase1_stream(cues, profile=prof)
    problems = tv.replay_trace(
        trace,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=shots),
        policy=fin.FinalizePolicy(),
        delivered=highest,
    )
    assert any("numeric minimum" in problem for problem in problems)


# ======================================================== E. the stability check


def test_stability_check_passes_on_a_fixed_point():
    prof, cues, shots = _r6_document()
    result = run(cues, prof, shots=shots)
    seed = fin.phase1_stream(cues, profile=prof)
    assert (
        tv.stability_check(
            tuple((c["start"], c["end"]) for c in result.cues),
            seed,
            profile=prof,
            evidence=fin.FinalizeEvidence(shots=shots),
            policy=fin.FinalizePolicy(),
            terminal=result.report.terminal,
        )
        == ()
    )


def test_stability_check_reports_a_corrupted_delivery():
    """Common-mode by construction: it catches corruption, never a shared rule bug."""
    prof, cues, shots = _r6_document()
    seed = fin.phase1_stream(cues, profile=prof)
    # Cue 0's end clipped after delivery: slot 1 rebases every end to its own
    # absolute desire, so one sweep puts it straight back and says so.
    corrupted = ((0.0, 0.5), (1.7, 3.0))
    assert tv.stability_check(
        corrupted,
        seed,
        profile=prof,
        evidence=fin.FinalizeEvidence(shots=shots),
        policy=fin.FinalizePolicy(),
        terminal="fixed-point",
    )


def test_stability_check_is_silent_on_non_fixed_point_terminals():
    """A cycle member is SUPPOSED to move, and a budget run is not a measurement."""
    prof, cues, shots, result = _cycle_run()
    seed = fin.phase1_stream(cues, profile=prof)
    delivered = tuple((c["start"], c["end"]) for c in result.cues)
    for terminal in ("cycle-adoption", "budget-exhausted"):
        assert (
            tv.stability_check(
                delivered,
                seed,
                profile=prof,
                evidence=fin.FinalizeEvidence(shots=shots),
                policy=fin.FinalizePolicy(),
                terminal=terminal,
            )
            == ()
        )


def test_finalizer_replay_trace_is_the_validators(monkeypatch):
    """`finalizer.replay_trace` is an alias, not a second implementation."""
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return ("sentinel",)

    monkeypatch.setattr(tv, "replay_trace", spy)
    assert fin.replay_trace(
        fin.Trace(legs=(), terminal="fixed-point", cycle=None, sweeps=0),
        (),
        profile=profile(),
        evidence=fin.FinalizeEvidence(),
        policy=fin.FinalizePolicy(),
        delivered=(),
    ) == ("sentinel",)
    assert len(calls) == 1
