# tests/test_finalizer_fuzz.py
"""Randomized determinism sweep over the TimelineFinalizer (P5 spec 2.5/2.6).

The goldens pin what the finalizer computes; this file pins that it computes the
SAME thing twice and touches nothing on the way. That is N8a in its exact spec
form -- two PRISTINE deep copies of one serialized seed must give bit-identical
results, with both input copies asserted byte-unchanged afterwards, so in-place
mutation cannot masquerade as determinism.

The draw is deliberately hostile in the two directions that break determinism in
practice. Cue spacing is allowed to go NEGATIVE, so seeds overlap and the ladder
runs; and one draw in five is the oscillator family, whose two cuts are spaced
UNDER 24 frames -- close enough that the 11-frame window pairs a boundary with
one cut on the way out and the other on the way back. That is the shape whose
answer would otherwise depend on which sweep the run happened to stop at, and it
is why the solver has a cycle rule at all rather than a "run it a few times"
heuristic.

Every terminal the run can reach is asserted to be one of the three typed ones,
and a ``budget-exhausted`` draw would be a typed invalid measurement rather than
a silent freeze (its semantics are pinned in ``test_finalizer_properties.py``).
"""

import json
import random

from voxweave.core.segdoc import DisplayProfile

F = 1.0 / 24.0
DRAWS = 600

EN_SURFACES = [
    ("ok",),
    ("Hello,", "world!!"),
    ("I", "I", "I"),
    ("the", "quick", "brown", "fox"),
    ("supercalifragilisticexpialidociousandthensome_extra_tail",),
    ("aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh", "iiii", "jjjj"),
]
JA_SURFACES = [
    tuple("あい"),
    tuple("これはとても"),
    tuple("ながいにほんごのぶんしょうです"),
]


def fin():
    from voxweave.core import finalizer as module

    return module


def draw_profile(rng):
    language = rng.choice(["en", "en", "ja"])
    japanese = language == "ja"
    return DisplayProfile(
        language=language,
        max_line_length=18 if japanese else 42,
        max_lines=1 if japanese else 2,
        clause_ms=400.0,
        vad_skip_ms=250.0,
        offline_ms=700.0,
        min_cue_s=rng.choice([0.0, 0.5, 1.0, 1.01]),
        max_cue_s=rng.choice([0.0, 0.0, 1.0, 7.0]),
        glue_gap_s=0.3,
        cps=rng.choice([0.0, 8.0, 12.0, 17.0]),
        lag_out_s=rng.choice([0.0, 0.25, 0.3]),
        shot_snap_s=11 * F,
    )


def draw_cue(rng, start, surfaces):
    """One cue; anchorless three times in ten, and free to truncate its own speech."""
    span = round(rng.uniform(0.05, 2.5), 3)
    end = round(start + span, 3)
    if rng.random() < 0.3:
        speech_start = speech_end = None
        word_span = [(None, None)] * len(surfaces)
    else:
        speech_start = start
        speech_end = round(start + span * rng.uniform(0.3, 1.3), 3)
        step = (speech_end - speech_start) / len(surfaces)
        word_span = [
            (round(speech_start + i * step, 4), round(speech_start + (i + 1) * step, 4))
            for i in range(len(surfaces))
        ]
    lang_join = " " if surfaces[0].isascii() else ""
    return {
        "text": lang_join.join(surfaces),
        "start": start,
        "end": end,
        "word_data": [
            {"text": text, "start": span_[0], "end": span_[1]}
            for text, span_ in zip(surfaces, word_span)
        ],
        "speech_start": speech_start,
        "speech_end": speech_end,
    }


def draw_document(rng, index, profile):
    """A document plus its shot list; one draw in five is the oscillator family."""
    pool = JA_SURFACES if profile.language == "ja" else EN_SURFACES
    if index % 5 == 0:
        # The 10f/22f shape at a random offset: an anchorless start sitting ten
        # frames after the first of two cuts spaced under 24 frames apart.
        base = round(rng.uniform(0.0, 5.0), 3)
        spacing = rng.choice([12, 16, 20, 21, 22, 23]) * F
        start = base + 10 * F
        cue = draw_cue(rng, start, rng.choice(pool))
        cue["speech_start"] = cue["speech_end"] = None
        cue["word_data"] = [
            {"text": u["text"], "start": None, "end": None} for u in cue["word_data"]
        ]
        cue["end"] = start + 3.0
        return [cue], (base, base + spacing)

    cues = []
    cursor = round(rng.uniform(0.0, 2.0), 3)
    for _ in range(rng.randint(1, 4)):
        cue = draw_cue(rng, cursor, rng.choice(pool))
        cues.append(cue)
        # A negative advance overlaps the next cue: the ladder's own input class.
        cursor = round(cue["end"] + rng.uniform(-0.25, 0.8), 3)
    shots = ()
    roll = rng.random()
    if roll < 0.45:
        base = round(rng.uniform(0.0, 6.0), 3)
        shots = (base, base + rng.choice([12, 18, 22, 23]) * F)
    elif roll < 0.7:
        shots = (round(rng.uniform(0.0, 6.0), 3),)
    return cues, shots


def run(cues, profile, shots):
    from voxweave.core.authority import AuthorityLedger

    module = fin()
    ledger = AuthorityLedger()
    capture = module.capture_v1_reference(cues, ledger=ledger)
    stream = module.phase1_from_v1_capture(
        capture,
        profile=profile,
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    return module.finalize(
        stream,
        profile=profile,
        evidence=module.FinalizeEvidence(shots=tuple(shots)),
        policy=module.FinalizePolicy(),
    )


def test_n8a_determinism_over_six_hundred_draws():
    """Two pristine copies of one serialized seed; inputs byte-unchanged after."""
    module = fin()
    rng = random.Random(20260827)
    terminals = {}

    for index in range(DRAWS):
        profile = draw_profile(rng)
        cues, shots = draw_document(rng, index, profile)
        blob = json.dumps(cues, sort_keys=True)
        left_input = json.loads(blob)
        right_input = json.loads(blob)
        pristine = json.loads(blob)

        left = run(left_input, profile, shots)
        right = run(right_input, profile, shots)
        where = f"draw {index} (blob={blob}, shots={shots!r})"

        assert left.cues == right.cues, where
        assert left.report.to_dict() == right.report.to_dict(), where
        assert left.trace.to_dict() == right.trace.to_dict(), where
        assert left.valid == right.valid, where

        # In-place mutation cannot masquerade as determinism: neither copy moved.
        assert left_input == pristine, where
        assert right_input == pristine, where
        assert json.dumps(left_input, sort_keys=True) == blob, where

        # word_data rides through by reference and is never dropped or rebuilt.
        assert [c["word_data"] for c in left.cues] == [
            c["word_data"] for c in pristine
        ], where

        terminal = left.report.terminal
        terminals[terminal] = terminals.get(terminal, 0) + 1
        assert terminal in module.TERMINALS, where
        assert left.valid is (terminal != "budget-exhausted"), where
        assert {tag.kind for tag in left.report.entries} <= set(module.REPORT_KINDS), (
            where
        )
        assert {w.kind for w in left.report.waivers} <= set(
            module.FINALIZER_WAIVER_KINDS
        ), where
        assert {leg.rule_id for leg in left.trace.legs} <= set(module.RULE_IDS), where

    # The oscillator family exists so this assertion is not vacuous: a run that
    # never reached a cycle would be pinning determinism only on the easy half.
    assert terminals.get("fixed-point", 0) > 0, terminals
    assert terminals.get("cycle-adoption", 0) > 0, terminals


def test_fuzz_terminals_carry_their_obligations():
    """Every cycle adopts its numeric minimum and SAYS which boundaries moved."""
    module = fin()
    rng = random.Random(20260827)
    checked_cycles = 0
    replayed = 0

    for index in range(DRAWS):
        profile = draw_profile(rng)
        cues, shots = draw_document(rng, index, profile)
        result = run(cues, profile, shots)
        where = f"draw {index}"

        if result.report.terminal == "cycle-adoption":
            cycle = result.trace.cycle
            assert cycle is not None, where
            assert len(cycle.members) >= 2, where
            assert cycle.adopted == min(cycle.members, key=module.state_key), where
            # ... and the adopted member is what SHIPPED, not merely what was
            # recorded: freezing at the last visited state would leave the
            # evidence block honest and the delivery arbitrary.
            assert (
                tuple((c["start"], c["end"]) for c in result.cues) == cycle.adopted
            ), where
            moved = {
                ref for ref, values in cycle.per_boundary_values if len(set(values)) > 1
            }
            reported = {
                module.BoundaryRef(tag.cue_index, tag.evidence["boundary"])
                for tag in result.report.entries
                if tag.kind == "shot-unhonored"
            }
            assert moved == reported, where
            checked_cycles += 1

        if result.report.terminal == "fixed-point":
            seed = module.phase1_stream(cues, profile=profile)
            delivered = tuple((c["start"], c["end"]) for c in result.cues)
            assert (
                module.replay_trace(
                    result.trace,
                    seed,
                    profile=profile,
                    evidence=module.FinalizeEvidence(shots=tuple(shots)),
                    policy=module.FinalizePolicy(),
                    delivered=delivered,
                )
                == ()
            ), where
            replayed += 1

    assert checked_cycles > 0
    assert replayed > DRAWS // 2
