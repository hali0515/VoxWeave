# tests/test_finalizer_properties.py
"""Architectural invariants of the TimelineFinalizer (P5 spec sections 2.3, 2.5, 2.6).

RED skeleton for W1: ``voxweave.core.finalizer`` does not exist yet.

Every idempotence claim the draft chain carried is a TOMBSTONE -- ``finalize`` is
not a projection and no test here may assert ``finalize(finalize(x))``. What
replaces it is narrower and actually true: N8a determinism (two pristine deep
copies of one serialized seed give bit-identical results, and neither input nor
the document is touched), N8b one root per row (``tests/test_authority.py``), and
N8c termination-as-policy -- the 10,000-sweep budget is an operational limit whose
exhaustion is a typed INVALID measurement, never a quietly frozen answer.
"""

import copy
import math
import struct

import pytest

from voxweave.core.segdoc import DisplayProfile

F = 1.0 / 24.0


def profile(language="en", **over):
    base = dict(
        language=language,
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


def cue(start, end, *, text="word", speech=(None, None), lyric=None):
    speech_start, speech_end = speech
    built = {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [{"text": text, "start": speech_start, "end": speech_end}],
        "speech_start": speech_start,
        "speech_end": speech_end,
    }
    if lyric is not None:
        built["lyric"] = lyric
    return built


def fin():
    from voxweave.core import finalizer as module

    return module


def run(cues, prof, *, shots=(), sings=()):
    """Seal a v1 capture and finalize it -- the only supported way in."""
    from voxweave.core.authority import AuthorityLedger

    module = fin()
    ledger = AuthorityLedger()
    capture = module.capture_v1_reference(cues, ledger=ledger)
    stream = module.phase1_from_v1_capture(
        capture,
        profile=prof,
        ledger=ledger,
        row_id="delivery_finalizer/v1",
        evaluation_id="e0",
    )
    return module.finalize(
        stream,
        profile=prof,
        evidence=module.FinalizeEvidence(shots=tuple(shots), sing_spans=tuple(sings)),
        policy=module.FinalizePolicy(),
    )


# ------------------------------------------------------------ N8a determinism


def test_n8a_two_pristine_copies_agree_bit_exactly():
    prof = profile(min_cue_s=1.0, cps=17.0, lag_out_s=0.3, max_cue_s=7.0)
    seed = [
        cue(0.0, 0.4, text="hello there", speech=(0.0, 0.4)),
        cue(1.0, 2.0, text="and again", speech=(1.0, 2.0)),
    ]
    left = run(copy.deepcopy(seed), prof, shots=(0.5, 2.5))
    right = run(copy.deepcopy(seed), prof, shots=(0.5, 2.5))
    assert left.cues == right.cues
    assert left.report.to_dict() == right.report.to_dict()
    assert left.trace.to_dict() == right.trace.to_dict()


def test_n8a_inputs_are_byte_unchanged():
    """In-place mutation cannot masquerade as determinism."""
    prof = profile(min_cue_s=1.0, cps=17.0)
    seed = [cue(0.0, 0.4, speech=(0.0, 0.4)), cue(1.0, 2.0, speech=(1.0, 2.0))]
    before = copy.deepcopy(seed)
    run(seed, prof)
    assert seed == before


def test_word_data_is_never_dropped_or_rebuilt():
    prof = profile(min_cue_s=1.0)
    seed = [cue(0.0, 0.4, speech=(0.0, 0.4))]
    result = run(copy.deepcopy(seed), prof)
    assert result.cues[0]["word_data"] == seed[0]["word_data"]


def test_lyric_flag_is_stamped_not_recomputed():
    """FD-2: the flag rides the evidence span, not the (widened) display span."""
    prof = profile(max_cue_s=7.0)
    seed = [cue(0.0, 1.25, speech=(None, None), lyric=True)]
    result = run(seed, prof, shots=(1.0,), sings=((0.0, 0.7),))
    assert result.cues[0]["end"] == 1.5  # cut + 12f, the R5 shape
    assert result.cues[0]["lyric"] is True


# ------------------------------------------------------- N8c budget semantics


def test_budget_exhaustion_is_a_typed_invalid_measurement(monkeypatch):
    """Exhaustion FREEZES at the last state and says so; it never guesses.

    The 10f/22f oscillator with a one-sweep budget: sweep 1 pairs the seed start
    ``10f`` with the cut at ``0.0`` (ten frames after it, so the landing zone
    pushes it to ``cut + 12f = 0.5``) and the budget is gone. What ships is that
    ``0.5`` -- the state the one sweep produced -- and NOT the seed, and not the
    cycle minimum ``10f`` the unbudgeted run adopts, because a run that never
    proved it had a cycle may not claim one.

    Everything else about the result is a typed refusal: ``valid=False``, the
    terminal, and a document-level report carrying the sweep count. The harness
    reads that as an invalid measurement (exit 2) rather than as an answer.
    """
    module = fin()
    monkeypatch.setattr(module, "SWEEP_BUDGET", 1)
    prof = profile()
    seed = [cue(10 * F, 5.0, speech=(None, None))]
    result = run(seed, prof, shots=(0.0, 22 * F))

    assert result.valid is False
    assert result.report.terminal == "budget-exhausted"
    assert result.trace.terminal == "budget-exhausted"
    assert result.trace.cycle is None
    assert result.report.max_sweeps_observed == 1
    assert result.trace.sweeps == 1
    assert result.cues[0]["start"] == 0.5
    assert result.cues[0]["start"] != 10 * F

    tag = next(t for t in result.report.entries if t.kind == "solver-budget-exhausted")
    assert tag.cue_index is None
    assert tag.evidence["sweeps"] == 1

    # The same seed with the shipped budget reaches the cycle and adopts its
    # minimum -- so the frozen answer above is the BUDGET's doing, not the shape's.
    monkeypatch.undo()
    unbudgeted = run(seed, prof, shots=(0.0, 22 * F))
    assert unbudgeted.report.terminal == "cycle-adoption"
    assert unbudgeted.valid is True


def test_document_level_reports_sort_ahead_of_every_cue(monkeypatch):
    """The ledger's order is a contract, because the artifact serializes it.

    ``solver-budget-exhausted`` is the only report with no cue index, and a sort
    key that treated "no cue" as "after every cue" would bury the one fact that
    says the whole measurement is invalid at the bottom of a list whose length
    is the cue count. The seed below carries per-cue reports too -- both cues are
    anchorless, so each mints ``fabricated-time`` for both sides -- which is what
    makes the position observable rather than vacuous.
    """
    module = fin()
    monkeypatch.setattr(module, "SWEEP_BUDGET", 1)
    prof = profile()
    seed = [
        cue(10 * F, 5.0, speech=(None, None)),
        cue(6.0, 7.0, speech=(None, None)),
    ]
    result = run(seed, prof, shots=(0.0, 22 * F))

    entries = result.report.entries
    assert result.report.terminal == "budget-exhausted"
    assert entries[0].kind == "solver-budget-exhausted"
    assert entries[0].cue_index is None
    assert {tag.cue_index for tag in entries[1:]} == {0, 1}


def test_budget_is_read_from_the_module_global_at_call_time(monkeypatch):
    """A default-argument binding would make the fixture above silently pass."""
    module = fin()
    assert module.SWEEP_BUDGET == 10_000
    monkeypatch.setattr(module, "SWEEP_BUDGET", 3)
    assert module.SWEEP_BUDGET == 3


def test_terminals_are_a_closed_vocabulary():
    module = fin()
    assert module.TERMINALS == ("budget-exhausted", "cycle-adoption", "fixed-point")


# ---------------------------------------------------------- preflight (2.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_preflight_refuses_non_finite_cue_times(bad):
    module = fin()
    with pytest.raises(module.NonFiniteTime):
        run([cue(0.0, bad, speech=(0.0, 1.0))], profile())


def test_preflight_refuses_a_non_finite_shot():
    module = fin()
    with pytest.raises(module.NonFiniteTime):
        run([cue(0.0, 1.0, speech=(0.0, 1.0))], profile(), shots=(float("nan"),))


@pytest.mark.parametrize("side", ["start", "end"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_preflight_refuses_a_non_finite_speech_anchor(side, bad):
    """An anchor is a time like any other; NaN there would poison every rule.

    A NaN speech end compares false against both ladder thresholds, so it would
    silently take branch 3 and stand as an unwaived overlap rather than raising.
    """
    module = fin()
    speech = (bad, 1.0) if side == "start" else (0.0, bad)
    with pytest.raises(module.NonFiniteTime):
        run([cue(0.0, 1.0, speech=speech)], profile())


def test_preflight_normalizes_signed_zero():
    module = fin()
    assert module.normalize_time(-0.0) == 0.0
    assert str(module.normalize_time(-0.0)) == "0.0"


def test_signed_zero_is_normalized_through_the_whole_pipeline():
    """``-0.0`` never reaches a delivered bound, an anchor or a packed state.

    ``-0.0 == 0.0`` is true while their bytes differ, so a run that let the sign
    through would produce two states that compare equal, hash apart, and
    serialize apart -- cycle detection and artifact bytes both wrong, and neither
    visibly. The packed pair below is the reason the normalization has to happen
    at the door rather than at the writer.
    """
    module = fin()
    result = run([cue(-0.0, 1.0, speech=(-0.0, 1.0))], profile())
    start = result.cues[0]["start"]
    assert start == 0.0
    assert math.copysign(1.0, start) == 1.0
    assert math.copysign(1.0, result.cues[0]["speech_start"]) == 1.0

    assert module.pack_state(((-0.0, 1.0),)) != module.pack_state(((0.0, 1.0),))
    assert module.pack_state(((start, 1.0),)) == module.pack_state(((0.0, 1.0),))


# ------------------------------------------- state canonicalization (2.5)


def test_packed_bytes_are_identity_only_and_never_the_minimum_key():
    """The little-endian trap: packed ``12f`` sorts BELOW packed ``10f``.

    Adopting the cycle minimum by byte order would return 12f where the golden
    requires 10f, so the minimum is taken over decoded numeric tuples and the byte
    representation is used for identity alone.
    """
    module = fin()
    ten = ((10 * F, 5.0),)
    twelve = ((12 * F, 5.0),)

    # The trap itself, stated as the negative it is: in little-endian bytes the
    # LARGER float sorts first, because the mantissa's low byte leads.
    assert struct.pack("<d", 12 * F) < struct.pack("<d", 10 * F)
    assert 10 * F < 12 * F

    # So the module packs big-endian, and even then packing is identity only:
    # the minimum policy and the cycle evidence order compare decoded numbers.
    assert module.pack_state(ten) == struct.pack(">2d", 10 * F, 5.0)
    assert module.state_key(ten) < module.state_key(twelve)
    packed = (module.pack_state(ten), module.pack_state(twelve))
    assert packed[0] != packed[1]
    assert min([ten, twelve], key=module.state_key) == ten
    assert min([ten, twelve], key=module.pack_state) == ten  # big-endian agrees


def test_state_identity_collisions_are_settled_by_a_full_compare():
    module = fin()
    state = ((0.0, 1.0), (1.5, 2.0))
    assert module.pack_state(state) == module.pack_state(copy.deepcopy(state))
    assert module.state_key(state) == (0.0, 1.0, 1.5, 2.0)


# ------------------------------------------------------------- vocabularies


def test_report_vocabulary_is_closed_and_sorted():
    module = fin()
    assert module.REPORT_KINDS == (
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


def test_only_waiver_kind_is_held_chain_duration():
    from voxweave.core.partition_check import WAIVER_KINDS

    module = fin()
    assert module.FINALIZER_WAIVER_KINDS == ("held-chain-duration",)
    assert WAIVER_KINDS == ("held-chain-duration",)


def test_rule_ids_are_the_frozen_slot_order():
    module = fin()
    assert module.RULE_IDS == (
        "duration-desire",
        "chain",
        "cap",
        "shot-in",
        "shot-out",
        "ladder-1",
        "ladder-2",
        "ladder-3",
    )


def test_fabricated_time_waives_nothing():
    """It is report-only: an anchorless cue still faces every hard predicate."""
    prof = profile(min_cue_s=1.0)
    result = run([cue(0.0, 0.2, speech=(None, None))], prof)
    kinds = {tag.kind for tag in result.report.entries}
    assert "fabricated-time" in kinds
    assert result.report.waivers == ()


def test_finalizer_stage_is_exit_driving():
    from voxweave.core.partition_check import EXIT_DRIVING_STAGES, STAGES

    assert "finalizer" in STAGES
    assert "finalizer" in EXIT_DRIVING_STAGES
