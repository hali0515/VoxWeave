# tests/test_finalizer_goldens.py
"""Hand-derived goldens for the TimelineFinalizer (P5 spec sections 2.2-2.4, 10.4).

RED skeleton for W1: ``voxweave.core.finalizer`` does not exist yet.

This file is the correctness AUTHORITY (N18). Every expected value is derived in
the test's own docstring or comment, and every number was produced by a read-only
probe of the real v1 primitives at base ``99f3605`` -- the rules the finalizer
ports are `timing._cleanup_cues` (slots 1-3) and `timing._snap_to_shots`
(slots 4-5), so a golden that disagrees with them is a port defect, not a policy
choice. Slot 6 (the overlap ladder) is new in P5 and its goldens come from spec
section 2.4 directly.

Section A pins phase 1 alone (pure, per-cue, order-free). Section B pins the
delivered stream, which is the fixed point of the sweep solver -- NOT the result
of one pass, and never the result of composing `finalize` with itself (that
composition is a tombstone).
"""

import pytest

from voxweave.core.segdoc import DisplayProfile

F = 1.0 / 24.0
TWO_FRAME_S = 2 * F


def profile(**over):
    """P_EN_BARE by default: every duration term OFF, so a rule can be isolated."""
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


P_EN = profile(min_cue_s=1.0, max_cue_s=7.0, cps=17.0, lag_out_s=0.3)
P_EN_CPS = profile(max_cue_s=7.0, cps=17.0)
P_JA = profile(
    language="ja",
    max_line_length=18,
    max_lines=1,
    min_cue_s=1.2,
    max_cue_s=6.0,
    cps=12.0,
)
P_ZONE = profile()


def cue(start, end, *, surfaces=("word",), speech=(None, None), words=None, lyric=None):
    """A cue in the shape both engines construct (atom-level word_data)."""
    speech_start, speech_end = speech
    timed = words if words is not None else [(speech_start, speech_end)] * len(surfaces)
    built = {
        "text": " ".join(surfaces),
        "start": start,
        "end": end,
        "word_data": [
            {"text": text, "start": span[0], "end": span[1]}
            for text, span in zip(surfaces, timed)
        ],
        "speech_start": speech_start,
        "speech_end": speech_end,
    }
    if lyric is not None:
        built["lyric"] = lyric
    return built


def fin():
    from voxweave.core import finalizer as module

    return module


def phase1(built, prof, index=0):
    return fin().phase1_cue(built, profile=prof, index=index)


def run(cues, prof, *, shots=(), sings=()):
    """Seal a v1 capture and finalize -- the only supported way into `finalize`."""
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


def kinds(result):
    return sorted(tag.kind for tag in result.report.entries)


def check_delivered(result, prof):
    """Violation kinds the FINALIZER-stage validator finds in a delivered stream.

    One source unit per cue, so text conservation is satisfied by construction
    and the run measures the timing predicates alone. The finalizer's own report
    ledger is handed over, because ``PF-2`` and ``PF-3`` are questions about the
    relationship between the delivered bounds and what the producer said.
    """
    from voxweave.core.partition_check import check_partition
    from voxweave.core.segdoc import SourceUnit

    delivered = list(result.cues)
    units = [
        SourceUnit(id=f"u{index}", surface=c["text"], start=c["start"], end=c["end"])
        for index, c in enumerate(delivered)
    ]
    checked = check_partition(
        list(range(1, len(delivered))),
        delivered,
        units=units,
        profile=prof,
        origin="v2",
        stage="finalizer",
        reports=result.report.entries,
    )
    return sorted(violation.kind for violation in checked.violations)


# ===================================================================== A. phase 1


def test_cf1_no_desire_term_fires():
    """CF-1: all three terms disabled -> `want` never leaves `end`; no report."""
    out = phase1(cue(1.0, 2.0, surfaces=("x",), speech=(1.0, 2.0)), P_ZONE)
    assert (out.start, out.end) == (1.0, 2.0)
    assert (out.text, out.lines, out.cell_widths, out.reading_chars) == (
        "x",
        ("x",),
        (1,),
        1,
    )
    assert out.reports == ()


def test_cf2_min_duration_floor_wins():
    """CF-2: floor 5.0+1.0=6.0 beats lag 5.4+0.3=5.7 and cps min(5.117…, 6.4)."""
    out = phase1(cue(5.0, 5.4, surfaces=("ok",), speech=(5.0, 5.4)), P_EN)
    assert out.end == 6.0
    assert out.reading_chars == 2


def test_cf3_lag_out_pad_wins():
    """CF-3: floor 6.0 < end 6.5; lag 6.5+0.3=6.8; cps min(5.117…, 7.5)."""
    out = phase1(cue(5.0, 6.5, surfaces=("ok",), speech=(5.0, 6.5)), P_EN)
    assert out.end == 6.8


def test_cf4_cps_linger_wins():
    """CF-4: 30 canonical chars / 17 = 1.7647058823529411, under the 1s ceiling."""
    thirty = ("aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff")
    out = phase1(cue(0.0, 1.0, surfaces=thirty, speech=(0.0, 1.0)), P_EN)
    assert out.reading_chars == 30
    assert out.end == 30 / 17.0
    assert out.end == 1.7647058823529411


def test_cf5_linger_cap_binds_on_the_speech_anchor():
    """CF-5: need 60/17=3.529… but the ceiling is speech_end+1.0, NOT display_end+1.0."""
    sixty = tuple(f"{c * 5}" for c in "abcdefghijkl")
    out = phase1(cue(0.0, 1.0, surfaces=sixty, speech=(0.0, 1.0)), P_EN)
    assert out.reading_chars == 60
    assert out.end == 2.0


def test_cf6_fd8_anchorless_is_verbatim():
    """CF-6: `speech_end is None` skips the WHOLE desire block -- the floor too."""
    thirty = ("aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff")
    out = phase1(cue(0.0, 0.2, surfaces=thirty, speech=(None, None)), P_EN)
    assert out.end == 0.2
    assert [(tag.kind, tag.evidence["side"]) for tag in out.reports] == [
        ("fabricated-time", "start"),
        ("fabricated-time", "end"),
    ]


def test_cf7_fd8_counterfactual_twin():
    """CF-7: the SAME cue with anchors -> floor 1.0, then min(1.7647…, 0.2+1.0)=1.2."""
    thirty = ("aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff")
    out = phase1(cue(0.0, 0.2, surfaces=thirty, speech=(0.0, 0.2)), P_EN)
    assert out.end == 1.2
    assert out.reports == ()


def test_cf8_half_anchored_reads_only_the_end_anchor():
    """CF-8: the desire never reads `speech_start`; the missing tag suppresses nothing."""
    thirty = ("aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "fffff")
    out = phase1(cue(0.0, 1.0, surfaces=thirty, speech=(None, 1.0)), P_EN)
    assert out.end == 1.7647058823529411
    assert [tag.evidence["side"] for tag in out.reports] == ["start"]


def test_cf9_fd1_canonical_reading_load():
    """CF-9: raw 13 non-space chars, canonical 10 -> 10/17, not 13/17."""
    built = cue(0.0, 0.5, surfaces=("Hello,", "world!!"), speech=(0.0, 0.5))
    out = phase1(built, P_EN_CPS)
    assert out.text == "Hello world"
    assert out.reading_chars == 10
    assert out.end == 10 / 17.0
    assert out.end == 0.5882352941176471


def test_cf10_bounded_stutter_load_is_the_hyphenated_form():
    """CF-10: 3 scans (stable), canonical 'I-I-I' -> 5 reading chars, not 3."""
    out = phase1(cue(0.0, 0.5, surfaces=("I", "I", "I"), speech=(0.0, 0.5)), P_EN_CPS)
    assert out.text == "I-I-I"
    assert out.reading_chars == 5
    assert out.end == 0.5  # 5/17 = 0.294… < 0.5, so the desire does not bind


def test_cf12_line_capacity_names_the_indivisible_token():
    """CF-12: the over-wide line SHIPS; the report carries the token."""
    token = "supercalifragilisticexpialidociousandthensome_extra_tail"
    out = phase1(cue(0.0, 1.0, surfaces=(token,), speech=(0.0, 1.0)), P_EN)
    assert out.cell_widths == (56,)
    assert out.end == 2.0  # min(56/17, 1.0+1.0) = 2.0
    tag = next(t for t in out.reports if t.kind == "line-capacity")
    assert (tag.evidence["width"], tag.evidence["budget"], tag.evidence["token"]) == (
        56,
        42,
        token,
    )


def test_cf13_over_wide_line_without_an_indivisible_token():
    """CF-13: ja, 21 kana = 42 cells, budget 36, max_lines 1 -> no split budget."""
    kana = tuple("これはとてもながいにほんごのぶんしょうです")
    out = phase1(cue(0.0, 1.0, surfaces=kana, speech=(0.0, 1.0)), P_JA)
    assert out.cell_widths == (42,)
    assert out.end == 1.75  # floor 1.2, then min(21/12, 2.0) = 1.75
    tag = next(t for t in out.reports if t.kind == "line-capacity")
    assert tag.evidence["token"] is None


def test_cf14_canonical_fallback_is_reported_not_silent():
    built = cue(0.0, 1.0, surfaces=("x",), speech=(0.0, 1.0))
    built["word_data"] = [{"start": 0.0, "end": 0.4}, {"start": 0.4, "end": 0.8}]
    built["text"] = "salvaged text"
    out = phase1(built, P_ZONE)
    assert out.text == "salvaged text"
    tag = next(t for t in out.reports if t.kind == "canonical-text-fallback")
    assert tag.evidence["reason"] == "empty-reconstruction"


def test_cf17_phase1_never_repairs_a_truncating_end():
    """CF-17: desire-only. The validator reports `speech-truncated-end`, not phase 1."""
    out = phase1(cue(0.0, 1.0, surfaces=("x",), speech=(0.0, 1.4)), P_ZONE)
    assert out.end == 1.0
    assert out.reports == ()


def test_cf18_start_never_moves_in_phase_1():
    out = phase1(cue(1.0, 2.0, surfaces=("x",), speech=(0.5, 2.0)), P_EN)
    assert out.start == 1.0


# ============================================================ B. golden matrix
#
# Shared setup for the zone goldens: one cue, cut at 10.0, snap window 11f, every
# duration term off, so only slots 4/5 can move anything.


@pytest.mark.parametrize(
    ("label", "start", "expected"),
    [
        ("already on the cut -> untouched", 10.0, 10.0),
        ("1-7 frames before -> onto the cut", 10.0 - 5 * F, 10.0),
        ("7 frames before (zone edge) -> onto the cut", 10.0 - 7 * F, 10.0),
        ("8 frames before (zone edge) -> cut - 12f", 10.0 - 8 * F, 9.5),
        ("8-11 frames before -> cut - 12f", 10.0 - 10 * F, 9.5),
        ("1-9 frames after -> back onto the cut", 10.0 + 5 * F, 10.0),
        ("9 frames after (zone edge) -> back onto the cut", 10.0 + 9 * F, 10.0),
        ("10-11 frames after -> cut + 12f", 10.0 + 10 * F, 10.5),
    ],
)
def test_in_time_zone_rules(label, start, expected):
    """Every in-time zone including both of its interior edges.

    The four zones are half-open, so the frame at which one ends and the next
    begins is where a port drifts silently. Derivations, cut ``C = 10.0``:

    * ``d == 0`` -- ``abs(cut - start) <= _EPS`` short-circuits before any zone
      is consulted, so a boundary already on the cut is left alone (and emits no
      leg at all).
    * ``d == -7f`` -- the first zone is ``-7f - _EPS <= d < 0``; the float
      subtraction gives ``-0.2916666666666661`` against a bound of
      ``-0.29166666766666666``, so the edge frame is INSIDE and the target is the
      cut. ``d == -8f`` (``-0.33333333333333393``) is outside and takes the
      lead-in to ``C - 12f = 9.5``.
    * ``d == 9f`` -- the third zone is ``d <= 9f + _EPS``; ``0.375 <= 0.375…001``
      holds, so 9 frames after still lands back ON the cut and only 10 frames
      after is pushed out to ``C + 12f = 10.5``.

    Every cue here is anchorless and three seconds long, so no speech veto and no
    duration term can interfere: the only rule with anything to say is slot 4.
    """
    result = run([cue(start, start + 3.0, speech=(None, None))], P_ZONE, shots=(10.0,))
    assert result.cues[0]["start"] == expected, label
    assert result.report.terminal == "fixed-point"


def test_in_time_snap_window_is_inclusive_at_eleven_frames():
    """The pairing window is ``<= snap_s``, so 11 frames pairs and 12 does not.

    Both halves DELIVER ``0.5``, which is exactly why the assertion is on the
    trace rather than on the number: at 11 frames after the cut the boundary is
    in the "10-11 frames after" zone and is PUSHED to ``cut + 12f = 0.5``; at 12
    frames it is already there and no cut was ever paired with it. A window test
    that only read the delivered value would pass with the window bound removed.

    Cut ``0.0`` keeps the arithmetic exact: ``abs(0.0 - 11f) == 11f`` is the
    inclusive edge itself, and ``12f == 0.5 > 11f`` is the first frame outside.
    """
    inside = run([cue(11 * F, 3.0, speech=(None, None))], P_ZONE, shots=(0.0,))
    assert inside.cues[0]["start"] == 0.5
    assert [(leg.rule_id, leg.to_value) for leg in inside.trace.legs] == [
        ("shot-in", 0.5)
    ]

    outside = run([cue(12 * F, 3.0, speech=(None, None))], P_ZONE, shots=(0.0,))
    assert outside.cues[0]["start"] == 0.5
    assert outside.trace.legs == ()


def test_in_time_refuses_a_snap_that_would_leave_under_two_frames():
    """``new_start < end - TWO_FRAME_S`` -- a cue may not be snapped to nothing.

    Start ``C - 5f = 9.791666666666666`` is in the flash-removal zone, so the raw
    target is the cut ``10.0``; the cue ends at ``10.05``, and
    ``10.05 - 2f = 9.966666666666667`` is EARLIER than the target, so the move
    would leave under two frames of display and is refused whole.

    ``max_cue_s = 0.3`` is switched on only to keep slot 5 quiet: the out-time
    pull-back to ``C - 2f`` is vetoed by the cue's own speech end (``10.05``),
    and the 12-frames-after last resort (``10.5``) then fails ``within_cap``
    because ``10.5 - 9.791666666666666 = 0.708…`` exceeds the cap. The cap is not
    the reason the START stays put -- the cue's own duration ``0.258…`` is well
    inside it, so slot 3 never fires either, and slot 4's own ``within_cap`` on
    ``(10.0, 10.05)`` passes: the two-frame guard is the sole refusal.
    """
    prof = profile(max_cue_s=0.3)
    result = run(
        [cue(10.0 - 5 * F, 10.05, speech=(None, 10.05), words=[(9.85, 10.05)])],
        prof,
        shots=(10.0,),
    )
    assert result.cues[0]["start"] == 9.791666666666666
    assert result.cues[0]["end"] == 10.05
    assert result.trace.legs == ()
    assert result.report.max_sweeps_observed == 1


def test_in_time_lead_in_respects_the_duration_cap():
    """A free lead-in still lengthens the cue, so the cap can refuse it.

    Start ``C - 10f = 9.583333333333334`` is in the 8-11-frames-before zone, so
    the target is ``C - 12f = 9.5`` and the cue would run ``9.5 -> 10.5``, one
    full second. With ``max_cue_s = 0.95`` that fails ``within_cap`` and the
    start stays; slot 3 does not fire on the way in, because the cue's own
    duration is ``10.5 - 9.583333333333334 = 0.916…``, under the cap.

    The counterfactual with the cap off is the "8-11 frames before" row of the
    zone table, and it lands on ``9.5``: one profile field is the whole
    difference.
    """
    start = 10.0 - 10 * F
    capped = run(
        [cue(start, 10.5, speech=(None, None))], profile(max_cue_s=0.95), shots=(10.0,)
    )
    assert capped.cues[0]["start"] == 9.583333333333334
    assert capped.trace.legs == ()

    uncapped = run([cue(start, 10.5, speech=(None, None))], P_ZONE, shots=(10.0,))
    assert uncapped.cues[0]["start"] == 9.5


def test_in_time_speech_start_veto_is_skipped_whole():
    """#24: a delaying move that would land past the cue's own first word is not clamped."""
    start = 10.0 - 5 * F  # 9.791666666666666; the zone target 10.0 sits past speech
    result = run(
        [cue(start, start + 3.0, speech=(9.85, 10.4), words=[(9.85, 10.4)])],
        P_ZONE,
        shots=(10.0,),
    )
    assert result.cues[0]["start"] == 9.791666666666666


@pytest.mark.parametrize(
    ("label", "end", "speech_end", "expected"),
    [
        ("<=12 frames before -> cut - 2f", 9.75, 9.5, 9.916666666666666),
        ("1-5 frames after -> pull back to cut - 2f", 10.125, 9.9, 9.916666666666666),
        ("pull-back vetoed -> cut + 12f last resort", 10.125, 10.0, 10.5),
        ("6-11 frames after -> cut + 12f", 10.0 + 8 * F, None, 10.5),
    ],
)
def test_out_time_zone_rules(label, end, speech_end, expected):
    words = None if speech_end is None else [(end - 2.5, speech_end)]
    result = run(
        [cue(end - 3.0, end, speech=(None, speech_end), words=words)],
        P_ZONE,
        shots=(10.0,),
    )
    assert result.cues[0]["end"] == expected, label


def test_out_time_extension_respects_the_next_cues_two_frames():
    """An out-time extension may not eat the following cue's separation.

    Cut ``C = 10.0``. A is ``[8.0, 9.75]``; B starts at ``C - 2f =
    9.916666666666666`` and is anchored so its own in-time move is vetoed (the
    target ``10.0`` sits past B's first word at ``9.93``, the #24 veto), which
    keeps B's start fixed while A's end is measured.

    A's slot 2 sees a gap of ``0.166…`` -- inside ``CHAIN_MAX_GAP_S`` and above
    two frames -- and chains A's end to ``B.start - 2f = 9.833333333333332``.
    Slot 5 then pairs that end with the cut (``0.166…`` away, inside the
    11-frame window) and computes the die-on-the-cut target ``C - 2f =
    9.916666666666666``, an EXTENSION. It is refused because
    ``target <= next_start - TWO_FRAME_S`` fails -- ``9.916666666666666`` is
    exactly the two frames B is owed.

    This is the general shape rather than a contrived one: whenever chaining has
    fired, the end already sits at ``next_start - 2f``, so any later extension is
    by construction an encroachment.

    The assertion has to be on the TRACE, not only on the delivered number: an
    extension made here would collapse the gap to zero, the ladder would trim it
    straight back to ``next.start - 2f``, and the delivered value would be
    identical. What the guard buys is that the move is never MADE -- so the legs
    are exactly slot 2 firing once per sweep plus slot 1's rebase, with no
    ``shot-out`` among them:

        sweep 1  chain            9.75 -> 9.833333333333332
        sweep 2  duration-desire  9.833333333333332 -> 9.75   (rebase to the seed)
        sweep 2  chain            9.75 -> 9.833333333333332   (fixed point)
    """
    b_start = 10.0 - TWO_FRAME_S
    a = cue(8.0, 9.75, speech=(None, None))
    b = cue(b_start, 12.0, speech=(9.93, 11.5), words=[(9.93, 11.5)])
    result = run([a, b], P_ZONE, shots=(10.0,))

    assert result.cues[1]["start"] == b_start  # the #24 veto held B in place
    assert result.cues[0]["end"] == b_start - TWO_FRAME_S
    assert result.cues[0]["end"] == 9.833333333333332
    assert result.cues[0]["end"] != 10.0 - TWO_FRAME_S  # the refused target
    assert "input-overlap" not in kinds(result)
    assert result.report.terminal == "fixed-point"
    assert [leg.rule_id for leg in result.trace.legs] == [
        "chain",
        "duration-desire",
        "chain",
    ]
    assert [leg.sweep for leg in result.trace.legs] == [1, 2, 2]


def test_out_time_extension_respects_the_duration_cap():
    """The same extension, refused by the cap instead of by a neighbour.

    A single cue ``[8.0, 9.75]`` pairs with the cut at ``10.0`` (six frames
    before, inside the window) and wants ``C - 2f = 9.916666666666666``. That
    would make the cue ``1.916666666666666`` long, so ``max_cue_s = 1.9`` refuses
    it -- while slot 3 stays silent, because the cue's own ``1.75`` is under the
    cap. With the cap off the same cue lands on ``9.916666666666666``.
    """
    seed = cue(8.0, 9.75, speech=(None, None))
    capped = run([seed], profile(max_cue_s=1.9), shots=(10.0,))
    assert capped.cues[0]["end"] == 9.75
    assert capped.trace.legs == ()

    uncapped = run([seed], P_ZONE, shots=(10.0,))
    assert uncapped.cues[0]["end"] == 9.916666666666666


def test_out_time_last_resort_respects_the_duration_cap():
    """The 12-frames-after last resort is a move like any other, cap included.

    End ``C + 3f = 10.125`` is in the 1-5-frames-after zone, so the target is the
    pull-back to ``C - 2f = 9.916666666666666``; speech runs to ``10.0``, so the
    pull-back is vetoed and the TTSG last resort proposes ``C + 12f = 10.5``.
    From a start of ``9.0`` that is a ``1.5`` second cue, which ``max_cue_s =
    1.4`` refuses -- so the end stays where the vetoed pull-back left it. Slot 3
    does not fire either: the delivered ``1.125`` is inside the cap.

    With the cap off this is the "pull-back vetoed -> cut + 12f" row of the
    out-time table, which lands on ``10.5``.
    """
    seed = cue(9.0, 10.125, speech=(None, 10.0), words=[(9.5, 10.0)])
    capped = run([seed], profile(max_cue_s=1.4), shots=(10.0,))
    assert capped.cues[0]["end"] == 10.125
    assert capped.trace.legs == ()

    uncapped = run([seed], P_ZONE, shots=(10.0,))
    assert uncapped.cues[0]["end"] == 10.5


def test_ladder_branch_1_mints_two_frames():
    """prev.speech_end 1.0 <= 1.2 - 2f = 1.1166666666666667 -> the full trim is legal."""
    prev = cue(0.0, 1.5, speech=(0.0, 1.0), words=[(0.0, 1.0)])
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)
    assert result.cues[0]["end"] == 1.2 - TWO_FRAME_S
    assert result.cues[0]["end"] == 1.1166666666666667
    assert "input-overlap" in kinds(result)  # the seed pair overlapped: 1.5 > 1.2


def test_ladder_branch_2_preserves_speech_and_reports():
    """speech_end 1.15 sits inside the two-frame band -> trim to speech, report the gap."""
    prev = cue(0.0, 1.5, speech=(0.0, 1.15), words=[(0.0, 1.15)])
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)
    assert result.cues[0]["end"] == 1.15
    tag = next(t for t in result.report.entries if t.kind == "min-gap-unmet")
    assert tag.cue_index == 0
    assert tag.evidence["resulting_gap"] == 1.2 - 1.15
    assert tag.evidence["resulting_gap"] == 0.050000000000000044


def test_ladder_branch_3_leaves_the_overlap_standing():
    """True crosstalk: speech_end 1.3 > next.start 1.2, so the trim stops at speech."""
    prev = cue(0.0, 1.5, speech=(0.0, 1.3), words=[(0.0, 1.3)])
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)
    assert result.cues[0]["end"] == 1.3
    assert "min-gap-unmet" not in kinds(result)
    assert "overlap" in check_delivered(result, P_ZONE)


def test_branch_3_on_a_touching_pair_stands_as_min_gap_not_overlap():
    """Branch 3 without a display overlap: the standing violation changes KIND.

    Section 2.4 describes branch 3's residue as an ``overlap`` that stands, and
    that is what the fixture above sees -- because there the cues really do
    overlap on screen. Branch 3 only requires speech past the NEXT START, so it
    also fires on a pair that merely touches: ``prev`` ends exactly where
    ``next`` begins while its speech runs on to ``1.5``.

    Nothing moves (every branch trims, and ``min(prev.end, speech_end)`` is
    already ``prev.end``), and the ladder mints no report, because only branch 2
    does. The delivered gap is therefore ``0.0`` with nothing to explain it, and
    the checker calls that ``min-gap`` -- ``overlap`` needs ``start < prev_end -
    EPS`` and this pair does not have it. Both kinds are unwaived and
    exit-driving, so the contract "branch 3 leaves a standing violation" holds;
    what does not hold is "branch 3 leaves an OVERLAP", and a harness that
    triages by kind has to know that.

    ``speech-truncated-end`` rides along by construction: branch 3 means speech
    past the next start, and the gap being non-negative puts it past this cue's
    own end too.
    """
    prev = cue(0.0, 1.0, speech=(0.0, 1.5), words=[(0.0, 1.5)])
    nxt = cue(1.0, 2.0, speech=(1.0, 2.0), words=[(1.0, 2.0)])
    result = run([prev, nxt], P_ZONE)

    assert result.cues[0]["end"] == 1.0
    assert result.trace.legs == ()
    assert "min-gap-unmet" not in kinds(result)
    assert check_delivered(result, P_ZONE) == ["min-gap", "speech-truncated-end"]


def ladder_leg(result):
    """The single ladder leg of a two-cue run, with its branch in the rule id."""
    legs = [leg for leg in result.trace.legs if leg.rule_id.startswith("ladder-")]
    assert legs, "no ladder leg was emitted"
    assert len({leg.rule_id for leg in legs}) == 1
    return legs[0]


def test_ladder_with_no_speech_anchor_takes_branch_1_unconditionally():
    """``speech_end is None`` -> branch 1, whatever the overlap looks like.

    An anchorless left cue has no speech for branches 2 and 3 to protect, so the
    ladder is free to mint the full two frames: prev ``[0.0, 1.5]`` against next
    start ``1.2`` trims to ``1.2 - 2f = 1.1166666666666667``, the same target
    branch 1 computes for an anchored cue whose speech clears it.

    The discriminator against branch 2 is the REPORT, not the number: a branch-2
    resolution would have added ``min-gap-unmet`` (and, under
    :func:`partition_check.check_partition`, would then be the only thing making
    the delivered gap legal). Nothing is reported here because the gap IS two
    frames.
    """
    prev = cue(0.0, 1.5, speech=(None, None))
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)

    assert result.cues[0]["end"] == 1.2 - TWO_FRAME_S
    assert result.cues[0]["end"] == 1.1166666666666667
    assert ladder_leg(result).rule_id == "ladder-1"
    assert "min-gap-unmet" not in kinds(result)
    assert "fabricated-time" in kinds(result)
    assert "input-overlap" in kinds(result)  # 1.5 > 1.2 in the SEED stream


def test_input_overlap_is_strict_so_a_touching_seed_pair_reports_nothing():
    """``input-overlap`` records ``prev.end > next.start``, and only that.

    The fixture above pins the positive side. This is the boundary: cues that
    END and BEGIN at the same instant do not overlap, they abut, and reporting
    them would put a fact in the ledger about a stream that never had it.

    The distinction is load-bearing rather than cosmetic. ``input-overlap`` is
    the trigger the delta registry reads for FD-3, so a report minted on an
    abutting pair claims the run exercised an overlap-resolution row it did not
    exercise -- and the diff classifier recomputes that trigger itself and fails
    on the mismatch. The ladder still runs here (a zero gap is under the floor)
    and mints its two frames, which is a separate fact with a separate name.
    """
    prev = cue(0.0, 1.2, speech=(0.0, 1.0), words=[(0.0, 1.0)])
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)

    assert "input-overlap" not in kinds(result)
    assert "FD-3" not in result.report.deltas_fired
    assert result.cues[0]["end"] == 1.2 - TWO_FRAME_S  # the ladder still fired
    assert ladder_leg(result).rule_id == "ladder-1"


def test_ladder_threshold_a_equality_stays_in_branch_1():
    """``prev.speech_end == next.start - TWO_FRAME_S`` is INSIDE branch 1.

    Both branches would deliver the same end here -- branch 1 trims to
    ``next.start - 2f`` and branch 2 would trim to ``speech_end``, and at this
    threshold those are the same number ``1.1166666666666667``. So the value
    cannot tell them apart and the pin is on what each branch OWES: branch 2
    would have to report ``min-gap-unmet``, and a run that reported it here would
    be claiming the two-frame floor was unmet when it was met exactly.
    """
    edge = 1.2 - TWO_FRAME_S
    prev = cue(0.0, 1.5, speech=(0.0, edge), words=[(0.0, edge)])
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)

    assert result.cues[0]["end"] == edge == 1.1166666666666667
    assert ladder_leg(result).rule_id == "ladder-1"
    assert "min-gap-unmet" not in kinds(result)


def test_ladder_threshold_b_equality_is_branch_2_not_branch_3():
    """``prev.speech_end == next.start`` is INSIDE branch 2, not crosstalk yet.

    Speech that ends exactly where the next cue starts does not overlap it, so
    the pair is a reported near-miss rather than the unwaivable crosstalk of
    branch 3. Again both branches deliver ``min(1.5, 1.2) == 1.2``; the tag is
    the difference, and it is load-bearing -- the delivered gap is ``0.0``, which
    ``PF-2`` calls a ``min-gap`` violation UNLESS this exact tag names cue 0.

    The tag's evidence has to survive ``PF-3``'s independent recomputation:
    ``resulting_gap == next_start - speech_end == 0.0``, inside
    ``[0, TWO_FRAME_S)``, and the branch-2 precondition
    ``1.1166666666666667 < 1.2 <= 1.2`` holds.
    """
    prev = cue(0.0, 1.5, speech=(0.0, 1.2), words=[(0.0, 1.2)])
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)

    assert result.cues[0]["end"] == 1.2
    assert ladder_leg(result).rule_id == "ladder-2"
    tag = next(t for t in result.report.entries if t.kind == "min-gap-unmet")
    assert tag.cue_index == 0
    assert tag.evidence["resulting_gap"] == 0.0
    assert tag.evidence["speech_end"] == 1.2
    assert tag.evidence["next_start"] == 1.2

    checked = check_delivered(result, P_ZONE)
    assert "min-gap" not in checked
    assert "forged-report" not in checked


def test_ladder_trigger_is_silent_at_exactly_two_frames():
    """The trigger is the exact negation of ``PF-2``, so neither fires at ``2f``.

    ``prev.end = 1.2 - 2f`` gives a gap of ``0.08333333333333326`` -- two frames
    to within one float ulp, and above ``TWO_FRAME_S - EPS``. The ladder declines
    it, and so does the validator: they read the same predicate from opposite
    sides, which is the point of stating the trigger with ``partition_check.EPS``
    rather than with the rules' own ``1e-9``.

    The left cue's speech end ``1.18`` sits inside the branch-2 window, so a
    ladder that DID fire here would emit ``min-gap-unmet`` (the value would not
    move -- every branch trims, and this end is already at or below all three
    targets). The absence of that report is the pin.
    """
    prev = cue(0.0, 1.2 - TWO_FRAME_S, speech=(0.0, 1.18), words=[(0.0, 1.18)])
    nxt = cue(1.2, 2.0, speech=(1.2, 2.0), words=[(1.2, 2.0)])
    result = run([prev, nxt], P_ZONE)

    assert result.cues[0]["end"] == 1.1166666666666667
    assert result.trace.legs == ()
    assert "min-gap-unmet" not in kinds(result)
    assert "min-gap" not in check_delivered(result, P_ZONE)


def test_ladder_trigger_and_pf2_meet_exactly_at_two_frames_minus_eps():
    """The seam of the exact negation, pinned ON the seam rather than near it.

    ``test_ladder_trigger_is_silent_at_exactly_two_frames`` lands a gap one ulp
    under ``TWO_FRAME_S``, which is comfortably above ``TWO_FRAME_S - EPS`` --
    both ``>=`` and ``>`` decline it, so it cannot tell the trigger's relation
    from its strict cousin. The single gap that can is the boundary value
    itself.

    ``prev.end`` is exactly ``0.0`` and ``next.start`` is the literal float
    ``TWO_FRAME_S - EPS``, so the subtraction is exact and the trigger's left
    side IS its right side. The ladder must decline (``>=``), and ``PF-2``,
    whose band is the half-open complement ``[-EPS, TWO_FRAME_S - EPS)``, must
    also decline. A strict trigger would trim ``prev.end`` to
    ``-1.000000000001e-06`` and leave the checker silent about it -- the exact
    hole the shared constant exists to close.
    """
    from voxweave.core.partition_check import EPS

    boundary = TWO_FRAME_S - EPS
    prev = cue(-1.0, 0.0, speech=(-0.9, -0.2), words=[(-0.9, -0.2)])
    nxt = cue(boundary, 1.0, speech=(boundary, 1.0), words=[(boundary, 1.0)])
    result = run([prev, nxt], P_ZONE)

    assert boundary == 0.08333233333333333
    assert result.cues[0]["end"] == 0.0
    assert result.trace.legs == ()
    assert "min-gap-unmet" not in kinds(result)
    assert "min-gap" not in check_delivered(result, P_ZONE)


def test_chaining_closes_a_sub_half_second_gap_to_two_frames():
    prev = cue(0.0, 1.0, speech=(0.0, 1.0), words=[(0.0, 1.0)])
    nxt = cue(1.3, 2.0, speech=(1.3, 2.0), words=[(1.3, 2.0)])
    result = run([prev, nxt], P_ZONE)
    assert result.cues[0]["end"] == 1.3 - TWO_FRAME_S
    assert result.cues[0]["end"] == 1.2166666666666668


def test_chaining_declines_a_gap_that_is_already_exactly_two_frames():
    """Slot 2 fires on a gap STRICTLY wider than two frames, never on the target.

    At exactly two frames the rule's own target ``next.start - TWO_FRAME_S``
    usually equals the end already, so a ``>=`` relation is invisible -- the
    no-op leg is dropped before it is recorded. It is invisible only USUALLY:
    when the two subtractions round differently the target is a ulp away from
    the end, and the cue silently loses that ulp every time the sweep runs.

    The pair below is such a case, found by exhaustive ulp search around
    ``end + TWO_FRAME_S``:

    * ``0.1145832858402704 - 0.031249952506937077 == TWO_FRAME_S`` exactly;
    * ``0.1145832858402704 - TWO_FRAME_S == 0.031249952506937073``, one ulp low.

    So the strict relation is not decoration. Nothing may move, and the trace
    must be empty -- an emitted leg here would be the rule chasing its own
    rounding error.
    """
    end = 0.031249952506937077
    next_start = 0.1145832858402704
    assert next_start - end == TWO_FRAME_S
    assert next_start - TWO_FRAME_S == 0.031249952506937073 != end

    prev = cue(0.0, end, speech=(0.0, end), words=[(0.0, end)])
    nxt = cue(next_start, 1.0, speech=(next_start, 1.0), words=[(next_start, 1.0)])
    result = run([prev, nxt], P_ZONE)

    assert result.cues[0]["end"] == end
    assert result.trace.legs == ()


def test_87fde9d_gap_preservation():
    """A gap already at the two-frame floor is never extended into (it would collapse)."""
    prof = profile(lag_out_s=0.3)
    prev = cue(0.0, 1.0, speech=(0.0, 1.0), words=[(0.0, 1.0)])
    nxt = cue(
        1.0 + TWO_FRAME_S,
        2.0,
        speech=(1.0 + TWO_FRAME_S, 2.0),
        words=[(1.0 + TWO_FRAME_S, 2.0)],
    )
    result = run([prev, nxt], prof)
    assert result.cues[0]["end"] == 1.0
    # phase 1 still PUBLISHED the desire (1.0 + 0.3); the movement ledger records both
    assert any(
        move.boundary.cue_index == 0 and move.phase1 == 1.3 and move.delivered == 1.0
        for move in result.report.movement
    )


def test_87fde9d_refuses_a_gap_that_is_exactly_two_frames():
    """The guard's floor is INCLUSIVE, pinned where ``>`` and ``>=`` disagree.

    ``test_87fde9d_gap_preservation`` builds its neighbour as ``1.0 +
    TWO_FRAME_S``, and that sum minus ``1.0`` comes back one ulp SHORT of two
    frames -- under the floor on either relation, so it cannot tell them apart.
    Here ``seed_end`` is exactly ``0.0``, which makes ``next.start - seed_end``
    the literal ``TWO_FRAME_S``: the one gap where "at the floor" and "under the
    floor" are different answers.

    The desire has to survive the guard to be observable, so it is kept inside
    ``EPS`` of the seed end (``-0.5 + 0.5000005 = 4.999999999588667e-07``,
    exact by Sterbenz). That is deliberate: a bigger desire would be clawed back
    by the ladder inside the same sweep and both relations would deliver the
    same number. At this size the ladder stays silent (the residual gap
    ``0.08333283333333337`` is still above ``TWO_FRAME_S - EPS``), so the grant
    would survive to the delivered stream -- and the refusal is what ships.
    """
    prof = profile(lag_out_s=0.5000005)
    prev = cue(-1.0, 0.0, speech=(-0.9, -0.5), words=[(-0.9, -0.5)])
    nxt = cue(TWO_FRAME_S, 1.0, speech=(None, None), words=[(None, None)])
    result = run([prev, nxt], prof)

    want = -0.5 + 0.5000005
    assert 0.0 < want <= 1e-6
    assert result.cues[0]["end"] == 0.0
    # phase 1 published the desire; the guard is what takes it back.
    assert any(
        move.boundary.cue_index == 0 and move.phase1 == want and move.delivered == 0.0
        for move in result.report.movement
    )


def test_cap_is_waived_by_a_continuous_held_chain():
    prof = profile(max_cue_s=7.0)
    held = cue(
        0.0, 9.0, surfaces=("a", "b"), speech=(0.0, 7.6), words=[(0.0, 3.0), (3.2, 7.6)]
    )
    result = run([held], prof)
    assert result.cues[0]["end"] == 7.6
    assert [w.kind for w in result.report.waivers] == ["held-chain-duration"]


def test_cap_refuses_to_cross_dead_air_for_a_stray_syllable():
    prof = profile(max_cue_s=7.0)
    dead = cue(
        0.0, 9.0, surfaces=("a", "b"), speech=(0.0, 7.6), words=[(0.0, 3.0), (5.5, 7.6)]
    )
    result = run([dead], prof)
    assert result.cues[0]["end"] == 7.0
    assert result.report.waivers == ()


def test_min_duration_short_is_reported_when_the_neighbour_blocks_the_floor():
    """Legacy clamps to next.start (a ZERO gap); slot 6 then mints the two frames."""
    prof = profile(min_cue_s=1.0)
    first = cue(0.0, 0.4, speech=(0.0, 0.4), words=[(0.0, 0.4)])
    second = cue(0.5, 2.0, speech=(0.5, 2.0), words=[(0.5, 2.0)])
    result = run([first, second], prof)
    assert result.cues[0]["end"] == 0.5 - TWO_FRAME_S
    assert result.cues[0]["end"] == 0.4166666666666667
    assert "min-duration-short" in kinds(result)


def test_10f_22f_cycle_adopts_the_numeric_minimum():
    """shots [0, 22f], snap 11f: 10f -> cut+12f -> 12f -> cut-12f -> 10f.

    The trajectory is a 2-cycle, so the solver adopts the LEXICOGRAPHICALLY MINIMAL
    member over decoded numeric tuples (10f) and says so; it does not freeze at the
    last state and does not refuse silently.
    """
    result = run([cue(10 * F, 5.0, speech=(None, None))], P_ZONE, shots=(0.0, 22 * F))
    assert result.cues[0]["start"] == 0.41666666666666663
    assert result.report.terminal == "cycle-adoption"
    tag = next(t for t in result.report.entries if t.kind == "shot-unhonored")
    assert tag.evidence["reason"] == "oscillation"
    assert sorted(tag.evidence["values"]) == [0.41666666666666663, 0.5]


def test_12f_seed_companion_returns_the_same_minimum():
    """The seed that separates cycle-min adoption from refusal-at-seed."""
    result = run([cue(12 * F, 5.0, speech=(None, None))], P_ZONE, shots=(0.0, 22 * F))
    assert result.cues[0]["start"] == 0.41666666666666663
    assert result.report.terminal == "cycle-adoption"


def test_anchorless_fd8_then_shot_extension():
    """R5: phase 1 refuses the legacy extension; slot 5 makes the same delivery move.

    Which is exactly why the delivered lyric flag may not be recomputed from the
    display span (FD-2): the sung fraction crosses 0.5 between 1.25 and 1.5.
    """
    prof = profile(max_cue_s=7.0, lag_out_s=0.25)
    seed = cue(0.0, 1.25, speech=(None, None), lyric=True)
    assert phase1(seed, prof).end == 1.25  # FD-8: no extension in phase 1
    result = run([seed], prof, shots=(1.0,), sings=((0.0, 0.7),))
    assert result.cues[0]["end"] == 1.5  # cut + 12f
    assert result.cues[0]["lyric"] is True


def test_non_frame_multiple_cap_target_after_a_start_move():
    """`max_cue_s` is an unclamped float, so cap targets sit on no grid at all.

    Sweep 1 caps at `0.7916666666666666 + 7.01`, which frees the in-time move onto
    the cut; sweep 2 re-caps from the moved start at `1.0 + 7.01`. A single pass
    would deliver the first number -- pinning the second is what proves the solver
    iterates rather than running production's pass order once.
    """
    prof = profile(max_cue_s=7.01)
    result = run([cue(1.0 - 5 * F, 8.5, speech=(None, None))], prof, shots=(1.0,))
    assert result.cues[0]["start"] == 1.0
    assert result.cues[0]["end"] == 1.0 + 7.01


def test_non_frame_multiple_min_target_after_a_start_move():
    """The min-duration floor is re-solved from the MOVED start, off any grid.

    ``min_cue_s = 1.01`` and one cue ``[1.0 - 5f, 1.5]`` whose speech begins at
    ``1.0``, with a cut at ``1.0``.

    Phase 1 wants ``0.7916666666666667 + 1.01 = 1.8016666666666667``. Sweep 1's
    slot 4 then moves the start onto the cut -- the raw target ``1.0`` is a
    delaying move of ``0.208…`` (inside half a second) that lands exactly ON the
    first word rather than past it, so the #24 veto does not fire. Sweep 2's slot
    1 re-solves the floor against the new start and gets ``1.0 + 1.01 = 2.01``;
    sweep 3 reproduces it, so the delivered end is ``2.01``.

    ``2.01 * 24`` is not an even number of frames (``…% 2 == 0.2399…``), which is
    the point: cap and floor targets are absolute sums of profile floats and sit
    on no quantization grid at all.
    """
    prof = profile(min_cue_s=1.01)
    seed = cue(1.0 - 5 * F, 1.5, speech=(1.0, 1.5), words=[(1.0, 1.5)])
    assert phase1(seed, prof).end == 1.8016666666666667

    result = run([seed], prof, shots=(1.0,))
    assert result.cues[0]["start"] == 1.0
    assert result.cues[0]["end"] == 1.0 + 1.01 == 2.01
    assert (result.cues[0]["end"] * 24.0) % 2.0 != 0.0
    assert result.report.max_sweeps_observed == 3
    assert "min-duration-short" not in kinds(result)
    assert any(
        move.boundary.cue_index == 0
        and move.boundary.side == "end"
        and move.phase1 == 1.8016666666666667
        and move.delivered == 2.01
        for move in result.report.movement
    )


def test_r5_mixed_anchored_anchorless_phase_1():
    """R5 run-1 (phase-1) golden: A desires 41f from 7 canonical chars at cps=8."""
    prof = profile(cps=8.0)
    a = cue(
        20 * F,
        21 * F,
        surfaces=("abcdefg",),
        speech=(20 * F, 21 * F),
        words=[(20 * F, 21 * F)],
    )
    out = phase1(a, prof)
    assert out.reading_chars == 7
    assert out.end == 41 * F
    assert out.end == 1.7083333333333333
    b = cue(60 * F, 72 * F, speech=(None, None))
    assert phase1(b, prof, index=1).end == 72 * F


def test_r5_mixed_anchored_anchorless_delivered():
    """R5 DELIVERED: the fixed point is (50f, 52f), NOT the run-1 pair (35f, 52f).

    Derived by hand from the slot table, with shots {37f, 52f}, snap 11f, the cap
    off and every duration term but CPS disabled.

    Phase 1: A = [20f, 41f] (7 canonical chars / cps 8 = 0.875 s past 20f, under
    the 1 s linger ceiling); B anchorless = [60f, 72f] verbatim (FD-8).

    Sweep 1. A slot 1: desire 41f, next.start 60f, 60f - seed_end 21f > 2f, so
    end = min(41f, 60f) = 41f. Slot 2: gap 60f - 41f = 19f >= CHAIN_MAX_GAP_S
    (0.5 s = 12f), no chain. Slot 5: nearest cut to 41f is 37f (4f away; 52f is
    11f away and loses the strictly-nearer test), 4f <= 5f so the target is
    37f - 2f = 35f, a pull-back that clears A's speech end 21f -> A.end = 35f.
    B slot 4: nearest cut to 60f is 52f (8f), in the 1-9-frames-after zone, so
    the target is the cut itself; the floor A.end + 2f = 37f does not bind and
    the move is earlier than the seed, so B.start = 52f. That pair (35f, 52f) is
    the legacy composite's run-1 answer.

    Sweep 2. A slot 1 re-proposes 41f from the IMMUTABLE seed end (never from the
    35f it was left at), and B's start has moved to 52f: 52f - 21f > 2f, so
    end = min(41f, 52f) = 41f. Slot 2 now sees gap 52f - 41f = 11f, inside
    CHAIN_MAX_GAP_S and above 2f -> A.end = 52f - 2f = 50f. Slot 5: 50f is
    already `cut - 2f` for the 52f cut, so the out-time rule is a no-op.

    Sweep 3 reproduces sweep 2, so the delivered pair is (50f, 52f).

    Section 10.4's "(run-1 golden)" wording predates the immutable-phase-1-basis
    resolution of exactly this counterexample (spec section 3.4, slot 1 reading
    `seed_end`); the fixed point is what the LAW now means, and it is what is
    pinned here.
    """
    prof = profile(cps=8.0)
    a = cue(
        20 * F,
        21 * F,
        surfaces=("abcdefg",),
        speech=(20 * F, 21 * F),
        words=[(20 * F, 21 * F)],
    )
    b = cue(60 * F, 72 * F, speech=(None, None))
    result = run([a, b], prof, shots=(37 * F, 52 * F))
    assert result.cues[1]["start"] == 52 * F
    assert result.cues[0]["end"] == 52 * F - TWO_FRAME_S
    assert (result.cues[0]["end"], result.cues[1]["start"]) == (
        2.083333333333333,
        2.1666666666666665,
    )
    assert result.report.terminal == "fixed-point"
    assert result.report.max_sweeps_observed == 3


def test_r3_shape_1_per_cue_interleaving_reads_a_final_floor():
    """R3 finding 1: the two-cue shape a start-first/end-second order gets wrong.

    One cut C = 10.0, snap 11f, cap off, no duration terms.
    A = [C-12f, C+5f] with speech_end = C-2f; B = [C+6f, C+31f], anchorless.

    Review 3 showed that closing every start before every end lands B on the
    stale floor A.end + 2f = C+7f, and that the next run then moves it to C --
    cue bytes changing between two runs of the same pass. The finalizer's sweep
    is per-cue interleaved exactly like production, so A's out-time pull-back
    (C+5f is 5f after the cut, speech end C-2f permits the pull-back) closes
    A.end to C-2f BEFORE B's in-time rule reads its floor. B's raw target is the
    cut itself (6f after, the 1-9 zone), the floor A.end + 2f is exactly C, and
    the move is earlier than B's seed, so B.start = C.

    The ladder does not fire on the delivered pair: the gap is exactly 2f. Sweep
    2 re-derives both moves from the same seeds, so C-2f/C is the fixed point --
    run 1 and run 2 agree, which is the property R3 said was missing.
    """
    prof = profile()
    cut = 10.0
    a = cue(
        cut - 12 * F,
        cut + 5 * F,
        speech=(None, cut - TWO_FRAME_S),
        words=[(cut - 12 * F, cut - TWO_FRAME_S)],
    )
    b = cue(cut + 6 * F, cut + 31 * F, speech=(None, None))
    result = run([a, b], prof, shots=(cut,))
    assert result.cues[0]["end"] == cut - TWO_FRAME_S
    assert result.cues[0]["end"] == 9.916666666666666
    assert result.cues[1]["start"] == cut
    assert result.report.terminal == "fixed-point"
    assert "min-gap" not in kinds(result)


def test_r3_shape_2_ladder_trims_are_inside_the_solve_loop():
    """R3 finding 2: a lawful ladder trim used to move again on the next run.

    Same cut C = 10.0, snap 11f, cap off, no duration terms.
    A = [C-12f, C+20f] with speech_end = C-4f; B = [C+5f, C+40f], anchorless.

    A's end is outside the snap window (20f > 11f), and B's in-time move is
    refused because the floor A.end + 2f = C+22f would delay B by 17f, past the
    12-frame landing displacement. The ladder then lawfully trims A to
    B.start - 2f = C+3f (branch 1: speech_end C-4f clears it).

    Under the legacy composite that trim is the end of the run, and run 2's
    out-time rule maps C+3f (3f after the cut, speech end C-4f permits the
    pull-back) to C-2f: two runs, two answers. Here the trim lives INSIDE the
    solve loop, and slot 1 rebases A's end to its immutable seed end C+20f at
    the top of every sweep -- so the out-time rule never sees C+3f, and sweep 2
    reproduces sweep 1's C+3f exactly.
    """
    prof = profile()
    cut = 10.0
    a = cue(
        cut - 12 * F,
        cut + 20 * F,
        speech=(None, cut - 4 * F),
        words=[(cut - 12 * F, cut - 4 * F)],
    )
    b = cue(cut + 5 * F, cut + 40 * F, speech=(None, None))
    result = run([a, b], prof, shots=(cut,))
    assert result.cues[1]["start"] == cut + 5 * F  # the in-time move stays refused
    assert result.cues[0]["end"] == (cut + 5 * F) - TWO_FRAME_S
    assert result.cues[0]["end"] == 10.125
    assert result.report.terminal == "fixed-point"


def test_r4_divergence_shape_terminates_under_fd8():
    """R4 finding 1: the anchorless lag-out ratchet, terminated by FD-8.

    Review 4's probe was one cue [0, 1] with no timed word_data, no shots and
    `lag_out_s = 0.25`. The legacy rule takes the CURRENT DISPLAY END as the lag
    anchor when a cue has no speech end, so each re-run adds another 0.25 s:
    with `max_cue_s = 7` it crawled 1.25, 1.50, ... and only stopped at the cap
    on sweep 25; with the cap disabled it was still growing at sweep 40 (11.0)
    and had no fixed point at all.

    FD-8 closes it analytically rather than by a budget: an anchorless cue gets
    no extension, so phase 1 publishes the input span verbatim and slot 1
    re-derives that same span from the immutable seed end every sweep. One sweep
    changes nothing and the run terminates as a fixed point -- under either cap.
    """
    for max_cue_s in (0.0, 7.0):
        prof = profile(lag_out_s=0.25, max_cue_s=max_cue_s)
        result = run([cue(0.0, 1.0, speech=(None, None))], prof)
        assert result.cues[0]["end"] == 1.0, max_cue_s
        assert result.report.terminal == "fixed-point"
        assert result.report.max_sweeps_observed == 1
        assert "fabricated-time" in kinds(result)

    # The counterfactual twin: with anchors the pad fires ONCE and stays there,
    # because the target is absolute (speech_end + lag_out_s), not relative to
    # the end the previous sweep produced.
    prof = profile(lag_out_s=0.25)
    anchored = run([cue(0.0, 1.0, speech=(0.0, 1.0), words=[(0.0, 1.0)])], prof)
    assert anchored.cues[0]["end"] == 1.25
    assert anchored.report.max_sweeps_observed == 1


def _composed_document():
    """CF-9 (FD-1) + chaining (FD-4) + an overlapping pair (FD-3/FD-6).

    cps 17 and nothing else, no shots; cue 0 is CF-9's `"Hello, world!!"` whose
    canonical load is 10 rather than the raw 13, cue 1 sits 0.21 s later (inside
    the chaining band), and cue 1 overlaps cue 2 by 0.3 s.
    """
    prof = profile(cps=17.0)
    first = cue(
        0.0,
        0.5,
        surfaces=("Hello,", "world!!"),
        speech=(0.0, 0.5),
        words=[(0.0, 0.25), (0.25, 0.5)],
    )
    second = cue(0.8, 1.5, surfaces=("ok",), speech=(0.8, 1.0), words=[(0.8, 1.0)])
    third = cue(1.2, 2.0, surfaces=("ok",), speech=(1.2, 2.0), words=[(1.2, 2.0)])
    return prof, [first, second, third]


def test_composed_fd1_fd4_fd6_trace_verified_leg_by_leg():
    """Six legs over two sweeps, replayed leg by leg against the validator's state.

    Phase 1: cue 0 wants 10/17 = 0.5882352941176471 (FD-1: the raw load 13 would
    have wanted 13/17); cues 1 and 2 want less than they already have.

    Sweep 1. Cue 0 slot 2 chains 0.588... up to 0.8 - 2f = 0.7166666666666667
    (gap 0.212 is inside CHAIN_MAX_GAP_S and above 2f). Cue 2 slot 6 sees the
    seeded overlap (1.2 < 1.5) and takes ladder branch 1, since cue 1's speech
    end 1.0 clears 1.2 - 2f = 1.1166666666666667.

    Sweep 2. Slot 1 rebases both moved ends to their absolute desires -- cue 0
    back to 0.588..., cue 1 back to its seed end 1.5 -- and slots 2 and 6
    re-derive the same two trims. The state after sweep 2 equals the state after
    sweep 1, so sweep 2 is the fixed point and the trace is exactly:

        chain(0), ladder-1(1), duration-desire(0), chain(0),
        duration-desire(1), ladder-1(1)

    (the parenthesised index is the boundary's cue, not the cue whose slot ran).
    """
    module = fin()
    prof, cues = _composed_document()
    result = run(cues, prof)

    assert result.cues[0]["end"] == 0.8 - TWO_FRAME_S
    assert result.cues[1]["end"] == 1.2 - TWO_FRAME_S
    assert result.report.terminal == "fixed-point"
    assert result.report.max_sweeps_observed == 2
    assert result.report.deltas_fired == ("FD-1", "FD-3", "FD-4", "FD-6", "FD-7")

    legs = result.trace.legs
    assert [leg.rule_id for leg in legs] == [
        "chain",
        "ladder-1",
        "duration-desire",
        "chain",
        "duration-desire",
        "ladder-1",
    ]
    assert [leg.target.cue_index for leg in legs] == [0, 1, 0, 0, 1, 1]
    assert {leg.target.side for leg in legs} == {"end"}
    assert [leg.sweep for leg in legs] == [1, 1, 2, 2, 2, 2]
    assert legs[0].from_value == 10 / 17.0
    assert legs[0].to_value == 0.8 - TWO_FRAME_S
    assert legs[1].from_value == 1.5
    assert legs[1].to_value == 1.2 - TWO_FRAME_S

    seed = module.phase1_stream(cues, profile=prof)
    delivered = tuple((c["start"], c["end"]) for c in result.cues)
    assert (
        module.replay_trace(
            result.trace,
            seed,
            profile=prof,
            evidence=module.FinalizeEvidence(),
            policy=module.FinalizePolicy(),
            delivered=delivered,
        )
        == ()
    )


def test_multi_boundary_cycle():
    """Two boundaries oscillating in ANTI-phase; the minimum is over the stream.

    Two independent cues, each with its own pair of cuts 22 frames apart and the
    11-frame window, so each start oscillates between `low = hi_cut - 12f` and
    `high = lo_cut + 12f` (the 10f/22f shape, twice). Cue 0 is seeded at its LOW
    value and cue 1 at its HIGH one, so the two members of the cycle are
    (low, high) and (high, low).

    Member A = (10f, 10.5) and member B = (12f, 10.416...). Numeric lexicographic
    order compares the first element first: 10f < 12f, so A is adopted -- even
    though A carries the LARGER of cue 1's two values. Adopting per boundary
    would have produced (10f, 10.416...), a state the solver never visited.
    """
    prof = P_ZONE
    offset = 10.0
    seed = [
        cue(10 * F, 5.0, speech=(None, None)),
        cue(offset + 12 * F, 15.0, speech=(None, None)),
    ]
    shots = (0.0, 22 * F, offset, offset + 22 * F)
    result = run(seed, prof, shots=shots)

    assert result.report.terminal == "cycle-adoption"
    assert result.cues[0]["start"] == 0.41666666666666663  # 10f, the lower member
    assert result.cues[1]["start"] == 10.5  # offset + 12f, the HIGHER member
    cycle = result.trace.cycle
    assert cycle is not None and len(cycle.members) == 2
    assert cycle.adopted == min(cycle.members, key=fin().state_key)

    oscillating = [t for t in result.report.entries if t.kind == "shot-unhonored"]
    assert [(t.cue_index, t.evidence["boundary"]) for t in oscillating] == [
        (0, "start"),
        (1, "start"),
    ]
    assert oscillating[0].evidence["values"] == [0.41666666666666663, 0.5]
    assert oscillating[1].evidence["values"] == [10.416666666666666, 10.5]


def test_cycle_minimum_is_numeric_where_byte_order_disagrees():
    """The adoption key is the decoded number, and here the bytes say otherwise.

    Spec section 2.5 forbids ordering by the packed representation. The
    little-endian form of that trap is pinned in ``test_finalizer_properties``;
    this is the BIG-endian one, which needs a negative time to appear: a
    negative double has its sign bit set, so its bytes sort ABOVE every
    non-negative one and byte order inverts exactly at zero.

    The 10f/22f oscillator, translated to sit across zero. Cuts at ``-0.5`` and
    ``-0.5 + 22f = 0.41666666666666663``, snap window 11f, seed start
    ``-0.5 + 10f = -0.08333333333333337``:

    * sweep 1 pairs the seed with the earlier cut (``10f`` away, inside the
      window; the later one is ``12f`` away and outside), ten frames after it,
      so the landing zone pushes the start to ``cut + 12f = 0.0``;
    * sweep 2 pairs ``0.0`` with the LATER cut (``10f`` away; the earlier is now
      ``12f`` away) at ten frames before it, so the lead-in pulls it back to
      ``cut - 12f``, the seed again.

    Adopting the numeric minimum delivers the negative member. Adopting by
    packed bytes -- in either endianness -- would deliver ``0.0``, and the
    assertion below pins that those two answers really are different here, so
    the golden is not quietly agreeing with the thing it forbids.
    """
    module = fin()
    base = -0.5
    start = base + 10 * F
    result = run(
        [cue(start, start + 5.0, speech=(None, None))],
        P_ZONE,
        shots=(base, base + 22 * F),
    )

    assert result.report.terminal == "cycle-adoption"
    assert result.cues[0]["start"] == -0.08333333333333337
    assert result.cues[0]["start"] == start

    cycle = result.trace.cycle
    assert cycle is not None
    assert cycle.adopted == min(cycle.members, key=module.state_key)
    assert min(cycle.members, key=module.pack_state) != cycle.adopted

    tag = next(t for t in result.report.entries if t.kind == "shot-unhonored")
    assert tag.evidence["values"] == [-0.08333333333333337, 0.0]


def test_forged_neighbour_snapshot_is_rejected():
    """A leg that lies about what it read is caught by the validator's own state.

    The trace is the composed document's, whose second leg (ladder branch 1)
    records reading cue 2's start at 1.2. Rewriting that recorded read to 1.4 --
    leaving `from`, `to` and the rule id untouched, so the leg is structurally
    perfect -- must be reported, because the validator checks every read against
    the state IT reconstructed rather than replaying the producer's claim.
    """
    import dataclasses

    module = fin()
    prof, cues = _composed_document()
    result = run(cues, prof)
    seed = module.phase1_stream(cues, profile=prof)
    delivered = tuple((c["start"], c["end"]) for c in result.cues)

    legs = list(result.trace.legs)
    honest = legs[1]
    assert honest.rule_id == "ladder-1"
    assert [read.value for read in honest.reads] == [1.2]
    legs[1] = dataclasses.replace(
        honest,
        reads=(dataclasses.replace(honest.reads[0], value=1.4),),
    )
    forged = dataclasses.replace(result.trace, legs=tuple(legs))

    problems = module.replay_trace(
        forged,
        seed,
        profile=prof,
        evidence=module.FinalizeEvidence(),
        policy=module.FinalizePolicy(),
        delivered=delivered,
    )
    assert problems
    assert any("1.4" in problem and "read" in problem for problem in problems)


def test_evidence_span_math_crosses_the_half_mark_between_the_two_spans():
    """The SPAN MATH under FD-2, pinned without the (W3) ``EvidenceSpan`` object.

    FD-2 says the delivered lyric flag rides the evidence span and is never
    recomputed from the display span. That rule is only worth having if the two
    spans can disagree, and this is the arithmetic that makes them disagree, run
    through the legacy classifier itself rather than through a restated constant:

    * evidence span ``[0.0, 1.25]`` against ``sing_spans = [(0.0, 0.7)]`` ->
      ``0.7 / 1.25 = 0.56``, at or above ``LYRIC_MIN_OVERLAP`` -> lyric;
    * delivered span ``[0.0, 1.5]`` (slot 5 extended the anchorless cue out to
      ``cut + 12f``) -> ``0.7 / 1.5 = 0.4666…``, below it -> NOT lyric.

    So a finalizer that re-classified on what it had just widened would drop a
    flag the evidence supports. The stamped flag survives, which is what
    ``test_anchorless_fd8_then_shot_extension`` asserts on the delivered cue.

    The ``EvidenceSpan`` object itself -- one constructor, endpoint kinds, the
    derived-prefix/suffix fixtures -- lands in W3 with the cost-side suppression
    that consumes it.
    """
    from voxweave.pipeline import LYRIC_MIN_OVERLAP, mark_lyric_cues

    sings = [(0.0, 0.7)]
    evidence_span = {"text": "word", "start": 0.0, "end": 1.25}
    display_span = {"text": "word", "start": 0.0, "end": 1.5}
    mark_lyric_cues([evidence_span, display_span], sings)

    assert 0.7 / 1.25 >= LYRIC_MIN_OVERLAP > 0.7 / 1.5
    assert evidence_span.get("lyric") is True
    assert display_span.get("lyric") is None

    prof = profile(max_cue_s=7.0, lag_out_s=0.25)
    result = run(
        [cue(0.0, 1.25, speech=(None, None), lyric=True)],
        prof,
        shots=(1.0,),
        sings=tuple(sings),
    )
    assert (result.cues[0]["start"], result.cues[0]["end"]) == (0.0, 1.5)
    assert result.cues[0]["lyric"] is True
