# tests/test_partition_check.py
"""The solver-independent hard-contract validator (P4 AD-4 / AD3-3).

``partition_check`` consumes a LOCKED partition plus its cue stream and nothing
else -- no lattice, no cost model, no solver -- because that is exactly the seam
P6 hands an aligner-locked partition through. Every test here therefore builds
the partition by hand: if a check needs the solver to be meaningful, it is in the
wrong module.

Two contracts are load-bearing and pinned individually: BOTH speech anchors are
checked (a validator that only guards the out-time cannot see the shot-snap
in-time class), and only unwaived ``origin="v2"`` violations at stages
``raw``/``core`` may drive the P4 exit -- v1 evidence and legacy-overlay damage
are P5 input, never a reason to fail an otherwise legal v2 shadow.
"""

import json

import pytest

from voxweave.core.partition_check import (
    EPS,
    ORIGINS,
    STAGES,
    VIOLATION_KINDS,
    WAIVER_KINDS,
    PartitionCheckResult,
    Violation,
    Waiver,
    check_partition,
    normalize_text,
    owned_unit_ids,
)
from voxweave.core.segdoc import DisplayProfile, SourceUnit

# ---------------------------------------------------------------- fixtures


def profile(language="en", **over):
    """A resolved profile; every knob explicit so a test never inherits a default."""
    base = dict(
        language=language,
        max_line_length=42,
        max_lines=2,
        clause_ms=400.0,
        vad_skip_ms=1000.0,
        offline_ms=700.0,
        min_cue_s=0.5,
        max_cue_s=7.0,
        glue_gap_s=0.3,
        cps=17.0,
        lag_out_s=0.25,
        shot_snap_s=0.458,
    )
    base.update(over)
    return DisplayProfile(**base)


def units(*spec):
    """``(surface, start, end)`` triples into positional ``SourceUnit``s."""
    return [
        SourceUnit(id=f"u{i}", surface=s, start=a, end=b)
        for i, (s, a, b) in enumerate(spec)
    ]


def cue(text, start, end, *, speech_start=..., speech_end=...):
    out = {"text": text, "start": start, "end": end, "word_data": []}
    if speech_start is not ...:
        out["speech_start"] = speech_start
    if speech_end is not ...:
        out["speech_end"] = speech_end
    return out


def clean_case():
    """Four units, one interior cut, two entirely legal cues."""
    us = units(
        ("the", 0.0, 0.4),
        ("cat", 0.4, 0.9),
        ("sat", 2.0, 2.4),
        ("down", 2.4, 3.0),
    )
    cues = [
        cue("the cat", 0.0, 0.9, speech_start=0.0, speech_end=0.9),
        cue("sat down", 2.0, 3.0, speech_start=2.0, speech_end=3.0),
    ]
    return [2], cues, us


def kinds(result):
    return sorted(v.kind for v in result.violations)


# ---------------------------------------------------------------- vocabulary


def test_eps_is_stated_once():
    """AD3-3: one epsilon, one definition -- not a literal per call site."""
    from pathlib import Path

    import voxweave.core.partition_check as mod

    assert EPS == pytest.approx(1e-6, rel=0, abs=0)
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert source.count("1e-6") == 1


def test_vocabularies_are_sorted_and_closed():
    assert list(VIOLATION_KINDS) == sorted(VIOLATION_KINDS)
    assert list(WAIVER_KINDS) == sorted(WAIVER_KINDS)
    assert "held-chain-duration" in WAIVER_KINDS
    assert set(ORIGINS) == {"v1", "v2"}
    # P5 section 10.3 adds the exit-driving ``finalizer`` stage; the set stays
    # closed, it just grew by exactly one member.
    assert set(STAGES) == {"raw", "core", "legacy-overlay", "finalizer"}


def test_owned_unit_ids_tiles_exactly():
    assert owned_unit_ids([2, 5], 7) == ((0, 2), (2, 5), (5, 7))
    assert owned_unit_ids([], 3) == ((0, 3),)


@pytest.mark.parametrize(
    "left,right",
    [
        ("the cat.", "the cat"),  # strip_punct_for_subtitles
        ("I I know", "I-I know"),  # _merge_stutters
        ("the cat\nsat down", "the cat sat down"),  # wrap_cue_text newline
        ("3.75 kg", "3.75 kg"),  # digit-internal dot survives on both sides
    ],
)
def test_normalize_text_absorbs_the_declared_delta(left, right):
    assert normalize_text(left) == normalize_text(right)


def test_normalize_text_still_separates_real_text():
    assert normalize_text("the cat") != normalize_text("the dog")


# ---------------------------------------------------------------- happy path


def test_clean_partition_has_no_violations():
    partition, cues, us = clean_case()
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert isinstance(result, PartitionCheckResult)
    assert result.violations == ()
    assert result.exit_driving == ()
    assert (result.cue_count, result.unit_count) == (2, 4)


def test_result_to_dict_is_json_stable():
    partition, cues, us = clean_case()
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="core"
    )
    blob = result.to_dict()
    assert blob["origin"] == "v2" and blob["stage"] == "core"
    assert json.dumps(blob, sort_keys=True) == json.dumps(
        result.to_dict(), sort_keys=True
    )


# ---------------------------------------------------------------- conservation


@pytest.mark.parametrize("partition", [[2, 2], [5, 1], [0], [4], [-1]])
def test_unit_conservation_rejects_a_malformed_partition(partition):
    _, cues, us = clean_case()
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert "unit-conservation" in kinds(result)


def test_unit_conservation_rejects_a_cue_count_mismatch():
    partition, cues, us = clean_case()
    result = check_partition(
        partition, cues[:1], units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert "unit-conservation" in kinds(result)


def test_text_conservation_catches_a_dropped_word():
    partition, cues, us = clean_case()
    cues[0]["text"] = "the"
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert kinds(result) == ["text-conservation"]
    assert result.violations[0].unit_ids == (0, 1)


def test_text_conservation_tolerates_strip_stutter_and_wrap():
    us = units(("I", 0.0, 0.2), ("I", 0.2, 0.4), ("know.", 0.4, 1.0))
    cues = [cue("I-I\nknow", 0.0, 1.0, speech_start=0.0, speech_end=1.0)]
    result = check_partition(
        [], cues, units=us, profile=profile(), origin="v2", stage="core"
    )
    assert result.violations == ()


# ---------------------------------------------------------------- time


def test_non_finite_and_reversed_times():
    partition, cues, us = clean_case()
    cues[0]["end"] = float("nan")
    cues[1]["start"] = 3.5
    cues[1]["end"] = 3.0
    cues[1].pop("speech_start")
    cues[1].pop("speech_end")
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert "non-finite-time" in kinds(result)
    assert "reversed-cue" in kinds(result)


def test_an_overlap_is_reported_once_and_not_also_as_non_monotone():
    """One defect, one violation.

    Bug pin. ``non-monotone-time`` used to fire on ``start < prev.end`` as well
    as on ``start < prev.start``, so every overlap raised both kinds and inflated
    the class by 2x in any count-based comparison. p4-api.md section 1 item 5
    defines monotonicity on the *starts* alone.
    """
    partition, cues, us = clean_case()
    cues[1]["start"] = 0.5
    cues[1]["speech_start"] = 0.5
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert kinds(result) == ["overlap"]


def test_non_monotone_time_fires_on_a_start_behind_the_previous_start():
    """Reachable without an overlap: a reversed predecessor ends before it starts."""
    partition, cues, us = clean_case()
    cues[0]["start"] = 2.5
    cues[0]["end"] = 1.9
    cues[0].pop("speech_start")
    cues[0].pop("speech_end")
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert kinds(result) == ["non-monotone-time", "reversed-cue"]


def test_an_off_whitelist_waiver_is_recorded_but_waives_nothing():
    """The whitelist is the point of having one."""
    partition, cues, us = clean_case()
    cues[0]["end"] = 30.0
    cues[1]["start"] = 30.1
    cues[1]["speech_start"] = 30.1
    cues[1]["end"] = 31.0
    cues[1]["speech_end"] = 31.0
    bogus = Waiver(
        kind="because-i-said-so",
        cue_index=0,
        unit_ids=(0, 1),
        span=(0.0, 30.0),
        cap=7.0,
    )
    result = check_partition(
        partition,
        cues,
        units=us,
        profile=profile(),
        origin="v2",
        stage="raw",
        waivers={0: bogus},
    )
    caps = [v for v in result.violations if v.kind == "duration-cap"]
    assert caps and all(not v.waived for v in caps)
    assert result.waivers == (bogus,)


def test_overlap_can_be_switched_off_for_a_lane_that_expects_it():
    partition, cues, us = clean_case()
    cues[1]["start"] = 0.85
    cues[1]["speech_start"] = 0.85
    result = check_partition(
        partition,
        cues,
        units=us,
        profile=profile(),
        origin="v2",
        stage="legacy-overlay",
        expect_no_overlap=False,
    )
    assert "overlap" not in kinds(result)


# ---------------------------------------------------------------- speech anchors


def test_speech_truncation_is_checked_on_the_start_anchor():
    """The shot-snap in-time class: display start moved later than first speech."""
    partition, cues, us = clean_case()
    cues[0]["start"] = 0.2  # speech_start stays 0.0
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="core"
    )
    assert kinds(result) == ["speech-truncated-start"]


def test_speech_truncation_is_checked_on_the_end_anchor():
    partition, cues, us = clean_case()
    cues[0]["end"] = 0.7  # speech_end stays 0.9
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="core"
    )
    assert kinds(result) == ["speech-truncated-end"]


def test_unknown_anchor_skips_only_its_own_side():
    partition, cues, us = clean_case()
    cues[0]["speech_start"] = None
    cues[0]["start"] = 0.2
    cues[0]["end"] = 0.7
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="core"
    )
    assert kinds(result) == ["speech-truncated-end"]


def test_speech_anchor_tolerance_is_exactly_eps():
    partition, cues, us = clean_case()
    cues[0]["start"] = 0.0 + EPS / 2
    cues[0]["end"] = 0.9 - EPS / 2
    assert (
        check_partition(
            partition, cues, units=us, profile=profile(), origin="v2", stage="core"
        ).violations
        == ()
    )


# ---------------------------------------------------------------- layout


def test_line_capacity_counts_rendered_lines():
    us = units(("word", 0.0, 0.4), ("word", 0.4, 0.7), ("word", 0.7, 1.0))
    cues = [cue("word\nword\nword", 0.0, 1.0, speech_start=0.0, speech_end=1.0)]
    result = check_partition(
        [], cues, units=us, profile=profile(), origin="v2", stage="core"
    )
    assert kinds(result) == ["line-capacity"]


def test_line_capacity_catches_an_overwide_line():
    us = units(("x" * 60, 0.0, 1.0))
    cues = [cue("x" * 60, 0.0, 1.0, speech_start=0.0, speech_end=1.0)]
    result = check_partition(
        [], cues, units=us, profile=profile(), origin="v2", stage="core"
    )
    assert "line-capacity" in kinds(result)


def test_an_empty_display_projection_fits_any_budget():
    """AD4-1: the all-invisible cue's empty rendered text must be legal here."""
    us = units(("!", 0.0, 0.0), ("!", 1.0, 1.0))
    cues = [cue("", 0.0, 1.0, speech_start=0.0, speech_end=1.0)]
    result = check_partition(
        [], cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert result.violations == ()


# ---------------------------------------------------------------- duration cap


def test_duration_cap_violation_and_its_waiver_provenance():
    us = units(("held", 0.0, 8.0))
    cues = [cue("held", 0.0, 8.0, speech_start=0.0, speech_end=8.0)]
    bare = check_partition(
        [], cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    assert kinds(bare) == ["duration-cap"]
    assert bare.exit_driving == bare.violations

    waiver = Waiver(
        kind="held-chain-duration",
        cue_index=0,
        unit_ids=(0,),
        span=(0.0, 8.0),
        cap=7.0,
        detail="single timed unit",
    )
    waived = check_partition(
        [],
        cues,
        units=us,
        profile=profile(),
        origin="v2",
        stage="raw",
        waivers={0: waiver},
    )
    assert kinds(waived) == ["duration-cap"]
    assert waived.violations[0].waived is True
    assert waived.violations[0].waived_by == waiver
    assert waived.unwaived == ()
    assert waived.exit_driving == ()
    assert waived.waivers == (waiver,)
    assert waived.violations[0].to_dict()["waived_by"]["cap"] == 7.0


def test_a_zero_cap_disables_the_duration_check():
    """AD-7/AD3-4: zero disables; a negative cap never reaches here (profile preflight)."""
    us = units(("held", 0.0, 8.0))
    cues = [cue("held", 0.0, 8.0, speech_start=0.0, speech_end=8.0)]
    result = check_partition(
        [], cues, units=us, profile=profile(max_cue_s=0.0), origin="v2", stage="raw"
    )
    assert result.violations == ()


# ---------------------------------------------------------------- attribution


@pytest.mark.parametrize(
    "origin,stage,drives",
    [
        ("v2", "raw", True),
        ("v2", "core", True),
        ("v2", "legacy-overlay", False),
        ("v1", "raw", False),
        ("v1", "core", False),
        ("v1", "legacy-overlay", False),
    ],
)
def test_only_unwaived_v2_raw_or_core_violations_drive_the_exit(origin, stage, drives):
    """AD3-3: v1 evidence and overlay damage are P5 input, never a P4 exit driver."""
    us = units(("held", 0.0, 8.0))
    cues = [cue("held", 0.0, 8.0, speech_start=0.0, speech_end=8.0)]
    result = check_partition(
        [], cues, units=us, profile=profile(), origin=origin, stage=stage
    )
    assert kinds(result) == ["duration-cap"]
    assert bool(result.exit_driving) is drives
    only = result.violations[0]
    assert (only.origin, only.stage) == (origin, stage)
    assert only.exit_driving is drives


def test_every_violation_carries_origin_and_stage():
    partition, cues, us = clean_case()
    cues[0]["text"] = "the"
    cues[1]["start"] = 0.5
    result = check_partition(
        partition,
        cues,
        units=us,
        profile=profile(),
        origin="v1",
        stage="legacy-overlay",
    )
    assert result.violations
    for violation in result.violations:
        assert isinstance(violation, Violation)
        assert violation.origin == "v1"
        assert violation.stage == "legacy-overlay"
        assert violation.kind in VIOLATION_KINDS


# ------------------------------------------- the finalizer stage (P5 section 10.3)
#
# Three predicates come alive only at ``stage="finalizer"``. Every one of them is
# a question about DELIVERED text or DELIVERED bounds, which is why it cannot be
# asked at ``raw``: a raw cue's text has not been through the wrap pass, and a
# raw stream has not been through the ladder.


def two_frame_case(gap):
    """Two legal cues separated by ``gap`` seconds."""
    partition, cues, us = clean_case()
    cues[0]["end"] = round(2.0 - gap, 6)
    return partition, cues, us


def min_gap_tag(cue_index, *, speech_end, next_start):
    from voxweave.core.partition_check import ReportTag

    return ReportTag(
        kind="min-gap-unmet",
        cue_index=cue_index,
        evidence={
            "next_start": next_start,
            "prev_end_before": 2.4,
            "resulting_gap": next_start - speech_end,
            "speech_end": speech_end,
        },
    )


def test_finalizer_stage_reports_an_unexplained_sub_two_frame_gap():
    partition, cues, us = two_frame_case(0.02)
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="finalizer"
    )
    assert "min-gap" in kinds(result)


def test_a_min_gap_unmet_tag_naming_the_left_cue_explains_the_gap():
    """Speech running into the two-frame band is legal WHEN it is reported."""
    partition, cues, us = two_frame_case(0.02)
    cues[0]["speech_end"] = 1.98
    result = check_partition(
        partition,
        cues,
        units=us,
        profile=profile(),
        origin="v2",
        stage="finalizer",
        reports=[min_gap_tag(0, speech_end=1.98, next_start=2.0)],
    )
    assert "min-gap" not in kinds(result)
    assert "forged-report" not in kinds(result)


def test_a_tag_whose_branch_does_not_recompute_is_a_forged_report():
    """PF-3: the evidence must re-derive the branch, or the tag is not evidence."""
    partition, cues, us = two_frame_case(0.02)
    cues[0]["speech_end"] = 1.98
    forged = min_gap_tag(0, speech_end=1.98, next_start=2.0)
    lying = type(forged)(
        kind=forged.kind,
        cue_index=0,
        evidence={**dict(forged.evidence), "resulting_gap": 0.0},
    )
    result = check_partition(
        partition,
        cues,
        units=us,
        profile=profile(),
        origin="v2",
        stage="finalizer",
        reports=[lying],
    )
    assert "forged-report" in kinds(result)


def test_exactly_two_frames_apart_is_legal():
    """The floor is inclusive: a gap AT two frames is what chaining aims for."""
    from voxweave.core.timing import TWO_FRAME_S

    partition, cues, us = two_frame_case(TWO_FRAME_S)
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="finalizer"
    )
    assert "min-gap" not in kinds(result)


def test_true_crosstalk_stays_an_overlap_and_no_tag_excuses_it():
    """Ladder branch 3: the trim stops at speech and the overlap STANDS.

    ``min-gap`` and ``overlap`` split the band at zero, and only the first is
    excusable by a report. A tag naming the left cue must not launder a real
    overlap into a reported near-miss -- which is the whole reason the two
    predicates are separate kinds rather than one severity.
    """
    partition, cues, us = two_frame_case(-0.1)
    cues[0]["speech_end"] = 2.1
    result = check_partition(
        partition,
        cues,
        units=us,
        profile=profile(),
        origin="v2",
        stage="finalizer",
        reports=[min_gap_tag(0, speech_end=2.1, next_start=2.0)],
    )
    assert "overlap" in kinds(result)
    assert "min-gap" not in kinds(result)
    # ... and the tag itself does not recompute as a branch-2 outcome.
    assert "forged-report" in kinds(result)


def test_a_tag_cannot_hide_crosstalk_inside_the_comparison_tolerance():
    """PF-3's band test is not implied by the other two, and this is the gap.

    The first two conditions -- the reported gap sits in ``[0, 2f)`` and equals
    ``next_start - speech_end`` -- pin the speech end only to within ``EPS``,
    because the second is an approximate equality. That slack is exactly wide
    enough to smuggle in a branch-3 fact: speech that runs 0.5 microseconds PAST
    the next start is true crosstalk, yet a tag claiming ``resulting_gap = 0.0``
    passes both tests (``|0.0 - (-5e-07)| = 5e-07 <= EPS``).

    The third condition is the branch-2 precondition itself (``speech_end <=
    next_start``), asked exactly rather than within tolerance, and it is the
    only one that refuses. The delivered stream here is otherwise clean -- the
    cues touch at ``2.0``, a zero gap the tag would legitimately explain -- so
    ``forged-report`` is the whole finding, not a side effect of a broken
    stream.
    """
    partition, cues, us = two_frame_case(0.0)
    cues[0]["speech_end"] = 2.0000005
    smuggled = min_gap_tag(0, speech_end=2.0000005, next_start=2.0)
    tag = type(smuggled)(
        kind=smuggled.kind,
        cue_index=0,
        evidence={**dict(smuggled.evidence), "resulting_gap": 0.0},
    )
    result = check_partition(
        partition,
        cues,
        units=us,
        profile=profile(),
        origin="v2",
        stage="finalizer",
        reports=[tag],
    )
    assert kinds(result) == ["forged-report"]


def test_the_new_predicates_are_silent_at_every_other_stage():
    partition, cues, us = two_frame_case(0.02)
    for stage in ("raw", "core", "legacy-overlay"):
        result = check_partition(
            partition, cues, units=us, profile=profile(), origin="v2", stage=stage
        )
        assert "min-gap" not in kinds(result), stage


def test_finalizer_line_capacity_bypasses_the_rewrap_excuse():
    """PF-1: what ships is measured, not what some rewrap could have folded.

    A 49-cell single line has no newline, so ``_wrappable`` answers "a wrap
    could still fit this" and the raw stage forgives it. At the finalizer stage
    the wrap has run and this IS the delivered line, so it is a violation.
    """
    text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj"
    us = units((text, 0.0, 3.0))
    cues = [cue(text, 0.0, 3.0, speech_start=0.0, speech_end=3.0)]
    forgiving = check_partition(
        [], cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    strict = check_partition(
        [], cues, units=us, profile=profile(), origin="v2", stage="finalizer"
    )
    assert "line-capacity" not in kinds(forgiving)
    assert "line-capacity" in kinds(strict)


def test_violations_are_sorted_for_byte_stable_artifacts():
    partition, cues, us = clean_case()
    cues[0]["text"] = "the"
    cues[0]["end"] = 0.7
    cues[1]["start"] = 0.5
    result = check_partition(
        partition, cues, units=us, profile=profile(), origin="v2", stage="raw"
    )
    keys = [
        (v.cue_index if v.cue_index is not None else -1, v.kind)
        for v in result.violations
    ]
    assert keys == sorted(keys)
