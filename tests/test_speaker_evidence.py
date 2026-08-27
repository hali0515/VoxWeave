"""P5 W3 speaker evidence, pricing, projection, and lineage goldens.

Every numeric expectation below is derived directly from LAW section 6.  The
tests intentionally work in source-unit coordinates: proportional child timing
is display timing and must never become speaker evidence.
"""

from __future__ import annotations

import copy

import pytest

from voxweave.core.boundary_cost import transition_time
from voxweave.core.schema import Cue
from voxweave.core.segdoc import DisplayProfile, SegDocument, SourceUnit
from voxweave.core.speaker_evidence import (
    BUCKET_KINDS,
    ENDPOINT_KINDS,
    EXPRESS_TOL_S,
    SPEAKER_EDGE_RUN_MIN_S,
    SPEAKER_MIN_RUN_S,
    SPEAKER_MULTI_MIN_FRAC,
    SPEAKER_UNIT_COVER_FRAC,
    TURN_STATES,
    W_SPEAKER_INTERIOR,
    BoundaryPoint,
    EvidenceSpan,
    LiveSpeakerEvent,
    SpeakerEvidenceError,
    SpeakerProjectionError,
    annotate_speaker_ids,
    injective_time_match,
    lyric_for_evidence,
    make_evidence_span,
    measure_speaker_events,
    named_multi_cues_unannotated,
    speaker_edge_cost,
    speaker_evidence,
)


def profile(language: str = "en", *, max_lines: int = 2) -> DisplayProfile:
    return DisplayProfile(
        language=language,
        max_line_length=42 if language == "en" else 18,
        max_lines=max_lines,
        clause_ms=400.0,
        vad_skip_ms=250.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=7.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=11 / 24,
    )


def unit(
    index: int,
    surface: str,
    start: float | None,
    end: float | None,
    *,
    provenance: str = "aligner",
) -> SourceUnit:
    return SourceUnit(
        id=f"u{index}",
        surface=surface,
        start=start,
        end=end,
        provenance=provenance,
    )


def document(
    units: list[SourceUnit],
    turns: list[tuple[float, float, str]] | None,
    *,
    language: str = "en",
) -> SegDocument:
    separator = " " if language == "en" else ""
    return SegDocument(
        language=language,
        units=units,
        profile=profile(language),
        vad_speech=None,
        shot_changes=None,
        sing_spans=None,
        speaker_turns=turns,
        manifest={},
        text=separator.join(item.surface for item in units),
    )


def cue(text: str = "x") -> Cue:
    return {
        "text": text,
        "start": 0.0,
        "end": 1.0,
        "word_data": [{"text": text, "start": 0.0, "end": 1.0}],
        "speech_start": 0.0,
        "speech_end": 1.0,
    }


# ---------------------------------------------------------------- constants


def test_w3_constants_and_closed_vocabularies_are_frozen():
    assert SPEAKER_UNIT_COVER_FRAC == 0.5
    assert SPEAKER_MULTI_MIN_FRAC == 0.25
    assert SPEAKER_MIN_RUN_S == 0.2
    assert SPEAKER_EDGE_RUN_MIN_S == 0.12
    assert W_SPEAKER_INTERIOR == 3.0
    assert EXPRESS_TOL_S == 0.5
    assert ENDPOINT_KINDS == ("exact", "fabricated")
    assert set(TURN_STATES) == {
        "absent",
        "overlap",
        "multi",
        "single",
        "unattributed",
    }
    assert BUCKET_KINDS == (
        "expressed",
        "policy_filtered",
        "survived_expressible_but_missed",
        "unattributed_loss",
        "unexpressible",
    )


def test_transition_time_uses_right_start_across_a_nonzero_gap():
    # R8-2: 1.4 is the cut-price coordinate; the 0.4 s gap midpoint and the
    # left end are both tempting but are not the frozen helper's answer.
    assert transition_time(1.0, 1.4) == 1.4
    assert transition_time(1.0, None) == 1.0
    assert transition_time(None, None) is None


# ------------------------------------------------------------- attribution


@pytest.mark.parametrize(
    ("turns", "expected_kind", "expected_label", "expected_support"),
    [
        ([(0.0, 0.5, "A")], "single", "A", ("A",)),
        # Parent classification is multi-first.  Exactly 50/50 and exactly
        # 75/25 therefore remain structural ambiguity, never a tie-picked label.
        ([(0.0, 0.5, "A"), (0.5, 1.0, "B")], "multi", None, ("A", "B")),
        ([(0.0, 0.75, "A"), (0.75, 1.0, "B")], "multi", None, ("A", "B")),
        # 0.80 >= cover while 0.15 < multi-min: this is single A.
        ([(0.0, 0.8, "A"), (0.8, 0.95, "B")], "single", "A", ("A",)),
        ([], "none", None, ()),
    ],
)
def test_parent_attribution_threshold_goldens(
    turns, expected_kind, expected_label, expected_support
):
    result = speaker_evidence(document([unit(0, "x", 0.0, 1.0)], turns))
    parent = result.parent_speakers[0]
    assert (parent.kind, parent.label, parent.support) == (
        expected_kind,
        expected_label,
        expected_support,
    )


@pytest.mark.parametrize(("start", "end"), [(1.0, 1.0), (2.0, 1.0), (None, 1.0)])
def test_zero_negative_or_unlocated_parent_is_none(start, end):
    result = speaker_evidence(document([unit(0, "x", start, end)], [(0.0, 3.0, "A")]))
    assert result.parent_speakers[0].kind == "none"
    assert result.labels == (None,)


def test_attribution_accumulates_disjoint_turns_of_the_same_label():
    # A covers 0.3 + 0.3 = 0.6 of the one-second parent.  Looking only at the
    # longest constituent would miss the 0.5 cover floor.
    turns = [(0.0, 0.3, "A"), (0.3, 0.4, "B"), (0.7, 1.0, "A")]
    result = speaker_evidence(document([unit(0, "x", 0.0, 1.0)], turns))
    assert result.parent_speakers[0].kind == "single"
    assert result.parent_speakers[0].label == "A"


def test_turn_track_absence_is_distinct_from_present_but_unattributed():
    parent = [unit(0, "x", 0.0, 1.0)]
    absent = speaker_evidence(document(parent, None))
    present = speaker_evidence(document(parent, []))
    span = EvidenceSpan(0.0, 1.0, "exact", "exact")
    assert (
        speaker_edge_cost(absent, (0, 1), evidence_span=span).features["turn_state"]
        == "absent"
    )
    assert (
        speaker_edge_cost(present, (0, 1), evidence_span=span).features["turn_state"]
        == "unattributed"
    )


# ------------------------------------------------------------ conditioning


def test_neighbour_fill_uses_left_label_and_fills_a_leading_prefix():
    units = [
        unit(0, "lead", 0.0, 1.0),
        unit(1, "A", 1.0, 2.0),
        unit(2, "hole", 2.0, 3.0),
        unit(3, "B", 3.0, 4.0),
    ]
    turns = [(1.0, 2.0, "A"), (3.0, 4.0, "B")]
    result = speaker_evidence(document(units, turns))
    assert result.labels == ("A", "A", "A", "B")
    assert result.stats.filled == 2


def test_neighbour_fill_never_crosses_the_robust_silence_threshold():
    # Edge silence = (250 + 50) ms = 0.300 s.  Equality is a region boundary,
    # so the unlabeled trailing region cannot inherit A.
    units = [unit(0, "A", 0.0, 1.0), unit(1, "hole", 1.3, 2.0)]
    result = speaker_evidence(document(units, [(0.0, 1.0, "A")]))
    assert result.edge_silence_s == 0.3
    assert result.labels == ("A", None)


def test_a_b_a_sub_floor_run_is_absorbed_to_a_fixpoint():
    units = [
        unit(0, "a", 0.0, 0.4),
        unit(1, "b", 0.4, 0.5),
        unit(2, "a", 0.5, 0.9),
    ]
    turns = [(0.0, 0.4, "A"), (0.4, 0.5, "B"), (0.5, 0.9, "A")]
    result = speaker_evidence(document(units, turns))
    assert result.labels == ("A", "A", "A")
    assert result.stats.runs_absorbed == 1
    assert result.stats.transitions_before == 2
    assert result.stats.transitions_after == 0


@pytest.mark.parametrize(
    ("duration", "expected"), [(0.16, ("A", "B")), (0.08, ("A", "A"))]
)
def test_edge_run_160ms_survives_but_80ms_noise_is_absorbed(duration, expected):
    units = [unit(0, "a", 0.0, 0.5), unit(1, "b", 0.5, 0.5 + duration)]
    turns = [(0.0, 0.5, "A"), (0.5, 0.5 + duration, "B")]
    assert speaker_evidence(document(units, turns)).labels == expected


def test_distinct_neighbour_tiny_run_absorbs_into_longer_right_tie_goes_left():
    units = [
        unit(0, "a", 0.0, 0.3),
        unit(1, "b", 0.3, 0.38),
        unit(2, "c", 0.38, 0.88),
    ]
    turns = [(0.0, 0.3, "A"), (0.3, 0.38, "B"), (0.38, 0.88, "C")]
    assert speaker_evidence(document(units, turns)).labels == ("A", "C", "C")

    tied = [
        unit(0, "a", 0.0, 0.4),
        unit(1, "b", 0.4, 0.48),
        unit(2, "c", 0.48, 0.88),
    ]
    tied_turns = [(0.0, 0.4, "A"), (0.4, 0.48, "B"), (0.48, 0.88, "C")]
    assert speaker_evidence(document(tied, tied_turns)).labels == ("A", "A", "C")


def test_absorb_runs_before_phrase_snap_on_composite_shape(monkeypatch):
    # If snap ran first, B's 0.15 s would beat A's 0.14 s in phrase [0,2),
    # producing B,B,A.  LAW order absorbs the A-B-A sandwich first -> A,A,A.
    monkeypatch.setattr(
        "voxweave.core.speaker_evidence._phrase_ranges",
        lambda _units, _lang: ((0, 2), (2, 3)),
    )
    units = [
        unit(0, "片", 0.0, 0.14),
        unit(1, "付ける", 0.14, 0.29),
        unit(2, "次", 0.29, 0.59),
    ]
    turns = [(0.0, 0.14, "A"), (0.14, 0.29, "B"), (0.29, 0.59, "A")]
    result = speaker_evidence(document(units, turns, language="ja"))
    assert result.labels == ("A", "A", "A")
    assert result.stats.runs_absorbed == 1


def test_phrase_snap_uses_duration_vote_and_all_none_stays_none(monkeypatch):
    monkeypatch.setattr(
        "voxweave.core.speaker_evidence._phrase_ranges",
        lambda _units, _lang: ((0, 2), (2, 3)),
    )
    units = [
        unit(0, "大", 0.0, 0.3),
        # B's 160 ms edge run survives absorb, then loses the duration vote.
        unit(1, "碴子", 0.3, 0.46),
        # A separate all-none silence region reaches phrase snap unchanged.
        unit(2, "。", 0.8, 1.2),
    ]
    turns = [(0.0, 0.3, "A"), (0.3, 0.46, "B")]
    result = speaker_evidence(document(units, turns, language="zh"))
    assert result.labels == ("A", "A", None)
    assert result.stats.phrase_snaps == 1


# --------------------------------------------------------- parent projection


def test_projection_uses_complete_origin_and_never_child_timing():
    parents = [
        unit(0, "single", 0.0, 1.0),
        unit(1, "multi", 1.0, 2.0),
        # The exact robust-silence boundary keeps this parent unfilled.
        unit(2, "none", 2.4, 3.4),
    ]
    turns = [(0.0, 1.0, "A"), (1.0, 1.5, "A"), (1.5, 2.0, "B")]
    # These derived child spans deliberately look like strong, alternating B/A
    # evidence.  Reading them would produce the wrong result.  The only lawful
    # ownership path is parent -> complete origin tuple.
    refined = (
        unit(0, "sin", 0.0, 0.5, provenance="subunit-per-char"),
        unit(1, "gle", 0.5, 1.0, provenance="subunit-per-char"),
        unit(2, "mul", 1.0, 1.5, provenance="subunit-per-char"),
        unit(3, "ti", 1.5, 2.0, provenance="subunit-per-char"),
        unit(4, "none", 2.4, 3.4),
    )
    result = speaker_evidence(
        document(parents, turns), refined_units=refined, origin=(0, 0, 1, 1, 2)
    )
    assert tuple(item.kind for item in result.unit_speakers) == (
        "single",
        "single",
        "ambiguous",
        "ambiguous",
        "none",
    )
    assert result.labels == ("A", "A", None, None, None)
    assert result.origin == (0, 0, 1, 1, 2)
    assert result.stats.multilabel_parents == 1


@pytest.mark.parametrize(
    ("origin", "message"),
    [((0,), "cardinality"), ((0, 2), "complete"), ((1, 0), "monotone")],
)
def test_projection_rejects_incomplete_or_nonmonotone_origin(origin, message):
    parent = document([unit(0, "a", 0.0, 1.0), unit(1, "b", 1.0, 2.0)], None)
    refined = (unit(0, "a", 0.0, 1.0), unit(1, "b", 1.0, 2.0))
    with pytest.raises(SpeakerEvidenceError, match=message):
        speaker_evidence(parent, refined_units=refined, origin=origin)


# ------------------------------------------------------------ EvidenceSpan


def test_evidence_span_resolves_each_endpoint_independently():
    units = [
        unit(0, "derived", 0.0, 1.0, provenance="subunit-phrase"),
        unit(1, "exact", 1.0, 2.0),
    ]
    span = make_evidence_span(units, (0, 2), input_start=-0.25, input_end=2.25)
    assert span == EvidenceSpan(-0.25, 2.0, "fabricated", "exact")

    units = [
        unit(0, "exact", 0.0, 1.0),
        unit(1, "derived", 1.0, 2.0, provenance="subunit-phrase"),
    ]
    span = make_evidence_span(units, (0, 2), input_start=-0.25, input_end=2.25)
    assert span == EvidenceSpan(0.0, 2.25, "exact", "fabricated")


def test_evidence_span_ghost_aligner_bound_is_fabricated():
    units = [unit(0, "ghost", None, 0.5), unit(1, "x", 0.5, 1.0)]
    assert make_evidence_span(
        units, (0, 2), input_start=0.25, input_end=1.0
    ) == EvidenceSpan(0.25, 1.0, "fabricated", "exact")


def test_lyric_threshold_is_inclusive_and_uses_evidence_not_delivered_span():
    # Evidence [0,1.25] overlaps singing for 0.625 s: exactly 0.5 -> lyric.
    # The shot-extended display [0,1.5] would be only 5/12 sung, but must never
    # be consulted by the v2 set-and-clear predicate.
    evidence = EvidenceSpan(0.0, 1.25, "fabricated", "fabricated")
    singing = [(0.0, 0.625)]
    assert lyric_for_evidence(evidence, singing) is True
    assert (
        lyric_for_evidence(EvidenceSpan(0.0, 1.5, "exact", "exact"), singing) is False
    )


def test_evidence_span_rejects_nonfinite_or_reversed_bounds():
    with pytest.raises(SpeakerEvidenceError):
        EvidenceSpan(float("nan"), 1.0, "exact", "exact")
    with pytest.raises(SpeakerEvidenceError):
        EvidenceSpan(2.0, 1.0, "exact", "exact")


# --------------------------------------------------------------- edge price


def test_edge_price_counts_only_label_changes_strictly_inside():
    units = [
        unit(0, "a", 0.0, 1.0),
        unit(1, "a", 1.0, 2.0),
        unit(2, "b", 2.0, 3.0),
    ]
    evidence = speaker_evidence(document(units, [(0.0, 2.0, "A"), (2.0, 3.0, "B")]))
    span = EvidenceSpan(0.0, 3.0, "exact", "exact")
    price = speaker_edge_cost(evidence, (0, 3), evidence_span=span)
    assert price.features == {
        "speaker_changes_in_cue_raw": 1.0,
        "suppressed_lyric": False,
        "turn_state": "multi",
        "two_speaker_raw": 1.0,
    }
    assert price.weighted_terms == {"speaker_interior": 3.0}
    assert price.total == 3.0

    # Cutting at unit 2 moves that change onto the edge endpoint; neither child
    # edge contains an interior transition.
    assert (
        speaker_edge_cost(
            evidence,
            (0, 2),
            evidence_span=EvidenceSpan(0.0, 2.0, "exact", "exact"),
        ).features["speaker_changes_in_cue_raw"]
        == 0.0
    )


def test_none_label_edges_are_not_transitions_and_state_remains_single():
    units = [unit(0, "none", 0.0, 1.0), unit(1, "a", 1.3, 2.0)]
    evidence = speaker_evidence(document(units, [(1.3, 2.0, "A")]))
    price = speaker_edge_cost(
        evidence,
        (0, 2),
        evidence_span=EvidenceSpan(0.0, 2.0, "exact", "exact"),
    )
    assert evidence.labels == (None, "A")
    assert price.features["speaker_changes_in_cue_raw"] == 0.0
    assert price.features["turn_state"] == "single"


def test_raw_simultaneous_distinct_turns_take_overlap_precedence():
    units = [unit(0, "x", 0.0, 1.0)]
    evidence = speaker_evidence(document(units, [(0.0, 0.8, "A"), (0.2, 1.0, "B")]))
    price = speaker_edge_cost(
        evidence,
        (0, 1),
        evidence_span=EvidenceSpan(0.0, 1.0, "exact", "exact"),
    )
    assert price.features["turn_state"] == "overlap"


def test_multi_parent_support_takes_multi_state_even_without_a_vote():
    units = [unit(0, "x", 0.0, 1.0)]
    evidence = speaker_evidence(document(units, [(0.0, 0.75, "A"), (0.75, 1.0, "B")]))
    price = speaker_edge_cost(
        evidence,
        (0, 1),
        evidence_span=EvidenceSpan(0.0, 1.0, "exact", "exact"),
    )
    assert price.features["turn_state"] == "multi"
    assert price.features["two_speaker_raw"] == 1.0


def test_lyric_suppresses_the_term_but_keeps_the_raw_feature():
    units = [unit(0, "a", 0.0, 1.0), unit(1, "b", 1.0, 2.0)]
    evidence = speaker_evidence(document(units, [(0.0, 1.0, "A"), (1.0, 2.0, "B")]))
    price = speaker_edge_cost(
        evidence,
        (0, 2),
        evidence_span=EvidenceSpan(0.0, 2.0, "exact", "exact"),
        sing_spans=[(0.0, 1.0)],
    )
    assert price.features["speaker_changes_in_cue_raw"] == 1.0
    assert price.features["suppressed_lyric"] is True
    assert price.weighted_terms["speaker_interior"] == 0.0


# --------------------------------------------------------- speaker id pass


def test_speaker_id_projection_uses_selected_ranges_and_only_metadata():
    units = [
        unit(0, "a", 0.0, 1.0),
        unit(1, "a", 1.0, 2.0),
        unit(2, "m", 2.0, 3.0),
        unit(3, "m", 3.0, 4.0),
        unit(4, "none", 4.4, 5.0),
    ]
    turns = [
        (0.0, 2.0, "A"),
        (2.0, 2.5, "A"),
        (2.5, 4.0, "B"),
    ]
    evidence = speaker_evidence(document(units, turns))
    cues = [cue("one"), cue("two"), cue("three")]
    cues[1]["speaker_ids"] = ["STALE"]
    cues[2]["speaker_ids"] = ["STALE"]
    before = copy.deepcopy(cues)

    assert (
        annotate_speaker_ids(cues, ((0, 2), (2, 4), (4, 5)), evidence.unit_speakers)
        is None
    )
    assert cues[0]["speaker_ids"] == ["A"]
    assert "speaker_ids" not in cues[1]
    assert "speaker_ids" not in cues[2]
    for index, current in enumerate(cues):
        assert {k: v for k, v in current.items() if k != "speaker_ids"} == {
            k: v for k, v in before[index].items() if k != "speaker_ids"
        }
    assert (
        named_multi_cues_unannotated(((0, 2), (2, 4), (4, 5)), evidence.unit_speakers)
        == 1
    )


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        (((0, 2),), "cardinality"),
        (((0, 1), (2, 3)), "contiguous"),
        (((0, 2), (1, 3)), "contiguous"),
        (((1, 2), (2, 3)), "start at 0"),
        (((0, 1), (1, 2)), "cover all"),
        (((0, 0), (0, 3)), "positive"),
    ],
)
def test_speaker_id_projection_rejects_bad_ownership(ranges, message):
    evidence = speaker_evidence(
        document(
            [unit(0, "a", 0.0, 1.0), unit(1, "b", 1.0, 2.0), unit(2, "c", 2.0, 3.0)],
            None,
        )
    )
    with pytest.raises(SpeakerProjectionError, match=message):
        annotate_speaker_ids([cue(), cue()], ranges, evidence.unit_speakers)


# ----------------------------------------------------------- event lineage


def test_injective_match_one_event_two_boundaries_chooses_earlier_boundary():
    events = (LiveSpeakerEvent("e0", 0, 1.0),)
    boundaries = (BoundaryPoint("b0", 0, 0.75), BoundaryPoint("b1", 1, 1.25))
    matches = injective_time_match(events, boundaries)
    assert [(item.event_id, item.boundary_id, item.distance) for item in matches] == [
        ("e0", "b0", 0.25)
    ]


def test_injective_match_equal_distance_duplicate_events_reserves_earlier_id():
    events = (LiveSpeakerEvent("e0", 0, 1.0), LiveSpeakerEvent("e1", 1, 1.0))
    boundaries = (BoundaryPoint("b0", 0, 1.1),)
    matches = injective_time_match(events, boundaries)
    assert [(item.event_id, item.boundary_id) for item in matches] == [("e0", "b0")]


def test_injective_match_ties_use_ids_not_caller_supplied_indices():
    events = (
        LiveSpeakerEvent("z-event", 0, 1.0),
        LiveSpeakerEvent("a-event", 99, 1.0),
    )
    boundaries = (
        BoundaryPoint("z-boundary", 0, 1.1),
        BoundaryPoint("a-boundary", 99, 1.1),
    )
    matches = injective_time_match(events, boundaries)
    assert [(item.event_id, item.boundary_id) for item in matches] == [
        ("a-event", "a-boundary"),
        ("z-event", "z-boundary"),
    ]


def test_injective_match_rejects_duplicate_endpoint_identities():
    events = (
        LiveSpeakerEvent("duplicate", 0, 1.0),
        LiveSpeakerEvent("duplicate", 1, 2.0),
    )
    with pytest.raises(SpeakerEvidenceError, match="unique"):
        injective_time_match(events, (BoundaryPoint("b0", 0, 1.0),))


def test_expression_matching_and_miss_bucket_conserve_raw_denominator():
    units = [unit(0, "a", 0.0, 1.0), unit(1, "b", 1.0, 2.0)]
    evidence = speaker_evidence(document(units, [(0.0, 1.0, "A"), (1.0, 2.0, "B")]))
    expressed = measure_speaker_events(evidence, delivered_boundaries=(0.75, 1.25))
    assert expressed.buckets == {
        "expressed": 1,
        "policy_filtered": 0,
        "survived_expressible_but_missed": 0,
        "unattributed_loss": 0,
        "unexpressible": 0,
    }
    assert expressed.matches[0].boundary_time == 0.75

    missed = measure_speaker_events(evidence, delivered_boundaries=())
    assert missed.buckets["survived_expressible_but_missed"] == 1
    assert sum(missed.buckets.values()) == missed.raw_in_speech_turn_changes == 1


def test_a_a_b_c_c_phrase_collision_preserves_b_to_c_ancestry(monkeypatch):
    # Attribution transitions are A->B@2 (e0) and B->C@3 (e1).  Phrase snap
    # yields A,A,A,C,C, so the sole output @3 matches e1 at distance 0; e0 is
    # unmatched and therefore policy_filtered.
    monkeypatch.setattr(
        "voxweave.core.speaker_evidence._phrase_ranges",
        lambda _units, _lang: ((0, 3), (3, 5)),
    )
    units = [unit(i, text, float(i), float(i + 1)) for i, text in enumerate("aabcc")]
    turns = [(0.0, 2.0, "A"), (2.0, 3.0, "B"), (3.0, 5.0, "C")]
    evidence = speaker_evidence(document(units, turns, language="ja"))
    measurement = measure_speaker_events(evidence, delivered_boundaries=(3.0,))
    assert evidence.labels == ("A", "A", "A", "C", "C")
    assert measurement.event_buckets == {"e0": "policy_filtered", "e1": "expressed"}


def test_unmatched_initial_event_takes_literal_policy_filtered_branch():
    # A labels parent 0.  Parents 1/2 each receive only 0.1/0.5 overlap and are
    # raw-none, then fill to A.  The raw A->B@1 event has no attribution-stage
    # transition to attach to; it is policy_filtered, not a catch-all miss.
    units = [
        unit(0, "a", 0.0, 0.5),
        unit(1, "x", 0.5, 1.0),
        unit(2, "y", 1.0, 1.5),
    ]
    turns = [(0.0, 0.6, "A"), (1.0, 1.1, "B")]
    evidence = speaker_evidence(document(units, turns))
    measurement = measure_speaker_events(evidence, delivered_boundaries=(1.0,))
    assert evidence.labels == ("A", "A", "A")
    assert measurement.event_buckets == {"e0": "policy_filtered"}


def test_unattributed_loss_precedes_the_unmatched_initial_branch():
    units = [
        unit(0, "x", 0.0, 0.5),
        unit(1, "y", 0.5, 1.0),
        unit(2, "z", 1.0, 1.5),
    ]
    turns = [(0.9, 1.0, "A"), (1.0, 1.1, "B")]
    evidence = speaker_evidence(document(units, turns))
    measurement = measure_speaker_events(evidence, delivered_boundaries=(1.0,))
    assert evidence.labels == (None, None, None)
    assert measurement.event_buckets == {"e0": "unattributed_loss"}


def test_absorb_fixpoint_is_one_lineage_step_and_filters_both_a_b_a_events():
    units = [
        unit(0, "a", 0.0, 0.4),
        unit(1, "b", 0.4, 0.5),
        unit(2, "a", 0.5, 0.9),
    ]
    turns = [(0.0, 0.4, "A"), (0.4, 0.5, "B"), (0.5, 0.9, "A")]
    evidence = speaker_evidence(document(units, turns))
    measurement = measure_speaker_events(evidence, delivered_boundaries=(0.4, 0.5))
    assert measurement.event_buckets == {
        "e0": "policy_filtered",
        "e1": "policy_filtered",
    }


def test_160ms_edge_event_survives_lineage_and_can_be_expressed():
    units = [unit(0, "a", 0.0, 0.5), unit(1, "b", 0.5, 0.66)]
    turns = [(0.0, 0.5, "A"), (0.5, 0.66, "B")]
    evidence = speaker_evidence(document(units, turns))
    measurement = measure_speaker_events(evidence, delivered_boundaries=(0.5,))
    assert evidence.labels == ("A", "B")
    assert measurement.event_buckets == {"e0": "expressed"}


def test_coarse_parent_event_stays_unexpressible_after_refinement():
    parents = [unit(0, "ab", 0.0, 2.0)]
    turns = [(0.0, 1.0, "A"), (1.0, 2.0, "B")]
    refined = (
        unit(0, "a", 0.0, 1.0, provenance="subunit-per-char"),
        unit(1, "b", 1.0, 2.0, provenance="subunit-per-char"),
    )
    evidence = speaker_evidence(
        document(parents, turns), refined_units=refined, origin=(0, 0)
    )
    measurement = measure_speaker_events(evidence, delivered_boundaries=(1.0,))
    assert measurement.event_buckets == {"e0": "unexpressible"}
    assert measurement.buckets["unexpressible"] == 1


def test_speaker_attributable_count_uses_off_rows_own_global_match():
    units = [unit(0, "a", 0.0, 1.0), unit(1, "b", 1.0, 2.0)]
    evidence = speaker_evidence(document(units, [(0.0, 1.0, "A"), (1.0, 2.0, "B")]))
    attributable = measure_speaker_events(
        evidence, delivered_boundaries=(1.0,), off_boundaries=()
    )
    shared = measure_speaker_events(
        evidence, delivered_boundaries=(1.0,), off_boundaries=(1.4,)
    )
    assert attributable.speaker_attributable_expressed_cuts == 1
    assert shared.speaker_attributable_expressed_cuts == 0


def test_hostile_turns_fail_closed_inside_the_shadow_module():
    doc = document([unit(0, "a", 0.0, 1.0)], [(0.5, 0.5, "A")])
    with pytest.raises(SpeakerEvidenceError, match="positive"):
        speaker_evidence(doc)


def test_hostile_non_string_label_fails_closed():
    doc = document([unit(0, "a", 0.0, 1.0)], [(0.0, 1.0, 7)])  # type: ignore[list-item]
    with pytest.raises(SpeakerEvidenceError, match="label"):
        speaker_evidence(doc)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), True])
def test_hostile_speaker_weights_fail_closed(weight):
    evidence = speaker_evidence(document([unit(0, "a", 0.0, 1.0)], [(0.0, 1.0, "A")]))
    with pytest.raises(SpeakerEvidenceError, match="weight"):
        speaker_edge_cost(
            evidence,
            (0, 1),
            evidence_span=EvidenceSpan(0.0, 1.0, "exact", "exact"),
            weight=weight,
        )
