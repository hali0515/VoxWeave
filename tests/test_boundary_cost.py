# tests/test_boundary_cost.py
"""``experimental_policy_1``: features, weights, and the pause uncertainty integral.

The cost module is the only place P4 encodes an opinion, so its whole surface is
pinned: every raw feature is recorded even when its weight is zero, every
weighted term is quantized before summation, and no number is read from the
environment. ``POLICY_VERSION`` is what a later policy revision has to bump.

The pause term is the part that earned three review rounds. It is the MEAN of the
linear ramp over the +/-50 ms uncertainty interval, evaluated analytically, with a
ramp that depends on which evidence actually exists -- 220 ms when VAD confirmed
the silence, ``220 * offline_ms / clause_ms`` when there is no VAD at all, and a
flat 1.5 when a bound is missing entirely. Those three regimes and their knees
are pinned numerically; a point estimate at the gap value would be a different
model wearing the same name.
"""

from pathlib import Path

import pytest

from voxweave.core import boundary_cost as cost_mod
from voxweave.core.boundary_cost import (
    CUE_BASE,
    PAUSE_MISSING_BOUNDS_COST,
    POLICY_NAME,
    POLICY_VERSION,
    PUNCT_AFFINITY_CHARS,
    QUANTUM,
    RAMP_KNOWN_MS,
    SHORT_FRAGMENT_LOOSE,
    SHORT_FRAGMENT_TIGHT,
    UNCERTAINTY_MS,
    VAD_STATES,
    W_BALANCE,
    W_LINE_COUNT,
    W_MIGRATION,
    W_MIN_DURATION,
    W_PARTICLE,
    W_PAUSE,
    W_POS,
    W_PUNCT_AFFINITY,
    W_READING,
    W_SENTENCE_CROSS,
    W_SHOT_PREVIEW,
    CostBreakdown,
    cut_cost,
    edge_cost,
    make_breakdown,
    offline_ramp_ms,
    pause_cut_cost,
    pause_evidence,
    pause_knees,
    quantize,
    ramp_integral_mean,
    sum_breakdowns,
)
from voxweave.core.boundary_lattice import Edge, LatticeAtom
from voxweave.core.segdoc import DisplayProfile
from voxweave.core.timing_preview import LegacyCleanupPreview

PREVIEW = LegacyCleanupPreview()

# ---------------------------------------------------------------- fixtures


def profile(language="en", **over):
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


def atom(index, text, *, start=None, end=None, **over):
    base = dict(
        index=index,
        text=text,
        start=start,
        end=end,
        unit_start=index,
        unit_end=index + 1,
        end_pen=0,
        start_pen=0,
        boundary_pen=0,
        phrase_start=True,
        forced_boundary=False,
        display=text,
        members=(index,),
    )
    base.update(over)
    return LatticeAtom(**base)


PAIR = [atom(0, "ab", start=0.0, end=0.5), atom(1, "cd", start=0.5, end=1.0)]


def edge(display="ab cd", *, lines=1, line_widths=(5,), start=0.0, end=1.0):
    return Edge(
        start_node=0,
        end_node=2,
        text=display,
        display_text=display,
        lines=lines,
        line_widths=line_widths,
        span_start=start,
        span_end=end,
        waiver=None,
    )


def terms(breakdown):
    return dict(breakdown.weighted_terms)


# ---------------------------------------------------------------- policy surface


def test_policy_identity_is_explicit_and_versioned():
    assert POLICY_VERSION == 1
    assert POLICY_NAME == "experimental_policy_1"
    assert QUANTUM == 1e-6


def test_weights_are_the_adjudicated_ones():
    assert (W_PARTICLE, W_POS, W_PAUSE) == (3.0, 1.0, 3.0)
    assert (W_PUNCT_AFFINITY, W_SHOT_PREVIEW, W_MIGRATION) == (-1.0, -0.5, 0.0)
    assert (CUE_BASE, SHORT_FRAGMENT_TIGHT, SHORT_FRAGMENT_LOOSE) == (2.0, 8.0, 4.0)
    assert (W_LINE_COUNT, W_BALANCE) == (1.0, 0.05)
    assert (W_READING, W_MIN_DURATION, W_SENTENCE_CROSS) == (10.0, 6.0, 6.0)
    assert (RAMP_KNOWN_MS, UNCERTAINTY_MS, PAUSE_MISSING_BOUNDS_COST) == (
        220.0,
        50.0,
        1.5,
    )


def test_the_vad_state_enum_is_four_valued():
    assert set(VAD_STATES) == {"silence", "speech-overlap", "absent", "missing-bounds"}
    assert list(VAD_STATES) == sorted(VAD_STATES)


@pytest.mark.parametrize(
    "module", ["boundary_cost", "boundary_lattice", "boundary_v2", "partition_check"]
)
def test_no_module_reads_the_environment(module):
    """C7: all constants are module level; the shadow must be reproducible from
    its inputs alone."""
    source = Path(
        __import__(f"voxweave.core.{module}", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert "getenv" not in source


def test_quantization_happens_before_summation():
    breakdown = make_breakdown({"a": 1.0}, {"x": 0.1234567, "y": 0.7654321})
    assert breakdown.weighted_terms["x"] == quantize(0.1234567)
    assert breakdown.total == quantize(quantize(0.1234567) + quantize(0.7654321))
    assert quantize(0.1234567) == pytest.approx(0.123457, abs=0.0)


# ---------------------------------------------------------------- pause integral


def test_offline_ramp_is_the_scaled_ramp_not_offline_ms():
    """AD3-2: the 'offline_ms cliff' claim was retracted; offline_ms only scales."""
    assert offline_ramp_ms(profile()) == pytest.approx(385.0)
    ja = profile("ja", clause_ms=560.0, offline_ms=980.0)
    assert offline_ramp_ms(ja) == pytest.approx(385.0)


@pytest.mark.parametrize(
    "eff,expected",
    [
        (0.0, 3.0),  # wholly inside the linear region: mean == midpoint value
        (100.0, 3.0 * (1.0 - 100.0 / 220.0)),
        (170.0, 3.0 * (1.0 - 170.0 / 220.0)),  # b == ramp, the lower knee
        (200.0, 3.0 * (220.0 - 150.0) ** 2 / (2 * 220.0) / 100.0),  # straddling
        (270.0, 0.0),  # a == ramp, the upper knee
        (400.0, 0.0),
    ],
)
def test_ramp_integral_is_analytic_and_piecewise(eff, expected):
    assert ramp_integral_mean(eff, RAMP_KNOWN_MS) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("knee", [170.0, 270.0])
def test_the_ramp_integral_is_continuous_at_both_knees(knee):
    below = ramp_integral_mean(knee - 1e-7, RAMP_KNOWN_MS)
    above = ramp_integral_mean(knee + 1e-7, RAMP_KNOWN_MS)
    assert below == pytest.approx(above, abs=1e-8)


def test_the_ramp_integral_is_non_increasing():
    values = [ramp_integral_mean(x, RAMP_KNOWN_MS) for x in range(0, 400, 7)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_pause_knees_are_resolved_from_the_documents_own_thresholds():
    knees = pause_knees(profile())
    assert set(knees) == set(VAD_STATES)
    assert 170.0 in knees["silence"] and 270.0 in knees["silence"]
    assert 335.0 in knees["absent"] and 435.0 in knees["absent"]
    for values in knees.values():
        assert list(values) == sorted(values)


# ---------------------------------------------------------------- pause evidence


def test_missing_bounds_is_a_flat_documented_policy_delta():
    ev = pause_evidence(None, 1.4, speech_spans=None, profile=profile())
    assert ev.vad_state == "missing-bounds"
    assert ev.effective_ms is None and ev.ramp_ms is None
    assert pause_cut_cost(ev) == PAUSE_MISSING_BOUNDS_COST


def test_absent_vad_uses_the_offline_ramp():
    ev = pause_evidence(1.0, 1.4, speech_spans=None, profile=profile())
    assert ev.vad_state == "absent"
    assert ev.gap_ms_raw == pytest.approx(400.0)
    assert ev.overlap_fraction == 0.0
    assert ev.effective_ms == pytest.approx(400.0)
    assert ev.ramp_ms == pytest.approx(385.0)
    assert pause_cut_cost(ev) == pytest.approx(ramp_integral_mean(400.0, 385.0))


def test_an_empty_speech_span_list_is_confirmed_silence_not_absence():
    """The three-way None / [] / populated semantics of ``gap_qualifies`` survive."""
    ev = pause_evidence(1.0, 1.4, speech_spans=[], profile=profile())
    assert ev.vad_state == "silence"
    assert ev.ramp_ms == pytest.approx(RAMP_KNOWN_MS)
    assert ev.effective_ms == pytest.approx(400.0)


def test_speech_overlap_discounts_the_gap_by_its_overlapped_fraction():
    ev = pause_evidence(1.0, 1.4, speech_spans=[(1.1, 1.3)], profile=profile())
    assert ev.vad_state == "speech-overlap"
    assert ev.overlap_fraction == pytest.approx(0.5)
    assert ev.effective_ms == pytest.approx(200.0)
    assert pause_cut_cost(ev) == pytest.approx(ramp_integral_mean(200.0, 220.0))


def test_a_non_positive_gap_is_zero_effective_silence():
    ev = pause_evidence(1.4, 1.0, speech_spans=[], profile=profile())
    assert ev.gap_ms_raw == 0.0
    assert pause_cut_cost(ev) == pytest.approx(W_PAUSE)


def test_pause_evidence_features_are_all_recorded():
    ev = pause_evidence(1.0, 1.4, speech_spans=[(1.1, 1.3)], profile=profile())
    features = ev.to_features()
    assert set(features) >= {
        "gap_ms_raw",
        "vad_state",
        "overlap_fraction",
        "uncertainty_ms",
        "effective_ms",
        "ramp_ms",
    }
    assert features["uncertainty_ms"] == UNCERTAINTY_MS


# ---------------------------------------------------------------- cut cost


def base_cut(**kwargs):
    left = atom(0, "cat.", start=0.5, end=1.0, end_pen=2)
    right = atom(1, "sat", start=1.4, end=2.0, start_pen=1, boundary_pen=3)
    call = dict(
        unit_id=1,
        profile=profile(),
        speech_spans=None,
        shot_changes=None,
        v1_cut_units=frozenset(),
    )
    call.update(kwargs)
    return cut_cost(left, right, **call)


def test_cut_cost_records_every_raw_feature_including_zero_weighted_ones():
    breakdown = base_cut()
    assert isinstance(breakdown, CostBreakdown)
    assert set(breakdown.features) >= {
        "particle_raw",
        "pos_raw",
        "gap_ms_raw",
        "vad_state",
        "overlap_fraction",
        "uncertainty_ms",
        "effective_ms",
        "ramp_ms",
        "punct_affinity_raw",
        "shot_preview_raw",
        "migration_raw",
    }
    assert breakdown.features["migration_raw"] == 1.0
    assert breakdown.weighted_terms["migration"] == 0.0


def test_cut_cost_weights_particle_and_pos_damage():
    breakdown = base_cut()
    assert breakdown.features["particle_raw"] == 3
    assert breakdown.features["pos_raw"] == 3
    assert terms(breakdown)["particle"] == pytest.approx(9.0)
    assert terms(breakdown)["pos"] == pytest.approx(3.0)


def test_punctuation_affinity_is_a_reward_on_the_left_surface():
    assert "." in PUNCT_AFFINITY_CHARS and "。" in PUNCT_AFFINITY_CHARS
    with_punct = base_cut()
    assert with_punct.features["punct_affinity_raw"] == 1.0
    assert terms(with_punct)["punct_affinity"] == pytest.approx(-1.0)

    plain = cut_cost(
        atom(0, "cat", start=0.5, end=1.0),
        atom(1, "sat", start=1.4, end=2.0),
        unit_id=1,
        profile=profile(),
        speech_spans=None,
        shot_changes=None,
    )
    assert plain.features["punct_affinity_raw"] == 0.0
    assert terms(plain)["punct_affinity"] == 0.0


def test_shot_preview_is_a_mild_affinity_inside_the_snap_window():
    breakdown = base_cut(shot_changes=[1.5])
    assert breakdown.features["shot_preview_raw"] == pytest.approx(0.1)
    assert terms(breakdown)["shot_preview"] == pytest.approx(
        W_SHOT_PREVIEW * (1.0 - 0.1 / 0.458)
    )


def test_shot_preview_ignores_a_cut_outside_the_snap_window():
    breakdown = base_cut(shot_changes=[9.0])
    assert breakdown.features["shot_preview_raw"] is None
    assert terms(breakdown)["shot_preview"] == 0.0


def test_a_disabled_shot_snap_removes_the_term_entirely():
    """AD-7: ``shot_snap_s <= 0`` -> feature None, term 0."""
    breakdown = base_cut(profile=profile(shot_snap_s=0.0), shot_changes=[1.5])
    assert breakdown.features["shot_preview_raw"] is None
    assert terms(breakdown)["shot_preview"] == 0.0


def test_migration_is_recorded_against_the_v1_reference():
    breakdown = base_cut(v1_cut_units=frozenset({1}))
    assert breakdown.features["migration_raw"] == 0.0
    assert terms(breakdown)["migration"] == 0.0


def test_cut_total_is_the_quantized_sum_of_its_terms():
    breakdown = base_cut()
    assert breakdown.total == quantize(sum(breakdown.weighted_terms.values()))


# ---------------------------------------------------------------- edge cost


def edge_terms(display="ab cd", *, prof=None, next_start=None, sentences=0, **over):
    breakdown = edge_cost(
        edge(display, **over),
        PAIR,
        profile=prof or profile(),
        preview=PREVIEW,
        next_start=next_start,
        sentence_cross_count=sentences,
    )
    return breakdown, terms(breakdown)


def test_a_comfortable_edge_costs_only_its_cue_base():
    breakdown, weighted = edge_terms()
    assert weighted["cue_base"] == pytest.approx(CUE_BASE)
    assert weighted["short_fragment"] == 0.0
    assert weighted["line_count"] == 0.0
    assert weighted["balance"] == 0.0
    assert weighted["reading"] == 0.0
    assert weighted["min_duration"] == 0.0
    assert weighted["sentence_cross"] == 0.0
    assert breakdown.total == pytest.approx(CUE_BASE)
    assert breakdown.features["available_s"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    "display,expected",
    [("ab", SHORT_FRAGMENT_TIGHT), ("abcd", SHORT_FRAGMENT_LOOSE), ("ab cd", 0.0)],
)
def test_short_fragment_tiers_use_the_true_visual_width(display, expected):
    _, weighted = edge_terms(display)
    assert weighted["short_fragment"] == pytest.approx(expected)


def test_line_count_and_balance_come_from_the_packer_state():
    breakdown, weighted = edge_terms("ab cd", lines=2, line_widths=(9, 4))
    assert breakdown.features["line_count_raw"] == 2
    assert weighted["line_count"] == pytest.approx(W_LINE_COUNT)
    assert breakdown.features["balance_raw"] == pytest.approx(5.0)
    assert weighted["balance"] == pytest.approx(5.0 * W_BALANCE)


def test_reading_pressure_fires_only_when_there_is_a_reading_load():
    # cps=1.0 -> need 4.0 s; the preview grants the capped linger, so avail is 2.0
    breakdown, slow = edge_terms(prof=profile(cps=1.0))
    assert breakdown.features["reading_need_s"] == pytest.approx(4.0)
    assert breakdown.features["available_s"] == pytest.approx(2.0)
    assert slow["reading"] == pytest.approx(W_READING * (4.0 - 2.0) / 4.0)

    breakdown, off = edge_terms(prof=profile(cps=0.0))
    assert breakdown.features["reading_need_s"] == 0.0
    assert off["reading"] == 0.0  # no division by a zero need


def test_min_duration_fires_when_the_preview_cannot_grant_the_floor():
    breakdown, weighted = edge_terms(prof=profile(min_cue_s=3.0), next_start=1.05)
    assert breakdown.features["available_s"] == pytest.approx(1.0)
    assert weighted["min_duration"] == pytest.approx(W_MIN_DURATION * (3.0 - 1.0) / 3.0)


def test_a_disabled_min_duration_removes_the_term():
    _, weighted = edge_terms(prof=profile(min_cue_s=0.0))
    assert weighted["min_duration"] == 0.0


def test_sentence_crossing_is_priced_per_crossed_end():
    _, weighted = edge_terms(sentences=2)
    assert weighted["sentence_cross"] == pytest.approx(2 * W_SENTENCE_CROSS)


def test_an_unresolvable_span_degrades_rather_than_raising():
    """Reachable only inside the AD4-1 all-invisible branch, whose chain is forced."""
    breakdown = edge_cost(
        edge(start=None, end=None),
        PAIR,
        profile=profile(),
        preview=PREVIEW,
        next_start=None,
        sentence_cross_count=0,
    )
    assert breakdown.features["available_s"] is None
    assert breakdown.total == pytest.approx(
        quantize(sum(breakdown.weighted_terms.values()))
    )


def test_the_sentence_algebra_prefers_splitting_a_clean_sentence_boundary():
    """C7's stated algebra: at a clean zero-gap sentence end, crossing must cost
    more than the two cues plus the cut that splitting there would buy."""
    crossing, _ = edge_terms(sentences=1)
    half, _ = edge_terms()
    clean_cut = cut_cost(
        atom(0, "ab", start=0.0, end=0.5),
        atom(1, "cd", start=0.5, end=1.0),
        unit_id=1,
        profile=profile(),
        speech_spans=[],
        shot_changes=None,
    )
    assert crossing.total > 2 * half.total + clean_cut.total


# ---------------------------------------------------------------- aggregation


def test_sum_breakdowns_pools_numeric_features_and_drops_categoricals():
    a = make_breakdown(
        {"gap_ms_raw": 100.0, "vad_state": "silence"}, {"pause_cut": 1.0}
    )
    b = make_breakdown({"gap_ms_raw": 50.0, "vad_state": "absent"}, {"pause_cut": 2.0})
    pooled = sum_breakdowns([a, b])
    assert pooled.features["gap_ms_raw"] == pytest.approx(150.0)
    assert "vad_state" not in pooled.features
    assert pooled.weighted_terms["pause_cut"] == pytest.approx(3.0)
    assert pooled.total == quantize(3.0)


def test_sum_breakdowns_of_nothing_is_zero():
    pooled = sum_breakdowns([])
    assert pooled.total == 0.0
    assert dict(pooled.weighted_terms) == {}


def test_breakdown_to_dict_is_sorted_and_json_ready():
    breakdown = base_cut()
    blob = breakdown.to_dict()
    assert list(blob["weighted_terms"]) == sorted(blob["weighted_terms"])
    assert list(blob["features"]) == sorted(blob["features"])
    assert blob["total"] == breakdown.total


def test_the_cost_module_does_not_depend_on_the_solver():
    """Import direction is one-way: cost may read the lattice, never boundary_v2."""
    source = Path(cost_mod.__file__).read_text(encoding="utf-8")
    assert "boundary_v2" not in source
