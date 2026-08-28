from __future__ import annotations

from types import SimpleNamespace


def _profile():
    return SimpleNamespace(
        language="en",
        min_cue_s=0.0,
        max_cue_s=0.0,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=0.0,
    )


def _semantic_facts(*, bad_text: bool = False, bad_leg: bool = False):
    relative_start = 0.2
    relative_end = 0.8
    physical_origin = 1601 / 16_000
    legacy_origin = 0.10001
    start = relative_start + physical_origin
    end = relative_end + physical_origin
    word = SimpleNamespace(
        unit_id="r0",
        call_index=0,
        call_unit_index=0,
        text="word",
        relative_start=relative_start,
        relative_end=relative_end,
        physical_origin_seconds=physical_origin,
        start=start,
        end=end,
        provenance="aligner",
        original_relative_start=relative_start,
        original_relative_end=relative_end,
    )
    physical_call = SimpleNamespace(
        call_index=0,
        sample_start=1601,
        sample_end=16_000,
        sample_rate=16_000,
        physical_origin_seconds=physical_origin,
        legacy_origin_seconds=legacy_origin,
        authority_origin_seconds=physical_origin,
        legacy_origin_kind="nominal-route",
    )
    authority_block = SimpleNamespace(
        source_index=0,
        authority_unit_ids=("r0",),
        word_data=(word,),
        speech_start=start,
        speech_end=end,
    )
    legacy_cue = SimpleNamespace(
        source_index=0,
        text="legacy",
        start=relative_start + legacy_origin,
        end=relative_end + legacy_origin,
        lyric=None,
    )
    v2_cue = SimpleNamespace(
        source_index=0,
        text="tampered" if bad_text else "word",
        start=start,
        end=end + (1.0 if bad_leg else 0.0),
        lyric=True,
    )
    phase1 = SimpleNamespace(
        index=0,
        start=start,
        end=end,
        seed_start=start,
        seed_end=end,
        speech_start=start,
        speech_end=end,
        text="word",
        lines=("word",),
        cell_widths=(4,),
        reading_chars=4,
        raw_reading_chars=4,
        word_data=[{"text": "word", "start": start, "end": end}],
        unit_range=(0, 1),
        lyric=True,
        reports=(),
    )
    legs = ()
    if bad_leg:
        legs = (
            SimpleNamespace(
                rule_id="duration-desire",
                sweep=1,
                cue_index=0,
                slot=1,
                target=SimpleNamespace(cue_index=0, side="end"),
                from_value=end,
                to_value=end + 1.0,
                reads=(
                    SimpleNamespace(
                        boundary=SimpleNamespace(cue_index=0, side="start"),
                        value=start,
                    ),
                ),
            ),
        )
    observation = SimpleNamespace(
        semantic_root_lineage=(
            (
                "evaluation",
                "align/delivery-finalizer/v2",
                "call",
                "fresh-seed",
                "phase1",
                None,
            ),
        ),
        phase1_seed=(phase1,),
        delivered=((start, v2_cue.end),),
        report=SimpleNamespace(terminal="fixed-point"),
        trace=SimpleNamespace(
            legs=legs,
            terminal="fixed-point",
            cycle=None,
            sweeps=1,
        ),
        partition_result=SimpleNamespace(exit_driving=False),
        trace_problems=(),
        stability_problems=(),
    )
    return {
        "route_kind": "qwen-crop",
        "physical_calls": (physical_call,),
        "authority_blocks": (authority_block,),
        "legacy": SimpleNamespace(cues=(legacy_cue,)),
        "v2": SimpleNamespace(cues=(v2_cue,)),
        "semantic_observation": observation,
        "profile": _profile(),
        "evidence": SimpleNamespace(sing_spans=((start, end),), shots=()),
    }


def test_semantic_oracle_accepts_only_its_independently_derived_relations():
    from voxweave.core.align_compare import compare_semantic_deltas

    comparison = compare_semantic_deltas(**_semantic_facts())
    outcomes = {outcome.delta_id: outcome for outcome in comparison.outcomes}

    assert comparison.active_classes == ("ALD-0", "ALD-1", "ALD-2", "ALD-5")
    assert comparison.violations == ()
    assert set(comparison.primitive_field_diffs) >= {
        "authority-time",
        "text",
        "end",
        "lyric",
    }
    assert all(outcome.passed for outcome in outcomes.values())


def test_semantic_oracle_rejects_wrong_v2_text_even_when_its_shape_is_valid():
    from voxweave.core.align_compare import compare_semantic_deltas

    comparison = compare_semantic_deltas(**_semantic_facts(bad_text=True))
    outcome = next(
        outcome for outcome in comparison.outcomes if outcome.delta_id == "ALD-1"
    )

    assert outcome.triggered is True
    assert outcome.passed is False
    assert "ALD-1" in comparison.violations


def test_semantic_oracle_replays_leg_arithmetic_instead_of_trusting_delivery():
    from voxweave.core.align_compare import compare_semantic_deltas

    comparison = compare_semantic_deltas(**_semantic_facts(bad_leg=True))
    outcome = next(
        outcome for outcome in comparison.outcomes if outcome.delta_id == "ALD-4"
    )

    assert outcome.triggered is True
    assert outcome.passed is False
    assert "ALD-4" in comparison.violations
