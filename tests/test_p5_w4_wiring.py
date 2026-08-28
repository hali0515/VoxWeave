"""End-to-end pins for the final P5 shadow wiring wave."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

from tests.test_calib_shadow import calib
from tests.test_shadow_hook import _case_plain, _case_speakers, _segment
from voxweave import pipeline


def test_live_artifact_is_complete_schema_two(monkeypatch) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None

    assert artifact["schema_version"] == 2
    assert {
        "coarse_caused_intervals",
        "dual_form_unmeasured",
        "named_multi_cues_unannotated",
    } <= set(artifact["coverage"])
    assert artifact["finalizer"] is not None
    assert artifact["speaker_evidence"]["measurement"] is not None
    assert artifact["speaker_evidence"]["projection"]["status"] == "verified"
    assert artifact["subunit_split"] is not None
    assert artifact["delta_registry"]
    assert (
        artifact["preview_fidelity"]["checked_edges"]
        == artifact["preview_fidelity"]["scored_edges"]
    )
    assert artifact["preview_fidelity"]["mismatches"] == []
    assert all(
        row["edge_count"] == row["cue_count"] and not row["mismatches"]
        for row in artifact["preview_fidelity"]["selected_rows"].values()
    )


def test_live_admission_refuses_a_premature_optimizer_schema_two(monkeypatch) -> None:
    from voxweave.core import boundary_v2

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    monkeypatch.setattr(boundary_v2, "SCHEMA_VERSION", 2)
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None
    assert artifact["schema_version"] == 1
    assert artifact["kind"] == "segmentation-shadow-error"
    assert "requires a schema-1 optimizer payload" in artifact["error"]["detail"]


def test_live_post_assembly_validator_rejects_a_deleted_required_block(
    monkeypatch,
) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    real_assembler = pipeline._shadow_v2_artifact

    def omit_authorities(*args, **kwargs):
        artifact = real_assembler(*args, **kwargs)
        artifact.pop("authorities")
        return artifact

    monkeypatch.setattr(pipeline, "_shadow_v2_artifact", omit_authorities)
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None
    assert artifact["schema_version"] == 1
    assert artifact["kind"] == "segmentation-shadow-error"
    assert "artifact: missing keys authorities" in artifact["error"]["detail"]


def test_required_schema_two_blocks_cannot_be_deleted(monkeypatch) -> None:
    from voxweave.core.shadow_schema import validate_shadow_v2_payload

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    original = _segment(_case_plain()).shadow
    assert original is not None
    for key in (
        "subunit_split",
        "delta_registry",
        "margin_summary",
        "diff_classification",
        "authorities",
        "refiner_comparison",
    ):
        artifact = dict(original)
        artifact.pop(key)
        errors = validate_shadow_v2_payload(artifact)
        assert any(f"missing keys {key}" in error for error in errors), key
        harness_errors = calib.shadow_measurement_errors(
            cast(Any, SimpleNamespace(id="schema-mutation")), artifact, {}
        )
        assert any("schema-2 structural error" in error for error in harness_errors)


def test_schema_two_nested_evidence_shapes_are_closed(monkeypatch) -> None:
    from voxweave.core.shadow_schema import validate_shadow_v2_payload

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    original = _segment(_case_speakers()).shadow
    assert original is not None

    mutations: list[tuple[dict[str, Any], str]] = []
    dual_form = cast(dict[str, Any], copy.deepcopy(original))
    dual_form["coverage"]["dual_form_unmeasured"] = 0
    mutations.append((dual_form, "dual_form_unmeasured: expected boolean"))

    legacy_margin = cast(dict[str, Any], copy.deepcopy(original))
    legacy_margin["intervals"][0]["low_margin"] = True
    mutations.append((legacy_margin, "legacy margin keys remain"))

    invalid_margin = cast(dict[str, Any], copy.deepcopy(original))
    invalid_margin["margin_summary"]["min"] = (
        float(invalid_margin["margin_summary"]["p50"]) + 1.0
    )
    mutations.append((invalid_margin, "expected min <= p05 <= p50"))

    invalid_diff = cast(dict[str, Any], copy.deepcopy(original))
    invalid_diff["diff_classification"]["relation_failures"] = "zero"
    mutations.append((invalid_diff, "relation_failures: expected non-negative integer"))

    malformed_trigger = cast(dict[str, Any], copy.deepcopy(original))
    malformed_trigger["diff_classification"]["independent_fired"] = [{}]
    mutations.append((malformed_trigger, "independent_fired: expected strings"))

    invalid_root = cast(dict[str, Any], copy.deepcopy(original))
    invalid_root["authorities"]["events"][0].pop("input_kind")
    mutations.append((invalid_root, "missing keys input_kind"))

    invalid_policy = cast(dict[str, Any], copy.deepcopy(original))
    invalid_policy["policy_deltas"] = []
    mutations.append((invalid_policy, "differs from the live policy-2 declaration"))

    for artifact, expected in mutations:
        assert any(
            expected in error for error in validate_shadow_v2_payload(artifact)
        ), expected


def test_report_alias_and_fallback_rechecks_are_closed_schema_invariants(
    monkeypatch,
) -> None:
    from voxweave.core.shadow_schema import validate_shadow_v2_payload

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    original = _segment(_case_plain()).shadow
    assert original is not None

    alias = cast(dict[str, Any], copy.deepcopy(original))
    finalizer = alias["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v1"][
        "finalizer"
    ]
    finalizer["entries"] = [
        {"cue_index": 0, "evidence": {"side": "start"}, "kind": "fabricated-time"}
    ]
    assert any(
        "entries/refusals report channels differ" in error
        for error in validate_shadow_v2_payload(alias)
    )

    orphan = cast(dict[str, Any], copy.deepcopy(original))
    orphan["canonical_fallback_rechecks"].append(
        {
            "cue_index": 0,
            "reason": "granularity-unreconciled",
            "row": "v1",
            "with_owned_footprint": "word-data",
            "with_owned_footprint_reason": None,
        }
    )
    assert any(
        "entries and independent rechecks do not match" in error
        for error in validate_shadow_v2_payload(orphan)
    )


def test_live_lane_and_row_matrix_uses_one_typed_root_per_materialized_row(
    monkeypatch,
) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(_case_speakers()).shadow
    assert artifact is not None

    assert set(artifact["lanes"]) == {
        pipeline.SHADOW_LANE_CORE,
        pipeline.SHADOW_LANE_DELIVERY_LEGACY,
        pipeline.SHADOW_LANE_FINALIZER,
        pipeline.SHADOW_LANE_LEGACY_DISPLAY,
    }
    assert set(artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]) == {
        "v1",
        "v2",
        "v2-speaker-off",
    }
    assert set(artifact["lanes"][pipeline.SHADOW_LANE_LEGACY_DISPLAY]["rows"]) == {"v1"}

    authorities = artifact["authorities"]
    assert authorities["violations"] == []
    assert authorities["expected"] == {
        "delivery_finalizer/v1": "v1-capture",
        "delivery_finalizer/v2": "optimizer-selection",
        "delivery_finalizer/v2-speaker-off": "optimizer-selection",
    }
    assert len(authorities["lineage"]) == 3


def test_every_finalizer_row_is_trace_verified_and_stage_checked(monkeypatch) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(_case_speakers()).shadow
    assert artifact is not None

    rows = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]
    for row_id, row in rows.items():
        assert row["finalizer"]["valid"] is True, row_id
        assert row["finalizer"]["trace_errors"] == [], row_id
        assert row["finalizer"]["stability_errors"] == [], row_id
        assert row["validator"]["stage"] == "finalizer", row_id

    assert artifact["validator"]["finalizer"] == rows["v2"]["validator"]


def test_n7_audits_every_scored_edge_against_phase_one(monkeypatch) -> None:
    """Mutation pin: changing preview facts cannot pass the corpus audit."""
    from voxweave.core.finalizer import FinalizerPreview

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    real = FinalizerPreview.preview_cue

    def corrupt_reading_load(self, candidate):
        preview = real(self, candidate)
        return replace(preview, reading_chars=preview.reading_chars + 1)

    monkeypatch.setattr(FinalizerPreview, "preview_cue", corrupt_reading_load)
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None

    assert artifact["schema_version"] == 1
    assert artifact["kind"] == "segmentation-shadow-error"
    assert "preview_fidelity.mismatches" in artifact["error"]["detail"]


def test_speaker_measurement_uses_the_counterfactual_and_v2_ids(monkeypatch) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(_case_speakers()).shadow
    assert artifact is not None

    evidence = artifact["speaker_evidence"]
    assert evidence["measurement"] is not None
    assert evidence["off_row_measurement"] is not None
    assert (
        sum(evidence["measurement"]["buckets"].values())
        == evidence["measurement"]["raw_in_speech_turn_changes"]
    )
    assert (
        sum(evidence["off_row_measurement"]["buckets"].values())
        == evidence["off_row_measurement"]["raw_in_speech_turn_changes"]
    )

    v2 = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v2"]
    off = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v2-speaker-off"]
    assert any(row["speaker_ids"] for row in v2["cues"])
    assert not any(row["speaker_ids"] for row in off["cues"])


def test_live_speaker_snapshot_precedes_refinement_and_projects_by_origin(
    monkeypatch,
) -> None:
    from voxweave.core import speaker_evidence as module

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    events: list[tuple[str, int, tuple[int, ...] | None]] = []
    real_snapshot = module.speaker_evidence
    real_project = module.project_speaker_evidence

    def snapshot(document, *args, **kwargs):
        events.append(("snapshot", len(document.units), None))
        return real_snapshot(document, *args, **kwargs)

    def project(evidence, *, refined_units, origin):
        events.append(("project", len(refined_units), tuple(origin)))
        return real_project(evidence, refined_units=refined_units, origin=origin)

    monkeypatch.setattr(module, "speaker_evidence", snapshot)
    monkeypatch.setattr(module, "project_speaker_evidence", project)
    artifact = _segment(
        {
            "language": "en",
            "word_segments": [
                {"text": "alpha bravo charlie delta", "start": 0.0, "end": 2.0}
            ],
            "speaker_turns": [[0.0, 2.0, "A"]],
        },
        smart_split_kwargs={"max_line_length": 5, "max_lines": 1},
    ).shadow
    assert artifact is not None

    assert [event[0] for event in events] == ["snapshot", "project"]
    assert events[0] == ("snapshot", 1, None)
    assert events[1][0] == "project"
    assert events[1][1] > 1
    assert set(events[1][2] or ()) == {0}
    assert artifact["kind"] == "segmentation-shadow-incomplete"
    assert artifact["schema_version"] == 1
    assert artifact["diagnostic"]["speaker_evidence"]["attribution"] == (
        "parent-projected"
    )


def test_margin_summary_replaces_interval_runner_up_fields(monkeypatch) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None

    assert "margin_summary" in artifact
    for interval in artifact["intervals"]:
        assert "margin" not in interval
        assert "low_margin" not in interval
        assert "runner_up_total" not in interval
        assert "margin_summary" in interval


def test_tracked_identity_row_runs_the_refiner_bypass_counterfactual(
    monkeypatch,
) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None
    assert artifact["refiner_comparison"] == {
        "byte_identical": True,
        "refined_parent_count": 0,
        "status": "tracked-identity",
    }


def test_v1_coordinates_reconcile_on_parents_then_translate_through_origin() -> None:
    from voxweave.core.segdoc import SourceUnit

    parent = SimpleNamespace(
        units=[
            SourceUnit("u0", "alpha", 0.0, 0.3),
            SourceUnit("u1", "bravo", 0.4, 0.7),
            SourceUnit("u2", "charlie", 0.8, 1.1),
        ]
    )
    cues = [
        {
            "word_data": [
                {"text": "alpha", "start": 0.0, "end": 0.3},
                {"text": "bravo", "start": 0.4, "end": 0.7},
            ]
        },
        {"word_data": [{"text": "charlie", "start": 0.8, "end": 1.1}]},
    ]
    # Parents 0 and 2 were refined. Parent cut 2 therefore becomes child cut 3.
    partition, mode = pipeline._shadow_v1_partition(
        cast(Any, parent), (0, 0, 1, 2, 2), cast(Any, cues)
    )
    assert partition == (3,)
    assert mode == "surface-footprint-parent-through-origin"


def test_n11_detects_a_missing_producer_trigger(monkeypatch) -> None:
    """Mutation pin: producer and independent FD registries cannot collude."""
    from voxweave.core import finalizer

    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    monkeypatch.setattr(finalizer, "_deltas_fired", lambda *args, **kwargs: ())
    artifact = _segment(_case_speakers()).shadow
    assert artifact is not None

    classification = artifact["diff_classification"]
    assert classification["trigger_mismatches"] == ["FD-4"]
    assert classification["independent_fired"] == ["FD-4"]
    assert classification["producer_fired"] == []


def test_n11_recomputes_fd7_from_phase_one_not_the_serialized_alias() -> None:
    from voxweave.core.finalizer import phase1_cue
    from voxweave.core.segdoc import DisplayProfile

    profile = DisplayProfile(
        language="en",
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
    seed = {
        "text": "fallback",
        "start": 0.0,
        "end": 0.4,
        "word_data": [{"text": "word", "start": 0.0, "end": 0.4}],
        "speech_start": 0.0,
        "speech_end": 0.4,
    }
    phase1 = phase1_cue(
        seed,
        profile=profile,
        index=0,
        expected_footprint="different",
        unit_range=(0, 1),
    )
    assert phase1.reports
    cue = {
        "end": phase1.end,
        "index": 0,
        "lines": len(phase1.lines),
        "lyric": False,
        "speaker_ids": [],
        "speech_end": phase1.speech_end,
        "speech_start": phase1.speech_start,
        "start": phase1.start,
        "text": phase1.text,
        "unit_range": [0, 1],
    }
    row = {
        "cues": [cue],
        "finalizer": {
            "deltas_fired": [],
            "entries": [],
            "refusals": [],
            "stability_errors": [],
            "trace": {"cycle": None, "legs": [], "terminal": "fixed-point"},
            "trace_errors": [],
        },
    }
    classification = pipeline._shadow_diff_classification(
        row,
        {"cues": [dict(cue)]},
        stream=SimpleNamespace(cues=(phase1,), profile=profile),
        seed_cues=[seed],
    )
    assert classification["independent_fired"] == ["FD-7"]
    assert classification["producer_fired"] == []
    assert classification["trigger_mismatches"] == ["FD-7"]


def test_n11_cross_checks_the_upstream_fd2_producer_fact(monkeypatch) -> None:
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    artifact = _segment(
        {
            "language": "en",
            "word_segments": [{"text": "word", "start": 0.0, "end": 0.4}],
            # Evidence span: 0.3 / 0.4 sung. Legacy display span after its
            # timing polish: 0.3 / 0.65 sung. The predicates must disagree.
            "sing_spans": [[0.0, 0.3]],
        }
    ).shadow
    assert artifact is not None

    row = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["v1"]
    assert row["finalizer"]["deltas_fired"] == ["FD-2"]
    assert artifact["diff_classification"] == {
        "alignment_error": False,
        "changed_fields": [
            {
                "allowed_relation": True,
                "field": "lyric",
                "from": False,
                "trigger_ids": ["FD-2"],
                "to": True,
                "unit_range": [0, 1],
            }
        ],
        "independent_fired": ["FD-2"],
        "producer_fired": ["FD-2"],
        "relation_failures": 0,
        "trigger_mismatches": [],
        "unclassified_field_diff": 0,
    }


def test_n11_detects_a_changed_value_outside_the_allowed_relation(monkeypatch) -> None:
    """Mutation pin: trigger eligibility alone cannot approve a corrupted value."""
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    real = pipeline._shadow_finalizer_row

    def corrupt_serialized_v1(*args, **kwargs):
        row, cues = real(*args, **kwargs)
        if kwargs["origin"] == "v1":
            row["cues"][0]["end"] = float(row["cues"][0]["end"]) + 0.125
        return row, cues

    monkeypatch.setattr(pipeline, "_shadow_finalizer_row", corrupt_serialized_v1)
    artifact = _segment(_case_speakers()).shadow
    assert artifact is not None

    classification = artifact["diff_classification"]
    assert classification["unclassified_field_diff"] == 0
    assert classification["relation_failures"] == 1
    changed = classification["changed_fields"]
    assert [
        (row["field"], row["trigger_ids"], row["allowed_relation"]) for row in changed
    ] == [("end", ["FD-4"], False)]


def test_perturbation_classes_pin_unit_shot_and_all_speaker_cliffs() -> None:
    original = [
        {"id": "u0", "text": "a", "start": 0.0, "end": 0.4},
        {"id": "u1", "text": "b", "start": 0.5, "end": 0.9},
    ]
    timing_only = [dict(row) for row in original]
    timing_only[0]["end"] = 0.401
    changed_unit = [dict(row) for row in timing_only]
    changed_unit[0]["text"] = "x"
    assert calib.perturbation_unit_stable(original, timing_only) is True
    assert calib.perturbation_unit_stable(original, changed_unit) is False

    shot = calib.shot_cycle_probe()
    assert shot["failures"] == []
    assert (shot["before_terminal"], shot["after_terminal"]) == (
        "fixed-point",
        "cycle-adoption",
    )
    assert shot["influence_cell"] == {
        "outside": [],
        "radius_units": 0,
        "unit_count": 1,
    }

    speaker = calib.speaker_cliff_diagnostics()
    assert speaker["failures"] == []
    assert {row["name"] for row in speaker["probes"]} == {
        "cover-frac",
        "MIN_RUN",
        "EDGE_RUN",
        "phrase-vote",
        "region-silence",
        "transition-crossing",
    }
    assert all(row["effective"] for row in speaker["probes"])
    assert set(speaker["attempted_by_turn_state"]) == {
        "absent",
        "overlap",
        "multi",
        "single",
        "unattributed",
    }
    assert speaker["warnings"] == [
        "P3/absent: zero effective probes (warning-uncovered)",
        "P3/overlap: zero effective probes (warning-uncovered)",
    ]


def _speaker_gate_case(
    *, raw: int, expressed: int, missed: int, off_expressed: int, attributable: int
):
    on_rest = raw - expressed - missed
    off_rest = raw - off_expressed
    return SimpleNamespace(
        case=SimpleNamespace(language="ja"),
        artifact={
            "speaker_evidence": {
                "measurement": {
                    "buckets": {
                        "expressed": expressed,
                        "policy_filtered": on_rest,
                        "survived_expressible_but_missed": missed,
                        "unattributed_loss": 0,
                        "unexpressible": 0,
                    },
                    "raw_in_speech_turn_changes": raw,
                    "speaker_attributable_expressed_cuts": attributable,
                },
                "off_row_measurement": {
                    "buckets": {
                        "expressed": off_expressed,
                        "policy_filtered": off_rest,
                        "survived_expressible_but_missed": 0,
                        "unattributed_loss": 0,
                        "unexpressible": 0,
                    },
                    "raw_in_speech_turn_changes": raw,
                    "speaker_attributable_expressed_cuts": 0,
                },
            }
        },
    )


def test_n3b_exact_four_gate_golden_and_unreachable_target_stop() -> None:
    passing = calib.speaker_gate_block(
        [
            _speaker_gate_case(
                raw=136,
                expressed=21,
                missed=21,
                off_expressed=20,
                attributable=1,
            )
        ]
    )
    assert [gate["id"] for gate in passing["gates"]] == [
        "N3b-activation",
        "N3b-expressed-rate",
        "N3b-attributable",
        "N3b-expressible-hit",
    ]
    assert [gate["status"] for gate in passing["gates"]] == ["pass"] * 4

    impossible = calib.speaker_gate_block(
        [
            _speaker_gate_case(
                raw=131,
                expressed=10,
                missed=5,
                off_expressed=8,
                attributable=2,
            )
        ]
    )
    rate = impossible["gates"][1]
    assert rate == {
        "absolute_status": "stopped",
        "comparison_status": "pass",
        "id": "N3b-expressed-rate",
        "possible_rate": 15 / 131,
        "status": "stopped",
        "target": 21 / 136,
        "value": 10 / 131,
    }

    regressed = calib.speaker_gate_block(
        [
            _speaker_gate_case(
                raw=131,
                expressed=4,
                missed=1,
                off_expressed=5,
                attributable=1,
            )
        ]
    )["gates"][1]
    assert regressed["absolute_status"] == "stopped"
    assert regressed["comparison_status"] == "fail"
    assert regressed["status"] == "fail"


def test_coarse_derivation_registry_drives_every_n4c_case_gate() -> None:
    tracked = calib.load_corpus(calib.DEFAULT_CORPUS)
    block = calib.run_coarse_gates(tracked)

    assert block["adjudication"] == "accepted deterministic derivation registry"
    assert block["failures"] == []
    assert [row["variant"] for row in block["cases"]] == [
        "width",
        "duration",
        "both",
        "per-char",
        "mixed",
    ]
    assert all(all(row["checks"].values()) for row in block["cases"])
    assert block["trigger_classes_exercised"] == {"duration": True, "width": True}
    assert block["evidence_exercised"] == {"per-char": True, "whitespace": True}
    duration = next(row for row in block["cases"] if row["variant"] == "duration")
    assert duration["case"] == "coarse-duration-zh"
    assert duration["refiner_comparison"]["materialized"] is True
    assert duration["refiner_comparison"]["diffs_confined_to_coarse_caused"] is True


def test_materialized_coarse_acceptance_regression_is_a_failure(monkeypatch) -> None:
    real_replay = calib.replay_shadow

    def replay_with_unconfined_duration(case):
        result = real_replay(case)
        if case.id == "coarse-duration-zh":
            artifact = result.shadow
            assert artifact["kind"] == "segmentation-shadow-incomplete"
            comparison = artifact["diagnostic"]["refiner_comparison"]
            assert comparison["materialized"] is True
            comparison["diffs_confined_to_coarse_caused"] = False
        return result

    monkeypatch.setattr(calib, "replay_shadow", replay_with_unconfined_duration)
    block = calib.run_coarse_gates(calib.load_corpus(calib.DEFAULT_CORPUS))
    assert (
        "coarse-duration-zh: refiner-off diff is not confined to a "
        "coarse_caused interval"
    ) in block["failures"]
    assert not any("coarse-duration-zh" in stop for stop in block["stops"])
