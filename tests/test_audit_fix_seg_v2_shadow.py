"""Regression pins for the v2 shadow optimizer and the segmentation adapter.

* a typed v1 fallback whose complete cues straddle a barrier absorbs the
  neighbouring interval instead of owning its units a second time;
* the global v1 reference records every rounded cut, matches chain cues to the
  v1 cue they came from, and prices v1 under the speaker rows' own rules;
* the refiner-off finalizer row is serialized in the refined unit space the
  schema-2 contract validates it against;
* the segmentation adapter builds the v2 delivery whenever the registry selects
  that family, not only when the shadow switch is on.
"""

from __future__ import annotations

import copy
import random

import pytest

from tests.test_boundary_v2 import document, profile
from tests.test_p6_segmentation_candidates import _issued
from voxweave import pipeline
from voxweave.core import boundary_v2
from voxweave.core.boundary_cost import quantize
from voxweave.core.boundary_v2 import V1Partition, optimize_document
from voxweave.core.finalizer import FinalizerPreview
from voxweave.core.speaker_evidence import project_speaker_evidence, speaker_evidence
from voxweave.core.subunit import refine_document


def _cue(text: str, start: float, end: float) -> dict:
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [],
        "speech_start": start,
        "speech_end": end,
    }


def _assert_tiled_without_v2_damage(solution, doc) -> None:
    cues = [cue for item in solution.solutions for cue in item.cues]
    assert " ".join(cue["text"] for cue in cues) == doc.text
    ranges = [item.unit_range for item in solution.solutions]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(doc.units)
    for left, right in zip(ranges, ranges[1:]):
        assert left[1] == right[0]
    raw = solution.artifact["validator"]["raw"]
    assert raw["exit_driving"] == 0
    assert not any(
        row["kind"] in {"text-conservation", "unit-conservation", "overlap"}
        for row in raw["violations"]
    )


# ------------------------------------------------ fallback adoption tiling


def test_fallback_expanding_backwards_absorbs_the_optimized_neighbour():
    """v1's cue [1, 4) straddles the barrier in front of the infeasible [2, 4)."""
    doc = document(
        [
            ("one", 0.0, 0.5),
            ("two", 0.6, 1.0),
            ("three", 20.0, 20.5),
            ("four", float("nan"), 21.0),
        ]
    )
    v1 = V1Partition(
        cuts=(1,), cues=(_cue("one", 0.0, 0.5), _cue("two three four", 0.6, 21.0))
    )
    solution = optimize_document(doc, v1=v1)
    # The symptom: v1's straddling cue used to own ``two`` a second time, which
    # the document pass reported as an exit-driving v2 text-conservation row.
    _assert_tiled_without_v2_damage(solution, doc)
    assert [item.interval.unit_start for item in solution.solutions] == [0, 2]
    assert all(not item.optimized for item in solution.solutions)
    head, tail = solution.solutions
    assert head.lattice.infeasible is None
    assert head.adopted is not None
    assert head.adopted.reason == boundary_v2.ADOPTION_ABSORBED
    assert head.unit_range == (0, 4)
    assert tail.unit_range == (4, 4)
    assert tail.cues == ()
    assert solution.artifact["totals"]["optimized_intervals"] == 0


def test_fallback_expanding_forwards_absorbs_the_optimized_neighbour():
    doc = document(
        [
            ("one", 0.0, 0.5),
            ("two", float("nan"), 1.0),
            ("three", 20.0, 20.5),
            ("four", 20.6, 21.0),
        ]
    )
    v1 = V1Partition(
        cuts=(1,), cues=(_cue("one", 0.0, 0.5), _cue("two three four", 0.6, 21.0))
    )
    solution = optimize_document(doc, v1=v1)
    _assert_tiled_without_v2_damage(solution, doc)
    assert all(not item.optimized for item in solution.solutions)
    assert solution.solutions[1].adopted is not None
    assert solution.solutions[1].adopted.reason == boundary_v2.ADOPTION_ABSORBED


def test_a_v1_cue_spanning_a_whole_interval_leaves_that_member_empty():
    """The middle interval (one unwrappable word) sits entirely inside v1's cue."""
    wide = "x" * 100
    doc = document(
        [
            ("one", 0.0, 0.5),
            ("two", 0.6, 1.0),
            (wide, 20.0, 20.5),
            ("four", 40.0, 40.5),
            ("five", 40.6, 41.0),
        ]
    )
    v1 = V1Partition(
        cuts=(1, 4),
        cues=(
            _cue("one", 0.0, 0.5),
            _cue(f"two {wide} four", 0.6, 40.5),
            _cue("five", 40.6, 41.0),
        ),
    )
    solution = optimize_document(doc, v1=v1)
    assert len(solution.solutions) == 3
    assert solution.solutions[1].lattice.infeasible is not None
    assert [item.unit_range for item in solution.solutions] == [(0, 4), (4, 4), (4, 5)]
    assert solution.solutions[1].validator_raw.violations == ()
    _assert_tiled_without_v2_damage(solution, doc)


def test_a_fallback_that_reaches_no_neighbour_is_unchanged():
    doc = document(
        [
            ("one", 0.0, 0.5),
            ("two", 0.6, 1.0),
            ("three", 20.0, 20.5),
            ("four", float("nan"), 21.0),
        ]
    )
    v1 = V1Partition(
        cuts=(2,), cues=(_cue("one two", 0.0, 1.0), _cue("three four", 20.0, 21.0))
    )
    solution = optimize_document(doc, v1=v1)
    head, tail = solution.solutions
    assert head.optimized
    assert tail.adopted is not None
    assert tail.unit_range == (2, 4)
    assert tail.adopted.fallback_expansion_units is None
    assert tail.adopted.reason == "span-preflight"


# ------------------------------------------------------ global v1 reference


def _folded_document():
    """``3``/``.``/``75`` fold into one atom, so units 2 and 3 have no node."""
    doc = document(
        [
            ("a", 0.0, 0.2),
            ("3", 0.3, 0.4),
            (".", 0.4, 0.5),
            ("75", 0.5, 0.7),
            ("b", 0.9, 1.1),
        ],
        prof=profile(max_line_length=12, max_lines=1),
    )
    object.__setattr__(doc, "text", "a 3.75 b")
    return doc


def test_a_v1_cut_inside_the_last_atom_is_recorded_not_dropped():
    doc = document(
        [("next", 0.0, 0.3), ("3", 0.5, 0.6), (".", 0.6, 0.7), ("75", 0.7, 0.9)],
        prof=profile(max_line_length=12, max_lines=1),
    )
    object.__setattr__(doc, "text", "next 3.75")
    solution = optimize_document(doc, v1=V1Partition(cuts=(1, 3)))
    rounded = solution.artifact["v1"]["rounded_cuts"]
    assert rounded == [{"landed_unit": 4, "node": 2, "unit": 3}]


def test_chain_cues_read_the_v1_cue_they_came_from_after_rounding(monkeypatch):
    """Cuts 2 and 4 both land on node 2, so v1 cue 2 collapses away.

    The chain is then one cue shorter than v1, and its last cue is v1's cue 3
    (``b``), not v1's cue 2 (``.75``) that merely shares its chain position.
    """
    doc = _folded_document()
    cues = (
        _cue("a", 0.0, 0.2),
        _cue("3", 0.3, 0.4),
        _cue(".75", 0.4, 0.7),
        _cue("b", 0.9, 1.1),
    )
    seen: list[str] = []
    real = boundary_v2.evidence_span_from_cue

    def spy(cue):
        seen.append(cue["text"])
        return real(cue)

    monkeypatch.setattr(boundary_v2, "evidence_span_from_cue", spy)
    solution = optimize_document(
        doc, v1=V1Partition(cuts=(1, 2, 4), cues=cues), speaker_weight=3.0
    )
    assert seen == ["a", "3", "b"]
    assert [row["unit"] for row in solution.artifact["v1"]["rounded_cuts"]] == [2]


def _production_solve(doc, v1=None):
    refined, split = refine_document(doc)
    speakers = project_speaker_evidence(
        speaker_evidence(doc), refined_units=split.units, origin=split.origin
    )
    return optimize_document(
        refined,
        v1=v1,
        preview=FinalizerPreview(refined.profile),
        subunit_split=split,
        speakers=speakers,
        speaker_weight=3.0,
    )


@pytest.mark.parametrize("seed", [220, 234, 235, 254, 291])
def test_v1_reference_prices_v2s_own_partition_at_v2s_price(seed):
    """Speaker rows: v1 is priced by the rules the speaker lattice priced v2 by.

    When v1's partition IS the v2 selection, the reference must reproduce the
    selected path's total exactly; it used to run the policy-1 packer and span
    defaults, and priced refined documents several points cheaper.
    """
    words = [
        "I",
        "well,",
        "no",
        "no!",
        "a",
        "the",
        "go",
        "x?",
        "one two three four five six seven",
        "alpha beta gamma delta",
    ]
    rng = random.Random(seed)
    spec = []
    t = 0.0
    for _ in range(rng.randint(4, 10)):
        duration = rng.choice([0.3, 0.5, 1.5])
        spec.append((rng.choice(words), round(t, 3), round(t + duration, 3)))
        t += duration + rng.choice([0.05, 0.1, 0.4])
    doc = document(
        spec,
        prof=profile(max_line_length=rng.choice([10, 16, 20]), max_lines=2),
    )
    first = _production_solve(doc)
    assert first.subunit_split.refined_parent_count > 0
    assert len(first.solutions) == 1 and first.solutions[0].optimized
    item = first.solutions[0]
    second = _production_solve(
        doc, v1=V1Partition(cuts=item.partition_units, cues=item.cues)
    )
    reference = second.v1_reference
    assert reference is not None
    # The premise: v1 IS the selection, so both totals price one partition.
    assert second.solutions[0].partition_units == item.partition_units
    selected = second.solutions[0].selection.policy_selected
    assert quantize(reference.global_cost.total) == quantize(selected.total)
    assert not [
        row for row in reference.hard_disagreements if row["kind"] == "over-budget"
    ]


# ------------------------------------------------------ refiner-off admission


def test_a_refined_document_with_a_projectable_v1_passes_schema_two(monkeypatch):
    monkeypatch.setenv(pipeline.SEG_V2_SHADOW_ENV, "1")
    units = [
        {"text": "Where", "start": 0.0, "end": 0.4},
        {"text": "did", "start": 0.5, "end": 0.8},
        {"text": "you", "start": 0.9, "end": 1.2},
        {"text": "go", "start": 1.4, "end": 2.0},
        {
            "text": "Nowhere special at all my friend, just around",
            "start": 2.4,
            "end": 5.0,
        },
        {"text": "okay", "start": 5.2, "end": 5.6},
    ]
    result = pipeline.segment_document(
        language="en",
        word_segments=copy.deepcopy(units),
        vad_speech=[(0.0, 2.0), (2.4, 5.6)],
        shot_changes=None,
        sing_spans=None,
        speaker_turns=None,
    )
    artifact = result.shadow
    assert artifact is not None
    assert artifact.get("error") is None, artifact.get("error")
    assert artifact["kind"] == "segmentation-shadow"
    assert artifact["schema_version"] == 2
    assert artifact["refiner_comparison"]["status"] == "refined-counterfactual"
    unit_count = artifact["coverage"]["unit_count"]
    assert unit_count > len(units)
    row = artifact["lanes"][pipeline.SHADOW_LANE_FINALIZER]["rows"]["refiner-off"]
    assert row["validator"]["unit_count"] == unit_count
    assert row["cues"][-1]["unit_range"][1] == unit_count


# ------------------------------------------------ segmentation adapter cutover


@pytest.fixture
def boundary_family(monkeypatch):
    """Issue contexts as if the registry had moved ``en`` to boundary-v2."""
    import voxweave.align_context as align_context

    monkeypatch.setattr(align_context, "engine_family_for", lambda _iso: "boundary-v2")


def test_registry_cutover_builds_and_selects_v2_without_the_shadow_switch(
    tmp_path, boundary_family
):
    from voxweave.segmentation_adapter import run_locked_segmentation_adapter
    from voxweave.segmentation_candidates import (
        encode_segmentation_candidates,
        project_selected_sdh_dialogue,
        select_segmentation_candidate,
        verify_selected_segmentation_projection,
    )

    context, issued = _issued(tmp_path)
    assert context.engine_family == "boundary-v2"
    result = run_locked_segmentation_adapter(context, issued, shadow_enabled=False)
    assert result.v2_status.kind == "valid"
    assert result.v2 is not None
    selected = select_segmentation_candidate(
        context, encode_segmentation_candidates(context, result)
    )
    assert selected.engine_family == "boundary-v2"
    verified = verify_selected_segmentation_projection(context, result, selected)
    assert verified.engine_family == "boundary-v2"
    dialogue = project_selected_sdh_dialogue(context, result, verified)
    assert " ".join(str(row["text"]) for row in dialogue) == "hello world"


def test_registry_cutover_raises_the_v2_failure_instead_of_hiding_it(
    tmp_path, boundary_family, monkeypatch
):
    import voxweave.segmentation_adapter as adapter
    from voxweave.align_failures import CanonicalFailure

    failure = CanonicalFailure(
        "segmentation-v2-invalid", "segmentation-adapter", "delivery-unit-range"
    )

    def fail(_record):
        raise adapter.SegmentationProductionError(failure)

    monkeypatch.setattr(adapter, "_build_boundary_delivery", fail)
    context, issued = _issued(tmp_path)
    with pytest.raises(adapter.SegmentationProductionError) as error:
        adapter.run_locked_segmentation_adapter(context, issued, shadow_enabled=False)
    assert error.value.failure == failure

    def crash(_record):
        raise RuntimeError("injected")

    monkeypatch.setattr(adapter, "_build_boundary_delivery", crash)
    context, issued = _issued(tmp_path)
    with pytest.raises(adapter.SegmentationProductionError) as error:
        adapter.run_locked_segmentation_adapter(context, issued, shadow_enabled=False)
    assert error.value.failure == CanonicalFailure(
        "shadow-internal-error", "segmentation-adapter", "w1-stage"
    )
    assert isinstance(error.value.__cause__, RuntimeError)


def test_legacy_family_still_skips_v2_and_types_shadow_failures(tmp_path, monkeypatch):
    import voxweave.segmentation_adapter as adapter

    context, issued = _issued(tmp_path)
    assert context.engine_family == "legacy-v1"
    result = adapter.run_locked_segmentation_adapter(
        context, issued, shadow_enabled=False
    )
    assert result.v2_status.kind == "not-requested"
    assert result.v2 is None

    def crash(_record):
        raise RuntimeError("injected")

    monkeypatch.setattr(adapter, "_build_boundary_delivery", crash)
    context, issued = _issued(tmp_path)
    result = adapter.run_locked_segmentation_adapter(
        context, issued, shadow_enabled=True
    )
    assert result.v2_status.kind == "invalid"
    assert result.v2_status.failure is not None
    assert result.v2_status.failure.kind == "shadow-internal-error"
