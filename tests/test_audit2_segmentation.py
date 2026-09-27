"""Regressions for the 2026-09-26 audit fixes in the segmentation cluster."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from voxweave import pipeline
from voxweave.config import gap_thresholds
from voxweave.core import shadow_v2
from voxweave.core.boundary_cost import (
    PauseEvidence,
    W_PAUSE,
    pause_cut_cost,
    quantize,
    ramp_integral_mean,
)
from voxweave.core.boundary_lattice import (
    INFEASIBLE_REASONS,
    UNTIMED_RUN,
    IncrementalPacker,
    build_document_lattice,
    preflight_profile,
)
from voxweave.core.boundary_v2 import (
    V1Partition,
    _pinned_neighbour_margins,
    build_cost_context,
    build_cost_tables,
    optimize_document,
    score_path,
    solve_interval,
)
from voxweave.core.layout import _join
from voxweave.core.segdoc import DisplayProfile, SegDocument, SourceUnit
from voxweave.core.subunit import (
    RefinementConservationError,
    assert_refinement_conserved,
)
from voxweave.core.unit_repair import repair_stranded_tails

FLAG = pipeline.SEG_V2_SHADOW_ENV

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    """Import a module from ``scripts/`` by path (it is not an installed package)."""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _profile(language: str = "en", **over) -> DisplayProfile:
    base = dict(
        language=language,
        max_line_length=42,
        max_lines=2,
        clause_ms=400.0,
        vad_skip_ms=1000.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=7.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=0.458,
    )
    base.update(over)
    return DisplayProfile(**base)


def _document(spec, *, prof: DisplayProfile | None = None) -> SegDocument:
    prof = prof or _profile()
    units = [
        SourceUnit(f"u{index}", surface, start, end)
        for index, (surface, start, end) in enumerate(spec)
    ]
    return SegDocument(
        language=prof.language,
        units=units,
        profile=prof,
        vad_speech=[(0.0, 60.0)],
        shot_changes=None,
        sing_spans=None,
        speaker_turns=None,
        manifest={},
        text=_join([unit.surface for unit in units], prof.language),
    )


def _timed(surfaces, *, dur: float = 0.3, gap: float = 0.1):
    out = []
    t = 0.0
    for surface in surfaces:
        out.append((surface, round(t, 6), round(t + dur, 6)))
        t += dur + gap
    return out


# ------------------------------------------------------------ followup ad3-1


AD3_1_SHAPES = [
    ("en", ["x", "...", "...", "100%"]),
    ("en", ["in", "2019", ",", "5", "people"]),
    ("en", [",", "5", "people"]),
    ("zh", ["好", ",", "5", "个"]),
    ("zh", ["好", ".", "5", "个"]),
]


@pytest.mark.parametrize("canonical_spaced", [False, True])
@pytest.mark.parametrize(("language", "surfaces"), AD3_1_SHAPES)
def test_a_mixed_interval_admits_no_edge_that_renders_nothing(
    language, surfaces, canonical_spaced
):
    """The '.' of '...' before '100%' survives only on the joined stream.

    A cue that ends after such an atom strips it, so an atom whose display needs
    what follows it is invisible and folds into a neighbour: every atom left in
    a mixed interval shows something in any cue, and no edge renders nothing.
    """
    doc = _document(_timed(surfaces), prof=_profile(language))
    lattice = build_document_lattice(doc, canonical_spaced=canonical_spaced)
    for interval in lattice.lattices:
        assert interval.all_invisible is False
        assert interval.infeasible is None
        assert all(atom.display for atom in interval.atoms)
        assert interval.edges
        assert all(edge.display_text.strip() for edge in interval.edges)


def test_a_lookahead_only_atom_folds_into_its_carrier():
    doc = _document(_timed(["x", "...", "...", "100%"]))
    (interval,) = build_document_lattice(doc).lattices
    assert [(atom.text, atom.display) for atom in interval.atoms] == [
        ("x ... ...", "x"),
        ("100%", "100%"),
    ]


def test_a_punctuation_only_interval_takes_the_all_invisible_chain():
    """Between two barriers, '...' alone renders nothing: a forced chain, not v1."""
    spec = [("hello", 0.0, 0.3), ("...", 3.0, 3.3), ("100%", 6.0, 6.3)]
    solution = optimize_document(_document(spec))
    middle = solution.lattice.lattices[1]
    assert middle.all_invisible is True
    assert middle.infeasible is None
    assert all(item.optimized for item in solution.solutions)
    assert solution.artifact["validator"]["raw"]["exit_driving"] == 0


@pytest.mark.parametrize(("language", "surfaces"), AD3_1_SHAPES)
def test_the_harness_n14_oracle_agrees_with_the_lattice(language, surfaces):
    """The harness rebuilds the legal edge set independently; both must agree."""
    pytest.importorskip("jsonschema")
    calib = _load_script("calib_segmentation")
    units = [
        SourceUnit(f"u{index}", surface, start, end)
        for index, (surface, start, end) in enumerate(_timed(surfaces))
    ]
    result = calib._n14_oracle(units, _profile(language), [(0, len(units))])
    assert result["checked"] > 0
    assert (result["false_negative"], result["false_positive"]) == (0, 0), result


# ------------------------------------------------------------------ seg-v2#19


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("key", ["clause_ms", "max_cue_s", "min_cue_s", "cps"])
def test_preflight_refuses_a_non_finite_knob(key, value):
    violations = preflight_profile(_profile(**{key: value}))
    assert [(v.key, v.reason) for v in violations] == [(key, "non-finite")]
    # The refusal serializes to valid JSON.
    json.dumps([v.to_dict() for v in violations], allow_nan=False)


def test_preflight_still_accepts_the_committed_defaults():
    assert preflight_profile(_profile()) == ()


# ------------------------------------------------------------------ seg-v2#23


def test_an_interval_blocked_only_by_untimed_words_is_typed_untimed_run():
    words = ["alpha", *(f"word{i}" for i in range(30)), "omega"]
    spec = [
        (word, 0.3 * i, 0.3 * i + 0.2)
        if i in (0, len(words) - 1)
        else (word, None, None)
        for i, word in enumerate(words)
    ]
    (interval,) = build_document_lattice(_document(spec)).lattices
    assert interval.infeasible is not None
    assert interval.infeasible.reason == UNTIMED_RUN
    assert UNTIMED_RUN in INFEASIBLE_REASONS


# ------------------------------------------------------------------ seg-v2#12


@pytest.mark.parametrize("language", ["zh", "ja"])
def test_the_incremental_packer_refuses_a_no_space_language(language):
    with pytest.raises(ValueError, match="spaced-language fold"):
        IncrementalPacker(language, 18, 1)


# ------------------------------------------------------------------ seg-v2#22


def test_pause_cut_cost_prices_the_evidences_own_uncertainty():
    evidence = PauseEvidence(
        gap_ms_raw=300.0,
        vad_state="silence",
        overlap_fraction=0.0,
        uncertainty_ms=120.0,
        effective_ms=300.0,
        ramp_ms=385.0,
    )
    assert pause_cut_cost(evidence) == ramp_integral_mean(
        300.0, 385.0, amplitude=W_PAUSE, uncertainty_ms=120.0
    )


# --------------------------------------------------------------- seg-v2#6/#9


def _solved_interval():
    words = ["one", "two", "three,", "four", "five.", "six", "seven", "eight"] * 4
    doc = _document(
        _timed(words, dur=0.25, gap=0.05), prof=_profile(max_line_length=16)
    )
    built = build_document_lattice(doc)
    (lattice,) = built.lattices
    tables = build_cost_tables(lattice, build_cost_context(doc, built))
    return lattice, tables


def test_pinned_margins_equal_rescoring_every_relocated_path():
    lattice, tables = _solved_interval()
    selected = solve_interval(lattice, tables).best
    assert selected.cuts, "the fixture must cut somewhere"
    edge_keys = {(edge.start_node, edge.end_node) for edge in lattice.edges}
    chain = (0, *selected.cuts, len(lattice.atoms))
    expected = []
    for index, cut in enumerate(selected.cuts):
        before, after = chain[index], chain[index + 2]
        deltas = []
        for node in lattice.nodes:
            if node == cut or not before < node < after:
                continue
            if (before, node) not in edge_keys or (node, after) not in edge_keys:
                continue
            relocated = list(selected.cuts)
            relocated[index] = node
            total = score_path(lattice, tables, relocated).total
            deltas.append(quantize(total - selected.total))
        if deltas:
            expected.append(min(deltas))
    assert _pinned_neighbour_margins(lattice, tables, selected) == tuple(expected)


# ------------------------------------------------------------------ seg-v2#24


def test_units_no_v1_cue_covers_are_v2_damage_without_a_reference():
    words = ["alpha", *(f"word{i}" for i in range(30)), "omega"]
    spec = [
        (word, 0.3 * i, 0.3 * i + 0.2)
        if i in (0, len(words) - 1)
        else (word, None, None)
        for i, word in enumerate(words)
    ]
    solution = optimize_document(_document(spec))
    (interval,) = solution.solutions
    assert interval.adopted is not None and not interval.adopted.cues
    assert interval.validator_raw.origin == "v2"
    assert interval.validator_raw.exit_driving
    raw = solution.artifact["validator"]["raw"]
    assert raw["origin"] == "v2"
    assert raw["exit_driving"] > 0


def _untimed_run(first: str, last: str, start: float):
    words = [first, *(f"{first}{i}" for i in range(30)), last]
    return [
        (word, start + 0.3 * i, start + 0.3 * i + 0.2)
        if i in (0, len(words) - 1)
        else (word, None, None)
        for i, word in enumerate(words)
    ]


def test_uncovered_units_stay_v2_damage_beside_an_adopted_interval():
    """One fallback adopts v1 cues, the next finds none: the gap is still v2's."""
    left = _untimed_run("alpha", "omega", 0.0)
    right = _untimed_run("beta", "gamma", 20.0)
    doc = _document(left + right)
    cue = {
        "text": _join([surface for surface, _, _ in left], "en"),
        "start": 0.0,
        "end": left[-1][2],
    }
    # Two v1 bounds but a single cue: the right interval has nothing to adopt.
    solution = optimize_document(doc, v1=V1Partition(cuts=(len(left),), cues=(cue,)))
    first, second = solution.solutions
    assert first.adopted is not None and first.adopted.cues
    assert second.adopted is not None and not second.adopted.cues
    raw = solution.artifact["validator"]["raw"]
    whole = [v for v in raw["violations"] if v["cue_index"] is None]
    assert [(v["kind"], v["origin"]) for v in whole] == [("unit-conservation", "v2")]
    assert solution.artifact["validator"]["interval_document_agree"] is True


def test_a_tiling_document_with_a_fallback_keeps_the_v2_default_origin():
    left = _untimed_run("alpha", "omega", 0.0)
    doc = _document(left)
    cue = {
        "text": _join([surface for surface, _, _ in left], "en"),
        "start": 0.0,
        "end": left[-1][2],
    }
    solution = optimize_document(doc, v1=V1Partition(cuts=(), cues=(cue,)))
    (only,) = solution.solutions
    assert only.adopted is not None and only.adopted.cues
    raw = solution.artifact["validator"]["raw"]
    assert raw["origin"] == "v2"
    assert not [v for v in raw["violations"] if v["cue_index"] is None]


# ------------------------------------------------------------------ seg-adapters#3


def test_refinement_conservation_checks_every_parent():
    parents = [SourceUnit("u0", "ab", 0.0, 1.0), SourceUnit("u1", "cd", 1.0, 2.0)]
    children = [
        SourceUnit("u0", "a", 0.0, 0.5),
        SourceUnit("u1", "b", 0.5, 1.0),
        SourceUnit("u2", "cd", 1.0, 2.0),
    ]
    assert_refinement_conserved(parents, children, [0, 0, 1], lang="zh")
    moved = [children[0], SourceUnit("u1", "c", 0.5, 1.0), children[2]]
    with pytest.raises(RefinementConservationError, match="parent 0"):
        assert_refinement_conserved(parents, moved, [0, 0, 1], lang="zh")


# ------------------------------------------------------------------ seg-core#17


def test_stranded_tail_repair_reads_surfaces_like_the_engine():
    """A ``text`` that is present but empty falls back to ``word``, as it does
    in the engine; ``text: None`` must not count as the four characters 'None'."""
    base = [
        {"text": "今日", "start": 0.0, "end": 0.4},
        {"text": "は", "start": 0.4, "end": 0.6},
        {"text": "い", "start": 0.6, "end": 0.8},
        {"text": "い", "start": 4.0, "end": 4.2},
        {"text": "天気", "start": 8.0, "end": 8.4},
    ]
    speech = [(0.0, 0.8), (4.0, 4.2), (8.0, 8.4)]
    expected = repair_stranded_tails(base, "ja", speech)
    for blank in ("", None):
        variant = [
            {
                "text": blank,
                "word": unit["text"],
                "start": unit["start"],
                "end": unit["end"],
            }
            for unit in base
        ]
        repaired = repair_stranded_tails(variant, "ja", speech)
        assert [(u["start"], u["end"]) for u in repaired] == [
            (u["start"], u["end"]) for u in expected
        ]


# ---------------------------------------------------------- shadow lane (flag on)


def _segment(case: dict, **kwargs) -> pipeline.SegmentationResult:
    return pipeline.segment_document(
        language=case["language"],
        word_segments=case["word_segments"],
        vad_speech=pipeline._spans_in(case.get("vad_speech")),
        shot_changes=[float(t) for t in case.get("shot_changes") or []] or None,
        sing_spans=pipeline._spans_in(case.get("sing_spans")),
        speaker_turns=pipeline._turns_in(case.get("speaker_turns")),
        **kwargs,
    )


_PLAIN = {
    "language": "en",
    "word_segments": [
        {"text": "Where", "start": 0.0, "end": 0.4},
        {"text": "did", "start": 0.5, "end": 0.8},
        {"text": "you", "start": 0.9, "end": 1.2},
        {"text": "go", "start": 1.4, "end": 2.0},
        {"text": "Nowhere", "start": 2.4, "end": 3.0},
        {"text": "special", "start": 3.1, "end": 3.6},
    ],
}


def test_the_lane_names_are_defined_once():
    import inspect

    from voxweave.core import shadow_lanes, shadow_schema

    names = (
        shadow_lanes.LANE_CORE,
        shadow_lanes.LANE_LEGACY,
        shadow_lanes.LANE_FINALIZER,
        shadow_lanes.LANE_DISPLAY,
    )
    assert (
        shadow_schema.LANE_CORE,
        shadow_schema.LANE_LEGACY,
        shadow_schema.LANE_FINALIZER,
        shadow_schema.LANE_DISPLAY,
    ) == names
    assert (
        shadow_v2.SHADOW_LANE_CORE,
        shadow_v2.SHADOW_LANE_DELIVERY_LEGACY,
        shadow_v2.SHADOW_LANE_FINALIZER,
        shadow_v2.SHADOW_LANE_LEGACY_DISPLAY,
    ) == names
    for module in (shadow_schema, shadow_v2):
        source = inspect.getsource(module)
        assert not [name for name in names if f'"{name}"' in source], module


def test_an_invalid_profile_names_the_refused_knob_and_types_the_reason(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    artifact = _segment(
        copy.deepcopy(_PLAIN), thresholds={**gap_thresholds("en"), "clause_ms": 0.0}
    ).shadow
    assert artifact is not None
    assert artifact["kind"] == "segmentation-shadow-incomplete"
    assert artifact["error"]["reason"] == "invalid-profile"
    assert "clause_ms=0.0 (not-positive)" in artifact["error"]["detail"]
    # The real ledgers ride on the envelope only; no empty placeholder in the
    # diagnostic contradicts them.
    assert "shadow_degraded" in artifact
    assert "shadow_degraded" not in artifact["diagnostic"]
    assert "production_degraded" not in artifact["diagnostic"]


def test_the_v1_finalizer_row_names_its_surface_projection(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    artifact = _segment(copy.deepcopy(_PLAIN)).shadow
    assert artifact is not None and artifact["schema_version"] == 2
    rows = artifact["lanes"][shadow_v2.SHADOW_LANE_FINALIZER]["rows"]
    assert rows["v1"]["projection"] == artifact["v1_projection"]["mode"]
    assert rows["v2"]["projection"] == "solver-partition"
    # One raw-stage answer, published in both places.
    assert artifact["raw"]["validator"] == artifact["validator"]["raw"]


def test_a_refused_admission_keeps_the_assembled_artifact(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    real = shadow_v2._shadow_v2_artifact

    def drop_authorities(*args, **kwargs):
        artifact = real(*args, **kwargs)
        artifact.pop("authorities")
        return artifact

    monkeypatch.setattr(shadow_v2, "_shadow_v2_artifact", drop_authorities)
    artifact = _segment(copy.deepcopy(_PLAIN)).shadow
    assert artifact is not None
    assert artifact["kind"] == "segmentation-shadow-error"
    assert artifact["error"]["type"] == "ShadowAdmissionError"
    assert "artifact: missing keys authorities" in artifact["admission_errors"]
    assert "lanes" in artifact["diagnostic"]
    assert "shadow_degraded" not in artifact["diagnostic"]


def test_an_admission_error_quotes_a_bounded_number_of_refusals():
    errors = [f"refusal {index}" for index in range(40)]
    exc = shadow_v2.ShadowAdmissionError({}, errors)
    assert exc.errors == tuple(errors)
    message = str(exc)
    assert "refusal 9" in message and "refusal 10;" not in message
    assert message.endswith(f"... and {40 - shadow_v2.ADMISSION_ERRORS_SHOWN} more")


# ------------------------------------------------------- seg-adapters#11/#22


def test_split_releases_its_issuance_records(tmp_path):
    from voxweave import segmentation_adapter, segmentation_candidates

    path = tmp_path / "episode.json"
    path.write_text(json.dumps(copy.deepcopy(_PLAIN)), encoding="utf-8")
    pipeline.split(path)
    before = (
        len(segmentation_adapter._LEGACY),
        len(segmentation_adapter._ADAPTER),
        len(segmentation_candidates._ENCODED),
        len(segmentation_candidates._VERIFIED),
    )
    pipeline.split(path)
    pipeline.split(path)
    after = (
        len(segmentation_adapter._LEGACY),
        len(segmentation_adapter._ADAPTER),
        len(segmentation_candidates._ENCODED),
        len(segmentation_candidates._VERIFIED),
    )
    assert after == before
