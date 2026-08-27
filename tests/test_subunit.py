"""P5 W2 sub-unit refinement, provenance folds, and coarse-corpus goldens.

Every numeric expectation in this file is derived directly from the LAW.  In
particular, derived child times use the parent's duration multiplied by the
child's share of ``_token_char_count``; none of the values were captured from
the implementation.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest

from voxweave.core.boundary_lattice import (
    CAP_EPS_S,
    Edge,
    IncrementalPacker,
    LatticeAtom,
    band_atoms,
)
from voxweave.core.boundary_v2 import (
    _document_partition,
    materialize_cues,
    optimize_document,
)
from voxweave.core.layout import _join, _line_budget_width, _vis_width
from voxweave.core.providers import degradation_capture
from voxweave.core.segdoc import DisplayProfile, SegDocument, SourceUnit
from voxweave.core.subunit import (
    EVIDENCE_KINDS,
    EVIDENCE_RANKING,
    RefineResult,
    RefinementConservationError,
    assert_refinement_conserved,
    refine_document,
    refine_units,
    speech_span_units,
)


ROOT = Path(__file__).resolve().parents[1]
COARSE_CORPUS = ROOT / "calibration/segmentation/corpus-coarse.json"
TRACKED_CORPUS = ROOT / "calibration/segmentation/corpus.json"


def profile(language: str = "ja", **over: object) -> DisplayProfile:
    base: dict[str, object] = {
        "language": language,
        "max_line_length": 18 if language in {"ja", "zh", "yue"} else 42,
        "max_lines": 1 if language in {"ja", "zh", "yue"} else 2,
        "clause_ms": 400.0,
        "vad_skip_ms": 1000.0,
        "offline_ms": 700.0,
        "min_cue_s": 0.5,
        "max_cue_s": 7.0,
        "glue_gap_s": 0.3,
        "cps": 17.0,
        "lag_out_s": 0.25,
        "shot_snap_s": 0.458,
    }
    base.update(over)
    return DisplayProfile(**base)  # type: ignore[arg-type]


def unit(
    surface: str,
    start: float | None,
    end: float | None,
    *,
    index: int = 0,
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
    units: list[SourceUnit], prof: DisplayProfile, *, text: str | None = None
) -> SegDocument:
    return SegDocument(
        language=prof.language,
        units=units,
        profile=prof,
        vad_speech=None,
        shot_changes=None,
        sing_spans=None,
        speaker_turns=None,
        manifest={"engine": "legacy-v1"},
        text=_join([item.surface for item in units], prof.language)
        if text is None
        else text,
    )


def atom(
    index: int,
    text: str,
    start: float | None,
    end: float | None,
    unit_start: int,
    unit_end: int,
) -> LatticeAtom:
    return LatticeAtom(
        index=index,
        text=text,
        start=start,
        end=end,
        unit_start=unit_start,
        unit_end=unit_end,
        end_pen=0,
        start_pen=0,
        boundary_pen=0,
        phrase_start=True,
        forced_boundary=False,
        display=text,
        members=(index,),
    )


def edge(start: int, end: int, atoms: list[LatticeAtom]) -> Edge:
    chunk = atoms[start:end]
    return Edge(
        start_node=start,
        end_node=end,
        text="".join(item.text for item in chunk),
        display_text="".join(item.text for item in chunk),
        lines=1,
        line_widths=(sum(_vis_width(item.text) for item in chunk),),
        span_start=next((item.start for item in chunk if item.start is not None), None),
        span_end=next(
            (item.end for item in reversed(chunk) if item.end is not None), None
        ),
        waiver=None,
    )


# ---------------------------------------------------------------- triggers


def test_trigger_thresholds_are_strict_and_cap_epsilon_is_imported() -> None:
    # ``max_lines`` is deliberately large: the trigger is one line's budget,
    # never total multi-line capacity.
    prof = profile("yue", max_line_length=2, max_lines=4, max_cue_s=7.0)
    budget = _line_budget_width(prof.max_line_length, prof.language)
    assert budget == 4

    width_equal = refine_units([unit("甲乙", 0.0, 1.0)], lang="yue", profile=prof)
    width_over = refine_units([unit("甲乙a", 0.0, 1.0)], lang="yue", profile=prof)
    assert width_equal.refined_parent_count == 0
    assert width_over.refined_parent_count == 1

    at_threshold = 7.0 + CAP_EPS_S
    just_over = math.nextafter(at_threshold, math.inf)
    duration_equal = refine_units(
        [unit("甲乙", 0.0, at_threshold)], lang="yue", profile=profile("yue")
    )
    duration_over = refine_units(
        [unit("甲乙", 0.0, just_over)], lang="yue", profile=profile("yue")
    )
    assert duration_equal.refined_parent_count == 0
    assert duration_over.refined_parent_count == 1


def test_disabled_cap_never_causes_a_duration_refinement() -> None:
    result = refine_units(
        [unit("甲乙", 0.0, 10_000.0)],
        lang="yue",
        profile=profile("yue", max_cue_s=0.0),
    )
    assert result.refined_parent_count == 0
    assert [item.surface for item in result.units] == ["甲乙"]


def test_invalid_profile_preflight_precedes_any_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave.core import subunit

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the splitter ran before profile preflight")

    monkeypatch.setattr(subunit, "_split_parent", forbidden)
    result = refine_units(
        [unit("甲乙丙丁", 0.0, 20.0)],
        lang="yue",
        profile=profile("yue", max_line_length=0),
    )
    assert result.refined_parent_count == 0


def test_trigger_is_unit_local_and_neighbour_free() -> None:
    target = unit("ab cd", 10.0, 18.0, index=7)
    prof = profile("en", max_line_length=42, max_lines=2, max_cue_s=7.0)
    alone = refine_units([target], lang="en", profile=prof)
    surrounded = refine_units(
        [unit("before", 0.0, 1.0), target, unit("after", 40.0, 41.0)],
        lang="en",
        profile=prof,
    )
    assert [(u.surface, u.start, u.end, u.provenance) for u in alone.units] == [
        (u.surface, u.start, u.end, u.provenance) for u in surrounded.units[1:3]
    ]


# ----------------------------------------------------------- ranked evidence


def test_whitespace_beats_punctuation_and_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave.core import subunit

    monkeypatch.setattr(
        subunit,
        "_phrase_pieces",
        lambda *_args: (_ for _ in ()).throw(AssertionError("phrase consulted")),
    )
    result = refine_units(
        [unit("ab, cd", 0.0, 8.0)],
        lang="en",
        profile=profile("en", max_cue_s=7.0),
    )
    assert [item.surface for item in result.units] == ["ab,", "cd"]
    assert {item.provenance for item in result.units} == {"subunit-whitespace"}
    assert result.evidence == {
        kind: 2 if kind == "whitespace" else 0 for kind in EVIDENCE_KINDS
    }


@pytest.mark.parametrize(
    ("language", "surface"),
    [
        ("en", "  alpha  beta \t"),
        ("ja", " 甲 乙 "),
    ],
)
def test_whitespace_evidence_preserves_hostile_spacing_exactly(
    language: str, surface: str
) -> None:
    result = refine_units(
        [unit(surface, 0.0, 8.0)],
        lang=language,
        profile=profile(language, max_cue_s=7.0),
    )
    assert result.refined_parent_count == 1
    assert _join([child.surface for child in result.units], language) == surface
    assert {child.provenance for child in result.units} == {"subunit-whitespace"}


def test_punctuation_beats_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    from voxweave.core import subunit

    monkeypatch.setattr(
        subunit,
        "_phrase_pieces",
        lambda *_args: (_ for _ in ()).throw(AssertionError("phrase consulted")),
    )
    result = refine_units(
        [unit("甲乙、丙丁", 0.0, 6.0)],
        lang="ja",
        profile=profile("ja", max_cue_s=5.0),
    )
    assert [item.surface for item in result.units] == ["甲乙、", "丙丁"]
    assert {item.provenance for item in result.units} == {"subunit-punct"}


def test_phrase_beats_per_character(monkeypatch: pytest.MonkeyPatch) -> None:
    from voxweave.core import subunit

    monkeypatch.setattr(
        subunit,
        "_phrase_pieces",
        lambda _text, _lang: (["これは", "テストです"], None),
    )
    result = refine_units(
        [unit("これはテストです", 0.0, 8.0)],
        lang="ja",
        profile=profile("ja", max_cue_s=7.0),
    )
    assert [item.surface for item in result.units] == ["これは", "テストです"]
    assert {item.provenance for item in result.units} == {"subunit-phrase"}


def test_providerless_language_uses_per_char_and_records_degradation() -> None:
    with degradation_capture(quiet=True) as degraded:
        result = refine_units(
            [unit("甲乙丙丁", 0.0, 8.0)],
            lang="yue",
            profile=profile("yue", max_cue_s=2.5),
        )
    assert [item.surface for item in result.units] == list("甲乙丙丁")
    assert {item.provenance for item in result.units} == {"subunit-per-char"}
    assert result.degraded == ("no-provider:per-char",)
    assert degraded == [
        {"slot": "subunit", "reason": "no-provider:per-char", "count": 1}
    ]


def test_providerless_degradation_ledger_counts_each_refined_parent() -> None:
    with degradation_capture(quiet=True) as degraded:
        result = refine_units(
            [unit("甲乙", 0.0, 8.0), unit("丙丁", 9.0, 17.0, index=1)],
            lang="yue",
            profile=profile("yue", max_cue_s=2.5),
        )
    assert result.degraded == ("no-provider:per-char",)
    assert degraded == [
        {"slot": "subunit", "reason": "no-provider:per-char", "count": 2}
    ]


def test_evidence_vocabulary_and_ranking_are_closed() -> None:
    assert EVIDENCE_RANKING == ("whitespace", "punct", "phrase", "per-char")
    assert EVIDENCE_KINDS == tuple(sorted(EVIDENCE_RANKING))


# ------------------------------------------------ times, identity, purity, N9


def test_proportional_times_are_the_hand_derived_character_load_golden() -> None:
    # "ab" and "cd" each carry two non-space characters, so the split of the
    # eight-second parent is exactly [10,14] + [14,18].
    source = unit("ab cd", 10.0, 18.0, index=9)
    result = refine_units([source], lang="en", profile=profile("en", max_cue_s=7.0))
    assert [(item.start, item.end) for item in result.units] == [
        (10.0, 14.0),
        (14.0, 18.0),
    ]
    assert [item.id for item in result.units] == ["u0", "u1"]
    assert result.origin == (0, 0)
    assert result.refined_parent_count == 1
    assert result.minted == 2


@pytest.mark.parametrize(("start", "end"), [(None, None), (0.0, None), (None, 8.0)])
def test_missing_parent_span_stays_missing_on_every_child(
    start: float | None, end: float | None
) -> None:
    result = refine_units(
        [unit("ab cd", start, end)],
        lang="en",
        profile=profile("en", max_line_length=4),
    )
    assert [(item.start, item.end) for item in result.units] == [
        (None, None),
        (None, None),
    ]


def test_refinement_is_deterministic_and_does_not_mutate_inputs() -> None:
    original = [
        unit("fine", 0.0, 1.0, index=0),
        unit("ab cd", 2.0, 10.0, index=1),
    ]
    pristine = copy.deepcopy(original)
    first = refine_units(original, lang="en", profile=profile("en"))
    second = refine_units(copy.deepcopy(original), lang="en", profile=profile("en"))
    assert first == second
    assert original == pristine
    assert first.origin == (0, 1, 1)
    assert [item.id for item in first.units] == ["u0", "u1", "u2"]


@pytest.mark.parametrize(
    ("language", "surface"),
    [
        ("en", "alpha beta gamma"),
        ("ja", "これは、テストです"),
        ("zh", "甲乙丙，丁戊己"),
        ("yue", "甲乙丙丁"),
    ],
)
def test_exact_language_join_conservation(language: str, surface: str) -> None:
    source = [unit(surface, 0.0, 12.0)]
    result = refine_units(
        source,
        lang=language,
        profile=profile(language, max_line_length=4, max_cue_s=3.0),
    )
    assert _join([item.surface for item in result.units], language) == surface
    assert_refinement_conserved(source, result.units, result.origin, lang=language)


def test_conservation_guard_rejects_a_spec_violating_stream() -> None:
    original = [unit("甲乙", 0.0, 2.0)]
    hostile = [
        SourceUnit("u0", "甲", 0.0, 1.0),
        SourceUnit("u1", "丙", 1.0, 2.0),
    ]
    with pytest.raises(RefinementConservationError, match="character stream"):
        assert_refinement_conserved(original, hostile, (0, 0), lang="ja")


def test_randomized_refinement_preserves_text_ownership_and_parent_envelopes() -> None:
    """Adversarial N9 sweep over both join policies and irregular parent spans."""
    rng = random.Random(0x5A2)
    alphabets = {
        "en": ("alpha", "b", "cc", "delta", "e"),
        "yue": tuple("甲乙丙丁戊己庚辛壬癸"),
    }
    for trial in range(64):
        language = "en" if trial % 2 == 0 else "yue"
        parents: list[SourceUnit] = []
        cursor = rng.uniform(0.0, 5.0)
        for parent_index in range(rng.randint(1, 5)):
            count = rng.randint(2, 12)
            tokens = [rng.choice(alphabets[language]) for _ in range(count)]
            surface = (" " if language == "en" else "").join(tokens)
            duration = rng.choice((0.25, 1.0, 2.5, 7.5, 12.0))
            parents.append(unit(surface, cursor, cursor + duration, index=parent_index))
            cursor += duration + rng.uniform(0.0, 0.75)

        prof = profile(
            language,
            max_line_length=rng.randint(2, 10),
            max_lines=1,
            max_cue_s=rng.choice((0.0, 2.0, 7.0)),
        )
        with degradation_capture(quiet=True):
            result = refine_units(parents, lang=language, profile=prof)
            replay = refine_units(copy.deepcopy(parents), lang=language, profile=prof)

        assert result == replay
        assert_refinement_conserved(parents, result.units, result.origin, lang=language)
        for parent_index, parent in enumerate(parents):
            children = [
                child
                for child, owner in zip(result.units, result.origin)
                if owner == parent_index
            ]
            assert children
            if len(children) > 1:
                assert children[0].start == parent.start
                assert children[-1].end == parent.end
                assert all(
                    left.end == right.start
                    for left, right in zip(children, children[1:])
                )
                assert all(
                    child.start is not None
                    and child.end is not None
                    and child.start <= child.end
                    for child in children
                )


def test_refine_document_remints_a_detached_shadow_copy() -> None:
    source = document(
        [unit("fine", 0.0, 1.0), unit("ab cd", 2.0, 10.0, index=1)],
        profile("en"),
    )
    shadow, split = refine_document(source)
    assert shadow is not source
    assert shadow.units is not source.units
    assert shadow.manifest is not source.manifest
    assert shadow.text == source.text
    assert tuple(shadow.units) == split.units
    shadow.manifest["mutated"] = True
    assert "mutated" not in source.manifest


# ----------------------------------------------------- provenance anchor fold


def test_speech_span_units_uses_endpoint_validity_not_interior_membership() -> None:
    derived = "subunit-phrase"
    prefix = [unit("d", 0.0, 1.0, provenance=derived), unit("a", 1.0, 2.0, index=1)]
    suffix = [unit("a", 0.0, 1.0), unit("d", 1.0, 2.0, index=1, provenance=derived)]
    middle = [
        unit("a", 0.0, 1.0),
        unit("d", 1.0, 2.0, index=1, provenance=derived),
        unit("b", 2.0, 3.0, index=2),
    ]
    assert speech_span_units(prefix) == (None, 2.0)
    assert speech_span_units(suffix) == (0.0, None)
    assert speech_span_units(middle) == (0.0, 3.0)


def test_speech_span_units_rejects_aligner_ghost_endpoints() -> None:
    assert speech_span_units([unit("ghost", None, None)]) == (None, None)
    assert speech_span_units(
        [unit("ghost", None, 1.0), unit("ok", 1.0, 2.0, index=1)]
    ) == (None, 2.0)
    assert speech_span_units(
        [unit("ok", 0.0, 1.0), unit("ghost", 1.0, None, index=1)]
    ) == (0.0, None)


def test_materializer_applies_provenance_per_selected_cue_endpoint() -> None:
    units = [
        unit("甲", 0.0, 1.0, provenance="subunit-per-char"),
        unit("乙", 1.0, 2.0, index=1),
        unit("丙", 2.0, 3.0, index=2, provenance="subunit-per-char"),
    ]
    atoms = [
        atom(0, "甲", 0.0, 1.0, 0, 1),
        atom(1, "乙", 1.0, 2.0, 1, 2),
        atom(2, "丙", 2.0, 3.0, 2, 3),
    ]
    cues = materialize_cues(
        [edge(0, 2, atoms), edge(2, 3, atoms)], atoms, "ja", units=units
    )
    assert [(cue["speech_start"], cue["speech_end"]) for cue in cues] == [
        (None, 2.0),
        (None, None),
    ]


def test_all_aligner_materialization_keeps_the_legacy_fold_bit_exact() -> None:
    units = [unit("a", 0.0, 1.0)]
    atoms = [atom(0, "a", 0.0, 1.0, 0, 1)]
    selected = edge(0, 1, atoms)
    # A hostile edge proves this takes the old edge fold, not a numerically
    # similar recomputation from SourceUnit.
    selected = Edge(
        **{
            **selected.__dict__,
            "span_start": 0.125,
            "span_end": 0.875,
        }
    )
    legacy = materialize_cues([selected], atoms, "en")
    aware = materialize_cues([selected], atoms, "en", units=units)
    assert aware == legacy
    assert aware[0]["speech_start"] == 0.125
    assert aware[0]["speech_end"] == 0.875


def test_all_aligner_random_partitions_match_legacy_bit_exactly() -> None:
    rng = random.Random(0xA11)
    for _trial in range(64):
        count = rng.randint(1, 14)
        units: list[SourceUnit] = []
        atoms: list[LatticeAtom] = []
        for index in range(count):
            ghost = rng.random() < 0.2
            start = None if ghost else float(index) + rng.random() / 10
            end = None if ghost else float(index + 1) + rng.random() / 10
            units.append(unit(chr(97 + index % 26), start, end, index=index))
            atoms.append(atom(index, units[-1].surface, start, end, index, index + 1))

        cuts = [index for index in range(1, count) if rng.random() < 0.35]
        bounds = (0, *cuts, count)
        edges = [edge(start, end, atoms) for start, end in zip(bounds, bounds[1:])]
        legacy = materialize_cues(edges, atoms, "en", fallback_start=3.25)
        aware = materialize_cues(edges, atoms, "en", fallback_start=3.25, units=units)
        assert aware == legacy


def test_optimizer_authority_rematerializes_and_seals_provenance() -> None:
    """W1's typed factory must carry W2 anchors without opening a seal gap."""
    from voxweave.core.authority import AuthorityLedger, SealBroken
    from voxweave.core.finalizer import (
        phase1_from_optimizer_selection,
        register_optimizer_selection,
    )

    prof = profile("en", max_cue_s=7.0)
    shadow, split = refine_document(document([unit("ab cd", 0.0, 8.0)], prof))
    solution = optimize_document(shadow, subunit_split=split)
    assert len([cue for item in solution.solutions for cue in item.cues]) == 2

    ledger = AuthorityLedger()
    registered = register_optimizer_selection(solution, ledger=ledger)
    stream = phase1_from_optimizer_selection(
        registered,
        ledger=ledger,
        row_id="delivery_finalizer/v2",
        evaluation_id="subunit-rematerialization",
    )
    assert [(cue.speech_start, cue.speech_end) for cue in stream.cues] == [
        (None, None),
        (None, None),
    ]
    assert [
        (report.kind, report.evidence["side"])
        for cue in stream.cues
        for report in cue.reports
        if report.kind == "fabricated-time"
    ] == [
        ("fabricated-time", "start"),
        ("fabricated-time", "end"),
        ("fabricated-time", "start"),
        ("fabricated-time", "end"),
    ]

    # A second authority isolates the negative case from the single-use success
    # above.  Provenance is materializer input, so changing it after registration
    # must break the selection seal before phase 1 can be minted.
    second_ledger = AuthorityLedger()
    second = register_optimizer_selection(solution, ledger=second_ledger)
    first = second.document.units[0]
    second.document.units[0] = SourceUnit(
        id=first.id,
        surface=first.surface,
        start=first.start,
        end=first.end,
        provenance="aligner",
        confidence=first.confidence,
    )
    with pytest.raises(SealBroken):
        phase1_from_optimizer_selection(
            second,
            ledger=second_ledger,
            row_id="delivery_finalizer/v2",
            evaluation_id="subunit-seal-negative",
        )


# --------------------------------------------- Probe A / artifact / fixtures


def test_probe_a_duration_trigger_makes_held_chain_waiver_unreachable() -> None:
    # LAW Probe A: 32 cells <= ja's 36-cell line, but 9 s > 7 s.  The comma is
    # real punctuation evidence and yields two 8-character, 4.5-second pieces.
    prof = profile("ja")
    source = document([unit("これはとても、ながいぶんです", 0.0, 9.0)], prof)
    shadow, split = refine_document(source)
    assert split.refined_parent_count == 1
    solution = optimize_document(shadow, subunit_split=split)
    assert all(item.adopted is None for item in solution.solutions)
    assert all(not item.waivers for item in solution.solutions)
    assert solution.artifact["totals"]["fallback_intervals"] == 0
    assert solution.artifact["totals"]["optimized_unit_ratio"] == 1.0
    assert solution.subunit_split is split
    assert solution.artifact["subunit_split"] == split.to_dict()
    assert solution.artifact["subunit_split"]["origin"] == [0, 0]

    from voxweave.core.authority import AuthorityLedger
    from voxweave.core.finalizer import (
        FinalizeEvidence,
        FinalizePolicy,
        finalize,
        phase1_from_optimizer_selection,
        register_optimizer_selection,
    )

    ledger = AuthorityLedger()
    registered = register_optimizer_selection(solution, ledger=ledger)
    phase1 = phase1_from_optimizer_selection(
        registered,
        ledger=ledger,
        row_id="delivery_finalizer/v2",
        evaluation_id="probe-a",
    )
    delivered = finalize(
        phase1,
        profile=prof,
        evidence=FinalizeEvidence(),
        policy=FinalizePolicy(),
    )
    assert delivered.report.waivers == ()


def test_unrefined_infeasible_shared_unit_is_tagged_coarse_caused() -> None:
    prof = profile("ja", max_line_length=4, max_lines=1, max_cue_s=7.0)
    coarse = document([unit("甲乙丙丁戊己庚辛", 0.0, 4.0)], prof)
    solution = optimize_document(coarse)
    assert any(item.lattice.infeasible is not None for item in solution.solutions)
    assert solution.artifact["intervals"][0]["coarse_caused"] is True
    assert solution.artifact["totals"]["coarse_caused_intervals"] == 1
    assert solution.artifact["coverage"]["coarse_caused_intervals"] == 1


def _tracked_profile_from_case(case: dict[str, object]) -> DisplayProfile:
    capture = case["capture"]
    assert isinstance(capture, dict)
    config = capture["config"]
    assert isinstance(config, dict)
    gaps = config["gap_thresholds"]
    assert isinstance(gaps, dict)
    return DisplayProfile(
        language=str(case["language"]),
        max_line_length=int(config["max_line_length"]),
        max_lines=int(config["max_lines"]),
        clause_ms=float(gaps["clause_ms"]),
        vad_skip_ms=float(gaps["vad_skip_ms"]),
        offline_ms=float(gaps["offline_ms"]),
        min_cue_s=float(config["min_cue_s"]),
        max_cue_s=float(config["max_cue_s"]),
        glue_gap_s=float(config["glue_gap_s"]),
        cps=float(config["cps"]),
        lag_out_s=float(config["lag_out_s"]),
        shot_snap_s=float(config["shot_snap_s"]),
    )


def _tracked_units_from_case(case: dict[str, object]) -> list[SourceUnit]:
    rows = case["word_segments"]
    assert isinstance(rows, list)
    out: list[SourceUnit] = []
    for row in rows:
        assert isinstance(row, dict)
        start = row.get("start")
        end = row.get("end")
        out.append(
            SourceUnit(
                id=str(row["id"]),
                surface=str(row["text"]),
                start=None if start is None else float(start),
                end=None if end is None else float(end),
            )
        )
    return out


_COARSE_BREAKERS = frozenset("。！？、，,.!?；;：:")


def _sentence_merged_document(
    fixture: dict[str, object],
) -> tuple[dict[str, object], SegDocument]:
    """Reproduce ``probe_prop_error.py`` with a language-correct text join."""
    relative = Path(str(fixture["source_case"]))
    cases_root = COARSE_CORPUS.parent.resolve()
    source_path = (COARSE_CORPUS.parent / relative).resolve()
    assert source_path.is_relative_to(cases_root)
    assert source_path.is_file()
    source_case = json.loads(source_path.read_text(encoding="utf-8"))

    overrides = fixture["profile_overrides"]
    assert isinstance(overrides, dict)
    prof = replace(_tracked_profile_from_case(source_case), **overrides)
    fine = _tracked_units_from_case(source_case)
    max_block = fixture["max_block_units"]
    assert isinstance(max_block, int) and not isinstance(max_block, bool)
    assert max_block > 0

    blocks: list[list[SourceUnit]] = []
    pending: list[SourceUnit] = []
    for source_unit in fine:
        pending.append(source_unit)
        if (source_unit.surface and source_unit.surface[-1] in _COARSE_BREAKERS) or len(
            pending
        ) >= max_block:
            blocks.append(pending)
            pending = []
    if pending:
        blocks.append(pending)

    coarse = [
        SourceUnit(
            id=f"u{index}",
            surface=_join([item.surface for item in block], prof.language),
            start=block[0].start,
            end=block[-1].end,
        )
        for index, block in enumerate(blocks)
    ]
    source = SegDocument(
        language=prof.language,
        units=coarse,
        profile=prof,
        vad_speech=[
            tuple(map(float, span)) for span in source_case.get("vad_speech", [])
        ],
        shot_changes=[float(value) for value in source_case.get("shot_changes", [])],
        sing_spans=[
            tuple(map(float, span)) for span in source_case.get("sing_spans", [])
        ],
        # Speaker projection is W3.  W2 preserves only evidence that the frozen
        # SegDocument contract can consume without pre-attribution.
        speaker_turns=None,
        manifest={
            "fixture": str(fixture["id"]),
            "source_case": str(fixture["source_case"]),
        },
        text=_join([item.surface for item in coarse], prof.language),
    )
    return source_case, source


def _truth_starts(rows: list[dict[str, object]]) -> list[float]:
    from voxweave.core.smart_split import _display_chars

    surfaces = [str(row["text"]) for row in rows]
    shown = _display_chars(surfaces)
    return [
        float(row["start"]) for row, chars in zip(rows, shown) for _character in chars
    ]


def _cue_start_errors(cues: list[dict[str, object]], truth: list[float]) -> list[float]:
    from voxweave.core.partition_check import normalize_text

    offset = 0
    errors: list[float] = []
    for cue in cues:
        # The first cue is fixed to the document's first source time and is not
        # an interior boundary.  ``probe_prop_error.py`` excludes it for the
        # same reason; the fixture measures only choices refinement can move.
        if 0 < offset < len(truth):
            errors.append(abs(float(cue["start"]) - truth[offset]))
        offset += len("".join(normalize_text(str(cue["text"])).split()))
    return errors


def test_tracked_corpus_trigger_is_code_absent_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B7 pin: all frozen cases are exact identity rows and never consult a splitter."""
    from voxweave.core import subunit

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("tracked-corpus refinement trigger fired")

    monkeypatch.setattr(subunit, "_split_parent", forbidden)
    registry = json.loads(TRACKED_CORPUS.read_text(encoding="utf-8"))
    for relative in registry["cases"]:
        path = TRACKED_CORPUS.parent / relative
        case = json.loads(path.read_text(encoding="utf-8"))
        units = _tracked_units_from_case(case)
        result = refine_units(
            units,
            lang=str(case["language"]),
            profile=_tracked_profile_from_case(case),
        )
        assert result.units == tuple(units), case["id"]
        assert result.origin == tuple(range(len(units))), case["id"]
        assert result.refined_parent_count == 0, case["id"]
        assert result.minted == 0, case["id"]
        assert not any(result.evidence.values()), case["id"]


def test_coarse_fixture_family_schema_and_per_case_gates() -> None:
    corpus = json.loads(COARSE_CORPUS.read_text(encoding="utf-8"))
    tracked = json.loads(TRACKED_CORPUS.read_text(encoding="utf-8"))
    assert set(corpus) == {"schema_version", "description", "cases"}
    assert corpus["schema_version"] == 1
    assert isinstance(corpus["description"], str) and corpus["description"]
    assert COARSE_CORPUS.name not in tracked["cases"]
    assert {case["variant"] for case in corpus["cases"]} == {
        "both",
        "duration",
        "mixed",
        "per-char",
        "width",
    }

    for case in corpus["cases"]:
        assert set(case) == {
            "id",
            "variant",
            "source_case",
            "max_block_units",
            "profile_overrides",
        }
        assert str(case["id"]).startswith("coarse-")
        assert case["source_case"] in tracked["cases"]
        source_case, source = _sentence_merged_document(case)
        prof = source.profile
        coarse_units = source.units
        pristine_source = copy.deepcopy(source)
        trigger_pairs = [
            (
                _vis_width(parent.surface)
                > _line_budget_width(prof.max_line_length, prof.language),
                prof.max_cue_s > 0
                and parent.start is not None
                and parent.end is not None
                and parent.end - parent.start > prof.max_cue_s + CAP_EPS_S,
            )
            for parent in coarse_units
        ]
        variant = case["variant"]
        if variant == "width":
            assert any(width and not duration for width, duration in trigger_pairs)
        elif variant == "duration":
            assert any(duration and not width for width, duration in trigger_pairs)
        elif variant == "both":
            assert any(width and duration for width, duration in trigger_pairs)
        elif variant == "per-char":
            assert any(width or duration for width, duration in trigger_pairs)
        elif variant == "mixed":
            assert any(width or duration for width, duration in trigger_pairs)
            assert any(not width and not duration for width, duration in trigger_pairs)

        shadow, split = refine_document(source)
        assert source == pristine_source, case["id"]
        assert split.refined_parent_count > 0, case["id"]
        if variant == "per-char":
            assert split.evidence["per-char"] > 0, case["id"]
        solution = optimize_document(shadow, subunit_split=split)
        totals = solution.artifact["totals"]
        assert totals["fallback_intervals"] == 0, case["id"]
        assert totals["optimized_unit_ratio"] == 1.0, case["id"]
        assert totals["coarse_caused_intervals"] == 0, case["id"]
        assert sum(bool(item.adopted) for item in solution.solutions) == 0, case["id"]
        assert solution.artifact["validator"]["raw"]["exit_driving"] == 0, case["id"]

        cues = [cue for item in solution.solutions for cue in item.cues]
        fine_rows = source_case["word_segments"]
        assert isinstance(fine_rows, list)
        errors = _cue_start_errors(cues, _truth_starts(fine_rows))
        assert errors, case["id"]
        ordered = sorted(errors)
        p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
        assert p90 <= 0.75, (case["id"], p90)
        assert max(errors) <= 2.0, (case["id"], max(errors))


def test_coarse_candidate_spans_match_canonical_legality_both_directions() -> None:
    """N14: direct refined-unit spans are admitted iff FinalText is legal."""
    from voxweave.core.canonical_text import canonical_legal, canonical_text

    corpus = json.loads(COARSE_CORPUS.read_text(encoding="utf-8"))
    checked = 0
    for fixture in corpus["cases"]:
        _source_case, source = _sentence_merged_document(fixture)
        shadow, _split = refine_document(source)
        if shadow.language == "en":
            continue
        limit = band_atoms(shadow.profile) + 2
        for start in range(len(shadow.units)):
            packer = IncrementalPacker(
                shadow.language,
                shadow.profile.max_line_length,
                shadow.profile.max_lines,
            )
            for end in range(start + 1, min(len(shadow.units), start + limit) + 1):
                owned = shadow.units[start:end]
                measure = packer.extend(owned[-1].surface)
                raw = _join([item.surface for item in owned], shadow.language)
                final = canonical_text(
                    [
                        {"text": item.surface, "start": item.start, "end": item.end}
                        for item in owned
                    ],
                    fallback_text=raw,
                    lang=shadow.language,
                    profile=shadow.profile,
                    expected_footprint=raw,
                )
                assert measure.fits is canonical_legal(final, shadow.profile), (
                    fixture["id"],
                    start,
                    end,
                    measure,
                    final,
                )
                checked += 1
    assert checked > 0


def test_coarse_finalizer_stage_has_no_exit_driving_violations() -> None:
    """N5: every refined coarse partition remains legal after finalization."""
    from voxweave.core.authority import AuthorityLedger
    from voxweave.core.finalizer import (
        FinalizeEvidence,
        FinalizePolicy,
        finalize,
        phase1_from_optimizer_selection,
        register_optimizer_selection,
    )
    from voxweave.core.partition_check import check_partition

    corpus = json.loads(COARSE_CORPUS.read_text(encoding="utf-8"))
    line_capacity = 0
    failures: dict[str, list[dict[str, object]]] = {}
    for fixture in corpus["cases"]:
        _source_case, source = _sentence_merged_document(fixture)
        shadow, split = refine_document(source)
        solution = optimize_document(shadow, subunit_split=split)
        ledger = AuthorityLedger()
        authority = register_optimizer_selection(solution, ledger=ledger)
        phase1 = phase1_from_optimizer_selection(
            authority,
            ledger=ledger,
            row_id="delivery_finalizer/v2",
            evaluation_id=f"coarse-n5:{fixture['id']}",
        )
        delivered = finalize(
            phase1,
            profile=shadow.profile,
            evidence=FinalizeEvidence(
                shots=tuple(shadow.shot_changes or ()),
                sing_spans=tuple(shadow.sing_spans or ()),
            ),
            policy=FinalizePolicy(),
        )
        checked = check_partition(
            _document_partition(solution.solutions, len(shadow.units)),
            delivered.cues,
            units=shadow.units,
            profile=shadow.profile,
            origin="v2",
            stage="finalizer",
            reports=delivered.report.entries,
            waivers={waiver.cue_index: waiver for waiver in delivered.report.waivers},
        )
        line_capacity += sum(
            violation.kind == "line-capacity" for violation in checked.exit_driving
        )
        if checked.exit_driving:
            failures[str(fixture["id"])] = [
                violation.to_dict() for violation in checked.exit_driving
            ]
    assert line_capacity == 0, (line_capacity, sorted(failures))
    assert failures == {}


def test_refinement_metadata_cannot_be_attached_to_another_document() -> None:
    left, split = refine_document(document([unit("ab cd", 0.0, 8.0)], profile("en")))
    with pytest.raises(ValueError, match="requires its audited"):
        optimize_document(left)

    right = document([unit("different", 0.0, 1.0)], profile("en"))
    assert isinstance(split, RefineResult)
    assert left.units != right.units
    with pytest.raises(ValueError, match="does not describe"):
        optimize_document(right, subunit_split=split)


def test_refinement_metadata_is_immutable_and_rejects_forged_accounting() -> None:
    _shadow, split = refine_document(document([unit("ab cd", 0.0, 8.0)], profile("en")))
    with pytest.raises(TypeError):
        split.evidence["whitespace"] = 999  # type: ignore[index]

    forged = dict(split.evidence)
    forged["whitespace"] += 1
    with pytest.raises(RefinementConservationError, match="evidence and minted"):
        RefineResult(
            units=split.units,
            origin=split.origin,
            refined_parent_count=split.refined_parent_count,
            minted=split.minted,
            evidence=forged,
            degraded=split.degraded,
        )

    rebalanced = dict(split.evidence)
    rebalanced["whitespace"] -= 1
    rebalanced["phrase"] += 1
    with pytest.raises(RefinementConservationError, match="unit provenance"):
        RefineResult(
            units=split.units,
            origin=split.origin,
            refined_parent_count=split.refined_parent_count,
            minted=split.minted,
            evidence=rebalanced,
            degraded=split.degraded,
        )
