"""Contract tests for the P5 shadow harness (``calib_segmentation.py shadow``).

The shadow lane measures BoundaryOptimizer v2 beside the v1 answer that actually
ships. Nothing it produces may reach the quality report, and nothing it reports
may be softer than the spec: these tests pin the parts a future edit could
quietly relax --

* ``SHADOW_GATES`` still has the tracked baseline's configured modes (three
  blocking, ``forbidden_end_rate`` warning), while the shared evaluator applies
  the same per-language sample promotion as the production quality report;
* the N4/N5 coverage, N1/N3 non-inferiority, and P1-P3 perturbation verdicts
  fold onto the shared 0/1/2 exits with "invalid" outranking "failed";
* a full run over a small synthetic corpus produces the complete lane/row matrix
  and a schema-valid aggregate;
* the perturbation runner reports both natural and pinned recomputations, freezes
  the barrier set in the pinned lane, and accounts for the influence cell;
* the overlapping-``adopted_v1`` duplicate-cue annotation is never read as
  conservation evidence.

The corpus is generated here rather than read from ``calibration/`` on purpose:
these are contract tests for the harness, and they must not start failing
because somebody recorded a new golden case.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
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


calib = _load_script("calib_segmentation")
capture = _load_script("capture_scenario")
cc = calib.cc

TRACKED_BASELINE = REPO_ROOT / "calibration" / "segmentation" / "baseline.json"


# --------------------------------------------------------------------------- #
# A minimal synthetic corpus
# --------------------------------------------------------------------------- #
#
# Two minimal documents carry every assertion below; the registry schema pins
# ``required_counts`` to the tracked shape (zh 7, ja 7, en 6) with ``const``, so
# they are replicated under distinct ids to make a loadable corpus rather than
# relaxed into one the loader would reject. Every test that does not need the
# whole corpus selects the cases it needs by id.

_EN_WORDS = (
    "the meeting starts at nine and the agenda covers the migration plan "
    "before we review the budget and then the team will walk through the "
    "remaining questions about the rollout schedule and the support rota"
).split()

_ZH_TEXT = (
    "今天天气很好我们一起去公园散步然后回家吃饭休息一下"
    "明天还要早起上班所以今晚不能睡得太晚这样才能保持精神"
)

_JA_TEXT = (
    "今日はとてもいい天気ですから公園を散歩してから家に帰ります"
    "明日も早く起きて仕事に行くので今夜は早めに休むつもりです"
)


#: Silence inserted after these word indexes of the en fixture, in seconds. The
#: 1.081 s one sits just above ``vad_skip_ms + 50`` so a +120 ms nudge to the
#: word before it deletes a robust-silence barrier -- which is exactly the event
#: the pinned perturbation lane has to suppress and the natural lane has to see.
_EN_LONG_GAPS = {14: 1.081, 20: 1.6}


def _spaced_units(
    words: list[str], *, dur: float = 0.45, gap: float = 0.05
) -> list[dict]:
    out: list[dict] = []
    t = 0.0
    for i, word in enumerate(words):
        out.append({"id": f"u{i}", "text": word, "start": t, "end": t + dur})
        t += dur + _EN_LONG_GAPS.get(i, gap)
    return out


def _char_units(text: str, *, dur: float = 0.22, gap: float = 0.03) -> list[dict]:
    out: list[dict] = []
    t = 0.0
    for i, ch in enumerate(text):
        out.append({"id": f"u{i}", "text": ch, "start": t, "end": t + dur})
        t += dur + gap
    return out


def _make_case(case_id: str, language: str, units: list[dict]) -> dict:
    """A schema-valid golden case, configured by the real capture path."""
    latest = max(u["end"] for u in units)
    dependency_versions = dict(calib.dependency_versions())
    dependency_versions["python"] = platform.python_version()
    return {
        "schema_version": 1,
        "id": case_id,
        "language": language,
        "description": f"synthetic {language} shadow fixture",
        "tags": ["synthetic"],
        "license": {
            "redistributable": True,
            "source_class": "synthetic-from-consented-speech",
            "spdx": None,
            "attribution": None,
        },
        "capture": {
            "voxweave_commit": "0" * 40,
            "source_digest": "0" * 64,
            "window_duration_s": round(latest + 0.5, 6),
            "dependency_versions": dependency_versions,
            "config": capture.segmentation_config(language),
            "missing_inputs": [],
        },
        "word_segments": units,
        "vad_speech": [[units[0]["start"], latest]],
        "shot_changes": [],
        "sing_spans": [],
        "speaker_turns": [],
    }


def _units_for(language: str) -> list[dict]:
    if language == "en":
        return _spaced_units(_EN_WORDS)
    return _char_units(_ZH_TEXT if language == "zh" else _JA_TEXT)


def _write_corpus(root: Path) -> Path:
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    relpaths: list[str] = []
    for language, count in (("zh", 7), ("ja", 7), ("en", 6)):
        for ordinal in range(1, count + 1):
            doc = _make_case(
                f"{language}-{ordinal:02d}", language, _units_for(language)
            )
            path = cases_dir / f"{doc['id']}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            relpaths.append(f"cases/{path.name}")
    registry = {
        "schema_version": 1,
        "metric_definition_version": calib.METRIC_DEFINITION_VERSION,
        "description": "synthetic corpus for the shadow harness tests",
        "cases": relpaths,
        "required_counts": {"zh": 7, "ja": 7, "en": 6},
        "required_tags": ["synthetic"],
    }
    registry_path = root / "corpus.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    return registry_path


def _cases(corpus: Any, *ids: str) -> list[Any]:
    wanted = set(ids)
    return [case for case in corpus.cases if case.id in wanted]


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path / "segmentation")


def _run_cli(argv: list[str]) -> int:
    try:
        return calib.main(argv)
    except cc.CalibrationError:
        return cc.EXIT_INVALID


# --------------------------------------------------------------------------- #
# The frozen gate table
# --------------------------------------------------------------------------- #


def test_shadow_gates_carry_the_tracked_baseline_modes() -> None:
    """The frozen literal must still describe the gate the corpus is armed with.

    ``SHADOW_GATES`` is deliberately not read from the baseline at run time -- a
    baseline edit would otherwise redefine the P5 acceptance criterion silently.
    This is the seam that makes such an edit visible: it fails here, in a test
    whose message says which mode moved.
    """
    baseline = json.loads(TRACKED_BASELINE.read_text(encoding="utf-8"))
    tracked = {metric: gate["mode"] for metric, gate in baseline["gates"].items()}
    frozen = {metric: gate["mode"] for metric, gate in calib.SHADOW_GATES.items()}
    assert frozen == tracked


def test_shadow_gates_are_three_blocking_and_one_warning() -> None:
    modes = {m: g["mode"] for m, g in calib.SHADOW_GATES.items()}
    assert set(calib.SHADOW_GATES) == set(calib.METRICS)
    assert sorted(m for m, mode in modes.items() if mode == "blocking") == [
        "cps_p90",
        "len_break_mid_phrase_rate",
        "over_7s_rate",
    ]
    assert modes["forbidden_end_rate"] == "warning"
    forbidden = calib.SHADOW_GATES["forbidden_end_rate"]
    assert forbidden["absolute_max"] is None
    assert forbidden["absolute_tolerance"] == calib.FORBIDDEN_END_BAD_SLACK
    assert forbidden["relative_tolerance"] == 0.0
    assert forbidden["min_samples"] == 100
    assert "forbidden_end_rate" in calib.COUNT_METRICS


def test_harness_names_match_the_hook_it_reads() -> None:
    """The flag and both lane names are literals here; a rename must be caught."""
    from voxweave import pipeline

    assert calib.SHADOW_ENV == pipeline.SEG_V2_SHADOW_ENV
    assert calib.SHADOW_LANE_CORE == pipeline.SHADOW_LANE_CORE
    assert calib.SHADOW_LANE_DELIVERY == pipeline.SHADOW_LANE_DELIVERY


def test_influence_radius_matches_the_optimizer_constant() -> None:
    """AD-2's radius is declared in two places; they must not drift apart."""
    from voxweave.core.boundary_lattice import INFLUENCE_RADIUS_UNITS

    assert calib.INFLUENCE_RADIUS_UNITS == INFLUENCE_RADIUS_UNITS


def test_ablation_table_names_real_cost_constants() -> None:
    from voxweave.core import boundary_cost as bc

    for term, names in calib.ABLATION_WEIGHTS.items():
        assert names, term
        for name in names:
            assert isinstance(getattr(bc, name), float), f"{term}: {name}"
    assert calib.ABLATION_TERM_PAUSE not in calib.ABLATION_WEIGHTS
    assert hasattr(bc, "pause_cut_cost")


def test_ablation_restores_every_weight_it_zeroed() -> None:
    from voxweave.core import boundary_cost as bc

    before = {
        name: getattr(bc, name)
        for names in calib.ABLATION_WEIGHTS.values()
        for name in names
    }
    pause = bc.pause_cut_cost
    for term in calib.ABLATION_TERMS:
        with calib._ablated_weight(term):
            pass
    after = {
        name: getattr(bc, name)
        for names in calib.ABLATION_WEIGHTS.values()
        for name in names
    }
    assert after == before
    assert bc.pause_cut_cost is pause


def test_forced_shadow_restores_the_operator_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(calib.SHADOW_ENV, "operator")
    with calib._forced_shadow(True):
        import os

        assert os.environ[calib.SHADOW_ENV] == "1"
    import os

    assert os.environ[calib.SHADOW_ENV] == "operator"
    monkeypatch.delenv(calib.SHADOW_ENV)
    with calib._forced_shadow(False):
        assert os.environ[calib.SHADOW_ENV] == "0"
    assert calib.SHADOW_ENV not in os.environ


# --------------------------------------------------------------------------- #
# Exit-code mapping
# --------------------------------------------------------------------------- #


def _gate(status: str, mode: str = "blocking") -> dict[str, Any]:
    return {"mode": mode, "status": status, "metric": "cps_p90", "group": "en"}


def test_clean_run_exits_zero() -> None:
    assert calib.shadow_exit_code([_gate("pass")], [], []) == cc.EXIT_OK


def test_blocking_gate_failure_exits_one() -> None:
    assert calib.shadow_exit_code([_gate("fail")], [], []) == cc.EXIT_GATE_FAILED


def test_blocking_thin_denominator_exits_two() -> None:
    assert (
        calib.shadow_exit_code([_gate("insufficient_samples")], [], [])
        == cc.EXIT_INVALID
    )


def test_warning_gate_never_changes_the_exit() -> None:
    results = [_gate("fail", mode="warning"), _gate("insufficient_samples", "warning")]
    assert calib.shadow_exit_code(results, [], []) == cc.EXIT_OK


def test_c13_failure_alone_exits_one() -> None:
    assert calib.shadow_exit_code([_gate("pass")], ["zh-01: fallback"], []) == (
        cc.EXIT_GATE_FAILED
    )


def test_perturbation_failure_alone_exits_one() -> None:
    assert calib.shadow_exit_code([_gate("pass")], [], [{"case": "en-01"}]) == (
        cc.EXIT_GATE_FAILED
    )


def test_invalid_measurement_outranks_a_failed_gate() -> None:
    """A run that could not answer the question may not call anything a regression."""
    results = [_gate("insufficient_samples"), _gate("fail")]
    assert calib.shadow_exit_code(results, ["zh-01: fallback"], []) == cc.EXIT_INVALID


# --------------------------------------------------------------------------- #
# End to end over the synthetic corpus
# --------------------------------------------------------------------------- #


def test_shadow_run_measures_both_lanes_and_both_engines(corpus_path: Path) -> None:
    corpus = calib.load_corpus(corpus_path)
    rows = [
        calib.run_shadow_case(case)
        for case in _cases(corpus, "en-01", "ja-01", "zh-01")
    ]
    assert len(rows) == 3
    for row in rows:
        assert row.artifact["kind"] == "segmentation-shadow"
        assert row.artifact["engine_v2"] == "boundary-optimizer-v2"
        for lane in calib.SHADOW_LANES:
            for engine in calib.SHADOW_LANE_ROWS[lane]:
                result = row.lanes[(lane, engine)]
                assert result.error is None, (lane, engine, result.error)
                assert result.measurement is not None
                assert result.measurement.diagnostics["unmapped_boundaries"] == 0
        # the v1 delivery lane is the stream production actually returns, so it
        # must reproduce the plain quality measurement exactly
        production = calib.measure_case(row.case, calib.replay(row.case))
        v1 = row.lanes[(calib.SHADOW_LANE_DELIVERY, "v1")].measurement
        assert v1 is not None
        assert {k: (r.bad, r.eligible) for k, r in v1.ratios.items()} == {
            k: (r.bad, r.eligible) for k, r in production.ratios.items()
        }


def test_lane_aggregates_are_schema_valid(corpus_path: Path) -> None:
    corpus = calib.load_corpus(corpus_path)
    rows = [
        calib.run_shadow_case(case)
        for case in _cases(corpus, "en-01", "ja-01", "zh-01")
    ]
    for lane in calib.SHADOW_LANES:
        for engine in calib.SHADOW_LANE_ROWS[lane]:
            groups = calib.lane_groups(rows, lane, engine)
            assert calib.shadow_group_errors(groups) == []


def test_shadow_cli_writes_a_report_and_leaves_quality_alone(
    corpus_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shadow.json"
    code = _run_cli(
        [
            "shadow",
            "--corpus",
            str(corpus_path),
            "--json-out",
            str(out),
            "--no-ablation",
        ]
    )
    assert code == cc.EXIT_OK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["kind"] == calib.SHADOW_REPORT_KIND
    assert report["metric_definition_digest"] == cc.canonical_digest(
        report["metric_definition"]
    )
    assert report["schema_version"] == calib.SHADOW_SCHEMA_VERSION == 2
    assert report["gated_lane"] == calib.SHADOW_LANE_FINALIZER
    assert report["gated_row"] == "v2"
    assert set(report["lanes"]) == set(calib.SHADOW_LANES)
    assert report["ablation"] is None
    assert report["perturbation"] is None
    assert "groups" not in report, "shadow numbers must never sit in a quality group"
    assert set(report["features"]) >= {"all", "en", "ja", "zh"}
    for block in report["features"].values():
        assert set(block["vad_state"]) <= set(
            ("absent", "missing-bounds", "silence", "speech-overlap")
        )
    for case in report["cases"]:
        assert set(case["lanes"]) == set(calib.SHADOW_LANES)
        assert case["coverage"]["fallback_intervals"] == 0
        assert case["coverage"]["optimized_unit_ratio"] == 1.0


def test_quality_report_never_learns_about_the_shadow(
    corpus_path: Path, tmp_path: Path
) -> None:
    """The two reports are separate files and the baseline can only see one.

    ``baseline_from_report`` copies ``report["groups"]`` verbatim into the
    tracked baseline, so a v2 number parked in that block would quietly become
    the reference the *v1* gate compares against on the next run.
    """
    quality = tmp_path / "quality.json"
    shadow = tmp_path / "shadow.json"
    assert (
        _run_cli(["evaluate", "--corpus", str(corpus_path), "--json-out", str(quality)])
        == cc.EXIT_OK
    )
    assert (
        _run_cli(
            [
                "shadow",
                "--corpus",
                str(corpus_path),
                "--json-out",
                str(shadow),
                "--no-ablation",
            ]
        )
        == cc.EXIT_OK
    )
    q = json.loads(quality.read_text(encoding="utf-8"))
    s = json.loads(shadow.read_text(encoding="utf-8"))
    assert "shadow" not in q
    assert "lanes" not in q
    assert calib.baseline_from_report(q)["groups"] == q["groups"]
    # the shadow's v1 delivery lane is the very stream the quality report graded
    v1 = s["lanes"][calib.SHADOW_LANE_DELIVERY]["v1"]
    for group in ("all", "en", "ja", "zh"):
        assert v1[group]["cue_count"] == q["groups"][group]["cue_count"]
        assert (
            v1[group]["len_break_mid_phrase_rate"]
            == q["groups"][group]["len_break_mid_phrase_rate"]
        )


def test_check_exit_is_the_verdict_the_report_states(
    corpus_path: Path, tmp_path: Path
) -> None:
    """``--check`` must exit exactly what the written report justifies."""
    out = tmp_path / "shadow.json"
    code = _run_cli(
        [
            "shadow",
            "--corpus",
            str(corpus_path),
            "--json-out",
            str(out),
            "--no-ablation",
            "--check",
        ]
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [r["metric"] for r in report["gate_results"]]
    assert code == calib.shadow_exit_code(
        report["gate_results"], report["coverage"]["failures"], []
    )


def test_partial_run_skips_the_gates(corpus_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "shadow.json"
    code = _run_cli(
        [
            "shadow",
            "--corpus",
            str(corpus_path),
            "--case",
            "en-01",
            "--json-out",
            str(out),
            "--no-ablation",
            "--check",
        ]
    )
    assert code == cc.EXIT_OK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["partial"] is True
    assert report["gate_results"] == []
    assert len(report["cases"]) == 1


def test_ablation_reports_one_row_per_term(corpus_path: Path) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")
    rows = [calib.run_shadow_case(c) for c in case]
    reference = calib.lane_groups(rows, calib.SHADOW_GATED_LANE, "v2")
    table = calib.run_ablation(case, reference)
    assert [row["term"] for row in table["rows"]] == list(calib.ABLATION_TERMS)
    for row in table["rows"]:
        assert row["weights"]
        assert "en" in row["deltas"]
        assert set(row["deltas"]["en"]) == {"cue_count", *calib.METRICS}
    migration = next(r for r in table["rows"] if r["term"] == "migration")
    # W_MIGRATION is already 0.0: zeroing it is the table's own control row.
    assert migration["deltas"]["en"]["cue_count"] == 0


def test_tracked_ablation_domain_is_fixed_recorded_and_three_language() -> None:
    corpus = calib.load_corpus(calib.DEFAULT_CORPUS)
    selected, record = calib.ablation_case_selection(
        corpus.cases, tracked_registry=True
    )
    assert {row["case"] for row in record["excluded_cases"]} == {"ja-02", "ja-03"}
    assert record["candidate_cases"] == [case.id for case in corpus.cases]
    assert record["requested_cases"] == [case.id for case in selected]
    assert {case.language for case in selected} == {"en", "ja", "zh"}
    assert len(selected) == 18
    custom, custom_record = calib.ablation_case_selection(
        corpus.cases, tracked_registry=False
    )
    assert list(custom) == list(corpus.cases)
    assert custom_record["excluded_cases"] == []


def test_ablation_with_zero_requested_evidence_is_invalid() -> None:
    table = calib.run_ablation([], {})
    assert table["complete"] is False
    assert len(table["unknown"]) == len(calib.ABLATION_TERMS)
    assert all(row["complete"] is False for row in table["rows"])
    assert all(row["coverage"]["requested_cases"] == [] for row in table["rows"])
    assert calib.shadow_exit_code([], [], [], [], table["unknown"]) == cc.EXIT_INVALID


def test_missing_n6_validator_block_is_invalid_not_zero() -> None:
    with pytest.raises(cc.CalibrationError, match="missing validator block"):
        calib._speech_truncation_count(None)


# --------------------------------------------------------------------------- #
# Perturbation
# --------------------------------------------------------------------------- #


def test_single_gap_probe_reports_both_lanes_and_pins_the_barriers(
    corpus_path: Path,
) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")[0]
    row = calib.run_shadow_case(case)
    block = calib.run_single_gap_probes(
        row, row.document, magnitudes=[20], max_probes=2
    )
    coverage = block["coverage"]
    assert coverage["candidate_gaps"] == len(case.units) - 1
    assert coverage["exhaustive"] is False  # capped by max_probes
    assert coverage["sample_rate"] == calib.PERTURB_SAMPLE_RATE
    assert set(coverage["near_cliff_by_state"]) <= set(
        coverage["vad_state_denominators"]
    )
    assert coverage["raw_knees"]
    assert set(coverage["knees"]) == {
        "absent",
        "missing-bounds",
        "silence",
        "speech-overlap",
    }
    summary = block["summary"]
    assert summary["probes"] <= 2
    assert summary["failures"] == 0
    assert summary["errors"] == 0
    assert summary["barrier_flips_pinned"] == 0
    assert summary["retained_probes"] == len(block["probes"])
    for probe in block["probes"]:
        assert probe["mode"] == "single_gap"
        assert probe["probe_unit"] == probe["gap_index"] + 1
        assert probe["applied_delta_ms"] != 0.0
        assert set(probe["lanes"]) == {"natural", "pinned"}
        for lane in probe["lanes"].values():
            assert lane["error"] is None
            assert isinstance(lane["barrier_flips"], int)
            assert isinstance(lane["crossed_interval_boundary"], bool)
            assert lane["moved_count"] == len(lane["moved_units"])
            assert set(lane["outside_cell"]) <= set(lane["moved_units"])
        # the pinned lane freezes the exogenous topology by construction
        assert probe["lanes"]["pinned"]["barrier_flips"] == 0


def test_pinned_lane_suppresses_a_barrier_flip_the_natural_lane_sees(
    corpus_path: Path,
) -> None:
    """A nudge that shrinks an atom-edge gap below ``vad_skip_ms + 50`` deletes a
    barrier; the pinned lane must keep the topology and still recompute costs."""
    from voxweave.core.boundary_lattice import build_atom_layer

    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")[0]
    row = calib.run_shadow_case(case)
    layer = build_atom_layer(row.document)
    threshold = row.document.profile.vad_skip_ms + 50.0
    target = None
    for node in range(1, len(layer.atoms)):
        left, right = layer.atoms[node - 1], layer.atoms[node]
        if left.end is None or right.start is None:
            continue
        if threshold <= (right.start - left.end) * 1000.0 < threshold + 200.0:
            target = layer.unit_bound(node) - 1
            break
    if target is None:
        pytest.skip("this fixture has no gap sitting just above the barrier threshold")

    units, applied = calib.perturb_single_gap(case.units, target, 120.0)
    assert applied > 0.0
    base = set(row.core_partition or ())
    natural = calib._cell_report(
        base,
        calib._probe_partition(case, units),
        row.barrier_units,
        (target + 1,),
    )
    with calib._pinned_barriers(row.barrier_units):
        pinned = calib._cell_report(
            base,
            calib._probe_partition(case, units),
            row.barrier_units,
            (target + 1,),
        )
    assert natural["barrier_flips"] >= 1
    assert pinned["barrier_flips"] == 0
    # and the patch is undone
    assert calib.barrier_unit_ids(row.document) == row.barrier_units


def test_influence_cell_accounting_is_anchored_on_probe_and_flips() -> None:
    radius = calib.INFLUENCE_RADIUS_UNITS
    probe = {
        "barrier_units": (900,),
        "cue_count": 3,
        "fallback_intervals": 0,
        "partition": {10, 200, 500, 940},
    }
    report = calib._cell_report({10, 200}, probe, (), (200,))
    # 500 sits outside the probe cell and no barrier flipped near it; 940 sits
    # inside the cell around the barrier that appeared at 900.
    assert report["moved_units"] == [500, 940]
    assert report["barrier_flips"] == 1
    assert report["flipped_barrier_units"] == [900]
    assert report["outside_cell"] == [500]
    assert report["crossed_interval_boundary"] is True
    assert report["influence_radius_units"] == min(abs(500 - 200), abs(500 - 900))

    near = {**probe, "partition": {10, 200, 200 + radius}}
    clean = calib._cell_report({10, 200}, near, (900,), (200,))
    assert clean["outside_cell"] == []
    assert clean["crossed_interval_boundary"] is False


def _probe_row(index: int, *, moved: list[int], outside: list[int]) -> dict[str, Any]:
    lane = {
        "barrier_flips": 0,
        "crossed_interval_boundary": bool(outside),
        "cue_count": 3,
        "error": None,
        "fallback_intervals": 0,
        "flipped_barrier_units": [],
        "influence_radius_units": max(moved, default=0),
        "moved_count": len(moved),
        "moved_units": moved,
        "outside_cell": outside,
    }
    return {
        "gap_index": index,
        "magnitude_ms": 20,
        "sign": 1,
        "lanes": {"natural": lane, "pinned": dict(lane)},
    }


def test_every_failing_probe_survives_the_report_trim() -> None:
    """A trimmed probe list may hide a quiet probe; it may never hide a failure."""
    quiet = [_probe_row(i, moved=[], outside=[]) for i in range(200)]
    movers = [_probe_row(500 + i, moved=[1, 2], outside=[]) for i in range(50)]
    failures = [_probe_row(900 + i, moved=[1], outside=[1]) for i in range(3)]
    retained, summary = calib.retain_probes([*quiet, *movers, *failures])
    assert summary["probes"] == 253
    assert summary["failures"] == 3
    assert summary["with_movement"] == 50
    assert len(retained) == 3 + calib.OFFENDER_LIMIT
    assert summary["retained_probes"] == len(retained)
    kept = {row["gap_index"] for row in retained}
    assert {900, 901, 902} <= kept
    assert not kept & {row["gap_index"] for row in quiet}


def test_probe_trim_ranks_the_biggest_movers_first() -> None:
    rows = [
        _probe_row(1, moved=[1], outside=[]),
        _probe_row(2, moved=[1, 2, 3], outside=[]),
        _probe_row(3, moved=[1, 2], outside=[]),
    ]
    retained, _ = calib.retain_probes(rows)
    assert [row["gap_index"] for row in retained] == [2, 3, 1]


def test_global_jitter_is_aggregate_only(corpus_path: Path) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")[0]
    row = calib.run_shadow_case(case)
    block = calib.run_global_jitter(row, magnitudes=[10])
    assert block["draws_per_magnitude"] == calib.PERTURB_JITTER_DRAWS
    assert len(block["rows"]) == calib.PERTURB_JITTER_DRAWS
    for entry in block["rows"]:
        assert entry["magnitude_ms"] == 10
        assert "crossed_interval_boundary" not in entry
        assert "outside_cell" not in entry
        assert entry["error"] is None
        assert isinstance(entry["barrier_flips"], int)


def test_global_jitter_is_seeded_and_reproducible(corpus_path: Path) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")[0]
    import random

    first = calib.perturb_global(
        case.units, 20, random.Random(f"{case.id}:global_jitter:20")
    )
    second = calib.perturb_global(
        case.units, 20, random.Random(f"{case.id}:global_jitter:20")
    )
    assert first == second
    assert first != [dict(u) for u in case.units]


def test_perturbed_streams_stay_legal_documents(corpus_path: Path) -> None:
    """A jitter that broke monotonicity would be graded as if it were capturable."""
    import random

    corpus = calib.load_corpus(corpus_path)
    for case in _cases(corpus, "en-01", "ja-01", "zh-01"):
        jittered = calib.perturb_global(case.units, 50, random.Random("seed"))
        rebuilt = calib.perturbed_case(case, jittered)  # re-runs the validators
        assert len(rebuilt.units) == len(case.units)
        assert all(u["end"] >= u["start"] for u in rebuilt.units)
        assert all(u["start"] >= 0.0 for u in rebuilt.units)
        starts = [u["start"] for u in rebuilt.units]
        assert starts == sorted(starts)
        ends = [u["end"] for u in rebuilt.units]
        assert ends == sorted(ends)


def test_illegal_perturbation_is_refused_not_graded(corpus_path: Path) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = corpus.cases[0]
    broken = [dict(u) for u in case.units]
    broken[3]["end"] = broken[3]["start"] - 1.0
    with pytest.raises(cc.CalibrationError):
        calib.perturbed_case(case, broken)


def test_single_gap_clamp_keeps_ends_monotone(corpus_path: Path) -> None:
    """The clamp must not manufacture the very fallback it is trying to observe."""
    corpus = calib.load_corpus(corpus_path)
    case = corpus.cases[0]
    for index in range(min(len(case.units) - 1, 12)):
        for delta in (-500.0, 500.0):
            units, _ = calib.perturb_single_gap(case.units, index, delta)
            assert units[index]["end"] >= units[index]["start"]
            if index:
                assert units[index]["end"] >= units[index - 1]["end"]
            assert units[index]["end"] <= units[index + 1]["start"]


def test_near_cliff_scan_splits_its_denominator_by_vad_state(
    corpus_path: Path,
) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "zh-01")[0]
    row = calib.run_shadow_case(case)
    scan = calib.near_cliff_scan(case, row.document, [10, 20, 50])
    assert scan["candidate_gaps"] == len(case.units) - 1
    assert sum(scan["vad_state_denominators"].values()) == scan["candidate_gaps"]
    assert len(scan["gap_states"]) == scan["candidate_gaps"]
    assert set(scan["near_cliff_by_state"]) <= set(scan["vad_state_denominators"])
    assert sum(scan["near_cliff_by_state"].values()) == len(scan["near_cliff"])
    assert all(0 <= i < scan["candidate_gaps"] for i in scan["near_cliff"])


def test_near_cliff_selection_catches_a_gap_straddling_a_knee(
    corpus_path: Path,
) -> None:
    """The classifier is geometric, so a gap parked on a knee must be selected."""
    from voxweave.core.boundary_cost import RAMP_KNOWN_MS, UNCERTAINTY_MS

    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")[0]
    row = calib.run_shadow_case(case)
    knee_s = (RAMP_KNOWN_MS + UNCERTAINTY_MS) / 1000.0
    units = [dict(u) for u in case.units]
    # park gap 5 exactly on the upper ramp knee, with no speech over it
    units[5]["end"] = float(units[6]["start"]) - knee_s
    doc = row.document
    doc.vad_speech = []
    parked = calib.perturbed_case(case, units)
    scan = calib.near_cliff_scan(parked, doc, [20])
    assert 5 in set(scan["near_cliff"])


def test_near_cliff_selection_sees_a_state_flip_at_the_shifted_endpoint(
    corpus_path: Path,
) -> None:
    """AD4-3's speech-overlap geometry, which the knee test cannot reach.

    The classifier re-derives ``overlap_fraction`` at the SHIFTED endpoints
    rather than assuming it constant, so a gap whose shifted end walks out of a
    speech span changes both its curve and its state -- and that state change is
    itself a cliff. The landed knee fixture sets ``vad_speech = []``, i.e. the
    ``silence`` state at ``overlap_fraction == 0``, so it never exercises this.
    """
    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")[0]
    row = calib.run_shadow_case(case)
    units = [dict(u) for u in case.units]
    # a 200 ms gap whose first 5 ms carry speech: state "speech-overlap"
    left_end = float(units[5]["end"])
    units[6]["start"] = left_end + 0.200
    doc = row.document
    doc.vad_speech = [(left_end, left_end + 0.005)]
    probed = calib.perturbed_case(case, units)

    base = calib.near_cliff_scan(probed, doc, [10])
    assert base["gap_states"][5] == "speech-overlap"
    # a -10 ms nudge pulls the gap's start behind the speech span entirely, so
    # the fraction goes to zero and the state flips to "silence"
    assert 5 in set(base["near_cliff"])


def test_perturbation_driver_shape(corpus_path: Path) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = _cases(corpus, "en-01")[0]
    row = calib.run_shadow_case(case)
    block = calib.run_perturbation(
        [row], modes=["single_gap"], magnitudes=[20], max_probes=2
    )
    assert block["influence_radius_units"] == calib.INFLUENCE_RADIUS_UNITS
    assert block["modes"] == ["single_gap"]
    assert block["magnitudes_ms"] == [20]
    assert block["global_jitter"] == []
    assert block["exhaustive"] is False  # capped by max_probes
    assert block["totals"]["probes"] == block["single_gap"][0]["summary"]["probes"]
    assert block["totals"]["failures"] == len(block["failures"])
    # every failing probe survives the trim: it is the exit driver
    assert block["totals"]["failures"] == sum(
        1 for p in block["single_gap"][0]["probes"] if calib._probe_failed(p)
    )


def test_perturbation_with_zero_evidence_is_nonexhaustive_and_invalid() -> None:
    block = calib.run_perturbation(
        [], modes=["single_gap"], magnitudes=[20], max_probes=0
    )
    assert block["exhaustive"] is False
    assert block["classes"]["P1-unit-stability"] == {
        "attempted": 0,
        "failures": 0,
        "status": "invalid",
    }
    assert block["unknown"]
    assert calib.shadow_exit_code([], [], [], block["unknown"]) == cc.EXIT_INVALID


def test_shadow_report_times_the_complete_command_phases(
    corpus_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shadow-timing.json"
    assert (
        _run_cli(
            [
                "shadow",
                "--corpus",
                str(corpus_path),
                "--json-out",
                str(out),
                "--no-ablation",
            ]
        )
        == cc.EXIT_OK
    )
    timing = json.loads(out.read_text(encoding="utf-8"))["timing"]
    assert set(timing) == {
        "ablation_wall_s",
        "base_case_sum_s",
        "base_wall_s",
        "coarse_wall_s",
        "other_wall_s",
        "perturbation_wall_s",
        "report_wall_s",
        "slowest_case",
        "slowest_wall_s",
        "total_wall_s",
    }
    assert timing["total_wall_s"] >= timing["base_wall_s"]
    assert (
        timing["total_wall_s"]
        >= sum(
            timing[key]
            for key in (
                "base_wall_s",
                "coarse_wall_s",
                "ablation_wall_s",
                "perturbation_wall_s",
                "report_wall_s",
            )
        )
        - 0.001
    )


# --------------------------------------------------------------------------- #
# The adopted_v1 duplicate-cue annotation
# --------------------------------------------------------------------------- #


def _artifact_with(violations: list[dict], *, duplicated: bool, fallbacks: int) -> dict:
    empty = {"violations": [], "waivers": []}
    lanes: dict[str, Any] = {}
    for lane, row_ids in calib.SHADOW_LANE_ROWS.items():
        rows = {row_id: {"validator": dict(empty)} for row_id in row_ids}
        lanes[lane] = (
            rows
            if lane in {calib.SHADOW_LANE_CORE, calib.SHADOW_LANE_DELIVERY}
            else {"rows": rows}
        )
    return {
        "schema_version": calib.SHADOW_SCHEMA_VERSION,
        "coverage": {
            "fallback_intervals": fallbacks,
            "fallback_ranges_overlap": duplicated,
            "fallback_unit_ranges": [[0, 4], [2, 8]] if duplicated else [],
            "optimized_intervals": 0,
            "optimized_unit_ratio": 1.0 if not fallbacks else 0.5,
            "named_multi_cues_unannotated": 0,
            "raw_conservation_trustworthy": not duplicated,
            "unit_count": 8,
        },
        "lanes": lanes,
        "validator": {
            "raw": {"violations": violations, "waivers": []},
            "core": dict(empty),
            "legacy_overlay": dict(empty),
            "finalizer": dict(empty),
            "raw_duplicate_v1_cues": duplicated,
        },
    }


def _violation(kind: str, stage: str = "raw", origin: str = "v2") -> dict:
    return {
        "cue_index": 1,
        "detail": f"{kind} at {stage}",
        "kind": kind,
        "origin": origin,
        "stage": stage,
        "waived": False,
    }


def test_duplicate_v1_cues_are_never_read_as_conservation_evidence() -> None:
    """Two adjacent ``adopted_v1`` fallbacks can adopt the same v1 cue.

    The raw-stage validator then sees that cue twice and reports a conservation
    violation about the *reporting shape*, not about the partition. Counting it
    as a v2 defect would turn a fallback into a fabricated conservation failure.
    """
    artifact = _artifact_with(
        [_violation("unit-conservation"), _violation("text-conservation")],
        duplicated=True,
        fallbacks=2,
    )
    counts = calib.shadow_violation_counts(artifact)
    assert counts["duplicate_v1_cues"] is True
    assert counts["not_conservation_evidence"] == 2
    assert counts["exit_driving"] == []
    # the rows are still reported, just not as evidence
    assert counts["stages"]["raw"]["violations"] == 2


def test_a_fallback_still_fails_c13_on_its_own_terms() -> None:
    """Suppressing the duplicate rows must not make a fallback look clean."""
    artifact = _artifact_with(
        [_violation("unit-conservation")], duplicated=True, fallbacks=2
    )
    case = calib.Case(
        path=Path("zh-01.json"),
        relpath="cases/zh-01.json",
        doc={"id": "zh-01", "language": "zh"},
        size_bytes=0,
    )
    failures = calib.c13_case_failures(
        case, artifact["coverage"], calib.shadow_violation_counts(artifact)
    )
    assert any("adopted_v1" in line for line in failures)
    assert any("optimized_unit_ratio" in line for line in failures)


def test_without_duplicates_a_conservation_violation_drives_the_exit() -> None:
    artifact = _artifact_with(
        [_violation("unit-conservation")], duplicated=False, fallbacks=0
    )
    counts = calib.shadow_violation_counts(artifact)
    assert counts["not_conservation_evidence"] == 0
    assert [row["kind"] for row in counts["exit_driving"]] == ["unit-conservation"]


def test_only_unwaived_v2_raw_and_core_rows_drive_the_exit() -> None:
    """AD3-3: v1 rows and every legacy-overlay row are evidence, never a verdict."""
    rows = [
        _violation("speech-truncated-start", stage="core", origin="v1"),
        _violation("speech-truncated-start", stage="legacy-overlay"),
        {**_violation("duration-cap", stage="core"), "waived": True},
        _violation("line-capacity", stage="core"),
    ]
    artifact = _artifact_with([], duplicated=False, fallbacks=0)
    artifact["validator"]["core"] = {"violations": rows, "waivers": []}
    counts = calib.shadow_violation_counts(artifact)
    assert [row["kind"] for row in counts["exit_driving"]] == ["line-capacity"]


def test_lane_cue_stream_refuses_an_unprojected_partition() -> None:
    units = [{"text": "a", "start": 0.0, "end": 1.0}]
    rows = [{"text": "a", "start": 0.0, "end": 1.0, "lyric": False, "unit_range": None}]
    assert calib.lane_cue_stream(rows, units) is None
    resolved = [{**rows[0], "unit_range": [0, 1]}]
    built = calib.lane_cue_stream(resolved, units)
    assert built is not None
    assert built[0]["word_data"] == [{"text": "a", "start": 0.0, "end": 1.0}]


# --------------------------------------------------------------------------- #
# An unmeasured stage is an invalid measurement, never a clean one
# --------------------------------------------------------------------------- #


def _shadow_case_stub(case_id: str = "zh-01") -> Any:
    return calib.Case(
        path=Path(f"{case_id}.json"),
        relpath=f"cases/{case_id}.json",
        doc={"id": case_id, "language": case_id.split("-")[0]},
        size_bytes=0,
    )


_MEASURABLE_ARTIFACT: dict[str, Any] | None = None


def _measurable(_artifact: dict) -> dict:
    """Return a real closed schema-2 payload for one-field mutation tests."""
    global _MEASURABLE_ARTIFACT
    if _MEASURABLE_ARTIFACT is None:
        from tests.test_shadow_hook import _case_plain, _segment

        with calib._forced_shadow(True):
            artifact = _segment(_case_plain()).shadow
        assert isinstance(artifact, dict) and artifact["schema_version"] == 2
        _MEASURABLE_ARTIFACT = artifact
    return copy.deepcopy(_MEASURABLE_ARTIFACT)


@pytest.mark.parametrize("stage", calib.SHADOW_REQUIRED_STAGES)
def test_a_missing_validator_stage_is_reported_and_is_not_a_clean_run(stage) -> None:
    """Bug pin: a falsy stage block used to read as "nothing to report".

    The hook has exactly one producer per stage, and dropping the ``core``
    assignment left the whole suite green while a real shipped defect could be
    reintroduced with the gate still at exit 0.
    """
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["validator"][stage] = None
    counts = calib.shadow_violation_counts(artifact)
    assert counts["missing_stages"] == [stage]
    assert counts["stages"][stage] is None
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any(stage in line for line in problems)


def test_all_stages_present_reports_no_measurement_error() -> None:
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    counts = calib.shadow_violation_counts(artifact)
    assert counts["missing_stages"] == []
    assert calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts) == []


def test_a_forged_finalizer_trace_invalidates_the_measurement() -> None:
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["lanes"][calib.SHADOW_LANE_FINALIZER]["rows"]["v1"]["finalizer"][
        "trace_errors"
    ] = ["forged neighbour snapshot"]
    counts = calib.shadow_violation_counts(artifact)
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any(
        "delivery_finalizer.rows.v1" in line and "trace_errors" in line
        for line in problems
    )


def test_a_preview_fidelity_mismatch_invalidates_the_measurement() -> None:
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["preview_fidelity"]["mismatches"] = [{"edge_index": 0}]
    counts = calib.shadow_violation_counts(artifact)
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any("preview_fidelity.mismatches" in line for line in problems)


def test_an_inconsistent_speaker_projection_invalidates_the_measurement() -> None:
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["speaker_evidence"]["projection"]["range_count"] = 2
    counts = calib.shadow_violation_counts(artifact)
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any("N19 speaker-id projection" in line for line in problems)


def test_speaker_on_off_conservation_is_an_invalid_measurement_precondition() -> None:
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["speaker_evidence"]["measurement"]["raw_in_speech_turn_changes"] = 1
    counts = calib.shadow_violation_counts(artifact)
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any("do not conserve" in line for line in problems)


def test_a_disagreeing_projection_cross_check_invalidates_the_measurement() -> None:
    """Every measured lane row is keyed by that structural projection.

    ``lane_cue_stream`` rebuilds cue ``word_data`` from the projected unit range,
    so a bad reconciliation moves CPS and mid-phrase measurements without moving
    a boundary.
    """
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["lanes"][calib.SHADOW_LANE_CORE]["v2"]["projection_cross_check"] = {
        "agrees": False,
        "mode": "surface-reconciled",
    }
    counts = calib.shadow_violation_counts(artifact)
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any("projection_cross_check.agrees" in line for line in problems)


def test_a_missing_projection_cross_check_invalidates_the_measurement() -> None:
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["lanes"][calib.SHADOW_LANE_CORE]["v2"].pop("projection_cross_check")
    counts = calib.shadow_violation_counts(artifact)
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any("projection_cross_check" in line for line in problems)


def test_an_unprojected_v1_stream_invalidates_the_measurement() -> None:
    """The tracked corpus is word-level and must project; if it stops, say so."""
    artifact = _measurable(_artifact_with([], duplicated=False, fallbacks=0))
    artifact["coverage"]["v1_unprojected"] = True
    artifact["v1_projection"]["mode"] = "stream-ends-at-char-26-of-33"
    counts = calib.shadow_violation_counts(artifact)
    problems = calib.shadow_measurement_errors(_shadow_case_stub(), artifact, counts)
    assert any(
        "v1_unprojected" in line or "v1_projection.unprojected" in line
        for line in problems
    )


def test_measurement_errors_outrank_a_failed_gate() -> None:
    assert (
        calib.shadow_exit_code(
            [_gate("fail")], ["zh-01: fallback"], [], [], ["zh-01: stage missing"]
        )
        == cc.EXIT_INVALID
    )


# --------------------------------------------------------------------------- #
# AD-2 probes that cannot be evaluated
# --------------------------------------------------------------------------- #


def test_an_unresolvable_probe_is_unknown_not_a_pass() -> None:
    """Bug pin: ``crossed_interval_boundary: False`` on a probe with no partition.

    AD-2's rule is "FAIL when any moved boundary lies outside the influence
    cell". A probe that could not report its moved set has not satisfied that
    rule, it has failed to answer it -- and at the exit it was indistinguishable
    from a clean one.
    """
    report = calib._cell_report(
        {1, 2},
        {
            "partition": None,
            "barrier_units": (),
            "cue_count": 0,
            "fallback_intervals": 0,
        },
        (),
        (3,),
    )
    assert report["crossed_interval_boundary"] is None
    assert report["error"] == "partition unresolved"
    probe = {
        "lanes": {"natural": report},
        "gap_index": 0,
        "magnitude_ms": 10,
        "sign": 1,
    }
    assert calib._probe_failed(probe) is False
    assert calib._probe_unknown(probe) is True


def test_a_probe_that_raised_is_unknown_too() -> None:
    probe = {
        "lanes": {},
        "error": "boom",
        "gap_index": 0,
        "magnitude_ms": 10,
        "sign": 1,
    }
    assert calib._probe_failed(probe) is False
    assert calib._probe_unknown(probe) is True


def test_a_measured_clean_probe_is_neither_failed_nor_unknown() -> None:
    report = calib._cell_report(
        {1, 2},
        {
            "partition": {1, 2},
            "barrier_units": (),
            "cue_count": 3,
            "fallback_intervals": 0,
        },
        (),
        (3,),
    )
    probe = {
        "lanes": {"natural": report},
        "gap_index": 0,
        "magnitude_ms": 10,
        "sign": 1,
    }
    assert report["crossed_interval_boundary"] is False
    assert calib._probe_failed(probe) is False
    assert calib._probe_unknown(probe) is False


def test_unknown_probes_are_retained_and_counted() -> None:
    unknown = {
        "lanes": {},
        "error": "boom",
        "gap_index": 4,
        "magnitude_ms": 10,
        "sign": 1,
    }
    retained, summary = calib.retain_probes([unknown])
    assert summary["unknown"] == 1
    assert summary["failures"] == 0
    assert retained == [unknown]


def test_an_unknown_probe_exits_invalid_not_ok() -> None:
    assert calib.shadow_exit_code([_gate("pass")], [], [], [{"case": "en-01"}]) == (
        cc.EXIT_INVALID
    )
    assert calib.shadow_exit_code([_gate("pass")], [], [], []) == cc.EXIT_OK


def test_the_frozen_shadow_target_arms_the_ad2_exit_driver() -> None:
    """AD-2's influence-cell FAIL is one of the three declared exits.

    It is unreachable without ``--perturb``, so a target that omits the flag
    leaves that exit structurally dead and the gate depends on a human
    remembering it. The slice is deliberately bounded (one magnitude, near-cliff
    gaps only, one case per language) -- pinned here so it stays a *bounded*
    run rather than being widened into something nobody will wait for, or
    quietly dropped again.
    """
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("quality-shadow-segmentation:", 1)[1].split("\n\n", 1)[0]
    assert "calib_segmentation.py shadow" in target
    assert "--perturb " in target
    assert "--perturb-near-cliff-only" in target
    assert "--perturb-magnitude 50" in target
    assert "--perturb-mode single_gap" in target
    assert "--perturb-max-probes 2" in target
    assert "--no-ablation" in target
    for case in ("en-01", "ja-01", "zh-01"):
        assert f"--perturb-case {case}" in target
    assert "--check" in target
    assert "quality-shadow-segmentation-full:" in makefile
    assert "4 h 58 min" in makefile


def test_exit_driving_stage_set_is_partition_checks_not_a_restated_literal() -> None:
    """The harness must honour partition_check.EXIT_DRIVING_STAGES verbatim.

    The rule used to be restated as a ("raw", "core") literal here, so a stage
    added to the validator vocabulary (P5 adds "finalizer") would drive the
    validator exit while the harness silently ignored it. Pin: a violation at
    every member of EXIT_DRIVING_STAGES is counted as exit-driving evidence.
    """
    from voxweave.core.partition_check import EXIT_DRIVING_STAGES

    assert EXIT_DRIVING_STAGES  # non-empty guard: an empty set would void the pin
    for stage in sorted(EXIT_DRIVING_STAGES):
        artifact = _artifact_with(
            [_violation("overlap", stage=stage)], duplicated=False, fallbacks=0
        )
        counts = calib.shadow_violation_counts(artifact)
        assert counts["exit_driving"], f"stage {stage!r} must drive the exit"
