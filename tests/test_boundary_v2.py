# tests/test_boundary_v2.py
"""The exact whole-interval solver, its selection policy, and the artifact.

AD-1 deleted the tile machinery, so every hard interval is solved EXACTLY. Two
tests carry that claim:

* **brute-force equality** -- on randomized intervals the DP's cut set and total
  must equal an independent enumeration of every legal path, including the
  canonical tie-break where several optima exist. Both sides read the same cost
  tables, so the comparison isolates the solver mechanics rather than re-deriving
  the policy;
* **the AD4-2 work gate** -- DP relaxations and packer extensions are counted
  separately and each must stay under ``atoms * (BAND + 2)`` on one fixture that
  combines the widest supported layout, minimum-width atoms, a disabled duration
  cap and a tie-heavy equal-cost graph.

The rest pins what a shadow reader is allowed to believe: ``adopted_v1`` output is
never scored as v2, ``policy_selected`` only migrates inside ``POLICY_MARGIN``,
materialized cues carry ``_chunk_to_cue``'s shapes, and the artifact is byte
stable across identical runs.
"""

import json
import random

import pytest

from voxweave.core.boundary_cost import quantize
from voxweave.core.boundary_lattice import (
    Edge,
    LatticeAtom,
    band_atoms,
    build_document_lattice,
)
from voxweave.core.boundary_v2 import (
    ENGINE_V2,
    POLICY_MARGIN,
    SCHEMA_VERSION,
    V1Partition,
    build_cost_context,
    build_cost_tables,
    materialize_cues,
    optimize_document,
    score_path,
    shadow_artifact,
    solve_interval,
)
from voxweave.core.segdoc import DisplayProfile, SegDocument, SourceUnit

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


def document(spec, *, prof=None, **kwargs):
    prof = prof or profile()
    us = [
        SourceUnit(id=f"u{i}", surface=s, start=a, end=b)
        for i, (s, a, b) in enumerate(spec)
    ]
    sep = "" if prof.language in {"ja", "zh", "yue"} else " "
    return SegDocument(
        language=prof.language,
        units=us,
        profile=prof,
        vad_speech=kwargs.pop("vad_speech", None),
        shot_changes=kwargs.pop("shot_changes", None),
        sing_spans=None,
        speaker_turns=None,
        manifest={},
        text=sep.join(u.surface for u in us),
    )


def timed(surfaces, *, dur=0.3, gap=0.05, start=0.0):
    out = []
    t = start
    for s in surfaces:
        out.append((s, round(t, 6), round(t + dur, 6)))
        t = t + dur + gap
    return out


def tie_document():
    """Six identical zero-length atoms at a budget that fits two, three or four.

    Every mid cue costs the same (the preview grants nothing, so the reading and
    min-duration terms saturate identically), every cut costs the same, and every
    last cue costs its bare cue base -- so ``[2,4]``, ``[3,3]`` and ``[4,2]`` are
    exactly equal optima. That is the cleanest available probe of the tie-break
    and of the migration rule.
    """
    return document(
        [("ab", 0.0, 0.0)] * 6, prof=profile(max_line_length=12, max_lines=1)
    )


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


# ------------------------------------------------- independent path enumeration


def enumerate_paths(lattice):
    """Every legal 0 -> N path, as its tuple of interior cut nodes."""
    target = len(lattice.atoms)
    found = []

    def walk(node, cuts):
        for edge in lattice.edges_from.get(node, ()):
            if edge.end_node == target:
                found.append(tuple(cuts))
            else:
                walk(edge.end_node, [*cuts, edge.end_node])

    walk(0, [])
    return found


def brute_total(lattice, tables, cuts):
    """The DP's accumulation order, replayed by hand."""
    nodes = (0, *cuts, len(lattice.atoms))
    total = 0.0
    for left, right in zip(nodes, nodes[1:]):
        total = quantize(total + tables.edges[(left, right)].total)
        if right != len(lattice.atoms):
            total = quantize(total + tables.cuts[right].total)
    return total


def canonical(paths):
    """AD3-1's local rule, read globally: minimise the last cut, then the one
    before it, and so on -- the zero-padded reversed cut tuple."""
    width = max((len(p) for p in paths), default=0)

    def key(cuts):
        rev = list(reversed(cuts))
        return rev + [0] * (width - len(rev))

    return min(paths, key=key)


def solved(document_obj, **kwargs):
    solution = optimize_document(document_obj, **kwargs)
    assert solution.solutions, "expected at least one interval"
    return solution


# ---------------------------------------------------------------- AD-1 exactness


@pytest.mark.parametrize("draw", range(100))
def test_dp_equals_brute_force_on_randomized_intervals(draw):
    rng = random.Random(f"p4-brute:{draw}")
    n = rng.randint(4, 12)
    words = [rng.choice(["a", "bo", "cat", "deed"]) for _ in range(n)]
    spec = []
    clock = 0.0
    for word in words:
        span = round(rng.uniform(0.05, 0.4), 3)
        spec.append((word, round(clock, 3), round(clock + span, 3)))
        clock += span + round(rng.uniform(0.0, 0.3), 3)
    doc = document(spec, prof=profile(max_line_length=10, max_lines=1, max_cue_s=3.0))

    built = build_document_lattice(doc)
    assert len(built.lattices) == 1
    lattice = built.lattices[0]
    ctx = build_cost_context(doc, built)
    tables = build_cost_tables(lattice, ctx)

    paths = enumerate_paths(lattice)
    assert paths, "the single-atom edges guarantee a path"
    totals = {path: brute_total(lattice, tables, path) for path in paths}
    best_total = min(totals.values())
    optima = [path for path in paths if totals[path] == best_total]

    result = solve_interval(lattice, tables)
    assert result.best.total == best_total
    assert result.best.cuts == canonical(optima)
    assert set(result.best.cuts) <= set(lattice.nodes)


@pytest.mark.parametrize("draw", range(6))
def test_dp_equals_brute_force_at_the_pinned_upper_bound(draw):
    """The spec's N <= 18 bound, exercised at the bound."""
    rng = random.Random(f"p4-brute-18:{draw}")
    n = rng.randint(15, 18)
    spec = timed([rng.choice(["a", "bo", "cat"]) for _ in range(n)], dur=0.1, gap=0.05)
    doc = document(spec, prof=profile(max_line_length=6, max_lines=1, max_cue_s=3.0))

    built = build_document_lattice(doc)
    assert len(built.lattices) == 1
    lattice = built.lattices[0]
    tables = build_cost_tables(lattice, build_cost_context(doc, built))
    paths = enumerate_paths(lattice)
    totals = {path: brute_total(lattice, tables, path) for path in paths}
    best_total = min(totals.values())
    result = solve_interval(lattice, tables)
    assert result.best.total == best_total
    assert result.best.cuts == canonical([p for p in paths if totals[p] == best_total])


def test_runner_up_is_the_best_path_distinct_alternative():
    doc = tie_document()
    built = build_document_lattice(doc)
    lattice = built.lattices[0]
    tables = build_cost_tables(lattice, build_cost_context(doc, built))
    paths = enumerate_paths(lattice)
    totals = {path: brute_total(lattice, tables, path) for path in paths}

    result = solve_interval(lattice, tables)
    others = [t for path, t in totals.items() if path != result.best.cuts]
    assert result.runner_up is not None
    assert result.runner_up.total == min(others)
    assert result.runner_up.cuts != result.best.cuts


def test_score_path_refuses_an_illegal_path():
    doc = tie_document()
    built = build_document_lattice(doc)
    lattice = built.lattices[0]
    tables = build_cost_tables(lattice, build_cost_context(doc, built))
    with pytest.raises(ValueError):
        score_path(lattice, tables, ())  # 6 atoms in one cue is over budget


def test_the_tie_fixture_really_ties_and_the_local_rule_picks_the_low_last_cut():
    doc = tie_document()
    built = build_document_lattice(doc)
    lattice = built.lattices[0]
    tables = build_cost_tables(lattice, build_cost_context(doc, built))
    totals = {p: brute_total(lattice, tables, p) for p in enumerate_paths(lattice)}
    best = min(totals.values())
    assert sorted(p for p in totals if totals[p] == best) == [(2,), (3,), (4,)]

    interval = solved(doc).solutions[0]
    assert interval.selection.raw_optimum.cuts == (2,)
    assert interval.selection.margin == pytest.approx(0.0)
    assert interval.selection.low_margin is True


# ---------------------------------------------------------------- AD4-2 work gate


def test_adversarial_work_fixture_stays_inside_the_resolved_band():
    """Widest layout + one-cell atoms + disabled cap + tie-heavy costs, at once."""
    prof = profile(max_line_length=42, max_lines=2, max_cue_s=0.0)
    doc = document([("a", 0.0, 0.0)] * 200, prof=prof)
    interval = solved(doc).solutions[0]

    assert interval.lattice.all_invisible is False
    atoms = len(interval.lattice.atoms)
    limit = atoms * (band_atoms(prof) + 2)
    assert interval.dp_relaxations < limit
    assert interval.packer_steps < limit
    # the early break really fires: no edge is wider than the proven band
    assert max(e.end_node - e.start_node for e in interval.lattice.edges) <= band_atoms(
        prof
    )


def test_the_all_invisible_branch_is_constant_work():
    """AD4-1: 2000 punctuation atoms produce ONE forced edge and no packing."""
    interval = solved(document([("!", 0.0, 0.0)] * 2000)).solutions[0]
    assert interval.lattice.all_invisible is True
    assert interval.packer_steps == 0
    assert interval.dp_relaxations == 1
    assert interval.optimized is True
    assert interval.partition_units == ()
    assert len(interval.cues) == 1


# ---------------------------------------------------------------- AD5-1 nine bangs


def test_nine_bang_family_yields_two_cap_legal_cues_and_no_v2_violation():
    """The round-5 counterexample: an all-invisible interval over the cap."""
    spec = [("!", round(1.01 * i, 4), round(1.01 * i, 4)) for i in range(9)]
    doc = document(spec)
    solution = solved(doc)
    interval = solution.solutions[0]

    assert interval.lattice.all_invisible is True
    assert interval.optimized is True
    assert interval.partition_units == (7,)
    assert len(interval.cues) == 2
    for cue in interval.cues:
        assert cue["end"] - cue["start"] <= doc.profile.max_cue_s + 1e-9
    assert interval.validator_raw.exit_driving == ()
    assert interval.validator_raw.unwaived == ()
    assert solution.artifact["totals"]["hard_violations"] == 0


# ---------------------------------------------------------------- C8 selection


def test_policy_selected_migrates_to_an_equally_good_v1_path():
    doc = tie_document()
    interval = solved(doc, v1=V1Partition(cuts=(4,))).solutions[0]
    assert interval.selection.v1_path_legal is True
    assert interval.selection.v1_illegality is None
    assert interval.selection.v1_cost_under_v2 is not None
    assert interval.selection.raw_optimum.cuts == (2,)
    assert interval.selection.policy_selected.cuts == (4,)
    assert interval.selection.selected_is_v1 is True


def test_policy_selected_refuses_a_v1_path_outside_the_margin():
    doc = tie_document()
    interval = solved(doc, v1=V1Partition(cuts=(1, 3))).solutions[0]
    assert interval.selection.v1_path_legal is True
    over = interval.selection.v1_cost_under_v2.total
    assert over > interval.selection.raw_optimum.total + POLICY_MARGIN
    assert interval.selection.selected_is_v1 is False
    assert interval.selection.policy_selected.cuts == (2,)


def test_an_illegal_v1_path_is_reported_not_scored():
    doc = tie_document()
    interval = solved(doc, v1=V1Partition(cuts=())).solutions[0]
    assert interval.selection.v1_path_legal is False
    assert interval.selection.v1_illegality
    assert interval.selection.v1_cost_under_v2 is None
    assert interval.selection.selected_is_v1 is False


def test_policy_margin_is_the_declared_constant():
    assert POLICY_MARGIN == 1.0


# ---------------------------------------------------------------- C9 fallback


def infeasible_document():
    """A span-preflight violation makes the whole (single) interval infeasible."""
    return document(
        [("one", 0.0, 0.5), ("two", 0.6, 1.0), ("three", float("nan"), 2.0)]
    )


def test_adopted_v1_is_never_counted_as_an_optimized_v2_result():
    doc = infeasible_document()
    v1_cues = [
        {
            "text": "one two",
            "start": 0.0,
            "end": 1.0,
            "word_data": [],
            "speech_start": 0.0,
            "speech_end": 1.0,
        },
        {
            "text": "three",
            "start": 1.0,
            "end": 2.0,
            "word_data": [],
            "speech_start": None,
            "speech_end": 2.0,
        },
    ]
    solution = solved(doc, v1=V1Partition(cuts=(2,), cues=tuple(v1_cues)))
    interval = solution.solutions[0]

    assert interval.optimized is False
    assert interval.selection is None
    assert interval.adopted is not None
    assert interval.adopted.reason == "span-preflight"
    assert interval.lattice.infeasible.reason == "span-preflight"

    totals = solution.artifact["totals"]
    assert totals["fallback_intervals"] == 1
    assert totals["optimized_intervals"] == 0
    assert totals["optimized_unit_ratio"] == 0.0

    block = solution.artifact["intervals"][0]
    assert block["adopted_v1"] is True
    assert block["raw_optimum"] is None
    assert block["policy_selected"] is None
    assert block["infeasible"]["reason"] == "span-preflight"


def test_a_healthy_document_reports_full_coverage():
    solution = solved(document(timed(["the", "cat", "sat", "down"])))
    totals = solution.artifact["totals"]
    assert totals["fallback_intervals"] == 0
    assert totals["optimized_unit_ratio"] == 1.0
    assert totals["optimized_intervals"] == len(solution.solutions)


# ---------------------------------------------------------------- materialization


def test_materialized_cues_mirror_chunk_to_cue_shapes():
    solution = solved(document(timed(["the", "cat", "sat", "down"])))
    cues = [cue for interval in solution.solutions for cue in interval.cues]
    assert cues
    for cue in cues:
        assert set(cue) >= {
            "text",
            "start",
            "end",
            "word_data",
            "speech_start",
            "speech_end",
        }
        assert isinstance(cue["start"], float) and isinstance(cue["end"], float)
        for unit in cue["word_data"]:
            assert "text" in unit  # atom level, never the raw ``word`` slice
            assert "word" not in unit
            assert set(unit) == {"text", "start", "end"}


def test_speech_anchors_take_no_fallback_when_the_chunk_is_untimed():
    """``_chunk_to_cue``'s rule: invented time is never laundered into the anchors."""
    atoms = [atom(0, "a"), atom(1, "b")]
    edges = [
        Edge(
            start_node=0,
            end_node=2,
            text="a b",
            display_text="a b",
            lines=1,
            line_widths=(3,),
            span_start=None,
            span_end=None,
            waiver=None,
        )
    ]
    cues = materialize_cues(edges, atoms, "en", fallback_start=2.5)
    assert cues[0]["text"] == "a b"
    assert cues[0]["start"] == 2.5 and cues[0]["end"] == 2.5
    assert cues[0]["speech_start"] is None and cues[0]["speech_end"] is None


def test_materialized_partitions_speak_unit_ids_in_the_artifact():
    solution = solved(tie_document())
    block = solution.artifact["intervals"][0]
    assert block["v2_partition"] == list(solution.solutions[0].partition_units)
    assert block["unit_range"] == [0, 6]


# ---------------------------------------------------------------- artifact


def test_artifact_carries_the_declared_schema():
    solution = solved(document(timed(["the", "cat", "sat"])))
    art = solution.artifact
    assert art["kind"] == "segmentation-shadow"
    assert art["schema_version"] == SCHEMA_VERSION
    assert art["engine_v2"] == ENGINE_V2
    assert art["policy_version"] == 1
    assert art["policy_name"] == "experimental_policy_1"
    for key in (
        "totals",
        "intervals",
        "validator",
        "production_degraded",
        "shadow_degraded",
        "influence_cell",
        "pause_knees",
        "policy_deltas",
    ):
        assert key in art
    assert art["validator"]["raw"] is not None
    assert art["production_degraded"] == []  # Wave B fills these
    assert art["shadow_degraded"] == []


def test_interval_blocks_expose_the_coverage_and_addendum_fields():
    block = solved(document(timed(["the", "cat", "sat"]))).artifact["intervals"][0]
    for key in (
        "coalesced_atoms",
        "all_invisible",
        "relief_injections",
        "waivers",
        "validator_raw",
        "dp_relaxations",
        "packer_steps",
        "v2_partition",
    ):
        assert key in block


def test_shadow_artifact_is_the_thin_wrapper():
    doc = document(timed(["the", "cat", "sat"]))
    assert shadow_artifact(doc) == optimize_document(doc).artifact


def test_the_artifact_is_byte_stable_across_identical_runs():
    first = shadow_artifact(document(timed(["the", "cat", "sat", "down"])))
    second = shadow_artifact(document(timed(["the", "cat", "sat", "down"])))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_an_invalid_profile_aborts_the_measurement_instead_of_degrading():
    doc = document(timed(["the", "cat"]), prof=profile(clause_ms=0.0))
    solution = optimize_document(doc)
    assert solution.solutions == ()
    assert [v.key for v in solution.invalid_profile] == ["clause_ms"]
    assert set(solution.artifact) == {
        "kind",
        "schema_version",
        "engine_v2",
        "policy_version",
        "invalid_profile",
    }


def test_a_negative_cap_is_refused_rather_than_reinterpreted():
    """AD3-4: the shadow never disagrees with cleanup about a negative cap."""
    doc = document(timed(["the", "cat"]), prof=profile(max_cue_s=-1.0))
    solution = optimize_document(doc)
    assert [v.key for v in solution.invalid_profile] == ["max_cue_s"]


# ------------------------------------------------- AD3-3 waiver bookkeeping


@pytest.mark.parametrize("surface", ["!", "held"])
def test_a_waived_cue_is_waived_in_the_document_pass_too(surface):
    """The two exit counters in one artifact must not contradict each other.

    ``totals.hard_violations`` sums the per-interval passes and
    ``validator.raw`` is a whole-document pass, so a waiver ledger that reached
    only the first would have the artifact call the same cue both exempt and
    exit-driving -- and which a Wave B reader believed would depend on the field
    it happened to read. The parametrisation covers both branches: ``!`` takes
    the all-invisible path, ``held`` the mixed one.
    """
    solution = optimize_document(document([(surface, 0.0, 8.0)]))
    interval = solution.solutions[0]
    assert [w.kind for w in interval.waivers] == ["held-chain-duration"]
    assert interval.validator_raw.exit_driving == ()

    raw = solution.artifact["validator"]["raw"]
    assert [v["kind"] for v in raw["violations"]] == ["duration-cap"]
    assert raw["violations"][0]["waived"] is True
    assert raw["exit_driving"] == solution.artifact["totals"]["hard_violations"] == 0
    assert len(raw["waivers"]) == 1


def test_document_waivers_are_stamped_with_document_cue_indices():
    """Two waived intervals: the second exemption must not point at cue 0."""
    doc = document(
        [("held", 0.0, 8.0), ("later", 20.0, 28.0)],
        prof=profile(vad_skip_ms=1000.0),
    )
    solution = optimize_document(doc)
    assert len(solution.solutions) == 2
    assert all(s.waivers for s in solution.solutions)
    raw = solution.artifact["validator"]["raw"]
    assert sorted(w["cue_index"] for w in raw["waivers"]) == [0, 1]
    assert raw["exit_driving"] == 0
