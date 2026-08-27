# tests/test_boundary_lattice.py
"""The P4 lattice: preflight, barriers, coalescing, hard edges, relief.

The lattice is where P4's boundedness contract lives. Two invariants carry the
whole complexity argument and are therefore pinned directly rather than inferred
from a timing measurement:

* **positive display progress** -- after canonical zero-display coalescing every
  atom is worth at least one half-width cell, so a legal edge spans at most
  ``BAND`` atoms and the monotone early break provably fires (AD3-1/AD4-2);
* **the all-invisible branch is total** -- an interval with no visible atom at
  all still obeys the duration cap, source-unit relief and the held-chain
  whitelist before it may call itself optimized (AD4-1/AD5-1/AD6-1).

Everything the lattice consumes from ``smart_split``/``layout`` is used verbatim;
a test that would pass only against a re-implementation of one of those helpers
is testing the wrong thing.
"""

import random

import pytest

from voxweave.core.canonical_text import canonical_legal, canonical_text
from voxweave.core.boundary_lattice import (
    BARRIER_KINDS,
    BARRIER_UNCERTAINTY_MS,
    CAP_EPS_S,
    HELD_CHAIN_MAX_GAP_S,
    INFEASIBLE_REASONS,
    INFLUENCE_RADIUS_UNITS,
    RELIEF_TRIGGER_FACTOR,
    SPAN_VIOLATION_REASONS,
    AtomLayer,
    HardBarrier,
    IncrementalPacker,
    IntervalLattice,
    LatticeAtom,
    band_atoms,
    build_atom_layer,
    build_barriers,
    build_document_lattice,
    build_intervals,
    candidate_nodes,
    coalesce_zero_display,
    exclusively_owned,
    granularity_check,
    greedy_cap_partition,
    held_chain_continuous,
    preflight_profile,
    preflight_units,
    relief_nodes,
    relief_trigger,
    resolve_cap_partition,
    sentence_end_nodes,
    span_max,
    span_min,
    split_candidate_at_unit,
    unit_edge_nodes,
)
from voxweave.core.layout import (
    _fits_budget,
    _join,
    _line_budget_width,
    _vis_width,
    split_subtitle,
    strip_punct_for_subtitles,
)
from voxweave.core.segdoc import DisplayProfile, SegDocument, SourceUnit
from voxweave.core.timing import HELD_WORD_MAX_GAP_S

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


def units_from(spec):
    return [
        SourceUnit(id=f"u{i}", surface=s, start=a, end=b)
        for i, (s, a, b) in enumerate(spec)
    ]


def document(spec, *, prof=None, text=None, **kwargs):
    """A ``SegDocument`` whose ``text`` is the exact join the v1 engine consumed."""
    us = units_from(spec)
    prof = prof or profile()
    return SegDocument(
        language=prof.language,
        units=us,
        profile=prof,
        vad_speech=kwargs.pop("vad_speech", None),
        shot_changes=kwargs.pop("shot_changes", None),
        sing_spans=None,
        speaker_turns=None,
        manifest={},
        text=_join([u.surface for u in us], prof.language) if text is None else text,
    )


def timed(surfaces, *, start=0.0, dur=0.3, gap=0.0):
    out = []
    t = start
    for s in surfaces:
        out.append((s, round(t, 6), round(t + dur, 6)))
        t = t + dur + gap
    return out


def atom(index, text, *, start=None, end=None, unit_start=None, unit_end=None, **over):
    """A ``LatticeAtom`` built by hand, for the branch tests that must not depend
    on ``_build_atoms``' own reconciliation."""
    unit_start = index if unit_start is None else unit_start
    unit_end = index + 1 if unit_end is None else unit_end
    base = dict(
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
        display=strip_punct_for_subtitles(text),
        members=(index,),
    )
    base.update(over)
    return LatticeAtom(**base)


def reachable(lattice):
    """Is there any 0 -> N path through the legal edge set?"""
    target = len(lattice.atoms)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for edge in lattice.edges_from.get(node, ()):
            if edge.end_node not in seen:
                seen.add(edge.end_node)
                stack.append(edge.end_node)
    return target in seen


# ---------------------------------------------------------------- C3 preflight


def test_none_bounds_are_tolerated_not_violations():
    """C3 (editorial correction, review 7): ``None`` is missing evidence, and C6/C7
    handle it -- it is never a span-preflight violation."""
    us = units_from([("a", None, None), ("b", 1.0, 2.0), ("c", None, 3.0)])
    assert preflight_units(us) == ()


def test_span_preflight_catches_every_declared_violation():
    cases = {
        "non-finite": [("a", float("nan"), 1.0)],
        "reversed": [("a", 2.0, 1.0)],
        "starts-non-monotone": [("a", 5.0, 6.0), ("b", 1.0, 7.0)],
        "ends-non-monotone": [("a", 0.0, 6.0), ("b", 1.0, 2.0)],
    }
    for reason, spec in cases.items():
        found = {v.reason for v in preflight_units(units_from(spec))}
        assert reason in found, (reason, found)
        assert found <= set(SPAN_VIOLATION_REASONS)


def test_span_preflight_rejects_bool_bounds():
    us = [SourceUnit(id="u0", surface="a", start=True, end=1.0)]
    assert {v.reason for v in preflight_units(us)} == {"bool-bound"}


def test_span_preflight_rejects_infinities():
    us = units_from([("a", 0.0, float("inf"))])
    assert {v.reason for v in preflight_units(us)} == {"non-finite"}


def test_span_violations_are_sorted_and_carry_identity():
    us = units_from([("a", 0.0, 1.0), ("b", float("nan"), 2.0), ("c", 5.0, 4.0)])
    violations = preflight_units(us)
    assert [v.unit_index for v in violations] == sorted(
        v.unit_index for v in violations
    )
    assert violations[0].unit_id == "u1"


# ---------------------------------------------------------------- AD3-2/AD3-4


@pytest.mark.parametrize(
    "over,key",
    [
        ({"clause_ms": 0.0}, "clause_ms"),
        ({"clause_ms": -1.0}, "clause_ms"),
        ({"offline_ms": 0.0}, "offline_ms"),
        ({"vad_skip_ms": 0.0}, "vad_skip_ms"),
        ({"max_cue_s": -1.0}, "max_cue_s"),
        ({"max_line_length": 0}, "max_line_length"),
        ({"max_lines": 0}, "max_lines"),
    ],
)
def test_profile_preflight_rejects_an_uninterpretable_profile(over, key):
    violations = preflight_profile(profile(**over))
    assert [v.key for v in violations] == [key]


def test_profile_preflight_accepts_a_zero_cap_and_a_disabled_snap():
    """AD3-4: zero DISABLES the cap; only a negative value is refused."""
    assert preflight_profile(profile(max_cue_s=0.0, shot_snap_s=0.0)) == ()


def test_a_healthy_profile_passes():
    assert preflight_profile(profile()) == ()
    assert preflight_profile(profile("ja", max_line_length=18, max_lines=1)) == ()


# ---------------------------------------------------------------- atom layer


def test_build_atom_layer_uses_the_documents_own_text():
    doc = document(timed(["the", "cat", "sat"]))
    layer = build_atom_layer(doc)
    assert isinstance(layer, AtomLayer)
    assert [a.text for a in layer.atoms] == ["the", "cat", "sat"]
    assert [(a.unit_start, a.unit_end) for a in layer.atoms] == [(0, 1), (1, 2), (2, 3)]
    assert layer.unit_count == 3
    assert layer.unit_bound(0) == 0 and layer.unit_bound(3) == 3


def test_build_atom_layer_refuses_to_rejoin_surfaces():
    """AD-6: the shadow reads ``text`` from the document, never re-derives it."""
    doc = document(timed(["the", "cat"]))
    doc.text = None
    with pytest.raises(ValueError):
        build_atom_layer(doc)


def test_display_projection_uses_the_joined_stream_normalization():
    """``[.,](?!\\d)`` is context-sensitive: the dot of ``3.75`` survives, a
    sentence-final one does not. Only the JOINED stream sees that context."""
    prof = profile("ja", max_line_length=18, max_lines=1)
    digits = build_atom_layer(document(timed(["3", ".", "75"]), prof=prof))
    assert "".join(a.display for a in digits.atoms) == "3.75"
    prose = build_atom_layer(document(timed(["あ", "。", "い"]), prof=prof))
    assert [a.display for a in prose.atoms] == ["あ", "", "い"]


def test_band_atoms_is_the_resolved_hard_band():
    assert band_atoms(profile()) == 2 * 42 + 1
    assert band_atoms(profile("ja", max_line_length=18, max_lines=1)) == (
        _line_budget_width(18, "ja") + 1
    )


def test_influence_radius_is_the_declared_locality_cell():
    assert INFLUENCE_RADIUS_UNITS == 96


# ---------------------------------------------------------------- AD3-1 coalescing


def test_trailing_invisible_run_folds_into_the_preceding_visible_atom():
    doc = document(timed(["the", "cat", ".", "!", "sat"]))
    layer = build_atom_layer(doc)
    result = coalesce_zero_display(layer.atoms)
    # the merged surface is the language join of its members, so the interval's
    # text still reconstructs byte for byte
    assert [a.text for a in result.atoms] == ["the", "cat . !", "sat"]
    assert [a.display for a in result.atoms] == ["the", "cat", "sat"]
    assert [(a.unit_start, a.unit_end) for a in result.atoms] == [
        (0, 1),
        (1, 4),
        (4, 5),
    ]
    assert result.atoms[1].members == (1, 2, 3)
    assert result.coalesced_atoms == 2
    assert result.all_invisible is False


def test_leading_invisible_run_folds_into_the_first_visible_atom():
    doc = document(timed([".", "!", "the", "cat"]))
    layer = build_atom_layer(doc)
    result = coalesce_zero_display(layer.atoms)
    assert [a.text for a in result.atoms] == [". ! the", "cat"]
    assert [(a.unit_start, a.unit_end) for a in result.atoms] == [(0, 3), (3, 4)]
    assert result.atoms[0].members == (0, 1, 2)
    assert result.coalesced_atoms == 2


def test_coalescing_conserves_ownership_exactly():
    doc = document(timed([".", "the", "!", "cat", ".", ",", "sat"]))
    layer = build_atom_layer(doc)
    result = coalesce_zero_display(layer.atoms)
    bounds = [(a.unit_start, a.unit_end) for a in result.atoms]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == layer.unit_count
    for left, right in zip(bounds, bounds[1:]):
        assert left[1] == right[0]
    assert sum(len(a.members) for a in result.atoms) == len(layer.atoms)


def test_coalescing_keeps_the_visible_carriers_penalties_and_spans():
    doc = document(timed(["the", "cat", "."], dur=0.4, gap=0.1))
    layer = build_atom_layer(doc)
    carrier = layer.atoms[1]
    merged = coalesce_zero_display(layer.atoms).atoms[1]
    assert merged.end_pen == carrier.end_pen
    assert merged.start_pen == carrier.start_pen
    assert merged.boundary_pen == carrier.boundary_pen
    assert merged.start == span_min([carrier.start, layer.atoms[2].start])
    assert merged.end == span_max([carrier.end, layer.atoms[2].end])


def test_every_coalesced_atom_has_positive_display_progress():
    """The invariant the AD4-2 work bound rests on."""
    doc = document(timed([".", "the", "!", "cat", ",", "sat", "."]))
    result = coalesce_zero_display(build_atom_layer(doc).atoms)
    assert all(_vis_width(a.display) >= 1 for a in result.atoms)


def test_an_all_invisible_run_folds_nothing_and_says_so():
    doc = document(timed(["!", "!", "!"]))
    result = coalesce_zero_display(build_atom_layer(doc).atoms)
    assert result.all_invisible is True
    assert result.coalesced_atoms == 0
    assert len(result.atoms) == 3


# ---------------------------------------------------------------- C4 barriers


def test_robust_silence_barrier_needs_the_full_uncertainty_band():
    assert BARRIER_UNCERTAINTY_MS == 50.0
    prof = profile()
    threshold_s = (prof.vad_skip_ms + BARRIER_UNCERTAINTY_MS) / 1000.0
    over = document([("a", 0.0, 0.5), ("b", 0.5 + threshold_s, 2.0)])
    under = document([("a", 0.0, 0.5), ("b", 0.5 + threshold_s - 0.01, 2.0)])
    kinds_over = [b.kind for b in build_barriers(build_atom_layer(over), prof)]
    kinds_under = [b.kind for b in build_barriers(build_atom_layer(under), prof)]
    assert kinds_over == ["document", "robust-silence", "document"]
    assert kinds_under == ["document", "document"]


def test_a_barrier_needs_both_bounds_real():
    prof = profile()
    doc = document([("a", 0.0, None), ("b", 3.0, 4.0)])
    assert [b.kind for b in build_barriers(build_atom_layer(doc), prof)] == [
        "document",
        "document",
    ]


def test_barriers_are_sorted_typed_and_speak_unit_ids():
    prof = profile()
    doc = document([("a", 0.0, 0.5), ("b", 2.0, 2.5), ("c", 4.0, 4.5)])
    barriers = build_barriers(build_atom_layer(doc), prof)
    assert [b.node for b in barriers] == sorted(b.node for b in barriers)
    assert all(isinstance(b, HardBarrier) and b.kind in BARRIER_KINDS for b in barriers)
    assert [b.unit_id for b in barriers] == [0, 1, 2, 3]
    assert barriers[0].gap_ms is None and barriers[-1].gap_ms is None
    assert barriers[1].gap_ms == pytest.approx(1500.0)


def test_intervals_tile_the_unit_stream():
    prof = profile()
    doc = document([("a", 0.0, 0.5), ("b", 2.0, 2.5), ("c", 4.0, 4.5)])
    layer = build_atom_layer(doc)
    intervals = build_intervals(layer, build_barriers(layer, prof))
    assert [(i.unit_start, i.unit_end) for i in intervals] == [(0, 1), (1, 2), (2, 3)]
    assert [i.index for i in intervals] == [0, 1, 2]


# ---------------------------------------------------------------- C1 nodes/seam


def test_candidate_nodes_are_language_conditioned():
    doc = document(timed(["one", "two", "three"]))
    spaced = candidate_nodes(build_atom_layer(doc).atoms, "en")
    assert spaced == (0, 1, 2, 3)

    prof = profile("ja", max_line_length=18, max_lines=1)
    ja_doc = document(timed(list("これは本です")), prof=prof)
    ja_atoms = build_atom_layer(ja_doc).atoms
    nodes = candidate_nodes(ja_atoms, "ja")
    assert nodes[0] == 0 and nodes[-1] == len(ja_atoms)
    assert set(nodes) <= set(range(len(ja_atoms) + 1))
    assert all(
        ja_atoms[n].phrase_start or ja_atoms[n].forced_boundary for n in nodes[1:-1]
    )


def test_split_candidate_at_unit_splits_by_footprint():
    us = units_from([("ab", 0.0, 0.5), ("cd", 1.0, 1.5)])
    parent = atom(
        0,
        "abcd",
        start=0.0,
        end=1.5,
        unit_start=0,
        unit_end=2,
        end_pen=3,
        start_pen=2,
        boundary_pen=1,
    )
    left, right = split_candidate_at_unit([parent], 1, us, "ja")
    assert (left.text, right.text) == ("ab", "cd")
    assert (left.unit_start, left.unit_end) == (0, 1)
    assert (right.unit_start, right.unit_end) == (1, 2)
    assert (left.start, left.end) == (0.0, 0.5)
    assert (right.start, right.end) == (1.0, 1.5)
    # penalty transport: the exposed edge carries no linguistic evidence
    assert (left.start_pen, left.boundary_pen, left.end_pen) == (2, 1, 0)
    assert (right.start_pen, right.boundary_pen, right.end_pen) == (0, 0, 3)
    assert right.phrase_start is True
    assert (left.index, right.index) == (0, 1)


def test_split_candidate_at_unit_refuses_an_already_exposed_boundary():
    us = units_from([("ab", 0.0, 0.5), ("cd", 1.0, 1.5)])
    atoms = [atom(0, "ab", start=0.0, end=0.5), atom(1, "cd", start=1.0, end=1.5)]
    with pytest.raises(ValueError):
        split_candidate_at_unit(atoms, 1, us, "ja")


# ---------------------------------------------------------------- C6 packer


def batch_measure(texts, lang, max_line_length, max_lines):
    stripped = strip_punct_for_subtitles(_join(list(texts), lang))
    lines = split_subtitle(stripped, max_line_length, lang).split("\n")
    return (
        _fits_budget(stripped, max_line_length, max_lines, lang),
        len(lines),
        tuple(_vis_width(line) for line in lines),
        stripped,
    )


@pytest.mark.parametrize("lang", ["en", "fr", "de", "es"])
def test_spaced_incremental_packer_matches_batch_on_every_prefix(lang):
    """Round-2 differential: the unchanged spaced-language fold stays exact."""
    vocab = [
        "a",
        "alpha",
        "café",
        "naïve",
        "über",
        "mañana",
        "3",
        ".",
        ",",
        "75",
        "e.g.",
        "10,000",
        "embedded  whitespace",
        "tab\tinside",
        "!",
        "?",
    ]
    rng = random.Random(f"spaced-packer:{lang}")
    checked = 0
    for _ in range(96):
        mll = rng.choice([8, 12, 18, 42])
        mlines = rng.choice([1, 2, 3])
        texts = [rng.choice(vocab) for _ in range(rng.randint(1, 64))]
        packer = IncrementalPacker(lang, mll, mlines)
        for k, text in enumerate(texts, 1):
            measure = packer.extend(text)
            fits, lines, widths, stripped = batch_measure(texts[:k], lang, mll, mlines)
            assert measure.fits is fits, (texts[:k], measure)
            assert measure.text == stripped
            assert measure.lines == lines
            assert measure.line_widths == widths
            checked += 1
    assert checked >= 2_500


def test_real_lattice_admission_matches_canonical_kinsoku_gap_pullback() -> None:
    """N14 both directions: a normalized gap may vanish during kinsoku."""
    source = "テスト「）kgGPT（000あ丙000甲kg3丁!」0003?丁これはテスト%（」あ%あ」乙「"
    assert "丁 」0003" in strip_punct_for_subtitles(source)
    prof = profile("ja", max_line_length=18, max_lines=2, max_cue_s=0.0)
    lattice = build_document_lattice(
        document(timed(list(source), dur=0.05), prof=prof, text=source)
    ).lattices[0]
    assert len(lattice.atoms) == 32

    actual = {(edge.start_node, edge.end_node) for edge in lattice.edges}
    expected: set[tuple[int, int]] = set()
    finals = {}
    for position, start in enumerate(lattice.nodes):
        for end in lattice.nodes[position + 1 :]:
            chunk = lattice.atoms[start:end]
            raw = _join([atom.text for atom in chunk], "ja")
            final = canonical_text(
                [
                    {"text": atom.text, "start": atom.start, "end": atom.end}
                    for atom in chunk
                ],
                fallback_text=raw,
                lang="ja",
                profile=prof,
                expected_footprint=raw,
            )
            finals[(start, end)] = final
            if canonical_legal(final, prof):
                expected.add((start, end))

    assert actual == expected
    full_span = (0, len(lattice.atoms))
    final = finals[full_span]
    assert canonical_legal(final, prof) is True
    assert final.cell_widths == (36, 35)
    assert final.lines[0].endswith("」")
    assert final.lines[1].startswith("0003")
    full_edge = next(
        edge for edge in lattice.edges if (edge.start_node, edge.end_node) == full_span
    )
    assert full_edge.line_widths == final.cell_widths
    assert lattice.packer_steps == 0
    assert lattice.canonical_chars > 0


def test_packer_counts_its_own_steps_and_reset_keeps_the_counter():
    packer = IncrementalPacker("en", 42, 2)
    for text in ["a", "b", "c"]:
        packer.extend(text)
    assert packer.steps == 3
    packer.reset()
    packer.extend("d")
    assert packer.steps == 4


def test_packer_balance_is_the_two_line_width_difference():
    packer = IncrementalPacker("en", 10, 2)
    measure = None
    for text in ["aaaa", "bbbb", "cccc"]:
        measure = packer.extend(text)
    assert measure.lines == 2
    assert measure.balance == pytest.approx(
        abs(measure.line_widths[0] - measure.line_widths[1])
    )


# ---------------------------------------------------------------- held chain


def test_held_chain_constant_is_the_inherited_one():
    assert HELD_CHAIN_MAX_GAP_S == HELD_WORD_MAX_GAP_S


def test_held_chain_walks_timed_units_in_start_order():
    continuous = units_from([("a", 0.0, 3.0), ("b", 3.5, 8.0)])
    broken = units_from([("a", 0.0, 0.5), ("b", 8.0, 8.5)])
    assert held_chain_continuous(continuous, 0, 2) is True
    assert held_chain_continuous(broken, 0, 2) is False


def test_a_single_timed_unit_is_trivially_a_held_span():
    """AD6-1 fixture 2's premise."""
    assert held_chain_continuous(units_from([("a", 0.0, 8.0)]), 0, 1) is True


def test_no_timed_unit_is_no_evidence():
    assert held_chain_continuous(units_from([("a", None, None)]), 0, 1) is False


# ---------------------------------------------------------------- AD5-1 greedy


def test_nine_bang_family_greedily_splits_into_two_cap_legal_chunks():
    """The round-5 adversarial family: nine zero-length ``!`` atoms 1.01 s apart."""
    atoms = [atom(i, "!", start=1.01 * i, end=1.01 * i) for i in range(9)]
    cuts = greedy_cap_partition(atoms, 7.0)
    assert cuts == (7,)
    assert (atoms[6].end - atoms[0].start) <= 7.0 + CAP_EPS_S
    assert (atoms[8].end - atoms[7].start) <= 7.0 + CAP_EPS_S


def test_greedy_cap_partition_is_a_noop_when_the_cap_is_disabled():
    atoms = [atom(i, "!", start=1.01 * i, end=1.01 * i) for i in range(9)]
    assert greedy_cap_partition(atoms, 0.0) == ()


def test_greedy_cap_partition_never_cuts_on_an_unresolvable_span():
    atoms = [atom(i, "!", start=None, end=None) for i in range(9)]
    assert greedy_cap_partition(atoms, 7.0) == ()


# ---------------------------------------------------------------- AD6-1 ordering


def test_over_cap_single_atom_takes_unit_relief_before_any_waiver():
    """AD6-1 fixture 1: discontinuous, individually cap-legal units."""
    us = units_from([("!", 0.0, 0.5), ("!", 8.0, 8.5)])
    atoms = [atom(0, "!!", start=0.0, end=8.5, unit_start=0, unit_end=2)]
    result = resolve_cap_partition(atoms, us, max_cue_s=7.0)
    assert result.infeasible is None
    assert result.waivers == ()
    assert result.relief_injections == 1
    assert len(result.atoms) == 2
    assert result.cuts == (1,)


def test_over_cap_single_unit_atom_gets_the_held_chain_waiver():
    """AD6-1 fixture 2: no internal edge exists, one timed span is held."""
    us = units_from([("!", 0.0, 8.0)])
    atoms = [atom(0, "!", start=0.0, end=8.0)]
    result = resolve_cap_partition(atoms, us, max_cue_s=7.0)
    assert result.infeasible is None
    assert result.cuts == ()
    assert result.relief_injections == 0
    assert len(result.waivers) == 1
    waiver = result.waivers[0]
    assert waiver.kind == "held-chain-duration"
    assert waiver.unit_ids == (0,)
    assert waiver.span == (0.0, 8.0)
    assert waiver.cap == 7.0


def test_duration_unwaivable_is_a_defensive_terminal_reached_only_by_injection(
    monkeypatch,
):
    """AD6-1: unreachable from a constructible document, so the CODE PATH is
    asserted by contradicting the held-chain evidence directly."""
    import voxweave.core.boundary_lattice as mod

    monkeypatch.setattr(mod, "held_chain_continuous", lambda *a, **k: False)
    us = units_from([("!", 0.0, 8.0)])
    atoms = [atom(0, "!", start=0.0, end=8.0)]
    result = resolve_cap_partition(atoms, us, max_cue_s=7.0)
    assert result.waivers == ()
    assert result.infeasible is not None
    assert result.infeasible.reason == "duration-unwaivable"
    assert result.infeasible.reason in INFEASIBLE_REASONS


def test_relief_split_pieces_that_still_exceed_the_cap_fall_back_to_the_waiver():
    us = units_from([("!", 0.0, 8.0), ("!", 8.2, 9.0)])
    atoms = [atom(0, "!!", start=0.0, end=9.0, unit_start=0, unit_end=2)]
    result = resolve_cap_partition(atoms, us, max_cue_s=7.0)
    assert result.relief_injections == 1
    assert result.infeasible is None
    assert [w.kind for w in result.waivers] == ["held-chain-duration"]


# ---------------------------------------------------------------- C16 relief


def test_relief_trigger_keeps_v1s_char_versus_cell_quirk():
    assert RELIEF_TRIGGER_FACTOR == 1.5
    atoms = [atom(i, "あ") for i in range(28)]
    assert relief_trigger(atoms, max_line_length=18, max_lines=1) is True
    assert relief_trigger(atoms[:27], max_line_length=18, max_lines=1) is False


def test_relief_nodes_land_on_budget_multiples_and_break_ties_low():
    atoms = [atom(i, "あ") for i in range(30)]
    assert relief_nodes(atoms, max_line_length=18, max_lines=1) == (18,)
    assert relief_nodes(atoms, max_line_length=6, max_lines=1) == (6, 12, 18, 24)


def test_relief_nodes_never_include_the_interval_ends():
    atoms = [atom(i, "あ") for i in range(18)]
    assert 0 not in relief_nodes(atoms, max_line_length=18, max_lines=1)
    assert 18 not in relief_nodes(atoms, max_line_length=18, max_lines=1)


def test_relief_regression_a_boundaryless_run_still_has_a_legal_path():
    """The defect the semantic path lacked, reproduced on ``"あ" * 200``."""
    prof = profile("ja", max_line_length=18, max_lines=1)
    doc = document(timed(["あ"] * 200, dur=0.1), prof=prof)
    built = build_document_lattice(doc)
    assert len(built.lattices) == 1
    lattice = built.lattices[0]
    assert lattice.infeasible is None
    assert reachable(lattice)


# ---------------------------------------------------------------- interval build


def test_a_span_violation_makes_only_its_own_interval_infeasible():
    prof = profile()
    spec = [("a", 0.0, 0.5), ("b", 3.0, 3.5), ("c", float("nan"), 7.0)]
    built = build_document_lattice(document(spec, prof=prof))
    reasons = [
        None if lattice.infeasible is None else lattice.infeasible.reason
        for lattice in built.lattices
    ]
    assert reasons.count("span-preflight") == 1
    assert reasons.count(None) == len(reasons) - 1


def test_all_invisible_interval_emits_a_forced_chain_with_zero_packing():
    """AD4-1: constant work, no layout search, ownership by fiat."""
    doc = document([("!", 0.0, 0.0)] * 2000)
    built = build_document_lattice(doc)
    assert len(built.lattices) == 1
    lattice = built.lattices[0]
    assert isinstance(lattice, IntervalLattice)
    assert lattice.all_invisible is True
    assert lattice.coalesced_atoms == 0
    assert lattice.packer_steps == 0
    assert len(lattice.edges) == 1
    assert (lattice.edges[0].start_node, lattice.edges[0].end_node) == (0, 2000)
    assert lattice.infeasible is None


def test_all_invisible_over_cap_interval_becomes_a_cap_legal_chain():
    spec = [("!", round(1.01 * i, 4), round(1.01 * i, 4)) for i in range(9)]
    built = build_document_lattice(document(spec))
    lattice = built.lattices[0]
    assert lattice.all_invisible is True
    assert [(e.start_node, e.end_node) for e in lattice.edges] == [(0, 7), (7, 9)]
    for edge in lattice.edges:
        assert edge.span_end - edge.span_start <= 7.0 + CAP_EPS_S


def test_mixed_interval_edges_are_sorted_and_layout_bounded():
    doc = document(timed(["word"] * 40, dur=0.05, gap=0.0))
    lattice = build_document_lattice(doc).lattices[0]
    keys = [(e.start_node, e.end_node) for e in lattice.edges]
    assert keys == sorted(keys)
    assert lattice.all_invisible is False
    band = band_atoms(doc.profile)
    assert all(e.end_node - e.start_node <= band for e in lattice.edges)
    assert reachable(lattice)


def test_edges_respect_the_duration_cap():
    doc = document(timed(["word"] * 12, dur=1.0, gap=0.0, start=0.0))
    lattice = build_document_lattice(doc).lattices[0]
    for edge in lattice.edges:
        assert edge.span_end - edge.span_start <= doc.profile.max_cue_s + CAP_EPS_S


def test_a_disabled_cap_removes_the_duration_predicate():
    """AD-7: ``max_cue_s <= 0`` disables the cap, no waiver machinery."""
    prof = profile(max_cue_s=0.0)
    doc = document(timed(["word"] * 12, dur=1.0, gap=0.0), prof=prof)
    lattice = build_document_lattice(doc).lattices[0]
    assert lattice.waivers == ()
    longest = max(e.end_node - e.start_node for e in lattice.edges)
    assert longest > 7  # duration no longer truncates the layout-legal span


# ---------------------------------------------------------------- sentences


def test_sentence_end_nodes_mirror_the_segmenter_and_record_misses():
    doc = document(timed(["one", "two.", "three", "four."]))
    ends = sentence_end_nodes(build_atom_layer(doc))
    assert ends.nodes == frozenset({2})
    assert isinstance(ends.missed, int) and ends.missed >= 0


# ---------------------------------------------------------------- helpers


def test_span_helpers_propagate_none_only_when_nothing_is_timed():
    assert span_min([None, 2.0, 1.0]) == 1.0
    assert span_max([None, 2.0, 1.0]) == 2.0
    assert span_min([None, None]) is None
    assert span_max([]) is None
    assert span_min([0.0]) == 0.0


# ------------------------------------------------- C1 unit-edge node space


def test_build_atoms_really_does_subdivide_a_source_unit():
    """The premise of the whole unit-edge restriction, stated as a fact.

    If this ever stops holding the restriction becomes a no-op rather than a
    bug, but it holds for every granularity the repo actually ships: word-level
    ja/zh aligner output and coarse ``word_data`` in any language.
    """
    ja = build_atom_layer(
        document(
            [("こんにちは", 0.0, 0.6), ("世界", 0.7, 1.1)],
            prof=profile("ja", max_line_length=18, max_lines=1),
        )
    )
    assert ja.unit_count == 2
    assert len(ja.atoms) == 7
    assert [a.unit_start for a in ja.atoms] == [0, 0, 0, 0, 0, 1, 1]

    en = build_atom_layer(document([("the quick brown", 0.0, 1.2), ("fox", 1.4, 1.9)]))
    assert [a.unit_start for a in en.atoms] == [0, 0, 0, 1]


def test_unit_edge_nodes_keeps_only_representable_cuts():
    atoms = [
        atom(0, "こ", unit_start=0, unit_end=1),
        atom(1, "ん", unit_start=0, unit_end=1),
        atom(2, "世", unit_start=1, unit_end=2),
        atom(3, "界", unit_start=1, unit_end=2),
    ]
    assert unit_edge_nodes(atoms) == (0, 2, 4)
    assert unit_edge_nodes([]) == (0,)
    tiling = [atom(i, "a") for i in range(3)]
    assert unit_edge_nodes(tiling) == (0, 1, 2, 3)


def test_candidate_nodes_never_offers_a_cut_inside_a_source_unit():
    """A break the partition could not express is not a break, however good."""
    atoms = [
        atom(0, "the", unit_start=0, unit_end=1),
        atom(1, "quick", unit_start=0, unit_end=1),
        atom(2, "brown", unit_start=0, unit_end=1),
        atom(3, "fox", unit_start=1, unit_end=2),
    ]
    assert candidate_nodes(atoms, "en") == (0, 3, 4)
    # where the aligner emitted one unit per atom the restriction removes nothing
    plain = [atom(i, "a") for i in range(4)]
    assert candidate_nodes(plain, "en") == (0, 1, 2, 3, 4)


def test_greedy_cap_partition_cuts_at_the_latest_unit_edge_not_the_atom_edge():
    """Over-cap at an interior atom falls back to the last representable edge."""
    atoms = [
        atom(0, "a", start=0.0, end=0.1, unit_start=0, unit_end=1),
        atom(1, "b", start=0.2, end=0.3, unit_start=1, unit_end=2),
        atom(2, "c", start=9.0, end=9.1, unit_start=1, unit_end=2),
    ]
    # atom 2 blows the cap, but node 2 is interior to unit 1: cut at node 1.
    assert greedy_cap_partition(atoms, 1.0) == (1,)
    # with nowhere legal to cut, no cut is invented -- the duration ladder runs.
    stuck = [
        atom(0, "a", start=0.0, end=0.1, unit_start=0, unit_end=1),
        atom(1, "b", start=9.0, end=9.1, unit_start=0, unit_end=1),
    ]
    assert greedy_cap_partition(stuck, 1.0) == ()


def test_relief_nodes_are_drawn_from_unit_edges_only():
    shared = [atom(i, "aaaa", unit_start=0, unit_end=1) for i in range(6)]
    assert relief_nodes(shared, max_line_length=4, max_lines=1) == ()
    split = [
        atom(i, "aaaa", unit_start=0 if i < 3 else 1, unit_end=1 if i < 3 else 2)
        for i in range(6)
    ]
    assert set(relief_nodes(split, max_line_length=4, max_lines=1)) <= {3}


def test_exclusively_owned_sees_a_shared_unit_on_either_side():
    shared = [
        atom(0, "う", unit_start=0, unit_end=1),
        atom(1, "う、", unit_start=0, unit_end=2),
        atom(2, "！", unit_start=2, unit_end=3),
    ]
    assert exclusively_owned(shared, 0) is False
    assert exclusively_owned(shared, 1) is False
    assert exclusively_owned(shared, 2) is True
    assert all(exclusively_owned([atom(i, "a") for i in range(3)], i) for i in range(3))


def test_relief_never_remints_a_unit_a_sibling_atom_still_holds():
    """The duplication bug, pinned end to end on the shape that produces it.

    A multi-glyph unit followed by punctuation coalesces into ``う`` +
    ``う、、！、``, whose footprint starts on the unit its left neighbour already
    carries. ``split_candidate_at_unit`` mints pieces from the *unit* stream, so
    relieving that atom re-emits the whole ``うう`` surface beside the sibling
    holding its first glyph -- the lattice then carries one more ``う`` than the
    document does, and the validator correctly calls it a conservation failure.

    Refusing relief here costs nothing that was available: the interval has no
    representable cut, so it reports the typed duration fallback instead.
    """
    doc = document(
        [("うう", 0.0, 0.3), ("、", 1.0, 1.1), ("、", None, None), ("！", 3.5, 3.7)],
        prof=profile("ja", max_line_length=18, max_lines=1, max_cue_s=3.0),
    )
    lattice = build_document_lattice(doc).lattices[0]
    assert lattice.relief_injections == 0
    assert "".join(a.text for a in lattice.atoms) == doc.text
    assert lattice.infeasible is not None
    assert lattice.infeasible.reason in INFEASIBLE_REASONS


def test_a_word_level_ja_document_conserves_its_text_end_to_end():
    """The live-shaped regression: word units, per-glyph atoms, zero violations."""
    doc = document(
        [
            ("こんにちは", 0.0, 0.6),
            ("世界", 0.7, 1.1),
            ("元気", 1.3, 1.7),
            ("ですか", 1.8, 2.3),
            ("今日", 2.5, 2.9),
        ],
        prof=profile("ja", max_line_length=8, max_lines=1),
    )
    built = build_document_lattice(doc)
    for lattice in built.lattices:
        legal = set(unit_edge_nodes(lattice.atoms))
        assert set(lattice.nodes) <= legal
        for edge in lattice.edges:
            assert edge.start_node in legal and edge.end_node in legal


# ------------------------------------------- AD6-1 duration ladder, in order


def test_a_cap_splittable_run_exposes_its_cut_instead_of_claiming_unwaivable():
    """Bug pin: a SUCCESSFUL cap resolution used to be read as the terminal.

    ``resolve_cap_partition`` returns ``waivers=()`` in two very different
    cases -- the run split cleanly into cap-legal chunks, and the run could not
    be split at all -- and the mixed branch used to treat the empty waiver list
    as ``duration-unwaivable``. That skipped the held-chain test AD6-1 orders
    before the terminal, short-circuited the C16 relief valve (which only runs
    where no ``infeasible`` was set), and falsified AD6-1's recorded conclusion
    that the terminal is unreachable from a constructible document.

    Twelve one-glyph ja units: the interval spans 8 s against a 7 s cap, so no
    single cue is legal, but a cut at unit 11 makes both halves legal -- and unit
    11 is a unit edge the no-space candidate set had hidden.
    """
    spec = [
        (ch, i * 0.55, i * 0.55 + 0.4) for i, ch in enumerate("あいうえおかきくけこさ")
    ]
    spec.append(("し", 6.4, 8.0))
    prof = profile("ja", max_line_length=18, max_lines=2, max_cue_s=7.0)
    doc = document(spec, prof=prof)
    lattice = build_document_lattice(doc).lattices[0]

    assert candidate_nodes(lattice.atoms, "ja") == (0, 12)
    assert lattice.infeasible is None
    assert lattice.cap_relief_nodes >= 1
    assert 11 in lattice.nodes
    assert reachable(lattice)
    for edge in lattice.edges:
        low, high = edge.span_start, edge.span_end
        assert low is None or high is None or high - low <= prof.max_cue_s + CAP_EPS_S


def test_relief_never_reintroduces_a_zero_display_atom_into_a_mixed_interval():
    """AD3-1 after relief: no invisible atom, and no empty-display edge.

    Relief splits by footprint, so a piece can own only punctuation and render
    to nothing -- which breaks the positive-display-progress invariant the band
    and the early break are proved from, and admits a candidate cue that renders
    to the empty string. Where re-coalescing folds the split straight back,
    relief simply was not available and the duration ladder falls through to the
    held-chain waiver, which is the AD6-1 ordering anyway.
    """
    doc = document(
        [
            ("あ", 0.0, 8.0),
            ("。", 8.0, 8.05),
            ("い", 8.2, 8.5),
            ("う", 8.6, 8.9),
            ("え", 9.0, 9.3),
        ],
        prof=profile("ja", max_line_length=18, max_lines=2, max_cue_s=7.0),
    )
    lattice = build_document_lattice(doc).lattices[0]

    assert lattice.all_invisible is False
    assert all(a.visible for a in lattice.atoms), [a.text for a in lattice.atoms]
    assert all(_vis_width(a.display) >= 1 for a in lattice.atoms)
    assert all(edge.display_text for edge in lattice.edges)
    assert [w.kind for w in lattice.waivers] == ["held-chain-duration"]
    assert lattice.infeasible is None


# ------------------------------------- coarse granularity (scope, not failure)


def coarse_ja_document():
    """Six sentence-level units, per-glyph atoms: candidates collapse to {0, N}."""
    sentences = [
        "これはテストです",
        "こんにちは世界",
        "今日はいい天気",
        "こんにちは世界",
        "こんにちは世界",
        "これはテストです",
    ]
    bounds = [
        (0.0, 1.16),
        (1.231, 2.793),
        (2.907, 4.171),
        (4.234, 4.909),
        (5.086, 6.588),
        (6.61, 7.451),
    ]
    return document(
        [(s, a, b) for s, (a, b) in zip(sentences, bounds)],
        prof=profile("ja", max_line_length=18, max_lines=2),
        text="".join(sentences),
    )


def test_a_collapsed_node_space_is_typed_coarse_granularity_not_no_path():
    """The class is a property of the INPUT, and says so.

    ``candidate_nodes`` intersects phrase starts with source-unit edges, and on a
    sentence-granularity ja stream the intersection is ``{0}``. No cue short
    enough to fit the budget can be expressed at all, so this is not a search
    that failed and must not be reported as one -- P5 resolves it by splitting
    below the source unit.
    """
    lattice = build_document_lattice(coarse_ja_document()).lattices[0]
    assert len(candidate_nodes(lattice.atoms, "ja")) == 2
    assert lattice.infeasible is not None
    assert lattice.infeasible.reason == "coarse-granularity"
    assert lattice.infeasible.reason in INFEASIBLE_REASONS
    # detected BEFORE solving: no edge scan ran
    assert lattice.edges == ()
    assert lattice.packer_steps == 0


def test_the_same_text_at_word_granularity_needs_no_fallback():
    """The minimal pair: identical text and timing, per-glyph units."""
    coarse = coarse_ja_document()
    spec = []
    for unit in coarse.units:
        step = (unit.end - unit.start) / len(unit.surface)
        for offset, ch in enumerate(unit.surface):
            spec.append(
                (ch, unit.start + offset * step, unit.start + (offset + 1) * step)
            )
    fine = document(spec, prof=coarse.profile, text=coarse.text)
    assert all(
        lattice.infeasible is None for lattice in build_document_lattice(fine).lattices
    )


def test_granularity_check_is_a_lower_bound_never_a_false_collapse():
    """A word-level document always has enough boundaries for its own width."""
    doc = document(
        timed(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"], dur=0.3),
        prof=profile(max_line_length=12, max_lines=1),
    )
    for lattice in build_document_lattice(doc).lattices:
        check = granularity_check(lattice.atoms, lattice.nodes, doc.profile)
        assert check.required_cuts >= 1
        assert check.collapsed is False
