"""The voiceprint-v1 clustering recipe, zero GPU, with a synthetic embedder.

The fake embedder knows who really speaks when (a ground-truth timeline) and
returns that speaker's base vector plus a small deterministic perturbation;
spans with nobody in them get a "noise" vector far from every speaker.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np
import pytest

from voxweave import speakercluster as vp

DIM = 16


def _basis(k: int) -> np.ndarray:
    v = np.zeros(DIM)
    v[k] = 1.0
    return v


SPEAKERS = {"alice": _basis(0), "bob": _basis(1), "carol": _basis(2), "dave": _basis(3)}
NOISE = _basis(15)


class FakeEmbed:
    """Truth = [(start, end, person)]; embedding = duration-weighted mix of who speaks."""

    def __init__(self, truth, jitter: float = 0.05):
        self.truth = truth
        self.jitter = jitter
        self.calls: list[list[tuple[float, float]]] = []

    def __call__(self, spans):
        self.calls.append(list(spans))
        rows = []
        for s, e in spans:
            acc = np.zeros(DIM)
            for a, b, who in self.truth:
                ov = min(e, b) - max(s, a)
                if ov > 0:
                    acc += ov * SPEAKERS[who]
            if not acc.any():
                acc = NOISE.copy()
            seed = int(round(s * 1000)) * 7919 + int(round(e * 1000))
            noise = np.random.default_rng(seed).normal(size=DIM)
            noise[15] = abs(noise[15])
            acc = acc / np.linalg.norm(acc) + self.jitter * noise / np.linalg.norm(
                noise
            )
            rows.append(acc / np.linalg.norm(acc))
        return np.array(rows).reshape(len(spans), DIM)


def _alternating(people, n_turns, length=3.0, gap=0.5, start=0.0):
    truth, t = [], start
    for k in range(n_turns):
        truth.append((t, t + length, people[k % len(people)]))
        t += length + gap
    return truth


def _labels(result):
    return [lab for _s, _e, lab in result.turns]


def _speakers(result) -> int:
    return len(set(_labels(result)))


def _partition(turns, truth):
    """Map each output label to the truth person it overlaps most."""
    owner = {}
    for s, e, lab in turns:
        best = max(truth, key=lambda t: min(e, t[1]) - max(s, t[0]))
        owner.setdefault(lab, set()).add(best[2])
    return owner


def _triangle():
    """Three voices, each pair talking over each other (75% of the shorter turn).

    Every speaker's anchors overlap anchors of both others, so cannot-link
    keeps all three clusters apart whatever their voiceprints say.
    """
    truth = []
    t = 0.0
    for _rep in range(4):
        for x, y in (("alice", "bob"), ("bob", "carol"), ("alice", "carol")):
            truth += [(t, t + 4.0, x), (t + 1.0, t + 5.0, y)]
            truth += [(t + 6.0, t + 9.0, x), (t + 10.0, t + 13.0, y)]
            t += 20.0
    turns = [(s, e, who[0].upper()) for s, e, who in truth]
    # The 1.0 s clean pieces of the overlapped turns sit on anchor_seconds.
    return truth, turns, {"anchor_seconds": 0.9}


# --------------------------------------------------------------------------
# Recipe identity
# --------------------------------------------------------------------------


def test_defaults_are_the_frozen_voiceprint_v1_recipe():
    # Any change here is a new recipe: give RECIPE a new id with it.
    assert vp.RECIPE == "voiceprint-v1"
    assert asdict(vp.ClusteringParams()) == {
        "method": "ahc",
        "anchor_seconds": 1.0,
        "min_piece_seconds": 0.3,
        "min_embed_seconds": 0.05,
        "cut_distance": 0.45,
        "cannot_link_overlap": 0.5,
        "spectral_pval": 0.012,
        "spectral_merge_similarity": 0.0,
        "spectral_max_speakers": 20,
        "merge_similarity": 0.55,
        "split_similarity": 0.3,
        "dissolve_seconds": 6.0,
        "local_seconds": 30.0,
        "local_turns": 15,
        "tau": 0.35,
        "margin": 0.1,
        "tau_global": 0.5,
        "margin_global": 0.1,
        "tau_floor": 0.1,
        "overlap_exclusion": True,
    }
    json.dumps(asdict(vp.ClusteringParams()), allow_nan=False)


# --------------------------------------------------------------------------
# Recipe behaviour
# --------------------------------------------------------------------------


def test_correct_input_is_kept_and_relabelled_by_first_appearance():
    truth = _alternating(["bob", "alice"], 20)
    turns = [(s, e, "X" if who == "bob" else "Y") for s, e, who in truth]
    result = vp.cluster_turns(turns, FakeEmbed(truth))
    assert _labels(result)[:2] == ["SPEAKER_00", "SPEAKER_01"]
    owners = _partition(result.turns, truth)
    assert owners == {"SPEAKER_00": {"bob"}, "SPEAKER_01": {"alice"}}
    assert result.audit["counts"]["clusters"] == 2
    assert result.audit["counts"]["turns_abstain"] == 0
    assert vp.PASSTHROUGH not in result.audit


def test_split_speaker_is_merged_and_short_turns_follow_voice():
    # pyannote split alice into a long-turn label A and a short-turn label S; S also
    # holds one of bob's backchannels.
    truth = _alternating(["alice", "bob"], 20)
    turns = [(s, e, "A" if who == "alice" else "B") for s, e, who in truth]
    extra_truth = []
    for k in range(10):
        t = 200.0 + 4 * k
        who = "bob" if k == 3 else "alice"
        extra_truth.append((t, t + 0.6, who))
        turns.append((t, t + 0.6, "S"))
    truth += extra_truth
    result = vp.cluster_turns(
        turns, FakeEmbed(truth), params=vp.ClusteringParams(dissolve_seconds=4.0)
    )
    owners = _partition(result.turns, truth)
    assert sorted(len(v) for v in owners.values()) == [1, 1]
    by_start = {round(s, 2): lab for s, _e, lab in result.turns}
    alice_label = by_start[0.0]
    bob_label = by_start[3.5]
    assert by_start[212.0] == bob_label  # the backchannel goes to bob, not to "S"
    assert by_start[200.0] == alice_label


def test_nonspeech_turns_are_abstained_and_dropped():
    truth = _alternating(["alice", "bob"], 16)
    turns = [(s, e, "A" if who == "alice" else "B") for s, e, who in truth]
    noise_turns = [(100.0 + 3 * k, 100.8 + 3 * k, "C") for k in range(6)]
    result = vp.cluster_turns(turns + noise_turns, FakeEmbed(truth))
    assert all(s < 100.0 for s, _e, _lab in result.turns)
    audit = result.audit
    assert audit["counts"]["turns_abstain"] == 6
    assert math.isclose(audit["abstained_seconds"], 6 * 0.8, rel_tol=1e-6)
    detailed = vp.assign_turns(turns + noise_turns, FakeEmbed(truth))
    assert detailed.labels[-6:] == [None] * 6
    assert detailed.routes[-6:] == [vp.ROUTE_ABSTAIN] * 6


def test_embed_receives_exactly_embedding_spans_in_one_call():
    truth = _alternating(["alice", "bob"], 8)
    turns = [(s, e, "A" if who == "alice" else "B") for s, e, who in truth]
    turns.append((1.0, 2.0, "B"))  # fully inside an A turn: no clean piece
    fake = FakeEmbed(truth)
    vp.cluster_turns(turns, fake)
    assert len(fake.calls) == 1
    assert fake.calls[0] == vp.embedding_spans(turns)
    assert (1.0, 2.0) in fake.calls[0]  # whole-span fallback for the overlapped turn
    assert (0.0, 1.0) in fake.calls[0] and (2.0, 3.0) in fake.calls[0]  # trimmed pieces


def test_cannot_link_keeps_overlapping_anchors_apart():
    # Same fake voice, but the two labels talk over each other for most of their turns.
    truth = [(0.0, 60.0, "alice")]
    turns = []
    for k in range(6):
        t = 10.0 * k
        turns.append((t, t + 6.0, "A"))
        turns.append((t + 2.0, t + 8.5, "B"))
    fake = FakeEmbed(truth)
    params = vp.ClusteringParams(
        anchor_seconds=1.0, dissolve_seconds=1.0, cannot_link_overlap=0.5
    )
    detailed = vp.assign_turns(turns, fake, params=params)
    a_labels = {
        lab
        for (s, e, orig), lab in zip(detailed.turns, detailed.labels)
        if orig == "A" and lab
    }
    b_labels = {
        lab
        for (s, e, orig), lab in zip(detailed.turns, detailed.labels)
        if orig == "B" and lab
    }
    assert detailed.audit["counts"]["cannot_link_pairs"] > 0
    # No cluster may hold two anchors that overlap by more than half the shorter one.
    anchors = [
        (s, e, lab)
        for (s, e, _o), lab, route in zip(
            detailed.turns, detailed.labels, detailed.routes
        )
        if route == vp.ROUTE_ANCHOR
    ]
    for i, (s1, e1, l1) in enumerate(anchors):
        for s2, e2, l2 in anchors[i + 1 :]:
            ov = min(e1, e2) - max(s1, s2)
            if ov > 0.5 * min(e1 - s1, e2 - s2):
                assert l1 != l2
    assert a_labels and b_labels


def test_speaker_bounds_clip_the_cluster_count():
    truth = _alternating(["alice", "bob", "carol", "dave"], 32)
    turns = [(s, e, who) for s, e, who in truth]
    fake = FakeEmbed(truth)
    free = vp.cluster_turns(turns, fake)
    assert free.audit["counts"]["clusters"] == 4
    capped = vp.cluster_turns(turns, fake, max_speakers=2)
    assert 1 <= capped.audit["counts"]["clusters"] <= 2
    assert capped.audit["speaker_bounds"]["satisfied"]
    loose = vp.ClusteringParams(cut_distance=1.9)
    merged = vp.cluster_turns(turns, fake, params=loose)
    assert merged.audit["counts"]["clusters"] == 1
    floor = vp.cluster_turns(turns, fake, params=loose, min_speakers=3)
    assert floor.audit["counts"]["clusters"] >= 3
    with pytest.raises(vp.ClusteringError):
        vp.cluster_turns(turns, fake, min_speakers=3, max_speakers=2)


@pytest.mark.parametrize("method", vp.METHODS)
def test_methods_are_deterministic_and_order_invariant(method):
    truth = _alternating(["alice", "bob", "carol"], 30)
    turns = [(s, e, who[0]) for s, e, who in truth]
    params = vp.ClusteringParams(method=method)
    one = vp.cluster_turns(turns, FakeEmbed(truth), params=params)
    two = vp.cluster_turns(list(reversed(turns)), FakeEmbed(truth), params=params)
    assert one.turns == two.turns
    assert one.audit == two.audit
    owners = _partition(one.turns, truth)
    assert sorted(len(v) for v in owners.values()) == [1, 1, 1]


def test_refine_splits_a_mixed_label():
    truth = _alternating(["alice", "bob", "carol"], 30)
    # pyannote put bob and carol under one label
    turns = [(s, e, "A" if who == "alice" else "M") for s, e, who in truth]
    params = vp.ClusteringParams(method="refine", split_similarity=0.5)
    result = vp.cluster_turns(turns, FakeEmbed(truth), params=params)
    owners = _partition(result.turns, truth)
    assert len(owners) == 3
    assert all(len(v) == 1 for v in owners.values())


def test_refine_merges_a_split_label():
    truth = _alternating(["alice", "bob"], 30)
    turns = [
        (s, e, ("A1" if k % 4 == 0 else "A2") if who == "alice" else "B")
        for k, (s, e, who) in enumerate(truth)
    ]
    params = vp.ClusteringParams(method="refine")
    result = vp.cluster_turns(turns, FakeEmbed(truth), params=params)
    assert result.audit["counts"]["clusters"] == 2


def test_empty_and_passthrough():
    empty = vp.cluster_turns([], FakeEmbed([]))
    assert empty.turns == [] and empty.audit["counts"]["turns"] == 0
    assert vp.PASSTHROUGH in empty.audit
    truth = [(0.0, 0.8, "alice"), (1.0, 1.6, "bob")]
    turns = [(0.0, 0.8, "p"), (1.0, 1.6, "q")]
    result = vp.cluster_turns(turns, FakeEmbed(truth))  # no anchor >= 1 s
    assert result.turns == [(0.0, 0.8, "SPEAKER_00"), (1.0, 1.6, "SPEAKER_01")]
    assert "no anchor cluster survived" in str(result.audit[vp.PASSTHROUGH])
    detailed = vp.assign_turns(turns, FakeEmbed(truth))
    assert detailed.routes == [vp.ROUTE_FALLBACK] * 2


def test_overlapped_turn_never_joins_the_overlapping_speakers_cluster():
    # bob's short interjection sits inside alice's long turn; the fake embedding
    # of that span is dominated by alice (it is the mixed audio), yet pyannote
    # said two voices, so the turn must not be merged into alice.
    truth = _alternating(["alice", "bob"], 20)
    turns = [(s, e, "A" if who == "alice" else "B") for s, e, who in truth]
    inner = (0.5, 1.2)
    truth_mix = truth + [(0.5, 0.7, "bob")]
    turns.append((*inner, "B"))
    params = vp.ClusteringParams(tau=0.0, margin=0.0, tau_global=0.0, tau_floor=-1.0)
    on = vp.assign_turns(turns, FakeEmbed(truth_mix), params=params)
    k = [t[:2] for t in on.turns].index(inner)
    alice = on.labels[[t[:2] for t in on.turns].index((0.0, 3.0))]
    assert on.labels[k] is not None and on.labels[k] != alice
    off = vp.assign_turns(
        turns,
        FakeEmbed(truth_mix),
        params=vp.ClusteringParams(
            tau=0.0, margin=0.0, tau_global=0.0, tau_floor=-1.0, overlap_exclusion=False
        ),
    )
    assert off.labels[k] == alice


def test_union_same_label_joins_overlaps_and_touching():
    turns = [
        (0.0, 2.0, "a"),
        (1.0, 3.0, "a"),
        (3.0, 4.0, "a"),
        (5.0, 6.0, "a"),
        (0.5, 1.0, "b"),
    ]
    assert vp.union_same_label(turns) == [
        (0.0, 4.0, "a"),
        (0.5, 1.0, "b"),
        (5.0, 6.0, "a"),
    ]


def test_overlap_trimmed_pieces():
    turns = [(0.0, 10.0, "a"), (2.0, 3.0, "b"), (9.8, 12.0, "b")]
    pieces = vp.overlap_trimmed_pieces(turns, 0.3)
    assert pieces[0] == [(0.0, 2.0), (3.0, 9.8)]
    assert pieces[1] == []
    assert pieces[2] == [(10.0, 12.0)]


def test_bad_inputs_raise():
    with pytest.raises(vp.ClusteringError):
        vp.ClusteringParams(method="kmeans")
    with pytest.raises(vp.ClusteringError):
        vp.ClusteringParams(tau=1.5)
    turns = [(0.0, 3.0, "a"), (4.0, 7.0, "b")]
    with pytest.raises(vp.ClusteringError):
        vp.cluster_turns(turns, lambda spans: np.zeros((len(spans) + 1, 4)))
    with pytest.raises(vp.ClusteringError):
        vp.cluster_turns(turns, lambda spans: np.full((len(spans), 4), np.nan))
    with pytest.raises(vp.ClusteringError):
        vp.cluster_turns([(0.0, math.inf, "a")], FakeEmbed([]))


def test_zero_embedding_rows_raise_instead_of_dropping_turns():
    # A zero row has no direction: accepting it would silently abstain the
    # turn (and drop it from the output) however long it is.
    truth = _alternating(["alice", "bob"], 12)
    turns = [(s, e, who) for s, e, who in truth]
    fake = FakeEmbed(truth)

    def some_zero(spans):
        rows = fake(spans)
        rows[[i for i, (s, _e) in enumerate(spans) if s >= 20.0]] = 0.0
        return rows

    with pytest.raises(vp.ClusteringError, match="all-zero row"):
        vp.cluster_turns(turns, some_zero)
    with pytest.raises(vp.ClusteringError, match="all-zero row"):
        vp.cluster_turns(turns, lambda spans: np.zeros((len(spans), 4)))


def test_local_assignment_prefers_nearby_clusters():
    # carol appears only early; a late short turn that sounds halfway between
    # alice and carol goes to alice, the only cluster active nearby.
    truth = _alternating(["alice", "carol"], 10) + _alternating(
        ["alice", "bob"], 20, start=200.0
    )
    turns = [(s, e, who) for s, e, who in truth]
    mixed = (300.0, 300.8)
    truth_mix = truth + [(300.0, 300.5, "alice"), (300.5, 300.8, "carol")]
    turns.append((*mixed, "alice"))
    params = vp.ClusteringParams(tau=0.3, margin=0.0)
    detailed = vp.assign_turns(turns, FakeEmbed(truth_mix), params=params)
    k = [t[:2] for t in detailed.turns].index(mixed)
    assert detailed.routes[k] == vp.ROUTE_LOCAL
    alice = detailed.labels[[t[:2] for t in detailed.turns].index((200.0, 203.0))]
    assert detailed.labels[k] == alice


def test_audit_is_json_serialisable():
    truth = _alternating(["alice", "bob"], 12)
    turns = [(s, e, who) for s, e, who in truth]
    result = vp.cluster_turns(turns, FakeEmbed(truth))
    text = json.dumps(result.audit, allow_nan=False)
    assert "anchor_seconds" in text
    assert result.audit["recipe"] == "voiceprint-v1"


# --------------------------------------------------------------------------
# Speaker bounds are hard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("max_speakers", [1, 2])
def test_max_speakers_blocked_by_cannot_link_raises(max_speakers):
    truth, turns, knobs = _triangle()
    params = vp.ClusteringParams(**knobs)
    free = vp.cluster_turns(turns, FakeEmbed(truth), params=params)
    assert _speakers(free) == 3
    assert free.audit["counts"]["cannot_link_pairs"] > 0
    fits = vp.cluster_turns(turns, FakeEmbed(truth), max_speakers=3, params=params)
    assert _speakers(fits) == 3

    # Every remaining merge joins overlapping anchors: no clustering has fewer
    # than three speakers, so the bound fails loudly instead of being ignored.
    with pytest.raises(vp.ClusteringError, match=f"max_speakers {max_speakers} "):
        vp.cluster_turns(
            turns, FakeEmbed(truth), max_speakers=max_speakers, params=params
        )
    with pytest.raises(vp.ClusteringError, match="cannot be met"):
        vp.assign_turns(
            turns, FakeEmbed(truth), max_speakers=max_speakers, params=params
        )


def test_refine_is_held_to_max_speakers_too():
    truth, turns, knobs = _triangle()
    params = vp.ClusteringParams(method="refine", **knobs)
    with pytest.raises(vp.ClusteringError, match=r"left 3 speaker\(s\), outside"):
        vp.cluster_turns(turns, FakeEmbed(truth), max_speakers=2, params=params)


@pytest.mark.parametrize("min_speakers", [5, 7])
def test_min_speakers_out_of_reach_raises_instead_of_passing_through(min_speakers):
    # Two voices, twelve 3 s anchors: a cluster needs two anchors (6 s) to
    # survive, and the merge levels give at most four such clusters. The old
    # walk stopped at zero merges, dissolved everything and silently handed
    # back pyannote's labels.
    truth = _alternating(["alice", "bob"], 12)
    turns = [(s, e, "A" if who == "alice" else "B") for s, e, who in truth]
    # A floor above the voices present is met by dividing them, as pyannote
    # does under the same bound (the recipe measured less harm doing so).
    reached = vp.cluster_turns(turns, FakeEmbed(truth), min_speakers=4)
    assert _speakers(reached) == 4
    assert vp.PASSTHROUGH not in reached.audit
    with pytest.raises(vp.ClusteringError, match=f"min_speakers {min_speakers} "):
        vp.cluster_turns(turns, FakeEmbed(truth), min_speakers=min_speakers)


def test_bounded_ahc_never_degrades_to_passthrough():
    # One 10 s voice and two 3.5 s voices too far apart to merge at the cut:
    # the cut keeps one cluster (the 10 s voice). Undoing merges only
    # dissolves it, so no level on the walk towards fewer merges has two
    # clusters. Merging the two small voices past the cut would give two, but
    # as one made-up speaker, and the min walk never merges more. This used
    # to come back as an unflagged-looking pass-through of pyannote's labels.
    vectors = {0.0: _basis(0), 6.0: _basis(0)}
    for start, sign in ((12.0, 1.0), (16.0, -1.0)):
        v = _basis(1) + sign * 0.8 * _basis(2)
        vectors[start] = v / np.linalg.norm(v)
    turns = [(0.0, 5.0, "a"), (6.0, 11.0, "a"), (12.0, 15.5, "b"), (16.0, 19.5, "c")]

    def embed(spans):
        return np.stack([vectors[start] for start, _end in spans])

    assert _speakers(vp.cluster_turns(turns, embed)) == 1
    with pytest.raises(
        vp.ClusteringError,
        match="min_speakers 2 cannot be met: from the cut down to no merges, at most 1 ",
    ):
        vp.cluster_turns(turns, embed, min_speakers=2)


def _bound_cases():
    triangle_truth, triangle_turns, knobs = _triangle()
    pair_truth = _alternating(["alice", "bob"], 12)
    pair_turns = [(s, e, "A" if w == "alice" else "B") for s, e, w in pair_truth]
    four_truth = _alternating(["alice", "bob", "carol", "dave"], 24)
    four_turns = [(s, e, w) for s, e, w in four_truth]
    return [
        (triangle_truth, triangle_turns, knobs),
        (pair_truth, pair_turns, {}),
        (four_truth, four_turns, {}),
    ]


@pytest.mark.parametrize("method", vp.METHODS)
def test_results_never_leave_the_speaker_bounds(method):
    bounds = [(lo, hi) for lo in (None, 1, 2, 3, 5) for hi in (None, 1, 2, 3, 6)]
    for truth, turns, knobs in _bound_cases():
        params = vp.ClusteringParams(method=method, **knobs)
        for lo, hi in bounds:
            if lo is not None and hi is not None and hi < lo:
                continue
            try:
                result = vp.cluster_turns(
                    turns,
                    FakeEmbed(truth),
                    min_speakers=lo,
                    max_speakers=hi,
                    params=params,
                )
            except vp.ClusteringError:
                continue
            if vp.PASSTHROUGH in result.audit:
                continue  # flagged: the caller keeps pyannote's answer
            count = _speakers(result)
            assert (lo or 1) <= count, (method, lo, hi, count)
            assert hi is None or count <= hi, (method, lo, hi, count)
            assert result.audit["speaker_bounds"]["satisfied"]
