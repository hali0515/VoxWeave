"""N17: the LAW section 9 typed policy-delta registry is immutable data."""

from __future__ import annotations

from voxweave.core.boundary_cost import POLICY_VERSION
from voxweave.core.policy_delta import DELTA_REGISTRY, delta_registry_bytes


EXPECTED_DELTA_REGISTRY_BYTES = (
    '[{"id":"FD-1","trigger":"canonical vs raw reading_chars differ",'
    '"affected_fields":["end"],"direction":"both",'
    '"allowed_relation":"delivered leg == classifier-recomputed phase-1 duration solve '
    '(independent reimplementation, mirror-pinned)","enforcement":"N11"},'
    '{"id":"FD-2","trigger":"evidence-span vs legacy display-span lyric '
    'classification differs","affected_fields":["lyric"],"direction":"both",'
    '"allowed_relation":"flag == EvidenceSpan predicate output","enforcement":"N11"},'
    '{"id":"FD-3","trigger":"input pair overlapped","affected_fields":["end"],'
    '"direction":"shrink","allowed_relation":"exact ladder-branch target (branch '
    'recomputed from evidence)","enforcement":"N11+N13"},'
    '{"id":"FD-4","trigger":"zone/separation outcome differs from legacy sequential '
    'sweeps","affected_fields":["start","end"],"direction":"both",'
    '"allowed_relation":"every trace leg = validator-recomputed rule application on ITS '
    'reconstructed state (§10.2)","enforcement":"N11 via trace"},'
    '{"id":"FD-5","trigger":"v2 lane never applies the speaker overlay",'
    '"affected_fields":["v2 construction"],"direction":"n/a",'
    '"allowed_relation":null,"enforcement":"construction + N3b"},'
    '{"id":"FD-6","trigger":"input gap < TWO_FRAME_S","affected_fields":["end"],'
    '"direction":"shrink","allowed_relation":"next_start − TWO_FRAME_S (b1) or '
    'speech_end + report (b2)","enforcement":"N11+N13"},'
    '{"id":"FD-7","trigger":"any veto/refusal fact","affected_fields":[],'
    '"direction":"n/a","allowed_relation":"report-only","enforcement":"report '
    'equality"},'
    '{"id":"FD-8","trigger":"anchorless cue where legacy would extend",'
    '"affected_fields":["end"],"direction":"both","allowed_relation":"delivered end == '
    'phase-1 input end or a composed FD-3/4/6 leg chain (each leg checked)",'
    '"enforcement":"N11 via trace"},'
    '{"id":"FD-9","trigger":"bounded stutter stable == False",'
    '"affected_fields":["text"],"direction":"n/a","allowed_relation":"to == '
    'bounded_canonical(raw)","enforcement":"N11 + injected fixture"},'
    '{"id":"PD-TEXT","trigger":"edge admission = canonical legality",'
    '"affected_fields":["lattice edge set"],"direction":"n/a",'
    '"allowed_relation":null,"enforcement":"N14 both-direction oracle"},'
    '{"id":"PD-SPK","trigger":"speaker edge term","affected_fields":["selection"],'
    '"direction":"n/a","allowed_relation":null,"enforcement":"speaker-off '
    'counterfactual"},'
    '{"id":"PD-SUBUNIT","trigger":"refinement of coarse units",'
    '"affected_fields":["unit space"],"direction":"n/a","allowed_relation":null,'
    '"enforcement":"refiner-off replay (§8)"}]'
).encode()


def test_delta_registry_has_the_frozen_typed_bytes():
    assert POLICY_VERSION == 2
    assert delta_registry_bytes() == EXPECTED_DELTA_REGISTRY_BYTES


def test_policy_two_has_exactly_one_speaker_partition_delta():
    records = {record.id: record for record in DELTA_REGISTRY}
    assert records["PD-SPK"].affected_fields == ("selection",)
    assert records["PD-SPK"].enforcement == "speaker-off counterfactual"
    assert [record.id for record in DELTA_REGISTRY].count("PD-SPK") == 1
