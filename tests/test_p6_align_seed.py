import dataclasses


def _unit(
    unit_id,
    surface,
    start,
    end,
    *,
    provenance="aligner",
    call_unit_index=0,
):
    from voxweave.align_acquisition import FreshUnit
    from voxweave.align_snapshot import freeze_json

    return FreshUnit(
        unit_id=unit_id,
        call_index=0,
        call_unit_index=call_unit_index,
        surface=surface,
        relative_start=start,
        relative_end=end,
        physical_origin_seconds=0.0,
        start=start,
        end=end,
        provenance=provenance,
        original_relative_start=start if provenance == "aligner" else None,
        original_relative_end=end if provenance == "aligner" else None,
        raw=freeze_json({"text": surface, "start": start, "end": end}),
    )


def _distribution(texts, surfaces):
    from voxweave.align_distribution import (
        AuthorityBlock,
        AuthorityCallInput,
        RouteClaim,
        RouteExpectation,
        build_authority_distribution,
    )

    blocks = tuple(AuthorityBlock(index, text) for index, text in enumerate(texts))
    ids = tuple(f"r{index}" for index in range(len(surfaces)))
    call = AuthorityCallInput(
        0,
        tuple(range(len(texts))),
        (0, len(surfaces)),
        ids,
        tuple(surfaces),
        "valid",
        None,
    )
    route = tuple(
        RouteExpectation(index, index, "call", 0) for index in range(len(texts))
    )
    claims = tuple(RouteClaim("call", 0, index, index) for index in range(len(texts)))
    receipt = build_authority_distribution(
        blocks=blocks,
        delivery_route=route,
        calls=(call,),
        skipped_blocks=(),
        route_claims=claims,
        iso="en",
    )
    assert receipt.status == "valid"
    return blocks, receipt


def test_seed_uses_exact_p5_endpoint_fold_and_all_four_display_branches():
    from voxweave.core.align_seed import build_align_seed

    texts = ("a", "b", "c", "d")
    blocks, distribution = _distribution(texts, texts)
    units = (
        _unit("r0", "a", 1.0, 1.0),
        _unit("r1", "b", 5.0, 6.0, provenance="align-interpolated", call_unit_index=1),
        _unit("r2", "c", 7.0, 8.0, provenance="align-interpolated", call_unit_index=2),
        _unit("r3", "d", 10.0, 11.0, call_unit_index=3),
    )
    seed = build_align_seed(
        blocks=blocks, units=units, distribution=distribution, iso="en"
    )
    assert seed.status == "valid"
    assert seed.reasons == ()
    assert seed.blocks is not None
    assert [(block.speech_start, block.speech_end) for block in seed.blocks] == [
        (1.0, 1.0),
        (None, None),
        (None, None),
        (10.0, 11.0),
    ]
    assert (seed.blocks[0].display_start, seed.blocks[0].display_end) == (1.0, 1.05)
    assert (seed.blocks[1].display_start, seed.blocks[1].display_end) == (3.0, 5.0)
    assert (seed.blocks[2].display_start, seed.blocks[2].display_end) == (7.0, 9.0)
    assert (seed.blocks[3].display_start, seed.blocks[3].display_end) == (10.0, 11.0)


def test_start_only_end_only_and_anchorless_seed_rules_are_exact():
    from voxweave.core.align_seed import build_align_seed

    blocks, distribution = _distribution(("a", "b", "c"), ("a", "b", "c"))
    units = (
        _unit("r0", "a", 5.0, 5.5, call_unit_index=0),
        _unit("r1", "b", 7.0, 8.0, provenance="align-interpolated", call_unit_index=1),
        _unit("r2", "c", 9.0, 10.0, call_unit_index=2),
    )
    # Force start-only, neither, and end-only without changing the absolute bounds.
    units = (
        dataclasses.replace(units[0], end=5.5, provenance="aligner"),
        units[1],
        dataclasses.replace(units[2], provenance="align-interpolated"),
    )
    # Separate one-unit owners cannot express asymmetric endpoint provenance, so use
    # two-unit owners for the explicit branches below.
    blocks2, distribution2 = _distribution(
        ("a x", "b y", "c"), ("a", "x", "b", "y", "c")
    )
    units2 = (
        _unit("r0", "a", 5.0, 5.2, call_unit_index=0),
        _unit("r1", "x", 5.3, 5.5, provenance="align-interpolated", call_unit_index=1),
        _unit("r2", "b", 7.0, 7.2, provenance="align-interpolated", call_unit_index=2),
        _unit("r3", "y", 7.3, 8.0, call_unit_index=3),
        _unit("r4", "c", 9.0, 10.0, provenance="align-interpolated", call_unit_index=4),
    )
    seed = build_align_seed(
        blocks=blocks2, units=units2, distribution=distribution2, iso="en"
    )
    assert seed.blocks is not None
    assert [(b.speech_start, b.speech_end) for b in seed.blocks] == [
        (5.0, None),
        (None, 8.0),
        (None, None),
    ]
    assert (seed.blocks[0].display_start, seed.blocks[0].display_end) == (5.0, 5.05)
    assert (seed.blocks[1].display_start, seed.blocks[1].display_end) == (5.95, 8.0)
    assert (seed.blocks[2].display_start, seed.blocks[2].display_end) == (10.0, 12.0)


def test_seed_rejects_negative_nonmonotone_and_reconciliation_defects():
    from voxweave.align_distribution import AuthorityBlock
    from voxweave.core.align_seed import build_align_seed

    blocks, distribution = _distribution(("a", "b"), ("a", "b"))
    negative = (
        _unit("r0", "a", -1.0, 0.0),
        _unit("r1", "b", 1.0, 2.0, call_unit_index=1),
    )
    result = build_align_seed(
        blocks=blocks, units=negative, distribution=distribution, iso="en"
    )
    assert result.status == "invalid"
    assert result.blocks is None
    assert result.reasons == ("absolute-bound-invalid",)

    nonmonotone = (
        _unit("r0", "a", 2.0, 3.0),
        _unit("r1", "b", 1.0, 4.0, call_unit_index=1),
    )
    result = build_align_seed(
        blocks=blocks, units=nonmonotone, distribution=distribution, iso="en"
    )
    assert result.reasons == ("absolute-order-invalid",)

    changed_blocks = (AuthorityBlock(0, "wrong"), AuthorityBlock(1, "b"))
    valid_units = (
        _unit("r0", "a", 0.0, 1.0),
        _unit("r1", "b", 1.0, 2.0, call_unit_index=1),
    )
    result = build_align_seed(
        blocks=changed_blocks,
        units=valid_units,
        distribution=distribution,
        iso="en",
    )
    assert result.reasons == ("footprint-reconciliation",)


def test_seed_materialization_returns_independent_mutable_w1_copies():
    from voxweave.core.align_seed import build_align_seed, materialize_seed_cues

    blocks, distribution = _distribution(("word",), ("word",))
    unit = _unit("r0", "word", 0.2, 0.8)
    seed = build_align_seed(
        blocks=blocks, units=(unit,), distribution=distribution, iso="en"
    )
    first = materialize_seed_cues(seed)
    second = materialize_seed_cues(seed)
    assert first == second
    assert first is not second
    assert first[0]["word_data"] is not second[0]["word_data"]
    assert first[0]["word_data"][0] is not second[0]["word_data"][0]
    first[0]["word_data"][0]["text"] = "mutated"
    assert second[0]["word_data"][0]["text"] == "word"
    assert seed.blocks is not None and seed.blocks[0].footprint == "word"
