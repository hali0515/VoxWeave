import math

import pytest


def _module():
    from voxweave import align_distribution

    return align_distribution


def _valid_job(
    *,
    block_text="word",
    surfaces=("word",),
    iso="en",
    profile=None,
    verifier_mutator=None,
):
    d = _module()
    blocks = (d.AuthorityBlock(0, block_text),)
    calls = (
        d.AuthorityCallInput(
            call_index=0,
            source_block_indices=(0,),
            raw_node_range=(0, len(surfaces)),
            raw_unit_ids=tuple(f"r{i}" for i in range(len(surfaces))),
            unit_surfaces=surfaces,
            strict_preflight_status="valid",
            strict_failure=None,
        ),
    )
    claims = (d.RouteClaim("call", 0, 0, 0),)
    route = (d.RouteExpectation(0, 0, "call", 0),)
    return d.build_authority_distribution(
        blocks=blocks,
        delivery_route=route,
        calls=calls,
        skipped_blocks=(),
        route_claims=claims,
        iso=iso,
        profile=profile,
        _verifier_cut_mutator=verifier_mutator,
    )


def test_production_limits_and_profile_digest_are_closed():
    d = _module()
    profile = d.production_authority_limit_profile()
    assert profile.kind == "production"
    assert profile.call == d.CallWorkLimits(1_000_000, 4_000_000, 1_000_000, 64_000_000)
    assert profile.job == d.JobWorkLimits(
        4_096, 4_000_000, 16_000_000, 4_000_000, 256_000_000
    )
    assert len(profile.profile_digest) == 64
    assert profile == d.production_authority_limit_profile()


def test_unqualified_changed_production_global_is_rejected(monkeypatch):
    d = _module()
    monkeypatch.setattr(d, "AUTH_ALLOC_STATE_LIMIT", 999_999)
    with pytest.raises(d.AuthorityLimitProfileError) as error:
        d.capture_authority_limit_profile()
    assert error.value.detail_code == "allocator-limit-profile"


def test_test_limit_qualification_is_lowered_single_use_and_task_local():
    d = _module()
    call = d.CallWorkLimits(10, 20, 10, 100)
    job = d.JobWorkLimits(2, 20, 40, 20, 200)
    token = d._issue_test_authority_limit_qualification("case", call, job)
    with d._with_test_authority_limit_qualification(token):
        profile = d.capture_authority_limit_profile()
        assert profile.kind == "test-only"
        assert profile.call == call
        assert profile.job == job
        with pytest.raises(d.AuthorityLimitProfileError):
            d.capture_authority_limit_profile()
    with pytest.raises(d.AuthorityLimitProfileError):
        with d._with_test_authority_limit_qualification(token):
            d.capture_authority_limit_profile()


def test_one_block_one_unit_has_exact_two_lane_counters_and_surface_chars():
    receipt = _valid_job()
    assert receipt.status == "valid"
    assert receipt.owners == (("r0",),)
    assert receipt.expected_counts == (1,)
    assert receipt.consumed_count == 1
    assert receipt.leftovers == ()
    assert receipt.reasons == ()
    assert receipt.work.status == "complete"
    assert receipt.work.charged_call_count == 1
    assert receipt.work.totals == _module().WorkCounters(4, 2, 2, 16)
    row = receipt.work.calls[0]
    assert row.surface_chars == 8
    assert row.allocator.counters == _module().WorkCounters(2, 1, 1, 8)
    assert row.verifier is not None
    assert row.verifier.counters == row.allocator.counters


def test_boundary_local_decimal_and_punctuation_allocation():
    d = _module()
    blocks = (d.AuthorityBlock(0, "3."), d.AuthorityBlock(1, "75"))
    calls = (
        d.AuthorityCallInput(
            0,
            (0, 1),
            (0, 2),
            ("r0", "r1"),
            ("3", "75"),
            "valid",
            None,
        ),
    )
    receipt = d.build_authority_distribution(
        blocks=blocks,
        delivery_route=(
            d.RouteExpectation(0, 0, "call", 0),
            d.RouteExpectation(1, 1, "call", 0),
        ),
        calls=calls,
        skipped_blocks=(),
        route_claims=(
            d.RouteClaim("call", 0, 0, 0),
            d.RouteClaim("call", 0, 1, 1),
        ),
        iso="en",
    )
    assert receipt.status == "valid"
    assert receipt.owners == (("r0",), ("r1",))


def test_true_decimal_can_own_two_units_under_no_space_join():
    receipt = _valid_job(block_text="3.75", surfaces=("3.", "75"), iso="ja")
    assert receipt.status == "valid"
    assert receipt.owners == (("r0", "r1"),)


@pytest.mark.parametrize(
    ("block_text", "surfaces", "detail"),
    [
        ("target", ("other",), "allocation-no-tiling"),
        ("a", ("a", "", "a"), "allocation-ambiguous"),
        ("", ("a",), "partial-empty-ownership"),
        ("...", ("a",), "punctuation-only-block"),
    ],
)
def test_allocator_closed_invalid_terminals(block_text, surfaces, detail):
    receipt = _valid_job(block_text=block_text, surfaces=surfaces)
    assert receipt.status == "invalid"
    assert receipt.owners is None
    assert receipt.consumed_count == 0
    assert receipt.leftovers == tuple(f"r{i}" for i in range(len(surfaces)))
    row = receipt.work.calls[0]
    assert row.allocator.status == "invalid"
    assert row.allocator.terminal_detail_code == detail
    assert receipt.reasons == (detail,)


@pytest.mark.parametrize(
    ("claims", "kind", "observation", "expected", "observed"),
    [
        ((("call", 0, 0, 0), ("call", 0, 2, 2)), "gap", None, 1, None),
        (
            (("call", 0, 0, 0), ("call", 0, 1, 1), ("call", 0, 1, 1)),
            "overlap",
            2,
            2,
            1,
        ),
        (
            (("call", 0, 0, 0), ("call", 0, 1, 1), ("call", 0, 9, 2)),
            "unexpected-index",
            2,
            2,
            9,
        ),
        (
            (("call", 0, 1, 1), ("call", 0, 0, 0), ("call", 0, 2, 2)),
            "reorder",
            0,
            0,
            1,
        ),
        (
            (("call", 0, 0, 9), ("call", 0, 1, 1), ("call", 0, 2, 2)),
            "owner-crosslink",
            0,
            0,
            0,
        ),
    ],
)
def test_route_mismatch_projection_is_lossless_and_precedence_ordered(
    claims, kind, observation, expected, observed
):
    d = _module()
    route = tuple(d.RouteExpectation(i, i, "call", 0) for i in range(3))
    calls = (
        d.AuthorityCallInput(
            0,
            (0, 1, 2),
            (0, 3),
            ("r0", "r1", "r2"),
            ("a", "b", "c"),
            "valid",
            None,
        ),
    )
    typed_claims = tuple(d.RouteClaim(*claim) for claim in claims)
    mismatch = d.project_route_mismatch(typed_claims, route, calls, ())
    assert mismatch == d.RouteMismatch(kind, observation, expected, observed)


def test_invalid_route_round_trips_claims_and_runs_zero_search_work():
    d = _module()
    calls = (
        d.AuthorityCallInput(
            0, (0, 1), (0, 2), ("r0", "r1"), ("a", "b"), "valid", None
        ),
    )
    claims = (d.RouteClaim("call", 0, 0, 0), d.RouteClaim("call", 0, 2, 1))
    receipt = d.build_authority_distribution(
        blocks=(d.AuthorityBlock(0, "a"), d.AuthorityBlock(1, "b")),
        delivery_route=(
            d.RouteExpectation(0, 0, "call", 0),
            d.RouteExpectation(1, 1, "call", 0),
        ),
        calls=calls,
        skipped_blocks=(),
        route_claims=claims,
        iso="en",
    )
    assert receipt.work.status == "not-run-route-invalid"
    assert receipt.work.route_claims == claims
    assert receipt.work.totals == d.WorkCounters(0, 0, 0, 0)
    assert receipt.work.calls[0].allocator.status == "not-run"
    assert receipt.reasons == ("route-owner-mismatch",)


def test_qwen_skip_is_one_zero_work_row_and_invalidates_authority():
    d = _module()
    skip = d.AuthoritySkippedBlockInput(
        delivery_index=0,
        source_index=0,
        route_skip_reason="empty-alignment-text",
        source_text_kind="whitespace",
    )
    receipt = d.build_authority_distribution(
        blocks=(d.AuthorityBlock(0, "   "),),
        delivery_route=(d.RouteExpectation(0, 0, "skip", 0),),
        calls=(),
        skipped_blocks=(skip,),
        route_claims=(d.RouteClaim("skip", 0, 0, 0),),
        iso="en",
    )
    assert receipt.work.status == "not-run-skip-invalid"
    assert receipt.work.skipped_blocks[0].counters == d.WorkCounters(0, 0, 0, 0)
    assert receipt.reasons == ("partial-empty-ownership",)


def test_multi_call_strict_failures_collect_all_locators_and_choose_canonical_first():
    d = _module()
    capture = d.StrictFailureLocator("strict-capture", 0, "strict-raw-node")
    geometry = d.StrictFailureLocator("sample-geometry", None, "sample-geometry")
    calls = (
        d.AuthorityCallInput(
            0, (0,), (0, 1), ("r0",), None, "capture-invalid", capture
        ),
        d.AuthorityCallInput(
            1, (1,), (1, 2), ("r1",), ("b",), "transform-invalid", geometry
        ),
    )
    receipt = d.build_authority_distribution(
        blocks=(d.AuthorityBlock(0, "a"), d.AuthorityBlock(1, "b")),
        delivery_route=(
            d.RouteExpectation(0, 0, "call", 0),
            d.RouteExpectation(1, 1, "call", 1),
        ),
        calls=calls,
        skipped_blocks=(),
        route_claims=(
            d.RouteClaim("call", 0, 0, 0),
            d.RouteClaim("call", 1, 1, 1),
        ),
        iso="en",
    )
    assert receipt.work.status == "not-run-strict-unavailable"
    assert receipt.work.terminal_call_index == 0
    assert receipt.work.calls[0].surface_chars is None
    assert receipt.work.calls[1].surface_chars == 2
    assert receipt.reasons == ("authority-transform-invalid",)
    failure = d.project_authority_failure(receipt)
    assert failure is not None
    assert failure.kind == "fresh-time-transform-invalid"
    assert failure.phase == "strict-capture"
    assert failure.detail_code == "strict-raw-node"
    assert failure.secondary == ()


def test_surface_chars_count_exact_python_codepoints_without_normalization():
    text = "e\u0301 \U0001f600!"
    receipt = _valid_job(block_text=text, surfaces=(text,))
    assert receipt.work.calls[0].surface_chars == len(text) * 2


def test_compound_route_skip_and_strict_reasons_are_registry_sorted():
    d = _module()
    locator = d.StrictFailureLocator("strict-capture", 0, "strict-raw-node")
    calls = (
        d.AuthorityCallInput(
            0, (1,), (0, 1), ("r0",), None, "capture-invalid", locator
        ),
    )
    skip = d.AuthoritySkippedBlockInput(0, 0, "missing-crop", "nonempty")
    receipt = d.build_authority_distribution(
        blocks=(d.AuthorityBlock(0, "a"), d.AuthorityBlock(1, "b")),
        delivery_route=(
            d.RouteExpectation(0, 0, "skip", 0),
            d.RouteExpectation(1, 1, "call", 0),
        ),
        calls=calls,
        skipped_blocks=(skip,),
        route_claims=(
            d.RouteClaim("skip", 0, 1, 0),
            d.RouteClaim("call", 0, 0, 1),
        ),
        iso="en",
    )
    assert receipt.work.status == "not-run-route-invalid"
    assert receipt.reasons == (
        "partial-empty-ownership",
        "authority-transform-invalid",
        "route-owner-mismatch",
    )


def test_atomic_denied_charge_records_call_scope_and_no_partial_counter_mutation():
    d = _module()
    profile = d._test_authority_limit_profile(
        call=d.CallWorkLimits(1, 20, 20, 100),
        job=d.JobWorkLimits(2, 20, 40, 20, 200),
    )
    receipt = _valid_job(profile=profile)
    assert receipt.work.status == "budget-exhausted"
    assert receipt.reasons == ("allocation-budget-exhausted",)
    denied = receipt.work.denied_charge
    assert denied is not None
    assert denied.lane == "allocator"
    assert denied.event_kind == "state-insert"
    assert denied.subject == (1, 1)
    assert denied.counters == (d.DeniedCounter("states", 1, ("call",)),)
    row = receipt.work.calls[0]
    assert row.allocator.counters.states == 1


def test_compound_interval_and_character_denial_is_one_event_in_counter_order():
    d = _module()
    profile = d._test_authority_limit_profile(
        call=d.CallWorkLimits(20, 20, 1, 2),
        job=d.JobWorkLimits(2, 40, 40, 20, 200),
    )
    receipt = _valid_job(block_text="z", surfaces=("x", "y"), profile=profile)
    denied = receipt.work.denied_charge
    assert denied is not None
    assert denied.event_kind == "interval-normalize"
    assert [counter.counter for counter in denied.counters] == [
        "intervals",
        "normalize_chars",
    ]
    assert denied.counters[0].scopes == ("call",)
    assert denied.counters[1].scopes == ("call",)
    assert receipt.work.calls[0].allocator.counters.intervals == 1
    assert receipt.work.calls[0].allocator.counters.normalize_chars == 2


def test_verifier_disagreement_is_transient_seal_mismatch_with_empty_reasons():
    d = _module()
    receipt = _valid_job(verifier_mutator=lambda cuts: tuple(reversed(cuts)))
    assert receipt.status == "invalid"
    assert receipt.work.status == "seal-mismatch"
    assert receipt.reasons == ()
    assert receipt.owners is None
    assert receipt.leftovers == ("r0",)
    failure = d.project_authority_failure(receipt)
    assert failure is not None
    assert failure.to_dict() == {
        "kind": "fresh-seal-broken",
        "phase": "authority-distribution",
        "detail_code": "distribution-seal",
        "secondary": [],
    }


def test_legacy_distribution_slices_before_shift_and_never_reads_surplus():
    d = _module()

    class Hostile:
        def __getitem__(self, key):
            raise AssertionError(f"surplus was read: {key}")

    raw = [
        {"text": "kept", "start": 1.0, "end": 2.0, "extra": "retained"},
        Hostile(),
    ]
    result = d.legacy_distribute_before_shift(
        raw,
        texts=("kept",),
        iso="en",
        origin=3.0,
        identity=False,
        raw_unit_ids=("r0", "r1"),
    )
    assert result.block_units == (({"text": "kept", "start": 4.0, "end": 5.0},),)
    assert result.receipt.leftover_unit_ids == ("r1",)


def test_whole_file_identity_performs_no_addition_and_preserves_unit_shape():
    d = _module()
    unit = {"text": "kept", "start": -0.0, "end": 2.0, "extra": 7}
    result = d.legacy_distribute_before_shift(
        [unit],
        texts=("kept",),
        iso="en",
        origin=0.0,
        identity=True,
        raw_unit_ids=("r0",),
    )
    assert result.block_units == ((unit,),)
    assert math.copysign(1.0, result.block_units[0][0]["start"]) == -1.0
