"""Independent reference replay for P6 authority-distribution receipts.

This module imports the producer's immutable receipt types, but no producer
allocator, route projector, profile validator, constants, or reason projector.
Its event machine and shipped constants are deliberately duplicated so a
producer regression cannot validate its own accounting at AO-16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityCallInput,
    AuthorityCallWorkReceipt,
    AuthorityDistributionReceipt,
    AuthorityJobWorkReceipt,
    AuthoritySkippedBlockInput,
    AuthoritySkippedBlockReceipt,
    CallWorkLimits,
    DeniedCharge,
    DeniedCounter,
    JobWorkLimits,
    RouteClaim,
    RouteMismatch,
    WorkCounters,
    WorkLaneReceipt,
)
from voxweave.align_snapshot import FrozenArray, FrozenInt, FrozenString
from voxweave.align_snapshot import frozen_json_digest
from voxweave.core.langsets import LANGUAGES_WITHOUT_SPACES
from voxweave.core.partition_check import normalize_text


class DistributionReferenceError(ValueError):
    pass


_PRODUCTION_CALL = CallWorkLimits(1_000_000, 4_000_000, 1_000_000, 64_000_000)
_PRODUCTION_JOB = JobWorkLimits(4_096, 4_000_000, 16_000_000, 4_000_000, 256_000_000)
_ZERO = WorkCounters(0, 0, 0, 0)
_COUNTERS = ("states", "edges", "intervals", "normalize_chars")
_LIMIT_NAMES = {
    "states": "state_limit",
    "edges": "edge_limit",
    "intervals": "interval_limit",
    "normalize_chars": "normalize_char_limit",
}
_REASON_ORDER = (
    "partial-empty-ownership",
    "punctuation-only-block",
    "authority-transform-invalid",
    "route-owner-mismatch",
    "allocation-no-tiling",
    "allocation-ambiguous",
    "allocation-budget-exhausted",
)
_STRICT_STAGES = ("strict-capture", "sample-geometry", "authority-transform")
_STRICT_DETAILS = (
    "strict-raw-node",
    "sample-geometry",
    "physical-origin-mismatch",
    "authority-recompute",
    "surplus-transform",
)


@dataclass(frozen=True)
class _Expectation:
    delivery_index: int
    source_index: int
    owner_kind: Literal["call", "skip"]
    owner_index: int


@dataclass
class _Counters:
    states: int = 0
    edges: int = 0
    intervals: int = 0
    normalize_chars: int = 0

    def frozen(self) -> WorkCounters:
        return WorkCounters(
            self.states, self.edges, self.intervals, self.normalize_chars
        )


@dataclass
class _Budget:
    call_limits: CallWorkLimits
    job_limits: JobWorkLimits
    calls: int = 0
    totals: _Counters | None = None
    event_ordinal: int = 0

    def __post_init__(self) -> None:
        self.totals = _Counters()

    def start_call(self, call_index: int) -> DeniedCharge | None:
        if self.calls + 1 > self.job_limits.call_limit:
            return DeniedCharge(
                "job",
                self.event_ordinal,
                "call-start",
                (call_index,),
                (DeniedCounter("calls", 1, ("job",)),),
            )
        self.calls += 1
        self.event_ordinal += 1
        return None

    def charge(
        self,
        lane_name: Literal["allocator", "verifier"],
        lane: _Counters,
        event_kind: Literal[
            "block-normalize", "state-insert", "edge-test", "interval-normalize"
        ],
        subject: tuple[int, ...],
        cost: WorkCounters,
    ) -> DeniedCharge | None:
        if self.totals is None:  # pragma: no cover - construction invariant
            raise DistributionReferenceError("reference budget was not initialized")
        denied: list[DeniedCounter] = []
        for counter in _COUNTERS:
            amount = getattr(cost, counter)
            if amount <= 0:
                continue
            limit_name = _LIMIT_NAMES[counter]
            scopes: list[Literal["job", "call"]] = []
            if getattr(self.totals, counter) + amount > getattr(
                self.job_limits, limit_name
            ):
                scopes.append("job")
            if getattr(lane, counter) + amount > getattr(self.call_limits, limit_name):
                scopes.append("call")
            if scopes:
                denied.append(DeniedCounter(counter, amount, tuple(scopes)))
        if denied:
            return DeniedCharge(
                lane_name,
                self.event_ordinal,
                event_kind,
                subject,
                tuple(denied),
            )
        for counter in _COUNTERS:
            amount = getattr(cost, counter)
            setattr(lane, counter, getattr(lane, counter) + amount)
            setattr(self.totals, counter, getattr(self.totals, counter) + amount)
        self.event_ordinal += 1
        return None


@dataclass(frozen=True)
class _LaneOutcome:
    status: Literal["unique", "invalid", "budget-exhausted"]
    counters: WorkCounters
    detail: str | None
    cuts: tuple[int, ...] | None
    denied: DeniedCharge | None


def _profile_digest(
    kind: Literal["production", "test-only"],
    call: CallWorkLimits,
    job: JobWorkLimits,
) -> str:
    return frozen_json_digest(
        FrozenArray(
            (
                FrozenString("authority-limit-profile"),
                FrozenString(kind),
                FrozenInt(call.state_limit),
                FrozenInt(call.edge_limit),
                FrozenInt(call.interval_limit),
                FrozenInt(call.normalize_char_limit),
                FrozenInt(job.call_limit),
                FrozenInt(job.state_limit),
                FrozenInt(job.edge_limit),
                FrozenInt(job.interval_limit),
                FrozenInt(job.normalize_char_limit),
            )
        )
    )


def _validate_profile(work: AuthorityJobWorkReceipt) -> CallWorkLimits:
    if work.limit_profile_kind not in ("production", "test-only"):
        raise DistributionReferenceError("allocator profile kind is invalid")
    call = work.calls[0].limits if work.calls else _PRODUCTION_CALL
    if any(row.limits != call for row in work.calls):
        raise DistributionReferenceError("allocator call limits disagree")
    if work.limit_profile_digest != _profile_digest(
        work.limit_profile_kind, call, work.limits
    ):
        raise DistributionReferenceError("allocator profile digest mismatch")
    if work.limit_profile_kind == "production":
        if call != _PRODUCTION_CALL or work.limits != _PRODUCTION_JOB:
            raise DistributionReferenceError("production allocator limits changed")
        return call
    values = (*call.__dict__.values(), *work.limits.__dict__.values())
    maxima = (*_PRODUCTION_CALL.__dict__.values(), *_PRODUCTION_JOB.__dict__.values())
    if (
        any(type(value) is not int or value < 1 for value in values)
        or any(value > maximum for value, maximum in zip(values, maxima, strict=True))
        or all(value == maximum for value, maximum in zip(values, maxima, strict=True))
    ):
        raise DistributionReferenceError("test allocator profile is not qualified")
    return call


def _route_mismatch(
    claims: tuple[RouteClaim, ...],
    route: tuple[_Expectation, ...],
    calls: tuple[AuthorityCallInput, ...],
    skipped: tuple[AuthoritySkippedBlockInput, ...],
) -> RouteMismatch | None:
    count = len(route)
    observed = tuple(claim.delivery_index for claim in claims)
    present = {index for index in observed if 0 <= index < count}
    for expected in range(count):
        if expected not in present:
            return RouteMismatch("gap", None, expected, None)
    for duplicated in range(count):
        positions = [
            position for position, index in enumerate(observed) if index == duplicated
        ]
        if len(positions) > 1:
            position = positions[1]
            return RouteMismatch(
                "overlap",
                position,
                position if position < count else None,
                observed[position],
            )
    for position, index in enumerate(observed):
        if index < 0 or index >= count:
            return RouteMismatch(
                "unexpected-index",
                position,
                position if position < count else None,
                index,
            )
    expected_order = tuple(range(count))
    if observed != expected_order:
        position = next(
            position
            for position, (left, right) in enumerate(
                zip(observed, expected_order, strict=True)
            )
            if left != right
        )
        return RouteMismatch("reorder", position, position, observed[position])
    call_indexes = {call.call_index for call in calls}
    skip_indexes = set(range(len(skipped)))
    for position, claim in enumerate(claims):
        expected = route[position]
        owner_exists = (
            claim.owner_index in call_indexes
            if claim.owner_kind == "call"
            else claim.owner_index in skip_indexes
        )
        if (
            not owner_exists
            or claim.source_index != expected.source_index
            or claim.owner_kind != expected.owner_kind
            or claim.owner_index != expected.owner_index
            or claim.delivery_index != expected.delivery_index
        ):
            return RouteMismatch(
                "owner-crosslink", position, position, claim.delivery_index
            )
    return None


def _joined_length(prefix: tuple[int, ...], lower: int, upper: int, iso: str) -> int:
    length = prefix[upper] - prefix[lower]
    if iso not in LANGUAGES_WITHOUT_SPACES:
        length += max(0, upper - lower - 1)
    return length


def _joined(surfaces: tuple[str, ...], lower: int, upper: int, iso: str) -> str:
    separator = "" if iso in LANGUAGES_WITHOUT_SPACES else " "
    return separator.join(surfaces[lower:upper])


def _run_lane(
    lane_name: Literal["allocator", "verifier"],
    blocks: tuple[AuthorityBlock, ...],
    surfaces: tuple[str, ...],
    iso: str,
    budget: _Budget,
) -> _LaneOutcome:
    lane = _Counters()
    normalized_blocks: dict[int, str] = {}
    normalized_intervals: dict[tuple[int, int], str] = {}
    prefix = [0]
    for surface in surfaces:
        prefix.append(prefix[-1] + len(surface))
    prefix_tuple = tuple(prefix)
    paths = [dict() for _ in range(len(blocks) + 1)]
    parents: dict[tuple[int, int], int] = {}
    for block_index, block in enumerate(blocks):
        denied = budget.charge(
            lane_name,
            lane,
            "block-normalize",
            (block_index,),
            WorkCounters(0, 0, 0, len(block.alignment_text)),
        )
        if denied is not None:
            return _LaneOutcome(
                "budget-exhausted",
                lane.frozen(),
                "allocation-budget",
                None,
                denied,
            )
        normalized_blocks[block_index] = normalize_text(block.alignment_text)
        if not block.alignment_text.strip():
            return _LaneOutcome(
                "invalid",
                lane.frozen(),
                "partial-empty-ownership",
                None,
                None,
            )
        if not normalized_blocks[block_index]:
            return _LaneOutcome(
                "invalid",
                lane.frozen(),
                "punctuation-only-block",
                None,
                None,
            )
    denied = budget.charge(
        lane_name,
        lane,
        "state-insert",
        (0, 0),
        WorkCounters(1, 0, 0, 0),
    )
    if denied is not None:
        return _LaneOutcome(
            "budget-exhausted",
            lane.frozen(),
            "allocation-budget",
            None,
            denied,
        )
    paths[0][0] = 1
    for block_index in range(len(blocks)):
        for lower in sorted(paths[block_index]):
            for upper in range(lower + 1, len(surfaces) + 1):
                denied = budget.charge(
                    lane_name,
                    lane,
                    "edge-test",
                    (block_index, lower, upper),
                    WorkCounters(0, 1, 0, 0),
                )
                if denied is not None:
                    return _LaneOutcome(
                        "budget-exhausted",
                        lane.frozen(),
                        "allocation-budget",
                        None,
                        denied,
                    )
                interval = (lower, upper)
                if interval not in normalized_intervals:
                    denied = budget.charge(
                        lane_name,
                        lane,
                        "interval-normalize",
                        interval,
                        WorkCounters(
                            0,
                            0,
                            1,
                            _joined_length(prefix_tuple, lower, upper, iso),
                        ),
                    )
                    if denied is not None:
                        return _LaneOutcome(
                            "budget-exhausted",
                            lane.frozen(),
                            "allocation-budget",
                            None,
                            denied,
                        )
                    normalized_intervals[interval] = normalize_text(
                        _joined(surfaces, lower, upper, iso)
                    )
                if normalized_blocks[block_index] != normalized_intervals[interval]:
                    continue
                if upper not in paths[block_index + 1]:
                    denied = budget.charge(
                        lane_name,
                        lane,
                        "state-insert",
                        (block_index + 1, upper),
                        WorkCounters(1, 0, 0, 0),
                    )
                    if denied is not None:
                        return _LaneOutcome(
                            "budget-exhausted",
                            lane.frozen(),
                            "allocation-budget",
                            None,
                            denied,
                        )
                    paths[block_index + 1][upper] = 0
                paths[block_index + 1][upper] = min(
                    2,
                    paths[block_index + 1][upper] + paths[block_index][lower],
                )
                if paths[block_index + 1][upper] == 1:
                    parents[block_index + 1, upper] = lower
                else:
                    parents.pop((block_index + 1, upper), None)
    path_count = paths[len(blocks)].get(len(surfaces), 0)
    if path_count == 0:
        return _LaneOutcome(
            "invalid",
            lane.frozen(),
            "allocation-no-tiling",
            None,
            None,
        )
    if path_count == 2:
        return _LaneOutcome(
            "invalid",
            lane.frozen(),
            "allocation-ambiguous",
            None,
            None,
        )
    cuts = [len(surfaces)]
    boundary = len(surfaces)
    for row_index in range(len(blocks), 0, -1):
        boundary = parents[row_index, boundary]
        cuts.append(boundary)
    cuts.reverse()
    return _LaneOutcome("unique", lane.frozen(), None, tuple(cuts), None)


def _surface_chars(
    call: AuthorityCallInput, blocks_by_source: dict[int, AuthorityBlock]
) -> int | None:
    if call.unit_surfaces is None:
        return None
    return sum(
        len(blocks_by_source[source].alignment_text)
        for source in call.source_block_indices
    ) + sum(len(surface) for surface in call.unit_surfaces)


def _base_row(
    call: AuthorityCallInput,
    claims: tuple[RouteClaim, ...],
    blocks_by_source: dict[int, AuthorityBlock],
    limits: CallWorkLimits,
    *,
    prior_terminal: bool = False,
) -> AuthorityCallWorkReceipt:
    return AuthorityCallWorkReceipt(
        call.call_index,
        tuple(
            position
            for position, claim in enumerate(claims)
            if claim.owner_kind == "call" and claim.owner_index == call.call_index
        ),
        call.source_block_indices,
        call.raw_node_range,
        len(call.source_block_indices),
        len(call.raw_unit_ids),
        None if call.unit_surfaces is None else len(call.unit_surfaces),
        _surface_chars(call, blocks_by_source),
        call.strict_preflight_status,
        call.strict_failure,
        limits,
        WorkLaneReceipt(
            "not-run-prior-terminal" if prior_terminal else "not-run",
            _ZERO,
            None,
        ),
        None,
    )


def _skip_rows(
    skipped: tuple[AuthoritySkippedBlockInput, ...],
    claims: tuple[RouteClaim, ...],
) -> tuple[AuthoritySkippedBlockReceipt, ...]:
    return tuple(
        AuthoritySkippedBlockReceipt(
            tuple(
                position
                for position, claim in enumerate(claims)
                if claim.owner_kind == "skip" and claim.owner_index == skip_index
            ),
            item.delivery_index,
            item.source_index,
            item.route_skip_reason,
            item.source_text_kind,
            "partial-empty-ownership",
            "not-run",
            _ZERO,
        )
        for skip_index, item in enumerate(skipped)
    )


def _first_strict_call(calls: tuple[AuthorityCallInput, ...]) -> int | None:
    facts = [
        (
            call.call_index,
            -1
            if call.strict_failure.call_unit_index is None
            else call.strict_failure.call_unit_index,
            _STRICT_STAGES.index(call.strict_failure.stage),
            _STRICT_DETAILS.index(call.strict_failure.detail_code),
        )
        for call in calls
        if call.strict_failure is not None
    ]
    return min(facts)[0] if facts else None


def _reasons(work: AuthorityJobWorkReceipt) -> tuple[str, ...]:
    if work.status == "seal-mismatch":
        return ()
    present: set[str] = set()
    if work.route_status == "invalid":
        present.add("route-owner-mismatch")
    if work.skipped_blocks:
        present.add("partial-empty-ownership")
    if any(row.strict_failure is not None for row in work.calls):
        present.add("authority-transform-invalid")
    for row in work.calls:
        if row.allocator.terminal_detail_code in {
            "partial-empty-ownership",
            "punctuation-only-block",
            "allocation-no-tiling",
            "allocation-ambiguous",
        }:
            present.add(row.allocator.terminal_detail_code)
    if work.status == "budget-exhausted":
        present.add("allocation-budget-exhausted")
    return tuple(reason for reason in _REASON_ORDER if reason in present)


def _job(
    *,
    status: str,
    mismatch: RouteMismatch | None,
    claims: tuple[RouteClaim, ...],
    route_count: int,
    calls: tuple[AuthorityCallInput, ...],
    skipped: tuple[AuthoritySkippedBlockInput, ...],
    profile_kind: Literal["production", "test-only"],
    profile_digest: str,
    job_limits: JobWorkLimits,
    charged_calls: int,
    totals: WorkCounters,
    terminal_call_index: int | None,
    denied: DeniedCharge | None,
    rows: tuple[AuthorityCallWorkReceipt, ...],
    skip_rows: tuple[AuthoritySkippedBlockReceipt, ...],
) -> AuthorityJobWorkReceipt:
    return AuthorityJobWorkReceipt(
        status,  # type: ignore[arg-type]
        "invalid" if mismatch is not None else "valid",
        mismatch,
        claims,
        route_count,
        len(calls),
        len(skipped),
        sum(len(call.raw_unit_ids) for call in calls),
        charged_calls,
        profile_kind,
        profile_digest,
        job_limits,
        totals,
        terminal_call_index,
        denied,
        rows,
        skip_rows,
    )


def replay_authority_distribution(
    *,
    blocks: tuple[AuthorityBlock, ...],
    calls: tuple[AuthorityCallInput, ...],
    skipped: tuple[AuthoritySkippedBlockInput, ...],
    receipt: AuthorityDistributionReceipt,
    iso: str,
) -> None:
    """Recompute and require the complete producer receipt exactly."""
    work = receipt.work
    call_limits = _validate_profile(work)
    if type(iso) is not str or not iso:
        raise DistributionReferenceError("allocator language is invalid")
    blocks_by_source = {block.source_index: block for block in blocks}
    if len(blocks_by_source) != len(blocks):
        raise DistributionReferenceError("allocator blocks are not unique")
    if tuple(call.call_index for call in calls) != tuple(range(len(calls))):
        raise DistributionReferenceError("allocator call indexes are not contiguous")
    raw_cursor = 0
    for call in calls:
        if call.raw_node_range != (
            raw_cursor,
            raw_cursor + len(call.raw_unit_ids),
        ):
            raise DistributionReferenceError("allocator raw ranges are not contiguous")
        raw_cursor = call.raw_node_range[1]
        if call.strict_preflight_status == "capture-invalid":
            if call.unit_surfaces is not None or call.strict_failure is None:
                raise DistributionReferenceError("capture-invalid facts disagree")
        elif call.unit_surfaces is None:
            raise DistributionReferenceError("capture-valid call lacks surfaces")

    call_owner = {
        source: call.call_index
        for call in calls
        for source in call.source_block_indices
    }
    if len(call_owner) != sum(len(call.source_block_indices) for call in calls):
        raise DistributionReferenceError("allocator call ownership overlaps")
    skipped_by_source = {item.source_index: index for index, item in enumerate(skipped)}
    if len(skipped_by_source) != len(skipped):
        raise DistributionReferenceError("allocator skip ownership overlaps")
    route: list[_Expectation] = []
    for delivery_index, block in enumerate(blocks):
        call_index = call_owner.get(block.source_index)
        skip_index = skipped_by_source.get(block.source_index)
        if (call_index is None) == (skip_index is None):
            raise DistributionReferenceError("allocator route is not lossless")
        if call_index is not None:
            route.append(
                _Expectation(delivery_index, block.source_index, "call", call_index)
            )
        else:
            if skip_index is None:  # pragma: no cover - exclusive check above
                raise DistributionReferenceError("allocator skip owner is absent")
            route.append(
                _Expectation(delivery_index, block.source_index, "skip", skip_index)
            )
    claims = work.route_claims
    mismatch = _route_mismatch(claims, tuple(route), calls, skipped)
    base_rows = tuple(
        _base_row(call, claims, blocks_by_source, call_limits) for call in calls
    )
    skip_rows = _skip_rows(skipped, claims)
    raw_ids = tuple(unit_id for call in calls for unit_id in call.raw_unit_ids)
    preflight: str | None = None
    terminal: int | None = None
    if mismatch is not None:
        preflight = "not-run-route-invalid"
    elif skipped:
        preflight = "not-run-skip-invalid"
    elif any(call.strict_failure is not None for call in calls):
        preflight = "not-run-strict-unavailable"
        terminal = _first_strict_call(calls)
    if preflight is not None:
        expected_work = _job(
            status=preflight,
            mismatch=mismatch,
            claims=claims,
            route_count=len(route),
            calls=calls,
            skipped=skipped,
            profile_kind=work.limit_profile_kind,
            profile_digest=work.limit_profile_digest,
            job_limits=work.limits,
            charged_calls=0,
            totals=_ZERO,
            terminal_call_index=terminal,
            denied=None,
            rows=base_rows,
            skip_rows=skip_rows,
        )
        expected = AuthorityDistributionReceipt(
            "invalid",
            tuple(item.source_index for item in route),
            None,
            None,
            _reasons(expected_work),
            0,
            raw_ids,
            expected_work,
        )
        if receipt != expected:
            raise DistributionReferenceError("allocator preflight receipt mismatch")
        return

    budget = _Budget(call_limits, work.limits)
    rows: list[AuthorityCallWorkReceipt] = []
    owners: dict[int, tuple[str, ...]] = {}
    job_status = "complete"
    denied_charge: DeniedCharge | None = None
    terminal_call: int | None = None
    stopped = False
    for call in calls:
        if stopped:
            rows.append(
                _base_row(
                    call,
                    claims,
                    blocks_by_source,
                    call_limits,
                    prior_terminal=True,
                )
            )
            continue
        denied = budget.start_call(call.call_index)
        row = _base_row(call, claims, blocks_by_source, call_limits)
        if denied is not None:
            rows.append(
                AuthorityCallWorkReceipt(
                    row.call_index,
                    row.route_claim_positions,
                    row.source_block_indices,
                    row.raw_node_range,
                    row.block_count,
                    row.raw_node_count,
                    row.typed_unit_count,
                    row.surface_chars,
                    row.strict_preflight_status,
                    row.strict_failure,
                    row.limits,
                    WorkLaneReceipt("budget-exhausted", _ZERO, "allocation-budget"),
                    None,
                )
            )
            job_status = "budget-exhausted"
            denied_charge = denied
            terminal_call = call.call_index
            stopped = True
            continue
        if call.unit_surfaces is None:  # pragma: no cover - preflight above
            raise DistributionReferenceError("allocator surfaces disappeared")
        call_blocks = tuple(
            blocks_by_source[source] for source in call.source_block_indices
        )
        allocator = _run_lane("allocator", call_blocks, call.unit_surfaces, iso, budget)
        allocator_receipt = WorkLaneReceipt(
            "complete" if allocator.status == "unique" else allocator.status,
            allocator.counters,
            allocator.detail,  # type: ignore[arg-type]
        )
        if allocator.status != "unique":
            rows.append(
                AuthorityCallWorkReceipt(
                    row.call_index,
                    row.route_claim_positions,
                    row.source_block_indices,
                    row.raw_node_range,
                    row.block_count,
                    row.raw_node_count,
                    row.typed_unit_count,
                    row.surface_chars,
                    row.strict_preflight_status,
                    row.strict_failure,
                    row.limits,
                    allocator_receipt,
                    None,
                )
            )
            job_status = (
                "budget-exhausted"
                if allocator.status == "budget-exhausted"
                else "invalid"
            )
            denied_charge = allocator.denied
            terminal_call = call.call_index
            stopped = True
            continue
        verifier = _run_lane("verifier", call_blocks, call.unit_surfaces, iso, budget)
        verifier_receipt = WorkLaneReceipt(
            "complete" if verifier.status == "unique" else verifier.status,
            verifier.counters,
            verifier.detail,  # type: ignore[arg-type]
        )
        rows.append(
            AuthorityCallWorkReceipt(
                row.call_index,
                row.route_claim_positions,
                row.source_block_indices,
                row.raw_node_range,
                row.block_count,
                row.raw_node_count,
                row.typed_unit_count,
                row.surface_chars,
                row.strict_preflight_status,
                row.strict_failure,
                row.limits,
                allocator_receipt,
                verifier_receipt,
            )
        )
        if verifier.status == "budget-exhausted":
            job_status = "budget-exhausted"
            denied_charge = verifier.denied
            terminal_call = call.call_index
            stopped = True
            continue
        if verifier.status != "unique" or verifier.cuts != allocator.cuts:
            rows[-1] = AuthorityCallWorkReceipt(
                row.call_index,
                row.route_claim_positions,
                row.source_block_indices,
                row.raw_node_range,
                row.block_count,
                row.raw_node_count,
                row.typed_unit_count,
                row.surface_chars,
                row.strict_preflight_status,
                row.strict_failure,
                row.limits,
                allocator_receipt,
                WorkLaneReceipt(
                    "seal-mismatch",
                    verifier.counters,
                    "distribution-seal",
                ),
            )
            job_status = "seal-mismatch"
            terminal_call = call.call_index
            stopped = True
            continue
        if allocator.cuts is None:  # pragma: no cover - unique invariant
            raise DistributionReferenceError("allocator unique result lacks cuts")
        groups = tuple(
            call.raw_unit_ids[allocator.cuts[index] : allocator.cuts[index + 1]]
            for index in range(len(call.source_block_indices))
        )
        for source, group in zip(call.source_block_indices, groups, strict=True):
            owners[source] = group

    if budget.totals is None:  # pragma: no cover - construction invariant
        raise DistributionReferenceError("reference totals disappeared")
    expected_work = _job(
        status=job_status,
        mismatch=None,
        claims=claims,
        route_count=len(route),
        calls=calls,
        skipped=skipped,
        profile_kind=work.limit_profile_kind,
        profile_digest=work.limit_profile_digest,
        job_limits=work.limits,
        charged_calls=budget.calls,
        totals=budget.totals.frozen(),
        terminal_call_index=terminal_call,
        denied=denied_charge,
        rows=tuple(rows),
        skip_rows=skip_rows,
    )
    sources = tuple(item.source_index for item in route)
    if job_status == "complete":
        owner_rows = tuple(owners[source] for source in sources)
        expected = AuthorityDistributionReceipt(
            "valid",
            sources,
            owner_rows,
            tuple(len(owner) for owner in owner_rows),
            (),
            len(raw_ids),
            (),
            expected_work,
        )
    else:
        expected = AuthorityDistributionReceipt(
            "invalid",
            sources,
            None,
            None,
            _reasons(expected_work),
            0,
            raw_ids,
            expected_work,
        )
    if receipt != expected:
        raise DistributionReferenceError("allocator receipt replay mismatch")


__all__ = ["DistributionReferenceError", "replay_authority_distribution"]
