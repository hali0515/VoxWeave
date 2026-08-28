"""Legacy slice parity and bounded all-unit authority distribution.

The selected legacy lane is deliberately tiny: compute count-only prefix
slices, then shift only retained slices.  Strict authority uses a separate,
bounded, boundary-local search over every captured unit.  Its producer and
verifier have independent traversal state but share the one atomic charge
ledger required to enforce command-wide limits.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal

from voxweave.align_failures import (
    AUTHORITY_REASON_ORDER,
    CanonicalFailure,
)
from voxweave.align_snapshot import (
    FrozenArray,
    FrozenInt,
    FrozenString,
    frozen_json_digest,
)
from voxweave.core.langsets import LANGUAGES_WITHOUT_SPACES
from voxweave.core.partition_check import normalize_text
from voxweave.timestamps import shift_units

# Shipped production limits.  The duplicate private literals are the immutable
# import-time qualification reference; mutating a public global without a
# one-use test qualification is an authority error, not a supported override.
AUTH_ALLOC_STATE_LIMIT = 1_000_000
AUTH_ALLOC_EDGE_LIMIT = 4_000_000
AUTH_ALLOC_INTERVAL_LIMIT = 1_000_000
AUTH_ALLOC_NORMALIZE_CHAR_LIMIT = 64_000_000
AUTH_ALLOC_JOB_CALL_LIMIT = 4_096
AUTH_ALLOC_JOB_STATE_LIMIT = 4_000_000
AUTH_ALLOC_JOB_EDGE_LIMIT = 16_000_000
AUTH_ALLOC_JOB_INTERVAL_LIMIT = 4_000_000
AUTH_ALLOC_JOB_NORMALIZE_CHAR_LIMIT = 256_000_000

_PRODUCTION_CALL_VALUES = (1_000_000, 4_000_000, 1_000_000, 64_000_000)
_PRODUCTION_JOB_VALUES = (4_096, 4_000_000, 16_000_000, 4_000_000, 256_000_000)

COUNTER_ORDER = ("calls", "states", "edges", "intervals", "normalize_chars")
SCOPE_ORDER = ("job", "call")


@dataclass(frozen=True)
class WorkCounters:
    states: int
    edges: int
    intervals: int
    normalize_chars: int


ZERO_COUNTERS = WorkCounters(0, 0, 0, 0)


@dataclass(frozen=True)
class CallWorkLimits:
    state_limit: int
    edge_limit: int
    interval_limit: int
    normalize_char_limit: int


@dataclass(frozen=True)
class JobWorkLimits:
    call_limit: int
    state_limit: int
    edge_limit: int
    interval_limit: int
    normalize_char_limit: int


@dataclass(frozen=True)
class AuthorityLimitProfile:
    kind: Literal["production", "test-only"]
    call: CallWorkLimits
    job: JobWorkLimits
    profile_digest: str


class AuthorityLimitProfileError(RuntimeError):
    def __init__(self, detail_code: str, message: str):
        super().__init__(message)
        self.detail_code = detail_code


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


def _production_values() -> tuple[CallWorkLimits, JobWorkLimits]:
    call = CallWorkLimits(
        AUTH_ALLOC_STATE_LIMIT,
        AUTH_ALLOC_EDGE_LIMIT,
        AUTH_ALLOC_INTERVAL_LIMIT,
        AUTH_ALLOC_NORMALIZE_CHAR_LIMIT,
    )
    job = JobWorkLimits(
        AUTH_ALLOC_JOB_CALL_LIMIT,
        AUTH_ALLOC_JOB_STATE_LIMIT,
        AUTH_ALLOC_JOB_EDGE_LIMIT,
        AUTH_ALLOC_JOB_INTERVAL_LIMIT,
        AUTH_ALLOC_JOB_NORMALIZE_CHAR_LIMIT,
    )
    return call, job


def production_authority_limit_profile() -> AuthorityLimitProfile:
    call, job = _production_values()
    if (
        tuple(call.__dict__.values()) != _PRODUCTION_CALL_VALUES
        or tuple(job.__dict__.values()) != _PRODUCTION_JOB_VALUES
    ):
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "production authority limits were modified"
        )
    return AuthorityLimitProfile(
        "production", call, job, _profile_digest("production", call, job)
    )


@dataclass(frozen=True, init=False)
class _AuthorityLimitQualification:
    case_id: str
    call: CallWorkLimits
    job: JobWorkLimits
    nonce: str
    scope: tuple[int, int | None]


@dataclass
class _QualificationRecord:
    token: _AuthorityLimitQualification
    consumed: bool = False


_QUALIFICATIONS: dict[str, _QualificationRecord] = {}
_ISSUED_TEST_PROFILES: dict[int, AuthorityLimitProfile] = {}
_ACTIVE_QUALIFICATION: contextvars.ContextVar[_AuthorityLimitQualification | None] = (
    contextvars.ContextVar("p6_authority_limit_qualification", default=None)
)


def _scope() -> tuple[int, int | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), id(task) if task is not None else None


def _exact_positive(values: Sequence[int]) -> bool:
    return all(type(value) is int and value >= 1 for value in values)


def _validate_test_limits(call: CallWorkLimits, job: JobWorkLimits) -> None:
    call_values = tuple(call.__dict__.values())
    job_values = tuple(job.__dict__.values())
    if not _exact_positive((*call_values, *job_values)):
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "test authority limits must be positive integers"
        )
    all_values = (*call_values, *job_values)
    production = (*_PRODUCTION_CALL_VALUES, *_PRODUCTION_JOB_VALUES)
    if any(value > maximum for value, maximum in zip(all_values, production)):
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "test authority limits may only decrease"
        )
    if all(value == maximum for value, maximum in zip(all_values, production)):
        raise AuthorityLimitProfileError(
            "allocator-limit-profile",
            "test qualification must lower at least one limit",
        )


def _issue_test_authority_limit_qualification(
    case_id: str, call: CallWorkLimits, job: JobWorkLimits
) -> _AuthorityLimitQualification:
    """Issue the private one-use calibration qualification."""
    if type(case_id) is not str or not case_id:
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "test qualification case id is required"
        )
    _validate_test_limits(call, job)
    import secrets

    token = object.__new__(_AuthorityLimitQualification)
    object.__setattr__(token, "case_id", case_id)
    object.__setattr__(token, "call", call)
    object.__setattr__(token, "job", job)
    object.__setattr__(token, "nonce", secrets.token_hex(32))
    object.__setattr__(token, "scope", _scope())
    _QUALIFICATIONS[token.nonce] = _QualificationRecord(token)
    return token


@contextmanager
def _with_test_authority_limit_qualification(
    token: _AuthorityLimitQualification,
) -> Iterator[None]:
    record = _QUALIFICATIONS.get(getattr(token, "nonce", ""))
    if (
        record is None
        or record.token is not token
        or record.consumed
        or token.scope != _scope()
    ):
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "test authority qualification is invalid"
        )
    context_token = _ACTIVE_QUALIFICATION.set(token)
    try:
        yield
    finally:
        _ACTIVE_QUALIFICATION.reset(context_token)


def capture_authority_limit_profile() -> AuthorityLimitProfile:
    qualification = _ACTIVE_QUALIFICATION.get()
    if qualification is None:
        return production_authority_limit_profile()
    record = _QUALIFICATIONS.get(qualification.nonce)
    if (
        record is None
        or record.token is not qualification
        or record.consumed
        or qualification.scope != _scope()
    ):
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "test authority qualification cannot be consumed"
        )
    record.consumed = True
    profile = AuthorityLimitProfile(
        "test-only",
        qualification.call,
        qualification.job,
        _profile_digest("test-only", qualification.call, qualification.job),
    )
    _ISSUED_TEST_PROFILES[id(profile)] = profile
    return profile


def validate_authority_limit_profile(
    profile: AuthorityLimitProfile,
) -> AuthorityLimitProfile:
    """Reject a forged, increased, or digest-substituted effective profile."""
    if not isinstance(profile, AuthorityLimitProfile):
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "authority limit profile is not issued"
        )
    if profile.kind == "production":
        call, job = _production_values()
        if profile.call != call or profile.job != job:
            raise AuthorityLimitProfileError(
                "allocator-limit-profile", "production authority limits changed"
            )
    elif profile.kind == "test-only":
        _validate_test_limits(profile.call, profile.job)
        if _ISSUED_TEST_PROFILES.get(id(profile)) is not profile:
            raise AuthorityLimitProfileError(
                "allocator-limit-profile",
                "test authority profile lacks its one-use qualification",
            )
    else:  # pragma: no cover - the dataclass annotation is not runtime authority
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "unknown authority limit profile kind"
        )
    expected = _profile_digest(profile.kind, profile.call, profile.job)
    if profile.profile_digest != expected:
        raise AuthorityLimitProfileError(
            "allocator-limit-profile", "authority limit profile digest changed"
        )
    return profile


@dataclass(frozen=True)
class RouteClaim:
    owner_kind: Literal["call", "skip"]
    owner_index: int
    delivery_index: int
    source_index: int


@dataclass(frozen=True)
class RouteExpectation:
    delivery_index: int
    source_index: int
    owner_kind: Literal["call", "skip"]
    owner_index: int


@dataclass(frozen=True)
class RouteMismatch:
    kind: Literal["gap", "overlap", "unexpected-index", "reorder", "owner-crosslink"]
    observation_index: int | None
    expected_delivery_index: int | None
    observed_delivery_index: int | None


@dataclass(frozen=True)
class StrictFailureLocator:
    stage: Literal["strict-capture", "sample-geometry", "authority-transform"]
    call_unit_index: int | None
    detail_code: Literal[
        "strict-raw-node",
        "sample-geometry",
        "physical-origin-mismatch",
        "authority-recompute",
        "surplus-transform",
    ]


@dataclass(frozen=True)
class AuthorityBlock:
    source_index: int
    alignment_text: str
    text: str | None = None


@dataclass(frozen=True)
class AuthorityCallInput:
    call_index: int
    source_block_indices: tuple[int, ...]
    raw_node_range: tuple[int, int]
    raw_unit_ids: tuple[str, ...]
    unit_surfaces: tuple[str, ...] | None
    strict_preflight_status: Literal["valid", "capture-invalid", "transform-invalid"]
    strict_failure: StrictFailureLocator | None


@dataclass(frozen=True)
class AuthoritySkippedBlockInput:
    delivery_index: int
    source_index: int
    route_skip_reason: Literal["missing-crop", "empty-alignment-text"]
    source_text_kind: Literal["empty", "whitespace", "nonempty"]


def project_route_mismatch(
    claims: tuple[RouteClaim, ...],
    route: tuple[RouteExpectation, ...],
    calls: tuple[AuthorityCallInput, ...],
    skipped: tuple[AuthoritySkippedBlockInput, ...],
) -> RouteMismatch | None:
    count = len(route)
    observed = tuple(claim.delivery_index for claim in claims)
    present = set(index for index in observed if 0 <= index < count)
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
            for position, (left, right) in enumerate(zip(observed, expected_order))
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


@dataclass(frozen=True)
class DeniedCounter:
    counter: Literal["calls", "states", "edges", "intervals", "normalize_chars"]
    amount: int
    scopes: tuple[Literal["job", "call"], ...]


@dataclass(frozen=True)
class DeniedCharge:
    lane: Literal["job", "allocator", "verifier"]
    event_ordinal: int
    event_kind: Literal[
        "call-start",
        "block-normalize",
        "state-insert",
        "edge-test",
        "interval-normalize",
    ]
    subject: tuple[int, ...]
    counters: tuple[DeniedCounter, ...]


LaneTerminalDetail = Literal[
    "partial-empty-ownership",
    "punctuation-only-block",
    "allocation-no-tiling",
    "allocation-ambiguous",
    "allocation-budget",
    "distribution-seal",
]


@dataclass(frozen=True)
class WorkLaneReceipt:
    status: Literal[
        "not-run",
        "not-run-prior-terminal",
        "complete",
        "invalid",
        "budget-exhausted",
        "seal-mismatch",
    ]
    counters: WorkCounters
    terminal_detail_code: LaneTerminalDetail | None


@dataclass(frozen=True)
class AuthorityCallWorkReceipt:
    call_index: int
    route_claim_positions: tuple[int, ...]
    source_block_indices: tuple[int, ...]
    raw_node_range: tuple[int, int]
    block_count: int
    raw_node_count: int
    typed_unit_count: int | None
    surface_chars: int | None
    strict_preflight_status: Literal["valid", "capture-invalid", "transform-invalid"]
    strict_failure: StrictFailureLocator | None
    limits: CallWorkLimits
    allocator: WorkLaneReceipt
    verifier: WorkLaneReceipt | None


@dataclass(frozen=True)
class AuthoritySkippedBlockReceipt:
    route_claim_positions: tuple[int, ...]
    delivery_index: int
    source_index: int
    route_skip_reason: Literal["missing-crop", "empty-alignment-text"]
    source_text_kind: Literal["empty", "whitespace", "nonempty"]
    detail_code: Literal["partial-empty-ownership"]
    work_status: Literal["not-run"]
    counters: WorkCounters


@dataclass(frozen=True)
class AuthorityJobWorkReceipt:
    status: Literal[
        "complete",
        "invalid",
        "budget-exhausted",
        "seal-mismatch",
        "not-run-route-invalid",
        "not-run-skip-invalid",
        "not-run-strict-unavailable",
    ]
    route_status: Literal["valid", "invalid"]
    route_mismatch: RouteMismatch | None
    route_claims: tuple[RouteClaim, ...]
    declared_delivery_block_count: int
    declared_call_count: int
    declared_skip_count: int
    declared_raw_node_count: int
    charged_call_count: int
    limit_profile_kind: Literal["production", "test-only"]
    limit_profile_digest: str
    limits: JobWorkLimits
    totals: WorkCounters
    terminal_call_index: int | None
    denied_charge: DeniedCharge | None
    calls: tuple[AuthorityCallWorkReceipt, ...]
    skipped_blocks: tuple[AuthoritySkippedBlockReceipt, ...]


@dataclass(frozen=True)
class AuthorityDistributionReceipt:
    status: Literal["valid", "invalid"]
    owner_source_indices: tuple[int, ...]
    owners: tuple[tuple[str, ...], ...] | None
    expected_counts: tuple[int, ...] | None
    reasons: tuple[str, ...]
    consumed_count: int
    leftovers: tuple[str, ...]
    work: AuthorityJobWorkReceipt


@dataclass
class _MutableCounters:
    states: int = 0
    edges: int = 0
    intervals: int = 0
    normalize_chars: int = 0

    def frozen(self) -> WorkCounters:
        return WorkCounters(
            self.states, self.edges, self.intervals, self.normalize_chars
        )


_LIMIT_NAME = {
    "states": "state_limit",
    "edges": "edge_limit",
    "intervals": "interval_limit",
    "normalize_chars": "normalize_char_limit",
}


@dataclass
class _BudgetLedger:
    profile: AuthorityLimitProfile
    calls: int = 0
    totals: _MutableCounters | None = None
    event_ordinal: int = 0

    def __post_init__(self) -> None:
        self.totals = _MutableCounters()

    def reserve_call(self, call_index: int) -> DeniedCharge | None:
        if self.calls + 1 > self.profile.job.call_limit:
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

    def reserve(
        self,
        lane: Literal["allocator", "verifier"],
        lane_counters: _MutableCounters,
        event_kind: Literal[
            "block-normalize", "state-insert", "edge-test", "interval-normalize"
        ],
        subject: tuple[int, ...],
        cost: WorkCounters,
    ) -> DeniedCharge | None:
        assert self.totals is not None
        denied: list[DeniedCounter] = []
        for counter in COUNTER_ORDER[1:]:
            amount = getattr(cost, counter)
            if amount <= 0:
                continue
            limit_name = _LIMIT_NAME[counter]
            scopes: list[Literal["job", "call"]] = []
            if getattr(self.totals, counter) + amount > getattr(
                self.profile.job, limit_name
            ):
                scopes.append("job")
            if getattr(lane_counters, counter) + amount > getattr(
                self.profile.call, limit_name
            ):
                scopes.append("call")
            if scopes:
                denied.append(DeniedCounter(counter, amount, tuple(scopes)))
        if denied:
            return DeniedCharge(
                lane, self.event_ordinal, event_kind, subject, tuple(denied)
            )
        for counter in COUNTER_ORDER[1:]:
            amount = getattr(cost, counter)
            setattr(lane_counters, counter, getattr(lane_counters, counter) + amount)
            setattr(self.totals, counter, getattr(self.totals, counter) + amount)
        self.event_ordinal += 1
        return None


@dataclass(frozen=True)
class _LaneResult:
    status: Literal["unique", "invalid", "budget-exhausted"]
    counters: WorkCounters
    detail: LaneTerminalDetail | None
    cuts: tuple[int, ...] | None
    denied: DeniedCharge | None


def _joined_length(prefix: tuple[int, ...], lo: int, hi: int, iso: str) -> int:
    chars = prefix[hi] - prefix[lo]
    if iso not in LANGUAGES_WITHOUT_SPACES:
        chars += max(0, hi - lo - 1)
    return chars


def _join_surfaces(surfaces: tuple[str, ...], lo: int, hi: int, iso: str) -> str:
    separator = "" if iso in LANGUAGES_WITHOUT_SPACES else " "
    return separator.join(surfaces[lo:hi])


def _run_allocator_lane(
    blocks: tuple[AuthorityBlock, ...],
    surfaces: tuple[str, ...],
    iso: str,
    budget: _BudgetLedger,
) -> _LaneResult:
    lane = _MutableCounters()
    block_cache: dict[int, str] = {}
    interval_cache: dict[tuple[int, int], str] = {}
    prefix = [0]
    for surface in surfaces:
        prefix.append(prefix[-1] + len(surface))
    prefix_tuple = tuple(prefix)
    ways = [dict() for _ in range(len(blocks) + 1)]
    predecessor: dict[tuple[int, int], int] = {}
    for index, block in enumerate(blocks):
        denied = budget.reserve(
            "allocator",
            lane,
            "block-normalize",
            (index,),
            WorkCounters(0, 0, 0, len(block.alignment_text)),
        )
        if denied:
            return _LaneResult(
                "budget-exhausted", lane.frozen(), "allocation-budget", None, denied
            )
        block_cache[index] = normalize_text(block.alignment_text)
        if not block.alignment_text.strip():
            return _LaneResult(
                "invalid", lane.frozen(), "partial-empty-ownership", None, None
            )
        if not block_cache[index]:
            return _LaneResult(
                "invalid", lane.frozen(), "punctuation-only-block", None, None
            )
    denied = budget.reserve(
        "allocator", lane, "state-insert", (0, 0), WorkCounters(1, 0, 0, 0)
    )
    if denied:
        return _LaneResult(
            "budget-exhausted", lane.frozen(), "allocation-budget", None, denied
        )
    ways[0][0] = 1
    for index in range(len(blocks)):
        for lo in sorted(ways[index]):
            for hi in range(lo + 1, len(surfaces) + 1):
                denied = budget.reserve(
                    "allocator",
                    lane,
                    "edge-test",
                    (index, lo, hi),
                    WorkCounters(0, 1, 0, 0),
                )
                if denied:
                    return _LaneResult(
                        "budget-exhausted",
                        lane.frozen(),
                        "allocation-budget",
                        None,
                        denied,
                    )
                key = (lo, hi)
                if key not in interval_cache:
                    denied = budget.reserve(
                        "allocator",
                        lane,
                        "interval-normalize",
                        key,
                        WorkCounters(
                            0, 0, 1, _joined_length(prefix_tuple, lo, hi, iso)
                        ),
                    )
                    if denied:
                        return _LaneResult(
                            "budget-exhausted",
                            lane.frozen(),
                            "allocation-budget",
                            None,
                            denied,
                        )
                    interval_cache[key] = normalize_text(
                        _join_surfaces(surfaces, lo, hi, iso)
                    )
                if block_cache[index] != interval_cache[key]:
                    continue
                if hi not in ways[index + 1]:
                    denied = budget.reserve(
                        "allocator",
                        lane,
                        "state-insert",
                        (index + 1, hi),
                        WorkCounters(1, 0, 0, 0),
                    )
                    if denied:
                        return _LaneResult(
                            "budget-exhausted",
                            lane.frozen(),
                            "allocation-budget",
                            None,
                            denied,
                        )
                    ways[index + 1][hi] = 0
                ways[index + 1][hi] = min(2, ways[index + 1][hi] + ways[index][lo])
                if ways[index + 1][hi] == 1:
                    predecessor[index + 1, hi] = lo
                else:
                    predecessor.pop((index + 1, hi), None)
    count = ways[len(blocks)].get(len(surfaces), 0)
    if count == 0:
        return _LaneResult("invalid", lane.frozen(), "allocation-no-tiling", None, None)
    if count == 2:
        return _LaneResult("invalid", lane.frozen(), "allocation-ambiguous", None, None)
    cuts = [len(surfaces)]
    boundary = len(surfaces)
    for row in range(len(blocks), 0, -1):
        boundary = predecessor[row, boundary]
        cuts.append(boundary)
    cuts.reverse()
    return _LaneResult("unique", lane.frozen(), None, tuple(cuts), None)


def _run_verifier_lane(
    blocks: tuple[AuthorityBlock, ...],
    surfaces: tuple[str, ...],
    iso: str,
    budget: _BudgetLedger,
) -> _LaneResult:
    # Separate caches, ways, predecessors, and traversal body are intentional:
    # the verifier cannot consume or warm the allocator's mutable search state.
    lane = _MutableCounters()
    normalized_blocks: dict[int, str] = {}
    normalized_intervals: dict[tuple[int, int], str] = {}
    prefix_lengths = [0]
    for surface in surfaces:
        prefix_lengths.append(prefix_lengths[-1] + len(surface))
    prefix_tuple = tuple(prefix_lengths)
    reachable = [dict() for _ in range(len(blocks) + 1)]
    parents: dict[tuple[int, int], int] = {}
    for block_index, block in enumerate(blocks):
        denied = budget.reserve(
            "verifier",
            lane,
            "block-normalize",
            (block_index,),
            WorkCounters(0, 0, 0, len(block.alignment_text)),
        )
        if denied:
            return _LaneResult(
                "budget-exhausted", lane.frozen(), "allocation-budget", None, denied
            )
        normalized_blocks[block_index] = normalize_text(block.alignment_text)
        if not block.alignment_text.strip():
            return _LaneResult(
                "invalid", lane.frozen(), "partial-empty-ownership", None, None
            )
        if not normalized_blocks[block_index]:
            return _LaneResult(
                "invalid", lane.frozen(), "punctuation-only-block", None, None
            )
    denied = budget.reserve(
        "verifier", lane, "state-insert", (0, 0), WorkCounters(1, 0, 0, 0)
    )
    if denied:
        return _LaneResult(
            "budget-exhausted", lane.frozen(), "allocation-budget", None, denied
        )
    reachable[0][0] = 1
    for block_index in range(len(blocks)):
        for lower in sorted(reachable[block_index]):
            for upper in range(lower + 1, len(surfaces) + 1):
                denied = budget.reserve(
                    "verifier",
                    lane,
                    "edge-test",
                    (block_index, lower, upper),
                    WorkCounters(0, 1, 0, 0),
                )
                if denied:
                    return _LaneResult(
                        "budget-exhausted",
                        lane.frozen(),
                        "allocation-budget",
                        None,
                        denied,
                    )
                interval = (lower, upper)
                if interval not in normalized_intervals:
                    denied = budget.reserve(
                        "verifier",
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
                    if denied:
                        return _LaneResult(
                            "budget-exhausted",
                            lane.frozen(),
                            "allocation-budget",
                            None,
                            denied,
                        )
                    normalized_intervals[interval] = normalize_text(
                        _join_surfaces(surfaces, lower, upper, iso)
                    )
                if normalized_blocks[block_index] != normalized_intervals[interval]:
                    continue
                if upper not in reachable[block_index + 1]:
                    denied = budget.reserve(
                        "verifier",
                        lane,
                        "state-insert",
                        (block_index + 1, upper),
                        WorkCounters(1, 0, 0, 0),
                    )
                    if denied:
                        return _LaneResult(
                            "budget-exhausted",
                            lane.frozen(),
                            "allocation-budget",
                            None,
                            denied,
                        )
                    reachable[block_index + 1][upper] = 0
                reachable[block_index + 1][upper] = min(
                    2,
                    reachable[block_index + 1][upper] + reachable[block_index][lower],
                )
                if reachable[block_index + 1][upper] == 1:
                    parents[block_index + 1, upper] = lower
                else:
                    parents.pop((block_index + 1, upper), None)
    count = reachable[len(blocks)].get(len(surfaces), 0)
    if count == 0:
        return _LaneResult("invalid", lane.frozen(), "allocation-no-tiling", None, None)
    if count == 2:
        return _LaneResult("invalid", lane.frozen(), "allocation-ambiguous", None, None)
    cuts = [len(surfaces)]
    boundary = len(surfaces)
    for row in range(len(blocks), 0, -1):
        boundary = parents[row, boundary]
        cuts.append(boundary)
    cuts.reverse()
    return _LaneResult("unique", lane.frozen(), None, tuple(cuts), None)


def _surface_chars(
    call: AuthorityCallInput, blocks_by_source: dict[int, AuthorityBlock]
) -> int | None:
    if call.unit_surfaces is None:
        return None
    return sum(
        len(blocks_by_source[source_index].alignment_text)
        for source_index in call.source_block_indices
    ) + sum(len(surface) for surface in call.unit_surfaces)


def _base_call_receipt(
    call: AuthorityCallInput,
    claims: tuple[RouteClaim, ...],
    blocks_by_source: dict[int, AuthorityBlock],
    limits: CallWorkLimits,
    *,
    prior_terminal: bool = False,
) -> AuthorityCallWorkReceipt:
    positions = tuple(
        position
        for position, claim in enumerate(claims)
        if claim.owner_kind == "call" and claim.owner_index == call.call_index
    )
    lane_status = "not-run-prior-terminal" if prior_terminal else "not-run"
    return AuthorityCallWorkReceipt(
        call_index=call.call_index,
        route_claim_positions=positions,
        source_block_indices=call.source_block_indices,
        raw_node_range=call.raw_node_range,
        block_count=len(call.source_block_indices),
        raw_node_count=len(call.raw_unit_ids),
        typed_unit_count=(
            len(call.unit_surfaces) if call.unit_surfaces is not None else None
        ),
        surface_chars=_surface_chars(call, blocks_by_source),
        strict_preflight_status=call.strict_preflight_status,
        strict_failure=call.strict_failure,
        limits=limits,
        allocator=WorkLaneReceipt(lane_status, ZERO_COUNTERS, None),
        verifier=None,
    )


def _skip_receipts(
    skipped: tuple[AuthoritySkippedBlockInput, ...], claims: tuple[RouteClaim, ...]
) -> tuple[AuthoritySkippedBlockReceipt, ...]:
    return tuple(
        AuthoritySkippedBlockReceipt(
            route_claim_positions=tuple(
                position
                for position, claim in enumerate(claims)
                if claim.owner_kind == "skip" and claim.owner_index == skip_index
            ),
            delivery_index=skip.delivery_index,
            source_index=skip.source_index,
            route_skip_reason=skip.route_skip_reason,
            source_text_kind=skip.source_text_kind,
            detail_code="partial-empty-ownership",
            work_status="not-run",
            counters=ZERO_COUNTERS,
        )
        for skip_index, skip in enumerate(skipped)
    )


_STRICT_STAGE_ORDER = ("strict-capture", "sample-geometry", "authority-transform")
_STRICT_DETAIL_ORDER = (
    "strict-raw-node",
    "sample-geometry",
    "physical-origin-mismatch",
    "authority-recompute",
    "surplus-transform",
)


def _first_strict_call(calls: tuple[AuthorityCallInput, ...]) -> int | None:
    facts = [
        (
            call.call_index,
            -1
            if call.strict_failure.call_unit_index is None
            else call.strict_failure.call_unit_index,
            _STRICT_STAGE_ORDER.index(call.strict_failure.stage),
            _STRICT_DETAIL_ORDER.index(call.strict_failure.detail_code),
        )
        for call in calls
        if call.strict_failure is not None
    ]
    return min(facts)[0] if facts else None


def project_authority_reasons(work: AuthorityJobWorkReceipt) -> tuple[str, ...]:
    if work.status == "seal-mismatch":
        return ()
    present: set[str] = set()
    if work.route_status == "invalid":
        present.add("route-owner-mismatch")
    if work.skipped_blocks:
        present.add("partial-empty-ownership")
    if any(call.strict_failure is not None for call in work.calls):
        present.add("authority-transform-invalid")
    detail_reason = {
        "partial-empty-ownership": "partial-empty-ownership",
        "punctuation-only-block": "punctuation-only-block",
        "allocation-no-tiling": "allocation-no-tiling",
        "allocation-ambiguous": "allocation-ambiguous",
    }
    for call in work.calls:
        detail = call.allocator.terminal_detail_code
        if detail in detail_reason:
            present.add(detail_reason[detail])
    if work.status == "budget-exhausted":
        present.add("allocation-budget-exhausted")
    return tuple(reason for reason in AUTHORITY_REASON_ORDER if reason in present)


def _job_receipt(
    *,
    status: Any,
    mismatch: RouteMismatch | None,
    route_claims: tuple[RouteClaim, ...],
    route_count: int,
    calls: tuple[AuthorityCallInput, ...],
    skipped: tuple[AuthoritySkippedBlockInput, ...],
    profile: AuthorityLimitProfile,
    charged_calls: int,
    totals: WorkCounters,
    terminal_call_index: int | None,
    denied: DeniedCharge | None,
    call_rows: tuple[AuthorityCallWorkReceipt, ...],
    skip_rows: tuple[AuthoritySkippedBlockReceipt, ...],
) -> AuthorityJobWorkReceipt:
    return AuthorityJobWorkReceipt(
        status=status,
        route_status="invalid" if mismatch is not None else "valid",
        route_mismatch=mismatch,
        route_claims=route_claims,
        declared_delivery_block_count=route_count,
        declared_call_count=len(calls),
        declared_skip_count=len(skipped),
        declared_raw_node_count=sum(len(call.raw_unit_ids) for call in calls),
        charged_call_count=charged_calls,
        limit_profile_kind=profile.kind,
        limit_profile_digest=profile.profile_digest,
        limits=profile.job,
        totals=totals,
        terminal_call_index=terminal_call_index,
        denied_charge=denied,
        calls=call_rows,
        skipped_blocks=skip_rows,
    )


def _build_authority_distribution_impl(
    *,
    blocks: tuple[AuthorityBlock, ...],
    delivery_route: tuple[RouteExpectation, ...],
    calls: tuple[AuthorityCallInput, ...],
    skipped_blocks: tuple[AuthoritySkippedBlockInput, ...],
    route_claims: tuple[RouteClaim, ...],
    iso: str,
    profile: AuthorityLimitProfile,
    _verifier_cut_mutator: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
) -> AuthorityDistributionReceipt:
    """Run preflight, producer, and verifier under one sealed effective profile."""
    effective_profile = validate_authority_limit_profile(profile)
    blocks_by_source = {block.source_index: block for block in blocks}
    if len(blocks_by_source) != len(blocks):
        raise ValueError("authority blocks must have unique source indices")
    if tuple(call.call_index for call in calls) != tuple(range(len(calls))):
        raise ValueError("authority call indexes must be increasing and gap-free")
    expected_raw_start = 0
    for call in calls:
        if call.raw_node_range != (
            expected_raw_start,
            expected_raw_start + len(call.raw_unit_ids),
        ):
            raise ValueError(
                "authority raw-node ranges must be increasing and gap-free"
            )
        expected_raw_start = call.raw_node_range[1]
        if call.strict_preflight_status == "capture-invalid":
            if call.unit_surfaces is not None or call.strict_failure is None:
                raise ValueError("capture-invalid call has impossible strict facts")
        elif call.unit_surfaces is None:
            raise ValueError("capture-valid call requires every captured surface")
    mismatch = project_route_mismatch(
        route_claims, delivery_route, calls, skipped_blocks
    )
    base_rows = tuple(
        _base_call_receipt(call, route_claims, blocks_by_source, effective_profile.call)
        for call in calls
    )
    skip_rows = _skip_receipts(skipped_blocks, route_claims)
    all_raw_ids = tuple(unit_id for call in calls for unit_id in call.raw_unit_ids)
    preflight_status: str | None = None
    terminal_call: int | None = None
    if mismatch is not None:
        preflight_status = "not-run-route-invalid"
    elif skipped_blocks:
        preflight_status = "not-run-skip-invalid"
    elif any(call.strict_failure is not None for call in calls):
        preflight_status = "not-run-strict-unavailable"
        terminal_call = _first_strict_call(calls)
    if preflight_status is not None:
        work = _job_receipt(
            status=preflight_status,
            mismatch=mismatch,
            route_claims=route_claims,
            route_count=len(delivery_route),
            calls=calls,
            skipped=skipped_blocks,
            profile=effective_profile,
            charged_calls=0,
            totals=ZERO_COUNTERS,
            terminal_call_index=terminal_call,
            denied=None,
            call_rows=base_rows,
            skip_rows=skip_rows,
        )
        reasons = project_authority_reasons(work)
        return AuthorityDistributionReceipt(
            "invalid",
            tuple(item.source_index for item in delivery_route),
            None,
            None,
            reasons,
            0,
            all_raw_ids,
            work,
        )

    budget = _BudgetLedger(effective_profile)
    rows: list[AuthorityCallWorkReceipt] = []
    owners_by_source: dict[int, tuple[str, ...]] = {}
    job_status: str = "complete"
    denied_charge: DeniedCharge | None = None
    terminal_call_index: int | None = None
    stopped = False
    for call in calls:
        if stopped:
            rows.append(
                _base_call_receipt(
                    call,
                    route_claims,
                    blocks_by_source,
                    effective_profile.call,
                    prior_terminal=True,
                )
            )
            continue
        call_denied = budget.reserve_call(call.call_index)
        if call_denied is not None:
            row = _base_call_receipt(
                call, route_claims, blocks_by_source, effective_profile.call
            )
            rows.append(
                replace(
                    row,
                    allocator=WorkLaneReceipt(
                        "budget-exhausted", ZERO_COUNTERS, "allocation-budget"
                    ),
                )
            )
            job_status = "budget-exhausted"
            denied_charge = call_denied
            terminal_call_index = call.call_index
            stopped = True
            continue
        assert call.unit_surfaces is not None
        call_blocks = tuple(
            blocks_by_source[source_index] for source_index in call.source_block_indices
        )
        allocator = _run_allocator_lane(call_blocks, call.unit_surfaces, iso, budget)
        row = _base_call_receipt(
            call, route_claims, blocks_by_source, effective_profile.call
        )
        allocator_receipt = WorkLaneReceipt(
            "complete" if allocator.status == "unique" else allocator.status,
            allocator.counters,
            allocator.detail,
        )
        if allocator.status != "unique":
            rows.append(replace(row, allocator=allocator_receipt))
            job_status = (
                "budget-exhausted"
                if allocator.status == "budget-exhausted"
                else "invalid"
            )
            denied_charge = allocator.denied
            terminal_call_index = call.call_index
            stopped = True
            continue
        verifier = _run_verifier_lane(call_blocks, call.unit_surfaces, iso, budget)
        verifier_receipt = WorkLaneReceipt(
            "complete" if verifier.status == "unique" else verifier.status,
            verifier.counters,
            verifier.detail,
        )
        if verifier.status == "budget-exhausted":
            rows.append(
                replace(
                    row,
                    allocator=allocator_receipt,
                    verifier=verifier_receipt,
                )
            )
            job_status = "budget-exhausted"
            denied_charge = verifier.denied
            terminal_call_index = call.call_index
            stopped = True
            continue
        verifier_cuts = verifier.cuts
        if verifier_cuts is not None and _verifier_cut_mutator is not None:
            verifier_cuts = _verifier_cut_mutator(verifier_cuts)
        if verifier.status != "unique" or verifier_cuts != allocator.cuts:
            rows.append(
                replace(
                    row,
                    allocator=allocator_receipt,
                    verifier=WorkLaneReceipt(
                        "seal-mismatch",
                        verifier.counters,
                        "distribution-seal",
                    ),
                )
            )
            job_status = "seal-mismatch"
            terminal_call_index = call.call_index
            stopped = True
            continue
        assert allocator.cuts is not None
        owner_groups = tuple(
            call.raw_unit_ids[allocator.cuts[index] : allocator.cuts[index + 1]]
            for index in range(len(call.source_block_indices))
        )
        for source_index, owner in zip(call.source_block_indices, owner_groups):
            owners_by_source[source_index] = owner
        rows.append(
            replace(
                row,
                allocator=allocator_receipt,
                verifier=verifier_receipt,
            )
        )

    assert budget.totals is not None
    work = _job_receipt(
        status=job_status,
        mismatch=None,
        route_claims=route_claims,
        route_count=len(delivery_route),
        calls=calls,
        skipped=skipped_blocks,
        profile=effective_profile,
        charged_calls=budget.calls,
        totals=budget.totals.frozen(),
        terminal_call_index=terminal_call_index,
        denied=denied_charge,
        call_rows=tuple(rows),
        skip_rows=skip_rows,
    )
    reasons = project_authority_reasons(work)
    owner_sources = tuple(item.source_index for item in delivery_route)
    if job_status != "complete":
        return AuthorityDistributionReceipt(
            "invalid",
            owner_sources,
            None,
            None,
            reasons,
            0,
            all_raw_ids,
            work,
        )
    owners = tuple(owners_by_source[source] for source in owner_sources)
    return AuthorityDistributionReceipt(
        "valid",
        owner_sources,
        owners,
        tuple(len(owner) for owner in owners),
        (),
        len(all_raw_ids),
        (),
        work,
    )


def build_authority_distribution(
    *,
    blocks: tuple[AuthorityBlock, ...],
    delivery_route: tuple[RouteExpectation, ...],
    calls: tuple[AuthorityCallInput, ...],
    skipped_blocks: tuple[AuthoritySkippedBlockInput, ...],
    route_claims: tuple[RouteClaim, ...],
    iso: str,
) -> AuthorityDistributionReceipt:
    """Build under the one effective profile captured in the current scope."""
    return _build_authority_distribution_impl(
        blocks=blocks,
        delivery_route=delivery_route,
        calls=calls,
        skipped_blocks=skipped_blocks,
        route_claims=route_claims,
        iso=iso,
        profile=capture_authority_limit_profile(),
    )


def _build_context_authority_distribution(
    *,
    blocks: tuple[AuthorityBlock, ...],
    delivery_route: tuple[RouteExpectation, ...],
    calls: tuple[AuthorityCallInput, ...],
    skipped_blocks: tuple[AuthoritySkippedBlockInput, ...],
    route_claims: tuple[RouteClaim, ...],
    iso: str,
    _limits: AuthorityLimitProfile,
    _verifier_cut_mutator: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
) -> AuthorityDistributionReceipt:
    """Private context-bound path used only by the sole acquisition issuer."""
    return _build_authority_distribution_impl(
        blocks=blocks,
        delivery_route=delivery_route,
        calls=calls,
        skipped_blocks=skipped_blocks,
        route_claims=route_claims,
        iso=iso,
        profile=_limits,
        _verifier_cut_mutator=_verifier_cut_mutator,
    )


def project_authority_failure(
    receipt: AuthorityDistributionReceipt,
) -> CanonicalFailure | None:
    work = receipt.work
    if work.status == "complete":
        return None
    if work.status == "not-run-route-invalid":
        return CanonicalFailure(
            "fresh-distribution-invalid",
            "authority-distribution",
            "route-owner-mismatch",
        )
    if work.status == "not-run-skip-invalid":
        return CanonicalFailure(
            "fresh-distribution-invalid",
            "authority-distribution",
            "partial-empty-ownership",
        )
    if work.status == "not-run-strict-unavailable":
        locators = [
            (call.call_index, call.strict_failure)
            for call in work.calls
            if call.strict_failure is not None
        ]
        _index, locator = min(
            locators,
            key=lambda item: (
                item[0],
                -1 if item[1].call_unit_index is None else item[1].call_unit_index,
                _STRICT_STAGE_ORDER.index(item[1].stage),
                _STRICT_DETAIL_ORDER.index(item[1].detail_code),
            ),
        )
        return CanonicalFailure(
            "fresh-time-transform-invalid", locator.stage, locator.detail_code
        )
    if work.status == "budget-exhausted":
        return CanonicalFailure(
            "fresh-distribution-invalid",
            "authority-distribution",
            "allocation-budget",
        )
    if work.status == "seal-mismatch":
        return CanonicalFailure(
            "fresh-seal-broken", "authority-distribution", "distribution-seal"
        )
    for call in work.calls:
        detail = call.allocator.terminal_detail_code
        if detail in {
            "partial-empty-ownership",
            "punctuation-only-block",
            "allocation-no-tiling",
            "allocation-ambiguous",
        }:
            return CanonicalFailure(
                "fresh-distribution-invalid", "authority-distribution", detail
            )
    raise ValueError("invalid authority receipt has no canonical failure projection")


@dataclass(frozen=True)
class LegacyCallDistributionReceipt:
    owner_source_indices: tuple[int, ...]
    expected_counts: tuple[int, ...]
    requested_ranges: tuple[tuple[int, int], ...]
    realized_ranges: tuple[tuple[int, int], ...]
    owner_unit_ids: tuple[tuple[str, ...], ...]
    final_cursor: int
    consumed_prefix_unit_ids: tuple[str, ...]
    shortage_source_indices: tuple[int, ...]
    leftover_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class LegacyDistributionResult:
    block_units: tuple[tuple[Any, ...], ...]
    receipt: LegacyCallDistributionReceipt
    relative_block_units: tuple[tuple[Any, ...], ...]


def _legacy_count(text: str, iso: str) -> int:
    stripped = (text or "").strip()
    if iso in LANGUAGES_WITHOUT_SPACES:
        return sum(1 for character in stripped if character.isalnum())
    return len(stripped.split())


def legacy_distribute_before_shift(
    relative_flat: Sequence[Any],
    *,
    texts: tuple[str, ...],
    iso: str,
    origin: float,
    identity: bool,
    raw_unit_ids: tuple[str, ...],
    source_indices: tuple[int, ...] | None = None,
) -> LegacyDistributionResult:
    """Freeze count-only slices, then transform only their retained members."""
    if len(raw_unit_ids) != len(relative_flat):
        raise ValueError("raw unit ids must cover the complete flat result")
    owner_sources = (
        tuple(range(len(texts))) if source_indices is None else tuple(source_indices)
    )
    if (
        len(owner_sources) != len(texts)
        or len(set(owner_sources)) != len(owner_sources)
        or any(type(index) is not int or index < 0 for index in owner_sources)
    ):
        raise ValueError("legacy source indices must be unique nonnegative integers")
    requested: list[tuple[int, int]] = []
    realized: list[tuple[int, int]] = []
    expected: list[int] = []
    owner_ids: list[tuple[str, ...]] = []
    retained: list[Sequence[Any]] = []
    shortages: list[int] = []
    cursor = 0
    raw_count = len(relative_flat)
    for delivery_index, text in enumerate(texts):
        count = _legacy_count(text, iso)
        lower, upper = cursor, cursor + count
        requested.append((lower, upper))
        realized_lower = min(lower, raw_count)
        realized_upper = min(upper, raw_count)
        realized.append((realized_lower, realized_upper))
        expected.append(count)
        retained.append(relative_flat[lower:upper])
        owner_ids.append(raw_unit_ids[lower:upper])
        if realized_upper - realized_lower < count:
            shortages.append(owner_sources[delivery_index])
        cursor += count
    relative_projected = tuple(
        tuple(
            {
                "text": unit["text"],
                "start": unit["start"],
                "end": unit["end"],
            }
            for unit in group
        )
        for group in retained
    )
    if identity:
        projected = tuple(
            tuple(dict(unit) for unit in group) for group in relative_projected
        )
    else:
        projected = tuple(
            tuple(shift_units(list(group), origin)) for group in relative_projected
        )
    consumed = min(cursor, raw_count)
    receipt = LegacyCallDistributionReceipt(
        owner_source_indices=owner_sources,
        expected_counts=tuple(expected),
        requested_ranges=tuple(requested),
        realized_ranges=tuple(realized),
        owner_unit_ids=tuple(owner_ids),
        final_cursor=cursor,
        consumed_prefix_unit_ids=raw_unit_ids[:consumed],
        shortage_source_indices=tuple(shortages),
        leftover_unit_ids=raw_unit_ids[consumed:],
    )
    return LegacyDistributionResult(projected, receipt, relative_projected)


def legacy_retain_qwen_before_shift(
    relative_flat: Sequence[Any],
    *,
    origin: float,
    identity: bool,
    raw_unit_ids: tuple[str, ...],
    source_indices: tuple[int, ...],
) -> LegacyDistributionResult:
    """Assign one Qwen call's complete raw result to its sole cue owner."""
    if len(raw_unit_ids) != len(relative_flat):
        raise ValueError("raw unit ids must cover the complete flat result")
    if (
        len(source_indices) != 1
        or type(source_indices[0]) is not int
        or source_indices[0] < 0
    ):
        raise ValueError("a Qwen call must have one nonnegative source owner")
    relative_owner = tuple(
        {
            "text": unit["text"],
            "start": unit["start"],
            "end": unit["end"],
        }
        for unit in relative_flat
    )
    if identity:
        owner = tuple(dict(unit) for unit in relative_owner)
    else:
        owner = tuple(shift_units(list(relative_owner), origin))
    raw_count = len(relative_flat)
    receipt = LegacyCallDistributionReceipt(
        owner_source_indices=source_indices,
        expected_counts=(raw_count,),
        requested_ranges=((0, raw_count),),
        realized_ranges=((0, raw_count),),
        owner_unit_ids=(raw_unit_ids,),
        final_cursor=raw_count,
        consumed_prefix_unit_ids=raw_unit_ids,
        shortage_source_indices=(),
        leftover_unit_ids=(),
    )
    return LegacyDistributionResult((owner,), receipt, (relative_owner,))
