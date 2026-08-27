"""Sealed issuance: proving which upstream authority a finalizer root came from.

The gate this module exists for (P5 spec section 2.6, N8b) is not "did the right
*type* reach ``finalize``". An ``isinstance`` check is satisfied by anything a
test -- or a well-meaning future call site -- can construct, so it proves only
that someone built the right shape. The question that matters is narrower: did
the stream a row delivered descend from THAT row's own upstream authority, and
did nothing else even try?

Three properties answer it:

* **authority is issuer identity plus a sealed payload digest.** A :class:`Seal`
  is minted only by an :class:`AuthorityLedger`, so a structurally perfect
  hand-built seal has no ledger entry and is rejected as unissued; and because
  the seal covers a digest of the payload, an edit made after issuance breaks
  verification instead of riding along.
* **the capability is single-use.** ``finalize`` consumes it, so a second
  finalize over the same seed is a raise rather than a quietly duplicated root.
* **an unused laundering attempt is itself a failure.** A recorded event nobody
  consumed still says someone minted a root the matrix did not expect, which is
  why :func:`check_roots` fails on unexpected events rather than only on
  unexpected deliveries.

Stdlib only, by design: this module sits below the finalizer and must never
drag a solver, a lattice or a cost model into a chain of custody check.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "AUTHORITY_KINDS",
    "LINEAGE_FIELDS",
    "AuthorityKind",
    "AuthorityLedger",
    "Capability",
    "CapabilityConsumed",
    "FactoryEvent",
    "LineageRecord",
    "Seal",
    "SealBroken",
    "UnissuedAuthority",
    "check_roots",
    "digest_payload",
    "lineage_tuples",
    "seal_chain",
]

AuthorityKind = Literal["v1-capture", "optimizer-selection"]

#: Closed and sorted: a row's seed chain must terminate in one of exactly these.
AUTHORITY_KINDS: tuple[AuthorityKind, ...] = ("optimizer-selection", "v1-capture")

#: The N8b probe record, in the spec's own field order. Stated once so the probe
#: and its tests cannot drift apart about which fields a lineage tuple carries.
LINEAGE_FIELDS: tuple[str, ...] = (
    "evaluation_id",
    "row_id",
    "call_id",
    "input_seed_id",
    "input_kind",
    "parent_finalize_call_id",
)

#: One probe record: the six fields above, positionally.
LineageRecord = tuple[str, str, str, str, str, str | None]


class UnissuedAuthority(RuntimeError):
    """A seal no ledger issued -- including a structurally valid hand-built one."""


class SealBroken(RuntimeError):
    """The sealed payload changed after issuance, so the seal no longer covers it."""


class CapabilityConsumed(RuntimeError):
    """A single-use capability was used twice; the second root is not a root."""


def _json_default(value: Any) -> Any:
    """Last-resort projection for a payload member JSON cannot encode.

    ``repr`` rather than a silent drop: an unencodable member must still change
    the digest, otherwise it is a hole an edit could hide in.
    """
    return repr(value)


def digest_payload(payload: Any) -> str:
    """sha256 over the canonical JSON of a normalized payload projection.

    ``sort_keys`` plus the compact separators make the digest independent of
    mapping insertion order and of ``PYTHONHASHSEED``, so two runs of the same
    evaluation seal to the same bytes.
    """
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Seal:
    """Issuer identity plus the digest of what was sealed.

    Authority is never ``isinstance`` (R8-1): the pair (``authority_id``,
    ``digest``) is what a ledger can confirm it minted, and both halves are
    needed -- the id alone would let a stale seal cover fresh content, the
    digest alone would let anyone mint one.
    """

    issuer: str
    authority_id: str
    kind: AuthorityKind
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "digest": self.digest,
            "issuer": self.issuer,
            "kind": self.kind,
        }


@dataclass
class Capability:
    """A single-use right to mint one root from one sealed payload.

    Mutable on purpose: consumption is state, and modelling it as a returned
    copy would let a caller keep the unconsumed original.
    """

    seal: Seal
    ledger: AuthorityLedger | None = None
    consumed: bool = field(default=False)

    def consume(self, payload: Any) -> Seal:
        """Verify the seal against ``payload`` and spend the capability.

        Raises :class:`CapabilityConsumed` on a second use,
        :class:`UnissuedAuthority` when no issuing ledger stands behind the
        seal, and :class:`SealBroken` when the payload no longer digests to what
        was sealed.
        """
        if self.consumed:
            raise CapabilityConsumed(
                f"capability {self.seal.authority_id!r} was already consumed"
            )
        if self.ledger is None:
            raise UnissuedAuthority(
                f"capability {self.seal.authority_id!r} carries no issuing ledger"
            )
        self.ledger.verify(self.seal, payload)
        self.consumed = True
        return self.seal


@dataclass(frozen=True)
class FactoryEvent:
    """One recorded root minting, in the shape the N8b probe reads.

    ``input_kind`` is recorded rather than inferred because the failure this
    gate exists for is precisely a root seeded from a *delivered* stream: the
    shape of such a stream is indistinguishable from a phase-1 one, and only the
    producer knows which it handed over.
    """

    evaluation_id: str
    row_id: str
    call_id: str
    input_seed_id: str
    input_kind: str
    parent_finalize_call_id: str | None
    authority_kind: AuthorityKind
    authority_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "authority_kind": self.authority_kind,
            "call_id": self.call_id,
            "evaluation_id": self.evaluation_id,
            "input_kind": self.input_kind,
            "input_seed_id": self.input_seed_id,
            "parent_finalize_call_id": self.parent_finalize_call_id,
            "row_id": self.row_id,
        }


class AuthorityLedger:
    """Per-evaluation registry of every seal minted and every root recorded.

    Seals are minted here and nowhere else. That is the whole mechanism: a
    forged seal cannot be in the registry, and a registered one cannot cover
    content it did not cover at issuance.
    """

    def __init__(self) -> None:
        self._issued: dict[str, Seal] = {}
        self._events: list[FactoryEvent] = []
        self._ids = itertools.count(1)

    def issue(self, *, issuer: str, kind: AuthorityKind, payload: Any) -> Capability:
        """Mint one capability over ``payload``; the digest is taken now."""
        authority_id = f"a{next(self._ids)}"
        seal = Seal(
            issuer=issuer,
            authority_id=authority_id,
            kind=kind,
            digest=digest_payload(payload),
        )
        self._issued[authority_id] = seal
        return Capability(seal=seal, ledger=self)

    def verify(self, seal: Seal, payload: Any) -> None:
        """Confirm this ledger issued ``seal`` and that it still covers ``payload``."""
        issued = self._issued.get(seal.authority_id)
        if issued is None or issued != seal:
            raise UnissuedAuthority(
                f"seal {seal.authority_id!r} from {seal.issuer!r} was not issued here"
            )
        if digest_payload(payload) != seal.digest:
            raise SealBroken(
                f"payload under seal {seal.authority_id!r} changed after issuance"
            )

    def record(self, event: FactoryEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> tuple[FactoryEvent, ...]:
        return tuple(self._events)

    @property
    def issued(self) -> tuple[Seal, ...]:
        return tuple(self._issued[key] for key in sorted(self._issued))

    def seal_for(self, authority_id: str) -> Seal | None:
        """The seal this ledger minted under ``authority_id``, if it minted one.

        ``None`` is not an error here: a probe may run over a ledger rebuilt from
        recorded events alone, where the seals live in the run that produced
        them. It IS an error for a chain link the ledger does say it issued to
        carry the wrong kind, which is what :func:`check_roots` reads this for.
        """
        return self._issued.get(authority_id)


def lineage_tuples(ledger: AuthorityLedger) -> tuple[LineageRecord, ...]:
    """Every recorded root as the N8b probe record, sorted.

    The probe is a tuple rather than the event object because it is a
    *published* record: the harness writes it into the artifact, a reviewer reads
    it, and the fields are exactly the six the gate reasons about. ``input_kind``
    is the load-bearing one -- a root seeded from a DELIVERED stream is
    structurally indistinguishable from one seeded from phase 1, so only the
    producer can say which it handed over, and this is where it says so.
    """
    records: list[LineageRecord] = [
        (
            event.evaluation_id,
            event.row_id,
            event.call_id,
            event.input_seed_id,
            event.input_kind,
            event.parent_finalize_call_id,
        )
        for event in ledger.events
    ]
    # ``None`` is a legal value in the last field, so the sort key substitutes a
    # string for it rather than letting the comparison raise.
    return tuple(
        sorted(
            records,
            key=lambda record: tuple("" if item is None else item for item in record),
        )
    )


def seal_chain(ledger: AuthorityLedger, event: FactoryEvent) -> tuple[Seal, ...]:
    """One root's chain of custody, nearest link first.

    Link 0 is the root's own sealed seed; link 1 is the upstream authority that
    seed was minted from -- the hook's v1 capture or the row's own optimizer
    selection. Links the ledger cannot resolve are omitted rather than faked, so
    a caller can tell "the chain ends in the wrong kind" from "this ledger never
    saw the chain".
    """
    chain: list[Seal] = []
    for authority_id in (event.authority_id, event.input_seed_id):
        seal = ledger.seal_for(authority_id)
        if seal is not None:
            chain.append(seal)
    return tuple(chain)


def check_roots(
    ledger: AuthorityLedger, *, expected: Mapping[str, AuthorityKind]
) -> tuple[str, ...]:
    """The N8b gate. Returns the violations; an empty tuple is a pass.

    Checked, in the spec's own order: exactly one finalize root per expected row
    and no root for an unexpected one; ``input_kind == "phase1"`` for every root;
    no root parented by a finalize call (a re-finalize is not a root, which is
    what keeps a ``FinalizeResult`` out of every chain); each row's seed chain
    terminating in the authority kind that row is entitled to -- resolved through
    the ledger's own issuance record, not read off the event's self-declaration,
    wherever the ledger issued the link; and no unexpected authority event, which
    includes an authority that was minted and never used. An unused laundering
    attempt is itself a failure: it says someone built a root the matrix did not
    expect, and whether they got round to delivering it is not the question.
    """
    problems: list[str] = []
    by_row: dict[str, list[FactoryEvent]] = {}
    for event in ledger.events:
        by_row.setdefault(event.row_id, []).append(event)

    for row, events in sorted(by_row.items()):
        if row not in expected:
            problems.append(
                f"unexpected row {row!r} minted {len(events)} finalizer root(s)"
            )
            continue
        if len(events) != 1:
            problems.append(
                f"row {row!r} minted {len(events)} roots, expected exactly one"
            )
        for event in events:
            if event.input_kind != "phase1":
                problems.append(
                    f"row {row!r} root {event.call_id!r} was seeded from "
                    f"{event.input_kind!r}, not phase1"
                )
            if event.parent_finalize_call_id is not None:
                problems.append(
                    f"row {row!r} root {event.call_id!r} descends from finalize call "
                    f"{event.parent_finalize_call_id!r}"
                )
            if event.authority_kind != expected[row]:
                problems.append(
                    f"row {row!r} root {event.call_id!r} carries authority "
                    f"{event.authority_kind!r}, expected {expected[row]!r}"
                )
            chain = seal_chain(ledger, event)
            if chain and chain[-1].kind != expected[row]:
                problems.append(
                    f"row {row!r} root {event.call_id!r} has a seed chain "
                    f"terminating in {chain[-1].kind!r} "
                    f"(issued by {chain[-1].issuer!r}), expected {expected[row]!r}"
                )

    for row in sorted(expected):
        if row not in by_row:
            problems.append(f"row {row!r} minted no finalizer root")

    referenced = {event.authority_id for event in ledger.events} | {
        event.input_seed_id for event in ledger.events
    }
    for seal in ledger.issued:
        if seal.authority_id not in referenced:
            problems.append(
                f"authority {seal.authority_id!r} ({seal.kind}) was minted by "
                f"{seal.issuer!r} and reached no finalizer root"
            )
    return tuple(problems)
