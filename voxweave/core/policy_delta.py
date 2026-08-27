"""Frozen typed policy/finalizer delta registry from P5 LAW section 9.

The registry is canonical data, not prose attached to a version constant.  Its
serialization is deliberately owned here and byte-pinned by N17 so a changed
trigger, relation, direction, field set, or enforcement cannot masquerade as
the same policy contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DELTA_REGISTRY",
    "DeltaRecord",
    "delta_registry_bytes",
    "delta_registry_data",
]


@dataclass(frozen=True)
class DeltaRecord:
    """One closed LAW section 9 delta record."""

    id: str
    trigger: str
    affected_fields: tuple[str, ...]
    direction: str
    allowed_relation: str | None
    enforcement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trigger": self.trigger,
            "affected_fields": list(self.affected_fields),
            "direction": self.direction,
            "allowed_relation": self.allowed_relation,
            "enforcement": self.enforcement,
        }


DELTA_REGISTRY: tuple[DeltaRecord, ...] = (
    DeltaRecord(
        "FD-1",
        "canonical vs raw reading_chars differ",
        ("end",),
        "both",
        "delivered leg == classifier-recomputed phase-1 duration solve "
        "(independent reimplementation, mirror-pinned)",
        "N11",
    ),
    DeltaRecord(
        "FD-2",
        "evidence-span vs legacy display-span lyric classification differs",
        ("lyric",),
        "both",
        "flag == EvidenceSpan predicate output",
        "N11",
    ),
    DeltaRecord(
        "FD-3",
        "input pair overlapped",
        ("end",),
        "shrink",
        "exact ladder-branch target (branch recomputed from evidence)",
        "N11+N13",
    ),
    DeltaRecord(
        "FD-4",
        "zone/separation outcome differs from legacy sequential sweeps",
        ("start", "end"),
        "both",
        "every trace leg = validator-recomputed rule application on ITS "
        "reconstructed state (§10.2)",
        "N11 via trace",
    ),
    DeltaRecord(
        "FD-5",
        "v2 lane never applies the speaker overlay",
        ("v2 construction",),
        "n/a",
        None,
        "construction + N3b",
    ),
    DeltaRecord(
        "FD-6",
        "input gap < TWO_FRAME_S",
        ("end",),
        "shrink",
        "next_start − TWO_FRAME_S (b1) or speech_end + report (b2)",
        "N11+N13",
    ),
    DeltaRecord(
        "FD-7",
        "any veto/refusal fact",
        (),
        "n/a",
        None,
        "report equality",
    ),
    DeltaRecord(
        "FD-8",
        "anchorless cue where legacy would extend",
        ("end",),
        "both",
        "delivered end == phase-1 input end or a composed FD-3/4/6 leg chain "
        "(each leg checked)",
        "N11 via trace",
    ),
    DeltaRecord(
        "FD-9",
        "bounded stutter stable == False",
        ("text",),
        "n/a",
        "to == bounded_canonical(raw)",
        "N11 + injected fixture",
    ),
    DeltaRecord(
        "PD-TEXT",
        "edge admission = canonical legality",
        ("lattice edge set",),
        "n/a",
        None,
        "N14 both-direction oracle",
    ),
    DeltaRecord(
        "PD-SPK",
        "speaker edge term",
        ("selection",),
        "n/a",
        None,
        "speaker-off counterfactual",
    ),
    DeltaRecord(
        "PD-SUBUNIT",
        "refinement of coarse units",
        ("unit space",),
        "n/a",
        None,
        "refiner-off replay (§8)",
    ),
)


def delta_registry_data() -> list[dict[str, Any]]:
    """Return detached artifact data in the registry's frozen order."""
    return [record.to_dict() for record in DELTA_REGISTRY]


def delta_registry_bytes() -> bytes:
    """Return the N17 canonical UTF-8 serialization."""
    return json.dumps(
        delta_registry_data(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
