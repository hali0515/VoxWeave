"""Dependency-neutral registry for P6 align primitive deltas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class AlignDeltaDefinition:
    delta_id: str
    title: str
    ratification: Literal["RAT-5"] | None
    phase: Literal["lazy-semantic", "mandatory-core"]
    primitive_fields: tuple[str, ...]
    relation: str


_REGISTRY = {
    "ALD-0": AlignDeltaDefinition(
        "ALD-0",
        "Qwen physical origin",
        "RAT-5",
        "lazy-semantic",
        ("authority-times", "anchors", "seed"),
        "physical-minus-nominal origin only",
    ),
    "ALD-1": AlignDeltaDefinition(
        "ALD-1",
        "canonical text/layout",
        None,
        "lazy-semantic",
        ("text",),
        "exact independent phase-one text and owner footprint",
    ),
    "ALD-2": AlignDeltaDefinition(
        "ALD-2",
        "duration desire",
        None,
        "lazy-semantic",
        ("end", "start"),
        "exact independent phase-one desire and trace replay",
    ),
    "ALD-3": AlignDeltaDefinition(
        "ALD-3",
        "fabricated display side",
        None,
        "lazy-semantic",
        ("start", "end", "anchors", "seed"),
        "exact endpoint provenance and display seed",
    ),
    "ALD-4": AlignDeltaDefinition(
        "ALD-4",
        "W1 phase-2 movement",
        None,
        "lazy-semantic",
        ("start", "end"),
        "every movement leg and final replay value",
    ),
    "ALD-5": AlignDeltaDefinition(
        "ALD-5",
        "EvidenceSpan lyric",
        None,
        "lazy-semantic",
        ("lyric",),
        "exact acoustic-anchor evidence predicate",
    ),
    "ALD-6": AlignDeltaDefinition(
        "ALD-6",
        "durable acquisition core",
        None,
        "mandatory-core",
        ("evidence-core",),
        "exact independent closed EvidenceCore projection",
    ),
}

ALIGN_DELTA_IDS = tuple(_REGISTRY)
ALIGN_DELTA_REGISTRY = MappingProxyType(_REGISTRY)


def canonical_align_delta_registry_bytes() -> bytes:
    projection = [
        {
            "id": definition.delta_id,
            "title": definition.title,
            "ratification": definition.ratification,
        }
        for definition in _REGISTRY.values()
    ]
    return (
        json.dumps(projection, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


ALIGN_DELTA_REGISTRY_SHA256 = hashlib.sha256(
    canonical_align_delta_registry_bytes()
).hexdigest()


__all__ = [
    "ALIGN_DELTA_IDS",
    "ALIGN_DELTA_REGISTRY",
    "ALIGN_DELTA_REGISTRY_SHA256",
    "AlignDeltaDefinition",
    "canonical_align_delta_registry_bytes",
]
