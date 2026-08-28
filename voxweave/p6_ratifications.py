"""Closed ratification record for the P6 implementation program.

These values are source-law constants, not runtime feature flags.  The law
owner approved RAT-1 through RAT-7 on 2026-08-28; RAT-6 selected option B and
is discharged by the later P7 cutover window.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class RatificationDecision:
    decision: str
    status: Literal["approved"]
    default: str
    approval_unlocks: str
    enabled_operation: str


_DECISIONS = {
    "RAT-1": RatificationDecision(
        "RAT-1",
        "approved",
        "acquisition-evidence-only",
        "fresh-alignment W1 authority and factory",
        "fresh-alignment",
    ),
    "RAT-2": RatificationDecision(
        "RAT-2",
        "approved",
        "persistence-scaffold-only",
        "durable transactional align-anchor evidence",
        "durable-align-evidence",
    ),
    "RAT-3": RatificationDecision(
        "RAT-3",
        "approved",
        "current-selected-writer",
        "raw last-occurrence speaker_turns carriage",
        "raw-speaker-turns-writer",
    ),
    "RAT-4": RatificationDecision(
        "RAT-4",
        "approved",
        "current-full-pass-order-and-budget-behavior",
        "lexical full-pass order and unsafe-hint refusal",
        "lexical-full-pass",
    ),
    "RAT-5": RatificationDecision(
        "RAT-5",
        "approved",
        "qwen-selected-legacy-nominal-origin",
        "selected Qwen v2 physical-origin delta",
        "qwen-physical-origin",
    ),
    "RAT-6": RatificationDecision(
        "RAT-6",
        "approved",
        "current-endpoint-semantic-mode",
        "one approved semantic authority model for P7-C",
        "p7-remove-semantic-mode",
    ),
    "RAT-7": RatificationDecision(
        "RAT-7",
        "approved",
        "j0-only",
        "split (J0,S0) commit generation recheck",
        "split-j0-s0-cas",
    ),
}

RATIFICATION_DEFAULTS = MappingProxyType(_DECISIONS)

FRESH_ALIGNMENT_W1_ENABLED = True
DURABLE_ALIGN_EVIDENCE_ENABLED = True
QWEN_SELECTED_V2_ENABLED = True
RAW_SPEAKER_TURNS_WRITER_ENABLED = True
LEXICAL_FULL_PASS_DELTA_ENABLED = True
SPEAKER_MAPPING_CAS_ENABLED = True


__all__ = [
    "DURABLE_ALIGN_EVIDENCE_ENABLED",
    "FRESH_ALIGNMENT_W1_ENABLED",
    "LEXICAL_FULL_PASS_DELTA_ENABLED",
    "RatificationDecision",
    "QWEN_SELECTED_V2_ENABLED",
    "RATIFICATION_DEFAULTS",
    "RAW_SPEAKER_TURNS_WRITER_ENABLED",
    "SPEAKER_MAPPING_CAS_ENABLED",
]
