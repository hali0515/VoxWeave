"""Closed pending-decision defaults for the P6 implementation program.

These are shipped law defaults, not feature flags.  Enabling any deferred
operation requires an approved governing amendment and a source change; no
environment, configuration, CLI, or plugin input can ratify a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class PendingRatification:
    decision: str
    status: Literal["pending"]
    default: str
    approval_unlocks: str
    enabled_operation: None = None


_DEFAULTS = {
    "RAT-1": PendingRatification(
        "RAT-1",
        "pending",
        "acquisition-evidence-only",
        "fresh-alignment W1 authority and factory",
    ),
    "RAT-2": PendingRatification(
        "RAT-2",
        "pending",
        "persistence-scaffold-only",
        "durable transactional align-anchor evidence",
    ),
    "RAT-3": PendingRatification(
        "RAT-3",
        "pending",
        "current-selected-writer",
        "raw last-occurrence speaker_turns carriage",
    ),
    "RAT-4": PendingRatification(
        "RAT-4",
        "pending",
        "current-full-pass-order-and-budget-behavior",
        "lexical full-pass order and unsafe-hint refusal",
    ),
    "RAT-5": PendingRatification(
        "RAT-5",
        "pending",
        "qwen-selected-legacy-nominal-origin",
        "selected Qwen v2 physical-origin delta",
    ),
    "RAT-6": PendingRatification(
        "RAT-6",
        "pending",
        "current-endpoint-semantic-mode",
        "one approved semantic authority model for P7-C",
    ),
    "RAT-7": PendingRatification(
        "RAT-7",
        "pending",
        "j0-only",
        "split (J0,S0) commit generation recheck",
    ),
}

RATIFICATION_DEFAULTS = MappingProxyType(_DEFAULTS)

FRESH_ALIGNMENT_W1_ENABLED = False
DURABLE_ALIGN_EVIDENCE_ENABLED = False
QWEN_SELECTED_V2_ENABLED = False
RAW_SPEAKER_TURNS_WRITER_ENABLED = False
LEXICAL_FULL_PASS_DELTA_ENABLED = False
SPEAKER_MAPPING_CAS_ENABLED = False


__all__ = [
    "DURABLE_ALIGN_EVIDENCE_ENABLED",
    "FRESH_ALIGNMENT_W1_ENABLED",
    "LEXICAL_FULL_PASS_DELTA_ENABLED",
    "PendingRatification",
    "QWEN_SELECTED_V2_ENABLED",
    "RATIFICATION_DEFAULTS",
    "RAW_SPEAKER_TURNS_WRITER_ENABLED",
    "SPEAKER_MAPPING_CAS_ENABLED",
]
