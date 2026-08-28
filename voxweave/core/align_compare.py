"""Lazy P6 semantic-comparison boundary.

RAT-1 is pending, so no semantic W1 candidate can exist honestly.  The module
still supplies the closed fail-closed boundary that later governing code may
amend; it deliberately defines no EvidenceCore projector and is never imported
by the ordinary legacy/off path.
"""

from __future__ import annotations

from typing import Never

from voxweave.p6_ratifications import FRESH_ALIGNMENT_W1_ENABLED


class SemanticComparisonUnavailable(RuntimeError):
    def __init__(self, decision: str):
        super().__init__(f"semantic align comparison is blocked by pending {decision}")
        self.decision = decision


def semantic_comparison_available() -> bool:
    return FRESH_ALIGNMENT_W1_ENABLED


def compare_semantic_deltas(*_args: object, **_kwargs: object) -> Never:
    """Fail closed until a genuine fresh W1 result can be produced."""
    raise SemanticComparisonUnavailable("RAT-1")


__all__ = [
    "SemanticComparisonUnavailable",
    "compare_semantic_deltas",
    "semantic_comparison_available",
]
