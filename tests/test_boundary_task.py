"""The boundary-task contract is importable from the core and shared by name.

Deep coverage of every ``__post_init__`` invariant lives in
``tests/test_semantic_breaks.py``; this file only pins the lift itself, so a
future move of the semantic module cannot silently fork the class.
"""

from __future__ import annotations

import pytest

from voxweave import semantic_breaks
from voxweave.core.boundary_task import BoundaryTask, _validate_selection


def _task() -> BoundaryTask:
    return BoundaryTask(
        atoms=("welcome", "to", "the", "show"),
        candidate_indices=(1, 2, 3),
        language="en",
        fallback_indices=(2,),
        allowed_edges=((0, 2), (2, 4), (0, 1), (1, 4)),
    )


def test_semantic_aliases_are_the_core_class() -> None:
    assert semantic_breaks.BoundaryTask is BoundaryTask
    assert semantic_breaks.SemanticBreakRequest is BoundaryTask
    assert isinstance(_task(), semantic_breaks.SemanticBreakRequest)


def test_equal_tasks_compare_equal_across_the_two_names() -> None:
    assert semantic_breaks.SemanticBreakRequest(
        atoms=("a", "b"), candidate_indices=(1,), language="english"
    ) == BoundaryTask(atoms=("a", "b"), candidate_indices=(1,), language="en")


def test_validator_round_trip() -> None:
    task = _task()
    assert _validate_selection(task, (2,)) is None
    assert _validate_selection(task, (1, 2)) == (
        "selected boundaries do not form a complete allowed path"
    )
    with pytest.raises(ValueError, match="fallback_indices must be a subset"):
        BoundaryTask(
            atoms=("a", "b", "c"),
            candidate_indices=(1,),
            language="en",
            fallback_indices=(2,),
        )
