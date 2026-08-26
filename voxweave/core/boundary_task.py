"""The shared, model-free boundary-task contract.

A :class:`BoundaryTask` states one segmentation question in a form that any
solver can answer: immutable atoms, an explicitly enumerated set of legal
boundary indices, an explicit DAG of legal cue edges over those indices, and
optional soft evidence (per-edge quality, per-boundary pause).  It never
carries text a solver may rewrite, and it never carries a model, a device, or
an endpoint -- constructing one imports nothing beyond the standard library and
:mod:`voxweave.lang`.

Both consumers of that contract live outside this module:

* the deterministic boundary optimizer, which searches the DAG for a minimum
  cost path;
* the optional endpoint-backed semantic path
  (:mod:`voxweave.semantic_breaks`), which asks a small language model to pick
  among host-approved paths and re-exports this class as
  ``SemanticBreakRequest``.

The invariants therefore belong here rather than in either consumer.
``__post_init__`` canonicalises every field (sorted, deduplicated, frozen) so
two structurally equal tasks compare equal and can be used as cache or diff
keys, and :func:`_validate_selection` decides whether a proposed set of breaks
is a complete path through the declared DAG, returning a human-readable reason
instead of raising.  Anything language-model specific -- prompts, batching,
scoring, fallback policy -- stays in the semantic module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from voxweave.lang import to_iso_or

__all__ = ["BoundaryTask"]


def _canonical_language(language: str) -> str:
    language = str(language or "").strip()
    if not language:
        raise ValueError("semantic break language is required")
    return (
        to_iso_or(language, None) or language.lower().replace("_", "-").split("-", 1)[0]
    )


def _clean_indices(
    values: Sequence[int], *, name: str, atom_count: int
) -> tuple[int, ...]:
    out: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain integer boundary indices")
        if not 0 < value < atom_count:
            raise ValueError(
                f"{name} index {value} is outside the valid range 1..{atom_count - 1}"
            )
        out.add(value)
    return tuple(sorted(out))


def _char_count(text: str) -> int:
    return sum(not ch.isspace() for ch in text)


@dataclass(frozen=True, slots=True)
class BoundaryTask:
    """One immutable-text boundary selection task.

    Boundary index ``i`` means "break immediately before ``atoms[i]``".  The
    caller passes only word/phrase boundaries its deterministic splitter
    already considers legal, so no solver can invent a cut the host would
    refuse to render.  ``fallback_indices`` is that splitter's own answer: the
    semantic path returns it unchanged whenever the optional model cannot
    produce a fully valid answer, and it doubles as the migration reference a
    solver can measure churn against.  It is validated like any other
    selection, so a task can never be built around an illegal fallback.

    ``target_chars`` is a soft hint.  ``max_segment_chars`` is a hard validator
    over non-whitespace characters and must also be satisfied by the fallback.
    ``pauses_ms`` supplies optional audio evidence as
    ``{boundary_index: preceding_pause_ms}`` without making pauses mandatory.
    When ``allowed_edges`` is supplied it is a directed acyclic graph over
    nodes ``0..len(atoms)``.  A complete selection must follow graph edges from
    0 through every chosen boundary to the terminal node.  The host constructs
    this graph from hard visual-width, maximum-duration, and pause constraints.
    ``edge_quality`` optionally assigns each allowed edge a soft 0..100 score
    derived from its achievable display time, minimum-duration target, and
    CPS/WPS load.
    """

    atoms: tuple[str, ...]
    candidate_indices: tuple[int, ...]
    language: str
    fallback_indices: tuple[int, ...] = ()
    required_indices: tuple[int, ...] = ()
    allowed_edges: tuple[tuple[int, int], ...] = ()
    edge_quality: tuple[tuple[int, int, int], ...] = ()
    pauses_ms: tuple[tuple[int, int], ...] = ()
    min_breaks: int = 0
    max_breaks: int | None = None
    target_chars: int | None = None
    max_segment_chars: int | None = None

    def __post_init__(self) -> None:
        atoms = tuple(self.atoms)
        if not atoms:
            raise ValueError("semantic break request needs at least one atom")
        if any(not isinstance(atom, str) or not atom for atom in atoms):
            raise ValueError("semantic break atoms must be non-empty strings")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "language", _canonical_language(self.language))

        candidates = _clean_indices(
            tuple(self.candidate_indices),
            name="candidate_indices",
            atom_count=len(atoms),
        )
        required = _clean_indices(
            tuple(self.required_indices),
            name="required_indices",
            atom_count=len(atoms),
        )
        fallback = _clean_indices(
            tuple(self.fallback_indices),
            name="fallback_indices",
            atom_count=len(atoms),
        )
        candidate_set = set(candidates)
        if not set(required) <= candidate_set:
            raise ValueError("required_indices must be a subset of candidate_indices")
        if not set(fallback) <= candidate_set:
            raise ValueError("fallback_indices must be a subset of candidate_indices")
        fallback = tuple(sorted(set(fallback) | set(required)))

        raw_edges: Any = self.allowed_edges
        allowed_edges: set[tuple[int, int]] = set()
        for edge in raw_edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError("allowed_edges entries must be (start, end) pairs")
            start, end = edge
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
            ):
                raise TypeError("allowed_edges nodes must be integers")
            if not 0 <= start < end <= len(atoms):
                raise ValueError(
                    "allowed_edges must move forward between nodes 0..atom_count"
                )
            if (start not in {0, len(atoms)} and start not in candidate_set) or (
                end not in {0, len(atoms)} and end not in candidate_set
            ):
                raise ValueError(
                    "allowed_edges internal nodes must be candidate boundary indices"
                )
            allowed_edges.add((start, end))

        raw_quality: Any = self.edge_quality
        edge_quality: dict[tuple[int, int], int] = {}
        for item in raw_quality:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError(
                    "edge_quality entries must be (start, end, score) triples"
                )
            start, end, score = item
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (start, end, score)
            ):
                raise TypeError("edge_quality values must be integers")
            edge = (start, end)
            if edge not in allowed_edges:
                raise ValueError("edge_quality may only score allowed_edges")
            if not 0 <= score <= 100:
                raise ValueError("edge_quality score must be between 0 and 100")
            if edge in edge_quality:
                raise ValueError("edge_quality cannot score an edge twice")
            edge_quality[edge] = score

        if isinstance(self.min_breaks, bool) or not isinstance(self.min_breaks, int):
            raise TypeError("min_breaks must be an integer")
        if self.min_breaks < 0:
            raise ValueError("min_breaks cannot be negative")
        max_breaks = len(candidates) if self.max_breaks is None else self.max_breaks
        if isinstance(max_breaks, bool) or not isinstance(max_breaks, int):
            raise TypeError("max_breaks must be an integer or None")
        if not self.min_breaks <= max_breaks <= len(candidates):
            raise ValueError(
                "max_breaks must be between min_breaks and candidate count"
            )
        if not self.min_breaks <= len(fallback) <= max_breaks:
            raise ValueError("fallback_indices must satisfy min_breaks/max_breaks")

        raw_pauses: Any = self.pauses_ms
        pause_items = (
            raw_pauses.items() if isinstance(raw_pauses, Mapping) else raw_pauses
        )
        pauses: dict[int, int] = {}
        for index, duration in pause_items:
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index not in candidate_set
            ):
                raise ValueError("pauses_ms keys must be candidate boundary indices")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, int)
                or duration < 0
            ):
                raise ValueError(
                    "pauses_ms values must be non-negative integer milliseconds"
                )
            pauses[index] = duration

        for name in ("target_chars", "max_segment_chars"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")

        object.__setattr__(self, "candidate_indices", candidates)
        object.__setattr__(self, "required_indices", required)
        object.__setattr__(self, "fallback_indices", fallback)
        object.__setattr__(self, "allowed_edges", tuple(sorted(allowed_edges)))
        object.__setattr__(
            self,
            "edge_quality",
            tuple(
                (start, end, score)
                for (start, end), score in sorted(edge_quality.items())
            ),
        )
        object.__setattr__(self, "pauses_ms", tuple(sorted(pauses.items())))
        object.__setattr__(self, "max_breaks", max_breaks)

        fallback_error = _validate_selection(self, fallback)
        if fallback_error is not None:
            raise ValueError(f"invalid deterministic fallback: {fallback_error}")


def _segment_char_counts(task: BoundaryTask, breaks: Sequence[int]) -> tuple[int, ...]:
    cuts = (0, *breaks, len(task.atoms))
    return tuple(
        _char_count("".join(task.atoms[start:end]))
        for start, end in zip(cuts, cuts[1:])
    )


def _validate_selection(task: BoundaryTask, indices: Sequence[int]) -> str | None:
    """Return why ``indices`` is not a legal selection for ``task``, else ``None``.

    A legal selection is a complete path: strictly increasing unique candidate
    indices, containing every required boundary, within the break-count bounds,
    within the optional hard character budget, and joined end to end by edges
    the task declared.  Reporting a reason rather than raising is what lets a
    solver treat one infeasible window as a typed local diagnostic.
    """

    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        return "breaks must contain integers only"
    chosen = tuple(indices)
    if chosen != tuple(sorted(set(chosen))):
        return "breaks must be unique and strictly increasing"
    if not set(chosen) <= set(task.candidate_indices):
        return "selected a boundary outside candidate_indices"
    if not set(task.required_indices) <= set(chosen):
        return "omitted a required boundary"
    if not task.min_breaks <= len(chosen) <= int(task.max_breaks or 0):
        return "break count is outside min_breaks/max_breaks"
    if task.max_segment_chars is not None and any(
        count > task.max_segment_chars for count in _segment_char_counts(task, chosen)
    ):
        return "a segment exceeds max_segment_chars"
    if task.allowed_edges:
        allowed = set(task.allowed_edges)
        nodes = (0, *chosen, len(task.atoms))
        if any(edge not in allowed for edge in zip(nodes, nodes[1:])):
            return "selected boundaries do not form a complete allowed path"
    return None
