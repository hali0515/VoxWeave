"""Voiceprint speaker clustering over pyannote's speaker turns.

pyannote finds *who speaks when* (turn boundaries, overlap); this module decides
*who is who*: it regroups the raw turns by speaker-embedding similarity instead
of trusting pyannote's own clustering, and may abstain on turns nobody can be
attributed to confidently (background voices, noise, fragments too short to
tell). ``voxweave.diarize`` calls :func:`cluster_turns` on the raw turns before
its smoothing pass when ``[diarize].clustering = "voiceprint"``.

PLACEHOLDER: the body of :func:`cluster_turns` below does not cluster yet. It
only normalizes the turns exactly as the real recipe's output contract requires
(first-appearance ``SPEAKER_NN`` labels, same-label overlaps unioned, sorted),
so the product plumbing around it can be built and tested. The real recipe is
ported here later, replacing this module's body wholesale; everything the rest
of the package relies on is the public contract below:

- :data:`RECIPE`, the recipe id recorded in diarization provenance (the audit
  dict carries it too);
- :class:`ClusteringParams`, frozen, every field defaulted, JSON-serializable
  through :func:`dataclasses.asdict`;
- :class:`ClusteringResult` and :func:`cluster_turns` with the signature below.

The module is pure: numpy (and optionally scipy) only, no torch import, no
randomness, no I/O. Embeddings come from the injected ``embed`` callable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

Turn = tuple[float, float, str]
# spans in seconds -> [N, D] L2-normalized rows, one per span, in span order.
Embed = Callable[[Sequence[tuple[float, float]]], np.ndarray]

RECIPE = "placeholder"
LABEL_FORMAT = "SPEAKER_{:02d}"


@dataclass(frozen=True)
class ClusteringParams:
    """Tuning parameters of the recipe; every field has a default.

    The placeholder has none. Values must be global (never per dataset or per
    language) and JSON-serializable, since they are recorded in provenance.
    """


@dataclass
class ClusteringResult:
    """Clustered turns plus a JSON-serializable audit of how they were formed.

    ``turns``: relabelled ``SPEAKER_NN`` by first appearance, abstained turns
    removed, same-label overlaps unioned, sorted by ``(start, end, label)``.
    ``audit``: counts, thresholds, per-cluster anchor seconds and abstained
    seconds; always carries ``"recipe"``.
    """

    turns: list[Turn]
    audit: dict[str, object] = field(default_factory=dict)


def _relabelled(turns: Sequence[Turn]) -> list[Turn]:
    """Sort turns and rename labels ``SPEAKER_NN`` in order of first appearance."""
    ordered = sorted(
        ((float(start), float(end), str(label)) for start, end, label in turns),
        key=lambda turn: (turn[0], turn[1], turn[2]),
    )
    names: dict[str, str] = {}
    for _start, _end, label in ordered:
        if label not in names:
            names[label] = LABEL_FORMAT.format(len(names))
    return [(start, end, names[label]) for start, end, label in ordered]


def _union_same_label(turns: Sequence[Turn]) -> list[Turn]:
    """Union overlapping turns of the same label; touching turns stay separate."""
    by_label: dict[str, list[tuple[float, float]]] = {}
    for start, end, label in turns:
        by_label.setdefault(label, []).append((start, end))
    merged: list[Turn] = []
    for label, spans in by_label.items():
        current: list[float] | None = None
        for start, end in sorted(spans):
            if current is not None and start < current[1]:
                current[1] = max(current[1], end)
                continue
            if current is not None:
                merged.append((current[0], current[1], label))
            current = [start, end]
        if current is not None:
            merged.append((current[0], current[1], label))
    return sorted(merged, key=lambda turn: (turn[0], turn[1], turn[2]))


def cluster_turns(
    turns: Sequence[Turn],
    embed: Embed,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    params: ClusteringParams | None = None,
) -> ClusteringResult:
    """Regroup raw pyannote ``turns`` by voiceprint similarity.

    ``embed`` maps ``(start, end)`` spans of the audio pyannote saw to unit
    rows; ``min_speakers``/``max_speakers`` bound the number of clusters like
    pyannote's own options. Deterministic for a given input.

    PLACEHOLDER: keeps pyannote's grouping (``embed`` is never called and no
    turn abstains); see the module docstring.
    """
    del embed, min_speakers, max_speakers  # used by the real recipe
    params = params or ClusteringParams()
    clustered = _union_same_label(_relabelled(turns))
    speech = sum(end - start for start, end, _label in clustered)
    return ClusteringResult(
        turns=clustered,
        audit={
            "recipe": RECIPE,
            "turns_in": len(turns),
            "turns_out": len(clustered),
            "speakers": len({label for _start, _end, label in clustered}),
            "abstained_turns": 0,
            "abstained_seconds": 0.0,
            "assigned_seconds": round(speech, 3),
        },
    )


__all__ = [
    "ClusteringParams",
    "ClusteringResult",
    "Embed",
    "LABEL_FORMAT",
    "RECIPE",
    "Turn",
    "cluster_turns",
]
