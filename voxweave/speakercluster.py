"""Voiceprint speaker clustering over pyannote turns (recipe ``voiceprint-v1``).

pyannote stays the segmenter (who speaks when, overlap); this module decides
*who is who* with an external speaker-embedding model and abstains on turns
nobody can attribute confidently (background voices, noise, very short
fragments). It is a pure algorithm: the caller injects ``embed`` (spans in
seconds -> ``[N, D]`` L2-normalised rows), so everything here runs without
torch or a GPU and is deterministic. numpy is the only import-time
dependency; the non-default ``spectral`` and ``refine`` methods import scipy
when they run. ``voxweave.diarize`` calls :func:`cluster_turns` on the raw
pyannote turns, before its smoothing pass, when
``[diarize].clustering = "voiceprint"``.

Pipeline (``cluster_turns``):

1. **Pieces.** Every turn minus the time any turn with another label is
   active (overlap-trimmed). Pieces shorter than ``min_piece_seconds`` are
   ignored.
2. **Anchors.** Turns whose usable clean speech is at least
   ``anchor_seconds``. A turn's embedding is the duration-weighted unit mean of
   its pieces' embeddings; a turn without a usable piece is embedded over its
   whole span (never an anchor).
3. **Anchor clustering**, one of three methods:

   * ``ahc``: average-linkage cosine AHC cut at ``cut_distance``, with a
     cannot-link constraint (two anchors overlapping in time by more than
     ``cannot_link_overlap`` of the shorter one never end up in one cluster);
   * ``spectral``: 3D-Speaker style spectral clustering (p-pruned cosine
     affinity, unnormalised Laplacian, eigengap speaker count, k-means);
   * ``refine``: start from pyannote's labels, split labels whose anchors
     form two well separated groups, merge labels whose anchors are close.

   Clusters holding fewer than ``dissolve_seconds`` of anchor speech are
   dissolved: their anchors are assigned like any other turn. ``min_speakers``
   / ``max_speakers`` are hard bounds on the number of surviving clusters:
   ``ahc`` takes the merge level nearest its cut that satisfies them, and when
   no clustering does (cannot-link keeping more than ``max_speakers`` anchor
   groups apart, too little anchor speech for ``min_speakers`` clusters),
   :class:`ClusteringError` is raised rather than returning a count outside
   the bounds. Every method is held to the bounds the same way.
4. **Centroids**: duration-weighted unit mean of each cluster's anchors.
5. **Assignment** of every other turn: the best *local* cluster (an anchor
   within ``local_seconds`` or ``local_turns`` turns) when its cosine is at
   least ``tau`` and beats the runner-up by ``margin``; otherwise the same
   test over all clusters with the stricter ``tau_global`` / ``margin_global``;
   otherwise pyannote's own label mapped to the cluster holding most of that
   label's anchor seconds, but only when the cosine to it is at least
   ``tau_floor`` (a non-speech guard); otherwise the turn is abstained.
   With ``overlap_exclusion`` a turn never joins the cluster of a turn with
   another pyannote label that overlaps it by more than ``cannot_link_overlap``
   of the shorter one: pyannote detected two voices there, and merging them
   would erase the overlapped speech (measured on dev: without it the missed
   speech grows by 2-4 pp and eats the confusion gain).
   Neighbouring labels are never used as evidence (a backchannel between two
   turns of speaker A usually comes from the listener, not from A).
6. Labels ``SPEAKER_NN`` by first appearance, abstained turns removed,
   overlapping or touching same-label turns unioned.

When no anchor cluster survives (nothing long or clean enough to anchor a
voiceprint on), pyannote's labels are passed through unchanged and
``audit[PASSTHROUGH]`` says why: callers must treat such a result as "not
clustered", not as a voiceprint answer.

The ``ClusteringParams`` defaults are the ``voiceprint-v1`` values, selected
on the dev splits of four public diarization sets (AMI, AliMeeting,
VoxConverse, JVS-conv) in the ReDimNet2-B6 embedding space. Changing any
default changes the recipe: give :data:`RECIPE` a new id with it.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

Turn = tuple[float, float, str]
Span = tuple[float, float]
# spans in seconds -> [N, D] L2-normalised rows, one per span, in span order.
EmbedFn = Callable[[Sequence[Span]], np.ndarray]
Embed = EmbedFn

RECIPE = "voiceprint-v1"
METHODS = ("ahc", "spectral", "refine")
LABEL_FORMAT = "SPEAKER_{:02d}"
# Audit key present (with the reason) when pyannote's labels were passed through.
PASSTHROUGH = "passthrough"

# Assignment routes recorded per turn.
ROUTE_ANCHOR = "anchor"
ROUTE_LOCAL = "local"
ROUTE_GLOBAL = "global"
ROUTE_FALLBACK = "fallback"
ROUTE_ABSTAIN = "abstain"
ROUTES = (ROUTE_ANCHOR, ROUTE_LOCAL, ROUTE_GLOBAL, ROUTE_FALLBACK, ROUTE_ABSTAIN)


class ClusteringError(ValueError):
    """Malformed input (bad parameters or embeddings) or unsatisfiable speaker bounds."""


@dataclass(frozen=True)
class ClusteringParams:
    """Every knob of ``voiceprint-v1``; cosine values are similarities in [-1, 1].

    ``cut_distance`` is a cosine *distance* (1 - similarity) as in pyannote's
    AgglomerativeClustering; all other thresholds are cosine similarities.
    """

    method: str = "ahc"
    anchor_seconds: float = 1.0
    min_piece_seconds: float = 0.3
    min_embed_seconds: float = 0.05
    # ahc
    cut_distance: float = 0.45
    cannot_link_overlap: float = 0.5
    # spectral
    spectral_pval: float = 0.012
    spectral_merge_similarity: float = 0.0
    spectral_max_speakers: int = 20
    # refine
    merge_similarity: float = 0.55
    split_similarity: float = 0.3
    # all methods
    dissolve_seconds: float = 6.0
    # assignment
    local_seconds: float = 30.0
    local_turns: int = 15
    tau: float = 0.35
    margin: float = 0.1
    tau_global: float = 0.5
    margin_global: float = 0.1
    tau_floor: float = 0.1
    # A turn overlapped (> cannot_link_overlap of the shorter) by a turn with
    # another pyannote label never joins that turn's cluster.
    overlap_exclusion: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ClusteringError(
                f"method must be one of {METHODS}, got {self.method!r}"
            )
        nonnegative = (
            "anchor_seconds",
            "min_piece_seconds",
            "min_embed_seconds",
            "dissolve_seconds",
            "local_seconds",
            "margin",
            "margin_global",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ClusteringError(
                    f"{name} must be a finite number >= 0, got {value!r}"
                )
        for name in (
            "tau",
            "tau_global",
            "tau_floor",
            "merge_similarity",
            "split_similarity",
            "spectral_merge_similarity",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ClusteringError(f"{name} must lie in [-1, 1], got {value!r}")
        if not 0.0 <= self.cut_distance <= 2.0:
            raise ClusteringError(
                f"cut_distance must lie in [0, 2], got {self.cut_distance!r}"
            )
        if not 0.0 < self.cannot_link_overlap <= 1.0:
            raise ClusteringError("cannot_link_overlap must lie in (0, 1]")
        if not 0.0 < self.spectral_pval <= 1.0:
            raise ClusteringError("spectral_pval must lie in (0, 1]")
        if self.local_turns < 0 or self.spectral_max_speakers < 1:
            raise ClusteringError(
                "local_turns must be >= 0 and spectral_max_speakers >= 1"
            )


@dataclass
class ClusteringResult:
    """Relabelled turns and what the clustering did.

    ``turns``: ``SPEAKER_NN`` by first appearance, abstained turns removed,
    same-label overlaps unioned, sorted. ``audit``: JSON-serialisable counts,
    thresholds, per-cluster anchor seconds and abstained seconds; it always
    carries ``"recipe"``, and carries ``PASSTHROUGH`` (the reason) when the
    turns are pyannote's labels passed through rather than a clustering.
    """

    turns: list[Turn]
    audit: dict[str, object]


@dataclass
class TurnAssignment:
    """Per-turn outcome, aligned with ``turns`` (the valid input turns, sorted).

    ``labels[i]`` is the final ``SPEAKER_NN`` of ``turns[i]`` or ``None`` when
    it was abstained; ``routes[i]`` says how it was decided (see ``ROUTES``).
    """

    turns: list[Turn]
    labels: list[str | None]
    routes: list[str]
    audit: dict[str, object]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def cluster_turns(
    turns: Sequence[Turn],
    embed: EmbedFn,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    params: ClusteringParams | None = None,
) -> ClusteringResult:
    """Re-cluster pyannote ``turns`` with voiceprints; see the module docstring.

    ``embed`` receives every span to embed in one call and must return one
    L2-normalised row per span. Abstained turns are dropped from the result.
    Raises :class:`ClusteringError` when the speaker bounds cannot be met.
    """
    assignment = assign_turns(
        turns,
        embed,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        params=params,
    )
    kept = [
        (start, end, label)
        for (start, end, _orig), label in zip(assignment.turns, assignment.labels)
        if label is not None
    ]
    return ClusteringResult(turns=union_same_label(kept), audit=assignment.audit)


def assign_turns(
    turns: Sequence[Turn],
    embed: EmbedFn,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    params: ClusteringParams | None = None,
) -> TurnAssignment:
    """The per-turn decisions behind ``cluster_turns`` (labels before dropping)."""
    params = params or ClusteringParams()
    lo, hi = _speaker_bounds(min_speakers, max_speakers)
    ordered = _valid_sorted(turns)
    n = len(ordered)
    pieces = overlap_trimmed_pieces(ordered, params.min_piece_seconds)
    vectors, has_vec, n_spans = _turn_vectors(ordered, pieces, embed, params)
    usable = np.array([sum(e - s for s, e in p) for p in pieces], dtype=np.float64)
    anchor_idx = [
        i for i in range(n) if has_vec[i] and usable[i] >= params.anchor_seconds
    ]
    anchor_idx_arr = np.array(anchor_idx, dtype=np.int64)
    anchor_vecs = (
        vectors[anchor_idx_arr] if anchor_idx else np.zeros((0, vectors.shape[1]))
    )
    anchor_secs = usable[anchor_idx_arr] if anchor_idx else np.zeros(0)
    cannot = _cannot_link(ordered, anchor_idx, params.cannot_link_overlap)

    if params.method == "ahc":
        raw = _ahc_clusters(anchor_vecs, anchor_secs, cannot, params, lo, hi)
    elif params.method == "spectral":
        raw = _spectral_clusters(anchor_vecs, anchor_secs, params, lo, hi)
    else:
        original = [ordered[i][2] for i in anchor_idx]
        raw = _refine_clusters(
            anchor_vecs, anchor_secs, original, cannot, params, lo, hi
        )
    n_before = int(len(set(raw.tolist()))) if raw.size else 0
    clusters = _dissolve(raw, anchor_secs, params.dissolve_seconds)
    survivors = sorted({int(c) for c in clusters.tolist() if c >= 0})

    labels_idx: list[int | None] = [None] * n
    routes = [ROUTE_ABSTAIN] * n
    if not survivors:
        # Nothing to anchor a voiceprint on: keep pyannote's labels untouched.
        return _passthrough(ordered, params, n_spans, len(anchor_idx), lo, hi)
    if len(survivors) < lo or (hi is not None and len(survivors) > hi):
        # Every surviving cluster keeps its anchors, so this is the output count.
        raise ClusteringError(
            f"{params.method} clustering left {len(survivors)} speaker(s), outside "
            f"the speaker bounds {_bounds_text(lo, hi)}"
        )

    remap = {c: k for k, c in enumerate(survivors)}
    cluster_of_anchor = np.array(
        [remap.get(int(c), -1) for c in clusters], dtype=np.int64
    )
    k_clusters = len(survivors)
    centroids = np.zeros((k_clusters, vectors.shape[1]))
    anchor_seconds = np.zeros(k_clusters)
    for pos, turn_index in enumerate(anchor_idx):
        c = int(cluster_of_anchor[pos])
        if c < 0:
            continue
        centroids[c] += anchor_secs[pos] * vectors[turn_index]
        anchor_seconds[c] += anchor_secs[pos]
        labels_idx[turn_index] = c
        routes[turn_index] = ROUTE_ANCHOR
    centroids = _unit_rows(centroids)

    # Fallback map: pyannote label -> cluster holding most of its anchor seconds.
    label_votes: dict[str, np.ndarray] = {}
    for pos, turn_index in enumerate(anchor_idx):
        c = int(cluster_of_anchor[pos])
        if c >= 0:
            votes = label_votes.setdefault(ordered[turn_index][2], np.zeros(k_clusters))
            votes[c] += anchor_secs[pos]
    label_home = {lab: int(np.argmax(v)) for lab, v in label_votes.items()}

    kept_anchor_pos = np.flatnonzero(cluster_of_anchor >= 0)
    kept_turn_idx = anchor_idx_arr[kept_anchor_pos]
    kept_cluster = cluster_of_anchor[kept_anchor_pos]
    starts = np.array([t[0] for t in ordered])
    ends = np.array([t[1] for t in ordered])
    pool = [i for i in range(n) if routes[i] != ROUTE_ANCHOR]
    conflicts = (
        _overlap_conflicts(ordered, params.cannot_link_overlap)
        if params.overlap_exclusion
        else [[] for _ in range(n)]
    )
    for i in pool:
        if not has_vec[i]:
            continue
        sims = centroids @ vectors[i]
        # pyannote says another speaker talks over this turn: that speaker's
        # cluster cannot be this turn's cluster (it would erase the overlap).
        forbidden = {labels_idx[j] for j in conflicts[i] if labels_idx[j] is not None}
        if forbidden:
            sims = sims.copy()
            sims[list(forbidden)] = -np.inf
        gap = np.maximum(
            0.0,
            np.maximum(starts[kept_turn_idx], starts[i])
            - np.minimum(ends[kept_turn_idx], ends[i]),
        )
        near = (gap <= params.local_seconds) | (
            np.abs(kept_turn_idx - i) <= params.local_turns
        )
        local = np.unique(kept_cluster[near])
        choice = _accept(sims, local, params.tau, params.margin)
        route = ROUTE_LOCAL
        if choice is None:
            choice = _accept(
                sims, np.arange(k_clusters), params.tau_global, params.margin_global
            )
            route = ROUTE_GLOBAL
        if choice is None:
            home = label_home.get(ordered[i][2])
            if home is not None and sims[home] >= params.tau_floor:
                choice, route = home, ROUTE_FALLBACK
        if choice is not None:
            labels_idx[i] = int(choice)
            routes[i] = route

    names = _first_appearance_names(ordered, labels_idx)
    labels = [names[c] if c is not None else None for c in labels_idx]
    audit = _audit(
        ordered,
        labels,
        routes,
        params,
        lo,
        hi,
        n_spans=n_spans,
        n_anchors=len(anchor_idx),
        n_before=n_before,
        cannot_pairs=len(cannot),
        dissolved_anchors=int((cluster_of_anchor < 0).sum()),
        anchor_seconds={
            names[c]: float(anchor_seconds[c]) for c in range(k_clusters) if c in names
        },
        anchor_counts={
            names[c]: int((cluster_of_anchor == c).sum())
            for c in range(k_clusters)
            if c in names
        },
    )
    return TurnAssignment(ordered, labels, routes, audit)


# --------------------------------------------------------------------------
# Turn geometry
# --------------------------------------------------------------------------


def _valid_sorted(turns: Sequence[Turn]) -> list[Turn]:
    out: list[Turn] = []
    for start, end, label in turns:
        s, e = float(start), float(end)
        if not (math.isfinite(s) and math.isfinite(e)):
            raise ClusteringError("turn boundaries must be finite")
        if e > s:
            out.append((s, e, str(label)))
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def _union(spans: Sequence[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _subtract(
    start: float, end: float, blockers: Sequence[Span], starts: Sequence[float]
) -> list[Span]:
    """``[start, end)`` minus sorted disjoint ``blockers`` (``starts`` = their starts)."""
    index = max(0, bisect.bisect_right(starts, start) - 1)
    pieces: list[Span] = []
    cursor = start
    while index < len(blockers):
        b_start, b_end = blockers[index]
        if b_start >= end:
            break
        if b_end > cursor:
            if b_start > cursor:
                pieces.append((cursor, min(b_start, end)))
            cursor = max(cursor, b_end)
            if cursor >= end:
                break
        index += 1
    if cursor < end:
        pieces.append((cursor, end))
    return [(s, e) for s, e in pieces if e > s]


def overlap_trimmed_pieces(
    turns: Sequence[Turn], min_piece_seconds: float
) -> list[list[Span]]:
    """Per turn: its span minus every other label's turns, pieces >= ``min_piece_seconds``."""
    labels = sorted({label for _s, _e, label in turns})
    others: dict[str, tuple[list[Span], list[float]]] = {}
    for label in labels:
        blockers = _union([(s, e) for s, e, lab in turns if lab != label])
        others[label] = (blockers, [b[0] for b in blockers])
    out: list[list[Span]] = []
    for start, end, label in turns:
        blockers, starts = others[label]
        pieces = _subtract(start, end, blockers, starts)
        out.append([(s, e) for s, e in pieces if e - s >= min_piece_seconds])
    return out


def union_same_label(turns: Sequence[Turn]) -> list[Turn]:
    """Join overlapping or touching turns of one label; sorted by (start, end, label)."""
    by_label: dict[str, list[Span]] = {}
    for start, end, label in turns:
        if end > start:
            by_label.setdefault(label, []).append((float(start), float(end)))
    out: list[Turn] = []
    for label, spans in by_label.items():
        out.extend((s, e, label) for s, e in _union(spans))
    return sorted(out, key=lambda t: (t[0], t[1], t[2]))


def _cannot_link(
    turns: Sequence[Turn], anchor_idx: Sequence[int], fraction: float
) -> set[tuple[int, int]]:
    """Anchor position pairs (a < b) whose turns overlap by > ``fraction`` of the shorter."""
    order = sorted(range(len(anchor_idx)), key=lambda p: turns[anchor_idx[p]][0])
    pairs: set[tuple[int, int]] = set()
    for rank, p in enumerate(order):
        s1, e1, _l1 = turns[anchor_idx[p]]
        for q in order[rank + 1 :]:
            s2, e2, _l2 = turns[anchor_idx[q]]
            if s2 >= e1:
                break
            overlap = min(e1, e2) - max(s1, s2)
            if overlap > fraction * min(e1 - s1, e2 - s2):
                pairs.add((min(p, q), max(p, q)))
    return pairs


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0)


def _embedding_plan(
    turns: Sequence[Turn], pieces: Sequence[Sequence[Span]], params: ClusteringParams
) -> list[list[Span]]:
    """Spans whose embeddings make up each turn's vector (whole span if no piece)."""
    plan: list[list[Span]] = []
    for (start, end, _label), own in zip(turns, pieces):
        spans = list(own)
        if not spans and end - start >= params.min_embed_seconds:
            spans = [(start, end)]
        plan.append(spans)
    return plan


def embedding_spans(
    turns: Sequence[Turn], params: ClusteringParams | None = None
) -> list[Span]:
    """Exactly the spans ``cluster_turns`` passes to ``embed`` (first-seen order)."""
    params = params or ClusteringParams()
    ordered = _valid_sorted(turns)
    plan = _embedding_plan(
        ordered, overlap_trimmed_pieces(ordered, params.min_piece_seconds), params
    )
    return list(dict.fromkeys(span for spans in plan for span in spans))


def _turn_vectors(
    turns: Sequence[Turn],
    pieces: Sequence[Sequence[Span]],
    embed: EmbedFn,
    params: ClusteringParams,
) -> tuple[np.ndarray, np.ndarray, int]:
    """(unit vector per turn [n, D], has-vector mask, number of embedded spans)."""
    plan = _embedding_plan(turns, pieces, params)
    wanted: dict[Span, int] = {}
    for spans in plan:
        for span in spans:
            wanted.setdefault(span, len(wanted))
    spans_list = list(wanted)
    if spans_list:
        rows = np.asarray(embed(spans_list), dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] != len(spans_list) or rows.shape[1] == 0:
            raise ClusteringError(
                f"embed returned shape {rows.shape}, expected ({len(spans_list)}, D)"
            )
        if not np.isfinite(rows).all():
            raise ClusteringError("embed returned non-finite values")
        rows = _unit_rows(rows)
        dim = rows.shape[1]
    else:
        rows = np.zeros((0, 1))
        dim = 1
    vectors = np.zeros((len(turns), dim))
    has_vec = np.zeros(len(turns), dtype=bool)
    for i, spans in enumerate(plan):
        if not spans:
            continue
        weights = np.array([e - s for s, e in spans])
        acc = (rows[[wanted[s] for s in spans]] * weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(acc))
        if norm > 0.0:
            vectors[i] = acc / norm
            has_vec[i] = True
    return vectors, has_vec, len(spans_list)


# --------------------------------------------------------------------------
# Anchor clustering
# --------------------------------------------------------------------------


def _speaker_bounds(
    min_speakers: int | None, max_speakers: int | None
) -> tuple[int, int | None]:
    lo = max(1, int(min_speakers)) if min_speakers is not None else 1
    hi = int(max_speakers) if max_speakers is not None else None
    if hi is not None and hi < lo:
        raise ClusteringError(f"max_speakers ({hi}) < min_speakers ({lo})")
    return lo, hi


def _bounds_text(lo: int, hi: int | None) -> str:
    return f"[{lo}, {hi}]" if hi is not None else f"[{lo}, unbounded]"


def _dissolve(clusters: np.ndarray, seconds: np.ndarray, minimum: float) -> np.ndarray:
    """Clusters with < ``minimum`` anchor seconds -> -1 (their anchors join the pool)."""
    out = clusters.copy()
    for c in np.unique(clusters):
        members = clusters == c
        if float(seconds[members].sum()) < minimum:
            out[members] = -1
    return out


def _surviving(clusters: np.ndarray, seconds: np.ndarray, minimum: float) -> int:
    return int(
        len({int(c) for c in _dissolve(clusters, seconds, minimum).tolist() if c >= 0})
    )


def _relabel(labels: Sequence[int]) -> np.ndarray:
    mapping: dict[int, int] = {}
    return np.array(
        [mapping.setdefault(int(x), len(mapping)) for x in labels], dtype=np.int64
    )


def average_linkage_history(
    x: np.ndarray, cannot: set[tuple[int, int]], stop_distance: float | None = None
) -> list[tuple[int, int, float]]:
    """Cosine average-linkage merges ``(keep, absorbed, distance)`` in order.

    Pairs of clusters containing a cannot-link pair are never merged. Stops at
    the first merge whose distance exceeds ``stop_distance`` (``None`` runs
    until no allowed merge remains). Ties break towards the lowest indices.
    """
    n = x.shape[0]
    if n < 2:
        return []
    sums = x @ x.T
    size = np.ones(n)
    blocked = np.zeros((n, n), dtype=bool)
    for a, b in cannot:
        blocked[a, b] = blocked[b, a] = True
    np.fill_diagonal(blocked, True)
    active = np.ones(n, dtype=bool)
    avg = np.where(blocked, -np.inf, sums)
    history: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        flat = int(np.argmax(avg))
        a, b = divmod(flat, n)
        best = avg[a, b]
        if not np.isfinite(best):
            break
        distance = 1.0 - float(best)
        if stop_distance is not None and distance > stop_distance:
            break
        keep, gone = (a, b) if a < b else (b, a)
        history.append((keep, gone, distance))
        sums[keep] += sums[gone]
        sums[:, keep] = sums[keep]
        size[keep] += size[gone]
        blocked[keep] |= blocked[gone]
        blocked[:, keep] = blocked[keep]
        active[gone] = False
        row = sums[keep] / (size[keep] * size)
        row[blocked[keep] | ~active] = -np.inf
        avg[keep] = row
        avg[:, keep] = row
        avg[gone] = -np.inf
        avg[:, gone] = -np.inf
    return history


def _replay(n: int, history: Sequence[tuple[int, int, float]], k: int) -> np.ndarray:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keep, gone, _d in history[:k]:
        parent[find(gone)] = find(keep)
    return _relabel([find(i) for i in range(n)])


def _fit_count(
    candidates: Sequence[int],
    labels_for: Callable[[int], np.ndarray],
    seconds: np.ndarray,
    minimum: float,
    lo: int,
    hi: int | None,
    start: int,
) -> np.ndarray:
    """The candidate nearest ``start`` whose surviving cluster count fits [lo, hi].

    Walks from ``start`` towards the bounds: more merges when there are too
    many clusters, fewer when there are too few. Every candidate on the way is
    tested, since dissolving makes the count non-monotonic in the number of
    merges. The walk never goes the other way: fewer merges can only lower
    the count by dissolving whole speakers, which is no answer to
    ``max_speakers``. Raises :class:`ClusteringError` when nothing on the walk
    fits (cannot-link blocking the merges ``max_speakers`` needs, too little
    anchor speech for ``min_speakers`` clusters), instead of returning a count
    outside the bounds.
    """

    def fits(count: int) -> bool:
        return count >= lo and (hi is None or count <= hi)

    labels = labels_for(start)
    count = _surviving(labels, seconds, minimum)
    if fits(count):
        return labels
    too_many = hi is not None and count > hi
    step = 1 if too_many else -1
    position = candidates.index(start) if start in candidates else 0
    reachable = [count]
    while 0 <= position + step < len(candidates):
        position += step
        labels = labels_for(candidates[position])
        count = _surviving(labels, seconds, minimum)
        if fits(count):
            return labels
        reachable.append(count)
    if too_many:
        raise ClusteringError(
            f"max_speakers {hi} cannot be met: anchors that overlap in time "
            f"cannot share a speaker, which keeps at least {min(reachable)} "
            "speakers apart"
        )
    raise ClusteringError(
        f"min_speakers {lo} cannot be met: at most {max(reachable)} anchor "
        f"cluster(s) hold the {minimum:g} s of anchor speech a speaker needs"
    )


def _ahc_clusters(
    x: np.ndarray,
    seconds: np.ndarray,
    cannot: set[tuple[int, int]],
    params: ClusteringParams,
    lo: int,
    hi: int | None,
) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    bounded = lo > 1 or hi is not None
    history = average_linkage_history(
        x, cannot, None if bounded else params.cut_distance
    )
    k_cut = 0
    while k_cut < len(history) and history[k_cut][2] <= params.cut_distance:
        k_cut += 1
    if not bounded:
        return _replay(n, history, k_cut)
    ks = list(range(len(history) + 1))
    return _fit_count(
        ks,
        lambda k: _replay(n, history, k),
        seconds,
        params.dissolve_seconds,
        lo,
        hi,
        k_cut,
    )


def _kmeans(
    points: np.ndarray, k: int, restarts: int = 10, iterations: int = 100
) -> np.ndarray:
    """Deterministic k-means++ / Lloyd with fixed seeds; best inertia wins."""
    best_labels = np.zeros(points.shape[0], dtype=np.int64)
    best_inertia = np.inf
    for seed in range(restarts):
        rng = np.random.default_rng(seed)
        centers = [points[int(rng.integers(points.shape[0]))]]
        for _ in range(1, k):
            d2 = np.min(
                ((points[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1
            )
            total = float(d2.sum())
            if total <= 0.0:
                centers.append(points[int(rng.integers(points.shape[0]))])
            else:
                centers.append(points[int(rng.choice(points.shape[0], p=d2 / total))])
        c = np.array(centers)
        labels = np.zeros(points.shape[0], dtype=np.int64)
        for _ in range(iterations):
            dist = ((points[:, None, :] - c[None]) ** 2).sum(-1)
            new = np.argmin(dist, axis=1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for j in range(k):
                members = points[labels == j]
                if len(members):
                    c[j] = members.mean(axis=0)
        inertia = float(((points - c[labels]) ** 2).sum())
        if inertia < best_inertia - 1e-12:
            best_inertia, best_labels = inertia, labels
    return best_labels


def _spectral_labels(x: np.ndarray, pval: float, k_lo: int, k_hi: int) -> np.ndarray:
    import scipy.linalg

    n = x.shape[0]
    if n < 3:
        return np.zeros(n, dtype=np.int64)
    sim = x @ x.T
    keep = pval if n * pval >= 6 else 6.0 / n
    n_drop = int((1 - keep) * n)
    pruned = sim.copy()
    if n_drop > 0:
        order = np.argsort(pruned, axis=1, kind="stable")[:, :n_drop]
        np.put_along_axis(pruned, order, 0.0, axis=1)
    sym = 0.5 * (pruned + pruned.T)
    np.fill_diagonal(sym, 0.0)
    laplacian = np.diag(np.abs(sym).sum(axis=1)) - sym
    eigvals, eigvecs = scipy.linalg.eigh(laplacian)
    window = eigvals[k_lo - 1 : k_hi + 1]
    k = k_lo if len(window) < 2 else int(np.argmax(np.diff(window))) + k_lo
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    return _relabel(_kmeans(eigvecs[:, :k], k).tolist())


def _merge_close(labels: np.ndarray, x: np.ndarray, threshold: float) -> np.ndarray:
    labels = labels.copy()
    while True:
        ids = np.unique(labels)
        if len(ids) < 2:
            break
        centers = _unit_rows(np.stack([x[labels == c].mean(axis=0) for c in ids]))
        sim = centers @ centers.T
        np.fill_diagonal(sim, -np.inf)
        a, b = divmod(int(np.argmax(sim)), len(ids))
        if sim[a, b] <= threshold:
            break
        labels[labels == ids[b]] = ids[a]
    return _relabel(labels.tolist())


def _spectral_clusters(
    x: np.ndarray,
    seconds: np.ndarray,
    params: ClusteringParams,
    lo: int,
    hi: int | None,
) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k_hi = min(hi if hi is not None else params.spectral_max_speakers, max(1, n - 1))
    labels = _spectral_labels(x, params.spectral_pval, min(lo, k_hi), k_hi)
    if params.spectral_merge_similarity > 0.0:
        labels = _merge_close(labels, x, params.spectral_merge_similarity)
    return labels


def _refine_clusters(
    x: np.ndarray,
    seconds: np.ndarray,
    original: Sequence[str],
    cannot: set[tuple[int, int]],
    params: ClusteringParams,
    lo: int,
    hi: int | None,
) -> np.ndarray:
    """Split, then merge, pyannote's labels on anchor evidence."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    groups: list[list[int]] = []
    by_label: dict[str, list[int]] = {}
    for pos, label in enumerate(original):
        by_label.setdefault(label, []).append(pos)
    pending = [
        by_label[label] for label in sorted(by_label, key=lambda lb: by_label[lb][0])
    ]

    def two_way(members: list[int]) -> tuple[list[int], list[int], float] | None:
        if len(members) < 2:
            return None
        sub = x[members]
        if len(members) == 2:
            sides = np.array([1, 2])
        else:
            tree = linkage(sub, method="average", metric="cosine")
            sides = fcluster(tree, 2, criterion="maxclust")
        left = [m for m, s in zip(members, sides) if s == sides[0]]
        right = [m for m, s in zip(members, sides) if s != sides[0]]
        if not left or not right:
            return None
        cross = float((x[left] @ x[right].T).mean())
        return left, right, cross

    while pending:
        members = pending.pop(0)
        split = two_way(members)
        if split is not None:
            left, right, cross = split
            if (
                cross < params.split_similarity
                and seconds[left].sum() >= params.dissolve_seconds
                and seconds[right].sum() >= params.dissolve_seconds
            ):
                pending[:0] = [left, right]
                continue
        groups.append(members)

    def group_sim(a: list[int], b: list[int]) -> float:
        return float((x[a] @ x[b].T).mean())

    def blocked(a: list[int], b: list[int]) -> bool:
        sa = set(a)
        return any((p in sa and q in b) or (q in sa and p in b) for p, q in cannot)

    def merge_best(threshold: float | None, only: Callable[[int], bool]) -> bool:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            if not only(i):
                continue
            for j in range(i + 1, len(groups)):
                if not only(j):
                    continue
                s = group_sim(groups[i], groups[j])
                if (threshold is None or s >= threshold) and (
                    best is None or s > best[0]
                ):
                    if not blocked(groups[i], groups[j]):
                        best = (s, i, j)
        if best is None:
            return False
        _s, i, j = best
        groups[i] = sorted(groups[i] + groups[j])
        del groups[j]
        return True

    while merge_best(params.merge_similarity, lambda _i: True):
        pass

    def labels_now() -> np.ndarray:
        out = np.zeros(n, dtype=np.int64)
        for g, members in enumerate(groups):
            out[members] = g
        return out

    def alive(i: int) -> bool:
        return float(seconds[groups[i]].sum()) >= params.dissolve_seconds

    while (
        hi is not None
        and _surviving(labels_now(), seconds, params.dissolve_seconds) > hi
    ):
        if not merge_best(None, alive):
            break
    while _surviving(labels_now(), seconds, params.dissolve_seconds) < lo:
        options = []
        for g, members in enumerate(groups):
            split = two_way(members)
            if split is not None:
                options.append((split[2], g, split[0], split[1]))
        if not options:
            break
        _cross, g, left, right = min(options, key=lambda o: (o[0], o[1]))
        groups[g : g + 1] = [left, right]
    return _relabel(labels_now().tolist())


# --------------------------------------------------------------------------
# Assignment and output
# --------------------------------------------------------------------------


def _overlap_conflicts(turns: Sequence[Turn], fraction: float) -> list[list[int]]:
    """Per turn: turns with another pyannote label overlapping it by > ``fraction``
    of the shorter of the two."""
    out: list[list[int]] = [[] for _ in turns]
    for i, (s1, e1, l1) in enumerate(turns):
        for j in range(i + 1, len(turns)):
            s2, e2, l2 = turns[j]
            if s2 >= e1:
                break
            if l1 == l2:
                continue
            if min(e1, e2) - max(s1, s2) > fraction * min(e1 - s1, e2 - s2):
                out[i].append(j)
                out[j].append(i)
    return out


def _accept(
    sims: np.ndarray, candidates: np.ndarray, tau: float, margin: float
) -> int | None:
    candidates = (
        candidates[np.isfinite(sims[candidates])] if candidates.size else candidates
    )
    if candidates.size == 0:
        return None
    values = sims[candidates]
    order = np.argsort(-values, kind="stable")
    top = float(values[order[0]])
    second = float(values[order[1]]) if candidates.size > 1 else -np.inf
    if top >= tau and top - second >= margin:
        return int(candidates[order[0]])
    return None


def _first_appearance_names(
    turns: Sequence[Turn], labels: Sequence[int | None]
) -> dict[int, str]:
    names: dict[int, str] = {}
    for _turn, c in zip(turns, labels):
        if c is not None and c not in names:
            names[c] = LABEL_FORMAT.format(len(names))
    return names


def _passthrough(
    turns: Sequence[Turn],
    params: ClusteringParams,
    n_spans: int,
    n_anchors: int,
    lo: int,
    hi: int | None,
) -> TurnAssignment:
    order: dict[str, int] = {}
    for _s, _e, label in turns:
        order.setdefault(label, len(order))
    labels: list[str | None] = [
        LABEL_FORMAT.format(order[lab]) for _s, _e, lab in turns
    ]
    routes = [ROUTE_FALLBACK] * len(turns)
    audit = _audit(
        turns,
        labels,
        routes,
        params,
        lo,
        hi,
        n_spans=n_spans,
        n_anchors=n_anchors,
        n_before=0,
        cannot_pairs=0,
        dissolved_anchors=n_anchors,
        anchor_seconds={},
        anchor_counts={},
    )
    audit[PASSTHROUGH] = "no anchor cluster survived; pyannote labels kept"
    return TurnAssignment(list(turns), labels, routes, audit)


def _audit(
    turns: Sequence[Turn],
    labels: Sequence[str | None],
    routes: Sequence[str],
    params: ClusteringParams,
    lo: int,
    hi: int | None,
    *,
    n_spans: int,
    n_anchors: int,
    n_before: int,
    cannot_pairs: int,
    dissolved_anchors: int,
    anchor_seconds: dict[str, float],
    anchor_counts: dict[str, int],
) -> dict[str, object]:
    route_counts = {route: 0 for route in ROUTES}
    route_seconds = {route: 0.0 for route in ROUTES}
    per_label: dict[str, dict[str, float]] = {}
    for (start, end, _orig), label, route in zip(turns, labels, routes):
        route_counts[route] += 1
        route_seconds[route] += end - start
        if label is not None:
            entry = per_label.setdefault(label, {"turns": 0, "seconds": 0.0})
            entry["turns"] += 1
            entry["seconds"] += end - start
    clusters = [
        {
            "label": label,
            "anchors": anchor_counts.get(label, 0),
            "anchor_seconds": round(anchor_seconds.get(label, 0.0), 3),
            "turns": int(per_label[label]["turns"]),
            "seconds": round(per_label[label]["seconds"], 3),
        }
        for label in sorted(per_label)
    ]
    n_out = len(per_label)
    return {
        "recipe": RECIPE,
        "method": params.method,
        "params": asdict(params),
        "speaker_bounds": {
            "min": lo,
            "max": hi,
            "satisfied": n_out >= lo and (hi is None or n_out <= hi),
        },
        "counts": {
            "turns": len(turns),
            "embedded_spans": n_spans,
            "anchors": n_anchors,
            "clusters_before_dissolve": n_before,
            "clusters": n_out,
            "dissolved_anchors": dissolved_anchors,
            "cannot_link_pairs": cannot_pairs,
            **{f"turns_{route}": route_counts[route] for route in ROUTES},
        },
        "seconds": {
            f"turns_{route}": round(route_seconds[route], 3) for route in ROUTES
        },
        "abstained_seconds": round(route_seconds[ROUTE_ABSTAIN], 3),
        "clusters": clusters,
    }


__all__ = [
    "ClusteringError",
    "ClusteringParams",
    "ClusteringResult",
    "Embed",
    "EmbedFn",
    "LABEL_FORMAT",
    "METHODS",
    "PASSTHROUGH",
    "RECIPE",
    "ROUTES",
    "ROUTE_ABSTAIN",
    "ROUTE_ANCHOR",
    "ROUTE_FALLBACK",
    "ROUTE_GLOBAL",
    "ROUTE_LOCAL",
    "Span",
    "Turn",
    "TurnAssignment",
    "assign_turns",
    "average_linkage_history",
    "cluster_turns",
    "embedding_spans",
    "overlap_trimmed_pieces",
    "union_same_label",
]
