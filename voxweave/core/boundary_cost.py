"""``experimental_policy_2``: the only place P4/P5 encodes a preference.

Legality lives in the lattice; everything here is opinion, and it is kept
separable on purpose. A cost that could veto is a hard rule in disguise, so no
term returns an infinity and no term reads the environment -- the whole policy is
module-level constants under :data:`POLICY_VERSION`, which is what a revision has
to bump before its numbers may be compared with an earlier run's.

Every raw feature is recorded even when its weight is zero. The zero-weighted
migration feature is the clearest case: it is not priced, but knowing how far a
candidate boundary sits from v1's is exactly what the shadow exists to measure,
and a feature that is only recorded when it is charged for cannot answer that.

**The pause term.** A cut is cheaper where speech actually paused, but the pause
is a *measurement*, and the ramp that prices it has a knee. Evaluating the ramp
at a point estimate makes the score jump across that knee for a gap that moved
by less than the measurement error. So the term is the mean of the ramp over the
uncertainty interval around the gap, in closed form -- continuous everywhere,
and provably identical to sampling with infinitely many samples. Which ramp
applies depends on what evidence exists, and the three regimes are genuinely
different claims:

* VAD confirmed the silence -> the short ramp;
* no VAD at all -> the ramp scaled by the offline-versus-clause threshold ratio,
  mirroring the stricter contract the gap classifier already uses when it is
  blind (note this scales the *ramp*: the offline threshold is not itself a
  cliff in this model, and the earlier claim that it was has been retracted);
* a bound is missing entirely -> a flat charge, a documented policy delta from
  v1's "missing evidence is a free cut".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

from .boundary_lattice import (
    BARRIER_UNCERTAINTY_MS,
    AtomLayer,
    Edge,
    LatticeAtom,
)
from .layout import (
    _line_budget_width,
    _two_line_break,
    _vis_width,
    _wrap_units,
)
from .schema import Unit
from .segdoc import DisplayProfile, SourceUnit
from .timing_preview import CueCandidate, DisplayTimingPreview, LegacyCleanupPreview


class _UnsetPreviewFact:
    """Distinguish an omitted override from an explicit acoustic ``None``."""


_UNSET_PREVIEW_FACT = _UnsetPreviewFact()

#: Bump before comparing two runs whose numbers were produced by different
#: weights: an artifact is only meaningful against its own policy version.
POLICY_VERSION: int = 2
POLICY_NAME: str = "experimental_policy_2"

#: Comparison quantum. Every weighted term is rounded to it before summation, so
#: two paths that differ only by float association compare exactly equal rather
#: than by a hair -- which is what makes the tie-break rule reachable at all.
QUANTUM: float = 1e-6

# --- cut terms -------------------------------------------------------------
W_PARTICLE: float = 3.0
W_POS: float = 1.0
W_PAUSE: float = 3.0
W_PUNCT_AFFINITY: float = -1.0
W_SHOT_PREVIEW: float = -0.5
W_MIGRATION: float = 0.0

# --- edge terms ------------------------------------------------------------
CUE_BASE: float = 2.0
SHORT_FRAGMENT_TIGHT: float = 8.0
SHORT_FRAGMENT_LOOSE: float = 4.0
SHORT_FRAGMENT_TIGHT_MAX_W: int = 2
SHORT_FRAGMENT_LOOSE_MAX_W: int = 4
W_LINE_COUNT: float = 1.0
W_BALANCE: float = 0.05
W_READING: float = 10.0
W_MIN_DURATION: float = 6.0
W_SENTENCE_CROSS: float = 6.0

# --- pause geometry --------------------------------------------------------
RAMP_KNOWN_MS: float = 220.0
UNCERTAINTY_MS: float = 50.0
PAUSE_MISSING_BOUNDS_COST: float = 1.5

#: Four-valued on purpose: "no VAD ran" and "a bound is missing" are different
#: kinds of not-knowing and are priced differently.
VAD_STATES: tuple[str, ...] = ("absent", "missing-bounds", "silence", "speech-overlap")

#: Characters whose presence at a cue's end makes that boundary a natural one.
PUNCT_AFFINITY_CHARS: str = "。！？!?.,、，;；"


def quantize(value: float) -> float:
    """Snap to the comparison quantum. Applied before summation, never after."""
    return round(value / QUANTUM) * QUANTUM


def offline_ramp_ms(profile: DisplayProfile) -> float:
    """The ramp used when no VAD evidence exists at all.

    Scaled by the ratio the gap classifier already uses to be stricter when
    blind, so at the committed defaults it is 385 ms -- *not* the offline
    threshold itself, and identical for ja because both sides scale together.
    The profile preflight guarantees the denominator is positive.
    """
    return RAMP_KNOWN_MS * (profile.offline_ms / profile.clause_ms)


def pause_knees(profile: DisplayProfile) -> dict[str, tuple[float, ...]]:
    """The resolved near-cliff probe points, per VAD state.

    Computed from the document's own thresholds rather than hardcoded, because a
    language preset scales them: probing 220 ms on a profile whose ramp sits at
    308 ms would test nothing. Each state gets the knees of *its own* curve plus
    the classifier thresholds that decide which curve applies.
    """
    shared = (
        profile.clause_ms,
        profile.vad_skip_ms,
        profile.vad_skip_ms + BARRIER_UNCERTAINTY_MS,
    )
    known = (RAMP_KNOWN_MS - UNCERTAINTY_MS, RAMP_KNOWN_MS + UNCERTAINTY_MS)
    offline = offline_ramp_ms(profile)
    absent = (offline - UNCERTAINTY_MS, offline + UNCERTAINTY_MS)
    return {
        "absent": tuple(sorted(set(absent + shared))),
        "missing-bounds": tuple(sorted(set(shared))),
        "silence": tuple(sorted(set(known + shared))),
        "speech-overlap": tuple(sorted(set(known + shared))),
    }


# ------------------------------------------------------------ pause evidence


@dataclass(frozen=True)
class PauseEvidence:
    """What is actually known about the silence at one candidate boundary."""

    gap_ms_raw: float | None
    vad_state: str
    overlap_fraction: float
    uncertainty_ms: float
    effective_ms: float | None
    ramp_ms: float | None

    def to_features(self) -> dict[str, float | str | None]:
        return {
            "effective_ms": self.effective_ms,
            "gap_ms_raw": self.gap_ms_raw,
            "overlap_fraction": self.overlap_fraction,
            "ramp_ms": self.ramp_ms,
            "uncertainty_ms": self.uncertainty_ms,
            "vad_state": self.vad_state,
        }


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _covered_length(
    low: float, high: float, spans: Sequence[tuple[float, float]]
) -> float:
    """Length of ``[low, high)`` covered by the UNION of ``spans``.

    A plain sum of per-span intersections double-counts wherever two spans
    overlap each other, so the clipped pieces are merged first. Only the pieces
    that actually reach into the window are sorted, which keeps the common case
    (a long, already-disjoint VAD track and a short gap) linear in the span count
    with a sort over a handful of survivors.
    """
    clipped: list[tuple[float, float]] = []
    for span_start, span_end in spans:
        start = max(low, span_start)
        end = min(high, span_end)
        if end > start:
            clipped.append((start, end))
    if not clipped:
        return 0.0
    clipped.sort()
    covered = 0.0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start > current_end:
            covered += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    return covered + (current_end - current_start)


def pause_evidence(
    prev_end: float | None,
    next_start: float | None,
    *,
    speech_spans: Sequence[tuple[float, float]] | None,
    profile: DisplayProfile,
) -> PauseEvidence:
    """Classify the gap at a boundary and resolve its effective silence.

    The three-way meaning of ``speech_spans`` is preserved exactly as the gap
    classifier defines it: ``None`` is *no VAD ran*, an empty list is *VAD ran
    and found no speech* (evidence of silence), and a populated list is
    measured. Conflating the first two would silently promote an unmeasured gap
    to a confirmed pause.

    One honest scoping note about that third value. The segmentation pipeline
    cannot currently produce it: ``pipeline._copied_spans`` -- the only writer of
    ``SegDocument.vad_speech`` on that path -- collapses an empty span list to
    ``None``, mirroring ``_spans_in``'s persisted-sibling contract that "no spans
    recorded" and "empty array" mean the same thing. So the empty-list branch is
    reachable only from a direct caller (a test, or a future capture path that
    distinguishes them). It is implemented rather than removed because the
    distinction is real evidence and the day the pipeline can carry it, the cost
    model must not have to relearn it -- but a reader must not conclude from this
    docstring that a shadow run over a captured document can report
    ``vad_state == "silence"`` on the strength of an empty list.

    The overlap fraction is a true intersection length with no jitter tolerance:
    the classifier's 50 ms epsilon is the slack a *boolean* needs, and applying
    it to a fraction would quantize a continuous measurement for no reason.
    "True" means the covered *union*: overlapping spans (which a hand-edited or
    foreign sibling JSON can carry, since ``_spans_in`` neither sorts nor merges)
    would otherwise have their shared time counted twice and could push the
    fraction past 1.0, under-reporting the effective silence and making the cut
    look more expensive than the evidence says.
    """
    if (
        prev_end is None
        or next_start is None
        or not _finite(prev_end)
        or not _finite(next_start)
    ):
        return PauseEvidence(
            gap_ms_raw=None,
            vad_state="missing-bounds",
            overlap_fraction=0.0,
            uncertainty_ms=UNCERTAINTY_MS,
            effective_ms=None,
            ramp_ms=None,
        )
    low = float(prev_end)
    high = float(next_start)
    gap_seconds = high - low
    gap_ms = max(0.0, gap_seconds * 1000.0)
    if speech_spans is None:
        return PauseEvidence(
            gap_ms_raw=gap_ms,
            vad_state="absent",
            overlap_fraction=0.0,
            uncertainty_ms=UNCERTAINTY_MS,
            effective_ms=gap_ms,
            ramp_ms=offline_ramp_ms(profile),
        )
    overlapped = _covered_length(low, high, speech_spans) if gap_seconds > 0 else 0.0
    fraction = 0.0 if gap_seconds <= 0 else min(1.0, overlapped / gap_seconds)
    return PauseEvidence(
        gap_ms_raw=gap_ms,
        vad_state="speech-overlap" if fraction > 0 else "silence",
        overlap_fraction=fraction,
        uncertainty_ms=UNCERTAINTY_MS,
        effective_ms=gap_ms * (1.0 - fraction),
        ramp_ms=RAMP_KNOWN_MS,
    )


def ramp_integral_mean(
    effective_ms: float,
    ramp_ms: float,
    *,
    amplitude: float = W_PAUSE,
    uncertainty_ms: float = UNCERTAINTY_MS,
) -> float:
    """Mean of ``amplitude * max(0, 1 - x/ramp)`` over the uncertainty interval.

    Piecewise-analytic, never sampled. With ``a = eff - U`` and ``b = eff + U``:

    * ``b <= ramp`` -- the interval is wholly inside the linear region, so the
      mean is the value at its midpoint;
    * ``a >= ramp`` -- wholly past the knee, zero;
    * otherwise the interval straddles the knee and the mean is the area of the
      remaining triangle divided by the interval width.

    ``a`` is deliberately not clamped at zero. The interval is an uncertainty
    band on a measurement, not a physical duration, and clamping it would make
    the term discontinuous exactly where the band first touches zero.
    """
    if ramp_ms <= 0:
        return 0.0
    low = effective_ms - uncertainty_ms
    high = effective_ms + uncertainty_ms
    if high <= ramp_ms:
        return amplitude * (1.0 - effective_ms / ramp_ms)
    if low >= ramp_ms:
        return 0.0
    return amplitude * (ramp_ms - low) ** 2 / (2.0 * ramp_ms) / (2.0 * uncertainty_ms)


def pause_cut_cost(evidence: PauseEvidence) -> float:
    """Price one boundary's pause evidence."""
    if evidence.effective_ms is None or evidence.ramp_ms is None:
        return PAUSE_MISSING_BOUNDS_COST
    return ramp_integral_mean(evidence.effective_ms, evidence.ramp_ms)


# --------------------------------------------------------------- breakdowns


@dataclass(frozen=True)
class CostBreakdown:
    """Raw features and the weighted terms they produced, plus the total.

    Both halves are kept because they answer different questions: the weighted
    terms say what the policy charged, the raw features say what the document
    actually looked like. Re-weighting an old artifact is only possible with the
    second.
    """

    features: Mapping[str, bool | float | str | None]
    weighted_terms: Mapping[str, float]
    total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": {key: self.features[key] for key in sorted(self.features)},
            "total": self.total,
            "weighted_terms": {
                key: self.weighted_terms[key] for key in sorted(self.weighted_terms)
            },
        }


def make_breakdown(
    features: Mapping[str, bool | float | str | None],
    weighted_terms: Mapping[str, float],
) -> CostBreakdown:
    """Quantize every term, then sum the quantized values -- in that order."""
    quantized = {key: quantize(value) for key, value in weighted_terms.items()}
    return CostBreakdown(
        features=dict(features),
        weighted_terms=quantized,
        total=quantize(sum(quantized.values())),
    )


def _numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def sum_breakdowns(parts: Iterable[CostBreakdown]) -> CostBreakdown:
    """Pool breakdowns for a whole interval or partition.

    Numeric features and weighted terms sum key-wise; a categorical feature
    (``vad_state``) has no sum and is dropped rather than being turned into a
    misleading aggregate. It survives on the per-cut breakdowns, which is where
    a reader should look for it anyway.
    """
    items = list(parts)
    feature_values: dict[str, list[Any]] = {}
    term_totals: dict[str, float] = {}
    for part in items:
        for key, value in part.features.items():
            feature_values.setdefault(key, []).append(value)
        for key, value in part.weighted_terms.items():
            term_totals[key] = term_totals.get(key, 0.0) + value
    features: dict[str, bool | float | str | None] = {}
    for key, values in feature_values.items():
        if len(values) == len(items) and all(_numeric(value) for value in values):
            features[key] = float(sum(values))
    return make_breakdown(features, term_totals)


# ------------------------------------------------------------------ context


@dataclass(frozen=True)
class CostContext:
    """Everything a cost term needs that is not the edge or the cut itself."""

    profile: DisplayProfile
    preview: DisplayTimingPreview
    speech_spans: Sequence[tuple[float, float]] | None
    shot_changes: Sequence[float] | None
    sentence_nodes: frozenset[int]
    v1_cut_units: frozenset[int]
    layer: AtomLayer
    # W3 additions.  ``speaker_evidence`` retains parent support/lineage while
    # ``unit_speakers`` is the index-aligned optimizer-space projection costs
    # consume.  Kept as ``Any`` here to avoid a boundary_cost ->
    # speaker_evidence import cycle: the speaker module itself returns this
    # module's CostBreakdown.
    units: Sequence[SourceUnit] = ()
    unit_speakers: tuple[Any, ...] = ()
    speaker_evidence: Any = None
    sing_spans: Sequence[tuple[float, float]] | None = None
    speaker_weight: float = 0.0

    def next_start_after(self, document_node: int) -> float | None:
        """The first known start at or after a document atom-stream node.

        Edge-local by construction. Defining this against the *partition* would
        make an edge's cost depend on which other edges were chosen, and a cost
        that is not a function of its own edge cannot be optimized by a shortest
        path at all.
        """
        for atom in self.layer.atoms[max(document_node, 0) :]:
            if atom.start is not None:
                return atom.start
        return None


# ----------------------------------------------------------------- cut cost


def _shot_distance(
    cut_time: float | None, shot_changes: Sequence[float] | None, window: float
) -> float | None:
    if cut_time is None or not shot_changes or window <= 0:
        return None
    nearest = min(abs(float(shot) - cut_time) for shot in shot_changes)
    return nearest if nearest < window else None


def transition_time(left_end: float | None, right_start: float | None) -> float | None:
    """The frozen source-boundary coordinate used by pricing and lineage.

    The right unit's start wins when present; only a missing right bound falls
    back to the left end.  In particular this is neither the midpoint nor the
    left edge of a nonzero gap (W1 R8-2).
    """
    return right_start if right_start is not None else left_end


def cut_cost(
    left: LatticeAtom,
    right: LatticeAtom,
    *,
    unit_id: int,
    profile: DisplayProfile,
    speech_spans: Sequence[tuple[float, float]] | None,
    shot_changes: Sequence[float] | None,
    v1_cut_units: AbstractSet[int] = frozenset(),
) -> CostBreakdown:
    """What one boundary between two atoms costs.

    The linguistic terms read penalties the engine's own analyzers already
    attached, rather than re-scoring the text: a second scorer would be a second
    opinion, and the shadow is supposed to be measuring the boundary decision,
    not the part-of-speech tagger.
    """
    evidence = pause_evidence(
        left.end, right.start, speech_spans=speech_spans, profile=profile
    )
    cut_time = transition_time(left.end, right.start)
    distance = _shot_distance(cut_time, shot_changes, profile.shot_snap_s)
    shot_term = (
        0.0
        if distance is None
        else W_SHOT_PREVIEW * (1.0 - distance / profile.shot_snap_s)
    )
    tail = left.text.rstrip()
    affinity = 1.0 if tail and tail[-1] in PUNCT_AFFINITY_CHARS else 0.0
    migration = 0.0 if unit_id in v1_cut_units else 1.0

    features: dict[str, float | str | None] = {
        "particle_raw": float(left.end_pen + right.start_pen),
        "pos_raw": float(right.boundary_pen),
        "punct_affinity_raw": affinity,
        "shot_preview_raw": distance,
        "migration_raw": migration,
    }
    features.update(evidence.to_features())
    return make_breakdown(
        features,
        {
            "particle": W_PARTICLE * (left.end_pen + right.start_pen),
            "pos": W_POS * right.boundary_pen,
            "pause_cut": pause_cut_cost(evidence),
            "punct_affinity": W_PUNCT_AFFINITY * affinity,
            "shot_preview": shot_term,
            "migration": W_MIGRATION * migration,
        },
    )


# ---------------------------------------------------------------- edge cost


#: How a candidate cue's priced line set was obtained. Categorical, so
#: ``sum_breakdowns`` drops it from interval aggregates on purpose -- it belongs
#: on the per-edge breakdown, next to the widths it explains.
LAYOUT_SOURCES: tuple[str, ...] = (
    "greedy-packer",
    "renderer-single-line",
    "renderer-two-line",
)


@dataclass(frozen=True)
class RenderedLayout:
    """The line set a candidate cue is priced against, and where it came from."""

    lines: int
    line_widths: tuple[int, ...]
    source: str

    @property
    def balance(self) -> float:
        if self.lines != 2:
            return 0.0
        return float(abs(self.line_widths[0] - self.line_widths[1]))


def rendered_layout(display_text: str, profile: DisplayProfile) -> RenderedLayout:
    """The lines the renderer will deliver for one candidate cue's display text.

    The packer that certifies an edge *legal* is a greedy first-fit -- the right
    oracle for "does this fit at all", and the wrong one for "how balanced does
    this read", because the pass that actually folds the cue
    (:func:`layout.wrap_cue_text`) picks a two-line break by exhaustive scoring.
    Pricing the greedy widths punishes exactly the candidates the renderer folds
    best: a cue that just fills one greedy line reads as maximally unbalanced
    while it ships as two near-equal halves.

    So the <=2-line case is priced through the renderer's own chooser
    (:func:`layout._two_line_break`), on the stripped text the edge already
    carries. Deeper wraps keep the greedy answer -- ``wrap_cue_text`` balances
    those greedily too, but at a different target -- and say so through
    ``layout_source`` rather than pretending to a fidelity they do not have.

    Two renderer passes are deliberately NOT mirrored, and both are content
    rewrites rather than break choices: ``_merge_stutters`` (ASCII repeats folded
    to a hyphenated form, which can shorten a line by a cell) and
    ``apply_kinsoku`` (ja/zh line-end/line-start repair, which moves at most one
    glyph across the break). Mirroring either would mean re-deriving the cue's
    text here, and the text is the lattice's answer, not the cost model's.
    """
    lang = profile.language
    budget = _line_budget_width(profile.max_line_length, lang)
    units = _wrap_units(display_text, lang)
    if len(units) <= 1:
        width = _vis_width(units[0][0]) if units else _vis_width(display_text)
        return RenderedLayout(1, (width,), "renderer-single-line")
    total = sum(_vis_width(atom) for atom, _gap in units) + sum(
        _vis_width(gap) for _atom, gap in units[:-1]
    )
    if total <= budget:
        return RenderedLayout(1, (total,), "renderer-single-line")
    if profile.max_lines == 2:
        chosen = _two_line_break(units, lang, budget)
        if chosen is not None:
            _index, top, bottom = chosen
            return RenderedLayout(2, (top, bottom), "renderer-two-line")
    return RenderedLayout(0, (), "greedy-packer")


def edge_cost(
    edge: Edge,
    atoms: Sequence[LatticeAtom],
    *,
    profile: DisplayProfile,
    preview: DisplayTimingPreview,
    next_start: float | None,
    sentence_cross_count: int,
    input_start: float | None | _UnsetPreviewFact = _UNSET_PREVIEW_FACT,
    input_end: float | None | _UnsetPreviewFact = _UNSET_PREVIEW_FACT,
    speech_start: float | None | _UnsetPreviewFact = _UNSET_PREVIEW_FACT,
    speech_end: float | None | _UnsetPreviewFact = _UNSET_PREVIEW_FACT,
    expected_footprint: str | None = None,
) -> CostBreakdown:
    """What one candidate cue costs, priced against its *display* duration.

    The available duration comes from the timing preview, not from the raw span:
    the pass that follows will extend a short cue, chain it to its neighbour and
    cap a long one, and scoring the raw span instead would mis-rank candidates
    systematically -- always in the same direction, which is worse than noise.

    An unresolvable span degrades to zero available time rather than raising.
    That is reachable only inside the all-invisible branch, whose chain is
    forced regardless of cost; the mixed branch declares such an edge illegal
    before it ever gets here.

    The layout terms are priced against :func:`rendered_layout` -- the lines the
    renderer will actually deliver -- not against the greedy packer state that
    decided legality. The packer's own answer is still recorded, as
    ``packed_line_count_raw``/``packed_balance_raw``, so the two readings can be
    compared in an artifact instead of being conflated in one field.
    """
    word_data: list[Unit] = [
        {"text": atom.text, "start": atom.start, "end": atom.end}
        for atom in atoms[edge.start_node : edge.end_node]
    ]
    start = (
        edge.span_start if isinstance(input_start, _UnsetPreviewFact) else input_start
    )
    end = edge.span_end if isinstance(input_end, _UnsetPreviewFact) else input_end
    acoustic_start = (
        edge.span_start if isinstance(speech_start, _UnsetPreviewFact) else speech_start
    )
    acoustic_end = (
        edge.span_end if isinstance(speech_end, _UnsetPreviewFact) else speech_end
    )
    cue_preview = preview.preview_cue(
        CueCandidate(
            start=start,
            end=end,
            next_start=next_start,
            text=edge.text,
            word_data=word_data,
            speech_start=acoustic_start,
            speech_end=acoustic_end,
            profile=profile,
            expected_footprint=expected_footprint,
        )
    )
    available: float | None = (
        None
        if cue_preview.display_start is None or cue_preview.display_end is None
        else cue_preview.available_s
    )
    usable = 0.0 if available is None else available

    if isinstance(preview, LegacyCleanupPreview):
        layout = rendered_layout(edge.display_text, profile)
        if layout.source == "greedy-packer":
            lines, balance = edge.lines, edge.balance
        else:
            lines, balance = layout.lines, layout.balance
        layout_source = layout.source
        width = edge.vis_width
    else:
        widths = tuple(_vis_width(line) for line in cue_preview.final_text.split("\n"))
        lines = cue_preview.line_count
        balance = (
            float(abs(widths[0] - widths[1]))
            if lines == 2 and len(widths) == 2
            else 0.0
        )
        layout_source = "preview-final-text"
        width = max(widths, default=0)
    if width <= SHORT_FRAGMENT_TIGHT_MAX_W:
        fragment = SHORT_FRAGMENT_TIGHT
    elif width <= SHORT_FRAGMENT_LOOSE_MAX_W:
        fragment = SHORT_FRAGMENT_LOOSE
    else:
        fragment = 0.0

    need = cue_preview.reading_chars / profile.cps if profile.cps > 0 else 0.0
    reading = W_READING * max(0.0, need - usable) / need if need > 0 else 0.0
    floor = profile.min_cue_s
    min_duration = (
        W_MIN_DURATION * max(0.0, floor - usable) / floor if floor > 0 else 0.0
    )

    features: dict[str, float | str | None] = {
        "available_s": available,
        "balance_raw": balance,
        "layout_source": layout_source,
        "line_count_raw": lines,
        "packed_balance_raw": edge.balance,
        "packed_line_count_raw": edge.lines,
        "preview_display_end": cue_preview.display_end,
        "preview_display_start": cue_preview.display_start,
        "preview_final_text": cue_preview.final_text,
        "preview_line_count": float(cue_preview.line_count),
        "preview_reading_chars": float(cue_preview.reading_chars),
        "preview_refusal_count": float(len(cue_preview.refusals)),
        "reading_need_s": need,
        "sentence_cross_raw": float(sentence_cross_count),
        "vis_width_raw": float(width),
    }
    return make_breakdown(
        features,
        {
            "cue_base": CUE_BASE,
            "short_fragment": fragment,
            "line_count": W_LINE_COUNT * (lines - 1),
            "balance": W_BALANCE * balance,
            "reading": reading,
            "min_duration": min_duration,
            "sentence_cross": W_SENTENCE_CROSS * sentence_cross_count,
        },
    )
