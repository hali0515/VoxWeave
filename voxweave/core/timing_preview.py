"""How long a candidate cue will actually be *displayed*, before shot snapping.

A partition optimizer scores candidate cues one at a time, and every duration
term it can ask about -- "is this cue readable at the configured CPS?", "does it
clear the minimum display duration?", "does it blow the duration cap?" -- is a
question about the *display* span, not the acoustic one. The two differ by every
rule in :func:`voxweave.core.timing._cleanup_cues`: the min-duration floor, the
lag-out pad measured from the speech anchor, the CPS reading linger capped at
``LINGER_CAP_S`` past that same anchor, the two-frame no-extend guard, gap
chaining, and the duration cap with its held-word waiver.

Scoring against the raw span -- or against a hand-rolled restatement of the pass
-- mis-ranks candidates systematically rather than randomly. The predictor this
module replaces was measured over-predicting by 0.75 s on short cues (it
hardcoded ``1.0`` for the CPS term instead of importing ``LINGER_CAP_S`` and
taking ``min(start + need, lag_anchor + LINGER_CAP_S)``), by 0.083 s across the
whole chaining band (it clamped to ``next_start`` rather than ``next_start -
TWO_FRAME_S``), by 0.05 s inside the two-frame band (it was blind to the
no-extend guard), and *under*-predicting by 0.45 s whenever a held word carried
a cue past the cap. A predictor with a per-regime bias is worse than no
predictor: it moves the optimum, quietly.

**``predicted_pre_shot``** is the published name of the quantity returned here.
Shot snapping (:func:`voxweave.core.timing._snap_to_shots`) runs *after* cleanup
and is deliberately outside the preview: it needs the neighbouring cue stream
and the shot-cut list, it may move a cue's *start* as well as its end, and its
effect on one candidate is not a function of that candidate alone. The optimizer
therefore scores the pre-snap duration and treats snap displacement as its own
cut-local feature.

:class:`DisplayTimingPreview` is the swap seam. P4 ships
:class:`LegacyCleanupPreview`, a faithful mirror of today's pass; P5 hands in
the finalizer's own preview and the cost model does not change a line.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .layout import _reading_chars
from .partition_check import ReportTag, Waiver
from .schema import Unit
from .segdoc import DisplayProfile
from .timing import (
    CHAIN_MAX_GAP_S,
    HELD_WORD_MAX_GAP_S,
    LINGER_CAP_S,
    TWO_FRAME_S,
)

__all__ = [
    "CueCandidate",
    "CuePreview",
    "DisplayTimingPreview",
    "LegacyCleanupPreview",
]


@dataclass(frozen=True)
class CueCandidate:
    """Everything a preview may read about one candidate cue.

    Carried whole rather than as loose arguments because the two
    implementations read *different* subsets and the seam must not grow a
    parameter every time one of them learns something: the legacy mirror scores
    ``text`` and the four thresholds, while the finalizer's preview projects
    ``word_data`` and reads the speech anchors. ``next_start`` is phase-2
    context and is present only so a caller need not carry two shapes -- the
    finalizer's preview ignores it by contract.
    """

    start: float | None
    end: float | None
    next_start: float | None
    text: str
    word_data: Sequence[Unit]
    speech_start: float | None
    speech_end: float | None
    profile: DisplayProfile
    expected_footprint: str | None = None


@dataclass(frozen=True)
class CuePreview:
    """What a preview promises about one candidate: span, text, load, refusals.

    The refusal channel is what makes the preview honest. A predictor that can
    only return a number has to encode "this cue cannot be laid out legally" as
    some number, and every such encoding is a silent lie to whoever ranks the
    candidates.
    """

    display_start: float | None
    display_end: float | None
    final_text: str
    line_count: int
    reading_chars: int
    #: Always empty in P5: the only waiver kind is minted by the duration cap,
    #: which is a sweep rule and therefore outside the preview by contract.
    waivers: tuple[Waiver, ...] = ()
    refusals: tuple[ReportTag, ...] = ()

    @property
    def available_s(self) -> float:
        """The previewed display duration; ``0.0`` when either bound is absent."""
        if self.display_start is None or self.display_end is None:
            return 0.0
        return self.display_end - self.display_start


@runtime_checkable
class DisplayTimingPreview(Protocol):
    """Predicts the display duration a timing pass will grant one cue.

    The implementation is a pure function of the candidate itself plus the one
    piece of context the timing pass uses, ``next_start`` -- deliberately not the
    whole cue stream, so an optimizer can call it per edge without materializing
    a partition. Everything a mirror of the current pass needs is explicit:
    ``word_data`` for the speech anchor and the held-word walk, ``text`` for the
    reading load, and the four cleanup thresholds.
    """

    def preview_display_span(
        self,
        start: float,
        end: float,
        next_start: float | None,
        *,
        text: str,
        word_data: Sequence[Unit],
        min_cue_s: float,
        max_cue_s: float,
        cps: float = 0.0,
        lag_out_s: float = 0.0,
    ) -> float:
        """Return ``predicted_pre_shot``: the cue's display duration in seconds.

        ``start``/``end`` are the candidate's raw span, ``next_start`` the next
        cue's start or ``None`` at the document end. The cue's start is never
        moved by the modelled pass, so the duration fully describes the result.
        """
        ...

    def preview_cue(self, candidate: CueCandidate) -> CuePreview:
        """Return the full promise: display span, delivered text, refusals.

        Wider than :meth:`preview_display_span` on purpose. A cost model that
        prices reading load off the raw string and duration off this scalar is
        pricing two different cues, and the P5 finalizer's own preview answers
        both questions from one projection.
        """
        ...


@dataclass(frozen=True)
class LegacyCleanupPreview:
    """Mirror of :func:`voxweave.core.timing._cleanup_cues` for a single cue.

    ``_cleanup_cues`` walks the stream mutating only ``end``, and reads its
    neighbour solely as ``out[i + 1]["start"]`` -- a value the pass never writes.
    Each cue's outcome is therefore an exact function of that cue plus
    ``next_start``, which is what makes this per-candidate mirror possible at
    all rather than merely approximate.

    Stateless and frozen: one instance can be shared, and its identity carries
    no configuration (the thresholds ride on every call, because the optimizer
    may score against more than one profile).
    """

    def preview_display_span(
        self,
        start: float,
        end: float,
        next_start: float | None,
        *,
        text: str,
        word_data: Sequence[Unit],
        min_cue_s: float,
        max_cue_s: float,
        cps: float = 0.0,
        lag_out_s: float = 0.0,
    ) -> float:
        """See :meth:`DisplayTimingPreview.preview_display_span`.

        Statement order matches the pass exactly, down to the operand order of
        every ``min``/``max``, so the two agree bit for bit in float arithmetic
        rather than merely to a tolerance.
        """
        # desired end: min-dur floor, CPS reading time (capped linger), tail pad.
        # Both the pad and the linger cap anchor on the cue's speech end when it
        # has timed word_data, which is what keeps the real pass idempotent.
        speech_end = _speech_end(word_data)
        lag_anchor = end if speech_end is None else speech_end
        want = end
        if min_cue_s > 0:
            want = max(want, start + min_cue_s)
        if lag_out_s > 0:
            want = max(want, lag_anchor + lag_out_s)
        if cps > 0:
            need = _reading_chars(text) / cps
            want = max(want, min(start + need, lag_anchor + LINGER_CAP_S))

        cur_end = end
        if want > cur_end:
            if next_start is None:
                cur_end = want
            elif next_start - cur_end > TWO_FRAME_S:
                cur_end = min(want, next_start)
            # else: the gap is already at/under the 2-frame floor -- no extension.

        # chaining: close small inter-cue gaps to 2 frames
        if next_start is not None:
            gap = next_start - cur_end
            if 0 <= gap < CHAIN_MAX_GAP_S and gap > TWO_FRAME_S:
                cur_end = next_start - TWO_FRAME_S

        # duration cap, waived for a word still sounding past it
        if max_cue_s and cur_end - start > max_cue_s:
            cap = start + max_cue_s
            timed = sorted(
                (
                    (s, e)
                    for w in word_data
                    if (s := w.get("start")) is not None
                    and (e := w.get("end")) is not None
                ),
                key=lambda unit: unit[0],
            )
            last_word_end = max((e for _s, e in timed), default=None)
            if last_word_end is not None and last_word_end > cap:
                held_end = timed[0][1]
                for (_ps, pe), (ns, ne) in zip(timed, timed[1:]):
                    if ns - pe > HELD_WORD_MAX_GAP_S:
                        break
                    held_end = ne
                target = held_end
                if next_start is not None:
                    target = min(target, next_start)
                cur_end = max(cap, target)
            else:
                cur_end = cap

        return cur_end - start

    def preview_cue(self, candidate: CueCandidate) -> CuePreview:
        """Wrap the scalar above with RAW text facts.

        The asymmetry against the finalizer's preview is declared, not
        accidental: this mirror reports the candidate's raw join, its raw
        newline count and its raw reading load, because that is what the pass it
        mirrors actually consumes. Registry FD-1 is exactly the duration
        difference that follows from the canonical projection stripping
        punctuation the raw load still charges for.

        ``display_end`` is defined as ``start + span`` and has no value without
        a start, so an untimed candidate previews ``None`` rather than being
        handed back its own end as though a span had been computed.
        """
        profile = candidate.profile
        display_end: float | None = None
        if candidate.start is not None and candidate.end is not None:
            display_end = candidate.start + self.preview_display_span(
                candidate.start,
                candidate.end,
                candidate.next_start,
                text=candidate.text,
                word_data=candidate.word_data,
                min_cue_s=profile.min_cue_s,
                max_cue_s=profile.max_cue_s,
                cps=profile.cps,
                lag_out_s=profile.lag_out_s,
            )
        return CuePreview(
            display_start=candidate.start,
            display_end=display_end,
            final_text=candidate.text,
            line_count=candidate.text.count("\n") + 1,
            reading_chars=_reading_chars(candidate.text),
            waivers=(),
            refusals=(),
        )


def _speech_end(word_data: Sequence[Unit]) -> float | None:
    """Latest word end, or ``None`` when nothing in the cue is timed.

    Mirrors :func:`voxweave.core.timing._speech_end`, which takes the cue dict;
    this preview is handed the ``word_data`` alone because a candidate cue does
    not exist as a cue yet.
    """
    ends = [e for w in word_data if (e := w.get("end")) is not None]
    return max(ends) if ends else None
