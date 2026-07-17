"""Typed schemas for the dict shapes flowing through segmentation and timing.

These are the de facto contracts of the sibling-file pipeline (see the JSON
``word_segments`` produced by transcribe/align); TypedDicts make them explicit
so a typo'd key is a type error instead of a silently-absent value.
``Unit``/``Atom`` are ``total=False`` (ghost units lack spans, legacy
word_data has no ``word``); ``Cue`` keys are required invariants.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class Unit(TypedDict, total=False):
    """One aligned token from an aligner / ``reinject_punct``.

    ``text`` is the unit's surface form (aligner output); pipeline word_data
    carries ``word`` instead (the ASR token used for cursor anchoring in
    ``split_at_sentence_end``). Spans are absolute seconds; either bound may be
    missing for ghost units.
    """

    text: str
    word: str
    start: float | None
    end: float | None


class Atom(TypedDict, total=False):
    """A non-breakable packing unit inside one cue (see ``_build_atoms``).

    Spaced langs: one word. No-space langs: one CJK char or Latin run.
    ``end_pen`` is the precomputed line-end break penalty attached by
    ``_attach_end_penalties`` (0 = clean break point).
    ``forced_boundary`` exposes spaces inside an overlong embedded Latin run.
    """

    text: str
    start: float | None
    end: float | None
    end_pen: int
    forced_boundary: bool


class Cue(TypedDict):
    """One subtitle cue: display text + span + its word-level timing source.

    All four keys are invariants of the cue stream (every constructor fills
    them; timing-less cues carry ``word_data=[]``), so they are required —
    subscript access is the normal pattern downstream. ``lyric`` marks a cue
    whose span is mostly sung (keep-lyrics mode); display layers wrap it with
    music notes while the stored text stays clean.
    """

    text: str
    start: float
    end: float
    word_data: list[Unit]
    lyric: NotRequired[bool]
