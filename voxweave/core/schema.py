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

    ``text`` is the unit's surface form (aligner output, and the packed atom's
    surface in the repacked word_data ``_chunk_to_cue`` writes); pipeline
    word_data carries ``word`` instead (the ASR token used for cursor anchoring
    in ``split_at_sentence_end``). Spans are absolute seconds; either bound may
    be missing for ghost units.

    A stream comes at one of several granularities — one entry per non-space
    character (aligner output, ``reinject_punct``), one per packed atom (a cue
    materialized by ``_chunk_to_cue``), or one per legacy sentence-sized ASR
    ``word`` — and **the key does not say which**: ``reinject_punct`` writes
    ``text`` on char-level units. Reconcile the stored surfaces against the text
    (``smart_split._unit_ranges``) instead of assuming a granularity.
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
    ``_unit_start``/``_unit_end`` are the atom's half-open footprint in the
    ``word_data`` it was built from — the only granularity-safe way to slice that
    stream back (``diarize.format_speaker_cues``, ``_repair_bound_particle_cues``).
    """

    text: str
    start: float | None
    end: float | None
    end_pen: int
    forced_boundary: bool
    _unit_start: int
    _unit_end: int


class Cue(TypedDict):
    """One subtitle cue: display text + span + its word-level timing source.

    All four keys are invariants of the cue stream (every constructor fills
    them; timing-less cues carry ``word_data=[]``), so they are required —
    subscript access is the normal pattern downstream. ``lyric`` marks a cue
    whose span is mostly sung (keep-lyrics mode); display layers wrap it with
    music notes while the stored text stays clean.

    ``speech_start``/``speech_end`` are the cue's immutable acoustic anchors,
    captured at construction from the raw unit/atom span (``None`` when the
    timing was fabricated — invented time is not acoustic evidence). Content
    folds (micro-merge, glue, bound-particle repair) recompute them from the
    material they folded; display passes (``_cleanup_cues``, ``_snap_to_shots``,
    the diarize overlap trim) must neither read nor write them, and the sibling
    writer projects them out so persisted ``segments[]`` keeps its legacy shape.
    Nothing in legacy-v1 reads them: they arm the P5 finalizer.
    """

    text: str
    start: float
    end: float
    word_data: list[Unit]
    lyric: NotRequired[bool]
    # Transient display metadata populated only while a named split replay is
    # rendering.  The sibling writer drops it alongside the acoustic anchors.
    speaker_ids: NotRequired[list[str]]
    speech_start: NotRequired[float | None]
    speech_end: NotRequired[float | None]
