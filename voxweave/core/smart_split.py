"""Subtitle splitting with gap-aware cue segmentation.

Two stages:
1. ``split_at_sentence_end`` — PySBD (or regex fallback) sentence boundaries,
   each grouped into comma clauses by ``_comma_clauses``. Stage 1 never splits
   for length: stage 2 does all length breaking.
2. ``split_long_cues_with_word_timings`` — word-level greedy packing into
   cues fitting ``max_lines × max_line_length``, with gap/duration breaks
   (a clause without word timings is wrapped by ``_split_without_timings``).

A stage-1 clause boundary is a cue boundary going into stage 2 but not
necessarily in the output: ``_repair_bound_particle_cues``
(zh/yue/ja: merges or repartitions a connected edge that strands a particle),
``_merge_micro_cues`` and ``_glue_short_cues`` (see ``timing``) can fold
adjacent cues across it. Each of those folds gates on a sub-pause gap, so none
crosses a real pause.

This module is the public facade and the orchestrator of the segmentation
*engine*: it keeps ``smart_split_segments``, ``split_long_cues_with_word_timings``
and ``_split_without_timings``. The engine code itself lives in sibling modules
and is re-exported here, so every ``smart_split.<name>`` import still resolves:

- ``split_types``: ``SplitThresholds`` and ``SplitContext``.
- ``split_atoms``: unit surface text, display-character ranges and atoms.
- ``split_sentences``: stage 1 (sentences, comma clauses, clause location).
- ``split_scoring``: break-point quality and length-break choice.
- ``split_packing``: stage 2 packing, hard wrap and untimed spread.
- ``split_repair``: the zh/yue/ja bound-particle repair.

Pure text helpers and display wrapping live in ``layout``; cue-stream timing
polish (glue/merge/cleanup/shot-snap) lives in ``timing``.
"""

from __future__ import annotations

import logging
from typing import Any

from .langsets import LANGUAGES_WITHOUT_SPACES as LANGUAGES_WITHOUT_SPACES  # re-export
from .layout import (
    _line_budget_width,
    _merge_stutters,
    _no_spaces,
    _tokens,
    default_max_line_length,
    default_max_lines,
    split_subtitle,
    strip_punct_for_subtitles,
    wrap_cue_text,
)
from .schema import Cue
from .split_atoms import (
    _build_atoms as _build_atoms,
    _cursor_ranges as _cursor_ranges,
    _display_chars as _display_chars,
    _phrase_boundary_atoms as _phrase_boundary_atoms,
    _span_end as _span_end,
    _span_start as _span_start,
    _surface_ranges as _surface_ranges,
    _unit_ranges as _unit_ranges,
    _unit_text as _unit_text,
)
from .split_packing import (
    UNTIMED_MIN_DISPLAY_S as UNTIMED_MIN_DISPLAY_S,
    _chunk_to_cue as _chunk_to_cue,
    _hard_wrap_surface as _hard_wrap_surface,
    _pack_atoms_into_chunks as _pack_atoms_into_chunks,
    _pack_with_oversized_fallback as _pack_with_oversized_fallback,
    _split_oversized_atom as _split_oversized_atom,
    _split_surface_mid as _split_surface_mid,
    _split_widest_until as _split_widest_until,
    _spread_untimed_cues as _spread_untimed_cues,
    _surface_parts_for_limits as _surface_parts_for_limits,
    _untimed_windows as _untimed_windows,
)
from .split_repair import (
    _BoundPair as _BoundPair,
    _analyze_bound_pair as _analyze_bound_pair,
    _best_repartition as _best_repartition,
    _bound_candidate_score as _bound_candidate_score,
    _repair_bound_particle_cues as _repair_bound_particle_cues,
    _repartitioned_cues as _repartitioned_cues,
    _try_merge_bound_pair as _try_merge_bound_pair,
)
from .split_scoring import (
    FORCE_BREAK_FACTOR as FORCE_BREAK_FACTOR,
    MICRO_FRAGMENT_DAMAGE as MICRO_FRAGMENT_DAMAGE,
    MICRO_FRAGMENT_WIDTHS as MICRO_FRAGMENT_WIDTHS,
    PAUSE_LADDER_DAMAGE as PAUSE_LADDER_DAMAGE,
    PAUSE_LADDER_MS as PAUSE_LADDER_MS,
    _atom_boundary_pen as _atom_boundary_pen,
    _atom_end_pen as _atom_end_pen,
    _atom_start_pen as _atom_start_pen,
    _attach_end_penalties as _attach_end_penalties,
    _best_len_break_pos as _best_len_break_pos,
    _classify_atom_break as _classify_atom_break,
    _gap_ms as _gap_ms,
    micro_fragment_damage as micro_fragment_damage,
    pause_damage as pause_damage,
)
from .split_sentences import (
    DEFAULT_COMMA_SPLIT_MIN_LEN as DEFAULT_COMMA_SPLIT_MIN_LEN,
    DEFAULT_COMMA_SPLIT_MIN_LEN_CJK as DEFAULT_COMMA_SPLIT_MIN_LEN_CJK,
    DEFAULT_DESIRED_WPS as DEFAULT_DESIRED_WPS,
    RESYNC_LOOKAHEAD_CLAUSES as RESYNC_LOOKAHEAD_CLAUSES,
    _ClausePlan as _ClausePlan,
    _anchor_cursor as _anchor_cursor,
    _clause_plans as _clause_plans,
    _comma_clauses as _comma_clauses,
    _comma_load as _comma_load,
    _find_clause_forward as _find_clause_forward,
    _locate_clause_units as _locate_clause_units,
    _proportional_units as _proportional_units,
    _resync_position as _resync_position,
    _segment_sentences as _segment_sentences,
    _snap_sentence_breaks as _snap_sentence_breaks,
    _split_keep_comma as _split_keep_comma,
    _unit_word_index as _unit_word_index,
    _units_match as _units_match,
    default_comma_split_min_len as default_comma_split_min_len,
    split_at_sentence_end as split_at_sentence_end,
)
from .split_types import (
    SplitContext as SplitContext,
    SplitThresholds as SplitThresholds,
)
from .timing import (
    _cleanup_cues,
    _glue_short_cues,
    _merge_micro_cues,
    _snap_to_shots,
)

log = logging.getLogger(__name__)


def _required_thresholds(value: object, caller: str) -> SplitThresholds:
    """The typed thresholds a caller must pass; a missing value is a TypeError."""
    if isinstance(value, SplitThresholds):
        return value
    if isinstance(value, dict):
        return SplitThresholds.from_mapping(value)
    raise TypeError(
        f"{caller}() needs thresholds (a SplitThresholds or a mapping such as "
        f"config.gap_thresholds(lang)), not {type(value).__name__}"
    )


def split_long_cues_with_word_timings(
    cues: list[Cue],
    max_line_length: int,
    max_lines: int,
    lang: str,
    *,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: SplitThresholds,
) -> list[Cue]:
    """Pack each cue's atoms into reading-sized cues using gap/duration/length breaks.

    ``thresholds`` is required (``SplitThresholds()`` gives the defaults). The
    arguments after ``lang`` are keyword-only, so a caller still passing the
    removed ``min_duration``/``desired_wps`` positionally gets a ``TypeError``
    instead of having them bound to other parameters.
    """
    ctx = SplitContext(
        lang=lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        th=_required_thresholds(thresholds, "split_long_cues_with_word_timings"),
        speech_spans=speech_spans,
    )
    new_cues: list[Cue] = []
    for cue in cues:
        word_data = list(cue.get("word_data") or [])
        if not word_data:
            new_cues.extend(
                _split_without_timings(cue, max_line_length, max_lines, lang, ctx=ctx)
            )
            continue
        atoms = _build_atoms(
            cue["text"],
            word_data,
            lang,
            max_atom_width=_line_budget_width(max_line_length, lang),
        )
        boundary = (
            _phrase_boundary_atoms(atoms, cue["text"], lang)
            if _no_spaces(lang)
            else None
        )
        if boundary is not None:
            boundary.update(
                i for i, atom in enumerate(atoms) if atom.get("forced_boundary")
            )
        _attach_end_penalties(atoms, boundary, lang)
        chunks = _pack_with_oversized_fallback(
            atoms, boundary=boundary, ctx=ctx, cue=cue
        )
        new_cues.extend(
            _spread_untimed_cues(
                [_chunk_to_cue(chunk, cue, lang) for chunk in chunks], ctx
            )
        )
    return new_cues


def _split_without_timings(
    cue: Cue,
    max_line_length: int,
    max_lines: int,
    lang: str,
    *,
    ctx: SplitContext | None = None,
) -> list[Cue]:
    # One indivisible token cannot be wrapped by the normal line grouper.  Use
    # the same proportional emergency fallback as timed coarse atoms.
    if ctx is not None and len(_tokens(cue["text"], lang)) == 1:
        pieces = _surface_parts_for_limits(cue["text"], cue["start"], cue["end"], ctx)
        if len(pieces) > 1:
            return [
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "word_data": [],
                    # Proportional subdivision of an untimed cue: no acoustic
                    # evidence exists for any of these pieces.
                    "speech_start": None,
                    "speech_end": None,
                }
                for text, start, end in pieces
            ]
    formatted = split_subtitle(cue["text"], max_line_length, lang)
    lines = formatted.split("\n")
    chunks: list[list[str]] = []
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) == max_lines:
            chunks.append(buf)
            buf = []
    if buf:
        chunks.append(buf)
    if not chunks:
        return [cue]

    sep = "" if _no_spaces(lang) else " "
    total_chars = sum(len(sep.join(c)) for c in chunks) or 1
    start = cue["start"]
    duration = cue["end"] - cue["start"]
    out: list[Cue] = []
    for c in chunks:
        # Join without \n — the downstream SubtitlesWriter handles display wrapping.
        text = sep.join(c)
        proportion = len(text) / total_chars
        end = start + duration * proportion if duration > 0 else start
        out.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "word_data": [],
                "speech_start": None,
                "speech_end": None,
            }
        )
        start = end
    return out


def smart_split_segments(
    segments: list[dict[str, Any]],
    lang: str,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    *,
    split_at_comma: bool = True,
    comma_split_min_len: int | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
    thresholds: SplitThresholds | dict,
    shot_changes: list[float] | None = None,
) -> list[Cue]:
    """Run the full smart-split pipeline over aligned segments.

    Each segment must have ``text`` and ``words`` (with ``start``/``end``).
    Returns a flat list of cues with ``text``, ``start``, ``end``, ``word_data``.
    Everything after ``max_lines`` is keyword-only, so a caller still passing the
    removed ``min_duration``/``desired_wps`` positionally gets a ``TypeError``
    instead of having them bound to ``split_at_comma``/``comma_split_min_len``.

    ``thresholds`` is required: a :class:`SplitThresholds`, or a mapping such as
    ``config.gap_thresholds(lang)`` (partial mappings fill the dataclass
    defaults). Omitting it, or passing ``None``, raises ``TypeError``.

    ``split_at_comma`` (default on) breaks at commas unless either side is
    shorter than ``comma_split_min_len`` visual chars. The sentence/comma
    clauses are packed by the timed atom stage and then the cue-stream folds
    run: ``_repair_bound_particle_cues``, ``_merge_micro_cues`` and
    ``_glue_short_cues`` can each merge adjacent cues across a clause boundary
    when the gap between them is below a real pause.
    """
    th = _required_thresholds(thresholds, "smart_split_segments")
    if max_line_length is None:
        max_line_length = default_max_line_length(lang)
    if max_lines is None:
        max_lines = default_max_lines(lang)  # zh/yue/ja -> 1, else 2
    # The packer budgets in native cells (a CJK preset counts one wide glyph per
    # cell) while the renderer measures half-width cells. Convert once so both
    # stages honour the same configured profile instead of the renderer silently
    # falling back to its built-in 42-column default.
    render_width = _line_budget_width(max_line_length, lang)
    all_cues: list[Cue] = []
    for segment in segments:
        text = segment.get("text", "")
        words = segment.get("words", []) or []
        if not text:
            continue
        all_cues.extend(
            split_at_sentence_end(
                text,
                words,
                lang,
                split_at_comma=split_at_comma,
                comma_split_min_len=comma_split_min_len,
            )
        )
    cues = split_long_cues_with_word_timings(
        all_cues,
        max_line_length=max_line_length,
        max_lines=max_lines,
        lang=lang,
        speech_spans=speech_spans,
        thresholds=th,
    )
    cues = _repair_bound_particle_cues(
        cues,
        lang=lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        max_cue_s=th.max_cue_s,
        connected_gap_s=th.clause_ms / 1000.0,
    )
    cues = _merge_micro_cues(
        cues,
        lang,
        max_gap_s=th.glue_gap_s,
        max_line_length=max_line_length,
        max_cue_s=th.max_cue_s,
        min_cue_s=th.min_cue_s,
        max_lines=max_lines,
    )
    cues = _glue_short_cues(
        cues,
        lang,
        max_gap_s=th.glue_gap_s,
        max_line_length=max_line_length,
        max_lines=max_lines,
        max_cue_s=th.max_cue_s,
    )
    cues = _cleanup_cues(
        cues,
        min_cue_s=th.min_cue_s,
        max_cue_s=th.max_cue_s,
        cps=th.cps,
        lag_out_s=th.lag_out_s,
    )
    if shot_changes:
        cues = _snap_to_shots(
            cues,
            sorted(shot_changes),
            snap_s=th.shot_snap_s,
            max_cue_s=th.max_cue_s,
        )
    for cue in cues:
        cue["text"] = _merge_stutters(strip_punct_for_subtitles(cue["text"]))
        # Display soft-wrap: fold over-budget cues into <=max_lines lines without
        # changing cue boundaries. Long Latin phrases inside CJK also collapse here.
        cue["text"] = wrap_cue_text(
            cue["text"], lang, max_lines, max_line_length=render_width
        )
    return cues
