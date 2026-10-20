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

This module owns the segmentation *engine*: clause/sentence splitting and the
atom packing loop. Pure text helpers and display wrapping live in ``layout``;
cue-stream timing polish (glue/merge/cleanup/shot-snap) lives in ``timing``.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple, cast

from .langsets import LANGUAGES_WITHOUT_SPACES as LANGUAGES_WITHOUT_SPACES  # re-export
from .layout import (
    _fits_budget,
    _join,
    _line_budget_width,
    _merge_stutters,
    _no_spaces,
    _tokens,
    _vis_width,
    default_max_line_length,
    default_max_lines,
    split_subtitle,
    strip_punct_for_subtitles,
    wrap_cue_text,
)
from .schema import Cue, Unit
from .timing import (
    _cleanup_cues,
    _glue_short_cues,
    _merge_micro_cues,
    _snap_to_shots,
    merge_cues,
)
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
from .split_types import (
    SplitContext as SplitContext,
    SplitThresholds as SplitThresholds,
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

log = logging.getLogger(__name__)


class _BoundPair(NamedTuple):
    """Atoms of two connected cues, with the edge between them located."""

    text: str
    units: list[Unit]
    atoms: list[dict]
    boundaries: set[int]
    original: int


def _analyze_bound_pair(
    left: Cue,
    right: Cue,
    *,
    lang: str,
    max_line_length: int,
    connected_gap_s: float,
) -> _BoundPair | None:
    """Atomize a connected pair, or ``None`` when its edge needs no repair.

    The guards run cheapest first: missing units, an unknown or unconnected
    gap, atoms that do not account for every source unit, an edge that cannot
    be located, and finally an edge that strands no particle or function word.
    """
    left_units = list(left.get("word_data") or [])
    right_units = list(right.get("word_data") or [])
    if not left_units or not right_units:
        return None
    speech_end = _span_end(left_units, left.get("end"))
    speech_start = _span_start(right_units, right.get("start"))
    if speech_end is None or speech_start is None:
        return None
    gap = float(speech_start) - float(speech_end)
    if gap < -1e-6 or gap >= connected_gap_s:
        return None

    left_text = left["text"].replace("\n", "").strip()
    right_text = right["text"].replace("\n", "").strip()
    combined_text = _join([left_text, right_text], lang)
    units = left_units + right_units
    atoms = _build_atoms(
        combined_text,
        units,
        lang,
        max_atom_width=_line_budget_width(max_line_length, lang),
    )
    # Repartitioning hands `units[:unit_cut]` to the left cue, so it is only
    # safe while the atoms account for every source unit exactly; a desynced
    # pair (atoms stop short, or the cursor ran past the end) keeps the
    # boundary it already has.
    covers_all = bool(atoms) and (
        atoms[0]["_unit_start"],
        atoms[-1]["_unit_end"],
    ) == (0, len(units))
    if not covers_all:
        return None

    # The original edge sits where the left cue's units run out. Counting
    # units (not characters) is what makes this hold for a repacked pair,
    # whose entries are whole atoms.
    original = next(
        (
            atom_index
            for atom_index, atom in enumerate(atoms)
            if atom["_unit_start"] == len(left_units)
        ),
        None,
    )
    if original is None or not 0 < original < len(atoms):
        return None

    boundaries = _phrase_boundary_atoms(atoms, combined_text, lang)
    boundaries.update({0, original})
    _attach_end_penalties(atoms, boundaries, lang)
    original_right = atoms[original]
    # Only repair high-confidence particle/function-word damage.  Ordinary
    # noun or clause boundaries are left to the main/model selector.
    if _atom_end_pen(atoms[original - 1]) < 2 and _atom_start_pen(original_right) < 2:
        return None
    return _BoundPair(combined_text, units, atoms, boundaries, original)


def _try_merge_bound_pair(
    left: Cue,
    right: Cue,
    pair: _BoundPair,
    *,
    lang: str,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
) -> Cue | None:
    """The two cues joined into one when the result fits width and duration."""
    outer_start = float(left["start"])
    outer_end = float(right["end"])
    if outer_end - outer_start <= max_cue_s + 1e-9 and _fits_budget(
        pair.text, max_line_length, max_lines, lang
    ):
        return cast(
            Cue,
            {
                "text": pair.text,
                "start": left["start"],
                "end": right["end"],
                **merge_cues(left, right, lang, text=pair.text),
            },
        )
    return None


def _bound_candidate_score(
    k: int,
    atoms: list[dict],
    original_tail_pen: int,
    lang: str,
) -> tuple[int, int, int, int, int]:
    """Rank a cut before atom ``k``; the smaller tuple is the better edge."""
    edge_right = atoms[k]
    # ``hard`` sums the two sides of the edge, so a bound particle sitting
    # at the left cue's END and the same particle heading the right cue
    # score an equal tie there and ``soft`` picks either side.  That lets a
    # repartition strand a 格助詞 / 的-class particle on a cue tail — the
    # >= 2 ``line_end_penalty`` position — where the original edge had a
    # clean one.  This leading component breaks the tie asymmetrically:
    # any candidate whose left tail scores WORSE than the original edge's
    # ranks strictly below it, so the pass can only relieve a dangling tail,
    # never manufacture one.  The signal is the same ``end_pen`` the rest of
    # the pass scores with (whole-word char table for zh/yue, UniDic Level-2
    # map for ja); candidates that leave the tail as good or better keep
    # competing on the old signals exactly as before.
    worsens_tail = 1 if _atom_end_pen(atoms[k - 1]) > original_tail_pen else 0
    hard = 3 * (_atom_end_pen(atoms[k - 1]) + _atom_start_pen(edge_right))
    prev_end = atoms[k - 1].get("end")
    next_start = edge_right.get("start")
    pause = pause_damage(prev_end, next_start)
    left_surface = _join([atom["text"] for atom in atoms[:k]], lang)
    right_surface = _join([atom["text"] for atom in atoms[k:]], lang)
    micro = micro_fragment_damage(_vis_width(left_surface), _vis_width(right_surface))
    soft = _atom_boundary_pen(edge_right) + pause + micro
    return (
        worsens_tail,
        hard,
        soft,
        abs(_vis_width(left_surface) - _vis_width(right_surface)),
        -k,
    )


def _best_repartition(
    left: Cue,
    right: Cue,
    pair: _BoundPair,
    *,
    lang: str,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
) -> tuple[int, int] | None:
    """``(split_at, unit_cut)`` of the best cut beating the original edge."""
    atoms = pair.atoms
    original = pair.original
    starts = sorted(pair.boundaries - {0, len(atoms)})
    # The tail the inherited edge already leaves: a repartition may relieve a
    # dangling one, but must never hand the left cue a worse tail than this.
    original_tail_pen = _atom_end_pen(atoms[original - 1])

    original_score = _bound_candidate_score(original, atoms, original_tail_pen, lang)
    choices: list[tuple[tuple[int, int, int, int, int], int, int]] = []
    for k in starts:
        if k == original:
            continue
        left_chunk, right_chunk = atoms[:k], atoms[k:]
        left_surface = _join([atom["text"] for atom in left_chunk], lang)
        right_surface = _join([atom["text"] for atom in right_chunk], lang)
        if not _fits_budget(
            left_surface, max_line_length, max_lines, lang
        ) or not _fits_budget(right_surface, max_line_length, max_lines, lang):
            continue
        left_start = _span_start(left_chunk, left.get("start"))
        left_end = _span_end(left_chunk, left.get("end"))
        right_start = _span_start(right_chunk, right.get("start"))
        right_end = _span_end(right_chunk, right.get("end"))
        if (
            left_start is None
            or left_end is None
            or right_start is None
            or right_end is None
        ):
            continue
        if (
            float(left_end) - float(left_start) > max_cue_s + 1e-9
            or float(right_end) - float(right_start) > max_cue_s + 1e-9
        ):
            continue
        unit_cut = right_chunk[0]["_unit_start"]
        score = _bound_candidate_score(k, atoms, original_tail_pen, lang)
        if score < original_score:
            choices.append((score, k, unit_cut))
    if not choices:
        return None

    _score, split_at, unit_cut = min(choices)
    return split_at, unit_cut


def _repartitioned_cues(
    left: Cue,
    right: Cue,
    pair: _BoundPair,
    split_at: int,
    unit_cut: int,
    lang: str,
) -> list[Cue]:
    """The pair rebuilt as two cues cut before atom ``split_at``."""
    atoms, units = pair.atoms, pair.units
    new_left_atoms, new_right_atoms = atoms[:split_at], atoms[split_at:]
    # Repartitioning moves material between the two cues, so each side's
    # anchor is re-derived from the atoms it now owns -- with no display
    # fallback, unlike the visible bounds beside it.
    return [
        {
            "text": _join([atom["text"] for atom in new_left_atoms], lang),
            "start": left["start"],
            "end": cast(float, _span_end(new_left_atoms, left.get("end"))),
            "word_data": units[:unit_cut],
            "speech_start": _span_start(new_left_atoms, None),
            "speech_end": _span_end(new_left_atoms, None),
        },
        {
            "text": _join([atom["text"] for atom in new_right_atoms], lang),
            "start": cast(float, _span_start(new_right_atoms, right.get("start"))),
            "end": right["end"],
            "word_data": units[unit_cut:],
            "speech_start": _span_start(new_right_atoms, None),
            "speech_end": _span_end(new_right_atoms, None),
        },
    ]


def _repair_bound_particle_cues(
    cues: list[Cue],
    *,
    lang: str,
    max_line_length: int,
    max_lines: int,
    max_cue_s: float,
    connected_gap_s: float,
) -> list[Cue]:
    """Remove connected cue edges that strand an independently tagged particle.

    This is a final safety net for boundaries inherited from separate ASR
    segments or an earlier hard layout decision.  A direct merge is preferred;
    if it would exceed width/duration, the two cues are repartitioned at a
    better phrase/POS boundary while retaining every source unit and timestamp.
    True pauses are never crossed.  Whole-token/POS scoring distinguishes the
    particles ``了/地`` from lexical words such as ``了解/地方``.  Repartitioning
    is one-directional: a candidate edge may relieve a dangling tail but is never
    allowed to leave the left cue a worse ``line_end_penalty`` tail than the edge
    it replaces (see ``_bound_candidate_score``).
    """

    if lang not in {"zh", "yue", "ja"} or len(cues) < 2 or connected_gap_s <= 0:
        return cues

    work = [cast(Cue, dict(cue)) for cue in cues]
    i = 0
    while i + 1 < len(work):
        left, right = work[i], work[i + 1]
        pair = _analyze_bound_pair(
            left,
            right,
            lang=lang,
            max_line_length=max_line_length,
            connected_gap_s=connected_gap_s,
        )
        if pair is None:
            i += 1
            continue

        merged = _try_merge_bound_pair(
            left,
            right,
            pair,
            lang=lang,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=max_cue_s,
        )
        if merged is not None:
            work[i : i + 2] = [merged]
            if i:
                i -= 1
            continue

        cut = _best_repartition(
            left,
            right,
            pair,
            lang=lang,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cue_s=max_cue_s,
        )
        if cut is not None:
            work[i : i + 2] = _repartitioned_cues(left, right, pair, *cut, lang)
        i += 1
    return work


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
