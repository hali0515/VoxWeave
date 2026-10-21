"""Bound-particle repair for zh, yue and ja cues.

Finds a cue edge that strands a bound particle, then either merges the two
cues or repartitions their atoms so the particle stays with its host. It
runs after packing and uses the break scores of ``split_scoring``.
"""

from __future__ import annotations

from typing import NamedTuple, cast

from .layout import _fits_budget, _join, _line_budget_width, _vis_width
from .schema import Cue, Unit
from .split_atoms import _build_atoms, _phrase_boundary_atoms, _span_end, _span_start
from .split_scoring import (
    _atom_boundary_pen,
    _atom_end_pen,
    _atom_start_pen,
    _attach_end_penalties,
    micro_fragment_damage,
    pause_damage,
)
from .timing import merge_cues


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
