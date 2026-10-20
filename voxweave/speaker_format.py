"""Speaker-aware cue formatting: the pure post-pass over smart_split cues.

Each cue's atoms get a speaker by time overlap with the persisted
``speaker_turns``; a cue containing two speakers becomes a Netflix
dual-speaker event (one line per speaker, leading hyphen, no space) when the
language allows two lines and both halves fit one line, otherwise the cue
splits at the speaker boundaries. This module holds no model or pyannote
state and never imports :mod:`voxweave.diarize`; that module re-exports every
name defined here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from voxweave.core.schema import Cue

if TYPE_CHECKING:
    from voxweave.core.smart_split import SplitThresholds

# Atom-level speaker assignment needs at least this much absolute overlap with a
# turn (seconds); below it the atom inherits its neighbors (guards 20ms grazes).
MIN_ATOM_OVERLAP_S = 0.05

# A speaker run whose total atom duration is below this is a candidate for
# absorption into an adjacent run (a word is never cut into two speaker cues).
# Whether a sub-floor run is actually absorbed is position-aware (see
# _absorb_tiny_runs): mid-phrase thrash goes, real edge utterances stay.
MIN_RUN_S = 0.2

# A sub-MIN_RUN_S run that is *not* a same-speaker (A-B-A) sandwich -- i.e. one at
# a cue edge or between two different speakers (A-B-C) -- is a real short
# utterance and is kept when at least this long; only shorter fragments are
# pyannote noise. Keeps e.g. a 160ms trailing '你好' as its own speaker cue while
# still dropping an 80ms mid-phrase fragment.
EDGE_RUN_MIN_S = 0.12

# A speaker boundary is also a cue boundary, so the left piece's tail is scored
# with the Level-2 (UniDic POS) end penalty smart_split uses for its own breaks.
# 2 = the tail cannot end an utterance at all (格助詞 with no head, 連体詞 with
# nothing to modify); splitting there is only acceptable when the cue has no
# cleaner edge to offer. ja only -- no validated equivalent grading exists for zh,
# and the Level-1 char table over-fires (準体の, adverbial に).
BAD_TAIL_PENALTY = 2

# Inter-run silence a bad-tail merge may span. A reply that starts after a beat is
# a genuine second speaker, whatever the left tail looks like -- swallowing it is
# the 2026-07-03 EDGE_RUN_MIN_S regression class. Only a boundary the diarizer
# placed inside continuous speech can plausibly be a mis-cut phrase, so the merge
# is confined to gaps below the shortest inter-speaker pause worth trusting
# (same order as MIN_RUN_S: shorter than any real turn-taking beat).
BAD_TAIL_MAX_GAP_S = 0.2

Turn = tuple[float, float, str]


def _span_speaker(
    start: float | None, end: float | None, turns: Sequence[Turn]
) -> str | None:
    """Dominant speaker for a time span by accumulated overlap (whisperX pattern).

    ``turns`` must be sorted by start (:func:`format_speaker_cues` sorts them).
    """
    if start is None or end is None or end <= start:
        return None
    overlap: dict[str, float] = {}
    for a, b, label in turns:
        if b <= start:
            continue
        if a >= end:
            break  # turns are sorted by start
        ov = min(end, b) - max(start, a)
        if ov > 0:
            overlap[label] = overlap.get(label, 0.0) + ov
    if not overlap:
        return None
    label, best = max(overlap.items(), key=lambda kv: kv[1])
    return label if best >= MIN_ATOM_OVERLAP_S else None


def _run_dur(atoms: list[dict]) -> float:
    """Total spoken duration of a run: sum of its atoms' positive durations."""
    total = 0.0
    for a in atoms:
        s, e = a.get("start"), a.get("end")
        if s is not None and e is not None and e > s:
            total += e - s
    return total


def _coalesce_runs(
    runs: list[tuple[str, list[dict]]],
) -> list[tuple[str, list[dict]]]:
    """Merge adjacent same-speaker runs (their atoms concatenate)."""
    out: list[tuple[str, list[dict]]] = []
    for lb, ats in runs:
        if out and out[-1][0] == lb:
            out[-1][1].extend(ats)
        else:
            out.append((lb, list(ats)))
    return out


def _absorbable(i: int, durs: list[float], runs: list[tuple[str, list[dict]]]) -> bool:
    """Whether run ``i`` (already known sub-``MIN_RUN_S``) should be absorbed.

    Position-aware policy:
    - A run sandwiched between two runs of the *same* speaker (A-B-A) is label
      thrash and is always absorbed, regardless of ``EDGE_RUN_MIN_S``.
    - A run at a cue edge, or between two *different* speakers (A-B-C), is a real
      short utterance: kept when >= ``EDGE_RUN_MIN_S``, absorbed only below it.
    """
    has_left = i > 0
    has_right = i + 1 < len(runs)
    sandwiched_same = has_left and has_right and runs[i - 1][0] == runs[i + 1][0]
    return sandwiched_same or durs[i] < EDGE_RUN_MIN_S


def _absorb_tiny_runs(
    runs: list[tuple[str, list[dict]]],
) -> list[tuple[str, list[dict]]]:
    """Fold absorbable sub-``MIN_RUN_S`` runs into their longer neighbor.

    Collapses A-B-A label thrash to a single run while keeping real short edge or
    A-B-C utterances (see ``_absorbable``). Repeats to a fixpoint: absorbing one
    run and re-coalescing can turn a surviving run into a new same-speaker
    sandwich, which the next pass then absorbs."""
    runs = [(lb, list(ats)) for lb, ats in runs]
    while len(runs) > 1:
        durs = [_run_dur(ats) for _, ats in runs]
        tiny = [
            (durs[i], i)
            for i in range(len(runs))
            if durs[i] < MIN_RUN_S and _absorbable(i, durs, runs)
        ]
        if not tiny:
            break
        _, i = min(tiny)  # shortest absorbable run first
        # len(runs) > 1, so at least one neighbour exists (durations are >= 0).
        left = durs[i - 1] if i > 0 else -1.0
        right = durs[i + 1] if i + 1 < len(runs) else -1.0
        if right > left:  # merge into the following (longer) run: prepend atoms
            runs[i + 1] = (runs[i + 1][0], runs[i][1] + runs[i + 1][1])
        else:  # merge into the preceding run: append atoms
            runs[i - 1] = (runs[i - 1][0], runs[i - 1][1] + runs[i][1])
        del runs[i]
        runs = _coalesce_runs(runs)
    return runs


def _snap_runs_to_phrases(
    runs: list[tuple[str, list[dict]]], lang: str
) -> list[tuple[str, list[dict]]]:
    """Snap run boundaries onto legal token edges so a run never cuts mid-word.

    No-space langs only (space-delimited atoms are already whole words). Each
    jieba/BudouX phrase is reassigned to its dominant speaker (by atom duration),
    which is exactly a boundary snap: a phrase spanning a mid-word label flip
    goes wholly to one speaker instead of splitting. Reuses the phrase-boundary
    machinery smart_split uses (``_phrase_boundary_atoms`` -> ``phrase_atoms``).
    """
    from voxweave.core.layout import _no_spaces

    if not _no_spaces(lang) or len(runs) < 2:
        return runs
    from voxweave.core.smart_split import _phrase_boundary_atoms

    flat = [a for _, ats in runs for a in ats]
    labels = [lb for lb, ats in runs for _ in ats]
    text = "".join(a["text"] for a in flat)
    boundaries = _phrase_boundary_atoms([{"text": a["text"]} for a in flat], text, lang)
    edges = sorted(boundaries | {0, len(flat)})
    new_labels = list(labels)
    for s, e in zip(edges, edges[1:]):
        weight: dict[str, float] = {}
        for k in range(s, e):  # edges strictly increase: never empty
            weight[labels[k]] = weight.get(labels[k], 0.0) + _run_dur([flat[k]])
        first = labels[s]
        best = max(weight, key=lambda lb: (weight[lb], lb == first))
        for k in range(s, e):
            new_labels[k] = best
    out: list[tuple[str, list[dict]]] = []
    for a, lb in zip(flat, new_labels):
        if out and out[-1][0] == lb:
            out[-1][1].append(a)
        else:
            out.append((lb, [a]))
    return out


def _run_gap(left: list[dict], right: list[dict]) -> float:
    """Silence between the end of ``left`` and the start of ``right``.

    ``inf`` when either side has no usable timestamp: an unmeasurable gap must
    never be read as "contiguous" by a caller that merges on contiguity.
    """
    end = left[-1].get("end") if left else None
    start = right[0].get("start") if right else None
    if end is None or start is None:
        return math.inf
    # +epsilon keeps BAD_TAIL_MAX_GAP_S the exclusive bound its comment documents:
    # 1.4 - 1.2 is 0.19999999999999996 in binary floating point, so without it a
    # nominally 0.2s gap slips under the ceiling and merges.
    return max(0.0, float(start) - float(end)) + 1e-9


def _ja_edge_penalties(flat: list[dict], edges: Sequence[int]) -> dict[int, int] | None:
    """Level-2 (UniDic POS) end penalty for each internal atom edge, or ``None``.

    Wired exactly like ``smart_split._attach_end_penalties``: the POS map is keyed
    by the non-space char offset of each token's LAST char, so the edge before
    atom ``e`` is scored at the cumulative offset of atom ``e - 1``.

    ``None`` means no Level-2 source (fugashi absent, or ``VOXWEAVE_JA_POS=0``).
    Unlike line breaking, this pass has no safe Level-1 fallback: the char table
    scores every trailing の/に 2, including 準体の (そうな|の) and the adverbial
    copula (そんな|に), and acting on those merges away a real speaker. So the
    caller must leave the boundary alone rather than degrade. Offsets the tagger
    does not end a token on (BudouX/MeCab disagreement) score 0 for the same
    reason.
    """
    from voxweave.core.kinsoku import ja_pos_end_penalties
    from voxweave.core.layout import _token_char_count

    pos = ja_pos_end_penalties("".join(a["text"] for a in flat), bound_tails_only=True)
    if pos is None:
        return None
    offsets: list[int] = []
    total = 0
    for a in flat:
        total += _token_char_count(a["text"])
        offsets.append(total)
    return {e: pos.get(offsets[e - 1] - 1, 0) for e in edges if 0 < e < len(flat)}


def _merge_bad_tail_runs(
    runs: list[tuple[str, list[dict]]], lang: str
) -> list[tuple[str, list[dict]]]:
    """Merge a speaker boundary that would strand a bound word on the left cue.

    ja only: the gate needs the Level-2 POS signal to tell a dangling 格助詞 from
    a legal clause end, and no validated equivalent grading exists for zh.

    Fires only when all three hold: the boundary's tail scores
    ``BAD_TAIL_PENALTY``, no other internal phrase edge of the cue would score
    below it, and the two runs are separated by less than ``BAD_TAIL_MAX_GAP_S``
    of silence. A cleaner edge means the boundary stays where the speaker signal
    put it (relocating it is a separate concern); an audible gap means the second
    run is a real reply, not a mis-cut phrase. The merged run keeps the longer
    speaker's label, the same duration comparison ``_absorb_tiny_runs`` uses.
    Duration alone never triggers this pass, so the tiny-run policy above is
    untouched.

    Repeats to a fixpoint: merging can join two runs of one speaker, and the
    re-coalesced neighbours form a boundary that was not there before.
    """
    if lang != "ja" or len(runs) < 2:
        return runs
    from voxweave.core.smart_split import _phrase_boundary_atoms

    runs = [(lb, list(ats)) for lb, ats in runs]
    while len(runs) > 1:
        flat = [a for _, ats in runs for a in ats]
        text = "".join(a["text"] for a in flat)
        edges = sorted(
            _phrase_boundary_atoms([{"text": a["text"]} for a in flat], text, lang)
            | {0, len(flat)}
        )
        pen = _ja_edge_penalties(flat, edges)
        if pen is None:
            break
        target = None
        cut = 0
        for i in range(len(runs) - 1):
            cut += len(runs[i][1])
            # An unknown cut is not a phrase edge at all -- treat it as clean and
            # leave it to _snap_runs_to_phrases rather than merging on a guess.
            if pen.get(cut, 0) < BAD_TAIL_PENALTY:
                continue
            if any(p < BAD_TAIL_PENALTY for e, p in pen.items() if e != cut):
                continue
            if _run_gap(runs[i][1], runs[i + 1][1]) >= BAD_TAIL_MAX_GAP_S:
                continue
            target = i
            break
        if target is None:
            break
        left, right = runs[target], runs[target + 1]
        label = right[0] if _run_dur(right[1]) > _run_dur(left[1]) else left[0]
        runs[target] = (label, left[1] + right[1])
        del runs[target + 1]
        runs = _coalesce_runs(runs)
    return runs


def _speaker_runs(
    atoms: list[dict], turns: Sequence[Turn], lang: str
) -> list[tuple[str, list[dict]]]:
    """Group a cue's atoms into consecutive same-speaker runs.

    Atoms without a confident speaker (no span / no overlap) inherit the current
    run; leading unassigned atoms join the first labeled run. Raw runs are then
    de-noised: sub-``MIN_RUN_S`` label thrash is absorbed into the longer
    neighbor, and surviving boundaries snap to jieba/BudouX phrase edges so a
    lexeme is never split across two speaker cues. Finally (ja only) a boundary
    that survives the snap but would leave the left cue ending on a bound word --
    with no cleaner edge available and no audible gap -- is merged away.
    """
    runs: list[tuple[str, list[dict]]] = []
    pending: list[dict] = []  # unassigned atoms before the first labeled one
    for atom in atoms:
        spk = _span_speaker(atom.get("start"), atom.get("end"), turns)
        if spk is None:
            (runs[-1][1] if runs else pending).append(atom)
            continue
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(atom)
        else:
            runs.append((spk, [atom]))
            if pending:
                runs[-1][1][:0] = pending
                pending = []
    if pending:  # no atom got a speaker at all
        return []
    if len(runs) <= 1:
        return runs
    runs = _absorb_tiny_runs(runs)
    runs = _snap_runs_to_phrases(runs, lang)
    return _merge_bad_tail_runs(runs, lang)


def _slice_text_by_runs(text: str, runs: list[tuple[str, list[dict]]]) -> list[str]:
    """Slice the cue's display text into one piece per run.

    Atoms cover exactly the text's non-space characters in order, so each run
    consumes its atoms' character count from the original string (interior
    spacing preserved, boundaries trimmed).
    """
    from voxweave.core.layout import _token_char_count

    pieces: list[str] = []
    i = 0
    for _, atoms in runs:
        need = sum(_token_char_count(a["text"]) for a in atoms)
        j = i
        seen = 0
        while j < len(text) and seen < need:
            if not text[j].isspace():
                seen += 1
            j += 1
        pieces.append(text[i:j].strip())
        i = j
    if i < len(text) and pieces:  # trailing slack (whitespace) sticks to the last piece
        pieces[-1] = (pieces[-1] + text[i:]).strip()
    return pieces


def _run_span(
    atoms: list[dict], fallback_start: float, fallback_end: float
) -> tuple[float, float]:
    starts = [s for a in atoms if (s := a.get("start")) is not None]
    ends = [e for a in atoms if (e := a.get("end")) is not None]
    return (
        float(min(starts)) if starts else fallback_start,
        float(max(ends)) if ends else fallback_end,
    )


def _run_speech_span(atoms: list[dict]) -> tuple[float | None, float | None]:
    """Acoustic anchor of one speaker run: its timed atoms, or nothing.

    Deliberately not :func:`_run_span` -- that falls back to the parent cue's
    DISPLAY bounds, and a run with no timed atom must stay anchorless rather
    than inherit a lag-out pad or a shot lead-in as if it were speech.
    """
    starts = [s for a in atoms if (s := a.get("start")) is not None]
    ends = [e for a in atoms if (e := a.get("end")) is not None]
    return (
        float(min(starts)) if starts else None,
        float(max(ends)) if ends else None,
    )


def format_speaker_cues(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    annotate_speakers: bool = False,
) -> list[Cue]:
    """Speaker-aware post-pass over smart_split's cues (pure, replayable).

    Single-speaker cues pass through. A two-speaker cue becomes one Netflix
    dual-speaker event (``-line\\n-line``, hyphen without space, one speaker per
    line) when the language renders two lines and both halves fit one line;
    otherwise — and for 3+ speakers — the cue splits at the speaker boundaries
    with word-accurate timing. Lyric cues pass through untouched (the music-note
    wrap owns that display).

    ``max_line_length`` and ``max_lines`` are the same per-cue budgets that
    smart_split used for this file (``None`` = the language defaults), so the
    dual gate and the piece re-wrap render for the configured player instead of
    the built-in layout profile.  ``annotate_speakers`` adds transient diarizer
    ids for the display layer; the sibling JSON writer always removes them.
    """
    if not turns:
        return cues
    # Persisted turns may arrive in any order; _span_speaker relies on start
    # order. Pipeline-produced turns are already in this order (a no-op).
    turns = sorted(turns, key=lambda t: (t[0], t[1], t[2]))
    from voxweave.core.layout import (
        _line_budget_width,
        _vis_width,
        default_max_line_length,
        default_max_lines,
        wrap_cue_text,
    )
    from voxweave.core.smart_split import _build_atoms

    effective_max_line_length = (
        default_max_line_length(lang) if max_line_length is None else max_line_length
    )
    effective_max_lines = default_max_lines(lang) if max_lines is None else max_lines
    # Half-width cells -- the unit _vis_width and wrap_cue_text measure in.
    budget = _line_budget_width(effective_max_line_length, lang)
    dual_ok = effective_max_lines >= 2
    out: list[Cue] = []
    for cue in cues:
        word_data = cue.get("word_data") or []
        if cue.get("lyric") or not word_data:
            if annotate_speakers:
                tagged = cast(Cue, dict(cue))
                label = _span_speaker(cue.get("start"), cue.get("end"), turns)
                if label is not None:
                    tagged["speaker_ids"] = [label]
                out.append(tagged)
            else:
                out.append(cue)
            continue
        atoms = _build_atoms(
            cue["text"],
            cast(list, word_data),
            lang,
            max_atom_width=budget,
        )
        runs = _speaker_runs(atoms, turns, lang)
        if len(runs) <= 1:
            if annotate_speakers and runs:
                tagged = cast(Cue, dict(cue))
                tagged["speaker_ids"] = [runs[0][0]]
                out.append(tagged)
            else:
                out.append(cue)
            continue
        pieces = _slice_text_by_runs(cue["text"], runs)
        # Collapse each piece to one logical line before the dual-budget test:
        # smart_split may have soft-wrapped the cue, so a piece can carry an
        # interior "\n" that _vis_width would (wrongly) count as width 1, letting
        # a really-3-line dual event slip past the guard.
        one_line = [" ".join(p.split()) for p in pieces]
        if (
            len(runs) == 2
            and dual_ok
            and all(_vis_width(f"-{t}") <= budget for t in one_line)
        ):
            # Netflix dual-speaker event: one line per speaker, both within budget.
            dual = cast(Cue, dict(cue))
            dual["text"] = f"-{one_line[0]}\n-{one_line[1]}"
            if annotate_speakers:
                dual["speaker_ids"] = [label for label, _atoms in runs]
            out.append(dual)
            continue
        wd_cursor = 0
        for index, ((_label, atoms_run), piece) in enumerate(zip(runs, pieces)):
            # Slice by each run's recorded word_data footprint, not by atom
            # count: a repacked cue stores one entry per atom while a
            # first-generation one stores one per character. Entries the display
            # dropped (punctuation) sit between footprints and go to the run that
            # follows them; the last run takes the remainder, so a trailing
            # dropped entry is kept rather than lost from the stream.
            if index == len(runs) - 1:
                wd_end = len(word_data)
            elif atoms_run:
                wd_end = min(
                    max(wd_cursor, int(atoms_run[-1]["_unit_end"])), len(word_data)
                )
            else:
                wd_end = wd_cursor
            start, end = _run_span(atoms_run, cue["start"], cue["end"])
            part = cast(Cue, dict(cue))
            # Re-wrap the piece for its language (the same layout machinery
            # smart_split uses): an en split re-flows to <=2 clean lines, a zh/ja
            # piece stays one line and never carries a stale "\n".
            part["text"] = wrap_cue_text(
                piece,
                lang,
                effective_max_lines,
                max_line_length=budget,
            )
            part["start"] = start
            part["end"] = end
            part["speech_start"], part["speech_end"] = _run_speech_span(atoms_run)
            part["word_data"] = list(word_data[wd_cursor:wd_end])
            if annotate_speakers:
                part["speaker_ids"] = [_label]
            wd_cursor = wd_end
            out.append(part)
    return out


def _ordered_speaker_format(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    annotate_speakers: bool = False,
) -> list[Cue]:
    """format_speaker_cues + re-sort + overlap trim (splits can abut)."""
    out = format_speaker_cues(
        cues,
        turns,
        lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        annotate_speakers=annotate_speakers,
    )
    out.sort(key=lambda c: (c["start"], c["end"]))
    for prev, nxt in zip(out, out[1:]):
        prev["end"] = min(prev["end"], nxt["start"])
    return out


def apply_speaker_format(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    thresholds: dict | SplitThresholds | None = None,
    max_line_length: int | None = None,
    max_lines: int | None = None,
    annotate_speakers: bool = False,
) -> list[Cue]:
    """Public entry: no-op without turns, otherwise format + keep cue order sane.

    When ``thresholds`` is given (the same gap thresholds smart_split used for this
    file), the formatted cues run through ``timing._cleanup_cues`` so speaker
    splits/dashes get the same timing polish as ordinary cues (short pieces extend
    into the following gap, sub-0.5s gaps chain) and never render as sub-flash
    cues. ``_cleanup_cues`` is timing-only and never merges content, so distinct
    speakers stay separate cues, and it is idempotent for cues carrying timed
    ``word_data``, so this second pass cannot stack another pad onto an already
    padded cue. ``thresholds=None`` keeps the pre-polish behavior for
    replay/back-compat callers. ``max_line_length`` and ``max_lines`` mirror
    smart_split's per-cue budgets (see :func:`format_speaker_cues`).
    """
    if not turns:
        return cues
    out = _ordered_speaker_format(
        cues,
        turns,
        lang,
        max_line_length=max_line_length,
        max_lines=max_lines,
        annotate_speakers=annotate_speakers,
    )
    if thresholds is not None:
        from voxweave.core.smart_split import SplitThresholds
        from voxweave.core.timing import _cleanup_cues

        th = (
            SplitThresholds.from_mapping(thresholds)
            if isinstance(thresholds, dict)
            else thresholds
        )
        out = _cleanup_cues(
            out,
            min_cue_s=th.min_cue_s,
            max_cue_s=th.max_cue_s,
            cps=th.cps,
            lag_out_s=th.lag_out_s,
        )
    return out
