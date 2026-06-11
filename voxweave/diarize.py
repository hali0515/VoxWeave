"""Speaker diarization (pyannote) + Netflix speaker-aware cue formatting.

Detection runs once per file on the separated-vocals 16k wav (clean speech beats
the BGM mix for speaker embeddings) and persists ``speaker_turns`` to the
sibling JSON. Formatting is a pure post-pass over smart_split's cues: each
cue's atoms get a speaker by time overlap with the turns; a cue containing two
speakers becomes a Netflix dual-speaker event (one line per speaker, leading
hyphen, no space) when the language allows two lines and both halves fit one
line, otherwise the cue splits at the speaker boundaries. ``split`` replays
formatting from the persisted turns without re-running pyannote.

Model: ``pyannote/speaker-diarization-3.1`` (pyannote.audio 3.x; MIT, ~1.6 GB
VRAM). The checkpoint is HF-gated: accept the conditions on the model card and
provide a token (VOXWEAVE_HF_TOKEN / HF_TOKEN env, or ``hf_token`` in the conf).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence, cast

from voxweave import config
from voxweave.core.schema import Cue

log = logging.getLogger("voxweave")

DIARIZE_MODEL = os.environ.get(
    "VOXWEAVE_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"
)

# Atom-level speaker assignment needs at least this much absolute overlap with a
# turn (seconds); below it the atom inherits its neighbors (guards 20ms grazes).
MIN_ATOM_OVERLAP_S = 0.05

Turn = tuple[float, float, str]

_pipeline = None  # pyannote Pipeline singleton -- lazy-loaded, released after use


def _get_pipeline(token: str):
    global _pipeline
    if _pipeline is None:
        try:
            import torch
            from pyannote.audio import (  # pyright: ignore[reportMissingImports]
                Pipeline,
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "diarization requires pyannote.audio (not installed); "
                "install with: pip install 'voxweave[diarize]'"
            ) from e
        pl = Pipeline.from_pretrained(
            DIARIZE_MODEL, use_auth_token=token, cache_dir=config.AUDIO_CACHE
        )
        if pl is None:
            raise RuntimeError(
                f"could not load {DIARIZE_MODEL}: accept the user conditions on "
                f"https://hf.co/{DIARIZE_MODEL} (and its segmentation model) and "
                "set VOXWEAVE_HF_TOKEN / HF_TOKEN"
            )
        if torch.cuda.is_available():
            pl.to(torch.device("cuda"))
        _pipeline = pl
        log.info("loaded diarization pipeline %s", DIARIZE_MODEL)
    return _pipeline


def release() -> None:
    """Drop the pipeline singleton and free its VRAM (mirrors backend.release)."""
    global _pipeline
    if _pipeline is None:
        return
    _pipeline = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def diarize_turns(
    wav_path: Path,
    *,
    token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Run diarization over ``wav_path`` and return sorted ``(start, end, label)`` turns."""
    token = token or config.conf_hf_token()
    if not token:
        raise RuntimeError(
            f"diarization needs a Hugging Face token for the gated {DIARIZE_MODEL} "
            "checkpoint: accept the conditions on its model card, then set "
            "VOXWEAVE_HF_TOKEN / HF_TOKEN (or hf_token in ~/.config/voxweave.conf)"
        )
    pl = _get_pipeline(token)
    kwargs: dict[str, int] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    annotation = pl(str(wav_path), **kwargs)
    turns = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort()
    log.info(
        "diarization: %d turn(s), %d speaker(s)",
        len(turns),
        len({lb for _, _, lb in turns}),
    )
    return turns


def _span_speaker(
    start: float | None, end: float | None, turns: Sequence[Turn]
) -> str | None:
    """Dominant speaker for a time span by accumulated overlap (whisperX pattern)."""
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


def _speaker_runs(
    atoms: list[dict], turns: Sequence[Turn]
) -> list[tuple[str, list[dict]]]:
    """Group a cue's atoms into consecutive same-speaker runs.

    Atoms without a confident speaker (no span / no overlap) inherit the current
    run; leading unassigned atoms join the first labeled run.
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
    return runs


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


def format_speaker_cues(
    cues: List[Cue], turns: Sequence[Turn] | None, lang: str
) -> List[Cue]:
    """Speaker-aware post-pass over smart_split's cues (pure, replayable).

    Single-speaker cues pass through. A two-speaker cue becomes one Netflix
    dual-speaker event (``-line\\n-line``, hyphen without space, one speaker per
    line) when the language renders two lines and both halves fit one line;
    otherwise — and for 3+ speakers — the cue splits at the speaker boundaries
    with word-accurate timing. Lyric cues pass through untouched (the music-note
    wrap owns that display).
    """
    if not turns:
        return cues
    from voxweave.core.layout import (
        DEFAULT_MAX_LINE_LENGTH,
        _vis_width,
        default_max_lines,
    )
    from voxweave.core.smart_split import _build_atoms

    dual_ok = default_max_lines(lang) >= 2
    out: List[Cue] = []
    for cue in cues:
        word_data = cue.get("word_data") or []
        if cue.get("lyric") or not word_data:
            out.append(cue)
            continue
        atoms = _build_atoms(cue["text"], cast(list, word_data), lang)
        runs = _speaker_runs(atoms, turns)
        if len(runs) <= 1:
            out.append(cue)
            continue
        pieces = _slice_text_by_runs(cue["text"], runs)
        if (
            len(runs) == 2
            and dual_ok
            and all(_vis_width(f"-{p}") <= DEFAULT_MAX_LINE_LENGTH for p in pieces)
        ):
            dual = cast(Cue, dict(cue))
            dual["text"] = f"-{pieces[0]}\n-{pieces[1]}"
            out.append(dual)
            continue
        wd_cursor = 0
        for (_label, atoms_run), piece in zip(runs, pieces):
            n = len(atoms_run)
            start, end = _run_span(atoms_run, cue["start"], cue["end"])
            part = cast(Cue, dict(cue))
            part["text"] = piece
            part["start"] = start
            part["end"] = end
            part["word_data"] = list(word_data[wd_cursor : wd_cursor + n])
            wd_cursor += n
            out.append(part)
    return out


def _ordered_speaker_format(
    cues: List[Cue], turns: Sequence[Turn] | None, lang: str
) -> List[Cue]:
    """format_speaker_cues + re-sort + overlap trim (splits can abut)."""
    out = format_speaker_cues(cues, turns, lang)
    out.sort(key=lambda c: (c["start"], c["end"]))
    for prev, nxt in zip(out, out[1:]):
        if prev["end"] > nxt["start"]:
            prev["end"] = nxt["start"]
    return out


def apply_speaker_format(
    cues: List[Cue], turns: Sequence[Turn] | None, lang: str
) -> List[Cue]:
    """Public entry: no-op without turns, otherwise format + keep cue order sane."""
    if not turns:
        return cues
    return _ordered_speaker_format(cues, turns, lang)
