"""SDH audio-event tags from PANNs (opt-in ``--sdh``).

Detects plot-relevant non-speech sounds (explosions, gunfire, sirens, knocks,
...) on the ORIGINAL audio mix -- vocal separation strips sound effects, so the
songdet vocals stem is the wrong input here -- and renders them as Netflix-style
``[lowercase label]`` cues in the speech-free gaps of the subtitle track.

Output is a sidecar ``<stem>.sdh.vtt`` (dialogue + event tags merged); the main
VTT/JSON contract is untouched, so ``align``/``split``/``translate`` never see
event cues. Detection reuses the songdet PANNs singleton and window pass.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("voxweave")

# AudioSet display name -> SDH label (lowercase per Netflix SDH convention).
# Curated to plot-pertinent, high-precision classes; broad/ambient classes
# (Music, Speech, Animal, ...) are deliberately absent.
TAG_BY_NAME: dict[str, str] = {
    "Explosion": "explosion",
    "Gunshot, gunfire": "gunfire",
    "Machine gun": "gunfire",
    "Fireworks": "fireworks",
    "Siren": "siren",
    "Civil defense siren": "siren",
    "Police car (siren)": "siren",
    "Ambulance (siren)": "siren",
    "Fire engine, fire truck (siren)": "siren",
    "Alarm": "alarm",
    "Alarm clock": "alarm ringing",
    "Smoke detector, smoke alarm": "alarm blaring",
    "Car alarm": "car alarm",
    "Telephone bell ringing": "phone ringing",
    "Ringtone": "phone ringing",
    "Doorbell": "doorbell rings",
    "Ding-dong": "doorbell rings",
    "Knock": "knocking",
    "Bark": "dog barking",
    "Howl": "dog howling",
    "Thunder": "thunder rumbling",
    "Thunderstorm": "thunder rumbling",
    "Applause": "applause",
    "Crying, sobbing": "sobbing",
    "Screaming": "screaming",
    "Shatter": "glass shattering",
    "Helicopter": "helicopter whirring",
    "Train horn": "train horn blaring",
    "Air horn, truck horn": "horn blaring",
}

# A window must score at least this on a mapped class to register an event.
EVENT_MIN_PROB = 0.5
# Netflix duration bounds: events shorter than the 5/6s floor are padded, and
# nothing runs past 7s (long ambience reads fine from one tag).
MIN_EVENT_SEC = 5.0 / 6.0
MAX_EVENT_SEC = 7.0


def events_from_scores(
    probs: np.ndarray,
    labels: Sequence[str],
    starts_sec: Sequence[float],
    *,
    threshold: float = EVENT_MIN_PROB,
    win_sec: float = 2.0,
) -> list[tuple[float, float, str]]:
    """Window probabilities -> merged ``(start, end, label)`` event spans.

    Per window, the highest-scoring mapped class above ``threshold`` wins;
    adjacent/overlapping windows with the same label merge into one span.
    Pure function (tests drive it with synthetic matrices, no GPU).
    """
    idx_tag = [(i, TAG_BY_NAME[n]) for i, n in enumerate(labels) if n in TAG_BY_NAME]
    merged: list[list[Any]] = []  # [start, end, tag]
    for w, t0 in enumerate(starts_sec):
        best: tuple[float, str] | None = None
        for i, tag in idx_tag:
            p = float(probs[w, i])
            if p >= threshold and (best is None or p > best[0]):
                best = (p, tag)
        if best is None:
            continue
        t1 = t0 + win_sec
        if merged and merged[-1][2] == best[1] and t0 <= merged[-1][1] + 1e-6:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1, best[1]])
    return [(float(a), float(b), tag) for a, b, tag in merged]


def fit_events_to_gaps(
    events: list[tuple[float, float, str]],
    cues: Sequence[Mapping[str, Any]],
    *,
    min_sec: float = MIN_EVENT_SEC,
    max_sec: float = MAX_EVENT_SEC,
) -> list[tuple[float, float, str]]:
    """Trim events to the speech-free gaps between dialogue cues.

    Dialogue always wins the screen: each event span has every cue interval
    subtracted; the longest remaining piece survives if it still clears
    ``min_sec`` (then padded to it when the gap allows nothing longer is
    needed -- pieces are real time, never padded into a cue), capped at
    ``max_sec``.
    """
    spans = sorted(
        (float(c["start"]), float(c["end"]))
        for c in cues
        if c.get("start") is not None and c.get("end") is not None
    )
    out: list[tuple[float, float, str]] = []
    for a, b, tag in events:
        pieces: list[tuple[float, float]] = []
        cur = a
        for s, e in spans:
            if e <= cur:
                continue
            if s >= b:
                break
            if s > cur:
                pieces.append((cur, min(s, b)))
            cur = max(cur, e)
            if cur >= b:
                break
        if cur < b:
            pieces.append((cur, b))
        if not pieces:
            continue
        ps, pe = max(pieces, key=lambda p: p[1] - p[0])
        if pe - ps < min_sec:
            continue
        out.append((ps, min(pe, ps + max_sec), tag))
    return sorted(out)


def detect_events(
    wav32k: Path, *, progress=None, threshold: float = EVENT_MIN_PROB
) -> list[tuple[float, float, str]]:
    """Run PANNs over the 32 kHz ORIGINAL mix and return merged event spans."""
    from panns_inference import labels as panns_labels

    from voxweave import songdet

    wp = songdet.window_probs(wav32k, progress=progress)
    if wp is None:
        return []
    probs, starts_sec = wp
    events = events_from_scores(
        probs, panns_labels, starts_sec, threshold=threshold, win_sec=songdet.WIN_SEC
    )
    log.info("SDH: %d raw event span(s)", len(events))
    return events


def render_sdh_vtt(
    cues: Sequence[Mapping[str, Any]],
    events: list[tuple[float, float, str]],
) -> str:
    """Merge dialogue cues and ``[label]`` event cues into one VTT, time-ordered.

    Lyric-flagged dialogue keeps its music-note wrap; event labels are never
    italicized (Netflix SDH rule).
    """
    from voxweave.pipeline import lyric_display_text
    from voxweave.realign import render_cues

    rows = [(c.get("start"), c.get("end"), lyric_display_text(c)) for c in cues]
    rows.extend((a, b, f"[{tag}]") for a, b, tag in events)
    rows.sort(key=lambda r: (r[0] is None, r[0] if r[0] is not None else 0.0))
    return render_cues(rows)
