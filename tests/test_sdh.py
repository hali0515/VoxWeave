# tests/test_sdh.py
# SDH event tags: window probabilities -> merged labeled spans (pure, no GPU),
# trimming to speech-free gaps (dialogue always wins), and the merged sidecar
# VTT rendering ([label] cues interleaved, never italicized).
import numpy as np

from voxweave.sdh import (
    EVENT_MIN_PROB,
    events_from_scores,
    fit_events_to_gaps,
    render_sdh_vtt,
)

LABELS = ["Speech", "Explosion", "Knock", "Music"]


def _probs(rows):
    return np.array(rows, dtype=np.float32)


def test_events_from_scores_merges_adjacent_windows():
    # windows at 0,1,2s; explosion fires in windows 0-1, knock in window 2
    probs = _probs(
        [
            [0.9, 0.8, 0.0, 0.0],
            [0.9, 0.7, 0.0, 0.0],
            [0.1, 0.0, 0.9, 0.0],
        ]
    )
    events = events_from_scores(probs, LABELS, [0.0, 1.0, 2.0], win_sec=2.0)
    assert events == [(0.0, 3.0, "explosion"), (2.0, 4.0, "knocking")]


def test_events_below_threshold_ignored():
    probs = _probs([[0.9, EVENT_MIN_PROB - 0.01, 0.0, 0.0]])
    assert events_from_scores(probs, LABELS, [0.0]) == []


def test_unmapped_classes_never_emit():
    # Speech and Music score high but are not in the curated tag map
    probs = _probs([[0.99, 0.0, 0.0, 0.99]])
    assert events_from_scores(probs, LABELS, [0.0]) == []


def test_fit_events_trims_to_gaps():
    cues = [{"start": 0.0, "end": 5.0}, {"start": 8.0, "end": 12.0}]
    # event spans the first cue and the 5-8s gap: the gap piece survives
    out = fit_events_to_gaps([(3.0, 8.0, "explosion")], cues)
    assert out == [(5.0, 8.0, "explosion")]


def test_fit_events_drops_fully_covered_and_tiny():
    cues = [{"start": 0.0, "end": 5.0}, {"start": 5.2, "end": 9.0}]
    # fully under dialogue -> dropped; 0.2s shard -> under the 5/6s floor
    assert fit_events_to_gaps([(1.0, 4.0, "siren"), (4.9, 5.3, "knocking")], cues) == []


def test_fit_events_caps_duration():
    out = fit_events_to_gaps([(0.0, 30.0, "rain")], [])
    assert out == [(0.0, 7.0, "rain")]


def test_render_sdh_vtt_interleaves_and_keeps_lyrics():
    cues = [
        {"text": "hello", "start": 0.0, "end": 2.0},
        {"text": "la la", "start": 10.0, "end": 12.0, "lyric": True},
    ]
    out = render_sdh_vtt(cues, [(4.0, 6.0, "explosion")])
    # time order: dialogue, event, lyric; event label plain (no italics markup)
    assert out.index("hello") < out.index("[explosion]") < out.index("♪ la la ♪")
    assert "<i>[" not in out
