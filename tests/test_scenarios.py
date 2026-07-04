"""Replay tests for the song-skip decision chain using real clips (zero GPU).

tests/scenarios/*.json are captured by scripts/capture_scenario.py (PANNs per-window scores + VAD segments).
These tests replay the full decision chain with pure functions (songdet + pipeline.plan_song_skip) and assert:
  - final song spans == golden snapshot (expected_song_spans)
  - every timestamp in assert.speech_present_at falls within a retained VAD segment (regression anchor: speech that should have subtitles must not be swallowed)
  - no chunk exceeds max_chunk_sec (both too-short and too-long chunks hurt ASR recall)

To add a new scenario: run `python scripts/capture_scenario.py <clip> <name>`, then manually fill in
speech_present_at in the generated json.  Afterwards `pytest tests/test_scenarios.py` guards it permanently
without needing to re-run the full clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voxweave import songdet
from voxweave.pipeline import MAX_CHUNK_SEC, MIN_SONG_SKIP_SEC, plan_song_skip

SCENARIO_DIR = Path(__file__).parent / "scenarios"
SCENARIOS = sorted(SCENARIO_DIR.glob("*.json")) if SCENARIO_DIR.exists() else []


def _replay(fx: dict):
    """fixture -> (final_song_spans, kept_segs, chunks), replaying the same pure decision chain as pipeline."""
    sc = fx["scores"]
    speech = np.array(sc["speech"], dtype="float32")
    sing = np.array(sc["sing"], dtype="float32")
    music = np.array(sc["music"], dtype="float32")
    t = sc["t"]
    song = songdet.merge_spans(songdet.song_flags_from_scores(speech, sing, music), t)
    sing_fl = songdet.sing_flags_from_scores(speech, sing, music)
    sing_starts = [tt for tt, f in zip(t, sing_fl) if f]
    sing_spans = [(a, b) for (a, b) in song if any(a <= x < b for x in sing_starts)]
    speech_spans = songdet.merge_spans(
        songdet.speech_flags_from_scores(speech, sing, music), t
    )
    segs = [{"start": a, "end": b} for a, b in fx["vad_segs"]]
    # silences captured by newer fixtures; older ones replay without snapping (None)
    silences = [(a, b) for a, b in fx["silences"]] if fx.get("silences") else None
    _, final, kept, chunks = plan_song_skip(
        song,
        sing_spans,
        segs,
        speech_spans=speech_spans,
        silences=silences,
        min_skip_sec=MIN_SONG_SKIP_SEC,
        max_chunk_sec=MAX_CHUNK_SEC,
    )
    return final, kept, chunks


def _covered(t: float, segs: list[dict]) -> bool:
    return any(s["start"] <= t <= s["end"] for s in segs)


@pytest.mark.skipif(not SCENARIOS, reason="no scenario fixtures")
@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_song_skip(path: Path):
    fx = json.loads(path.read_text(encoding="utf-8"))
    a = fx.get("assert", {})
    final, kept, chunks = _replay(fx)

    # 1) final song spans == golden snapshot (logic changes surface here; re-run capture intentionally)
    got = [[round(x, 1), round(y, 1)] for x, y in final]
    assert got == a.get("expected_song_spans", got), (
        f"{path.stem}: song spans changed {got} != {a.get('expected_song_spans')}"
    )

    # 2) every timestamp that should have speech must fall within a retained VAD segment (regression anchor: must not be swallowed by song-skip)
    for tp in a.get("speech_present_at", []):
        assert _covered(tp, kept), (
            f"{path.stem}: speech at t={tp}s was swallowed by song-skip (should be retained)"
        )

    # 3) no oversized chunks (both too-short and too-long chunks hurt ASR recall)
    cap = a.get("max_chunk_sec", MAX_CHUNK_SEC)
    for c in chunks:
        assert c["end"] - c["start"] <= cap + 1e-6, (
            f"{path.stem}: chunk {c['start']:.1f}-{c['end']:.1f} exceeds {cap}s"
        )


# --------------------------------------------------------------------------- #
# Synthetic decision-chain tests (no fixture): short-hum excision inside a dialogue block
# --------------------------------------------------------------------------- #
def test_short_hum_excised_within_mixed_segment():
    # "speech, brief pause, a hummed bar, speech again" inside ONE VAD segment.
    # The hum (5s singing span, < min_skip_sec) must not absorb the block, must not be
    # filtered away, and must be cut out of the segment with the dialogue kept.
    segs = [
        {"start": 95.0, "end": 99.0},  # earlier dialogue, same voiced block
        {
            "start": 100.0,
            "end": 130.0,
        },  # mixed: speech 100-112, hum 112-117, speech 117-130
    ]
    sing_spans = [(112.0, 117.0)]
    silences = [(111.7, 112.1), (117.2, 117.6)]  # fine-VAD pauses around the hum
    _, final, kept, chunks = plan_song_skip(
        [(112.0, 117.0)],
        sing_spans,
        segs,
        speech_spans=[(100.0, 111.0), (118.0, 130.0)],
        silences=silences,
        min_skip_sec=MIN_SONG_SKIP_SEC,
        max_chunk_sec=MAX_CHUNK_SEC,
    )
    # span start 112.0 already sits inside a silence (kept as-is); span end 117.0 is in
    # speech and snaps to the next silence midpoint 117.4
    assert final == [(112.0, 117.4)]
    assert kept == [
        {"start": 95.0, "end": 99.0},
        {"start": 100.0, "end": 112.0},  # dialogue before the hum survives
        {"start": 117.4, "end": 130.0},  # dialogue after the hum survives
    ]
    # chunks must not bridge the excised hum (slice_wav cuts contiguously)
    for c in chunks:
        assert not (c["start"] < 112.0 and c["end"] > 117.4)


def test_short_instrumental_span_still_kept_as_content():
    # Cecilia guard: a brief pure-instrumental span (not in sing_spans) is still
    # filtered out entirely — transcribed as content, no excision, no block split.
    segs = [{"start": 148.5, "end": 156.3}]
    _, final, kept, chunks = plan_song_skip(
        [(148.0, 151.0)],
        [],  # no singing
        segs,
        speech_spans=[(153.0, 156.0)],
        silences=[],
        min_skip_sec=MIN_SONG_SKIP_SEC,
        max_chunk_sec=MAX_CHUNK_SEC,
    )
    assert final == []
    assert kept == segs
    assert chunks == [{"start": 148.5, "end": 156.3, "offset": 148.5}]


def test_expected_fixtures_present():
    # Guard against silent fixture loss (untracked file, bad glob): every named replay
    # guard must be discovered, or its regression protection quietly disappears.
    names = {p.stem for p in SCENARIOS}
    assert names >= {
        "cecilia-bgm-debut",
        "isekai-ed-overeat",
        "isekai-op-sting",
        "meido-head-rescue",
        "yofukashi-rap-op",
    }, f"missing scenario fixtures: {names}"
