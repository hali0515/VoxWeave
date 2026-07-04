#!/usr/bin/env python
"""Run the GPU pipeline once and capture the song-skip decision artifacts as tests/scenarios/<name>.json.

Usage:
    python scripts/capture_scenario.py <media> <name> [--desc "..."] [--lang en]

Captures GPU-stage outputs (PANNs per-window scores + separated-vocals VAD segments).
tests/test_scenarios.py can then replay these with pure functions (songdet +
pipeline.plan_song_skip) and assert against them -- **zero GPU required**.
The fixture is pre-populated with the current (correct) behavior as a golden snapshot;
manually fill in ``assert.speech_present_at`` with timestamps where speech should be
present (regression anchors).

Only captures what is needed for the song-skip decision; for smart_split / reinject
scenarios, save ASR units separately (--with-units flag, not shown here).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxweave import backend
from voxweave.chunking import decode_to_wav, silence_gaps, vad_speech_segments
from voxweave import songdet
from voxweave.pipeline import (
    plan_song_skip,
    ASR_LOUDNORM,
    MIN_SONG_SKIP_SEC,
    MAX_CHUNK_SEC,
    SONG_FINE_SILENCE_MS,
    cache_vocals_path,
)

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("name")
    ap.add_argument("--desc", default="")
    ap.add_argument("--lang", default="")
    ap.add_argument(
        "--no-separate",
        action="store_true",
        help="use original audio (default: separate vocals, route ii)",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="apply the pipeline's loudnorm filter to the VAD input (mirror `normalize=true` runs;"
        " loudnorm changes VAD segment boundaries and thus voiced-block structure — a fixture"
        " meant to replay a normalized run MUST capture with this flag). PANNs scoring input"
        " stays unfiltered, exactly like the pipeline.",
    )
    args = ap.parse_args()

    media = Path(args.media)
    af = ASR_LOUDNORM if args.normalize else None
    if args.no_separate:
        print(f"[capture] decode (no separation): {media.name}")
        voc16 = decode_to_wav(media, audio_filter=af)
        voc32 = decode_to_wav(media, sample_rate=songdet.SR)
    else:
        cache = cache_vocals_path(media)
        if cache.exists():
            print(f"[capture] reuse cached vocals: {cache}")
            voc16 = decode_to_wav(cache, audio_filter=af)
            voc32 = cache
        else:
            print(f"[capture] decode + separate: {media.name}")
            fb = decode_to_wav(media, sample_rate=44100, mono=False)
            voc = backend.separate_vocals(fb)
            voc16 = decode_to_wav(voc, audio_filter=af)
            voc32 = decode_to_wav(voc, sample_rate=songdet.SR)

    # PANNs per-window scoring (mirrors the internals of detect_song_spans)
    data, sr = sf.read(str(voc32), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    win, hop = int(songdet.WIN_SEC * sr), int(songdet.HOP_SEC * sr)
    starts_idx = list(range(0, len(data) - win + 1, hop))
    wins = np.stack([data[s : s + win] for s in starts_idx])
    model = songdet._get_model()
    probs = np.concatenate(
        [model.inference(wins[i : i + 32])[0] for i in range(0, len(wins), 32)]
    )
    speech, sing, music = songdet.reduce_scores(probs)
    t = [round(s / sr, 2) for s in starts_idx]

    segs = vad_speech_segments(voc16)
    vad_segs = [[round(s["start"], 3), round(s["end"], 3)] for s in segs]
    # Fine VAD silences: excision snaps its cut points into these (mirrors pipeline)
    fine = vad_speech_segments(voc16, min_silence_ms=SONG_FINE_SILENCE_MS)
    silences = [[round(a, 3), round(b, 3)] for a, b in silence_gaps(fine)]

    # Current (correct) behavior -> golden snapshot
    song = songdet.merge_spans(songdet.song_flags(probs), t)
    sing_starts = [tt for tt, f in zip(t, songdet.sing_flags(probs)) if f]
    sing_spans = [(a, b) for (a, b) in song if any(a <= x < b for x in sing_starts)]
    speech_spans = songdet.merge_spans(songdet.speech_flags(probs), t)
    _, final, kept, chunks = plan_song_skip(
        song,
        sing_spans,
        segs,
        speech_spans=speech_spans,
        silences=[(a, b) for a, b in silences],
        min_skip_sec=MIN_SONG_SKIP_SEC,
        max_chunk_sec=MAX_CHUNK_SEC,
    )

    fixture = {
        "name": args.name,
        "desc": args.desc,
        "lang": args.lang,
        "normalize": args.normalize,
        "win_sec": songdet.WIN_SEC,
        "hop_sec": songdet.HOP_SEC,
        "scores": {
            "t": t,
            "speech": [round(float(x), 4) for x in speech],
            "sing": [round(float(x), 4) for x in sing],
            "music": [round(float(x), 4) for x in music],
        },
        "vad_segs": vad_segs,
        "silences": silences,
        "assert": {
            "expected_song_spans": [[round(a, 1), round(b, 1)] for a, b in final],
            "speech_present_at": [],  # fill in manually: timestamps where speech should be present (regression anchors)
            "max_chunk_sec": MAX_CHUNK_SEC,
        },
    }
    out = (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "scenarios"
        / f"{args.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[capture] wrote {out}")
    print(
        f"  raw song={[(round(a, 1), round(b, 1)) for a, b in song]}  final={fixture['assert']['expected_song_spans']}"
    )
    print(
        f"  vad_segs={len(vad_segs)}  chunks={[(round(c['start'], 1), round(c['end'], 1)) for c in chunks]}"
    )
    print(
        "  -> now manually fill in assert.speech_present_at with timestamps where speech should be present"
    )


if __name__ == "__main__":
    main()
