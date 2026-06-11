"""Print PANNs speech/sing/music window scores for a time span of a media file.

Diagnoses song-skip misses: shows, per 2s window, the three scores against the
song_flags gates (sing >= SING_MIN or music >= MUSIC_MIN, AND speech < SPEECH_MAX)
so a missed span reveals WHICH gate failed (e.g. sung vocals scoring speech-like).

Usage:
    uv run python scripts/song_scores.py <media> <start_sec> <end_sec>

Prefers the run's separated-vocals cache (cache/<stem>.vocals.32k.flac) next to
the media -- scores are only meaningful on separated vocals (route ii); falls
back to the raw media with a warning.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from voxweave import songdet
from voxweave.pipeline import cache_vocals_path


def _slice_32k(src: Path, start: float, end: float) -> Path:
    out = Path(tempfile.mkstemp(suffix=".wav")[1])
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(songdet.SR),
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    media, start, end = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    src = cache_vocals_path(media)
    if not src.exists():
        src = media
        print(
            "WARNING: no vocals cache found -- scoring the raw mix; "
            "thresholds are tuned for separated vocals, expect inflated music scores"
        )
    print(f"source: {src}")
    wav = _slice_32k(src, start, end)
    try:
        wp = songdet.window_probs(wav)
    finally:
        wav.unlink(missing_ok=True)
    if wp is None:
        sys.exit(f"span too short for one {songdet.WIN_SEC}s window")
    probs, starts = wp
    speech, sing, music = songdet.reduce_scores(probs)
    flags = songdet.song_flags(probs)
    print(
        f"gates: (sing >= {songdet.SING_MIN} OR music >= {songdet.MUSIC_MIN}) "
        f"AND speech < {songdet.SPEECH_MAX}\n"
        f"{'t':>7}  {'speech':>6}  {'sing':>6}  {'music':>6}  verdict"
    )
    for t, sp, si, mu, f in zip(starts, speech, sing, music, flags, strict=True):
        gate = (
            "SONG"
            if f
            else (
                "miss:speech-gate"
                if (si >= songdet.SING_MIN or mu >= songdet.MUSIC_MIN)
                else "miss:level-gate"
            )
        )
        print(f"{start + t:7.1f}  {sp:6.2f}  {si:6.2f}  {mu:6.2f}  {gate}")
    spans = songdet.merge_spans(flags, starts)
    print(
        f"\nmerged song spans within slice (>={songdet.MIN_SPAN_SEC}s): "
        + (", ".join(f"{start + a:.1f}-{start + b:.1f}" for a, b in spans) or "none")
    )


if __name__ == "__main__":
    main()
