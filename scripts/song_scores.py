"""Print PANNs speech/sing/music window scores for a time span of a media file.

Diagnoses song-skip misses: shows, per 2s window, the three scores against the
song_flags criterion -- (sing > speech AND sing > SING_MIN) OR
(music > MUSIC_MIN AND speech < SPEECH_MAX) -- so a missed window reveals which
condition of each branch failed (e.g. sung vocals scoring speech-like).

Usage:
    uv run --extra cuda python scripts/song_scores.py <media> <start_sec> <end_sec>

Prefers the run's separated-vocals entry in the per-media VoxWeave artifact
cache (or an existing legacy media-adjacent cache) -- scores are only meaningful
on separated vocals (route ii); falls back to the raw media with a warning.
Read-only: never creates cache state. The slice start is snapped down to the
pipeline's HOP_SEC window grid so the windows match the ones song detection saw.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from voxweave import artifacts, songdet
from voxweave.pipeline import CACHE_DIRNAME


def _vocals_source(media: Path) -> Path | None:
    """Existing separated-vocals cache for ``media`` (legacy first), without claiming one."""
    legacy = media.parent / CACHE_DIRNAME / f"{media.stem}.vocals.32k.flac"
    managed = artifacts.inspect_paths(media)
    for candidate in (legacy, managed.vocals_cache if managed else None):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _slice_32k(src: Path, start: float, end: float) -> Path:
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)  # ffmpeg writes by path
    out = Path(name)
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
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
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed slicing {src} (rc={proc.returncode}):\n{proc.stderr.strip()}"
        )
    return out


def _verdict(sp: float, si: float, mu: float, flagged: bool) -> str:
    """SONG, or which condition of each song_flags branch failed for a missed window."""
    if flagged:
        return "SONG"
    sing_fail = [
        why
        for why, failed in (
            ("sing<=speech", si <= sp),
            ("sing<=SING_MIN", si <= songdet.SING_MIN),
        )
        if failed
    ]
    music_fail = [
        why
        for why, failed in (
            ("music<=MUSIC_MIN", mu <= songdet.MUSIC_MIN),
            ("speech>=SPEECH_MAX", sp >= songdet.SPEECH_MAX),
        )
        if failed
    ]
    return f"miss  sing:[{','.join(sing_fail)}] music:[{','.join(music_fail)}]"


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    media, start, end = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    # Song detection windows the whole file from t=0 in HOP_SEC steps; slicing on
    # that grid reproduces its windows instead of scoring off-grid neighbours.
    start = math.floor(start / songdet.HOP_SEC) * songdet.HOP_SEC
    src = _vocals_source(media)
    if src is None:
        src = media
        print(
            "WARNING: no vocals cache found -- scoring the raw mix; "
            "thresholds are tuned for separated vocals, expect inflated music scores"
        )
    print(f"source: {src}")
    print(
        f"slice: {start:.1f}-{end:.1f}s (start snapped to the {songdet.HOP_SEC}s grid)"
    )
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
        f"song if: (sing > speech AND sing > {songdet.SING_MIN}) "
        f"OR (music > {songdet.MUSIC_MIN} AND speech < {songdet.SPEECH_MAX})\n"
        f"{'t':>7}  {'speech':>6}  {'sing':>6}  {'music':>6}  verdict"
    )
    for t, sp, si, mu, f in zip(starts, speech, sing, music, flags, strict=True):
        verdict = _verdict(sp, si, mu, bool(f))
        print(f"{start + t:7.1f}  {sp:6.2f}  {si:6.2f}  {mu:6.2f}  {verdict}")
    spans = songdet.merge_spans(flags, starts)
    print(
        f"\nmerged song spans within slice (>={songdet.MIN_SPAN_SEC}s): "
        + (", ".join(f"{start + a:.1f}-{start + b:.1f}" for a, b in spans) or "none")
    )


if __name__ == "__main__":
    main()
