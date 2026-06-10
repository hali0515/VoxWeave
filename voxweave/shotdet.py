"""Shot-change detection via ffmpeg scene scores.

Cue boundaries that land just off a hard cut flash across it — the classic
amateur-subtitle tell. ``detect_shot_changes`` decodes the video at reduced
resolution through ffmpeg's scene-score select filter and returns sorted cut
timestamps; smart_split's snap pass then nudges nearby cue boundaries onto
them. Returns ``None`` when the media has no video stream or ffmpeg fails,
so audio-only pipelines skip snapping transparently.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# select gt(scene,t): ffmpeg docs call 0.3-0.5 reasonable; anime hard cuts score
# high, and the snap pass is conservative, so favor recall.
SCENE_THRESHOLD = 0.3
# Decode at this width for scene scoring: scene scores are stable under scaling
# and a full-res decode of movie-length media would dominate pipeline runtime.
_SCALE_WIDTH = 320
_PTS_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def _scene_threshold() -> float:
    raw = os.environ.get("VOXWEAVE_SHOT_SCENE", "")
    try:
        return float(raw) if raw.strip() else SCENE_THRESHOLD
    except ValueError:
        return SCENE_THRESHOLD


def detect_shot_changes(
    media: Path,
    threshold: float | None = None,
    timeout_s: int = 3600,
) -> list[float] | None:
    """Return sorted shot-change timestamps (seconds) for ``media``'s first video
    stream, or ``None`` when undetectable (no video stream, no ffmpeg, timeout).

    One ffmpeg pass: downscale -> scene-score select -> showinfo; cut times are
    parsed from showinfo's pts_time lines on stderr. ``-nostdin`` + DEVNULL per
    the project ffmpeg contract (a captured stdin hangs looped invocations).
    """
    th = threshold if threshold is not None else _scene_threshold()
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",  # showinfo logs at info level; quieter levels lose the cut times
        "-i",
        str(media),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={_SCALE_WIDTH}:-2,select='gt(scene,{th})',showinfo",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        log.debug("ffmpeg not found; shot detection skipped")
        return None
    except subprocess.TimeoutExpired:
        log.warning("shot detection timed out after %ds; skipped", timeout_s)
        return None
    if proc.returncode != 0:
        # typical: audio-only media (no 0:v:0 stream to map)
        log.debug("shot detection unavailable for %s (ffmpeg rc=%d)", media.name, proc.returncode)
        return None
    cuts = sorted({float(m.group(1)) for m in _PTS_RE.finditer(proc.stderr)})
    log.info("detected %d shot changes in %s", len(cuts), media.name)
    return cuts
