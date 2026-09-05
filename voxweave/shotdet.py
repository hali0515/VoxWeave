"""Shot-change detection via ffmpeg scene scores.

Cue boundaries that land just off a hard cut flash across it — the classic
amateur-subtitle tell. ``detect_shot_changes`` decodes the video at reduced
resolution through ffmpeg's scene-score select filter and returns sorted cut
timestamps; smart_split's snap pass then nudges nearby cue boundaries onto
them. Returns ``None`` when the media has no video stream or ffmpeg fails,
so audio-only pipelines skip snapping transparently.

The pass is CPU-only and independent of transcription, so the pipeline runs it
as a :class:`ShotDetectionJob` started before the GPU stages and collected after
them; ``detect_shot_changes`` is the blocking convenience over the same job.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

# select gt(scene,t): ffmpeg docs call 0.3-0.5 reasonable; anime hard cuts score
# high, and the snap pass is conservative, so favor recall.
SCENE_THRESHOLD = 0.3
# Decode at this width for scene scoring: scene scores are stable under scaling
# and a full-res decode of movie-length media would dominate pipeline runtime.
_SCALE_WIDTH = 320
_PTS_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")
# ffmpeg exits on SIGTERM promptly; escalate to SIGKILL only past this grace.
_CANCEL_GRACE_S = 5.0


def _scene_threshold() -> float:
    raw = os.environ.get("VOXWEAVE_SHOT_SCENE", "")
    try:
        return float(raw) if raw.strip() else SCENE_THRESHOLD
    except ValueError:
        return SCENE_THRESHOLD


def _ffmpeg_command(media: Path, threshold: float) -> list[str]:
    """One ffmpeg pass: downscale -> scene-score select -> showinfo on stderr."""
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",  # showinfo logs at info level; quieter levels lose the cut times
        "-i",
        str(media),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={_SCALE_WIDTH}:-2,select='gt(scene,{threshold})',showinfo",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]


class ShotDetectionJob:
    """One shot-detection ffmpeg pass running in the background.

    ``start`` launches ffmpeg and returns at once so the caller can overlap the
    pass with GPU work; ``result`` joins it with exactly the outcome
    :func:`detect_shot_changes` reports (``None`` for no ffmpeg / timeout /
    non-zero exit, sorted unique cut times otherwise) and caches that outcome;
    ``cancel`` reaps the child when the caller bails out before collecting.
    A daemon thread drains stderr while ffmpeg runs -- showinfo is chatty and a
    full pipe would block ffmpeg forever. Single consumer: call ``result`` and
    ``cancel`` from the thread that owns the job.
    """

    def __init__(self) -> None:
        self._media: Path | None = None
        self._timeout_s = 3600
        self._deadline = 0.0
        self._proc: subprocess.Popen[str] | None = None
        self._stderr: list[str] = []
        self._drain: threading.Thread | None = None
        self._outcome: list[float] | None = None
        self._finished = False

    def start(
        self,
        media: Path,
        threshold: float | None = None,
        timeout_s: int = 3600,
    ) -> ShotDetectionJob:
        """Launch the ffmpeg pass on ``media``'s first video stream and return self.

        ``-nostdin`` + DEVNULL per the project ffmpeg contract (a captured stdin
        hangs looped invocations). ``timeout_s`` is wall-clock from launch, as
        ``subprocess.run(timeout=)`` would count it. A missing ffmpeg finishes
        the job immediately with ``None``.
        """
        if self._proc is not None or self._finished:
            raise RuntimeError("shot detection job already started")
        self._media = Path(media)
        self._timeout_s = timeout_s
        self._deadline = time.monotonic() + timeout_s
        th = threshold if threshold is not None else _scene_threshold()
        try:
            proc = subprocess.Popen(
                _ffmpeg_command(self._media, th),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            log.debug("ffmpeg not found; shot detection skipped")
            self._finish(None)
            return self
        assert proc.stderr is not None
        self._proc = proc
        self._drain = threading.Thread(
            target=self._pump_stderr,
            args=(proc.stderr,),
            name="voxweave-shotdet-stderr",
            daemon=True,
        )
        self._drain.start()
        return self

    def result(self) -> list[float] | None:
        """Join the pass and return its cut times (``None`` when undetectable).

        Idempotent: the first call settles the outcome, later calls return it.
        A job that was never started detects nothing.
        """
        if self._finished:
            return self._outcome
        proc = self._proc
        if proc is None:
            return self._finish(None)
        media = self._media
        assert media is not None
        try:
            proc.wait(timeout=max(0.0, self._deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            log.warning("shot detection timed out after %ds; skipped", self._timeout_s)
            proc.kill()
            proc.wait()
            self._join_drain()
            return self._finish(None)
        self._join_drain()
        if proc.returncode != 0:
            # typical: audio-only media (no 0:v:0 stream to map)
            log.debug(
                "shot detection unavailable for %s (ffmpeg rc=%d)",
                media.name,
                proc.returncode,
            )
            return self._finish(None)
        cuts = sorted(
            {float(m.group(1)) for m in _PTS_RE.finditer("".join(self._stderr))}
        )
        log.info("detected %d shot changes in %s", len(cuts), media.name)
        return self._finish(cuts)

    def cancel(self) -> None:
        """Stop a running pass and settle the outcome to ``None``.

        Terminates, then kills if ffmpeg lingers past the grace period. Idempotent,
        a no-op before ``start`` and after the outcome has settled.
        """
        proc = self._proc
        if proc is None or self._finished:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_CANCEL_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._join_drain()
        media = self._media
        assert media is not None
        log.debug("shot detection cancelled for %s", media.name)
        self._finish(None)

    def _pump_stderr(self, stream: IO[str]) -> None:
        try:
            self._stderr.append(stream.read())
        except (OSError, ValueError) as exc:
            log.debug("shot detection stderr drain stopped: %r", exc)
        finally:
            stream.close()

    def _join_drain(self) -> None:
        if self._drain is not None:
            self._drain.join()
            self._drain = None

    def _finish(self, outcome: list[float] | None) -> list[float] | None:
        self._outcome = outcome
        self._finished = True
        return outcome


def detect_shot_changes(
    media: Path,
    threshold: float | None = None,
    timeout_s: int = 3600,
) -> list[float] | None:
    """Return sorted shot-change timestamps (seconds) for ``media``'s first video
    stream, or ``None`` when undetectable (no video stream, no ffmpeg, timeout).

    Blocking convenience over :class:`ShotDetectionJob`: start the pass and
    collect it immediately.
    """
    return ShotDetectionJob().start(media, threshold, timeout_s).result()
