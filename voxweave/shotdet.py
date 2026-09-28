"""Shot-change detection via ffmpeg scene scores.

Cue boundaries that land just off a hard cut flash across it — the classic
amateur-subtitle tell. :class:`ShotDetectionJob` decodes the video at reduced
resolution through ffmpeg's scene-score select filter and reports sorted cut
timestamps; smart_split's snap pass then nudges nearby cue boundaries onto
them. The outcome is ``None`` when the media has no video stream or ffmpeg
fails, so audio-only pipelines skip snapping transparently.

The pass is CPU-only and independent of transcription, so the pipeline starts
the job before the GPU stages and collects it after them.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
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
    """``VOXWEAVE_SHOT_SCENE`` (ffmpeg scene score, 0 < t <= 1) or the default.

    An unparsable or out-of-range value warns and falls back to the default:
    scene scores lie in [0, 1], so gt(scene,t) fires on nearly every frame for
    t <= 0 and never for t > 1.
    """
    raw = os.environ.get("VOXWEAVE_SHOT_SCENE", "").strip()
    if not raw:
        return SCENE_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        value = None
    if value is None or not 0.0 < value <= 1.0:
        log.warning(
            "ignoring VOXWEAVE_SHOT_SCENE=%r (expected a number in (0, 1]); using %s",
            raw,
            SCENE_THRESHOLD,
        )
        return SCENE_THRESHOLD
    return value


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
    pass with GPU work; ``result`` joins it and caches the outcome (``None``
    for no ffmpeg / no video stream / timeout / non-zero exit / unreadable
    stderr, sorted unique cut times otherwise); ``cancel`` reaps the child when
    the caller bails out before collecting.
    A daemon thread drains stderr while ffmpeg runs -- showinfo is chatty and a
    full pipe would block ffmpeg forever. Single consumer: call ``result`` and
    ``cancel`` from the thread that owns the job.
    """

    def __init__(self) -> None:
        self._media: Path | None = None
        self._timeout_s = 3600
        self._proc: subprocess.Popen[str] | None = None
        self._stderr: list[str] = []
        self._drain: threading.Thread | None = None
        # Set by the drain thread when it could not read stderr to EOF; the
        # buffer is then untrustworthy and ``result`` must not parse it.
        self._drain_error: Exception | None = None
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
        hangs looped invocations). ``timeout_s`` is the budget ``result`` grants
        the pass once the caller starts waiting for it, not wall-clock from
        launch: a legitimately slow pass must not be killed just because the
        work it overlapped took longer than the budget (a caller that collects
        immediately sees the two readings coincide). A missing ffmpeg finishes
        the job immediately with ``None``.

        stderr is decoded as UTF-8 with undecodable bytes replaced: ffmpeg
        echoes the input path and container tags verbatim, and a strict decode
        would abort the drain on a non-UTF-8 filename while the ASCII
        ``pts_time`` lines stay perfectly parseable.
        """
        if self._proc is not None or self._finished:
            raise RuntimeError("shot detection job already started")
        self._media = Path(media)
        self._timeout_s = timeout_s
        th = threshold if threshold is not None else _scene_threshold()
        try:
            proc = subprocess.Popen(
                _ffmpeg_command(self._media, th),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            log.debug("ffmpeg not found; shot detection skipped")
            self._finish(None)
            return self
        assert proc.stderr is not None
        self._proc = proc
        self._drain = threading.Thread(
            target=self._pump_stderr,
            args=(proc, proc.stderr),
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
            proc.wait(timeout=self._timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("shot detection timed out after %ds; skipped", self._timeout_s)
            proc.kill()
            proc.wait()
            self._join_drain()
            return self._finish(None)
        self._join_drain()
        if self._drain_error is not None:
            # The buffer may be truncated; parsing it would silently drop cuts.
            log.warning(
                "shot detection could not read ffmpeg output for %s (%r); skipped",
                media.name,
                self._drain_error,
            )
            return self._finish(None)
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

    def _pump_stderr(self, proc: subprocess.Popen[str], stream: IO[str]) -> None:
        try:
            self._stderr.append(stream.read())
        except (OSError, ValueError) as exc:
            self._drain_error = exc
            log.debug("shot detection stderr drain stopped: %r", exc)
        finally:
            stream.close()
        # EOF means ffmpeg is exiting: reap it now (best effort, non-blocking) so a
        # fast-failing child does not sit as a zombie until the caller collects.
        proc.poll()

    def _join_drain(self) -> None:
        if self._drain is not None:
            self._drain.join()
            self._drain = None

    def _finish(self, outcome: list[float] | None) -> list[float] | None:
        self._outcome = outcome
        self._finished = True
        return outcome
