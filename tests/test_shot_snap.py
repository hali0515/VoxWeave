# tests/test_shot_snap.py
# Shot-change snapping per Netflix TTSG zones (24fps frames): in-times 1-7
# before / 1-9 after a cut land on it, 8-11 before pull out to 12 before, 10-11
# after push to 12 after; out-times die on cut-2frames (up to 12 before / 1-5
# after) or land 12 after (6-11 after, or as last resort when speech crosses
# the cut). Speech is never sacrificed: ends never pull below the last word and
# a start move that would land after the cue's own first word is vetoed.
# Detection itself is one ffmpeg pass parsed from showinfo stderr; audio-only
# media degrades to None.
import io
import json
import logging
import shutil
import subprocess

import pytest

from voxweave import pipeline, shotdet
from voxweave.core.timing import _FRAME_S, _SHOT_LANDING_S, TWO_FRAME_S, _snap_to_shots


def _cue(start, end, speech_end=None, speech_start=None, text="x"):
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [
            {
                "start": start if speech_start is None else speech_start,
                "end": speech_end if speech_end else end,
            }
        ],
    }


def test_end_extends_to_die_on_cut():
    out = _snap_to_shots([_cue(1.0, 2.0)], [2.15], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["end"] == pytest.approx(2.15 - TWO_FRAME_S)


def test_end_pull_back_respects_speech():
    # cut shortly before the cue end: pull back only if speech already finished
    out = _snap_to_shots(
        [_cue(1.0, 2.3, speech_end=2.0)], [2.2], snap_s=0.24, max_cue_s=7.0
    )
    assert out[0]["end"] == pytest.approx(2.2 - TWO_FRAME_S)
    # speech runs through the cut -> never cut a word short; the subtitle
    # legitimately crosses, so it lands 12 frames after the cut instead of
    # flashing out just past it (TTSG last resort)
    out = _snap_to_shots(
        [_cue(1.0, 2.3, speech_end=2.25)], [2.2], snap_s=0.24, max_cue_s=7.0
    )
    assert out[0]["end"] == pytest.approx(2.2 + _SHOT_LANDING_S)


def test_start_leads_in_to_cut():
    out = _snap_to_shots([_cue(1.0, 3.0)], [0.85], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(0.85)


def test_start_flash_removal_vetoed_when_it_would_clip_speech():
    # cue starts 0.15s before a cut -> text would flash across it, and removing
    # the flash means delaying the start onto the cut. The first word is already
    # sounding at 1.0, so the delay would swallow it: the move is vetoed whole
    # (never clamped to the word, which would land outside every TTSG zone).
    out = _snap_to_shots([_cue(1.0, 3.0)], [1.15], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(1.0)


def test_far_cut_untouched():
    out = _snap_to_shots([_cue(1.0, 2.0)], [5.0], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == 1.0 and out[0]["end"] == 2.0


def test_lead_in_clamp_vetoed_when_it_would_clip_speech():
    cues = [_cue(0.0, 0.95), _cue(1.0, 3.0)]
    out = _snap_to_shots(cues, [0.9], snap_s=0.24, max_cue_s=7.0)
    # second cue wants to sit on the 0.9 cut, but cue 1 ends at 0.95, so the
    # 2-frame separation clamp turns the move into a *delay* to 1.0333 -- past
    # its own first word at 1.0. Separation is not worth a clipped word: the cue
    # keeps the start it came in with (the pass never created that tight gap).
    assert out[1]["start"] == pytest.approx(1.0)


def test_end_extension_respects_next_cue_and_cap():
    cues = [_cue(1.0, 2.0), _cue(2.1, 3.0)]
    out = _snap_to_shots(cues, [2.2], snap_s=0.24, max_cue_s=7.0)
    # extending to 2.2-2f would collide with next start 2.1 -> stay put
    assert out[0]["end"] == pytest.approx(2.0)


def test_start_zone_8_to_11_before_pulls_out_to_12():
    # start 9 frames before the cut -> free lead-in out to 12 frames before
    cut = 5.0
    start = cut - 9 * _FRAME_S
    out = _snap_to_shots([_cue(start, 7.0)], [cut], snap_s=0.458, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(cut - _SHOT_LANDING_S)


def test_start_zone_10_to_11_after_vetoed_when_it_would_clip_speech():
    # start 10 frames after the cut -> the zone wants a push out to 12 frames
    # after, but that delays past the first word (sounding at the cue start),
    # so the push is vetoed and the cue keeps its start.
    cut = 5.0
    start = cut + 10 * _FRAME_S
    out = _snap_to_shots([_cue(start, 7.0)], [cut], snap_s=0.458, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(start)


def test_start_delays_apply_when_speech_is_clear():
    # Both delaying zones still fire when the landing does not clip the cue's
    # own speech -- the veto is about speech, not about delays.
    # 1-7 frames before the cut: flash removal onto the cut.
    out = _snap_to_shots(
        [_cue(1.0, 3.0, speech_start=1.2)], [1.15], snap_s=0.24, max_cue_s=7.0
    )
    assert out[0]["start"] == pytest.approx(1.15)
    # 10-11 frames after the cut: push out to the 12-frames-after landing zone.
    cut = 5.0
    start = cut + 10 * _FRAME_S
    out = _snap_to_shots(
        [_cue(start, 7.0, speech_start=cut + _SHOT_LANDING_S)],
        [cut],
        snap_s=0.458,
        max_cue_s=7.0,
    )
    assert out[0]["start"] == pytest.approx(cut + _SHOT_LANDING_S)


def test_lead_in_clamp_applies_when_speech_starts_later():
    # same geometry as the vetoed clamp above, but speech starts at 1.1, after
    # the clamped landing -> the 2-frame separation from the previous cue holds.
    cues = [_cue(0.0, 0.95), _cue(1.0, 3.0, speech_start=1.1)]
    out = _snap_to_shots(cues, [0.9], snap_s=0.24, max_cue_s=7.0)
    assert out[1]["start"] >= out[0]["end"] + TWO_FRAME_S - 1e-9


def test_earlier_zones_snap_regardless_of_speech_start():
    # Moving earlier can never clip speech, so the guard never touches the
    # lead-in zones even though the cue's first word starts at the cue start.
    out = _snap_to_shots([_cue(1.0, 3.0)], [0.85], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(0.85)  # 1-7 frames before -> the cut
    cut = 5.0
    out = _snap_to_shots(
        [_cue(cut - 9 * _FRAME_S, 7.0)], [cut], snap_s=0.458, max_cue_s=7.0
    )
    assert out[0]["start"] == pytest.approx(cut - _SHOT_LANDING_S)  # 8-11 before


def test_untimed_cue_start_still_snaps():
    # No word_data -> no acoustic evidence to protect; the zone rules apply as
    # before (this is the stream align/legacy cues without word timing produce).
    cue = {"text": "x", "start": 1.0, "end": 3.0, "word_data": []}
    out = _snap_to_shots([cue], [1.15], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(1.15)
    cut = 5.0
    cue = {"text": "x", "start": cut + 10 * _FRAME_S, "end": 7.0, "word_data": []}
    out = _snap_to_shots([cue], [cut], snap_s=0.458, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(cut + _SHOT_LANDING_S)


def test_end_zone_6_to_11_after_lands_12_after():
    # end 8 frames after the cut -> extends out to 12 frames after, not pulled
    # back across the cut
    cut = 5.0
    end = cut + 8 * _FRAME_S
    out = _snap_to_shots(
        [_cue(3.0, end, speech_end=end)], [cut], snap_s=0.458, max_cue_s=7.0
    )
    assert out[0]["end"] == pytest.approx(cut + _SHOT_LANDING_S)


def test_snap_disabled_when_window_zero():
    out = _snap_to_shots([_cue(1.0, 2.0)], [2.1], snap_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == 2.0


# --------------------------------------------------------------------------- #
# detection: ffmpeg stderr parsing + graceful degradation
# --------------------------------------------------------------------------- #


SHOWINFO = (
    "[Parsed_showinfo_2 @ 0x1] n:   0 pts:  12345 pts_time:12.345 duration...\n"
    "[Parsed_showinfo_2 @ 0x1] n:   1 pts:  23456 pts_time:23.4 duration...\n"
    "frame=    2 fps=0.0 q=-0.0 Lsize=N/A\n"
)


class _BrokenStderr:
    """A stderr pipe whose read fails, as a torn-down or unreadable stream would."""

    def read(self):
        raise OSError("stderr read failed")

    def close(self):
        pass


class _FakePopen:
    """ffmpeg stand-in: exits with ``rc`` after writing ``stderr``, or hangs.

    ``stderr`` is the text the child emits, or a ready-made stream object such as
    :class:`_BrokenStderr`. A hung fake keeps ``wait`` timing out until it is
    signalled; a ``stubborn`` one also ignores SIGTERM, so cancel has to escalate
    to ``kill`` exactly as it would on a stuck child. Every ``wait`` budget is
    recorded in ``wait_timeouts``.
    """

    def __init__(self, rc=0, stderr="", hang=False, stubborn=False):
        self.returncode = None
        self._rc = rc
        self.stderr = io.StringIO(stderr) if isinstance(stderr, str) else stderr
        self.hang = hang
        self.stubborn = stubborn
        self.calls = []
        self.wait_timeouts = []

    def poll(self):
        if self.hang:
            return None
        self.returncode = self._rc
        return self.returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.hang:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout or 0)
        self.returncode = self._rc
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if not self.stubborn:
            self.hang = False
            self._rc = -15

    def kill(self):
        self.calls.append("kill")
        self.hang = False
        self._rc = -9


def _install_popen(monkeypatch, **fake_kwargs):
    """Patch Popen with a one-shot factory and return the launches it records."""
    launches = []

    def popen(cmd, **kwargs):
        proc = _FakePopen(**fake_kwargs)
        launches.append((cmd, kwargs, proc))
        return proc

    monkeypatch.setattr(shotdet.subprocess, "Popen", popen)
    return launches


def _install_missing_ffmpeg(monkeypatch):
    def popen(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(shotdet.subprocess, "Popen", popen)


def test_detect_parses_showinfo(monkeypatch, tmp_path):
    _install_popen(monkeypatch, rc=0, stderr=SHOWINFO)
    cuts = shotdet.detect_shot_changes(tmp_path / "v.mkv")
    assert cuts == [12.345, 23.4]


def test_detect_none_on_no_video(monkeypatch, tmp_path):
    _install_popen(monkeypatch, rc=1)
    assert shotdet.detect_shot_changes(tmp_path / "a.wav") is None


def test_detect_none_on_missing_ffmpeg(monkeypatch, tmp_path):
    _install_missing_ffmpeg(monkeypatch)
    assert shotdet.detect_shot_changes(tmp_path / "v.mkv") is None


def test_detect_none_on_timeout(monkeypatch, tmp_path):
    launches = _install_popen(monkeypatch, hang=True)
    assert shotdet.detect_shot_changes(tmp_path / "v.mkv", timeout_s=1) is None
    assert launches[0][2].calls == ["kill"]


def test_job_launch_follows_ffmpeg_contract(monkeypatch, tmp_path):
    launches = _install_popen(monkeypatch, rc=0, stderr=SHOWINFO)
    shotdet.ShotDetectionJob().start(tmp_path / "v.mkv", threshold=0.42)
    (cmd, kwargs, _proc), *rest = launches
    assert not rest
    # Byte-identical to the argv the blocking detect_shot_changes built before
    # the job existed; anything else changes what ffmpeg detects.
    assert cmd == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",
        "-i",
        str(tmp_path / "v.mkv"),
        "-map",
        "0:v:0",
        "-vf",
        "scale=320:-2,select='gt(scene,0.42)',showinfo",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True
    # Lenient decoding: ffmpeg echoes the input path and container tags raw.
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_job_result_matches_detect_and_is_idempotent(monkeypatch, tmp_path):
    launches = _install_popen(monkeypatch, rc=0, stderr=SHOWINFO)
    expected = shotdet.detect_shot_changes(tmp_path / "v.mkv")
    job = shotdet.ShotDetectionJob().start(tmp_path / "v.mkv")
    first = job.result()
    assert first == expected == [12.345, 23.4]
    assert job.result() == first
    assert len(launches) == 2
    assert launches[1][2].calls == []
    with pytest.raises(RuntimeError):
        job.start(tmp_path / "v.mkv")


def test_job_result_none_on_nonzero_exit(monkeypatch, tmp_path):
    _install_popen(monkeypatch, rc=1, stderr=SHOWINFO)
    job = shotdet.ShotDetectionJob().start(tmp_path / "a.wav")
    assert job.result() is None
    assert job.result() is None


def test_job_timeout_kills_and_returns_none(monkeypatch, tmp_path):
    launches = _install_popen(monkeypatch, hang=True)
    job = shotdet.ShotDetectionJob().start(tmp_path / "v.mkv", timeout_s=1)
    assert job.result() is None
    proc = launches[0][2]
    assert proc.calls == ["kill"]
    # The budget is granted at collection time, not consumed since launch: a
    # pass that overlapped a long transcription still gets the full timeout.
    assert proc.wait_timeouts[0] == 1
    assert job.result() is None
    job.cancel()
    assert proc.calls == ["kill"]


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_job_keeps_cuts_around_undecodable_stderr_bytes(monkeypatch, tmp_path):
    # ffmpeg -v info echoes the input filename and container tags to stderr
    # verbatim; a non-UTF-8 byte there must not abort the drain and turn every
    # real cut into an empty list.
    script = "printf 'pts_time:1.5\\n' >&2; printf '\\377\\n' >&2; printf 'pts_time:3.5\\n' >&2"
    monkeypatch.setattr(
        shotdet, "_ffmpeg_command", lambda _media, _th: ["sh", "-c", script]
    )
    job = shotdet.ShotDetectionJob().start(tmp_path / "v.mkv")
    assert job.result() == [1.5, 3.5]
    assert shotdet.detect_shot_changes(tmp_path / "v.mkv") == [1.5, 3.5]


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_job_reaps_a_fast_failing_child_before_collection(monkeypatch, tmp_path):
    # Audio-only media makes ffmpeg exit at once; the drain thread reaps it on
    # EOF so it does not linger as a zombie for the whole transcription.
    monkeypatch.setattr(
        shotdet, "_ffmpeg_command", lambda _media, _th: ["sh", "-c", "exit 3"]
    )
    job = shotdet.ShotDetectionJob().start(tmp_path / "a.wav")
    job._join_drain()
    assert job._proc is not None and job._proc.returncode == 3
    assert job.result() is None


def test_job_drain_failure_settles_none_not_partial_cuts(monkeypatch, tmp_path, caplog):
    # A drain that could not read to EOF leaves a truncated buffer; parsing it
    # would silently drop cuts, so the outcome must be "undetectable" instead.
    _install_popen(monkeypatch, rc=0, stderr=_BrokenStderr())
    job = shotdet.ShotDetectionJob().start(tmp_path / "v.mkv")
    with caplog.at_level(logging.WARNING, logger="voxweave.shotdet"):
        assert job.result() is None
    assert "could not read ffmpeg output" in caplog.text
    assert job.result() is None


def test_job_cancel_before_finish_kills_and_settles_none(monkeypatch, tmp_path):
    launches = _install_popen(monkeypatch, hang=True, stubborn=True)
    job = shotdet.ShotDetectionJob().start(tmp_path / "v.mkv")
    job.cancel()
    proc = launches[0][2]
    assert proc.calls == ["terminate", "kill"]
    assert job.result() is None
    job.cancel()
    assert proc.calls == ["terminate", "kill"]


def test_job_cancel_stops_at_terminate_when_the_child_exits(monkeypatch, tmp_path):
    launches = _install_popen(monkeypatch, hang=True)
    job = shotdet.ShotDetectionJob().start(tmp_path / "v.mkv")
    job.cancel()
    assert launches[0][2].calls == ["terminate"]
    assert job.result() is None


def test_job_cancel_is_safe_before_start_and_after_finish(monkeypatch, tmp_path):
    launches = _install_popen(monkeypatch, rc=0, stderr=SHOWINFO)
    job = shotdet.ShotDetectionJob()
    job.cancel()
    assert job.start(tmp_path / "v.mkv").result() == [12.345, 23.4]
    job.cancel()
    assert launches[0][2].calls == []
    assert job.result() == [12.345, 23.4]


def test_job_result_before_start_is_none():
    job = shotdet.ShotDetectionJob()
    assert job.result() is None


def test_job_missing_ffmpeg_settles_none(monkeypatch, tmp_path):
    _install_missing_ffmpeg(monkeypatch)
    job = shotdet.ShotDetectionJob().start(tmp_path / "v.mkv")
    assert job.result() is None
    job.cancel()
    assert job.result() is None


# --------------------------------------------------------------------------- #
# persistence: split replays shot_changes from the sibling JSON
# --------------------------------------------------------------------------- #


def test_split_replays_shot_changes(tmp_path):
    units = [
        {"text": "hello", "start": 1.0, "end": 1.4},
        {"text": "there", "start": 1.5, "end": 2.0},
    ]
    json_path = tmp_path / "clip.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "en",
                "word_segments": units,
                "segments": [],
                "vad_speech": [],
                "shot_changes": [2.2],
            }
        ),
        encoding="utf-8",
    )
    pipeline.split(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["shot_changes"] == [2.2]  # round-trips through the re-split
    seg = data["segments"][-1]
    # cue end snapped onto the 2.2 cut (minus 2 frames), not left at lag-padded end
    assert seg["end"] == pytest.approx(2.2 - TWO_FRAME_S, abs=1e-6)


def test_the_speech_veto_declines_a_separation_repair_it_cannot_make_for_free():
    """The 2-frame floor bounds a move this pass MAKES; it is not a postcondition.

    A start already inside the previous cue's guard band arrives that way. The
    prev-end clamp can lift a snap out of it -- but if the lifted landing would
    fall past the cue's own first word, the speech veto declines the whole move
    and the pre-existing gap stands. Speech beats layout, so this pass repairs a
    separation violation only when the repair costs no audible word. Pinned
    because the docstring used to promise the postcondition unconditionally.
    """
    timed = [
        _cue(0.0, 1.00, speech_start=0.0, speech_end=1.00, text="first"),
        # 0.02 s after the previous end: already inside the 2-frame guard, and
        # its first word starts exactly at the display start.
        _cue(1.02, 3.00, speech_start=1.02, speech_end=3.00, text="second"),
    ]
    out = _snap_to_shots(timed, [1.10], snap_s=0.458, max_cue_s=7.0)
    assert out[1]["start"] == pytest.approx(1.02)
    assert out[1]["start"] - out[0]["end"] < TWO_FRAME_S

    # the same document with no acoustic evidence: nothing to sacrifice, so the
    # move (and with it the separation repair) goes ahead
    untimed = [
        {"text": "first", "start": 0.0, "end": 1.00, "word_data": []},
        {"text": "second", "start": 1.02, "end": 3.00, "word_data": [{"text": "s"}]},
    ]
    out = _snap_to_shots(untimed, [1.10], snap_s=0.458, max_cue_s=7.0)
    assert out[1]["start"] == pytest.approx(1.10)
    assert out[1]["start"] - out[0]["end"] >= TWO_FRAME_S
