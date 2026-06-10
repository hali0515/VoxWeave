# tests/test_shot_snap.py
# Shot-change snapping: cue boundaries within the snap window move onto cuts —
# start may lead in earlier (or shift later only within the window, removing a
# pre-cut flash), end may extend to cut-2frames inside the gap or pull back only
# down to the last word's end. Detection itself is one ffmpeg pass parsed from
# showinfo stderr; audio-only media degrades to None.
import json
import subprocess

import pytest

from voxweave import pipeline, shotdet
from voxweave.core.smart_split import TWO_FRAME_S, _snap_to_shots


def _cue(start, end, speech_end=None, text="x"):
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [{"start": start, "end": speech_end if speech_end else end}],
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
    # speech runs through the cut -> no pull-back (never cut a word short)
    out = _snap_to_shots(
        [_cue(1.0, 2.3, speech_end=2.25)], [2.2], snap_s=0.24, max_cue_s=7.0
    )
    assert out[0]["end"] == pytest.approx(2.3)


def test_start_leads_in_to_cut():
    out = _snap_to_shots([_cue(1.0, 3.0)], [0.85], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(0.85)


def test_start_shift_later_removes_precut_flash():
    # cue starts 0.15s before a cut -> text would flash across it; start moves to the cut
    out = _snap_to_shots([_cue(1.0, 3.0)], [1.15], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(1.15)


def test_far_cut_untouched():
    out = _snap_to_shots([_cue(1.0, 2.0)], [5.0], snap_s=0.24, max_cue_s=7.0)
    assert out[0]["start"] == 1.0 and out[0]["end"] == 2.0


def test_lead_in_respects_previous_cue():
    cues = [_cue(0.0, 0.95), _cue(1.0, 3.0)]
    out = _snap_to_shots(cues, [0.9], snap_s=0.24, max_cue_s=7.0)
    # second cue wants to lead in to 0.9 but cue 1 ends at ~0.9; keep 2 frames clear
    assert out[1]["start"] >= out[0]["end"] + TWO_FRAME_S - 1e-9


def test_end_extension_respects_next_cue_and_cap():
    cues = [_cue(1.0, 2.0), _cue(2.1, 3.0)]
    out = _snap_to_shots(cues, [2.2], snap_s=0.24, max_cue_s=7.0)
    # extending to 2.2-2f would collide with next start 2.1 -> stay put
    assert out[0]["end"] == pytest.approx(2.0)


def test_snap_disabled_when_window_zero():
    out = _snap_to_shots([_cue(1.0, 2.0)], [2.1], snap_s=0.0, max_cue_s=7.0)
    assert out[0]["end"] == 2.0


# --------------------------------------------------------------------------- #
# detection: ffmpeg stderr parsing + graceful degradation
# --------------------------------------------------------------------------- #


class _Proc:
    def __init__(self, rc, stderr=""):
        self.returncode = rc
        self.stderr = stderr


def test_detect_parses_showinfo(monkeypatch, tmp_path):
    stderr = (
        "[Parsed_showinfo_2 @ 0x1] n:   0 pts:  12345 pts_time:12.345 duration...\n"
        "[Parsed_showinfo_2 @ 0x1] n:   1 pts:  23456 pts_time:23.4 duration...\n"
        "frame=    2 fps=0.0 q=-0.0 Lsize=N/A\n"
    )
    monkeypatch.setattr(
        shotdet.subprocess, "run", lambda *a, **k: _Proc(0, stderr=stderr)
    )
    cuts = shotdet.detect_shot_changes(tmp_path / "v.mkv")
    assert cuts == [12.345, 23.4]


def test_detect_none_on_no_video(monkeypatch, tmp_path):
    monkeypatch.setattr(shotdet.subprocess, "run", lambda *a, **k: _Proc(1))
    assert shotdet.detect_shot_changes(tmp_path / "a.wav") is None


def test_detect_none_on_missing_ffmpeg(monkeypatch, tmp_path):
    def _raise(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(shotdet.subprocess, "run", _raise)
    assert shotdet.detect_shot_changes(tmp_path / "v.mkv") is None


def test_detect_none_on_timeout(monkeypatch, tmp_path):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(shotdet.subprocess, "run", _raise)
    assert shotdet.detect_shot_changes(tmp_path / "v.mkv") is None


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
