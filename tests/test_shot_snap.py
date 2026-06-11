# tests/test_shot_snap.py
# Shot-change snapping per Netflix TTSG zones (24fps frames): in-times 1-7
# before / 1-9 after a cut land on it, 8-11 before pull out to 12 before, 10-11
# after push to 12 after; out-times die on cut-2frames (up to 12 before / 1-5
# after) or land 12 after (6-11 after, or as last resort when speech crosses
# the cut). Speech is never sacrificed: ends never pull below the last word.
# Detection itself is one ffmpeg pass parsed from showinfo stderr; audio-only
# media degrades to None.
import json
import subprocess

import pytest

from voxweave import pipeline, shotdet
from voxweave.core.timing import _FRAME_S, _SHOT_LANDING_S, TWO_FRAME_S, _snap_to_shots


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


def test_start_zone_8_to_11_before_pulls_out_to_12():
    # start 9 frames before the cut -> free lead-in out to 12 frames before
    cut = 5.0
    start = cut - 9 * _FRAME_S
    out = _snap_to_shots([_cue(start, 7.0)], [cut], snap_s=0.458, max_cue_s=7.0)
    assert out[0]["start"] == pytest.approx(cut - _SHOT_LANDING_S)


def test_start_zone_10_to_11_after_pushes_out_to_12():
    # start 10 frames after the cut -> pushed out to the 12-frames-after zone
    cut = 5.0
    start = cut + 10 * _FRAME_S
    out = _snap_to_shots([_cue(start, 7.0)], [cut], snap_s=0.458, max_cue_s=7.0)
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
