"""Pipeline wiring regressions: shot re-snapping after speaker formatting, and
explicit release of the detection singletons (PANNs, silero) once their last
consumer in a job is done.

Model-free: transcription/diarization are stubbed, everything under test is the
deterministic layout + timing chain.
"""

import json
from pathlib import Path

import pytest

from voxweave import backend, chunking, pipeline, songdet
from voxweave.config import gap_thresholds
from voxweave.core.smart_split import smart_split_segments
from voxweave.diarize import apply_speaker_format

FRAME_S = 1.0 / 24.0
TWO_FRAME_S = 2 * FRAME_S
LANDING_S = 12 * FRAME_S

# Two speakers, one exchange: A holds 0.0-2.0, B answers 2.4-3.6. Cuts sit right
# after each speaker's last word, so the cue ends are inside the shot-change
# adjustment zone and the Netflix landing rules apply.
WORDS = [
    ("Where", 0.0, 0.4),
    ("did", 0.5, 0.8),
    ("you", 0.9, 1.2),
    ("go", 1.4, 2.0),
    ("Nowhere", 2.4, 3.0),
    ("special", 3.1, 3.6),
]
UNITS = [{"text": w, "start": s, "end": e} for w, s, e in WORDS]
TURNS = [(0.0, 2.2, "SPEAKER_00"), (2.3, 3.8, "SPEAKER_01")]
SHOTS = [2.3, 3.9]


def _zone_violations(cues, cuts, snap_s):
    """Cue ends paired with a cut but sitting on no legal landing spot.

    Netflix TTSG: an out-time within the pairing window either dies two frames
    before the cut or clears it by twelve frames. Anything in between flashes
    across the cut.
    """
    bad = []
    for c in cues:
        end = float(c["end"])
        for cut in cuts:
            if abs(end - cut) > snap_s:
                continue
            on_cut = abs(end - (cut - TWO_FRAME_S)) <= 1e-6
            cleared = end >= cut + LANDING_S - 1e-6
            if not (on_cut or cleared):
                bad.append((c["text"], end, cut))
    return bad


def _write_json(path: Path, **extra) -> Path:
    payload = {"language": "en", "word_segments": UNITS}
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_speaker_format_cleanup_undoes_shot_snapping():
    """Precondition for the fix: without a second snap the cue flashes.

    smart_split already snapped the exchange's out-time to two frames before the
    3.9s cut; the speaker-format cleanup then re-extends it to speech end + the
    lag-out pad, landing inside the cut's adjustment zone.
    """
    th = gap_thresholds("en")
    seg = pipeline._units_to_seg(UNITS, "en")
    cues = smart_split_segments([seg], lang="en", thresholds=th, shot_changes=SHOTS)
    assert not _zone_violations(cues, SHOTS, th["shot_snap_s"])

    formatted = apply_speaker_format(cues, TURNS, "en", thresholds=th)
    assert _zone_violations(formatted, SHOTS, th["shot_snap_s"]), (
        "speaker formatting no longer disturbs the snapped out-time; "
        "this test no longer covers the regression"
    )


def test_split_replay_resnaps_shots_after_speaker_format(tmp_path):
    """split replay: stored speaker_turns + shot_changes -> no cue flashes."""
    th = gap_thresholds("en")
    j = _write_json(
        tmp_path / "ep.json",
        speaker_turns=[[s, e, lb] for s, e, lb in TURNS],
        shot_changes=SHOTS,
    )
    pipeline.split(j)

    cues = json.loads(j.read_text(encoding="utf-8"))["segments"]
    assert any("-" in c["text"] for c in cues), "speaker formatting did not run"
    assert _zone_violations(cues, SHOTS, th["shot_snap_s"]) == []
    # the dual-speaker cue dies two frames before the 3.9s cut
    assert cues[-1]["end"] == 3.9 - TWO_FRAME_S


def test_process_resnaps_shots_after_speaker_format(tmp_path, monkeypatch):
    """Same invariant on the process path (fresh transcription + shot detection)."""
    th = gap_thresholds("en")
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"x")

    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *a, **kw: ("en", UNITS, [(0.0, 3.6)], [], list(TURNS)),
    )
    monkeypatch.setattr(
        "voxweave.shotdet.detect_shot_changes", lambda *a, **kw: list(SHOTS)
    )

    pipeline.process(media)

    cues = json.loads((tmp_path / "ep.json").read_text(encoding="utf-8"))["segments"]
    assert any("-" in c["text"] for c in cues), "speaker formatting did not run"
    assert _zone_violations(cues, SHOTS, th["shot_snap_s"]) == []


def test_split_replay_without_shots_keeps_speaker_format(tmp_path):
    """No stored cuts: re-snapping is a no-op, formatting still applies."""
    j = _write_json(
        tmp_path / "ep.json", speaker_turns=[[s, e, lb] for s, e, lb in TURNS]
    )
    pipeline.split(j)
    cues = json.loads(j.read_text(encoding="utf-8"))["segments"]
    assert any("-" in c["text"] for c in cues)


@pytest.mark.parametrize(
    "mapping",
    ["", "{", '{"version":2,"speakers":{}}', '{"version":1,"speakers":[]}'],
)
def test_split_ignores_corrupt_speaker_mapping_once(tmp_path, caplog, mapping):
    j = _write_json(
        tmp_path / "ep.json", speaker_turns=[[s, e, label] for s, e, label in TURNS]
    )
    (tmp_path / "ep.speakers.json").write_text(mapping, encoding="utf-8")

    with caplog.at_level("WARNING", logger="voxweave"):
        out = pipeline.split(j)

    assert out.exists()
    assert "<v " not in out.read_text(encoding="utf-8")
    assert (
        sum(
            "ignoring unreadable speaker mapping" in record.message
            for record in caplog.records
        )
        == 1
    )


def test_split_passes_layout_budget_to_speaker_format(tmp_path, monkeypatch):
    """Layout overrides must reach both packer and speaker formatter."""
    seen: dict = {}

    def fake_format(cues, turns, lang, **kw):
        seen.update(kw)
        return list(cues)

    monkeypatch.setattr("voxweave.diarize.apply_speaker_format", fake_format)
    j = _write_json(
        tmp_path / "ep.json", speaker_turns=[[s, e, lb] for s, e, lb in TURNS]
    )
    pipeline.split(j, max_line_length=20, max_lines=1)
    assert seen["max_line_length"] == 20
    assert seen["max_lines"] == 1


def _stub_transcribe_models(monkeypatch, tmp_path):
    """Stub every model call inside transcribe (decode/VAD/ASR/alignment)."""
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *a, **kw: wav)
    monkeypatch.setattr(
        pipeline, "vad_speech_segments", lambda *a, **kw: [{"start": 0.0, "end": 2.0}]
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *a, **kw: wav)
    monkeypatch.setattr(backend, "chunk_pass_count", lambda *a, **kw: 2)
    monkeypatch.setattr(backend, "release", lambda: None)
    monkeypatch.setattr(
        backend,
        "transcribe_chunks",
        lambda *a, **kw: [("English", "hello world", list(UNITS[:2]))],
    )


def test_transcribe_releases_panns_and_silero(tmp_path, monkeypatch):
    """Plain transcribe: both detection singletons are dropped before returning."""
    released: list[str] = []
    _stub_transcribe_models(monkeypatch, tmp_path)
    monkeypatch.setattr(songdet, "release_model", lambda: released.append("panns"))
    monkeypatch.setattr(
        chunking, "release_silero_vad", lambda: released.append("silero")
    )

    media = tmp_path / "ep.mkv"
    media.write_bytes(b"x")
    iso, units, *_ = pipeline.transcribe(media, separate=False)

    assert iso == "en" and units
    assert "panns" in released and "silero" in released


def test_transcribe_releases_panns_even_when_it_fails(tmp_path, monkeypatch):
    """A failed run must not strand PANNs, deferral requested or not."""
    released: list[str] = []
    _stub_transcribe_models(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "vad_speech_segments", lambda *a, **kw: [])
    monkeypatch.setattr(songdet, "release_model", lambda: released.append("panns"))
    monkeypatch.setattr(chunking, "release_silero_vad", lambda: None)

    media = tmp_path / "ep.mkv"
    media.write_bytes(b"x")
    try:
        pipeline.transcribe(media, separate=False, release_panns=False)
    except RuntimeError:
        pass
    else:  # pragma: no cover - guards the fixture, not the behavior
        raise AssertionError("expected 'no speech detected'")
    assert released == ["panns"]


def test_process_defers_panns_release_to_the_sdh_pass(tmp_path, monkeypatch):
    """--sdh reuses the PANNs singleton on the original mix: transcribe hands it
    over, and the sidecar pass is what releases it."""
    order: list[str] = []
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"x")

    def fake_transcribe(*a, **kw):
        order.append(f"transcribe(release_panns={kw['release_panns']})")
        return ("en", UNITS, None, [], [])

    def fake_sidecar(*a, **kw):
        order.append("sdh")

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "_write_sdh_sidecar", fake_sidecar)
    monkeypatch.setattr(songdet, "release_model", lambda: order.append("release"))

    pipeline.process(media, sdh=True, shot_snap=False)
    assert order == ["transcribe(release_panns=False)", "sdh", "release"]


def test_process_releases_panns_after_failed_sdh_sidecar(tmp_path, monkeypatch):
    """The sidecar can explode (PANNs OOM); the model must still be released."""
    order: list[str] = []
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"x")

    def boom(*a, **kw):
        raise RuntimeError("PANNs exploded")

    monkeypatch.setattr(
        pipeline, "transcribe", lambda *a, **kw: ("en", UNITS, None, [], [])
    )
    monkeypatch.setattr(pipeline, "_write_sdh_sidecar", boom)
    monkeypatch.setattr(songdet, "release_model", lambda: order.append("release"))

    pipeline.process(media, sdh=True, shot_snap=False)
    assert order == ["release"]
