import json
import os
import stat
from pathlib import Path

import pytest

from voxweave import backend, chunking, pipeline, songdet
from voxweave.mediasnapshot import SnapshotUnavailable
from voxweave.voicebase import (
    load_voiceprints,
    media_fingerprint,
    validate_voiceprint_conjunction,
)
from voxweave.voiceepisode import episode_lock_path
from voxweave.vocalscache import (
    load_cache_companion,
    publish_cache_companion,
    validate_cache_pair,
)


UNIT = {"text": "hello", "start": 0.0, "end": 1.0}
TURN = (0.0, 1.0, "SPEAKER_00")
VECTOR = [1.0, *([0.0] * 15)]
SEPARATOR = {
    "repo": "example/separator",
    "file": "weights.ckpt",
    "checkpoint": "b" * 64,
    "config_sha256": "c" * 64,
}
PROVENANCE = {
    "diarization_model": "example/diarizer",
    "outer_config_sha256": "d" * 64,
    "embedding_model": "example/embedder",
    "embedding_checkpoint": "e" * 64,
    "embedding_dim": 16,
    "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
    "pyannote_version": "3.4.0",
    "torch_version": "test",
}


@pytest.fixture(autouse=True)
def _private_snapshot_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(tmp_path / "cache-root"))


def _capture(turns):
    return pipeline.VoiceprintCapture(
        centroids={"SPEAKER_00": list(VECTOR)},
        provenance=dict(PROVENANCE),
        turns=turns,
    )


def test_process_uses_snapshot_and_commits_bound_pair(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    source_paths: list[Path] = []

    def fake_transcribe(source, **kwargs):
        source = Path(source)
        source_paths.append(source)
        assert source != media
        assert source.exists()
        assert source.read_bytes() == media.read_bytes()
        assert kwargs["cache_vocals"] == pipeline.cache_vocals_path(media)
        assert kwargs["source_fingerprint"] == media_fingerprint(media)
        turns = [TURN]
        return "en", [dict(UNIT)], [(0.0, 1.0)], [], turns, _capture(turns)

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    out = pipeline.process(
        media,
        diarize=True,
        voiceprints=True,
        shot_snap=False,
    )

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    sidecar, validated = load_voiceprints(tmp_path / "episode.voiceprints.json")
    assert out == tmp_path / "episode.vtt"
    assert sibling["voiceprint_capture"] == validated.capture_id
    assert sibling["voiceprint_media"] == media_fingerprint(media)
    validate_voiceprint_conjunction(sidecar, sibling, media_fingerprint(media))
    assert source_paths and not source_paths[0].exists()
    lock_path = episode_lock_path(media)
    assert lock_path.exists()
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_capture_logs_biometric_notice_once_per_process(tmp_path, monkeypatch, caplog):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")

    def fake_transcribe(_source, **_kwargs):
        turns = [TURN]
        return "en", [dict(UNIT)], [], [], turns, _capture(turns)

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "_voiceprint_notice_logged", False)
    with caplog.at_level("WARNING", logger="voxweave"):
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    assert caplog.text.count("sensitive voice-biometric sidecar") == 1


def test_process_routes_capture_shot_detection_through_snapshot(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    seen: dict[str, Path] = {}

    def fake_transcribe(source, **_kwargs):
        turns = [TURN]
        seen["transcribe"] = Path(source)
        return "en", [dict(UNIT)], [], [], turns, _capture(turns)

    def fake_shots(source):
        seen["shots"] = Path(source)
        return []

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr("voxweave.shotdet.detect_shot_changes", fake_shots)
    pipeline.process(media, diarize=True, voiceprints=True)

    assert seen["transcribe"] == seen["shots"]
    assert seen["transcribe"] != media


def test_live_media_mismatch_drops_pair_and_machine_artifacts(
    tmp_path, monkeypatch, caplog
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    for suffix in (".voiceprints.json", ".speakers.suggest.json", ".speakers.html"):
        pipeline.swap_ext(media, suffix).write_text("stale", encoding="utf-8")

    def fake_transcribe(_source, **_kwargs):
        turns = [TURN]
        return "en", [dict(UNIT)], [], [], turns, _capture(turns)

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "media_fingerprint", lambda _path: "f" * 64)
    with caplog.at_level("WARNING", logger="voxweave"):
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert "voiceprint_capture" not in sibling
    assert "voiceprint_media" not in sibling
    for suffix in (".voiceprints.json", ".speakers.suggest.json", ".speakers.html"):
        assert not pipeline.swap_ext(media, suffix).exists()
    assert "live media changed" in caplog.text


def test_snapshot_unavailable_continues_without_capture(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    (tmp_path / "episode.voiceprints.json").write_text("stale", encoding="utf-8")

    class BrokenSnapshot:
        def __init__(self, _source):
            pass

        def __enter__(self):
            raise SnapshotUnavailable("no private copy")

        def __exit__(self, *_args):
            return False

    def fake_transcribe(source, **kwargs):
        assert Path(source) == media
        assert kwargs["voiceprints"] is False
        return "en", [dict(UNIT)], [], [], [], None

    monkeypatch.setattr(pipeline, "MediaSnapshot", BrokenSnapshot)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert "voiceprint_capture" not in sibling
    assert not (tmp_path / "episode.voiceprints.json").exists()


def _stub_transcribe_tail(tmp_path, monkeypatch):
    wav = tmp_path / "speech.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(
        pipeline,
        "vad_speech_segments",
        lambda *_args, **_kwargs: [{"start": 0.0, "end": 1.0}],
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(backend, "chunk_pass_count", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(
        backend,
        "transcribe_chunks",
        lambda *_args, **_kwargs: [("English", "hello", [dict(UNIT)])],
    )
    monkeypatch.setattr(backend, "release", lambda: None)
    monkeypatch.setattr(chunking, "release_silero_vad", lambda: None)
    monkeypatch.setattr(songdet, "release_model", lambda: None)
    return wav


def test_capture_cache_hit_validates_pair_before_decode(tmp_path, monkeypatch):
    media = tmp_path / "snapshot.mkv"
    media.write_bytes(b"source")
    cache = tmp_path / "cache" / "episode.vocals.32k.flac"
    cache.parent.mkdir()
    cache.write_bytes(b"bound flac")
    fingerprint = media_fingerprint(media)
    publish_cache_companion(
        cache,
        media_fingerprint=fingerprint,
        separator=SEPARATOR,
    )
    wav = _stub_transcribe_tail(tmp_path, monkeypatch)
    decoded: list[Path] = []

    def fake_decode(source, **_kwargs):
        decoded.append(Path(source))
        return wav

    monkeypatch.setattr(backend, "separator_identity", lambda: dict(SEPARATOR))
    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
    monkeypatch.setattr(
        pipeline,
        "_separate_to_16k_32k",
        lambda *_args, **_kwargs: pytest.fail("bound cache should be a hit"),
    )

    result = pipeline.transcribe(
        media,
        separate=True,
        voiceprints=True,
        source_fingerprint=fingerprint,
        cache_vocals=cache,
    )

    assert result[0] == "en"
    assert decoded == [cache.resolve()]
    assert Path(f"{cache.resolve()}.lock").exists()


def test_capture_cache_mismatch_reseparates_and_rebinds(tmp_path, monkeypatch):
    media = tmp_path / "snapshot.mkv"
    media.write_bytes(b"new source")
    cache = tmp_path / "cache" / "episode.vocals.32k.flac"
    cache.parent.mkdir()
    cache.write_bytes(b"old flac")
    publish_cache_companion(
        cache,
        media_fingerprint="a" * 64,
        separator=SEPARATOR,
    )
    fingerprint = media_fingerprint(media)
    wav = _stub_transcribe_tail(tmp_path, monkeypatch)
    fullband = tmp_path / "fullband.wav"
    vocals = tmp_path / "vocals.flac"
    voc32 = tmp_path / "voc32.wav"
    for path in (fullband, vocals, voc32):
        path.write_bytes(path.name.encode())
    separated_from: list[Path] = []

    def fake_separate(source, **_kwargs):
        separated_from.append(Path(source))
        return fullband, vocals, wav, voc32

    def fake_encode(_source, destination):
        Path(destination).write_bytes(b"new bound flac")

    monkeypatch.setattr(backend, "separator_identity", lambda: dict(SEPARATOR))
    monkeypatch.setattr(pipeline, "_separate_to_16k_32k", fake_separate)
    monkeypatch.setattr(pipeline, "_encode_flac", fake_encode)
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)

    pipeline.transcribe(
        media,
        separate=True,
        voiceprints=True,
        source_fingerprint=fingerprint,
        cache_vocals=cache,
    )

    companion, _validated = load_cache_companion(Path(f"{cache.resolve()}.meta.json"))
    validate_cache_pair(
        companion,
        cache,
        media_fingerprint=fingerprint,
        separator=SEPARATOR,
    )
    assert separated_from == [media]


def test_episode_lock_canonicalizes_parent_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    assert episode_lock_path(real / "episode.mkv") == episode_lock_path(
        alias / "episode.mkv"
    )
    assert os.path.realpath(real / "episode.episode.lock") == os.fspath(
        episode_lock_path(real / "episode.mkv")
    )
