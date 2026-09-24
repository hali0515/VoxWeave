import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from voxweave import (
    artifacts,
    backend,
    chunking,
    diarize,
    episode_transaction,
    pipeline,
    songdet,
)
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
    sidecar, validated = load_voiceprints(artifacts.claim_paths(media).voiceprints)
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

    class _ShotJob:
        def start(self, source, *_a, **_k):
            seen["shots"] = Path(source)
            return self

        def result(self):
            return []

        def cancel(self):
            pass

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr("voxweave.shotdet.ShotDetectionJob", _ShotJob)
    pipeline.process(media, diarize=True, voiceprints=True)

    assert seen["transcribe"] == seen["shots"]
    assert seen["transcribe"] != media


def test_capture_raw_decoder_reads_stable_snapshot_through_truncate_aba(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    original = b"A" * 64
    media.write_bytes(original)
    wav = tmp_path / "speech.wav"
    wav.write_bytes(b"wav")
    decoded_sources: list[Path] = []

    def fake_decode(source, **_kwargs):
        source = Path(source)
        decoded_sources.append(source)
        assert source != media
        with media.open("r+b") as live:
            live.truncate(0)
            live.write(b"B" * len(original))
            live.flush()
            os.fsync(live.fileno())
            live.seek(0)
            live.truncate(0)
            live.write(original)
            live.flush()
            os.fsync(live.fileno())
        assert source.read_bytes() == original
        return wav

    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
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
    monkeypatch.setattr(
        diarize,
        "diarize_turns",
        lambda *_args, **_kwargs: diarize.DiarizationResult(
            turns=[TURN],
            centroids={"SPEAKER_00": list(VECTOR)},
            provenance=dict(PROVENANCE),
        ),
    )
    monkeypatch.setattr(diarize, "release", lambda: None)
    # Legacy lane: the faked diarizer supplies the centroids itself.
    monkeypatch.setenv("VOXWEAVE_VOICEPRINT_MODEL", "pyannote")

    pipeline.process(
        media,
        separate=False,
        diarize=True,
        voiceprints=True,
        shot_snap=False,
    )

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert sibling["voiceprint_media"] == media_fingerprint(media)
    assert decoded_sources and not decoded_sources[0].exists()


def test_live_media_mismatch_stale_aborts_without_machine_artifact_cleanup(
    tmp_path, monkeypatch, caplog
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    for suffix in (".voiceprints.json", ".speakers.suggest.json", ".speakers.html"):
        pipeline.swap_ext(media, suffix).write_text("stale", encoding="utf-8")

    def fake_transcribe(_source, **_kwargs):
        turns = [TURN]
        media.write_bytes(b"replacement media")
        return "en", [dict(UNIT)], [], [], turns, _capture(turns)

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    with caplog.at_level("WARNING", logger="voxweave"):
        with pytest.raises(episode_transaction.MediaStaleError) as caught:
            pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    assert caught.value.failure.detail_code == "media-generation"
    assert not (tmp_path / "episode.json").exists()
    assert not (tmp_path / "episode.vtt").exists()
    for suffix in (".voiceprints.json", ".speakers.suggest.json", ".speakers.html"):
        assert pipeline.swap_ext(media, suffix).read_text(encoding="utf-8") == "stale"


def test_process_capture_publishes_primaries_then_machine_artifact(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"A" * 64)
    turns = [TURN]
    for suffix in (".voiceprints.json", ".speakers.suggest.json", ".speakers.html"):
        pipeline.swap_ext(media, suffix).write_text("stale", encoding="utf-8")

    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: (
            "en",
            [dict(UNIT)],
            [],
            [],
            turns,
            _capture(turns),
        ),
    )
    real_replace = episode_transaction._replace_stage
    order: list[str] = []

    def observed_replace(stage):
        order.append(stage.target.name)
        real_replace(stage)

    monkeypatch.setattr(episode_transaction, "_replace_stage", observed_replace)

    pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert order == ["episode.json", "episode.vtt", "episode.voiceprints.json"]
    assert isinstance(sibling["voiceprint_capture"], str)
    assert sibling["voiceprint_capture"].startswith("c")
    assert sibling["voiceprint_media"] == media_fingerprint(media)
    assert (tmp_path / "episode.voiceprints.json").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()
    assert not (tmp_path / "episode.speakers.html").exists()


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


def test_no_diarize_rewrite_deletes_complete_machine_artifact_set(tmp_path):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    for suffix in (".voiceprints.json", ".speakers.suggest.json", ".speakers.html"):
        pipeline.swap_ext(media, suffix).write_text("stale", encoding="utf-8")

    pipeline.process(
        media,
        word_segments=("en", [dict(UNIT)]),
        shot_snap=False,
    )

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert "voiceprint_capture" not in sibling
    assert "voiceprint_media" not in sibling
    for suffix in (".voiceprints.json", ".speakers.suggest.json", ".speakers.html"):
        assert not pipeline.swap_ext(media, suffix).exists()


@pytest.mark.parametrize("failed_suffix", [".json", ".vtt"])
def test_process_primary_write_failure_exits_with_no_false_sidecar(
    tmp_path, monkeypatch, failed_suffix
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    turns = [TURN]

    def fake_transcribe(_source, **_kwargs):
        return "en", [dict(UNIT)], [], [], turns, _capture(turns)

    real_replace = episode_transaction._replace_stage

    def failing_replace(stage):
        if stage.target.suffix == failed_suffix:
            raise OSError(f"injected {failed_suffix} failure")
        return real_replace(stage)

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(episode_transaction, "_replace_stage", failing_replace)

    with pytest.raises(
        episode_transaction.TransactionOperationError, match="injected"
    ) as caught:
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    assert caught.value.failure.kind == "commit-failed"
    assert caught.value.failure.detail_code == (
        "main-json-replace" if failed_suffix == ".json" else "vtt-replace"
    )
    assert not (tmp_path / "episode.voiceprints.json").exists()
    if failed_suffix == ".vtt":
        sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
        assert "voiceprint_capture" in sibling
        assert not (tmp_path / "episode.vtt").exists()
    else:
        assert not (tmp_path / "episode.json").exists()


def test_process_sidecar_write_failure_names_landed_primary_outputs(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    turns = [TURN]
    for suffix in (".speakers.suggest.json", ".speakers.html"):
        pipeline.swap_ext(media, suffix).write_text("stale", encoding="utf-8")
    voiceprints_path = artifacts.claim_paths(media).voiceprints

    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: (
            "en",
            [dict(UNIT)],
            [],
            [],
            turns,
            _capture(turns),
        ),
    )
    real_replace = episode_transaction._replace_stage

    def fail_sidecar_replace(stage):
        if stage.target == voiceprints_path:
            raise OSError("disk full")
        return real_replace(stage)

    monkeypatch.setattr(episode_transaction, "_replace_stage", fail_sidecar_replace)

    with pytest.raises(episode_transaction.TransactionOperationError) as caught:
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    assert caught.value.failure.kind == "commit-failed"
    assert caught.value.failure.detail_code == "machine-artifact-replace"
    assert caught.value.landed == (
        tmp_path / "episode.json",
        tmp_path / "episode.vtt",
    )
    assert caught.value.machine_landed == ()
    assert caught.value.leftovers == ()

    assert (tmp_path / "episode.json").exists()
    assert (tmp_path / "episode.vtt").exists()
    assert not voiceprints_path.exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()
    assert not (tmp_path / "episode.speakers.html").exists()


def test_process_unlink_failure_names_landed_outputs_and_leftover(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    suggest = tmp_path / "episode.speakers.suggest.json"
    suggest.write_text("stale", encoding="utf-8")
    turns = [TURN]
    original_unlink = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if Path(path) == suggest:
            raise OSError("permission denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: (
            "en",
            [dict(UNIT)],
            [],
            [],
            turns,
            _capture(turns),
        ),
    )
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(
        RuntimeError,
        match=r"primary JSON/VTT outputs landed.*could not delete .*suggest",
    ):
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    assert (tmp_path / "episode.json").exists()
    assert (tmp_path / "episode.vtt").exists()
    assert suggest.exists()
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


class _RawSegment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _SmoothingAnnotation:
    def itertracks(self, *, yield_label=False):
        rows = (
            (_RawSegment(0.0, 2.0), "track-a", "SPEAKER_A"),
            (_RawSegment(0.5, 0.55), "track-b", "SPEAKER_B"),
        )
        for row in rows:
            yield row if yield_label else row[:2]

    def labels(self):
        return ["SPEAKER_A", "SPEAKER_B"]


class _SmoothingPipeline:
    def __call__(self, _source, **kwargs):
        assert "return_embeddings" not in kwargs
        embeddings = np.array(
            [
                [1.0, *([0.0] * 15)],
                [0.0, 1.0, *([0.0] * 14)],
            ],
            dtype=np.float32,
        )
        annotation = _SmoothingAnnotation()
        return SimpleNamespace(
            speaker_diarization=annotation,
            exclusive_speaker_diarization=annotation,
            speaker_embeddings=embeddings,
        )


def test_smoothing_active_capture_publishes_valid_four_part_conjunction(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    wav = _stub_transcribe_tail(tmp_path, monkeypatch)
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    # diarize_turns resolves the HF token before reaching the stubbed pipeline;
    # provide one so the test does not depend on ambient credentials.
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(
        diarize, "_get_pipeline", lambda _token, _model: _SmoothingPipeline()
    )
    monkeypatch.setattr(
        diarize,
        "_build_provenance",
        lambda *_args, **_kwargs: dict(PROVENANCE),
    )
    monkeypatch.setattr(diarize, "release", lambda: None)
    monkeypatch.setenv("VOXWEAVE_VOICEPRINT_MODEL", "pyannote")

    pipeline.process(
        media,
        separate=False,
        diarize=True,
        voiceprints=True,
        shot_snap=False,
    )

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    sidecar, validated = load_voiceprints(artifacts.claim_paths(media).voiceprints)
    assert sibling["speaker_turns"] == [[0.0, 2.0, "SPEAKER_A"]]
    assert set(validated.speakers) == {"SPEAKER_A"}
    validate_voiceprint_conjunction(sidecar, sibling, media_fingerprint(media))


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


def test_capture_cache_autocast_change_reseparates_and_rebinds(tmp_path, monkeypatch):
    # Same media, same checkpoint, different separation numerics: bf16 stems are not
    # the fp32 stems the companion claims, so the cached vocals cannot be reused.
    media = tmp_path / "snapshot.mkv"
    media.write_bytes(b"source")
    cache = tmp_path / "cache" / "episode.vocals.32k.flac"
    cache.parent.mkdir()
    cache.write_bytes(b"fp32 flac")
    fingerprint = media_fingerprint(media)
    publish_cache_companion(
        cache,
        media_fingerprint=fingerprint,
        separator=SEPARATOR,  # recorded as the fp32 path
    )
    wav = _stub_transcribe_tail(tmp_path, monkeypatch)
    fullband = tmp_path / "fullband.wav"
    vocals = tmp_path / "vocals.flac"
    voc32 = tmp_path / "voc32.wav"
    for path in (fullband, vocals, voc32):
        path.write_bytes(path.name.encode())
    mixed_precision = {**SEPARATOR, "autocast": "bf16"}
    separated_from: list[Path] = []

    def fake_separate(source, **kwargs):
        assert kwargs["return_separator_identity"] is True
        separated_from.append(Path(source))
        return fullband, vocals, wav, voc32, dict(mixed_precision)

    monkeypatch.setattr(backend, "separator_identity", lambda: dict(mixed_precision))
    monkeypatch.setattr(pipeline, "_separate_to_16k_32k", fake_separate)
    monkeypatch.setattr(
        pipeline,
        "_encode_flac",
        lambda _source, destination: Path(destination).write_bytes(b"bf16 flac"),
    )
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)

    pipeline.transcribe(
        media,
        separate=True,
        voiceprints=True,
        source_fingerprint=fingerprint,
        cache_vocals=cache,
    )

    assert separated_from == [media]
    companion, validated = load_cache_companion(Path(f"{cache.resolve()}.meta.json"))
    assert validated.separator.autocast == "bf16"
    validate_cache_pair(
        companion,
        cache,
        media_fingerprint=fingerprint,
        separator=mixed_precision,
    )


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
    loaded_separator = {**SEPARATOR, "checkpoint": "d" * 64}
    captured_audio_profiles: list[dict[str, object]] = []

    def fake_separate(source, **kwargs):
        assert kwargs["return_separator_identity"] is True
        separated_from.append(Path(source))
        return fullband, vocals, wav, voc32, dict(loaded_separator)

    def fake_encode(_source, destination):
        Path(destination).write_bytes(b"new bound flac")

    def fake_diarize(_wav, **kwargs):
        captured_audio_profiles.append(dict(kwargs["audio_profile"]))
        return diarize.DiarizationResult(turns=[], centroids=None, provenance={})

    monkeypatch.setattr(backend, "separator_identity", lambda: dict(SEPARATOR))
    monkeypatch.setattr(pipeline, "_separate_to_16k_32k", fake_separate)
    monkeypatch.setattr(pipeline, "_encode_flac", fake_encode)
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(diarize, "diarize_turns", fake_diarize)
    monkeypatch.setattr(diarize, "release", lambda: None)

    pipeline.transcribe(
        media,
        separate=True,
        diarize=True,
        voiceprints=True,
        source_fingerprint=fingerprint,
        cache_vocals=cache,
    )

    companion, _validated = load_cache_companion(Path(f"{cache.resolve()}.meta.json"))
    validate_cache_pair(
        companion,
        cache,
        media_fingerprint=fingerprint,
        separator=loaded_separator,
    )
    assert separated_from == [media]
    assert captured_audio_profiles == [
        {
            "separated": True,
            "normalized": False,
            "sample_rate": 16000,
            "separator": loaded_separator,
        }
    ]


def test_episode_lock_canonicalizes_parent_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    assert episode_lock_path(real / "episode.mkv") == episode_lock_path(
        alias / "episode.mkv"
    )
    lock = episode_lock_path(real / "episode.mkv")
    assert lock == artifacts.claim_paths(real / "episode.mkv").episode_lock
    assert lock.parent != real
    assert not (real / "episode.episode.lock").exists()


# --------------------------------------------------------------------------
# Decoupled voiceprint lane (dedicated embedder after diarization)
# --------------------------------------------------------------------------


class _FakeEmbeddingNetwork:
    def __init__(self):
        self.inputs = []

    def __call__(self, wave):
        import torch

        samples = wave.detach().cpu().numpy()
        self.inputs.append(samples.copy())
        row = torch.zeros((samples.shape[0], 192), dtype=torch.float32)
        row[:, 0] = 1.0
        return row


def _install_fake_voiceprint_embedder(monkeypatch):
    from voxweave import voiceembed

    requested = []
    network = _FakeEmbeddingNetwork()

    def fake_get_embedder(spec):
        requested.append(spec)
        return voiceembed.LoadedEmbedder(
            spec=spec,
            checkpoint_sha256=spec.sha256,
            network=network,
            device="cpu",
        )

    monkeypatch.setattr(voiceembed, "get_embedder", fake_get_embedder)
    return requested, network


def _run_smoothing_capture(tmp_path, monkeypatch, *, lane):
    media = tmp_path / lane / "episode.mkv"
    media.parent.mkdir()
    media.write_bytes(b"stable media bytes")
    wav = _stub_transcribe_tail(tmp_path / lane, monkeypatch)
    sf.write(wav, np.zeros(3 * 16000, dtype=np.float32), 16000)
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(
        diarize, "_get_pipeline", lambda _token, _model: _SmoothingPipeline()
    )
    monkeypatch.setattr(
        diarize,
        "_build_provenance",
        lambda *_args, **_kwargs: dict(PROVENANCE),
    )
    monkeypatch.setattr(diarize, "release", lambda: None)
    monkeypatch.setenv("VOXWEAVE_VOICEPRINT_MODEL", lane)

    pipeline.process(
        media,
        separate=False,
        diarize=True,
        voiceprints=True,
        shot_snap=False,
    )

    sibling = json.loads((media.parent / "episode.json").read_text(encoding="utf-8"))
    sidecar, validated = load_voiceprints(artifacts.claim_paths(media).voiceprints)
    validate_voiceprint_conjunction(sidecar, sibling, media_fingerprint(media))
    return sibling, sidecar, validated


def test_decoupled_capture_publishes_embedder_voiceprints_with_identical_turns(
    tmp_path, monkeypatch
):
    from voxweave import voiceembed

    requested, network = _install_fake_voiceprint_embedder(monkeypatch)
    wanted: list[bool] = []
    real_diarize = diarize.diarize_turns

    def spy_diarize(*args, **kwargs):
        wanted.append(kwargs["want_embeddings"])
        return real_diarize(*args, **kwargs)

    monkeypatch.setattr(diarize, "diarize_turns", spy_diarize)

    legacy_sibling, legacy_sidecar, _ = _run_smoothing_capture(
        tmp_path, monkeypatch, lane="pyannote"
    )
    sibling, sidecar, validated = _run_smoothing_capture(
        tmp_path, monkeypatch, lane="auto"
    )

    # The embedder lane never changes what pyannote persisted.
    assert sibling["speaker_turns"] == legacy_sibling["speaker_turns"]
    assert wanted == [True, False]
    assert requested and all(spec is voiceembed.REDIMNET2_B6 for spec in requested)
    assert len(network.inputs) == 1  # one 2 s turn -> one forward
    assert set(validated.speakers) == {"SPEAKER_A"}
    assert validated.embedding_dim == 192
    provenance = sidecar["provenance"]
    assert provenance["embedding_lane"] == "decoupled"
    assert provenance["embedding_model"] == voiceembed.REDIMNET2_B6.name
    assert provenance["embedding_checkpoint"] == voiceembed.REDIMNET2_B6.sha256
    assert provenance["embedding_recipe"] == "centroid-v1"
    assert provenance["diarization_model"] == PROVENANCE["diarization_model"]
    assert provenance["audio"] == PROVENANCE["audio"]
    assert "embedding_lane" not in legacy_sidecar["provenance"]


def _stub_decoupled_transcribe(tmp_path, monkeypatch, *, language="English"):
    wav = _stub_transcribe_tail(tmp_path, monkeypatch)
    sf.write(wav, np.zeros(5 * 16000, dtype=np.float32), 16000)
    monkeypatch.setattr(
        backend,
        "transcribe_chunks",
        lambda *_args, **_kwargs: [(language, "hello", [dict(UNIT)])],
    )
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    seen: dict[str, object] = {}

    def fake_diarize(wav_path, **kwargs):
        seen["wav"] = Path(wav_path)
        seen.update(kwargs)
        return diarize.DiarizationResult(
            turns=[(0.0, 3.0, "SPEAKER_00")],
            centroids=None,
            provenance=dict(PROVENANCE),
        )

    monkeypatch.setattr(diarize, "diarize_turns", fake_diarize)
    monkeypatch.setattr(diarize, "release", lambda: None)
    return wav, seen


def test_japanese_episode_routes_to_the_anime_voice_actor_embedder(
    tmp_path, monkeypatch
):
    from voxweave import voiceembed

    wav, seen = _stub_decoupled_transcribe(tmp_path, monkeypatch, language="Japanese")
    requested, _network = _install_fake_voiceprint_embedder(monkeypatch)
    read_from: list[Path] = []
    real_read = voiceembed.read_mono_16k

    def spy_read(path):
        read_from.append(Path(path))
        return real_read(path)

    released: list[str] = []
    monkeypatch.setattr(voiceembed, "read_mono_16k", spy_read)
    monkeypatch.setattr(voiceembed, "release", lambda: released.append("released"))
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"source")

    result = pipeline.transcribe(
        media,
        separate=False,
        diarize=True,
        voiceprints=True,
    )

    capture = result[5]
    assert capture is not None
    assert capture.turns is result[4]
    assert seen["want_embeddings"] is False
    assert read_from == [seen["wav"]] == [wav]
    assert requested and all(spec is voiceembed.ANIME_VA for spec in requested)
    assert capture.provenance["embedding_model"] == voiceembed.ANIME_VA.name
    assert released == ["released"]


def test_embedder_failure_drops_the_capture_but_keeps_the_subtitles(
    tmp_path, monkeypatch, caplog
):
    from voxweave import voiceembed

    _stub_decoupled_transcribe(tmp_path, monkeypatch)

    def unavailable(_spec):
        raise voiceembed.VoiceEmbeddingError("could not download voiceprint model")

    monkeypatch.setattr(voiceembed, "get_embedder", unavailable)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")

    with caplog.at_level("WARNING", logger="voxweave"):
        out = pipeline.process(
            media,
            separate=False,
            diarize=True,
            voiceprints=True,
            shot_snap=False,
        )

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert out.exists()
    assert sibling["speaker_turns"] == [[0.0, 3.0, "SPEAKER_00"]]
    assert "voiceprint_capture" not in sibling
    assert not artifacts.claim_paths(media).voiceprints.exists()
    assert "voiceprint capture unavailable" in caplog.text


def test_unknown_voiceprint_model_fails_before_any_audio_work(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    monkeypatch.setenv("VOXWEAVE_VOICEPRINT_MODEL", "wespeaker")
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_a, **_k: pytest.fail("must fail before transcription"),
    )

    with pytest.raises(ValueError, match="unknown voiceprint model"):
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)


def test_cli_voiceprint_model_is_validated_and_reaches_process(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from voxweave.cli import cli

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    seen: dict[str, object] = {}

    def fake_process(_media, **kwargs):
        seen.update(kwargs)
        return tmp_path / "episode.vtt"

    monkeypatch.setattr(pipeline, "process", fake_process)
    ok = CliRunner().invoke(
        cli,
        ["--diarize", "--voiceprints", "--voiceprint-model", "Anime-VA", str(media)],
    )
    bad = CliRunner().invoke(
        cli,
        ["--diarize", "--voiceprints", "--voiceprint-model", "ecapa", str(media)],
    )

    assert ok.exit_code == 0, ok.output
    assert seen["voiceprint_model"] == "anime-va-ecapa-gn"
    assert bad.exit_code != 0
    assert "unknown voiceprint model" in bad.output


def test_cli_voiceprint_model_is_validated_even_without_voiceprints(
    tmp_path, monkeypatch, caplog
):
    from click.testing import CliRunner

    from voxweave.cli import cli

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    calls: list[dict[str, object]] = []

    def fake_process(_media, **kwargs):
        calls.append(kwargs)
        return tmp_path / "episode.vtt"

    monkeypatch.setattr(pipeline, "process", fake_process)
    # install_logging(force=True) would detach caplog's root handler.
    monkeypatch.setattr("voxweave.cli.install_logging", lambda **_kwargs: None)
    bad = CliRunner().invoke(cli, ["--voiceprint-model", "ecapa", str(media)])
    assert bad.exit_code != 0
    assert "unknown voiceprint model" in bad.output
    assert calls == []

    with caplog.at_level("WARNING", logger="voxweave"):
        ok = CliRunner().invoke(cli, ["--voiceprint-model", "anime-va", str(media)])
    assert ok.exit_code == 0, ok.output
    assert calls[0]["voiceprints"] is False
    assert "--voiceprint-model has no effect" in caplog.text
    assert "add --voiceprints" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="voxweave"):
        quiet = CliRunner().invoke(cli, [str(media)])
    assert quiet.exit_code == 0, quiet.output
    assert "--voiceprint-model" not in caplog.text


# --------------------------------------------------------------------------
# Checkpoint prefetch at the start of the run
# --------------------------------------------------------------------------


def test_prefetch_failure_turns_capture_off_but_keeps_the_subtitles(
    tmp_path, monkeypatch, caplog
):
    from voxweave import voiceembed

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    seen: dict[str, object] = {}

    def unavailable(cli_value, language_iso):
        seen["prefetch"] = (cli_value, language_iso)
        raise voiceembed.VoiceEmbeddingError(
            "could not download voiceprint model redimnet2: timed out"
        )

    def fake_transcribe(source, **kwargs):
        assert "prefetch" in seen, "the prefetch must run before any audio work"
        seen["source"] = Path(source)
        seen["voiceprints"] = kwargs["voiceprints"]
        return "en", [dict(UNIT)], [(0.0, 1.0)], [], [TURN], None

    monkeypatch.setattr(voiceembed, "prefetch_checkpoints", unavailable)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        pipeline,
        "MediaSnapshot",
        lambda *_a, **_k: pytest.fail("no private snapshot without a capture"),
    )

    with caplog.at_level("WARNING", logger="voxweave"):
        out = pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)

    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert out.exists()
    assert seen["prefetch"] == (None, None)
    assert seen["source"] == media
    assert seen["voiceprints"] is False
    assert sibling["speaker_turns"] == [list(TURN)]
    assert "voiceprint_capture" not in sibling
    assert not artifacts.claim_paths(media).voiceprints.exists()
    assert "voiceprint models unavailable" in caplog.text
    assert "timed out" in caplog.text


@pytest.mark.parametrize(
    ("lang_override", "expected"),
    [
        (None, None),
        ("  ", None),
        ("ja", "ja"),
        ("Japanese", "ja"),
        ("xx-unknown", "en"),
    ],
)
def test_prefetch_follows_a_forced_language(
    tmp_path, monkeypatch, lang_override, expected
):
    from voxweave import voiceembed

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")
    calls: list[tuple[object, object]] = []

    def fake_prefetch(cli_value, language_iso):
        calls.append((cli_value, language_iso))
        return ()

    def fake_transcribe(_source, **kwargs):
        assert kwargs["voiceprints"] is True
        turns = [TURN]
        return "en", [dict(UNIT)], [], [], turns, _capture(turns)

    monkeypatch.setattr(voiceembed, "prefetch_checkpoints", fake_prefetch)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)

    pipeline.process(
        media,
        lang_override=lang_override,
        diarize=True,
        voiceprints=True,
        voiceprint_model="auto",
        shot_snap=False,
    )

    assert calls == [("auto", expected)]


def test_transcribe_prefetch_failure_skips_the_capture(tmp_path, monkeypatch, caplog):
    from voxweave import voiceembed

    _wav, seen = _stub_decoupled_transcribe(tmp_path, monkeypatch, language="Japanese")
    requested, _network = _install_fake_voiceprint_embedder(monkeypatch)
    prefetched: list[tuple[object, object]] = []

    def unavailable(cli_value, language_iso):
        prefetched.append((cli_value, language_iso))
        raise voiceembed.VoiceEmbeddingError("anime-va checkpoint is not reachable")

    monkeypatch.setattr(voiceembed, "prefetch_checkpoints", unavailable)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"source")

    with caplog.at_level("WARNING", logger="voxweave"):
        result = pipeline.transcribe(
            media,
            lang_override="ja",
            separate=False,
            diarize=True,
            voiceprints=True,
        )

    assert prefetched == [(None, "ja")]
    assert result[4] == [(0.0, 3.0, "SPEAKER_00")]
    assert result[5] is None
    assert seen["want_embeddings"] is False
    assert requested == []
    assert "voiceprint models unavailable" in caplog.text
