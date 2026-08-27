import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from voxweave import pipeline, speakers
from voxweave.voicebase import (
    canonical_turns_digest,
    media_fingerprint,
    write_voiceprints,
)
from voxweave.voicematch import load_suggest
from voxweave.voicestore import (
    enroll_exemplar,
    exclusive_store_lock as real_exclusive_store_lock,
    load_voice_store,
    new_voice_store,
    write_voice_store,
)


VECTOR_A = [1.0, *([0.0] * 15)]
VECTOR_B = [0.0, 1.0, *([0.0] * 14)]
PROVENANCE = {
    "diarization_model": "example/diarizer",
    "outer_config_sha256": "a" * 64,
    "embedding_model": "example/embedder",
    "embedding_checkpoint": "b" * 64,
    "embedding_dim": 16,
    "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
    "pyannote_version": "3.4.0",
    "torch_version": "test",
}


@pytest.fixture(autouse=True)
def _isolate_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(tmp_path / "cache-root"))
    monkeypatch.delenv("VOXWEAVE_VOICES_ACCEPT", raising=False)
    monkeypatch.delenv("VOXWEAVE_VOICES_SUGGEST", raising=False)
    monkeypatch.delenv("VOXWEAVE_VOICES_MARGIN", raising=False)


def _episode(
    root: Path,
    *,
    capture: str,
    vector=None,
    mapping_name: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    media = root / "episode.mkv"
    media.write_bytes(f"media:{capture}".encode())
    fingerprint = media_fingerprint(media)
    turns = [[0.0, 4.0, "SPEAKER_00"]]
    sibling = {
        "language": "en",
        "segments": [],
        "word_segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
        "vad_speech": [[0.0, 4.0]],
        "speaker_turns": turns,
        "voiceprint_capture": capture,
        "voiceprint_media": fingerprint,
    }
    (root / "episode.json").write_text(
        json.dumps(sibling, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_voiceprints(
        root / "episode.voiceprints.json",
        {
            "version": 1,
            "capture_id": capture,
            "provenance": copy.deepcopy(PROVENANCE),
            "binding": {
                "turns_digest": canonical_turns_digest(turns),
                "media_fingerprint": fingerprint,
                "media_stem": "episode",
                "created": "2026-08-28T00:00:00Z",
            },
            "speakers": {"SPEAKER_00": list(vector or VECTOR_A)},
        },
    )
    if mapping_name is not None:
        (root / "episode.speakers.json").write_text(
            json.dumps({"version": 1, "speakers": {"SPEAKER_00": mapping_name}}),
            encoding="utf-8",
        )
    return media


def _store(path: Path) -> None:
    store = new_voice_store("Example Show", PROVENANCE)
    store = enroll_exemplar(
        store,
        raw_name="Aqua",
        capture_id="c" + "f" * 32,
        media_fingerprint="f" * 64,
        episode="prior",
        vector=VECTOR_A,
    ).store
    write_voice_store(path, store)


def _write_clip(_source, _start, _end, output) -> None:
    Path(output).write_bytes(b"mp3")


def test_two_speaker_generators_leave_winner_outputs_intact(tmp_path, monkeypatch):
    media = _episode(tmp_path, capture="c" + "1" * 32)
    store_path = tmp_path / "voices.json"
    _store(store_path)
    both_staged = threading.Barrier(2)
    original_load = speakers._load_generation_store

    def synchronized_load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        both_staged.wait(timeout=5)
        return result

    monkeypatch.setattr(speakers, "_load_generation_store", synchronized_load)
    monkeypatch.setattr(speakers, "extract_clip", _write_clip)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(speakers.create_speaker_audition, media, voices=store_path)
            for _ in range(2)
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except RuntimeError as exc:
                outcomes.append(exc)

    assert sum(isinstance(value, Path) for value in outcomes) == 1
    assert sum(isinstance(value, RuntimeError) for value in outcomes) == 1
    assert (tmp_path / "episode.speakers.json").exists()
    assert (tmp_path / "episode.speakers.html").exists()
    assert load_suggest(tmp_path / "episode.speakers.suggest.json")["version"] == 1


def test_process_commit_invalidates_staged_speaker_generation(tmp_path, monkeypatch):
    media = _episode(tmp_path, capture="c" + "2" * 32)
    staged = threading.Event()
    release = threading.Event()

    def blocking_clip(_source, _start, _end, output):
        staged.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test coordination timed out")
        Path(output).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "extract_clip", blocking_clip)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(speakers.create_speaker_audition, media)
        assert staged.wait(timeout=5)
        pipeline.process(
            media,
            word_segments=("en", [{"text": "new", "start": 0.0, "end": 1.0}]),
        )
        release.set()
        with pytest.raises(RuntimeError, match="input changed"):
            future.result(timeout=10)

    assert not (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.voiceprints.json").exists()
    assert "voiceprint_capture" not in json.loads(
        (tmp_path / "episode.json").read_text(encoding="utf-8")
    )


def test_speaker_generation_and_purge_serialize_as_one_episode_set(
    tmp_path, monkeypatch
):
    media = _episode(tmp_path, capture="c" + "3" * 32)
    store_path = tmp_path / "voices.json"
    _store(store_path)
    monkeypatch.setattr(speakers, "extract_clip", _write_clip)
    publish_entered = threading.Event()
    publish_release = threading.Event()
    purge_started = threading.Event()
    original_publish = speakers._publish_audition

    def blocking_publish(*args, **kwargs):
        publish_entered.set()
        if not publish_release.wait(timeout=5):
            raise RuntimeError("test coordination timed out")
        return original_publish(*args, **kwargs)

    def run_purge():
        purge_started.set()
        return speakers.purge_voiceprints(media)

    monkeypatch.setattr(speakers, "_publish_audition", blocking_publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        generation = pool.submit(
            speakers.create_speaker_audition,
            media,
            voices=store_path,
        )
        assert publish_entered.wait(timeout=5)
        purge = pool.submit(run_purge)
        assert purge_started.wait(timeout=5)
        assert not purge.done()
        publish_release.set()
        generation.result(timeout=10)
        removed = purge.result(timeout=10)

    assert len(removed) == 3
    assert (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.voiceprints.json").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()
    assert not (tmp_path / "episode.speakers.html").exists()


def test_process_commit_and_purge_serialize_as_one_episode_set(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media")
    write_entered = threading.Event()
    write_release = threading.Event()
    purge_started = threading.Event()
    original_write = pipeline._write_siblings
    turns = [(0.0, 1.0, "SPEAKER_00")]

    def fake_transcribe(_source, **_kwargs):
        return (
            "en",
            [{"text": "hello", "start": 0.0, "end": 1.0}],
            [(0.0, 1.0)],
            [],
            turns,
            pipeline.VoiceprintCapture(
                centroids={"SPEAKER_00": list(VECTOR_A)},
                provenance=copy.deepcopy(PROVENANCE),
                turns=turns,
            ),
        )

    def blocking_write(*args, **kwargs):
        write_entered.set()
        if not write_release.wait(timeout=5):
            raise RuntimeError("test coordination timed out")
        return original_write(*args, **kwargs)

    def run_purge():
        purge_started.set()
        return speakers.purge_voiceprints(media)

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "_write_siblings", blocking_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        processing = pool.submit(
            pipeline.process,
            media,
            diarize=True,
            voiceprints=True,
            shot_snap=False,
        )
        assert write_entered.wait(timeout=5)
        purge = pool.submit(run_purge)
        assert purge_started.wait(timeout=5)
        assert not purge.done()
        write_release.set()
        processing.result(timeout=10)
        removed = purge.result(timeout=10)

    assert pipeline.voiceprints_path(media) in removed
    assert not pipeline.voiceprints_path(media).exists()


def test_two_enrollments_serialize_through_real_and_symlink_store_paths(
    tmp_path, monkeypatch
):
    first = _episode(
        tmp_path / "first",
        capture="c" + "4" * 32,
        mapping_name="Aqua",
    )
    second = _episode(
        tmp_path / "second",
        capture="c" + "5" * 32,
        vector=VECTOR_B,
        mapping_name="Blaze",
    )
    store_root = tmp_path / "store"
    store_root.mkdir()
    alias_root = tmp_path / "store-alias"
    alias_root.symlink_to(store_root, target_is_directory=True)
    real_path = store_root / "voices.json"
    alias_path = alias_root / "voices.json"
    both_ready = threading.Barrier(2)

    @contextmanager
    def synchronized_exclusive(path: Path):
        both_ready.wait(timeout=5)
        with real_exclusive_store_lock(path) as handle:
            yield handle

    monkeypatch.setattr(speakers, "exclusive_store_lock", synchronized_exclusive)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(
                speakers.enroll_speaker_voices,
                first,
                voices=real_path,
                show="Example Show",
                episode="first",
            ),
            pool.submit(
                speakers.enroll_speaker_voices,
                second,
                voices=alias_path,
                show="Example Show",
                episode="second",
            ),
        )
        assert {future.result(timeout=10) for future in futures} == {real_path}

    store, validated = load_voice_store(real_path)
    assert validated.revision == 2
    assert {identity["display_name"] for identity in store["identities"].values()} == {
        "Aqua",
        "Blaze",
    }


def test_generation_shared_store_lock_blocks_enrollment_mutation(tmp_path, monkeypatch):
    generator_media = _episode(tmp_path / "generation", capture="c" + "6" * 32)
    enrollment_media = _episode(
        tmp_path / "enrollment",
        capture="c" + "7" * 32,
        vector=VECTOR_B,
        mapping_name="Blaze",
    )
    store_path = tmp_path / "voices.json"
    _store(store_path)
    monkeypatch.setattr(speakers, "extract_clip", _write_clip)
    publish_entered = threading.Event()
    publish_release = threading.Event()
    enrollment_started = threading.Event()
    mutation_reached = threading.Event()
    original_publish = speakers._publish_audition
    original_enroll = speakers.enroll_exemplar

    def blocking_publish(*args, **kwargs):
        publish_entered.set()
        if not publish_release.wait(timeout=5):
            raise RuntimeError("test coordination timed out")
        return original_publish(*args, **kwargs)

    def observed_enroll(*args, **kwargs):
        mutation_reached.set()
        return original_enroll(*args, **kwargs)

    def run_enrollment():
        enrollment_started.set()
        return speakers.enroll_speaker_voices(
            enrollment_media,
            voices=store_path,
            episode="new-episode",
        )

    monkeypatch.setattr(speakers, "_publish_audition", blocking_publish)
    monkeypatch.setattr(speakers, "enroll_exemplar", observed_enroll)
    with ThreadPoolExecutor(max_workers=2) as pool:
        generation = pool.submit(
            speakers.create_speaker_audition,
            generator_media,
            voices=store_path,
        )
        assert publish_entered.wait(timeout=5)
        enrollment = pool.submit(run_enrollment)
        assert enrollment_started.wait(timeout=5)
        assert not mutation_reached.wait(timeout=0.2)
        publish_release.set()
        generation.result(timeout=10)
        enrollment.result(timeout=10)

    store, validated = load_voice_store(store_path)
    assert validated.revision == 2
    assert {identity["display_name"] for identity in store["identities"].values()} == {
        "Aqua",
        "Blaze",
    }
