from __future__ import annotations

import json
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from voxweave import artifacts, pipeline
from voxweave.voiceepisode import episode_lock, episode_lock_path


def _media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"media")
    return path


def test_claim_layout_keeps_transcript_and_deliverables_adjacent(tmp_path):
    media = _media(tmp_path / "episode.mp3")

    paths = artifacts.claim_paths(media)

    assert paths.directory == tmp_path / ".voxweave-cache/artifacts/episode"
    assert paths.speaker_mapping.name == "speakers.json"
    assert paths.speaker_suggest.name == "speakers.suggest.json"
    assert paths.voiceprints.name == "voiceprints.json"
    assert paths.episode_lock.name == "episode.episode.lock"
    assert paths.vocals_cache.name == "vocals.32k.flac"
    assert paths.align_evidence(tmp_path / "episode.vtt").name == (
        "episode.align-evidence.json"
    )
    assert paths.asrfix_audit(tmp_path / "episode.vtt").name == "episode.asrfix.json"
    assert paths.translation_progress(tmp_path / "episode.vtt", "zh").name == (
        "episode.zh.progress.json"
    )
    assert not (paths.directory / "episode.json").exists()
    assert json.loads(paths.marker.read_bytes()) == {
        "version": 1,
        "source": str(media.resolve()),
    }


def test_cache_root_is_read_at_call_time(tmp_path, monkeypatch):
    media = _media(tmp_path / "episode.mp3")
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(first))
    assert artifacts.claim_paths(media).directory.parent.parent == first
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(second))
    assert artifacts.claim_paths(media).directory.parent.parent == second


def test_same_stem_collision_uses_path_digest_fallback(tmp_path):
    first = _media(tmp_path / "one/episode.mp3")
    second = _media(tmp_path / "two/episode.mp3")

    first_paths = artifacts.claim_paths(first)
    second_paths = artifacts.claim_paths(second)

    assert first_paths.directory.name == "episode"
    assert second_paths.directory.name.startswith("episode--")
    assert len(second_paths.directory.name) == len("episode--") + 8
    digest = hashlib.sha1(
        str(second.resolve()).encode(), usedforsecurity=False
    ).hexdigest()[:8]
    assert second_paths.directory.name == f"episode--{digest}"
    assert json.loads(second_paths.marker.read_bytes())["source"] == str(
        second.resolve()
    )


def test_concurrent_claimers_choose_one_identical_directory(tmp_path):
    media = _media(tmp_path / "episode.mp3")

    with ThreadPoolExecutor(max_workers=8) as executor:
        directories = tuple(
            executor.map(
                lambda _index: artifacts.claim_paths(media).directory, range(32)
            )
        )

    assert len(set(directories)) == 1


def test_occupied_collision_fallback_fails_closed(tmp_path, monkeypatch):
    first = _media(tmp_path / "one/episode.mp3")
    second = _media(tmp_path / "two/episode.mp3")
    third = _media(tmp_path / "three/episode.mp3")
    monkeypatch.setattr(artifacts, "_claim_digest", lambda _source: "12345678")

    artifacts.claim_paths(first)
    artifacts.claim_paths(second)
    with pytest.raises(artifacts.ArtifactCollisionError):
        artifacts.claim_paths(third)


@pytest.mark.parametrize("node_kind", ["symlink", "fifo"])
def test_marker_rejects_nonregular_nodes(tmp_path, node_kind):
    media = _media(tmp_path / "episode.mp3")
    directory = artifacts.artifacts_root() / "episode"
    directory.mkdir(parents=True)
    marker = directory / "source.json"
    if node_kind == "symlink":
        victim = tmp_path / "victim"
        victim.write_text("not a marker", encoding="utf-8")
        marker.symlink_to(victim)
    else:
        os.mkfifo(marker)

    with pytest.raises(artifacts.ArtifactMarkerError):
        artifacts.claim_paths(media)


def test_marker_replacement_between_lstat_and_open_fails_closed(tmp_path, monkeypatch):
    media = _media(tmp_path / "episode.mp3")
    paths = artifacts.claim_paths(media)
    replacement = tmp_path / "replacement.json"
    replacement.write_text(paths.marker.read_text(encoding="utf-8"), encoding="utf-8")
    real_open = artifacts.os.open
    replaced = False

    def replace_then_open(path, flags, *args):
        nonlocal replaced
        if Path(path) == paths.marker and not replaced:
            replaced = True
            replacement.replace(paths.marker)
        return real_open(path, flags, *args)

    monkeypatch.setattr(artifacts.os, "open", replace_then_open)
    with pytest.raises(artifacts.ArtifactMarkerError):
        artifacts.inspect_paths(media)


def test_legacy_mapping_wins_without_inspecting_poisoned_cache(tmp_path):
    media = _media(tmp_path / "episode.mp3")
    legacy = tmp_path / "episode.speakers.json"
    legacy.write_text('{"version":1,"speakers":{}}\n', encoding="utf-8")
    directory = artifacts.artifacts_root() / "episode"
    directory.mkdir(parents=True)
    (directory / "source.json").write_text("not-json", encoding="utf-8")

    assert artifacts.speaker_mapping_path(media) == legacy
    assert artifacts.inspect_speaker_mapping_path(media) == legacy


@pytest.mark.parametrize("node_kind", ["symlink", "fifo"])
def test_cached_episode_lock_rejects_nonregular_nodes(tmp_path, node_kind):
    media = _media(tmp_path / "episode.mp3")
    lock = artifacts.claim_paths(media).episode_lock
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    original_mode = victim.stat().st_mode
    if node_kind == "symlink":
        lock.symlink_to(victim)
    else:
        os.mkfifo(lock)

    with pytest.raises(OSError):
        with episode_lock(media):
            pass

    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert victim.stat().st_mode == original_mode


def test_derived_subtitles_share_media_owner_and_lock(tmp_path):
    media = _media(tmp_path / "episode.mp4")
    expected = episode_lock_path(media)

    for name in (
        "episode.sdh.vtt",
        "episode.asrfix.vtt",
        "episode.zh.asrfix.vtt",
        "episode.asrfix.zh.vtt",
    ):
        subtitle = tmp_path / name
        subtitle.write_text("WEBVTT\n", encoding="utf-8")
        assert pipeline._artifact_owner(subtitle) == media
        assert episode_lock_path(subtitle) == expected


def test_exact_derived_stem_media_wins_and_unknown_tag_is_not_stripped(tmp_path):
    base = _media(tmp_path / "episode.mp4")
    exact = _media(tmp_path / "episode.asrfix.mp4")
    derived = tmp_path / "episode.asrfix.vtt"
    unknown = tmp_path / "episode.editorial.vtt"
    derived.write_text("WEBVTT\n", encoding="utf-8")
    unknown.write_text("WEBVTT\n", encoding="utf-8")

    assert pipeline._artifact_owner(derived) == exact
    assert pipeline._artifact_owner(unknown) == unknown
    assert pipeline._artifact_owner(unknown) != base


def test_same_stem_media_collision_uses_one_publication_lock_domain(tmp_path):
    first = _media(tmp_path / "episode.mp4")
    second = _media(tmp_path / "episode.mp3")
    unrelated = _media(tmp_path / "other/episode.mp4")
    assert artifacts.episode_domain_lock_path(
        first
    ) == artifacts.episode_domain_lock_path(second)
    assert artifacts.episode_domain_lock_path(
        first
    ) != artifacts.episode_domain_lock_path(unrelated)
    started = threading.Event()
    acquired = threading.Event()

    def take_second_lock() -> None:
        started.set()
        with episode_lock(second):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with episode_lock(first):
            future = executor.submit(take_second_lock)
            assert started.wait(1)
            assert not acquired.wait(0.05)
        future.result(timeout=1)
    assert acquired.is_set()


def test_existing_legacy_lock_and_cached_lock_are_acquired_together(tmp_path):
    media = _media(tmp_path / "episode.mp4")
    legacy = tmp_path / "episode.episode.lock"
    legacy.touch()
    started = threading.Event()
    acquired = threading.Event()

    def take_new_lock() -> None:
        started.set()
        with episode_lock(media):
            acquired.set()

    import fcntl

    descriptor = os.open(legacy, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(take_new_lock)
            assert started.wait(1)
            assert not acquired.wait(0.05)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            future.result(timeout=1)
    finally:
        os.close(descriptor)
    assert acquired.is_set()
    assert legacy.exists()


def test_cached_media_owner_remains_discoverable_after_media_is_removed(tmp_path):
    media = _media(tmp_path / "episode.mp4")
    paths = artifacts.claim_paths(media)
    paths.speaker_mapping.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":"Aoi"}}\n',
        encoding="utf-8",
    )
    transcript = tmp_path / "episode.json"
    transcript.write_text('{"language":"en","word_segments":[]}', encoding="utf-8")
    media.unlink()

    assert pipeline._artifact_owner(transcript) == media.resolve()
    assert (
        pipeline.inspect_speakers_mapping_path(transcript, reference=transcript)
        == paths.speaker_mapping
    )
    assert episode_lock_path(transcript) == paths.episode_lock


def test_transcribe_debug_sink_is_rooted_in_the_artifact_claim(tmp_path, monkeypatch):
    media = _media(tmp_path / "episode.mp4")
    captured: dict[str, object] = {}

    class CapturingSink:
        def __init__(self, stem, *, root=None, **_kwargs):
            captured["stem"] = stem
            captured["root"] = root

    class Stop(RuntimeError):
        pass

    monkeypatch.setattr(pipeline, "FileDebugSink", CapturingSink)
    monkeypatch.setattr(
        pipeline,
        "decode_to_wav",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Stop("stop after sink")),
    )

    with pytest.raises(Stop, match="stop after sink"):
        pipeline.transcribe(media, separate=False, debug=True)

    assert captured == {
        "stem": "episode",
        "root": artifacts.claim_paths(media).debug,
    }
    assert not (tmp_path / "debug").exists()
