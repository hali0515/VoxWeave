"""Episode ownership: the episode lock and the artifact cache agree on the owner."""

from __future__ import annotations

from pathlib import Path

from voxweave import artifacts, pipeline, voiceepisode


def _media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"media")
    return path


def test_live_media_under_a_peeled_stem_beats_a_recorded_tagged_source(tmp_path):
    # A cache claim recorded for "episode.zh.mkv", which is gone now, while
    # "episode.mkv" is on disk: the artifact cache resolves "episode.zh.vtt" to
    # the live media, so the episode lock must do the same.
    tagged = _media(tmp_path / "episode.zh.mkv")
    artifacts.claim_paths(tagged)
    tagged.unlink()
    live = _media(tmp_path / "episode.mkv")
    subtitle = tmp_path / "episode.zh.vtt"
    subtitle.write_text("WEBVTT\n", encoding="utf-8")

    assert pipeline._artifact_owner(subtitle) == live
    assert voiceepisode._episode_owner(subtitle) == live
    assert voiceepisode.episode_lock_path(subtitle) == voiceepisode.episode_lock_path(
        live
    )


def test_a_recorded_source_still_owns_its_subtitles_when_no_media_is_on_disk(tmp_path):
    media = _media(tmp_path / "episode.mkv")
    artifacts.claim_paths(media)
    media.unlink()
    subtitle = tmp_path / "episode.zh.vtt"
    subtitle.write_text("WEBVTT\n", encoding="utf-8")

    assert pipeline._artifact_owner(subtitle) == media
    assert voiceepisode._episode_owner(subtitle) == media
