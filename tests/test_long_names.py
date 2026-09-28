"""Media names near the filesystem's name limit (NAME_MAX, 255 bytes here).

Cache names derived from the stem are shortened only when they would not fit;
deliverables beside the media are never shortened, and one that cannot fit is
refused before any work.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from voxweave import artifacts, fsio, llm_commands, mux, pipeline, speakers, vocals
from voxweave import translate as translate_engine
from voxweave.paths import OutputNameTooLongError, name_bytes, swap_ext
from voxweave.voiceepisode import episode_lock
from voxweave.voicematch import delete_suggest

# 250 ASCII bytes and 83 three-byte characters (249 bytes): the media name fits
# with a four-byte extension, and so do <stem>.json and <stem>.vtt, but
# <stem>.episode.lock does not.
LONG_STEMS = pytest.param("a" * 250, id="ascii"), pytest.param("字" * 83, id="cjk")
UNITS = [
    {"text": "hello", "start": 0.0, "end": 1.0},
    {"text": "world", "start": 1.0, "end": 2.0},
]


@pytest.fixture(autouse=True)
def _name_max_is_255(tmp_path: Path) -> None:
    if os.pathconf(tmp_path, "PC_NAME_MAX") != 255:
        pytest.skip("these names are sized for a 255-byte NAME_MAX")


def _digest(stem: str) -> str:
    return hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]


def _assert_shortened(name: str, stem: str, suffix: str) -> None:
    tail = f"--{_digest(stem)}{suffix}"
    assert name.endswith(tail)
    assert stem.startswith(name[: -len(tail)])
    assert 0 < name_bytes(name) <= 255
    name.encode("utf-8")  # no broken multi-byte character


def _cache_names(directory: Path) -> set[str]:
    return {entry.name for entry in directory.iterdir()}


def test_names_that_fit_stay_exactly_as_they_are(tmp_path: Path) -> None:
    stem = "a" * 242  # "<stem>.episode.lock" is exactly 255 bytes
    assert artifacts.fitted_name(tmp_path, stem, ".episode.lock") == (
        f"{stem}.episode.lock"
    )
    assert artifacts.fitted_name(tmp_path, "episode", ".episode.lock") == (
        "episode.episode.lock"
    )
    paths = artifacts.claim_paths(tmp_path / "episode.mkv")
    assert paths.directory == tmp_path / "cache" / "episode"
    assert paths.episode_lock.name == "episode.episode.lock"
    assert paths.align_evidence(tmp_path / "episode.vtt").name == (
        "episode.align-evidence.json"
    )


@pytest.mark.parametrize("stem", LONG_STEMS)
def test_over_long_names_are_shortened_deterministically(
    tmp_path: Path, stem: str
) -> None:
    first = artifacts.fitted_name(tmp_path, stem, ".episode.lock")
    assert first == artifacts.fitted_name(tmp_path, stem, ".episode.lock")
    _assert_shortened(first, stem, ".episode.lock")
    # The stem is cut to the last whole character that fits.
    kept = first[: -len(f"--{_digest(stem)}.episode.lock")]
    assert name_bytes(first) + len(stem[len(kept)].encode("utf-8")) > 255


@pytest.mark.parametrize("stem", LONG_STEMS)
def test_process_writes_and_rereads_a_long_named_episode(
    tmp_path: Path, stem: str
) -> None:
    media = tmp_path / f"{stem}.mkv"
    media.write_bytes(b"media")

    vtt = pipeline.process(media, word_segments=("en", [dict(u) for u in UNITS]))

    # Deliverables keep their full names beside the media.
    assert vtt == tmp_path / f"{stem}.vtt"
    assert json.loads(swap_ext(media, ".json").read_text(encoding="utf-8"))
    # The claim directory name fits and stays plain; only the lock is shortened.
    claim = tmp_path / "cache" / stem
    paths = artifacts.inspect_paths(media)
    assert paths is not None and paths.directory == claim
    _assert_shortened(paths.episode_lock.name, stem, ".episode.lock")
    names = _cache_names(claim)
    assert paths.episode_lock.name in names
    assert {"source.json", ".episode-domain.lock"} <= names

    # A second run and a replay take the same lock and write nothing new.
    pipeline.process(media, word_segments=("en", [dict(u) for u in UNITS]))
    pipeline.split(swap_ext(media, ".json"))
    assert _cache_names(claim) == names
    with episode_lock(media):
        pass
    assert artifacts.claim_paths(media) == paths
    assert artifacts.claimed_sources(tmp_path, stem) == (media,)
    assert vocals.cache_vocals_path(media) == claim / "vocals.32k.flac"


@pytest.mark.parametrize("stem", LONG_STEMS)
def test_per_subtitle_cache_files_round_trip(tmp_path: Path, stem: str) -> None:
    media = tmp_path / f"{stem}.mkv"
    media.write_bytes(b"media")
    subtitle = swap_ext(media, ".vtt")
    written = {
        ".zh.progress.json": artifacts.translation_progress_path(media, subtitle, "zh"),
        ".align-evidence.json": artifacts.align_evidence_path(media, subtitle),
        ".asrfix.json": artifacts.asrfix_audit_path(media, subtitle),
    }
    for suffix, path in written.items():
        _assert_shortened(path.name, stem, suffix)
        fsio.atomic_write_text(path, suffix, private=True)

    assert (
        artifacts.translation_progress_path(media, subtitle, "zh")
        == (written[".zh.progress.json"])
    )
    assert artifacts.translation_progress_candidates(media, subtitle, "zh") == (
        written[".zh.progress.json"],
    )
    assert (
        artifacts.align_evidence_path(media, subtitle)
        == (written[".align-evidence.json"])
    )
    assert written[".align-evidence.json"] in (
        artifacts.align_evidence_candidates(media, subtitle)
    )
    assert artifacts.asrfix_audit_path(media, subtitle) == written[".asrfix.json"]
    for suffix, path in written.items():
        assert path.read_text(encoding="utf-8") == suffix


def test_same_stem_collision_fallback_directory_is_shortened_to_fit(
    tmp_path: Path,
) -> None:
    stem = "b" * 247  # "<stem>--<8 hex>" would be 257 bytes
    first = tmp_path / f"{stem}.mkv"
    second = tmp_path / f"{stem}.mp3"
    for media in (first, second):
        media.write_bytes(b"media")

    primary = artifacts.claim_paths(first)
    fallback = artifacts.claim_paths(second)

    assert primary.directory.name == stem
    claim_digest = hashlib.sha1(second.name.encode()).hexdigest()[:8]
    _assert_shortened(fallback.directory.name, stem, f"--{claim_digest}")
    assert artifacts.inspect_paths(second) == fallback
    assert set(artifacts.claimed_sources(tmp_path, stem)) == {first, second}
    with episode_lock(second):
        pass
    _assert_shortened(fallback.episode_lock.name, stem, ".episode.lock")
    assert fallback.episode_lock.exists()


def test_readers_find_a_name_shortened_under_another_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Written on a filesystem with a 143-byte limit (eCryptfs), read where the
    # plain name would fit: the existing shortened entry is still the one used.
    stem = "c" * 130
    media = tmp_path / f"{stem}.mkv"
    subtitle = swap_ext(media, ".vtt")
    monkeypatch.setattr(artifacts, "name_limit", lambda _directory: 143)
    small = artifacts.align_evidence_path(media, subtitle)
    _assert_shortened(small.name, stem, ".align-evidence.json")
    assert name_bytes(small.name) <= 143
    small.write_text("{}", encoding="utf-8")
    monkeypatch.undo()

    assert artifacts.fitted_name(small.parent, stem, ".align-evidence.json") == (
        f"{stem}.align-evidence.json"
    )
    assert artifacts.align_evidence_path(media, subtitle) == small
    # A plain entry, where one exists, is read under its plain name.
    plain = small.with_name(f"{stem}.asrfix.json")
    plain.write_text("{}", encoding="utf-8")
    assert artifacts.asrfix_audit_path(media, subtitle) == plain


def test_process_refuses_a_deliverable_name_that_cannot_exist(tmp_path: Path) -> None:
    media = tmp_path / f"{'d' * 251}.mkv"  # 255 bytes, so <stem>.json is 256

    with pytest.raises(OutputNameTooLongError) as caught:
        pipeline.process(media, word_segments=("en", [dict(u) for u in UNITS]))

    assert caught.value.filename == str(swap_ext(media, ".json"))
    assert "256 bytes" in str(caught.value) and str(media.parent) in str(caught.value)
    assert not (tmp_path / "cache").exists()


def test_process_checks_the_sdh_sidecar_name_before_any_audio_work(
    tmp_path: Path,
) -> None:
    media = tmp_path / f"{'e' * 248}.mkv"  # <stem>.sdh.vtt is 256 bytes

    with pytest.raises(OutputNameTooLongError) as caught:
        pipeline.process(media, sdh=True)

    assert caught.value.filename == str(swap_ext(media, ".sdh.vtt"))
    assert not (tmp_path / "cache").exists()


def _long_vtt(tmp_path: Path, stem: str) -> Path:
    vtt = tmp_path / f"{stem}.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8")
    return vtt


def test_subtitle_commands_refuse_output_names_that_cannot_exist(
    tmp_path: Path,
) -> None:
    vtt = _long_vtt(tmp_path, "f" * 251)  # 255 bytes

    with pytest.raises(OutputNameTooLongError) as translated:
        llm_commands.translate(vtt, to="zh", model="unused")
    assert translated.value.filename == str(swap_ext(vtt, ".zh.vtt"))

    with pytest.raises(OutputNameTooLongError) as corrected:
        llm_commands.correct(vtt, model="unused")
    assert corrected.value.filename == str(swap_ext(vtt, ".asrfix.vtt"))

    with pytest.raises(OutputNameTooLongError) as aligned:
        pipeline.align(vtt)
    assert aligned.value.filename == str(swap_ext(vtt, ".json"))
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("stem", LONG_STEMS)
def test_speaker_audition_and_purge_of_a_long_named_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stem: str
) -> None:
    # The legacy adjacent sidecars (<stem>.speakers.suggest.json, ...) are too
    # long to exist; clearing and purging them must treat them as absent.
    media = tmp_path / f"{stem}.mkv"
    media.write_bytes(b"media")
    turns = {"speaker_turns": [[0.0, 8.0, "SPEAKER_00"]], "vad_speech": [[0.0, 8.0]]}
    swap_ext(media, ".json").write_text(json.dumps(turns), encoding="utf-8")

    def fake_extract(_media, _start, _end, output):
        Path(output).write_bytes(b"clip-bytes")

    monkeypatch.setattr(speakers, "extract_clip", fake_extract)
    audition = speakers.create_speaker_audition(media)

    paths = artifacts.claim_paths(media)
    assert audition.mapping_path == paths.speaker_mapping
    assert json.loads(paths.speaker_mapping.read_text(encoding="utf-8")) == {
        "version": 1,
        "speakers": {"SPEAKER_00": ""},
    }
    legacy_suggest = artifacts.legacy_path(media, ".speakers.suggest.json")
    delete_suggest(legacy_suggest)  # absent, not ENAMETOOLONG
    assert artifacts.fixed_candidates(
        media, ".speakers.suggest.json", "speaker_suggest"
    ) == (paths.speaker_suggest,)
    paths.speaker_split_undo.write_text("{}", encoding="utf-8")
    speakers.purge_voiceprints(media)
    assert not paths.speaker_split_undo.exists()
    assert paths.speaker_mapping.exists()  # purge keeps the reviewed names


def test_translate_resumes_through_a_shortened_progress_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stem = "字" * 80  # "<stem>.zh.vtt" fits, "<stem>.zh.progress.json" does not
    vtt = _long_vtt(tmp_path, stem)
    seen: list[Path] = []

    def fake_translate(payload, **kwargs):
        progress = Path(kwargs["progress_path"])
        seen.append(progress)
        fsio.atomic_write_text(progress, "{}", private=True)
        return {int(item["i"]): "你好" for item in payload}

    monkeypatch.setattr(translate_engine, "translate_cues", fake_translate)
    out = llm_commands.translate(vtt, to="zh", model="fake-model")

    assert out == tmp_path / f"{stem}.zh.vtt"
    assert "你好" in out.read_text(encoding="utf-8")
    (progress,) = seen
    _assert_shortened(progress.name, stem, ".zh.progress.json")
    assert progress.parent == tmp_path / "cache" / stem
    assert not progress.exists()  # removed once the translation landed


def test_pack_and_burn_refuse_an_output_name_that_cannot_exist_before_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stem = "g" * 245
    media = tmp_path / f"{stem}.mkv"
    media.write_bytes(b"media")
    vtt = _long_vtt(tmp_path, f"{stem}.zh")

    def no_ffmpeg(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ffmpeg ran before the output name was checked")

    monkeypatch.setattr(mux, "probe_streams", no_ffmpeg)
    monkeypatch.setattr(mux, "pick_encoder", no_ffmpeg)
    with pytest.raises(OutputNameTooLongError) as packed:
        mux.pack([vtt], media=media)
    assert packed.value.filename == str(tmp_path / f"{stem}.zh.pack.mkv")
    with pytest.raises(OutputNameTooLongError) as burned:
        mux.burn(vtt, media=media)
    assert burned.value.filename == str(tmp_path / f"{stem}.zh.burn.mp4")
