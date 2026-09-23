"""speakers enroll/serve against the global voice library (no models, no ffmpeg)."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from voxweave import artifacts, speakers, voicelibrary
from voxweave.voicebase import (
    canonical_turns_digest,
    media_fingerprint,
    write_voiceprints,
)
from voxweave.voicematch import load_suggest
from voxweave.voicestore import (
    EnrollmentRefusal,
    enroll_exemplar,
    new_voice_store,
    write_voice_store,
)

VECTOR = [1.0, *([0.0] * 15)]
OTHER = [0.0, 1.0, *([0.0] * 14)]
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
def _quiet_thresholds(monkeypatch):
    for name in (
        "VOXWEAVE_VOICES_ACCEPT",
        "VOXWEAVE_VOICES_SUGGEST",
        "VOXWEAVE_VOICES_MARGIN",
        "VOXWEAVE_VOICES_GLOBAL_SUGGEST",
    ):
        monkeypatch.delenv(name, raising=False)
    speakers._WARNED_LEGACY_STORES.clear()
    monkeypatch.setattr(
        speakers,
        "extract_clip",
        lambda _source, _start, _end, output: Path(output).write_bytes(b"mp3"),
    )


def _episode(
    folder: Path,
    name="episode",
    *,
    capture_digit="1",
    vectors=None,
    provenance=None,
    names=None,
):
    folder.mkdir(parents=True, exist_ok=True)
    media = folder / f"{name}.mkv"
    media.write_bytes(f"media {folder.name} {name}".encode())
    vectors = vectors or {"SPEAKER_00": list(VECTOR)}
    turns = []
    cursor = 0.0
    for label in vectors:
        turns.append([cursor, cursor + 4.0, label])
        cursor += 4.0
    fingerprint = media_fingerprint(media)
    capture = "c" + capture_digit * 32
    sibling = {
        "language": "en",
        "segments": [],
        "word_segments": [],
        "vad_speech": [[0.0, cursor]],
        "speaker_turns": turns,
        "voiceprint_capture": capture,
        "voiceprint_media": fingerprint,
    }
    (folder / f"{name}.json").write_text(json.dumps(sibling), encoding="utf-8")
    write_voiceprints(
        folder / f"{name}.voiceprints.json",
        {
            "version": 1,
            "capture_id": capture,
            "provenance": copy.deepcopy(provenance or PROVENANCE),
            "binding": {
                "turns_digest": canonical_turns_digest(turns),
                "media_fingerprint": fingerprint,
                "media_stem": name,
                "created": "2026-09-23T00:00:00Z",
            },
            "speakers": vectors,
        },
    )
    if names is not None:
        _name(media, names)
    return media


def _name(media: Path, names: dict[str, str]) -> None:
    from voxweave import pipeline

    pipeline.speakers_mapping_path(media).write_text(
        json.dumps({"version": 1, "speakers": names}), encoding="utf-8"
    )


def _library(root=None):
    root = root or voicelibrary.resolve_voices_dir().root
    with voicelibrary.library_lock(root, exclusive=False):
        return voicelibrary.read_state(root)


def test_enroll_defaults_to_the_library_scoped_by_folder(tmp_path):
    media = _episode(tmp_path / "Season 1", names={"SPEAKER_00": "Aqua"})
    mapping = artifacts.claim_paths(media).speaker_mapping.read_bytes()

    root = speakers.enroll_speaker_voices(media)

    assert root == tmp_path / ".xdg-data" / "voxweave" / "voices"
    state = _library(root)
    [(identity_id, identity)] = state.identity_map.items()
    assert identity["display_name"] == "Aqua"
    assert identity["scopes"] == ["Season 1"]
    [space] = state.spaces
    [exemplar] = state.space_exemplars(space)[identity_id]
    assert exemplar["source"]["media_path"] == str(media)
    assert exemplar["source"]["speaker_label"] == "SPEAKER_00"
    assert exemplar["source"]["episode"] == "episode"
    assert artifacts.claim_paths(media).speaker_mapping.read_bytes() == mapping
    assert not (tmp_path / "Season 1" / "voxweave.voices.json").exists()


def test_show_names_the_scope_and_voices_dir_moves_the_library(tmp_path):
    media = _episode(tmp_path / "S01", names={"SPEAKER_00": "Aqua"})
    (tmp_path / "nas").mkdir()
    root = tmp_path / "nas" / "voices"

    assert (
        speakers.enroll_speaker_voices(media, voices_dir=root, show="KonoSuba") == root
    )
    assert _library(root).identity_map
    [identity] = _library(root).identity_map.values()
    assert identity["scopes"] == ["KonoSuba"]
    assert not _library().identity_map  # the default library stays untouched


def test_an_unmounted_configured_library_is_refused_and_warned(tmp_path, caplog):
    media = _episode(tmp_path / "S01", names={"SPEAKER_00": "Aqua"})
    mount = tmp_path / "nas"  # the share is not mounted: an empty directory
    mount.mkdir()
    root = mount / "voxweave" / "voices"

    with pytest.raises(voicelibrary.VoiceLibraryError, match="not mounted"):
        speakers.enroll_speaker_voices(media, voices_dir=root)
    assert list(mount.iterdir()) == []

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        speakers.create_speaker_audition(media, voices_dir=root)
    assert "does not exist; is a network share not mounted?" in caplog.text
    assert list(mount.iterdir()) == []


def test_voices_and_voices_dir_are_exclusive(tmp_path):
    media = _episode(tmp_path / "S01", names={"SPEAKER_00": "Aqua"})
    with pytest.raises(ValueError, match="not both"):
        speakers.enroll_speaker_voices(
            media, voices=tmp_path / "v.json", voices_dir=tmp_path / "lib"
        )
    with pytest.raises(ValueError, match="not both"):
        speakers.create_speaker_audition(
            media, voices=tmp_path / "v.json", voices_dir=tmp_path / "lib"
        )


def test_other_folder_gets_a_labelled_tier_two_suggestion(tmp_path):
    first = _episode(tmp_path / "Season 1", names={"SPEAKER_00": "Aqua"})
    speakers.enroll_speaker_voices(first)
    second = _episode(tmp_path / "Season 2", capture_digit="2")

    page = speakers.create_speaker_audition(second).page

    suggest = load_suggest(artifacts.claim_paths(second).speaker_suggest)
    match = suggest["speakers"]["SPEAKER_00"]
    assert match["candidates"] == []
    assert match["decision"] == "none"
    [candidate] = match["secondary"]["candidates"]
    assert candidate["display_name"] == "Aqua"
    assert candidate["scopes"] == ["Season 1"]
    assert suggest["voices"]["show"] == "Season 2"
    assert "global_suggest" in suggest["thresholds"]
    assert 'data-tier="2"' in page
    assert "Other scopes" in page
    assert "Aqua (1.00) from Season 1 [use]" in page
    assert "This scope (Season 2)" in page
    assert "machine-suggested" not in page


def test_same_scope_is_tier_one(tmp_path):
    first = _episode(tmp_path / "a", names={"SPEAKER_00": "Aqua"})
    speakers.enroll_speaker_voices(first, show="KonoSuba")
    second = _episode(tmp_path / "b", capture_digit="2")

    speakers.create_speaker_audition(second, show="KonoSuba")

    match = load_suggest(artifacts.claim_paths(second).speaker_suggest)["speakers"][
        "SPEAKER_00"
    ]
    assert [c["display_name"] for c in match["candidates"]] == ["Aqua"]
    assert match["secondary"]["candidates"] == []


def test_using_a_suggestion_links_the_identity_across_scopes(tmp_path):
    first = _episode(tmp_path / "Season 1", names={"SPEAKER_00": "Aqua"})
    speakers.enroll_speaker_voices(first)
    second = _episode(tmp_path / "Season 2", capture_digit="2")
    speakers.create_speaker_audition(second)
    _name(second, {"SPEAKER_00": "Aqua"})

    speakers.enroll_speaker_voices(second)

    [(identity_id, identity)] = _library().identity_map.items()
    assert identity["scopes"] == ["Season 1", "Season 2"]
    [space] = _library().spaces
    assert len(_library().space_exemplars(space)[identity_id]) == 2


def test_a_bare_name_never_links_scopes(tmp_path):
    first = _episode(tmp_path / "Show A", names={"SPEAKER_00": "Alex"})
    speakers.enroll_speaker_voices(first)
    # A different "Alex" in another show, reviewed without any suggestion.
    second = _episode(
        tmp_path / "Show B",
        capture_digit="2",
        vectors={"SPEAKER_00": list(OTHER)},
        names={"SPEAKER_00": "Alex"},
    )

    speakers.enroll_speaker_voices(second)

    identities = _library().identity_map
    assert len(identities) == 2
    assert sorted(i["scopes"][0] for i in identities.values()) == ["Show A", "Show B"]


def test_same_scope_name_resolves_and_repeat_is_a_noop(tmp_path):
    first = _episode(tmp_path / "Show", names={"SPEAKER_00": "Aqua"})
    speakers.enroll_speaker_voices(first)
    second = _episode(
        tmp_path / "Show", "episode2", capture_digit="2", names={"SPEAKER_00": "Aqua"}
    )
    speakers.enroll_speaker_voices(second)
    root = voicelibrary.resolve_voices_dir().root
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    speakers.enroll_speaker_voices(second)

    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    [identity_id] = _library().identity_map
    [space] = _library().spaces
    assert len(_library().space_exemplars(space)[identity_id]) == 2


def test_re_enrolling_after_a_speaker_split_needs_replace(tmp_path):
    folder = tmp_path / "Show"
    media = _episode(folder, names={"SPEAKER_00": "Aqua"})
    speakers.enroll_speaker_voices(media)
    # A confirmed split rewrites the centroids and speaker turns of the same
    # capture (same capture id, same media).
    resplit = [0.8, 0.6, *([0.0] * 14)]
    _episode(
        folder,
        vectors={"SPEAKER_00": resplit, "SPEAKER_01": list(OTHER)},
        names={"SPEAKER_00": "Aqua", "SPEAKER_01": "Kazuma"},
    )

    with pytest.raises(EnrollmentRefusal, match="use --replace"):
        speakers.enroll_speaker_voices(media)
    speakers.enroll_speaker_voices(media, replace_episode=True)

    state = _library()
    [space] = state.spaces
    vectors = {
        state.identity_map[identity_id]["display_name"]: [
            item["vector"] for item in items
        ]
        for identity_id, items in state.space_exemplars(space).items()
    }
    assert vectors == {"Aqua": [resplit], "Kazuma": [list(OTHER)]}


def test_unresolved_space_is_refused_before_touching_the_library(tmp_path):
    provenance = copy.deepcopy(PROVENANCE)
    provenance["outer_config_sha256"] = "unresolved"
    media = _episode(
        tmp_path / "Show", provenance=provenance, names={"SPEAKER_00": "Aqua"}
    )
    with pytest.raises(EnrollmentRefusal, match="unresolved"):
        speakers.enroll_speaker_voices(media)
    assert not (voicelibrary.resolve_voices_dir().root / "identities.json").exists()


def _legacy_folder_store(folder: Path, *, name="Aqua", vector=None):
    store = new_voice_store("Example Show", PROVENANCE)
    store = enroll_exemplar(
        store,
        raw_name=name,
        capture_id="c" + "9" * 32,
        media_fingerprint="f" * 64,
        episode="prior",
        vector=vector or VECTOR,
        at="2026-09-01T00:00:00Z",
    ).store
    path = folder / "voxweave.voices.json"
    write_voice_store(path, store)
    return path, store


def test_legacy_folder_store_is_read_only_with_a_one_time_hint(tmp_path, caplog):
    media = _episode(tmp_path / "Show")
    legacy_path, _store = _legacy_folder_store(tmp_path / "Show")
    before = legacy_path.read_bytes()

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        page = speakers.create_speaker_audition(media).page
        speakers.create_speaker_audition(media)

    hints = [r for r in caplog.records if "voxweave voices import" in r.getMessage()]
    assert len(hints) == 1
    assert "Aqua (1.00) [use]" in page  # tier 1: the store lives in this folder
    assert legacy_path.read_bytes() == before
    assert not (voicelibrary.resolve_voices_dir().root / "identities.json").exists()


def test_imported_legacy_store_stops_the_hint_and_the_library_copy_wins(
    tmp_path, caplog
):
    media = _episode(tmp_path / "Show")
    legacy_path, store = _legacy_folder_store(tmp_path / "Show")
    root = voicelibrary.resolve_voices_dir().root
    with voicelibrary.library_lock(root, exclusive=True, create_parents=True):
        state = voicelibrary.read_state(
            root, spaces=[voicelibrary.space_identity(PROVENANCE)[0]]
        )
        change, _summary = voicelibrary.import_store(
            state, store, scope="Show", source_label=str(legacy_path)
        )
        voicelibrary.commit(state, change)

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        speakers.create_speaker_audition(media)

    assert "voxweave voices import" not in caplog.text
    match = load_suggest(artifacts.claim_paths(media).speaker_suggest)["speakers"][
        "SPEAKER_00"
    ]
    assert len(match["candidates"]) == 1  # not duplicated by the legacy copy


def _forget(identity_id):
    root = voicelibrary.resolve_voices_dir().root
    with voicelibrary.library_lock(root, exclusive=True):
        state = voicelibrary.read_state(root)
        change, _removed = voicelibrary.forget_identity(state, identity_id)
        voicelibrary.commit(state, change)


def test_a_review_page_older_than_a_forget_cannot_bring_the_id_back(tmp_path):
    first = _episode(tmp_path / "Season 1", names={"SPEAKER_00": "Aqua"})
    speakers.enroll_speaker_voices(first)
    [forgotten] = _library().identity_map
    second = _episode(tmp_path / "Season 2", capture_digit="2")
    speakers.create_speaker_audition(second)  # the page offers `forgotten`
    _forget(forgotten)
    _name(second, {"SPEAKER_00": "Aqua"})  # the reviewer used that suggestion

    with pytest.raises(EnrollmentRefusal, match="forgotten since"):
        speakers.enroll_speaker_voices(second)
    assert _library().identity_map == {}

    # A fresh review no longer offers the forgotten person.
    page = speakers.create_speaker_audition(second).page
    assert "Aqua (" not in page  # no suggestion button
    assert not artifacts.path_present(artifacts.claim_paths(second).speaker_suggest)


def test_a_forgotten_id_stays_out_of_the_folder_store_suggestions(tmp_path, caplog):
    media = _episode(tmp_path / "Show")
    legacy_path, store = _legacy_folder_store(tmp_path / "Show")
    [identity_id] = store["identities"]
    root = voicelibrary.resolve_voices_dir().root
    with voicelibrary.library_lock(root, exclusive=True, create_parents=True):
        state = voicelibrary.read_state(
            root, spaces=[voicelibrary.space_identity(PROVENANCE)[0]]
        )
        change, _summary = voicelibrary.import_store(
            state, store, scope="Show", source_label=str(legacy_path)
        )
        voicelibrary.commit(state, change)
    _forget(identity_id)

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        page = speakers.create_speaker_audition(media).page

    assert "Aqua (" not in page  # no suggestion button
    assert "voxweave voices import" not in caplog.text
    assert not artifacts.path_present(artifacts.claim_paths(media).speaker_suggest)


def test_a_folder_store_suggestion_is_adopted_by_its_id(tmp_path):
    media = _episode(tmp_path / "Show")
    _legacy_path, store = _legacy_folder_store(tmp_path / "Show")
    [identity_id] = store["identities"]
    speakers.create_speaker_audition(media)
    match = load_suggest(artifacts.claim_paths(media).speaker_suggest)["speakers"][
        "SPEAKER_00"
    ]
    assert [(c["identity"], c["origin"]) for c in match["candidates"]] == [
        (identity_id, "legacy")
    ]
    _name(media, {"SPEAKER_00": "Aqua"})

    speakers.enroll_speaker_voices(media)

    assert list(_library().identity_map) == [identity_id]


def test_unusable_library_leaves_the_review_manual(tmp_path, caplog):
    media = _episode(tmp_path / "Show")
    root = voicelibrary.resolve_voices_dir().root
    root.mkdir(parents=True)
    (root / "identities.json").write_text("{broken", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        audition = speakers.create_speaker_audition(media)

    assert "is unusable" in caplog.text
    assert audition.page.startswith("<!doctype html>")
    assert not artifacts.path_present(artifacts.claim_paths(media).speaker_suggest)


def test_library_scope_and_names_are_escaped_in_the_page(tmp_path):
    hostile = '<b>"Show"</b>'
    first = _episode(tmp_path / "a", names={"SPEAKER_00": "<i>Aqua</i>"})
    speakers.enroll_speaker_voices(first, show=hostile)
    second = _episode(tmp_path / "b", capture_digit="2")

    page = speakers.create_speaker_audition(second).page

    assert "<b>" not in page and "<i>" not in page
    assert "&lt;b&gt;" in page and "&lt;i&gt;Aqua&lt;/i&gt;" in page
    assert str(VECTOR) not in page


def test_manual_review_never_reads_the_library(tmp_path, monkeypatch):
    media = _episode(tmp_path / "Show")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("--manual must not resolve the voice library")

    monkeypatch.setattr(voicelibrary, "resolve_voices_dir", forbidden)
    speakers.create_speaker_audition(media, no_match=True)
