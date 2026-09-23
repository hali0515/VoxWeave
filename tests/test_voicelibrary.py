"""Global voice library: location, layout, transitions, locking, and import."""

from __future__ import annotations

import copy
import json
import logging
import os
import stat
import threading
from pathlib import Path

import pytest

from voxweave import voicelibrary, voicestore
from voxweave.voicebase import Phase2DataError

NOW = "2026-09-23T01:00:00Z"
LATER = "2026-09-23T02:00:00Z"
AUDIO = {"separated": False, "normalized": False, "sample_rate": 16000}
LEGACY = {
    "diarization_model": "example/diarizer",
    "outer_config_sha256": "a" * 64,
    "embedding_model": "example/embedder",
    "embedding_checkpoint": "b" * 64,
    "embedding_dim": 16,
    "audio": AUDIO,
    "pyannote_version": "3.4.0",
    "torch_version": "test",
}


def _decoupled(model="redimnet2-b6-vb2-vox2-cnc2-lm"):
    return {
        "diarization_model": "example/diarizer",
        "embedding_lane": "decoupled",
        "embedding_model": model,
        "embedding_checkpoint": "e" * 64,
        "embedding_dim": 16,
        "embedding_recipe": "centroid-v1",
        "audio": AUDIO,
        "torch_version": "test",
    }


def _unit(index=0):
    vector = [0.0] * 16
    vector[index] = 1.0
    return vector


def _source(number=1, *, episode="ep01"):
    return voicelibrary.EpisodeSource(
        media_path=f"/media/show/ep{number:02d}.mkv",
        media_fingerprint=f"{number:064x}",
        capture_id=f"c{number:032x}",
        turns_digest=f"{number + 100:064x}",
        episode=episode,
    )


class _Ids:
    """Deterministic id factories (the library's injectable randomness)."""

    def __init__(self):
        self.identity = 0
        self.exemplar = 0

    def identity_id(self):
        self.identity += 1
        return f"v{self.identity:012x}"

    def exemplar_id(self):
        self.exemplar += 1
        return f"x{self.exemplar:08x}"


def _enroll(
    root,
    entries,
    *,
    provenance=None,
    scope="Show A",
    source=None,
    ids=None,
    at=NOW,
    replace=False,
):
    provenance = provenance or _decoupled()
    ids = ids or _Ids()
    name, _fingerprint = voicelibrary.space_identity(provenance)
    with voicelibrary.library_lock(root, exclusive=True):
        state = voicelibrary.read_state(root, spaces=[name])
        change, outcomes = voicelibrary.enroll_entries(
            state,
            provenance=provenance,
            scope=scope,
            source=source or _source(),
            entries=entries,
            replace_episode=replace,
            at=at,
            identity_id_factory=ids.identity_id,
            exemplar_id_factory=ids.exemplar_id,
        )
        voicelibrary.commit(state, change)
    return outcomes


def _entry(name="Aqua", vector=None, identity_id=None, label="SPEAKER_00"):
    return voicelibrary.EnrollEntry(
        identity_id=identity_id,
        raw_name=name,
        speaker_label=label,
        vector=vector or _unit(),
    )


def _read(root, spaces=None):
    with voicelibrary.library_lock(root, exclusive=False):
        return voicelibrary.read_state(root, spaces=spaces)


def _history(root):
    path = Path(root) / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# --------------------------------------------------------------------------
# Location and layout
# --------------------------------------------------------------------------


def test_location_precedence_is_cli_env_conf_default(tmp_path, monkeypatch):
    conf = tmp_path / "conf" / "voxweave.conf"
    conf.parent.mkdir()
    conf.write_text('[voices]\ndir = "shared/voices"\n', encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    monkeypatch.setenv("VOXWEAVE_VOICES_DIR", str(tmp_path / "env"))

    cli = voicelibrary.resolve_voices_dir(tmp_path / "cli")
    assert (cli.root, cli.source) == (tmp_path / "cli", "--voices-dir")
    env = voicelibrary.resolve_voices_dir(None)
    assert env.root == tmp_path / "env"
    assert env.source == "environment VOXWEAVE_VOICES_DIR"

    monkeypatch.delenv("VOXWEAVE_VOICES_DIR")
    configured = voicelibrary.resolve_voices_dir()
    # A relative [voices].dir is relative to the config file, not the cwd.
    assert configured.root == conf.parent / "shared" / "voices"
    assert configured.source.startswith("config [voices].dir")

    conf.write_text("", encoding="utf-8")
    fallback = voicelibrary.resolve_voices_dir()
    assert fallback.root == tmp_path / ".xdg-data" / "voxweave" / "voices"
    assert fallback.source == "built-in default ($XDG_DATA_HOME)"


def test_default_ignores_a_relative_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    location = voicelibrary.default_voices_dir({"XDG_DATA_HOME": "relative/data"})
    assert (
        location.root == tmp_path / "home" / ".local" / "share" / "voxweave" / "voices"
    )
    assert location.source == "built-in default"


def test_config_voices_key_is_known_and_type_checked(tmp_path, monkeypatch, caplog):
    from voxweave import config

    conf = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    conf.write_text("[voices]\ndir = 3\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert config.conf_voices_dir() is None
    assert "wrong type" in caplog.text
    assert "unknown config key" not in caplog.text
    conf.write_text('[voices]\ndir = "~/voices"\n', encoding="utf-8")
    assert config.conf_voices_dir() == Path("~/voices").expanduser()
    assert "[voices]" in config._TEMPLATE


def test_space_names_follow_model_and_fingerprint():
    legacy_name, legacy_fp = voicelibrary.space_identity(LEGACY)
    assert legacy_name == f"pyannote-{legacy_fp[:12]}"
    name, fingerprint = voicelibrary.space_identity(_decoupled())
    assert name == f"redimnet2-b6-vb2-vox2-cnc2-lm-{fingerprint[:12]}"
    odd, _ = voicelibrary.space_identity(_decoupled("Org/Model_V2+X"))
    assert odd.startswith("org-model-v2-x-")

    unresolved = copy.deepcopy(LEGACY)
    unresolved["outer_config_sha256"] = "unresolved"
    with pytest.raises(voicelibrary.VoiceLibraryError, match="unresolved"):
        voicelibrary.space_identity(unresolved)


def test_episode_scope_is_show_or_media_folder(tmp_path):
    media = tmp_path / "Season 2" / "ep01.mkv"
    assert voicelibrary.episode_scope(media) == "Season 2"
    assert voicelibrary.episode_scope(media, "  My\tShow ") == "My Show"


# --------------------------------------------------------------------------
# Enrollment transitions and commit
# --------------------------------------------------------------------------


def test_first_enrollment_creates_private_layout_and_a_notice(tmp_path, caplog):
    root = tmp_path / "voices"
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        outcomes = _enroll(root, [_entry()])
    assert [o.outcome for o in outcomes] == ["enroll"]
    assert outcomes[0].created

    notices = [r for r in caplog.records if "voice biometrics" in r.getMessage()]
    assert len(notices) == 1
    assert "voxweave voices forget" in notices[0].getMessage()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for path in (
        root / ".library.lock",
        root / "identities.json",
        root / "history.jsonl",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    state = _read(root)
    [space_name] = state.spaces
    identity = state.identity_map["v000000000001"]
    assert identity["display_name"] == "Aqua"
    assert identity["scopes"] == ["Show A"]
    [exemplar] = state.space_exemplars(space_name)["v000000000001"]
    assert exemplar["scope"] == "Show A"
    assert exemplar["source"] == {
        "media_path": "/media/show/ep01.mkv",
        "media_fingerprint": f"{1:064x}",
        "capture_id": f"c{1:032x}",
        "turns_digest": f"{101:064x}",
        "speaker_label": "SPEAKER_00",
        "episode": "ep01",
    }
    assert state.identities["revision"] == 1
    assert state.spaces[space_name]["revision"] == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        _enroll(root, [_entry("Kazuma", _unit(1))], source=_source(2, episode="ep02"))
    assert "voice biometrics" not in caplog.text


def test_repeat_enrollment_is_a_byte_noop(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    outcomes = _enroll(root, [_entry(identity_id="v000000000001")])
    assert [o.outcome for o in outcomes] == ["noop"]
    after = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before


def test_same_episode_needs_replace_within_its_scope(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    later = voicelibrary.EpisodeSource(
        media_path="/other/ep01.mkv",
        media_fingerprint="9" * 64,
        capture_id="c" + "9" * 32,
        turns_digest=None,
        episode="ep01",
    )
    with pytest.raises(voicestore.EnrollmentRefusal, match="replace-episode"):
        _enroll(root, [_entry(identity_id="v000000000001")], source=later)
    ids = _Ids()
    ids.exemplar = 10
    outcomes = _enroll(
        root,
        [_entry(identity_id="v000000000001")],
        source=later,
        replace=True,
        ids=ids,
        at=LATER,
    )
    assert [o.outcome for o in outcomes] == ["replace"]
    state = _read(root)
    identity = state.identity_map["v000000000001"]
    assert identity["scopes"] == ["Show A"]
    assert [row["action"] for row in _history(root)] == ["create", "enroll", "replace"]


def test_an_episode_label_repeats_freely_across_scopes(tmp_path):
    # "Episode 01" exists in every season folder: the episode index is scoped.
    root = tmp_path / "voices"
    ids = _Ids()
    _enroll(root, [_entry()], source=_source(1, episode="Episode 01"), ids=ids)
    outcomes = _enroll(
        root,
        [_entry(identity_id="v000000000001")],
        source=_source(2, episode="Episode 01"),
        scope="Show B",
        ids=ids,
        at=LATER,
    )
    assert [o.outcome for o in outcomes] == ["enroll"]
    identity = _read(root).identity_map["v000000000001"]
    assert identity["scopes"] == ["Show A", "Show B"]
    assert identity["updated"] == LATER


def test_exemplar_cap_and_eviction_are_per_space(tmp_path):
    root = tmp_path / "voices"
    ids = _Ids()
    for number in range(1, voicestore.MAX_EXEMPLARS + 2):
        _enroll(
            root,
            [_entry(identity_id="v000000000001" if number > 1 else None)],
            source=_source(number, episode=f"ep{number:02d}"),
            ids=ids,
            at=f"2026-09-23T01:00:{number:02d}Z",
        )
    # The same person in another embedding space starts its own list.
    _enroll(
        root,
        [_entry(identity_id="v000000000001")],
        provenance=_decoupled("anime-va-ecapa-gn"),
        source=_source(99, episode="ep99"),
        ids=ids,
    )
    state = _read(root)
    counts = voicelibrary.identity_summary(state, "v000000000001")["exemplars"]
    assert sorted(counts.values()) == [1, voicestore.MAX_EXEMPLARS]
    redimnet = next(name for name in counts if name.startswith("redimnet2"))
    episodes = [
        item["source"]["episode"]
        for item in state.space_exemplars(redimnet)["v000000000001"]
    ]
    assert "ep01" not in episodes  # the oldest was displaced
    evictions = [row for row in _history(root) if row["action"] == "evict"]
    assert len(evictions) == 1 and evictions[0]["episode"] == "ep01"


def test_history_never_holds_vectors_or_names(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry(), _entry("Kazuma", _unit(1), label="SPEAKER_01")])
    raw = (root / "history.jsonl").read_text()
    assert "vector" not in raw
    assert "Aqua" not in raw and "Kazuma" not in raw
    with pytest.raises(voicelibrary.VoiceLibraryError, match="vector"):
        voicelibrary.history_row("enroll", NOW, detail={"vector": [1.0]})


def test_two_names_resolving_to_one_identity_are_refused(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    with pytest.raises(voicestore.EnrollmentRefusal, match="same identity|resolve"):
        _enroll(
            root,
            [
                _entry(identity_id="v000000000001"),
                _entry("Aqua 2", _unit(1), identity_id="v000000000001"),
            ],
            source=_source(2, episode="ep02"),
        )


# --------------------------------------------------------------------------
# Rename, forget, lookups
# --------------------------------------------------------------------------


def test_rename_applies_to_every_space_with_one_write(tmp_path):
    root = tmp_path / "voices"
    ids = _Ids()
    _enroll(root, [_entry()], ids=ids)
    _enroll(
        root,
        [_entry(identity_id="v000000000001")],
        provenance=_decoupled("anime-va-ecapa-gn"),
        source=_source(2, episode="ep02"),
        ids=ids,
    )
    spaces_before = {p: p.read_bytes() for p in (root / "spaces").iterdir()}
    with voicelibrary.library_lock(root, exclusive=True):
        state = voicelibrary.read_state(root, spaces=[])
        change = voicelibrary.rename_identity(
            state, "v000000000001", "Amamiya", at=LATER
        )
        voicelibrary.commit(state, change)
    assert {p: p.read_bytes() for p in (root / "spaces").iterdir()} == spaces_before
    state = _read(root)
    assert state.identity_map["v000000000001"]["display_name"] == "Amamiya"
    assert _history(root)[-1] == {
        "at": LATER,
        "action": "rename",
        "identity": "v000000000001",
    }
    with pytest.raises(voicelibrary.UnknownIdentity):
        voicelibrary.rename_identity(state, "v0000000000ff", "Nobody")


def test_forget_removes_every_trace_across_spaces(tmp_path):
    root = tmp_path / "voices"
    ids = _Ids()
    _enroll(root, [_entry(), _entry("Kazuma", _unit(1), label="SPEAKER_01")], ids=ids)
    _enroll(
        root,
        [_entry(identity_id="v000000000001")],
        provenance=_decoupled("anime-va-ecapa-gn"),
        source=_source(2, episode="ep02"),
        ids=ids,
    )
    with voicelibrary.library_lock(root, exclusive=True):
        partial = voicelibrary.read_state(root, spaces=[])
        with pytest.raises(voicelibrary.VoiceLibraryError, match="every embedding"):
            voicelibrary.forget_identity(partial, "v000000000001")
        state = voicelibrary.read_state(root)
        change, removed = voicelibrary.forget_identity(state, "v000000000001", at=LATER)
        assert not change.identities_first
        voicelibrary.commit(state, change)
    assert sorted(removed.values()) == [1, 1]
    state = _read(root)
    assert list(state.identity_map) == ["v000000000002"]
    for name in state.spaces:
        assert "v000000000001" not in state.space_exemplars(name)
    stored = b"".join(p.read_bytes() for p in root.rglob("*.json"))
    assert b"Aqua" not in stored
    row = _history(root)[-1]
    assert row["action"] == "forget" and row["identity"] == "v000000000001"
    assert "Aqua" not in json.dumps(row)


def test_find_identities_by_id_or_any_name(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry("Alex"), _entry("Alex", _unit(1), label="SPEAKER_01")])
    state = _read(root)
    # Two identities may share a display name; lookup reports both.
    assert voicelibrary.find_identities(state, " Alex ") == [
        "v000000000001",
        "v000000000002",
    ]
    assert voicelibrary.find_identities(state, "v000000000002") == ["v000000000002"]
    assert voicelibrary.find_identities(state, "nobody") == []
    assert voicelibrary.identities_named(state, "Alex", scope="Show B") == []


# --------------------------------------------------------------------------
# Locking, conflicts, and network-filesystem behaviour
# --------------------------------------------------------------------------


def test_second_writer_detects_a_change_it_did_not_read(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    # Two writers staged against the same revision; as if flock were a no-op
    # (an NFS mount without working locks), the first one commits.
    first = voicelibrary.read_state(root, spaces=[])
    second = voicelibrary.read_state(root, spaces=[])
    voicelibrary.commit(
        first, voicelibrary.rename_identity(first, "v000000000001", "First", at=LATER)
    )
    change = voicelibrary.rename_identity(second, "v000000000001", "Second", at=LATER)
    with pytest.raises(voicelibrary.LibraryConflict, match="re-run"):
        voicelibrary.commit(second, change)
    assert _read(root).identity_map["v000000000001"]["display_name"] == "First"
    assert [row["action"] for row in _history(root)].count("rename") == 1


def test_exclusive_lock_serializes_writers(tmp_path):
    root = tmp_path / "voices"
    order: list[str] = []
    held = threading.Event()
    release = threading.Event()

    def holder():
        with voicelibrary.library_lock(root, exclusive=True):
            order.append("holder")
            held.set()
            release.wait(5)
            order.append("holder-release")

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(5)
    waiter_done = threading.Event()

    def waiter():
        with voicelibrary.library_lock(root, exclusive=False):
            order.append("reader")
        waiter_done.set()

    reader = threading.Thread(target=waiter)
    reader.start()
    assert not waiter_done.wait(0.2)
    release.set()
    thread.join(5)
    reader.join(5)
    assert order == ["holder", "holder-release", "reader"]


def test_conflict_check_compares_content_not_inodes(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    state = voicelibrary.read_state(root, spaces=[])
    identities = root / "identities.json"
    # Another client rewrote the same bytes through a fresh file (new inode,
    # new mtime), as an NFS writer replacing via rename would.
    same = identities.read_bytes()
    replacement = root / "identities.json.tmp"
    replacement.write_bytes(same)
    os.replace(replacement, identities)
    change = voicelibrary.rename_identity(state, "v000000000001", "Renamed", at=LATER)
    voicelibrary.commit(state, change)
    assert _read(root).identity_map["v000000000001"]["display_name"] == "Renamed"


def test_a_space_file_of_another_full_fingerprint_is_refused(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    name, fingerprint = voicelibrary.space_identity(_decoupled())
    state = _read(root, spaces=[name])
    voicelibrary.require_same_space(state, name, fingerprint)
    with pytest.raises(voicelibrary.VoiceLibraryError, match="collision"):
        voicelibrary.require_same_space(state, name, fingerprint[:12] + "0" * 52)


def test_writes_use_rename_only_never_hard_links(tmp_path, monkeypatch):
    def no_links(*_args, **_kwargs):
        raise OSError("hard links are not available on this share")

    monkeypatch.setattr(os, "link", no_links)
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    assert _read(root).identity_map


def test_shared_lock_never_creates_a_missing_library(tmp_path):
    root = tmp_path / "absent"
    with voicelibrary.library_lock(root, exclusive=False) as held:
        assert held is False
        state = voicelibrary.read_state(root)
    assert state.identity_map == {} and state.spaces == {}
    assert not root.exists()


def test_existing_shared_directory_keeps_its_mode(tmp_path):
    root = tmp_path / "shared"
    root.mkdir(mode=0o770)
    os.chmod(root, 0o770)
    _enroll(root, [_entry()])
    assert stat.S_IMODE(root.stat().st_mode) == 0o770


def test_orphan_exemplars_and_misfiled_spaces_are_rejected(tmp_path):
    root = tmp_path / "voices"
    _enroll(root, [_entry()])
    state = _read(root)
    [name] = state.spaces
    identities = root / "identities.json"
    document = json.loads(identities.read_text())
    document["identities"] = {}
    identities.write_text(json.dumps(document))
    with pytest.raises(voicelibrary.VoiceLibraryError, match="unknown identity"):
        _read(root)

    misfiled = root / "spaces" / "pyannote-000000000000.json"
    (root / "spaces" / f"{name}.json").rename(misfiled)
    identities.write_text(json.dumps(state.identities))
    with pytest.raises(voicelibrary.VoiceLibraryError, match="holds space"):
        _read(root)


# --------------------------------------------------------------------------
# Import of a pre-library per-show store
# --------------------------------------------------------------------------


def _legacy_store(path, *, show="Example Show", names=("Aqua", "Kazuma")):
    store = voicestore.new_voice_store(show, LEGACY)
    for index, name in enumerate(names):
        store = voicestore.enroll_exemplar(
            store,
            raw_name=name,
            capture_id=f"c{index + 1:032x}",
            media_fingerprint=f"{index + 1:064x}",
            episode=f"ep{index + 1:02d}",
            vector=_unit(index),
            at=NOW,
        ).store
    voicestore.write_voice_store(path, store)
    return store


def _import(root, store, *, scope="Example Show", at=LATER):
    name, _ = voicelibrary.space_identity(store["provenance"])
    with voicelibrary.library_lock(root, exclusive=True):
        state = voicelibrary.read_state(root, spaces=[name])
        change, summary = voicelibrary.import_store(
            state, store, scope=scope, source_label="legacy.json", at=at
        )
        voicelibrary.commit(state, change)
    return change, summary


def test_import_is_idempotent_and_keeps_ids(tmp_path):
    root = tmp_path / "voices"
    store = _legacy_store(tmp_path / "legacy.json")
    change, summary = _import(root, store)
    assert (summary.identities_created, summary.exemplars_added) == (2, 2)
    state = _read(root)
    assert set(state.identity_map) == set(store["identities"])
    legacy_ids = {
        exemplar["id"]
        for identity in store["identities"].values()
        for exemplar in identity["exemplars"]
    }
    library_ids = {
        exemplar["id"]
        for items in state.space_exemplars(summary.space).values()
        for exemplar in items
    }
    assert library_ids == legacy_ids
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    again, second = _import(root, store)
    assert again.empty
    assert (second.identities_created, second.exemplars_added) == (0, 0)
    assert second.exemplars_present == 2
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert [row["action"] for row in _history(root)].count("import") == 1


def test_reimport_after_rename_keeps_the_library_name(tmp_path):
    root = tmp_path / "voices"
    store = _legacy_store(tmp_path / "legacy.json", names=("Aqua",))
    _import(root, store)
    [identity_id] = store["identities"]
    with voicelibrary.library_lock(root, exclusive=True):
        state = voicelibrary.read_state(root, spaces=[])
        voicelibrary.commit(
            state, voicelibrary.rename_identity(state, identity_id, "Amamiya")
        )
    _import(root, store, scope="Other")
    assert _read(root).identity_map[identity_id]["display_name"] == "Amamiya"
    assert _read(root).identity_map[identity_id]["scopes"] == ["Example Show"]


def test_matching_pools_split_by_scope_and_legacy_coverage(tmp_path):
    root = tmp_path / "voices"
    store = _legacy_store(tmp_path / "legacy.json")
    name, _ = voicelibrary.space_identity(LEGACY)
    state = _read(root, spaces=[name])
    empty = voicelibrary.matching_pools(state, name, "Example Show")
    pools, unimported = voicelibrary.add_legacy_store(
        empty, store, state=state, space_name=name, in_scope=True
    )
    assert unimported
    assert set(pools.in_scope) == set(store["identities"])
    assert all(scopes == ("Example Show",) for scopes in pools.scopes.values())

    _import(root, store)
    state = _read(root, spaces=[name])
    pools = voicelibrary.matching_pools(state, name, "Other Show")
    assert set(pools.other_scopes) == set(store["identities"])
    covered, unimported = voicelibrary.add_legacy_store(
        pools, store, state=state, space_name=name, in_scope=True
    )
    assert not unimported
    assert covered.in_scope == {}  # the library copy wins


def test_invalid_documents_are_library_errors(tmp_path):
    root = tmp_path / "voices"
    root.mkdir()
    (root / "identities.json").write_text('{"version": 1, "version": 1}')
    with pytest.raises(voicelibrary.VoiceLibraryError, match="duplicate"):
        _read(root)
    assert issubclass(voicelibrary.VoiceLibraryError, Phase2DataError)
