"""Voice-store schema, normalized namespace, locking, and transition law."""

import multiprocessing
from pathlib import Path

import pytest

from voxweave import voicebase, voicestore

NOW = "2026-08-27T05:00:00Z"


def _unit(index=0, dim=16):
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def _provenance(dim=16):
    return {
        "diarization_model": "repo/model",
        "embedding_dim": dim,
        "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
    }


def _exemplar(number=1, *, episode="ep01", index=0, added=NOW):
    return {
        "id": f"x{number:08x}",
        "vector": _unit(index),
        "episode": episode,
        "capture_id": f"c{number:032x}",
        "media_fingerprint": f"{number:064x}",
        "added": added,
    }


def _store(exemplars=None):
    store = voicestore.new_voice_store("Show", _provenance())
    store["revision"] = 7
    store["identities"] = {
        "v000000000001": {
            "display_name": "Aqua",
            "aliases": ["Aqua-sama"],
            "exemplars": exemplars if exemplars is not None else [_exemplar()],
        }
    }
    voicestore.validate_voice_store(store)
    return store


def _lock_worker(path, exclusive, connection):
    context = (
        voicestore.exclusive_store_lock(path)
        if exclusive
        else voicestore.shared_store_lock(path)
    )
    connection.send("started")
    with context as handle:
        connection.send(("acquired", str(handle.store_path), str(handle.lock_path)))
        connection.recv()
    connection.close()


def test_new_store_is_valid_and_revision_zero():
    store = voicestore.new_voice_store("Show", _provenance())
    validated = voicestore.validate_voice_store(store)
    assert validated.show == "Show"
    assert validated.revision == 0
    assert validated.embedding_dim == 16


@pytest.mark.parametrize("revision", [True, 1.0, "1", -1])
def test_revision_is_an_exact_nonnegative_counter(revision):
    store = _store()
    store["revision"] = revision
    with pytest.raises(voicestore.VoiceStoreError, match="revision"):
        voicestore.validate_voice_store(store)


def test_normalization_is_nfc_sanitizer_without_casefold_or_nfkc():
    assert voicestore.normalize_speaker_key("  Cafe\u0301\tStar  ") == "Café Star"
    assert voicestore.normalize_speaker_key("Aqua") != voicestore.normalize_speaker_key(
        "aqua"
    )
    assert voicestore.normalize_speaker_key("Ａ") != voicestore.normalize_speaker_key(
        "A"
    )


def test_nfc_twins_collide_across_display_and_alias_namespace():
    store = _store([])
    store["identities"] = {
        "v000000000001": {
            "display_name": "Café",
            "aliases": [],
            "exemplars": [],
        },
        "v000000000002": {
            "display_name": "Other",
            "aliases": ["Cafe\u0301"],
            "exemplars": [],
        },
    }
    with pytest.raises(voicestore.VoiceStoreError, match="namespace collision"):
        voicestore.validate_voice_store(store)


def test_case_and_compatibility_variants_can_have_distinct_owners():
    store = _store([])
    store["identities"] = {
        "v000000000001": {"display_name": "A", "aliases": [], "exemplars": []},
        "v000000000002": {"display_name": "a", "aliases": [], "exemplars": []},
        "v000000000003": {"display_name": "Ａ", "aliases": [], "exemplars": []},
    }
    voicestore.validate_voice_store(store)


def test_alias_and_display_can_repeat_only_for_the_same_owner():
    store = _store([])
    identity = store["identities"]["v000000000001"]
    identity["aliases"] = [" Aqua "]
    voicestore.validate_voice_store(store)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("identities", 65, "at most 64"),
        ("aliases", 9, "aliases exceeds 8"),
        ("exemplars", 6, "exemplars exceeds 5"),
        ("log", 1025, "at most 1024"),
    ],
)
def test_store_cardinality_bounds(field, value, message):
    store = _store([])
    if field == "identities":
        store["identities"] = {
            f"v{index:012x}": {
                "display_name": f"name-{index}",
                "aliases": [],
                "exemplars": [],
            }
            for index in range(value)
        }
    elif field == "aliases":
        store["identities"]["v000000000001"]["aliases"] = [
            f"alias-{index}" for index in range(value)
        ]
    elif field == "exemplars":
        store["identities"]["v000000000001"]["exemplars"] = [
            _exemplar(index + 1, episode=f"ep-{index}") for index in range(value)
        ]
    else:
        store["log"] = [
            {
                "at": NOW,
                "action": "enroll",
                "identity": "v000000000001",
                "exemplar": f"x{index:08x}",
                "episode": "ep",
            }
            for index in range(value)
        ]
    with pytest.raises(voicestore.VoiceStoreError, match=message):
        voicestore.validate_voice_store(store)


def test_store_rejects_duplicate_active_ids_and_normalized_episode_twins():
    first = _exemplar(1, episode="Café")
    second = _exemplar(2, episode="Cafe\u0301")
    second["id"] = first["id"]
    store = _store([])
    store["identities"]["v000000000001"]["exemplars"] = [first, second]
    with pytest.raises(voicestore.VoiceStoreError, match="duplicate active exemplar"):
        voicestore.validate_voice_store(store)
    second["id"] = "x00000002"
    with pytest.raises(voicestore.VoiceStoreError, match="duplicate episode"):
        voicestore.validate_voice_store(store)


def test_store_rejects_duplicate_capture_and_source_indexes():
    first = _exemplar(1, episode="ep1")
    second = _exemplar(2, episode="ep2")
    second["capture_id"] = first["capture_id"]
    with pytest.raises(voicestore.VoiceStoreError, match="duplicate capture"):
        voicestore.validate_voice_store(_store([first, second]))
    second["capture_id"] = "c" + "2" * 32
    second["media_fingerprint"] = first["media_fingerprint"]
    with pytest.raises(voicestore.VoiceStoreError, match="duplicate source"):
        voicestore.validate_voice_store(_store([first, second]))


def test_maximum_cardinality_and_string_bounds_round_trip(tmp_path):
    dim = 768
    store = voicestore.new_voice_store("S" * 256, _provenance(dim))
    store["provenance"]["detail"] = "P" * 512
    identities = {}
    for identity_index in range(64):
        prefix = f"D{identity_index:02d}"
        display = prefix + "d" * (256 - len(prefix))
        aliases = []
        for alias_index in range(8):
            alias_prefix = f"A{identity_index:02d}{alias_index}"
            aliases.append(alias_prefix + "a" * (256 - len(alias_prefix)))
        exemplars = []
        for exemplar_index in range(5):
            number = identity_index * 5 + exemplar_index + 1
            episode_prefix = f"E{exemplar_index}"
            exemplars.append(
                {
                    "id": f"x{number:08x}",
                    "vector": _unit(dim=dim),
                    "episode": episode_prefix + "e" * (128 - len(episode_prefix)),
                    "capture_id": f"c{number:032x}",
                    "media_fingerprint": f"{number:064x}",
                    "added": NOW,
                }
            )
        identities[f"v{identity_index:012x}"] = {
            "display_name": display,
            "aliases": aliases,
            "exemplars": exemplars,
        }
    store["identities"] = identities
    store["log"] = [
        {
            "at": NOW,
            "action": "evict",
            "identity": f"v{index % 64:012x}",
            "exemplar": f"x{index + 10000:08x}",
            "episode": "L" * 128,
        }
        for index in range(1024)
    ]
    path = tmp_path / "voxweave.voices.json"
    voicestore.write_voice_store(path, store)
    loaded, validated = voicestore.load_voice_store(path)
    assert loaded == store
    assert len(validated.identities) == 64
    assert len(loaded["log"]) == 1024
    assert path.stat().st_size < voicebase.VOICES_STORE_MAX_BYTES


def test_store_writer_byte_preflight_preserves_target(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text("existing", encoding="utf-8")
    store = _store()
    store["future"] = "x" * voicebase.VOICES_STORE_MAX_BYTES
    with pytest.raises(voicebase.Phase2DataError, match="encoded JSON exceeds"):
        voicestore.write_voice_store(path, store)
    assert path.read_text(encoding="utf-8") == "existing"


def test_content_digest_catches_manual_edit_without_revision_change():
    store = _store()
    before = voicestore.voice_store_digest(store)
    store["identities"]["v000000000001"]["display_name"] = "Aquamarine"
    after = voicestore.voice_store_digest(store)
    assert store["revision"] == 7
    assert before != after


def test_realpath_and_symlink_spellings_share_exclusive_process_lock(tmp_path):
    real = tmp_path / "voices.json"
    real.write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_lock_worker, args=(alias, True, child))
    try:
        with voicestore.exclusive_store_lock(real) as handle:
            assert handle.store_path == real
            assert handle.lock_path == Path(f"{real}.lock")
            process.start()
            assert parent.recv() == "started"
            assert not parent.poll(0.2)
        assert parent.poll(2)
        acquired = parent.recv()
        assert acquired == ("acquired", str(real), str(real) + ".lock")
        parent.send("release")
        process.join(2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_shared_store_locks_do_not_block_each_other(tmp_path):
    real = tmp_path / "voices.json"
    real.write_text("{}", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_lock_worker, args=(real, False, child))
    try:
        with voicestore.shared_store_lock(real):
            process.start()
            assert parent.recv() == "started"
            assert parent.poll(2)
            assert parent.recv()[0] == "acquired"
            parent.send("release")
        process.join(2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_symlink_writer_replaces_real_target_without_destroying_alias(tmp_path):
    real = tmp_path / "voices.json"
    real.write_text("old", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    voicestore.write_voice_store(alias, _store())
    assert alias.is_symlink()
    assert voicestore.load_voice_store(real)[0] == _store()


def test_identity_resolution_uses_display_and_alias_namespace():
    store = _store()
    assert voicestore.resolve_identity_id(store, " Aqua-sama ") == "v000000000001"
    assert voicestore.resolve_identity_id(store, "Unknown") is None


def test_fresh_identity_preserves_bounded_raw_display_name():
    store = voicestore.new_voice_store("Show", _provenance())
    result = voicestore.enroll_exemplar(
        store,
        raw_name="  Aoi  ",
        capture_id="c" + "3" * 32,
        media_fingerprint="3" * 64,
        episode=" ep01 ",
        vector=_unit(),
        at=NOW,
        identity_id_factory=lambda: "v000000000003",
        exemplar_id_factory=lambda: "x00000003",
    )
    identity = result.store["identities"]["v000000000003"]
    assert identity["display_name"] == "  Aoi  "
    assert identity["exemplars"][0]["episode"] == "ep01"
    assert result.store["revision"] == 1


def test_overlong_raw_enrollment_name_is_typed_refusal():
    with pytest.raises(voicebase.Phase2DataError, match="256 UTF-8 bytes"):
        voicestore.enroll_exemplar(
            _store(),
            raw_name="é" * 129,
            capture_id="c" + "3" * 32,
            media_fingerprint="3" * 64,
            episode="ep02",
            vector=_unit(),
        )


def test_enrollment_never_coerces_replace_flag():
    with pytest.raises(voicestore.EnrollmentRefusal, match="must be a boolean"):
        voicestore.enroll_exemplar(
            _store(),
            raw_name="Aqua",
            capture_id=f"c{2:032x}",
            media_fingerprint=f"{2:064x}",
            episode="ep02",
            vector=_unit(1),
            replace_episode=1,
        )


def test_exact_capture_repeat_is_noop_without_revision_or_log_change():
    store = _store()
    result = voicestore.enroll_exemplar(
        store,
        raw_name="Aqua",
        capture_id="c" + f"{1:032x}",
        media_fingerprint=f"{1:064x}",
        episode="ep01",
        vector=_unit(),
        at=NOW,
    )
    assert result.outcome == "noop"
    assert result.store == store
    assert result.store["revision"] == 7
    assert result.store["log"] == []


@pytest.mark.parametrize(
    ("media", "episode", "vector", "message"),
    [
        (f"{1:064x}", "ep01", _unit(1), "capture integrity"),
        (f"{9:064x}", "ep01", _unit(), "capture integrity"),
        (f"{1:064x}", "ep02", _unit(), "already enrolled as"),
    ],
)
def test_capture_hit_enforces_all_four_evidence_fields(media, episode, vector, message):
    with pytest.raises(voicestore.EnrollmentRefusal, match=message):
        voicestore.enroll_exemplar(
            _store(),
            raw_name="Aqua",
            capture_id=f"c{1:032x}",
            media_fingerprint=media,
            episode=episode,
            vector=vector,
            at=NOW,
        )


def test_fresh_capture_on_same_source_refuses_without_replace():
    with pytest.raises(voicestore.EnrollmentRefusal, match="use --replace-episode"):
        voicestore.enroll_exemplar(
            _store(),
            raw_name="Aqua",
            capture_id=f"c{2:032x}",
            media_fingerprint=f"{1:064x}",
            episode="ep01",
            vector=_unit(1),
            at=NOW,
        )


def test_fresh_capture_on_same_source_cannot_take_a_new_label_even_with_replace():
    with pytest.raises(voicestore.EnrollmentRefusal, match="same source media"):
        voicestore.enroll_exemplar(
            _store(),
            raw_name="Aqua",
            capture_id=f"c{2:032x}",
            media_fingerprint=f"{1:064x}",
            episode="ep02",
            vector=_unit(1),
            replace_episode=True,
            at=NOW,
        )


def test_same_source_replace_mints_fresh_slot_and_audits_old_new_capture():
    result = voicestore.enroll_exemplar(
        _store(),
        raw_name="Aqua",
        capture_id=f"c{2:032x}",
        media_fingerprint=f"{1:064x}",
        episode="ep01",
        vector=_unit(1),
        replace_episode=True,
        at="2026-08-27T06:00:00Z",
        exemplar_id_factory=lambda: "x00000009",
    )
    assert result.outcome == "replace"
    assert result.exemplar_id == "x00000009"
    assert result.store["revision"] == 8
    assert result.store["log"] == [
        {
            "at": "2026-08-27T06:00:00Z",
            "action": "replace",
            "identity": "v000000000001",
            "old_exemplar": "x00000001",
            "new_exemplar": "x00000009",
            "episode": "ep01",
            "old_capture": f"c{1:032x}",
            "new_capture": f"c{2:032x}",
        }
    ]


def test_existing_episode_refuses_or_replaces_different_source():
    kwargs = {
        "raw_name": "Aqua",
        "capture_id": f"c{2:032x}",
        "media_fingerprint": f"{2:064x}",
        "episode": "ep01",
        "vector": _unit(1),
        "at": NOW,
    }
    with pytest.raises(voicestore.EnrollmentRefusal, match="episode 'ep01'"):
        voicestore.enroll_exemplar(_store(), **kwargs)
    result = voicestore.enroll_exemplar(
        _store(),
        **kwargs,
        replace_episode=True,
        exemplar_id_factory=lambda: "x00000002",
    )
    assert result.outcome == "replace"


def test_fresh_evidence_enrolls_new_slot_and_log():
    result = voicestore.enroll_exemplar(
        _store(),
        raw_name="Aqua-sama",
        capture_id=f"c{2:032x}",
        media_fingerprint=f"{2:064x}",
        episode="ep02",
        vector=_unit(1),
        at=NOW,
        exemplar_id_factory=lambda: "x00000002",
    )
    assert result.outcome == "enroll"
    assert len(result.store["identities"]["v000000000001"]["exemplars"]) == 2
    assert result.store["log"][-1]["action"] == "enroll"


def test_cross_index_resolution_to_different_exemplars_refuses():
    store = _store(
        [
            _exemplar(1, episode="ep01", index=0),
            _exemplar(2, episode="ep02", index=1),
        ]
    )
    with pytest.raises(voicestore.CrossIndexRefusal, match="x00000001, x00000002"):
        voicestore.enroll_exemplar(
            store,
            raw_name="Aqua",
            capture_id=f"c{1:032x}",
            media_fingerprint=f"{2:064x}",
            episode="ep03",
            vector=_unit(),
            at=NOW,
        )
    assert store["revision"] == 7
    assert store["log"] == []


def test_sixth_enrollment_evicts_oldest_added_with_vector_free_audit():
    exemplars = [
        _exemplar(
            index + 1,
            episode=f"ep{index + 1}",
            index=index,
            added=f"2026-08-2{index + 1}T05:00:00Z",
        )
        for index in range(5)
    ]
    result = voicestore.enroll_exemplar(
        _store(exemplars),
        raw_name="Aqua",
        capture_id=f"c{6:032x}",
        media_fingerprint=f"{6:064x}",
        episode="ep6",
        vector=_unit(5),
        at=NOW,
        exemplar_id_factory=lambda: "x00000006",
    )
    assert result.evicted_exemplar_id == "x00000001"
    active = result.store["identities"]["v000000000001"]["exemplars"]
    assert {item["id"] for item in active} == {
        "x00000002",
        "x00000003",
        "x00000004",
        "x00000005",
        "x00000006",
    }
    assert [row["action"] for row in result.store["log"]] == ["evict", "enroll"]
    assert all("vector" not in row for row in result.store["log"])


def test_log_is_drop_oldest_bounded_during_real_mutation():
    store = _store()
    store["log"] = [
        {
            "at": NOW,
            "action": "enroll",
            "identity": "v000000000001",
            "exemplar": f"x{index + 100:08x}",
            "episode": f"history-{index}",
        }
        for index in range(1024)
    ]
    result = voicestore.enroll_exemplar(
        store,
        raw_name="Aqua",
        capture_id=f"c{2:032x}",
        media_fingerprint=f"{2:064x}",
        episode="ep02",
        vector=_unit(1),
        at=NOW,
        exemplar_id_factory=lambda: "x00000002",
    )
    assert len(result.store["log"]) == 1024
    assert result.store["log"][0]["episode"] == "history-1"
    assert result.store["log"][-1]["action"] == "enroll"
