"""Threshold, compatibility, max-dot, collision, and suggestion policy gates."""

import copy

import pytest

from voxweave import config, voicebase, voicematch, voicestore

NOW = "2026-08-27T05:00:00Z"


def _unit(index=0, dim=16):
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def _provenance(*, separated=True):
    audio = {
        "separated": separated,
        "normalized": False,
        "sample_rate": 16000,
    }
    if separated:
        audio["separator"] = {
            "repo": "audio/separator",
            "file": "model.ckpt",
            "checkpoint": "blob-123",
            "config_sha256": "c" * 64,
        }
    return {
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "outer_config_sha256": "a" * 64,
        "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "embedding_checkpoint": "blob-456",
        "embedding_dim": 16,
        "audio": audio,
        "pyannote_version": "3.4.0",
        "torch_version": "2.11.0",
    }


def _store(vectors=None):
    vectors = vectors or [_unit()]
    store = voicestore.new_voice_store("Show", _provenance())
    identities = {}
    for index, vector in enumerate(vectors, start=1):
        identity_id = f"v{index:012x}"
        identities[identity_id] = {
            "display_name": f"Name {index}",
            "aliases": [],
            "exemplars": [
                {
                    "id": f"x{index:08x}",
                    "vector": vector,
                    "episode": f"ep{index}",
                    "capture_id": f"c{index:032x}",
                    "media_fingerprint": f"{index:064x}",
                    "added": NOW,
                }
            ],
        }
    store["identities"] = identities
    store["revision"] = 3
    voicestore.validate_voice_store(store)
    return store


def _thresholds(*, accept=None, suggest=0.45, margin=0.05):
    return voicematch.MatchThresholds(
        accept=accept,
        suggest=suggest,
        margin=margin,
    )


def _record(store=None, centroids=None):
    store = store or _store()
    centroids = centroids or {"SPEAKER_00": _unit()}
    thresholds = _thresholds()
    matches = voicematch.match_speakers(centroids, store, thresholds)
    compatibility = voicematch.build_compatibility_fingerprint(_provenance())
    return voicematch.build_suggest_record(
        matches,
        capture_id="c" + "f" * 32,
        voiceprints_content_digest="d" * 64,
        compatibility=compatibility,
        thresholds=thresholds,
        store_path=voicestore.canonical_store_path("voices.json"),
        store=store,
        generated=NOW,
    )


def test_default_thresholds_ship_suggest_only():
    thresholds = voicematch.parse_thresholds({})
    assert thresholds == _thresholds()
    assert thresholds.as_mapping() == {
        "accept": "off",
        "suggest": 0.45,
        "margin": 0.05,
    }


def test_threshold_parser_accepts_explicit_finite_policy():
    thresholds = voicematch.parse_thresholds(
        {
            voicematch.ENV_ACCEPT: "0.8",
            voicematch.ENV_SUGGEST: "-0.25",
            voicematch.ENV_MARGIN: "0",
        }
    )
    assert thresholds == _thresholds(accept=0.8, suggest=-0.25, margin=0.0)
    assert voicematch.parse_thresholds({voicematch.ENV_ACCEPT: " OFF "}).accept is None


@pytest.mark.parametrize(
    "env",
    [
        {voicematch.ENV_ACCEPT: "nan"},
        {voicematch.ENV_ACCEPT: "inf"},
        {voicematch.ENV_ACCEPT: "invalid"},
        {voicematch.ENV_ACCEPT: "1.01"},
        {voicematch.ENV_ACCEPT: "0.4", voicematch.ENV_SUGGEST: "0.5"},
        {voicematch.ENV_SUGGEST: "-1.01"},
        {voicematch.ENV_SUGGEST: "nan"},
        {voicematch.ENV_MARGIN: "-0.001"},
        {voicematch.ENV_MARGIN: "inf"},
    ],
)
def test_threshold_preflight_matrix_refuses_without_defaulting(env):
    with pytest.raises(voicematch.ThresholdError):
        voicematch.parse_thresholds(env)


def test_compatibility_fingerprint_is_canonical_and_torch_is_descriptive():
    provenance = _provenance()
    first = voicematch.build_compatibility_fingerprint(provenance)
    reordered = dict(reversed(list(provenance.items())))
    reordered["torch_version"] = "99.0"
    second = voicematch.build_compatibility_fingerprint(reordered)
    assert isinstance(first, voicematch.CompatibilityFingerprint)
    assert voicematch.compatibility_equal(first, second)

    changed = copy.deepcopy(provenance)
    changed["pyannote_version"] = "3.5.0"
    assert not voicematch.compatibility_equal(
        first, voicematch.build_compatibility_fingerprint(changed)
    )


def test_raw_profile_is_resolved_without_separator():
    result = voicematch.build_compatibility_fingerprint(_provenance(separated=False))
    assert isinstance(result, voicematch.CompatibilityFingerprint)


@pytest.mark.parametrize(
    "field_path",
    [
        ("diarization_model",),
        ("outer_config_sha256",),
        ("embedding_model",),
        ("embedding_checkpoint",),
        ("embedding_dim",),
        ("pyannote_version",),
        ("audio", "sample_rate"),
        ("audio", "separator", "repo"),
        ("audio", "separator", "file"),
        ("audio", "separator", "checkpoint"),
        ("audio", "separator", "config_sha256"),
    ],
)
def test_any_unresolved_strict_component_makes_compatibility_typed_unknown(
    field_path,
):
    provenance = copy.deepcopy(_provenance())
    target = provenance
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = "unresolved"
    result = voicematch.build_compatibility_fingerprint(provenance)
    assert isinstance(result, voicematch.CompatibilityUnknown)
    assert ".".join(field_path) in result.unresolved_fields


def test_equal_unknowns_never_establish_compatibility():
    unknown = voicematch.build_compatibility_fingerprint(
        {**_provenance(), "outer_config_sha256": "unresolved"}
    )
    other = voicematch.build_compatibility_fingerprint(
        {**_provenance(), "outer_config_sha256": "unresolved"}
    )
    assert isinstance(unknown, voicematch.CompatibilityUnknown)
    assert unknown != unknown
    assert not voicematch.compatibility_equal(unknown, other)
    with pytest.raises(voicematch.CompatibilityError, match="unknown"):
        voicematch.require_known_compatibility(unknown)


def test_compatibility_requires_literal_runtime_profile_types():
    provenance = _provenance()
    provenance["audio"]["separated"] = 1
    with pytest.raises(voicematch.CompatibilityError, match="booleans"):
        voicematch.build_compatibility_fingerprint(provenance)
    provenance = _provenance()
    del provenance["audio"]["separator"]
    with pytest.raises(voicematch.CompatibilityError, match="audio.separator"):
        voicematch.build_compatibility_fingerprint(provenance)
    provenance = _provenance()
    provenance["torch_version"] = "x" * 513
    with pytest.raises(voicematch.CompatibilityError, match="torch_version"):
        voicematch.build_compatibility_fingerprint(provenance)


def test_max_dot_chooses_winning_exemplar_with_stable_tie_break():
    store = _store()
    identity = store["identities"]["v000000000001"]
    identity["exemplars"].append(
        {
            "id": "x00000002",
            "vector": _unit(1),
            "episode": "ep2",
            "capture_id": f"c{2:032x}",
            "media_fingerprint": f"{2:064x}",
            "added": NOW,
        }
    )
    matches = voicematch.match_speakers({"S": _unit(1)}, store, _thresholds(suggest=0))
    assert matches["S"].candidates[0].exemplar_id == "x00000002"
    identity["exemplars"][0]["vector"] = _unit(1)
    matches = voicematch.match_speakers({"S": _unit(1)}, store, _thresholds(suggest=0))
    assert matches["S"].candidates[0].exemplar_id == "x00000001"


def test_candidate_ties_order_by_identity_id():
    store = _store([_unit(), _unit()])
    matches = voicematch.match_speakers({"S": _unit()}, store, _thresholds(suggest=0))
    assert [item.identity_id for item in matches["S"].candidates] == [
        "v000000000001",
        "v000000000002",
    ]


def test_single_identity_margin_is_vacuously_satisfied():
    matches = voicematch.match_speakers(
        {"S": _unit()},
        _store(),
        _thresholds(accept=0.9, suggest=0.4, margin=999),
    )
    assert matches["S"].decision == "prefill"


def test_top_two_margin_blocks_prefill_but_keeps_suggestion():
    second = [0.99995, 0.01] + [0.0] * 14
    norm = sum(value * value for value in second) ** 0.5
    second = [value / norm for value in second]
    matches = voicematch.match_speakers(
        {"S": _unit()},
        _store([_unit(), second]),
        _thresholds(accept=0.9, suggest=0.4, margin=0.01),
    )
    assert matches["S"].decision == "suggest"


def test_collision_demotes_every_eligible_local_id():
    matches = voicematch.match_speakers(
        {"A": _unit(), "B": _unit()},
        _store(),
        _thresholds(accept=0.9, suggest=0.4),
    )
    assert {match.decision for match in matches.values()} == {"collision"}


def test_below_suggest_local_id_is_not_collision_eligible():
    matches = voicematch.match_speakers(
        {"A": _unit(), "B": _unit(1)},
        _store(),
        _thresholds(accept=0.9, suggest=0.4),
    )
    assert matches["A"].decision == "prefill"
    assert matches["B"].decision == "none"
    assert matches["B"].candidates == ()


def test_candidate_cap_records_exact_dropped_count():
    store = _store([_unit()] * 7)
    match = voicematch.match_speakers({"S": _unit()}, store, _thresholds(suggest=0))[
        "S"
    ]
    assert len(match.candidates) == 5
    assert match.truncated == 2


def test_matching_rejects_more_than_sidecar_speaker_cap():
    centroids = {f"S{index:02}": _unit() for index in range(65)}
    with pytest.raises(voicebase.Phase2DataError, match="at most 64"):
        voicematch.match_speakers(centroids, _store(), _thresholds())


def test_all_none_run_still_builds_full_reproducible_record():
    store = _store()
    matches = voicematch.match_speakers(
        {"S": _unit(1)}, store, _thresholds(suggest=0.5)
    )
    record = voicematch.build_suggest_record(
        matches,
        capture_id="c" + "f" * 32,
        voiceprints_content_digest="d" * 64,
        compatibility=voicematch.build_compatibility_fingerprint(_provenance()),
        thresholds=_thresholds(suggest=0.5),
        store_path=voicestore.canonical_store_path("voices.json"),
        store=store,
        generated=NOW,
    )
    assert record["speakers"]["S"] == {
        "candidates": [],
        "truncated": 0,
        "decision": "none",
    }
    assert record["capture_id"] == "c" + "f" * 32
    assert record["voiceprints_digest"] == "d" * 64
    assert record["voices"]["revision"] == 3
    assert record["voices"]["content_digest"] == voicestore.voice_store_digest(store)


def test_suggest_record_bytes_are_deterministic_across_input_order():
    store = _store([_unit(), _unit(1)])
    thresholds = _thresholds(suggest=0)
    forward = voicematch.match_speakers(
        {"B": _unit(1), "A": _unit()}, store, thresholds
    )
    reverse = voicematch.match_speakers(
        {"A": _unit(), "B": _unit(1)}, store, thresholds
    )
    kwargs = {
        "capture_id": "c" + "f" * 32,
        "voiceprints_content_digest": "d" * 64,
        "compatibility": "e" * 64,
        "thresholds": thresholds,
        "store_path": voicestore.canonical_store_path("voices.json"),
        "store": store,
        "generated": NOW,
    }
    first = voicematch.build_suggest_record(forward, **kwargs)
    second = voicematch.build_suggest_record(reverse, **kwargs)
    assert voicematch.suggest_bytes(first) == voicematch.suggest_bytes(second)


def test_suggest_write_load_delete_helpers(tmp_path):
    path = tmp_path / "ep.speakers.suggest.json"
    record = _record()
    voicematch.write_suggest(path, record)
    assert voicematch.load_suggest(path) == record
    voicematch.delete_suggest(path)
    assert not path.exists()
    voicematch.delete_suggest(path)


def test_suggest_writer_preflights_cap_without_touching_target(tmp_path):
    path = tmp_path / "ep.speakers.suggest.json"
    path.write_text("old", encoding="utf-8")
    record = _record()
    record["future"] = "x" * voicebase.SUGGEST_MAX_BYTES
    with pytest.raises(voicebase.Phase2DataError, match="encoded JSON exceeds"):
        voicematch.write_suggest(path, record)
    assert path.read_text(encoding="utf-8") == "old"


def test_suggest_validator_rejects_nondeterministic_candidates():
    record = _record(store=_store([_unit(), _unit()]))
    candidates = record["speakers"]["SPEAKER_00"]["candidates"]
    candidates.reverse()
    with pytest.raises(voicebase.Phase2DataError, match="ordered"):
        voicematch.validate_suggest_record(record)


def test_suggest_validator_never_coerces_numeric_strings():
    record = _record()
    record["thresholds"]["suggest"] = "0.45"
    with pytest.raises(voicebase.Phase2DataError, match="non-bool number"):
        voicematch.validate_suggest_record(record)
    record = _record()
    record["thresholds"]["suggest"] = 10**1000
    with pytest.raises(voicebase.Phase2DataError, match="finite range"):
        voicematch.validate_suggest_record(record)


def _mismatch(episode_provenance, store_provenance):
    return voicematch.describe_compatibility_mismatch(
        episode_provenance,
        store_provenance,
        episode=voicematch.build_compatibility_fingerprint(episode_provenance),
        store=voicematch.build_compatibility_fingerprint(store_provenance),
    )


def test_default_flip_mismatch_names_both_models_and_the_way_back():
    store_provenance = _provenance()
    store_provenance["diarization_model"] = config.LEGACY_DIARIZE_MODEL
    episode_provenance = copy.deepcopy(store_provenance)
    episode_provenance["diarization_model"] = config.DEFAULT_DIARIZE_MODEL
    episode_provenance["embedding_checkpoint"] = "blob-789"

    detail = _mismatch(episode_provenance, store_provenance)

    assert "compatibility differs" in detail
    assert config.LEGACY_DIARIZE_MODEL in detail
    assert config.DEFAULT_DIARIZE_MODEL in detail
    assert "blob-456" in detail and "blob-789" in detail
    assert "--diarize-model 3.1" in detail
    assert '[diarize].model = "3.1"' in detail
    assert "re-enroll" in detail


def test_unresolved_compatibility_keeps_its_own_wording():
    store_provenance = _provenance()
    store_provenance["diarization_model"] = config.LEGACY_DIARIZE_MODEL
    episode_provenance = copy.deepcopy(store_provenance)
    episode_provenance["diarization_model"] = config.DEFAULT_DIARIZE_MODEL
    episode_provenance["embedding_dim"] = "unresolved"

    detail = _mismatch(episode_provenance, store_provenance)

    assert "unresolved" in detail
    assert "embedding_dim" in detail
    assert "compatibility differs" not in detail
    assert "--diarize-model" not in detail


def test_recovery_hint_only_points_back_from_the_new_default():
    store_provenance = _provenance()
    store_provenance["diarization_model"] = config.DEFAULT_DIARIZE_MODEL
    episode_provenance = copy.deepcopy(store_provenance)
    episode_provenance["diarization_model"] = config.LEGACY_DIARIZE_MODEL

    detail = _mismatch(episode_provenance, store_provenance)

    assert config.DEFAULT_DIARIZE_MODEL in detail
    assert config.LEGACY_DIARIZE_MODEL in detail
    assert "--diarize-model" not in detail


def test_unnamed_provenance_difference_still_reports_a_reason():
    store_provenance = _provenance()
    episode_provenance = copy.deepcopy(store_provenance)
    episode_provenance["outer_config_sha256"] = "d" * 64

    detail = _mismatch(episode_provenance, store_provenance)

    assert "compatibility differs" in detail
    assert "pipeline configuration" in detail


def test_unknown_compatibility_cannot_build_suggest_record():
    unknown = voicematch.build_compatibility_fingerprint(
        {**_provenance(), "outer_config_sha256": "unresolved"}
    )
    with pytest.raises(voicematch.CompatibilityError, match="unknown"):
        voicematch.build_suggest_record(
            {},
            capture_id="c" + "f" * 32,
            voiceprints_content_digest="d" * 64,
            compatibility=unknown,
            thresholds=_thresholds(),
            store_path=voicestore.canonical_store_path("voices.json"),
            store=_store(),
            generated=NOW,
        )


# --------------------------------------------------------------------------
# Embedding lanes: legacy digest pin, decoupled fingerprint, reporting
# --------------------------------------------------------------------------

# Digests of legacy (pyannote-lane) provenance computed before the decoupled
# lane existed. Every sidecar and voice store in the wild fingerprints like
# this; a change here orphans all of them.
LEGACY_PLAIN = {
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "outer_config_sha256": "a" * 64,
    "embedding_model": (
        "pyannote/speaker-diarization-community-1@"
        "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee#subfolder=embedding"
    ),
    "embedding_checkpoint": (
        "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929"
    ),
    "embedding_dim": 256,
    "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
    "pyannote_version": "4.0.7",
    "torch_version": "2.11.0",
}
LEGACY_SEPARATED = {
    **LEGACY_PLAIN,
    "audio": {
        "separated": True,
        "normalized": True,
        "sample_rate": 16000,
        "separator": {
            "repo": "KimberleyJSN/melbandroformer",
            "file": "MelBandRoformer.ckpt",
            "checkpoint": "b" * 64,
            "config_sha256": "c" * 64,
        },
    },
}


@pytest.mark.parametrize(
    ("provenance", "digest"),
    [
        (
            LEGACY_PLAIN,
            "bdfcde76bf029e051a7e453b8355fcce0233c5774c15b84b4e08cf2c748a7302",
        ),
        (
            LEGACY_SEPARATED,
            "0376adb94f19be9eb102af5538c32b0e089c80f5b7972374f1092ef482b9ff2e",
        ),
    ],
)
def test_legacy_fingerprint_is_byte_identical_to_the_pre_decoupling_digest(
    provenance, digest
):
    assert voicematch.build_compatibility_fingerprint(provenance) == (
        voicematch.CompatibilityFingerprint(digest)
    )


def _decoupled(**changes):
    provenance = {
        **LEGACY_SEPARATED,
        "embedding_lane": "decoupled",
        "embedding_model": "redimnet2-b6-vb2-vox2-cnc2-lm",
        "embedding_checkpoint": "e" * 64,
        "embedding_dim": 192,
        "embedding_recipe": "centroid-v1",
    }
    provenance.update(changes)
    return provenance


@pytest.mark.parametrize(
    "changes",
    [
        {"diarization_model": config.LEGACY_DIARIZE_MODEL},
        {"outer_config_sha256": "unresolved"},
        {"pyannote_version": "5.0.0"},
        {"torch_version": "3.0.0"},
    ],
)
def test_decoupled_fingerprint_ignores_the_diarization_pipeline(changes):
    base = voicematch.build_compatibility_fingerprint(_decoupled())
    other = voicematch.build_compatibility_fingerprint(_decoupled(**changes))

    assert isinstance(base, voicematch.CompatibilityFingerprint)
    assert voicematch.compatibility_equal(base, other)


@pytest.mark.parametrize(
    "changes",
    [
        {"embedding_model": "anime-va-ecapa-gn"},
        {"embedding_checkpoint": "f" * 64},
        {"embedding_dim": 256},
        {"embedding_recipe": "centroid-v2"},
        {"audio": LEGACY_PLAIN["audio"]},
    ],
)
def test_decoupled_fingerprint_tracks_the_embedding_space(changes):
    base = voicematch.build_compatibility_fingerprint(_decoupled())
    other = voicematch.build_compatibility_fingerprint(_decoupled(**changes))

    assert isinstance(other, voicematch.CompatibilityFingerprint)
    assert not voicematch.compatibility_equal(base, other)


def test_lanes_never_share_a_fingerprint():
    legacy = voicematch.build_compatibility_fingerprint(LEGACY_SEPARATED)
    decoupled = voicematch.build_compatibility_fingerprint(_decoupled())

    assert not voicematch.compatibility_equal(legacy, decoupled)


_CLUSTERING_BLOCKS = [
    {
        "method": "voiceprint",
        "recipe": "voiceprint-v1",
        "embedder": "redimnet2-b6-vb2-vox2-cnc2-lm",
        "embedder_checkpoint": "9" * 64,
        "params": {"anchor_seconds": 1.5, "tau": 0.5},
        "audit": {"anchors": 12, "abstained_turns": 3, "abstained_seconds": 2.4},
    },
    {
        "method": "pyannote",
        "requested": "voiceprint",
        "reason": "VoiceEmbeddingError: could not download voiceprint model",
    },
]


@pytest.mark.parametrize("block", _CLUSTERING_BLOCKS)
def test_fingerprints_ignore_the_speaker_clustering_block(block):
    # Clustering regroups the turns; it never changes an embedding space, so
    # neither lane's fingerprint may read the diarization "clustering" block.
    assert voicematch.build_compatibility_fingerprint(
        {**LEGACY_SEPARATED, "clustering": block}
    ) == voicematch.CompatibilityFingerprint(
        "0376adb94f19be9eb102af5538c32b0e089c80f5b7972374f1092ef482b9ff2e"
    )
    assert voicematch.compatibility_equal(
        voicematch.build_compatibility_fingerprint(_decoupled(clustering=block)),
        voicematch.build_compatibility_fingerprint(_decoupled()),
    )


@pytest.mark.parametrize(
    "field",
    [
        "embedding_model",
        "embedding_checkpoint",
        "embedding_dim",
        "embedding_recipe",
        "audio",
    ],
)
def test_missing_decoupled_field_is_unknown(field):
    provenance = _decoupled()
    del provenance[field]

    result = voicematch.build_compatibility_fingerprint(provenance)

    assert isinstance(result, voicematch.CompatibilityUnknown)
    assert field in result.unresolved_fields


def test_unresolved_decoupled_values_are_unknown_and_malformed_ones_refused():
    unresolved = voicematch.build_compatibility_fingerprint(
        _decoupled(embedding_checkpoint="unresolved", embedding_dim="unresolved")
    )
    assert isinstance(unresolved, voicematch.CompatibilityUnknown)
    assert unresolved.unresolved_fields == (
        "embedding_checkpoint",
        "embedding_dim",
    )
    with pytest.raises(voicematch.CompatibilityError):
        voicematch.build_compatibility_fingerprint(
            _decoupled(embedding_checkpoint="not-a-sha")
        )
    with pytest.raises(voicematch.CompatibilityError):
        voicematch.build_compatibility_fingerprint(_decoupled(embedding_dim=4))


def test_unknown_embedding_lane_is_never_compatible():
    result = voicematch.build_compatibility_fingerprint(
        _decoupled(embedding_lane="future")
    )

    assert isinstance(result, voicematch.CompatibilityUnknown)
    assert result.unresolved_fields == ("embedding_lane",)


def test_legacy_store_against_decoupled_run_names_both_spaces_and_the_way_back():
    store_provenance = copy.deepcopy(LEGACY_SEPARATED)
    episode_provenance = _decoupled()

    detail = _mismatch(episode_provenance, store_provenance)

    assert "store was built with pyannote embeddings (" in detail
    assert "this run uses redimnet2-b6-vb2-vox2-cnc2-lm embeddings" in detail
    assert "--voiceprint-model pyannote" in detail
    assert '[voiceprint].model = "pyannote"' in detail
    assert "--diarize-model" not in detail


def test_legacy_31_store_against_decoupled_run_also_names_the_diarizer_way_back():
    store_provenance = {
        **copy.deepcopy(LEGACY_SEPARATED),
        "diarization_model": config.LEGACY_DIARIZE_MODEL,
    }
    episode_provenance = _decoupled(diarization_model=config.DEFAULT_DIARIZE_MODEL)

    detail = _mismatch(episode_provenance, store_provenance)

    assert "--voiceprint-model pyannote" in detail
    assert "--diarize-model 3.1" in detail


def test_decoupled_store_from_another_embedder_names_its_alias():
    store_provenance = _decoupled(
        embedding_model="anime-va-ecapa-gn", embedding_checkpoint="f" * 64
    )
    episode_provenance = _decoupled()

    detail = _mismatch(episode_provenance, store_provenance)

    assert "store was built with anime-va-ecapa-gn embeddings" in detail
    assert "this run uses redimnet2-b6-vb2-vox2-cnc2-lm embeddings" in detail
    assert "--voiceprint-model anime-va" in detail
    assert "pyannote version" not in detail
    assert "diarization model" not in detail


def test_decoupled_audio_difference_is_reported_without_a_model_hint():
    store_provenance = _decoupled(audio=LEGACY_PLAIN["audio"])
    episode_provenance = _decoupled()

    detail = _mismatch(episode_provenance, store_provenance)

    assert "audio profile" in detail
    assert "store was built with" not in detail
    assert "--voiceprint-model" not in detail


def test_threshold_defaults_follow_the_embedding_space():
    from voxweave import voiceembed

    assert voicematch.threshold_defaults(None) == (0.45, 0.05)
    assert voicematch.threshold_defaults(LEGACY_SEPARATED) == (0.45, 0.05)
    assert voicematch.threshold_defaults(_decoupled()) == (
        voiceembed.REDIMNET2_B6.suggest,
        voiceembed.REDIMNET2_B6.margin,
    )
    assert voicematch.threshold_defaults(
        _decoupled(embedding_model="anime-va-ecapa-gn")
    ) == (voiceembed.ANIME_VA.suggest, voiceembed.ANIME_VA.margin)
    assert voicematch.threshold_defaults(
        _decoupled(embedding_model="future-embedder")
    ) == (0.45, 0.05)


def test_environment_thresholds_still_win_over_embedder_defaults():
    anime = _decoupled(embedding_model="anime-va-ecapa-gn")

    defaults = voicematch.parse_thresholds({}, provenance=anime)
    overridden = voicematch.parse_thresholds(
        {"VOXWEAVE_VOICES_SUGGEST": "0.6", "VOXWEAVE_VOICES_MARGIN": "0.1"},
        provenance=anime,
    )

    assert (defaults.suggest, defaults.margin) == voicematch.threshold_defaults(anime)
    assert (overridden.suggest, overridden.margin) == (0.6, 0.1)


# --------------------------------------------------------------------------
# Two-tier library matching
# --------------------------------------------------------------------------


def test_global_suggest_follows_the_space_and_is_never_looser_than_tier_one():
    from voxweave import voiceembed

    anime = _decoupled(embedding_model="anime-va-ecapa-gn")
    assert voicematch.global_suggest_default(None) == voicematch.DEFAULT_GLOBAL_SUGGEST
    assert voicematch.global_suggest_default(LEGACY_SEPARATED) == (
        voicematch.DEFAULT_GLOBAL_SUGGEST
    )
    assert voicematch.global_suggest_default(_decoupled()) == (
        voiceembed.REDIMNET2_B6.global_suggest
    )
    assert (
        voicematch.global_suggest_default(anime) == voiceembed.ANIME_VA.global_suggest
    )
    assert voicematch.parse_global_suggest({}, provenance=anime, suggest=0.35) == (
        voiceembed.ANIME_VA.global_suggest
    )
    env = {voicematch.ENV_GLOBAL_SUGGEST: "0.8"}
    assert voicematch.parse_global_suggest(env, provenance=anime, suggest=0.35) == 0.8
    # A global bar below tier 1 is raised to it rather than widening tier 2.
    low = {voicematch.ENV_GLOBAL_SUGGEST: "0.1"}
    assert voicematch.parse_global_suggest(low, suggest=0.45) == 0.45
    for raw in ("nan", "1.5", "loose"):
        with pytest.raises(voicematch.ThresholdError):
            voicematch.parse_global_suggest(
                {voicematch.ENV_GLOBAL_SUGGEST: raw}, suggest=0.45
            )


def _pool(entries):
    """``{identity: (name, vector)}`` -> library-tier identities."""
    return {
        identity_id: {
            "display_name": name,
            "exemplars": [{"id": f"x{index:08x}", "vector": vector}],
        }
        for index, (identity_id, (name, vector)) in enumerate(
            sorted(entries.items()), start=1
        )
    }


def _tiers(centroids, in_scope, other, scopes, *, accept=None, global_suggest=0.6):
    return voicematch.match_tiers(
        centroids,
        in_scope=_pool(in_scope),
        other_scopes=_pool(other),
        scopes=scopes,
        embedding_dim=16,
        thresholds=_thresholds(accept=accept),
        global_suggest=global_suggest,
    )


def test_tier_two_is_stricter_never_prefilled_and_carries_scopes():
    near = [0.7, (1 - 0.7**2) ** 0.5, *([0.0] * 14)]
    matches = _tiers(
        {"SPEAKER_00": _unit(), "SPEAKER_01": _unit(2)},
        {"v000000000001": ("Aqua", _unit())},
        {
            "v000000000002": ("Kazuma", _unit(2)),
            # 0.70 >= 0.6: suggested, but only as a secondary candidate.
            "v000000000003": ("Aqua VA", near),
        },
        {
            "v000000000001": ("Show A",),
            "v000000000002": ("Show B", "Show C"),
            "v000000000003": ("Show B",),
        },
        accept=0.9,
    )

    first = matches["SPEAKER_00"]
    assert first.decision == "prefill"
    assert [c.identity_id for c in first.candidates] == ["v000000000001"]
    assert first.candidates[0].scopes == ("Show A",)
    assert first.secondary is not None
    assert first.secondary.decision == "suggest"
    assert [c.display_name for c in first.secondary.candidates] == ["Aqua VA"]
    assert first.secondary.candidates[0].scopes == ("Show B",)

    second = matches["SPEAKER_01"]
    assert second.decision == "none"
    assert second.secondary is not None
    # A perfect cross-scope hit still only suggests: prefill is tier 1 only.
    assert second.secondary.decision == "suggest"
    assert second.secondary.candidates[0].scopes == ("Show B", "Show C")


def test_collision_is_decided_within_each_tier():
    # Equidistant from both speakers: cosine 0.707 with each.
    shared = [0.5**0.5, 0.5**0.5, *([0.0] * 14)]
    matches = _tiers(
        {"SPEAKER_A": _unit(), "SPEAKER_B": _unit(1)},
        {"v000000000001": ("A", _unit()), "v000000000002": ("B", _unit(1))},
        {"v000000000009": ("Everyone", shared)},
        {},
    )
    for local_id in ("SPEAKER_A", "SPEAKER_B"):
        match = matches[local_id]
        # Tier 1 tops differ, so tier 1 never collides...
        assert match.decision == "suggest"
        # ...while both speakers share the same tier-2 top identity.
        assert match.secondary is not None
        assert match.secondary.decision == "collision"


def _at_similarity(similarity, rng):
    """A 16-d unit vector whose dot with e0 is exactly ``similarity``."""
    tail = [rng.gauss(0.0, 1.0) for _ in range(15)]
    norm = sum(value * value for value in tail) ** 0.5
    scale = (1.0 - similarity * similarity) ** 0.5 / norm
    return [similarity, *(value * scale for value in tail)]


def test_large_library_impostors_do_not_reach_the_secondary_suggestions():
    import random

    rng = random.Random(20260923)
    impostors = {
        f"v{index:012x}": (f"Impostor {index}", _at_similarity(sim, rng))
        for index, sim in enumerate(
            (0.30 + 0.29 * rng.random() for _ in range(2000)), start=1
        )
    }
    true_id = "v0000000fffff"
    other = {**impostors, true_id: ("Same VA", _at_similarity(0.9, rng))}
    tier_one_bar = _thresholds().suggest
    passing_tier_one_bar = sum(
        1 for _name, vector in impostors.values() if vector[0] >= tier_one_bar
    )
    # Hundreds of impostors would be suggested at the same-scope bar ...
    assert passing_tier_one_bar > 100

    matches = _tiers({"SPEAKER_00": _unit()}, {}, other, {}, global_suggest=0.6)

    secondary = matches["SPEAKER_00"].secondary
    assert secondary is not None
    # ... while the stricter cross-scope bar keeps only the real voice.
    assert [c.identity_id for c in secondary.candidates] == [true_id]
    assert secondary.truncated == 0
    assert matches["SPEAKER_00"].decision == "none"


def test_tier_thresholds_must_be_ordered():
    with pytest.raises(voicematch.ThresholdError):
        _tiers({"S": _unit()}, {}, {}, {}, global_suggest=0.2)


def _library_record(matches, *, global_suggest=0.6):
    return voicematch.build_library_suggest_record(
        matches,
        capture_id="c" + "f" * 32,
        voiceprints_content_digest="d" * 64,
        compatibility="e" * 64,
        thresholds=_thresholds(),
        global_suggest=global_suggest,
        library_path=voicestore.canonical_store_path("voices"),
        scope="Show A",
        revision=4,
        content_digest="a" * 64,
        generated=NOW,
    )


def test_library_record_round_trips_tiers_and_rejects_a_prefilled_tier_two():
    matches = _tiers(
        {"SPEAKER_00": _unit()},
        {"v000000000001": ("Aqua", _unit())},
        {"v000000000002": ("Aqua VA", _unit())},
        {"v000000000001": ("Show A",), "v000000000002": ("Show B",)},
    )
    record = _library_record(matches)

    speaker = record["speakers"]["SPEAKER_00"]
    assert speaker["candidates"][0]["scopes"] == ["Show A"]
    assert speaker["secondary"]["candidates"][0]["scopes"] == ["Show B"]
    assert record["thresholds"]["global_suggest"] == 0.6
    assert record["voices"]["show"] == "Show A"
    voicematch.validate_suggest_record(record)

    tampered = copy.deepcopy(record)
    tampered["speakers"]["SPEAKER_00"]["secondary"]["decision"] = "prefill"
    with pytest.raises(voicebase.Phase2DataError, match="secondary.decision"):
        voicematch.validate_suggest_record(tampered)
    tampered = copy.deepcopy(record)
    tampered["thresholds"]["global_suggest"] = 0.1
    with pytest.raises(voicebase.Phase2DataError, match="global_suggest"):
        voicematch.validate_suggest_record(tampered)


def test_library_candidates_record_where_their_identity_came_from():
    library = {"display_name": "Aqua", "origin": "library", "exemplars": []}
    legacy = {"display_name": "Kazuma", "origin": "legacy", "exemplars": []}
    library["exemplars"] = [{"id": "x00000001", "vector": _unit()}]
    legacy["exemplars"] = [{"id": "x00000002", "vector": _unit()}]
    matches = voicematch.match_tiers(
        {"SPEAKER_00": _unit()},
        in_scope={"v000000000001": library, "v000000000002": legacy},
        other_scopes={},
        scopes={"v000000000001": ("Show A",), "v000000000002": ("Show A",)},
        embedding_dim=16,
        thresholds=_thresholds(),
        global_suggest=0.6,
    )
    record = _library_record(matches)
    candidates = record["speakers"]["SPEAKER_00"]["candidates"]
    assert {c["identity"]: c["origin"] for c in candidates} == {
        "v000000000001": "library",
        "v000000000002": "legacy",
    }
    tampered = copy.deepcopy(record)
    tampered["speakers"]["SPEAKER_00"]["candidates"][0]["origin"] = "elsewhere"
    with pytest.raises(voicebase.Phase2DataError, match="origin"):
        voicematch.validate_suggest_record(tampered)


def test_per_show_records_keep_their_single_tier_shape():
    record = _record()
    assert "secondary" not in record["speakers"]["SPEAKER_00"]
    assert "global_suggest" not in record["thresholds"]
    assert all(
        "scopes" not in candidate
        for candidate in record["speakers"]["SPEAKER_00"]["candidates"]
    )
