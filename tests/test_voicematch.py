"""Threshold, compatibility, max-dot, collision, and suggestion policy gates."""

import copy

import pytest

from voxweave import voicebase, voicematch, voicestore

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
