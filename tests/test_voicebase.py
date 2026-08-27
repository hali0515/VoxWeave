"""Strict phase-2 JSON, vector, binding, fingerprint, and HTML primitives."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from voxweave import voicebase


def _unit(dim=16, index=0):
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def _sidecar(*, turns=None, media=None, speakers=None, dim=16):
    turns = turns or [[0, 1, "SPEAKER_00"]]
    media = media or ("a" * 64)
    speakers = speakers or {"SPEAKER_00": _unit(dim)}
    return {
        "version": 1,
        "capture_id": "c" + "1" * 32,
        "provenance": {
            "diarization_model": "repo/model",
            "embedding_dim": dim,
            "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
        },
        "binding": {
            "turns_digest": voicebase.canonical_turns_digest(turns),
            "media_fingerprint": media,
            "media_stem": "episode",
            "created": "2026-08-27T05:00:00Z",
        },
        "speakers": speakers,
    }


def _sibling(*, turns=None, media=None):
    return {
        "voiceprint_capture": "c" + "1" * 32,
        "voiceprint_media": media or ("a" * 64),
        "speaker_turns": turns or [[0, 1, "SPEAKER_00"]],
    }


def test_strict_json_rejects_duplicate_keys_at_any_depth():
    with pytest.raises(voicebase.DuplicateKeyError, match="duplicate JSON key: x"):
        voicebase.strict_json_loads(b'{"outer":{"x":1,"x":2}}', max_bytes=100)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_strict_json_rejects_every_nonfinite_number(literal):
    with pytest.raises(voicebase.Phase2DataError, match="non-finite"):
        voicebase.strict_json_loads(f'{{"x":{literal}}}', max_bytes=100)


def test_strict_json_checks_byte_cap_before_parse(monkeypatch):
    def must_not_parse(*_args, **_kwargs):
        raise AssertionError("oversized bytes reached the parser")

    monkeypatch.setattr(voicebase.json, "loads", must_not_parse)
    with pytest.raises(voicebase.Phase2DataError, match="exceeds"):
        voicebase.strict_json_loads(b"{} ", max_bytes=2)


def test_strict_json_rejects_invalid_utf8_and_non_object():
    with pytest.raises(voicebase.Phase2DataError, match="UTF-8"):
        voicebase.strict_json_loads(b'"\xff"', max_bytes=10)
    with pytest.raises(voicebase.Phase2DataError, match="top-level object"):
        voicebase.strict_json_object_loads(b"[]", max_bytes=10)


@pytest.mark.parametrize("version", [True, 1.0, "1", None, 2])
def test_version_one_is_an_exact_integer(version):
    with pytest.raises(voicebase.Phase2DataError, match="integer 1"):
        voicebase.require_version_one(version)


def test_canonical_json_is_key_order_independent_and_forbids_nan():
    left = {"z": [2, 1], "a": {"b": "雪"}}
    right = {"a": {"b": "雪"}, "z": [2, 1]}
    assert voicebase.canonical_json_bytes(left) == voicebase.canonical_json_bytes(right)
    assert voicebase.canonical_json_digest(left) == voicebase.canonical_json_digest(
        right
    )
    with pytest.raises(voicebase.Phase2DataError, match="strict JSON"):
        voicebase.canonical_json_bytes({"bad": float("nan")})


def test_writer_preflights_encoded_bytes_without_touching_destination(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(voicebase.Phase2DataError, match="encoded JSON exceeds"):
        voicebase.write_json_object(path, {"x": "é" * 20}, max_bytes=20)
    assert path.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_utc_timestamps_are_canonical_and_timezone_aware():
    local = timezone(timedelta(hours=10))
    now = datetime(2026, 8, 27, 15, 4, 5, 999, tzinfo=local)
    assert voicebase.utc_timestamp(now) == "2026-08-27T05:04:05Z"
    assert (
        voicebase.require_utc_timestamp("2026-08-27T05:04:05Z", "at")
        == "2026-08-27T05:04:05Z"
    )
    with pytest.raises(voicebase.Phase2DataError, match="timezone-aware"):
        voicebase.utc_timestamp(datetime(2026, 8, 27))
    with pytest.raises(voicebase.Phase2DataError, match="valid UTC"):
        voicebase.require_utc_timestamp("2026-02-30T00:00:00Z", "at")


@pytest.mark.parametrize(
    ("validator", "good", "bad"),
    [
        (voicebase.require_capture_id, "c" + "a" * 32, "c" + "A" * 32),
        (voicebase.require_identity_id, "v" + "a" * 12, "v" + "a" * 11),
        (voicebase.require_exemplar_id, "x" + "a" * 8, "x" + "g" * 8),
        (voicebase.require_sha256, "a" * 64, "A" * 64),
    ],
)
def test_id_and_digest_grammars_are_exact(validator, good, bad):
    assert validator(good) == good
    with pytest.raises(voicebase.Phase2DataError):
        validator(bad)


def test_capture_id_mint_retries_current_value(monkeypatch):
    values = iter(["1" * 32, "2" * 32])
    monkeypatch.setattr(voicebase.secrets, "token_hex", lambda _count: next(values))
    assert voicebase.mint_capture_id(current="c" + "1" * 32) == "c" + "2" * 32


def test_vector_accepts_finite_non_bool_unit_values_at_dimension_bounds():
    assert voicebase.validate_vector(_unit(16)) == tuple(_unit(16))
    assert len(voicebase.validate_vector(_unit(768))) == 768
    near = [0.25] * 16
    assert voicebase.validate_vector(near) == tuple(near)


@pytest.mark.parametrize(
    "vector",
    [
        [0.0] * 16,
        [True] + [0.0] * 15,
        [float("nan")] + [0.0] * 15,
        [float("inf")] + [0.0] * 15,
        [1e308] + [0.0] * 15,
        [1.00001] + [0.0] * 15,
        _unit(15),
        _unit(769),
    ],
)
def test_vector_law_rejects_adversarial_values(vector):
    with pytest.raises(voicebase.Phase2DataError):
        voicebase.validate_vector(vector)


def test_vector_requires_exact_declared_dimension():
    with pytest.raises(voicebase.Phase2DataError, match="exactly 17"):
        voicebase.validate_vector(_unit(16), dim=17)


def test_turn_projection_and_digest_are_numeric_deterministic():
    ints = [[0, 1, "A"], [1, 2, "B"]]
    floats = [[0.0, 1.0, "A"], [1.0, 2.0, "B"]]
    assert voicebase.strict_turn_projection(ints) == tuple(map(tuple, floats))
    assert voicebase.canonical_turns_digest(ints) == voicebase.canonical_turns_digest(
        floats
    )


@pytest.mark.parametrize(
    "turns",
    [
        [[0, 1]],
        [[True, 1, "A"]],
        [[0, float("nan"), "A"]],
        [[-1, 1, "A"]],
        [[1, 1, "A"]],
        [[0, 1, ""]],
        [[0, 1, "é" * 33]],
        [[1, 2, "B"], [0, 1, "A"]],
        [[0, 1, "B"], [0, 1, "A"]],
    ],
)
def test_turn_projection_rejects_malformed_or_unsorted_entries(turns):
    with pytest.raises(voicebase.Phase2DataError):
        voicebase.strict_turn_projection(turns)


@pytest.mark.parametrize("size", [0, 1, 1024 * 1024, 1024 * 1024 + 17, 2 * 1024 * 1024])
def test_media_fingerprint_uses_nonoverlapping_frame(tmp_path, size):
    payload = bytes((index % 251 for index in range(size)))
    path = tmp_path / f"media-{size}.bin"
    path.write_bytes(payload)
    head = payload[: 1024 * 1024]
    remaining = max(0, size - 1024 * 1024)
    tail_size = min(1024 * 1024, remaining)
    tail = payload[size - tail_size :] if tail_size else b""
    expected = hashlib.sha256(size.to_bytes(8, "big") + head + tail).hexdigest()
    assert voicebase.media_fingerprint(path) == expected


def test_voiceprints_schema_round_trip_accepts_unknown_top_key(tmp_path):
    sidecar = _sidecar()
    sidecar["future"] = {"safe": True}
    path = tmp_path / "ep.voiceprints.json"
    voicebase.write_voiceprints(path, sidecar)
    raw, validated = voicebase.load_voiceprints(path)
    assert raw == sidecar
    assert validated.embedding_dim == 16
    assert validated.speakers["SPEAKER_00"] == tuple(_unit())


def test_voiceprints_max_speaker_cardinality_round_trip(tmp_path):
    turns = [[index, index + 0.5, f"S{index:02}"] for index in range(64)]
    speakers = {f"S{index:02}": _unit(index=index % 16) for index in range(64)}
    sidecar = _sidecar(turns=turns, speakers=speakers)
    sidecar["provenance"]["detail"] = "é" * 256
    sidecar["binding"]["media_stem"] = "é" * 127 + "a"
    path = tmp_path / "maximum.voiceprints.json"
    voicebase.write_voiceprints(path, sidecar)
    assert len(voicebase.load_voiceprints(path)[1].speakers) == 64


def test_voiceprints_rejects_ragged_vectors_and_too_many_speakers():
    ragged = _sidecar(speakers={"SPEAKER_00": _unit(17)}, dim=16)
    with pytest.raises(voicebase.Phase2DataError, match="exactly 16"):
        voicebase.validate_voiceprints_mapping(ragged)
    too_many = _sidecar(speakers={f"S{index:02}": _unit() for index in range(65)})
    with pytest.raises(voicebase.Phase2DataError, match="at most 64"):
        voicebase.validate_voiceprints_mapping(too_many)


def test_voiceprints_writer_cap_preserves_existing_file(tmp_path):
    path = tmp_path / "ep.voiceprints.json"
    path.write_bytes(b"existing")
    sidecar = _sidecar()
    sidecar["future"] = "x" * voicebase.VOICEPRINTS_MAX_BYTES
    with pytest.raises(voicebase.Phase2DataError, match="encoded JSON exceeds"):
        voicebase.write_voiceprints(path, sidecar)
    assert path.read_bytes() == b"existing"


def test_four_part_voiceprint_conjunction_accepts_only_complete_match():
    sidecar = _sidecar()
    sibling = _sibling()
    validated = voicebase.validate_voiceprint_conjunction(sidecar, sibling, "a" * 64)
    assert validated.capture_id == sibling["voiceprint_capture"]

    mutations = [
        ({**sidecar, "capture_id": "c" + "2" * 32}, sibling, "a" * 64),
        (
            {
                **sidecar,
                "binding": {**sidecar["binding"], "media_fingerprint": "b" * 64},
            },
            sibling,
            "a" * 64,
        ),
        (sidecar, {**sibling, "speaker_turns": [[0, 2, "SPEAKER_00"]]}, "a" * 64),
        (sidecar, sibling, "b" * 64),
    ]
    for bad_sidecar, bad_sibling, consumer in mutations:
        assert not voicebase.voiceprint_conjunction_valid(
            bad_sidecar, bad_sibling, consumer
        )


def test_conjunction_requires_sidecar_labels_to_be_in_strict_turns():
    sidecar = _sidecar(speakers={"OTHER": _unit()})
    with pytest.raises(voicebase.Phase2DataError, match="subset"):
        voicebase.validate_voiceprint_conjunction(sidecar, _sibling(), "a" * 64)


def test_html_helpers_neutralize_malicious_values_in_all_contexts():
    malicious = '"><script>alert("x")</script>\u2028\u2029&\''
    text = voicebase.html_text(malicious)
    attribute = voicebase.html_attribute(malicious)
    script = voicebase.script_json({"name": malicious})

    assert "<script>" not in text
    assert "<script>" not in attribute
    assert "&quot;" in text and "&#x27;" in text
    assert "</script>" not in script.lower()
    assert "<\\/script>" in script
    assert "\u2028" not in script and "\u2029" not in script
    assert "\\u2028" in script and "\\u2029" in script
    assert json.loads(script.replace("<\\/", "</"))["name"] == malicious
