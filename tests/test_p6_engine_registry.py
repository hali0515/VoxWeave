import hashlib

import pytest


EXPECTED_BYTES = (
    b'{"de":"legacy-v1","en":"legacy-v1","es":"legacy-v1",'
    b'"fr":"legacy-v1","it":"legacy-v1","ja":"legacy-v1",'
    b'"ko":"legacy-v1","pt":"legacy-v1","ru":"legacy-v1",'
    b'"yue":"legacy-v1","zh":"legacy-v1"}\n'
)


def test_registry_is_complete_all_legacy_and_byte_pinned():
    from voxweave import engine_registry as registry

    assert dict(registry.LANGUAGE_ENGINE_FAMILY) == {
        "de": "legacy-v1",
        "en": "legacy-v1",
        "es": "legacy-v1",
        "fr": "legacy-v1",
        "it": "legacy-v1",
        "ja": "legacy-v1",
        "ko": "legacy-v1",
        "pt": "legacy-v1",
        "ru": "legacy-v1",
        "yue": "legacy-v1",
        "zh": "legacy-v1",
    }
    assert registry.REGISTRY_CANONICAL_BYTES == EXPECTED_BYTES
    assert registry.REGISTRY_SHA256 == hashlib.sha256(EXPECTED_BYTES).hexdigest()


@pytest.mark.parametrize(
    ("raw", "canonical", "family"),
    [
        ("English", "en", "legacy-v1"),
        ("JA-jp", "ja", "legacy-v1"),
        ("yue_HK", "yue", "legacy-v1"),
        (None, None, "legacy-v1"),
        ("", None, "legacy-v1"),
        ("klingon", None, "legacy-v1"),
    ],
)
def test_registry_canonicalization_is_total_and_never_guesses_english(
    raw, canonical, family
):
    from voxweave import engine_registry as registry

    assert registry.canonical_registry_iso(raw) == canonical
    assert registry.engine_family_for(raw) == family


def test_registry_and_manifest_maps_are_immutable():
    from voxweave import engine_registry as registry

    with pytest.raises(TypeError):
        registry.LANGUAGE_ENGINE_FAMILY["en"] = "boundary-v2"
    with pytest.raises(TypeError):
        registry.MANIFEST_ENGINE_BY_FAMILY["legacy-v1"] = "other"
