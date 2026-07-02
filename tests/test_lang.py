import logging

import pytest

from voxweave.lang import is_supported, to_aligner_name, to_iso, to_iso3


def test_full_name_to_iso():
    assert to_iso("English") == "en"
    assert to_iso("Chinese") == "zh"
    assert to_iso("Cantonese") == "yue"


def test_iso_to_iso_passthrough():
    assert to_iso("zh") == "zh"
    assert to_iso("EN") == "en"


def test_to_aligner_name_from_iso_and_name():
    assert to_aligner_name("zh") == "chinese"
    assert to_aligner_name("Chinese") == "chinese"
    assert to_aligner_name("en") == "english"


def test_is_supported():
    assert is_supported("zh")
    assert is_supported("English")
    assert not is_supported("klingon")
    assert not is_supported(
        "th"
    )  # smart_split knows th but the aligner does not support it


def test_unknown_raises():
    with pytest.raises(ValueError):
        to_iso("klingon")
    with pytest.raises(ValueError):
        to_aligner_name("klingon")


# --- #41: BCP-47 region/script subtags must be stripped before lookup ---


def test_to_iso_strips_bcp47_region_tag():
    assert to_iso("en-US") == "en"
    assert to_iso("zh-CN") == "zh"
    assert to_iso("ja-JP") == "ja"


def test_to_iso_strips_bcp47_underscore_variant():
    assert to_iso("en_US") == "en"


def test_is_supported_with_bcp47_region_tag():
    assert is_supported("en-US")
    assert is_supported("zh-CN")


# --- #42: to_iso3 unknown-code passthrough must warn ---


def test_to_iso3_unknown_code_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = to_iso3("xx")
    assert result == "xx"  # passthrough shape preserved (mux tagging must not crash)
    assert any("unknown language" in rec.message.lower() for rec in caplog.records)
