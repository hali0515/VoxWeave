import logging

import pytest

from voxweave.lang import (
    detected_language_candidates,
    is_supported,
    reconcile_detected_language,
    to_aligner_name,
    to_iso,
    to_iso3,
    transcript_content_weight,
)


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


# --- transcript-aware auto-detection -------------------------------------- #


HAN_HEAVY_TECH_TEXT = (
    "GPT Red。该模型可自动模拟各类网络攻击，用于检测 AI 大模型的安全漏洞。"
    "采用自博弈强化学习训练，简单说就是用一个网络攻击模型和一个网络防御模型"
    "同步进行对抗。采用这种方式训练和迭代模型。OpenAI"
)


def test_detected_language_candidates_handles_qwen_multilingual_labels():
    assert detected_language_candidates(
        " English, Chinese，ENGLISH； Japanese ,, "
    ) == ["English", "Chinese", "Japanese"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [("English", "zh"), ("English,Chinese", "Chinese")],
)
def test_reconcile_han_heavy_transcript_misreported_as_english(label, expected):
    assert reconcile_detected_language(label, HAN_HEAVY_TECH_TEXT) == expected


def test_reconcile_explicit_language_always_wins():
    assert (
        reconcile_detected_language(
            "English,Chinese", HAN_HEAVY_TECH_TEXT, override="en"
        )
        == "en"
    )


def test_reconcile_does_not_flip_short_proper_noun_snippet():
    assert (
        reconcile_detected_language("Chinese,English", "OpenAI Codex Micro")
        == "Chinese"
    )


def test_reconcile_sparse_kana_is_still_japanese_script_evidence():
    assert reconcile_detected_language("en", "東京都が新計画を発表") == "ja"


def test_reconcile_han_only_prefers_supplied_japanese_candidate():
    assert (
        reconcile_detected_language("English,Japanese", "東京証券取引所株価指数")
        == "Japanese"
    )


def test_reconcile_han_only_preserves_cantonese_candidate():
    assert reconcile_detected_language("English,Cantonese", "今日香港天氣很好") == (
        "Cantonese"
    )


def test_transcript_content_weight_ignores_alignment_and_punctuation():
    assert transcript_content_weight("你好，AI!") == 4
