import math
import struct

import pytest


def _vtt(*cues: tuple[str, str, str]) -> bytes:
    rows = ["WEBVTT", ""]
    for start, end, text in cues:
        rows.extend([f"{start} --> {end}", text, ""])
    return "\n".join(rows).encode()


def test_vtt_content_stays_lexical_while_qwen_route_is_separate():
    from voxweave.align_snapshot import decode_align_snapshot

    snapshot = decode_align_snapshot(
        "episode.vtt",
        _vtt(
            ("00:01:40.000", "00:01:42.000", "FIRST"),
            ("00:00:00.000", "00:00:01.000", "SECOND"),
        ),
        None,
        effective_iso="en",
    )
    assert [block.text for block in snapshot.blocks] == ["FIRST", "SECOND"]
    assert [block.source_index for block in snapshot.blocks] == [0, 1]
    assert snapshot.qwen_delivery_order == (1, 0)
    assert [(bound.start, bound.end) for bound in snapshot.route_bounds] == [
        (100.0, 102.0),
        (0.0, 1.0),
    ]


def test_route_view_swaps_reversed_bounds_without_mutating_content():
    from voxweave.align_snapshot import decode_align_snapshot

    snapshot = decode_align_snapshot(
        "episode.vtt",
        _vtt(("00:00:05.000", "00:00:03.000", "kept")),
        None,
        effective_iso="en",
    )
    assert snapshot.blocks[0].text == "kept"
    assert snapshot.route_bounds[0].start == 3.0
    assert snapshot.route_bounds[0].end == 5.0


def test_snapshot_rejects_renamed_ass_and_empty_vtt_with_historical_messages():
    from voxweave.align_snapshot import decode_subtitle_snapshot

    with pytest.raises(RuntimeError, match="content is ASS/SSA.*extension says vtt"):
        decode_subtitle_snapshot(
            "renamed.vtt", b"[Script Info]\n[Events]\nFormat: Start, End, Text\n"
        )
    with pytest.raises(RuntimeError, match=r"^no cues in empty\.vtt$"):
        decode_subtitle_snapshot("empty.vtt", b"WEBVTT\n\nNOTE only\nignored\n")


def test_snapshot_uses_existing_bom_and_fallback_decoder(caplog):
    from voxweave.align_snapshot import decode_subtitle_snapshot

    text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nCaf\u00e9\n"
    utf16 = decode_subtitle_snapshot("utf16.vtt", text.encode("utf-16"))
    assert utf16.blocks[0].text == "Caf\u00e9"
    cp1252 = decode_subtitle_snapshot("western.vtt", text.encode("cp1252"))
    assert cp1252.blocks[0].text == "Caf\u00e9"
    assert any("decoded as cp1252" in record.message for record in caplog.records)


def test_dual_json_projection_retains_duplicates_and_legacy_last_key_wins():
    from voxweave.align_snapshot import (
        FrozenFloat,
        FrozenObject,
        decode_sibling_json_snapshot,
    )

    raw = b'{"speaker_turns":[1],"speaker_turns":{"x":1,"x":2},"nested":{"value":NaN}}'
    snapshot = decode_sibling_json_snapshot("episode.json", raw)
    semantic = snapshot.thaw_legacy()
    assert semantic["speaker_turns"] == {"x": 2}
    assert math.isnan(semantic["nested"]["value"])
    assert isinstance(snapshot.lexical, FrozenObject)
    assert [key for key, _value in snapshot.lexical.items].count("speaker_turns") == 2
    nested = dict(snapshot.lexical.items)["nested"]
    assert isinstance(nested, FrozenObject)
    assert isinstance(dict(nested.items)["value"], FrozenFloat)
    assert snapshot.strict_input_status.detail_code == "sibling-json-duplicate-key"
    carrier = snapshot.carrier("speaker_turns")
    assert carrier.present is True
    assert isinstance(carrier.value, FrozenObject)
    assert [key for key, _value in carrier.value.items] == ["x", "x"]


def test_nonfinite_is_the_strict_status_when_no_duplicate_precedes_it():
    from voxweave.align_snapshot import decode_sibling_json_snapshot

    snapshot = decode_sibling_json_snapshot("episode.json", b'{"x":Infinity}')
    assert snapshot.strict_input_status.kind == "invalid"
    assert snapshot.strict_input_status.detail_code == "sibling-json-nonfinite"


def test_absent_sibling_is_distinct_from_present_empty_object():
    from voxweave.align_snapshot import FrozenAbsent, decode_sibling_json_snapshot

    absent = decode_sibling_json_snapshot("episode.json", None)
    empty = decode_sibling_json_snapshot("episode.json", b"{}")
    assert absent.present is False
    assert isinstance(absent.lexical, FrozenAbsent)
    assert empty.present is True
    assert absent.digest != empty.digest
    assert absent.carrier("speaker_turns").present is False


@pytest.mark.parametrize(
    "left,right",
    [
        (None, "__ABSENT__"),
        ({}, []),
        ({"x": 1}, [["x", 1]]),
        (True, 1),
        (1, 1.0),
        (-0.0, 0.0),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        ([1, [2]], [[1, 2]]),
    ],
)
def test_frozen_json_encoding_is_injective(left, right):
    from voxweave.align_snapshot import FROZEN_ABSENT, encode_frozen_json, freeze_json

    left_value = FROZEN_ABSENT if left == "__ABSENT__" else left
    right_value = FROZEN_ABSENT if right == "__ABSENT__" else right
    assert encode_frozen_json(freeze_json(left_value)) != encode_frozen_json(
        freeze_json(right_value)
    )


def test_float_freeze_preserves_type_bits_signed_zero_and_nonfinite_class():
    from voxweave.align_snapshot import FrozenFloat, freeze_json

    neg_zero = freeze_json(-0.0)
    pos_zero = freeze_json(0.0)
    assert isinstance(neg_zero, FrozenFloat)
    assert isinstance(pos_zero, FrozenFloat)
    assert neg_zero.binary64_hex == struct.pack(">d", -0.0).hex()
    assert pos_zero.binary64_hex == struct.pack(">d", 0.0).hex()
    assert freeze_json(float("inf")).source_class == "positive-infinity"
    assert freeze_json(float("-inf")).source_class == "negative-infinity"
    assert freeze_json(float("nan")).source_class == "nan"


def test_frozen_json_rejects_cycles_and_non_string_object_keys():
    from voxweave.align_snapshot import FrozenJSONDomainError, freeze_json

    cycle = []
    cycle.append(cycle)
    with pytest.raises(FrozenJSONDomainError) as cycle_error:
        freeze_json(cycle)
    assert cycle_error.value.detail_code == "selected-json-cycle"
    with pytest.raises(FrozenJSONDomainError) as key_error:
        freeze_json({1: "not a JSON object key"})
    assert key_error.value.detail_code == "non-string-selected-object-key"


def test_each_legacy_thaw_is_independent_and_does_not_change_the_seal():
    from voxweave.align_snapshot import decode_sibling_json_snapshot

    snapshot = decode_sibling_json_snapshot("episode.json", b'{"x":[1,{"y":2}]}')
    before = snapshot.digest
    first = snapshot.thaw_legacy()
    first["x"][1]["y"] = 99
    assert snapshot.thaw_legacy() == {"x": [1, {"y": 2}]}
    assert snapshot.digest == before


def test_content_digest_ignores_timestamps_but_includes_voice_metadata():
    from voxweave.align_snapshot import decode_align_snapshot

    first = decode_align_snapshot(
        "episode.vtt",
        _vtt(("00:00:00.000", "00:00:01.000", "<v Alice>Hello</v>")),
        None,
        effective_iso="en",
    )
    shifted = decode_align_snapshot(
        "episode.vtt",
        _vtt(("00:01:00.000", "00:01:01.000", "<v Alice>Hello</v>")),
        None,
        effective_iso="en",
    )
    renamed = decode_align_snapshot(
        "episode.vtt",
        _vtt(("00:00:00.000", "00:00:01.000", "<v Bob>Hello</v>")),
        None,
        effective_iso="en",
    )
    assert first.block_content_sha256 == shifted.block_content_sha256
    assert first.block_content_sha256 != renamed.block_content_sha256
