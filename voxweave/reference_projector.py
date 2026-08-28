"""Independent byte projection for an already selected align delivery."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Literal, cast

from voxweave.align_adapter import AlignDelivery, AlignProjectionInputs
from voxweave.align_snapshot import (
    FrozenAbsent,
    FrozenArray,
    FrozenBool,
    FrozenFloat,
    FrozenInt,
    FrozenJSON,
    FrozenNull,
    FrozenObject,
    FrozenString,
    RawJSONCarrier,
    freeze_json,
    thaw_json,
)
from voxweave.segmentation_adapter import (
    SegmentationDelivery,
    SegmentationProjectionInputs,
)
from voxweave.speakers import voice_text_for_block, voice_text_for_ids


@dataclass(frozen=True)
class ReferenceAlignProjection:
    vtt_bytes: bytes
    main_json_bytes: bytes


@dataclass(frozen=True)
class ReferenceSegmentationProjection:
    vtt_bytes: bytes
    main_json_bytes: bytes


class ReferenceAlignProjectionError(ValueError):
    def __init__(
        self,
        detail_code: Literal["cue-source-map", "unit-coverage"],
    ) -> None:
        super().__init__(detail_code)
        self.detail_code: Literal["cue-source-map", "unit-coverage"] = detail_code


def _validate_align_projection_domain(
    delivery: AlignDelivery,
    inputs: AlignProjectionInputs,
) -> None:
    source_indices = tuple(item.source_index for item in inputs.source_blocks)
    cue_indices = tuple(cue.source_index for cue in delivery.cues)
    if (
        len(set(source_indices)) != len(source_indices)
        or len(set(cue_indices)) != len(cue_indices)
        or len(source_indices) != len(cue_indices)
        or set(source_indices) != set(cue_indices)
    ):
        raise ReferenceAlignProjectionError("cue-source-map")
    covered_units = tuple(unit for cue in delivery.cues for unit in cue.word_data)
    if covered_units != delivery.word_segments:
        raise ReferenceAlignProjectionError("unit-coverage")


def _reference_json_bytes(value: FrozenJSON, *, allow_nan: bool) -> bytes:
    """Independently encode ordered frozen pairs with the legacy JSON layout."""

    def scalar(node: FrozenJSON) -> str:
        if isinstance(node, FrozenAbsent):
            raise TypeError("absent has no JSON document representation")
        if isinstance(node, FrozenNull):
            projected: Any = None
        elif isinstance(node, FrozenBool):
            projected = node.value
        elif isinstance(node, FrozenInt):
            projected = node.value
        elif isinstance(node, FrozenFloat):
            projected = struct.unpack(">d", bytes.fromhex(node.binary64_hex))[0]
        elif isinstance(node, FrozenString):
            projected = node.value
        else:
            raise TypeError("compound JSON value reached scalar projection")
        return json.dumps(
            projected,
            ensure_ascii=False,
            allow_nan=allow_nan,
        )

    def render(node: FrozenJSON, depth: int) -> str:
        if isinstance(node, FrozenArray):
            if not node.items:
                return "[]"
            member_indent = " " * (2 * (depth + 1))
            close_indent = " " * (2 * depth)
            members = [
                member_indent + render(member, depth + 1) for member in node.items
            ]
            return "[\n" + ",\n".join(members) + "\n" + close_indent + "]"
        if isinstance(node, FrozenObject):
            if not node.items:
                return "{}"
            member_indent = " " * (2 * (depth + 1))
            close_indent = " " * (2 * depth)
            members = [
                member_indent
                + json.dumps(key, ensure_ascii=False)
                + ": "
                + render(member, depth + 1)
                for key, member in node.items
            ]
            return "{\n" + ",\n".join(members) + "\n" + close_indent + "}"
        return scalar(node)

    return render(value, 0).encode("utf-8")


def _timestamp(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _decoration(inputs: AlignProjectionInputs, source_index: int) -> dict[str, Any]:
    matches = [
        item for item in inputs.source_blocks if item.source_index == source_index
    ]
    if len(matches) != 1:
        raise ValueError("selected cue decoration is not unique")
    item = matches[0]
    value: dict[str, Any] = {}
    if item.speaker is not None:
        value["speaker"] = item.speaker
    if item.speakers is not None:
        value["speakers"] = list(item.speakers)
    return value


def _vtt(delivery: AlignDelivery, inputs: AlignProjectionInputs) -> bytes:
    lines = ["WEBVTT", ""]
    for cue in delivery.cues:
        lines.append(f"{_timestamp(cue.start)} --> {_timestamp(cue.end)}")
        text = f"♪ {cue.text} ♪" if cue.lyric is True else cue.text
        lines.append(voice_text_for_block(text, _decoration(inputs, cue.source_index)))
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _reference_speaker_turns_value(
    inputs: AlignProjectionInputs,
) -> tuple[bool, FrozenJSON | None]:
    turns = inputs.speaker_turns
    if isinstance(turns, RawJSONCarrier):
        if not turns.present:
            return False, None
        if turns.value is None:
            raise ValueError("present speaker_turns carrier lacks a value")
        return True, turns.value
    if turns is None:
        return False, None
    return True, freeze_json(
        [
            [float(start), float(end), str(label)]
            for start, end, label in cast(Any, turns)
        ]
    )


def _main(delivery: AlignDelivery, inputs: AlignProjectionInputs) -> FrozenObject:
    value: dict[str, Any] = {
        "language": inputs.language,
        "segments": [
            {"text": cue.text, "start": cue.start, "end": cue.end}
            | ({"lyric": True} if cue.lyric is True else {})
            for cue in delivery.cues
        ],
        "word_segments": [
            {"text": unit.text, "start": unit.start, "end": unit.end}
            for unit in delivery.word_segments
        ],
    }
    if inputs.vad_speech is not None:
        value["vad_speech"] = [
            [float(start), float(end)] for start, end in inputs.vad_speech
        ]
    if inputs.shot_changes is not None:
        value["shot_changes"] = [float(item) for item in inputs.shot_changes]
    if inputs.sing_spans is not None:
        value["sing_spans"] = [
            [float(start), float(end)] for start, end in inputs.sing_spans
        ]
    turns_present, turns = _reference_speaker_turns_value(inputs)
    if turns_present:
        assert turns is not None
        value["speaker_turns"] = turns
    if inputs.voiceprint_capture is not None and inputs.voiceprint_media is not None:
        value["voiceprint_capture"] = inputs.voiceprint_capture
        value["voiceprint_media"] = inputs.voiceprint_media
    if inputs.segmentation is not None:
        value["segmentation"] = inputs.segmentation
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenObject):  # pragma: no cover - mapping invariant
        raise TypeError("reference align projection is not an object")
    return frozen


def reference_align_projection(
    delivery: AlignDelivery,
    inputs: AlignProjectionInputs,
    *,
    strict: bool,
) -> ReferenceAlignProjection:
    _validate_align_projection_domain(delivery, inputs)
    return ReferenceAlignProjection(
        _vtt(delivery, inputs),
        _reference_json_bytes(_main(delivery, inputs), allow_nan=not strict),
    )


def _segmentation_vtt(
    delivery: SegmentationDelivery,
    inputs: SegmentationProjectionInputs,
) -> bytes:
    names = dict(inputs.speaker_names)
    lines = ["WEBVTT", ""]
    for cue in delivery.cues:
        if inputs.timestamps:
            lines.append(f"{_timestamp(cue.start)} --> {_timestamp(cue.end)}")
        text = f"♪ {cue.text} ♪" if cue.lyric is True else cue.text
        if names:
            text = voice_text_for_ids(text, cue.speaker_ids, names)
        lines.extend((text, ""))
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _segmentation_main(delivery: SegmentationDelivery) -> FrozenObject:
    segments: list[dict[str, Any]] = []
    for cue in delivery.cues:
        row: dict[str, Any] = {
            "text": cue.text,
            "start": cue.start,
            "end": cue.end,
            "word_data": [thaw_json(unit) for unit in cue.word_data],
        }
        if cue.lyric is not None:
            row["lyric"] = cue.lyric
        segments.append(row)
    value: dict[str, Any] = {
        "language": delivery.language,
        "segments": segments,
        "word_segments": [thaw_json(unit) for unit in delivery.top_level_word_segments],
        "vad_speech": [thaw_json(row) for row in delivery.carriers.vad_speech],
    }
    if delivery.carriers.shot_changes is not None:
        value["shot_changes"] = [
            thaw_json(row) for row in delivery.carriers.shot_changes
        ]
    if delivery.carriers.sing_spans is not None:
        value["sing_spans"] = [thaw_json(row) for row in delivery.carriers.sing_spans]
    turns = delivery.carriers.speaker_turns
    if turns.present:
        if turns.value is None:
            raise ValueError("present speaker_turns carrier lacks a value")
        value["speaker_turns"] = turns.value
    if (
        delivery.carriers.voiceprint_capture is not None
        and delivery.carriers.voiceprint_media is not None
    ):
        value["voiceprint_capture"] = delivery.carriers.voiceprint_capture
        value["voiceprint_media"] = delivery.carriers.voiceprint_media
    value["segmentation"] = delivery.manifest
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenObject):  # pragma: no cover - mapping invariant
        raise TypeError("reference segmentation projection is not an object")
    return frozen


def reference_segmentation_projection(
    delivery: SegmentationDelivery,
    inputs: SegmentationProjectionInputs,
    *,
    strict: bool,
) -> ReferenceSegmentationProjection:
    return ReferenceSegmentationProjection(
        _segmentation_vtt(delivery, inputs),
        _reference_json_bytes(_segmentation_main(delivery), allow_nan=not strict),
    )


__all__ = [
    "ReferenceAlignProjection",
    "ReferenceAlignProjectionError",
    "ReferenceSegmentationProjection",
    "reference_align_projection",
    "reference_segmentation_projection",
]
