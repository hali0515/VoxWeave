"""Independent byte projection for an already selected align delivery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from voxweave.align_adapter import AlignDelivery, AlignProjectionInputs
from voxweave.align_snapshot import RawJSONCarrier, thaw_json
from voxweave.core.segdoc import normalize_speaker_turn_bounds
from voxweave.p6_ratifications import RAW_SPEAKER_TURNS_WRITER_ENABLED
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


def _main(delivery: AlignDelivery, inputs: AlignProjectionInputs) -> dict[str, Any]:
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
    if inputs.speaker_turns is not None:
        value["speaker_turns"] = [
            [float(start), float(end), str(label)]
            for start, end, label in inputs.speaker_turns
        ]
    if inputs.voiceprint_capture is not None and inputs.voiceprint_media is not None:
        value["voiceprint_capture"] = inputs.voiceprint_capture
        value["voiceprint_media"] = inputs.voiceprint_media
    if inputs.segmentation is not None:
        value["segmentation"] = thaw_json(inputs.segmentation)
    return value


def reference_align_projection(
    delivery: AlignDelivery,
    inputs: AlignProjectionInputs,
    *,
    strict: bool,
) -> ReferenceAlignProjection:
    return ReferenceAlignProjection(
        _vtt(delivery, inputs),
        json.dumps(
            _main(delivery, inputs),
            ensure_ascii=False,
            indent=2,
            allow_nan=not strict,
        ).encode("utf-8"),
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


def _reference_legacy_turns(carrier: RawJSONCarrier) -> list[list[object]] | None:
    if not carrier.present or carrier.value is None:
        return None
    raw = thaw_json(carrier.value)
    if not raw:
        return None
    try:
        values = iter(raw)
    except TypeError:
        return None
    rows: list[list[object]] = []
    for value in values:
        try:
            start_value, end_value, label = value
            start, end = normalize_speaker_turn_bounds(
                float(start_value), float(end_value)
            )
        except (TypeError, ValueError):
            continue
        rows.append([start, end, str(label)])
    return rows or None


def _reference_turns(carrier: RawJSONCarrier) -> Any | None:
    if RAW_SPEAKER_TURNS_WRITER_ENABLED:
        return None if carrier.value is None else thaw_json(carrier.value)
    return _reference_legacy_turns(carrier)


def _segmentation_main(delivery: SegmentationDelivery) -> dict[str, Any]:
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
    turns = _reference_turns(delivery.carriers.speaker_turns)
    if turns is not None:
        value["speaker_turns"] = turns
    if (
        delivery.carriers.voiceprint_capture is not None
        and delivery.carriers.voiceprint_media is not None
    ):
        value["voiceprint_capture"] = delivery.carriers.voiceprint_capture
        value["voiceprint_media"] = delivery.carriers.voiceprint_media
    value["segmentation"] = thaw_json(delivery.manifest)
    return value


def reference_segmentation_projection(
    delivery: SegmentationDelivery,
    inputs: SegmentationProjectionInputs,
    *,
    strict: bool,
) -> ReferenceSegmentationProjection:
    return ReferenceSegmentationProjection(
        _segmentation_vtt(delivery, inputs),
        json.dumps(
            _segmentation_main(delivery),
            ensure_ascii=False,
            indent=2,
            allow_nan=not strict,
        ).encode("utf-8"),
    )


__all__ = [
    "ReferenceAlignProjection",
    "ReferenceSegmentationProjection",
    "reference_align_projection",
    "reference_segmentation_projection",
]
