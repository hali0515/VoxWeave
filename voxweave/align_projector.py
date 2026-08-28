"""Producer projection for immutable align deliveries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from voxweave import realign
from voxweave.align_adapter import AlignDelivery, AlignProjectionInputs
from voxweave.align_snapshot import thaw_json
from voxweave.speakers import voice_text_for_block


@dataclass(frozen=True)
class AlignPrimaryProjection:
    vtt_bytes: bytes
    main_json_bytes: bytes


def _source_block(inputs: AlignProjectionInputs, source_index: int) -> dict[str, Any]:
    item = next(
        (block for block in inputs.source_blocks if block.source_index == source_index),
        None,
    )
    if item is None:
        raise ValueError("delivery cue lacks its sealed source decoration")
    block: dict[str, Any] = {}
    if item.speaker is not None:
        block["speaker"] = item.speaker
    if item.speakers is not None:
        block["speakers"] = list(item.speakers)
    return block


def _main_value(
    delivery: AlignDelivery, inputs: AlignProjectionInputs
) -> dict[str, Any]:
    segments = [
        {"text": cue.text, "start": cue.start, "end": cue.end}
        | ({"lyric": True} if cue.lyric is True else {})
        for cue in delivery.cues
    ]
    units = [
        {"text": unit.text, "start": unit.start, "end": unit.end}
        for unit in delivery.word_segments
    ]
    data: dict[str, Any] = {
        "language": inputs.language,
        "segments": segments,
        "word_segments": units,
    }
    if inputs.vad_speech is not None:
        data["vad_speech"] = [
            [float(start), float(end)] for start, end in inputs.vad_speech
        ]
    if inputs.shot_changes is not None:
        data["shot_changes"] = [float(value) for value in inputs.shot_changes]
    if inputs.sing_spans is not None:
        data["sing_spans"] = [
            [float(start), float(end)] for start, end in inputs.sing_spans
        ]
    if inputs.speaker_turns is not None:
        data["speaker_turns"] = [
            [float(start), float(end), str(label)]
            for start, end, label in inputs.speaker_turns
        ]
    if inputs.voiceprint_capture is not None and inputs.voiceprint_media is not None:
        data["voiceprint_capture"] = inputs.voiceprint_capture
        data["voiceprint_media"] = inputs.voiceprint_media
    if inputs.segmentation is not None:
        data["segmentation"] = thaw_json(inputs.segmentation)
    return data


def project_align_delivery(
    delivery: AlignDelivery,
    inputs: AlignProjectionInputs,
    *,
    strict: bool,
) -> AlignPrimaryProjection:
    rows = []
    for cue in delivery.cues:
        text = f"♪ {cue.text} ♪" if cue.lyric is True else cue.text
        text = voice_text_for_block(text, _source_block(inputs, cue.source_index))
        rows.append((cue.start, cue.end, text))
    vtt_bytes = realign.render_cues(rows).encode("utf-8")
    main_json_bytes = json.dumps(
        _main_value(delivery, inputs),
        ensure_ascii=False,
        indent=2,
        allow_nan=not strict,
    ).encode("utf-8")
    return AlignPrimaryProjection(vtt_bytes, main_json_bytes)


__all__ = ["AlignPrimaryProjection", "project_align_delivery"]
