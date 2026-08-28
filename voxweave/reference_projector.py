"""Independent byte projection for an already selected align delivery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from voxweave.align_adapter import AlignDelivery, AlignProjectionInputs
from voxweave.align_snapshot import thaw_json
from voxweave.speakers import voice_text_for_block


@dataclass(frozen=True)
class ReferenceAlignProjection:
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


__all__ = ["ReferenceAlignProjection", "reference_align_projection"]
