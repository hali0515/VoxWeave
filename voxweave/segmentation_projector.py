"""Pure producer projection for typed process and split deliveries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voxweave import realign
from voxweave.align_snapshot import (
    FrozenObject,
    encode_frozen_json_document,
    freeze_json,
    thaw_json,
)
from voxweave.segmentation_adapter import (
    SegmentationDelivery,
    SegmentationProjectionInputs,
)
from voxweave.speakers import voice_text_for_ids


@dataclass(frozen=True)
class SegmentationPrimaryProjection:
    vtt_bytes: bytes
    main_json_bytes: bytes


def _main_value(delivery: SegmentationDelivery) -> FrozenObject:
    segments: list[dict[str, Any]] = []
    for cue in delivery.cues:
        value: dict[str, Any] = {
            "text": cue.text,
            "start": cue.start,
            "end": cue.end,
            "word_data": [thaw_json(unit) for unit in cue.word_data],
        }
        if cue.lyric is not None:
            value["lyric"] = cue.lyric
        segments.append(value)
    data: dict[str, Any] = {
        "language": delivery.language,
        "segments": segments,
        "word_segments": [thaw_json(unit) for unit in delivery.top_level_word_segments],
        "vad_speech": [thaw_json(row) for row in delivery.carriers.vad_speech],
    }
    if delivery.carriers.shot_changes is not None:
        data["shot_changes"] = [
            thaw_json(value) for value in delivery.carriers.shot_changes
        ]
    if delivery.carriers.sing_spans is not None:
        data["sing_spans"] = [
            thaw_json(value) for value in delivery.carriers.sing_spans
        ]
    turns = delivery.carriers.speaker_turns
    if turns.present:
        if turns.value is None:
            raise ValueError("present speaker_turns carrier lacks a value")
        data["speaker_turns"] = turns.value
    if (
        delivery.carriers.voiceprint_capture is not None
        and delivery.carriers.voiceprint_media is not None
    ):
        data["voiceprint_capture"] = delivery.carriers.voiceprint_capture
        data["voiceprint_media"] = delivery.carriers.voiceprint_media
    data["segmentation"] = delivery.manifest
    frozen = freeze_json(data)
    if not isinstance(frozen, FrozenObject):  # pragma: no cover - mapping invariant
        raise TypeError("segmentation main projection is not an object")
    return frozen


def project_segmentation_delivery(
    delivery: SegmentationDelivery,
    inputs: SegmentationProjectionInputs,
    *,
    strict: bool,
) -> SegmentationPrimaryProjection:
    names = dict(inputs.speaker_names)
    rows: list[tuple[float | None, float | None, str]] = []
    for cue in delivery.cues:
        text = f"♪ {cue.text} ♪" if cue.lyric is True else cue.text
        if names:
            text = voice_text_for_ids(text, cue.speaker_ids, names)
        rows.append(
            (
                cue.start if inputs.timestamps else None,
                cue.end if inputs.timestamps else None,
                text,
            )
        )
    return SegmentationPrimaryProjection(
        realign.render_cues(rows).encode("utf-8"),
        encode_frozen_json_document(_main_value(delivery), allow_nan=not strict),
    )


__all__ = ["SegmentationPrimaryProjection", "project_segmentation_delivery"]
