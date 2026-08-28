"""Pure producer projection for typed process and split deliveries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from voxweave import realign
from voxweave.align_snapshot import RawJSONCarrier, thaw_json
from voxweave.core.segdoc import normalize_speaker_turn_bounds
from voxweave.p6_ratifications import RAW_SPEAKER_TURNS_WRITER_ENABLED
from voxweave.segmentation_adapter import (
    SegmentationDelivery,
    SegmentationProjectionInputs,
)
from voxweave.speakers import voice_text_for_ids


@dataclass(frozen=True)
class SegmentationPrimaryProjection:
    vtt_bytes: bytes
    main_json_bytes: bytes


def _legacy_turns(carrier: RawJSONCarrier) -> list[list[object]] | None:
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


def _turns(carrier: RawJSONCarrier) -> Any | None:
    if RAW_SPEAKER_TURNS_WRITER_ENABLED:
        return None if carrier.value is None else thaw_json(carrier.value)
    return _legacy_turns(carrier)


def _main_value(delivery: SegmentationDelivery) -> dict[str, Any]:
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
    turns = _turns(delivery.carriers.speaker_turns)
    if turns is not None:
        data["speaker_turns"] = turns
    if (
        delivery.carriers.voiceprint_capture is not None
        and delivery.carriers.voiceprint_media is not None
    ):
        data["voiceprint_capture"] = delivery.carriers.voiceprint_capture
        data["voiceprint_media"] = delivery.carriers.voiceprint_media
    data["segmentation"] = thaw_json(delivery.manifest)
    return data


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
        json.dumps(
            _main_value(delivery),
            ensure_ascii=False,
            indent=2,
            allow_nan=not strict,
        ).encode("utf-8"),
    )


__all__ = ["SegmentationPrimaryProjection", "project_segmentation_delivery"]
