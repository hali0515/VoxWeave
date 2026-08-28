"""Producer projection for immutable align deliveries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from voxweave import realign
from voxweave.align_adapter import AlignDelivery, AlignProjectionInputs
from voxweave.align_snapshot import (
    FrozenJSON,
    FrozenObject,
    RawJSONCarrier,
    encode_frozen_json_document,
    freeze_json,
)
from voxweave.speakers import voice_text_for_block


@dataclass(frozen=True)
class AlignPrimaryProjection:
    vtt_bytes: bytes
    main_json_bytes: bytes


class AlignProjectionEncodeError(RuntimeError):
    def __init__(self, detail_code: Literal["main-json-encode", "vtt-encode"]):
        super().__init__(detail_code)
        self.detail_code: Literal["main-json-encode", "vtt-encode"] = detail_code


def _source_block(inputs: AlignProjectionInputs, source_index: int) -> dict[str, Any]:
    matches = tuple(
        block for block in inputs.source_blocks if block.source_index == source_index
    )
    if len(matches) != 1:
        raise ValueError("delivery cue source decoration is missing or duplicated")
    item = matches[0]
    block: dict[str, Any] = {}
    if item.speaker is not None:
        block["speaker"] = item.speaker
    if item.speakers is not None:
        block["speakers"] = list(item.speakers)
    return block


def _speaker_turns_value(
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


def _main_value(delivery: AlignDelivery, inputs: AlignProjectionInputs) -> FrozenObject:
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
    turns_present, turns = _speaker_turns_value(inputs)
    if turns_present:
        assert turns is not None
        data["speaker_turns"] = turns
    if inputs.voiceprint_capture is not None and inputs.voiceprint_media is not None:
        data["voiceprint_capture"] = inputs.voiceprint_capture
        data["voiceprint_media"] = inputs.voiceprint_media
    if inputs.segmentation is not None:
        data["segmentation"] = inputs.segmentation
    frozen = freeze_json(data)
    if not isinstance(frozen, FrozenObject):  # pragma: no cover - mapping invariant
        raise TypeError("align main projection is not an object")
    return frozen


def project_align_vtt_bytes(
    delivery: AlignDelivery,
    inputs: AlignProjectionInputs,
) -> bytes:
    """Render only the VTT member of one candidate family."""
    rows = []
    for cue in delivery.cues:
        text = f"♪ {cue.text} ♪" if cue.lyric is True else cue.text
        text = voice_text_for_block(text, _source_block(inputs, cue.source_index))
        rows.append((cue.start, cue.end, text))
    return realign.render_cues(rows).encode("utf-8")


def project_align_main_json_bytes(
    delivery: AlignDelivery,
    inputs: AlignProjectionInputs,
    *,
    strict: bool,
) -> bytes:
    """Encode only the main-JSON member of one candidate family."""
    return encode_frozen_json_document(
        _main_value(delivery, inputs),
        allow_nan=not strict,
    )


def project_align_delivery(
    delivery: AlignDelivery,
    inputs: AlignProjectionInputs,
    *,
    strict: bool,
) -> AlignPrimaryProjection:
    try:
        vtt_bytes = project_align_vtt_bytes(delivery, inputs)
    except Exception as exc:
        raise AlignProjectionEncodeError("vtt-encode") from exc
    try:
        main_json_bytes = project_align_main_json_bytes(
            delivery,
            inputs,
            strict=strict,
        )
    except Exception as exc:
        raise AlignProjectionEncodeError("main-json-encode") from exc
    return AlignPrimaryProjection(vtt_bytes, main_json_bytes)


__all__ = [
    "AlignPrimaryProjection",
    "AlignProjectionEncodeError",
    "project_align_delivery",
    "project_align_main_json_bytes",
    "project_align_vtt_bytes",
]
