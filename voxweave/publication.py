"""Pure VTT and sibling-JSON byte encoding for process, split, and align."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from voxweave import realign
from voxweave.align_failures import CanonicalFailure
from voxweave.align_snapshot import (
    FrozenArray,
    FrozenJSON,
    FrozenObject,
    RawJSONCarrier,
    encode_frozen_json_document,
    freeze_json,
)
from voxweave.core.schema import Cue
from voxweave.speakers import voice_text_for_ids


@dataclass(frozen=True)
class EncodedPrimaryOutputs:
    """Encoded VTT and sibling JSON byte payloads ready for transaction staging."""

    vtt_bytes: bytes
    main_json_bytes: bytes


class PublicationEncodeError(RuntimeError):
    """Raised when VTT or sibling JSON serialization fails before staging."""

    def __init__(
        self,
        detail_code: Literal["main-json-encode", "vtt-encode"],
        cause: BaseException | None = None,
    ):
        msg = f"{detail_code}: {cause}" if cause is not None else detail_code
        super().__init__(msg)
        self.detail_code: Literal["main-json-encode", "vtt-encode"] = detail_code
        self.failure = CanonicalFailure("preencode-failed", "encoder", detail_code)


def semantic_speaker_turns_carrier(
    turns: Sequence[tuple[float, float, str]] | None,
) -> RawJSONCarrier:
    """Wrap fresh ``speaker_turns`` tuples into a :class:`RawJSONCarrier`."""
    if turns is None:
        return RawJSONCarrier(False, None)
    frozen = freeze_json(
        [[float(start), float(end), str(label)] for start, end, label in turns]
    )
    if not isinstance(frozen, FrozenArray):  # pragma: no cover - list invariant
        raise TypeError("speaker_turns carrier is not an array")
    return RawJSONCarrier(True, frozen)


def _speaker_turns_json(
    turns: RawJSONCarrier | Sequence[tuple[float, float, str]] | None,
) -> tuple[bool, FrozenJSON | None]:
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


def encode_segmentation_outputs(
    *,
    language: str,
    cues: Sequence[Cue],
    top_level_units: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    vad_speech: Sequence[tuple[float, float]] | None,
    shot_changes: Sequence[float] | None,
    sing_spans: Sequence[tuple[float, float]] | None,
    speaker_turns: RawJSONCarrier | Sequence[tuple[float, float, str]] | None,
    voiceprint_pair: tuple[str, str] | None,
    timestamps: bool,
    speaker_names: Sequence[tuple[str, str]],
) -> tuple[EncodedPrimaryOutputs, tuple[dict[str, object], ...]]:
    """Encode VTT + sibling JSON bytes and SDH dialogue view for ``process`` / ``split``."""
    names = dict(speaker_names)
    try:
        rows: list[tuple[float | None, float | None, str]] = []
        for cue in cues:
            raw_text = str(cue["text"])
            lyric = cue.get("lyric") is True
            text = f"♪ {raw_text} ♪" if lyric else raw_text
            if names:
                speaker_ids_val = cue.get("speaker_ids")
                speaker_ids = (
                    None
                    if speaker_ids_val is None
                    else tuple(str(v) for v in speaker_ids_val)
                )
                text = voice_text_for_ids(text, speaker_ids, names)
            start = float(cue["start"])
            end = float(cue["end"])
            rows.append(
                (start if timestamps else None, end if timestamps else None, text)
            )
        vtt_bytes = realign.render_cues(rows).encode("utf-8")
    except Exception as exc:
        raise PublicationEncodeError("vtt-encode", exc) from exc

    try:
        segments: list[dict[str, Any]] = []
        for cue in cues:
            seg: dict[str, Any] = {
                "text": str(cue["text"]),
                "start": float(cue["start"]),
                "end": float(cue["end"]),
                "word_data": [dict(unit) for unit in cue.get("word_data") or ()],
            }
            if cue.get("lyric") is True:
                seg["lyric"] = True
            segments.append(seg)

        data: dict[str, Any] = {
            "language": language,
            "segments": segments,
            "word_segments": [dict(unit) for unit in top_level_units],
            "vad_speech": [[float(s), float(e)] for s, e in (vad_speech or ())],
        }
        if shot_changes is not None:
            data["shot_changes"] = [float(v) for v in shot_changes]
        if sing_spans is not None:
            data["sing_spans"] = [[float(s), float(e)] for s, e in sing_spans]
        turns_present, turns_val = _speaker_turns_json(speaker_turns)
        if turns_present:
            assert turns_val is not None
            data["speaker_turns"] = turns_val
        if voiceprint_pair is not None:
            data["voiceprint_capture"] = voiceprint_pair[0]
            data["voiceprint_media"] = voiceprint_pair[1]
        data["segmentation"] = dict(manifest)

        frozen = freeze_json(data)
        if not isinstance(frozen, FrozenObject):  # pragma: no cover - dict invariant
            raise TypeError("segmentation main projection is not an object")
        main_json_bytes = encode_frozen_json_document(frozen, allow_nan=True)
    except Exception as exc:
        raise PublicationEncodeError("main-json-encode", exc) from exc

    sdh_dialogue = tuple(
        {
            "text": str(cue["text"]),
            "start": float(cue["start"]),
            "end": float(cue["end"]),
            **({"lyric": True} if cue.get("lyric") is True else {}),
        }
        for cue in cues
    )
    return EncodedPrimaryOutputs(vtt_bytes, main_json_bytes), sdh_dialogue


def encode_align_outputs(
    *,
    blocks: Sequence[Mapping[str, Any]],
    spans: Sequence[tuple[float, float]],
    all_units: Sequence[Mapping[str, Any]],
    language: str,
    vad_speech: Sequence[tuple[float, float]] | None,
    shot_changes: Sequence[float] | None,
    sing_spans: Sequence[tuple[float, float]] | None,
    speaker_turns: RawJSONCarrier | Sequence[tuple[float, float, str]] | None,
    voiceprint_pair: tuple[str, str] | None,
    manifest: Mapping[str, Any] | None,
) -> EncodedPrimaryOutputs:
    """Encode VTT + sibling JSON bytes for ``align``."""
    try:
        vtt_blocks: list[dict[str, Any]] = []
        vtt_spans: list[tuple[float, float]] = []
        for block, span in zip(blocks, spans, strict=True):
            item: dict[str, Any] = {}
            if isinstance(block.get("speaker"), str):
                item["speaker"] = block["speaker"]
            speakers_val = block.get("speakers")
            if isinstance(speakers_val, (list, tuple)):
                item["speakers"] = [
                    (None if row[0] is None else str(row[0]), str(row[1]))
                    for row in speakers_val
                    if isinstance(row, (list, tuple)) and len(row) == 2
                ]
            item["text"] = str(block["text"])
            item["lyric"] = block.get("lyric") is True
            vtt_blocks.append(item)
            vtt_spans.append((float(span[0]), float(span[1])))
        vtt_bytes = realign.render_vtt(vtt_blocks, vtt_spans).encode("utf-8")
    except Exception as exc:
        raise PublicationEncodeError("vtt-encode", exc) from exc

    try:
        segments = [
            {
                "text": str(block["text"]),
                "start": float(span[0]),
                "end": float(span[1]),
            }
            | ({"lyric": True} if block.get("lyric") is True else {})
            for block, span in zip(blocks, spans, strict=True)
        ]
        units = [
            {
                "text": str(unit["text"]),
                "start": float(unit["start"]),
                "end": float(unit["end"]),
            }
            for unit in all_units
        ]
        data: dict[str, Any] = {
            "language": language,
            "segments": segments,
            "word_segments": units,
        }
        if vad_speech is not None:
            data["vad_speech"] = [[float(s), float(e)] for s, e in vad_speech]
        if shot_changes is not None:
            data["shot_changes"] = [float(v) for v in shot_changes]
        if sing_spans is not None:
            data["sing_spans"] = [[float(s), float(e)] for s, e in sing_spans]
        turns_present, turns_val = _speaker_turns_json(speaker_turns)
        if turns_present:
            assert turns_val is not None
            data["speaker_turns"] = turns_val
        if voiceprint_pair is not None:
            data["voiceprint_capture"] = voiceprint_pair[0]
            data["voiceprint_media"] = voiceprint_pair[1]
        if manifest is not None:
            data["segmentation"] = dict(manifest)

        frozen = freeze_json(data)
        if not isinstance(frozen, FrozenObject):  # pragma: no cover - dict invariant
            raise TypeError("align main projection is not an object")
        main_json_bytes = encode_frozen_json_document(frozen, allow_nan=True)
    except Exception as exc:
        raise PublicationEncodeError("main-json-encode", exc) from exc

    return EncodedPrimaryOutputs(vtt_bytes, main_json_bytes)


__all__ = [
    "EncodedPrimaryOutputs",
    "PublicationEncodeError",
    "encode_align_outputs",
    "encode_segmentation_outputs",
    "semantic_speaker_turns_carrier",
]
