"""Public-command bridge into the typed segmentation candidate schedule."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from voxweave import segmentation_candidates
from voxweave.align_context import (
    IssuedSegmentationContext,
    issue_segmentation_context,
    retire_live_context_roles,
    verify_context_roles_terminal,
)
from voxweave.align_snapshot import (
    FrozenArray,
    FrozenObject,
    RawJSONCarrier,
    freeze_json,
    frozen_json_digest,
)
from voxweave.candidate_encoder import VerifiedEncodedCandidate
from voxweave.core.partition_check import owned_unit_ids
from voxweave.core.schema import Cue
from voxweave.core.segdoc import SegDocument
from voxweave.episode_transaction import (
    FileGeneration,
    ProcessSourceMode,
    SpeakerMappingGeneration,
    bind_split_speaker_mapping_generation,
    release_split_speaker_mapping_generation,
)
from voxweave.segmentation_adapter import (
    SegmentationAdapterResult,
    SegmentationCarriers,
    SegmentationDelivery,
    SegmentationDeliveryCue,
    SegmentationProjectionInputs,
    issue_legacy_segmentation,
    run_locked_segmentation_adapter,
)


SegmentationCommand = Literal["process", "split"]


def _swap_ext(path: Path, new_ext: str) -> Path:
    target = Path(path)
    if target.suffix:
        return target.with_name(target.name[: -len(target.suffix)] + new_ext)
    return target.with_name(target.name + new_ext)


@dataclass(frozen=True)
class SegmentationSelection:
    context: IssuedSegmentationContext
    result: SegmentationAdapterResult
    verified: VerifiedEncodedCandidate
    sdh_dialogue: tuple[dict[str, object], ...]


def _object(value: object) -> FrozenObject:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenObject):
        raise TypeError("segmentation object projection is not an object")
    return frozen


def _array(value: object) -> FrozenArray:
    frozen = freeze_json(value)
    if not isinstance(frozen, FrozenArray):
        raise TypeError("segmentation array projection is not an array")
    return frozen


def semantic_speaker_turns_carrier(
    turns: Sequence[tuple[float, float, str]] | None,
) -> RawJSONCarrier:
    if turns is None:
        return RawJSONCarrier(False, None)
    return RawJSONCarrier(
        True,
        _array([[float(start), float(end), str(label)] for start, end, label in turns]),
    )


def _generation_value(generation: FileGeneration) -> dict[str, object]:
    return {
        "present": generation.present,
        "size": None if generation.bytes_value is None else len(generation.bytes_value),
        "sha256": generation.sha256,
    }


def _mapping_value(
    generation: SpeakerMappingGeneration | None,
) -> dict[str, object] | None:
    if generation is None:
        return None
    return {
        "kind": generation.kind,
        "readable_size": (
            None if generation.bytes_value is None else len(generation.bytes_value)
        ),
        "readable_sha256": (
            None
            if generation.bytes_value is None
            else hashlib.sha256(generation.bytes_value).hexdigest()
        ),
        "loader_status": generation.loader_status,
        "names": [list(item) for item in generation.names],
    }


def _partition(
    document: SegDocument, cues: Sequence[Cue]
) -> tuple[tuple[int, int], ...]:
    from voxweave.core.smart_split import _surface_ranges

    word_data = [entry for cue in cues for entry in cue.get("word_data") or ()]
    ranges = _surface_ranges([unit.surface for unit in document.units], word_data)
    if ranges is None or len(ranges) != len(document.units):
        return tuple((0, 0) for _cue in cues)
    boundaries = {
        ranges[index - 1][1]: index
        for index in range(1, len(ranges))
        if ranges[index - 1][1] == ranges[index][0]
    }
    cursor = 0
    cuts: list[int] = []
    for cue in cues[:-1]:
        cursor += len(cue.get("word_data") or ())
        cut = boundaries.get(cursor)
        if cut is None:
            return tuple((0, 0) for _cue in cues)
        cuts.append(cut)
    if cues:
        cursor += len(cues[-1].get("word_data") or ())
    if cursor != len(word_data):
        return tuple((0, 0) for _cue in cues)
    projected = tuple(owned_unit_ids(tuple(cuts), len(document.units)))
    if len(projected) != len(cues) or any(low >= high for low, high in projected):
        return tuple((0, 0) for _cue in cues)
    return projected


def _delivery_cue(cue: Cue, unit_range: tuple[int, int]) -> SegmentationDeliveryCue:
    word_data = tuple(_object(dict(unit)) for unit in cue.get("word_data") or ())
    speaker_ids_value = cue.get("speaker_ids")
    speaker_ids = (
        None
        if speaker_ids_value is None
        else tuple(str(value) for value in speaker_ids_value)
    )
    speech_start = cue.get("speech_start")
    speech_end = cue.get("speech_end")
    return SegmentationDeliveryCue(
        unit_range,
        str(cue["text"]),
        float(cue["start"]),
        float(cue["end"]),
        word_data,
        None if speech_start is None else float(speech_start),
        None if speech_end is None else float(speech_end),
        True if cue.get("lyric") is True else None,
        speaker_ids,
    )


def _carriers(
    *,
    vad_speech: Sequence[tuple[float, float]] | None,
    shot_changes: Sequence[float] | None,
    sing_spans: Sequence[tuple[float, float]] | None,
    speaker_turns: RawJSONCarrier,
    voiceprint_pair: tuple[str, str] | None,
) -> SegmentationCarriers:
    return SegmentationCarriers(
        tuple(_array([float(start), float(end)]) for start, end in vad_speech or ()),
        (
            None
            if shot_changes is None
            else tuple(freeze_json(float(value)) for value in shot_changes)
        ),
        (
            None
            if sing_spans is None
            else tuple(_array([float(start), float(end)]) for start, end in sing_spans)
        ),
        speaker_turns,
        None if voiceprint_pair is None else voiceprint_pair[0],
        None if voiceprint_pair is None else voiceprint_pair[1],
    )


def build_segmentation_selection(
    *,
    command: SegmentationCommand,
    target_path: Path,
    sibling_path: Path,
    language: str,
    cues: Sequence[Cue],
    top_level_units: Sequence[Mapping[str, Any]],
    document: SegDocument,
    manifest: Mapping[str, Any],
    vad_speech: Sequence[tuple[float, float]] | None,
    shot_changes: Sequence[float] | None,
    sing_spans: Sequence[tuple[float, float]] | None,
    speaker_turns: RawJSONCarrier,
    voiceprint_pair: tuple[str, str] | None,
    timestamps: bool,
    speaker_names: Sequence[tuple[str, str]],
    expected_json: FileGeneration,
    expected_vtt: FileGeneration | None,
    source_mode: ProcessSourceMode | None,
    mapping_generation: SpeakerMappingGeneration | None,
    shadow_enabled: bool,
    semantic_selector_enabled: bool,
) -> SegmentationSelection:
    """Run adapter, one composite encode, selection, and independent projection."""
    if command == "process" and source_mode is None:
        raise ValueError("process selection requires a source mode")
    if command == "split" and source_mode is not None:
        raise ValueError("split selection cannot carry a process source mode")
    stable = _object(
        {
            "command": command,
            "source_mode": source_mode,
            "json_generation": _generation_value(expected_json),
            "vtt_generation": (
                None if expected_vtt is None else _generation_value(expected_vtt)
            ),
            "speaker_mapping": _mapping_value(mapping_generation),
            "timestamps": timestamps,
            "semantic_selector_enabled": semantic_selector_enabled,
            "speaker_turns": {
                "present": speaker_turns.present,
                "digest": (
                    None
                    if speaker_turns.value is None
                    else frozen_json_digest(speaker_turns.value)
                ),
            },
            "voiceprint_pair": (
                None if voiceprint_pair is None else list(voiceprint_pair)
            ),
            "manifest": dict(manifest),
            "top_level_units": [dict(unit) for unit in top_level_units],
        }
    )
    context = issue_segmentation_context(
        stable_fields=stable,
        target_path=target_path,
        sibling_path=sibling_path,
        effective_iso=language,
    )
    if command == "split":
        if mapping_generation is None:
            retire_live_context_roles(context)
            verify_context_roles_terminal(context)
            raise ValueError("split selection requires its S0 mapping generation")
        bind_split_speaker_mapping_generation(
            context,
            _swap_ext(sibling_path, ".speakers.json"),
            mapping_generation,
        )
    elif mapping_generation is not None:
        retire_live_context_roles(context)
        verify_context_roles_terminal(context)
        raise ValueError("process selection cannot bind a speaker mapping generation")
    try:
        ranges = _partition(document, cues)
        delivery = SegmentationDelivery(
            context.context_content_digest,
            "legacy-v1",
            language,
            tuple(
                _delivery_cue(cue, unit_range) for cue, unit_range in zip(cues, ranges)
            ),
            tuple(_object(dict(unit)) for unit in top_level_units),
            _carriers(
                vad_speech=vad_speech,
                shot_changes=shot_changes,
                sing_spans=sing_spans,
                speaker_turns=speaker_turns,
                voiceprint_pair=voiceprint_pair,
            ),
            _object(dict(manifest)),
        )
        projection_inputs = SegmentationProjectionInputs(
            timestamps=timestamps,
            speaker_names=tuple(speaker_names),
        )
        issued = issue_legacy_segmentation(
            context,
            delivery=delivery,
            projection_inputs=projection_inputs,
            document=document,
        )
        result = run_locked_segmentation_adapter(
            context,
            issued,
            shadow_enabled=shadow_enabled,
            semantic_selector_enabled=semantic_selector_enabled,
        )
        candidates = segmentation_candidates.encode_segmentation_candidates(
            context, result
        )
        selected = segmentation_candidates.select_segmentation_candidate(
            context, candidates
        )
        verified = segmentation_candidates.verify_selected_segmentation_projection(
            context, result, selected
        )
        sdh_dialogue = segmentation_candidates.project_selected_sdh_dialogue(
            context, result, verified
        )
    except BaseException:
        release_split_speaker_mapping_generation(context)
        retire_live_context_roles(context)
        verify_context_roles_terminal(context)
        raise
    return SegmentationSelection(context, result, verified, sdh_dialogue)


def retire_segmentation_selection(selection: SegmentationSelection) -> None:
    release_split_speaker_mapping_generation(selection.context)
    retire_live_context_roles(selection.context)
    verify_context_roles_terminal(selection.context)


__all__ = [
    "SegmentationSelection",
    "build_segmentation_selection",
    "retire_segmentation_selection",
    "semantic_speaker_turns_carrier",
]
