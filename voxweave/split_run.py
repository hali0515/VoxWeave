"""The plan and commit stages of one split call.

``pipeline.split`` reads the sibling JSON, lays the cues out, then runs these
two stages: bind the laid-out cues to the generations that were read and build
the verified selection, then commit it. ``SplitSource`` is what the read stage
hands over. This module imports nothing from ``voxweave.pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxweave import episode_transaction
from voxweave.core.shadow_v2 import shadow_enabled
from voxweave.paths import swap_ext
from voxweave.segmentation import SegmentationResult


class VttEditsKeptError(RuntimeError):
    """``split(keep_vtt_edits=True)`` found cues the JSON would not reproduce."""


def _cue_content(vtt_path: Path, raw: bytes) -> list[tuple[Any, Any, str, bool]] | None:
    """Cue times, text and lyric flag of a VTT, without speaker names (None = unreadable)."""
    from voxweave.subformats import load_subtitle_blocks_bytes

    try:
        blocks = load_subtitle_blocks_bytes(vtt_path, raw)
    except (RuntimeError, ValueError, UnicodeError):
        return None
    return [
        (block.get("start"), block.get("end"), block["text"], bool(block.get("lyric")))
        for block in blocks
    ]


@dataclass(frozen=True)
class SplitSource:
    # Everything ``split`` reads before it lays anything out: the exact sibling
    # JSON bytes, the replay carriers decoded from them, and the speaker mapping.
    json_path: Path
    owner: Path
    input_bytes: bytes
    voiceprint_pair: tuple[str, str] | None
    pair_declared: bool
    iso: str
    units: list[dict]
    speech_spans: list[tuple[float, float]] | None
    shot_changes: list[float] | None
    sing_spans: list[tuple[float, float]] | None
    speaker_turns: list[tuple[float, float, str]] | None
    annotate_speakers: bool
    mapping_path: Path
    mapping_generation: episode_transaction.SpeakerMappingGeneration


@dataclass(frozen=True)
class SplitCommit:
    # What ``split`` publishes: the verified selection and the generations the
    # commit must still find on disk.
    vtt_path: Path
    selection: Any
    expected_json: episode_transaction.FileGeneration
    expected_vtt: episode_transaction.FileGeneration | None
    cleanup: tuple[episode_transaction.ArtifactCleanup, ...]


def plan_split_commit(
    source: SplitSource,
    segmented: SegmentationResult,
    *,
    timestamps: bool,
    keep_vtt_edits: bool,
) -> SplitCommit:
    # Select stage of split: bind the laid-out cues to the generations read
    # earlier and build the verified selection with its cleanup list.
    from voxweave import segmentation_orchestration
    from voxweave.align_snapshot import decode_sibling_json_snapshot

    json_path = source.json_path
    if segmented.document is None or segmented.manifest is None:
        raise RuntimeError("segmentation result lacks its production authority")
    expected_json = episode_transaction.FileGeneration(True, source.input_bytes)
    vtt_path = swap_ext(json_path, ".vtt")
    expected_vtt: episode_transaction.FileGeneration | None = None
    if keep_vtt_edits:
        try:
            expected_vtt = episode_transaction.FileGeneration(
                True, vtt_path.read_bytes()
            )
        except FileNotFoundError:
            expected_vtt = episode_transaction.FileGeneration(False, None)
    lexical_snapshot = decode_sibling_json_snapshot(json_path.name, source.input_bytes)
    selection = segmentation_orchestration.build_segmentation_selection(
        command="split",
        target_path=swap_ext(json_path, ".vtt"),
        sibling_path=json_path,
        language=source.iso,
        cues=segmented.cues,
        top_level_units=segmented.units,
        document=segmented.document,
        manifest=segmented.manifest,
        vad_speech=source.speech_spans,
        shot_changes=source.shot_changes,
        sing_spans=source.sing_spans,
        speaker_turns=lexical_snapshot.carrier("speaker_turns"),
        voiceprint_pair=source.voiceprint_pair,
        timestamps=timestamps,
        speaker_names=source.mapping_generation.names,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        source_mode=None,
        mapping_generation=source.mapping_generation,
        mapping_path=source.mapping_path,
        shadow_enabled=shadow_enabled(),
    )
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if source.pair_declared and source.voiceprint_pair is None:
        cleanup.extend(
            episode_transaction.speaker_artifact_cleanups(source.owner, json_path)
        )
    cleanup.extend(
        episode_transaction.evidence_cleanups(
            swap_ext(json_path, ".vtt"), media=source.owner
        )
    )
    return SplitCommit(vtt_path, selection, expected_json, expected_vtt, tuple(cleanup))


def commit_split(source: SplitSource, plan: SplitCommit) -> None:
    # Write stage of split: refuse to replace hand edits when asked to keep them,
    # commit, and retire the selection however the commit ends.
    from voxweave import segmentation_orchestration

    json_path = source.json_path
    vtt_path = plan.vtt_path
    selection = plan.selection
    expected_vtt = plan.expected_vtt
    try:
        if expected_vtt is not None and expected_vtt.bytes_value is not None:
            old = _cue_content(vtt_path, expected_vtt.bytes_value)
            if old is None or old != _cue_content(
                vtt_path, selection.verified.vtt_bytes
            ):
                raise VttEditsKeptError(
                    f"{vtt_path.name} was not overwritten: its cue text or timing"
                    f" differs from what {json_path.name} produces (hand edits, align"
                    " output, or other render options); run"
                    f" 'voxweave render {json_path}' to replace it anyway"
                )
        episode_transaction.commit_primary_outputs(
            command="split",
            episode_path=source.owner,
            json_path=json_path,
            vtt_path=vtt_path,
            expected_json=plan.expected_json,
            expected_vtt=expected_vtt,
            main_json_bytes=selection.verified.main_json_bytes,
            vtt_bytes=selection.verified.vtt_bytes,
            cleanup_paths=plan.cleanup,
            context=selection.context,
            speaker_mapping_path=source.mapping_path,
            expected_speaker_mapping=source.mapping_generation,
        )
    finally:
        segmentation_orchestration.retire_segmentation_selection(selection)
