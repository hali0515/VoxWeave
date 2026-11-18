"""The stages of one align call after its inputs are read.

``pipeline.align`` reads the inputs, then runs these stages in order: acquire
the 16 kHz audio, select the aligned spans, encode the primary outputs, commit
them, and dispose of the run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voxweave import backend, config, episode_transaction, publication, realign
from voxweave.align_route import AlignRoute
from voxweave.core.overlay import spans_in
from voxweave.failure_annotations import (
    _attach_canonical_failure,
    _record_disposal_failure,
)
from voxweave.mediasnapshot import MediaSnapshot
from voxweave.progress import Reporter

log = logging.getLogger("voxweave")

# Align-stage cue duration floor. Default 0 (disabled): enforce_min_duration only
# resolves overlaps without padding, so short back-channels keep their real ~0.6s.
# Set VOXWEAVE_MIN_CUE_SEC=0.8 to re-enable padding. Distinct from VOXWEAVE_SEG_MIN_CUE_SEC.
MIN_CUE_SEC = config._env_float("VOXWEAVE_MIN_CUE_SEC", 0.0)
# Flash-cue rescue (orthogonal to MIN_CUE_SEC): genuine flash cues (so/あ at 0.1-0.2s)
# are extended to TINY_CUE_TARGET, allowed to overlap only the immediately following cue.
# VOXWEAVE_TINY_CUE_SEC=0 disables.
TINY_CUE_SEC = config._env_float("VOXWEAVE_TINY_CUE_SEC", 0.2)
TINY_CUE_TARGET = config._env_float("VOXWEAVE_TINY_CUE_TARGET", 0.5)


@dataclass(frozen=True)
class SelectedLegacyAlignResult:
    block_units: tuple[tuple[Mapping[str, Any], ...], ...]
    spans: tuple[tuple[float, float], ...]
    all_units: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CachedMediaIdentity:
    """Selected-media identity for align when a bound vocals cache serves the audio."""

    fingerprint: str


@dataclass(frozen=True)
class AlignInputs:
    # Everything align reads before any model work: the VTT and sibling JSON
    # bytes, their decoded snapshots, the selected media and the route plan.
    vtt_path: Path
    json_path: Path
    vtt_input_bytes: bytes
    expected_json: episode_transaction.FileGeneration
    json_input_bytes: bytes | None
    sibling_snapshot: Any
    data: dict
    pair_declared: bool
    voiceprint_pair: tuple[str, str] | None
    iso: str
    input_snapshot: Any
    media: Path
    media_input_fingerprint: str
    explicit_media: bool
    mms: bool
    ctc_model: str | None
    route: AlignRoute
    blocks: list[dict[str, Any]]
    crops: list[tuple[float, float] | None]


@dataclass
class AlignRun:
    # Mutable state of one align call. The disposal stage reads it however the
    # call ends, so each field is bound at the point the call reaches it.
    tmp: list[Path] = field(default_factory=list)
    tmp_chunks: list[Path] = field(default_factory=list)
    snapshots: ExitStack = field(default_factory=ExitStack)
    selected_snapshot: MediaSnapshot | CachedMediaIdentity | None = None
    aligned_cue_count: int = 0
    aligned_unit_count: int = 0


def seal_selected_legacy_align_result(
    block_units: Sequence[Sequence[Mapping[str, Any]]],
    spans: Sequence[tuple[float, float]],
    all_units: Sequence[Mapping[str, Any]],
) -> SelectedLegacyAlignResult:
    """Seal the selected alignment result."""

    def seal_unit(unit: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "text": unit["text"],
            "start": unit["start"],
            "end": unit["end"],
        }

    return SelectedLegacyAlignResult(
        tuple(tuple(seal_unit(unit) for unit in owner) for owner in block_units),
        tuple((start, end) for start, end in spans),
        tuple(seal_unit(unit) for unit in all_units),
    )


def begin_fresh_acquisition(
    inputs: AlignInputs,
    run: AlignRun,
    *,
    acquire_audio: Callable[
        ..., tuple[Path, MediaSnapshot | CachedMediaIdentity | None]
    ],
    rep: Reporter,
    separate: bool,
    normalize: bool,
) -> Path:
    wav, run.selected_snapshot = acquire_audio(
        inputs.media,
        separate=separate,
        normalize=normalize,
        bind_voiceprint=inputs.voiceprint_pair is not None,
        media_input_fingerprint=inputs.media_input_fingerprint,
        rep=rep,
        tmp=run.tmp,
        snapshots=run.snapshots,
    )
    return wav


def build_align_selection(
    inputs: AlignInputs,
    run: AlignRun,
    selected_legacy: SelectedLegacyAlignResult,
) -> tuple[
    publication.EncodedPrimaryOutputs, bool, list[episode_transaction.ArtifactCleanup]
]:
    data = inputs.data
    voiceprint_pair = inputs.voiceprint_pair
    selected_snapshot = run.selected_snapshot
    # Preserve vad_speech / shot_changes from the original JSON (computed by
    # transcribe from the original media; align does not recompute them).
    keep_vad = spans_in(data.get("vad_speech"))
    keep_shots = [float(t) for t in data.get("shot_changes") or []] or None
    keep_sing = spans_in(data.get("sing_spans"))
    keep_turns = inputs.sibling_snapshot.carrier("speaker_turns")
    # align never re-segments, so the segmentation manifest is preserved
    # verbatim (and stays absent when the document never had one).
    stored_manifest = data.get("segmentation")
    keep_manifest = stored_manifest if isinstance(stored_manifest, Mapping) else None
    preserve_pair = bool(
        voiceprint_pair is not None
        and selected_snapshot is not None
        and selected_snapshot.fingerprint == voiceprint_pair[1]
    )
    if (
        voiceprint_pair is not None
        and selected_snapshot is not None
        and not preserve_pair
    ):
        log.warning(
            "voiceprint binding omitted during align: "
            "selected media does not match the sibling binding"
        )
    encoded = publication.encode_align_outputs(
        blocks=inputs.blocks,
        spans=selected_legacy.spans,
        all_units=selected_legacy.all_units,
        language=inputs.iso,
        vad_speech=keep_vad,
        shot_changes=keep_shots,
        sing_spans=keep_sing,
        speaker_turns=keep_turns,
        voiceprint_pair=voiceprint_pair if preserve_pair else None,
        manifest=keep_manifest,
    )
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if inputs.pair_declared and not preserve_pair:
        cleanup.extend(
            episode_transaction.speaker_artifact_cleanups(inputs.media, inputs.vtt_path)
        )
    return encoded, preserve_pair, cleanup


def commit_align_outputs(
    inputs: AlignInputs,
    run: AlignRun,
    encoded: publication.EncodedPrimaryOutputs,
    preserve_pair: bool,
    cleanup: list[episode_transaction.ArtifactCleanup],
    *,
    rep: Reporter,
) -> None:
    media = inputs.media
    vtt_path = inputs.vtt_path
    voiceprint_pair = inputs.voiceprint_pair
    rep.step("write outputs")
    rep.stage("write VTT + JSON")
    episode_transaction.commit_primary_outputs(
        command="align",
        episode_path=media,
        json_path=inputs.json_path,
        vtt_path=vtt_path,
        expected_json=inputs.expected_json,
        expected_vtt=episode_transaction.FileGeneration(True, inputs.vtt_input_bytes),
        main_json_bytes=encoded.main_json_bytes,
        vtt_bytes=encoded.vtt_bytes,
        cleanup_paths=tuple(cleanup),
        media_path=media,
        expected_media_fingerprint=inputs.media_input_fingerprint,
        expected_voiceprint_media_fingerprint=(
            voiceprint_pair[1]
            if voiceprint_pair is not None and run.selected_snapshot is not None
            else None
        ),
        expected_pair_decision=(
            preserve_pair
            if voiceprint_pair is not None and run.selected_snapshot is not None
            else None
        ),
    )


def select_legacy_alignment(
    block_units: Sequence[Sequence[Mapping[str, Any]]],
    final: list[tuple[float, float] | None],
    all_units: Sequence[Mapping[str, Any]],
    media: Path,
) -> SelectedLegacyAlignResult:
    if not all_units:
        exc = RuntimeError(f"no aligned units for {media.name}")
        _attach_canonical_failure(
            exc,
            kind="no-aligned-units",
            phase="fresh-acquisition",
            detail_code="all-block-units-empty",
        )
        raise exc
    filled = realign.fill_insert_blocks(final)
    duration_enforced = realign.enforce_min_duration(
        filled,
        min_dur=MIN_CUE_SEC,
    )
    rescued = realign.rescue_tiny_cues(
        duration_enforced,
        trig=TINY_CUE_SEC,
        target=TINY_CUE_TARGET,
    )
    spans_filled = realign.clamp_spans(rescued)
    return seal_selected_legacy_align_result(
        block_units,
        spans_filled,
        all_units,
    )


def dispose_align_run(run: AlignRun, production_failure: BaseException | None) -> None:
    disposal_failure: BaseException | None = None
    try:
        run.snapshots.close()
    except BaseException as exc:
        disposal_failure = _record_disposal_failure(
            production_failure,
            disposal_failure,
            exc,
            detail_code="media-snapshot-residue",
        )
    try:
        backend.release()
    except Exception as exc:
        log.warning("releasing the alignment models failed: %r", exc)
    for p in run.tmp:
        try:
            p.unlink(missing_ok=True)
        except BaseException as exc:
            disposal_failure = _record_disposal_failure(
                production_failure,
                disposal_failure,
                exc,
                detail_code="audio-temp-residue",
            )
    for c in run.tmp_chunks:
        try:
            c.unlink(missing_ok=True)
        except BaseException as exc:
            disposal_failure = _record_disposal_failure(
                production_failure,
                disposal_failure,
                exc,
                detail_code="audio-temp-residue",
            )
    if production_failure is None and disposal_failure is not None:
        raise disposal_failure
