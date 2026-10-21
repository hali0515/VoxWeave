from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxweave import artifacts, episode_transaction
from voxweave import sdh as sdh_mod
from voxweave.chunking import decode_to_wav
from voxweave.core.shadow_v2 import shadow_enabled
from voxweave.paths import swap_ext
from voxweave.progress import Reporter, progress_bridge
from voxweave.segmentation import segment_document
from voxweave.sidecars import inspect_speakers_mapping_path, voiceprints_path
from voxweave.speakers import named_speaker_ids
from voxweave.vocals import SONGDET_SR
from voxweave.voicebase import (
    Phase2DataError,
    VOICEPRINTS_MAX_BYTES,
    canonical_turns_digest,
    encode_json_bytes,
    media_fingerprint,
    mint_capture_id,
    utc_timestamp,
    validate_voiceprints_mapping,
)
from voxweave.voiceprint_capture import VoiceprintCapture

log = logging.getLogger("voxweave")


@dataclass(frozen=True)
class _ProcessPublication:
    path: Path
    landed: tuple[Path, ...]
    auxiliary_landed: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _ProcessSource:
    """The sealed identity of one process run's source and sibling generations."""

    path: Path
    snapshot_fingerprint: str | None
    expected_json: episode_transaction.FileGeneration
    expected_vtt: episode_transaction.FileGeneration
    expected_media_fingerprint: str | None
    mode: episode_transaction.ProcessSourceMode


def _voiceprint_capture_from_generation(
    generation: episode_transaction.FileGeneration,
) -> str | None:
    if generation.bytes_value is None:
        return None
    try:
        value = json.loads(generation.bytes_value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(value, dict) and isinstance(value.get("voiceprint_capture"), str):
        return value["voiceprint_capture"]
    return None


def _warn_stale_speaker_names(
    media_path: Path,
    previous_json: episode_transaction.FileGeneration,
    speaker_turns: Sequence[tuple[float, float, str]] | None,
) -> None:
    """Warn when saved speaker names were given to turns this run replaced.

    The speaker mapping keys names by ``SPEAKER_NN`` id alone. A re-run whose
    turns differ (another ``--speaker-clustering`` or ``--diarize-model``, a
    voiceprint stage that fell back to pyannote, new audio) can hand an id to
    another voice, and its saved name follows the id into rendered subtitles,
    the speakers page and ``speakers enroll`` (which would store that voice
    under the name in the voice library). Names whose id kept exactly the
    same turns are not reported. Never raises: this is advice, not a gate.
    """
    if not speaker_turns or previous_json.bytes_value is None:
        return
    try:
        previous = json.loads(previous_json.bytes_value.decode("utf-8"))
        raw_old = previous.get("speaker_turns") if isinstance(previous, dict) else None
        mapping = inspect_speakers_mapping_path(media_path)
        if not raw_old or not artifacts.path_present(mapping):
            return
        named = named_speaker_ids(mapping)
        old = [(float(s), float(e), str(label)) for s, e, label in raw_old]
    except (OSError, RuntimeError, UnicodeError, ValueError, TypeError):
        return

    def spans_by_id(turns: Sequence[tuple[float, float, str]]) -> dict[str, list]:
        spans: dict[str, list] = {}
        for start, end, label in turns:
            spans.setdefault(label, []).append((float(start), float(end)))
        return {label: sorted(values) for label, values in spans.items()}

    before, after = spans_by_id(old), spans_by_id(speaker_turns)
    stale = sorted(i for i in named if before.get(i) != after.get(i))
    if stale:
        log.warning(
            "%s names %s, but this run's speaker turns for %s differ from those "
            "the names were given to (changing --speaker-clustering or "
            "--diarize-model renumbers speakers), so a name may now label another "
            "voice; review the names with `voxweave speakers %s` before "
            "`voxweave speakers enroll`",
            mapping.name,
            ", ".join(stale),
            "that id" if len(stale) == 1 else "those ids",
            media_path.name,
        )


def _voiceprints_document(
    media_path: Path,
    capture: VoiceprintCapture,
    *,
    capture_id: str,
    source_fingerprint: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "capture_id": capture_id,
        "provenance": capture.provenance,
        "binding": {
            "turns_digest": canonical_turns_digest(capture.turns),
            "media_fingerprint": source_fingerprint,
            "media_stem": media_path.stem,
            "created": utc_timestamp(),
        },
        "speakers": capture.centroids,
    }
    validate_voiceprints_mapping(value)
    return value


def _finish_process_from_units(
    media_path: Path,
    *,
    rep: Reporter,
    iso: str,
    units: list[dict],
    vad_speech: list[tuple[float, float]] | None,
    shot_changes: list[float] | None,
    sing_spans: list[tuple[float, float]] | None,
    speaker_turns: list[tuple[float, float, str]] | None,
    timestamps: bool,
    capture: VoiceprintCapture | None,
    source: _ProcessSource,
    sdh_enabled: bool,
) -> _ProcessPublication:
    """Finish segmentation and publication after source ownership is sealed."""
    rep.step("layout subtitles")
    rep.stage("smart_split layout")
    segmented = segment_document(
        language=iso,
        word_segments=units,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
    )
    units, cues = segmented.units, segmented.cues

    rep.step("write outputs")
    rep.stage("write siblings")
    from voxweave import segmentation_orchestration

    if segmented.document is None or segmented.manifest is None:
        raise RuntimeError("segmentation result lacks its production authority")
    if capture is not None and capture.turns is not speaker_turns:
        raise RuntimeError("voiceprint capture turns diverged from sibling turns")

    voiceprint_pair: tuple[str, str] | None = None
    machine_artifact: episode_transaction.MachineArtifactPublication | None = None
    if capture is not None and source.snapshot_fingerprint is not None:
        capture_id = mint_capture_id(
            current=_voiceprint_capture_from_generation(source.expected_json)
        )
        try:
            sidecar = _voiceprints_document(
                media_path,
                capture,
                capture_id=capture_id,
                source_fingerprint=source.snapshot_fingerprint,
            )
            sidecar_bytes = encode_json_bytes(sidecar, max_bytes=VOICEPRINTS_MAX_BYTES)
        except Phase2DataError as exc:
            log.warning("voiceprint capture dropped: %s", exc)
        else:
            voiceprint_pair = (capture_id, source.snapshot_fingerprint)
            machine_artifact = episode_transaction.MachineArtifactPublication(
                voiceprints_path(media_path), sidecar_bytes
            )

    selection = segmentation_orchestration.build_segmentation_selection(
        command="process",
        target_path=swap_ext(media_path, ".vtt"),
        sibling_path=swap_ext(media_path, ".json"),
        language=iso,
        cues=cues,
        top_level_units=units,
        document=segmented.document,
        manifest=segmented.manifest,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=segmentation_orchestration.semantic_speaker_turns_carrier(
            speaker_turns
        ),
        voiceprint_pair=voiceprint_pair,
        timestamps=timestamps,
        speaker_names=(),
        expected_json=source.expected_json,
        expected_vtt=source.expected_vtt,
        source_mode=source.mode,
        mapping_generation=None,
        mapping_path=None,
        shadow_enabled=shadow_enabled(),
    )
    cleanup = episode_transaction.speaker_artifact_cleanups(
        media_path, media_path, voiceprints=machine_artifact is None
    )
    cleanup.extend(
        episode_transaction.evidence_cleanups(
            swap_ext(media_path, ".vtt"), media=media_path
        )
    )
    try:
        receipt = episode_transaction.commit_primary_outputs(
            command="process",
            episode_path=media_path,
            json_path=swap_ext(media_path, ".json"),
            vtt_path=swap_ext(media_path, ".vtt"),
            expected_json=source.expected_json,
            expected_vtt=source.expected_vtt,
            main_json_bytes=selection.verified.main_json_bytes,
            vtt_bytes=selection.verified.vtt_bytes,
            cleanup_paths=tuple(cleanup),
            context=selection.context,
            media_path=(
                media_path if source.expected_media_fingerprint is not None else None
            ),
            expected_media_fingerprint=source.expected_media_fingerprint,
            machine_artifact=machine_artifact,
        )
    finally:
        segmentation_orchestration.retire_segmentation_selection(selection)
    vtt_out = swap_ext(media_path, ".vtt")
    landed = receipt.landed
    selected_sdh_cues: Sequence[Mapping[str, Any]] = selection.sdh_dialogue
    if machine_artifact is not None:
        log.info("wrote voice-biometric sidecar %s", machine_artifact.path.name)
    log.info("wrote %s + .json (%d cues, lang=%s)", vtt_out.name, len(cues), iso)
    _warn_stale_speaker_names(media_path, source.expected_json, speaker_turns)
    auxiliary_landed: tuple[Path, ...] = ()
    if sdh_enabled and source.mode == "transcribed-media":
        rep.step("create SDH sidecar")
        committed_json = episode_transaction.capture_file_generation(
            swap_ext(media_path, ".json")
        )
        committed_vtt = episode_transaction.capture_file_generation(vtt_out)
        try:
            sidecar = _write_sdh_sidecar(
                media_path,
                selected_sdh_cues,
                rep,
                expected_json_generation=committed_json,
                expected_vtt_generation=committed_vtt,
                expected_media_fingerprint=source.expected_media_fingerprint,
            )
            if sidecar is not None:
                auxiliary_landed = (sidecar,)
        except Exception as exc:
            log.warning("SDH sidecar failed (non-fatal): %r", exc)
    return _ProcessPublication(vtt_out, landed, auxiliary_landed)


def _write_sdh_sidecar(
    media_path: Path,
    cues: Sequence[Mapping[str, Any]],
    rep: Reporter,
    *,
    expected_json_generation: episode_transaction.FileGeneration | None = None,
    expected_vtt_generation: episode_transaction.FileGeneration | None = None,
    expected_media_fingerprint: str | None = None,
) -> Path | None:
    """Detect non-speech events on the ORIGINAL mix (effects are stripped from the
    separated-vocals stem) and write ``<stem>.sdh.vtt`` (dialogue + event tags).
    Returns None when panns-inference is missing (warned, non-fatal)."""
    rep.stage("SDH event detection (PANNs)")
    wav32 = decode_to_wav(media_path, sample_rate=SONGDET_SR)
    try:
        events = sdh_mod.detect_events(
            wav32, progress=progress_bridge(rep, "SDH event detection (PANNs)")
        )
    except ModuleNotFoundError as e:
        log.warning(
            "SDH detection requires panns-inference (not installed: %s); skipping sidecar",
            e,
        )
        return None
    finally:
        wav32.unlink(missing_ok=True)
    events = sdh_mod.fit_events_to_gaps(events, cues)
    path = swap_ext(media_path, ".sdh.vtt")
    json_path = swap_ext(media_path, ".json")
    vtt_path = swap_ext(media_path, ".vtt")
    expected_json = (
        expected_json_generation
        or episode_transaction.capture_file_generation(json_path)
    )
    expected_vtt = (
        expected_vtt_generation or episode_transaction.capture_file_generation(vtt_path)
    )
    expected_media = expected_media_fingerprint or media_fingerprint(media_path)
    landed = episode_transaction.commit_auxiliary_sdh(
        episode_path=media_path,
        sidecar_path=path,
        sidecar_bytes=sdh_mod.render_sdh_vtt(cues, events).encode("utf-8"),
        json_path=json_path,
        expected_json=expected_json,
        vtt_path=vtt_path,
        expected_vtt=expected_vtt,
        media_path=media_path,
        expected_media_fingerprint=expected_media,
    )
    if not landed:
        log.warning("stale SDH sidecar discarded; primaries or media changed")
        return None
    log.info("wrote %s (%d event tag(s))", path.name, len(events))
    return path
