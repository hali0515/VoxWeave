"""The stages of one align call after its inputs are read.

``pipeline.align`` reads the inputs, then runs these stages in order: open the
fresh acquisition, select the legacy result, build the verified selection,
commit the outputs and dispose of the run. The records here carry what the
stages share; ``AlignRun`` holds the state that disposal must see however the
call ends. The command layer passes in the audio acquisition step, so this
module imports nothing from ``voxweave.pipeline``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voxweave import artifacts, backend, config, episode_transaction, realign
from voxweave.align_route import (
    AlignRoute,
    _align_route_facts,
    _prepared_audio_geometry,
    _stable_fact_digest,
)
from voxweave.align_runtime import (
    align_runtime_activity,
    bind_align_runtime_identity,
)
from voxweave.core.overlay import spans_in
from voxweave.core.shadow_v2 import shadow_enabled
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
    """Selected-media identity for align when a bound vocals cache serves the audio.

    Stands where a :class:`MediaSnapshot` would: the media bytes are never read,
    so only the fingerprint the cache companion was validated against is needed.
    """

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
    align_context: Any = None
    completed_selection: Any = None
    observation_input: Mapping[str, Any] | None = None
    prepared_audio_sha256: str | None = None
    aligned_cue_count: int = 0
    aligned_unit_count: int = 0


@dataclass(frozen=True)
class FreshAlignBinding:
    # What the align route needs once the fresh acquisition is authorized.
    wav: Path
    legacy_policy: Any
    strict_shot_changes: Any
    strict_sing_spans: Any
    fresh_session: Any
    capture_raw_call: Callable[..., Any]
    invoke_backend_call: Callable[..., Any]
    invoke_qwen_call: Callable[..., Any]


def invoke_physical_preparation(operation: Callable[[], Any]) -> Any:
    with align_runtime_activity("AO-06", "physical-call-preparation"):
        return operation()


def invoke_legacy_distribution(operation: Callable[[], Any]) -> Any:
    with align_runtime_activity("AO-08", "legacy-owner-slice"):
        return operation()


def invoke_legacy_shift(operation: Callable[[], Any]) -> Any:
    with align_runtime_activity("AO-09", "legacy-time-transform"):
        return operation()


def seal_selected_legacy_align_result(
    block_units: Sequence[Sequence[Mapping[str, Any]]],
    spans: Sequence[tuple[float, float]],
    all_units: Sequence[Mapping[str, Any]],
) -> SelectedLegacyAlignResult:
    """Seal the exact selected-legacy projection before strict AO work begins."""

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


def _align_observation_input(
    inputs: AlignInputs, align_context: Any, stored_manifest_value: object
) -> dict[str, Any]:
    # The input summary the shadow observer later receives with the selection.
    from voxweave import align_orchestration

    return {
        "context_content_digest": align_context.context_content_digest,
        "vtt_sha256": hashlib.sha256(inputs.vtt_input_bytes).hexdigest(),
        "sibling_present": inputs.json_input_bytes is not None,
        "sibling_sha256": (
            None
            if inputs.json_input_bytes is None
            else hashlib.sha256(inputs.json_input_bytes).hexdigest()
        ),
        "media_fingerprint": inputs.media_input_fingerprint,
        "media_logical_id": align_orchestration._media_logical_identity(
            inputs.media,
            explicit_media=inputs.explicit_media,
        ),
        "effective_iso": inputs.iso,
        "route": inputs.route.kind,
        "block_count": len(inputs.blocks),
        "block_content_sha256": inputs.input_snapshot.block_content_sha256,
        "profile_source": (
            "manifest-absent"
            if not isinstance(stored_manifest_value, Mapping)
            else "stored-or-default"
        ),
    }


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
    expected_vtt_sha256: str | None,
) -> FreshAlignBinding:
    # Prepare stage of align (AO-04 and AO-05): produce the 16k audio, issue the
    # align context and open the fresh acquisition that authorizes backend calls.
    data = inputs.data
    route_kind = inputs.route.kind
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
    from voxweave import align_orchestration

    with align_runtime_activity("AO-04", "prepared-audio-digest"):
        run.prepared_audio_sha256 = align_orchestration.file_sha256(wav)
    from voxweave.align_inputs import LegacyAlignPolicy

    legacy_policy = LegacyAlignPolicy(
        MIN_CUE_SEC,
        TINY_CUE_SEC,
        TINY_CUE_TARGET,
    )
    stored_manifest_value = data.get("segmentation")
    strict_shot_changes = data.get("shot_changes")
    strict_sing_spans = data.get("sing_spans")
    with align_runtime_activity("AO-05", "context-and-limit-profile-issuance"):
        run.align_context = align_orchestration.issue_public_align_context(
            target_path=inputs.vtt_path,
            sibling_path=inputs.json_path,
            media_path=inputs.media,
            prepared_audio_path=wav,
            expected_vtt=episode_transaction.FileGeneration(
                True, inputs.vtt_input_bytes
            ),
            expected_json=inputs.expected_json,
            expected_vtt_sha256=expected_vtt_sha256,
            media_fingerprint=inputs.media_input_fingerprint,
            effective_iso=inputs.iso,
            route_kind=route_kind,
            blocks=inputs.blocks,
            prepared_audio_sha256=run.prepared_audio_sha256,
            legacy_policy=legacy_policy,
            stored_language=data.get("language"),
            segmentation=stored_manifest_value,
            strict_shot_changes=strict_shot_changes,
            strict_sing_spans=strict_sing_spans,
            explicit_media=inputs.explicit_media,
            block_content_sha256=inputs.input_snapshot.block_content_sha256,
        )
    align_context = run.align_context
    bind_align_runtime_identity(
        route_kind=align_context.route_kind,
        engine_family=align_context.engine_family,
    )
    run.observation_input = _align_observation_input(
        inputs, align_context, stored_manifest_value
    )
    from voxweave.align_acquisition import (
        _fresh_alignment_backend_invoker,
        _fresh_alignment_call_observer,
        _fresh_alignment_qwen_invoker,
        begin_fresh_alignment,
    )

    prepared_sample_rate, prepared_sample_count = _prepared_audio_geometry(wav)

    model_facts = {
        "route": route_kind,
        "language": inputs.iso,
        "backend": inputs.route.backend_label(lambda: backend._use_mlx()),
        "model": inputs.route.model_label(backend.ALIGNER_MODEL),
        "sample_rate": prepared_sample_rate,
    }
    route_facts = _align_route_facts(
        route_kind, inputs.iso, inputs.blocks, inputs.crops
    )

    with align_runtime_activity("AO-05", "acquisition-authorization"):
        fresh_session = begin_fresh_alignment(
            align_context,
            alignment_texts=tuple(
                str(block.get("alignment_text", block["text"]))
                for block in inputs.blocks
            ),
            source_texts=tuple(str(block["text"]) for block in inputs.blocks),
            source_indices=tuple(int(block["source_index"]) for block in inputs.blocks),
            language=inputs.iso,
            prepared_audio_sample_count=prepared_sample_count,
            sample_rate=prepared_sample_rate,
            backend_model_config_digest=_stable_fact_digest(model_facts),
            route_input_digest=_stable_fact_digest(route_facts),
            backend_model_config_facts=model_facts,
            route_input_facts=route_facts,
        )
    return FreshAlignBinding(
        wav=wav,
        legacy_policy=legacy_policy,
        strict_shot_changes=strict_shot_changes,
        strict_sing_spans=strict_sing_spans,
        fresh_session=fresh_session,
        capture_raw_call=_fresh_alignment_call_observer(fresh_session),
        invoke_backend_call=_fresh_alignment_backend_invoker(fresh_session),
        invoke_qwen_call=_fresh_alignment_qwen_invoker(fresh_session),
    )


def build_align_selection(
    inputs: AlignInputs,
    run: AlignRun,
    binding: FreshAlignBinding,
    acquisition: Any,
    selected_legacy: SelectedLegacyAlignResult,
) -> tuple[Any, bool, list[episode_transaction.ArtifactCleanup]]:
    # Select stage of align: bind the sealed legacy result and the sealed
    # acquisition to the preserved carriers and build the verified selection.
    from voxweave import align_orchestration

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
    assert run.align_context is not None
    selection = align_orchestration.build_align_selection(
        context=run.align_context,
        acquisition=acquisition,
        blocks=inputs.blocks,
        block_units=selected_legacy.block_units,
        spans=selected_legacy.spans,
        all_units=selected_legacy.all_units,
        language=inputs.iso,
        vad_speech=keep_vad,
        shot_changes=keep_shots,
        sing_spans=keep_sing,
        speaker_turns=keep_turns,
        voiceprint_pair=voiceprint_pair if preserve_pair else None,
        manifest=keep_manifest,
        shadow_requested=shadow_enabled(),
        strict_input_status=inputs.sibling_snapshot.strict_input_status,
        legacy_policy=binding.legacy_policy,
        stored_language=data.get("language"),
        strict_shot_changes=binding.strict_shot_changes,
        strict_sing_spans=binding.strict_sing_spans,
    )
    if run.observation_input is not None:
        run.observation_input = {
            **run.observation_input,
            "profile_source": selection.profile_status.source,
        }
    cleanup: list[episode_transaction.ArtifactCleanup] = []
    if inputs.pair_declared and not preserve_pair:
        cleanup.extend(
            episode_transaction.speaker_artifact_cleanups(inputs.media, inputs.vtt_path)
        )
    return selection, preserve_pair, cleanup


def commit_align_outputs(
    inputs: AlignInputs,
    run: AlignRun,
    selection: Any,
    preserve_pair: bool,
    cleanup: list[episode_transaction.ArtifactCleanup],
    *,
    rep: Reporter,
) -> None:
    # Write stage of align (AO-22): encode the evidence first, then commit the
    # VTT, the sibling JSON and the evidence sidecar as one transaction.
    media = inputs.media
    vtt_path = inputs.vtt_path
    voiceprint_pair = inputs.voiceprint_pair
    rep.step("write outputs")
    rep.stage("write VTT + JSON")
    from voxweave.align_evidence import encode_align_evidence

    with align_runtime_activity("AO-22", "selected-evidence-preencode"):
        try:
            evidence_bytes = encode_align_evidence(selection.evidence)
        except BaseException as exc:
            _attach_canonical_failure(
                exc,
                kind="preencode-failed",
                phase="preencode",
                detail_code="evidence-encode",
            )
            raise
    evidence_path = artifacts.align_evidence_path(media, vtt_path)
    cleanup.extend(
        episode_transaction.evidence_cleanups(vtt_path, media=media, keep=evidence_path)
    )
    evidence_artifact = episode_transaction.EvidencePublication(
        evidence_path,
        evidence_bytes,
    )
    episode_transaction.commit_primary_outputs(
        command="align",
        episode_path=media,
        json_path=inputs.json_path,
        vtt_path=vtt_path,
        expected_json=inputs.expected_json,
        expected_vtt=episode_transaction.FileGeneration(True, inputs.vtt_input_bytes),
        main_json_bytes=selection.verified.main_json_bytes,
        vtt_bytes=selection.verified.vtt_bytes,
        cleanup_paths=tuple(cleanup),
        context=selection.context,
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
        evidence_artifact=evidence_artifact,
    )


def select_legacy_alignment(
    block_units: Sequence[Sequence[Mapping[str, Any]]],
    final: list[tuple[float, float] | None],
    all_units: Sequence[Mapping[str, Any]],
    media: Path,
) -> SelectedLegacyAlignResult:
    # AO-10 after the spans are grouped: the historical helper chain that turns the
    # grouped units into cue spans, then the seal of that selected result.
    with align_runtime_activity("AO-10", "common-all-empty-decision"):
        if not all_units:
            exc = RuntimeError(f"no aligned units for {media.name}")
            _attach_canonical_failure(
                exc,
                kind="no-aligned-units",
                phase="fresh-acquisition",
                detail_code="all-block-units-empty",
            )
            raise exc
    # Preserve the exact historical helper chain.  The selected result is sealed
    # before seal_fresh_alignment may begin AO-11 strict recursive capture.
    with align_runtime_activity("AO-10", "fill-insert-blocks"):
        filled = realign.fill_insert_blocks(final)
    with align_runtime_activity("AO-10", "enforce-min-duration"):
        duration_enforced = realign.enforce_min_duration(
            filled,
            min_dur=MIN_CUE_SEC,
        )
    with align_runtime_activity("AO-10", "rescue-tiny-cues"):
        rescued = realign.rescue_tiny_cues(
            duration_enforced,
            trig=TINY_CUE_SEC,
            target=TINY_CUE_TARGET,
        )
    with align_runtime_activity("AO-10", "clamp-spans"):
        spans_filled = realign.clamp_spans(rescued)
    with align_runtime_activity("AO-10", "seal-selected-legacy-result"):
        return seal_selected_legacy_align_result(
            block_units,
            spans_filled,
            all_units,
        )


def dispose_align_run(run: AlignRun, production_failure: BaseException | None) -> None:
    # Disposal stage of align (AO-24): runs on every exit. A disposal failure
    # surfaces only when the call itself succeeded, so it never masks the error.
    # Release aligner singleton VRAM (separation self-releases earlier).
    disposal_failure: BaseException | None = None
    with align_runtime_activity("AO-24", "media-snapshot-disposal"):
        try:
            run.snapshots.close()
        except BaseException as exc:
            disposal_failure = _record_disposal_failure(
                production_failure,
                disposal_failure,
                exc,
                detail_code="media-snapshot-residue",
            )
    with align_runtime_activity("AO-24", "backend-and-audio-temp-disposal"):
        # Guarded: a release that raises (e.g. torch.cuda.empty_cache after a CUDA
        # fault) must neither mask the original error nor skip the temp cleanup.
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
    with align_runtime_activity("AO-24", "selection-role-retirement"):
        if run.align_context is not None:
            from voxweave import align_orchestration

            align_orchestration.retire_align_selection(run.align_context)
    if production_failure is None and disposal_failure is not None:
        raise disposal_failure
