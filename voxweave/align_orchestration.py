"""Public-align bridge into acquisition, candidates, and the locked transaction."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxweave import candidate_encoder
from voxweave.align_acquisition import (
    IssuedFreshAlignment,
    _bind_fresh_adapter_payload,
    _fresh_producer_core_inputs,
    _fresh_reference_core_inputs,
    _fresh_seed,
)
from voxweave.align_adapter import (
    AlignDelivery,
    AlignDeliveryCue,
    AlignEvaluatedResult,
    AlignProjectionInputs,
    PersistedAlignUnit,
    SourceBlockDecoration,
    _adapter_semantic_observation,
    issue_align_evaluated_result,
    run_locked_align_adapter,
)
from voxweave.align_context import (
    IssuedAlignContext,
    issue_align_context,
    retire_live_context_roles,
    verify_context_expected_vtt_generation,
    verify_context_roles_terminal,
)
from voxweave.align_distribution import AuthorityDistributionReceipt
from voxweave.align_evidence import FinalAlignEvidence, bind_align_evidence
from voxweave.align_evidence_core import (
    build_evidence_core,
    evaluate_ald6,
    project_evidence_core,
)
from voxweave.align_failures import CanonicalFailure
from voxweave.align_inputs import (
    EvidenceStatus,
    LegacyAlignPolicy,
    ProfileStatus,
    V2PolicyStatus,
)
from voxweave.align_runtime import align_runtime_activity
from voxweave.align_snapshot import (
    FrozenObject,
    RawJSONCarrier,
    StrictInputStatus,
    freeze_json,
    frozen_json_digest,
)
from voxweave.episode_transaction import FileGeneration


ALIGN_AO_PHASE_ORDER = (
    "AO-01",
    "AO-02",
    "AO-03",
    "AO-04",
    "AO-05",
    "AO-06",
    "AO-07",
    "AO-08",
    "AO-09",
    "AO-10",
    "AO-11",
    "AO-12",
    "AO-13",
    "AO-14",
    "AO-15",
    "AO-16",
    "AO-17",
    "AO-18",
    "AO-19",
    "AO-20",
    "AO-21",
    "AO-22",
    "AO-23",
    "AO-24",
    "AO-25",
)


@dataclass(frozen=True)
class AlignSelection:
    context: IssuedAlignContext
    result: AlignEvaluatedResult
    verified: candidate_encoder.VerifiedEncodedCandidate
    legacy_vtt_sha256: str
    legacy_main_json_sha256: str
    evidence: FinalAlignEvidence
    distribution: AuthorityDistributionReceipt
    strict_input_status: StrictInputStatus
    v2_policy_status: V2PolicyStatus
    profile_status: ProfileStatus
    evidence_status: EvidenceStatus
    observation_failure: CanonicalFailure | None


@dataclass(frozen=True)
class _FailedProfileResolution:
    profile: None
    status: ProfileStatus
    unexpected_failure: CanonicalFailure


def _classify_unchanged_exception(
    exc: BaseException,
    failure: CanonicalFailure,
) -> None:
    try:
        if not isinstance(getattr(exc, "failure", None), CanonicalFailure):
            setattr(exc, "failure", failure)
    except Exception:
        pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_value(generation: FileGeneration) -> dict[str, object]:
    return {
        "present": generation.present,
        "size": None if generation.bytes_value is None else len(generation.bytes_value),
        "sha256": generation.sha256,
    }


def _original_source_indices(
    blocks: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    sources: list[int] = []
    for block in blocks:
        source = block.get("source_index")
        if type(source) is not int or source < 0:
            raise ValueError("align block lacks an original source index")
        sources.append(source)
    if len(set(sources)) != len(sources):
        raise ValueError("align block source indices are duplicated")
    return tuple(sources)


def _media_logical_identity(media: Path, *, explicit_media: bool) -> str:
    name = unicodedata.normalize("NFC", media.name)
    suffix = unicodedata.normalize("NFC", media.suffix.lower())
    if (explicit_media and not name) or (not explicit_media and not suffix):
        exc = ValueError("media logical identity is unavailable")
        setattr(
            exc,
            "failure",
            CanonicalFailure(
                "media-identity-invalid",
                "media",
                "media-logical-id",
            ),
        )
        raise exc
    return f"explicit:{name}" if explicit_media else f"sibling:{suffix}"


def issue_public_align_context(
    *,
    target_path: Path,
    sibling_path: Path,
    media_path: Path,
    prepared_audio_path: Path,
    expected_vtt: FileGeneration,
    expected_json: FileGeneration,
    expected_vtt_sha256: str | None,
    media_fingerprint: str,
    effective_iso: str,
    route_kind: str,
    blocks: Sequence[Mapping[str, Any]],
    prepared_audio_sha256: str,
    legacy_policy: LegacyAlignPolicy,
    stored_language: object,
    segmentation: object,
    strict_shot_changes: object,
    strict_sing_spans: object,
    explicit_media: bool = False,
    block_content_sha256: str | None = None,
) -> IssuedAlignContext:
    """Seal stable invocation facts, then authorize the physical call lane."""
    if type(explicit_media) is not bool:
        raise TypeError("explicit_media must be an exact bool")
    prepared = Path(prepared_audio_path)
    media = Path(media_path)
    source_indices = _original_source_indices(blocks)
    normalized_target_name = unicodedata.normalize("NFC", Path(target_path).name)
    media_logical_id = _media_logical_identity(media, explicit_media=explicit_media)
    stable_blocks = [
        {
            "source_index": source_index,
            "text": str(block["text"]),
            "alignment_text": str(block.get("alignment_text", block["text"])),
            "start": block.get("start"),
            "end": block.get("end"),
            "lyric": block.get("lyric") is True,
            "speaker": block.get("speaker"),
            "speakers": (
                [
                    list(row) if isinstance(row, tuple) else row
                    for row in block["speakers"]
                ]
                if isinstance(block.get("speakers"), (list, tuple))
                else block.get("speakers")
            ),
        }
        for source_index, block in zip(source_indices, blocks, strict=True)
    ]
    derived_block_digest = frozen_json_digest(freeze_json(stable_blocks))
    sealed_block_digest = (
        derived_block_digest if block_content_sha256 is None else block_content_sha256
    )
    if (
        type(sealed_block_digest) is not str
        or len(sealed_block_digest) != 64
        or any(c not in "0123456789abcdef" for c in sealed_block_digest)
    ):
        raise ValueError("block_content_sha256 must be lowercase SHA-256")
    profile_input_sha256 = frozen_json_digest(
        freeze_json(
            {
                "stored_language": stored_language,
                "segmentation": segmentation,
            }
        )
    )
    evidence_carriers_sha256 = frozen_json_digest(
        freeze_json(
            {
                "shot_changes": strict_shot_changes,
                "sing_spans": strict_sing_spans,
            }
        )
    )
    stable = freeze_json(
        {
            "schema_version": 2,
            "vtt_generation": _generation_value(expected_vtt),
            "sibling_generation": _generation_value(expected_json),
            "expected_vtt_sha256": expected_vtt_sha256,
            "block_content_sha256": sealed_block_digest,
            "blocks": stable_blocks,
            "media_fingerprint": media_fingerprint,
            "media_logical_id": media_logical_id,
            "media_display_name": media.name,
            "target_logical_id": normalized_target_name,
            "prepared_audio_size": prepared.stat().st_size,
            "prepared_audio_sha256": prepared_audio_sha256,
            "profile_input_sha256": profile_input_sha256,
            "evidence_carriers_sha256": evidence_carriers_sha256,
            "adapter_inputs": {
                "legacy_policy": {
                    "min_cue_sec": legacy_policy.min_cue_sec,
                    "tiny_cue_sec": legacy_policy.tiny_cue_sec,
                    "tiny_cue_target": legacy_policy.tiny_cue_target,
                },
                "stored_language": stored_language,
                "segmentation": segmentation,
                "shot_changes": strict_shot_changes,
                "sing_spans": strict_sing_spans,
            },
        }
    )
    if not isinstance(stable, FrozenObject):
        raise TypeError("align context stable projection is not an object")
    context = issue_align_context(
        stable_fields=stable,
        target_path=target_path,
        sibling_path=sibling_path,
        media_path=media_path,
        effective_iso=effective_iso,
        route_kind=route_kind,  # type: ignore[arg-type]
    )
    verify_context_expected_vtt_generation(
        context,
        observed_vtt_sha256=expected_vtt.sha256,
    )
    return context


def _persisted_unit(value: Mapping[str, Any]) -> PersistedAlignUnit:
    return PersistedAlignUnit(
        str(value.get("text", value.get("word", ""))),
        float(value["start"]),
        float(value["end"]),
    )


def _source_decoration(index: int, block: Mapping[str, Any]) -> SourceBlockDecoration:
    speaker_value = block.get("speaker")
    speakers_value = block.get("speakers")
    speakers = None
    if isinstance(speakers_value, (list, tuple)):
        speakers = tuple(
            (
                None if row[0] is None else str(row[0]),
                str(row[1]),
            )
            for row in speakers_value
            if isinstance(row, (list, tuple)) and len(row) == 2
        )
    return SourceBlockDecoration(
        index,
        speaker_value if isinstance(speaker_value, str) else None,
        speakers,
    )


def build_align_selection(
    *,
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
    blocks: Sequence[Mapping[str, Any]],
    block_units: Sequence[Sequence[Mapping[str, Any]]],
    spans: Sequence[tuple[float, float]],
    all_units: Sequence[Mapping[str, Any]],
    language: str,
    vad_speech: Sequence[tuple[float, float]] | None,
    shot_changes: Sequence[float] | None,
    sing_spans: Sequence[tuple[float, float]] | None,
    speaker_turns: object | None,
    voiceprint_pair: tuple[str, str] | None,
    manifest: Mapping[str, Any] | None,
    shadow_requested: bool,
    strict_input_status: StrictInputStatus,
    legacy_policy: LegacyAlignPolicy,
    stored_language: object,
    strict_shot_changes: object,
    strict_sing_spans: object,
) -> AlignSelection:
    """Run the sole AO-14 through AO-21 align selection schedule."""
    if not isinstance(acquisition, IssuedFreshAlignment):
        raise TypeError("align selection requires one issued fresh acquisition")
    from voxweave import align_inputs

    source_indices = _original_source_indices(blocks)
    v2_policy_status = align_inputs.validate_v2_policy(legacy_policy)
    try:
        profile = align_inputs.resolve_align_profile(
            manifest,
            effective_iso=language,
            stored_iso=stored_language if isinstance(stored_language, str) else None,
        )
    except Exception as exc:
        profile_failure = CanonicalFailure(
            "shadow-internal-error",
            "profile-stage",
            "profile-stage",
        )
        if context.engine_family == "boundary-v2":
            _classify_unchanged_exception(exc, profile_failure)
            raise
        profile = _FailedProfileResolution(
            None,
            ProfileStatus(
                "invalid",
                "manifest-absent" if manifest is None else "stored-profile",
                "profile-shape",
            ),
            profile_failure,
        )
    evidence_resolution = align_inputs.resolve_finalize_evidence(
        shot_changes=strict_shot_changes,  # type: ignore[arg-type]
        sing_spans=strict_sing_spans,  # type: ignore[arg-type]
    )
    seed = _fresh_seed(context, acquisition)
    seed_blocks = getattr(seed, "blocks", None)
    anchors = (
        {}
        if seed_blocks is None
        else {block.source_index: block for block in seed_blocks}
    )
    delivery_cues = tuple(
        AlignDeliveryCue(
            source_index,
            str(block["text"]),
            float(span[0]),
            float(span[1]),
            True if block.get("lyric") is True else None,
            anchors[source_index].owner_unit_ids if source_index in anchors else (),
            tuple(_persisted_unit(unit) for unit in block_units[delivery_index]),
            anchors[source_index].speech_start if source_index in anchors else None,
            anchors[source_index].speech_end if source_index in anchors else None,
        )
        for delivery_index, (source_index, block, span) in enumerate(
            zip(source_indices, blocks, spans, strict=True)
        )
    )
    delivery = AlignDelivery(
        context.context_content_digest,
        acquisition.receipt_digest,
        "legacy-v1",
        context.route_kind,
        delivery_cues,
        tuple(_persisted_unit(unit) for unit in all_units),
    )
    frozen_manifest = None if manifest is None else freeze_json(dict(manifest))
    if frozen_manifest is not None and not isinstance(frozen_manifest, FrozenObject):
        raise TypeError("align segmentation manifest is not an object")
    if isinstance(speaker_turns, RawJSONCarrier) or speaker_turns is None:
        sealed_turns = speaker_turns
    else:
        sealed_turns = tuple(speaker_turns)  # type: ignore[arg-type]
    projection_inputs = AlignProjectionInputs(
        language,
        tuple(
            _source_decoration(source_index, block)
            for source_index, block in zip(source_indices, blocks, strict=True)
        ),
        None if vad_speech is None else tuple(vad_speech),
        None if shot_changes is None else tuple(float(value) for value in shot_changes),
        None if sing_spans is None else tuple(sing_spans),
        sealed_turns,  # type: ignore[arg-type]
        None if voiceprint_pair is None else voiceprint_pair[0],
        None if voiceprint_pair is None else voiceprint_pair[1],
        frozen_manifest,
    )
    with align_runtime_activity("AO-14", "adapter-binding-and-admission"):
        _bind_fresh_adapter_payload(
            context,
            acquisition,
            legacy_delivery=delivery,
            projection_inputs=projection_inputs,
            strict_input_status=strict_input_status,
            v2_policy_status=v2_policy_status,
            profile_resolution=profile,
            evidence_resolution=evidence_resolution,
        )
    adapter_result = run_locked_align_adapter(
        context,
        acquisition,
        shadow_enabled=shadow_requested,
    )
    semantic_observation = _adapter_semantic_observation(context, adapter_result)

    with align_runtime_activity("AO-16", "mandatory-evidence-core-and-ald6"):
        producer_core = build_evidence_core(
            _fresh_producer_core_inputs(
                context,
                acquisition,
                strict_input_status=strict_input_status,
                v2_policy_status=v2_policy_status,
                profile_status=profile.status,
                evidence_status=evidence_resolution.status,
            )
        )
        reference_core = project_evidence_core(
            _fresh_reference_core_inputs(context, acquisition)
        )
        if not evaluate_ald6(producer_core, reference_core).passed:
            raise RuntimeError("independent EvidenceCore projection disagreed")

    comparison = None
    if adapter_result.v2_status.kind == "valid":
        if adapter_result.v2 is None:
            raise RuntimeError("valid adapter result lacks its v2 delivery")
        from voxweave.core.align_compare import compare_semantic_deltas

        if semantic_observation is None:
            raise RuntimeError("valid adapter result lacks its AO-15 observation")
        if profile.profile is None:
            raise RuntimeError("valid adapter result lacks its display profile")

        with align_runtime_activity("AO-17", "semantic-comparison"):
            try:
                comparison = compare_semantic_deltas(
                    route_kind=context.route_kind,
                    physical_calls=producer_core.physical_calls,
                    authority_blocks=producer_core.blocks,
                    legacy=adapter_result.legacy,
                    v2=adapter_result.v2,
                    semantic_observation=semantic_observation,
                    profile=profile.profile,
                    evidence=evidence_resolution,
                )
            except Exception as exc:
                comparator_failure = CanonicalFailure(
                    "shadow-internal-error",
                    "comparator-stage",
                    "comparator-stage",
                )
                if context.engine_family == "boundary-v2":
                    _classify_unchanged_exception(exc, comparator_failure)
                    raise
                comparison = comparator_failure
    with align_runtime_activity("AO-18", "selected-v2-admissibility"):
        result = issue_align_evaluated_result(
            context,
            adapter_result,
            evidence_core=producer_core,
            comparison=comparison,
        )
    with align_runtime_activity("AO-19", "composite-candidate-encode"):
        candidates = candidate_encoder.encode_align_candidates(context, result)
    with align_runtime_activity("AO-20", "selector-and-independent-projection"):
        legacy_candidate = candidates.outcome_for("legacy-v1")
        if not isinstance(legacy_candidate, candidate_encoder.EncodedCandidate):
            raise RuntimeError("legacy alignment candidate is unavailable")
        selected = candidate_encoder.select_align_candidate(context, candidates)
        verified = candidate_encoder.verify_selected_align_projection(
            context, result, selected
        )
    boundary_candidate = candidates.outcome_for("boundary-v2")
    observation_failure = (
        boundary_candidate.failure
        if (
            context.engine_family == "legacy-v1"
            and result.v2_status.kind == "valid"
            and isinstance(boundary_candidate, candidate_encoder.CandidateFailure)
        )
        else None
    )
    with align_runtime_activity("AO-21", "selected-hashes-and-evidence-bind"):
        bound_evidence = bind_align_evidence(
            context,
            producer_core,
            acquisition=acquisition,
            strict_input_status=strict_input_status,
            v2_policy_status=v2_policy_status,
            profile_status=profile.status,
            evidence_status=evidence_resolution.status,
            engine_family=verified.engine_family,
            vtt_sha256=verified.vtt_sha256,
            main_json_sha256=verified.main_json_sha256,
        )
    return AlignSelection(
        context,
        result,
        verified,
        legacy_candidate.vtt_sha256,
        legacy_candidate.main_json_sha256,
        bound_evidence,
        acquisition.distribution,
        strict_input_status,
        v2_policy_status,
        profile.status,
        evidence_resolution.status,
        observation_failure,
    )


def retire_align_selection(selection: AlignSelection | IssuedAlignContext) -> None:
    context = (
        selection if isinstance(selection, IssuedAlignContext) else selection.context
    )
    retire_live_context_roles(context)
    verify_context_roles_terminal(context)


__all__ = [
    "ALIGN_AO_PHASE_ORDER",
    "AlignSelection",
    "build_align_selection",
    "file_sha256",
    "issue_public_align_context",
    "retire_align_selection",
]
