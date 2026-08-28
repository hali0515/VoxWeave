"""Public-align bridge into acquisition, candidates, and the locked transaction."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxweave import candidate_encoder
from voxweave.align_acquisition import (
    AuthorityTransformResult,
    StrictCaptureResult,
    capture_strict_units,
    transform_strict_units,
)
from voxweave.align_adapter import (
    AlignDelivery,
    AlignDeliveryCue,
    AlignEvaluatedResult,
    AlignProjectionInputs,
    PersistedAlignUnit,
    SourceBlockDecoration,
    issue_legacy_align_evaluated_result,
)
from voxweave.align_context import (
    IssuedAlignContext,
    consume_context_role,
    issue_align_context,
    retire_live_context_roles,
)
from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityCallInput,
    AuthorityDistributionReceipt,
    AuthoritySkippedBlockInput,
    RouteClaim,
    RouteExpectation,
    build_authority_distribution,
)
from voxweave.align_evidence import FinalAlignEvidence, bind_align_evidence
from voxweave.align_evidence_core import evaluate_ald6, project_evidence_core
from voxweave.align_failures import AUTHORITY_REASON_ORDER, CanonicalFailure
from voxweave.align_inputs import (
    EvidenceStatus,
    LegacyAlignPolicy,
    ProfileStatus,
    V2PolicyStatus,
)
from voxweave.align_snapshot import FrozenObject, StrictInputStatus, freeze_json
from voxweave.episode_transaction import FileGeneration


@dataclass(frozen=True)
class RawAlignmentCall:
    source_indices: tuple[int, ...]
    alignment_texts: tuple[str, ...]
    raw_units: tuple[Any, ...]
    physical_origin_seconds: float
    identity_transform: bool


@dataclass(frozen=True)
class AlignSelection:
    context: IssuedAlignContext
    result: AlignEvaluatedResult
    verified: candidate_encoder.VerifiedEncodedCandidate
    evidence: FinalAlignEvidence
    distribution: AuthorityDistributionReceipt
    strict_input_status: StrictInputStatus
    v2_policy_status: V2PolicyStatus
    profile_status: ProfileStatus
    evidence_status: EvidenceStatus


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
) -> IssuedAlignContext:
    """Seal stable invocation facts, then authorize the physical call lane."""
    prepared = Path(prepared_audio_path)
    stable = freeze_json(
        {
            "schema_version": 2,
            "vtt_generation": _generation_value(expected_vtt),
            "sibling_generation": _generation_value(expected_json),
            "expected_vtt_sha256": expected_vtt_sha256,
            "blocks": [
                {
                    "source_index": index,
                    "text": str(block["text"]),
                    "start": block.get("start"),
                    "end": block.get("end"),
                    "lyric": True if block.get("lyric") is True else None,
                }
                for index, block in enumerate(blocks)
            ],
            "media_fingerprint": media_fingerprint,
            "media_logical_id": Path(media_path).name,
            "prepared_audio_size": prepared.stat().st_size,
            "prepared_audio_sha256": prepared_audio_sha256,
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
    consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
    return context


def capture_raw_alignment_call(
    *,
    source_indices: Sequence[int],
    alignment_texts: Sequence[str],
    raw_units: Sequence[Any],
    physical_origin_seconds: float,
    identity_transform: bool,
) -> RawAlignmentCall:
    """Detach a physical backend result from mutable backend-owned containers."""
    return RawAlignmentCall(
        tuple(int(value) for value in source_indices),
        tuple(str(value) for value in alignment_texts),
        tuple(copy.deepcopy(list(raw_units))),
        float(physical_origin_seconds),
        bool(identity_transform),
    )


def _strict_acquisition(
    calls: Sequence[RawAlignmentCall],
) -> tuple[
    tuple[StrictCaptureResult, ...],
    tuple[AuthorityTransformResult, ...],
    tuple[AuthorityCallInput, ...],
]:
    captures: list[StrictCaptureResult] = []
    transforms: list[AuthorityTransformResult] = []
    inputs: list[AuthorityCallInput] = []
    raw_cursor = 0
    unit_cursor = 0
    for call_index, call in enumerate(calls):
        ids = tuple(
            f"r{index}"
            for index in range(unit_cursor, unit_cursor + len(call.raw_units))
        )
        unit_cursor += len(ids)
        capture = capture_strict_units(
            call.raw_units,
            call_index=call_index,
            raw_unit_ids=ids,
        )
        transform = transform_strict_units(
            capture,
            physical_origin_seconds=call.physical_origin_seconds,
            identity=call.identity_transform,
        )
        if capture.status != "valid":
            status = "capture-invalid"
            surfaces = None
            failure = capture.failure
        elif transform.status != "valid":
            status = "transform-invalid"
            surfaces = tuple(unit.surface for unit in capture.units or ())
            failure = transform.failure
        else:
            status = "valid"
            surfaces = tuple(unit.surface for unit in capture.units or ())
            failure = None
        inputs.append(
            AuthorityCallInput(
                call_index,
                call.source_indices,
                (raw_cursor, raw_cursor + len(ids)),
                ids,
                surfaces,
                status,  # type: ignore[arg-type]
                failure,
            )
        )
        raw_cursor += len(ids)
        captures.append(capture)
        transforms.append(transform)
    return tuple(captures), tuple(transforms), tuple(inputs)


def _authority_distribution(
    *,
    alignment_texts: Sequence[str],
    calls: Sequence[RawAlignmentCall],
    call_inputs: tuple[AuthorityCallInput, ...],
    iso: str,
) -> AuthorityDistributionReceipt:
    blocks = tuple(
        AuthorityBlock(index, str(text)) for index, text in enumerate(alignment_texts)
    )
    owners = {
        source_index: call_index
        for call_index, call in enumerate(calls)
        for source_index in call.source_indices
    }
    skipped: list[AuthoritySkippedBlockInput] = []
    route: list[RouteExpectation] = []
    claims: list[RouteClaim] = []
    for delivery_index, text in enumerate(alignment_texts):
        owner_index = owners.get(delivery_index)
        if owner_index is not None:
            expectation = RouteExpectation(
                delivery_index, delivery_index, "call", owner_index
            )
            claim = RouteClaim("call", owner_index, delivery_index, delivery_index)
        else:
            stripped = str(text).strip()
            skip_index = len(skipped)
            skipped.append(
                AuthoritySkippedBlockInput(
                    delivery_index,
                    delivery_index,
                    "empty-alignment-text" if not stripped else "missing-crop",
                    "empty"
                    if not text
                    else "whitespace"
                    if not stripped
                    else "nonempty",
                )
            )
            expectation = RouteExpectation(
                delivery_index, delivery_index, "skip", skip_index
            )
            claim = RouteClaim("skip", skip_index, delivery_index, delivery_index)
        route.append(expectation)
        claims.append(claim)
    return build_authority_distribution(
        blocks=blocks,
        delivery_route=tuple(route),
        calls=call_inputs,
        skipped_blocks=tuple(skipped),
        route_claims=tuple(claims),
        iso=iso,
    )


def _persisted_unit(value: Mapping[str, Any]) -> PersistedAlignUnit:
    return PersistedAlignUnit(
        str(value.get("text", value.get("word", ""))),
        float(value["start"]),
        float(value["end"]),
    )


def _first_v2_failure(
    *,
    strict_input_status: StrictInputStatus,
    captures: Sequence[StrictCaptureResult],
    transforms: Sequence[AuthorityTransformResult],
    distribution: AuthorityDistributionReceipt,
    seed_status: str,
    seed_reasons: Sequence[str],
    v2_policy_status: V2PolicyStatus,
    profile_status: ProfileStatus,
    evidence_status: EvidenceStatus,
) -> CanonicalFailure | None:
    if strict_input_status.kind == "invalid":
        if strict_input_status.detail_code is None:
            raise ValueError("invalid strict input status lacks its detail")
        return CanonicalFailure(
            "v2-input-invalid",
            "strict-input",
            strict_input_status.detail_code,
        )
    for result in (*captures, *transforms):
        locator = result.failure
        if locator is not None:
            return CanonicalFailure(
                "fresh-time-transform-invalid",
                locator.stage,
                locator.detail_code,
            )
    detail_by_reason = {
        "partial-empty-ownership": "partial-empty-ownership",
        "punctuation-only-block": "punctuation-only-block",
        "route-owner-mismatch": "route-owner-mismatch",
        "allocation-no-tiling": "allocation-no-tiling",
        "allocation-ambiguous": "allocation-ambiguous",
        "allocation-budget-exhausted": "allocation-budget",
    }
    if distribution.status == "invalid":
        for reason in AUTHORITY_REASON_ORDER:
            detail = detail_by_reason.get(reason)
            if reason in distribution.reasons and detail is not None:
                return CanonicalFailure(
                    "fresh-distribution-invalid",
                    "authority-distribution",
                    detail,
                )
        raise ValueError("invalid authority distribution lacks a canonical failure")
    if "footprint-reconciliation" in seed_reasons:
        return CanonicalFailure(
            "fresh-reconciliation-invalid",
            "seed-reconciliation",
            "footprint-reconciliation",
        )
    if any(
        reason in seed_reasons
        for reason in (
            "absolute-bound-invalid",
            "absolute-order-invalid",
            "display-seed-invalid",
        )
    ):
        return CanonicalFailure(
            "fresh-seed-invalid",
            "seed-admission",
            "seed-admission",
        )
    if seed_status != "valid":
        raise ValueError("invalid align seed lacks a canonical failure")
    if v2_policy_status.kind == "invalid":
        if v2_policy_status.detail_code is None:
            raise ValueError("invalid v2 policy status lacks its detail")
        return CanonicalFailure(
            "v2-policy-invalid",
            "v2-policy",
            v2_policy_status.detail_code,
        )
    if profile_status.kind == "invalid":
        if profile_status.detail_code is None:
            raise ValueError("invalid profile status lacks its detail")
        return CanonicalFailure(
            "profile-invalid",
            "display-profile",
            profile_status.detail_code,
        )
    if evidence_status.kind == "invalid":
        if evidence_status.detail_code is None:
            raise ValueError("invalid evidence status lacks its detail")
        return CanonicalFailure(
            "evidence-invalid",
            "finalizer-evidence",
            evidence_status.detail_code,
        )
    return None


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
    blocks: Sequence[Mapping[str, Any]],
    alignment_texts: Sequence[str],
    block_units: Sequence[Sequence[Mapping[str, Any]]],
    spans: Sequence[tuple[float, float]],
    all_units: Sequence[Mapping[str, Any]],
    raw_calls: Sequence[RawAlignmentCall],
    language: str,
    vad_speech: Sequence[tuple[float, float]] | None,
    shot_changes: Sequence[float] | None,
    sing_spans: Sequence[tuple[float, float]] | None,
    speaker_turns: Sequence[tuple[float, float, str]] | None,
    voiceprint_pair: tuple[str, str] | None,
    manifest: Mapping[str, Any] | None,
    shadow_requested: bool,
    strict_input_status: StrictInputStatus,
    legacy_policy: LegacyAlignPolicy,
    stored_language: object,
    strict_shot_changes: object,
    strict_sing_spans: object,
) -> AlignSelection:
    """Project mandatory evidence, encode once, select, and independently verify."""
    captures, transforms, call_inputs = _strict_acquisition(raw_calls)
    distribution = _authority_distribution(
        alignment_texts=alignment_texts,
        calls=raw_calls,
        call_inputs=call_inputs,
        iso=language,
    )
    authority_blocks = tuple(
        AuthorityBlock(index, str(text)) for index, text in enumerate(alignment_texts)
    )
    from voxweave.core import align_seed

    fresh_units = tuple(
        unit
        for transform in transforms
        for unit in (transform.units if transform.units is not None else ())
    )
    seed = align_seed.build_align_seed(
        blocks=authority_blocks,
        units=fresh_units,
        distribution=distribution,
        iso=language,
    )
    from voxweave import align_inputs

    v2_policy_status = align_inputs.validate_v2_policy(legacy_policy)
    profile = align_inputs.resolve_align_profile(
        manifest,
        effective_iso=language,
        stored_iso=stored_language if isinstance(stored_language, str) else None,
    )
    evidence_resolution = align_inputs.resolve_finalize_evidence(
        shot_changes=strict_shot_changes,  # type: ignore[arg-type]
        sing_spans=strict_sing_spans,  # type: ignore[arg-type]
    )
    evidence_core = project_evidence_core(
        context_content_digest=context.context_content_digest,
        blocks=authority_blocks,
        captures=captures,
        transforms=transforms,
        distribution=distribution,
        seed_status=seed.status,
        seed_reasons=seed.reasons,
    )
    if not evaluate_ald6(evidence_core, evidence_core).passed:
        raise RuntimeError("independent EvidenceCore projection disagreed")

    owner_ids: dict[int, tuple[str, ...]] = {}
    if distribution.owners is not None:
        owner_ids = dict(
            zip(distribution.owner_source_indices, distribution.owners, strict=True)
        )
    anchors = {block.source_index: block for block in evidence_core.blocks}
    delivery_cues = tuple(
        AlignDeliveryCue(
            index,
            str(block["text"]),
            float(span[0]),
            float(span[1]),
            True if block.get("lyric") is True else None,
            owner_ids.get(index, ()),
            tuple(_persisted_unit(unit) for unit in block_units[index]),
            anchors[index].speech_start if index in anchors else None,
            anchors[index].speech_end if index in anchors else None,
        )
        for index, (block, span) in enumerate(zip(blocks, spans, strict=True))
    )
    delivery = AlignDelivery(
        context.context_content_digest,
        evidence_core.core_digest,
        "legacy-v1",
        context.route_kind,
        delivery_cues,
        tuple(_persisted_unit(unit) for unit in all_units),
    )
    frozen_manifest = None if manifest is None else freeze_json(dict(manifest))
    if frozen_manifest is not None and not isinstance(frozen_manifest, FrozenObject):
        raise TypeError("align segmentation manifest is not an object")
    projection_inputs = AlignProjectionInputs(
        language,
        tuple(_source_decoration(index, block) for index, block in enumerate(blocks)),
        None if vad_speech is None else tuple(vad_speech),
        None if shot_changes is None else tuple(float(value) for value in shot_changes),
        None if sing_spans is None else tuple(sing_spans),
        None if speaker_turns is None else tuple(speaker_turns),
        None if voiceprint_pair is None else voiceprint_pair[0],
        None if voiceprint_pair is None else voiceprint_pair[1],
        frozen_manifest,
    )
    v2_failure = _first_v2_failure(
        strict_input_status=strict_input_status,
        captures=captures,
        transforms=transforms,
        distribution=distribution,
        seed_status=seed.status,
        seed_reasons=seed.reasons,
        v2_policy_status=v2_policy_status,
        profile_status=profile.status,
        evidence_status=evidence_resolution.status,
    )
    result = issue_legacy_align_evaluated_result(
        context,
        delivery=delivery,
        projection_inputs=projection_inputs,
        evidence_core=evidence_core,
        shadow_requested=shadow_requested,
        v2_failure=v2_failure,
    )
    candidates = candidate_encoder.encode_align_candidates(context, result)
    selected = candidate_encoder.select_align_candidate(context, candidates)
    verified = candidate_encoder.verify_selected_align_projection(
        context, result, selected
    )
    bound_evidence = bind_align_evidence(
        context,
        evidence_core,
        engine_family=verified.engine_family,
        vtt_sha256=verified.vtt_sha256,
        main_json_sha256=verified.main_json_sha256,
    )
    return AlignSelection(
        context,
        result,
        verified,
        bound_evidence,
        distribution,
        strict_input_status,
        v2_policy_status,
        profile.status,
        evidence_resolution.status,
    )


def retire_align_selection(selection: AlignSelection | IssuedAlignContext) -> None:
    context = (
        selection if isinstance(selection, IssuedAlignContext) else selection.context
    )
    retire_live_context_roles(context)


__all__ = [
    "AlignSelection",
    "RawAlignmentCall",
    "build_align_selection",
    "capture_raw_alignment_call",
    "file_sha256",
    "issue_public_align_context",
    "retire_align_selection",
]
