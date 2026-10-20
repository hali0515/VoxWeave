"""Closed RAT-2 evidence binding and independent durable verification.

The binder receives only already-issued context/acquisition values and the
producer EvidenceCore, which AO-16 has already checked against the independent
reference projection. The path verifier deliberately does not consult any
in-memory issuer registry.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from voxweave import artifacts
from voxweave.align_acquisition import IssuedFreshAlignment
from voxweave.align_context import (
    IssuedAlignContext,
    consume_context_role,
)
from voxweave.align_evidence_core import (
    EvidenceCore,
    evidence_core_value,
)
from voxweave.align_evidence_schema import (
    _TOP_LEVEL_KEYS as _TOP_LEVEL_KEYS,
    _SOURCE_FACT_KEYS as _SOURCE_FACT_KEYS,
    _MODEL_FACT_KEYS as _MODEL_FACT_KEYS,
    _DEFAULT_MODEL_FACT_KEYS as _DEFAULT_MODEL_FACT_KEYS,
    _ROUTE_FACT_KEYS as _ROUTE_FACT_KEYS,
    _DEFAULT_ROUTE_FACT_KEYS as _DEFAULT_ROUTE_FACT_KEYS,
    _ROUTE_FACT_BLOCK_KEYS as _ROUTE_FACT_BLOCK_KEYS,
    _INPUT_HISTORY_KEYS as _INPUT_HISTORY_KEYS,
    _POLICY_KEYS as _POLICY_KEYS,
    _ROUTE_PLAN_KEYS as _ROUTE_PLAN_KEYS,
    _ROUTE_ENTRY_KEYS as _ROUTE_ENTRY_KEYS,
    _PHYSICAL_CALL_KEYS as _PHYSICAL_CALL_KEYS,
    _STRICT_FAILURE_KEYS as _STRICT_FAILURE_KEYS,
    _LEGACY_DISTRIBUTION_KEYS as _LEGACY_DISTRIBUTION_KEYS,
    _LEGACY_CALL_KEYS as _LEGACY_CALL_KEYS,
    _AUTHORITY_DISTRIBUTION_KEYS as _AUTHORITY_DISTRIBUTION_KEYS,
    _WORK_KEYS as _WORK_KEYS,
    _ROUTE_MISMATCH_KEYS as _ROUTE_MISMATCH_KEYS,
    _ROUTE_CLAIM_KEYS as _ROUTE_CLAIM_KEYS,
    _JOB_LIMIT_KEYS as _JOB_LIMIT_KEYS,
    _CALL_LIMIT_KEYS as _CALL_LIMIT_KEYS,
    _COUNTER_KEYS as _COUNTER_KEYS,
    _DENIED_CHARGE_KEYS as _DENIED_CHARGE_KEYS,
    _DENIED_COUNTER_KEYS as _DENIED_COUNTER_KEYS,
    _WORK_CALL_KEYS as _WORK_CALL_KEYS,
    _LANE_KEYS as _LANE_KEYS,
    _SKIP_KEYS as _SKIP_KEYS,
    _BLOCK_KEYS as _BLOCK_KEYS,
    _WORD_KEYS as _WORD_KEYS,
    _STRICT_STATUS_KEYS as _STRICT_STATUS_KEYS,
    _SEED_STATUS_KEYS as _SEED_STATUS_KEYS,
    _PROFILE_STATUS_KEYS as _PROFILE_STATUS_KEYS,
    _SELECTED_OUTPUT_KEYS as _SELECTED_OUTPUT_KEYS,
    _WORK_STATUSES as _WORK_STATUSES,
    _LANE_STATUSES as _LANE_STATUSES,
    _LANE_DETAILS as _LANE_DETAILS,
    _PROFILE_SOURCES as _PROFILE_SOURCES,
    _is_sha256 as _is_sha256,
    _same_float as _same_float,
    _same_optional_float as _same_optional_float,
    _digest_projection as _digest_projection,
    _DuplicateEvidenceKey as _DuplicateEvidenceKey,
    _EvidenceInvalid as _EvidenceInvalid,
    _invalid as _invalid,
    _object_without_duplicates as _object_without_duplicates,
    _reject_nonfinite as _reject_nonfinite,
    _closed_mapping as _closed_mapping,
    _list as _list,
    _exact_int as _exact_int,
    _nullable_int as _nullable_int,
    _finite_float as _finite_float,
    _nullable_float as _nullable_float,
    _sha_or_none as _sha_or_none,
    _int_array as _int_array,
    _string_array as _string_array,
    _pair as _pair,
    _validate_generation as _validate_generation,
    _validate_history as _validate_history,
    _durable_fact_digest as _durable_fact_digest,
    _fact_hex_float as _fact_hex_float,
    _validate_source_facts as _validate_source_facts,
    _validate_route_plan as _validate_route_plan,
    _validate_strict_failure as _validate_strict_failure,
    _validate_physical_calls as _validate_physical_calls,
    _crosslink_source_facts as _crosslink_source_facts,
    _validate_legacy as _validate_legacy,
    _validate_limits as _validate_limits,
    _project_route_mismatch as _project_route_mismatch,
    _validate_lane as _validate_lane,
    _validate_denied_charge as _validate_denied_charge,
    _work_reason_projection as _work_reason_projection,
)
from voxweave.align_evidence_validation import (
    _validate_work as _validate_work,
    _validate_authority as _validate_authority,
    _validate_blocks as _validate_blocks,
    _validate_statuses as _validate_statuses,
    _lane_receipt_value as _lane_receipt_value,
    _work_receipt_value as _work_receipt_value,
    _receipt_digest as _receipt_digest,
    _validate_evidence_value as _validate_evidence_value,
)
from voxweave.align_failures import (
    CanonicalFailure,
)
from voxweave.align_inputs import EvidenceStatus, ProfileStatus, V2PolicyStatus
from voxweave.align_snapshot import (
    StrictInputStatus,
)
from voxweave.candidate_encoder import _verified_hash_binding
from voxweave.engine_registry import EngineFamily
from voxweave.paths import find_sibling_media, find_subtitle_media, swap_ext
from voxweave.voicebase import media_fingerprint


@dataclass(frozen=True)
class SelectedOutputs:
    engine_family: EngineFamily
    vtt_present: Literal[True]
    vtt_sha256: str
    json_present: Literal[True]
    json_sha256: str


@dataclass(frozen=True, init=False)
class FinalAlignEvidence:
    core: EvidenceCore
    selected_outputs: SelectedOutputs
    durable_authority: Literal[True]
    _projection: Mapping[str, Any] = field(repr=False, compare=True)

    def __init__(self) -> None:
        raise TypeError("FinalAlignEvidence is binder-only")


@dataclass(frozen=True)
class AlignEvidenceVerification:
    integrity: bool
    w1_usable: bool
    detail_code: str | None


@dataclass(frozen=True)
class _EvidenceRecord:
    evidence: FinalAlignEvidence
    context: IssuedAlignContext
    snapshot: FinalAlignEvidence


_EVIDENCE: dict[int, _EvidenceRecord] = {}
_LOCK = threading.RLock()


class EvidenceBindingError(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(
            f"{failure.kind}/{failure.phase}/{failure.detail_code}: align stopped"
            " because its alignment evidence failed an internal consistency check;"
            " no subtitle file was written. This is a VoxWeave bug, not a problem"
            " with your files; please report it with this message"
        )
        self.failure = failure


def _binding_failure(detail: str = "evidence-binding") -> EvidenceBindingError:
    return EvidenceBindingError(
        CanonicalFailure("final-evidence-invalid", "evidence-bind", detail)
    )


def _release_evidence(context: IssuedAlignContext) -> None:
    """Forget the evidence bound for ``context``.

    See :func:`voxweave.align_orchestration.release_align_selection`.
    """
    with _LOCK:
        for key in [key for key, row in _EVIDENCE.items() if row.context is context]:
            del _EVIDENCE[key]


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            member.name: _json_value(getattr(value, member.name))
            for member in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_json_value(member) for member in value]
    if isinstance(value, list):
        return [_json_value(member) for member in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(member) for key, member in value.items()}
    return value


def _core_value(
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
    evidence_core: EvidenceCore,
    *,
    strict_input_status: StrictInputStatus,
    v2_policy_status: V2PolicyStatus,
    profile_status: ProfileStatus,
    evidence_status: EvidenceStatus,
) -> dict[str, Any]:
    projection = evidence_core_value(evidence_core)
    strict_value = {
        "kind": strict_input_status.kind,
        "detail_code": strict_input_status.detail_code,
    }
    policy_value = {
        "kind": v2_policy_status.kind,
        "detail_code": v2_policy_status.detail_code,
    }
    profile_value = {
        "kind": profile_status.kind,
        "source": profile_status.source,
        "detail_code": profile_status.detail_code,
    }
    evidence_value = {
        "kind": evidence_status.kind,
        "detail_code": evidence_status.detail_code,
    }
    if (
        projection.get("schema_version") != 8
        or projection.get("kind") != "fresh-alignment"
        or projection.get("context_content_digest") != context.context_content_digest
        or projection.get("receipt_digest") != acquisition.receipt_digest
        or projection.get("language") != context.effective_iso
        or projection.get("route") != context.route_kind
        or projection.get("raw_unit_count") != evidence_core.raw_unit_count
        or projection.get("strict_input_status") != strict_value
        or projection.get("v2_policy_status") != policy_value
        or projection.get("profile_status") != profile_value
        or projection.get("evidence_status") != evidence_value
        or projection.get("seed_status")
        != {
            "kind": evidence_core.seed_status,
            "reasons": list(evidence_core.seed_reasons),
        }
        or evidence_core.receipt_digest != acquisition.receipt_digest
        or len(evidence_core.physical_calls) != len(acquisition.physical_calls)
        or evidence_core.authority_status != acquisition.distribution.status
        or evidence_core.authority_work != acquisition.distribution.work
        or evidence_core.seed_status != acquisition.seed_status
        or evidence_core.seed_reasons != acquisition.seed_reasons
    ):
        raise _binding_failure("independent-projection")
    return projection


def bind_align_evidence(
    context: IssuedAlignContext,
    evidence_core: EvidenceCore,
    *,
    acquisition: IssuedFreshAlignment,
    strict_input_status: StrictInputStatus,
    v2_policy_status: V2PolicyStatus,
    profile_status: ProfileStatus,
    evidence_status: EvidenceStatus,
    engine_family: EngineFamily,
    vtt_sha256: str,
    main_json_sha256: str,
) -> FinalAlignEvidence:
    """Bind a complete closed core only after independent primary verification."""
    if (
        not isinstance(context, IssuedAlignContext)
        or not isinstance(acquisition, IssuedFreshAlignment)
        or not isinstance(evidence_core, EvidenceCore)
        or not isinstance(strict_input_status, StrictInputStatus)
        or not isinstance(v2_policy_status, V2PolicyStatus)
        or not isinstance(profile_status, ProfileStatus)
        or not isinstance(evidence_status, EvidenceStatus)
        or evidence_core.context_content_digest != context.context_content_digest
        or acquisition.context_content_digest != context.context_content_digest
        or engine_family != context.engine_family
        or not _is_sha256(vtt_sha256)
        or not _is_sha256(main_json_sha256)
    ):
        raise _binding_failure()
    if not _verified_hash_binding(
        context,
        evidence_core,
        engine_family,
        vtt_sha256,
        main_json_sha256,
    ):
        raise _binding_failure("selected-hash-link")
    consume_context_role(context, "evidence-bind", consumer="bind_align_evidence")
    try:
        projection = _core_value(
            context,
            acquisition,
            evidence_core,
            strict_input_status=strict_input_status,
            v2_policy_status=v2_policy_status,
            profile_status=profile_status,
            evidence_status=evidence_status,
        )
        selected = SelectedOutputs(
            engine_family, True, vtt_sha256, True, main_json_sha256
        )
        _validate_evidence_value(
            {**projection, "selected_outputs": _json_value(selected)}
        )
    except EvidenceBindingError:
        raise
    except Exception as exc:
        raise _binding_failure("closed-schema") from exc
    evidence = object.__new__(FinalAlignEvidence)
    object.__setattr__(evidence, "core", evidence_core)
    object.__setattr__(evidence, "selected_outputs", selected)
    object.__setattr__(evidence, "durable_authority", True)
    object.__setattr__(evidence, "_projection", copy.deepcopy(projection))
    with _LOCK:
        _EVIDENCE[id(evidence)] = _EvidenceRecord(
            evidence, context, copy.deepcopy(evidence)
        )
    return evidence


def _final_value(evidence: FinalAlignEvidence) -> dict[str, Any]:
    return {
        **copy.deepcopy(dict(evidence._projection)),
        "selected_outputs": _json_value(evidence.selected_outputs),
    }


def encode_align_evidence(evidence: FinalAlignEvidence) -> bytes:
    """Encode one genuine bound record in canonical UTF-8/LF form."""
    with _LOCK:
        record = _EVIDENCE.get(id(evidence))
    if (
        record is None
        or record.evidence is not evidence
        or evidence != record.snapshot
        or evidence.durable_authority is not True
    ):
        raise _binding_failure()
    value = _final_value(evidence)
    try:
        _validate_evidence_value(value)
        return (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except EvidenceBindingError:
        raise
    except Exception as exc:
        raise _binding_failure("closed-schema") from exc


def _normalized_basename(path: Path) -> str:
    return unicodedata.normalize("NFC", Path(path).name)


def _media_integrity(
    value: Mapping[str, Any],
    vtt_path: Path,
    explicit_media_path: Path | None,
    corpus_root: Path | None,
) -> bool:
    history = value["input_history"]
    logical_id = history["media_logical_id"]
    expected = history["media_fingerprint"]
    media: Path
    if logical_id.startswith("explicit:"):
        if explicit_media_path is None or corpus_root is not None:
            return False
        media = Path(explicit_media_path)
        if _normalized_basename(media) != logical_id.removeprefix("explicit:"):
            return False
    elif logical_id.startswith("corpus:"):
        if corpus_root is None or explicit_media_path is not None:
            return False
        locator = logical_id.removeprefix("corpus:")
        pure = PurePosixPath(locator)
        if (
            pure.is_absolute()
            or not locator
            or "\\" in locator
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            return False
        root = Path(corpus_root).resolve()
        media = root.joinpath(*pure.parts).resolve()
        try:
            media.relative_to(root)
        except ValueError:
            return False
    elif logical_id.startswith("sibling:"):
        if explicit_media_path is not None or corpus_root is not None:
            return False
        suffix = logical_id.removeprefix("sibling:")
        if (
            not suffix.startswith(".")
            or suffix != suffix.lower()
            or "/" in suffix
            or "\\" in suffix
        ):
            return False
        try:
            resolved = find_sibling_media(vtt_path)
        except OSError:
            return False
        if resolved is None or resolved.suffix.lower() != suffix:
            return False
        media = resolved
    else:
        return False
    try:
        return media.is_file() and media_fingerprint(media) == expected
    except OSError:
        return False


def _w1_usable_audit(root: Mapping[str, Any]) -> bool:
    """Return the unsigned W1 usability conjunction: production limit profile,
    valid V2 admission, full (non-default) source facts, and owner ranges that
    cover every raw unit exactly once."""
    try:
        history = root["input_history"]
        source_facts = root["source_facts"]
        model_facts = source_facts["backend_model_config"]
        route_facts = source_facts["route_input"]
        authority = root["authority_distribution"]
        work = authority["work"]
        owner_ranges = authority["owner_unit_ids"]
        raw_unit_count = root["raw_unit_count"]
    except (KeyError, TypeError):
        return False
    if (
        not isinstance(history, Mapping)
        or not isinstance(source_facts, Mapping)
        or not isinstance(model_facts, Mapping)
        or not isinstance(route_facts, Mapping)
        or tuple(model_facts) != _MODEL_FACT_KEYS
        or tuple(route_facts) != _ROUTE_FACT_KEYS
        or not isinstance(authority, Mapping)
        or not isinstance(work, Mapping)
        or history.get("authority_limit_profile_kind") != "production"
        or work.get("limit_profile_kind") != "production"
        or root.get("v2_admission_status") != "valid"
        or type(raw_unit_count) is not int
        or raw_unit_count < 0
        or not isinstance(owner_ranges, (list, tuple))
        or not owner_ranges
    ):
        return False
    flattened: list[str] = []
    for owner_range in owner_ranges:
        if not isinstance(owner_range, (list, tuple)) or not owner_range:
            return False
        if any(type(unit_id) is not str for unit_id in owner_range):
            return False
        flattened.extend(owner_range)
    return len(flattened) == raw_unit_count and len(set(flattened)) == raw_unit_count


def verify_align_evidence(
    vtt_path: Path,
    *,
    explicit_media_path: Path | None = None,
    corpus_root: Path | None = None,
) -> AlignEvidenceVerification:
    """Verify canonical sidecar bytes and live selected-primary/media links."""
    target = Path(vtt_path)
    legacy_evidence = swap_ext(target, ".align-evidence.json")
    artifact_media = (
        Path(explicit_media_path) if explicit_media_path is not None else None
    )
    if artifact_media is None:
        try:
            artifact_media = find_subtitle_media(target)
        except OSError:
            artifact_media = None
    if artifacts.path_present(legacy_evidence):
        evidence_path = legacy_evidence
    else:
        cached = artifacts.inspect_paths(artifact_media or target)
        evidence_path = (
            legacy_evidence if cached is None else cached.align_evidence(target)
        )
    json_path = swap_ext(target, ".json")
    try:
        encoded = evidence_path.read_bytes()
    except OSError:
        return AlignEvidenceVerification(False, False, "evidence-read")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite,
        )
        canonical = (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        if encoded != canonical:
            _invalid("evidence bytes are not canonical")
        root = _validate_evidence_value(value)
        selected = root["selected_outputs"]
    except (KeyError, TypeError, UnicodeError, ValueError):
        return AlignEvidenceVerification(False, False, "evidence-schema")
    try:
        vtt_bytes = target.read_bytes()
        json_bytes = json_path.read_bytes()
    except OSError:
        return AlignEvidenceVerification(False, False, "selected-output-read")
    if (
        hashlib.sha256(vtt_bytes).hexdigest() != selected["vtt_sha256"]
        or hashlib.sha256(json_bytes).hexdigest() != selected["json_sha256"]
    ):
        return AlignEvidenceVerification(False, False, "selected-output-hash")
    if not _media_integrity(root, target, explicit_media_path, corpus_root):
        return AlignEvidenceVerification(False, False, "media-identity")
    # This audit result never remints or replaces a fresh W1 capability.
    return AlignEvidenceVerification(True, _w1_usable_audit(root), None)


__all__ = [
    "AlignEvidenceVerification",
    "EvidenceBindingError",
    "FinalAlignEvidence",
    "SelectedOutputs",
    "bind_align_evidence",
    "encode_align_evidence",
    "verify_align_evidence",
]
