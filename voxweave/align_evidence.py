"""Post-render in-memory evidence binding under the pending RAT-2 default."""

from __future__ import annotations

import dataclasses
import json
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from voxweave.align_context import IssuedAlignContext, consume_context_role
from voxweave.align_evidence_core import EvidenceCore
from voxweave.align_failures import CanonicalFailure
from voxweave.candidate_encoder import _verified_hash_binding
from voxweave.engine_registry import EngineFamily
from voxweave.p6_ratifications import DURABLE_ALIGN_EVIDENCE_ENABLED


@dataclass(frozen=True)
class SelectedOutputs:
    engine_family: EngineFamily
    vtt_present: Literal[True]
    vtt_sha256: str
    json_present: Literal[True]
    json_sha256: str

    @property
    def main_json_sha256(self) -> str:
        return self.json_sha256


@dataclass(frozen=True)
class FinalAlignEvidence:
    core: EvidenceCore
    selected_outputs: SelectedOutputs
    durable_authority: Literal[False] = False


@dataclass(frozen=True)
class _EvidenceRecord:
    evidence: FinalAlignEvidence
    context: IssuedAlignContext
    snapshot: FinalAlignEvidence


_EVIDENCE: dict[int, _EvidenceRecord] = {}
_LOCK = threading.RLock()


class EvidenceBindingError(RuntimeError):
    def __init__(self, failure: CanonicalFailure):
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


class DurableEvidenceUnavailable(RuntimeError):
    def __init__(self, decision: Literal["RAT-2"] = "RAT-2"):
        super().__init__(f"{decision} remains pending")
        self.decision = decision


def _binding_failure() -> EvidenceBindingError:
    return EvidenceBindingError(
        CanonicalFailure(
            "final-evidence-invalid",
            "evidence-bind",
            "evidence-binding",
        )
    )


def _is_sha256(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def bind_align_evidence(
    context: IssuedAlignContext,
    evidence_core: EvidenceCore,
    *,
    engine_family: EngineFamily,
    vtt_sha256: str,
    main_json_sha256: str,
) -> FinalAlignEvidence:
    """Bind only hashes already reproduced by the independent primary check."""
    if (
        not isinstance(evidence_core, EvidenceCore)
        or evidence_core.context_content_digest != context.context_content_digest
        or engine_family != context.engine_family
        or not _is_sha256(vtt_sha256)
        or not _is_sha256(main_json_sha256)
        or not _verified_hash_binding(
            context,
            evidence_core,
            engine_family,
            vtt_sha256,
            main_json_sha256,
        )
    ):
        raise _binding_failure()
    consume_context_role(context, "evidence-bind", consumer="bind_align_evidence")
    evidence = FinalAlignEvidence(
        evidence_core,
        SelectedOutputs(
            engine_family,
            True,
            vtt_sha256,
            True,
            main_json_sha256,
        ),
    )
    with _LOCK:
        _EVIDENCE[id(evidence)] = _EvidenceRecord(evidence, context, deepcopy(evidence))
    return evidence


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_json_value(member) for member in value]
    if isinstance(value, list):
        return [_json_value(member) for member in value]
    if isinstance(value, dict):
        return {str(key): _json_value(member) for key, member in value.items()}
    return value


def _final_value(evidence: FinalAlignEvidence) -> dict[str, Any]:
    core = _json_value(evidence.core)
    if not isinstance(core, dict):
        raise _binding_failure()
    core["selected_outputs"] = _json_value(evidence.selected_outputs)
    return core


def encode_align_evidence(evidence: FinalAlignEvidence) -> bytes:
    """Encode the closed scaffold once; it is not durable authority while pending."""
    with _LOCK:
        record = _EVIDENCE.get(id(evidence))
    if (
        record is None
        or record.evidence is not evidence
        or evidence != record.snapshot
        or evidence.durable_authority is not False
    ):
        raise _binding_failure()
    value = _final_value(evidence)
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def verify_durable_align_evidence(
    evidence: FinalAlignEvidence,
) -> None:
    """Refuse durable trust until the governing RAT-2 amendment exists."""
    with _LOCK:
        record = _EVIDENCE.get(id(evidence))
    if record is None or record.evidence is not evidence or evidence != record.snapshot:
        raise _binding_failure()
    if DURABLE_ALIGN_EVIDENCE_ENABLED:
        raise RuntimeError(
            "RAT-2 cannot be enabled without its governing implementation"
        )
    raise DurableEvidenceUnavailable()


__all__ = [
    "DurableEvidenceUnavailable",
    "EvidenceBindingError",
    "FinalAlignEvidence",
    "SelectedOutputs",
    "bind_align_evidence",
    "encode_align_evidence",
    "verify_durable_align_evidence",
]
