"""Post-render binding and durable path verification for RAT-2 evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from voxweave.align_context import IssuedAlignContext, consume_context_role
from voxweave.align_evidence_core import EvidenceCore
from voxweave.align_failures import CanonicalFailure
from voxweave.candidate_encoder import _verified_hash_binding
from voxweave.engine_registry import EngineFamily
from voxweave.align_snapshot import freeze_json, frozen_json_digest
from voxweave.voicebase import media_fingerprint


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
    durable_authority: Literal[True] = True


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
        super().__init__(f"{failure.kind}/{failure.phase}/{failure.detail_code}")
        self.failure = failure


def _binding_failure() -> EvidenceBindingError:
    return EvidenceBindingError(
        CanonicalFailure(
            "final-evidence-invalid",
            "evidence-bind",
            "evidence-binding",
        )
    )


def _is_sha256(value: object) -> bool:
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
    core = {
        "schema_version": core.pop("schema_version"),
        "kind": "fresh-alignment",
        **core,
    }
    core["selected_outputs"] = _json_value(evidence.selected_outputs)
    return core


def encode_align_evidence(evidence: FinalAlignEvidence) -> bytes:
    """Encode the bound durable record once in its canonical LF form."""
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
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


_TOP_LEVEL_KEYS = (
    "schema_version",
    "kind",
    "context_content_digest",
    "physical_calls",
    "authority_status",
    "authority_reasons",
    "authority_work",
    "call_surface_chars",
    "blocks",
    "raw_unit_count",
    "seed_status",
    "seed_reasons",
    "core_digest",
    "selected_outputs",
)
_SELECTED_OUTPUT_KEYS = (
    "engine_family",
    "vtt_present",
    "vtt_sha256",
    "json_present",
    "json_sha256",
)


class _DuplicateEvidenceKey(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise _DuplicateEvidenceKey(key)
        value[key] = member
    return value


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"nonfinite JSON token {token}")


def _swap_ext(path: Path, new_ext: str) -> Path:
    target = Path(path)
    if target.suffix:
        return target.with_name(target.name[: -len(target.suffix)] + new_ext)
    return target.with_name(target.name + new_ext)


def _core_digest(value: dict[str, Any]) -> str:
    calls = value["physical_calls"]
    work = value["authority_work"]
    blocks = value["blocks"]
    projected = {
        "schema_version": 8,
        "context_content_digest": value["context_content_digest"],
        "physical_calls": [
            {
                "call_index": call["call_index"],
                "strict_unit_status": call["strict_unit_status"],
                "strict_failure": call["strict_failure"],
                "raw_units_sha256": call["raw_units_sha256"],
                "relative_units_sha256": call["relative_units_sha256"],
                "authority_transform_status": call["authority_transform_status"],
                "authority_absolute_sha256": call["authority_absolute_sha256"],
                "raw_unit_ids": call["raw_unit_ids"],
            }
            for call in calls
        ],
        "authority_status": value["authority_status"],
        "authority_reasons": value["authority_reasons"],
        "work_status": work["status"],
        "route_claims": [
            [
                claim["owner_kind"],
                claim["owner_index"],
                claim["delivery_index"],
                claim["source_index"],
            ]
            for claim in work["route_claims"]
        ],
        "work_totals": [
            work["totals"]["states"],
            work["totals"]["edges"],
            work["totals"]["intervals"],
            work["totals"]["normalize_chars"],
        ],
        "call_surface_chars": value["call_surface_chars"],
        "blocks": [
            {
                "source_index": block["source_index"],
                "authority_unit_ids": block["authority_unit_ids"],
                "word_data": (
                    None
                    if block["word_data"] is None
                    else [
                        [
                            word["unit_id"],
                            word["call_index"],
                            word["call_unit_index"],
                            word["text"],
                            word["relative_start"],
                            word["relative_end"],
                            word["physical_origin_seconds"],
                            word["start"],
                            word["end"],
                            word["provenance"],
                            word["original_relative_start"],
                            word["original_relative_end"],
                        ]
                        for word in block["word_data"]
                    ]
                ),
                "speech_start": block["speech_start"],
                "speech_end": block["speech_end"],
            }
            for block in blocks
        ],
        "raw_unit_count": value["raw_unit_count"],
        "seed_status": value["seed_status"],
        "seed_reasons": value["seed_reasons"],
    }
    return frozen_json_digest(freeze_json(projected))


def _media_integrity(
    value: dict[str, Any],
    vtt_path: Path,
    explicit_media_path: Path | None,
    corpus_root: Path | None,
) -> bool:
    history = value.get("input_history")
    if history is None:
        return explicit_media_path is None and corpus_root is None
    if not isinstance(history, dict):
        return False
    logical_id = history.get("media_logical_id")
    expected = history.get("media_fingerprint")
    if type(logical_id) is not str or not _is_sha256(expected):
        return False
    media: Path
    if logical_id.startswith("explicit:"):
        if explicit_media_path is None or corpus_root is not None:
            return False
        media = Path(explicit_media_path)
        if media.name != logical_id.removeprefix("explicit:"):
            return False
    elif logical_id.startswith("corpus:"):
        if corpus_root is None or explicit_media_path is not None:
            return False
        root = Path(corpus_root).resolve()
        media = (root / logical_id.removeprefix("corpus:")).resolve()
        try:
            media.relative_to(root)
        except ValueError:
            return False
    elif logical_id.startswith("sibling:"):
        if explicit_media_path is not None or corpus_root is not None:
            return False
        suffix = logical_id.removeprefix("sibling:")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            return False
        base = _swap_ext(vtt_path, "").name
        matches = sorted(
            child
            for child in vtt_path.parent.iterdir()
            if child.is_file()
            and _swap_ext(child, "").name == base
            and child.suffix.lower() == suffix.lower()
        )
        if len(matches) != 1:
            return False
        media = matches[0]
    else:
        return False
    try:
        return media.is_file() and media_fingerprint(media) == expected
    except OSError:
        return False


def verify_align_evidence(
    vtt_path: Path,
    *,
    explicit_media_path: Path | None = None,
    corpus_root: Path | None = None,
) -> AlignEvidenceVerification:
    """Verify one sidecar only through its current VTT-relative durable path."""
    target = Path(vtt_path)
    evidence_path = _swap_ext(target, ".align-evidence.json")
    json_path = _swap_ext(target, ".json")
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
        if not isinstance(value, dict):
            raise ValueError("evidence root is not an object")
        canonical = (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        if encoded != canonical:
            raise ValueError("evidence bytes are not canonical")
        if (
            tuple(value) != _TOP_LEVEL_KEYS
            or value["schema_version"] != 8
            or value["kind"] != "fresh-alignment"
            or not _is_sha256(value["context_content_digest"])
            or not _is_sha256(value["core_digest"])
            or _core_digest(value) != value["core_digest"]
        ):
            raise ValueError("evidence schema or core digest is invalid")
        selected = value["selected_outputs"]
        if (
            not isinstance(selected, dict)
            or tuple(selected) != _SELECTED_OUTPUT_KEYS
            or selected["engine_family"] not in ("legacy-v1", "boundary-v2")
            or selected["vtt_present"] is not True
            or selected["json_present"] is not True
            or not _is_sha256(selected["vtt_sha256"])
            or not _is_sha256(selected["json_sha256"])
        ):
            raise ValueError("selected output binding is invalid")
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
    if not _media_integrity(
        value,
        target,
        explicit_media_path,
        corpus_root,
    ):
        return AlignEvidenceVerification(False, False, "media-identity")
    # The unsigned durable sidecar is audit evidence only.  It never remints
    # the live fresh-call capability even when every recorded predicate passes.
    return AlignEvidenceVerification(True, False, None)


__all__ = [
    "AlignEvidenceVerification",
    "EvidenceBindingError",
    "FinalAlignEvidence",
    "SelectedOutputs",
    "bind_align_evidence",
    "encode_align_evidence",
    "verify_align_evidence",
]
