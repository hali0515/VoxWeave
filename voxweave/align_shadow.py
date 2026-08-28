"""Rich immutable observation for a completed selected align transaction."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from voxweave.align_delta_registry import (
    ALIGN_DELTA_IDS,
    ALIGN_DELTA_REGISTRY_SHA256,
)
from voxweave.align_evidence import encode_align_evidence
from voxweave.align_failures import CanonicalFailure, OUTCOME_DETAILS


_RICH_KEYS = (
    "schema_version",
    "artifact_kind",
    "status",
    "failure",
    "input",
    "fresh",
    "legacy",
    "v2",
    "comparison",
    "selected",
)
_INPUT_KEYS = (
    "context_content_digest",
    "vtt_sha256",
    "sibling_present",
    "sibling_sha256",
    "media_fingerprint",
    "media_logical_id",
    "effective_iso",
    "route",
    "block_count",
    "block_content_sha256",
    "profile_source",
)
_FRESH_KEYS = (
    "receipt_digest",
    "prepared_audio_sha256",
    "physical_call_count",
    "raw_unit_count",
    "legacy_distribution_digest",
    "authority_distribution_digest",
    "authority_distribution_status",
    "seed_status",
    "strict_input_status",
    "v2_policy_status",
    "profile_status",
    "evidence_status",
    "v2_admission_status",
)
_LEGACY_KEYS = ("normalized_delivery", "vtt_sha256", "json_sha256")
_SEMANTIC_KEYS = (
    "semantic_root_lineage",
    "phase1_seed",
    "delivered",
    "report",
    "trace",
)
_LINEAGE_KEYS = (
    "expected_row_id",
    "input_kind",
    "authority_kind",
    "parent_absent",
    "record_count",
    "matching_row_count",
    "stable_input_digest",
)
_VALIDATOR_KEYS = (
    "partition_result",
    "trace_problems",
    "stability_problems",
)
_COMPARISON_KEYS = (
    "registry_sha256",
    "active_classes",
    "primitive_field_diffs",
    "violations",
)
_SELECTED_KEYS = (
    "engine_family",
    "vtt_sha256",
    "json_sha256",
    "evidence_sha256",
)
_PRIMITIVE_FIELD_ORDER = ("authority-time", "text", "start", "end", "lyric")


def _immutable(value: Any) -> Any:
    """Freeze a JSON projection while preserving mapping and sequence order."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("rich align shadow mapping key is not a string")
        return MappingProxyType(
            {key: _immutable(member) for key, member in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(member) for member in value)
    return value


def _json_value(value: Any) -> Any:
    """Project the closed observation value types without a fallback ``repr``."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("rich align shadow contains a nonfinite float")
        return value
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, member in value.items():
            if type(key) is not str:
                raise TypeError("rich align shadow mapping key is not a string")
            projected[key] = _json_value(member)
        return projected
    if isinstance(value, (list, tuple)):
        return [_json_value(member) for member in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    raise TypeError(
        f"rich align shadow member has unsupported type {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ordered_keys(value: object, expected: tuple[str, ...], *, label: str) -> None:
    if not isinstance(value, Mapping) or tuple(value) != expected:
        raise ValueError(f"rich align shadow {label} keys are not closed and ordered")


def _failure_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or tuple(value) != (
        "kind",
        "phase",
        "detail_code",
        "secondary",
    ):
        return False
    kind = value["kind"]
    phase = value["phase"]
    detail = value["detail_code"]
    if (
        type(kind) is not str
        or type(phase) is not str
        or not phase
        or type(detail) is not str
        or detail not in OUTCOME_DETAILS.get(kind, ())
    ):
        return False
    secondary = value["secondary"]
    return isinstance(secondary, list) and all(
        isinstance(item, Mapping)
        and tuple(item) == ("kind", "phase", "detail_code")
        and type(item["kind"]) is str
        and type(item["phase"]) is str
        and bool(item["phase"])
        and type(item["detail_code"]) is str
        and item["detail_code"] in OUTCOME_DETAILS.get(item["kind"], ())
        for item in secondary
    )


def _validate_rich_artifact(value: Mapping[str, Any]) -> None:
    """Validate the exact rich union member without a runtime schema dependency."""
    _ordered_keys(value, _RICH_KEYS, label="top-level")
    if value["schema_version"] != 2 or value["artifact_kind"] != "rich":
        raise ValueError("rich align shadow discriminator is invalid")
    if value["status"] not in ("valid", "invalid"):
        raise ValueError("rich align shadow status is invalid")
    failure = value["failure"]
    if failure is not None and not _failure_valid(failure):
        raise ValueError("rich align shadow failure is invalid")
    if (value["status"] == "valid") != (failure is None):
        raise ValueError("rich align shadow failure does not match status")

    _ordered_keys(value["input"], _INPUT_KEYS, label="input")
    _ordered_keys(value["fresh"], _FRESH_KEYS, label="fresh")
    _ordered_keys(value["legacy"], _LEGACY_KEYS, label="legacy")
    _ordered_keys(value["v2"], ("semantic", "validators"), label="v2")
    _ordered_keys(value["comparison"], ("result",), label="comparison")
    _ordered_keys(value["selected"], _SELECTED_KEYS, label="selected")

    input_value = value["input"]
    if (
        not _sha256(input_value["context_content_digest"])
        or not _sha256(input_value["vtt_sha256"])
        or type(input_value["sibling_present"]) is not bool
        or (
            _sha256(input_value["sibling_sha256"])
            if input_value["sibling_present"]
            else input_value["sibling_sha256"] is None
        )
        is not True
        or not _sha256(input_value["media_fingerprint"])
        or type(input_value["media_logical_id"]) is not str
        or not input_value["media_logical_id"]
        or type(input_value["effective_iso"]) is not str
        or not input_value["effective_iso"]
        or input_value["route"] not in ("ctc-full", "mms-full", "qwen-crop")
        or type(input_value["block_count"]) is not int
        or input_value["block_count"] < 0
        or not _sha256(input_value["block_content_sha256"])
        or input_value["profile_source"]
        not in (
            "language-override",
            "unsupported-manifest",
            "profile-absent",
            "stored-profile",
            "manifest-absent",
        )
    ):
        raise ValueError("rich align shadow input facts are invalid")

    fresh = value["fresh"]
    if (
        not all(
            _sha256(fresh[key])
            for key in (
                "receipt_digest",
                "prepared_audio_sha256",
                "legacy_distribution_digest",
                "authority_distribution_digest",
            )
        )
        or any(
            type(fresh[key]) is not int or fresh[key] < 0
            for key in ("physical_call_count", "raw_unit_count")
        )
        or any(
            fresh[key] not in ("valid", "invalid")
            for key in (
                "authority_distribution_status",
                "seed_status",
                "strict_input_status",
                "v2_policy_status",
                "profile_status",
                "evidence_status",
                "v2_admission_status",
            )
        )
    ):
        raise ValueError("rich align shadow fresh facts are invalid")

    legacy = value["legacy"]
    if (
        not isinstance(legacy["normalized_delivery"], list)
        or not _sha256(legacy["vtt_sha256"])
        or not _sha256(legacy["json_sha256"])
    ):
        raise ValueError("rich align shadow legacy facts are invalid")
    for cue in legacy["normalized_delivery"]:
        _ordered_keys(
            cue,
            ("source_index", "text", "start", "end", "lyric", "unit_ids"),
            label="legacy delivery cue",
        )
        if (
            type(cue["source_index"]) is not int
            or cue["source_index"] < 0
            or type(cue["text"]) is not str
            or type(cue["start"]) is not float
            or type(cue["end"]) is not float
            or cue["lyric"] not in (None, True)
            or not isinstance(cue["unit_ids"], list)
            or not all(type(unit_id) is str for unit_id in cue["unit_ids"])
        ):
            raise ValueError("rich align shadow legacy delivery is invalid")

    selected = value["selected"]
    if selected["engine_family"] not in ("legacy-v1", "boundary-v2"):
        raise ValueError("rich align shadow selected family is invalid")
    if not all(_sha256(selected[key]) for key in _SELECTED_KEYS[1:]):
        raise ValueError("rich align shadow selected hash is invalid")
    semantic = value["v2"]["semantic"]
    validators = value["v2"]["validators"]
    comparison = value["comparison"]["result"]
    if semantic is None and validators is not None:
        raise ValueError("rich align shadow validators lack semantic facts")
    if semantic is not None:
        _ordered_keys(semantic, _SEMANTIC_KEYS, label="semantic")
        lineage = semantic["semantic_root_lineage"]
        _ordered_keys(lineage, _LINEAGE_KEYS, label="semantic root lineage")
        if (
            lineage["expected_row_id"] != "align/delivery-finalizer/v2"
            or lineage["input_kind"] != "phase1"
            or lineage["authority_kind"] != "fresh-alignment"
            or lineage["parent_absent"] is not True
            or lineage["record_count"] != 1
            or lineage["matching_row_count"] != 1
            or lineage["stable_input_digest"] != _digest(semantic["phase1_seed"])
            or not isinstance(semantic["phase1_seed"], list)
            or not isinstance(semantic["delivered"], list)
            or not isinstance(semantic["report"], Mapping)
            or not isinstance(semantic["trace"], Mapping)
        ):
            raise ValueError("rich align shadow semantic facts are invalid")
    if validators is not None:
        _ordered_keys(validators, _VALIDATOR_KEYS, label="validators")
        if (
            not isinstance(validators["partition_result"], Mapping)
            or not isinstance(validators["trace_problems"], list)
            or not isinstance(validators["stability_problems"], list)
        ):
            raise ValueError("rich align shadow validator facts are invalid")
    if comparison is not None:
        _ordered_keys(comparison, _COMPARISON_KEYS, label="comparison result")
        if semantic is None:
            raise ValueError("rich align shadow comparison lacks semantic facts")
        if comparison["registry_sha256"] != ALIGN_DELTA_REGISTRY_SHA256:
            raise ValueError("rich align shadow comparison registry mismatch")
        active = comparison["active_classes"]
        primitive_fields = comparison["primitive_field_diffs"]
        violations = comparison["violations"]
        if (
            not isinstance(active, list)
            or active != [delta for delta in ALIGN_DELTA_IDS[:-1] if delta in active]
            or len(set(active)) != len(active)
            or not isinstance(primitive_fields, list)
            or primitive_fields
            != [field for field in _PRIMITIVE_FIELD_ORDER if field in primitive_fields]
            or len(set(primitive_fields)) != len(primitive_fields)
            or not isinstance(violations, list)
            or violations
            != [delta for delta in ALIGN_DELTA_IDS[:-1] if delta in violations]
            or len(set(violations)) != len(violations)
            or any(delta not in active for delta in violations)
        ):
            raise ValueError("rich align shadow comparison values are invalid")

    if value["status"] == "valid":
        if semantic is None or validators is None or comparison is None:
            raise ValueError("valid rich align shadow lacks complete v2 facts")
        if any(
            fresh[key] != "valid"
            for key in (
                "authority_distribution_status",
                "seed_status",
                "strict_input_status",
                "v2_policy_status",
                "profile_status",
                "evidence_status",
                "v2_admission_status",
            )
        ):
            raise ValueError("valid rich align shadow carries an invalid status")
        if validators["trace_problems"] or validators["stability_problems"]:
            raise ValueError("valid rich align shadow carries validator problems")
        if comparison["violations"]:
            raise ValueError("valid rich align shadow carries comparison violations")
    _canonical_bytes(value)


@dataclass(frozen=True)
class RichAlignShadowArtifact:
    schema_version: Literal[2]
    artifact_kind: Literal["rich"]
    status: Literal["valid", "invalid"]
    failure: CanonicalFailure | None
    input: Mapping[str, Any]
    fresh: Mapping[str, Any]
    legacy: Mapping[str, Any]
    v2: Mapping[str, Any]
    comparison: Mapping[str, Any]
    selected: Mapping[str, str]

    def _value(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "input": self.input,
            "fresh": self.fresh,
            "legacy": self.legacy,
            "v2": self.v2,
            "comparison": self.comparison,
            "selected": self.selected,
        }

    def to_canonical_bytes(self) -> bytes:
        value = self._value()
        projected = _json_value(value)
        assert isinstance(projected, Mapping)
        _validate_rich_artifact(projected)
        return _canonical_bytes(value)


def _normalized_delivery(cues: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_index": cue.source_index,
            "text": cue.text,
            "start": cue.start,
            "end": cue.end,
            "lyric": cue.lyric,
            "unit_ids": list(cue.unit_ids),
        }
        for cue in cues
    ]


def _cue_values(cues: object) -> list[Any]:
    if not isinstance(cues, (list, tuple)):
        raise TypeError("rich semantic cue group is not an ordered sequence")
    values: list[Any] = []
    for cue in cues:
        projected = _json_value(cue)
        if isinstance(projected, Mapping):
            projected = {
                key: member
                for key, member in projected.items()
                if not key.startswith("_")
            }
        values.append(projected)
    return values


def _semantic_root_lineage(
    records: object, *, stable_input_digest: str
) -> dict[str, Any]:
    if not isinstance(records, (list, tuple)):
        raise TypeError("rich semantic lineage is not an ordered sequence")
    normalized: list[tuple[Any, ...]] = []
    for record in records:
        if not isinstance(record, (list, tuple)) or len(record) != 6:
            raise ValueError("rich semantic lineage record has the wrong shape")
        normalized.append(tuple(record))
    expected_row = "align/delivery-finalizer/v2"
    return {
        "expected_row_id": expected_row,
        "input_kind": "phase1",
        "authority_kind": "fresh-alignment",
        "parent_absent": all(record[5] is None for record in normalized),
        "record_count": len(normalized),
        "matching_row_count": sum(record[1] == expected_row for record in normalized),
        "stable_input_digest": stable_input_digest,
    }


def _semantic_groups(
    selection: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from voxweave.align_adapter import _align_semantic_observation

    observed = _align_semantic_observation(selection.context, selection.result)
    if observed is None:
        return None, None
    phase1_seed = _cue_values(observed.phase1_seed)
    semantic = {
        "semantic_root_lineage": _semantic_root_lineage(
            observed.semantic_root_lineage,
            stable_input_digest=_digest(phase1_seed),
        ),
        "phase1_seed": phase1_seed,
        "delivered": _cue_values(observed.delivered),
        "report": _json_value(observed.report),
        "trace": _json_value(observed.trace),
    }
    validators = None
    if (
        observed.partition_result is not None
        and observed.trace_problems is not None
        and observed.stability_problems is not None
    ):
        validators = {
            "partition_result": _json_value(observed.partition_result),
            "trace_problems": _json_value(observed.trace_problems),
            "stability_problems": _json_value(observed.stability_problems),
        }
    return semantic, validators


def _comparison_result(comparison: Any) -> dict[str, Any] | None:
    if comparison is None:
        return None
    return {
        "registry_sha256": comparison.registry_sha256,
        "active_classes": list(comparison.active_classes),
        "primitive_field_diffs": list(comparison.primitive_field_diffs),
        "violations": list(comparison.violations),
    }


def build_rich_align_shadow_artifact(
    *,
    selection: Any,
    input_summary: Mapping[str, Any],
    prepared_audio_sha256: str,
) -> RichAlignShadowArtifact:
    """Construct and canonicalize the complete post-commit rich observation."""
    result = selection.result
    verified = selection.verified
    core = result.evidence_core
    failure = selection.observation_failure or result.v2_status.failure
    status: Literal["valid", "invalid"] = "invalid" if failure is not None else "valid"
    distribution_value = dataclasses.asdict(selection.distribution)
    evidence_sha256 = hashlib.sha256(
        encode_align_evidence(selection.evidence)
    ).hexdigest()
    legacy_delivery = _normalized_delivery(result.legacy.cues)
    semantic, validators = _semantic_groups(selection)
    comparison = _comparison_result(result.comparison)
    artifact = RichAlignShadowArtifact(
        2,
        "rich",
        status,
        failure,
        _immutable(dict(input_summary)),
        _immutable(
            {
                "receipt_digest": result.receipt_digest,
                "prepared_audio_sha256": prepared_audio_sha256,
                "physical_call_count": len(core.physical_calls),
                "raw_unit_count": core.raw_unit_count,
                "legacy_distribution_digest": _digest(legacy_delivery),
                "authority_distribution_digest": _digest(distribution_value),
                "authority_distribution_status": core.authority_status,
                "seed_status": core.seed_status,
                "strict_input_status": selection.strict_input_status.kind,
                "v2_policy_status": selection.v2_policy_status.kind,
                "profile_status": selection.profile_status.kind,
                "evidence_status": selection.evidence_status.kind,
                "v2_admission_status": (
                    "invalid"
                    if selection.observation_failure is not None
                    else "valid"
                    if result.v2_status.kind == "valid"
                    else result.v2_status.kind
                ),
            }
        ),
        _immutable(
            {
                "normalized_delivery": legacy_delivery,
                "vtt_sha256": selection.legacy_vtt_sha256,
                "json_sha256": selection.legacy_main_json_sha256,
            }
        ),
        _immutable({"semantic": semantic, "validators": validators}),
        _immutable({"result": comparison}),
        _immutable(
            {
                "engine_family": verified.engine_family,
                "vtt_sha256": verified.vtt_sha256,
                "json_sha256": verified.main_json_sha256,
                "evidence_sha256": evidence_sha256,
            }
        ),
    )
    value = _json_value(artifact._value())
    assert isinstance(value, Mapping)
    _validate_rich_artifact(value)
    artifact.to_canonical_bytes()
    return artifact


__all__ = ["RichAlignShadowArtifact", "build_rich_align_shadow_artifact"]
