"""Row builders of the BoundaryOptimizer v2 shadow lane.

These functions serialize a cue stream into the artifact's schema rows: the
per-cue rows, one engine's stream block at one lane, and the finalizer row with
its verification payload. They call only each other. They moved here verbatim
from :mod:`voxweave.core.shadow_v2`, which re-exports every name so that the
test seams on that module (``_shadow_finalizer_row``) keep biting: all callers
of the seams stay in ``shadow_v2``. Every v2 module the rows need (partition
checker, trace validator, finalizer, cost model) is imported lazily inside the
function that uses it, so importing this module pulls none of them in.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from voxweave.core.schema import Cue
from voxweave.core.segdoc import SegDocument

if TYPE_CHECKING:  # the v2 modules are import-free unless the flag is on
    from voxweave.core.partition_check import Origin, Stage


def _shadow_cue_rows(
    cues: Sequence[Cue], partition: Sequence[int] | None, unit_count: int
) -> list[dict[str, Any]]:
    """The artifact projection of one cue stream: display facts plus ownership."""
    from voxweave.core.partition_check import owned_unit_ids

    bounds = (
        owned_unit_ids(partition, unit_count)
        if partition is not None and len(partition) + 1 == len(cues)
        else None
    )
    rows: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        text = str(cue["text"])
        rows.append(
            {
                "end": cue.get("end"),
                "index": index,
                "lines": len(text.split("\n")),
                "lyric": bool(cue.get("lyric", False)),
                "speaker_ids": list(cue.get("speaker_ids") or ()),
                "speech_end": cue.get("speech_end"),
                "speech_start": cue.get("speech_start"),
                "start": cue.get("start"),
                "text": text,
                "unit_range": None if bounds is None else list(bounds[index]),
            }
        )
    return rows


def _shadow_stream_block(
    cues: Sequence[Cue],
    partition: Sequence[int] | None,
    projection: str,
    *,
    document: SegDocument,
    origin: Origin,
    stage: Stage,
    waivers: Mapping[int, Any] | None = None,
    origins: Mapping[int, Origin] | None = None,
    extra: Mapping[str, Any] | None = None,
    validator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One engine's stream at one lane: rows, its partition, and its validator.

    ``validator`` supplies a check another stage already ran on this exact
    stream instead of running a second one.
    """
    from voxweave.core.partition_check import check_partition

    if validator is None and partition is not None:
        validator = check_partition(
            partition,
            cues,
            units=document.units,
            profile=document.profile,
            origin=origin,
            stage=stage,
            waivers=waivers,
            origins=origins,
        ).to_dict()
    block: dict[str, Any] = {
        "cue_count": len(cues),
        "cues": _shadow_cue_rows(cues, partition, len(document.units)),
        "partition": None if partition is None else list(partition),
        "projection": projection,
        "validator": validator,
    }
    block.update(extra or {})
    return block


def _shadow_boundary_times(cues: Sequence[Cue]) -> tuple[float, ...]:
    """Delivered interior boundaries in the one frozen transition coordinate."""
    from voxweave.core.boundary_cost import transition_time

    values: list[float] = []
    for left, right in zip(cues, cues[1:]):
        value = transition_time(left.get("end"), right.get("start"))
        if value is not None:
            values.append(float(value))
    return tuple(values)


def _shadow_movement_distribution(movement: Sequence[Any]) -> dict[str, Any]:
    """Compact absolute phase-2 movement distribution, nearest-rank."""
    import math

    def summary(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)

        def rank(percentile: float) -> float | None:
            if not ordered:
                return None
            return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

        return {
            "count": len(ordered),
            "max": max(ordered) if ordered else None,
            "p50": rank(0.50),
            "p90": rank(0.90),
        }

    return {
        side: summary(
            [abs(float(item.delta)) for item in movement if item.boundary.side == side]
        )
        for side in ("start", "end")
    }


def _shadow_source_units(units: Sequence[Any]) -> list[dict[str, Any]]:
    """Closed source-unit authority used by schema-side partition replay."""
    return [
        {
            "confidence": unit.confidence,
            "end": unit.end,
            "id": unit.id,
            "provenance": unit.provenance,
            "start": unit.start,
            "surface": unit.surface,
        }
        for unit in units
    ]


def _shadow_finalizer_verification(
    stream: Any, evidence: Any, policy: Any
) -> dict[str, Any]:
    """Digest-bound canonical inputs for schema-side finalizer replay."""
    from voxweave.core.finalizer import _stream_payload

    seed = _stream_payload(
        stream.cues,
        stream.profile,
        stream.row_id,
        stream.evaluation_id,
    )
    return {
        "authority_id": stream.seed_id,
        "authority_kind": stream.authority_kind,
        "evidence": {
            "shots": list(evidence.shots),
            "sing_spans": [list(span) for span in evidence.sing_spans],
        },
        "policy": {
            "grid": policy.grid,
            "min_gap": policy.min_gap,
            "overlap_policy": policy.overlap_policy,
        },
        "seed_digest": stream.capability.seal.digest,
        "seed_payload": seed,
    }


def _shadow_finalizer_row(
    result: Any,
    stream: Any,
    partition: Sequence[int] | None,
    *,
    document: SegDocument,
    origin: Any,
    evidence: Any,
    policy: Any,
    projection: str = "solver-partition",
) -> tuple[dict[str, Any], list[Cue]]:
    """Verify and serialize one finalizer root without trusting its trace.

    ``projection`` names where ``partition`` came from: the solver for the v2
    rows, the surface projection (``v1_projection``) for the v1 row.
    """
    from voxweave.core.partition_check import check_partition
    from voxweave.core.trace_validator import replay_trace, stability_check

    cues = [copy.deepcopy(cue) for cue in result.cues]
    delivered = tuple((float(cue["start"]), float(cue["end"])) for cue in cues)
    trace_errors = replay_trace(
        result.trace,
        stream.cues,
        profile=document.profile,
        evidence=evidence,
        policy=policy,
        delivered=delivered,
    )
    stability_errors = stability_check(
        delivered,
        stream.cues,
        profile=document.profile,
        evidence=evidence,
        policy=policy,
        terminal=result.trace.terminal,
    )
    validator = None
    partition_cardinality_ok = partition is not None and len(partition) + 1 == len(cues)
    if result.valid and partition is not None and len(partition) + 1 == len(cues):
        validator = check_partition(
            partition,
            cues,
            units=document.units,
            profile=document.profile,
            origin=origin,
            stage="finalizer",
            reports=result.report.entries,
            waivers={waiver.cue_index: waiver for waiver in result.report.waivers},
        ).to_dict()
    finalizer = {
        **result.report.to_dict(),
        "movement_distribution": _shadow_movement_distribution(result.report.movement),
        "refusals": [entry.to_dict() for entry in result.report.entries],
        "stability_errors": list(stability_errors),
        "trace": result.trace.to_dict(),
        "trace_errors": list(trace_errors),
        "valid": bool(result.valid),
    }
    return (
        {
            "cue_count": len(cues),
            "cues": _shadow_cue_rows(cues, partition, len(document.units)),
            "finalizer": finalizer,
            "partition": None if partition is None else list(partition),
            "projection": (
                projection
                if partition_cardinality_ok
                else "cue/range-cardinality-mismatch"
                if partition is not None
                else "unresolved"
            ),
            "validator": validator,
            "verification": _shadow_finalizer_verification(stream, evidence, policy),
        },
        cues,
    )
