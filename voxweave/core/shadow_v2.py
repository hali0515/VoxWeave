"""BoundaryOptimizer v2 shadow lane, measured beside the shipped v1 answer.

This is the code that used to sit inside :mod:`voxweave.pipeline` between the
v1 engine call and the legacy overlays. It moved here verbatim so the pipeline
module carries only the hook: ``pipeline.segment_document`` reads
:data:`SEG_V2_SHADOW_ENV` first and reaches :func:`run_shadow` only when the
flag is on. Every v2 module the lane needs (optimizer, finalizer, schema
validator, ...) is imported lazily inside the function that uses it, never at
module scope, so a flag-off run still pulls none of them into the process.

Import direction: this module never imports :mod:`voxweave.pipeline` at module
scope. The handful of pipeline helpers the lane replays (``mark_lyric_cues``,
``_copied_spans``, ``_copied_turns``, ``_resnap_shots``, ``LYRIC_MIN_OVERLAP``)
are imported lazily inside the functions that use them; by the time any of
those runs ``pipeline`` is already loaded, because it is the only caller. The
lane constants below are re-exported from ``pipeline`` for downstream readers.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from voxweave.core.providers import degradation_capture
from voxweave.core.schema import Cue, Unit
from voxweave.core.segdoc import SegDocument
from voxweave.core.shadow_lanes import (
    LANE_CORE,
    LANE_DISPLAY,
    LANE_FINALIZER,
    LANE_LEGACY,
)

if TYPE_CHECKING:  # the v2 modules are import-free unless the flag is on
    from voxweave.core.boundary_v2 import DocumentSolution
    from voxweave.core.finalizer import FinalizerPreview
    from voxweave.core.partition_check import Origin, Stage
    from voxweave.core.timing_preview import CueCandidate, CuePreview

log = logging.getLogger("voxweave")

#: Opt-in for the BoundaryOptimizer v2 shadow measurement. Parsed with the
#: exact-``"1"`` test the manifest already uses for the VAD mask: a paired
#: ``--no-`` flag writes the literal ``"0"``, which is truthy as a *string*, so
#: ``bool()`` would latch the shadow on for the run that turned it off.
#:
#: Deliberately environment-only, and deliberately absent from the manifest. The
#: manifest is the record of what produced the *shipped* output and a shadow run
#: ships v1, so mentioning it there would move persisted bytes for a lane that
#: changes nothing. A ``[defaults]`` conf key is worse still: inner keys are
#: never validated, so a typo is silent and a latched value could leave a user
#: running a measurement build for months.
SEG_V2_SHADOW_ENV = "VOXWEAVE_SEG_V2_SHADOW"

#: P5's lane names, from the one leaf both this hook and the schema read. Core
#: and the renamed legacy delivery proxy retain the P4 evidence; the finalizer
#: row matrix is gated, and the legacy-display isolation comparator supplies
#: N3a/N11 without speaker overlay or resnapping.
SHADOW_LANE_CORE = LANE_CORE
SHADOW_LANE_DELIVERY_LEGACY = LANE_LEGACY
SHADOW_LANE_FINALIZER = LANE_FINALIZER
SHADOW_LANE_LEGACY_DISPLAY = LANE_DISPLAY
# Compatibility name for downstream readers that imported the P4 constant.
SHADOW_LANE_DELIVERY = SHADOW_LANE_DELIVERY_LEGACY

#: Machine-readable causes of a ``segmentation-shadow-incomplete`` envelope,
#: carried as ``error.reason`` beside the human ``error.detail``.
INCOMPLETE_REASONS: tuple[str, ...] = (
    "invalid-profile",
    "optimizer-selection-unavailable",
    "refiner-off-unavailable",
    "v1-unprojected",
)

#: How many schema-2 refusals an error detail quotes; the full list rides on
#: the error envelope as ``admission_errors``.
ADMISSION_ERRORS_SHOWN = 10

#: Longest error detail the failure warning quotes.
LOG_DETAIL_CHARS = 500


def _shadow_surface_partition(
    units: Sequence[Any], cues: Sequence[Cue]
) -> tuple[tuple[int, ...] | None, str]:
    """Project cue boundaries by stored surfaces, never by a character cursor."""
    from voxweave.core.smart_split import _surface_ranges

    if not cues:
        return (), "empty"
    word_data = [entry for cue in cues for entry in cue.get("word_data") or ()]
    ranges = _surface_ranges([unit.surface for unit in units], word_data)
    if ranges is None or len(ranges) != len(units):
        return None, "surface-reconciliation-failed"
    boundaries = {
        ranges[index - 1][1]: index
        for index in range(1, len(ranges))
        if ranges[index - 1][1] == ranges[index][0]
    }
    cursor = 0
    cuts: list[int] = []
    for cue in cues[:-1]:
        cursor += len(cue.get("word_data") or ())
        cut = boundaries.get(cursor)
        if cut is None:
            return None, f"surface-boundary-unresolved-at-entry-{cursor}"
        cuts.append(cut)
    cursor += len(cues[-1].get("word_data") or ())
    if cursor != len(word_data):
        return None, f"surface-stream-ends-at-entry-{cursor}-of-{len(word_data)}"
    if any(left >= right for left, right in zip(cuts, cuts[1:])):
        return None, "surface-cuts-non-monotone"
    return tuple(cuts), "surface-footprint"


def _shadow_v1_partition(
    parent: SegDocument,
    origin: Sequence[int],
    cues: Sequence[Cue],
) -> tuple[tuple[int, ...] | None, str]:
    """Resolve v1 structurally, translating parent coordinates through origin."""
    import bisect

    parent_cuts, parent_mode = _shadow_surface_partition(parent.units, cues)
    if parent_cuts is not None:
        translated = tuple(bisect.bisect_left(origin, cut) for cut in parent_cuts)
        return translated, f"{parent_mode}-parent-through-origin"
    # Translation through ``origin`` cannot fail; only the parent projection can.
    return None, f"parent:{parent_mode}"


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


def _restamp_by_footprint(
    waivers: Mapping[int, Any], partition: Sequence[int] | None, unit_count: int
) -> dict[int, Any]:
    """Re-point an interval-minted waiver ledger at another stage's cue indices.

    A waiver's cue index is only meaningful against the stream it was minted for,
    and the later stages re-time, re-wrap and (in the overlay lane) split and
    merge cues -- so index identity is not a contract. Source-unit ownership is:
    the exemption names the units it covers, and every stage still owns those
    units in exactly one cue. Handing the checker no ledger at all -- the shape
    this replaced -- made it re-report an exemption the solver had granted as an
    *unwaived*, exit-driving violation, which is the failure mode
    ``_document_waivers`` exists to prevent, reproduced one level up.
    """
    from dataclasses import replace

    from voxweave.core.partition_check import owned_unit_ids

    if partition is None or not waivers:
        return {}
    bounds = owned_unit_ids(partition, unit_count)
    out: dict[int, Any] = {}
    for waiver in waivers.values():
        if not waiver.unit_ids:
            continue
        low, high = min(waiver.unit_ids), max(waiver.unit_ids) + 1
        for index, (start, end) in enumerate(bounds):
            if start <= low and high <= end:
                out[index] = replace(waiver, cue_index=index)
                break
    return out


def _refined_partition(
    parent_partition: Sequence[int], origin: Sequence[int]
) -> list[int]:
    """Re-express parent-unit cuts as refined-unit cuts through ``origin``.

    ``origin`` is monotone and names every parent, so parent ``p`` begins at the
    first refined unit it owns.
    """
    first_child: dict[int, int] = {}
    for index, parent in enumerate(origin):
        first_child.setdefault(parent, index)
    return [first_child[cut] for cut in parent_partition]


def _origins_by_footprint(
    fallback_ranges: Sequence[Sequence[int]],
    partition: Sequence[int] | None,
    unit_count: int,
) -> dict[int, Origin]:
    """Which engine produced each cue of a stage's stream, by unit ownership.

    AD3-3 attributes a violation to the engine that produced the violating cue.
    A document with one adopted-v1 interval is not a v1 document, so typing the
    whole stage "v2" blames v2 for v1's damage and typing it "v1" excuses v2's.
    """
    from voxweave.core.partition_check import owned_unit_ids

    if partition is None or not fallback_ranges:
        return {}
    bounds = owned_unit_ids(partition, unit_count)
    return {
        index: "v1"
        for index, (start, end) in enumerate(bounds)
        if any(start < high and end > low for low, high in fallback_ranges)
    }


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


def _shadow_lane_block(
    lane: str, stage: Stage, v1: Mapping[str, Any], v2: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair the two engines at one lane and state where they disagree."""
    agreement: dict[str, Any] | None = None
    if v1["partition"] is not None and v2["partition"] is not None:
        left = set(v1["partition"])
        right = set(v2["partition"])
        agreement = {
            "identical_cuts": len(left & right),
            "v1_cut_count": len(left),
            "v1_only": sorted(left - right),
            "v2_cut_count": len(right),
            "v2_only": sorted(right - left),
        }
    return {
        "agreement": agreement,
        "lane": lane,
        "stage": stage,
        "v1": dict(v1),
        "v2": dict(v2),
    }


def _shadow_core_cues(
    solution: DocumentSolution, document: SegDocument, thresholds: Mapping[str, Any]
) -> list[Cue]:
    """v2's raw materialization, finished the way v1 finishes its own stream.

    Only the passes that are *not* boundary decisions are replayed: the timing
    cleanup, the shot snap and the text finalization. The merge/glue/repair
    passes are boundary decisions and are precisely what v2 replaces, so
    replaying them here would grade v2 on v1's repairs.
    """
    from voxweave.core.layout import (
        _line_budget_width,
        _merge_stutters,
        strip_punct_for_subtitles,
        wrap_cue_text,
    )
    from voxweave.core.smart_split import SplitThresholds
    from voxweave.core.timing import _cleanup_cues, _snap_to_shots

    profile = document.profile
    lang = profile.language
    th = SplitThresholds.from_mapping(dict(thresholds))
    cues: list[Cue] = [
        copy.deepcopy(cue) for item in solution.solutions for cue in item.cues
    ]
    cues = _cleanup_cues(
        cues,
        min_cue_s=th.min_cue_s,
        max_cue_s=th.max_cue_s,
        cps=th.cps,
        lag_out_s=th.lag_out_s,
    )
    if document.shot_changes:
        cues = _snap_to_shots(
            cues,
            sorted(document.shot_changes),
            snap_s=th.shot_snap_s,
            max_cue_s=th.max_cue_s,
        )
    width = _line_budget_width(profile.max_line_length, lang)
    for cue in cues:
        cue["text"] = wrap_cue_text(
            _merge_stutters(strip_punct_for_subtitles(cue["text"])),
            lang,
            profile.max_lines,
            max_line_length=width,
        )
    return cues


def _shadow_overlay_cues(
    cues: Sequence[Cue], document: SegDocument, thresholds: Mapping[str, Any]
) -> list[Cue]:
    """The legacy overlays applied to a copy of a stream, for the delivery lane.

    Every input the overlays touch is copied first: production runs these same
    overlays on the real cue stream immediately after the hook returns, so a
    formatter that mutated its ``turns`` list here would change shipped bytes.
    """
    # Deferred: these are the production overlays and stay in ``pipeline``; the
    # lane must never import that module at module scope (see module docstring).
    from voxweave.pipeline import (
        _copied_spans,
        _copied_turns,
        _resnap_shots,
        mark_lyric_cues,
    )

    out: list[Cue] = [copy.deepcopy(cue) for cue in cues]
    mark_lyric_cues(out, _copied_spans(document.sing_spans))
    turns = _copied_turns(document.speaker_turns)
    if turns:
        from voxweave.diarize import apply_speaker_format

        out = apply_speaker_format(
            out,
            turns,
            document.language,
            thresholds=dict(thresholds),
            max_line_length=document.profile.max_line_length,
            max_lines=document.profile.max_lines,
        )
        out = _resnap_shots(
            out, list(document.shot_changes or ()) or None, dict(thresholds)
        )
    return out


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


def _shadow_fallback_rechecks(
    stream: Any,
    footprints: Sequence[str],
    *,
    row_id: str,
) -> list[dict[str, Any]]:
    """Show whether W1's missing factory footprint is the sole fallback cause."""
    from voxweave.core.canonical_text import canonical_text

    checks: list[dict[str, Any]] = []
    if len(footprints) != len(stream.cues):
        return [
            {
                "cue_index": None,
                "reason": "footprint-cardinality-mismatch",
                "row": row_id,
                "with_owned_footprint": None,
            }
        ]
    for cue, footprint in zip(stream.cues, footprints):
        fallback = next(
            (
                report
                for report in cue.reports
                if report.kind == "canonical-text-fallback"
            ),
            None,
        )
        if fallback is None:
            continue
        replayed = canonical_text(
            cue.word_data,
            fallback_text=cue.text,
            lang=stream.profile.language,
            profile=stream.profile,
            expected_footprint=footprint,
        )
        checks.append(
            {
                "cue_index": cue.index,
                "reason": fallback.evidence.get("reason"),
                "row": row_id,
                "with_owned_footprint": replayed.source,
                "with_owned_footprint_reason": replayed.fallback_reason,
            }
        )
    return checks


def _shadow_stamp_comparator_deltas(
    finalizer_row: Mapping[str, Any], comparator_row: Mapping[str, Any]
) -> None:
    """Join W4's upstream FD-2 producer fact into the finalizer row report."""
    finalizer = finalizer_row.get("finalizer")
    if not isinstance(finalizer, dict):
        return
    evidence_flags = {
        tuple(row["unit_range"]): bool(row.get("lyric"))
        for row in finalizer_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    legacy_flags = {
        tuple(row["unit_range"]): bool(row.get("lyric"))
        for row in comparator_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    fired: set[str] = set(finalizer.get("deltas_fired") or ())
    if any(
        evidence_flags[unit_range] != legacy_flags[unit_range]
        for unit_range in evidence_flags.keys() & legacy_flags.keys()
    ):
        fired.add("FD-2")
    finalizer["deltas_fired"] = sorted(fired)


def _shadow_diff_classification(
    finalizer_row: Mapping[str, Any],
    comparator_row: Mapping[str, Any],
    *,
    stream: Any,
    seed_cues: Sequence[Cue],
    sing_spans: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    """N11: independently recompute per-cue triggers and allowed relations.

    A changed field is approved only against a fact recomputed here from the
    phase-1 ``stream`` and the immutable ``seed_cues``; a cue with no such fact
    (the two disagree in cardinality) is counted unclassified, never approved on
    the producer's own claim.
    """
    from voxweave.core.speaker_evidence import (
        evidence_span_from_cue,
        lyric_for_evidence,
    )
    from voxweave.core.timing import LINGER_CAP_S, TWO_FRAME_S
    from voxweave.pipeline import LYRIC_MIN_OVERLAP

    finalizer = finalizer_row["finalizer"]
    producer_fired = set(finalizer.get("deltas_fired") or ())
    trace_clean = not finalizer.get("trace_errors") and not finalizer.get(
        "stability_errors"
    )
    permitted = {
        "text": {"FD-9"},
        "start": {"FD-4"},
        "end": {"FD-1", "FD-3", "FD-4", "FD-6", "FD-8"},
        "lyric": {"FD-2"},
    }
    facts: dict[int, dict[str, Any]] = {}
    independent_fired: set[str] = set()
    if len(stream.cues) == len(seed_cues):
        extends = (
            stream.profile.min_cue_s > 0
            or stream.profile.lag_out_s > 0
            or stream.profile.cps > 0
        )
        trace_legs = finalizer.get("trace", {}).get("legs") or ()
        for index, (phase1, seed) in enumerate(zip(stream.cues, seed_cues)):
            per_field = {field: set() for field in permitted}
            if phase1.reading_chars != phase1.raw_reading_chars:
                per_field["end"].add("FD-1")
                independent_fired.add("FD-1")
            if phase1.speech_end is None and extends:
                per_field["end"].add("FD-8")
                independent_fired.add("FD-8")
            if any(
                report.kind == "stutter-not-proven-fixed-within-4-scans"
                for report in phase1.reports
            ):
                per_field["text"].add("FD-9")
                independent_fired.add("FD-9")
            if index + 1 < len(seed_cues):
                next_seed = seed_cues[index + 1]
                if float(seed["end"]) > float(next_seed["start"]):
                    per_field["end"].add("FD-3")
                    independent_fired.add("FD-3")
                if float(next_seed["start"]) - float(seed["end"]) < TWO_FRAME_S:
                    per_field["end"].add("FD-6")
                    independent_fired.add("FD-6")
            target_legs = [
                leg
                for leg in trace_legs
                if int(leg["target"]["cue_index"]) == index
                and leg["target"]["side"] in ("start", "end")
            ]
            for leg in target_legs:
                if leg["rule_id"] in ("chain", "shot-in", "shot-out") or str(
                    leg["rule_id"]
                ).startswith("ladder-"):
                    per_field[str(leg["target"]["side"])].add("FD-4")
                    independent_fired.add("FD-4")

            evidence_lyric = lyric_for_evidence(
                evidence_span_from_cue(seed), sing_spans
            )
            start, end = float(seed["start"]), float(seed["end"])
            duration = end - start
            overlap = sum(
                max(0.0, min(end, high) - max(start, low)) for low, high in sing_spans
            )
            legacy_lyric = duration > 0 and overlap / duration >= LYRIC_MIN_OVERLAP
            if evidence_lyric != legacy_lyric:
                per_field["lyric"].add("FD-2")
                independent_fired.add("FD-2")

            wanted_end = float(seed["end"])
            if phase1.speech_end is not None:
                if stream.profile.min_cue_s > 0:
                    wanted_end = max(
                        wanted_end,
                        float(seed["start"]) + stream.profile.min_cue_s,
                    )
                if stream.profile.lag_out_s > 0:
                    wanted_end = max(
                        wanted_end,
                        float(phase1.speech_end) + stream.profile.lag_out_s,
                    )
                if stream.profile.cps > 0:
                    needed = sum(not char.isspace() for char in phase1.text) / (
                        stream.profile.cps
                    )
                    wanted_end = max(
                        wanted_end,
                        min(
                            float(seed["start"]) + needed,
                            float(phase1.speech_end) + LINGER_CAP_S,
                        ),
                    )
            facts[index] = {
                "evidence_lyric": evidence_lyric,
                "expected_phase1_end": wanted_end,
                "legacy_lyric": legacy_lyric,
                "target_legs": target_legs,
                "triggers": {field: sorted(ids) for field, ids in per_field.items()},
            }
        # FD-7 is derived from the phase-1 stream, immutable seed, delivered
        # state, and trace terminal -- never from either serialized report
        # channel.  ``entries``/``refusals`` are producer output and are checked
        # for exact equality by the shared schema validator; reading either here
        # would let a serializer mutation erase both sides of the N11 check.
        has_report = any(cue.reports for cue in stream.cues)
        has_report = has_report or any(
            float(left["end"]) > float(right["start"])
            for left, right in zip(seed_cues, seed_cues[1:])
        )
        delivered = finalizer_row.get("cues") or ()
        if len(delivered) == len(stream.cues):
            for index, (row, cue) in enumerate(zip(delivered, stream.cues)):
                start, end = float(row["start"]), float(row["end"])
                if (
                    stream.profile.min_cue_s > 0
                    and end - start < stream.profile.min_cue_s - 1e-9
                ):
                    has_report = True
                if index + 1 >= len(delivered):
                    continue
                next_start = float(delivered[index + 1]["start"])
                if (
                    next_start - end < TWO_FRAME_S - 1e-9
                    and cue.speech_end is not None
                    and next_start - TWO_FRAME_S < cue.speech_end <= next_start
                ):
                    has_report = True
        trace = finalizer.get("trace") or {}
        cycle = trace.get("cycle")
        if isinstance(cycle, Mapping):
            has_report = has_report or any(
                len(set(row.get("values") or ())) > 1
                for row in cycle.get("per_boundary_values") or ()
                if isinstance(row, Mapping)
            )
        if trace.get("terminal") == "budget-exhausted":
            has_report = True
        if has_report:
            independent_fired.add("FD-7")

    left = {
        tuple(row["unit_range"]): row
        for row in finalizer_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    right = {
        tuple(row["unit_range"]): row
        for row in comparator_row.get("cues") or ()
        if row.get("unit_range") is not None
    }
    changed: list[dict[str, Any]] = []
    unclassified = 0
    relation_failures = 0
    alignment_error = set(left) != set(right)
    movement = {
        (
            int(item["boundary"]["cue_index"]),
            str(item["boundary"]["side"]),
        ): item
        for item in finalizer.get("movement") or ()
    }
    for unit_range in sorted(set(left) & set(right)):
        before, after = right[unit_range], left[unit_range]
        index = int(after["index"])
        fact = facts.get(index)
        for field in ("text", "start", "end", "lyric"):
            if before.get(field) == after.get(field):
                continue
            eligible = [] if fact is None else list(fact["triggers"][field])
            relation_ok = False
            if fact is not None and field == "lyric" and "FD-2" in eligible:
                relation_ok = (
                    bool(after.get("lyric")) == fact["evidence_lyric"]
                    and bool(before.get("lyric")) == fact["legacy_lyric"]
                )
            elif fact is not None and field == "text" and "FD-9" in eligible:
                relation_ok = str(after.get("text")) == stream.cues[index].text
            elif fact is not None and field in ("start", "end"):
                move = movement.get((index, field))
                delivered_matches = move is not None and float(
                    move["delivered"]
                ) == float(after[field])
                targeted = any(
                    leg["target"]["side"] == field for leg in fact["target_legs"]
                )
                if field == "start":
                    relation_ok = delivered_matches and trace_clean and targeted
                else:
                    phase1_matches = move is not None and float(
                        move["phase1"]
                    ) == float(fact["expected_phase1_end"])
                    relation_ok = delivered_matches and any(
                        (
                            trigger == "FD-1"
                            and phase1_matches
                            and (not targeted or trace_clean)
                        )
                        or (
                            trigger in {"FD-3", "FD-4", "FD-6"}
                            and targeted
                            and trace_clean
                        )
                        or (
                            trigger == "FD-8"
                            and phase1_matches
                            and (not targeted or trace_clean)
                        )
                        for trigger in eligible
                    )
            if not eligible:
                unclassified += 1
            elif not relation_ok:
                relation_failures += 1
            changed.append(
                {
                    "allowed_relation": relation_ok,
                    "field": field,
                    "from": before.get(field),
                    "trigger_ids": eligible,
                    "to": after.get(field),
                    "unit_range": list(unit_range),
                }
            )
    return {
        "alignment_error": alignment_error,
        "changed_fields": changed,
        "independent_fired": sorted(independent_fired),
        "producer_fired": sorted(producer_fired),
        "relation_failures": relation_failures,
        "trigger_mismatches": sorted(independent_fired ^ producer_fired),
        "unclassified_field_diff": unclassified,
    }


class _AuditedFinalizerPreview:
    """N7: compare every consumed preview with phase 1 itself, exactly.

    Wraps the finalizer's own preview for the optimizer: every scored candidate
    is re-derived through ``phase1_cue`` and compared field for field, and
    :meth:`check_selected` bridges each row's selected edges to the seed that
    row's factory actually minted.
    """

    def __init__(self, delegate: FinalizerPreview) -> None:
        self.delegate = delegate
        self.scored_edges = 0
        self.checked_edges = 0
        self.uncheckable_edges = 0
        self.mismatches: list[dict[str, Any]] = []
        self.selected_rows: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _facts(value: CuePreview) -> dict[str, Any]:
        return {
            "display_end": value.display_end,
            "display_start": value.display_start,
            "final_text": value.final_text,
            "line_count": value.line_count,
            "reading_chars": value.reading_chars,
            "refusals": [row.to_dict() for row in value.refusals],
            "waivers": [row.to_dict() for row in value.waivers],
        }

    def preview_cue(self, candidate: CueCandidate) -> CuePreview:
        from voxweave.core.finalizer import phase1_cue
        from voxweave.core.timing_preview import CuePreview

        consumed = self.delegate.preview_cue(candidate)
        edge_index = self.scored_edges
        self.scored_edges += 1
        if candidate.start is None or candidate.end is None:
            self.uncheckable_edges += 1
            return consumed
        seed: Cue = {
            "text": candidate.text,
            "start": candidate.start,
            "end": candidate.end,
            "word_data": list(candidate.word_data),
            "speech_start": candidate.speech_start,
            "speech_end": candidate.speech_end,
        }
        phase1 = phase1_cue(
            seed,
            profile=candidate.profile,
            index=0,
            expected_footprint=candidate.expected_footprint,
        )
        expected = CuePreview(
            display_start=phase1.start,
            display_end=phase1.end,
            final_text=phase1.text,
            line_count=len(phase1.lines),
            reading_chars=phase1.reading_chars,
            waivers=(),
            refusals=phase1.reports,
        )
        self.checked_edges += 1
        if consumed != expected:
            self.mismatches.append(
                {
                    "consumed": self._facts(consumed),
                    "edge_index": edge_index,
                    "phase1": self._facts(expected),
                }
            )
        return consumed

    def preview_display_span(
        self,
        start: float,
        end: float,
        next_start: float | None,
        *,
        text: str,
        word_data: Sequence[Unit],
        min_cue_s: float,
        max_cue_s: float,
        cps: float = 0.0,
        lag_out_s: float = 0.0,
    ) -> float:
        return self.delegate.preview_display_span(
            start,
            end,
            next_start,
            text=text,
            word_data=word_data,
            min_cue_s=min_cue_s,
            max_cue_s=max_cue_s,
            cps=cps,
            lag_out_s=lag_out_s,
        )

    def check_selected(self, row_id: str, solution: Any, stream: Any) -> None:
        """Bridge scored selected-edge facts to the factory's actual seed."""
        edge_facts = [
            part.features
            for interval in solution.solutions
            if interval.selection is not None
            for part in interval.selection.policy_selected.edge_breakdowns
        ]
        mismatches: list[dict[str, Any]] = []
        if len(edge_facts) != len(stream.cues):
            mismatches.append(
                {
                    "cue_count": len(stream.cues),
                    "edge_count": len(edge_facts),
                    "reason": "cardinality",
                }
            )
        for index, (facts, cue) in enumerate(zip(edge_facts, stream.cues)):
            consumed = {
                "display_end": facts.get("preview_display_end"),
                "display_start": facts.get("preview_display_start"),
                "final_text": facts.get("preview_final_text"),
                "line_count": facts.get("preview_line_count"),
                "reading_chars": facts.get("preview_reading_chars"),
                "refusal_count": facts.get("preview_refusal_count"),
            }
            phase1 = {
                "display_end": cue.end,
                "display_start": cue.start,
                "final_text": cue.text,
                "line_count": len(cue.lines),
                "reading_chars": cue.reading_chars,
                "refusal_count": len(cue.reports),
            }
            if consumed != phase1:
                mismatches.append(
                    {
                        "consumed": consumed,
                        "cue_index": index,
                        "phase1": phase1,
                        "reason": "facts",
                    }
                )
        self.selected_rows[row_id] = {
            "cue_count": len(stream.cues),
            "edge_count": len(edge_facts),
            "mismatches": mismatches,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_edges": self.checked_edges,
            "mismatches": list(self.mismatches),
            "scored_edges": self.scored_edges,
            "selected_rows": copy.deepcopy(self.selected_rows),
            "uncheckable_edges": self.uncheckable_edges,
        }


class ShadowAdmissionError(ValueError):
    """A fully assembled artifact that the schema-2 contract refuses.

    Carries the artifact and every refusal, so the error envelope can keep them
    as a diagnostic instead of discarding the whole measurement. The message
    quotes at most :data:`ADMISSION_ERRORS_SHOWN` refusals.
    """

    def __init__(self, artifact: dict[str, Any], errors: Sequence[str]) -> None:
        self.artifact = artifact
        self.errors = tuple(errors)
        shown = "; ".join(self.errors[:ADMISSION_ERRORS_SHOWN])
        hidden = len(self.errors) - ADMISSION_ERRORS_SHOWN
        if hidden > 0:
            shown += f"; ... and {hidden} more"
        super().__init__("invalid live shadow schema 2: " + shown)


def _incomplete(
    artifact: dict[str, Any], *, reason: str, detail: str
) -> dict[str, Any]:
    """The fail-open envelope for an artifact that cannot claim schema 2.

    ``reason`` is one of :data:`INCOMPLETE_REASONS`; ``detail`` is the human
    sentence. The optimizer artifact carries empty placeholder degradation
    ledgers, while the real ones ride on this envelope (``shadow_degraded``
    from :func:`run_shadow`, ``production_degraded`` from the pipeline hook), so
    the placeholders are dropped rather than left to contradict them.
    """
    if reason not in INCOMPLETE_REASONS:
        raise ValueError(f"unknown incomplete-shadow reason {reason!r}")
    artifact.pop("production_degraded", None)
    artifact.pop("shadow_degraded", None)
    return {
        "diagnostic": artifact,
        "error": {
            "detail": detail,
            "reason": reason,
            "type": "IncompleteShadowArtifact",
        },
        "kind": "segmentation-shadow-incomplete",
        "schema_version": 1,
    }


def _shadow_v2_artifact(
    document: SegDocument, v1_cues: Sequence[Cue], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the complete P5 row matrix and assemble one schema-2 artifact.

    Returns the admitted artifact, or an :func:`_incomplete` envelope when a row
    the contract needs cannot be materialized. Raises
    :class:`ShadowAdmissionError` when the assembled artifact fails the
    contract.
    """
    from voxweave.core.authority import (
        AuthorityKind,
        AuthorityLedger,
        check_roots,
        digest_payload,
        lineage_tuples,
    )
    from voxweave.core.boundary_v2 import (
        V1Partition,
        _document_partition,
        _document_waivers,
        _optimization_reuse,
        optimize_document,
        selected_evidence_spans,
    )
    from voxweave.core.finalizer import (
        FinalizeEvidence,
        FinalizePolicy,
        FinalizerPreview,
        capture_v1_reference,
        finalize,
        phase1_from_optimizer_selection,
        phase1_from_v1_capture,
        register_optimizer_selection,
    )
    from voxweave.core.layout import _join
    from voxweave.core.partition_check import owned_unit_ids
    from voxweave.core.speaker_evidence import (
        W_SPEAKER_INTERIOR,
        annotate_speaker_ids,
        evidence_span_from_cue,
        lyric_for_evidence,
        measure_speaker_events,
        named_multi_cues_unannotated,
        project_speaker_evidence,
        speaker_evidence,
    )
    from voxweave.core.subunit import empty_refine_result, refine_document
    from voxweave.pipeline import _copied_spans, mark_lyric_cues

    # Capture the committed v1 bytes before any legacy overlay. The finalizer
    # input is a separate evidence-stamped copy; the delivery tripwire retains
    # this raw reference byte for byte.
    reference: list[Cue] = [copy.deepcopy(cue) for cue in v1_cues]
    parent_speakers = speaker_evidence(document)
    finalizer_reference = [copy.deepcopy(cue) for cue in reference]
    for cue in finalizer_reference:
        lyric = lyric_for_evidence(evidence_span_from_cue(cue), document.sing_spans)
        if lyric:
            cue["lyric"] = True
        else:
            cue.pop("lyric", None)
    ledger = AuthorityLedger()
    capture = capture_v1_reference(finalizer_reference, ledger=ledger)

    # Refinement is the first v2 topology operation and acts on a detached copy.
    shadow_document, split = refine_document(document)
    projected_speakers = project_speaker_evidence(
        parent_speakers, refined_units=split.units, origin=split.origin
    )
    v1_partition, v1_projection = _shadow_v1_partition(
        document, split.origin, reference
    )
    v1_reference_input = (
        None
        if v1_partition is None
        else V1Partition(cuts=v1_partition, cues=tuple(reference))
    )
    preview = _AuditedFinalizerPreview(FinalizerPreview(shadow_document.profile))
    pricing_reuse = _optimization_reuse(shadow_document, canonical_spaced=True)
    solution = optimize_document(
        shadow_document,
        v1=v1_reference_input,
        preview=preview,
        subunit_split=split,
        speakers=projected_speakers,
        speaker_weight=W_SPEAKER_INTERIOR,
        _reuse=pricing_reuse,
    )
    speaker_off = optimize_document(
        shadow_document,
        v1=v1_reference_input,
        preview=preview,
        subunit_split=split,
        speakers=projected_speakers,
        speaker_weight=0.0,
        _reuse=pricing_reuse,
    )
    optimizer_artifact_bytes = json.dumps(
        solution.artifact, sort_keys=True, separators=(",", ":")
    )
    artifact = solution.artifact
    artifact["units"] = _shadow_source_units(shadow_document.units)
    artifact["preview_fidelity"] = preview.to_dict()
    artifact["v1_projection"] = {
        "cut_count": None if v1_partition is None else len(v1_partition),
        "mode": v1_projection,
        "unprojected": v1_partition is None,
    }
    if solution.invalid_profile:
        refused = "; ".join(
            f"{violation.key}={violation.value} ({violation.reason})"
            for violation in solution.invalid_profile
        )
        return _incomplete(
            artifact,
            reason="invalid-profile",
            detail=f"optimizer profile preflight failed: {refused}",
        )

    unit_count = len(shadow_document.units)
    raw_partition = _document_partition(solution.solutions, unit_count)
    off_partition = _document_partition(speaker_off.solutions, unit_count)
    solver_waivers = _document_waivers(solution.solutions)
    fallback_ranges = [
        list(item.unit_range) for item in solution.solutions if not item.optimized
    ]

    raw_cues = [copy.deepcopy(cue) for item in solution.solutions for cue in item.cues]
    # The raw stage is the optimizer's own stream, which ``optimize_document``
    # already checked with its per-interval waivers and attribution. Reusing
    # that check keeps ``raw.validator`` and ``validator.raw`` one answer (the
    # one ``interval_document_agree`` was computed against).
    raw_v2 = _shadow_stream_block(
        raw_cues,
        raw_partition,
        "solver-partition",
        document=shadow_document,
        origin="v2",
        stage="raw",
        validator=copy.deepcopy(artifact["validator"]["raw"]),
    )
    artifact["raw"] = raw_v2

    def owned_footprints(partition: Sequence[int] | None) -> list[str]:
        if partition is None:
            return []
        return [
            _join(
                [unit.surface for unit in shadow_document.units[low:high]],
                shadow_document.language,
            )
            for low, high in owned_unit_ids(partition, unit_count)
        ]

    core_cues = _shadow_core_cues(solution, shadow_document, thresholds)
    core_projection, core_projection_mode = _shadow_surface_partition(
        shadow_document.units, core_cues
    )
    core_v2 = _shadow_stream_block(
        core_cues,
        raw_partition,
        "solver-partition",
        document=shadow_document,
        origin="v2",
        stage="core",
        waivers=_restamp_by_footprint(solver_waivers, raw_partition, unit_count),
        origins=_origins_by_footprint(fallback_ranges, raw_partition, unit_count),
        extra={
            "projection_cross_check": {
                "agrees": core_projection == raw_partition,
                "mode": core_projection_mode,
            }
        },
    )
    core_v1 = _shadow_stream_block(
        reference,
        v1_partition,
        v1_projection,
        document=shadow_document,
        origin="v1",
        stage="core",
    )

    delivery_v2_cues = _shadow_overlay_cues(core_cues, shadow_document, thresholds)
    delivery_v2_partition, delivery_v2_mode = _shadow_surface_partition(
        shadow_document.units, delivery_v2_cues
    )
    delivery_v1_cues = _shadow_overlay_cues(reference, shadow_document, thresholds)
    delivery_v1_partition, delivery_v1_mode = _shadow_surface_partition(
        shadow_document.units, delivery_v1_cues
    )
    delivery_v2 = _shadow_stream_block(
        delivery_v2_cues,
        delivery_v2_partition,
        delivery_v2_mode,
        document=shadow_document,
        origin="v2",
        stage="legacy-overlay",
        waivers=_restamp_by_footprint(
            solver_waivers, delivery_v2_partition, unit_count
        ),
        origins=_origins_by_footprint(
            fallback_ranges, delivery_v2_partition, unit_count
        ),
    )
    delivery_v1 = _shadow_stream_block(
        delivery_v1_cues,
        delivery_v1_partition,
        delivery_v1_mode,
        document=shadow_document,
        origin="v1",
        stage="legacy-overlay",
    )

    # Typed fallbacks adopt COMPLETE v1 cues, and the optimizer's adoption plan
    # grows neighbouring fallbacks until their ranges tile. An overlap here means
    # that invariant broke: the raw-stage document validator then sees a v1 cue
    # twice, a reporting artifact rather than a conservation result, so it is
    # flagged where a reader meets it instead of being mistaken for evidence. It
    # cannot arise on the public corpus, where the C13 gate forbids fallbacks.
    overlapping = any(
        left[1] > right[0] for left, right in zip(fallback_ranges, fallback_ranges[1:])
    )
    # Every row gets one typed upstream authority and one finalizer root.
    evaluation_id = (
        "p5:"
        + digest_payload(
            {
                "language": shadow_document.language,
                "profile": artifact["profile"],
                "units": [unit.surface for unit in shadow_document.units],
            }
        )[:20]
    )
    finalizer_evidence = FinalizeEvidence(
        shots=tuple(shadow_document.shot_changes or ()),
        sing_spans=tuple(shadow_document.sing_spans or ()),
    )
    finalizer_policy = FinalizePolicy()
    unavailable = {
        "v2": [
            item.interval.index for item in solution.solutions if not item.optimized
        ],
        "v2-speaker-off": [
            item.interval.index for item in speaker_off.solutions if not item.optimized
        ],
    }
    v1_stream = phase1_from_v1_capture(
        capture,
        profile=shadow_document.profile,
        ledger=ledger,
        row_id=f"{SHADOW_LANE_FINALIZER}/v1",
        evaluation_id=evaluation_id,
    )

    def v1_rows() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """The v1 finalizer row, its legacy-display comparator, and N11 on both.

        The comparator is the exact captured input with only the legacy
        display-span lyric classification applied (set and clear): no speaker
        overlay, no resnap.
        """
        v1_finalized = finalize(
            v1_stream,
            profile=shadow_document.profile,
            evidence=finalizer_evidence,
            policy=finalizer_policy,
        )
        v1_row, _v1_final_cues = _shadow_finalizer_row(
            v1_finalized,
            v1_stream,
            v1_partition,
            document=shadow_document,
            origin="v1",
            evidence=finalizer_evidence,
            policy=finalizer_policy,
            projection=v1_projection,
        )
        comparator_cues = [copy.deepcopy(cue) for cue in capture.cues]
        for cue in comparator_cues:
            cue.pop("lyric", None)
        mark_lyric_cues(comparator_cues, _copied_spans(shadow_document.sing_spans))
        comparator_row = _shadow_stream_block(
            comparator_cues,
            v1_partition,
            v1_projection,
            document=shadow_document,
            origin="v1",
            stage="core",
        )
        _shadow_stamp_comparator_deltas(v1_row, comparator_row)
        classification = _shadow_diff_classification(
            v1_row,
            comparator_row,
            stream=v1_stream,
            seed_cues=capture.cues,
            sing_spans=tuple(shadow_document.sing_spans or ()),
        )
        return v1_row, comparator_row, classification

    def publish_validators(finalizer: dict[str, Any] | None) -> None:
        artifact["validator"]["core"] = core_v2["validator"]
        artifact["validator"]["legacy_overlay"] = delivery_v2["validator"]
        artifact["validator"]["raw_duplicate_v1_cues"] = overlapping
        artifact["validator"]["finalizer"] = finalizer

    def lanes(
        finalizer_rows: dict[str, Any], comparator_row: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            SHADOW_LANE_CORE: _shadow_lane_block(
                SHADOW_LANE_CORE, "core", core_v1, core_v2
            ),
            SHADOW_LANE_DELIVERY_LEGACY: _shadow_lane_block(
                SHADOW_LANE_DELIVERY_LEGACY,
                "legacy-overlay",
                delivery_v1,
                delivery_v2,
            ),
            SHADOW_LANE_FINALIZER: {
                "lane": SHADOW_LANE_FINALIZER,
                "rows": finalizer_rows,
                "stage": "finalizer",
            },
            SHADOW_LANE_LEGACY_DISPLAY: {
                "lane": SHADOW_LANE_LEGACY_DISPLAY,
                "rows": {"v1": comparator_row},
                "stage": "legacy-display",
            },
        }

    def authorities(expected: dict[str, AuthorityKind]) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in ledger.events],
            "expected": expected,
            "lineage": [list(record) for record in lineage_tuples(ledger)],
            "violations": list(check_roots(ledger, expected=expected)),
        }

    if any(unavailable.values()):
        # The frozen optimizer factory correctly refuses an adopted-v1 interval:
        # it has no optimizer selection to seal. Preserve the useful core/legacy
        # diagnostics and state the unmaterialized rows instead of letting that
        # typed precondition collapse the whole fail-open artifact to ``error``.
        v1_row, comparator_row, diff_classification = v1_rows()
        actual_fallbacks = sum(not item.optimized for item in solution.solutions)
        optimized_units = sum(
            item.interval.unit_end - item.interval.unit_start
            for item in solution.solutions
            if item.optimized
        )
        artifact["coverage"] = {
            **artifact["coverage"],
            "coarse_granularity_intervals": sum(
                item.lattice.infeasible is not None
                and item.lattice.infeasible.reason == "coarse-granularity"
                for item in solution.solutions
            ),
            "fallback_intervals": actual_fallbacks,
            "fallback_ranges_overlap": overlapping,
            "fallback_unit_ranges": fallback_ranges,
            "optimized_intervals": len(solution.solutions) - actual_fallbacks,
            "optimized_unit_ratio": (
                1.0 if unit_count == 0 else optimized_units / unit_count
            ),
            "raw_conservation_trustworthy": not overlapping,
            "unit_count": unit_count,
            "v1_unprojected": v1_partition is None,
        }
        publish_validators(None)
        unmaterialized = {
            "materialized": False,
            "reason": "adopted-v1-has-no-optimizer-authority",
        }
        artifact["lanes"] = lanes(
            {
                "v1": v1_row,
                "v2": dict(unmaterialized),
                "v2-speaker-off": dict(unmaterialized),
            },
            comparator_row,
        )
        artifact["authorities"] = authorities(
            {f"{SHADOW_LANE_FINALIZER}/v1": "v1-capture"}
        )
        artifact["diff_classification"] = diff_classification
        artifact["canonical_fallback_rechecks"] = _shadow_fallback_rechecks(
            v1_stream,
            owned_footprints(v1_partition),
            row_id="v1",
        )
        artifact["finalizer"] = None
        artifact["invalid_optimizer_rows"] = unavailable
        artifact["refiner_comparison"] = {
            "materialized": False,
            "reason": "optimizer-selection-unavailable",
            "refined_parent_count": split.refined_parent_count,
            "status": "unmaterialized",
        }
        artifact["speaker_evidence"]["measurement_refusal"] = (
            "optimizer-selection-unavailable"
        )
        artifact["preview_fidelity"] = preview.to_dict()
        return _incomplete(
            artifact,
            reason="optimizer-selection-unavailable",
            detail="optimizer selection authority unavailable for one or more rows",
        )

    on_authority = register_optimizer_selection(solution, ledger=ledger)
    on_stream = phase1_from_optimizer_selection(
        on_authority,
        ledger=ledger,
        row_id=f"{SHADOW_LANE_FINALIZER}/v2",
        evaluation_id=evaluation_id,
    )
    off_authority = register_optimizer_selection(speaker_off, ledger=ledger)
    off_stream = phase1_from_optimizer_selection(
        off_authority,
        ledger=ledger,
        row_id=f"{SHADOW_LANE_FINALIZER}/v2-speaker-off",
        evaluation_id=evaluation_id,
    )
    preview.check_selected("v2", solution, on_stream)
    preview.check_selected("v2-speaker-off", speaker_off, off_stream)
    canonical_fallback_rechecks = [
        *_shadow_fallback_rechecks(
            v1_stream,
            owned_footprints(v1_partition),
            row_id="v1",
        ),
        *_shadow_fallback_rechecks(
            on_stream,
            owned_footprints(raw_partition),
            row_id="v2",
        ),
        *_shadow_fallback_rechecks(
            off_stream,
            owned_footprints(off_partition),
            row_id="v2-speaker-off",
        ),
    ]
    v1_row, comparator_row, diff_classification = v1_rows()
    on_finalized = finalize(
        on_stream,
        profile=shadow_document.profile,
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    off_finalized = finalize(
        off_stream,
        profile=shadow_document.profile,
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    on_row, on_final_cues = _shadow_finalizer_row(
        on_finalized,
        on_stream,
        raw_partition,
        document=shadow_document,
        origin="v2",
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )
    off_row, off_final_cues = _shadow_finalizer_row(
        off_finalized,
        off_stream,
        off_partition,
        document=shadow_document,
        origin="v2",
        evidence=finalizer_evidence,
        policy=finalizer_policy,
    )

    # Speaker measurement uses one selected evidence basis for both independent
    # global boundary matching runs. Budget-invalid rows short-circuit here.
    measurement_refusal: str | None = None
    row_cardinality_ok = len(raw_partition) + 1 == len(on_final_cues) and len(
        off_partition
    ) + 1 == len(off_final_cues)
    if on_finalized.valid and off_finalized.valid and not row_cardinality_ok:
        measurement_refusal = "cue/range-cardinality-mismatch"
    elif on_finalized.valid and off_finalized.valid:
        try:
            evidence_spans = selected_evidence_spans(solution)
        except ValueError as exc:
            measurement_refusal = str(exc)
        else:
            on_measurement = measure_speaker_events(
                projected_speakers,
                evidence_spans=evidence_spans,
                delivered_boundaries=_shadow_boundary_times(on_final_cues),
                off_boundaries=_shadow_boundary_times(off_final_cues),
            )
            off_measurement = measure_speaker_events(
                projected_speakers,
                evidence_spans=evidence_spans,
                delivered_boundaries=_shadow_boundary_times(off_final_cues),
            )
            for name, measured in (
                ("v2", on_measurement),
                ("v2-speaker-off", off_measurement),
            ):
                if (
                    sum(measured.buckets.values())
                    != measured.raw_in_speech_turn_changes
                ):
                    raise ValueError(f"{name} speaker bucket conservation failed")
            artifact["speaker_evidence"]["measurement"] = on_measurement.to_dict()
            artifact["speaker_evidence"]["off_row_measurement"] = (
                off_measurement.to_dict()
            )
            off_row["speaker_measurement"] = off_measurement.to_dict()
            ranges = owned_unit_ids(raw_partition, unit_count)
            annotate_speaker_ids(
                on_final_cues, ranges, projected_speakers.unit_speakers
            )
            projected_named_multi = named_multi_cues_unannotated(
                ranges, projected_speakers.unit_speakers
            )
            if projected_named_multi != int(
                artifact["coverage"]["named_multi_cues_unannotated"]
            ):
                raise ValueError(
                    "speaker projection counter disagrees with selected ownership"
                )
            artifact["speaker_evidence"]["projection"] = {
                "cue_count": len(on_final_cues),
                "named_multi_cues_unannotated": projected_named_multi,
                "range_count": len(ranges),
                "status": "verified",
            }
            on_row["cues"] = _shadow_cue_rows(on_final_cues, raw_partition, unit_count)
    artifact["speaker_evidence"]["measurement_refusal"] = measurement_refusal

    # Refiner bypass is an exact identity gate on tracked rows and a typed
    # diagnostic on genuinely refined rows. Only a fallback-free bypass can own
    # an optimizer authority/finalizer root under the frozen W1 API.
    parent_v1_cuts, parent_v1_mode = _shadow_surface_partition(
        document.units, reference
    )
    parent_v1 = (
        None
        if parent_v1_cuts is None
        else V1Partition(parent_v1_cuts, tuple(reference))
    )
    refiner_off = optimize_document(
        document,
        v1=parent_v1,
        preview=preview,
        subunit_split=empty_refine_result(document.units, language=document.language),
        speakers=parent_speakers,
        speaker_weight=W_SPEAKER_INTERIOR,
    )
    if split.refined_parent_count == 0:
        refiner_comparison: dict[str, Any] = {
            "byte_identical": optimizer_artifact_bytes
            == json.dumps(refiner_off.artifact, sort_keys=True, separators=(",", ":")),
            "refined_parent_count": 0,
            "status": "tracked-identity",
        }
    else:
        off_fallbacks = sum(not item.optimized for item in refiner_off.solutions)
        refiner_off_partition = _document_partition(
            refiner_off.solutions, len(document.units)
        )
        coarse_ranges = [
            list(item.unit_range)
            for item in refiner_off.solutions
            if item.coarse_caused
        ]
        mapped_on: set[int] = set()
        internal_on: list[int] = []
        for cut in raw_partition:
            left_parent = split.origin[cut - 1]
            right_parent = split.origin[cut]
            if left_parent == right_parent:
                internal_on.append(left_parent)
            else:
                mapped_on.add(right_parent)
        external_diff = sorted(mapped_on ^ set(refiner_off_partition))

        def covered(parent: int) -> bool:
            return any(low <= parent < high for low, high in coarse_ranges)

        diffs_confined = all(covered(parent) for parent in internal_on) and all(
            covered(max(0, cut - 1)) or covered(cut) for cut in external_diff
        )
        refiner_comparison = {
            "byte_identical": False,
            "coarse_caused_intervals": refiner_off.artifact["coverage"][
                "coarse_caused_intervals"
            ],
            "coarse_caused_unit_ranges": coarse_ranges,
            "diffs_confined_to_coarse_caused": diffs_confined,
            "external_parent_cut_diff": external_diff,
            "fallback_intervals": off_fallbacks,
            "internal_refinement_cut_parents": sorted(internal_on),
            "materialized": off_fallbacks == 0,
            "off_partition": list(refiner_off_partition),
            "on_parent_edge_partition": sorted(mapped_on),
            "parent_v1_projection": parent_v1_mode,
            "refined_parent_count": split.refined_parent_count,
            "status": "refined-counterfactual",
        }

    rows: dict[str, Any] = {
        "v1": v1_row,
        "v2": on_row,
        "v2-speaker-off": off_row,
    }
    expected: dict[str, AuthorityKind] = {
        f"{SHADOW_LANE_FINALIZER}/v1": "v1-capture",
        f"{SHADOW_LANE_FINALIZER}/v2": "optimizer-selection",
        f"{SHADOW_LANE_FINALIZER}/v2-speaker-off": "optimizer-selection",
    }
    if split.refined_parent_count and refiner_comparison["materialized"]:
        # The counterfactual is solved over the parent units, but every row of
        # the artifact is read against the refined ``units`` block. A parent
        # cut is a refined cut at the parent's first child, and each parent's
        # children join back to its surface, so the row keeps its cue text.
        refiner_partition = _refined_partition(
            _document_partition(refiner_off.solutions, len(document.units)),
            split.origin,
        )
        refiner_authority = register_optimizer_selection(refiner_off, ledger=ledger)
        refiner_stream = phase1_from_optimizer_selection(
            refiner_authority,
            ledger=ledger,
            row_id=f"{SHADOW_LANE_FINALIZER}/refiner-off",
            evaluation_id=evaluation_id,
        )
        preview.check_selected("refiner-off", refiner_off, refiner_stream)
        refiner_evidence = FinalizeEvidence(
            shots=tuple(document.shot_changes or ()),
            sing_spans=tuple(document.sing_spans or ()),
        )
        refiner_finalized = finalize(
            refiner_stream,
            profile=document.profile,
            evidence=refiner_evidence,
            policy=finalizer_policy,
        )
        refiner_row, _refiner_cues = _shadow_finalizer_row(
            refiner_finalized,
            refiner_stream,
            refiner_partition,
            document=shadow_document,
            origin="v2",
            evidence=refiner_evidence,
            policy=finalizer_policy,
        )
        rows["refiner-off"] = refiner_row
        expected[f"{SHADOW_LANE_FINALIZER}/refiner-off"] = "optimizer-selection"
        canonical_fallback_rechecks.extend(
            _shadow_fallback_rechecks(
                refiner_stream,
                owned_footprints(refiner_partition),
                row_id="refiner-off",
            )
        )

    totals = artifact["totals"]
    artifact["coverage"] = {
        **artifact["coverage"],
        "coarse_granularity_intervals": totals["coarse_granularity_intervals"],
        "fallback_intervals": totals["fallback_intervals"],
        "fallback_ranges_overlap": overlapping,
        "fallback_unit_ranges": fallback_ranges,
        "optimized_intervals": totals["optimized_intervals"],
        "optimized_unit_ratio": totals["optimized_unit_ratio"],
        "raw_conservation_trustworthy": not overlapping,
        "unit_count": totals["unit_count"],
        "v1_unprojected": v1_partition is None,
    }
    publish_validators(on_row["validator"])
    artifact["finalizer"] = on_row["finalizer"]
    artifact["lanes"] = lanes(rows, comparator_row)
    artifact["authorities"] = authorities(expected)
    artifact["diff_classification"] = diff_classification
    artifact["canonical_fallback_rechecks"] = canonical_fallback_rechecks
    artifact["preview_fidelity"] = preview.to_dict()
    artifact["refiner_comparison"] = refiner_comparison
    artifact["invalid_finalizer_rows"] = [
        name for name, row in rows.items() if not row["finalizer"]["valid"]
    ]
    if v1_partition is None:
        return _incomplete(
            artifact,
            reason="v1-unprojected",
            detail="v1 source partition could not be projected",
        )
    if (
        split.refined_parent_count
        and refiner_comparison.get("materialized") is not True
    ):
        return _incomplete(
            artifact,
            reason="refiner-off-unavailable",
            detail="refiner-off optimizer selection authority unavailable",
        )
    # The optimizer artifact remains schema 1 until this exact completed
    # payload passes the one shared live/harness contract.  Validation ignores
    # only the version field for this pre-admission pass; every other top-level
    # key, lane/row, evidence block, and cross-block cardinality is live.
    from voxweave.core.shadow_schema import (
        LIVE_SHADOW_SCHEMA_VERSION,
        validate_shadow_v2_payload,
    )

    if artifact.get("schema_version") != 1:
        raise ValueError("live shadow admission requires a schema-1 optimizer payload")
    errors = validate_shadow_v2_payload(artifact, require_version=False)
    if errors:
        raise ShadowAdmissionError(artifact, errors)
    artifact["schema_version"] = LIVE_SHADOW_SCHEMA_VERSION
    return artifact


def run_shadow(
    document: SegDocument, cues: Sequence[Cue], *, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure BoundaryOptimizer v2 beside the shipped v1 answer.

    ``pipeline._maybe_shadow_v2`` has already read :data:`SEG_V2_SHADOW_ENV` and
    only enters here when it is on, so the optimizer is imported only after the
    flag passes: an off run costs one environment read and a branch and never
    pulls a v2 module into the process at all.

    The shadow opens its OWN nested degradation capture. ``note_degraded``
    aggregates by ``(slot, reason)`` and bumps a count on repeats, so a shadow
    that re-tokenizes inside production's capture would change the persisted
    manifest -- breaking the very contract this lane exists to respect. Nesting
    restores the outer capture on exit, so production's ledger never sees a
    shadow event. The capture is also ``quiet``: the one-shot warning log is
    process-global, and a measurement that re-runs the same providers would
    otherwise win that latch and silence the line the shipping run owed its
    operator.

    Nothing here may fail the run: a measurement that can crash the pipeline is
    worse than no measurement, so an unexpected error is recorded as a typed
    ``error`` block and the shipped cues are returned untouched. An artifact the
    schema-2 contract refuses keeps its assembled payload as ``diagnostic`` and
    every refusal as ``admission_errors``.
    """
    with degradation_capture(quiet=True) as shadow_degraded:
        try:
            artifact = _shadow_v2_artifact(document, cues, thresholds)
            # AD4-4: the shadow's own ledger, collected at hook time and
            # deliberately kept out of the persisted manifest. Copied off the
            # live list so a later capture cannot append to published evidence.
            artifact["shadow_degraded"] = list(shadow_degraded)
            from voxweave.core.shadow_schema import (
                LIVE_SHADOW_SCHEMA_VERSION,
                validate_shadow_v2_payload,
            )

            if artifact.get("schema_version") == LIVE_SHADOW_SCHEMA_VERSION:
                # Validated again as published: the ledger above was attached
                # after the assembler's own admission, and the contract closes
                # over it as well.
                errors = validate_shadow_v2_payload(artifact)
                if errors:
                    raise ShadowAdmissionError(artifact, errors)
        except Exception as exc:  # noqa: BLE001 - a measurement never fails the run
            detail = str(exc)
            shown = (
                detail
                if len(detail) <= LOG_DETAIL_CHARS
                else detail[:LOG_DETAIL_CHARS] + " ..."
            )
            log.warning(
                "v2 shadow lane failed; shipped output is unaffected (%s)", shown
            )
            log.debug("v2 shadow lane failure", exc_info=True)
            artifact = {
                "error": {"detail": detail, "type": type(exc).__name__},
                "kind": "segmentation-shadow-error",
                "schema_version": 1,
                "shadow_degraded": list(shadow_degraded),
            }
            if isinstance(exc, ShadowAdmissionError):
                # The envelope carries the degradation ledgers; the refused
                # artifact's own copies are placeholders or duplicates.
                exc.artifact.pop("production_degraded", None)
                exc.artifact.pop("shadow_degraded", None)
                artifact["admission_errors"] = list(exc.errors)
                artifact["diagnostic"] = exc.artifact
    return artifact
