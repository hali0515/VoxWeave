"""Independent oracles of the BoundaryOptimizer v2 shadow lane.

Two checks live here, each deliberately self-contained so that neither can be
satisfied by the code it audits: :func:`_shadow_diff_classification` (N11)
recomputes per-cue triggers and allowed relations from the phase-1 stream, and
:class:`_AuditedFinalizerPreview` (N7) re-derives every consumed preview
through ``phase1_cue``. They share no helper with each other and never import
finalizer producer code at module scope. Both moved here verbatim from
:mod:`voxweave.core.shadow_v2`, which re-exports them.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from voxweave.core.overlay import LYRIC_MIN_OVERLAP
from voxweave.core.schema import Cue, Unit

if TYPE_CHECKING:  # the finalizer modules stay lazy: they load only with the flag on
    from voxweave.core.finalizer import FinalizerPreview
    from voxweave.core.timing_preview import CueCandidate, CuePreview


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
