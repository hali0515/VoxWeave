"""Pure absolute-authority seed and footprint construction for P6 align.

The result is acquisition evidence while RAT-1 remains pending.  This module
does not mint a TimelineFinalizer root, call a P5 factory, or launder the fresh
receipt through either existing authority kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from voxweave import realign
from voxweave.align_acquisition import FreshUnit
from voxweave.align_distribution import AuthorityBlock, AuthorityDistributionReceipt
from voxweave.align_failures import SEED_REASON_ORDER
from voxweave.core.langsets import LANGUAGES_WITHOUT_SPACES
from voxweave.core.partition_check import normalize_text
from voxweave.core.schema import Unit
from voxweave.core.segdoc import SourceUnit
from voxweave.core.smart_split import _display_chars, _surface_ranges
from voxweave.core.subunit import speech_span_units


@dataclass(frozen=True)
class AlignSeedBlock:
    source_index: int
    text: str
    owner_unit_ids: tuple[str, ...]
    unit_range: tuple[int, int]
    units: tuple[FreshUnit, ...]
    speech_start: float | None
    speech_end: float | None
    display_start: float
    display_end: float
    footprint: str


@dataclass(frozen=True)
class AlignSeedResult:
    status: Literal["valid", "invalid"]
    reasons: tuple[str, ...]
    blocks: tuple[AlignSeedBlock, ...] | None
    ordered_units: tuple[FreshUnit, ...]


def _ordered_reasons(present: set[str]) -> tuple[str, ...]:
    return tuple(reason for reason in SEED_REASON_ORDER if reason in present)


def _exact_bound(value: object) -> bool:
    return value is None or (
        type(value) is float and math.isfinite(value) and value >= 0.0
    )


def _language_join(surfaces: tuple[str, ...], iso: str) -> str:
    separator = "" if iso in LANGUAGES_WITHOUT_SPACES else " "
    return separator.join(surfaces)


def _complete_footprint(block_text: str, owned: tuple[FreshUnit, ...]) -> bool:
    word_data: list[Unit] = [
        {"text": unit.surface, "start": unit.start, "end": unit.end} for unit in owned
    ]
    ranges = _surface_ranges([block_text], word_data)
    if ranges is None or len(ranges) != 1:
        return False
    lower, upper = ranges[0]
    visible = _display_chars([unit.surface for unit in owned])
    if lower > 0 and all(not value for value in visible[:lower]):
        lower = 0
    if upper < len(owned) and all(not value for value in visible[upper:]):
        upper = len(owned)
    return (lower, upper) == (0, len(owned))


def _anchors(owned: tuple[FreshUnit, ...]) -> tuple[float | None, float | None]:
    projected = tuple(
        SourceUnit(
            unit.unit_id,
            unit.surface,
            unit.start,
            unit.end,
            unit.provenance,
            None,
        )
        for unit in owned
    )
    return speech_span_units(projected)


def build_align_seed(
    *,
    blocks: tuple[AuthorityBlock, ...],
    units: tuple[FreshUnit, ...],
    distribution: AuthorityDistributionReceipt,
    iso: str,
) -> AlignSeedResult:
    """Recompute absolute bounds, reconciliation, anchors, and display seed."""
    if type(iso) is not str or not iso:
        raise ValueError("seed language must be one exact nonempty string")
    if distribution.work.status == "seal-mismatch":
        raise ValueError("a transient distribution seal mismatch cannot form a seed")
    if distribution.status != "valid":
        return AlignSeedResult(
            "invalid", _ordered_reasons(set(distribution.reasons)), None, ()
        )
    if distribution.owners is None or distribution.expected_counts is None:
        raise ValueError("valid authority distribution lacks owners")

    block_by_source = {block.source_index: block for block in blocks}
    if len(block_by_source) != len(blocks) or any(
        type(block.source_index) is not int or type(block.alignment_text) is not str
        for block in blocks
    ):
        raise ValueError("seed blocks have an invalid source domain")
    unit_by_id = {unit.unit_id: unit for unit in units}
    if len(unit_by_id) != len(units):
        raise ValueError("fresh unit ids must be globally unique")

    owner_units: list[tuple[FreshUnit, ...]] = []
    observed_ids: list[str] = []
    for source_index, owner_ids in zip(
        distribution.owner_source_indices, distribution.owners
    ):
        if source_index not in block_by_source or not owner_ids:
            raise ValueError("authority owner names no nonempty seed block")
        try:
            owned = tuple(unit_by_id[unit_id] for unit_id in owner_ids)
        except KeyError as exc:
            raise ValueError("authority owner names an unknown fresh unit") from exc
        owner_units.append(owned)
        observed_ids.extend(owner_ids)
    if (
        len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != set(unit_by_id)
        or distribution.consumed_count != len(units)
        or distribution.leftovers
    ):
        raise ValueError("authority owners do not cover fresh units exactly once")

    ordered_units = tuple(unit for owner in owner_units for unit in owner)
    reasons: set[str] = set()
    for unit in ordered_units:
        if not _exact_bound(unit.start) or not _exact_bound(unit.end):
            reasons.add("absolute-bound-invalid")
        elif unit.start is not None and unit.end is not None and unit.start > unit.end:
            reasons.add("absolute-bound-invalid")
    starts = [unit.start for unit in ordered_units if unit.start is not None]
    ends = [unit.end for unit in ordered_units if unit.end is not None]
    if any(later < earlier for earlier, later in zip(starts, starts[1:])) or any(
        later < earlier for earlier, later in zip(ends, ends[1:])
    ):
        reasons.add("absolute-order-invalid")

    anchor_pairs: list[tuple[float, float] | None] = []
    anchors: list[tuple[float | None, float | None]] = []
    for source_index, owned in zip(distribution.owner_source_indices, owner_units):
        block = block_by_source[source_index]
        surfaces = tuple(unit.surface for unit in owned)
        if (
            not normalize_text(block.alignment_text)
            or normalize_text(_language_join(surfaces, iso))
            != normalize_text(block.alignment_text)
            or not _complete_footprint(block.alignment_text, owned)
        ):
            reasons.add("footprint-reconciliation")
        speech_start, speech_end = _anchors(owned)
        anchors.append((speech_start, speech_end))
        anchor_pairs.append(
            (speech_start, speech_end)
            if speech_start is not None and speech_end is not None
            else None
        )
    if reasons:
        return AlignSeedResult(
            "invalid", _ordered_reasons(reasons), None, ordered_units
        )

    fabricated = realign.fill_insert_blocks(
        anchor_pairs, gap_sec=realign.GAP_SEC, default_dur=2.0
    )
    seed_blocks: list[AlignSeedBlock] = []
    cursor = 0
    for source_index, owned, anchor, fabricated_pair in zip(
        distribution.owner_source_indices, owner_units, anchors, fabricated
    ):
        speech_start, speech_end = anchor
        fabricated_start, fabricated_end = fabricated_pair
        if speech_start is not None and speech_end is not None:
            display_start = speech_start
            display_end = (
                speech_start + 0.05 if speech_start == speech_end else speech_end
            )
        elif speech_start is not None:
            display_start = speech_start
            display_end = max(fabricated_end, speech_start + 0.05)
        elif speech_end is not None:
            display_start = max(0.0, min(fabricated_start, speech_end - 0.05))
            display_end = speech_end
        else:
            display_start, display_end = fabricated_pair
        if (
            not math.isfinite(display_start)
            or not math.isfinite(display_end)
            or display_start < 0.0
            or display_start >= display_end
        ):
            reasons.add("display-seed-invalid")
            continue
        block = block_by_source[source_index]
        unit_range = (cursor, cursor + len(owned))
        cursor = unit_range[1]
        seed_blocks.append(
            AlignSeedBlock(
                source_index,
                block.alignment_text,
                tuple(unit.unit_id for unit in owned),
                unit_range,
                owned,
                speech_start,
                speech_end,
                display_start,
                display_end,
                block.alignment_text,
            )
        )
    if reasons:
        return AlignSeedResult(
            "invalid", _ordered_reasons(reasons), None, ordered_units
        )
    return AlignSeedResult("valid", (), tuple(seed_blocks), ordered_units)


def materialize_seed_cues(seed: AlignSeedResult) -> list[dict[str, Any]]:
    """Return a fresh mutable cue/word-data copy without minting W1 authority."""
    if seed.status != "valid" or seed.blocks is None:
        raise ValueError("only a valid align seed can be materialized")
    return [
        {
            "text": block.text,
            "start": block.display_start,
            "end": block.display_end,
            "word_data": [
                {"text": unit.surface, "start": unit.start, "end": unit.end}
                for unit in block.units
            ],
            "speech_start": block.speech_start,
            "speech_end": block.speech_end,
            "lyric": None,
            "unit_range": block.unit_range,
        }
        for block in seed.blocks
    ]


__all__ = [
    "AlignSeedBlock",
    "AlignSeedResult",
    "build_align_seed",
    "materialize_seed_cues",
]
