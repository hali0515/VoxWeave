"""Legacy count-only slice distribution for the selected parity lane."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from voxweave.core.langsets import ALIGNER_NO_SPACE_LANGS
from voxweave.timestamps import shift_units


@dataclass(frozen=True)
class LegacyCallDistributionReceipt:
    owner_source_indices: tuple[int, ...]
    expected_counts: tuple[int, ...]
    requested_ranges: tuple[tuple[int, int], ...]
    realized_ranges: tuple[tuple[int, int], ...]
    owner_unit_ids: tuple[tuple[str, ...], ...]
    final_cursor: int
    consumed_prefix_unit_ids: tuple[str, ...]
    shortage_source_indices: tuple[int, ...]
    leftover_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class LegacyDistributionResult:
    block_units: tuple[tuple[Any, ...], ...]
    receipt: LegacyCallDistributionReceipt
    relative_block_units: tuple[tuple[Any, ...], ...]


def _legacy_count(text: str, iso: str) -> int:
    """Owner size exactly as ``align_common._distribute_units`` counts it.

    The legacy lane mirrors that slicer, so it uses the aligners' own no-space set
    (``realign.NO_SPACE_LANGS``), not the wider layout set: a th/lo/my full pass is
    sliced by words there, and counting characters here would disagree with the
    delivered units.
    """
    stripped = (text or "").strip()
    if iso in ALIGNER_NO_SPACE_LANGS:
        return sum(1 for character in stripped if character.isalnum())
    return len(stripped.split())


def legacy_distribute_before_shift(
    relative_flat: Sequence[Any],
    *,
    texts: tuple[str, ...],
    iso: str,
    origin: float,
    identity: bool,
    raw_unit_ids: tuple[str, ...],
    source_indices: tuple[int, ...] | None = None,
) -> LegacyDistributionResult:
    """Freeze count-only slices, then transform only their retained members."""
    if len(raw_unit_ids) != len(relative_flat):
        raise ValueError("raw unit ids must cover the complete flat result")
    owner_sources = (
        tuple(range(len(texts))) if source_indices is None else tuple(source_indices)
    )
    if (
        len(owner_sources) != len(texts)
        or len(set(owner_sources)) != len(owner_sources)
        or any(type(index) is not int or index < 0 for index in owner_sources)
    ):
        raise ValueError("legacy source indices must be unique nonnegative integers")
    requested: list[tuple[int, int]] = []
    realized: list[tuple[int, int]] = []
    expected: list[int] = []
    owner_ids: list[tuple[str, ...]] = []
    retained: list[Sequence[Any]] = []
    shortages: list[int] = []
    cursor = 0
    raw_count = len(relative_flat)
    for delivery_index, text in enumerate(texts):
        count = _legacy_count(text, iso)
        lower, upper = cursor, cursor + count
        requested.append((lower, upper))
        realized_lower = min(lower, raw_count)
        realized_upper = min(upper, raw_count)
        realized.append((realized_lower, realized_upper))
        expected.append(count)
        retained.append(relative_flat[lower:upper])
        owner_ids.append(raw_unit_ids[lower:upper])
        if realized_upper - realized_lower < count:
            shortages.append(owner_sources[delivery_index])
        cursor += count
    relative_projected = tuple(
        tuple(
            {
                "text": unit["text"],
                "start": unit["start"],
                "end": unit["end"],
            }
            for unit in group
        )
        for group in retained
    )
    if identity:
        projected = tuple(
            tuple(dict(unit) for unit in group) for group in relative_projected
        )
    else:
        projected = tuple(
            tuple(shift_units(list(group), origin)) for group in relative_projected
        )
    consumed = min(cursor, raw_count)
    receipt = LegacyCallDistributionReceipt(
        owner_source_indices=owner_sources,
        expected_counts=tuple(expected),
        requested_ranges=tuple(requested),
        realized_ranges=tuple(realized),
        owner_unit_ids=tuple(owner_ids),
        final_cursor=cursor,
        consumed_prefix_unit_ids=raw_unit_ids[:consumed],
        shortage_source_indices=tuple(shortages),
        leftover_unit_ids=raw_unit_ids[consumed:],
    )
    return LegacyDistributionResult(projected, receipt, relative_projected)


def legacy_retain_qwen_before_shift(
    relative_flat: Sequence[Any],
    *,
    origin: float,
    identity: bool,
    raw_unit_ids: tuple[str, ...],
    source_indices: tuple[int, ...],
) -> LegacyDistributionResult:
    """Assign one Qwen call's complete raw result to its sole cue owner."""
    if len(raw_unit_ids) != len(relative_flat):
        raise ValueError("raw unit ids must cover the complete flat result")
    if (
        len(source_indices) != 1
        or type(source_indices[0]) is not int
        or source_indices[0] < 0
    ):
        raise ValueError("a Qwen call must have one nonnegative source owner")
    relative_owner = tuple(
        {
            "text": unit["text"],
            "start": unit["start"],
            "end": unit["end"],
        }
        for unit in relative_flat
    )
    if identity:
        owner = tuple(dict(unit) for unit in relative_owner)
    else:
        owner = tuple(shift_units(list(relative_owner), origin))
    raw_count = len(relative_flat)
    receipt = LegacyCallDistributionReceipt(
        owner_source_indices=source_indices,
        expected_counts=(raw_count,),
        requested_ranges=((0, raw_count),),
        realized_ranges=((0, raw_count),),
        owner_unit_ids=(raw_unit_ids,),
        final_cursor=raw_count,
        consumed_prefix_unit_ids=raw_unit_ids,
        shortage_source_indices=(),
        leftover_unit_ids=(),
    )
    return LegacyDistributionResult((owner,), receipt, (relative_owner,))
