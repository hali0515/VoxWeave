"""Fresh acquisition seals are digested once, yet still catch swapped fields."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, cast

import pytest


def _sealed_acquisition(tmp_path):
    """Seal one two-block ctc-full acquisition whose phase-1 seed is valid."""
    from voxweave import align_acquisition
    from voxweave.align_acquisition import (
        _fresh_alignment_call_observer,
        begin_fresh_alignment,
        seal_fresh_alignment,
    )
    from voxweave.align_context import issue_align_context
    from voxweave.align_snapshot import FrozenObject, freeze_json

    stable_fields = freeze_json({"input": "seal-digest"})
    assert isinstance(stable_fields, FrozenObject)
    context = issue_align_context(
        stable_fields=stable_fields,
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )
    session = begin_fresh_alignment(
        context,
        alignment_texts=("alpha bravo", "charlie delta"),
        source_indices=(0, 1),
        language="en",
        prepared_audio_sample_count=64_000,
    )
    units = tuple(
        {"text": word, "start": 0.5 * index, "end": 0.5 * index + 0.4}
        for index, word in enumerate(("alpha", "bravo", "charlie", "delta"))
    )
    _fresh_alignment_call_observer(session)(units, None, (0, 1), 0.0)
    acquisition = seal_fresh_alignment(session)
    record = align_acquisition._FRESH[id(acquisition)]
    assert cast(Any, record.seed).status == "valid"
    return context, acquisition, record


def _count_digests(monkeypatch):
    from voxweave import align_acquisition

    calls: list[object] = []
    original = align_acquisition._stable_digest

    def counting(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(align_acquisition, "_stable_digest", counting)
    return calls


def test_repeated_fresh_record_access_does_not_redigest_the_seals(
    tmp_path, monkeypatch
):
    from voxweave.align_acquisition import (
        _fresh_core_inputs,
        _fresh_record,
        _fresh_seed,
    )

    calls = _count_digests(monkeypatch)
    context, acquisition, record = _sealed_acquisition(tmp_path)

    def phase1_digests() -> int:
        return sum(
            1
            for value in calls
            if isinstance(value, tuple) and value and value[0] is record.seed
        )

    assert phase1_digests() == 1
    sealed = len(calls)
    for _ in range(8):
        assert _fresh_record(context, acquisition) is record
        _fresh_core_inputs(context, acquisition)
        _fresh_seed(context, acquisition)
    assert len(calls) == sealed
    assert phase1_digests() == 1


def test_swapped_but_equal_field_is_redigested_once_then_trusted(tmp_path, monkeypatch):
    from voxweave.align_acquisition import _fresh_core_inputs

    calls = _count_digests(monkeypatch)
    context, acquisition, record = _sealed_acquisition(tmp_path)
    _fresh_core_inputs(context, acquisition)
    record.seed = copy.deepcopy(record.seed)
    before = len(calls)
    _fresh_core_inputs(context, acquisition)
    redigested = calls[before:]
    assert len(redigested) == 1
    assert cast(tuple, redigested[0])[0] is record.seed
    after = len(calls)
    _fresh_core_inputs(context, acquisition)
    assert len(calls) == after


def _tamper(detail_code, context, acquisition, record):
    if detail_code == "context-seal":
        object.__setattr__(context, "route_kind", "mms-full")
    elif detail_code == "raw-seal":
        record.captures = (
            replace(record.captures[0], raw_units_digest="0" * 64),
            *record.captures[1:],
        )
    elif detail_code == "relative-seal":
        record.captures = (
            replace(record.captures[0], normalized_relative_digest="0" * 64),
            *record.captures[1:],
        )
    elif detail_code == "legacy-slice-seal":
        record.legacy_receipts = (
            replace(record.legacy_receipts[0], final_cursor=99),
            *record.legacy_receipts[1:],
        )
    elif detail_code == "authority-seal":
        record.transforms = (
            replace(record.transforms[0], authority_absolute_digest="0" * 64),
            *record.transforms[1:],
        )
    elif detail_code == "distribution-seal":
        record.distribution = replace(record.distribution, consumed_count=99)
    elif detail_code == "issued-distribution-seal":
        object.__setattr__(
            acquisition,
            "distribution",
            replace(acquisition.distribution, consumed_count=99),
        )
    else:
        record.seed = replace(
            cast(Any, record.seed), reasons=("absolute-bound-invalid",)
        )


@pytest.mark.parametrize(
    ("tamper", "detail_code"),
    (
        ("context-seal", "context-seal"),
        ("raw-seal", "raw-seal"),
        ("relative-seal", "relative-seal"),
        ("legacy-slice-seal", "legacy-slice-seal"),
        ("authority-seal", "authority-seal"),
        ("distribution-seal", "distribution-seal"),
        ("issued-distribution-seal", "distribution-seal"),
        ("phase1-seal", "phase1-seal"),
    ),
)
def test_tampering_after_first_access_is_still_detected(tmp_path, tamper, detail_code):
    from voxweave.align_acquisition import FreshSealBroken, _fresh_core_inputs

    context, acquisition, record = _sealed_acquisition(tmp_path)
    for _ in range(2):
        _fresh_core_inputs(context, acquisition)
    _tamper(tamper, context, acquisition, record)
    for _ in range(2):
        with pytest.raises(FreshSealBroken) as caught:
            _fresh_core_inputs(context, acquisition)
        assert caught.value.failure.kind == "fresh-seal-broken"
        assert caught.value.failure.detail_code == detail_code
