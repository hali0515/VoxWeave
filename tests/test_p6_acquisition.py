import math

import pytest


def test_strict_capture_is_all_or_none_and_assigns_ids_before_recursive_freeze():
    from voxweave.align_acquisition import capture_strict_units

    good = {"text": "a", "start": 0.0, "end": 1.0, "extra": {"x": 1}}
    cycle = {"text": "b", "start": 1.0, "end": 2.0}
    cycle["extra"] = cycle
    result = capture_strict_units(
        (good, cycle), call_index=3, raw_unit_ids=("r7", "r8")
    )
    assert result.status == "invalid"
    assert result.units is None
    assert result.raw_units_digest is None
    assert result.normalized_relative_digest is None
    assert result.failure is not None
    assert result.failure.stage == "strict-capture"
    assert result.failure.call_unit_index == 1
    assert result.observed_unit_ids == ("r7", "r8")


def test_strict_capture_preserves_original_bounds_and_interpolation_provenance():
    from voxweave.align_acquisition import capture_strict_units

    current = ({"text": "a", "start": 0.5, "end": 0.5, "extra": True},)
    original = ({"text": "a", "start": None, "end": None, "extra": True},)
    result = capture_strict_units(
        current,
        original_units=original,
        call_index=0,
        raw_unit_ids=("r0",),
    )
    assert result.status == "valid"
    assert result.units is not None
    unit = result.units[0]
    assert unit.provenance == "align-interpolated"
    assert unit.original_relative_start is None
    assert unit.original_relative_end is None
    assert unit.relative_start == unit.relative_end == 0.5


@pytest.mark.parametrize(
    ("node", "index"),
    [
        ({"start": 0.0, "end": 1.0}, 0),
        ({"text": 1, "start": 0.0, "end": 1.0}, 0),
        ({"text": "a", "start": True, "end": 1.0}, 0),
    ],
)
def test_strict_capture_rejects_the_closed_unit_domain(node, index):
    from voxweave.align_acquisition import capture_strict_units

    result = capture_strict_units((node,), call_index=0, raw_unit_ids=("r0",))
    assert result.status == "invalid"
    assert result.failure is not None
    assert result.failure.call_unit_index == index


def test_physical_transform_uses_identity_without_plus_zero():
    from voxweave.align_acquisition import capture_strict_units, transform_strict_units

    capture = capture_strict_units(
        ({"text": "a", "start": -0.0, "end": 1.0},),
        call_index=0,
        raw_unit_ids=("r0",),
    )
    transformed = transform_strict_units(
        capture, physical_origin_seconds=0.0, identity=True
    )
    assert transformed.status == "valid"
    assert transformed.units is not None
    assert math.copysign(1.0, transformed.units[0].start) == -1.0


def test_qwen_nominal_and_sample_origins_remain_distinct():
    from voxweave.align_acquisition import qwen_sample_geometry

    geometry = qwen_sample_geometry(
        nominal_start=0.10001,
        nominal_end=1.0,
        sample_rate=16_000,
        sample_count=32_000,
    )
    assert geometry.sample_start == int(0.10001 * 16_000)
    assert geometry.physical_origin_seconds == geometry.sample_start / 16_000
    assert geometry.legacy_origin_seconds == 0.10001
    assert geometry.authority_origin_seconds == geometry.physical_origin_seconds
    assert geometry.legacy_origin_seconds != geometry.physical_origin_seconds


def test_qwen_geometry_clamps_negative_start_and_media_end():
    from voxweave.align_acquisition import qwen_sample_geometry

    geometry = qwen_sample_geometry(
        nominal_start=-3.0,
        nominal_end=99.0,
        sample_rate=10,
        sample_count=20,
    )
    assert (geometry.sample_start, geometry.sample_end) == (0, 20)
    assert geometry.physical_origin_seconds == 0.0
    assert geometry.legacy_origin_seconds == -3.0


def test_transform_failure_retains_complete_capture_and_exact_locator():
    from voxweave.align_acquisition import capture_strict_units, transform_strict_units

    capture = capture_strict_units(
        ({"text": "a", "start": float("inf"), "end": 1.0},),
        call_index=2,
        raw_unit_ids=("r4",),
    )
    assert capture.status == "valid"
    transformed = transform_strict_units(
        capture, physical_origin_seconds=3.0, identity=False
    )
    assert transformed.status == "invalid"
    assert transformed.units is None
    assert transformed.failure is not None
    assert transformed.failure.stage == "authority-transform"
    assert transformed.failure.call_unit_index == 0
    assert transformed.failure.detail_code == "authority-recompute"
    assert capture.units is not None


def _issued_context(tmp_path):
    from voxweave.align_context import consume_context_role, issue_align_context
    from voxweave.align_snapshot import freeze_json

    stable = freeze_json({"input": "x"})
    context = issue_align_context(
        stable_fields=stable,
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )
    consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
    return context


def test_seal_mismatch_appends_exactly_one_private_terminal_before_disposal_and_raise(
    tmp_path,
):
    from voxweave.align_acquisition import (
        AcquisitionAdmissionLedger,
        FreshSealBroken,
        raise_distribution_seal_mismatch,
    )
    from voxweave.align_context import retire_live_context_roles, role_vector

    context = _issued_context(tmp_path)
    ledger = AcquisitionAdmissionLedger()
    order = []
    with pytest.raises(FreshSealBroken) as error:
        raise_distribution_seal_mismatch(
            context,
            terminal_call_index=4,
            ledger=ledger,
            dispose=lambda: order.append("disposed"),
        )
    assert order == ["disposed"]
    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event.terminal == "acquisition-failed"
    assert len(event.subject) == 2
    assert event.payload == (
        "AO-13",
        "fresh-seal-broken",
        "authority-distribution",
        "distribution-seal",
        4,
        (),
    )
    assert error.value.failure.secondary == ()
    retire_live_context_roles(context)
    assert role_vector(context) == ("C", "R", "R", "R", "R")
    with pytest.raises(FreshSealBroken):
        raise_distribution_seal_mismatch(
            context,
            terminal_call_index=4,
            ledger=ledger,
            dispose=lambda: None,
        )
    assert len(ledger.events) == 1


def test_disposal_failure_is_secondary_without_mutating_private_terminal(tmp_path):
    from voxweave.align_acquisition import (
        AcquisitionAdmissionLedger,
        FreshSealBroken,
        raise_distribution_seal_mismatch,
    )

    context = _issued_context(tmp_path)
    ledger = AcquisitionAdmissionLedger()

    def fail_dispose():
        raise OSError("injected disposal failure")

    with pytest.raises(FreshSealBroken) as error:
        raise_distribution_seal_mismatch(
            context,
            terminal_call_index=1,
            ledger=ledger,
            dispose=fail_dispose,
        )
    assert ledger.events[0].payload[-1] == ()
    assert len(error.value.failure.secondary) == 1
    assert error.value.failure.secondary[0].detail_code == "stage-residue"
