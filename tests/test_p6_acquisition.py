import math
from dataclasses import replace
from typing import Any, Literal, cast

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
    assert transformed.units[0].start is not None
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
    from voxweave.align_snapshot import FrozenObject, freeze_json

    stable = freeze_json({"input": "x"})
    assert isinstance(stable, FrozenObject)
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


def _fresh_session(
    tmp_path,
    *,
    route_kind: Literal["ctc-full", "mms-full", "qwen-crop"] = "ctc-full",
):
    from voxweave.align_acquisition import begin_fresh_alignment
    from voxweave.align_context import issue_align_context
    from voxweave.align_snapshot import FrozenObject, freeze_json

    stable_fields = freeze_json({"input": "registry-closure"})
    assert isinstance(stable_fields, FrozenObject)
    context = issue_align_context(
        stable_fields=stable_fields,
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind=route_kind,
    )
    session = begin_fresh_alignment(
        context,
        alignment_texts=("word",),
        source_indices=(0,),
        language="en",
        prepared_audio_sample_count=32_000,
    )
    return context, session


@pytest.mark.parametrize(
    ("raw_unit", "detail_code", "message"),
    (
        ({"start": 0.0, "end": 1.0}, "retained-unit-text", "'text'"),
        ({"text": "word", "end": 1.0}, "retained-unit-start", "'start'"),
        ({"text": "word", "start": 0.0}, "retained-unit-end", "'end'"),
        (
            {"text": "word", "start": "bad", "end": 1.0},
            "retained-unit-operand",
            'can only concatenate str (not "float") to str',
        ),
    ),
)
def test_legacy_time_transform_preserves_exception_and_attaches_exact_failure(
    tmp_path,
    raw_unit,
    detail_code,
    message,
):
    from voxweave.align_acquisition import (
        _fresh_alignment_call_observer,
        seal_fresh_alignment,
    )

    _context, session = _fresh_session(tmp_path)
    _fresh_alignment_call_observer(session)(
        (raw_unit,),
        None,
        (0,),
        1.0,
        audio_sample_start=16_000,
        audio_sample_end=32_000,
        sample_rate=16_000,
        sample_count=32_000,
    )
    expected_class = KeyError if detail_code != "retained-unit-operand" else TypeError
    with pytest.raises(expected_class) as caught:
        seal_fresh_alignment(session)
    assert str(caught.value) == message
    failure = getattr(caught.value, "failure")
    assert failure.kind == "legacy-time-transform-failed"
    assert failure.detail_code == detail_code


@pytest.mark.parametrize(
    "detail_code",
    ("backend-call-shape", "backend-raised", "relative-normalization"),
)
def test_fresh_backend_output_preserves_exception_and_attaches_exact_failure(
    tmp_path,
    detail_code,
):
    from voxweave.align_acquisition import (
        _fresh_alignment_backend_invoker,
        _fresh_alignment_call_observer,
        seal_fresh_alignment,
    )

    _context, session = _fresh_session(tmp_path)
    if detail_code == "backend-call-shape":
        with pytest.raises(
            TypeError, match="backend result must be one sized sequence"
        ) as caught:
            _fresh_alignment_call_observer(session)(object(), None, (0,), 0.0)
    elif detail_code == "backend-raised":

        def fail_backend():
            raise RuntimeError("injected backend failure")

        with pytest.raises(RuntimeError, match="injected backend failure") as caught:
            _fresh_alignment_backend_invoker(session)(fail_backend)
    else:
        _fresh_alignment_call_observer(session)(
            ({"text": "word", "start": 0.0, "end": 1.0},),
            (),
            (0,),
            0.0,
        )
        with pytest.raises(
            ValueError,
            match="original_units must match the current raw result length",
        ) as caught:
            seal_fresh_alignment(session)
    failure = getattr(caught.value, "failure")
    assert failure.kind == "fresh-backend-output-invalid"
    assert failure.detail_code == detail_code


def test_backend_raised_invoker_is_wired_to_both_live_full_pass_leaves(
    tmp_path,
    monkeypatch,
):
    import inspect
    from types import SimpleNamespace

    from voxweave import align_ctc, align_mms, pipeline
    from voxweave.align_acquisition import _fresh_alignment_backend_invoker

    def fail_backend(*_args, **_kwargs):
        raise RuntimeError("leaf backend failure")

    monkeypatch.setattr(align_ctc, "_ctc_emit_full", fail_backend)
    monkeypatch.setattr(align_mms, "_mms_emit_units", fail_backend)

    _context, ctc_session = _fresh_session(tmp_path)
    ctc_invoker = _fresh_alignment_backend_invoker(ctc_session)
    with pytest.raises(RuntimeError, match="leaf backend failure") as ctc_caught:
        align_ctc._ctc_full_pass(
            SimpleNamespace(invocab={}),
            object(),
            ["word"],
            False,
            "en",
            _backend_invoker=ctc_invoker,
        )
    ctc_failure = getattr(ctc_caught.value, "failure")
    assert ctc_failure.kind == "fresh-backend-output-invalid"
    assert ctc_failure.detail_code == "backend-raised"

    _context, mms_session = _fresh_session(tmp_path)
    mms_invoker = _fresh_alignment_backend_invoker(mms_session)
    with pytest.raises(RuntimeError, match="leaf backend failure") as mms_caught:
        align_mms._mms_full_pass(
            object(),
            ["word"],
            "en",
            _backend_invoker=mms_invoker,
        )
    mms_failure = getattr(mms_caught.value, "failure")
    assert mms_failure.kind == "fresh-backend-output-invalid"
    assert mms_failure.detail_code == "backend-raised"

    source = inspect.getsource(pipeline._align_blocks)
    assert source.count("_backend_invoker=backend_invoker") == 2


@pytest.mark.parametrize(
    "detail_code",
    (
        "context-seal",
        "raw-seal",
        "relative-seal",
        "legacy-slice-seal",
        "authority-seal",
        "phase1-seal",
    ),
)
def test_fresh_acquisition_component_seals_fail_with_exact_detail(
    tmp_path,
    detail_code,
):
    from voxweave import align_acquisition
    from voxweave.align_acquisition import (
        FreshSealBroken,
        _fresh_alignment_call_observer,
        _fresh_core_inputs,
        seal_fresh_alignment,
    )

    context, session = _fresh_session(tmp_path)
    _fresh_alignment_call_observer(session)(
        ({"text": "word", "start": 0.0, "end": 1.0},),
        None,
        (0,),
        0.0,
    )
    acquisition = seal_fresh_alignment(session)
    record = align_acquisition._FRESH[id(acquisition)]
    if detail_code == "context-seal":
        object.__setattr__(context, "route_kind", "mms-full")
    elif detail_code == "raw-seal":
        record.captures = (replace(record.captures[0], raw_units_digest="0" * 64),)
    elif detail_code == "relative-seal":
        record.captures = (
            replace(record.captures[0], normalized_relative_digest="0" * 64),
        )
    elif detail_code == "legacy-slice-seal":
        record.legacy_receipts = (replace(record.legacy_receipts[0], final_cursor=99),)
    elif detail_code == "authority-seal":
        record.transforms = (
            replace(record.transforms[0], authority_absolute_digest="0" * 64),
        )
    else:
        record.seed = replace(
            cast(Any, record.seed), reasons=("absolute-bound-invalid",)
        )

    with pytest.raises(FreshSealBroken) as caught:
        _fresh_core_inputs(context, acquisition)
    assert caught.value.failure.kind == "fresh-seal-broken"
    assert caught.value.failure.detail_code == detail_code
