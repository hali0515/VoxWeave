import inspect


def test_all_ratifications_remain_pending_with_closed_defaults():
    from voxweave.p6_ratifications import RATIFICATION_DEFAULTS

    assert tuple(RATIFICATION_DEFAULTS) == tuple(
        f"RAT-{index}" for index in range(1, 8)
    )
    assert all(
        decision.status == "pending" for decision in RATIFICATION_DEFAULTS.values()
    )
    assert RATIFICATION_DEFAULTS["RAT-1"].enabled_operation is None
    assert RATIFICATION_DEFAULTS["RAT-5"].enabled_operation is None
    assert RATIFICATION_DEFAULTS["RAT-7"].default == "j0-only"


def test_pending_rat1_does_not_amend_or_call_timeline_finalizer():
    from voxweave.core import authority, finalizer
    from voxweave.core import align_seed

    assert authority.AUTHORITY_KINDS == ("optimizer-selection", "v1-capture")
    assert "fresh-alignment" not in authority.AUTHORITY_KINDS
    assert not hasattr(finalizer, "phase1_from_fresh_alignment")
    source = inspect.getsource(align_seed)
    assert "phase1_stream(" not in source
    assert "finalize(" not in source
    assert "capture_v1_reference" not in source


def test_pending_rat5_and_rat7_defaults_are_not_runtime_switches():
    from voxweave.p6_ratifications import (
        QWEN_SELECTED_V2_ENABLED,
        SPEAKER_MAPPING_CAS_ENABLED,
    )

    assert QWEN_SELECTED_V2_ENABLED is False
    assert SPEAKER_MAPPING_CAS_ENABLED is False
