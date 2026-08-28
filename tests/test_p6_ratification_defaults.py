import inspect


def test_ratification_record_is_closed_and_approved():
    from voxweave.p6_ratifications import RATIFICATION_DEFAULTS

    assert tuple(RATIFICATION_DEFAULTS) == tuple(
        f"RAT-{index}" for index in range(1, 8)
    )
    assert all(
        decision.status == "approved" for decision in RATIFICATION_DEFAULTS.values()
    )
    assert RATIFICATION_DEFAULTS["RAT-1"].enabled_operation == "fresh-alignment"
    assert RATIFICATION_DEFAULTS["RAT-5"].enabled_operation == "qwen-physical-origin"
    assert RATIFICATION_DEFAULTS["RAT-7"].enabled_operation == "split-j0-s0-cas"


def test_ratified_rat1_adds_only_the_closed_fresh_alignment_factory():
    from voxweave.core import authority, finalizer
    from voxweave.core import align_seed

    assert authority.AUTHORITY_KINDS == (
        "fresh-alignment",
        "optimizer-selection",
        "v1-capture",
    )
    assert hasattr(finalizer, "phase1_from_fresh_alignment")
    source = inspect.getsource(align_seed)
    assert "phase1_stream(" not in source
    assert "finalize(" not in source
    assert "capture_v1_reference" not in source


def test_ratified_dependent_operations_are_source_constants_not_runtime_switches():
    from voxweave.p6_ratifications import (
        DURABLE_ALIGN_EVIDENCE_ENABLED,
        FRESH_ALIGNMENT_W1_ENABLED,
        LEXICAL_FULL_PASS_DELTA_ENABLED,
        QWEN_SELECTED_V2_ENABLED,
        RAW_SPEAKER_TURNS_WRITER_ENABLED,
        SPEAKER_MAPPING_CAS_ENABLED,
    )

    assert FRESH_ALIGNMENT_W1_ENABLED is True
    assert DURABLE_ALIGN_EVIDENCE_ENABLED is True
    assert RAW_SPEAKER_TURNS_WRITER_ENABLED is True
    assert LEXICAL_FULL_PASS_DELTA_ENABLED is True
    assert QWEN_SELECTED_V2_ENABLED is True
    assert SPEAKER_MAPPING_CAS_ENABLED is True
