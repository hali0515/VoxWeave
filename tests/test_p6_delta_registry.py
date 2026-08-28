import hashlib
import importlib
import sys


EXPECTED_BYTES = (
    b'[{"id":"ALD-0","title":"Qwen physical origin","ratification":"RAT-5"},'
    b'{"id":"ALD-1","title":"canonical text/layout","ratification":null},'
    b'{"id":"ALD-2","title":"duration desire","ratification":null},'
    b'{"id":"ALD-3","title":"fabricated display side","ratification":null},'
    b'{"id":"ALD-4","title":"W1 phase-2 movement","ratification":null},'
    b'{"id":"ALD-5","title":"EvidenceSpan lyric","ratification":null},'
    b'{"id":"ALD-6","title":"durable acquisition core","ratification":null}]\n'
)


def test_align_delta_registry_is_closed_and_has_pinned_canonical_bytes():
    from voxweave.align_delta_registry import (
        ALIGN_DELTA_IDS,
        ALIGN_DELTA_REGISTRY,
        canonical_align_delta_registry_bytes,
    )

    assert ALIGN_DELTA_IDS == tuple(f"ALD-{index}" for index in range(7))
    assert tuple(ALIGN_DELTA_REGISTRY) == ALIGN_DELTA_IDS
    assert canonical_align_delta_registry_bytes() == EXPECTED_BYTES
    assert len(hashlib.sha256(EXPECTED_BYTES).hexdigest()) == 64
    assert ALIGN_DELTA_REGISTRY["ALD-0"].ratification == "RAT-5"
    assert ALIGN_DELTA_REGISTRY["ALD-6"].phase == "mandatory-core"


def test_registry_import_is_dependency_neutral():
    sys.modules.pop("voxweave.align_delta_registry", None)
    before = set(sys.modules)
    importlib.import_module("voxweave.align_delta_registry")
    loaded = set(sys.modules) - before
    forbidden = {
        "voxweave.pipeline",
        "voxweave.core.finalizer",
        "voxweave.core.align_compare",
        "voxweave.align_evidence_core",
    }
    assert loaded.isdisjoint(forbidden)


def test_semantic_gate_is_available_after_rat1_approval():
    from voxweave.core.align_compare import semantic_comparison_available

    assert semantic_comparison_available() is True
