import pytest


EXPECTED_KINDS = (
    "subtitle-snapshot-failed",
    "align-input-decode-invalid",
    "v2-input-invalid",
    "carrier-extraction-failed",
    "media-identity-invalid",
    "media-snapshot-unavailable",
    "semantic-backend-unavailable",
    "qwen-route-invalid",
    "qwen-window-operation-failed",
    "dp-route-hints-invalid",
    "cache-lock-failed",
    "cache-companion-invalid",
    "cache-operation-failed",
    "context-authority-invalid",
    "legacy-time-transform-failed",
    "fresh-backend-output-invalid",
    "fresh-time-transform-invalid",
    "fresh-distribution-invalid",
    "fresh-reconciliation-invalid",
    "fresh-seed-invalid",
    "v2-policy-invalid",
    "fresh-authority-invalid",
    "fresh-seal-broken",
    "profile-invalid",
    "evidence-invalid",
    "segmentation-v2-invalid",
    "finalizer-budget-exhausted",
    "finalizer-output-invalid",
    "finalizer-partition-failed",
    "finalizer-trace-failed",
    "finalizer-stability-failed",
    "align-delta-invalid",
    "no-aligned-units",
    "selected-render-invalid",
    "final-evidence-invalid",
    "shadow-internal-error",
    "shadow-artifact-unavailable",
    "preencode-failed",
    "stage-failed",
    "episode-lock-failed",
    "input-stale",
    "media-stale",
    "commit-failed",
    "artifact-cleanup-failed",
    "model-release-failed",
    "observer-failed",
    "snapshot-dispose-failed",
)


def test_failure_and_reason_registries_are_exact_and_closed():
    from voxweave.align_failures import (
        AUTHORITY_REASON_ORDER,
        OUTCOME_KIND_ORDER,
        RATIFICATION_DORMANT_DETAILS,
        SEED_REASON_ORDER,
    )

    assert OUTCOME_KIND_ORDER == EXPECTED_KINDS
    assert AUTHORITY_REASON_ORDER == (
        "partial-empty-ownership",
        "punctuation-only-block",
        "authority-transform-invalid",
        "route-owner-mismatch",
        "allocation-no-tiling",
        "allocation-ambiguous",
        "allocation-budget-exhausted",
    )
    assert SEED_REASON_ORDER == AUTHORITY_REASON_ORDER + (
        "absolute-bound-invalid",
        "absolute-order-invalid",
        "display-seed-invalid",
        "footprint-reconciliation",
    )
    assert RATIFICATION_DORMANT_DETAILS == ()


def test_every_outcome_has_a_nonempty_closed_detail_tuple():
    from voxweave.align_failures import OUTCOME_DETAILS

    assert tuple(OUTCOME_DETAILS) == EXPECTED_KINDS
    assert all(details for details in OUTCOME_DETAILS.values())
    assert OUTCOME_DETAILS["stage-failed"] == (
        "main-json-stage",
        "vtt-stage",
        "evidence-stage",
        "machine-artifact-stage",
    )
    assert OUTCOME_DETAILS["commit-failed"] == (
        "main-json-replace",
        "vtt-replace",
        "evidence-replace",
        "machine-artifact-replace",
    )
    assert (
        len({detail for details in OUTCOME_DETAILS.values() for detail in details}) > 40
    )


def test_canonical_failure_validates_parent_detail_and_secondary_shape():
    from voxweave.align_failures import (
        CanonicalFailure,
        FailureRegistryError,
        SecondaryFailure,
    )

    failure = CanonicalFailure(
        kind="fresh-distribution-invalid",
        phase="authority-distribution",
        detail_code="allocation-budget",
        secondary=(
            SecondaryFailure(
                kind="snapshot-dispose-failed",
                phase="dispose",
                detail_code="stage-residue",
            ),
        ),
    )
    assert failure.to_dict() == {
        "kind": "fresh-distribution-invalid",
        "phase": "authority-distribution",
        "detail_code": "allocation-budget",
        "secondary": [
            {
                "kind": "snapshot-dispose-failed",
                "phase": "dispose",
                "detail_code": "stage-residue",
            }
        ],
    }
    with pytest.raises(FailureRegistryError):
        CanonicalFailure(
            kind="fresh-distribution-invalid",
            phase="authority-distribution",
            detail_code="not-registered",
        )
    with pytest.raises(FailureRegistryError):
        SecondaryFailure(
            kind="not-a-kind", phase="dispose", detail_code="stage-residue"
        )


def test_rat7_detail_is_registered_and_active():
    from voxweave.align_failures import OUTCOME_DETAILS, is_detail_dormant

    assert "speaker-mapping-generation" in OUTCOME_DETAILS["input-stale"]
    assert is_detail_dormant("input-stale", "speaker-mapping-generation") is False
    assert is_detail_dormant("input-stale", "vtt-generation") is False
