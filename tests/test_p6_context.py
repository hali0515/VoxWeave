from pathlib import Path

import pytest


def _stable_fields():
    from voxweave.align_snapshot import FrozenObject, freeze_json

    stable_fields = freeze_json(
        {
            "vtt_sha256": "a" * 64,
            "sibling_json_sha256": "b" * 64,
            "media_fingerprint": "c" * 64,
            "prepared_audio_sha256": "d" * 64,
        }
    )
    assert isinstance(stable_fields, FrozenObject)
    return stable_fields


def _issue(tmp_path: Path):
    from voxweave.align_context import issue_align_context

    return issue_align_context(
        stable_fields=_stable_fields(),
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )


def test_align_context_issues_exact_distinct_roles_and_registry_family(tmp_path):
    from voxweave.align_context import ALIGN_ROLE_ORDER, role_vector

    context = _issue(tmp_path)
    assert context.engine_family == "legacy-v1"
    assert context.effective_iso == "en"
    assert context.route_kind == "ctc-full"
    assert ALIGN_ROLE_ORDER == (
        "acquisition",
        "adapter",
        "encoder",
        "evidence-bind",
        "commit",
    )
    assert role_vector(context) == ("L", "L", "L", "L", "L")


def test_roles_are_single_use_scoped_and_terminal_exactly_once(tmp_path):
    from voxweave.align_context import (
        ContextAuthorityError,
        consume_context_role,
        retire_live_context_roles,
        role_events,
        role_vector,
    )

    context = _issue(tmp_path)
    consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
    with pytest.raises(ContextAuthorityError) as reused:
        consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
    assert reused.value.detail_code == "context-consumed"
    retire_live_context_roles(context)
    assert role_vector(context) == ("C", "R", "R", "R", "R")
    assert [(event.role, event.terminal) for event in role_events(context)] == [
        ("acquisition", "consumed"),
        ("adapter", "retired"),
        ("encoder", "retired"),
        ("evidence-bind", "retired"),
        ("commit", "retired"),
    ]
    retire_live_context_roles(context)
    assert len(role_events(context)) == 5


def test_wrong_consumer_and_wrong_role_are_rejected_without_consumption(tmp_path):
    from voxweave.align_context import (
        ContextAuthorityError,
        consume_context_role,
        role_vector,
    )

    context = _issue(tmp_path)
    with pytest.raises(ContextAuthorityError) as wrong_consumer:
        consume_context_role(context, "adapter", consumer="encode_align_candidates")
    assert wrong_consumer.value.detail_code == "context-role"
    with pytest.raises(ContextAuthorityError) as wrong_role:
        consume_context_role(context, "unknown", consumer="unknown")
    assert wrong_role.value.detail_code == "context-role"
    assert role_vector(context) == ("L", "L", "L", "L", "L")


def test_private_path_binding_rejects_target_swap_but_stable_digest_relocates(tmp_path):
    from voxweave.align_context import ContextAuthorityError, verify_context_binding

    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first = _issue(first_root)
    second = _issue(second_root)
    assert first.context_content_digest == second.context_content_digest
    assert first.context_binding_digest != second.context_binding_digest
    verify_context_binding(
        first,
        target_path=first_root / "episode.vtt",
        sibling_path=first_root / "episode.json",
        media_path=first_root / "episode.mkv",
    )
    with pytest.raises(ContextAuthorityError) as swapped:
        verify_context_binding(
            first,
            target_path=second_root / "episode.vtt",
            sibling_path=second_root / "episode.json",
            media_path=second_root / "episode.mkv",
        )
    assert swapped.value.detail_code == "context-binding"


def test_hand_built_context_and_cross_context_binding_are_unissued(tmp_path):
    from voxweave.align_context import (
        ContextAuthorityError,
        IssuedAlignContext,
        consume_context_role,
        issue_align_context,
    )

    forged = object.__new__(IssuedAlignContext)
    object.__setattr__(forged, "context_content_digest", "0" * 64)
    object.__setattr__(forged, "context_binding_digest", "1" * 64)
    object.__setattr__(forged, "engine_family", "legacy-v1")
    object.__setattr__(forged, "effective_iso", "en")
    object.__setattr__(forged, "route_kind", "ctc-full")
    with pytest.raises(ContextAuthorityError) as unissued:
        consume_context_role(forged, "acquisition", consumer="FreshAlignmentIssuer")
    assert unissued.value.detail_code == "context-unissued"

    foreign = issue_align_context(
        stable_fields=_stable_fields(),
        target_path=tmp_path / "foreign.vtt",
        sibling_path=tmp_path / "foreign.json",
        media_path=tmp_path / "foreign.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )
    assert foreign.context_binding_digest != _issue(tmp_path).context_binding_digest


def test_segmentation_context_has_only_adapter_encoder_commit_roles(tmp_path):
    from voxweave.align_context import (
        consume_context_role,
        issue_segmentation_context,
        retire_live_context_roles,
        role_vector,
    )

    context = issue_segmentation_context(
        stable_fields=_stable_fields(),
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        effective_iso="ja",
    )
    assert role_vector(context) == ("L", "L", "L")
    consume_context_role(context, "adapter", consumer="run_locked_segmentation_adapter")
    retire_live_context_roles(context)
    assert role_vector(context) == ("C", "R", "R")


def test_context_closure_classifies_unused_roles_and_expected_vtt_generation(
    tmp_path,
):
    from voxweave.align_context import (
        ContextAuthorityError,
        issue_align_context,
        verify_context_expected_vtt_generation,
        verify_context_roles_terminal,
    )
    from voxweave.align_snapshot import FrozenObject, freeze_json

    stable_fields = freeze_json(
        {
            "expected_vtt_sha256": "a" * 64,
            "vtt_generation": {"sha256": "b" * 64},
        }
    )
    assert isinstance(stable_fields, FrozenObject)
    context = issue_align_context(
        stable_fields=stable_fields,
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )

    with pytest.raises(ContextAuthorityError) as unused:
        verify_context_roles_terminal(context)
    assert unused.value.failure.kind == "context-authority-invalid"
    assert unused.value.failure.detail_code == "context-unused-role"

    with pytest.raises(ContextAuthorityError) as generation:
        verify_context_expected_vtt_generation(
            context,
            observed_vtt_sha256="b" * 64,
        )
    assert generation.value.failure.kind == "context-authority-invalid"
    assert generation.value.failure.detail_code == "expected-vtt-generation"
