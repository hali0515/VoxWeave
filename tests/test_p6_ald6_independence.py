from __future__ import annotations

import dataclasses

import pytest


def _issued_facts(tmp_path):
    from voxweave.align_acquisition import (
        _fresh_alignment_call_observer,
        _fresh_core_inputs,
        begin_fresh_alignment,
        seal_fresh_alignment,
    )
    from voxweave.align_context import issue_align_context
    from voxweave.align_snapshot import FrozenObject, freeze_json

    stable = freeze_json({"input": "ald6-independent-digest-probe"})
    assert isinstance(stable, FrozenObject)
    context = issue_align_context(
        stable_fields=stable,
        target_path=tmp_path / "episode.vtt",
        sibling_path=tmp_path / "episode.json",
        media_path=tmp_path / "episode.mkv",
        effective_iso="en",
        route_kind="ctc-full",
    )
    session = begin_fresh_alignment(
        context,
        alignment_texts=("word",),
        source_indices=(0,),
        language="en",
        prepared_audio_sample_count=16_000,
    )
    _fresh_alignment_call_observer(session)(
        ({"text": "word", "start": 0.2, "end": 0.8},),
        None,
        (0,),
        0.0,
    )
    acquisition = seal_fresh_alignment(session)
    return context, acquisition, _fresh_core_inputs(context, acquisition)


@pytest.mark.parametrize(
    "field",
    (
        "legacy_slice_digest",
        "legacy_absolute_digest",
        "backend_model_config_digest",
        "route_input_digest",
    ),
)
def test_ald6_reference_rejects_alternate_legacy_model_and_route_digests(
    tmp_path,
    field,
):
    from voxweave.align_evidence_core import (
        EvidenceCoreProjectionError,
        project_evidence_core,
    )

    context, acquisition, inputs = _issued_facts(tmp_path)
    corrupt_call = dataclasses.replace(inputs[6][0], **{field: "d" * 64})
    with pytest.raises(EvidenceCoreProjectionError, match="digest|cross-link"):
        project_evidence_core(
            context_content_digest=context.context_content_digest,
            blocks=inputs[0],
            captures=inputs[1],
            transforms=inputs[2],
            distribution=inputs[3],
            seed_status=inputs[4],
            seed_reasons=inputs[5],
            physical_calls=(corrupt_call,),
            receipt_digest=acquisition.receipt_digest,
            language="en",
        )
