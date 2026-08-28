import inspect


def test_reference_projector_does_not_import_producer_encoder_or_writer():
    import voxweave.reference_projector as module

    source = inspect.getsource(module)
    for forbidden in (
        "align_projector",
        "candidate_encoder",
        "pipeline",
        "fsio",
        "atomic_write",
    ):
        assert forbidden not in source


def test_evidence_binder_cannot_reproject_the_mandatory_core():
    import voxweave.align_evidence as module

    source = inspect.getsource(module)
    assert "project_evidence_core" not in source
    assert "project_align_delivery" not in source
    assert "render_cues" not in source
