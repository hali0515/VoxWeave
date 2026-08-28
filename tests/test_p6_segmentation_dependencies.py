import inspect


def test_segmentation_adapter_never_reads_pipeline_or_shadow_artifact_output():
    import voxweave.segmentation_adapter as module

    source = inspect.getsource(module)
    for forbidden in (
        "pipeline",
        "shadow_schema",
        "_shadow_v2_artifact",
        "_cleanup_cues",
        "_snap_to_shots",
        "mark_lyric_cues",
        "apply_speaker_format",
    ):
        assert forbidden not in source


def test_segmentation_projector_has_no_writer_or_transaction_dependency():
    import voxweave.segmentation_projector as module

    source = inspect.getsource(module)
    for forbidden in ("pipeline", "fsio", "atomic_write", "episode_lock"):
        assert forbidden not in source


def test_reference_projector_does_not_import_segmentation_producer():
    import voxweave.reference_projector as module

    source = inspect.getsource(module)
    assert "segmentation_projector" not in source
    assert "segmentation_candidates" not in source
