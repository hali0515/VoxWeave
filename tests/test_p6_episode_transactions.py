from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from voxweave import pipeline, sdh, songdet


def _vtt(text: str = "hello", *, settings: str = "") -> str:
    suffix = f" {settings}" if settings else ""
    return f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000{suffix}\n{text}\n"


def _units(text: str = "hello") -> list[dict[str, object]]:
    return [{"text": text, "start": 0.0, "end": 1.0}]


def test_align_exposes_only_the_two_permitted_private_transaction_seams():
    parameters = inspect.signature(pipeline.align).parameters
    assert tuple(parameters)[-2:] == ("_shadow_observer", "_expected_vtt_sha256")
    assert parameters["_shadow_observer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["_expected_vtt_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["_shadow_observer"].default is None
    assert parameters["_expected_vtt_sha256"].default is None


def test_align_expected_generation_rejects_before_media_or_backend(
    tmp_path, monkeypatch
):
    from voxweave.episode_transaction import InputStaleError

    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_vtt(), encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("generation mismatch reached media/backend work")

    monkeypatch.setattr(pipeline, "_find_sibling_media", forbidden)
    with pytest.raises(InputStaleError) as caught:
        pipeline.align(vtt, _expected_vtt_sha256="0" * 64)
    assert caught.value.failure.detail_code == "vtt-generation"
    assert vtt.read_text(encoding="utf-8") == _vtt()


def test_correct_apply_stale_generation_mutates_nothing(tmp_path, monkeypatch):
    from voxweave.episode_transaction import InputStaleError

    vtt = tmp_path / "episode.vtt"
    original = _vtt()
    concurrent = _vtt("concurrent winner")
    vtt.write_text(original, encoding="utf-8")
    evidence = tmp_path / "episode.align-evidence.json"
    evidence.write_text('{"kept":true}\n', encoding="utf-8")

    def concurrent_fix(_payload, **_kwargs):
        vtt.write_text(concurrent, encoding="utf-8")
        return [{"i": 0, "orig": "hello", "fixed": "hallo", "reason": "typo"}]

    monkeypatch.setattr(pipeline.asrfix_mod, "correct_cues", concurrent_fix)
    with pytest.raises(InputStaleError) as caught:
        pipeline.correct(vtt, apply=True)
    assert caught.value.failure.detail_code == "correct-generation"
    assert vtt.read_text(encoding="utf-8") == concurrent
    assert evidence.read_text(encoding="utf-8") == '{"kept":true}\n'


def test_correct_all_rejected_canonical_rewrite_deletes_stale_evidence(
    tmp_path, monkeypatch
):
    vtt = tmp_path / "episode.vtt"
    original = _vtt(settings="align:start position:20%")
    vtt.write_text(original, encoding="utf-8")
    evidence = tmp_path / "episode.align-evidence.json"
    evidence.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        pipeline.asrfix_mod,
        "correct_cues",
        lambda _payload, **_kwargs: [
            {"i": 0, "orig": "wrong", "fixed": "ignored", "reason": "bad quote"}
        ],
    )

    result = pipeline.correct(vtt, apply=True, align_after=True)

    assert result["applied"] == []
    assert result["aligned"] is False
    assert vtt.read_bytes() != original.encode("utf-8")
    assert not evidence.exists()
    assert tuple(result) == (
        "out",
        "audit",
        "applied",
        "rejected",
        "n_cues",
        "applied_in_place",
        "aligned",
    )


def test_correct_byte_identical_rewrite_retains_evidence(tmp_path, monkeypatch):
    vtt = tmp_path / "episode.vtt"
    original = _vtt()
    vtt.write_text(original, encoding="utf-8")
    evidence = tmp_path / "episode.align-evidence.json"
    evidence.write_text("still-current", encoding="utf-8")
    monkeypatch.setattr(
        pipeline.asrfix_mod, "correct_cues", lambda _payload, **_kwargs: []
    )

    result = pipeline.correct(vtt, apply=True, align_after=True)

    assert vtt.read_text(encoding="utf-8") == original
    assert evidence.read_text(encoding="utf-8") == "still-current"
    assert result["aligned"] is False


def test_correct_real_fix_hands_exact_committed_hash_to_align(tmp_path, monkeypatch):
    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_vtt(), encoding="utf-8")
    evidence = tmp_path / "episode.align-evidence.json"
    evidence.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        pipeline.asrfix_mod,
        "correct_cues",
        lambda _payload, **_kwargs: [
            {"i": 0, "orig": "hello", "fixed": "hallo", "reason": "typo"}
        ],
    )
    seen: dict[str, object] = {}

    def fake_align(path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        return path

    monkeypatch.setattr(pipeline, "align", fake_align)
    result = pipeline.correct(vtt, apply=True, align_after=True)

    committed = vtt.read_bytes()
    assert seen["path"] == vtt
    assert seen["_expected_vtt_sha256"] == hashlib.sha256(committed).hexdigest()
    assert result["aligned"] is True
    assert not evidence.exists()


def test_process_injected_words_is_media_free_with_absent_nominal_path(
    tmp_path, monkeypatch
):
    nominal = tmp_path / "absent.input.movie.mkv"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("injected words touched media or PANNs")

    monkeypatch.setattr(pipeline, "transcribe", forbidden)
    monkeypatch.setattr(pipeline, "decode_to_wav", forbidden)
    monkeypatch.setattr(pipeline, "media_fingerprint", forbidden)
    monkeypatch.setattr(songdet, "release_model", forbidden)

    out = pipeline.process(
        nominal,
        word_segments=("en", _units()),
        sdh=True,
        shot_snap=True,
    )

    assert out == tmp_path / "absent.input.movie.vtt"
    assert out.exists()
    assert (tmp_path / "absent.input.movie.json").exists()
    assert not (tmp_path / "absent.input.movie.sdh.vtt").exists()


def test_process_output_generation_change_stale_aborts_without_overwrite(
    tmp_path, monkeypatch
):
    from voxweave.episode_transaction import InputStaleError

    nominal = tmp_path / "episode.mkv"
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    json_path.write_bytes(b'{"old":true}')
    vtt_path.write_bytes(b"old vtt")
    concurrent_json = b'{"concurrent":true}'
    concurrent_vtt = b"concurrent vtt"
    real_segment = pipeline.segment_document

    def concurrent_segment(**kwargs):
        result = real_segment(**kwargs)
        json_path.write_bytes(concurrent_json)
        vtt_path.write_bytes(concurrent_vtt)
        return result

    monkeypatch.setattr(pipeline, "segment_document", concurrent_segment)
    with pytest.raises(InputStaleError) as caught:
        pipeline.process(nominal, word_segments=("en", _units()))
    assert caught.value.failure.detail_code == "process-output-generation"
    assert json_path.read_bytes() == concurrent_json
    assert vtt_path.read_bytes() == concurrent_vtt


def test_process_json_only_generation_change_stale_aborts(tmp_path, monkeypatch):
    from voxweave.episode_transaction import InputStaleError

    nominal = tmp_path / "episode.mkv"
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    json_path.write_bytes(b'{"old":true}')
    vtt_path.write_bytes(b"old vtt")
    concurrent_json = b'{"concurrent":true}'
    real_segment = pipeline.segment_document

    def concurrent_segment(**kwargs):
        result = real_segment(**kwargs)
        json_path.write_bytes(concurrent_json)
        return result

    monkeypatch.setattr(pipeline, "segment_document", concurrent_segment)
    with pytest.raises(InputStaleError) as caught:
        pipeline.process(nominal, word_segments=("en", _units()))

    assert caught.value.failure.detail_code == "process-output-generation"
    assert json_path.read_bytes() == concurrent_json
    assert vtt_path.read_bytes() == b"old vtt"


def test_process_releases_panns_after_handoff_when_segmentation_fails(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    released: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: ("en", _units(), None, [], [], None),
    )
    monkeypatch.setattr(
        pipeline,
        "segment_document",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("segmentation failed")),
    )
    monkeypatch.setattr(songdet, "release_model", lambda: released.append("panns"))

    with pytest.raises(RuntimeError, match="segmentation failed"):
        pipeline.process(media, sdh=True, shot_snap=False)
    assert released == ["panns"]


def test_process_panns_release_failure_preserves_exception_and_landed_accounting(
    tmp_path, monkeypatch
):
    class ReleaseFailure(OSError):
        pass

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    sidecar = tmp_path / "episode.sdh.vtt"
    release_calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: ("en", _units(), None, [], [], None),
    )

    def write_sidecar(*_args, **_kwargs):
        sidecar.write_bytes(b"auxiliary")
        return sidecar

    def fail_release():
        release_calls.append("panns")
        raise ReleaseFailure("release failed exactly")

    monkeypatch.setattr(pipeline, "_write_sdh_sidecar", write_sidecar)
    monkeypatch.setattr(songdet, "release_model", fail_release)

    with pytest.raises(ReleaseFailure, match="release failed exactly") as caught:
        pipeline.process(media, sdh=True, shot_snap=False)

    assert release_calls == ["panns"]
    assert caught.value.failure.kind == "model-release-failed"
    assert caught.value.failure.phase == "dispose"
    assert caught.value.failure.detail_code == "panns-release"
    assert caught.value.landed == (
        tmp_path / "episode.json",
        tmp_path / "episode.vtt",
    )
    assert caught.value.auxiliary_landed == (sidecar,)


def test_process_panns_release_failure_is_secondary_to_earlier_failure(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    release_calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: ("en", _units(), None, [], [], None),
    )
    monkeypatch.setattr(
        pipeline,
        "segment_document",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("segmentation failed")),
    )

    def fail_release():
        release_calls.append("panns")
        raise OSError("late release failed")

    monkeypatch.setattr(songdet, "release_model", fail_release)

    with pytest.raises(RuntimeError, match="segmentation failed") as caught:
        pipeline.process(media, sdh=True, shot_snap=False)

    assert release_calls == ["panns"]
    assert len(caught.value.secondary_failures) == 1
    assert caught.value.secondary_failures[0].kind == "model-release-failed"
    assert caught.value.secondary_failures[0].phase == "dispose"
    assert caught.value.secondary_failures[0].detail_code == "panns-release"


def test_sdh_generation_change_retains_existing_auxiliary(
    tmp_path, monkeypatch, caplog
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    sidecar = tmp_path / "episode.sdh.vtt"
    sidecar.write_bytes(b"existing sidecar")
    wav = tmp_path / "events.wav"
    released: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: ("en", _units(), None, [], [], None),
    )
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(songdet, "release_model", lambda: released.append("panns"))

    def concurrent_event_detection(*_args, **_kwargs):
        (tmp_path / "episode.json").write_text('{"later":true}', encoding="utf-8")
        return []

    monkeypatch.setattr(sdh, "detect_events", concurrent_event_detection)
    with caplog.at_level("WARNING", logger="voxweave"):
        out = pipeline.process(media, sdh=True, shot_snap=False)

    assert out == tmp_path / "episode.vtt"
    assert sidecar.read_bytes() == b"existing sidecar"
    assert released == ["panns"]
    assert any("stale SDH" in record.message for record in caplog.records)


def test_tolerant_mapping_observation_is_exact_and_rat7_recheck_is_dormant(
    tmp_path,
):
    from voxweave.episode_transaction import (
        capture_speaker_mapping,
        same_speaker_mapping_generation,
    )
    from voxweave.p6_ratifications import SPEAKER_MAPPING_CAS_ENABLED

    warnings: list[str] = []
    mapping = tmp_path / "episode.speakers.json"
    invalid = b'{"version":1,"speakers":[]}'
    mapping.write_bytes(invalid)
    readable = capture_speaker_mapping(
        mapping,
        known_ids={"S0"},
        warn=warnings.append,
    )
    assert readable.kind == "readable-bytes"
    assert readable.bytes_value == invalid
    assert readable.loader_status == "tolerated-invalid"
    assert readable.names == ()
    assert readable.private_observation is None
    assert len(warnings) == 1

    mapping.unlink()
    mapping.mkdir()
    warnings.clear()
    first = capture_speaker_mapping(mapping, known_ids={"S0"}, warn=warnings.append)
    second = capture_speaker_mapping(
        mapping, known_ids={"S0"}, warn=lambda _message: None
    )
    assert first.kind == "tolerated-unreadable"
    assert first.bytes_value is None
    assert first.loader_status == "unreadable"
    assert first.private_observation is not None
    assert first.private_observation.read_exception_class is IsADirectoryError
    assert first.private_observation.lstat_value is not None
    assert same_speaker_mapping_generation(first, second)
    assert len(warnings) == 1
    assert SPEAKER_MAPPING_CAS_ENABLED is False


def test_tolerant_mapping_distinguishes_absence_from_broken_symlink(tmp_path):
    from voxweave.episode_transaction import capture_speaker_mapping

    absent_warnings: list[str] = []
    absent = capture_speaker_mapping(
        tmp_path / "absent.speakers.json",
        known_ids={"S0"},
        warn=absent_warnings.append,
    )
    assert absent.kind == "absent"
    assert absent_warnings == []

    mapping = tmp_path / "episode.speakers.json"
    mapping.symlink_to(tmp_path / "missing-target.json")
    warnings: list[str] = []
    broken = capture_speaker_mapping(
        mapping,
        known_ids={"S0"},
        warn=warnings.append,
    )
    assert broken.kind == "tolerated-unreadable"
    assert broken.private_observation is not None
    assert broken.private_observation.read_exception_class is FileNotFoundError
    assert broken.private_observation.lstat_value is not None
    assert len(warnings) == 1


def test_file_generation_distinguishes_absence_from_broken_symlink(tmp_path):
    from voxweave.episode_transaction import capture_file_generation

    absent = tmp_path / "absent.json"
    assert capture_file_generation(absent).present is False

    broken = tmp_path / "episode.json"
    broken.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(FileNotFoundError):
        capture_file_generation(broken)


def test_transaction_module_has_no_model_renderer_or_pipeline_dependency():
    source = inspect.getsource(
        __import__("voxweave.episode_transaction", fromlist=["*"])
    )
    for forbidden in (
        "import pipeline",
        "from voxweave import pipeline",
        "import backend",
        "import segmentation_projector",
        "import align_projector",
    ):
        assert forbidden not in source


def test_align_transaction_stages_then_publishes_json_before_vtt(tmp_path, monkeypatch):
    from voxweave import episode_transaction

    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    cleanup = tmp_path / "episode.voiceprints.json"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")
    cleanup.write_bytes(b"old machine artifact")
    expected_json = episode_transaction.capture_file_generation(json_path)
    expected_vtt = episode_transaction.capture_file_generation(vtt_path)
    order: list[str] = []
    real_replace = episode_transaction._replace_stage

    def observed_replace(stage):
        order.append(stage.target.name)
        real_replace(stage)

    monkeypatch.setattr(episode_transaction, "_replace_stage", observed_replace)
    receipt = episode_transaction.commit_primary_outputs(
        command="align",
        episode_path=vtt_path,
        json_path=json_path,
        vtt_path=vtt_path,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        main_json_bytes=b"new json",
        vtt_bytes=b"new vtt",
        cleanup_paths=(
            episode_transaction.ArtifactCleanup(cleanup, "voiceprints-unlink"),
        ),
    )

    assert order == ["episode.json", "episode.vtt"]
    assert receipt.landed == (json_path, vtt_path)
    assert not cleanup.exists()
    assert not tuple(tmp_path.glob("*.part*"))


def test_primary_stage_failure_is_canonical_and_mutates_nothing(tmp_path, monkeypatch):
    from voxweave import episode_transaction

    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")
    expected_json = episode_transaction.capture_file_generation(json_path)
    expected_vtt = episode_transaction.capture_file_generation(vtt_path)
    real_stage = episode_transaction._stage_bytes

    def fail_json_stage(target, value):
        if target == json_path:
            raise OSError("stage disk full")
        return real_stage(target, value)

    monkeypatch.setattr(episode_transaction, "_stage_bytes", fail_json_stage)
    with pytest.raises(episode_transaction.TransactionOperationError) as caught:
        episode_transaction.commit_primary_outputs(
            command="align",
            episode_path=vtt_path,
            json_path=json_path,
            vtt_path=vtt_path,
            expected_json=expected_json,
            expected_vtt=expected_vtt,
            main_json_bytes=b"new json",
            vtt_bytes=b"new vtt",
        )
    assert caught.value.failure.kind == "stage-failed"
    assert caught.value.failure.phase == "stage"
    assert caught.value.failure.detail_code == "main-json-stage"
    assert caught.value.landed == ()
    assert json_path.read_bytes() == b"old json"
    assert vtt_path.read_bytes() == b"old vtt"


def test_vtt_replace_failure_reports_json_as_the_only_landed_primary(
    tmp_path, monkeypatch
):
    from voxweave import episode_transaction

    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")
    expected_json = episode_transaction.capture_file_generation(json_path)
    expected_vtt = episode_transaction.capture_file_generation(vtt_path)
    real_replace = episode_transaction._replace_stage

    def fail_vtt_replace(stage):
        if stage.target == vtt_path:
            raise OSError("replace failed")
        return real_replace(stage)

    monkeypatch.setattr(episode_transaction, "_replace_stage", fail_vtt_replace)
    with pytest.raises(episode_transaction.TransactionOperationError) as caught:
        episode_transaction.commit_primary_outputs(
            command="align",
            episode_path=vtt_path,
            json_path=json_path,
            vtt_path=vtt_path,
            expected_json=expected_json,
            expected_vtt=expected_vtt,
            main_json_bytes=b"new json",
            vtt_bytes=b"new vtt",
        )
    assert caught.value.failure.kind == "commit-failed"
    assert caught.value.failure.phase == "commit"
    assert caught.value.failure.detail_code == "vtt-replace"
    assert caught.value.landed == (json_path,)
    assert caught.value.leftovers == ()
    assert json_path.read_bytes() == b"new json"
    assert vtt_path.read_bytes() == b"old vtt"
    assert not tuple(tmp_path.glob("*.part*"))


def test_episode_lock_failure_is_canonical_and_discards_stages(tmp_path, monkeypatch):
    from voxweave import episode_transaction

    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")

    def fail_lock(_path):
        raise OSError("lock unavailable")

    monkeypatch.setattr(episode_transaction, "episode_lock", fail_lock)
    with pytest.raises(episode_transaction.TransactionOperationError) as caught:
        episode_transaction.commit_primary_outputs(
            command="align",
            episode_path=vtt_path,
            json_path=json_path,
            vtt_path=vtt_path,
            expected_json=episode_transaction.capture_file_generation(json_path),
            expected_vtt=episode_transaction.capture_file_generation(vtt_path),
            main_json_bytes=b"new json",
            vtt_bytes=b"new vtt",
        )
    assert caught.value.failure.kind == "episode-lock-failed"
    assert caught.value.failure.phase == "episode-lock"
    assert caught.value.failure.detail_code == "episode-lock-acquire"
    assert caught.value.landed == ()
    assert caught.value.leftovers == ()
    assert json_path.read_bytes() == b"old json"
    assert vtt_path.read_bytes() == b"old vtt"
    assert not tuple(tmp_path.glob("*.part*"))


def test_segmentation_transaction_consumes_commit_only_after_recheck(tmp_path):
    from voxweave.align_context import (
        consume_context_role,
        issue_segmentation_context,
        role_vector,
    )
    from voxweave.align_snapshot import FrozenObject, freeze_json
    from voxweave.episode_transaction import (
        capture_file_generation,
        commit_primary_outputs,
    )

    stable = freeze_json({"case": "transaction"})
    assert isinstance(stable, FrozenObject)
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    context = issue_segmentation_context(
        stable_fields=stable,
        target_path=vtt_path,
        sibling_path=json_path,
        effective_iso="en",
    )
    consume_context_role(
        context,
        "adapter",
        consumer="run_locked_segmentation_adapter",
    )
    consume_context_role(
        context,
        "encoder",
        consumer="encode_segmentation_candidates",
    )
    expected_json = capture_file_generation(json_path)
    expected_vtt = capture_file_generation(vtt_path)

    commit_primary_outputs(
        command="process",
        episode_path=vtt_path,
        json_path=json_path,
        vtt_path=vtt_path,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        main_json_bytes=b"{}",
        vtt_bytes=b"WEBVTT\n",
        context=context,
    )

    assert role_vector(context) == ("C", "C", "C")


def test_split_mapping_change_does_not_reprobe_before_rat7(tmp_path, monkeypatch):
    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "en",
                "word_segments": _units(),
                "speaker_turns": [[0.0, 1.0, "S0"]],
            }
        ),
        encoding="utf-8",
    )
    mapping = tmp_path / "episode.speakers.json"
    mapping.write_text('{"version":1,"speakers":{"S0":"Alice"}}', encoding="utf-8")
    real_segment = pipeline.segment_document

    def change_mapping(**kwargs):
        result = real_segment(**kwargs)
        mapping.write_text('{"version":1,"speakers":{"S0":"Bob"}}', encoding="utf-8")
        return result

    monkeypatch.setattr(pipeline, "segment_document", change_mapping)
    out = pipeline.split(json_path)
    assert "<v Alice>" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize("command", ["process", "split"])
def test_public_segmentation_requires_independent_selected_projection(
    command, tmp_path, monkeypatch
):
    from voxweave import segmentation_candidates

    json_path = tmp_path / "episode.json"
    if command == "split":
        json_path.write_text(
            json.dumps({"language": "en", "word_segments": _units()}),
            encoding="utf-8",
        )

    def reject_projection(*_args, **_kwargs):
        raise RuntimeError("independent projection rejected")

    monkeypatch.setattr(
        segmentation_candidates,
        "verify_selected_segmentation_projection",
        reject_projection,
    )
    with pytest.raises(RuntimeError, match="independent projection rejected"):
        if command == "process":
            pipeline.process(tmp_path / "episode.mkv", word_segments=("en", _units()))
        else:
            pipeline.split(json_path)
    assert not (tmp_path / "episode.vtt").exists()
    if command == "process":
        assert not json_path.exists()


def test_process_selected_candidate_enters_context_bound_transaction(
    tmp_path, monkeypatch
):
    from voxweave import episode_transaction
    from voxweave.align_context import IssuedSegmentationContext

    real_commit = episode_transaction.commit_primary_outputs
    seen: dict[str, object] = {}

    def observed_commit(**kwargs):
        seen.update(kwargs)
        return real_commit(**kwargs)

    monkeypatch.setattr(episode_transaction, "commit_primary_outputs", observed_commit)
    out = pipeline.process(tmp_path / "episode.mkv", word_segments=("en", _units()))
    assert out.exists()
    assert seen["command"] == "process"
    assert isinstance(seen["context"], IssuedSegmentationContext)
    assert seen["main_json_bytes"] == (tmp_path / "episode.json").read_bytes()
    assert seen["vtt_bytes"] == out.read_bytes()


def test_process_semantic_configuration_precedes_voiceprint_snapshot(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    events: list[str] = []

    def fail_semantic(_enabled):
        events.append("semantic-config")
        raise RuntimeError("semantic endpoint unavailable")

    class ForbiddenSnapshot:
        def __init__(self, _path):
            events.append("media-snapshot")
            raise AssertionError("snapshot ran before semantic configuration")

    monkeypatch.setattr(pipeline, "_make_semantic_engine", fail_semantic)
    monkeypatch.setattr(pipeline, "MediaSnapshot", ForbiddenSnapshot)

    with pytest.raises(RuntimeError, match="semantic endpoint unavailable"):
        pipeline.process(
            media,
            diarize=True,
            voiceprints=True,
            semantic_split=True,
        )

    assert events == ["semantic-config"]


def test_process_output_snapshot_precedes_voiceprint_media_snapshot(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    (tmp_path / "episode.json").mkdir()
    events: list[str] = []

    class ForbiddenSnapshot:
        def __init__(self, _path):
            events.append("media-snapshot")
            raise AssertionError("media snapshot ran before output snapshot")

    monkeypatch.setattr(pipeline, "MediaSnapshot", ForbiddenSnapshot)

    with pytest.raises(IsADirectoryError) as caught:
        pipeline.process(media, diarize=True, voiceprints=True)

    assert events == []
    assert caught.value.failure.kind == "subtitle-snapshot-failed"
    assert caught.value.failure.phase == "snapshot"
    assert caught.value.failure.detail_code == "sibling-read"


def test_process_voiceprint_candidate_uses_the_same_context_bound_transaction(
    tmp_path, monkeypatch
):
    from voxweave import episode_transaction
    from voxweave.align_context import IssuedSegmentationContext

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    turns = [(0.0, 1.0, "SPEAKER_00")]
    capture = pipeline.VoiceprintCapture(
        centroids={"SPEAKER_00": [1.0, *([0.0] * 15)]},
        provenance={
            "diarization_model": "example/diarizer",
            "outer_config_sha256": "d" * 64,
            "embedding_model": "example/embedder",
            "embedding_checkpoint": "e" * 64,
            "embedding_dim": 16,
            "audio": {
                "separated": False,
                "normalized": False,
                "sample_rate": 16000,
            },
            "pyannote_version": "3.4.0",
            "torch_version": "test",
        },
        turns=turns,
    )
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *_args, **_kwargs: ("en", _units(), None, [], turns, capture),
    )
    real_commit = episode_transaction.commit_primary_outputs
    seen: dict[str, object] = {}

    def observed_commit(**kwargs):
        seen.update(kwargs)
        return real_commit(**kwargs)

    monkeypatch.setattr(episode_transaction, "commit_primary_outputs", observed_commit)
    out = pipeline.process(
        media,
        diarize=True,
        voiceprints=True,
        shot_snap=False,
    )

    assert isinstance(seen["context"], IssuedSegmentationContext)
    assert seen["main_json_bytes"] == (tmp_path / "episode.json").read_bytes()
    assert seen["vtt_bytes"] == out.read_bytes()
    assert seen["machine_artifact"].path == tmp_path / "episode.voiceprints.json"


@pytest.mark.parametrize("command", ["process", "split"])
def test_successful_segmentation_retires_stale_align_evidence(command, tmp_path):
    evidence = tmp_path / "episode.align-evidence.json"
    evidence.write_text("stale", encoding="utf-8")
    if command == "process":
        pipeline.process(tmp_path / "episode.mkv", word_segments=("en", _units()))
    else:
        json_path = tmp_path / "episode.json"
        json_path.write_text(
            json.dumps({"language": "en", "word_segments": _units()}),
            encoding="utf-8",
        )
        pipeline.split(json_path)
    assert not evidence.exists()


def test_split_invalid_declared_pair_cleans_full_machine_artifact_set(tmp_path, caplog):
    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "en",
                "word_segments": _units(),
                "voiceprint_capture": "c" + "1" * 32,
            }
        ),
        encoding="utf-8",
    )
    artifacts = (
        tmp_path / "episode.voiceprints.json",
        tmp_path / "episode.speakers.suggest.json",
        tmp_path / "episode.speakers.html",
    )
    for artifact in artifacts:
        artifact.write_text("stale", encoding="utf-8")

    with caplog.at_level("WARNING", logger="voxweave"):
        pipeline.split(json_path)

    assert not any(artifact.exists() for artifact in artifacts)
    assert any(
        "dropping invalid voiceprint replay pair" in row.message
        for row in caplog.records
    )


def test_public_align_publishes_preencoded_primaries_through_transaction(
    tmp_path, monkeypatch
):
    from voxweave import backend, episode_transaction
    from voxweave.align_context import IssuedAlignContext

    media = tmp_path / "episode.wav"
    media.write_bytes(b"media")
    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": [
                    {"text": "你", "start": 0.0, "end": 0.5},
                    {"text": "好", "start": 0.5, "end": 1.0},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text("WEBVTT\n\n你好\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_args, **_kwargs: media)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [
            {"text": value, "start": float(index), "end": float(index) + 0.5}
            for index, value in enumerate(text)
        ],
    )
    real_commit = episode_transaction.commit_primary_outputs
    seen: dict[str, object] = {}

    def observed_commit(**kwargs):
        seen.update(kwargs)
        return real_commit(**kwargs)

    monkeypatch.setattr(episode_transaction, "commit_primary_outputs", observed_commit)
    assert pipeline.align(vtt_path) == vtt_path
    assert seen["command"] == "align"
    assert isinstance(seen["context"], IssuedAlignContext)
    assert seen["main_json_bytes"] == json_path.read_bytes()
    assert seen["vtt_bytes"] == vtt_path.read_bytes()


def test_public_align_requires_independent_selected_projection(tmp_path, monkeypatch):
    from voxweave import backend, candidate_encoder

    media = tmp_path / "episode.wav"
    media.write_bytes(b"media")
    json_path = tmp_path / "episode.json"
    original_json = json.dumps(
        {
            "language": "zh",
            "word_segments": [
                {"text": "你", "start": 0.0, "end": 0.5},
                {"text": "好", "start": 0.5, "end": 1.0},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    json_path.write_bytes(original_json)
    vtt_path = tmp_path / "episode.vtt"
    original_vtt = "WEBVTT\n\n你好\n".encode()
    vtt_path.write_bytes(original_vtt)
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_args, **_kwargs: media)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [
            {"text": value, "start": float(index), "end": float(index) + 0.5}
            for index, value in enumerate(text)
        ],
    )
    monkeypatch.setattr(
        candidate_encoder,
        "verify_selected_align_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("independent align projection rejected")
        ),
    )

    with pytest.raises(RuntimeError, match="independent align projection rejected"):
        pipeline.align(vtt_path)

    assert json_path.read_bytes() == original_json
    assert vtt_path.read_bytes() == original_vtt


def _stub_public_shadow_align(tmp_path, monkeypatch):
    from voxweave import backend

    media = tmp_path / "episode.wav"
    media.write_bytes(b"media")
    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": [
                    {"text": "你", "start": 0.0, "end": 0.5},
                    {"text": "好", "start": 0.5, "end": 1.0},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    vtt_path = tmp_path / "episode.vtt"
    vtt_path.write_text("WEBVTT\n\n你好\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_args, **_kwargs: media)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [
            {"text": value, "start": float(index), "end": float(index) + 0.5}
            for index, value in enumerate(text)
        ],
    )
    return media, json_path, vtt_path


def test_align_shadow_observer_runs_once_after_disposal(tmp_path, monkeypatch):
    from voxweave import backend

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    released: list[str] = []
    observed: list[object] = []
    monkeypatch.setattr(backend, "release", lambda: released.append("backend"))

    def observer(artifact):
        assert released == ["backend"]
        observed.append(artifact)

    assert pipeline.align(vtt_path, _shadow_observer=observer) == vtt_path
    assert len(observed) == 1
    artifact = observed[0]
    assert artifact.artifact_kind == "rich"
    assert artifact.status == "invalid"
    assert artifact.failure.detail_code == "w1-root-event"
    assert (
        artifact.selected["vtt_sha256"]
        == hashlib.sha256(vtt_path.read_bytes()).hexdigest()
    )
    assert (
        artifact.selected["json_sha256"]
        == hashlib.sha256(json_path.read_bytes()).hexdigest()
    )
    canonical = json.loads(artifact.to_canonical_bytes())
    assert tuple(canonical) == (
        "schema_version",
        "artifact_kind",
        "status",
        "failure",
        "input",
        "fresh",
        "legacy",
        "v2",
        "comparison",
        "selected",
    )
    assert tuple(canonical["v2"]) == ("semantic", "validators")
    assert canonical["v2"] == {"semantic": None, "validators": None}
    assert canonical["comparison"] == {"result": None}


def test_align_shadow_rich_failure_notifies_with_minimal_artifact(
    tmp_path, monkeypatch
):
    from voxweave import align_shadow

    _media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    monkeypatch.setattr(
        align_shadow,
        "build_rich_align_shadow_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("rich failed")),
    )
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path
    assert len(observed) == 1
    assert observed[0].artifact_kind == "minimal-failure"
    assert observed[0].failure.detail_code == "w1-root-event"
    assert observed[0].failure.secondary[-1].detail_code == (
        "rich-artifact-construction"
    )


def test_align_shadow_reports_strict_j0_failure_before_pending_w1(
    tmp_path, monkeypatch
):
    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    json_path.write_text(
        '{"language":"zh","language":"zh","word_segments":'
        '[{"text":"你","start":0.0,"end":0.5},'
        '{"text":"好","start":0.5,"end":1.0}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    assert len(observed) == 1
    artifact = observed[0]
    assert artifact.failure.kind == "v2-input-invalid"
    assert artifact.failure.detail_code == "sibling-json-duplicate-key"
    assert artifact.fresh["strict_input_status"] == "invalid"


def test_align_shadow_reports_strict_raw_failure_before_pending_w1(
    tmp_path, monkeypatch
):
    from voxweave import backend

    _media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [
            {"text": value, "start": index, "end": index + 1}
            for index, value in enumerate(text)
        ],
    )
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    artifact = observed[0]
    assert artifact.failure.kind == "fresh-time-transform-invalid"
    assert artifact.failure.phase == "strict-capture"
    assert artifact.failure.detail_code == "strict-raw-node"
    assert artifact.fresh["authority_distribution_status"] == "invalid"


def test_align_shadow_runs_seed_reconciliation_before_pending_w1(tmp_path, monkeypatch):
    from voxweave.core import align_seed

    _media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    monkeypatch.setattr(
        align_seed,
        "build_align_seed",
        lambda **_kwargs: align_seed.AlignSeedResult(
            "invalid",
            ("footprint-reconciliation",),
            None,
            (),
        ),
    )
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    artifact = observed[0]
    assert artifact.failure.kind == "fresh-reconciliation-invalid"
    assert artifact.failure.detail_code == "footprint-reconciliation"
    assert artifact.fresh["seed_status"] == "invalid"


def test_align_shadow_runs_profile_admission_before_pending_w1(tmp_path, monkeypatch):
    from voxweave import align_inputs

    _media, _json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    monkeypatch.setattr(
        align_inputs,
        "resolve_align_profile",
        lambda *_args, **_kwargs: align_inputs.ProfileResolution(
            None,
            align_inputs.ProfileStatus(
                "invalid",
                "stored-profile",
                "profile-domain",
            ),
        ),
    )
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    artifact = observed[0]
    assert artifact.failure.kind == "profile-invalid"
    assert artifact.failure.detail_code == "profile-domain"
    assert artifact.fresh["profile_status"] == "invalid"


def test_qwen_align_block_keeps_nominal_legacy_and_observes_sample_origin(
    tmp_path, monkeypatch
):
    import numpy as np
    import soundfile as sf

    from voxweave import backend

    wav = tmp_path / "prepared.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda *_args, **_kwargs: [{"text": "x", "start": 0.0, "end": 0.1}],
    )
    chunks: list = []
    origins: list[float] = []

    delivered = pipeline._align_blocks(
        wav,
        [{"text": "x", "start": 0.10001, "end": 0.5}],
        "zh",
        mms=False,
        ctc_model=None,
        crops=[(0.10001, 0.5)],
        reporter=pipeline.Reporter(),
        tmp_chunks=chunks,
        raw_call_observer=lambda _raw, _sources, origin, _identity: origins.append(
            origin
        ),
    )
    try:
        assert delivered[0][0]["start"] == 0.10001
        assert origins == [int(0.10001 * 16_000) / 16_000]
        assert origins[0] != delivered[0][0]["start"]
    finally:
        for path in chunks:
            path.unlink(missing_ok=True)


def test_align_shadow_observer_failure_cannot_change_selected_return(
    tmp_path, monkeypatch, caplog
):
    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")

    def fail_observer(_artifact):
        raise RuntimeError("observer failed")

    with caplog.at_level("WARNING", logger="voxweave"):
        assert pipeline.align(vtt_path, _shadow_observer=fail_observer) == vtt_path

    assert json_path.exists() and vtt_path.exists()
    assert any("align shadow observer failed" in row.message for row in caplog.records)
