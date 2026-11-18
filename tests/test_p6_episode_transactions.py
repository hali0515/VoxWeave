from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from voxweave import artifacts, episode_transaction, pipeline, publication, sdh, songdet
from voxweave import llm_commands
from voxweave import paths
from voxweave import process_publication
from voxweave import transcription
from voxweave import vocals


def _vtt(text: str = "hello", *, settings: str = "") -> str:
    suffix = f" {settings}" if settings else ""
    return f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000{suffix}\n{text}\n"


def _units(text: str = "hello") -> list[dict[str, object]]:
    return [{"text": text, "start": 0.0, "end": 1.0}]


def test_align_exposes_only_the_permitted_private_transaction_seam():
    parameters = inspect.signature(pipeline.align).parameters
    assert tuple(parameters)[-1:] == ("_expected_vtt_sha256",)
    assert parameters["_expected_vtt_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["_expected_vtt_sha256"].default is None


def test_align_expected_generation_rejects_before_media_or_backend(
    tmp_path, monkeypatch
):
    from voxweave.episode_transaction import InputStaleError

    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_vtt(), encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("generation mismatch reached media/backend work")

    monkeypatch.setattr(paths, "find_sibling_media", forbidden)
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

    def concurrent_fix(_payload, **_kwargs):
        vtt.write_text(concurrent, encoding="utf-8")
        return [{"i": 0, "orig": "hello", "fixed": "hallo", "reason": "typo"}]

    monkeypatch.setattr(llm_commands.asrfix_mod, "correct_cues", concurrent_fix)
    with pytest.raises(InputStaleError) as caught:
        llm_commands.correct(vtt, apply=True)
    assert caught.value.failure.detail_code == "correct-generation"
    assert vtt.read_text(encoding="utf-8") == concurrent


def test_correct_all_rejected_canonical_rewrite_returns_expected_keys(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mp4"
    media.write_bytes(b"media")
    vtt = tmp_path / "episode.vtt"
    original = _vtt(settings="align:start position:20%")
    vtt.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        llm_commands.asrfix_mod,
        "correct_cues",
        lambda _payload, **_kwargs: [
            {"i": 0, "orig": "wrong", "fixed": "ignored", "reason": "bad quote"}
        ],
    )

    result = llm_commands.correct(vtt, apply=True, align_after=True)

    assert result["applied"] == []
    assert result["aligned"] is False
    assert vtt.read_bytes() != original.encode("utf-8")
    assert tuple(result) == (
        "out",
        "audit",
        "applied",
        "rejected",
        "n_cues",
        "applied_in_place",
        "aligned",
    )


def test_correct_byte_identical_rewrite_skips_align(tmp_path, monkeypatch):
    vtt = tmp_path / "episode.vtt"
    original = _vtt()
    vtt.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        llm_commands.asrfix_mod, "correct_cues", lambda _payload, **_kwargs: []
    )

    result = llm_commands.correct(vtt, apply=True, align_after=True)

    assert vtt.read_text(encoding="utf-8") == original
    assert result["aligned"] is False


def test_correct_real_fix_hands_exact_committed_hash_to_align(tmp_path, monkeypatch):
    vtt = tmp_path / "episode.vtt"
    vtt.write_text(_vtt(), encoding="utf-8")
    monkeypatch.setattr(
        llm_commands.asrfix_mod,
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
    result = llm_commands.correct(vtt, apply=True, align_after=True)

    committed = vtt.read_bytes()
    assert seen["path"] == vtt
    assert seen["_expected_vtt_sha256"] == hashlib.sha256(committed).hexdigest()
    assert result["aligned"] is True


def test_process_injected_words_is_media_free_with_absent_nominal_path(
    tmp_path, monkeypatch
):
    nominal = tmp_path / "absent.input.movie.mkv"

    def forbidden(*_args, **_kwargs):
        pytest.fail("injected words touched media or PANNs")

    monkeypatch.setattr(pipeline, "transcribe", forbidden)
    monkeypatch.setattr(process_publication, "decode_to_wav", forbidden)
    monkeypatch.setattr(transcription, "decode_to_wav", forbidden)
    monkeypatch.setattr(vocals, "decode_to_wav", forbidden)
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

    monkeypatch.setattr(process_publication, "segment_document", concurrent_segment)
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

    monkeypatch.setattr(process_publication, "segment_document", concurrent_segment)
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
        process_publication,
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

    monkeypatch.setattr(process_publication, "_write_sdh_sidecar", write_sidecar)
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
        process_publication,
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
    monkeypatch.setattr(
        process_publication, "decode_to_wav", lambda *_args, **_kwargs: wav
    )
    monkeypatch.setattr(vocals, "decode_to_wav", lambda *_args, **_kwargs: wav)
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


def test_tolerant_mapping_observation_is_exact(tmp_path):
    from voxweave.episode_transaction import (
        capture_speaker_mapping,
        same_speaker_mapping_generation,
    )

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
    ):
        assert forbidden not in source


def test_align_transaction_stages_then_publishes_json_before_vtt(tmp_path, monkeypatch):
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


def test_split_mapping_change_is_stale(tmp_path, monkeypatch):
    from voxweave.episode_transaction import InputStaleError

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
    with pytest.raises(InputStaleError) as caught:
        pipeline.split(json_path)
    assert caught.value.failure.detail_code == "speaker-mapping-generation"
    assert not (tmp_path / "episode.vtt").exists()


def test_process_publishes_preencoded_primaries_through_transaction(
    tmp_path, monkeypatch
):
    real_commit = episode_transaction.commit_primary_outputs
    seen: dict[str, object] = {}

    def observed_commit(**kwargs):
        seen.update(kwargs)
        return real_commit(**kwargs)

    monkeypatch.setattr(episode_transaction, "commit_primary_outputs", observed_commit)
    out = pipeline.process(tmp_path / "episode.mkv", word_segments=("en", _units()))
    assert out.exists()
    assert seen["command"] == "process"
    assert seen["main_json_bytes"] == (tmp_path / "episode.json").read_bytes()
    assert seen["vtt_bytes"] == out.read_bytes()


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
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_test_token")

    with pytest.raises(IsADirectoryError) as caught:
        pipeline.process(media, diarize=True, voiceprints=True)

    assert events == []
    assert caught.value.failure.kind == "subtitle-snapshot-failed"
    assert caught.value.failure.phase == "snapshot"
    assert caught.value.failure.detail_code == "sibling-read"


def test_process_voiceprint_candidate_uses_the_same_transaction(tmp_path, monkeypatch):
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
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_test_token")
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

    assert seen["main_json_bytes"] == (tmp_path / "episode.json").read_bytes()
    assert seen["vtt_bytes"] == out.read_bytes()
    assert seen["machine_artifact"].path == artifacts.claim_paths(media).voiceprints


def test_split_reports_encode_failure_with_its_cause(tmp_path, monkeypatch):
    from voxweave import realign

    json_path = tmp_path / "episode.json"
    json_path.write_text(
        json.dumps({"language": "en", "word_segments": _units()}), encoding="utf-8"
    )

    def fail_render(*_args, **_kwargs):
        raise OSError("render target vanished")

    monkeypatch.setattr(realign, "render_cues", fail_render)
    with pytest.raises(publication.PublicationEncodeError) as caught:
        pipeline.split(json_path)
    assert caught.value.detail_code == "vtt-encode"
    assert not (tmp_path / "episode.vtt").exists()


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
    stale_artifacts = (
        tmp_path / "episode.voiceprints.json",
        tmp_path / "episode.speakers.suggest.json",
        tmp_path / "episode.speakers.html",
    )
    for artifact in stale_artifacts:
        artifact.write_text("stale", encoding="utf-8")

    with caplog.at_level("WARNING", logger="voxweave"):
        pipeline.split(json_path)

    assert not any(artifact.exists() for artifact in stale_artifacts)
    assert any(
        "dropping invalid voiceprint replay pair" in row.message
        for row in caplog.records
    )


def test_public_align_publishes_preencoded_primaries_through_transaction(
    tmp_path, monkeypatch
):
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
    real_commit = episode_transaction.commit_primary_outputs
    seen: dict[str, object] = {}

    def observed_commit(**kwargs):
        seen.update(kwargs)
        return real_commit(**kwargs)

    monkeypatch.setattr(episode_transaction, "commit_primary_outputs", observed_commit)
    assert pipeline.align(vtt_path) == vtt_path
    assert seen["command"] == "align"
    assert seen["main_json_bytes"] == json_path.read_bytes()
    assert seen["vtt_bytes"] == vtt_path.read_bytes()
