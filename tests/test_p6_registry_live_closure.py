from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxweave import pipeline


def _stub_public_shadow_align(tmp_path, monkeypatch):
    import wave

    from voxweave import backend
    from voxweave.align_acquisition import qwen_sample_geometry

    media = tmp_path / "episode.wav"
    with wave.open(str(media), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 32_000)
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

    def fake_slice(
        _wav,
        start,
        end,
        *,
        _sample_geometry_observer=None,
        **_kwargs,
    ):
        geometry = qwen_sample_geometry(
            nominal_start=float(start),
            nominal_end=float(end),
            sample_rate=16_000,
            sample_count=32_000,
        )
        if _sample_geometry_observer is not None:
            _sample_geometry_observer(
                geometry.sample_start,
                geometry.sample_end,
                geometry.sample_rate,
                geometry.sample_count,
            )
        crop = tmp_path / "qwen-crop.wav"
        crop.write_bytes(media.read_bytes())
        return crop

    monkeypatch.setattr(pipeline, "slice_wav", fake_slice)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [
            {"text": value, "start": float(index), "end": float(index) + 0.5}
            for index, value in enumerate(text)
        ],
    )
    return media, json_path, vtt_path


def test_shadow_profile_exception_is_typed_and_selected_commit_is_contained(
    tmp_path, monkeypatch
):
    from voxweave import align_inputs

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    monkeypatch.setattr(
        align_inputs,
        "resolve_align_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("profile producer failed exactly")
        ),
    )
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    assert json_path.exists() and vtt_path.exists()
    assert len(observed) == 1
    assert observed[0].failure.kind == "shadow-internal-error"
    assert observed[0].failure.phase == "profile-stage"
    assert observed[0].failure.detail_code == "profile-stage"


def test_shadow_comparator_exception_is_typed_and_selected_commit_is_contained(
    tmp_path, monkeypatch
):
    from voxweave.core import align_compare

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    monkeypatch.setattr(
        align_compare,
        "compare_semantic_deltas",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("comparator producer failed exactly")
        ),
    )
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    assert json_path.exists() and vtt_path.exists()
    assert len(observed) == 1
    assert observed[0].failure.kind == "shadow-internal-error"
    assert observed[0].failure.phase == "comparator-stage"
    assert observed[0].failure.detail_code == "comparator-stage"


def test_shadow_renderer_exception_is_typed_and_selected_commit_is_contained(
    tmp_path, monkeypatch
):
    from voxweave import candidate_encoder

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    real_encode = candidate_encoder._encode_family

    def fail_boundary(context, result, family, delivery, projection_inputs):
        if family == "boundary-v2":
            raise RuntimeError("renderer producer failed exactly")
        return real_encode(context, result, family, delivery, projection_inputs)

    monkeypatch.setattr(candidate_encoder, "_encode_family", fail_boundary)
    observed: list[object] = []

    assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    assert json_path.exists() and vtt_path.exists()
    assert len(observed) == 1
    assert observed[0].failure.kind == "shadow-internal-error"
    assert observed[0].failure.phase == "renderer-stage"
    assert observed[0].failure.detail_code == "renderer-stage"


def test_minimal_shadow_failure_is_classified_without_callback_or_rollback(
    tmp_path, monkeypatch, caplog
):
    from voxweave import align_shadow, align_shadow_minimal

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    monkeypatch.setattr(
        align_shadow,
        "build_rich_align_shadow_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("rich failed")),
    )
    monkeypatch.setattr(
        align_shadow_minimal,
        "build_minimal_align_shadow_failure_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("minimal failed")),
    )
    observed: list[object] = []

    with caplog.at_level("WARNING", logger="voxweave"):
        assert pipeline.align(vtt_path, _shadow_observer=observed.append) == vtt_path

    assert json_path.exists() and vtt_path.exists()
    assert observed == []
    record = next(
        row
        for row in caplog.records
        if "align shadow artifact unavailable" in row.message
    )
    assert record.failure.kind == "shadow-artifact-unavailable"
    assert record.failure.phase == "minimal-artifact"
    assert record.failure.detail_code == "minimal-artifact-construction"


def test_evidence_encode_failure_preserves_public_exception_and_prewrite_bytes(
    tmp_path, monkeypatch
):
    from voxweave import align_evidence

    class EvidenceEncodeFailure(OSError):
        pass

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    original_json = json_path.read_bytes()
    original_vtt = vtt_path.read_bytes()

    def fail_encode(_evidence):
        raise EvidenceEncodeFailure("evidence encode failed exactly")

    monkeypatch.setattr(align_evidence, "encode_align_evidence", fail_encode)

    with pytest.raises(
        EvidenceEncodeFailure, match="evidence encode failed exactly"
    ) as caught:
        pipeline.align(vtt_path)

    assert type(caught.value) is EvidenceEncodeFailure
    assert str(caught.value) == "evidence encode failed exactly"
    assert caught.value.failure.kind == "preencode-failed"
    assert caught.value.failure.phase == "preencode"
    assert caught.value.failure.detail_code == "evidence-encode"
    assert json_path.read_bytes() == original_json
    assert vtt_path.read_bytes() == original_vtt


def test_pair_decision_recheck_is_typed_and_precedes_primary_replacement(
    tmp_path, monkeypatch
):
    from voxweave import episode_transaction

    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    media_path = tmp_path / "episode.wav"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")
    media_path.write_bytes(b"media")
    fingerprints = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        episode_transaction,
        "media_fingerprint",
        lambda _path: next(fingerprints),
    )

    with pytest.raises(
        episode_transaction.MediaStaleError,
        match="selected media pair decision changed during processing; re-run",
    ) as caught:
        episode_transaction.commit_primary_outputs(
            command="align",
            episode_path=vtt_path,
            json_path=json_path,
            vtt_path=vtt_path,
            expected_json=episode_transaction.capture_file_generation(json_path),
            expected_vtt=episode_transaction.capture_file_generation(vtt_path),
            main_json_bytes=b"new json",
            vtt_bytes=b"new vtt",
            media_path=media_path,
            expected_media_fingerprint="a" * 64,
            expected_voiceprint_media_fingerprint="a" * 64,
            expected_pair_decision=True,
        )

    assert caught.value.failure.kind == "media-stale"
    assert caught.value.failure.phase == "recheck"
    assert caught.value.failure.detail_code == "pair-decision"
    assert caught.value.landed == ()
    assert json_path.read_bytes() == b"old json"
    assert vtt_path.read_bytes() == b"old vtt"
    assert not tuple(tmp_path.glob("*.part*"))


def test_media_snapshot_disposal_preserves_public_exception_after_selected_commit(
    tmp_path, monkeypatch
):
    class SnapshotDisposeFailure(OSError):
        pass

    media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    media_sha256 = pipeline.media_fingerprint(media)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["voiceprint_capture"] = "c" + "1" * 32
    payload["voiceprint_media"] = media_sha256
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class FailingSnapshot:
        def __init__(self, _source):
            self.path = media
            self.fingerprint = media_sha256

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise SnapshotDisposeFailure("snapshot dispose failed exactly")

    monkeypatch.setattr(pipeline, "MediaSnapshot", FailingSnapshot)

    with pytest.raises(
        SnapshotDisposeFailure, match="snapshot dispose failed exactly"
    ) as caught:
        pipeline.align(vtt_path)

    assert type(caught.value) is SnapshotDisposeFailure
    assert str(caught.value) == "snapshot dispose failed exactly"
    assert caught.value.failure.kind == "snapshot-dispose-failed"
    assert caught.value.failure.phase == "dispose"
    assert caught.value.failure.detail_code == "media-snapshot-residue"
    assert json_path.exists() and vtt_path.exists()
    assert (tmp_path / "episode.align-evidence.json").exists()


def test_audio_temp_disposal_preserves_public_exception_after_selected_commit(
    tmp_path, monkeypatch
):
    class AudioDisposeFailure(OSError):
        pass

    _media, json_path, vtt_path = _stub_public_shadow_align(tmp_path, monkeypatch)
    real_unlink = Path.unlink

    def fail_crop_unlink(path, *args, **kwargs):
        if path.name == "qwen-crop.wav":
            raise AudioDisposeFailure("audio temp dispose failed exactly")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_crop_unlink)

    with pytest.raises(
        AudioDisposeFailure, match="audio temp dispose failed exactly"
    ) as caught:
        pipeline.align(vtt_path)

    assert type(caught.value) is AudioDisposeFailure
    assert str(caught.value) == "audio temp dispose failed exactly"
    assert caught.value.failure.kind == "snapshot-dispose-failed"
    assert caught.value.failure.phase == "dispose"
    assert caught.value.failure.detail_code == "audio-temp-residue"
    assert json_path.exists() and vtt_path.exists()
    assert (tmp_path / "episode.align-evidence.json").exists()
