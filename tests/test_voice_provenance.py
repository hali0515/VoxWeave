from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import soundfile as sf

from voxweave import backend, config, diarize, voicematch


EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"


def _cache_hf_file(
    cache_root: Path,
    repo_id: str,
    filename: str,
    payload: bytes,
    *,
    revision: str = "1" * 40,
    subfolder: str | None = None,
) -> tuple[Path, str]:
    repo_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
    digest = hashlib.sha256(payload).hexdigest()
    blob_path = repo_cache / "blobs" / digest
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)
    relative_path = Path(filename)
    if subfolder is not None:
        relative_path = Path(subfolder) / relative_path
    snapshot_path = repo_cache / "snapshots" / revision / relative_path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.symlink_to(os.path.relpath(blob_path, snapshot_path.parent))
    return snapshot_path, digest


def _community_cache(
    tmp_path: Path,
) -> tuple[Path, Path, Path, bytes, bytes, str]:
    cache_root = tmp_path / "audio-cache"
    revision = "c" * 40
    outer_bytes = json.dumps(
        {
            "version": "3.1.0",
            "pipeline": {
                "name": "pyannote.audio.pipelines.SpeakerDiarization",
                "params": {
                    "clustering": "VBxClustering",
                    "embedding": "$model/embedding",
                    "plda": "$model/plda",
                    "segmentation": "$model/segmentation",
                },
            },
        },
        sort_keys=True,
    ).encode()
    embedding_bytes = b"community-1 nested embedding checkpoint"
    config_path, _config_digest = _cache_hf_file(
        cache_root,
        COMMUNITY_MODEL,
        "config.yaml",
        outer_bytes,
        revision=revision,
    )
    embedding_path, _embedding_digest = _cache_hf_file(
        cache_root,
        COMMUNITY_MODEL,
        diarize.EMBEDDING_CHECKPOINT_FILE,
        embedding_bytes,
        revision=revision,
        subfolder="embedding",
    )
    return (
        cache_root,
        config_path,
        embedding_path,
        outer_bytes,
        embedding_bytes,
        revision,
    )


def _patch_community_downloads(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_path: Path,
    embedding_path: Path,
) -> list[tuple[str, str, dict[str, object]]]:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def download(repo_id: str, filename: str, **kwargs: object) -> str:
        calls.append((repo_id, filename, kwargs))
        if filename == "config.yaml" and kwargs.get("subfolder") is None:
            return os.fspath(config_path)
        if (
            filename == diarize.EMBEDDING_CHECKPOINT_FILE
            and kwargs.get("subfolder") == "embedding"
        ):
            return os.fspath(embedding_path)
        raise AssertionError(
            f"unexpected Hub request: {repo_id}/{kwargs.get('subfolder')}/{filename}"
        )

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def _write_local_pipeline_config(tmp_path: Path, embedding: Path) -> Path:
    path = tmp_path / "local-pipeline.yaml"
    path.write_text(
        json.dumps(
            {
                "pipeline": {
                    "name": "example.CustomPipeline",
                    "params": {
                        "clustering": "CustomClustering",
                        "embedding": os.fspath(embedding),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


class _Segment:
    start = 0.0
    end = 1.0


class _Annotation:
    def itertracks(self, *, yield_label: bool = False):
        row = (_Segment(), "track", "SPEAKER_00")
        yield row if yield_label else row[:2]

    def labels(self) -> list[str]:
        return ["SPEAKER_00"]


class _MutatingPipeline:
    def __init__(self, mutate) -> None:
        self._mutate = mutate
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, audio: object, **kwargs: object) -> object:
        self.calls.append((audio, kwargs))
        self._mutate()
        return SimpleNamespace(
            speaker_diarization=_Annotation(),
            exclusive_speaker_diarization=_Annotation(),
            speaker_embeddings=[[1.0, *([0.0] * 255)]],
        )


def _silent_wav(tmp_path: Path) -> Path:
    path = tmp_path / "silence.wav"
    sf.write(str(path), [0.0] * 1600, 16000)
    return path


def test_embedding_authority_pins_hub_revision_in_private_audio_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "private-audio-cache"
    revision = "a" * 40
    checkpoint_bytes = b"resolved wespeaker checkpoint"
    snapshot, digest = _cache_hf_file(
        cache_root,
        EMBEDDING_REPO,
        diarize.EMBEDDING_CHECKPOINT_FILE,
        checkpoint_bytes,
        revision=revision,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def download(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return os.fspath(snapshot)

    import huggingface_hub

    monkeypatch.setattr(config, "AUDIO_CACHE", str(cache_root))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    authority = diarize._embedding_load_authority(
        f"{EMBEDDING_REPO}@release-candidate",
        token="hf_secret",
    )

    assert authority is not None
    assert authority.binding.path == snapshot.resolve(strict=True)
    assert authority.binding.sha256 == digest
    assert authority.loaded_path == snapshot.resolve(strict=True)
    assert authority.provenance_value == f"{EMBEDDING_REPO}@{revision}"
    assert authority.loader_value == {
        "checkpoint": EMBEDDING_REPO,
        "revision": revision,
        "strict": False,
        "token": "hf_secret",
        "cache_dir": str(cache_root),
    }
    assert calls == [
        (
            (EMBEDDING_REPO, diarize.EMBEDDING_CHECKPOINT_FILE),
            {
                "subfolder": None,
                "revision": "release-candidate",
                "token": "hf_secret",
                "cache_dir": str(cache_root),
            },
        )
    ]
    assert "hf_secret" not in authority.provenance_value
    assert str(cache_root) not in authority.provenance_value


def test_community_plan_pins_nested_identity_and_private_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        cache_root,
        config_path,
        embedding_path,
        outer_bytes,
        embedding_bytes,
        revision,
    ) = _community_cache(tmp_path)
    monkeypatch.setattr(config, "AUDIO_CACHE", str(cache_root))
    calls = _patch_community_downloads(
        monkeypatch,
        config_path=config_path,
        embedding_path=embedding_path,
    )

    plan = diarize._prepare_pipeline_load(COMMUNITY_MODEL, "hf_private")

    assert isinstance(plan.checkpoint, dict)
    params = plan.checkpoint["pipeline"]["params"]
    common = {
        "checkpoint": COMMUNITY_MODEL,
        "revision": revision,
        "token": "hf_private",
        "cache_dir": str(cache_root),
    }
    assert params["segmentation"] == {**common, "subfolder": "segmentation"}
    assert params["plda"] == {**common, "subfolder": "plda"}
    assert params["embedding"] == {**common, "subfolder": "embedding"}
    assert plan.revision is None
    assert plan.outer_config_sha256 == hashlib.sha256(outer_bytes).hexdigest()
    assert plan.authority is not None
    assert plan.authority.binding.path == embedding_path.resolve(strict=True)
    assert plan.authority.binding.sha256 == hashlib.sha256(embedding_bytes).hexdigest()
    assert plan.authority.provenance_value == (
        f"{COMMUNITY_MODEL}@{revision}#subfolder=embedding"
    )
    assert [call[:2] for call in calls] == [
        (COMMUNITY_MODEL, "config.yaml"),
        (COMMUNITY_MODEL, diarize.EMBEDDING_CHECKPOINT_FILE),
    ]
    assert all(call[2]["cache_dir"] == str(cache_root) for call in calls)
    assert calls[1][2]["subfolder"] == "embedding"
    assert calls[1][2]["revision"] == revision
    assert config_path.read_bytes() == outer_bytes


def test_nested_checkpoint_mutation_during_construction_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        cache_root,
        config_path,
        embedding_path,
        _outer_bytes,
        embedding_bytes,
        revision,
    ) = _community_cache(tmp_path)
    monkeypatch.setattr(config, "AUDIO_CACHE", str(cache_root))
    _patch_community_downloads(
        monkeypatch,
        config_path=config_path,
        embedding_path=embedding_path,
    )
    constructed: list[SimpleNamespace] = []

    def construct(
        _pipeline_cls: object,
        _token: str | None,
        checkpoint: object,
        *,
        revision: str | None = None,
    ) -> SimpleNamespace:
        assert revision is None
        assert isinstance(checkpoint, dict)
        assert checkpoint["pipeline"]["params"]["embedding"] == {
            "checkpoint": COMMUNITY_MODEL,
            "revision": "c" * 40,
            "subfolder": "embedding",
            "token": "hf_private",
            "cache_dir": str(cache_root),
        }
        embedding_path.write_bytes(b"changed while pyannote constructed the model")
        pipeline = SimpleNamespace()
        constructed.append(pipeline)
        return pipeline

    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", construct)

    with pytest.raises(
        diarize.EmbeddingCheckpointChangedError,
        match="changed during pipeline construction",
    ):
        diarize._load_pipeline(
            object,
            "hf_private",
            model=COMMUNITY_MODEL,
        )

    assert revision == "c" * 40
    assert (
        hashlib.sha256(embedding_bytes).hexdigest()
        != hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    )
    assert len(constructed) == 1
    assert not hasattr(constructed[0], diarize._EMBEDDING_BINDING_ATTR)


def test_local_checkpoint_mutation_during_construction_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "embedding.ckpt"
    initial_bytes = b"local checkpoint consumed during construction"
    checkpoint.write_bytes(initial_bytes)
    pipeline_config = _write_local_pipeline_config(tmp_path, checkpoint)
    constructed: list[SimpleNamespace] = []

    def construct(
        _pipeline_cls: object,
        _token: str | None,
        load_document: object,
        *,
        revision: str | None = None,
    ) -> SimpleNamespace:
        assert revision is None
        assert isinstance(load_document, dict)
        assert load_document["pipeline"]["params"]["embedding"] == os.fspath(
            checkpoint.resolve(strict=True)
        )
        checkpoint.write_bytes(b"replacement before constructor returned")
        pipeline = SimpleNamespace()
        constructed.append(pipeline)
        return pipeline

    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", construct)

    with pytest.raises(
        diarize.EmbeddingCheckpointChangedError,
        match="changed during pipeline construction",
    ):
        diarize._load_pipeline(
            object,
            "hf_private",
            model=os.fspath(pipeline_config),
        )

    assert (
        hashlib.sha256(initial_bytes).hexdigest()
        != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert len(constructed) == 1
    assert not hasattr(constructed[0], diarize._EMBEDDING_BINDING_ATTR)


def test_inference_mutation_cannot_change_recorded_nested_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        cache_root,
        config_path,
        embedding_path,
        outer_bytes,
        embedding_bytes,
        revision,
    ) = _community_cache(tmp_path)
    monkeypatch.setattr(config, "AUDIO_CACHE", str(cache_root))
    _patch_community_downloads(
        monkeypatch,
        config_path=config_path,
        embedding_path=embedding_path,
    )

    def mutate_loaded_sources() -> None:
        embedding_path.write_bytes(b"replacement after the resident model was loaded")
        config_path.write_bytes(b"replacement outer config after construction")

    pipeline = _MutatingPipeline(mutate_loaded_sources)
    monkeypatch.setattr(
        diarize,
        "_call_pipeline_from_pretrained",
        lambda *_args, **_kwargs: pipeline,
    )
    loaded = diarize._load_pipeline(object, "hf_private", model=COMMUNITY_MODEL)
    assert loaded is pipeline
    monkeypatch.setattr(
        diarize,
        "_get_pipeline",
        lambda _token, _model: pipeline,
    )
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "4.0.7")

    first = diarize.diarize_turns(
        _silent_wav(tmp_path),
        token="hf_private",
        model=COMMUNITY_MODEL,
        want_embeddings=True,
    )
    second = diarize.diarize_turns(
        _silent_wav(tmp_path),
        token="hf_private",
        model=COMMUNITY_MODEL,
        want_embeddings=True,
    )

    embedding_digest = hashlib.sha256(embedding_bytes).hexdigest()
    outer_digest = hashlib.sha256(outer_bytes).hexdigest()
    embedding_source = f"{COMMUNITY_MODEL}@{revision}#subfolder=embedding"
    assert embedding_path.read_bytes() != embedding_bytes
    assert config_path.read_bytes() != outer_bytes
    assert first.provenance["diarization_model"] == COMMUNITY_MODEL
    assert first.provenance["outer_config_sha256"] == outer_digest
    assert first.provenance["embedding_model"] == embedding_source
    assert first.provenance["embedding_checkpoint"] == embedding_digest
    assert first.provenance["embedding_dim"] == 256
    assert first.provenance["pyannote_version"] == "4.0.7"
    assert second.provenance == first.provenance
    assert diarize._checkpoint_identity(pipeline) == embedding_digest
    assert first.centroids == {"SPEAKER_00": [1.0, *([0.0] * 255)]}
    assert len(pipeline.calls) == 2
    assert all("return_embeddings" not in kwargs for _audio, kwargs in pipeline.calls)

    compatibility = voicematch.build_compatibility_fingerprint(first.provenance)
    assert isinstance(compatibility, voicematch.CompatibilityFingerprint)
    other_model_provenance = dict(first.provenance)
    other_model_provenance["diarization_model"] = config.DEFAULT_DIARIZE_MODEL
    other_compatibility = voicematch.build_compatibility_fingerprint(
        other_model_provenance
    )
    assert isinstance(other_compatibility, voicematch.CompatibilityFingerprint)
    assert not voicematch.compatibility_equal(compatibility, other_compatibility)


def test_separator_identity_uses_hf_blob_or_explicit_local_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "audio"
    snapshot_path, blob_name = _cache_hf_file(
        cache_root,
        backend.SEPARATOR_REPO,
        backend.SEPARATOR_REPO_FILE,
        b"cached separator checkpoint",
    )
    separator_config = tmp_path / "separator.yaml"
    separator_config.write_bytes(b"model: {}\n")
    monkeypatch.setattr(
        backend,
        "_resolve_separator_files",
        lambda: (snapshot_path, separator_config),
    )
    assert backend.separator_identity()["checkpoint"] == blob_name

    local_payload = b"explicit local separator checkpoint"
    local_checkpoint = tmp_path / "separator.ckpt"
    local_checkpoint.write_bytes(local_payload)
    monkeypatch.setattr(
        backend,
        "_resolve_separator_files",
        lambda: (local_checkpoint, separator_config),
    )
    assert (
        backend.separator_identity()["checkpoint"]
        == hashlib.sha256(local_payload).hexdigest()
    )

    mismatched_blob = tmp_path / "blobs" / ("f" * 64)
    mismatched_payload = b"not the digest named by this path"
    mismatched_blob.parent.mkdir(exist_ok=True)
    mismatched_blob.write_bytes(mismatched_payload)
    monkeypatch.setattr(
        backend,
        "_resolve_separator_files",
        lambda: (mismatched_blob, separator_config),
    )
    resolved = backend.separator_identity()["checkpoint"]
    assert resolved == hashlib.sha256(mismatched_payload).hexdigest()
    assert resolved != mismatched_blob.name
