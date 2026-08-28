import builtins
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest
import soundfile as sf
import yaml

from voxweave import backend, config, diarize, voicematch


EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
_PipelineT = TypeVar("_PipelineT")


def _cache_hf_file(
    cache_root: Path,
    repo_id: str,
    filename: str,
    payload: bytes,
    *,
    revision: str = "1" * 40,
    ref_name: str | None = "main",
) -> tuple[Path, str]:
    repo_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
    blob_name = hashlib.sha256(payload).hexdigest()
    blob_path = repo_cache / "blobs" / blob_name
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)
    if ref_name is not None:
        ref_path = repo_cache / "refs" / ref_name
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(revision, encoding="utf-8")
    snapshot_path = repo_cache / "snapshots" / revision / filename
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.symlink_to(os.path.relpath(blob_path, snapshot_path.parent))
    return snapshot_path, blob_name


def _real_shape_pipeline(repo_id: str = EMBEDDING_REPO) -> SimpleNamespace:
    return SimpleNamespace(
        embedding=repo_id,
        _embedding=SimpleNamespace(embedding=repo_id),
    )


def _bind_pipeline(pipeline: _PipelineT) -> _PipelineT:
    embedder = getattr(pipeline, "_embedding", None)
    authority = diarize._embedding_load_authority(getattr(embedder, "embedding", None))
    diarize._store_embedding_checkpoint(
        pipeline,
        authority.binding if authority is not None else None,
    )
    return pipeline


def _pipeline_config(tmp_path: Path, embedding: str | Path) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        json.dumps(
            {
                "pipeline": {
                    "name": "example.Pipeline",
                    "params": {"embedding": os.fspath(embedding)},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _capture_pyannote_cache(monkeypatch, cache_dir: Path | str | None) -> None:
    model_module = types.ModuleType("pyannote.audio.core.model")
    setattr(model_module, "CACHE_DIR", cache_dir)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", model_module)


class _Segment:
    start = 0.0
    end = 1.0


class _Annotation:
    def itertracks(self, *, yield_label=False):
        row = (_Segment(), "track", "SPEAKER_00")
        yield row if yield_label else row[:2]

    def labels(self):
        return ["SPEAKER_00"]


class _MutatingPipeline:
    def __init__(self, embedding: str | Path, mutate) -> None:
        self.embedding = os.fspath(embedding)
        self._embedding = SimpleNamespace(embedding=embedding)
        self._mutate = mutate
        self.calls = 0

    def __call__(self, _file, **_kwargs):
        self.calls += 1
        self._mutate(self.calls)
        return _Annotation(), [[1.0, *([0.0] * 15)]]


def _silent_wav(tmp_path: Path) -> Path:
    path = tmp_path / "silence.wav"
    sf.write(str(path), [0.0] * 1600, 16000)
    return path


def test_checkpoint_identity_resolves_real_pyannote_hub_id_shape(tmp_path, monkeypatch):
    cache_root = tmp_path / "audio"
    _snapshot, blob_name = _cache_hf_file(
        cache_root / diarize.PYANNOTE_CACHE_SUBDIR,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"real-shaped wespeaker checkpoint",
    )
    monkeypatch.setattr(config, "AUDIO_CACHE", str(cache_root))
    _capture_pyannote_cache(monkeypatch, cache_root / diarize.PYANNOTE_CACHE_SUBDIR)

    pipeline = _bind_pipeline(_real_shape_pipeline())
    assert diarize._checkpoint_identity(pipeline) == blob_name


def test_checkpoint_identity_uses_captured_cache_when_private_and_user_conflict(
    tmp_path, monkeypatch
):
    audio_cache = tmp_path / "audio"
    private_cache = audio_cache / diarize.PYANNOTE_CACHE_SUBDIR
    user_cache = tmp_path / "user-pyannote"
    _private_snapshot, private_digest = _cache_hf_file(
        private_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"unused private checkpoint",
    )
    _user_snapshot, user_digest = _cache_hf_file(
        user_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"checkpoint selected by pyannote",
    )
    monkeypatch.setattr(config, "AUDIO_CACHE", str(audio_cache))
    monkeypatch.setenv(diarize.PYANNOTE_CACHE_ENV, str(user_cache))
    _capture_pyannote_cache(monkeypatch, user_cache)

    pipeline = _bind_pipeline(_real_shape_pipeline())
    resolved = diarize._checkpoint_identity(pipeline)
    assert resolved == user_digest
    assert resolved != private_digest


def test_unused_private_cache_appearance_does_not_change_loaded_identity(
    tmp_path, monkeypatch
):
    audio_cache = tmp_path / "audio"
    private_cache = audio_cache / diarize.PYANNOTE_CACHE_SUBDIR
    user_cache = tmp_path / "user-pyannote"
    _snapshot, user_digest = _cache_hf_file(
        user_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"stable loaded checkpoint",
    )
    monkeypatch.setattr(config, "AUDIO_CACHE", str(audio_cache))
    _capture_pyannote_cache(monkeypatch, user_cache)

    pipeline = _bind_pipeline(_real_shape_pipeline())
    before = diarize._checkpoint_identity(pipeline)
    _cache_hf_file(
        private_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"new but unused private checkpoint",
    )
    after = diarize._checkpoint_identity(pipeline)

    assert before == after == user_digest


def test_checkpoint_identity_honors_cache_captured_before_voxweave_configures_env(
    tmp_path, monkeypatch
):
    audio_cache = tmp_path / "audio"
    private_cache = audio_cache / diarize.PYANNOTE_CACHE_SUBDIR
    legacy_cache = tmp_path / "legacy-pyannote"
    _cache_hf_file(
        private_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"unused private checkpoint",
    )
    _snapshot, legacy_digest = _cache_hf_file(
        legacy_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"checkpoint loaded before voxweave import",
    )
    monkeypatch.setattr(config, "AUDIO_CACHE", str(audio_cache))
    monkeypatch.delenv(diarize.PYANNOTE_CACHE_ENV, raising=False)
    _capture_pyannote_cache(monkeypatch, legacy_cache)

    diarize._configure_pyannote_cache()

    assert os.environ[diarize.PYANNOTE_CACHE_ENV] == str(private_cache)
    pipeline = _bind_pipeline(_real_shape_pipeline())
    assert diarize._checkpoint_identity(pipeline) == legacy_digest


def test_checkpoint_identity_resolves_named_pinned_revision(tmp_path, monkeypatch):
    cache_root = tmp_path / "captured-pyannote"
    _snapshot, digest = _cache_hf_file(
        cache_root,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"pinned checkpoint",
        revision="2" * 40,
        ref_name="release-2026",
    )
    _capture_pyannote_cache(monkeypatch, cache_root)

    pipeline = _bind_pipeline(_real_shape_pipeline(f"{EMBEDDING_REPO}@release-2026"))
    assert diarize._checkpoint_identity(pipeline) == digest


def test_pyannote_cache_defaults_under_audio_cache_when_unset(tmp_path, monkeypatch):
    audio_cache = tmp_path / "audio"
    monkeypatch.setattr(config, "AUDIO_CACHE", str(audio_cache))
    monkeypatch.delenv(diarize.PYANNOTE_CACHE_ENV, raising=False)

    diarize._configure_pyannote_cache()

    assert os.environ[diarize.PYANNOTE_CACHE_ENV] == str(
        audio_cache / diarize.PYANNOTE_CACHE_SUBDIR
    )


def test_pyannote_cache_preserves_user_location(tmp_path, monkeypatch):
    user_cache = tmp_path / "user-selected"
    monkeypatch.setattr(config, "AUDIO_CACHE", str(tmp_path / "audio"))
    monkeypatch.setenv(diarize.PYANNOTE_CACHE_ENV, str(user_cache))

    diarize._configure_pyannote_cache()

    assert os.environ[diarize.PYANNOTE_CACHE_ENV] == str(user_cache)


def test_get_pipeline_defaults_pyannote_cache_before_import(tmp_path, monkeypatch):
    audio_cache = tmp_path / "audio"
    expected_cache = str(audio_cache / diarize.PYANNOTE_CACHE_SUBDIR)
    monkeypatch.setattr(config, "AUDIO_CACHE", str(audio_cache))
    monkeypatch.delenv(diarize.PYANNOTE_CACHE_ENV, raising=False)
    monkeypatch.setattr(diarize, "_pipeline", None)
    monkeypatch.setattr(diarize, "_ensure_torchaudio_compat", lambda: None)

    fake_package = types.ModuleType("pyannote")
    fake_package.__path__ = []
    fake_audio = types.ModuleType("pyannote.audio")
    setattr(fake_audio, "Pipeline", object())
    monkeypatch.setitem(sys.modules, "pyannote", fake_package)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    import_observations: list[str | None] = []
    real_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            import_observations.append(os.environ.get(diarize.PYANNOTE_CACHE_ENV))
        return real_import(name, *args, **kwargs)

    loaded = SimpleNamespace(to=lambda _device: None)
    monkeypatch.setattr(builtins, "__import__", tracking_import)
    monkeypatch.setattr(diarize, "_load_pipeline", lambda _cls, _token: loaded)

    assert diarize._get_pipeline("hf_test") is loaded
    assert import_observations == [expected_cache]


def test_load_pipeline_pins_hub_revision_and_stores_digest(tmp_path, monkeypatch):
    audio_cache = tmp_path / "audio"
    pyannote_cache = audio_cache / diarize.PYANNOTE_CACHE_SUBDIR
    snapshot, digest = _cache_hf_file(
        pyannote_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"construction-edge checkpoint",
    )
    pipeline_config = _pipeline_config(tmp_path, EMBEDDING_REPO)
    monkeypatch.setattr(config, "AUDIO_CACHE", str(audio_cache))
    monkeypatch.setattr(diarize, "DIARIZE_MODEL", os.fspath(pipeline_config))
    monkeypatch.setenv(diarize.PYANNOTE_CACHE_ENV, str(pyannote_cache))
    _capture_pyannote_cache(monkeypatch, pyannote_cache)

    loaded = _MutatingPipeline(EMBEDDING_REPO, lambda _call: None)
    seen_embedding: list[str] = []

    def construct(_cls, _token, checkpoint_path):
        document = yaml.safe_load(Path(checkpoint_path).read_text(encoding="utf-8"))
        seen_embedding.append(document["pipeline"]["params"]["embedding"])
        return loaded

    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", construct)

    assert diarize._load_pipeline(object, "hf_test") is loaded
    assert seen_embedding == [os.fspath(snapshot.resolve(strict=True))]
    binding = getattr(loaded, diarize._EMBEDDING_BINDING_ATTR)
    assert binding.path == snapshot.resolve(strict=True)
    assert binding.sha256 == digest
    assert diarize._checkpoint_identity(loaded) == digest


def test_load_pipeline_records_unresolved_when_hub_pin_is_unavailable(
    tmp_path, monkeypatch
):
    empty_cache = tmp_path / "empty-pyannote-cache"
    pipeline_config = _pipeline_config(tmp_path, EMBEDDING_REPO)
    monkeypatch.setattr(diarize, "DIARIZE_MODEL", os.fspath(pipeline_config))
    _capture_pyannote_cache(monkeypatch, empty_cache)

    loaded = _MutatingPipeline(EMBEDDING_REPO, lambda _call: None)
    seen_embedding: list[str] = []

    def construct(_cls, _token, checkpoint_path):
        document = yaml.safe_load(Path(checkpoint_path).read_text(encoding="utf-8"))
        seen_embedding.append(document["pipeline"]["params"]["embedding"])
        return loaded

    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", construct)

    assert diarize._load_pipeline(object, "hf_test") is loaded
    assert seen_embedding == [EMBEDDING_REPO]
    assert diarize._checkpoint_identity(loaded) == "unresolved"


def test_hub_ref_mutated_inside_construction_keeps_pinned_identity(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "captured-pyannote"
    snapshot_a, digest_a = _cache_hf_file(
        cache_root,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"checkpoint consumed during construction",
        revision="1" * 40,
    )
    _snapshot_b, digest_b = _cache_hf_file(
        cache_root,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"checkpoint selected by the flipped ref",
        revision="2" * 40,
        ref_name=None,
    )
    repo_cache = cache_root / f"models--{EMBEDDING_REPO.replace('/', '--')}"
    ref = repo_cache / "refs" / "main"
    pipeline_config = _pipeline_config(tmp_path, EMBEDDING_REPO)
    monkeypatch.setattr(diarize, "DIARIZE_MODEL", os.fspath(pipeline_config))
    monkeypatch.setenv(diarize.PYANNOTE_CACHE_ENV, str(cache_root))
    _capture_pyannote_cache(monkeypatch, cache_root)

    consumed: list[str] = []

    def construct(_cls, _token, checkpoint_path):
        document = yaml.safe_load(Path(checkpoint_path).read_text(encoding="utf-8"))
        pinned = document["pipeline"]["params"]["embedding"]
        loaded_path = Path(pinned)
        assert loaded_path == snapshot_a.resolve(strict=True)
        consumed.append(hashlib.sha256(loaded_path.read_bytes()).hexdigest())
        ref.write_text("2" * 40, encoding="utf-8")
        return _MutatingPipeline(pinned, lambda _call: None)

    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", construct)
    pipeline = diarize._load_pipeline(object, "hf_test")
    monkeypatch.setattr(diarize, "_get_pipeline", lambda _token: pipeline)
    wav = _silent_wav(tmp_path)

    first = diarize.diarize_turns(wav, token="hf_test", want_embeddings=True)
    second = diarize.diarize_turns(wav, token="hf_test", want_embeddings=True)

    assert consumed == [digest_a]
    assert digest_a != digest_b
    assert ref.read_text(encoding="utf-8") == "2" * 40
    assert first.provenance["embedding_model"] == EMBEDDING_REPO
    assert first.provenance["embedding_checkpoint"] == digest_a
    assert second.provenance["embedding_checkpoint"] == digest_a


def test_local_checkpoint_mutated_inside_construction_refuses_capture(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "embedding.ckpt"
    loaded_bytes = b"local checkpoint consumed during construction"
    replacement = b"local checkpoint replaced before constructor returned"
    checkpoint.write_bytes(loaded_bytes)
    pipeline_config = _pipeline_config(tmp_path, checkpoint)
    monkeypatch.setattr(diarize, "DIARIZE_MODEL", os.fspath(pipeline_config))

    constructed: list[_MutatingPipeline] = []
    consumed: list[str] = []

    def construct(_cls, _token, checkpoint_path):
        document = yaml.safe_load(Path(checkpoint_path).read_text(encoding="utf-8"))
        configured = Path(document["pipeline"]["params"]["embedding"])
        assert configured == checkpoint.resolve(strict=True)
        consumed.append(hashlib.sha256(configured.read_bytes()).hexdigest())
        configured.write_bytes(replacement)
        loaded = _MutatingPipeline(configured, lambda _call: None)
        constructed.append(loaded)
        return loaded

    monkeypatch.setattr(diarize, "_call_pipeline_from_pretrained", construct)
    monkeypatch.setattr(
        diarize,
        "_get_pipeline",
        lambda token: diarize._load_pipeline(object, token),
    )

    with pytest.raises(
        diarize.EmbeddingCheckpointChangedError,
        match="changed during pipeline construction",
    ):
        diarize.diarize_turns(
            _silent_wav(tmp_path), token="hf_test", want_embeddings=True
        )

    assert consumed == [hashlib.sha256(loaded_bytes).hexdigest()]
    assert checkpoint.read_bytes() == replacement
    assert len(constructed) == 1
    assert not hasattr(constructed[0], diarize._EMBEDDING_BINDING_ATTR)


def test_hub_ref_mutated_during_inference_keeps_construction_identity(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "captured-pyannote"
    _snapshot_a, digest_a = _cache_hf_file(
        cache_root,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"checkpoint loaded into resident model",
        revision="1" * 40,
    )
    _snapshot_b, digest_b = _cache_hf_file(
        cache_root,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"checkpoint selected during inference",
        revision="2" * 40,
        ref_name=None,
    )
    ref = cache_root / f"models--{EMBEDDING_REPO.replace('/', '--')}" / "refs" / "main"
    _capture_pyannote_cache(monkeypatch, cache_root)

    def mutate_ref(_call):
        ref.write_text("2" * 40, encoding="utf-8")

    pipeline = _MutatingPipeline(EMBEDDING_REPO, mutate_ref)
    _bind_pipeline(pipeline)
    monkeypatch.setattr(diarize, "_get_pipeline", lambda _token: pipeline)

    result = diarize.diarize_turns(
        _silent_wav(tmp_path), token="hf_test", want_embeddings=True
    )

    assert ref.read_text(encoding="utf-8") == "2" * 40
    assert digest_a != digest_b
    assert result.provenance["embedding_checkpoint"] == digest_a


def test_local_checkpoint_mutated_during_inference_keeps_construction_identity(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "embedding.ckpt"
    loaded_bytes = b"local checkpoint loaded into resident model"
    replacement = b"local checkpoint replaced during inference"
    checkpoint.write_bytes(loaded_bytes)
    pipeline = _MutatingPipeline(
        checkpoint,
        lambda _call: checkpoint.write_bytes(replacement),
    )
    _bind_pipeline(pipeline)
    monkeypatch.setattr(diarize, "_get_pipeline", lambda _token: pipeline)

    result = diarize.diarize_turns(
        _silent_wav(tmp_path), token="hf_test", want_embeddings=True
    )

    assert checkpoint.read_bytes() == replacement
    assert (
        result.provenance["embedding_checkpoint"]
        == hashlib.sha256(loaded_bytes).hexdigest()
    )


def test_construction_identity_survives_multiple_diarize_calls(tmp_path, monkeypatch):
    checkpoint = tmp_path / "embedding.ckpt"
    loaded_bytes = b"one construction-edge checkpoint"
    checkpoint.write_bytes(loaded_bytes)

    def mutate_each_call(call: int) -> None:
        checkpoint.write_bytes(f"replacement {call}".encode())

    pipeline = _MutatingPipeline(checkpoint, mutate_each_call)
    _bind_pipeline(pipeline)
    monkeypatch.setattr(diarize, "_get_pipeline", lambda _token: pipeline)
    wav = _silent_wav(tmp_path)

    first = diarize.diarize_turns(wav, token="hf_test", want_embeddings=True)
    second = diarize.diarize_turns(wav, token="hf_test", want_embeddings=True)

    digest = hashlib.sha256(loaded_bytes).hexdigest()
    assert pipeline.calls == 2
    assert first.provenance["embedding_checkpoint"] == digest
    assert second.provenance["embedding_checkpoint"] == digest
    assert diarize._checkpoint_identity(pipeline) == digest


def test_checkpoint_identity_hashes_resolved_bytes_not_blob_basename(
    tmp_path,
):
    blob_path = tmp_path / "blobs" / ("a" * 64)
    blob_path.parent.mkdir()
    blob_payload = b"bytes that do not match the claimed blob name"
    blob_path.write_bytes(blob_payload)
    cached = _bind_pipeline(
        SimpleNamespace(_embedding=SimpleNamespace(embedding=blob_path))
    )
    resolved = diarize._checkpoint_identity(cached)
    assert resolved == hashlib.sha256(blob_payload).hexdigest()
    assert resolved != blob_path.name

    local_path = tmp_path / "pytorch_model.bin"
    local_payload = b"explicit local model"
    local_path.write_bytes(local_payload)
    local = _bind_pipeline(
        SimpleNamespace(_embedding=SimpleNamespace(embedding=local_path))
    )
    assert (
        diarize._checkpoint_identity(local) == hashlib.sha256(local_payload).hexdigest()
    )


def test_checkpoint_identity_is_unresolved_when_hub_checkpoint_is_not_cached(
    tmp_path, monkeypatch
):
    _capture_pyannote_cache(monkeypatch, tmp_path / "empty-pyannote-cache")
    pipeline = _bind_pipeline(_real_shape_pipeline())
    assert diarize._checkpoint_identity(pipeline) == "unresolved"


def test_checkpoint_identity_refuses_to_guess_without_captured_loader_cache(
    tmp_path, monkeypatch
):
    audio_cache = tmp_path / "audio"
    _cache_hf_file(
        audio_cache / diarize.PYANNOTE_CACHE_SUBDIR,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"present but not tied to the loader",
    )
    monkeypatch.setattr(config, "AUDIO_CACHE", str(audio_cache))
    monkeypatch.delitem(sys.modules, "pyannote.audio.core.model", raising=False)

    pipeline = _bind_pipeline(_real_shape_pipeline())
    assert diarize._checkpoint_identity(pipeline) == "unresolved"


def test_separator_identity_uses_hf_blob_or_explicit_local_digest(
    tmp_path, monkeypatch
):
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


def test_real_shape_capture_provenance_builds_known_compatibility(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "audio"
    outer_config = b"pipeline:\n  name: SpeakerDiarization\n"
    _cache_hf_file(
        cache_root,
        "pyannote/speaker-diarization-3.1",
        "config.yaml",
        outer_config,
    )
    _embedding_snapshot, embedding_blob = _cache_hf_file(
        cache_root / diarize.PYANNOTE_CACHE_SUBDIR,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"real-shaped wespeaker checkpoint",
    )
    separator_snapshot, separator_blob = _cache_hf_file(
        cache_root,
        backend.SEPARATOR_REPO,
        backend.SEPARATOR_REPO_FILE,
        b"cached separator checkpoint",
    )
    separator_config = tmp_path / "separator.yaml"
    separator_config.write_bytes(b"model: {}\n")
    monkeypatch.setattr(config, "AUDIO_CACHE", str(cache_root))
    _capture_pyannote_cache(monkeypatch, cache_root / diarize.PYANNOTE_CACHE_SUBDIR)
    monkeypatch.setattr(
        diarize,
        "DIARIZE_MODEL",
        "pyannote/speaker-diarization-3.1",
    )
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "3.4.0")
    monkeypatch.setattr(
        backend,
        "_resolve_separator_files",
        lambda: (separator_snapshot, separator_config),
    )

    separator = backend.separator_identity()
    pipeline = _bind_pipeline(_real_shape_pipeline())
    provenance = diarize._build_provenance(
        pipeline,
        embedding_dim=256,
        audio_profile={
            "separated": True,
            "normalized": False,
            "sample_rate": 16000,
            "separator": separator,
        },
        torch_version="2.11.0",
    )

    assert provenance["embedding_checkpoint"] == embedding_blob
    assert provenance["outer_config_sha256"] == hashlib.sha256(outer_config).hexdigest()
    assert separator["checkpoint"] == separator_blob
    compatibility = voicematch.build_compatibility_fingerprint(provenance)
    assert isinstance(compatibility, voicematch.CompatibilityFingerprint)
