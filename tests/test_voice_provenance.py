import builtins
import hashlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from voxweave import backend, config, diarize, voicematch


EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"


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


def _capture_pyannote_cache(monkeypatch, cache_dir: Path | str | None) -> None:
    model_module = types.ModuleType("pyannote.audio.core.model")
    setattr(model_module, "CACHE_DIR", cache_dir)
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.model", model_module)


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

    assert diarize._checkpoint_identity(_real_shape_pipeline()) == blob_name


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

    resolved = diarize._checkpoint_identity(_real_shape_pipeline())
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

    before = diarize._checkpoint_identity(_real_shape_pipeline())
    _cache_hf_file(
        private_cache,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"new but unused private checkpoint",
    )
    after = diarize._checkpoint_identity(_real_shape_pipeline())

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
    assert diarize._checkpoint_identity(_real_shape_pipeline()) == legacy_digest


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

    pipeline = _real_shape_pipeline(f"{EMBEDDING_REPO}@release-2026")
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


def test_checkpoint_identity_hashes_resolved_bytes_not_blob_basename(
    tmp_path,
):
    blob_path = tmp_path / "blobs" / ("a" * 64)
    blob_path.parent.mkdir()
    blob_payload = b"bytes that do not match the claimed blob name"
    blob_path.write_bytes(blob_payload)
    cached = SimpleNamespace(_embedding=SimpleNamespace(embedding=blob_path))
    resolved = diarize._checkpoint_identity(cached)
    assert resolved == hashlib.sha256(blob_payload).hexdigest()
    assert resolved != blob_path.name

    local_path = tmp_path / "pytorch_model.bin"
    local_payload = b"explicit local model"
    local_path.write_bytes(local_payload)
    local = SimpleNamespace(_embedding=SimpleNamespace(embedding=local_path))
    assert (
        diarize._checkpoint_identity(local) == hashlib.sha256(local_payload).hexdigest()
    )


def test_checkpoint_identity_is_unresolved_when_hub_checkpoint_is_not_cached(
    tmp_path, monkeypatch
):
    _capture_pyannote_cache(monkeypatch, tmp_path / "empty-pyannote-cache")
    assert diarize._checkpoint_identity(_real_shape_pipeline()) == "unresolved"


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

    assert diarize._checkpoint_identity(_real_shape_pipeline()) == "unresolved"


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
    provenance = diarize._build_provenance(
        _real_shape_pipeline(),
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
