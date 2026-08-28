import hashlib
import os
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
) -> tuple[Path, str]:
    repo_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
    blob_name = hashlib.sha256(payload).hexdigest()
    blob_path = repo_cache / "blobs" / blob_name
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)
    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")
    snapshot_path = repo_cache / "snapshots" / revision / filename
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.symlink_to(os.path.relpath(blob_path, snapshot_path.parent))
    return snapshot_path, blob_name


def _real_shape_pipeline(repo_id: str = EMBEDDING_REPO) -> SimpleNamespace:
    return SimpleNamespace(
        embedding=repo_id,
        _embedding=SimpleNamespace(embedding=repo_id),
    )


def test_checkpoint_identity_resolves_real_pyannote_hub_id_shape(tmp_path, monkeypatch):
    cache_root = tmp_path / "audio"
    _snapshot, blob_name = _cache_hf_file(
        cache_root,
        EMBEDDING_REPO,
        "pytorch_model.bin",
        b"real-shaped wespeaker checkpoint",
    )
    monkeypatch.setattr(config, "AUDIO_CACHE", str(cache_root))

    assert diarize._checkpoint_identity(_real_shape_pipeline()) == blob_name


def test_checkpoint_identity_keeps_blob_path_branch_and_rejects_mutable_path(
    tmp_path,
):
    blob_path = tmp_path / "blobs" / ("a" * 64)
    blob_path.parent.mkdir()
    blob_path.write_bytes(b"cached")
    cached = SimpleNamespace(_embedding=SimpleNamespace(embedding=blob_path))
    assert diarize._checkpoint_identity(cached) == blob_path.name

    local_path = tmp_path / "pytorch_model.bin"
    local_path.write_bytes(b"mutable local model")
    local = SimpleNamespace(_embedding=SimpleNamespace(embedding=local_path))
    assert diarize._checkpoint_identity(local) == "unresolved"


def test_checkpoint_identity_is_unresolved_when_hub_checkpoint_is_not_cached(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "AUDIO_CACHE", str(tmp_path / "empty-audio-cache"))
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
        cache_root,
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
