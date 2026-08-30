from __future__ import annotations

from pathlib import Path

import pytest

from voxweave import diarize


def _cached_embedding(
    tmp_path: Path,
    *,
    model_id: str,
    revision: str,
    subfolder: str | None = None,
) -> Path:
    relative = Path(diarize.EMBEDDING_CHECKPOINT_FILE)
    if subfolder is not None:
        relative = Path(subfolder) / relative
    checkpoint = (
        tmp_path
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
        / relative
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"embedding checkpoint")
    return checkpoint


def test_model_subfolder_child_revision_overrides_parent_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diarize.config, "AUDIO_CACHE", "/private/audio-cache")
    document: dict[str, object] = {"embedding": "$model/embedding@child-release"}

    diarize._expand_model_references(
        document,
        model_id="acme/custom-diarizer",
        revision="outer-release",
        token="hf_private",
    )

    assert document["embedding"] == {
        "checkpoint": "acme/custom-diarizer",
        "revision": "child-release",
        "subfolder": "embedding",
        "token": "hf_private",
        "cache_dir": "/private/audio-cache",
    }


def test_local_config_model_reference_is_based_at_config_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pipeline" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "pipeline:\n"
        "  name: pyannote.audio.pipelines.SpeakerDiarization\n"
        "  params:\n"
        "    embedding: $model/embedding\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        diarize, "_embedding_load_authority", lambda *_args, **_kwargs: None
    )

    plan = diarize._prepare_pipeline_load(str(config_path), "hf_private")

    assert isinstance(plan.checkpoint, dict)
    embedding = plan.checkpoint["pipeline"]["params"]["embedding"]
    assert isinstance(embedding, dict)
    assert Path(str(embedding["checkpoint"])) == config_path.parent.resolve()
    assert embedding["subfolder"] == "embedding"
    assert "revision" not in embedding


def test_embedding_mapping_preserves_original_loader_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "acme/custom-embedding"
    commit = "a" * 40
    checkpoint = _cached_embedding(
        tmp_path,
        model_id=model_id,
        revision=commit,
        subfolder="weights",
    )
    monkeypatch.setattr(diarize.config, "AUDIO_CACHE", str(tmp_path))

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *_args, **_kwargs: str(checkpoint),
    )

    authority = diarize._embedding_load_authority(
        {
            "checkpoint": model_id,
            "revision": "requested-release",
            "subfolder": "weights",
            "map_location": "cpu",
            "strict": True,
        },
        token="hf_private",
    )

    assert authority is not None
    assert isinstance(authority.loader_value, dict)
    assert authority.loader_value["map_location"] == "cpu"
    assert authority.loader_value["strict"] is True


def test_hub_string_to_mapping_preserves_string_loader_strict_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "acme/custom-embedding"
    commit = "b" * 40
    checkpoint = _cached_embedding(
        tmp_path,
        model_id=model_id,
        revision=commit,
    )
    monkeypatch.setattr(diarize.config, "AUDIO_CACHE", str(tmp_path))

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *_args, **_kwargs: str(checkpoint),
    )

    authority = diarize._embedding_load_authority(
        f"{model_id}@requested-release",
        token="hf_private",
    )

    assert authority is not None
    assert isinstance(authority.loader_value, dict)
    assert authority.loader_value["strict"] is False
