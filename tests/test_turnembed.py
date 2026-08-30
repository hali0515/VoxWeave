import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from voxweave import turnembed


def _vector(first: float, second: float) -> list[float]:
    return [first, second, *([0.0] * 14)]


class _FakeEmbeddingModel:
    sample_rate = turnembed.SAMPLE_RATE

    def __init__(self, rows: list[list[float]], *, min_num_samples: int) -> None:
        self.rows = iter(rows)
        self.min_num_samples = min_num_samples
        self.inputs: list[np.ndarray] = []

    def __call__(self, waveforms) -> np.ndarray:
        self.inputs.append(waveforms.detach().cpu().numpy().copy())
        return np.asarray([next(self.rows)], dtype=np.float32)


def _install_fake_provider(
    monkeypatch,
    model: _FakeEmbeddingModel,
) -> turnembed.EmbeddingIdentity:
    waveform = np.arange(20, dtype=np.float32)
    identity = turnembed.EmbeddingIdentity(
        model=turnembed.EMBEDDING_MODEL,
        checkpoint_sha256="b" * 64,
        pyannote_version="3.4.0",
    )
    monkeypatch.setattr(turnembed, "_read_mono_16k", lambda _path: waveform)
    monkeypatch.setattr(turnembed, "_get_inference", lambda: model)
    monkeypatch.setattr(turnembed, "_inference_identity", identity)
    return identity


def test_checkpoint_download_uses_explicit_token_and_cache(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "resolved.bin"
    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    assert turnembed._download_checkpoint(tmp_path, "hf_test") == checkpoint
    assert seen == {
        "repo_id": turnembed.EMBEDDING_MODEL,
        "filename": turnembed.EMBEDDING_CHECKPOINT_FILE,
        "cache_dir": tmp_path,
        "token": "hf_test",
    }


def test_bound_checkpoint_accepts_stable_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    payload = b"stable checkpoint bytes"
    checkpoint.write_bytes(payload)
    sentinel = object()
    seen: list[Path] = []

    def construct(resolved: Path):
        seen.append(resolved)
        return sentinel

    value, digest = turnembed._construct_bound_checkpoint(checkpoint, construct)

    assert value is sentinel
    assert seen == [checkpoint.resolve()]
    assert digest == hashlib.sha256(payload).hexdigest()


def test_bound_checkpoint_refuses_mutation_during_construction(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"before")

    def mutate(resolved: Path):
        resolved.write_bytes(b"after")
        return object()

    with pytest.raises(
        turnembed.TurnEmbeddingError,
        match="checkpoint changed during construction",
    ):
        turnembed._construct_bound_checkpoint(checkpoint, mutate)


def test_checkpoint_identity_hashes_contents_not_filename(tmp_path) -> None:
    checkpoint = tmp_path / ("a" * 64)
    payload = b"the filename is not an identity"
    checkpoint.write_bytes(payload)

    _, digest = turnembed._construct_bound_checkpoint(
        checkpoint,
        lambda _resolved: object(),
    )

    assert digest == hashlib.sha256(payload).hexdigest()
    assert digest != checkpoint.name


def test_inference_identity_is_published_with_singleton(monkeypatch) -> None:
    inference = object()
    identity = turnembed.EmbeddingIdentity(
        model=turnembed.EMBEDDING_MODEL,
        checkpoint_sha256="b" * 64,
        pyannote_version="3.4.0",
    )
    calls = 0

    def load_inference():
        nonlocal calls
        calls += 1
        return inference, identity

    monkeypatch.setattr(turnembed, "_inference", None)
    monkeypatch.setattr(turnembed, "_inference_identity", None)
    monkeypatch.setattr(turnembed, "_load_inference", load_inference)

    assert turnembed._get_inference() is inference
    assert turnembed._inference_identity == identity
    assert calls == 1


def test_attested_embeddings_are_dict_compatible_with_frozen_identity() -> None:
    identity = turnembed.EmbeddingIdentity(
        model=turnembed.EMBEDDING_MODEL,
        checkpoint_sha256="c" * 64,
        pyannote_version="3.4.0",
    )
    embeddings = turnembed.AttestedTurnEmbeddings(
        {3: _vector(1.0, 0.0)},
        identity=identity,
    )

    assert isinstance(embeddings, dict)
    assert embeddings == {3: _vector(1.0, 0.0)}
    assert embeddings.identity is identity
    embeddings[4] = _vector(0.0, 1.0)
    assert list(embeddings) == [3, 4]
    with pytest.raises(AttributeError):
        setattr(
            embeddings,
            "identity",
            turnembed.EmbeddingIdentity(
                model="replacement",
                checkpoint_sha256="d" * 64,
                pyannote_version="replacement",
            ),
        )


def test_turn_embeddings_contract_and_model_driven_padding(monkeypatch) -> None:
    model = _FakeEmbeddingModel(
        [_vector(3.0, 4.0), _vector(3.0, 4.0)],
        min_num_samples=8,
    )
    identity = _install_fake_provider(monkeypatch, model)
    monkeypatch.setattr(
        turnembed,
        "MIN_TURN_SECONDS",
        8 / turnembed.SAMPLE_RATE,
    )

    embeddings = turnembed.turn_embeddings(
        Path("unused.wav"),
        [
            (0.0, 4 / turnembed.SAMPLE_RATE, "SPEAKER_00"),
            (4 / turnembed.SAMPLE_RATE, 14 / turnembed.SAMPLE_RATE, "SPEAKER_00"),
        ],
    )

    assert list(embeddings) == [0, 1]
    assert isinstance(embeddings, turnembed.AttestedTurnEmbeddings)
    assert embeddings.identity is identity
    assert all(isinstance(value, list) for value in embeddings.values())
    assert all(
        isinstance(item, float) for value in embeddings.values() for item in value
    )
    assert model.inputs[0].shape == (1, 1, 8)
    assert model.inputs[0][0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    assert model.inputs[1].shape == (1, 1, 10)
    assert model.inputs[1][0, 0].tolist() == list(map(float, range(4, 14)))


def test_turn_embeddings_l2_normalizes_model_rows(monkeypatch) -> None:
    model = _FakeEmbeddingModel([_vector(3.0, 4.0)], min_num_samples=1)
    _install_fake_provider(monkeypatch, model)
    monkeypatch.setattr(turnembed, "MIN_TURN_SECONDS", 0.0)

    vector = turnembed.turn_embeddings(
        Path("unused.wav"),
        [(0.0, 1 / turnembed.SAMPLE_RATE, "SPEAKER_00")],
    )[0]

    assert vector[:2] == pytest.approx([0.6, 0.8])
    assert math.sqrt(math.fsum(value * value for value in vector)) == pytest.approx(1.0)
    assert turnembed.normalized_centroid([vector, vector]) == pytest.approx(vector)


def test_turn_embeddings_refuses_zero_model_row(monkeypatch) -> None:
    model = _FakeEmbeddingModel([_vector(0.0, 0.0)], min_num_samples=1)
    _install_fake_provider(monkeypatch, model)
    monkeypatch.setattr(turnembed, "MIN_TURN_SECONDS", 0.0)

    with pytest.raises(turnembed.TurnEmbeddingError, match="zero or non-finite"):
        turnembed.turn_embeddings(
            Path("unused.wav"),
            [(0.0, 1 / turnembed.SAMPLE_RATE, "SPEAKER_00")],
        )


def test_bisect_embeddings_is_deterministic_for_two_blobs() -> None:
    embeddings = {
        20: [-0.99, -0.08],
        2: [1.0, 0.05],
        13: [-1.0, 0.06],
        7: [0.98, -0.07],
    }

    expected = {2: "A", 7: "A", 13: "B", 20: "B"}
    assert turnembed.bisect_embeddings(embeddings) == expected
    assert turnembed.bisect_embeddings(dict(reversed(embeddings.items()))) == expected


def test_bisect_embeddings_refuses_degenerate_vectors() -> None:
    with pytest.raises(
        turnembed.UnsplittableSpeakerError,
        match="indistinguishable voice embeddings",
    ):
        turnembed.bisect_embeddings(
            {
                0: [1.0, 0.0],
                1: [1.0, 0.0],
                2: [1.0, 0.0],
            }
        )


def test_bisect_embeddings_refuses_symmetric_tied_top_eigenspace() -> None:
    with pytest.raises(
        turnembed.UnsplittableSpeakerError,
        match="ambiguous principal voice axis",
    ):
        turnembed.bisect_embeddings(
            {
                0: [1.0, 0.0],
                1: [0.0, 1.0],
                2: [-1.0, 0.0],
                3: [0.0, -1.0],
            }
        )
