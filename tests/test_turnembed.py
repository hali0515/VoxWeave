import hashlib
import io
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
    monkeypatch.setattr(turnembed, "_get_inference", lambda _identity=None: model)
    monkeypatch.setattr(turnembed, "_inference_identity", identity)
    return identity


@pytest.mark.parametrize(
    ("repository", "subfolder"),
    [
        (turnembed.EMBEDDING_MODEL, None),
        ("pyannote/speaker-diarization-community-1", "embedding"),
    ],
)
def test_checkpoint_download_uses_pinned_authority_token_and_cache(
    tmp_path,
    monkeypatch,
    repository,
    subfolder,
) -> None:
    revision = "f" * 40
    checkpoint = tmp_path / "resolved.bin"
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(checkpoint)

    monkeypatch.setattr(turnembed.config, "AUDIO_CACHE", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    authority = turnembed._EmbeddingAuthority(
        repository,
        revision,
        subfolder,
    )

    assert turnembed._download_checkpoint(authority, "hf_test") == checkpoint
    assert calls == [
        (
            (
                repository,
                turnembed.EMBEDDING_CHECKPOINT_FILE,
            ),
            {
                "cache_dir": str(tmp_path),
                "revision": revision,
                "subfolder": subfolder,
                "token": "hf_test",
            },
        )
    ]


@pytest.mark.parametrize(
    ("repository", "subfolder"),
    [
        (turnembed.EMBEDDING_MODEL, None),
        ("pyannote/speaker-diarization-community-1", "embedding"),
    ],
)
def test_pyannote4_loader_binds_exact_snapshot_authorities(
    tmp_path,
    monkeypatch,
    repository,
    subfolder,
) -> None:
    commit = "a" * 40
    cache_dir = tmp_path / "audio-cache"
    checkpoint = (
        cache_dir / f"models--{repository.replace('/', '--')}" / "snapshots" / commit
    )
    if subfolder is not None:
        checkpoint = checkpoint / subfolder
    checkpoint = checkpoint / turnembed.EMBEDDING_CHECKPOINT_FILE
    checkpoint.parent.mkdir(parents=True)
    payload = b"pyannote 4 checkpoint"
    checkpoint.write_bytes(payload)
    model = object()
    inference = _FakeEmbeddingModel([], min_num_samples=1)
    model_source = f"{repository}@{commit}"
    if subfolder is not None:
        model_source = f"{model_source}#subfolder={subfolder}"
    requested = turnembed.EmbeddingIdentity(
        model=model_source,
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        pyannote_version="4.0.7",
    )
    authorities = []
    model_calls = []
    inference_calls = []

    from pyannote.audio import Model
    from pyannote.audio.pipelines import speaker_verification

    def fake_from_pretrained(path, **kwargs):
        model_calls.append((path, kwargs))
        return model

    def fake_pretrained(embedding, **kwargs):
        inference_calls.append((embedding, kwargs))
        return inference

    monkeypatch.setattr(turnembed.config, "AUDIO_CACHE", str(cache_dir))
    monkeypatch.setattr(turnembed.config, "conf_hf_token", lambda: "hf_test")
    monkeypatch.setattr(turnembed.runtime, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        turnembed,
        "_download_checkpoint",
        lambda authority, token: authorities.append((authority, token)) or checkpoint,
    )
    monkeypatch.setattr(turnembed, "_pyannote_version", lambda: "4.0.7")
    monkeypatch.setattr(Model, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        speaker_verification,
        "PretrainedSpeakerEmbedding",
        fake_pretrained,
    )

    loaded, identity = turnembed._load_inference(requested)

    assert loaded is inference
    assert authorities == [
        (
            turnembed._EmbeddingAuthority(
                repository,
                commit,
                subfolder,
            ),
            "hf_test",
        )
    ]
    assert len(model_calls) == 1
    loaded_checkpoint, model_kwargs = model_calls[0]
    assert isinstance(loaded_checkpoint, io.BytesIO)
    assert loaded_checkpoint.getvalue() == payload
    assert str(model_kwargs.pop("map_location")) == "cpu"
    assert model_kwargs == {
        "strict": True,
        "token": "hf_test",
        "cache_dir": str(cache_dir),
    }
    assert len(inference_calls) == 1
    loaded_model, inference_kwargs = inference_calls[0]
    assert loaded_model is model
    assert str(inference_kwargs.pop("device")) == "cpu"
    assert inference_kwargs == {
        "token": "hf_test",
        "cache_dir": str(cache_dir),
    }
    assert identity == requested


def test_bound_checkpoint_accepts_stable_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    payload = b"stable checkpoint bytes"
    checkpoint.write_bytes(payload)
    sentinel = object()
    seen: list[bytes] = []

    def construct(bound: io.BytesIO):
        seen.append(bound.getvalue())
        return sentinel

    value, digest = turnembed._construct_bound_checkpoint(checkpoint, construct)

    assert value is sentinel
    assert seen == [payload]
    assert digest == hashlib.sha256(payload).hexdigest()


def test_bound_checkpoint_refuses_mutation_during_construction(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"before")

    def mutate(bound: io.BytesIO):
        bound.seek(0)
        bound.write(b"after")
        bound.truncate()
        return object()

    with pytest.raises(
        turnembed.TurnEmbeddingError,
        match="checkpoint changed during construction",
    ):
        turnembed._construct_bound_checkpoint(checkpoint, mutate)


def test_bound_checkpoint_ignores_path_aba_during_construction(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    payload = b"attested checkpoint bytes"
    checkpoint.write_bytes(payload)

    def replace_path(bound: io.BytesIO):
        checkpoint.write_bytes(b"unattested replacement")
        loaded = bound.getvalue()
        checkpoint.write_bytes(payload)
        return loaded

    value, digest = turnembed._construct_bound_checkpoint(checkpoint, replace_path)

    assert value == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def test_bound_checkpoint_refuses_requested_digest_mismatch_before_construction(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"actual")
    constructed = False

    def construct(_bound: io.BytesIO):
        nonlocal constructed
        constructed = True
        return object()

    with pytest.raises(
        turnembed.TurnEmbeddingError,
        match="does not match requested identity",
    ):
        turnembed._construct_bound_checkpoint(
            checkpoint,
            construct,
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )

    assert constructed is False


def test_requested_pyannote_version_mismatch_refuses_before_download(
    monkeypatch,
) -> None:
    identity = turnembed.EmbeddingIdentity(
        model=turnembed.EMBEDDING_MODEL,
        checkpoint_sha256="a" * 64,
        pyannote_version="4.0.6",
    )
    monkeypatch.setattr(turnembed, "_pyannote_version", lambda: "4.0.7")
    monkeypatch.setattr(
        turnembed,
        "_download_checkpoint",
        lambda *_args: pytest.fail("version mismatch must precede download"),
    )

    with pytest.raises(
        turnembed.TurnEmbeddingError,
        match="version does not match requested identity",
    ):
        turnembed._load_inference(identity)


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

    def load_inference(_expected_identity=None):
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


def test_attested_turn_request_is_a_frozen_sequence() -> None:
    identity = turnembed.EmbeddingIdentity(
        model="pyannote/example@" + ("e" * 40) + "#subfolder=embedding",
        checkpoint_sha256="f" * 64,
        pyannote_version="4.0.7",
    )
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_00")]
    request = turnembed.AttestedTurnRequest(turns, identity=identity)

    assert list(request) == turns
    assert request[1:] == (turns[1],)
    assert request.identity is identity
    with pytest.raises(AttributeError):
        setattr(request, "identity", identity)


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

    turns = turnembed.AttestedTurnRequest(
        [
            (0.0, 4 / turnembed.SAMPLE_RATE, "SPEAKER_00"),
            (4 / turnembed.SAMPLE_RATE, 14 / turnembed.SAMPLE_RATE, "SPEAKER_00"),
        ],
        identity=identity,
    )
    embeddings = turnembed.turn_embeddings(Path("unused.wav"), turns)

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


# --------------------------------------------------------------------------
# Decoupled lane: voiceembed provider behind the same attested contract
# --------------------------------------------------------------------------


class _FakeVoiceprintNetwork:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def __call__(self, wave):
        import torch

        samples = wave.detach().cpu().numpy()
        self.inputs.append(samples.copy())
        row = torch.zeros((samples.shape[0], 192), dtype=torch.float32)
        row[:, 0] = 3.0
        row[:, 1] = 4.0
        return row


def _install_fake_voiceprint_embedder(monkeypatch):
    from voxweave import voiceembed

    network = _FakeVoiceprintNetwork()
    requested = []

    def fake_get_embedder(spec):
        requested.append(spec)
        return voiceembed.LoadedEmbedder(
            spec=spec,
            checkpoint_sha256=spec.sha256,
            network=network,
            device="cpu",
        )

    monkeypatch.setattr(voiceembed, "get_embedder", fake_get_embedder)
    monkeypatch.setattr(
        turnembed,
        "_load_inference",
        lambda *_a, **_k: pytest.fail("the decoupled lane never loads pyannote"),
    )
    return network, requested


def test_turn_minimum_is_shared_with_the_voiceprint_recipe() -> None:
    from voxweave import voiceembed

    assert turnembed.MIN_TURN_SECONDS == voiceembed.MIN_TURN_SECONDS == 2.0


def test_decoupled_identity_embeds_turns_with_the_registered_embedder(
    monkeypatch,
) -> None:
    from voxweave import voiceembed

    network, requested = _install_fake_voiceprint_embedder(monkeypatch)
    waveform = np.ones(4 * turnembed.SAMPLE_RATE, dtype=np.float32)
    monkeypatch.setattr(turnembed, "_read_mono_16k", lambda _path: waveform)
    identity = turnembed.EmbeddingIdentity.decoupled(
        voiceembed.REDIMNET2_B6.name, voiceembed.REDIMNET2_B6.sha256
    )
    request = turnembed.AttestedTurnRequest(
        [(0.0, 0.5, "SPEAKER_00"), (1.0, 3.5, "SPEAKER_00")],
        identity=identity,
    )

    embeddings = turnembed.turn_embeddings(Path("unused.wav"), request)

    assert embeddings.identity == identity
    assert embeddings.identity.pyannote_version is None
    assert list(embeddings) == [0, 1]
    assert embeddings[0][:2] == pytest.approx([0.6, 0.8])
    assert requested and all(spec is voiceembed.REDIMNET2_B6 for spec in requested)
    # The short turn is repeated up to the embedder's 1 s minimum.
    assert [sample.shape for sample in network.inputs] == [
        (1, turnembed.SAMPLE_RATE),
        (1, round(2.5 * turnembed.SAMPLE_RATE)),
    ]


@pytest.mark.parametrize(
    ("model", "checkpoint", "message"),
    [
        ("unknown-embedder", "a" * 64, "not available"),
        ("redimnet2-b6-vb2-vox2-cnc2-lm", "a" * 64, "does not match"),
    ],
)
def test_decoupled_identity_refuses_unknown_or_unpinned_models(
    monkeypatch, model, checkpoint, message
) -> None:
    _install_fake_voiceprint_embedder(monkeypatch)
    monkeypatch.setattr(
        turnembed,
        "_read_mono_16k",
        lambda _path: pytest.fail("identity is checked before audio"),
    )
    request = turnembed.AttestedTurnRequest(
        [(0.0, 1.0, "SPEAKER_00")],
        identity=turnembed.EmbeddingIdentity.decoupled(model, checkpoint),
    )

    with pytest.raises(turnembed.TurnEmbeddingError, match=message):
        turnembed.turn_embeddings(Path("unused.wav"), request)


def test_legacy_loader_refuses_a_decoupled_identity() -> None:
    identity = turnembed.EmbeddingIdentity.decoupled("any", "a" * 64)

    with pytest.raises(turnembed.TurnEmbeddingError, match="identity is invalid"):
        turnembed._load_inference(identity)


def test_recipe_centroids_run_centroid_v1_for_the_requested_labels(
    monkeypatch,
) -> None:
    from voxweave import voiceembed

    _install_fake_voiceprint_embedder(monkeypatch)
    waveform = np.ones(12 * turnembed.SAMPLE_RATE, dtype=np.float32)
    monkeypatch.setattr(turnembed, "_read_mono_16k", lambda _path: waveform)
    identity = turnembed.EmbeddingIdentity.decoupled(
        voiceembed.ANIME_VA.name, voiceembed.ANIME_VA.sha256
    )
    turns = [
        (0.0, 3.0, "SPEAKER_00"),
        (3.0, 6.0, "SPEAKER_01"),
        (6.0, 9.0, "B"),
        (9.0, 9.3, "SPEAKER_00"),
    ]

    centroids = turnembed.recipe_centroids(
        Path("unused.wav"), turns, ("SPEAKER_00", "B"), identity
    )

    assert set(centroids) == {"SPEAKER_00", "B"}
    assert centroids["B"][:2] == pytest.approx([0.6, 0.8])
    with pytest.raises(turnembed.TurnEmbeddingError, match="identity is invalid"):
        turnembed.recipe_centroids(
            Path("unused.wav"),
            turns,
            ("SPEAKER_00",),
            turnembed.EmbeddingIdentity(
                model=turnembed.EMBEDDING_MODEL,
                checkpoint_sha256="a" * 64,
                pyannote_version="4.0.7",
            ),
        )
