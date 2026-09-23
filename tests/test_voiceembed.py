"""Decoupled voiceprint embedders: registry, recipe, checkpoint pinning, inference.

Everything here runs offline on CPU: downloads and model forwards are faked at
the module boundaries, and the vendored networks are exercised with random
weights only.
"""

from __future__ import annotations

import hashlib
import io
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voxweave import config, voiceembed
from voxweave.voicebase import validate_vector

# The architecture of the pinned b6-vb2+vox2+cnc2_v0-lm.pt release asset, as its
# embedded model_config records it.
B6_MODEL_CONFIG = {
    "C": 64,
    "F": 72,
    "agg_gnorm": True,
    "block_1d_type": "conv+att",
    "block_2d_type": "basic_resnet",
    "causal": "none",
    "compress_tconvs": True,
    "dual_agg": False,
    "emb_bn": False,
    "embed_dim": 192,
    "feat_type": "tf",
    "fm_weigthing_type": "NC",
    "global_context_att": True,
    "group_divisor": 1,
    "hop_length": 160,
    "out_channels": 224,
    "pooling_func": "ASTP",
    "return_2d_output": True,
    "spec_in_channels": 1,
    "spec_params": {"do_preemph": True, "do_spec_aug": False, "norm_signal": True},
    "stages_setup": [
        [[1, 1], 3, 3, [[3, 3]], 64],
        [[2, 1], 4, 2, [[3, 3]], 64],
        [[1, 2], 5, 2, [[3, 3]], 48],
        [[2, 1], 5, 1, [[3, 3]], 48],
        [[1, 2], 4, 0.75, [[3, 3]], 32],
        [[2, 1], 3, 0.5, [[3, 3]], 24],
    ],
}


class _FakeNetwork:
    """Records every forward input; returns a vector keyed on the signal level."""

    def __init__(self, dim: int = 192) -> None:
        self.dim = dim
        self.inputs: list[np.ndarray] = []

    def __call__(self, wave):
        import torch

        samples = wave.detach().cpu().numpy()
        self.inputs.append(samples.copy())
        level = float(np.abs(samples).mean())
        row = torch.zeros((samples.shape[0], self.dim), dtype=torch.float32)
        row[:, 0] = level
        row[:, 1] = 1.0
        return row


def _install_fake_embedder(monkeypatch, spec=voiceembed.REDIMNET2_B6):
    network = _FakeNetwork(spec.embedding_dim)
    loaded = voiceembed.LoadedEmbedder(
        spec=spec,
        checkpoint_sha256=spec.sha256,
        network=network,
        device="cpu",
    )
    monkeypatch.setattr(voiceembed, "get_embedder", lambda _spec: loaded)
    return network


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("ja", voiceembed.ANIME_VA),
        ("JA", voiceembed.ANIME_VA),
        ("en", voiceembed.REDIMNET2_B6),
        ("zh", voiceembed.REDIMNET2_B6),
        (None, voiceembed.REDIMNET2_B6),
    ],
)
def test_auto_routes_japanese_to_anime_va_and_everything_else_to_redimnet2(
    language, expected
):
    assert voiceembed.resolve_voiceprint_model(None, language) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("redimnet2", voiceembed.REDIMNET2_B6),
        ("ReDimNet2", voiceembed.REDIMNET2_B6),
        ("anime-va", voiceembed.ANIME_VA),
        ("pyannote", voiceembed.LEGACY),
        (voiceembed.REDIMNET2_B6.name, voiceembed.REDIMNET2_B6),
        (voiceembed.ANIME_VA.name, voiceembed.ANIME_VA),
    ],
)
def test_explicit_choices_ignore_the_language(value, expected):
    assert voiceembed.resolve_voiceprint_model(value, "en") is expected


def test_choice_precedence_is_cli_then_env_then_conf(tmp_path, monkeypatch):
    conf = tmp_path / "voxweave.conf"
    conf.write_text('[voiceprint]\nmodel = "anime-va"\n', encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))

    assert voiceembed.resolve_voiceprint_choice(None) == voiceembed.ANIME_VA.name
    monkeypatch.setenv(voiceembed.ENV_MODEL, "pyannote")
    assert voiceembed.resolve_voiceprint_choice(None) == "pyannote"
    assert voiceembed.resolve_voiceprint_choice("redimnet2") == (
        voiceembed.REDIMNET2_B6.name
    )
    monkeypatch.delenv(voiceembed.ENV_MODEL)
    conf.write_text("", encoding="utf-8")
    assert voiceembed.resolve_voiceprint_choice(None) == voiceembed.AUTO


@pytest.mark.parametrize(
    ("where", "source"),
    [
        ("cli", "--voiceprint-model"),
        ("env", "environment VOXWEAVE_VOICEPRINT_MODEL"),
        ("conf", "config [voiceprint].model"),
    ],
)
def test_unknown_choice_fails_naming_its_source(tmp_path, monkeypatch, where, source):
    cli_value = None
    if where == "cli":
        cli_value = "ecapa"
    elif where == "env":
        monkeypatch.setenv(voiceembed.ENV_MODEL, "ecapa")
    else:
        conf = tmp_path / "voxweave.conf"
        conf.write_text('[voiceprint]\nmodel = "ecapa"\n', encoding="utf-8")
        monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))

    with pytest.raises(ValueError, match="unknown voiceprint model") as caught:
        voiceembed.resolve_voiceprint_model(cli_value, "en")
    assert source in str(caught.value)
    assert "auto, redimnet2, anime-va, pyannote" in str(caught.value)


def test_explicit_anime_va_on_another_language_warns(caplog):
    with caplog.at_level("WARNING", logger="voxweave"):
        assert voiceembed.resolve_voiceprint_model("anime-va", "en") is (
            voiceembed.ANIME_VA
        )
    assert "trained for ja only" in caplog.text


def test_voiceprint_section_is_a_known_documented_config_key():
    assert "voiceprint" in config._KNOWN_KEYS
    assert "[voiceprint]" in config._TEMPLATE
    assert '# model = "auto"' in config._TEMPLATE


def test_conf_voiceprint_model_ignores_wrong_types(tmp_path, monkeypatch, caplog):
    conf = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    conf.write_text("voiceprint = 3\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="voxweave"):
        assert config.conf_voiceprint_model() is None
    conf.write_text("[voiceprint]\nmodel = 3\n", encoding="utf-8")
    assert config.conf_voiceprint_model() is None
    conf.write_text('[voiceprint]\nmodel = "  "\n', encoding="utf-8")
    assert config.conf_voiceprint_model() is None
    assert "wrong type" in caplog.text


def test_registry_pins_both_checkpoints():
    assert voiceembed.REDIMNET2_B6.sha256 == (
        "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868"
    )
    assert voiceembed.REDIMNET2_B6.url == (
        "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/"
        "b6-vb2%2Bvox2%2Bcnc2_v0-lm.pt"
    )
    assert voiceembed.ANIME_VA.hf_repo == (
        "litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm"
    )
    assert voiceembed.ANIME_VA.hf_revision is not None
    assert len(voiceembed.ANIME_VA.hf_revision) == 40
    assert len(voiceembed.ANIME_VA.sha256) == 64
    for spec in voiceembed.EMBEDDERS.values():
        assert spec.sample_rate == voiceembed.SAMPLE_RATE
        assert spec.embedding_dim == 192
        assert 0.0 < spec.suggest < 1.0
        assert spec.margin >= 0.0


def test_import_touches_neither_torch_nor_the_network():
    code = (
        "import sys; import voxweave.voiceembed, voxweave.voicematch; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------------
# centroid-v1 recipe
# --------------------------------------------------------------------------


def test_centroid_segments_trim_overlap_with_other_speakers():
    turns = [
        (0.0, 5.0, "A"),
        (4.0, 6.0, "B"),
        (6.0, 9.0, "A"),
        (7.0, 7.5, "C"),
    ]

    # A keeps 0-4 (B overlaps 4-5) and 6-7 / 7.5-9 (C overlaps 7-7.5). The
    # latter two are below MIN_TURN_SECONDS, so only the 4 s piece qualifies.
    assert voiceembed.centroid_segments(turns, "A") == [(0.0, 4.0)]
    assert voiceembed.centroid_segments(turns, "B") == [(5.0, 6.0)]


def test_centroid_segments_fall_back_to_half_second_pieces():
    turns = [
        (0.0, 1.5, "A"),
        (1.5, 4.0, "B"),
        (4.0, 4.3, "A"),
        (5.0, 5.8, "A"),
    ]

    assert voiceembed.centroid_segments(turns, "A") == [(0.0, 1.5), (5.0, 5.8)]


def test_centroid_segments_leave_speakers_without_usable_audio_out():
    turns = [(0.0, 0.4, "A"), (0.0, 10.0, "B")]

    assert voiceembed.centroid_segments(turns, "A") == []


def test_centroid_segments_keep_the_longest_twenty_in_time_order():
    turns = [
        (index * 10.0, index * 10.0 + 2.0 + index * 0.1, "A") for index in range(30)
    ]

    segments = voiceembed.centroid_segments(turns, "A")

    assert len(segments) == voiceembed.MAX_SEGMENTS_PER_SPEAKER
    assert segments == sorted(segments)
    assert segments[0][0] == 100.0  # turns 10..29 are the longest


def test_window_bounds_split_long_segments_into_equal_windows():
    rate = voiceembed.SAMPLE_RATE
    assert voiceembed.window_bounds(0, 10 * rate) == [(0, 10 * rate)]
    windows = voiceembed.window_bounds(rate, rate + 25 * rate)
    assert len(windows) == 3
    assert windows[0][0] == rate and windows[-1][1] == 26 * rate
    lengths = [high - low for low, high in windows]
    assert max(lengths) - min(lengths) <= 1
    assert max(lengths) <= voiceembed.WINDOW_SECONDS * rate


def test_weighted_unit_mean_weights_by_duration_and_renormalizes():
    vector = voiceembed.weighted_unit_mean(
        [[2.0, 0.0], [0.0, 5.0]],
        [3.0, 1.0],
    )

    expected = np.array([0.75, 0.25]) / math.hypot(0.75, 0.25)
    assert vector == pytest.approx(expected)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_weighted_unit_mean_refuses_bad_weights():
    with pytest.raises(voiceembed.VoiceEmbeddingError):
        voiceembed.weighted_unit_mean([[1.0, 0.0]], [0.0])
    with pytest.raises(voiceembed.VoiceEmbeddingError):
        voiceembed.weighted_unit_mean([[1.0, 0.0]], [1.0, 2.0])


# --------------------------------------------------------------------------
# Inference contract (fake network)
# --------------------------------------------------------------------------


def test_embed_segments_runs_one_unbatched_forward_per_window(monkeypatch):
    network = _install_fake_embedder(monkeypatch)
    rate = voiceembed.SAMPLE_RATE
    waveform = np.linspace(-0.5, 0.5, 40 * rate, dtype=np.float32)

    rows = voiceembed.embed_segments(
        waveform,
        [(0.0, 0.5), (1.0, 4.0), (5.0, 30.0)],
        voiceembed.REDIMNET2_B6,
    )

    assert rows.shape == (3, 192)
    assert np.linalg.norm(rows, axis=1) == pytest.approx(np.ones(3))
    shapes = [tuple(sample.shape) for sample in network.inputs]
    # 0.5 s is repeated up to min_seconds (1 s), 3 s is one pass, 25 s is
    # three equal windows -- every forward sees a batch of exactly one.
    assert shapes[0] == (1, rate)
    assert shapes[1] == (1, 3 * rate)
    assert len(shapes) == 5
    assert all(shape[0] == 1 for shape in shapes)
    assert sum(shape[1] for shape in shapes[2:]) == 25 * rate


def test_short_segments_repeat_speech_instead_of_padding_silence(monkeypatch):
    network = _install_fake_embedder(monkeypatch)
    rate = voiceembed.SAMPLE_RATE
    waveform = np.zeros(2 * rate, dtype=np.float32)
    waveform[: rate // 2] = np.arange(rate // 2, dtype=np.float32) + 1.0

    voiceembed.embed_segments(waveform, [(0.0, 0.5)], voiceembed.REDIMNET2_B6)

    fed = network.inputs[0][0]
    assert fed.shape == (rate,)
    assert np.count_nonzero(fed) == rate
    assert fed[rate // 2] == fed[0]


def test_embed_segments_refuses_out_of_range_spans(monkeypatch):
    _install_fake_embedder(monkeypatch)
    waveform = np.ones(voiceembed.SAMPLE_RATE, dtype=np.float32)

    with pytest.raises(voiceembed.VoiceEmbeddingError, match="outside the audio"):
        voiceembed.embed_segments(waveform, [(2.0, 3.0)], voiceembed.REDIMNET2_B6)
    with pytest.raises(voiceembed.VoiceEmbeddingError, match="start < end"):
        voiceembed.embed_segments(waveform, [(0.5, 0.5)], voiceembed.REDIMNET2_B6)


def test_loaded_embedder_refuses_a_wrong_output_dimension():
    loaded = voiceembed.LoadedEmbedder(
        spec=voiceembed.REDIMNET2_B6,
        checkpoint_sha256=voiceembed.REDIMNET2_B6.sha256,
        network=_FakeNetwork(dim=256),
        device="cpu",
    )

    with pytest.raises(voiceembed.VoiceEmbeddingError, match="256-dim"):
        loaded.embed_samples(np.ones(16_000, dtype=np.float32))


def test_speaker_centroids_are_duration_weighted_unit_vectors(monkeypatch):
    rate = voiceembed.SAMPLE_RATE
    waveform = np.zeros(20 * rate, dtype=np.float32)
    waveform[0 : 3 * rate] = 0.9  # A, 3 s, loud
    waveform[5 * rate : 7 * rate] = 0.1  # A, 2 s, quiet
    waveform[10 * rate : 14 * rate] = 0.5  # B, 4 s
    _install_fake_embedder(monkeypatch)
    turns = [
        (0.0, 3.0, "SPEAKER_00"),
        (5.0, 7.0, "SPEAKER_00"),
        (10.0, 14.0, "SPEAKER_01"),
        (15.0, 15.2, "SPEAKER_02"),
    ]

    centroids = voiceembed.speaker_centroids(waveform, turns, voiceembed.REDIMNET2_B6)

    assert set(centroids) == {"SPEAKER_00", "SPEAKER_01"}
    for vector in centroids.values():
        validate_vector(vector, dim=192)
    loud = np.array([0.9, 1.0]) / math.hypot(0.9, 1.0)
    quiet = np.array([0.1, 1.0]) / math.hypot(0.1, 1.0)
    mixed = (3.0 * loud + 2.0 * quiet) / 5.0
    mixed /= np.linalg.norm(mixed)
    assert centroids["SPEAKER_00"][:2] == pytest.approx(list(mixed))


def test_decoupled_provenance_replaces_only_the_embedding_space():
    base = {
        "diarization_model": "pyannote/speaker-diarization-community-1",
        "outer_config_sha256": "a" * 64,
        "embedding_model": "pyannote/embedding@x#subfolder=embedding",
        "embedding_checkpoint": "b" * 64,
        "embedding_dim": "unresolved",
        "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
        "pyannote_version": "4.0.7",
        "torch_version": "2.11.0",
    }

    provenance = voiceembed.decoupled_provenance(
        base,
        voiceembed.REDIMNET2_B6,
        checkpoint_sha256=voiceembed.REDIMNET2_B6.sha256,
    )

    assert provenance == {
        **base,
        "embedding_lane": "decoupled",
        "embedding_model": "redimnet2-b6-vb2-vox2-cnc2-lm",
        "embedding_checkpoint": voiceembed.REDIMNET2_B6.sha256,
        "embedding_dim": 192,
        "embedding_recipe": "centroid-v1",
    }
    assert base["embedding_dim"] == "unresolved"


def test_capture_voiceprints_reads_the_diarized_wav(tmp_path, monkeypatch):
    import soundfile as sf

    rate = voiceembed.SAMPLE_RATE
    wav = tmp_path / "speech.wav"
    signal = np.zeros(6 * rate, dtype=np.float32)
    signal[: 3 * rate] = 0.25
    sf.write(wav, signal, rate)
    _install_fake_embedder(monkeypatch)
    base = {"audio": {"separated": False, "normalized": False, "sample_rate": rate}}

    captured = voiceembed.capture_voiceprints(
        wav,
        [(0.0, 3.0, "SPEAKER_00")],
        voiceembed.REDIMNET2_B6,
        base,
    )

    assert captured is not None
    centroids, provenance = captured
    assert set(centroids) == {"SPEAKER_00"}
    assert provenance["embedding_lane"] == "decoupled"
    assert (
        voiceembed.capture_voiceprints(wav, [], voiceembed.REDIMNET2_B6, base) is None
    )


# --------------------------------------------------------------------------
# Checkpoint acquisition and verification
# --------------------------------------------------------------------------


def _spec_for(payload: bytes, **changes) -> voiceembed.EmbedderSpec:
    values = {
        "name": "test-embedder",
        "languages": None,
        "embedding_dim": 16,
        "sample_rate": voiceembed.SAMPLE_RATE,
        "min_seconds": 1.0,
        "suggest": 0.5,
        "margin": 0.05,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "filename": "test.pt",
        "checkpoint_env": "VOXWEAVE_TEST_EMBEDDER_CKPT",
        "loader": lambda _state: (_ for _ in ()).throw(AssertionError("unused")),
        "url": "https://example.invalid/test.pt",
        "cache_subdir": "test-embedder",
    }
    values.update(changes)
    return voiceembed.EmbedderSpec(**values)


def test_env_override_names_an_explicit_local_checkpoint(tmp_path, monkeypatch):
    spec = _spec_for(b"weights")
    local = tmp_path / "local.pt"
    local.write_bytes(b"weights")
    monkeypatch.setenv(spec.checkpoint_env, str(local))
    monkeypatch.setattr(
        voiceembed,
        "_download_url",
        lambda *_a: pytest.fail("an explicit checkpoint must not download"),
    )

    assert voiceembed.checkpoint_path(spec) == local
    monkeypatch.setenv(spec.checkpoint_env, str(tmp_path / "missing.pt"))
    with pytest.raises(voiceembed.VoiceEmbeddingError, match="is not a file"):
        voiceembed.checkpoint_path(spec)


class _FakeResponse:
    """A streaming HTTP response: hands out ``payload`` in small reads."""

    def __init__(self, payload: bytes, *, fail_after: int | None = None) -> None:
        self.payload = payload
        self.offset = 0
        self.fail_after = fail_after
        self.reads: list[int] = []
        self.closed = False

    def read(self, size: int) -> bytes:
        self.reads.append(size)
        if self.fail_after is not None and self.offset >= self.fail_after:
            raise TimeoutError("timed out")
        chunk = self.payload[self.offset : self.offset + 3]
        self.offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.closed = True


def _serve(monkeypatch, response):
    calls: list[tuple[str, float]] = []

    def fake_open(url, timeout):
        calls.append((url, timeout))
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(voiceembed, "_open_url", fake_open)
    return calls


def _cache_leftovers(root: Path) -> list[str]:
    return sorted(path.name for path in root.rglob("*") if path.is_file())


def test_release_asset_streams_into_the_audio_cache_with_a_timeout(
    tmp_path, monkeypatch
):
    spec = _spec_for(b"pinned weights")
    response = _FakeResponse(b"pinned weights")
    calls = _serve(monkeypatch, response)
    monkeypatch.setattr(voiceembed.config, "AUDIO_CACHE", str(tmp_path / "audio"))

    path = voiceembed.checkpoint_path(spec)

    assert path == tmp_path / "audio" / "test-embedder" / "test.pt"
    assert path.read_bytes() == b"pinned weights"
    assert calls == [(spec.url, voiceembed.DOWNLOAD_TIMEOUT_SECONDS)]
    assert 0 < voiceembed.DOWNLOAD_TIMEOUT_SECONDS <= 120
    assert len(response.reads) > 2  # streamed, not slurped
    assert response.closed
    assert _cache_leftovers(tmp_path / "audio") == ["test.pt"]
    # A cached asset is reused without touching the network again.
    assert voiceembed.checkpoint_path(spec) == path
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("served", "reason"),
    [
        (b"pinned weightz", "SHA-256"),  # same size, other bytes
        (b"pinned weights and more", "more than the pinned"),
        (b"pinned", "stopped after 6 of 14 bytes"),
    ],
)
def test_a_foreign_download_never_lands_in_the_cache(
    tmp_path, monkeypatch, served, reason
):
    spec = _spec_for(b"pinned weights")
    _serve(monkeypatch, _FakeResponse(served))
    monkeypatch.setattr(voiceembed.config, "AUDIO_CACHE", str(tmp_path / "audio"))

    with pytest.raises(voiceembed.VoiceEmbeddingError) as caught:
        voiceembed.checkpoint_path(spec)

    assert reason in str(caught.value)
    assert spec.checkpoint_env in str(caught.value)
    assert _cache_leftovers(tmp_path / "audio") == []


def test_a_stalled_download_fails_with_the_manual_route_and_no_partial_file(
    tmp_path, monkeypatch
):
    spec = _spec_for(b"pinned weights")
    _serve(monkeypatch, _FakeResponse(b"pinned weights", fail_after=6))
    monkeypatch.setattr(voiceembed.config, "AUDIO_CACHE", str(tmp_path / "audio"))

    with pytest.raises(voiceembed.VoiceEmbeddingError) as caught:
        voiceembed.checkpoint_path(spec)

    message = str(caught.value)
    assert "could not download voiceprint model test-embedder" in message
    assert "timed out" in message
    assert spec.checkpoint_env in message
    assert _cache_leftovers(tmp_path / "audio") == []


def test_an_unreachable_host_names_the_manual_route(tmp_path, monkeypatch):
    import urllib.error

    spec = _spec_for(b"pinned weights")
    _serve(monkeypatch, urllib.error.URLError("Name or service not known"))
    monkeypatch.setattr(voiceembed.config, "AUDIO_CACHE", str(tmp_path / "audio"))

    with pytest.raises(voiceembed.VoiceEmbeddingError) as caught:
        voiceembed.checkpoint_path(spec)
    assert "Name or service not known" in str(caught.value)
    assert spec.source in str(caught.value)


def test_open_url_passes_the_timeout_to_urlopen(monkeypatch):
    seen = {}

    def fake_urlopen(request, *, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return "response"

    monkeypatch.setattr(voiceembed.urllib.request, "urlopen", fake_urlopen)

    assert voiceembed._open_url("https://example.invalid/x.pt", 12.5) == "response"
    assert seen == {"url": "https://example.invalid/x.pt", "timeout": 12.5}


def test_hub_checkpoint_downloads_the_pinned_revision_into_the_audio_cache(
    tmp_path, monkeypatch
):
    calls = []

    def fake_hub(repo, filename, *, revision, cache_dir):
        calls.append((repo, filename, revision, cache_dir))
        return str(tmp_path / "hub" / filename)

    monkeypatch.setattr(voiceembed.config, "AUDIO_CACHE", str(tmp_path / "audio"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hub)

    path = voiceembed.checkpoint_path(voiceembed.ANIME_VA)

    assert path == tmp_path / "hub" / "embedding_model.pth"
    assert calls == [
        (
            voiceembed.ANIME_VA.hf_repo,
            "embedding_model.pth",
            voiceembed.ANIME_VA.hf_revision,
            str(tmp_path / "audio"),
        )
    ]


def test_hash_mismatch_refuses_before_deserializing(tmp_path, monkeypatch):
    spec = _spec_for(b"expected bytes")
    local = tmp_path / "tampered.pt"
    local.write_bytes(b"tampered bytes")  # same size, other content
    monkeypatch.setenv(spec.checkpoint_env, str(local))
    monkeypatch.setattr(
        "torch.load", lambda *_a, **_k: pytest.fail("must not deserialize")
    )

    with pytest.raises(voiceembed.VoiceEmbeddingError, match="has SHA-256"):
        voiceembed._construct(spec)


def test_a_wrong_size_checkpoint_is_refused_before_it_is_read(tmp_path, monkeypatch):
    spec = _spec_for(b"expected bytes")
    huge = tmp_path / "huge.pt"
    with open(huge, "wb") as handle:
        handle.truncate(4 * 1024**3)  # sparse: nothing is allocated or read
    reads = []
    real_open = open

    def spy_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if Path(path) == huge:
            original_read = handle.read

            def read(*read_args):
                reads.append(read_args)
                return original_read(*read_args)

            handle.read = read
        return handle

    monkeypatch.setattr("builtins.open", spy_open)

    with pytest.raises(voiceembed.VoiceEmbeddingError) as caught:
        voiceembed.verify_checkpoint(spec, huge)

    assert f"is {4 * 1024**3} bytes" in str(caught.value)
    assert "pinned to a 14-byte file" in str(caught.value)
    assert spec.checkpoint_env in str(caught.value)
    assert reads == []


def test_verified_reads_return_exactly_the_pinned_bytes(tmp_path):
    payload = bytes(range(256)) * 9000  # several read chunks
    spec = _spec_for(payload)
    local = tmp_path / "weights.pt"
    local.write_bytes(payload)

    assert voiceembed.verify_checkpoint(spec, local) is None
    verified = voiceembed.read_verified_checkpoint(spec, local)
    assert bytes(verified) == payload


# --------------------------------------------------------------------------
# Prefetch at the start of a run
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("choice", "language", "expected"),
    [
        (None, None, ("redimnet2-b6-vb2-vox2-cnc2-lm", "anime-va-ecapa-gn")),
        (None, "ja", ("anime-va-ecapa-gn",)),
        (None, "en", ("redimnet2-b6-vb2-vox2-cnc2-lm",)),
        ("anime-va", None, ("anime-va-ecapa-gn",)),
        ("redimnet2", "ja", ("redimnet2-b6-vb2-vox2-cnc2-lm",)),
        ("pyannote", None, ()),
    ],
)
def test_prefetch_covers_every_route_the_run_may_take(choice, language, expected):
    specs = voiceembed.prefetch_specs(choice, language)

    assert tuple(spec.name for spec in specs) == expected
    if language is not None and specs:
        assert specs == (voiceembed.resolve_voiceprint_model(choice, language),)


@pytest.mark.real_voiceprint_prefetch
def test_prefetch_downloads_and_verifies_each_needed_checkpoint(monkeypatch):
    fetched = []
    verified = []

    def fake_path(spec):
        fetched.append(spec.name)
        return Path(f"/weights/{spec.name}")

    monkeypatch.setattr(voiceembed, "checkpoint_path", fake_path)
    monkeypatch.setattr(
        voiceembed,
        "verify_checkpoint",
        lambda spec, path: verified.append((spec.name, path)),
    )

    specs = voiceembed.prefetch_checkpoints(None, None)

    assert specs == (voiceembed.REDIMNET2_B6, voiceembed.ANIME_VA)
    assert fetched == [voiceembed.REDIMNET2_B6.name, voiceembed.ANIME_VA.name]
    assert verified == [(name, Path(f"/weights/{name}")) for name in fetched]


@pytest.mark.real_voiceprint_prefetch
def test_prefetch_surfaces_a_download_failure(tmp_path, monkeypatch):
    import urllib.error

    _serve(monkeypatch, urllib.error.URLError("network is unreachable"))
    monkeypatch.setattr(voiceembed.config, "AUDIO_CACHE", str(tmp_path / "audio"))

    with pytest.raises(voiceembed.VoiceEmbeddingError, match="network is unreachable"):
        voiceembed.prefetch_checkpoints("redimnet2", None)
    with pytest.raises(ValueError, match="unknown voiceprint model"):
        voiceembed.prefetch_checkpoints("ecapa", None)


def _saved(value: object) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def test_construct_loads_weights_only_and_checks_the_declared_dim(
    tmp_path, monkeypatch
):
    import torch

    payload = _saved({"weight": torch.zeros(2), "config": {"dim": 16}})
    seen = []

    def loader(state):
        seen.append(state)
        return torch.nn.Identity(), 16

    spec = _spec_for(payload, loader=loader)
    local = tmp_path / "weights.pt"
    local.write_bytes(payload)
    monkeypatch.setenv(spec.checkpoint_env, str(local))
    monkeypatch.setattr(voiceembed.runtime, "get_device", lambda: "cpu")
    real_load = torch.load
    load_kwargs = []

    def spy_load(*args, **kwargs):
        load_kwargs.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr("torch.load", spy_load)

    loaded = voiceembed._construct(spec)

    assert loaded.checkpoint_sha256 == spec.sha256
    assert load_kwargs == [{"map_location": "cpu", "weights_only": True}]
    assert seen[0]["config"] == {"dim": 16}

    wrong = _spec_for(payload, loader=lambda _state: (torch.nn.Identity(), 32))
    monkeypatch.setenv(wrong.checkpoint_env, str(local))
    with pytest.raises(voiceembed.VoiceEmbeddingError, match="declares 32-dim"):
        voiceembed._construct(wrong)


def test_resident_embedder_is_keyed_by_name_and_checkpoint(monkeypatch):
    built = []

    def construct(spec):
        built.append(spec.name)
        return voiceembed.LoadedEmbedder(
            spec=spec,
            checkpoint_sha256=spec.sha256,
            network=_FakeNetwork(),
            device="cpu",
        )

    monkeypatch.setattr(voiceembed, "_resident", None)
    monkeypatch.setattr(voiceembed, "_construct", construct)

    first = voiceembed.get_embedder(voiceembed.REDIMNET2_B6)
    assert voiceembed.get_embedder(voiceembed.REDIMNET2_B6) is first
    voiceembed.get_embedder(voiceembed.ANIME_VA)
    assert built == [voiceembed.REDIMNET2_B6.name, voiceembed.ANIME_VA.name]
    voiceembed.release()
    assert voiceembed._resident is None


# --------------------------------------------------------------------------
# Vendored networks with random weights (architecture + checkpoint layout)
# --------------------------------------------------------------------------


def test_redimnet2_b6_architecture_round_trips_its_release_layout():
    import torch

    from voxweave import voiceembed_models
    from voxweave.vendor.redimnet2 import ReDimNet2Wrap

    torch.manual_seed(0)
    reference = ReDimNet2Wrap(**B6_MODEL_CONFIG)
    checkpoint = {
        "model_config": B6_MODEL_CONFIG,
        "state_dict": reference.state_dict(),
    }

    network, declared = voiceembed_models.build_redimnet2(checkpoint)
    network.eval()
    with torch.inference_mode():
        output = network(torch.randn(1, 24_000))

    assert declared == 192
    assert tuple(output.shape) == (1, 192)
    assert torch.isfinite(output).all()


def test_anime_va_network_swaps_every_batchnorm_and_scales_the_waveform(
    monkeypatch,
):
    import torch

    from voxweave import voiceembed_models

    torch.manual_seed(0)
    reference = voiceembed_models.AnimeVoiceActorEmbedder()
    state = reference.state_dict()
    assert "fbank.compute_deltas.kernel" in state
    assert not any(
        isinstance(module, torch.nn.BatchNorm1d) for module in reference.modules()
    )
    groups = [
        module
        for module in reference.modules()
        if isinstance(module, torch.nn.GroupNorm)
    ]
    assert groups and all(
        module.num_groups == voiceembed_models.ANIME_VA_NORM_GROUPS for module in groups
    )

    network, declared = voiceembed_models.build_anime_va(state)
    network.eval()
    seen = []
    original_fbank = network.fbank.forward

    def spy_fbank(wave):
        seen.append(float(wave.abs().max()))
        return original_fbank(wave)

    monkeypatch.setattr(network.fbank, "forward", spy_fbank)
    wave = torch.zeros(1, 24_000)
    wave[0, 0] = 0.5
    with torch.inference_mode():
        output = network(wave)
        network(wave * 4.0)  # clipping input: peak-normalized first

    assert declared == 192
    assert tuple(output.shape) == (1, 192)
    assert seen == pytest.approx([0.5 * 32768.0, 32768.0])


def test_anime_va_builder_refuses_a_foreign_state_dict():
    from voxweave import voiceembed_models

    with pytest.raises(RuntimeError, match="Missing key"):
        voiceembed_models.build_anime_va({"backbone.unrelated": 1})
    with pytest.raises(voiceembed_models.CheckpointLayoutError):
        voiceembed_models.build_redimnet2({"state_dict": {}})
