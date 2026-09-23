"""Voiceprint speaker clustering: knob, diarize hook, fallback, provenance, wiring.

Zero GPU: pyannote, the ReDimNet2 embedder and (where the recipe itself is not
under test) ``speakercluster.cluster_turns`` are replaced by fakes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from voxweave import (
    backend,
    chunking,
    config,
    diarize,
    pipeline,
    songdet,
    speakercluster,
    voiceembed,
)
from voxweave.voicebase import canonical_json_bytes

MODEL = "pyannote/speaker-diarization-community-1"
ENV = "VOXWEAVE_DIARIZE_CLUSTERING"
OUTER_SHA = "f" * 64
ATTESTED = "9" * 64

# Raw pyannote turns: the 0.1 s "A" fragment sits inside "B" and is removed by
# smoothing, so seeing it in cluster_turns proves the hook runs before smoothing.
RAW = [
    (0.0, 2.0, "B"),
    (0.5, 0.6, "A"),
    (2.5, 4.0, "A"),
    (4.1, 5.0, "B"),
]
PYANNOTE_TURNS = [(0.0, 2.0, "B"), (2.5, 4.0, "A"), (4.1, 5.0, "B")]


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.delenv("VOXWEAVE_DIARIZE_MODEL", raising=False)
    monkeypatch.setattr(diarize, "_pipeline", None)
    monkeypatch.setattr(diarize, "_pipeline_model", None)


def _write_config(tmp_path: Path, text: str) -> None:
    (tmp_path / "voxweave.conf").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# speakercluster placeholder contract
# --------------------------------------------------------------------------


def test_placeholder_relabels_by_first_appearance_and_unions_overlaps():
    def never(_spans):
        raise AssertionError("the placeholder does not embed")

    result = speakercluster.cluster_turns(
        [
            (3.0, 4.0, "X"),
            (0.0, 2.0, "Y"),
            (1.5, 2.5, "Y"),  # overlaps the previous Y turn -> unioned
            (2.5, 3.0, "Y"),  # touches, does not overlap -> kept apart
            (1.0, 1.2, "X"),
        ],
        never,
        min_speakers=1,
        max_speakers=4,
    )

    assert result.turns == [
        (0.0, 2.5, "SPEAKER_00"),
        (1.0, 1.2, "SPEAKER_01"),
        (2.5, 3.0, "SPEAKER_00"),
        (3.0, 4.0, "SPEAKER_01"),
    ]
    assert result.audit["recipe"] == speakercluster.RECIPE
    assert result.audit["turns_in"] == 5
    assert result.audit["speakers"] == 2
    canonical_json_bytes(result.audit)  # strict-JSON serializable
    canonical_json_bytes(asdict(speakercluster.ClusteringParams()))


def test_placeholder_accepts_empty_input():
    result = speakercluster.cluster_turns([], lambda _spans: np.zeros((0, 192)))
    assert result.turns == []
    assert result.audit["speakers"] == 0


# --------------------------------------------------------------------------
# Knob resolution
# --------------------------------------------------------------------------


def test_builtin_default_is_the_single_constant(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV, raising=False)
    assert config.resolve_diarize_clustering() == config.DEFAULT_DIARIZE_CLUSTERING
    assert config.DEFAULT_DIARIZE_CLUSTERING in config.DIARIZE_CLUSTERING_CHOICES


def test_conf_env_cli_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV, raising=False)
    _write_config(tmp_path, '[diarize]\nclustering = "voiceprint"\n')
    assert config.resolve_diarize_clustering() == "voiceprint"

    monkeypatch.setenv(ENV, "pyannote")
    assert config.resolve_diarize_clustering() == "pyannote"

    monkeypatch.setenv(ENV, "voiceprint")
    assert config.resolve_diarize_clustering("pyannote") == "pyannote"

    _write_config(tmp_path, '[diarize]\nclustering = "pyannote"\n')
    assert config.resolve_diarize_clustering() == "voiceprint"


def test_blank_values_fall_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_config(tmp_path, '[diarize]\nclustering = "voiceprint"\n')
    monkeypatch.setenv(ENV, "  ")
    assert config.resolve_diarize_clustering("") == "voiceprint"


def test_values_are_case_insensitive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV, " VoicePrint ")
    assert config.resolve_diarize_clustering() == "voiceprint"
    assert config.resolve_diarize_clustering("PYANNOTE") == "pyannote"


@pytest.mark.parametrize(
    ("layer", "source"),
    [
        ("cli", "--speaker-clustering"),
        ("env", f"environment {ENV}"),
        ("conf", "config [diarize].clustering"),
    ],
)
def test_unknown_value_names_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str, source: str
):
    monkeypatch.delenv(ENV, raising=False)
    cli_value = None
    if layer == "cli":
        cli_value = "wespeaker"
    elif layer == "env":
        monkeypatch.setenv(ENV, "wespeaker")
    else:
        _write_config(tmp_path, '[diarize]\nclustering = "wespeaker"\n')

    with pytest.raises(ValueError) as excinfo:
        config.resolve_diarize_clustering(cli_value)

    message = str(excinfo.value)
    assert message.startswith(source)
    assert "'wespeaker'" in message
    assert "voiceprint, pyannote" in message


def test_wrong_conf_type_is_warned_and_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.delenv(ENV, raising=False)
    _write_config(tmp_path, "[diarize]\nclustering = 1\n")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert config.resolve_diarize_clustering() == config.DEFAULT_DIARIZE_CLUSTERING
    assert "[diarize].clustering has wrong type" in caplog.text


def test_conf_template_documents_the_knob_and_its_default():
    template = config._TEMPLATE
    default = config.DEFAULT_DIARIZE_CLUSTERING
    assert f'# clustering = "{default}"' in template
    assert "VOXWEAVE_DIARIZE_CLUSTERING" in template
    assert "@DIARIZE_CLUSTERING@" not in template


# --------------------------------------------------------------------------
# diarize hook
# --------------------------------------------------------------------------


class _Segment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _Annotation:
    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, *, yield_label: bool = False):
        for index, (start, end, label) in enumerate(self._turns):
            row = (_Segment(start, end), f"track-{index}", label)
            yield row if yield_label else row[:2]

    def labels(self) -> list[str]:
        return list(dict.fromkeys(label for _s, _e, label in self._turns))


class _FakePipeline:
    def __init__(self, turns: list[tuple[float, float, str]], *, embeddings=False):
        annotation = _Annotation(turns)
        self.result = SimpleNamespace(
            speaker_diarization=annotation,
            exclusive_speaker_diarization=annotation,
            speaker_embeddings=(
                np.eye(2, 16, dtype=np.float32) if embeddings else None
            ),
        )
        # Local provenance only: no config download for the outer digest.
        setattr(self, diarize._OUTER_CONFIG_ATTR, OUTER_SHA)
        setattr(self, diarize._EMBEDDING_MODEL_ATTR, "example/embedding@rev")
        self.calls: list[dict[str, object]] = []

    def __call__(self, _audio, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _signal(seconds: float, sample_rate: int) -> np.ndarray:
    count = int(seconds * sample_rate)
    return (0.1 * np.sin(np.linspace(0.0, 400.0, count))).astype(np.float32)


def _wav(tmp_path: Path, *, sample_rate: int = 16000) -> Path:
    path = tmp_path / f"speech-{sample_rate}.wav"
    sf.write(path, _signal(6.0, sample_rate), sample_rate, subtype="FLOAT")
    return path


def _install_pipeline(monkeypatch, *, embeddings=False) -> _FakePipeline:
    fake = _FakePipeline(RAW, embeddings=embeddings)
    monkeypatch.setattr(diarize, "_get_pipeline", lambda _token, _model: fake)
    monkeypatch.setattr(diarize, "_package_version", lambda _name: "4.0.7")
    return fake


class _EmbedderSpy:
    """Fake ReDimNet2: unit rows, an attested checkpoint, counted releases."""

    def __init__(self, monkeypatch, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[np.ndarray, list[tuple[float, float]], object]] = []
        self.released = 0
        self.fail = fail
        monkeypatch.setattr(voiceembed, "embed_segments", self.embed_segments)
        monkeypatch.setattr(
            voiceembed,
            "get_embedder",
            lambda _spec: SimpleNamespace(checkpoint_sha256=ATTESTED),
        )
        monkeypatch.setattr(voiceembed, "release", self.release)

    def embed_segments(self, waveform, spans, spec):
        if self.fail is not None:
            raise self.fail
        self.calls.append((waveform, list(spans), spec))
        rows = np.zeros((len(spans), 192))
        rows[:, 0] = 1.0
        return rows

    def release(self) -> None:
        self.released += 1


class _ClusterSpy:
    """Fake recipe: embeds two anchors and returns a fixed regrouping."""

    turns = [
        (0.0, 2.0, "SPEAKER_00"),
        (0.5, 0.6, "SPEAKER_01"),  # contained + short: smoothing removes it
        (2.5, 5.0, "SPEAKER_01"),
    ]
    audit = {
        "recipe": "test-recipe",
        "anchors": 2,
        "abstained_turns": 1,
        "abstained_seconds": 0.9,
        "anchor_seconds": {"SPEAKER_00": 2.0, "SPEAKER_01": 1.5},
    }

    def __init__(self, monkeypatch, *, result_turns=None, error=None) -> None:
        self.calls: list[dict[str, object]] = []
        self.rows: list[np.ndarray] = []
        self.result_turns = self.turns if result_turns is None else result_turns
        self.error = error
        monkeypatch.setattr(speakercluster, "cluster_turns", self)

    def __call__(self, turns, embed, *, min_speakers, max_speakers, params):
        self.calls.append(
            {
                "turns": list(turns),
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "params": params,
            }
        )
        if self.error is not None:
            raise self.error
        self.rows.append(embed([(0.0, 2.0), (2.5, 4.0)]))
        return speakercluster.ClusteringResult(
            turns=list(self.result_turns), audit=dict(self.audit)
        )


def _forbid_voiceprint_stage(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the voiceprint stage must not run")

    monkeypatch.setattr(speakercluster, "cluster_turns", forbidden)
    monkeypatch.setattr(voiceembed, "embed_segments", forbidden)
    monkeypatch.setattr(voiceembed, "get_embedder", forbidden)
    monkeypatch.setattr(voiceembed, "release", forbidden)


def _expected_base_provenance() -> dict[str, object]:
    import torch

    return {
        "diarization_model": MODEL,
        "outer_config_sha256": OUTER_SHA,
        "embedding_model": "example/embedding@rev",
        "embedding_checkpoint": "unresolved",
        "embedding_dim": "unresolved",
        "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
        "pyannote_version": "4.0.7",
        "torch_version": str(torch.__version__),
    }


@pytest.mark.parametrize("via", ["argument", "environment"])
def test_pyannote_clustering_is_byte_identical_to_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, via: str
):
    fake = _install_pipeline(monkeypatch)
    _forbid_voiceprint_stage(monkeypatch)
    monkeypatch.setenv(ENV, "pyannote")
    clustering = "pyannote" if via == "argument" else None

    result = diarize.diarize_turns(
        _wav(tmp_path), token="hf_test", model=MODEL, clustering=clustering
    )

    # Exactly what the code recorded before the knob existed: smoothed raw
    # turns, and the eight provenance fields with no clustering block.
    assert result.turns == PYANNOTE_TURNS == diarize._smooth_turns(RAW)
    assert result.centroids is None
    assert result.provenance == _expected_base_provenance()
    assert canonical_json_bytes(result.provenance) == canonical_json_bytes(
        _expected_base_provenance()
    )
    assert fake.calls == [{}]


def test_voiceprint_clustering_runs_on_raw_turns_before_smoothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = _install_pipeline(monkeypatch)
    embedder = _EmbedderSpy(monkeypatch)
    cluster = _ClusterSpy(monkeypatch)
    wav = _wav(tmp_path)

    result = diarize.diarize_turns(
        wav,
        token="hf_test",
        model=MODEL,
        min_speakers=2,
        max_speakers=3,
        clustering="voiceprint",
    )

    (call,) = cluster.calls
    assert call["turns"] == RAW  # raw pyannote turns, fragment included
    assert call["min_speakers"] == 2 and call["max_speakers"] == 3
    assert fake.calls == [{"min_speakers": 2, "max_speakers": 3}]
    # The embedder sees the exact 16 kHz mono samples pyannote was given.
    ((waveform, spans, spec),) = embedder.calls
    assert spec is voiceembed.REDIMNET2_B6
    assert spans == [(0.0, 2.0), (2.5, 4.0)]
    assert waveform.ndim == 1 and waveform.dtype == np.float32
    np.testing.assert_array_equal(waveform, sf.read(wav, dtype="float32")[0])
    assert cluster.rows[0].shape == (2, 192)
    # Smoothing runs on the clustered turns.
    assert result.turns == [(0.0, 2.0, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_01")]
    assert embedder.released == 1


def test_voiceprint_provenance_records_the_clustering_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_pipeline(monkeypatch)
    _EmbedderSpy(monkeypatch)
    _ClusterSpy(monkeypatch)

    result = diarize.diarize_turns(
        _wav(tmp_path), token="hf_test", model=MODEL, clustering="voiceprint"
    )

    provenance = dict(result.provenance)
    block = provenance.pop("clustering")
    assert provenance == _expected_base_provenance()
    assert block == {
        "method": "voiceprint",
        "recipe": "test-recipe",
        "embedder": voiceembed.REDIMNET2_B6.name,
        "embedder_checkpoint": ATTESTED,
        "params": asdict(speakercluster.ClusteringParams()),
        # Scalar counts only; the nested per-cluster detail stays in the log.
        "audit": {
            "recipe": "test-recipe",
            "anchors": 2,
            "abstained_turns": 1,
            "abstained_seconds": 0.9,
        },
    }
    canonical_json_bytes(result.provenance)


def test_real_placeholder_recipe_is_wired_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_pipeline(monkeypatch)
    embedder = _EmbedderSpy(monkeypatch)
    monkeypatch.setenv(ENV, "voiceprint")  # env layer, no argument

    result = diarize.diarize_turns(_wav(tmp_path), token="hf_test", model=MODEL)

    # First appearance: B -> SPEAKER_00, A -> SPEAKER_01.
    assert result.turns == [
        (0.0, 2.0, "SPEAKER_00"),
        (2.5, 4.0, "SPEAKER_01"),
        (4.1, 5.0, "SPEAKER_00"),
    ]
    block = result.provenance["clustering"]
    assert block["method"] == "voiceprint"
    assert block["recipe"] == speakercluster.RECIPE
    # Never embedded: the pinned identity is recorded.
    assert block["embedder_checkpoint"] == voiceembed.REDIMNET2_B6.sha256
    assert embedder.calls == []
    assert embedder.released == 1


def test_waveform_at_another_rate_is_resampled_to_16k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_pipeline(monkeypatch)
    embedder = _EmbedderSpy(monkeypatch)
    _ClusterSpy(monkeypatch)

    diarize.diarize_turns(
        _wav(tmp_path, sample_rate=8000),
        token="hf_test",
        model=MODEL,
        clustering="voiceprint",
    )

    ((waveform, _spans, _spec),) = embedder.calls
    assert waveform.shape == (6 * 16000,)


def test_caller_can_keep_the_embedder_for_the_voiceprint_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_pipeline(monkeypatch)
    embedder = _EmbedderSpy(monkeypatch)
    _ClusterSpy(monkeypatch)

    diarize.diarize_turns(
        _wav(tmp_path),
        token="hf_test",
        model=MODEL,
        clustering="voiceprint",
        release_embedder=False,
    )

    assert embedder.released == 0


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("recipe-error", "RuntimeError: CUDA out of memory"),
        ("embedder-error", "VoiceEmbeddingError: could not download"),
        ("malformed-result", "ValueError: clustering returned an invalid span"),
        ("oversized-label", "ValueError: clustering returned an oversized label"),
    ],
)
def test_any_stage_failure_keeps_pyannote_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: str,
    reason: str,
):
    _install_pipeline(monkeypatch)
    embedder = _EmbedderSpy(
        monkeypatch,
        fail=(
            voiceembed.VoiceEmbeddingError("could not download voiceprint model")
            if case == "embedder-error"
            else None
        ),
    )
    _ClusterSpy(
        monkeypatch,
        error=RuntimeError("CUDA out of memory") if case == "recipe-error" else None,
        result_turns={
            "malformed-result": [(2.0, 1.0, "SPEAKER_00")],
            "oversized-label": [(0.0, 2.0, "S" * 65)],
        }.get(case),
    )

    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = diarize.diarize_turns(
            _wav(tmp_path), token="hf_test", model=MODEL, clustering="voiceprint"
        )

    assert result.turns == PYANNOTE_TURNS
    provenance = dict(result.provenance)
    block = provenance.pop("clustering")
    assert provenance == _expected_base_provenance()
    assert block["method"] == "pyannote"
    assert block["requested"] == "voiceprint"
    assert str(block["reason"]).startswith(reason)
    assert "keeping pyannote's speakers" in caplog.text
    assert embedder.released == 1


def test_fallback_reason_is_bounded_for_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_pipeline(monkeypatch)
    _EmbedderSpy(monkeypatch)
    _ClusterSpy(monkeypatch, error=RuntimeError("失败" * 400))

    result = diarize.diarize_turns(
        _wav(tmp_path), token="hf_test", model=MODEL, clustering="voiceprint"
    )

    reason = str(result.provenance["clustering"]["reason"])  # type: ignore[index]
    assert len(reason.encode("utf-8")) <= 300
    assert reason.endswith("...")


def test_legacy_voiceprint_lane_keeps_pyannote_clustering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _install_pipeline(monkeypatch, embeddings=True)
    _forbid_voiceprint_stage(monkeypatch)

    with caplog.at_level(logging.INFO, logger="voxweave"):
        result = diarize.diarize_turns(
            _wav(tmp_path),
            token="hf_test",
            model=MODEL,
            want_embeddings=True,
            clustering="voiceprint",
        )
    monkeypatch.setattr(diarize, "_pipeline", None)
    baseline = diarize.diarize_turns(
        _wav(tmp_path),
        token="hf_test",
        model=MODEL,
        want_embeddings=True,
        clustering="pyannote",
    )

    assert result.turns == baseline.turns == PYANNOTE_TURNS
    assert result.centroids == baseline.centroids
    assert set(result.centroids or {}) == {"A", "B"}
    provenance = dict(result.provenance)
    assert provenance.pop("clustering") == {
        "method": "pyannote",
        "requested": "voiceprint",
        "reason": "legacy pyannote voiceprint lane",
    }
    assert provenance == baseline.provenance
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("keep pyannote's clustering" in m for m in messages) == 1


def test_invalid_knob_fails_before_pyannote_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = _install_pipeline(monkeypatch)
    monkeypatch.setenv(ENV, "spectral")

    with pytest.raises(ValueError, match=f"environment {ENV}"):
        diarize.diarize_turns(_wav(tmp_path), token="hf_test", model=MODEL)
    assert fake.calls == []


# --------------------------------------------------------------------------
# pipeline threading
# --------------------------------------------------------------------------

UNIT = {"text": "hello", "start": 0.0, "end": 1.0}


def _stub_asr(tmp_path: Path, monkeypatch, *, language="English") -> Path:
    wav = tmp_path / "speech.wav"
    sf.write(wav, _signal(6.0, 16000), 16000, subtype="FLOAT")
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_a, **_k: wav)
    monkeypatch.setattr(
        pipeline,
        "vad_speech_segments",
        lambda *_a, **_k: [{"start": 0.0, "end": 1.0}],
    )
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_a, **_k: wav)
    monkeypatch.setattr(backend, "chunk_pass_count", lambda *_a, **_k: 2)
    monkeypatch.setattr(
        backend,
        "transcribe_chunks",
        lambda *_a, **_k: [(language, "hello", [dict(UNIT)])],
    )
    monkeypatch.setattr(backend, "release", lambda: None)
    monkeypatch.setattr(chunking, "release_silero_vad", lambda: None)
    monkeypatch.setattr(songdet, "release_model", lambda: None)
    return wav


def _stub_transcribe(tmp_path: Path, monkeypatch, *, language="English"):
    _stub_asr(tmp_path, monkeypatch, language=language)
    events: list[object] = []

    def fake_diarize(_wav_path, **kwargs):
        events.append(("diarize", kwargs["clustering"], kwargs["release_embedder"]))
        return diarize.DiarizationResult(
            turns=[(0.0, 3.0, "SPEAKER_00")],
            centroids=None,
            provenance={
                "diarization_model": MODEL,
                "outer_config_sha256": OUTER_SHA,
                "embedding_model": "example/embedding",
                "embedding_checkpoint": "e" * 64,
                "embedding_dim": 16,
                "audio": {
                    "separated": False,
                    "normalized": False,
                    "sample_rate": 16000,
                },
                "pyannote_version": "4.0.7",
                "torch_version": "test",
            },
        )

    monkeypatch.setattr(diarize, "diarize_turns", fake_diarize)
    monkeypatch.setattr(diarize, "release", lambda: events.append("pyannote released"))
    monkeypatch.setattr(
        voiceembed, "release", lambda: events.append("embedder released")
    )
    return events


@pytest.mark.parametrize(
    ("clustering", "language", "voiceprints", "keep"),
    [
        ("voiceprint", "English", True, True),  # capture reuses ReDimNet2
        ("voiceprint", "Japanese", True, False),  # capture uses anime-va
        ("voiceprint", "English", False, False),  # no capture
        ("pyannote", "English", True, False),
    ],
)
def test_transcribe_threads_the_knob_and_the_embedder_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clustering: str,
    language: str,
    voiceprints: bool,
    keep: bool,
):
    events = _stub_transcribe(tmp_path, monkeypatch, language=language)
    monkeypatch.setattr(
        voiceembed,
        "capture_voiceprints",
        lambda *_a, **_k: events.append("capture") or None,
    )
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    pipeline.transcribe(
        media,
        separate=False,
        diarize=True,
        voiceprints=voiceprints,
        speaker_clustering=clustering,
    )

    assert events[0] == ("diarize", clustering, not keep)
    assert events[1] == "pyannote released"
    if voiceprints:
        # The capture runs after pyannote is gone, and releases the embedder.
        assert events[2:4] == ["capture", "embedder released"]
    if keep:
        assert events[-1] == "embedder released"


def test_transcribe_reads_the_configured_knob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events = _stub_transcribe(tmp_path, monkeypatch)
    monkeypatch.setenv(ENV, "voiceprint")
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    pipeline.transcribe(media, separate=False, diarize=True)

    assert events[0] == ("diarize", "voiceprint", True)


def test_transcribe_rejects_a_bad_knob_before_audio_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ENV, "spectral")
    monkeypatch.setattr(
        pipeline,
        "decode_to_wav",
        lambda *_a, **_k: pytest.fail("must fail before decoding"),
    )
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    with pytest.raises(ValueError, match="unknown speaker clustering"):
        pipeline.transcribe(media, separate=False, diarize=True)


class _FakeNetwork:
    def __init__(self) -> None:
        self.forwards = 0

    def __call__(self, wave):
        import torch

        self.forwards += 1
        row = torch.zeros((wave.shape[0], 192), dtype=torch.float32)
        row[:, 0] = 1.0
        return row


def test_process_capture_reuses_the_clustering_embedder_and_keeps_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from voxweave import artifacts, voicematch
    from voxweave.voicebase import (
        load_voiceprints,
        media_fingerprint,
        validate_voiceprint_conjunction,
    )

    _stub_asr(tmp_path, monkeypatch)
    _install_pipeline(monkeypatch)
    _ClusterSpy(monkeypatch)
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_test_token")
    network = _FakeNetwork()
    constructed: list[str] = []

    def construct(spec):
        constructed.append(spec.name)
        return voiceembed.LoadedEmbedder(
            spec=spec, checkpoint_sha256=spec.sha256, network=network, device="cpu"
        )

    monkeypatch.setattr(voiceembed, "_construct", construct)
    monkeypatch.setattr(voiceembed, "_resident", None)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable media bytes")

    pipeline.process(
        media,
        separate=False,
        diarize=True,
        voiceprints=True,
        shot_snap=False,
        speaker_clustering="voiceprint",
    )

    # One ReDimNet2 load serves clustering (2 anchors) and the capture
    # (2 speakers), and nothing stays resident afterwards.
    assert constructed == [voiceembed.REDIMNET2_B6.name]
    assert network.forwards == 4
    assert voiceembed._resident is None
    sibling = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert sibling["speaker_turns"] == [
        [0.0, 2.0, "SPEAKER_00"],
        [2.5, 5.0, "SPEAKER_01"],
    ]
    sidecar, validated = load_voiceprints(artifacts.claim_paths(media).voiceprints)
    validate_voiceprint_conjunction(sidecar, sibling, media_fingerprint(media))
    assert set(validated.speakers) == {"SPEAKER_00", "SPEAKER_01"}
    provenance = dict(sidecar["provenance"])
    assert provenance["embedding_lane"] == "decoupled"
    block = provenance.pop("clustering")
    assert block["method"] == "voiceprint"
    assert block["embedder_checkpoint"] == voiceembed.REDIMNET2_B6.sha256
    assert voicematch.compatibility_equal(
        voicematch.build_compatibility_fingerprint(sidecar["provenance"]),
        voicematch.build_compatibility_fingerprint(provenance),
    )


def test_process_passes_the_knob_to_transcribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def fake_transcribe(_media, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    with pytest.raises(_Stop):
        pipeline.process(
            media,
            separate=False,
            diarize=True,
            shot_snap=False,
            speaker_clustering="voiceprint",
        )
    assert seen["speaker_clustering"] == "voiceprint"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def cli_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    def fake_process(_media, **kwargs):
        calls.append(kwargs)
        return tmp_path / "episode.vtt"

    monkeypatch.setattr(pipeline, "process", fake_process)
    monkeypatch.setattr(config, "ensure_default_config", lambda: None)
    monkeypatch.setattr("voxweave.cli.install_logging", lambda **_kwargs: None)
    monkeypatch.delenv("VOXWEAVE_VOICEPRINTS", raising=False)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    return SimpleNamespace(calls=calls, media=str(media))


def _invoke(args: list[str]):
    from voxweave.cli import cli

    return CliRunner().invoke(cli, args, terminal_width=200)


def test_cli_flag_is_case_insensitive_and_reaches_process(cli_process):
    result = _invoke(
        ["--diarize", "--speaker-clustering", "VoicePrint", cli_process.media]
    )
    assert result.exit_code == 0, result.output
    assert cli_process.calls[-1]["speaker_clustering"] == "voiceprint"
    assert cli_process.calls[-1]["diarize"] is True


def test_cli_flag_beats_env_and_env_beats_default(cli_process, monkeypatch):
    monkeypatch.setenv(ENV, "voiceprint")
    from_env = _invoke(["--diarize", cli_process.media])
    from_cli = _invoke(
        ["--diarize", "--speaker-clustering", "pyannote", cli_process.media]
    )
    assert from_env.exit_code == from_cli.exit_code == 0
    assert [c["speaker_clustering"] for c in cli_process.calls] == [
        "voiceprint",
        "pyannote",
    ]


def test_cli_rejects_unknown_values(cli_process, monkeypatch):
    bad_flag = _invoke(["--diarize", "--speaker-clustering", "vbx", cli_process.media])
    assert bad_flag.exit_code == 2
    assert "voiceprint" in bad_flag.output and "pyannote" in bad_flag.output

    monkeypatch.setenv(ENV, "vbx")
    bad_env = _invoke(["--diarize", cli_process.media])
    assert bad_env.exit_code == 2
    assert f"environment {ENV}" in bad_env.output
    assert cli_process.calls == []


def test_cli_bad_env_is_ignored_without_diarization(cli_process, monkeypatch):
    monkeypatch.setenv(ENV, "vbx")
    result = _invoke([cli_process.media])
    assert result.exit_code == 0, result.output
    assert cli_process.calls[-1]["diarize"] is False


def test_cli_flag_without_diarize_warns(cli_process, caplog):
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        result = _invoke(["--speaker-clustering", "voiceprint", cli_process.media])
    assert result.exit_code == 0, result.output
    assert "--speaker-clustering has no effect" in caplog.text


def test_cli_help_documents_the_flag(cli_process):
    result = _invoke(["transcribe", "--help"])
    assert result.exit_code == 0, result.output
    text = re.sub(r"[\s│┃]+", " ", result.output)
    assert "--speaker-clustering" in text
    assert "[voiceprint|pyannote]" in text
    assert f"Default: {config.DEFAULT_DIARIZE_CLUSTERING}." in text
    assert ENV in text
    assert "[diarize].clustering" in text
