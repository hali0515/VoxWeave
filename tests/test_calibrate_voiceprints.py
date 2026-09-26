"""Contract tests for ``scripts/calibrate_voiceprints.py`` (no models, no audio I/O)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    """Import a module from ``scripts/`` by path (it is not an installed package)."""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calib = _load_script("calibrate_voiceprints")
voiceembed = calib.voiceembed


def test_error_rates_for_perfectly_separated_scores():
    rates = calib.error_rates([0.8, 0.9, 0.7], [0.1, 0.2, 0.3])

    assert rates["eer"] == pytest.approx(0.0)
    assert 0.3 < rates["eer_threshold"] <= 0.7
    assert rates["far1_threshold"] == pytest.approx(0.7)
    assert rates["frr_at_far1"] == pytest.approx(0.0)


def test_error_rates_interpolate_the_crossing():
    # Accept = score >= t. At t=0.5: FAR 1/2, FRR 1/2 exactly.
    rates = calib.error_rates([0.4, 0.6], [0.3, 0.5])

    assert rates["eer"] == pytest.approx(0.5)
    assert rates["far1_threshold"] == pytest.approx(0.6)
    assert rates["frr_at_far1"] == pytest.approx(0.5)


def test_error_rates_without_both_populations_are_undefined():
    assert calib.error_rates([], [0.1])["eer"] is None
    assert calib.error_rates([0.9], [])["far1_threshold"] is None


def test_calibration_segments_trim_overlaps_and_keep_time_order():
    turns = [
        (0.0, 3.0, "A"),
        (2.5, 6.0, "B"),
        (6.0, 9.0, "A"),
        (9.0, 9.5, "A"),
    ]

    # A's first turn loses 2.5-3.0 to B; the 0.5 s tail turn is too short.
    assert calib.calibration_segments(turns, "A", max_segments=40) == [
        (0.0, 2.5),
        (6.0, 9.0),
    ]
    assert calib.calibration_segments(turns, "A", max_segments=1) == [(6.0, 9.0)]
    assert calib.calibration_segments(turns, "B", max_segments=40) == [(3.0, 6.0)]


def test_episode_scores_split_even_and_odd_segments():
    spans = [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0), (9.0, 11.0)]
    a_vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    b_vectors = np.array([[1.0, 0.0]], dtype=np.float64)

    targets, non_targets = calib.episode_scores(
        {"A": (spans, a_vectors), "B": ([(0.0, 2.0)], b_vectors)}
    )

    # even = segments 0, 2 -> [1, 0]; odd = segments 1, 3 -> [0, 1].
    assert targets == [{"speaker": "A", "score": pytest.approx(0.0), "segments": 4}]
    assert non_targets == [{"speakers": ["A", "B"], "score": pytest.approx(2**-0.5)}]


def test_calibrate_reports_every_model_and_releases_it():
    episodes = [
        calib.Episode(
            media=Path("ep.mkv"),
            turns=[
                (0.0, 2.0, "A"),
                (2.0, 4.0, "B"),
                (4.0, 6.0, "A"),
                (6.0, 8.0, "B"),
            ],
            audio_source=Path("ep.mkv"),
            separated=False,
        )
    ]
    waveform = np.zeros(8 * 16000, dtype=np.float32)
    released = []

    def embed(_waveform, spans):
        # Speaker A lives on axis 0, speaker B on axis 1.
        return np.array(
            [[1.0, 0.0] if start in (0.0, 4.0) else [0.0, 1.0] for start, _end in spans]
        )

    embedder = calib.Embedder(
        name="fake", embed=embed, release=lambda: released.append(True)
    )

    models = calib.calibrate(episodes, [waveform], [embedder], max_segments=40)

    result = models["fake"]
    assert result["target"]["count"] == 2
    assert result["target"]["mean"] == pytest.approx(1.0)
    assert result["non_target"]["count"] == 1
    assert result["non_target"]["mean"] == pytest.approx(0.0)
    assert result["eer"] == pytest.approx(0.0)
    assert released == [True]


def test_main_writes_the_json_report(tmp_path, monkeypatch, capsys):
    media = tmp_path / "ep.mkv"
    episode = calib.Episode(
        media=media,
        turns=[(0.0, 2.0, "A"), (2.0, 4.0, "B"), (4.0, 6.0, "A")],
        audio_source=media,
        separated=False,
    )
    monkeypatch.setattr(
        calib,
        "build_embedder",
        lambda name: calib.Embedder(
            name=name,
            embed=lambda _w, spans: np.ones((len(spans), 2)),
            release=lambda: None,
        ),
    )
    monkeypatch.setattr(calib, "load_episode", lambda _media, **_kw: episode)
    monkeypatch.setattr(
        calib, "decode_episode", lambda _episode, **_kw: np.zeros(96_000)
    )
    out = tmp_path / "report.json"

    code = calib.main([str(media), "--models", "redimnet2", "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["recipe"] == "centroid-v1"
    assert set(report["models"]) == {"redimnet2"}
    captured = capsys.readouterr()
    assert "no separated-vocals cache" in captured.err
    assert "redimnet2" in captured.out


def test_unknown_model_is_an_input_error(tmp_path, capsys):
    code = calib.main([str(tmp_path / "ep.mkv"), "--models", "ecapa"])

    assert code == 2
    assert "unknown voiceprint model" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["auto", "pyannote", "ecapa", "RedimNet2"])
def test_model_error_lists_exactly_the_accepted_models(tmp_path, capsys, name):
    code = calib.main(
        [str(tmp_path / "ep.mkv"), "--models", name, "--out", str(tmp_path / "r.json")]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 1
    listed = err.rsplit("choose from ", 1)[1].strip()
    assert listed.split(", ") == [
        "redimnet2",
        "anime-va",
        "pyannote-community-1",
        "pyannote-embedding",
    ]


def test_help_documents_models_out_and_max_segments(capsys):
    with pytest.raises(SystemExit) as excinfo:
        calib.main(["--help"])

    assert excinfo.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "pyannote-embedding" in text
    assert "(default: redimnet2,anime-va,pyannote-community-1)" in text
    assert "(default: voiceprints-calibration.json)" in text
    assert "(default: 40)" in text


def _refuse_loading(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("must fail before any episode or model is loaded")

    monkeypatch.setattr(calib, "load_episode", boom)
    monkeypatch.setattr(calib, "build_embedder", boom)


@pytest.mark.parametrize("value", ["1", "0", "-3"])
def test_max_segments_below_two_is_an_input_error(tmp_path, monkeypatch, capsys, value):
    _refuse_loading(monkeypatch)

    code = calib.main(
        [
            str(tmp_path / "ep.mkv"),
            f"--max-segments={value}",
            "--out",
            str(tmp_path / "r.json"),
        ]
    )

    assert code == 2
    assert "--max-segments must be at least 2" in capsys.readouterr().err


def test_unwritable_out_is_refused_before_any_work(tmp_path, monkeypatch, capsys):
    _refuse_loading(monkeypatch)

    missing = calib.main(
        [str(tmp_path / "ep.mkv"), "--out", str(tmp_path / "nope" / "r.json")]
    )
    directory = calib.main([str(tmp_path / "ep.mkv"), "--out", str(tmp_path)])

    assert (missing, directory) == (2, 2)
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert "is a directory" in err


def _stub_run(monkeypatch, tmp_path, *, embed=None, decode=None):
    media = tmp_path / "ep.mkv"
    episode = calib.Episode(
        media=media,
        turns=[(0.0, 2.0, "A"), (2.0, 4.0, "B"), (4.0, 6.0, "A")],
        audio_source=media,
        separated=True,
    )
    monkeypatch.setattr(
        calib,
        "build_embedder",
        lambda name: calib.Embedder(
            name=name,
            embed=embed or (lambda _w, spans: np.ones((len(spans), 2))),
            release=lambda: None,
        ),
    )
    monkeypatch.setattr(calib, "load_episode", lambda _media, **_kw: episode)
    monkeypatch.setattr(
        calib, "decode_episode", decode or (lambda _episode, **_kw: np.zeros(96_000))
    )
    return media


def test_decode_failure_is_exit_2_without_a_traceback(tmp_path, monkeypatch, capsys):
    def decode(_episode, **_kw):
        raise RuntimeError("ffmpeg failed to decode ep.mkv: moov atom not found")

    media = _stub_run(monkeypatch, tmp_path, decode=decode)
    out = tmp_path / "r.json"

    code = calib.main([str(media), "--models", "redimnet2", "--out", str(out)])

    assert code == 2
    assert "error: ffmpeg failed to decode ep.mkv" in capsys.readouterr().err
    assert not out.exists()


def test_embedder_failure_names_model_episode_and_speaker(
    tmp_path, monkeypatch, capsys
):
    def embed(_waveform, _spans):
        raise voiceembed.VoiceEmbeddingError("segment 0 falls outside the audio")

    media = _stub_run(monkeypatch, tmp_path, embed=embed)

    code = calib.main(
        [str(media), "--models", "anime-va", "--out", str(tmp_path / "r.json")]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "anime-va on ep.mkv, speaker A: segment 0 falls outside the audio" in err


def test_report_write_failure_is_exit_2(tmp_path, monkeypatch, capsys):
    media = _stub_run(monkeypatch, tmp_path)
    out = tmp_path / "r.json"

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", refuse)

    code = calib.main([str(media), "--models", "redimnet2", "--out", str(out)])

    assert code == 2
    assert f"error: cannot write {out}" in capsys.readouterr().err


def _episode_files(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"")
    (tmp_path / "ep.json").write_text(
        json.dumps({"speaker_turns": [[0.0, 2.0, "A"], [2.0, 4.0, "B"]]}),
        encoding="utf-8",
    )
    cache = tmp_path / "cache" / "ep.vocals.32k.flac"
    cache.parent.mkdir()
    cache.write_bytes(b"flac")
    return media, cache


def test_fresh_vocals_cache_is_scored(tmp_path, monkeypatch):
    from voxweave import pipeline

    media, cache = _episode_files(tmp_path)
    checked = []
    monkeypatch.setattr(
        pipeline,
        "_vocals_cache_fresh",
        lambda candidate, source: checked.append((candidate, source)) or True,
    )

    episode = calib.load_episode(media, use_vocals_cache=True)

    assert (episode.audio_source, episode.separated) == (cache, True)
    assert checked == [(cache, media)]


def test_stale_vocals_cache_falls_back_to_the_mix(tmp_path, monkeypatch):
    from voxweave import pipeline

    media, _cache = _episode_files(tmp_path)
    monkeypatch.setattr(pipeline, "_vocals_cache_fresh", lambda _c, _m: False)

    episode = calib.load_episode(media, use_vocals_cache=True)

    assert (episode.audio_source, episode.separated) == (media, False)


def test_the_default_pyannote_baseline_is_community_1():
    assert calib.DEFAULT_MODELS == ("redimnet2", "anime-va", "pyannote-community-1")
    assert calib.pyannote_source("pyannote-community-1") == (
        "pyannote/speaker-diarization-community-1#subfolder=embedding"
    )
    # The 3.1-era standalone WeSpeaker checkpoint stays available by name.
    assert calib.pyannote_source("pyannote-embedding") == (
        "pyannote/wespeaker-voxceleb-resnet34-LM"
    )


class _FakePyannoteInference:
    min_num_samples = 400

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, tensor):
        self.shapes.append(tuple(tensor.shape))
        return np.array([[3.0, 4.0]])


@pytest.mark.parametrize(
    ("name", "source"),
    [
        (
            "pyannote-community-1",
            "pyannote/speaker-diarization-community-1#subfolder=embedding",
        ),
        ("pyannote-embedding", "pyannote/wespeaker-voxceleb-resnet34-LM"),
    ],
)
def test_pyannote_baselines_load_their_own_checkpoint(monkeypatch, name, source):
    from voxweave import turnembed

    inference = _FakePyannoteInference()
    requested: list[tuple[object, object]] = []

    def fake_load(expected_identity=None, *, source=None):
        requested.append((expected_identity, source))
        return inference, turnembed.EmbeddingIdentity(
            model=f"{source}@rev",
            checkpoint_sha256="a" * 64,
            pyannote_version="4.0.7",
        )

    monkeypatch.setattr(turnembed, "_load_inference", fake_load)
    embedder = calib.build_embedder(name)
    assert embedder.name == name
    assert embedder.identity == {"source": source}

    rows = embedder.embed(np.ones(3 * 16000, dtype=np.float32), [(0.0, 2.5)])

    assert requested == [(None, source)]
    assert rows.shape == (1, 2)
    assert rows[0] == pytest.approx([0.6, 0.8])
    assert embedder.identity == {
        "source": source,
        "model": f"{source}@rev",
        "checkpoint_sha256": "a" * 64,
        "pyannote_version": "4.0.7",
    }
    embedder.release()


def test_pyannote_span_outside_the_audio_is_refused(monkeypatch):
    from voxweave import turnembed

    inference = _FakePyannoteInference()
    monkeypatch.setattr(
        turnembed,
        "_load_inference",
        lambda _identity=None, *, source=None: (
            inference,
            turnembed.EmbeddingIdentity("m@rev", "a" * 64, "4.0.7"),
        ),
    )
    embedder = calib.build_embedder("pyannote-community-1")

    with pytest.raises(voiceembed.VoiceEmbeddingError, match="segment 1 falls outside"):
        embedder.embed(np.ones(3 * 16000, dtype=np.float32), [(0.0, 2.5), (4.0, 6.0)])
    embedder.release()


def test_turnembed_loads_the_community_1_embedding_submodel(monkeypatch):
    from voxweave import turnembed

    downloads: list[object] = []

    def fake_download(authority, token):
        downloads.append((authority, token))
        raise turnembed.TurnEmbeddingError("stop after resolving the authority")

    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(turnembed, "_download_checkpoint", fake_download)

    with pytest.raises(turnembed.TurnEmbeddingError, match="stop after"):
        turnembed._load_inference(
            None, source=calib.pyannote_source("pyannote-community-1")
        )
    with pytest.raises(turnembed.TurnEmbeddingError, match="either"):
        turnembed._load_inference(
            turnembed.EmbeddingIdentity("m", "a" * 64, "4.0.7"), source="m"
        )

    ((authority, token),) = downloads
    assert authority.checkpoint == "pyannote/speaker-diarization-community-1"
    assert authority.subfolder == "embedding"
    assert authority.revision is None
    assert token == "hf_test_token"
