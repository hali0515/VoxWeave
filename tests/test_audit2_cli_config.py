"""Audit follow-ups for the CLI/config/pipeline cluster.

Covers the diarization preflight wired into ``process``, the bound align
vocals-cache companion, the production sibling writer (the legacy test-only
writer chain is gone), config warnings, and the vocals-cache autocast default.
"""

import json
import logging
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from voxweave import backend, config, pipeline, vocalscache
from voxweave.cli import cli
from voxweave.voicebase import media_fingerprint
from voxweave.vocalscache import (
    cache_companion_path,
    load_cache_companion,
    validate_cache_pair,
)

UNITS = [
    {"text": "Where", "start": 0.0, "end": 0.4},
    {"text": "did", "start": 0.5, "end": 0.8},
    {"text": "you", "start": 0.9, "end": 1.2},
    {"text": "go.", "start": 1.4, "end": 2.0},
]
SEPARATOR = {
    "repo": "example/separator",
    "file": "weights.ckpt",
    "checkpoint": "b" * 64,
    "config_sha256": "c" * 64,
}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _no_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """No token from any source, whatever the developer's machine has stored."""
    for name in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "conf_hf_token", lambda: None)


def _refuse_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:
    def transcribe(*_args, **_kwargs):
        raise AssertionError("audio work started before the diarization preflight")

    monkeypatch.setattr(pipeline, "transcribe", transcribe)


# --- diarization preflight in process() ----------------------------------------


def test_process_refuses_a_gated_diarization_without_token_before_audio_work(
    tmp_path, monkeypatch
):
    _no_hf_token(monkeypatch)
    _refuse_transcribe(monkeypatch)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    with pytest.raises(RuntimeError, match="model-card"):
        pipeline.process(media, diarize=True, shot_snap=False)

    assert not (tmp_path / "episode.vtt").exists()
    assert not (tmp_path / "episode.json").exists()


def test_process_refuses_impossible_speaker_bounds_before_audio_work(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_test_token")
    _refuse_transcribe(monkeypatch)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    with pytest.raises(ValueError, match="min_speakers"):
        pipeline.process(
            media, diarize=True, min_speakers=3, max_speakers=2, shot_snap=False
        )


def test_process_preflight_runs_before_the_voiceprint_prefetch(tmp_path, monkeypatch):
    _no_hf_token(monkeypatch)
    _refuse_transcribe(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_prefetch_voiceprint_models",
        lambda *_a, **_k: pytest.fail("voiceprint checkpoints fetched first"),
    )
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    with pytest.raises(RuntimeError, match="model-card"):
        pipeline.process(media, diarize=True, voiceprints=True, shot_snap=False)


def test_process_with_injected_words_needs_no_diarization_token(tmp_path, monkeypatch):
    # Injected words skip transcription, so nothing is diarized and nothing is gated.
    _no_hf_token(monkeypatch)
    media = tmp_path / "episode.mkv"

    out = pipeline.process(media, diarize=True, word_segments=("en", list(UNITS)))

    assert out == tmp_path / "episode.vtt"


def test_cli_diarize_without_token_fails_fast_with_the_error_panel(
    tmp_path, monkeypatch
):
    _no_hf_token(monkeypatch)
    _refuse_transcribe(monkeypatch)
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    result = CliRunner().invoke(cli, [str(media), "--diarize", "--no-shot-snap"])

    assert result.exit_code == 1
    assert "model-card" in _ANSI.sub("", result.output)


# --- bound align vocals cache ---------------------------------------------------


def test_bound_align_cache_miss_publishes_a_companion_the_next_run_reuses(
    tmp_path, monkeypatch
):
    source = tmp_path / "snapshot.wav"
    source.write_bytes(b"snapshot bytes")
    cache_owner = tmp_path / "episode.wav"
    cache_owner.write_bytes(source.read_bytes())
    fingerprint = media_fingerprint(source)
    parts = tuple(
        tmp_path / name
        for name in ("full.wav", "vocals.wav", "speech.wav", "vocals32.wav")
    )
    separations: list[dict] = []

    def fake_separate(_media, **kwargs):
        separations.append(kwargs)
        assert kwargs.get("return_separator_identity") is True
        return (*parts, dict(SEPARATOR))

    decoded = tmp_path / "decoded.wav"
    decoded.write_bytes(b"decoded")
    monkeypatch.setattr(backend, "separator_identity", lambda: dict(SEPARATOR))
    monkeypatch.setattr(pipeline, "_separate_to_16k_32k", fake_separate)
    monkeypatch.setattr(
        pipeline, "_encode_flac", lambda _src, dst: Path(dst).write_bytes(b"flac")
    )
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_a, **_k: decoded)

    def prepare() -> Path:
        return pipeline._prepare_16k_for_align(
            source,
            separate=True,
            normalize=False,
            reporter=pipeline.Reporter(),
            tmp=[],
            cache_media=cache_owner,
            source_fingerprint=fingerprint,
        )

    assert prepare() == parts[2]
    cache = pipeline.cache_vocals_path(cache_owner)
    companion, _validated = load_cache_companion(cache_companion_path(cache))
    validate_cache_pair(
        companion, cache, media_fingerprint=fingerprint, separator=SEPARATOR
    )

    assert prepare() == decoded  # a hit: no second separation
    assert len(separations) == 1


# --- the production sibling writer ---------------------------------------------


def test_process_json_keeps_vad_speech_and_drops_in_memory_cue_keys(
    tmp_path, monkeypatch
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")

    def fake_transcribe(_source, **_kwargs):
        return "en", [dict(u) for u in UNITS], [(0.0, 1.0), (1.3, 2.1)], [], [], None

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    pipeline.process(media, separate=False, skip_songs=False, shot_snap=False)

    data = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert data["vad_speech"] == [[0.0, 1.0], [1.3, 2.1]]
    assert list(data)[-1] == "segmentation"
    for segment in data["segments"]:
        assert not {"speech_start", "speech_end", "speaker_ids"} & set(segment)


def test_injected_words_write_an_empty_vad_speech(tmp_path):
    media = tmp_path / "episode.mkv"
    pipeline.process(media, word_segments=("en", [dict(u) for u in UNITS]))
    data = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
    assert data["vad_speech"] == []


# --- config warnings -------------------------------------------------------------


@pytest.fixture
def conf_at(tmp_path, monkeypatch):
    path = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(path))
    return path


def _warnings(caplog, needle: str) -> int:
    return sum(needle in r.getMessage() for r in caplog.records)


def test_a_config_typo_is_reported_once_per_run(conf_at, caplog):
    conf_at.write_text('[llm]\nbase-url = "http://x"\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        for _ in range(3):
            config.conf_default_flag("separate", True)
            config.resolve_llm_base_url(None)
    assert _warnings(caplog, "unknown config key 'base-url'") == 1


@pytest.mark.parametrize(
    ("text", "call", "expected"),
    [
        ("batch = 4\n", lambda: config.conf_batch("mms"), 4),
        ('fusion = "large-v3"\n', config.conf_fusion_whisper, "large-v3"),
        (
            "defaults = true\n",
            lambda: config.conf_default_flag("diarize", False),
            False,
        ),
        (
            'align = "mms"\n',
            lambda: config.align_model_for("en"),
            config.DEFAULT_ALIGN_MODELS["en"],
        ),
        ('llm = "gpt"\n', lambda: config.resolve_llm_base_url(None), None),
    ],
)
def test_a_non_table_section_is_warned_about_and_ignored(
    conf_at, caplog, monkeypatch, text, call, expected
):
    for name in ("VOXWEAVE_MMS_BATCH", "VOXWEAVE_FUSION_WHISPER", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    conf_at.write_text(text, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        value = call()
        call()
    assert value == expected  # the built-in default
    assert _warnings(caplog, "expected table") == 1


def test_invalid_batch_values_are_warned_about(conf_at, caplog, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CTC_BATCH", "lots")
    conf_at.write_text('[batch]\nctc = "many"\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert config.conf_batch("ctc") == 1
    assert "VOXWEAVE_CTC_BATCH" in caplog.text
    assert "[batch].ctc must be an integer" in caplog.text


def test_non_string_fusion_and_token_values_are_warned_about(
    conf_at, caplog, monkeypatch
):
    monkeypatch.delenv("VOXWEAVE_FUSION_QWEN", raising=False)
    conf_at.write_text("hf_token = 1\n[fusion]\nqwen = 2\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        assert config.conf_fusion_qwen() == config.DEFAULT_FUSION_QWEN
        config._load()  # the token check needs no hub lookup to be exercised
        for name in ("VOXWEAVE_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        config.conf_hf_token()
    assert "[fusion].qwen has wrong type" in caplog.text
    assert "'hf_token' has wrong type" in caplog.text


# --- vocals cache: companions older than the autocast setting ------------------


def test_absent_autocast_means_the_fp32_path_whatever_the_configured_default(
    monkeypatch,
):
    monkeypatch.setattr(config, "SEP_AUTOCAST_DEFAULT", "bf16")
    identity = vocalscache.validate_separator_identity(dict(SEPARATOR))
    assert identity.autocast == vocalscache.PRE_AUTOCAST_MODE == "off"


# --- CLI help --------------------------------------------------------------------


def test_min_speakers_help_matches_the_measured_advice():
    from voxweave.cli import cmd_transcribe

    helps = {p.name: getattr(p, "help", "") or "" for p in cmd_transcribe.params}
    assert "when the count is known" not in helps["min_speakers"]
    assert "only when a missing speaker matters more" in helps["min_speakers"]
    assert "over-splitting" in helps["max_speakers"]


# --- missing-dependency hints ---------------------------------------------------


def test_cuda_install_hint_keeps_the_onnxruntime_override():
    from voxweave import runtime

    # Without the override the CPU onnxruntime wheel shadows onnxruntime-gpu.
    for hint in (runtime._MISSING_HINT, runtime._MISSING_WHISPER):
        assert "--overrides" in hint
        assert "onnxruntime; sys_platform == 'darwin'" in hint
