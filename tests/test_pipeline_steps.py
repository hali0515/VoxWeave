"""Workflow totals follow enabled work, not task/download/retry event counts."""

import json
import wave
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from voxweave import backend, chunking, diarize, pipeline, shotdet, songdet
from voxweave.progress import Reporter


class RecordingReporter(Reporter):
    def __init__(self) -> None:
        self.planned: tuple[str, ...] = ()
        self.entered: list[str] = []
        self.details: list[tuple] = []

    def plan(self, steps: Sequence[str]) -> None:
        assert not self.planned, "a nested operation replaced the command plan"
        self.planned = tuple(steps)

    def step(self, label: str) -> None:
        assert label == self.planned[len(self.entered)]
        self.entered.append(label)

    def stage(self, label: str) -> None:
        self.details.append(("stage", label))

    def status(self, label: str) -> None:
        self.details.append(("status", label))

    def task(self, label: str, total: int) -> None:
        self.details.append(("task", label, total))

    def advance(self, n: int = 1) -> None:
        self.details.append(("advance", n))

    def download(self, label: str, done: int, total: int | None) -> None:
        self.details.append(("download", label, done, total))

    def assert_complete(self, expected: Sequence[str]) -> None:
        assert self.planned == tuple(expected)
        assert self.entered == list(expected)


UNITS = [
    {"text": "hello", "start": 0.0, "end": 0.5},
    {"text": "world", "start": 0.5, "end": 1.0},
]


def _subtitle(tmp_path: Path, text: str = "hello") -> Path:
    path = tmp_path / "episode.vtt"
    path.write_text(
        f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{text}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def stub_transcription(tmp_path, monkeypatch):
    """Keep real process/transcribe control flow while replacing GPU/media work."""
    counter = 0

    def audio(*_args, **_kwargs):
        nonlocal counter
        counter += 1
        path = tmp_path / f"decoded-{counter}.wav"
        path.write_bytes(b"audio")
        return path

    monkeypatch.setattr(pipeline, "decode_to_wav", audio)
    monkeypatch.setattr(pipeline, "slice_wav", audio)
    monkeypatch.setattr(
        pipeline,
        "_separate_to_16k_32k",
        lambda *_a, **_k: tuple(audio() for _ in range(4)),
    )
    monkeypatch.setattr(
        pipeline,
        "_encode_flac",
        lambda source, destination: destination.write_bytes(source.read_bytes()),
    )
    monkeypatch.setattr(
        pipeline, "vad_speech_segments", lambda *_a, **_k: [{"start": 0.0, "end": 1.0}]
    )
    monkeypatch.setattr(pipeline, "detect_song_spans", lambda *_a, **_k: ([], [], []))
    monkeypatch.setattr(backend, "chunk_pass_count", lambda _model: 1)

    def transcribe_chunks(chunks, _language, **kwargs):
        for index in range(len(chunks)):
            kwargs["on_done"](index)
        return [("en", "hello world", [dict(unit) for unit in UNITS]) for _ in chunks]

    monkeypatch.setattr(backend, "transcribe_chunks", transcribe_chunks)
    monkeypatch.setattr(backend, "release", lambda: None)
    monkeypatch.setattr(chunking, "release_silero_vad", lambda: None)
    monkeypatch.setattr(songdet, "release_model", lambda: None)
    monkeypatch.setattr(shotdet, "detect_shot_changes", lambda _path: [])
    monkeypatch.setattr(
        diarize,
        "diarize_turns",
        lambda *_a, **_k: SimpleNamespace(turns=[], centroids={}),
    )
    monkeypatch.setattr(diarize, "release", lambda: None)
    monkeypatch.setattr(pipeline, "_write_sdh_sidecar", lambda *_a, **_k: None)


@pytest.mark.parametrize(
    ("options", "middle", "tail"),
    [
        ({}, [], []),
        ({"skip_songs": True}, ["detect songs"], []),
        ({"keep_lyrics": True}, ["detect songs"], []),
        ({"separate": False, "skip_songs": True}, [], []),
        ({"diarize": True}, [], ["identify speakers"]),
        ({"shot_snap": True}, [], ["detect shot changes"]),
        (
            {"skip_songs": True, "diarize": True, "shot_snap": True, "sdh": True},
            ["detect songs"],
            ["identify speakers", "detect shot changes"],
        ),
    ],
)
def test_process_steps_match_enabled_work(
    tmp_path, stub_transcription, options, middle, tail
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    rep = RecordingReporter()

    out = pipeline.process(media, reporter=rep, **({"shot_snap": False} | options))

    expected = (
        ["inspect source", "prepare audio"]
        + middle
        + ["find speech", "transcribe and align"]
        + tail
        + ["layout subtitles", "write outputs"]
    )
    if options.get("sdh"):
        expected.append("create SDH sidecar")
    rep.assert_complete(expected)
    assert out.exists()
    assert any(detail[0] == "task" for detail in rep.details)


def test_process_injected_words_omit_all_media_only_steps(tmp_path):
    rep = RecordingReporter()
    pipeline.process(
        tmp_path / "episode.mkv",
        word_segments=("en", [dict(unit) for unit in UNITS]),
        skip_songs=True,
        diarize=True,
        shot_snap=True,
        sdh=True,
        reporter=rep,
    )
    rep.assert_complete(("inspect source", "layout subtitles", "write outputs"))


def test_process_cache_hit_keeps_audio_preparation_step(
    tmp_path, monkeypatch, stub_transcription
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    cache = tmp_path / "vocals.flac"
    cache.write_bytes(b"cached audio")
    monkeypatch.setattr(pipeline, "cache_vocals_path", lambda _media: cache)
    monkeypatch.setattr(pipeline, "_vocals_cache_fresh", lambda *_a: True)

    def unexpected_separation(*_args, **_kwargs):
        pytest.fail("a cache hit must not run separation")

    monkeypatch.setattr(pipeline, "_separate_to_16k_32k", unexpected_separation)
    rep = RecordingReporter()
    pipeline.process(media, skip_songs=True, shot_snap=False, reporter=rep)

    rep.assert_complete(
        (
            "inspect source",
            "prepare audio",
            "detect songs",
            "find speech",
            "transcribe and align",
            "layout subtitles",
            "write outputs",
        )
    )
    assert ("stage", "vocals cache (32k)") in rep.details


def test_split_progress_never_reaches_smart_split_options(tmp_path):
    output = pipeline.process(
        tmp_path / "episode.mkv", word_segments=("en", [dict(unit) for unit in UNITS])
    )
    rep = RecordingReporter()
    assert pipeline.split(output, reporter=rep) == output
    rep.assert_complete(("read subtitle data", "layout subtitles", "write outputs"))


def test_translate_retry_stays_in_the_same_step(tmp_path, monkeypatch):
    vtt = _subtitle(tmp_path)
    with vtt.open("a", encoding="utf-8") as stream:
        stream.write("\n00:00:02.000 --> 00:00:03.000\nworld\n")
    rep = RecordingReporter()
    calls = []

    def translate(payload, **kwargs):
        assert rep.entered == ["read subtitles", "translate cues"]
        assert kwargs["reporter"] is rep
        calls.append(payload)
        rep.task("LLM response", len(payload))
        rep.advance()
        return {0: "hello translated"} if len(calls) == 1 else {1: "world translated"}

    monkeypatch.setattr(pipeline.translate_mod, "translate_cues", translate)
    original = vtt.read_bytes()
    out = pipeline.translate(vtt, reporter=rep)

    rep.assert_complete(("read subtitles", "translate cues", "write translation"))
    assert len(calls) == 2
    assert [cue["i"] for cue in calls[1]] == [1]
    assert vtt.read_bytes() == original
    assert "world translated" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("apply", "align_after", "changed"),
    [
        (False, False, True),
        (False, True, True),
        (True, False, True),
        (True, True, True),
        (True, True, False),
    ],
)
def test_correct_nested_alignment_preserves_outer_plan(
    tmp_path, monkeypatch, apply, align_after, changed
):
    vtt = _subtitle(tmp_path)
    rep = RecordingReporter()
    fixes = [{"i": 0, "orig": "hello", "fixed": "hallo", "reason": "test"}]
    monkeypatch.setattr(
        pipeline.asrfix_mod, "correct_cues", lambda *_a, **_k: fixes if changed else []
    )
    aligned = []

    def align(path, *, reporter, **_kwargs):
        aligned.append(path)
        reporter.plan(("nested read", "nested align"))
        reporter.step("nested read")
        reporter.stage("read nested subtitles")
        reporter.status("timing reference ready")
        reporter.step("nested align")
        reporter.download("alignment weights", 1, 2)
        reporter.task("alignment", 1)
        reporter.advance()
        return path

    monkeypatch.setattr(pipeline, "align", align)
    result = pipeline.correct(vtt, apply=apply, align_after=align_after, reporter=rep)

    expected = ["read subtitles", "correct text", "write correction"]
    if apply and align_after:
        expected.append("check and refresh timing")
    rep.assert_complete(expected)
    assert result["aligned"] is bool(apply and align_after and changed)
    assert bool(aligned) is result["aligned"]
    if aligned:
        assert ("stage", "read nested subtitles") in rep.details
        assert ("status", "timing reference ready") in rep.details
        assert ("download", "alignment weights", 1, 2) in rep.details
        assert ("task", "alignment", 1) in rep.details
        assert ("advance", 1) in rep.details
    elif apply and align_after:
        assert ("stage", "no text changes; alignment not needed") in rep.details


def test_align_reports_real_order_without_counting_backend_tasks(tmp_path, monkeypatch):
    media = tmp_path / "episode.wav"
    with wave.open(str(media), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 32000)
    (tmp_path / "episode.json").write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": [
                    {"text": "你", "start": 0.0, "end": 0.5},
                    {"text": "好", "start": 0.5, "end": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    vtt = _subtitle(tmp_path, "你好")
    rep = RecordingReporter()

    def prepare(*_args, **_kwargs):
        assert rep.entered == ["read subtitles", "prepare audio"]
        rep.stage("cached vocals")
        return media

    def align_text(_wav, text, _iso):
        assert rep.entered == ["read subtitles", "prepare audio", "align subtitles"]
        return [
            {"text": value, "start": index * 0.5, "end": (index + 1) * 0.5}
            for index, value in enumerate(text)
        ]

    monkeypatch.setattr(pipeline, "_prepare_16k_for_align", prepare)
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_a, **_k: media)
    monkeypatch.setattr(backend, "align_text", align_text)
    monkeypatch.setattr(backend, "release", lambda: None)

    assert pipeline.align(vtt, reporter=rep) == vtt
    rep.assert_complete(
        ("read subtitles", "prepare audio", "align subtitles", "write outputs")
    )
    assert ("task", "per-cue alignment", 1) in rep.details
