import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

from voxweave import backend, episode_transaction, pipeline
from voxweave.cli import cli
from voxweave.mediasnapshot import SnapshotUnavailable
from voxweave.voicebase import media_fingerprint
from voxweave.vocalscache import (
    cache_companion_path,
    cache_lock as real_cache_lock,
    publish_cache_companion,
)


CAPTURE = "c" + "1" * 32
SEPARATOR = {
    "repo": "example/separator",
    "file": "weights.ckpt",
    "checkpoint": "b" * 64,
    "config_sha256": "c" * 64,
}


@pytest.fixture(autouse=True)
def _private_snapshot_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(tmp_path / "cache-root"))
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))


def _sibling(*, pair_media: str | None = None) -> dict:
    data = {
        "language": "zh",
        "segments": [],
        "word_segments": [
            {"text": "你", "start": 0.0, "end": 0.5},
            {"text": "好", "start": 0.5, "end": 1.0},
        ],
        "speaker_turns": [[0.0, 1.0, "SPEAKER_00"]],
    }
    if pair_media is not None:
        data["voiceprint_capture"] = CAPTURE
        data["voiceprint_media"] = pair_media
    return data


def _write_split_input(path: Path, *, pair_media: str | None = None) -> bytes:
    raw = json.dumps(
        _sibling(pair_media=pair_media),
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _write_align_input(
    tmp_path: Path,
    *,
    media_bytes: bytes = b"selected media",
    pair_media: str | None = None,
) -> tuple[Path, Path, Path, bytes, bytes]:
    media = tmp_path / "episode.wav"
    media.write_bytes(media_bytes)
    json_path = tmp_path / "episode.json"
    json_bytes = json.dumps(
        _sibling(pair_media=pair_media),
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    json_path.write_bytes(json_bytes)
    vtt_path = tmp_path / "episode.vtt"
    vtt_bytes = b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\xe4\xbd\xa0\xe5\xa5\xbd\n"
    vtt_path.write_bytes(vtt_bytes)
    return media, vtt_path, json_path, vtt_bytes, json_bytes


def _stub_align(
    tmp_path: Path,
    monkeypatch,
    *,
    inspect_prepare=None,
) -> None:
    prepared = tmp_path / "prepared.wav"
    prepared.write_bytes(b"prepared")
    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"chunk")

    def fake_prepare(source, **kwargs):
        if inspect_prepare is not None:
            inspect_prepare(Path(source), kwargs)
        return prepared

    monkeypatch.setattr(pipeline, "_prepare_16k_for_align", fake_prepare)
    monkeypatch.setattr(pipeline, "slice_wav", lambda *_args, **_kwargs: chunk)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _lang: [
            {"text": char, "start": index * 0.4, "end": index * 0.4 + 0.3}
            for index, char in enumerate(text)
        ],
    )


def test_split_passes_valid_pair_verbatim(tmp_path):
    json_path = tmp_path / "episode.json"
    _write_split_input(json_path, pair_media="a" * 64)

    pipeline.split(json_path)

    replayed = json.loads(json_path.read_text(encoding="utf-8"))
    assert replayed["voiceprint_capture"] == CAPTURE
    assert replayed["voiceprint_media"] == "a" * 64


def test_split_warns_and_drops_invalid_pair(tmp_path, caplog):
    json_path = tmp_path / "episode.json"
    data = _sibling(pair_media="a" * 64)
    data["voiceprint_capture"] = "invalid"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level("WARNING", logger="voxweave"):
        pipeline.split(json_path)

    replayed = json.loads(json_path.read_text(encoding="utf-8"))
    assert "voiceprint_capture" not in replayed
    assert "voiceprint_media" not in replayed
    assert "dropping invalid voiceprint replay pair" in caplog.text


def test_split_input_cas_aborts_without_output(tmp_path, monkeypatch):
    json_path = tmp_path / "episode.json"
    _write_split_input(json_path)
    original_segment = pipeline.segment_document
    external = b'{"external": true}\n'

    def mutate_after_compute(**kwargs):
        result = original_segment(**kwargs)
        json_path.write_bytes(external)
        return result

    monkeypatch.setattr(pipeline, "segment_document", mutate_after_compute)

    with pytest.raises(RuntimeError, match="input changed during replay"):
        pipeline.split(json_path)

    assert json_path.read_bytes() == external
    assert not (tmp_path / "episode.vtt").exists()


def test_align_same_content_alternate_media_preserves_pair_and_snapshot_source(
    tmp_path, monkeypatch
):
    original = tmp_path / "original.wav"
    original.write_bytes(b"same episode bytes")
    fingerprint = media_fingerprint(original)
    _media, vtt_path, json_path, _vtt, _json = _write_align_input(
        tmp_path,
        pair_media=fingerprint,
    )
    alternate = tmp_path / "alternate.wav"
    alternate.write_bytes(original.read_bytes())
    seen_snapshot: list[Path] = []

    def inspect_prepare(source: Path, kwargs: dict) -> None:
        seen_snapshot.append(source)
        assert source != alternate
        assert source.read_bytes() == alternate.read_bytes()
        original_bytes = alternate.read_bytes()
        with alternate.open("r+b") as live:
            live.truncate(0)
            live.write(b"B" * len(original_bytes))
            live.flush()
            live.seek(0)
            live.truncate(0)
            live.write(original_bytes)
            live.flush()
        assert source.read_bytes() == original_bytes
        assert kwargs["cache_media"] == alternate
        assert kwargs["source_fingerprint"] == fingerprint

    _stub_align(tmp_path, monkeypatch, inspect_prepare=inspect_prepare)

    result = CliRunner().invoke(
        cli,
        ["align", str(vtt_path), "--media", str(alternate), "--no-separate"],
    )

    assert result.exit_code == 0, result.output
    replayed = json.loads(json_path.read_text(encoding="utf-8"))
    assert replayed["voiceprint_capture"] == CAPTURE
    assert replayed["voiceprint_media"] == fingerprint
    assert seen_snapshot and not seen_snapshot[0].exists()


def test_align_live_change_after_snapshot_stale_aborts_without_cleanup(
    tmp_path, monkeypatch
):
    media, vtt_path, json_path, _vtt, _json = _write_align_input(tmp_path)
    fingerprint = media_fingerprint(media)
    json_path.write_text(
        json.dumps(_sibling(pair_media=fingerprint)),
        encoding="utf-8",
    )
    artifacts = [
        tmp_path / "episode.voiceprints.json",
        tmp_path / "episode.speakers.suggest.json",
        tmp_path / "episode.speakers.html",
    ]
    for artifact in artifacts:
        artifact.write_text("sensitive", encoding="utf-8")
    original_vtt = vtt_path.read_bytes()
    original_json = json_path.read_bytes()

    def inspect_prepare(source: Path, _kwargs: dict) -> None:
        assert source != media
        assert source.read_bytes() == media.read_bytes()
        media.write_bytes(b"replacement media")

    _stub_align(tmp_path, monkeypatch, inspect_prepare=inspect_prepare)

    with pytest.raises(episode_transaction.MediaStaleError) as caught:
        pipeline.align(vtt_path)

    assert caught.value.failure.detail_code == "media-generation"
    assert vtt_path.read_bytes() == original_vtt
    assert json_path.read_bytes() == original_json
    assert all(path.read_text(encoding="utf-8") == "sensitive" for path in artifacts)


def test_align_publishes_json_before_vtt_after_final_media_recheck(
    tmp_path, monkeypatch
):
    media, vtt_path, json_path, _vtt, _json = _write_align_input(
        tmp_path,
        media_bytes=b"A" * 64,
    )
    fingerprint = media_fingerprint(media)
    json_path.write_text(
        json.dumps(_sibling(pair_media=fingerprint)),
        encoding="utf-8",
    )
    artifacts = [
        tmp_path / "episode.voiceprints.json",
        tmp_path / "episode.speakers.suggest.json",
        tmp_path / "episode.speakers.html",
    ]
    for artifact in artifacts:
        artifact.write_text("sensitive", encoding="utf-8")
    _stub_align(tmp_path, monkeypatch)
    real_replace = episode_transaction._replace_stage
    order: list[str] = []

    def observed_replace(stage):
        order.append(stage.target.name)
        real_replace(stage)

    monkeypatch.setattr(episode_transaction, "_replace_stage", observed_replace)

    pipeline.align(vtt_path)

    replayed = json.loads(json_path.read_text(encoding="utf-8"))
    assert order == ["episode.json", "episode.vtt"]
    assert replayed["voiceprint_capture"] == CAPTURE
    assert replayed["voiceprint_media"] == fingerprint
    assert all(path.read_text(encoding="utf-8") == "sensitive" for path in artifacts)


def test_align_omit_unlink_failure_names_landed_outputs_and_leftover(
    tmp_path, monkeypatch
):
    prior = tmp_path / "prior.wav"
    prior.write_bytes(b"prior media")
    fingerprint = media_fingerprint(prior)
    _media, vtt_path, json_path, _vtt, _json = _write_align_input(
        tmp_path,
        media_bytes=b"replacement media",
        pair_media=fingerprint,
    )
    sidecar = tmp_path / "episode.voiceprints.json"
    sidecar.write_text("sensitive", encoding="utf-8")
    original_unlink = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if Path(path) == sidecar:
            raise OSError("permission denied")
        return original_unlink(path, *args, **kwargs)

    _stub_align(tmp_path, monkeypatch)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(
        RuntimeError,
        match=r"primary JSON/VTT outputs landed.*could not delete .*voiceprints",
    ):
        pipeline.align(vtt_path)

    replayed = json.loads(json_path.read_text(encoding="utf-8"))
    assert "voiceprint_capture" not in replayed
    assert "voiceprint_media" not in replayed
    assert sidecar.exists()


def test_align_different_selected_media_omits_pair_and_deletes_artifacts(
    tmp_path, monkeypatch
):
    prior = tmp_path / "prior.wav"
    prior.write_bytes(b"prior media")
    fingerprint = media_fingerprint(prior)
    _media, vtt_path, json_path, _vtt, _json = _write_align_input(
        tmp_path,
        media_bytes=b"replacement media",
        pair_media=fingerprint,
    )
    artifacts = [
        tmp_path / "episode.voiceprints.json",
        tmp_path / "episode.speakers.suggest.json",
        tmp_path / "episode.speakers.html",
    ]
    for artifact in artifacts:
        artifact.write_text("sensitive", encoding="utf-8")
    _stub_align(tmp_path, monkeypatch)

    pipeline.align(vtt_path)

    replayed = json.loads(json_path.read_text(encoding="utf-8"))
    assert "voiceprint_capture" not in replayed
    assert "voiceprint_media" not in replayed
    assert not any(path.exists() for path in artifacts)


def test_bound_align_snapshot_failure_continues_unbound_and_cleans(
    tmp_path, monkeypatch
):
    media, vtt_path, json_path, _vtt, _json = _write_align_input(tmp_path)
    fingerprint = media_fingerprint(media)
    data = _sibling(pair_media=fingerprint)
    json_path.write_text(json.dumps(data), encoding="utf-8")
    artifact = tmp_path / "episode.voiceprints.json"
    artifact.write_text("sensitive", encoding="utf-8")

    class BrokenSnapshot:
        def __init__(self, _source):
            pass

        def __enter__(self):
            raise SnapshotUnavailable("no private copy")

        def __exit__(self, *_args):
            return False

    def inspect_prepare(source: Path, kwargs: dict) -> None:
        assert source == media
        assert kwargs["source_fingerprint"] is None

    monkeypatch.setattr(pipeline, "MediaSnapshot", BrokenSnapshot)
    _stub_align(tmp_path, monkeypatch, inspect_prepare=inspect_prepare)

    pipeline.align(vtt_path)

    replayed = json.loads(json_path.read_text(encoding="utf-8"))
    assert "voiceprint_capture" not in replayed
    assert not artifact.exists()


@pytest.mark.parametrize("changed_input", ["json", "vtt"])
def test_align_input_cas_aborts_without_own_writes(
    tmp_path, monkeypatch, changed_input
):
    _media, vtt_path, json_path, vtt_bytes, json_bytes = _write_align_input(tmp_path)
    _stub_align(tmp_path, monkeypatch)
    original_group = pipeline.realign.group_block_spans
    external_json = b'{"external": true}\n'
    external_vtt = b"WEBVTT\n\nexternal edit\n"

    def mutate_after_compute(block_units):
        result = original_group(block_units)
        if changed_input == "json":
            json_path.write_bytes(external_json)
        else:
            vtt_path.write_bytes(external_vtt)
        return result

    monkeypatch.setattr(pipeline.realign, "group_block_spans", mutate_after_compute)

    with pytest.raises(RuntimeError, match="input changed during replay"):
        pipeline.align(vtt_path)

    assert json_path.read_bytes() == (
        external_json if changed_input == "json" else json_bytes
    )
    assert vtt_path.read_bytes() == (
        external_vtt if changed_input == "vtt" else vtt_bytes
    )


def test_bound_align_rejects_duration_only_and_legacy_cache(tmp_path, monkeypatch):
    source = tmp_path / "snapshot.wav"
    source.write_bytes(b"snapshot bytes")
    cache_owner = tmp_path / "episode.wav"
    cache_owner.write_bytes(source.read_bytes())
    fingerprint = media_fingerprint(source)
    cache = pipeline.cache_vocals_path(cache_owner)
    cache.parent.mkdir()
    cache.write_bytes(b"wrong unbound cache")
    cache_companion_path(cache).write_text("stale", encoding="utf-8")
    pipeline.cache_16k_path(cache_owner).write_bytes(b"legacy cache")
    parts = tuple(
        tmp_path / name
        for name in ("full.wav", "vocals.wav", "speech.wav", "vocals32.wav")
    )
    separated_from: list[Path] = []

    def fake_separate(media, **_kwargs):
        separated_from.append(Path(media))
        return parts

    def fake_encode(_source, destination):
        Path(destination).write_bytes(b"new cache")

    monkeypatch.setattr(backend, "separator_identity", lambda: dict(SEPARATOR))
    monkeypatch.setattr(pipeline, "_probe_duration", lambda _path: 10.0)
    monkeypatch.setattr(pipeline, "_separate_to_16k_32k", fake_separate)
    monkeypatch.setattr(pipeline, "_encode_flac", fake_encode)
    monkeypatch.setattr(
        pipeline,
        "decode_to_wav",
        lambda *_args, **_kwargs: pytest.fail("unbound cache must not be decoded"),
    )

    got = pipeline._prepare_16k_for_align(
        source,
        separate=True,
        normalize=False,
        reporter=pipeline.Reporter(),
        tmp=[],
        cache_media=cache_owner,
        source_fingerprint=fingerprint,
    )

    assert got == parts[2]
    assert separated_from == [source]
    assert cache.read_bytes() == b"new cache"
    assert not cache_companion_path(cache).exists()


def test_bound_align_validates_cache_while_lock_is_held(tmp_path, monkeypatch):
    source = tmp_path / "snapshot.wav"
    source.write_bytes(b"snapshot bytes")
    cache_owner = tmp_path / "episode.wav"
    cache_owner.write_bytes(source.read_bytes())
    fingerprint = media_fingerprint(source)
    cache = pipeline.cache_vocals_path(cache_owner)
    cache.parent.mkdir()
    cache.write_bytes(b"bound cache")
    publish_cache_companion(
        cache,
        media_fingerprint=fingerprint,
        separator=SEPARATOR,
    )
    decoded = tmp_path / "decoded.wav"
    decoded.write_bytes(b"decoded")
    held = {"value": False}

    @contextmanager
    def tracking_lock(path: Path) -> Iterator:
        with real_cache_lock(path) as handle:
            held["value"] = True
            try:
                yield handle
            finally:
                held["value"] = False

    def fake_decode(path, **_kwargs):
        assert Path(path) == cache
        assert held["value"] is True
        return decoded

    monkeypatch.setattr(backend, "separator_identity", lambda: dict(SEPARATOR))
    monkeypatch.setattr(pipeline, "cache_lock", tracking_lock)
    monkeypatch.setattr(pipeline, "decode_to_wav", fake_decode)
    monkeypatch.setattr(
        pipeline,
        "_separate_to_16k_32k",
        lambda *_args, **_kwargs: pytest.fail("validated cache must be a hit"),
    )

    got = pipeline._prepare_16k_for_align(
        source,
        separate=True,
        normalize=False,
        reporter=pipeline.Reporter(),
        tmp=[],
        cache_media=cache_owner,
        source_fingerprint=fingerprint,
    )

    assert got == decoded
    assert held["value"] is False


def test_unbound_align_never_creates_media_snapshot(tmp_path, monkeypatch):
    _media, vtt_path, _json_path, _vtt, _json = _write_align_input(tmp_path)

    class ForbiddenSnapshot:
        def __init__(self, _source):
            raise AssertionError("unbound align must remain snapshot-free")

    monkeypatch.setattr(pipeline, "MediaSnapshot", ForbiddenSnapshot)
    _stub_align(tmp_path, monkeypatch)

    assert pipeline.align(vtt_path) == vtt_path
