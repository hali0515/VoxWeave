from __future__ import annotations

import base64
import copy
import http.client
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

from voxweave import artifacts, pipeline, speakers, speakerserve, turnembed, vocalscache
from voxweave.voicebase import (
    VOICEPRINTS_MAX_BYTES,
    canonical_turns_digest,
    encode_json_bytes,
    media_fingerprint,
    validate_voiceprint_conjunction,
    write_voiceprints,
)
from voxweave.voicestore import load_voice_store


CAPTURE_ID = "c" + "1" * 32
VECTOR_A = [1.0, *([0.0] * 15)]
VECTOR_B = [0.0, 1.0, *([0.0] * 14)]
PROVENANCE = {
    "diarization_model": "example/diarizer",
    "outer_config_sha256": "a" * 64,
    "embedding_model": "example/embedder",
    "embedding_checkpoint": "b" * 64,
    "embedding_dim": 16,
    "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
    "pyannote_version": "3.4.0",
    "torch_version": "test",
}


def _fixture_embedding_identity(
    *,
    model: str | None = None,
    checkpoint_sha256: str | None = None,
    pyannote_version: str | None = None,
) -> turnembed.EmbeddingIdentity:
    return turnembed.EmbeddingIdentity(
        model=model or cast(str, PROVENANCE["embedding_model"]),
        checkpoint_sha256=checkpoint_sha256
        or cast(str, PROVENANCE["embedding_checkpoint"]),
        pyannote_version=pyannote_version or cast(str, PROVENANCE["pyannote_version"]),
    )


def _json_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _request(
    server: speakerserve.SpeakerHTTPServer,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
):
    host = str(server.server_address[0])
    connection = http.client.HTTPConnection(host, server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, response.headers, payload


def _post(
    server: speakerserve.SpeakerHTTPServer,
    path: str,
    value: object,
    *,
    token: str | None = None,
    origin: str | None = None,
    host: str | None = None,
):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-VoxWeave-Token"] = token
    if origin is not None:
        headers["Origin"] = origin
    if host is not None:
        headers["Host"] = host
    return _request(
        server,
        "POST",
        path,
        body=json.dumps(value).encode("utf-8"),
        headers=headers,
    )


@contextmanager
def _running_split_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[
    tuple[
        speakerserve.SpeakerHTTPServer,
        dict[str, Path],
        dict[str, bytes],
        list[str],
        dict[str, object],
    ]
]:
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"stable episode media")
    fingerprint = media_fingerprint(media)
    turns = [
        [0.0, 3.0, "SPEAKER_00"],
        [3.0, 6.0, "SPEAKER_01"],
        [6.0, 9.0, "SPEAKER_00"],
    ]
    sibling = {
        "language": "en",
        "segments": [],
        "word_segments": [
            {"text": "first", "start": 0.0, "end": 1.0},
            {"text": "other", "start": 3.0, "end": 4.0},
            {"text": "second", "start": 6.0, "end": 7.0},
        ],
        "speaker_turns": turns,
        "voiceprint_capture": CAPTURE_ID,
        "voiceprint_media": fingerprint,
        "fixture_marker": "preserve me",
    }
    sibling_path = tmp_path / "episode.json"
    sibling_path.write_bytes(_json_bytes(sibling))

    sidecar = {
        "version": 1,
        "capture_id": CAPTURE_ID,
        "provenance": copy.deepcopy(PROVENANCE),
        "binding": {
            "turns_digest": canonical_turns_digest(turns),
            "media_fingerprint": fingerprint,
            "media_stem": "episode",
            "created": "2026-08-30T00:00:00Z",
        },
        "speakers": {
            "SPEAKER_00": list(VECTOR_A),
            "SPEAKER_01": list(VECTOR_B),
        },
    }
    sidecar_path = tmp_path / "episode.voiceprints.json"
    write_voiceprints(sidecar_path, sidecar)

    mapping = {
        "version": 1,
        "speakers": {"SPEAKER_00": "Aoi", "SPEAKER_01": "Ren"},
    }
    mapping_path = tmp_path / "episode.speakers.json"
    mapping_path.write_bytes(_json_bytes(mapping, newline=True))
    suggest_path = tmp_path / "episode.speakers.suggest.json"
    suggest_path.write_bytes(b"stale suggestion bytes\n")

    fake_wav = tmp_path / "prepared.wav"
    fake_wav.write_bytes(b"mock 16 kHz mono wav")
    observations: dict[str, object] = {"clips": []}

    def fake_prepare(path: Path, staged: object) -> Path:
        observations["prepared_from"] = Path(path)
        observations["staged"] = staged
        return fake_wav

    def fake_embeddings(path: Path, selected_turns: object):
        observations["embedding_wav"] = Path(path)
        observations["embedding_turns"] = list(selected_turns)  # type: ignore[arg-type]
        return turnembed.AttestedTurnEmbeddings(
            {0: list(VECTOR_A), 1: list(VECTOR_B)},
            identity=_fixture_embedding_identity(),
        )

    def fake_bisect(embeddings: object):
        observations["bisect_embeddings"] = embeddings
        return {0: "A", 1: "B"}

    def fake_extract(
        source: Path,
        start: float,
        end: float,
        output: Path,
    ) -> None:
        observations["clips"].append((Path(source), start, end))  # type: ignore[union-attr]
        output.write_bytes(f"clip:{start:.1f}:{end:.1f}".encode("ascii"))

    monkeypatch.setattr(speakerserve, "_prepare_split_wav", fake_prepare)
    monkeypatch.setattr(speakerserve.turnembed, "turn_embeddings", fake_embeddings)
    monkeypatch.setattr(speakerserve.turnembed, "bisect_embeddings", fake_bisect)
    monkeypatch.setattr("voxweave.speakers.extract_clip", fake_extract)

    logs: list[str] = []
    server = speakerserve.make_server(
        page="<!doctype html><title>split audition</title>",
        media_path=media,
        mapping_path=mapping_path,
        sibling_path=sibling_path,
        speaker_ids=("SPEAKER_00", "SPEAKER_01"),
        port=0,
        report=logs.append,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    paths = {
        "media": media,
        "sibling": sibling_path,
        "sidecar": sidecar_path,
        "mapping": mapping_path,
        "suggest": suggest_path,
        "undo": artifacts.speaker_split_undo_path(media),
    }
    originals = {
        name: path.read_bytes()
        for name, path in paths.items()
        if name in {"sibling", "sidecar", "mapping", "suggest"}
    }
    try:
        yield server, paths, originals, logs, observations
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


NEW_ROUTES = (
    ("/split", {"speaker_id": "SPEAKER_00"}),
    (
        "/split-confirm",
        {"speaker_id": "SPEAKER_00", "assignment": {"0": "A", "2": "B"}},
    ),
    ("/split-undo", {}),
)


def _audio_stage(
    tmp_path: Path,
    *,
    fingerprint: str = "f" * 64,
    separated: bool,
    normalized: bool,
    separator: vocalscache.SeparatorIdentity | None = None,
) -> speakerserve._StagedSplit:
    return speakerserve._StagedSplit(
        sibling_bytes=b"{}",
        voiceprints_path=tmp_path / "voiceprints.json",
        voiceprints_bytes=b"{}",
        media_fingerprint=fingerprint,
        turns=(),
        selected_indices=(),
        embedding_dim=16,
        embedding_model=cast(str, PROVENANCE["embedding_model"]),
        embedding_checkpoint=cast(str, PROVENANCE["embedding_checkpoint"]),
        pyannote_version=cast(str, PROVENANCE["pyannote_version"]),
        audio_separated=separated,
        audio_normalized=normalized,
        audio_separator=separator,
    )


SEPARATOR = vocalscache.SeparatorIdentity(
    repo="example/separator",
    file="separator.ckpt",
    checkpoint="d" * 64,
    config_sha256="e" * 64,
)


@pytest.mark.parametrize("normalized", (False, True))
def test_prepare_split_wav_reproduces_unseparated_capture_and_ignores_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    normalized: bool,
) -> None:
    media = tmp_path / "episode.mkv"
    output = tmp_path / "decoded.wav"
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_decode(source: Path, **kwargs: object) -> Path:
        calls.append((Path(source), kwargs))
        return output

    monkeypatch.setattr(
        pipeline,
        "cache_vocals_path",
        lambda _media: pytest.fail("raw capture must ignore a vocals cache"),
    )
    monkeypatch.setattr("voxweave.chunking.decode_to_wav", fake_decode)

    result = speakerserve._prepare_split_wav(
        media,
        _audio_stage(
            tmp_path,
            separated=False,
            normalized=normalized,
        ),
    )

    assert result == output
    assert calls == [
        (
            media,
            {
                "sample_rate": turnembed.SAMPLE_RATE,
                "mono": True,
                "audio_filter": pipeline.ASR_LOUDNORM if normalized else None,
            },
        )
    ]


def test_prepare_split_wav_validates_and_reproduces_separated_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "episode.mkv"
    cache = tmp_path / "vocals.32k.flac"
    cache.write_bytes(b"bound separated vocals")
    fingerprint = "f" * 64
    companion = vocalscache.build_cache_companion(
        cache,
        media_fingerprint=fingerprint,
        separator=SEPARATOR,
    )
    vocalscache.write_cache_companion(
        vocalscache.cache_companion_path(cache),
        companion,
    )
    output = tmp_path / "decoded.wav"
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_decode(source: Path, **kwargs: object) -> Path:
        calls.append((Path(source), kwargs))
        return output

    monkeypatch.setattr(pipeline, "cache_vocals_path", lambda _media: cache)
    monkeypatch.setattr("voxweave.chunking.decode_to_wav", fake_decode)

    result = speakerserve._prepare_split_wav(
        media,
        _audio_stage(
            tmp_path,
            fingerprint=fingerprint,
            separated=True,
            normalized=True,
            separator=SEPARATOR,
        ),
    )

    assert result == output
    assert calls == [
        (
            cache.resolve(),
            {
                "sample_rate": turnembed.SAMPLE_RATE,
                "mono": True,
                "audio_filter": pipeline.ASR_LOUDNORM,
            },
        )
    ]


@pytest.mark.parametrize(
    "invalid",
    ("missing-companion", "media", "separator", "cache-bytes"),
)
def test_prepare_split_wav_refuses_unbound_separated_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    media = tmp_path / "episode.mkv"
    cache = tmp_path / "vocals.32k.flac"
    cache.write_bytes(b"bound separated vocals")
    fingerprint = "f" * 64
    companion_path = vocalscache.cache_companion_path(cache)
    vocalscache.write_cache_companion(
        companion_path,
        vocalscache.build_cache_companion(
            cache,
            media_fingerprint=fingerprint,
            separator=SEPARATOR,
        ),
    )
    staged_fingerprint = fingerprint
    staged_separator = SEPARATOR
    if invalid == "missing-companion":
        companion_path.unlink()
    elif invalid == "media":
        staged_fingerprint = "0" * 64
    elif invalid == "separator":
        staged_separator = vocalscache.SeparatorIdentity(
            repo=SEPARATOR.repo,
            file=SEPARATOR.file,
            checkpoint="0" * 64,
            config_sha256=SEPARATOR.config_sha256,
        )
    else:
        cache.write_bytes(b"changed cache bytes")

    monkeypatch.setattr(pipeline, "cache_vocals_path", lambda _media: cache)
    monkeypatch.setattr(
        "voxweave.chunking.decode_to_wav",
        lambda *_args, **_kwargs: pytest.fail("invalid cache must not be decoded"),
    )

    with pytest.raises(
        speakerserve.SplitConflict,
        match="does not match this voiceprint capture",
    ):
        speakerserve._prepare_split_wav(
            media,
            _audio_stage(
                tmp_path,
                fingerprint=staged_fingerprint,
                separated=True,
                normalized=False,
                separator=staged_separator,
            ),
        )


@pytest.mark.parametrize(("route", "payload"), NEW_ROUTES)
@pytest.mark.parametrize("guard", ("host", "origin", "token"))
def test_new_routes_match_save_host_origin_and_token_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    payload: object,
    guard: str,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        _observations,
    ):
        kwargs: dict[str, str | None] = {"token": server.token}
        if guard == "host":
            kwargs["host"] = "attacker.invalid"
        elif guard == "origin":
            kwargs["origin"] = "https://attacker.invalid"
        else:
            kwargs["token"] = "wrong-token"

        status, _headers, _body = _post(
            server,
            route,
            payload,
            **kwargs,
        )

        assert status == 403
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert paths["suggest"].read_bytes() == originals["suggest"]
        assert not paths["undo"].exists()


@pytest.mark.parametrize(("route", "_payload"), NEW_ROUTES)
def test_new_routes_match_save_body_cap_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    _payload: object,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        _observations,
    ):
        status, _headers, _body = _request(
            server,
            "POST",
            route,
            body=b"",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(speakerserve.MAX_BODY_BYTES + 1),
                "X-VoxWeave-Token": server.token,
            },
        )

        assert status == 413
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert paths["suggest"].read_bytes() == originals["suggest"]
        assert not paths["undo"].exists()


def _preview_split(server: speakerserve.SpeakerHTTPServer) -> dict[str, object]:
    status, headers, body = _post(
        server,
        "/split",
        {"speaker_id": "SPEAKER_00"},
        token=server.token,
        origin=server.origin,
    )
    assert status == 200, body
    assert headers["Cache-Control"] == "no-store"
    return json.loads(body)


def _confirm_split(
    server: speakerserve.SpeakerHTTPServer,
    proposal: dict[str, object],
) -> tuple[int, object]:
    status, _headers, body = _post(
        server,
        "/split-confirm",
        {
            "speaker_id": proposal["speaker_id"],
            "assignment": proposal["assignment"],
        },
        token=server.token,
    )
    return status, json.loads(body)


def test_split_returns_mocked_groups_and_does_not_mutate_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        observations,
    ):
        proposal = _preview_split(server)

        assert proposal["speaker_id"] == "SPEAKER_00"
        assert proposal["assignment"] == {"0": "A", "2": "B"}
        assert proposal["groups"] == [
            {
                "label": "A",
                "turn_indices": [0],
                "turn_count": 1,
                "total_duration": 3.0,
                "samples": [
                    {
                        "start": 0.0,
                        "end": 3.0,
                        "src": "data:audio/mpeg;base64,"
                        + base64.b64encode(b"clip:0.0:3.0").decode("ascii"),
                    }
                ],
            },
            {
                "label": "B",
                "turn_indices": [2],
                "turn_count": 1,
                "total_duration": 3.0,
                "samples": [
                    {
                        "start": 6.0,
                        "end": 9.0,
                        "src": "data:audio/mpeg;base64,"
                        + base64.b64encode(b"clip:6.0:9.0").decode("ascii"),
                    }
                ],
            },
        ]
        assert observations["prepared_from"] == paths["media"]
        assert observations["embedding_turns"] == [
            (0.0, 3.0, "SPEAKER_00"),
            (6.0, 9.0, "SPEAKER_00"),
        ]
        assert observations["clips"] == [
            (tmp_path / "prepared.wav", 0.0, 3.0),
            (tmp_path / "prepared.wav", 6.0, 9.0),
        ]
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert paths["suggest"].read_bytes() == originals["suggest"]
        assert not paths["undo"].exists()


@pytest.mark.parametrize(
    "field",
    ("model", "checkpoint_sha256", "pyannote_version"),
)
def test_split_refuses_result_identity_mismatch_before_clustering_or_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        observations,
    ):
        values = {
            "model": cast(str, PROVENANCE["embedding_model"]),
            "checkpoint_sha256": cast(str, PROVENANCE["embedding_checkpoint"]),
            "pyannote_version": cast(str, PROVENANCE["pyannote_version"]),
        }
        values[field] = "mismatch"

        def mismatched_embeddings(path: Path, selected_turns: object):
            observations["embedding_wav"] = Path(path)
            observations["embedding_turns"] = list(selected_turns)  # type: ignore[arg-type]
            return turnembed.AttestedTurnEmbeddings(
                {0: list(VECTOR_A), 1: list(VECTOR_B)},
                identity=turnembed.EmbeddingIdentity(**values),
            )

        monkeypatch.setattr(
            speakerserve.turnembed,
            "turn_embeddings",
            mismatched_embeddings,
        )

        status, headers, body = _post(
            server,
            "/split",
            {"speaker_id": "SPEAKER_00"},
            token=server.token,
        )

        assert status == 409
        assert headers["Cache-Control"] == "no-store"
        assert "does not match the voiceprint capture" in json.loads(body)["error"]
        assert observations["prepared_from"] == paths["media"]
        assert observations["embedding_wav"] == tmp_path / "prepared.wav"
        assert "bisect_embeddings" not in observations
        assert observations["clips"] == []
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert not paths["undo"].exists()


def test_split_refuses_provider_rows_without_result_local_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        observations,
    ):

        def unattested_embeddings(path: Path, selected_turns: object):
            observations["embedding_wav"] = Path(path)
            observations["embedding_turns"] = list(selected_turns)  # type: ignore[arg-type]
            return {0: list(VECTOR_A), 1: list(VECTOR_B)}

        monkeypatch.setattr(
            speakerserve.turnembed,
            "turn_embeddings",
            unattested_embeddings,
        )

        status, headers, body = _post(
            server,
            "/split",
            {"speaker_id": "SPEAKER_00"},
            token=server.token,
        )

        assert status == 500
        assert headers["Cache-Control"] == "no-store"
        assert "did not attest its loaded identity" in json.loads(body)["error"]
        assert observations["prepared_from"] == paths["media"]
        assert observations["embedding_wav"] == tmp_path / "prepared.wav"
        assert "bisect_embeddings" not in observations
        assert observations["clips"] == []
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert not paths["undo"].exists()


def test_split_refuses_unresolved_voiceprint_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        _originals,
        _logs,
        observations,
    ):
        sidecar = json.loads(paths["sidecar"].read_bytes())
        sidecar["provenance"]["embedding_checkpoint"] = "unresolved"
        write_voiceprints(paths["sidecar"], sidecar)

        status, headers, body = _post(
            server,
            "/split",
            {"speaker_id": "SPEAKER_00"},
            token=server.token,
        )

        assert status == 409
        assert headers["Cache-Control"] == "no-store"
        assert "compatibility is unknown" in json.loads(body)["error"]
        assert "prepared_from" not in observations
        assert "embedding_wav" not in observations
        assert not paths["undo"].exists()


def test_split_normalizes_scaled_provider_rows_before_centroid_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        _originals,
        _logs,
        _observations,
    ):
        monkeypatch.setattr(
            speakerserve.turnembed,
            "turn_embeddings",
            lambda _path, _turns: turnembed.AttestedTurnEmbeddings(
                {
                    0: [value * 3.0 for value in VECTOR_A],
                    1: [value * 7.0 for value in VECTOR_B],
                },
                identity=_fixture_embedding_identity(),
            ),
        )

        proposal = _preview_split(server)
        assert _confirm_split(server, proposal) == (200, {"new_id": "SPEAKER_02"})

        sidecar = json.loads(paths["sidecar"].read_bytes())
        assert sidecar["speakers"]["SPEAKER_00"] == VECTOR_A
        assert sidecar["speakers"]["SPEAKER_02"] == VECTOR_B


def test_confirm_rewrites_bound_episode_deletes_suggest_and_terminalizes_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        logs,
        _observations,
    ):
        proposal = _preview_split(server)
        status, response = _confirm_split(server, proposal)

        assert status == 200
        assert response == {"new_id": "SPEAKER_02"}

        sibling = json.loads(originals["sibling"])
        sibling["speaker_turns"][2][2] = "SPEAKER_02"
        expected_sibling = _json_bytes(sibling)
        assert paths["sibling"].read_bytes() == expected_sibling

        sidecar = json.loads(originals["sidecar"])
        sidecar["binding"]["turns_digest"] = canonical_turns_digest(
            sibling["speaker_turns"]
        )
        sidecar["speakers"]["SPEAKER_00"] = list(VECTOR_A)
        sidecar["speakers"]["SPEAKER_02"] = list(VECTOR_B)
        expected_sidecar = encode_json_bytes(
            sidecar,
            max_bytes=VOICEPRINTS_MAX_BYTES,
        )
        assert paths["sidecar"].read_bytes() == expected_sidecar
        validate_voiceprint_conjunction(
            json.loads(expected_sidecar),
            json.loads(expected_sibling),
            media_fingerprint(paths["media"]),
        )

        mapping = json.loads(originals["mapping"])
        mapping["speakers"]["SPEAKER_02"] = ""
        expected_mapping = _json_bytes(mapping, newline=True)
        assert paths["mapping"].read_bytes() == expected_mapping
        assert not paths["suggest"].exists()
        assert paths["undo"].is_file()
        assert logs == [
            "Split SPEAKER_00 into SPEAKER_00 and SPEAKER_02",
            "Restart `voxweave speakers` to re-audition",
        ]

        save_status, save_headers, save_body = _post(
            server,
            "/save",
            {
                "version": 1,
                "speakers": {"SPEAKER_00": "Changed", "SPEAKER_01": "Ren"},
            },
            token=server.token,
        )
        assert save_status == 409
        assert save_headers["Cache-Control"] == "no-store"
        assert json.loads(save_body)["error"]
        assert paths["mapping"].read_bytes() == expected_mapping


def test_confirm_rolls_back_a_write_that_interrupts_after_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        _observations,
    ):
        _preview_split(server)
        proposal = server.split_proposal
        assert proposal is not None
        real_write = speakerserve._write_bytes
        interrupted = False

        def write_then_interrupt(path: Path, raw: bytes) -> None:
            nonlocal interrupted
            real_write(path, raw)
            if Path(path) == paths["sidecar"] and not interrupted:
                interrupted = True
                raise KeyboardInterrupt

        monkeypatch.setattr(speakerserve, "_write_bytes", write_then_interrupt)

        with pytest.raises(KeyboardInterrupt):
            speakerserve._confirm_split(server, proposal)

        assert interrupted
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert paths["suggest"].read_bytes() == originals["suggest"]
        assert not paths["undo"].exists()


def test_undo_restores_sibling_voiceprints_and_mapping_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        _observations,
    ):
        proposal = _preview_split(server)
        assert _confirm_split(server, proposal) == (200, {"new_id": "SPEAKER_02"})

        status, headers, body = _post(
            server,
            "/split-undo",
            {},
            token=server.token,
        )

        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert json.loads(body) == {"undone": True}
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert not paths["suggest"].exists()
        assert not paths["undo"].exists()


def test_undo_rolls_back_a_write_that_interrupts_after_replace_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        _observations,
    ):
        proposal = _preview_split(server)
        assert _confirm_split(server, proposal) == (200, {"new_id": "SPEAKER_02"})
        split_bytes = {
            field: paths[field].read_bytes()
            for field in ("sibling", "sidecar", "mapping")
        }
        real_write = speakerserve._write_bytes
        interrupted = False

        def write_then_interrupt(path: Path, raw: bytes) -> None:
            nonlocal interrupted
            real_write(path, raw)
            if Path(path) == paths["sidecar"] and not interrupted:
                interrupted = True
                raise KeyboardInterrupt

        monkeypatch.setattr(speakerserve, "_write_bytes", write_then_interrupt)

        with pytest.raises(KeyboardInterrupt):
            speakerserve._undo_split(server)

        assert interrupted
        assert {
            field: paths[field].read_bytes()
            for field in ("sibling", "sidecar", "mapping")
        } == split_bytes
        assert paths["undo"].is_file()

        monkeypatch.setattr(speakerserve, "_write_bytes", real_write)
        speakerserve._undo_split(server)

        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert not paths["undo"].exists()


@pytest.mark.parametrize(
    "already_before",
    (
        ("sibling",),
        ("sidecar", "mapping"),
        ("sibling", "sidecar", "mapping"),
    ),
)
def test_undo_recovers_trusted_mixed_before_and_after_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_before: tuple[str, ...],
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        originals,
        _logs,
        _observations,
    ):
        proposal = _preview_split(server)
        assert _confirm_split(server, proposal) == (200, {"new_id": "SPEAKER_02"})
        for field in already_before:
            paths[field].write_bytes(originals[field])

        status, headers, body = _post(
            server,
            "/split-undo",
            {},
            token=server.token,
        )

        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert json.loads(body) == {"undone": True}
        assert paths["sibling"].read_bytes() == originals["sibling"]
        assert paths["sidecar"].read_bytes() == originals["sidecar"]
        assert paths["mapping"].read_bytes() == originals["mapping"]
        assert not paths["undo"].exists()


def test_confirmed_split_replays_and_enrolls_both_recomputed_centroids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        _originals,
        _logs,
        _observations,
    ):
        proposal = _preview_split(server)
        assert _confirm_split(server, proposal) == (200, {"new_id": "SPEAKER_02"})

        mapping = json.loads(paths["mapping"].read_bytes())
        mapping["speakers"] = {
            "SPEAKER_00": "Aoi",
            "SPEAKER_01": "",
            "SPEAKER_02": "Beryl",
        }
        named_mapping_bytes = _json_bytes(mapping, newline=True)
        paths["mapping"].write_bytes(named_mapping_bytes)

        split_sibling = json.loads(paths["sibling"].read_bytes())
        expected_turns = copy.deepcopy(split_sibling["speaker_turns"])
        expected_pair = (
            split_sibling["voiceprint_capture"],
            split_sibling["voiceprint_media"],
        )
        sidecar_bytes = paths["sidecar"].read_bytes()

        vtt_path = pipeline.split(paths["sibling"])

        replayed = json.loads(paths["sibling"].read_bytes())
        assert replayed["speaker_turns"] == expected_turns
        assert (
            replayed["voiceprint_capture"],
            replayed["voiceprint_media"],
        ) == expected_pair
        assert paths["sidecar"].read_bytes() == sidecar_bytes
        assert paths["mapping"].read_bytes() == named_mapping_bytes
        validate_voiceprint_conjunction(
            json.loads(sidecar_bytes),
            replayed,
            media_fingerprint(paths["media"]),
        )
        rendered = vtt_path.read_text(encoding="utf-8")
        assert "<v Aoi>" in rendered
        assert "<v Beryl>" in rendered

        store_path = tmp_path / "example-show.voices.json"
        speakers.enroll_speaker_voices(
            paths["media"],
            voices=store_path,
            show="Example Show",
        )
        store, validated_store = load_voice_store(store_path)
        identities = cast(dict[str, dict[str, Any]], store["identities"])
        enrolled = {
            identity["display_name"]: identity["exemplars"][0]["vector"]
            for identity in identities.values()
        }
        assert validated_store.revision == 1
        assert enrolled == {"Aoi": VECTOR_A, "Beryl": VECTOR_B}


@pytest.mark.parametrize("changed", ("sibling", "sidecar", "mapping"))
def test_undo_refuses_changed_post_split_inputs_without_restoring_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        _originals,
        _logs,
        _observations,
    ):
        proposal = _preview_split(server)
        assert _confirm_split(server, proposal) == (200, {"new_id": "SPEAKER_02"})
        paths[changed].write_bytes(paths[changed].read_bytes() + b" ")
        before_undo = {
            field: paths[field].read_bytes()
            for field in ("sibling", "sidecar", "mapping")
        }

        status, headers, body = _post(
            server,
            "/split-undo",
            {},
            token=server.token,
        )

        assert status == 409
        assert headers["Cache-Control"] == "no-store"
        assert "changed since the split" in json.loads(body)["error"]
        assert {
            field: paths[field].read_bytes()
            for field in ("sibling", "sidecar", "mapping")
        } == before_undo
        assert paths["undo"].is_file()


def test_undo_treats_a_deleted_rewritten_input_as_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _running_split_server(tmp_path, monkeypatch) as (
        server,
        paths,
        _originals,
        _logs,
        _observations,
    ):
        proposal = _preview_split(server)
        assert _confirm_split(server, proposal) == (200, {"new_id": "SPEAKER_02"})
        paths["mapping"].unlink()
        sibling_after = paths["sibling"].read_bytes()
        sidecar_after = paths["sidecar"].read_bytes()

        status, headers, body = _post(
            server,
            "/split-undo",
            {},
            token=server.token,
        )

        assert status == 409
        assert headers["Cache-Control"] == "no-store"
        error = json.loads(body)["error"]
        assert "mapping" in error and "changed since the split" in error
        assert paths["sibling"].read_bytes() == sibling_after
        assert paths["sidecar"].read_bytes() == sidecar_after
        assert not paths["mapping"].exists()
        assert paths["undo"].is_file()
