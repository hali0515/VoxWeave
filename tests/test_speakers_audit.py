"""Regression tests for the speaker review server and audition audit fixes."""

from __future__ import annotations

import http.client
import json
import shlex
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.test_voicecontrollers import _fake_clips, _write_episode
from voxweave import speakers, speakerserve


@pytest.fixture(autouse=True)
def _isolate_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(tmp_path / "cache-root"))
    monkeypatch.delenv("VOXWEAVE_VOICEPRINTS", raising=False)


@contextmanager
def _running_server(tmp_path: Path, *, speakers_on_disk: dict[str, str], page=None):
    mapping = tmp_path / "episode.speakers.json"
    mapping.write_text(
        json.dumps({"version": 1, "speakers": speakers_on_disk}), encoding="utf-8"
    )
    logs: list[str] = []
    server = speakerserve.make_server(
        page=page or "<!doctype html><title>audition</title>",
        media_path=tmp_path / "episode.mkv",
        mapping_path=mapping,
        sibling_path=tmp_path / "my episode.json",
        speaker_ids=("SPEAKER_00", "SPEAKER_01"),
        port=0,
        report=logs.append,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, mapping, logs
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(server, method: str, path: str, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, response.headers, payload


def _save(server, token: str, value: object):
    return _request(
        server,
        "POST",
        "/save",
        body=json.dumps(value).encode(),
        headers={"Content-Type": "application/json", "X-VoxWeave-Token": token},
    )


def test_serve_info_drops_stale_ids_left_by_rediarization(tmp_path):
    on_disk = {"SPEAKER_00": "Aoi", "SPEAKER_01": "Ren", "SPEAKER_02": "Gone"}
    with _running_server(tmp_path, speakers_on_disk=on_disk) as (server, mapping, logs):
        for _ in range(2):
            status, _headers, body = _request(server, "GET", "/serve-info")
            assert status == 200
            info = json.loads(body)
            assert info["speakers"] == {"SPEAKER_00": "Aoi", "SPEAKER_01": "Ren"}
        stale_reports = [line for line in logs if "SPEAKER_02" in line]
        assert len(stale_reports) == 1
        assert "no longer in this episode" in stale_reports[0]

        # The POST payload keeps the strict known-id check.
        status, _headers, _body = _save(
            server, info["token"], {"version": 1, "speakers": on_disk}
        )
        assert status == 400
        assert json.loads(mapping.read_bytes())["speakers"] == on_disk

        status, _headers, _body = _save(
            server, info["token"], {"version": 1, "speakers": info["speakers"]}
        )
        assert status == 200
        assert json.loads(mapping.read_bytes())["speakers"] == info["speakers"]


def test_serve_info_read_failure_is_reported_to_the_terminal(tmp_path):
    with _running_server(tmp_path, speakers_on_disk={}) as (server, mapping, logs):
        mapping.write_text("{not json", encoding="utf-8")
        status, headers, body = _request(server, "GET", "/serve-info")
        assert status == 500
        assert headers["Cache-Control"] == "no-store"
        assert json.loads(body) == {"error": "mapping could not be read"}
        assert server.token.encode() not in body
        assert any("Could not read the speaker mapping" in line for line in logs)


def test_save_disk_failure_replies_json_500_and_reports(tmp_path, monkeypatch):
    with _running_server(tmp_path, speakers_on_disk={}) as (server, mapping, logs):
        token = json.loads(_request(server, "GET", "/serve-info")[2])["token"]
        before = mapping.read_bytes()

        def disk_full(_path, _text):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(speakerserve.fsio, "atomic_write_text", disk_full)
        status, headers, body = _save(
            server, token, {"version": 1, "speakers": {"SPEAKER_00": "Aoi"}}
        )

        assert status == 500
        assert headers["Cache-Control"] == "no-store"
        assert json.loads(body) == {
            "error": "could not save the speaker mapping: No space left on device"
        }
        assert mapping.read_bytes() == before
        assert logs == [
            "Save failed: could not save the speaker mapping: No space left on device"
        ]
        # The session stays usable once the disk recovers.
        monkeypatch.undo()
        status, _headers, _body = _save(
            server, token, {"version": 1, "speakers": {"SPEAKER_00": "Aoi"}}
        )
        assert status == 200


def test_save_next_step_quotes_the_sibling_path(tmp_path):
    with _running_server(tmp_path, speakers_on_disk={}) as (server, _mapping, logs):
        token = json.loads(_request(server, "GET", "/serve-info")[2])["token"]
        assert _save(server, token, {"version": 1, "speakers": {}})[0] == 200
    sibling = tmp_path / "my episode.json"
    assert logs[-1] == f"Next: voxweave render {shlex.quote(str(sibling))}"
    assert "'" in logs[-1]


@pytest.mark.parametrize(
    ("host", "mentions_ngrok"),
    [
        ("localhost", False),
        ("attacker.invalid", False),
        ("demo.ngrok-free.app", True),
        ("demo.ngrok.io", True),
    ],
)
def test_forbidden_host_explains_what_is_accepted(tmp_path, host, mentions_ngrok):
    with _running_server(tmp_path, speakers_on_disk={}) as (server, _mapping, _logs):
        port = server.server_port
        status, _headers, body = _request(
            server, "GET", "/", headers={"Host": f"{host}:{port}"}
        )
    assert status == 403
    text = body.decode("utf-8")
    assert f"Host '{host}:{port}' is not allowed" in text
    assert "DNS rebinding" in text
    assert f"http://127.0.0.1:{port}/" in text
    assert "--host 0.0.0.0" in text
    assert ("--ngrok" in text) is mentions_ngrok


def test_idle_connections_time_out(tmp_path, monkeypatch):
    assert speakerserve._SpeakerRequestHandler.timeout == 30
    monkeypatch.setattr(speakerserve._SpeakerRequestHandler, "timeout", 0.2)
    with _running_server(tmp_path, speakers_on_disk={}) as (server, _mapping, _logs):
        with socket.create_connection(("127.0.0.1", server.server_port)) as idle:
            idle.settimeout(5)
            started = time.monotonic()
            assert idle.recv(1) == b""
            assert time.monotonic() - started < 4
        # The server keeps answering other clients.
        assert _request(server, "GET", "/")[0] == 200


def test_large_pages_are_written_intact(tmp_path):
    page = "<!doctype html>" + "x" * (3 * speakerserve._WRITE_CHUNK_BYTES + 17)
    with _running_server(tmp_path, speakers_on_disk={}, page=page) as (
        server,
        _mapping,
        _logs,
    ):
        status, headers, body = _request(server, "GET", "/")
    assert status == 200
    assert int(headers["Content-Length"]) == len(body) == len(page)
    assert body.decode("utf-8") == page


@pytest.mark.parametrize(
    ("host", "ngrok", "expected"),
    [
        ("127.0.0.1", False, None),
        ("0.0.0.0", False, "this machine on port 3210"),
        ("127.0.0.1", True, "the ngrok URL"),
        ("0.0.0.0", True, "this machine on port 3210 or the ngrok URL"),
    ],
)
def test_network_exposure_prints_a_no_password_warning(
    tmp_path, monkeypatch, host, ngrok, expected
):
    class InterruptingServer:
        origin = f"http://{host}:3210"
        server_port = 3210

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(
        speakerserve, "make_server", lambda **_kwargs: InterruptingServer()
    )
    reports: list[str] = []
    speakerserve.serve(
        page="page",
        media_path=tmp_path / "episode.mp4",
        mapping_path=tmp_path / "speakers.json",
        sibling_path=tmp_path / "episode.json",
        speaker_ids=(),
        host=host,
        ngrok=ngrok,
        open_browser=False,
        report=reports.append,
    )
    warnings = [line for line in reports if line.startswith("Warning:")]
    if expected is None:
        assert warnings == []
        return
    assert warnings == [
        f"Warning: anyone who can reach {expected} can play the episode audio, "
        "read and change speaker names and run splits — there is no password; "
        "stop the server when you are done."
    ]
    assert not warnings[0].startswith("http://")


def test_audition_page_does_not_offer_save_after_a_failed_session_load():
    page = speakers._render_audition_html(
        "episode.mkv",
        "speakers.json",
        {"SPEAKER_00": [((1.0, 3.0), "data:audio/mpeg;base64,Y2xpcA==")]},
        None,
    )
    load_failure = page[page.index("}).catch((error) => {") :]
    load_failure = load_failure[: load_failure.index("});")]
    assert "Could not load the saved names" in load_failure
    assert "(see the terminal); use Copy JSON" in load_failure
    assert "save.disabled = true;" in load_failure
    assert "Save failed" not in load_failure
    assert "splitErrorMessage(payload, `serve-info ${response.status}`)" in page


def test_purged_voiceprints_get_a_targeted_manual_hint(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    _fake_clips(monkeypatch)
    assert speakers.purge_voiceprints(media)

    with pytest.raises(RuntimeError) as refused:
        speakers.create_speaker_audition(media)

    message = str(refused.value)
    assert "voiceprints were removed (voxweave speakers purge)" in message
    assert f"`voxweave speakers serve {shlex.quote(str(media))} --manual`" in message
    assert "--no-match" not in message
    assert speakers.create_speaker_audition(media, no_match=True).speaker_ids == (
        "SPEAKER_00",
    )


def test_unusable_voiceprints_recommend_manual_not_no_match(tmp_path, monkeypatch):
    media, _sibling, sidecar = _write_episode(tmp_path)
    _fake_clips(monkeypatch)
    sidecar_path = tmp_path / "episode.voiceprints.json"
    sidecar_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="declared but not usable") as refused:
        speakers.create_speaker_audition(media)

    message = str(refused.value)
    assert "--no-match" not in message
    assert f"`voxweave speakers serve {shlex.quote(str(media))} --manual`" in message
    assert "speakers purge" not in message
