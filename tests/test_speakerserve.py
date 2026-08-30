import http.client
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from voxweave import speakerserve


@contextmanager
def _running_server(tmp_path: Path):
    mapping = tmp_path / "episode.speakers.json"
    mapping.write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {"SPEAKER_00": "Aoi", "SPEAKER_01": ""},
            }
        ),
        encoding="utf-8",
    )
    logs: list[str] = []
    server = speakerserve.make_server(
        page="<!doctype html><title>audition</title>",
        media_path=tmp_path / "episode.mkv",
        mapping_path=mapping,
        sibling_path=tmp_path / "episode.json",
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
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, response.headers, payload


def _serve_info(server) -> dict[str, object]:
    status, headers, body = _request(server, "GET", "/serve-info")
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    return json.loads(body)


def _save(server, token: str, value: object, *, origin: str | None = None):
    headers = {
        "Content-Type": "application/json",
        "X-VoxWeave-Token": token,
    }
    if origin is not None:
        headers["Origin"] = origin
    return _request(
        server,
        "POST",
        "/save",
        body=json.dumps(value).encode(),
        headers=headers,
    )


def test_get_page_and_serve_info_rereads_saved_mapping(tmp_path):
    with _running_server(tmp_path) as (server, mapping, _logs):
        status, headers, body = _request(server, "GET", "/")
        assert status == 200
        assert headers.get_content_type() == "text/html"
        assert body == b"<!doctype html><title>audition</title>"

        first = _serve_info(server)
        assert first["mapping_name"] == "episode.speakers.json"
        assert first["speakers"] == {"SPEAKER_00": "Aoi", "SPEAKER_01": ""}
        assert isinstance(first["token"], str) and first["token"]

        mapping.write_text(
            '{"version":1,"speakers":{"SPEAKER_00":"Ren"}}\n',
            encoding="utf-8",
        )
        assert _serve_info(server)["speakers"] == {"SPEAKER_00": "Ren"}


def test_save_writes_exact_bytes_in_skeleton_order_and_can_overwrite(tmp_path):
    with _running_server(tmp_path) as (server, mapping, logs):
        token = _serve_info(server)["token"]
        status, _headers, body = _save(
            server,
            token,
            {
                "version": 1,
                "speakers": {"SPEAKER_01": "Ren", "SPEAKER_00": "Aoi"},
            },
        )
        assert status == 200
        assert json.loads(body) == {"saved": True}
        assert mapping.read_bytes() == (
            b'{\n  "version": 1,\n  "speakers": {\n'
            b'    "SPEAKER_00": "Aoi",\n    "SPEAKER_01": "Ren"\n  }\n}\n'
        )

        status, _headers, _body = _save(
            server,
            token,
            {"version": 1, "speakers": {"SPEAKER_00": "Aster"}},
        )
        assert status == 200
        assert json.loads(mapping.read_text(encoding="utf-8")) == {
            "version": 1,
            "speakers": {"SPEAKER_00": "Aster"},
        }
        assert logs == [
            f"Saved {mapping}",
            f"Next: voxweave split {tmp_path / 'episode.json'}",
            f"Saved {mapping}",
            f"Next: voxweave split {tmp_path / 'episode.json'}",
        ]


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_save_requires_session_token(tmp_path, token):
    with _running_server(tmp_path) as (server, mapping, _logs):
        before = mapping.read_bytes()
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["X-VoxWeave-Token"] = token
        status, _headers, _body = _request(
            server,
            "POST",
            "/save",
            body=b'{"version":1,"speakers":{}}',
            headers=headers,
        )
        assert status == 403
        assert mapping.read_bytes() == before


def test_save_rejects_foreign_origin(tmp_path):
    with _running_server(tmp_path) as (server, mapping, _logs):
        before = mapping.read_bytes()
        token = _serve_info(server)["token"]
        status, _headers, _body = _save(
            server,
            token,
            {"version": 1, "speakers": {}},
            origin="https://attacker.invalid",
        )
        assert status == 403
        assert mapping.read_bytes() == before


def test_save_rejects_foreign_host_before_other_failures(tmp_path):
    with _running_server(tmp_path) as (server, mapping, _logs):
        before = mapping.read_bytes()
        status, _headers, _body = _request(
            server,
            "POST",
            "/save",
            body=b"x" * 1_000_001,
            headers={
                "Host": "attacker.invalid",
                "Origin": "https://attacker.invalid",
                "X-VoxWeave-Token": "wrong-token",
            },
        )
        assert status == 403
        assert mapping.read_bytes() == before


def test_save_checks_token_before_body_limit(tmp_path):
    with _running_server(tmp_path) as (server, mapping, _logs):
        before = mapping.read_bytes()
        status, _headers, _body = _request(
            server,
            "POST",
            "/save",
            body=b"x" * 1_000_001,
            headers={"X-VoxWeave-Token": "wrong-token"},
        )
        assert status == 403
        assert mapping.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1, "speakers": {"UNKNOWN": "Name"}},
        {"version": 1, "speakers": []},
        {"version": True, "speakers": {}},
        {"version": 1, "speakers": {"SPEAKER_00": 7}},
        {"version": 1, "speakers": {"SPEAKER_00": "x" * 501}},
        {"version": 1, "speakers": {}, "extra": True},
    ],
)
def test_save_rejects_unknown_ids_and_bad_shapes(tmp_path, payload):
    with _running_server(tmp_path) as (server, mapping, _logs):
        before = mapping.read_bytes()
        token = _serve_info(server)["token"]
        status, _headers, _body = _save(server, token, payload)
        assert status == 400
        assert mapping.read_bytes() == before


def test_save_rejects_oversized_body_before_reading_it(tmp_path):
    with _running_server(tmp_path) as (server, mapping, _logs):
        before = mapping.read_bytes()
        token = _serve_info(server)["token"]
        status, _headers, _body = _request(
            server,
            "POST",
            "/save",
            body=b"x" * 1_000_001,
            headers={"X-VoxWeave-Token": token},
        )
        assert status == 413
        assert mapping.read_bytes() == before


def test_server_never_creates_an_html_artifact(tmp_path):
    with _running_server(tmp_path) as (server, _mapping, _logs):
        assert _request(server, "GET", "/")[0] == 200
    assert not list(tmp_path.glob("*.html"))
