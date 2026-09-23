import http.client
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from voxweave import artifacts, ngrok, speakerserve


def _generation(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


@contextmanager
def _running_server(tmp_path: Path, *, host: str = "127.0.0.1", ngrok: bool = False):
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
        host=host,
        ngrok=ngrok,
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


def _request(
    server, method: str, path: str, body=None, headers=None, *, connect_host=None
):
    host, port = server.server_address
    host = connect_host or ("127.0.0.1" if host == "0.0.0.0" else host)
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
            f"Next: voxweave render {tmp_path / 'episode.json'}",
            f"Saved {mapping}",
            f"Next: voxweave render {tmp_path / 'episode.json'}",
        ]


@pytest.mark.parametrize("token", [None, "wrong-token"])
@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_save_requires_session_token(tmp_path, token, host):
    with _running_server(tmp_path, host=host) as (server, mapping, _logs):
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


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_save_rejects_foreign_origin(tmp_path, host):
    with _running_server(tmp_path, host=host) as (server, mapping, _logs):
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


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_reads_reject_foreign_host(tmp_path, host):
    with _running_server(tmp_path, host=host) as (server, _mapping, _logs):
        for path in ("/", "/serve-info"):
            status, _headers, body = _request(
                server,
                "GET",
                path,
                headers={"Host": "attacker.invalid"},
            )
            assert status == 403
            assert server.token.encode() not in body


@pytest.mark.parametrize("connect_host", ["127.0.0.1", "127.0.0.2"])
def test_wildcard_binding_supports_page_info_and_save(tmp_path, connect_host):
    with _running_server(tmp_path, host="0.0.0.0") as (server, mapping, _logs):
        assert server.server_address[0] == "0.0.0.0"
        assert server.origin == f"http://0.0.0.0:{server.server_port}"
        assert _request(server, "GET", "/", connect_host=connect_host)[0] == 200
        status, _, body = _request(
            server, "GET", "/serve-info", connect_host=connect_host
        )
        assert status == 200
        token = json.loads(body)["token"]
        status, _, _ = _request(
            server,
            "POST",
            "/save",
            body=b'{"version":1,"speakers":{"SPEAKER_00":"Ren"}}',
            headers={
                "Origin": f"http://{connect_host}:{server.server_port}",
                "X-VoxWeave-Token": token,
            },
            connect_host=connect_host,
        )
        assert status == 200
        assert json.loads(mapping.read_bytes())["speakers"] == {"SPEAKER_00": "Ren"}


def test_wildcard_binding_rejects_foreign_ip_port_and_mismatched_origin(tmp_path):
    with _running_server(tmp_path, host="0.0.0.0") as (server, mapping, _logs):
        before = mapping.read_bytes()
        for authority in (
            f"192.0.2.1:{server.server_port}",
            "127.0.0.1:0",
            f"localhost:{server.server_port}",
        ):
            for route in ("/", "/serve-info", "/save"):
                method = "POST" if route == "/save" else "GET"
                assert (
                    _request(server, method, route, headers={"Host": authority})[0]
                    == 403
                )
        token = _serve_info(server)["token"]
        assert (
            _save(
                server,
                token,
                {"version": 1, "speakers": {}},
                origin=server.origin,
            )[0]
            == 403
        )
        assert mapping.read_bytes() == before


@pytest.mark.parametrize("rewrite_host", [False, True])
def test_ngrok_discovers_late_tunnel_and_supports_https_save(
    tmp_path, monkeypatch, rewrite_host
):
    endpoints = []
    monkeypatch.setattr(ngrok, "_agent_endpoints", lambda: endpoints)
    tick = [10.0]
    monkeypatch.setattr(ngrok.time, "monotonic", lambda: tick[0])
    with _running_server(tmp_path, ngrok=True) as (server, mapping, _logs):
        authority = "random-tunnel.ngrok.app"
        headers = {"Host": authority}
        assert _request(server, "GET", "/", headers=headers)[0] == 403
        endpoints.append((f"https://{authority}", f"localhost:{server.server_port}"))
        tick[0] += 2
        assert _request(server, "GET", "/", headers=headers)[0] == 200
        if rewrite_host:
            headers["Host"] = f"localhost:{server.server_port}"
        status, _, body = _request(server, "GET", "/serve-info", headers=headers)
        assert status == 200
        token = json.loads(body)["token"]
        payload = b'{"version":1,"speakers":{"SPEAKER_00":"Ren"}}'
        headers["Origin"] = f"https://{authority}"
        assert _request(server, "POST", "/save", payload, headers)[0] == 403
        headers["X-VoxWeave-Token"] = token
        assert _request(server, "POST", "/save", payload, headers)[0] == 200
        assert json.loads(mapping.read_bytes())["speakers"] == {"SPEAKER_00": "Ren"}
        headers["Origin"] = "https://attacker.invalid"
        for route in ("/save", "/split", "/split-confirm", "/split-undo"):
            assert _request(server, "POST", route, payload, headers)[0] == 403

        endpoints[:] = [
            ("https://new-tunnel.ngrok.app", f"localhost:{server.server_port}")
        ]
        tick[0] += 2
        assert _request(server, "GET", "/", headers={"Host": authority})[0] == 403
        assert (
            _request(server, "GET", "/", headers={"Host": "new-tunnel.ngrok.app"})[0]
            == 200
        )
        endpoints.clear()
        tick[0] += 2
        assert (
            _request(server, "GET", "/", headers={"Host": "new-tunnel.ngrok.app"})[0]
            == 403
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_ngrok_rejects_unrelated_tunnels_and_forged_forwarded_headers(
    tmp_path, monkeypatch, enabled
):
    with _running_server(tmp_path, ngrok=enabled) as (server, _mapping, _logs):
        monkeypatch.setattr(
            ngrok,
            "_agent_endpoints",
            lambda: [
                ("https://ours.ngrok.app", f"localhost:{server.server_port}"),
                ("https://other.ngrok.app", "localhost:0"),
            ],
        )
        headers = {
            "Host": "ours.ngrok.app",
            "X-Forwarded-Host": "ours.ngrok.app",
            "X-Forwarded-Proto": "https",
        }
        assert _request(server, "GET", "/", headers=headers)[0] == (
            200 if enabled else 403
        )
        for host in ("other.ngrok.app", "attacker.invalid"):
            headers["Host"] = host
            assert _request(server, "GET", "/serve-info", headers=headers)[0] == 403


def test_save_accepts_its_exact_self_origin(tmp_path):
    with _running_server(tmp_path) as (server, mapping, _logs):
        token = _serve_info(server)["token"]
        status, _headers, _body = _save(
            server,
            token,
            {"version": 1, "speakers": {"SPEAKER_00": "Aoi"}},
            origin=server.origin,
        )
        assert status == 200
        assert json.loads(mapping.read_bytes())["speakers"] == {"SPEAKER_00": "Aoi"}


def test_save_rejects_foreign_host_before_other_failures(tmp_path):
    with _running_server(tmp_path) as (server, mapping, _logs):
        before = mapping.read_bytes()
        status, _headers, _body = _request(
            server,
            "POST",
            "/save",
            body=b"x",
            headers={
                "Host": "attacker.invalid",
                "Origin": "https://attacker.invalid",
                "X-VoxWeave-Token": "wrong-token",
                "Content-Length": str(speakerserve.MAX_BODY_BYTES + 1),
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
            body=b"x",
            headers={
                "X-VoxWeave-Token": "wrong-token",
                "Content-Length": str(speakerserve.MAX_BODY_BYTES + 1),
            },
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
            # Declare an oversized length instead of transmitting one: the server
            # rejects on the declared length without reading, and actually sending
            # the bytes races its early close into a client-side broken pipe.
            body=b"x",
            headers={
                "X-VoxWeave-Token": token,
                "Content-Length": str(speakerserve.MAX_BODY_BYTES + 1),
            },
        )
        assert status == 413
        assert mapping.read_bytes() == before


def test_server_never_creates_an_html_artifact(tmp_path):
    with _running_server(tmp_path) as (server, _mapping, _logs):
        assert _request(server, "GET", "/")[0] == 200
    assert not list(tmp_path.glob("*.html"))


@pytest.mark.parametrize(
    "method", ["HEAD", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"]
)
def test_other_http_methods_are_method_not_allowed(tmp_path, method):
    with _running_server(tmp_path) as (server, _mapping, _logs):
        assert _request(server, method, "/")[0] == 405


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize("open_browser", [False, True])
@pytest.mark.parametrize("ngrok", [False, True])
def test_serve_closes_cleanly_after_keyboard_interrupt(
    tmp_path, monkeypatch, host, open_browser, ngrok
):
    class InterruptingServer:
        origin = f"http://{host}:3210"
        closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True

    server = InterruptingServer()
    seen = {}
    monkeypatch.setattr(
        speakerserve, "make_server", lambda **kwargs: seen.update(kwargs) or server
    )
    opened = []
    monkeypatch.setattr(speakerserve.webbrowser, "open", opened.append)
    reports: list[str] = []

    result = speakerserve.serve(
        page="page",
        media_path=tmp_path / "episode.mp4",
        mapping_path=tmp_path / "speakers.json",
        sibling_path=tmp_path / "episode.json",
        speaker_ids=(),
        host=host,
        ngrok=ngrok,
        open_browser=open_browser,
        report=reports.append,
    )

    assert result == f"http://{host}:3210/"
    assert seen["host"] == host
    assert seen["ngrok"] is ngrok
    assert reports[0] == result
    assert len(reports) == 1 + (host == "0.0.0.0") + ngrok
    assert opened == (["http://127.0.0.1:3210/"] if open_browser else [])
    assert server.closed is True


def test_server_switches_to_legacy_writeback_if_adjacent_mapping_appears(tmp_path):
    media = tmp_path / "episode.mp4"
    media.write_bytes(b"media")
    cached = artifacts.claim_paths(media).speaker_mapping
    cached.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":"Cached"}}\n',
        encoding="utf-8",
    )
    server = speakerserve.make_server(
        page="page",
        media_path=media,
        mapping_path=cached,
        sibling_path=tmp_path / "episode.json",
        speaker_ids=("SPEAKER_00",),
        pristine_mapping_generation=_generation(cached),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        legacy = tmp_path / "episode.speakers.json"
        legacy.write_text(
            '{"version":1,"speakers":{"SPEAKER_00":""}}\n',
            encoding="utf-8",
        )
        info = _serve_info(server)
        assert info["speakers"] == {"SPEAKER_00": ""}
        status, _headers, _body = _save(
            server,
            info["token"],
            {"version": 1, "speakers": {"SPEAKER_00": "Winner"}},
        )
        assert status == 200
        assert json.loads(legacy.read_bytes())["speakers"] == {"SPEAKER_00": "Winner"}
        assert json.loads(cached.read_bytes())["speakers"] == {"SPEAKER_00": "Cached"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_pristine_skeleton_is_omitted_until_even_an_empty_save_occurs(tmp_path):
    media = tmp_path / "episode.mp4"
    media.write_bytes(b"media")
    mapping = artifacts.claim_paths(media).speaker_mapping
    mapping.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":""}}\n',
        encoding="utf-8",
    )
    server = speakerserve.make_server(
        page="page",
        media_path=media,
        mapping_path=mapping,
        sibling_path=tmp_path / "episode.json",
        speaker_ids=("SPEAKER_00",),
        pristine_mapping_generation=_generation(mapping),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        info = _serve_info(server)
        assert info["speakers"] == {}
        status, _headers, _body = _save(
            server,
            info["token"],
            {"version": 1, "speakers": {"SPEAKER_00": ""}},
        )
        assert status == 200
        assert _serve_info(server)["speakers"] == {"SPEAKER_00": ""}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_edit_between_audition_creation_and_server_start_is_not_suppressed(tmp_path):
    media = tmp_path / "episode.mp4"
    media.write_bytes(b"media")
    mapping = artifacts.claim_paths(media).speaker_mapping
    mapping.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":""}}\n',
        encoding="utf-8",
    )
    pristine_generation = _generation(mapping)
    mapping.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":"Saved before serve"}}\n',
        encoding="utf-8",
    )

    server = speakerserve.make_server(
        page="page",
        media_path=media,
        mapping_path=mapping,
        sibling_path=tmp_path / "episode.json",
        speaker_ids=("SPEAKER_00",),
        pristine_mapping_generation=pristine_generation,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _serve_info(server)["speakers"] == {"SPEAKER_00": "Saved before serve"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_legacy_mapping_save_ignores_an_unselected_poisoned_cache_claim(tmp_path):
    media = tmp_path / "episode.mp4"
    media.write_bytes(b"media")
    legacy = tmp_path / "episode.speakers.json"
    legacy.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":"Before"}}\n',
        encoding="utf-8",
    )
    poisoned = artifacts.artifacts_root(media) / "episode"
    poisoned.mkdir(parents=True)
    (poisoned / "source.json").write_text("not-json", encoding="utf-8")

    server = speakerserve.make_server(
        page="page",
        media_path=media,
        mapping_path=legacy,
        sibling_path=tmp_path / "episode.json",
        speaker_ids=("SPEAKER_00",),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        info = _serve_info(server)
        status, _headers, _body = _save(
            server,
            info["token"],
            {"version": 1, "speakers": {"SPEAKER_00": "After"}},
        )
        assert status == 200
        assert json.loads(legacy.read_bytes())["speakers"] == {"SPEAKER_00": "After"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
