import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voxweave import ngrok


@pytest.mark.parametrize("legacy", [False, True])
def test_agent_api_discovers_only_the_selected_local_upstream(monkeypatch, legacy):
    entries = []
    for public, upstream in (
        ("https://selected.ngrok.app", "http://127.0.0.1:9999"),
        ("https://another.ngrok.app", "http://localhost:8888"),
        ("https://remote.ngrok.app", "http://192.0.2.1:9999"),
    ):
        entries.append(
            {"public_url": public, "config": {"addr": upstream}}
            if legacy
            else {"url": public, "upstream": {"url": upstream}}
        )
    requests = []

    class AgentHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(self.path)
            if legacy and self.path == "/api/endpoints":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"tunnels" if legacy else "endpoints": entries}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    monkeypatch.setattr(ngrok, "_AGENT_PORT", server.server_port)
    # Discovery must stay on the loopback socket even with proxy environment variables.
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:1")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert ngrok.NgrokOrigins(9999).get() == {"https://selected.ngrok.app"}
        assert requests == (
            ["/api/endpoints", "/api/tunnels"] if legacy else ["/api/endpoints"]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "public,upstream,expected",
    [
        ("https://random.ngrok.app/", "9999", "https://random.ngrok.app"),
        ("https://RANDOM.ngrok.app:443", "localhost:9999", "https://random.ngrok.app"),
        ("http://custom.example:8080", "0.0.0.0:9999", "http://custom.example:8080"),
        ("https://random.ngrok.app", "localhost:invalid", None),
        ("https://random.ngrok.app", "https://localhost:9999", None),
        ("https://random.ngrok.app/path", "localhost:9999", None),
        ("https://user@random.ngrok.app", "localhost:9999", None),
        ("https://*.ngrok.app", "localhost:9999", None),
        ("tcp://0.tcp.ngrok.io:1234", "localhost:9999", None),
    ],
)
def test_discovery_normalizes_origins_and_filters_invalid_endpoints(
    public, upstream, expected
):
    assert ngrok._origin_for_port(public, upstream, 9999) == expected


def test_agent_failure_expires_previous_domains_and_recovers(monkeypatch):
    tick = [10.0]
    monkeypatch.setattr(ngrok.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(
        ngrok, "_agent_endpoints", lambda: [("https://old.ngrok.app", "localhost:9999")]
    )
    origins = ngrok.NgrokOrigins(9999)
    assert origins.get() == {"https://old.ngrok.app"}

    def unavailable():
        raise ConnectionRefusedError

    monkeypatch.setattr(ngrok, "_agent_endpoints", unavailable)
    tick[0] += 2
    assert origins.get() == set()
    monkeypatch.setattr(
        ngrok, "_agent_endpoints", lambda: [("https://new.ngrok.app", "localhost:9999")]
    )
    tick[0] += 2
    assert origins.get() == {"https://new.ngrok.app"}
