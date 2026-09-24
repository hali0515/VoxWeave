"""Discover public origins from the local ngrok agent without creating tunnels."""

from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection, HTTPException
from urllib.parse import urlsplit

_AGENT_PORT = 4040
_MAX_RESPONSE_BYTES = 1_000_000


def _agent_endpoints() -> list[tuple[str, str]]:
    for route, key in (("/api/endpoints", "endpoints"), ("/api/tunnels", "tunnels")):
        connection = HTTPConnection("127.0.0.1", _AGENT_PORT, timeout=1)
        try:
            connection.request("GET", route)
            response = connection.getresponse()
            if response.status == 404 and key == "endpoints":
                continue
            if response.status != 200:
                return []
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(body) > _MAX_RESPONSE_BYTES:
                return []
            data = json.loads(body)
        finally:
            connection.close()
        entries = data.get(key) if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        result = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            public = entry.get("url" if key == "endpoints" else "public_url")
            config = entry.get("upstream" if key == "endpoints" else "config")
            upstream = (
                config.get("url" if key == "endpoints" else "addr")
                if isinstance(config, dict)
                else None
            )
            if isinstance(public, str) and isinstance(upstream, str):
                result.append((public, upstream))
        return result
    return []


def _origin_for_port(public: str, upstream: str, port: int) -> str | None:
    if upstream.isdecimal():
        upstream = f"127.0.0.1:{upstream}"
    if "://" not in upstream:
        upstream = f"http://{upstream}"
    try:
        target = urlsplit(upstream)
        target_port = target.port if target.port is not None else 80
        if (
            target.scheme != "http"
            or target.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}
            or target_port != port
        ):
            return None
        url = urlsplit(public)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or any(c.isspace() or c in "\\*" for c in public)
        ):
            return None
        host = url.hostname.encode("idna").decode("ascii")
        if ":" in host:
            host = f"[{host}]"
        if url.port is not None and url.port != (443 if url.scheme == "https" else 80):
            host = f"{host}:{url.port}"
        return f"{url.scheme}://{host}"
    except (ValueError, UnicodeError):
        return None


class NgrokOrigins:
    """Track only HTTP endpoints forwarding to this server's local port."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._lock = threading.Lock()
        self._expires = 0.0
        self._origins: frozenset[str] = frozenset()

    def get(self) -> frozenset[str]:
        with self._lock:
            if time.monotonic() < self._expires:
                return self._origins
            try:
                endpoints = _agent_endpoints()
            except (OSError, HTTPException, ValueError):
                endpoints = []
            self._origins = frozenset(
                origin
                for public, upstream in endpoints
                if (origin := _origin_for_port(public, upstream, self.port)) is not None
            )
            self._expires = time.monotonic() + 1
            return self._origins
