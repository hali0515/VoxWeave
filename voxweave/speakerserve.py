"""Local-only HTTP serving for the speaker audition page."""

from __future__ import annotations

import json
import secrets
import webbrowser
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from voxweave import fsio
from voxweave.voiceepisode import episode_lock

HOST = "127.0.0.1"
MAX_BODY_BYTES = 1_000_000
MAX_NAME_CHARS = 500


class SpeakerHTTPServer(ThreadingHTTPServer):
    """A loopback server carrying one in-memory audition session."""

    daemon_threads = True

    def __init__(
        self,
        *,
        page: str,
        media_path: Path,
        mapping_path: Path,
        sibling_path: Path,
        speaker_ids: Sequence[str],
        pristine_mapping_generation: fsio.FileGeneration | None,
        port: int,
        report: Callable[[str], None],
    ) -> None:
        self.page_bytes = page.encode("utf-8")
        self.media_path = Path(media_path)
        self.mapping_path = Path(mapping_path)
        self.sibling_path = Path(sibling_path)
        self.speaker_ids = tuple(speaker_ids)
        self.pristine_mapping_path = (
            self.mapping_path if pristine_mapping_generation is not None else None
        )
        self.pristine_mapping_generation = pristine_mapping_generation
        self.token = secrets.token_urlsafe(32)
        self.report = report
        super().__init__((HOST, port), _SpeakerRequestHandler)

    @property
    def authority(self) -> str:
        return f"{HOST}:{self.server_port}"

    @property
    def origin(self) -> str:
        return f"http://{self.authority}"


class _SpeakerRequestHandler(BaseHTTPRequestHandler):
    server: SpeakerHTTPServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _reply(
        self,
        status: HTTPStatus,
        body: bytes = b"",
        *,
        content_type: str = "text/plain; charset=utf-8",
        no_store: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_reply(
        self,
        status: HTTPStatus,
        value: object,
        *,
        no_store: bool = False,
    ) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self._reply(
            status,
            body,
            content_type="application/json; charset=utf-8",
            no_store=no_store,
        )

    def _host_allowed(self) -> bool:
        # A DNS-rebinding page resolves its own hostname to 127.0.0.1 and can
        # then read same-origin responses, so every route — not just the write
        # path — must refuse a Host other than the bound authority (the page
        # embeds private audio and serve-info discloses the save token).
        return self.headers.get_all("Host", []) == [self.server.authority]

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._reply(HTTPStatus.FORBIDDEN)
            return
        if self.path == "/":
            self._reply(
                HTTPStatus.OK,
                self.server.page_bytes,
                content_type="text/html; charset=utf-8",
                no_store=True,
            )
            return
        if self.path == "/serve-info":
            try:
                from voxweave import pipeline

                mapping_path = pipeline.speakers_mapping_path(self.server.media_path)
                speakers, generation = _mapping_entries(
                    mapping_path,
                    self.server.speaker_ids,
                )
                if (
                    mapping_path == self.server.pristine_mapping_path
                    and generation == self.server.pristine_mapping_generation
                ):
                    speakers = {}
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                self._json_reply(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "mapping could not be read"},
                    no_store=True,
                )
                return
            self._json_reply(
                HTTPStatus.OK,
                {
                    "token": self.server.token,
                    "mapping_name": mapping_path.name,
                    "speakers": speakers,
                },
                no_store=True,
            )
            return
        self._reply(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/save":
            self._reply(HTTPStatus.NOT_FOUND)
            return
        if not self._host_allowed():
            self._reply(HTTPStatus.FORBIDDEN)
            return
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] != self.server.origin):
            self._reply(HTTPStatus.FORBIDDEN)
            return
        tokens = self.headers.get_all("X-VoxWeave-Token", [])
        token = tokens[0] if len(tokens) == 1 else ""
        if not secrets.compare_digest(token, self.server.token):
            self._reply(HTTPStatus.FORBIDDEN)
            return
        length_values = self.headers.get_all("Content-Length", [])
        length_value = length_values[0] if len(length_values) == 1 else ""
        length = int(length_value) if length_value.isdecimal() else -1
        if length > MAX_BODY_BYTES:
            self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        if length < 0:
            self._reply(HTTPStatus.BAD_REQUEST)
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._reply(HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = _strict_json_loads(self.rfile.read(length))
            mapping = _validated_mapping(payload, self.server.speaker_ids)
        except (UnicodeError, ValueError, json.JSONDecodeError):
            self._reply(HTTPStatus.BAD_REQUEST)
            return
        ordered = {
            speaker_id: mapping[speaker_id]
            for speaker_id in self.server.speaker_ids
            if speaker_id in mapping
        }
        document = {"version": 1, "speakers": ordered}
        with episode_lock(self.server.media_path):
            from voxweave import pipeline

            mapping_path = pipeline.speakers_mapping_path(self.server.media_path)
            fsio.atomic_write_text(
                mapping_path,
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            )
        self.server.mapping_path = mapping_path
        self.server.pristine_mapping_generation = None
        self.server.pristine_mapping_path = None
        self.server.report(f"Saved {mapping_path}")
        self.server.report(f"Next: voxweave split {self.server.sibling_path}")
        self._json_reply(HTTPStatus.OK, {"saved": True})

    def do_HEAD(self) -> None:
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED)

    def do_PUT(self) -> None:
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED)

    def do_PATCH(self) -> None:
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED)

    def do_DELETE(self) -> None:
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED)

    def do_OPTIONS(self) -> None:
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED)

    def do_TRACE(self) -> None:
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED)

    def do_CONNECT(self) -> None:
        self._reply(HTTPStatus.METHOD_NOT_ALLOWED)


def _strict_json_loads(raw: bytes) -> Any:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )


def _validated_mapping(
    value: object,
    speaker_ids: Sequence[str],
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"version", "speakers"}:
        raise ValueError("mapping must contain only version and speakers")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("mapping version must be 1")
    speakers = value["speakers"]
    if not isinstance(speakers, dict):
        raise ValueError("speakers must be an object")
    known = set(speaker_ids)
    if any(not isinstance(key, str) or key not in known for key in speakers):
        raise ValueError("mapping contains an unknown speaker id")
    if any(
        not isinstance(name, str) or len(name) > MAX_NAME_CHARS
        for name in speakers.values()
    ):
        raise ValueError("speaker names must be strings of at most 500 characters")
    return speakers


def _file_generation(path: Path) -> tuple[int, int, int, int]:
    metadata = Path(path).stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _mapping_entries(
    path: Path, speaker_ids: Sequence[str]
) -> tuple[dict[str, str], tuple[int, int, int, int]]:
    before = _file_generation(path)
    value = _strict_json_loads(Path(path).read_bytes())
    after = _file_generation(path)
    if before != after:
        raise ValueError("mapping changed while reading")
    if not isinstance(value, dict):
        raise ValueError("mapping must be an object")
    if type(value.get("version")) is not int or value.get("version") != 1:
        raise ValueError("mapping version must be 1")
    speakers = value.get("speakers")
    if not isinstance(speakers, dict):
        raise ValueError("speakers must be an object")
    known = set(speaker_ids)
    if any(
        not isinstance(key, str)
        or key not in known
        or not isinstance(name, str)
        or len(name) > MAX_NAME_CHARS
        for key, name in speakers.items()
    ):
        raise ValueError("mapping contains invalid speaker entries")
    return speakers, after


def make_server(
    *,
    page: str,
    media_path: Path,
    mapping_path: Path,
    sibling_path: Path,
    speaker_ids: Sequence[str],
    pristine_mapping_generation: fsio.FileGeneration | None = None,
    port: int = 0,
    report: Callable[[str], None] = print,
) -> SpeakerHTTPServer:
    """Bind and return one loopback-only audition server."""
    return SpeakerHTTPServer(
        page=page,
        media_path=media_path,
        mapping_path=mapping_path,
        sibling_path=sibling_path,
        speaker_ids=speaker_ids,
        pristine_mapping_generation=pristine_mapping_generation,
        port=port,
        report=report,
    )


def serve(
    *,
    page: str,
    media_path: Path,
    mapping_path: Path,
    sibling_path: Path,
    speaker_ids: Sequence[str],
    pristine_mapping_generation: fsio.FileGeneration | None = None,
    port: int = 0,
    open_browser: bool = True,
    report: Callable[[str], None] = print,
) -> str:
    """Serve an audition until interrupted and return its loopback URL."""
    server = make_server(
        page=page,
        media_path=media_path,
        mapping_path=mapping_path,
        sibling_path=sibling_path,
        speaker_ids=speaker_ids,
        pristine_mapping_generation=pristine_mapping_generation,
        port=port,
        report=report,
    )
    url = f"{server.origin}/"
    report(url)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return url


__all__ = ["SpeakerHTTPServer", "make_server", "serve"]
