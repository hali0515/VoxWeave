"""HTTP serving for the speaker audition page, with loopback binding by default.

Access is by link: the URL the server prints and opens carries a random access
key (``/?k=...``, new for every start). Opening it sets an HttpOnly,
SameSite=Strict session cookie and moves the browser to the clean ``/``; every
route (the page, ``/serve-info``, ``/save`` and the split routes) then requires
that cookie, on 127.0.0.1 too. The Host/Origin checks against DNS rebinding and
the per-session save token of the POST routes apply on top of it.
"""

from __future__ import annotations

import json
import secrets
import shlex
import threading
import webbrowser
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from voxweave import fsio, sidecars, turnembed
from voxweave.ngrok import NgrokOrigins
from voxweave.speakerserve_mapping import (
    MAX_NAME_CHARS as MAX_NAME_CHARS,
    _check_names as _check_names,
    _file_generation as _file_generation,
    _mapping_entries as _mapping_entries,
    _mapping_speakers as _mapping_speakers,
    _strict_json_loads as _strict_json_loads,
    _validated_mapping as _validated_mapping,
    _validated_speakers as _validated_speakers,
)
from voxweave.speakerserve_split import (
    MAX_UNDO_BYTES as MAX_UNDO_BYTES,
    _SPLIT_B_PLACEHOLDER as _SPLIT_B_PLACEHOLDER,
    _SPLIT_GROUP_LABEL as _SPLIT_GROUP_LABEL,
    _UNDO_FILE_LABELS as _UNDO_FILE_LABELS,
    SplitConflict as SplitConflict,
    _absolute as _absolute,
    _build_split_proposal as _build_split_proposal,
    _capture_mismatch as _capture_mismatch,
    _confirm_split as _confirm_split,
    _decoded_undo_record as _decoded_undo_record,
    _json_bytes as _json_bytes,
    _load_undo as _load_undo,
    _mapping_document as _mapping_document,
    _next_speaker_id as _next_speaker_id,
    _prepare_split_wav as _prepare_split_wav,
    _proposal_groups as _proposal_groups,
    _recheck_split_inputs as _recheck_split_inputs,
    _require_embedding_identity as _require_embedding_identity,
    _restore_files as _restore_files,
    _sha256 as _sha256,
    _SplitProposal as _SplitProposal,
    _split_recipe_centroids as _split_recipe_centroids,
    _stage_split_inputs as _stage_split_inputs,
    _StagedProvenance as _StagedProvenance,
    _staged_provenance as _staged_provenance,
    _StagedSplit as _StagedSplit,
    _undo_bytes as _undo_bytes,
    _undo_record as _undo_record,
    _undo_split as _undo_split,
    _validate_confirmation as _validate_confirmation,
    _validated_speaker_request as _validated_speaker_request,
    _write_bytes as _write_bytes,
)
from voxweave.voicebase import (
    Phase2DataError,
)
from voxweave.voiceepisode import episode_lock

HOST = "127.0.0.1"
MAX_BODY_BYTES = 10_000_000
_POST_ROUTES = frozenset({"/save", "/split", "/split-confirm", "/split-undo"})
_INVALID_BODY = object()
# Per socket operation. sendall() treats its timeout as a total deadline, so
# replies are written in chunks: a slow but progressing client is not cut off.
_SOCKET_TIMEOUT_S = 30
_WRITE_CHUNK_BYTES = 64 * 1024
# The query parameter of the access link (``/?k=<access key>``).
_ACCESS_KEY_PARAM = "k"
_NO_SESSION_MESSAGE = (
    b"This audition needs its access link: open the /?k=... link that "
    b"voxweave speakers serve printed when it started.\n"
)
_BAD_KEY_MESSAGE = (
    b"This access key is not valid for this server (a new one is made every "
    b"time voxweave speakers serve starts); open the link it printed.\n"
)
# The answer to a valid access link. Not a 3xx: after a cross-site navigation
# (a link tapped in a chat or mail app) browsers withhold SameSite=Strict
# cookies from the redirected request too, so a 303 would land on a 403. A
# navigation this same-origin page starts is same-site and carries the cookie;
# location.replace also drops the key from the tab's history.
_ENTER_PAGE = (
    b'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    b'<meta name="referrer" content="no-referrer">\n<title>VoxWeave</title>\n'
    b"</head>\n<body>\n<script>location.replace('/');</script>\n"
    b'<noscript><a href="/">Open the audition</a></noscript>\n</body>\n</html>\n'
)
# Hosts that can only be reached through an ngrok tunnel.
_NGROK_HOST_SUFFIXES = (
    ".ngrok-free.app",
    ".ngrok-free.dev",
    ".ngrok.app",
    ".ngrok.dev",
    ".ngrok.io",
)


class SpeakerHTTPServer(ThreadingHTTPServer):
    """An HTTP server carrying one in-memory audition session."""

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
        host: str = HOST,
        ngrok: bool = False,
        render: Callable[[], str] | None = None,
    ) -> None:
        if host not in (HOST, "0.0.0.0"):
            raise ValueError("host must be 127.0.0.1 or 0.0.0.0")
        self.page_bytes = page.encode("utf-8")
        self.media_path = Path(media_path)
        self.sibling_path = Path(sibling_path)
        self.speaker_ids = tuple(speaker_ids)
        # Every read resolves the mapping path afresh; this one only identifies
        # the untouched skeleton the audition was generated with.
        self.pristine_mapping_path = (
            Path(mapping_path) if pristine_mapping_generation is not None else None
        )
        self.pristine_mapping_generation = pristine_mapping_generation
        # Access link key -> session cookie (see the module docstring); the
        # token guards the POST routes as before.
        self.access_key = secrets.token_urlsafe(32)
        self.session_cookie = secrets.token_urlsafe(32)
        self.token = secrets.token_urlsafe(32)
        self.report = report
        # Re-renders the episode's subtitles after a Save and returns the VTT's
        # name; None leaves rendering to the user (speakers serve --no-render).
        self.render = render
        self.action_lock = threading.Lock()
        self.split_proposal: _SplitProposal | None = None
        self.session_terminal = False
        self.reported_stale_ids: set[str] = set()
        super().__init__((host, port), _SpeakerRequestHandler)
        self.ngrok_origins = NgrokOrigins(self.server_port) if ngrok else None

    @property
    def authority(self) -> str:
        return f"{self.server_address[0]}:{self.server_port}"

    @property
    def origin(self) -> str:
        return f"http://{self.authority}"

    @property
    def access_url(self) -> str:
        """The link to open: the page's URL plus this session's access key."""
        return f"{self.origin}/?{_ACCESS_KEY_PARAM}={self.access_key}"

    @property
    def cookie_name(self) -> str:
        # Cookies are not scoped by port: a per-port name keeps two auditions
        # served from one host from replacing each other's session cookie.
        return f"voxweave_session_{self.server_port}"


def _same_secret(candidate: str, expected: str) -> bool:
    """Constant-time comparison that also accepts non-ASCII input."""
    return secrets.compare_digest(
        candidate.encode("utf-8", "surrogateescape"), expected.encode("utf-8")
    )


class _SpeakerRequestHandler(BaseHTTPRequestHandler):
    server: SpeakerHTTPServer
    # StreamRequestHandler applies this to the socket, so idle or trickling
    # connections cannot pin handler threads when the server is reachable
    # from the network (--host 0.0.0.0 / --ngrok).
    timeout = _SOCKET_TIMEOUT_S

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _reply(
        self,
        status: HTTPStatus,
        body: bytes = b"",
        *,
        content_type: str = "text/plain; charset=utf-8",
        no_store: bool = False,
        extra_headers: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        view = memoryview(body)
        for offset in range(0, len(view), _WRITE_CHUNK_BYTES):
            self.wfile.write(view[offset : offset + _WRITE_CHUNK_BYTES])

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

    def _allowed_origins(self) -> set[str]:
        # Every route checks Host because the page embeds audio and serve-info
        # discloses the save token. Wildcard binding also accepts the connection's
        # local IP, never arbitrary hostnames that could enable DNS rebinding.
        authorities = {self.server.authority}
        if self.server.server_address[0] == "0.0.0.0":
            local_host = self.connection.getsockname()[0]
            authorities.add(f"{local_host}:{self.server.server_port}")
        if self.server.ngrok_origins is not None:
            authorities.add(f"localhost:{self.server.server_port}")
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            return set()
        host = hosts[0]
        local = host in authorities
        origins = {f"http://{host}"} if local else set()
        if self.server.ngrok_origins is not None and (
            not local
            or any(o not in origins for o in self.headers.get_all("Origin", []))
        ):
            origins.update(
                origin
                for origin in self.server.ngrok_origins.get()
                if local or urlsplit(origin).netloc == host
            )
        return origins

    def _forbidden_host_message(self) -> bytes:
        if self.server.ngrok_origins is not None:
            return (
                b"No matching ngrok endpoint. Check the local agent at 127.0.0.1:4040 "
                b"and forward it to this server's port.\n"
            )
        hosts = self.headers.get_all("Host", [])
        host = hosts[0] if len(hosts) == 1 else ""
        # Never the access key itself: whoever sent this request may not have it.
        message = (
            f"Host '{host[:200]}' is not allowed (hostnames are refused to prevent "
            f"DNS rebinding); open the http://{HOST}:{self.server.server_port}/?"
            f"{_ACCESS_KEY_PARAM}=... link that voxweave speakers serve printed, "
            "or with --host 0.0.0.0 use this machine's IP address in it."
        )
        if host.split(":", 1)[0].lower().endswith(_NGROK_HOST_SUFFIXES):
            message += (
                " For an ngrok tunnel, start voxweave speakers serve with --ngrok."
            )
        return f"{message}\n".encode()

    def _has_session(self) -> bool:
        """Whether the request carries this server's session cookie."""
        name = self.server.cookie_name
        for header in self.headers.get_all("Cookie", []):
            for pair in header.split(";"):
                key, separator, value = pair.strip().partition("=")
                if (
                    separator
                    and key == name
                    and _same_secret(value, self.server.session_cookie)
                ):
                    return True
        return False

    def _enter_with_access_key(self) -> bool:
        """Answer ``/?k=...``: trade a valid access key for the session cookie.

        Returns False for any other path, which the caller routes as usual.
        """
        parts = urlsplit(self.path)
        if parts.path != "/" or not parts.query:
            return False
        try:
            query = parse_qs(parts.query, keep_blank_values=True, strict_parsing=True)
        except ValueError:
            query = {}
        keys = query.get(_ACCESS_KEY_PARAM, [])
        if (
            set(query) != {_ACCESS_KEY_PARAM}
            or len(keys) != 1
            or not _same_secret(keys[0], self.server.access_key)
        ):
            self._reply(HTTPStatus.FORBIDDEN, _BAD_KEY_MESSAGE)
            return True
        cookie = (
            f"{self.server.cookie_name}={self.server.session_cookie}; "
            "Path=/; HttpOnly; SameSite=Strict"
        )
        hosts = self.headers.get_all("Host", [])
        if f"https://{hosts[0]}" in self._allowed_origins():
            # Reached through an HTTPS tunnel: never send the cookie in clear.
            cookie += "; Secure"
        self._reply(
            HTTPStatus.OK,
            _ENTER_PAGE,
            content_type="text/html; charset=utf-8",
            no_store=True,
            extra_headers=(
                ("Set-Cookie", cookie),
                ("Referrer-Policy", "no-referrer"),
            ),
        )
        return True

    def do_GET(self) -> None:
        if not self._allowed_origins():
            self._reply(
                HTTPStatus.FORBIDDEN,
                self._forbidden_host_message(),
            )
            return
        if self._enter_with_access_key():
            return
        if not self._has_session():
            self._reply(HTTPStatus.FORBIDDEN, _NO_SESSION_MESSAGE)
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
                with self.server.action_lock:
                    mapping_path = sidecars.speakers_mapping_path(
                        self.server.media_path
                    )
                    speakers, stale, generation = _mapping_entries(
                        mapping_path,
                        self.server.speaker_ids,
                    )
                    if (
                        mapping_path == self.server.pristine_mapping_path
                        and generation == self.server.pristine_mapping_generation
                    ):
                        speakers = {}
                    unreported = sorted(set(stale) - self.server.reported_stale_ids)
                    if unreported:
                        self.server.reported_stale_ids.update(unreported)
                        self.server.report(
                            f"{mapping_path.name}: ignoring speaker id(s) no longer "
                            f"in this episode: {', '.join(unreported)}; the next "
                            "Save drops them"
                        )
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                self.server.report(f"Could not read the speaker mapping: {exc}")
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
        if not self._allowed_origins() or not self._has_session():
            self._reply(HTTPStatus.FORBIDDEN)
            return
        if self.path not in _POST_ROUTES:
            self._reply(HTTPStatus.NOT_FOUND)
            return
        payload = self._guarded_json_body()
        if payload is _INVALID_BODY:
            return
        with self.server.action_lock:
            if self.path == "/save":
                self._handle_save(payload)
            elif self.path == "/split":
                self._handle_split(payload)
            elif self.path == "/split-confirm":
                self._handle_split_confirm(payload)
            else:
                self._handle_split_undo(payload)

    def _guarded_json_body(self) -> object:
        allowed_origins = self._allowed_origins()
        if not allowed_origins:
            self._reply(HTTPStatus.FORBIDDEN)
            return _INVALID_BODY
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] not in allowed_origins):
            self._reply(HTTPStatus.FORBIDDEN)
            return _INVALID_BODY
        tokens = self.headers.get_all("X-VoxWeave-Token", [])
        token = tokens[0] if len(tokens) == 1 else ""
        if not _same_secret(token, self.server.token):
            self._reply(HTTPStatus.FORBIDDEN)
            return _INVALID_BODY
        length_values = self.headers.get_all("Content-Length", [])
        length_value = length_values[0] if len(length_values) == 1 else ""
        length = int(length_value) if length_value.isdecimal() else -1
        if length > MAX_BODY_BYTES:
            self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return _INVALID_BODY
        if length < 0:
            self._reply(HTTPStatus.BAD_REQUEST)
            return _INVALID_BODY
        if self.headers.get("Transfer-Encoding") is not None:
            self._reply(HTTPStatus.BAD_REQUEST)
            return _INVALID_BODY
        try:
            return _strict_json_loads(self.rfile.read(length))
        except (UnicodeError, ValueError, json.JSONDecodeError):
            self._reply(HTTPStatus.BAD_REQUEST)
            return _INVALID_BODY

    def _handle_save(self, payload: object) -> None:
        if self.server.session_terminal:
            self._json_reply(
                HTTPStatus.CONFLICT,
                {"error": "restart voxweave speakers serve before saving again"},
                no_store=True,
            )
            return
        try:
            mapping = _validated_mapping(payload, self.server.speaker_ids)
        except ValueError:
            self._reply(HTTPStatus.BAD_REQUEST)
            return
        ordered = {
            speaker_id: mapping[speaker_id]
            for speaker_id in self.server.speaker_ids
            if speaker_id in mapping
        }
        document = {"version": 1, "speakers": ordered}
        try:
            with episode_lock(self.server.media_path):
                mapping_path = sidecars.speakers_mapping_path(self.server.media_path)
                fsio.atomic_write_text(
                    mapping_path,
                    json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    private=True,
                )
        except OSError as exc:
            message = f"could not save the speaker mapping: {exc.strerror or exc}"
            self.server.report(f"Save failed: {message}")
            self._json_reply(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": message},
                no_store=True,
            )
            return
        self.server.pristine_mapping_generation = None
        self.server.pristine_mapping_path = None
        self.server.report(f"Saved {mapping_path}")
        reply: dict[str, object] = {"saved": True}
        if self.server.render is None:
            self.server.report(
                f"Next: voxweave render {shlex.quote(str(self.server.sibling_path))}"
            )
        else:
            try:
                reply["rendered"] = self.server.render()
            except Exception as exc:  # noqa: BLE001 - the save stands; say why render did not
                message = str(exc) or type(exc).__name__
                self.server.report(f"Subtitles not updated: {message}")
                reply["render_error"] = message
            else:
                self.server.report(f"Rendered {reply['rendered']}")
        self._json_reply(HTTPStatus.OK, reply)

    def _handle_split(self, payload: object) -> None:
        if self.server.session_terminal:
            self._json_reply(
                HTTPStatus.CONFLICT,
                {"error": "restart voxweave speakers serve before another split"},
                no_store=True,
            )
            return
        try:
            speaker_id = _validated_speaker_request(
                payload,
                self.server.speaker_ids,
            )
            self.server.split_proposal = None
            proposal, response = _build_split_proposal(self.server, speaker_id)
        except turnembed.UnsplittableSpeakerError as exc:
            self._json_reply(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": str(exc)},
                no_store=True,
            )
            return
        except (SplitConflict, Phase2DataError) as exc:
            self._json_reply(
                HTTPStatus.CONFLICT,
                {"error": str(exc)},
                no_store=True,
            )
            return
        except ValueError:
            self._reply(HTTPStatus.BAD_REQUEST)
            return
        except (OSError, RuntimeError) as exc:
            self._json_reply(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"speaker split failed: {exc}"},
                no_store=True,
            )
            return
        self.server.split_proposal = proposal
        self._json_reply(HTTPStatus.OK, response, no_store=True)

    def _handle_split_confirm(self, payload: object) -> None:
        if self.server.session_terminal:
            self._json_reply(
                HTTPStatus.CONFLICT,
                {"error": "the audition session already changed; restart it"},
                no_store=True,
            )
            return
        try:
            proposal = self.server.split_proposal
            if proposal is None:
                raise SplitConflict(
                    "no current split proposal; preview the split again"
                )
            _validate_confirmation(payload, proposal)
            new_id = _confirm_split(self.server, proposal)
        except (SplitConflict, Phase2DataError, turnembed.TurnEmbeddingError) as exc:
            self._json_reply(
                HTTPStatus.CONFLICT,
                {"error": str(exc)},
                no_store=True,
            )
            return
        except ValueError:
            self._reply(HTTPStatus.BAD_REQUEST)
            return
        except (OSError, RuntimeError) as exc:
            self._json_reply(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"speaker split could not be applied: {exc}"},
                no_store=True,
            )
            return
        self._json_reply(HTTPStatus.OK, {"new_id": new_id}, no_store=True)

    def _handle_split_undo(self, payload: object) -> None:
        if not isinstance(payload, dict) or payload:
            self._reply(HTTPStatus.BAD_REQUEST)
            return
        try:
            _undo_split(self.server)
        except (SplitConflict, Phase2DataError, ValueError) as exc:
            self._json_reply(
                HTTPStatus.CONFLICT,
                {"error": str(exc)},
                no_store=True,
            )
            return
        except (OSError, RuntimeError) as exc:
            self._json_reply(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"speaker split undo failed: {exc}"},
                no_store=True,
            )
            return
        self._json_reply(HTTPStatus.OK, {"undone": True}, no_store=True)

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


def make_server(
    *,
    page: str,
    media_path: Path,
    mapping_path: Path,
    sibling_path: Path,
    speaker_ids: Sequence[str],
    pristine_mapping_generation: fsio.FileGeneration | None = None,
    host: str = HOST,
    ngrok: bool = False,
    port: int = 0,
    report: Callable[[str], None] = print,
    render: Callable[[], str] | None = None,
) -> SpeakerHTTPServer:
    """Bind and return one audition server."""
    return SpeakerHTTPServer(
        page=page,
        media_path=media_path,
        mapping_path=mapping_path,
        sibling_path=sibling_path,
        speaker_ids=speaker_ids,
        pristine_mapping_generation=pristine_mapping_generation,
        host=host,
        ngrok=ngrok,
        port=port,
        report=report,
        render=render,
    )


def _exposure_warning(*, host: str) -> str:
    """Warn that a network-reachable audition is only as private as its link."""
    warning = (
        "Warning: anyone who has the access link (the one with ?k=) can play the "
        "episode audio, read and change speaker names and run splits; share it "
        "only with people you trust and stop the server when you are done."
    )
    if host != HOST:
        warning += (
            " On the local network the connection is plain HTTP, so the link and "
            "the audio are not encrypted."
        )
    return warning


def serve(
    *,
    page: str,
    media_path: Path,
    mapping_path: Path,
    sibling_path: Path,
    speaker_ids: Sequence[str],
    pristine_mapping_generation: fsio.FileGeneration | None = None,
    host: str = HOST,
    ngrok: bool = False,
    port: int = 0,
    open_browser: bool = True,
    report: Callable[[str], None] = print,
    render: Callable[[], str] | None = None,
) -> str:
    """Serve an audition until interrupted and return its access link.

    The link (the listening URL plus ``?k=`` and the access key) is the only
    way in; it is printed first and is what the browser is opened with.
    ``render`` runs after every successful Save (see :class:`SpeakerHTTPServer`).
    """
    server = make_server(
        page=page,
        media_path=media_path,
        mapping_path=mapping_path,
        sibling_path=sibling_path,
        speaker_ids=speaker_ids,
        pristine_mapping_generation=pristine_mapping_generation,
        host=host,
        ngrok=ngrok,
        port=port,
        report=report,
        render=render,
    )
    url = server.access_url
    report(url)
    if ngrok:
        report(
            "ngrok discovery enabled via the local agent on 127.0.0.1:4040; open "
            f"the tunnel's URL with /?{_ACCESS_KEY_PARAM}={server.access_key} "
            "appended."
        )
    if host == "0.0.0.0":
        report(
            "For access from another device, replace 0.0.0.0 with this machine's IP address."
        )
    if host != HOST or ngrok:
        report(_exposure_warning(host=host))
    if open_browser:
        try:
            webbrowser.open(url.replace("//0.0.0.0:", f"//{HOST}:"))
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
