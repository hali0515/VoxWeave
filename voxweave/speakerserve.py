"""Local-only HTTP serving for the speaker audition page."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
import secrets
import tempfile
import threading
import webbrowser
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from voxweave import artifacts, fsio, turnembed
from voxweave.voicebase import (
    MAX_PROVENANCE_STRING_BYTES,
    VOICEPRINTS_MAX_BYTES,
    Phase2DataError,
    canonical_turns_digest,
    encode_json_bytes,
    media_fingerprint,
    require_mapping,
    require_string,
    strict_json_object_loads,
    strict_turn_projection,
    validate_voiceprint_conjunction,
)
from voxweave.voiceepisode import episode_lock
from voxweave.voicematch import (
    build_compatibility_fingerprint,
    delete_suggest,
    require_known_compatibility,
)
from voxweave.vocalscache import SeparatorIdentity, validate_separator_identity

HOST = "127.0.0.1"
MAX_BODY_BYTES = 1_000_000
MAX_NAME_CHARS = 500
MAX_UNDO_BYTES = 64 * 1024 * 1024
_POST_ROUTES = frozenset({"/save", "/split", "/split-confirm", "/split-undo"})
_INVALID_BODY = object()


class SplitConflict(RuntimeError):
    """The staged split no longer matches authoritative episode inputs."""


@dataclass(frozen=True, slots=True)
class _SplitProposal:
    speaker_id: str
    sibling_bytes: bytes
    voiceprints_path: Path
    voiceprints_bytes: bytes
    media_fingerprint: str
    assignment: tuple[tuple[int, str], ...]
    embeddings: tuple[tuple[int, tuple[float, ...]], ...]


@dataclass(frozen=True, slots=True)
class _StagedSplit:
    sibling_bytes: bytes
    voiceprints_path: Path
    voiceprints_bytes: bytes
    media_fingerprint: str
    turns: tuple[tuple[float, float, str], ...]
    selected_indices: tuple[int, ...]
    embedding_dim: int
    embedding_model: str
    embedding_checkpoint: str
    pyannote_version: str
    audio_separated: bool
    audio_normalized: bool
    audio_separator: SeparatorIdentity | None


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
        self.action_lock = threading.Lock()
        self.split_proposal: _SplitProposal | None = None
        self.session_terminal = False
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
                with self.server.action_lock:
                    from voxweave import pipeline

                    mapping_path = pipeline.speakers_mapping_path(
                        self.server.media_path
                    )
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
        if not self._host_allowed():
            self._reply(HTTPStatus.FORBIDDEN)
            return _INVALID_BODY
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] != self.server.origin):
            self._reply(HTTPStatus.FORBIDDEN)
            return _INVALID_BODY
        tokens = self.headers.get_all("X-VoxWeave-Token", [])
        token = tokens[0] if len(tokens) == 1 else ""
        if not secrets.compare_digest(token, self.server.token):
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
                {"error": "restart voxweave speakers before saving again"},
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

    def _handle_split(self, payload: object) -> None:
        if self.server.session_terminal:
            self._json_reply(
                HTTPStatus.CONFLICT,
                {"error": "restart voxweave speakers before another split"},
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


def _absolute(path: Path) -> Path:
    return Path(os.path.realpath(os.path.abspath(os.fspath(Path(path)))))


def _validated_speaker_request(value: object, speaker_ids: Sequence[str]) -> str:
    if not isinstance(value, dict) or set(value) != {"speaker_id"}:
        raise ValueError("split request must contain only speaker_id")
    speaker_id = value["speaker_id"]
    if not isinstance(speaker_id, str) or speaker_id not in set(speaker_ids):
        raise ValueError("split request contains an unknown speaker id")
    return speaker_id


def _prepare_split_wav(media_path: Path, staged: _StagedSplit) -> Path:
    """Reproduce the audio transform recorded by the bound voiceprints."""
    from voxweave import pipeline
    from voxweave.chunking import decode_to_wav
    from voxweave.vocalscache import (
        cache_lock,
        load_cache_companion,
        validate_cache_pair,
    )

    audio_filter = pipeline.ASR_LOUDNORM if staged.audio_normalized else None
    if not staged.audio_separated:
        return decode_to_wav(
            media_path,
            sample_rate=turnembed.SAMPLE_RATE,
            mono=True,
            audio_filter=audio_filter,
        )

    if staged.audio_separator is None:
        raise SplitConflict("separated voiceprints lack a resolved separator identity")
    cache_path = pipeline.cache_vocals_path(media_path)
    with cache_lock(cache_path) as handle:
        try:
            companion, _validated = load_cache_companion(handle.companion_path)
            validate_cache_pair(
                companion,
                handle.cache_path,
                media_fingerprint=staged.media_fingerprint,
                separator=staged.audio_separator,
            )
        except (OSError, Phase2DataError) as exc:
            raise SplitConflict(
                "the separated vocals cache does not match this voiceprint capture"
            ) from exc
        return decode_to_wav(
            handle.cache_path,
            sample_rate=turnembed.SAMPLE_RATE,
            mono=True,
            audio_filter=audio_filter,
        )


def _staged_provenance(
    sidecar: Mapping[str, object],
) -> tuple[str, str, str, bool, bool, SeparatorIdentity | None]:
    provenance = require_mapping(sidecar.get("provenance"), "provenance")
    require_known_compatibility(build_compatibility_fingerprint(provenance))
    embedding_model = require_string(
        provenance.get("embedding_model"),
        "provenance.embedding_model",
        max_bytes=MAX_PROVENANCE_STRING_BYTES,
    )
    embedding_checkpoint = require_string(
        provenance.get("embedding_checkpoint"),
        "provenance.embedding_checkpoint",
        max_bytes=MAX_PROVENANCE_STRING_BYTES,
    )
    pyannote_version = require_string(
        provenance.get("pyannote_version"),
        "provenance.pyannote_version",
        max_bytes=MAX_PROVENANCE_STRING_BYTES,
    )
    audio = require_mapping(provenance.get("audio"), "provenance.audio")
    separated = audio.get("separated")
    normalized = audio.get("normalized")
    sample_rate = audio.get("sample_rate")
    if type(separated) is not bool or type(normalized) is not bool:
        raise Phase2DataError("provenance.audio separated/normalized must be booleans")
    if type(sample_rate) is not int or sample_rate != turnembed.SAMPLE_RATE:
        raise Phase2DataError(
            f"provenance.audio.sample_rate must be {turnembed.SAMPLE_RATE}"
        )
    separator = (
        validate_separator_identity(audio.get("separator")) if separated else None
    )
    return (
        embedding_model,
        embedding_checkpoint,
        pyannote_version,
        separated,
        normalized,
        separator,
    )


def _stage_split_inputs(
    server: SpeakerHTTPServer,
    speaker_id: str,
) -> _StagedSplit:
    from voxweave import pipeline

    with episode_lock(server.media_path):
        sibling_bytes = server.sibling_path.read_bytes()
        sibling = strict_json_object_loads(
            sibling_bytes,
            max_bytes=max(1, len(sibling_bytes)),
            source=server.sibling_path.name,
        )
        turns = strict_turn_projection(sibling.get("speaker_turns"))
        selected = tuple(
            index
            for index, (_start, _end, label) in enumerate(turns)
            if label == speaker_id
        )
        if len(selected) < 2:
            raise turnembed.UnsplittableSpeakerError(
                "a speaker needs at least two turns to split"
            )
        sidecar_path = pipeline.voiceprints_path(server.media_path)
        try:
            sidecar_bytes = sidecar_path.read_bytes()
        except OSError as exc:
            raise SplitConflict(
                "speaker splitting requires bound voiceprints; rerun with "
                "--diarize --voiceprints"
            ) from exc
        sidecar = strict_json_object_loads(
            sidecar_bytes,
            max_bytes=VOICEPRINTS_MAX_BYTES,
            source=sidecar_path.name,
        )
        fingerprint = media_fingerprint(server.media_path)
        validated = validate_voiceprint_conjunction(
            sidecar,
            sibling,
            fingerprint,
        )
        (
            embedding_model,
            embedding_checkpoint,
            pyannote_version,
            audio_separated,
            audio_normalized,
            audio_separator,
        ) = _staged_provenance(sidecar)
        return _StagedSplit(
            sibling_bytes=sibling_bytes,
            voiceprints_path=sidecar_path,
            voiceprints_bytes=sidecar_bytes,
            media_fingerprint=fingerprint,
            turns=turns,
            selected_indices=selected,
            embedding_dim=validated.embedding_dim,
            embedding_model=embedding_model,
            embedding_checkpoint=embedding_checkpoint,
            pyannote_version=pyannote_version,
            audio_separated=audio_separated,
            audio_normalized=audio_normalized,
            audio_separator=audio_separator,
        )


def _proposal_groups(
    wav_path: Path,
    turns: Sequence[tuple[float, float, str]],
    assignment: Mapping[int, str],
) -> list[dict[str, object]]:
    from voxweave import speakers

    groups: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="voxweave_split_clips_") as temp_dir:
        root = Path(temp_dir)
        for label in ("A", "B"):
            indices = sorted(
                index for index, group in assignment.items() if group == label
            )
            spans = [(turns[index][0], turns[index][1]) for index in indices]
            samples: list[dict[str, object]] = []
            for clip_index, (start, end) in enumerate(
                speakers._pick_spread(spans, speakers.MAX_SNIPPETS_PER_SPEAKER)
            ):
                clip_path = root / f"{label}-{clip_index}.mp3"
                speakers.extract_clip(wav_path, start, end, clip_path)
                encoded = base64.b64encode(clip_path.read_bytes()).decode("ascii")
                samples.append(
                    {
                        "start": start,
                        "end": end,
                        "src": f"data:audio/mpeg;base64,{encoded}",
                    }
                )
            groups.append(
                {
                    "label": label,
                    "turn_indices": indices,
                    "turn_count": len(indices),
                    "total_duration": math.fsum(
                        turns[index][1] - turns[index][0] for index in indices
                    ),
                    "samples": samples,
                }
            )
    return groups


def _recheck_split_inputs(server: SpeakerHTTPServer, staged: _StagedSplit) -> None:
    from voxweave import pipeline

    with episode_lock(server.media_path):
        if server.sibling_path.read_bytes() != staged.sibling_bytes:
            raise SplitConflict("speaker turns changed during split preview; retry")
        current_sidecar_path = pipeline.voiceprints_path(server.media_path)
        if current_sidecar_path != staged.voiceprints_path:
            raise SplitConflict(
                "voiceprints storage changed during split preview; retry"
            )
        if current_sidecar_path.read_bytes() != staged.voiceprints_bytes:
            raise SplitConflict("voiceprints changed during split preview; retry")
        if media_fingerprint(server.media_path) != staged.media_fingerprint:
            raise SplitConflict("media changed during split preview; retry")


def _require_embedding_identity(
    staged: _StagedSplit,
    embeddings: object,
) -> None:
    if not isinstance(embeddings, turnembed.AttestedTurnEmbeddings):
        raise turnembed.TurnEmbeddingError(
            "turn embedding provider did not attest its loaded identity"
        )
    identity = embeddings.identity
    if (
        identity.model != staged.embedding_model
        or identity.checkpoint_sha256 != staged.embedding_checkpoint
        or identity.pyannote_version != staged.pyannote_version
    ):
        raise SplitConflict(
            "the turn embedding provider does not match the voiceprint capture"
        )


def _build_split_proposal(
    server: SpeakerHTTPServer,
    speaker_id: str,
) -> tuple[_SplitProposal, dict[str, object]]:
    staged = _stage_split_inputs(server, speaker_id)
    selected_turns = [staged.turns[index] for index in staged.selected_indices]
    embedding_request = turnembed.AttestedTurnRequest(
        selected_turns,
        identity=turnembed.EmbeddingIdentity(
            model=staged.embedding_model,
            checkpoint_sha256=staged.embedding_checkpoint,
            pyannote_version=staged.pyannote_version,
        ),
    )
    wav_path = _prepare_split_wav(server.media_path, staged)
    try:
        provider_embeddings = turnembed.turn_embeddings(wav_path, embedding_request)
        _require_embedding_identity(staged, provider_embeddings)
        expected = set(range(len(selected_turns)))
        if set(provider_embeddings) != expected:
            raise turnembed.TurnEmbeddingError(
                "turn embedding provider did not return every requested turn"
            )
        local_embeddings = {
            index: turnembed.normalized_centroid([provider_embeddings[index]])
            for index in sorted(expected)
        }
        if any(
            len(local_embeddings[index]) != staged.embedding_dim for index in expected
        ):
            raise turnembed.TurnEmbeddingError(
                "turn embeddings do not match the bound voiceprint dimension"
            )
        local_assignment = turnembed.bisect_embeddings(local_embeddings)
        assignment = {
            staged.selected_indices[local_index]: group
            for local_index, group in local_assignment.items()
        }
        embeddings = {
            staged.selected_indices[local_index]: tuple(
                float(value) for value in local_embeddings[local_index]
            )
            for local_index in sorted(local_embeddings)
        }
        groups = _proposal_groups(wav_path, staged.turns, assignment)
    finally:
        wav_path.unlink(missing_ok=True)
    _recheck_split_inputs(server, staged)
    ordered_assignment = tuple(sorted(assignment.items()))
    proposal = _SplitProposal(
        speaker_id=speaker_id,
        sibling_bytes=staged.sibling_bytes,
        voiceprints_path=staged.voiceprints_path,
        voiceprints_bytes=staged.voiceprints_bytes,
        media_fingerprint=staged.media_fingerprint,
        assignment=ordered_assignment,
        embeddings=tuple(sorted(embeddings.items())),
    )
    response: dict[str, object] = {
        "speaker_id": speaker_id,
        "assignment": {str(index): group for index, group in ordered_assignment},
        "groups": groups,
    }
    return proposal, response


def _validate_confirmation(value: object, proposal: _SplitProposal) -> None:
    if not isinstance(value, dict) or set(value) != {"speaker_id", "assignment"}:
        raise ValueError("confirmation must contain speaker_id and assignment")
    if value["speaker_id"] != proposal.speaker_id:
        raise ValueError("confirmation speaker id does not match the proposal")
    raw_assignment = value["assignment"]
    if not isinstance(raw_assignment, dict):
        raise ValueError("confirmation assignment must be an object")
    expected = {str(index): group for index, group in proposal.assignment}
    if raw_assignment != expected:
        raise ValueError("confirmation assignment does not match the proposal")


def _mapping_document(
    raw: bytes, *, source: str
) -> tuple[dict[str, object], dict[str, str]]:
    value = _strict_json_loads(raw)
    if not isinstance(value, dict) or set(value) != {"version", "speakers"}:
        raise SplitConflict(f"{source} has an invalid speaker mapping schema")
    if type(value["version"]) is not int or value["version"] != 1:
        raise SplitConflict(f"{source} has an unsupported speaker mapping version")
    speakers = value["speakers"]
    if not isinstance(speakers, dict) or any(
        not isinstance(key, str)
        or not isinstance(name, str)
        or len(name) > MAX_NAME_CHARS
        for key, name in speakers.items()
    ):
        raise SplitConflict(f"{source} contains invalid speaker mapping entries")
    return value, speakers


def _next_speaker_id(used: set[str]) -> str:
    index = 0
    while True:
        candidate = f"SPEAKER_{index:02d}"
        if candidate not in used:
            return candidate
        index += 1


def _json_bytes(value: object, *, newline: bool) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _undo_record(path: Path, before: bytes, after: bytes) -> dict[str, object]:
    return {
        "path": os.fspath(_absolute(path)),
        "before": base64.b64encode(before).decode("ascii"),
        "after_size": len(after),
        "after_sha256": _sha256(after),
    }


def _undo_bytes(
    *,
    fingerprint: str,
    sibling_path: Path,
    sibling_before: bytes,
    sibling_after: bytes,
    voiceprints_path: Path,
    voiceprints_before: bytes,
    voiceprints_after: bytes,
    mapping_path: Path,
    mapping_before: bytes,
    mapping_after: bytes,
) -> bytes:
    value = {
        "version": 1,
        "media_fingerprint": fingerprint,
        "files": {
            "sibling": _undo_record(sibling_path, sibling_before, sibling_after),
            "voiceprints": _undo_record(
                voiceprints_path,
                voiceprints_before,
                voiceprints_after,
            ),
            "mapping": _undo_record(mapping_path, mapping_before, mapping_after),
        },
    }
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if len(raw) > MAX_UNDO_BYTES:
        raise SplitConflict("speaker split undo snapshot is too large")
    return raw


def _write_bytes(path: Path, raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SplitConflict(f"{path.name} is not valid UTF-8") from exc
    fsio.atomic_write_text(path, text)


def _restore_files(files: Iterable[tuple[Path, bytes]]) -> None:
    for path, raw in files:
        _write_bytes(path, raw)


def _confirm_split(server: SpeakerHTTPServer, proposal: _SplitProposal) -> str:
    from voxweave import pipeline

    with episode_lock(server.media_path):
        sibling_bytes = server.sibling_path.read_bytes()
        if sibling_bytes != proposal.sibling_bytes:
            raise SplitConflict("speaker turns changed after the split preview; retry")
        sidecar_path = pipeline.voiceprints_path(server.media_path)
        if sidecar_path != proposal.voiceprints_path:
            raise SplitConflict("voiceprints storage changed after the split preview")
        sidecar_bytes = sidecar_path.read_bytes()
        if sidecar_bytes != proposal.voiceprints_bytes:
            raise SplitConflict("voiceprints changed after the split preview; retry")
        fingerprint = media_fingerprint(server.media_path)
        if fingerprint != proposal.media_fingerprint:
            raise SplitConflict("media changed after the split preview; retry")

        sibling = strict_json_object_loads(
            sibling_bytes,
            max_bytes=max(1, len(sibling_bytes)),
            source=server.sibling_path.name,
        )
        turns = strict_turn_projection(sibling.get("speaker_turns"))
        sidecar = strict_json_object_loads(
            sidecar_bytes,
            max_bytes=VOICEPRINTS_MAX_BYTES,
            source=sidecar_path.name,
        )
        validated = validate_voiceprint_conjunction(sidecar, sibling, fingerprint)

        mapping_path = pipeline.speakers_mapping_path(server.media_path)
        mapping_bytes = mapping_path.read_bytes()
        mapping, mapping_entries = _mapping_document(
            mapping_bytes,
            source=mapping_path.name,
        )
        sidecar_speakers = sidecar.get("speakers")
        if not isinstance(sidecar_speakers, dict):
            raise Phase2DataError("voiceprints speakers must be an object")
        used_ids = {
            *(label for _start, _end, label in turns),
            *sidecar_speakers.keys(),
            *mapping_entries.keys(),
        }
        new_id = _next_speaker_id(used_ids)

        assignments = dict(proposal.assignment)
        vectors = dict(proposal.embeddings)
        if set(assignments) != set(vectors):
            raise SplitConflict("split proposal embeddings are incomplete")
        for index in assignments:
            if index >= len(turns) or turns[index][2] != proposal.speaker_id:
                raise SplitConflict("split proposal no longer addresses the same turns")
        group_a = [
            vectors[index] for index, group in assignments.items() if group == "A"
        ]
        group_b = [
            vectors[index] for index, group in assignments.items() if group == "B"
        ]
        if any(
            len(vector) != validated.embedding_dim for vector in (*group_a, *group_b)
        ):
            raise SplitConflict("split proposal embedding dimension changed")
        centroid_a = turnembed.normalized_centroid(group_a)
        centroid_b = turnembed.normalized_centroid(group_b)

        updated_sibling = copy.deepcopy(sibling)
        raw_turns = updated_sibling.get("speaker_turns")
        if not isinstance(raw_turns, list):
            raise Phase2DataError("speaker_turns must be an array")
        for index, group in assignments.items():
            raw_turn = raw_turns[index]
            if not isinstance(raw_turn, list) or len(raw_turn) != 3:
                raise Phase2DataError(f"speaker_turns[{index}] must be an array")
            if group == "B":
                raw_turn[2] = new_id
        updated_turns = strict_turn_projection(updated_sibling.get("speaker_turns"))

        updated_sidecar = copy.deepcopy(sidecar)
        updated_speakers = updated_sidecar.get("speakers")
        updated_binding = updated_sidecar.get("binding")
        if not isinstance(updated_speakers, dict) or not isinstance(
            updated_binding, dict
        ):
            raise Phase2DataError("voiceprints speakers and binding must be objects")
        updated_speakers[proposal.speaker_id] = centroid_a
        updated_speakers[new_id] = centroid_b
        updated_binding["turns_digest"] = canonical_turns_digest(updated_turns)
        validate_voiceprint_conjunction(
            updated_sidecar,
            updated_sibling,
            fingerprint,
        )

        updated_mapping_entries = dict(mapping_entries)
        updated_mapping_entries[new_id] = ""
        mapping["speakers"] = updated_mapping_entries
        sibling_after = _json_bytes(updated_sibling, newline=False)
        sidecar_after = encode_json_bytes(
            updated_sidecar,
            max_bytes=VOICEPRINTS_MAX_BYTES,
        )
        mapping_after = _json_bytes(mapping, newline=True)

        undo_path = artifacts.speaker_split_undo_path(server.media_path)
        previous_undo = undo_path.read_bytes() if undo_path.exists() else None
        snapshot = _undo_bytes(
            fingerprint=fingerprint,
            sibling_path=server.sibling_path,
            sibling_before=sibling_bytes,
            sibling_after=sibling_after,
            voiceprints_path=sidecar_path,
            voiceprints_before=sidecar_bytes,
            voiceprints_after=sidecar_after,
            mapping_path=mapping_path,
            mapping_before=mapping_bytes,
            mapping_after=mapping_after,
        )
        suggest_path = pipeline.speakers_suggest_path(server.media_path)
        _write_bytes(undo_path, snapshot)
        written: list[tuple[Path, bytes]] = []
        try:
            for path, before, after in (
                (sidecar_path, sidecar_bytes, sidecar_after),
                (server.sibling_path, sibling_bytes, sibling_after),
                (mapping_path, mapping_bytes, mapping_after),
            ):
                written.append((path, before))
                _write_bytes(path, after)
            delete_suggest(suggest_path)
        except BaseException:
            _restore_files(reversed(written))
            if previous_undo is None:
                undo_path.unlink(missing_ok=True)
            else:
                _write_bytes(undo_path, previous_undo)
            raise

    server.speaker_ids = (*server.speaker_ids, new_id)
    server.mapping_path = mapping_path
    server.pristine_mapping_path = None
    server.pristine_mapping_generation = None
    server.split_proposal = None
    server.session_terminal = True
    server.report(
        f"Split {proposal.speaker_id} into {proposal.speaker_id} and {new_id}"
    )
    server.report("Restart `voxweave speakers` to re-audition")
    return new_id


def _decoded_undo_record(
    value: object,
    *,
    field: str,
) -> tuple[Path, bytes, int, str]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "before",
        "after_size",
        "after_sha256",
    }:
        raise ValueError(f"undo {field} record has an invalid schema")
    raw_path = value["path"]
    encoded = value["before"]
    after_size = value["after_size"]
    after_hash = value["after_sha256"]
    if not isinstance(raw_path, str) or not isinstance(encoded, str):
        raise ValueError(f"undo {field} path and bytes must be strings")
    if type(after_size) is not int or after_size < 0:
        raise ValueError(f"undo {field} size must be a non-negative integer")
    if (
        not isinstance(after_hash, str)
        or len(after_hash) != 64
        or any(character not in "0123456789abcdef" for character in after_hash)
    ):
        raise ValueError(f"undo {field} digest is invalid")
    try:
        before = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"undo {field} bytes are invalid base64") from exc
    path = Path(raw_path)
    if not path.is_absolute() or path != _absolute(path):
        raise ValueError(f"undo {field} path is not normalized")
    return path, before, after_size, after_hash


def _load_undo(path: Path) -> tuple[str, dict[str, tuple[Path, bytes, int, str]]]:
    if not path.is_file():
        raise SplitConflict("there is no speaker split to undo")
    raw = path.read_bytes()
    value = strict_json_object_loads(
        raw,
        max_bytes=MAX_UNDO_BYTES,
        source=path.name,
    )
    if set(value) != {"version", "media_fingerprint", "files"}:
        raise ValueError("speaker split undo snapshot has an invalid schema")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("speaker split undo snapshot has an invalid version")
    fingerprint = value["media_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("speaker split undo media fingerprint is invalid")
    files = value["files"]
    if not isinstance(files, dict) or set(files) != {
        "sibling",
        "voiceprints",
        "mapping",
    }:
        raise ValueError("speaker split undo files have an invalid schema")
    return fingerprint, {
        field: _decoded_undo_record(files[field], field=field)
        for field in ("sibling", "voiceprints", "mapping")
    }


def _undo_split(server: SpeakerHTTPServer) -> None:
    from voxweave import pipeline

    undo_path = artifacts.speaker_split_undo_path(server.media_path)
    with episode_lock(server.media_path):
        fingerprint, records = _load_undo(undo_path)
        try:
            current_paths = {
                "sibling": _absolute(server.sibling_path),
                "voiceprints": _absolute(pipeline.voiceprints_path(server.media_path)),
                "mapping": _absolute(pipeline.speakers_mapping_path(server.media_path)),
            }
        except OSError as exc:
            raise SplitConflict(
                "artifact storage changed since the split; undo refused"
            ) from exc
        for field, expected_path in current_paths.items():
            recorded_path, _before, _size, _digest = records[field]
            if recorded_path != expected_path:
                raise SplitConflict(
                    f"{field} storage changed since the split; undo refused"
                )
        try:
            current_fingerprint = media_fingerprint(server.media_path)
        except OSError as exc:
            raise SplitConflict("media changed since the split; undo refused") from exc
        if current_fingerprint != fingerprint:
            raise SplitConflict("media changed since the split; undo refused")

        current: dict[str, bytes] = {}
        for field, path in current_paths.items():
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise SplitConflict(
                    f"{field} changed since the split; undo refused"
                ) from exc
            _recorded_path, before, expected_size, expected_hash = records[field]
            matches_after = len(raw) == expected_size and _sha256(raw) == expected_hash
            if raw != before and not matches_after:
                raise SplitConflict(f"{field} changed since the split; undo refused")
            current[field] = raw

        sibling_before = records["sibling"][1]
        sidecar_before = records["voiceprints"][1]
        mapping_before = records["mapping"][1]
        sibling = strict_json_object_loads(
            sibling_before,
            max_bytes=max(1, len(sibling_before)),
            source=current_paths["sibling"].name,
        )
        sidecar = strict_json_object_loads(
            sidecar_before,
            max_bytes=VOICEPRINTS_MAX_BYTES,
            source=current_paths["voiceprints"].name,
        )
        validate_voiceprint_conjunction(sidecar, sibling, fingerprint)
        _mapping_document(mapping_before, source=current_paths["mapping"].name)

        restored: list[tuple[Path, bytes]] = []
        try:
            for field, before in (
                ("voiceprints", sidecar_before),
                ("sibling", sibling_before),
                ("mapping", mapping_before),
            ):
                path = current_paths[field]
                restored.append((path, current[field]))
                _write_bytes(path, before)
            undo_path.unlink()
        except BaseException:
            _restore_files(reversed(restored))
            raise

    restored_turns = strict_turn_projection(sibling.get("speaker_turns"))
    server.speaker_ids = tuple(
        dict.fromkeys(label for _start, _end, label in restored_turns)
    )
    server.mapping_path = current_paths["mapping"]
    server.pristine_mapping_path = None
    server.pristine_mapping_generation = None
    server.split_proposal = None
    server.session_terminal = True
    server.report("Restored the previous speaker split generation")
    server.report("Restart `voxweave speakers` to re-audition")


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
