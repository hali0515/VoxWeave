"""Strict, seam-independent primitives for phase-2 speaker artifacts.

Phase-1 speaker mappings deliberately keep their tolerant reader.  This module
is for phase-2 machine artifacts only: it rejects ambiguous JSON, validates the
shared vector and binding laws, and serializes deterministically before any
file is touched.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from voxweave import fsio

VOICEPRINTS_MAX_BYTES = 2 * 1024 * 1024
VOICES_STORE_MAX_BYTES = 8 * 1024 * 1024
SUGGEST_MAX_BYTES = 1024 * 1024
CACHE_COMPANION_MAX_BYTES = 4 * 1024

MAX_SIDECAR_SPEAKERS = 64
MIN_EMBEDDING_DIM = 16
MAX_EMBEDDING_DIM = 768
MAX_PROVENANCE_STRING_BYTES = 512
MAX_SIDECAR_LABEL_BYTES = 64
MAX_MEDIA_STEM_BYTES = 255

CAPTURE_ID_RE = re.compile(r"^c[0-9a-f]{32}$")
IDENTITY_ID_RE = re.compile(r"^v[0-9a-f]{12}$")
EXEMPLAR_ID_RE = re.compile(r"^x[0-9a-f]{8}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_FINGERPRINT_WINDOW = 1024 * 1024


class Phase2DataError(ValueError):
    """A phase-2 artifact or explicit in-memory value violates the schema."""


class DuplicateKeyError(Phase2DataError):
    """A JSON object contains the same member name more than once."""


@dataclass(frozen=True)
class ValidatedVoiceprints:
    """The validated fields a phase-2 consumer may use."""

    capture_id: str
    media_fingerprint: str
    turns_digest: str
    embedding_dim: int
    speakers: dict[str, tuple[int | float, ...]]


def _invalid(message: str) -> NoReturn:
    raise Phase2DataError(message)


def utf8_size(value: str) -> int:
    """Return a string's encoded size, rejecting ill-formed surrogates."""
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise Phase2DataError("string is not valid UTF-8") from exc


def require_string(
    value: object,
    field: str,
    *,
    max_bytes: int,
    nonempty: bool = True,
) -> str:
    """Validate a UTF-8 string with a byte, rather than code-point, bound."""
    if not isinstance(value, str):
        _invalid(f"{field} must be a string")
    if nonempty and not value:
        _invalid(f"{field} must not be empty")
    if utf8_size(value) > max_bytes:
        _invalid(f"{field} exceeds {max_bytes} UTF-8 bytes")
    return value


def require_exact_int(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate an integer without accepting bool or integral floats."""
    if type(value) is not int:
        _invalid(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        _invalid(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        _invalid(f"{field} must be at most {maximum}")
    return value


def require_version_one(value: object) -> None:
    """Enforce the phase-2 exact-int version predicate."""
    if type(value) is not int or value != 1:
        _invalid("version must be the integer 1")


def require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _invalid(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        _invalid(f"{field} keys must be strings")
    return value


def require_sha256(value: object, field: str = "sha256") -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _invalid(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def require_capture_id(value: object, field: str = "capture_id") -> str:
    if not isinstance(value, str) or CAPTURE_ID_RE.fullmatch(value) is None:
        _invalid(f"{field} has an invalid capture id")
    return value


def require_identity_id(value: object, field: str = "identity_id") -> str:
    if not isinstance(value, str) or IDENTITY_ID_RE.fullmatch(value) is None:
        _invalid(f"{field} has an invalid identity id")
    return value


def require_exemplar_id(value: object, field: str = "exemplar_id") -> str:
    if not isinstance(value, str) or EXEMPLAR_ID_RE.fullmatch(value) is None:
        _invalid(f"{field} has an invalid exemplar id")
    return value


def mint_capture_id(*, current: str | None = None) -> str:
    """Mint a 128-bit capture id, retrying the one forbidden current value."""
    while True:
        candidate = f"c{secrets.token_hex(16)}"
        if candidate != current:
            return candidate


def utc_timestamp(now: datetime | None = None) -> str:
    """Render an aware time as the phase-2 UTC, whole-second timestamp."""
    instant = now if now is not None else datetime.now(timezone.utc)
    if instant.tzinfo is None or instant.utcoffset() is None:
        _invalid("timestamp input must be timezone-aware")
    return (
        instant.astimezone(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def require_utc_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or UTC_TIMESTAMP_RE.fullmatch(value) is None:
        _invalid(f"{field} must be a UTC ISO8601 timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Phase2DataError(f"{field} must be a valid UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        _invalid(f"{field} must be a canonical UTC timestamp")
    return value


def _reject_constant(token: str) -> NoReturn:
    raise Phase2DataError(f"non-finite JSON number {token} is forbidden")


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise Phase2DataError(f"non-finite JSON number {token} is forbidden")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(
    payload: bytes | str,
    *,
    max_bytes: int,
    source: str = "JSON",
) -> object:
    """Parse bounded UTF-8 JSON while rejecting duplicate keys and NaN/Inf."""
    if isinstance(payload, str):
        try:
            raw = payload.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise Phase2DataError(f"{source} is not valid UTF-8") from exc
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise TypeError("payload must be bytes or str")
    if len(raw) > max_bytes:
        _invalid(f"{source} exceeds the {max_bytes}-byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Phase2DataError(f"{source} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except Phase2DataError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise Phase2DataError(f"invalid {source}: {exc}") from exc


def strict_json_object_loads(
    payload: bytes | str,
    *,
    max_bytes: int,
    source: str = "JSON",
) -> dict[str, object]:
    value = strict_json_loads(payload, max_bytes=max_bytes, source=source)
    if not isinstance(value, dict):
        _invalid(f"{source} must contain a top-level object")
    return value


def load_json_object(
    path: Path,
    *,
    max_bytes: int,
) -> dict[str, object]:
    """Read a phase-2 object, checking its cap both before and after reading."""
    source = Path(path)
    try:
        if source.stat().st_size > max_bytes:
            _invalid(f"{source.name} exceeds the {max_bytes}-byte limit")
        payload = source.read_bytes()
    except Phase2DataError:
        raise
    except OSError:
        raise
    return strict_json_object_loads(
        payload,
        max_bytes=max_bytes,
        source=source.name,
    )


def canonical_path(path: Path) -> Path:
    """Resolve a target to its realpath so sibling lock/companion keys are stable.

    Every phase-2 store derives its ``.lock``/``.meta.json`` companions from this
    one normalization, so two aliases of the same file always take the same lock.
    """
    return Path(os.path.realpath(os.fspath(Path(path))))


def canonical_json_bytes(value: object) -> bytes:
    """Encode canonical UTF-8 JSON for digests and reproducible artifacts."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Phase2DataError(f"value cannot be encoded as strict JSON: {exc}") from exc


def encode_json_bytes(
    value: object,
    *,
    max_bytes: int,
    newline: bool = True,
) -> bytes:
    """Serialize and byte-preflight a phase-2 artifact entirely in memory."""
    payload = canonical_json_bytes(value) + (b"\n" if newline else b"")
    if len(payload) > max_bytes:
        _invalid(f"encoded JSON exceeds the {max_bytes}-byte limit")
    return payload


def write_json_object(
    path: Path,
    value: Mapping[str, object],
    *,
    max_bytes: int,
) -> None:
    """Atomically write preflighted strict JSON."""
    payload = encode_json_bytes(value, max_bytes=max_bytes)
    fsio.atomic_write_text(Path(path), payload.decode("utf-8"))


def canonical_json_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_dimension(value: object, field: str = "embedding_dim") -> int:
    return require_exact_int(
        value,
        field,
        minimum=MIN_EMBEDDING_DIM,
        maximum=MAX_EMBEDDING_DIM,
    )


def validate_vector(
    value: object,
    *,
    dim: int | None = None,
    field: str = "vector",
) -> tuple[int | float, ...]:
    """Validate the shared exact-length, finite, near-unit vector law."""
    if not isinstance(value, (list, tuple)):
        _invalid(f"{field} must be an array")
    actual_dim = len(value)
    if dim is None:
        require_dimension(actual_dim, f"{field} dimension")
    elif actual_dim != dim:
        _invalid(f"{field} must contain exactly {dim} elements")

    checked: list[int | float] = []
    squares: list[float] = []
    for index, element in enumerate(value):
        if type(element) not in (int, float):
            _invalid(f"{field}[{index}] must be a non-bool number")
        try:
            numeric = float(element)
        except OverflowError as exc:
            raise Phase2DataError(f"{field}[{index}] exceeds finite range") from exc
        if not math.isfinite(numeric):
            _invalid(f"{field}[{index}] must be finite")
        if abs(numeric) > 1.0 + 1e-6:
            _invalid(f"{field}[{index}] exceeds the unit-vector bound")
        checked.append(element)
        squares.append(numeric * numeric)
    norm = math.sqrt(math.fsum(squares))
    if not 0.999 <= norm <= 1.001:
        _invalid(f"{field} L2 norm {norm!r} is outside [0.999, 1.001]")
    return tuple(checked)


def strict_turn_projection(
    turns: object,
) -> tuple[tuple[float, float, str], ...]:
    """Validate and canonically project persisted ``speaker_turns``.

    Numeric values are accepted only as exact JSON numbers (never bool or
    strings).  The explicit float projection is the digest law shared by the
    producer and consumers.
    """
    if not isinstance(turns, (list, tuple)):
        _invalid("speaker_turns must be an array")
    projected: list[tuple[float, float, str]] = []
    for index, entry in enumerate(turns):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            _invalid(f"speaker_turns[{index}] must be [start, end, label]")
        start, end, raw_label = entry
        if type(start) not in (int, float) or type(end) not in (int, float):
            _invalid(f"speaker_turns[{index}] bounds must be non-bool numbers")
        try:
            start_f = float(start)
            end_f = float(end)
        except OverflowError as exc:
            raise Phase2DataError(
                f"speaker_turns[{index}] bounds exceed finite range"
            ) from exc
        if not math.isfinite(start_f) or not math.isfinite(end_f):
            _invalid(f"speaker_turns[{index}] bounds must be finite")
        if not 0 <= start_f < end_f:
            _invalid(f"speaker_turns[{index}] must satisfy 0 <= start < end")
        label = require_string(
            raw_label,
            f"speaker_turns[{index}] label",
            max_bytes=MAX_SIDECAR_LABEL_BYTES,
        )
        projected.append((start_f, end_f, label))
    if projected != sorted(projected, key=lambda turn: (turn[0], turn[1], turn[2])):
        _invalid("speaker_turns must be ordered by (start, end, label)")
    return tuple(projected)


def canonical_turns_bytes(turns: object) -> bytes:
    projected = strict_turn_projection(turns)
    arrays = [[start, end, label] for start, end, label in projected]
    return canonical_json_bytes(arrays)


def canonical_turns_digest(turns: object) -> str:
    return hashlib.sha256(canonical_turns_bytes(turns)).hexdigest()


def _read_exact_at(fd: int, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    consumed = 0
    while consumed < length:
        chunk = os.pread(fd, length - consumed, offset + consumed)
        if not chunk:
            raise Phase2DataError("media changed or ended while fingerprinting")
        chunks.append(chunk)
        consumed += len(chunk)
    return b"".join(chunks)


def media_fingerprint_from_fd(fd: int, *, size: int | None = None) -> str:
    """Hash the frozen sampled-media frame from an already-open descriptor."""
    actual_size = os.fstat(fd).st_size if size is None else size
    if type(actual_size) is not int or not 0 <= actual_size < 2**64:
        _invalid("media size cannot be represented as an unsigned 64-bit integer")
    head_size = min(_FINGERPRINT_WINDOW, actual_size)
    tail_size = min(_FINGERPRINT_WINDOW, max(0, actual_size - _FINGERPRINT_WINDOW))
    digest = hashlib.sha256(actual_size.to_bytes(8, "big"))
    if head_size:
        digest.update(_read_exact_at(fd, 0, head_size))
    if tail_size:
        digest.update(_read_exact_at(fd, actual_size - tail_size, tail_size))
    return digest.hexdigest()


def media_fingerprint(path: Path) -> str:
    """Compute the sampled media identity from an explicit pathname."""
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        return media_fingerprint_from_fd(descriptor)
    finally:
        os.close(descriptor)


def _validate_provenance_strings(value: object, field: str) -> None:
    if isinstance(value, str):
        require_string(
            value,
            field,
            max_bytes=MAX_PROVENANCE_STRING_BYTES,
            nonempty=False,
        )
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                _invalid(f"{field} keys must be strings")
            _validate_provenance_strings(nested, f"{field}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_provenance_strings(nested, f"{field}[{index}]")


def validate_provenance(value: object) -> tuple[Mapping[str, object], int]:
    provenance = require_mapping(value, "provenance")
    _validate_provenance_strings(provenance, "provenance")
    dim = require_dimension(provenance.get("embedding_dim"))
    return provenance, dim


def validate_voiceprints_mapping(value: object) -> ValidatedVoiceprints:
    """Validate a decoded v1 voiceprints sidecar, accepting future top keys."""
    root = require_mapping(value, "voiceprints")
    require_version_one(root.get("version"))
    capture_id = require_capture_id(root.get("capture_id"))
    _provenance, dim = validate_provenance(root.get("provenance"))

    binding = require_mapping(root.get("binding"), "binding")
    turns_digest = require_sha256(binding.get("turns_digest"), "binding.turns_digest")
    media_hash = require_sha256(
        binding.get("media_fingerprint"),
        "binding.media_fingerprint",
    )
    require_string(
        binding.get("media_stem"),
        "binding.media_stem",
        max_bytes=MAX_MEDIA_STEM_BYTES,
    )
    require_utc_timestamp(binding.get("created"), "binding.created")

    speakers = require_mapping(root.get("speakers"), "speakers")
    if not speakers:
        _invalid("speakers must contain at least one usable centroid")
    if len(speakers) > MAX_SIDECAR_SPEAKERS:
        _invalid(f"speakers may contain at most {MAX_SIDECAR_SPEAKERS} entries")
    checked: dict[str, tuple[int | float, ...]] = {}
    for label, vector in speakers.items():
        require_string(
            label,
            "speaker label",
            max_bytes=MAX_SIDECAR_LABEL_BYTES,
        )
        checked[label] = validate_vector(vector, dim=dim, field=f"speakers.{label}")
    return ValidatedVoiceprints(
        capture_id=capture_id,
        media_fingerprint=media_hash,
        turns_digest=turns_digest,
        embedding_dim=dim,
        speakers=checked,
    )


def load_voiceprints(path: Path) -> tuple[dict[str, object], ValidatedVoiceprints]:
    raw = load_json_object(path, max_bytes=VOICEPRINTS_MAX_BYTES)
    return raw, validate_voiceprints_mapping(raw)


def voiceprints_digest(value: Mapping[str, object]) -> str:
    validate_voiceprints_mapping(value)
    return canonical_json_digest(value)


def write_voiceprints(path: Path, value: Mapping[str, object]) -> None:
    validate_voiceprints_mapping(value)
    write_json_object(path, value, max_bytes=VOICEPRINTS_MAX_BYTES)


def validate_voiceprint_conjunction(
    sidecar: Mapping[str, object],
    sibling: Mapping[str, object],
    consumer_fingerprint: str,
) -> ValidatedVoiceprints:
    """Enforce capture, media, turns, and consumer-media binding together."""
    validated = validate_voiceprints_mapping(sidecar)
    sibling_capture = require_capture_id(
        sibling.get("voiceprint_capture"),
        "sibling.voiceprint_capture",
    )
    sibling_media = require_sha256(
        sibling.get("voiceprint_media"),
        "sibling.voiceprint_media",
    )
    consumer_media = require_sha256(consumer_fingerprint, "consumer fingerprint")
    if validated.capture_id != sibling_capture:
        _invalid("voiceprints capture does not match the sibling capture")
    if validated.media_fingerprint != sibling_media:
        _invalid("voiceprints media does not match the sibling media")
    turns = sibling.get("speaker_turns")
    digest = canonical_turns_digest(turns)
    if validated.turns_digest != digest:
        _invalid("voiceprints turns digest does not match the sibling turns")
    labels = {label for _start, _end, label in strict_turn_projection(turns)}
    if not set(validated.speakers).issubset(labels):
        _invalid("voiceprints speaker labels are not a subset of sibling turns")
    if consumer_media != sibling_media:
        _invalid("consumer media does not match the sibling media")
    return validated


def voiceprint_conjunction_valid(
    sidecar: Mapping[str, object],
    sibling: Mapping[str, object],
    consumer_fingerprint: str,
) -> bool:
    try:
        validate_voiceprint_conjunction(sidecar, sibling, consumer_fingerprint)
    except Phase2DataError:
        return False
    return True


def html_text(value: str) -> str:
    """Escape an untrusted value for HTML text content."""
    return html.escape(value, quote=True)


def html_attribute(value: str) -> str:
    """Escape an untrusted value for a quoted HTML attribute."""
    return html.escape(value, quote=True)


def script_json(value: object) -> str:
    """Encode a script value as JSON and neutralize every HTML close tag."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Phase2DataError(f"script value cannot be encoded as JSON: {exc}") from exc
    return encoded.replace("</", "<\\/")


__all__ = [
    "CACHE_COMPANION_MAX_BYTES",
    "CAPTURE_ID_RE",
    "DuplicateKeyError",
    "EXEMPLAR_ID_RE",
    "IDENTITY_ID_RE",
    "MAX_EMBEDDING_DIM",
    "MIN_EMBEDDING_DIM",
    "Phase2DataError",
    "SHA256_RE",
    "SUGGEST_MAX_BYTES",
    "VOICES_STORE_MAX_BYTES",
    "VOICEPRINTS_MAX_BYTES",
    "ValidatedVoiceprints",
    "canonical_json_bytes",
    "canonical_json_digest",
    "canonical_path",
    "canonical_turns_bytes",
    "canonical_turns_digest",
    "encode_json_bytes",
    "html_attribute",
    "html_text",
    "load_json_object",
    "load_voiceprints",
    "media_fingerprint",
    "media_fingerprint_from_fd",
    "mint_capture_id",
    "require_capture_id",
    "require_dimension",
    "require_exact_int",
    "require_exemplar_id",
    "require_identity_id",
    "require_mapping",
    "require_sha256",
    "require_string",
    "require_utc_timestamp",
    "require_version_one",
    "script_json",
    "strict_json_loads",
    "strict_json_object_loads",
    "strict_turn_projection",
    "utc_timestamp",
    "utf8_size",
    "validate_provenance",
    "validate_vector",
    "validate_voiceprint_conjunction",
    "validate_voiceprints_mapping",
    "voiceprint_conjunction_valid",
    "voiceprints_digest",
    "write_json_object",
    "write_voiceprints",
]
