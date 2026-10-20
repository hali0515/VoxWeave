"""Strict JSON reading and speaker-mapping validation for the audition server."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MAX_NAME_CHARS = 500


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


def _mapping_speakers(value: object) -> dict[str, Any]:
    """The version-1 envelope every mapping reader of this server shares.

    Returns the ``speakers`` object unchecked; extra top-level keys are allowed
    (only the POST payload is held to exactly ``{version, speakers}``, see
    :func:`_validated_mapping`).
    """
    if not isinstance(value, dict):
        raise ValueError("mapping must be an object")
    if type(value.get("version")) is not int or value.get("version") != 1:
        raise ValueError("mapping version must be 1")
    speakers = value.get("speakers")
    if not isinstance(speakers, dict):
        raise ValueError("speakers must be an object")
    return speakers


def _check_names(speakers: Mapping[str, object]) -> None:
    if any(
        not isinstance(name, str) or len(name) > MAX_NAME_CHARS
        for name in speakers.values()
    ):
        raise ValueError("mapping contains invalid speaker entries")


def _validated_speakers(
    value: object,
    speaker_ids: Sequence[str],
    *,
    stale: list[str] | None = None,
) -> dict[str, str]:
    """The mapping's names, all of them for ids in ``speaker_ids``.

    Used for the ``/save`` payload and the ``/serve-info`` read. When ``stale``
    is given, entries for ids outside ``speaker_ids`` (left behind by a
    re-diarization) are dropped and collected there instead of rejected, as
    :func:`voxweave.speakers.load_speaker_mapping_bytes` does.
    """
    speakers = _mapping_speakers(value)
    known = set(speaker_ids)
    if stale is not None:
        stale.extend(key for key in speakers if key not in known)
        speakers = {key: name for key, name in speakers.items() if key in known}
    if any(key not in known for key in speakers):
        raise ValueError("mapping contains invalid speaker entries")
    _check_names(speakers)
    return speakers


def _validated_mapping(
    value: object,
    speaker_ids: Sequence[str],
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"version", "speakers"}:
        raise ValueError("mapping must contain only version and speakers")
    return _validated_speakers(value, speaker_ids)


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
) -> tuple[dict[str, str], list[str], tuple[int, int, int, int]]:
    """Read the on-disk mapping, dropping (and returning) ids no longer known."""
    before = _file_generation(path)
    value = _strict_json_loads(Path(path).read_bytes())
    after = _file_generation(path)
    if before != after:
        raise ValueError("mapping changed while reading")
    stale: list[str] = []
    speakers = _validated_speakers(value, speaker_ids, stale=stale)
    return speakers, stale, after
