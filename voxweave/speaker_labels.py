"""Speaker names in mapping files and in subtitle text.

Decodes the per-media speaker mapping, sanitizes display names, and reads or
writes the SRT ``-`` prefixes and the VTT ``<v>`` voice tags.
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voxweave.voicebase import sanitize_speaker_name

log = logging.getLogger("voxweave")


MAPPING_VERSION = 1

SpeakerLine = tuple[str | None, str]


_VOICE_WRAP_RE = re.compile(
    r"\A<v(?:[ \t]+([^>]*?))?>(.*)</v>\Z", re.IGNORECASE | re.DOTALL
)
_VOICE_SPAN_RE = re.compile(
    r"<v(?:[ \t]+([^>]*?))?>(.*?)</v>", re.IGNORECASE | re.DOTALL
)
_VOICE_TAG_RE = re.compile(r"(?:<v(?:[ \t]+[^>]*?)?>|</v>)", re.IGNORECASE)


def _mapping_entries_bytes(raw_bytes: bytes, *, source: str) -> dict[str, Any]:
    """Decode version-1 mapping entries from one exact byte observation."""
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid speaker mapping JSON in {source}: {exc}") from exc
    version = raw.get("version") if isinstance(raw, dict) else None
    if type(version) is not int or version != MAPPING_VERSION:
        raise RuntimeError(
            f"{source} must use speaker mapping version {MAPPING_VERSION}"
        )
    speakers = raw.get("speakers")
    if not isinstance(speakers, dict):
        raise RuntimeError(f"{source} must contain a speakers object")
    return speakers


def _mapping_entries(path: Path) -> dict[str, Any]:
    """Read and validate the entries object from a version-1 mapping."""
    path = Path(path)
    return _mapping_entries_bytes(path.read_bytes(), source=path.name)


def named_speaker_ids(path: Path) -> set[str]:
    """The speaker ids a version-1 mapping gives a non-empty name."""
    return {
        str(raw_id)
        for raw_id, raw_name in _mapping_entries(path).items()
        if isinstance(raw_name, str) and raw_name.strip()
    }


def load_speaker_display_names(path: Path) -> list[str]:
    """Return the non-empty names in a mapping, in stable mapping order."""
    names: list[str] = []
    for raw_name in _mapping_entries(path).values():
        if isinstance(raw_name, str) and raw_name.strip():
            names.append(raw_name)
    return list(dict.fromkeys(names))


def load_speaker_mapping_bytes(
    raw_bytes: bytes,
    known_ids: Sequence[str] | set[str],
    *,
    source: str,
) -> dict[str, str]:
    """Project names from one exact mapping-byte observation.

    Empty names and unknown speaker ids are ignored. Unknown ids are reported
    in one logger call so a stale phase-1 mapping is visible without flooding
    an episode replay. Non-empty names are preserved exactly as entered;
    surrounding whitespace only decides whether a value is effectively empty.
    """
    speakers = _mapping_entries_bytes(raw_bytes, source=source)

    known = set(known_ids)
    unknown: list[str] = []
    names: dict[str, str] = {}
    for raw_id, raw_name in speakers.items():
        speaker_id = str(raw_id)
        if speaker_id not in known:
            unknown.append(speaker_id)
            continue
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        names[speaker_id] = raw_name
    if unknown:
        log.warning(
            "%s: ignoring unknown speaker id(s): %s",
            source,
            ", ".join(sorted(unknown)),
        )
    return names


def sanitize_ass_speaker_name(name: str) -> str:
    """Normalize a name and escape ASS's unquotable comma field delimiter."""
    return sanitize_speaker_name(name).replace(",", "，")


def _normalized_name(value: object) -> str | None:
    """Return one safe non-empty name, or None for non-name values."""
    if not isinstance(value, str):
        return None
    name = sanitize_speaker_name(value)
    return name if name.strip() else None


def strip_srt_speaker_prefixes(
    text: str, known_names: Sequence[str]
) -> tuple[str, str | None, list[SpeakerLine] | None]:
    """Recover speaker metadata from SRT prefixes emitted by voxweave.

    A prefix is recognized only when its name appears in the sibling speaker
    mapping.  This avoids interpreting ordinary dialogue containing a colon as
    metadata.  Multi-line dash cues retain line-level attribution; a normal
    multi-line cue with only its first line prefixed retains cue-level metadata.
    """
    names = sorted(
        {rendered for name in known_names if (rendered := sanitize_speaker_name(name))},
        key=len,
        reverse=True,
    )
    if not names:
        return text, None, None

    def strip_line(line: str) -> tuple[str, str | None]:
        for name in names:
            prefix = f"{name}: "
            if line.startswith(prefix):
                return line[len(prefix) :], name
        return line, None

    parsed = [strip_line(line) for line in text.split("\n")]
    line_names = [name for _line, name in parsed]
    if not any(line_names):
        return text, None, None
    plain_lines = [line for line, _name in parsed]
    plain = "\n".join(plain_lines)
    if len(parsed) == 1:
        return plain, line_names[0], None

    matched = sum(name is not None for name in line_names)
    is_dash = all(re.match(r"^-\s*\S", line) for line in plain_lines)
    if matched > 1 or line_names[0] is None or is_dash:
        return plain, None, list(zip(line_names, plain_lines))
    return plain, line_names[0], None


def _voice_name(name: str) -> str:
    """Escape a display name for a WebVTT voice annotation."""
    return html.escape(sanitize_speaker_name(name), quote=False)


def speaker_layout(
    text: str,
    *,
    speaker: str | None = None,
    speakers: Sequence[SpeakerLine] | None = None,
) -> tuple[str | None, list[str | None] | None]:
    """Resolve safe cue/line names by exact source-line content.

    Per-line names never follow a positional index across wrapping or edits.
    Every current line independently inherits a name only when its text matches
    an unambiguous stored source line. An explicit cue-level name owns the whole
    cue and therefore survives line-count changes.
    """
    cue_name = _normalized_name(speaker)
    ownership: dict[str, set[str | None]] = {}
    for name, source_line in speakers or ():
        if not isinstance(source_line, str):
            continue
        ownership.setdefault(source_line, set()).add(_normalized_name(name))
    if ownership:
        line_names: list[str | None] = []
        for line in text.split("\n"):
            candidates = ownership.get(line, set())
            line_names.append(next(iter(candidates)) if len(candidates) == 1 else None)
        if any(line_names):
            return None, line_names
    return (cue_name or None), None


def voice_tag_text(
    text: str,
    *,
    speaker: str | None = None,
    speakers: Sequence[SpeakerLine] | None = None,
) -> str:
    """Apply one cue-level or several line-level WebVTT voice tags."""
    cue_name, line_names = speaker_layout(text, speaker=speaker, speakers=speakers)
    lines = text.split("\n")
    if line_names is not None:
        return "\n".join(
            f"<v {_voice_name(name)}>{line}</v>" if name else line
            for line, name in zip(lines, line_names)
        )
    if cue_name:
        return f"<v {_voice_name(cue_name)}>{text}</v>"
    return text


def voice_text_for_block(text: str, block: Mapping[str, Any]) -> str:
    """Restore parsed speaker-name metadata around rendered cue text."""
    line_names = block.get("speakers")
    if isinstance(line_names, list):
        return voice_tag_text(text, speakers=line_names)
    name = block.get("speaker")
    return voice_tag_text(text, speaker=name if isinstance(name, str) else None)


def voice_text_for_ids(
    text: str,
    speaker_ids: Sequence[str] | None,
    names: Mapping[str, str] | None,
) -> str:
    """Render transient diarizer ids through a phase-1 name mapping."""
    if not speaker_ids or not names:
        return text
    resolved = [names.get(speaker_id) for speaker_id in speaker_ids]
    if len(resolved) == 1:
        return voice_tag_text(text, speaker=resolved[0])
    return voice_tag_text(text, speakers=list(zip(resolved, text.split("\n"))))


def strip_voice_tags(text: str) -> tuple[str, str | None, list[SpeakerLine] | None]:
    """Strip full-cue or per-line VTT voice tags into display metadata.

    Returns ``(plain_text, speaker, speakers)``. ``speaker`` is used for one
    wrapper around the whole cue; ``speakers`` stores ``(name, line_text)``
    pairs so later renders match attribution by content rather than position.
    """

    def voice_name(raw: str | None) -> str | None:
        return _normalized_name(html.unescape(raw or ""))

    def unwrap(value: str) -> tuple[str, str | None] | None:
        match = _VOICE_WRAP_RE.fullmatch(value)
        if match is None:
            return None
        return match.group(2), voice_name(match.group(1))

    whole = unwrap(text)
    if whole is not None and _VOICE_TAG_RE.search(whole[0]) is None:
        plain, name = whole
        return plain, name, None

    plain_lines: list[str] = []
    names: list[str | None] = []
    tagged = False
    for line in text.split("\n"):
        spans = list(_VOICE_SPAN_RE.finditer(line))
        if not spans:
            plain_lines.append(line)
            names.append(None)
            continue
        plain = _VOICE_SPAN_RE.sub(lambda match: match.group(2), line)
        if _VOICE_TAG_RE.search(plain) is not None:
            plain_lines.append(line)
            names.append(None)
            continue
        name = (
            voice_name(spans[0].group(1))
            if len(spans) == 1 and spans[0].span() == (0, len(line))
            else None
        )
        plain_lines.append(plain)
        names.append(name)
        tagged = True
    if tagged:
        clean = "\n".join(plain_lines)
        pairs = list(zip(names, plain_lines))
        return clean, None, pairs if any(name for name, _line in pairs) else None
    return text, None, None


def speaker_metadata(
    block: Mapping[str, Any],
) -> tuple[str | None, list[SpeakerLine] | None]:
    """Return normalized cue-level and content-bound line metadata."""
    speaker = block.get("speaker")
    raw_lines = block.get("speakers")
    lines: list[SpeakerLine] = []
    if isinstance(raw_lines, list):
        for entry in raw_lines:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            name, line_text = entry
            if isinstance(line_text, str):
                lines.append((_normalized_name(name), line_text))
    return (
        _normalized_name(speaker),
        lines or None,
    )
