"""Speaker-name mapping, audition snippets, and display metadata helpers.

Phase 1 deliberately keeps names out of the transcription sibling JSON.  The
``.speakers.json`` sidecar maps diarizer ids to display names, while VTT voice
tags are parsed into transient cue metadata before any text-processing stage.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

from voxweave import fsio
from voxweave.mediasnapshot import MediaSnapshot, SnapshotUnavailable
from voxweave.voicebase import (
    VOICES_STORE_MAX_BYTES,
    VOICEPRINTS_MAX_BYTES,
    Phase2DataError,
    html_attribute,
    html_text,
    media_fingerprint,
    require_capture_id,
    require_sha256,
    strict_turn_projection,
    strict_json_object_loads,
    validate_voiceprint_conjunction,
    validate_voiceprints_mapping,
    voiceprints_digest,
)
from voxweave.voiceepisode import episode_lock
from voxweave.voicematch import (
    CompatibilityError,
    MatchThresholds,
    SpeakerMatch,
    ThresholdError,
    build_compatibility_fingerprint,
    build_suggest_record,
    compatibility_equal,
    delete_suggest,
    match_speakers,
    parse_thresholds,
    write_suggest,
)
from voxweave.voicestore import (
    EnrollmentRefusal,
    canonical_store_path,
    enroll_exemplar,
    exclusive_store_lock,
    new_voice_store,
    normalize_episode,
    normalize_show,
    normalize_speaker_key,
    resolve_identity_id,
    shared_store_lock,
    validate_voice_store,
    write_voice_store,
)

log = logging.getLogger("voxweave")

MAPPING_VERSION = 1
MIN_SNIPPET_S = 2.0
MAX_SNIPPET_S = 6.0
MAX_SNIPPETS_PER_SPEAKER = 3
MIN_SNIPPET_GAP_S = 1.0
FFMPEG_TIMEOUT = float(os.environ.get("VOXWEAVE_FFMPEG_TIMEOUT", "3600"))

Span = tuple[float, float]
Turn = tuple[float, float, str]
SpeakerLine = tuple[str | None, str]


def _read_exact_object(path: Path, max_bytes: int) -> tuple[bytes, dict[str, object]]:
    raw = Path(path).read_bytes()
    return raw, strict_json_object_loads(
        raw,
        max_bytes=max_bytes,
        source=Path(path).name,
    )


def _read_sibling_exact(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = Path(path).read_bytes()
    return raw, strict_json_object_loads(
        raw,
        max_bytes=max(1, len(raw)),
        source=Path(path).name,
    )


def _declared_voiceprint_pair(
    sibling: Mapping[str, object],
) -> tuple[str, str] | None:
    capture_present = "voiceprint_capture" in sibling
    media_present = "voiceprint_media" in sibling
    if not capture_present and not media_present:
        return None
    if not capture_present or not media_present:
        raise Phase2DataError("sibling voiceprint keys are not a complete pair")
    return (
        require_capture_id(
            sibling.get("voiceprint_capture"), "sibling.voiceprint_capture"
        ),
        require_sha256(sibling.get("voiceprint_media"), "sibling.voiceprint_media"),
    )


def _resolved_voices_path(media: Path, voices: Path | None) -> tuple[Path, bool]:
    explicit = voices is not None
    raw = Path(voices) if voices is not None else media.parent / "voxweave.voices.json"
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    path = canonical_store_path(raw)
    media_parent = Path(os.path.realpath(os.fspath(media.parent)))
    if path.parent == media_parent and path.name.startswith(f"{media.stem}."):
        raise RuntimeError(
            f"voices store {path.name} is inside the {media.stem}.* episode namespace"
        )
    return path, explicit


def _load_generation_store(
    media: Path,
    *,
    voices: Path | None,
    show: str | None,
) -> tuple[Path, bytes, dict[str, object]] | None:
    path, explicit = _resolved_voices_path(media, voices)
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"explicit voices store not found: {path}")
        if show is not None:
            raise FileNotFoundError(
                f"no discovered voices store for --show {show!r}: expected {path}"
            )
        return None
    try:
        raw, store = _read_exact_object(path, VOICES_STORE_MAX_BYTES)
        validated = validate_voice_store(store)
    except (OSError, Phase2DataError) as exc:
        if explicit:
            raise RuntimeError(
                f"explicit voices store {path} is unusable: {exc}"
            ) from exc
        log.warning(
            "discovered voices store %s is unusable; matching skipped: %s", path, exc
        )
        return None
    if show is None and not explicit:
        log.info(
            "found voices store %s; pass --show %r to activate matching",
            path,
            validated.show,
        )
        return None
    if show is not None and normalize_show(show) != normalize_show(validated.show):
        if explicit:
            raise RuntimeError(
                f"--show {show!r} does not match voices store show {validated.show!r}"
            )
        log.info(
            "discovered voices store %s belongs to a different show; matching skipped",
            path,
        )
        return None
    return path, raw, store


def _load_voiceprints_exact(path: Path) -> tuple[bytes, dict[str, object]]:
    raw, sidecar = _read_exact_object(path, VOICEPRINTS_MAX_BYTES)
    validate_voiceprints_mapping(sidecar)
    return raw, sidecar


def _matching_record(
    sidecar: Mapping[str, object],
    store_path: Path,
    store: Mapping[str, object],
) -> tuple[dict[str, SpeakerMatch], dict[str, object], MatchThresholds] | None:
    sidecar_provenance = cast(Mapping[str, object], sidecar["provenance"])
    store_provenance = cast(Mapping[str, object], store["provenance"])
    left = build_compatibility_fingerprint(sidecar_provenance)
    right = build_compatibility_fingerprint(store_provenance)
    if not compatibility_equal(left, right):
        log.warning("voice compatibility differs or is unresolved; matching skipped")
        return None
    if sidecar_provenance.get("torch_version") != store_provenance.get("torch_version"):
        log.warning("voice evidence was produced by a different torch version")
    try:
        thresholds = parse_thresholds()
    except ThresholdError as exc:
        log.warning("voice matching thresholds are invalid; matching skipped: %s", exc)
        return None
    centroids = cast(Mapping[str, object], sidecar["speakers"])
    matches = match_speakers(centroids, store, thresholds)
    record = build_suggest_record(
        matches,
        capture_id=cast(str, sidecar["capture_id"]),
        voiceprints_content_digest=voiceprints_digest(sidecar),
        compatibility=left,
        thresholds=thresholds,
        store_path=store_path,
        store=store,
    )
    return matches, record, thresholds


_VOICE_WRAP_RE = re.compile(
    r"\A<v(?:[ \t]+([^>]*?))?>(.*)</v>\Z", re.IGNORECASE | re.DOTALL
)
_VOICE_SPAN_RE = re.compile(
    r"<v(?:[ \t]+([^>]*?))?>(.*?)</v>", re.IGNORECASE | re.DOTALL
)
_VOICE_TAG_RE = re.compile(r"(?:<v(?:[ \t]+[^>]*?)?>|</v>)", re.IGNORECASE)


def _valid_spans(spans: Sequence[Span] | None) -> list[Span]:
    """Return finite, positive spans in chronological order."""
    out: list[Span] = []
    for start, end in spans or ():
        a, b = float(start), float(end)
        if a < b and math.isfinite(a) and math.isfinite(b):
            out.append((a, b))
    return sorted(out)


def _merge_spans(spans: Sequence[Span] | None) -> list[Span]:
    """Union overlapping or touching spans."""
    merged: list[Span] = []
    for start, end in _valid_spans(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersect_spans(left: Sequence[Span], right: Sequence[Span]) -> list[Span]:
    """Intersection of two span unions."""
    a = _merge_spans(left)
    b = _merge_spans(right)
    out: list[Span] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract_spans(source: Sequence[Span], removed: Sequence[Span]) -> list[Span]:
    """Subtract the union of ``removed`` from the union of ``source``."""
    cuts = _merge_spans(removed)
    out: list[Span] = []
    for start, end in _merge_spans(source):
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor:
                continue
            if cut_start >= end:
                break
            if cut_start > cursor:
                out.append((cursor, min(cut_start, end)))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return out


def _window(span: Span, target: float | None = None) -> Span:
    """Make a 2-6 second audition window inside ``span`` near ``target``."""
    start, end = span
    length = min(MAX_SNIPPET_S, end - start)
    center = (start + end) / 2.0 if target is None else target
    clip_start = min(max(center - length / 2.0, start), end - length)
    return (clip_start, clip_start + length)


def _pick_spread(regions: Sequence[Span], limit: int) -> list[Span]:
    """Pick long clean windows near the early, middle, and late extent.

    A single 6-18 second run yields two longest-possible windows with at least
    one second between them. Otherwise the minimum gap applies only to cuts
    from the same continuous run; separately voiced runs remain eligible even
    when their natural pause is shorter.
    """
    clean = [
        span for span in _merge_spans(regions) if span[1] - span[0] >= MIN_SNIPPET_S
    ]
    if not clean or limit <= 0:
        return []
    if len(clean) == 1 and limit >= 2:
        start, end = clean[0]
        run_length = end - start
        if 6.0 <= run_length <= 18.0:
            clip_length = min(MAX_SNIPPET_S, (run_length - MIN_SNIPPET_GAP_S) / 2.0)
            return [
                (start, start + clip_length),
                (end - clip_length, end),
            ]
    extent_start, extent_end = clean[0][0], clean[-1][1]
    bin_count = min(limit, MAX_SNIPPETS_PER_SPEAKER)
    bin_width = (extent_end - extent_start) / bin_count
    bins = [
        (extent_start + bin_width * index, extent_start + bin_width * (index + 1))
        for index in range(bin_count)
    ]
    picked: list[tuple[Span, int]] = []

    def separated(candidate: Span, source_index: int) -> bool:
        for (start, end), picked_source in picked:
            if source_index != picked_source:
                continue
            if not (
                candidate[1] + MIN_SNIPPET_GAP_S <= start
                or end + MIN_SNIPPET_GAP_S <= candidate[0]
            ):
                return False
        return True

    for bin_start, bin_end in bins:
        target = (bin_start + bin_end) / 2.0
        candidates = [
            (_window(span, target), source_index)
            for source_index, span in enumerate(clean)
            if span[0] < bin_end and span[1] > bin_start
        ]
        candidates = [
            (span, source_index)
            for span, source_index in candidates
            if separated(span, source_index)
        ]
        if not candidates:
            continue
        best, source_index = max(
            candidates,
            key=lambda item: (
                item[0][1] - item[0][0],
                -abs((item[0][0] + item[0][1]) / 2.0 - target),
                -item[0][0],
            ),
        )
        picked.append((best, source_index))
        if len(picked) == limit:
            return sorted(span for span, _source in picked)

    # A narrow or empty time bin can leave another usable window unselected.
    # Fill with centered whole-span windows, retaining the same-run gap above.
    while len(picked) < limit:
        candidates = [
            (_window(span), source_index) for source_index, span in enumerate(clean)
        ]
        candidates = [
            (span, source_index)
            for span, source_index in candidates
            if separated(span, source_index)
        ]
        if not candidates:
            break
        best, source_index = max(
            candidates,
            key=lambda item: (item[0][1] - item[0][0], -item[0][0]),
        )
        picked.append((best, source_index))
    return sorted(span for span, _source in picked)


def select_snippets(
    turns: Sequence[Turn],
    vad_speech: Sequence[Span] | None,
    sing_spans: Sequence[Span] | None = None,
    *,
    max_per_speaker: int = MAX_SNIPPETS_PER_SPEAKER,
) -> dict[str, list[Span]]:
    """Select clean audition spans for every diarized speaker.

    A candidate must be inside that speaker's turn, outside every other
    speaker's turn, inside VAD speech, and outside detected singing.  Returned
    clips are always 2-6 seconds and are ordered chronologically.
    """
    labels = list(dict.fromkeys(str(label) for _, _, label in turns))
    by_speaker: dict[str, list[Span]] = {label: [] for label in labels}
    for start, end, label in turns:
        if float(end) > float(start):
            by_speaker[str(label)].append((float(start), float(end)))

    voiced = _merge_spans(vad_speech)
    singing = _merge_spans(sing_spans)
    selected: dict[str, list[Span]] = {}
    for label in labels:
        other = [
            (float(start), float(end))
            for start, end, other_label in turns
            if str(other_label) != label and float(end) > float(start)
        ]
        exclusive = _subtract_spans(by_speaker[label], other)
        clean = _subtract_spans(_intersect_spans(exclusive, voiced), singing)
        selected[label] = _pick_spread(clean, max_per_speaker)
    return selected


def _mapping_entries(path: Path) -> dict[str, Any]:
    """Read and validate the entries object from a version-1 mapping."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid speaker mapping JSON in {path.name}: {exc}"
        ) from exc
    version = raw.get("version") if isinstance(raw, dict) else None
    if type(version) is not int or version != MAPPING_VERSION:
        raise RuntimeError(
            f"{path.name} must use speaker mapping version {MAPPING_VERSION}"
        )
    speakers = raw.get("speakers")
    if not isinstance(speakers, dict):
        raise RuntimeError(f"{path.name} must contain a speakers object")
    return speakers


def load_speaker_display_names(path: Path) -> list[str]:
    """Return the non-empty names in a mapping, in stable mapping order."""
    names: list[str] = []
    for raw_name in _mapping_entries(path).values():
        if isinstance(raw_name, str) and raw_name.strip():
            names.append(raw_name)
    return list(dict.fromkeys(names))


def load_speaker_mapping(
    path: Path, known_ids: Sequence[str] | set[str]
) -> dict[str, str]:
    """Read a version-1 mapping, ignoring empty names and unknown speaker ids.

    Unknown ids are reported in one logger call so a stale phase-1 mapping is
    visible without flooding an episode replay.  Non-empty names are preserved
    exactly as entered; surrounding whitespace only decides whether a value is
    effectively empty.
    """
    path = Path(path)
    speakers = _mapping_entries(path)

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
            path.name,
            ", ".join(sorted(unknown)),
        )
    return names


_NAME_RECORD_SEPARATORS = "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_NAME_TRANSLATION = str.maketrans({char: " " for char in _NAME_RECORD_SEPARATORS})
_NAME_ASCII_WHITESPACE_RE = re.compile(r"[ \t]+")


def sanitize_speaker_name(name: str) -> str:
    """Normalize record separators without changing display punctuation.

    Only ASCII layout runs are collapsed. Unicode whitespace is stripped at
    name edges but remains meaningful inside a name. ASS applies its comma
    escape separately.
    """
    normalized = name.translate(_NAME_TRANSLATION)
    return _NAME_ASCII_WHITESPACE_RE.sub(" ", normalized).strip()


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


def build_clip_command(
    media: Path, start: float, end: float, output: Path
) -> list[str]:
    """Build one 16 kHz mono MP3 clip using ffmpeg's default audio stream.

    The transcription decoder also leaves stream choice to ffmpeg; keeping the
    same selection rule is essential for multi-audio media.
    """
    duration = float(end) - float(start)
    if duration <= 0:
        raise ValueError("speaker snippet end must be after start")
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start):.3f}",
        "-i",
        str(media),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "32k",
        str(output),
    ]


def run_clip_command(cmd: list[str]) -> None:
    """Execute a clip command with the repository's non-interactive contract."""
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFMPEG_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found; install ffmpeg and make sure it is on your PATH"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg speaker clip timed out after {FFMPEG_TIMEOUT:g}s"
        ) from exc
    if proc.returncode != 0:
        detail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        suffix = f"\n{detail}" if detail else ""
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}){suffix}")


def extract_clip(media: Path, start: float, end: float, output: Path) -> None:
    """Extract one clip atomically; command construction remains separately testable."""
    with fsio.atomic_path(output) as tmp:
        run_clip_command(build_clip_command(media, start, end, tmp))


def _render_audition_html(
    title: str,
    mapping_name: str,
    snippets: Mapping[str, Sequence[tuple[Span, str]]],
    matches: Mapping[str, SpeakerMatch] | None = None,
) -> str:
    """Build the self-contained offline audition page."""
    cards: list[str] = []
    for speaker_id, clips in snippets.items():
        audio = []
        for (start, end), data_uri in clips:
            audio.append(
                "<figure><figcaption>"
                f"{start:.1f}s- {end:.1f}s"
                f'</figcaption><audio controls preload="none" src="{data_uri}"></audio></figure>'
            )
        if not audio:
            audio.append(
                '<p class="empty">No clean 2-6 second snippet was available.</p>'
            )
        sid = html.escape(speaker_id, quote=True)
        if matches is None:
            cards.append(
                '<section class="speaker">'
                f'<div class="speaker-head"><code>{html.escape(speaker_id)}</code>'
                f'<input type="text" data-speaker="{sid}" aria-label="Name for {sid}" '
                'placeholder="Enter display name" autocomplete="off"></div>'
                f'<div class="clips">{"".join(audio)}</div></section>'
            )
            continue

        match = matches.get(speaker_id)
        candidates = match.candidates if match is not None else ()
        prefill = (
            candidates[0].display_name
            if match is not None and match.decision == "prefill" and candidates
            else ""
        )
        suggestion_buttons = "".join(
            '<button type="button" class="use-suggestion" '
            f'data-use="{html_attribute(candidate.display_name)}">'
            f"{html_text(candidate.display_name)} ({candidate.similarity:.2f}) [use]"
            "</button>"
            for candidate in candidates
        )
        if not suggestion_buttons:
            suggestion_buttons = '<span class="no-suggestion">No stored match.</span>'
        machine_mark = (
            '<span class="machine-mark">machine-suggested; review before saving</span>'
            if prefill
            else ""
        )
        cards.append(
            '<section class="speaker">'
            f'<div class="speaker-head"><code>{html.escape(speaker_id)}</code>'
            f'<input type="text" data-speaker="{sid}" aria-label="Name for {sid}" '
            f'value="{html_attribute(prefill)}" placeholder="Enter display name" '
            f'autocomplete="off">{machine_mark}</div>'
            f'<div class="suggestions">{suggestion_buttons}</div>'
            f'<div class="clips">{"".join(audio)}</div></section>'
        )

    safe_title = html.escape(title)
    safe_mapping = html.escape(mapping_name)
    matching_css = (
        "\n.suggestions { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .75rem; }"
        "\n.machine-mark, .no-suggestion { color: #777; font-size: .8rem; }"
        if matches is not None
        else ""
    )
    matching_script = (
        "for (const button of document.querySelectorAll('.use-suggestion')) {"
        "\n  button.addEventListener('click', () => {"
        "\n    const field = button.closest('.speaker').querySelector('[data-speaker]');"
        "\n    field.value = button.dataset.use; field.dispatchEvent(new Event('input'));"
        "\n  });"
        "\n}\n"
        if matches is not None
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title} - speaker audition</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 960px; margin: 0 auto; padding: 2rem; line-height: 1.45; }}
h1 {{ margin-bottom: .25rem; }}
.lede {{ color: #777; margin-top: 0; }}
.speaker {{ border: 1px solid #8886; border-radius: .75rem; padding: 1rem; margin: 1rem 0; }}
.speaker-head {{ display: grid; grid-template-columns: minmax(8rem, 1fr) 2fr; gap: 1rem; align-items: center; }}
input {{ font: inherit; padding: .6rem .75rem; border: 1px solid #8888; border-radius: .4rem; }}
.clips {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .75rem; margin-top: 1rem; }}
figure {{ margin: 0; }} figcaption {{ font-size: .8rem; color: #777; }} audio {{ width: 100%; }}
.empty {{ color: #777; font-style: italic; }}
.output {{ position: sticky; bottom: 0; background: Canvas; border: 1px solid #8886; border-radius: .75rem; padding: 1rem; box-shadow: 0 0 2rem #0003; }}
.output-head {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; }}
button {{ font: inherit; padding: .45rem .8rem; cursor: pointer; }}
pre {{ overflow-x: auto; margin-bottom: 0; user-select: all; }}{matching_css}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<p class="lede">Listen, enter names, then copy the JSON into <code>{safe_mapping}</code>. Everything on this page is embedded and works offline.</p>
{"".join(cards)}
<section class="output">
<div class="output-head"><strong>Speaker mapping JSON</strong><button id="copy" type="button">Copy JSON</button></div>
<pre id="json" aria-live="polite"></pre>
</section>
<script>
const fields = [...document.querySelectorAll('[data-speaker]')];
const output = document.querySelector('#json');
function update() {{
  const speakers = {{}};
  for (const field of fields) speakers[field.dataset.speaker] = field.value;
  output.textContent = JSON.stringify({{version: 1, speakers}}, null, 2);
}}
for (const field of fields) field.addEventListener('input', update);
{matching_script}document.querySelector('#copy').addEventListener('click', async (event) => {{
  try {{ await navigator.clipboard.writeText(output.textContent); event.target.textContent = 'Copied'; }}
  catch (_) {{ const range = document.createRange(); range.selectNodeContents(output); getSelection().removeAllRanges(); getSelection().addRange(range); event.target.textContent = 'Select and copy'; }}
}});
update();
</script>
</body>
</html>
"""


def _mapping_exists_refusal(mapping_path: Path) -> RuntimeError:
    return RuntimeError(
        f"refusing to overwrite existing {mapping_path.name}; edit it directly, "
        "or move it aside before regenerating the audition page"
    )


def _delete_stale_suggest_for_refusal(
    media: Path,
    *,
    sibling_path: Path,
    sibling_bytes: bytes,
) -> None:
    from voxweave import pipeline

    mapping_path = pipeline.swap_ext(media, ".speakers.json")
    with episode_lock(media):
        if mapping_path.exists():
            return
        try:
            unchanged = sibling_path.read_bytes() == sibling_bytes
        except OSError:
            return
        if unchanged:
            delete_suggest(pipeline.speakers_suggest_path(media))


def _publish_audition(
    media: Path,
    *,
    mapping_path: Path,
    html_path: Path,
    page: str,
    skeleton: Mapping[str, object],
    suggest_record: Mapping[str, object] | None,
) -> None:
    from voxweave import pipeline

    suggest_path = pipeline.speakers_suggest_path(media)
    if mapping_path.exists():
        raise _mapping_exists_refusal(mapping_path)
    try:
        if suggest_record is None:
            delete_suggest(suggest_path)
        else:
            write_suggest(suggest_path, suggest_record)
        fsio.atomic_write_text(html_path, page)
        fsio.atomic_write_text_new(
            mapping_path, json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n"
        )
    except FileExistsError as exc:
        delete_suggest(suggest_path)
        html_path.unlink(missing_ok=True)
        raise _mapping_exists_refusal(mapping_path) from exc
    except BaseException:
        # These outputs are wholly regenerable and owned by this invocation.
        # The protected mapping either does not exist or was created atomically
        # as the final successful edge.
        if not mapping_path.exists():
            delete_suggest(suggest_path)
            html_path.unlink(missing_ok=True)
        raise


def create_speaker_audition(
    media: Path,
    *,
    voices: Path | None = None,
    show: str | None = None,
    no_match: bool = False,
) -> Path:
    """Create ``<stem>.speakers.html`` and a new empty mapping skeleton.

    Existing mapping files are user data and cause an early refusal before
    ffmpeg or either output writer runs.
    """
    from voxweave import pipeline

    media = Path(media)
    if not media.is_file():
        raise FileNotFoundError(f"media file not found: {media}")
    json_path = pipeline.swap_ext(media, ".json")
    mapping_path = pipeline.swap_ext(media, ".speakers.json")
    html_path = pipeline.swap_ext(media, ".speakers.html")
    if mapping_path.exists():
        raise _mapping_exists_refusal(mapping_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"sibling transcript {json_path.name} not found; run voxweave {media.name} --diarize first"
        )
    try:
        sibling_bytes = json_path.read_bytes()
        data = json.loads(sibling_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid sibling JSON in {json_path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid sibling JSON in {json_path.name}: expected object")

    declared = "voiceprint_capture" in data or "voiceprint_media" in data
    pair: tuple[str, str] | None = None
    sidecar_bytes: bytes | None = None
    sidecar: dict[str, object] | None = None
    if declared and not no_match:
        try:
            data = strict_json_object_loads(
                sibling_bytes,
                max_bytes=max(1, len(sibling_bytes)),
                source=json_path.name,
            )
            pair = _declared_voiceprint_pair(data)
            sidecar_path = pipeline.voiceprints_path(media)
            sidecar_bytes, sidecar = _load_voiceprints_exact(sidecar_path)
        except (OSError, Phase2DataError) as exc:
            _delete_stale_suggest_for_refusal(
                media,
                sibling_path=json_path,
                sibling_bytes=sibling_bytes,
            )
            raise RuntimeError(
                "voiceprint evidence is declared but not usable; rerun "
                f"--diarize --voiceprints or use --no-match: {exc}"
            ) from exc

    try:
        store_stage = (
            None
            if no_match
            else _load_generation_store(media, voices=voices, show=show)
        )
    except (OSError, RuntimeError, Phase2DataError):
        _delete_stale_suggest_for_refusal(
            media,
            sibling_path=json_path,
            sibling_bytes=sibling_bytes,
        )
        raise

    turns = pipeline._turns_in(data.get("speaker_turns"))
    if not turns:
        _delete_stale_suggest_for_refusal(
            media,
            sibling_path=json_path,
            sibling_bytes=sibling_bytes,
        )
        raise RuntimeError(
            f"{json_path.name} has no speaker_turns; run voxweave {media.name} --diarize first"
        )
    vad_speech = pipeline._spans_in(data.get("vad_speech"))
    sing_spans = pipeline._spans_in(data.get("sing_spans"))
    picks = select_snippets(turns, vad_speech, sing_spans)

    def generate_from(source: Path, snapshot_fingerprint: str | None) -> Path:
        if pair is not None:
            assert sidecar is not None
            assert snapshot_fingerprint is not None
            try:
                validate_voiceprint_conjunction(sidecar, data, snapshot_fingerprint)
            except Phase2DataError as exc:
                _delete_stale_suggest_for_refusal(
                    media,
                    sibling_path=json_path,
                    sibling_bytes=sibling_bytes,
                )
                raise RuntimeError(
                    "voiceprint evidence does not bind this episode; rerun "
                    f"--diarize --voiceprints or use --no-match: {exc}"
                ) from exc

        matches: dict[str, SpeakerMatch] | None = None
        suggest_record: dict[str, object] | None = None
        if pair is not None and sidecar is not None and store_stage is not None:
            store_path, _store_bytes, store = store_stage
            try:
                matched = _matching_record(sidecar, store_path, store)
            except (CompatibilityError, Phase2DataError) as exc:
                log.warning("voice matching skipped: %s", exc)
                matched = None
            if matched is not None:
                matches, suggest_record, _thresholds = matched

        embedded: dict[str, list[tuple[Span, str]]] = {label: [] for label in picks}
        with tempfile.TemporaryDirectory(prefix="voxweave_speakers_") as tmp_dir:
            root = Path(tmp_dir)
            for speaker_index, (speaker_id, spans) in enumerate(picks.items()):
                for clip_index, (start, end) in enumerate(spans):
                    clip_path = root / f"speaker-{speaker_index}-{clip_index}.mp3"
                    extract_clip(source, start, end, clip_path)
                    encoded = base64.b64encode(clip_path.read_bytes()).decode("ascii")
                    embedded[speaker_id].append(
                        ((start, end), f"data:audio/mpeg;base64,{encoded}")
                    )

        page = _render_audition_html(
            media.name,
            mapping_path.name,
            embedded,
            matches,
        )
        skeleton = {
            "version": MAPPING_VERSION,
            "speakers": {speaker_id: "" for speaker_id in picks},
        }
        with episode_lock(media):
            if json_path.read_bytes() != sibling_bytes:
                raise RuntimeError("input changed during speaker generation; re-run")
            if pair is not None:
                assert sidecar_bytes is not None
                assert sidecar is not None
                if pipeline.voiceprints_path(media).read_bytes() != sidecar_bytes:
                    delete_suggest(pipeline.speakers_suggest_path(media))
                    raise RuntimeError(
                        "input changed during speaker generation; re-run"
                    )
                assert snapshot_fingerprint is not None
                validate_voiceprint_conjunction(sidecar, data, snapshot_fingerprint)
                if media_fingerprint(media) != snapshot_fingerprint:
                    delete_suggest(pipeline.speakers_suggest_path(media))
                    raise RuntimeError(
                        "media changed during speaker generation; re-run"
                    )
            if mapping_path.exists():
                raise _mapping_exists_refusal(mapping_path)
            if suggest_record is not None and store_stage is not None:
                store_path, store_bytes, _store = store_stage
                with shared_store_lock(store_path) as lock_handle:
                    current_bytes, current_store = _read_exact_object(
                        lock_handle.store_path,
                        VOICES_STORE_MAX_BYTES,
                    )
                    validate_voice_store(current_store)
                    if current_bytes != store_bytes:
                        delete_suggest(pipeline.speakers_suggest_path(media))
                        raise RuntimeError(
                            "voices store changed during speaker generation; re-run"
                        )
                    _publish_audition(
                        media,
                        mapping_path=mapping_path,
                        html_path=html_path,
                        page=page,
                        skeleton=skeleton,
                        suggest_record=suggest_record,
                    )
            else:
                _publish_audition(
                    media,
                    mapping_path=mapping_path,
                    html_path=html_path,
                    page=page,
                    skeleton=skeleton,
                    suggest_record=None,
                )
        return html_path

    if pair is not None:
        snapshots = ExitStack()
        try:
            snapshot = snapshots.enter_context(MediaSnapshot(media))
        except SnapshotUnavailable as exc:
            snapshots.close()
            _delete_stale_suggest_for_refusal(
                media,
                sibling_path=json_path,
                sibling_bytes=sibling_bytes,
            )
            raise RuntimeError(
                f"cannot validate declared voiceprints without a media snapshot: {exc}"
            ) from exc
        else:
            with snapshots:
                output = generate_from(snapshot.path, snapshot.fingerprint)
    else:
        output = generate_from(media, None)
    log.info("wrote %s and %s", html_path.name, mapping_path.name)
    return output


def enroll_speaker_voices(
    media: Path,
    *,
    voices: Path | None = None,
    show: str | None = None,
    episode: str | None = None,
    replace_episode: bool = False,
) -> Path:
    """Enroll human-named, bound episode centroids into one show store."""
    from voxweave import pipeline

    media = Path(media)
    if not media.is_file():
        raise FileNotFoundError(f"media file not found: {media}")
    json_path = pipeline.swap_ext(media, ".json")
    mapping_path = pipeline.swap_ext(media, ".speakers.json")
    sidecar_path = pipeline.voiceprints_path(media)
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"speaker mapping {mapping_path.name} not found; generate and review it first"
        )
    store_path, explicit = _resolved_voices_path(media, voices)
    create_store = not store_path.exists()
    if create_store and (not explicit or show is None):
        raise RuntimeError(
            "creating a voices store requires explicit --voices PATH and --show NAME"
        )
    if not create_store and not explicit and show is None:
        raise RuntimeError(
            f"found voices store {store_path}; pass --show NAME to confirm enrollment"
        )
    if create_store:
        store_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sibling_bytes, sibling = _read_sibling_exact(json_path)
        pair = _declared_voiceprint_pair(sibling)
        if pair is None:
            raise Phase2DataError("sibling does not declare voiceprint evidence")
        sidecar_bytes, _sidecar = _load_voiceprints_exact(sidecar_path)
        mapping_bytes = mapping_path.read_bytes()
    except (OSError, Phase2DataError) as exc:
        raise RuntimeError(f"cannot stage enrollment evidence: {exc}") from exc

    try:
        snapshot_context = MediaSnapshot(media)
        snapshot = snapshot_context.__enter__()
    except SnapshotUnavailable as exc:
        raise RuntimeError(f"cannot snapshot media for enrollment: {exc}") from exc
    try:
        with episode_lock(media):
            with exclusive_store_lock(store_path) as lock_handle:
                if json_path.read_bytes() != sibling_bytes:
                    raise RuntimeError("input changed during enrollment; re-run")
                if sidecar_path.read_bytes() != sidecar_bytes:
                    raise RuntimeError("input changed during enrollment; re-run")
                if mapping_path.read_bytes() != mapping_bytes:
                    raise RuntimeError("input changed during enrollment; re-run")

                _current_sibling_bytes, current_sibling = _read_sibling_exact(json_path)
                _current_sidecar_bytes, current_sidecar = _load_voiceprints_exact(
                    sidecar_path
                )
                validated_sidecar = validate_voiceprint_conjunction(
                    current_sidecar,
                    current_sibling,
                    snapshot.fingerprint,
                )
                if media_fingerprint(media) != snapshot.fingerprint:
                    raise RuntimeError("media changed during enrollment; re-run")

                if lock_handle.store_path.exists():
                    _store_bytes, store = _read_exact_object(
                        lock_handle.store_path,
                        VOICES_STORE_MAX_BYTES,
                    )
                    validated_store = validate_voice_store(store)
                    if show is not None and normalize_show(show) != normalize_show(
                        validated_store.show
                    ):
                        raise EnrollmentRefusal(
                            f"--show {show!r} does not match the voices store"
                        )
                else:
                    assert show is not None
                    store = new_voice_store(
                        show,
                        cast(Mapping[str, object], current_sidecar["provenance"]),
                    )

                side_compat = build_compatibility_fingerprint(
                    cast(Mapping[str, object], current_sidecar["provenance"])
                )
                store_compat = build_compatibility_fingerprint(
                    cast(Mapping[str, object], store["provenance"])
                )
                if not compatibility_equal(side_compat, store_compat):
                    raise EnrollmentRefusal(
                        "voice compatibility differs or is unresolved; enrollment refused"
                    )

                mapping = _mapping_entries(mapping_path)
                named = {
                    str(local_id): raw_name
                    for local_id, raw_name in mapping.items()
                    if isinstance(raw_name, str) and raw_name.strip()
                }
                if not named:
                    raise EnrollmentRefusal(
                        "speaker mapping has no human-entered names to enroll"
                    )
                for raw_name in named.values():
                    normalize_speaker_key(raw_name)

                durations: dict[str, float] = {}
                for start, end, local_id in strict_turn_projection(
                    current_sibling.get("speaker_turns")
                ):
                    durations[local_id] = durations.get(local_id, 0.0) + end - start

                groups: dict[
                    tuple[str, str],
                    list[tuple[str, str, tuple[int | float, ...], float]],
                ] = {}
                for local_id, raw_name in named.items():
                    vector = validated_sidecar.speakers.get(local_id)
                    if vector is None:
                        log.info(
                            "skipping named local speaker %s without a usable centroid",
                            local_id,
                        )
                        continue
                    identity_id = resolve_identity_id(store, raw_name)
                    group_key = (
                        ("identity", identity_id)
                        if identity_id is not None
                        else ("name", normalize_speaker_key(raw_name))
                    )
                    groups.setdefault(group_key, []).append(
                        (local_id, raw_name, vector, durations.get(local_id, 0.0))
                    )
                if not groups:
                    raise EnrollmentRefusal("nothing to enroll")

                selected: list[tuple[str, str, tuple[int | float, ...]]] = []
                for candidates in groups.values():
                    winner = min(candidates, key=lambda item: (-item[3], item[0]))
                    selected.append((winner[0], winner[1], winner[2]))
                    for skipped in sorted(
                        local_id
                        for local_id, *_rest in candidates
                        if local_id != winner[0]
                    ):
                        log.info(
                            "skipping over-split local speaker %s in favor of %s",
                            skipped,
                            winner[0],
                        )

                episode_key = normalize_episode(episode or media.stem)
                working = store
                mutations = 0
                noops = 0
                for _local_id, raw_name, vector in selected:
                    result = enroll_exemplar(
                        working,
                        raw_name=raw_name,
                        capture_id=validated_sidecar.capture_id,
                        media_fingerprint=validated_sidecar.media_fingerprint,
                        episode=episode_key,
                        vector=vector,
                        replace_episode=replace_episode,
                    )
                    working = result.store
                    if result.outcome == "noop":
                        noops += 1
                    else:
                        mutations += 1
                if mutations:
                    write_voice_store(lock_handle.store_path, working)
                    log.info(
                        "enrolled %d voice exemplar(s) into %s", mutations, store_path
                    )
                elif noops:
                    log.info("already enrolled; voices store unchanged")
                else:  # defensive: selected is nonempty and every transition is total
                    raise EnrollmentRefusal("nothing to enroll")
                return lock_handle.store_path
    finally:
        snapshot_context.__exit__(None, None, None)


def purge_voiceprints(media: Path) -> tuple[Path, ...]:
    """Delete the complete per-episode biometric/derived artifact set."""
    from voxweave import pipeline

    media = Path(media)
    targets = (
        pipeline.voiceprints_path(media),
        pipeline.speakers_suggest_path(media),
        pipeline.speakers_html_path(media),
    )
    removed: list[Path] = []
    with episode_lock(media):
        for target in targets:
            if target.exists():
                target.unlink()
                removed.append(target)
                log.info("deleted %s", target)
    return tuple(removed)


__all__ = [
    "FFMPEG_TIMEOUT",
    "MAPPING_VERSION",
    "MAX_SNIPPET_S",
    "MIN_SNIPPET_GAP_S",
    "MIN_SNIPPET_S",
    "build_clip_command",
    "create_speaker_audition",
    "enroll_speaker_voices",
    "extract_clip",
    "load_speaker_display_names",
    "load_speaker_mapping",
    "purge_voiceprints",
    "run_clip_command",
    "sanitize_ass_speaker_name",
    "sanitize_speaker_name",
    "select_snippets",
    "speaker_layout",
    "speaker_metadata",
    "strip_srt_speaker_prefixes",
    "strip_voice_tags",
    "voice_tag_text",
    "voice_text_for_block",
    "voice_text_for_ids",
]
