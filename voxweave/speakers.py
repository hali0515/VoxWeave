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
from pathlib import Path
from typing import Any

from voxweave import fsio

log = logging.getLogger("voxweave")

MAPPING_VERSION = 1
MIN_SNIPPET_S = 2.0
MAX_SNIPPET_S = 6.0
MAX_SNIPPETS_PER_SPEAKER = 3
MIN_SNIPPET_GAP_S = 1.0
FFMPEG_TIMEOUT = float(os.environ.get("VOXWEAVE_FFMPEG_TIMEOUT", "3600"))

Span = tuple[float, float]
Turn = tuple[float, float, str]

_VOICE_WRAP_RE = re.compile(
    r"\A<v(?:[ \t]+([^>]*?))?>(.*)</v>\Z", re.IGNORECASE | re.DOTALL
)


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

    Windows are always cut from whole clean spans, so a single 6-second
    utterance stays one useful clip. The minimum gap applies only to multiple
    cuts from the same continuous run; separately voiced runs remain eligible
    even when their natural pause is shorter.
    """
    clean = [
        span for span in _merge_spans(regions) if span[1] - span[0] >= MIN_SNIPPET_S
    ]
    if not clean or limit <= 0:
        return []
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

    Only ASCII layout whitespace is collapsed; meaningful NBSP and ideographic
    spaces remain as entered. ASS applies its comma escape separately.
    """
    normalized = name.translate(_NAME_TRANSLATION)
    return _NAME_ASCII_WHITESPACE_RE.sub(" ", normalized).strip(" \t")


def sanitize_ass_speaker_name(name: str) -> str:
    """Normalize a name and escape ASS's unquotable comma field delimiter."""
    return sanitize_speaker_name(name).replace(",", "，")


def _normalized_name(value: object) -> str | None:
    """Return one safe non-empty name, or None for non-name values."""
    if not isinstance(value, str):
        return None
    name = sanitize_speaker_name(value)
    return name or None


def strip_srt_speaker_prefixes(
    text: str, known_names: Sequence[str]
) -> tuple[str, str | None, list[str | None] | None]:
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
        return plain, None, line_names
    return plain, line_names[0], None


def _voice_name(name: str) -> str:
    """Escape a display name for a WebVTT voice annotation."""
    return html.escape(sanitize_speaker_name(name), quote=False)


def speaker_layout(
    text: str,
    *,
    speaker: str | None = None,
    speakers: Sequence[str | None] | None = None,
) -> tuple[str | None, list[str | None] | None]:
    """Resolve safe cue/line names for the current rendered text layout.

    Text-mutating stages can collapse or wrap lines after names were parsed. A
    single remaining identity can safely become a cue label; several distinct
    identities stay unnamed instead of producing an unmappable composite.
    """
    cue_name = _normalized_name(speaker)
    line_names = (
        [_normalized_name(name) for name in speakers] if speakers is not None else None
    )
    if line_names is not None and len(line_names) == len(text.split("\n")):
        if any(line_names):
            return None, line_names
    if line_names:
        distinct = list(dict.fromkeys(name for name in line_names if name))
        if len(distinct) == 1:
            return distinct[0], None
    return (cue_name or None), None


def voice_tag_text(
    text: str,
    *,
    speaker: str | None = None,
    speakers: Sequence[str | None] | None = None,
) -> str:
    """Apply one cue-level or several line-level WebVTT voice tags."""
    speaker, speakers = speaker_layout(text, speaker=speaker, speakers=speakers)
    lines = text.split("\n")
    if speakers is not None:
        return "\n".join(
            f"<v {_voice_name(name)}>{line}</v>" if name else line
            for line, name in zip(lines, speakers)
        )
    if speaker:
        return f"<v {_voice_name(speaker)}>{text}</v>"
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
    return voice_tag_text(text, speakers=resolved)


def strip_voice_tags(text: str) -> tuple[str, str | None, list[str | None] | None]:
    """Strip full-cue or per-line VTT voice tags into display metadata.

    Returns ``(plain_text, speaker, speakers)``.  ``speaker`` is used for one
    wrapper around the whole cue; ``speakers`` preserves line positions for a
    dash cue where named and unnamed speakers may be mixed.
    """

    def unwrap(value: str) -> tuple[str, str | None] | None:
        match = _VOICE_WRAP_RE.fullmatch(value)
        if match is None:
            return None
        name = html.unescape((match.group(1) or "").strip())
        return match.group(2), name if name.strip() else None

    whole = unwrap(text)
    if whole is not None and not re.search(r"</?v(?:\s|>)", whole[0], re.IGNORECASE):
        plain, name = whole
        return plain, name, None

    plain_lines: list[str] = []
    names: list[str | None] = []
    tagged = False
    for line in text.split("\n"):
        parsed = unwrap(line)
        if parsed is None:
            plain_lines.append(line)
            names.append(None)
        else:
            plain, name = parsed
            plain_lines.append(plain)
            names.append(name)
            tagged = True
    if tagged:
        return "\n".join(plain_lines), None, names
    return text, None, None


def speaker_metadata(
    block: Mapping[str, Any],
) -> tuple[str | None, list[str | None] | None]:
    """Return normalized cue-level and line-level name metadata."""
    speaker = block.get("speaker")
    line_names = block.get("speakers")
    return (
        _normalized_name(speaker),
        [_normalized_name(name) for name in line_names]
        if isinstance(line_names, list)
        else None,
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
        cards.append(
            '<section class="speaker">'
            f'<div class="speaker-head"><code>{html.escape(speaker_id)}</code>'
            f'<input type="text" data-speaker="{sid}" aria-label="Name for {sid}" '
            'placeholder="Enter display name" autocomplete="off"></div>'
            f'<div class="clips">{"".join(audio)}</div></section>'
        )

    safe_title = html.escape(title)
    safe_mapping = html.escape(mapping_name)
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
pre {{ overflow-x: auto; margin-bottom: 0; user-select: all; }}
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
document.querySelector('#copy').addEventListener('click', async (event) => {{
  try {{ await navigator.clipboard.writeText(output.textContent); event.target.textContent = 'Copied'; }}
  catch (_) {{ const range = document.createRange(); range.selectNodeContents(output); getSelection().removeAllRanges(); getSelection().addRange(range); event.target.textContent = 'Select and copy'; }}
}});
update();
</script>
</body>
</html>
"""


def create_speaker_audition(media: Path) -> Path:
    """Create ``<stem>.speakers.html`` and a new empty mapping skeleton.

    Existing mapping files are user data and cause an early refusal before
    ffmpeg or either output writer runs.
    """
    from voxweave import pipeline

    media = Path(media)
    json_path = pipeline.swap_ext(media, ".json")
    mapping_path = pipeline.swap_ext(media, ".speakers.json")
    html_path = pipeline.swap_ext(media, ".speakers.html")
    if mapping_path.exists():
        raise RuntimeError(
            f"refusing to overwrite existing {mapping_path.name}; edit it directly, "
            "or move it aside before regenerating the audition page"
        )
    if not json_path.exists():
        raise FileNotFoundError(
            f"sibling transcript {json_path.name} not found; run voxweave {media.name} --diarize first"
        )
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid sibling JSON in {json_path.name}: {exc}") from exc

    turns = pipeline._turns_in(data.get("speaker_turns"))
    if not turns:
        raise RuntimeError(
            f"{json_path.name} has no speaker_turns; run voxweave {media.name} --diarize first"
        )
    vad_speech = pipeline._spans_in(data.get("vad_speech"))
    sing_spans = pipeline._spans_in(data.get("sing_spans"))
    picks = select_snippets(turns, vad_speech, sing_spans)

    embedded: dict[str, list[tuple[Span, str]]] = {label: [] for label in picks}
    with tempfile.TemporaryDirectory(prefix="voxweave_speakers_") as tmp_dir:
        root = Path(tmp_dir)
        for speaker_index, (speaker_id, spans) in enumerate(picks.items()):
            for clip_index, (start, end) in enumerate(spans):
                clip_path = root / f"speaker-{speaker_index}-{clip_index}.mp3"
                extract_clip(media, start, end, clip_path)
                encoded = base64.b64encode(clip_path.read_bytes()).decode("ascii")
                embedded[speaker_id].append(
                    ((start, end), f"data:audio/mpeg;base64,{encoded}")
                )

    page = _render_audition_html(media.name, mapping_path.name, embedded)
    skeleton = {
        "version": MAPPING_VERSION,
        "speakers": {speaker_id: "" for speaker_id in picks},
    }
    # Write the regenerable page first.  If the protected mapping write fails,
    # the command remains rerunnable because no user-data sentinel was created.
    fsio.atomic_write_text(html_path, page)
    try:
        fsio.atomic_write_text_new(
            mapping_path, json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n"
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"refusing to overwrite existing {mapping_path.name}; edit it directly, "
            "or move it aside before regenerating the audition page"
        ) from exc
    log.info("wrote %s and %s", html_path.name, mapping_path.name)
    return html_path


__all__ = [
    "FFMPEG_TIMEOUT",
    "MAPPING_VERSION",
    "MAX_SNIPPET_S",
    "MIN_SNIPPET_GAP_S",
    "MIN_SNIPPET_S",
    "build_clip_command",
    "create_speaker_audition",
    "extract_clip",
    "load_speaker_display_names",
    "load_speaker_mapping",
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
