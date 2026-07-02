"""Load common subtitle formats (VTT/SRT/ASS/SSA) into the shared cue-block shape.

``realign.parse_vtt_blocks`` already reads SRT (the numeric index line and the
comma-decimal timing line are within its tolerance), so only ASS/SSA needs a
dedicated parser. All loaders return the realign block contract:
``[{text, start, end, lyric?}]``.
"""

from __future__ import annotations

import re
from pathlib import Path

from voxweave.realign import _parse_ts, parse_vtt_blocks

# Subtitle formats the file-based commands (export/translate/pack/burn) accept.
SUBTITLE_EXTS = (".vtt", ".srt", ".ass", ".ssa")

# Fallback Events field order (the standard v4.00+ layout) when no Format line
# precedes the first Dialogue line.
_ASS_DEFAULT_FIELDS = [
    "layer",
    "start",
    "end",
    "style",
    "name",
    "marginl",
    "marginr",
    "marginv",
    "effect",
    "text",
]

_ASS_FULL_ITALIC_RE = re.compile(r"\{\\i1\}(.*)\{\\i0\}\Z", re.DOTALL)
_ASS_ITALIC_ON_RE = re.compile(r"\{\\i1\}")
_ASS_ITALIC_OFF_RE = re.compile(r"\{\\i0?\}")
_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")


def _ass_plain_text(raw: str) -> str:
    """ASS event text -> cue text: ``\\N``/``\\n`` become line breaks, ``\\h``
    a space, italic overrides become inline ``<i>``/``</i>`` tags (the form the
    SRT/ASS renderers understand), all other override blocks are dropped.

    A whole-line italic wrap is removed entirely: it is styling (typically a
    song line), and keeping it would mask the music-note lyric detection below.
    """
    t = raw.strip()
    m = _ASS_FULL_ITALIC_RE.fullmatch(t)
    if m and "{" not in m.group(1):
        t = m.group(1)
    t = _ASS_ITALIC_ON_RE.sub("<i>", t)
    t = _ASS_ITALIC_OFF_RE.sub("</i>", t)
    t = _ASS_OVERRIDE_RE.sub("", t)
    t = t.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    return "\n".join(ln.strip() for ln in t.split("\n")).strip()


def _lyric_block(cue: str, start: float | None, end: float | None) -> dict | None:
    """Cue text -> block dict, stripping a music-note wrap into the ``lyric``
    flag (same convention as ``parse_vtt_blocks``); None when nothing remains."""
    if not cue:
        return None
    block: dict = {"text": cue, "start": start, "end": end}
    if len(cue) > 2 and cue.startswith("♪") and cue.endswith("♪"):
        block["text"] = cue[1:-1].strip()
        block["lyric"] = True
        if not block["text"]:
            return None
    return block


def parse_ass_blocks(text: str) -> list[dict]:
    """Parse ASS/SSA ``[Events]`` Dialogue lines -> ordered cue blocks.

    Honors the section's Format line for field order (falling back to the
    standard v4.00+ layout), skips Comment lines, and sorts by start time
    (ASS events are not required to be chronological).
    """
    blocks: list[dict] = []
    fields: list[str] | None = None
    in_events = False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            fields = None
            continue
        if not in_events or not line:
            continue
        key, _, val = line.partition(":")
        k = key.strip().lower()
        if k == "format":
            fields = [f.strip().lower() for f in val.split(",")]
            continue
        if k != "dialogue":
            continue
        names = fields or _ASS_DEFAULT_FIELDS
        parts = val.split(",", len(names) - 1)
        if len(parts) < len(names):
            continue
        row = dict(zip(names, parts))
        start = _parse_ts(row.get("start", ""))
        end = _parse_ts(row.get("end", ""))
        if start is None or end is None:
            continue
        block = _lyric_block(_ass_plain_text(row.get("text", "")), start, end)
        if block is not None:
            blocks.append(block)
    blocks.sort(key=lambda b: (b["start"], b["end"]))
    return blocks


def require_subtitle(path: Path, *, exts: tuple[str, ...] = SUBTITLE_EXTS) -> Path:
    """Reject inputs whose extension is not an accepted subtitle format."""
    p = Path(path)
    if p.suffix.lower() not in exts:
        allowed = "/".join(e.lstrip(".") for e in exts)
        raise ValueError(
            f"{p.name}: unsupported subtitle format"
            f" (got {p.suffix or 'no extension'!r}, expected {allowed})"
        )
    return p


def load_subtitle_blocks(path: Path) -> list[dict]:
    """Read and parse a subtitle file by extension -> cue blocks; raise when the
    format is unsupported or the file yields no cues."""
    p = require_subtitle(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".ass", ".ssa"):
        blocks = parse_ass_blocks(text)
    else:  # .vtt and .srt share a parser
        blocks = parse_vtt_blocks(text)
    if not blocks:
        raise RuntimeError(f"no cues in {p.name}")
    return blocks
