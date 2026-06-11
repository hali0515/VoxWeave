"""Export sibling VTT subtitles to SRT and ASS.

The VTT + JSON pair stays the source of truth; export renders presentation
formats from it. SRT is a plain re-rendering (inline ``<i>`` tags pass through
unchanged -- mainstream players honor them). ASS carries a Default dialogue
style and translates ``<i>``/``</i>`` into ``{\\i1}``/``{\\i0}`` override tags,
giving styled features (lyrics italics, raised positioning) a native target.
"""

from __future__ import annotations

import re
from pathlib import Path

from voxweave.realign import parse_vtt_blocks

# 1080p canvas: 22px outline text with generous margins matches the default
# rendering of mainstream players at this resolution.
_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,3,1.5,2,120,120,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_ITALIC_OPEN_RE = re.compile(r"<i>", re.IGNORECASE)
_ITALIC_CLOSE_RE = re.compile(r"</i>", re.IGNORECASE)
_OTHER_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def _srt_ts(seconds: float) -> str:
    """Seconds -> SRT timestamp ``HH:MM:SS,mmm``."""
    ms = round(max(0.0, seconds) * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_ts(seconds: float) -> str:
    """Seconds -> ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    cs = round(max(0.0, seconds) * 100)
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _timed_rows(
    blocks: list[dict],
) -> list[tuple[float, float, str]]:
    """Cue blocks -> (start, end, text) rows; raises when the VTT carries no
    timestamps at all (a plain-text edit draft -- run ``align`` first)."""
    rows = [
        (float(b["start"]), float(b["end"]), str(b["text"]))
        for b in blocks
        if b.get("start") is not None and b.get("end") is not None
    ]
    if not rows:
        raise ValueError(
            "VTT has no cue timestamps (plain-text edit draft?); run 'voxweave align' first"
        )
    return rows


def render_srt(rows: list[tuple[float, float, str]]) -> str:
    """Render numbered SRT cues. Inline ``<i>`` tags pass through."""
    out: list[str] = []
    for n, (start, end, text) in enumerate(rows, start=1):
        out.append(str(n))
        out.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        out.append(text)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _ass_text(text: str) -> str:
    """Cue text -> ASS event text: ``\\N`` line breaks, ``<i>`` to ``{\\i1}``
    overrides, other tags dropped, brace characters neutralized (ASS reads
    ``{...}`` as override blocks)."""
    t = text.replace("{", "(").replace("}", ")")
    t = _ITALIC_OPEN_RE.sub(r"{\\i1}", t)
    t = _ITALIC_CLOSE_RE.sub(r"{\\i0}", t)
    t = _OTHER_TAG_RE.sub("", t)
    return t.replace("\n", "\\N")


def render_ass(rows: list[tuple[float, float, str]]) -> str:
    """Render an ASS script with a single Default style. Lyric cues (wrapped in
    music notes by keep-lyrics mode) render italic per the Netflix convention."""
    events = []
    for start, end, text in rows:
        body = _ass_text(text)
        if text.startswith("♪") and text.endswith("♪"):
            body = f"{{\\i1}}{body}{{\\i0}}"
        events.append(
            f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},Default,,0,0,0,,{body}"
        )
    return _ASS_HEADER + "\n".join(events) + "\n"


_RENDERERS = {"srt": render_srt, "ass": render_ass}


def export_subtitles(vtt_path: Path, formats: tuple[str, ...]) -> list[Path]:
    """Render ``vtt_path`` into each requested format next to it; return the
    written paths. Unknown format names raise ValueError."""
    from voxweave.pipeline import swap_ext

    unknown = [f for f in formats if f not in _RENDERERS]
    if unknown:
        raise ValueError(f"unknown export format(s): {', '.join(unknown)}")
    rows = _timed_rows(parse_vtt_blocks(vtt_path.read_text(encoding="utf-8")))
    out: list[Path] = []
    for fmt in dict.fromkeys(formats):  # dedupe, keep order
        path = swap_ext(vtt_path, f".{fmt}")
        path.write_text(_RENDERERS[fmt](rows), encoding="utf-8")
        out.append(path)
    return out
