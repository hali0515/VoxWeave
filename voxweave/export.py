"""Export subtitles between formats (VTT/SRT/ASS in, VTT/SRT/ASS out).

The VTT + JSON pair stays the source of truth; export renders presentation
formats from it. SRT is a plain re-rendering (inline ``<i>`` tags pass through
unchanged -- mainstream players honor them). ASS carries a Default dialogue
style and translates ``<i>``/``</i>`` into ``{\\i1}``/``{\\i0}`` override tags,
giving styled features (lyrics italics, raised positioning) a native target.
Foreign SRT/ASS files can also be exported to VTT to enter the voxweave
editing workflow (they carry no word-level JSON, so align works from scratch).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voxweave import fsio
from voxweave.realign import render_cues
from voxweave.speakers import (
    sanitize_ass_speaker_name,
    speaker_layout,
    speaker_metadata,
    voice_text_for_block,
)


def ass_header(
    *,
    width: int = 1920,
    height: int = 1080,
    font: str = "Arial",
    font_size: int | None = None,
) -> str:
    """ASS script header with a single Default style sized for the given canvas.

    All metric values (font size, outline, shadow, margins) are tuned for a 1080p
    canvas and scale linearly with the actual height, so burning onto e.g. a 2160p
    frame keeps the same visual proportions.
    """
    # ASS "Style:" lines are comma-delimited; a comma in the font name would
    # shift every field after Fontname, so strip commas from it.
    font = font.replace(",", "")
    scale = height / 1080
    size = font_size if font_size is not None else round(72 * scale)
    outline = round(3 * scale, 1)
    shadow = round(1.5 * scale, 1)
    margin_lr = round(120 * scale)
    margin_v = round(60 * scale)
    return f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# 1080p canvas: outline text with generous margins matches the default
# rendering of mainstream players at this resolution.
_ASS_HEADER = ass_header()

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
    """Cue blocks -> (start, end, text) rows; raises when the file carries no
    timestamps at all (a plain-text edit draft -- run ``align`` first).

    Lyric-flagged blocks (parsers strip the music-note wrap into the flag) get
    their display wrap restored here so renderers see the on-screen text."""
    rows = [
        (
            float(b["start"]),
            float(b["end"]),
            f"♪ {b['text']} ♪" if b.get("lyric") else str(b["text"]),
        )
        for b in blocks
        if b.get("start") is not None and b.get("end") is not None
    ]
    if not rows:
        raise ValueError(
            "no cue timestamps found (plain-text edit draft?); run 'voxweave align' first"
        )
    return rows


def _srt_speaker_text(text: str, block: Mapping[str, Any] | None) -> str:
    """Apply ``NAME: `` prefixes without changing the underlying cue text."""
    if block is None:
        return text
    speaker, speakers = speaker_metadata(block)
    speaker, speakers = speaker_layout(text, speaker=speaker, speakers=speakers)
    lines = text.split("\n")
    if speakers is not None:
        return "\n".join(
            f"{name}: {line}" if name else line for name, line in zip(speakers, lines)
        )
    return f"{speaker}: {text}" if speaker else text


def render_srt(
    rows: list[tuple[float, float, str]],
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render numbered SRT cues, including speaker-name prefixes when supplied."""
    out: list[str] = []
    for n, (start, end, text) in enumerate(rows, start=1):
        out.append(str(n))
        out.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        block = blocks[n - 1] if blocks is not None and n - 1 < len(blocks) else None
        out.append(_srt_speaker_text(text, block))
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


def render_ass(
    rows: list[tuple[float, float, str]],
    *,
    header: str | None = None,
    blocks: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render an ASS script with a single Default style. Lyric cues (wrapped in
    music notes by keep-lyrics mode) render italic per the Netflix convention.

    ``header`` overrides the default 1080p script header (see :func:`ass_header`);
    the burn path passes one sized to the actual video frame.
    """
    events = []
    for index, (start, end, text) in enumerate(rows):
        body = _ass_text(text)
        if text.startswith("♪") and text.endswith("♪"):
            body = f"{{\\i1}}{body}{{\\i0}}"
        name = ""
        if blocks is not None and index < len(blocks):
            speaker, speakers = speaker_metadata(blocks[index])
            if speaker:
                name = speaker
            elif speakers:
                # ASS has one Name field per Dialogue event.  Preserve the dual
                # cue as one on-screen event and retain both actor names here.
                name = " / ".join(dict.fromkeys(n for n in speakers if n))
        name = sanitize_ass_speaker_name(name)
        events.append(
            f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},Default,{name},0,0,0,,{body}"
        )
    return (header if header is not None else _ASS_HEADER) + "\n".join(events) + "\n"


def render_vtt_rows(
    rows: list[tuple[float, float, str]],
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render timed rows as WebVTT, restoring any speaker voice tags."""
    if blocks is None:
        return render_cues([(s, e, t) for s, e, t in rows])
    return render_cues(
        [
            (
                start,
                end,
                voice_text_for_block(text, blocks[index])
                if index < len(blocks)
                else text,
            )
            for index, (start, end, text) in enumerate(rows)
        ]
    )


_EXPORT_FORMATS = {"srt", "ass", "vtt"}


def export_subtitles(sub_path: Path, formats: tuple[str, ...]) -> list[Path]:
    """Render ``sub_path`` (VTT/SRT/ASS/SSA) into each requested format next to
    it; return the written paths. Unknown format names and a target format equal
    to the source raise ValueError."""
    from voxweave.pipeline import swap_ext
    from voxweave.subformats import load_subtitle_blocks

    unknown = [f for f in formats if f not in _EXPORT_FORMATS]
    if unknown:
        raise ValueError(f"unknown export format(s): {', '.join(unknown)}")
    src_fmt = sub_path.suffix.lower().lstrip(".")
    if src_fmt in formats:
        raise ValueError(f"{sub_path.name} is already .{src_fmt}; pick another --to")
    blocks = load_subtitle_blocks(sub_path)
    timed_blocks = [
        block
        for block in blocks
        if block.get("start") is not None and block.get("end") is not None
    ]
    rows = _timed_rows(blocks)
    out: list[Path] = []
    for fmt in dict.fromkeys(formats):  # dedupe, keep order
        path = swap_ext(sub_path, f".{fmt}")
        if fmt == "srt":
            rendered = render_srt(rows, blocks=timed_blocks)
        elif fmt == "ass":
            rendered = render_ass(rows, blocks=timed_blocks)
        else:
            rendered = render_vtt_rows(rows, blocks=timed_blocks)
        fsio.atomic_write_text(path, rendered)
        out.append(path)
    return out
