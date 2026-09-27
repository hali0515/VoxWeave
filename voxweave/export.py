"""Export subtitles between formats (VTT/SRT/ASS in, VTT/SRT/ASS out).

The VTT + JSON pair stays the source of truth; export renders presentation
formats from it. SRT is a plain re-rendering (inline ``<i>`` tags and the
``{\\an8}`` position tag pass through -- mainstream players honor them). ASS
carries a Default dialogue style, translates ``<i>``/``</i>`` into
``{\\i1}``/``{\\i0}`` override tags and keeps ``{\\anN}``/``{\\aN}`` position
tags as real overrides, giving styled features (lyrics italics, raised
positioning) a native target. WebVTT output drops position tags (it has no
inline form for them).
Foreign SRT/ASS files can also be exported to VTT to enter the voxweave
editing workflow (they carry no word-level JSON, so align works from scratch).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voxweave import fsio
from voxweave.realign import fmt_ts, render_cues
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
# The position tag mainstream SRT players honor ({\an8} = top center), plus the
# legacy SSA numbering ({\a6}: 1-3 bottom, 5-7 top, 9-11 middle).
_POSITION_TAG_RE = re.compile(r"\{\\(an[1-9]|a(?:1[01]|[1235679]))\}")


def _srt_ts(seconds: float) -> str:
    """Seconds -> SRT timestamp ``HH:MM:SS,mmm``.

    Identical to the VTT timestamp (including its round-to-whole-milliseconds
    fix) apart from the fractional separator, so reuse realign.fmt_ts instead of
    carrying a second copy of the divmod chain.
    """
    return fmt_ts(seconds).replace(".", ",", 1)


def _ass_ts(seconds: float) -> str:
    """Seconds -> ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    cs = round(max(0.0, seconds) * 100)
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def require_timed_blocks(blocks: Sequence[Mapping[str, Any]], name: str) -> None:
    """Raise ValueError unless every cue of the file ``name`` carries timestamps.

    A file without any is a plain-text edit draft. A partly timed one is refused
    too, as translate refuses it for SRT/ASS: every timed output (SRT/ASS
    export, burn, pack) would silently lose its untimed cues. The message names
    the first untimed cues (1-based, in file order) and the fix."""
    untimed = [
        (n, b)
        for n, b in enumerate(blocks, start=1)
        if b.get("start") is None or b.get("end") is None
    ]
    if not untimed:
        return
    if name.lower().endswith(".vtt"):
        remedy = f"run 'voxweave align {name}' first"
    else:
        remedy = "add their timing lines first"
    if len(untimed) == len(blocks):
        raise ValueError(
            f"{name}: no cue timestamps found (plain-text edit draft?); {remedy}"
        )
    shown = ", ".join(str(n) for n, _b in untimed[:5])
    if len(untimed) > 5:
        shown += f", ... {len(untimed)} in all"
    first = " ".join(str(untimed[0][1].get("text", "")).split())
    if len(first) > 40:
        first = first[:37] + "..."
    raise ValueError(
        f"{name} has cues without timestamps (cue {shown}; first: {first!r}); {remedy}"
    )


def _timed_rows(
    blocks: list[dict],
    *,
    name: str,
) -> list[tuple[float, float, str]]:
    """Cue blocks of the file ``name`` -> (start, end, text) rows; raises unless
    every cue is timed (see :func:`require_timed_blocks`).

    Lyric-flagged blocks (parsers strip the music-note wrap into the flag) get
    their display wrap restored here so renderers see the on-screen text."""
    require_timed_blocks(blocks, name)
    return [
        (
            float(b["start"]),
            float(b["end"]),
            f"♪ {b['text']} ♪" if b.get("lyric") else str(b["text"]),
        )
        for b in blocks
    ]


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


def _split_position(text: str) -> tuple[str | None, str]:
    """Cue text -> (its position override such as ``"an8"`` or None, the text
    without position tags). Only the first tag counts, as in libass."""
    tags = _POSITION_TAG_RE.findall(text)
    return (tags[0] if tags else None), _POSITION_TAG_RE.sub("", text)


def render_srt(
    rows: list[tuple[float, float, str]],
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render numbered SRT cues, including speaker-name prefixes when supplied.
    A ``{\\an8}``-style position tag is kept, at the start of the cue."""
    out: list[str] = []
    for n, (start, end, text) in enumerate(rows, start=1):
        out.append(str(n))
        out.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        block = blocks[n - 1] if blocks is not None and n - 1 < len(blocks) else None
        position, text = _split_position(text)
        body = _srt_speaker_text(text, block)
        out.append(f"{{\\{position}}}{body}" if position else body)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _ass_text(text: str) -> str:
    """Cue text -> ASS event text: ``\\N`` line breaks, ``<i>`` to ``{\\i1}``
    overrides, a ``{\\an8}``/``{\\a6}`` position tag to a leading override of
    its own, other tags dropped, brace characters neutralized (ASS reads
    ``{...}`` as override blocks)."""
    position, text = _split_position(text)
    t = text.replace("{", "(").replace("}", ")")
    t = _ITALIC_OPEN_RE.sub(r"{\\i1}", t)
    t = _ITALIC_CLOSE_RE.sub(r"{\\i0}", t)
    t = _OTHER_TAG_RE.sub("", t)
    t = t.replace("\n", "\\N")
    return f"{{\\{position}}}{t}" if position else t


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
                name = " / ".join(
                    dict.fromkeys(name for name, _line in speakers if name)
                )
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
    """Render timed rows as WebVTT, restoring any speaker voice tags. SRT-style
    position tags are dropped: WebVTT has no inline form for them and the cue
    writer emits no cue settings."""
    rows = [(s, e, _POSITION_TAG_RE.sub("", t)) for s, e, t in rows]
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
        example = "vtt" if src_fmt == "srt" else "srt"
        raise ValueError(
            f"{sub_path.name} is already .{src_fmt}; "
            f"pick another --format (e.g. -f {example})"
        )
    blocks = load_subtitle_blocks(sub_path)
    rows = _timed_rows(blocks, name=sub_path.name)  # raises unless all are timed
    out: list[Path] = []
    for fmt in dict.fromkeys(formats):  # dedupe, keep order
        path = swap_ext(sub_path, f".{fmt}")
        if fmt == "srt":
            rendered = render_srt(rows, blocks=blocks)
        elif fmt == "ass":
            rendered = render_ass(rows, blocks=blocks)
        else:
            rendered = render_vtt_rows(rows, blocks=blocks)
        fsio.atomic_write_text(path, rendered)
        out.append(path)
    return out
