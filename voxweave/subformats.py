"""Load common subtitle formats (VTT/SRT/ASS/SSA) into the shared cue-block shape.

``realign.parse_vtt_blocks`` already reads SRT (the numeric index line and the
comma-decimal timing line are within its tolerance), so only ASS/SSA needs a
dedicated parser. All loaders return the realign block contract:
``[{text, start, end, lyric?}]``.
"""

from __future__ import annotations

import codecs
import logging
import re
from pathlib import Path

from voxweave.realign import _parse_ts, parse_vtt_blocks

log = logging.getLogger("voxweave")

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

_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
# \i1 / \i0 anywhere inside an override block, even packed with other tags
# ({\i1\fad(200,200)}); the (?![0-9]) guard keeps \i10 etc. from matching.
_ASS_ITALIC_TAG_RE = re.compile(r"\\i([01])(?![0-9])")
_ITALIC_WRAP_RE = re.compile(r"<i>(.*)</i>\Z", re.DOTALL)
_ITALIC_TOKEN_RE = re.compile(r"(<i>|</i>)")


def _override_to_italic(m: re.Match) -> str:
    """Override block -> its italic effect: the last \\i toggle wins; blocks
    without one are dropped."""
    tags = _ASS_ITALIC_TAG_RE.findall(m.group(0))
    if not tags:
        return ""
    return "<i>" if tags[-1] == "1" else "</i>"


def _balance_italics(t: str) -> str:
    """Drop dangling ``</i>`` and auto-close unclosed ``<i>`` (ASS italics run
    to end of line), so renderers never see a stray tag."""
    out: list[str] = []
    depth = 0
    for tok in _ITALIC_TOKEN_RE.split(t):
        if tok == "<i>":
            depth += 1
        elif tok == "</i>":
            if depth == 0:
                continue
            depth -= 1
        out.append(tok)
    out.append("</i>" * depth)
    return "".join(out)


def _ass_plain_text(raw: str) -> str:
    """ASS event text -> cue text: ``\\N``/``\\n`` become line breaks, ``\\h``
    a space, italic overrides become inline ``<i>``/``</i>`` tags (the form the
    SRT/ASS renderers understand) even when packed with other tags, all other
    override blocks are dropped, and stray italic tags are balanced.

    A whole-line italic wrap is removed entirely: it is styling (typically a
    song line), and keeping it would mask the music-note lyric detection below.
    """
    t = _ASS_OVERRIDE_RE.sub(_override_to_italic, raw.strip())
    m = _ITALIC_WRAP_RE.fullmatch(t)
    if m and "<i>" not in m.group(1) and "</i>" not in m.group(1):
        t = m.group(1)
    t = _balance_italics(t)
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
    music_only = 0
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
            log.warning(
                "malformed ASS Dialogue line dropped (%d fields, expected %d): %.60s",
                len(parts),
                len(names),
                line,
            )
            continue
        row = dict(zip(names, parts))
        start = _parse_ts(row.get("start", ""))
        end = _parse_ts(row.get("end", ""))
        if start is None or end is None:
            continue
        cue = _ass_plain_text(row.get("text", ""))
        block = _lyric_block(cue, start, end)
        if block is None:
            if cue:
                music_only += 1
            continue
        blocks.append(block)
    if music_only:
        log.info(
            "dropped %d music-only cue(s) (no text inside the note wrap)", music_only
        )
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


# BOM checks are decisive; UTF-32 first (its LE BOM starts with the UTF-16 LE
# BOM bytes). The utf-*-sig/utf-16/utf-32 codecs strip the BOM themselves.
_BOM_ENCODINGS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
# BOM-less non-UTF-8 fallbacks, tried in order. gb18030 goes first: it fails
# fast on most Western single-byte text while GBK-encoded CJK subtitles are the
# common wild case here; cp1252 catches the Western leftovers.
_FALLBACK_ENCODINGS = ("gb18030", "cp1252")


def read_subtitle_text(path: Path) -> str:
    """Read a subtitle file tolerating the encodings found in the wild: any BOM
    (UTF-8/16/32) decides outright, then strict UTF-8, then the fallback chain
    (logged, since the guess can be wrong). Raises RuntimeError with a
    convert-to-UTF-8 hint when nothing decodes."""
    p = Path(path)
    data = p.read_bytes()
    for bom, enc in _BOM_ENCODINGS:
        if data.startswith(bom):
            return data.decode(enc)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in _FALLBACK_ENCODINGS:
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        log.warning("%s is not UTF-8; decoded as %s", p.name, enc)
        return text
    raise RuntimeError(
        f"{p.name}: cannot determine text encoding"
        f" (tried utf-8, {', '.join(_FALLBACK_ENCODINGS)}); convert the file to UTF-8"
    )


def sniff_format(text: str) -> str | None:
    """Guess the parser family from the first non-empty line: ``"vtt"`` for a
    WEBVTT header, ``"ass"`` for an ASS/SSA section header ([Script Info],
    [V4* Styles], or a headerless [Events]); None when undecided (SRT has no
    magic, so it never sniffs)."""
    for line in text.lstrip("\ufeff").splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("webvtt"):
            return "vtt"
        if low.startswith(("[script info]", "[v4", "[events]")):
            return "ass"
        return None
    return None


def load_subtitle_blocks(path: Path) -> list[dict]:
    """Read and parse a subtitle file by extension -> cue blocks; raise when the
    format is unsupported, the content belongs to the other parser family (an
    ASS file renamed ``.vtt`` would otherwise parse into garbage cues), or the
    file yields no cues."""
    p = require_subtitle(path)
    text = read_subtitle_text(p)
    is_ass = p.suffix.lower() in (".ass", ".ssa")
    sniffed = sniff_format(text)
    if sniffed is not None and (sniffed == "ass") != is_ass:
        actual = "ASS/SSA" if sniffed == "ass" else "WebVTT"
        raise RuntimeError(
            f"{p.name}: content is {actual} but the extension says"
            f" {p.suffix.lower().lstrip('.')}; rename the file to its real format"
        )
    blocks = parse_ass_blocks(text) if is_ass else parse_vtt_blocks(text)
    if not blocks:
        raise RuntimeError(f"no cues in {p.name}")
    _sanitize_block_order(blocks, p.name)
    return blocks


def _sanitize_block_order(blocks: list[dict], name: str) -> None:
    """Swap inverted timestamps (logged) and sort timed cues by start, so
    VTT/SRT input behaves like the ASS parser (which already sorts) and
    downstream global alignment never sees a non-monotonic cue list."""
    for b in blocks:
        s, e = b.get("start"), b.get("end")
        if s is not None and e is not None and s > e:
            log.warning("%s: inverted timestamps (%.3f > %.3f), swapping", name, s, e)
            b["start"], b["end"] = e, s
    if all(b.get("start") is not None for b in blocks):
        blocks.sort(
            key=lambda b: (b["start"], b["end"] if b["end"] is not None else b["start"])
        )
