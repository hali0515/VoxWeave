"""Sibling-path primitives: dot-safe extension swaps and media/subtitle lookup.

A leaf module: it imports nothing from voxweave beyond ``lang``, so every layer
(artifacts, subtitle I/O, mux, speakers, the pipeline) can share one
implementation instead of reaching up into the orchestration module.
"""

from __future__ import annotations

import logging
from pathlib import Path

from voxweave import lang

log = logging.getLogger("voxweave")

# Extensions tried when locating the source media by stem (align only receives the VTT).
MEDIA_EXTS = (
    ".mkv",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".ts",
    ".m4v",
    ".flac",
    ".wav",
    ".m4a",
    ".mp3",
    ".aac",
    ".opus",
    ".ogg",
)

# Subtitle formats the file-based commands (export/translate/pack/burn) accept.
SUBTITLE_EXTS = (".vtt", ".srt", ".ass", ".ssa")


def swap_ext(path: Path, new_ext: str) -> Path:
    """Replace the trailing extension of path with new_ext (include leading dot; "" removes it).

    Do NOT use ``Path.with_suffix`` for sibling paths: filenames with interior dots
    (e.g. YouTube titles containing ``...``) cause with_suffix to misidentify the first
    interior dot as the suffix, silently truncating the name. This function only replaces
    ``path.suffix``, leaving interior dots untouched.
    """
    if path.suffix:
        return path.with_name(path.name[: -len(path.suffix)] + new_ext)
    return path.with_name(path.name + new_ext)


def detect_subtitle_language(sub: Path) -> str | None:
    """ISO code from a language-tagged subtitle filename ("X.zh.vtt" -> "zh"), else None."""
    p = Path(sub)
    stem = p.name
    if p.suffix.lower() in SUBTITLE_EXTS:
        stem = stem[: -len(p.suffix)]
    if "." not in stem:
        return None
    return lang.to_iso_or(stem.rsplit(".", 1)[1], None)


def find_sibling_media(ref: Path) -> Path | None:
    """Find the source media alongside ref by stem, matching extensions case-insensitively.

    Returns the best candidate by ``MEDIA_EXTS`` order (``.mkv`` before ``.mp4`` ...);
    when more than one media sibling exists the ambiguity is logged and the selection
    stays deterministic. ``None`` when no sibling is found.
    """
    ref = Path(ref)
    base = swap_ext(ref, "").name  # stem without the reference extension (dot-safe)
    order = {ext: i for i, ext in enumerate(MEDIA_EXTS)}
    matches: list[tuple[int, Path]] = []
    parent = ref.parent if str(ref.parent) else Path(".")
    if parent.exists():
        for p in parent.iterdir():
            if p.is_file() and swap_ext(p, "").name == base:
                rank = order.get(p.suffix.lower())
                if rank is not None:
                    matches.append((rank, p))
    if not matches:
        return None
    matches.sort(key=lambda m: (m[0], str(m[1])))
    if len(matches) > 1:
        log.warning(
            "multiple sibling media files for %s: %s; using %s",
            ref.name,
            [m[1].name for m in matches],
            matches[0][1].name,
        )
    return matches[0][1]


def find_subtitle_media(ref: Path) -> Path | None:
    """Find exact-stem media, then peel closed subtitle-derived suffix tags."""
    found = find_sibling_media(ref)
    if found is not None:
        return found
    carrier = swap_ext(ref, "")
    while carrier.suffix:
        tag = carrier.suffix[1:].casefold()
        if tag not in {"asrfix", "sdh"} and detect_subtitle_language(carrier) is None:
            return None
        found = find_sibling_media(carrier)
        if found is not None:
            return found
        carrier = swap_ext(carrier, "")
    return None
