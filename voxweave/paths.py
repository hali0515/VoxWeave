"""Sibling-path primitives: dot-safe extension swaps and media/subtitle lookup.

A leaf module: it imports nothing from voxweave beyond ``lang``, so every layer
(artifacts, subtitle I/O, mux, speakers, the pipeline) can share one
implementation instead of reaching up into the orchestration module.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

from voxweave import lang

log = logging.getLogger("voxweave")

# Bytes per file name when the filesystem cannot say (every filesystem voxweave
# targets allows at least this).
DEFAULT_NAME_MAX = 255

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


def name_limit(directory: Path) -> int:
    """Longest file name, in bytes, the filesystem holding ``directory`` accepts.

    ``directory`` may not exist yet: its nearest existing ancestor answers for
    it, since that is where it would be created. Falls back to 255.
    """
    for candidate in (Path(directory), *Path(directory).parents):
        try:
            return int(os.pathconf(candidate, "PC_NAME_MAX"))
        except FileNotFoundError:
            continue
        except (AttributeError, OSError, ValueError):
            break
    return DEFAULT_NAME_MAX


def name_bytes(name: str) -> int:
    """Length of ``name`` as the filesystem stores it (UTF-8 bytes)."""
    return len(os.fsencode(name))


class OutputNameTooLongError(OSError):
    """A deliverable's file name exceeds its filesystem's name limit."""

    def __init__(self, path: Path, size: int, limit: int) -> None:
        super().__init__(errno.ENAMETOOLONG, os.strerror(errno.ENAMETOOLONG), str(path))
        self.size = size
        self.limit = limit

    def __str__(self) -> str:
        return (
            f"cannot write {self.filename}: its file name is {self.size} bytes and "
            f"this filesystem allows at most {self.limit}; give the input a "
            "shorter name (or pass a shorter output path) and run again"
        )


def name_fits(path: Path) -> bool:
    """Whether ``path``'s file name is short enough to exist on its filesystem."""
    path = Path(path)
    return name_bytes(path.name) <= name_limit(path.parent)


def require_output_name(path: Path) -> Path:
    """Return ``path``, or raise if its file name cannot exist on its filesystem.

    Deliverables beside the media (``<stem>.json``, ``.vtt``, derived and
    translated subtitles, pack/burn outputs) keep their user-facing names and
    are never shortened, so a name that is too long is refused up front, before
    a command does any work, instead of failing at the final write.
    """
    path = Path(path)
    size = name_bytes(path.name)
    limit = name_limit(path.parent)
    if size > limit:
        raise OutputNameTooLongError(path, size, limit)
    return path


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
