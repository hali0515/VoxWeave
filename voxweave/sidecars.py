"""Episode sidecar paths resolved from any reference to the episode.

Commands receive a media file, a subtitle (possibly language- or
``.sdh``/``.asrfix``-tagged) or a sibling JSON. :func:`artifact_owner` maps each
to the media that owns the episode's machine artifacts, and the helpers below
resolve the speaker/voiceprint/evidence sidecars through that owner, so every
command finds the same files. The per-owner locations themselves live in
:mod:`voxweave.artifacts`.
"""

from __future__ import annotations

import os
from pathlib import Path

from voxweave import artifacts
from voxweave.paths import (
    MEDIA_EXTS,
    detect_subtitle_language,
    find_subtitle_media,
    swap_ext,
)


def artifact_owner(reference: Path) -> Path:
    """Return live or cache-recorded media, else the standalone source itself."""
    value = Path(reference)
    if value.suffix.lower() in MEDIA_EXTS:
        return value
    live = find_subtitle_media(value)
    if live is not None:
        return live
    normalized_parent = Path(os.path.realpath(os.fspath(value.parent)))
    carrier = swap_ext(value, "")
    order = {suffix: index for index, suffix in enumerate(MEDIA_EXTS)}
    while True:
        recorded = sorted(
            (
                order[source.suffix.lower()],
                str(source),
                source,
            )
            for source in artifacts.claimed_sources(normalized_parent, carrier.name)
            if source.suffix.lower() in order
        )
        if recorded:
            return recorded[0][2]
        if not carrier.suffix:
            break
        tag = carrier.suffix[1:].casefold()
        if tag not in {"asrfix", "sdh"} and detect_subtitle_language(carrier) is None:
            break
        carrier = swap_ext(carrier, "")
    return value


def voiceprints_path(path: Path) -> Path:
    return artifacts.voiceprints_path(artifact_owner(Path(path)))


def speakers_suggest_path(path: Path) -> Path:
    return artifacts.speaker_suggest_path(artifact_owner(Path(path)))


def speakers_mapping_path(path: Path, *, reference: Path | None = None) -> Path:
    return artifacts.speaker_mapping_path(
        artifact_owner(Path(path)),
        reference=reference,
    )


def inspect_speakers_mapping_path(
    path: Path,
    *,
    reference: Path | None = None,
) -> Path:
    owner = artifact_owner(Path(path))
    if reference is not None and owner == Path(path):
        ref = Path(reference)
        exact = swap_ext(ref, ".speakers.json")
        if artifacts.path_present(exact):
            return exact
        if detect_subtitle_language(ref) is not None:
            untagged = swap_ext(swap_ext(ref, ""), ".speakers.json")
            if artifacts.path_present(untagged):
                return untagged
    return artifacts.inspect_speaker_mapping_path(
        owner,
        reference=reference,
    )


def voiceprints_candidates(path: Path) -> tuple[Path, ...]:
    return artifacts.fixed_candidates(
        artifact_owner(Path(path)),
        ".voiceprints.json",
        "voiceprints",
    )


def speaker_suggest_candidates(path: Path) -> tuple[Path, ...]:
    return artifacts.fixed_candidates(
        artifact_owner(Path(path)),
        ".speakers.suggest.json",
        "speaker_suggest",
    )


def align_evidence_candidates(
    subtitle: Path,
    *,
    media: Path | None = None,
) -> tuple[Path, ...]:
    owner = Path(media) if media is not None else artifact_owner(Path(subtitle))
    return artifacts.align_evidence_candidates(owner, Path(subtitle))


def speakers_html_path(path: Path) -> Path:
    return swap_ext(Path(path), ".speakers.html")
