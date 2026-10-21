from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxweave.lang import to_iso_or
from voxweave.progress import Reporter

log = logging.getLogger("voxweave")


@dataclass(frozen=True)
class VoiceprintCapture:
    """Embedding evidence kept alive from diarization through episode commit."""

    centroids: dict[str, list[float]]
    provenance: dict[str, object]
    turns: list[tuple[float, float, str]]


def _prefetch_voiceprint_models(
    voiceprint_model: str | None,
    lang_override: str | None,
    reporter: Reporter,
) -> bool:
    """Fetch and verify the voiceprint checkpoint(s) before any audio work.

    The capture itself only runs after separation, ASR and diarization; a
    missing or stalled download discovered there would waste the whole run.
    ``auto`` needs both embedders unless ``--language`` already fixes the language
    (resolved to ISO exactly as :func:`transcribe` resolves it). ``False``
    (after a warning) means voiceprints are off for this run; the subtitles are
    still produced. An invalid model choice raises ``ValueError``.
    """
    from voxweave import voiceembed

    forced = lang_override.strip() if lang_override else ""
    language_iso = to_iso_or(forced, "en") if forced else None
    reporter.stage("voiceprint models")
    try:
        voiceembed.prefetch_checkpoints(voiceprint_model, language_iso)
    except voiceembed.VoiceEmbeddingError as exc:
        log.warning(
            "voiceprint models unavailable; continuing without voiceprint capture: %s",
            exc,
        )
        return False
    return True


def _decoupled_voiceprint_capture(
    wav: Path,
    diarization: Any,
    spec: Any,
    *,
    reporter: Reporter,
) -> VoiceprintCapture | None:
    """Voiceprints from a dedicated embedder over the audio pyannote heard.

    ``wav`` is the exact 16 kHz mono file the diarizer was fed, so the centroids
    describe the signal the provenance ``audio`` block records. The embedder is
    released before returning. A failure to fetch or run it drops the capture
    with a warning (the subtitles are the primary output); it never falls back
    to another embedding space.
    """
    from voxweave import voiceembed

    if not diarization.turns:
        return None
    reporter.stage(f"speaker voiceprints ({spec.name})")
    try:
        captured = voiceembed.capture_voiceprints(
            wav,
            diarization.turns,
            spec,
            diarization.provenance,
        )
    except (voiceembed.VoiceEmbeddingError, OSError) as exc:
        log.warning(
            "voiceprint capture unavailable; continuing without capture: %s", exc
        )
        return None
    finally:
        voiceembed.release()
    if captured is None:
        return None
    centroids, provenance = captured
    return VoiceprintCapture(
        centroids=centroids,
        provenance=provenance,
        turns=diarization.turns,
    )
