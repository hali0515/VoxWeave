"""The 16 kHz speech input for ASR, diarization and alignment, and its vocals cache.

With separation on, the input comes from the separated vocals: Roformer runs on
the full-band 44.1 kHz stereo decode, the stem is resampled to 32 kHz mono (the
PANNs input and what the vocals cache stores) and the 16 kHz input is decoded
from that. :func:`acquire_16k` is the one flow that reuses the cache or
separates and writes it back, shared by ``transcribe`` and ``align``.

The strict companion format and the cache lock live in
:mod:`voxweave.vocalscache`; this module adds the separator and ffmpeg side on
top of them.
"""

from __future__ import annotations

import logging
import os
import stat
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from voxweave import artifacts, backend, config
from voxweave.chunking import decode_to_wav
from voxweave.progress import Reporter, progress_bridge
from voxweave.vocalscache import (
    cache_companion_path,
    cache_lock,
    cache_publish_path,
    cache_write_window,
    classify_cache_decode_failure,
    load_cache_companion,
    publish_cache_companion,
    validate_cache_pair,
)
from voxweave.voicebase import Phase2DataError

log = logging.getLogger("voxweave")

# Loudness normalization applied only to the 16k VAD/ASR path; 44.1k separation path is untouched.
ASR_LOUDNORM = os.environ.get("VOXWEAVE_LOUDNORM", "loudnorm=I=-16:TP=-1.5:LRA=11")
# PANNs Cnn14 is trained at 32k.
SONGDET_SR = 32000

# Vocals cache: per-media artifact claim ``vocals.32k.flac`` (32k mono, no BGM).
# Shared by process and align; PANNs eats it directly, ASR/alignment downsample to 16k.
# Existing media-adjacent 32k/16k caches remain writeback/read compatibility lanes.
# A cache hit also requires the durations to match (_vocals_cache_fresh): a replaced or
# trimmed source silently invalidates the old separation, so it is re-run and overwritten.
CACHE_DIRNAME = "cache"
# Max |cache - media| duration drift still treated as the same source. Covers
# container-vs-stream duration jitter; real source edits move duration by seconds.
CACHE_DUR_TOL_SEC = config._env_float("VOXWEAVE_CACHE_DUR_TOL_SEC", 0.5)


def cache_vocals_path(media_path: Path) -> Path:
    """Resolve the vocals cache, preserving an existing legacy cache pair."""
    media_path = Path(media_path)
    legacy = media_path.parent / CACHE_DIRNAME / f"{media_path.stem}.vocals.32k.flac"
    if (
        artifacts.path_present(legacy)
        or artifacts.path_present(cache_companion_path(legacy))
        or artifacts.path_present(Path(f"{legacy.resolve()}.lock"))
    ):
        return legacy
    managed = artifacts.claim_paths(media_path).vocals_cache
    for candidate in (
        managed,
        Path(f"{managed}.meta.json"),
        Path(f"{managed}.lock"),
    ):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise artifacts.ArtifactMarkerError(
                f"cannot inspect managed vocals cache node {candidate}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise artifacts.ArtifactMarkerError(
                f"managed vocals cache node is not a regular file: {candidate}"
            )
    return managed


def cache_16k_path(media_path: Path) -> Path:
    """Return the legacy 16k vocals cache path: <media_dir>/cache/<stem>.16k.flac (read-only backward compat)."""
    media_path = Path(media_path)
    return media_path.parent / CACHE_DIRNAME / f"{media_path.stem}.16k.flac"


class _ProbeUnavailable(RuntimeError):
    """ffprobe itself could not answer (not installed, or timed out): says nothing
    about whether the probed file is readable."""


def _probe_duration(path: Path) -> float | None:
    """Media duration in seconds via ffprobe, or None if unreadable.

    Raises :class:`_ProbeUnavailable` when ffprobe is missing from PATH or times
    out, so callers do not mistake a tool problem for an unreadable file.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as e:
        raise _ProbeUnavailable("ffprobe not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise _ProbeUnavailable(f"ffprobe timed out on {path.name}") from e
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _vocals_cache_fresh(cache: Path, media: Path) -> bool:
    """True if the cached vocals still match the source media duration.

    An unreadable cache is stale (truncated/corrupt flac must not be trusted);
    an unprobeable media keeps the cache hit — decoding will surface the real
    error later, and burning a separation pass on a maybe-valid cache helps nobody.
    When ffprobe itself is unavailable (missing or timed out) the cache cannot be
    validated, so it is re-separated too, but the warning names the real cause.
    """
    try:
        cache_dur = _probe_duration(cache)
    except _ProbeUnavailable as e:
        log.warning("%s; cannot validate the vocals cache, re-separating: %s", e, cache)
        return False
    if cache_dur is None:
        log.warning("vocals cache unreadable, re-separating: %s", cache)
        return False
    try:
        media_dur = _probe_duration(media)
    except _ProbeUnavailable as e:
        log.warning("%s; keeping the vocals cache unvalidated: %s", e, cache)
        return True
    if media_dur is None:
        return True
    if abs(cache_dur - media_dur) <= CACHE_DUR_TOL_SEC:
        return True
    log.warning(
        "vocals cache stale (cache %.2fs vs media %.2fs), re-separating: %s",
        cache_dur,
        media_dur,
        cache,
    )
    return False


def _encode_flac(src_wav: Path, dst_flac: Path) -> None:
    """Encode wav to flac for caching (lossless); caller treats failure as non-fatal.

    Atomic: an interrupted encode must not leave a truncated flac at the cache
    path — every later run would treat it as a cache hit and fail obscurely.
    """
    dst_flac.parent.mkdir(parents=True, exist_ok=True)
    with cache_publish_path(dst_flac) as tmp:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(src_wav), "-c:a", "flac", str(tmp)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _vocals_to_16k(voc32: Path, *, normalize: bool) -> Path:
    """Decode 32 kHz mono vocals to the 16 kHz ASR/diarization input (a temp wav).

    ``voc32`` is either a first run's fresh 32k wav or the ``vocals.32k.flac``
    the vocals cache stores (a lossless copy of it). Every such decode (a first
    run, and a cache hit in ``transcribe`` or ``align``) goes through here, so a
    re-run feeds ASR and diarization exactly the samples its first run did:
    loudnorm measures its input, and a different source, rate or filter would
    move the speaker turns between runs.
    """
    return decode_to_wav(voc32, audio_filter=ASR_LOUDNORM if normalize else None)


@overload
def _separate_to_16k_32k(
    media: Path,
    *,
    reporter: Reporter,
    normalize: bool,
    return_separator_identity: Literal[False] = False,
) -> tuple[Path, Path, Path, Path]: ...


@overload
def _separate_to_16k_32k(
    media: Path,
    *,
    reporter: Reporter,
    normalize: bool,
    return_separator_identity: Literal[True],
) -> tuple[Path, Path, Path, Path, dict[str, object]]: ...


def _separate_to_16k_32k(
    media: Path,
    *,
    reporter: Reporter,
    normalize: bool,
    return_separator_identity: bool = False,
) -> tuple[Path, Path, Path, Path] | tuple[Path, Path, Path, Path, dict[str, object]]:
    """Decode full-band 44.1k stereo -> Roformer separate -> resample, returning
    ``(fullband, vocals, wav_16k, voc32_32k)`` and, when requested, the
    load-bound separator identity as a fifth item.

    The full-band 44.1k stereo feed is a hard constraint (Roformer is trained at 44.1k);
    downsampling to 16k/32k happens only after separation. Callers own temp bookkeeping,
    debug dumps, and caching of the returned paths.

    The 16k ASR/diarization input is derived from the 32k mono vocals, never from the
    44.1k stereo stem: ``voc32`` is exactly what the vocals cache stores, so a first run
    and a later cache hit (which decodes ``vocals.32k.flac`` with the same filter) feed
    loudnorm the same samples. Normalizing the stereo stem instead measured 2.5-2.6 dB
    louder and moved ~10-14% of diarization frames between the two runs.

    On a clean return the caller registers the paths in its own ``tmp`` list (cleaned in its
    ``finally``). Since that registration only runs after this returns, the helper self-cleans
    its partial outputs if a later step raises (or is interrupted) — otherwise an OOM/ffmpeg
    failure or a Ctrl-C mid-separation would orphan the already-decoded temp files.
    """
    created: list[Path] = []
    try:
        reporter.stage("decode fullband 44.1k")
        fullband = decode_to_wav(media, sample_rate=44100, mono=False)
        created.append(fullband)
        reporter.stage("vocal separation (Roformer)")
        if return_separator_identity:
            vocals, separator_identity = backend.separate_vocals(
                fullband,
                progress=progress_bridge(reporter, "vocal separation (Roformer)"),
                return_identity=True,
            )
        else:
            vocals = backend.separate_vocals(
                fullband,
                progress=progress_bridge(reporter, "vocal separation (Roformer)"),
            )
        created.append(vocals)
        reporter.stage("resample 16k")
        voc32 = decode_to_wav(
            vocals, sample_rate=SONGDET_SR
        )  # 32k mono: PANNs + cache source
        created.append(voc32)
        # Same source and filter as a vocals-cache hit (32k mono -> 16k mono).
        wav = _vocals_to_16k(voc32, normalize=normalize)
        created.append(wav)
        if return_separator_identity:
            return fullband, vocals, wav, voc32, separator_identity
        return fullband, vocals, wav, voc32
    except BaseException:
        # BaseException: a Ctrl-C mid-separation must not orphan the multi-hundred-MB
        # full-band WAV either.
        for p in created:
            p.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class Acquired16k:
    """What :func:`acquire_16k` produced. Every temp file is already in ``tmp``."""

    #: The 16 kHz mono ASR/diarization input (a temp wav).
    wav: Path
    #: 32 kHz mono vocals, the PANNs input: a fresh temp wav, or on a cache hit the
    #: cache file itself (not a temp). ``None`` without separation or on a legacy
    #: 16k cache hit.
    voc32: Path | None = None
    #: The full-band 44.1 kHz stereo decode (the original-mix VAD reference);
    #: only when separation ran.
    fullband: Path | None = None
    #: The separator identity the vocals are bound to; only on the bound lane.
    separator: dict[str, object] | None = None


def acquire_16k(
    media: Path,
    *,
    separate: bool,
    normalize: bool,
    reporter: Reporter,
    tmp: list[Path],
    cache: Path | None = None,
    owner: Path | None = None,
    fingerprint: str | None = None,
    separator: Mapping[str, object] | None = None,
    legacy_16k: bool = False,
    purpose: str,
    on_separated: Callable[[Path, Path], None] | None = None,
) -> Acquired16k:
    """Produce the 16 kHz input for ``media``, reusing or refreshing the vocals cache.

    Without ``separate`` the media itself is decoded to 16 kHz. With it, the
    vocals ``cache`` (``None`` = no cache) is reused when it is still valid and
    otherwise ``media`` is separated and the cache rewritten (a failed write is
    only a warning). ``tmp`` receives every temp file as soon as it exists.

    Validity has two lanes. Unbound (``fingerprint`` is ``None``): the cache
    duration must match ``owner``'s (default ``media``), and with
    ``legacy_16k`` an old media-adjacent ``<stem>.16k.flac`` is still read
    before separating. Bound: only a companion recording exactly this media
    ``fingerprint`` and ``separator`` establishes a hit (``separator=None``
    resolves the configured one, and only once a cache is there to validate),
    and a rewritten cache gets its companion. ``purpose`` names the bound
    consumer in the refusal log. ``on_separated(fullband, vocals)`` runs after a
    fresh separation, before the cache write.
    """
    media = Path(media)
    if not separate:
        reporter.stage("decode 16k")
        wav = decode_to_wav(media, audio_filter=ASR_LOUDNORM if normalize else None)
        tmp.append(wav)
        return Acquired16k(wav)
    reference = Path(owner) if owner is not None else media
    bound = fingerprint is not None
    if cache is not None:
        with cache_lock(Path(cache)) as cache_handle:
            cache_path = cache_handle.cache_path
            cache_hit = False
            bound_separator: dict[str, object] | None = None
            if cache_path.exists():
                if bound:
                    try:
                        companion, _validated = load_cache_companion(
                            cache_handle.companion_path
                        )
                        current_separator = (
                            separator
                            if separator is not None
                            else backend.separator_identity()
                        )
                        validated = validate_cache_pair(
                            companion,
                            cache_path,
                            media_fingerprint=fingerprint or "",
                            separator=current_separator,
                        )
                        bound_separator = validated.separator.as_mapping()
                        cache_hit = True
                    except (OSError, Phase2DataError):
                        log.info(
                            "vocals cache is not bound to this %s; re-separating: %s",
                            purpose,
                            cache_path,
                        )
                else:
                    cache_hit = _vocals_cache_fresh(cache_path, reference)
            if cache_hit:
                # Keep the cache lock through decoder completion. A validated
                # hash followed by an unlocked open is not a stable read.
                reporter.stage("vocals cache (32k)")
                log.info("reuse cached vocals %s", cache_path)
                try:
                    wav = _vocals_to_16k(cache_path, normalize=normalize)
                except BaseException as exc:
                    classify_cache_decode_failure(exc)
                    raise
                tmp.append(wav)
                # Cache hit: skip Roformer; PANNs eats 32k directly, ASR downsamples to 16k.
                return Acquired16k(wav, voc32=cache_path, separator=bound_separator)
    if legacy_16k and not bound:
        legacy = cache_16k_path(reference)
        if any(
            artifacts.path_present(candidate)
            for candidate in (
                legacy,
                cache_companion_path(legacy),
                Path(f"{legacy.resolve()}.lock"),
            )
        ):
            with cache_lock(legacy) as legacy_handle:
                if legacy_handle.cache_path.exists() and _vocals_cache_fresh(
                    legacy_handle.cache_path,
                    reference,
                ):
                    reporter.stage("vocals cache (16k legacy)")
                    log.info("reuse legacy 16k vocals %s", legacy_handle.cache_path)
                    try:
                        wav = decode_to_wav(
                            legacy_handle.cache_path,
                            audio_filter=ASR_LOUDNORM if normalize else None,
                        )
                    except BaseException as exc:
                        classify_cache_decode_failure(exc)
                        raise
                    tmp.append(wav)
                    return Acquired16k(wav)
    separated_by: dict[str, object] | None = None
    if bound:
        fullband, vocals, wav, voc32, separated_by = _separate_to_16k_32k(
            media,
            reporter=reporter,
            normalize=normalize,
            return_separator_identity=True,
        )
    else:
        fullband, vocals, wav, voc32 = _separate_to_16k_32k(
            media, reporter=reporter, normalize=normalize
        )
    # Own every temp before anything else can raise (a debug dump onto a full
    # disk would otherwise leave them in /tmp).
    tmp.extend((fullband, vocals, voc32, wav))
    if on_separated is not None:
        on_separated(fullband, vocals)
    if cache is not None:
        try:
            with cache_write_window(Path(cache)) as cache_handle:
                _encode_flac(voc32, cache_handle.cache_path)
                if bound:
                    # Without a companion no later bound run could reuse these vocals.
                    publish_cache_companion(
                        cache_handle.cache_path,
                        media_fingerprint=fingerprint or "",
                        separator=separated_by or {},
                        companion_path=cache_handle.companion_path,
                    )
            log.info("cached vocals 32k → %s", cache)
        except (OSError, subprocess.CalledProcessError, Phase2DataError) as e:
            log.warning("cache vocals failed (non-fatal): %r", e)
    return Acquired16k(wav, voc32=voc32, fullband=fullband, separator=separated_by)
