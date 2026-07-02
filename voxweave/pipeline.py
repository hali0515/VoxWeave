from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voxweave import backend
from voxweave import fsio
from voxweave.core.schema import Cue
from voxweave.chunking import (
    decode_to_wav,
    pack_speech_segments,
    silence_gaps,
    slice_wav,
    vad_speech_segments,
)
from voxweave.debug import DebugSink, FileDebugSink
from voxweave.lang import is_supported, to_iso_or
from voxweave.progress import Reporter
from voxweave import realign
from voxweave import translate as translate_mod
from voxweave import asrfix as asrfix_mod
from voxweave.songdet import (
    detect_song_spans,
    excise_spans_from_segments,
    expand_spans_to_voiced_blocks,
    filter_short_spans,
    group_segments_by_spans,
)
from voxweave.timestamps import shift_units

log = logging.getLogger("voxweave")

# ≤120s: long chunks occasionally trigger ASR repetition loops (stuck token ->
# zero-duration wall). Do NOT raise this to pack more; the risk and blast radius grow.
MAX_CHUNK_SEC = float(os.environ.get("VOXWEAVE_MAX_CHUNK_SEC", "120"))
# Spans shorter than this after expansion are kept as dialogue, not skipped.
# Real OP/ED runs 30-90s; short instrumental BGM scattered through speech would hurt ASR
# if dropped (env VOXWEAVE_MIN_SONG_SKIP_SEC).
MIN_SONG_SKIP_SEC = float(os.environ.get("VOXWEAVE_MIN_SONG_SKIP_SEC", "8"))
# Loudness normalization applied only to the 16k VAD/ASR path; 44.1k separation path is untouched.
ASR_LOUDNORM = os.environ.get("VOXWEAVE_LOUDNORM", "loudnorm=I=-16:TP=-1.5:LRA=11")
# PANNs Cnn14 is trained at 32k.
SONGDET_SR = 32000
# Sensitive VAD threshold for snapping zero-duration units to original (pre-separation)
# audio. Silero default 0.5 misses back-channels (はい/ええ) attenuated by vocal separation;
# 0.25 catches them. Used only for snap positioning, not for chunk boundary decisions.
SNAP_VAD_THRESHOLD = float(os.environ.get("VOXWEAVE_SNAP_VAD_THRESHOLD", "0.25"))
# Fine VAD pass for song excision: a small min-silence (vs the 300ms chunking default)
# surfaces brief intra-segment pauses, so excision cut points land in real silence and
# never bisect a dialogue word. Only runs when song spans were detected.
SONG_FINE_SILENCE_MS = int(os.environ.get("VOXWEAVE_SONG_FINE_SILENCE_MS", "100"))
# Align-stage cue duration floor. Default 0 (disabled): enforce_min_duration only
# resolves overlaps without padding, so short back-channels keep their real ~0.6s.
# Set VOXWEAVE_MIN_CUE_SEC=0.8 to re-enable padding. Distinct from VOXWEAVE_SEG_MIN_CUE_SEC.
MIN_CUE_SEC = float(os.environ.get("VOXWEAVE_MIN_CUE_SEC", "0"))
# Flash-cue rescue (orthogonal to MIN_CUE_SEC): genuine flash cues (so/あ at 0.1-0.2s)
# are extended to TINY_CUE_TARGET, allowed to overlap only the immediately following cue.
# VOXWEAVE_TINY_CUE_SEC=0 disables.
TINY_CUE_SEC = float(os.environ.get("VOXWEAVE_TINY_CUE_SEC", "0.2"))
TINY_CUE_TARGET = float(os.environ.get("VOXWEAVE_TINY_CUE_TARGET", "0.5"))


# Vocals cache: <media_dir>/cache/<stem>.vocals.32k.flac (32k mono, no BGM).
# Shared by process and align; PANNs eats it directly, ASR/alignment downsample to 16k.
# Legacy <stem>.16k.flac caches are still accepted by align for backward compatibility.
CACHE_DIRNAME = "cache"
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


def _progress_bridge(rep: Reporter, label: str):
    """Convert the ``(done, total)`` callback from backend/songdet into a Reporter task bar.

    Keeps backend/songdet free of any rich dependency.
    """
    started = {"v": False}

    def cb(done: int, total: int) -> None:
        if not started["v"]:
            rep.task(label, total)
            started["v"] = True
        rep.advance(1)

    return cb


def cache_vocals_path(media_path: Path) -> Path:
    """Return the canonical vocals cache path: <media_dir>/cache/<stem>.vocals.32k.flac."""
    media_path = Path(media_path)
    return media_path.parent / CACHE_DIRNAME / f"{media_path.stem}.vocals.32k.flac"


def cache_16k_path(media_path: Path) -> Path:
    """Return the legacy 16k vocals cache path: <media_dir>/cache/<stem>.16k.flac (read-only backward compat)."""
    media_path = Path(media_path)
    return media_path.parent / CACHE_DIRNAME / f"{media_path.stem}.16k.flac"


def _encode_flac(src_wav: Path, dst_flac: Path) -> None:
    """Encode wav to flac for caching (lossless); caller treats failure as non-fatal.

    Atomic: an interrupted encode must not leave a truncated flac at the cache
    path — every later run would treat it as a cache hit and fail obscurely.
    """
    dst_flac.parent.mkdir(parents=True, exist_ok=True)
    with fsio.atomic_path(dst_flac) as tmp:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(src_wav), "-c:a", "flac", str(tmp)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


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


def require_vtt(path: Path) -> Path:
    """Reject non-VTT inputs for align/correct; return the path unchanged.

    Both commands write VTT back (align overwrites the input in place, correct's
    ``--apply`` does too), so running them on ``.srt``/``.ass`` would corrupt the
    file with VTT content. translate/pack/burn/export accept the other formats
    via :func:`voxweave.subformats.require_subtitle`.
    """
    from voxweave.subformats import require_subtitle

    return require_subtitle(path, exts=(".vtt",))


def _find_sibling_media(ref: Path) -> Path | None:
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


def _separate_to_16k_32k(
    media: Path, *, reporter: Reporter, normalize: bool
) -> tuple[Path, Path, Path, Path]:
    """Decode full-band 44.1k stereo -> Roformer separate -> resample, returning
    ``(fullband, vocals, wav_16k, voc32_32k)``.

    The full-band 44.1k stereo feed is a hard constraint (Roformer is trained at 44.1k);
    downsampling to 16k/32k happens only after separation. Callers own temp bookkeeping,
    debug dumps, and caching of the returned paths.

    On a clean return the caller registers the paths in its own ``tmp`` list (cleaned in its
    ``finally``). Since that registration only runs after this returns, the helper self-cleans
    its partial outputs if a later step raises — otherwise an OOM/ffmpeg failure mid-separation
    would orphan the already-decoded temp files.
    """
    af = ASR_LOUDNORM if normalize else None
    created: list[Path] = []
    try:
        reporter.stage("decode fullband 44.1k")
        fullband = decode_to_wav(media, sample_rate=44100, mono=False)
        created.append(fullband)
        reporter.stage("vocal separation (Roformer)")
        vocals = backend.separate_vocals(
            fullband,
            progress=_progress_bridge(reporter, "vocal separation (Roformer)"),
        )
        created.append(vocals)
        reporter.stage("resample 16k")
        wav = decode_to_wav(vocals, audio_filter=af)
        created.append(wav)
        voc32 = decode_to_wav(
            vocals, sample_rate=SONGDET_SR
        )  # 32k mono: PANNs + cache source
        return fullband, vocals, wav, voc32
    except Exception:
        for p in created:
            p.unlink(missing_ok=True)
        raise


def _load_sibling_json(json_path: Path, *, require: str | None = None) -> dict:
    """Load a sibling ``.json`` with readable failures: a corrupt file or a
    missing required key names the file and points at regeneration instead of
    surfacing a bare JSONDecodeError/KeyError deep in the stack."""
    json_path = Path(json_path)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{json_path.name} is corrupt JSON ({e.msg} at line {e.lineno});"
            " re-run transcribe/process to regenerate it"
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(
            f"{json_path.name}: expected a JSON object, got {type(data).__name__};"
            " re-run transcribe/process to regenerate it"
        )
    if require is not None and require not in data:
        raise RuntimeError(
            f"{json_path.name} has no {require!r} key;"
            " re-run transcribe/process to regenerate it"
        )
    return data


def _load_cues(vtt_path: Path) -> list[dict]:
    """Parse subtitle cue blocks by extension (VTT/SRT/ASS/SSA); raise if the
    file has no cues. Shared guard for align/translate/correct."""
    from voxweave.subformats import load_subtitle_blocks

    return load_subtitle_blocks(Path(vtt_path))


def plan_song_skip(
    song_spans: list[tuple[float, float]],
    sing_spans: list[tuple[float, float]],
    segs: list[dict],
    *,
    speech_spans: list[tuple[float, float]] | None = None,
    silences: list[tuple[float, float]] | None = None,
    min_skip_sec: float,
    max_chunk_sec: float,
) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], list[dict], list[dict]
]:
    """Pure song-skip decision chain: expand -> filter -> excise -> group -> pack.

    Returns (expanded_spans, final_spans, kept_segs, chunks). No side effects, no GPU
    calls -- shared with scenario replay tests.

    Two song scales, two treatments:
    - Long singing spans (>= min_skip_sec) anchor OP/ED sequences: they absorb their whole
      voiced block (rap verses PANNs hears as Speech ride along), clean dialogue is trimmed
      from the block edges (``protect=speech_spans``), and instrumental-only spans still
      shorter than min_skip_sec after expansion are kept as content (Cecilia guard).
    - Short singing spans (< min_skip_sec — a hummed bar inside a dialogue block) must NOT
      absorb their block and must not be discarded by the length filter either: they go
      straight to excision.

    Excision replaces whole-segment dropping for everything: song intervals are cut OUT of
    the VAD segments (cut points snapped into real silences), so a segment mixing
    "speech, brief pause, humming, speech" keeps its dialogue and loses only the
    song + flanking silence.
    """
    long_sing = [sp for sp in sing_spans if sp[1] - sp[0] >= min_skip_sec]
    expanded = expand_spans_to_voiced_blocks(
        segs, song_spans, expandable=long_sing, protect=speech_spans
    )
    final_long = filter_short_spans(expanded, min_sec=min_skip_sec)
    short_sing = [
        (a, b)
        for a, b in sing_spans
        if b - a < min_skip_sec
        and not any(max(a, fa) < min(b, fb) for fa, fb in final_long)
    ]
    to_cut = sorted(final_long + short_sing)
    if not to_cut:
        return expanded, [], segs, pack_speech_segments(segs, max_sec=max_chunk_sec)
    kept, final = excise_spans_from_segments(segs, to_cut, silences=silences)
    chunks: list[dict] = []
    for group in group_segments_by_spans(kept, final):
        chunks.extend(pack_speech_segments(group, max_sec=max_chunk_sec))
    return expanded, final, kept, chunks


def transcribe(
    media_path: Path,
    *,
    lang_override: str | None = None,
    separate: bool = True,
    skip_songs: bool = False,
    keep_lyrics: bool = False,
    diarize: bool = False,
    normalize: bool = False,
    reporter: Reporter | None = None,
    debug: bool = False,
    cache_vocals: Path | None = None,
    asr_model: str | None = None,
    context: str | None = None,
) -> tuple[
    str,
    list[dict],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float, str]],
]:
    """Run separation -> song skip -> VAD chunking -> ASR -> alignment.

    Returns ``(iso_language, word_segments, vad_spans, sing_spans, speaker_turns)``.
    vad_spans are the original-audio speech intervals, persisted to JSON for gap
    splitting. ``keep_lyrics`` runs song detection but skips excision: sung regions go
    through ASR/alignment like dialogue, and the detected singing spans come back so
    :func:`process` can flag lyric cues (empty unless keep_lyrics). ``diarize`` runs
    pyannote on the separated-vocals wav and returns the speaker turns (empty unless
    set). All models run in-process (no network calls). smart_split and file writing
    are handled by :func:`process`.
    """
    media_path = Path(media_path)
    rep = reporter or Reporter()
    dbg: DebugSink = FileDebugSink(media_path.stem) if debug else DebugSink()
    af = ASR_LOUDNORM if normalize else None
    tmp: list[
        Path
    ] = []  # intermediate files (fullband/vocals/16k/32k wav), deleted at end
    tmp_chunks: list[Path] = []
    try:
        vocals: Path | None = None
        fullband: Path | None = None
        voc32: Path | None = None  # 32k mono vocals: PANNs input + cache source
        if separate:
            if cache_vocals is not None and Path(cache_vocals).exists():
                # Cache hit: skip Roformer; PANNs eats 32k directly, ASR downsamples to 16k.
                rep.stage("vocals cache (32k)")
                log.info("reuse cached vocals %s", cache_vocals)
                voc32 = Path(cache_vocals)
                wav = decode_to_wav(voc32, audio_filter=af)  # 32k flac -> 16k mono
            else:
                fullband, vocals, wav, voc32 = _separate_to_16k_32k(
                    media_path, reporter=rep, normalize=normalize
                )
                tmp.append(fullband)
                dbg.audio("00_fullband_44k.wav", fullband)
                tmp.append(vocals)
                dbg.audio("01_vocals.flac", vocals)
                tmp.append(voc32)
                log.info("separated vocals (local Roformer)")
                if cache_vocals is not None:
                    try:
                        _encode_flac(voc32, Path(cache_vocals))
                        log.info("cached vocals 32k → %s", cache_vocals)
                    except (OSError, subprocess.CalledProcessError) as e:
                        log.warning("cache vocals failed (non-fatal): %r", e)
        else:
            rep.stage("decode 16k")
            wav = decode_to_wav(media_path, audio_filter=af)
        tmp.append(wav)
        dbg.audio("02_speech_16k.wav", wav)

        # Song detection must run on clean separated vocals; BGM causes speech/music confusion.
        song_spans: list[tuple[float, float]] = []
        sing_spans: list[tuple[float, float]] = []  # subset triggering block expansion
        speech_spans: list[tuple[float, float]] = []  # trimmed from song core edges
        if skip_songs or keep_lyrics:
            if not separate or voc32 is None:
                # --no-separate + skip-songs is valid (clean input); skip detection silently.
                log.debug(
                    "song detection requires separated vocals; skipping with --no-separate"
                )
            else:
                try:
                    rep.stage("song detection (PANNs)")
                    song_spans, sing_spans, speech_spans = detect_song_spans(
                        voc32, progress=_progress_bridge(rep, "song detection (PANNs)")
                    )
                    if song_spans:
                        log.info(
                            "song spans: %s",
                            [(round(a, 1), round(b, 1)) for a, b in song_spans],
                        )
                except ModuleNotFoundError as e:
                    # panns-inference not installed; continue without song skip.
                    # Install voxweave[songdet] or pass --no-skip-songs to suppress.
                    log.warning(
                        "song detection requires panns-inference (not installed: %s) -- "
                        "continuing without song skip; install voxweave[songdet] or pass --no-skip-songs",
                        e,
                    )

        rep.stage("VAD chunking")
        segs = vad_speech_segments(wav)
        if keep_lyrics:
            # Keep-lyrics: sung regions stay in the chunk stream and get ASR'd like
            # dialogue; only the singing spans (human vocals) are kept for cue marking.
            if sing_spans:
                log.info(
                    "keeping %d singing span(s) as lyrics: %s",
                    len(sing_spans),
                    [(round(a, 1), round(b, 1)) for a, b in sing_spans],
                )
            song_spans = []  # disable excision + VAD-reference exclusion below
            chunks = pack_speech_segments(segs, max_sec=MAX_CHUNK_SEC)
        elif song_spans:
            # Fine VAD (small min-silence) exposes brief intra-segment pauses; excision
            # snaps its cut points into these so dialogue words are never bisected.
            fine = vad_speech_segments(wav, min_silence_ms=SONG_FINE_SILENCE_MS)
            silences = silence_gaps(fine)
            # Decision chain lives in plan_song_skip (pure, shared with scenario tests).
            before = sum(s["end"] - s["start"] for s in segs)
            expanded, song_spans, segs, chunks = plan_song_skip(
                song_spans,
                sing_spans,
                segs,
                speech_spans=speech_spans,
                silences=silences,
                min_skip_sec=MIN_SONG_SKIP_SEC,
                max_chunk_sec=MAX_CHUNK_SEC,
            )
            log.info(
                "song spans (expanded): %s",
                [(round(a, 1), round(b, 1)) for a, b in expanded],
            )
            short = [
                (round(a, 1), round(b, 1))
                for a, b in song_spans
                if (b - a) < MIN_SONG_SKIP_SEC
            ]
            if short:
                log.info(
                    "short singing spans excised in-segment (<%.0fs): %s",
                    MIN_SONG_SKIP_SEC,
                    short,
                )
            after = sum(s["end"] - s["start"] for s in segs)
            log.info("excised %.1fs of speech-segment time as song", before - after)
        else:
            chunks = pack_speech_segments(segs, max_sec=MAX_CHUNK_SEC)
        if not chunks:
            raise RuntimeError(f"no speech detected in {media_path.name}")

        rep.stage("load ASR/alignment models")
        # Slice all chunk waveforms upfront so dual-pass (full ASR -> release -> full
        # alignment) can shave VRAM peak.
        cwavs: list[Path] = []
        for ch in chunks:
            cwav = slice_wav(wav, ch["start"], ch["end"])
            tmp_chunks.append(cwav)
            cwavs.append(cwav)
        from voxweave.config import conf_load_strategy

        strategy = conf_load_strategy()
        rep.chunks(len(chunks) * backend.chunk_pass_count(asr_model, strategy))
        # full_wav + bounds let CTC/MMS languages run ONE full-file alignment pass over
        # the whole audio (chunk windows as DP silence anchors) instead of N per-chunk
        # calls; Qwen-aligned languages (zh/yue) keep per-chunk inside transcribe_chunks.
        results = backend.transcribe_chunks(
            cwavs,
            lang_override,
            asr_model=asr_model,
            context=context,
            on_done=lambda _i: rep.chunk_done(),
            strategy=strategy,
            full_wav=wav,
            bounds=[(ch["start"], ch["end"]) for ch in chunks],
            # post-excise speech segments on the separated wav: the CTC full pass
            # soft-masks emissions outside these so words cannot park in music/silence
            speech_spans=[(s["start"], s["end"]) for s in segs],
        )
        # reinject_punct runs after language resolution (tokenization must match iso),
        # so punctuation cannot be reinjected per-chunk.
        chunk_pairs: list[tuple[str, list[dict]]] = []
        detected: list[str] = []  # per-chunk detected language (debug meta only)
        lang_weight: Counter[str] = Counter()  # vote weighted by aligned unit count
        for idx, (ch, cwav, (det_lang, text, units)) in enumerate(
            zip(chunks, cwavs, results)
        ):
            if not text.strip():
                log.warning("empty ASR for chunk @%.1fs, skipping", ch["start"])
                dbg.chunk(
                    idx,
                    wav=cwav,
                    start=ch["start"],
                    end=ch["end"],
                    raw=text,
                    text=text,
                    lang=det_lang,
                    units=None,
                )
                continue
            if det_lang:
                detected.append(det_lang)
                lang_weight[det_lang] += len(units)
            dbg.chunk(
                idx,
                wav=cwav,
                start=ch["start"],
                end=ch["end"],
                raw=text,
                text=text,
                lang=det_lang,
                units=units,
            )
            chunk_pairs.append((text, shift_units(units, ch["offset"])))

        if not chunk_pairs:
            raise RuntimeError(f"no aligned units for {media_path.name}")

        # Unit-count weighting lets long dialogue dominate over short cold-open/insert segments.
        if lang_override:
            lang_name = lang_override
        elif lang_weight:
            lang_name = lang_weight.most_common(1)[0][0]
        else:
            lang_name = "english"
        if not is_supported(lang_name):
            log.warning(
                "language %r not in aligner set; smart_split may misbehave", lang_name
            )
        iso = to_iso_or(lang_name, "en")

        # Aligner strips punctuation; reinject_punct reattaches it by time so smart_split
        # can use it for sentence breaking and space insertion.
        all_units: list[dict] = []
        for txt, u in chunk_pairs:
            all_units.extend(realign.reinject_punct(txt, u, iso))
        if not all_units:
            raise RuntimeError(f"no aligned units for {media_path.name}")
        # Zero-duration snap: the aligner collapses short words after a pause (e.g. はい)
        # to zero duration. We snap them into the actual speech region using VAD.
        # Vocal separation attenuates secondary-speaker back-channels, so separated-vocals
        # VAD misses them. We run VAD on the ORIGINAL audio (retains attenuated speech) as
        # the timing reference, excluding song spans to avoid snapping onto singing.
        # vad_spans are persisted to .json (vad_speech) for reuse by split.
        # SNAP_VAD_THRESHOLD (0.25) catches attenuated back-channels; --no-separate uses
        # silero default (0.5) since the original audio is not available separately.
        if separate and fullband is not None:
            orig16k = decode_to_wav(fullband)
            tmp.append(orig16k)
            orig_segs = vad_speech_segments(orig16k, threshold=SNAP_VAD_THRESHOLD)
            if song_spans:
                # Subtract only the truly sung/instrumental parts: clean-dialogue
                # windows inside expanded song spans (dialogue spoken OVER the
                # song) must survive in vad_speech, otherwise snapping and the
                # emission mask forbid those words' true location and the aligner
                # smears them across the song (observed on movie dialogue-over-
                # montage: a 15s exchange stretched over 65s).
                from voxweave.songdet import subtract_spans

                orig_segs, _ = excise_spans_from_segments(
                    orig_segs, subtract_spans(song_spans, speech_spans)
                )
            vad_spans = [(s["start"], s["end"]) for s in orig_segs]
        else:
            vad_spans = [(s["start"], s["end"]) for s in segs]
        # Qwen aligner has no CTC blank token, so word durations bleed into silence.
        # position_units_with_vad carves true gaps, giving smart_split an accurate signal.
        all_units = realign.position_units_with_vad(all_units, vad_spans)
        dbg.meta(
            {
                "media": str(media_path),
                "separate": separate,
                "skip_songs": skip_songs,
                "song_spans": song_spans,
                "language": iso,
                "detected": detected,
                "chunks": len(chunks),
                "units": len(all_units),
            }
        )
        speaker_turns: list[tuple[float, float, str]] = []
        if diarize:
            from voxweave import diarize as diarize_mod

            rep.stage("speaker diarization (pyannote)")
            try:
                speaker_turns = diarize_mod.diarize_turns(wav)
            finally:
                diarize_mod.release()
        return (
            iso,
            all_units,
            vad_spans,
            sing_spans if keep_lyrics else [],
            speaker_turns,
        )
    finally:
        # Release ASR/alignment singleton VRAM (separation self-releases earlier).
        backend.release()
        for p in tmp:
            p.unlink(missing_ok=True)
        for c in tmp_chunks:
            c.unlink(missing_ok=True)


def _spans_in(raw: Any) -> list[tuple[float, float]] | None:
    """Parse a persisted ``vad_speech`` array (``[[start, end], ...]``) to float tuples.

    Malformed entries (wrong arity, non-numeric bounds) are skipped with a warning
    rather than crashing the whole re-split. None if absent/empty or nothing survives.
    """
    if not raw:
        return None
    out: list[tuple[float, float]] = []
    for entry in raw:
        try:
            s, e = entry
            out.append((float(s), float(e)))
        except (TypeError, ValueError):
            log.warning("skipping malformed vad_speech entry: %r", entry)
    return out or None


def _turns_in(raw: Any) -> list[tuple[float, float, str]] | None:
    """Parse persisted ``speaker_turns`` (``[[start, end, label], ...]``).

    Malformed entries (wrong arity, non-numeric bounds) are skipped with a warning.
    None if absent/empty or nothing survives.
    """
    if not raw:
        return None
    out: list[tuple[float, float, str]] = []
    for entry in raw:
        try:
            s, e, lb = entry
            out.append((float(s), float(e), str(lb)))
        except (TypeError, ValueError):
            log.warning("skipping malformed speaker_turns entry: %r", entry)
    return out or None


def _maybe_adaptive_thresholds(th: dict, units: list[dict]) -> dict:
    """Scale clause/offline gap thresholds to this file's gap distribution.

    EXPERIMENTAL, default off: opt in via VOXWEAVE_GAP_ADAPTIVE=1. Replaces the
    static clause_ms (and offline_ms at the same clause:offline ratio) with a
    per-file estimate from the inter-unit gap distribution; vad_skip_ms is
    untouched. Validate against scripts/calib_segmentation.py before trusting.
    """
    if os.environ.get("VOXWEAVE_GAP_ADAPTIVE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return th
    from voxweave.core.gap_split import adaptive_clause_ms

    gaps_ms: list[float] = []
    for prev, nxt in zip(units, units[1:]):
        pe, ns = prev.get("end"), nxt.get("start")
        if pe is not None and ns is not None:
            gaps_ms.append((float(ns) - float(pe)) * 1000.0)
    clause = adaptive_clause_ms(gaps_ms)
    if clause is None:
        return th
    ratio = th["offline_ms"] / th["clause_ms"] if th.get("clause_ms") else 1.75
    out = dict(th)
    out["clause_ms"] = clause
    out["offline_ms"] = round(clause * ratio)
    log.info(
        "adaptive gap thresholds: clause %dms offline %dms (static %s/%s)",
        out["clause_ms"],
        out["offline_ms"],
        th.get("clause_ms"),
        th.get("offline_ms"),
    )
    return out


# A cue is a lyric when at least this fraction of its span overlaps detected singing.
LYRIC_MIN_OVERLAP = 0.5


def mark_lyric_cues(
    cues: Sequence["Cue"], sing_spans: list[tuple[float, float]] | None
) -> None:
    """Flag cues whose span mostly overlaps detected singing (``lyric=True``).

    The stored cue text stays clean; display layers (`_write_siblings` VTT rows,
    SRT/ASS export) wrap flagged cues with music notes per the Netflix lyric
    convention. Runs in place after smart_split so flags ride the final cues.
    """
    if not sing_spans:
        return
    for c in cues:
        start, end = c.get("start"), c.get("end")
        if start is None or end is None or end - start <= 0:
            continue
        overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in sing_spans)
        if overlap / (end - start) >= LYRIC_MIN_OVERLAP:
            c["lyric"] = True


def lyric_display_text(cue: Mapping[str, Any]) -> str:
    """Cue display text: lyric cues get the Netflix music-note wrap (note + space
    at the start and end of the subtitle), others pass through unchanged."""
    text = str(cue["text"])
    return f"♪ {text} ♪" if cue.get("lyric") else text


def _dump_sibling_json(
    json_path: Path,
    *,
    language: str,
    segments: Sequence[Mapping[str, Any]],
    units: list[dict],
    vad_speech: list[tuple[float, float]] | None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
) -> None:
    """Write the sibling JSON document (language + segments + word_segments + optional
    vad_speech / shot_changes / sing_spans / speaker_turns).

    ``vad_speech=None`` omits the key; a list (even empty) writes it coerced to
    ``[[float, float], ...]``. ``shot_changes`` behaves the same (written only when not
    None, so ``split`` can replay shot snapping without re-decoding the video), as do
    ``sing_spans`` (lyric re-flagging without PANNs) and ``speaker_turns`` (speaker
    re-formatting without pyannote). Single source of truth for the sibling-JSON shape
    shared by process and align.
    """
    data: dict = {"language": language, "segments": segments, "word_segments": units}
    if vad_speech is not None:
        data["vad_speech"] = [[float(s), float(e)] for s, e in vad_speech]
    if shot_changes is not None:
        data["shot_changes"] = [float(t) for t in shot_changes]
    if sing_spans is not None:
        data["sing_spans"] = [[float(s), float(e)] for s, e in sing_spans]
    if speaker_turns is not None:
        data["speaker_turns"] = [
            [float(s), float(e), str(lb)] for s, e, lb in speaker_turns
        ]
    fsio.atomic_write_text(json_path, json.dumps(data, ensure_ascii=False, indent=2))


def _write_siblings(
    src: Path,
    cues: Sequence[Mapping[str, Any]],
    units: list[dict],
    lang: str,
    vad_speech: list[tuple[float, float]] | None = None,
    timestamps: bool = True,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
) -> Path:
    """Write sibling .json (ground truth) and .vtt alongside src; return the .vtt path.

    ``timestamps=True`` writes a timing line before each cue (word-level precision); cues
    missing start/end fall back to plain text. ``timestamps=False`` writes a plain-text
    edit draft for human editing before re-running ``align``. Both formats are accepted by
    ``realign.parse_vtt_blocks``. Lyric-flagged cues render with the music-note wrap in
    the VTT only; the JSON keeps clean text + the flag. Uses ``swap_ext`` (not
    ``with_suffix``) to preserve interior dots in filenames.
    """
    _dump_sibling_json(
        swap_ext(src, ".json"),
        language=lang,
        segments=cues,
        units=units,
        vad_speech=vad_speech or [],
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
    )
    rows = [
        (
            c.get("start") if timestamps else None,
            c.get("end") if timestamps else None,
            lyric_display_text(c),
        )
        for c in cues
    ]
    vtt_path = swap_ext(src, ".vtt")
    fsio.atomic_write_text(vtt_path, realign.render_cues(rows))
    return vtt_path


def _units_to_seg(units: list[dict], iso: str) -> dict:
    """Flatten word_segments into a single segment dict for smart_split.

    Units already carry punctuation from reinject_punct. No-space languages join without
    separator; smart_split uses punctuation for sentence breaking and converts it to spaces.
    """
    sep = "" if iso in realign.NO_SPACE_LANGS else " "
    words = [{"word": u["text"], "start": u["start"], "end": u["end"]} for u in units]
    return {
        "start": units[0]["start"],
        "end": units[-1]["end"],
        "text": sep.join(u["text"] for u in units),
        "words": words,
    }


def process(
    media_path: Path,
    lang_override: str | None = None,
    separate: bool = True,
    reporter: Reporter | None = None,
    debug: bool = False,
    normalize: bool = False,
    skip_songs: bool = False,
    keep_lyrics: bool = False,
    sdh: bool = False,
    diarize: bool = False,
    word_segments: tuple[str, list[dict]] | None = None,
    asr_model: str | None = None,
    context: str | None = None,
    timestamps: bool = True,
    shot_snap: bool = True,
) -> Path:
    """Full pipeline: transcribe -> smart_split -> write siblings. Return the .vtt path.

    Pass ``word_segments`` to skip transcription (tests / special cases); that path
    also skips shot detection (no media decode in unit tests). ``keep_lyrics``
    transcribes detected songs instead of excising them and flags the sung cues
    (rendered with a music-note wrap; spans persist to JSON for ``split`` replay).
    ``sdh`` additionally writes a ``<stem>.sdh.vtt`` sidecar with PANNs-detected
    non-speech event tags merged into the dialogue (main VTT/JSON untouched).
    ``diarize`` runs pyannote speaker diarization and formats multi-speaker cues
    (dual-speaker hyphens / speaker-boundary splits; turns persist to JSON).
    """
    media_path = Path(media_path)
    rep = reporter or Reporter()
    vad_speech: list[tuple[float, float]] | None = None
    shot_changes: list[float] | None = None
    sing_spans: list[tuple[float, float]] | None = None
    speaker_turns: list[tuple[float, float, str]] | None = None
    if word_segments is not None:
        iso, units = word_segments
    else:
        iso, units, vad_speech, sing_spans, speaker_turns = transcribe(
            media_path,
            lang_override=lang_override,
            separate=separate,
            skip_songs=skip_songs,
            keep_lyrics=keep_lyrics,
            diarize=diarize,
            normalize=normalize,
            reporter=reporter,
            debug=debug,
            cache_vocals=cache_vocals_path(media_path),
            asr_model=asr_model,
            context=context,
        )
        sing_spans = sing_spans or None
        speaker_turns = speaker_turns or None
        if shot_snap:
            from voxweave import shotdet

            rep.stage("shot detection")
            shot_changes = shotdet.detect_shot_changes(media_path)

    from voxweave.core.smart_split import smart_split_segments
    from voxweave.config import gap_thresholds

    # zh: Qwen punctuation can drift up to one character; snap to jieba word boundary
    # to prevent smart_split from splitting mid-word (e.g. 数据|中心 instead of 数据中心).
    units = realign.snap_break_punct(units, iso)
    seg = _units_to_seg(units, iso)
    rep.stage("smart_split layout")
    cues = smart_split_segments(
        [seg],
        lang=iso,
        speech_spans=vad_speech,
        thresholds=_maybe_adaptive_thresholds(gap_thresholds(iso), units),
        shot_changes=shot_changes,
    )
    mark_lyric_cues(cues, sing_spans)
    if speaker_turns:
        from voxweave.diarize import apply_speaker_format

        cues = apply_speaker_format(cues, speaker_turns, iso)

    rep.stage("write siblings")
    vtt_out = _write_siblings(
        media_path,
        cues,
        units,
        iso,
        vad_speech=vad_speech,
        timestamps=timestamps,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
    )
    log.info("wrote %s + .json (%d cues, lang=%s)", vtt_out.name, len(cues), iso)
    if sdh and word_segments is None:
        # The main VTT/JSON already landed; an SDH sidecar failure (PANNs OOM, missing
        # model) must not lose that primary output.
        try:
            _write_sdh_sidecar(media_path, cues, rep)
        except Exception as e:
            log.warning("SDH sidecar failed (non-fatal): %r", e)
    return vtt_out


def _write_sdh_sidecar(
    media_path: Path, cues: Sequence[Cue], rep: Reporter
) -> Path | None:
    """Detect non-speech events on the ORIGINAL mix (effects are stripped from the
    separated-vocals stem) and write ``<stem>.sdh.vtt`` (dialogue + event tags).
    Returns None when panns-inference is missing (warned, non-fatal)."""
    from voxweave import sdh as sdh_mod

    rep.stage("SDH event detection (PANNs)")
    wav32 = decode_to_wav(media_path, sample_rate=SONGDET_SR)
    try:
        events = sdh_mod.detect_events(
            wav32, progress=_progress_bridge(rep, "SDH event detection (PANNs)")
        )
    except ModuleNotFoundError as e:
        log.warning(
            "SDH detection requires panns-inference (not installed: %s); skipping sidecar",
            e,
        )
        return None
    finally:
        wav32.unlink(missing_ok=True)
    events = sdh_mod.fit_events_to_gaps(events, cues)
    path = swap_ext(media_path, ".sdh.vtt")
    fsio.atomic_write_text(path, sdh_mod.render_sdh_vtt(cues, events))
    log.info("wrote %s (%d event tag(s))", path.name, len(events))
    return path


def split(json_path: Path, timestamps: bool = True, **smart_split_kwargs) -> Path:
    """Re-run smart_split from persisted word_segments without any model calls.

    Reuses ``vad_speech`` from the sibling JSON for gap splitting; falls back to gap-only
    mode if absent. ``timestamps`` behaves as in :func:`process`.
    """
    from voxweave.core.smart_split import smart_split_segments
    from voxweave.config import gap_thresholds

    # Accept the .vtt sibling too: `voxweave split foo.vtt` should not feed
    # WEBVTT bytes to json.loads.
    json_path = swap_ext(Path(json_path), ".json")
    data = _load_sibling_json(json_path, require="word_segments")
    units = data["word_segments"]
    iso = data.get("language", "en")
    speech_spans = _spans_in(data.get("vad_speech"))
    shot_changes = [float(t) for t in data.get("shot_changes") or []] or None
    sing_spans = _spans_in(data.get("sing_spans"))
    speaker_turns = _turns_in(data.get("speaker_turns"))
    units = realign.snap_break_punct(
        units, iso
    )  # zh: snap to jieba boundary (same as process)
    seg = _units_to_seg(units, iso)
    cues = smart_split_segments(
        [seg],
        lang=iso,
        speech_spans=speech_spans,
        thresholds=_maybe_adaptive_thresholds(gap_thresholds(iso), units),
        shot_changes=shot_changes,
        **smart_split_kwargs,
    )
    mark_lyric_cues(cues, sing_spans)
    if speaker_turns:
        from voxweave.diarize import apply_speaker_format

        cues = apply_speaker_format(cues, speaker_turns, iso)
    vtt_out = _write_siblings(
        json_path,
        cues,
        units,
        iso,
        vad_speech=speech_spans,
        timestamps=timestamps,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
    )
    log.info("re-split %s → %d cues", vtt_out.name, len(cues))
    return vtt_out


def _prepare_16k_for_align(
    media: Path,
    *,
    separate: bool,
    normalize: bool,
    reporter: Reporter,
    tmp: list[Path],
) -> Path:
    """Prepare 16k vocals for align; append temp paths to tmp. Return the 16k wav path.

    Cache priority: 32k vocals flac -> legacy 16k flac -> re-run separation -> decode raw.
    """
    af = ASR_LOUDNORM if normalize else None
    if separate:
        cache = cache_vocals_path(media)
        if cache.exists():
            reporter.stage("vocals cache (32k)")
            log.info("reuse cached vocals %s", cache)
            wav = decode_to_wav(cache, audio_filter=af)  # 32k flac -> 16k
            tmp.append(wav)
            return wav
        legacy = cache_16k_path(media)
        if legacy.exists():
            reporter.stage("vocals cache (16k legacy)")
            log.info("reuse legacy 16k vocals %s", legacy)
            return legacy
        fullband, vocals, wav, voc32 = _separate_to_16k_32k(
            media, reporter=reporter, normalize=normalize
        )
        tmp.extend((fullband, vocals, wav, voc32))
        try:
            _encode_flac(voc32, cache)
            log.info("cached vocals 32k → %s", cache)
        except (OSError, subprocess.CalledProcessError) as e:
            log.warning("cache vocals failed (non-fatal): %r", e)
        return wav
    reporter.stage("decode 16k")
    wav = decode_to_wav(media, audio_filter=af)
    tmp.append(wav)
    return wav


def _write_align_json(
    json_path: Path,
    blocks: list[dict],
    spans: list[tuple[float, float]],
    units: list[dict],
    lang: str,
    vad_speech: list[tuple[float, float]] | None = None,
    shot_changes: list[float] | None = None,
    sing_spans: list[tuple[float, float]] | None = None,
    speaker_turns: list[tuple[float, float, str]] | None = None,
) -> None:
    """Update the sibling JSON with new alignment timing. Passes vad_speech,
    shot_changes, sing_spans, and speaker_turns through so split and subsequent
    align runs can reuse them without recomputing. Lyric flags survive on the
    re-timed segments.
    """
    segments = [
        {"text": b["text"], "start": a, "end": e}
        | ({"lyric": True} if b.get("lyric") else {})
        for b, (a, e) in zip(blocks, spans)
    ]
    _dump_sibling_json(
        json_path,
        language=lang,
        segments=segments,
        units=units,
        vad_speech=vad_speech,
        shot_changes=shot_changes,
        sing_spans=sing_spans,
        speaker_turns=speaker_turns,
    )


def _align_blocks(
    wav: Path,
    blocks: list[dict],
    iso: str,
    *,
    mms: bool,
    ctc_model: str | None,
    crops: list[tuple[float, float] | None],
    reporter: Reporter,
    tmp_chunks: list[Path],
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[list[dict]]:
    """Route blocks to the configured aligner and return per-block units.

    Three paths — these ARE the hard-constraint full-pass routing; do NOT collapse them:
    - ja MMS: one full-file pass (``align_blocks_full_mms``).
    - en wav2vec2 CTC: one full-file windowed-emission pass (``align_blocks_full_ctc``).
    - zh·yue (no CTC config): per-cue tight-crop Qwen — each cue gets its own audio slice so
      error is contained within the sentence and inter-sentence pauses are preserved.

    Per-cue slices are appended to ``tmp_chunks`` for the caller's ``finally`` to clean up.
    """
    # cue (start,end) bounds are used ONLY as silence anchors to split movie-length audio
    # into memory-sized chunks when it overflows the single-pass DP budget — NOT to crop/route
    # per cue. align is routing-free because the input VTT timestamps are exactly what may be
    # wrong (the reason to re-align); the global DP self-locates every word. None for cues
    # without timestamps. See memory voxweave-alignment-timing.
    bounds = [
        (b["start"], b["end"])
        if b["start"] is not None and b["end"] is not None
        else None
        for b in blocks
    ]
    if mms:
        reporter.task("full-file alignment (MMS)", 1)
        units = backend.align_blocks_full_mms(
            wav, [b["text"] for b in blocks], iso, bounds=bounds
        )
        reporter.advance(1)
        return units
    if ctc_model:  # en wav2vec2: windowed emission + single global DP (routing-free)
        reporter.task("full-file alignment (CTC)", 1)
        units = backend.align_blocks_full_ctc(
            wav,
            [b["text"] for b in blocks],
            iso,
            ctc_model,
            bounds=bounds,
            speech_spans=speech_spans,
        )
        reporter.advance(1)
        return units
    reporter.task("per-cue alignment", len(blocks))
    block_units: list[list[dict]] = [[] for _ in blocks]
    for i, crop in enumerate(crops):
        text = realign.join_block_texts([blocks[i]["text"]], iso)
        if crop is None or not text:  # insertion block or empty: skip
            reporter.advance(1)
            continue
        cs, ce = crop
        cwav = slice_wav(wav, cs, ce)
        tmp_chunks.append(cwav)
        block_units[i] = shift_units(backend.align_text(cwav, text, iso), cs)
        reporter.advance(1)
    return block_units


def align(
    vtt_path: Path,
    *,
    media_path: Path | None = None,
    separate: bool = True,
    normalize: bool = False,
    lang_override: str | None = None,
    reporter: Reporter | None = None,
) -> Path:
    """Re-align edited VTT text against original audio; overwrite VTT and update JSON.

    Routes each block to its audio window (via word_segments or VTT timestamps), slices
    and aligns locally, interpolates insertion blocks, then writes timing. ASR is not
    re-run; smart_split is not touched. All models run in-process (no network calls).
    """
    vtt_path = require_vtt(Path(vtt_path))  # align overwrites the input as VTT
    rep = reporter or Reporter()
    json_path = swap_ext(vtt_path, ".json")
    data = _load_sibling_json(json_path) if json_path.exists() else {}
    word_segments = data.get("word_segments", [])

    blocks = _load_cues(vtt_path)

    lang_name = lang_override or data.get("language") or "english"
    iso = to_iso_or(lang_name, "en")

    media = Path(media_path) if media_path else _find_sibling_media(vtt_path)
    if media is None or not media.exists():
        raise FileNotFoundError(
            f"source media for {vtt_path.name} not found (expected sibling with same stem); "
            f"align needs the original file to re-align, or specify --media"
        )

    # Full-file single-pass alignment (whisperx fork align_ctc) for both MMS (ja) and wav2vec2
    # CTC (en): concatenate all cue text, run one global monotone forced-align over the whole
    # audio, slice units back per cue by char/word count. The global path self-locates every
    # token (blank / <star> absorbs silence + song spans), immune to per-cue cropping drift
    # (observed: wrong coarse crop displaced エルダドワーフ by 11s; crammed en "blocks" into dead
    # air). Needs no has_ts/route/crop. ja MMS emission is windowed inside ctc-forced-aligner;
    # en wav2vec2 emission is windowed in align_blocks_full_ctc (full-file xlsr is O(T^2) -> OOM
    # at 23min). zh·yue have no CTC config -> per-cue tight-crop Qwen (routing+crop below). Do
    # NOT revert ja to per-cue MMS: repeated small ONNX calls corrupt the heap (~180-226 cues).
    from voxweave.config import align_model_for

    mms = backend.uses_mms(iso)
    ctc_model = None if mms else align_model_for(iso)
    full_pass = mms or bool(ctc_model)
    crops: list[
        tuple[float, float] | None
    ] = []  # set + looped only on the per-cue (zh·yue) path
    if not full_pass:
        has_ts = all(b["start"] is not None and b["end"] is not None for b in blocks)
        if not has_ts and not word_segments:
            raise RuntimeError(
                f"{json_path.name} has no word_segments and VTT has no timestamps; "
                f"cannot route audio windows"
            )
        spans = realign.route_blocks(blocks, word_segments)
        crops = realign.crop_blocks(spans)
        if all(c is None for c in crops):
            raise RuntimeError(
                "routing failed: no alignable blocks (text completely mismatches word_segments?)"
            )

    tmp: list[Path] = []
    tmp_chunks: list[Path] = []
    try:
        wav = _prepare_16k_for_align(
            media,
            separate=separate,
            normalize=normalize,
            reporter=rep,
            tmp=tmp,
        )
        block_units = _align_blocks(
            wav,
            blocks,
            iso,
            mms=mms,
            ctc_model=ctc_model,
            crops=crops,
            reporter=rep,
            tmp_chunks=tmp_chunks,
            # vad_speech persisted by transcribe (same media timeline): lets the CTC
            # full pass mask non-speech emissions; absent/empty -> no masking
            speech_spans=_spans_in(data.get("vad_speech")),
        )

        # Tight cropping eliminates "last word drifts into inter-sentence silence", so
        # position_units_with_vad is not needed here (unlike the transcribe path).
        final, all_units = realign.group_block_spans(block_units)
        if not all_units:
            raise RuntimeError(f"no aligned units for {media.name}")
        # fill_insert -> enforce_min_duration -> rescue_tiny_cues (extend flash cues like
        # so/あ, overlap allowed with next-neighbor only) -> clamp.
        spans_filled = realign.clamp_spans(
            realign.rescue_tiny_cues(
                realign.enforce_min_duration(
                    realign.fill_insert_blocks(final), min_dur=MIN_CUE_SEC
                ),
                trig=TINY_CUE_SEC,
                target=TINY_CUE_TARGET,
            )
        )

        rep.stage("write VTT + JSON")
        fsio.atomic_write_text(vtt_path, realign.render_vtt(blocks, spans_filled))
        # Preserve vad_speech / shot_changes from the original JSON (computed by
        # transcribe from the original media; align does not recompute them).
        keep_vad = _spans_in(data.get("vad_speech"))
        keep_shots = [float(t) for t in data.get("shot_changes") or []] or None
        keep_sing = _spans_in(data.get("sing_spans"))
        keep_turns = _turns_in(data.get("speaker_turns"))
        _write_align_json(
            json_path,
            blocks,
            spans_filled,
            all_units,
            iso,
            keep_vad,
            keep_shots,
            keep_sing,
            keep_turns,
        )
        log.info(
            "aligned %s → %d cues, %d units", vtt_path.name, len(blocks), len(all_units)
        )
        return vtt_path
    finally:
        # Release aligner singleton VRAM (separation self-releases earlier).
        backend.release()
        for p in tmp:
            p.unlink(missing_ok=True)
        for c in tmp_chunks:
            c.unlink(missing_ok=True)


def translate(
    vtt_path: Path,
    *,
    to: str = "zh",
    context: str | None = None,
    glossary: dict[str, str] | str | None = None,
    model: str = translate_mod.TRANSLATE_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    reporter: Reporter | None = None,
) -> Path:
    """Translate subtitle cues via OpenAI; write <stem>.<to>.<ext> (source untouched).

    Accepts VTT/SRT/ASS/SSA; the output mirrors the input format (SSA is written
    back as ASS). Missing translations are retried once; any remaining are
    back-filled with source text. Output cue count always equals input cue count.
    """
    from voxweave.subformats import require_subtitle

    vtt_path = require_subtitle(Path(vtt_path))
    ext = ".ass" if vtt_path.suffix.lower() == ".ssa" else vtt_path.suffix.lower()
    rep = reporter or Reporter()
    blocks = _load_cues(vtt_path)
    if any(b.get("start") is None for b in blocks):
        if ext != ".vtt":
            raise ValueError(
                f"{vtt_path.name} has cues without timestamps; cannot render {ext}"
            )
        log.warning(
            "%s has no timestamps; translated output will be plain-text blocks (run align first)",
            vtt_path.name,
        )

    payload = translate_mod.build_payload(blocks)
    # Progress sidecar: completed windows survive a mid-run failure (network,
    # rate limit), so rerunning the same command resumes instead of restarting.
    progress_path = swap_ext(vtt_path, f".{to}.progress.json")
    tx_kwargs: dict[str, Any] = dict(
        to=to,
        model=model,
        context=context,
        glossary=glossary,
        base_url=base_url,
        api_key=api_key,
        progress_path=progress_path,
        progress_sig=translate_mod.payload_signature(payload),
    )
    rep.stage(f"translate {len(payload)} cues -> {to}")
    try:
        trans = translate_mod.translate_cues(payload, **tx_kwargs, reporter=rep)

        missing = translate_mod.validate_and_fill(blocks, trans)
        if missing:
            rep.stage(f"retry translate {len(missing)} cues")
            retry_payload = [payload[i] for i in missing]
            # Continuity tail: hand the retry window the already-translated cues that
            # precede the first gap so the model keeps register/terminology consistent.
            tail = [
                (payload[j]["t"], trans[j])
                for j in range(missing[0])
                if trans.get(j, "").strip()
            ][-translate_mod.CONTEXT_TAIL :]
            trans.update(
                translate_mod.translate_cues(
                    retry_payload, **tx_kwargs, tail=tail, reporter=rep
                )
            )
            still = translate_mod.validate_and_fill(blocks, trans)
            if still:
                log.warning(
                    "%d cues still untranslated, back-filling with source text: %s",
                    len(still),
                    still,
                )
    except Exception:
        if progress_path.exists():
            log.warning(
                "translation interrupted; progress saved to %s -- rerun the same "
                "command to resume",
                progress_path.name,
            )
        raise

    rep.stage(f"write translated {ext.lstrip('.').upper()}")
    rows = translate_mod.translated_rows(blocks, trans, to_iso=to_iso_or(to, None))
    if ext == ".vtt":
        content = realign.render_cues(rows)
    else:
        from voxweave.export import render_ass, render_srt

        timed = [
            (float(s), float(e), t)
            for s, e, t in rows
            if s is not None and e is not None
        ]
        content = render_srt(timed) if ext == ".srt" else render_ass(timed)
    out_path = swap_ext(vtt_path, f".{to}{ext}")
    fsio.atomic_write_text(out_path, content)
    progress_path.unlink(missing_ok=True)  # translation landed; sidecar done
    log.info("wrote %s (%d cues → %s)", out_path.name, len(blocks), to)
    return out_path


def correct(
    vtt_path: Path,
    *,
    glossary: dict[str, str] | str | None = None,
    model: str = asrfix_mod.FIX_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    apply: bool = False,
    align_after: bool = False,
    media_path: Path | None = None,
    separate: bool = True,
    normalize: bool = False,
    lang_override: str | None = None,
    reporter: Reporter | None = None,
) -> dict[str, Any]:
    """LLM ASR correction (run before align): send VTT to the LLM for a conservative diff.

    Default (review): writes sidecar ``<stem>.asrfix.vtt`` + audit ``<stem>.asrfix.json``,
    source untouched. ``apply``: overwrites the original VTT in place and writes **no audit
    json** (the diff is shown in the summary). When ``align_after`` and a real change was
    applied, immediately re-runs :func:`align` to refresh timestamps (text edits change
    word counts) and update the sibling ``<stem>.json``.

    Returns ``{out, audit, applied, rejected, n_cues, applied_in_place, aligned}``.
    """
    vtt_path = require_vtt(Path(vtt_path))  # --apply overwrites the input as VTT
    rep = reporter or Reporter()
    blocks = _load_cues(vtt_path)

    payload = asrfix_mod.build_payload(blocks)
    rep.stage(f"LLM correction {len(payload)} cues (model={model})")
    fixes = asrfix_mod.correct_cues(
        payload, model=model, glossary=glossary, base_url=base_url, api_key=api_key
    )
    new_texts, applied, rejected = asrfix_mod.apply_fixes(blocks, fixes)
    rendered = asrfix_mod.render_vtt(blocks, new_texts)

    audit_path: Path | None = None
    if apply:
        # in-place edit: overwrite the original, no sidecar json (diff lives in the summary)
        rep.stage("overwrite VTT in place")
        fsio.atomic_write_text(vtt_path, rendered)
        out_path = vtt_path
    else:
        rep.stage("write sidecar VTT + audit json")
        out_path = swap_ext(vtt_path, ".asrfix.vtt")
        fsio.atomic_write_text(out_path, rendered)
        audit_path = swap_ext(vtt_path, ".asrfix.json")
        try:
            fsio.atomic_write_text(
                audit_path,
                json.dumps(
                    {"applied": applied, "rejected": rejected},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception:
            # Sidecar VTT + audit JSON are a pair; if the audit write fails, unlink
            # the VTT so no orphaned half-pair is left behind (source stays untouched).
            out_path.unlink(missing_ok=True)
            raise
    log.info(
        "asrfix %s: %d applied / %d rejected → %s",
        vtt_path.name,
        len(applied),
        len(rejected),
        out_path.name,
    )

    # apply means "change the file for real" -> refresh timing right away (only worth it
    # if something actually changed; an empty diff leaves the VTT identical).
    aligned = False
    if apply and align_after and applied:
        align(
            out_path,
            media_path=media_path,
            separate=separate,
            normalize=normalize,
            lang_override=lang_override,
            reporter=rep,
        )
        aligned = True

    return {
        "out": out_path,
        "audit": audit_path,
        "applied": applied,
        "rejected": rejected,
        "n_cues": len(blocks),
        "applied_in_place": apply,
        "aligned": aligned,
    }
