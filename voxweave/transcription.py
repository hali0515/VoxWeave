from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from voxweave import artifacts, backend, chunking, config, realign, songdet
from voxweave.chunking import (
    decode_to_wav,
    pack_speech_segments,
    silence_gaps,
    slice_wav,
    vad_speech_segments,
)
from voxweave.debug import DebugSink, FileDebugSink
from voxweave.lang import is_supported, to_asr_iso, to_iso_or
from voxweave.progress import Reporter, progress_bridge
from voxweave.songdet import (
    detect_song_spans,
    excise_spans_from_segments,
    expand_spans_to_voiced_blocks,
    filter_short_spans,
    group_segments_by_spans,
    rescue_speech_segments,
    subtract_spans,
)
from voxweave.timestamps import shift_units
from voxweave.transcript_language import _select_transcript_language
from voxweave.vocals import acquire_16k
from voxweave.voicebase import require_sha256
from voxweave.voiceprint_capture import (
    VoiceprintCapture,
    _decoupled_voiceprint_capture,
    _prefetch_voiceprint_models,
)

log = logging.getLogger("voxweave")

# ≤120s: long chunks occasionally trigger ASR repetition loops (stuck token ->
# zero-duration wall). Do NOT raise this to pack more; the risk and blast radius grow.
MAX_CHUNK_SEC = config._env_float("VOXWEAVE_MAX_CHUNK_SEC", 120.0)
# Spans shorter than this after expansion are kept as dialogue, not skipped.
# Real OP/ED runs 30-90s; short instrumental BGM scattered through speech would hurt ASR
# if dropped (env VOXWEAVE_MIN_SONG_SKIP_SEC).
MIN_SONG_SKIP_SEC = config._env_float("VOXWEAVE_MIN_SONG_SKIP_SEC", 8.0)
# Sensitive VAD threshold for snapping zero-duration units to original (pre-separation)
# audio. Silero default 0.5 misses back-channels (はい/ええ) attenuated by vocal separation;
# 0.25 catches them. Used only for snap positioning, not for chunk boundary decisions.
SNAP_VAD_THRESHOLD = config._env_float("VOXWEAVE_SNAP_VAD_THRESHOLD", 0.25)
# Fine VAD pass for song excision: a small min-silence (vs the 300ms chunking default)
# surfaces brief intra-segment pauses, so excision cut points land in real silence and
# never bisect a dialogue word. Only runs when song spans were detected.
SONG_FINE_SILENCE_MS = config._env_int("VOXWEAVE_SONG_FINE_SILENCE_MS", 100)


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

    After expansion (and before excision), PANNs clean-dialogue spans rescue waveform-VAD
    misses: a >=3s stretch silero left uncovered but PANNs scored as Speech joins ``segs``
    (see :func:`songdet.rescue_speech_segments`), so the cold-open dialogue silero
    under-scores still reaches ASR. Rescue is deliberately AFTER expansion — rescue
    segments widen chunk coverage only and must not reshape voiced blocks or the edge trim.
    """
    long_sing = [sp for sp in sing_spans if sp[1] - sp[0] >= min_skip_sec]
    expanded = expand_spans_to_voiced_blocks(
        segs, song_spans, expandable=long_sing, protect=speech_spans
    )
    # Rescue AFTER expansion, before excision: rescue segments must only widen chunk/ASR
    # coverage — they must not reshape voiced blocks. A rescue segment can fill the one >3s
    # gap separating a dialogue block from a song block, gluing them into one block whose
    # edge trim then stops at the first non-clean segment far from the song (observed:
    # +131s over-excision). Songs are still cut out of rescued segments by excision below.
    if speech_spans:
        rescued = rescue_speech_segments(speech_spans, segs)
        if rescued:
            log.info(
                "speech rescue: %d PANNs-only segment(s) silero missed: %s",
                len(rescued),
                [(round(s["start"], 1), round(s["end"], 1)) for s in rescued],
            )
            segs = sorted(segs + rescued, key=lambda s: s["start"])
            if silences:
                # Fine-VAD calls the whole rescued region silence (silero under-scored
                # it — that is why it needed rescuing), so excision snapping could pull
                # a cut up to SNAP_SEC into rescued dialogue. Remove rescued intervals
                # from the snap targets; genuine silences elsewhere still snap.
                silences = subtract_spans(
                    list(silences), [(r["start"], r["end"]) for r in rescued]
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


# What the chunk planning stage of transcribe() decided: the packed ASR chunks, the
# VAD segments they were packed from (rescue merged, song excised), the song spans
# left for the later stages (empty under keep-lyrics) and the rescued speech islands.
@dataclass(frozen=True)
class _ChunkPlan:
    chunks: list[dict]
    segs: list[dict]
    song_spans: list[tuple[float, float]]
    rescued_spans: list[tuple[float, float]]


def _plan_chunks(
    wav: Path,
    segs: list[dict],
    *,
    song_spans: list[tuple[float, float]],
    sing_spans: list[tuple[float, float]],
    speech_spans: list[tuple[float, float]],
    keep_lyrics: bool,
) -> _ChunkPlan:
    # PANNs speech rescue, computed once against the raw silero segs (same inputs as
    # plan_song_skip's internal rescue, so the two lists are identical on the song
    # branch). Used to (a) widen chunk coverage on the keep-lyrics / no-song branches,
    # which never reach plan_song_skip, and (b) union into the persisted vad_speech
    # below — without (b) the timing reference calls the rescued dialogue silence and
    # a later `align --vad-mask` would evict exactly the words the rescue recovered.
    rescued = rescue_speech_segments(speech_spans, segs) if speech_spans else []
    rescued_spans = [(r["start"], r["end"]) for r in rescued]
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
        if rescued:
            log.info(
                "speech rescue: %d PANNs-only segment(s) silero missed: %s",
                len(rescued),
                rescued_spans,
            )
            segs = sorted(segs + rescued, key=lambda s: s["start"])
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
        # skip_songs on but nothing detected: rescue still applies (silero misses are
        # orthogonal to song presence — a recap episode has no OP yet can lose a cold open).
        if rescued:
            log.info(
                "speech rescue: %d PANNs-only segment(s) silero missed: %s",
                len(rescued),
                rescued_spans,
            )
            segs = sorted(segs + rescued, key=lambda s: s["start"])
        chunks = pack_speech_segments(segs, max_sec=MAX_CHUNK_SEC)
    return _ChunkPlan(
        chunks=chunks,
        segs=segs,
        song_spans=song_spans,
        rescued_spans=rescued_spans,
    )


# Pure interval union behind the persisted vad_speech key: the timing-reference
# spans plus the rescued speech islands (minus excised song overlap), merged.
def _union_rescued_spans(
    vad_spans: list[tuple[float, float]],
    rescued_spans: list[tuple[float, float]],
    song_spans: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    # Rescued regions are PANNs-confirmed speech BOTH silero passes under-score
    # (the separated pass by premise; the original-mix pass credibly too — same
    # theatrical delivery plus BGM). Union them into the timing reference, minus
    # any excised song overlap, or downstream consumers (position_units_with_vad,
    # smart_split gaps, a later `align --vad-mask`) treat the rescued dialogue as
    # silence — the mask would evict exactly the words the rescue recovered.
    add = (
        subtract_spans(rescued_spans, sorted(song_spans))
        if song_spans
        else rescued_spans
    )
    merged = sorted(vad_spans + [(a, b) for a, b in add if b > a])
    vad_spans = []
    for a, b in merged:
        if vad_spans and a <= vad_spans[-1][1]:
            vad_spans[-1] = (vad_spans[-1][0], max(vad_spans[-1][1], b))
        else:
            vad_spans.append((a, b))
    return vad_spans


# Zero-duration snap: the aligner collapses short words after a pause (e.g. はい)
# to zero duration. We snap them into the actual speech region using VAD.
# Vocal separation attenuates secondary-speaker back-channels, so separated-vocals
# VAD misses them. We run VAD on the ORIGINAL audio (retains attenuated speech) as
# the timing reference, excluding song spans to avoid snapping onto singing.
# vad_spans are persisted to .json (vad_speech) for reuse by split.
# SNAP_VAD_THRESHOLD (0.25) catches attenuated back-channels. The original is the
# full-band stem on a fresh separation, or the source media itself on a vocals-cache
# hit (no stem exists then). The two decode to slightly different 16k PCM, so a
# re-run's reference can differ a little from the first run's (VAD at this
# threshold has absorbed the difference on the clips checked so far; the samples
# themselves are not identical). --no-separate reuses the chunking VAD
# (silero default 0.5): the decoded 16k input already IS the original audio.
def _timing_reference_spans(
    media_path: Path,
    fullband: Path | None,
    plan: _ChunkPlan,
    speech_spans: list[tuple[float, float]],
    *,
    separate: bool,
    tmp: list[Path],
) -> list[tuple[float, float]]:
    if separate:
        orig16k = decode_to_wav(fullband if fullband is not None else media_path)
        tmp.append(orig16k)
        orig_segs = vad_speech_segments(orig16k, threshold=SNAP_VAD_THRESHOLD)
        if plan.song_spans:
            # Subtract only the truly sung/instrumental parts: clean-dialogue
            # windows inside expanded song spans (dialogue spoken OVER the
            # song) must survive in vad_speech, otherwise snapping and the
            # emission mask forbid those words' true location and the aligner
            # smears them across the song (observed on movie dialogue-over-
            # montage: a 15s exchange stretched over 65s).
            orig_segs, _ = excise_spans_from_segments(
                orig_segs, subtract_spans(plan.song_spans, speech_spans)
            )
        vad_spans = [(s["start"], s["end"]) for s in orig_segs]
    else:
        vad_spans = [(s["start"], s["end"]) for s in plan.segs]
    if plan.rescued_spans:
        vad_spans = _union_rescued_spans(vad_spans, plan.rescued_spans, plan.song_spans)
    return vad_spans


# The pyannote run configuration of the speaker identification stage: exactly the
# four values diarize_turns takes from the caller.
@dataclass(frozen=True)
class _DiarizeSettings:
    model: str | None
    clustering: str | None
    min_speakers: int | None
    max_speakers: int | None


# What diarize_turns records about the audio it was given. Pure: built from the
# separation facts of the run, so building it before the diarization stage starts
# changes nothing observable.
def _diarize_audio_profile(
    *,
    separate: bool,
    normalize: bool,
    separator_identity: dict | None,
) -> dict:
    return {
        "separated": separate,
        "normalized": normalize,
        "sample_rate": 16000,
        **(
            {
                "separator": {
                    **(
                        separator_identity
                        or {
                            "repo": backend.SEPARATOR_REPO,
                            "file": backend.SEPARATOR_REPO_FILE,
                            "checkpoint": "unresolved",
                            "config_sha256": "unresolved",
                        }
                    ),
                }
            }
            if separate
            else {}
        ),
    }


# Speaker identification stage of transcribe(): pyannote turns on the 16k wav, then
# the voiceprint capture. The two try levels stay separate on purpose: the inner
# finally frees pyannote before the capture loads its embedder, the outer one frees
# a kept clustering embedder on any path that never reaches the capture.
def _identify_speakers(
    wav: Path,
    iso: str,
    settings: _DiarizeSettings,
    *,
    voiceprints: bool,
    voiceprint_model: str | None,
    audio_profile: dict,
    rep: Reporter,
) -> tuple[list[tuple[float, float, str]], VoiceprintCapture | None]:
    from voxweave import diarize as diarize_mod
    from voxweave import voiceembed

    rep.step("identify speakers")
    voiceprint_target: voiceembed.EmbedderSpec | voiceembed.LegacyLane | None = (
        voiceembed.resolve_voiceprint_model(voiceprint_model, iso)
        if voiceprints
        else None
    )
    # Voiceprint clustering embeds with ReDimNet2; when the capture below
    # uses the same model, keep it resident across the two stages. The
    # capture releases it, and so does the finally below on any path
    # that never reaches the capture.
    reuse_clustering_embedder = (
        settings.clustering == config.DIARIZE_CLUSTERING_VOICEPRINT
        and isinstance(voiceprint_target, voiceembed.EmbedderSpec)
        and voiceprint_target.name == voiceembed.REDIMNET2_B6.name
    )
    rep.stage("speaker diarization (pyannote)")
    voiceprint_capture: VoiceprintCapture | None = None
    try:
        try:
            diarization = diarize_mod.diarize_turns(
                wav,
                model=settings.model,
                min_speakers=settings.min_speakers,
                max_speakers=settings.max_speakers,
                clustering=settings.clustering,
                release_embedder=not reuse_clustering_embedder,
                # Only the legacy lane reads the pipeline's own embeddings;
                # a dedicated embedder computes its voiceprints below.
                want_embeddings=voiceprint_target is voiceembed.LEGACY,
                audio_profile=audio_profile,
            )
        finally:
            diarize_mod.release()
        speaker_turns = diarization.turns
        if isinstance(voiceprint_target, voiceembed.EmbedderSpec):
            voiceprint_capture = _decoupled_voiceprint_capture(
                wav,
                diarization,
                voiceprint_target,
                reporter=rep,
            )
        elif voiceprint_target is voiceembed.LEGACY and diarization.centroids:
            voiceprint_capture = VoiceprintCapture(
                centroids=diarization.centroids,
                provenance=diarization.provenance,
                turns=diarization.turns,
            )
    finally:
        if reuse_clustering_embedder:
            voiceembed.release()  # idempotent after the capture's own
    return speaker_turns, voiceprint_capture


def transcribe(
    media_path: Path,
    *,
    lang_override: str | None = None,
    separate: bool = True,
    skip_songs: bool = False,
    keep_lyrics: bool = False,
    diarize: bool = False,
    diarize_model: str | None = None,
    voiceprints: bool = False,
    voiceprint_model: str | None = None,
    normalize: bool = False,
    reporter: Reporter | None = None,
    debug: bool = False,
    debug_root: Path | None = None,
    cache_vocals: Path | None = None,
    source_fingerprint: str | None = None,
    asr_model: str | None = None,
    context: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    release_panns: bool = True,
    speaker_clustering: str | None = None,
) -> tuple[
    str,
    list[dict],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float, str]],
    VoiceprintCapture | None,
]:
    """Run separation -> song skip -> VAD chunking -> ASR -> alignment.

    Returns ``(iso_language, word_segments, vad_spans, sing_spans, speaker_turns,
    voiceprint_capture)``. The original five positions remain unchanged.
    vad_spans are the original-audio speech intervals, persisted to JSON for gap
    splitting. ``keep_lyrics`` runs song detection but skips excision: sung regions go
    through ASR/alignment like dialogue, and the detected singing spans come back so
    :func:`process` can flag lyric cues (empty unless keep_lyrics). ``diarize`` runs
    pyannote on the separated-vocals wav and returns the speaker turns (empty unless
    set). With ``voiceprints`` the per-speaker centroids come from the embedder
    ``voiceprint_model`` resolves to for the detected language (see
    :func:`voxweave.voiceembed.resolve_voiceprint_model`); only its ``pyannote``
    legacy lane reads the diarization pipeline's own embeddings; its checkpoint(s)
    are fetched and verified before any audio work, and if that fails voiceprints
    are off for the run (warning) while everything else proceeds.
    ``speaker_clustering`` picks how diarization groups its turns into speakers
    (:func:`voxweave.config.resolve_diarize_clustering`; resolved and validated
    before any audio work). All models run
    in-process (weights are fetched once into the voxweave cache). smart_split and
    file writing are handled by :func:`process`.

    ``release_panns=False`` keeps the PANNs singleton resident on return: the caller
    has a second detection pass queued (the ``--sdh`` sidecar tags the ORIGINAL mix
    with the same model) and would otherwise pay a reload. That caller owns the
    release. Every other singleton this function loads is released before it returns.
    """
    media_path = Path(media_path)
    rep = reporter or Reporter()
    if lang_override and lang_override.strip():
        # transcribe_chunks checks again, but only after separation and VAD.
        to_asr_iso(lang_override)
    if diarize:
        # Fail on a bad env/conf value now, not after separation and ASR.
        speaker_clustering = config.resolve_diarize_clustering(speaker_clustering)
    dbg: DebugSink = DebugSink()
    if debug:
        if debug_root is None:
            debug_root = artifacts.claim_paths(media_path).debug
        dbg = FileDebugSink(root=debug_root)
    tmp: list[
        Path
    ] = []  # intermediate files (fullband/vocals/16k/32k wav), deleted at end
    tmp_chunks: list[Path] = []
    # Set only on a successful return with release_panns=False: a caller can inherit
    # the loaded model, but never as the fallout of a failed run.
    panns_handoff = False
    try:
        rep.step("prepare audio")
        if voiceprints:
            # Idempotent after process()'s own prefetch: a cached checkpoint is
            # only re-verified. A failure turns voiceprints off for this run.
            voiceprints = _prefetch_voiceprint_models(
                voiceprint_model, lang_override, rep
            )
        if separate and voiceprints:
            require_sha256(source_fingerprint, "capture media fingerprint")

        def dump_separation(fullband: Path, vocals: Path) -> None:
            dbg.audio("00_fullband_44k.wav", fullband)
            dbg.audio("01_vocals.flac", vocals)
            log.info("separated vocals (local Roformer)")

        acquired = acquire_16k(
            media_path,
            separate=separate,
            normalize=normalize,
            reporter=rep,
            tmp=tmp,
            cache=cache_vocals,
            # A capture binds the cache to the snapshot it separated; otherwise
            # the duration check against the source decides.
            fingerprint=(source_fingerprint or "") if voiceprints else None,
            purpose="capture",
            on_separated=dump_separation,
            want_voc32=skip_songs or keep_lyrics,
        )
        wav = acquired.wav
        fullband = acquired.fullband
        voc32 = acquired.voc32  # 32k mono vocals: PANNs input + cache source
        separator_identity = acquired.separator
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
                    rep.step("detect songs")
                    rep.stage("song detection (PANNs)")
                    song_spans, sing_spans, speech_spans = detect_song_spans(
                        voc32, progress=progress_bridge(rep, "song detection (PANNs)")
                    )
                    if song_spans:
                        log.info(
                            "song spans: %s",
                            [(round(a, 1), round(b, 1)) for a, b in song_spans],
                        )
                except ModuleNotFoundError as e:
                    # panns-inference (or one of its deps) is missing from a broken
                    # environment: it is a core dependency, so reinstalling fixes it.
                    log.warning(
                        "song detection unavailable (missing module: %s) -- continuing "
                        "without song skip; panns-inference is a core dependency, so "
                        "reinstall voxweave, or pass --no-skip-songs (and drop "
                        "--keep-lyrics) to silence this",
                        e,
                    )
        if release_panns:
            # Last PANNs consumer of this job is done (an --sdh run defers this to
            # its own pass). Drop the ~300MB before the ASR/aligner weights load, so
            # the two never share the card. No-op when detection never ran.
            songdet.release_model()

        rep.step("find speech")
        rep.stage("VAD chunking")
        segs = vad_speech_segments(wav)
        plan = _plan_chunks(
            wav,
            segs,
            song_spans=song_spans,
            sing_spans=sing_spans,
            speech_spans=speech_spans,
            keep_lyrics=keep_lyrics,
        )
        chunks = plan.chunks
        segs = plan.segs
        song_spans = plan.song_spans
        if not chunks:
            raise RuntimeError(f"no speech detected in {media_path.name}")

        rep.step("transcribe and align")
        rep.stage("load ASR/alignment models")
        # Slice all chunk waveforms upfront so dual-pass (full ASR -> release -> full
        # alignment) can shave VRAM peak.
        cwavs: list[Path] = []
        for ch in chunks:
            cwav = slice_wav(wav, ch["start"], ch["end"])
            tmp_chunks.append(cwav)
            cwavs.append(cwav)
        strategy = config.conf_load_strategy()
        rep.chunks(len(chunks) * backend.chunk_pass_count(asr_model))
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
            # final excised song intervals: muted in the full-pass waveform so mid-file
            # songs (which survive the envelope crop) cannot host smeared sentence
            # fragments. Empty under --keep-lyrics (songs stay transcribed -> unmuted).
            song_spans=song_spans or None,
        )
        # reinject_punct runs after language resolution (tokenization must match iso),
        # so punctuation cannot be reinjected per-chunk.
        chunk_pairs: list[tuple[str, list[dict]]] = []
        detected: list[str] = []  # per-chunk detected language (debug meta only)
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
                    text=text,
                    lang=det_lang,
                    units=None,
                )
                continue
            if det_lang:
                detected.append(det_lang)
            dbg.chunk(
                idx,
                wav=cwav,
                start=ch["start"],
                end=ch["end"],
                text=text,
                lang=det_lang,
                units=units,
            )
            chunk_pairs.append((text, shift_units(units, ch["offset"])))

        if not chunk_pairs:
            raise RuntimeError(f"no aligned units for {media_path.name}")

        # Transcript-content weighting lets long dialogue dominate without
        # trusting the aligner whose language choice we are validating.
        lang_name = _select_transcript_language(results, lang_override)
        if not is_supported(lang_name):
            log.warning(
                "language %r is not officially supported; subtitle line breaking may"
                " be poor — pass --language to override detection",
                lang_name,
            )
        iso = to_iso_or(lang_name, "en")

        # Aligner strips punctuation; reinject_punct reattaches it by time so smart_split
        # can use it for sentence breaking and space insertion.
        all_units: list[dict] = []
        for txt, u in chunk_pairs:
            all_units.extend(realign.reinject_punct(txt, u, iso))
        if not all_units:
            raise RuntimeError(f"no aligned units for {media_path.name}")
        vad_spans = _timing_reference_spans(
            media_path, fullband, plan, speech_spans, separate=separate, tmp=tmp
        )
        # Qwen aligner has no CTC blank token, so word durations bleed into silence.
        # position_units_with_vad carves true gaps, giving smart_split an accurate signal.
        # Routed through the sink so --debug also records the pre/post snapshots and the
        # zero-duration repair accounting; the no-op sink calls the pass unchanged.
        all_units = dbg.position_units(all_units, vad_spans, language=lang_name)
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
                # Wall-clock seconds per step entered so far (the open step counts up
                # to now); later steps (speaker identification, layout) are not in it.
                "timings": dict(rep.timings()),
            }
        )
        speaker_turns: list[tuple[float, float, str]] = []
        voiceprint_capture: VoiceprintCapture | None = None
        if diarize:
            speaker_turns, voiceprint_capture = _identify_speakers(
                wav,
                iso,
                _DiarizeSettings(
                    diarize_model, speaker_clustering, min_speakers, max_speakers
                ),
                voiceprints=voiceprints,
                voiceprint_model=voiceprint_model,
                audio_profile=_diarize_audio_profile(
                    separate=separate,
                    normalize=normalize,
                    separator_identity=separator_identity,
                ),
                rep=rep,
            )
        panns_handoff = not release_panns
        return (
            iso,
            all_units,
            vad_spans,
            sing_spans if keep_lyrics else [],
            speaker_turns,
            voiceprint_capture,
        )
    finally:
        # Each release is guarded on its own: a release that raises (e.g.
        # torch.cuda.empty_cache after a CUDA fault) must neither mask the original
        # error nor skip the remaining releases and the temp-file cleanup below.
        # Release ASR/alignment singleton VRAM (separation self-releases earlier).
        try:
            backend.release()
        except Exception as e:
            log.warning("releasing the ASR/alignment models failed: %r", e)
        # No VAD pass can follow: every vad_speech_segments call lives above.
        try:
            chunking.release_silero_vad()
        except Exception as e:
            log.warning("releasing the VAD model failed: %r", e)
        if not panns_handoff:
            # Safety net: an exception before the post-detection release (or before
            # the sdh caller can take over) would otherwise strand PANNs on the card.
            # Idempotent when the release above already ran.
            try:
                songdet.release_model()
            except Exception as e:
                log.warning("releasing the song-detection model failed: %r", e)
        for p in tmp:
            p.unlink(missing_ok=True)
        for c in tmp_chunks:
            c.unlink(missing_ok=True)
