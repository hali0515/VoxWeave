from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from voxweave import config

log = logging.getLogger("voxweave")

# PANNs Cnn14 training SR — must feed 32k; 16k input causes a sample-rate mismatch.
SR = 32000

# Explicit local path wins; otherwise pulled from HF (-> config.AUDIO_CACHE) instead of
# panns_inference's default Zenodo ~/panns_data, keeping all weights under the HF cache root.
PANNS_CKPT = os.path.expanduser(os.environ.get("VOXWEAVE_PANNS_CKPT", ""))
PANNS_REPO = os.environ.get("VOXWEAVE_PANNS_REPO", "thelou1s/panns-inference")
PANNS_REPO_FILE = os.environ.get("VOXWEAVE_PANNS_REPO_FILE", "Cnn14_mAP=0.431.pth")
# panns_inference.config reads ~/panns_data/class_labels_indices.csv at import time; canonical source.
PANNS_LABELS_FILE = "class_labels_indices.csv"
PANNS_LABELS_URL = os.environ.get(
    "VOXWEAVE_PANNS_LABELS_URL",
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
)

# AudioSet class indices (class_labels_indices.csv)
IDX_SPEECH = [0, 4, 6, 7]  # Speech, Conversation, Babbling, Speech synthesizer
IDX_SING = [27, 28, 29, 30, 31, 36, 37, 254, 255, 266]  # Singing/Choir/Chant/Rap/...
IDX_MUSIC = [137, 267, 268, 270]  # Music / Background / Theme / Soundtrack

# Tuned on Yofukashi no Uta ep1 separated vocals.
# Either branch (sing or music) hitting its threshold counts as "song/music".
SING_MIN = 0.15
MUSIC_MIN = 0.30
SPEECH_MAX = 0.25
# Clean-dialogue signature on separated vocals: speech dominant, almost no singing/instrumental
# residue. Used during span expansion to trim dialogue flush against the song-core edge rather
# than absorb it. Interior rap verses still carry rhythmic residue (sing >= SING_QUIET_MAX or
# music >= MUSIC_QUIET_MAX), so they do NOT match and are still absorbed (preserves pit-2 protection).
SPEECH_CLEAN_MIN = 0.5
SING_QUIET_MAX = 0.10
MUSIC_QUIET_MAX = 0.20
WIN_SEC = 2.0
HOP_SEC = 1.0
GAP_MERGE_SEC = 2.0
MIN_SPAN_SEC = 3.0
BLOCK_GAP_SEC = 3.0  # adjacent VAD segments within this gap are treated as one voiced block (for span expansion)
# Song-core clustering: song spans within this gap of a long expandable span are the same song
# (tolerates rap/instrumental interludes between sung windows — e.g. a 3s sung OP intro sitting
# ~12s before the chorus). A short song span farther than this from any long core is an isolated
# sting embedded in dialogue; it must NOT stop the edge trim (else it anchors the whole dialogue
# tail into the song). Bounds the two known cases: yofukashi rap-OP intro (~12s) is protected,
# an Isekai in-dialogue sting (~49s from the OP) is isolated.
SONG_CORE_MERGE_SEC = float(os.environ.get("VOXWEAVE_SONG_CORE_MERGE_SEC", "15.0"))
# Intra-segment excision: PANNs span edges are coarse (2s windows), so cut points are
# snapped into the nearest real silence within SNAP_SEC (fine-VAD gaps) — the song goes
# out together with its flanking silence, and dialogue words are never bisected.
SNAP_SEC = 1.5
MIN_KEEP_SEC = 0.4  # excised remainders shorter than this are noise shards, dropped
# Speech rescue: silero occasionally scores real dialogue far below threshold for many seconds
# (observed: a 14s theatrical cold open at speech-prob < 0.25 while PANNs scores the same
# separated audio 0.66-0.83 Speech). A PANNs clean-dialogue stretch of at least this length
# with NO waveform-VAD coverage is a genuine miss and is rescued into the chunk stream;
# shorter remainders are inter-sentence pauses (PANNs 2s windows blur them) and stay out.
SPEECH_RESCUE_MIN_SEC = float(os.environ.get("VOXWEAVE_SPEECH_RESCUE_MIN_S", "3.0"))

_model = None  # AudioTagging singleton — lazy-loaded, reused within the process


def _resolve_panns_ckpt() -> str:
    """Return local path to Cnn14 checkpoint; explicit env path wins, else download from HF (cached)."""
    if PANNS_CKPT and os.path.exists(PANNS_CKPT):
        return PANNS_CKPT
    from voxweave.runtime import _hf_download

    return _hf_download(PANNS_REPO, PANNS_REPO_FILE, cache_dir=config.AUDIO_CACHE)


def _ensure_panns_labels() -> None:
    """Pre-place ~/panns_data/class_labels_indices.csv so importing panns_inference never shells out
    to `wget` (its config.py wgets this file at import time — wget is absent on macOS). Pulls from the
    same HF repo as the checkpoint when available, else the canonical AudioSet URL via urllib."""
    dst = Path.home() / "panns_data" / PANNS_LABELS_FILE
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:  # prefer HF (cached under AUDIO_CACHE, same source as the checkpoint)
        import shutil

        from voxweave.runtime import _hf_download

        src = _hf_download(PANNS_REPO, PANNS_LABELS_FILE, cache_dir=config.AUDIO_CACHE)
        shutil.copyfile(src, dst)
        return
    except Exception:  # noqa: BLE001 -- repo may not host the csv; fall back to the canonical URL
        pass
    import urllib.request

    with urllib.request.urlopen(PANNS_LABELS_URL) as r:  # noqa: S310 -- fixed trusted host
        data = r.read()
    dst.write_bytes(data)
    log.info("downloaded PANNs labels -> %s", dst)


def _get_model():
    global _model
    if _model is None:
        _ensure_panns_labels()  # must precede the import: panns_inference.config reads the csv eagerly
        import torch
        from panns_inference import AudioTagging

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Pass checkpoint explicitly so panns_inference never falls back to its ~/panns_data Zenodo download.
        ckpt = _resolve_panns_ckpt()
        _model = AudioTagging(checkpoint_path=ckpt, device=device)
        log.info("PANNs Cnn14 loaded on %s (%s)", device, ckpt)
    return _model


def reduce_scores(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, 527) probabilities → ``(speech, sing, music)`` per-window maxima.

    Extracted so tests can store just these three small arrays and drive
    ``*_from_scores`` variants without running PANNs or a GPU."""
    return (
        probs[:, IDX_SPEECH].max(axis=1),
        probs[:, IDX_SING].max(axis=1),
        probs[:, IDX_MUSIC].max(axis=1),
    )


def song_flags_from_scores(
    speech: np.ndarray,
    sing: np.ndarray,
    music: np.ndarray,
    *,
    sing_min: float = SING_MIN,
    music_min: float = MUSIC_MIN,
    speech_max: float = SPEECH_MAX,
) -> np.ndarray:
    """Three per-window score arrays → boolean: True = song/music. Pure function (core criterion of song_flags)."""
    return ((sing > speech) & (sing > sing_min)) | (
        (music > music_min) & (speech < speech_max)
    )


def sing_flags_from_scores(
    speech: np.ndarray,
    sing: np.ndarray,
    music: np.ndarray,
    *,
    sing_min: float = SING_MIN,
) -> np.ndarray:
    """Three per-window score arrays → boolean: True = contains singing (sing branch only, excludes pure instrumental). Pure function."""
    return (sing > speech) & (sing > sing_min)


def song_flags(
    probs: np.ndarray,
    *,
    sing_min: float = SING_MIN,
    music_min: float = MUSIC_MIN,
    speech_max: float = SPEECH_MAX,
) -> np.ndarray:
    """Per-window (n, 527) probabilities → boolean array: True = song/music, False = speech/silence. Pure function."""
    return song_flags_from_scores(
        *reduce_scores(probs),
        sing_min=sing_min,
        music_min=music_min,
        speech_max=speech_max,
    )


def sing_flags(
    probs: np.ndarray,
    *,
    sing_min: float = SING_MIN,
) -> np.ndarray:
    """(n, 527) → boolean: True = singing/rap/chant dominant (excludes pure instrumental).

    Only spans that pass this test trigger voiced-block expansion (to catch rap verses
    PANNs misclassifies as Speech). Pure-instrumental BGM (music dominant, sing~0) does
    NOT trigger expansion, preventing adjacent dialogue from being absorbed."""
    return sing_flags_from_scores(*reduce_scores(probs), sing_min=sing_min)


def speech_flags_from_scores(
    speech: np.ndarray,
    sing: np.ndarray,
    music: np.ndarray,
    *,
    speech_min: float = SPEECH_CLEAN_MIN,
    sing_max: float = SING_QUIET_MAX,
    music_max: float = MUSIC_QUIET_MAX,
) -> np.ndarray:
    """Per-window scores → boolean: True = clean dialogue (speech dominant, minimal singing/instrumental).

    Used during expansion edge-trimming: genuine dialogue (sing~0/music~0.03) at the song
    boundary is trimmed rather than absorbed. Interior rap verses carry residue and do NOT
    match, so they are still absorbed (preserves pit-2 protection)."""
    return (speech > speech_min) & (sing < sing_max) & (music < music_max)


def speech_flags(
    probs: np.ndarray,
    *,
    speech_min: float = SPEECH_CLEAN_MIN,
    sing_max: float = SING_QUIET_MAX,
    music_max: float = MUSIC_QUIET_MAX,
) -> np.ndarray:
    """Per-window (n, 527) → boolean: True = clean dialogue. Pure function. See speech_flags_from_scores."""
    return speech_flags_from_scores(
        *reduce_scores(probs),
        speech_min=speech_min,
        sing_max=sing_max,
        music_max=music_max,
    )


def merge_spans(
    flags: np.ndarray,
    starts: list[float],
    *,
    win_sec: float = WIN_SEC,
    gap_merge: float = GAP_MERGE_SEC,
    min_span: float = MIN_SPAN_SEC,
) -> list[tuple[float, float]]:
    """Consecutive flagged windows → time spans [(start, end)]; gaps <= gap_merge are merged, spans shorter than min_span are dropped. Pure function."""
    spans: list[list[float]] = []
    cur: list[float] | None = None
    for flag, t in zip(flags, starts, strict=True):
        if not flag:
            continue
        if cur is None:
            cur = [t, t + win_sec]
        elif t - cur[1] <= gap_merge:
            cur[1] = t + win_sec
        else:
            spans.append(cur)
            cur = [t, t + win_sec]
    if cur is not None:
        spans.append(cur)
    return [(a, b) for a, b in spans if b - a >= min_span]


def _snap_cut(c: float, silences: list[tuple[float, float]], snap_sec: float) -> float:
    """Snap a cut point into the nearest real silence within ±snap_sec; return c unchanged if none.

    A point already inside a silence stays put. Otherwise the candidate is the silence
    midpoint pulled into the snap window (long silences cut at the window edge — still
    inside the silence, just closer to the song). Pure function."""
    best, bd = c, None
    for sa, sb in silences:
        if sa <= c <= sb:
            return c  # already in silence — a fine cut point as-is
        p = min(max((sa + sb) / 2.0, c - snap_sec), c + snap_sec)
        if p < sa or p > sb:
            continue  # silence out of snapping reach
        d = abs(p - c)
        if bd is None or d < bd:
            best, bd = p, d
    return best


def excise_spans_from_segments(
    segments: list[dict],
    spans: list[tuple[float, float]],
    *,
    silences: list[tuple[float, float]] | None = None,
    snap_sec: float = SNAP_SEC,
    min_keep_sec: float = MIN_KEEP_SEC,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Cut song spans OUT of VAD segments instead of dropping whole segments.

    A VAD segment often mixes dialogue and song ("speech, brief pause, a hummed bar,
    speech again" — silero does not split on sub-threshold pauses). Whole-segment
    dropping loses the dialogue; whole-segment keeping feeds the song to ASR. This
    excises only the song interval: each segment is split into the parts outside the
    spans, so flanking dialogue survives with its own timestamps.

    Span edges come from 2s PANNs windows, so each cut point is snapped into the nearest
    real silence (``silences`` from a fine VAD pass, see chunking.silence_gaps) within
    ``snap_sec`` — the song leaves together with its flanking silence and dialogue words
    are never bisected. With no silence in reach the span edge is used as-is. Remainders
    shorter than ``min_keep_sec`` are dropped as shards.

    Returns ``(kept_segments, cut_spans)``: cut_spans are the snapped intervals actually
    excised (one per input span, sorted/merged) — use these for chunk grouping and as the
    final song spans so downstream holes match what was really removed. Pure function.
    """
    if not spans or not segments:
        return segments, list(spans)
    sil = silences or []
    cuts: list[tuple[float, float]] = []
    for a, b in spans:
        ca, cb = _snap_cut(a, sil, snap_sec), _snap_cut(b, sil, snap_sec)
        if cb <= ca:  # snapping inverted a brief span — keep the raw edges
            ca, cb = a, b
        cuts.append((ca, cb))
    cuts.sort()
    merged: list[list[float]] = [list(cuts[0])]
    for a, b in cuts[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cut_spans = [(a, b) for a, b in merged]

    kept: list[dict] = []
    for seg in segments:
        pieces = [(seg["start"], seg["end"])]
        for ca, cb in cut_spans:
            nxt: list[tuple[float, float]] = []
            for s, e in pieces:
                if ca >= e or cb <= s:  # no overlap with this cut
                    nxt.append((s, e))
                    continue
                if s < ca:
                    nxt.append((s, ca))
                if cb < e:
                    nxt.append((cb, e))
            pieces = nxt
        for s, e in pieces:
            if e - s >= min_keep_sec:
                kept.append({**seg, "start": s, "end": e})
    return kept, cut_spans


def filter_short_spans(
    spans: list[tuple[float, float]], *, min_sec: float
) -> list[tuple[float, float]]:
    """Drop song spans shorter than ``min_sec``.

    Real OP/ED segments (30-90s) far exceed any sane threshold. A brief BGM burst that
    slips through would cause group_segments_by_spans to split the audio there, turning
    the second half into a BGM-dominated chunk that breaks Qwen ASR recall. Apply after
    voiced-block expansion (singing OPs are already stretched long by then)."""
    return [(a, b) for (a, b) in spans if b - a >= min_sec]


def _overlaps(seg: dict, spans: list[tuple[float, float]]) -> bool:
    return any(max(seg["start"], a) < min(seg["end"], b) for a, b in spans)


def expand_spans_to_voiced_blocks(
    segments: list[dict],
    spans: list[tuple[float, float]],
    *,
    expandable: list[tuple[float, float]] | None = None,
    protect: list[tuple[float, float]] | None = None,
    block_gap: float = BLOCK_GAP_SEC,
) -> list[tuple[float, float]]:
    """Absorb entire voiced blocks that overlap a song span; return expanded and merged spans.

    Adjacent VAD segments within block_gap form one voiced block. OP/ED sequences are often
    a continuous "rap verse -> singing chorus" block: PANNs detects only the chorus, but
    anchoring on it pulls the whole block (rap included) into the song span.

    ``expandable`` (default None = all spans may expand) restricts which spans trigger
    whole-block absorption — typically singing spans from :func:`sing_flags`. Pure-instrumental
    BGM spans are excluded here; without this, a BGM cue followed immediately by speech
    (gap < block_gap) would absorb the speech and drop its subtitles.

    ``protect`` (default None = no trimming) lists clean-dialogue spans (see
    :func:`speech_flags`): before absorbing a block, dialogue segments at the leading/trailing
    edges are trimmed inward until a non-dialogue segment is hit. Interior segments (rap verses
    between song windows) are NOT trimmed and are still absorbed (preserves pit-2 protection).
    The song **core** (an expandable OP/ED span plus any song span contiguous with it, gap
    <= block_gap) is always preserved; trimming only affects what block expansion adds. The
    trim stop keys on that core, NOT on every ``spans`` entry: a brief embedded sting (a short,
    non-expandable song span sitting inside a dialogue block far from any core — e.g. a 3s
    musical hit) must not anchor the whole dialogue tail into the song. Such stings are trimmed
    through here and re-covered by the short-span excision path (``plan_song_skip`` short_sing).
    A short sung intro flush against a long core stays part of the core and still stops the trim.
    """
    if not spans or not segments:
        return spans
    exp = spans if expandable is None else expandable
    prot = protect or []

    blocks: list[list[dict]] = [[segments[0]]]
    for s in segments[1:]:
        if s["start"] - blocks[-1][-1]["end"] <= block_gap:
            blocks[-1].append(s)
        else:
            blocks.append([s])

    # Song "cores": song spans clustered within SONG_CORE_MERGE_SEC, keeping the MEMBER spans of
    # clusters that reach a long expandable span — NOT the cluster hull. A hull would also cover
    # the gaps BETWEEN member spans, and clean dialogue sitting in such a gap (e.g. a line between
    # the OP and a title-card sting 12s later) must stay trimmable. A short sung intro/outro near
    # a long OP/ED core is part of the song and must stop the edge trim; an isolated sting
    # embedded in dialogue (no long core within SONG_CORE_MERGE_SEC) is NOT a core, so the trim
    # passes through it — the dialogue on the far side is freed and the sting is re-covered by
    # plan_song_skip's short_sing excision.
    cores: list[tuple[float, float]] = []
    if exp:

        def _touches_exp(cluster: list[tuple[float, float]]) -> bool:
            return any(max(a, ea) < min(b, eb) for a, b in cluster for ea, eb in exp)

        cluster: list[tuple[float, float]] = []
        cluster_end = 0.0
        for a, b in sorted(spans):
            if cluster and a - cluster_end <= SONG_CORE_MERGE_SEC:
                cluster.append((a, b))
                cluster_end = max(cluster_end, b)
            else:
                if cluster and _touches_exp(cluster):
                    cores.extend(cluster)
                cluster = [(a, b)]
                cluster_end = b
        if cluster and _touches_exp(cluster):
            cores.extend(cluster)

    def _clean_speech(seg: dict) -> bool:
        # Never trim a segment on a song core (expandable OP/ED + its contiguous short spans),
        # even if its speech score is high; trim clean dialogue everywhere else.
        return _overlaps(seg, prot) and not _overlaps(seg, cores)

    out = list(spans)
    for blk in blocks:
        ba, bb = blk[0]["start"], blk[-1]["end"]
        if not any(max(ba, a) < min(bb, b) for a, b in exp):
            continue
        lo, hi = 0, len(blk) - 1
        while lo <= hi and _clean_speech(blk[lo]):
            lo += 1
        while hi >= lo and _clean_speech(blk[hi]):
            hi -= 1
        if lo > hi:  # entire block is clean dialogue — do not absorb (defensive)
            continue
        out.append((blk[lo]["start"], blk[hi]["end"]))

    out.sort()
    merged: list[list[float]] = [list(out[0])]
    for a, b in out[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def group_segments_by_spans(
    segments: list[dict], spans: list[tuple[float, float]]
) -> list[list[dict]]:
    """Split speech segments into groups at song-span boundaries.

    Breaks whenever a song span falls between two adjacent segments, ensuring the
    contiguous [start, end] interval of each packed chunk does not cross a song span.
    Without this, slice_wav's contiguous cut would pull the skipped song back into the
    audio fed to ASR (skipping a span is not the same as excising it from the waveform).
    """
    if not spans or not segments:
        return [segments] if segments else []
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for seg in segments:
        if cur:
            gap_a, gap_b = cur[-1]["end"], seg["start"]
            if any(max(gap_a, a) < min(gap_b, b) for a, b in spans):
                groups.append(cur)
                cur = []
        cur.append(seg)
    if cur:
        groups.append(cur)
    return groups


def subtract_spans(
    spans: list[tuple[float, float]], keep: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Interval subtraction: ``spans`` minus every interval in ``keep`` (both sorted).

    Used to carve clean-dialogue windows out of expanded song spans before they
    are subtracted from the VAD timing reference: dialogue spoken OVER a song
    must survive in vad_speech or the emission mask forbids its true location.
    """
    out: list[tuple[float, float]] = []
    for a, b in spans:
        cur = a
        for ka, kb in keep:
            if kb <= cur:
                continue
            if ka >= b:
                break
            if ka > cur:
                out.append((cur, ka))
            cur = max(cur, kb)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def rescue_speech_segments(
    speech_spans: list[tuple[float, float]],
    segs: list[dict],
    *,
    min_sec: float = SPEECH_RESCUE_MIN_SEC,
) -> list[dict]:
    """Synthetic segments for PANNs clean-dialogue stretches the waveform VAD missed.

    silero can score real dialogue far below threshold for many seconds (theatrical
    delivery; observed on a 14s cold open) while PANNs Speech on the same separated
    vocals is unambiguous. Remainders of ``speech_spans`` not covered by ``segs`` that
    are at least ``min_sec`` long become rescue segments so the chunk stream (and thus
    ASR) covers them. Sub-``min_sec`` remainders are inter-sentence pauses — PANNs 2s
    windows blur across them — and are NOT rescued (they would only reshape voiced
    blocks and chunk packing without adding speech).
    """
    cover = [(s["start"], s["end"]) for s in segs]
    return [
        {"start": a, "end": b}
        for a, b in subtract_spans(speech_spans, cover)
        if b - a >= min_sec
    ]


def window_probs(
    wav_path: Path, *, batch: int = 32, progress=None
) -> tuple[np.ndarray, list[float]] | None:
    """Run PANNs over ``wav_path`` (32 kHz mono) in sliding windows.

    Returns ``(probs, starts_sec)`` — the full ``(n_windows, 527)`` AudioSet
    probability matrix and each window's start time — or None when the audio is
    shorter than one window. Shared by song detection (separated vocals input)
    and SDH event detection (original-mix input); ``progress(done, total)`` is
    optional, called per batch.
    """
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == SR, f"expected {SR} Hz, got {sr!r} — decode_to_wav(sample_rate={SR})"

    win, hop = int(WIN_SEC * SR), int(HOP_SEC * SR)
    if len(data) < win:
        return None
    starts_idx = list(range(0, len(data) - win + 1, hop))
    wins = np.stack([data[s : s + win] for s in starts_idx])

    model = _get_model()
    batch_starts = list(range(0, len(wins), batch))
    nb = len(batch_starts)
    probs = []
    for bi, i in enumerate(batch_starts):
        out, _ = model.inference(wins[i : i + batch])
        probs.append(out)
        if progress is not None:
            progress(bi + 1, nb)
    return np.concatenate(probs), [s / SR for s in starts_idx]


def detect_song_spans(
    wav_path: Path, *, batch: int = 32, progress=None
) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]
]:
    """Run PANNs on a 32 kHz mono separated-vocals WAV.

    Returns ``(song/music spans, singing spans, clean-dialogue spans)``:
    - song/music spans: used to drop VAD segments.
    - singing spans (subset): spans with human vocals; only these trigger voiced-block
      expansion. Pure-instrumental BGM is absent here, so it never swallows adjacent dialogue.
    - clean-dialogue spans: trimmed from voiced-block boundaries during expansion rather
      than absorbed (see :func:`expand_spans_to_voiced_blocks`).

    Input must be separated vocals (route ii): instruments stripped, so singing vs. speech
    scores are cleanly separated. ``progress(done, total)`` is optional, called per batch.
    """
    wp = window_probs(wav_path, batch=batch, progress=progress)
    if wp is None:
        return [], [], []
    P, starts_sec = wp
    spans = merge_spans(song_flags(P), starts_sec)
    sing_starts = [t for t, f in zip(starts_sec, sing_flags(P), strict=True) if f]
    sing_spans = [(a, b) for (a, b) in spans if any(a <= t < b for t in sing_starts)]
    speech_spans = merge_spans(speech_flags(P), starts_sec)
    log.info(
        "song-detect: %d span(s) (%d with singing), %.1fs total; %d clean-dialogue span(s)",
        len(spans),
        len(sing_spans),
        sum(b - a for a, b in spans),
        len(speech_spans),
    )
    return spans, sing_spans, speech_spans
