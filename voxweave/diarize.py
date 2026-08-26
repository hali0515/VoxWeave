"""Speaker diarization (pyannote) + Netflix speaker-aware cue formatting.

Detection runs once per file on the separated-vocals 16k wav (clean speech beats
the BGM mix for speaker embeddings) and persists ``speaker_turns`` to the
sibling JSON. Formatting is a pure post-pass over smart_split's cues: each
cue's atoms get a speaker by time overlap with the turns; a cue containing two
speakers becomes a Netflix dual-speaker event (one line per speaker, leading
hyphen, no space) when the language allows two lines and both halves fit one
line, otherwise the cue splits at the speaker boundaries. ``split`` replays
formatting from the persisted turns without re-running pyannote.

Model: ``pyannote/speaker-diarization-3.1`` (pyannote.audio 3.x; MIT, ~1.6 GB
VRAM). The checkpoint is HF-gated: accept the conditions on the model card and
provide a token (VOXWEAVE_HF_TOKEN / HF_TOKEN env, or ``hf_token`` in the conf).
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from voxweave import config
from voxweave.core.schema import Cue

if TYPE_CHECKING:
    from voxweave.core.smart_split import SplitThresholds

log = logging.getLogger("voxweave")

DIARIZE_MODEL = os.environ.get(
    "VOXWEAVE_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"
)

# Atom-level speaker assignment needs at least this much absolute overlap with a
# turn (seconds); below it the atom inherits its neighbors (guards 20ms grazes).
MIN_ATOM_OVERLAP_S = 0.05

# A speaker run whose total atom duration is below this is a candidate for
# absorption into an adjacent run (a word is never cut into two speaker cues).
# Whether a sub-floor run is actually absorbed is position-aware (see
# _absorb_tiny_runs): mid-phrase thrash goes, real edge utterances stay.
MIN_RUN_S = 0.2

# A sub-MIN_RUN_S run that is *not* a same-speaker (A-B-A) sandwich -- i.e. one at
# a cue edge or between two different speakers (A-B-C) -- is a real short
# utterance and is kept when at least this long; only shorter fragments are
# pyannote noise. Keeps e.g. a 160ms trailing '你好' as its own speaker cue while
# still dropping an 80ms mid-phrase fragment.
EDGE_RUN_MIN_S = 0.12

# A speaker boundary is also a cue boundary, so the left piece's tail is scored
# with the Level-2 (UniDic POS) end penalty smart_split uses for its own breaks.
# 2 = the tail cannot end an utterance at all (格助詞 with no head, 連体詞 with
# nothing to modify); splitting there is only acceptable when the cue has no
# cleaner edge to offer. ja only -- no validated equivalent grading exists for zh,
# and the Level-1 char table over-fires (準体の, adverbial に).
BAD_TAIL_PENALTY = 2

# Inter-run silence a bad-tail merge may span. A reply that starts after a beat is
# a genuine second speaker, whatever the left tail looks like -- swallowing it is
# the 2026-07-03 EDGE_RUN_MIN_S regression class. Only a boundary the diarizer
# placed inside continuous speech can plausibly be a mis-cut phrase, so the merge
# is confined to gaps below the shortest inter-speaker pause worth trusting
# (same order as MIN_RUN_S: shorter than any real turn-taking beat).
BAD_TAIL_MAX_GAP_S = 0.2

# Turn-list smoothing (raw pyannote turns are noisy: 16-31% run <0.5s, and
# overlap-track fragments sit fully inside another speaker's turn). Module
# constants, overridable via env.
DIARIZE_MERGE_GAP_S = (
    0.35  # merge consecutive same-speaker turns across a gap below this
)
DIARIZE_DROP_CONTAINED_S = (
    0.2  # drop turns shorter than this fully inside another speaker's turn
)

Turn = tuple[float, float, str]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v is not None and v.strip() else default
    except ValueError:
        return default


_pipeline = None  # pyannote Pipeline singleton -- lazy-loaded, released after use


def _ensure_torchaudio_compat() -> None:
    """Restore torchaudio symbols pyannote.audio 3.4 imports but 2.11 removed.

    ``pyannote/audio/core/io.py`` annotates a function ``-> torchaudio.AudioMetaData``
    at import time and calls ``torchaudio.info`` / ``torchaudio.list_audio_backends``
    for file-path inputs; torchaudio 2.11 dropped all three. We never touch
    torchaudio I/O at runtime (diarize_turns feeds a decoded waveform dict), but
    the import-time annotation still resolves the attribute, so the shims must
    exist before ``import pyannote.audio``.

    Each shim installs only if missing (hasattr guard) so it is idempotent and an
    older torchaudio that still ships the real symbols is left untouched.
    """
    import torchaudio

    ta: Any = torchaudio
    if not hasattr(ta, "AudioMetaData"):

        @dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 0
            encoding: str = ""

        ta.AudioMetaData = AudioMetaData
    if not hasattr(ta, "info"):

        def info(filepath, *_a, **_k):
            import soundfile as sf

            meta = sf.info(str(filepath))
            return ta.AudioMetaData(
                sample_rate=int(meta.samplerate),
                num_frames=int(meta.frames),
                num_channels=int(meta.channels),
            )

        ta.info = info
    if not hasattr(ta, "list_audio_backends"):

        def list_audio_backends():
            return ["soundfile"]

        ta.list_audio_backends = list_audio_backends


def _load_pipeline(pipeline_cls, token: str):
    """Load the pyannote checkpoint under torch 2.11's weights_only default.

    torch >= 2.6 loads with ``weights_only=True``, which rejects the plain Python
    objects pyannote pickles into its checkpoints ("Unsupported global"). We
    allowlist exactly the classes the load names -- all in trusted torch /
    pyannote namespaces, discovered empirically against the official
    ``pyannote/speaker-diarization-3.1`` repo (a gated repo the user has accepted;
    the token is verified before we get here) -- via
    ``torch.serialization.safe_globals``. That context is a no-op on older torch
    that already defaults to ``weights_only=False``.
    """
    import torch
    from pyannote.audio.core.task import (  # pyright: ignore[reportMissingImports]
        Problem,
        Resolution,
        Specifications,
    )
    from torch.torch_version import TorchVersion

    allow = [TorchVersion, Specifications, Problem, Resolution]
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:  # torch < 2.4: weights_only already defaults to False
        return pipeline_cls.from_pretrained(
            DIARIZE_MODEL, use_auth_token=token, cache_dir=config.AUDIO_CACHE
        )
    with safe_globals(allow):
        return pipeline_cls.from_pretrained(
            DIARIZE_MODEL, use_auth_token=token, cache_dir=config.AUDIO_CACHE
        )


def _get_pipeline(token: str):
    global _pipeline
    if _pipeline is None:
        try:
            import torch

            _ensure_torchaudio_compat()  # must precede the pyannote import
            from pyannote.audio import (  # pyright: ignore[reportMissingImports]
                Pipeline,
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "diarization requires pyannote.audio (not installed); "
                "install with: pip install 'voxweave[diarize]'"
            ) from e
        pl = _load_pipeline(Pipeline, token)
        if pl is None:
            raise RuntimeError(
                f"could not load {DIARIZE_MODEL}: accept the user conditions on "
                f"https://hf.co/{DIARIZE_MODEL} (and its segmentation model) and "
                "set VOXWEAVE_HF_TOKEN / HF_TOKEN"
            )
        if torch.cuda.is_available():
            pl.to(torch.device("cuda"))
        _pipeline = pl
        log.info("loaded diarization pipeline %s", DIARIZE_MODEL)
    return _pipeline


def release() -> None:
    """Drop the pipeline singleton and free its VRAM (mirrors backend.release)."""
    global _pipeline
    if _pipeline is None:
        return
    _pipeline = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def diarize_turns(
    wav_path: Path,
    *,
    token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Run diarization over ``wav_path`` and return sorted ``(start, end, label)`` turns."""
    token = token or config.conf_hf_token()
    if not token:
        raise RuntimeError(
            f"diarization needs a Hugging Face token for the gated {DIARIZE_MODEL} "
            "checkpoint: accept the conditions on its model card, then either log "
            "in once with `hf auth login`, or set VOXWEAVE_HF_TOKEN / HF_TOKEN "
            "(or hf_token in ~/.config/voxweave.conf)"
        )
    import soundfile as sf
    import torch

    # pyannote's reproducibility guard force-disables TF32 process-wide and warns
    # whenever it finds it enabled (pyannote-audio#1370) -- which it always does,
    # because torch enables cudnn TF32 by default and the separator deliberately
    # opts into TF32 matmuls. Pre-comply for the diarization span so the guard
    # stays silent, then restore the process policy so pyannote cannot turn the
    # separator's TF32 off behind our back.
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        pl = _get_pipeline(token)
        kwargs: dict[str, int] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        # Feed a decoded waveform dict rather than a path: pyannote's file-path branch
        # goes through torchaudio.info/load, which torchaudio 2.11 broke. This dict
        # form is a first-class pyannote input and sidesteps its runtime audio I/O.
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (T, C)
        wav = torch.from_numpy(data).T.contiguous()  # (C, T)
        if wav.shape[0] > 1:  # defensive stereo downmix -> mono (1, T)
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(torch.float32)
        with warnings.catch_warnings():
            # pyannote's stat pooling hits std() on single-frame reductions for
            # very short speaker segments; the result is guarded internally and
            # the warning is pure noise on every episode.
            warnings.filterwarnings(
                "ignore", message=r"std\(\): degrees of freedom is <= 0"
            )
            warnings.filterwarnings(
                "ignore", message=r"TensorFloat-32 \(TF32\) has been disabled"
            )
            annotation = pl({"waveform": wav, "sample_rate": int(sr)}, **kwargs)
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    turns = [
        (float(seg.start), float(seg.end), str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns = _smooth_turns(turns)
    log.info(
        "diarization: %d turn(s), %d speaker(s)",
        len(turns),
        len({lb for _, _, lb in turns}),
    )
    return turns


def _smooth_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Smooth raw pyannote turns before persisting (pure, order-preserving).

    Two passes, both robust to noisy input:
    - drop turns shorter than ``VOXWEAVE_DIARIZE_DROP_CONTAINED_S`` that are fully
      contained inside a *different* speaker's turn (overlap-track fragments);
      standalone short interjections (not contained) are spared.
    - merge consecutive same-speaker turns separated by a gap below
      ``VOXWEAVE_DIARIZE_MERGE_GAP_S`` (a single speaker split by a micro-pause).

    Containment is tested against the original turn set, so dropping never depends
    on merge order. Clean input is returned unchanged.
    """
    if not turns:
        return []
    ordered = sorted(turns, key=lambda t: (t[0], t[1]))
    drop_s = _env_float("VOXWEAVE_DIARIZE_DROP_CONTAINED_S", DIARIZE_DROP_CONTAINED_S)
    merge_gap = _env_float("VOXWEAVE_DIARIZE_MERGE_GAP_S", DIARIZE_MERGE_GAP_S)
    kept: list[Turn] = []
    for i, (a, b, lb) in enumerate(ordered):
        if b - a < drop_s and any(
            olb != lb and oa <= a and b <= ob
            for j, (oa, ob, olb) in enumerate(ordered)
            if j != i
        ):
            continue
        kept.append((a, b, lb))
    merged: list[Turn] = []
    for a, b, lb in kept:
        if merged and merged[-1][2] == lb and a - merged[-1][1] < merge_gap:
            pa, pb, plb = merged[-1]
            merged[-1] = (pa, max(pb, b), plb)
        else:
            merged.append((a, b, lb))
    return merged


def _span_speaker(
    start: float | None, end: float | None, turns: Sequence[Turn]
) -> str | None:
    """Dominant speaker for a time span by accumulated overlap (whisperX pattern)."""
    if start is None or end is None or end <= start:
        return None
    overlap: dict[str, float] = {}
    for a, b, label in turns:
        if b <= start:
            continue
        if a >= end:
            break  # turns are sorted by start
        ov = min(end, b) - max(start, a)
        if ov > 0:
            overlap[label] = overlap.get(label, 0.0) + ov
    if not overlap:
        return None
    label, best = max(overlap.items(), key=lambda kv: kv[1])
    return label if best >= MIN_ATOM_OVERLAP_S else None


def _run_dur(atoms: list[dict]) -> float:
    """Total spoken duration of a run: sum of its atoms' positive durations."""
    total = 0.0
    for a in atoms:
        s, e = a.get("start"), a.get("end")
        if s is not None and e is not None and e > s:
            total += e - s
    return total


def _coalesce_runs(
    runs: list[tuple[str, list[dict]]],
) -> list[tuple[str, list[dict]]]:
    """Merge adjacent same-speaker runs (their atoms concatenate)."""
    out: list[tuple[str, list[dict]]] = []
    for lb, ats in runs:
        if out and out[-1][0] == lb:
            out[-1][1].extend(ats)
        else:
            out.append((lb, list(ats)))
    return out


def _absorbable(i: int, durs: list[float], runs: list[tuple[str, list[dict]]]) -> bool:
    """Whether run ``i`` (already known sub-``MIN_RUN_S``) should be absorbed.

    Position-aware policy:
    - A run sandwiched between two runs of the *same* speaker (A-B-A) is label
      thrash and is always absorbed, regardless of ``EDGE_RUN_MIN_S``.
    - A run at a cue edge, or between two *different* speakers (A-B-C), is a real
      short utterance: kept when >= ``EDGE_RUN_MIN_S``, absorbed only below it.
    """
    has_left = i > 0
    has_right = i + 1 < len(runs)
    sandwiched_same = has_left and has_right and runs[i - 1][0] == runs[i + 1][0]
    return sandwiched_same or durs[i] < EDGE_RUN_MIN_S


def _absorb_tiny_runs(
    runs: list[tuple[str, list[dict]]],
) -> list[tuple[str, list[dict]]]:
    """Fold absorbable sub-``MIN_RUN_S`` runs into their longer neighbor.

    Collapses A-B-A label thrash to a single run while keeping real short edge or
    A-B-C utterances (see ``_absorbable``). Repeats to a fixpoint: absorbing one
    run and re-coalescing can turn a surviving run into a new same-speaker
    sandwich, which the next pass then absorbs."""
    runs = [(lb, list(ats)) for lb, ats in runs]
    while len(runs) > 1:
        durs = [_run_dur(ats) for _, ats in runs]
        tiny = [
            (durs[i], i)
            for i in range(len(runs))
            if durs[i] < MIN_RUN_S and _absorbable(i, durs, runs)
        ]
        if not tiny:
            break
        _, i = min(tiny)  # shortest absorbable run first
        left = durs[i - 1] if i > 0 else -1.0
        right = durs[i + 1] if i + 1 < len(runs) else -1.0
        if left < 0 and right < 0:
            break
        if right > left:  # merge into the following (longer) run: prepend atoms
            runs[i + 1] = (runs[i + 1][0], runs[i][1] + runs[i + 1][1])
        else:  # merge into the preceding run: append atoms
            runs[i - 1] = (runs[i - 1][0], runs[i - 1][1] + runs[i][1])
        del runs[i]
        runs = _coalesce_runs(runs)
    return runs


def _snap_runs_to_phrases(
    runs: list[tuple[str, list[dict]]], lang: str
) -> list[tuple[str, list[dict]]]:
    """Snap run boundaries onto legal token edges so a run never cuts mid-word.

    No-space langs only (space-delimited atoms are already whole words). Each
    jieba/BudouX phrase is reassigned to its dominant speaker (by atom duration),
    which is exactly a boundary snap: a phrase spanning a mid-word label flip
    goes wholly to one speaker instead of splitting. Reuses the phrase-boundary
    machinery smart_split uses (``_phrase_boundary_atoms`` -> ``phrase_atoms``).
    """
    from voxweave.core.layout import _no_spaces

    if not _no_spaces(lang) or len(runs) < 2:
        return runs
    from voxweave.core.smart_split import _phrase_boundary_atoms

    flat = [a for _, ats in runs for a in ats]
    labels = [lb for lb, ats in runs for _ in ats]
    text = "".join(a["text"] for a in flat)
    boundaries = _phrase_boundary_atoms([{"text": a["text"]} for a in flat], text, lang)
    edges = sorted(boundaries | {0, len(flat)})
    new_labels = list(labels)
    for s, e in zip(edges, edges[1:]):
        weight: dict[str, float] = {}
        for k in range(s, e):
            weight[labels[k]] = weight.get(labels[k], 0.0) + _run_dur([flat[k]])
        if not weight:
            continue
        first = labels[s]
        best = max(weight, key=lambda lb: (weight[lb], lb == first))
        for k in range(s, e):
            new_labels[k] = best
    out: list[tuple[str, list[dict]]] = []
    for a, lb in zip(flat, new_labels):
        if out and out[-1][0] == lb:
            out[-1][1].append(a)
        else:
            out.append((lb, [a]))
    return out


def _run_gap(left: list[dict], right: list[dict]) -> float:
    """Silence between the end of ``left`` and the start of ``right``.

    ``inf`` when either side has no usable timestamp: an unmeasurable gap must
    never be read as "contiguous" by a caller that merges on contiguity.
    """
    end = left[-1].get("end") if left else None
    start = right[0].get("start") if right else None
    if end is None or start is None:
        return math.inf
    # +epsilon keeps BAD_TAIL_MAX_GAP_S the exclusive bound its comment documents:
    # 1.4 - 1.2 is 0.19999999999999996 in binary floating point, so without it a
    # nominally 0.2s gap slips under the ceiling and merges.
    return max(0.0, float(start) - float(end)) + 1e-9


def _ja_edge_penalties(flat: list[dict], edges: Sequence[int]) -> dict[int, int] | None:
    """Level-2 (UniDic POS) end penalty for each internal atom edge, or ``None``.

    Wired exactly like ``smart_split._attach_end_penalties``: the POS map is keyed
    by the non-space char offset of each token's LAST char, so the edge before
    atom ``e`` is scored at the cumulative offset of atom ``e - 1``.

    ``None`` means no Level-2 source (fugashi absent, or ``VOXWEAVE_JA_POS=0``).
    Unlike line breaking, this pass has no safe Level-1 fallback: the char table
    scores every trailing の/に 2, including 準体の (そうな|の) and the adverbial
    copula (そんな|に), and acting on those merges away a real speaker. So the
    caller must leave the boundary alone rather than degrade. Offsets the tagger
    does not end a token on (BudouX/MeCab disagreement) score 0 for the same
    reason.
    """
    from voxweave.core.kinsoku import ja_pos_end_penalties
    from voxweave.core.layout import _token_char_count

    pos = ja_pos_end_penalties("".join(a["text"] for a in flat), bound_tails_only=True)
    if pos is None:
        return None
    offsets: list[int] = []
    total = 0
    for a in flat:
        total += _token_char_count(a["text"])
        offsets.append(total)
    return {e: pos.get(offsets[e - 1] - 1, 0) for e in edges if 0 < e < len(flat)}


def _merge_bad_tail_runs(
    runs: list[tuple[str, list[dict]]], lang: str
) -> list[tuple[str, list[dict]]]:
    """Merge a speaker boundary that would strand a bound word on the left cue.

    ja only: the gate needs the Level-2 POS signal to tell a dangling 格助詞 from
    a legal clause end, and no validated equivalent grading exists for zh.

    Fires only when all three hold: the boundary's tail scores
    ``BAD_TAIL_PENALTY``, no other internal phrase edge of the cue would score
    below it, and the two runs are separated by less than ``BAD_TAIL_MAX_GAP_S``
    of silence. A cleaner edge means the boundary stays where the speaker signal
    put it (relocating it is a separate concern); an audible gap means the second
    run is a real reply, not a mis-cut phrase. The merged run keeps the longer
    speaker's label, the same duration comparison ``_absorb_tiny_runs`` uses.
    Duration alone never triggers this pass, so the tiny-run policy above is
    untouched.

    Repeats to a fixpoint: merging can join two runs of one speaker, and the
    re-coalesced neighbours form a boundary that was not there before.
    """
    if lang != "ja" or len(runs) < 2:
        return runs
    from voxweave.core.smart_split import _phrase_boundary_atoms

    runs = [(lb, list(ats)) for lb, ats in runs]
    while len(runs) > 1:
        flat = [a for _, ats in runs for a in ats]
        text = "".join(a["text"] for a in flat)
        edges = sorted(
            _phrase_boundary_atoms([{"text": a["text"]} for a in flat], text, lang)
            | {0, len(flat)}
        )
        pen = _ja_edge_penalties(flat, edges)
        if pen is None:
            break
        target = None
        cut = 0
        for i in range(len(runs) - 1):
            cut += len(runs[i][1])
            # An unknown cut is not a phrase edge at all -- treat it as clean and
            # leave it to _snap_runs_to_phrases rather than merging on a guess.
            if pen.get(cut, 0) < BAD_TAIL_PENALTY:
                continue
            if any(p < BAD_TAIL_PENALTY for e, p in pen.items() if e != cut):
                continue
            if _run_gap(runs[i][1], runs[i + 1][1]) >= BAD_TAIL_MAX_GAP_S:
                continue
            target = i
            break
        if target is None:
            break
        left, right = runs[target], runs[target + 1]
        label = right[0] if _run_dur(right[1]) > _run_dur(left[1]) else left[0]
        runs[target] = (label, left[1] + right[1])
        del runs[target + 1]
        runs = _coalesce_runs(runs)
    return runs


def _speaker_runs(
    atoms: list[dict], turns: Sequence[Turn], lang: str
) -> list[tuple[str, list[dict]]]:
    """Group a cue's atoms into consecutive same-speaker runs.

    Atoms without a confident speaker (no span / no overlap) inherit the current
    run; leading unassigned atoms join the first labeled run. Raw runs are then
    de-noised: sub-``MIN_RUN_S`` label thrash is absorbed into the longer
    neighbor, and surviving boundaries snap to jieba/BudouX phrase edges so a
    lexeme is never split across two speaker cues. Finally (ja only) a boundary
    that survives the snap but would leave the left cue ending on a bound word --
    with no cleaner edge available and no audible gap -- is merged away.
    """
    runs: list[tuple[str, list[dict]]] = []
    pending: list[dict] = []  # unassigned atoms before the first labeled one
    for atom in atoms:
        spk = _span_speaker(atom.get("start"), atom.get("end"), turns)
        if spk is None:
            (runs[-1][1] if runs else pending).append(atom)
            continue
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(atom)
        else:
            runs.append((spk, [atom]))
            if pending:
                runs[-1][1][:0] = pending
                pending = []
    if pending:  # no atom got a speaker at all
        return []
    if len(runs) <= 1:
        return runs
    runs = _absorb_tiny_runs(runs)
    runs = _snap_runs_to_phrases(runs, lang)
    return _merge_bad_tail_runs(runs, lang)


def _slice_text_by_runs(text: str, runs: list[tuple[str, list[dict]]]) -> list[str]:
    """Slice the cue's display text into one piece per run.

    Atoms cover exactly the text's non-space characters in order, so each run
    consumes its atoms' character count from the original string (interior
    spacing preserved, boundaries trimmed).
    """
    from voxweave.core.layout import _token_char_count

    pieces: list[str] = []
    i = 0
    for _, atoms in runs:
        need = sum(_token_char_count(a["text"]) for a in atoms)
        j = i
        seen = 0
        while j < len(text) and seen < need:
            if not text[j].isspace():
                seen += 1
            j += 1
        pieces.append(text[i:j].strip())
        i = j
    if i < len(text) and pieces:  # trailing slack (whitespace) sticks to the last piece
        pieces[-1] = (pieces[-1] + text[i:]).strip()
    return pieces


def _run_span(
    atoms: list[dict], fallback_start: float, fallback_end: float
) -> tuple[float, float]:
    starts = [s for a in atoms if (s := a.get("start")) is not None]
    ends = [e for a in atoms if (e := a.get("end")) is not None]
    return (
        float(min(starts)) if starts else fallback_start,
        float(max(ends)) if ends else fallback_end,
    )


def format_speaker_cues(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    max_line_length: int | None = None,
) -> list[Cue]:
    """Speaker-aware post-pass over smart_split's cues (pure, replayable).

    Single-speaker cues pass through. A two-speaker cue becomes one Netflix
    dual-speaker event (``-line\\n-line``, hyphen without space, one speaker per
    line) when the language renders two lines and both halves fit one line;
    otherwise — and for 3+ speakers — the cue splits at the speaker boundaries
    with word-accurate timing. Lyric cues pass through untouched (the music-note
    wrap owns that display).

    ``max_line_length`` is the same per-cue budget in native cells that
    smart_split used for this file (``None`` = the language default), so the dual
    gate and the piece re-wrap render for the configured player instead of the
    built-in 42-column profile.
    """
    if not turns:
        return cues
    from voxweave.core.layout import (
        _line_budget_width,
        _vis_width,
        default_max_line_length,
        default_max_lines,
        wrap_cue_text,
    )
    from voxweave.core.smart_split import _build_atoms

    max_lines = default_max_lines(lang)
    # Half-width cells -- the unit _vis_width and wrap_cue_text measure in.
    budget = _line_budget_width(
        default_max_line_length(lang) if max_line_length is None else max_line_length,
        lang,
    )
    dual_ok = max_lines >= 2
    out: list[Cue] = []
    for cue in cues:
        word_data = cue.get("word_data") or []
        if cue.get("lyric") or not word_data:
            out.append(cue)
            continue
        atoms = _build_atoms(cue["text"], cast(list, word_data), lang)
        runs = _speaker_runs(atoms, turns, lang)
        if len(runs) <= 1:
            out.append(cue)
            continue
        pieces = _slice_text_by_runs(cue["text"], runs)
        # Collapse each piece to one logical line before the dual-budget test:
        # smart_split may have soft-wrapped the cue, so a piece can carry an
        # interior "\n" that _vis_width would (wrongly) count as width 1, letting
        # a really-3-line dual event slip past the guard.
        one_line = [" ".join(p.split()) for p in pieces]
        if (
            len(runs) == 2
            and dual_ok
            and all(_vis_width(f"-{t}") <= budget for t in one_line)
        ):
            # Netflix dual-speaker event: one line per speaker, both within budget.
            dual = cast(Cue, dict(cue))
            dual["text"] = f"-{one_line[0]}\n-{one_line[1]}"
            out.append(dual)
            continue
        wd_cursor = 0
        for index, ((_label, atoms_run), piece) in enumerate(zip(runs, pieces)):
            # Slice by each run's recorded word_data footprint, not by atom
            # count: a repacked cue stores one entry per atom while a
            # first-generation one stores one per character. Entries the display
            # dropped (punctuation) sit between footprints and go to the run that
            # follows them; the last run takes the remainder, so a trailing
            # dropped entry is kept rather than lost from the stream.
            if index == len(runs) - 1:
                wd_end = len(word_data)
            elif atoms_run:
                wd_end = min(
                    max(wd_cursor, int(atoms_run[-1]["_unit_end"])), len(word_data)
                )
            else:
                wd_end = wd_cursor
            start, end = _run_span(atoms_run, cue["start"], cue["end"])
            part = cast(Cue, dict(cue))
            # Re-wrap the piece for its language (the same layout machinery
            # smart_split uses): an en split re-flows to <=2 clean lines, a zh/ja
            # piece stays one line and never carries a stale "\n".
            part["text"] = wrap_cue_text(piece, lang, max_lines, max_line_length=budget)
            part["start"] = start
            part["end"] = end
            part["word_data"] = list(word_data[wd_cursor:wd_end])
            wd_cursor = wd_end
            out.append(part)
    return out


def _ordered_speaker_format(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    max_line_length: int | None = None,
) -> list[Cue]:
    """format_speaker_cues + re-sort + overlap trim (splits can abut)."""
    out = format_speaker_cues(cues, turns, lang, max_line_length=max_line_length)
    out.sort(key=lambda c: (c["start"], c["end"]))
    for prev, nxt in zip(out, out[1:]):
        prev["end"] = min(prev["end"], nxt["start"])
    return out


def apply_speaker_format(
    cues: list[Cue],
    turns: Sequence[Turn] | None,
    lang: str,
    *,
    thresholds: dict | SplitThresholds | None = None,
    max_line_length: int | None = None,
) -> list[Cue]:
    """Public entry: no-op without turns, otherwise format + keep cue order sane.

    When ``thresholds`` is given (the same gap thresholds smart_split used for this
    file), the formatted cues run through ``timing._cleanup_cues`` so speaker
    splits/dashes get the same timing polish as ordinary cues (short pieces extend
    into the following gap, sub-0.5s gaps chain) and never render as sub-flash
    cues. ``_cleanup_cues`` is timing-only and never merges content, so distinct
    speakers stay separate cues, and it is idempotent for cues carrying timed
    ``word_data``, so this second pass cannot stack another pad onto an already
    padded cue. ``thresholds=None`` keeps the pre-polish behavior for
    replay/back-compat callers. ``max_line_length`` mirrors smart_split's per-cue
    budget (see :func:`format_speaker_cues`).
    """
    if not turns:
        return cues
    out = _ordered_speaker_format(cues, turns, lang, max_line_length=max_line_length)
    if thresholds is not None:
        from voxweave.core.smart_split import SplitThresholds
        from voxweave.core.timing import _cleanup_cues

        th = (
            SplitThresholds.from_mapping(thresholds)
            if isinstance(thresholds, dict)
            else thresholds
        )
        out = _cleanup_cues(
            out,
            min_cue_s=th.min_cue_s,
            max_cue_s=th.max_cue_s,
            cps=th.cps,
            lag_out_s=th.lag_out_s,
        )
    return out
