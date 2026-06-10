"""wav2vec2 CTC forced alignment (English and other spaced languages).

Full-pass design: windowed emissions bound the encoder's O(T^2) self-attention,
word-level ``<star>`` wildcards absorb untranscribed gaps, and one global
forced-align DP self-locates every word (routing-free, immune to per-cue
cropping drift). Movie-length audio is DP-chunked at silence anchors via
``align_common._dp_chunked_pass``.
"""

from __future__ import annotations

import logging
import os
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

from voxweave import config
from voxweave.align_common import (
    _distribute_units,
    _dp_chunked_pass,
    _load_mono,
    _mask_emissions_outside_speech,
    _strip_trailing_punct,
    interp_missing,
)
from voxweave.runtime import _empty_cache, get_device

log = logging.getLogger("voxweave")

# wav2vec2 CTC aligner; cached by iso. blank/sep_id/invocab come from the model, not hardcoded.
# proc is only set on the HF path (required for z-score normalization).
CtcAligner = namedtuple("CtcAligner", "kind model sr blank sep_id invocab proc")
_ctc = None  # CtcAligner
_ctc_lang = None  # iso of the loaded CTC singleton (reloaded on language change)

# wav2vec2 CTC windowed emission (mirrors ctc-forced-aligner generate_emissions): encode the
# waveform in CTC_EMIT_WINDOW_S windows with CTC_EMIT_CONTEXT_S overlap each side (so edge frames
# stay well-attended), drop the context frames, concatenate -> bounds the encoder's O(T^2)
# self-attention so the full-file CTC pass survives long audio (full-file xlsr OOMs at 23min).
CTC_EMIT_WINDOW_S = float(os.environ.get("VOXWEAVE_CTC_WINDOW_S", "30"))
CTC_EMIT_CONTEXT_S = float(os.environ.get("VOXWEAVE_CTC_CONTEXT_S", "2"))


def _get_ctc_aligner(iso: str, model_name: str):
    """Lazy-load wav2vec2 CTC aligner singleton, cached by iso. Reloads on language change.

    model_name in torchaudio.pipelines -> torchaudio bundle; otherwise -> HF Wav2Vec2ForCTC
    (auto-downloads to config.ALIGN_CACHE). Returns a CtcAligner namedtuple.
    """
    global _ctc, _ctc_lang
    if _ctc is not None and _ctc_lang != iso:
        _ctc = None
        _ctc_lang = None
        _empty_cache()
    if _ctc is None:
        import torchaudio

        dev = get_device()
        if (
            model_name in torchaudio.pipelines.__all__
        ):  # torchaudio bundle (English large)
            bundle = torchaudio.pipelines.__dict__[model_name]
            model = bundle.get_model().to(dev).eval()
            labels = bundle.get_labels()
            sep_id = labels.index("|") if "|" in labels else -1
            invocab = {c: i for i, c in enumerate(labels) if i not in (0, sep_id)}
            _ctc = CtcAligner(
                "torchaudio", model, bundle.sample_rate, 0, sep_id, invocab, None
            )
        else:  # HF Wav2Vec2ForCTC (English LV60K-self, Japanese xlsr etc.): blank=pad_id, invocab=full vocab
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            # Auto-downloads to config.ALIGN_CACHE on first run; cache hit on subsequent runs.
            proc = Wav2Vec2Processor.from_pretrained(
                model_name, cache_dir=config.ALIGN_CACHE
            )
            model = (
                Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=config.ALIGN_CACHE)
                .to(dev)
                .eval()
            )
            vocab = proc.tokenizer.get_vocab()  # {char: id}
            _ctc = CtcAligner(
                "hf",
                model,
                proc.feature_extractor.sampling_rate,
                proc.tokenizer.pad_token_id,
                vocab.get("|", -1),
                vocab,
                proc,
            )
        _ctc_lang = iso
        log.info(
            "loaded CTC aligner lang=%s model=%s kind=%s on %s",
            iso,
            model_name,
            _ctc.kind,
            dev,
        )
    return _ctc


def _ctc_logp(al, wav):
    """Single forward pass: 1D waveform tensor @ al.sr -> [T,V] log-probs (softmax of raw logits).

    HF models need the processor's z-score normalization (skipping it shifts argmax); torchaudio
    bundles take the raw waveform. Must log_softmax the raw logits before forced_align.
    """
    import torch

    if wav.shape[-1] < 400:  # wav2vec2 conv minimum input length, prevents crash
        wav = torch.nn.functional.pad(wav, (0, 400 - wav.shape[-1]))
    with torch.inference_mode():
        if al.kind == "hf":  # HF: processor z-score normalization required
            inp = al.proc(
                wav.cpu().numpy(), sampling_rate=al.sr, return_tensors="pt"
            ).input_values
            logits = al.model(inp.to(get_device())).logits[0]
        else:  # torchaudio bundle: raw wav, returns (emissions, lengths)
            emis, _ = al.model(wav.unsqueeze(0).to(get_device()))
            logits = emis[0]
        return torch.log_softmax(logits, dim=-1)  # [T,V]


def _ctc_emit_full(al, wav):
    """Long waveform -> seamless [T,V] log-probs via windowed forward passes.

    Mirrors ctc-forced-aligner generate_emissions for wav2vec2: encode in CTC_EMIT_WINDOW_S
    windows padded by CTC_EMIT_CONTEXT_S of overlap each side (edge frames stay well-attended),
    drop the context frames, concatenate. The kept interior of each window tiles the file
    gap-free; the encoder never sees more than window+2*context, bounding O(T^2) self-attention.
    """
    import torch

    sr = al.sr
    win = int(CTC_EMIT_WINDOW_S * sr)
    ctx = int(CTC_EMIT_CONTEXT_S * sr)
    n = wav.shape[-1]
    if n <= win + 2 * ctx:
        return _ctc_logp(al, wav)  # short enough for a single pass
    parts = []
    pos = 0
    while pos < n:
        a = max(0, pos - ctx)
        b = min(n, pos + win + ctx)
        lp = _ctc_logp(al, wav[a:b])  # [t,V]
        stride = (b - a) / lp.shape[0]  # samples/frame for this window (~320)
        end = min(pos + win, n)
        lo = round((pos - a) / stride)  # drop prepended left-context frames
        hi = lp.shape[0] - (round((b - end) / stride) if end < n else 0)
        lo = max(0, min(lo, lp.shape[0]))
        hi = max(lo, min(hi, lp.shape[0]))
        parts.append(lp[lo:hi])
        pos = end
    return torch.cat(parts, dim=0)


def _ctc_align_logp(al, logp, toks, meta, words, nospace, total_samples):
    """[T,V] log-probs + tokens -> word/char units. Shared by per-cue and full-file CTC.

    Appends a wildcard column for OOV tokens (WhisperX technique), runs forced_align + merge,
    maps frames to seconds via total_samples/T/sr. No-space langs get a last-resort span fill.
    """
    import torch
    import torchaudio.functional as AF

    toks = list(toks)
    if any(
        t is None for t in toks
    ):  # OOV wildcard: max non-blank score per frame column
        cols = [i for i in range(logp.shape[1]) if i != al.blank]
        star = logp[:, cols].max(dim=1).values
        logp = torch.cat([logp, star.unsqueeze(1)], dim=1)
        star_id = logp.shape[1] - 1
        toks = [star_id if t is None else t for t in toks]
    # torchaudio.forced_align has no MPS kernel, so on Apple Silicon run the (cheap) DP on CPU —
    # emissions stay on MPS for the forward. CUDA is left untouched: forced_align runs on the GPU
    # exactly as before (the DP is deterministic, so CPU/CUDA give identical alignments).
    if logp.device.type == "mps":
        logp = logp.detach().to("cpu")
    targets = torch.tensor([toks], dtype=torch.int32, device=logp.device)
    aligned, scores = AF.forced_align(
        logp.unsqueeze(0).contiguous(), targets, blank=al.blank
    )
    spans = AF.merge_tokens(aligned[0], scores[0], blank=al.blank)
    ratio = total_samples / logp.shape[0] / al.sr
    units = _ctc_words_from_spans(spans, meta, words, ratio)
    if nospace:  # last-resort span fill; never drops a character
        units = interp_missing(units)
    return units


def _ctc_words_from_spans(
    spans, meta: list[int], words: list[str], ratio: float
) -> list[dict]:
    """Token-level spans + word idx -> word-level units [{text,start,end}].

    Groups tokens by word idx (separator meta<0 skipped); start/end = frame range * ratio (-> seconds).
    Pure logic; spans just need .start/.end attributes.
    """
    groups: dict[int, list] = {}  # dict insertion order == word order (Py3.7+)
    for span, m in zip(spans, meta):
        if m < 0:
            continue
        groups.setdefault(m, []).append(span)
    units: list[dict] = []
    for widx, sps in groups.items():
        start = min(s.start for s in sps) * ratio
        end = max(s.end for s in sps) * ratio
        units.append(
            {
                "text": _strip_trailing_punct(words[widx]),
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
    return units


def _ctc_build_tokens(norm: list[str], nospace: bool, al):
    """Build the <star>-interleaved token stream for full-pass CTC over cue texts `norm`.

    Token stream: <star> word0 <star> word1 <star> ... wordN <star>. A wildcard star sits at
    EVERY word boundary (not just cue boundaries) and both edges. The star is a None token
    (-> wildcard column in _ctc_align_logp) at meta=-1 (-> skipped in word grouping AND in
    _distribute_units, which counts real words/chars). Because a star sits between every pair of
    words regardless of how cues are grouped, the global monotone path absorbs ANY inter-word gap
    -- intra-cue music/silence included -- instead of cramming the later word forward (the
    failure: "...these <2-3s gap> blocks" placed blocks right after these, ignoring the pause).
    Returns (toks, meta, words); words are flattened in cue order for _distribute_units.
    """
    toks: list[int | None] = []
    meta: list[int] = []
    words: list[str] = []

    def _star() -> None:
        toks.append(None)
        meta.append(-1)

    _star()
    for t in norm:
        if not t:
            continue
        for it in list(t) if nospace else t.split():
            if nospace and not it.isalnum():
                continue
            widx = len(words)
            # no case-fold for no-space vocabs (xlsr-ja has uppercase A/C/P only); upper otherwise
            toks.extend(al.invocab.get(c if nospace else c.upper()) for c in it)
            meta.extend(widx for _ in it)
            words.append(it)
            _star()  # wildcard after every word absorbs the inter-word gap
    return toks, meta, words


def _ctc_full_pass(
    al,
    wav,
    norm: list[str],
    nospace: bool,
    iso: str,
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[list[dict]]:
    """One windowed-emission + global forced_align over `wav` for cue texts `norm`.

    Times are relative to the start of `wav` (caller offsets when `wav` is a chunk). Returns
    per-cue units in `norm` order; empty/wordless cues get []. The single DP is O(T*L).
    ``speech_spans`` (seconds, wav-relative) soft-mask non-speech emissions before the DP.
    """
    toks, meta, words = _ctc_build_tokens(norm, nospace, al)
    if not words:
        return [[] for _ in norm]
    logp = _ctc_emit_full(al, wav)
    if speech_spans:
        logp = _mask_emissions_outside_speech(
            logp, speech_spans, wav.shape[-1], al.sr, al.blank
        )
    units = _ctc_align_logp(al, logp, toks, meta, words, nospace, wav.shape[-1])
    return _distribute_units(units, norm, iso)


def align_blocks_full_ctc(
    wav_path: Path,
    texts: list[str],
    iso: str,
    model_name: str,
    bounds: Sequence[tuple[float, float] | None] | None = None,
    speech_spans: list[tuple[float, float]] | None = None,
) -> list[list[dict]]:
    """Full-audio single-pass wav2vec2 CTC alignment (en analogue of align_blocks_full_mms).

    Runs ONE windowed-emission + global forced_align over the whole audio, then slices flat
    units back to each block by word/char count. The global monotone CTC path self-locates every
    word, immune to the per-cue cropping drift that crammed words into dead air (en "blocks"
    displaced into a 2.6s silence); inter-cue stars absorb untranscribed gaps (music/silence
    between cues) so the path never stretches a real word across a gap. Units are absolute
    timestamps relative to the full wav. Movie-length audio is DP-chunked at silence anchors
    via cue `bounds` (see _dp_chunked_pass). ``speech_spans`` (VAD, absolute seconds)
    soft-mask non-speech emissions so words cannot park in music/silence — opt-in via
    VOXWEAVE_VAD_EMISSION_MASK=1 (see _mask_emissions_outside_speech for why).
    """
    from voxweave.realign import NO_SPACE_LANGS

    if os.environ.get("VOXWEAVE_VAD_EMISSION_MASK", "").strip() != "1":
        speech_spans = None
    al = _get_ctc_aligner(iso, model_name)
    nospace = iso in NO_SPACE_LANGS
    norm = [(t or "").strip() for t in texts]
    wav = _load_mono(wav_path, al.sr)

    def _pass(w, sub: list[str], offset_s: float = 0.0) -> list[list[dict]]:
        spans_rel = None
        if speech_spans:
            end_s = offset_s + w.shape[-1] / al.sr
            spans_rel = [
                (max(0.0, s - offset_s), min(end_s, e) - offset_s)
                for s, e in speech_spans
                if e > offset_s and s < end_s
            ]
        out = _ctc_full_pass(al, w, sub, nospace, iso, speech_spans=spans_rel)
        _empty_cache()
        return out

    return _dp_chunked_pass(wav, al.sr, norm, bounds, _pass, "CTC")


def align_text_ctc(wav_path: Path, text: str, iso: str, model_name: str) -> list[dict]:
    """wav2vec2 CTC forced alignment: blank absorbs silence giving tight boundaries.

    Spaced langs -> word-level units. No-space langs (NO_SPACE_LANGS) -> per-char units
    (punctuation skipped; OOV kanji use wildcard; missing spans filled by interp_missing).
    Same star-interleaved full pass as the align subcommand (_ctc_full_pass with a single
    text): a wildcard at every word boundary absorbs intra-chunk gaps (pauses the ASR did
    not transcribe) instead of cramming the next word forward, and the windowed emission
    bounds the encoder's O(T^2) self-attention on long chunks.
    Exceptions propagate; align_text catches and falls back to Qwen.
    """
    from voxweave.realign import NO_SPACE_LANGS

    text = (text or "").strip()
    if not text:
        return []
    al = _get_ctc_aligner(iso, model_name)
    nospace = iso in NO_SPACE_LANGS
    wav = _load_mono(wav_path, al.sr)
    units = _ctc_full_pass(al, wav, [text], nospace, iso)[0]
    _empty_cache()
    return units


def release_ctc() -> None:
    """Drop the wav2vec2 CTC singleton (backend.release() calls this)."""
    global _ctc, _ctc_lang
    _ctc = None
    _ctc_lang = None
