"""Align the blocks of one align call with the aligner their route names.

``align_blocks`` picks the aligner from ``_ROUTE_ALIGNERS`` by route kind (see
:class:`voxweave.align_route.AlignRoute`). The three routes are the hard-constraint
full-pass routing; do NOT collapse them:

- ja MMS: one full-file pass (``align_blocks_full_mms``).
- en wav2vec2 CTC: one full-file windowed-emission pass (``align_blocks_full_ctc``).
- zh·yue (no CTC config): per-cue tight-crop Qwen. Each cue gets its own audio
  slice so error is contained within the sentence and inter-sentence pauses are
  preserved.

The caller passes the audio-slicing, time-shifting and owner-slicing steps of the
Qwen route in, so this module depends on no caller state.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from voxweave import backend, realign
from voxweave.progress import Reporter

log = logging.getLogger("voxweave")


@dataclass(frozen=True)
class AlignBlocksRequest:
    wav: Path
    blocks: list[dict]
    iso: str
    ctc_model: str | None
    crops: list[tuple[float, float] | None]
    reporter: Reporter
    tmp_chunks: list[Path]
    speech_spans: list[tuple[float, float]] | None
    slice_audio: Callable[..., Path]
    shift_units: Callable[[list[dict], float], list[dict]]
    retain_owner_slice: Callable[[Sequence[dict]], list[dict]]


@dataclass(frozen=True)
class _PreparedQwenCall:
    source_index: int
    wav_path: Path
    text: str
    nominal_start: float


def _full_pass_bounds(blocks: list[dict]) -> list[tuple[float, float] | None]:
    # MMS/CTC full passes: cue (start,end) bounds are used ONLY as silence anchors to split
    # movie-length audio into memory-sized chunks when it overflows the single-pass DP
    # budget — NOT to crop/route per cue.
    return [
        (b["start"], b["end"])
        if b["start"] is not None and b["end"] is not None
        else None
        for b in blocks
    ]


def _align_full_mms(request: AlignBlocksRequest) -> list[list[dict]]:
    reporter = request.reporter
    reporter.task("full-file alignment (MMS)", 1)
    units = backend.align_blocks_full_mms(
        request.wav,
        [b.get("alignment_text", b["text"]) for b in request.blocks],
        request.iso,
        bounds=_full_pass_bounds(request.blocks),
    )
    reporter.advance(1)
    return units


def _align_full_ctc(request: AlignBlocksRequest) -> list[list[dict]]:
    # en wav2vec2: windowed emission + single global DP (routing-free)
    reporter = request.reporter
    reporter.task("full-file alignment (CTC)", 1)
    units = backend.align_blocks_full_ctc(
        request.wav,
        [b.get("alignment_text", b["text"]) for b in request.blocks],
        request.iso,
        cast(str, request.ctc_model),
        bounds=_full_pass_bounds(request.blocks),
        speech_spans=request.speech_spans,
    )
    reporter.advance(1)
    return units


def _prepare_qwen_calls(request: AlignBlocksRequest) -> list[_PreparedQwenCall]:
    blocks = request.blocks
    reporter = request.reporter
    prepared: list[_PreparedQwenCall] = []
    for i, crop in enumerate(request.crops):
        text = realign.join_block_texts(
            [blocks[i].get("alignment_text", blocks[i]["text"])], request.iso
        )
        if crop is None or not text:  # insertion block or empty: skip
            reporter.advance(1)
            continue
        cs, ce = crop
        cwav = request.slice_audio(
            request.wav,
            cs,
            ce,
            _canonical_qwen_failures=True,
        )
        request.tmp_chunks.append(cwav)
        prepared.append(
            _PreparedQwenCall(
                source_index=i,
                wav_path=cwav,
                text=text,
                nominal_start=float(cs),
            )
        )
    return prepared


def _align_per_cue(request: AlignBlocksRequest) -> list[list[dict]]:
    reporter = request.reporter
    if os.environ.get("VOXWEAVE_VAD_EMISSION_MASK", "").strip() == "1":
        # The VAD emission mask is wired into the wav2vec2 CTC pass only.
        log.warning(
            "--vad-mask / VOXWEAVE_VAD_EMISSION_MASK has no effect on Qwen alignment"
            " (language %s); only the wav2vec2 CTC aligner applies it",
            request.iso,
        )
    reporter.task("per-cue alignment", len(request.blocks))
    block_units: list[list[dict]] = [[] for _ in request.blocks]

    prepared_qwen = _prepare_qwen_calls(request)
    pending_qwen: list[tuple[int, list[dict], float]] = []
    for prepared in prepared_qwen:
        raw_units = backend.align_text(prepared.wav_path, prepared.text, request.iso)
        pending_qwen.append((prepared.source_index, raw_units, prepared.nominal_start))
        reporter.advance(1)

    for index, raw_units, origin in pending_qwen:
        owner_units = request.retain_owner_slice(raw_units)
        block_units[index] = request.shift_units(owner_units, origin)
    return block_units


#: One aligner per route kind (:attr:`voxweave.align_route.AlignRoute.kind`).
_ROUTE_ALIGNERS: dict[str, Callable[[AlignBlocksRequest], list[list[dict]]]] = {
    "mms-full": _align_full_mms,
    "ctc-full": _align_full_ctc,
    "qwen-crop": _align_per_cue,
}


def align_blocks(route_kind: str, request: AlignBlocksRequest) -> list[list[dict]]:
    """Align ``request.blocks`` with the aligner of ``route_kind``; per-block units."""
    return _ROUTE_ALIGNERS[route_kind](request)
