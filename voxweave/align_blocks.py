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
from functools import partial
from pathlib import Path
from typing import Any, cast

from voxweave import backend, realign
from voxweave.align_runtime import align_runtime_activity
from voxweave.progress import Reporter

log = logging.getLogger("voxweave")

#: Runs one zero-argument operation inside a runtime phase and returns its result.
PhaseInvoker = Callable[[Callable[[], Any]], Any]


@dataclass(frozen=True)
class AlignBlocksRequest:
    # One align call's blocks, the phase hooks the P6 acquisition wraps around the
    # backend, and the three collaborators of the Qwen per-cue route.
    wav: Path
    blocks: list[dict]
    iso: str
    ctc_model: str | None
    crops: list[tuple[float, float] | None]
    reporter: Reporter
    tmp_chunks: list[Path]
    speech_spans: list[tuple[float, float]] | None
    raw_call_observer: Callable[..., None] | None
    qwen_invoker: Callable[..., Sequence[Any]] | None
    backend_invoker: PhaseInvoker | None
    physical_preparation_invoker: PhaseInvoker | None
    legacy_distribution_invoker: PhaseInvoker | None
    legacy_shift_invoker: PhaseInvoker | None
    slice_audio: Callable[..., Path]
    shift_units: Callable[[list[dict], float], list[dict]]
    retain_owner_slice: Callable[[Sequence[dict]], list[dict]]


@dataclass(frozen=True)
class _PreparedQwenCall:
    source_index: int
    wav_path: Path
    text: str
    nominal_start: float
    nominal_end: float
    sample_geometry: tuple[int, int, int, int] | None


def _full_pass_bounds(blocks: list[dict]) -> list[tuple[float, float] | None]:
    # MMS/CTC full passes: cue (start,end) bounds are used ONLY as silence anchors to split
    # movie-length audio into memory-sized chunks when it overflows the single-pass DP
    # budget — NOT to crop/route per cue. These routes are routing-free because the input
    # VTT timestamps are exactly what may be wrong (the reason to re-align); the global DP
    # self-locates every word. None for cues without timestamps. (The Qwen per-cue route
    # crops each cue from ``request.crops`` instead.)
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
        _raw_call_observer=request.raw_call_observer,
        _backend_invoker=request.backend_invoker,
        _preparation_invoker=request.physical_preparation_invoker,
        _legacy_distribution_invoker=request.legacy_distribution_invoker,
        _legacy_shift_invoker=request.legacy_shift_invoker,
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
        _raw_call_observer=request.raw_call_observer,
        _backend_invoker=request.backend_invoker,
        _preparation_invoker=request.physical_preparation_invoker,
        _legacy_distribution_invoker=request.legacy_distribution_invoker,
        _legacy_shift_invoker=request.legacy_shift_invoker,
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
        observed_geometry: tuple[int, int, int, int] | None = None

        def observe_sample_geometry(
            sample_start: int,
            sample_end: int,
            sample_rate: int,
            sample_count: int,
        ) -> None:
            nonlocal observed_geometry
            if observed_geometry is not None:
                raise RuntimeError("Qwen slice reported sample geometry twice")
            observed_geometry = (
                sample_start,
                sample_end,
                sample_rate,
                sample_count,
            )

        cwav = request.slice_audio(
            request.wav,
            cs,
            ce,
            _sample_geometry_observer=observe_sample_geometry,
            _canonical_qwen_failures=True,
        )
        request.tmp_chunks.append(cwav)
        prepared.append(
            _PreparedQwenCall(
                source_index=i,
                wav_path=cwav,
                text=text,
                nominal_start=float(cs),
                nominal_end=float(ce),
                sample_geometry=observed_geometry,
            )
        )
    return prepared


def _retain_owners(
    request: AlignBlocksRequest, pending_qwen: list[tuple[int, list[dict], float]]
) -> list[tuple[int, list[dict], float]]:
    return [
        (index, request.retain_owner_slice(raw_units), origin)
        for index, raw_units, origin in pending_qwen
    ]


def _shift_owners(
    request: AlignBlocksRequest,
    retained_owners: list[tuple[int, list[dict], float]],
    block_units: list[list[dict]],
) -> list[list[dict]]:
    for index, owner_units, origin in retained_owners:
        block_units[index] = request.shift_units(owner_units, origin)
    return block_units


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

    prepared_qwen = (
        _prepare_qwen_calls(request)
        if request.physical_preparation_invoker is None
        else cast(
            list[_PreparedQwenCall],
            request.physical_preparation_invoker(partial(_prepare_qwen_calls, request)),
        )
    )
    pending_qwen: list[tuple[int, list[dict], float]] = []
    for prepared in prepared_qwen:
        geometry = prepared.sample_geometry
        if request.qwen_invoker is None:
            raw_units = backend.align_text(
                prepared.wav_path, prepared.text, request.iso
            )
        else:
            raw_units = cast(
                list[dict[str, Any]],
                request.qwen_invoker(
                    lambda: backend.align_text(
                        prepared.wav_path,
                        prepared.text,
                        request.iso,
                    ),
                    prepared.source_index,
                    prepared.nominal_start,
                    prepared.nominal_end,
                    audio_sample_start=None if geometry is None else geometry[0],
                    audio_sample_end=None if geometry is None else geometry[1],
                    sample_rate=None if geometry is None else geometry[2],
                    sample_count=None if geometry is None else geometry[3],
                ),
            )

        pending_qwen.append((prepared.source_index, raw_units, prepared.nominal_start))
        # One tick per aligned cue as it lands (skipped cues ticked at preparation),
        # so the bar moves with the work instead of jumping from 0/N at the end.
        reporter.advance(1)

    if not pending_qwen:
        with align_runtime_activity("AO-07", "no-physical-calls"):
            pass

    retained_owners = (
        _retain_owners(request, pending_qwen)
        if request.legacy_distribution_invoker is None
        else cast(
            list[tuple[int, list[dict], float]],
            request.legacy_distribution_invoker(
                partial(_retain_owners, request, pending_qwen)
            ),
        )
    )
    return (
        _shift_owners(request, retained_owners, block_units)
        if request.legacy_shift_invoker is None
        else cast(
            list[list[dict]],
            request.legacy_shift_invoker(
                partial(_shift_owners, request, retained_owners, block_units)
            ),
        )
    )


#: One aligner per route kind (:attr:`voxweave.align_route.AlignRoute.kind`).
_ROUTE_ALIGNERS: dict[str, Callable[[AlignBlocksRequest], list[list[dict]]]] = {
    "mms-full": _align_full_mms,
    "ctc-full": _align_full_ctc,
    "qwen-crop": _align_per_cue,
}


def align_blocks(route_kind: str, request: AlignBlocksRequest) -> list[list[dict]]:
    """Align ``request.blocks`` with the aligner of ``route_kind``; per-block units."""
    if (request.legacy_distribution_invoker is None) != (
        request.legacy_shift_invoker is None
    ):
        raise ValueError("legacy projection phase invokers must be supplied together")
    return _ROUTE_ALIGNERS[route_kind](request)
