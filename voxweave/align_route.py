from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from voxweave import realign
from voxweave.align_runtime import align_runtime_activity
from voxweave.failure_annotations import _attach_canonical_failure


@dataclass(frozen=True)
class AlignRoute:
    """Aligner route chosen for a language: the one place the route strings live.

    ``kind`` feeds persisted digests and evidence, so its values are fixed.
    """

    kind: Literal["mms-full", "ctc-full", "qwen-crop"]
    full_pass: bool
    ctc_model: str | None

    def backend_label(self, use_mlx: Callable[[], bool]) -> str:
        # ``use_mlx`` is only consulted on the qwen route.
        if self.kind == "mms-full":
            return "mms"
        if self.ctc_model:
            return "ctc"
        return "mlx-qwen" if use_mlx() else "qwen-asr"

    def model_label(self, qwen_model: str) -> str:
        if self.kind == "mms-full":
            return "mms"
        return self.ctc_model if self.ctc_model else qwen_model


def resolve_align_route(mms: bool, ctc_model: str | None) -> AlignRoute:
    return AlignRoute(
        kind="mms-full" if mms else "ctc-full" if ctc_model else "qwen-crop",
        full_pass=mms or bool(ctc_model),
        ctc_model=ctc_model,
    )


# Builds the per-block dicts that every align route consumes, in delivery order.
# input_snapshot is a voxweave.align_snapshot.AlignSnapshot (imported lazily by align).
def _align_blocks_from_snapshot(
    input_snapshot: Any, delivery_order: tuple[int, ...]
) -> list[dict[str, Any]]:
    bounds_by_source = {
        bound.source_index: bound for bound in input_snapshot.route_bounds
    }
    blocks: list[dict[str, Any]] = []
    for source_index in delivery_order:
        content = input_snapshot.blocks[source_index]
        bound = bounds_by_source[source_index]
        blocks.append(
            {
                "text": content.text,
                "alignment_text": content.alignment_text,
                "start": bound.start,
                "end": bound.end,
                "lyric": content.lyric,
                "speaker": content.speaker,
                "speakers": (
                    None if content.speakers is None else list(content.speakers)
                ),
                "source_index": content.source_index,
            }
        )
    return blocks


# Reads the prepared 16k wav geometry (sample rate, sample count); unavailable
# geometry falls back to (16_000, 0) so the sealed facts mark it invalid.
def _prepared_audio_geometry(wav: Path) -> tuple[int, int]:
    try:
        import soundfile as sf

        prepared_info = sf.info(str(wav))
        prepared_sample_rate = int(prepared_info.samplerate)
        prepared_sample_count = int(prepared_info.frames)
    except Exception:  # noqa: BLE001 - unavailable geometry is sealed as invalid
        prepared_sample_rate = 16_000
        prepared_sample_count = 0
    return prepared_sample_rate, prepared_sample_count


# Route facts whose stable digest becomes route_input_digest. Floats are hex
# encoded so the digest does not depend on decimal formatting.
def _align_route_facts(
    route_kind: str,
    iso: str,
    blocks: list[dict[str, Any]],
    crops: list[tuple[float, float] | None],
) -> dict[str, Any]:
    route_facts = {
        "route": route_kind,
        "language": iso,
        "blocks": [
            {
                "source_index": block["source_index"],
                "alignment_text": block["alignment_text"],
                "start": (
                    None if block["start"] is None else float(block["start"]).hex()
                ),
                "end": (None if block["end"] is None else float(block["end"]).hex()),
            }
            for block in blocks
        ],
        "crops": [
            None if crop is None else [float(crop[0]).hex(), float(crop[1]).hex()]
            for crop in crops
        ],
    }
    return route_facts


def _stable_fact_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Per-cue (zh/yue Qwen) route planning: route every block to its audio window, then
# crop each window. Every unusable route raises a canonical failure.
def _plan_qwen_crops(
    blocks: list[dict[str, Any]], word_segments: Any, json_path: Path
) -> list[tuple[float, float] | None]:
    has_ts = all(b["start"] is not None and b["end"] is not None for b in blocks)
    if not has_ts and not word_segments:
        untimed = sum(1 for b in blocks if b["start"] is None or b["end"] is None)
        if untimed == len(blocks):
            message = (
                f"{json_path.name} has no word_segments and VTT has no timestamps; "
                f"cannot route audio windows"
            )
        else:
            message = (
                f"{json_path.name} has no word_segments and {untimed} of "
                f"{len(blocks)} VTT cues have no timestamps; cannot route audio "
                f"windows (give every cue a timing line, or restore the sibling "
                f"JSON's word_segments)"
            )
        exc = RuntimeError(message)
        _attach_canonical_failure(
            exc,
            kind="qwen-route-invalid",
            phase="route-plan",
            detail_code="no-route-source",
        )
        raise exc
    with align_runtime_activity("AO-03", "qwen-route-plan"):
        try:
            spans = realign.route_blocks(blocks, word_segments)
            crops = realign.crop_blocks(spans)
        except (IndexError, KeyError) as exc:
            _attach_canonical_failure(
                exc,
                kind="qwen-window-operation-failed",
                phase="route-plan",
                detail_code="route-bound-access",
            )
            raise
        except (ArithmeticError, TypeError, ValueError) as exc:
            _attach_canonical_failure(
                exc,
                kind="qwen-window-operation-failed",
                phase="route-plan",
                detail_code="route-bound-arithmetic",
            )
            raise
    if all(c is None for c in crops):
        exc = RuntimeError(
            "routing failed: no alignable blocks (text completely mismatches word_segments?)"
        )
        _attach_canonical_failure(
            exc,
            kind="qwen-route-invalid",
            phase="route-plan",
            detail_code="all-crops-none",
        )
        raise exc
    return crops
