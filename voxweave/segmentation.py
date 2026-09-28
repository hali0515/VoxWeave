"""The v1 segmentation entry point: aligned word segments in, final cues out.

:func:`segment_document` is the one orchestration around ``smart_split`` that
``process``, ``split``/``render`` replay and offline calibration all run, and it
records the ``segmentation`` manifest the sibling JSON persists. It is pure: no
filesystem writes, no models.
"""

from __future__ import annotations

import copy
import importlib.metadata
import logging
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from voxweave import realign
from voxweave.core.overlay import (
    copied_spans,
    copied_turns,
    mark_lyric_cues,
    resnap_shots,
)
from voxweave.core.providers import degradation_capture, provider_snapshot
from voxweave.core.schema import Cue
from voxweave.core.segdoc import (
    THRESHOLD_KEYS,
    DisplayProfile,
    SegDocument,
    build_seg_document,
)
from voxweave.core.shadow_v2 import SEG_V2_SHADOW_ENV

log = logging.getLogger("voxweave")


def resolve_segmentation_manifest(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """The segmentation manifest of a loaded sibling document.

    Every sibling written before P3 carries no ``segmentation`` key at all, and
    that absence is itself the label: such a document was produced by the legacy
    v1 engine, so it resolves to ``{"engine": "legacy-v1", "inferred": True}``
    rather than to nothing. ``inferred`` distinguishes the deduction from a
    recorded manifest, which is returned exactly as stored (a non-mapping value
    is not one and falls back to the inference).

    The engine name here is a literal on purpose and must NOT become
    :data:`SEGMENTATION_ENGINE`: that constant tracks whichever engine this build
    runs, while a manifest-less file was written by v1 no matter what this build
    does now.
    """
    found = data.get("segmentation")
    if isinstance(found, Mapping):
        return found
    return {"engine": "legacy-v1", "inferred": True}


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


def _units_to_seg(units: list[dict], iso: str) -> dict:
    """Flatten word_segments into a single segment dict for smart_split.

    Units already carry punctuation from reinject_punct. No-space languages join without
    separator; smart_split uses punctuation for sentence breaking and converts it to spaces.
    Surfaces are read through the tolerant accessor: units legally carry their text under
    ``text`` or ``word`` (see ``schema.Unit``), and replayed sibling JSONs use either.
    """
    from voxweave.core.smart_split import _unit_text

    sep = "" if iso in realign.NO_SPACE_LANGS else " "
    words = [
        {"word": _unit_text(u), "start": u["start"], "end": u["end"]} for u in units
    ]
    return {
        "start": units[0]["start"],
        "end": units[-1]["end"],
        "text": sep.join(_unit_text(u) for u in units),
        "words": words,
    }


#: Shape version of the sibling JSON's ``segmentation`` block. Bump when a field
#: changes meaning; a reader that does not know the version must not guess.
SEGMENTATION_MANIFEST_VERSION = 1
#: The segmentation engine this build runs. P3 records the pre-strangler one so a
#: later document can be told apart from every file written before the manifest.
SEGMENTATION_ENGINE = "legacy-v1"


def _voxweave_version() -> str:
    """Installed voxweave version, or ``"unknown"`` running from a source tree."""
    try:
        return importlib.metadata.version("voxweave")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class SegmentationResult:
    """Output of :func:`segment_document`: the cue stream plus what produced it.

    ``units`` is the (copied) unit stream after punctuation snapping -- the same
    stream the callers persist as ``word_segments``, so a replay writes back what
    it actually split. ``diagnostics`` records which optional passes ran (all
    values deterministic, so two identical inputs compare equal); the effective
    gap/duration thresholds, after the optional adaptive pass, are the ones
    ``manifest["profile"]`` quotes.

    ``manifest`` is the ``SegmentationManifest`` the callers persist as the
    sibling JSON's ``segmentation`` key, and ``document`` is the
    :class:`~voxweave.core.segdoc.SegDocument` holding that same manifest object
    -- minted before the engine runs, so it describes the inputs rather than
    summarizing the output. Both are additive and default to ``None`` so
    existing constructors keep working; in legacy-v1 the engine consumes
    neither.

    ``shadow`` is the BoundaryOptimizer v2 measurement artifact, present only
    when :data:`SEG_V2_SHADOW_ENV` is on. It is *returned* rather than written
    because ``segment_document`` is pinned pure; persisting it is a caller's or
    the harness's job. Nothing in the shipped output depends on it -- a run with
    the flag on and a run with it off produce byte-identical siblings.
    """

    cues: list[Cue]
    language: str
    units: list[dict]
    diagnostics: dict[str, Any]
    manifest: dict[str, Any] | None = None
    document: SegDocument | None = None
    shadow: dict[str, Any] | None = None


# --------------------------------------------------------- v2 shadow lane


def _maybe_shadow_v2(
    document: SegDocument, cues: Sequence[Cue], *, thresholds: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Measure BoundaryOptimizer v2 beside the shipped v1 answer, or do nothing.

    The flag is read FIRST and the lane (:func:`voxweave.core.shadow_v2.run_shadow`)
    is entered only after it passes, so an off run costs one environment read
    and a branch and never pulls a v2 module into the process at all: the lane
    module itself imports nothing at module scope that the v1 engine does not
    already load, and every optimizer import inside it is deferred to the call.
    Nothing in the lane may fail the run: ``run_shadow`` records an unexpected
    error as a typed ``error`` block and the shipped cues are returned untouched.
    """
    if os.environ.get(SEG_V2_SHADOW_ENV, "").strip() != "1":
        return None
    from voxweave.core.shadow_v2 import run_shadow

    return run_shadow(document, cues, thresholds=thresholds)


def segment_document(
    *,
    language: str,
    word_segments: Sequence[Mapping[str, Any]],
    vad_speech: Sequence[tuple[float, float]] | None = (),
    shot_changes: Sequence[float] | None = (),
    sing_spans: Sequence[tuple[float, float]] | None = (),
    speaker_turns: Sequence[tuple[float, float, str]] | None = (),
    thresholds: Mapping[str, Any] | None = None,
    smart_split_kwargs: Mapping[str, Any] | None = None,
    annotate_speakers: bool = False,
) -> SegmentationResult:
    """Turn aligned word segments into the final cue stream. Pure and deterministic.

    This is the single segmentation orchestration shared by
    :func:`voxweave.pipeline.process` (the post-ASR half),
    :func:`voxweave.pipeline.split` (sibling-JSON replay) and offline calibration
    replay -- nobody re-implements the adapter logic around ``smart_split``.
    The pass order is exactly what production runs:

    1. snap sentence-break punctuation onto word boundaries (zh only),
    2. repair stranded word tails (``repair_stranded_tails``) on the stream cue
       formation sees; ``result.units`` keeps the raw aligner timings,
    3. flatten the units into one segment,
    4. resolve the effective thresholds (optional adaptive gap scaling),
    5. record the manifest and mint the :class:`SegDocument` (see below),
    6. ``smart_split_segments`` (content breaks + timing cleanup + shot snap),
    7. lyric marking from ``sing_spans``,
    8. speaker formatting from ``speaker_turns`` (which re-runs timing cleanup),
    9. re-snap to ``shot_changes`` because step 8 moved boundaries again.

    Step 5 sits *before* the engine on purpose: the document is the single
    authority describing what this segmentation runs on, so a later engine takes
    it as input instead of being reverse-engineered from its own output. Only
    ``degraded`` cannot be known that early; the manifest reserves the key at
    build time and the ledger is written into it once the capture closes, so the
    persisted block is byte-identical either way.

    With :data:`SEG_V2_SHADOW_ENV` on, the v2 optimizer is measured between steps
    6 and 7 and its artifact is returned on ``result.shadow``. It ships nothing:
    the cue stream, the units and the persisted manifest are byte-identical to a
    run with the flag off.

    No filesystem writes, no model loads, no ASR. Every input sequence is copied
    before use, so callers can reuse their own lists afterwards.

    ``vad_speech`` distinguishes absent (``None``/empty -> single-gap-threshold
    degradation in ``gap_split``) from real spans; ``shot_changes``,
    ``sing_spans`` and ``speaker_turns`` treat absent and empty alike.
    ``thresholds`` defaults to ``config.gap_thresholds(language)``.
    ``smart_split_kwargs`` forwards layout overrides (``max_line_length``,
    ``max_lines``, ...) to ``smart_split_segments``; ``max_line_length`` also
    reaches the speaker formatter so both measure the same budget. The layout
    pair is resolved here rather than inside the engine, so the manifest records
    the values that actually ran instead of re-deriving them from a second copy
    of the defaulting rule.
    """
    from voxweave.config import gap_thresholds
    from voxweave.core.layout import default_max_line_length, default_max_lines
    from voxweave.core.smart_split import SplitThresholds, smart_split_segments

    iso = language
    units: list[dict] = [copy.deepcopy(dict(u)) for u in word_segments]
    speech_spans = copied_spans(vad_speech)
    cuts = [float(t) for t in shot_changes] if shot_changes else None
    sings = copied_spans(sing_spans)
    turns = copied_turns(speaker_turns)
    extra: dict[str, Any] = dict(smart_split_kwargs or {})
    # Resolve the layout pair once and hand the resolved values to the engine:
    # passing them explicitly is what the engine would have defaulted to anyway,
    # so output is unchanged, and the manifest/profile can then quote what ran.
    max_line_length = extra.get("max_line_length")
    if max_line_length is None:
        max_line_length = default_max_line_length(iso)
    max_lines = extra.get("max_lines")
    if max_lines is None:
        max_lines = default_max_lines(iso)
    extra["max_line_length"] = max_line_length
    extra["max_lines"] = max_lines

    # zh: Qwen punctuation can drift up to one character; snap to jieba word boundary
    # to prevent smart_split from splitting mid-word (e.g. 数据|中心 instead of 数据中心).
    snapped = realign.snap_break_punct(units, iso)
    # Stranded word tails (aligner parked a word-final char across dead air) are
    # repaired only on the stream cue formation sees; ``result.units`` keeps the
    # raw aligner timings, so persisted siblings stay alignment evidence and
    # every replay re-derives the repair.
    from voxweave.core.unit_repair import repair_stranded_tails

    repaired = repair_stranded_tails(snapped, iso, speech_spans)
    seg = _units_to_seg(repaired, iso)
    base = dict(thresholds) if thresholds is not None else gap_thresholds(iso)
    effective = _maybe_adaptive_thresholds(base, snapped)
    # The nine threshold values the engine really ran on: ``effective`` is the
    # caller's mapping, which ``smart_split_segments`` normalizes through
    # ``SplitThresholds.from_mapping`` (partial mappings fill dataclass defaults).
    # Quoting that same normalization keeps the profile honest for a partial
    # mapping AND keeps the recorder strict -- ``DisplayProfile.from_resolved``
    # still raises on a missing key, it is just never handed an incomplete one.
    resolved_th = SplitThresholds.from_mapping(effective)
    profile_thresholds = {key: getattr(resolved_th, key) for key in THRESHOLD_KEYS}
    # Mirror the ONLY consumer, ``align_ctc.align_blocks_full_ctc``, which masks
    # iff the value is exactly "1". ``--no-vad-mask`` writes the literal "0",
    # which is truthy as a string, so ``bool()`` would record masking as ON for
    # the run that explicitly turned it off.
    vad_mask_on = os.environ.get("VOXWEAVE_VAD_EMISSION_MASK", "").strip() == "1"
    manifest: dict[str, Any] = {
        "manifest_version": SEGMENTATION_MANIFEST_VERSION,
        "engine": SEGMENTATION_ENGINE,
        "voxweave": _voxweave_version(),
        "python": platform.python_version(),
        "language": iso,
        # Verbatim: the profile's whole value is saying what ran, so no clamp and
        # no renormalization (the tree carries two disagreeing default sets).
        "profile": {
            "max_line_length": max_line_length,
            "max_lines": max_lines,
            **profile_thresholds,
        },
        "env": {
            # True only when the adaptive pass actually replaced values, not
            # merely because the opt-in env var was set: the pass can hand back a
            # fresh dict whose estimate happens to equal the static one.
            "gap_adaptive": effective != base,
            "vad_emission_mask": vad_mask_on,
        },
        "providers": provider_snapshot(iso),
        # Placeholder in its final position: the ledger only exists once the run
        # is over, but the key is inserted here so the persisted key order does
        # not depend on when the value arrives.
        "degraded": [],
    }
    # The document is the single authority for this segmentation, so it is minted
    # before the engine runs, not reconstructed from its output. It holds the
    # manifest by reference, which is what lets ``degraded`` be filled in below
    # without the document and the sibling JSON drifting apart.
    document = build_seg_document(
        language=iso,
        units=repaired,
        profile=DisplayProfile.from_resolved(
            iso,
            profile_thresholds,
            max_line_length=max_line_length,
            max_lines=max_lines,
        ),
        manifest=manifest,
        vad_speech=speech_spans,
        shot_changes=cuts,
        sing_spans=sings,
        speaker_turns=turns,
        text=seg["text"],
    )
    # Everything the language providers touch runs inside the capture, so the
    # manifest can say which fallbacks actually fired on this document. Nothing
    # above reaches a provider (``_units_to_seg`` joins surfaces, the threshold
    # passes read config/env, and ``provider_snapshot`` only *reports* identity),
    # so narrowing the window to the engine run drops no event.
    with degradation_capture() as degraded:
        cues = smart_split_segments(
            [seg],
            lang=iso,
            speech_spans=speech_spans,
            thresholds=effective,
            shot_changes=cuts,
            **extra,
        )
        # Immediately after the v1 answer and before any overlay: the shadow
        # sees exactly the stream v1 produced from exactly the inputs v1 read.
        shadow = _maybe_shadow_v2(document, cues, thresholds=effective)
        mark_lyric_cues(cues, sings)
        split_cue_count = len(cues)
        if turns:
            from voxweave.diarize import apply_speaker_format

            # Same thresholds AND line budget as smart_split so speaker splits get the
            # same timing polish and wrap width the deterministic layout just used.
            cues = apply_speaker_format(
                cues,
                turns,
                iso,
                thresholds=effective,
                max_line_length=max_line_length,
                max_lines=max_lines,
                annotate_speakers=annotate_speakers,
            )
            # ... and its cleanup can push a boundary back across a cut, so snap again.
            cues = resnap_shots(cues, cuts, effective)
    # Fill the reserved key in place: assigning an existing key never reorders a
    # dict, so the persisted ``segmentation`` block is byte-identical to the one
    # built after the run.
    manifest["degraded"] = degraded
    if shadow is not None:
        # AD3-5/AD4-4: two origin-typed ledgers and never one merged ``degraded``.
        # ``production_degraded`` can only be copied here, once the outer capture
        # has closed and the manifest's reserved key holds the real list;
        # ``shadow_degraded`` was collected by the hook's own nested capture.
        shadow["production_degraded"] = copy.deepcopy(manifest["degraded"])
        # AD3-5's other half. Without it an artifact reader cannot tell which
        # language providers the measured run actually resolved -- and a run
        # whose ja POS tagger fell back to the character table is measuring a
        # different boundary decision than one whose tagger loaded.
        shadow["providers"] = copy.deepcopy(manifest["providers"])
    diagnostics: dict[str, Any] = {
        "unit_count": len(snapped),
        "punct_snapped": snapped is not units,
        "adaptive_thresholds": effective is not base,
        "speech_span_count": len(speech_spans or ()),
        "shot_change_count": len(cuts or ()),
        "sing_span_count": len(sings or ()),
        "speaker_turn_count": len(turns or ()),
        "split_cue_count": split_cue_count,
        "lyric_cue_count": sum(1 for c in cues if c.get("lyric")),
        "speaker_formatted": bool(turns),
        "shot_resnapped": bool(turns and cuts),
        "cue_count": len(cues),
        "shadow_v2": shadow is not None,
    }
    return SegmentationResult(
        cues=cues,
        language=iso,
        units=snapped,
        diagnostics=diagnostics,
        manifest=manifest,
        document=document,
        shadow=shadow,
    )
