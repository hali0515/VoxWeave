"""The ``translate`` and ``correct`` commands: LLM passes over a finished subtitle.

Both read a subtitle, send its cues to an OpenAI-compatible endpoint through
:mod:`voxweave.translate` / :mod:`voxweave.asrfix` and write the result. Neither
runs the audio pipeline; ``correct --apply`` only calls
:func:`voxweave.pipeline.align` afterwards to refresh the timing of the text it
changed, which is why this module sits above the pipeline and not inside it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voxweave import (
    artifacts,
    config,
    episode_transaction,
    fsio,
    pipeline,
    realign,
    sidecars,
)
from voxweave import asrfix as asrfix_mod
from voxweave import translate as translate_mod
from voxweave.export import render_ass, render_srt
from voxweave.lang import to_iso_or
from voxweave.paths import swap_ext
from voxweave.progress import NestedReporter, Reporter
from voxweave.subformats import (
    load_subtitle_blocks,
    load_subtitle_blocks_bytes,
    require_subtitle,
)

log = logging.getLogger("voxweave")


def _load_cues(vtt_path: Path) -> list[dict]:
    """Parse subtitle cue blocks by extension (VTT/SRT/ASS/SSA); raise if the
    file has no cues. Used by translate (align and correct decode the exact bytes
    they snapshot instead)."""
    return load_subtitle_blocks(Path(vtt_path))


def translate(
    vtt_path: Path,
    *,
    to: str = "zh",
    context: str | None = None,
    glossary: dict[str, str] | str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    concurrency: int | None = None,
    window_cues: int | None = None,
    allow_partial: bool = False,
    reporter: Reporter | None = None,
) -> Path:
    """Translate subtitle cues via an OpenAI-compatible endpoint; write
    <stem>.<to>.<ext> (source untouched).

    Accepts VTT/SRT/ASS/SSA; the output mirrors the input format (SSA is written
    back as ASS). Output cue count always equals input cue count.

    ``concurrency`` / ``window_cues`` (None = env > conf ``[llm]`` > built-in 8 /
    100) run bounded windows in parallel; ``concurrency=1`` is the single
    whole-episode request with translated-tail continuity. Missing translations
    are retried once, sequentially. Cues still missing afterwards fail the run
    with :class:`voxweave.translate.PartialTranslationError` and keep the
    progress sidecar so a rerun resumes; ``allow_partial=True`` writes the file
    anyway with those cues back-filled from the source text (logged).
    """
    vtt_path = require_subtitle(Path(vtt_path))
    # None = env VOXWEAVE_TRANSLATE_MODEL > conf [llm].model > built-in, read now
    # rather than at import so a library caller sees the current config.
    model = model or config.resolve_llm_model(
        None, task_envvar="VOXWEAVE_TRANSLATE_MODEL"
    )
    base_url = config.resolve_llm_base_url(base_url)
    reasoning_effort = config.resolve_llm_reasoning_effort(reasoning_effort)
    concurrency = config.resolve_llm_concurrency(concurrency)
    window_cues = config.resolve_llm_window_cues(window_cues)
    ext = ".ass" if vtt_path.suffix.lower() == ".ssa" else vtt_path.suffix.lower()
    rep = reporter or Reporter()
    rep.plan(("read subtitles", "translate cues", "write translation"))
    rep.step("read subtitles")
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
    progress_owner = sidecars.artifact_owner(vtt_path)
    progress_path = artifacts.translation_progress_path(
        progress_owner,
        vtt_path,
        to,
    )
    progress_candidates = artifacts.translation_progress_candidates(
        progress_owner,
        vtt_path,
        to,
    )
    tx_kwargs: dict[str, Any] = dict(
        to=to,
        model=model,
        context=context,
        glossary=glossary,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
        progress_path=progress_path,
        progress_sig=translate_mod.payload_signature(payload),
        concurrency=concurrency,
        window_cues=window_cues,
    )
    rep.step("translate cues")
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
            # The retry stage is sequential (translated-tail continuity); after a
            # concurrent main stage it keeps the operator's window size so a large
            # gap is not re-requested as one oversized window.
            retry_kwargs: dict[str, Any] = {**tx_kwargs, "concurrency": 1}
            if concurrency > 1:
                retry_kwargs["batch"] = window_cues
            trans.update(
                translate_mod.translate_cues(
                    retry_payload, **retry_kwargs, tail=tail, reporter=rep
                )
            )
            still = translate_mod.validate_and_fill(blocks, trans)
            if still and not allow_partial:
                raise translate_mod.PartialTranslationError(still, len(blocks))
            if still:
                log.warning(
                    "%d of %d cues still untranslated after retry; --allow-partial: "
                    "back-filling them with source text: %s",
                    len(still),
                    len(blocks),
                    still,
                )
    except Exception as exc:
        if progress_path.exists():
            # PartialTranslationError is not an interruption: the run finished and
            # refused to write a part-source file.
            log.warning(
                "%s; progress saved to %s -- rerun the same command to resume",
                "translation incomplete"
                if isinstance(exc, translate_mod.PartialTranslationError)
                else "translation interrupted",
                progress_path.name,
            )
        raise

    rep.step("write translation")
    rep.stage(f"write translated {ext.lstrip('.').upper()}")
    rows = translate_mod.translated_rows(
        blocks,
        trans,
        to_iso=to_iso_or(to, None),
        voice_tags=ext == ".vtt",
    )
    if ext == ".vtt":
        content = realign.render_cues(rows)
    else:
        timed = [
            (float(s), float(e), t)
            for s, e, t in rows
            if s is not None and e is not None
        ]
        timed_blocks = [
            translate_mod._speaker_block_for_rendered(block, text)
            for block, (start, end, text) in zip(blocks, rows)
            if start is not None and end is not None
        ]
        content = (
            render_srt(timed, blocks=timed_blocks)
            if ext == ".srt"
            else render_ass(timed, blocks=timed_blocks)
        )
    out_path = swap_ext(vtt_path, f".{to}{ext}")
    fsio.atomic_write_text(out_path, content)
    try:
        live_progress_candidates = artifacts.translation_progress_candidates(
            progress_owner,
            vtt_path,
            to,
        )
    except (artifacts.ArtifactMarkerError, OSError) as exc:
        live_progress_candidates = None
        log.warning("translation progress cleanup authority changed: %s", exc)
    cleanup_failed = live_progress_candidates != progress_candidates
    if not cleanup_failed:
        for candidate in (
            path for path in progress_candidates if path != progress_path
        ):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_failed = True
                log.warning(
                    "translation progress cleanup failed for %s: %s", candidate, exc
                )
    if cleanup_failed:
        log.warning(
            "translation progress cleanup is incomplete; preserving selected lane %s",
            progress_path,
        )
    else:
        try:
            progress_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning(
                "translation progress cleanup failed for selected lane %s: %s",
                progress_path,
                exc,
            )
    log.info("wrote %s (%d cues → %s)", out_path.name, len(blocks), to)
    return out_path


def _warn_correction_not_realigned(vtt_path: Path, applied: Sequence[Mapping]) -> None:
    """Warn that ``vtt_path`` was corrected but not re-aligned, listing the diff."""
    shown = [
        f"  #{fix.get('i')}: {fix.get('orig')!r} -> {fix.get('fixed')!r}"
        for fix in applied[:20]
    ]
    if len(applied) > len(shown):
        shown.append(f"  ... and {len(applied) - len(shown)} more")
    log.warning(
        "%s was corrected in place (%d change(s)) but re-alignment failed, so its"
        " timing is stale; run `voxweave align %s` (add --media if the source is not"
        " beside it) to refresh it. Applied changes:\n%s",
        vtt_path.name,
        len(applied),
        vtt_path,
        "\n".join(shown),
    )


def correct(
    vtt_path: Path,
    *,
    glossary: dict[str, str] | str | None = None,
    model: str | None = None,
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

    Default (review): writes adjacent sidecar ``<stem>.asrfix.vtt`` plus an audit in the
    per-media artifact cache or an existing adjacent legacy audit lane, source untouched.
    ``apply``: overwrites the original VTT in place and writes **no audit json** (the diff is
    shown in the summary). When ``align_after`` and a real change was applied, immediately
    re-runs :func:`voxweave.pipeline.align` to refresh timestamps (text edits change word
    counts) and update the sibling ``<stem>.json``. An explicit ``media_path`` that does not
    exist is rejected before the LLM call; if the re-alignment fails after the in-place write,
    the applied diff is logged (re-running correct would find nothing left to change) before
    the error propagates.

    Returns ``{out, audit, applied, rejected, n_cues, applied_in_place, aligned}``.
    """
    # --apply overwrites the input as VTT
    vtt_path = pipeline.require_vtt(Path(vtt_path))
    # None = env VOXWEAVE_FIX_MODEL > conf [llm].model > built-in (see translate()).
    model = model or config.resolve_llm_model(None, task_envvar="VOXWEAVE_FIX_MODEL")
    if apply and align_after and media_path is not None:
        # --apply commits before the re-alignment, so a --media that align would
        # reject must fail here, before the LLM call and the in-place overwrite.
        if not Path(media_path).exists():
            raise pipeline._media_not_found_error(vtt_path)
    owner = (
        Path(media_path)
        if media_path is not None
        else sidecars.artifact_owner(vtt_path)
    )
    rep = reporter or Reporter()
    steps = ["read subtitles", "correct text", "write correction"]
    if apply and align_after:
        steps.append("check and refresh timing")
    rep.plan(steps)
    rep.step("read subtitles")
    try:
        vtt_input_bytes = vtt_path.read_bytes()
    except OSError as exc:
        pipeline._attach_canonical_failure(
            exc,
            kind="subtitle-snapshot-failed",
            phase="snapshot",
            detail_code="vtt-read",
        )
        raise
    blocks = load_subtitle_blocks_bytes(vtt_path, vtt_input_bytes)

    payload = asrfix_mod.build_payload(blocks)
    rep.step("correct text")
    rep.stage(f"LLM correction {len(payload)} cues (model={model})")
    fixes = asrfix_mod.correct_cues(
        payload, model=model, glossary=glossary, base_url=base_url, api_key=api_key
    )
    new_texts, applied, rejected = asrfix_mod.apply_fixes(blocks, fixes)
    rendered = asrfix_mod.render_vtt(blocks, new_texts)

    rep.step("write correction")
    audit_path: Path | None = None
    if apply:
        # in-place edit: overwrite the original, no sidecar json (diff lives in the summary)
        rep.stage("overwrite VTT in place")
        rendered_bytes = rendered.encode("utf-8")
        episode_transaction.commit_correction(
            episode_path=owner,
            vtt_path=vtt_path,
            expected_vtt=episode_transaction.FileGeneration(True, vtt_input_bytes),
            rendered_vtt_bytes=rendered_bytes,
            evidence_paths=artifacts.align_evidence_candidates(
                owner,
                vtt_path,
            ),
        )
        out_path = vtt_path
    else:
        rep.stage("write sidecar VTT + audit json")
        out_path = swap_ext(vtt_path, ".asrfix.vtt")
        audit_path = artifacts.asrfix_audit_path(owner, vtt_path)
        fsio.atomic_write_text(out_path, rendered)
        try:
            fsio.atomic_write_text(
                audit_path,
                json.dumps(
                    {"applied": applied, "rejected": rejected},
                    ensure_ascii=False,
                    indent=2,
                ),
                private=True,
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
    if apply and align_after:
        rep.step("check and refresh timing")
        if not applied:
            rep.stage("no text changes; alignment not needed")
    if apply and align_after and applied:
        try:
            pipeline.align(
                out_path,
                media_path=media_path,
                separate=separate,
                normalize=normalize,
                lang_override=lang_override,
                reporter=NestedReporter(rep),
                _expected_vtt_sha256=hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
            )
        except BaseException:
            # The text is already committed: re-running correct finds nothing left to
            # change, so keep the diff the summary would have shown and say how to
            # finish the job.
            _warn_correction_not_realigned(out_path, applied)
            raise
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
