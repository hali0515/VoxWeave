"""LLM-based ASR error correction, run between ``process`` and ``align``.

Fixes clear transcription errors (homophones, wrong hanzi/kanji, split words, garbled
proper nouns) while changing as little as possible. Corrected text feeds into ``align``
which re-derives timestamps, absorbing any length changes.

Conservative by design: the model is told to substitute only, never invent or rephrase.
A hard SAFETY GATE (:func:`apply_fixes`) applies a fix only when its quoted ``orig``
matches the actual cue text and the replacement stays a minimal edit, rejecting
cross-cue word splits, hallucinated quotes, no-ops, cue erasure, expansions, and
wholesale rewrites. Enforced in code, not trusted to the prompt.

Reuses OpenAI plumbing from :mod:`voxweave.translate` -- including its request
ladder (:func:`voxweave.translate.request_with_json_fallback`: retries, then one
plain-chat attempt) -- so a flaky endpoint behaves the same for both commands;
only the system prompt and response schema differ (fixes diff instead of
translations). The whole transcript still goes out as ONE request; splitting is
a last resort for a response the endpoint truncated (see :func:`correct_cues`).
"""

from __future__ import annotations

import difflib
import json
import logging

from voxweave import config
from voxweave.realign import render_cues
from voxweave.speakers import voice_text_for_block
from voxweave.translate import (
    IncompleteResponse,
    _call,
    _call_options,
    _loads_salvage,
    _make_client,
    build_payload,
    format_glossary,
    request_with_json_fallback,
    resolve_model,
    restore_dash_layout,
)

log = logging.getLogger("voxweave")

# Built-in only; env / conf [llm] resolve at call time (see translate.TRANSLATE_MODEL).
FIX_MODEL = config.DEFAULT_LLM_MODEL
# Preceding cue texts handed to a half after a split, so it keeps a little of the
# whole-transcript context the single unsplit request has (proper-noun continuity).
SPLIT_CONTEXT_TAIL = 3

SYSTEM_PROMPT = """\
You are an expert subtitle transcription proofreader. The input is an automatic
speech recognition (ASR) transcript of ONE video, split into numbered cues, in
its original language. ASR hears the sounds correctly but often writes the WRONG
characters/words: homophones, wrong hanzi/kanji, split words, and especially
garbled proper nouns (people, brands, products, technical terms, tickers, model
numbers).

Your job: fix CLEAR transcription errors while changing as LITTLE as possible.

HARD RULES (breaking these is worse than leaving an error in):
1. NEVER ADD MEANING. Do NOT insert words, complete unfinished sentences, fill
   in dropped words, or "restore" missing content. If a cue looks truncated or is
   missing words, LEAVE IT UNCHANGED. Inventing words that were not spoken is the
   worst possible error.
2. NO REPHRASING. Do not paraphrase, summarize, reorder, or improve wording,
   tone, grammar, or style. Keep the exact wording; only correct mis-written
   characters/words, and rejoin words ASR split apart (e.g. "ne ed" -> "need").
3. You MAY DELETE only obvious ASR artifacts: a duplicated character/word the
   speaker did not repeat (e.g. "做到到" -> "做到"), or a stray inserted token.
   Deleting is allowed ONLY for clear ASR doubling/garbage, never to shorten or
   clean up real speech.
4. ONE CUE AT A TIME. Each fix must stay WITHIN a single cue. If a word is split
   across two cues (one cue ends mid-word, the next begins with the rest), LEAVE
   BOTH UNCHANGED — never pull text across the cue boundary.
5. DO NOT add, remove, or change punctuation.
6. PRESERVE STRUCTURE: exactly one corrected text per changed cue, same index.
   Never merge or split cues.
7. STAY IN THE ORIGINAL LANGUAGE. Do NOT translate.
8. WHEN UNSURE, LEAVE IT UNCHANGED. A correct-but-unusual real name stays as-is.

PROPER NOUNS & RECURRING ENTITIES (highest-value fixes):
- ASR often writes the SAME name/term DIFFERENTLY across cues (several garbled
  variants of one entity).
- The GLOSSARY below (if provided) is AUTHORITATIVE: it lists canonical entities
  for this video. Map every garbled phonetic variant to its glossary entry and
  apply it CONSISTENTLY across all cues. Trust the glossary over your own guess.
- For recurring entities NOT in the glossary, infer the canonical form from the
  video's topic/domain and normalize all occurrences consistently; if you cannot
  identify it confidently, leave it unchanged rather than guess.
- If the transcript is clearly from a SPECIFIC identifiable work (film, show,
  book, game, franchise), you MAY use that work's established canonical spellings
  for its characters, places, and INVENTED terminology to repair obviously
  phonetic ASR garbles (e.g. a sci-fi coinage the ASR spelled by sound). Apply
  this ONLY to a clear garble of a term you are confident belongs to that work,
  and prefer the work's specific coined term over a generic common word when the
  same garbled form recurs (e.g. a coined organism name, not "amoeba"). Never
  rename a plausibly-correct word and never alter ordinary dialogue.

OUTPUT: JSON only. Report ONLY cues you actually changed (the fixed text MUST
differ from the original), so the edit is a reviewable diff:
{"fixes":[{"i":<index>,"orig":"<original cue text, verbatim>","fixed":"<corrected text>","reason":"<short reason>"}]}
If nothing needs fixing, return {"fixes":[]}."""


def build_messages(
    payload: list[dict],
    *,
    glossary: dict[str, str] | str | None = None,
    source_tail: list[str] | None = None,
) -> list[dict]:
    """system (prompt + optional glossary + optional preceding cues) + user (numbered cue JSON).

    ``source_tail`` carries the cue texts immediately before ``payload`` as
    read-only context; only a split request (see :func:`correct_cues`) uses it, and
    those cues live in the system message so they can never be mistaken for cues to
    report fixes for.
    """
    system = SYSTEM_PROMPT
    gl = format_glossary(glossary)
    if gl:
        system += (
            "\n\nGLOSSARY (canonical entities for THIS video — authoritative):\n" + gl
        )
    if source_tail:
        tail_txt = "\n".join(t for t in source_tail if t)
        if tail_txt:
            system += (
                "\n\nPRECEDING CUES (context only — earlier cues of the same "
                "transcript, for entity consistency; they are NOT in the cue list "
                "below, so never report a fix for them):\n" + tail_txt
            )
    user = json.dumps({"cues": payload}, ensure_ascii=False)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_fixes(raw: object) -> list[dict]:
    """Model response → list of ``{i, orig, fixed, reason}``; salvages the first
    JSON object from dirty text, returns [] on failure."""
    items = _loads_salvage(raw).get("fixes", [])
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            out.append(
                {
                    "i": int(it["i"]),
                    "orig": str(it.get("orig", "")),
                    "fixed": str(it.get("fixed", "")),
                    "reason": str(it.get("reason", "")),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _norm(s: str) -> str:
    """Collapse all whitespace (cue text may carry wrap ``\\n``; the model quotes
    it space-joined) for tolerant orig/fixed comparison."""
    return " ".join(s.split())


# Semantic gates on the fixed text. Growth budget: legit fixes substitute or
# shrink; the only legal growth is a short garble replaced by a canonical term
# (often cross-script, e.g. a 3-hanzi garble -> a Latin glossary entry), which
# the absolute slack covers. The similarity gate only judges cues long enough
# for SequenceMatcher to be meaningful; short cues rely on the growth budget.
_GROWTH_FACTOR = 1.5
_GROWTH_SLACK = 8
_REWRITE_MIN_LEN = 12
_REWRITE_MIN_RATIO = 0.4


def _semantic_reject(actual_n: str, fixed_n: str) -> str | None:
    """Why ``fixed`` violates the minimal-edit promise, or None when it is fine."""
    if not fixed_n:
        return "empty replacement"
    if len(fixed_n) > _GROWTH_FACTOR * len(actual_n) + _GROWTH_SLACK:
        return "expansion (added content)"
    if len(actual_n) >= _REWRITE_MIN_LEN:
        ratio = difflib.SequenceMatcher(None, actual_n, fixed_n).ratio()
        if ratio < _REWRITE_MIN_RATIO:
            return f"rewrite (similarity {ratio:.2f})"
    return None


def apply_fixes(
    blocks: list[dict], fixes: list[dict]
) -> tuple[list[str], list[dict], list[dict]]:
    """SAFETY GATE. Returns ``(new_texts, applied, rejected)``.

    A fix is applied only when: index is in range, ``orig`` matches the actual cue text
    (whitespace-normalized), ``fixed`` differs, and ``fixed`` honors the minimal-edit
    promise (non-empty, within the growth budget, not a wholesale rewrite). Everything
    else is rejected with a ``_why`` reason — this blocks cross-cue word splits,
    hallucinated quotes, no-ops, cue erasure, and sentence "completion". Enforced in
    code, never trusted to the model.

    Several fixes for one cue compose: a later fix must quote the cue as the earlier
    ones left it (it then applies on top, still gated against the original cue), else
    it is rejected as a ``"duplicate index"`` -- never silently overwriting an earlier
    fix while the audit reports both as applied.
    """
    n = len(blocks)
    new_texts = [b["text"] for b in blocks]
    changed: set[int] = set()
    applied: list[dict] = []
    rejected: list[dict] = []
    for f in fixes:
        i = f["i"]
        if not (0 <= i < n):
            rejected.append({**f, "_why": "index out of range"})
            continue
        actual = new_texts[i]  # the cue as earlier fixes left it
        actual_n = _norm(actual)
        fixed_n = _norm(f["fixed"])
        if _norm(f["orig"]) != actual_n:
            why = (
                "duplicate index"
                if i in changed
                else "orig != cue (cross-cue split / misquote)"
            )
            rejected.append({**f, "_why": why})
            continue
        if fixed_n == actual_n:
            rejected.append({**f, "_why": "no-op"})
            continue
        # the minimal-edit promise holds against the transcribed cue, so composed
        # fixes cannot creep past the budgets one step at a time
        why = _semantic_reject(_norm(blocks[i]["text"]), fixed_n)
        if why is not None:
            rejected.append({**f, "_why": why})
            continue
        fixed = f["fixed"]
        if isinstance(blocks[i].get("speakers"), list):
            fixed = restore_dash_layout(actual, fixed)
        new_texts[i] = fixed
        changed.add(i)
        applied.append({**f, "orig": actual, "fixed": fixed})
    return new_texts, applied, rejected


def render_vtt(blocks: list[dict], texts: list[str]) -> str:
    """Render cues with corrected ``texts``, preserving each block's timestamps
    when present (text-only otherwise). Structure-preserving: one cue in, one out;
    lyric and speaker display metadata are restored after correction."""
    return render_cues(
        [
            (
                block.get("start"),
                block.get("end"),
                voice_text_for_block(
                    f"♪ {text} ♪" if block.get("lyric") else text, block
                ),
            )
            for block, text in zip(blocks, texts)
        ]
    )


class _UnparseableFixes(IncompleteResponse):
    """The response carried no fix list at all — not even an empty one.

    An :class:`~voxweave.translate.IncompleteResponse` on purpose, so the retry
    ladder re-requests it: reading garbage as "the model found nothing to fix"
    would silently pass a whole transcript through uncorrected.
    """

    def __init__(self, raw: object) -> None:
        super().__init__(
            None,
            raw if isinstance(raw, str) else "",
            detail="no parseable fix list in the response",
        )


class _OutputCapped(RuntimeError):
    """Carries a ``finish_reason == "length"`` :class:`IncompleteResponse` past the
    request ladder.

    Deliberately NOT an IncompleteResponse: translate._retryable retries every one of
    those and request_with_json_fallback then re-sends it as plain chat, yet the same
    cue set hits the same output cap every time. :func:`_request_fixes` unwraps it so
    :func:`_correct_window` splits the cue set straight away.
    """

    def __init__(self, response: IncompleteResponse) -> None:
        super().__init__(str(response))
        self.response = response


def _payload_indices(payload: list[dict]) -> set[int]:
    """Cue indices a request asks about (:func:`build_payload` numbers every entry).
    Entries without a usable ``i`` are skipped; an entirely unnumbered payload
    yields an empty set, which disables index filtering rather than dropping
    every fix."""
    out: set[int] = set()
    for entry in payload:
        try:
            out.add(int(entry["i"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _request_label(payload: list[dict]) -> str:
    ids = _payload_indices(payload)
    if not ids:
        return f"correct ({len(payload)} cues)"
    return f"correct (cues {min(ids)}-{max(ids)})"


def _parse_fix_window(raw: object, wanted: set[int], *, label: str) -> list[dict]:
    """One response → its fix list, restricted to the cues this request asked about.

    An answer with no ``fixes`` array raises :class:`_UnparseableFixes` so the
    retry ladder runs; ``{"fixes": []}`` is a legitimate "nothing to fix" answer
    and returns ``[]``. Indices the request never sent are dropped here so a split
    request can never report a fix against a cue outside its own half.
    """
    doc = _loads_salvage(raw)
    if not isinstance(doc.get("fixes"), list):
        raise _UnparseableFixes(raw)
    fixes = parse_fixes(doc)
    if not wanted:
        return fixes
    stray = [f["i"] for f in fixes if f["i"] not in wanted]
    if stray:
        log.warning("%s: dropping out-of-request fix indices %s", label, stray)
    return [f for f in fixes if f["i"] in wanted]


def _request_fixes(
    client,
    model: str,
    payload: list[dict],
    *,
    glossary: dict[str, str] | str | None,
    source_tail: list[str] | None,
    label: str,
) -> list[dict]:
    """One cue set → its fix list, using translate's request ladder (json_object
    with retries, then a single plain-chat attempt). Still-incomplete propagates.

    A response cut off by the output cap (``finish_reason == "length"``) skips the
    ladder and propagates at once: re-requesting the same cue set cannot fit it.
    """
    messages = build_messages(payload, glossary=glossary, source_tail=source_tail)
    wanted = _payload_indices(payload)

    def attempt(json_mode: bool) -> list[dict]:
        try:
            raw = _call(client, model, messages, **_call_options(None, json_mode))
        except IncompleteResponse as exc:
            if exc.finish_reason == "length":
                raise _OutputCapped(exc) from exc
            raise
        return _parse_fix_window(raw, wanted, label=label)

    try:
        return request_with_json_fallback(
            attempt, label=label, call_label=f"{label} correct call"
        )
    except _OutputCapped as capped:
        raise capped.response from None


def _splittable(exc: IncompleteResponse) -> bool:
    """Whether a smaller request could plausibly succeed where this one failed.

    Only two failures are about size: the output cap (``finish_reason == "length"``
    — retrying the same request cannot help) and an answer that stayed unparseable
    through the plain-chat fallback. A server-side abort or a dropped stream is not
    a size problem, so it raises instead of fanning out requests at a sick endpoint.
    """
    return isinstance(exc, _UnparseableFixes) or exc.finish_reason == "length"


def _correct_window(
    client,
    model: str,
    payload: list[dict],
    *,
    glossary: dict[str, str] | str | None,
    source_tail: list[str] | None,
) -> list[dict]:
    """Request fixes for ``payload``, halving it when the answer will not fit.

    Fix indices stay absolute throughout: each half keeps the ``i`` values
    :func:`build_payload` assigned, so the merged list needs no remapping and
    :func:`apply_fixes` stays the sole authority on what is applied.
    """
    label = _request_label(payload)
    try:
        return _request_fixes(
            client,
            model,
            payload,
            glossary=glossary,
            source_tail=source_tail,
            label=label,
        )
    except IncompleteResponse as exc:
        if len(payload) < 2 or not _splittable(exc):
            raise
        log.warning(
            "%s: %s (%s); splitting the cue set in half and correcting each half "
            "separately",
            label,
            "response hit the output length cap"
            if exc.finish_reason == "length"
            else "still incomplete after the plain-chat fallback",
            exc,
        )
    mid = len(payload) // 2
    head, tail = payload[:mid], payload[mid:]
    fixes = _correct_window(
        client, model, head, glossary=glossary, source_tail=source_tail
    )
    fixes += _correct_window(
        client,
        model,
        tail,
        glossary=glossary,
        source_tail=[str(entry.get("t", "")) for entry in head][-SPLIT_CONTEXT_TAIL:],
    )
    return fixes


def correct_cues(
    payload: list[dict],
    *,
    model: str = FIX_MODEL,
    glossary: dict[str, str] | str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    client=None,
) -> list[dict]:
    """Numbered payload → raw fix list (pre-gate).

    Normally a single call over the whole transcript: full context and the glossary
    are what make entity normalization consistent, so correct does NOT window by
    default the way translate does. No progress bar: the model emits only changed
    cues so a per-cue bar would barely move. ``client`` injectable for tests.

    A response that does not finish with ``"stop"`` is never read as the model's
    full review. It goes through translate's request ladder instead — retries in
    ``json_object`` mode, then one plain-chat attempt for servers whose
    structured-output path aborts (vLLM's grammar FSM). A size failure halves the
    cue set -- immediately for ``finish_reason == "length"`` (the same request would
    hit the same output cap, so the ladder is skipped), after the ladder for an answer
    that stays unparseable -- and each half is requested separately, carrying the preceding
    cues as context; the halves keep their original cue indices and their fixes are
    merged. A single cue that still fails raises
    :class:`voxweave.translate.IncompleteResponse`.
    """
    if not payload:
        return []
    client = client or _make_client(base_url, api_key)
    model = resolve_model(client, model)
    return _correct_window(
        client, model, list(payload), glossary=glossary, source_tail=None
    )


__all__ = [
    "FIX_MODEL",
    "SPLIT_CONTEXT_TAIL",
    "SYSTEM_PROMPT",
    "build_messages",
    "parse_fixes",
    "apply_fixes",
    "render_vtt",
    "correct_cues",
    "build_payload",
]
