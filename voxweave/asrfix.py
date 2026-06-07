"""LLM-based ASR error correction, run between ``process`` and ``align``.

Fixes clear transcription errors (homophones, wrong hanzi/kanji, split words, garbled
proper nouns) while changing as little as possible. Corrected text feeds into ``align``
which re-derives timestamps, absorbing any length changes.

Conservative by design: the model is told to substitute only, never invent or rephrase.
A hard SAFETY GATE (:func:`apply_fixes`) applies a fix only when its quoted ``orig``
matches the actual cue text, rejecting cross-cue word splits, hallucinated quotes, and
no-ops. Enforced in code, not trusted to the prompt.

Reuses OpenAI plumbing from :mod:`voxweave.translate`; only the system prompt and
response schema differ (fixes diff instead of translations).
"""

from __future__ import annotations

import json
import logging
import os

from voxweave.realign import render_cues
from voxweave.translate import (
    _call,
    _loads_salvage,
    _make_client,
    build_payload,
    format_glossary,
)

log = logging.getLogger("voxweave")

FIX_MODEL = os.environ.get("VOXWEAVE_FIX_MODEL", "gpt-5.3-chat-latest")

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
    payload: list[dict], *, glossary: dict[str, str] | str | None = None
) -> list[dict]:
    """system (prompt + optional glossary) + user (numbered cue JSON)."""
    system = SYSTEM_PROMPT
    gl = format_glossary(glossary)
    if gl:
        system += (
            "\n\nGLOSSARY (canonical entities for THIS video — authoritative):\n" + gl
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


def apply_fixes(
    blocks: list[dict], fixes: list[dict]
) -> tuple[list[str], list[dict], list[dict]]:
    """SAFETY GATE. Returns ``(new_texts, applied, rejected)``.

    A fix is applied only when: index is in range, ``orig`` matches the actual cue text
    (whitespace-normalized), and ``fixed`` differs. Everything else is rejected with a
    ``_why`` reason — this blocks cross-cue word splits, hallucinated quotes, and no-ops
    observed in testing. Enforced in code, never trusted to the model.
    """
    n = len(blocks)
    new_texts = [b["text"] for b in blocks]
    applied: list[dict] = []
    rejected: list[dict] = []
    for f in fixes:
        i = f["i"]
        if not (0 <= i < n):
            rejected.append({**f, "_why": "index out of range"})
            continue
        actual = blocks[i]["text"]
        if _norm(f["orig"]) != _norm(actual):
            rejected.append({**f, "_why": "orig != cue (cross-cue split / misquote)"})
            continue
        if _norm(f["fixed"]) == _norm(actual):
            rejected.append({**f, "_why": "no-op"})
            continue
        new_texts[i] = f["fixed"]
        applied.append(f)
    return new_texts, applied, rejected


def render_vtt(blocks: list[dict], texts: list[str]) -> str:
    """Render cues with corrected ``texts``, preserving each block's timestamps
    when present (text-only otherwise). Structure-preserving: one cue in, one out."""
    return render_cues(
        [(b.get("start"), b.get("end"), text) for b, text in zip(blocks, texts)]
    )


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

    Single call over the whole transcript (full context ensures consistent entity
    normalization); not windowed unlike translate. No progress bar: the model emits
    only changed cues so a per-cue bar would barely move. ``client`` injectable for tests.
    """
    if not payload:
        return []
    client = client or _make_client(base_url, api_key)
    messages = build_messages(payload, glossary=glossary)
    raw = _call(client, model, messages)
    return parse_fixes(raw)


__all__ = [
    "FIX_MODEL",
    "SYSTEM_PROMPT",
    "build_messages",
    "parse_fixes",
    "apply_fixes",
    "render_vtt",
    "correct_cues",
    "build_payload",
]
