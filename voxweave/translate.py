from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from voxweave.realign import render_cues

log = logging.getLogger("voxweave")

TRANSLATE_MODEL = os.environ.get("VOXWEAVE_TRANSLATE_MODEL", "gpt-5.3-chat-latest")
# Set high (800) so a typical episode (300-500 cues) fits in one call — full-episode context
# beats windowing: seams cause disambiguation errors and inconsistent proper-noun rendering.
# Only very long compilations (>800 cues) fall back to sequential windows.
BATCH_THRESHOLD = int(os.environ.get("VOXWEAVE_TRANSLATE_BATCH", "800"))
# Tail cues from the previous window carried into the next for stylistic continuity.
CONTEXT_TAIL = int(os.environ.get("VOXWEAVE_TRANSLATE_CONTEXT_TAIL", "3"))


# Decorative punctuation GPT commonly produces but not covered by voxweave’s base stripper.
# Excludes · ・ (name joiners), ー (kana long-vowel mark), - (hyphen/digits).
_EXTRA_PUNCT_TO_SPACE_RE = re.compile(
    "["
    "「」『』｢｣〝〟"  # CJK quotation marks
    "“”‘’"  # curly quotes
    "（）()【】〔〕《》〈〉［］｛｝"  # brackets (full-width and half-width)
    "…⋯"  # ellipses
    "—–―"  # em dash, en dash, horizontal bar
    "]"
)


def strip_punct_for_subtitles(text: str) -> str:
    """Strip decorative punctuation from translated text, then apply voxweave's base stripper.

    Decimal/thousands separators (digit.digit, digit,digit) and name joiners · ・ are preserved.
    """
    from voxweave.core.layout import strip_punct_for_subtitles

    return strip_punct_for_subtitles(_EXTRA_PUNCT_TO_SPACE_RE.sub(" ", text))


def build_payload(blocks: list[dict]) -> list[dict]:
    """Cue blocks -> numbered list [{i, t}]; multi-line cue text is joined into a single string."""
    return [
        {"i": idx, "t": " ".join(b["text"].split("\n")).strip()}
        for idx, b in enumerate(blocks)
    ]


def _loads_salvage(raw: object) -> dict:
    """Decode a model response to a dict, salvaging the first complete JSON object from dirty
    text (markdown fences / prose preamble). Returns {} when nothing parseable is found.

    Shared by parse_response (translations) and asrfix.parse_fixes (fixes).
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}  # non-str/non-dict (e.g. None): tolerate like the original inline parsers
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            return {}
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}


def parse_response(raw: object) -> dict[int, str]:
    """Structured model output -> {index: translated text}; salvages the first complete JSON object from dirty text, returns {} on failure."""
    items = _loads_salvage(raw).get("translations", [])
    out: dict[int, str] = {}
    for it in items:
        try:
            out[int(it["i"])] = str(it["t"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def validate_and_fill(blocks: list[dict], trans: dict[int, str]) -> list[int]:
    """Return indices of cues with missing or blank translations (for retry)."""
    return [i for i in range(len(blocks)) if not trans.get(i, "").strip()]


def _layout_translated(text: str, to_iso: str | None) -> str:
    """Soft-wrap a translated cue to the TARGET language's display budget.

    Cue count and timing are fixed by the source speech, so wrapping is the only
    layout valve for translations that outgrow the line budget (an en cue packed
    to 2x42 chars can translate into 30+ zh chars). <=2 lines for every target:
    short text stays on one line (the wrap only triggers past the visual budget),
    CJK gets kinsoku inside wrap_cue_text.
    """
    if not to_iso:
        return text
    from voxweave.core.layout import wrap_cue_text

    return wrap_cue_text(text, to_iso, 2)


def render_translated_vtt(
    blocks: list[dict], trans: dict[int, str], to_iso: str | None = None
) -> str:
    """Translated text + per-block timestamps -> VTT; missing translations fall back
    to the original text; blocks without timestamps produce plain-text cues.
    ``to_iso`` enables target-language soft-wrap (see _layout_translated)."""
    rows = [
        (
            b.get("start"),
            b.get("end"),
            _layout_translated(
                strip_punct_for_subtitles(trans.get(i, "").strip() or b["text"]),
                to_iso,
            ),
        )
        for i, b in enumerate(blocks)
    ]
    return render_cues(rows)


def format_glossary(glossary: dict[str, str] | str | None) -> str:
    """Glossary -> prompt fragment; dict is rendered as 'source -> translation' lines, str is returned as-is."""
    if not glossary:
        return ""
    if isinstance(glossary, str):
        return glossary
    return "\n".join(f"{k} -> {v}" for k, v in glossary.items())


def load_glossary(path) -> dict[str, str] | str:
    """Load a glossary file: .json is parsed into a dict; any other extension is returned as a raw string."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    return text.strip()


def build_messages(
    payload: list[dict],
    *,
    to: str,
    context: str | None = None,
    glossary: dict[str, str] | str | None = None,
    tail: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Assemble OpenAI chat messages: system (instructions + context + glossary + continuity tail) + user (numbered payload JSON)."""
    parts = [
        f"You are a professional subtitle translator. Translate each subtitle cue into the target language: {to}.",
        "Translate every cue in light of the full context: use the surrounding cues to disambiguate "
        "meaning, read the scene and situation, and infer who is speaking to whom. Keep proper nouns "
        "and terminology rendered consistently across the whole episode.",
        "Match the tone and register of the original rather than translating word for word -- "
        "casual vs. formal, calm vs. urgent, playful vs. serious, intimate vs. distant. Render each "
        "line the way a native speaker would actually say it in that situation and emotional moment.",
        "Translate sense-for-sense, not word-for-word: short idiomatic lines shift meaning with "
        "the scene (e.g. 'This is it' can mean the end has come, here we are, or this is the one) "
        "-- always pick the rendering that matches what is happening in that exact moment.",
        "Express the full meaning of each line: do not flatten emotional color or drop nuance "
        "for brevity. The words on screen must carry what the line actually means in context.",
        "Preserve the cue count: never merge, split, add, or drop cues -- each source cue maps to "
        "exactly one translated cue.",
        "Keep it colloquial and concise, fit for on-screen subtitles.",
        "Only condense when the translation would be too long to read comfortably as a subtitle, "
        "and even then drop only redundancies, false starts, and hesitation fillers -- never the "
        "meaning, nuance, or tone. Do not over-omit.",
        "Numbers follow target-language subtitle convention: spell out small numbers (one through "
        "ten) in dialogue; use digits for times, dates, scores, measurements, and longer numbers. "
        "Do not convert units of measurement -- render the original units naturally.",
        "Avoid abbreviations and acronyms unless they are the standard spoken form in the target "
        "language.",
        "Do not output decorative punctuation such as quotation marks, brackets, ellipses, or dashes "
        "(subtitle style).",
        'Return JSON only, in the form: {"translations": [{"i": <source index>, "t": "<translation>"}, ...]}, '
        "where every input index maps to exactly one translation.",
    ]
    if context:
        parts.append(f"\nBackground / tone:\n{context}")
    gl = format_glossary(glossary)
    if gl:
        parts.append(
            f"\nRender the following terms / names with these fixed translations:\n{gl}"
        )
    if tail:
        tail_txt = "\n".join(f"{o} => {t}" for o, t in tail if t)
        if tail_txt:
            parts.append(
                "\nPreceding context (already translated, for stylistic continuity only -- "
                f"do not re-output):\n{tail_txt}"
            )
    user = json.dumps({"cues": payload}, ensure_ascii=False)
    return [
        {"role": "system", "content": "\n".join(parts)},
        {"role": "user", "content": user},
    ]


def _make_client(base_url: str | None, api_key: str | None):
    """Lazily import openai and construct a client; raises a clear error if the package is missing."""
    try:
        from openai import OpenAI
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "translate requires openai (not installed); install with: pip install 'voxweave[translate]' or make install"
        ) from e
    kw = {}
    if api_key:
        kw["api_key"] = api_key
    if base_url:
        kw["base_url"] = base_url
    return OpenAI(**kw)


def _call(client, model: str, messages: list[dict], on_entry=None) -> str:
    """Call the model and return the JSON text.

    With ``on_entry``: streams and counts ``"i"`` keys as they arrive (one per cue),
    calling ``on_entry(delta)`` to advance the progress bar. Counting ``"i"`` is reliable
    because index values are numbers and translated strings JSON-escape interior quotes to
    ``\\"`` — no bare ``"i"`` tokens leak through. Without ``on_entry``: single blocking call.
    """
    if on_entry is None:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    buf: list[str] = []
    seen = 0
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        stream=True,
    )
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue  # trailing usage-only chunks have no choices
        piece = getattr(chunk.choices[0].delta, "content", None)
        if not piece:
            continue
        buf.append(piece)
        n = "".join(buf).count('"i"')
        if n > seen:
            on_entry(n - seen)
            seen = n
    return "".join(buf)


def translate_cues(
    payload: list[dict],
    *,
    to: str,
    model: str = TRANSLATE_MODEL,
    context: str | None = None,
    glossary: dict[str, str] | str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    client=None,
    batch: int = BATCH_THRESHOLD,
    context_tail: int = CONTEXT_TAIL,
    reporter=None,
) -> dict[int, str]:
    """Numbered payload -> {index: translated text}.

    Single call when len <= batch (whole-episode context); sequential windows with continuity
    tail otherwise. ``client`` injectable for tests.
    """
    if not payload:
        return {}
    client = client or _make_client(base_url, api_key)
    if len(payload) <= batch:
        windows = [payload]
    else:
        windows = [payload[i : i + batch] for i in range(0, len(payload), batch)]
    # With streaming the bar advances as entries arrive; also works for a single batch.
    on_entry = None
    if reporter is not None:
        reporter.task(f"translate -> {to}", len(payload))
        on_entry = reporter.advance
    result: dict[int, str] = {}
    tail: list[tuple[str, str]] = []
    for win in windows:
        msgs = build_messages(win, to=to, context=context, glossary=glossary, tail=tail)
        result.update(parse_response(_call(client, model, msgs, on_entry=on_entry)))
        tail = [(c["t"], result.get(c["i"], "")) for c in win[-context_tail:]]
    return result
