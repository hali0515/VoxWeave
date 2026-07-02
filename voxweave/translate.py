from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from voxweave import fsio
from voxweave.realign import render_cues

log = logging.getLogger("voxweave")

# Injectable for tests (backoff sleeps must not slow the suite down).
_sleep = time.sleep
# Two backoffs = three attempts per window; long enough to ride out rate-limit
# blips, short enough that a hard failure surfaces within seconds.
_RETRY_DELAYS = (2.0, 8.0)

TRANSLATE_MODEL = os.environ.get("VOXWEAVE_TRANSLATE_MODEL", "gpt-5.3-chat-latest")
# Set high (800) so a typical episode (300-500 cues) fits in one call — full-episode context
# beats windowing: seams cause disambiguation errors and inconsistent proper-noun rendering.
# Only very long compilations (>800 cues) fall back to sequential windows.
BATCH_THRESHOLD = int(os.environ.get("VOXWEAVE_TRANSLATE_BATCH", "800"))
# Tail cues from the previous window carried into the next for stylistic continuity.
CONTEXT_TAIL = int(os.environ.get("VOXWEAVE_TRANSLATE_CONTEXT_TAIL", "3"))
# Second windowing gate: total cue characters per window. The cue-count cap alone
# lets a dense compilation (800 long cues) build a prompt past the model's context
# window; CJK text runs ~1 token/char and the response echoes the input size, so
# 60k chars keeps request+response comfortably inside current context limits.
WINDOW_CHARS = int(os.environ.get("VOXWEAVE_TRANSLATE_WINDOW_CHARS", "60000"))


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


# Netflix dual-speaker cue: exactly two lines, each starting with a hyphen, one
# speaker per line (``-line A\n-line B``, hyphen without a following space --
# emitted by voxweave.diarize). Detected on the ASCII hyphen the diarizer writes.
_DASH_SOURCE_LINE_RE = re.compile(r"^-\s*\S")
# Any leading dash (ASCII/en/em, possibly repeated, with surrounding space) the
# model may prepend to a half despite instructions -- stripped so our code owns
# the hyphen formatting.
_LEADING_DASH_RE = re.compile(r"^\s*[-–—]+\s*")
# Stable, collision-free index namespace for the second half of a dash cue. Block
# indices are far below this (subtitle files hold well under a million cues), so a
# dash cue's halves take indices ``k`` and ``k + _DASH_UNIT_BASE`` regardless of
# whether translate_cues is called with the full payload or a retry subset -- keeps
# the progress sidecar consistent across both.
_DASH_UNIT_BASE = 1_000_000


def is_dash_cue(text: str) -> bool:
    """True when ``text`` is a two-line dual-speaker cue (each line a hyphen +
    content). Single lines that start with '-' and 3+ line texts are ordinary."""
    lines = text.split("\n")
    return len(lines) == 2 and all(_DASH_SOURCE_LINE_RE.match(ln) for ln in lines)


def _dash_parts(text: str) -> list[str]:
    """Two-line dash cue -> its two speaker halves with the leading hyphen removed."""
    return [_LEADING_DASH_RE.sub("", ln, count=1).strip() for ln in text.split("\n")]


def _clean_half(text: str) -> str:
    """Strip any model-prepended dash and flatten internal whitespace so a half is
    always a single line (dash cues recombine two halves on one ``\\n``)."""
    return " ".join(_LEADING_DASH_RE.sub("", text.strip(), count=1).split())


def build_payload(blocks: list[dict]) -> list[dict]:
    """Cue blocks -> numbered list [{i, t}]; multi-line cue text is joined into a
    single string. A dual-speaker dash cue (see :func:`is_dash_cue`) additionally
    carries ``parts`` = its two speaker halves, so translate_cues can translate
    each speaker as a separate unit without bleeding across them."""
    out: list[dict] = []
    for idx, b in enumerate(blocks):
        text = b["text"]
        entry = {"i": idx, "t": " ".join(text.split("\n")).strip()}
        if not b.get("lyric") and is_dash_cue(text):
            entry["parts"] = _dash_parts(text)
        out.append(entry)
    return out


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
            idx = int(it["i"])
            txt = str(it["t"])
        except (KeyError, TypeError, ValueError):
            log.warning("dropping malformed translation entry: %r", it)
            continue
        if idx in out:
            log.warning("duplicate index %d in translations; last one wins", idx)
        out[idx] = txt
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


def _layout_dash_cue(block: dict, trans_text: str | None) -> str:
    """Render a dual-speaker dash cue as exactly two lines ``-X\\n-Y``, one per
    speaker. Each half is punct-stripped but NEVER soft-wrapped: a dash cue must
    stay two lines (one per speaker) even when a half is over budget, and the two
    halves must never merge onto one line. The leading hyphen is re-applied here
    (our code owns it); a half missing its translation falls back to the source."""
    src = _dash_parts(block["text"])
    halves = trans_text.split("\n") if trans_text and trans_text.strip() else []
    out: list[str] = []
    for idx in range(2):
        raw = _clean_half(halves[idx]) if idx < len(halves) else ""
        if not raw:
            raw = src[idx] if idx < len(src) else ""
        out.append(f"-{strip_punct_for_subtitles(raw)}")
    return "\n".join(out)


def translated_rows(
    blocks: list[dict], trans: dict[int, str], to_iso: str | None = None
) -> list[tuple[float | None, float | None, str]]:
    """Translated text + per-block timestamps -> (start, end, text) rows;
    missing translations fall back to the original text. ``to_iso`` enables
    target-language soft-wrap (see _layout_translated). Lyric-flagged blocks
    get their music-note wrap restored after layout. Dual-speaker dash cues are
    rendered as two unwrapped speaker lines (see :func:`_layout_dash_cue`)."""

    def _text(i: int, b: dict) -> str:
        if not b.get("lyric") and is_dash_cue(b["text"]):
            return _layout_dash_cue(b, trans.get(i))
        t = _layout_translated(
            strip_punct_for_subtitles(trans.get(i, "").strip() or b["text"]), to_iso
        )
        return f"♪ {t} ♪" if b.get("lyric") else t

    return [(b.get("start"), b.get("end"), _text(i, b)) for i, b in enumerate(blocks)]


def render_translated_vtt(
    blocks: list[dict], trans: dict[int, str], to_iso: str | None = None
) -> str:
    """Translated blocks -> VTT (see :func:`translated_rows`); blocks without
    timestamps produce plain-text cues."""
    return render_cues(translated_rows(blocks, trans, to_iso))


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
    if not p.exists():
        raise RuntimeError(f"glossary file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"invalid JSON in glossary {p.name}: {e}") from e
        if not isinstance(data, dict):
            raise RuntimeError(f"glossary {p.name} must be a JSON object")
        return data
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
    if not to or not to.strip() or "\n" in to:
        raise ValueError("target language must be a non-empty single-line string")
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
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; pass api_key= or set the OPENAI_API_KEY environment variable"
        )
    if base_url:
        return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=key)


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


def _call_with_retry(client, model: str, messages: list[dict], on_entry=None) -> str:
    """:func:`_call` with exponential backoff on transient failures (network,
    rate limit, 5xx). A streamed retry re-counts entries already reported to
    ``on_entry`` — the bar may overshoot, the translations do not."""
    for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
        try:
            return _call(client, model, messages, on_entry=on_entry)
        except Exception as e:
            if delay is None:
                raise
            log.warning(
                "translate call failed (attempt %d/%d: %s); retrying in %.0fs",
                attempt + 1,
                len(_RETRY_DELAYS) + 1,
                e,
                delay,
            )
            _sleep(delay)
    raise AssertionError("unreachable")


def payload_signature(payload: list[dict]) -> str:
    """Stable fingerprint of a translate payload; guards progress files against
    being replayed onto a since-edited subtitle."""
    doc = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(doc.encode("utf-8")).hexdigest()


def save_progress(path: Path, sig: str | None, trans: dict[int, str]) -> None:
    """Persist completed translations (atomic) so an interrupted multi-window
    run can resume instead of re-translating from zero."""
    doc = {"sig": sig, "translations": {str(i): t for i, t in trans.items()}}
    fsio.atomic_write_text(Path(path), json.dumps(doc, ensure_ascii=False))


def load_progress(path: Path, sig: str | None) -> dict[int, str]:
    """Load a progress file written by :func:`save_progress`; empty dict when
    the file is missing, unreadable, or was written for a different payload."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(doc, dict) or doc.get("sig") != sig:
        log.warning("%s does not match the current subtitle; ignoring it", p.name)
        return {}
    items = doc.get("translations", {})
    if not isinstance(items, dict):
        return {}
    out: dict[int, str] = {}
    for k, v in items.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


def _plan_windows(
    payload: list[dict], *, batch: int, char_budget: int
) -> list[list[dict]]:
    """Split the payload into sequential windows capped by BOTH cue count and
    total cue characters. A single cue larger than the budget still gets its own
    window (translated, never dropped). One window when everything fits."""
    windows: list[list[dict]] = []
    cur: list[dict] = []
    cur_chars = 0
    for c in payload:
        clen = len(c["t"])
        if cur and (len(cur) >= batch or cur_chars + clen > char_budget):
            windows.append(cur)
            cur, cur_chars = [], 0
        cur.append(c)
        cur_chars += clen
    if cur:
        windows.append(cur)
    return windows


def _expand_to_units(payload: list[dict]) -> list[dict]:
    """Block payload -> per-speaker translation units. A dash cue (carrying
    ``parts``) becomes two adjacent units with indices ``k`` and
    ``k + _DASH_UNIT_BASE`` so each speaker translates independently; every other
    cue passes through unchanged (index == block index, so ordinary cues behave
    exactly as before)."""
    units: list[dict] = []
    for c in payload:
        parts = c.get("parts")
        if parts:
            units.append({"i": c["i"], "t": parts[0]})
            units.append({"i": c["i"] + _DASH_UNIT_BASE, "t": parts[1]})
        else:
            units.append({"i": c["i"], "t": c["t"]})
    return units


def _collapse_units(payload: list[dict], unit_trans: dict[int, str]) -> dict[int, str]:
    """Per-unit translations -> per-block translations (external cue count). A dash
    cue recombines its two halves as ``X\\n-less`` joined by ``\\n`` only when BOTH
    are present -- a partial cue is left absent so the pipeline retries the whole
    cue instead of emitting a blank speaker line. Ordinary cues pass through."""
    out: dict[int, str] = {}
    for c in payload:
        k = c["i"]
        if c.get("parts"):
            a = _clean_half(unit_trans.get(k, ""))
            b = _clean_half(unit_trans.get(k + _DASH_UNIT_BASE, ""))
            if a and b:
                out[k] = f"{a}\n{b}"
        elif k in unit_trans:
            out[k] = unit_trans[k]
    return out


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
    char_budget: int = WINDOW_CHARS,
    context_tail: int = CONTEXT_TAIL,
    tail: list[tuple[str, str]] | None = None,
    reporter=None,
    progress_path: Path | None = None,
    progress_sig: str | None = None,
) -> dict[int, str]:
    """Numbered payload -> {index: translated text}.

    Single call when the whole payload fits both the cue-count cap and the char
    budget (whole-episode context); sequential windows with continuity tail
    otherwise. ``client`` injectable for tests. ``tail`` seeds the first window's
    continuity context (e.g. an already-translated preceding cue on retry), the
    same channel inter-window tails already use. With ``progress_path``, completed
    windows are persisted after each call and fully-covered windows are skipped
    on a rerun, so a mid-run failure costs only the window that failed.
    """
    if not payload:
        return {}
    client = client or _make_client(base_url, api_key)
    # Dual-speaker dash cues expand into two per-speaker units before windowing so
    # each speaker translates in isolation; the result is collapsed back to one
    # entry per block afterwards, keeping the external cue count conserved.
    units = _expand_to_units(payload)
    windows = _plan_windows(units, batch=batch, char_budget=char_budget)
    # With streaming the bar advances as entries arrive; also works for a single batch.
    on_entry = None
    if reporter is not None:
        reporter.task(f"translate -> {to}", len(units))
        on_entry = reporter.advance
    result: dict[int, str] = {}
    if progress_path is not None:
        result.update(load_progress(progress_path, progress_sig))
    tail = list(tail) if tail else []
    for win in windows:
        if all(result.get(c["i"], "").strip() for c in win):
            if on_entry is not None:
                on_entry(len(win))  # resumed from progress: count it as done
        else:
            msgs = build_messages(
                win, to=to, context=context, glossary=glossary, tail=tail
            )
            parsed = parse_response(
                _call_with_retry(client, model, msgs, on_entry=on_entry)
            )
            win_ids = {c["i"] for c in win}
            stray = [i for i in parsed if i not in win_ids]
            if stray:
                log.warning("dropping out-of-window indices %s from response", stray)
                for i in stray:
                    del parsed[i]
            result.update(parsed)
            if progress_path is not None:
                save_progress(progress_path, progress_sig, result)
        tail = [(c["t"], result.get(c["i"], "")) for c in win[-context_tail:]]
    return _collapse_units(payload, result)
