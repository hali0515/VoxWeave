from __future__ import annotations

import logging
import re
import unicodedata
from typing import overload

log = logging.getLogger("voxweave")

# 11 languages supported by Qwen3-ForcedAligner; ISO <-> Qwen full name (all lowercase)
_ISO_TO_NAME = {
    "yue": "cantonese",
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}
_NAME_TO_ISO = {name: iso for iso, name in _ISO_TO_NAME.items()}

# ISO-639-1 -> ISO-639-3 for uroman/ctc-forced-aligner.
# zh maps to "chi" not "zho": preprocess_text checks for "chi" to enable per-character mode.
_ISO1_TO_ISO3 = {
    "en": "eng",
    "zh": "chi",
    "ja": "jpn",
    "ko": "kor",
    "yue": "chi",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "ar": "ara",
    "hi": "hin",
    "nl": "nld",
    "tr": "tur",
    "vi": "vie",
    "th": "tha",
    "id": "ind",
    "uk": "ukr",
    "pl": "pol",
}


def _canon(raw: str | None) -> str:
    if not raw or not raw.strip():
        raise ValueError("language is required")
    # Drop BCP-47 region/script subtags: "en-US"/"zh_CN" -> "en"/"zh".
    return raw.strip().lower().replace("_", "-").split("-", 1)[0]


def is_supported(raw: str) -> bool:
    """Return True if the language is within the aligner's 11 supported languages (accepts ISO code or full name)."""
    try:
        key = _canon(raw)
    except ValueError:
        return False
    return key in _ISO_TO_NAME or key in _NAME_TO_ISO


def to_iso(raw: str) -> str:
    """Normalize full name (English) or ISO code (en) to ISO (en); used by smart_split."""
    key = _canon(raw)
    if key in _ISO_TO_NAME:
        return key
    if key in _NAME_TO_ISO:
        return _NAME_TO_ISO[key]
    raise ValueError(f"unsupported language {raw!r}")


def to_aligner_name(raw: str) -> str:
    """Normalize full name or ISO code to the Qwen full name (all lowercase); used by aligner/align."""
    key = _canon(raw)
    if key in _NAME_TO_ISO:
        return key
    if key in _ISO_TO_NAME:
        return _ISO_TO_NAME[key]
    raise ValueError(f"unsupported language {raw!r}")


@overload
def to_iso_or(raw: str | None, default: str) -> str: ...
@overload
def to_iso_or(raw: str | None, default: None) -> str | None: ...
def to_iso_or(raw: str | None, default: str | None) -> str | None:
    """Normalize to ISO when supported, else return default (None/empty/unknown -> default).

    Folds the open-coded `to_iso(x) if is_supported(x) else default` idiom into one place.
    """
    try:
        key = _canon(raw)
    except ValueError:
        return default
    if key in _ISO_TO_NAME:
        return key
    if key in _NAME_TO_ISO:
        return _NAME_TO_ISO[key]
    return default


_LANGUAGE_LABEL_SEP_RE = re.compile(r"[,，;；|/]+")
_LATIN_LANGUAGE_ISOS = {"en", "fr", "de", "it", "pt", "es"}
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def detected_language_candidates(raw: str | None) -> list[str]:
    """Return ordered, de-duplicated labels from an ASR language field.

    Qwen can report multilingual audio as a comma-separated string such as
    ``"English,Chinese"``.  Treating that value as one label (or blindly taking
    its first item) makes the result depend on model formatting rather than the
    transcript.  Full-width punctuation is accepted because localized wrappers
    occasionally preserve it.
    """
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in _LANGUAGE_LABEL_SEP_RE.split(str(raw)):
        label = part.strip().strip("\"'").strip()
        if not label:
            continue
        key = to_iso_or(label, None) or label.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def transcript_content_weight(text: str) -> int:
    """Stable language-vote mass derived from transcript content, not alignment.

    Alignment unit counts are an unsafe vote: selecting the wrong tokenizer can
    itself collapse or multiply units.  Unicode alphanumeric characters exist
    before alignment and therefore give every engine the same evidence.
    """
    return sum(ch.isalnum() for ch in text)


def _is_han(codepoint: int) -> bool:
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x323AF
    )


def _dominant_transcript_script(text: str) -> str | None:
    """Return only strong script evidence; short/mixed snippets stay undecided."""
    han = kana = hangul = latin = other = 0
    for ch in text:
        cp = ord(ch)
        if _is_han(cp):
            han += 1
        elif 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9D:
            kana += 1
        elif (
            0x1100 <= cp <= 0x11FF
            or 0x3130 <= cp <= 0x318F
            or 0xA960 <= cp <= 0xA97F
            or 0xAC00 <= cp <= 0xD7AF
            or 0xD7B0 <= cp <= 0xD7FF
        ):
            hangul += 1
        elif ch.isalpha() and "LATIN" in unicodedata.name(ch, ""):
            latin += 1
        elif ch.isalpha():
            other += 1

    total = han + kana + hangul + latin + other
    if not total:
        return None
    if hangul >= 4 and hangul * 5 >= total * 3:
        return "ko"

    # Japanese normally mixes kana and Han.  Requiring a meaningful kana share
    # avoids classifying Chinese prose containing a short Japanese product name.
    cjk = han + kana
    if kana >= 2 and cjk >= 4 and cjk * 5 >= total * 3 and kana * 6 >= cjk:
        return "ja"
    if han >= 4 and han * 5 >= total * 3 and kana * 6 < cjk:
        return "han"

    # Latin cannot distinguish English from the other supported Latin-script
    # languages.  It is useful only for selecting one of the model's candidates,
    # and only for sentence-like text rather than a couple of product names.
    if (
        latin >= 12
        and len(_LATIN_WORD_RE.findall(text)) >= 4
        and latin * 5 >= total * 4
    ):
        return "latin"
    return None


def reconcile_detected_language(
    detected: str | None, text: str, override: str | None = None
) -> str | None:
    """Choose the effective ASR language using labels plus transcript script.

    An explicit CLI language is authoritative.  In auto mode, strong script
    evidence selects the matching item from a multilingual Qwen label.  It can
    also repair an impossible single-label result (most importantly a Han-heavy
    transcript reported as English).  Weak evidence deliberately keeps the first
    model label so short mixed/proper-noun chunks cannot flip the file language.
    """
    if override and override.strip():
        return override.strip()

    candidates = detected_language_candidates(detected)
    primary = candidates[0] if candidates else None
    pairs = [(label, to_iso_or(label, None)) for label in candidates]
    script = _dominant_transcript_script(text)

    if script == "han":
        # Han alone cannot distinguish Chinese, Cantonese, and Japanese.
        # Prefer a model-supplied CJK candidate (especially when it is the only
        # one) instead of manufacturing Chinese from an inherently ambiguous
        # script.  Candidate order remains the model's tie-break for multiple
        # CJK labels.
        for label, iso in pairs:
            if iso in {"zh", "yue", "ja"}:
                return label
        return "zh"
    if script in {"ja", "ko"}:
        for label, iso in pairs:
            if iso == script:
                return label
        return script
    if script == "latin":
        for label, iso in pairs:
            if iso in _LATIN_LANGUAGE_ISOS:
                return label
    return primary


def to_iso3(iso: str) -> str:
    """ISO-639-1 -> ISO-639-3 for uroman/ctc-forced-aligner. Unknown codes pass through unchanged."""
    code = (iso or "").lower()
    if code not in _ISO1_TO_ISO3:
        log.warning("unknown language %r; passing through unchanged", code)
    return _ISO1_TO_ISO3.get(code, code)


def display_name(raw: str) -> str:
    """Capitalized English language name for UI/track titles ("zh" -> "Chinese")."""
    return _ISO_TO_NAME[to_iso(raw)].capitalize()
