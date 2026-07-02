from __future__ import annotations

import logging
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


def to_iso3(iso: str) -> str:
    """ISO-639-1 -> ISO-639-3 for uroman/ctc-forced-aligner. Unknown codes pass through unchanged."""
    code = (iso or "").lower()
    if code not in _ISO1_TO_ISO3:
        log.warning("unknown language %r; passing through unchanged", code)
    return _ISO1_TO_ISO3.get(code, code)


def display_name(raw: str) -> str:
    """Capitalized English language name for UI/track titles ("zh" -> "Chinese")."""
    return _ISO_TO_NAME[to_iso(raw)].capitalize()
