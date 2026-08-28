"""The single language-to-engine-family selection datum.

P6 ships every supported language on the historical family.  The mapping is
immutable and deliberately has no environment, configuration, CLI, or stored
manifest input.  A later cutover is therefore one reviewed data edit after the
P7 qualification program, not a second selection mechanism.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from voxweave import lang

EngineFamily = Literal["legacy-v1", "boundary-v2"]

MANIFEST_ENGINE_BY_FAMILY = MappingProxyType(
    {
        "legacy-v1": "legacy-v1",
        "boundary-v2": "boundary-optimizer-v2",
    }
)

# Keep this literal complete and visibly reviewable.  Importing lang's private
# table here would turn a change in a helper module into an implicit registry
# edit, defeating the one-datum cutover rule.
LANGUAGE_ENGINE_FAMILY: Mapping[str, EngineFamily] = MappingProxyType(
    {
        "de": "legacy-v1",
        "en": "legacy-v1",
        "es": "legacy-v1",
        "fr": "legacy-v1",
        "it": "legacy-v1",
        "ja": "legacy-v1",
        "ko": "legacy-v1",
        "pt": "legacy-v1",
        "ru": "legacy-v1",
        "yue": "legacy-v1",
        "zh": "legacy-v1",
    }
)

REGISTRY_CANONICAL_BYTES = (
    json.dumps(
        dict(LANGUAGE_ENGINE_FAMILY),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    + "\n"
).encode("utf-8")
REGISTRY_SHA256 = hashlib.sha256(REGISTRY_CANONICAL_BYTES).hexdigest()


def canonical_registry_iso(raw: str | None) -> str | None:
    """Return a supported canonical ISO, or ``None`` without guessing."""
    return lang.to_iso_or(raw, None)


def engine_family_for(raw: str | None) -> EngineFamily:
    """Resolve one family; missing and unsupported labels stay on legacy."""
    iso = canonical_registry_iso(raw)
    if iso is None:
        return "legacy-v1"
    return LANGUAGE_ENGINE_FAMILY.get(iso, "legacy-v1")
