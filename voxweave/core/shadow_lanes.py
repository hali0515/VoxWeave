"""The four shadow lane names, defined once.

A leaf with no imports: the live hook (``shadow_v2``) and the schema-2
validator (``shadow_schema``) both read these, and the validator must stay
importable without pulling in the hook.
"""

from __future__ import annotations

#: Core partition, before any legacy overlay.
LANE_CORE = "core_partition_pre_overlay"
#: The legacy delivery proxy: the overlays applied to each engine's stream.
LANE_LEGACY = "delivery_v1_legacy"
#: The finalizer row matrix.
LANE_FINALIZER = "delivery_finalizer"
#: The legacy-display isolation comparator.
LANE_DISPLAY = "legacy_display"

__all__ = ["LANE_CORE", "LANE_DISPLAY", "LANE_FINALIZER", "LANE_LEGACY"]
