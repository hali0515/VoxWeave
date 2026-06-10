"""voxweave.core — subtitle segmentation and layout core (pure logic, no models).

- ``layout`` — text primitives, per-language line budgets, display wrapping,
  punctuation stripping.
- ``timing`` — timing-only polish over the final cue stream (glue/merge,
  duration cleanup, shot-change snapping).
- ``smart_split`` — the segmentation engine (sentence/clause splitting,
  gap-aware atom packing) and the ``smart_split_segments`` orchestrator.
- ``kinsoku`` / ``breakpoints`` / ``conjunctions`` / ``gap_split`` /
  ``langsets`` — leaf tables and scoring shared by the above.
"""
