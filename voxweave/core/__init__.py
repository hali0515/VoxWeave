"""voxweave.core — subtitle segmentation, layout and display-timing core.

The modules fall into three groups with different contracts.

**v1 engine** -- the shipped segmentation path. Pure text/timing logic: no
audio, no ASR/aligner models and no imports from outside ``voxweave.core``; the
only external dependencies are the optional language providers (pysbd, BudouX,
jieba, fugashi) that ``providers`` records.

- ``layout`` — text primitives, per-language line budgets, display wrapping,
  punctuation stripping.
- ``smart_split`` — the public facade of the segmentation engine and the
  ``smart_split_segments`` orchestrator. It re-exports the names of the
  modules below, so every ``smart_split.<name>`` import keeps working.
- ``split_types`` — the per-call ``SplitThresholds`` and ``SplitContext``.
- ``split_atoms`` — unit surface text, display-character ranges and the break
  atoms built from them.
- ``split_sentences`` — stage 1: sentence splitting, comma clauses and clause
  location in the units.
- ``split_scoring`` — break-point quality: atom penalties, the pause and
  micro-fragment ladders, length-break choice.
- ``split_packing`` — stage 2: gap-aware atom packing into cues, the hard wrap
  and the spread of untimed text.
- ``split_repair`` — the zh/yue/ja bound-particle repair over packed cues.
- ``timing`` — timing-only polish over the final cue stream (glue/merge,
  duration cleanup, shot-change snapping).
- ``overlay`` — the overlays after the engine (lyric flags, shot re-snap,
  the lyric display wrap) and the persisted span/turn inputs they read.
- ``unit_repair`` — pre-segmentation repair of aligner artifacts in the unit
  stream.
- ``kinsoku`` / ``breakpoints`` / ``gap_split`` / ``langsets`` — leaf
  tables and scoring shared by the above.

**Records** -- what one segmentation ran on and with.

- ``schema`` — TypedDicts for the unit and cue dicts.
- ``segdoc`` — the immutable segmentation IR (``SegDocument``, ``SourceUnit``,
  ``DisplayProfile``).
- ``providers`` — language-provider identity and the degradation ledger.
"""
