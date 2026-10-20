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

**P5/P6 subsystem** -- the boundary optimizer, the TimelineFinalizer and the
align machinery, run as a shadow lane beside v1 and behind the ``boundary-v2``
delivery family. The "pure logic" claim above does not extend to this group,
which imports upward: ``align_seed`` imports ``voxweave.align_acquisition``,
``align_distribution``, ``align_failures`` and ``realign``; ``align_compare``
imports ``voxweave.align_delta_registry``; ``finalizer`` lazily imports
``voxweave.align_acquisition``; ``shadow_v2`` lazily imports
``voxweave.diarize``.

- ``boundary_lattice`` / ``boundary_atoms`` / ``boundary_nodes`` /
  ``boundary_packing`` / ``boundary_cost`` / ``boundary_v2`` — the hard-legal
  lattice, its data model, its node sets, its packing measure, the cost model
  and the exact whole-interval solver.
- ``boundary_select`` — path choice over one interval's cost table: the
  forward DP, path scoring and the margin policy.
- ``boundary_interval`` — one interval's committed answer: select a path or
  adopt v1's cues, then materialize the cues.
- ``subunit`` / ``speaker_evidence`` / ``timing_preview`` / ``canonical_text``
  / ``policy_delta`` — solver inputs: sub-unit refinement, speaker evidence,
  display-duration preview, delivered text, the policy delta registry.
- ``finalizer`` / ``trace_validator`` / ``partition_check`` / ``authority`` —
  the TimelineFinalizer, its independent checks and sealed issuance.
- ``shadow_v2`` / ``shadow_schema`` / ``shadow_lanes`` — the shadow lane, its
  artifact contract and the lane names both share.
- ``shadow_oracles`` / ``shadow_rows`` — the lane's independent diff and preview
  oracles, and the row builders for cues, streams and finalizer rows.
- ``shadow_schema_fields`` / ``shadow_schema_finalizer`` — the schema's field
  readers, and the finalizer row decode and replay.
- ``align_seed`` / ``align_compare`` — P6 align seed construction and semantic
  comparison.
"""
