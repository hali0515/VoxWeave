"""voxweave.core — subtitle segmentation, layout and display-timing core.

The modules fall into three groups with different contracts.

**v1 engine** -- the shipped segmentation path. Pure text/timing logic: no
audio, no ASR/aligner models and no imports from outside ``voxweave.core``; the
only external dependencies are the optional language providers (pysbd, BudouX,
jieba, fugashi) that ``providers`` records.

- ``layout`` — text primitives, per-language line budgets, display wrapping,
  punctuation stripping.
- ``smart_split`` — the segmentation engine (sentence/clause splitting,
  gap-aware atom packing) and the ``smart_split_segments`` orchestrator.
- ``timing`` — timing-only polish over the final cue stream (glue/merge,
  duration cleanup, shot-change snapping).
- ``unit_repair`` — pre-segmentation repair of aligner artifacts in the unit
  stream.
- ``kinsoku`` / ``breakpoints`` / ``conjunctions`` / ``gap_split`` /
  ``langsets`` — leaf tables and scoring shared by the above.

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
``voxweave.pipeline`` and ``voxweave.diarize``.

- ``boundary_lattice`` / ``boundary_cost`` / ``boundary_v2`` — the hard-legal
  lattice, the cost model and the exact whole-interval solver.
- ``subunit`` / ``speaker_evidence`` / ``timing_preview`` / ``canonical_text``
  / ``policy_delta`` — solver inputs: sub-unit refinement, speaker evidence,
  display-duration preview, delivered text, the policy delta registry.
- ``finalizer`` / ``trace_validator`` / ``partition_check`` / ``authority`` —
  the TimelineFinalizer, its independent checks and sealed issuance.
- ``shadow_v2`` / ``shadow_schema`` — the shadow lane and its artifact contract.
- ``align_seed`` / ``align_compare`` — P6 align seed construction and semantic
  comparison.
"""
