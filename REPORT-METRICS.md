# Segmentation metric-definition delta: v2 to v3

## Provenance

This is a definition-isolated comparison over the same 20 tracked case files.
No production voxweave module and no case JSON changed between the two runs.

- v2 artifact: build/calibration/segmentation-report-definition-v2.json
- v2 commit: 806c7941ec3673e93d478468772fabca78302b3c
- v3 artifact: build/calibration/segmentation-report-definition-v3.json
- v3 implementation commit: 83233811692eb1a1b1ab0b859cb77236ad1f4a34
- metric definition: 2 -> 3
- metric-definition digest: 89bf5b31753e9d80b132f0df154ac341a12082e0e340eee37b68342264842425
- corpus digest: 6c4e05cec8a674b0... -> 23df693e438693bc...

The corpus digest changes only because the registry records metric definition 3.
The machine cross-check found no per-case change in len_break_mid_phrase_rate,
over_7s_rate, or cps_p90: OTHER_METRIC_DIFFS = [].

## What definition 3 measures

Definition 3 combines all requirements from TASK.md and TASK-ADDENDUM-1.md:

1. An eligible document-final cue contributes one forbidden-end row. It is
   excluded only for lyric/manual exceptions or sentence-terminal source
   punctuation; a clause comma is not a terminal.
2. Internal alternatives are enumerated from phrase_start_offsets on the
   pre-split source-unit stream, within the two adjacent cues' source range.
3. Japanese numerator and alternative legality use the same UniDic Level-2
   lens. The ruler locates the tail-bearing atom in the punctuated source phrase
   lattice and calls kinsoku.ja_pos_end_penalties on that atom.
4. The active lens is ja-unidic-level2, fugashi 1.5.2 with unidic-lite 1.0.8
   (UnidicFeatures26). A missing POS source or missing token-end offset falls
   back to ja-char-table-level1.
5. Lens identity and context are stored in metric_definition and included in
   metric_definition_digest. A baseline with a different lens is exit 2.
6. The metric still reports rate plus bad/eligible, but its gate compares the
   raw bad count: current_bad <= baseline_bad + 1.

## Language-level definition-isolated delta

| group | v2 bad/n | v2 rate | v3 bad/n | v3 rate | delta bad | delta n |
|---|---:|---:|---:|---:|---:|---:|
| all | 1/399 | 0.002506266 | 3/415 | 0.007228916 | +2 | +16 |
| en | 0/48 | 0.000000000 | 3/54 | 0.055555556 | +3 | +6 |
| ja | 1/141 | 0.007092199 | 0/145 | 0.000000000 | -1 | +4 |
| zh | 0/210 | 0.000000000 | 0/216 | 0.000000000 | 0 | +6 |

There are 18 eligible final tails: 6 per language. ja-03 and zh-07 are the two
sentence-terminal exclusions. The newly visible bad final tails are en-01
(to), en-04 (to), and en-05 (the). Japanese improves to 0 because あの is
POS-clean and the apparent ja-04 alternative is also POS-bad, so that boundary
is not denominator-eligible.

The task brief mentions an old English n of 77. The reproducible v2 artifact
and the tracked v2 baseline both contain en n=48. The evidence-backed movement
in this branch is therefore 48 -> 54.

## Per-case definition-isolated delta

The no-alt column shows source-lattice and POS-legality reclassification. Rates
are always accompanied by bad/n.

| case | lang | v2 bad/n (rate) | v3 bad/n (rate) | no-alt v2->v3 | final eligible | terminal excluded |
|---|---|---:|---:|---:|---:|---:|
| zh-01 | zh | 0/14 (0.000000000) | 0/15 (0.000000000) | 0->0 | 1 | 0 |
| zh-02 | zh | 0/25 (0.000000000) | 0/26 (0.000000000) | 0->0 | 1 | 0 |
| zh-03 | zh | 0/34 (0.000000000) | 0/35 (0.000000000) | 0->0 | 1 | 0 |
| zh-04 | zh | 0/33 (0.000000000) | 0/34 (0.000000000) | 1->1 | 1 | 0 |
| zh-05 | zh | 0/39 (0.000000000) | 0/40 (0.000000000) | 1->1 | 1 | 0 |
| zh-06 | zh | 0/28 (0.000000000) | 0/29 (0.000000000) | 0->0 | 1 | 0 |
| zh-07 | zh | 0/37 (0.000000000) | 0/37 (0.000000000) | 1->1 | 0 | 1 |
| ja-01 | ja | 0/17 (0.000000000) | 0/18 (0.000000000) | 4->4 | 1 | 0 |
| ja-02 | ja | 1/11 (0.090909091) | 0/12 (0.000000000) | 10->10 | 1 | 0 |
| ja-03 | ja | 0/18 (0.000000000) | 0/17 (0.000000000) | 3->4 | 0 | 1 |
| ja-04 | ja | 0/20 (0.000000000) | 0/19 (0.000000000) | 11->13 | 1 | 0 |
| ja-05 | ja | 0/25 (0.000000000) | 0/27 (0.000000000) | 18->17 | 1 | 0 |
| ja-06 | ja | 0/27 (0.000000000) | 0/27 (0.000000000) | 5->6 | 1 | 0 |
| ja-07 | ja | 0/23 (0.000000000) | 0/25 (0.000000000) | 13->12 | 1 | 0 |
| en-01 | en | 0/6 (0.000000000) | 1/7 (0.142857143) | 0->0 | 1 | 0 |
| en-02 | en | 0/11 (0.000000000) | 0/12 (0.000000000) | 0->0 | 1 | 0 |
| en-03 | en | 0/10 (0.000000000) | 0/11 (0.000000000) | 0->0 | 1 | 0 |
| en-04 | en | 0/8 (0.000000000) | 1/9 (0.111111111) | 0->0 | 1 | 0 |
| en-05 | en | 0/3 (0.000000000) | 1/4 (0.250000000) | 0->0 | 1 | 0 |
| en-06 | en | 0/10 (0.000000000) | 0/11 (0.000000000) | 0->0 | 1 | 0 |

## Re-recorded tracked baseline diff

The old tracked baseline predates the live v2 replay, so this table includes
pre-existing replay drift as well as the metric-definition change.

| group | cues old->new | mid-phrase old->new | over-max old->new | cps p90 old->new | forbidden-end old->new |
|---|---:|---:|---:|---:|---:|
| all | 2082->2081 | 3/1778->1/1778 | 0/2082->0/2081 | 16.5143->16.5143 | 1/405->3/415 |
| en | 565->565 | 0/475->0/475 | 0/565->0/565 | 20.7122->20.7122 | 0/48->3/54 |
| ja | 660->659 | 3/464->1/464 | 0/660->0/659 | 7.1429->7.1429 | 1/147->0/145 |
| zh | 857->857 | 0/839->0/839 | 0/857->0/857 | 9.6154->9.6154 | 0/210->0/216 |

Other baseline changes:

- metric_definition_version is 3 and the complete lens block is recorded.
- corpus digest is 23df693e438693bc5192acd1f605b394d93a2bd5ada39e63ba38a3fef6fae9c6.
- generated_from_commit is 83233811692eb1a1b1ab0b859cb77236ad1f4a34.
- diagnostics add final_tail_eligible=18 and terminal_final_tails=2.
- aggregate no_legal_alternative changes 70 -> 69.
- the first three configured gate modes remain blocking.
- forbidden_end_rate remains configured warning with min_samples=100, but its
  comparison changes from a rate ceiling/tolerance to bad-count slack 1.

## Promotion status

### Production evaluation

| group | baseline bad/n | current bad/n | configured | effective | promoted | result |
|---|---:|---:|---|---|---|---|
| en | 3/54 | 3/54 | warning | warning | no | insufficient samples |
| ja | 0/145 | 0/145 | warning | blocking | yes | pass |
| zh | 0/216 | 0/216 | warning | blocking | yes | pass |

The configured gate table is unchanged by promotion. Every forbidden-end gate
result contains numerator, denominator, reported_rate, compared bad_count, and
baseline sample count.

## Shadow v1 versus v2 under definition 3

### Core partition, before overlays

| group | v1 bad/n (rate) | v2 bad/n (rate) |
|---|---:|---:|
| all | 3/403 (0.0074) | 3/343 (0.0087) |
| en | 3/54 (0.0556) | 3/74 (0.0405) |
| ja | 0/135 (0.0000) | 0/93 (0.0000) |
| zh | 0/214 (0.0000) | 0/176 (0.0000) |

### Delivery proxy, after overlays (gated lane)

| group | v1 bad/n (rate) | v2 bad/n (rate) |
|---|---:|---:|
| all | 3/415 (0.0072) | 3/365 (0.0082) |
| en | 3/54 (0.0556) | 3/74 (0.0405) |
| ja | 0/145 (0.0000) | 0/113 (0.0000) |
| zh | 0/216 (0.0000) | 0/178 (0.0000) |

The addendum's expected Japanese result is confirmed: all four ja
lane/engine combinations have bad=0. Their denominators differ because v1/v2
choose different partitions; none is hidden behind a bare rate.

| group | baseline bad/n | v2 delivery bad/n | configured | effective | promoted | result |
|---|---:|---:|---|---|---|---|
| en | 3/54 | 3/74 | warning | warning | no | insufficient samples |
| ja | 0/145 | 0/113 | warning | blocking | yes | pass |
| zh | 0/216 | 0/178 | warning | blocking | yes | pass |

The final shadow command exits 0 with failures=0, warnings=1, and projection
agreement on 20/20 cases.
