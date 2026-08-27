# Ruler blind-spots implementation report

## Outcome

The segmentation ruler now closes both original blind spots and incorporates
the addendum without a second metric-version bump or a second baseline commit.

- Document-final non-terminal tails are measured.
- Internal alternatives come from the pre-split source phrase lattice.
- Japanese numerator and legality use the same UniDic Level-2 source-atom lens.
- Lens identity is part of a machine-checked metric-definition digest.
- forbidden_end_rate reports bad/eligible and rate, but gates the bad count
  against baseline_bad + 1.
- Warning gates promote per language only when both baseline and current
  denominators reach min_samples.
- Metric definition 3 is recorded once in the tracked baseline.
- Production evaluate and the v1/v2 shadow both pass with exit 0.

No production file under voxweave/ was changed and nothing was pushed.

Implementation commits before this report:

- 7c529b7 fix: close segmentation ruler blind spots
- 8323381 fix: align forbidden-tail ruler with source POS
- 6a2161a chore: rerecord segmentation quality baseline
- 6161637 test: cover metric definition baseline schema

## Implementation

### Final-cue eligibility

After lyric and manual-exception filtering, the final cue contributes one row
unless the final source tail has sentence-terminal punctuation. There is no
following gap or alternative boundary to test. Clause punctuation is not a
document terminator. Diagnostics expose final_tail_eligible and
terminal_final_tails.

The corpus contributes 18 eligible final tails and 2 exclusions:

- en: 6 eligible, 0 terminal
- ja: 6 eligible, 1 terminal
- zh: 6 eligible, 1 terminal

### Pre-split alternative lattice

Each mapped boundary carries the source-unit range owned by the adjacent cues.
has_legal_alternative enumerates only global phrase-start offsets within that
range, excluding the actual cut. A candidate is legal only if:

- both rendered halves are non-empty;
- both fit captured max_line_length and max_lines;
- the candidate left tail is below forbidden penalty 2 under the active lens.

This removes post-split re-segmentation circularity while keeping feasibility
checks intact.

### Japanese Level-2 lens

For Japanese, the ruler derives the punctuated source span, locates the atom
that owns the actual tail offset in the source phrase lattice, and calls
kinsoku.ja_pos_end_penalties on that atom. Numerator and alternative legality
share the same cache and scorer. Punctuation-only source units omitted from
word_data remain in POS context, while the lookup stays on the lexical tail.

Recorded identity:

- id: ja-unidic-level2
- provider: fugashi-unidic 1.5.2
- dictionary: unidic-lite 1.0.8 (UnidicFeatures26)
- context: punctuated-source-phrase-atom
- missing-source/offset fallback: ja-char-table-level1
- metric-definition digest:
  89bf5b31753e9d80b132f0df154ac341a12082e0e340eee37b68342264842425

load_baseline and record-baseline compare the complete live definition and its
digest. A Level-2 baseline cannot be used by a Level-1 fallback run.

### Count gate and promotion

forbidden_end_rate remains the report field name and remains configured
warning with min_samples=100. Its compared value is now bad_count:

current_bad <= baseline_bad + 1

The old absolute rate ceiling is removed; absolute_tolerance is 1 event and
relative_tolerance is 0. Each gate result includes numerator, denominator,
reported_rate, bad_count, and baseline_samples. The same policy is frozen in
SHADOW_GATES.

Effective production status:

| group | baseline/current bad/n | configured | effective | result |
|---|---:|---|---|---|
| en | 3/54 -> 3/54 | warning | warning | insufficient samples |
| ja | 0/145 -> 0/145 | warning | blocking | pass |
| zh | 0/216 -> 0/216 | warning | blocking | pass |

Promotion never rewrites the tracked configured mode.

## Metric and shadow results

Definition-isolated v2 -> v3:

| group | v2 bad/n | v3 bad/n |
|---|---:|---:|
| all | 1/399 | 3/415 |
| en | 0/48 | 3/54 |
| ja | 1/141 | 0/145 |
| zh | 0/210 | 0/216 |

The task's old English n=77 is not present in this branch. Both the captured v2
report and tracked v2 baseline use n=48; the reproducible movement is 48 -> 54.
REPORT-METRICS.md contains every case delta and the full baseline diff.

The addendum's Japanese shadow prediction is confirmed:

| lane | v1 bad/n | v2 bad/n |
|---|---:|---:|
| core_partition_pre_overlay | 0/135 | 0/93 |
| delivery_proxy_post_overlay | 0/145 | 0/113 |

The final shadow gated lane also has en 3/74 and zh 0/178. ja and zh promote to
blocking and pass; en remains warning because its baseline n=54 is below 100.
Projection cross-check agrees on all 20 cases.

## Baseline record

The baseline is isolated in commit 6a2161a. It records:

- metric definition version 3;
- corpus digest 23df693e438693bc5192acd1f605b394d93a2bd5ada39e63ba38a3fef6fae9c6;
- source implementation commit 83233811692eb1a1b1ab0b859cb77236ad1f4a34;
- the complete Japanese lens identity and metric-definition digest;
- the original configured modes: three blocking gates and one warning gate;
- forbidden-end count slack 1 with no absolute rate ceiling.

The baseline diff is broader than the isolated definition delta because the
old tracked baseline predates the live v2 replay. In particular, 2082 -> 2081
cues and 3/1778 -> 1/1778 mid-phrase were already present in the live v2 run.

## Assertions changed

Five old denominator assertions were deliberately updated for final-tail
eligibility:

- test_forbidden_end_counts_a_dangling_article_when_an_alternative_existed:
  1/1 -> 1/2.
- test_forbidden_end_denominator_drops_boundaries_with_no_alternative:
  0/0 null -> 0/1.
- test_forbidden_end_drops_boundaries_forced_by_a_long_pause:
  0/0 -> 0/1.
- test_forbidden_end_drops_boundaries_the_source_punctuated:
  0/0 -> 0/1.
- test_unavoidable_forbidden_end_exception_excludes_only_that_metric:
  0/0 -> 0/1.

Contract fixtures were also updated:

- synthetic corpus and schema fixtures now use metric definition 3;
- test_baseline_ratio_and_metric_shapes_validate includes the required
  metric_definition and metric_definition_digest blocks;
- report assertions require explicit forbidden numerator/denominator/rate;
- shadow assertions pin count slack, min_samples=100, and definition digest.

New tests cover:

- bad and terminal-excluded document-final tails;
- pre-split source-lattice alternatives;
- promotion at and below the two-sided sample threshold;
- Level-2 あの versus Level-1 fallback;
- Level-2 legality for the 連体詞 いわゆる;
- punctuated source-atom selection and punctuation-only context;
- count slack pass/fail independent of denominator dilution;
- rejection of a baseline recorded under another Japanese lens.

No unrelated output assertion was relaxed.

## Verification gates

### Full pytest

Command:

    uv run --extra cuda pytest /mnt/Dev/Git/qsub-worktrees/fix10-ruler/tests/ -q

Result:

    2302 passed in 9.23s

### Ruff

Commands:

    uv run --no-project --with ruff ruff check .
    uv run --no-project --with ruff ruff format --check .

Results:

    All checks passed!
    141 files already formatted

### Pyright

Command:

    uv run --extra cuda pyright

Result:

    0 errors, 0 warnings, 0 informations

### Production quality gate

Command:

    uv run --extra cuda python /mnt/Dev/Git/qsub-worktrees/fix10-ruler/scripts/calib_segmentation.py evaluate --corpus /mnt/Dev/Git/qsub-worktrees/fix10-ruler/calibration/segmentation/corpus.json --baseline /mnt/Dev/Git/qsub-worktrees/fix10-ruler/calibration/segmentation/baseline.json --json-out /mnt/Dev/Git/qsub-worktrees/fix10-ruler/build/calibration/segmentation-report.json --check

Relevant tail:

    [n<min] en forbidden_end_rate warning bad_count=3 rate=0.0556 (3/54)
    [ok] ja forbidden_end_rate blocking bad_count=0 rate=0.0000 (0/145)
    [ok] zh forbidden_end_rate blocking bad_count=0 rate=0.0000 (0/216)
    QUALITY segmentation status=pass cases=20 failures=0 warnings=1

### Shadow quality gate

Command:

    uv run --extra cuda python /mnt/Dev/Git/qsub-worktrees/fix10-ruler/scripts/calib_segmentation.py shadow --corpus /mnt/Dev/Git/qsub-worktrees/fix10-ruler/calibration/segmentation/corpus.json --baseline /mnt/Dev/Git/qsub-worktrees/fix10-ruler/calibration/segmentation/baseline.json --json-out /mnt/Dev/Git/qsub-worktrees/fix10-ruler/build/calibration/segmentation-shadow-report.json --check

Relevant tail:

    [n<min] en forbidden_end_rate warning bad_count=3 rate=0.0405 (3/74)
    [ok] ja forbidden_end_rate blocking bad_count=0 rate=0.0000 (0/113)
    [ok] zh forbidden_end_rate blocking bad_count=0 rate=0.0000 (0/178)
    projection cross-check agrees on 20/20 cases
    QUALITY segmentation-shadow status=pass cases=20 failures=0 warnings=1

## Rebase notes

- Keep the metric-version bump, corpus registry, both segmentation schemas, and
  baseline together.
- Preserve the source-lattice span fields on Boundary and the shared
  Japanese-tail scorer used by numerator and legality.
- Preserve SHADOW_GATES as three blocking literals plus forbidden warning,
  min_samples=100, count slack 1.
- Keep baseline commit 6a2161a as the standalone chore commit.
- TASK.md and TASK-ADDENDUM-1.md are task inputs and are intentionally not
  included in commits.
