# Dead selector cleanup report

## Status

Completed on branch `chore/remove-semantic-residue` after merging latest
`main` at `ee92eb3e74a1fa9b1517455bdb08ef4aea747d41`. The closed failure
registry remains byte-identical to its pinned baseline, the two retired
coverage rows are now honest structural reserves under controller authority,
and every required gate passes.

The initial cleanup changed `voxweave/align_failures.py`. Its SHA-256 became
`f167624754a9f8f12a4dc656ac50081633986c5c10952dff80c2becd0443280f`;
the P6 manifest pins
`69471bfcc890ef4d59b2f8cc36e37c48a5e971202527f13bf4e1b9b9cccdc9cd`.
The required oracle command therefore stopped at manifest validation.

## Controller ruling

The controller ruled that `voxweave/align_failures.py` is a closed historical
law vocabulary: retired error-code rows remain for compatibility and replay.
The file was restored byte-for-byte from baseline `2b5cb52`; its restored
SHA-256 is the pinned
`69471bfcc890ef4d59b2f8cc36e37c48a5e971202527f13bf4e1b9b9cccdc9cd`.
The retained `semantic-backend-unavailable` and
`semantic-selector-unmodelled` rows are string constants only. They import
no deleted module or symbol and require no restored execution path. The oracle
manifest was not re-pinned.

## Authorized coverage amendment

On 2026-08-31 the controller authorized changing exactly these
`failure_registry_coverage` entries:

- `semantic-backend-unavailable/endpoint-not-configured`
- `segmentation-v2-invalid/semantic-selector-unmodelled`

Each now has `status: structural-reserve`, an empty `evidence` list, and the
authorized rationale: the 2026-08-31 controller ruling keeps the row in the
closed historical failure vocabulary for compatibility and replay after the
retired selector's raising implementation was removed.

No other manifest row changed. No registry, schema, input, or execution digest
changed. The amendment is isolated in its own commit,
`chore: reclassify retired failure rows as structural-reserve`.

## Pre-edit inventory

I ran the required command before changing files:

```text
grep -rn semantic voxweave/ tests/ scripts/ README.md pyproject.toml
```

It returned 458 matches across 51 files. The complete occurrence
inventory is below as `path (match count): every matching baseline line`.
`README.md` and `pyproject.toml` had zero matches.

- `scripts/calib_alignment.py` (1): 1159
- `scripts/calib_segmentation.py` (5): 298, 411, 527, 4856, 4868
- `scripts/p6_oracle.py` (2): 680, 906
- `scripts/p6_oracle_release_refresh.py` (1): 203
- `tests/test_boundary_cost.py` (1): 243
- `tests/test_boundary_task.py` (8): 4, 5, 12, 26, 27, 28, 29, 33
- `tests/test_capture_units.py` (1): 218
- `tests/test_cli.py` (12): 138, 139, 147, 150, 158, 159, 160, 163, 300, 301, 306, 307
- `tests/test_config.py` (7): 64, 67, 68, 72, 73, 76, 77
- `tests/test_default_diarize_packaging.py` (2): 13, 153
- `tests/test_diarize.py` (2): 243, 245
- `tests/test_finalizer_fuzz.py` (1): 21
- `tests/test_finalizer_properties.py` (1): 129
- `tests/test_kinsoku.py` (1): 104
- `tests/test_p6_align_candidates.py` (5): 21, 143, 147, 149, 155
- `tests/test_p6_align_shadow_corpus.py` (9): 68, 446, 447, 448, 454, 475, 476, 519, 541
- `tests/test_p6_delta_registry.py` (3): 46, 47, 49
- `tests/test_p6_episode_transactions.py` (11): 770, 777, 778, 779, 784, 786, 789, 794, 797, 1149, 1150
- `tests/test_p6_evidence_core.py` (1): 67
- `tests/test_p6_failure_registry.py` (1): 11
- `tests/test_p6_fix_authority.py` (2): 223, 251
- `tests/test_p6_oracle_vectors.py` (2): 235, 247
- `tests/test_p6_registry_live_closure.py` (1): 114
- `tests/test_p6_segmentation_candidates.py` (7): 157, 166, 267, 279, 283, 381, 399
- `tests/test_p6_semantic_comparator.py` (16): 17, 121, 150, 156, 157, 159, 174, 176, 187, 189, 199, 200, 202, 212, 213, 215
- `tests/test_p6_snapshot.py` (3): 80, 81, 82
- `tests/test_pipeline.py` (2): 434, 441
- `tests/test_segment_document.py` (1): 283
- `tests/test_semantic_breaks.py` (7): 8, 16, 798, 801, 802, 803, 804
- `tests/test_semantic_removal.py` (19): 1, 11, 24, 25, 26, 27, 28, 36, 38, 39, 40, 43, 49, 50, 51, 52, 55, 56, 58
- `tests/test_smart_split_semantic.py` (19): 13, 82, 90, 91, 94, 95, 106, 143, 171, 207, 259, 284, 304, 348, 377, 398, 404, 411, 412
- `tests/test_timing_preview.py` (1): 32
- `voxweave/align_adapter.py` (29): 123, 134, 139, 243, 263, 286, 290, 400, 435, 436, 448, 519, 520, 540, 541, 557, 558, 600, 678, 701, 717, 729, 769, 775, 784, 789, 821, 855, 859
- `voxweave/align_delta_registry.py` (7): 17, 27, 35, 43, 51, 59, 67
- `voxweave/align_failures.py` (3): 22, 94, 189
- `voxweave/align_orchestration.py` (7): 27, 449, 472, 474, 479, 481, 487
- `voxweave/align_shadow.py` (31): 64, 219, 314, 317, 318, 319, 320, 321, 322, 330, 331, 332, 333, 334, 336, 347, 348, 371, 444, 458, 462, 466, 480, 483, 485, 489, 490, 491, 510, 541, 579
- `voxweave/align_snapshot.py` (17): 4, 232, 435, 442, 458, 468, 476, 478, 485, 493, 506, 513, 515, 518, 519, 532, 540
- `voxweave/asrfix.py` (2): 150, 193
- `voxweave/core/align_compare.py` (14): 1, 49, 68, 888, 907, 914, 918, 921, 923, 930, 935, 972, 1040, 1041
- `voxweave/core/boundary_lattice.py` (1): 137
- `voxweave/core/boundary_task.py` (7): 15, 16, 26, 43, 76, 110, 112
- `voxweave/core/kinsoku.py` (1): 47
- `voxweave/core/smart_split.py` (66): 75, 480, 1114, 1903, 1929, 1934, 1974, 1981, 2002, 2027, 2057, 2073, 2108, 2128, 2131, 2133, 2134, 2145, 2163, 2184, 2202, 2217, 2241, 2264, 2271, 2274, 2288, 2295, 2308, 2321, 2325, 2365, 2398, 2412, 2423, 2424, 2450, 2479, 2527, 2590, 2632, 2633, 2661, 2663, 2679, 2681, 2682, 2686, 2751, 2753, 2765, 2767, 2772, 2784, 2786, 2788, 2790, 2793, 2803, 2806, 2814, 2815, 2817, 2820, 2824, 2835
- `voxweave/p6_ratifications.py` (3): 63, 64, 65
- `voxweave/pipeline.py` (53): 302, 306, 307, 359, 367, 3582, 3583, 3616, 3738, 3739, 3786, 3806, 3807, 3811, 3813, 3818, 4095, 4096, 4109, 4116, 4118, 4120, 4132, 4136, 4164, 4165, 4236, 4237, 4268, 4269, 4273, 4326, 4330, 4331, 4348, 4349, 4373, 4374, 4392, 4402, 4403, 4406, 4451, 4463, 4595, 4596, 4603, 4654, 4656, 4666, 4667, 4672, 4703
- `voxweave/progress.py` (1): 16
- `voxweave/segmentation_adapter.py` (4): 500, 503, 514, 520
- `voxweave/segmentation_orchestration.py` (5): 79, 222, 239, 308, 340
- `voxweave/semantic_breaks.py` (50): 1, 31, 86, 88, 89, 91, 144, 178, 184, 241, 253, 295, 577, 600, 627, 661, 673, 677, 682, 688, 690, 704, 707, 708, 712, 909, 936, 943, 973, 976, 994, 1031, 1043, 1048, 1070, 1092, 1126, 1139, 1154, 1176, 1187, 1196, 1227, 1232, 1243, 1260, 1261, 1287, 1288, 1289
- `voxweave/ui.py` (2): 175, 180

### Inventory classification

Deleted execution residue:

- The endpoint implementation `voxweave/semantic_breaks.py` and its sole
  shared task contract `voxweave/core/boundary_task.py`.
- The optional engine branch, preparation graph, model arguments, imports, and
  constants in `voxweave/core/smart_split.py`.
- `semantic_split`, `semantic_model`, and `semantic_engine` parameters,
  factory/release ownership, error attachment, diagnostics, and internal
  threading in `voxweave/pipeline.py`.
- The retired selector switch in segmentation orchestration/adapter code, its
  execution branch, UI hint, positive tests, and stale test prose.
- Entire positive-only test modules
  `tests/test_semantic_breaks.py`,
  `tests/test_smart_split_semantic.py`, and
  `tests/test_boundary_task.py`; isolated positive test cases were removed
  from the remaining suites.

Kept because they are live or intentionally negative:

- `segmentation_orchestration.semantic_speaker_turns_carrier`. Here the name
  distinguishes an in-memory JSON value from the lexical snapshot carrier; it
  is unrelated to model boundary selection.
- P6 semantic-fact comparison, evidence, shadow, delta-registry, snapshot, and
  oracle terminology in `align_*`, `core/align_compare.py`, scripts, and
  their tests. These describe equivalence/relation facts, not the removed
  endpoint lane.
- Generic English uses in ASR correction, kinsoku, boundary-cost, finalizer,
  calibration, progress, schema-validation, and loader-behavior tests.
- Negative public-absence tests in `test_cli.py`, `test_config.py`,
  `test_default_diarize_packaging.py`, and `test_semantic_removal.py`.
  The latter now also checks the deleted module and internal parameters remain
  absent.
- The historical RAT-6 entry in `p6_ratifications.py`.
- The closed failure kinds/details in `align_failures.py` and the compatible
  align-shadow schema enums, per the controller ruling.
- `openai>=1.40` in `pyproject.toml`: it had no inventory hit and remains a
  live translation dependency.

## Load-bearing manifest and provider findings

The live P3 provider ledger has only `sentences`, `atoms`, and `pos`;
there is no retired-selector provider slot in production.

Outside the required inventory roots, the frozen sibling fixture
`calibration/p6-oracle/inputs/public-selected-v2-segmentation.json` contains
`segmentation.providers.semantic = "disabled"`. The oracle manifest pins that
786-byte file at SHA-256
`54956fea38bd9946a5440e9a636ce572297af3965963eb0d85d84f31c2d2489d`.
It was treated as load-bearing and left unchanged.

The P6 manifest and align-shadow compatibility schema still name the retired
failure rows. They are controller-owned law/compatibility data and were not
re-pinned or broadly rewritten. The initial attempted registry deletion exposed
the first stop; the registry is now restored exactly.

## Cache finding

A repository search found no reference to
`~/.cache/voxweave/semantic/`, no semantic cache creation call, and no
`mkdir` path capable of recreating that directory. The cleanup does not touch
the existing on-disk directory; its deletion remains an owner action.

## Latest-main reconciliation

Fetched `origin/main` and fast-forwarded the branch base from `2b5cb52` to
`ee92eb3` with Git's autostash handling the uncommitted cleanup. The stash
reapplied without conflicts.

The merged media-adjacent cache relocation is intact: `_artifact_owner()`
uses `claimed_sources(normalized_parent, carrier.name)`, matching the new
`<media-directory>/cache/<stem>/` ownership model. None of the cleanup hunks
overwrote that change.

## Validation

The initial pre-ruling checks passed 419 focused tests, Ruff checks,
compilation, and `git diff --check`. The first oracle run stopped on the
attempted registry deletion; the registry was restored exactly. After the first
controller ruling and latest-main merge, the full suite exposed the two stale
`reachable` coverage dispositions. The authorized amendment above resolved
that final law mismatch without changing a digest.

Final full gates, run after the manifest amendment:

```text
$ uv run --extra cuda pytest tests/ -q
3713 passed, 1 skipped in 344.29s (0:05:44)

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
256 files already formatted

$ uv run pyright
0 errors, 0 warnings, 0 informations

$ make quality-p6-oracle VARIANT=cuda
validate: exit 0
compare --check: exit 0
source-gates --check: exit 0

$ git diff --check
(exit 0)
```

The branch is committed in logical English/conventional commits, with the
manifest amendment isolated as required. Nothing was pushed.
