# Dead and duplicate code cleanup report

## Summary

**Implemented 32 / 32, skipped 0.** Net line delta **-694** (21 files, +159 / -853; production `voxweave/` -640, `tests/` -54).

Branch `chore/dead-dup-cleanup`, 3 logical commits + a merge of latest `main` (which had advanced 4 commits — the semantic-residue removal).

```
38c4a55 Merge branch 'main' into chore/dead-dup-cleanup
0e4a849 refactor: remove vestigial parameters and receipt fields
1de5acd refactor: consolidate duplicated helpers onto one implementation
4f4da33 refactor: drop dead evidence, sibling-JSON and phase-2 helpers
```

## Gate results

| Gate | Result |
|---|---|
| `pytest tests/ -q` (pre-merge) | **3773 passed, 1 skipped** (360s), exit 0 |
| `pytest tests/ -q` (post-merge-main) | **3713 passed, 1 skipped** (347s), exit 0 |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 255 files already formatted |
| `pyright` | 0 errors, 0 warnings, 0 informations |
| `make quality-p6-oracle VARIANT=cuda` | exit 0 — before **and** after the merge (validate / compare --check / source-gates --check). No revert-by-law was needed. |
| `git diff --check` | clean |

## Per-finding disposition

Every finding was re-verified in the worktree before cutting (repo-wide `grep -rIn` across production, `tests/` incl. monkeypatch target strings, `cli.py`, `README.md`, `__all__`, dynamic access), plus a static module-level import-closure check before each new cross-module import.

| # | Symbol | Disposition |
|---|---|---|
| 1 | `backend.chunk_pass_count` | `strategy` dropped end-to-end (signature, `del`, pipeline call site, `test_backend.py`). `strategy` still computed in pipeline for the live `transcribe_chunks(strategy=…)`. |
| 2 | `backend_mlx._load/_snapshot` | `_snapshot` deleted; both loaders use `runtime._hf_snapshot` (gains `_hf_error` hint + reporter progress). MLX-specific `_require` kept for the `mlx_audio` import. Test patch target updated. |
| 3 | `align_evidence` 8 builders | Deleted (238 lines) **+ closure**: `_work_value`, `_strict_failure`, `_lane`, `_denied_charge` (each re-verified to have exactly one repo-wide occurrence after the cut) and 8 unused imports → 377 lines. |
| 4 | `align_evidence._swap_ext` | Deleted; deferred `from voxweave.pipeline import swap_ext` (same pattern the function already uses for `_find_subtitle_media`). |
| 5 | `realign.fuse_punct_into_text(units)` | Removed from signature + all 16 call sites; dead `units = [...]` test fixtures dropped. |
| 6-9 | `pipeline._load_sibling_json`, `_write_align_json`, `_encode_align_json_bytes`, `_encode_sibling_json_bytes` | All deleted. `_dump_sibling_json` / `_write_siblings` untouched. |
| 10 | `diarize._embedding_load_authority` parents | Params + unreachable `"$model/"` branch removed. **Audit note corrected:** no test calls this with a raw `$model/` string (those tests feed `_expand_model_references` / `_write_pipeline_config`), so nothing broke. |
| 11+29 | `_snapshot_commit` | One implementation kept in `diarize` (the general `filename=` form its `config.yaml` site needs); turnembed's deleted. |
| 12+30 | `_canonical_embedding_source` | Algorithm kept in `diarize`; turnembed's is now a one-line dataclass view. |
| 13 | `diarize._sha256_file` | Deleted; imports `backend._sha256_file`. |
| 14 | turnembed community-1 literal | Now `diarize.COMMUNITY_DIARIZE_MODEL`. |
| 15-16 | `voicebase.bounded_delete`, `IdFactory` | Deleted (+ orphaned `Callable` import). |
| 17 | `speakerserve._validated_mapping/_mapping_entries` | **Deviation** — see below. |
| 18 | `speakerserve._undo_bytes` | Now `encode_json_bytes(value, max_bytes=MAX_UNDO_BYTES)`. |
| 19 | `speakers._subtract_spans` | Sweep from `songdet.subtract_spans`; only the `_merge_spans` normalization songdet's contract requires stays local. |
| 20 | `subformats.read_subtitle_text` | Deleted. |
| 21 | `mux._find_sibling_media` | Deleted; `resolve_media` uses `pipeline._find_sibling_media` (which the mux docstring and `test_mux_sibling.py` already claimed). |
| 22 | `episode_transaction.require_primary_generations` | Deleted incl. `__all__` entry. |
| 23 | `TransactionReceipt.auxiliary_landed/.leftovers` | Removed. Re-verified all same-named reads are on the annotated exception or on `pipeline._ProcessPublication`. |
| 24 | `artifacts._stem/_swap_ext` | `_swap_ext` forwards to `pipeline.swap_ext` via deferred import (module scope would cycle: pipeline imports artifacts); `_stem` built on it. |
| 25 | `MediaSnapshot(janitor_age_seconds)` | Param + attribute removed; `__enter__` reads `SNAPSHOT_MAX_AGE_SECONDS`. Not in README. |
| 26 | `vocalscache.cache_lock` | Uses `cache_lock_path(resolved)` / `cache_companion_path(resolved)` — passing the already-resolved path keeps results byte-identical without re-realpathing the caller's alias. |
| 27 | `speakers.FFMPEG_TIMEOUT` | Imported from `chunking`; still re-exported via `speakers.__all__`. |
| 28 | `export._srt_ts` | `fmt_ts(seconds).replace(".", ",", 1)`. |
| 31 | `vocalscache.canonical_cache_path` | Primitive hoisted to `voicebase.canonical_path`; both `canonical_cache_path` and `canonical_store_path` remain as their modules' public names (both in `__all__`, both used) but are one implementation. |
| 32 | `vocalscache.cache_publish_path` | Staging delegated to `fsio.atomic_path`; only the cache `CanonicalFailure` classification stays local. |

## Deviations from the literal finding text

1. **[17]** The finding wanted `_mapping_entries` to return `_validated_mapping(value, ids), after`. That is *not* behaviour-preserving: `_validated_mapping` also rejects top-level keys outside `{version, speakers}`, so an on-disk mapping with an extra key would start failing HTTP 500 instead of loading. The shared schema core was extracted into `_validated_speakers` and each caller keeps its own envelope rule. Duplication gone, accepted-input sets unchanged. The three distinct `ValueError` messages collapse to one, but both call sites discard the message (fixed HTTP 400 / 500 bodies).
2. **[31]** `duplicate_of` names `voicestore.canonical_store_path`, but the reuse is realized by hoisting the one-liner into `voicebase` (both depend on it; neither imports the other).
3. **[32]** The finding also names `episode_transaction._stage_bytes/_replace_stage` as a third copy. That is P6 transaction machinery and not the registered subject of any finding, so it was deliberately left alone.
4. **[3]** Removing the 8 named builders orphaned 4 more private helpers in the same file that were reachable only from them; the closure was removed and each was individually re-verified.

## Behavioural notes (non-default paths only)

- **[21]** `pipeline._find_sibling_media` case-folds only the *extension* (mux's copy folded the whole filename), and lets a `PermissionError` from `iterdir()` propagate where mux's copy returned `None`. Both are uncovered edge cases; the mux docstring and its tests already documented pipeline's function as the intended implementation.
- **[18]** On the over-`MAX_UNDO_BYTES` guard the raised type/message changes from `SplitConflict("…too large")` to `Phase2DataError("encoded JSON exceeds the …-byte limit")`. The sole call site catches both in the same `except` → HTTP 409 either way; success-path bytes are identical.
- **Import closures**: `voxweave.diarize` now transitively imports `voxweave.backend`, and `voxweave.speakers` imports `chunking` + `songdet`. Static closure check confirms no cycle in either direction (backend does not import diarize; songdet/chunking do not import speakers).
- **[2]** MLX HF downloads now surface the shared `repo`/`HF_TOKEN` hint and feed the CLI download bar.

## Independent verification before merge

A four-agent verification workflow (dangling references + import cycles, behavior
preservation of every consolidation, and a dedicated audit of the four deviations)
raised a single candidate finding — loss of mux's whole-filename case-insensitive
sibling matching — and refuted it: that behavior was an incidental artifact of a
lowercased-name dict, contradicted the helper's own docstring and tests (which
document delegation to `pipeline._find_sibling_media`), diverged from the
stem-exact contract every other media-resolution path enforces, and degrades only
to an actionable `FileNotFoundError` naming the `--media` escape hatch. Zero
findings were confirmed.
