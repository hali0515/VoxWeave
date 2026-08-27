# Speaker labeling rework round 2

All binding rulings and all re-verification findings were accepted and fixed. No
finding was refuted. The sidecar-less SRT inference introduced in round 1 has
been deleted, generated SRT text is again provenance-safe, and corrupt speaker
mappings now degrade without blocking unrelated subtitle commands.

## Binding-ruling dispositions

| Ruling | Disposition | Regression pin |
| --- | --- | --- |
| R-A | Deleted `infer_srt_speaker_names`, its candidate regex, its export, and every call site. An SRT is parsed as literal text unless its exact sibling mapping is valid and contains the exact sanitized prefix. Recurring SDH labels and unrelated colon dialogue remain text. | `test_load_srt_without_mapping_preserves_recurring_sdh_prefixes`, `test_load_srt_without_mapping_preserves_colons_beside_dual_dash`, `test_named_srt_round_trip_recovers_clean_text_and_dash_metadata` |
| R-B | Removed synthesized composite names from display renderers. Matching line layouts retain per-line VTT tags and SRT prefixes; an irrecoverable multi-name collapse renders unnamed in VTT/SRT. ASS may still retain the deduplicated composite in its non-display `Name` field. | `test_distinct_line_names_render_unnamed_after_unrecoverable_collapse`, `test_collapsed_names_do_not_bake_into_srt_round_trip`, `test_correct_reapplies_only_unchanged_speaker_lines_by_content` |
| R-C | Split structural display-name normalization from the ASS delimiter escape. Record/line separators and ASCII layout runs collapse everywhere; VTT/SRT preserve commas and meaningful interior NBSP/ideographic spaces. Only ASS changes comma to full-width comma. SRT matching uses the same display sanitizer as SRT writing. | `test_all_named_renderers_share_structural_name_sanitization`, `test_name_sanitization_preserves_non_ascii_display_spaces`, `test_load_named_srt_matches_srt_sanitized_mapping_name` |
| R-D | SRT-side mapping lookup catches corrupt JSON/schema/version, decoding, permission, and other I/O failures. It logs exactly one `voxweave` warning per read and continues with literal SRT text. Export, translate, pack, and burn integration tests all cover this degradation. | `test_load_srt_ignores_corrupt_speaker_sidecar_once`, `test_load_srt_ignores_unreadable_speaker_sidecar_once`, plus the export/translate/pack/burn tests named below |
| R-E | Restored the completed-temp + fsync + hard-link publication path. `EPERM`, `EOPNOTSUPP`, and `EXDEV` fall back to an `O_EXCL` destination claim followed by rename over that owned claim. Existing destinations remain protected. The docstring records the fallback's small empty-claim crash window; R-D handles such truncated files. | `test_atomic_write_text_new_prefers_content_atomic_hard_link`, `test_atomic_write_text_new_claims_then_replaces_when_links_unavailable`, `test_atomic_write_text_new_fallback_still_refuses_existing_file`, `test_atomic_write_text_new_does_not_publish_incomplete_content` |
| R-F | Gap enforcement is now scoped to multiple cuts from the same continuous clean run. Separately voiced runs remain eligible even when their natural pause is below one second. | `test_select_snippets_keeps_close_but_disjoint_clean_utterances` |
| R-G | `apply_fixes` copies the post-`restore_dash_layout` string into each applied record, so the returned result and JSON audit match the VTT text written. | `test_correct_audit_records_reflowed_text_written_to_vtt` |
| R-H | Empty and whitespace-only WebVTT voice annotations unwrap to clean text with no speaker metadata, including entity-decoded whitespace. | `test_whitespace_only_voice_annotations_strip_as_clean_text` |

## Re-verification finding dispositions

| Finding | Disposition and evidence |
| --- | --- |
| F3 not fixed | Fixed at the root by R-A. The no-sidecar inference API and branch no longer exist. Mapping-gated SRT round trips still recover cue/per-line metadata, while sidecar-less translation sends the full literal SDH text—including the label—to the translator. `test_pipeline_translate_sidecarless_sdh_prefixes_as_dialogue` pins the model payload. |
| F4 not fixed | Fixed under the controller's R-B policy. A correction that can restore a dash boundary retains `Aoi`/`Ren` per line. An unrecoverable collapse emits clean unnamed VTT/SRT text, so `Aoi / Ren` cannot bake into a later SRT import; ASS alone retains the composite as metadata. |
| SWEEP-1 | Fixed. Recurring `MAN:`/`WOMAN:` text survives SRT parsing and export byte-for-display, with no voice metadata. `test_export_sidecarless_sdh_srt_preserves_literal_speaker_labels` checks VTT and ASS; `test_burn_ignores_corrupt_sidecar_and_keeps_sdh_prefix` inspects burn's generated ASS and confirms `MAN:` remains visible. |
| SWEEP-2 | Fixed by deleting document-wide inference. `test_load_srt_without_mapping_preserves_colons_beside_dual_dash` combines the reported dual-dash cue with `Listen:` and `Rule one:` controls and asserts that every prefix remains literal. |
| SWEEP-3 | Fixed with format-specific comma handling and one shared structural sanitizer. `Smith, Jr.` remains unchanged in VTT/SRT and becomes `Smith， Jr.` only in ASS; SRT mapping lookup closes that exact round trip. |
| SWEEP-4 | Fixed with content-atomic hard-link publication and the ruled fallback. Tests verify content is fsynced before the trusted destination appears, all three fallback errnos, exclusive no-replace behavior, and no temp residue. |
| SWEEP-5 | Fixed. Invalid and unreadable mappings degrade once at the shared reader. `test_export_srt_ignores_corrupt_speaker_sidecar_once`, `test_pipeline_translate_ignores_corrupt_speaker_sidecar_once`, `test_pack_ignores_corrupt_srt_speaker_sidecar_at_gate`, and `test_burn_ignores_corrupt_sidecar_and_keeps_sdh_prefix` cover every named consumer. |
| SWEEP-6 | Partially fixed in round 2 by associating each selected window with its originating clean run: the exact three 2.5-second utterances separated by 0.5 seconds all survive. The separate single-continuous-run concern remained unchanged in round 2 and is completed in round 3. |
| SWEEP-7 | Fixed as a documentation defect. The fill-pass comment states that it may select another usable window and that the same-run gap remains enforced; the round-3-renamed `test_select_snippets_fill_can_manufacture_gap_within_one_clean_run` pins the reported `(100,106)` / `(107,113)` behavior without describing that manufactured gap as a real pause. |
| SWEEP-8 | Fixed. Both the returned applied list and `.asrfix.json` now contain `-Stay here\n-Go now`, exactly matching the reflowed VTT body. |

## Required gate tails

### Full test suite

Command: `uv run --extra cuda pytest tests/ -q`

```text
........................................................................ [ 97%]
..................................................                       [100%]
2354 passed in 8.87s
```

### Ruff check

Command: `uv run --extra cuda ruff check .`

```text
All checks passed!
```

### Ruff format check

Command: `uv run --extra cuda ruff format --check .`

```text
145 files already formatted
```

### Pyright

Command: `uv run --extra cuda pyright`

```text
0 errors, 0 warnings, 0 informations
```

### Byte baseline

Command: `uv run --extra cuda python /tmp/claude-1000/-mnt-Dev-Git-qsub/76be7f61-2a96-4094-80b0-3b9311d0223b/scratchpad/p3_byte_baseline.py --check /tmp/claude-1000/-mnt-Dev-Git-qsub/76be7f61-2a96-4094-80b0-3b9311d0223b/scratchpad/p3_byte_baseline.json`

```text
20/20 cases match
```
