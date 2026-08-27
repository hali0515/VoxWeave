# Speaker labeling rework round 3

This final rework implements the two binding rulings and the closed list of nine
required dispositions from `REVIEW-FINDINGS-3.md`. The expressly out-of-scope
notes (additional hard-link errnos, non-string mapping diagnostics, abandoned
temporary files, and WebVTT class/newline tag syntax) were not changed.

## Binding rulings

### R-I — content-bound per-line speakers

Per-line metadata is now stored as `(name, original_line_text)` pairs. VTT and
SRT renderers resolve each current line through an exact content lookup rather
than its index: reordered matching lines retain the correct owner, an edited or
rewrapped line alone becomes unnamed, and an explicit cue-level speaker remains
attached across any line-count change. Ambiguous duplicate source text degrades
unnamed. Structured dash translation retains its existing per-half names by
rebinding the separately keyed translation halves to their source-content
identities before the same exact-match render.

Pins:

- `test_per_line_speakers_follow_content_when_lines_move`
- `test_per_line_speaker_mismatch_degrades_only_that_line`
- `test_cue_level_speaker_survives_line_count_change`
- `test_translate_rewrap_never_attributes_speakers_by_line_index`
- `test_correct_reapplies_only_unchanged_speaker_lines_by_content`

### R-J — language-tag-aware sidecar discovery

SRT mapping lookup now follows the existing sibling convention: it first checks
the exact sibling, then uses `detect_subtitle_language` and `swap_ext` to retry
without a recognized language tag. Thus `ep.zh.srt` finds `ep.speakers.json`
without enabling any sidecar-less inference.

Pin: `test_translated_srt_reuses_mapping_after_language_tag_is_stripped`.

## Required narrow-item dispositions

| Item | Disposition | Pin |
| --- | --- | --- |
| 1. RA-F3 burn | `burn` now retains the filtered blocks alongside timed rows and passes them to `render_ass`; mapping-recovered identities therefore reach the temporary ASS `Name` field. README documents that the visible dialogue stays prefix-free. | `test_burn_passes_mapping_recovered_srt_speakers_to_ass` |
| 2. RB-F4 / sweep major | Implemented R-I for both VTT and SRT. The reported equal-line-count rewrap no longer assigns `Ren` to Aoi's moved phrase, while unchanged lines can retain their own metadata instead of losing the entire cue. | R-I tests above |
| 3. RC Unicode edges | `sanitize_speaker_name` is now the single write/read normalizer. It strips Unicode whitespace only at name edges, preserves interior NBSP/ideographic spaces, and remains the input to the ASS-only comma escape. Voice-tag parsing calls that same function. | `test_edge_unicode_spaces_use_one_name_normalizer_for_srt_round_trip` |
| 4. Translated SRT | Implemented R-J. The real `ep.srt` → `ep.zh.srt` path reparses cue-level and per-line names through `ep.speakers.json`, keeping prefixes out of transcript text and preserving dash units. | `test_translated_srt_reuses_mapping_after_language_tag_is_stripped` |
| 5. Corrupt mapping during split | `split` now catches the same mapping I/O, encoding, JSON, version, and schema failures as the other consumers, emits exactly one `voxweave` warning, and replays without display names. | `test_split_ignores_corrupt_speaker_mapping_once` (empty, truncated, version 2, wrong schema) |
| 6. Audit `orig` | Applied audit records now replace the model's flattened quote with the actual pre-edit block text. Both `orig` and `fixed` therefore match their respective on-disk strings, including dash line breaks. | `test_correct_audit_records_reflowed_text_written_to_vtt` |
| 7. Multiple voice spans | Balanced voice spans are stripped independently across a physical line. A line containing multiple spans keeps all display text, receives no ambiguous line owner, leaves no tag residue, and sends clean text to translation. | `test_multiple_voice_spans_strip_balanced_without_line_attribution` (named and unnamed outer shapes) |
| 8. One continuous 6–18 second run | A qualifying run now yields two longest-possible 2–6 second windows with at least the mandated one-second inter-window gap. Separate short utterances retain the R-F behavior. | `test_select_snippets_keeps_two_separated_windows_in_one_long_run` (6, 12, and 18 seconds) |
| 9. Misleading round-2 wording | Renamed the fill test to say that its one-second gap is manufactured inside a continuous run. `REPORT-REWORK-2.md` now marks its SWEEP-6 work as partial and identifies the round-3 completion instead of claiming both halves were already fixed. | `test_select_snippets_fill_can_manufacture_gap_within_one_clean_run` |

## Required gate tails

### Full test suite

Command: `uv run --extra cuda pytest tests/ -q`

```text
........................................................................ [ 97%]
..................................................................       [100%]
2370 passed in 9.18s
```

### Ruff check

Command: `uv run --extra cuda ruff check .`

```text
All checks passed!
```

### Ruff format check

Command: `uv run --extra cuda ruff format --check .`

```text
147 files already formatted
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
