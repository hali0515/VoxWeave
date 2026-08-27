# Speaker labeling phase 1 rework report

## Outcome

All nine findings against `191c438` were confirmed and fixed. No finding was
refuted. The fixes preserve the phase-1 display-only model, keep the no-mapping
byte baseline intact, and add regression coverage for each executed failure
scenario.

## Per-finding disposition

### F1 — fixed: newline-bearing names corrupted ASS/SRT

Speaker names now pass through one structural sanitizer before rendering. ASCII
and Unicode line/record separators collapse to spaces, and commas become the
full-width comma used for the delimiter-free ASS Name field. Literal newlines and
`&#10;` annotations therefore render as `Ren Kai` rather than splitting an ASS
Dialogue record or injecting an SRT line.

Regression: `test_named_exports_sanitize_literal_and_entity_line_breaks` exports
the review's three-cue VTT and reparses all three ASS cues successfully.

### F2 — fixed: filtered translate rows used unfiltered speaker blocks

`pipeline.translate` now builds `timed_blocks` with the same start-and-end
predicate as `timed` and passes that aligned list to the SRT/ASS renderers.

Regression: `test_pipeline_translate_filters_srt_speaker_blocks_with_timed_rows`
uses the review's single-digit malformed end timestamp and verifies that the two
surviving cues remain `Ren: B` and `Kai: C`, with no one-position shift.

### F3 — fixed: named SRT prefixes were write-only

SRT input now recovers voxweave's `NAME: ` prefixes as cue metadata before model
payload construction. An exact sibling speaker mapping is the primary safe gate.
The no-sidecar round-trip from the review is also recognized through the
distinctive two-line prefixed dash pattern; otherwise a name must recur in
separate cues. A lone ordinary `Aoi: dialogue` cue without a mapping stays
literal, so the reader does not use a bare-colon heuristic.

Regression: `test_named_srt_round_trip_recovers_clean_text_and_dash_metadata`
pins VTT -> SRT -> parse/payload -> VTT without a mapping sidecar. It verifies
name-free model text, restored `parts`, and byte-identical voice tags. Two
additional subformat tests pin the mapping gate and isolated-colon control.

### F4 — fixed: line-count changes discarded per-line names

Correction now restores a flattened dual-dash cue's original speaker boundary
before rendering, so the exact in-house path remains two lines with `<v Aoi>`
and `<v Ren>`. For other text mutations where line attribution cannot be
reconstructed, VTT and SRT share a deterministic cue-level fallback containing
all distinct names (`Aoi / Ren`) instead of silently dropping metadata.

Regressions: `test_correct_keeps_all_names_when_a_fix_collapses_dual_speaker_lines`,
`test_restore_dash_layout_recovers_flat_correction_boundary`, and
`test_translate_keeps_all_names_when_per_line_text_collapses` pin both the
correction and third-party translation reproductions, including the SRT twin.

### F5 — fixed: foreign ASS Actor/Name leaked into display text

`parse_ass_blocks` no longer promotes a foreign ASS/SSA Dialogue Name/Actor field
to speaker metadata. The field remains opaque authoring metadata. Named ASS
output from a VTT source is unchanged; the unsafe read direction is removed.

Regression: `test_export_does_not_promote_foreign_ass_actor_names` uses the
review's `sign` / `TS note: fix later` / empty fixture and verifies clean SRT and
VTT text with no speaker keys or voice tags.

### F6 — fixed: sanitizer differed by output format

`sanitize_speaker_name` is now the single rule used by VTT, SRT, and ASS render
paths, including cue-level, per-line, and combined fallback labels. It handles
CR/LF, vertical/form feed, ASCII file/group/record separators, NEL, Unicode line
and paragraph separators, and the ASS comma delimiter.

Regression: `test_all_named_renderers_share_structural_name_sanitization` feeds
one name containing comma, CRLF, ASCII record separator, and Unicode line
separator through all three renderers and verifies the same safe spelling.

### F7 — fixed: audition forced audio stream 0

The audition clip command no longer passes `-map 0:a:0`. It now leaves audio
selection to ffmpeg, matching both `decode_to_wav` paths used for transcription
and diarization.

Regression: `test_clip_builder_leaves_audio_stream_selection_to_ffmpeg` pins the
production argv. The review's real sine-tone probe was rerun after the fix: all
five MKV/MP4 layouts reported `SAME`, including a later default track, a later
5.1 track with cleared dispositions, and a three-audio-stream file.

### F8 — fixed: one 6-second run became three adjacent fragments

Time bins now select full windows from original clean spans rather than clipping
spans at bin boundaries. Separate picks require a one-second gap, and the fill
pass only considers whole original spans. This preserves long audition windows
without manufacturing adjacent fragments.

Regression: `test_select_snippets_keeps_one_six_second_run_as_one_long_clip`
pins the exact `(100.0, 106.0)` reproduction. The review's length probe now gives
one 6.0-second clip at lengths 6.0, 6.01, 8.0, and 12.0; longer runs yield only
separated 6.0-second clips.

### F9 — fixed: protected creation required hard links

`atomic_write_text_new` now uses `O_CREAT | O_EXCL` and a fsynced write. It keeps
race-safe no-overwrite semantics, removes a partial destination on an exception,
and has no hard-link requirement.

Regression: `test_atomic_write_text_new_does_not_require_hard_links` replaces
`os.link` with an EPERM stub and verifies successful creation. The review's real
CLI reproduction was rerun: with `os.link` forced to EPERM, the first invocation
exited 0 and created both `ep.speakers.html` and `ep.speakers.json`; the second
correctly refused to overwrite the completed mapping.

## Executed reproduction checks

The supplied review drivers were rerun against the rework:

- newline-name ASS round-trip: 2 cues in, 2 cues out; no split Dialogue lines;
- foreign ASS Actor export: clean SRT/VTT text, no promoted names;
- malformed translate timing: surviving names are `Ren`, then `Kai`;
- named SRT round-trip: clean payload with `parts`, restored VTT voice tags;
- correction end-to-end: both original per-line voice tags survive;
- dual-audio frequency probe: 5/5 layouts use the transcription-selected stream;
- snippet length probe: the 6.0-second cliff is gone;
- hard-link-less CLI probe: exit 0 with both artifacts present.

## Final gate tails

```text
$ uv run --extra cuda pytest tests/ -q
2326 passed in 8.86s
```

```text
$ uv run --no-project --with ruff ruff check .
All checks passed!
$ uv run --no-project --with ruff ruff format --check .
142 files already formatted
```

```text
$ uv run --extra cuda pyright
0 errors, 0 warnings, 0 informations
```

```text
$ uv run --extra cuda python .../p3_byte_baseline.py --check .../p3_byte_baseline.json
20/20 cases match
```
