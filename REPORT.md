# Speaker labeling phase 1 report

## Outcome

Phase 1 is complete. VoxWeave can now generate an offline speaker-audition page,
accept a minimal id-to-name mapping, render names without contaminating transcript
text or sibling JSON, and preserve name metadata through align, correct, export,
and translate workflows.

The implementation intentionally contains no cross-episode matching or embedding
storage. The versioned mapping file is the only phase-2 seam.

## CLI walkthrough

1. Produce the diarized transcript and persisted `speaker_turns`:

   ```text
   voxweave episode.mkv --diarize
   ```

2. Generate the audition artifacts:

   ```text
   voxweave speakers episode.mkv
   ```

   This writes:

   - `episode.speakers.html`: one self-contained `file://` page with embedded
     16 kHz mono MP3 snippets, a name input for every diarizer id, and live JSON.
   - `episode.speakers.json`: a directly editable skeleton with empty names.

   Each speaker gets at most three 2-6 second snippets. Selection intersects the
   speaker turn with VAD speech, removes every other speaker's turns and all
   `sing_spans`, then prefers long material across the early/middle/late thirds.

   If `speaker_turns` is missing, the command exits through the standard error
   panel with `run voxweave <media> --diarize first`. If the mapping already
   exists, the command refuses before ffmpeg runs. The protected mapping creation
   is atomic and no-replace, including against a concurrent writer.

3. Enter names in the HTML page and copy its JSON over the skeleton, or edit the
   skeleton directly.

4. Render the named VTT without rerunning diarization:

   ```text
   voxweave split episode.json
   ```

5. Export presentation formats as usual:

   ```text
   voxweave export episode.vtt --to srt --to ass
   ```

Names are display metadata. `segments[].text`, `word_segments`, and all text sent
to correction/translation models remain name-free.

## Mapping schema

```json
{
  "version": 1,
  "speakers": {
    "SPEAKER_00": "Aoi",
    "SPEAKER_01": ""
  }
}
```

- `version` must be `1`.
- `speakers` maps persisted diarizer ids to display names.
- Missing, non-string, whitespace-only, and empty names remain unlabeled.
- Unknown ids are ignored and reported in one `voxweave` logger warning per
  mapping read.
- The reader does not mutate or rewrite the mapping.

## Rendering matrix

| Format | Single named cue | Named dash cue | Unnamed cue/line |
| --- | --- | --- | --- |
| WebVTT | `<v Aoi>Hello</v>` | `<v Aoi>-Stay</v>` plus `<v Ren>-Go</v>` on the second line | Original text, with no voice tag |
| SRT | `Aoi: Hello` | `Aoi: -Stay` plus `Ren: -Go` on the second line | Original text, with no prefix |
| ASS | Dialogue `Name` is `Aoi`; text is `Hello` | One display-preserving Dialogue event; `Name` is `Aoi / Ren`, text remains `-Stay\N-Go` | Empty Dialogue `Name`; original text |

WebVTT parsing strips full-cue and per-line `<v ...>` wrappers into `speaker` /
`speakers` cue metadata. Rendering restores the wrappers after text work. Lyric
metadata remains independent, so `<v Aoi>♪ la la ♪</v>` round-trips with clean
`text == "la la"`, `speaker == "Aoi"`, and `lyric == true`.

During `split`, the existing diarization overlap/run logic supplies transient
speaker ids for single, split, and two-line dash cues. Those ids are explicitly
dropped alongside other in-memory-only cue fields before sibling JSON is written.

## Verification gates

Final gate tails from this worktree:

```text
$ uv run --extra cuda pytest tests/ -q
2313 passed in 8.86s
```

```text
$ uv run --no-project --with ruff ruff check .
All checks passed!
$ uv run --no-project --with ruff ruff format --check .
141 files already formatted
```

```text
$ uv run --extra cuda pyright
0 errors, 0 warnings, 0 informations
```

```text
$ uv run --extra cuda python .../p3_byte_baseline.py --check .../p3_byte_baseline.json
20/20 cases match
```

The added coverage exercises clean snippet geometry, temporal selection,
ffmpeg command/execution separation, timeout/stdin/atomic contracts, mapping
tolerance, CLI routing/refusal paths, VTT tag idempotence, lyric interaction,
SRT/ASS rendering, dash-line naming, translation payload isolation, correction
round-tripping, split attribution, and sibling JSON cleanliness.

## Phase 2 open questions

- What embedding representation and matching threshold should define a stable
  cross-episode identity, and how should uncertain matches be surfaced rather
  than silently accepted?
- Should canonical identities live in a separate show-level file, leaving this
  per-episode version-1 mapping as an override, or should phase 2 mint a new
  mapping version?
- How should one episode's diarizer split/merge errors reconcile with a canonical
  person when several local ids match one voice, or one local id contains several
  voices?
- What precedence should manual names have over embedding matches, and what audit
  trail should be retained when an automatic match changes?
- Should embeddings remain local-only, and what retention/deletion controls are
  required for voice-biometric data?

No choice above is encoded or scaffolded in phase 1.
