# Speaker serve and artifact-cache report

## Feature A: localhost speaker audition and write-back

`voxweave speakers <media>` now returns an in-memory audition from the generation
layer and serves it only on `127.0.0.1`. The CLI prints the selected URL, opens it
best-effort unless `--no-open` is set, and accepts `--port` with an ephemeral-port
default. No audition HTML is written to disk.

The browser fetches the current mapping for each session, lets saved values override
machine suggestions, and writes reviewed names back through an authenticated `POST
/save`. The server validates Host, optional Origin, a per-session secret, body size,
and a closed JSON shape before taking the media's episode lock and atomically replacing
the mapping. It preserves skeleton speaker order and prints the adjacent transcript path
that can be passed to `voxweave split`. The Copy button remains as a fallback.

Existing mappings are now the normal edit path. Generation keeps the existing
snapshot, fingerprint, sibling, voiceprint, voices-store, and lock rechecks; it installs
an empty skeleton only when the mapping is absent and treats an atomic-install race as
an existing authoritative mapping. Legacy `.speakers.html` files are still removed by
the purge operation, but no new HTML file is produced.

### Feature A gates

- `uv run --extra cuda pytest tests/ -q`: 3,645 passed, 0 failed in 311.52s on
  the post-fix replay (the initial cold-environment run also passed, with four
  dependency syntax warnings).
- `uv run --no-project --with ruff ruff check .`: passed.
- `uv run --no-project --with ruff ruff format --check .`: passed after formatting
  one touched source file.
- `uv run --extra cuda pyright`: 0 errors, 0 warnings, 0 informations.

## Feature B: machine artifact cache

The amended owner decision is implemented: `<stem>.json` remains the adjacent
word-segment truth source, and all subtitle-family deliverables remain next to the
media. Speaker state, transaction state, vocals cache data, translation progress,
alignment evidence, correction audits, and debug artifacts now use one per-source
cache claim by default.

`VOXWEAVE_CACHE_ROOT` is evaluated at call time. Each claim has an owner-only directory
and a canonical `source.json` marker containing the normalized absolute source path.
The plain `<stem>/` directory is used when unclaimed or already owned by that source;
a collision uses `<stem>--<sha1(abs_source)[:8]>/` and a second collision fails closed.
Marker, artifact-lock, and vocals-lock handling rejects symlinks, FIFOs, non-regular
nodes, oversized/non-canonical markers, and marker replacement races.

### Exact current layout

For `episode.mkv`, the default tree is:

```text
${VOXWEAVE_CACHE_ROOT:-~/.cache/voxweave}/artifacts/episode/
├── source.json
├── episode.episode.lock
├── .episode-domain-<sha256>.lock
├── speakers.json
├── speakers.suggest.json
├── voiceprints.json
├── vocals.32k.flac
├── vocals.32k.flac.meta.json
├── vocals.32k.flac.lock
├── episode.zh.progress.json
├── episode.align-evidence.json
├── episode.asrfix.json
└── debug/
```

The hidden domain lock serializes media with the same parent/stem before a collision
claim exists. The per-claim episode locks then serialize every known claim in a stable
order. An existing adjacent `<stem>.episode.lock` is also acquired; a new adjacent lock
is never created. If an unselected cache marker is corrupt, an operation using a valid
legacy adjacent lane can still proceed under the stable domain lock, while an actual
cache access continues to validate and fail closed.

### Compatibility and placement matrix

| Artifact | New/default placement | Existing adjacent lane | Write/cleanup behavior |
| --- | --- | --- | --- |
| `<stem>.json` transcript | Beside media | Current, not legacy | All process/align/split behavior remains adjacent. |
| VTT/SRT/ASS and `.sdh.vtt`/`.asrfix.vtt` | Beside media | Current, not legacy | Deliverable paths and bytes remain unchanged. |
| Speaker mapping | `speakers.json` | `<stem>.speakers.json` wins | Saves, generation, split, and enroll write back to the selected legacy file; purge preserves mappings. |
| Speaker suggestions | `speakers.suggest.json` | `<stem>.speakers.suggest.json` wins | Generation writes back; invalidation and purge cover both known lanes. |
| Voiceprints | `voiceprints.json` | `<stem>.voiceprints.json` wins | Capture/enroll read or write back; invalidation and purge cover both known lanes. |
| Episode transaction lock | `<stem>.episode.lock` plus the domain lock above | Existing adjacent lock is joined | Cache lock is mode `0600`; no new adjacent lock is created or deleted. |
| Vocals FLAC, companion, lock | `vocals.32k.flac[.meta.json/.lock]` | Existing `media_dir/cache/<stem>.vocals.32k.flac` lane wins | Reads and refreshes stay in the legacy lane; the older 16k cache is read-only compatibility. A fresh run never creates adjacent `cache/`. |
| Translation resume state | `<input-stem>.<target>.progress.json` | Adjacent progress wins | Resume/write-back uses the selected lane; success retires both valid known lanes. Cleanup cannot turn a landed subtitle into failure or expose a poisoned unselected lane. |
| Alignment evidence | `<input-stem>.align-evidence.json` | Adjacent evidence wins | Align writes back to the selected lane; segmentation/correction invalidation covers both known lanes. |
| Correction audit | `<input-stem>.asrfix.json` | Adjacent audit wins | Review mode replaces the selected audit; `.asrfix.vtt` remains adjacent. |
| Debug bundle | `debug/` in the claim | No implicit legacy resolver | Production and direct sinks require an explicit cache/caller root; `debug/<stem>/` is no longer created implicitly. |
| Legacy audition HTML | Never produced | `<stem>.speakers.html` is purge-only | Purge continues removing old files. |
| Show voice database | User-selected/discovered path | Unchanged | `voxweave.voices.json` was deliberately not relocated. |

The speaker server re-resolves its mapping inside the episode lock, so an adjacent
legacy mapping that appears during a session becomes the authoritative write-back lane.
The exact generation installed for a pristine skeleton is carried from the atomic
install into the server, preventing an edit between generation and server startup from
being mistaken for the pristine empty skeleton.

### P6 oracle disposition

The first untouched runner invocation exposed a path-only failure: it still looked for
`episode.align-evidence.json` beside the fixture while production correctly published it
under the new cache root. The runner now pins a distinct temporary
`VOXWEAVE_CACHE_ROOT` for every isolated public-command case and reads that cached
evidence path. The standalone projector remains the sole byte authority.

No calibration expected artifact, oracle manifest, execution-record field, or production
golden byte changed. The release-refresh procedure was not invoked: there was no version,
dependency-lock digest, or container-digest change for that sanctioned procedure to
record, and the path-only runner adaptation made compare pass byte-for-byte.

### Feature B gates

- `uv run --extra cuda pytest tests/ -q`: **3,695 passed**, 0 failed in 322.52s.
- `uv run --no-project --with ruff ruff check .`: `All checks passed!`.
- `uv run --no-project --with ruff ruff format --check .`: 251 files already formatted.
- `uv run --extra cuda pyright`: 0 errors, 0 warnings, 0 informations.
- `make quality-segmentation`: pass, 20 cases, 0 failures, 1 existing warning;
  corpus digest prefix `23df693e438693bc`, metric prefix `89bf5b31753e9d80`.
- `make quality-p6-oracle VARIANT=cuda`: validate, compare/check, and source-gates/check
  all exited 0 with the manifest and expected bytes unchanged.
- Focused artifact/server/transaction replay: 371 passed, 0 failed.
- `git diff --check`: passed.

### Deliberately unchanged

- No relocation of `<stem>.json`, VTT/SRT/ASS deliverables, or the show-level voices
  database.
- No diarization/default/config change, model change, subtitle byte change, or production
  baseline re-record.
- No new audition HTML and no new adjacent `cache/` or episode-lock file.
- No P6 release refresh, no push, and no mutation of another checkout or worktree.
