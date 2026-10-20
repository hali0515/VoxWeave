# Data contract

Each input keeps its editable delivery set beside the media:

- **`<stem>.json`** — the source of truth: word/character-level segments, language, VAD speech,
  plus optional replay data (`shot_changes`, `sing_spans`, `speaker_turns`) so `render` can
  redo shot snapping, lyric flagging, and speaker formatting without re-running any model.
- **`<stem>.vtt`** — editable subtitles. By default cues carry word-level timestamps (same
  precision as `align` output, ready to use); `--no-timestamps` writes a plain-text editing
  draft for hand-correction, which `align` re-times.
- **Subtitle-family deliverables** — translated `.vtt`/`.srt`/`.ass` files and derived files
  such as `<stem>.sdh.vtt` and `<stem>.asrfix.vtt` also stay beside the media.

Other episode state is stored beside the media, under `<media directory>/cache/<stem>/`.
For `episode.mkv`, the layout is:

```text
cache/episode/
├── source.json                    # source file-name claim (never an absolute path)
├── episode.episode.lock           # episode transaction lock
├── .episode-domain.lock           # same-directory stem publication lock
├── speakers.json                  # reviewed diarizer-id-to-name mapping
├── speakers.suggest.json          # regenerable match suggestions
├── voiceprints.json               # optional biometric centroids
├── speaker-split.undo.json        # files a confirmed split rewrote (one level)
├── vocals.32k.flac                # separated-vocals cache
├── vocals.32k.flac.meta.json      # source-bound integrity companion, when required
├── vocals.32k.flac.lock           # vocals-cache lock
├── episode.zh.progress.json       # interrupted translation resume state
├── episode.align-evidence.json    # durable alignment evidence
├── episode.asrfix.json            # correction audit
└── debug/                         # optional --debug bundle
```

Because the claim records only the source file name, moving or renaming the media
_directory_ (local disk to NAS, season folder reorganizations) keeps every artifact valid —
the cache travels with the media. The marker normally claims the media file; for a supported
standalone subtitle command with no discoverable sibling media, it claims that input file
instead. If another same-stem file in the same directory already owns the plain stem
(`episode.mkv` next to `episode.mp3`), VoxWeave uses `<stem>--<sha1-of-file-name-first-8>/`
for the second claim.

Every cache name above keeps its plain form while it fits the filesystem's file-name limit
(`NAME_MAX`, 255 bytes on most filesystems, counted in UTF-8 bytes). Only a name that would not
fit is shortened: `<stem, cut to fit>--<first 8 hex digits of sha1(stem)><suffix>`, for example
`<stem>.episode.lock` once the media stem reaches about 243 bytes, or the collision directory
at about 246. The shortened name is deterministic, and an entry is found under
whichever form it was written. Deliverables beside the media (`<stem>.json`, `<stem>.vtt`,
derived and translated subtitles, `pack`/`burn` outputs) are never shortened: when one of them
would not fit, the command stops before doing any work and names the file, so rename the media
(or pass a shorter `-o`) and run it again.

Existing adjacent machine sidecars remain compatible: when an adjacent
`<stem>.speakers.json`, `<stem>.speakers.suggest.json`, `<stem>.voiceprints.json`, translation
progress file, alignment-evidence file, correction audit, or old `cache/` vocals entry exists,
VoxWeave reads it first and writes updates back to that same legacy lane. New artifacts use the
cache. An existing adjacent `<stem>.episode.lock` is joined to the cache lock so old and new
VoxWeave processes still serialize the same episode. The adjacent `<stem>.json` transcript is
not a legacy exception: it remains the current source of truth by design. The
explicit/discovered show-level `voxweave.voices.json` is a user-managed database and also stays
where the user placed it.

## Sensitive and derived speaker data

Treat voiceprint artifacts, speaker-split undo snapshots, show-level voices stores, suggestion
records, in-memory auditions, calibration reports, crash-temporary files, and media snapshots
as sensitive or derived data.
The served audition contains embedded audio but is never saved by VoxWeave; a snapshot contains
private media bytes rather than biometrics. Snapshots live under
`${VOXWEAVE_CACHE_ROOT:-~/.cache/voxweave}/snapshots`, are mode
`0600` in an owner-only directory, and inactive crash residue older than one hour is cleaned on
a later snapshot creation. Active snapshots are lock-protected from that janitor.

The purge command removes only the episode voiceprints, suggestion record, split undo snapshot,
and any legacy audition HTML; it preserves the reviewed mapping, transcript, subtitles, and
unrelated cache state. Remove a show voices store, calibration report, inactive snapshot residue, or
crash-temporary file manually when it is no longer needed. Do not remove an active snapshot.
Backups, filesystem snapshots, synced folders, and manual copies are outside VoxWeave's control
and are not erased by purge; delete them separately according to their retention policy.

## Testing

- Unit tests (models mocked, no network): `make test` (= `uv run --extra $(VARIANT) pytest tests/ -v`).
  The suite ignores inherited `VOXWEAVE_*` settings; with a real-model gate such as
  `VOXWEAVE_REAL_WEIGHT_TESTS=1` set it keeps `VOXWEAVE_CACHE_ROOT`, `VOXWEAVE_MODEL_DIR` and
  `VOXWEAVE_HF_TOKEN`, so those tests find weights you already have. Its P6 oracle tests also
  need the interpreter recorded in `calibration/p6-oracle/manifest.json` (`make dev` syncs it,
  as CI does), Linux x86_64, a full clone (a shallow one lacks the oracle's reference commits)
  and the ambient environment the oracle records, which CI sets:
  `env -u TZ -u PYTHONHASHSEED LC_ALL=C.UTF-8 make test`.
- Lint / format: `make lint` (`make lint-check` is the read-only form CI runs; both use the
  pinned ruff version)
