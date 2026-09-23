# Migration notes

What changed per release, newest first. Only releases that need an action or a
heads-up appear here.

## 0.18.0 (unreleased)

Most of this release concerns `--voiceprints` (cross-episode voice matching and **Split
this speaker**). One fix changes the audio a first run transcribes and diarizes, so a
first run's subtitles can differ slightly from 0.17.0.

### First runs hear the same audio as re-runs

With vocal separation on (the default), a first run used to normalize the full-band
stereo vocals for ASR and diarization, while every later run of the same media, which
reuses the cached `vocals.32k.flac`, normalized the 32 kHz mono copy. The two inputs
differed by about 2.5 dB and 10-14% of diarization frames, so re-running an episode could
change its speaker turns. A first run now derives its 16 kHz input from the 32 kHz mono
vocals too, exactly as a re-run does. Expect small differences against a 0.17.0 first
run; re-runs from an existing cache are unchanged.

### Opt-in voiceprint speaker clustering

`--diarize` can group the speaker turns by voiceprint instead of by pyannote's own
clustering: `--speaker-clustering voiceprint`, `VOXWEAVE_DIARIZE_CLUSTERING=voiceprint` or
`[diarize].clustering = "voiceprint"`. pyannote still finds who speaks when; ReDimNet2-B6
(the `redimnet2` voiceprint model: weights CC BY-NC-SA 4.0, a 51 MB download on first use)
decides who is who, and turns it cannot attribute confidently are dropped from
`speaker_turns`, so their words stay with the surrounding speaker. The default stays
`pyannote`, whose output is unchanged. If the voiceprint stage fails (download, out of
memory), the run warns and keeps pyannote's speakers. `--voiceprint-model pyannote` always
keeps pyannote's clustering, because its voiceprints are keyed by pyannote's labels.

### Voiceprints come from a dedicated embedder

`--voiceprints` used to store the diarization pipeline's own speaker embeddings. It now
keeps pyannote for the speaker turns but computes the voiceprints with a separate
speaker-embedding model chosen per language (`--voiceprint-model` /
`VOXWEAVE_VOICEPRINT_MODEL` / `[voiceprint].model`, default `auto`):

- `auto`: `anime-va` for Japanese, `redimnet2` for every other language;
- `redimnet2`: ReDimNet2-B6 (weights non-commercial, CC BY-NC-SA 4.0);
- `anime-va`: anime voice-actor ECAPA-TDNN (Japanese only);
- `pyannote`: the previous behavior.

A `--voiceprints` run fetches the checkpoint it may need (51 MB for `redimnet2`, 83 MB
for `anime-va`; with `auto` and no `--lang`, both) into `~/.cache/voxweave/audio/` and
verifies its SHA-256 before any audio work. If that fails (no network, a stalled
download), the run warns and continues without voiceprints; the subtitles are written as
usual. For an offline host:

- `redimnet2`: copy `b6-vb2+vox2+cnc2_v0-lm.pt` into `~/.cache/voxweave/audio/redimnet2/`,
  or point `VOXWEAVE_REDIMNET2_CKPT` at the file;
- `anime-va`: point `VOXWEAVE_ANIME_VA_CKPT` at `embedding_model.pth`. Its cache entry
  uses the Hugging Face hub layout (`models--litagin--.../snapshots/<revision>/`), so a
  file copied into `~/.cache/voxweave/audio/` is not found.

Either way the file must be the pinned checkpoint; its size and SHA-256 are verified.

**Existing voiceprints and voice stores do not match new captures.** A voice store is
tied to one embedding space, and the new embedders define new ones. Nothing is rewritten
or deleted, but with the new default:

- `speakers serve` skips an existing store with a warning that names both spaces
  (`store was built with pyannote embeddings (...); this run uses redimnet2-... embeddings`);
- `speakers enroll` refuses to add a new episode to it.

To keep using an existing store, capture new episodes on the legacy lane:

```bash
voxweave episode.mkv --diarize --voiceprints --voiceprint-model pyannote
```

or set it once in `~/.config/voxweave.conf`:

```toml
[voiceprint]
model = "pyannote"
```

Alternatively start a new store with the new embedder: re-run the reviewed episodes with
`--diarize --voiceprints` and `speakers enroll` them into a new `--voices` file.

In exchange, a store built by `redimnet2` or `anime-va` no longer depends on the
diarization pipeline: switching `--diarize-model` keeps it matching. The default
matching thresholds now follow the embedding space (`anime-va` uses a lower suggest
threshold, 0.35); `VOXWEAVE_VOICES_SUGGEST` / `VOXWEAVE_VOICES_MARGIN` still override
them. Those defaults are provisional; `scripts/calibrate_voiceprints.py` measures them on
your own diarized episodes.

**Split this speaker** embeds turns with whichever embedder the episode's voiceprints
were captured with, so legacy episodes keep splitting as before.

### Enrolled voices go to a global voice library

`speakers enroll` without `--voices` used to need a per-folder `voxweave.voices.json`
(created with `--voices PATH --show NAME`). It now saves into one voice library shared by
every media folder, by default `~/.local/share/voxweave/voices` (or
`$XDG_DATA_HOME/voxweave/voices`); `--voices-dir`, `VOXWEAVE_VOICES_DIR` or `[voices].dir`
move it, for example onto a NAS shared by several machines. `voxweave voices where` shows
which directory is in use. The first enrollment prints a notice that the library holds
voice biometrics; `voxweave voices forget ID` removes one person.

`speakers serve` suggests names from the library in two tiers: voices enrolled under the
episode's scope (`--show`, else the media folder's name) as before, and voices of every
other scope only above a stricter threshold (`VOXWEAVE_VOICES_GLOBAL_SUGGEST`), labelled
with where they were heard and never prefilled.

What stays the same, and what to do:

- `--voices FILE` still selects a per-show store explicitly and behaves exactly as before;
  it cannot be combined with `--voices-dir`.
- An existing `voxweave.voices.json` beside the media is still used for suggestions, now
  without `--show`, but is never written again. Merge it once with
  `voxweave voices import path/to/voxweave.voices.json`. Its voices keep both scopes the
  file served before, its folder's and its show's, so the same suggestions stay in the
  first tier (`--scope NAME` picks one scope instead; importing again with another
  `--scope` adds it). Importing keeps its ids, so running it again adds nothing,
  and the file is left untouched; delete it yourself once you no longer need it. Such a
  file keeps the vectors of everyone in it: `voxweave voices forget ID` stops VoxWeave
  from reading them for that person and lists the files that still hold them, but only
  deleting those files removes the vectors from disk.
- `--show` now also names the scope of an enrollment. Without it, the scope is the media
  folder's name, qualified with its parent for generic names such as `Season 1`, `S02`,
  `Disc 1` or `Specials` (`Frieren / Season 1`); pass `--show` where two unrelated folders
  share a distinctive name.
- Enrolling a name does not merge it with a same-named person of another scope unless you
  used that person's suggestion on the review page; use `voxweave voices list`,
  `show`, `rename` and `forget` to curate the result.

## 0.17.0

Two performance settings were added, both off by default, and successful runs gained
one extra line of stderr output; neither affects an existing run. The translate
change below is the one item here that needs your attention: it alters what an
unchanged `translate` command does.

### Translate: windowed requests and partial output

`translate` now sends the episode as bounded windows with several requests in
flight (`--concurrency`, `--window`; `[llm].concurrency` / `[llm].window_cues`;
defaults 8 / 100). Set `concurrency = 1` to keep the previous single
whole-episode request. Windows translated in parallel see the preceding source
cues as context instead of their neighbours' translations; use `--glossary` and
`--context` for cross-window consistency.

A run whose cues stay untranslated after the retry stage now fails instead of
writing a file with those cues in source text; the progress file is kept, so
rerunning the same command resumes. Pass `--allow-partial` for the previous
back-fill behavior. Responses that do not finish with `stop` (truncated or
aborted by the server) are retried, and the final attempt for a window drops
`response_format` (plain JSON) to get past structured-output failures on
self-hosted servers.

### Opt-in batched ASR decode

`[batch].asr` / `VOXWEAVE_ASR_BATCH` decodes several VAD chunks per Qwen3-ASR call.
It defaults to `1`, which is the previous per-chunk call, so an unchanged
configuration transcribes exactly as before. Raising it is a deliberate trade:
measured on an RTX PRO 4000, batch 4 is 1.34x faster at 6.4 GiB peak and batch 8 is
1.49x at 8.9 GiB, but batched transcripts drift ~1.5% CER from batch 1. Whisper and
the Apple Silicon MLX adapter ignore the setting (they take one chunk per call).

### Opt-in separator autocast

`[separate].autocast` / `VOXWEAVE_SEP_AUTOCAST` accepts `off` (default), `bf16`, or
`fp16` for the vocal-separation forward pass. `off` is the previous fp32 path, so an
unchanged configuration separates exactly as before. `bf16` measured 1.35x faster
with peak VRAM 1.69 → 1.57 GiB, at the cost of a slightly different stem (~52 dB SNR
against the fp32 stem) whose downstream ASR drifts ~2.3% CER. The setting is CUDA
only and is ignored on CPU/MPS; an unrecognized value warns once and falls back to
`off`.

Both knobs are described in more detail under
[Performance knobs](README.md#performance-knobs) in the README.

### Stage timing on stderr

A run that finishes cleanly now prints one extra muted line to stderr, after the
progress display:

```text
timing: inspect source 0.4s | prepare audio 1m17s | ... | total 3m41s
```

It lists wall-clock seconds per workflow step, under a minute as seconds and above
as `NmSSs`. Failed runs and runs without a declared plan print nothing extra. stdout
is unchanged — result paths and the speaker-service URL still go there alone — so a
script that reads stdout is unaffected; a script that captures stderr will see the
new line. A transcription run with `--debug` additionally records the steps finished
by mid-run in `debug/meta.json` under a `timings` key (a prefix of the printed line;
no other command writes that file).

Shot-change detection now runs as a background ffmpeg pass started before
transcription instead of a serial step afterwards. No option changed; its timing
entry is simply near zero because the work overlapped the GPU stages.

## 0.16.0

This release gives existing operations clearer names and makes the explicit
`transcribe` command work alongside `voxweave <media>`. Update commands and scripts
using the mappings below. File formats, output naming, configuration keys, and
environment variables are unchanged by this rename batch.

### Canonical command names

| Previous form | Use now |
| --- | --- |
| `voxweave split episode.json` | `voxweave render episode.json` |
| `voxweave speakers episode.mkv` | Still supported; shorthand for `voxweave speakers serve episode.mkv` |
| `voxweave speakers episode.mkv --enroll ...` | `voxweave speakers enroll episode.mkv ...` |
| `voxweave speakers episode.mkv --purge-voiceprints` | `voxweave speakers purge episode.mkv` |
| `voxweave speakers episode.mkv --no-match` | `voxweave speakers serve episode.mkv --manual` |
| `voxweave speakers episode.mkv --enroll --replace-episode ...` | `voxweave speakers enroll episode.mkv --replace ...` |

`split` and the old speaker action/option spellings remain accepted but are hidden
from help. Each deprecated spelling warns at most once per process, on stderr.
Bare `speakers <media>` is not deprecated and does not warn. No removal release
is scheduled here.

`speakers list EPISODE [--json]` is a read-only view of one episode's speaker turns,
reviewed names, and voiceprint state. It does not inspect a show-level voices store.
The speaker commands accept a media path or its JSON/VTT sibling. Store enrollment
still requires the existing explicit store/show selection; the rename does not
change how a discovered store becomes active.

### Options with one meaning

| Command | Use now | Hidden compatibility alias |
| --- | --- | --- |
| `transcribe` or bare media | `-m, --asr-model MODEL` | `--model MODEL` |
| `translate` | `-t, --target LANGUAGE` | `--to LANGUAGE` |
| `export` | `-f, --format FORMAT` | `--to FORMAT` |
| `pack`, `burn` | `--container CONTAINER` | `--to CONTAINER` |

These option aliases are permanent and silent. `--model` remains the canonical
model option for `translate` and `correct`: only the ASR option was renamed.
The translation endpoint, authentication, and `--reasoning-effort` options are
unchanged.

Use either the old or the new spelling for an option, never both in the same
command. Mixing them is a usage error even when their values agree. For multiple
export formats, repeat the canonical option:

```bash
voxweave translate episode.vtt --target zh
voxweave export episode.vtt --format srt --format ass
voxweave pack episode.zh.vtt --container mp4
voxweave burn episode.zh.vtt --container mkv
voxweave transcribe episode.mkv --asr-model qwen3-asr-1.7B
```

The repeated legacy form `export --to srt --to ass` still works, but combining
`--to` and `--format` does not. The same no-mixing rule applies to `--manual` versus
`--no-match`, and `--replace` versus `--replace-episode`.

### Input routing

Both transcription forms run the same operation:

```bash
voxweave episode.mkv
voxweave transcribe episode.mkv
```

Known commands take precedence over filenames. Use a path such as `./render` if a
media file's name is also a command. Unknown command words now produce a command
error instead of being treated as missing media files.

A bare subtitle or JSON path produces a usage error with an explicit next command;
it does not run transcription or silently select an in-place operation:

- Edited VTT: `voxweave align episode.vtt`.
- Subtitle format conversion: `voxweave export downloaded.srt --format vtt`.
- Layout from saved word timings: `voxweave render episode.json`.

`render` accepts the JSON, its VTT sibling, or the media path. It derives the sibling
JSON and rewrites the working VTT and JSON, just as `split` did. Save or align manual
VTT edits first. Saving speaker names still requires a separate `render` invocation
to put those names into the VTT; it does not automatically render on Save.

### Scope and unchanged behavior

This batch does not add a settings registry, `doctor`, `config`, or `status`
commands; backup or `--force` guards; batch processing; or a new exit-code taxonomy.
The 0.16.0 release does not change the existing in-place write behavior.
In particular, the `render` name is not a preview mode or an overwrite safeguard.

Existing translation endpoint configuration and numbered, shared-style progress
remain available. Progress and deprecation warnings go to stderr. Successful
processing commands retain their result paths on stdout; speaker serving retains
its URL output, and `speakers list --json` emits its requested inspection data.
