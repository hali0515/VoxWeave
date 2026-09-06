# Migration notes

What changed per release, newest first. Only releases that need an action or a
heads-up appear here.

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
