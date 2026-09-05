# CLI migration: 0.16.0

This change gives existing operations clearer names and makes the explicit
`transcribe` command work alongside `voxweave <media>`. Update commands and scripts
using the mappings below. File formats, output naming, configuration keys, and
environment variables are unchanged by this rename batch.

## Canonical command names

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

## Options with one meaning

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

## Input routing

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

## Scope and unchanged behavior

This batch does not add a settings registry, `doctor`, `config`, or `status`
commands; backup or `--force` guards; batch processing; or a new exit-code taxonomy.
The 0.16.0 release does not change the existing in-place write behavior.
In particular, the `render` name is not a preview mode or an overwrite safeguard.

Existing translation endpoint configuration and numbered, shared-style progress
remain available. Progress and deprecation warnings go to stderr. Successful
processing commands retain their result paths on stdout; speaker serving retains
its URL output, and `speakers list --json` emits its requested inspection data.
