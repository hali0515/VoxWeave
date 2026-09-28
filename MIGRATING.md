# Migration notes

What changed per release, newest first. Only releases that need an action or a
heads-up appear here.

## 0.18.0 (unreleased)

Most of this release concerns `--voiceprints` (cross-episode voice matching and **Split
this speaker**). Several fixes change output: a first run transcribes and diarizes the
same audio as a re-run, and subtitle layout changed for Latin/Cyrillic/Greek text, CJK
line wraps and cue gaps, so subtitles can differ slightly from 0.17.0. `speakers serve`
now opens only through the access link it prints, and deliverables are written with
your umask instead of `0600`.

### First runs transcribe and diarize the same audio as re-runs

With vocal separation on (the default), a first run used to derive its 16 kHz ASR and
diarization input straight from the 44.1 kHz stereo vocals, while every later run of the
same media decodes the cached 32 kHz mono copy (`vocals.32k.flac`). A first run now goes
through the same 32 kHz mono vocals, so both runs feed ASR and diarization identical
samples. Without `--normalize` (the default) the old gap was only a resampling difference.
With `--normalize` it was larger: loudness normalization measured the two inputs about
2.5 dB apart and moved 10-14% of diarization frames, so a re-run could change the speaker
turns. Expect small differences against a 0.17.0 first run. Re-runs from an existing cache
feed ASR and diarization the same samples as before, and now also find the speech timing
reference (gap splitting, snapping words to speech) on the original mix, as a first run
does. They used the separated vocals for it, so re-running an episode (for example to add
`--diarize`) could move cue timing; expect small timing differences against a 0.17.0 re-run.

### ASR models that cannot load stop the run at once

A model that cannot be loaded (a missing package, an unknown `--asr-model` id, no network for
the first download) used to be retried on every chunk, and the run failed only after the whole
ASR pass with "ASR failed on all N chunks". It now stops immediately with the original error.
Under `--hybrid` this covers both engines: a whisper or Qwen engine that cannot load fails the
run instead of producing a single-engine transcript. If an engine loads but then fails on every
chunk, `--hybrid` keeps the other engine's transcript and says so in a warning.

### TF32 is limited to vocal separation

On CUDA, vocal separation runs its matrix multiplies in TF32 (`VOXWEAVE_TF32=0` turns this
off). The setting used to stay on for the rest of the process, so on a run that separated
vocals, later fp32 stages (song detection, `--sdh` scoring, the English wav2vec2 aligner) also
ran in TF32, while a re-run from the vocals cache ran them in full fp32. It is now restored
when separation finishes: a first run and a cached re-run compute the same numbers, and a first
run's song spans and English word timings can differ very slightly from 0.17.0.

### Model directory

- With `VOXWEAVE_CACHE_ROOT` set, an explicitly placed separator checkpoint
  (`vocals_mel_band_roformer.ckpt` + `.yaml`) is looked for under that root, like every other
  model, unless `VOXWEAVE_MODEL_DIR` says otherwise.
- Importing voxweave no longer moves a pre-rename `~/.cache/qsub` directory into place.

### Subtitle layout changes

`voxweave <media>` and `render` lay out some cues differently from 0.17.0; run `render` on an
existing episode's JSON to re-lay it out with the new rules.

- Latin, Cyrillic and Greek letters count as one column and combining marks as none. Every
  non-ASCII character used to count as two, so Russian and accented Latin lines were packed
  and wrapped at about half the configured width.
- Two-line CJK cues prefer to wrap at a jieba/BudouX word boundary over a break inside a
  word, and と/まで/より are kept whole.
- Extending a cue (minimum duration, linger, tail pad) stops two frames before the next cue
  instead of running up to it.
- A word without timing is shown over its own stretch instead of its whole parent span, and
  a cue left shorter than the minimum duration borrows spare display time from its neighbours.

### `--diarize` refuses a run that cannot succeed before the audio work

`voxweave MEDIA --diarize` now checks the Hugging Face token for a gated diarization model
(the default community-1 and 3.1 are both gated) and the `--min-speakers`/`--max-speakers`
bounds before vocal separation and ASR. Previously it only failed after them. Accept the model
card and run `hf auth login` (or set `VOXWEAVE_HF_TOKEN` / `HF_TOKEN` / `hf_token` in the
config) before the run.

### `--debug` bundle

Each `--debug` run now replaces the previous `cache/<stem>/debug/` bundle instead of mixing its
chunk files with the old ones, and `chunks/*.raw.txt` is no longer written (it repeated
`*.text.txt`). Library callers: `pipeline.transcribe()` no longer takes `debug_stem`, and
`debug.FileDebugSink` takes the bundle directory as its only argument.

### Opt-in voiceprint speaker clustering

`--diarize` can group the speaker turns by voiceprint instead of by pyannote's own
clustering. It is opt-in: `--speaker-clustering voiceprint`,
`VOXWEAVE_DIARIZE_CLUSTERING=voiceprint` or `[diarize].clustering = "voiceprint"`. pyannote
still finds who speaks when; the `voiceprint-v1` recipe decides who is who with ReDimNet2-B6
(the `redimnet2` voiceprint model: weights CC BY-NC-SA 4.0, a 51 MB download on first use).
It clusters the turns that hold at least 1 s of speech nobody talks over (turns overlapping
in time for most of their length never share a speaker), gives the other turns to the
closest voice nearby, and drops from `speaker_turns` the turns no voice is close enough to,
so their words stay with the surrounding speaker. In our measurements it confused speakers less than pyannote on three of
four public test sets and got more short backchannels right in an online meeting, but it can
report more speakers than there are, and a speaker who talks very little may be merged into
others or dropped; that is why the default stays `pyannote`, whose output is unchanged.
`--min-speakers`/`--max-speakers` bind the voiceprint stage too; a `--min-speakers` above the
number of voices it finds is met by dividing those voices, as pyannote does under that bound
(it cost the stage less accuracy than pyannote in our measurements, but more than no bound).
If it fails (download, out of memory), cannot meet those bounds, or finds no turn long enough
to anchor a voiceprint, the run warns and keeps pyannote's speakers.
`--voiceprint-model pyannote` always keeps pyannote's clustering, because its voiceprints are
keyed by pyannote's labels.

Saved speaker names are keyed by speaker id (`SPEAKER_00`, ...), and switching
`--speaker-clustering` (like switching `--diarize-model`) renumbers the speakers of an episode
you already named. A transcription run now warns when a named id's turns changed; review the names with
`voxweave speakers <media>` before running `voxweave speakers enroll`, or a voice can be
stored in the voice library under someone else's name.

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
for `anime-va`; with `auto` and no `--language`, both) into `~/.cache/voxweave/audio/` and
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

Alternatively re-run the reviewed episodes with `--diarize --voiceprints` and
`voxweave speakers enroll EPISODE`; the voice library keeps the new embedder's voices in their
own embedding space.

In exchange, a store built by `redimnet2` or `anime-va` no longer depends on the
diarization pipeline: switching `--diarize-model` keeps it matching. The default
matching thresholds now follow the embedding space (`anime-va` uses a lower suggest
threshold, 0.35); `VOXWEAVE_VOICES_SUGGEST` / `VOXWEAVE_VOICES_MARGIN` still override
them. Those defaults are provisional; `scripts/calibrate_voiceprints.py` measures them on
your own diarized episodes.

**Split this speaker** embeds turns with whichever embedder the episode's voiceprints
were captured with, so legacy episodes keep splitting as before.

### `speakers serve` opens through its access link

The audition page, `/serve-info`, saves and splits now require a session cookie that the
printed link (`http://127.0.0.1:PORT/?k=...`) sets; a bookmark or a hand-typed
`http://127.0.0.1:PORT/` gets 403. Open the link the command prints (it is also what the
browser is opened with); with `--host 0.0.0.0` replace `0.0.0.0` in it with the machine's IP;
with `--ngrok` append the printed `/?k=...` to the tunnel URL. The key changes on every start,
so restart links do not carry over.

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
- A library belongs to one user account. Machines that share it on a NAS must reach it as
  the same user (the same uid on NFS); sharing one library between accounts is not supported.
- A library whose `identities.json` is missing is refused with a message saying what to
  restore, and voice samples whose identity is missing from a rolled-back `identities.json`
  are kept on disk until it is restored. Only samples of an identity removed with
  `voxweave voices forget` are deleted.

### `burn` output is no larger than the source

`burn` still encodes at constant quality, but now caps the video bitrate at the source's.
Constant quality alone could inflate a low-bitrate source: a 49-minute 3440x1440 screen
recording at 374 kb/s H.264 (170 MB) came out of the default NVENC `-cq 23` at 873 kb/s HEVC
(340 MB). Sources that need fewer bits than the cap still get fewer, so the output only
changes where it used to exceed the source rate. Pass `--no-bitrate-cap` for the old
behaviour, for example when burning an AV1 or VP9 source to h264, which needs more bits than
the source for the same picture. When the source rate cannot be read, `burn` warns and
encodes without a cap.

### `pack` and `burn` name their output after the subtitle language

Without `-o`, the output used to be `<media stem>.<container>` (`.pack`/`.burn` was added
only when that name was the source itself), so packing or burning a second language into
the same media silently replaced the first result. The default name now always carries the
command and the subtitle languages taken from the subtitle file names:

- `burn episode.zh.vtt` writes `episode.zh.burn.mp4` (was `episode.mp4`);
  `burn episode.vtt` writes `episode.burn.mp4`.
- `pack episode.zh.vtt episode.ja.vtt` writes `episode.zh.ja.pack.mkv` (languages in
  argument order, each once); with no language in any file name it is `episode.pack.mkv`.

Running the same command again still replaces its own previous output. Scripts that pick up
the old name should pass `-o` or use the new one.

### `pack` and `burn` take the container from `-o`

An `-o` path ending in `.mkv`, `.mp4` or `.webm` now decides the container, and the ffmpeg
command is built for it (for example mov_text subtitles for `-o out.mp4` from an mkv source).
Before, the container came from the source or `--container` while ffmpeg picked its muxer from
the file name. A `--container` that contradicts such an `-o` is overridden with a warning. Any
other extension (`.m4v`, `.mov`) keeps `--container` and forces its muxer. `burn -o x.webm` is
refused (burn writes mp4 or mkv). `pack` into mp4 now refuses up front audio the mp4 muxer
cannot store (`pcm_u8`, `wmav2`, `truehd`, ...; pack into mkv instead), and `pack` into webm
drops cover art. Warnings ffmpeg or libass print during a successful `pack`/`burn` (for example
a font without glyphs for the subtitle text) are now shown instead of discarded. A hidden
`.<name>.<random>.part.<ext>` file that a killed `pack` or `burn` left next to the media is now
deleted by the next voxweave write in that directory once nothing has written it for 5
minutes. Before, it stayed until you deleted it.

### Output files honour your umask

Subtitles (VTT/SRT/ASS), the sibling `.json`, the `.sdh.vtt`/`.asrfix.vtt` sidecars and
`pack`/`burn` outputs used to be written with mode `0600` whatever your umask. They are now
created like any other file (`0644` under the usual umask `022`), and rewriting an existing
file keeps its current mode. Files written by earlier versions therefore keep `0600` when
rewritten; if other accounts or services need to read them, `chmod 644` them once (for
example `chmod 644 *.vtt *.srt *.json`). Private data stays private: the voice library and
its locks, speaker mappings, voiceprints, align evidence, translation progress, the
correction audit and the vocals cache are written `0600`, and everything under
`cache/<stem>/` (including `--debug` dumps) lives in a `0700` directory.

### `pack` into mkv keeps mov_text subtitles and a single default track

Packing into mkv (the default for `.mov` sources) failed when the source carried mov_text
(3GPP timed text) subtitles, which Matroska cannot store; those tracks are now converted to
SRT. The first packed track was flagged default, but a source subtitle track flagged default
kept its flag, so players could still pick the old track; that flag is now cleared (other
flags such as forced stay).

### Subtitle conversion refuses partly timed files and keeps `{\an8}`

- `export`, `burn` and `pack` refuse a subtitle file in which only some cues have
  timestamps, as `translate` does for SRT/ASS, instead of silently leaving the untimed cues
  out. The error names the first untimed cues; run `voxweave align` on a VTT, or add the
  missing timing lines to an SRT.
- The SRT position tag `{\an8}` (and the SSA form `{\a6}`) no longer shows up as literal
  `(\an8)` text in ASS export or burned video: it is kept as an ASS override, stays in SRT
  output and is dropped from VTT output.
- ASS vector drawings (`{\p1}m 0 0 l ...`) are no longer read as dialogue; an event that only
  draws a shape produces no cue.

### `align` on long media accepts nested cues

Above the single-pass alignment budget (about 30 min, `ctc_max_dp_frames`), English and
Japanese alignment is split at silences between cues. Overlapping or nested cue times (such
as a short cue stretched across its successor) used to refuse the whole file; they are now
planned on their combined extent. If a split would still cut a cue off from its own audio
(for example one cue whose start was mistyped far too early), `align` refuses and names the
overlapping cues and their times; fix those cue times and run it again. VTTs whose cue times
never overlap are planned exactly as before, and plans that fit the budget before still do.

### Library API: smart_split parameters

`voxweave.core.smart_split.smart_split_segments` and `split_long_cues_with_word_timings` no
longer accept `min_duration` or `desired_wps`; neither argument had any effect. Their optional
arguments are now keyword-only: everything after `max_lines` in `smart_split_segments`
(`split_at_comma`, `comma_split_min_len`, `speech_spans`, `thresholds`, `shot_changes`), and
`speech_spans`/`thresholds` in `split_long_cues_with_word_timings`. Passing a removed argument,
by keyword or positionally, including through `pipeline.split(..., **kwargs)`, raises
`TypeError`: drop it, and pass the remaining optional arguments by keyword.

### Library API: `pipeline` split into smaller modules

`voxweave.pipeline` keeps `transcribe`, `process`, `split` and `align`. The helpers it also
held moved to their own modules, and its other public names (`swap_ext`, `require_vtt`,
`MEDIA_EXTS`, `segment_document`, `SegmentationResult`, `cache_vocals_path`, the
`speakers_*_path` and `voiceprints_path` helpers, `lyric_display_text`, ...) stay importable
from `pipeline`. Three things changed:

- `pipeline.translate` and `pipeline.correct` are now `voxweave.llm_commands.translate` and
  `voxweave.llm_commands.correct`, with the same signatures. `correct --apply` calls
  `pipeline.align`, so `pipeline` cannot import them back.
- `SHADOW_LANE_DELIVERY` is gone from `pipeline` and `voxweave.core.shadow_v2`; use
  `SHADOW_LANE_DELIVERY_LEGACY`, which has the same value.
- `SegmentationResult` no longer has `thresholds_used`: read the thresholds a run used from
  `result.manifest["profile"]` (the nine gap/duration keys, as the engine ran them). The
  field was positional, so construct a `SegmentationResult` by keyword.

Code that imported or patched private `pipeline` helpers finds them here:
`voxweave.paths` (`swap_ext`, `find_sibling_media`, `find_subtitle_media`),
`voxweave.sidecars` (`artifact_owner` and the speaker and voiceprint sidecar paths),
`voxweave.vocals` (the vocals cache and `acquire_16k`, the one 16 kHz input flow shared by
`transcribe` and `align`), `voxweave.segmentation` (`segment_document`) and
`voxweave.core.overlay` (`spans_in`, `turns_in`, lyric marking and the shot re-snap).
`voxweave.mux.detect_subtitle_language` and `voxweave.subformats.SUBTITLE_EXTS` still import
from their old modules.

### Smaller changes

- `--language` accepts ISO codes such as `ja` with the Qwen engine; before, every chunk failed
  after the whole separation and ASR pass. An unsupported value fails before any model loads.
- `correct --apply` re-aligns with the `[defaults]` `separate`/`normalize`/`vad_mask` settings,
  like `align`.
- A `translate` or `correct` request times out after 300 s (`VOXWEAVE_LLM_TIMEOUT_S`) and the
  OpenAI SDK no longer retries underneath VoxWeave's own retries; before, one attempt could
  wait 600 s three times. Raise the timeout for a slow self-hosted endpoint.
- A malformed `VOXWEAVE_*` number or config value warns once and falls back to its default,
  instead of crashing every command at import or being ignored silently.
- `--vad-mask` only ever applied to wav2vec2 CTC alignment (English by default); with
  Japanese MMS alignment it now warns that it has no effect.
- `align` no longer drops plain-text draft lines that begin with "Note", "Style" or "Region".
- Aligning Thai, Lao or Burmese with a configured CTC or MMS aligner (`[align] th = "mms"`)
  no longer aborts.
- `--hybrid` keeps the decimal point in numbers (`14.2`, not `142`) and no longer doubles
  punctuation Whisper already has.
- Voiceprint clustering keeps pyannote's speakers for recordings with more than 5000 anchor
  turns instead of running for a very long time.
- `--help` no longer writes `~/.config/voxweave.conf`.
- `burn` works with subtitle paths that contain `:` or quotes (`Star Wars: A New Hope.ass`).
- `pack` and `burn` accept GBK/Big5 subtitles that the other commands read, instead of
  losing CJK cues or failing, and find the media for `X.sdh.vtt` / `X.asrfix.vtt` as `align`
  does.
- After an episode is re-diarized, the audition page's Save and Split work again: a saved
  mapping entry for a speaker id that no longer exists is ignored (and reported) instead of
  failing the page.

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
