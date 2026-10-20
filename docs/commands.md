# Commands

## Progress output

All processing commands share numbered workflow steps, a segmented dot bar, and
elapsed time. For example, while translating:

```text
[2/3] translate cues
translate -> zh       [⣿⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀] 128/400 0:00:12
```

The denominator reflects enabled steps, not an estimate of elapsed time. Downloads
and retries stay inside their current step. Cyan means active, green complete,
yellow warning, and red failure; elapsed time is subdued. Unknown totals use a
pulse instead of a fabricated percentage. Burn reports actual encoded frames/time.
Non-terminal runs emit static step lines and occasional encoding updates to stderr,
with no animation; stdout remains reserved for result paths or the speaker-service URL.

A run that finishes cleanly closes with one muted stderr line of wall-clock time per
workflow step, so "where did the time go" is answerable from any run:

```text
timing: inspect source 0.4s | prepare audio 1m17s | detect songs 31.6s | find speech 9.1s | transcribe and align 1m41s | detect shot changes 0.1s | layout subtitles 1.2s | write outputs 0.3s | total 3m41s
```

Anything under a minute is printed as seconds, anything above as `NmSSs`. A step
entered more than once accumulates; `total` is the sum of the listed steps. Failed
runs end in the error panel instead, and a run without a declared plan prints no
summary. A transcription run with `--debug` additionally records the steps finished
by mid-run in `debug/meta.json` under a `timings` key — that file is written while
`transcribe and align` is still open, so it holds a prefix of the printed line, and
only the transcribe path writes it.

## Transcribe

`voxweave transcribe <media>` (or `voxweave <media>`) — separation → song-skip → VAD chunking → ASR + forced alignment →
smart_split → writes `<stem>.vtt` (editable) + `<stem>.json` (word-level timestamp source of
truth). Models load in-process (see `voxweave.separator` and `voxweave.backend`); the separator is released from VRAM
before ASR+alignment load, so peak usage is ≈ max(sep, asr) rather than their sum.

```bash
voxweave episode.mkv
voxweave transcribe episode.mkv             # explicit form of the same command
voxweave clip.mp4 --no-separate          # clean speech (podcast/lecture): skip separation
voxweave episode.mkv --asr-model qwen3-asr-1.7B   # larger, more accurate ASR
voxweave episode.mkv --context "Ryland Grace, Astrophage, Hail Mary"   # bias names/terms
```

Unknown command words produce a command error. Bare subtitle or JSON paths are not
transcribed: use `align` for edited VTT, `export` for subtitle format conversion,
or `render` for layout from the sibling JSON. These inputs are never automatically
routed to an in-place editing command. See the [migration notes](https://github.com/hali0515/VoxWeave/blob/main/MIGRATING.md) for old names.

<details>
<summary><b>Options</b></summary>

| Option                              | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--language`                        | Force language (ISO code or full name); default auto-detect.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `--no-separate`                     | Skip vocal separation (for clean speech) to save GPU time.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `--no-skip-songs`                   | Keep lyrics / transcribe purely musical content (song-skip is on by default).                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `-m, --asr-model`                   | Local ASR model (default `Qwen3-ASR-0.6B`; `qwen3-asr-1.7B` is more accurate).                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `--context`                         | ASR bias prompt: names/terms likely to appear (comma or newline separated). Bare term lists are auto-framed as `Proper nouns: ...` for Qwen — a bare list actually _regresses_ accuracy ([details](https://github.com/TypeWhisper/typewhisper-mac/issues/321)); prose or pre-framed text passes through. Same as `VOXWEAVE_ASR_CONTEXT`.                                                                                                                                                                                                           |
| `--hybrid`                          | Dual-ASR fusion: Whisper text + Qwen punctuation. Whisper's error bias is the opposite of Qwen's (it hallucinates rather than omits), so use this when Qwen drops uncertain words.                                                                                                                                                                                                                                                                                                                                                                 |
| `--normalize/--no-normalize`        | Apply loudness normalization (`loudnorm`) to the 16k ASR input — helps when quiet words get dropped; off by default since it also amplifies noise.                                                                                                                                                                                                                                                                                                                                                                                                 |
| `--timestamps/--no-timestamps`      | VTT carries cue timing lines (word-level precision from alignment; default on); `--no-timestamps` writes a plain-text editing draft.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `--keep-lyrics`                     | Transcribe detected songs instead of skipping them; sung cues are wrapped `♪ ... ♪` (italic in ASS export). Requires vocal separation (warns and has no effect with `--no-separate`).                                                                                                                                                                                                                                                                                                                                                              |
| `--sdh`                             | Also write `<stem>.sdh.vtt`: PANNs non-speech event tags (`[explosion]`, `[phone ringing]`, ...) in speech-free gaps.                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `--diarize`                         | Opt in to the default-installed pyannote speaker diarizer: multi-speaker cues split at speaker boundaries; on two-line languages a short exchange becomes a Netflix dual-speaker event (`-line` per speaker). The gated checkpoint requires `VOXWEAVE_HF_TOKEN`, `HF_TOKEN`, config `hf_token`, or a prior `hf auth login`. Speaker turns persist to the sibling JSON, so `voxweave render` replays the formatting without re-running the model.                                                                                                   |
| `--diarize-model`                   | Select `community-1` (the default), `3.1`, or any full Hugging Face pipeline id. The same setting is available as `VOXWEAVE_DIARIZE_MODEL` or `[diarize].model`; precedence is CLI > env > config > default.                                                                                                                                                                                                                                                                                                                                       |
| `--speaker-clustering`              | How `--diarize` groups its turns into speakers: `pyannote` (the default: the pipeline's own clustering) or `voiceprint` (opt-in: regroup with ReDimNet2 voiceprints, recipe `voiceprint-v1`; turns nobody can be attributed to are dropped and their words stay with the surrounding speaker). The same setting is available as `VOXWEAVE_DIARIZE_CLUSTERING` or `[diarize].clustering`; precedence is CLI > env > config > default, and an unknown value is an error. See [Speaker models](speakers.md#speaker-models) for the model, its licence and the measured trade-offs. |
| `--voiceprints/--no-voiceprints`    | Opt in to a voice-biometric centroid sidecar for reviewed cross-episode speaker suggestions. Requires a fresh `--diarize` run and is off by default. Precedence: CLI, `VOXWEAVE_VOICEPRINTS`, `[defaults].voiceprints`, then off.                                                                                                                                                                                                                                                                                                                  |
| `--voiceprint-model`                | Speaker-embedding model for `--voiceprints`: `auto` (default: `anime-va` for Japanese, `redimnet2` otherwise), `redimnet2`, `anime-va`, or `pyannote` (legacy: the diarization pipeline's own embeddings, which matches pre-existing voice stores). Precedence: CLI, `VOXWEAVE_VOICEPRINT_MODEL`, `[voiceprint].model`, then `auto`; an unknown value is an error, and the CLI value is validated (with a no-effect warning) even when voiceprints are off. See [Speaker models](speakers.md#speaker-models) for the models and their licences.                                 |
| `--min-speakers` / `--max-speakers` | Bound the diarizer's speaker count. `--max-speakers` (e.g. `2` for an interview) is the single best lever against over-splitting on noisy material; set `--min-speakers` only when a missing speaker matters more than a split one, since a lower bound above the voices actually found splits voices (see [Speaker models](speakers.md#speaker-models)).                                                                                                                                                                                                                       |
| `--no-shot-snap`                    | Disable shot-change detection/snapping (cue boundaries otherwise land on cuts per the Netflix zone rules; `VOXWEAVE_SHOT_SNAP_MS` sets how far from a cut a boundary is moved).                                                                                                                                                                                                                                                                                                                                                                    |
| `--vad-mask/--no-vad-mask`          | Suppress wav2vec2 CTC emissions outside speech spans during alignment so words cannot park in music/silence (recommended for sparse-dialogue movies with songs; keep off when VAD may misjudge sung/whispered speech). Only languages aligned with a wav2vec2 CTC model use it (English by default); Japanese (MMS) and Chinese/Cantonese (Qwen) alignment ignore it. Same as `VOXWEAVE_VAD_EMISSION_MASK=1`.                                                                                                                                      |
| `--debug`                           | Write intermediate artifacts (full-band / vocals / per-chunk VAD + ASR + alignment) under `<media directory>/cache/<stem>/debug/`. Each run replaces the previous bundle.                                                                                                                                                                                                                                                                                                                                                                          |

The boolean flags (`--separate`, `--skip-songs`, `--normalize`, `--diarize`, `--voiceprints`, `--timestamps`,
`--shot-snap`, `--vad-mask`) can have their defaults set persistently via the `[defaults]`
section of `~/.config/voxweave.conf` — an explicit CLI flag always wins for that run.

</details>

## Re-align after editing

`voxweave align <vtt>` — takes the edited VTT text and **re-runs forced alignment against the
original audio**, overwriting the timestamped VTT and updating the JSON. Does not re-run ASR
or touch smart_split. Aligns on separated 16k vocals by default (prevents BGM interference);
prefers the episode's cached `vocals.32k.flac` under the artifact cache, otherwise re-separates
and caches it there. Existing media-adjacent `cache/<stem>.vocals.32k.flac` entries remain a
legacy read/write-back lane. When a voiceprint pair requires source binding, both managed and
legacy vocals caches are accepted only when their matching integrity companion validates.

Both VTT forms are accepted: the timestamped VTT and the plain-text editing draft that
`--no-timestamps` writes. The aligner strips punctuation as a hard constraint; ASR punctuation is
re-injected by time so the final output has correct spacing and breaks without stray marks.

On media longer than the single-pass alignment budget (about 30 min, `ctc_max_dp_frames`),
English and Japanese alignment runs in pieces split at silences between cues, so there the VTT
needs its cue timing lines. Overlapping or nested cues are fine; if a split would still cut a cue
off from its own audio (for example a cue whose start was mistyped far too early), `align`
refuses and names the overlapping cues so you can fix their times. Raising `ctc_max_dp_frames`
aligns the whole file in one pass instead, at the cost of more memory.

```bash
voxweave align episode.vtt                 # finds episode.<ext> in the same dir
voxweave align episode.vtt --media original.mkv
voxweave align episode.vtt --no-separate   # align on the original audio (clean sources)
```

<details>
<summary><b>Options</b></summary>

| Option                       | Description                                                                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--media`                    | Source media path (default: same-name file in the same directory).                                                                                    |
| `--language`                 | Force language (ISO code or full name); default: read from JSON.                                                                                      |
| `--no-separate`              | Align on the original audio instead of separated vocals.                                                                                              |
| `--normalize/--no-normalize` | Apply `loudnorm` to the 16k alignment input.                                                                                                          |
| `--vad-mask/--no-vad-mask`   | Suppress wav2vec2 CTC emissions outside the JSON's `vad_speech` spans (see the transcribe option of the same name; MMS and Qwen alignment ignore it). |

`--separate`, `--normalize`, and `--vad-mask` also honor the `[defaults]` section of
`~/.config/voxweave.conf` when not passed explicitly.

</details>

## Re-layout offline

`voxweave render <episode>` — re-run deterministic layout from `<stem>.json` without any models
(adjust line width / sentence breaks instantly).

```bash
voxweave render episode.json --max-line-length 14 --max-lines 1
voxweave render episode.json --no-timestamps   # plain-text editing draft
```

Pass the JSON, its VTT sibling, or the media path; each resolves to the same sibling JSON.
This command rewrites the working VTT and JSON and re-lays out every cue from the word
timings. It does not back up or protect manual VTT edits: save or `align` them first.

## ASR correction

`voxweave correct <vtt>` — optional **pre-align** LLM pass that fixes obvious ASR typos, split
words, and garbled proper nouns, producing a reviewable diff. Conservative substitution only
(no completion/rewrite), gated by a code check that the matched text equals the original
line-for-line. By default writes only an adjacent sidecar `<stem>.asrfix.vtt` plus an audit
JSON in the episode artifact cache — the original VTT is untouched. To accept a reviewed sidecar,
replace the VTT with it and run `align`. `--apply` instead lets the LLM rewrite the VTT in place
and then re-runs alignment automatically (skip that with `--no-align`, and pass `--media` when
the media is not a same-stem sibling).

```bash
voxweave correct episode.vtt --glossary names.json           # review the sidecar
voxweave correct episode.vtt --glossary names.json --apply   # rewrite in place + re-align
```

<details>
<summary><b>Options</b></summary>

| Option                         | Description                                                                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--glossary`                   | Term/name glossary (`.json` → mapping; other → raw prompt). Strongly recommended for ambiguous proper nouns.                                  |
| `--apply`                      | Let the LLM rewrite the original VTT in place, then re-align it (default: sidecar only, for review).                                          |
| `--align/--no-align`           | With `--apply`, re-run alignment afterwards to refresh timestamps (default: on).                                                              |
| `--media`                      | Media for that re-alignment when it is not a same-stem sibling of the VTT.                                                                    |
| `--model`                      | Correction model (default `VOXWEAVE_FIX_MODEL` env, `[llm].model` in the config, or `gpt-6-luna`; `auto` = the endpoint's only served model). |
| `--base-url` / `--api-key-env` | OpenAI-compatible endpoint + which env var holds the key (defaults from `[llm]` in the config; see [Configuration](configuration.md#configuration)).          |

</details>

## Translate

`voxweave translate <subtitle>` — **after align**, translate each cue with whole-episode
context, preserving cue count, into `<stem>.<target>.<ext>` (the original is left unchanged).
Accepts `.vtt`/`.srt`/`.ass`/`.ssa`; the output mirrors the input format
(`episode.srt` → `episode.zh.srt`). An ASS/SSA input keeps its script: only the text of each
translated event changes, behind the override tags that open it, so styles, positions
(`{\an8}`, `\pos`), Comment lines and drawings stay as they were. Position tags never reach
the model; an SRT cue's `{\an8}` is put back on its translation.

```bash
voxweave translate episode.vtt --target zh
voxweave translate episode.vtt --target en --context "sci-fi, formal register" --glossary terms.json
voxweave translate downloaded.srt -t zh               # foreign SRT in, SRT out
voxweave translate episode.vtt --target zh --reasoning-effort low
voxweave translate episode.vtt --target zh --concurrency 1   # one whole-episode request
```

Translation accepts any OpenAI-compatible Chat Completions endpoint, including a
self-hosted Qwen server. Configure it once in `~/.config/voxweave.conf`:

```toml
[llm]
base_url = "http://127.0.0.1:8000/v1"
model = "auto"              # or the exact served model ID from /v1/models
api_key_env = ""            # keyless server; otherwise name an environment variable
reasoning_effort = "low"    # translate only; omit to keep the server default
concurrency = 8             # translate windows in flight; 1 = single whole-episode request
window_cues = 100           # cues per window when concurrency > 1
```

By default the episode is translated as bounded windows (`window_cues` cues each)
with several requests in flight (`concurrency`); each window sees the preceding
source cues as context. This trades some cross-window stylistic continuity for
throughput and resilience on a self-hosted server -- `--glossary` and `--context`
are the consistency tools there. `--concurrency 1` sends one whole-episode request
(sequential windows with translated-tail continuity only past 800 cues), the best
choice for a hosted API and the default when no `base_url` is set.

Every response is checked for completeness: an answer that does not finish with
`stop` (length cap, a server aborting its structured-output grammar) is retried,
and the last attempt for that window runs without `response_format` (plain JSON).
Cues still untranslated after the retry stage fail the command and keep the
progress file, so rerunning resumes; `--allow-partial` writes the file anyway with
those cues in source text.

CLI options override environment variables, which override this configuration.
`--model` accepts a served model name without a built-in model list. `auto` requires
the endpoint to advertise exactly one model. For authenticated endpoints, set
`api_key_env = "MY_LLM_KEY"` and put the secret in that environment variable.

`--reasoning-effort` sends the standard top-level `reasoning_effort` field in the
request. Accepted values depend on the served model: for example, Qwen3.8-27B
supports `low`, `medium`, and `xhigh`. Use `--reasoning-effort default` to override
a configured value and omit the field. VoxWeave does not inject thinking template
switches or silently replace an effort rejected by the endpoint. Reasoning output
is kept out of subtitle text. Interrupted translations resume only when the
input, endpoint, resolved model, effort, context, glossary, and target match.

<details>
<summary><b>Options</b></summary>

| Option                         | Description                                                                                                                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-t, --target`                 | Target language code, written to `<stem>.<target>.<ext>` (default `zh`); use `export --format` to change file formats.                                                                         |
| `--context`                    | Show/tone context injected into the prompt.                                                                                                                                                    |
| `--glossary`                   | Term/name glossary (`.json` → mapping; other → raw prompt).                                                                                                                                    |
| `--model`                      | Translation model (default `VOXWEAVE_TRANSLATE_MODEL` env, `[llm].model` in the config, or `gpt-6-luna`; `auto` = the endpoint's only served model).                                           |
| `--base-url` / `--api-key-env` | OpenAI-compatible endpoint + which env var holds the key (defaults from `[llm]` in the config; see [Configuration](configuration.md#configuration)).                                                           |
| `--reasoning-effort`           | Model-specific effort. `VOXWEAVE_TRANSLATE_REASONING_EFFORT` > `[llm].reasoning_effort` > endpoint default; `default` explicitly omits the field.                                              |
| `--concurrency N`              | Windows in flight at once (`VOXWEAVE_TRANSLATE_CONCURRENCY` > `[llm].concurrency` > 8 with a base URL, 1 for api.openai.com). `1` = one whole-episode request with translated-tail continuity. |
| `--window N`                   | Cues per window when `--concurrency` > 1 (`VOXWEAVE_TRANSLATE_WINDOW_CUES` > `[llm].window_cues` > 100).                                                                                       |
| `--allow-partial`              | Write the output even when cues stay untranslated after the retry (they keep their source text). Default: fail and keep the progress file for a resumed rerun.                                 |

</details>

## Export

`voxweave export <subtitle>` — convert between subtitle formats: VTT/SRT/ASS/SSA in,
SRT/ASS/VTT out (written next to the input; the VTT + JSON pair stays the source of truth
for voxweave-produced subtitles). ASS output carries a Default style; lyric cues (`♪ ... ♪`)
render italic. Named VTT cues become `NAME: text` in SRT and use the ASS Dialogue `Name`
field. The position tag `{\an8}` (raise to the top; an ASS input's `\anN`/`\aN` override too)
stays in SRT output, becomes a real override in ASS and is dropped from VTT. Foreign SRT/ASS
files can be exported to VTT to enter the editing workflow. Every cue needs timestamps: a plain-text edit draft, or a file in
which only some cues are timed, is refused with the untimed cues named (run `align` on a VTT
first; an SRT needs its missing timing lines).

```bash
voxweave export episode.vtt --format srt
voxweave export episode.vtt -f srt -f ass
voxweave export downloaded.ass --format vtt        # foreign ASS -> VTT for editing/translate
```

Repeat `-f, --format` to request multiple output formats in one run.

## Pack (soft subtitles)

`voxweave pack <subtitle>...` — remux the source media with the subtitle file(s)
(VTT/SRT/ASS) added as proper subtitle tracks. Pure stream copy (instant, lossless,
reversible); each track is titled `VoxWeave <Language>` with the container language tag
taken from the filename (`episode.zh.vtt` → `chi` / "VoxWeave Chinese"), and the first
packed track is flagged default (the source tracks lose that flag) so players select it.
ASS inputs keep their styling in mkv targets (mp4/webm store text-only codecs, so styling
is dropped there).

```bash
voxweave pack episode.zh.vtt                    # finds episode.<ext>, keeps its container -> episode.zh.pack.mkv
voxweave pack episode.zh.vtt episode.ja.vtt     # several tracks at once -> episode.zh.ja.pack.mkv
voxweave pack episode.zh.vtt --container mp4    # mov_text in mp4 (image subs are dropped)
voxweave pack episode.zh.vtt --media other.mkv -o out.mkv
```

The output goes next to the source as `<media stem>.<languages>.pack.<container>`, the
languages taken from the subtitle file names (just `.pack` when none has one), so packing
another language never replaces an earlier result; `-o` picks any other path, and its
extension (`.mkv`/`.mp4`/`.webm`) picks the container. mkv targets keep every source stream
(including attachments; mov_text subtitles, which Matroska cannot store, are converted to
SRT); mp4/webm targets keep video+audio and existing _text_ subtitle tracks only (webm also
drops cover art). HEVC video muxed into mp4 is tagged `hvc1` for Apple players.

## Burn (hard subtitles)

`voxweave burn <subtitle>` — render the subtitles (VTT/SRT/ASS) into the pixels and write a
clean file with **all subtitle tracks removed**. For VTT/SRT input a styled ASS is generated
at the actual frame size (same look as `export`, lyric cues italic); ASS/SSA input goes to
libass as-is, keeping its own styling. The video is re-encoded at constant quality with
hardware acceleration when available: **NVENC** on NVIDIA, **VideoToolbox** on macOS,
libx264/libx265/libsvt-av1 software fallback. Audio is stream-copied (mp4 targets re-encode
mp4-incompatible codecs to AAC). The output goes next to the source as
`<media stem>.<language>.burn.<container>` (`episode.burn.mp4` when the subtitle file name
carries no language) unless `-o` names one.

```bash
voxweave burn episode.zh.vtt                          # hevc, auto hw encoder, -> episode.zh.burn.mp4
voxweave burn episode.zh.vtt --codec h264             # legacy-device compatibility
voxweave burn episode.zh.vtt --codec av1 --container mkv   # max compression, recent hardware
voxweave burn episode.zh.vtt --quality 20 --font "Noto Sans CJK SC"
```

<details>
<summary>Burn options & encoding policy</summary>

| Option             | Meaning                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `--codec`          | `hevc` (default: 10-bit capable, ~40% smaller than h264, plays everywhere as `hvc1` mp4) / `h264` / `av1`. |
| `--encoder`        | Force a specific ffmpeg encoder (default: auto-probe with a test encode).                                  |
| `--quality`        | Constant quality: NVENC `-cq` / software `-crf` (lower = better); VideoToolbox `-q:v` (higher = better).   |
| `--no-bitrate-cap` | Drop the default cap at the source's video bitrate, so constant quality alone decides the size.            |
| `--container`      | `mp4` (default, maximum compatibility) or `mkv`; an `-o` ending in `.mkv`/`.mp4` picks it.                 |
| `--font`           | Subtitle font family (fontconfig resolves fallbacks; e.g. `Noto Sans CJK SC`).                             |
| `--font-size`      | Override the default 72-at-1080p scaled size.                                                              |

Bitrate is never targeted: constant quality (`-b:v 0` on NVENC) lets the encoder spend bits
where the content needs them. It is capped, though, at the source's video bitrate (`-maxrate`;
capped CRF on x264/x265/SVT-AV1, a one-second data rate limit on VideoToolbox), because
constant quality alone can inflate a low-bitrate source: a 3440x1440 screen recording at
374 kb/s H.264 came out of NVENC `-cq 23` at 873 kb/s HEVC. Under the cap the output is no
larger than the source; content that needs fewer bits still gets fewer. The rate comes from
the stream (`bit_rate`, or matroska's `BPS` tag), else from the video stream's packet sizes
(one demux-only ffprobe pass), else from the file size over the duration minus attachments and
the other streams' known rates; when none can be read the cap is skipped with a warning.
Turn it off with `--no-bitrate-cap` when the target codec is less efficient than the
source's (an AV1 or VP9 source burnt to h264 needs more bits for the same picture). Output bit
depth follows the source dynamically (8-bit stays 8-bit, 10-bit stays 10-bit; 12-bit is kept
on libx265 and clamped to 10 on NVENC/VideoToolbox/SVT-AV1, which top out there) — except on
h264 paths, which are always 8-bit for player compatibility (NVENC h264 cannot encode 10-bit
at all).

</details>

Progress is rendered with rich: countable stages (demix windows / PANNs batches / per-chunk
ASR+alignment / align per-cue / translate streaming per-line) show a real `x/N` bar with
elapsed time; indeterminate stages (decode / file write) show a pulse bar. `-v/--verbose`
enables DEBUG logging.
