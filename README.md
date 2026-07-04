<div align="center">

<img src="resources/VoxWeave_icon.png" alt="VoxWeave" width="200"/>

# VoxWeave

**BGM-robust subtitles for anime, film, and clips.**

Vocal separation and song-skip so ASR never hallucinates on background music, OP/ED, or
insert songs. Local-first Qwen3 ASR, forced alignment, and edit-and-resync — CJK-aware.

[![CI](https://github.com/hali0515/VoxWeave/actions/workflows/ci.yml/badge.svg)](https://github.com/hali0515/VoxWeave/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/voxweave)](https://pypi.org/project/voxweave/)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![CUDA cu128](https://img.shields.io/badge/CUDA-cu128-76B900?logo=nvidia&logoColor=white)
![Apple Silicon MLX](https://img.shields.io/badge/Apple_Silicon-MLX-000000?logo=apple&logoColor=white)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/hali0515)

<https://github.com/user-attachments/assets/e75b6dd3-fa37-4afe-89db-b6ee2c28f6bc>

<sub>Sliced clip under heavy BGM · <code>voxweave Test.mp4</code> · Qwen3-ASR-1.7B</sub>

</div>

> [!NOTE]
> **Local-first.** Separation, ASR, and forced alignment all run in-process on your GPU — no
> network endpoints, no audio leaves the machine. Runs on **NVIDIA CUDA** (PyTorch) and on
> **Apple Silicon**, where ASR + alignment use the native **MLX** Qwen3 models. Weights download
> once on first run. (Translation and ASR-correction are the only optional features that call an
> external LLM, and only when you invoke them.)

> [!NOTE]
> **Hardware.** The default pipeline (`Qwen3-ASR-0.6B`, `peak` load strategy) runs in **~8 GB of
> VRAM** — the separator is freed before ASR + alignment load, so peak ≈ `max(stage)`, not their
> sum. `--model qwen3-asr-1.7B` adds roughly **+2 GB** and still fits 8 GB under the default `peak`
> strategy. `load_strategy = "sum"` (concurrent, faster on big cards) makes peak the **sum** of the
> resident models — plan for 12 GB+. On **Apple Silicon** the MLX 8-bit weights roughly halve the
> Qwen footprint (so 1.7B fits comfortably in 16 GB unified memory). `--hybrid` loads a Whisper
> engine alongside Qwen to trade VRAM for accuracy — it does **not** save memory; the only knob that
> lowers it is staying on `0.6B` + the `peak` strategy.

VoxWeave derives from the WhisperX "edit-and-resync" workflow: transcribe once, then edit
the text and re-align it against the original audio for frame-accurate timestamps. Where it
differs is the front end — vocal separation and song-skip keep background music out of the
ASR, and a CJK-aware layout/alignment stack (MMS-300m for Japanese, BudouX/jieba for line
breaks) handles Chinese/Japanese/English as first-class.

## Contents

- [Why VoxWeave](#why-voxweave)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [Transcribe (`voxweave <media>`)](#transcribe)
  - [Re-align after editing (`align`)](#re-align-after-editing)
  - [Re-layout offline (`split`)](#re-layout-offline)
  - [ASR correction (`correct`)](#asr-correction)
  - [Translate (`translate`)](#translate)
  - [Export (`export`)](#export)
  - [Pack soft subtitles (`pack`)](#pack-soft-subtitles)
  - [Burn hard subtitles (`burn`)](#burn-hard-subtitles)
- [The edit-and-resync workflow](#the-edit-and-resync-workflow)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Data contract](#data-contract)
- [Testing](#testing)
- [Support](#support)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Why VoxWeave

- **BGM removal before ASR.** A Mel-Band Roformer vocal separator (pure torch, full-band
  44.1k) strips music first, so ASR doesn't transcribe lyrics or hallucinate on score.
- **Song-skip.** PANNs detects singing/music on the separated vocals and skips OP/ED and
  insert songs before ASR — on by default, `--no-skip-songs` to keep them.
- **Local Qwen3 ASR + forced alignment.** Text and word-level timestamps in one pass, fully
  on-device — in-process PyTorch on NVIDIA, or the native MLX Qwen3 models on Apple Silicon. A
  Whisper hybrid engine is also available for when you prefer Whisper text (faster-whisper on
  NVIDIA, the native MLX Whisper port on Apple Silicon).
- **Edit-and-resync.** Fix the transcript by hand, then `align` re-derives timestamps from
  the audio — timestamps are _never_ hand-written.
- **CJK-aware.** Japanese aligns with MMS-300m + uroman (zero-OOV, immune to the per-cue
  drift that breaks wav2vec2-xlsr on rare kanji); line breaks use BudouX phrase atoms + jieba.
- **Optional LLM steps.** `correct` cleans up ASR typos/garbled names before alignment;
  `translate` does whole-episode context-aware translation while preserving cue count
  (dual-speaker `-line`/`-line` cues are translated one speaker at a time and re-assembled).
- **Ship the result.** `pack` soft-muxes finished VTTs into the media as titled subtitle
  tracks (instant stream copy); `burn` hardcodes them at constant quality with NVENC /
  VideoToolbox acceleration, matching the source bit depth.

## Setup

Two install variants: **`voxweave[cuda]`** (NVIDIA GPU — Blackwell sm_120 / cu128 by default)
and **`voxweave[mps]`** (Apple Silicon / macOS). Both need `ffmpeg` on PATH.

<details>
<summary><b>Install ffmpeg</b></summary>

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg
# Arch Linux
sudo pacman -S ffmpeg
# macOS (Homebrew)
brew install ffmpeg
```

</details>

<details>
<summary><b>CUDA / PyTorch notes</b></summary>

On the `[cuda]` variant the torch wheel is pinned to the **cu128** build (Blackwell sm_120) and
installed into an isolated `uv` tool venv. The CUDA toolkit does **not** need to be installed
separately — the cu128 wheel bundles the required runtime libraries; only an NVIDIA driver is
required on the host. The `[mps]` variant uses the default PyPI torch wheel (Metal/MPS built in).
Override the torch index per-invocation: `make install TORCH_BACKEND=cpu`.

</details>

**Install from PyPI** (puts the global `voxweave` command on PATH):

```bash
# NVIDIA / Linux:
uv tool install --torch-backend=cu128 "voxweave[cuda]"   # full pipeline + faster-whisper hybrid
# Apple Silicon / macOS:
uv tool install "voxweave[mps]"                          # full pipeline + MLX Whisper hybrid
```

The full local pipeline — vocal separation, ASR, forced alignment (incl. MMS-300m for
Japanese/CJK), layout, song-skip — plus CJK line-break and translation are baked into the
**core dependencies**. The variant selects the compute platform **and the ASR/alignment backend**:

- `[cuda]` (NVIDIA/Linux): the in-process PyTorch Qwen3-ASR + forced aligner (`qwen-asr`), GPU
  onnxruntime (CUDAExecutionProvider for MMS alignment), and the faster-whisper hybrid engine.
- `[mps]` (Apple Silicon/macOS): **ASR** runs on the native MLX Qwen3-ASR from
  [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) (Metal kernels + quantization). **Alignment**
  keeps the same per-language stack as `[cuda]`: English on wav2vec2 CTC (torch, runs on MPS;
  the forced-align DP falls to CPU as torchaudio has no Metal kernel), Japanese/CJK on the ONNX
  MMS aligner (CoreML/CPU — onnxruntime has no Metal provider). Only the Qwen fallback (zh·yue,
  or any CTC failure) is served by the MLX Qwen3-ForcedAligner, since the torch `qwen-asr` aligner
  is absent here. The Whisper hybrid/fusion engines (`--model large-v3`, `--hybrid`) run on the
  native [`mlx-whisper`](https://pypi.org/project/mlx-whisper/) Metal port instead of faster-whisper
  (ctranslate2 has no Metal backend). Vocal separation (MelBandRoformer) + PANNs song-skip stay on
  torch-MPS. `qwen-asr` is excluded because its `transformers==4.57.6` pin conflicts with mlx-audio,
  so `[cuda]` and `[mps]` are mutually exclusive — pick one per host.

**From source** (for development or pulling new code):

```bash
make install       # auto-detects the platform: Apple Silicon -> [mps], anything else -> [cuda]
                   # (cu128 torch wheel with an NVIDIA driver, CPU wheel without); includes
                   # [diarize] by default and keeps whatever extras the existing venv already has
make reinstall     # after pulling new code (same auto-detection, preserves installed extras)
make uninstall
```

Override the detection per invocation: `make install VARIANT=mps`, `make install EXTRAS=`
(no extras), `make install TORCH_BACKEND=cpu`.

<details>
<summary><b>Extras & what each pulls</b></summary>

- The core pulls `qwen-asr` (hard-pins `transformers==4.57.6` + `accelerate==1.12.0`) + a
  pure-torch Mel-Band Roformer vendored in `voxweave.vendor` (**no onnx/onnxruntime** —
  `audio-separator` is intentionally avoided because it eagerly imports onnxruntime at the
  top level) + MMS-300m forced aligner (`ctc-forced-aligner`) + layout (`pysbd`) + song-skip
  (`panns-inference`) + CJK break (`budoux` + `jieba`) + translation (`openai`).
- **`[cuda]`** (NVIDIA/Linux): `qwen-asr` + `onnxruntime-gpu` + `faster-whisper`. **`[mps]`**
  (Apple Silicon/macOS): `mlx-audio` + plain `onnxruntime`. Declared **conflicting** in
  `[tool.uv]` (incompatible `transformers` pins), so `uv` resolves each in its own fork — pick one
  per host (`make dev VARIANT=mps` on Apple Silicon).
- **`[diarize]`** (stackable on either variant): `pyannote-audio` 3.x for `--diarize`. The
  `pyannote/speaker-diarization-3.1` checkpoint is HF-gated — accept its model-card conditions
  (and segmentation-3.0's) once, then any stored token works (`hf auth login` is enough).
  Stay on pyannote 3.x: 4.x peaks ~9.5 GB VRAM on the same model (pyannote-audio#1963).
- The device is auto-detected at runtime (cuda → mps → cpu); override with `VOXWEAVE_DEVICE`. On
  mps the MLX backend is selected automatically; force it either way with `VOXWEAVE_BACKEND=mlx|torch`.
- **Development**: `make dev` (= `uv sync --extra cuda --dev`; on Apple Silicon use
  `make dev VARIANT=mps` — `[cuda]`/`[mps]` are conflicting extras and can't be synced together).

</details>

## Quickstart

```bash
# Transcribe a video to a timestamped VTT (+ a JSON source of truth)
voxweave episode.mkv

# ...edit episode.vtt by hand (fix wording, line breaks)...

# Re-align the edited text against the original audio
voxweave align episode.vtt

# Optionally translate the aligned subtitles to Chinese
voxweave translate episode.vtt --to zh
```

## Usage

### Transcribe

`voxweave <media>` — separation → song-skip → VAD chunking → ASR + forced alignment →
smart_split → writes `<stem>.vtt` (editable) + `<stem>.json` (word-level timestamp source of
truth). Models load in-process (see `voxweave.backend`); the separator is released from VRAM
before ASR+alignment load, so peak usage is ≈ max(sep, asr) rather than their sum.

```bash
voxweave episode.mkv
voxweave clip.mp4 --no-separate          # clean speech (podcast/lecture): skip separation
voxweave episode.mkv --model qwen3-asr-1.7B   # larger, more accurate ASR
voxweave episode.mkv --context "Ryland Grace, Astrophage, Hail Mary"   # bias names/terms
```

<details>
<summary><b>Options</b></summary>

| Option                         | Description                                                                                                                                                                                                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--language`                   | Force language (ISO code or full name); default auto-detect.                                                                                                                                                                                                                                             |
| `--no-separate`                | Skip vocal separation (for clean speech) to save GPU time.                                                                                                                                                                                                                                               |
| `--no-skip-songs`              | Keep lyrics / transcribe purely musical content (song-skip is on by default).                                                                                                                                                                                                                            |
| `--model`                      | Local ASR model (default `Qwen3-ASR-0.6B`; `qwen3-asr-1.7B` is more accurate).                                                                                                                                                                                                                           |
| `--context`                    | ASR bias prompt: names/terms likely to appear (comma or newline separated). Bare term lists are auto-framed as `Proper nouns: ...` for Qwen — a bare list actually _regresses_ accuracy ([details](https://github.com/TypeWhisper/typewhisper-mac/issues/321)); prose or pre-framed text passes through. |
| `--hybrid`                     | Dual-ASR fusion: Whisper text + Qwen punctuation. Whisper's error bias is the opposite of Qwen's (it hallucinates rather than omits), so use this when Qwen drops uncertain words.                                                                                                                       |
| `--normalize/--no-normalize`   | Apply loudness normalization (`loudnorm`) to the 16k ASR input — helps when quiet words get dropped; off by default since it also amplifies noise.                                                                                                                                                       |
| `--timestamps/--no-timestamps` | VTT carries word-level timestamps (default on); `--no-timestamps` writes a plain-text editing draft.                                                                                                                                                                                                     |
| `--keep-lyrics`                | Transcribe detected songs instead of skipping them; sung cues are wrapped `♪ ... ♪` (italic in ASS export).                                                                                                                                                                                              |
| `--sdh`                        | Also write `<stem>.sdh.vtt`: PANNs non-speech event tags (`[explosion]`, `[phone ringing]`, ...) in speech-free gaps.                                                                                                                                                                                    |
| `--diarize`                    | pyannote speaker diarization: multi-speaker cues split at speaker boundaries; on two-line languages a short exchange becomes a Netflix dual-speaker event (`-line` per speaker). Needs `voxweave[diarize]` + an HF token for the gated checkpoint (`VOXWEAVE_HF_TOKEN` / `HF_TOKEN` / conf `hf_token`, or just `hf auth login` once). Speaker turns persist to the sibling JSON, so `voxweave split` replays the formatting without re-running the model. |
| `--min-speakers` / `--max-speakers` | Bound the diarizer's speaker count when you know it (e.g. `--max-speakers 2` for an interview) — the single best lever against over-splitting on noisy material.                                                                                                       |
| `--no-shot-snap`               | Disable shot-change detection/snapping (cue boundaries otherwise land on cuts per the Netflix zone rules).                                                                                                                                                                                               |
| `--vad-mask/--no-vad-mask`     | Suppress CTC emissions outside speech spans during alignment so words cannot park in music/silence (recommended for sparse-dialogue movies with songs; keep off when VAD may misjudge sung/whispered speech). Same as `VOXWEAVE_VAD_EMISSION_MASK=1`.                                                    |
| `--debug`                      | Write intermediate artifacts (full-band / vocals / per-chunk VAD + ASR + alignment) to `debug/<stem>/`.                                                                                                                                                                                                  |

The boolean flags (`--separate`, `--skip-songs`, `--normalize`, `--diarize`, `--timestamps`,
`--shot-snap`, `--vad-mask`) can have their defaults set persistently via the `[defaults]`
section of `~/.config/voxweave.conf` — an explicit CLI flag always wins for that run.

</details>

### Re-align after editing

`voxweave align <vtt>` — takes the edited VTT text and **re-runs forced alignment against the
original audio**, overwriting the timestamped VTT and updating the JSON. Does not re-run ASR
or touch smart_split. Aligns on separated 16k vocals by default (prevents BGM interference);
prefers a cached `cache/<stem>.16k.flac`, otherwise re-separates and caches.

```bash
voxweave align episode.vtt                 # finds episode.<ext> in the same dir
voxweave align episode.vtt --media original.mkv
voxweave align episode.vtt --no-separate   # align on the original audio (clean sources)
```

<details>
<summary><b>Options</b></summary>

| Option          | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| `--media`       | Source media path (default: same-name file in the same directory). |
| `--language`    | Force language (ISO code or full name); default: read from JSON.   |
| `--no-separate` | Align on the original audio instead of separated vocals.           |
| `--normalize/--no-normalize` | Apply `loudnorm` to the 16k alignment input.          |
| `--vad-mask/--no-vad-mask`   | Suppress CTC emissions outside the JSON's `vad_speech` spans (see the transcribe option of the same name). |

`--separate`, `--normalize`, and `--vad-mask` also honor the `[defaults]` section of
`~/.config/voxweave.conf` when not passed explicitly.

</details>

### Re-layout offline

`voxweave split <json>` — re-run smart_split from `<stem>.json` **without any models** (adjust
line width / sentence breaks instantly).

```bash
voxweave split episode.json --max-line-length 14 --max-lines 1
voxweave split episode.json --no-timestamps   # plain-text editing draft
```

### ASR correction

`voxweave correct <vtt>` — optional **pre-align** LLM pass that fixes obvious ASR typos, split
words, and garbled proper nouns, producing a reviewable diff. Conservative substitution only
(no completion/rewrite), gated by a code check that the matched text equals the original
line-for-line. By default writes only a sidecar `<stem>.asrfix.vtt` + audit JSON — the
original VTT is untouched. Use `--apply` to overwrite, **then run `align`** to reassign timing.

```bash
voxweave correct episode.vtt --glossary names.json   # review the sidecar
voxweave correct episode.vtt --glossary names.json --apply
voxweave align episode.vtt
```

<details>
<summary><b>Options</b></summary>

| Option                         | Description                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `--glossary`                   | Term/name glossary (`.json` → mapping; other → raw prompt). Strongly recommended for ambiguous proper nouns. |
| `--apply`                      | Overwrite the original VTT (default: sidecar only, for review).                                              |
| `--model`                      | Correction model (default `VOXWEAVE_FIX_MODEL` env or `gpt-5.3-chat-latest`).                                |
| `--base-url` / `--api-key-env` | OpenAI-compatible endpoint + which env var holds the key.                                                    |

</details>

### Translate

`voxweave translate <subtitle>` — **after align**, translate each cue with whole-episode
context, preserving cue count, into `<stem>.<to>.<ext>` (the original is left unchanged).
Accepts `.vtt`/`.srt`/`.ass`/`.ssa`; the output mirrors the input format
(`episode.srt` → `episode.zh.srt`).

```bash
voxweave translate episode.vtt --to zh
voxweave translate episode.vtt --to en --context "sci-fi, formal register" --glossary terms.json
voxweave translate downloaded.srt --to zh               # foreign SRT in, SRT out
```

<details>
<summary><b>Options</b></summary>

| Option                         | Description                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| `--to`                         | Target language code, written to `<stem>.<to>.<ext>` (default `zh`).                 |
| `--context`                    | Show/tone context injected into the prompt.                                          |
| `--glossary`                   | Term/name glossary (`.json` → mapping; other → raw prompt).                          |
| `--model`                      | Translation model (default `VOXWEAVE_TRANSLATE_MODEL` env or `gpt-5.3-chat-latest`). |
| `--base-url` / `--api-key-env` | OpenAI-compatible endpoint + which env var holds the key.                            |

</details>

### Export

`voxweave export <subtitle>` — convert between subtitle formats: VTT/SRT/ASS/SSA in,
SRT/ASS/VTT out (written next to the input; the VTT + JSON pair stays the source of truth
for voxweave-produced subtitles). ASS output carries a Default style; lyric cues (`♪ ... ♪`)
render italic. Foreign SRT/ASS files can be exported to VTT to enter the editing workflow.

```bash
voxweave export episode.vtt --to srt
voxweave export episode.vtt --to srt --to ass
voxweave export downloaded.ass --to vtt        # foreign ASS -> VTT for editing/translate
```

### Pack (soft subtitles)

`voxweave pack <subtitle>...` — remux the source media with the subtitle file(s)
(VTT/SRT/ASS) added as proper subtitle tracks. Pure stream copy (instant, lossless,
reversible); each track is titled `VoxWeave <Language>` with the container language tag
taken from the filename (`episode.zh.vtt` → `chi` / "VoxWeave Chinese"), and the first
packed track is flagged default so players select it. ASS inputs keep their styling in
mkv targets (mp4/webm store text-only codecs, so styling is dropped there).

```bash
voxweave pack episode.zh.vtt                    # finds episode.<ext>, keeps its container
voxweave pack episode.zh.vtt episode.ja.vtt     # several tracks at once
voxweave pack episode.zh.vtt --to mp4           # mov_text in mp4 (image subs are dropped)
voxweave pack episode.zh.vtt --media other.mkv -o out.mkv
```

mkv targets keep every source stream (including attachments); mp4/webm targets keep
video+audio and existing _text_ subtitle tracks only. HEVC video muxed into mp4 is tagged
`hvc1` for Apple players.

### Burn (hard subtitles)

`voxweave burn <subtitle>` — render the subtitles (VTT/SRT/ASS) into the pixels and write a
clean file with **all subtitle tracks removed**. For VTT/SRT input a styled ASS is generated
at the actual frame size (same look as `export`, lyric cues italic); ASS/SSA input goes to
libass as-is, keeping its own styling. The video is re-encoded at constant quality with
hardware acceleration when available: **NVENC** on NVIDIA, **VideoToolbox** on macOS,
libx264/libx265/libsvt-av1 software fallback. Audio is stream-copied (mp4 targets re-encode
mp4-incompatible codecs to AAC).

```bash
voxweave burn episode.zh.vtt                          # hevc, auto hw encoder, -> episode.mp4
voxweave burn episode.zh.vtt --codec h264             # legacy-device compatibility
voxweave burn episode.zh.vtt --codec av1 --to mkv     # max compression, recent hardware
voxweave burn episode.zh.vtt --quality 20 --font "Noto Sans CJK SC"
```

<details>
<summary>Burn options & encoding policy</summary>

| Option        | Meaning                                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------------------- |
| `--codec`     | `hevc` (default: 10-bit capable, ~40% smaller than h264, plays everywhere as `hvc1` mp4) / `h264` / `av1`. |
| `--encoder`   | Force a specific ffmpeg encoder (default: auto-probe with a test encode).                                  |
| `--quality`   | Constant quality: NVENC `-cq` / software `-crf` (lower = better); VideoToolbox `-q:v` (higher = better).   |
| `--to`        | `mp4` (default, maximum compatibility) or `mkv`.                                                           |
| `--font`      | Subtitle font family (fontconfig resolves fallbacks; e.g. `Noto Sans CJK SC`).                             |
| `--font-size` | Override the default 72-at-1080p scaled size.                                                              |

Bitrate is never targeted: pure constant-quality (`-b:v 0` on NVENC) lets the encoder spend
bits where the content needs them, with no overshoot against the source rate. Output bit
depth follows the source dynamically (8-bit stays 8-bit, 10-bit stays 10-bit; 12-bit is kept
on libx265 and clamped to 10 on NVENC/VideoToolbox/SVT-AV1, which top out there) — except on
h264 paths, which are always 8-bit for player compatibility (NVENC h264 cannot encode 10-bit
at all).

</details>

Progress is rendered with rich: countable stages (demix windows / PANNs batches / per-chunk
ASR+alignment / align per-cue / translate streaming per-line) show a real `x/N` bar with
elapsed time; indeterminate stages (decode / file write) show a pulse bar. `-v/--verbose`
enables DEBUG logging.

## The edit-and-resync workflow

```
voxweave episode.mkv          # 1. transcribe  -> episode.vtt + episode.json
  └─ (optional) correct       # 2. LLM ASR fix -> episode.asrfix.vtt (--apply to commit)
edit episode.vtt by hand      # 3. fix wording / line breaks
voxweave align episode.vtt    # 4. re-derive timestamps from audio (overwrites VTT + JSON)
voxweave translate episode.vtt --to zh   # 5. context-aware translation
voxweave pack episode.zh.vtt             # 6. soft-mux into the media (or burn for hardsubs)
```

Timestamps are **always** derived from the audio by the forced aligner — you never hand-edit
them. Edit the text freely; `align` puts the timing back.

## How it works

| Stage           | What runs                                                                                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Separation**  | Mel-Band Roformer (full-band 44.1k stereo, vendored pure-torch) isolates vocals; downsampled to 16k afterwards.                                                  |
| **Song-skip**   | PANNs (route ii) flags singing/music on the separated vocals before ASR; songs are excised mid-segment with cuts snapped into real silences, and PANNs clean-speech evidence rescues dialogue the waveform VAD under-scores. |
| **Chunking**    | Silero VAD splits speech into ≤120s chunks (longer risks ASR repetition-loop collapse).                                                                          |
| **ASR + align** | Qwen3-ASR (default, text + units in one pass) / Whisper hybrid (faster-whisper on cuda, mlx-whisper on mps) / dual-ASR fusion — the pipeline is engine-agnostic. |
| **Alignment**   | `ja` → MMS-300m + uroman, `en` → wav2vec2-LV60K CTC (both full-file single pass, WhisperX-gold); `zh`·`yue` → Qwen. During transcribe the pass is cropped to the transcribed envelope and excised songs are muted, so untranscribed music can never host stretched words. |
| **Layout**      | gap-aware `smart_split`: word-level gaps + BudouX phrase atoms + line-length, on a shared timeline forked per language.                                          |

## Configuration

Precedence: **CLI flag > env var > `~/.config/voxweave.conf` > built-in default.** A commented
default config is written on first run (migrated automatically from a pre-rename `qsub.conf`).

<details>
<summary><b>Environment variables</b></summary>

**Models**

- `VOXWEAVE_ASR_MODEL` (default `Qwen/Qwen3-ASR-0.6B`; same as `--model`)
- `VOXWEAVE_ALIGNER_MODEL` (default `Qwen/Qwen3-ForcedAligner-0.6B`)
- `VOXWEAVE_DEVICE` (default: auto-detect `cuda:0` → `mps` → `cpu`)
- `VOXWEAVE_BACKEND` (`mlx` | `torch`; default: `mlx` on mps, else `torch`) — picks the ASR/alignment backend
- `VOXWEAVE_OFFLINE` (`1` to enable) — once all models are cached, sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` so loading skips the per-file HEAD revalidation + optional-file probing huggingface_hub/transformers otherwise do on every run (no network on a cache hit). Leave off for the first download.
- `VOXWEAVE_MLX_ASR_REPO` / `VOXWEAVE_MLX_ALIGNER_REPO` / `VOXWEAVE_MLX_WHISPER_REPO` — MLX backend
  repos. By default the ASR repo tracks `--model` size (`--model 1.7b` → `mlx-community/Qwen3-ASR-1.7B-8bit`)
  and the Whisper repo tracks the Whisper size (`--model large-v3` → `mlx-community/whisper-large-v3-mlx`);
  set the matching var to hard-pin a specific quant (e.g. a 4-bit build) regardless of `--model`.

All model weights (torch + MLX) are cached under `~/.cache/voxweave/{asr,align,audio}`
(auto-downloaded on first use; override the root with `VOXWEAVE_CACHE_ROOT`), so a container only
needs to bind-mount that one directory. Each model exposes an env override to swap the HF repo, or
to point at an explicit local file (which, if it exists, skips the HF download):

- `VOXWEAVE_SEPARATOR_REPO` / `VOXWEAVE_SEPARATOR_REPO_FILE` (default `KimberleyJSN/melbandroformer` /
  `MelBandRoformer.ckpt`), or `VOXWEAVE_SEPARATOR_CKPT` / `VOXWEAVE_SEPARATOR_CONFIG` for explicit
  weights + matching yaml
- `VOXWEAVE_PANNS_REPO` / `VOXWEAVE_PANNS_REPO_FILE` (default `thelou1s/panns-inference` /
  `Cnn14_mAP=0.431.pth`), or `VOXWEAVE_PANNS_CKPT` for an explicit checkpoint (song-skip CNN)
- `VOXWEAVE_MMS_REPO` / `VOXWEAVE_MMS_REPO_FILE` (default `deskpai/ctc_forced_aligner` /
  `04ac86b67129634da93aea76e0147ef3.onnx`), or `VOXWEAVE_MMS_MODEL` for an explicit onnx path
  (Japanese/CJK MMS-300m aligner)

**Tuning**

- `VOXWEAVE_MAX_CHUNK_SEC` (default 120; shorter chunks reduce ASR repetition loops on long segments)
- `VOXWEAVE_LOUDNORM` (default `loudnorm=I=-16:TP=-1.5:LRA=11`; the `-af` filter for `--normalize`)
- `VOXWEAVE_MIN_CUE_SEC` (default 0.8; minimum cue display duration in `align`)
- `VOXWEAVE_SNAP_VAD_THRESHOLD` (default 0.25; sensitive VAD used when repositioning
  zero-duration units against the original audio)
- `VOXWEAVE_SONG_CORE_MERGE_SEC` (default 15; song spans within this gap of a long OP/ED
  cluster into one song "core" that stops the dialogue edge trim — an isolated brief sting
  farther away is trimmed through instead of anchoring dialogue into the excised song)
- `VOXWEAVE_SPEECH_RESCUE_MIN_S` (default 3; minimum length of a PANNs clean-dialogue
  stretch with no silero coverage to be rescued into the chunk stream — catches dialogue
  silero under-scores, e.g. theatrical delivery)
- `VOXWEAVE_CTC_ENVELOPE_PAD_SEC` (default 2; lead-in/out pad when the full-file alignment
  pass is cropped to the transcribed chunk envelope during transcribe, keeping a skipped
  leading/trailing song out of the aligner's waveform)

</details>

<details>
<summary><b>Config file (<code>~/.config/voxweave.conf</code>, TOML)</b></summary>

Every key below is optional — delete a line to fall back to its built-in default. The values
shown are a usable starting point, not the defaults (the auto-written template has everything
commented out).

```toml
# ~/.config/voxweave.conf  —  TOML
# Precedence: CLI flag > env var > this file > built-in default.

# Default ASR model (= --model). Short name (qwen3-asr-0.6b | qwen3-asr-1.7b) or full HF id.
# Special value "hybrid" (= --hybrid) -> dual-ASR fusion (whisper text + Qwen punctuation).
asr_model = "Qwen/Qwen3-ASR-1.7B"        # built-in default: Qwen/Qwen3-ASR-0.6B

# Model load strategy:
#   "peak" (default) — serial peak-shaving: all-chunk ASR -> release -> all-chunk align;
#                      ASR and aligner never co-reside, peak VRAM = max(models). Works on 8 GB.
#   "sum"            — concurrent per-chunk ASR+align; peak VRAM = sum(models), but skips two
#                      model swap round-trips (faster on large-VRAM cards).
load_strategy = "sum"

# Inference batch sizes: windows per GPU forward (env: VOXWEAVE_SEP_BATCH / VOXWEAVE_CTC_BATCH /
# VOXWEAVE_MMS_BATCH). On an 8 GB-class card batch=1 already saturates compute — measured no
# speedup at 2/4, just ~+0.8 GiB VRAM per extra separation window — so the defaults stay at 1.
# Only worth raising on much wider GPUs, and only after measuring.
[batch]
separate = 1                             # vocal separation (MelBandRoformer) 8s windows
ctc      = 1                             # wav2vec2 CTC emission 30s windows (en aligner)
mms      = 4                             # MMS-300m emission batch (ja aligner)

# Default on/off for the boolean pipeline flags. An explicit CLI flag always wins
# (e.g. separate = false here, --separate on the command line for one run).
[defaults]
separate   = true                        # vocal separation before ASR/alignment (--separate/--no-separate)
skip_songs = true                        # PANNs music detection + skip before ASR (--skip-songs/--no-skip-songs)
normalize  = false                       # loudnorm on the 16k input (--normalize/--no-normalize)
diarize    = false                       # pyannote speaker diarization (--diarize/--no-diarize; needs voxweave[diarize] + HF token)
timestamps = true                        # word-level timestamps in the VTT (--timestamps/--no-timestamps)
shot_snap  = true                        # snap cue boundaries onto shot changes (--shot-snap/--no-shot-snap)
vad_mask   = false                       # suppress CTC emissions outside speech (--vad-mask/--no-vad-mask)

# dual-ASR fusion sub-models — only consulted when running with --hybrid.
[fusion]
whisper = "large-v3-turbo"               # Whisper size: large-v3 (best) | large-v3-turbo (~5x faster); faster-whisper on cuda, mlx-whisper on mps
qwen    = "Qwen/Qwen3-ASR-1.7B"          # punctuation model; must emit punctuation -> 1.7B, not 0.6B

# Per-language forced-alignment model. Key = ISO-639-1 code; unlisted languages use Qwen3-ForcedAligner.
# Values:
#   "mms"   — MMS-300m + uroman, full-file single pass (immune to per-cue drift; the gold standard).
#   HF id   — wav2vec2 CTC via HF transformers; weights land in ~/.cache/voxweave/align (per-cue crop).
#   bundle  — torchaudio bundle name, e.g. "WAV2VEC2_ASR_LARGE_LV60K_960H" (same model, cached in ~/.cache/torch).
#   ""      — explicitly fall back to Qwen for that language.
[align]
en = "facebook/wav2vec2-large-960h-lv60-self"  # English: LV60K-self CTC, per-cue crop (HF hub)
ja = "mms"                                      # Japanese: MMS-300m + uroman full-file (= whisperx fork align_ctc)
# zh  = "mms"                                   # Chinese can also use MMS; default is Qwen (native CJK char-level)
# yue = ""                                      # force Qwen for Cantonese
```

</details>

## Data contract

Each input produces two sibling files:

- **`<stem>.json`** — the source of truth: word/character-level segments, language, VAD speech,
  plus optional replay data (`shot_changes`, `sing_spans`, `speaker_turns`) so `split` can
  redo shot snapping, lyric flagging, and speaker formatting without re-running any model.
- **`<stem>.vtt`** — editable subtitles. By default cues carry word-level timestamps (same
  precision as `align` output, ready to use); `--no-timestamps` writes a plain-text editing
  draft for hand-correction, which `align` re-times.

Both VTT forms are accepted by `align`. The aligner strips punctuation as a hard constraint;
ASR punctuation is re-injected by time so the final output has correct spacing and breaks
without stray marks.

## Testing

- Unit tests (models mocked, no network): `make test` (= `uv run pytest tests/`)
- Lint / format: `make lint`

## Support

If VoxWeave saves you time, you can support development here:

<a href="https://buymeacoffee.com/hali0515"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me A Coffee"/></a>

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) — the forced-alignment + edit-and-resync
  workflow this project builds on; the Japanese MMS full-file alignment path is a faithful
  port of its `ctc` align backend.
- [stable-ts](https://github.com/jianfch/stable-ts) — inspiration for timestamp post-processing
  and documentation structure.
- [Qwen3-ASR / Qwen3-ForcedAligner](https://github.com/QwenLM) (Alibaba) — local ASR + aligner.
- [MMS-300m](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) (Meta) via
  [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) — zero-OOV CJK alignment.
- [Mel-Band Roformer](https://github.com/lucidrains/BS-RoFormer) (lucidrains) +
  [KimberleyJSN](https://huggingface.co/KimberleyJSN/melbandroformer) weights — vocal separation.
- [BudouX](https://github.com/google/budoux), [jieba](https://github.com/fxsjy/jieba),
  [PySBD](https://github.com/nipunsadvilkar/pySBD) — CJK/sentence line-break.
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) — song/music detection.
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection.
