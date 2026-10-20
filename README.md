<div align="center">

<img src="https://raw.githubusercontent.com/hali0515/VoxWeave/main/resources/VoxWeave_icon.png" alt="VoxWeave" width="200"/>

# VoxWeave

**BGM-robust subtitles for anime, film, and clips.**

Vocal separation and song-skip so ASR never hallucinates on background music, OP/ED, or
insert songs. Local-first Qwen3 ASR, word-level forced alignment, speaker labels, and
edit-and-resync — CJK-aware.

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
> **Apple Silicon**, where ASR and the Qwen aligner use the native **MLX** Qwen3 models. Weights
> download once on first run. (Translation and ASR-correction are the only optional features
> that call an external LLM, and only when you invoke them.)

> [!NOTE]
> **Hardware.** The default pipeline runs in about 8 GB of VRAM on NVIDIA, or in 16 GB of
> unified memory on Apple Silicon. Details: [Setup](https://github.com/hali0515/VoxWeave/blob/main/docs/install.md#hardware).

VoxWeave derives from the WhisperX "edit-and-resync" workflow: transcribe once, then edit
the text and re-align it against the original audio for word-level timestamps. Where it
differs is the front end — vocal separation and song-skip keep background music out of the
ASR, and a CJK-aware layout/alignment stack (MMS-300m for Japanese, BudouX/jieba for line
breaks) handles Chinese/Japanese/English as first-class.

## Why VoxWeave

- **BGM removal before ASR.** A Mel-Band Roformer vocal separator (pure torch, full-band
  44.1k) strips music first, so ASR doesn't transcribe lyrics or hallucinate on score.
- **Song-skip.** PANNs detects singing/music on the separated vocals and skips OP/ED and
  insert songs before ASR — on by default, `--no-skip-songs` to keep them.
- **Local Qwen3 ASR + forced alignment.** Text and word-level timestamps in one pass, fully
  on-device — in-process PyTorch on NVIDIA, or the native MLX Qwen3 models on Apple Silicon. A
  Whisper hybrid engine is also available for when you prefer Whisper text (faster-whisper on
  NVIDIA, the native MLX Whisper port on Apple Silicon).
- **Word-level timestamps.** Each word gets its own start and end from CTC forced alignment
  against the audio (MMS-300m for Japanese, wav2vec2 for English, Qwen3-ForcedAligner for
  Chinese), not from ASR segment times. Cue edges then snap to shot changes under Netflix
  timing rules.
- **Speaker labels.** `--diarize` splits cues at speaker changes and sets short two-person
  exchanges as Netflix `-line` pairs. `voxweave speakers serve` opens a page where you audition
  each speaker, name them, and split two people that were merged. Enrolled voices go into one
  library, so names carry over to later episodes. Opt-in; needs a Hugging Face token for the
  gated pyannote model (see [Setup](https://github.com/hali0515/VoxWeave/blob/main/docs/install.md#setup)).
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

### Pipeline at a glance

<p align="center">
  <img src="https://raw.githubusercontent.com/hali0515/VoxWeave/main/resources/pipeline.svg" alt="Media goes through vocal separation, song-skip, VAD chunking, ASR, forced alignment and cue splitting into a VTT and JSON pair; speaker diarization is an optional branch that splits cues at speaker changes" width="100%"/>
</p>

<sub>Fig. 1. Processing pipeline. Solid lines carry the audio and text; the dashed box and lines
are the optional `--diarize` branch. Every stage runs locally; see
[How it works](https://github.com/hali0515/VoxWeave/blob/main/docs/internals.md#how-it-works).</sub>

## Install

```bash
# NVIDIA / Linux
uv tool install --torch-backend=cu128 \
  --overrides <(printf "onnxruntime; sys_platform == 'darwin'\n") \
  "voxweave[cuda]"
# Apple Silicon / macOS
uv tool install "voxweave[mps]"
```

From a clone, `make install` does the same and picks the variant, the torch build and the
overrides for you (`make reinstall` after pulling new code).

Both variants need `ffmpeg` on PATH. The default pipeline fits in about 8 GB of VRAM.
`--diarize` also needs a Hugging Face token for a gated model; see
[Speaker models](https://github.com/hali0515/VoxWeave/blob/main/docs/speakers.md#speaker-models). Everything else about installing
(variants, CUDA notes, overrides, dependencies) is in [Setup](https://github.com/hali0515/VoxWeave/blob/main/docs/install.md#setup).

## Quickstart

```bash
# Transcribe a video to a timestamped VTT (+ a JSON source of truth)
voxweave episode.mkv

# Opt in to speaker diarization (requires gated-model access; see Setup)
voxweave interview.mkv --diarize

# Stay on the older 3.1 pipeline if that is the gate you have accepted
voxweave interview.mkv --diarize --diarize-model 3.1

# ...edit episode.vtt by hand (fix wording, line breaks)...

# Re-align the edited text against the original audio
voxweave align episode.vtt

# Optionally translate the aligned subtitles to Chinese
voxweave translate episode.vtt --target zh
```

### Recipes

**A meeting or interview recording**, with named speakers in the subtitles. Audio files work
as well as video. When the background is clean, `--no-separate` skips the vocal separator.
Bound the speaker count if you know it: `--max-speakers` is the best lever against one person
being split into two (see [Transcribe](https://github.com/hali0515/VoxWeave/blob/main/docs/commands.md#transcribe)).

```bash
voxweave meeting.m4a --no-separate --diarize --max-speakers 4

# Listen to each speaker, type a name, press Save.
voxweave speakers serve meeting.m4a
```

<p align="center">
  <img src="https://raw.githubusercontent.com/hali0515/VoxWeave/main/resources/speakers-flow.svg" alt="Diarize the recording, serve the speaker page, name each speaker and press Save to re-render the VTT with voice tags; optionally enroll the names into the voice library so the next recording suggests them" width="100%"/>
</p>

<sub>Fig. 2. Naming speakers. Each Save re-renders the VTT; the dashed boxes are the optional
voice library, which suggests the names again in later recordings.</sub>

Each Save re-renders `meeting.vtt` with the names as `<v Name>` voice tags. If two people
share one speaker card, use **Split this speaker**; it needs `--voiceprints` on the first run
(see [Label speakers](https://github.com/hali0515/VoxWeave/blob/main/docs/speakers.md#label-speakers)).

**A recurring group** (a weekly meeting, a show with a fixed cast). Name people once, and
later recordings suggest the names:

```bash
voxweave week1.m4a --no-separate --diarize --voiceprints
voxweave speakers serve week1.m4a      # name and Save
voxweave speakers enroll week1.m4a     # store the names in the voice library

voxweave week2.m4a --no-separate --diarize --voiceprints
voxweave speakers serve week2.m4a      # names are suggested; check them, then Save
```

**Anime or film with music.** The default pipeline separates vocals and skips OP/ED and
insert songs, so no flags are needed:

```bash
voxweave episode.mkv
voxweave episode.mkv --diarize         # add speaker labels
```

**Fix the text, keep the timing.** Edit `episode.vtt` by hand, then re-align:

```bash
voxweave align episode.vtt
```

**Ship the result:**

```bash
voxweave translate episode.vtt --target zh
voxweave pack episode.zh.vtt           # soft track, stream copy
voxweave burn episode.zh.vtt           # hard subtitles
```

## Commands

| Command                                                 | What it does                                                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `voxweave <media>`                                      | Transcribe: separate vocals, skip songs, ASR, align, split into cues. Writes `<stem>.vtt` and `<stem>.json`. |
| `voxweave speakers serve <media>`                       | Open the page where you name the speakers; each Save re-renders the VTT.                                     |
| `voxweave speakers enroll <media>`                      | Store the named voices in the shared voice library.                                                          |
| `voxweave align <vtt>`                                  | Re-derive timestamps from the audio after you edit the text.                                                 |
| `voxweave render <media>`                               | Re-split cues from the saved JSON, without any model.                                                        |
| `voxweave correct <vtt>`                                | Optional LLM pass that fixes obvious ASR errors before alignment.                                            |
| `voxweave translate <subtitle> --target zh`             | Whole-episode translation; the cue count stays the same.                                                     |
| `voxweave export <subtitle> --to srt`                   | Convert between VTT, SRT and ASS.                                                                            |
| `voxweave pack <subtitle>` / `voxweave burn <subtitle>` | Add subtitles to the media as tracks, or burn them into the picture.                                         |

Options and details: [Commands](https://github.com/hali0515/VoxWeave/blob/main/docs/commands.md).

## Documentation

| Page                                                                                  | Covers                                                               |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [Setup](https://github.com/hali0515/VoxWeave/blob/main/docs/install.md)               | Install variants, CUDA and Apple Silicon notes, dependencies         |
| [Commands](https://github.com/hali0515/VoxWeave/blob/main/docs/commands.md)           | Every command with its options                                       |
| [Speakers](https://github.com/hali0515/VoxWeave/blob/main/docs/speakers.md)           | Diarization models, voiceprints, the speaker page, the voice library |
| [How it works](https://github.com/hali0515/VoxWeave/blob/main/docs/internals.md)      | The edit-and-resync workflow and the pipeline stages                 |
| [Configuration](https://github.com/hali0515/VoxWeave/blob/main/docs/configuration.md) | `voxweave.conf`, environment variables, performance knobs            |
| [Data contract](https://github.com/hali0515/VoxWeave/blob/main/docs/data-contract.md) | Files written next to the media, JSON schema, tests                  |
| [Migration notes](https://github.com/hali0515/VoxWeave/blob/main/MIGRATING.md)        | Changes between releases                                             |

## Support

If VoxWeave saves you time, you can support development here:

<a href="https://buymeacoffee.com/hali0515"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me A Coffee"/></a>

## License

MIT — see [LICENSE](https://github.com/hali0515/VoxWeave/blob/main/LICENSE).

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
- [ReDimNet2](https://github.com/PalabraAI/redimnet2) (Palabra.ai) — the default voiceprint
  embedder; [SpeechBrain](https://github.com/speechbrain/speechbrain) ECAPA-TDNN with the
  [anime voice-actor weights](https://huggingface.co/litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm)
  (litagin) — the Japanese one.
