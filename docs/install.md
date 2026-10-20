# Setup

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
uv tool install --torch-backend=cu128 \
  --overrides <(printf "onnxruntime; sys_platform == 'darwin'\n") \
  "voxweave[cuda]"                                       # full pipeline + faster-whisper hybrid
# Apple Silicon / macOS:
uv tool install "voxweave[mps]"                          # full pipeline + MLX Whisper hybrid
```

`--torch-backend` in `uv tool install` needs uv 0.9.19 or newer (`uv self update`). On Linux the
`--overrides` line keeps the CPU `onnxruntime` wheel (pulled in by ctc-forced-aligner) from
shadowing `onnxruntime-gpu`, which would silently move Japanese MMS alignment onto the CPU; the
Makefile passes the same rule from `overrides.txt`.

The full local pipeline — vocal separation, ASR, forced alignment (incl. MMS-300m for
Japanese/CJK), layout, song-skip, and speaker-diarization support — plus CJK line-break and
translation — is baked into the **core dependencies**. The variant selects the compute platform
**and the ASR/alignment backend**:

- `[cuda]` (NVIDIA/Linux): the in-process PyTorch Qwen3-ASR + forced aligner (`qwen-asr`), GPU
  onnxruntime (CUDAExecutionProvider for MMS alignment), and the faster-whisper hybrid engine.
- `[mps]` (Apple Silicon/macOS): **ASR** runs on the native MLX Qwen3-ASR from
  [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) (Metal kernels + quantization). **Alignment**
  keeps the same per-language stack as `[cuda]`: English on wav2vec2 CTC (torch, runs on MPS;
  the forced-align DP falls to CPU as torchaudio has no Metal kernel), Japanese/CJK on the ONNX
  MMS aligner (CPU onnxruntime; CoreML is avoided because its Metal context crashes alongside MLX). Only the Qwen fallback (zh·yue,
  or any CTC failure) is served by the MLX Qwen3-ForcedAligner, since the torch `qwen-asr` aligner
  is absent here. The Whisper hybrid/fusion engines (`--asr-model large-v3`, `--hybrid`) run on the
  native [`mlx-whisper`](https://pypi.org/project/mlx-whisper/) Metal port instead of faster-whisper
  (ctranslate2 has no Metal backend). Vocal separation (MelBandRoformer) + PANNs song-skip stay on
  torch-MPS. `qwen-asr` is excluded because its `transformers==4.57.6` pin conflicts with mlx-audio,
  so `[cuda]` and `[mps]` are mutually exclusive — pick one per host.

<details>
<summary><b>Dependencies & what each variant pulls</b></summary>

- The core pulls a pure-torch Mel-Band Roformer vendored in `voxweave.vendor`
  (**no onnx/onnxruntime** —
  `audio-separator` is intentionally avoided because it eagerly imports onnxruntime at the
  top level) + MMS-300m forced aligner (`ctc-forced-aligner`) + layout (`pysbd`) + song-skip
  (`panns-inference`) + CJK break (`budoux` + `jieba`) + translation (`openai`) +
  `pyannote-audio>=4,<5` for opt-in speaker diarization.
- **`[cuda]`** (NVIDIA/Linux): `qwen-asr` + `onnxruntime-gpu` + `faster-whisper`. **`[mps]`**
  (Apple Silicon/macOS): `mlx-audio` + plain `onnxruntime`. Declared **conflicting** in
  `[tool.uv]` (incompatible `transformers` pins), so `uv` resolves each in its own fork — pick one
  per host (`make dev VARIANT=mps` on Apple Silicon).
- **Diarization ships by default but runs only with `--diarize`.** Both supported pyannote
  pipelines are gated independently. Accept the
  [`speaker-diarization-community-1`](https://hf.co/pyannote/speaker-diarization-community-1)
  conditions for the default, or the
  [`speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) conditions (and
  segmentation-3.0's) before selecting `--diarize-model 3.1`. Then use `hf auth login`
  (`uvx --from huggingface_hub hf auth login` after a `uv tool` install),
  `VOXWEAVE_HF_TOKEN`, or `HF_TOKEN`.
- The device is auto-detected at runtime (cuda → mps → cpu); override with `VOXWEAVE_DEVICE`. On
  mps the MLX backend is selected automatically; force it either way with `VOXWEAVE_BACKEND=mlx|torch`.
- **Development**: `make dev` (= `uv sync --extra cuda --dev`; on Apple Silicon use
  `make dev VARIANT=mps` — `[cuda]`/`[mps]` are conflicting extras and can't be synced together).

</details>

**From source** (for development or pulling new code):

```bash
make install       # auto-detects the platform: Apple Silicon -> [mps], anything else -> [cuda]
                   # (cu128 torch wheel with an NVIDIA driver, CPU wheel without)
make reinstall     # after pulling new code (same platform detection)
make uninstall
```

Override the detection per invocation: `make install VARIANT=mps` or
`make install TORCH_BACKEND=cpu`.

## Hardware

The default pipeline (`Qwen3-ASR-0.6B`, `peak` load strategy) runs in **~8 GB of
VRAM** — the separator is freed before ASR + alignment load, so peak ≈ `max(stage)`, not their
sum. `--asr-model qwen3-asr-1.7B` adds roughly **+2 GB** and still fits 8 GB under the default `peak`
strategy. `load_strategy = "sum"` (keeps models resident between passes, skipping reloads on big cards) makes peak the **sum** of the
resident models — plan for 12 GB+. On **Apple Silicon** the MLX 8-bit weights roughly halve the
Qwen footprint (so 1.7B fits comfortably in 16 GB unified memory). `--hybrid` loads a Whisper
engine alongside Qwen to trade VRAM for accuracy — it does **not** save memory; the only knob that
lowers it is staying on `0.6B` + the `peak` strategy.
