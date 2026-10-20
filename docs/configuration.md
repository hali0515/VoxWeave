# Configuration

Precedence: **CLI flag > env var > `~/.config/voxweave.conf` > built-in default.** A commented
default config is written on first run (migrated automatically from a pre-rename `qsub.conf`).
Set `VOXWEAVE_CONFIG` to read (and create) the config file somewhere else.

<details>
<summary><b>Environment variables</b></summary>

**Models**

- `VOXWEAVE_ASR_MODEL` (default `Qwen/Qwen3-ASR-0.6B`; same as `--asr-model`)
- `VOXWEAVE_ASR_CONTEXT` (default none; the ASR bias prompt, same as `--context`)
- `VOXWEAVE_ALIGNER_MODEL` (default `Qwen/Qwen3-ForcedAligner-0.6B`)
- `VOXWEAVE_FUSION_WHISPER` / `VOXWEAVE_FUSION_QWEN` (defaults `large-v3` /
  `Qwen/Qwen3-ASR-1.7B`; the `--hybrid` sub-models, same as `[fusion]` in the config)
- `VOXWEAVE_LOAD_STRATEGY` (`peak` default, or `sum`; same as `load_strategy` in the config)
- `VOXWEAVE_CTC_MAX_DP_FRAMES` (default 90000 ≈ 30 min; same as `ctc_max_dp_frames` in the config)
- `VOXWEAVE_DIARIZE_MODEL` (default `pyannote/speaker-diarization-community-1`; short names `3.1`
  and `community-1`, or any full Hugging Face pipeline id; same as `--diarize-model`)
- `VOXWEAVE_DIARIZE_CLUSTERING` (default `[diarize].clustering` in the config, else `pyannote`;
  `pyannote` or `voiceprint`; same as `--speaker-clustering`)
- `VOXWEAVE_VOICEPRINT_MODEL` (default `[voiceprint].model` in the config, else `auto`; one of
  `auto`, `redimnet2`, `anime-va`, `pyannote`; same as `--voiceprint-model`)
- `VOXWEAVE_TRANSLATE_MODEL` / `VOXWEAVE_FIX_MODEL` (default `[llm].model` in the config, else
  `gpt-6-luna`; same as `--model` on `translate` / `correct`; `auto` = the endpoint's only served model)
- `OPENAI_BASE_URL` (default `[llm].base_url` in the config, else api.openai.com; same as `--base-url`)
- `VOXWEAVE_TRANSLATE_REASONING_EFFORT` (default `[llm].reasoning_effort`, else the endpoint default;
  same as `translate --reasoning-effort`; `default` leaves the request field unset)
- `VOXWEAVE_TRANSLATE_CONCURRENCY` / `VOXWEAVE_TRANSLATE_WINDOW_CUES` (default `[llm].concurrency` /
  `[llm].window_cues`, else 8 (1 for api.openai.com) / 100; same as `translate --concurrency` / `--window`)
- `VOXWEAVE_LLM_TIMEOUT_S` (default 300) — seconds one `translate` / `correct` request may take
  before it is retried; raise it for a slow self-hosted endpoint
- `VOXWEAVE_DEVICE` (default: auto-detect `cuda:0` → `mps` → `cpu`)
- `VOXWEAVE_BACKEND` (`mlx` | `torch`; default: `mlx` on mps, else `torch`) — picks the ASR/alignment backend
- `VOXWEAVE_HF_TOKEN` / `HF_TOKEN` — authentication for gated models, including both pyannote
  diarizers; alternatively authenticate once with `hf auth login` (after a `uv tool` install:
  `uvx --from huggingface_hub hf auth login`)
- `VOXWEAVE_OFFLINE` (`1` to enable) — once all models are cached, sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` so loading skips the per-file HEAD revalidation + optional-file probing huggingface_hub/transformers otherwise do on every run (no network on a cache hit). Leave off for the first download.
- `VOXWEAVE_MLX_ASR_REPO` / `VOXWEAVE_MLX_ALIGNER_REPO` / `VOXWEAVE_MLX_WHISPER_REPO` — MLX backend
  repos. By default the ASR repo tracks `--asr-model` size (`--asr-model 1.7b` → `mlx-community/Qwen3-ASR-1.7B-8bit`)
  and the Whisper repo tracks the Whisper size (`--asr-model large-v3` → `mlx-community/whisper-large-v3-mlx`);
  set the matching var to hard-pin a specific quant (e.g. a 4-bit build) regardless of `--asr-model`.

Model weights (torch + MLX) and private media snapshots live under `~/.cache/voxweave/`
(override the root with `VOXWEAVE_CACHE_ROOT`), so a container only needs to bind-mount that
one directory for models. Per-media machine artifacts live beside the media itself under
`<media directory>/cache/<stem>/` (see "Files on disk" below), so they travel with the media.
Model weights are auto-downloaded under the `asr`,
`align`, and `audio` subdirectories on first use. Each model exposes an env override to swap the
HF repo, or to point at an explicit local file (which, if it exists, skips the HF download):

- `VOXWEAVE_SEPARATOR_REPO` / `VOXWEAVE_SEPARATOR_REPO_FILE` (default `KimberleyJSN/melbandroformer` /
  `MelBandRoformer.ckpt`), or `VOXWEAVE_SEPARATOR_CKPT` / `VOXWEAVE_SEPARATOR_CONFIG` for explicit
  weights + matching yaml (default `vocals_mel_band_roformer.ckpt` / `.yaml` under
  `VOXWEAVE_MODEL_DIR`, which itself defaults to the cache root)
- `VOXWEAVE_PANNS_REPO` / `VOXWEAVE_PANNS_REPO_FILE` (default `thelou1s/panns-inference` /
  `Cnn14_mAP=0.431.pth`), or `VOXWEAVE_PANNS_CKPT` for an explicit checkpoint (song-skip CNN)
- `VOXWEAVE_MMS_REPO` / `VOXWEAVE_MMS_REPO_FILE` (default `deskpai/ctc_forced_aligner` /
  `04ac86b67129634da93aea76e0147ef3.onnx`), or `VOXWEAVE_MMS_MODEL` for an explicit onnx path
  (Japanese/CJK MMS-300m aligner; its weights are licensed **CC BY-NC 4.0 (non-commercial)**,
  see `THIRD_PARTY_NOTICES.md` for every model's licence)
- `VOXWEAVE_REDIMNET2_CKPT` / `VOXWEAVE_ANIME_VA_CKPT` — an explicit local copy of the
  `redimnet2` / `anime-va` voiceprint checkpoint (offline hosts). Unlike the overrides above,
  the file must still be the pinned checkpoint: its SHA-256 is verified, because the
  checkpoint defines the voice-store embedding space. For `anime-va` this variable is the
  offline route: its cache entry uses the Hugging Face hub layout, so a file copied into
  `~/.cache/voxweave/audio/` is not picked up (`redimnet2` also accepts a copy at
  `~/.cache/voxweave/audio/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt`)

**Tuning**

- `VOXWEAVE_VOICEPRINTS` (`1/0`, `true/false`, `yes/no`, or `on/off`; opt-in capture,
  overridden by the explicit CLI flag)
- `VOXWEAVE_VOICES_ACCEPT` (default `off`; finite `[-1,1]` enables reviewed audition-page prefills)
- `VOXWEAVE_VOICES_SUGGEST` (minimum similarity shown as a suggestion) and
  `VOXWEAVE_VOICES_MARGIN` (minimum top-two margin for a prefill). Unset, each embedding space
  uses its own default: `0.45` / `0.05` for `redimnet2` and the legacy `pyannote` lane,
  `0.35` / `0.05` for `anime-va` (whose same-speaker cosines run lower). The dedicated
  embedders' values are provisional; measure yours with `scripts/calibrate_voiceprints.py`
- `VOXWEAVE_VOICES_GLOBAL_SUGGEST` (the stricter bar for voice-library suggestions from other
  scopes; unset: `0.60` for `redimnet2` and `pyannote`, `0.50` for `anime-va`, all provisional;
  never below the suggest threshold)
- `VOXWEAVE_VOICES_DIR` (the voice library directory; same as `--voices-dir`, default
  `[voices].dir` in the config, else `$XDG_DATA_HOME/voxweave/voices` or
  `~/.local/share/voxweave/voices`)

- `VOXWEAVE_MAX_CHUNK_SEC` (default 120; shorter chunks reduce ASR repetition loops on long segments)
- `VOXWEAVE_LOUDNORM` (default `loudnorm=I=-16:TP=-1.5:LRA=11`; the `-af` filter for `--normalize`)
- `VOXWEAVE_MIN_CUE_SEC` (default 0 = off; set e.g. 0.8 to pad `align`-stage cues to a minimum
  display duration; distinct from the segmentation floor `VOXWEAVE_SEG_MIN_CUE_SEC`)
- `VOXWEAVE_SNAP_VAD_THRESHOLD` (default 0.25; sensitive VAD used when repositioning
  zero-duration units against the original audio)
- `VOXWEAVE_SHOT_SCENE` (default 0.3, range (0, 1]; ffmpeg scene-change score above which a
  frame counts as a shot cut for `--shot-snap`; an invalid value warns and uses the default)
- `VOXWEAVE_SHOT_SNAP_MS` (default 458 = 11 frames at 24 fps; `0` disables snapping) — how far
  from a shot cut a cue boundary is still moved onto it, the outermost Netflix adjustment zone
- `VOXWEAVE_FFMPEG_TIMEOUT` (default 3600) — seconds one ffmpeg audio decode may take; raise it
  for very long media
- `VOXWEAVE_SONG_CORE_MERGE_SEC` (default 15; song spans within this gap of a long OP/ED
  cluster into one song "core" that stops the dialogue edge trim — an isolated brief sting
  farther away is trimmed through instead of anchoring dialogue into the excised song)
- `VOXWEAVE_SPEECH_RESCUE_MIN_S` (default 3; minimum length of a PANNs clean-dialogue
  stretch with no silero coverage to be rescued into the chunk stream — catches dialogue
  silero under-scores, e.g. theatrical delivery)
- `VOXWEAVE_CTC_ENVELOPE_PAD_SEC` (default 2; lead-in/out pad when the full-file alignment
  pass is cropped to the transcribed chunk envelope during transcribe, keeping a skipped
  leading/trailing song out of the aligner's waveform)

**Throughput (opt-in; see [Performance knobs](#performance-knobs))**

- `VOXWEAVE_SEP_BATCH` / `VOXWEAVE_CTC_BATCH` / `VOXWEAVE_MMS_BATCH` / `VOXWEAVE_ASR_BATCH`
  (defaults 1 / 1 / 4 / 1; same as `[batch].separate` / `.ctc` / `.mms` / `.asr`) — windows or
  chunks per GPU forward pass. Values below 1 are clamped to 1; a non-integer value is ignored
  and the next source in the precedence chain applies
- `VOXWEAVE_ASR_BATCH_MIN_CPS` (default 0.5) / `VOXWEAVE_ASR_BATCH_MIN_CHECK_SEC` (default 2.0)
  — the qwen-asr #207 guard on the batched ASR path: a batched result with fewer than `MIN_CPS`
  alphanumeric characters per second of audio, for a chunk of at least `MIN_CHECK_SEC`, is
  rejected and that chunk is re-run alone. Shorter chunks are exempt (a cough legitimately
  transcribes to nothing)
- `VOXWEAVE_SEP_AUTOCAST` (`off` (default) | `bf16` | `fp16`; same as `[separate].autocast`) —
  mixed precision for the vocal-separation forward pass. CUDA only; ignored on CPU/MPS. An
  unrecognized value warns once and falls back to `off` rather than to the config file
- `VOXWEAVE_TF32` (default on; `0` / `false` / `off` disables) — TF32 matrix multiplies for the
  vocal-separation forward pass on CUDA (Ampere or newer). It applies to separation only and is
  restored afterwards, so later stages keep strict fp32 matrix multiplies

</details>

<details>
<summary><b>Config file (<code>~/.config/voxweave.conf</code>, TOML)</b></summary>

Every key below is optional — delete a line to fall back to its built-in default. The values
shown are a usable starting point, not the defaults (the auto-written template has everything
commented out).

```toml
# ~/.config/voxweave.conf  —  TOML
# Precedence: CLI flag > env var > this file > built-in default.

# Default ASR model (= --asr-model). Short name (qwen3-asr-0.6b | qwen3-asr-1.7b) or full HF id.
# Special value "hybrid" (= --hybrid) -> dual-ASR fusion (whisper text + Qwen punctuation).
asr_model = "Qwen/Qwen3-ASR-1.7B"        # built-in default: Qwen/Qwen3-ASR-0.6B

# Hugging Face token for the gated diarization models (top-level key; prefer the
# VOXWEAVE_HF_TOKEN / HF_TOKEN environment variables over storing it here).
# hf_token = "hf_..."

# Single-pass CTC alignment budget in emission frames (default 90000 ≈ 30 min at 50 fps);
# longer audio is split at silence anchors. Raise it on large cards for long movies.
# ctc_max_dp_frames = 150000

# Model load strategy:
#   "peak" (default) — serial peak-shaving: all-chunk ASR -> release -> all-chunk align;
#                      ASR and aligner never co-reside, peak VRAM = max(models). Works on 8 GB.
#   "sum"            — same passes as "peak", but the ASR model(s) stay resident while the aligner
#                      loads; peak VRAM = sum(models), but skips two model swap round-trips
#                      (faster on large-VRAM cards).
load_strategy = "sum"

# Inference batch sizes: windows per GPU forward (env: VOXWEAVE_SEP_BATCH / VOXWEAVE_CTC_BATCH /
# VOXWEAVE_MMS_BATCH / VOXWEAVE_ASR_BATCH). On an 8 GB-class card batch=1 already saturates
# compute — measured no speedup at 2/4, just ~+0.8 GiB VRAM per extra separation window — so the
# defaults stay at 1. Only worth raising on much wider GPUs, and only after measuring.
[batch]
separate = 1                             # vocal separation (MelBandRoformer) 8s windows
ctc      = 1                             # wav2vec2 CTC emission 30s windows (en aligner)
mms      = 4                             # MMS-300m emission batch (ja aligner)
asr      = 1                             # Qwen3-ASR chunks per decode call; 1 = the per-chunk call.
                                         # Opt-in: faster but its transcripts differ from batch 1 —
                                         # see "Performance knobs" below

# Vocal separation numerics (= env VOXWEAVE_SEP_AUTOCAST). autocast wraps only the model
# forward; the overlap-add accumulation stays fp32. CUDA only (ignored on CPU / MPS).
#   "off" (default) — fp32 forward, the reference output, byte-identical run to run.
#   "bf16" | "fp16" — mixed-precision forward: faster and slightly leaner, at the cost of tiny
#                     waveform differences in the stem. See "Performance knobs" below.
[separate]
autocast = "off"

# Diarization pipeline (= --diarize-model / env VOXWEAVE_DIARIZE_MODEL).
# Values: "community-1" (built-in default), "3.1", or any full Hugging Face pipeline id.
# Only legacy (voiceprint model "pyannote") stores depend on it: their centroids come from the
# pipeline's own embeddings, so switching pipelines starts a fresh legacy store.
# clustering: who is who once the pipeline has found the turns (= --speaker-clustering /
# env VOXWEAVE_DIARIZE_CLUSTERING). "pyannote" (built-in default) = the pipeline's own
# clustering; "voiceprint" = regroup with ReDimNet2 (CC BY-NC-SA 4.0 weights) and drop
# turns nobody can be attributed to.
[diarize]
model = "community-1"
clustering = "pyannote"

# Voiceprint embedder (= --voiceprint-model / env VOXWEAVE_VOICEPRINT_MODEL); used with --voiceprints.
#   "auto" (default) — anime-va for Japanese, redimnet2 for every other language
#   "redimnet2"      — ReDimNet2-B6; weights non-commercial (CC BY-NC-SA 4.0)
#   "anime-va"       — anime voice-actor ECAPA-TDNN; Japanese only, unclear data provenance
#   "pyannote"       — legacy: the diarization pipeline's embeddings (matches older stores)
# Voice stores are per embedding space; changing the diarizer does not affect redimnet2/anime-va.
[voiceprint]
model = "auto"

# Voice library (= --voices-dir / env VOXWEAVE_VOICES_DIR): where `speakers enroll` saves named
# voices, shared by every media folder. Default: $XDG_DATA_HOME/voxweave/voices, else
# ~/.local/share/voxweave/voices. May be a NAS path shared by several machines of the same user
# account (needs working NFS locking, see "Voice library"); several accounts cannot share one library.
# A relative path is relative to this file's directory.
[voices]
# dir = "/mnt/nas/voxweave/voices"

# Default on/off for the boolean pipeline flags. An explicit CLI flag always wins
# (e.g. separate = false here, --separate on the command line for one run).
[defaults]
separate   = true                        # vocal separation before ASR/alignment (--separate/--no-separate)
skip_songs = true                        # PANNs music detection + skip before ASR (--skip-songs/--no-skip-songs)
normalize  = false                       # loudnorm on the 16k input (--normalize/--no-normalize)
diarize    = false                       # pyannote speaker diarization (--diarize/--no-diarize; gated-model token required)
voiceprints = false                      # opt-in biometric centroid capture; requires diarize
timestamps = true                        # cue timing lines in the VTT (--timestamps/--no-timestamps)
shot_snap  = true                        # snap cue boundaries onto shot changes (--shot-snap/--no-shot-snap)
vad_mask   = false                       # suppress wav2vec2 CTC emissions outside speech; MMS/Qwen ignore it (--vad-mask/--no-vad-mask)

# LLM for translate / correct: any OpenAI-compatible chat-completions endpoint.
# Precedence per key: CLI option > env (VOXWEAVE_TRANSLATE_MODEL / VOXWEAVE_FIX_MODEL,
# OPENAI_BASE_URL) > this section > built-in (gpt-6-luna on api.openai.com, key from OPENAI_API_KEY).
[llm]
model = "auto"                           # or a model name; "auto" = the endpoint's only served model
base_url = "http://127.0.0.1:8000/v1"    # e.g. a local vLLM; remove for api.openai.com
api_key_env = ""                         # "" = keyless endpoint; else the env var holding the key
# reasoning_effort = "low"               # translate only; accepted values depend on the served model
# concurrency = 8                        # translate windows in flight; 1 = single whole-episode request
# window_cues = 100                      # cues per window when concurrency > 1

# dual-ASR fusion sub-models — only consulted when running with --hybrid.
[fusion]
whisper = "large-v3-turbo"               # Whisper size: large-v3 (best) | large-v3-turbo (~5x faster); faster-whisper on cuda, mlx-whisper on mps
qwen    = "Qwen/Qwen3-ASR-1.7B"          # punctuation model; must emit punctuation -> 1.7B, not 0.6B

# Per-language forced-alignment model. Key = ISO-639-1 code; unlisted languages use Qwen3-ForcedAligner.
# Values:
#   "mms"   — MMS-300m + uroman, full-file single pass (immune to per-cue drift; the gold standard).
#   HF id   — wav2vec2 CTC via HF transformers, also full-file; weights land in ~/.cache/voxweave/align.
#   bundle  — torchaudio bundle name, e.g. "WAV2VEC2_ASR_LARGE_LV60K_960H" (same model, cached in ~/.cache/torch).
#   ""      — explicitly fall back to Qwen for that language.
[align]
en = "facebook/wav2vec2-large-960h-lv60-self"  # English: LV60K-self CTC, full-file (HF hub)
ja = "mms"                                      # Japanese: MMS-300m + uroman full-file (= whisperx fork align_ctc)
# zh  = "mms"                                   # Chinese can also use MMS; default is Qwen (native CJK char-level)
# yue = ""                                      # force Qwen for Cantonese
```

</details>

## Performance knobs

Two GPU throughput settings are **opt-in and off by default**, because both change the
output. They are worth enabling only on a wide GPU, and only if you have re-checked the
result on your own material — there is no truth ruler in the pipeline that can tell you
whether the changed output is better or worse.

| Knob                                            | Default              | Measured on an RTX PRO 4000 (24 GB)                                                             | What changes                                                                                               |
| ----------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `[batch].asr` / `VOXWEAVE_ASR_BATCH`            | `1` (per-chunk call) | 4 → 1.34x faster at 6.4 GiB peak; 8 → 1.49x at 8.9 GiB (Qwen3-ASR-1.7B, greedy, 24-min episode) | Transcripts drift ~1.5% CER from batch 1 — scattered small edits from bf16 batched kernels, no chunk lost  |
| `[separate].autocast` / `VOXWEAVE_SEP_AUTOCAST` | `off` (fp32)         | `bf16` → 1.35x faster separation, peak VRAM 1.69 → 1.57 GiB                                     | The vocal stem differs slightly (~52 dB SNR against the fp32 stem), and the ASR run on it drifts ~2.3% CER |

Neither has a CLI flag; precedence for both is env var > config file > built-in default.
`[batch].asr` applies to the torch Qwen engine only — Whisper and the Apple Silicon MLX
adapter take one chunk per ASR call whatever it says — and autocast applies to CUDA only,
being ignored on CPU/MPS. Raising `[batch].separate` above 1 bought no speedup on the same
GPU (the separator is already compute-bound), so it stays at 1 as well.

Batched ASR is guarded against qwen-asr #207, where a mixed-length batch can corrupt its
shorter item into a lone `!`: chunks are grouped by duration to keep each batch's lengths
close, and any batched result that comes back implausibly empty for its chunk's duration is
re-run alone through the per-chunk call (thresholds: `VOXWEAVE_ASR_BATCH_MIN_CPS`,
`VOXWEAVE_ASR_BATCH_MIN_CHECK_SEC`). A batch whose call raises is likewise redone chunk by
chunk, so one poisoned chunk degrades alone.

Shot-change detection normally needs no setting (`VOXWEAVE_SHOT_SCENE` tunes its
sensitivity): it is a CPU-only ffmpeg pass that now starts before transcription and is
joined at its workflow step, so it overlaps the GPU stages and the `detect shot changes` entry in the timing line is normally ~0s.
