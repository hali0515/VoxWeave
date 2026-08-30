# Community-1 diarization integration report

Branch: `feat/diarize-community1`

Base: `main@e23941fd76f7baa7d1652078edfa4672e5fd7ddd`

## Phase 0 measurement spike

Phase 0 completed before any production-code change. Measurements used:

- GPU: NVIDIA RTX PRO 4000 Blackwell SFF Edition, 24,467 MiB;
- pyannote.audio 4.0.7;
- PyTorch 2.11.0+cu128;
- lab environment: `/tmp/claude-1000/-mnt-Dev-Git-qsub/76be7f61-2a96-4094-80b0-3b9311d0223b/scratchpad/diarize-ab/.venv`;
- audio: `diarize-ab/wav/ja.16k.wav`, 1,431.424 seconds, SHA-256 `22be7401701b7d1fb317607dde25c4a251a767a136c853794b4d616aaf2c84e5`;
- fresh process per model, waveform dictionary input at 16 kHz mono, `min_speakers=1`, `max_speakers=10`;
- measurement script: `diarize-ab/measure_phase0.py`, SHA-256 `1a66b492b2b19115eb1c59155090ca12906fe2f268817def0aa41cdfa66f35e4`.

| Pipeline | Load | Inference | Peak allocated | Peak reserved | Output | Embeddings | Exclusive output |
|---|---:|---:|---:|---:|---|---|---|
| `pyannote/speaker-diarization-3.1` under 4.0.7 | 10.470 s | 59.562 s | 2,723.963 MiB | 3,890 MiB | `DiarizeOutput`, 687 normal turns | NumPy `[10, 256]` | `Annotation`, 663 turns |
| `pyannote/speaker-diarization-community-1` | 2.499 s | 60.879 s | 2,723.963 MiB | 3,890 MiB | `DiarizeOutput`, 743 normal turns | NumPy `[10, 256]` | `Annotation`, 731 turns |

The 3.1-under-4.x peak is 2.660 GiB allocated: close to the decision table's approximate
2.5 GiB low-memory bound and decisively not the reported ~9.5 GiB regression. Embeddings work
for both models. The low-memory decision-table branch therefore applies:

- core dependency becomes `pyannote-audio>=4,<5`;
- the default remains `pyannote/speaker-diarization-3.1` so existing authenticated users do
  not encounter a new community-1 gate;
- configuration selects `3.1`, `community-1`, or an arbitrary Hugging Face pipeline ID.

### pyannote 4.x API observations

- Both models use `Pipeline.from_pretrained(..., token=..., cache_dir=...)`; `use_auth_token`
  is removed.
- Both accept the existing `{"waveform": Tensor[channel,time], "sample_rate": int}` input,
  avoiding the observed torchcodec CUDA-library warning.
- `num_speakers`, `min_speakers`, and `max_speakers` remain call keyword arguments.
- Both return `DiarizeOutput`. Normal turns are under `.speaker_diarization`, exclusive turns
  under `.exclusive_speaker_diarization`, and label-ordered centroids under
  `.speaker_embeddings`. Embeddings are returned by default; 4.x has no
  `return_embeddings` switch.
- Normal annotations support both direct `(segment, speaker)` iteration and
  `itertracks(yield_label=True)`; VoxWeave will retain the latter and will not silently switch
  to exclusive turns.
- The existing waveform-dict path is required on this host because torchcodec cannot load
  `libnppicc.so.12`; this remains a warning, not a diarization failure.

## Implementation decisions

### Dependency and selection contract

- Core dependency: `pyannote-audio>=4,<5`; the empty `diarize` compatibility extra remains
  available and the CUDA/MPS conflict declarations are unchanged. `uv.lock` resolves
  pyannote.audio 4.0.7.
- Default: `pyannote/speaker-diarization-3.1`.
- Aliases: `3.1` resolves to the default repository and `community-1` resolves to
  `pyannote/speaker-diarization-community-1`. Other non-blank values pass through as full
  pipeline identifiers.
- Selection precedence: CLI `--diarize-model` > `VOXWEAVE_DIARIZE_MODEL` >
  `[diarize].model` > the 3.1 default. Blank values fall through rather than masking the next
  source.
- Model selection is resolved at the public CLI boundary, threaded through `process`,
  `_process_from_source` and `transcribe`, then idempotently normalized by `diarize_turns` for
  direct API callers; diarization remains opt-in and no `[defaults]` behavior changed.

### Loader, cache, and runtime behavior

- The pyannote 4 loader uses `token` and the VoxWeave private cache root for the outer pipeline
  and every nested Hub reference. Repository revisions, nested subfolders and child revisions,
  local pipeline files, and structured embedding loader arguments are preserved.
- Pipeline config bytes are read once and those same bytes are both hashed and parsed. Nested
  embedding checkpoints are pinned before construction, rehashed after construction, and bound
  into provenance so a replacement during loading fails closed.
- The 3.4-era torchaudio compatibility shim and safe-global whitelist are retired. The existing
  waveform-dictionary input and TF32 save/restore contract remain intact.
- pyannote.audio 4.0.7's 3.1 pipeline attempts to obtain the community-1 PLDA artifact even
  though 3.1 declares agglomerative clustering and does not use PLDA. A narrow construction-only
  compatibility context suppresses that request only for the exact 3.1/agglomerative/no-PLDA
  plan. Every other model and clustering plan fails closed. Construction, singleton replacement,
  inference, and release are serialized so the temporary compatibility seam cannot affect a
  concurrent community-1 load.
- A gate failure names the selected model and its own Hugging Face model-card URL; access to 3.1
  and community-1 remains separately gated.

### Embeddings and provenance

- pyannote 4 `DiarizeOutput` and legacy tuple/annotation output shapes are normalized to the
  existing turn contract. Label-ordered 256-dimensional rows from `speaker_embeddings` are
  normalized into the existing per-speaker centroid contract.
- Provenance records the resolved diarization pipeline, exact outer config digest, resolved
  embedding source, exact embedding checkpoint digest, embedding dimension, pyannote.audio
  version, torch version, and audio identity. Tokens and private cache paths are excluded.
- Voiceprint compatibility now includes the embedding authority and selected pipeline. A store
  from another model or embedding checkpoint is valid as data but is not eligible for matching;
  existing pyannote.audio 3.4 sidecars continue to validate and replay.

### P6 execution record

Regenerating `uv.lock` changes the P6 recorded environment. The oracle manifest's
`dependency_lock_sha256` and derived `container_digest` were refreshed; its package version,
registry, fixtures, and every expected byte artifact are unchanged. The canonical three-part
oracle gate passes.

## User model switching

After accepting the selected repository's conditions and authenticating with `hf auth login`,
`VOXWEAVE_HF_TOKEN`, or `HF_TOKEN`, users can select a model in exactly one of these ways:

```console
# CLI (highest precedence)
voxweave interview.mkv --diarize --diarize-model community-1

# Environment
VOXWEAVE_DIARIZE_MODEL=community-1 voxweave interview.mkv --diarize
```

Or in `~/.config/voxweave.conf`:

```toml
[diarize]
model = "community-1"
```

Use `3.1` to select the compatibility default explicitly. A full Hugging Face pipeline ID can
replace either alias.

## Verification

- `uv lock --check`: pass, 212 packages.
- `uv run --extra cuda --no-sync pytest -q`: **3,711 passed, 1 skipped** in 368.13 seconds.
  The skip is the explicitly opt-in live GPU test.
- Live community-1 GPU E2E on the frozen Japanese waveform through public `diarize_turns`:
  **1 passed** in 64.82 seconds; turns, label-keyed normalized 256-dimensional centroids, and
  bound provenance were all asserted. The torchcodec shared-library warning is expected because
  the tested waveform-dictionary path bypasses torchcodec.
- Real pyannote.audio 4.0.7 constructor smoke against the private cache: pass for both
  `speaker-diarization-3.1` and `speaker-diarization-community-1`, including resolved embedding
  checkpoint provenance.
- `uv run --extra cuda --no-sync ruff check .`: pass.
- `uv run --extra cuda --no-sync ruff format --check .`: pass, 257 files already formatted.
- `uv run --extra cuda --no-sync pyright`: pass, 0 errors and 0 warnings.
- `make quality-p6-oracle VARIANT=cuda`: pass for validate, byte comparison, and source gates.
- `git diff --check`: pass.

Per the task boundary, `make reinstall` and its installed-environment grep verification remain
for the controller after merge. Nothing was pushed.
