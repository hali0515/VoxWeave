# Speaker split cluster report

## Delivery

- Branch: `feat/speaker-split-cluster`
- Worktree: `/mnt/Dev/Git/qsub-worktrees/splitcluster`
- Base: `e23941f` (`main` when the worktree was created)
- `main` later advanced independently to `6fccb29` with the sibling community-1 work;
  this branch was not rebased across that in-flight change.
- Scope: the served speaker-audition workflow, artifact paths, documentation, and tests
- `voxweave/diarize.py` was not changed.
- Nothing was pushed.

## Implementation

### Turn embeddings and deterministic bisection

Added `voxweave/turnembed.py` with the swappable provider seam:

```python
turn_embeddings(wav_path, turns) -> dict[index, list[float]]
```

The default provider lazily loads `pyannote/wespeaker-voxceleb-resnet34-LM`, uses the
existing Hugging Face token chain, downloads into the Voxweave cache hierarchy, and
runs on the configured CUDA/MPS/CPU device. Audio is decoded as float32, mixed to mono,
resampled to 16 kHz, cropped per persisted turn, and padded before inference. Returned
vectors are finite, non-zero, dimensionally consistent, and L2-normalized. Model
construction hashes the resolved checkpoint immediately before and after loading, then
attaches its model, byte hash, and pyannote version immutably to that returned dict batch.
The server therefore makes only the specified two-argument provider call while retaining
race-free provenance attestation.

The same module implements seed-free deterministic bisection: a sign-canonicalized
principal component supplies the initial partition and bounded Lloyd iterations refine
it. Group A is canonicalized to contain the earliest persisted turn so dictionary order
cannot change the assignment. Degenerate inputs and tied/numerically ambiguous leading
eigenspaces raise typed errors instead of depending on a BLAS-specific eigenvector.

### Served split workflow

Added guarded `POST` routes to `voxweave/speakerserve.py`:

- `/split` validates the current sibling JSON and voiceprints conjunction, refuses
  unresolved provenance before inference, and verifies the returned batch identity before
  clustering, clips, or proposal publication. It reproduces the recorded audio lane: raw
  captures ignore vocals caches, while separated captures
  require the cache companion, media binding, separator identity, size, and full FLAC
  hash to validate under the cache lock. Recorded normalization and 16 kHz mono settings
  are mirrored. It then returns A/B counts, durations, and up to three 2–6 second data-URI
  sample clips per group.
- `/split-confirm` accepts only the exact active proposal. Under the episode lock it
  assigns B the lowest free `SPEAKER_NN`, recomputes normalized centroids for A and B,
  refreshes the voiceprints binding, appends a blank mapping entry without changing
  existing names, removes the stale suggestion record, and returns the new ID.
- `/split-undo` restores the single saved pre-confirm state only when media and all
  post-confirm inputs still match. Sibling JSON, voiceprints, and mapping bytes are
  restored exactly; any changed input causes a conflict and leaves the snapshot intact.
  Exact before/after mixtures are accepted so the snapshot can recover an interrupted
  multi-file publication, while arbitrary bytes are still refused.

All three routes share `/save`'s host, origin, token, content-length/body-cap, transfer
encoding, and strict JSON checks. Split responses are `no-store`; server actions are
serialized; confirm and undo terminalize the audition session so a restart must reload
the coherent artifact set. The undo snapshot is cache-only and is included in purge.

### Audition page and documentation

Each speaker card now has a hidden `Split this speaker` action. It becomes available only
after the served-endpoint probe succeeds, so an exported/static page remains inert. A
proposal renders both groups with counts, durations, safe text, and audio controls, then
offers apply/cancel actions. Applying disables further mutations and displays:

> split applied — restart `voxweave speakers` to re-audition

README documentation covers prerequisites, the preview/confirm behavior, restart, the
one-level undo policy, and cache/purge handling.

## Verification

Model-free focused coverage includes checkpoint binding, provider padding/normalization
and failures, deterministic/tie-stable clustering, identity mismatch refusal, raw and
separated audio provenance, cache corruption, all route guards, proposal clips, exact
rewrites, interrupted publication rollback, mixed-generation and byte-identical undo,
changed/deleted input refusal, mapping/suggestion behavior, post-confirm `voxweave split`
replay, and post-confirm enrollment using both centroids.

| Gate | Result |
| --- | --- |
| `uv run --extra cuda pytest tests/test_turnembed.py tests/test_speakerserve_split.py -q` | 50 passed |
| `uv run --extra cuda pytest tests/ -q` | 3748 passed in 337.66s |
| `uv run --extra cuda ruff check .` | passed |
| `uv run --extra cuda ruff format --check .` | 254 files already formatted |
| `uv run --extra cuda pyright` | 0 errors, 0 warnings |
| `git diff --check` | passed |

## Optional real-GPU check

The requested `ja.16k.wav` was embedded with the real WeSpeaker model on `cuda:0`, using
12 spread `SPEAKER_05` turns from `out/ja.31.json`. The provider produced finite,
normalized 256-dimensional vectors and a deterministic 5/7 split in 3.683 seconds. The
two recomputed centroids had cosine similarity `0.713902`. The loader bound pyannote
`3.4.0` and checkpoint SHA-256
`366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56`. With warnings
forced to `always`, only third-party Matplotlib/SWIG import deprecations were observed;
there were no short-turn, non-finite-vector, or inference warnings.

## Handoff

The full audition process should be restarted after confirm or undo, as documented. The
controller can review and merge the branch; no repository push was performed.
