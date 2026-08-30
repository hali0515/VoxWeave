# Speaker split cluster report

## Delivery

- Branch: `feat/speaker-split-cluster`
- Worktree: `/mnt/Dev/Git/qsub-worktrees/splitcluster`
- Original base: `e23941f` (`main` when the worktree was created)
- Reconciled base: `6fccb29`, merged without rebasing in `db7b8b8`
- Scope: the served speaker-audition workflow, artifact paths, documentation, and tests
- `voxweave/diarize.py` was not hand-edited by the split work; its community-1 changes
  came only from the merge of `main`.
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
construction reads the resolved checkpoint once, hashes that frozen payload, and gives
pyannote the same in-memory byte buffer before rechecking it. The returned dict batch then
attaches its model, byte hash, and pyannote version immutably. The server therefore makes only
the specified two-argument provider call without reopening a mutable checkpoint pathname.

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

## Original verification

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

## Original optional real-GPU check

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

## Reconciliation with community-1

### Merge

Fetched and merged `main` at `6fccb29` into `feat/speaker-split-cluster` as merge commit
`db7b8b8`. The original feature commit `2f43940` remains intact. Git resolved the README and
speaker-controller test additions automatically, with no textual conflict markers. The semantic
conflict was the embedding authority: current voiceprints record a pinned repository revision,
and community-1 records its nested `#subfolder=embedding` checkpoint, while the original split
provider attested only the bare standalone model id.

No reconciliation edit was made to `voxweave/diarize.py`. The primary purge documentation was
updated to name the one-level speaker-split undo snapshot alongside the other removed artifacts.

### Semantic reconciliation

`AttestedTurnRequest` now carries the exact staged embedding model authority, checkpoint SHA-256,
and pyannote version through the existing two-argument provider call. The provider parses pinned
`repo@revision` and `repo@revision#subfolder=embedding` authorities, resolves them inside the
VoxWeave audio cache with the existing token chain, and reloads its singleton when the requested
identity changes. It checks the installed pyannote version before download, checks expected bytes
before construction, constructs from that same frozen `BytesIO`, checks the buffer again afterward,
and exact-matches the final identity before inference results can reach clustering or sample
generation. A regression test replaces and restores the checkpoint pathname during construction
and verifies that only the already-attested buffer can be loaded.

The obsolete pyannote 3.x import shim was removed. Model loading now uses the pyannote.audio 4.x
API with explicit `token=`, `cache_dir=`, `map_location=`, and strict checkpoint loading. A direct
CPU smoke check loaded the current community-1 nested checkpoint as
`PyannoteAudioPretrainedSpeakerEmbedding` at 16 kHz, with exact identity:

- model: `pyannote/speaker-diarization-community-1@3533c8cf8e369892e6b79ff1bf80f7b0286a54ee#subfolder=embedding`
- checkpoint SHA-256: `6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929`
- pyannote.audio: `4.0.7`

The split endpoint now builds that exact request from the current provenance contract. New
community-1-shaped coverage uses a pinned nested authority and 256-dimensional centroids, applies
the proposal, validates the rewritten sibling/voiceprint conjunction, and passes the result
through the current matching compatibility gate. The confirmation preserves all embedding-space
and selected-pipeline provenance while refreshing the turns digest and both affected centroids.
A store bound to a different embedding authority is refused, and a provider returning a different
identity is rejected before clustering, clips, or mutation.

### Reconciliation gates

| Gate | Result |
| --- | --- |
| `uv run --extra cuda pytest tests/test_turnembed.py tests/test_speakerserve_split.py -q` | 59 passed in 19.51s |
| `uv run --extra cuda pytest tests/test_voice_provenance.py -q` | 6 passed in 0.83s |
| `uv run --extra cuda pytest tests/test_diarize_models.py -q` | 17 passed in 2.26s |
| `uv run --extra cuda pytest tests/ -q` | 3772 passed, 1 skipped in 355.31s |
| `uv run --extra cuda ruff check .` | passed |
| `uv run --extra cuda ruff format --check .` | 260 files already formatted |
| `uv run --extra cuda pyright` | 0 errors, 0 warnings |
| `git diff --check` | passed |
| `git diff main...HEAD --check` | passed |

The pytest warnings were limited to third-party SWIG import deprecations. Direct construction of
the community-1 wrapper also emitted pyannote's upstream one-sample pooling warning; production
turns are padded to the provider's safe two-second minimum before inference.

### Reconciled real-GPU check

The same `ja.16k.wav` check selected 12 spread turns from the 193 `SPEAKER_05` turns in
`out/ja.31.json`. On `cuda:0` (`NVIDIA RTX PRO 4000 Blackwell SFF Edition`), the provider returned
12 finite, normalized 256-dimensional vectors in 2.949 seconds. It bound:

- model: `pyannote/wespeaker-voxceleb-resnet34-LM@837717ddb9ff5507820346191109dc79c958d614`
- checkpoint SHA-256: `366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56`
- pyannote.audio: `4.0.7`

The split remained 5/7 with A indexes `0, 1, 3, 4, 9` and B indexes
`2, 5, 6, 7, 8, 10, 11`. Repeating the bisection and reversing input dictionary order produced
the identical assignment. The recomputed centroid cosine was `0.713902`; vector norms ranged
from `0.9999999999999999` to `1.0`. Only third-party SWIG import deprecations were observed.

Nothing was pushed.
