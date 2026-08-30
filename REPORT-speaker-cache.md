# Speaker serve and artifact-cache report

## Feature A: localhost speaker audition and write-back

`voxweave speakers <media>` now returns an in-memory audition from the generation
layer and serves it only on `127.0.0.1`. The CLI prints the selected URL, opens it
best-effort unless `--no-open` is set, and accepts `--port` with an ephemeral-port
default. No audition HTML is written to disk.

The browser fetches the current mapping for each session, lets saved values override
machine suggestions, and writes reviewed names back through an authenticated `POST
/save`. The server validates Host, optional Origin, a per-session secret, body size,
and a closed JSON shape before taking the media's episode lock and atomically replacing
the mapping. It preserves skeleton speaker order and prints the cached/legacy sibling
path that can be passed to `voxweave split`. The Copy button remains as a fallback.

Existing mappings are now the normal edit path. Generation keeps the existing
snapshot, fingerprint, sibling, voiceprint, voices-store, and lock rechecks; it installs
an empty skeleton only when the mapping is absent and treats an atomic-install race as
an existing authoritative mapping. Legacy `.speakers.html` files are still removed by
the purge operation, but no new HTML file is produced.

### Feature A gates

- `uv run --extra cuda pytest tests/ -q`: 3,645 passed, 0 failed in 311.52s on
  the post-fix replay (the initial cold-environment run also passed, with four
  dependency syntax warnings).
- `uv run --no-project --with ruff ruff check .`: passed.
- `uv run --no-project --with ruff ruff format --check .`: passed after formatting
  one touched source file.
- `uv run --extra cuda pyright`: 0 errors, 0 warnings, 0 informations.

## Feature B: machine artifact cache

Pending in the second commit.
