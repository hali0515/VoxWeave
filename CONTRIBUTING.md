# Contributing to VoxWeave

Thank you for contributing to VoxWeave. This guide covers environment setup, local verification gates, and codebase conventions.

## Prerequisites

- **Python**: 3.11, 3.12, or 3.13 (`uv` downloads the pinned interpreter automatically).
- **uv**: `>= 0.9.19` (`uv --version`; update with `uv self update`).
- **FFmpeg**: `ffmpeg` and `ffprobe` on `PATH` (optional for unit tests, required for end-to-end media runs and two real-ffmpeg cache tests).

## Development setup

Clone with full history so the detached P6 oracle can verify its reference commits:

```bash
git clone https://github.com/hali0515/VoxWeave.git
cd VoxWeave
```

Sync the development environment for your platform:

```bash
# Linux / Windows WSL2 (CUDA or CPU fallback):
make dev

# Apple Silicon macOS (Metal + MLX):
make dev VARIANT=mps
```

`[cuda]` and `[mps]` are mutually exclusive in `pyproject.toml` because `qwen-asr` and `mlx-audio` pin incompatible `transformers` majors. Always pass `--extra cuda` or `--extra mps` when running `uv run` directly (or use the `Makefile` targets, which auto-detect `VARIANT`). A bare `uv run` without an extra will re-sync `.venv` to the base dependency set.

## Verification gates

Run these checks before opening a pull request:

```bash
# 1. Lint and format (pinned ruff version)
make lint
make lint-check

# 2. Static type check (pyright basic mode on voxweave/, zero errors)
make typecheck

# 3. Unit and contract test suite (offline, models mocked)
make test

# 4. Zero-GPU subtitle cue-breaking quality baseline
make quality-segmentation
```

### P6 alignment and projection oracle

`calibration/p6-oracle/manifest.json` pins exact deterministic outputs across reference alignment and cue-rendering scenarios. Its host-pinned execution tests require:

1. The recorded interpreter micro (`CPython 3.13.12`, synced by `make dev` on Linux).
2. `Linux x86_64`.
3. A non-shallow git clone.

On macOS, other Python versions (`3.11`, `3.12`), or shallow clones, `make test` skips the 14 host-pinned oracle execution tests automatically while still running all schema, vector, and unit tests. On a matching Linux x86_64 checkout, run the full detached oracle gate directly:

```bash
make quality-p6-oracle
```

CI enforces `VOXWEAVE_STRICT_ORACLE=1` on the primary Linux check job so host-pinned oracle tests can never skip there.

If a reviewed change intentionally updates `uv.lock` or bumps the package version in `pyproject.toml`, refresh the recorded execution triplet with:

```bash
uv run --extra cuda python scripts/p6_oracle_release_refresh.py
```

## Codebase conventions

- **Keep CLI startup fast**: `voxweave --help`, `--version`, `doctor`, `config`, `models list`, `completions`, and `render` must not import heavy ML libraries (`torch`, `torchaudio`, `qwen_asr`, `mlx_audio`, `faster_whisper`, `pyannote.audio`) at module import time. Keep heavy imports lazy inside execution paths.
- **Module and function size**: Keep production Python files under 1,500 lines and functions focused (aim for under 150 lines). Extract cohesive CLI subcommands into `voxweave/cli_*.py` modules when `cli.py` grows.
- **stdout vs stderr**: Reserve `stdout` for primary machine-readable deliverables (output file paths or `--json` payloads). Route progress bars, Rich panels, warnings, and logs through `voxweave.ui` (`stderr`).
- **Configuration precedence**: CLI flag -> `VOXWEAVE_*` environment variable -> `voxweave.conf` -> built-in default. Never write secrets (`hf_token`, API keys) into logs, `--json` output, or config templates.
- **Commit messages**: Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

## Reporting bugs

Open a GitHub issue using the bug report template and attach the output of:

```bash
voxweave doctor
```

