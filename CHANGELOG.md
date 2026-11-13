# Changelog

All notable user-facing changes to VoxWeave are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). For upgrade actions and subtle output differences between releases, see [MIGRATING.md](MIGRATING.md).

## [Unreleased]

### Added

- `voxweave doctor` (`--json`) environment diagnostics checking `ffmpeg` hardware encoders, PyTorch CUDA/MPS compute, ASR backend packages, `onnxruntime` provider shadowing, `voxweave.conf` syntax and unknown keys, Hugging Face authentication, and cache disk usage.
- Multi-file `transcribe` batching (`voxweave ep01.mkv ep02.mkv ...`) retaining resident ASR and VAD weights across files when VRAM load strategy permits, plus `--skip-existing` and `--no-clobber` output guards.
- `voxweave models` (`path`, `list`, `pull`, `clean`) to inspect model licenses and cache status, pre-download weights for offline runs, and reclaim disk space.
- `voxweave config` (`path`, `show`, `init`) with secret redaction and backward-compatible `$XDG_CONFIG_HOME/voxweave.conf` discovery.
- `voxweave completions` (`bash`, `zsh`, `fish`) shell tab-completion script generator.
- Global `-q` / `--quiet` mode and `--json` machine-readable output on `transcribe`, `render`, `align`, `translate`, `export`, `pack`, `burn`, and `correct`.
- CUDA runtime `Dockerfile` (`.dockerignore`) and Windows WSL2 / fish / PowerShell setup instructions in `docs/install.md`.
- Cross-platform (`macos-latest` `mps` + `ubuntu-latest` `cuda`) and multi-Python (`3.11`, `3.12`, `3.13`) CI compatibility matrix, `CONTRIBUTING.md`, and structured GitHub bug-report template.

### Changed

- `Ctrl+C` (`KeyboardInterrupt`) now exits with code `130` cleanly without printing a Python traceback.
- Host-pinned P6 oracle tests skip automatically on non-recorded Python versions, macOS, or shallow git clones unless `VOXWEAVE_STRICT_ORACLE=1` is set; `make test` pins `LC_ALL=C.UTF-8` with `TZ` and `PYTHONHASHSEED` unset.

## [0.20.0] - 2026-11-06

### Changed

- `voxweave speakers` (`speakers serve`) stops automatically after a complete Save once subtitles are rendered (or immediately with `--no-render`), keeping the server alive on failure so you can retry.
- Same-stem media files (`episode.wav` and `episode.mp3`) share one speaker mapping, and `align` selects the media bound to the episode's voiceprints.
- Refined PyPI package description, keywords, and classifiers.

## [0.19.0] - 2026-10-25

### Changed

- Qwen3-ForcedAligner decoding on PyTorch now solves for the best non-decreasing word path with soft 3-class boundary refinement instead of grid-locked 80 ms argmax repair, sharply reducing zero-duration words on Chinese and Cantonese episodes.
- Streamlined `README.md` and extracted detailed command, speaker, configuration, internals, and data-contract reference guides under `docs/`.

## [0.18.0] - 2026-10-19

### Added

- Cross-episode speaker voiceprint matching (`--voiceprints`), shared voice library (`voxweave speakers enroll`), and interactive speaker cluster separation (**Split this speaker**) in `speakers serve`.
- Pre-flight Hugging Face authentication and speaker-bound checks for `--diarize` before running vocal separation or ASR.

### Changed

- Unified first-run and cached re-run audio paths through the same 32 kHz mono vocals representation and restored CUDA TF32 state after vocal separation for deterministic re-runs.
- Improved subtitle cue layout for Latin/Cyrillic/Greek column widths, CJK word-boundary wrapping, inter-cue frame gaps, and minimum-duration borrowing.
- Restricted `speakers serve` access to its printed token URL and honored user `umask` on deliverable files.

