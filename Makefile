.PHONY: install reinstall cuda mps uninstall dev test lint typecheck

# Install as a global uv tool (end-user mode): puts the voxweave command on PATH.
# The separation / layout / song-skip / CJK-break / translation pipeline is baked into the core
# deps; the install variant selects the compute platform AND the ASR/alignment backend:
#   VARIANT=cuda (default) -> NVIDIA/Linux: torch Qwen3-ASR+aligner (qwen-asr) + onnxruntime-gpu +
#                             faster-whisper, on the cu128 torch wheel (Blackwell sm_120, no auto-detect)
#   VARIANT=mps            -> Apple Silicon/macOS: native MLX Qwen3-ASR+aligner (mlx-audio) on the
#                             default torch wheel (MPS built in for the separator; no whisper engine)
# Convenience targets: `make cuda` / `make mps` == `make install VARIANT=<x>`.
# Everything lands in an isolated uv tool venv (a bare `uv pip` cannot reach that venv).
# Override the torch index per-invocation if needed, e.g. CPU-only: make install TORCH_BACKEND=cpu
VARIANT ?= cuda

# Optional extras stacked on top of the platform variant. Defaults to diarize (pyannote
# speaker diarization; the feature itself stays opt-in behind --diarize and needs a HF
# token for the gated checkpoint). Disable with `make install EXTRAS=` or stack more:
# `make install EXTRAS=diarize,foo`.
EXTRAS ?= diarize
comma := ,
INSTALL_SPEC = .[$(VARIANT)$(if $(EXTRAS),$(comma)$(EXTRAS))]

ifeq ($(VARIANT),mps)
  TORCH_BACKEND ?= auto
else
  TORCH_BACKEND ?= cu128
endif

# --overrides is required: `uv tool install` ignores [tool.uv] override-dependencies in
# pyproject.toml, so without it the CPU `onnxruntime` (pulled by ctc-forced-aligner /
# faster-whisper) races onnxruntime-gpu for the shared import directory and can silently
# drop CUDAExecutionProvider. See overrides.txt.
install:
	uv tool install --force --torch-backend=$(TORCH_BACKEND) --overrides overrides.txt "$(INSTALL_SPEC)"
	@voxweave --version
	@git diff --quiet 2>/dev/null && echo "installed (git $$(git rev-parse --short HEAD))" || echo "installed (git $$(git rev-parse --short HEAD), uncommitted changes present)"

# Platform shorthands.
cuda:
	$(MAKE) install VARIANT=cuda

mps:
	$(MAKE) install VARIANT=mps

# Force reinstall after pulling new code.
reinstall:
	uv tool install --force --reinstall --torch-backend=$(TORCH_BACKEND) --overrides overrides.txt "$(INSTALL_SPEC)"
	@voxweave --version
	@git diff --quiet 2>/dev/null && echo "reinstalled (git $$(git rev-parse --short HEAD))" || echo "reinstalled (git $$(git rev-parse --short HEAD), uncommitted changes present)"

uninstall:
	uv tool uninstall voxweave

# Development environment (for code changes, matches CI). [cuda] and [mps] are mutually
# exclusive (conflicting transformers pins), so sync exactly one — defaults to cuda; on Apple
# Silicon use: make dev VARIANT=mps
dev:
	uv sync --extra $(VARIANT) --dev

# Unit tests (no network).
test:
	uv run pytest tests/ -v

# Lint / format (project-wide; repo has no ruff config but this is the canonical invocation).
lint:
	uv run --no-project --with ruff ruff check --fix .
	uv run --no-project --with ruff ruff format .

# Static type check (pyright, basic mode, production code only -- see [tool.pyright]).
# Zero errors is the bar; CI enforces it so type noise cannot accumulate again.
typecheck:
	uv run pyright
