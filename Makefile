.PHONY: install reinstall uninstall dev test lint typecheck

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

# ---- Platform auto-detection -------------------------------------------------
# Explicit VARIANT=cuda|mps always wins. Otherwise: Apple Silicon -> mps; everything
# else -> cuda (on Intel macs the [cuda] extra degrades cleanly: its GPU wheels carry
# non-darwin markers, so only the torch-CPU stack lands).
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
  VARIANT ?= mps
else
  VARIANT ?= cuda
endif

# TORCH_BACKEND: macOS resolves torch from the default index (MPS is built in); on
# Linux use the cu128 wheel only when an NVIDIA driver is actually present, else fall
# back to the CPU wheel instead of pulling gigabytes of unusable CUDA blobs.
ifeq ($(VARIANT),mps)
  TORCH_BACKEND ?= auto
endif
ifeq ($(UNAME_S),Darwin)
  TORCH_BACKEND ?= auto
else ifneq ($(shell command -v nvidia-smi 2>/dev/null),)
  TORCH_BACKEND ?= cu128
else
  TORCH_BACKEND ?= cpu
endif

# ---- Extras ------------------------------------------------------------------
# Explicit EXTRAS=... always wins (EXTRAS= for none; stack with commas). Otherwise
# preserve what the existing tool venv already has, so a plain `make reinstall`
# never silently drops diarize (detected via its pyannote package); a first install
# defaults to diarize (the feature stays opt-in behind --diarize + the HF token).
TOOL_SITE := $(firstword $(wildcard $(HOME)/.local/share/uv/tools/voxweave/lib/python*/site-packages))
ifeq ($(TOOL_SITE),)
  EXTRAS ?= diarize
else ifneq ($(wildcard $(TOOL_SITE)/pyannote),)
  EXTRAS ?= diarize
else
  EXTRAS ?=
endif
comma := ,
INSTALL_SPEC = .[$(VARIANT)$(if $(EXTRAS),$(comma)$(EXTRAS))]

# --overrides is required: `uv tool install` ignores [tool.uv] override-dependencies in
# pyproject.toml, so without it the CPU `onnxruntime` (pulled by ctc-forced-aligner /
# faster-whisper) races onnxruntime-gpu for the shared import directory and can silently
# drop CUDAExecutionProvider. See overrides.txt.
install:
	@echo "detected: variant=$(VARIANT) torch-backend=$(TORCH_BACKEND) extras=$(or $(EXTRAS),none)"
	uv tool install --force --torch-backend=$(TORCH_BACKEND) --overrides overrides.txt "$(INSTALL_SPEC)"
	@voxweave --version
	@git diff --quiet 2>/dev/null && echo "installed (git $$(git rev-parse --short HEAD))" || echo "installed (git $$(git rev-parse --short HEAD), uncommitted changes present)"

# Force reinstall after pulling new code.
reinstall:
	@echo "detected: variant=$(VARIANT) torch-backend=$(TORCH_BACKEND) extras=$(or $(EXTRAS),none)"
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
