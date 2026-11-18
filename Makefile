.PHONY: install reinstall uninstall dev test lint lint-check typecheck \
	quality quality-segmentation quality-record-segmentation

# Install as a global uv tool (end-user mode): puts the voxweave command on PATH.
# Separation / layout / song-skip / CJK-break / translation support is baked into
# the core deps; add EXTRAS=diarize for speaker diarization (--diarize). The install
# variant selects the compute platform AND the ASR/alignment backend:
#   VARIANT=cuda (default except on Apple Silicon) -> NVIDIA/Linux: torch Qwen3-ASR+aligner
#                             (qwen-asr) + onnxruntime-gpu + faster-whisper, on the cu128 torch wheel
#                             when nvidia-smi is present (Blackwell sm_120), else the CPU wheel
#   VARIANT=mps            -> Apple Silicon/macOS: native MLX Qwen3-ASR+aligner (mlx-audio) +
#                             mlx-whisper for the hybrid engines, on the default torch wheel
#                             (MPS built in for the separator)
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

# ---- Optional extras ---------------------------------------------------------
# Pass EXTRAS=diarize to install optional speaker diarization (pyannote.audio 4.x).
EXTRAS ?=
comma := ,
INSTALL_SPEC = .[$(VARIANT)$(if $(EXTRAS),$(comma)$(EXTRAS))]

# `uv tool install --torch-backend` needs uv 0.9.19 or newer; older releases reject the flag.
UV_MIN_VERSION := 0.9.19
UV_VERSION_CHECK = uv --version | awk -v min=$(UV_MIN_VERSION) '{ split($$2, have, "."); split(min, need, "."); for (i = 1; i <= 3; i++) { if (have[i] + 0 > need[i] + 0) exit 0; if (have[i] + 0 < need[i] + 0) exit 1 } }' || { echo "uv $(UV_MIN_VERSION) or newer is required: run 'uv self update'" >&2; exit 1; }
# The tool's own entry point: the uv tool bin directory need not be on PATH yet.
TOOL_VOXWEAVE = "$$(uv tool dir --bin)/voxweave"
# Untracked and staged changes count as well as unstaged ones.
GIT_STATE = $$(git rev-parse --short HEAD)$$(test -z "$$(git status --porcelain 2>/dev/null)" || echo ", uncommitted changes present")

# --overrides is required: `uv tool install` ignores [tool.uv] override-dependencies in
# pyproject.toml, so without it the CPU `onnxruntime` (pulled by ctc-forced-aligner /
# faster-whisper) races onnxruntime-gpu for the shared import directory and can silently
# drop CUDAExecutionProvider. See overrides.txt.
install:
	@echo "detected: variant=$(VARIANT) torch-backend=$(TORCH_BACKEND) extras=$(or $(EXTRAS),none)"
	@$(UV_VERSION_CHECK)
	uv tool install --force --torch-backend=$(TORCH_BACKEND) --overrides overrides.txt "$(INSTALL_SPEC)"
	@$(TOOL_VOXWEAVE) --version
	@echo "installed (git $(GIT_STATE))"

# Force reinstall after pulling new code.
reinstall:
	@echo "detected: variant=$(VARIANT) torch-backend=$(TORCH_BACKEND) extras=$(or $(EXTRAS),none)"
	@$(UV_VERSION_CHECK)
	uv tool install --force --reinstall --torch-backend=$(TORCH_BACKEND) --overrides overrides.txt "$(INSTALL_SPEC)"
	@$(TOOL_VOXWEAVE) --version
	@echo "reinstalled (git $(GIT_STATE))"

uninstall:
	uv tool uninstall voxweave

# Development environment for code changes, synced like CI: Python 3.13, the platform
# variant ([cuda] or [mps]), the [diarize] extra (so diarization tests run), and dev tools.
# [cuda] and [mps] are mutually exclusive (conflicting transformers pins), so sync
# exactly one; on Apple Silicon use: make dev VARIANT=mps.
dev:
	uv sync --extra $(VARIANT) --extra diarize --dev --python 3.13

# Unit tests (no network). --extra $(VARIANT) --extra diarize keeps this hermetic on a
# fresh clone. Pins LC_ALL=C.UTF-8 with TZ and PYTHONHASHSEED unset to match CI.
test:
	env -u TZ -u PYTHONHASHSEED LC_ALL=C.UTF-8 uv run --extra $(VARIANT) --extra diarize pytest tests/ -v

# Lint / format (project-wide; rules and the vendor exclusion come from [tool.ruff] in
# pyproject.toml). Ruff is pinned so a release that changes its style cannot turn untouched
# code red; CI runs `lint-check` with the same pin. Bump it deliberately, reformatting in
# the same commit.
RUFF = uv run --no-project --with ruff==0.16.9 ruff
lint:
	$(RUFF) check --fix .
	$(RUFF) format .

# The read-only form of `lint` that CI runs.
lint-check:
	$(RUFF) check .
	$(RUFF) format --check .

# Static type check (pyright, basic mode, production code only -- see [tool.pyright]).
# Zero errors is the bar; CI enforces it so type noise cannot accumulate again.
typecheck:
	uv run --extra $(VARIANT) --extra diarize pyright

# ---- Quality rulers ----------------------------------------------------------
# `make test` answers "did behaviour change unintentionally". These answer "is the
# output any good": the segmentation ruler replays a tracked corpus of captured unit
# streams through the production entry point and gates four metrics against a
# recorded baseline. Exit codes: 0 pass, 1 gate regression, 2 invalid corpus/baseline.
# The tracked baseline is always passed, so a missing one is an invalid run (exit 2),
# never a silent downgrade of every gate to warning-only.
SEG_CORPUS ?= calibration/segmentation/corpus.json
SEG_BASELINE ?= calibration/segmentation/baseline.json
SEG_REPORT ?= build/calibration/segmentation-report.json

# `quality` is only the public, zero-GPU, deterministic lane: no media, no model, no
# network, runnable from a bare checkout. The alignment ruler needs private media and
# MFA truth, so it is invoked explicitly and never wired in here.
quality: quality-segmentation

quality-segmentation:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py evaluate \
	  --corpus $(SEG_CORPUS) \
	  --baseline $(SEG_BASELINE) \
	  --json-out $(SEG_REPORT) --check

# Deliberately not part of `quality`, and never run by CI: recording a baseline is a
# reviewed human action, or a regression can be laundered into the new normal.
quality-record-segmentation:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py record-baseline \
	  --corpus $(SEG_CORPUS) \
	  --report $(SEG_REPORT) \
	  --output $(SEG_BASELINE)
