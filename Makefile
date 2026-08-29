.PHONY: install reinstall uninstall dev test lint typecheck \
	quality quality-segmentation quality-shadow-segmentation \
	quality-shadow-segmentation-full \
	quality-p6-oracle \
	quality-record-segmentation

# Install as a global uv tool (end-user mode): puts the voxweave command on PATH.
# Separation / layout / song-skip / diarization / CJK-break / translation support is baked into
# the core deps; the install variant selects the compute platform AND the ASR/alignment backend:
#   VARIANT=cuda (default) -> NVIDIA/Linux: torch Qwen3-ASR+aligner (qwen-asr) + onnxruntime-gpu +
#                             faster-whisper, on the cu128 torch wheel (Blackwell sm_120, no auto-detect)
#   VARIANT=mps            -> Apple Silicon/macOS: native MLX Qwen3-ASR+aligner (mlx-audio) on the
#                             default torch wheel (MPS built in for the separator; no whisper engine)
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

# ---- Compatibility extras ----------------------------------------------------
# Diarization is a core dependency. Keep explicit extras available so older source-install
# commands such as EXTRAS=diarize still resolve through the empty compatibility alias.
EXTRAS ?=
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

# ---- Quality rulers ----------------------------------------------------------
# `make test` answers "did behaviour change unintentionally". These answer "is the
# output any good": the segmentation ruler replays a tracked corpus of captured unit
# streams through the production entry point and gates four metrics against a
# recorded baseline. Exit codes: 0 pass, 1 gate regression, 2 invalid corpus/baseline.
SEG_CORPUS ?= calibration/segmentation/corpus.json
SEG_BASELINE ?= calibration/segmentation/baseline.json
SEG_REPORT ?= build/calibration/segmentation-report.json
SEG_SHADOW_REPORT ?= build/calibration/segmentation-shadow-report.json
P6_ORACLE_MANIFEST ?= calibration/p6-oracle/manifest.json
P6_ORACLE_REPORT ?= build/p6-oracle-report.json
P6_ORACLE_ENV = env -i PATH="$(PATH)" LANG=zh_CN.UTF-8 LC_ALL=C.UTF-8

# `quality` is only the public, zero-GPU, deterministic lane: no media, no model, no
# network, runnable from a bare checkout. The alignment ruler needs private media and
# MFA truth, so it is invoked explicitly and never wired in here.
quality: quality-segmentation

quality-segmentation:
	uv run python scripts/calib_segmentation.py evaluate \
	  --corpus $(SEG_CORPUS) \
	  $(if $(wildcard $(SEG_BASELINE)),--baseline $(SEG_BASELINE),) \
	  --json-out $(SEG_REPORT) --check

# Canonical P6 oracle entry point. The runner intentionally validates its recorded
# environment; this target supplies that environment explicitly instead of depending on
# the invoking shell's locale, timezone, hash seed, or project-specific variables.
quality-p6-oracle:
	$(P6_ORACLE_ENV) uv run --extra $(VARIANT) python scripts/p6_oracle.py validate \
	  --manifest $(P6_ORACLE_MANIFEST)
	$(P6_ORACLE_ENV) uv run --extra $(VARIANT) python scripts/p6_oracle.py compare \
	  --manifest $(P6_ORACLE_MANIFEST) --check --json-out $(P6_ORACLE_REPORT)
	$(P6_ORACLE_ENV) uv run --extra $(VARIANT) python scripts/p6_oracle.py source-gates \
	  --manifest $(P6_ORACLE_MANIFEST) --check

# P5: full optimizer/finalizer/speaker shadow matrix beside the shipped v1 answer.
# Same corpus, same baseline, same environment as `quality-segmentation` -- the
# non-inferiority numbers are only comparable against a baseline recorded here.
# Deliberately NOT part of `quality`: the shadow ships nothing, so a v2 regression
# during soak must not block a PR that changed neither engine.
#
# The routine lane is an explicitly bounded smoke slice: base corpus + coarse
# gates + two near-cliff probes in one case per language, with ablation skipped.
# It keeps AD-2's exit driver live and is budgeted in minutes, not hours. The
# exhaustive historical slice (1,551 probes plus 14 ablations) is retained below
# as `quality-shadow-segmentation-full`; an independent frozen-entry-point run on
# 2026-08-28 took 4 h 58 min, so it is never described as the routine lane.
quality-shadow-segmentation:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py shadow \
	  --corpus $(SEG_CORPUS) \
	  $(if $(wildcard $(SEG_BASELINE)),--baseline $(SEG_BASELINE),) \
	  --no-ablation \
	  --perturb --perturb-mode single_gap --perturb-magnitude 50 \
	  --perturb-near-cliff-only --perturb-max-probes 2 \
	  --perturb-case en-01 --perturb-case ja-01 --perturb-case zh-01 \
	  --json-out $(SEG_SHADOW_REPORT) --check

quality-shadow-segmentation-full:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py shadow \
	  --corpus $(SEG_CORPUS) \
	  $(if $(wildcard $(SEG_BASELINE)),--baseline $(SEG_BASELINE),) \
	  --perturb --perturb-mode single_gap --perturb-magnitude 50 \
	  --perturb-near-cliff-only \
	  --perturb-case en-01 --perturb-case ja-01 --perturb-case zh-01 \
	  --json-out $(SEG_SHADOW_REPORT) --check

# Deliberately not part of `quality`, and never run by CI: recording a baseline is a
# reviewed human action, or a regression can be laundered into the new normal.
quality-record-segmentation:
	uv run python scripts/calib_segmentation.py record-baseline \
	  --corpus $(SEG_CORPUS) \
	  --report $(SEG_REPORT) \
	  --output $(SEG_BASELINE)
