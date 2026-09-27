.PHONY: install reinstall uninstall dev test lint lint-check typecheck \
	quality quality-segmentation quality-shadow-segmentation \
	quality-shadow-segmentation-full \
	quality-p6-oracle \
	quality-record-segmentation

# Install as a global uv tool (end-user mode): puts the voxweave command on PATH.
# Separation / layout / song-skip / diarization / CJK-break / translation support is baked into
# the core deps; the install variant selects the compute platform AND the ASR/alignment backend:
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

# ---- Compatibility extras ----------------------------------------------------
# Diarization is a core dependency. Keep explicit extras available so older source-install
# commands such as EXTRAS=diarize still resolve through the empty compatibility alias.
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

# The interpreter the P6 oracle manifest was recorded under. CI syncs the same pin, and
# the oracle tests refuse any other interpreter; empty when python3 is unavailable.
ORACLE_PYTHON := $(shell python3 -c "import json; print(json.load(open('calibration/p6-oracle/manifest.json'))['execution']['interpreter'].split()[1])" 2>/dev/null)

# Development environment for code changes, synced like CI: the oracle's interpreter pin,
# the [cuda] extra and the dev group. [cuda] and [mps] are mutually exclusive (conflicting
# transformers pins), so sync exactly one; on Apple Silicon use: make dev VARIANT=mps.
# The P6 oracle tests also need the oracle's recorded platform (Linux x86_64) and the full
# git history (a shallow clone lacks its reference commits); elsewhere they fail with a
# message naming the mismatch.
dev:
	uv sync --extra $(VARIANT) --dev $(if $(ORACLE_PYTHON),--python $(ORACLE_PYTHON))

# Unit tests (no network). --extra $(VARIANT) keeps this hermetic on a fresh
# clone; without it the run env lacks the backend deps and imports fail.
test:
	uv run --extra $(VARIANT) pytest tests/ -v

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
	uv run --extra $(VARIANT) pyright

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
SEG_SHADOW_REPORT ?= build/calibration/segmentation-shadow-report.json
P6_ORACLE_MANIFEST ?= calibration/p6-oracle/manifest.json
P6_ORACLE_REPORT ?= build/p6-oracle-report.json
P6_ORACLE_ENV = env -i PATH="$(PATH)" LANG=zh_CN.UTF-8 LC_ALL=C.UTF-8
# The command the oracle runner starts under. Tests pass their own interpreter
# (P6_ORACLE_PYTHON=/path/to/python) so they never re-sync the environment or the lock.
P6_ORACLE_PYTHON ?= uv run --extra $(VARIANT) python

# `quality` is only the public, zero-GPU, deterministic lane: no media, no model, no
# network, runnable from a bare checkout. The alignment ruler needs private media and
# MFA truth, so it is invoked explicitly and never wired in here.
quality: quality-segmentation

quality-segmentation:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py evaluate \
	  --corpus $(SEG_CORPUS) \
	  --baseline $(SEG_BASELINE) \
	  --json-out $(SEG_REPORT) --check

# Canonical P6 oracle entry point. The runner intentionally validates its recorded
# environment; this target supplies that environment explicitly instead of depending on
# the invoking shell's locale, timezone, hash seed, or project-specific variables.
quality-p6-oracle:
	$(P6_ORACLE_ENV) $(P6_ORACLE_PYTHON) scripts/p6_oracle.py validate \
	  --manifest $(P6_ORACLE_MANIFEST)
	$(P6_ORACLE_ENV) $(P6_ORACLE_PYTHON) scripts/p6_oracle.py compare \
	  --manifest $(P6_ORACLE_MANIFEST) --check --json-out $(P6_ORACLE_REPORT)
	$(P6_ORACLE_ENV) $(P6_ORACLE_PYTHON) scripts/p6_oracle.py source-gates \
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
# exhaustive slice (every near-cliff probe of the three cases, plus the ablations) is
# retained below as `quality-shadow-segmentation-full`; an independent frozen-entry-point
# run on 2026-08-28 took 4 h 58 min (the near-cliff probe count has shrunk since), so it
# is never described as the routine lane.
quality-shadow-segmentation:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py shadow \
	  --corpus $(SEG_CORPUS) \
	  --baseline $(SEG_BASELINE) \
	  --no-ablation \
	  --perturb --perturb-mode single_gap --perturb-magnitude 50 \
	  --perturb-near-cliff-only --perturb-max-probes 2 \
	  --perturb-case en-01 --perturb-case ja-01 --perturb-case zh-01 \
	  --json-out $(SEG_SHADOW_REPORT) --check

quality-shadow-segmentation-full:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py shadow \
	  --corpus $(SEG_CORPUS) \
	  --baseline $(SEG_BASELINE) \
	  --perturb --perturb-mode single_gap --perturb-magnitude 50 \
	  --perturb-near-cliff-only \
	  --perturb-case en-01 --perturb-case ja-01 --perturb-case zh-01 \
	  --json-out $(SEG_SHADOW_REPORT) --check

# Deliberately not part of `quality`, and never run by CI: recording a baseline is a
# reviewed human action, or a regression can be laundered into the new normal.
quality-record-segmentation:
	uv run --extra $(VARIANT) python scripts/calib_segmentation.py record-baseline \
	  --corpus $(SEG_CORPUS) \
	  --report $(SEG_REPORT) \
	  --output $(SEG_BASELINE)
