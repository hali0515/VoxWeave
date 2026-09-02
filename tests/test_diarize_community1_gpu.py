"""GPU end-to-end test for the Community-1 diarization pipeline.

Opt-in gates (all unset by default, so this file is a no-op in CI):

- ``VOXWEAVE_RUN_DIARIZE_C1_GPU_E2E=1`` runs the test at all; otherwise it
  is skipped.
- ``VOXWEAVE_DIARIZE_C1_E2E_WAV`` points at a local 16kHz Japanese WAV
  fixture with multiple speakers; the test is skipped with an explicit
  reason naming this variable when it is unset or the file is missing.
- ``VOXWEAVE_DIARIZE_C1_E2E_WAV_SHA256`` is an optional sha256 anchor for
  that fixture; when set, the test asserts the file's digest matches it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path

import pytest

from voxweave import config, diarize


GPU_E2E_ENV = "VOXWEAVE_RUN_DIARIZE_C1_GPU_E2E"
WAV_PATH_ENV = "VOXWEAVE_DIARIZE_C1_E2E_WAV"
WAV_SHA256_ENV = "VOXWEAVE_DIARIZE_C1_E2E_WAV_SHA256"
COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"


@pytest.mark.skipif(
    os.environ.get(GPU_E2E_ENV) != "1",
    reason=f"set {GPU_E2E_ENV}=1 to run the Community-1 GPU E2E",
)
def test_community1_public_diarize_turns_gpu_with_embeddings() -> None:
    import torch

    assert torch.cuda.is_available(), "the Community-1 E2E requires a CUDA GPU"
    wav_path = os.environ.get(WAV_PATH_ENV)
    if not wav_path:
        pytest.skip(f"set {WAV_PATH_ENV} to a Japanese multi-speaker WAV fixture")
    ja_wav = Path(wav_path)
    if not ja_wav.is_file():
        pytest.skip(f"{WAV_PATH_ENV} points at a missing file: {ja_wav}")
    expected_sha256 = os.environ.get(WAV_SHA256_ENV)
    if expected_sha256:
        actual_sha256 = hashlib.sha256(ja_wav.read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256, (
            f"{WAV_PATH_ENV} content changed: expected sha256 "
            f"{expected_sha256}, got {actual_sha256}"
        )
    token = config.conf_hf_token()
    assert token, "authenticate with `hf auth login` or set VOXWEAVE_HF_TOKEN/HF_TOKEN"

    diarize.release()
    try:
        result = diarize.diarize_turns(
            ja_wav,
            token=token,
            model="community-1",
            min_speakers=1,
            max_speakers=10,
            want_embeddings=True,
        )
    finally:
        diarize.release()

    assert result.turns
    speakers = {label for _start, _end, label in result.turns}
    assert speakers
    assert all(
        math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end and label
        for start, end, label in result.turns
    )

    assert result.centroids
    assert set(result.centroids) <= speakers
    assert all(len(vector) == 256 for vector in result.centroids.values())
    assert all(
        math.sqrt(math.fsum(value * value for value in vector))
        == pytest.approx(1.0, abs=1e-6)
        for vector in result.centroids.values()
    )

    provenance = result.provenance
    assert provenance["diarization_model"] == COMMUNITY_MODEL
    assert provenance["pyannote_version"] == importlib.metadata.version(
        "pyannote-audio"
    )
    assert provenance["embedding_dim"] == 256
    assert provenance["embedding_model"] != "unresolved"
    assert provenance["embedding_checkpoint"] != "unresolved"
    assert provenance["outer_config_sha256"] != "unresolved"
