from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path

import pytest

from voxweave import config, diarize


GPU_E2E_ENV = "VOXWEAVE_RUN_DIARIZE_C1_GPU_E2E"
COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
JA_WAV = Path(
    "/tmp/claude-1000/-mnt-Dev-Git-qsub/"
    "76be7f61-2a96-4094-80b0-3b9311d0223b/scratchpad/"
    "diarize-ab/wav/ja.16k.wav"
)
JA_WAV_SHA256 = "22be7401701b7d1fb317607dde25c4a251a767a136c853794b4d616aaf2c84e5"


@pytest.mark.skipif(
    os.environ.get(GPU_E2E_ENV) != "1",
    reason=f"set {GPU_E2E_ENV}=1 to run the Community-1 GPU E2E",
)
def test_community1_public_diarize_turns_gpu_with_embeddings() -> None:
    import torch

    assert torch.cuda.is_available(), "the Community-1 E2E requires a CUDA GPU"
    assert JA_WAV.is_file(), f"frozen Japanese fixture is missing: {JA_WAV}"
    assert hashlib.sha256(JA_WAV.read_bytes()).hexdigest() == JA_WAV_SHA256
    token = config.conf_hf_token()
    assert token, "authenticate with `hf auth login` or set VOXWEAVE_HF_TOKEN/HF_TOKEN"

    diarize.release()
    try:
        result = diarize.diarize_turns(
            JA_WAV,
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
