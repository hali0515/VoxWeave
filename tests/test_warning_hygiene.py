"""Runtime warning hygiene: pyannote TF32 handling and jieba import noise.

The diarization span must pre-comply with pyannote's reproducibility guard
(so it never warns) and restore the process float32-matmul policy afterwards
(so pyannote cannot silently disable the separator's TF32 opt-in). The jieba
import sites must not surface setuptools' pkg_resources deprecation warning.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest
import soundfile as sf

from voxweave import diarize
from voxweave.core.breakpoints import quiet_import_jieba


class _FakeAnnotation:
    def itertracks(self, yield_label: bool = False):
        return iter(())


class _FakePipeline:
    def __init__(self, *, warn: bool = False, raise_error: bool = False):
        self.warn = warn
        self.raise_error = raise_error
        self.seen_matmul_tf32: bool | None = None
        self.seen_cudnn_tf32: bool | None = None

    def __call__(self, payload, **kwargs):
        import torch

        self.seen_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        self.seen_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        if self.raise_error:
            raise RuntimeError("boom")
        if self.warn:
            warnings.warn(
                "std(): degrees of freedom is <= 0. Correction should be strictly"
                " less than the reduction factor",
                UserWarning,
                stacklevel=1,
            )
            warnings.warn(
                "TensorFloat-32 (TF32) has been disabled as it might lead to"
                " reproducibility issues and lower accuracy.",
                UserWarning,
                stacklevel=1,
            )
        return _FakeAnnotation()


@pytest.fixture
def tiny_wav(tmp_path):
    path = tmp_path / "tiny.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
    return path


@pytest.fixture
def fake_pipeline(monkeypatch):
    def install(pl):
        monkeypatch.setattr(diarize, "_pipeline", pl)
        return pl

    yield install


def test_diarize_disables_tf32_for_inference_and_restores(
    tiny_wav, fake_pipeline
) -> None:
    import torch

    pl = fake_pipeline(_FakePipeline())
    prev_precision = torch.get_float32_matmul_precision()
    prev_cudnn = bool(torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    try:
        turns = diarize.diarize_turns(tiny_wav, token="dummy")
        assert turns == []
        assert pl.seen_matmul_tf32 is False
        assert pl.seen_cudnn_tf32 is False
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def test_diarize_restores_tf32_policy_when_pipeline_raises(
    tiny_wav, fake_pipeline
) -> None:
    import torch

    fake_pipeline(_FakePipeline(raise_error=True))
    prev_precision = torch.get_float32_matmul_precision()
    prev_cudnn = bool(torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    try:
        with pytest.raises(RuntimeError, match="boom"):
            diarize.diarize_turns(tiny_wav, token="dummy")
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def test_diarize_swallows_known_pipeline_warnings(tiny_wav, fake_pipeline) -> None:
    fake_pipeline(_FakePipeline(warn=True))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        diarize.diarize_turns(tiny_wav, token="dummy")
    noisy = [
        r
        for r in records
        if "degrees of freedom" in str(r.message) or "TensorFloat-32" in str(r.message)
    ]
    assert noisy == []


def test_quiet_import_jieba_returns_working_modules() -> None:
    jieba = quiet_import_jieba()
    assert jieba is not None
    assert list(jieba.cut("数据中心"))
    pseg = quiet_import_jieba(posseg=True)
    assert pseg is not None
    assert [pair.word for pair in pseg.cut("数据中心")]


def test_quiet_import_jieba_fresh_interpreter_with_warning_as_error() -> None:
    """A cold import through the helper must not surface the pkg_resources warning.

    -W turns that exact warning into an error, so an unfiltered import path
    fails loudly here even though a warm test process would stay silent.
    """
    code = (
        "from voxweave.core.breakpoints import quiet_import_jieba\n"
        "m = quiet_import_jieba(posseg=True)\n"
        "raise SystemExit(0 if m is not None else 3)\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error:pkg_resources is deprecated:UserWarning",
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
