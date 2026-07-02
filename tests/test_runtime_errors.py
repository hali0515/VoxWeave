"""RED tests for #22: runtime._hf_download / _hf_snapshot must wrap raw
huggingface_hub exceptions into a readable RuntimeError (repo id + HF_TOKEN hint),
instead of letting library-internal exceptions (OSError, etc.) escape bare.
"""

import pytest

from voxweave import runtime


def test_hf_download_wraps_underlying_error_with_repo_id(monkeypatch):
    def boom(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(RuntimeError, match="acme/repo"):
        runtime._hf_download("acme/repo", "model.bin")


def test_hf_download_error_hints_hf_token(monkeypatch):
    def boom(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        runtime._hf_download("acme/repo", "model.bin")


def test_hf_snapshot_wraps_underlying_error_with_repo_id(monkeypatch, tmp_path):
    def boom(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    with pytest.raises(RuntimeError, match="acme/repo"):
        runtime._hf_snapshot("acme/repo", str(tmp_path))


def test_hf_snapshot_error_hints_hf_token(monkeypatch, tmp_path):
    def boom(*a, **kw):
        raise OSError("network down")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        runtime._hf_snapshot("acme/repo", str(tmp_path))
