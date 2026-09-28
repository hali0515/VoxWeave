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


@pytest.mark.parametrize(
    ("target", "call"),
    [
        ("huggingface_hub.hf_hub_download", lambda: runtime._hf_download("a/r", "f")),
        ("huggingface_hub.snapshot_download", lambda: runtime._hf_snapshot("a/r", "c")),
    ],
)
def test_hf_helpers_pass_the_configured_token(monkeypatch, tmp_path, target, call):
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return "/local"

    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.setenv("VOXWEAVE_HF_TOKEN", "hf_from_voxweave_env")
    monkeypatch.setattr(target, fake)
    assert call() == "/local"
    assert seen["token"] == "hf_from_voxweave_env"


def test_missing_dependency_hint_names_the_pypi_install_first():
    hint = str(runtime._require("qwen_asr"))
    assert hint.index("uv tool install") < hint.index("make install")
    assert "qwen_asr" in hint
