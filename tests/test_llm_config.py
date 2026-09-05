# tests/test_llm_config.py
# The [llm] config section for translate/correct: model / base_url / api_key_env resolution
# (CLI > env > conf > built-in), the keyless-endpoint contract, the "auto" served-model
# probe, and the JSON salvage that survives vLLM's guided-decoding artifact.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from voxweave import asrfix, config, pipeline, translate
from voxweave.cli import cli


@pytest.fixture
def conf_at(tmp_path, monkeypatch):
    p = tmp_path / "voxweave.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(p))
    for name in (
        "VOXWEAVE_TRANSLATE_MODEL",
        "VOXWEAVE_FIX_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return p


# --- resolution precedence -----------------------------------------------------


def test_builtin_llm_default_is_a_pinned_live_model(conf_at):
    # The previous default (gpt-5.3-chat-latest) was a floating alias that OpenAI
    # retired; the built-in must be a concrete model name, never a "-latest" alias.
    assert config.DEFAULT_LLM_MODEL
    assert "latest" not in config.DEFAULT_LLM_MODEL
    assert config.resolve_llm_model(None, task_envvar="VOXWEAVE_TRANSLATE_MODEL") == (
        config.DEFAULT_LLM_MODEL
    )


def test_conf_llm_model_used_when_cli_and_env_absent(conf_at):
    conf_at.write_text('[llm]\nmodel = "Qwen3.8-27B-FP8"\n', encoding="utf-8")
    assert (
        config.resolve_llm_model(None, task_envvar="VOXWEAVE_TRANSLATE_MODEL")
        == "Qwen3.8-27B-FP8"
    )
    assert (
        config.resolve_llm_model(None, task_envvar="VOXWEAVE_FIX_MODEL")
        == "Qwen3.8-27B-FP8"
    )


def test_task_env_beats_conf_and_cli_beats_env(conf_at, monkeypatch):
    conf_at.write_text('[llm]\nmodel = "conf-model"\n', encoding="utf-8")
    monkeypatch.setenv("VOXWEAVE_TRANSLATE_MODEL", "env-model")
    assert (
        config.resolve_llm_model(None, task_envvar="VOXWEAVE_TRANSLATE_MODEL")
        == "env-model"
    )
    # The other task's env var does not leak across.
    assert (
        config.resolve_llm_model(None, task_envvar="VOXWEAVE_FIX_MODEL") == "conf-model"
    )
    assert (
        config.resolve_llm_model("cli-model", task_envvar="VOXWEAVE_TRANSLATE_MODEL")
        == "cli-model"
    )


def test_base_url_precedence(conf_at, monkeypatch):
    assert config.resolve_llm_base_url(None) is None
    conf_at.write_text(
        '[llm]\nbase_url = "http://100.88.155.80:1234/v1"\n', encoding="utf-8"
    )
    assert config.resolve_llm_base_url(None) == "http://100.88.155.80:1234/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env:1/v1")
    assert config.resolve_llm_base_url(None) == "http://env:1/v1"
    assert config.resolve_llm_base_url("http://cli:2/v1") == "http://cli:2/v1"


def test_api_key_env_default_and_keyless_endpoint(conf_at):
    assert config.resolve_llm_api_key_env(None) == "OPENAI_API_KEY"
    conf_at.write_text('[llm]\napi_key_env = "MY_KEY"\n', encoding="utf-8")
    assert config.resolve_llm_api_key_env(None) == "MY_KEY"
    # An explicit empty string declares the endpoint keyless (local vLLM).
    conf_at.write_text('[llm]\napi_key_env = ""\n', encoding="utf-8")
    assert config.resolve_llm_api_key_env(None) == ""
    assert config.resolve_llm_api_key_env("CLI_KEY") == "CLI_KEY"


def test_llm_section_is_a_known_key_and_in_template(conf_at, caplog):
    conf_at.write_text('[llm]\nmodel = "x"\n', encoding="utf-8")
    with caplog.at_level("WARNING", logger="voxweave"):
        config._load()
    assert "unknown config key" not in caplog.text
    assert "[llm]" in config._TEMPLATE
    assert "gpt-5.3" not in config._TEMPLATE


def test_llm_wrong_types_fall_back(conf_at, caplog):
    conf_at.write_text("[llm]\nmodel = 3\nbase_url = 1\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="voxweave"):
        assert (
            config.resolve_llm_model(None, task_envvar="VOXWEAVE_FIX_MODEL")
            == config.DEFAULT_LLM_MODEL
        )
        assert config.resolve_llm_base_url(None) is None
    assert "wrong type" in caplog.text


# --- "auto": use the endpoint's only served model ----------------------------


def _fake_client(ids):
    return SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: [SimpleNamespace(id=i) for i in ids],
        )
    )


def test_resolve_model_auto_uses_single_served_model():
    assert (
        translate.resolve_model(_fake_client(["Qwen3.8-27B-FP8"]), "auto")
        == "Qwen3.8-27B-FP8"
    )


def test_resolve_model_auto_refuses_ambiguous_endpoint():
    with pytest.raises(RuntimeError, match="serves 2 models"):
        translate.resolve_model(_fake_client(["a", "b"]), "auto")
    with pytest.raises(RuntimeError, match="serves no models"):
        translate.resolve_model(_fake_client([]), "auto")


def test_resolve_model_passthrough_never_probes():
    client = SimpleNamespace()  # no .models attribute at all
    assert translate.resolve_model(client, "gpt-5.5") == "gpt-5.5"


def test_translate_cues_resolves_auto_before_calling(monkeypatch):
    seen = {}

    def fake_call(client, model, messages, on_entry=None):
        seen["model"] = model
        return '{"translations": [{"i": 0, "t": "早上好"}]}'

    monkeypatch.setattr(translate, "_call", fake_call)
    out = translate.translate_cues(
        [{"i": 0, "t": "good morning"}],
        to="zh",
        model="auto",
        client=_fake_client(["Qwen3.8-27B-FP8"]),
    )
    assert seen["model"] == "Qwen3.8-27B-FP8"
    assert out == {0: "早上好"}


def test_correct_cues_resolves_auto_before_calling(monkeypatch):
    seen = {}

    def fake_call(client, model, messages, on_entry=None):
        seen["model"] = model
        return '{"fixes": []}'

    monkeypatch.setattr(asrfix, "_call", fake_call)
    asrfix.correct_cues(
        [{"i": 0, "t": "hi"}],
        model="auto",
        client=_fake_client(["Qwen3.8-27B-FP8"]),
    )
    assert seen["model"] == "Qwen3.8-27B-FP8"


# --- salvage survives vLLM guided-decoding artifact ----------------------------


def test_salvage_skips_spurious_object_prefix():
    # vLLM json_object / json_schema guided decoding on Qwen3.8 emits a stray '{"'
    # before the real object; the first '{' therefore never decodes.
    raw = '{"{"translations":[{"i":0,"t":"早上好"}]}'
    assert translate.parse_response(raw) == {0: "早上好"}
    assert asrfix.parse_fixes('{"{"fixes":[]}') == []


def test_salvage_still_returns_empty_for_garbage():
    assert translate.parse_response('{"{"{"nope') == {}
    assert translate.parse_response("{ { {") == {}


# --- CLI wiring ------------------------------------------------------------------


def _vtt(tmp_path):
    p = tmp_path / "ep.vtt"
    p.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n", encoding="utf-8")
    return p


def test_cli_translate_uses_conf_llm_without_openai_key(conf_at, tmp_path):
    conf_at.write_text(
        '[llm]\nmodel = "Qwen3.8-27B-FP8"\n'
        'base_url = "http://100.88.155.80:1234/v1"\napi_key_env = ""\n',
        encoding="utf-8",
    )
    captured = {}
    with patch.object(
        pipeline,
        "translate",
        lambda path, **kw: captured.update(kw) or (tmp_path / "ep.zh.vtt"),
    ):
        res = CliRunner().invoke(cli, ["translate", str(_vtt(tmp_path))])
    assert res.exit_code == 0, res.output
    assert captured["model"] == "Qwen3.8-27B-FP8"
    assert captured["base_url"] == "http://100.88.155.80:1234/v1"
    assert captured["api_key"]  # a placeholder key, never empty


def test_cli_correct_uses_conf_llm_without_openai_key(conf_at, tmp_path):
    conf_at.write_text(
        '[llm]\nmodel = "auto"\nbase_url = "http://h:1/v1"\napi_key_env = ""\n',
        encoding="utf-8",
    )
    captured = {}
    with patch.object(
        pipeline,
        "correct",
        lambda path, **kw: (
            captured.update(kw)
            or {
                "out": tmp_path / "ep.asrfix.vtt",
                "audit": None,
                "applied": [],
                "rejected": [],
                "n_cues": 1,
                "applied_in_place": False,
                "aligned": False,
            }
        ),
    ):
        res = CliRunner().invoke(cli, ["correct", str(_vtt(tmp_path))])
    assert res.exit_code == 0, res.output
    assert captured["model"] == "auto"
    assert captured["base_url"] == "http://h:1/v1"


def test_cli_translate_still_requires_key_for_openai(conf_at, tmp_path):
    res = CliRunner().invoke(cli, ["translate", str(_vtt(tmp_path))])
    assert res.exit_code == 1
    assert "OPENAI_API_KEY" in res.output


def test_cli_llm_help_names_the_live_default():
    for cmd in ("translate", "correct"):
        res = CliRunner().invoke(cli, [cmd, "--help"])
        assert res.exit_code == 0
        assert config.DEFAULT_LLM_MODEL in res.output
        assert "gpt-5.3" not in res.output
        assert "[llm]" in res.output


def test_readme_no_longer_documents_the_retired_model():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "gpt-5.3-chat-latest" not in text
    assert "[llm]" in text


def test_module_constants_never_read_the_user_conf(conf_at):
    # Import-time snapshots of the user conf were the original sin: a conf with
    # model = "auto" leaked into every library call and test. The module-level
    # names stay built-in; env / conf resolve at call time in the CLI and pipeline.
    conf_at.write_text('[llm]\nmodel = "auto"\n', encoding="utf-8")
    assert translate.TRANSLATE_MODEL == config.DEFAULT_LLM_MODEL
    assert asrfix.FIX_MODEL == config.DEFAULT_LLM_MODEL
    assert config.resolve_llm_model(None, task_envvar="VOXWEAVE_FIX_MODEL") == "auto"
