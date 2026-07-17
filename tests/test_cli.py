from unittest.mock import patch

import pytest
from click.testing import CliRunner

from voxweave import pipeline
from voxweave.cli import cli


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    # CLI tests must not read the developer's real ~/.config/voxweave.conf (non-hermetic: user settings
    # like asr_model would pollute default-value assertions). Point to an empty tmp path so
    # ensure_default_config writes the commented template (asr_model commented out) and
    # conf_asr_model() returns None, exercising the built-in defaults.
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))


def _media(tmp_path):
    m = tmp_path / "a.wav"
    m.write_bytes(b"x")
    out = tmp_path / "a.vtt"
    out.write_text("WEBVTT\n", encoding="utf-8")
    return m, out


def test_process_default_separate(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is True


def test_process_no_separate(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--no-separate", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is False


def test_process_debug_flag(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--debug", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["debug"] is True


def test_process_subcommand_name_removed(tmp_path):
    # `voxweave process <media>` is no longer a valid invocation (process subcommand has been removed)
    media, _ = _media(tmp_path)
    with patch("voxweave.pipeline.process") as m:
        r = CliRunner().invoke(cli, ["process", str(media)])
    assert r.exit_code != 0
    assert not m.called


def test_process_debug_default_off(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["debug"] is False


def test_process_normalize_flag(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--normalize", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["normalize"] is True


def test_process_normalize_default_off(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["normalize"] is False


def test_process_skip_songs_default_on(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["skip_songs"] is True


def test_process_no_skip_songs(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--no-skip-songs", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["skip_songs"] is False


def test_process_default_asr_model(tmp_path):
    # bare voxweave: no --model override -> asr_model None -> backend uses its default (0.6B)
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["asr_model"] is None  # no override -> backend uses 0.6B


def test_process_model_override(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--model", "qwen3-asr-1.7B", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["asr_model"] == "qwen3-asr-1.7B"


def test_media_shorthand_routes_to_process(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.called


def test_split_passes_kwargs(tmp_path):
    j = tmp_path / "a.json"
    j.write_text("{}", encoding="utf-8")
    out = tmp_path / "a.vtt"
    with patch("voxweave.pipeline.split", return_value=out) as m:
        r = CliRunner().invoke(cli, ["split", str(j), "--max-lines", "2"])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs == {
        "max_lines": 2,
        "timestamps": True,
        "semantic_split": False,
        "semantic_model": "Qwen/Qwen3.5-0.8B",
    }


def test_process_semantic_split_is_optional_and_selects_default_model(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--semantic-split", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["semantic_split"] is True
    assert m.call_args.kwargs["semantic_model"] == "Qwen/Qwen3.5-0.8B"


def test_process_semantic_model_override_does_not_enable_feature(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--semantic-model", "local/custom", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["semantic_split"] is False
    assert m.call_args.kwargs["semantic_model"] == "local/custom"


def test_split_semantic_flags_pass_through(tmp_path):
    j = tmp_path / "a.json"
    j.write_text("{}", encoding="utf-8")
    out = tmp_path / "a.vtt"
    with patch("voxweave.pipeline.split", return_value=out) as m:
        r = CliRunner().invoke(
            cli,
            [
                "split",
                str(j),
                "--semantic-split",
                "--semantic-model",
                "local/custom",
            ],
        )
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["semantic_split"] is True
    assert m.call_args.kwargs["semantic_model"] == "local/custom"


def test_process_error_renders_panel_and_exits(tmp_path):
    media, _ = _media(tmp_path)
    with patch("voxweave.pipeline.process", side_effect=RuntimeError("boom")):
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 1


def _vtt(tmp_path):
    v = tmp_path / "a.vtt"
    v.write_text("WEBVTT\n\nhi\n", encoding="utf-8")
    return v


def test_align_missing_arg_errors():
    # not a stub: missing VTT argument -> click usage error
    r = CliRunner().invoke(cli, ["align"])
    assert r.exit_code == 2


def test_align_default_separate(tmp_path):
    v = _vtt(tmp_path)
    with patch("voxweave.pipeline.align", return_value=v) as m:
        r = CliRunner().invoke(cli, ["align", str(v)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is True
    assert m.call_args.kwargs["media_path"] is None


def test_align_no_separate(tmp_path):
    v = _vtt(tmp_path)
    with patch("voxweave.pipeline.align", return_value=v) as m:
        r = CliRunner().invoke(cli, ["align", "--no-separate", str(v)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is False


def test_align_media_override(tmp_path):
    v = _vtt(tmp_path)
    media = tmp_path / "a.mkv"
    media.write_bytes(b"x")
    with patch("voxweave.pipeline.align", return_value=v) as m:
        r = CliRunner().invoke(cli, ["align", "--media", str(media), str(v)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["media_path"] == media


def test_cli_translate_invokes_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n", encoding="utf-8")
    captured = {}

    def fake_translate(path, **kw):
        captured["path"] = path
        captured["to"] = kw.get("to")
        captured["model"] = kw.get("model")
        return tmp_path / "ep.zh.vtt"

    monkeypatch.setattr(pipeline, "translate", fake_translate)
    runner = CliRunner()
    res = runner.invoke(cli, ["translate", str(vtt), "--to", "zh"])
    assert res.exit_code == 0, res.output
    assert captured["to"] == "zh"
    assert captured["path"] == vtt


def test_cli_translate_missing_api_key_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n", encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(cli, ["translate", str(vtt)])
    assert res.exit_code == 1


def test_cli_translate_loads_glossary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n", encoding="utf-8")
    g = tmp_path / "g.json"
    g.write_text('{"A": "甲"}', encoding="utf-8")
    captured = {}
    monkeypatch.setattr(
        pipeline,
        "translate",
        lambda path, **kw: captured.update(kw) or (tmp_path / "ep.zh.vtt"),
    )
    runner = CliRunner()
    res = runner.invoke(cli, ["translate", str(vtt), "--glossary", str(g)])
    assert res.exit_code == 0, res.output
    assert captured["glossary"] == {"A": "甲"}


# --- conf [defaults] flag resolution (CLI flag > conf > builtin) ---


def _write_conf(tmp_path, body):
    (tmp_path / "voxweave.conf").write_text(body, encoding="utf-8")


def test_process_conf_default_separate_off(tmp_path):
    _write_conf(tmp_path, "[defaults]\nseparate = false\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is False


def test_process_cli_flag_beats_conf_default(tmp_path):
    _write_conf(tmp_path, "[defaults]\nseparate = false\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--separate", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is True


def test_process_conf_default_normalize_on(tmp_path):
    _write_conf(tmp_path, "[defaults]\nnormalize = true\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["normalize"] is True


def test_process_conf_default_semantic_split_on_and_cli_can_disable(tmp_path):
    _write_conf(tmp_path, "[defaults]\nsemantic_split = true\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["semantic_split"] is True

    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--no-semantic-split", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["semantic_split"] is False


def test_process_no_normalize_beats_conf_default(tmp_path):
    _write_conf(tmp_path, "[defaults]\nnormalize = true\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--no-normalize", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["normalize"] is False


def test_align_conf_default_separate_off(tmp_path):
    _write_conf(tmp_path, "[defaults]\nseparate = false\n")
    v = _vtt(tmp_path)
    with patch("voxweave.pipeline.align", return_value=v) as m:
        r = CliRunner().invoke(cli, ["align", str(v)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is False


def test_split_conf_default_timestamps_off(tmp_path):
    _write_conf(tmp_path, "[defaults]\ntimestamps = false\n")
    j = tmp_path / "a.json"
    j.write_text("{}", encoding="utf-8")
    with patch("voxweave.pipeline.split", return_value=tmp_path / "a.vtt") as m:
        r = CliRunner().invoke(cli, ["split", str(j)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["timestamps"] is False


def test_process_conf_default_vad_mask_sets_env(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("VOXWEAVE_VAD_EMISSION_MASK", raising=False)
    _write_conf(tmp_path, "[defaults]\nvad_mask = true\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out):
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert os.environ.get("VOXWEAVE_VAD_EMISSION_MASK") == "1"


def test_process_no_vad_mask_beats_env(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out):
        r = CliRunner().invoke(cli, ["--no-vad-mask", str(media)])
    assert r.exit_code == 0, r.output
    assert os.environ.get("VOXWEAVE_VAD_EMISSION_MASK") == "0"


def test_process_env_vad_mask_beats_conf_off(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    _write_conf(tmp_path, "[defaults]\nvad_mask = false\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out):
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert os.environ.get("VOXWEAVE_VAD_EMISSION_MASK") == "1"


def test_process_conf_default_diarize_on(tmp_path):
    _write_conf(tmp_path, "[defaults]\ndiarize = true\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["diarize"] is True


def test_process_no_diarize_beats_conf_default(tmp_path):
    _write_conf(tmp_path, "[defaults]\ndiarize = true\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--no-diarize", str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["diarize"] is False


def test_process_diarize_default_off(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["diarize"] is False
