from unittest.mock import patch

import pytest
from click.testing import CliRunner

from voxweave import pipeline
from voxweave.cli import cli
from voxweave.ui import RichReporter


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
    reporter = m.call_args.kwargs["reporter"]
    assert isinstance(reporter, RichReporter)
    assert m.call_args.kwargs == {
        "max_lines": 2,
        "timestamps": True,
        "reporter": reporter,
    }


@pytest.mark.parametrize("flag", ["--semantic-split", "--no-semantic-split"])
def test_process_semantic_split_flag_is_removed(tmp_path, flag):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [flag, str(media)])
    assert r.exit_code == 2
    assert not m.called


def test_process_semantic_model_flag_is_removed(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--semantic-model", "local/custom", str(media)])
    assert r.exit_code == 2
    assert not m.called


@pytest.mark.parametrize(
    "args",
    [
        ["--semantic-split"],
        ["--no-semantic-split"],
        ["--semantic-model", "local/custom"],
    ],
)
def test_split_semantic_flags_are_removed(tmp_path, args):
    j = tmp_path / "a.json"
    j.write_text("{}", encoding="utf-8")
    out = tmp_path / "a.vtt"
    with patch("voxweave.pipeline.split", return_value=out) as m:
        r = CliRunner().invoke(cli, ["split", str(j), *args])
    assert r.exit_code == 2
    assert not m.called


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


def test_process_ignores_retired_semantic_split_config(tmp_path):
    _write_conf(tmp_path, "[defaults]\nsemantic_split = true\n")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert "semantic_split" not in m.call_args.kwargs
    assert "semantic_model" not in m.call_args.kwargs


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


def _flat(text: str) -> str:
    """Collapse rich's wrapping so log lines can be matched as one sentence."""
    return " ".join(text.split())


def _correct_result(v):
    return {
        "out": v,
        "audit": None,
        "applied": [],
        "rejected": [],
        "n_cues": 1,
        "applied_in_place": True,
        "aligned": False,
    }


def test_correct_apply_realign_honours_conf_defaults(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "")  # restored after the test
    _write_conf(
        tmp_path, "[defaults]\nseparate = false\nnormalize = true\nvad_mask = true\n"
    )
    v = _vtt(tmp_path)
    with patch("voxweave.pipeline.correct", return_value=_correct_result(v)) as m:
        r = CliRunner().invoke(cli, ["correct", "--apply", str(v)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["separate"] is False
    assert m.call_args.kwargs["normalize"] is True
    assert os.environ["VOXWEAVE_VAD_EMISSION_MASK"] == "1"


@pytest.mark.parametrize("command", ["translate", "correct"])
def test_bad_glossary_renders_error_panel(tmp_path, monkeypatch, command):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    v = _vtt(tmp_path)
    g = tmp_path / "g.json"
    g.write_text("{not json", encoding="utf-8")
    with patch(f"voxweave.pipeline.{command}") as m:
        r = CliRunner().invoke(cli, [command, str(v), "--glossary", str(g)])
    assert r.exit_code == 1
    assert isinstance(r.exception, SystemExit)  # error panel, not a traceback
    assert "invalid JSON in glossary g.json" in _flat(r.output)
    assert not m.called


@pytest.mark.parametrize("option", ["--min-speakers", "--max-speakers"])
def test_speaker_bounds_must_be_positive(tmp_path, option):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--diarize", option, "0", str(media)])
    assert r.exit_code == 2
    assert not m.called


def test_min_speakers_above_max_speakers_is_usage_error(tmp_path):
    media, out = _media(tmp_path)
    args = ["--diarize", "--min-speakers", "3", "--max-speakers", "2", str(media)]
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, args)
    assert r.exit_code == 2
    assert "is greater than" in _flat(r.output)
    assert not m.called


@pytest.mark.parametrize(
    "args",
    [
        ["--min-speakers", "2"],
        ["--max-speakers", "4"],
        ["--diarize-model", "3.1"],
        ["--speaker-clustering", "pyannote"],
    ],
)
def test_speaker_options_warn_when_diarization_is_off(tmp_path, args):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out):
        r = CliRunner().invoke(cli, ["--no-diarize", *args, str(media)])
    assert r.exit_code == 0, r.output
    assert (
        f"{args[0]} has no effect: diarization is off (from CLI --no-diarize)"
        in _flat(r.output)
    )


def test_keep_lyrics_without_separation_warns(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, ["--no-separate", "--keep-lyrics", str(media)])
    assert r.exit_code == 0, r.output
    assert "--keep-lyrics has no effect" in _flat(r.output)
    assert m.call_args.kwargs["keep_lyrics"] is True


def test_blank_voiceprints_env_counts_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_VOICEPRINTS", "  ")
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out) as m:
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert m.call_args.kwargs["voiceprints"] is False


@pytest.mark.parametrize(
    "args", [["transcribe", "--help"], ["render", "-h"], ["-v", "align", "--help"]]
)
def test_subcommand_help_does_not_write_default_config(tmp_path, args):
    r = CliRunner().invoke(cli, args)
    assert r.exit_code == 0, r.output
    assert not (tmp_path / "voxweave.conf").exists()
    assert "created default config" not in r.output


def test_real_run_still_writes_default_config(tmp_path):
    media, out = _media(tmp_path)
    with patch("voxweave.pipeline.process", return_value=out):
        r = CliRunner().invoke(cli, [str(media)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "voxweave.conf").exists()


def test_debug_summary_reads_the_claim_without_creating_one(tmp_path):
    from voxweave import artifacts

    media, out = _media(tmp_path)
    with (
        patch("voxweave.pipeline.process", return_value=out),
        patch("voxweave.cli.summary_panel") as panel,
    ):
        r = CliRunner().invoke(cli, ["--debug", str(media)])
    assert r.exit_code == 0, r.output
    assert panel.call_args.kwargs["debug_dir"] is None
    assert artifacts.inspect_paths(media) is None  # nothing claimed for display

    debug_dir = artifacts.claim_paths(media).debug
    with (
        patch("voxweave.pipeline.process", return_value=out),
        patch("voxweave.cli.summary_panel") as panel,
    ):
        r = CliRunner().invoke(cli, ["--debug", str(media)])
    assert r.exit_code == 0, r.output
    assert panel.call_args.kwargs["debug_dir"] == debug_dir
