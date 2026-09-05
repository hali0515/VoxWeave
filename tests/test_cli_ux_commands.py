"""Public CLI dispatch and rename contracts, without models or media processing."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import rich_click as click
from click.testing import CliRunner

from voxweave import cli as cli_module
from voxweave import config, export, mux, pipeline
from voxweave.progress import Reporter


class _Reporter(Reporter):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    """Keep real parsing and logging while replacing every expensive callback."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "cli-test-key")
    monkeypatch.delenv("VOXWEAVE_ASR_MODEL", raising=False)
    monkeypatch.delenv("VOXWEAVE_VOICEPRINTS", raising=False)
    monkeypatch.setattr(config, "ensure_default_config", lambda: None)
    monkeypatch.setattr(cli_module, "RichReporter", _Reporter)
    for name in (
        "summary_panel",
        "success_panel",
        "translate_summary_panel",
    ):
        monkeypatch.setattr(cli_module, name, Mock())

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"mock media")
    subtitle = tmp_path / "episode.vtt"
    subtitle.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n")
    episode = tmp_path / "episode.json"
    episode.write_text("{}")
    targets = {
        "transcribe": (pipeline, "process"),
        "render": (pipeline, "split"),
        "translate": (pipeline, "translate"),
        "export": (export, "export_subtitles"),
        "pack": (mux, "pack"),
        "burn": (mux, "burn"),
    }
    calls = {}
    for command, (module, name) in targets.items():
        result = [tmp_path / "episode.srt"] if command == "export" else subtitle
        calls[command] = Mock(return_value=result)
        monkeypatch.setattr(module, name, calls[command])

    root_logger = logging.getLogger()
    old_handlers, old_level = root_logger.handlers[:], root_logger.level
    try:
        yield SimpleNamespace(
            runner=CliRunner(),
            media=media,
            subtitle=subtitle,
            episode=episode,
            calls=calls,
        )
    finally:
        root_logger.handlers = old_handlers
        root_logger.setLevel(old_level)


def _invoke(case, args):
    return case.runner.invoke(cli_module.cli, args, terminal_width=140)


def _pipeline_call(mock):
    call = mock.call_args
    return call.args, {
        key: value for key, value in call.kwargs.items() if key != "reporter"
    }


@pytest.mark.parametrize("verbosity", [[], ["-v"], ["--verbose"], ["-vv"]])
@pytest.mark.parametrize("debug", [[], ["--debug"]])
def test_explicit_transcribe_matches_media_shorthand(cli_case, verbosity, debug):
    bare = _invoke(cli_case, [*verbosity, *debug, str(cli_case.media)])
    assert bare.exit_code == 0, bare.output
    bare_call = _pipeline_call(cli_case.calls["transcribe"])
    cli_case.calls["transcribe"].reset_mock()

    explicit = _invoke(
        cli_case, [*verbosity, "transcribe", *debug, str(cli_case.media)]
    )
    assert explicit.exit_code == 0, explicit.output
    cli_case.calls["transcribe"].assert_called_once()
    assert _pipeline_call(cli_case.calls["transcribe"]) == bare_call
    assert bare_call[1]["debug"] is bool(debug)
    assert bare.stdout.strip() == explicit.stdout.strip() == str(cli_case.subtitle)


@pytest.mark.parametrize("verbosity", ["-v", "--verbose"])
def test_group_verbosity_after_transcribe_option_is_usage_error(cli_case, verbosity):
    result = _invoke(cli_case, ["--debug", verbosity, str(cli_case.media)])
    assert result.exit_code == 2, result.output
    assert "No such option" in result.output
    cli_case.calls["transcribe"].assert_not_called()


@pytest.mark.parametrize("typo", ["pak", "transcrib", "redner"])
def test_unknown_verb_is_not_reported_as_missing_media(cli_case, typo):
    result = _invoke(cli_case, [typo, str(cli_case.media)])
    assert result.exit_code == 2, result.output
    assert "No such command" in result.output
    assert typo in result.output
    assert "does not exist" not in result.output
    cli_case.calls["transcribe"].assert_not_called()


@pytest.mark.parametrize("command", ["render", "pack"])
def test_known_command_wins_over_existing_same_named_file(cli_case, command):
    cli_case.media.with_name(command).write_bytes(b"extensionless media")
    source = cli_case.episode if command == "render" else cli_case.subtitle
    result = _invoke(cli_case, [command, str(source)])
    assert result.exit_code == 0, result.output
    cli_case.calls[command].assert_called_once()
    cli_case.calls["transcribe"].assert_not_called()


@pytest.mark.parametrize("args", [["./pack"], ["--", "pack"]])
def test_explicit_path_escape_can_transcribe_file_named_like_command(cli_case, args):
    cli_case.media.with_name("pack").write_bytes(b"extensionless media")
    result = _invoke(cli_case, args)
    assert result.exit_code == 0, result.output
    cli_case.calls["transcribe"].assert_called_once()
    assert cli_case.calls["transcribe"].call_args.args[0].name == "pack"
    cli_case.calls["pack"].assert_not_called()


@pytest.mark.parametrize(
    ("suffix", "hint"),
    [
        (".vtt", "align"),
        (".srt", "export"),
        (".ass", "export"),
        (".ssa", "export"),
        (".json", "render"),
    ],
)
def test_bare_nonmedia_input_explains_next_command(cli_case, suffix, hint):
    source = cli_case.media.with_suffix(suffix)
    source.touch(exist_ok=True)
    result = _invoke(cli_case, [str(source)])
    assert result.exit_code == 2, result.output
    assert hint in result.output.lower()
    cli_case.calls["transcribe"].assert_not_called()


def test_group_help_exposes_workflow_and_hides_command_aliases(cli_case):
    result = _invoke(cli_case, ["--help"])
    assert result.exit_code == 0, result.output
    for panel in ("Capture", "Revise", "Deliver"):
        assert panel in result.output
    for command in ("transcribe", "render"):
        assert re.search(rf"\b{command}\b", result.output)
    assert not re.search(r"(?m)^\s*[│┃|]?\s*(?:split|help)\s{2,}", result.output)


@pytest.mark.parametrize(
    "command", [None, "transcribe", "render", "translate", "export", "pack", "burn"]
)
def test_hidden_help_command_matches_help_option(cli_case, command):
    path = [command] if command else []
    option = _invoke(cli_case, [*path, "--help"])
    alias = _invoke(cli_case, ["help", *path])
    assert option.exit_code == alias.exit_code == 0, alias.output
    assert alias.output == option.output
    assert all(not mock.called for mock in cli_case.calls.values())


_OPTION_CASES = [
    ("transcribe", "--asr-model", "-m", "--model", "qwen3-asr-1.7B", "asr_model"),
    ("translate", "--target", "-t", "--to", "ja", "to"),
    ("export", "--format", "-f", "--to", "ass", None),
    ("pack", "--container", None, "--to", "mkv", "container"),
    ("burn", "--container", None, "--to", "mkv", "container"),
]
_OPTION_STYLES = [
    (*case, style)
    for case in _OPTION_CASES
    for style in ("separate", "equals", "short-attached")
    if style != "short-attached" or case[2] is not None
]


def _source(case, command):
    return str(case.media if command == "transcribe" else case.subtitle)


def _option_args(flag, value, style):
    if style == "separate":
        return [flag, value]
    if flag.startswith("--"):
        return [f"{flag}={value}"]
    return [f"{flag}{value}"]


@pytest.mark.parametrize(
    "command,canonical,short,legacy,value,key,style", _OPTION_STYLES
)
def test_canonical_and_legacy_options_forward_same_value_silently(
    cli_case, command, canonical, short, legacy, value, key, style
):
    expected = None
    primary = short if style == "short-attached" else canonical
    for flag in (primary, legacy):
        result = _invoke(
            cli_case,
            [command, _source(cli_case, command), *_option_args(flag, value, style)],
        )
        assert result.exit_code == 0, result.output
        assert result.stderr == ""
        actual = _pipeline_call(cli_case.calls[command])
        if key:
            assert actual[1][key] == value
        else:
            assert actual[0][1] == (value,)
        if expected is None:
            expected = actual
        assert actual == expected


@pytest.mark.parametrize("command,canonical,short,legacy,value,key", _OPTION_CASES)
def test_help_shows_canonical_option_and_hides_legacy_spelling(
    cli_case, command, canonical, short, legacy, value, key
):
    result = _invoke(cli_case, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert canonical in result.output
    assert not re.search(rf"(?<![\w-]){re.escape(legacy)}(?![\w-])", result.output)


@pytest.mark.parametrize(
    "command,canonical,short,legacy,value,key,style", _OPTION_STYLES
)
@pytest.mark.parametrize("reverse", [False, True])
def test_canonical_and_legacy_spellings_conflict_even_with_same_value(
    cli_case, command, canonical, short, legacy, value, key, style, reverse
):
    primary = short if style == "short-attached" else canonical
    canonical_args = _option_args(primary, value, style)
    legacy_args = _option_args(legacy, value, style)
    options = legacy_args + canonical_args if reverse else canonical_args + legacy_args
    result = _invoke(cli_case, [command, _source(cli_case, command), *options])
    assert result.exit_code == 2, result.output
    assert canonical in result.output
    assert legacy in result.output
    cli_case.calls[command].assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        ["--format", "srt", "--format", "ass"],
        ["-fsrt", "-fass"],
        ["--to", "srt", "--to", "ass"],
    ],
)
def test_export_keeps_repeatable_formats_for_each_spelling(cli_case, options):
    result = _invoke(cli_case, ["export", str(cli_case.subtitle), *options])
    assert result.exit_code == 0, result.output
    assert cli_case.calls["export"].call_args.args[1] == ("srt", "ass")


def test_split_alias_warns_once_per_process_and_calls_render_pipeline(
    cli_case, monkeypatch
):
    from voxweave import cli_compat

    monkeypatch.setattr(cli_compat, "_WARNED_DEPRECATIONS", set())
    first = _invoke(cli_case, ["split", str(cli_case.episode)])
    second = _invoke(cli_case, ["split", str(cli_case.episode)])
    assert first.exit_code == second.exit_code == 0, first.output + second.output
    assert cli_case.calls["render"].call_count == 2
    assert first.stdout.strip() == second.stdout.strip() == str(cli_case.subtitle)
    assert "split" in first.stderr and "render" in first.stderr
    assert "deprecat" in first.stderr.lower()
    assert "deprecat" not in second.stderr.lower()


def _group_with_value_options(group_class):
    """Exercise dispatch against native Click using a small, model-free group."""
    seen = {}

    @click.group(cls=group_class)
    @click.option("-c", "--config")
    @click.option("-P", "--pair", nargs=2)
    @click.option("-v", "--verbose", is_flag=True)
    def group(**kwargs):
        seen["group"] = kwargs

    @group.command("transcribe")
    @click.argument("media")
    @click.option("--debug", is_flag=True)
    def transcribe(**kwargs):
        seen["transcribe"] = kwargs

    group.default_cmd = transcribe
    group.media_extensions = (".mkv",)
    return group, seen


@pytest.mark.parametrize(
    "prefix",
    [
        ["--config", "settings.toml"],
        ["--config=settings.toml"],
        ["-csettings.toml"],
        ["-vcsettings.toml"],
        ["-v", "--config", "transcribe"],
        ["--config", "--debug"],
        ["--config="],
        ["--pair", "one", "two"],
        ["--pair=one", "two"],
        ["-Pone", "two"],
        ["--pair", "transcribe", "pack"],
        ["--config", "settings.toml", "--pair", "one", "two", "-v"],
    ],
)
def test_group_option_values_remain_native_click_values_before_media(prefix):
    from voxweave.cli_compat import DefaultGroup

    shortcut, shortcut_seen = _group_with_value_options(DefaultGroup)
    native, native_seen = _group_with_value_options(click.RichGroup)
    runner = CliRunner()
    expected = runner.invoke(native, [*prefix, "transcribe", "--debug", "episode.mkv"])
    actual = runner.invoke(shortcut, [*prefix, "--debug", "episode.mkv"])
    assert expected.exit_code == 0, expected.output
    assert actual.exit_code == 0, actual.output
    assert shortcut_seen == native_seen
    assert shortcut_seen["transcribe"] == {"media": "episode.mkv", "debug": True}


@pytest.mark.parametrize(
    "args",
    [
        ["--config"],
        ["-c"],
        ["--pair", "one"],
        ["--pair=one"],
        ["-Pone"],
    ],
)
def test_missing_group_option_values_remain_usage_errors(args):
    from voxweave.cli_compat import DefaultGroup

    shortcut, seen = _group_with_value_options(DefaultGroup)
    native, _native_seen = _group_with_value_options(click.RichGroup)
    runner = CliRunner()
    expected = runner.invoke(native, args)
    actual = runner.invoke(shortcut, args)
    assert expected.exit_code == actual.exit_code == 2, actual.output
    assert seen == {}


def test_native_click_suggestion_is_not_duplicated(monkeypatch):
    from voxweave.cli_compat import DefaultGroup

    class NativeCommandError(click.UsageError):
        possibilities = ["transcribe"]

        def format_message(self):
            return self.message + " Did you mean 'transcribe'?"

    def unknown_command(self, ctx, args):
        raise NativeCommandError("No such command 'transcrib'.", ctx)

    group, _seen = _group_with_value_options(DefaultGroup)
    monkeypatch.setattr(click.RichGroup, "resolve_command", unknown_command)
    result = CliRunner().invoke(group, ["transcrib"])
    assert result.exit_code == 2
    assert result.output.count("Did you mean") == 1
    assert "pass a path" in result.output


def test_speakers_list_does_not_create_first_run_config(tmp_path, monkeypatch):
    from voxweave.cli_speakers import _list_episode

    assert callable(_list_episode)
    episode = tmp_path / "episode.json"
    episode.write_text('{"speaker_turns": [[0, 1, "SPEAKER_00"]]}')
    conf = tmp_path / "not-created.conf"
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(conf))
    result = CliRunner().invoke(
        cli_module.cli, ["speakers", "list", str(episode), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert not conf.exists()
    assert not (tmp_path / ".voxweave").exists()
