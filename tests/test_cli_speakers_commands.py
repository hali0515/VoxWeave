"""Canonical speaker commands preserve legacy routes and keep inspection read-only."""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path

import pytest
import rich_click as click
from click.testing import CliRunner

from voxweave import artifacts, cli_compat, speakers, speakerserve
from voxweave.cli_speakers import build_speakers_group


@pytest.fixture
def speaker_group():
    previous_warnings = set(cli_compat._WARNED_DEPRECATIONS)
    cli_compat._WARNED_DEPRECATIONS.clear()

    def run(fn, *, reporter=True):
        assert reporter is False
        try:
            return fn(None)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

    def report(message):
        click.echo(message, err=not message.startswith("http://"))

    yield build_speakers_group(run, report)
    cli_compat._WARNED_DEPRECATIONS.clear()
    cli_compat._WARNED_DEPRECATIONS.update(previous_warnings)


@pytest.fixture
def audition(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    (tmp_path / "episode.json").write_text("{}", encoding="utf-8")
    (tmp_path / "episode.vtt").write_text("WEBVTT\n", encoding="utf-8")
    value = speakers.SpeakerAudition(
        page="<html></html>",
        media_path=media,
        sibling_json_path=tmp_path / "episode.json",
        mapping_path=tmp_path / "episode.speakers.json",
        speaker_ids=("SPEAKER_00",),
    )
    seen = {"create": [], "serve": []}

    def create(path, **kwargs):
        seen["create"].append((path, kwargs))
        return value

    def serve(**kwargs):
        seen["serve"].append(kwargs)
        kwargs["report"]("http://127.0.0.1:41533/")
        kwargs["report"]("Saved speaker names")
        return "http://127.0.0.1:41533/"

    monkeypatch.setattr(speakers, "create_speaker_audition", create)
    monkeypatch.setattr(speakerserve, "serve", serve)
    return media, seen


@pytest.mark.parametrize(
    "args",
    [
        ["serve", "EPISODE", "--no-open"],
        ["EPISODE", "--no-open"],
        ["--no-open", "EPISODE"],
        ["serve", "--no-open", "EPISODE"],
    ],
)
def test_serve_canonical_and_bare_forms(speaker_group, audition, args):
    media, seen = audition
    result = CliRunner().invoke(
        speaker_group, [str(media) if arg == "EPISODE" else arg for arg in args]
    )
    assert result.exit_code == 0, result.output
    assert seen["create"] == [(media, {})]
    assert seen["serve"][0]["open_browser"] is False
    assert result.stdout == "http://127.0.0.1:41533/\n"
    assert result.stderr == "Saved speaker names\n"


@pytest.mark.parametrize("suffix", [".vtt", ".json"])
def test_serve_resolves_episode_reference(speaker_group, audition, suffix):
    media, seen = audition
    result = CliRunner().invoke(
        speaker_group, ["serve", str(media.with_suffix(suffix))]
    )
    assert result.exit_code == 0, result.output
    assert seen["create"] == [(media, {})]


def test_manual_and_legacy_no_match_warn_once(speaker_group, audition):
    media, seen = audition
    runner = CliRunner()
    canonical = runner.invoke(
        speaker_group, ["serve", str(media), "--manual", "--open"]
    )
    first = runner.invoke(speaker_group, [str(media), "--no-match"])
    second = runner.invoke(speaker_group, ["--no-match", str(media)])
    assert all(result.exit_code == 0 for result in (canonical, first, second))
    assert all(kwargs["no_match"] for _, kwargs in seen["create"])
    assert "deprecated" not in canonical.stderr
    assert "--no-match is deprecated" in first.stderr
    assert "deprecated" not in second.stderr
    assert seen["serve"][0]["open_browser"] is True


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("before_media", [False, True])
def test_enroll_preserves_options(
    speaker_group, audition, monkeypatch, legacy, before_media
):
    media, seen = audition
    enrolled = {}
    output = media.parent / "voices.json"
    monkeypatch.setattr(
        speakers,
        "enroll_speaker_voices",
        lambda path, **kwargs: enrolled.update(path=path, **kwargs) or output,
    )
    options = ["--voices", str(output), "--show", "Show", "--episode", "S01E01"]
    options += ["--replace-episode"] if legacy else ["--replace"]
    if legacy:
        options += ["--enroll"]
    args = (options + [str(media)]) if before_media else ([str(media)] + options)
    if not legacy:
        args.insert(0, "enroll")
    result = CliRunner().invoke(speaker_group, args)
    assert result.exit_code == 0, result.output
    assert enrolled == {
        "path": media,
        "voices": output,
        "show": "Show",
        "episode": "S01E01",
        "replace_episode": True,
    }
    assert seen["create"] == []
    assert seen["serve"] == []
    assert result.stdout == f"{output}\n"
    assert ("deprecated" in result.stderr) is legacy


@pytest.mark.parametrize(
    "args",
    [
        ["purge", "EPISODE"],
        ["--purge-voiceprints", "EPISODE"],
        ["EPISODE", "--purge-voiceprints"],
    ],
)
def test_purge_allows_missing_media_and_preserves_names(speaker_group, tmp_path, args):
    media = tmp_path / "episode.mkv"
    mapping = tmp_path / "episode.speakers.json"
    mapping.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":"Aoi"}}', encoding="utf-8"
    )
    voiceprints = tmp_path / "episode.voiceprints.json"
    voiceprints.write_text("{}", encoding="utf-8")
    result = CliRunner().invoke(
        speaker_group, [str(media) if arg == "EPISODE" else arg for arg in args]
    )
    assert result.exit_code == 0, result.output
    assert str(voiceprints) in result.stdout
    assert not voiceprints.exists()
    assert mapping.exists()
    assert not media.exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--manual", "--no-match"],
        ["--enroll", "--manual"],
        ["--enroll", "--no-open"],
        ["--enroll", "--port", "1234"],
        ["--purge-voiceprints", "--enroll"],
        ["--purge-voiceprints", "--show", "Show"],
        ["--episode", "S01E01"],
        ["--replace-episode"],
    ],
)
def test_invalid_legacy_combinations_still_fail(speaker_group, audition, options):
    media, seen = audition
    result = CliRunner().invoke(speaker_group, [str(media), *options])
    assert result.exit_code == 2, result.output
    assert seen["create"] == []
    assert seen["serve"] == []


def test_replace_alias_conflict_is_explicit(speaker_group, audition):
    media, _ = audition
    result = CliRunner().invoke(
        speaker_group, ["enroll", str(media), "--replace", "--replace-episode"]
    )
    assert result.exit_code == 2
    assert "not both" in result.output


def test_help_lists_commands_and_hides_legacy_options(speaker_group):
    runner = CliRunner()
    group_help = runner.invoke(speaker_group, ["--help"])
    serve_help = runner.invoke(speaker_group, ["serve", "--help"])
    enroll_help = runner.invoke(speaker_group, ["enroll", "--help"])
    for name in ("serve", "enroll", "purge", "list"):
        assert name in group_help.output
    assert "--manual" in serve_help.output
    assert "--open" in serve_help.output
    for flag in ("--no-match", "--enroll", "--purge-voiceprints", "--replace-episode"):
        assert flag not in serve_help.output
    assert "--replace" in enroll_help.output
    assert "--replace-episode" not in enroll_help.output


def test_misspelled_subcommand_uses_command_error(speaker_group):
    result = CliRunner().invoke(speaker_group, ["enrol", "episode.mkv"])
    assert result.exit_code == 2
    assert "No such command" in result.output
    assert "enroll" in result.output


@pytest.mark.parametrize("media_present", [False, True])
def test_list_json_is_read_only_even_with_cached_artifacts(
    speaker_group, tmp_path, monkeypatch, media_present
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    sibling = tmp_path / "episode.json"
    sibling.write_text(
        json.dumps(
            {
                "speaker_turns": [
                    [0, 2, "SPEAKER_00"],
                    [3, 4, "SPEAKER_01"],
                    [5, 7.5, "SPEAKER_00"],
                ]
            }
        ),
        encoding="utf-8",
    )
    cached = artifacts.claim_paths(media)
    cached.speaker_mapping.write_text(
        json.dumps({"version": 1, "speakers": {"SPEAKER_00": "[cyan]Aoi"}}),
        encoding="utf-8",
    )
    cache_root = cached.directory.parent
    cache_root.chmod(0o755)
    if not media_present:
        media.unlink()
    before = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")}

    def forbidden(*args, **kwargs):
        raise AssertionError("list attempted a filesystem mutation")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(os, "chmod", forbidden)
    monkeypatch.setattr(artifacts, "claim_paths", forbidden)
    result = CliRunner().invoke(speaker_group, ["list", str(sibling), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["speakers"] == [
        {
            "id": "SPEAKER_00",
            "name": "[cyan]Aoi",
            "turns": 2,
            "duration_seconds": 4.5,
            "voiceprint": "absent",
        },
        {
            "id": "SPEAKER_01",
            "name": None,
            "turns": 1,
            "duration_seconds": 1.0,
            "voiceprint": "absent",
        },
    ]
    assert data["voiceprints"]["media_checked"] is False
    assert result.stderr == ""
    assert before == {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")}
    assert cache_root.stat().st_mode & 0o777 == 0o755


def test_list_without_cache_does_not_create_it(speaker_group, tmp_path, monkeypatch):
    sibling = tmp_path / "episode.json"
    sibling.write_text('{"speaker_turns":[[0,1,"SPEAKER_00"]]}', encoding="utf-8")
    original_import = builtins.__import__

    def without_models(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "pyannote", "transformers"}:
            raise AssertionError(f"list attempted to import a model dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_models)
    result = CliRunner().invoke(speaker_group, ["list", str(sibling)])
    assert result.exit_code == 0, result.output
    assert "SPEAKER_00" in result.stdout
    assert "media not checked" in result.stdout
    assert not (tmp_path / "cache").exists()


def test_list_preserves_literal_names_and_legacy_mapping_priority(
    speaker_group, tmp_path
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    sibling = tmp_path / "episode.json"
    sibling.write_text('{"speaker_turns":[[0,1,"SPEAKER_00"]]}', encoding="utf-8")
    cached = artifacts.claim_paths(media)
    cached.speaker_mapping.write_text(
        '{"version":1,"speakers":{"SPEAKER_00":"cached"}}', encoding="utf-8"
    )
    (tmp_path / "episode.speakers.json").write_text(
        '{"version":1,"speakers":{"SPEAKER_00":"[red]Aoi"}}', encoding="utf-8"
    )
    result = CliRunner().invoke(speaker_group, ["list", str(media)])
    assert result.exit_code == 0, result.output
    assert "[red]Aoi" in result.stdout
    assert "cached" not in result.stdout


@pytest.mark.parametrize("state", ["recorded", "stale", "unbound", "invalid"])
def test_list_describes_saved_voiceprint_binding_without_checking_media(
    speaker_group, tmp_path, monkeypatch, state
):
    from voxweave import voicebase

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media need not match the saved fingerprint")
    turns = [[0, 1, "SPEAKER_00"]]
    capture = "c" + "1" * 32
    fingerprint = "a" * 64
    sibling = {"speaker_turns": turns}
    if state != "unbound":
        sibling.update(voiceprint_capture=capture, voiceprint_media=fingerprint)
    (tmp_path / "episode.json").write_text(json.dumps(sibling), encoding="utf-8")
    sidecar = {
        "version": 1,
        "capture_id": "c" + "2" * 32 if state == "stale" else capture,
        "provenance": {
            "diarization_model": "repo/model",
            "embedding_dim": 16,
            "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
        },
        "binding": {
            "turns_digest": voicebase.canonical_turns_digest(turns),
            "media_fingerprint": fingerprint,
            "media_stem": "episode",
            "created": "2026-09-05T00:00:00Z",
        },
        "speakers": {"SPEAKER_00": [1.0, *([0.0] * 15)]},
    }
    (tmp_path / "episode.voiceprints.json").write_text(
        "not-json" if state == "invalid" else json.dumps(sidecar), encoding="utf-8"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("list must not fingerprint the media")

    monkeypatch.setattr(voicebase, "media_fingerprint", forbidden)
    result = CliRunner().invoke(speaker_group, ["list", str(media), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["voiceprints"]["state"] == state
    assert data["voiceprints"]["media_checked"] is False
    assert data["speakers"][0]["voiceprint"] == state


def test_read_only_artifact_inspection_rejects_symlink_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "cache").symlink_to(outside, target_is_directory=True)
    media = tmp_path / "episode.mkv"
    with pytest.raises(artifacts.ArtifactMarkerError, match="private directory"):
        artifacts.inspect_paths(media)
    with pytest.raises(artifacts.ArtifactMarkerError, match="private directory"):
        artifacts.claimed_sources(tmp_path, "episode")
