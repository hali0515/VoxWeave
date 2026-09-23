"""``voxweave voices`` and the library options of ``speakers``, via CliRunner."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from voxweave import cli as cli_module
from voxweave import voicelibrary, voicestore

AUDIO = {"separated": False, "normalized": False, "sample_rate": 16000}
DECOUPLED = {
    "diarization_model": "example/diarizer",
    "embedding_lane": "decoupled",
    "embedding_model": "redimnet2-b6-vb2-vox2-cnc2-lm",
    "embedding_checkpoint": "e" * 64,
    "embedding_dim": 16,
    "embedding_recipe": "centroid-v1",
    "audio": AUDIO,
    "torch_version": "test",
}
LEGACY = {
    "diarization_model": "example/diarizer",
    "outer_config_sha256": "a" * 64,
    "embedding_model": "example/embedder",
    "embedding_checkpoint": "b" * 64,
    "embedding_dim": 16,
    "audio": AUDIO,
    "pyannote_version": "3.4.0",
    "torch_version": "test",
}


def _unit(index):
    vector = [0.0] * 16
    vector[index] = 1.0
    return vector


@pytest.fixture
def invoke():
    root_logger = logging.getLogger()
    handlers, level = root_logger.handlers[:], root_logger.level
    runner = CliRunner()

    def run(*args, input=None):
        return runner.invoke(
            cli_module.cli, [str(arg) for arg in args], input=input, terminal_width=160
        )

    try:
        yield run
    finally:
        root_logger.handlers = handlers
        root_logger.setLevel(level)


@pytest.fixture
def library(tmp_path):
    """A library with two "Alex" identities in two scopes and two spaces."""
    root = tmp_path / "library"
    counter = iter(range(1, 100))

    def enroll(provenance, identity_id, name, index, scope, number):
        space, _ = voicelibrary.space_identity(provenance)
        with voicelibrary.library_lock(root, exclusive=True):
            state = voicelibrary.read_state(root, spaces=[space])
            change, _ = voicelibrary.enroll_entries(
                state,
                provenance=provenance,
                scope=scope,
                source=voicelibrary.EpisodeSource(
                    media_path=f"/media/{scope}/ep{number}.mkv",
                    media_fingerprint=f"{number:064x}",
                    capture_id=f"c{number:032x}",
                    turns_digest=None,
                    episode=f"ep{number}",
                ),
                entries=[
                    voicelibrary.EnrollEntry(
                        identity_id, name, "SPEAKER_00", _unit(index)
                    )
                ],
                identity_id_factory=lambda: f"v{next(counter):012x}",
            )
            voicelibrary.commit(state, change)

    enroll(DECOUPLED, None, "Alex", 0, "Show A", 1)
    enroll(DECOUPLED, None, "Alex", 1, "Show B", 2)
    enroll(LEGACY, "v000000000001", "Alex", 2, "Show A", 3)
    return root


def test_where_reports_the_layer_that_chose_the_directory(
    tmp_path, invoke, monkeypatch
):
    default = invoke("voices", "where")
    assert default.exit_code == 0, default.output
    lines = default.output.splitlines()
    assert lines[0] == str(tmp_path / ".xdg-data" / "voxweave" / "voices")
    assert lines[1] == "source: built-in default ($XDG_DATA_HOME)"

    monkeypatch.setenv("VOXWEAVE_VOICES_DIR", str(tmp_path / "env"))
    env = invoke("voices", "where")
    assert env.output.splitlines() == [
        str(tmp_path / "env"),
        "source: environment VOXWEAVE_VOICES_DIR",
    ]
    cli = invoke("voices", "where", "--voices-dir", tmp_path / "cli")
    assert cli.output.splitlines()[1] == "source: --voices-dir"
    assert not (tmp_path / "env").exists() and not (tmp_path / "cli").exists()


def test_list_json_and_scope_filter(library, invoke):
    empty = invoke("voices", "list", "--voices-dir", library.parent / "none")
    assert empty.exit_code == 0 and "No saved voices" in empty.output

    result = invoke("voices", "list", "--json", "--voices-dir", library)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["library"] == str(library)
    rows = {row["id"]: row for row in payload["identities"]}
    assert set(rows) == {"v000000000001", "v000000000002"}
    assert sorted(rows["v000000000001"]["exemplars"].values()) == [1, 1]
    assert rows["v000000000002"]["scopes"] == ["Show B"]

    scoped = invoke(
        "voices", "list", "--scope", "Show B", "--json", "--voices-dir", library
    )
    assert [row["id"] for row in json.loads(scoped.stdout)["identities"]] == [
        "v000000000002"
    ]
    table = invoke("voices", "list", "--voices-dir", library)
    assert "v000000000001" in table.output and "Show A" in table.output


def test_show_lists_candidates_for_an_ambiguous_name(library, invoke):
    ambiguous = invoke("voices", "show", "Alex", "--voices-dir", library)
    assert ambiguous.exit_code == 1
    assert "names 2 identities" in ambiguous.stderr
    assert "v000000000001" in ambiguous.stdout and "v000000000002" in ambiguous.stdout

    detail = invoke(
        "voices", "show", "v000000000001", "--json", "--voices-dir", library
    )
    assert detail.exit_code == 0, detail.output
    data = json.loads(detail.stdout)
    assert len(data["spaces"]) == 2
    assert "vector" not in detail.stdout

    human = invoke("voices", "show", "v000000000002", "--voices-dir", library)
    assert "Show B / ep2" in human.stdout
    missing = invoke("voices", "show", "Nobody", "--voices-dir", library)
    assert missing.exit_code != 0 and "no saved identity" in missing.output


def test_rename_by_id(library, invoke):
    result = invoke(
        "voices", "rename", "v000000000002", "Alex (B)", "--voices-dir", library
    )
    assert result.exit_code == 0, result.output
    shown = invoke("voices", "show", "Alex", "--json", "--voices-dir", library)
    assert json.loads(shown.stdout)["id"] == "v000000000001"
    unknown = invoke("voices", "rename", "v0000000000ff", "X", "--voices-dir", library)
    assert unknown.exit_code == 1


def test_forget_asks_first_and_removes_every_space(library, invoke):
    declined = invoke(
        "voices", "forget", "v000000000001", "--voices-dir", library, input="n\n"
    )
    assert declined.exit_code != 0
    assert "Forget Alex (v000000000001) and 2 voice sample(s)?" in declined.output
    with voicelibrary.library_lock(library, exclusive=False):
        assert "v000000000001" in voicelibrary.read_state(library).identity_map

    result = invoke(
        "voices", "forget", "v000000000001", "--yes", "--voices-dir", library
    )
    assert result.exit_code == 0, result.output
    assert "2 voice sample(s) removed" in result.output
    with voicelibrary.library_lock(library, exclusive=False):
        state = voicelibrary.read_state(library)
    assert list(state.identity_map) == ["v000000000002"]
    assert all("v000000000001" not in state.space_exemplars(n) for n in state.spaces)


def test_import_is_idempotent_and_leaves_the_store_alone(tmp_path, invoke):
    store = voicestore.new_voice_store("Example Show", LEGACY)
    store = voicestore.enroll_exemplar(
        store,
        raw_name="Aqua",
        capture_id="c" + "5" * 32,
        media_fingerprint="5" * 64,
        episode="ep05",
        vector=_unit(5),
        at="2026-09-01T00:00:00Z",
    ).store
    legacy = tmp_path / "old" / "voxweave.voices.json"
    legacy.parent.mkdir()
    voicestore.write_voice_store(legacy, store)
    before = legacy.read_bytes()
    root = tmp_path / "library"

    first = invoke("voices", "import", legacy, "--voices-dir", root)
    assert first.exit_code == 0, first.output
    assert "imported 1 identities and 1 voice sample(s)" in first.output
    # A voxweave.voices.json served its own folder and its show before.
    assert "scopes 'old', 'Example Show'" in first.output
    repeat = invoke("voices", "import", legacy, "--voices-dir", root)
    assert repeat.exit_code == 0, repeat.output
    assert "imported 0 identities and 0 voice sample(s)" in repeat.output
    assert "1 already present" in repeat.output
    second = invoke(
        "voices", "import", legacy, "--scope", "Other", "--voices-dir", root
    )
    assert second.exit_code == 0, second.output
    assert "imported 0 identities and 0 voice sample(s)" in second.output
    assert "1 already present; 1 scope(s) added" in second.output
    assert legacy.read_bytes() == before
    history = [
        json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()
    ]
    assert [row["action"] for row in history].count("import") == 2
    shown = json.loads(invoke("voices", "list", "--json", "--voices-dir", root).stdout)[
        "identities"
    ]
    assert shown[0]["scopes"] == ["old", "Example Show", "Other"]


def test_forget_names_per_folder_stores_that_still_hold_the_id(tmp_path, invoke):
    store = voicestore.new_voice_store("Example Show", LEGACY)
    store = voicestore.enroll_exemplar(
        store,
        raw_name="Aqua",
        capture_id="c" + "5" * 32,
        media_fingerprint="5" * 64,
        episode="ep05",
        vector=_unit(5),
        at="2026-09-01T00:00:00Z",
    ).store
    [identity_id] = store["identities"]
    legacy = tmp_path / "Show A" / "voxweave.voices.json"
    legacy.parent.mkdir()
    voicestore.write_voice_store(legacy, store)
    root = tmp_path / "library"
    assert invoke("voices", "import", legacy, "--voices-dir", root).exit_code == 0

    result = invoke("voices", "forget", identity_id, "--yes", "--voices-dir", root)
    assert result.exit_code == 0, result.output
    assert f"per-folder store {legacy.resolve()} still holds 1 voice" in result.stderr
    assert "deleted" in result.stderr

    again = invoke("voices", "import", legacy, "--voices-dir", root)
    assert again.exit_code == 0, again.output
    assert f"skipped {identity_id}: forgotten" in again.stderr
    assert "imported 0 identities and 0 voice sample(s)" in again.stdout
    listed = invoke("voices", "list", "--json", "--voices-dir", root)
    assert json.loads(listed.stdout)["identities"] == []


def test_speakers_rejects_both_store_kinds(tmp_path, invoke):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    for command in ("enroll", "serve"):
        result = invoke(
            "speakers",
            command,
            media,
            "--voices",
            tmp_path / "v.json",
            "--voices-dir",
            tmp_path / "lib",
        )
        assert result.exit_code == 2, result.output
        assert "not both" in result.output


def test_help_lists_voices_and_the_alias_matches(invoke):
    top = invoke("--help")
    assert re.search(r"\bvoices\b", top.output)
    option = invoke("voices", "--help")
    alias = invoke("help", "voices")
    assert option.exit_code == alias.exit_code == 0
    assert option.output == alias.output
    for name in ("list", "show", "rename", "forget", "import", "where"):
        assert name in option.output


def test_voices_commands_do_not_write_a_config_template(tmp_path, invoke):
    invoke("voices", "where")
    assert not Path(tmp_path / "voxweave.conf").exists()
