"""Episode transaction publication: permissions, partial-publication reporting,
cleanup failures, read-error messages and temp-file hygiene."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from voxweave import episode_transaction as et
from voxweave import fsio


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _primaries(tmp_path: Path) -> tuple[Path, Path]:
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")
    return json_path, vtt_path


def _commit(json_path: Path, vtt_path: Path, *, command="process", **kwargs):
    return et.commit_primary_outputs(
        command=command,
        episode_path=vtt_path,
        json_path=json_path,
        vtt_path=vtt_path,
        expected_json=et.capture_file_generation(json_path),
        expected_vtt=et.capture_file_generation(vtt_path),
        main_json_bytes=b"new json",
        vtt_bytes=b"new vtt",
        **kwargs,
    )


def _fail_unlink_of(monkeypatch, *targets: Path) -> None:
    original = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if Path(path) in targets:
            raise PermissionError(13, "Permission denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)


# -- permissions --------------------------------------------------------------


def test_primaries_keep_their_mode_and_machine_artifacts_stay_private(
    tmp_path, umask_022
):
    json_path, vtt_path = _primaries(tmp_path)
    os.chmod(json_path, 0o664)
    vtt_path.unlink()
    machine = tmp_path / "voiceprints.json"
    machine.write_bytes(b"old machine")
    os.chmod(machine, 0o644)

    et.commit_primary_outputs(
        command="process",
        episode_path=vtt_path,
        json_path=json_path,
        vtt_path=vtt_path,
        expected_json=et.capture_file_generation(json_path),
        expected_vtt=et.capture_file_generation(vtt_path),
        main_json_bytes=b"new json",
        vtt_bytes=b"new vtt",
        machine_artifact=et.MachineArtifactPublication(machine, b"new machine"),
    )

    assert _mode(json_path) == 0o664  # an existing deliverable keeps its mode
    assert _mode(vtt_path) == 0o644  # a new deliverable honours the umask
    assert _mode(machine) == 0o600  # voice biometrics stay private


def test_align_evidence_is_private(tmp_path, umask_022):
    json_path, vtt_path = _primaries(tmp_path)
    evidence = tmp_path / "episode.align-evidence.json"
    _commit(
        json_path,
        vtt_path,
        command="align",
        evidence_artifact=et.EvidencePublication(evidence, b"{}\n"),
    )
    assert _mode(evidence) == 0o600
    assert _mode(json_path) == 0o644


def test_sdh_and_correction_are_deliverables(tmp_path, umask_022):
    json_path, vtt_path = _primaries(tmp_path)
    os.chmod(vtt_path, 0o640)
    sidecar = tmp_path / "episode.sdh.vtt"
    assert et.commit_auxiliary_sdh(
        episode_path=vtt_path,
        sidecar_path=sidecar,
        sidecar_bytes=b"WEBVTT\n",
        json_path=json_path,
        expected_json=et.capture_file_generation(json_path),
        vtt_path=vtt_path,
        expected_vtt=et.capture_file_generation(vtt_path),
    )
    et.commit_correction(
        episode_path=vtt_path,
        vtt_path=vtt_path,
        expected_vtt=et.capture_file_generation(vtt_path),
        rendered_vtt_bytes=b"corrected",
        evidence_paths=(),
    )
    assert _mode(sidecar) == 0o644
    assert _mode(vtt_path) == 0o640


def test_cache_machine_data_is_private(tmp_path, umask_022, monkeypatch):
    from voxweave import artifacts, pipeline, translate, vocalscache

    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    paths = artifacts.claim_paths(media)
    assert _mode(paths.marker) == 0o600

    progress = paths.translation_progress(tmp_path / "episode.vtt", "zh")
    translate.save_progress(progress, "sig", {0: "hello"})
    assert _mode(progress) == 0o600

    with vocalscache.cache_publish_path(paths.vocals_cache) as staged:
        staged.write_bytes(b"flac")
    assert _mode(paths.vocals_cache) == 0o600

    vtt = tmp_path / "episode.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline.asrfix_mod,
        "correct_cues",
        lambda payload, **kw: [{"i": 0, "orig": "hello", "fixed": "hallo"}],
    )
    pipeline.correct(vtt)
    (audit,) = tmp_path.rglob("*.asrfix.json")
    assert _mode(audit) == 0o600
    assert _mode(tmp_path / "episode.asrfix.vtt") == 0o644  # a deliverable


def test_names_too_long_to_exist_count_as_absent(tmp_path, monkeypatch):
    from voxweave import artifacts, voiceepisode

    media = tmp_path / ("m" * 245 + ".mkv")
    media.write_bytes(b"media")
    too_long = tmp_path / ("m" * 245 + ".speakers.suggest.json")
    assert artifacts.path_present(too_long) is False
    # Only the legacy adjacent lock probe is under test here: its name
    # (<stem>.episode.lock) exceeds NAME_MAX and must read as "no legacy lock".
    monkeypatch.setattr(voiceepisode, "_artifact_lock_paths", lambda _owner: ())
    with voiceepisode.episode_lock(media):
        pass


# -- cleanup failures ---------------------------------------------------------


def test_cleanup_failure_still_publishes_the_machine_artifact(tmp_path, monkeypatch):
    json_path, vtt_path = _primaries(tmp_path)
    suggest = tmp_path / "episode.speakers.suggest.json"
    html = tmp_path / "episode.speakers.html"
    evidence = tmp_path / "episode.align-evidence.json"
    for path in (suggest, html, evidence):
        path.write_bytes(b"stale")
    machine = tmp_path / "voiceprints.json"
    _fail_unlink_of(monkeypatch, suggest, html)

    with pytest.raises(et.ArtifactCleanupError) as caught:
        _commit(
            json_path,
            vtt_path,
            cleanup_paths=(
                et.ArtifactCleanup(suggest, "suggest-unlink"),
                et.ArtifactCleanup(html, "html-unlink"),
                et.ArtifactCleanup(evidence, "evidence-unlink"),
            ),
            machine_artifact=et.MachineArtifactPublication(machine, b"new machine"),
        )

    error = caught.value
    assert error.failure.detail_code == "suggest-unlink"
    assert [s.detail_code for s in error.failure.secondary] == ["html-unlink"]
    assert error.landed == (json_path, vtt_path)
    assert error.machine_landed == (machine,)
    assert machine.read_bytes() == b"new machine"
    assert not evidence.exists()  # the later, independent cleanup still ran
    assert suggest.exists() and html.exists()
    assert str(error).startswith("primary JSON/VTT outputs landed")
    assert "already written: episode.json, episode.vtt, voiceprints.json" in str(error)
    assert not tuple(tmp_path.glob(".*.part*"))


def test_cleanup_failure_withholds_align_evidence(tmp_path, monkeypatch):
    json_path, vtt_path = _primaries(tmp_path)
    voiceprints = tmp_path / "episode.voiceprints.json"
    voiceprints.write_bytes(b"sensitive")
    evidence = tmp_path / "episode.align-evidence.json"
    _fail_unlink_of(monkeypatch, voiceprints)

    with pytest.raises(et.ArtifactCleanupError) as caught:
        _commit(
            json_path,
            vtt_path,
            command="align",
            cleanup_paths=(et.ArtifactCleanup(voiceprints, "voiceprints-unlink"),),
            evidence_artifact=et.EvidencePublication(evidence, b"{}\n"),
        )

    assert caught.value.landed == (json_path, vtt_path)
    assert not evidence.exists()
    assert not tuple(tmp_path.glob(".*.part*"))


def test_machine_replace_failure_after_cleanup_failure_is_secondary(
    tmp_path, monkeypatch
):
    json_path, vtt_path = _primaries(tmp_path)
    suggest = tmp_path / "episode.speakers.suggest.json"
    suggest.write_bytes(b"stale")
    machine = tmp_path / "voiceprints.json"
    _fail_unlink_of(monkeypatch, suggest)
    real_replace = et._replace_stage

    def fail_machine(stage):
        if stage.target == machine:
            raise OSError("disk full")
        return real_replace(stage)

    monkeypatch.setattr(et, "_replace_stage", fail_machine)
    with pytest.raises(et.ArtifactCleanupError) as caught:
        _commit(
            json_path,
            vtt_path,
            cleanup_paths=(et.ArtifactCleanup(suggest, "suggest-unlink"),),
            machine_artifact=et.MachineArtifactPublication(machine, b"new machine"),
        )

    assert caught.value.failure.detail_code == "suggest-unlink"
    assert [(s.kind, s.detail_code) for s in caught.value.failure.secondary] == [
        ("commit-failed", "machine-artifact-replace")
    ]
    assert caught.value.machine_landed == ()
    assert not machine.exists()
    assert not tuple(tmp_path.glob(".*.part*"))


def test_correction_cleanup_failure_names_only_the_vtt(tmp_path, monkeypatch):
    _json_path, vtt_path = _primaries(tmp_path)
    evidence = tmp_path / "episode.align-evidence.json"
    evidence.write_bytes(b"stale")
    _fail_unlink_of(monkeypatch, evidence)

    with pytest.raises(et.ArtifactCleanupError) as caught:
        et.commit_correction(
            episode_path=vtt_path,
            vtt_path=vtt_path,
            expected_vtt=et.capture_file_generation(vtt_path),
            rendered_vtt_bytes=b"corrected",
            evidence_paths=(evidence,),
        )

    message = str(caught.value)
    assert message.startswith("the corrected VTT landed")
    assert "JSON" not in message
    assert caught.value.landed == (vtt_path,)


# -- partial publication and read errors reach the message --------------------


def test_partial_publication_is_named_in_the_error_message(tmp_path, monkeypatch):
    json_path, vtt_path = _primaries(tmp_path)
    real_replace = et._replace_stage

    def fail_vtt(stage):
        if stage.target == vtt_path:
            raise OSError("replace failed")
        return real_replace(stage)

    monkeypatch.setattr(et, "_replace_stage", fail_vtt)
    with pytest.raises(et.TransactionOperationError) as caught:
        _commit(json_path, vtt_path, command="align")
    assert str(caught.value) == "replace failed; already written: episode.json"


def test_unreadable_primary_is_not_reported_as_a_change(tmp_path):
    json_path, vtt_path = _primaries(tmp_path)
    expected_json = et.capture_file_generation(json_path)
    expected_vtt = et.capture_file_generation(vtt_path)
    json_path.unlink()
    json_path.mkdir()  # reading it now fails with IsADirectoryError

    with pytest.raises(et.InputStaleError) as caught:
        et.commit_primary_outputs(
            command="align",
            episode_path=vtt_path,
            json_path=json_path,
            vtt_path=vtt_path,
            expected_json=expected_json,
            expected_vtt=expected_vtt,
            main_json_bytes=b"new json",
            vtt_bytes=b"new vtt",
        )

    assert caught.value.failure.detail_code == "sibling-generation"
    assert "could not re-read episode.json" in str(caught.value)
    assert "changed during" not in str(caught.value)
    assert vtt_path.read_bytes() == b"old vtt"


def test_unreadable_media_is_not_reported_as_a_change(tmp_path):
    with pytest.raises(et.MediaStaleError) as caught:
        et.require_media_generation(tmp_path / "gone.mkv", "0" * 64)
    assert caught.value.failure.detail_code == "media-generation"
    assert "could not re-read the selected media gone.mkv" in str(caught.value)


def test_mapping_stat_does_not_swallow_interrupts():
    class Interrupting:
        def lstat(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        et._mapping_stat(Interrupting())  # type: ignore[arg-type]


def test_import_does_not_load_the_speaker_or_numpy_stack():
    # the module docstring promises no model/renderer dependency
    probe = (
        "import sys, voxweave.episode_transaction; "
        "print(sorted(m for m in ('voxweave.speakers', 'numpy') if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]"


# -- staging names, crash residue, durability ---------------------------------


def test_long_output_names_can_be_staged(tmp_path):
    stem = "字" * 83  # 249 bytes: the longest name that still takes ".vtt"
    json_path = tmp_path / f"{stem[:-1]}.json"
    vtt_path = tmp_path / f"{stem}.vtt"
    json_path.write_bytes(b"old json")
    vtt_path.write_bytes(b"old vtt")
    receipt = et.commit_primary_outputs(
        command="process",
        episode_path=tmp_path / "episode.mkv",
        json_path=json_path,
        vtt_path=vtt_path,
        expected_json=et.capture_file_generation(json_path),
        expected_vtt=et.capture_file_generation(vtt_path),
        main_json_bytes=b"new json",
        vtt_bytes=b"new vtt",
    )
    assert receipt.landed == (json_path, vtt_path)
    assert vtt_path.read_bytes() == b"new vtt"
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == sorted(
        [json_path.name, vtt_path.name]
    )


def test_staging_sweeps_stale_residue_and_fsyncs_the_directory(tmp_path, monkeypatch):
    json_path, vtt_path = _primaries(tmp_path)
    old = 1_000_000_000
    residue = tmp_path / ".episode.abcdefgh.part.vtt"
    residue.write_bytes(b"crashed stage")
    os.utime(residue, (old, old))
    synced: list[Path] = []
    real_fsync_directory = fsio.fsync_directory

    def record(directory):
        synced.append(Path(directory))
        real_fsync_directory(directory)

    monkeypatch.setattr(fsio, "fsync_directory", record)
    _commit(json_path, vtt_path)

    assert not residue.exists()
    # after each primary rename (the episode lock's own cache writes aside)
    assert [directory for directory in synced if directory == tmp_path] == [
        tmp_path,
        tmp_path,
    ]
