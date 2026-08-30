import copy
import errno
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from voxweave import artifacts, pipeline, speakers
from voxweave.cli import cli
from voxweave.voicebase import (
    canonical_turns_digest,
    media_fingerprint,
    write_voiceprints,
)
from voxweave.voicematch import load_suggest
from voxweave.voicestore import (
    enroll_exemplar,
    load_voice_store,
    new_voice_store,
    write_voice_store,
)


CAPTURE = "c" + "1" * 32
PRIOR_CAPTURE = "c" + "2" * 32
QUERY_VECTOR = [1.0, *([0.0] * 15)]
OTHER_VECTOR = [0.0, 1.0, *([0.0] * 14)]
PROVENANCE = {
    "diarization_model": "example/diarizer",
    "outer_config_sha256": "a" * 64,
    "embedding_model": "example/embedder",
    "embedding_checkpoint": "b" * 64,
    "embedding_dim": 16,
    "audio": {"separated": False, "normalized": False, "sample_rate": 16000},
    "pyannote_version": "3.4.0",
    "torch_version": "test",
}


@pytest.fixture(autouse=True)
def _isolate_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_CONFIG", str(tmp_path / "voxweave.conf"))
    monkeypatch.setenv("VOXWEAVE_CACHE_ROOT", str(tmp_path / "cache-root"))
    monkeypatch.delenv("VOXWEAVE_VOICEPRINTS", raising=False)
    monkeypatch.delenv("VOXWEAVE_VOICES_ACCEPT", raising=False)
    monkeypatch.delenv("VOXWEAVE_VOICES_SUGGEST", raising=False)
    monkeypatch.delenv("VOXWEAVE_VOICES_MARGIN", raising=False)


def _write_episode(
    tmp_path: Path,
    *,
    labels=("SPEAKER_00",),
    vectors=None,
    provenance=None,
):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"episode media")
    vectors = vectors or {label: list(QUERY_VECTOR) for label in labels}
    turns = []
    units = []
    cursor = 0.0
    for index, label in enumerate(labels):
        duration = 4.0 if index == 0 else 2.0
        turns.append([cursor, cursor + duration, label])
        units.append(
            {
                "text": f"line{index}",
                "start": cursor,
                "end": cursor + duration,
            }
        )
        cursor += duration
    fingerprint = media_fingerprint(media)
    sibling = {
        "language": "en",
        "segments": [],
        "word_segments": units,
        "vad_speech": [[0.0, cursor]],
        "speaker_turns": turns,
        "voiceprint_capture": CAPTURE,
        "voiceprint_media": fingerprint,
    }
    (tmp_path / "episode.json").write_text(
        json.dumps(sibling, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sidecar = {
        "version": 1,
        "capture_id": CAPTURE,
        "provenance": copy.deepcopy(provenance or PROVENANCE),
        "binding": {
            "turns_digest": canonical_turns_digest(turns),
            "media_fingerprint": fingerprint,
            "media_stem": "episode",
            "created": "2026-08-28T00:00:00Z",
        },
        "speakers": vectors,
    }
    write_voiceprints(tmp_path / "episode.voiceprints.json", sidecar)
    return media, sibling, sidecar


def _write_store(path: Path, *, name="Aqua", vector=None, provenance=None):
    store = new_voice_store("Example Show", provenance or PROVENANCE)
    store = enroll_exemplar(
        store,
        raw_name=name,
        capture_id=PRIOR_CAPTURE,
        media_fingerprint="f" * 64,
        episode="prior",
        vector=vector or QUERY_VECTOR,
    ).store
    write_voice_store(path, store)
    return store


def _fake_clips(monkeypatch, seen=None):
    def extract(source, _start, _end, output):
        if seen is not None:
            seen.append(Path(source))
        Path(output).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "extract_clip", extract)


def test_matching_prefill_stays_html_only_and_mapping_empty(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    store_path = tmp_path / "voices.json"
    _write_store(store_path)
    seen_sources: list[Path] = []
    _fake_clips(monkeypatch, seen_sources)
    monkeypatch.setenv("VOXWEAVE_VOICES_ACCEPT", "0.9")

    audition = speakers.create_speaker_audition(media, voices=store_path)

    paths = artifacts.claim_paths(media)
    mapping = json.loads(paths.speaker_mapping.read_text())
    assert mapping == {"version": 1, "speakers": {"SPEAKER_00": ""}}
    page = audition.page
    assert 'value="Aqua"' in page
    assert "machine-suggested" in page
    assert audition.pristine_mapping_generation is not None
    suggestion = load_suggest(paths.speaker_suggest)
    assert suggestion["speakers"]["SPEAKER_00"]["decision"] == "prefill"
    assert seen_sources and all(source != media for source in seen_sources)
    assert all(not source.exists() for source in seen_sources)

    rendered = pipeline.split(tmp_path / "episode.json").read_text(encoding="utf-8")
    assert "Aqua" not in rendered


def test_store_names_are_context_escaped_and_vectors_never_render(
    tmp_path, monkeypatch
):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    store_path = tmp_path / "voices.json"
    malicious = '"><script>alert(1)</script>\u2028"quoted"'
    _write_store(store_path, name=malicious)
    _fake_clips(monkeypatch)

    page = speakers.create_speaker_audition(
        media,
        voices=store_path,
    ).page

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&quot;" in page
    assert str(QUERY_VECTOR) not in page


def test_no_match_is_manual_escape_and_mints_no_snapshot(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    sibling_path = tmp_path / "episode.json"
    sibling = json.loads(sibling_path.read_text())
    sibling["voiceprint_capture"] = "invalid"
    sibling_path.write_text(json.dumps(sibling), encoding="utf-8")
    (tmp_path / "episode.speakers.suggest.json").write_text("stale", encoding="utf-8")
    _fake_clips(monkeypatch)

    class ForbiddenSnapshot:
        def __init__(self, _path):
            raise AssertionError("--no-match must stay snapshot-free")

    monkeypatch.setattr(speakers, "MediaSnapshot", ForbiddenSnapshot)
    speakers.create_speaker_audition(media, no_match=True)

    assert artifacts.claim_paths(media).speaker_mapping.exists()
    assert not (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()


def test_declared_missing_sidecar_refuses_without_mapping(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    (tmp_path / "episode.voiceprints.json").unlink()
    (tmp_path / "episode.speakers.suggest.json").write_text("stale", encoding="utf-8")
    _fake_clips(monkeypatch)

    with pytest.raises(RuntimeError, match="declared but not usable"):
        speakers.create_speaker_audition(media)

    assert not (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()


def test_discovery_requires_show_confirmation(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    _write_store(tmp_path / "voxweave.voices.json")
    _fake_clips(monkeypatch)

    page = speakers.create_speaker_audition(media).page

    assert "machine-suggested" not in page
    assert not (tmp_path / "episode.speakers.suggest.json").exists()


def test_explicit_corrupt_store_is_fatal_but_discovered_corrupt_store_is_manual(
    tmp_path, monkeypatch
):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{broken", encoding="utf-8")
    _fake_clips(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit voices store.*unusable"):
        speakers.create_speaker_audition(media, voices=explicit)
    assert not (tmp_path / "episode.speakers.json").exists()

    discovered = tmp_path / "voxweave.voices.json"
    discovered.write_text("{broken", encoding="utf-8")
    audition = speakers.create_speaker_audition(media)
    assert audition.page.startswith("<!doctype html>")
    assert artifacts.claim_paths(media).speaker_mapping.exists()
    assert not (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()


def test_voices_store_cannot_use_episode_artifact_namespace(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    forbidden = tmp_path / "episode.voices.json"
    _write_store(forbidden)
    _fake_clips(monkeypatch)

    with pytest.raises(RuntimeError, match="episode namespace"):
        speakers.create_speaker_audition(media, voices=forbidden)

    assert not (tmp_path / "episode.speakers.json").exists()


def test_generation_detects_sibling_change_before_mapping_commit(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    sibling_path = tmp_path / "episode.json"

    def mutate(_source, _start, _end, output):
        payload = json.loads(sibling_path.read_text())
        payload["language"] = "changed"
        sibling_path.write_text(json.dumps(payload), encoding="utf-8")
        Path(output).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "extract_clip", mutate)
    with pytest.raises(RuntimeError, match="input changed"):
        speakers.create_speaker_audition(media)

    assert not (tmp_path / "episode.speakers.json").exists()


def test_first_clip_reads_stable_snapshot_through_truncate_aba(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    original = media.read_bytes()
    seen: list[Path] = []

    def mutate_live(source, _start, _end, output):
        source = Path(source)
        seen.append(source)
        assert source != media
        with media.open("r+b") as live:
            live.truncate(0)
            live.write(b"B" * len(original))
            live.flush()
            live.seek(0)
            live.truncate(0)
            live.write(original)
            live.flush()
        assert source.read_bytes() == original
        Path(output).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "extract_clip", mutate_live)

    speakers.create_speaker_audition(media)

    assert artifacts.claim_paths(media).speaker_mapping.exists()
    assert not (tmp_path / "episode.speakers.json").exists()
    assert seen and not seen[0].exists()


def test_live_media_change_after_first_clip_aborts_without_mapping(
    tmp_path, monkeypatch
):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    original = media.read_bytes()

    def replace_live(source, _start, _end, output):
        assert Path(source) != media
        assert Path(source).read_bytes() == original
        media.write_bytes(b"replacement media")
        Path(output).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "extract_clip", replace_live)

    with pytest.raises(RuntimeError, match="media changed"):
        speakers.create_speaker_audition(media)

    assert not (tmp_path / "episode.speakers.json").exists()


def test_failed_no_match_clip_clears_stale_suggestion(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    suggest_path = tmp_path / "episode.speakers.suggest.json"
    suggest_path.write_text('{"stale": true}', encoding="utf-8")
    monkeypatch.setattr(
        speakers,
        "extract_clip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("clip edge")),
    )

    with pytest.raises(OSError, match="clip edge"):
        speakers.create_speaker_audition(media, no_match=True)

    assert not suggest_path.exists()
    assert not (tmp_path / "episode.speakers.json").exists()


@pytest.mark.parametrize(
    ("sibling_bytes", "message"),
    [
        (b"{broken", "invalid sibling JSON"),
        (b"\xff\xfe", "invalid sibling JSON"),
        (b"[]", "expected object"),
    ],
    ids=["malformed-json", "invalid-utf8", "non-object-root"],
)
def test_invalid_sibling_clears_stale_suggestion(tmp_path, sibling_bytes, message):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    (tmp_path / "episode.json").write_bytes(sibling_bytes)
    suggest_path = tmp_path / "episode.speakers.suggest.json"
    suggest_path.write_text('{"stale": true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        speakers.create_speaker_audition(media, no_match=True)

    assert not (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.speakers.html").exists()
    assert not suggest_path.exists()


def test_store_reread_failure_clears_stale_suggestion(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    store_path = tmp_path / "voices.json"
    _write_store(store_path)
    suggest_path = tmp_path / "episode.speakers.suggest.json"
    suggest_path.write_text('{"stale": true}', encoding="utf-8")

    def corrupt_store(_source, _start, _end, output):
        store_path.write_text("{broken", encoding="utf-8")
        Path(output).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "extract_clip", corrupt_store)

    with pytest.raises(Exception, match="invalid voices.json"):
        speakers.create_speaker_audition(media, voices=store_path)

    assert not suggest_path.exists()
    assert not (tmp_path / "episode.speakers.json").exists()


def test_generation_rechecks_live_media_after_matching_decision(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    store_path = tmp_path / "voices.json"
    _write_store(store_path)
    _fake_clips(monkeypatch)
    original_matching = speakers._matching_record

    def mutate_after_matching(*args, **kwargs):
        result = original_matching(*args, **kwargs)
        media.write_bytes(b"replacement after matching work")
        return result

    monkeypatch.setattr(speakers, "_matching_record", mutate_after_matching)

    with pytest.raises(RuntimeError, match="media changed"):
        speakers.create_speaker_audition(media, voices=store_path)

    assert not (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.speakers.html").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()


def test_generation_rechecks_live_media_at_mapping_install_edge(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    store_path = tmp_path / "voices.json"
    _write_store(store_path)
    _fake_clips(monkeypatch)
    mapping_path = artifacts.claim_paths(media).speaker_mapping
    original_mapping_write = speakers.fsio.atomic_write_text_new
    mutated = []

    def mutate_before_mapping_install(path, content, **kwargs):
        if Path(path) == mapping_path:
            media.write_bytes(b"replacement media in sampled region")
            mutated.append(True)
        return original_mapping_write(path, content, **kwargs)

    monkeypatch.setattr(
        speakers.fsio,
        "atomic_write_text_new",
        mutate_before_mapping_install,
    )

    with pytest.raises(RuntimeError, match="media changed"):
        speakers.create_speaker_audition(media, voices=store_path)

    assert mutated == [True]
    assert not mapping_path.exists()
    assert not (tmp_path / "episode.speakers.html").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()


@pytest.mark.parametrize(
    "install_route",
    ["hard-link", "o-excl-fallback"],
)
def test_generation_combined_mapping_and_media_race_cleans_only_machine_outputs(
    tmp_path, monkeypatch, install_route
):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    store_path = tmp_path / "voices.json"
    _write_store(store_path)
    _fake_clips(monkeypatch)
    mapping_path = tmp_path / "episode.speakers.json"
    suggest_path = tmp_path / "episode.speakers.suggest.json"
    human_mapping = {
        "version": 1,
        "speakers": {"SPEAKER_00": "Human-reviewed name"},
    }
    human_bytes = json.dumps(human_mapping, ensure_ascii=False).encode("utf-8")
    original_publish = speakers._publish_audition
    race_injected = []
    callback_calls = []

    def inject_race():
        mapping_path.write_bytes(human_bytes)
        media.write_bytes(b"replacement media in sampled region")
        race_injected.append(True)

    def publish_with_race(*args, **kwargs):
        original_callback = kwargs["before_mapping_install"]

        def callback_with_race():
            callback_calls.append(True)
            if install_route == "hard-link" or len(callback_calls) == 2:
                inject_race()
            original_callback()

        kwargs["before_mapping_install"] = callback_with_race
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(speakers, "_publish_audition", publish_with_race)
    if install_route == "o-excl-fallback":

        def hard_links_unavailable(*_args, **_kwargs):
            raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

        monkeypatch.setattr(speakers.fsio.os, "link", hard_links_unavailable)

    with pytest.raises(RuntimeError, match="media changed"):
        speakers.create_speaker_audition(media, voices=store_path)

    assert race_injected == [True]
    assert mapping_path.read_bytes() == human_bytes
    assert not suggest_path.exists()
    assert not (tmp_path / "episode.speakers.html").exists()


def test_same_token_sidecar_content_change_aborts_revalidation(tmp_path, monkeypatch):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    sidecar_path = tmp_path / "episode.voiceprints.json"
    suggest_path = tmp_path / "episode.speakers.suggest.json"
    suggest_path.write_text('{"stale": true}', encoding="utf-8")

    def mutate_sidecar(_source, _start, _end, output):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["speakers"]["SPEAKER_00"] = list(OTHER_VECTOR)
        write_voiceprints(sidecar_path, sidecar)
        Path(output).write_bytes(b"mp3")

    monkeypatch.setattr(speakers, "extract_clip", mutate_sidecar)

    with pytest.raises(RuntimeError, match="input changed"):
        speakers.create_speaker_audition(media)

    assert not (tmp_path / "episode.speakers.json").exists()
    assert not suggest_path.exists()


@pytest.mark.parametrize("failed_edge", ["suggest", "mapping"])
def test_generation_publication_edge_failure_leaves_no_mapping_or_machine_outputs(
    tmp_path, monkeypatch, failed_edge
):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    store_path = tmp_path / "voices.json"
    _write_store(store_path)
    _fake_clips(monkeypatch)
    if failed_edge == "suggest":
        monkeypatch.setattr(
            speakers,
            "write_suggest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("suggest edge")),
        )
    else:
        monkeypatch.setattr(
            speakers.fsio,
            "atomic_write_text_new",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("mapping edge")),
        )

    with pytest.raises(OSError, match=f"{failed_edge} edge"):
        speakers.create_speaker_audition(media, voices=store_path)

    assert not (tmp_path / "episode.speakers.json").exists()
    assert not (tmp_path / "episode.speakers.suggest.json").exists()
    assert not (tmp_path / "episode.speakers.html").exists()


def test_enrollment_store_write_failure_preserves_mapping_and_no_store(
    tmp_path, monkeypatch
):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    mapping_path = tmp_path / "episode.speakers.json"
    mapping_path.write_text(
        json.dumps({"version": 1, "speakers": {"SPEAKER_00": "Aqua"}}),
        encoding="utf-8",
    )
    before = mapping_path.read_bytes()
    store_path = tmp_path / "voices.json"
    monkeypatch.setattr(
        speakers,
        "write_voice_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("store edge")),
    )

    with pytest.raises(OSError, match="store edge"):
        speakers.enroll_speaker_voices(
            media,
            voices=store_path,
            show="Example Show",
        )

    assert mapping_path.read_bytes() == before
    assert not store_path.exists()


def test_purge_io_failure_is_nonzero(tmp_path):
    media = tmp_path / "missing.mkv"
    sidecar_path = tmp_path / "missing.voiceprints.json"
    sidecar_path.mkdir()

    with pytest.raises(OSError):
        speakers.purge_voiceprints(media)


def test_enrollment_creates_store_from_human_mapping(tmp_path):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    mapping_path = tmp_path / "episode.speakers.json"
    mapping_path.write_text(
        json.dumps({"version": 1, "speakers": {"SPEAKER_00": "Aqua"}}),
        encoding="utf-8",
    )
    before_mapping = mapping_path.read_bytes()
    store_path = tmp_path / "new-voices.json"

    output = speakers.enroll_speaker_voices(
        media,
        voices=store_path,
        show="Example Show",
    )

    store, validated = load_voice_store(output)
    identity = next(iter(store["identities"].values()))
    exemplar = identity["exemplars"][0]
    assert validated.show == "Example Show"
    assert identity["display_name"] == "Aqua"
    assert exemplar["capture_id"] == CAPTURE
    assert mapping_path.read_bytes() == before_mapping


def test_enrollment_decodes_each_in_lock_evidence_observation_once(
    tmp_path, monkeypatch
):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    sibling_path = tmp_path / "episode.json"
    sidecar_path = tmp_path / "episode.voiceprints.json"
    mapping_path = tmp_path / "episode.speakers.json"
    mapping_path.write_text(
        json.dumps({"version": 1, "speakers": {"SPEAKER_00": "Aqua"}}),
        encoding="utf-8",
    )
    observed = {path: 0 for path in (sibling_path, sidecar_path, mapping_path)}
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path):
        if path in observed:
            observed[path] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    speakers.enroll_speaker_voices(
        media,
        voices=tmp_path / "voices.json",
        show="Example Show",
    )

    # One staging observation and one authoritative observation under both
    # locks. Parsing must use those bytes rather than opening the paths again.
    assert observed == {
        sibling_path: 2,
        sidecar_path: 2,
        mapping_path: 2,
    }


def test_enrollment_uses_longest_turn_for_one_identity(tmp_path):
    media, _sibling, _sidecar = _write_episode(
        tmp_path,
        labels=("SPEAKER_LONG", "SPEAKER_SHORT"),
        vectors={
            "SPEAKER_LONG": list(QUERY_VECTOR),
            "SPEAKER_SHORT": list(OTHER_VECTOR),
        },
    )
    (tmp_path / "episode.speakers.json").write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {"SPEAKER_LONG": "Aqua", "SPEAKER_SHORT": "Aqua"},
            }
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "voices.json"

    speakers.enroll_speaker_voices(
        media,
        voices=store_path,
        show="Example Show",
    )

    store, _validated = load_voice_store(store_path)
    identity = next(iter(store["identities"].values()))
    assert identity["exemplars"][0]["vector"] == QUERY_VECTOR


def test_multi_identity_enrollment_increments_revision_once_per_write(tmp_path):
    media, _sibling, _sidecar = _write_episode(
        tmp_path,
        labels=("SPEAKER_A", "SPEAKER_B"),
        vectors={
            "SPEAKER_A": list(QUERY_VECTOR),
            "SPEAKER_B": list(OTHER_VECTOR),
        },
    )
    (tmp_path / "episode.speakers.json").write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": {"SPEAKER_A": "Aqua", "SPEAKER_B": "Beryl"},
            }
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "voices.json"

    speakers.enroll_speaker_voices(
        media,
        voices=store_path,
        show="Example Show",
    )

    store, validated = load_voice_store(store_path)
    assert len(store["identities"]) == 2
    assert validated.revision == 1


def test_exact_repeat_enrollment_is_byte_noop(tmp_path):
    media, _sibling, _sidecar = _write_episode(tmp_path)
    (tmp_path / "episode.speakers.json").write_text(
        json.dumps({"version": 1, "speakers": {"SPEAKER_00": "Aqua"}}),
        encoding="utf-8",
    )
    store_path = tmp_path / "voices.json"
    speakers.enroll_speaker_voices(
        media,
        voices=store_path,
        show="Example Show",
    )
    before = store_path.read_bytes()

    speakers.enroll_speaker_voices(media, voices=store_path)

    assert store_path.read_bytes() == before


def test_unknown_compatibility_refuses_store_creation(tmp_path):
    provenance = copy.deepcopy(PROVENANCE)
    provenance["outer_config_sha256"] = "unresolved"
    media, _sibling, _sidecar = _write_episode(tmp_path, provenance=provenance)
    (tmp_path / "episode.speakers.json").write_text(
        json.dumps({"version": 1, "speakers": {"SPEAKER_00": "Aqua"}}),
        encoding="utf-8",
    )
    store_path = tmp_path / "voices.json"

    with pytest.raises(Exception, match="unresolved"):
        speakers.enroll_speaker_voices(
            media,
            voices=store_path,
            show="Example Show",
        )

    assert not store_path.exists()


def test_purge_works_after_media_deletion(tmp_path):
    media = tmp_path / "missing.mkv"
    targets = [
        tmp_path / "missing.voiceprints.json",
        tmp_path / "missing.speakers.suggest.json",
        tmp_path / "missing.speakers.html",
    ]
    for target in targets:
        target.write_text("sensitive", encoding="utf-8")

    removed = speakers.purge_voiceprints(media)

    assert set(removed) == set(targets)
    assert not any(target.exists() for target in targets)


def test_cli_voiceprints_env_requires_diarization_source(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    monkeypatch.setenv("VOXWEAVE_VOICEPRINTS", "1")

    result = CliRunner().invoke(cli, [str(media)])

    assert result.exit_code != 0
    assert "environment VOXWEAVE_VOICEPRINTS" in result.output
    assert "built-in default" in result.output


def test_cli_voiceprints_config_error_names_both_config_sources(tmp_path):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    (tmp_path / "voxweave.conf").write_text(
        "[defaults]\ndiarize = false\nvoiceprints = true\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, [str(media)])

    assert result.exit_code != 0
    assert "config [defaults].voiceprints" in result.output
    assert "config [defaults].diarize" in result.output


def test_cli_voiceprints_precedence_reaches_process(tmp_path, monkeypatch):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    output = tmp_path / "episode.vtt"
    seen = {}
    monkeypatch.setenv("VOXWEAVE_VOICEPRINTS", "1")

    def fake_process(_media, **kwargs):
        seen.update(kwargs)
        return output

    monkeypatch.setattr(pipeline, "process", fake_process)
    result = CliRunner().invoke(
        cli,
        ["--diarize", "--no-voiceprints", str(media)],
    )

    assert result.exit_code == 0, result.output
    assert seen["diarize"] is True
    assert seen["voiceprints"] is False


def test_cli_purge_accepts_missing_media_and_rejects_cross_mode_options(tmp_path):
    media = tmp_path / "missing.mkv"
    sidecar = tmp_path / "missing.voiceprints.json"
    sidecar.write_text("sensitive", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["speakers", str(media), "--purge-voiceprints"],
    )
    assert result.exit_code == 0, result.output
    assert str(sidecar) in result.output
    assert not sidecar.exists()

    invalid = CliRunner().invoke(
        cli,
        ["speakers", str(media), "--purge-voiceprints", "--no-match"],
    )
    assert invalid.exit_code == 2
    assert "mutually exclusive" in invalid.output
