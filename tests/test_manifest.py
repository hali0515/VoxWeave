# tests/test_manifest.py
"""P3 PIN 3 -- the SegmentationManifest: build, persistence, legacy inference.

The sibling JSON has carried no configuration fingerprint at all, so every file
in the wild is indistinguishable from a default-profile one. P3 adds exactly one
top-level key, ``segmentation``, on the process/split path:

* ``split``/``process`` REGENERATE it (they re-segment),
* ``align`` PRESERVES it verbatim (it never re-segments) and never invents one,
* a document without the key is legacy-v1 by definition
  (``pipeline.resolve_segmentation_manifest``).

Key order is pinned: ``segmentation`` goes last, after ``speaker_turns``, so the
byte-diff harness can strip it and compare the rest of the document unchanged.
"""

import copy
import json
import os
import platform
from unittest.mock import patch

import pytest

from voxweave import pipeline
from voxweave.core import providers
from voxweave.core.smart_split import SplitThresholds

THRESHOLD_KEYS = (
    "clause_ms",
    "vad_skip_ms",
    "offline_ms",
    "min_cue_s",
    "max_cue_s",
    "glue_gap_s",
    "cps",
    "lag_out_s",
    "shot_snap_s",
)
PROFILE_KEYS = set(THRESHOLD_KEYS) | {"max_line_length", "max_lines"}
#: The persisted ``segmentation`` block, in insertion order -- the order IS the
#: bytes, so it is pinned rather than compared as a set.
MANIFEST_KEY_ORDER = (
    "manifest_version",
    "engine",
    "voxweave",
    "python",
    "language",
    "profile",
    "env",
    "providers",
    "degraded",
)
PROFILE_KEY_ORDER = ("max_line_length", "max_lines", *THRESHOLD_KEYS)

EN_UNITS = [
    {"text": "Where", "start": 0.0, "end": 0.4},
    {"text": "did", "start": 0.5, "end": 0.8},
    {"text": "you", "start": 0.9, "end": 1.2},
    {"text": "go", "start": 1.4, "end": 2.0},
    {"text": "Nowhere", "start": 2.4, "end": 3.0},
    {"text": "special", "start": 3.1, "end": 3.6},
]
FULL_CONTEXT = {
    "vad_speech": [[0.0, 2.0], [2.4, 3.6]],
    "shot_changes": [2.3, 3.9],
    "sing_spans": [[0.0, 0.5]],
    "speaker_turns": [[0.0, 2.2, "SPEAKER_00"], [2.3, 3.8, "SPEAKER_01"]],
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VOXWEAVE_GAP_ADAPTIVE", raising=False)
    monkeypatch.delenv("VOXWEAVE_VAD_EMISSION_MASK", raising=False)


def _write_case(tmp_path, name="ep", language="en", units=None, **extra):
    doc = {
        "language": language,
        "word_segments": copy.deepcopy(EN_UNITS if units is None else units),
    }
    doc.update(extra)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_json_scalars(value):
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str)
            _assert_json_scalars(v)
    elif isinstance(value, list):
        for item in value:
            _assert_json_scalars(item)
    else:
        assert value is None or isinstance(value, (str, int, float, bool))


# --- build + persistence on the split path -----------------------------------


def test_split_writes_a_segmentation_manifest(tmp_path):
    path = _write_case(tmp_path)
    pipeline.split(path)
    manifest = _read(path)["segmentation"]

    assert manifest["manifest_version"] == 1
    assert manifest["engine"] == "legacy-v1"
    assert manifest["language"] == "en"
    assert isinstance(manifest["voxweave"], str) and manifest["voxweave"]
    assert manifest["python"] == platform.python_version()
    assert set(manifest["profile"]) == PROFILE_KEYS
    assert manifest["env"] == {"gap_adaptive": False, "vad_emission_mask": False}
    assert manifest["providers"] == providers.provider_snapshot("en")
    assert manifest["degraded"] == []


def test_persisted_manifest_key_order_is_pinned(tmp_path):
    """JSON preserves insertion order, so the block's key order IS its bytes.

    ``degraded`` is the last key even though it is the only value that cannot be
    known before the engine runs: it is inserted empty with the rest and filled
    in place afterwards, which is what keeps this order independent of when the
    ledger arrives.
    """
    path = _write_case(tmp_path)
    pipeline.split(path)
    manifest = _read(path)["segmentation"]

    assert list(manifest) == list(MANIFEST_KEY_ORDER)
    assert list(manifest["profile"]) == list(PROFILE_KEY_ORDER)
    assert list(manifest["env"]) == ["gap_adaptive", "vad_emission_mask"]
    assert list(manifest["providers"]) == ["sentences", "atoms", "pos"]


def test_manifest_profile_records_what_actually_ran(tmp_path):
    path = _write_case(tmp_path)
    pipeline.split(path)
    manifest = _read(path)["segmentation"]

    result = pipeline.segment_document(
        language="en", word_segments=copy.deepcopy(EN_UNITS)
    )
    for key in THRESHOLD_KEYS:
        assert manifest["profile"][key] == result.thresholds_used[key], key
    # no layout override -> the language defaults the engine actually used
    assert manifest["profile"]["max_line_length"] == 42
    assert manifest["profile"]["max_lines"] == 2


def test_manifest_profile_records_a_layout_override(tmp_path):
    path = _write_case(tmp_path)
    pipeline.split(path, max_line_length=12, max_lines=1)
    profile = _read(path)["segmentation"]["profile"]
    assert profile["max_line_length"] == 12
    assert profile["max_lines"] == 1


def test_manifest_profile_records_the_language_layout_defaults(tmp_path):
    zh_units = [
        {"text": ch, "start": 0.3 * i, "end": 0.3 * i + 0.25}
        for i, ch in enumerate("今天的天气真好。我们一起出去走走吧。")
    ]
    path = _write_case(tmp_path, name="zh", language="zh", units=zh_units)
    pipeline.split(path)
    manifest = _read(path)["segmentation"]
    assert manifest["language"] == "zh"
    assert manifest["profile"]["max_line_length"] == 18
    assert manifest["profile"]["max_lines"] == 1
    assert manifest["providers"] == providers.provider_snapshot("zh")


def test_manifest_env_reports_the_vad_emission_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    path = _write_case(tmp_path)
    pipeline.split(path)
    assert _read(path)["segmentation"]["env"]["vad_emission_mask"] is True


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("1", True),
        (" 1 ", True),  # the consumer strips before comparing
        ("0", False),  # what ``cli._apply_vad_mask`` writes for --no-vad-mask
        ("", False),
        ("true", False),  # truthy as a string, but the consumer wants "1"
        ("off", False),
        (None, False),  # unset
    ],
)
def test_manifest_env_vad_emission_mask_mirrors_its_only_consumer(
    env_value, expected, monkeypatch
):
    """The field must agree with ``align_ctc``, which masks iff the value is "1".

    ``--no-vad-mask`` writes the literal string ``"0"``, so a plain ``bool()`` of
    the env var would record masking as ON for the run that explicitly turned it
    off -- a persisted, permanent lie in a record whose whole point is saying
    what ran.
    """
    if env_value is None:
        monkeypatch.delenv("VOXWEAVE_VAD_EMISSION_MASK", raising=False)
    else:
        monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", env_value)
    result = pipeline.segment_document(
        language="en", word_segments=copy.deepcopy(EN_UNITS)
    )
    assert result.manifest is not None
    assert result.manifest["env"]["vad_emission_mask"] is expected
    # ... and it agrees with what the aligner would actually do with that value.
    masks = os.environ.get("VOXWEAVE_VAD_EMISSION_MASK", "").strip() == "1"
    assert masks is expected


def test_manifest_env_gap_adaptive_is_true_only_when_values_changed(monkeypatch):
    """``gap_adaptive`` records whether the adaptive pass actually replaced
    values -- not merely that the env var was set."""
    monkeypatch.setenv("VOXWEAVE_GAP_ADAPTIVE", "1")
    units = []
    t = 0.0
    for _ in range(100):
        units.append({"text": "x", "start": t, "end": t + 0.2})
        t += 0.8
    result = pipeline.segment_document(language="en", word_segments=units)
    assert result.diagnostics["adaptive_thresholds"] is True
    assert result.manifest["env"]["gap_adaptive"] is True

    # too few samples to estimate -> the mapping is untouched
    short = pipeline.segment_document(
        language="en", word_segments=copy.deepcopy(EN_UNITS)
    )
    assert short.diagnostics["adaptive_thresholds"] is False
    assert short.manifest["env"]["gap_adaptive"] is False


def test_manifest_env_gap_adaptive_compares_values_not_object_identity(monkeypatch):
    """An estimate equal to the static clause replaced nothing.

    ``_maybe_adaptive_thresholds`` hands back a NEW dict whenever the estimator
    produces a number, so an identity check calls that run adaptive even though
    every value is the one the static profile already had. The spec asks for
    "actually replaced values", which only a value comparison answers.
    """
    from voxweave.config import gap_thresholds
    from voxweave.core import gap_split

    monkeypatch.setenv("VOXWEAVE_GAP_ADAPTIVE", "1")
    static = gap_thresholds("en")
    # the estimator "fires" but lands exactly on the static clause_ms, so the
    # derived offline_ms (same clause:offline ratio) is unchanged too
    monkeypatch.setattr(
        gap_split, "adaptive_clause_ms", lambda gaps: static["clause_ms"]
    )
    units = []
    t = 0.0
    for _ in range(100):
        units.append({"text": "x", "start": t, "end": t + 0.2})
        t += 0.8
    result = pipeline.segment_document(language="en", word_segments=units)

    assert result.manifest is not None
    assert result.thresholds_used == static  # value-identical to the static set
    assert result.manifest["env"]["gap_adaptive"] is False


def test_segment_document_accepts_a_partial_thresholds_mapping():
    """A partial mapping is legal input -- the engine fills the rest.

    ``smart_split_segments`` normalizes whatever mapping it is handed through
    ``SplitThresholds.from_mapping`` ("possibly partial ... filling defaults"),
    so quoting the caller's raw mapping in the manifest would both crash on a
    partial one and, if it did not, describe values the engine never used.
    """
    partial = {"clause_ms": 400}
    result = pipeline.segment_document(
        language="en", word_segments=copy.deepcopy(EN_UNITS), thresholds=partial
    )
    assert result.cues  # the call completes instead of raising KeyError
    assert result.manifest is not None
    profile = result.manifest["profile"]
    assert set(profile) == PROFILE_KEYS
    assert profile["clause_ms"] == 400  # the one value the caller supplied
    # The other eight are the dataclass defaults the engine ran on -- NOT
    # ``config.gap_thresholds`` (which would say cps=17.0, lag_out_s=0.25).
    defaults = SplitThresholds.from_mapping(dict(partial))
    for key in THRESHOLD_KEYS:
        assert profile[key] == getattr(defaults, key), key
    assert profile["cps"] == 0.0
    assert profile["lag_out_s"] == 0.0
    assert profile["max_cue_s"] == 7.0
    assert profile["shot_snap_s"] == pytest.approx(11.0 / 24.0)
    # the IR profile records the same resolved values
    assert result.document is not None
    for key in THRESHOLD_KEYS:
        assert getattr(result.document.profile, key) == float(profile[key]), key


def test_segment_document_accepts_an_empty_thresholds_mapping():
    result = pipeline.segment_document(
        language="en", word_segments=copy.deepcopy(EN_UNITS), thresholds={}
    )
    assert result.cues
    assert result.manifest is not None
    assert result.manifest["profile"]["clause_ms"] == SplitThresholds().clause_ms


def test_manifest_is_json_serializable_scalars_only(tmp_path):
    path = _write_case(tmp_path, **FULL_CONTEXT)
    pipeline.split(path)
    _assert_json_scalars(_read(path)["segmentation"])


def test_segmentation_is_the_last_top_level_key(tmp_path):
    path = _write_case(tmp_path, **FULL_CONTEXT)
    pipeline.split(path)
    assert list(_read(path)) == [
        "language",
        "segments",
        "word_segments",
        "vad_speech",
        "shot_changes",
        "sing_spans",
        "speaker_turns",
        "segmentation",
    ]


def test_split_regenerates_a_stale_manifest(tmp_path):
    stale = {
        "manifest_version": 0,
        "engine": "some-future-engine",
        "language": "de",
        "marker": "stale",
    }
    path = _write_case(tmp_path, segmentation=stale)
    pipeline.split(path)
    manifest = _read(path)["segmentation"]
    assert manifest["engine"] == "legacy-v1"
    assert manifest["manifest_version"] == 1
    assert manifest["language"] == "en"
    assert "marker" not in manifest


# --- the process path also persists it ---------------------------------------


def test_process_writes_a_segmentation_manifest(tmp_path):
    """``process`` is the primary user path and re-segments, so it REGENERATES.

    Pinned separately from ``split``: both call sites pass ``result.manifest`` to
    ``_write_siblings``, and dropping the kwarg from the process one alone left
    the whole suite green -- every ``voxweave <media>`` output would silently
    lose the key while CI stayed happy.
    """
    media = tmp_path / "ep.mkv"  # word_segments bypasses transcription
    pipeline.process(media, word_segments=("en", copy.deepcopy(EN_UNITS)))
    data = _read(tmp_path / "ep.json")

    assert list(data)[-1] == "segmentation"
    manifest = data["segmentation"]
    assert manifest["manifest_version"] == 1
    assert manifest["engine"] == "legacy-v1"
    assert manifest["language"] == "en"
    assert manifest["python"] == platform.python_version()
    assert set(manifest["profile"]) == PROFILE_KEYS
    assert manifest["env"] == {"gap_adaptive": False, "vad_emission_mask": False}
    assert manifest["providers"] == providers.provider_snapshot("en")
    assert manifest["degraded"] == []
    _assert_json_scalars(manifest)


def test_process_regenerates_the_manifest_of_an_edited_sibling(tmp_path):
    """A stale manifest left by a previous run does not survive a re-process."""
    media = tmp_path / "ep.mkv"
    json_path = tmp_path / "ep.json"
    json_path.write_text(
        json.dumps(
            {"language": "de", "segmentation": {"engine": "some-future-engine"}}
        ),
        encoding="utf-8",
    )
    pipeline.process(media, word_segments=("en", copy.deepcopy(EN_UNITS)))
    assert _read(json_path)["segmentation"]["engine"] == "legacy-v1"


# --- a REAL degradation reaches the persisted manifest ------------------------


# yue is declared no-space but ships neither jieba word segmentation nor a BudouX
# model, so ``phrase_atoms`` lands on per-char every time -- a deterministic,
# provider-less degradation that needs no monkeypatching to reproduce.
YUE_TEXT = "你哋今日食咗飯未呀我哋一齊去街啦"
YUE_UNITS = [
    {"text": ch, "start": round(0.3 * i, 3), "end": round(0.3 * i + 0.25, 3)}
    for i, ch in enumerate(YUE_TEXT)
]


def _degraded_index(entries):
    return {(e["slot"], e["reason"]): e["count"] for e in entries}


def test_split_persists_a_real_degradation_in_the_manifest(tmp_path):
    """The ledger -> manifest seam, end to end on a document that really degrades.

    Every other manifest test asserts ``degraded == []``, which both "hardcode
    ``[]``" and "drop the capture context" satisfy. This one fails under either.
    """
    path = _write_case(tmp_path, name="yue", language="yue", units=YUE_UNITS)
    pipeline.split(path)
    manifest = _read(path)["segmentation"]

    counts = _degraded_index(manifest["degraded"])
    assert ("atoms", "no-provider:per-char") in counts, manifest["degraded"]
    assert counts[("atoms", "no-provider:per-char")] >= 1
    # the snapshot agrees with the event: no atom provider ran at all
    assert manifest["providers"]["atoms"]["provider"] == "per_char"
    _assert_json_scalars(manifest["degraded"])


def test_segment_document_ledger_matches_the_manifest_it_persists(tmp_path):
    """The in-memory result and the file agree about what degraded."""
    result = pipeline.segment_document(
        language="yue", word_segments=copy.deepcopy(YUE_UNITS)
    )
    assert result.manifest is not None
    path = _write_case(tmp_path, name="yue2", language="yue", units=YUE_UNITS)
    pipeline.split(path)
    assert _degraded_index(_read(path)["segmentation"]["degraded"]) == _degraded_index(
        result.manifest["degraded"]
    )


def test_a_monkeypatched_missing_provider_is_recorded_in_the_written_manifest(
    tmp_path, monkeypatch
):
    """Second, independent route to a non-empty ``degraded``: a zh document on a
    host where jieba cannot be imported falls back to BudouX and says so."""
    from voxweave.core import breakpoints

    monkeypatch.setattr(breakpoints, "_load_jieba", lambda: None)
    zh_units = [
        {"text": ch, "start": round(0.3 * i, 3), "end": round(0.3 * i + 0.25, 3)}
        for i, ch in enumerate("今天的天气真好我们一起出去走走吧")
    ]
    path = _write_case(tmp_path, name="zh", language="zh", units=zh_units)
    pipeline.split(path)
    counts = _degraded_index(_read(path)["segmentation"]["degraded"])
    assert counts.get(("atoms", "jieba-missing:budoux-fallback"), 0) >= 1


# --- the read-side label exists at the chokepoints ----------------------------


def test_split_labels_the_document_it_replays(tmp_path, monkeypatch):
    """PIN 3: nothing consumes the label in P3, but it must be produced at the
    chokepoint so P4's adapter has a place to hook."""
    seen = []

    def spy(data):
        seen.append(dict(data))
        return {"engine": "legacy-v1", "inferred": True}

    monkeypatch.setattr(pipeline, "resolve_segmentation_manifest", spy)
    path = _write_case(tmp_path)
    pipeline.split(path)
    assert len(seen) == 1
    assert seen[0]["language"] == "en"


def test_align_labels_the_document_it_retimes(tmp_path, monkeypatch):
    seen = []

    def spy(data):
        seen.append(dict(data))
        return {"engine": "legacy-v1", "inferred": True}

    monkeypatch.setattr(pipeline, "resolve_segmentation_manifest", spy)
    media, vtt, _json_path = _align_setup(tmp_path)
    _run_align(media, vtt)
    assert len(seen) == 1
    assert seen[0]["language"] == "zh"


def test_align_labels_even_without_a_sibling_json(tmp_path, monkeypatch):
    """align's ``{}`` no-file branch is labelled too -- an unlabelled document is
    still a legacy-v1 one."""
    seen = []

    def spy(data):
        seen.append(dict(data))
        return {"engine": "legacy-v1", "inferred": True}

    monkeypatch.setattr(pipeline, "resolve_segmentation_manifest", spy)
    media, vtt, json_path = _align_setup(tmp_path)
    json_path.unlink()
    # Without the sibling there is no word_segments to route from, so the VTT has
    # to carry its own timestamps; ``lang_override`` keeps the per-cue (patched)
    # aligner route the sibling's ``language`` would otherwise have selected.
    vtt.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n你好\n\n"
        "00:00:02.000 --> 00:00:03.000\n世界\n",
        encoding="utf-8",
    )
    _run_align(media, vtt, lang_override="zh")
    assert seen == [{}]


# --- the writers -------------------------------------------------------------


def test_dump_sibling_json_omits_segmentation_without_a_manifest(tmp_path):
    path = tmp_path / "x.json"
    pipeline._dump_sibling_json(
        path,
        language="en",
        segments=[],
        units=[],
        vad_speech=None,
    )
    assert "segmentation" not in _read(path)


def test_dump_sibling_json_appends_the_manifest_last(tmp_path):
    path = tmp_path / "x.json"
    manifest = {"engine": "legacy-v1"}
    pipeline._dump_sibling_json(
        path,
        language="en",
        segments=[],
        units=[],
        vad_speech=[(0.0, 1.0)],
        speaker_turns=[(0.0, 1.0, "SPEAKER_00")],
        manifest=manifest,
    )
    data = _read(path)
    assert list(data)[-1] == "segmentation"
    assert data["segmentation"] == manifest


def test_dump_sibling_json_does_not_mutate_the_caller_manifest(tmp_path):
    path = tmp_path / "x.json"
    manifest = {"engine": "legacy-v1", "degraded": []}
    before = copy.deepcopy(manifest)
    pipeline._dump_sibling_json(
        path, language="en", segments=[], units=[], vad_speech=None, manifest=manifest
    )
    assert manifest == before


def test_write_siblings_forwards_the_manifest(tmp_path):
    src = tmp_path / "clip.mkv"
    cues = [{"text": "hi", "start": 0.0, "end": 1.0, "word_data": []}]
    units = [{"text": "hi", "start": 0.0, "end": 1.0}]
    pipeline._write_siblings(
        src, cues, units, "en", manifest={"engine": "legacy-v1", "manifest_version": 1}
    )
    data = _read(tmp_path / "clip.json")
    assert data["segmentation"] == {"engine": "legacy-v1", "manifest_version": 1}
    assert list(data)[-1] == "segmentation"


# --- align: preserve, never regenerate, never invent -------------------------


def _align_setup(tmp_path, **extra):
    stem = tmp_path / "ep"
    media = stem.with_suffix(".wav")
    media.write_bytes(b"x")
    doc = {
        "language": "zh",
        "word_segments": [
            {"text": "你", "start": 0.0, "end": 0.5},
            {"text": "好", "start": 0.5, "end": 1.0},
            {"text": "世", "start": 2.0, "end": 2.5},
            {"text": "界", "start": 2.5, "end": 3.0},
        ],
    }
    doc.update(extra)
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    vtt = stem.with_suffix(".vtt")
    vtt.write_text("WEBVTT\n\n你好\n\n世界\n", encoding="utf-8")
    return media, vtt, json_path


def _fake_align(cwav, text, lang):
    return [
        {"text": c, "start": float(i), "end": float(i) + 0.5}
        for i, c in enumerate(text)
    ]


def _run_align(media, vtt, **kwargs):
    with (
        patch("voxweave.pipeline._prepare_16k_for_align", return_value=media),
        patch("voxweave.pipeline.slice_wav", return_value=media),
        patch("voxweave.backend.align_text", side_effect=_fake_align),
        patch("voxweave.pipeline.vad_speech_segments", return_value=[]),
    ):
        return pipeline.align(vtt, **kwargs)


def test_align_preserves_an_existing_manifest_verbatim(tmp_path):
    manifest = {
        "manifest_version": 1,
        "engine": "legacy-v1",
        "language": "zh",
        "profile": {"max_line_length": 18, "max_lines": 1},
        "providers": {"atoms": {"provider": "jieba", "version": "0.42.1"}},
        "degraded": [{"slot": "pos", "reason": "pos-offset-disagreement", "count": 2}],
    }
    media, vtt, json_path = _align_setup(tmp_path, segmentation=manifest)
    _run_align(media, vtt)
    data = _read(json_path)
    assert data["segmentation"] == manifest
    assert list(data)[-1] == "segmentation"


def test_align_does_not_invent_a_manifest_when_absent(tmp_path):
    media, vtt, json_path = _align_setup(tmp_path)
    _run_align(media, vtt)
    assert "segmentation" not in _read(json_path)


# --- legacy inference on the read side ---------------------------------------


def test_resolve_segmentation_manifest_infers_legacy_when_absent():
    assert pipeline.resolve_segmentation_manifest({"language": "en"}) == {
        "engine": "legacy-v1",
        "inferred": True,
    }


def test_resolve_segmentation_manifest_infers_legacy_for_an_empty_document():
    assert pipeline.resolve_segmentation_manifest({}) == {
        "engine": "legacy-v1",
        "inferred": True,
    }


def test_resolve_segmentation_manifest_returns_a_present_manifest():
    manifest = {"manifest_version": 1, "engine": "legacy-v1"}
    assert (
        pipeline.resolve_segmentation_manifest({"segmentation": manifest}) is manifest
    )


def test_resolve_segmentation_manifest_ignores_a_non_mapping_value():
    for bad in ([], "legacy-v1", 1, None):
        assert pipeline.resolve_segmentation_manifest({"segmentation": bad}) == {
            "engine": "legacy-v1",
            "inferred": True,
        }


# --- SegmentationResult carries the IR ---------------------------------------


def test_segmentation_result_carries_the_manifest_and_document():
    result = pipeline.segment_document(
        language="en", word_segments=copy.deepcopy(EN_UNITS)
    )
    assert result.manifest is not None
    assert result.manifest["engine"] == "legacy-v1"
    assert result.document is not None
    # the document records the same manifest object, not a copy
    assert result.document.manifest is result.manifest
    assert result.document.language == "en"
    assert [u.id for u in result.document.units] == [
        f"u{i}" for i in range(len(result.units))
    ]
    for key in THRESHOLD_KEYS:
        assert getattr(result.document.profile, key) == float(
            result.thresholds_used[key]
        ), key
    assert result.document.profile.max_line_length == 42
    assert result.document.profile.max_lines == 2


def test_segmentation_result_document_carries_the_engine_text():
    """``document.text`` is the exact join ``_units_to_seg`` handed the engine.

    Recorded rather than re-derived: a consumer that re-joined the surfaces would
    have to re-implement the no-space-language rule, and could then disagree with
    the stream that actually ran.
    """
    result = pipeline.segment_document(
        language="en", word_segments=copy.deepcopy(EN_UNITS)
    )
    assert result.document is not None
    assert result.document.text == "Where did you go Nowhere special"

    ja = pipeline.segment_document(
        language="ja",
        word_segments=[
            {"text": "こんにちは", "start": 0.0, "end": 0.6},
            {"text": "世界", "start": 0.7, "end": 1.4},
        ],
    )
    assert ja.document is not None
    assert ja.document.text == "こんにちは世界"  # no separator for a no-space language


def test_the_document_is_built_before_the_engine_runs():
    """AD-6: the IR is an *input*, so it exists before ``smart_split_segments``.

    Ordering is observed through the two seams themselves rather than by probing
    a private attribute: a wrapper on each records the call order, and the engine
    wrapper still delegates to the real one so the run is a normal one.
    """
    from voxweave.core import smart_split as smart_split_module

    order = []
    real_build = pipeline.build_seg_document
    real_split = smart_split_module.smart_split_segments

    def build(**kwargs):
        order.append("document")
        return real_build(**kwargs)

    def split(*args, **kwargs):
        order.append("engine")
        return real_split(*args, **kwargs)

    with patch.object(pipeline, "build_seg_document", build):
        with patch.object(smart_split_module, "smart_split_segments", split):
            result = pipeline.segment_document(
                language="en", word_segments=copy.deepcopy(EN_UNITS)
            )

    assert order == ["document", "engine"]
    assert result.document is not None


def test_the_manifest_is_complete_before_the_engine_and_only_degraded_is_filled_after():
    """Every manifest field except ``degraded`` is final before the engine runs.

    yue degrades for real (per-char atoms), so this also pins the direction of
    the fill: empty at build time, populated afterwards, in the same dict object
    the document holds.
    """
    seen = {}
    real_build = pipeline.build_seg_document

    def build(**kwargs):
        manifest = kwargs["manifest"]
        seen["keys"] = list(manifest)
        seen["snapshot"] = copy.deepcopy(manifest)
        return real_build(**kwargs)

    with patch.object(pipeline, "build_seg_document", build):
        result = pipeline.segment_document(
            language="yue", word_segments=copy.deepcopy(YUE_UNITS)
        )

    assert result.manifest is not None
    # the key set (and its order) is settled before the engine, degraded included
    assert seen["keys"] == list(result.manifest)
    # ... and every value except degraded is already the final one
    for key, value in seen["snapshot"].items():
        if key == "degraded":
            continue
        assert result.manifest[key] == value, key
    assert seen["snapshot"]["degraded"] == []
    assert (
        _degraded_index(result.manifest["degraded"])[("atoms", "no-provider:per-char")]
        >= 1
    )
    assert result.document is not None
    assert result.document.manifest is result.manifest


def test_the_pre_engine_phase_raises_no_degradation_event():
    """Why narrowing the capture to the engine run loses nothing.

    The work that now sits ahead of the capture -- the joined text, the threshold
    resolution and the provider *snapshot* -- reaches no fallback: only the three
    scorers inside the engine call ``note_degraded``. ``provider_snapshot`` is
    the interesting one, because it does load the very providers whose absence is
    a degradation, and yue is the language that has none.
    """
    for iso in ("en", "ja", "zh", "yue"):
        with providers.degradation_capture() as ledger:
            providers.provider_snapshot(iso)
        assert ledger == [], iso

    # ... while the engine run on the same language does record the fallback.
    result = pipeline.segment_document(
        language="yue", word_segments=copy.deepcopy(YUE_UNITS)
    )
    assert result.manifest is not None
    assert result.manifest["degraded"]


def test_segmentation_result_document_carries_the_replay_evidence():
    result = pipeline.segment_document(
        language="en",
        word_segments=copy.deepcopy(EN_UNITS),
        vad_speech=[(0.0, 2.0)],
        shot_changes=[2.3],
        sing_spans=[(0.0, 0.5)],
        speaker_turns=[(0.0, 2.2, "SPEAKER_00"), (2.3, 3.8, "SPEAKER_01")],
    )
    doc = result.document
    assert doc is not None
    assert doc.vad_speech == [(0.0, 2.0)]
    assert doc.shot_changes == [2.3]
    assert doc.sing_spans == [(0.0, 0.5)]
    assert doc.speaker_turns == [
        (0.0, 2.2, "SPEAKER_00"),
        (2.3, 3.8, "SPEAKER_01"),
    ]
