"""Contract tests for ``scripts/calib_alignment.py``.

The alignment ruler is only worth what its matcher is worth, so these pin the
properties that decide whether a reported number means anything:

* the matcher pairs on text and **never** on time -- shuffling every hypothesis
  timestamp must leave the pairing byte-identical;
* 1:N and N:1 merges/splits are matched as groups, not silently skipped;
* a Japanese lane refuses an English release track (exit 2, not a soft warning);
* an empty lane reports ``n = 0`` with ``null`` aggregates, never a fake ``0.0``;
* ``report`` -> ``record-baseline`` -> ``check`` round-trips, the gates are
  one-way, and a changed corpus is *invalid* rather than a regression.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "calibration" / "alignment" / "fixtures"
EXAMPLE_MANIFEST = REPO_ROOT / "calibration" / "alignment" / "manifest.example.json"


def _load_script(name: str) -> Any:
    """Import a module from ``scripts/`` by path (it is not an installed package)."""
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ca = _load_script("calib_alignment")
cc = _load_script("calib_common")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def read_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def cue_segments(doc: dict[str, Any], language: str) -> list[Any]:
    return ca.make_segments(doc["cues"], language=language, prefix="hyp")


def reference_of(name: str) -> tuple[list[Any], int]:
    doc = read_fixture(name)
    return ca.reference_segments(doc, language=doc["language"])


def pairing_shape(result: Any) -> list[tuple[int, int, int, int]]:
    """Pairing identity, deliberately excluding similarity and every timestamp."""
    return [(g.hyp_start, g.hyp_count, g.ref_start, g.ref_count) for g in result.groups]


def write_manifest(
    tmp_path: Path, items: list[dict[str, Any]], **defaults: Any
) -> Path:
    doc: dict[str, Any] = {"schema_version": 1, "items": items}
    if defaults:
        doc["defaults"] = defaults
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def fixture_item(item_id: str, language: str, hyp: str, ref: str, kind: str) -> dict:
    return {
        "id": item_id,
        "language": language,
        "hypothesis": {"path": str(FIXTURES / hyp)},
        "references": [
            {
                "id": f"{item_id}-ref",
                "kind": kind,
                "language": language,
                "path": str(FIXTURES / ref),
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def test_normalization_strips_markup_but_keeps_words() -> None:
    raw = "{\\i1}<i>MARIA:</i> - Well, it's <b>over</b>! [door slams] ♪{\\i0}"
    assert ca.normalize_text(raw, "en") == "well it's over"


def test_normalization_drops_cjk_punctuation_and_spaces() -> None:
    assert (
        ca.normalize_text("灯台守は、嵐の前に 眠らない。", "ja")
        == "灯台守は嵐の前に眠らない"
    )


def test_normalization_is_pure() -> None:
    text = "The keeper's lamp — it failed."
    assert ca.normalize_text(text, "en") == ca.normalize_text(text, "en")
    assert text == "The keeper's lamp — it failed."


def test_empty_normalization_is_counted_not_dropped() -> None:
    segments = ca.make_segments(
        [
            {"text": "♪", "start": 0.0, "end": 1.0},
            {"text": "[thunder]", "start": 1.0, "end": 2.0},
            {"text": "Real words here", "start": 2.0, "end": 3.0},
        ],
        language="en",
        prefix="h",
    )
    keep, empty = ca.prepare_segments(segments)
    assert [s.norm for s in keep] == ["real words here"]
    assert empty == 2


# --------------------------------------------------------------------------- #
# Language detection and same-language enforcement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The lighthouse keeper never sleeps before the storm arrives", "en"),
        ("灯台守は嵐の前に眠らない 彼は波を数えて記録する", "ja"),
        ("灯塔守夜人从不在暴风雨来临之前入睡他数着海浪", "zh"),
        ("ok", None),
    ],
)
def test_detect_text_language(text: str, expected: str | None) -> None:
    assert ca.detect_text_language(text) == expected


def test_japanese_lane_rejects_an_english_reference(tmp_path: Path) -> None:
    """A ja lane fed an English release track is invalid, never "close enough"."""
    manifest = write_manifest(
        tmp_path,
        [
            fixture_item(
                "ja-english-track",
                "ja",
                "synthetic-ja.hypothesis.json",
                "synthetic-ja-english-track.reference.json",
                "commercial_cues",
            )
        ],
    )
    report = ca.evaluate(ca.load_manifest(manifest))
    assert report["status"] == "invalid"
    codes = [f["code"] for f in report["failures"]]
    assert codes == ["reference_language_mismatch"]
    lane = report["lanes"][0]
    assert lane["status"] == "invalid"
    # A rejected reference contributes no samples at all: the lane must not be
    # able to publish a number derived from a track it refused.
    assert all(block["n"] == 0 for block in lane["metrics"].values())


def test_language_mismatch_exits_2(tmp_path: Path, capsys: Any) -> None:
    manifest = write_manifest(
        tmp_path,
        [
            fixture_item(
                "ja-english-track",
                "ja",
                "synthetic-ja.hypothesis.json",
                "synthetic-ja-english-track.reference.json",
                "commercial_cues",
            )
        ],
    )
    with pytest.raises(SystemExit) as exc:
        ca.main(
            [
                "report",
                "--manifest",
                str(manifest),
                "--json-out",
                str(tmp_path / "report.json"),
            ]
        )
    assert exc.value.code == cc.EXIT_INVALID
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["status"] == "invalid"


def test_cross_language_manifest_is_rejected_by_the_loader(tmp_path: Path) -> None:
    item = fixture_item(
        "mixed",
        "ja",
        "synthetic-ja.hypothesis.json",
        "synthetic-1to2.reference.json",
        "manual_cues",
    )
    item["references"][0]["language"] = "en"
    manifest = write_manifest(tmp_path, [item])
    with pytest.raises(cc.CalibrationError, match="cross-language"):
        ca.load_manifest(manifest)


# --------------------------------------------------------------------------- #
# Matcher: shapes and time independence
# --------------------------------------------------------------------------- #


def test_one_reference_cue_matches_two_hypothesis_cues() -> None:
    hyp = cue_segments(read_fixture("synthetic-1to2.hypothesis.json"), "en")
    ref, excluded = reference_of("synthetic-1to2.reference.json")
    assert excluded == 0
    result = ca.pair_monotonic(
        hyp, ref, language="en", level="cue", min_pair_similarity=0.6
    )
    shapes = [g.shape for g in result.groups]
    assert shapes.count("N:1") == 1
    merged = next(g for g in result.groups if g.shape == "N:1")
    assert (merged.hyp_count, merged.ref_count) == (2, 1)
    assert merged.ref_start == 1
    coverage = ca.coverage_of(result)
    assert coverage["hyp_chars"]["value"] == 1.0
    assert coverage["ref_chars"]["value"] == 1.0
    assert coverage["hyp_unmatched_segments"] == 0
    assert coverage["ref_unmatched_segments"] == 0


def test_two_reference_cues_match_one_hypothesis_cue() -> None:
    hyp = cue_segments(read_fixture("synthetic-2to1.hypothesis.json"), "en")
    ref, _ = reference_of("synthetic-2to1.reference.json")
    result = ca.pair_monotonic(
        hyp, ref, language="en", level="cue", min_pair_similarity=0.6
    )
    split = [g for g in result.groups if g.shape == "1:N"]
    assert len(split) == 1
    assert (split[0].hyp_count, split[0].ref_count) == (1, 2)
    assert ca.coverage_of(result)["ref_chars"]["value"] == 1.0


def test_matcher_ignores_timestamps() -> None:
    """Shuffle every hypothesis timestamp; the pairing must not move.

    This is the property the whole ruler rests on. If time leaked into the
    matcher, a hypothesis with destroyed timing would pair with itself and score
    well -- the metric would be measuring its own input.
    """
    doc = read_fixture("synthetic-1to2.hypothesis.json")
    ref, _ = reference_of("synthetic-1to2.reference.json")
    ordered = cue_segments(doc, "en")
    baseline = ca.pair_monotonic(
        ordered, ref, language="en", level="cue", min_pair_similarity=0.6
    )

    rng = random.Random(20260826)
    scrambled_doc = copy.deepcopy(doc)
    times = [(c["start"], c["end"]) for c in scrambled_doc["cues"]]
    rng.shuffle(times)
    for cue, (start, end) in zip(scrambled_doc["cues"], times, strict=True):
        cue["start"], cue["end"] = start + 137.0, end + 137.0
    scrambled = cue_segments(scrambled_doc, "en")

    assert [s.start for s in scrambled] != [s.start for s in ordered]
    assert pairing_shape(
        ca.pair_monotonic(
            scrambled, ref, language="en", level="cue", min_pair_similarity=0.6
        )
    ) == pairing_shape(baseline)


def test_matcher_ignores_timestamps_for_word_lanes() -> None:
    doc = read_fixture("synthetic-ja-words.hypothesis.json")
    ref, _ = reference_of("synthetic-ja-words.reference.json")
    ordered = ca.make_segments(doc["word_segments"], language="ja", prefix="hyp")
    baseline = ca.pair_monotonic(
        ordered, ref, language="ja", level="word", min_pair_similarity=0.6
    )
    reversed_doc = copy.deepcopy(doc)
    for unit in reversed_doc["word_segments"]:
        unit["start"], unit["end"] = 99.0 - unit["end"], 99.0 - unit["start"]
    reversed_units = ca.make_segments(
        reversed_doc["word_segments"], language="ja", prefix="hyp"
    )
    assert pairing_shape(
        ca.pair_monotonic(
            reversed_units, ref, language="ja", level="word", min_pair_similarity=0.6
        )
    ) == pairing_shape(baseline)


def test_japanese_characters_group_onto_one_reference_word() -> None:
    doc = read_fixture("synthetic-ja-words.hypothesis.json")
    ref, _ = reference_of("synthetic-ja-words.reference.json")
    hyp = ca.make_segments(doc["word_segments"], language="ja", prefix="hyp")
    result = ca.pair_monotonic(
        hyp, ref, language="ja", level="word", min_pair_similarity=0.6
    )
    assert len(result.groups) == len(ref)
    assert all(g.ref_count == 1 for g in result.groups)
    assert any(g.hyp_count == 2 for g in result.groups)
    assert ca.coverage_of(result)["hyp_chars"]["value"] == 1.0


def test_repeated_lines_do_not_make_the_matcher_jump() -> None:
    """A line repeated verbatim must stay in order rather than pair across the file."""
    lines = [
        "who is out there tonight",
        "nobody answered the call",
        "who is out there tonight",
        "the fog swallowed the harbour lights",
        "who is out there tonight",
        "then the bell rang once and stopped",
    ]
    ref = ca.make_segments(
        [
            {"text": t, "start": 10.0 * i, "end": 10.0 * i + 3.0}
            for i, t in enumerate(lines)
        ],
        language="en",
        prefix="ref",
    )
    hyp = ca.make_segments(
        [
            {"text": t, "start": 10.0 * i + 0.1, "end": 10.0 * i + 3.1}
            for i, t in enumerate(lines)
        ],
        language="en",
        prefix="hyp",
    )
    result = ca.pair_monotonic(
        hyp, ref, language="en", level="cue", min_pair_similarity=0.6
    )
    assert pairing_shape(result) == [(i, 1, i, 1) for i in range(len(lines))]


def test_missing_and_extra_cues_are_skipped_not_forced() -> None:
    ref_lines = [
        "the harbour master keeps a spare key under the third plank",
        "he swears the gulls have learned to lift it",
        "one summer a boy borrowed the boat before dawn",
        "the master rowed after him without saying a word",
        "they came back with the sail down and both of them laughing",
    ]
    ref = ca.make_segments(
        [
            {"text": t, "start": 10.0 * i, "end": 10.0 * i + 4.0}
            for i, t in enumerate(ref_lines)
        ],
        language="en",
        prefix="ref",
    )
    hyp_lines = [
        ref_lines[0],
        ref_lines[1],
        "completely unrelated interjection about nothing at all",
        ref_lines[3],
        ref_lines[4],
    ]
    hyp = ca.make_segments(
        [
            {"text": t, "start": 10.0 * i + 0.2, "end": 10.0 * i + 4.2}
            for i, t in enumerate(hyp_lines)
        ],
        language="en",
        prefix="hyp",
    )
    result = ca.pair_monotonic(
        hyp, ref, language="en", level="cue", min_pair_similarity=0.6
    )
    coverage = ca.coverage_of(result)
    assert coverage["hyp_unmatched_segments"] == 1
    assert coverage["ref_unmatched_segments"] == 1
    assert coverage["hyp_chars"]["value"] is not None
    assert coverage["hyp_chars"]["value"] < 1.0
    assert (2, 1, 2, 1) not in pairing_shape(result)


def test_matcher_is_deterministic_across_repeated_runs() -> None:
    hyp = cue_segments(read_fixture("synthetic-1to2.hypothesis.json"), "en")
    ref, _ = reference_of("synthetic-1to2.reference.json")
    runs = [
        pairing_shape(
            ca.pair_monotonic(
                hyp, ref, language="en", level="cue", min_pair_similarity=0.6
            )
        )
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_empty_lane_reports_null_not_zero(tmp_path: Path) -> None:
    """Every reference segment excluded -> n = 0 and null aggregates."""
    manifest = write_manifest(
        tmp_path,
        [
            fixture_item(
                "empty-lane",
                "en",
                "synthetic-1to2.hypothesis.json",
                "synthetic-empty.reference.json",
                "manual_cues",
            )
        ],
    )
    report = ca.evaluate(ca.load_manifest(manifest))
    assert report["status"] == "pass"
    lane = report["lanes"][0]
    assert lane["status"] == "insufficient_samples"
    block = lane["metrics"]["start_abs_s"]
    assert block["n"] == 0
    assert block["median"] is None and block["p90"] is None and block["mae"] is None
    assert block["pct_le_0_25"] is None
    assert lane["coverage"]["excluded_reference_segments"] == 4
    assert lane["coverage"]["ref_chars"]["value"] is None


def test_lanes_are_never_pooled(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        [
            fixture_item(
                "en-cues",
                "en",
                "synthetic-1to2.hypothesis.json",
                "synthetic-1to2.reference.json",
                "manual_cues",
            ),
            fixture_item(
                "ja-words",
                "ja",
                "synthetic-ja-words.hypothesis.json",
                "synthetic-ja-words.reference.json",
                "mfa_words",
            ),
        ],
    )
    report = ca.evaluate(ca.load_manifest(manifest))
    keys = [(lane["source_kind"], lane["language"]) for lane in report["lanes"]]
    assert keys == [("manual_cues", "en"), ("mfa_words", "ja")]
    cue_lane = report["lanes"][0]
    word_lane = report["lanes"][1]
    assert "start_abs_s" in cue_lane["metrics"]
    assert "word_start_abs_s" not in cue_lane["metrics"]
    assert "word_start_abs_s" in word_lane["metrics"]
    assert "start_abs_s" not in word_lane["metrics"]
    # Cue lanes report the display thresholds, word lanes the acoustic ones.
    assert cue_lane["metrics"]["start_abs_s"]["pct_le_1_0"] is not None
    assert "pct_le_1_0" not in word_lane["metrics"]["word_start_abs_s"]
    assert word_lane["metrics"]["word_start_abs_s"]["pct_le_0_025"] is not None


def test_multi_reference_word_groups_are_not_word_mae() -> None:
    """One hypothesis unit covering two reference words is a lexical span."""
    ref = ca.make_segments(
        [
            {"text": "do", "start": 1.0, "end": 1.2},
            {"text": "not", "start": 1.2, "end": 1.5},
            {"text": "worry", "start": 1.5, "end": 2.0},
            {"text": "about", "start": 2.0, "end": 2.4},
            {"text": "the", "start": 2.4, "end": 2.55},
            {"text": "weather", "start": 2.55, "end": 3.1},
        ],
        language="en",
        prefix="ref",
    )
    hyp = ca.make_segments(
        [
            {"text": "don't", "start": 1.02, "end": 1.52},
            {"text": "worry", "start": 1.52, "end": 2.02},
            {"text": "about", "start": 2.02, "end": 2.42},
            {"text": "the", "start": 2.42, "end": 2.57},
            {"text": "weather", "start": 2.57, "end": 3.12},
        ],
        language="en",
        prefix="hyp",
    )
    result = ca.pair_monotonic(
        hyp, ref, language="en", level="word", min_pair_similarity=0.6
    )
    contraction = [g for g in result.groups if g.ref_count == 2]
    assert len(contraction) == 1
    metrics, primary = ca.lane_metrics(
        "word",
        ca.boundary_errors(result, cluster_prefix="t"),
        uncertainty=None,
        bootstrap_samples=0,
    )
    assert primary == 4
    assert metrics["word_start_abs_s"]["n"] == 4
    assert metrics["lexical_span_start_abs_s"]["n"] == 1


def test_reference_uncertainty_is_reported_not_subtracted() -> None:
    errors = [
        ca.BoundaryError(("h",), ("r",), 1, 1.0, 0.05, 0.05, 0.05, 0.05, "c1"),
        ca.BoundaryError(("h",), ("r",), 1, 1.0, 0.09, 0.09, 0.09, 0.09, "c2"),
    ]
    metrics, _ = ca.lane_metrics("word", errors, uncertainty=0.02, bootstrap_samples=0)
    block = metrics["word_start_abs_s"]
    assert block["median"] == pytest.approx(0.07)
    assert block["interpretive_lower_bound"] == pytest.approx(0.05)
    assert block["mae"] == pytest.approx(0.07)


def test_coverage_keeps_numerator_and_denominator(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        [
            fixture_item(
                "en-cues",
                "en",
                "synthetic-1to2.hypothesis.json",
                "synthetic-1to2.reference.json",
                "manual_cues",
            )
        ],
    )
    lane = ca.evaluate(ca.load_manifest(manifest))["lanes"][0]
    for side in ("hyp_chars", "ref_chars", "hyp_segments", "ref_segments"):
        block = lane["coverage"][side]
        assert set(block) == {"matched", "total", "value"}
        assert block["total"] > 0
        assert block["value"] == pytest.approx(block["matched"] / block["total"])


def test_insufficient_coverage_is_invalid_not_a_regression(tmp_path: Path) -> None:
    hyp = tmp_path / "hyp.json"
    hyp.write_text(
        json.dumps(
            {
                "language": "en",
                "cues": [
                    {
                        "text": "The lighthouse keeper never sleeps",
                        "start": 1.0,
                        "end": 4.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    item = fixture_item(
        "thin",
        "en",
        "synthetic-1to2.hypothesis.json",
        "synthetic-1to2.reference.json",
        "manual_cues",
    )
    item["hypothesis"]["path"] = str(hyp)
    manifest = write_manifest(tmp_path, [item])
    report = ca.evaluate(ca.load_manifest(manifest))
    assert report["status"] == "invalid"
    assert report["failures"][0]["code"] == "insufficient_coverage"
    assert "coverage" in report["failures"][0]["message"]


# --------------------------------------------------------------------------- #
# Track discovery
# --------------------------------------------------------------------------- #

FFPROBE_STREAMS = {
    "streams": [
        {
            "index": 2,
            "codec_name": "hdmv_pgs_subtitle",
            "tags": {"language": "eng", "title": "Full"},
            "disposition": {"default": 1, "forced": 0},
        },
        {
            "index": 3,
            "codec_name": "subrip",
            "tags": {"language": "jpn", "title": "Signs & Songs"},
            "disposition": {"default": 0, "forced": 0},
        },
        {
            "index": 4,
            "codec_name": "subrip",
            "tags": {"language": "jpn", "title": "Japanese"},
            "disposition": {"default": 0, "forced": 0},
        },
        {
            "index": 5,
            "codec_name": "subrip",
            "tags": {"language": "jpn", "title": "English (mislabelled)"},
            "disposition": {"default": 0, "forced": 0},
        },
        {
            "index": 6,
            "codec_name": "subrip",
            "tags": {"title": "Untagged"},
            "disposition": {"default": 0, "forced": 1},
        },
    ]
}

JA_TRACK = [
    ("灯台守は嵐の前に眠らない", 1.0, 4.0),
    ("彼は波を数えて革の手帳に記録する", 4.6, 8.4),
    ("村の誰もその手帳を読まない", 9.0, 11.9),
    ("去年の冬に灯りが初めて消えた", 12.4, 15.6),
    ("守は港まで降りて提灯を借りた", 16.1, 19.4),
    ("それから彼は替えの芯を二本持ち歩く", 20.0, 23.2),
]
EN_TRACK = [
    ("The lighthouse keeper never sleeps before the storm", 1.0, 4.0),
    ("He counts the waves and writes them in a leather book", 4.6, 8.4),
    ("Nobody in the village ever reads that book", 9.0, 11.9),
    ("Last winter the lamp went out for the first time", 12.4, 15.6),
    ("The keeper went down to the harbour and borrowed a lantern", 16.1, 19.4),
    ("Since then he carries two spare wicks with him", 20.0, 23.2),
]


def _srt(rows: list[tuple[str, float, float]]) -> str:
    def stamp(t: float) -> str:
        ms = round(t * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    return "\n".join(
        f"{i}\n{stamp(s)} --> {stamp(e)}\n{text}\n"
        for i, (text, s, e) in enumerate(rows, 1)
    )


def fake_probe(_media: Path) -> list[Any]:
    doc = json.loads(json.dumps(FFPROBE_STREAMS))
    streams = []
    for entry in doc["streams"]:
        tags = entry.get("tags") or {}
        disposition = entry.get("disposition") or {}
        streams.append(
            ca.SubtitleStream(
                index=int(entry["index"]),
                codec=str(entry["codec_name"]),
                language=cc.canonical_language_or(tags.get("language"), None),
                raw_language=tags.get("language"),
                title=str(tags.get("title") or ""),
                default=bool(disposition.get("default")),
                forced=bool(disposition.get("forced")),
                hearing_impaired=False,
            )
        )
    return streams


def fake_extract(_media: Path, index: int, dest: Path, _codec: str) -> Path:
    rows = JA_TRACK if index == 4 else EN_TRACK
    dest.write_text(_srt(rows), encoding="utf-8")
    return dest


def test_probe_parses_ffprobe_json_without_touching_the_filesystem() -> None:
    captured: list[list[str]] = []

    def run(args):
        captured.append(list(args))
        return json.dumps(FFPROBE_STREAMS)

    streams = ca.probe_subtitle_streams(Path("episode.mkv"), run=run)
    assert [s.index for s in streams] == [2, 3, 4, 5, 6]
    assert [s.language for s in streams] == ["en", "ja", "ja", "ja", None]
    assert captured[0][0] == "ffprobe"
    assert "0:s:0" not in " ".join(captured[0])


def test_track_selection_never_assumes_the_first_stream(tmp_path: Path) -> None:
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"")
    hyp = ca.make_segments(
        [{"text": t, "start": s, "end": e} for t, s, e in JA_TRACK],
        language="ja",
        prefix="hyp",
    )
    selection = ca.select_subtitle_track(
        media,
        language="ja",
        hypothesis_norm="".join(s.norm for s in hyp),
        probe=fake_probe,
        extract=fake_extract,
    )
    assert selection.stream.index == 4
    assert selection.detected_language == "ja"
    rejections = {c.stream.index: c.rejected for c in selection.candidates}
    assert rejections[2] == "bitmap_subtitle_codec"
    assert rejections[3] == "partial_track_title"
    assert rejections[5] == "detected_language(en)"
    assert rejections[6] == "untagged_language"


def test_explicit_stream_index_is_still_language_checked(tmp_path: Path) -> None:
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"")
    with pytest.raises(ca.ReferenceLanguageMismatch, match="detects as 'en'"):
        ca.select_subtitle_track(
            media,
            language="ja",
            hypothesis_norm="灯台守は嵐の前に眠らない",
            explicit_index=5,
            probe=fake_probe,
            extract=fake_extract,
        )


def test_commercial_track_reference_runs_end_to_end(tmp_path: Path) -> None:
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"")
    manifest = write_manifest(
        tmp_path,
        [
            {
                "id": "ja-episode",
                "language": "ja",
                "hypothesis": {"path": str(FIXTURES / "synthetic-ja.hypothesis.json")},
                "references": [
                    {
                        "id": "ja-episode-release",
                        "kind": "commercial_cues",
                        "language": "ja",
                        "media": str(media),
                    }
                ],
            }
        ],
    )
    report = ca.evaluate(
        ca.load_manifest(manifest), probe=fake_probe, extract=fake_extract
    )
    assert report["status"] == "pass"
    lane = report["lanes"][0]
    assert lane["source_kind"] == "commercial_cues"
    assert lane["items"][0]["track"]["stream_index"] == 4
    assert lane["coverage"]["ref_chars"]["value"] == 1.0


def test_inspect_tracks_reports_a_verdict_per_stream(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"")
    monkeypatch.setattr(ca, "probe_subtitle_streams", fake_probe)
    monkeypatch.setattr(
        ca,
        "extract_subtitle_track",
        lambda m, i, d, codec="": fake_extract(m, i, d, codec),
    )
    assert (
        ca.main(
            [
                "inspect-tracks",
                str(media),
                "--lang",
                "ja",
                "--hypothesis",
                str(FIXTURES / "synthetic-ja.hypothesis.json"),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    verdicts = {row["index"]: row["rejected"] for row in payload["streams"]}
    assert verdicts[2] == "bitmap_subtitle_codec"
    assert verdicts[4] is None
    assert verdicts[5] == "detected_language(en)"
    chosen = next(row for row in payload["streams"] if row["index"] == 4)
    assert chosen["coverage"] == pytest.approx(1.0)


def test_ambiguous_tracks_require_an_explicit_index(tmp_path: Path) -> None:
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"")

    def probe(_media: Path) -> list[Any]:
        return [
            ca.SubtitleStream(
                4, "subrip", "ja", "jpn", "Japanese", False, False, False
            ),
            ca.SubtitleStream(
                7, "subrip", "ja", "jpn", "Japanese 2", False, False, False
            ),
        ]

    def extract(_media: Path, _index: int, dest: Path, _codec: str) -> Path:
        dest.write_text(_srt(JA_TRACK), encoding="utf-8")
        return dest

    hyp_norm = "".join(ca.normalize_text(t, "ja") for t, _, _ in JA_TRACK)
    with pytest.raises(cc.CalibrationError, match="ambiguous"):
        ca.select_subtitle_track(
            media,
            language="ja",
            hypothesis_norm=hyp_norm,
            probe=probe,
            extract=extract,
        )


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #


def test_example_manifest_loads() -> None:
    manifest = ca.load_manifest(EXAMPLE_MANIFEST)
    assert {item.id for item in manifest.items} >= {
        "synthetic-en-split",
        "synthetic-en-merge",
        "synthetic-ja-words",
    }
    assert manifest.digest == cc.canonical_digest(manifest.document)


def test_invalid_manifest_exits_2(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 2, "items": []}), encoding="utf-8")
    with pytest.raises(cc.CalibrationError, match="schema validation"):
        ca.load_manifest(path)


def test_calib_root_placeholder_must_stay_inside_the_root(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("VOXWEAVE_CALIB_ROOT", str(tmp_path / "root"))
    (tmp_path / "root").mkdir()
    inside = ca.resolve_manifest_path(
        "${VOXWEAVE_CALIB_ROOT}/truth/a.json",
        base=tmp_path,
        root_env="VOXWEAVE_CALIB_ROOT",
    )
    assert inside == (tmp_path / "root" / "truth" / "a.json").resolve()
    with pytest.raises(cc.CalibrationError, match="outside"):
        ca.resolve_manifest_path(
            "${VOXWEAVE_CALIB_ROOT}/../escape.json",
            base=tmp_path,
            root_env="VOXWEAVE_CALIB_ROOT",
        )


def test_ranges_select_the_window_and_must_be_well_formed(tmp_path: Path) -> None:
    item = fixture_item(
        "ranged",
        "en",
        "synthetic-1to2.hypothesis.json",
        "synthetic-1to2.reference.json",
        "manual_cues",
    )
    item["include_ranges"] = [[0.0, 12.0]]
    manifest = write_manifest(tmp_path, [item])
    lane = ca.evaluate(ca.load_manifest(manifest))["lanes"][0]
    assert lane["coverage"]["ref_segments"]["total"] == 3

    item["include_ranges"] = [[12.0, 12.0]]
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    with pytest.raises(cc.CalibrationError, match="end must exceed start"):
        ca.load_manifest(write_manifest(bad_dir, [item]))


def test_reference_document_invariants_are_checked() -> None:
    doc = read_fixture("synthetic-1to2.reference.json")
    doc["segments"][2]["end"] = doc["segments"][2]["start"]
    with pytest.raises(cc.CalibrationError, match="end .* must exceed start"):
        ca.reference_segments(doc, language="en")


def test_excluded_reference_segments_are_counted_not_dropped() -> None:
    doc = read_fixture("synthetic-1to2.reference.json")
    doc["segments"][3]["excluded"] = True
    doc["segments"][3]["exclude_reason"] = "spn"
    segments, excluded = ca.reference_segments(doc, language="en")
    assert excluded == 1
    assert len(segments) == len(doc["segments"]) - 1


# --------------------------------------------------------------------------- #
# CLI: report -> record-baseline -> check
# --------------------------------------------------------------------------- #


@pytest.fixture()
def gated_corpus(tmp_path: Path, monkeypatch: Any) -> Path:
    """A manifest whose lanes clear the sample floors, so the gates have teeth."""
    monkeypatch.setattr(ca, "MIN_CUE_GROUPS", 4)
    monkeypatch.setattr(ca, "MIN_WORD_SAMPLES", 4)
    return write_manifest(
        tmp_path,
        [
            fixture_item(
                "en-split",
                "en",
                "synthetic-1to2.hypothesis.json",
                "synthetic-1to2.reference.json",
                "manual_cues",
            ),
            fixture_item(
                "ja-words",
                "ja",
                "synthetic-ja-words.hypothesis.json",
                "synthetic-ja-words.reference.json",
                "mfa_words",
            ),
        ],
        bootstrap_samples=25,
    )


def test_report_check_round_trip(gated_corpus: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"

    assert (
        ca.main(
            ["report", "--manifest", str(gated_corpus), "--json-out", str(report_path)]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert [lane["status"] for lane in report["lanes"]] == ["pass", "pass"]

    assert (
        ca.main(
            [
                "record-baseline",
                "--manifest",
                str(gated_corpus),
                "--report",
                str(report_path),
                "--output",
                str(baseline_path),
            ]
        )
        == 0
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["manifest_digest"] == report["manifest_digest"]
    assert baseline["text_norm_version"] == ca.TEXT_NORM_VERSION
    assert all(
        ca.is_gated_metric(name)
        for lane in baseline["lanes"]
        for name in lane["metrics"]
    )

    assert (
        ca.main(
            ["check", "--manifest", str(gated_corpus), "--baseline", str(baseline_path)]
        )
        == 0
    )


def test_check_fails_on_a_regression(gated_corpus: Path, tmp_path: Path) -> None:
    report = ca.evaluate(ca.load_manifest(gated_corpus))
    baseline = ca.baseline_document(report)
    for lane in baseline["lanes"]:
        for block in lane["metrics"].values():
            for key in ("mae", "median", "p90"):
                if block.get(key) is not None:
                    block[key] = float(block[key]) / 10.0
    failures = ca.apply_gates(report, baseline)
    assert failures
    assert report["status"] == "fail"
    assert any(lane["status"] == "fail" for lane in report["lanes"])
    assert all(f["direction"] == "max" for f in failures)


def test_gates_are_one_way(gated_corpus: Path) -> None:
    """An improvement never fails: every error metric halved still passes."""
    report = ca.evaluate(ca.load_manifest(gated_corpus))
    baseline = ca.baseline_document(report)
    for lane in report["lanes"]:
        for block in lane["metrics"].values():
            for key in ("mae", "median", "p90"):
                if block.get(key) is not None:
                    block[key] = float(block[key]) / 2.0
            for key in ca.GATED_RATE_FIELDS:
                if block.get(key) is not None:
                    block[key] = 1.0
    assert ca.apply_gates(report, baseline) == []
    assert report["status"] == "pass"


def test_changed_corpus_is_invalid_not_a_regression(gated_corpus: Path) -> None:
    report = ca.evaluate(ca.load_manifest(gated_corpus))
    baseline = ca.baseline_document(report)
    baseline["manifest_digest"] = "0" * 64
    with pytest.raises(cc.CalibrationError, match="manifest digest"):
        ca.apply_gates(report, baseline)


def test_baseline_from_a_different_normalization_is_invalid(gated_corpus: Path) -> None:
    report = ca.evaluate(ca.load_manifest(gated_corpus))
    baseline = ca.baseline_document(report)
    baseline["text_norm_version"] = ca.TEXT_NORM_VERSION + 1
    with pytest.raises(cc.CalibrationError, match="text_norm_version"):
        ca.apply_gates(report, baseline)


def test_record_baseline_refuses_a_foreign_report(
    gated_corpus: Path, tmp_path: Path
) -> None:
    report = ca.evaluate(ca.load_manifest(gated_corpus))
    report["manifest_digest"] = "1" * 64
    report_path = tmp_path / "foreign.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ca.main(
            [
                "record-baseline",
                "--manifest",
                str(gated_corpus),
                "--report",
                str(report_path),
                "--output",
                str(tmp_path / "baseline.json"),
            ]
        )
    assert exc.value.code == cc.EXIT_INVALID


def test_check_exits_1_on_a_gate_failure(gated_corpus: Path, tmp_path: Path) -> None:
    report = ca.evaluate(ca.load_manifest(gated_corpus))
    baseline = ca.baseline_document(report)
    for lane in baseline["lanes"]:
        for block in lane["metrics"].values():
            for key in ("mae", "median", "p90"):
                if block.get(key) is not None:
                    block[key] = 0.0
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ca.main(
            ["check", "--manifest", str(gated_corpus), "--baseline", str(baseline_path)]
        )
    assert exc.value.code == cc.EXIT_GATE_FAILED


def test_report_validates_against_the_tracked_schema(gated_corpus: Path) -> None:
    report = ca.evaluate(ca.load_manifest(gated_corpus))
    assert cc.schema_errors(report, "alignment-report") == []


def test_per_group_pairs_are_kept_for_spot_checking(gated_corpus: Path) -> None:
    """The worst pairs are retained by default, and the selection names itself."""
    report = ca.evaluate(ca.load_manifest(gated_corpus), pairs="worst", pairs_limit=3)
    item = report["lanes"][0]["items"][0]
    assert item["pairs_kept"] == "worst_3"
    assert item["pairs_total"] > len(item["worst_pairs"]) == 3
    worst = item["worst_pairs"]
    assert set(worst[0]) == {
        "hyp_ids",
        "ref_ids",
        "match_shape",
        "similarity",
        "start_abs_s",
        "end_abs_s",
        "start_signed_s",
        "end_signed_s",
    }
    magnitudes = [max(p["start_abs_s"], p["end_abs_s"]) for p in worst]
    assert magnitudes == sorted(magnitudes, reverse=True)

    everything = ca.evaluate(ca.load_manifest(gated_corpus), pairs="all")
    detail = everything["lanes"][0]["items"][0]
    assert len(detail["pairs"]) == detail["pairs_total"]

    nothing = ca.evaluate(ca.load_manifest(gated_corpus), pairs="none")
    assert "worst_pairs" not in nothing["lanes"][0]["items"][0]


def test_source_and_item_filters_narrow_the_run(gated_corpus: Path) -> None:
    report = ca.evaluate(ca.load_manifest(gated_corpus), source_filter=("mfa_words",))
    assert [lane["source_kind"] for lane in report["lanes"]] == ["mfa_words"]
    report = ca.evaluate(ca.load_manifest(gated_corpus), item_filter=("en-split",))
    assert [lane["items"][0]["item_id"] for lane in report["lanes"]] == ["en-split"]
    empty = ca.evaluate(ca.load_manifest(gated_corpus), item_filter=("nothing",))
    assert empty["status"] == "invalid"
    assert empty["failures"][0]["code"] == "no_references_selected"
