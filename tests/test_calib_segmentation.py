"""Contract tests for ``scripts/calib_segmentation.py``.

A quality ruler that is itself unverified measures nothing, so these pin the
parts a number silently depends on:

* the four metric definitions, on hand-built cue streams whose expected
  numerator and denominator are written out in the test;
* micro aggregation -- summed counts, never averaged per-case percentages;
* one-sided gates -- a regression fails, an improvement never does;
* the 0/1/2 exit contract, including at the real process boundary;
* corpus validation: coverage, size, digest and baseline mismatch are all
  "invalid" (2), never "regressed" (1);
* replay determinism, because a gate on a non-deterministic replay is noise.

The synthetic corpus is generated here rather than read from
``calibration/segmentation/``: these tests must keep working before the real
corpus is captured, and they must not start passing or failing because someone
added an episode.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[1]


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


calib = _load_script("calib_segmentation")
capture = _load_script("capture_scenario")
cc = calib.cc


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def spaced_units(
    words: list[str], *, start: float = 0.0, dur: float = 0.6, gap: float = 0.0
) -> list[dict[str, Any]]:
    """One unit per word on a fixed grid -- the shape an English aligner emits."""
    out: list[dict[str, Any]] = []
    t = start
    for i, word in enumerate(words):
        out.append({"id": f"u{i}", "text": word, "start": t, "end": t + dur})
        t += dur + gap
    return out


def char_units(
    text: str, *, start: float = 0.0, dur: float = 0.3, gap: float = 0.0
) -> list[dict[str, Any]]:
    """One unit per character -- the shape a CJK aligner emits."""
    out: list[dict[str, Any]] = []
    t = start
    for i, ch in enumerate(text):
        out.append({"id": f"u{i}", "text": ch, "start": t, "end": t + dur})
        t += dur + gap
    return out


def cue(text: str, start: float, end: float, units: list[dict[str, Any]]) -> dict:
    """A cue carrying the exact ``word_data`` spans of its source units.

    Timing passes concatenate and slice ``word_data`` but never rewrite a span,
    which is what lets the harness map a boundary back onto the source stream.
    """
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [
            {"word": u["text"], "start": u["start"], "end": u["end"]} for u in units
        ],
    }


class Replayed:
    """Stand-in for ``SegmentationResult`` so a metric test can hand-build cues."""

    def __init__(self, cues: list[dict[str, Any]]) -> None:
        self.cues = cues
        self.diagnostics: dict[str, Any] = {}


def make_case(
    case_id: str,
    language: str,
    units: list[dict[str, Any]],
    *,
    tags: list[str] | None = None,
    vad_speech: list[list[float]] | None = None,
    shot_changes: list[float] | None = None,
    sing_spans: list[list[float]] | None = None,
    speaker_turns: list[dict[str, Any]] | None = None,
    exceptions: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A schema-valid golden case around ``units``.

    ``capture.config`` comes from ``capture_scenario.segmentation_config`` rather
    than a literal, so the test corpus is shaped by the real capture path and a
    capture/replay contract break shows up here.
    """
    latest = max(u["end"] for u in units)
    dependency_versions = dict(calib.dependency_versions())
    dependency_versions["python"] = platform.python_version()
    return {
        "schema_version": 1,
        "id": case_id,
        "language": language,
        "description": f"synthetic {language} fixture",
        "tags": tags or ["synthetic"],
        "license": {
            "redistributable": True,
            "source_class": "synthetic-from-consented-speech",
            "spdx": None,
            "attribution": None,
        },
        "capture": {
            "voxweave_commit": "0" * 40,
            "source_digest": "0" * 64,
            "window_duration_s": round(latest + 0.5, 6),
            "dependency_versions": dependency_versions,
            "config": config or capture.segmentation_config(language),
            "missing_inputs": [],
        },
        "word_segments": units,
        "vad_speech": vad_speech or [],
        "shot_changes": shot_changes or [],
        "sing_spans": sing_spans or [],
        "speaker_turns": speaker_turns or [],
        **({"exceptions": exceptions} if exceptions else {}),
    }


def as_case(doc: dict[str, Any]) -> Any:
    """Wrap a case document the way ``load_corpus`` would, without touching disk."""
    return calib.Case(
        path=Path(f"{doc['id']}.json"),
        relpath=f"cases/{doc['id']}.json",
        doc=doc,
        size_bytes=len(json.dumps(doc).encode("utf-8")),
    )


REQUIRED_TAGS = [
    "fast",
    "sparse-tail",
    "interruption",
    "noisy-punctuation",
    "code-switch",
    "shot",
    "speaker",
]

# Long enough that a replay produces several cues per case: a single-cue case has
# no internal boundary, so two of the four metrics would never be exercised
# end to end.
_WORDS = [
    (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliett "
        "kilo lima mike november oscar papa quebec romeo sierra tango "
        "uniform victor whiskey xray yankee zulu apple bridge cedar dawn"
    ),
    (
        "the meeting starts at nine and the agenda covers the migration plan "
        "before we review the budget and then the team will walk through "
        "the remaining questions about the rollout schedule and support"
    ),
    (
        "we walked to the old bridge yesterday and watched the river run "
        "under the arches while the evening light faded behind the hills "
        "and the town slowly settled into an unusually quiet evening"
    ),
]
_ZH = (
    "今天天气很好我们一起去公园散步然后回家吃饭休息一下"
    "明天还要早起上班所以今晚不能睡得太晚这样才能保持精神"
)
_JA = (
    "今日はとてもいい天気ですから公園を散歩してから家に帰ります"
    "明日も早く起きて仕事に行くので今夜は早めに休むつもりです"
)


def _corpus_case(index: int, language: str, ordinal: int) -> dict[str, Any]:
    """One member of the generated 20-case corpus.

    Every required tag is covered, and a few cases carry shot changes, speaker
    turns and an exception so the replay exercises those passes rather than only
    the plain content path.
    """
    case_id = f"{language}-{ordinal:02d}"
    tags = ["synthetic", REQUIRED_TAGS[index % len(REQUIRED_TAGS)]]
    extra: dict[str, Any] = {}
    if language == "en":
        units = spaced_units(_WORDS[index % len(_WORDS)].split(), dur=0.6)
    elif language == "zh":
        units = char_units(_ZH, dur=0.28)
    else:
        units = char_units(_JA, dur=0.3)
    span_end = units[-1]["end"]
    extra["vad_speech"] = [[units[0]["start"], span_end]]
    if "shot" in tags:
        extra["shot_changes"] = [round(span_end / 3, 3), round(2 * span_end / 3, 3)]
    if "speaker" in tags:
        middle = round(span_end / 2, 3)
        extra["speaker_turns"] = [
            {"start": units[0]["start"], "end": middle, "speaker": "S0"},
            {"start": middle, "end": span_end, "speaker": "S1"},
        ]
    return make_case(case_id, language, units, tags=tags, **extra)


def write_corpus(root: Path) -> Path:
    """Write a corpus that satisfies ``required_counts`` (zh 7, ja 7, en 6)."""
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    relpaths: list[str] = []
    index = 0
    for language, count in (("zh", 7), ("ja", 7), ("en", 6)):
        for ordinal in range(1, count + 1):
            doc = _corpus_case(index, language, ordinal)
            index += 1
            path = cases_dir / f"{doc['id']}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            relpaths.append(f"cases/{path.name}")
    registry = {
        "schema_version": 1,
        "metric_definition_version": calib.METRIC_DEFINITION_VERSION,
        "description": "synthetic corpus for the harness tests",
        "cases": relpaths,
        "required_counts": {"zh": 7, "ja": 7, "en": 6},
        "required_tags": REQUIRED_TAGS,
    }
    registry_path = root / "corpus.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    return registry_path


def run_cli(argv: list[str]) -> int:
    """Invoke ``main`` with the same error mapping ``cc.run_cli`` applies."""
    try:
        return calib.main(argv)
    except cc.CalibrationError:
        return cc.EXIT_INVALID
    except SystemExit as exc:  # pragma: no cover - defensive
        return int(exc.code or 0)


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "segmentation")


# --------------------------------------------------------------------------- #
# Metric 2: cue duration
# --------------------------------------------------------------------------- #


def test_over_7s_counts_cues_past_the_configured_max() -> None:
    units = spaced_units(["one", "two", "three"], dur=6.0)
    doc = make_case("en-01", "en", units)
    max_cue_s = doc["capture"]["config"]["max_cue_s"]
    assert max_cue_s == 7.0
    cues = [
        cue("one", 0.0, 3.0, units[:1]),
        cue("two", 6.0, 13.5, units[1:2]),  # 7.5s -> bad
        cue("three", 14.0, 22.0, units[2:3]),  # 8.0s -> bad
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    ratio = measurement.ratios["over_7s_rate"]
    assert (ratio.bad, ratio.eligible) == (2, 3)
    assert ratio.value == pytest.approx(2 / 3)
    assert [row["cue_index"] for row in measurement.offenders["over_7s_rate"]] == [2, 1]


def test_held_speech_exception_leaves_the_cue_in_coverage() -> None:
    """An exception removes an offender from its metric, not from the corpus.

    Dropping it from ``cue_count`` too would let a case shrink its own
    denominator by declaring exceptions; the coverage number is what makes that
    visible.
    """
    units = spaced_units(["one", "two", "three"], dur=6.0)
    doc = make_case(
        "en-01",
        "en",
        units,
        exceptions=[
            {
                "kind": "held_speech_over_7s",
                "range": [14.0, 22.0],
                "reason": "sung note held across the whole line",
            }
        ],
    )
    cues = [
        cue("one", 0.0, 3.0, units[:1]),
        cue("two", 6.0, 13.5, units[1:2]),
        cue("three", 14.0, 22.0, units[2:3]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    ratio = measurement.ratios["over_7s_rate"]
    assert (ratio.bad, ratio.eligible) == (1, 2)
    assert measurement.cue_count == 3
    assert measurement.diagnostics["exempted_cues"] == 1


# --------------------------------------------------------------------------- #
# Metric 3: reading speed
# --------------------------------------------------------------------------- #


def test_cps_uses_production_reading_chars_and_type7_percentile() -> None:
    units = spaced_units(["ab", "cd", "ef"], dur=1.0)
    doc = make_case("en-01", "en", units)
    cues = [
        cue("abcdef", 0.0, 2.0, units[:1]),  # 6 chars / 2.0s  -> 3.0
        cue("ab cd", 2.0, 3.0, units[1:2]),  # 4 chars / 1.0s  -> 4.0 (space free)
        cue("abcde", 3.0, 4.0, units[2:3]),  # 5 chars / 1.0s  -> 5.0
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    assert measurement.cps_samples == pytest.approx([3.0, 4.0, 5.0])

    groups = calib.aggregate([measurement])
    # type-7 on [3,4,5] at p90: idx = 2*0.9 = 1.8 -> 4 + 0.8*(5-4) = 4.8
    assert groups["en"]["cps_p90"]["value"] == pytest.approx(4.8)
    assert groups["en"]["cps_p90"]["n"] == 3
    assert groups["en"]["cps_p90"]["value"] == pytest.approx(
        cc.percentile([3.0, 4.0, 5.0], 90.0)
    )
    # The absolute ceiling is derived per language, never one global CPS number.
    assert groups["en"]["cps_p90"]["target_cps"] == pytest.approx(17.0)
    assert groups["en"]["cps_p90"]["absolute_max"] == pytest.approx(21.25)


def test_non_positive_cue_duration_is_invalid_not_infinite_cps() -> None:
    units = spaced_units(["ab", "cd"], dur=1.0)
    doc = make_case("en-01", "en", units)
    cues = [
        cue("ab", 0.0, 1.0, units[:1]),
        cue("cd", 1.0, 1.0, units[1:2]),  # zero duration
    ]
    with pytest.raises(cc.CalibrationError):
        calib.measure_case(as_case(doc), Replayed(cues))


# --------------------------------------------------------------------------- #
# Metric 1: length-driven mid-phrase breaks
# --------------------------------------------------------------------------- #


@pytest.fixture
def one_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the whole document is a single unbreakable phrase.

    Isolates the boundary bookkeeping (silence / punctuation / exception) from
    whichever segmenter happens to be installed: with one atom, every internal
    boundary is mid-phrase unless one of those rules excludes it.
    """
    monkeypatch.setattr(
        calib, "_phrase_atoms", lambda text, iso: [text.replace(" ", "")]
    )


@pytest.fixture
def per_char_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend every character is its own phrase: no break can be mid-phrase."""
    monkeypatch.setattr(
        calib, "_phrase_atoms", lambda text, iso: [c for c in text if not c.isspace()]
    )


def _zh_three_cues(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        cue(
            "".join(u["text"] for u in units[0:2]),
            units[0]["start"],
            units[1]["end"],
            units[0:2],
        ),
        cue(
            "".join(u["text"] for u in units[2:5]),
            units[2]["start"],
            units[4]["end"],
            units[2:5],
        ),
        cue(
            "".join(u["text"] for u in units[5:8]),
            units[5]["start"],
            units[7]["end"],
            units[5:8],
        ),
    ]


def test_len_break_mid_phrase_counts_every_length_driven_boundary(
    one_phrase: None,
) -> None:
    units = char_units("一二三四五六七八", dur=0.5)
    doc = make_case("zh-01", "zh", units)
    measurement = calib.measure_case(as_case(doc), Replayed(_zh_three_cues(units)))
    ratio = measurement.ratios["len_break_mid_phrase_rate"]
    assert (ratio.bad, ratio.eligible) == (2, 2)
    assert measurement.diagnostics["phrase_granularity"] == "phrase"


def test_acoustic_silence_break_leaves_the_denominator(one_phrase: None) -> None:
    """A boundary the speaker's pause forced is not a layout decision.

    zh ``offline_ms`` is 700 ms, so the 1.0 s hole before the second cue makes
    that boundary a silence break: it drops out of numerator *and* denominator.
    """
    units = char_units("一二三四五六七八", dur=0.5)
    for unit in units[2:]:
        unit["start"] += 1.0
        unit["end"] += 1.0
    doc = make_case("zh-01", "zh", units)
    measurement = calib.measure_case(as_case(doc), Replayed(_zh_three_cues(units)))
    ratio = measurement.ratios["len_break_mid_phrase_rate"]
    assert (ratio.bad, ratio.eligible) == (1, 1)
    assert measurement.diagnostics["silence_breaks"] == 1


def test_source_punctuation_makes_a_break_intended_not_bad(one_phrase: None) -> None:
    """Punctuation rides on the source unit, and it is the speaker's own boundary."""
    units = char_units("一二三四五六七八", dur=0.5)
    units[1]["text"] = "二。"
    doc = make_case("zh-01", "zh", units)
    measurement = calib.measure_case(as_case(doc), Replayed(_zh_three_cues(units)))
    ratio = measurement.ratios["len_break_mid_phrase_rate"]
    assert (ratio.bad, ratio.eligible) == (1, 2)


def test_break_at_a_phrase_start_is_never_bad(per_char_phrase: None) -> None:
    units = char_units("一二三四五六七八", dur=0.5)
    doc = make_case("zh-01", "zh", units)
    measurement = calib.measure_case(as_case(doc), Replayed(_zh_three_cues(units)))
    ratio = measurement.ratios["len_break_mid_phrase_rate"]
    assert (ratio.bad, ratio.eligible) == (0, 2)


def test_spaced_language_reports_word_granularity_and_no_mid_phrase(
    one_phrase: None,
) -> None:
    """For English an atom is a word, so "inside a phrase" is not expressible.

    The denominator still counts the boundaries -- the gate keeps its samples --
    but the report labels the granularity so a structural zero cannot be read as
    an achievement.
    """
    units = spaced_units(["we", "walked", "to", "old", "bridge", "now"], dur=0.5)
    doc = make_case("en-01", "en", units)
    cues = [
        cue("we walked", 0.0, 1.0, units[0:2]),
        cue("to old", 1.0, 2.0, units[2:4]),
        cue("bridge now", 2.0, 3.0, units[4:6]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    ratio = measurement.ratios["len_break_mid_phrase_rate"]
    assert (ratio.bad, ratio.eligible) == (0, 2)
    assert measurement.diagnostics["phrase_granularity"] == "word"


def test_phrase_truth_comes_from_the_source_stream_not_the_cue_text() -> None:
    """The segmenter runs on the input document, so agreement is not tautological.

    Punctuation only exists in the source units (the splitter strips it), so a
    phrase map computed from cue text cannot equal this one -- which is the whole
    point of measuring against the input.
    """
    units = char_units("一二三四", dur=0.5)
    units[1]["text"] = "二。"
    starts, offsets = calib.phrase_start_offsets(units, "zh")
    assert offsets == [0, 1, 3, 4]  # "二。" occupies two reading characters
    assert 0 in starts and 5 in starts  # document start and end are phrase edges


# --------------------------------------------------------------------------- #
# Metric 4: forbidden line ends
# --------------------------------------------------------------------------- #


def test_forbidden_end_counts_a_dangling_article_when_an_alternative_existed() -> None:
    units = spaced_units(
        ["we", "walked", "to", "the", "old", "bridge", "yesterday"], dur=0.5
    )
    doc = make_case("en-01", "en", units)
    cues = [
        cue("we walked to the", 0.0, 2.0, units[0:4]),
        cue("old bridge yesterday", 2.0, 3.5, units[4:7]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    ratio = measurement.ratios["forbidden_end_rate"]
    assert (ratio.bad, ratio.eligible) == (1, 2)
    assert measurement.offenders["forbidden_end_rate"][0]["note"] == "the"


def test_forbidden_end_counts_a_document_final_dangling_tail() -> None:
    units = spaced_units(["we", "walked", "to", "the"], dur=0.5)
    doc = make_case("en-01", "en", units)
    cues = [cue("we walked to the", 0.0, 2.0, units)]

    measurement = calib.measure_case(as_case(doc), Replayed(cues))

    assert measurement.ratios["forbidden_end_rate"] == cc.Ratio(1, 1)
    assert measurement.diagnostics["final_tail_eligible"] == 1
    assert measurement.offenders["forbidden_end_rate"][0]["cue_index"] == 0


def test_document_final_sentence_punctuation_excludes_the_tail() -> None:
    units = spaced_units(["we", "walked", "to", "the."], dur=0.5)
    doc = make_case("en-01", "en", units)
    cues = [cue("we walked to the", 0.0, 2.0, units)]

    measurement = calib.measure_case(as_case(doc), Replayed(cues))

    assert measurement.ratios["forbidden_end_rate"] == cc.Ratio(0, 0)
    assert measurement.diagnostics["final_tail_eligible"] == 0
    assert measurement.diagnostics["terminal_final_tails"] == 1


def test_ja_final_tail_uses_level2_pos_instead_of_the_char_false_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UniDic identifies final ``あの`` as a legal filler even though L1 sees の."""
    units = char_units("あの", dur=0.5)
    doc = make_case("ja-01", "ja", units)
    cues = [cue("あの", 0.0, 1.0, units)]

    monkeypatch.setattr(calib, "_ja_pos_end_penalties", lambda text: {1: 0})
    level2 = calib.measure_case(as_case(doc), Replayed(cues))
    assert level2.ratios["forbidden_end_rate"] == cc.Ratio(0, 1)

    monkeypatch.setattr(calib, "_ja_pos_end_penalties", lambda text: None)
    fallback = calib.measure_case(as_case(doc), Replayed(cues))
    assert fallback.ratios["forbidden_end_rate"] == cc.Ratio(1, 1)


def test_ja_alternative_legality_uses_the_same_level2_source_offset_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An L1-clean 連体詞 tail is not a legal alternative under UniDic L2."""
    units = char_units("いわゆる猫犬", dur=0.5)
    starts = {0, 4, 5, 6}
    offsets = list(range(6))

    # Level 1 sees final る as clean. Level 2 scores the whole-source token
    # いわゆる (ending at offset 3) as a forward-binding 連体詞.
    monkeypatch.setattr(calib, "_ja_pos_end_penalties", lambda text: None)
    assert calib.has_legal_alternative(
        units,
        starts,
        offsets,
        span_start_unit=0,
        actual_right_unit=5,
        span_end_unit=5,
        iso="ja",
        max_line_length=5,
        max_lines=1,
    )
    monkeypatch.setattr(calib, "_ja_pos_end_penalties", lambda text: {3: 2})
    assert not calib.has_legal_alternative(
        units,
        starts,
        offsets,
        span_start_unit=0,
        actual_right_unit=5,
        span_end_unit=5,
        iso="ja",
        max_line_length=5,
        max_lines=1,
    )


def test_ja_level2_map_uses_the_atom_from_the_punctuated_source_lattice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    units = char_units("これは、あの", dur=0.4)
    seen: list[str] = []

    def capture_source(text: str) -> dict[int, int]:
        seen.append(text)
        return {1: 0}

    monkeypatch.setattr(calib, "_ja_pos_end_penalties", capture_source)
    doc = make_case("ja-01", "ja", units)
    cues = [cue("これは あの", 0.0, units[-1]["end"], units)]
    calib.measure_case(as_case(doc), Replayed(cues))

    assert calib._source_span_text(units, 0, len(units) - 1, "ja") == "これは、あの"
    assert seen == ["あの"]


def test_ja_level2_tail_keeps_punctuation_only_units_in_its_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    units = char_units("お、猫", dur=0.4)
    _, offsets = calib.phrase_start_offsets(units, "ja")
    seen: list[str] = []

    def punctuation_disambiguates(text: str) -> dict[int, int]:
        seen.append(text)
        return {0: 0, 1: 0}

    monkeypatch.setattr(calib, "_ja_pos_end_penalties", punctuation_disambiguates)
    penalty = calib._forbidden_tail_penalty(
        units,
        offsets,
        start_unit=0,
        end_unit=0,
        context_end_unit=1,
        display_text="お",
        iso="ja",
        ja_pos_cache={},
    )

    assert penalty == 0
    assert seen == ["お、"]


def test_legal_alternative_uses_the_pre_split_source_lattice(monkeypatch) -> None:
    units = char_units("甲乙丙丁戊己", dur=0.5)
    units[-1]["text"] = "己。"

    def context_sensitive_atoms(text: str, iso: str) -> list[str]:
        compact = text.replace(" ", "")
        if compact == "甲乙丙丁戊己。":
            return ["甲乙", "丙丁", "戊己。"]
        return [compact]

    monkeypatch.setattr(calib, "_phrase_atoms", context_sensitive_atoms)
    monkeypatch.setattr(
        calib,
        "_line_end_penalty",
        lambda text, iso: 2 if text.endswith("丙") else 0,
    )
    starts, offsets = calib.phrase_start_offsets(units, "zh")

    # Re-segmenting the two post-split cues yields only their actual boundary,
    # so the old view had no alternative. The whole-source lattice has legal
    # boundaries at offsets 2 and 4.
    assert context_sensitive_atoms("甲乙丙", "zh") + context_sensitive_atoms(
        "丁戊己", "zh"
    ) == ["甲乙丙", "丁戊己"]
    assert calib.has_legal_alternative(
        units,
        starts,
        offsets,
        span_start_unit=0,
        actual_right_unit=3,
        span_end_unit=5,
        iso="zh",
        max_line_length=4,
        max_lines=1,
    )

    config = capture.segmentation_config("zh")
    config.update({"max_line_length": 4, "max_lines": 1})
    doc = make_case("zh-01", "zh", units, config=config)
    cues = [
        cue("甲乙丙", 0.0, 1.5, units[:3]),
        cue("丁戊己", 1.5, 3.0, units[3:]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    assert measurement.ratios["forbidden_end_rate"] == cc.Ratio(1, 1)
    assert measurement.diagnostics["no_legal_alternative"] == 0


def test_forbidden_end_denominator_drops_boundaries_with_no_alternative() -> None:
    """Two single-word cues cannot be repacked, so the boundary is not a defect.

    Counting it would make the metric unreachable: a rate that punishes
    unsolvable boundaries can never reach its target no matter what the splitter
    does.
    """
    units = spaced_units(["the", "cat"], dur=0.5)
    doc = make_case("en-01", "en", units)
    cues = [
        cue("the", 0.0, 0.5, units[0:1]),
        cue("cat", 0.5, 1.0, units[1:2]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    ratio = measurement.ratios["forbidden_end_rate"]
    assert (ratio.bad, ratio.eligible) == (0, 1)
    assert ratio.value == 0.0
    assert measurement.diagnostics["no_legal_alternative"] == 1


def test_forbidden_end_drops_boundaries_forced_by_a_long_pause() -> None:
    units = spaced_units(
        ["we", "walked", "to", "the", "old", "bridge", "yesterday"], dur=0.5
    )
    for unit in units[4:]:  # vad_skip_ms defaults to 1000 ms
        unit["start"] += 1.5
        unit["end"] += 1.5
    doc = make_case("en-01", "en", units)
    cues = [
        cue("we walked to the", 0.0, 2.0, units[0:4]),
        cue("old bridge yesterday", 3.5, 5.0, units[4:7]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    ratio = measurement.ratios["forbidden_end_rate"]
    assert (ratio.bad, ratio.eligible) == (0, 1)
    assert measurement.diagnostics["forced_breaks"] == 1


def test_forbidden_end_drops_boundaries_the_source_punctuated() -> None:
    units = spaced_units(
        ["we", "walked", "to", "the,", "old", "bridge", "yesterday"], dur=0.5
    )
    doc = make_case("en-01", "en", units)
    cues = [
        cue("we walked to the", 0.0, 2.0, units[0:4]),
        cue("old bridge yesterday", 2.0, 3.5, units[4:7]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    ratio = measurement.ratios["forbidden_end_rate"]
    assert (ratio.bad, ratio.eligible) == (0, 1)
    assert measurement.diagnostics["punctuation_breaks"] == 1


def test_unavoidable_forbidden_end_exception_excludes_only_that_metric() -> None:
    units = spaced_units(
        ["we", "walked", "to", "the", "old", "bridge", "yesterday"], dur=0.5
    )
    doc = make_case(
        "en-01",
        "en",
        units,
        exceptions=[
            {
                "kind": "unavoidable_forbidden_end",
                "range": [0.0, 2.0],
                "reason": "speaker pauses mid-phrase, no in-budget alternative",
            }
        ],
    )
    cues = [
        cue("we walked to the", 0.0, 2.0, units[0:4]),
        cue("old bridge yesterday", 2.0, 3.5, units[4:7]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    assert measurement.ratios["forbidden_end_rate"] == cc.Ratio(0, 1)
    # ... while the cue-duration metric still sees both cues.
    assert measurement.ratios["over_7s_rate"].eligible == 2


def test_known_bad_source_unit_exception_excludes_every_metric(
    one_phrase: None,
) -> None:
    """A defective input unit is not evidence about the splitter, in any metric."""
    units = char_units("一二三四五六七八", dur=0.5)
    doc = make_case(
        "zh-01",
        "zh",
        units,
        exceptions=[
            {
                "kind": "known_bad_source_unit",
                "range": [0.0, 4.0],
                "reason": "aligner collapsed this run to zero duration",
            }
        ],
    )
    measurement = calib.measure_case(as_case(doc), Replayed(_zh_three_cues(units)))
    assert measurement.ratios["len_break_mid_phrase_rate"] == cc.Ratio(0, 0)
    assert measurement.ratios["over_7s_rate"] == cc.Ratio(0, 0)
    assert measurement.ratios["forbidden_end_rate"] == cc.Ratio(0, 0)
    assert measurement.cps_samples == []
    assert measurement.cue_count == 3  # ... but the coverage still counts them


# --------------------------------------------------------------------------- #
# Unit health: the zero-duration ledger split by mechanism
# --------------------------------------------------------------------------- #


def test_unit_health_tells_collapse_from_quantization() -> None:
    """Same total zero rate, opposite shapes.

    Quantization (the ja MMS lane) produces ordered zeros on distinct
    timestamps; collapse (the zh NAR failure) lands a run on one identical
    timestamp. Only the second is a wall.
    """
    quantized = [
        {"text": "あ", "start": 0.0, "end": 0.1},
        {"text": "い", "start": 0.1, "end": 0.1},
        {"text": "う", "start": 0.2, "end": 0.2},
        {"text": "え", "start": 0.3, "end": 0.4},
    ]
    health = calib.unit_health(quantized)
    assert health["lexical_zero"] == 2
    assert health["same_time_wall_max"] == 1
    assert health["lexical_zero_run_max"] == 2

    collapsed = [
        {"text": "T", "start": 0.0, "end": 0.1},
        {"text": "r", "start": 0.5, "end": 0.5},
        {"text": "a", "start": 0.5, "end": 0.5},
        {"text": "n", "start": 0.5, "end": 0.5},
        {"text": "s", "start": 0.6, "end": 0.7},
    ]
    health = calib.unit_health(collapsed)
    assert health["same_time_wall_max"] == 3
    assert health["lexical_zero_run_max"] == 3


def test_unit_health_punct_zeros_are_a_separate_column() -> None:
    """reinject_punct's zero-width punctuation must not count as lexical."""
    units = [
        {"text": "词", "start": 0.0, "end": 0.2},
        {"text": "。", "start": 0.2, "end": 0.2},
        {"text": "再", "start": 1.0, "end": 1.2},
    ]
    health = calib.unit_health(units)
    assert health["lexical_zero"] == 0
    assert health["punct_zero"] == 1
    assert health["lexical_count"] == 2


def test_unit_health_stranded_tail_needs_adjacent_word_units() -> None:
    """A big gap behind punctuation is a sentence pause, not a stranded tail."""
    stranded = [
        {"text": "弱", "start": 22.5, "end": 23.0},
        {"text": "い", "start": 29.5, "end": 30.2},
    ]
    health = calib.unit_health(stranded)
    assert health["stranded_gap_count"] == 1
    assert health["stranded_gap_max_s"] == 6.5

    paused = [
        {"text": "弱", "start": 22.5, "end": 23.0},
        {"text": "。", "start": 23.0, "end": 23.0},
        {"text": "い", "start": 29.5, "end": 30.2},
    ]
    assert calib.unit_health(paused)["stranded_gap_count"] == 0


def test_unit_health_long_unit_vad_coverage() -> None:
    """A >1s unit reports how much of it actually lies in speech."""
    units = [
        {"text": "っ", "start": 0.0, "end": 2.0},
        {"text": "あ", "start": 2.0, "end": 2.2},
    ]
    health = calib.unit_health(units, vad_speech=[(0.0, 1.0)])
    assert health["long_unit_count"] == 1
    assert health["long_unit_min_vad_coverage"] == 0.5
    assert calib.unit_health(units)["long_unit_min_vad_coverage"] is None


def test_unit_health_counts_nonmonotonic_pairs() -> None:
    """Order regressions in the source stream are visible, not smoothed over."""
    units = [
        {"text": "上", "start": 94.9, "end": 94.9},
        {"text": "，", "start": 7.5, "end": 7.5},
        {"text": "次", "start": 8.0, "end": 8.2},
    ]
    assert calib.unit_health(units)["nonmonotonic_pairs"] == 1


# --------------------------------------------------------------------------- #
# Boundary mapping
# --------------------------------------------------------------------------- #


def test_unmappable_boundary_is_reported_not_guessed() -> None:
    """A fabricated ``word_data`` span must not be silently matched to a unit.

    The ghost sits in a long cue stream so its two lost boundaries stay under
    the invalid-measurement ceiling: they are excluded and counted, not
    guessed at, and the rest of the stream is still measured.
    """
    words = [f"w{i}" for i in range(221)]
    units = spaced_units(words, dur=0.5)
    doc = make_case("en-01", "en", units)
    cues = [
        cue(w, i * 0.5, (i + 1) * 0.5, units[i : i + 1]) for i, w in enumerate(words)
    ]
    cues[110] = cue("ghost", 55.0, 55.5, [])
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    assert measurement.diagnostics["unmapped_boundaries"] == 2


def test_heavy_unmapped_is_an_invalid_measurement() -> None:
    """Losing more than 1% of boundaries fails the case as a measurement error.

    The metrics graded a cue stream they mostly could not see, so the run must
    exit 2 (invalid), never report numbers from the visible remainder.
    """
    units = spaced_units(["a", "b", "c", "d"], dur=0.5)
    doc = make_case("en-01", "en", units)
    ghost = cue("b c", 0.5, 1.5, [])
    cues = [cue("a", 0.0, 0.5, units[0:1]), ghost, cue("d", 1.5, 2.0, units[3:4])]
    with pytest.raises(cc.CalibrationError):
        calib.measure_case(as_case(doc), Replayed(cues))


def test_atom_level_word_data_maps_by_span_edges() -> None:
    """A repacked (atom-level) entry resolves through its edge timestamps.

    ``_chunk_to_cue`` emits word_data whose span aggregates several units; the
    exact ``(start, end)`` key then matches no single unit, but the entry still
    starts on its first unit and ends on its last one.
    """
    units = spaced_units(["a", "b", "c", "d"], dur=0.5)
    doc = make_case("en-01", "en", units)
    merged = {"text": "b c", "start": units[1]["start"], "end": units[2]["end"]}
    cues = [
        cue("a", 0.0, 0.5, units[0:1]),
        cue("b c", 0.5, 1.5, [merged]),
        cue("d", 1.5, 2.0, units[3:4]),
    ]
    measurement = calib.measure_case(as_case(doc), Replayed(cues))
    assert measurement.diagnostics["unmapped_boundaries"] == 0


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_aggregation_is_micro_not_an_average_of_case_rates() -> None:
    """A short clip must not weigh the same as a long one.

    Case A is 1 bad of 1 boundary (100%), case B is 1 bad of 9 (11.1%). The
    macro average both cases used to feed would be 55.6%; the micro rate is
    2/10 = 20%, and the counts stay in the report to prove it.
    """
    small = calib.CaseMeasurement(
        case_id="zh-01",
        language="zh",
        cue_count=2,
        ratios={"len_break_mid_phrase_rate": cc.Ratio(1, 1)},
        diagnostics={"target_cps": 9.0, "phrase_granularity": "phrase"},
    )
    large = calib.CaseMeasurement(
        case_id="zh-02",
        language="zh",
        cue_count=10,
        ratios={"len_break_mid_phrase_rate": cc.Ratio(1, 9)},
        diagnostics={"target_cps": 9.0, "phrase_granularity": "phrase"},
    )
    groups = calib.aggregate([small, large])
    block = groups["zh"]["len_break_mid_phrase_rate"]
    assert (block["bad"], block["eligible"]) == (2, 10)
    assert block["value"] == pytest.approx(0.2)
    macro_average = (1 / 1 + 1 / 9) / 2
    assert block["value"] != pytest.approx(macro_average)
    assert groups["zh"]["case_count"] == 2
    assert groups["all"]["len_break_mid_phrase_rate"]["eligible"] == 10


def test_empty_denominator_reports_null_not_zero() -> None:
    measurement = calib.CaseMeasurement(
        case_id="en-01",
        language="en",
        cue_count=1,
        ratios={"forbidden_end_rate": cc.Ratio(0, 0)},
        diagnostics={"target_cps": 17.0, "phrase_granularity": "word"},
    )
    groups = calib.aggregate([measurement])
    assert groups["en"]["forbidden_end_rate"]["value"] is None
    assert groups["en"]["cps_p90"]["value"] is None


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def _groups_with(language: str, **values: Any) -> dict[str, Any]:
    block = {
        "case_count": 1,
        "cue_count": 200,
        "len_break_mid_phrase_rate": {"bad": 10, "eligible": 200, "value": 0.05},
        "over_7s_rate": {"bad": 0, "eligible": 200, "value": 0.0},
        "forbidden_end_rate": {"bad": 2, "eligible": 200, "value": 0.01},
        "cps_p90": {"n": 200, "value": 9.0, "target_cps": 9.0, "absolute_max": 11.25},
    }
    block.update(values)
    return {language: block}


def _gate(metric: str, **overrides: Any) -> dict[str, Any]:
    gate = dict(calib.DEFAULT_GATES[metric])
    gate["mode"] = "blocking"
    gate["min_samples"] = 1
    gate.update(overrides)
    return gate


def test_regression_fails_and_improvement_passes() -> None:
    """Gates are one-sided: only a move in the wrong direction can fail."""
    gates = {m: _gate(m) for m in calib.METRICS}
    baseline_worse = {"groups": _groups_with("zh", cps_p90={"n": 200, "value": 12.0})}
    baseline_better = {"groups": _groups_with("zh", cps_p90={"n": 200, "value": 6.0})}
    current = _groups_with("zh")

    improved = next(
        r
        for r in calib.evaluate_gates(current, gates, baseline_worse)
        if r["metric"] == "cps_p90"
    )
    assert improved["status"] == "pass"
    assert calib.gate_exit_code([improved]) == cc.EXIT_OK

    regressed = next(
        r
        for r in calib.evaluate_gates(current, gates, baseline_better)
        if r["metric"] == "cps_p90"
    )
    assert regressed["status"] == "fail"
    assert regressed["baseline_value"] == pytest.approx(6.0)
    assert calib.gate_exit_code([regressed]) == cc.EXIT_GATE_FAILED


def test_absolute_ceiling_fails_even_without_a_baseline() -> None:
    gates = {m: _gate(m) for m in calib.METRICS}
    current = _groups_with(
        "zh", cps_p90={"n": 200, "value": 40.0, "absolute_max": 11.25}
    )
    result = next(
        r
        for r in calib.evaluate_gates(current, gates, None)
        if r["metric"] == "cps_p90"
    )
    assert result["status"] == "fail"
    assert "absolute" in result["reasons"][0]


def test_over_7s_is_gated_on_the_raw_count_not_a_diluted_rate() -> None:
    """One long cue is a defect whether the corpus holds 200 cues or 20 000."""
    gates = {m: _gate(m) for m in calib.METRICS}
    current = _groups_with(
        "zh", over_7s_rate={"bad": 1, "eligible": 20000, "value": 0.00005}
    )
    result = next(
        r
        for r in calib.evaluate_gates(current, gates, None)
        if r["metric"] == "over_7s_rate"
    )
    assert result["measure"] == "count"
    assert result["value"] == pytest.approx(1.0)
    assert result["status"] == "fail"


def test_forbidden_end_gate_compares_bad_count_with_one_event_slack() -> None:
    gates = {m: _gate(m, mode="disabled") for m in calib.METRICS}
    gates["forbidden_end_rate"] = _gate("forbidden_end_rate")
    baseline = {
        "groups": _groups_with(
            "zh", forbidden_end_rate={"bad": 2, "eligible": 100, "value": 0.02}
        )
    }

    at_slack = _groups_with(
        "zh", forbidden_end_rate={"bad": 3, "eligible": 400, "value": 0.0075}
    )
    result = next(
        row
        for row in calib.evaluate_gates(at_slack, gates, baseline)
        if row["metric"] == "forbidden_end_rate"
    )
    assert result["measure"] == "count"
    assert result["value"] == 3.0
    assert result["allowed_by_baseline"] == 3.0
    assert result["status"] == "pass"
    assert (result["numerator"], result["denominator"]) == (3, 400)
    assert result["reported_rate"] == pytest.approx(0.0075)

    over_slack = _groups_with(
        "zh", forbidden_end_rate={"bad": 4, "eligible": 1000, "value": 0.004}
    )
    result = next(
        row
        for row in calib.evaluate_gates(over_slack, gates, baseline)
        if row["metric"] == "forbidden_end_rate"
    )
    assert result["status"] == "fail"


def test_thin_denominator_is_invalid_not_a_pass() -> None:
    """A gate that cannot be measured has no standing to say "pass"."""
    gates = {m: _gate(m, min_samples=100) for m in calib.METRICS}
    current = _groups_with(
        "zh", forbidden_end_rate={"bad": 0, "eligible": 3, "value": 0.0}
    )
    result = next(
        r
        for r in calib.evaluate_gates(current, gates, None)
        if r["metric"] == "forbidden_end_rate"
    )
    assert result["status"] == "insufficient_samples"
    assert calib.gate_exit_code([result]) == cc.EXIT_INVALID


def test_warning_mode_never_changes_the_exit_code() -> None:
    gates = {m: _gate(m, mode="warning") for m in calib.METRICS}
    current = _groups_with(
        "zh", cps_p90={"n": 200, "value": 40.0, "absolute_max": 11.25}
    )
    results = calib.evaluate_gates(current, gates, None)
    assert any(r["status"] == "fail" for r in results)
    assert calib.gate_exit_code(results) == cc.EXIT_OK


def test_warning_gate_promotes_when_both_sample_counts_reach_the_minimum() -> None:
    gates = {m: _gate(m, mode="disabled") for m in calib.METRICS}
    gates["forbidden_end_rate"] = _gate(
        "forbidden_end_rate", mode="warning", min_samples=100
    )
    current = _groups_with(
        "zh", forbidden_end_rate={"bad": 3, "eligible": 100, "value": 0.03}
    )
    baseline = {
        "groups": _groups_with(
            "zh", forbidden_end_rate={"bad": 0, "eligible": 100, "value": 0.0}
        )
    }

    result = next(
        row
        for row in calib.evaluate_gates(current, gates, baseline)
        if row["metric"] == "forbidden_end_rate"
    )

    assert result["configured_mode"] == "warning"
    assert result["mode"] == "blocking"
    assert result["promoted"] is True
    assert result["baseline_samples"] == 100
    assert result["status"] == "fail"
    assert calib.gate_exit_code([result]) == cc.EXIT_GATE_FAILED


@pytest.mark.parametrize(("current_n", "baseline_n"), [(99, 100), (100, 99)])
def test_warning_gate_stays_warning_until_both_sample_counts_are_ready(
    current_n: int, baseline_n: int
) -> None:
    gates = {m: _gate(m, mode="disabled") for m in calib.METRICS}
    gates["forbidden_end_rate"] = _gate(
        "forbidden_end_rate", mode="warning", min_samples=100
    )
    current = _groups_with(
        "zh",
        forbidden_end_rate={"bad": 0, "eligible": current_n, "value": 0.0},
    )
    baseline = {
        "groups": _groups_with(
            "zh",
            forbidden_end_rate={"bad": 0, "eligible": baseline_n, "value": 0.0},
        )
    }

    result = next(
        row
        for row in calib.evaluate_gates(current, gates, baseline)
        if row["metric"] == "forbidden_end_rate"
    )

    assert result["configured_mode"] == "warning"
    assert result["mode"] == "warning"
    assert result["promoted"] is False


# --------------------------------------------------------------------------- #
# Corpus validation
# --------------------------------------------------------------------------- #


def test_generated_corpus_validates(corpus_path: Path) -> None:
    assert run_cli(["validate-corpus", "--corpus", str(corpus_path)]) == cc.EXIT_OK
    corpus = calib.load_corpus(corpus_path)
    assert len(corpus.cases) == 20
    assert len(corpus.digest) == 64


def test_missing_required_tag_is_invalid(corpus_path: Path) -> None:
    registry = json.loads(corpus_path.read_text(encoding="utf-8"))
    registry["required_tags"] = [*registry["required_tags"], "no-case-has-this"]
    corpus_path.write_text(json.dumps(registry), encoding="utf-8")
    assert run_cli(["validate-corpus", "--corpus", str(corpus_path)]) == cc.EXIT_INVALID


def test_wrong_case_count_is_invalid(corpus_path: Path) -> None:
    registry = json.loads(corpus_path.read_text(encoding="utf-8"))
    registry["cases"] = [c for c in registry["cases"] if not c.endswith("zh-07.json")]
    corpus_path.write_text(json.dumps(registry), encoding="utf-8")
    assert run_cli(["validate-corpus", "--corpus", str(corpus_path)]) == cc.EXIT_INVALID


def test_oversized_case_is_invalid(corpus_path: Path) -> None:
    case_path = corpus_path.parent / "cases" / "en-01.json"
    doc = json.loads(case_path.read_text(encoding="utf-8"))
    doc["description"] = "x" * 200
    doc["tags"] = [f"pad-{i}" for i in range(30000)]
    case_path.write_text(json.dumps(doc), encoding="utf-8")
    assert case_path.stat().st_size > calib.MAX_CASE_BYTES
    assert run_cli(["validate-corpus", "--corpus", str(corpus_path)]) == cc.EXIT_INVALID


def test_case_time_past_its_declared_window_is_invalid(corpus_path: Path) -> None:
    case_path = corpus_path.parent / "cases" / "en-02.json"
    doc = json.loads(case_path.read_text(encoding="utf-8"))
    doc["capture"]["window_duration_s"] = 1.0
    case_path.write_text(json.dumps(doc), encoding="utf-8")
    assert run_cli(["validate-corpus", "--corpus", str(corpus_path)]) == cc.EXIT_INVALID


def test_missing_corpus_is_invalid(tmp_path: Path) -> None:
    assert (
        run_cli(["validate-corpus", "--corpus", str(tmp_path / "nope.json")])
        == cc.EXIT_INVALID
    )


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


def test_replay_is_deterministic(corpus_path: Path) -> None:
    """Two replays of one case must be byte-identical, or a gate measures noise."""
    corpus = calib.load_corpus(corpus_path)
    for case in corpus.cases[:6]:
        first = calib.replay(case)
        second = calib.replay(case)
        assert cc.canonical_digest(first.cues) == cc.canonical_digest(second.cues)
        assert first.thresholds_used == second.thresholds_used


def test_replay_uses_the_captured_thresholds_not_the_environment(
    corpus_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env override in the operator's shell must not redefine the corpus."""
    corpus = calib.load_corpus(corpus_path)
    case = corpus.cases[0]
    before = calib.replay(case)
    monkeypatch.setenv("VOXWEAVE_MAX_CUE_SEC", "2.0")
    monkeypatch.setenv("VOXWEAVE_CPS", "1.0")
    after = calib.replay(case)
    assert cc.canonical_digest(before.cues) == cc.canonical_digest(after.cues)
    assert after.thresholds_used["max_cue_s"] == pytest.approx(7.0)


def test_replay_pins_gap_adaptive_to_the_captured_value(
    corpus_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = calib.load_corpus(corpus_path)
    case = corpus.cases[0]
    assert case.config["gap_adaptive"] is False
    baseline = calib.replay(case)
    monkeypatch.setenv("VOXWEAVE_GAP_ADAPTIVE", "1")
    pinned = calib.replay(case)
    assert cc.canonical_digest(baseline.cues) == cc.canonical_digest(pinned.cues)
    # ... and the operator's environment survives the replay unchanged.
    import os

    assert os.environ["VOXWEAVE_GAP_ADAPTIVE"] == "1"


def test_each_case_replays_well_under_the_wall_budget(
    corpus_path: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"
    assert (
        run_cli(
            [
                "evaluate",
                "--corpus",
                str(corpus_path),
                "--json-out",
                str(report_path),
            ]
        )
        == cc.EXIT_OK
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["timing"]["slowest_wall_s"] < calib.CASE_WALL_TARGET_S


# --------------------------------------------------------------------------- #
# Report and baseline round trip
# --------------------------------------------------------------------------- #


def _evaluate(corpus_path: Path, out: Path, *extra: str) -> tuple[int, dict[str, Any]]:
    code = run_cli(
        ["evaluate", "--corpus", str(corpus_path), "--json-out", str(out), *extra]
    )
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return code, report


def test_report_keeps_numerator_denominator_and_offenders(
    corpus_path: Path, tmp_path: Path
) -> None:
    code, report = _evaluate(corpus_path, tmp_path / "report.json")
    assert code == cc.EXIT_OK
    for language in ("all", "zh", "ja", "en"):
        block = report["groups"][language]
        for metric in (
            "len_break_mid_phrase_rate",
            "over_7s_rate",
            "forbidden_end_rate",
        ):
            assert set(block[metric]) == {"bad", "eligible", "value"}
        assert "n" in block["cps_p90"]
    assert set(report["offenders"]) == set(calib.METRICS)
    assert report["corpus"]["case_count"] == 20
    assert len(report["cases"]) == 20
    assert report["metric_definition_digest"] == cc.canonical_digest(
        report["metric_definition"]
    )
    lens = report["metric_definition"]["forbidden_end"]["ja_tail_lens"]
    assert lens["id"] in {calib.JA_TAIL_LENS_LEVEL2, calib.JA_TAIL_LENS_LEVEL1}
    forbidden_claims = [
        row for row in report["gate_results"] if row["metric"] == "forbidden_end_rate"
    ]
    assert forbidden_claims
    assert all(
        {"numerator", "denominator", "reported_rate"} <= set(row)
        for row in forbidden_claims
    )


def test_report_groups_match_the_tracked_baseline_contract(
    corpus_path: Path, tmp_path: Path
) -> None:
    _, report = _evaluate(corpus_path, tmp_path / "report.json")
    group_schema = calib.group_schema()
    for block in report["groups"].values():
        assert cc.schema_errors(block, group_schema) == []
    assert cc.schema_errors(report["gates"], calib.gates_schema()) == []


def test_record_baseline_round_trips_and_then_passes(
    corpus_path: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"
    assert _evaluate(corpus_path, report_path)[0] == cc.EXIT_OK
    assert (
        run_cli(
            [
                "record-baseline",
                "--corpus",
                str(corpus_path),
                "--report",
                str(report_path),
                "--output",
                str(baseline_path),
            ]
        )
        == cc.EXIT_OK
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert cc.schema_errors(baseline, "segmentation-baseline") == []
    assert baseline["corpus_digest"] == calib.load_corpus(corpus_path).digest
    # Re-running against the freshly recorded baseline is a no-change comparison.
    assert (
        _evaluate(
            corpus_path,
            tmp_path / "report2.json",
            "--baseline",
            str(baseline_path),
            "--check",
        )[0]
        == cc.EXIT_OK
    )


def _record(corpus_path: Path, tmp_path: Path) -> Path:
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"
    _evaluate(corpus_path, report_path)
    run_cli(
        [
            "record-baseline",
            "--corpus",
            str(corpus_path),
            "--report",
            str(report_path),
            "--output",
            str(baseline_path),
        ]
    )
    return baseline_path


def _blocking(baseline_path: Path, metric: str, **mutate: Any) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for name in calib.METRICS:
        baseline["gates"][name]["mode"] = "disabled"
    baseline["gates"][metric].update(
        {
            "mode": "blocking",
            "min_samples": 1,
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0,
            **mutate,
        }
    )
    return baseline


def test_end_to_end_regression_exits_1(corpus_path: Path, tmp_path: Path) -> None:
    baseline_path = _record(corpus_path, tmp_path)
    baseline = _blocking(baseline_path, "cps_p90")
    for language in ("zh", "ja", "en"):
        value = baseline["groups"][language]["cps_p90"]["value"]
        baseline["groups"][language]["cps_p90"]["value"] = value * 0.5
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    code, report = _evaluate(
        corpus_path, tmp_path / "r.json", "--baseline", str(baseline_path), "--check"
    )
    assert code == cc.EXIT_GATE_FAILED
    assert any(
        r["metric"] == "cps_p90" and r["status"] == "fail"
        for r in report["gate_results"]
    )


def test_end_to_end_improvement_exits_0(corpus_path: Path, tmp_path: Path) -> None:
    baseline_path = _record(corpus_path, tmp_path)
    baseline = _blocking(baseline_path, "cps_p90")
    for language in ("zh", "ja", "en"):
        value = baseline["groups"][language]["cps_p90"]["value"]
        baseline["groups"][language]["cps_p90"]["value"] = value * 2.0
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    code, report = _evaluate(
        corpus_path, tmp_path / "r.json", "--baseline", str(baseline_path), "--check"
    )
    assert code == cc.EXIT_OK
    assert all(
        r["status"] in {"pass", "disabled"}
        for r in report["gate_results"]
        if r["mode"] == "blocking"
    )


def test_baseline_digest_mismatch_exits_2(corpus_path: Path, tmp_path: Path) -> None:
    baseline_path = _record(corpus_path, tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["corpus_digest"] = "f" * 64
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert (
        run_cli(
            [
                "evaluate",
                "--corpus",
                str(corpus_path),
                "--baseline",
                str(baseline_path),
                "--check",
            ]
        )
        == cc.EXIT_INVALID
    )


def test_baseline_from_another_ja_lens_exits_2(
    corpus_path: Path, tmp_path: Path
) -> None:
    baseline_path = _record(corpus_path, tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    lens = baseline["metric_definition"]["forbidden_end"]["ja_tail_lens"]
    if lens["id"] == calib.JA_TAIL_LENS_LEVEL2:
        lens.update(
            {
                "id": calib.JA_TAIL_LENS_LEVEL1,
                "source": "kinsoku.line_end_penalty",
                "provider": None,
                "provider_version": None,
                "dictionary": None,
                "missing_offset_fallback": None,
            }
        )
    else:
        lens.update(
            {
                "id": calib.JA_TAIL_LENS_LEVEL2,
                "source": "kinsoku.ja_pos_end_penalties",
                "provider": "fugashi-unidic",
                "provider_version": "different",
                "dictionary": "different",
                "missing_offset_fallback": calib.JA_TAIL_LENS_LEVEL1,
            }
        )
    baseline["metric_definition_digest"] = cc.canonical_digest(
        baseline["metric_definition"]
    )
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    assert (
        run_cli(
            [
                "evaluate",
                "--corpus",
                str(corpus_path),
                "--baseline",
                str(baseline_path),
                "--check",
            ]
        )
        == cc.EXIT_INVALID
    )


def test_corpus_edit_invalidates_the_baseline(
    corpus_path: Path, tmp_path: Path
) -> None:
    """Editing a case changes the digest, so yesterday's numbers stop applying."""
    baseline_path = _record(corpus_path, tmp_path)
    case_path = corpus_path.parent / "cases" / "en-01.json"
    doc = json.loads(case_path.read_text(encoding="utf-8"))
    doc["description"] = "edited"
    case_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert (
        run_cli(
            [
                "evaluate",
                "--corpus",
                str(corpus_path),
                "--baseline",
                str(baseline_path),
                "--check",
            ]
        )
        == cc.EXIT_INVALID
    )


def test_baseline_environment_drift_exits_2(corpus_path: Path, tmp_path: Path) -> None:
    baseline_path = _record(corpus_path, tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["environment"]["dependencies"]["jieba"] = "0.0.0-not-installed"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    argv = [
        "evaluate",
        "--corpus",
        str(corpus_path),
        "--baseline",
        str(baseline_path),
        "--check",
    ]
    assert run_cli(argv) == cc.EXIT_INVALID
    assert run_cli([*argv, "--allow-environment-drift"]) == cc.EXIT_OK


def test_record_baseline_refuses_a_partial_report(
    corpus_path: Path, tmp_path: Path
) -> None:
    """A single-case run must never become the corpus baseline."""
    report_path = tmp_path / "partial.json"
    code, report = _evaluate(corpus_path, report_path, "--case", "en-01")
    assert code == cc.EXIT_OK
    assert report["partial"] is True
    assert report["gate_results"] == []
    assert (
        run_cli(
            [
                "record-baseline",
                "--corpus",
                str(corpus_path),
                "--report",
                str(report_path),
                "--output",
                str(tmp_path / "baseline.json"),
            ]
        )
        == cc.EXIT_INVALID
    )


def test_record_baseline_refuses_a_stale_report(
    corpus_path: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"
    _evaluate(corpus_path, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["corpus_digest"] = "a" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert (
        run_cli(
            [
                "record-baseline",
                "--corpus",
                str(corpus_path),
                "--report",
                str(report_path),
                "--output",
                str(tmp_path / "baseline.json"),
            ]
        )
        == cc.EXIT_INVALID
    )


def test_unknown_case_filter_is_invalid(corpus_path: Path) -> None:
    assert (
        run_cli(["evaluate", "--corpus", str(corpus_path), "--case", "zh-99"])
        == cc.EXIT_INVALID
    )


# --------------------------------------------------------------------------- #
# Process boundary and wiring
# --------------------------------------------------------------------------- #


def _run_process(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "calib_segmentation.py"), *argv],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
        check=False,
    )


def test_process_exit_codes_and_machine_summary(
    corpus_path: Path, tmp_path: Path
) -> None:
    """CI reads the exit code and the last stdout line; both are the contract."""
    ok = _run_process(["validate-corpus", "--corpus", str(corpus_path)])
    assert ok.returncode == cc.EXIT_OK, ok.stderr
    last = ok.stdout.strip().splitlines()[-1]
    assert last.startswith("QUALITY segmentation status=pass cases=20 ")

    broken = tmp_path / "broken.json"
    broken.write_text('{"schema_version": 1}', encoding="utf-8")
    bad = _run_process(["validate-corpus", "--corpus", str(broken)])
    assert bad.returncode == cc.EXIT_INVALID


def test_compare_video_dir_measures_private_siblings_without_ffmpeg(
    tmp_path: Path,
) -> None:
    """The legacy lane runs the same four metrics over a media directory.

    It reads sibling JSON only: no subtitle track is extracted and no ASS is
    parsed, so an empty placeholder file is enough to stand in for the media.
    """
    (tmp_path / "s01e01.mkv").write_bytes(b"")
    units = spaced_units(_WORDS[2].split(), dur=0.6)
    (tmp_path / "s01e01.json").write_text(
        json.dumps(
            {
                "language": "english",
                "word_segments": units,
                "vad_speech": [[units[0]["start"], units[-1]["end"]]],
            }
        ),
        encoding="utf-8",
    )
    assert run_cli(["compare-video-dir", str(tmp_path)]) == cc.EXIT_OK


def test_compare_video_dir_without_siblings_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "s01e01.mkv").write_bytes(b"")
    assert run_cli(["compare-video-dir", str(tmp_path)]) == cc.EXIT_INVALID


def test_makefile_exposes_the_quality_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "quality-segmentation:" in makefile
    assert "quality-record-segmentation:" in makefile
    assert "calib_segmentation.py evaluate" in makefile


def test_ci_runs_the_gate_armed_and_never_records(tmp_path: Path) -> None:
    """The quality job is armed (soak ended 2026-08-26): it may block a PR,
    and it must still never rewrite the baseline — recording stays a
    deliberate human act. The job also pins the baseline's python minor so
    environment_drift does not refuse every CI run.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "continue-on-error" not in workflow
    assert "make quality-segmentation" in workflow
    assert "--python 3.13" in workflow
    assert "record-baseline" not in workflow
    assert "quality-record-segmentation" not in workflow


def test_no_subtitle_parser_survives_in_the_ruler() -> None:
    """The ASS parser and the commercial-track column belong to the alignment ruler.

    A release subtitle is same-language cue truth for *alignment*; it is not a
    reference for segmentation quality, and a second parser next to
    ``voxweave.subformats`` was a maintenance liability besides.
    """
    source = (REPO_ROOT / "scripts" / "calib_segmentation.py").read_text(
        encoding="utf-8"
    )
    assert "Dialogue:" not in source
    assert "0:s:0" not in source
    assert "ffmpeg" not in source
