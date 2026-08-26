"""``pipeline.segment_document``: the pure segmentation replay entry.

Three properties matter for a function that production and offline calibration
both call:

* purity -- caller inputs are never mutated and nothing touches the filesystem,
* equivalence -- replaying a sibling JSON through :func:`pipeline.split` yields
  exactly what rendering ``segment_document``'s result directly yields,
* determinism -- the same inputs produce the same cues, thresholds and
  diagnostics on every call.

Model-free: no ASR, no diarization, no shot detection. All fixtures are
synthetic word streams in the persisted ``word_segments`` shape.
"""

import copy
import json
from pathlib import Path

from voxweave import pipeline, realign

# --- fixtures ---------------------------------------------------------------

# 1) plain English dialogue, no optional context at all.
PLAIN_UNITS = [
    {"text": "Where", "start": 0.0, "end": 0.4},
    {"text": "did", "start": 0.5, "end": 0.8},
    {"text": "you", "start": 0.9, "end": 1.2},
    {"text": "go", "start": 1.4, "end": 2.0},
    {"text": "Nowhere", "start": 2.4, "end": 3.0},
    {"text": "special", "start": 3.1, "end": 3.6},
]
PLAIN_VAD = [(0.0, 2.0), (2.4, 3.6)]

# 2) same exchange with two speakers and cuts right after each speaker's last
#    word: exercises lyric marking, speaker formatting and the post-format
#    re-snap in one pass chain.
TURNS = [(0.0, 2.2, "SPEAKER_00"), (2.3, 3.8, "SPEAKER_01")]
SHOTS = [2.3, 3.9]

# 3) Chinese, character-grained units over a sung span: lyric flags must survive
#    into the rendered music-note wrap.
ZH_UNITS = [
    {"text": ch, "start": 0.3 * i, "end": 0.3 * i + 0.25}
    for i, ch in enumerate("今天的天气真好。我们一起出去走走吧。")
]
ZH_SING = [(0.0, 2.5)]
ZH_VAD = [(0.0, 5.4)]


def _case_plain() -> dict:
    return {"language": "en", "word_segments": copy.deepcopy(PLAIN_UNITS)}


def _case_speakers() -> dict:
    return {
        "language": "en",
        "word_segments": copy.deepcopy(PLAIN_UNITS),
        "vad_speech": [list(span) for span in PLAIN_VAD],
        "shot_changes": list(SHOTS),
        "speaker_turns": [[s, e, label] for s, e, label in TURNS],
    }


def _case_lyrics() -> dict:
    return {
        "language": "zh",
        "word_segments": copy.deepcopy(ZH_UNITS),
        "vad_speech": [list(span) for span in ZH_VAD],
        "sing_spans": [list(span) for span in ZH_SING],
    }


CASES = {
    "plain": _case_plain,
    "speakers": _case_speakers,
    "lyrics": _case_lyrics,
}


def _segment_case(case: dict, **kwargs) -> pipeline.SegmentationResult:
    """Run segment_document over a sibling-JSON-shaped case, split's way."""
    return pipeline.segment_document(
        language=case["language"],
        word_segments=case["word_segments"],
        vad_speech=pipeline._spans_in(case.get("vad_speech")),
        shot_changes=[float(t) for t in case.get("shot_changes") or []] or None,
        sing_spans=pipeline._spans_in(case.get("sing_spans")),
        speaker_turns=pipeline._turns_in(case.get("speaker_turns")),
        **kwargs,
    )


# Raw acoustic anchors are in-memory-only cue state: ``_write_siblings`` projects
# them out so the persisted ``segments[]`` keeps its legacy shape byte for byte.
SPEECH_KEYS = ("speech_start", "speech_end")


def _projected(cues: list) -> list[dict]:
    """The cue stream as ``_write_siblings`` persists it (anchors dropped)."""
    return [{k: v for k, v in cue.items() if k not in SPEECH_KEYS} for cue in cues]


def _render(result: pipeline.SegmentationResult) -> str:
    """Reproduce the VTT body _write_siblings writes for these cues."""
    return realign.render_cues(
        [
            (c.get("start"), c.get("end"), pipeline.lyric_display_text(c))
            for c in result.cues
        ]
    )


# --- purity -----------------------------------------------------------------


def test_inputs_are_not_mutated():
    """Every sequence the caller owns must come back untouched (deep copies in)."""
    for name, build in CASES.items():
        case = build()
        units = case["word_segments"]
        vad = pipeline._spans_in(case.get("vad_speech"))
        shots = [float(t) for t in case.get("shot_changes") or []] or None
        sings = pipeline._spans_in(case.get("sing_spans"))
        turns = pipeline._turns_in(case.get("speaker_turns"))
        before = copy.deepcopy((units, vad, shots, sings, turns))

        pipeline.segment_document(
            language=case["language"],
            word_segments=units,
            vad_speech=vad,
            shot_changes=shots,
            sing_spans=sings,
            speaker_turns=turns,
        )

        assert (units, vad, shots, sings, turns) == before, f"{name}: input mutated"


def test_returned_units_are_detached_from_the_input():
    units = copy.deepcopy(PLAIN_UNITS)
    result = pipeline.segment_document(language="en", word_segments=units)
    assert [dict(u) for u in result.units] == units
    result.units[0]["text"] = "MUTATED"
    assert units[0]["text"] == "Where"


def test_writes_no_files(tmp_path: Path, monkeypatch):
    """Pure means pure: replaying a document creates nothing on disk."""
    monkeypatch.chdir(tmp_path)
    for build in CASES.values():
        _segment_case(build())
    assert list(tmp_path.iterdir()) == []


# --- equivalence with the production replay path ----------------------------


def test_split_replay_matches_direct_segmentation(tmp_path: Path):
    """pipeline.split's sibling output == rendering segment_document directly."""
    for name, build in CASES.items():
        case = build()
        json_path = tmp_path / f"{name}.json"
        json_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")

        # split repairs a stale persisted language before segmenting; these
        # fixtures are already labelled correctly, so the replay input it hands
        # to segment_document is the case as written.
        iso, reconciled = pipeline._reconcile_word_segment_language(
            case["language"], case["word_segments"]
        )
        assert iso == case["language"]
        assert reconciled == case["word_segments"]

        vtt_path = pipeline.split(json_path)
        result = _segment_case(build())

        assert vtt_path.read_text(encoding="utf-8") == _render(result), name
        written = json.loads(json_path.read_text(encoding="utf-8"))
        assert written["segments"] == json.loads(json.dumps(_projected(result.cues))), (
            name
        )
        assert not any(
            key in segment for segment in written["segments"] for key in SPEECH_KEYS
        ), name
        assert written["word_segments"] == json.loads(json.dumps(result.units)), name
        assert written["language"] == result.language, name


def test_split_replay_matches_direct_segmentation_with_layout_overrides(
    tmp_path: Path,
):
    """A --max-line-length override reaches both the layout and the formatter."""
    case = _case_speakers()
    json_path = tmp_path / "narrow.json"
    json_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")

    vtt_path = pipeline.split(json_path, max_line_length=12)
    result = _segment_case(_case_speakers(), smart_split_kwargs={"max_line_length": 12})

    assert vtt_path.read_text(encoding="utf-8") == _render(result)


def test_lyric_flags_reach_the_rendered_music_notes(tmp_path: Path):
    """Guards the lyric fixture: it must actually exercise the ♪ display path."""
    result = _segment_case(_case_lyrics())
    assert result.diagnostics["lyric_cue_count"] >= 1
    assert "♪" in _render(result)


def test_speaker_fixture_exercises_formatting_and_resnap():
    """Guards the speaker fixture: formatting and the second snap both run."""
    result = _segment_case(_case_speakers())
    assert result.diagnostics["speaker_formatted"] is True
    assert result.diagnostics["shot_resnapped"] is True
    assert any("-" in c["text"] for c in result.cues)


# --- determinism ------------------------------------------------------------


def test_repeated_calls_are_identical():
    for name, build in CASES.items():
        first = _segment_case(build())
        second = _segment_case(build())
        assert json.dumps(first.cues, ensure_ascii=False) == json.dumps(
            second.cues, ensure_ascii=False
        ), name
        assert first.units == second.units, name
        assert first.language == second.language, name
        assert first.thresholds_used == second.thresholds_used, name
        assert first.diagnostics == second.diagnostics, name


def test_same_inputs_reused_across_calls_stay_stable():
    """Calling twice with the *same* input objects must not drift (no in-place
    state carried over from the first pass)."""
    case = _case_speakers()
    units = case["word_segments"]
    vad = pipeline._spans_in(case["vad_speech"])
    shots = [float(t) for t in case["shot_changes"]]
    turns = pipeline._turns_in(case["speaker_turns"])
    kwargs = {
        "language": "en",
        "word_segments": units,
        "vad_speech": vad,
        "shot_changes": shots,
        "speaker_turns": turns,
    }

    first = pipeline.segment_document(**kwargs)
    second = pipeline.segment_document(**kwargs)

    assert json.dumps(first.cues) == json.dumps(second.cues)
    assert first.diagnostics == second.diagnostics


# --- resolved context -------------------------------------------------------


def test_thresholds_default_to_the_language_profile_and_can_be_overridden():
    from voxweave.config import gap_thresholds

    default = pipeline.segment_document(language="en", word_segments=PLAIN_UNITS)
    assert default.thresholds_used == gap_thresholds("en")

    override = dict(gap_thresholds("en"))
    override["clause_ms"] = 10_000
    forced = pipeline.segment_document(
        language="en", word_segments=PLAIN_UNITS, thresholds=override
    )
    assert forced.thresholds_used == override


def test_absent_optional_context_is_reported_as_empty():
    result = pipeline.segment_document(language="en", word_segments=PLAIN_UNITS)
    assert result.diagnostics["speech_span_count"] == 0
    assert result.diagnostics["shot_change_count"] == 0
    assert result.diagnostics["sing_span_count"] == 0
    assert result.diagnostics["speaker_turn_count"] == 0
    assert result.diagnostics["speaker_formatted"] is False
    assert result.diagnostics["shot_resnapped"] is False
    assert result.diagnostics["semantic_engine"] is False
    assert result.diagnostics["cue_count"] == len(result.cues)
    assert result.diagnostics["unit_count"] == len(result.units)


def test_accepts_tuples_for_every_span_input():
    """Calibration cases arrive as immutable sequences; they must not need lists."""
    result = pipeline.segment_document(
        language="en",
        word_segments=tuple(PLAIN_UNITS),
        vad_speech=tuple(PLAIN_VAD),
        shot_changes=tuple(SHOTS),
        sing_spans=((0.0, 1.0),),
        speaker_turns=tuple(TURNS),
    )
    assert result.cues
    assert result.diagnostics["shot_change_count"] == len(SHOTS)
