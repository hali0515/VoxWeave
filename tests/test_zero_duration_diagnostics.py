"""Zero-duration repair accounting for the VAD positioning pass.

The upstream Qwen NAR aligner emits word spans with ``start == end``
(QwenLM/Qwen3-ASR#197); ``position_units_with_vad`` repairs them. These tests pin the
diagnostics that make the repair visible: exact counters on synthetic unit streams with
known collapse counts, the ``raw == repaired + residual`` identity, and the guarantee
that attaching an accumulator changes nothing about the result.
"""

from __future__ import annotations

import copy
import json

import pytest

from voxweave import realign
from voxweave.debug import DebugSink, FileDebugSink

ZeroDurationDiagnostics = realign.ZeroDurationDiagnostics


# --------------------------------------------------------------------------- #
# Fixtures: unit streams with known zero-duration counts
# --------------------------------------------------------------------------- #
# One collapsed unit with isolated speech in the gap -> snap relocates it; the
# following inflated unit is carved back to the speech offset.
REPAIRABLE = (
    [
        {"text": "な", "start": 11.9, "end": 12.06},
        {"text": "はい", "start": 12.06, "end": 12.06},
        {"text": "Oh", "start": 13.8, "end": 16.0},
    ],
    [(11.0, 12.06), (13.0, 13.6), (14.0, 14.5)],
)

# Collapsed unit whose gap holds no isolated VAD segment -> nothing to snap to.
UNREPAIRABLE = (
    [
        {"text": "a", "start": 1.0, "end": 2.0},
        {"text": "b", "start": 2.0, "end": 2.0},
        {"text": "c", "start": 4.0, "end": 5.0},
    ],
    [(1.0, 2.0), (4.0, 5.0)],
)

# Nine collapsed units in a row: an ASR repetition wall, above max_run -> untouched.
LONG_RUN = (
    [{"text": "lead", "start": 2.0, "end": 3.0}]
    + [{"text": f"w{k}", "start": 3.0, "end": 3.0} for k in range(9)]
    + [{"text": "tail", "start": 6.0, "end": 7.0}],
    [(2.0, 3.0), (6.0, 7.0)],
)

# Point-timestamp run that drifted past the speech offset -> pulled back, no zeros.
STRANDED = (
    [
        {"text": "行", "start": 5.2, "end": 5.3},
        {"text": "く", "start": 5.3, "end": 5.4},
    ],
    [(4.0, 5.0)],
)

# Punctuation promoted by reinject_punct is zero-width by design, not a defect.
PUNCT_ONLY = (
    [
        {"text": "そう", "start": 1.0, "end": 1.5},
        {"text": "。", "start": 1.5, "end": 1.5},
        {"text": " ", "start": 1.5, "end": 1.5},
    ],
    [(1.0, 1.5)],
)

ALL_FIXTURES = {
    "repairable": REPAIRABLE,
    "unrepairable": UNREPAIRABLE,
    "long_run": LONG_RUN,
    "stranded": STRANDED,
    "punct_only": PUNCT_ONLY,
}


def _run(fixture, diagnostics=None):
    units, vad = fixture
    return realign.position_units_with_vad(
        copy.deepcopy(units), list(vad), diagnostics=diagnostics
    )


# --------------------------------------------------------------------------- #
# Exact counters
# --------------------------------------------------------------------------- #
def test_repairable_collapse_counted_per_step():
    diag = ZeroDurationDiagnostics()
    out = _run(REPAIRABLE, diag)

    assert (out[1]["start"], out[1]["end"]) == (13.0, 13.6)  # snapped
    assert out[2]["end"] == 14.5  # carved
    assert diag.calls == 1
    assert (diag.units_seen, diag.lexical_units) == (3, 3)
    assert diag.raw_exact_zero == 1
    assert diag.repaired_exact_zero == 1
    assert diag.residual_exact_zero == 0
    assert diag.raw_collapse_candidates == 1
    assert diag.repaired_collapse_candidates == 1
    assert diag.residual_collapse_candidates == 0
    assert diag.changed_by_step == {
        "snap_zero_duration": 1,
        "snap_silence_stranded": 0,
        "carve_over_silence": 1,
    }
    assert diag.run_outcomes["relocated"] == 1
    assert diag.run_length_histogram() == {"1": 1, "2-3": 0, "4-8": 0, ">8": 0}
    # midpoint moved 12.06 -> 13.3, duration 0 -> 0.6
    assert diag.midpoint_shifts == pytest.approx([1.24])
    assert diag.duration_increases == pytest.approx([0.6])


def test_collapse_without_speech_in_gap_stays_residual():
    diag = ZeroDurationDiagnostics()
    out = _run(UNREPAIRABLE, diag)

    assert out[1]["start"] == out[1]["end"] == 2.0  # unchanged
    assert diag.raw_exact_zero == 1
    assert diag.repaired_exact_zero == 0
    assert diag.residual_exact_zero == 1
    assert diag.residual_collapse_candidates == 1
    assert diag.run_outcomes["no_speech_gap"] == 1
    assert diag.changed_by_step["snap_zero_duration"] == 0
    assert diag.midpoint_shifts == [] and diag.duration_increases == []


def test_run_above_max_run_is_left_alone_and_flagged():
    diag = ZeroDurationDiagnostics()
    out = _run(LONG_RUN, diag)

    assert all(u["start"] == u["end"] == 3.0 for u in out[1:10])
    assert diag.raw_exact_zero == 9
    assert diag.repaired_exact_zero == 0
    assert diag.residual_exact_zero == 9
    assert diag.run_outcomes == {
        "relocated": 0,
        "over_max_run": 1,
        "no_lexical_unit": 0,
        "no_speech_gap": 0,
    }
    # ">8" is the loud bucket: it is exactly what snap_zero_duration_units refuses.
    assert diag.run_length_histogram() == {"1": 0, "2-3": 0, "4-8": 0, ">8": 1}


def test_silence_stranded_rescue_is_counted():
    diag = ZeroDurationDiagnostics()
    out = _run(STRANDED, diag)

    assert out[-1]["end"] == pytest.approx(5.0)  # pulled back to the speech offset
    assert diag.changed_by_step["snap_silence_stranded"] == 2
    assert diag.raw_exact_zero == 0
    assert diag.repaired_exact_zero == diag.residual_exact_zero == 0


def test_punctuation_is_not_a_denominator():
    diag = ZeroDurationDiagnostics()
    _run(PUNCT_ONLY, diag)

    assert diag.units_seen == 3
    assert diag.lexical_units == 1  # "。" and " " carry no text
    assert diag.raw_exact_zero == 0
    assert diag.raw_collapse_candidates == 0


def test_counters_accumulate_across_calls():
    diag = ZeroDurationDiagnostics()
    _run(REPAIRABLE, diag)
    _run(UNREPAIRABLE, diag)

    assert diag.calls == 2
    assert diag.units_seen == 6
    assert diag.raw_exact_zero == 2
    assert diag.repaired_exact_zero == 1
    assert diag.residual_exact_zero == 1
    assert diag.accounting_balanced


def test_no_vad_leaves_collapse_unrepaired():
    diag = ZeroDurationDiagnostics()
    units = [{"text": "は", "start": 1.0, "end": 1.0}]
    assert realign.position_units_with_vad(units, [], diagnostics=diag) == units
    assert (
        diag.raw_exact_zero,
        diag.repaired_exact_zero,
        diag.residual_exact_zero,
    ) == (
        1,
        0,
        1,
    )


# --------------------------------------------------------------------------- #
# Accounting identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_accounting_identity_holds(name):
    diag = ZeroDurationDiagnostics()
    _run(ALL_FIXTURES[name], diag)
    assert diag.raw_exact_zero == diag.repaired_exact_zero + diag.residual_exact_zero
    assert diag.accounting_balanced


def test_accounting_identity_holds_for_the_whole_corpus_in_one_accumulator():
    diag = ZeroDurationDiagnostics()
    for fixture in ALL_FIXTURES.values():
        _run(fixture, diag)
    assert diag.raw_exact_zero == 11  # 1 + 1 + 9 + 0 + 0
    assert diag.raw_exact_zero == diag.repaired_exact_zero + diag.residual_exact_zero
    assert diag.desynced_calls == 0


def test_index_desync_counts_everything_as_residual():
    # Defensive path: a step that stopped preserving indices makes repair attribution
    # meaningless, so nothing may be claimed as repaired.
    diag = ZeroDurationDiagnostics()
    units, _ = UNREPAIRABLE
    diag.begin_call(copy.deepcopy(units))
    diag.end_call([{"text": "a", "start": 1.0, "end": 2.0}])

    assert diag.desynced_calls == 1
    assert diag.raw_exact_zero == diag.residual_exact_zero == 1
    assert diag.repaired_exact_zero == 0
    assert diag.accounting_balanced


def test_unknown_step_and_outcome_are_rejected():
    diag = ZeroDurationDiagnostics()
    with pytest.raises(ValueError):
        diag.record_change(0, "carve_over_speech")
    with pytest.raises(ValueError):
        diag.record_zero_run(1, "fixed")


def test_run_length_histogram_buckets():
    diag = ZeroDurationDiagnostics()
    for length in (1, 3, 4, 8, 9, 40):
        diag.record_zero_run(length, "over_max_run")
    assert diag.run_length_histogram() == {"1": 1, "2-3": 1, "4-8": 2, ">8": 2}


# --------------------------------------------------------------------------- #
# diagnostics=None keeps the production path byte-identical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_accumulator_does_not_change_results(name):
    fixture = ALL_FIXTURES[name]
    assert _run(fixture, ZeroDurationDiagnostics()) == _run(fixture, None)


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_individual_steps_ignore_the_accumulator(name):
    units, vad = ALL_FIXTURES[name]
    diag = ZeroDurationDiagnostics()
    for step in (
        realign.snap_zero_duration_units,
        realign.snap_silence_stranded_units,
        realign.carve_units_over_silence,
    ):
        assert step(copy.deepcopy(units), list(vad), diagnostics=diag) == step(
            copy.deepcopy(units), list(vad)
        )


def test_pass_does_not_mutate_its_input():
    units, vad = REPAIRABLE
    supplied = copy.deepcopy(units)
    realign.position_units_with_vad(
        supplied, list(vad), diagnostics=ZeroDurationDiagnostics()
    )
    assert supplied == units


# --------------------------------------------------------------------------- #
# debug sink wiring
# --------------------------------------------------------------------------- #
def test_noop_sink_positions_units_without_instrumentation(tmp_path):
    units, vad = REPAIRABLE
    out = DebugSink().position_units(
        copy.deepcopy(units), list(vad), language="Japanese"
    )
    assert out == _run(REPAIRABLE, None)
    assert list(tmp_path.iterdir()) == []


def _sink(tmp_path) -> FileDebugSink:
    return FileDebugSink("clip", base=tmp_path / "debug")


def test_file_sink_writes_snapshots_and_health(tmp_path):
    sink = _sink(tmp_path)
    units, vad = REPAIRABLE
    out = sink.position_units(copy.deepcopy(units), list(vad), language="Japanese")
    assert out == _run(REPAIRABLE, None)  # same result as production

    root = tmp_path / "debug" / "clip"
    pre = json.loads((root / "03-pre-position.units.json").read_text())
    post = json.loads((root / "04-post-position.units.json").read_text())
    assert [u["unit_id"] for u in pre] == [u["unit_id"] for u in post]
    assert pre[1]["start"] == pre[1]["end"] == 12.06  # collapsed before the pass
    assert post[1]["end"] > post[1]["start"]  # repaired after it

    health = json.loads((root / "alignment-health.json").read_text())
    ja = health["languages"]["ja"]
    assert health["totals"]["accounting_balanced"] is True
    assert ja["counters"]["raw_exact_zero"] == 1
    assert ja["counters"]["repaired_exact_zero"] == 1
    assert ja["rates"]["raw_exact_zero_unit_rate"] == {
        "bad": 1,
        "eligible": 3,
        "value": pytest.approx(1 / 3),
    }
    assert ja["rates"]["residual_exact_zero_unit_rate"]["bad"] == 0
    assert ja["rates"]["repaired_exact_zero_rate"]["value"] == 1.0


def test_health_rate_value_is_null_when_nothing_is_eligible(tmp_path):
    sink = _sink(tmp_path)
    sink.position_units([], [], language="English")
    rates = sink.health_report()["languages"]["en"]["rates"]
    assert rates["raw_exact_zero_unit_rate"] == {"bad": 0, "eligible": 0, "value": None}
    assert rates["chunk_affected_exact_zero_rate"]["value"] is None


def test_health_groups_by_language(tmp_path):
    sink = _sink(tmp_path)
    sink.position_units(
        copy.deepcopy(REPAIRABLE[0]), list(REPAIRABLE[1]), language="ja"
    )
    sink.position_units(
        copy.deepcopy(UNREPAIRABLE[0]), list(UNREPAIRABLE[1]), language="English"
    )
    languages = sink.health_report()["languages"]

    assert sorted(languages) == ["en", "ja"]
    assert languages["ja"]["counters"]["repaired_exact_zero"] == 1
    assert languages["en"]["counters"]["residual_exact_zero"] == 1
    assert sink.health_report()["totals"] == {
        "units_seen": 6,
        "lexical_units": 6,
        "raw_exact_zero": 2,
        "repaired_exact_zero": 1,
        "residual_exact_zero": 1,
        "chunks_with_units": 0,
        "chunks_with_exact_zero": 0,
        "accounting_balanced": True,
    }


def test_chunk_level_affected_rate(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFfake")
    sink = _sink(tmp_path)
    common = {"wav": src, "start": 0.0, "end": 1.0, "raw": "r", "text": "t"}
    sink.chunk(
        0, lang="Japanese", units=[{"text": "あ", "start": 0.0, "end": 0.2}], **common
    )
    sink.chunk(
        1, lang="Japanese", units=[{"text": "い", "start": 1.0, "end": 1.0}], **common
    )
    sink.chunk(
        2, lang="Japanese", units=[{"text": "。", "start": 2.0, "end": 2.0}], **common
    )
    sink.chunk(3, lang="Japanese", units=None, **common)  # empty ASR: not eligible
    sink.position_units([], [], language="Japanese")

    ja = sink.health_report()["languages"]["ja"]
    assert ja["chunks"] == {
        "with_units": 3,
        "with_exact_zero": 1,  # only the collapsed い; the "。" is by-design zero-width
        "exact_zero_units": 1,
    }
    assert ja["rates"]["chunk_affected_exact_zero_rate"]["value"] == pytest.approx(
        1 / 3
    )


def test_undetected_chunk_language_falls_back_to_the_pass_language(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"RIFFfake")
    sink = _sink(tmp_path)
    sink.chunk(
        0,
        wav=src,
        start=0.0,
        end=1.0,
        raw="r",
        text="t",
        lang=None,
        units=[{"text": "い", "start": 1.0, "end": 1.0}],
    )
    sink.position_units([], [], language="Japanese")

    languages = sink.health_report()["languages"]
    assert sorted(languages) == ["ja"]
    assert languages["ja"]["chunks"]["with_exact_zero"] == 1
