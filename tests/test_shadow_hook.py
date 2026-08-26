"""The v2 shadow hook in ``pipeline.segment_document``.

A shadow lane earns its place only if it is invisible. Every test here defends
one half of that claim:

* **invisible when off** -- the flag is read before anything else and no v2
  module reaches ``sys.modules``, so a production build pays one environment
  read for a feature it does not use;
* **invisible when on** -- the shipped cues, the returned units and the
  persisted sibling bytes are identical with the flag on, including on a
  document whose provider ledger is non-empty (the case where a naive shadow
  would double the manifest's degradation counts);
* **honest about itself** -- the artifact is deterministic, its two degradation
  ledgers are kept apart by origin, and an optimizer that raises degrades to a
  typed ``error`` block rather than taking the run down with it.
"""

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from voxweave import pipeline
from voxweave.config import gap_thresholds
from voxweave.core import boundary_lattice, boundary_v2
from voxweave.core import providers
from voxweave.core.providers import note_degraded

FLAG = pipeline.SEG_V2_SHADOW_ENV

# --- fixtures ---------------------------------------------------------------

EN_UNITS = [
    {"text": "Where", "start": 0.0, "end": 0.4},
    {"text": "did", "start": 0.5, "end": 0.8},
    {"text": "you", "start": 0.9, "end": 1.2},
    {"text": "go", "start": 1.4, "end": 2.0},
    {"text": "Nowhere", "start": 2.4, "end": 3.0},
    {"text": "special", "start": 3.1, "end": 3.6},
]
EN_VAD = [(0.0, 2.0), (2.4, 3.6)]
EN_TURNS = [(0.0, 2.2, "SPEAKER_00"), (2.3, 3.8, "SPEAKER_01")]
EN_SHOTS = [2.3, 3.9]

ZH_UNITS = [
    {"text": ch, "start": 0.3 * i, "end": 0.3 * i + 0.25}
    for i, ch in enumerate("今天的天气真好。我们一起出去走走吧。")
]
YUE_UNITS = [
    {"text": ch, "start": 0.3 * i, "end": 0.3 * i + 0.25}
    for i, ch in enumerate("今日天氣真好。我哋一齊出去行下啦。")
]


def _case_plain() -> dict:
    return {"language": "en", "word_segments": copy.deepcopy(EN_UNITS)}


def _case_speakers() -> dict:
    return {
        "language": "en",
        "word_segments": copy.deepcopy(EN_UNITS),
        "vad_speech": [list(span) for span in EN_VAD],
        "shot_changes": list(EN_SHOTS),
        "speaker_turns": [[s, e, label] for s, e, label in EN_TURNS],
    }


def _case_lyrics() -> dict:
    return {
        "language": "zh",
        "word_segments": copy.deepcopy(ZH_UNITS),
        "vad_speech": [[0.0, 5.4]],
        "sing_spans": [[0.0, 2.5]],
    }


#: Two dialogue runs separated by a silence far past ``vad_skip_ms + 50``, so
#: the lattice raises a robust-silence barrier and the document really does have
#: more than one hard interval (which the fallback-overlap test needs).
TWO_INTERVAL_UNITS = [
    {"text": "Where", "start": 0.0, "end": 0.4},
    {"text": "did", "start": 0.5, "end": 0.8},
    {"text": "you", "start": 0.9, "end": 1.2},
    {"text": "go", "start": 1.4, "end": 2.0},
    {"text": "Nowhere", "start": 5.0, "end": 5.6},
    {"text": "special", "start": 5.7, "end": 6.2},
    {"text": "today", "start": 6.3, "end": 6.9},
]


def _case_two_intervals() -> dict:
    return {
        "language": "en",
        "word_segments": copy.deepcopy(TWO_INTERVAL_UNITS),
        "vad_speech": [[0.0, 2.0], [5.0, 6.9]],
    }


def _case_yue() -> dict:
    """A document whose providers really do degrade -- the ledger pin needs one."""
    return {
        "language": "yue",
        "word_segments": copy.deepcopy(YUE_UNITS),
        "vad_speech": [[0.0, 5.4]],
    }


CASES = {
    "plain": _case_plain,
    "speakers": _case_speakers,
    "lyrics": _case_lyrics,
    "two_intervals": _case_two_intervals,
    "yue": _case_yue,
}


def _segment(case: dict, **kwargs) -> pipeline.SegmentationResult:
    return pipeline.segment_document(
        language=case["language"],
        word_segments=case["word_segments"],
        vad_speech=pipeline._spans_in(case.get("vad_speech")),
        shot_changes=[float(t) for t in case.get("shot_changes") or []] or None,
        sing_spans=pipeline._spans_in(case.get("sing_spans")),
        speaker_turns=pipeline._turns_in(case.get("speaker_turns")),
        **kwargs,
    )


def _canonical(value: object) -> str:
    """Byte-stable JSON: sorted keys, so a dict's build order cannot leak in."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _persisted(case: dict, tmp_path: Path, name: str) -> tuple[bytes, bytes]:
    """Replay a sibling JSON through ``split`` and return (VTT, JSON) bytes."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    json_path = tmp_path / f"{name}.json"
    json_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    vtt_path = pipeline.split(json_path)
    return Path(vtt_path).read_bytes(), json_path.read_bytes()


@pytest.fixture
def shadow_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def shadow_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(FLAG, raising=False)


# --- invisible when off -----------------------------------------------------


def test_flag_off_returns_no_shadow(shadow_off):
    for name, build in CASES.items():
        result = _segment(build())
        assert result.shadow is None, name
        assert result.diagnostics["shadow_v2"] is False, name


@pytest.mark.parametrize("value", ["", "0", "false", "true", "yes", " 1 x"])
def test_only_the_exact_string_one_turns_the_shadow_on(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    """``--no-<flag>`` writes the literal ``"0"``, which is a truthy string."""
    monkeypatch.setenv(FLAG, value)
    assert _segment(_case_plain()).shadow is None


def test_off_path_imports_no_v2_module(tmp_path: Path):
    """A fresh interpreter must not have the optimizer in ``sys.modules``.

    Run out of process on purpose: the rest of the suite imports the v2 modules
    directly, so an in-process assertion would only prove that some earlier test
    got there first.
    """
    script = tmp_path / "off_path.py"
    script.write_text(
        "import json, os, sys\n"
        f"os.environ.pop({FLAG!r}, None)\n"
        "from voxweave import pipeline\n"
        f"units = {EN_UNITS!r}\n"
        "result = pipeline.segment_document(language='en', word_segments=units)\n"
        "leaked = sorted(\n"
        "    name\n"
        "    for name in sys.modules\n"
        "    if name.startswith('voxweave.core.boundary')\n"
        "    or name == 'voxweave.core.partition_check'\n"
        ")\n"
        "print(json.dumps({'shadow': result.shadow, 'leaked': leaked,\n"
        "                  'cues': len(result.cues)}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    )
    report = json.loads(proc.stdout.splitlines()[-1])
    assert report["shadow"] is None
    assert report["cues"] > 0
    assert report["leaked"] == []


# --- invisible when on ------------------------------------------------------


def test_flag_on_leaves_the_returned_stream_identical(
    monkeypatch: pytest.MonkeyPatch,
):
    for name, build in CASES.items():
        monkeypatch.delenv(FLAG, raising=False)
        off = _segment(build())
        monkeypatch.setenv(FLAG, "1")
        on = _segment(build())

        assert on.shadow is not None, name
        assert on.diagnostics["shadow_v2"] is True, name
        assert _canonical(on.cues) == _canonical(off.cues), name
        assert _canonical(on.units) == _canonical(off.units), name
        assert _canonical(on.manifest) == _canonical(off.manifest), name
        assert on.thresholds_used == off.thresholds_used, name
        assert on.language == off.language, name
        # The only diagnostics delta is the boolean the hook itself records.
        assert {k: v for k, v in on.diagnostics.items() if k != "shadow_v2"} == {
            k: v for k, v in off.diagnostics.items() if k != "shadow_v2"
        }, name


def test_flag_on_persists_identical_vtt_and_json_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for name, build in CASES.items():
        monkeypatch.delenv(FLAG, raising=False)
        off_vtt, off_json = _persisted(build(), tmp_path / "off", f"{name}")
        monkeypatch.setenv(FLAG, "1")
        on_vtt, on_json = _persisted(build(), tmp_path / "on", f"{name}")
        assert on_vtt == off_vtt, name
        assert on_json == off_json, name


def test_inputs_are_not_mutated_with_the_shadow_on(shadow_on):
    """The delivery lane runs the real overlays -- on copies, never the inputs."""
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

        assert (units, vad, shots, sings, turns) == before, name


def test_writes_no_files_with_the_shadow_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.chdir(tmp_path)
    for build in CASES.values():
        _segment(build())
    assert list(tmp_path.iterdir()) == []


# --- determinism ------------------------------------------------------------


def test_repeated_runs_produce_identical_artifact_bytes(shadow_on):
    for name, build in CASES.items():
        first = _segment(build()).shadow
        second = _segment(build()).shadow
        assert first is not None and second is not None, name
        assert _canonical(first) == _canonical(second), name


def test_the_artifact_is_json_serializable(shadow_on):
    """It is returned, not written -- but a harness has to be able to write it."""
    for name, build in CASES.items():
        artifact = _segment(build()).shadow
        assert json.loads(_canonical(artifact)) == json.loads(
            json.dumps(artifact, ensure_ascii=False)
        ), name


# --- artifact shape ---------------------------------------------------------


def test_artifact_carries_both_lanes_with_stage_attribution(shadow_on):
    artifact = _segment(_case_speakers()).shadow
    assert artifact is not None
    assert artifact["kind"] == "segmentation-shadow"
    assert artifact["engine_v2"] == boundary_v2.ENGINE_V2

    lanes = artifact["lanes"]
    core = lanes[pipeline.SHADOW_LANE_CORE]
    delivery = lanes[pipeline.SHADOW_LANE_DELIVERY]
    assert core["stage"] == "core"
    assert delivery["stage"] == "legacy-overlay"
    for lane, stage in ((core, "core"), (delivery, "legacy-overlay")):
        for side, origin in (("v1", "v1"), ("v2", "v2")):
            block = lane[side]
            assert block["cue_count"] == len(block["cues"])
            assert block["validator"] is not None, (lane["lane"], side)
            assert block["validator"]["origin"] == origin
            assert block["validator"]["stage"] == stage
            for violation in block["validator"]["violations"]:
                assert violation["origin"] == origin
                assert violation["stage"] == stage
                # AD3-3: only unwaived v2 findings at raw/core drive the exit.
                if origin == "v1" or stage == "legacy-overlay":
                    assert violation["exit_driving"] is False

    # The solver's own partition and the character projection the overlay lane
    # must rely on are independent derivations; they have to agree.
    assert core["v2"]["projection_cross_check"]["agrees"] is True
    assert core["v2"]["projection"] == "solver-partition"
    assert core["agreement"]["identical_cuts"] >= 0


def test_coverage_block_reports_the_c13_fields(shadow_on):
    for name, build in CASES.items():
        artifact = _segment(build()).shadow
        assert artifact is not None
        coverage = artifact["coverage"]
        assert coverage["unit_count"] == len(build()["word_segments"])
        assert coverage["fallback_intervals"] == 0, name
        assert coverage["optimized_unit_ratio"] == 1.0, name
        assert coverage["fallback_unit_ranges"] == [], name
        # No fallback means no overlapping adoption, so the raw-stage validator
        # cannot be double-counting a v1 cue.
        assert coverage["fallback_ranges_overlap"] is False, name
        assert coverage["raw_conservation_trustworthy"] is True, name
        assert artifact["validator"]["raw_duplicate_v1_cues"] is False, name


def test_the_fixture_really_has_two_hard_intervals(shadow_on):
    """Guards the fixture the fallback-overlap test depends on."""
    artifact = _segment(_case_two_intervals()).shadow
    assert artifact is not None
    assert artifact["totals"]["interval_count"] >= 2
    assert artifact["totals"]["barrier_count"] >= 3


def test_overlapping_fallbacks_are_flagged_as_unusable_conservation_evidence(
    shadow_on, monkeypatch: pytest.MonkeyPatch
):
    """Adjacent typed fallbacks can adopt the SAME v1 cue.

    ``AdoptedV1`` adopts complete v1 cues and expands to their span, so two
    neighbouring fallbacks can both own a cue that straddles their shared
    boundary -- and the raw-stage document validator then sees that cue twice.
    That is a property of the fallback contract, not a conservation result, so
    the artifact has to say so where a harness meets it. The public corpus can
    never reach the state (the C13 gate forbids fallbacks outright), so it is
    injected here; only the annotation is under test, not the fabricated totals.
    """
    real = boundary_v2.optimize_document

    def with_overlapping_fallbacks(document, **kwargs):
        solution = real(document, **kwargs)
        assert len(solution.solutions) >= 2
        whole = (0, len(document.units))
        patched = tuple(
            replace(
                item,
                adopted=boundary_v2.AdoptedV1(
                    unit_range=whole,
                    fallback_expansion_units=whole,
                    cues=item.cues,
                    reason="no-path",
                ),
            )
            for item in solution.solutions[:2]
        ) + tuple(solution.solutions[2:])
        return replace(solution, solutions=patched)

    monkeypatch.setattr(boundary_v2, "optimize_document", with_overlapping_fallbacks)
    artifact = _segment(_case_two_intervals()).shadow
    assert artifact is not None
    coverage = artifact["coverage"]
    assert coverage["fallback_unit_ranges"][:2] == [[0, 7], [0, 7]]
    assert coverage["fallback_ranges_overlap"] is True
    assert coverage["raw_conservation_trustworthy"] is False
    assert artifact["validator"]["raw_duplicate_v1_cues"] is True


def test_cue_rows_carry_the_unit_ranges_they_own(shadow_on):
    artifact = _segment(_case_lyrics()).shadow
    assert artifact is not None
    block = artifact["lanes"][pipeline.SHADOW_LANE_CORE]["v2"]
    rows = block["cues"]
    assert rows
    assert rows[0]["unit_range"][0] == 0
    assert rows[-1]["unit_range"][1] == artifact["coverage"]["unit_count"]
    for left, right in zip(rows, rows[1:]):
        assert left["unit_range"][1] == right["unit_range"][0]


def test_post_overlay_lane_runs_the_diarize_formatter(shadow_on):
    """A turn-bearing document must actually exercise the delivery lane."""
    result = _segment(_case_speakers())
    artifact = result.shadow
    assert artifact is not None
    assert result.diagnostics["speaker_formatted"] is True

    core = artifact["lanes"][pipeline.SHADOW_LANE_CORE]["v2"]
    delivery = artifact["lanes"][pipeline.SHADOW_LANE_DELIVERY]["v2"]
    assert delivery["cue_count"] >= core["cue_count"]
    # The formatter either split at the speaker turn or dashed a shared cue;
    # either way the delivery stream is not simply the core stream renamed.
    assert delivery["cue_count"] > core["cue_count"] or any(
        row["text"].startswith("-") for row in delivery["cues"]
    )


def test_lyric_flags_only_appear_in_the_delivery_lane(shadow_on):
    artifact = _segment(_case_lyrics()).shadow
    assert artifact is not None
    core = artifact["lanes"][pipeline.SHADOW_LANE_CORE]["v2"]["cues"]
    delivery = artifact["lanes"][pipeline.SHADOW_LANE_DELIVERY]["v2"]["cues"]
    assert not any(row["lyric"] for row in core)
    assert any(row["lyric"] for row in delivery)


# --- the two degradation ledgers (AD3-5 / AD4-4) ----------------------------


def test_yue_production_ledger_is_the_persisted_manifest_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """AD3-5: the artifact quotes the completed manifest ledger, non-empty."""
    monkeypatch.delenv(FLAG, raising=False)
    off = _segment(_case_yue())
    off_vtt, off_json = _persisted(_case_yue(), tmp_path / "off", "yue")

    monkeypatch.setenv(FLAG, "1")
    on = _segment(_case_yue())
    on_vtt, on_json = _persisted(_case_yue(), tmp_path / "on", "yue")

    assert on.manifest is not None and on.shadow is not None
    assert on.manifest["degraded"], "fixture no longer degrades any provider"
    assert on.shadow["production_degraded"] == on.manifest["degraded"]
    # Quoted, not aliased: a later mutation of the artifact cannot reach the
    # dict the caller is about to persist.
    assert on.shadow["production_degraded"] is not on.manifest["degraded"]
    # The shadow re-tokenizes the same document, so without the nested capture
    # its events would have inflated these counts.
    assert off.manifest is not None
    assert on.manifest["degraded"] == off.manifest["degraded"]
    assert (on_vtt, on_json) == (off_vtt, off_json)


def test_shadow_only_degradation_never_reaches_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """AD4-4(b): a provider that fails only under v2 lands in one ledger only."""
    monkeypatch.delenv(FLAG, raising=False)
    off = _segment(_case_plain())
    off_vtt, off_json = _persisted(_case_plain(), tmp_path / "off", "plain")

    original = boundary_lattice._attach_end_penalties

    def failing(*args, **kwargs):
        note_degraded("pos", "shadow-test-injection")
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary_lattice, "_attach_end_penalties", failing)
    monkeypatch.setenv(FLAG, "1")
    on = _segment(_case_plain())
    on_vtt, on_json = _persisted(_case_plain(), tmp_path / "on", "plain")

    assert on.shadow is not None and on.manifest is not None
    injected = ("pos", "shadow-test-injection")

    def pairs(ledger: list[dict]) -> set[tuple[str, str]]:
        return {(entry["slot"], entry["reason"]) for entry in ledger}

    assert injected in pairs(on.shadow["shadow_degraded"])
    assert injected not in pairs(on.shadow["production_degraded"])
    assert off.manifest is not None
    assert on.manifest["degraded"] == off.manifest["degraded"]
    assert (on_vtt, on_json) == (off_vtt, off_json)


def test_both_ledgers_are_present_and_separate(shadow_on):
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None
    assert artifact["production_degraded"] == []
    assert artifact["shadow_degraded"] == []
    # AD4-4 forbids one merged field: a reader must never have to guess origin.
    assert "degraded" not in artifact


# --- failure containment ----------------------------------------------------


def test_an_optimizer_failure_degrades_to_a_typed_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv(FLAG, raising=False)
    off = _segment(_case_plain())
    off_vtt, off_json = _persisted(_case_plain(), tmp_path / "off", "plain")

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(boundary_v2, "optimize_document", boom)
    monkeypatch.setenv(FLAG, "1")
    on = _segment(_case_plain())
    on_vtt, on_json = _persisted(_case_plain(), tmp_path / "on", "plain")

    assert on.shadow is not None
    assert on.shadow["error"] == {
        "detail": "synthetic optimizer failure",
        "type": "RuntimeError",
    }
    assert on.shadow["shadow_degraded"] == []
    assert "lanes" not in on.shadow
    assert _canonical(on.cues) == _canonical(off.cues)
    assert (on_vtt, on_json) == (off_vtt, off_json)


def test_an_invalid_profile_reports_itself_and_nothing_else(shadow_on):
    """AD3-2: an unusable knob is an invalid measurement, never a degraded one."""
    artifact = _segment(
        _case_plain(), thresholds={**gap_thresholds("en"), "clause_ms": 0.0}
    ).shadow
    assert artifact is not None
    assert [v["key"] for v in artifact["invalid_profile"]] == ["clause_ms"]
    assert artifact["invalid_profile"][0]["reason"] == "not-positive"
    assert "lanes" not in artifact
    assert "totals" not in artifact
    # The ledgers still ride along: the harness reads them to decide the exit.
    assert artifact["shadow_degraded"] == []
    assert artifact["production_degraded"] == []


# ------------------------------------- the exit driver must not be blindable


def test_every_mandatory_validator_stage_is_produced_end_to_end(shadow_on):
    """Bug pin: the core stage had exactly one producer and no consumer test.

    ``artifact["validator"]["core"]`` is the only source of the stage that drives
    the P4 exit, and the harness treated a falsy block as "nothing to report".
    Mutating that single assignment to ``None`` left the whole suite green while
    a real shipped defect could be reintroduced and the gate stayed at exit 0 --
    because every harness test supplied the stage blocks itself. This is the join
    the fixtures could not make: produced here, counted by
    ``calib_segmentation.shadow_violation_counts``.
    """
    for name, build in CASES.items():
        artifact = _segment(build()).shadow
        assert artifact is not None
        validator = artifact["validator"]
        for stage in ("raw", "core", "legacy_overlay"):
            assert validator[stage] is not None, (name, stage)
            assert validator[stage]["origin"] == "v2", (name, stage)
            assert validator[stage]["cue_count"] > 0, (name, stage)
        assert validator["raw"]["stage"] == "raw"
        assert validator["core"]["stage"] == "core"
        assert validator["legacy_overlay"]["stage"] == "legacy-overlay"
        # the two counters the artifact says it cross-checks
        assert (
            validator["interval_hard_violations"]
            == (artifact["totals"]["hard_violations"])
        )
        assert validator["interval_document_agree"] is True


def test_the_projection_cross_check_really_compares_two_derivations(
    shadow_on, monkeypatch: pytest.MonkeyPatch
):
    """Bug pin: hardcoding ``agrees=True`` passed everything and nothing noticed.

    The GATED delivery lane has no solver partition of its own: it reads every
    cue's unit ownership off the character cursor, so a wrong cursor moves
    cps_p90 and the mid-phrase rate without moving a boundary. This is the only
    automated evidence that cursor is sound.
    """
    honest = _segment(_case_two_intervals()).shadow
    assert honest is not None
    core = honest["lanes"][pipeline.SHADOW_LANE_CORE]["v2"]
    assert core["projection_cross_check"]["agrees"] is True
    assert core["partition"], "the fixture must cut somewhere for this to bite"

    real = boundary_v2._document_partition

    def wrong(solutions, unit_count):
        """A partition that is legal-looking but not the one the cues express."""
        cuts = real(solutions, unit_count)
        return tuple(cut for cut in cuts if cut != cuts[0]) if cuts else (1,)

    monkeypatch.setattr(boundary_v2, "_document_partition", wrong)
    artifact = _segment(_case_two_intervals()).shadow
    assert artifact is not None
    check = artifact["lanes"][pipeline.SHADOW_LANE_CORE]["v2"]["projection_cross_check"]
    assert check["agrees"] is False


def test_a_held_chain_waiver_survives_into_the_core_and_overlay_stages(shadow_on):
    """Bug pin: the exemption was granted at ``raw`` and re-reported at ``core``.

    One word sounding for 20 s against the 7 s cap is exactly the evidence the
    C6/AD-4 waiver exists for. The later stages used to be handed no waiver map
    at all, so the same violation came back unwaived and EXIT DRIVING one stage
    on -- a false exit-1 for a condition the design explicitly waives.
    """
    result = pipeline.segment_document(
        language="en",
        word_segments=[
            {"text": "Stop", "start": 0.0, "end": 0.4},
            {"text": "riiiiiight", "start": 0.5, "end": 20.5},
            {"text": "there", "start": 20.6, "end": 21.0},
        ],
        vad_speech=[(0.0, 21.0)],
    )
    artifact = result.shadow
    assert artifact is not None
    assert [w["kind"] for w in artifact["totals"]["waivers"]] == ["held-chain-duration"]
    for stage in ("raw", "core", "legacy_overlay"):
        block = artifact["validator"][stage]
        caps = [v for v in block["violations"] if v["kind"] == "duration-cap"]
        assert caps, stage
        assert all(v["waived"] for v in caps), stage
        assert all(not v["exit_driving"] for v in caps), stage
        assert block["waivers"], stage
        # provenance, not just a label
        assert block["waivers"][0]["unit_ids"] == [1]
        assert block["waivers"][0]["span"] == [0.5, 20.5]


def test_the_artifact_reports_the_resolved_language_providers(shadow_on):
    """AD3-5's other half: which providers the measured run actually resolved."""
    result = _segment(_case_yue())
    artifact = result.shadow
    assert artifact is not None
    assert result.manifest is not None
    assert artifact["providers"] == result.manifest["providers"]
    assert artifact["providers"], "the manifest snapshot is never empty"
    # copied, not shared: a later mutation of one must not move the other
    assert artifact["providers"] is not result.manifest["providers"]


def test_an_unprojectable_v1_stream_is_measured_without_a_v1_reference(
    shadow_on, monkeypatch: pytest.MonkeyPatch
):
    """Adjudication B: never fabricate an empty v1 partition.

    ``_shadow_partition`` returns ``None`` when the character cursor cannot land
    on an atom edge. ``None or ()`` used to collapse that into "v1 chose no
    cuts", which is indistinguishable from the legitimate single-cue answer --
    and then wins the C8 policy comparison outright, because every legal path is
    trivially within margin of a reference that made no cuts.
    """
    monkeypatch.setattr(
        pipeline, "_shadow_partition", lambda *a, **k: (None, "unresolved-at-char-7")
    )
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None
    assert artifact["v1_projection"]["unprojected"] is True
    assert artifact["v1_projection"]["cut_count"] is None
    assert artifact["coverage"]["v1_unprojected"] is True
    assert artifact["v1"] is None
    for block in artifact["intervals"]:
        assert block["selected_is_v1"] is None
        assert block["v1_path_legal"] is None
        assert block["v1_cost_under_v2"] is None
    for lane in (pipeline.SHADOW_LANE_CORE, pipeline.SHADOW_LANE_DELIVERY):
        assert artifact["lanes"][lane]["v1"]["validator"] is None


def test_a_projected_run_records_v1_and_says_so(shadow_on):
    artifact = _segment(_case_plain()).shadow
    assert artifact is not None
    assert artifact["v1_projection"]["unprojected"] is False
    assert artifact["coverage"]["v1_unprojected"] is False
    assert artifact["v1"] is not None
    assert artifact["intervals"][0]["selected_is_v1"] in (True, False)


def test_coverage_reports_the_coarse_granularity_class_separately(shadow_on):
    """Adjudication A: an input-granularity limit is not an optimizer failure.

    It still counts in ``fallback_intervals`` -- the public C13 gate is unmoved
    in either direction -- but the artifact says which class it was, because P5
    resolves this one by splitting below the source unit and P4 cannot.
    """
    sentences = ["これはテストです", "こんにちは世界", "今日はいい天気"] * 2
    bounds = [(1.3 * i, 1.3 * i + 1.1) for i in range(len(sentences))]
    artifact = pipeline.segment_document(
        language="ja",
        word_segments=[
            {"text": s, "start": a, "end": b} for s, (a, b) in zip(sentences, bounds)
        ],
        smart_split_kwargs={"max_line_length": 18, "max_lines": 2},
    ).shadow
    assert artifact is not None
    coverage = artifact["coverage"]
    assert coverage["coarse_granularity_intervals"] >= 1
    assert coverage["fallback_intervals"] >= coverage["coarse_granularity_intervals"]
    reasons = {
        block["infeasible"]["reason"]
        for block in artifact["intervals"]
        if block["infeasible"]
    }
    assert "coarse-granularity" in reasons


def test_a_healthy_word_level_document_reports_no_coarse_intervals(shadow_on):
    for name, build in CASES.items():
        artifact = _segment(build()).shadow
        assert artifact is not None
        assert artifact["coverage"]["coarse_granularity_intervals"] == 0, name
        assert artifact["coverage"]["v1_unprojected"] is False, name


def test_the_shadow_never_claims_productions_once_per_process_warning(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """Bug pin: the measurement could win the latch and silence the shipped run.

    ``providers._WARNED`` is process-global, so whichever context reaches a
    ``(slot, reason)`` pair first emits the single ``log.warning``. The shadow
    re-tokenizes the same document, so a pair only the v2 path can reach would be
    logged by the measurement and never by the run that actually shipped -- an
    operator would see a degradation attributed to a lane that ships nothing, or
    (for a pair production hits on a *later* file) nothing at all.
    """
    monkeypatch.setattr(providers, "_WARNED", set())
    original = boundary_lattice._attach_end_penalties
    pair = ("pos", "shadow-latch-probe")

    def failing(*args, **kwargs):
        note_degraded(*pair)
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary_lattice, "_attach_end_penalties", failing)
    monkeypatch.setenv(FLAG, "1")
    with caplog.at_level("WARNING", logger="voxweave"):
        result = _segment(_case_plain())
    assert result.shadow is not None
    assert {(e["slot"], e["reason"]) for e in result.shadow["shadow_degraded"]} >= {
        pair
    }
    assert not [r for r in caplog.records if pair[1] in r.getMessage()]

    # the latch is still unclaimed, so the shipping context gets its line
    with caplog.at_level("WARNING", logger="voxweave"):
        with providers.degradation_capture():
            note_degraded(*pair)
    assert [r for r in caplog.records if pair[1] in r.getMessage()]
