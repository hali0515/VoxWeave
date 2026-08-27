"""Contract tests for the calibration schemas and scripts/calib_common.py.

These lock the parts every calibration harness shares: the tracked schemas are
themselves valid JSON Schema, an invalid case is rejected with exit code 2, and
percentile / digest / micro aggregation keep exactly one definition each.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

jsonschema = pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "calibration" / "schemas"
SCHEMA_NAMES = (
    "alignment-manifest",
    "alignment-reference",
    "alignment-report",
    "segmentation-case",
    "segmentation-corpus",
    "segmentation-baseline",
)


def _load_calib_common() -> Any:
    """Import scripts/calib_common.py by path (scripts/ is not an installed package)."""
    cached = sys.modules.get("calib_common")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "calib_common", REPO_ROOT / "scripts" / "calib_common.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calib_common"] = module
    spec.loader.exec_module(module)
    return module


cc = _load_calib_common()

DIGEST_64 = "a" * 64


def valid_case() -> dict[str, Any]:
    """A minimal segmentation case that satisfies every required field."""
    return {
        "schema_version": 1,
        "id": "zh-01",
        "language": "zh",
        "description": "two-speaker dialogue, normal pace",
        "tags": ["dialogue", "shot"],
        "license": {
            "redistributable": True,
            "source_class": "self-recorded",
            "spdx": None,
            "attribution": None,
        },
        "capture": {
            "voxweave_commit": "aaea548",
            "source_digest": DIGEST_64,
            "window_duration_s": 12.0,
            "dependency_versions": {
                "python": "3.11.9",
                "pysbd": "0.3.4",
                "fugashi": None,
            },
            "config": {"max_cue_s": 7.0, "max_chars": 20, "max_lines": 2},
            "missing_inputs": ["sing_spans"],
        },
        "word_segments": [
            {"id": "u0", "text": "你好", "start": 0.0, "end": 0.42},
            {"id": "u1", "text": "早上好", "start": 0.5, "end": 1.1},
        ],
        "vad_speech": [[0.0, 1.2]],
        "shot_changes": [0.48],
        "sing_spans": [],
        "speaker_turns": [{"start": 0.0, "end": 1.2, "speaker": "S0"}],
        "exceptions": [],
    }


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_file_is_valid_draft_2020_12(name: str) -> None:
    schema = json.loads(
        (SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(f"{name}.schema.json")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_load_schema_accepts_short_and_full_name(name: str) -> None:
    assert cc.load_schema(name) is cc.load_schema(f"{name}.schema.json")


def test_schema_dir_holds_exactly_the_tracked_contracts() -> None:
    on_disk = sorted(p.name for p in SCHEMA_DIR.glob("*.schema.json"))
    assert on_disk == sorted(f"{n}.schema.json" for n in SCHEMA_NAMES)


def test_load_schema_missing_file_is_calibration_error() -> None:
    with pytest.raises(cc.CalibrationError):
        cc.load_schema("no-such-contract")


# --------------------------------------------------------------------------- #
# Round trip: valid case passes, invalid case is exit 2
# --------------------------------------------------------------------------- #


def test_valid_segmentation_case_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "zh-01.json"
    path.write_text(json.dumps(valid_case()), encoding="utf-8")
    doc = cc.load_validated_json(path, "segmentation-case")
    assert doc["id"] == "zh-01"
    assert cc.schema_errors(doc, "segmentation-case") == []
    cc.validate_or_exit2(doc, "segmentation-case", label=str(path))


@pytest.mark.parametrize(
    ("mutate", "expect_in_message"),
    [
        (lambda c: c.pop("license"), "license"),
        (lambda c: c.update(id="zh-1"), "zh-1"),
        (lambda c: c.update(word_segments=[]), "word_segments"),
        (lambda c: c.update(language="fr"), "fr"),
        (lambda c: c.update(unexpected_key=1), "unexpected_key"),
        (lambda c: c["capture"].update(source_digest="deadbeef"), "source_digest"),
        (lambda c: c["speaker_turns"][0].update(speaker="Alice"), "Alice"),
        (lambda c: c["word_segments"][0].update(start=-1.0), "word_segments"),
    ],
)
def test_invalid_segmentation_case_is_rejected(mutate, expect_in_message: str) -> None:
    case = valid_case()
    mutate(case)
    errors = cc.schema_errors(case, "segmentation-case")
    assert errors, "mutation should have broken the schema"
    assert any(expect_in_message in err for err in errors)


def test_invalid_case_exits_with_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    case = valid_case()
    case.pop("license")
    with pytest.raises(SystemExit) as excinfo:
        cc.validate_or_exit2(case, "segmentation-case", label="zh-01.json")
    assert excinfo.value.code == cc.EXIT_INVALID == 2
    err = capsys.readouterr().err
    assert "zh-01.json" in err and "license" in err


def test_load_validated_json_raises_for_invalid_document(tmp_path: Path) -> None:
    case = valid_case()
    case["capture"]["window_duration_s"] = 0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(cc.CalibrationError) as excinfo:
        cc.load_validated_json(path, "segmentation-case")
    assert excinfo.value.details


def test_unparsable_json_is_calibration_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(cc.CalibrationError):
        cc.read_json(path)


def test_read_json_or_exit2_uses_exit_code_2(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cc.read_json_or_exit2(tmp_path / "missing.json")
    assert excinfo.value.code == 2


def test_corpus_registry_rejects_escaping_paths() -> None:
    registry = {
        "schema_version": 1,
        "metric_definition_version": 3,
        "cases": ["cases/zh-01.json"],
        "required_counts": {"zh": 7, "ja": 7, "en": 6},
        "required_tags": ["fast", "sparse-tail"],
    }
    assert cc.schema_errors(registry, "segmentation-corpus") == []
    for bad in ("../secret/zh-01.json", "/abs/zh-01.json", "cases/zh-01.vtt"):
        assert cc.schema_errors({**registry, "cases": [bad]}, "segmentation-corpus")
    assert cc.schema_errors(
        {**registry, "required_counts": {"zh": 6, "ja": 7, "en": 6}},
        "segmentation-corpus",
    )


def test_baseline_ratio_and_metric_shapes_validate() -> None:
    baseline = {
        "schema_version": 1,
        "metric_definition_version": 3,
        "metric_definition": {
            "version": 3,
            "forbidden_end": {
                "tail_scope": "eligible-internal-plus-document-final",
                "alternative_source": "pre-split-punctuated-source-phrase-lattice",
                "reported_measure": "rate-with-bad-and-eligible",
                "gate_measure": "bad-count",
                "baseline_bad_slack": 1.0,
                "ja_tail_lens": {
                    "id": "ja-char-table-level1",
                    "source": "kinsoku.line_end_penalty",
                    "provider": None,
                    "provider_version": None,
                    "dictionary": None,
                    "context": "punctuated-source-phrase-atom",
                    "missing_offset_fallback": None,
                },
            },
        },
        "metric_definition_digest": DIGEST_64,
        "corpus_digest": DIGEST_64,
        "generated_from_commit": "aaea548",
        "environment": {"python": "3.11.9", "dependencies": {"pysbd": "0.3.4"}},
        "groups": {
            group: {
                "case_count": 1,
                "cue_count": 10,
                "len_break_mid_phrase_rate": cc.Ratio(1, 20).to_dict(),
                "over_7s_rate": cc.Ratio(0, 10).to_dict(),
                "cps_p90": {"n": 10, "value": 14.5},
                "forbidden_end_rate": cc.Ratio(0, 8).to_dict(),
            }
            for group in ("all", "zh", "ja", "en")
        },
        "gates": {
            name: {
                "direction": "lower_is_better",
                "mode": "warning",
                "absolute_max": 0.1,
                "absolute_tolerance": 0.01,
                "relative_tolerance": 0.1,
                "min_samples": 100,
            }
            for name in (
                "len_break_mid_phrase_rate",
                "over_7s_rate",
                "cps_p90",
                "forbidden_end_rate",
            )
        },
    }
    assert cc.schema_errors(baseline, "segmentation-baseline") == []


def test_alignment_report_accepts_metric_block() -> None:
    report = {
        "schema_version": 1,
        "metric_definition_version": 1,
        "manifest_digest": DIGEST_64,
        "status": "pass",
        "lanes": [
            {
                "source_kind": "mfa_words",
                "language": "ja",
                "reference_id": "ja-ep01-mfa",
                "status": "pass",
                "coverage": {"hyp": 0.93, "ref": 0.91},
                "metrics": {"word_start": cc.metric_block([0.01, 0.2, 0.4])},
                "items": [],
            }
        ],
    }
    assert cc.schema_errors(report, "alignment-report") == []


# --------------------------------------------------------------------------- #
# Percentile
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("values", "p", "expected"),
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90.0, 9.1),  # type-7: idx 8.1 -> 9*.9 + 10*.1
        ([1, 2, 3, 4], 50.0, 2.5),
        ([1, 2, 3, 4], 0.0, 1.0),
        ([1, 2, 3, 4], 100.0, 4.0),
        ([5.0], 90.0, 5.0),
        ([10, 1, 4, 7], 25.0, 3.25),  # unsorted input is sorted internally
    ],
)
def test_percentile_is_type_7(values, p, expected) -> None:
    assert cc.percentile(values, p) == pytest.approx(expected)


def test_percentile_matches_legacy_calib_segmentation_definition() -> None:
    # scripts/calib_segmentation.py's percentile() is the definition baselines
    # were recorded with; the shared helper must not move it.
    data = [0.4, 1.2, 3.3, 3.9, 5.0, 8.8, 9.1, 12.0, 13.7, 21.0, 30.5]
    for p in (0, 5, 25, 50, 75, 90, 95, 99, 100):
        s = sorted(data)
        n = len(s)
        idx = (n - 1) * p / 100.0
        lo = int(idx)
        hi = lo + 1
        frac = idx - lo
        legacy = s[-1] if hi >= n else s[lo] * (1 - frac) + s[hi] * frac
        assert cc.percentile(data, p) == pytest.approx(legacy)


def test_percentile_empty_returns_default() -> None:
    assert cc.percentile([], 90.0) is None
    assert cc.percentile([], 90.0, default=0.0) == 0.0


def test_percentile_rejects_out_of_range_p() -> None:
    with pytest.raises(ValueError):
        cc.percentile([1.0, 2.0], 101.0)


# --------------------------------------------------------------------------- #
# Canonical JSON digest
# --------------------------------------------------------------------------- #


def test_canonical_json_is_sorted_and_compact() -> None:
    assert cc.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_digest_is_key_order_and_whitespace_independent() -> None:
    a = {"b": 1, "a": {"y": 2, "x": [3, 4]}}
    b = json.loads(json.dumps({"a": {"x": [3, 4], "y": 2}, "b": 1}, indent=4))
    assert cc.canonical_digest(a) == cc.canonical_digest(b)


def test_digest_matches_declared_construction() -> None:
    obj = {"cases": ["cases/zh-01.json"], "schema_version": 1}
    expected = hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert cc.canonical_digest(obj) == expected
    # Pinned literal: a digest convention change must break this, not silently
    # invalidate every recorded baseline.
    assert cc.canonical_digest(obj) == (
        "ae25afa2e6205afde4c7fcf437bdbc7c23d1e2fce3c337cbdf7beb0b1ee2d7f6"
    )


def test_digest_is_sensitive_to_values() -> None:
    assert cc.canonical_digest({"a": 1}) != cc.canonical_digest({"a": 2})
    assert cc.canonical_digest({"a": 1}) != cc.canonical_digest({"a": "1"})


def test_digest_handles_non_ascii_stably() -> None:
    assert cc.canonical_digest({"text": "你好"}) == cc.canonical_digest(
        {"text": "你好"}
    )


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = b"voxweave" * 1024
    path.write_bytes(payload)
    assert cc.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_write_json_creates_parents_and_is_readable(tmp_path: Path) -> None:
    out = tmp_path / "build" / "calibration" / "report.json"
    cc.write_json(out, {"status": "pass"})
    assert cc.read_json(out) == {"status": "pass"}
    assert not list(out.parent.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# Language canonicalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en", "en"),
        ("EN", "en"),
        ("en-US", "en"),
        ("zh_CN", "zh"),
        ("zh-Hans-CN", "zh"),
        ("ja-JP", "ja"),
        ("jpn", "ja"),  # ffprobe / MFA style ISO-639-3
        ("eng", "en"),
        ("zho", "zh"),
        ("chi", "zh"),
        ("cmn", "zh"),
        ("Japanese", "ja"),  # aligner full names
        ("chinese", "zh"),
        (" english ", "en"),
    ],
)
def test_canonical_language(raw: str, expected: str) -> None:
    assert cc.canonical_language(raw) == expected


def test_canonical_language_rejects_unknown() -> None:
    for raw in ("", None, "klingon", "xx-YY"):
        assert cc.canonical_language_or(raw, "fallback") == "fallback"
        with pytest.raises(cc.CalibrationError):
            cc.canonical_language(raw)


def test_require_calibration_language_excludes_other_supported_languages() -> None:
    assert cc.require_calibration_language("ja-JP") == "ja"
    # yue is a voxweave language but not part of the calibration corpora.
    with pytest.raises(cc.CalibrationError):
        cc.require_calibration_language("yue")


def test_languages_match_is_never_cross_language() -> None:
    assert cc.languages_match("ja", "jpn")
    assert cc.languages_match("zh-CN", "chinese")
    assert not cc.languages_match("ja", "en")
    assert not cc.languages_match("ja", "klingon")
    assert not cc.languages_match(None, None)


def test_group_keys_pairs_all_with_language() -> None:
    assert cc.group_keys("en-GB") == ("all", "en")


# --------------------------------------------------------------------------- #
# Metric blocks and micro aggregation
# --------------------------------------------------------------------------- #


def test_metric_block_shape_and_values() -> None:
    block = cc.metric_block([0.0, 0.02, 0.06, 0.3, 2.0])
    assert block["n"] == 5
    assert block["mae"] == pytest.approx((0.0 + 0.02 + 0.06 + 0.3 + 2.0) / 5)
    assert block["median"] == pytest.approx(0.06)
    assert block["p90"] == pytest.approx(1.32)
    assert block["pct_le_0_025"] == pytest.approx(2 / 5)
    assert block["pct_le_0_05"] == pytest.approx(2 / 5)
    assert block["pct_le_0_10"] == pytest.approx(3 / 5)
    assert block["pct_le_0_25"] == pytest.approx(3 / 5)
    assert block["pct_le_1_0"] == pytest.approx(4 / 5)


def test_metric_block_threshold_is_inclusive() -> None:
    block = cc.metric_block([0.25, 0.25, 0.2500000001])
    assert block["pct_le_0_25"] == pytest.approx(1.0)


def test_metric_block_empty_reports_zero_n_not_zero_error() -> None:
    block = cc.metric_block([])
    assert block["n"] == 0
    assert block["mae"] is None and block["median"] is None and block["p90"] is None
    assert block["pct_le_0_25"] is None
    report_schema = cc.load_schema("alignment-report")
    metric_only = {
        "$schema": report_schema["$schema"],
        "$ref": "#/$defs/metric",
        "$defs": report_schema["$defs"],
    }
    assert cc.schema_errors(block, metric_only) == []


def test_metric_block_rejects_broken_samples() -> None:
    with pytest.raises(cc.CalibrationError):
        cc.metric_block([0.1, float("inf")])
    with pytest.raises(cc.CalibrationError):
        cc.metric_block([0.1, float("nan")])
    with pytest.raises(cc.CalibrationError):
        cc.metric_block([-0.1])


def test_metric_block_optional_fields() -> None:
    block = cc.metric_block([0.1], ci95=(0.05, 0.2), interpretive_lower_bound=0.02)
    assert block["ci95_low"] == 0.05
    assert block["ci95_high"] == 0.2
    assert block["interpretive_lower_bound"] == 0.02


def test_ratio_keeps_numerator_and_denominator() -> None:
    r = cc.Ratio(3, 12)
    assert r.value == pytest.approx(0.25)
    assert r.to_dict() == {"bad": 3, "eligible": 12, "value": pytest.approx(0.25)}


def test_ratio_with_no_eligible_boundaries_is_null_not_zero() -> None:
    assert cc.Ratio(0, 0).value is None
    assert cc.Ratio().to_dict()["value"] is None


def test_ratio_rejects_impossible_counts() -> None:
    with pytest.raises(cc.CalibrationError):
        cc.Ratio(5, 2)
    with pytest.raises(cc.CalibrationError):
        cc.Ratio(-1, 2)


def test_micro_aggregation_is_not_a_mean_of_rates() -> None:
    # Short case 1/2 (50%) and long case 1/98 (~1%): the micro rate is 2/100,
    # the (wrong) macro average would be ~25.5%.
    total = cc.merge_ratios([cc.Ratio(1, 2), cc.Ratio(1, 98)])
    assert total == cc.Ratio(2, 100)
    assert total.value == pytest.approx(0.02)


def test_micro_aggregator_pools_ratios_and_samples() -> None:
    agg = cc.MicroAggregator()
    for group in cc.group_keys("zh"):
        agg.add_ratio(group, "forbidden_end_rate", 1, 40)
        agg.add_samples(group, "cps", [12.0, 18.0])
    for group in cc.group_keys("ja"):
        agg.add_ratio(group, "forbidden_end_rate", 3, 60)
        agg.add_samples(group, "cps", [9.0])

    assert agg.ratio("all", "forbidden_end_rate") == cc.Ratio(4, 100)
    assert agg.ratio("zh", "forbidden_end_rate") == cc.Ratio(1, 40)
    assert agg.ratio("en", "forbidden_end_rate") == cc.Ratio(0, 0)
    assert sorted(agg.samples("all", "cps")) == [9.0, 12.0, 18.0]
    assert agg.groups() == ["all", "ja", "zh"]
    assert agg.metrics("zh") == ["cps", "forbidden_end_rate"]


# --------------------------------------------------------------------------- #
# Exit-code contract
# --------------------------------------------------------------------------- #


def test_exit_code_contract() -> None:
    assert cc.exit_code(valid=True, gates_passed=True) == 0
    assert cc.exit_code(valid=True, gates_passed=False) == 1
    assert cc.exit_code(valid=False, gates_passed=True) == 2
    assert cc.exit_code(valid=False, gates_passed=False) == 2


def test_die_helpers_use_the_shared_codes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as invalid:
        cc.die_invalid("bad manifest", ["items: too short"])
    assert invalid.value.code == 2
    with pytest.raises(SystemExit) as gate:
        cc.die_gate("over_7s_rate regressed")
    assert gate.value.code == 1
    err = capsys.readouterr().err
    assert "items: too short" in err and "over_7s_rate regressed" in err


def test_run_cli_maps_calibration_error_to_2() -> None:
    def broken() -> int:
        raise cc.CalibrationError("corpus digest mismatch")

    with pytest.raises(SystemExit) as excinfo:
        cc.run_cli(broken)
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as ok:
        cc.run_cli(lambda: cc.EXIT_OK)
    assert ok.value.code == 0


def test_calib_common_does_not_pull_the_inference_stack() -> None:
    # The harness must stay runnable in a bare environment (jsonschema only);
    # importing torch here would also make every schema check pay for it.
    source = (REPO_ROOT / "scripts" / "calib_common.py").read_text(encoding="utf-8")
    for banned in (
        "import torch",
        "import numpy",
        "from voxweave.backend",
        "from voxweave.pipeline",
    ):
        assert banned not in source
