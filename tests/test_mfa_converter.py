"""Contract tests for ``scripts/mfa_to_word_segments.py``.

An MFA truth file is only worth what its converter is worth, so these pin the
parts a downstream metric silently trusts: that both Praat TextGrid dialects
parse to the same grid, that every source interval is accounted for
(``included + excluded + outside_window == total``), that ``spn``/OOV words are
written as excluded segments with a reason instead of vanishing, that shard
offsets and ``--range`` rebasing are plain arithmetic on the raw MFA times, that
the nominal MFA uncertainty is stored and never applied, that what is written
validates against ``alignment-reference.schema.json``, and that a malformed
TextGrid exits 2 rather than producing a plausible-looking reference.
"""

from __future__ import annotations

import importlib.util
import json
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


mfa = _load_script("mfa_to_word_segments")
cc = mfa.cc


# --------------------------------------------------------------------------- #
# Fixtures: one grid, written in both TextGrid dialects
# --------------------------------------------------------------------------- #
#
# words tier (7 intervals):   "" / hello / world / sil / spn / Kaguya / ""
# phones tier mirrors it; "Kaguya" is realized entirely as `spn`, i.e. an OOV
# the phone tier exposes even though the word tier still spells it out.

LONG_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 3
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 3
        intervals: size = 7
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 0.9
            text = "hello"
        intervals [3]:
            xmin = 0.9
            xmax = 1.4
            text = "world"
        intervals [4]:
            xmin = 1.4
            xmax = 1.6
            text = "sil"
        intervals [5]:
            xmin = 1.6
            xmax = 2.1
            text = "spn"
        intervals [6]:
            xmin = 2.1
            xmax = 2.6
            text = "Kaguya"
        intervals [7]:
            xmin = 2.6
            xmax = 3
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 3
        intervals: size = 8
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 0.7
            text = "HH"
        intervals [3]:
            xmin = 0.7
            xmax = 0.9
            text = "AH0"
        intervals [4]:
            xmin = 0.9
            xmax = 1.4
            text = "W"
        intervals [5]:
            xmin = 1.4
            xmax = 1.6
            text = "sil"
        intervals [6]:
            xmin = 1.6
            xmax = 2.1
            text = "spn"
        intervals [7]:
            xmin = 2.1
            xmax = 2.6
            text = "spn"
        intervals [8]:
            xmin = 2.6
            xmax = 3
            text = ""
"""

SHORT_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

0
3
<exists>
2
"IntervalTier"
"words"
0
3
7
0
0.5
""
0.5
0.9
"hello"
0.9
1.4
"world"
1.4
1.6
"sil"
1.6
2.1
"spn"
2.1
2.6
"Kaguya"
2.6
3
""
"IntervalTier"
"phones"
0
3
8
0
0.5
""
0.5
0.7
"HH"
0.7
0.9
"AH0"
0.9
1.4
"W"
1.4
1.6
"sil"
1.6
2.1
"spn"
2.1
2.6
"spn"
2.6
3
""
"""

#: Same shape, MFA's multi-speaker tier naming plus an escaped quote in a label.
SPEAKER_TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "speaker1 - words"
        xmin = 0
        xmax = 2
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.4
            text = ""
        intervals [2]:
            xmin = 0.4
            xmax = 1.0
            text = "say ""hi"""
        intervals [3]:
            xmin = 1.0
            xmax = 2
            text = "there"
'''

#: Two speaker word tiers plus a point tier (which carries no boundaries and
#: must be consumed without shifting the tiers that follow it).
SPEAKER_PAIR_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

0
2
<exists>
3
"IntervalTier"
"A - words"
0
2
2
0
0.5
"alpha"
0.5
2
""
"TextTier"
"marks"
0
2
1
0.75
"x"
"IntervalTier"
"B - words"
0
2
2
0
1
""
1
1.5
"beta"
"""

#: Declares 7 intervals and supplies 2 -- the classic truncated/edited TextGrid.
MALFORMED_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 3
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 3
        intervals: size = 7
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 0.9
            text = "hello"
"""

PROVENANCE_ARGS = [
    "--mfa-version",
    "3.0.6",
    "--acoustic-model",
    "english_us_arpa",
    "--dictionary",
    "english_us_arpa",
    "--created-at",
    "2026-08-25T00:00:00Z",
    "--mfa-command",
    "mfa align corpus dict model out",
]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def mfa_out(tmp_path: Path) -> Path:
    """An MFA output directory holding one aligned shard."""
    out = tmp_path / "mfa_out"
    _write(out / "ep01-0000.TextGrid", LONG_TEXTGRID)
    return out


def _run(argv: list[str]) -> int:
    return mfa.main(argv)


def _base_argv(mfa_out: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--mfa-output",
        str(mfa_out),
        "--language",
        "en",
        "--reference-id",
        "en-ep01-mfa",
        "--output",
        str(out),
        *PROVENANCE_ARGS,
        *extra,
    ]


# --------------------------------------------------------------------------- #
# TextGrid parsing
# --------------------------------------------------------------------------- #


def test_long_and_short_formats_parse_to_the_same_grid(tmp_path: Path) -> None:
    long_grid = mfa.parse_textgrid(_write(tmp_path / "long.TextGrid", LONG_TEXTGRID))
    short_grid = mfa.parse_textgrid(_write(tmp_path / "short.TextGrid", SHORT_TEXTGRID))
    assert long_grid == short_grid
    words = mfa.select_tiers(long_grid, "words")
    assert len(words) == 1
    assert [iv.text for iv in words[0].intervals] == [
        "",
        "hello",
        "world",
        "sil",
        "spn",
        "Kaguya",
        "",
    ]
    assert words[0].intervals[1].xmin == 0.5
    assert words[0].intervals[1].xmax == 0.9


def test_speaker_prefixed_tier_and_escaped_quotes(tmp_path: Path) -> None:
    grid = mfa.parse_textgrid(_write(tmp_path / "spk.TextGrid", SPEAKER_TEXTGRID))
    tiers = mfa.select_tiers(grid, "words")
    assert [t.name for t in tiers] == ["speaker1 - words"]
    assert tiers[0].intervals[1].text == 'say "hi"'


def test_short_format_with_utf16_bom(tmp_path: Path) -> None:
    path = tmp_path / "utf16.TextGrid"
    path.write_bytes(SHORT_TEXTGRID.encode("utf-16"))
    assert mfa.parse_textgrid(path) == mfa.parse_textgrid(
        _write(tmp_path / "utf8.TextGrid", SHORT_TEXTGRID)
    )


@pytest.mark.parametrize(
    "text",
    [
        MALFORMED_TEXTGRID,
        'File type = "ooTextFile"\nObject class = "Sound"\n\n0\n1\n',
        LONG_TEXTGRID.replace('class = "IntervalTier"', 'class = "MysteryTier"', 1),
        LONG_TEXTGRID.replace("xmin = 0.5", "xmin = later", 1),
    ],
    ids=["truncated", "not-a-textgrid", "unknown-tier-class", "non-numeric-time"],
)
def test_malformed_textgrid_is_invalid(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path / "bad.TextGrid", text)
    with pytest.raises(cc.CalibrationError):
        mfa.parse_textgrid(path)


def test_malformed_textgrid_exits_2(tmp_path: Path) -> None:
    """End to end: a broken TextGrid must exit 2, never 0 or 1."""
    out_dir = tmp_path / "mfa_out"
    _write(out_dir / "bad.TextGrid", MALFORMED_TEXTGRID)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "mfa_to_word_segments.py"),
            *_base_argv(out_dir, tmp_path / "ref.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == cc.EXIT_INVALID, proc.stderr
    assert not (tmp_path / "ref.json").exists()


# --------------------------------------------------------------------------- #
# Conversion: accounting, exclusions, schema validity
# --------------------------------------------------------------------------- #


def test_reference_is_schema_valid_and_records_provenance(
    mfa_out: Path, tmp_path: Path
) -> None:
    out = tmp_path / "truth" / "en-ep01.words.json"
    report_path = tmp_path / "report.json"
    assert _run(_base_argv(mfa_out, out, "--report", str(report_path))) == cc.EXIT_OK

    reference = json.loads(out.read_text(encoding="utf-8"))
    assert cc.schema_errors(reference, "alignment-reference") == []
    assert reference["kind"] == "mfa_words"
    assert reference["language"] == "en"
    assert reference["timebase"] == "seconds"

    prov = reference["provenance"]
    assert prov["tool_version"] == "3.0.6"
    assert prov["acoustic_model"] == "english_us_arpa"
    assert prov["dictionary"] == "english_us_arpa"
    assert prov["command"] == "mfa align corpus dict model out"
    assert prov["source_digest"] == mfa.source_digest(
        [mfa.Shard("ep01-0000", mfa_out / "ep01-0000.TextGrid", 0.0)], mfa_out
    )
    assert prov["annotators"] == 0


def test_exclusion_accounting_identity(mfa_out: Path, tmp_path: Path) -> None:
    """included + excluded == total when no window is applied; nothing is dropped."""
    out = tmp_path / "ref.json"
    report_path = tmp_path / "report.json"
    _run(_base_argv(mfa_out, out, "--report", str(report_path)))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    counts = report["intervals"]
    assert counts["total"] == 7
    assert counts["outside_window"] == 0
    assert counts["included"] + counts["excluded"] == counts["total"]
    assert counts["included"] == 2
    assert report["excluded_by_reason"] == {
        "oov_spn_phones": 1,
        "silence": 3,
        "spn": 1,
    }
    assert sum(report["excluded_by_reason"].values()) == counts["excluded"]
    assert len(report["excluded_intervals"]) == counts["excluded"]
    assert report["excluded_ratio"] == {
        "bad": 5,
        "eligible": 7,
        "value": pytest.approx(5 / 7),
    }
    # spn + the phone-tier OOV, against the words this reference can vouch for.
    assert report["lexical_excluded_ratio"] == {
        "bad": 2,
        "eligible": 4,
        "value": pytest.approx(0.5),
    }


def test_spn_and_oov_are_excluded_segments_silence_is_report_only(
    mfa_out: Path, tmp_path: Path
) -> None:
    out = tmp_path / "ref.json"
    report_path = tmp_path / "report.json"
    _run(_base_argv(mfa_out, out, "--report", str(report_path)))
    reference = json.loads(out.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    truth = [s for s in reference["segments"] if not s.get("excluded")]
    excluded = [s for s in reference["segments"] if s.get("excluded")]
    assert [s["text"] for s in truth] == ["hello", "world"]
    assert {(s["text"], s["exclude_reason"]) for s in excluded} == {
        ("spn", "spn"),
        ("Kaguya", "oov_spn_phones"),
    }
    # Silence has no schema segment (text needs >= 1 char) but is still named.
    assert not any(s["text"] in {"", "sil"} for s in reference["segments"])
    silence = [r for r in report["excluded_intervals"] if r["reason"] == "silence"]
    assert [r["label"] for r in silence] == ["", "sil", ""]


def test_oov_list_marks_words_without_a_phone_tier(
    mfa_out: Path, tmp_path: Path
) -> None:
    oov = _write(tmp_path / "oovs.txt", "# found by mfa find_oovs\nWORLD\t3\n")
    out = tmp_path / "ref.json"
    report_path = tmp_path / "report.json"
    _run(
        _base_argv(
            mfa_out,
            out,
            "--oov-list",
            str(oov),
            "--phone-tier",
            "",
            "--report",
            str(report_path),
        )
    )
    reference = json.loads(out.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    by_text = {s["text"]: s for s in reference["segments"]}
    assert by_text["world"]["excluded"] is True
    assert by_text["world"]["exclude_reason"] == "oov_dictionary"
    # Without the phone tier "Kaguya" is no longer detectable as OOV.
    assert "excluded" not in by_text["Kaguya"]
    assert report["excluded_by_reason"] == {"oov_dictionary": 1, "silence": 3, "spn": 1}
    counts = report["intervals"]
    assert counts["included"] + counts["excluded"] == counts["total"]


def test_segment_times_are_raw_mfa_times(mfa_out: Path, tmp_path: Path) -> None:
    """No smoothing, no padding, and no uncertainty applied to the boundaries."""
    out = tmp_path / "ref.json"
    _run(_base_argv(mfa_out, out))
    reference = json.loads(out.read_text(encoding="utf-8"))
    hello = next(s for s in reference["segments"] if s["text"] == "hello")
    assert (hello["start"], hello["end"]) == (0.5, 0.9)


# --------------------------------------------------------------------------- #
# Shards, offsets and the window
# --------------------------------------------------------------------------- #


@pytest.fixture()
def two_shards(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "mfa_out"
    _write(out / "ep01-0000.TextGrid", LONG_TEXTGRID)
    _write(out / "ep01-0001.TextGrid", SHORT_TEXTGRID)
    shard_map = _write(
        tmp_path / "shard-map.json",
        json.dumps(
            {
                "shards": [
                    {
                        "id": "ep01-0000",
                        "textgrid": "ep01-0000.TextGrid",
                        "global_offset_s": 0.0,
                    },
                    {
                        "id": "ep01-0001",
                        "textgrid": "ep01-0001.TextGrid",
                        "global_offset_s": 10.0,
                    },
                ]
            }
        ),
    )
    return out, shard_map


def test_shard_offsets_are_added_to_every_time(
    two_shards: tuple[Path, Path], tmp_path: Path
) -> None:
    mfa_out, shard_map = two_shards
    out = tmp_path / "ref.json"
    report_path = tmp_path / "report.json"
    _run(
        _base_argv(
            mfa_out, out, "--shard-map", str(shard_map), "--report", str(report_path)
        )
    )
    reference = json.loads(out.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    truth = [s for s in reference["segments"] if not s.get("excluded")]
    assert [(s["text"], s["start"], s["end"], s["utterance_id"]) for s in truth] == [
        ("hello", 0.5, 0.9, "ep01-0000"),
        ("world", 0.9, 1.4, "ep01-0000"),
        ("hello", 10.5, 10.9, "ep01-0001"),
        ("world", 10.9, 11.4, "ep01-0001"),
    ]
    counts = report["intervals"]
    assert counts["total"] == 14
    assert counts["included"] + counts["excluded"] == counts["total"]
    assert cc.schema_errors(reference, "alignment-reference") == []


def test_range_keeps_only_fully_contained_words_and_rebases(
    two_shards: tuple[Path, Path], tmp_path: Path
) -> None:
    mfa_out, shard_map = two_shards
    out = tmp_path / "ref.json"
    report_path = tmp_path / "report.json"
    _run(
        _base_argv(
            mfa_out,
            out,
            "--shard-map",
            str(shard_map),
            "--range",
            "10.4:11.5",
            "--report",
            str(report_path),
        )
    )
    reference = json.loads(out.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert [(s["text"], s["start"], s["end"]) for s in reference["segments"]] == [
        ("hello", 0.1, 0.5),
        ("world", 0.5, 1.0),
    ]
    assert reference["media_duration_s"] == 1.1
    counts = report["intervals"]
    assert counts["total"] == 14
    assert counts["included"] == 2
    assert counts["excluded"] == 0
    assert counts["outside_window"] == 12
    assert counts["included"] + counts["excluded"] + counts["outside_window"] == 14


def test_window_dropping_every_word_is_invalid(
    two_shards: tuple[Path, Path], tmp_path: Path
) -> None:
    mfa_out, shard_map = two_shards
    with pytest.raises(cc.CalibrationError):
        _run(
            _base_argv(
                mfa_out,
                tmp_path / "ref.json",
                "--shard-map",
                str(shard_map),
                "--range",
                "100.0:120.0",
            )
        )


def test_several_shards_without_a_map_are_invalid(
    two_shards: tuple[Path, Path], tmp_path: Path
) -> None:
    mfa_out, _ = two_shards
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(_base_argv(mfa_out, tmp_path / "ref.json"))
    assert "--shard-map" in excinfo.value.message


def test_shard_map_must_describe_every_textgrid(
    two_shards: tuple[Path, Path], tmp_path: Path
) -> None:
    mfa_out, _ = two_shards
    partial = _write(tmp_path / "partial.json", json.dumps({"ep01-0000": 0.0}))
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(_base_argv(mfa_out, tmp_path / "ref.json", "--shard-map", str(partial)))
    assert "ep01-0001" in excinfo.value.render()


def test_overlapping_shard_offsets_are_invalid(
    two_shards: tuple[Path, Path], tmp_path: Path
) -> None:
    mfa_out, _ = two_shards
    overlapping = _write(
        tmp_path / "overlap.json", json.dumps({"ep01-0000": 0.0, "ep01-0001": 1.0})
    )
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(
            _base_argv(mfa_out, tmp_path / "ref.json", "--shard-map", str(overlapping))
        )
    assert "overlap" in excinfo.value.message


def test_flat_shard_map_shape_is_accepted(
    two_shards: tuple[Path, Path], tmp_path: Path
) -> None:
    mfa_out, _ = two_shards
    flat = _write(
        tmp_path / "flat.json", json.dumps({"ep01-0000": 0.0, "ep01-0001": 10.0})
    )
    out = tmp_path / "ref.json"
    assert _run(_base_argv(mfa_out, out, "--shard-map", str(flat))) == cc.EXIT_OK
    reference = json.loads(out.read_text(encoding="utf-8"))
    assert reference["segments"][-1]["utterance_id"] == "ep01-0001"


def test_media_duration_bound_is_enforced(mfa_out: Path, tmp_path: Path) -> None:
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(_base_argv(mfa_out, tmp_path / "ref.json", "--media-duration", "1.0"))
    assert "outside the media" in excinfo.value.message


def test_media_offset_is_stored_not_baked(mfa_out: Path, tmp_path: Path) -> None:
    out = tmp_path / "ref.json"
    _run(_base_argv(mfa_out, out, "--media-offset", "2.5"))
    reference = json.loads(out.read_text(encoding="utf-8"))
    assert reference["offset_s"] == 2.5
    hello = next(s for s in reference["segments"] if s["text"] == "hello")
    assert hello["start"] == 0.5  # compare time applies the offset, not the file


# --------------------------------------------------------------------------- #
# Provenance and uncertainty
# --------------------------------------------------------------------------- #


def test_nominal_uncertainty_is_metadata_only(mfa_out: Path, tmp_path: Path) -> None:
    out = tmp_path / "ref.json"
    report_path = tmp_path / "report.json"
    _run(_base_argv(mfa_out, out, "--report", str(report_path)))
    reference = json.loads(out.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert reference["provenance"]["reference_uncertainty_s"] == pytest.approx(0.020)
    assert report["reference_uncertainty_s"] == pytest.approx(0.020)
    assert "never subtracted" in report["uncertainty_note"]
    # A 20 ms nominal uncertainty must not have moved a single boundary.
    hello = next(s for s in reference["segments"] if s["text"] == "hello")
    world = next(s for s in reference["segments"] if s["text"] == "world")
    assert (hello["start"], hello["end"]) == (0.5, 0.9)
    assert (world["start"], world["end"]) == (0.9, 1.4)


def test_missing_mfa_identity_is_invalid(mfa_out: Path, tmp_path: Path) -> None:
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(
            [
                "--mfa-output",
                str(mfa_out),
                "--language",
                "en",
                "--reference-id",
                "en-ep01-mfa",
                "--output",
                str(tmp_path / "ref.json"),
            ]
        )
    details = excinfo.value.render()
    assert "tool_version" in details and "acoustic_model" in details


def test_floating_model_identifier_is_invalid(mfa_out: Path, tmp_path: Path) -> None:
    argv = _base_argv(mfa_out, tmp_path / "ref.json")
    argv[argv.index("--mfa-version") + 1] = "latest"
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(argv)
    assert "exact version" in excinfo.value.message


def test_provenance_file_merges_and_flags_win(mfa_out: Path, tmp_path: Path) -> None:
    side = _write(
        tmp_path / "prov.json",
        json.dumps(
            {
                "created_by": "hali",
                "tool_version": "3.0.0",
                "acoustic_model": "japanese_mfa",
                "dictionary": "japanese_mfa",
                "license": "self-recorded",
            }
        ),
    )
    out = tmp_path / "ref.json"
    _run(_base_argv(mfa_out, out, "--provenance", str(side)))
    prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert prov["created_by"] == "hali"
    assert prov["license"] == "self-recorded"
    assert prov["tool_version"] == "3.0.6"  # the flag wins over the file


def test_provenance_file_rejects_unknown_keys(mfa_out: Path, tmp_path: Path) -> None:
    side = _write(
        tmp_path / "prov.json", json.dumps({"created_by": "hali", "source_digest": "x"})
    )
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(_base_argv(mfa_out, tmp_path / "ref.json", "--provenance", str(side)))
    assert "source_digest" in excinfo.value.render()


def test_model_path_is_hashed_when_it_is_a_real_file(
    mfa_out: Path, tmp_path: Path
) -> None:
    model = _write(tmp_path / "japanese_mfa.zip", "not really a zip")
    out = tmp_path / "ref.json"
    argv = _base_argv(mfa_out, out)
    argv[argv.index("--acoustic-model") + 1] = str(model)
    _run(argv)
    prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert prov["acoustic_model_sha256"] == cc.sha256_file(model)
    assert prov["dictionary_sha256"] is None  # a bare name cannot be hashed


# --------------------------------------------------------------------------- #
# Output handling
# --------------------------------------------------------------------------- #


def test_existing_reference_is_not_clobbered(mfa_out: Path, tmp_path: Path) -> None:
    out = _write(tmp_path / "ref.json", "{}")
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(_base_argv(mfa_out, out))
    assert "--force" in excinfo.value.message
    assert out.read_text(encoding="utf-8") == "{}"
    assert _run(_base_argv(mfa_out, out, "--force")) == cc.EXIT_OK
    assert json.loads(out.read_text(encoding="utf-8"))["id"] == "en-ep01-mfa"


def test_reference_id_must_be_manifest_safe(mfa_out: Path, tmp_path: Path) -> None:
    argv = _base_argv(mfa_out, tmp_path / "ref.json")
    argv[argv.index("--reference-id") + 1] = "EN Episode 01"
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(argv)
    assert "manifest-safe" in excinfo.value.message


def test_language_outside_the_calibration_set_is_invalid(
    mfa_out: Path, tmp_path: Path
) -> None:
    argv = _base_argv(mfa_out, tmp_path / "ref.json")
    argv[argv.index("--language") + 1] = "ko"
    with pytest.raises(cc.CalibrationError):
        _run(argv)


def test_language_tag_is_canonicalized(mfa_out: Path, tmp_path: Path) -> None:
    argv = _base_argv(mfa_out, tmp_path / "ref.json")
    argv[argv.index("--language") + 1] = "en-US"
    _run(argv)
    assert (
        json.loads((tmp_path / "ref.json").read_text(encoding="utf-8"))["language"]
        == "en"
    )


def test_missing_word_tier_names_what_is_present(mfa_out: Path, tmp_path: Path) -> None:
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(_base_argv(mfa_out, tmp_path / "ref.json", "--tier", "mots"))
    assert "words" in excinfo.value.render()


# --------------------------------------------------------------------------- #
# Multi-speaker tiers and point tiers
# --------------------------------------------------------------------------- #


def test_point_tier_is_consumed_without_shifting_later_tiers(tmp_path: Path) -> None:
    grid = mfa.parse_textgrid(_write(tmp_path / "pair.TextGrid", SPEAKER_PAIR_TEXTGRID))
    assert [t.name for t in grid.tiers] == ["A - words", "marks", "B - words"]
    assert grid.tiers[1].intervals == ()
    assert [t.name for t in mfa.select_tiers(grid, "words")] == [
        "A - words",
        "B - words",
    ]


def test_two_speaker_tiers_merge_in_time_order(tmp_path: Path) -> None:
    out_dir = tmp_path / "mfa_out"
    _write(out_dir / "ep01-0000.TextGrid", SPEAKER_PAIR_TEXTGRID)
    out = tmp_path / "ref.json"
    _run(_base_argv(out_dir, out))
    reference = json.loads(out.read_text(encoding="utf-8"))
    assert [(s["text"], s["start"], s["end"]) for s in reference["segments"]] == [
        ("alpha", 0.0, 0.5),
        ("beta", 1.0, 1.5),
    ]
    assert cc.schema_errors(reference, "alignment-reference") == []


def test_overlapping_speaker_tiers_are_invalid(tmp_path: Path) -> None:
    out_dir = tmp_path / "mfa_out"
    _write(
        out_dir / "ep01-0000.TextGrid",
        SPEAKER_PAIR_TEXTGRID.replace('1\n1.5\n"beta"', '0.4\n0.9\n"beta"'),
    )
    with pytest.raises(cc.CalibrationError) as excinfo:
        _run(_base_argv(out_dir, tmp_path / "ref.json"))
    assert "--tier" in excinfo.value.render()


# --------------------------------------------------------------------------- #
# Interval classification unit rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (mfa.Interval(0.0, 0.5, "hello"), (True, None)),
        (mfa.Interval(0.0, 0.5, ""), (False, mfa.REASON_SILENCE)),
        (mfa.Interval(0.0, 0.5, "  sil "), (False, mfa.REASON_SILENCE)),
        (mfa.Interval(0.0, 0.5, "<eps>"), (False, mfa.REASON_SILENCE)),
        (mfa.Interval(0.0, 0.5, "SPN"), (False, mfa.REASON_SPN)),
        (mfa.Interval(0.5, 0.5, "hello"), (False, mfa.REASON_ZERO_DURATION)),
        (mfa.Interval(0.9, 0.5, "hello"), (False, mfa.REASON_ZERO_DURATION)),
        (mfa.Interval(float("nan"), 0.5, "hello"), (False, mfa.REASON_NON_FINITE)),
    ],
)
def test_classify_interval_rules(
    interval: Any, expected: tuple[bool, str | None]
) -> None:
    assert mfa.classify_interval(interval) == expected


def test_uncertainty_from_a_provenance_file_survives(
    mfa_out: Path, tmp_path: Path
) -> None:
    side = _write(
        tmp_path / "prov.json",
        json.dumps(
            {
                "created_by": "hali",
                "tool_version": "3.0.6",
                "acoustic_model": "japanese_mfa",
                "dictionary": "japanese_mfa",
                "reference_uncertainty_s": 0.011,
            }
        ),
    )
    out = tmp_path / "ref.json"
    _run(
        [
            "--mfa-output",
            str(mfa_out),
            "--language",
            "en",
            "--reference-id",
            "en-ep01-mfa",
            "--output",
            str(out),
            "--provenance",
            str(side),
        ]
    )
    prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert prov["reference_uncertainty_s"] == pytest.approx(0.011)


def test_uncertainty_flag_overrides_the_default(mfa_out: Path, tmp_path: Path) -> None:
    out = tmp_path / "ref.json"
    _run(_base_argv(mfa_out, out, "--nominal-uncertainty-s", "0.03"))
    prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert prov["reference_uncertainty_s"] == pytest.approx(0.03)
