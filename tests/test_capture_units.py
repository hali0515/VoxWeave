"""Contract tests for ``scripts/capture_scenario.py --with-units``.

The segmentation corpus is only worth what its capture tool is worth, so these
pin the parts a golden case silently depends on: the window rule (a word is
never cut in half), span clipping and shot rebasing, the refusal to clobber an
existing case, schema validity of what is written, the fail-closed license
default, and -- the reason ``--units-only`` exists at all -- that the units path
imports no model code.
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


capture = _load_script("capture_scenario")
cc = capture.cc


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def sibling_document() -> dict[str, Any]:
    """A tiny synthetic sibling JSON with every optional replay key present.

    Units sit on a 1s grid from t=10 so a ``--range 10:14`` window has one unit
    straddling each edge; spans and shots are placed to exercise clip/filter.
    """
    return {
        "language": "zh",
        "segments": [],
        "word_segments": [
            {"text": "早", "start": 9.5, "end": 10.2},  # straddles the window start
            {"text": "上", "start": 10.5, "end": 11.0},
            {"text": "好", "start": 11.2, "end": 11.6},
            {"text": "你", "start": 12.0, "end": 12.4},
            {"text": "在", "start": 12.6, "end": 13.0},
            {"text": "吗", "start": 13.5, "end": 14.4},  # straddles the window end
            {"text": "呢", "start": 15.0, "end": 15.4},  # fully outside
        ],
        "vad_speech": [[9.0, 11.5], [12.0, 16.0]],
        "shot_changes": [9.9, 11.4, 13.0, 14.8],
        "sing_spans": [[13.8, 20.0]],
        "speaker_turns": [
            [9.0, 11.9, "SPEAKER_01"],
            [11.9, 14.2, "SPEAKER_00"],
            [14.2, 16.0, "SPEAKER_01"],
        ],
    }


@pytest.fixture()
def sibling(tmp_path: Path) -> Path:
    path = tmp_path / "Show.S01E01.1080p.json"
    path.write_text(json.dumps(sibling_document()), encoding="utf-8")
    return path


def _args(**overrides: Any) -> Any:
    """A namespace shaped like ``build_parser().parse_args`` output."""
    import argparse

    base: dict[str, Any] = {
        "media": "/nonexistent/Show.S01E01.1080p.mkv",
        "name": "zh-03",
        "desc": "",
        "lang": "",
        "no_separate": False,
        "normalize": False,
        "with_units": "auto",
        "units_only": True,
        "range": None,
        "case_out": None,
        "force": False,
        "tags": "dialogue",
        "license_class": "self-recorded",
        "attribution": None,
        "spdx": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _build(sibling: Path, window: capture.Window, **kwargs: Any) -> dict[str, Any]:
    case, _ = capture.build_case(
        case_id=kwargs.pop("case_id", "zh-03"),
        sibling=sibling,
        document=json.loads(sibling.read_text(encoding="utf-8")),
        window=window,
        source_class=kwargs.pop("source_class", "self-recorded"),
        **kwargs,
    )
    return case


# --------------------------------------------------------------------------- #
# Backward compatibility of the legacy flags
# --------------------------------------------------------------------------- #


def test_legacy_invocation_requests_no_segmentation_case() -> None:
    args = capture.build_parser().parse_args(["ep.mkv", "meido-e12", "--desc", "x"])
    assert args.with_units is None
    assert args.units_only is False


def test_bare_with_units_means_auto() -> None:
    args = capture.build_parser().parse_args(["ep.mkv", "zh-03", "--with-units"])
    assert args.with_units == "auto"
    explicit = capture.build_parser().parse_args(
        ["ep.mkv", "zh-03", "--with-units", "other.json"]
    )
    assert explicit.with_units == "other.json"


def test_units_only_without_with_units_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        capture.main(["ep.mkv", "zh-03", "--units-only"])
    assert excinfo.value.code == 2


def _record_flows(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    ran: list[str] = []
    monkeypatch.setattr(
        capture, "capture_units", lambda _a: ran.append("units") or Path("case.json")
    )
    monkeypatch.setattr(
        capture,
        "capture_songdet",
        lambda _a: ran.append("songdet") or Path("scenario.json"),
    )
    return ran


def test_legacy_run_captures_only_the_song_skip_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _record_flows(monkeypatch)
    assert capture.main(["ep.mkv", "meido-e12"]) == 0
    assert ran == ["songdet"]


def test_units_only_run_skips_the_song_skip_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _record_flows(monkeypatch)
    assert capture.main(["ep.mkv", "zh-03", "--with-units", "--units-only"]) == 0
    assert ran == ["units"]


def test_with_units_alone_captures_both_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _record_flows(monkeypatch)
    assert capture.main(["ep.mkv", "zh-03", "--with-units"]) == 0
    assert ran == ["units", "songdet"]


# --------------------------------------------------------------------------- #
# Sibling discovery
# --------------------------------------------------------------------------- #


def test_auto_sibling_is_the_json_next_to_the_media() -> None:
    media = Path("/media/Show.S01E01.1080p.WEB-DL.mkv")
    assert capture.sibling_json_for(media) == media.with_name(
        "Show.S01E01.1080p.WEB-DL.json"
    )


def test_auto_sibling_keeps_interior_dots() -> None:
    # Path.with_suffix would truncate at the first interior dot.
    media = Path("/media/Some... Title.mkv")
    assert capture.sibling_json_for(media).name == "Some... Title.json"


def test_auto_discovery_finds_the_sibling(sibling: Path, tmp_path: Path) -> None:
    media = sibling.with_name("Show.S01E01.1080p.mkv")
    out = tmp_path / "zh-03.json"
    capture.capture_units(_args(media=str(media), case_out=str(out), range="10:14"))
    assert json.loads(out.read_text(encoding="utf-8"))["id"] == "zh-03"


def test_missing_sibling_is_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(cc.CalibrationError):
        capture.capture_units(_args(media=str(tmp_path / "absent.mkv")))


# --------------------------------------------------------------------------- #
# --range semantics
# --------------------------------------------------------------------------- #


def test_range_keeps_only_units_wholly_inside_and_rebases(sibling: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 14.0))
    assert [u["text"] for u in case["word_segments"]] == ["上", "好", "你", "在"]
    assert case["word_segments"][0]["start"] == pytest.approx(0.5)
    assert case["word_segments"][0]["end"] == pytest.approx(1.0)
    assert case["word_segments"][-1]["end"] == pytest.approx(3.0)
    assert [u["id"] for u in case["word_segments"]] == ["u0", "u1", "u2", "u3"]


def test_range_never_splits_a_word(sibling: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 14.0))
    texts = [u["text"] for u in case["word_segments"]]
    assert "早" not in texts  # starts at 9.5, before the window
    assert "吗" not in texts  # ends at 14.4, after the window
    assert all(0.0 <= u["start"] <= u["end"] <= 4.0 for u in case["word_segments"])


def test_range_clips_and_rebases_spans(sibling: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 14.0))
    assert case["vad_speech"] == [[0.0, 1.5], [2.0, 4.0]]
    # sing_spans 13.8..20.0 survives only as its 0.2s intersection
    assert case["sing_spans"] == [[3.8, 4.0]]


def test_range_drops_spans_outside_the_window(sibling: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 13.0))
    assert case["sing_spans"] == []
    assert "sing_spans" not in case["capture"]["missing_inputs"]


def test_range_filters_and_rebases_shot_points(sibling: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 14.0))
    assert case["shot_changes"] == [1.4, 3.0]  # 9.9 before, 14.8 after


def test_range_clips_turns_and_renumbers_speakers(sibling: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 14.0))
    assert case["speaker_turns"] == [
        {"start": 0.0, "end": 1.9, "speaker": "S0"},
        {"start": 1.9, "end": 4.0, "speaker": "S1"},
    ]


def test_window_duration_is_the_range_length(sibling: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 14.0))
    assert case["capture"]["window_duration_s"] == pytest.approx(4.0)


def test_without_range_the_whole_stream_is_captured(sibling: Path) -> None:
    case = _build(sibling, capture.Window())
    assert len(case["word_segments"]) == 7
    assert case["word_segments"][0]["start"] == pytest.approx(9.5)
    # unbounded window: duration is the last time actually captured
    assert case["capture"]["window_duration_s"] == pytest.approx(20.0)


def test_empty_window_is_invalid_input(sibling: Path) -> None:
    with pytest.raises(cc.CalibrationError):
        _build(sibling, capture.Window(100.0, 120.0))


def test_window_longer_than_the_schema_cap_points_at_range(tmp_path: Path) -> None:
    doc = sibling_document()
    doc["word_segments"] = [{"text": "x", "start": 0.0, "end": 400.0}]
    doc["sing_spans"] = []
    path = tmp_path / "long.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(cc.CalibrationError) as excinfo:
        _build(path, capture.Window())
    assert "--range" in excinfo.value.render()


@pytest.mark.parametrize("spec", ["10", "10:", "a:b", "14:10", "10:10", "-1:5", ""])
def test_bad_range_spec_is_invalid_input(spec: str) -> None:
    with pytest.raises(cc.CalibrationError):
        capture.parse_range(spec)


def test_range_spec_parses_to_a_window() -> None:
    window = capture.parse_range(" 615.0:735.0 ")
    assert (window.start, window.end) == (615.0, 735.0)
    assert window.bounded


# --------------------------------------------------------------------------- #
# Unit projection details
# --------------------------------------------------------------------------- #


def test_word_key_is_mirrored_only_when_the_source_has_one(tmp_path: Path) -> None:
    doc = sibling_document()
    doc["word_segments"] = [
        {"text": "a", "word": "a.", "start": 0.0, "end": 0.5},
        {"text": "b", "start": 0.6, "end": 1.0},
    ]
    path = tmp_path / "w.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    case = _build(path, capture.Window(), language="en")
    assert case["word_segments"][0]["word"] == "a."
    assert "word" not in case["word_segments"][1]


def test_units_without_usable_spans_are_skipped_and_counted(tmp_path: Path) -> None:
    doc = sibling_document()
    doc["word_segments"] = [
        {"text": "ghost", "start": None, "end": None},
        {"text": "real", "start": 0.0, "end": 0.5},
        {"start": 1.0, "end": 1.5},
    ]
    path = tmp_path / "g.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    case, skipped = capture.build_case(
        case_id="en-01",
        sibling=path,
        document=doc,
        window=capture.Window(),
        language="en",
        source_class="self-recorded",
    )
    assert skipped == 2
    assert [u["text"] for u in case["word_segments"]] == ["real"]


def test_sibling_without_word_segments_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    path.write_text(json.dumps({"language": "zh"}), encoding="utf-8")
    with pytest.raises(cc.CalibrationError):
        capture.build_case(
            case_id="zh-03",
            sibling=path,
            document=json.loads(path.read_text(encoding="utf-8")),
            window=capture.Window(),
            source_class="self-recorded",
        )


# --------------------------------------------------------------------------- #
# capture block
# --------------------------------------------------------------------------- #


def test_missing_inputs_lists_absent_keys_and_writes_empty_arrays(
    tmp_path: Path,
) -> None:
    doc = sibling_document()
    for key in ("shot_changes", "speaker_turns"):
        doc.pop(key)
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    case = _build(path, capture.Window(10.0, 14.0))
    assert case["capture"]["missing_inputs"] == ["shot_changes", "speaker_turns"]
    assert case["shot_changes"] == []
    assert case["speaker_turns"] == []


def test_source_digest_is_the_sha256_of_the_sibling_bytes(sibling: Path) -> None:
    import hashlib

    case = _build(sibling, capture.Window(10.0, 14.0))
    expected = hashlib.sha256(sibling.read_bytes()).hexdigest()
    assert case["capture"]["source_digest"] == expected


def test_dependency_versions_cover_every_optional_segmenter() -> None:
    versions = capture.dependency_versions()
    assert set(versions) == {"python", *capture.SEGMENTER_DISTRIBUTIONS}
    assert versions["python"]
    assert all(v is None or isinstance(v, str) for v in versions.values())


def test_config_records_the_segmentation_relevant_values() -> None:
    config = capture.segmentation_config("zh")
    for key in (
        "language",
        "max_line_length",
        "max_lines",
        "max_cue_s",
        "cps",
        "lag_out_s",
        "gap_thresholds",
        "shot_snap_frames",
        "diarize_format",
    ):
        assert key in config, key
    assert config["language"] == "zh"
    assert set(config["gap_thresholds"]) == {"clause_ms", "vad_skip_ms", "offline_ms"}
    assert config["shot_snap_frames"] == pytest.approx(11.0, abs=0.05)


def test_config_follows_an_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXWEAVE_MAX_CUE_SEC", "5.5")
    assert capture.segmentation_config("en")["max_cue_s"] == pytest.approx(5.5)


def test_voxweave_commit_is_a_hash() -> None:
    if not (REPO_ROOT / ".git").exists():  # pragma: no cover - checkout shape
        pytest.skip("not a git checkout")
    assert capture._COMMIT_RE.match(capture.voxweave_commit())


def test_voxweave_commit_failure_is_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(cc.CalibrationError):
        capture.voxweave_commit()


# --------------------------------------------------------------------------- #
# License
# --------------------------------------------------------------------------- #


def test_undeclared_license_refuses_to_build(sibling: Path) -> None:
    with pytest.raises(cc.CalibrationError) as excinfo:
        _build(sibling, capture.Window(10.0, 14.0), source_class="undeclared")
    assert "redistributable" in str(excinfo.value)


def test_license_default_is_undeclared() -> None:
    args = capture.build_parser().parse_args(["ep.mkv", "zh-03", "--with-units"])
    assert args.license_class == capture.UNDECLARED_LICENSE_CLASS
    assert args.license_class not in capture.REDISTRIBUTABLE_CLASSES


def test_attribution_and_spdx_reach_the_license_block(sibling: Path) -> None:
    case = _build(
        sibling,
        capture.Window(10.0, 14.0),
        attribution="Recorded by the author",
        spdx="CC-BY-4.0",
        source_class="cc",
    )
    assert case["license"] == {
        "redistributable": True,
        "source_class": "cc",
        "spdx": "CC-BY-4.0",
        "attribution": "Recorded by the author",
    }


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def test_written_case_validates_against_the_schema(
    sibling: Path, tmp_path: Path
) -> None:
    out = tmp_path / "cases" / "zh-03.json"
    capture.capture_units(
        _args(media=str(sibling.with_suffix(".mkv")), case_out=str(out), range="10:14")
    )
    doc = cc.load_validated_json(out, "segmentation-case")
    assert doc["id"] == "zh-03"
    assert cc.schema_errors(doc, "segmentation-case") == []


def test_case_id_must_match_the_corpus_pattern(sibling: Path, tmp_path: Path) -> None:
    out = tmp_path / "bad.json"
    with pytest.raises(SystemExit) as excinfo:
        capture.capture_units(
            _args(
                media=str(sibling.with_suffix(".mkv")),
                name="not-an-id",
                case_out=str(out),
                range="10:14",
            )
        )
    assert excinfo.value.code == 2
    assert not out.exists()


def test_existing_case_is_not_overwritten_without_force(
    sibling: Path, tmp_path: Path
) -> None:
    out = tmp_path / "zh-03.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(cc.CalibrationError) as excinfo:
        capture.capture_units(
            _args(
                media=str(sibling.with_suffix(".mkv")),
                case_out=str(out),
                range="10:14",
            )
        )
    assert "--force" in str(excinfo.value)
    assert out.read_text(encoding="utf-8") == "{}"


def test_force_overwrites_an_existing_case(sibling: Path, tmp_path: Path) -> None:
    out = tmp_path / "zh-03.json"
    out.write_text("{}", encoding="utf-8")
    capture.capture_units(
        _args(
            media=str(sibling.with_suffix(".mkv")),
            case_out=str(out),
            range="10:14",
            force=True,
        )
    )
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == 1


def test_default_case_out_is_the_corpus_directory(
    sibling: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capture, "DEFAULT_CASE_DIR", tmp_path / "cases")
    capture.capture_units(_args(media=str(sibling.with_suffix(".mkv")), range="10:14"))
    assert (tmp_path / "cases" / "zh-03.json").is_file()


def test_language_override_wins_over_the_sibling(sibling: Path, tmp_path: Path) -> None:
    case = _build(sibling, capture.Window(10.0, 14.0), language="ja")
    assert case["language"] == "ja"
    assert case["capture"]["config"]["language"] == "ja"


def test_language_outside_the_corpus_set_is_invalid(tmp_path: Path) -> None:
    doc = sibling_document()
    doc["language"] = "fr"
    path = tmp_path / "fr.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(cc.CalibrationError):
        _build(path, capture.Window(10.0, 14.0))


def test_tags_are_split_deduplicated_and_required(
    sibling: Path, tmp_path: Path
) -> None:
    out = tmp_path / "zh-03.json"
    capture.capture_units(
        _args(
            media=str(sibling.with_suffix(".mkv")),
            case_out=str(out),
            range="10:14",
            tags="dialogue, shot ,dialogue",
        )
    )
    assert json.loads(out.read_text(encoding="utf-8"))["tags"] == ["dialogue", "shot"]
    with pytest.raises(cc.CalibrationError):
        capture.capture_units(
            _args(
                media=str(sibling.with_suffix(".mkv")),
                case_out=str(tmp_path / "other.json"),
                tags=" , ",
            )
        )


# --------------------------------------------------------------------------- #
# The zero-GPU guarantee
# --------------------------------------------------------------------------- #


def test_units_only_imports_no_model_code(sibling: Path, tmp_path: Path) -> None:
    """--units-only must not drag torch (or any inference stack) into the process."""
    heavy = ("torch", "torchaudio", "panns_inference", "onnxruntime", "transformers")
    preloaded = {name for name in heavy if name in sys.modules}
    capture.capture_units(
        _args(
            media=str(sibling.with_suffix(".mkv")),
            case_out=str(tmp_path / "zh-03.json"),
            range="10:14",
        )
    )
    assert {name for name in heavy if name in sys.modules} == preloaded


def test_units_only_run_is_torch_free_in_a_fresh_interpreter(
    sibling: Path, tmp_path: Path
) -> None:
    """End-to-end guard: a real ``--units-only`` process never imports torch."""
    out = tmp_path / "zh-03.json"
    argv = [
        str(sibling.with_suffix(".mkv")),
        "zh-03",
        "--with-units",
        str(sibling),
        "--units-only",
        "--range",
        "10:14",
        "--license-class",
        "self-recorded",
        "--case-out",
        str(out),
    ]
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('capture_scenario', {str(REPO_ROOT / 'scripts' / 'capture_scenario.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['capture_scenario'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        f"code = mod.main({argv!r})\n"
        "sys.exit(0 if code == 0 and 'torch' not in sys.modules else 3)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
