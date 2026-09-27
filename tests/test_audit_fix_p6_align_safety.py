"""Regressions for the over-budget route hints, VTT decode classes and the oracle
dependency gate."""

from __future__ import annotations

import codecs
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voxweave import align_dp_safety
from voxweave.align_dp_safety import DpRouteHintsInvalid, validate_over_budget_hints

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_RUNNER = REPO_ROOT / "scripts" / "p6_oracle.py"

_BUDGET = {
    "sample_rate": 16_000,
    "max_dp_frames": 1250,
    "frame_stride": 320,
    "chunk_fraction": 0.8,
}


def _validate(bounds: Any, *, audio_end: float = 120.0) -> None:
    validate_over_budget_hints(
        bounds,
        block_count=len(bounds),
        audio_end=audio_end,
        **_BUDGET,
    )


def _rescued_spans() -> list[tuple[float, float]]:
    """Spans shaped by the align finalizer: a flash cue is stretched across its
    shorter successor, which then ends before it (a nested pair)."""

    from voxweave import realign

    spans = realign.rescue_tiny_cues(
        [(0.0, 8.0), (10.0, 10.1), (10.15, 10.4), (10.5, 18.0), (30.0, 38.0)],
        trig=0.2,
        target=0.5,
    )
    assert spans[1] == (10.0, 10.5) and spans[2] == (10.15, 10.4)
    return spans


# -- over-budget route hints ------------------------------------------------------


def test_nested_cues_written_by_align_are_accepted() -> None:
    _validate(_rescued_spans())


def test_overlapping_cue_that_starts_before_its_predecessor_is_accepted() -> None:
    _validate([(0.0, 5.0), (10.0, 14.0), (9.5, 12.0), (20.0, 25.0)])


def test_cue_entirely_before_an_earlier_cue_is_refused_by_number() -> None:
    with pytest.raises(DpRouteHintsInvalid) as caught:
        _validate([(0.0, 1.0), (100.0, 101.0), (2.0, 3.0)])
    assert caught.value.failure.detail_code == "hint-nonmonotone"
    message = str(caught.value)
    assert "DP budget" in message
    assert "cue 3" in message and "cue 2" in message


def test_unsafe_cue_geometry_names_the_cue() -> None:
    with pytest.raises(DpRouteHintsInvalid) as caught:
        _validate([(0.0, 1.0), (-1.0, 2.0)])
    assert caught.value.failure.detail_code == "hint-nonmonotone"
    assert "cue 2" in str(caught.value)


@pytest.mark.parametrize("bounds", ([None, None], None))
def test_untimed_draft_refusal_explains_the_remedy(bounds: Any) -> None:
    with pytest.raises(DpRouteHintsInvalid) as caught:
        validate_over_budget_hints(
            bounds,
            block_count=2,
            audio_end=120.0,
            **_BUDGET,
        )
    assert caught.value.failure.detail_code == "hint-shape"
    message = str(caught.value)
    assert "DP budget" in message
    assert "timestamp" in message
    assert "VOXWEAVE_CTC_MAX_DP_FRAMES" in message
    assert "--no-timestamps" in message


@pytest.mark.parametrize(("second_hi", "named"), ((3, "(cues 2-3)"), (2, "(cue 2)")))
def test_over_budget_plan_refusal_names_its_cues(second_hi: int, named: str) -> None:
    with pytest.raises(DpRouteHintsInvalid) as caught:
        align_dp_safety.validate_over_budget_plans(
            [
                {"lo": 0, "hi": 1, "start": 0.0, "end": 10.0},
                {"lo": 1, "hi": second_hi, "start": 10.0, "end": 39.0},
            ],
            block_count=second_hi,
            audio_end=40.0,
            sample_count=40 * 16_000,
            **_BUDGET,
        )
    assert caught.value.failure.detail_code == "crop-over-budget"
    assert named in str(caught.value)


def test_envelope_is_the_identity_for_monotone_hints() -> None:
    bounds = ((0.0, 8.0), None, (10, 18), (18.0, 18.0), None)
    assert align_dp_safety.route_hint_envelope(bounds) is bounds


def test_envelope_widens_overlapping_cues_to_a_monotone_cover() -> None:
    bounds = [(0.0, 5.0), (10.0, 14.0), None, (9.5, 12.0), (20.0, 25.0)]
    envelope = align_dp_safety.route_hint_envelope(bounds)
    assert envelope == [(0.0, 5.0), (9.5, 14.0), None, (9.5, 14.0), (20.0, 25.0)]
    for original, widened in zip(bounds, envelope, strict=True):
        if original is None:
            assert widened is None
            continue
        assert widened is not None
        assert widened[0] <= original[0] and original[1] <= widened[1]


def test_over_budget_chunks_never_cut_through_a_nested_cue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_common

    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    sample_rate = 16_000
    wav = np.zeros(40 * sample_rate, dtype=np.float32)
    # Cue 1 runs to 12 s; cue 2 is nested inside it. Planning on the raw ends would
    # cut in the 2 s..14 s "gap" after cue 2 and crop cue 1's tail off.
    bounds = [(0.0, 12.0), (1.0, 2.0), (14.0, 30.0), (32.0, 39.0)]
    texts = ["A", "B", "C", "D"]
    calls = align_common._prepare_dp_calls(wav, sample_rate, texts, bounds, "MMS")
    assert len(calls) > 1
    cursor = 0
    for call in calls:
        owned = range(cursor, cursor + len(call.texts))
        cursor += len(call.texts)
        for index in owned:
            start, end = bounds[index]
            assert call.audio_sample_start <= int(start * sample_rate)
            assert int(end * sample_rate) <= call.audio_sample_end
    assert cursor == len(texts)


def test_monotone_over_budget_plan_is_unchanged_by_the_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_common
    from voxweave.chunking import plan_dp_chunks

    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    sample_rate = 16_000
    wav = np.zeros(40 * sample_rate, dtype=np.float32)
    bounds = [(0.0, 8.0), None, (10.0, 18.0), (22.0, 30.0), (32.0, 39.0)]
    texts = ["A", "B", "C", "D", "E"]
    calls = align_common._prepare_dp_calls(wav, sample_rate, texts, bounds, "CTC")
    plans = plan_dp_chunks(bounds, max_sec=20.0, audio_end=40.0)
    assert [
        (call.audio_sample_start, call.audio_sample_end, tuple(call.texts))
        for call in calls
    ] == [
        (
            max(0, int(plan["start"] * sample_rate)),
            min(len(wav), int(plan["end"] * sample_rate)),
            tuple(texts[plan["lo"] : plan["hi"]]),
        )
        for plan in plans
    ]


def _two_second_grid() -> list[tuple[float, float]]:
    """60 one-second cues every 2 s over 120 s: many 1 s gaps, no 1.5 s silence."""

    return [(2.0 * k, 2.0 * k + 1.0) for k in range(60)]


def _prepare_long(
    monkeypatch: pytest.MonkeyPatch, bounds: list[tuple[float, float] | None]
) -> list[Any]:
    from voxweave import align_common

    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)  # 20 s pieces
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    wav = np.zeros(120 * 16_000, dtype=np.float32)
    texts = [f"w{index}" for index in range(len(bounds))]
    return align_common._prepare_dp_calls(wav, 16_000, texts, bounds, "MMS")


def _cues_outside_their_crop(
    calls: list[Any], bounds: list[tuple[float, float] | None]
) -> list[int]:
    outside: list[int] = []
    cursor = 0
    for call in calls:
        for index in range(cursor, cursor + len(call.texts)):
            pair = bounds[index]
            if pair is None:
                continue
            if int(pair[0] * 16_000) < call.audio_sample_start or (
                int(pair[1] * 16_000) > call.audio_sample_end
            ):
                outside.append(index + 1)
        cursor += len(call.texts)
    assert cursor == len(bounds)
    return outside


def test_early_start_typo_that_forces_a_split_inside_it_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cue 16 really plays at 30-31 s but is typed from 0 s. It still overlaps cue
    # 15, so the hints pass, yet it leaves no silence before it: every split in the
    # first 20 s piece would cut correctly timed cues off from their own audio.
    bounds: list[tuple[float, float] | None] = list(_two_second_grid())
    bounds[15] = (0.0, 31.0)
    with pytest.raises(DpRouteHintsInvalid) as caught:
        _prepare_long(monkeypatch, bounds)
    assert caught.value.failure.detail_code == "hint-nonmonotone"
    message = str(caught.value)
    assert "cue 16 (00:00:00.000 --> 00:00:31.000)" in message
    assert "between cues 10 and 11" in message


def test_widening_that_stays_inside_one_piece_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounds: list[tuple[float, float] | None] = list(_two_second_grid())
    bounds[3] = (0.5, 7.0)  # overlaps cues 1-3 but no split has to fall there
    calls = _prepare_long(monkeypatch, bounds)
    assert len(calls) > 1
    assert _cues_outside_their_crop(calls, bounds) == []


def test_widened_hints_never_crop_a_cue_out_of_its_own_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    rng = random.Random(20260927)
    accepted = refused = 0
    for _ in range(400):
        bounds: list[tuple[float, float] | None] = list(_two_second_grid())
        for index in rng.sample(range(60), rng.randint(1, 2)):
            pair = bounds[index]
            assert pair is not None
            start, end = pair
            kind = rng.randrange(3)
            if kind == 0:  # start typed too early
                bounds[index] = (rng.uniform(0.0, start), end)
            elif kind == 1:  # end typed too late
                bounds[index] = (start, min(119.0, end + rng.uniform(0.5, 40.0)))
            else:  # stretched across its successors, which then nest inside it
                bounds[index] = (start, min(119.0, end + rng.uniform(1.0, 6.0)))
        try:
            calls = _prepare_long(monkeypatch, bounds)
        except DpRouteHintsInvalid:
            refused += 1
            continue
        accepted += 1
        assert _cues_outside_their_crop(calls, bounds) == [], bounds
    assert accepted > 50 and refused > 50


def test_widened_plan_refusal_without_an_overlap_names_the_cropped_cue() -> None:
    hints: list[tuple[float, float] | None] = [
        (0.0, 10.0),
        (2.0, 3.0),
        None,
        (10.2, 12.0),
    ]
    envelope = align_dp_safety.route_hint_envelope(hints)
    assert envelope is not hints
    with pytest.raises(DpRouteHintsInvalid) as caught:
        align_dp_safety.validate_widened_plans(
            [
                {"lo": 0, "hi": 3, "start": 0.0, "end": 10.5},
                {"lo": 3, "hi": 4, "start": 10.5, "end": 20.0},
            ],
            hints=hints,
            envelope=envelope,
        )
    assert caught.value.failure.detail_code == "hint-nonmonotone"
    message = str(caught.value)
    assert "cue 4 (00:00:10.200 --> 00:00:12.000)" in message
    assert "between cues 3 and 4" in message


def test_plans_on_unwidened_hints_are_not_rechecked() -> None:
    # A forward overlap keeps its previous planning, including a split inside it.
    hints = [(0.0, 10.0), (9.0, 20.0)]
    align_dp_safety.validate_widened_plans(
        [
            {"lo": 0, "hi": 1, "start": 0.0, "end": 9.5},
            {"lo": 1, "hi": 2, "start": 9.5, "end": 20.5},
        ],
        hints=hints,
        envelope=align_dp_safety.route_hint_envelope(hints),
    )


class _ModelReached(Exception):
    pass


def _long_ja_episode(tmp_path: Path, vtt: bytes) -> tuple[Path, Path]:
    vtt_path = tmp_path / "episode.vtt"
    media_path = tmp_path / "episode.wav"
    vtt_path.write_bytes(vtt)
    (tmp_path / "episode.json").write_text(
        json.dumps({"language": "ja"}), encoding="utf-8"
    )
    media_path.write_bytes(b"synthetic-media")
    return vtt_path, media_path


def _configure_long_mms(
    monkeypatch: pytest.MonkeyPatch, media_path: Path, *, seconds: int
) -> list[str]:
    from voxweave import align_common, align_mms, backend, config, pipeline

    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media_path
    )
    monkeypatch.setattr(backend, "uses_mms", lambda _iso: True)
    monkeypatch.setattr(config, "align_model_for", lambda _iso: None)
    monkeypatch.setattr(
        align_mms,
        "_read_wav_16k",
        lambda _path: np.zeros(seconds * align_mms.MMS_SR, dtype=np.float32),
    )
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    monkeypatch.setattr(align_mms, "_empty_cache", lambda: None)
    reached: list[str] = []

    def emit(_wav: Any, text: str, _iso: str) -> list[dict]:
        reached.append(text)
        raise _ModelReached

    monkeypatch.setattr(align_mms, "_mms_emit_units", emit)
    return reached


def test_long_media_realign_of_rescued_cues_reaches_the_aligner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxweave import pipeline, realign

    spans = _rescued_spans()
    texts = ["あ", "い", "う", "え", "お"]
    vtt = realign.render_cues(
        [(start, end, text) for (start, end), text in zip(spans, texts, strict=True)]
    )
    vtt_path, media_path = _long_ja_episode(tmp_path, vtt.encode("utf-8"))
    reached = _configure_long_mms(monkeypatch, media_path, seconds=120)

    with pytest.raises(_ModelReached):
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert reached


def test_long_media_plain_draft_refusal_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxweave import pipeline

    vtt_path, media_path = _long_ja_episode(tmp_path, "WEBVTT\n\nあ\n\nい\n".encode())
    reached = _configure_long_mms(monkeypatch, media_path, seconds=120)

    with pytest.raises(DpRouteHintsInvalid) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert caught.value.failure.detail_code == "hint-shape"
    assert "VOXWEAVE_CTC_MAX_DP_FRAMES" in str(caught.value)
    assert not reached


# -- VTT decode classification ----------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "exception_type"),
    (
        # A UTF-8 BOM followed by bytes that are not UTF-8.
        (
            codecs.BOM_UTF8 + b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\xff\xfe\n",
            UnicodeDecodeError,
        ),
        # No BOM, and undecodable as UTF-8, GB18030 and CP1252 alike.
        (b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\x81\n", RuntimeError),
    ),
)
def test_undecodable_vtt_is_classified_as_an_encoding_failure(
    payload: bytes,
    exception_type: type[BaseException],
    tmp_path: Path,
) -> None:
    from voxweave import pipeline

    vtt_path, media_path = _long_ja_episode(tmp_path, payload)
    with pytest.raises(exception_type) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    failure = caught.value.failure  # type: ignore[attr-defined]
    assert failure.kind == "align-input-decode-invalid"
    assert failure.phase == "decode"
    assert failure.detail_code == "vtt-encoding"


@pytest.mark.parametrize(
    ("payload", "detail_code"),
    (
        (b"WEBVTT\n\n", "vtt-no-cues"),
        (b"[Script Info]\nTitle: x\n", "vtt-format-mismatch"),
    ),
)
def test_other_vtt_decode_failures_keep_their_detail(
    payload: bytes, detail_code: str, tmp_path: Path
) -> None:
    from voxweave import pipeline

    vtt_path, media_path = _long_ja_episode(tmp_path, payload)
    with pytest.raises(RuntimeError) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    failure = caught.value.failure  # type: ignore[attr-defined]
    assert failure.kind == "align-input-decode-invalid"
    assert failure.detail_code == detail_code


# -- oracle dependency gate --------------------------------------------------------


@pytest.fixture
def oracle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "p6_oracle_dependency_gate_under_test", ORACLE_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    return module


def _source(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("relative", "text", "expected"),
    (
        ("voxweave/align_dp_safety.py", "from voxweave import pipeline\n", None),
        ("voxweave/align_dp_safety.py", "from . import pipeline\n", None),
        ("voxweave/align_dp_safety.py", "from .pipeline import align\n", None),
        ("voxweave/core/align_compare.py", "from .. import pipeline\n", None),
        ("voxweave/core/align_compare.py", "from ..pipeline import align\n", None),
        (
            "voxweave/core/align_compare.py",
            "from . import finalizer\n",
            "voxweave.core.finalizer",
        ),
        (
            "voxweave/core/__init__.py",
            "from . import finalizer\n",
            "voxweave.core.finalizer",
        ),
        ("voxweave/align_dp_safety.py", "import voxweave.pipeline as p\n", None),
    ),
)
def test_dependency_gate_resolves_package_and_relative_imports(
    oracle: Any, tmp_path: Path, relative: str, text: str, expected: str | None
) -> None:
    path = _source(tmp_path, relative, text)
    assert (expected or "voxweave.pipeline") in oracle._imports(path)


def test_dependency_gate_keeps_plain_absolute_imports(
    oracle: Any, tmp_path: Path
) -> None:
    path = _source(
        tmp_path,
        "voxweave/align_dp_safety.py",
        "import math\nfrom voxweave.align_failures import CanonicalFailure\n",
    )
    observed = oracle._imports(path)
    assert {"math", "voxweave.align_failures"} <= observed
    assert "voxweave.pipeline" not in observed


def test_dependency_gate_rejects_a_relative_import_beyond_the_top_package(
    oracle: Any, tmp_path: Path
) -> None:
    path = _source(tmp_path, "voxweave/align_dp_safety.py", "from ... import x\n")
    with pytest.raises(oracle.OracleInvalid):
        oracle._imports(path)


def test_dependency_gate_reports_a_package_form_layering_violation(
    oracle: Any, tmp_path: Path
) -> None:
    for relative in (
        "voxweave/align_evidence_core.py",
        "voxweave/core/align_compare.py",
        "voxweave/reference_projector.py",
        "voxweave/episode_transaction.py",
    ):
        _source(tmp_path, relative, "import math\n")
    _source(tmp_path, "voxweave/align_dp_safety.py", "from voxweave import backend\n")
    assert oracle._check_dependencies() == [
        "dependency violation in voxweave/align_dp_safety.py: ['voxweave.backend']"
    ]


def test_dependency_gate_is_clean_on_the_checked_in_sources() -> None:
    spec = importlib.util.spec_from_file_location(
        "p6_oracle_dependency_gate_live", ORACLE_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._check_dependencies() == []
