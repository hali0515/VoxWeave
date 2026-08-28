from __future__ import annotations

import codecs
import importlib.util
import json
import threading
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


class _PhysicalCallReached(RuntimeError):
    pass


def _load_oracle_runner() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "p6_oracle.py"
    spec = importlib.util.spec_from_file_location("p6_oracle_vector_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_test_evidence_must_name_the_exact_detail() -> None:
    oracle = _load_oracle_runner()
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "calibration" / "p6-oracle" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    row = manifest["failure_registry_coverage"]["qwen-route-invalid"]["no-route-source"]
    row["evidence"] = [
        "tests/test_p6_oracle_vectors.py::test_snapshot_decodes_utf32_bom_without_a_phantom_header"
    ]
    failures = oracle._check_failure_registry_coverage(manifest)
    assert (
        "reachable registry row has no exact evidence: "
        "qwen-route-invalid/no-route-source"
    ) in failures


@pytest.mark.parametrize(
    ("shot_changes", "sing_spans", "detail_code"),
    (
        (["not-a-number"], None, "shot-shape"),
        (None, ["not-a-span"], "sing-shape"),
    ),
)
def test_finalizer_evidence_shape_failures_keep_their_exact_domain(
    shot_changes: object,
    sing_spans: object,
    detail_code: str,
) -> None:
    from voxweave.align_inputs import resolve_finalize_evidence

    resolution = resolve_finalize_evidence(
        shot_changes=shot_changes,
        sing_spans=sing_spans,
    )
    assert resolution.shots is None
    assert resolution.sing_spans is None
    assert resolution.status.kind == "invalid"
    assert resolution.status.detail_code == detail_code


@pytest.mark.parametrize(
    ("terminal", "valid", "root_error"),
    tuple(
        (terminal, valid, root_error)
        for terminal in (
            "fixed-point",
            "cycle-adoption",
            "budget-exhausted",
            "unknown-terminal",
        )
        for valid in (False, True)
        for root_error in (False, True)
    ),
)
def test_finalizer_terminal_table_precedes_root_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
    valid: bool,
    root_error: bool,
) -> None:
    from tests.test_p6_align_candidates import _evaluated
    from voxweave.core import authority, finalizer

    original_finalize = finalizer.finalize
    root_calls = 0

    def inject_terminal(*args: Any, **kwargs: Any) -> Any:
        result = original_finalize(*args, **kwargs)
        return replace(
            result,
            report=replace(result.report, terminal=terminal),
            trace=replace(result.trace, terminal=terminal),
            valid=valid,
        )

    def observed_roots(*args: Any, **kwargs: Any) -> tuple[str, ...]:
        nonlocal root_calls
        root_calls += 1
        return ("injected-root-error",) if root_error else ()

    monkeypatch.setattr(finalizer, "finalize", inject_terminal)
    monkeypatch.setattr(authority, "check_roots", observed_roots)
    _context, result = _evaluated(tmp_path, shadow_requested=True)

    if terminal in ("fixed-point", "cycle-adoption") and valid:
        assert root_calls == 1
        if root_error:
            assert result.v2_status.kind == "invalid"
            assert result.v2_status.failure is not None
            assert (
                result.v2_status.failure.kind,
                result.v2_status.failure.phase,
                result.v2_status.failure.detail_code,
            ) == (
                "fresh-authority-invalid",
                "w1-admission",
                "w1-root-event",
            )
        elif terminal == "fixed-point":
            assert result.v2_status.kind == "valid"
            assert result.v2_status.failure is None
        return

    assert root_calls == 0
    assert result.v2_status.kind == "invalid"
    assert result.v2_status.failure is not None
    if terminal == "budget-exhausted" and valid is False:
        assert (
            result.v2_status.failure.kind,
            result.v2_status.failure.phase,
            result.v2_status.failure.detail_code,
        ) == (
            "finalizer-budget-exhausted",
            "w1-finalizer",
            "sweep-budget",
        )
    else:
        assert (
            result.v2_status.failure.kind,
            result.v2_status.failure.phase,
            result.v2_status.failure.detail_code,
        ) == (
            "finalizer-output-invalid",
            "w1-finalizer",
            "terminal-validity",
        )


@pytest.mark.parametrize("root_error", (False, True))
def test_finalizer_terminal_disagreement_stops_before_root_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_error: bool,
) -> None:
    from tests.test_p6_align_candidates import _evaluated
    from voxweave.core import authority, finalizer

    original_finalize = finalizer.finalize
    root_calls = 0

    def inject_disagreement(*args: Any, **kwargs: Any) -> Any:
        result = original_finalize(*args, **kwargs)
        return replace(
            result,
            report=replace(result.report, terminal="fixed-point"),
            trace=replace(result.trace, terminal="cycle-adoption"),
            valid=True,
        )

    def observed_roots(*args: Any, **kwargs: Any) -> tuple[str, ...]:
        nonlocal root_calls
        root_calls += 1
        return ("injected-root-error",) if root_error else ()

    monkeypatch.setattr(finalizer, "finalize", inject_disagreement)
    monkeypatch.setattr(authority, "check_roots", observed_roots)
    _context, result = _evaluated(tmp_path, shadow_requested=True)

    assert root_calls == 0
    assert result.v2_status.kind == "invalid"
    assert result.v2_status.failure is not None
    assert (
        result.v2_status.failure.kind,
        result.v2_status.failure.phase,
        result.v2_status.failure.detail_code,
    ) == (
        "finalizer-output-invalid",
        "w1-finalizer",
        "terminal-validity",
    )


def test_finalizer_canonical_text_fallback_is_a_footprint_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_p6_align_candidates import _evaluated
    from voxweave.core import finalizer
    from voxweave.core.partition_check import ReportTag

    original_factory = finalizer.phase1_from_fresh_alignment

    def inject_fallback(*args: Any, **kwargs: Any) -> Any:
        stream = original_factory(*args, **kwargs)
        first = stream.cues[0]
        fallback = ReportTag(
            kind="canonical-text-fallback",
            cue_index=0,
            evidence={"reason": "footprint-mismatch"},
        )
        changed = replace(first, reports=first.reports + (fallback,))
        return replace(stream, cues=(changed, *stream.cues[1:]))

    monkeypatch.setattr(finalizer, "phase1_from_fresh_alignment", inject_fallback)
    _context, result = _evaluated(tmp_path, shadow_requested=True)
    assert result.v2_status.kind == "invalid"
    assert result.v2_status.failure is not None
    assert result.v2_status.failure.kind == "finalizer-output-invalid"
    assert result.v2_status.failure.phase == "w1-finalizer"
    assert result.v2_status.failure.detail_code == "footprint-fallback"


def test_semantic_registry_digest_corruption_is_typed_before_primitive_relations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_p6_align_candidates import _evaluated
    from voxweave.core import align_compare

    monkeypatch.setattr(align_compare, "ALIGN_DELTA_REGISTRY_SHA256", "0" * 64)
    _context, result = _evaluated(tmp_path, shadow_requested=True)
    assert result.v2_status.kind == "invalid"
    assert result.v2_status.failure is not None
    assert result.v2_status.failure.kind == "align-delta-invalid"
    assert result.v2_status.failure.phase == "semantic-comparison"
    assert result.v2_status.failure.detail_code == "registry-digest"


@pytest.mark.parametrize(
    "detail_code",
    (
        "cue-source-map",
        "unit-coverage",
        "vtt-projection",
        "json-projection",
        "candidate-family-manifest",
    ),
)
def test_selected_projection_failures_keep_their_exact_domain(
    detail_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_p6_align_candidates import _evaluated
    from voxweave import candidate_encoder

    context, result = _evaluated(tmp_path)
    selected = candidate_encoder.select_align_candidate(
        context,
        candidate_encoder.encode_align_candidates(context, result),
    )
    record = candidate_encoder._ENCODED[id(selected)]
    if detail_code == "cue-source-map":
        source = record.projection_inputs.source_blocks[0]
        inputs = replace(
            record.projection_inputs,
            source_blocks=(replace(source, source_index=source.source_index + 1),),
        )
        candidate_encoder._ENCODED[id(selected)] = replace(
            record, projection_inputs=inputs
        )
    elif detail_code == "unit-coverage":
        delivery = replace(record.delivery, word_segments=())
        candidate_encoder._ENCODED[id(selected)] = replace(record, delivery=delivery)
    elif detail_code in ("vtt-projection", "json-projection"):
        original_reference = candidate_encoder.reference_align_projection

        def corrupt_reference(*args: Any, **kwargs: Any) -> Any:
            projection = original_reference(*args, **kwargs)
            if detail_code == "vtt-projection":
                return replace(projection, vtt_bytes=projection.vtt_bytes + b"x")
            return replace(
                projection, main_json_bytes=projection.main_json_bytes + b"x"
            )

        monkeypatch.setattr(
            candidate_encoder, "reference_align_projection", corrupt_reference
        )
    else:
        object.__setattr__(selected, "engine_family", "boundary-v2")

    with pytest.raises(candidate_encoder.SelectedRenderError) as caught:
        candidate_encoder.verify_selected_align_projection(context, result, selected)
    assert caught.value.failure.kind == "selected-render-invalid"
    assert caught.value.failure.phase == "renderer"
    assert caught.value.failure.detail_code == detail_code


def _episode(
    tmp_path: Path,
    *,
    language: str,
    texts: tuple[str, ...] = ("FIRST", "SECOND"),
    timed: bool,
) -> tuple[Path, Path, Path]:
    vtt_path = tmp_path / "episode.vtt"
    json_path = tmp_path / "episode.json"
    media_path = tmp_path / "episode.wav"
    if timed:
        rows = ["WEBVTT", ""]
        for index, text in enumerate(texts):
            rows.extend(
                (
                    f"00:00:{index * 2:02d}.000 --> 00:00:{index * 2 + 1:02d}.000",
                    text,
                    "",
                )
            )
    else:
        rows = ["WEBVTT", ""]
        for text in texts:
            rows.extend((text, ""))
    vtt_path.write_text("\n".join(rows), encoding="utf-8")
    json_path.write_text(json.dumps({"language": language}), encoding="utf-8")
    media_path.write_bytes(b"synthetic-media")
    return vtt_path, json_path, media_path


@pytest.mark.parametrize(
    ("route", "language"),
    (("ctc", "en"), ("mms", "ja")),
)
def test_public_direct_full_pass_accepts_untimed_vtt_in_lexical_order(
    route: str,
    language: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import backend, config, pipeline

    vtt_path, _json_path, media_path = _episode(
        tmp_path, language=language, timed=False
    )
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media_path
    )
    monkeypatch.setattr(backend, "uses_mms", lambda _iso: route == "mms")
    monkeypatch.setattr(
        config,
        "align_model_for",
        lambda _iso: None if route == "mms" else "synthetic-ctc",
    )
    observed: dict[str, Any] = {}

    def stop_at_call(
        _wav: Path,
        texts: list[str],
        _iso: str,
        *_args: object,
        bounds: object = None,
        **_kwargs: object,
    ) -> list[list[dict[str, Any]]]:
        observed.update(texts=texts, bounds=bounds)
        raise _PhysicalCallReached

    target = "align_blocks_full_mms" if route == "mms" else "align_blocks_full_ctc"
    monkeypatch.setattr(backend, target, stop_at_call)
    with pytest.raises(_PhysicalCallReached):
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert observed == {"texts": ["FIRST", "SECOND"], "bounds": [None, None]}


@pytest.mark.parametrize(
    ("bom", "encoding"),
    (
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
    ),
)
def test_snapshot_decodes_utf32_bom_without_a_phantom_header(
    bom: bytes,
    encoding: str,
) -> None:
    from voxweave.align_snapshot import decode_subtitle_snapshot

    document = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n"
    snapshot = decode_subtitle_snapshot("episode.vtt", bom + document.encode(encoding))
    assert [block.text for block in snapshot.blocks] == ["hello"]


def test_ctc_safe_over_budget_uses_real_planner_and_exact_physical_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_common, align_ctc

    sample_rate = align_ctc.CTC_AUDIO_SR
    waveform = np.zeros(40 * sample_rate, dtype=np.float32)
    monkeypatch.setattr(align_ctc, "_load_mono", lambda *_args, **_kwargs: waveform)
    monkeypatch.setattr(
        align_ctc,
        "_get_ctc_aligner",
        lambda *_args, **_kwargs: SimpleNamespace(sr=sample_rate),
    )
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    monkeypatch.setattr(align_ctc, "_empty_cache", lambda: None)
    calls: list[tuple[int, tuple[str, ...]]] = []
    observed: list[tuple[tuple[int, ...], dict[str, Any]]] = []

    def fake_pass(
        _aligner: object,
        wav: np.ndarray,
        texts: list[str],
        _nospace: bool,
        _iso: str,
        *,
        speech_spans: object = None,
        _raw_result_observer: Any = None,
        _raw_original_observer: Any = None,
        _backend_invoker: Any = None,
    ) -> list[list[dict[str, Any]]]:
        del speech_spans, _backend_invoker
        calls.append((len(wav), tuple(texts)))
        flat = [
            {"text": text, "start": float(index), "end": float(index) + 0.5}
            for index, text in enumerate(texts)
        ]
        if _raw_original_observer is not None:
            _raw_original_observer(None)
        if _raw_result_observer is not None:
            _raw_result_observer(flat)
        return [[unit] for unit in flat]

    def observe(
        _post: list[dict[str, Any]],
        _original: list[dict[str, Any]] | None,
        sources: tuple[int, ...],
        _legacy_origin: float,
        **geometry: Any,
    ) -> None:
        observed.append((sources, geometry))

    monkeypatch.setattr(align_ctc, "_ctc_full_pass", fake_pass)
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    result = align_ctc.align_blocks_full_ctc(
        wav_path,
        ["A", "B", "C", "D"],
        "en",
        "synthetic",
        bounds=((0.0, 8.0), (10.0, 18.0), (22.0, 30.0), (32.0, 39.0)),
        _raw_call_observer=observe,
    )

    assert calls == [
        (20 * sample_rate, ("A", "B")),
        (20 * sample_rate, ("C", "D")),
    ]
    assert [sources for sources, _geometry in observed] == [(0, 1), (2, 3)]
    assert [
        (
            geometry["audio_sample_start"],
            geometry["audio_sample_end"],
            geometry["sample_rate"],
            geometry["sample_count"],
        )
        for _sources, geometry in observed
    ] == [
        (0, 20 * sample_rate, sample_rate, 40 * sample_rate),
        (20 * sample_rate, 40 * sample_rate, sample_rate, 40 * sample_rate),
    ]
    assert len(result) == 4


def test_one_overlong_mms_cue_is_refused_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_common, align_mms
    from voxweave.align_dp_safety import DpRouteHintsInvalid

    waveform = np.zeros(40 * align_mms.MMS_SR, dtype=np.float32)
    monkeypatch.setattr(align_mms, "_read_wav_16k", lambda _path: waveform)
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    monkeypatch.setattr(
        align_mms,
        "_mms_emit_units",
        lambda *_args, **_kwargs: pytest.fail("overlong cue reached model work"),
    )
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with pytest.raises(DpRouteHintsInvalid) as caught:
        align_mms.align_blocks_full_mms(
            wav_path,
            ["A"],
            "ja",
            bounds=((0.0, 39.0),),
        )
    assert caught.value.failure.detail_code == "crop-over-budget"


def test_overlapping_planner_crops_are_refused_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_common, align_mms, chunking
    from voxweave.align_dp_safety import DpRouteHintsInvalid

    waveform = np.zeros(40 * align_mms.MMS_SR, dtype=np.float32)
    monkeypatch.setattr(align_mms, "_read_wav_16k", lambda _path: waveform)
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(align_common, "CTC_DP_CHUNK_FRAC", 0.8)
    monkeypatch.setattr(
        chunking,
        "plan_dp_chunks",
        lambda *_args, **_kwargs: [
            {"lo": 0, "hi": 1, "start": 0.0, "end": 18.0},
            {"lo": 1, "hi": 2, "start": 17.0, "end": 35.0},
        ],
    )
    monkeypatch.setattr(
        align_mms,
        "_mms_emit_units",
        lambda *_args, **_kwargs: pytest.fail("overlapping plan reached model work"),
    )
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with pytest.raises(DpRouteHintsInvalid) as caught:
        align_mms.align_blocks_full_mms(
            wav_path,
            ["A", "B"],
            "ja",
            bounds=((0.0, 8.0), (22.0, 39.0)),
        )
    assert caught.value.failure.detail_code == "crop-geometry"


def test_explicit_media_wins_over_a_same_stem_sibling_and_survives_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import pipeline

    vtt_path, _json_path, sibling = _episode(tmp_path, language="en", timed=True)
    sibling.write_bytes(b"different sibling content")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    explicit = first_root / "renamed-source.wav"
    explicit.write_bytes(b"stable explicit content")
    seen: list[Path] = []

    def stop(media: Path, **_kwargs: object) -> Path:
        seen.append(media)
        raise _PhysicalCallReached

    monkeypatch.setattr(pipeline, "_prepare_16k_for_align", stop)
    with pytest.raises(_PhysicalCallReached):
        pipeline.align(vtt_path, media_path=explicit, separate=False)
    relocated = second_root / "another-name.wav"
    explicit.replace(relocated)
    with pytest.raises(_PhysicalCallReached):
        pipeline.align(vtt_path, media_path=relocated, separate=False)
    assert seen == [first_root / "renamed-source.wav", relocated]


def test_missing_explicit_media_is_a_canonical_pre_backend_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import pipeline

    vtt_path, _json_path, sibling = _episode(tmp_path, language="en", timed=True)
    sibling.unlink()
    monkeypatch.setattr(
        pipeline,
        "_prepare_16k_for_align",
        lambda *_args, **_kwargs: pytest.fail("missing explicit media reached decode"),
    )
    with pytest.raises(FileNotFoundError) as caught:
        pipeline.align(
            vtt_path,
            media_path=tmp_path / "missing.wav",
            separate=False,
        )
    assert caught.value.failure.kind == "media-identity-invalid"
    assert caught.value.failure.detail_code == "media-not-found"


def test_sibling_media_shadowing_uses_closed_extension_precedence(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from voxweave import pipeline

    reference = tmp_path / "episode.part.vtt"
    reference.write_bytes(b"WEBVTT\n")
    wav = tmp_path / "episode.part.wav"
    mkv = tmp_path / "episode.part.mkv"
    wav.write_bytes(b"wav")
    mkv.write_bytes(b"mkv")
    with caplog.at_level("WARNING", logger="voxweave"):
        selected = pipeline._find_sibling_media(reference)
    assert selected == mkv
    assert any(
        "multiple sibling media files" in record.message for record in caplog.records
    )


def _qwen_episode(
    tmp_path: Path,
    *,
    timed: bool,
    word_segments: list[dict[str, Any]] | None,
) -> tuple[Path, Path]:
    vtt_path, json_path, media_path = _episode(
        tmp_path,
        language="zh",
        texts=("你好",),
        timed=timed,
    )
    json_path.write_text(
        json.dumps(
            {
                "language": "zh",
                "word_segments": word_segments or [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with wave.open(str(media_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 32_000)
    return vtt_path, media_path


def _force_qwen_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from voxweave import backend, config

    monkeypatch.setattr(backend, "uses_mms", lambda _iso: False)
    monkeypatch.setattr(config, "align_model_for", lambda _iso: None)


def test_qwen_no_route_source_is_canonical_and_pre_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import pipeline

    vtt_path, media_path = _qwen_episode(tmp_path, timed=False, word_segments=None)
    _force_qwen_route(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_prepare_16k_for_align",
        lambda *_args, **_kwargs: pytest.fail("no-route source reached audio work"),
    )
    with pytest.raises(RuntimeError) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert caught.value.failure.kind == "qwen-route-invalid"
    assert caught.value.failure.detail_code == "no-route-source"


def test_qwen_all_crops_none_is_canonical_and_pre_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import pipeline, realign

    vtt_path, media_path = _qwen_episode(tmp_path, timed=True, word_segments=None)
    _force_qwen_route(monkeypatch)
    monkeypatch.setattr(realign, "crop_blocks", lambda _spans: [None])
    monkeypatch.setattr(
        pipeline,
        "_prepare_16k_for_align",
        lambda *_args, **_kwargs: pytest.fail("all-crops-none reached audio work"),
    )
    with pytest.raises(RuntimeError) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert caught.value.failure.kind == "qwen-route-invalid"
    assert caught.value.failure.detail_code == "all-crops-none"


def test_qwen_all_aligned_lists_empty_is_canonical_before_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import backend, pipeline, realign

    vtt_path, media_path = _qwen_episode(tmp_path, timed=True, word_segments=None)
    _force_qwen_route(monkeypatch)
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media_path
    )
    monkeypatch.setattr(realign, "crop_blocks", lambda _spans: [(0.0, 1.0)])
    monkeypatch.setattr(backend, "align_text", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(backend, "release", lambda: None)
    with pytest.raises(RuntimeError) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert caught.value.failure.kind == "no-aligned-units"
    assert caught.value.failure.detail_code == "all-block-units-empty"


@pytest.mark.parametrize("source_mode", ("vtt", "word-segments"))
@pytest.mark.parametrize(
    ("crop", "exception_type", "detail_code"),
    (
        ((float("nan"), 1.0), ValueError, "sample-start-index"),
        ((float("inf"), 1.0), OverflowError, "sample-start-index"),
        ((0.0, float("nan")), ValueError, "sample-end-index"),
        ((0.0, float("inf")), OverflowError, "sample-end-index"),
    ),
)
def test_qwen_nonfinite_route_bounds_from_both_sources_are_canonical_pre_model(
    source_mode: str,
    crop: tuple[float, float],
    exception_type: type[BaseException],
    detail_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import backend, pipeline, realign

    words = (
        None if source_mode == "vtt" else [{"text": "你好", "start": 0.0, "end": 1.0}]
    )
    vtt_path, media_path = _qwen_episode(
        tmp_path,
        timed=source_mode == "vtt",
        word_segments=words,
    )
    _force_qwen_route(monkeypatch)
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media_path
    )
    monkeypatch.setattr(realign, "crop_blocks", lambda _spans: [crop])
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda *_args, **_kwargs: pytest.fail("invalid route bound reached model work"),
    )
    monkeypatch.setattr(backend, "release", lambda: None)
    with pytest.raises(exception_type) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert caught.value.failure.kind == "qwen-window-operation-failed"
    assert caught.value.failure.detail_code == detail_code


@pytest.mark.parametrize("source_mode", ("timestamp", "word-segments"))
@pytest.mark.parametrize("position", ("start", "end"))
@pytest.mark.parametrize(
    "value_kind",
    ("missing", "string", "bool", "nan", "positive-infinity", "negative-infinity"),
)
def test_qwen_special_route_bound_matrix_preserves_real_statement_order(
    source_mode: str,
    position: str,
    value_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from voxweave import align_snapshot, backend, pipeline, realign

    missing = object()
    special: object
    if value_kind == "missing":
        special = missing
    elif value_kind == "string":
        special = "odd"
    elif value_kind == "bool":
        special = True
    elif value_kind == "nan":
        special = float("nan")
    elif value_kind == "positive-infinity":
        special = float("inf")
    else:
        special = float("-inf")

    start: object = 0.0
    end: object = 1.0
    if position == "start":
        start = None if special is missing else special
    else:
        end = None if special is missing else special

    words: list[dict[str, Any]] | None = None
    if source_mode == "word-segments":
        word = {"text": "你好"}
        word[position] = special
        if special is missing:
            del word[position]
        word["end" if position == "start" else "start"] = (
            1.0 if position == "start" else 0.0
        )
        words = [word]
    vtt_path, media_path = _qwen_episode(
        tmp_path,
        timed=source_mode == "timestamp",
        word_segments=words,
    )
    _force_qwen_route(monkeypatch)

    if source_mode == "timestamp":
        original_decode = align_snapshot.decode_align_snapshot

        def decode_with_special_bound(*args: Any, **kwargs: Any) -> Any:
            snapshot = original_decode(*args, **kwargs)
            return replace(
                snapshot,
                route_bounds=(align_snapshot.RouteBound(0, start, end),),
                qwen_delivery_order=(0,),
            )

        monkeypatch.setattr(
            align_snapshot, "decode_align_snapshot", decode_with_special_bound
        )

    route_results: list[list[tuple[object, object] | None]] = []
    crop_results: list[list[tuple[object, object] | None]] = []
    slice_inputs: list[tuple[object, object]] = []
    slice_outputs: list[Path] = []
    prepare_calls = 0
    backend_calls = 0
    original_route = realign.route_blocks
    original_crop = realign.crop_blocks
    original_slice = pipeline.slice_wav

    def observe_route(*args: Any, **kwargs: Any) -> Any:
        result = original_route(*args, **kwargs)
        route_results.append(result)
        return result

    def observe_crop(*args: Any, **kwargs: Any) -> Any:
        result = original_crop(*args, **kwargs)
        crop_results.append(result)
        return result

    def observe_slice(
        wav: Path, crop_start: object, crop_end: object, **kwargs: Any
    ) -> Path:
        slice_inputs.append((crop_start, crop_end))
        output = original_slice(wav, crop_start, crop_end, **kwargs)
        slice_outputs.append(output)
        return output

    def prepare(*_args: Any, **_kwargs: Any) -> Path:
        nonlocal prepare_calls
        prepare_calls += 1
        return media_path

    def stop_at_backend(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        nonlocal backend_calls
        backend_calls += 1
        raise _PhysicalCallReached

    monkeypatch.setattr(realign, "route_blocks", observe_route)
    monkeypatch.setattr(realign, "crop_blocks", observe_crop)
    monkeypatch.setattr(pipeline, "slice_wav", observe_slice)
    monkeypatch.setattr(pipeline, "_prepare_16k_for_align", prepare)
    monkeypatch.setattr(backend, "align_text", stop_at_backend)
    monkeypatch.setattr(backend, "release", lambda: None)

    if value_kind == "missing":
        expected_type: type[BaseException] = (
            RuntimeError if source_mode == "timestamp" else KeyError
        )
    elif value_kind == "string":
        expected_type = TypeError
    elif value_kind == "positive-infinity":
        expected_type = OverflowError
    elif value_kind == "nan" and position == "end":
        expected_type = ValueError
    else:
        expected_type = _PhysicalCallReached

    with pytest.raises(expected_type) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)

    if value_kind == "missing" and source_mode == "timestamp":
        assert str(caught.value) == (
            "episode.json has no word_segments and VTT has no timestamps; "
            "cannot route audio windows"
        )
        assert caught.value.failure.kind == "qwen-route-invalid"
        assert caught.value.failure.detail_code == "no-route-source"
        assert route_results == crop_results == []
        assert prepare_calls == backend_calls == 0
        assert slice_inputs == []
    elif value_kind == "missing":
        assert caught.value.args == (position,)
        assert caught.value.failure.kind == "qwen-window-operation-failed"
        assert caught.value.failure.phase == "route-plan"
        assert caught.value.failure.detail_code == "route-bound-access"
        assert route_results == crop_results == []
        assert prepare_calls == backend_calls == 0
        assert slice_inputs == []
    elif value_kind == "string":
        expected_message = (
            "unsupported operand type(s) for -: 'str' and 'float'"
            if position == "start"
            else 'can only concatenate str (not "float") to str'
        )
        assert str(caught.value) == expected_message
        assert caught.value.failure.kind == "qwen-window-operation-failed"
        assert caught.value.failure.phase == "route-plan"
        assert caught.value.failure.detail_code == "route-bound-arithmetic"
        assert len(route_results) == 1
        assert crop_results == []
        assert prepare_calls == backend_calls == 0
        assert slice_inputs == []
    else:
        assert len(route_results) == len(crop_results) == 1
        assert len(crop_results[0]) == 1
        crop = crop_results[0][0]
        assert crop is not None
        crop_start, crop_end = crop
        if value_kind == "bool":
            assert (crop_start, crop_end) == (
                (0.9, 1.1) if position == "start" else (0.0, 1.1)
            )
        elif value_kind == "nan":
            if position == "start":
                assert (crop_start, crop_end) == (0.0, 1.1)
            else:
                assert crop_start == 0.0
                assert crop_end != crop_end
        elif value_kind == "positive-infinity":
            assert (crop_start, crop_end) == (
                (float("inf"), float("inf"))
                if position == "start"
                else (0.0, float("inf"))
            )
        else:
            assert (crop_start, crop_end) == (
                (0.0, 1.1) if position == "start" else (0.0, 0.1)
            )
        assert slice_inputs == [(crop_start, crop_end)]
        assert prepare_calls == 1
        if expected_type in (ValueError, OverflowError):
            assert str(caught.value) == (
                "cannot convert float NaN to integer"
                if expected_type is ValueError
                else "cannot convert float infinity to integer"
            )
            assert caught.value.failure.kind == "qwen-window-operation-failed"
            assert caught.value.failure.phase == "qwen-window"
            assert caught.value.failure.detail_code == f"sample-{position}-index"
            assert backend_calls == 0
            assert slice_outputs == []
        else:
            assert backend_calls == 1
            assert len(slice_outputs) == 1
    assert all(not path.exists() for path in slice_outputs)


@pytest.mark.parametrize(
    ("edge", "detail_code"),
    (
        ("open", "sample-open"),
        ("seek", "sample-seek-read"),
        ("read", "sample-seek-read"),
        ("temp-create", "sample-temp-create"),
        ("write", "sample-write"),
    ),
)
def test_qwen_window_io_failures_are_canonical_pre_model_and_cleanup_owned_files(
    edge: str,
    detail_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import backend, chunking, pipeline

    vtt_path, media_path = _qwen_episode(tmp_path, timed=True, word_segments=None)
    _force_qwen_route(monkeypatch)
    monkeypatch.setattr(
        pipeline, "_prepare_16k_for_align", lambda *_args, **_kwargs: media_path
    )
    backend_calls = 0

    def forbid_backend(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        nonlocal backend_calls
        backend_calls += 1
        pytest.fail("failed Qwen window reached model work")

    monkeypatch.setattr(backend, "align_text", forbid_backend)
    monkeypatch.setattr(backend, "release", lambda: None)

    closed = False

    class FakeSoundFile:
        samplerate = 16_000

        def __len__(self) -> int:
            return 32_000

        def seek(self, _offset: int) -> None:
            if edge == "seek":
                raise OSError("seek failed")

        def read(self, count: int, *, dtype: str) -> np.ndarray:
            assert dtype == "float32"
            if edge == "read":
                raise OSError("read failed")
            return np.zeros(count, dtype=np.float32)

        def close(self) -> None:
            nonlocal closed
            closed = True

    if edge == "open":
        monkeypatch.setattr(
            chunking.sf,
            "SoundFile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("open failed")),
        )
    else:
        monkeypatch.setattr(
            chunking.sf, "SoundFile", lambda *_args, **_kwargs: FakeSoundFile()
        )

    original_mkstemp = chunking.tempfile.mkstemp
    if edge == "temp-create":
        monkeypatch.setattr(
            chunking.tempfile,
            "mkstemp",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("temp-create failed")
            ),
        )
    elif edge == "write":

        def controlled_mkstemp(*, suffix: str, prefix: str) -> tuple[int, str]:
            return original_mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=tmp_path,
            )

        monkeypatch.setattr(chunking.tempfile, "mkstemp", controlled_mkstemp)
        monkeypatch.setattr(
            chunking.sf,
            "write",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
        )

    with pytest.raises(OSError) as caught:
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert str(caught.value) == f"{edge} failed"
    assert caught.value.failure.kind == "qwen-window-operation-failed"
    assert caught.value.failure.phase == "qwen-window"
    assert caught.value.failure.detail_code == detail_code
    assert backend_calls == 0
    assert closed is (edge != "open")
    assert not list(tmp_path.glob("voxweave_chunk_*"))


def test_final_evidence_bind_classifies_independent_projection_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_p6_align_evidence import _verified
    from voxweave import align_evidence

    context, result, verified, acquisition, strict, policy, profile, evidence_status = (
        _verified(tmp_path)
    )
    frozen_inputs = align_evidence._fresh_evidence_inputs(context, acquisition)
    monkeypatch.setattr(
        align_evidence,
        "_fresh_evidence_inputs",
        lambda *_args, **_kwargs: frozen_inputs,
    )
    object.__setattr__(acquisition, "receipt_digest", "0" * 64)
    with pytest.raises(align_evidence.EvidenceBindingError) as caught:
        align_evidence.bind_align_evidence(
            context,
            result.evidence_core,
            acquisition=acquisition,
            strict_input_status=strict,
            v2_policy_status=policy,
            profile_status=profile,
            evidence_status=evidence_status,
            engine_family=verified.engine_family,
            vtt_sha256=verified.vtt_sha256,
            main_json_sha256=verified.main_json_sha256,
        )
    assert caught.value.failure.kind == "final-evidence-invalid"
    assert caught.value.failure.phase == "evidence-bind"
    assert caught.value.failure.detail_code == "independent-projection"


def test_final_evidence_bind_classifies_closed_schema_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_p6_align_evidence import _verified
    from voxweave import align_evidence

    context, result, verified, acquisition, strict, policy, profile, evidence_status = (
        _verified(tmp_path)
    )
    monkeypatch.setattr(
        align_evidence,
        "_validate_evidence_value",
        lambda _value: (_ for _ in ()).throw(ValueError("schema rejected")),
    )
    with pytest.raises(align_evidence.EvidenceBindingError) as caught:
        align_evidence.bind_align_evidence(
            context,
            result.evidence_core,
            acquisition=acquisition,
            strict_input_status=strict,
            v2_policy_status=policy,
            profile_status=profile,
            evidence_status=evidence_status,
            engine_family=verified.engine_family,
            vtt_sha256=verified.vtt_sha256,
            main_json_sha256=verified.main_json_sha256,
        )
    assert caught.value.failure.kind == "final-evidence-invalid"
    assert caught.value.failure.phase == "evidence-bind"
    assert caught.value.failure.detail_code == "closed-schema"


def _commit_simple(
    episode_path: Path,
    json_path: Path,
    vtt_path: Path,
    *,
    expected_json: Any,
    expected_vtt: Any,
    main_json_bytes: bytes,
    vtt_bytes: bytes,
) -> Any:
    from voxweave import episode_transaction

    return episode_transaction.commit_primary_outputs(
        command="process",
        episode_path=episode_path,
        json_path=json_path,
        vtt_path=vtt_path,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        main_json_bytes=main_json_bytes,
        vtt_bytes=vtt_bytes,
    )


def test_primary_byte_generation_defines_same_byte_aba_as_the_same_generation(
    tmp_path: Path,
) -> None:
    from voxweave import episode_transaction

    episode = tmp_path / "episode.mkv"
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    episode.write_bytes(b"media")
    json_path.write_bytes(b"old-json")
    vtt_path.write_bytes(b"old-vtt")
    expected_json = episode_transaction.capture_file_generation(json_path)
    expected_vtt = episode_transaction.capture_file_generation(vtt_path)
    json_path.write_bytes(b"intermediate")
    json_path.write_bytes(b"old-json")
    receipt = _commit_simple(
        episode,
        json_path,
        vtt_path,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        main_json_bytes=b"new-json",
        vtt_bytes=b"new-vtt",
    )
    assert receipt.landed == (json_path, vtt_path)
    assert json_path.read_bytes() == b"new-json"
    assert vtt_path.read_bytes() == b"new-vtt"


def test_nonparticipating_editor_after_recheck_is_replaced_by_the_atomic_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import episode_transaction

    episode = tmp_path / "episode.mkv"
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    episode.write_bytes(b"media")
    json_path.write_bytes(b"old-json")
    vtt_path.write_bytes(b"old-vtt")
    expected_json = episode_transaction.capture_file_generation(json_path)
    expected_vtt = episode_transaction.capture_file_generation(vtt_path)

    def concurrent_edit(*_args: object, **_kwargs: object) -> None:
        json_path.write_bytes(b"foreign-json")
        vtt_path.write_bytes(b"foreign-vtt")

    monkeypatch.setattr(episode_transaction, "_consume_commit_role", concurrent_edit)
    receipt = _commit_simple(
        episode,
        json_path,
        vtt_path,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        main_json_bytes=b"new-json",
        vtt_bytes=b"new-vtt",
    )
    assert receipt.landed == (json_path, vtt_path)
    assert json_path.read_bytes() == b"new-json"
    assert vtt_path.read_bytes() == b"new-vtt"


def test_cooperating_writer_after_recheck_waits_and_then_stale_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import episode_transaction

    episode = tmp_path / "episode.mkv"
    json_path = tmp_path / "episode.json"
    vtt_path = tmp_path / "episode.vtt"
    episode.write_bytes(b"media")
    json_path.write_bytes(b"old-json")
    vtt_path.write_bytes(b"old-vtt")
    expected_json = episode_transaction.capture_file_generation(json_path)
    expected_vtt = episode_transaction.capture_file_generation(vtt_path)
    attempted = threading.Event()
    writer_errors: list[BaseException] = []
    writer: threading.Thread | None = None
    original_consume = episode_transaction._consume_commit_role

    def run_writer() -> None:
        attempted.set()
        try:
            _commit_simple(
                episode,
                json_path,
                vtt_path,
                expected_json=expected_json,
                expected_vtt=expected_vtt,
                main_json_bytes=b"writer-json",
                vtt_bytes=b"writer-vtt",
            )
        except BaseException as exc:
            writer_errors.append(exc)

    def start_writer(*args: object, **kwargs: object) -> None:
        nonlocal writer
        original_consume(*args, **kwargs)
        if threading.current_thread() is threading.main_thread():
            writer = threading.Thread(target=run_writer, name="cooperating-writer")
            writer.start()
            assert attempted.wait(timeout=2)

    monkeypatch.setattr(episode_transaction, "_consume_commit_role", start_writer)
    receipt = _commit_simple(
        episode,
        json_path,
        vtt_path,
        expected_json=expected_json,
        expected_vtt=expected_vtt,
        main_json_bytes=b"new-json",
        vtt_bytes=b"new-vtt",
    )
    assert writer is not None
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert receipt.landed == (json_path, vtt_path)
    assert len(writer_errors) == 1
    assert isinstance(writer_errors[0], episode_transaction.InputStaleError)
    assert json_path.read_bytes() == b"new-json"
    assert vtt_path.read_bytes() == b"new-vtt"


def test_transaction_stages_dotted_targets_in_each_target_directory(
    tmp_path: Path,
) -> None:
    from voxweave import episode_transaction

    json_root = tmp_path / "json-root"
    vtt_root = tmp_path / "vtt-root"
    json_root.mkdir()
    vtt_root.mkdir()
    episode = tmp_path / "episode.part.mkv"
    json_path = json_root / "episode.part.final.json"
    vtt_path = vtt_root / "episode.part.final.vtt"
    episode.write_bytes(b"media")
    json_path.write_bytes(b"old-json")
    vtt_path.write_bytes(b"old-vtt")
    receipt = _commit_simple(
        episode,
        json_path,
        vtt_path,
        expected_json=episode_transaction.capture_file_generation(json_path),
        expected_vtt=episode_transaction.capture_file_generation(vtt_path),
        main_json_bytes=b"new-json",
        vtt_bytes=b"new-vtt",
    )
    assert receipt.landed == (json_path, vtt_path)
    assert list(json_root.iterdir()) == [json_path]
    assert list(vtt_root.iterdir()) == [vtt_path]


def _sdh_inputs(tmp_path: Path) -> tuple[Path, Path]:
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"media")
    (tmp_path / "episode.json").write_bytes(b"json")
    (tmp_path / "episode.vtt").write_bytes(b"vtt")
    wav = tmp_path / "decoded.wav"
    wav.write_bytes(b"wav")
    return media, wav


@pytest.mark.parametrize("events", ([], [{"label": "door", "start": 1.0, "end": 1.2}]))
def test_sdh_success_and_empty_detection_commit_selected_sidecar(
    events: list[dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import pipeline, sdh

    media, wav = _sdh_inputs(tmp_path)
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(sdh, "detect_events", lambda *_args, **_kwargs: events)
    monkeypatch.setattr(sdh, "fit_events_to_gaps", lambda found, _cues: found)
    monkeypatch.setattr(sdh, "render_sdh_vtt", lambda _cues, found: repr(found))
    result = pipeline._write_sdh_sidecar(media, (), pipeline.Reporter())
    assert result == tmp_path / "episode.sdh.vtt"
    assert result.read_bytes() == repr(events).encode("utf-8")
    assert not wav.exists()


def test_sdh_missing_dependency_and_detector_exception_retain_preexisting_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import pipeline, sdh

    media, wav = _sdh_inputs(tmp_path)
    sidecar = tmp_path / "episode.sdh.vtt"
    sidecar.write_bytes(b"existing")
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(
        sdh,
        "detect_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError("panns_inference")
        ),
    )
    assert pipeline._write_sdh_sidecar(media, (), pipeline.Reporter()) is None
    assert sidecar.read_bytes() == b"existing"

    wav.write_bytes(b"wav")
    monkeypatch.setattr(
        sdh,
        "detect_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("detector failed")
        ),
    )
    with pytest.raises(RuntimeError, match="detector failed"):
        pipeline._write_sdh_sidecar(media, (), pipeline.Reporter())
    assert sidecar.read_bytes() == b"existing"


def test_sdh_atomic_replace_failure_retains_preexisting_sidecar_and_no_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import episode_transaction, pipeline, sdh

    media, wav = _sdh_inputs(tmp_path)
    sidecar = tmp_path / "episode.sdh.vtt"
    sidecar.write_bytes(b"existing")
    monkeypatch.setattr(pipeline, "decode_to_wav", lambda *_args, **_kwargs: wav)
    monkeypatch.setattr(sdh, "detect_events", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(sdh, "fit_events_to_gaps", lambda found, _cues: found)
    monkeypatch.setattr(sdh, "render_sdh_vtt", lambda _cues, _events: "candidate")
    monkeypatch.setattr(
        episode_transaction,
        "_replace_stage",
        lambda _stage: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        pipeline._write_sdh_sidecar(media, (), pipeline.Reporter())
    assert sidecar.read_bytes() == b"existing"
    assert not wav.exists()
    assert not [
        path for path in tmp_path.iterdir() if path.name.startswith(".episode.sdh.")
    ]
