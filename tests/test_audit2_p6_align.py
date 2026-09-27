"""Regressions for the second audit pass over the P6 align modules."""

from __future__ import annotations

import dataclasses
import decimal
import itertools
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_p6_runtime_ao import _stub_public_route, _write_public_align_episode


def test_reversed_aligner_unit_does_not_abort_align(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit the aligner returns with end < start is legacy data, not a schema breach."""
    from voxweave import backend, pipeline
    from voxweave.align_evidence import verify_align_evidence

    vtt_path, media_path = _write_public_align_episode(tmp_path, route="qwen-crop")
    _stub_public_route(monkeypatch, tmp_path, route="qwen-crop", media_path=media_path)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [{"text": text, "start": 0.8, "end": 0.2}],
    )

    assert pipeline.align(vtt_path, media_path=media_path, separate=False) == vtt_path
    assert "你" in vtt_path.read_text(encoding="utf-8")
    verified = verify_align_evidence(vtt_path, explicit_media_path=media_path)
    assert verified.integrity is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (None, 30.0),
        ("12.5", 12.5),
        ("abc", 30.0),
        ("0", 30.0),
        ("-3", 30.0),
        ("nan", 30.0),
        ("inf", 30.0),
    ),
)
def test_env_float_knobs_fall_back_instead_of_breaking(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    from voxweave.align_common import _env_float_in_range

    if raw is None:
        monkeypatch.delenv("VOXWEAVE_TEST_WINDOW_S", raising=False)
    else:
        monkeypatch.setenv("VOXWEAVE_TEST_WINDOW_S", raw)
    assert _env_float_in_range("VOXWEAVE_TEST_WINDOW_S", 30.0, low=1.0) == expected


def test_env_fraction_knob_rejects_zero_and_values_above_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave.align_common import _env_float_in_range

    for raw, expected in (("0", 0.8), ("1.5", 0.8), ("1", 1.0), ("0.5", 0.5)):
        monkeypatch.setenv("VOXWEAVE_TEST_FRAC", raw)
        assert (
            _env_float_in_range(
                "VOXWEAVE_TEST_FRAC", 0.8, low=0.0, high=1.0, low_inclusive=False
            )
            == expected
        )


def test_ctc_transcribe_path_refuses_unsafe_plan_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a preparation invoker, the CTC pass plans and checks before model work."""
    from voxweave import align_common, align_ctc, chunking
    from voxweave.align_dp_safety import DpRouteHintsInvalid

    waveform = np.zeros(40 * align_ctc.CTC_AUDIO_SR, dtype=np.float32)
    monkeypatch.setattr(align_ctc, "_load_mono", lambda *_args, **_kwargs: waveform)
    monkeypatch.setattr(align_common, "CTC_MAX_DP_FRAMES", 1250)
    monkeypatch.setattr(
        chunking,
        "plan_dp_chunks",
        lambda *_args, **_kwargs: [
            {"lo": 0, "hi": 1, "start": 0.0, "end": 18.0},
            {"lo": 1, "hi": 2, "start": 17.0, "end": 35.0},
        ],
    )
    monkeypatch.setattr(
        align_ctc,
        "_get_ctc_aligner",
        lambda *_args, **_kwargs: pytest.fail("unsafe plan reached model loading"),
    )
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with pytest.raises(DpRouteHintsInvalid) as caught:
        align_ctc.align_blocks_full_ctc(
            wav_path,
            ["A", "B"],
            "en",
            "synthetic",
            bounds=((0.0, 8.0), (22.0, 39.0)),
        )
    assert caught.value.failure.detail_code == "crop-geometry"


def test_ctc_transcribe_path_refuses_a_non_16k_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from voxweave import align_ctc

    waveform = np.zeros(align_ctc.CTC_AUDIO_SR, dtype=np.float32)
    monkeypatch.setattr(align_ctc, "_load_mono", lambda *_args, **_kwargs: waveform)
    monkeypatch.setattr(
        align_ctc, "_get_ctc_aligner", lambda *_args: SimpleNamespace(sr=8000)
    )
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with pytest.raises(RuntimeError, match="sample rate differs"):
        align_ctc.align_blocks_full_ctc(wav_path, ["A"], "en", "synthetic")


def test_vad_mask_on_mms_alignment_is_reported_as_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from voxweave import align_mms

    waveform = np.zeros(align_mms.MMS_SR, dtype=np.float32)
    monkeypatch.setattr(align_mms, "_read_wav_16k", lambda _path: waveform)
    monkeypatch.setattr(align_mms, "_mms_emit_units", lambda *_args: [])
    monkeypatch.setattr(align_mms, "_empty_cache", lambda: None)
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    wav_path = tmp_path / "episode.wav"
    wav_path.write_bytes(b"unused")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        align_mms.align_blocks_full_mms(wav_path, ["あ"], "ja")
    assert any("no effect on MMS" in record.message for record in caplog.records)

    caplog.clear()
    monkeypatch.delenv("VOXWEAVE_VAD_EMISSION_MASK")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        align_mms.align_blocks_full_mms(wav_path, ["あ"], "ja")
    assert not any("no effect on MMS" in record.message for record in caplog.records)


def test_vad_mask_on_the_qwen_align_route_is_reported_as_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from voxweave import backend, pipeline

    vtt_path, media_path = _write_public_align_episode(tmp_path, route="qwen-crop")
    _stub_public_route(monkeypatch, tmp_path, route="qwen-crop", media_path=media_path)
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _iso: [{"text": text, "start": 0.2, "end": 0.8}],
    )
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", "1")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    warnings = [r for r in caplog.records if "no effect on Qwen" in r.message]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("mask", "language", "warnings"),
    (("1", "Chinese", 1), ("1", "English", 0), ("", "Chinese", 0)),
)
def test_transcribe_reports_vad_mask_ignored_once_per_qwen_aligned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mask: str,
    language: str,
    warnings: int,
) -> None:
    from voxweave import backend

    monkeypatch.setattr(backend, "_use_mlx", lambda: False)
    monkeypatch.setattr(
        backend, "_asr_only", lambda *_args: (language, "hello there", language)
    )
    monkeypatch.setattr(
        backend,
        "align_text",
        lambda _wav, text, _lang: [{"text": text, "start": 0.0, "end": 0.5}],
    )
    monkeypatch.setattr(backend, "_release_qwen_asr", lambda: None)
    monkeypatch.setattr(backend, "_empty_cache", lambda: None)
    monkeypatch.setenv("VOXWEAVE_VAD_EMISSION_MASK", mask)
    wavs = [tmp_path / "c0.wav", tmp_path / "c1.wav"]
    for wav in wavs:
        wav.write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="voxweave"):
        backend.transcribe_chunks(wavs, None, asr_model="qwen3-asr-1.7b")
    found = [r for r in caplog.records if "no effect on Qwen" in r.message]
    assert len(found) == warnings


@pytest.mark.parametrize("iso", ("th", "lo", "my", "ja", "en"))
def test_legacy_parity_lane_slices_like_the_full_pass_distributor(iso: str) -> None:
    """The legacy lane mirrors align_common._distribute_units for every language."""
    from voxweave.align_common import _distribute_units
    from voxweave.align_distribution import legacy_distribute_before_shift
    from voxweave.align_evidence_core import _r_legacy_count

    texts = ("สวัสดี ครับ", "ไป ไหน มา")
    flat = [
        {"text": f"u{index}", "start": float(index), "end": float(index) + 0.5}
        for index in range(40)
    ]
    expected = _distribute_units(flat, list(texts), iso)
    result = legacy_distribute_before_shift(
        flat,
        texts=texts,
        iso=iso,
        origin=0.0,
        identity=True,
        raw_unit_ids=tuple(f"r{index}" for index in range(len(flat))),
    )
    assert [list(owner) for owner in result.block_units] == expected
    assert result.receipt.expected_counts == tuple(len(owner) for owner in expected)
    assert tuple(_r_legacy_count(text, iso) for text in texts) == (
        result.receipt.expected_counts
    )


def _quadratic_overlap(observed: tuple[int, ...], count: int) -> int | None:
    for duplicated in range(count):
        positions = [
            position for position, index in enumerate(observed) if index == duplicated
        ]
        if len(positions) > 1:
            return positions[1]
    return None


def test_route_overlap_detection_matches_the_quadratic_definition() -> None:
    from voxweave.align_distribution import (
        RouteClaim,
        RouteExpectation,
        project_route_mismatch,
    )
    from voxweave.align_distribution_reference import _route_mismatch

    rng = random.Random(7)
    for _ in range(400):
        count = rng.randint(1, 7)
        length = rng.randint(count, count + 4)
        observed = [rng.randint(-1, count) for _ in range(length)]
        for index in range(count):  # no gap, so the overlap rule is reached
            if index not in observed:
                observed[rng.randrange(length)] = index
        if any(index not in observed for index in range(count)):
            continue
        claims = tuple(
            RouteClaim("call", 0, index, position)
            for position, index in enumerate(observed)
        )
        position = _quadratic_overlap(tuple(observed), count)
        route = tuple(
            RouteExpectation(index, index, "call", 0) for index in range(count)
        )
        produced = project_route_mismatch(claims, route, (), ())
        replayed = _route_mismatch(claims, route, (), ())  # type: ignore[arg-type]
        if position is None:
            assert produced is None or produced.kind != "overlap"
            assert replayed is None or replayed.kind != "overlap"
        else:
            assert produced is not None and produced.kind == "overlap"
            assert produced.observation_index == position
            assert replayed == produced


def test_claim_positions_are_grouped_per_owner_in_claim_order() -> None:
    from voxweave.align_distribution import RouteClaim, _claim_positions
    from voxweave.align_distribution_reference import _owner_positions

    claims = tuple(
        RouteClaim(kind, owner, delivery, delivery)
        for delivery, (kind, owner) in enumerate(
            itertools.islice(
                itertools.cycle((("call", 0), ("skip", 0), ("call", 1))), 9
            )
        )
    )
    expected = {("call", 0): (0, 3, 6), ("skip", 0): (1, 4, 7), ("call", 1): (2, 5, 8)}
    assert _claim_positions(claims) == expected
    assert _owner_positions(claims) == expected


def test_changed_authority_limits_carry_their_registered_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_distribution as d

    monkeypatch.setattr(d, "AUTH_ALLOC_STATE_LIMIT", 999_999)
    with pytest.raises(d.AuthorityLimitProfileError) as caught:
        d.capture_authority_limit_profile()
    assert caught.value.failure.to_dict() == {
        "kind": "context-authority-invalid",
        "phase": "context",
        "detail_code": "allocator-limit-profile",
        "secondary": [],
    }


def test_unsealable_retained_unit_fails_at_the_legacy_digest(tmp_path: Path) -> None:
    from tests.test_p6_acquisition import _fresh_session
    from voxweave.align_acquisition import (
        _fresh_alignment_call_observer,
        seal_fresh_alignment,
    )

    _context, session = _fresh_session(tmp_path)
    _fresh_alignment_call_observer(session)(
        ({"text": "word", "start": decimal.Decimal("0.0"), "end": 1.0},),
        None,
        (0,),
        0.0,
    )
    with pytest.raises(TypeError) as caught:
        seal_fresh_alignment(session)
    failure = getattr(caught.value, "failure")
    assert (failure.kind, failure.phase, failure.detail_code) == (
        "legacy-time-transform-failed",
        "legacy-time-transform",
        "retained-unit-operand",
    )


def test_failed_legacy_encode_names_its_cause_on_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxweave.candidate_encoder as encoder
    from tests.test_p6_align_candidates import _evaluated

    context, result = _evaluated(tmp_path, shadow_requested=True)
    injected = UnicodeError("injected legacy renderer failure")

    def fail_legacy(*_args: Any, **_kwargs: Any) -> Any:
        raise injected

    monkeypatch.setattr(encoder, "project_align_delivery", fail_legacy)
    candidates = encoder.encode_align_candidates(context, result)
    legacy = candidates.outcome_for("legacy-v1")
    assert isinstance(legacy, encoder.CandidateFailure)
    assert legacy.cause is injected
    with pytest.raises(encoder.SelectedCandidateError) as caught:
        encoder.select_align_candidate(context, candidates)
    assert caught.value.failure.detail_code == "selected-candidate-missing"
    assert caught.value.__cause__ is injected
    message = str(caught.value)
    assert "injected legacy renderer failure" in message
    assert "preencode-failed/encoder/main-json-encode" in message
    assert message.endswith(
        "[selected-render-invalid/renderer/selected-candidate-missing]"
    )


def test_public_align_reports_a_legacy_render_failure_with_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxweave.candidate_encoder as encoder
    from voxweave import pipeline

    vtt_path, media_path = _write_public_align_episode(tmp_path, route="ctc-full")
    _stub_public_route(monkeypatch, tmp_path, route="ctc-full", media_path=media_path)
    before = vtt_path.read_bytes()

    def fail_render(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("render target vanished")

    monkeypatch.setattr(encoder, "project_align_delivery", fail_render)
    with pytest.raises(encoder.SelectedCandidateError, match="render target vanished"):
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    assert vtt_path.read_bytes() == before


def test_first_core_difference_names_the_disagreeing_field(tmp_path: Path) -> None:
    from tests.test_p6_ald6_independence import _issued_facts
    from voxweave.align_evidence_core import build_evidence_core
    from voxweave.align_orchestration import _first_core_difference

    producer_facts, _reference = _issued_facts(tmp_path)
    core = build_evidence_core(producer_facts)
    assert _first_core_difference(core, core) == "no field differs"
    changed = dataclasses.replace(core, raw_unit_count=core.raw_unit_count + 1)
    assert _first_core_difference(core, changed) == "first difference: raw_unit_count"


def _align_registries() -> tuple[Any, ...]:
    from voxweave import (
        align_acquisition,
        align_adapter,
        align_context,
        align_evidence,
        candidate_encoder,
    )

    return (
        align_context._ISSUED,
        align_acquisition._FRESH,
        align_acquisition._VERIFIED_FRESH,
        align_acquisition._FRESH_SESSIONS,
        align_adapter._ADAPTERS,
        align_adapter._EVALUATED,
        candidate_encoder._SETS,
        candidate_encoder._ENCODED,
        candidate_encoder._VERIFIED,
        align_evidence._EVIDENCE,
    )


def _capture_releases(
    monkeypatch: pytest.MonkeyPatch, registries: tuple[Any, ...]
) -> list[tuple[Any, list[int]]]:
    """Record each context align releases, with the registry sizes just before."""
    from voxweave import align_orchestration

    released: list[tuple[Any, list[int]]] = []
    release = align_orchestration.release_align_selection

    def capture_release(context: Any) -> None:
        released.append((context, [len(registry) for registry in registries]))
        release(context)

    monkeypatch.setattr(align_orchestration, "release_align_selection", capture_release)
    return released


def test_release_forgets_every_registry_entry_of_a_finished_align(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxweave import align_orchestration, pipeline
    from voxweave.align_context import ContextAuthorityError, role_vector

    registries = _align_registries()
    released = _capture_releases(monkeypatch, registries)
    # The shadow lane also exercises the verified-transfer registry.
    monkeypatch.setenv("VOXWEAVE_SEG_V2_SHADOW", "1")
    vtt_path, media_path = _write_public_align_episode(tmp_path, route="ctc-full")
    _stub_public_route(monkeypatch, tmp_path, route="ctc-full", media_path=media_path)
    baseline = [len(registry) for registry in registries]
    observed: list[list[int]] = []

    def observer(_artifact: object) -> None:
        observed.append([len(registry) for registry in registries])

    assert (
        pipeline.align(
            vtt_path, media_path=media_path, separate=False, _shadow_observer=observer
        )
        == vtt_path
    )
    # align released its context itself, after the shadow observer had run.
    ((context, grown),) = released
    assert [after > before for before, after in zip(baseline, grown)] == [True] * 10
    assert observed == [grown]
    assert [len(registry) for registry in registries] == baseline
    with pytest.raises(ContextAuthorityError) as caught:
        role_vector(context)
    assert caught.value.detail_code == "context-unissued"
    align_orchestration.release_align_selection(context)  # idempotent
    assert [len(registry) for registry in registries] == baseline


def test_failed_align_releases_its_registries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voxweave import episode_transaction, pipeline

    registries = _align_registries()
    released = _capture_releases(monkeypatch, registries)
    vtt_path, media_path = _write_public_align_episode(tmp_path, route="ctc-full")
    _stub_public_route(monkeypatch, tmp_path, route="ctc-full", media_path=media_path)
    baseline = [len(registry) for registry in registries]
    before = vtt_path.read_bytes()

    def fail_commit(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(episode_transaction, "commit_primary_outputs", fail_commit)
    with pytest.raises(OSError, match="disk full"):
        pipeline.align(vtt_path, media_path=media_path, separate=False)
    ((_context, grown),) = released
    assert grown != baseline
    assert [len(registry) for registry in registries] == baseline
    assert vtt_path.read_bytes() == before


def test_sibling_snapshot_digest_is_computed_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxweave import align_snapshot

    calls: list[object] = []
    original = align_snapshot._sibling_digest

    def counting(**kwargs: Any) -> str:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(align_snapshot, "_sibling_digest", counting)
    snapshot = align_snapshot.decode_sibling_json_snapshot(
        "episode.json", b'{"word_segments": []}'
    )
    assert calls == []
    assert snapshot.digest == snapshot.digest
    assert len(calls) == 2


def test_frozen_json_primitives_do_not_import_the_subtitle_stack() -> None:
    probe = (
        "import sys, voxweave.align_snapshot; "
        "print(sorted(m for m in ('voxweave.realign', 'voxweave.subformats',"
        " 'voxweave.speakers') if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "[]"


def test_stored_profile_key_order_is_not_significant() -> None:
    from voxweave.align_inputs import resolve_align_profile
    from voxweave.core.segdoc import THRESHOLD_KEYS

    thresholds = dict.fromkeys(THRESHOLD_KEYS, 1)
    profile = {"max_line_length": 42, "max_lines": 2, **thresholds}
    manifest = {
        "manifest_version": 1,
        "engine": "legacy-v1",
        "language": "en",
        "profile": dict(reversed(list(profile.items()))),
    }
    reordered = resolve_align_profile(manifest, effective_iso="en")
    manifest["profile"] = profile
    ordered = resolve_align_profile(manifest, effective_iso="en")
    assert reordered.status == ordered.status
    manifest["profile"] = {**profile, "extra": 1}
    extra = resolve_align_profile(manifest, effective_iso="en")
    assert (extra.status.kind, extra.status.detail_code) == ("invalid", "profile-shape")
