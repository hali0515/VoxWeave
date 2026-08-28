#!/usr/bin/env python3
"""Isolated production-command worker for the detached P6 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("public oracle request is not an object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _write_wav(path: Path, *, sample_rate: int, sample_count: int) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        remaining = sample_count
        silence = b"\0\0" * min(sample_count, sample_rate)
        while remaining:
            count = min(remaining, sample_rate)
            stream.writeframesraw(silence[: count * 2])
            remaining -= count


def _select_family(language: str, family: str) -> None:
    if family == "legacy-v1":
        return
    if family != "boundary-v2":
        raise ValueError(f"unsupported public oracle family {family!r}")
    from voxweave import engine_registry

    mapping = dict(engine_registry.LANGUAGE_ENGINE_FAMILY)
    mapping[language] = "boundary-v2"
    engine_registry.LANGUAGE_ENGINE_FAMILY = MappingProxyType(mapping)
    encoded = (
        json.dumps(mapping, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    engine_registry.REGISTRY_CANONICAL_BYTES = encoded
    engine_registry.REGISTRY_SHA256 = hashlib.sha256(encoded).hexdigest()


def _install_align_seams(
    fixture: dict[str, Any],
    *,
    episode_root: Path,
    media_path: Path,
    route_evidence_path: Path,
) -> None:
    from voxweave import backend, config, pipeline

    route = fixture["route"]
    sample_rate = int(fixture["media"]["sample_rate"])
    sample_count = int(fixture["media"]["sample_count"])
    pipeline._prepare_16k_for_align = lambda *_args, **_kwargs: media_path
    backend.release = lambda: None
    backend.uses_mms = lambda _iso: route == "mms-full"
    config.align_model_for = lambda _iso: (
        "detached-synthetic-ctc" if route == "ctc-full" else None
    )

    if route in {"ctc-full", "mms-full"}:
        block_units = fixture["block_units"]

        def full_pass(
            _wav: Path,
            texts: list[str],
            _iso: str,
            *_args: Any,
            bounds: Any = None,
            _raw_call_observer: Any = None,
            _backend_invoker: Any = None,
            **_kwargs: Any,
        ) -> list[list[dict[str, Any]]]:
            groups = deepcopy(block_units)
            remaining = list(groups)
            ordered: list[list[dict[str, Any]]] = []
            for text in texts:
                matches = [
                    index
                    for index, group in enumerate(remaining)
                    if text
                    in {
                        "".join(str(unit["text"]) for unit in group),
                        " ".join(str(unit["text"]) for unit in group),
                    }
                ]
                if len(matches) != 1:
                    break
                ordered.append(remaining.pop(matches[0]))
            if len(ordered) == len(groups) and not remaining:
                groups = ordered
            raw = [unit for group in groups for unit in group]
            if _backend_invoker is not None:
                raw = _backend_invoker(lambda: raw)
            if _raw_call_observer is not None:
                _raw_call_observer(
                    raw,
                    list(raw) if route == "ctc-full" else None,
                    tuple(range(len(texts))),
                    0.0,
                    audio_sample_start=0,
                    audio_sample_end=sample_count,
                    sample_rate=sample_rate,
                    sample_count=sample_count,
                    nominal_end_seconds=None,
                )
            _write_json(
                route_evidence_path,
                {"bounds": bounds, "route": route, "texts": texts},
            )
            return groups

        if route == "ctc-full":
            backend.align_blocks_full_ctc = full_pass
        else:
            backend.align_blocks_full_mms = full_pass
        return

    if route != "qwen-crop":
        raise ValueError(f"unsupported public oracle route {route!r}")
    call_units = iter(deepcopy(fixture["qwen_call_units"]))
    crop_index = 0

    def slice_wav(
        _wav: Path,
        start: float,
        end: float,
        *,
        _sample_geometry_observer: Any = None,
        **_kwargs: Any,
    ) -> Path:
        nonlocal crop_index
        a = max(0, int(float(start) * sample_rate))
        b = min(sample_count, int(float(end) * sample_rate))
        if _sample_geometry_observer is not None:
            _sample_geometry_observer(a, b, sample_rate, sample_count)
        crop = episode_root / f".oracle-crop-{crop_index}.wav"
        crop_index += 1
        _write_wav(crop, sample_rate=sample_rate, sample_count=max(0, b - a))
        return crop

    def align_text(_wav: Path, _text: str, _iso: str) -> list[dict[str, Any]]:
        try:
            return next(call_units)
        except StopIteration as exc:
            raise RuntimeError("public oracle Qwen receipt was exhausted") from exc

    pipeline.slice_wav = slice_wav
    backend.align_text = align_text


def _install_correct_seams() -> None:
    from voxweave import asrfix

    os.environ["P6_ORACLE_API_KEY"] = "detached-no-network"
    asrfix.correct_cues = lambda *_args, **_kwargs: []


def _install_failure_injections(injections: list[str]) -> None:
    if not injections:
        return

    def fail_w1(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected public-oracle AO-15 failure")

    def fail_ao16(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected public-oracle AO-16 failure")

    for injection in injections:
        if injection == "ao15-w1":
            from voxweave import align_adapter

            align_adapter._w1_delivery = fail_w1
        elif injection == "ao16-core":
            from voxweave import align_orchestration

            align_orchestration.build_evidence_core = fail_ao16
        else:
            raise ValueError(f"unknown public oracle injection {injection!r}")


def _materialize(fixture: dict[str, Any], episode_root: Path) -> tuple[Path, Path]:
    target = episode_root / fixture["target"]
    json_path = episode_root / "episode.json"
    if "vtt" in fixture:
        target.write_text(fixture["vtt"], encoding="utf-8")
    if "json" in fixture:
        json_path.write_text(fixture["json"], encoding="utf-8")
    if "speaker_mapping" in fixture:
        (episode_root / "episode.speakers.json").write_text(
            fixture["speaker_mapping"],
            encoding="utf-8",
        )
    media_path = episode_root / "episode.wav"
    if "media" in fixture:
        _write_wav(
            media_path,
            sample_rate=int(fixture["media"]["sample_rate"]),
            sample_count=int(fixture["media"]["sample_count"]),
        )
    return target, media_path


def _invoke(request: dict[str, Any]) -> None:
    from voxweave import cli as cli_module

    cli_module.cli.main(
        args=list(request["arguments"]),
        prog_name="voxweave",
        standalone_mode=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--trace-out", type=Path, required=True)
    parser.add_argument("--evidence-verification-out", type=Path, required=True)
    parser.add_argument("--outcome-out", type=Path, required=True)
    parser.add_argument("--route-evidence-out", type=Path, required=True)
    arguments = parser.parse_args()
    request = _read_json(arguments.request)
    fixture = request["fixture"]
    if not isinstance(fixture, dict):
        raise TypeError("public oracle fixture is not an object")
    arguments.episode_root.mkdir(parents=True, exist_ok=False)
    target, media_path = _materialize(fixture, arguments.episode_root)
    os.chdir(arguments.episode_root)
    _select_family(fixture["language"], request["expected_family"])
    if request.get("shadow_requested") is True:
        os.environ["VOXWEAVE_SEG_V2_SHADOW"] = "1"
    else:
        os.environ.pop("VOXWEAVE_SEG_V2_SHADOW", None)
    if request["command"] == "align":
        _install_align_seams(
            fixture,
            episode_root=arguments.episode_root,
            media_path=media_path,
            route_evidence_path=arguments.route_evidence_out,
        )
    elif request["command"] == "correct":
        _install_correct_seams()
    _install_failure_injections(list(request.get("injections", ())))

    if request.get("historical") is True:
        _invoke(request)
        _write_json(arguments.trace_out, None)
        _write_json(arguments.evidence_verification_out, None)
        _write_json(
            arguments.outcome_out,
            {"exception_class": None, "success": True},
        )
        return 0

    if request["command"] == "align":
        from voxweave.align_runtime import capture_align_runtime_trace

        failure: BaseException | None = None
        with capture_align_runtime_trace() as capture:
            try:
                _invoke(request)
            except BaseException as exc:
                failure = exc
        _write_json(arguments.trace_out, capture.snapshot().as_record())
        _write_json(
            arguments.outcome_out,
            {
                "exception_class": (
                    None if failure is None else type(failure).__name__
                ),
                "success": failure is None,
            },
        )
        if failure is not None:
            if request.get("expect_failure") is True:
                _write_json(arguments.evidence_verification_out, None)
                return 0
            raise failure
        from voxweave.align_evidence import verify_align_evidence

        verification = verify_align_evidence(
            target,
            explicit_media_path=media_path,
        )
        _write_json(
            arguments.evidence_verification_out,
            {
                "detail_code": verification.detail_code,
                "integrity": verification.integrity,
                "w1_usable": verification.w1_usable,
            },
        )
    else:
        _invoke(request)
        _write_json(arguments.trace_out, None)
        _write_json(arguments.evidence_verification_out, None)
        _write_json(
            arguments.outcome_out,
            {"exception_class": None, "success": True},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
