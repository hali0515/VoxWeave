#!/usr/bin/env python
"""Calibrate voiceprint matching thresholds per speaker-embedding model.

For every media file given, the diarized ``speaker_turns`` of its sibling
``<stem>.json`` are cut into centroid-v1 segments (overlap-trimmed turns of at
least ``voiceembed.MIN_TURN_SECONDS``), each model embeds them, and two score
populations are collected:

* **target**: cosine between the duration-weighted centroid of a speaker's
  even-numbered segments and that of its odd-numbered segments (same episode,
  disjoint audio, same label);
* **non-target**: cosine between the centroids of two different speaker labels
  of the same episode.

Per model it prints the counts, mean / std of both populations, the equal error
rate with its threshold, and the threshold at a 1% false-accept rate, and writes
everything (including the raw scores) to a JSON report. Use the thresholds to
replace the provisional ``suggest`` / ``margin`` defaults in
``voxweave.voiceembed``.

Caveats: labels come from diarization, so a speaker the diarizer split in two
shows up as a (wrong) non-target pair, and FAR 1% needs well over a hundred
non-target pairs to mean anything -- calibrate on several episodes.

Audio: the separated-vocals cache ``voxweave transcribe`` left next to the
media is used when present (the same signal captured voiceprints are computed
on); otherwise the original mix is decoded, with a warning.

Models (``--models``, comma separated; default
``redimnet2,anime-va,pyannote-community-1``):

* ``redimnet2``, ``anime-va``: the registered voiceembed checkpoints.
* ``pyannote-community-1`` (the default baseline): the embedding submodel of
  the default diarization pipeline, ``embedding/pytorch_model.bin`` of the
  ``pyannote/speaker-diarization-community-1`` snapshot -- what the legacy
  ``pyannote`` voiceprint lane stores with the default diarizer.
* ``pyannote-embedding``: the standalone WeSpeaker ResNet34 checkpoint
  ``pyannote/wespeaker-voxceleb-resnet34-LM`` the 3.1 pipeline embeds with. It
  is a different file (different SHA-256) from community-1's submodel; keep it
  for comparisons with voice stores built under ``--diarize-model 3.1``. (At
  community-1 revision 3533c8c and WeSpeaker revision 837717d the two files
  hold bit-identical weights and differ only in their pyannote metadata, so
  their scores match; the report still names the checkpoint that was scored.)

Both pyannote baselines load standalone the way the split feature loads them
and need the Hugging Face token used for diarization. The report records the
resolved checkpoint (revision and SHA-256) of each.

Usage::

    python scripts/calibrate_voiceprints.py ep01.mkv ep02.mkv \\
        --models redimnet2,anime-va,pyannote-community-1 --out voiceprints-calib.json

Exit codes: 0 = report written, 2 = invalid input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxweave import voiceembed  # noqa: E402

PYANNOTE_COMMUNITY_1 = "pyannote-community-1"
PYANNOTE_EMBEDDING = "pyannote-embedding"
PYANNOTE_BASELINES = (PYANNOTE_COMMUNITY_1, PYANNOTE_EMBEDDING)
DEFAULT_MODELS = ("redimnet2", "anime-va", PYANNOTE_COMMUNITY_1)
# Segments per speaker fed to the target split (longest first, then back to
# time order); more than the capture cap so odd/even halves stay meaningful.
DEFAULT_MAX_SEGMENTS = 40
TARGET_FAR = 0.01

Turn = tuple[float, float, str]
Span = tuple[float, float]


# --------------------------------------------------------------------------
# Pure scoring helpers
# --------------------------------------------------------------------------


def calibration_segments(
    turns: Sequence[Turn], label: str, *, max_segments: int
) -> list[Span]:
    """Overlap-trimmed segments of ``label`` that reach MIN_TURN_SECONDS."""
    others = [(start, end) for start, end, other in turns if other != label]
    pieces: list[Span] = []
    for start, end, owner in turns:
        if owner == label and end > start:
            pieces.extend(voiceembed._subtract(float(start), float(end), others))
    qualifying = [p for p in pieces if p[1] - p[0] >= voiceembed.MIN_TURN_SECONDS]
    ranked = sorted(qualifying, key=lambda p: (-(p[1] - p[0]), p[0], p[1]))
    return sorted(ranked[:max_segments])


def centroid(vectors: np.ndarray, spans: Sequence[Span]) -> np.ndarray:
    return voiceembed.weighted_unit_mean(vectors, [end - start for start, end in spans])


def episode_scores(
    embeddings: Mapping[str, tuple[Sequence[Span], np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Target (odd vs even halves) and non-target (label vs label) scores."""
    targets: list[dict[str, object]] = []
    full: dict[str, np.ndarray] = {}
    for label in sorted(embeddings):
        spans, vectors = embeddings[label]
        if len(spans) == 0:
            continue
        full[label] = centroid(vectors, spans)
        if len(spans) < 2:
            continue
        even = list(range(0, len(spans), 2))
        odd = list(range(1, len(spans), 2))
        score = float(
            np.dot(
                centroid(vectors[even], [spans[i] for i in even]),
                centroid(vectors[odd], [spans[i] for i in odd]),
            )
        )
        targets.append({"speaker": label, "score": score, "segments": len(spans)})
    non_targets: list[dict[str, object]] = []
    labels = sorted(full)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            non_targets.append(
                {
                    "speakers": [left, right],
                    "score": float(np.dot(full[left], full[right])),
                }
            )
    return targets, non_targets


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def error_rates(
    targets: Sequence[float], non_targets: Sequence[float]
) -> dict[str, float | None]:
    """EER with its threshold, and the threshold reaching FAR <= 1%.

    A score ``>= threshold`` is an accept. The EER is interpolated between the
    two candidate thresholds where FRR and FAR cross.
    """
    if not targets or not non_targets:
        return {
            "eer": None,
            "eer_threshold": None,
            "far1_threshold": None,
            "frr_at_far1": None,
        }
    target = np.sort(np.asarray(targets, dtype=np.float64))
    impostor = np.sort(np.asarray(non_targets, dtype=np.float64))
    candidates = np.unique(np.concatenate([target, impostor, [np.inf]]))

    def far(threshold: float) -> float:
        return float(np.mean(impostor >= threshold))

    def frr(threshold: float) -> float:
        return float(np.mean(target < threshold))

    best: tuple[float, float, float] | None = None
    previous: tuple[float, float, float] | None = None
    for threshold in candidates:
        point = (float(threshold), far(threshold), frr(threshold))
        if point[2] >= point[1]:
            if previous is None:
                best = point
            else:
                # Linear interpolation of the crossing between the two points.
                (t0, far0, frr0), (t1, far1, frr1) = previous, point
                gap0, gap1 = far0 - frr0, frr1 - far1
                weight = gap0 / (gap0 + gap1) if gap0 + gap1 > 0 else 0.0
                eer = far0 + weight * (far1 - far0)
                threshold_at = t0 if math.isinf(t1) else t0 + weight * (t1 - t0)
                best = (threshold_at, eer, eer)
            break
        previous = point
    assert best is not None  # the +inf candidate always has FRR 1 >= FAR 0
    eer_threshold, eer_far, eer_frr = best
    far1 = next(
        (float(t) for t in candidates if far(float(t)) <= TARGET_FAR),
        float("inf"),
    )
    return {
        "eer": (eer_far + eer_frr) / 2.0,
        "eer_threshold": None if math.isinf(eer_threshold) else eer_threshold,
        "far1_threshold": None if math.isinf(far1) else far1,
        "frr_at_far1": frr(far1),
    }


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------


@dataclass
class Embedder:
    """One model under calibration: embeds ``(start, end)`` spans of 16 kHz audio."""

    name: str
    embed: Callable[[np.ndarray, Sequence[Span]], np.ndarray]
    release: Callable[[], None]
    identity: dict[str, object] = field(default_factory=dict)


def _voiceembed_embedder(alias: str) -> Embedder:
    choice = voiceembed.normalize_voiceprint_choice(alias, source="--models")
    spec = voiceembed.spec_by_name(choice)
    if spec is None:
        raise ValueError(
            f"--models entry {alias!r} is not a dedicated embedder; the pyannote "
            f"baselines are {', '.join(PYANNOTE_BASELINES)}"
        )
    return Embedder(
        name=alias,
        embed=lambda waveform, spans: voiceembed.embed_segments(waveform, spans, spec),
        release=voiceembed.release,
        identity={"model": spec.name, "checkpoint_sha256": spec.sha256},
    )


def pyannote_source(name: str) -> str:
    """Embedding checkpoint of a pyannote baseline, in turnembed's source grammar."""
    from voxweave import config, turnembed

    if name == PYANNOTE_COMMUNITY_1:
        return f"{config.COMMUNITY_DIARIZE_MODEL}#subfolder=embedding"
    if name == PYANNOTE_EMBEDDING:
        return turnembed.EMBEDDING_MODEL
    raise ValueError(f"--models entry {name!r} is not a pyannote baseline")


def _pyannote_embedder(name: str) -> Embedder:
    from voxweave import turnembed

    source = pyannote_source(name)
    state: dict[str, Any] = {}
    # Filled in with the resolved revision and checkpoint once loaded.
    identity: dict[str, object] = {"source": source}

    def load() -> tuple[Any, int]:
        if "inference" not in state:
            inference, loaded = turnembed._load_inference(None, source=source)
            state["inference"] = inference
            state["minimum"] = turnembed._minimum_samples(inference)
            identity.update(
                {
                    "model": loaded.model,
                    "checkpoint_sha256": loaded.checkpoint_sha256,
                    "pyannote_version": loaded.pyannote_version,
                }
            )
        return state["inference"], state["minimum"]

    def embed(waveform: np.ndarray, spans: Sequence[Span]) -> np.ndarray:
        import torch

        inference, minimum = load()
        rows = []
        for start, end in spans:
            first = max(0, math.floor(start * voiceembed.SAMPLE_RATE))
            last = min(len(waveform), math.ceil(end * voiceembed.SAMPLE_RATE))
            windows = voiceembed.window_bounds(first, last)
            vectors = []
            for low, high in windows:
                segment = waveform[low:high]
                if len(segment) < minimum:
                    segment = np.resize(segment, minimum)
                tensor = torch.from_numpy(np.ascontiguousarray(segment)).reshape(
                    1, 1, -1
                )
                row = np.asarray(inference(tensor), dtype=np.float64).reshape(-1)
                vectors.append(voiceembed.unit_vector(row))
            rows.append(
                voiceembed.weighted_unit_mean(
                    vectors, [float(high - low) for low, high in windows]
                )
            )
        return np.stack(rows)

    def release() -> None:
        state.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass

    return Embedder(name=name, embed=embed, release=release, identity=identity)


def build_embedder(name: str) -> Embedder:
    if name in PYANNOTE_BASELINES:
        return _pyannote_embedder(name)
    return _voiceembed_embedder(name)


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------


@dataclass
class Episode:
    media: Path
    turns: list[Turn]
    audio_source: Path
    separated: bool


def load_episode(media: Path, *, use_vocals_cache: bool) -> Episode:
    from voxweave import artifacts, pipeline
    from voxweave.voicebase import strict_turn_projection

    sibling = pipeline.swap_ext(media, ".json")
    try:
        document = json.loads(sibling.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{media}: cannot read sibling {sibling}: {exc}") from exc
    turns = list(strict_turn_projection(document.get("speaker_turns") or []))
    if not turns:
        raise ValueError(f"{media}: {sibling.name} has no speaker_turns (--diarize)")
    source, separated = media, False
    if use_vocals_cache:
        legacy = media.parent / pipeline.CACHE_DIRNAME / f"{media.stem}.vocals.32k.flac"
        managed = artifacts.inspect_paths(media)
        for candidate in (legacy, managed.vocals_cache if managed else None):
            if candidate is not None and candidate.is_file():
                source, separated = candidate, True
                break
    return Episode(media=media, turns=turns, audio_source=source, separated=separated)


def decode_episode(episode: Episode, *, normalize: bool) -> np.ndarray:
    from voxweave import pipeline
    from voxweave.chunking import decode_to_wav

    wav = decode_to_wav(
        episode.audio_source,
        sample_rate=voiceembed.SAMPLE_RATE,
        mono=True,
        audio_filter=pipeline.ASR_LOUDNORM if normalize else None,
    )
    try:
        return voiceembed.read_mono_16k(wav)
    finally:
        wav.unlink(missing_ok=True)


def calibrate(
    episodes: Sequence[Episode],
    waveforms: Sequence[np.ndarray],
    embedders: Sequence[Embedder],
    *,
    max_segments: int,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for embedder in embedders:
        targets: list[float] = []
        non_targets: list[float] = []
        per_episode = []
        try:
            for episode, waveform in zip(episodes, waveforms, strict=True):
                embedded: dict[str, tuple[Sequence[Span], np.ndarray]] = {}
                for label in sorted({label for _s, _e, label in episode.turns}):
                    spans = calibration_segments(
                        episode.turns, label, max_segments=max_segments
                    )
                    if spans:
                        embedded[label] = (spans, embedder.embed(waveform, spans))
                episode_targets, episode_non_targets = episode_scores(embedded)
                targets.extend(float(row["score"]) for row in episode_targets)
                non_targets.extend(float(row["score"]) for row in episode_non_targets)
                per_episode.append(
                    {
                        "media": str(episode.media),
                        "target": episode_targets,
                        "non_target": episode_non_targets,
                    }
                )
        finally:
            embedder.release()
        models[embedder.name] = {
            "identity": embedder.identity,
            "target": _stats(targets),
            "non_target": _stats(non_targets),
            **error_rates(targets, non_targets),
            "episodes": per_episode,
        }
    return models


def _format_row(name: str, result: Mapping[str, Any]) -> str:
    def number(value: object) -> str:
        return "-" if value is None else f"{float(value):.3f}"

    target, non_target = result["target"], result["non_target"]
    return (
        f"{name:<20} target n={target['count']:<4} mean={number(target['mean'])} "
        f"std={number(target['std'])}  non-target n={non_target['count']:<4} "
        f"mean={number(non_target['mean'])} std={number(non_target['std'])}  "
        f"EER={number(result['eer'])} @ {number(result['eer_threshold'])}  "
        f"FAR1%@{number(result['far1_threshold'])} "
        f"(FRR {number(result['frr_at_far1'])})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0] if __doc__ else None
    )
    parser.add_argument("media", nargs="+", type=Path)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--out", type=Path, default=Path("voiceprints-calibration.json")
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="apply the ASR loudnorm filter (match transcribe --normalize)",
    )
    parser.add_argument(
        "--no-vocals-cache",
        action="store_true",
        help="always decode the original mix instead of the separated vocals",
    )
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SEGMENTS)
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.models.split(",") if name.strip()]
    try:
        embedders = [build_embedder(name) for name in names]
        episodes = [
            load_episode(Path(media), use_vocals_cache=not args.no_vocals_cache)
            for media in args.media
        ]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not embedders:
        print("error: --models is empty", file=sys.stderr)
        return 2
    for episode in episodes:
        if not episode.separated:
            print(
                f"warning: {episode.media.name}: no separated-vocals cache; "
                "scoring the original mix",
                file=sys.stderr,
            )
    waveforms = [
        decode_episode(episode, normalize=args.normalize) for episode in episodes
    ]
    models = calibrate(episodes, waveforms, embedders, max_segments=args.max_segments)
    report = {
        "version": 1,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recipe": voiceembed.CENTROID_RECIPE,
        "min_segment_seconds": voiceembed.MIN_TURN_SECONDS,
        "max_segments": args.max_segments,
        "normalize": args.normalize,
        "episodes": [
            {
                "media": str(episode.media),
                "audio": str(episode.audio_source),
                "separated": episode.separated,
            }
            for episode in episodes
        ],
        "models": models,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in models.items():
        print(_format_row(name, result))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
