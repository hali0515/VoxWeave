#!/usr/bin/env python
"""Capture replayable fixtures from a real media file: song-skip scenarios and segmentation cases.

Two independent artifacts, selected by flags:

1. **Song-skip scenario** (default, unchanged): runs the GPU stages once and writes
   ``tests/scenarios/<name>.json`` with the PANNs per-window scores plus the
   separated-vocals VAD segments. ``tests/test_scenarios.py`` replays those with
   pure functions (songdet + ``pipeline.plan_song_skip``) -- zero GPU required.
   The fixture is pre-populated with the current (correct) behavior as a golden
   snapshot; fill in ``assert.speech_present_at`` manually with timestamps where
   speech must be present (regression anchors).

2. **Segmentation case** (``--with-units``): writes an additional
   ``calibration/segmentation/cases/<name>.json`` validated against
   ``calibration/schemas/segmentation-case.schema.json``. The source is the
   sibling JSON a previous voxweave run already produced -- its final
   ``word_segments`` plus whichever of ``vad_speech`` / ``shot_changes`` /
   ``sing_spans`` / ``speaker_turns`` it recorded. Detectors are never re-run:
   an input the sibling does not carry is written as an empty array and named in
   ``capture.missing_inputs``, so a replay can never silently invent it.

   ``--units-only`` is the zero-GPU path: the sibling JSON is the only input read
   and no model module is imported at all.

Usage::

    # song-skip scenario only (legacy behavior)
    python scripts/capture_scenario.py episode.mkv meido-e12 --desc "cold open"

    # segmentation case only, no GPU, 120s window
    python scripts/capture_scenario.py episode.mkv zh-03 \\
        --with-units episode.json --units-only --range 615.0:735.0 \\
        --lang zh --license-class self-recorded \\
        --case-out calibration/segmentation/cases/zh-03.json

Exit codes follow the shared calibration contract: 0 = written, 2 = invalid
input / schema / tooling. The segmentation case is built and validated before
the GPU stages run, so a bad window or a missing license declaration fails in
milliseconds instead of after a separation pass.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent


def _load_calib_common() -> Any:
    """Import ``scripts/calib_common.py`` by path -- ``scripts/`` is not an installed package.

    Loading by path (rather than mutating ``sys.path``) keeps this module
    importable from a test that loads it the same way, without the two copies
    disagreeing about which ``calib_common`` is authoritative.
    """
    cached = sys.modules.get("calib_common")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "calib_common", _SCRIPTS_DIR / "calib_common.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise RuntimeError(f"cannot load {_SCRIPTS_DIR / 'calib_common.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["calib_common"] = module
    spec.loader.exec_module(module)
    return module


cc = _load_calib_common()

#: Where a segmentation case lands unless ``--case-out`` says otherwise.
DEFAULT_CASE_DIR = REPO_ROOT / "calibration" / "segmentation" / "cases"

CASE_SCHEMA = "segmentation-case"
CASE_SCHEMA_VERSION = 1

#: Sibling-JSON keys a case replays; absent ones are recorded in ``missing_inputs``.
OPTIONAL_INPUTS = ("vad_speech", "shot_changes", "sing_spans", "speaker_turns")

#: Source classes the schema accepts, i.e. the ones that are redistributable.
REDISTRIBUTABLE_CLASSES = (
    "self-recorded",
    "cc",
    "public-domain",
    "synthetic-from-consented-speech",
)
#: Default ``--license-class``: declaring nothing must refuse, never assume rights.
UNDECLARED_LICENSE_CLASS = "undeclared"

#: Optional segmenters whose version silently changes where breaks land.
SEGMENTER_DISTRIBUTIONS = ("pysbd", "budoux", "jieba", "fugashi")

#: Times are stored at microsecond resolution -- far below any audio boundary we
#: can measure, and enough to keep a rebased float from printing 17 digits.
TIME_DECIMALS = 6

#: Netflix shot-snap zones are specified in frames at 24 fps (see core/timing.py).
SHOT_SNAP_FPS = 24.0

_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


# --------------------------------------------------------------------------- #
# Window arithmetic
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Window:
    """The capture window in source time. ``end=None`` means "to the end of the file".

    Every time written into a case is expressed relative to ``start``, so a case
    always begins at t=0 regardless of where in the episode it was cut from.
    """

    start: float = 0.0
    end: float | None = None

    @property
    def bounded(self) -> bool:
        return self.end is not None

    def rebase(self, t: float) -> float:
        """Shift a source timestamp into window time (never negative)."""
        return round(max(0.0, float(t) - self.start), TIME_DECIMALS)

    def contains(self, start: float, end: float) -> bool:
        """True when ``[start, end]`` lies wholly inside the window (words are never split)."""
        eps = 1e-9
        if start < self.start - eps:
            return False
        return self.end is None or end <= self.end + eps

    def clip(self, start: float, end: float) -> tuple[float, float] | None:
        """Intersect a span with the window and rebase it; ``None`` when nothing survives."""
        lo = max(float(start), self.start)
        hi = float(end) if self.end is None else min(float(end), self.end)
        if hi <= lo:
            return None
        return self.rebase(lo), self.rebase(hi)

    def holds_point(self, t: float) -> bool:
        eps = 1e-9
        if t < self.start - eps:
            return False
        return self.end is None or t <= self.end + eps


def parse_range(spec: str) -> Window:
    """Parse ``START:END`` seconds into a :class:`Window`."""
    parts = str(spec).strip().split(":")
    if len(parts) != 2:
        raise cc.CalibrationError(f"--range must be START:END seconds, got {spec!r}")
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError:
        raise cc.CalibrationError(
            f"--range bounds must be numbers, got {spec!r}"
        ) from None
    if not (math.isfinite(start) and math.isfinite(end)):
        raise cc.CalibrationError(f"--range bounds must be finite, got {spec!r}")
    if start < 0:
        raise cc.CalibrationError(f"--range start must be >= 0, got {start!r}")
    if end <= start:
        raise cc.CalibrationError(f"--range end must be > start, got {spec!r}")
    return Window(start, end)


# --------------------------------------------------------------------------- #
# Sibling JSON -> case inputs
# --------------------------------------------------------------------------- #


def sibling_json_for(media: Path) -> Path:
    """The sibling ``.json`` next to ``media`` (``--with-units auto``).

    Uses ``pipeline.swap_ext``: ``Path.with_suffix`` truncates at the first
    interior dot, which real release filenames are full of.
    """
    from voxweave.pipeline import swap_ext

    return swap_ext(Path(media), ".json")


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def select_units(
    word_segments: Sequence[Mapping[str, Any]], window: Window
) -> tuple[list[dict[str, Any]], int]:
    """Pick the lexical units wholly inside ``window``, rebased and given stable ids.

    A unit straddling a window edge is dropped whole -- half a word is not a
    lexical unit and would change what the segmenter reads. Returns the units
    plus the number of source entries skipped for missing text or non-finite
    spans (ghost units from a failed aligner cannot be replayed).
    """
    out: list[dict[str, Any]] = []
    skipped = 0
    for raw in word_segments:
        if not isinstance(raw, Mapping):
            skipped += 1
            continue
        word = raw.get("word")
        text = raw.get("text", word)
        start, end = raw.get("start"), raw.get("end")
        if text is None or not _finite(start) or not _finite(end):
            skipped += 1
            continue
        if not window.contains(float(start), float(end)):
            continue
        unit: dict[str, Any] = {
            "id": f"u{len(out)}",
            "text": str(text),
            "start": window.rebase(float(start)),
            "end": window.rebase(float(end)),
        }
        # The schema keeps `word` optional on purpose: only mirror it when the
        # source sibling actually carried one.
        if word is not None:
            unit["word"] = str(word)
        out.append(unit)
    return out, skipped


def clip_spans(raw: Any, window: Window) -> list[list[float]]:
    """Clip ``[[start, end], ...]`` to the window and rebase; degenerate spans drop out."""
    out: list[list[float]] = []
    for entry in raw or ():
        try:
            start, end = entry
        except (TypeError, ValueError):
            continue
        if not (_finite(start) and _finite(end)):
            continue
        clipped = window.clip(float(start), float(end))
        if clipped is not None:
            out.append([clipped[0], clipped[1]])
    return out


def clip_turns(raw: Any, window: Window) -> list[dict[str, Any]]:
    """Clip persisted ``[[start, end, label], ...]`` turns and relabel to ``S0``, ``S1``, ...

    pyannote emits ``SPEAKER_00``-style labels; the case schema pins ``^S[0-9]+$``
    so a case never leaks a detector's naming scheme. Numbering follows first
    appearance in the clipped stream, which keeps it deterministic per window.
    """
    kept: list[tuple[float, float, str]] = []
    for entry in raw or ():
        try:
            start, end, label = entry
        except (TypeError, ValueError):
            continue
        if not (_finite(start) and _finite(end)):
            continue
        clipped = window.clip(float(start), float(end))
        if clipped is not None:
            kept.append((clipped[0], clipped[1], str(label)))
    order: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for start, end, label in kept:
        speaker = order.get(label)
        if speaker is None:
            speaker = f"S{len(order)}"
            order[label] = speaker
        out.append({"start": start, "end": end, "speaker": speaker})
    return out


def filter_shots(raw: Any, window: Window) -> list[float]:
    """Keep shot-change points inside the window and rebase them."""
    out: list[float] = []
    for value in raw or ():
        if not _finite(value):
            continue
        t = float(value)
        if window.holds_point(t):
            out.append(window.rebase(t))
    return out


# --------------------------------------------------------------------------- #
# capture provenance block
# --------------------------------------------------------------------------- #


def voxweave_commit() -> str:
    """``git rev-parse HEAD`` for the repo this script lives in."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise cc.CalibrationError(
            "cannot resolve the voxweave commit (`git rev-parse HEAD` failed)",
            [str(exc)],
        ) from None
    commit = proc.stdout.strip()
    if not _COMMIT_RE.match(commit):
        raise cc.CalibrationError(
            f"`git rev-parse HEAD` returned {commit!r}, which is not a commit hash"
        )
    return commit


def dependency_versions() -> dict[str, str | None]:
    """Versions of the interpreter and every optional segmenter (``None`` when absent).

    Read from distribution metadata, never by importing the package: an absent
    fugashi silently swaps ja POS scoring for the character table, and that swap
    must be visible in the case rather than inferred from a rerun.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in SEGMENTER_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def segmentation_config(language: str) -> dict[str, Any]:
    """The effective segmentation knobs for ``language`` at capture time.

    Everything here can move a cue boundary, and every value is resolved through
    the same helpers production uses, so an env override in the capture shell is
    recorded instead of vanishing.
    """
    from voxweave import diarize
    from voxweave.config import gap_thresholds
    from voxweave.core.layout import default_max_line_length, default_max_lines

    th = gap_thresholds(language)
    shot_snap_s = float(th["shot_snap_s"])
    return {
        "language": language,
        "max_line_length": default_max_line_length(language),
        "max_lines": default_max_lines(language),
        "max_cue_s": float(th["max_cue_s"]),
        "min_cue_s": float(th["min_cue_s"]),
        "cps": float(th["cps"]),
        "lag_out_s": float(th["lag_out_s"]),
        "glue_gap_s": float(th["glue_gap_s"]),
        "gap_thresholds": {
            "clause_ms": int(th["clause_ms"]),
            "vad_skip_ms": int(th["vad_skip_ms"]),
            "offline_ms": int(th["offline_ms"]),
        },
        "gap_adaptive": os.environ.get("VOXWEAVE_GAP_ADAPTIVE", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "shot_snap_s": shot_snap_s,
        "shot_snap_frames": round(shot_snap_s * SHOT_SNAP_FPS, 3),
        "diarize_format": {
            "min_atom_overlap_s": float(diarize.MIN_ATOM_OVERLAP_S),
            "min_run_s": float(diarize.MIN_RUN_S),
            "edge_run_min_s": float(diarize.EDGE_RUN_MIN_S),
            "merge_gap_s": float(diarize.DIARIZE_MERGE_GAP_S),
            "drop_contained_s": float(diarize.DIARIZE_DROP_CONTAINED_S),
        },
    }


def license_block(
    source_class: str, attribution: str | None, spdx: str | None
) -> dict[str, Any]:
    """Build the license block, refusing anything that is not redistributable.

    A golden case is tracked in git and shipped with the repo, so "I did not say"
    has to fail closed: the default source class is undeclared and lands here.
    """
    if source_class not in REDISTRIBUTABLE_CLASSES:
        raise cc.CalibrationError(
            f"refusing to write a case with source class {source_class!r}: a tracked "
            "golden case must be redistributable",
            [f"pass --license-class from: {', '.join(REDISTRIBUTABLE_CLASSES)}"],
        )
    return {
        "redistributable": True,
        "source_class": source_class,
        "spdx": spdx,
        "attribution": attribution,
    }


# --------------------------------------------------------------------------- #
# Case assembly
# --------------------------------------------------------------------------- #


def _max_window_duration() -> float | None:
    """The longest window a case may cover, read from the schema (one source of truth)."""
    schema = cc.load_schema(CASE_SCHEMA)
    try:
        value = schema["properties"]["capture"]["properties"]["window_duration_s"][
            "maximum"
        ]
    except (KeyError, TypeError):  # pragma: no cover - schema shape changed
        return None
    return float(value)


def _max_time(
    units: Sequence[Mapping[str, Any]],
    span_groups: Iterable[Sequence[Sequence[float]]],
    turns: Sequence[Mapping[str, Any]],
    shots: Sequence[float],
) -> float:
    times: list[float] = [float(u["end"]) for u in units]
    for spans in span_groups:
        times.extend(float(span[1]) for span in spans)
    times.extend(float(t["end"]) for t in turns)
    times.extend(float(t) for t in shots)
    return max(times, default=0.0)


def build_case(
    *,
    case_id: str,
    sibling: Path,
    document: Mapping[str, Any],
    window: Window,
    language: str | None = None,
    description: str = "",
    tags: Sequence[str] = ("uncurated",),
    source_class: str = UNDECLARED_LICENSE_CLASS,
    attribution: str | None = None,
    spdx: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Assemble one segmentation case from an already-parsed sibling JSON.

    Pure apart from reading the sibling's bytes for the digest and asking git for
    HEAD; no detector is run and no model is imported. Returns the case plus the
    number of unusable source units that were skipped.
    """
    raw_units = document.get("word_segments")
    if not isinstance(raw_units, list):
        raise cc.CalibrationError(
            f"{sibling} has no `word_segments` array -- not a voxweave sibling JSON"
        )
    iso = cc.require_calibration_language(language or document.get("language"))

    units, skipped = select_units(raw_units, window)
    if not units:
        raise cc.CalibrationError(
            f"no lexical unit lies wholly inside the capture window of {sibling}",
            [f"window: {window.start}s .. {window.end if window.bounded else 'EOF'}"],
        )
    vad = clip_spans(document.get("vad_speech"), window)
    sing = clip_spans(document.get("sing_spans"), window)
    turns = clip_turns(document.get("speaker_turns"), window)
    shots = filter_shots(document.get("shot_changes"), window)
    missing = [key for key in OPTIONAL_INPUTS if key not in document]

    if window.bounded:
        assert window.end is not None
        duration = round(window.end - window.start, TIME_DECIMALS)
    else:
        duration = _max_time(units, (vad, sing), turns, shots)
    cap = _max_window_duration()
    if cap is not None and duration > cap:
        # The schema would reject this anyway; say what to do about it.
        raise cc.CalibrationError(
            f"capture window is {duration:.1f}s, over the {cap:g}s a case may cover",
            ["cut a window with --range START:END (a case is a clip, not an episode)"],
        )

    case: dict[str, Any] = {
        "schema_version": CASE_SCHEMA_VERSION,
        "id": case_id,
        "language": iso,
        **({"description": description} if description else {}),
        "tags": list(dict.fromkeys(tags)),
        "license": license_block(source_class, attribution, spdx),
        "capture": {
            "voxweave_commit": voxweave_commit(),
            "source_digest": cc.sha256_file(sibling),
            "window_duration_s": duration,
            "dependency_versions": dependency_versions(),
            "config": segmentation_config(iso),
            "missing_inputs": missing,
        },
        "word_segments": units,
        "vad_speech": vad,
        "shot_changes": shots,
        "sing_spans": sing,
        "speaker_turns": turns,
    }
    return case, skipped


def write_case(path: Path, case: Mapping[str, Any], *, force: bool = False) -> Path:
    """Validate and atomically write a case; refuse to clobber unless ``force``."""
    path = Path(path)
    if path.exists() and not force:
        raise cc.CalibrationError(
            f"{path} already exists; pass --force to overwrite it",
            ["a tracked golden case is a baseline -- silent replacement hides drift"],
        )
    cc.validate_or_exit2(case, CASE_SCHEMA, label=str(path))
    return cc.write_json(path, case)


# --------------------------------------------------------------------------- #
# The two capture flows
# --------------------------------------------------------------------------- #


def capture_units(args: argparse.Namespace) -> Path:
    """Read the sibling JSON and write the segmentation case. No models, no GPU."""
    media = Path(args.media)
    if args.with_units == "auto":
        sibling = sibling_json_for(media)
    else:
        sibling = Path(args.with_units)
    if not sibling.is_file():
        raise cc.CalibrationError(f"sibling JSON not found: {sibling}")

    window = parse_range(args.range) if args.range else Window()
    case_out = (
        Path(args.case_out) if args.case_out else DEFAULT_CASE_DIR / f"{args.name}.json"
    )
    if case_out.exists() and not args.force:
        raise cc.CalibrationError(
            f"{case_out} already exists; pass --force to overwrite it",
            ["a tracked golden case is a baseline -- silent replacement hides drift"],
        )

    tags = [t.strip() for t in str(args.tags).split(",") if t.strip()]
    if not tags:
        raise cc.CalibrationError("--tags must name at least one tag")

    document = cc.read_json(sibling)
    if not isinstance(document, dict):
        raise cc.CalibrationError(f"{sibling} is not a JSON object")

    case, skipped = build_case(
        case_id=args.name,
        sibling=sibling,
        document=document,
        window=window,
        language=args.lang or None,
        description=args.desc,
        tags=tags,
        source_class=args.license_class,
        attribution=args.attribution,
        spdx=args.spdx,
    )
    out = write_case(case_out, case, force=args.force)
    capture = case["capture"]
    print(f"[capture] wrote {out}")
    print(
        f"  units={len(case['word_segments'])} vad={len(case['vad_speech'])} "
        f"shots={len(case['shot_changes'])} sing={len(case['sing_spans'])} "
        f"speakers={len(case['speaker_turns'])} window={capture['window_duration_s']}s"
    )
    if capture["missing_inputs"]:
        print(
            f"  missing inputs (empty arrays, never re-detected): {capture['missing_inputs']}"
        )
    if skipped:
        print(f"  skipped {skipped} source unit(s) without usable text/spans")
    return out


def capture_songdet(args: argparse.Namespace) -> Path:
    """Run the GPU stages once and write the song-skip scenario fixture.

    Every model-bearing import lives in here so that ``--units-only`` can stay a
    genuinely zero-GPU path (no torch in ``sys.modules``).
    """
    import json

    import numpy as np
    import soundfile as sf

    from voxweave import backend, songdet
    from voxweave.chunking import decode_to_wav, silence_gaps, vad_speech_segments
    from voxweave.pipeline import (
        ASR_LOUDNORM,
        MAX_CHUNK_SEC,
        MIN_SONG_SKIP_SEC,
        SONG_FINE_SILENCE_MS,
        cache_vocals_path,
        plan_song_skip,
    )

    media = Path(args.media)
    af = ASR_LOUDNORM if args.normalize else None
    if args.no_separate:
        print(f"[capture] decode (no separation): {media.name}")
        voc16 = decode_to_wav(media, audio_filter=af)
        voc32 = decode_to_wav(media, sample_rate=songdet.SR)
    else:
        cache = cache_vocals_path(media)
        if cache.exists():
            print(f"[capture] reuse cached vocals: {cache}")
            voc16 = decode_to_wav(cache, audio_filter=af)
            voc32 = cache
        else:
            print(f"[capture] decode + separate: {media.name}")
            fb = decode_to_wav(media, sample_rate=44100, mono=False)
            voc = backend.separate_vocals(fb)
            voc16 = decode_to_wav(voc, audio_filter=af)
            voc32 = decode_to_wav(voc, sample_rate=songdet.SR)

    # PANNs per-window scoring (mirrors the internals of detect_song_spans)
    data, sr = sf.read(str(voc32), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    win, hop = int(songdet.WIN_SEC * sr), int(songdet.HOP_SEC * sr)
    starts_idx = list(range(0, len(data) - win + 1, hop))
    wins = np.stack([data[s : s + win] for s in starts_idx])
    model = songdet._get_model()
    probs = np.concatenate(
        [model.inference(wins[i : i + 32])[0] for i in range(0, len(wins), 32)]
    )
    speech, sing, music = songdet.reduce_scores(probs)
    t = [round(s / sr, 2) for s in starts_idx]

    segs = vad_speech_segments(voc16)
    vad_segs = [[round(s["start"], 3), round(s["end"], 3)] for s in segs]
    # Fine VAD silences: excision snaps its cut points into these (mirrors pipeline)
    fine = vad_speech_segments(voc16, min_silence_ms=SONG_FINE_SILENCE_MS)
    silences = [[round(a, 3), round(b, 3)] for a, b in silence_gaps(fine)]

    # Current (correct) behavior -> golden snapshot
    song = songdet.merge_spans(songdet.song_flags(probs), t)
    sing_starts = [tt for tt, f in zip(t, songdet.sing_flags(probs)) if f]
    sing_spans = [(a, b) for (a, b) in song if any(a <= x < b for x in sing_starts)]
    speech_spans = songdet.merge_spans(songdet.speech_flags(probs), t)
    _, final, kept, chunks = plan_song_skip(
        song,
        sing_spans,
        segs,
        speech_spans=speech_spans,
        silences=[(a, b) for a, b in silences],
        min_skip_sec=MIN_SONG_SKIP_SEC,
        max_chunk_sec=MAX_CHUNK_SEC,
    )

    fixture = {
        "name": args.name,
        "desc": args.desc,
        "lang": args.lang,
        "normalize": args.normalize,
        "win_sec": songdet.WIN_SEC,
        "hop_sec": songdet.HOP_SEC,
        "scores": {
            "t": t,
            "speech": [round(float(x), 4) for x in speech],
            "sing": [round(float(x), 4) for x in sing],
            "music": [round(float(x), 4) for x in music],
        },
        "vad_segs": vad_segs,
        "silences": silences,
        "assert": {
            "expected_song_spans": [[round(a, 1), round(b, 1)] for a, b in final],
            "speech_present_at": [],  # fill in manually: timestamps where speech should be present (regression anchors)
            "max_chunk_sec": MAX_CHUNK_SEC,
        },
    }
    out = REPO_ROOT / "tests" / "scenarios" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[capture] wrote {out}")
    print(
        f"  raw song={[(round(a, 1), round(b, 1)) for a, b in song]}  final={fixture['assert']['expected_song_spans']}"
    )
    print(
        f"  vad_segs={len(vad_segs)}  chunks={[(round(c['start'], 1), round(c['end'], 1)) for c in chunks]}"
    )
    print(
        "  -> now manually fill in assert.speech_present_at with timestamps where speech should be present"
    )
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("media")
    ap.add_argument(
        "name", help="fixture name; also the segmentation case id (e.g. zh-03)"
    )
    ap.add_argument("--desc", default="")
    ap.add_argument("--lang", default="")
    ap.add_argument(
        "--no-separate",
        action="store_true",
        help="use original audio (default: separate vocals, route ii)",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="apply the pipeline's loudnorm filter to the VAD input (mirror `normalize=true` runs;"
        " loudnorm changes VAD segment boundaries and thus voiced-block structure — a fixture"
        " meant to replay a normalized run MUST capture with this flag). PANNs scoring input"
        " stays unfiltered, exactly like the pipeline.",
    )
    seg = ap.add_argument_group("segmentation case (calibration/segmentation/cases/)")
    seg.add_argument(
        "--with-units",
        nargs="?",
        const="auto",
        default=None,
        metavar="SIBLING_JSON",
        help="also write a segmentation case from a sibling JSON; bare flag = the"
        " sibling next to MEDIA",
    )
    seg.add_argument(
        "--units-only",
        action="store_true",
        help="skip the song-skip capture entirely (requires --with-units): the"
        " sibling JSON is the only input and no model is imported",
    )
    seg.add_argument(
        "--range",
        metavar="START:END",
        default=None,
        help="seconds; keep only units wholly inside the window, clip spans, rebase"
        " every time by -START",
    )
    seg.add_argument(
        "--case-out",
        metavar="PATH",
        default=None,
        help=f"case destination (default: {DEFAULT_CASE_DIR}/<name>.json)",
    )
    seg.add_argument(
        "--force", action="store_true", help="overwrite an existing case file"
    )
    seg.add_argument(
        "--tags",
        default="uncurated",
        metavar="A,B",
        help="comma-separated corpus tags (default: uncurated)",
    )
    seg.add_argument(
        "--license-class",
        default=UNDECLARED_LICENSE_CLASS,
        choices=(UNDECLARED_LICENSE_CLASS, *REDISTRIBUTABLE_CLASSES),
        help="source class; the default is undeclared and refuses to write",
    )
    seg.add_argument("--attribution", default=None, help="credit line for the source")
    seg.add_argument(
        "--spdx", default=None, help="SPDX identifier of the source license"
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.units_only and not args.with_units:
        ap.error("--units-only requires --with-units")
    # The case is cheap and fails fast: a bad window or an undeclared license must
    # not be discovered after a separation pass has already burned GPU minutes.
    if args.with_units:
        capture_units(args)
    if not args.units_only:
        capture_songdet(args)
    return cc.EXIT_OK


if __name__ == "__main__":
    cc.run_cli(main)
