"""Segmentation quality ruler: replay the golden corpus, measure four metrics, gate.

The unit suite answers "did behaviour change unintentionally". This answers a
different question -- "is the subtitle segmentation any good" -- on a tracked,
zero-GPU corpus of real captured unit streams.

How it works::

    calibration/segmentation/corpus.json   registry: case paths, counts, tags
      -> calibration/segmentation/cases/*.json   one captured word_segments
         stream + the production replay inputs (vad_speech, shot_changes,
         sing_spans, speaker_turns) + the config that produced it
      -> voxweave.pipeline.segment_document   the *production* entry point
      -> four metrics, micro-aggregated per language
      -> one-sided comparison against calibration/segmentation/baseline.json

Every case replays through the same function ``process`` and ``split`` call, so
this harness cannot drift away from what ships. No media, no model, no network:
a case is JSON, and a replay is arithmetic plus the segmenters.

The four gated metrics (design doc 4.5):

``len_break_mid_phrase_rate``
    Of the internal cue boundaries that acoustic silence did *not* force, how
    many land inside a lexical phrase of the *source* document.
``over_7s_rate``
    Dialogue cues longer than the configured ``max_cue_s``.
``cps_p90``
    90th percentile reading speed, pooled per language over cue samples.
``forbidden_end_rate``
    Of the boundaries that had a legal in-budget alternative, how many leave a
    forward-binding token dangling at the end of a cue.

Three deliberate properties, each a fix for a flaw the 2026-08-25 audit found in
the previous version of this script:

* **Micro aggregation.** Numerators and denominators are summed across cases and
  kept in the report; per-case percentages are never averaged, so a 60 s clip
  cannot outweigh a 150 s one.
* **Stable exit codes.** ``0`` valid and gates passed, ``1`` valid but a gate
  failed, ``2`` corpus/schema/baseline invalid -- a broken corpus must never be
  reported as a quality regression.
* **Non-circular phrase truth.** The phrase segmentation used to judge a break is
  computed over the *source unit stream*, not over the cue text the splitter just
  produced, and the denominator excludes boundaries no alternative could improve.

Subcommands::

    validate-corpus   schema + coverage + size + digest, no replay
    evaluate          replay, measure, compare to baseline, write the report
    record-baseline   promote a report to the tracked baseline (never run by CI)
    compare-video-dir legacy: run the same metrics over private media siblings

The last stdout line is always a machine summary::

    QUALITY segmentation status=pass cases=20 failures=0 warnings=0 report=...
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent


def _load_calib_common() -> Any:
    """Import ``scripts/calib_common.py`` by path -- ``scripts/`` is not a package.

    Loading by path (rather than mutating ``sys.path``) keeps this module
    importable from a test that loads it the same way, without two copies
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

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

CORPUS_SCHEMA = "segmentation-corpus"
CASE_SCHEMA = "segmentation-case"
BASELINE_SCHEMA = "segmentation-baseline"

SCHEMA_VERSION = 1
#: Bumping this invalidates every recorded baseline on purpose: a metric whose
#: definition moved is not comparable to a number recorded under the old one.
METRIC_DEFINITION_VERSION = 2

DEFAULT_CORPUS = REPO_ROOT / "calibration" / "segmentation" / "corpus.json"
DEFAULT_BASELINE = REPO_ROOT / "calibration" / "segmentation" / "baseline.json"

#: Tracked-size ceilings from design 4.4; exceeding either is exit 2, not a gate.
MAX_CASE_BYTES = 256 * 1024
MAX_CORPUS_BYTES = 3 * 1024 * 1024

#: Optional segmenters whose version silently moves where breaks land.
SEGMENTER_DISTRIBUTIONS = ("pysbd", "budoux", "jieba", "fugashi")

#: Absolute cue-duration comparison slack -- 1 microsecond, far below any
#: boundary we can measure, present only to absorb float subtraction noise.
DURATION_EPS = 1e-6

#: Absolute cps_p90 ceiling as a multiple of the language's configured target.
CPS_ABSOLUTE_FACTOR = 1.25

#: How many worst samples per metric the report keeps for human triage.
OFFENDER_LIMIT = 20

#: Per-case replay budget (design 7, Phase A acceptance). Reported, not gated:
#: wall clock on a shared CI runner is not a quality signal.
CASE_WALL_TARGET_S = 1.0

GROUP_ALL = "all"
METRICS = (
    "len_break_mid_phrase_rate",
    "over_7s_rate",
    "cps_p90",
    "forbidden_end_rate",
)
#: ``over_7s_rate`` is gated on the raw bad *count*: one 12 s cue is a defect
#: whether the corpus holds 200 cues or 20 000, and a rate would dilute it away.
COUNT_METRICS = frozenset({"over_7s_rate"})

#: Initial gate table (design 4.6). ``warning`` is the landing mode: the soak
#: phase fixes metric/corpus problems without blocking PRs; flipping a gate to
#: ``blocking`` is a reviewed edit of the tracked baseline, not a code change.
#: ``cps_p90`` has no single ``absolute_max`` because the ceiling is derived per
#: language from that language's configured target CPS (see CPS_ABSOLUTE_FACTOR).
DEFAULT_GATES: dict[str, dict[str, Any]] = {
    "len_break_mid_phrase_rate": {
        "direction": "lower_is_better",
        "mode": "warning",
        "absolute_max": 0.10,
        "absolute_tolerance": 0.01,
        "relative_tolerance": 0.10,
        "min_samples": 100,
    },
    "over_7s_rate": {
        "direction": "lower_is_better",
        "mode": "warning",
        "absolute_max": 0.0,
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "min_samples": 1,
    },
    "cps_p90": {
        "direction": "lower_is_better",
        "mode": "warning",
        "absolute_max": None,
        "absolute_tolerance": 0.5,
        "relative_tolerance": 0.05,
        "min_samples": 100,
    },
    "forbidden_end_rate": {
        "direction": "lower_is_better",
        "mode": "warning",
        "absolute_max": 0.02,
        "absolute_tolerance": 0.005,
        "relative_tolerance": 0.10,
        "min_samples": 100,
    },
}

#: Punctuation that makes a break explicitly intended by the source text.
#: Sentence terminals, clause commas and the closers that trail them: a cue that
#: ends here was not chosen by the layout, it was chosen by the speaker.
_TERMINAL_PUNCT = "。．.!！?？…‥⋯"
_CLAUSE_PUNCT = "、，,;；:：·・"
_CLOSERS = "」』）)》〉】\"'”’"
_BREAK_PUNCT = frozenset(_TERMINAL_PUNCT + _CLAUSE_PUNCT + _CLOSERS)

#: Time key rounding when matching a cue's word_data back to a source unit.
#: Timing passes concatenate word_data lists but never rewrite their spans, so
#: an exact (start, end) key is a reliable identity -- microsecond rounding only
#: absorbs JSON float round-tripping.
_TIME_DECIMALS = 6

_EXCEPTION_METRICS = {
    "held_speech_over_7s": ("over_7s_rate",),
    "unavoidable_forbidden_end": ("forbidden_end_rate",),
    "known_bad_source_unit": METRICS,
}


# --------------------------------------------------------------------------- #
# Environment and provenance
# --------------------------------------------------------------------------- #


def dependency_versions() -> dict[str, str | None]:
    """Installed versions of every optional segmenter (``None`` when absent).

    Read from distribution metadata, never by importing: an absent fugashi
    silently swaps ja POS scoring for the character table, and that swap has to
    be visible in the report rather than inferred from a rerun.
    """
    versions: dict[str, str | None] = {}
    for name in SEGMENTER_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_block() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
    }


def repo_commit() -> str | None:
    """``git rev-parse HEAD`` for this repo, or ``None`` outside a checkout."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    return commit if commit and len(commit) >= 7 else None


def _python_minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


# --------------------------------------------------------------------------- #
# Corpus loading and validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    """One validated golden case: the document plus where it came from."""

    path: Path
    relpath: str
    doc: dict[str, Any]
    size_bytes: int

    @property
    def id(self) -> str:
        return str(self.doc["id"])

    @property
    def language(self) -> str:
        return str(self.doc["language"])

    @property
    def units(self) -> list[dict[str, Any]]:
        return list(self.doc["word_segments"])

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.doc["capture"]["config"])

    @property
    def tags(self) -> list[str]:
        return list(self.doc["tags"])

    @property
    def exceptions(self) -> list[dict[str, Any]]:
        return list(self.doc.get("exceptions") or [])


@dataclass(frozen=True)
class Corpus:
    """A validated registry plus its cases, and the digest that identifies both."""

    path: Path
    registry: dict[str, Any]
    cases: list[Case]
    digest: str
    total_bytes: int

    @property
    def required_counts(self) -> dict[str, int]:
        return dict(self.registry["required_counts"])


def corpus_digest(registry: Mapping[str, Any], cases: Sequence[Case]) -> str:
    """Digest of the registry and every case body.

    Keyed by the registry-relative path so reordering ``cases`` in the registry
    changes the digest (it changes evaluation order and therefore nothing about
    the metrics -- but it *is* a corpus edit, and a baseline should be reviewed
    against the corpus it was recorded on).
    """
    payload = {
        "registry": dict(registry),
        "cases": {case.relpath: case.doc for case in cases},
    }
    return cc.canonical_digest(payload)


def _validate_case_semantics(case: Case) -> None:
    """Check what JSON Schema cannot: ordering, uniqueness and window bounds.

    Anything caught here is a corpus defect (exit 2). A case that lies about its
    own window or repeats a unit id cannot support a metric, so it must not be
    allowed to quietly shift a denominator.
    """
    doc = case.doc
    problems: list[str] = []
    where = f"{case.relpath} ({case.id})"

    if case.id.split("-", 1)[0] != case.language:
        problems.append(f"id {case.id!r} does not match language {case.language!r}")

    units = doc["word_segments"]
    seen: set[str] = set()
    previous_start: float | None = None
    for i, unit in enumerate(units):
        uid = str(unit["id"])
        if uid in seen:
            problems.append(f"word_segments[{i}]: duplicate unit id {uid!r}")
        seen.add(uid)
        start, end = float(unit["start"]), float(unit["end"])
        if not (math.isfinite(start) and math.isfinite(end)):
            problems.append(f"word_segments[{i}]: non-finite span")
            continue
        if end < start:
            problems.append(f"word_segments[{i}]: end {end} < start {start}")
        if previous_start is not None and start < previous_start - DURATION_EPS:
            problems.append(
                f"word_segments[{i}]: start {start} goes back before {previous_start}"
            )
        previous_start = start

    for key in ("vad_speech", "sing_spans"):
        for i, span in enumerate(doc.get(key) or ()):
            if float(span[1]) <= float(span[0]):
                problems.append(f"{key}[{i}]: end must be greater than start")
    for i, turn in enumerate(doc.get("speaker_turns") or ()):
        if float(turn["end"]) <= float(turn["start"]):
            problems.append(f"speaker_turns[{i}]: end must be greater than start")
    for i, exc in enumerate(doc.get("exceptions") or ()):
        rng = exc["range"]
        if float(rng[1]) <= float(rng[0]):
            problems.append(f"exceptions[{i}]: range end must be greater than start")

    window = float(doc["capture"]["window_duration_s"])
    limit = window + 0.5
    latest = max(
        [float(u["end"]) for u in units]
        + [float(s[1]) for s in doc.get("vad_speech") or ()]
        + [float(s[1]) for s in doc.get("sing_spans") or ()]
        + [float(t["end"]) for t in doc.get("speaker_turns") or ()]
        + [float(t) for t in doc.get("shot_changes") or ()],
        default=0.0,
    )
    if latest > limit:
        problems.append(
            f"latest time {latest:.3f}s exceeds window_duration_s + 0.5 ({limit:.3f}s)"
        )

    if problems:
        raise cc.CalibrationError(f"{where} is not a usable case", problems)


def load_corpus(path: str | Path, *, strict_size: bool = True) -> Corpus:
    """Load and fully validate a corpus registry and every case it names.

    ``strict_size`` exists only so a private extension corpus can be larger than
    the tracked-in-git ceiling; the public corpus is always checked.
    """
    registry_path = Path(path)
    if not registry_path.is_file():
        raise cc.CalibrationError(
            f"corpus registry not found: {registry_path}",
            [
                "the tracked corpus lives at calibration/segmentation/corpus.json",
                "capture cases with: scripts/capture_scenario.py --with-units --units-only",
            ],
        )
    registry = cc.load_validated_json(
        registry_path, CORPUS_SCHEMA, label=str(registry_path)
    )
    if registry["metric_definition_version"] != METRIC_DEFINITION_VERSION:
        raise cc.CalibrationError(
            f"{registry_path} declares metric_definition_version "
            f"{registry['metric_definition_version']}, this harness implements "
            f"{METRIC_DEFINITION_VERSION}"
        )

    base = registry_path.parent
    cases: list[Case] = []
    total = 0
    problems: list[str] = []
    for relpath in registry["cases"]:
        case_path = base / relpath
        if not case_path.is_file():
            problems.append(f"{relpath}: listed in the registry but missing on disk")
            continue
        size = case_path.stat().st_size
        total += size
        if strict_size and size > MAX_CASE_BYTES:
            problems.append(
                f"{relpath}: {size} bytes exceeds the {MAX_CASE_BYTES}-byte case ceiling"
            )
        doc = cc.load_validated_json(case_path, CASE_SCHEMA, label=relpath)
        cases.append(Case(case_path, relpath, doc, size))
    if problems:
        raise cc.CalibrationError(f"{registry_path} is not a usable corpus", problems)
    if strict_size and total > MAX_CORPUS_BYTES:
        raise cc.CalibrationError(
            f"tracked corpus is {total} bytes, over the {MAX_CORPUS_BYTES}-byte ceiling",
            ["shorten a window with --range, or move the case to a private corpus"],
        )

    for case in cases:
        _validate_case_semantics(case)

    _check_coverage(registry_path, registry, cases)
    digest = corpus_digest(registry, cases)
    return Corpus(registry_path, registry, cases, digest, total)


def _check_coverage(
    registry_path: Path, registry: Mapping[str, Any], cases: Sequence[Case]
) -> None:
    """Per-language counts and tag coverage must match the registry exactly.

    A corpus that quietly lost its ``sparse-tail`` case still produces numbers;
    those numbers just stop answering the question the corpus was built to
    answer. That is an invalid run, not a passing one.
    """
    problems: list[str] = []
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate case ids: {', '.join(duplicates)}")

    counts: dict[str, int] = {}
    for case in cases:
        counts[case.language] = counts.get(case.language, 0) + 1
    for language, expected in registry["required_counts"].items():
        actual = counts.get(language, 0)
        if actual != expected:
            problems.append(
                f"required_counts[{language}] is {expected} but the corpus has {actual}"
            )
    extra = sorted(set(counts) - set(registry["required_counts"]))
    if extra:
        problems.append(f"cases in unrequired languages: {', '.join(extra)}")

    covered = {tag for case in cases for tag in case.tags}
    missing = [tag for tag in registry["required_tags"] if tag not in covered]
    if missing:
        problems.append(f"required_tags not covered by any case: {', '.join(missing)}")

    if problems:
        raise cc.CalibrationError(f"{registry_path} coverage is incomplete", problems)


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


def thresholds_from_case(case: Case) -> dict[str, Any]:
    """Rebuild the ``gap_thresholds`` mapping the case was captured with.

    Injected explicitly rather than re-read from the environment: an env
    override in the operator's shell must not silently redefine what the corpus
    measures.
    """
    config = case.config
    gaps = config.get("gap_thresholds") or {}
    missing = [
        key for key in ("clause_ms", "vad_skip_ms", "offline_ms") if key not in gaps
    ] + [
        key
        for key in (
            "min_cue_s",
            "max_cue_s",
            "glue_gap_s",
            "cps",
            "lag_out_s",
            "shot_snap_s",
        )
        if key not in config
    ]
    if missing:
        raise cc.CalibrationError(
            f"{case.relpath}: capture.config is missing segmentation knobs",
            [f"absent: {', '.join(missing)}"],
        )
    return {
        "clause_ms": int(gaps["clause_ms"]),
        "vad_skip_ms": int(gaps["vad_skip_ms"]),
        "offline_ms": int(gaps["offline_ms"]),
        "min_cue_s": float(config["min_cue_s"]),
        "max_cue_s": float(config["max_cue_s"]),
        "glue_gap_s": float(config["glue_gap_s"]),
        "cps": float(config["cps"]),
        "lag_out_s": float(config["lag_out_s"]),
        "shot_snap_s": float(config["shot_snap_s"]),
    }


def layout_kwargs_from_case(case: Case) -> dict[str, Any]:
    """Line budget overrides for ``segment_document`` (and the speaker formatter)."""
    config = case.config
    out: dict[str, Any] = {}
    if "max_line_length" in config:
        out["max_line_length"] = int(config["max_line_length"])
    if "max_lines" in config:
        out["max_lines"] = int(config["max_lines"])
    return out


@contextlib.contextmanager
def _forced_gap_adaptive(enabled: bool) -> Iterator[None]:
    """Pin ``VOXWEAVE_GAP_ADAPTIVE`` to the value the case was captured under.

    The adaptive-threshold pass reads the environment inside
    ``segment_document``, so it is the one knob a case cannot inject. Forcing it
    here makes the replay hermetic: the same corpus produces the same numbers in
    any shell.
    """
    key = "VOXWEAVE_GAP_ADAPTIVE"
    previous = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _diarize_config_drift(case: Case) -> list[str]:
    """Speaker-format constants that differ from the ones recorded in the case.

    These live as module constants, so unlike the thresholds they cannot be
    injected. Reported for every case; fatal only when the case actually carries
    speaker turns, because only then does the drift move a boundary.
    """
    recorded = case.config.get("diarize_format")
    if not isinstance(recorded, Mapping):
        return []
    from voxweave import diarize

    live = {
        "min_atom_overlap_s": float(diarize.MIN_ATOM_OVERLAP_S),
        "min_run_s": float(diarize.MIN_RUN_S),
        "edge_run_min_s": float(diarize.EDGE_RUN_MIN_S),
        "merge_gap_s": float(diarize.DIARIZE_MERGE_GAP_S),
        "drop_contained_s": float(diarize.DIARIZE_DROP_CONTAINED_S),
    }
    return [
        f"diarize_format.{key}: case {recorded[key]!r} != installed {live[key]!r}"
        for key in sorted(live)
        if key in recorded and float(recorded[key]) != live[key]
    ]


def replay(case: Case) -> Any:
    """Run one case through the production segmentation entry point.

    Pure and deterministic: no ASR, no GPU, no filesystem writes, no detector is
    re-run. Whatever the case does not carry stays empty -- a replay may never
    invent an input the capture did not record.
    """
    from voxweave.pipeline import segment_document

    doc = case.doc
    turns = [
        (float(t["start"]), float(t["end"]), str(t["speaker"]))
        for t in doc.get("speaker_turns") or ()
    ]
    drift = _diarize_config_drift(case)
    if drift and turns:
        raise cc.CalibrationError(
            f"{case.relpath}: speaker-format constants moved since capture",
            [
                *drift,
                "re-capture the case, or pin the installed voxweave to the commit",
            ],
        )
    with _forced_gap_adaptive(bool(case.config.get("gap_adaptive"))):
        return segment_document(
            language=case.language,
            word_segments=case.units,
            vad_speech=[(float(a), float(b)) for a, b in doc.get("vad_speech") or ()],
            shot_changes=[float(t) for t in doc.get("shot_changes") or ()],
            sing_spans=[(float(a), float(b)) for a, b in doc.get("sing_spans") or ()],
            speaker_turns=turns,
            thresholds=thresholds_from_case(case),
            smart_split_kwargs=layout_kwargs_from_case(case),
        )


def validate_result_contract(case: Case, cues: Sequence[Mapping[str, Any]]) -> int:
    """Terminal invariants of the cue stream; returns the overlap count.

    A non-positive duration is a contract failure rather than an infinity fed
    into a percentile (design 4.5), so it fails the run as invalid instead of
    poisoning ``cps_p90`` with a number nobody can interpret.
    """
    if not cues:
        raise cc.CalibrationError(f"{case.relpath}: replay produced no cues")
    problems: list[str] = []
    overlaps = 0
    previous_end: float | None = None
    previous_start: float | None = None
    for i, cue in enumerate(cues):
        start, end = cue.get("start"), cue.get("end")
        if start is None or end is None:
            problems.append(f"cue[{i}] has no span")
            continue
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end)):
            problems.append(f"cue[{i}] span is not finite")
            continue
        if end - start <= 0.0:
            problems.append(f"cue[{i}] duration is {end - start:.6f}s (must be > 0)")
        if previous_start is not None and start < previous_start - DURATION_EPS:
            problems.append(f"cue[{i}] starts before cue[{i - 1}]")
        if previous_end is not None and start < previous_end - DURATION_EPS:
            overlaps += 1
        previous_start, previous_end = start, end
    if problems:
        raise cc.CalibrationError(
            f"{case.relpath}: cue stream violates a terminal invariant", problems
        )
    return overlaps


# --------------------------------------------------------------------------- #
# Metric primitives (all sourced from the production modules)
# --------------------------------------------------------------------------- #


def _flat(text: str) -> str:
    return str(text).replace("\n", " ")


def _tail_token(text: str, iso: str) -> str:
    """The trailing word of a cue, as ``line_end_penalty`` expects it.

    Spaced langs: the last whitespace token. No-space langs: the last phrase atom
    of the last whitespace run, because the zh/ja penalty tables are keyed on
    whole words, not on the trailing character of a whole cue.
    """
    from voxweave.core.breakpoints import phrase_atoms

    parts = _flat(text).split()
    if not parts:
        return ""
    tail = parts[-1]
    if _no_space(iso):
        atoms = phrase_atoms(tail, iso)
        if atoms:
            tail = atoms[-1]
    return tail


def _no_space(iso: str) -> bool:
    from voxweave.core.langsets import LANGUAGES_WITHOUT_SPACES

    return iso in LANGUAGES_WITHOUT_SPACES


def _has_multichar_phrases(iso: str) -> bool:
    """True when the language's phrase segmenter can span more than one source unit.

    For spaced languages ``phrase_atoms`` is ``str.split``: an atom is exactly one
    word, and a source unit is exactly one word, so "a break inside a phrase" is
    not expressible -- ``len_break_mid_phrase_rate`` is structurally 0 there. The
    report says so per group (``phrase_granularity``) instead of letting a
    guaranteed zero read as an achievement.
    """
    return _no_space(iso)


def _reading_chars(text: str) -> int:
    from voxweave.core.layout import _reading_chars as production_reading_chars

    return production_reading_chars(text)


def _fits_budget(text: str, max_line_length: int, max_lines: int, iso: str) -> bool:
    from voxweave.core.layout import _fits_budget as production_fits_budget

    return production_fits_budget(text, max_line_length, max_lines, iso)


def _line_end_penalty(text: str, iso: str) -> int:
    from voxweave.core.kinsoku import line_end_penalty

    return line_end_penalty(text, iso)


def _phrase_atoms(text: str, iso: str) -> list[str]:
    from voxweave.core.breakpoints import phrase_atoms

    return phrase_atoms(text, iso)


def _ends_with_break_punct(text: str) -> bool:
    """True when the unit's surface form closes a sentence or clause.

    Punctuation rides on the preceding unit (``reinject_punct``), so this is the
    source stream's own statement that a boundary belongs here. Two characters
    are inspected so a closer after a terminal (``。」``) still counts.
    """
    stripped = str(text).rstrip()
    return any(ch in _BREAK_PUNCT for ch in stripped[-2:])


def _starts_with_break_punct(text: str) -> bool:
    stripped = str(text).lstrip()
    return bool(stripped) and stripped[0] in _BREAK_PUNCT


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


# --------------------------------------------------------------------------- #
# Mapping cue boundaries back onto the source unit stream
# --------------------------------------------------------------------------- #


class UnitLocator:
    """Resolve a cue's ``word_data`` entry back to its index in the source stream.

    Timing passes concatenate and slice ``word_data`` but never rewrite a span,
    so ``(start, end)`` is a stable identity. Lookup prefers the first candidate
    at or after a monotonic cursor, which disambiguates the repeated keys that
    zero-duration units produce. An entry the splitter fabricated (the logged
    proportional-timing desync path) simply does not resolve, and its boundary is
    excluded and counted -- never guessed at.
    """

    def __init__(self, units: Sequence[Mapping[str, Any]]) -> None:
        self._index: dict[tuple[float, float], list[int]] = {}
        for i, unit in enumerate(units):
            self._index.setdefault(self._key(unit), []).append(i)

    @staticmethod
    def _key(unit: Mapping[str, Any]) -> tuple[float, float]:
        start, end = unit.get("start"), unit.get("end")
        return (
            round(float(start), _TIME_DECIMALS) if start is not None else math.nan,
            round(float(end), _TIME_DECIMALS) if end is not None else math.nan,
        )

    def locate(self, entry: Mapping[str, Any], cursor: int) -> int | None:
        candidates = self._index.get(self._key(entry))
        if not candidates:
            return None
        for i in candidates:
            if i >= cursor:
                return i
        return candidates[-1]


@dataclass(frozen=True)
class Boundary:
    """One internal cue boundary, resolved onto the source units."""

    cue_index: int
    left_unit: int
    right_unit: int
    gap: float


def map_boundaries(
    units: Sequence[Mapping[str, Any]], cues: Sequence[Mapping[str, Any]]
) -> tuple[list[Boundary | None], int]:
    """Map every internal cue boundary onto ``(left_unit, right_unit)`` indices.

    Returns one entry per internal boundary (``None`` when unresolvable) plus the
    count of unresolved ones, so a report can show how much of the cue stream the
    metrics could actually see.
    """
    locator = UnitLocator(units)
    out: list[Boundary | None] = []
    unmapped = 0
    cursor = 0
    firsts: list[int | None] = []
    lasts: list[int | None] = []
    for cue in cues:
        word_data = list(cue.get("word_data") or ())
        if not word_data:
            firsts.append(None)
            lasts.append(None)
            continue
        first = locator.locate(word_data[0], cursor)
        last = locator.locate(word_data[-1], first if first is not None else cursor)
        firsts.append(first)
        lasts.append(last)
        if last is not None:
            cursor = last + 1
    for i in range(len(cues) - 1):
        left, right = lasts[i], firsts[i + 1]
        if left is None or right is None or right <= left:
            out.append(None)
            unmapped += 1
            continue
        gap = float(units[right]["start"]) - float(units[left]["end"])
        out.append(Boundary(i, left, right, gap))
    return out, unmapped


def phrase_start_offsets(
    units: Sequence[Mapping[str, Any]], iso: str
) -> tuple[set[int], list[int]]:
    """Phrase starts of the *source* document, plus each unit's character offset.

    This is the de-circularisation the audit asked for. The previous version
    segmented the concatenated *cue* text -- the splitter's own output -- and
    then asked whether the splitter had split it where the segmenter would. Here
    the truth is the input document: the units as captured, punctuation intact,
    segmented once with the whole sentence for context. The splitter never sees
    that segmentation (it works cue-locally, on punctuation-stripped text), so
    agreement is evidence rather than tautology.

    Offsets are counted in non-whitespace characters, the only index space in
    which ``phrase_atoms`` and the unit stream agree (atoms drop whitespace).
    """
    sep = "" if _no_space(iso) else " "
    text = sep.join(str(u.get("text") or "") for u in units)
    unit_offsets: list[int] = []
    offset = 0
    for unit in units:
        unit_offsets.append(offset)
        offset += _reading_chars(str(unit.get("text") or ""))
    starts = {0}
    offset = 0
    for atom in _phrase_atoms(text, iso):
        offset += _reading_chars(atom)
        starts.add(offset)
    return starts, unit_offsets


def has_legal_alternative(
    left_text: str,
    right_text: str,
    iso: str,
    max_line_length: int,
    max_lines: int,
) -> bool:
    """True when this boundary could have moved and still fit the layout budget.

    The denominator of ``forbidden_end_rate`` is the boundaries where a better
    choice existed. Repacking the two cues' atoms, an alternative counts only if
    both halves are non-empty, both fit the configured line budget, and the new
    left half does *not* end on a forward-binding token. Without this, a cue
    whose only in-budget break is a bad one would be scored as an algorithm
    defect, and a rate that punishes unsolvable boundaries can never reach 0.
    """
    left_atoms = _phrase_atoms(_flat(left_text), iso)
    right_atoms = _phrase_atoms(_flat(right_text), iso)
    atoms = left_atoms + right_atoms
    if len(atoms) < 2:
        return False
    join = "" if _no_space(iso) else " "
    actual = len(left_atoms)
    for k in range(1, len(atoms)):
        if k == actual:
            continue
        if _line_end_penalty(atoms[k - 1], iso) >= 2:
            continue
        lhs = join.join(atoms[:k])
        rhs = join.join(atoms[k:])
        if not lhs.strip() or not rhs.strip():
            continue
        if not _fits_budget(lhs, max_line_length, max_lines, iso):
            continue
        if not _fits_budget(rhs, max_line_length, max_lines, iso):
            continue
        return True
    return False


# --------------------------------------------------------------------------- #
# Per-case measurement
# --------------------------------------------------------------------------- #


@dataclass
class CaseMeasurement:
    """Everything one case contributes: ratios, samples, offenders, diagnostics."""

    case_id: str
    language: str
    cue_count: int
    ratios: dict[str, cc.Ratio] = field(default_factory=dict)
    cps_samples: list[float] = field(default_factory=list)
    offenders: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0


def _exception_ranges(case: Case) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {metric: [] for metric in METRICS}
    for exc in case.exceptions:
        span = (float(exc["range"][0]), float(exc["range"][1]))
        for metric in _EXCEPTION_METRICS.get(str(exc["kind"]), ()):
            out[metric].append(span)
    return out


def measure_case(case: Case, result: Any) -> CaseMeasurement:
    """Reduce one replayed case to the four metrics plus diagnostics.

    All four are counted here with their numerators and denominators intact; the
    aggregation step only sums them. Nothing is turned into a percentage before
    it reaches the report.
    """
    iso = case.language
    cues: list[dict[str, Any]] = list(result.cues)
    units = case.units
    config = case.config
    thresholds = thresholds_from_case(case)
    offline_s = float(thresholds["offline_ms"]) / 1000.0
    vad_skip_s = float(thresholds["vad_skip_ms"]) / 1000.0
    max_cue_s = float(thresholds["max_cue_s"])
    max_line_length = int(config.get("max_line_length") or _default_line_length(iso))
    max_lines = int(config.get("max_lines") or _default_lines(iso))
    exceptions = _exception_ranges(case)

    overlaps = validate_result_contract(case, cues)
    boundaries, unmapped = map_boundaries(units, cues)
    starts, unit_offsets = phrase_start_offsets(units, iso)
    multichar = _has_multichar_phrases(iso)

    measurement = CaseMeasurement(case_id=case.id, language=iso, cue_count=len(cues))
    spans = [(float(c["start"]), float(c["end"])) for c in cues]
    lyric = [bool(c.get("lyric")) for c in cues]

    def exempt(metric: str, index: int) -> bool:
        return any(_overlaps(spans[index], rng) for rng in exceptions[metric])

    # --- metric 2: cue duration ------------------------------------------- #
    over_bad = over_eligible = over_exempt = 0
    duration_offenders: list[dict[str, Any]] = []
    for i, cue in enumerate(cues):
        if lyric[i] or exempt("over_7s_rate", i):
            over_exempt += 1
            continue
        over_eligible += 1
        duration = spans[i][1] - spans[i][0]
        if duration > max_cue_s + DURATION_EPS:
            over_bad += 1
            duration_offenders.append(
                _offender(case, i, cue, duration=duration, value=duration)
            )

    # --- metric 3: reading speed ------------------------------------------ #
    cps_samples: list[float] = []
    cps_offenders: list[dict[str, Any]] = []
    for i, cue in enumerate(cues):
        if lyric[i] or exempt("cps_p90", i):
            continue
        duration = spans[i][1] - spans[i][0]
        if duration <= 0.0:  # pragma: no cover - validate_result_contract rejects it
            raise cc.CalibrationError(
                f"{case.relpath}: cue[{i}] has non-positive duration {duration!r}"
            )
        value = _reading_chars(str(cue.get("text") or "")) / duration
        cps_samples.append(value)
        cps_offenders.append(_offender(case, i, cue, duration=duration, value=value))

    # --- metrics 1 and 4: boundary quality -------------------------------- #
    mid_bad = mid_eligible = 0
    forbidden_bad = forbidden_eligible = 0
    silence_breaks = forced_breaks = punctuation_breaks = no_alternative = 0
    mid_offenders: list[dict[str, Any]] = []
    forbidden_offenders: list[dict[str, Any]] = []

    for boundary in boundaries:
        if boundary is None:
            continue
        i = boundary.cue_index
        left_unit = units[boundary.left_unit]
        right_unit = units[boundary.right_unit]
        left_cue, right_cue = cues[i], cues[i + 1]
        is_lyric = lyric[i] or lyric[i + 1]

        # metric 1 -- length/format-driven boundaries only.
        if not (
            is_lyric
            or exempt("len_break_mid_phrase_rate", i)
            or exempt("len_break_mid_phrase_rate", i + 1)
        ):
            if boundary.gap >= offline_s:
                silence_breaks += 1
            else:
                mid_eligible += 1
                at_phrase_start = unit_offsets[boundary.right_unit] in starts
                explicit = _ends_with_break_punct(
                    str(left_unit.get("text") or "")
                ) or _starts_with_break_punct(str(right_unit.get("text") or ""))
                if multichar and not at_phrase_start and not explicit:
                    mid_bad += 1
                    mid_offenders.append(
                        _offender(
                            case,
                            i,
                            left_cue,
                            # Ranked by how far below the silence threshold the
                            # break sat: the quieter the pause, the more purely
                            # this was the layout's own decision.
                            value=max(0.0, offline_s - boundary.gap),
                            note=(
                                f"{_tail_token(str(left_cue.get('text') or ''), iso)}"
                                f" | {str(right_cue.get('text') or '')[:12]}"
                            ),
                        )
                    )

        # metric 4 -- boundaries a legal alternative could have improved.
        if (
            is_lyric
            or exempt("forbidden_end_rate", i)
            or exempt("forbidden_end_rate", i + 1)
        ):
            continue
        if boundary.gap >= vad_skip_s:
            forced_breaks += 1
            continue
        if _ends_with_break_punct(str(left_unit.get("text") or "")):
            punctuation_breaks += 1
            continue
        if not has_legal_alternative(
            str(left_cue.get("text") or ""),
            str(right_cue.get("text") or ""),
            iso,
            max_line_length,
            max_lines,
        ):
            no_alternative += 1
            continue
        forbidden_eligible += 1
        tail = _tail_token(str(left_cue.get("text") or ""), iso)
        if tail and _line_end_penalty(tail, iso) >= 2:
            forbidden_bad += 1
            forbidden_offenders.append(
                # Ranked the same way: a dangling particle with no pause behind
                # it is a worse line end than one the speaker nearly justified.
                _offender(
                    case,
                    i,
                    left_cue,
                    value=max(0.0, vad_skip_s - boundary.gap),
                    note=tail,
                )
            )

    measurement.ratios = {
        "len_break_mid_phrase_rate": cc.Ratio(mid_bad, mid_eligible),
        "over_7s_rate": cc.Ratio(over_bad, over_eligible),
        "forbidden_end_rate": cc.Ratio(forbidden_bad, forbidden_eligible),
    }
    measurement.cps_samples = cps_samples
    # Trimmed per case: the global worst ``OFFENDER_LIMIT`` is a subset of the
    # union of the per-case worst ``OFFENDER_LIMIT``, so this loses nothing.
    measurement.offenders = {
        "len_break_mid_phrase_rate": _worst(mid_offenders),
        "over_7s_rate": _worst(duration_offenders),
        "cps_p90": _worst(cps_offenders),
        "forbidden_end_rate": _worst(forbidden_offenders),
    }
    measurement.diagnostics = {
        "unit_count": len(units),
        "cue_count": len(cues),
        "lyric_cue_count": sum(lyric),
        "internal_boundaries": max(len(cues) - 1, 0),
        "unmapped_boundaries": unmapped,
        "overlapping_cues": overlaps,
        "silence_breaks": silence_breaks,
        "forced_breaks": forced_breaks,
        "punctuation_breaks": punctuation_breaks,
        "no_legal_alternative": no_alternative,
        "exempted_cues": over_exempt,
        "phrase_granularity": "phrase" if multichar else "word",
        "target_cps": float(thresholds["cps"]),
        "max_cue_s": max_cue_s,
        "dependency_drift": _dependency_drift(case),
        "config_drift": _diarize_config_drift(case),
        "replay": dict(result.diagnostics),
    }
    return measurement


def _default_line_length(iso: str) -> int:
    from voxweave.core.layout import default_max_line_length

    return default_max_line_length(iso)


def _default_lines(iso: str) -> int:
    from voxweave.core.layout import default_max_lines

    return default_max_lines(iso)


def _dependency_drift(case: Case) -> list[str]:
    """Segmenter versions that moved since the case was captured (informational).

    A different jieba does not invalidate the *case*; it changes where breaks
    land, which is exactly why the report has to say so out loud rather than let
    the number shift under a silent upgrade.
    """
    recorded = case.doc["capture"].get("dependency_versions") or {}
    live = dependency_versions()
    live["python"] = platform.python_version()
    drift: list[str] = []
    for name, version in sorted(recorded.items()):
        if name not in live:
            continue
        current = live[name]
        if name == "python":
            if _python_minor(str(version)) != _python_minor(str(current)):
                drift.append(f"python: case {version} != running {current}")
            continue
        if version != current:
            drift.append(f"{name}: case {version!r} != installed {current!r}")
    return drift


def _worst(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``OFFENDER_LIMIT`` highest-``value`` rows, worst first."""
    return sorted(rows, key=lambda row: float(row["value"]), reverse=True)[
        :OFFENDER_LIMIT
    ]


def _offender(
    case: Case,
    index: int,
    cue: Mapping[str, Any],
    *,
    value: float,
    duration: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    start, end = float(cue["start"]), float(cue["end"])
    out: dict[str, Any] = {
        "case": case.id,
        "language": case.language,
        "cue_index": index,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration if duration is not None else end - start, 3),
        "value": round(float(value), 4),
        "text": _flat(str(cue.get("text") or ""))[:80],
    }
    if note:
        out["note"] = note[:60]
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(measurements: Sequence[CaseMeasurement]) -> dict[str, dict[str, Any]]:
    """Micro-aggregate every case into the ``all`` summary and one group per language.

    Numerators and denominators are summed; per-case percentages are never
    averaged. ``cps_p90`` pools the raw cue samples, because a percentile of
    per-case percentiles is not a percentile of anything.
    """
    agg = cc.MicroAggregator()
    case_counts: dict[str, int] = {}
    cue_counts: dict[str, int] = {}
    targets: dict[str, set[float]] = {}
    granularity: dict[str, set[str]] = {}
    diagnostics: dict[str, dict[str, int]] = {}

    for measurement in measurements:
        groups = cc.group_keys(measurement.language)
        for group in groups:
            case_counts[group] = case_counts.get(group, 0) + 1
            cue_counts[group] = cue_counts.get(group, 0) + measurement.cue_count
            for metric, ratio in measurement.ratios.items():
                agg.add_ratio(group, metric, ratio.bad, ratio.eligible)
            agg.add_samples(group, "cps_p90", measurement.cps_samples)
            bucket = diagnostics.setdefault(group, {})
            for key, value in measurement.diagnostics.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    bucket[key] = bucket.get(key, 0) + value
            targets.setdefault(group, set()).add(
                float(measurement.diagnostics["target_cps"])
            )
            granularity.setdefault(group, set()).add(
                str(measurement.diagnostics["phrase_granularity"])
            )

    out: dict[str, dict[str, Any]] = {}
    for group in sorted(set(case_counts) | {GROUP_ALL, *cc.CALIBRATION_LANGUAGES}):
        samples = agg.samples(group, "cps_p90")
        target_set = targets.get(group, set())
        # The ``all`` group mixes languages with different reading-speed targets,
        # so it has no single ceiling -- which is why gates run per language.
        target = next(iter(target_set)) if len(target_set) == 1 else None
        block: dict[str, Any] = {
            "case_count": case_counts.get(group, 0),
            "cue_count": cue_counts.get(group, 0),
            "cps_p90": {
                "n": len(samples),
                "value": cc.percentile(samples, 90.0),
                "median": cc.percentile(samples, 50.0),
                "p95": cc.percentile(samples, 95.0),
                "target_cps": target,
                "absolute_max": (
                    round(target * CPS_ABSOLUTE_FACTOR, 4)
                    if target is not None
                    else None
                ),
            },
            "phrase_granularity": sorted(granularity.get(group, ())),
            "diagnostics": diagnostics.get(group, {}),
        }
        for metric in METRICS:
            if metric == "cps_p90":
                continue
            block[metric] = agg.ratio(group, metric).to_dict()
        out[group] = block
    return out


def collect_offenders(
    measurements: Sequence[CaseMeasurement],
) -> dict[str, list[dict[str, Any]]]:
    """The ``OFFENDER_LIMIT`` worst samples per metric, worst first.

    A rate with no examples is un-actionable: the report must be able to answer
    "show me the cue" without a second run.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for metric in METRICS:
        pool: list[dict[str, Any]] = []
        for measurement in measurements:
            pool.extend(measurement.offenders.get(metric, ()))
        pool.sort(key=lambda row: float(row["value"]), reverse=True)
        out[metric] = pool[:OFFENDER_LIMIT]
    return out


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def _measure(block: Mapping[str, Any], metric: str) -> tuple[float | None, int, str]:
    """The compared quantity, its sample count and its unit, for one metric."""
    if metric == "cps_p90":
        cps = block["cps_p90"]
        return cps["value"], int(cps["n"]), "cps"
    ratio = block[metric]
    if metric in COUNT_METRICS:
        return float(ratio["bad"]), int(ratio["eligible"]), "count"
    return ratio["value"], int(ratio["eligible"]), "rate"


def _absolute_max(
    gate: Mapping[str, Any], block: Mapping[str, Any], metric: str
) -> float | None:
    """The absolute ceiling, resolving ``cps_p90``'s per-language derivation."""
    if metric == "cps_p90":
        return block["cps_p90"].get("absolute_max")
    value = gate.get("absolute_max")
    return None if value is None else float(value)


def evaluate_gates(
    groups: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """One-sided comparison per language group; ``all`` is a summary only.

    Every rule is lower-is-better, so an improvement can never fail::

        allowed = baseline + max(absolute_tolerance, baseline * relative_tolerance)
        passed  = value <= allowed and (absolute_max is None or value <= absolute_max)

    A group whose denominator is under ``min_samples`` reports
    ``insufficient_samples``: with the corpus fixed at 20 cases that is a corpus
    defect, not a pass, and the caller turns it into exit 2 for a blocking gate.
    """
    results: list[dict[str, Any]] = []
    baseline_groups = (baseline or {}).get("groups") or {}
    for language in cc.CALIBRATION_LANGUAGES:
        block = groups.get(language)
        if block is None:
            continue
        for metric in METRICS:
            gate = gates.get(metric) or DEFAULT_GATES[metric]
            mode = str(gate.get("mode", "warning"))
            value, samples, unit = _measure(block, metric)
            ceiling = _absolute_max(gate, block, metric)
            result: dict[str, Any] = {
                "group": language,
                "metric": metric,
                "mode": mode,
                "measure": unit,
                "value": value,
                "samples": samples,
                "min_samples": int(gate["min_samples"]),
                "absolute_max": ceiling,
                "baseline_value": None,
                "allowed_by_baseline": None,
                "reasons": [],
            }
            if mode == "disabled":
                result["status"] = "disabled"
                results.append(result)
                continue
            if samples < int(gate["min_samples"]):
                result["status"] = "insufficient_samples"
                result["reasons"].append(
                    f"{samples} samples < min_samples {gate['min_samples']}"
                )
                results.append(result)
                continue
            if value is None:
                result["status"] = "insufficient_samples"
                result["reasons"].append("metric has no value (empty denominator)")
                results.append(result)
                continue

            passed = True
            if ceiling is not None and value > ceiling + DURATION_EPS:
                passed = False
                result["reasons"].append(
                    f"absolute: {value:.4f} > absolute_max {ceiling:.4f}"
                )
            base_block = baseline_groups.get(language)
            if base_block is not None:
                base_value, _, _ = _measure(base_block, metric)
                if base_value is not None:
                    allowed = float(base_value) + max(
                        float(gate["absolute_tolerance"]),
                        float(base_value) * float(gate["relative_tolerance"]),
                    )
                    result["baseline_value"] = base_value
                    result["allowed_by_baseline"] = allowed
                    if value > allowed + DURATION_EPS:
                        passed = False
                        result["reasons"].append(
                            f"regression: {value:.4f} > allowed {allowed:.4f} "
                            f"(baseline {base_value:.4f})"
                        )
            result["status"] = "pass" if passed else "fail"
            results.append(result)
    return results


def gate_exit_code(results: Sequence[Mapping[str, Any]]) -> int:
    """Map gate outcomes onto the shared contract.

    ``insufficient_samples`` on a blocking gate is exit 2, not exit 1: the corpus
    could not answer the question, so this run has no standing to call it a
    regression (design 4.6).
    """
    blocking = [r for r in results if r["mode"] == "blocking"]
    if any(r["status"] == "insufficient_samples" for r in blocking):
        return cc.EXIT_INVALID
    if any(r["status"] == "fail" for r in blocking):
        return cc.EXIT_GATE_FAILED
    return cc.EXIT_OK


# --------------------------------------------------------------------------- #
# Baseline handling
# --------------------------------------------------------------------------- #


def load_baseline(path: str | Path, corpus: Corpus) -> dict[str, Any]:
    """Load a baseline and refuse to compare against one that does not fit.

    A digest, metric-definition or segmenter-version mismatch is exit 2 and
    demands a reviewed ``record-baseline``. Silently comparing today's corpus to
    yesterday's numbers is how a gate stops meaning anything.
    """
    baseline = cc.load_validated_json(path, BASELINE_SCHEMA, label=str(path))
    problems: list[str] = []
    if baseline["metric_definition_version"] != METRIC_DEFINITION_VERSION:
        problems.append(
            f"metric_definition_version {baseline['metric_definition_version']} "
            f"!= {METRIC_DEFINITION_VERSION} implemented here"
        )
    if baseline["corpus_digest"] != corpus.digest:
        problems.append(
            f"corpus_digest {baseline['corpus_digest'][:12]}... != current "
            f"{corpus.digest[:12]}... (the corpus changed)"
        )
    if problems:
        raise cc.CalibrationError(
            f"{path} does not describe this corpus",
            [*problems, "re-record deliberately: make quality-record-segmentation"],
        )
    return baseline


def environment_drift(baseline: Mapping[str, Any]) -> list[str]:
    """Segmenter/python differences between the baseline and this environment."""
    recorded = baseline.get("environment") or {}
    drift: list[str] = []
    base_python = str(recorded.get("python") or "")
    if base_python and _python_minor(base_python) != _python_minor(
        platform.python_version()
    ):
        drift.append(
            f"python: baseline {base_python} != running {platform.python_version()}"
        )
    live = dependency_versions()
    for name, version in sorted((recorded.get("dependencies") or {}).items()):
        if name in live and live[name] != version:
            drift.append(f"{name}: baseline {version!r} != installed {live[name]!r}")
    return drift


def baseline_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project a report onto the tracked baseline shape (groups + gate rules)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "corpus_digest": report["corpus_digest"],
        "generated_from_commit": report["generated_from_commit"],
        "environment": report["environment"],
        "groups": report["groups"],
        "gates": report["gates"],
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def group_schema() -> dict[str, Any]:
    """The baseline schema's ``group`` definition, usable as a standalone schema.

    There is no separate report schema: a report's groups are the same objects a
    baseline records, so they are held to the same contract instead of a second,
    drifting one.
    """
    schema = cc.load_schema(BASELINE_SCHEMA)
    return {**schema["$defs"]["group"], "$defs": schema["$defs"]}


def gates_schema() -> dict[str, Any]:
    schema = cc.load_schema(BASELINE_SCHEMA)
    return {
        **schema["properties"]["gates"],
        "$defs": schema["$defs"],
    }


def validate_report(report: Mapping[str, Any]) -> None:
    """Hold a report to the tracked contracts it shares with the baseline."""
    errors: list[str] = []
    group_def = group_schema()
    for name, block in report["groups"].items():
        errors.extend(f"groups/{name}/{e}" for e in cc.schema_errors(block, group_def))
    errors.extend(
        f"gates/{e}" for e in cc.schema_errors(report["gates"], gates_schema())
    )
    if errors:
        raise cc.CalibrationError("the generated report is not schema-valid", errors)


def build_report(
    corpus: Corpus,
    measurements: Sequence[CaseMeasurement],
    groups: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any]],
    gate_results: Sequence[Mapping[str, Any]],
    *,
    partial: bool,
    private: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the report the gates were evaluated on -- one aggregation, not two."""
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "kind": "segmentation-report",
        "corpus": {
            "path": str(corpus.path),
            "case_count": len(corpus.cases),
            "total_bytes": corpus.total_bytes,
            "required_counts": corpus.required_counts,
        },
        "corpus_digest": corpus.digest,
        "generated_from_commit": repo_commit(),
        "environment": environment_block(),
        "partial": partial,
        "groups": dict(groups),
        "gates": {metric: dict(gates[metric]) for metric in METRICS},
        "gate_results": [dict(r) for r in gate_results],
        "cases": [
            {
                "id": m.case_id,
                "language": m.language,
                "cue_count": m.cue_count,
                "wall_time_s": round(m.wall_time_s, 4),
                "metrics": {
                    **{name: ratio.to_dict() for name, ratio in m.ratios.items()},
                    "cps_p90": {
                        "n": len(m.cps_samples),
                        "value": cc.percentile(m.cps_samples, 90.0),
                    },
                },
                "diagnostics": m.diagnostics,
            }
            for m in measurements
        ],
        "offenders": collect_offenders(measurements),
        "warnings": list(warnings),
    }
    slowest = max(measurements, key=lambda m: m.wall_time_s, default=None)
    report["timing"] = {
        "total_wall_s": round(sum(m.wall_time_s for m in measurements), 4),
        "slowest_case": slowest.case_id if slowest else None,
        "slowest_wall_s": round(slowest.wall_time_s, 4) if slowest else None,
        "case_wall_target_s": CASE_WALL_TARGET_S,
        "cases_over_target": [
            m.case_id for m in measurements if m.wall_time_s > CASE_WALL_TARGET_S
        ],
    }
    if private is not None:
        report["private"] = dict(private)
    validate_report(report)
    return report


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #


def run_cases(cases: Sequence[Case]) -> list[CaseMeasurement]:
    measurements: list[CaseMeasurement] = []
    for case in cases:
        started = time.perf_counter()
        result = replay(case)
        measurement = measure_case(case, result)
        measurement.wall_time_s = time.perf_counter() - started
        measurements.append(measurement)
    return measurements


def private_corpus_path() -> Path | None:
    """``VOXWEAVE_CALIB_ROOT``'s segmentation registry, when one is configured.

    A private corpus is reported in its own block and never touches a gate: it
    must not change the denominator of a public PR gate (design 4.4).
    """
    root = os.environ.get("VOXWEAVE_CALIB_ROOT", "").strip()
    if not root:
        return None
    candidate = Path(root) / "segmentation" / "corpus.json"
    return candidate if candidate.is_file() else None


def evaluate_private() -> dict[str, Any] | None:
    path = private_corpus_path()
    if path is None:
        return None
    corpus = load_corpus(path, strict_size=False)
    measurements = run_cases(corpus.cases)
    return {
        "path": str(path),
        "corpus_digest": corpus.digest,
        "case_count": len(corpus.cases),
        "groups": aggregate(measurements),
        "gated": False,
    }


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ratio_cell(ratio: Mapping[str, Any]) -> str:
    """``rate (bad/eligible)`` -- a percentage never appears without its counts."""
    return f"{_fmt(ratio['value'])} ({ratio['bad']}/{ratio['eligible']})"


def print_summary(report: Mapping[str, Any]) -> None:
    groups = report["groups"]
    print(f"corpus   : {report['corpus']['path']}")
    print(f"digest   : {report['corpus_digest'][:16]}...")
    print(f"cases    : {report['corpus']['case_count']}")
    print()
    print(
        f"  {'group':<6} {'cases':>5} {'cues':>6}  {'mid-phrase':<20}"
        f"  {'over-max-cue':<14} {'cps_p90':>8}  {'bad-end':<20}"
    )
    print("  " + "-" * 84)
    for name in (GROUP_ALL, *cc.CALIBRATION_LANGUAGES):
        block = groups.get(name)
        if block is None or not block["case_count"]:
            continue
        over = block["over_7s_rate"]
        over_cell = "{}/{}".format(over["bad"], over["eligible"])
        print(
            f"  {name:<6} {block['case_count']:>5} {block['cue_count']:>6}"
            f"  {_ratio_cell(block['len_break_mid_phrase_rate']):<20}"
            f"  {over_cell:<14}"
            f" {_fmt(block['cps_p90']['value'], 2):>8}"
            f"  {_ratio_cell(block['forbidden_end_rate']):<20}"
        )
    results = report["gate_results"]
    if results:
        print()
        print("  gates")
        for result in results:
            marker = {
                "pass": "ok  ",
                "fail": "FAIL",
                "insufficient_samples": "n<min",
                "disabled": "off ",
            }.get(str(result["status"]), "?   ")
            reasons = "; ".join(result["reasons"])
            print(
                f"    [{marker}] {result['group']:<3} {result['metric']:<26}"
                f" {result['mode']:<8} value={_fmt(result['value'])}"
                + (f"  {reasons}" if reasons else "")
            )
    for warning in report.get("warnings") or ():
        print(f"  warning: {warning}")


def machine_summary(
    status: str, cases: int, failures: int, warnings: int, report_path: str
) -> str:
    return (
        f"QUALITY segmentation status={status} cases={cases} failures={failures} "
        f"warnings={warnings} report={report_path}"
    )


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_validate_corpus(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    print(f"corpus   : {corpus.path}")
    print(f"cases    : {len(corpus.cases)} ({corpus.total_bytes} bytes tracked)")
    print(f"digest   : {corpus.digest}")
    for case in corpus.cases:
        print(
            f"  {case.id:<8} {case.language:<3} {len(case.units):>5} units  "
            f"{case.size_bytes:>7}B  {', '.join(case.tags)}"
        )
    print(machine_summary("pass", len(corpus.cases), 0, 0, "-"))
    return cc.EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    cases = list(corpus.cases)
    partial = False
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.id in wanted]
        unknown = wanted - {c.id for c in cases}
        if unknown:
            raise cc.CalibrationError(
                f"no such case in {corpus.path}: {', '.join(sorted(unknown))}"
            )
        partial = True

    warnings: list[str] = []
    baseline: dict[str, Any] | None = None
    gates: dict[str, Mapping[str, Any]] = dict(DEFAULT_GATES)
    if args.baseline:
        baseline = load_baseline(args.baseline, corpus)
        gates = dict(baseline["gates"])
        drift = environment_drift(baseline)
        if drift:
            if args.allow_environment_drift:
                warnings.extend(f"environment drift ignored: {d}" for d in drift)
            else:
                raise cc.CalibrationError(
                    f"{args.baseline} was recorded in a different environment",
                    [
                        *drift,
                        "segmenter versions move where breaks land",
                        "re-record deliberately, or pass --allow-environment-drift",
                    ],
                )

    measurements = run_cases(cases)
    for measurement in measurements:
        warnings.extend(
            f"{measurement.case_id}: {d}"
            for d in measurement.diagnostics["dependency_drift"]
        )
        if measurement.diagnostics["unmapped_boundaries"]:
            warnings.append(
                f"{measurement.case_id}: "
                f"{measurement.diagnostics['unmapped_boundaries']} cue boundaries "
                "could not be mapped back to source units"
            )

    groups = aggregate(measurements)
    if partial:
        gate_results: list[dict[str, Any]] = []
        warnings.append("partial run (--case): gates skipped, not a baseline candidate")
    else:
        gate_results = evaluate_gates(groups, gates, baseline)

    report = build_report(
        corpus,
        measurements,
        groups,
        gates,
        gate_results,
        partial=partial,
        private=evaluate_private() if args.private else None,
        warnings=warnings,
    )

    destination = "-"
    if args.json_out:
        destination = str(cc.write_json(args.json_out, report))

    print_summary(report)

    failures = sum(
        1 for r in gate_results if r["mode"] == "blocking" and r["status"] != "pass"
    )
    warned = sum(
        1 for r in gate_results if r["mode"] == "warning" and r["status"] != "pass"
    )
    code = gate_exit_code(gate_results) if args.check else cc.EXIT_OK
    if code == cc.EXIT_INVALID:
        print(
            machine_summary("invalid", len(measurements), failures, warned, destination)
        )
        raise cc.CalibrationError(
            "a blocking gate has fewer samples than it requires",
            [
                f"{r['group']}/{r['metric']}: {r['reasons'][0] if r['reasons'] else ''}"
                for r in gate_results
                if r["mode"] == "blocking" and r["status"] == "insufficient_samples"
            ],
        )
    status = "fail" if code == cc.EXIT_GATE_FAILED else "pass"
    print(machine_summary(status, len(measurements), failures, warned, destination))
    return code


def cmd_record_baseline(args: argparse.Namespace) -> int:
    """Promote a report to the tracked baseline. Never run by CI, by design.

    Refuses a partial report, a stale digest or a metric-definition mismatch, so
    a regression cannot be laundered into a new baseline by rerunning the
    harness until the numbers look acceptable.
    """
    corpus = load_corpus(args.corpus)
    report = cc.read_json(args.report)
    problems: list[str] = []
    if report.get("kind") != "segmentation-report":
        problems.append(f"{args.report} is not a segmentation report")
    if report.get("partial"):
        problems.append("report is partial (--case); record from a full run")
    if report.get("metric_definition_version") != METRIC_DEFINITION_VERSION:
        problems.append(
            f"report metric_definition_version {report.get('metric_definition_version')}"
            f" != {METRIC_DEFINITION_VERSION}"
        )
    if report.get("corpus_digest") != corpus.digest:
        problems.append(
            f"report corpus_digest {str(report.get('corpus_digest'))[:12]}... != "
            f"current corpus {corpus.digest[:12]}... (re-run evaluate first)"
        )
    if not report.get("generated_from_commit"):
        problems.append(
            "report has no generated_from_commit (run inside a git checkout)"
        )
    if problems:
        raise cc.CalibrationError(f"refusing to record {args.output}", problems)

    baseline = baseline_from_report(report)
    cc.validate_or_exit2(baseline, BASELINE_SCHEMA, label=str(args.output))
    path = cc.write_json(args.output, baseline)
    print(f"recorded {path}")
    print(f"  corpus_digest : {baseline['corpus_digest']}")
    print(f"  commit        : {baseline['generated_from_commit']}")
    for metric in METRICS:
        gate = baseline["gates"][metric]
        print(f"  gate {metric:<26} mode={gate['mode']}")
    print(machine_summary("pass", len(corpus.cases), 0, 0, str(path)))
    return cc.EXIT_OK


_MEDIA_EXTS = {".mkv", ".mp4", ".webm", ".mov", ".m4v", ".avi", ".ts"}


def _sibling_json(media: Path) -> Path:
    """Sibling ``.json``, replacing only the trailing extension.

    Never ``Path.with_suffix``: a name with an interior dot would be truncated at
    the first one (same contract as ``pipeline.swap_ext``).
    """
    return media.with_name(media.name[: -len(media.suffix)] + ".json")


def _case_from_sibling(media: Path, document: Mapping[str, Any]) -> Case:
    """Wrap a private sibling JSON as an in-memory case (never written to disk).

    Legacy convenience only: it lets the same four metrics run over private media
    that can never be tracked in git. The document is not schema-validated as a
    golden case, because it is not one.
    """
    from voxweave.config import gap_thresholds
    from voxweave.core.layout import default_max_line_length, default_max_lines

    iso = cc.require_calibration_language(document.get("language"))
    th = gap_thresholds(iso)
    config = {
        "language": iso,
        "max_line_length": default_max_line_length(iso),
        "max_lines": default_max_lines(iso),
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
        "shot_snap_s": float(th["shot_snap_s"]),
    }
    units = [
        {
            "id": str(i),
            "text": str(u.get("text") or ""),
            "start": float(u["start"]),
            "end": float(u["end"]),
        }
        for i, u in enumerate(document.get("word_segments") or ())
        if u.get("start") is not None and u.get("end") is not None
    ]
    if not units:
        raise cc.CalibrationError(f"{media}: sibling JSON has no usable word_segments")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "id": f"{iso}-00",
        "language": iso,
        "tags": ["private"],
        "capture": {
            "window_duration_s": max(u["end"] for u in units) + 0.5,
            "dependency_versions": {},
            "config": config,
        },
        "word_segments": units,
        "vad_speech": [list(s) for s in document.get("vad_speech") or ()],
        "shot_changes": list(document.get("shot_changes") or ()),
        "sing_spans": [list(s) for s in document.get("sing_spans") or ()],
        "speaker_turns": [
            {"start": float(a), "end": float(b), "speaker": str(label)}
            for a, b, label in document.get("speaker_turns") or ()
        ],
    }
    return Case(media, media.name, doc, 0)


def cmd_compare_video_dir(args: argparse.Namespace) -> int:
    """Legacy: run the corpus metrics over a directory of private media siblings.

    This is the knob-validation path referenced from ``gap_split.adaptive_clause_ms``
    and ``pipeline._maybe_adaptive_thresholds``: run it twice, once with
    ``VOXWEAVE_GAP_ADAPTIVE=1``, and compare the four numbers. It reads sibling
    JSON only -- no subtitle track is extracted and no ASS is parsed here; a
    commercial release track is the *alignment* ruler's ground truth, not this
    one's (design 3.1).
    """
    directory = Path(args.directory)
    if not directory.is_dir():
        raise cc.CalibrationError(f"not a directory: {directory}")
    media_files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _MEDIA_EXTS
    )
    measurements: list[CaseMeasurement] = []
    skipped: list[str] = []
    for media in media_files:
        sibling = _sibling_json(media)
        if not sibling.is_file():
            continue
        try:
            document = cc.read_json(sibling)
            case = _case_from_sibling(media, document)
            started = time.perf_counter()
            measurement = measure_case(case, replay(case))
            measurement.wall_time_s = time.perf_counter() - started
            measurement.case_id = media.stem
        except cc.CalibrationError as exc:
            skipped.append(f"{media.name}: {exc.message}")
            continue
        measurements.append(measurement)
        ratios = measurement.ratios
        cps = cc.percentile(measurement.cps_samples, 90.0)
        print(
            f"  {media.stem[:34]:<34} cues={measurement.cue_count:>4} "
            f"mid={_fmt(ratios['len_break_mid_phrase_rate'].value)} "
            f"over={ratios['over_7s_rate'].bad}/{ratios['over_7s_rate'].eligible} "
            f"cps_p90={_fmt(cps, 2)} "
            f"end={_fmt(ratios['forbidden_end_rate'].value)}"
        )
    if not measurements:
        raise cc.CalibrationError(
            f"no media file in {directory} has a usable sibling JSON",
            skipped or ["nothing matched the known media extensions"],
        )
    print()
    for name, block in aggregate(measurements).items():
        if not block["case_count"]:
            continue
        mid = block["len_break_mid_phrase_rate"]
        over = block["over_7s_rate"]
        bad = block["forbidden_end_rate"]
        print(
            f"  {name:<5} cases={block['case_count']:>3} cues={block['cue_count']:>5} "
            f"mid={_fmt(mid['value'])} ({mid['bad']}/{mid['eligible']}) "
            f"over={over['bad']}/{over['eligible']} "
            f"cps_p90={_fmt(block['cps_p90']['value'], 2)} "
            f"end={_fmt(bad['value'])} ({bad['bad']}/{bad['eligible']})"
        )
    for line in skipped:
        print(f"  skipped {line}")
    print(machine_summary("pass", len(measurements), 0, 0, "-"))
    return cc.EXIT_OK


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calib_segmentation.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate-corpus", help="schema, coverage, size and digest checks only"
    )
    validate.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    validate.set_defaults(func=cmd_validate_corpus)

    evaluate = sub.add_parser(
        "evaluate", help="replay the corpus, measure, compare against the baseline"
    )
    evaluate.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    evaluate.add_argument(
        "--baseline",
        default=None,
        help="tracked baseline to compare against; omitted = absolute gates only",
    )
    evaluate.add_argument("--json-out", default=None, help="where to write the report")
    evaluate.add_argument(
        "--case",
        action="append",
        default=None,
        help="restrict to one case id (repeatable); produces a partial report",
    )
    evaluate.add_argument(
        "--check",
        action="store_true",
        help="apply the gates: exit 1 when a blocking gate regresses",
    )
    evaluate.add_argument(
        "--private",
        action="store_true",
        help="also evaluate $VOXWEAVE_CALIB_ROOT/segmentation (reported, never gated)",
    )
    evaluate.add_argument(
        "--allow-environment-drift",
        action="store_true",
        help="downgrade a segmenter-version mismatch with the baseline to a warning",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    record = sub.add_parser(
        "record-baseline", help="promote a report to the tracked baseline (never in CI)"
    )
    record.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    record.add_argument("--report", required=True)
    record.add_argument("--output", default=str(DEFAULT_BASELINE))
    record.set_defaults(func=cmd_record_baseline)

    legacy = sub.add_parser(
        "compare-video-dir",
        help="legacy: same metrics over a directory of private media siblings",
    )
    legacy.add_argument("directory")
    legacy.set_defaults(func=cmd_compare_video_dir)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    cc.run_cli(main)
