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
    Of the internal boundaries that had a legal in-budget source-lattice
    alternative, plus eligible document-final tails, how many leave a
    forward-binding token dangling at the end of a cue. The rate is reported,
    but its gate compares the raw bad count against ``baseline_bad + 1`` because
    one event is larger than a stable rate tolerance at this corpus size.

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
    shadow            P5: measure the full finalizer lane/row matrix beside v1
    compare-video-dir legacy: run the same metrics over private media siblings

The last stdout line is always a machine summary::

    QUALITY segmentation status=pass cases=20 failures=0 warnings=0 report=...
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import importlib.util
import math
import os
import platform
import random
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
METRIC_DEFINITION_VERSION = 3

DEFAULT_CORPUS = REPO_ROOT / "calibration" / "segmentation" / "corpus.json"
DEFAULT_BASELINE = REPO_ROOT / "calibration" / "segmentation" / "baseline.json"
DEFAULT_COARSE_CORPUS = (
    REPO_ROOT / "calibration" / "segmentation" / "corpus-coarse.json"
)
DEFAULT_AUTHORIZED_DEFERRALS = (
    REPO_ROOT / "calibration" / "segmentation" / "p5-authorized-deferrals.txt"
)

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
#: ``forbidden_end_rate`` is also a count gate: at current language-group
#: denominators a single boundary is about one percentage point, so a fractional
#: tolerance below that quantum merely turns every event into a regression.
COUNT_METRICS = frozenset({"over_7s_rate", "forbidden_end_rate"})

#: One newly bad forbidden tail is the smallest honest tolerance. The rate and
#: its numerator/denominator remain report columns; only the compared quantity
#: changes to ``bad <= baseline_bad + FORBIDDEN_END_BAD_SLACK``.
FORBIDDEN_END_BAD_SLACK = 1.0

JA_TAIL_LENS_LEVEL2 = "ja-unidic-level2"
JA_TAIL_LENS_LEVEL1 = "ja-char-table-level1"

#: Initial gate table (design 4.6). ``warning`` is the landing mode: the soak
#: phase fixes metric/corpus problems without blocking PRs. Once both the
#: baseline and current language group reach ``min_samples``, evaluation promotes
#: that group's warning result to blocking without mutating this tracked policy.
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
        "absolute_max": None,
        "absolute_tolerance": FORBIDDEN_END_BAD_SLACK,
        "relative_tolerance": 0.0,
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

#: Ceiling on the share of internal cue boundaries the mapper may fail to
#: resolve before the whole case is an invalid *measurement* (exit 2), not a
#: quality result. Every unresolved boundary is a boundary no metric saw, so a
#: run that loses more than this is silently grading a different cue stream.
_UNMAPPED_MAX_RATIO = 0.01

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


def forbidden_end_ja_lens() -> dict[str, Any]:
    """Identity of the Japanese tail scorer this process will actually use.

    ``provider_snapshot`` reads the same cached tagger state as
    :func:`kinsoku.ja_pos_end_penalties`, including ``VOXWEAVE_JA_POS=0``. A
    Level-2 run still names its documented per-offset Level-1 fallback: MeCab
    and the source-unit lattice need not agree on every token boundary.
    """
    from voxweave.core.providers import provider_snapshot

    pos = provider_snapshot("ja")["pos"]
    enabled = bool(pos.get("ja_pos_enabled"))
    return {
        "id": JA_TAIL_LENS_LEVEL2 if enabled else JA_TAIL_LENS_LEVEL1,
        "source": (
            "kinsoku.ja_pos_end_penalties" if enabled else "kinsoku.line_end_penalty"
        ),
        "provider": pos.get("provider"),
        "provider_version": pos.get("version"),
        "dictionary": pos.get("dict"),
        "context": "punctuated-source-phrase-atom",
        "missing_offset_fallback": JA_TAIL_LENS_LEVEL1 if enabled else None,
    }


def metric_definition_block() -> dict[str, Any]:
    """Machine-readable definition details that can vary without a code edit.

    The integer definition version covers reviewed semantic changes. The lens
    identity additionally binds a baseline to the actual optional provider and
    dictionary selected at runtime, so a Level-2 baseline can never be compared
    with a Level-1 fallback run under the same integer version.
    """
    return {
        "version": METRIC_DEFINITION_VERSION,
        "forbidden_end": {
            "tail_scope": "eligible-internal-plus-document-final",
            "alternative_source": "pre-split-punctuated-source-phrase-lattice",
            "reported_measure": "rate-with-bad-and-eligible",
            "gate_measure": "bad-count",
            "baseline_bad_slack": FORBIDDEN_END_BAD_SLACK,
            "ja_tail_lens": forbidden_end_ja_lens(),
        },
    }


def metric_definition_digest() -> str:
    """Digest of :func:`metric_definition_block` for baseline compatibility."""
    return cc.canonical_digest(metric_definition_block())


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


def _ja_pos_end_penalties(text: str) -> dict[int, int] | None:
    """The production Level-2 scorer, wrapped as a testable ruler seam."""
    from voxweave.core.kinsoku import ja_pos_end_penalties

    return ja_pos_end_penalties(text)


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


def _ends_with_terminal_punct(text: str) -> bool:
    """True when a source tail closes the document with sentence punctuation."""
    stripped = str(text).rstrip().rstrip(_CLOSERS)
    return bool(stripped) and stripped[-1] in _TERMINAL_PUNCT


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
    so ``(start, end)`` is a stable identity for unit-level entries. The
    length-break repack path (``_chunk_to_cue``) instead emits *atom*-level
    entries whose span aggregates several units -- ``(start, end)`` then matches
    no single unit, but the entry's start still names the atom's first unit and
    its end the atom's last unit, so lookup falls back to those edge indexes
    (this was the silent-exclusion mechanism behind up to 10% unmapped
    boundaries on len-break-heavy zh cases). Lookup prefers the first candidate
    at or after a monotonic cursor, which disambiguates the repeated keys that
    zero-duration units produce. An entry the splitter fabricated (the logged
    proportional-timing desync path) still resolves to nothing, and its boundary
    is excluded and counted -- never guessed at.
    """

    def __init__(self, units: Sequence[Mapping[str, Any]]) -> None:
        self._index: dict[tuple[float, float], list[int]] = {}
        self._by_start: dict[float, list[int]] = {}
        self._by_end: dict[float, list[int]] = {}
        for i, unit in enumerate(units):
            key = self._key(unit)
            self._index.setdefault(key, []).append(i)
            self._by_start.setdefault(key[0], []).append(i)
            self._by_end.setdefault(key[1], []).append(i)

    @staticmethod
    def _key(unit: Mapping[str, Any]) -> tuple[float, float]:
        start, end = unit.get("start"), unit.get("end")
        return (
            round(float(start), _TIME_DECIMALS) if start is not None else math.nan,
            round(float(end), _TIME_DECIMALS) if end is not None else math.nan,
        )

    @staticmethod
    def _pick(candidates: list[int] | None, cursor: int) -> int | None:
        if not candidates:
            return None
        for i in candidates:
            if i >= cursor:
                return i
        return candidates[-1]

    def locate_first(self, entry: Mapping[str, Any], cursor: int) -> int | None:
        """Index of the unit this entry *starts* on (edge fallback for atoms)."""
        key = self._key(entry)
        found = self._pick(self._index.get(key), cursor)
        if found is not None:
            return found
        return self._pick(self._by_start.get(key[0]), cursor)

    def locate_last(self, entry: Mapping[str, Any], cursor: int) -> int | None:
        """Index of the unit this entry *ends* on (edge fallback for atoms)."""
        key = self._key(entry)
        found = self._pick(self._index.get(key), cursor)
        if found is not None:
            return found
        return self._pick(self._by_end.get(key[1]), cursor)


@dataclass(frozen=True)
class Boundary:
    """One internal cue boundary, resolved onto the source units."""

    cue_index: int
    left_unit: int
    right_unit: int
    span_start_unit: int
    span_end_unit: int
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
        first = locator.locate_first(word_data[0], cursor)
        last = locator.locate_last(
            word_data[-1], first if first is not None else cursor
        )
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
        out.append(
            Boundary(
                cue_index=i,
                left_unit=left,
                right_unit=right,
                span_start_unit=firsts[i] if firsts[i] is not None else left,
                span_end_unit=(lasts[i + 1] if lasts[i + 1] is not None else right),
                gap=gap,
            )
        )
    return out, unmapped


# --------------------------------------------------------------------------- #
# Source-unit health: the zero-duration ledger split by mechanism
# --------------------------------------------------------------------------- #

#: A zero-duration span, with microsecond slack for JSON float round-tripping.
_ZERO_S = 1e-6

#: Adjacent lexical units further apart than this, with no punctuation between
#: them, are counted in the ledger. Natural mid-sentence pauses land here too
#: (ASR under-punctuates), so the count is data, not an alarm.
_STRANDED_GAP_S = 1.0

#: Only a gap this extreme between directly adjacent word units earns a warning
#: line: a pause that long without punctuation is not phrasing, it is the
#: aligner parking a word tail on a later speech island.
_STRANDED_WARN_S = 5.0

#: A single lexical unit longer than this gets its VAD coverage checked: a held
#: vowel is fine, a unit stretched across silence is an alignment overhang.
_LONG_UNIT_S = 1.0

#: Wall/run sizes at or above this are worth a warning line; mirrors
#: ``voxweave.realign.ZERO_DURATION_MAX_RUN`` (the repair pass gives up past
#: this run length, so anything this long survived into the shipped units).
_ZERO_SHAPE_WARN = 8


def _is_lexical(unit: Mapping[str, Any]) -> bool:
    return any(ch.isalnum() for ch in str(unit.get("text") or ""))


def _vad_coverage(
    start: float, end: float, spans: Sequence[tuple[float, float]]
) -> float:
    dur = end - start
    if dur <= 0.0:
        return 1.0
    covered = sum(max(0.0, min(end, b) - max(start, a)) for a, b in spans)
    return covered / dur


def unit_health(
    units: Sequence[Mapping[str, Any]],
    vad_speech: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """The zero-duration ledger split by mechanism, plus alignment-shape checks.

    A single zero-duration rate conflates three mechanisms that mean different
    things: ``reinject_punct``'s by-design zero-width punctuation, lexical
    collapse (the zh NAR aligner's known failure, which lands runs of units on
    one identical timestamp), and point-alignment quantization (the ja MMS
    lane, whose zeros are ordered and unique). The shape columns tell them
    apart: a same-time wall is collapse evidence even at a low overall rate,
    while a high rate with no wall and short runs is quantization. Stranded
    gaps and undercovered long units are the aligner parking part of a word
    somewhere the speech is not.
    """
    spans = [(float(a), float(b)) for a, b in vad_speech or ()]
    lexical = [u for u in units if _is_lexical(u)]
    punct_zero = sum(
        1
        for u in units
        if not _is_lexical(u) and float(u["end"]) - float(u["start"]) <= _ZERO_S
    )

    zero_starts: dict[float, int] = {}
    lexical_zero = 0
    zero_run = zero_run_max = 0
    long_count = 0
    long_min_coverage: float | None = None
    for u in lexical:
        start, end = float(u["start"]), float(u["end"])
        if end - start <= _ZERO_S:
            lexical_zero += 1
            zero_run += 1
            zero_run_max = max(zero_run_max, zero_run)
            key = round(start, _TIME_DECIMALS)
            zero_starts[key] = zero_starts.get(key, 0) + 1
        else:
            zero_run = 0
        if end - start > _LONG_UNIT_S:
            long_count += 1
            if spans:
                coverage = _vad_coverage(start, end, spans)
                if long_min_coverage is None or coverage < long_min_coverage:
                    long_min_coverage = coverage

    # Stream-adjacent lexical pairs only: a punctuation unit between two words
    # marks a legitimate sentence pause, so it exempts the gap. Two directly
    # adjacent word units this far apart mean the aligner parked the right one
    # on a later speech island (the stranded-tail signature).
    stranded_count = 0
    stranded_max = 0.0
    for a, b in zip(units, units[1:]):
        if not (_is_lexical(a) and _is_lexical(b)):
            continue
        gap = float(b["start"]) - float(a["end"])
        if gap > _STRANDED_GAP_S:
            stranded_count += 1
            stranded_max = max(stranded_max, gap)

    nonmonotonic = sum(
        1 for a, b in zip(units, units[1:]) if float(b["start"]) < float(a["start"])
    )
    return {
        "lexical_count": len(lexical),
        "lexical_zero": lexical_zero,
        "punct_zero": punct_zero,
        "same_time_wall_max": max(zero_starts.values(), default=0),
        "lexical_zero_run_max": zero_run_max,
        "nonmonotonic_pairs": nonmonotonic,
        "stranded_gap_count": stranded_count,
        "stranded_gap_max_s": round(stranded_max, 3),
        "long_unit_count": long_count,
        "long_unit_min_vad_coverage": (
            round(long_min_coverage, 3) if long_min_coverage is not None else None
        ),
    }


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


def _source_span_text(
    units: Sequence[Mapping[str, Any]], start_unit: int, end_unit: int, iso: str
) -> str:
    """One inclusive, punctuated source span in the phrase-lattice join mode."""
    join = "" if _no_space(iso) else " "
    return join.join(
        str(unit.get("text") or "") for unit in units[start_unit : end_unit + 1]
    )


def _forbidden_tail_penalty(
    units: Sequence[Mapping[str, Any]],
    unit_offsets: Sequence[int],
    *,
    start_unit: int,
    end_unit: int,
    context_end_unit: int | None = None,
    display_text: str,
    iso: str,
    ja_pos_cache: dict[str, dict[int, int] | None] | None,
) -> int:
    """Score one cue/candidate tail with the language's metric lens.

    Japanese locates the atom carrying this tail in the pre-split phrase lattice
    of the complete *punctuated source prefix*, then runs Level 2 on that source
    atom. It never reads a display token. ``context_end_unit`` lets a mapped
    lexical tail retain punctuation-only units omitted from ``word_data``
    (``お、 | うじゃ``), while the lookup stays on ``end_unit`` rather than
    grading the comma. The right-hand lexical head is never included. Maps are
    cached by atom text. When POS is unavailable, or the tail offset lands
    inside a MeCab token, this follows the production scorer's documented
    Level-1 char-table fallback. Other languages retain their display-tail
    surface lens.
    """
    if iso != "ja":
        return _line_end_penalty(_tail_token(display_text, iso), iso)
    context_end = end_unit if context_end_unit is None else context_end_unit
    if not (
        0 <= start_unit <= end_unit <= context_end < len(units)
        and len(unit_offsets) == len(units)
    ):
        return 0
    source_tail = str(units[end_unit].get("text") or "")
    source_text = _source_span_text(units, start_unit, context_end, iso)
    target_offset = (
        unit_offsets[end_unit]
        - unit_offsets[start_unit]
        + _reading_chars(source_tail)
        - 1
    )
    atom_start = 0
    for atom in _phrase_atoms(source_text, iso):
        atom_width = _reading_chars(atom)
        if atom_width <= 0:
            continue
        if atom_start <= target_offset < atom_start + atom_width:
            if ja_pos_cache is not None and atom in ja_pos_cache:
                pos_penalties = ja_pos_cache[atom]
            else:
                pos_penalties = _ja_pos_end_penalties(atom)
                if ja_pos_cache is not None:
                    ja_pos_cache[atom] = pos_penalties
            if pos_penalties is not None:
                pos_penalty = pos_penalties.get(target_offset - atom_start)
                if pos_penalty is not None:
                    return int(pos_penalty)
            break
        atom_start += atom_width
    return _line_end_penalty(source_tail, iso)


def has_legal_alternative(
    units: Sequence[Mapping[str, Any]],
    phrase_starts: set[int],
    unit_offsets: Sequence[int],
    *,
    span_start_unit: int,
    actual_right_unit: int,
    span_end_unit: int,
    iso: str,
    max_line_length: int,
    max_lines: int,
    ja_pos_cache: dict[str, dict[int, int] | None] | None = None,
) -> bool:
    """True when a source-lattice alternative fits the same two-cue budget.

    The denominator of ``forbidden_end_rate`` is the boundaries where a better
    choice existed. Candidate cuts come from the phrase lattice computed once
    over the pre-split source stream, not from re-segmenting each already-split
    cue without its original context. An alternative counts only if both source
    spans render non-empty, both fit the configured line budget, and the new
    left half does *not* end on a forward-binding token. Without this, a cue
    whose only in-budget break is a bad one would be scored as an algorithm
    defect, and a rate that punishes unsolvable boundaries can never reach 0.
    """
    if not (
        0 <= span_start_unit < actual_right_unit <= span_end_unit < len(units)
        and len(unit_offsets) == len(units)
    ):
        return False
    join = "" if _no_space(iso) else " "
    from voxweave.core.layout import strip_punct_for_subtitles

    for unit_cut in range(span_start_unit + 1, span_end_unit + 1):
        if unit_cut == actual_right_unit:
            continue
        if unit_offsets[unit_cut] not in phrase_starts:
            continue
        lhs = strip_punct_for_subtitles(
            join.join(
                str(unit.get("text") or "") for unit in units[span_start_unit:unit_cut]
            )
        )
        rhs = strip_punct_for_subtitles(
            join.join(
                str(unit.get("text") or "")
                for unit in units[unit_cut : span_end_unit + 1]
            )
        )
        if not lhs.strip() or not rhs.strip():
            continue
        if (
            _forbidden_tail_penalty(
                units,
                unit_offsets,
                start_unit=span_start_unit,
                end_unit=unit_cut - 1,
                display_text=lhs,
                iso=iso,
                ja_pos_cache=ja_pos_cache,
            )
            >= 2
        ):
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
    internal = max(len(cues) - 1, 0)
    if internal and unmapped / internal > _UNMAPPED_MAX_RATIO:
        raise cc.CalibrationError(
            f"{case.relpath}: {unmapped}/{internal} cue boundaries could not be"
            " mapped back to source units",
            [
                "the metrics cannot see this much of the cue stream, so the run is"
                " an invalid measurement, not a quality result",
                "likely a word_data provenance change in the splitter -- fix the"
                " mapper, do not relax this gate",
            ],
        )
    starts, unit_offsets = phrase_start_offsets(units, iso)
    ja_pos_cache: dict[str, dict[int, int] | None] | None = {} if iso == "ja" else None
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
    final_tail_eligible = terminal_final_tails = 0
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
            units,
            starts,
            unit_offsets,
            span_start_unit=boundary.span_start_unit,
            actual_right_unit=boundary.right_unit,
            span_end_unit=boundary.span_end_unit,
            iso=iso,
            max_line_length=max_line_length,
            max_lines=max_lines,
            ja_pos_cache=ja_pos_cache,
        ):
            no_alternative += 1
            continue
        forbidden_eligible += 1
        tail = _tail_token(str(left_cue.get("text") or ""), iso)
        if tail and (
            _forbidden_tail_penalty(
                units,
                unit_offsets,
                start_unit=boundary.span_start_unit,
                end_unit=boundary.left_unit,
                context_end_unit=boundary.right_unit - 1,
                display_text=str(left_cue.get("text") or ""),
                iso=iso,
                ja_pos_cache=ja_pos_cache,
            )
            >= 2
        ):
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

    # A document-final tail has no following gap and no movable two-cue
    # boundary. After the ordinary lyric/manual-exception filters it is therefore
    # always eligible unless the source explicitly closes with sentence-final
    # punctuation. A clause comma is not a document terminator and remains
    # scoreable. With no pause evidence, a bad final tail gets the maximum
    # boundary-band severity for offender ordering.
    if cues:
        i = len(cues) - 1
        if not (lyric[i] or exempt("forbidden_end_rate", i)):
            source_tail = str(units[-1].get("text") or "") if units else ""
            if _ends_with_terminal_punct(source_tail):
                terminal_final_tails += 1
            else:
                final_tail_eligible += 1
                forbidden_eligible += 1
                tail = _tail_token(str(cues[i].get("text") or ""), iso)
                if tail and (
                    _forbidden_tail_penalty(
                        units,
                        unit_offsets,
                        start_unit=0,
                        end_unit=len(units) - 1,
                        display_text=str(cues[i].get("text") or ""),
                        iso=iso,
                        ja_pos_cache=ja_pos_cache,
                    )
                    >= 2
                ):
                    forbidden_bad += 1
                    forbidden_offenders.append(
                        _offender(
                            case,
                            i,
                            cues[i],
                            value=vad_skip_s,
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
        "final_tail_eligible": final_tail_eligible,
        "terminal_final_tails": terminal_final_tails,
        "exempted_cues": over_exempt,
        "phrase_granularity": "phrase" if multichar else "word",
        "unit_health": unit_health(
            units,
            [(float(a), float(b)) for a, b in case.doc.get("vad_speech") or ()],
        ),
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

    For ``forbidden_end_rate``, ``value`` is the raw bad count and its configured
    tolerances reduce that formula to ``baseline_bad + 1``. The result also
    carries the reported rate and its numerator/denominator explicitly.

    A group whose denominator is under ``min_samples`` reports
    ``insufficient_samples``: with the corpus fixed at 20 cases that is a corpus
    defect, not a pass, and the caller turns it into exit 2 for a blocking gate.
    A configured warning promotes to blocking for one language only when both
    its baseline and current sample counts reach ``min_samples``. The gate table
    remains unchanged, so promotion is evidence-driven and reversible per run.
    """
    results: list[dict[str, Any]] = []
    baseline_groups = (baseline or {}).get("groups") or {}
    for language in cc.CALIBRATION_LANGUAGES:
        block = groups.get(language)
        if block is None:
            continue
        for metric in METRICS:
            gate = gates.get(metric) or DEFAULT_GATES[metric]
            configured_mode = str(gate.get("mode", "warning"))
            value, samples, unit = _measure(block, metric)
            min_samples = int(gate["min_samples"])
            base_block = baseline_groups.get(language)
            baseline_samples: int | None = None
            if base_block is not None:
                _, baseline_samples, _ = _measure(base_block, metric)
            promoted = bool(
                configured_mode == "warning"
                and samples >= min_samples
                and baseline_samples is not None
                and baseline_samples >= min_samples
            )
            mode = "blocking" if promoted else configured_mode
            ceiling = _absolute_max(gate, block, metric)
            result: dict[str, Any] = {
                "group": language,
                "metric": metric,
                "mode": mode,
                "configured_mode": configured_mode,
                "promoted": promoted,
                "measure": unit,
                "value": value,
                "samples": samples,
                "baseline_samples": baseline_samples,
                "min_samples": min_samples,
                "absolute_max": ceiling,
                "baseline_value": None,
                "allowed_by_baseline": None,
                "reasons": [],
            }
            if metric == "forbidden_end_rate":
                ratio = block[metric]
                result.update(
                    {
                        "numerator": int(ratio["bad"]),
                        "denominator": int(ratio["eligible"]),
                        "reported_rate": ratio["value"],
                    }
                )
            if mode == "disabled":
                result["status"] = "disabled"
                results.append(result)
                continue
            if samples < min_samples:
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
    live_definition = metric_definition_block()
    live_definition_digest = cc.canonical_digest(live_definition)
    if baseline["metric_definition_version"] != METRIC_DEFINITION_VERSION:
        problems.append(
            f"metric_definition_version {baseline['metric_definition_version']} "
            f"!= {METRIC_DEFINITION_VERSION} implemented here"
        )
    if baseline["metric_definition_digest"] != live_definition_digest:
        recorded_lens = (
            baseline.get("metric_definition", {})
            .get("forbidden_end", {})
            .get("ja_tail_lens", {})
            .get("id")
        )
        live_lens = live_definition["forbidden_end"]["ja_tail_lens"]["id"]
        problems.append(
            "metric_definition_digest "
            f"{baseline['metric_definition_digest'][:12]}... != current "
            f"{live_definition_digest[:12]}... (ja lens {recorded_lens!r} != "
            f"{live_lens!r}, or another metric-definition detail changed)"
        )
    elif baseline["metric_definition"] != live_definition:
        problems.append(
            "metric_definition content does not match its current digest definition"
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
        "metric_definition": report["metric_definition"],
        "metric_definition_digest": report["metric_definition_digest"],
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


def metric_definition_schema() -> dict[str, Any]:
    schema = cc.load_schema(BASELINE_SCHEMA)
    return {**schema["$defs"]["metric_definition"], "$defs": schema["$defs"]}


def validate_report(report: Mapping[str, Any]) -> None:
    """Hold a report to the tracked contracts it shares with the baseline."""
    errors: list[str] = []
    group_def = group_schema()
    for name, block in report["groups"].items():
        errors.extend(f"groups/{name}/{e}" for e in cc.schema_errors(block, group_def))
    errors.extend(
        f"gates/{e}" for e in cc.schema_errors(report["gates"], gates_schema())
    )
    errors.extend(
        f"metric_definition/{e}"
        for e in cc.schema_errors(
            report["metric_definition"], metric_definition_schema()
        )
    )
    if report["metric_definition_digest"] != cc.canonical_digest(
        report["metric_definition"]
    ):
        errors.append("metric_definition_digest does not match metric_definition")
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
    definition = metric_definition_block()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "metric_definition": definition,
        "metric_definition_digest": cc.canonical_digest(definition),
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


def _gate_value_cell(result: Mapping[str, Any]) -> str:
    """Human gate claim, preserving forbidden-end counts beside its rate."""
    if result["metric"] == "forbidden_end_rate":
        value = result.get("value")
        count = "n/a" if value is None else str(int(float(value)))
        return (
            f"bad_count={count} rate={_fmt(result.get('reported_rate'))} "
            f"({result.get('numerator')}/{result.get('denominator')})"
        )
    return f"value={_fmt(result.get('value'))}"


def print_summary(report: Mapping[str, Any]) -> None:
    groups = report["groups"]
    lens = report["metric_definition"]["forbidden_end"]["ja_tail_lens"]
    print(f"corpus   : {report['corpus']['path']}")
    print(f"digest   : {report['corpus_digest'][:16]}...")
    print(
        f"metric   : {report['metric_definition_digest'][:16]}... ja-tail={lens['id']}"
    )
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
                f" {result['mode']:<8} {_gate_value_cell(result)}"
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
        health = measurement.diagnostics["unit_health"]
        if health["same_time_wall_max"] >= _ZERO_SHAPE_WARN:
            warnings.append(
                f"{measurement.case_id}: {health['same_time_wall_max']} lexical"
                " units collapsed onto one timestamp (aligner collapse wall)"
            )
        if health["lexical_zero_run_max"] >= _ZERO_SHAPE_WARN:
            warnings.append(
                f"{measurement.case_id}: run of {health['lexical_zero_run_max']}"
                " consecutive zero-duration lexical units"
            )
        if health["nonmonotonic_pairs"]:
            warnings.append(
                f"{measurement.case_id}: {health['nonmonotonic_pairs']}"
                " non-monotonic source unit pairs"
            )
        if health["stranded_gap_max_s"] >= _STRANDED_WARN_S:
            warnings.append(
                f"{measurement.case_id}: stranded word tail"
                f" ({health['stranded_gap_max_s']}s between adjacent word units;"
                f" {health['stranded_gap_count']} gaps over {_STRANDED_GAP_S}s)"
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
    live_definition = metric_definition_block()
    live_definition_digest = cc.canonical_digest(live_definition)
    if report.get("kind") != "segmentation-report":
        problems.append(f"{args.report} is not a segmentation report")
    if report.get("partial"):
        problems.append("report is partial (--case); record from a full run")
    if report.get("metric_definition_version") != METRIC_DEFINITION_VERSION:
        problems.append(
            f"report metric_definition_version {report.get('metric_definition_version')}"
            f" != {METRIC_DEFINITION_VERSION}"
        )
    if report.get("metric_definition") != live_definition:
        problems.append(
            "report metric_definition does not match the scorer selected in this process"
        )
    if report.get("metric_definition_digest") != live_definition_digest:
        problems.append(
            "report metric_definition_digest does not match the current metric definition"
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
# P5 shadow harness: optimizer, finalizer, speaker counterfactual, and comparators
# --------------------------------------------------------------------------- #
#
# The shadow ships nothing. ``segment_document`` returns v1's cues either way and
# writes the same bytes; with ``VOXWEAVE_SEG_V2_SHADOW=1`` it *also* returns a
# measurement artifact on ``result.shadow``. Everything below reads that artifact
# and answers the independent coverage, correctness, non-inferiority, and
# perturbation questions in the P5 gate law:
#
#   N4/N5  did v2 optimize the whole corpus without an unwaived contract breach?
#   N1/N3  are finalizer/v2 and the v1 isolation treatment non-inferior?
#   N7/N11/N19  are the preview, classifier, and speaker projections verified?
#   P1-P3  are perturbations evaluable and confined to their declared classes?
#
# None of it touches the quality report: this writes its own file, and no shadow
# number ever enters ``report["groups"]`` (``baseline_from_report`` copies that
# block verbatim into the tracked baseline, so a v2 number parked there would
# silently become the v1 gate's reference).

SHADOW_SCHEMA_VERSION = 2
SHADOW_REPORT_KIND = "segmentation-shadow-report"

#: The hook's opt-in. Pinned around the replay and restored afterwards, exactly
#: as ``_forced_gap_adaptive`` pins the adaptive-threshold knob: an operator's
#: shell must neither turn the shadow off for a shadow run nor leak it into the
#: quality run that may follow in the same process.
SHADOW_ENV = "VOXWEAVE_SEG_V2_SHADOW"

#: P5's lane/row matrix, named by ``pipeline.SHADOW_LANE_*``.
SHADOW_LANE_CORE = "core_partition_pre_overlay"
SHADOW_LANE_DELIVERY = "delivery_v1_legacy"
SHADOW_LANE_FINALIZER = "delivery_finalizer"
SHADOW_LANE_LEGACY_DISPLAY = "legacy_display"
SHADOW_LANES = (
    SHADOW_LANE_CORE,
    SHADOW_LANE_DELIVERY,
    SHADOW_LANE_FINALIZER,
    SHADOW_LANE_LEGACY_DISPLAY,
)
SHADOW_ENGINES = ("v1", "v2")
SHADOW_LANE_ROWS: dict[str, tuple[str, ...]] = {
    SHADOW_LANE_CORE: SHADOW_ENGINES,
    SHADOW_LANE_DELIVERY: SHADOW_ENGINES,
    SHADOW_LANE_FINALIZER: ("v1", "v2", "v2-speaker-off"),
    SHADOW_LANE_LEGACY_DISPLAY: ("v1",),
}

#: N1's binding W4 retarget. The tracked baseline remains the shipped delivery
#: stream, while the candidate is now the complete P5 finalizer/v2 row. The
#: legacy-proxy lane survives solely as the byte tripwire and P4 continuity
#: evidence; the core lane remains ungated boundary evidence.
SHADOW_GATED_LANE = SHADOW_LANE_FINALIZER
SHADOW_GATED_ROW = "v2"

#: N1: frozen before the optimizer was written, and deliberately a literal
#: rather than a read of ``baseline["gates"]``. The gate table is what "not
#: worse" *means*; loading it from the same file the comparison targets would let
#: a future baseline edit silently redefine the P5 acceptance criterion. A test
#: asserts these modes still match the tracked baseline, so a deliberate edit is
#: visible as a failing test rather than as a quietly moved goalpost.
#: The shared ``evaluate_gates`` still applies per-group sample promotion to the
#: warning literal, exactly as it does for the production quality report.
SHADOW_GATES: dict[str, dict[str, Any]] = {
    "len_break_mid_phrase_rate": {
        "direction": "lower_is_better",
        "mode": "blocking",
        "absolute_max": 0.10,
        "absolute_tolerance": 0.01,
        "relative_tolerance": 0.10,
        "min_samples": 100,
    },
    "over_7s_rate": {
        "direction": "lower_is_better",
        "mode": "blocking",
        "absolute_max": 0.0,
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "min_samples": 1,
    },
    "cps_p90": {
        "direction": "lower_is_better",
        "mode": "blocking",
        "absolute_max": None,
        "absolute_tolerance": 0.5,
        "relative_tolerance": 0.05,
        "min_samples": 100,
    },
    "forbidden_end_rate": {
        "direction": "lower_is_better",
        "mode": "warning",
        "absolute_max": None,
        "absolute_tolerance": FORBIDDEN_END_BAD_SLACK,
        "relative_tolerance": 0.0,
        "min_samples": 100,
    },
}

DEFAULT_SHADOW_REPORT = (
    REPO_ROOT / "build" / "calibration" / "segmentation-shadow-report.json"
)

#: AD-2. Mirrors ``boundary_lattice.INFLUENCE_RADIUS_UNITS``; asserted equal by a
#: test rather than imported, so the harness keeps working in an environment
#: where the optimizer is not importable and a divergence is still caught.
INFLUENCE_RADIUS_UNITS = 96

PERTURB_MODES = ("single_gap", "global_jitter")
PERTURB_MAGNITUDES_MS = (10, 20, 50)
PERTURB_SIGNS = (-1, 1)
#: AD4-3: every near-cliff gap is probed exhaustively, plus this share of the
#: rest, chosen by a seed derived from the case and the magnitude.
PERTURB_SAMPLE_RATE = 0.10
#: Global jitter is aggregate-only (no probe unit, so no influence cell); a
#: handful of seeded draws per magnitude is enough to see a stability cliff.
PERTURB_JITTER_DRAWS = 3

#: One-at-a-time weight ablation: term name -> the ``boundary_cost`` constants
#: that make it up. ``pause_cut`` is absent on purpose -- its amplitude reaches
#: the ramp through a *keyword default* bound at definition time, so rebinding
#: ``W_PAUSE`` would not change a single score. It is ablated by zeroing the term
#: function instead, which is what "this term contributes nothing" means anyway.
ABLATION_WEIGHTS: dict[str, tuple[str, ...]] = {
    "balance": ("W_BALANCE",),
    "cue_base": ("CUE_BASE",),
    "line_count": ("W_LINE_COUNT",),
    "migration": ("W_MIGRATION",),
    "min_duration": ("W_MIN_DURATION",),
    "particle": ("W_PARTICLE",),
    "pos": ("W_POS",),
    "punct_affinity": ("W_PUNCT_AFFINITY",),
    "reading": ("W_READING",),
    "sentence_cross": ("W_SENTENCE_CROSS",),
    "short_fragment": ("SHORT_FRAGMENT_TIGHT", "SHORT_FRAGMENT_LOOSE"),
    "shot_preview": ("W_SHOT_PREVIEW",),
}
ABLATION_TERM_PAUSE = "pause_cut"
ABLATION_TERM_SPEAKER = "speaker"
ABLATION_TERMS = (
    *sorted(ABLATION_WEIGHTS),
    ABLATION_TERM_PAUSE,
    ABLATION_TERM_SPEAKER,
)
# Fixed metric-domain exclusions for the tracked ablation registry.  These are
# input/counterfactual facts, not results filtered after a replay: with the
# short-fragment term genuinely zero, the captured zero-time collapse in each
# case becomes a standalone delivered cue, for which CPS has no finite value.
# Keeping one common 18-case denominator across every OAT row makes deltas
# comparable; the selection is serialized in the report and an empty selection
# still fails completeness.
ABLATION_TRACKED_EXCLUSIONS: dict[str, str] = {
    "ja-02": "short_fragment=0 isolates zero-time cues; CPS is undefined",
    "ja-03": "short_fragment=0 isolates a zero-time cue; CPS is undefined",
}


@contextlib.contextmanager
def _forced_shadow(enabled: bool) -> Iterator[None]:
    """Pin the shadow flag around one call and restore the operator's value.

    Written with the literal ``"0"``/``"1"`` the hook parses, because the hook
    tests the string exactly: a paired ``--no-`` flag elsewhere in the tree
    writes ``"0"``, which is truthy as a string, so anything looser would latch
    the shadow on for a run that asked for it off.
    """
    previous = os.environ.get(SHADOW_ENV)
    os.environ[SHADOW_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SHADOW_ENV, None)
        else:
            os.environ[SHADOW_ENV] = previous


def replay_shadow(case: Case) -> Any:
    """Replay one case with the shadow hook armed."""
    with _forced_shadow(True):
        return replay(case)


def shadow_artifact_of(case: Case, result: Any) -> dict[str, Any]:
    """The artifact off one replay, or a typed invalid-measurement failure.

    Three distinct not-a-result cases, all exit 2 rather than a zero: an absent
    artifact means the hook did not fire, ``error`` means the hook caught an
    exception (it may never fail the pipeline, so it records instead), and
    ``invalid_profile`` is AD3-2's refusal to interpret a knob that has no
    meaning. A quality number derived from any of them would be a fiction.
    """
    artifact = getattr(result, "shadow", None)
    if not isinstance(artifact, Mapping):
        raise cc.CalibrationError(
            f"{case.relpath}: the replay returned no shadow artifact",
            [
                f"the hook is gated on {SHADOW_ENV}=1 and this run pinned it on",
                "the installed voxweave predates the P5 hook, or the flag was"
                " consumed by a different process",
            ],
        )
    if "error" in artifact:
        error = artifact["error"]
        raise cc.CalibrationError(
            f"{case.relpath}: the shadow lane failed on this document",
            [f"{error.get('type')}: {error.get('detail')}"],
        )
    if "invalid_profile" in artifact:
        raise cc.CalibrationError(
            f"{case.relpath}: the display profile is not interpretable (AD3-2)",
            [
                f"{v.get('key')}={v.get('value')}: {v.get('reason')}"
                for v in artifact["invalid_profile"]
            ],
        )
    from voxweave.core.shadow_schema import validate_shadow_v2_payload

    schema_errors = validate_shadow_v2_payload(artifact)
    if schema_errors:
        raise cc.CalibrationError(
            f"{case.relpath}: the live shadow artifact is not schema 2",
            schema_errors,
        )
    return dict(artifact)


class LaneStream:
    """Adapter that lets ``measure_case`` read one artifact lane.

    ``measure_case`` wants a ``SegmentationResult``; a lane is a list of display
    rows plus a unit range per row. Rebuilding ``word_data`` from that range is
    not a shortcut around the mapper -- the range *is* the partition the lane
    committed to, and handing the mapper the units it names measures exactly the
    stream the optimizer produced. Both engines go through this same adapter, so
    neither is measured on evidence the other did not get.
    """

    def __init__(
        self, cues: list[dict[str, Any]], diagnostics: Mapping[str, Any] | None = None
    ) -> None:
        self.cues = cues
        self.diagnostics: dict[str, Any] = dict(diagnostics or {})


def lane_cue_stream(
    rows: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]] | None:
    """Artifact rows -> cue dicts the metric code can read, or ``None``.

    ``None`` when any row has no resolved unit range: a partition the artifact
    could not project is a stream no metric can see, and inventing a mapping for
    it would report a number about a partition nobody chose.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        span = row.get("unit_range")
        if span is None:
            return None
        low, high = int(span[0]), int(span[1])
        out.append(
            {
                "text": row.get("text") or "",
                "start": row.get("start"),
                "end": row.get("end"),
                "lyric": bool(row.get("lyric")),
                "word_data": [
                    {
                        "text": unit.get("text"),
                        "start": unit.get("start"),
                        "end": unit.get("end"),
                    }
                    for unit in units[low:high]
                ],
            }
        )
    return out


@dataclass
class LaneResult:
    """One (lane, engine) pair measured, or the reason it could not be."""

    lane: str
    engine: str
    projection: str
    cue_count: int
    measurement: CaseMeasurement | None = None
    error: str | None = None


def measure_lane(
    case: Case, artifact: Mapping[str, Any], lane: str, engine: str
) -> LaneResult:
    """Run the four metrics over one lane of one case's artifact."""
    lane_block = artifact["lanes"][lane]
    block = lane_block["rows"][engine] if "rows" in lane_block else lane_block[engine]
    result = LaneResult(
        lane=lane,
        engine=engine,
        projection=str(block.get("projection")),
        cue_count=int(block.get("cue_count") or 0),
    )
    if block.get("materialized") is False:
        result.error = str(block.get("reason") or "row was not materialized")
        return result
    cues = lane_cue_stream(block.get("cues") or (), case.units)
    if cues is None:
        result.error = f"partition unresolved ({result.projection})"
        return result
    try:
        result.measurement = measure_case(case, LaneStream(cues))
    except cc.CalibrationError as exc:
        # Kept as a per-lane fact rather than aborting the document: an
        # ungradable *core* lane still leaves the delivery lane measurable, and
        # the caller decides which lanes it is not allowed to lose.
        result.error = exc.message
    return result


# ------------------------------------------------- Wave A reads (no solving)


def seg_document_of(case: Case, result: Any) -> Any:
    """The ``SegDocument`` the replay minted, which the artifact is derived from."""
    document = getattr(result, "document", None)
    if document is None:
        raise cc.CalibrationError(
            f"{case.relpath}: the replay returned no SegDocument",
            ["the installed voxweave predates the P3 IR"],
        )
    return document


def barrier_unit_ids(document: Any) -> tuple[int, ...]:
    """The robust-silence barrier set of one document, in source-unit ids.

    Barriers are the only exogenous topology v2 has, so this is what the pinned
    perturbation lane freezes and what ``barrier_flips`` counts. Unit ids rather
    than atom nodes: a node index is an internal coordinate that a coalescing or
    relief change could renumber, and a frozen set has to survive that.
    """
    from voxweave.core.boundary_lattice import build_atom_layer, build_barriers

    layer = build_atom_layer(document)
    return tuple(
        sorted(
            barrier.unit_id
            for barrier in build_barriers(layer, document.profile)
            if barrier.kind == "robust-silence"
        )
    )


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "p50": cc.percentile(list(values), 50.0),
        "p90": cc.percentile(list(values), 90.0),
    }


def cut_feature_scan(document: Any, selected_units: Sequence[int]) -> dict[str, Any]:
    """Pause/linguistic features over the candidate space and over v2's choices.

    The artifact carries per-interval *sums* only: ``sum_breakdowns`` drops a
    categorical feature rather than inventing an aggregate for it, so
    ``vad_state`` -- the one feature that says what kind of evidence a boundary
    rested on -- survives only per cut. Recomputing it here from the same helpers
    the optimizer used is the cheapest honest way to get a distribution, and it
    costs one atom-layer build plus one pause evaluation per inter-atom gap.
    """
    from voxweave.core.boundary_cost import pause_evidence
    from voxweave.core.boundary_lattice import build_atom_layer

    layer = build_atom_layer(document)
    profile = document.profile
    speech = document.vad_speech
    chosen = set(selected_units)

    states: dict[str, int] = {}
    selected_states: dict[str, int] = {}
    effective: list[float] = []
    selected_effective: list[float] = []
    gaps: list[float] = []
    counts = {
        "candidates": 0,
        "particle_nonzero": 0,
        "pos_nonzero": 0,
        "punct_affinity": 0,
        "selected": 0,
        "selected_particle_nonzero": 0,
        "selected_punct_affinity": 0,
    }
    from voxweave.core.boundary_cost import PUNCT_AFFINITY_CHARS

    for node in range(1, len(layer.atoms)):
        left, right = layer.atoms[node - 1], layer.atoms[node]
        evidence = pause_evidence(
            left.end, right.start, speech_spans=speech, profile=profile
        )
        tail = left.text.rstrip()
        affinity = bool(tail) and tail[-1] in PUNCT_AFFINITY_CHARS
        particle = (left.end_pen + right.start_pen) != 0
        counts["candidates"] += 1
        counts["particle_nonzero"] += int(particle)
        counts["pos_nonzero"] += int(right.boundary_pen != 0)
        counts["punct_affinity"] += int(affinity)
        states[evidence.vad_state] = states.get(evidence.vad_state, 0) + 1
        if evidence.gap_ms_raw is not None:
            gaps.append(float(evidence.gap_ms_raw))
        if evidence.effective_ms is not None:
            effective.append(float(evidence.effective_ms))
        if layer.unit_bound(node) not in chosen:
            continue
        counts["selected"] += 1
        counts["selected_particle_nonzero"] += int(particle)
        counts["selected_punct_affinity"] += int(affinity)
        selected_states[evidence.vad_state] = (
            selected_states.get(evidence.vad_state, 0) + 1
        )
        if evidence.effective_ms is not None:
            selected_effective.append(float(evidence.effective_ms))
    return {
        "counts": counts,
        "effective_ms": effective,
        "gap_ms": gaps,
        "selected_effective_ms": selected_effective,
        "selected_vad_state": selected_states,
        "vad_state": states,
    }


def merge_feature_scans(scans: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool per-case scans into one distribution (counts sum, samples pool)."""
    counts: dict[str, int] = {}
    states: dict[str, int] = {}
    selected_states: dict[str, int] = {}
    effective: list[float] = []
    gaps: list[float] = []
    selected_effective: list[float] = []
    for scan in scans:
        for key, value in scan["counts"].items():
            counts[key] = counts.get(key, 0) + int(value)
        for key, value in scan["vad_state"].items():
            states[key] = states.get(key, 0) + int(value)
        for key, value in scan["selected_vad_state"].items():
            selected_states[key] = selected_states.get(key, 0) + int(value)
        effective.extend(float(v) for v in scan["effective_ms"])
        gaps.extend(float(v) for v in scan["gap_ms"])
        selected_effective.extend(float(v) for v in scan["selected_effective_ms"])
    return {
        "counts": dict(sorted(counts.items())),
        "effective_ms": _percentiles(effective),
        "gap_ms": _percentiles(gaps),
        "selected_effective_ms": _percentiles(selected_effective),
        "selected_vad_state": dict(sorted(selected_states.items())),
        "vad_state": dict(sorted(states.items())),
    }


# ----------------------------------------------------- C13 and the validator


#: The validator stages every valid measurement must carry. AD-4 runs the
#: checker at all three, and a stage that is simply *absent* used to read here as
#: "nothing to report" -- which made the whole exit driver blindable by deleting
#: one assignment in the hook, with no test and no schema noticing.
SHADOW_REQUIRED_STAGES = ("raw", "core", "legacy_overlay", "finalizer")


def shadow_violation_counts(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Every validator stage of one artifact, counted by kind and by origin.

    ``raw_duplicate_v1_cues`` is honoured here rather than left for a reader to
    notice: adjacent ``adopted_v1`` fallbacks can expand onto the same v1 cue, so
    a fallback-carrying document's *raw* stage sees that cue twice and reports a
    conservation violation about the reporting shape, not about the partition.
    Those rows are labelled and excluded from the C13 driver -- which costs
    nothing, because a document with a fallback already fails C13 on
    ``fallback_intervals`` alone.

    A missing mandatory stage is recorded in ``missing_stages`` and is an INVALID
    measurement for the caller, never a clean one. The distinction matters more
    than it looks: an absent stage and a stage that found nothing are reported
    identically by a counter that treats falsy as empty, so the difference
    between "the core validator says this partition is clean" and "the core
    validator never ran" would come down to which field a reader happened to
    open. The v1 lane stages stay optional -- an unprojectable v1 stream has no
    partition to check, which is a fact about v1, not a broken measurement.
    """
    from voxweave.core.partition_check import EXIT_DRIVING_STAGES

    duplicated = bool(artifact["validator"].get("raw_duplicate_v1_cues"))
    stages: dict[str, Any] = {}
    exit_driving: list[dict[str, Any]] = []
    missing: list[str] = []
    suppressed = 0
    sources: list[tuple[str, Any]] = [
        (name, artifact["validator"].get(name)) for name in SHADOW_REQUIRED_STAGES
    ]
    for lane, row_ids in SHADOW_LANE_ROWS.items():
        lane_block = artifact["lanes"][lane]
        for row_id in row_ids:
            block = (
                lane_block["rows"].get(row_id)
                if "rows" in lane_block
                else lane_block.get(row_id)
            )
            if not isinstance(block, Mapping):
                continue
            sources.append((f"{lane}:{row_id}", block.get("validator")))
    for name, block in sources:
        if not block:
            stages[name] = None
            if name in SHADOW_REQUIRED_STAGES:
                missing.append(name)
            continue
        kinds: dict[str, int] = {}
        for violation in block["violations"]:
            key = "{}/{}/{}".format(
                violation["origin"],
                violation["stage"],
                violation["kind"],
            )
            if violation["waived"]:
                key += "/waived"
            kinds[key] = kinds.get(key, 0) + 1
            if not violation["waived"] and violation["origin"] == "v2":
                # Single source of truth: partition_check owns the exit-driving
                # stage set; restating it as a literal here silently ignored any
                # stage added there (P5 adds "finalizer").
                if violation["stage"] not in EXIT_DRIVING_STAGES:
                    continue
                conservation = violation["kind"] in (
                    "text-conservation",
                    "unit-conservation",
                )
                if duplicated and violation["stage"] == "raw" and conservation:
                    suppressed += 1
                    continue
                exit_driving.append(
                    {
                        "cue_index": violation.get("cue_index"),
                        "detail": violation.get("detail"),
                        "kind": violation["kind"],
                        "stage": violation["stage"],
                    }
                )
        stages[name] = {
            "kinds": dict(sorted(kinds.items())),
            "violations": len(block["violations"]),
            "waivers": len(block["waivers"]),
        }
    return {
        "duplicate_v1_cues": duplicated,
        "exit_driving": exit_driving,
        "missing_stages": missing,
        "not_conservation_evidence": suppressed,
        "stages": stages,
    }


def _preview_fidelity_valid(block: Any) -> bool:
    """N7's complete scored-edge and selected-factory bridge predicate."""
    if not isinstance(block, Mapping):
        return False
    scored = int(block.get("scored_edges") or 0)
    checked = int(block.get("checked_edges") or 0)
    selected = block.get("selected_rows")
    return (
        scored > 0
        and checked == scored
        and not int(block.get("uncheckable_edges") or 0)
        and not (block.get("mismatches") or ())
        and isinstance(selected, Mapping)
        and all(
            isinstance(row := selected.get(row_id), Mapping)
            and int(row.get("edge_count") or 0) == int(row.get("cue_count") or 0)
            and not (row.get("mismatches") or ())
            for row_id in ("v2", "v2-speaker-off")
        )
    )


def shadow_measurement_errors(
    case: Case, artifact: Mapping[str, Any], violations: Mapping[str, Any]
) -> list[str]:
    """Reasons this document's shadow run is not a measurement at all (exit 2).

    Kept apart from :func:`c13_case_failures` on purpose. A C13 failure says v2
    did something wrong; every reason here says the run cannot support a verdict
    either way, and folding the two together would let an unmeasured stage read
    as a passing one.

    The checks cover:

    * a mandatory validator stage did not run -- see :func:`shadow_violation_counts`;
    * the two independent derivations of v2's partition (the solver's own cut
      list and structural surface reconciliation) disagree. That projection is
      not decorative: every measured row is keyed by its owned unit range, so a
      bad reconciliation moves CPS and mid-phrase measurements without moving a
      single boundary;
    * the v1 stream could not be projected onto source units. The tracked corpus
      is word-level and must project; if it does not, the comparison the whole
      report is built on has no v1 side;
    * N7's edge-by-edge preview audit or N19's selected speaker projection is
      incomplete or inconsistent.
    """
    from voxweave.core.shadow_schema import validate_shadow_v2_payload

    structural = validate_shadow_v2_payload(artifact)
    if structural:
        return [
            f"{case.id}: schema-2 structural error: {error}" for error in structural
        ]

    problems = [
        f"{case.id}: validator stage {name!r} did not run (AD-4 requires all of"
        f" {', '.join(SHADOW_REQUIRED_STAGES)})"
        for name in violations["missing_stages"]
    ]
    if artifact.get("schema_version") != 2:
        problems.append(
            f"{case.id}: live shadow schema is {artifact.get('schema_version')!r}, expected 2"
        )
    fidelity = artifact.get("preview_fidelity")
    if not isinstance(fidelity, Mapping):
        problems.append(f"{case.id}: N7 preview-fidelity audit is absent")
    else:
        scored = int(fidelity.get("scored_edges") or 0)
        checked = int(fidelity.get("checked_edges") or 0)
        uncheckable = int(fidelity.get("uncheckable_edges") or 0)
        mismatches = fidelity.get("mismatches") or ()
        if not _preview_fidelity_valid(fidelity):
            problems.append(
                f"{case.id}: N7 preview fidelity is incomplete "
                f"(scored={scored}, checked={checked}, "
                f"uncheckable={uncheckable}, mismatches={len(mismatches)})"
            )
    invalid_finalizers = artifact.get("invalid_finalizer_rows") or ()
    if invalid_finalizers:
        problems.append(
            f"{case.id}: finalizer budget/validity short-circuit on rows "
            + ", ".join(map(str, invalid_finalizers))
        )
    finalizer_rows = artifact["lanes"][SHADOW_LANE_FINALIZER].get("rows") or {}
    for row_id in SHADOW_LANE_ROWS[SHADOW_LANE_FINALIZER]:
        row = finalizer_rows.get(row_id)
        if not isinstance(row, Mapping) or row.get("materialized") is False:
            problems.append(f"{case.id}: finalizer row {row_id!r} was not materialized")
            continue
        finalizer = row.get("finalizer")
        if not isinstance(finalizer, Mapping):
            problems.append(f"{case.id}: finalizer row {row_id!r} has no report")
            continue
        for check in ("trace_errors", "stability_errors"):
            errors = finalizer.get(check) or ()
            if errors:
                problems.append(
                    f"{case.id}: finalizer row {row_id!r} has {len(errors)} {check}"
                )
        if not isinstance(row.get("validator"), Mapping):
            problems.append(
                f"{case.id}: finalizer row {row_id!r} has no finalizer-stage validator"
            )
    speaker = artifact.get("speaker_evidence")
    if not isinstance(speaker, Mapping):
        problems.append(f"{case.id}: speaker evidence block is absent")
    else:
        refusal = speaker.get("measurement_refusal")
        if refusal:
            problems.append(f"{case.id}: speaker measurement refused: {refusal}")
        measurements = {
            "v2": speaker.get("measurement"),
            "v2-speaker-off": speaker.get("off_row_measurement"),
        }
        raw_counts: set[int] = set()
        for row_id, measurement in measurements.items():
            if not isinstance(measurement, Mapping):
                problems.append(
                    f"{case.id}: speaker measurement for {row_id!r} is absent"
                )
                continue
            raw = int(measurement.get("raw_in_speech_turn_changes") or 0)
            buckets = measurement.get("buckets")
            if (
                not isinstance(buckets, Mapping)
                or sum(map(int, buckets.values())) != raw
            ):
                problems.append(
                    f"{case.id}: speaker measurement for {row_id!r} does not conserve"
                )
            raw_counts.add(raw)
        if len(raw_counts) > 1:
            problems.append(
                f"{case.id}: speaker on/off rows do not share one raw denominator"
            )
        projection = speaker.get("projection")
        v2_row = finalizer_rows.get("v2") or {}
        cue_rows = v2_row.get("cues") or ()
        ranges = [
            tuple(row["unit_range"])
            for row in cue_rows
            if row.get("unit_range") is not None
        ]
        unit_count = int((artifact.get("coverage") or {}).get("unit_count") or 0)
        ranges_valid = (
            len(ranges) == len(cue_rows)
            and (bool(ranges) or unit_count == 0)
            and (not ranges or ranges[0][0] == 0)
            and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
            and (not ranges or ranges[-1][1] == unit_count)
        )
        if (
            not isinstance(projection, Mapping)
            or projection.get("status") != "verified"
            or int(projection.get("cue_count") or 0) != len(cue_rows)
            or int(projection.get("range_count") or 0) != len(ranges)
            or int(projection.get("named_multi_cues_unannotated") or 0)
            != int(
                (artifact.get("coverage") or {}).get("named_multi_cues_unannotated")
                or 0
            )
            or not ranges_valid
        ):
            problems.append(f"{case.id}: N19 speaker-id projection is not verified")
    cross = (artifact["lanes"][SHADOW_LANE_CORE]["v2"] or {}).get(
        "projection_cross_check"
    )
    if not isinstance(cross, Mapping):
        problems.append(
            f"{case.id}: the core lane carries no projection cross-check, so the"
            " unit ranges the gated lane is measured on are unverified"
        )
    elif not cross.get("agrees"):
        problems.append(
            f"{case.id}: the solver partition and structural surface projection"
            f" disagree (mode {cross.get('mode')!r})"
        )
    if artifact["coverage"].get("v1_unprojected"):
        problems.append(
            f"{case.id}: v1's cue stream could not be projected onto source units"
            f" ({artifact['v1_projection']['mode']}), so this case has no v1"
            " reference to compare against"
        )
    return problems


def c13_case_failures(
    case: Case, coverage: Mapping[str, Any], violations: Mapping[str, Any]
) -> list[str]:
    """N4/N5 (formerly C13): per-document shadow coverage failures."""
    problems: list[str] = []
    if int(coverage["fallback_intervals"]):
        problems.append(
            f"{case.id}: {coverage['fallback_intervals']} interval(s) fell back to"
            " adopted_v1 (C13 requires zero on the public corpus)"
        )
    ratio = float(coverage["optimized_unit_ratio"])
    if ratio < 1.0:
        problems.append(
            f"{case.id}: optimized_unit_ratio {ratio:.4f} < 1.0"
            " (C13 requires the whole document optimized)"
        )
    if int(coverage.get("coarse_caused_intervals") or 0):
        problems.append(
            f"{case.id}: {coverage['coarse_caused_intervals']} coarse-caused "
            "interval(s) remain after refinement"
        )
    for row in violations["exit_driving"]:
        problems.append(
            f"{case.id}: unwaived v2 {row['kind']} at stage {row['stage']}"
            f" (cue {row['cue_index']}): {row['detail']}"
        )
    return problems


def merge_violation_counts(
    shadow_cases: Sequence[ShadowCase],
) -> dict[str, dict[str, dict[str, int]]]:
    """Validator rows pooled per reporting group, keyed origin/stage/kind.

    The v1 streams go through the same validator at the same stages, so the two
    engines are directly comparable here. That comparison is the evidence a
    reader needs for the remaining P5 policy diagnostic: ja
    documents inherit a ``speech-truncated-start`` class from the shared shot
    snap, and whether v2 owning fewer of them than v1 counts as acceptable is a
    decision, not a measurement. This block states the counts and takes no view.
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    for row in shadow_cases:
        counts = shadow_violation_counts(row.artifact)
        for group in cc.group_keys(row.case.language):
            bucket = out.setdefault(group, {})
            for stage, block in counts["stages"].items():
                if not block:
                    continue
                target = bucket.setdefault(stage, {})
                for key, value in block["kinds"].items():
                    target[key] = target.get(key, 0) + int(value)
    return {
        group: {
            stage: dict(sorted(kinds.items())) for stage, kinds in sorted(x.items())
        }
        for group, x in sorted(out.items())
    }


def interval_changes(case: Case, artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-interval disagreement rows, worst first, for the report's triage list."""
    v1_cuts = sorted(int(u) for u in (artifact.get("v1") or {}).get("cut_units") or ())
    rows: list[dict[str, Any]] = []
    for interval in artifact["intervals"]:
        low, high = int(interval["unit_range"][0]), int(interval["unit_range"][1])
        inside_v1 = {cut for cut in v1_cuts if low < cut < high}
        inside_v2 = {int(cut) for cut in interval["v2_partition"]}
        moved = inside_v1 ^ inside_v2
        if not moved:
            continue
        selected = interval.get("policy_selected") or {}
        v1_cost = interval.get("v1_cost_under_v2") or {}
        rows.append(
            {
                "adopted_v1": bool(interval["adopted_v1"]),
                "atom_count": int(interval["atom_count"]),
                "case": case.id,
                "interval_index": int(interval["interval_index"]),
                "language": case.language,
                "margin_summary": interval.get("margin_summary"),
                "selected_is_v1": bool(interval["selected_is_v1"]),
                "unit_range": [low, high],
                "v1_cost_total": v1_cost.get("total"),
                "v1_cuts": sorted(inside_v1),
                "v1_path_legal": bool(interval["v1_path_legal"]),
                "v2_cost_total": selected.get("total"),
                "v2_cuts": sorted(inside_v2),
                "value": float(len(moved)),
            }
        )
    return rows


# ---------------------------------------------------------- the shadow run


@dataclass
class ShadowCase:
    """Everything one replayed case contributes to the shadow report."""

    case: Case
    artifact: dict[str, Any]
    document: Any
    lanes: dict[tuple[str, str], LaneResult]
    features: dict[str, Any]
    barrier_units: tuple[int, ...]
    production: CaseMeasurement
    tripwire_error: str | None
    wall_time_s: float = 0.0

    @property
    def core_partition(self) -> tuple[int, ...] | None:
        partition = self.artifact["lanes"][SHADOW_LANE_CORE]["v2"]["partition"]
        return None if partition is None else tuple(int(u) for u in partition)


def run_shadow_case(case: Case) -> ShadowCase:
    """Replay one case with the flag on and reduce it to lanes plus features."""
    started = time.perf_counter()
    result = replay_shadow(case)
    artifact = shadow_artifact_of(case, result)
    document = seg_document_of(case, result)
    lanes = {
        (lane, engine): measure_lane(case, artifact, lane, engine)
        for lane in SHADOW_LANES
        for engine in SHADOW_LANE_ROWS[lane]
        if engine in (artifact["lanes"][lane].get("rows") or artifact["lanes"][lane])
    }
    partition = artifact["lanes"][SHADOW_LANE_CORE]["v2"]["partition"] or ()
    production = measure_case(case, result)
    legacy = lanes[(SHADOW_LANE_DELIVERY, "v1")].measurement
    tripwire_error: str | None = None
    if legacy is None:
        tripwire_error = "delivery_v1_legacy/v1 is not measurable"
    elif (
        legacy.cue_count != production.cue_count
        or legacy.cps_samples != production.cps_samples
        or {key: (value.bad, value.eligible) for key, value in legacy.ratios.items()}
        != {
            key: (value.bad, value.eligible) for key, value in production.ratios.items()
        }
    ):
        tripwire_error = "delivery_v1_legacy/v1 drifted from production replay"
    shadow_case = ShadowCase(
        case=case,
        artifact=artifact,
        document=document,
        lanes=lanes,
        features=cut_feature_scan(document, [int(u) for u in partition]),
        barrier_units=barrier_unit_ids(document),
        production=production,
        tripwire_error=tripwire_error,
    )
    shadow_case.wall_time_s = time.perf_counter() - started
    return shadow_case


def lane_groups(
    shadow_cases: Sequence[ShadowCase], lane: str, engine: str
) -> dict[str, dict[str, Any]]:
    """Micro-aggregate one (lane, engine) pair across every measured case."""
    return aggregate(
        [
            row.lanes[(lane, engine)].measurement
            for row in shadow_cases
            if (lane, engine) in row.lanes
            and row.lanes[(lane, engine)].measurement is not None
        ]
    )


def shadow_group_errors(groups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Hold shadow group blocks to the same schema the baseline's groups obey."""
    definition = group_schema()
    return [
        f"{name}/{error}"
        for name, block in sorted(groups.items())
        for error in cc.schema_errors(block, definition)
    ]


def finalizer_vs_legacy_gates(
    shadow_cases: Sequence[ShadowCase],
) -> list[dict[str, Any]]:
    """N3a: identical committed v1 input, finalizer versus legacy display."""
    current = lane_groups(shadow_cases, SHADOW_LANE_FINALIZER, "v1")
    reference = lane_groups(shadow_cases, SHADOW_LANE_LEGACY_DISPLAY, "v1")
    results = evaluate_gates(current, SHADOW_GATES, {"groups": reference})
    for result in results:
        result["family"] = "N3a-finalizer-vs-legacy-display"
    return results


def speaker_gate_block(shadow_cases: Sequence[ShadowCase]) -> dict[str, Any]:
    """N3b's four ja gates plus non-blocking zh/en diagnostics."""

    def aggregate_language(language: str) -> dict[str, Any]:
        raw = expressed = missed = off_expressed = attributable = 0
        eligible_ceiling = 0
        activation_cases = 0
        case_count = 0
        for row in shadow_cases:
            if (
                cc.canonical_language_or(row.case.language, row.case.language)
                != language
            ):
                continue
            evidence = row.artifact.get("speaker_evidence") or {}
            on = evidence.get("measurement")
            off = evidence.get("off_row_measurement")
            if not isinstance(on, Mapping) or not isinstance(off, Mapping):
                continue
            case_count += 1
            on_buckets = on["buckets"]
            off_buckets = off["buckets"]
            raw += int(on["raw_in_speech_turn_changes"])
            expressed += int(on_buckets["expressed"])
            missed += int(on_buckets["survived_expressible_but_missed"])
            off_expressed += int(off_buckets["expressed"])
            attributable += int(on["speaker_attributable_expressed_cuts"])
            eligible_ceiling += int(on_buckets["expressed"]) + int(
                on_buckets["survived_expressible_but_missed"]
            )
            if int(on["speaker_attributable_expressed_cuts"]) > 0 or int(
                on_buckets["expressed"]
            ) > int(off_buckets["expressed"]):
                activation_cases += 1
        rate = 0.0 if raw == 0 else expressed / raw
        off_rate = 0.0 if raw == 0 else off_expressed / raw
        hit = None if expressed + missed == 0 else expressed / (expressed + missed)
        return {
            "activation_cases": activation_cases,
            "case_count": case_count,
            "eligible_ceiling": eligible_ceiling,
            "expressed": expressed,
            "expressed_rate": rate,
            "expressible_hit_rate": hit,
            "missed": missed,
            "off_expressed": off_expressed,
            "off_expressed_rate": off_rate,
            "raw_in_speech_turn_changes": raw,
            "speaker_attributable_expressed_cuts": attributable,
        }

    diagnostics = {
        language: aggregate_language(language) for language in cc.CALIBRATION_LANGUAGES
    }
    ja = diagnostics["ja"]
    target = 21.0 / 136.0
    possible_rate = (
        0.0
        if ja["raw_in_speech_turn_changes"] == 0
        else ja["eligible_ceiling"] / ja["raw_in_speech_turn_changes"]
    )
    comparison_status = (
        "pass" if ja["expressed_rate"] >= ja["off_expressed_rate"] else "fail"
    )
    absolute_status = (
        "pass"
        if ja["expressed_rate"] >= target
        else "stopped"
        if possible_rate < target
        else "fail"
    )
    # The frozen absolute ceiling can only adjudicate the absolute clause.  It
    # has no authority to excuse a regression below the independently measured
    # speaker-off row.
    rate_status = (
        "fail"
        if comparison_status == "fail"
        else "pass"
        if absolute_status == "pass"
        else absolute_status
    )
    gates = [
        {
            "id": "N3b-activation",
            "status": "pass" if ja["activation_cases"] > 0 else "fail",
            "value": ja["activation_cases"],
        },
        {
            "absolute_status": absolute_status,
            "comparison_status": comparison_status,
            "id": "N3b-expressed-rate",
            "possible_rate": possible_rate,
            "status": rate_status,
            "target": max(target, ja["off_expressed_rate"]),
            "value": ja["expressed_rate"],
        },
        {
            "id": "N3b-attributable",
            "status": (
                "pass" if ja["speaker_attributable_expressed_cuts"] >= 1 else "fail"
            ),
            "value": ja["speaker_attributable_expressed_cuts"],
        },
        {
            "id": "N3b-expressible-hit",
            "status": (
                "pass"
                if ja["expressible_hit_rate"] is not None
                and ja["expressible_hit_rate"] >= 0.5
                else "fail"
            ),
            "target": 0.5,
            "value": ja["expressible_hit_rate"],
        },
    ]
    return {"diagnostics": diagnostics, "gates": gates}


_SPEECH_TRUNCATION_KINDS = frozenset({"speech-truncated-start", "speech-truncated-end"})


def _speech_truncation_count(block: Mapping[str, Any] | None) -> int:
    if not isinstance(block, Mapping):
        raise cc.CalibrationError(
            "N6 cannot measure a missing validator block",
            ["speech-truncation absence is not zero evidence"],
        )
    return sum(
        not bool(row.get("waived")) and str(row.get("kind")) in _SPEECH_TRUNCATION_KINDS
        for row in block.get("violations") or ()
    )


def speech_truncation_gates(
    shadow_cases: Sequence[ShadowCase],
) -> dict[str, Any]:
    """N6: finalizer/v2 may not exceed shipped legacy truncation by language."""
    languages: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for language in cc.CALIBRATION_LANGUAGES:
        current = legacy = 0
        for row in shadow_cases:
            if (
                cc.canonical_language_or(row.case.language, row.case.language)
                != language
            ):
                continue
            lanes = row.artifact["lanes"]
            current += _speech_truncation_count(
                lanes[SHADOW_LANE_FINALIZER]["rows"]["v2"].get("validator")
            )
            legacy += _speech_truncation_count(
                lanes[SHADOW_LANE_DELIVERY]["v1"].get("validator")
            )
        passed = current <= legacy
        languages[language] = {
            "legacy": legacy,
            "status": "pass" if passed else "fail",
            "v2": current,
        }
        if not passed:
            failures.append(
                f"N6/{language}: finalizer/v2 speech truncations {current} "
                f"exceed legacy {legacy}"
            )
    return {"failures": failures, "languages": languages}


def speaker_cliff_diagnostics() -> dict[str, Any]:
    """P3's six deterministic threshold probes plus per-turn-state coverage."""
    from voxweave.core import speaker_evidence as se
    from voxweave.core.segdoc import DisplayProfile, SegDocument, SourceUnit

    def profile(language: str = "en") -> DisplayProfile:
        return DisplayProfile(
            language=language,
            max_line_length=42 if language == "en" else 18,
            max_lines=2,
            clause_ms=400.0,
            vad_skip_ms=250.0,
            offline_ms=700.0,
            min_cue_s=0.0,
            max_cue_s=7.0,
            glue_gap_s=0.3,
            cps=0.0,
            lag_out_s=0.0,
            shot_snap_s=11 / 24,
        )

    def document(
        spans: Sequence[tuple[str, float, float]],
        turns: Sequence[tuple[float, float, str]] | None,
        *,
        language: str = "en",
    ) -> SegDocument:
        units = [
            SourceUnit(f"u{index}", surface, start, end)
            for index, (surface, start, end) in enumerate(spans)
        ]
        return SegDocument(
            language=language,
            units=units,
            profile=profile(language),
            vad_speech=None,
            shot_changes=None,
            sing_spans=None,
            speaker_turns=None if turns is None else list(turns),
            manifest={},
            text=(" " if language == "en" else "").join(
                surface for surface, _start, _end in spans
            ),
        )

    def signature(evidence: Any) -> dict[str, Any]:
        return {
            "labels": list(evidence.labels),
            "phrase_snaps": evidence.stats.phrase_snaps,
            "runs_absorbed": evidence.stats.runs_absorbed,
            "transitions_after": evidence.stats.transitions_after,
        }

    def turn_state(evidence: Any) -> str:
        starts = [
            unit.start for unit in evidence.parent_units if unit.start is not None
        ]
        ends = [unit.end for unit in evidence.parent_units if unit.end is not None]
        span = se.EvidenceSpan(min(starts), max(ends), "exact", "exact")
        return str(
            se.speaker_edge_cost(
                evidence, (0, len(evidence.unit_speakers)), evidence_span=span
            ).features["turn_state"]
        )

    probes: list[dict[str, Any]] = []

    def compare(name: str, before_doc: SegDocument, after_doc: SegDocument) -> None:
        before = se.speaker_evidence(before_doc)
        after = se.speaker_evidence(after_doc)
        before_signature, after_signature = signature(before), signature(after)
        probes.append(
            {
                "after": after_signature,
                "before": before_signature,
                "effective": before_signature != after_signature,
                "name": name,
                "turn_states": sorted({turn_state(before), turn_state(after)}),
            }
        )

    compare(
        "cover-frac",
        document([("x", 0.0, 1.0)], [(0.0, 0.499, "A")]),
        document([("x", 0.0, 1.0)], [(0.0, 0.5, "A")]),
    )
    compare(
        "MIN_RUN",
        document(
            [("a", 0.0, 0.4), ("b", 0.4, 0.599), ("a", 0.599, 1.0)],
            [(0.0, 0.4, "A"), (0.4, 0.599, "B"), (0.599, 1.0, "A")],
        ),
        document(
            [("a", 0.0, 0.4), ("b", 0.4, 0.601), ("a", 0.601, 1.0)],
            [(0.0, 0.4, "A"), (0.4, 0.601, "B"), (0.601, 1.0, "A")],
        ),
    )
    compare(
        "EDGE_RUN",
        document(
            [("a", 0.0, 0.5), ("b", 0.5, 0.619)],
            [(0.0, 0.5, "A"), (0.5, 0.619, "B")],
        ),
        document(
            [("a", 0.0, 0.5), ("b", 0.5, 0.62)],
            [(0.0, 0.5, "A"), (0.5, 0.62, "B")],
        ),
    )

    saved_phrase_ranges = se._phrase_ranges
    se._phrase_ranges = lambda _units, _lang: ((0, 2),)
    try:
        compare(
            "phrase-vote",
            document(
                [("大", 0.0, 0.3), ("碴子", 0.3, 0.59)],
                [(0.0, 0.3, "A"), (0.3, 0.59, "B")],
                language="zh",
            ),
            document(
                [("大", 0.0, 0.3), ("碴子", 0.3, 0.61)],
                [(0.0, 0.3, "A"), (0.3, 0.61, "B")],
                language="zh",
            ),
        )
    finally:
        se._phrase_ranges = saved_phrase_ranges

    compare(
        "region-silence",
        document(
            [("a", 0.0, 1.0), ("hole", 1.299, 2.0)],
            [(0.0, 1.0, "A")],
        ),
        document(
            [("a", 0.0, 1.0), ("hole", 1.3, 2.0)],
            [(0.0, 1.0, "A")],
        ),
    )

    crossing = se.speaker_evidence(
        document(
            [("a", 0.0, 1.0), ("b", 1.0, 2.0)],
            [(0.0, 1.0, "A"), (1.0, 2.0, "B")],
        )
    )
    span = (se.EvidenceSpan(0.0, 2.0, "exact", "exact"),)
    before_measure = se.measure_speaker_events(
        crossing, evidence_spans=span, delivered_boundaries=(1.5,)
    )
    after_measure = se.measure_speaker_events(
        crossing, evidence_spans=span, delivered_boundaries=(1.500001,)
    )
    probes.append(
        {
            "after": after_measure.to_dict(),
            "before": before_measure.to_dict(),
            "effective": before_measure.buckets != after_measure.buckets,
            "name": "transition-crossing",
            "turn_states": [turn_state(crossing)],
        }
    )

    state_examples = {
        "absent": se.speaker_evidence(document([("x", 0.0, 1.0)], None)),
        "overlap": se.speaker_evidence(
            document(
                [("x", 0.0, 1.0)],
                [(0.0, 0.8, "A"), (0.2, 1.0, "B")],
            )
        ),
        "unattributed": se.speaker_evidence(document([("x", 0.0, 1.0)], [])),
    }
    attempted = {state: 0 for state in se.TURN_STATES}
    effective = {state: 0 for state in se.TURN_STATES}
    for probe in probes:
        for state in probe["turn_states"]:
            attempted[state] += 1
            effective[state] += int(probe["effective"])
    for expected, evidence in state_examples.items():
        observed = turn_state(evidence)
        if observed != expected:
            raise cc.CalibrationError(
                f"P3 state fixture expected {expected!r}, observed {observed!r}"
            )
        attempted[observed] += 1
    warnings = [
        f"P3/{state}: zero effective probes (warning-uncovered)"
        for state in se.TURN_STATES
        if effective[state] == 0
    ]
    failures = [
        f"P3/{probe['name']}: threshold probe was ineffective"
        for probe in probes
        if not probe["effective"]
    ]
    return {
        "attempted_by_turn_state": attempted,
        "effective_by_turn_state": effective,
        "failures": failures,
        "probes": probes,
        "warnings": warnings,
    }


_COARSE_BREAKERS = frozenset("。！？、，,.!?；;：:")


def load_coarse_manifest(path: str | Path = DEFAULT_COARSE_CORPUS) -> dict[str, Any]:
    """Load W2's shadow-only derivation registry under a closed local schema."""
    source = Path(path)
    try:
        payload = cc.read_json(source)
    except OSError as exc:
        raise cc.CalibrationError(f"coarse corpus not found: {source}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "description",
        "cases",
    }:
        raise cc.CalibrationError(f"{source}: invalid coarse registry shape")
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("description"), str
    ):
        raise cc.CalibrationError(f"{source}: invalid coarse registry header")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise cc.CalibrationError(f"{source}: coarse registry has no cases")
    expected_variants = {"width", "duration", "both", "per-char", "mixed"}
    if len(rows) != len(expected_variants):
        raise cc.CalibrationError(
            f"{source}: coarse registry must contain exactly five variants"
        )
    required = {
        "id",
        "variant",
        "source_case",
        "max_block_units",
        "profile_overrides",
    }
    variants: set[str] = set()
    ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise cc.CalibrationError(
                f"{source}: coarse cases[{index}] has an invalid shape"
            )
        case_id = row["id"]
        block = row["max_block_units"]
        variant = row["variant"]
        source_case = row["source_case"]
        if (
            not isinstance(case_id, str)
            or not case_id.startswith("coarse-")
            or case_id in ids
            or not isinstance(variant, str)
            or variant not in expected_variants
            or variant in variants
            or not isinstance(source_case, str)
            or not source_case.startswith("cases/")
            or isinstance(block, bool)
            or not isinstance(block, int)
            or block <= 0
            or not isinstance(row["profile_overrides"], dict)
        ):
            raise cc.CalibrationError(
                f"{source}: coarse cases[{index}] has invalid values"
            )
        ids.add(case_id)
        variants.add(variant)
    if variants != expected_variants:
        raise cc.CalibrationError(f"{source}: coarse variants are incomplete")
    return payload


def _derived_coarse_case(source: Case, fixture: Mapping[str, Any]) -> Case:
    """Apply the manifest's deterministic sentence/block merge derivation."""
    from voxweave.core.layout import _join

    blocks: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    limit = int(fixture["max_block_units"])
    for source_unit in source.units:
        pending.append(source_unit)
        surface = str(source_unit["text"])
        if (surface and surface[-1] in _COARSE_BREAKERS) or len(pending) >= limit:
            blocks.append(pending)
            pending = []
    if pending:
        blocks.append(pending)

    overrides = dict(fixture["profile_overrides"])
    language = str(overrides.pop("language", source.language))
    doc = copy.deepcopy(source.doc)
    doc["id"] = str(fixture["id"])
    doc["language"] = language
    doc["word_segments"] = [
        {
            "id": f"u{index}",
            "text": _join([str(unit["text"]) for unit in block], language),
            "start": block[0]["start"],
            "end": block[-1]["end"],
        }
        for index, block in enumerate(blocks)
    ]
    config = doc["capture"]["config"]
    gaps = config["gap_thresholds"]
    for key, value in overrides.items():
        if key in gaps:
            gaps[key] = value
        elif key in config:
            config[key] = value
        else:
            raise cc.CalibrationError(
                f"{fixture['id']}: unknown profile override {key!r}"
            )
    return Case(source.path, str(fixture["source_case"]), doc, source.size_bytes)


def _coarse_start_errors(fine: Case, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Boundary start error against the frozen fine stream, by display cursor."""
    from voxweave.core.partition_check import normalize_text
    from voxweave.core.smart_split import _display_chars

    truth = [
        float(unit["start"])
        for unit, chars in zip(
            fine.units,
            _display_chars([str(unit["text"]) for unit in fine.units]),
        )
        for _character in chars
    ]
    offset = 0
    errors: list[float] = []
    for row in rows:
        if 0 < offset < len(truth):
            errors.append(abs(float(row["start"]) - truth[offset]))
        offset += len("".join(normalize_text(str(row["text"])).split()))
    return errors


def _n14_profile(value: Any) -> Any:
    """Decode the closed artifact profile for an independent N14 replay."""
    from voxweave.core.segdoc import DisplayProfile

    if not isinstance(value, Mapping):
        raise ValueError("profile is missing")
    keys = {
        "clause_ms",
        "cps",
        "glue_gap_s",
        "lag_out_s",
        "language",
        "max_cue_s",
        "max_line_length",
        "max_lines",
        "min_cue_s",
        "offline_ms",
        "shot_snap_s",
        "vad_skip_ms",
    }
    if set(value) != keys:
        raise ValueError("profile is not closed")
    return DisplayProfile(**{key: value[key] for key in keys})


def _n14_units(value: Any) -> list[Any]:
    """Decode only serialized source-unit facts; never read producer objects."""
    from voxweave.core.segdoc import SourceUnit

    if not isinstance(value, list):
        raise ValueError("source units are missing")
    units = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"unit {index} is not a mapping")
        if set(raw) != {"confidence", "end", "id", "provenance", "start", "surface"}:
            raise ValueError(f"unit {index} is not closed")
        units.append(
            SourceUnit(
                id=str(raw["id"]),
                surface=str(raw["surface"]),
                start=raw["start"],
                end=raw["end"],
                provenance=str(raw["provenance"]),
                confidence=raw["confidence"],
            )
        )
    return units


def _n14_work_replay(
    units: Sequence[Any], profile: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    """Instrument fresh canonical work from serialized facts and the live profile."""
    from voxweave.core.boundary_lattice import (
        COARSE_GRANULARITY,
        build_document_lattice,
    )
    from voxweave.core.layout import _join
    from voxweave.core.segdoc import SegDocument

    document = SegDocument(
        language=profile.language,
        units=list(units),
        profile=profile,
        vad_speech=None,
        shot_changes=None,
        sing_spans=None,
        speaker_turns=None,
        manifest={},
        text=_join([unit.surface for unit in units], profile.language),
    )
    built = build_document_lattice(
        document,
        cache_speaker_evidence=False,
        canonical_spaced=True,
    )
    measurements: list[dict[str, Any]] = []
    errors: list[str] = []
    cursor = 0
    for index, lattice in enumerate(built.lattices):
        low = lattice.interval.unit_start
        high = lattice.interval.unit_end
        if low != cursor or not low < high <= len(units):
            errors.append(
                f"interval {index} replay range {[low, high]!r} does not tile units"
            )
        cursor = high
        reason = None if lattice.infeasible is None else lattice.infeasible.reason
        scanned = (
            bool(lattice.atoms)
            and not lattice.all_invisible
            and reason != COARSE_GRANULARITY
        )
        measurements.append(
            {
                "canonical_chars": lattice.canonical_chars,
                "scanned": scanned,
                "unit_range": [low, high],
            }
        )
    if cursor != len(units):
        errors.append(f"replay ranges stop at {cursor}/{len(units)} units")
    return measurements, errors


def _n14_oracle(
    units: Sequence[Any],
    profile: Any,
    interval_ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Rebuild text-only admission and derive the legal edge set independently."""
    from dataclasses import replace

    from voxweave.core.boundary_lattice import band_atoms, build_document_lattice
    from voxweave.core.canonical_text import canonical_legal, canonical_text
    from voxweave.core.layout import _join
    from voxweave.core.segdoc import SegDocument

    text_profile = replace(profile, max_cue_s=0.0)
    checked = false_negative = false_positive = 0
    mismatch_examples: list[dict[str, Any]] = []
    unknown: list[str] = []
    limit = band_atoms(text_profile) + 2
    for interval_index, (low, high) in enumerate(interval_ranges):
        owned_units = list(units[low:high])
        document = SegDocument(
            language=text_profile.language,
            units=owned_units,
            profile=text_profile,
            vad_speech=None,
            shot_changes=None,
            sing_spans=None,
            speaker_turns=None,
            manifest={},
            text=_join([unit.surface for unit in owned_units], text_profile.language),
        )
        built = build_document_lattice(
            document,
            cache_speaker_evidence=False,
            canonical_spaced=True,
        )
        if len(built.lattices) != 1:
            unknown.append(
                f"interval {interval_index}: text-only replay produced "
                f"{len(built.lattices)} intervals"
            )
            continue
        lattice = built.lattices[0]
        actual = {(edge.start_node, edge.end_node) for edge in lattice.edges}
        expected: set[tuple[int, int]] = set()
        for position, start in enumerate(lattice.nodes):
            for end in lattice.nodes[position + 1 :]:
                if end - start > limit:
                    break
                atoms = lattice.atoms[start:end]
                raw = _join([atom.text for atom in atoms], text_profile.language)
                final = canonical_text(
                    [
                        {"text": atom.text, "start": atom.start, "end": atom.end}
                        for atom in atoms
                    ],
                    fallback_text=raw,
                    lang=text_profile.language,
                    profile=text_profile,
                    expected_footprint=raw,
                )
                checked += 1
                if (
                    any(atom.start is not None for atom in atoms)
                    and any(atom.end is not None for atom in atoms)
                    and canonical_legal(final, text_profile)
                ):
                    expected.add((start, end))
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        false_negative += len(missing)
        false_positive += len(extra)
        if missing or extra:
            mismatch_examples.append(
                {
                    "extra": [list(pair) for pair in extra[:5]],
                    "interval": interval_index,
                    "missing": [list(pair) for pair in missing[:5]],
                }
            )
    return {
        "checked": checked,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "mismatch_examples": mismatch_examples,
        "unknown": unknown,
    }


def n14_artifact_evidence(
    artifact: Mapping[str, Any], *, case_id: str, corpus: str
) -> dict[str, Any]:
    """Derive N14 work, FD-9 and both-direction facts from serialized authority."""
    from voxweave.core.boundary_lattice import band_atoms
    from voxweave.core.canonical_text import CANONICAL_PASS_FACTOR
    from voxweave.core.layout import _no_spaces

    failures: list[str] = []
    unknown: list[str] = []
    work_rows: list[dict[str, Any]] = []
    finalizer_rows: list[dict[str, Any]] = []
    interval_ranges: list[tuple[int, int]] = []
    try:
        profile = _n14_profile(artifact.get("profile"))
        units = _n14_units(artifact.get("units"))
    except (TypeError, ValueError) as exc:
        return {
            "case": case_id,
            "corpus": corpus,
            "failures": [],
            "language_class": "unknown",
            "oracle": {"checked": 0},
            "unknown": [f"{case_id}: {exc}"],
        }

    replayed_work, replay_errors = _n14_work_replay(units, profile)
    unknown.extend(
        f"{case_id}: independent work replay: {detail}" for detail in replay_errors
    )

    intervals = artifact.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        unknown.append(f"{case_id}: interval work evidence is missing")
    else:
        if len(intervals) != len(replayed_work):
            unknown.append(
                f"{case_id}: independent work measurement has "
                f"{len(replayed_work)} intervals for {len(intervals)} emitted rows"
            )
        cursor = 0
        for index, raw in enumerate(intervals):
            if not isinstance(raw, Mapping):
                unknown.append(f"{case_id}: interval {index} is not a mapping")
                continue
            unit_range = raw.get("unit_range")
            actual = raw.get("canonical_chars")
            if (
                not isinstance(unit_range, list)
                or len(unit_range) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in unit_range
                )
                or isinstance(actual, bool)
                or not isinstance(actual, int)
                or actual < 0
            ):
                unknown.append(f"{case_id}: interval {index} work row is malformed")
                continue
            low, high = unit_range
            if low != cursor or not low < high <= len(units):
                unknown.append(
                    f"{case_id}: interval {index} range {unit_range!r} does not tile units"
                )
                continue
            cursor = high
            interval_ranges.append((low, high))
            raw_chars = sum(len(unit.surface) for unit in units[low:high])
            bound = CANONICAL_PASS_FACTOR * raw_chars * (band_atoms(profile) + 2) ** 2
            replay = replayed_work[index] if index < len(replayed_work) else None
            if replay is None:
                unknown.append(
                    f"{case_id}: interval {index} independent work measurement is absent"
                )
                continue
            verified = replay["canonical_chars"]
            range_agrees = replay["unit_range"] == unit_range
            if not range_agrees:
                unknown.append(
                    f"{case_id}: interval {index} work measurement range "
                    f"{replay['unit_range']!r} disagrees with emitted {unit_range!r}"
                )
            positive = not replay["scanned"] or verified > 0
            if not positive:
                unknown.append(
                    f"{case_id}: interval {index} independent work measurement is "
                    "zero for a nonempty canonical scan"
                )
            measurement_agrees = actual == verified
            if not measurement_agrees:
                unknown.append(
                    f"{case_id}: interval {index} work measurement disagrees: "
                    f"emitted {actual}, independently replayed {verified}"
                )
            within_bound = verified <= bound
            passed = range_agrees and positive and measurement_agrees and within_bound
            if range_agrees and positive and measurement_agrees and not within_bound:
                failures.append(
                    f"{case_id}: interval {index} canonical_chars {verified} exceeds "
                    f"independent bound {bound}"
                )
            work_rows.append(
                {
                    "bound": bound,
                    "canonical_chars": verified,
                    "emitted_canonical_chars": actual,
                    "interval": index,
                    "measurement_agrees": measurement_agrees,
                    "passed": passed,
                    "raw_chars": raw_chars,
                    "scanned": replay["scanned"],
                    "unit_range": [low, high],
                }
            )
        if cursor != len(units):
            unknown.append(
                f"{case_id}: interval work ranges stop at {cursor}/{len(units)} units"
            )

    totals = artifact.get("totals")
    reported_total = (
        totals.get("canonical_chars") if isinstance(totals, Mapping) else None
    )
    recomputed_total = sum(row["canonical_chars"] for row in replayed_work)
    if isinstance(reported_total, bool) or not isinstance(reported_total, int):
        unknown.append(f"{case_id}: totals.canonical_chars is missing or malformed")
    elif reported_total != recomputed_total:
        unknown.append(
            f"{case_id}: totals.canonical_chars {reported_total} disagrees with "
            f"independently replayed interval sum {recomputed_total}"
        )

    lanes = artifact.get("lanes")
    delivery = lanes.get(SHADOW_LANE_FINALIZER) if isinstance(lanes, Mapping) else None
    rows = delivery.get("rows") if isinstance(delivery, Mapping) else None
    required_rows = {"v1", "v2", "v2-speaker-off"}
    comparison = artifact.get("refiner_comparison")
    if isinstance(comparison, Mapping) and comparison.get("materialized") is True:
        required_rows.add("refiner-off")
    if not isinstance(rows, Mapping):
        unknown.append(f"{case_id}: authoritative finalizer rows are missing")
    else:
        required_rows.update(
            str(row_id)
            for row_id, row in rows.items()
            if isinstance(row, Mapping) and isinstance(row.get("finalizer"), Mapping)
        )
        for row_id in sorted(required_rows):
            row = rows.get(row_id)
            finalizer = row.get("finalizer") if isinstance(row, Mapping) else None
            if not isinstance(finalizer, Mapping):
                unknown.append(f"{case_id}: finalizer row {row_id} has no evidence")
                continue
            entries = finalizer.get("entries")
            deltas = finalizer.get("deltas_fired")
            if not isinstance(entries, list) or not isinstance(deltas, list):
                unknown.append(
                    f"{case_id}: finalizer row {row_id} evidence is malformed"
                )
                continue
            organic = sum(
                isinstance(entry, Mapping)
                and entry.get("kind") == "stutter-not-proven-fixed-within-4-scans"
                for entry in entries
            )
            delta_count = sum(delta == "FD-9" for delta in deltas)
            if organic or delta_count:
                failures.append(
                    f"{case_id}: finalizer row {row_id} fired organic FD-9 "
                    f"({organic} report(s), {delta_count} delta(s))"
                )
            finalizer_rows.append(
                {
                    "fd9_deltas": delta_count,
                    "fd9_reports": organic,
                    "row": row_id,
                }
            )

    oracle = (
        _n14_oracle(units, profile, interval_ranges)
        if interval_ranges
        else {
            "checked": 0,
            "false_negative": 0,
            "false_positive": 0,
            "mismatch_examples": [],
            "unknown": ["no valid interval ranges"],
        }
    )
    unknown.extend(
        f"{case_id}: oracle: {detail}" for detail in oracle.get("unknown", ())
    )
    if int(oracle.get("checked") or 0) == 0:
        unknown.append(f"{case_id}: both-direction oracle denominator is zero")
    if int(oracle.get("false_negative") or 0) or int(oracle.get("false_positive") or 0):
        failures.append(
            f"{case_id}: both-direction oracle found "
            f"{oracle['false_negative']} missing and {oracle['false_positive']} extra edge(s)"
        )
    return {
        "case": case_id,
        "corpus": corpus,
        "failures": failures,
        "finalizer_rows": finalizer_rows,
        "language": profile.language,
        "language_class": "no-space" if _no_spaces(profile.language) else "spaced",
        "oracle": oracle,
        "reported_total": reported_total,
        "recomputed_total": recomputed_total,
        "unknown": unknown,
        "work": work_rows,
    }


def n14_matrix_unknown(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require positive oracle denominators in both corpora and language classes."""
    required = {
        (corpus, language_class)
        for corpus in ("tracked", "coarse")
        for language_class in ("spaced", "no-space")
    }
    observed = {
        (str(row.get("corpus")), str(row.get("language_class")))
        for row in evidence
        if int((row.get("oracle") or {}).get("checked") or 0) > 0
    }
    return [
        f"N14 missing positive denominator for {corpus}/{language_class}"
        for corpus, language_class in sorted(required - observed)
    ]


def n14_exit_code(
    evidence: Sequence[Mapping[str, Any]], *, require_matrix: bool = False
) -> int:
    """Fold N14 evidence: missing is invalid, a disproved property is a failure."""
    if not evidence:
        return cc.EXIT_INVALID
    unknown = [item for row in evidence for item in row.get("unknown") or ()]
    if require_matrix:
        unknown.extend(n14_matrix_unknown(evidence))
    if unknown:
        return cc.EXIT_INVALID
    if any(row.get("failures") for row in evidence):
        return cc.EXIT_GATE_FAILED
    return cc.EXIT_OK


def run_coarse_gates(
    tracked: Corpus,
    path: str | Path = DEFAULT_COARSE_CORPUS,
) -> dict[str, Any]:
    """N4c/PD-SUBUNIT over W2's isolated, derived coarse family."""
    manifest = load_coarse_manifest(path)
    by_relpath = {case.relpath: case for case in tracked.cases}
    rows: list[dict[str, Any]] = []
    n14_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    stops: list[dict[str, str]] = []
    for fixture in manifest["cases"]:
        source = by_relpath.get(str(fixture["source_case"]))
        if source is None:
            raise cc.CalibrationError(
                f"{fixture['id']}: source case is not in the tracked registry"
            )
        derived = _derived_coarse_case(source, fixture)
        result = replay_shadow(derived)
        raw_artifact = getattr(result, "shadow", None)
        raw_error = (
            raw_artifact.get("error") if isinstance(raw_artifact, Mapping) else None
        )
        diagnostic = (
            raw_artifact.get("diagnostic")
            if isinstance(raw_artifact, Mapping)
            else None
        )
        unavailable_comparison = (
            diagnostic.get("refiner_comparison")
            if isinstance(diagnostic, Mapping)
            else None
        )
        frozen_projection_gap = (
            isinstance(raw_error, Mapping)
            and raw_error.get("detail") == "v1 source partition could not be projected"
            and isinstance(unavailable_comparison, Mapping)
            and unavailable_comparison.get("status") == "refined-counterfactual"
            and isinstance(diagnostic, Mapping)
            and (diagnostic.get("v1_projection") or {}).get("unprojected") is True
        )
        frozen_optimizer_gap = (
            isinstance(raw_error, Mapping)
            and raw_error.get("detail")
            == "optimizer selection authority unavailable for one or more rows"
            and isinstance(unavailable_comparison, Mapping)
            and unavailable_comparison.get("status") == "unmaterialized"
            and unavailable_comparison.get("reason")
            == "optimizer-selection-unavailable"
        )
        incomplete = (
            isinstance(raw_artifact, Mapping)
            and raw_artifact.get("kind") == "segmentation-shadow-incomplete"
            and raw_artifact.get("schema_version") == 1
            and isinstance(raw_error, Mapping)
            and raw_error.get("type") == "IncompleteShadowArtifact"
            and isinstance(diagnostic, Mapping)
            and (frozen_projection_gap or frozen_optimizer_gap)
        )
        artifact = (
            dict(diagnostic) if incomplete else shadow_artifact_of(derived, result)
        )
        final_row = artifact["lanes"][SHADOW_LANE_FINALIZER]["rows"].get("v2") or {}
        errors = _coarse_start_errors(source, final_row.get("cues") or ())
        ordered = sorted(errors)
        p90 = (
            None if not ordered else ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
        )
        maximum = max(ordered, default=None)
        adopted = sum(bool(item["adopted_v1"]) for item in artifact["intervals"])
        raw_validator = artifact["validator"].get("raw") or {}
        finalizer_validator = artifact["validator"].get("finalizer") or {}
        trigger_counts = {"duration": 0, "width": 0}
        from voxweave.core.boundary_lattice import CAP_EPS_S
        from voxweave.core.layout import _line_budget_width, _vis_width

        parent_document = seg_document_of(derived, result)
        n14 = n14_artifact_evidence(
            artifact, case_id=str(fixture["id"]), corpus="coarse"
        )
        n14_rows.append(n14)
        failures.extend(n14["failures"])
        width_budget = _line_budget_width(
            parent_document.profile.max_line_length,
            parent_document.profile.language,
        )
        for unit in parent_document.units:
            if _vis_width(unit.surface) > width_budget:
                trigger_counts["width"] += 1
            if (
                parent_document.profile.max_cue_s > 0
                and unit.start is not None
                and unit.end is not None
                and unit.end - unit.start
                > parent_document.profile.max_cue_s + CAP_EPS_S
            ):
                trigger_counts["duration"] += 1
        evidence = artifact["subunit_split"]["evidence"]
        variant = str(fixture["variant"])
        variant_exercised = {
            "width": trigger_counts["width"] > 0 and trigger_counts["duration"] == 0,
            "duration": trigger_counts["duration"] > 0 and trigger_counts["width"] == 0,
            "both": trigger_counts["width"] > 0 and trigger_counts["duration"] > 0,
            "per-char": int(evidence["per-char"]) > 0,
            "mixed": int(evidence["whitespace"]) > 0,
        }[variant]
        checks = {
            "adopted_v1": adopted == 0,
            "coarse_caused_intervals": int(
                artifact["coverage"]["coarse_caused_intervals"]
            )
            == 0,
            "fallback_intervals": int(artifact["coverage"]["fallback_intervals"]) == 0,
            "max_start_error": maximum is not None and maximum <= 2.0,
            "optimized_unit_ratio": float(artifact["coverage"]["optimized_unit_ratio"])
            == 1.0,
            "p90_start_error": p90 is not None and p90 <= 0.75,
            "preview_fidelity": _preview_fidelity_valid(
                artifact.get("preview_fidelity")
            ),
            "variant_exercised": variant_exercised,
            "zero_unwaived_v2_raw_finalizer": (
                int(raw_validator.get("unwaived") or 0) == 0
                and int(finalizer_validator.get("unwaived") or 0) == 0
            ),
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{fixture['id']}: {name} failed")
        comparison = artifact.get("refiner_comparison") or {}
        if comparison.get("materialized") is not True:
            stops.append(
                {
                    "detail": (
                        "refiner-off optimizer root unavailable because the frozen "
                        "factory refuses adopted-v1 intervals"
                    ),
                    "id": f"PD-SUBUNIT/{fixture['id']}",
                }
            )
        elif comparison.get("diffs_confined_to_coarse_caused") is not True:
            failures.append(
                f"{fixture['id']}: refiner-off diff is not confined to a "
                "coarse_caused interval"
            )
        rows.append(
            {
                "case": fixture["id"],
                "checks": checks,
                "max_start_error_s": maximum,
                "p90_start_error_s": p90,
                "refiner_comparison": comparison,
                "subunit_split": artifact["subunit_split"],
                "trigger_counts": trigger_counts,
                "variant": fixture["variant"],
            }
        )
    exercised = {
        trigger: any(int(row["trigger_counts"][trigger]) > 0 for row in rows)
        for trigger in ("width", "duration")
    }
    evidence_exercised = {
        "per-char": any(
            int(row["subunit_split"]["evidence"]["per-char"]) > 0 for row in rows
        ),
        "whitespace": any(
            int(row["subunit_split"]["evidence"]["whitespace"]) > 0 for row in rows
        ),
    }
    for name, passed in {**exercised, **evidence_exercised}.items():
        if not passed:
            failures.append(f"coarse family did not exercise {name}")
    return {
        "adjudication": "accepted deterministic derivation registry",
        "cases": rows,
        "evidence_exercised": evidence_exercised,
        "failures": failures,
        "n14": n14_rows,
        "path": str(path),
        "stops": stops,
        "trigger_classes_exercised": exercised,
    }


# ------------------------------------------------------- weight ablation (OAT)


@contextlib.contextmanager
def _ablated_weight(term: str) -> Iterator[None]:
    """Zero exactly one cost term for the duration of the block.

    A deliberate, restored monkeypatch of a production module. The alternative --
    an env knob or an injection point inside ``boundary_cost`` -- would put a
    measurement-only switch in the code path that ships, which is the thing this
    whole lane exists to avoid. Single-threaded, and the ``finally`` restores the
    exact previous object rather than the module's declared default.
    """
    from voxweave.core import boundary_cost as bc

    saved: list[tuple[str, Any]] = []
    speaker_saved: float | None = None
    if term == ABLATION_TERM_PAUSE:
        saved.append(("pause_cut_cost", bc.pause_cut_cost))
        setattr(bc, "pause_cut_cost", lambda evidence: 0.0)
    elif term == ABLATION_TERM_SPEAKER:
        from voxweave.core import speaker_evidence as se

        speaker_saved = se.W_SPEAKER_INTERIOR
        se.W_SPEAKER_INTERIOR = 0.0
    else:
        for name in ABLATION_WEIGHTS[term]:
            saved.append((name, getattr(bc, name)))
            setattr(bc, name, 0.0)
    try:
        yield
    finally:
        for name, value in reversed(saved):
            setattr(bc, name, value)
        if speaker_saved is not None:
            from voxweave.core import speaker_evidence as se

            se.W_SPEAKER_INTERIOR = speaker_saved


def ablation_case_selection(
    cases: Sequence[Case], *, tracked_registry: bool
) -> tuple[list[Case], dict[str, Any]]:
    """Return the fixed metric domain and its explicit exclusions."""
    excluded = ABLATION_TRACKED_EXCLUSIONS if tracked_registry else {}
    selected = [case for case in cases if case.id not in excluded]
    present_exclusions = [
        {"case": case.id, "reason": excluded[case.id]}
        for case in cases
        if case.id in excluded
    ]
    return selected, {
        "candidate_cases": [case.id for case in cases],
        "excluded_cases": present_exclusions,
        "kind": "fixed-metric-domain",
        "requested_cases": [case.id for case in selected],
    }


def run_ablation(
    cases: Sequence[Case], reference: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """One-at-a-time: zero each term, re-solve the corpus, report the deltas.

    Cost is |terms| extra corpus replays -- twelve legacy weights plus the pause
    and speaker terms, so fourteen. That is bounded and known rather than swept:
    a sweep would need a search budget and a stopping rule, and an OAT table
    answers the P5 stability question about the weights ("does any one
    of them carry the whole result").
    """
    rows: list[dict[str, Any]] = []
    unknown: list[str] = []
    requested_groups = {GROUP_ALL: len(cases)}
    for case in cases:
        language = cc.canonical_language_or(case.language, case.language)
        requested_groups[language] = requested_groups.get(language, 0) + 1
    for term in ABLATION_TERMS:
        started = time.perf_counter()
        measured: list[CaseMeasurement] = []
        measured_cases: list[str] = []
        errors: list[dict[str, str]] = []
        with _ablated_weight(term):
            for case in cases:
                try:
                    lane = measure_lane(
                        case,
                        shadow_artifact_of(case, replay_shadow(case)),
                        SHADOW_GATED_LANE,
                        "v2",
                    )
                except cc.CalibrationError as exc:
                    errors.append({"case": case.id, "error": exc.message})
                    continue
                if lane.measurement is None:
                    errors.append(
                        {
                            "case": case.id,
                            "error": lane.error or "gated lane produced no measurement",
                        }
                    )
                    continue
                measured.append(lane.measurement)
                measured_cases.append(case.id)
        groups = aggregate(measured)
        group_counts = {
            group: int((groups.get(group) or {}).get("case_count") or 0)
            for group in sorted(requested_groups)
        }
        missing_groups = [
            group
            for group, requested in sorted(requested_groups.items())
            if requested <= 0 or group_counts[group] != requested
        ]
        missing_cases = sorted({case.id for case in cases} - set(measured_cases))
        complete = (
            bool(cases) and not errors and not missing_cases and not missing_groups
        )
        if not complete:
            unknown.append(
                f"ablation/{term}: incomplete evidence "
                f"({len(measured_cases)}/{len(cases)} cases; "
                f"missing groups={','.join(missing_groups) or 'none'})"
            )
        rows.append(
            {
                "complete": complete,
                "coverage": {
                    "errors": errors,
                    "group_case_counts": group_counts,
                    "measured_cases": measured_cases,
                    "missing_cases": missing_cases,
                    "missing_groups": missing_groups,
                    "requested_cases": [case.id for case in cases],
                    "requested_groups": dict(sorted(requested_groups.items())),
                },
                "deltas": {
                    group: _metric_deltas(groups.get(group), reference.get(group))
                    for group in sorted(set(groups) | set(reference))
                },
                "term": term,
                "wall_time_s": round(time.perf_counter() - started, 4),
                "weights": (
                    ["pause_cut_cost"]
                    if term == ABLATION_TERM_PAUSE
                    else ["W_SPEAKER_INTERIOR"]
                    if term == ABLATION_TERM_SPEAKER
                    else list(ABLATION_WEIGHTS[term])
                ),
            }
        )
    return {
        "complete": not unknown,
        "kind": "one-at-a-time",
        "note": (
            "each row re-solves the whole corpus with one term zeroed; deltas are"
            " ablated minus reference on the gated lane, so a positive delta means"
            " the term was doing work"
        ),
        "rows": rows,
        "terms": list(ABLATION_TERMS),
        "unknown": unknown,
    }


def _metric_deltas(
    block: Mapping[str, Any] | None, reference: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Per-metric ``ablated - reference`` for one group, plus the cue-count move."""
    if block is None or reference is None:
        return {}
    out: dict[str, Any] = {
        "cue_count": int(block["cue_count"]) - int(reference["cue_count"])
    }
    for metric in METRICS:
        value, _, _ = _measure(block, metric)
        base, _, _ = _measure(reference, metric)
        out[metric] = (
            None if value is None or base is None else round(float(value - base), 6)
        )
    return out


# ------------------------------------------------------------- perturbation


@contextlib.contextmanager
def _pinned_barriers(unit_ids: Sequence[int]) -> Iterator[None]:
    """Freeze the robust-silence topology at the unperturbed set (AD-2).

    Every cost and every edge still recomputes on the perturbed stream; only the
    exogenous topology is held still, which is what isolates "the optimizer
    changed its mind" from "the document was cut into different intervals".
    Barriers are re-anchored by *unit id* rather than by node, so the pin
    survives any atom renumbering, and the document ends are always kept because
    an interval set without them is not a partition of anything.

    Both module bindings are patched: ``build_document_lattice`` calls its own
    module global, and ``boundary_v2.score_v1_global`` holds an imported
    reference of its own.
    """
    from voxweave.core import boundary_lattice as bl
    from voxweave.core import boundary_v2 as bv

    frozen = sorted({int(u) for u in unit_ids})
    original = bl.build_barriers

    def pinned(layer: Any, profile: Any) -> Any:
        keep = {b.node: b for b in original(layer, profile) if b.kind == "document"}
        edges: dict[int, int] = {}
        for node in sorted(bl.unit_edge_nodes(layer.atoms)):
            edges.setdefault(layer.unit_bound(node), node)
        for unit_id in frozen:
            node = edges.get(unit_id)
            if node is None or node in keep or not 0 < node < len(layer.atoms):
                continue
            left, right = layer.atoms[node - 1].end, layer.atoms[node].start
            gap = (
                None
                if left is None or right is None
                else max(0.0, (float(right) - float(left)) * 1000.0)
            )
            keep[node] = bl.HardBarrier(
                node=node, unit_id=unit_id, kind="robust-silence", gap_ms=gap
            )
        return tuple(keep[node] for node in sorted(keep))

    bl.build_barriers = pinned
    bv.build_barriers = pinned
    try:
        yield
    finally:
        bl.build_barriers = original
        bv.build_barriers = original


def perturbed_case(case: Case, units: Sequence[Mapping[str, Any]]) -> Case:
    """A case carrying a different unit stream, re-validated as if captured.

    ``load_corpus``'s semantic validators never see a perturbed stream, so they
    are re-run here: a jitter that pushed a unit below zero, inverted a span or
    broke monotonicity would otherwise be graded silently as if it were a
    document somebody could have captured.
    """
    doc = {**case.doc, "word_segments": [dict(u) for u in units]}
    out = Case(
        path=case.path,
        relpath=case.relpath,
        doc=doc,
        size_bytes=case.size_bytes,
    )
    _validate_case_semantics(out)
    return out


def perturbation_unit_stable(
    original: Sequence[Mapping[str, Any]], perturbed: Sequence[Mapping[str, Any]]
) -> bool:
    """P1: a timing probe may alter only start/end, never unit topology."""
    if len(original) != len(perturbed):
        return False
    return all(
        {key: value for key, value in left.items() if key not in {"start", "end"}}
        == {key: value for key, value in right.items() if key not in {"start", "end"}}
        for left, right in zip(original, perturbed)
    )


def shifted_gap_end(
    units: Sequence[Mapping[str, Any]], index: int, delta_ms: float
) -> tuple[float, float]:
    """Where ``units[index]['end']`` lands under a nudge, and by how much.

    Clamped into the window its neighbours leave free -- and the lower bound
    includes the *previous* unit's end, not just this unit's start, because the
    optimizer's own span preflight rejects non-monotone ends and would mark the
    enclosing interval infeasible. A probe that manufactured a fallback would be
    measuring the preflight, not the boundary decision.

    Split out from :func:`perturb_single_gap` because the near-cliff classifier
    asks this question for every gap at every magnitude and needs one float, not
    a copy of the whole stream.
    """
    low = float(units[index]["start"])
    if index > 0:
        low = max(low, float(units[index - 1]["end"]))
    high = float(units[index + 1]["start"])
    before = float(units[index]["end"])
    after = min(max(before + delta_ms / 1000.0, low), high)
    return after, round((after - before) * 1000.0, 6)


def perturb_single_gap(
    units: Sequence[Mapping[str, Any]], index: int, delta_ms: float
) -> tuple[list[dict[str, Any]], float]:
    """Move one unit's end, changing exactly one gap. Returns the applied delta."""
    after, applied = shifted_gap_end(units, index, delta_ms)
    out = [copy.deepcopy(dict(u)) for u in units]
    out[index]["end"] = after
    return out, applied


def perturb_global(
    units: Sequence[Mapping[str, Any]], magnitude_ms: float, rng: random.Random
) -> list[dict[str, Any]]:
    """Jitter every bound, then repair into a stream the validators accept."""
    delta = magnitude_ms / 1000.0
    out = [copy.deepcopy(dict(u)) for u in units]
    start_floor = 0.0
    end_floor = 0.0
    for unit in out:
        start = max(0.0, float(unit["start"]) + rng.uniform(-delta, delta))
        end = float(unit["end"]) + rng.uniform(-delta, delta)
        start = max(start, start_floor)
        end = max(end, start, end_floor)
        unit["start"], unit["end"] = start, end
        start_floor, end_floor = start, end
    return out


def _crosses(low: float | None, high: float | None, knees: Sequence[float]) -> bool:
    """Does moving from ``low`` to ``high`` step over one of these knees?"""
    if low is None or high is None or low == high:
        return False
    lo, hi = (low, high) if low <= high else (high, low)
    return any(lo <= knee <= hi for knee in knees)


def near_cliff_scan(
    case: Case, document: Any, magnitudes: Sequence[int]
) -> dict[str, Any]:
    """AD4-3: classify every gap by whether a nudge crosses a knee of *its* curve.

    Not a scalar list of raw gaps: which knees apply depends on the gap's own VAD
    state, and the overlap fraction is re-derived at the shifted endpoints rather
    than assumed constant -- a gap whose shifted end walks out of a speech span
    changes both its curve and its state, and that state change is itself a
    cliff. The classifier thresholds are tested in raw-gap space as well, because
    the barrier rule reads the raw gap and not the effective one.
    """
    from voxweave.core.boundary_cost import pause_evidence, pause_knees
    from voxweave.core.boundary_lattice import BARRIER_UNCERTAINTY_MS

    profile = document.profile
    speech = document.vad_speech
    knees = pause_knees(profile)
    raw_knees = (
        float(profile.clause_ms),
        float(profile.vad_skip_ms),
        float(profile.vad_skip_ms) + BARRIER_UNCERTAINTY_MS,
    )
    units = case.units
    by_state: dict[str, int] = {}
    near_by_state: dict[str, int] = {}
    near: list[int] = []
    states: list[str] = []
    for index in range(len(units) - 1):
        base = pause_evidence(
            units[index].get("end"),
            units[index + 1].get("start"),
            speech_spans=speech,
            profile=profile,
        )
        states.append(base.vad_state)
        by_state[base.vad_state] = by_state.get(base.vad_state, 0) + 1
        hit = False
        for magnitude in magnitudes:
            for sign in PERTURB_SIGNS:
                end, applied = shifted_gap_end(units, index, sign * magnitude)
                if applied == 0.0:
                    continue
                moved = pause_evidence(
                    end,
                    units[index + 1].get("start"),
                    speech_spans=speech,
                    profile=profile,
                )
                if moved.vad_state != base.vad_state:
                    hit = True
                    break
                candidates = sorted(
                    set(knees.get(base.vad_state, ()))
                    | set(knees.get(moved.vad_state, ()))
                )
                if _crosses(base.effective_ms, moved.effective_ms, candidates):
                    hit = True
                    break
                if _crosses(base.gap_ms_raw, moved.gap_ms_raw, raw_knees):
                    hit = True
                    break
            if hit:
                break
        if hit:
            near.append(index)
            near_by_state[base.vad_state] = near_by_state.get(base.vad_state, 0) + 1
    return {
        "candidate_gaps": len(units) - 1,
        "gap_states": states,
        "knees": {state: list(values) for state, values in sorted(knees.items())},
        "near_cliff": near,
        "near_cliff_by_state": dict(sorted(near_by_state.items())),
        "raw_knees": list(raw_knees),
        "vad_state_denominators": dict(sorted(by_state.items())),
    }


def _probe_partition(case: Case, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay a perturbed stream and read back v2's core partition and barriers."""
    probe = perturbed_case(case, units)
    result = replay_shadow(probe)
    artifact = shadow_artifact_of(probe, result)
    partition = artifact["lanes"][SHADOW_LANE_CORE]["v2"]["partition"]
    return {
        "barrier_units": barrier_unit_ids(seg_document_of(probe, result)),
        "cue_count": int(artifact["lanes"][SHADOW_LANE_CORE]["v2"]["cue_count"]),
        "fallback_intervals": int(artifact["coverage"]["fallback_intervals"]),
        "partition": None if partition is None else {int(u) for u in partition},
    }


def _cell_report(
    base_partition: set[int],
    probe: Mapping[str, Any],
    base_barriers: Sequence[int],
    centres: Sequence[int],
) -> dict[str, Any]:
    """Compare one probe against the unperturbed run inside the influence cell."""
    partition = probe["partition"]
    flips = sorted(set(base_barriers) ^ set(probe["barrier_units"]))
    if partition is None:
        return {
            "barrier_flips": len(flips),
            # ``None``, not ``False``. AD-2's rule is "FAIL when any moved
            # boundary lies outside the influence cell"; a probe that cannot
            # report its moved set has not satisfied that rule, it has failed to
            # answer it. Reporting False made an unmeasurable probe indis-
            # tinguishable from a clean one at the exit, so a probe class that
            # became systematically unreplayable would silently shrink the AD-2
            # denominator while the run kept reporting "no interval crossings".
            "crossed_interval_boundary": None,
            "cue_count": probe["cue_count"],
            "error": "partition unresolved",
            "fallback_intervals": probe["fallback_intervals"],
            "flipped_barrier_units": flips,
            "influence_radius_units": None,
            "moved_count": None,
            "moved_units": [],
            "outside_cell": [],
        }
    moved = sorted(base_partition ^ partition)
    anchors = sorted({*(int(c) for c in centres), *flips})
    outside = [
        unit
        for unit in moved
        if not any(abs(unit - anchor) <= INFLUENCE_RADIUS_UNITS for anchor in anchors)
    ]
    radius = max((min(abs(unit - a) for a in anchors) for unit in moved), default=0)
    return {
        "barrier_flips": len(flips),
        "crossed_interval_boundary": bool(outside),
        "cue_count": probe["cue_count"],
        "error": None,
        "fallback_intervals": probe["fallback_intervals"],
        "flipped_barrier_units": flips,
        "influence_radius_units": radius if moved else 0,
        "moved_count": len(moved),
        "moved_units": moved,
        "outside_cell": outside,
    }


def run_single_gap_probes(
    shadow_case: ShadowCase,
    document: Any,
    *,
    magnitudes: Sequence[int],
    max_probes: int,
    near_cliff_only: bool = False,
) -> dict[str, Any]:
    """AD-2/AD4-3: exhaustive near-cliff probes plus a seeded 10% of the rest.

    ``near_cliff_only`` drops the seeded remainder. It exists so a frozen entry
    point can afford to exercise the AD-2 exit driver at all: near-cliff gaps are
    where a decision can move, and probing only those is a bounded slice rather
    than a reduced-confidence version of the full run. The report says which it
    was (``sampled_skipped``) so the two are never confused.
    """
    case = shadow_case.case
    base_partition = shadow_case.core_partition
    scan = near_cliff_scan(case, document, magnitudes)
    near = set(scan["near_cliff"])
    sampled: dict[int, list[int]] = {}
    for magnitude in magnitudes if not near_cliff_only else ():
        rng = random.Random(f"{case.id}:single_gap:{magnitude}")
        for index in range(scan["candidate_gaps"]):
            if index in near:
                continue
            if rng.random() < PERTURB_SAMPLE_RATE:
                sampled.setdefault(index, []).append(magnitude)

    plan: list[tuple[int, int, int]] = []
    for index in sorted(near):
        for magnitude in magnitudes:
            for sign in PERTURB_SIGNS:
                plan.append((index, magnitude, sign))
    for index in sorted(sampled):
        for magnitude in sorted(sampled[index]):
            for sign in PERTURB_SIGNS:
                plan.append((index, magnitude, sign))
    full_plan = list(plan)
    planned_by_magnitude = {
        str(magnitude): sum(row[1] == magnitude for row in full_plan)
        for magnitude in magnitudes
    }
    truncated = bool(max_probes) and len(plan) > max_probes
    if truncated:
        plan = plan[:max_probes]
    selected_by_magnitude = {
        str(magnitude): sum(row[1] == magnitude for row in plan)
        for magnitude in magnitudes
    }

    probes: list[dict[str, Any]] = []
    skipped = 0
    executed_by_magnitude = {str(magnitude): 0 for magnitude in magnitudes}
    for index, magnitude, sign in plan:
        units, applied = perturb_single_gap(case.units, index, sign * magnitude)
        if applied == 0.0:
            skipped += 1
            continue
        executed_by_magnitude[str(magnitude)] += 1
        row: dict[str, Any] = {
            "applied_delta_ms": applied,
            "case": case.id,
            "gap_index": index,
            "language": case.language,
            "magnitude_ms": magnitude,
            "mode": "single_gap",
            "near_cliff": index in near,
            "probe_unit": index + 1,
            "sign": sign,
            "unit_stable": perturbation_unit_stable(case.units, units),
            "vad_state": scan["gap_states"][index],
        }
        if not row["unit_stable"]:
            row["error"] = "P1 unit-stability precondition failed"
            row["lanes"] = {}
            row["value"] = 0.0
            probes.append(row)
            continue
        lanes: dict[str, Any] = {}
        if base_partition is None:
            row["error"] = "unperturbed partition unresolved"
            row["lanes"] = {}
            row["value"] = 0.0
            probes.append(row)
            continue
        base_set = set(base_partition)
        try:
            lanes["natural"] = _cell_report(
                base_set,
                _probe_partition(case, units),
                shadow_case.barrier_units,
                (index + 1,),
            )
            with _pinned_barriers(shadow_case.barrier_units):
                lanes["pinned"] = _cell_report(
                    base_set,
                    _probe_partition(case, units),
                    shadow_case.barrier_units,
                    (index + 1,),
                )
        except cc.CalibrationError as exc:
            # The clamp is supposed to keep every probe a capturable document; if
            # one still is not, that is a fact about this probe, not a licence to
            # abandon the other several thousand.
            row["error"] = exc.message
            row["lanes"] = {}
            row["value"] = 0.0
            probes.append(row)
            continue
        row["lanes"] = lanes
        row["value"] = float(
            max((len(lane["outside_cell"]) for lane in lanes.values()), default=0)
        )
        probes.append(row)

    retained, summary = retain_probes(probes)
    return {
        "coverage": {
            "candidate_gaps": scan["candidate_gaps"],
            "executed_by_magnitude": executed_by_magnitude,
            "exhaustive": bool(full_plan) and not truncated,
            "knees": scan["knees"],
            "near_cliff": len(near),
            "near_cliff_by_state": scan["near_cliff_by_state"],
            "planned_by_magnitude": planned_by_magnitude,
            "planned_probes": len(full_plan),
            "raw_knees": scan["raw_knees"],
            "sampled_gaps": len(sampled),
            "sample_rate": 0.0 if near_cliff_only else PERTURB_SAMPLE_RATE,
            "sampled_skipped": near_cliff_only,
            "selected_by_magnitude": selected_by_magnitude,
            "skipped_clamped": skipped,
            "vad_state_denominators": scan["vad_state_denominators"],
        },
        "probes": retained,
        "summary": summary,
    }


def _probe_failed(probe: Mapping[str, Any]) -> bool:
    return any(
        lane["crossed_interval_boundary"] is True
        for lane in probe.get("lanes", {}).values()
    )


def _probe_unknown(probe: Mapping[str, Any]) -> bool:
    """A probe that could not answer AD-2's question, in either lane.

    Two shapes reach here: a replay that raised (``lanes`` empty, ``error`` set)
    and a probe whose partition would not resolve (``crossed_interval_boundary``
    is ``None``). Neither is a pass. They are counted separately from failures
    because they mean different things -- a failure is evidence against the
    locality claim, an unknown is the absence of evidence for it -- but both
    refuse the run a clean verdict.
    """
    lanes = probe.get("lanes") or {}
    if not lanes:
        return bool(probe.get("error"))
    return any(lane["crossed_interval_boundary"] is None for lane in lanes.values())


def _probe_movement(probe: Mapping[str, Any]) -> int:
    """How many boundaries this probe moved, over its worst lane."""
    return max(
        (len(lane["moved_units"]) for lane in probe.get("lanes", {}).values()),
        default=0,
    )


def retain_probes(
    probes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Trim a probe set for the report, and count what it was trimmed from.

    Exhaustive near-cliff coverage means thousands of probes per case, and a
    report nobody can open is a report nobody reads. Every FAILING probe is kept
    -- it is the exit driver and must never be summarised away -- and so is every
    UNKNOWN one, plus the worst movers by count. The summary states the true
    totals so a trimmed list can never be mistaken for the whole run.
    """
    failures = [dict(p) for p in probes if _probe_failed(p)]
    unknown = [dict(p) for p in probes if not _probe_failed(p) and _probe_unknown(p)]
    moved = sorted(
        (
            dict(p)
            for p in probes
            if not _probe_failed(p) and not _probe_unknown(p) and _probe_movement(p) > 0
        ),
        key=lambda p: (
            -_probe_movement(p),
            p["gap_index"],
            p["magnitude_ms"],
            p["sign"],
        ),
    )
    retained = [*failures, *unknown, *moved[:OFFENDER_LIMIT]]
    return retained, {
        "barrier_flips_natural": sum(
            int(p["lanes"]["natural"]["barrier_flips"]) for p in probes if p["lanes"]
        ),
        "barrier_flips_pinned": sum(
            int(p["lanes"]["pinned"]["barrier_flips"]) for p in probes if p["lanes"]
        ),
        "errors": sum(1 for p in probes if p.get("error")),
        "failures": len(failures),
        "max_influence_radius_units": max(
            (
                int(lane["influence_radius_units"] or 0)
                for p in probes
                for lane in p["lanes"].values()
            ),
            default=0,
        ),
        "probes": len(probes),
        "retained_probes": len(retained),
        "unknown": len(unknown),
        "with_movement": len(moved),
    }


def run_global_jitter(
    shadow_case: ShadowCase, *, magnitudes: Sequence[int]
) -> dict[str, Any]:
    """Aggregate-only stability under aligner-noise-shaped jitter.

    No influence cell: a global jitter has no probe unit, so there is no centre a
    moved boundary could be near or far from. Reported as churn and barrier
    flips, which is what a stability claim can honestly rest on.
    """
    case = shadow_case.case
    base_partition = shadow_case.core_partition
    rows: list[dict[str, Any]] = []
    for magnitude in magnitudes:
        rng = random.Random(f"{case.id}:global_jitter:{magnitude}")
        for draw in range(PERTURB_JITTER_DRAWS):
            units = perturb_global(case.units, magnitude, rng)
            unit_stable = perturbation_unit_stable(case.units, units)
            if not unit_stable:
                rows.append(
                    {
                        "case": case.id,
                        "draw": draw,
                        "error": "P1 unit-stability precondition failed",
                        "magnitude_ms": magnitude,
                        "unit_stable": False,
                    }
                )
                continue
            try:
                probe = _probe_partition(case, units)
            except cc.CalibrationError as exc:
                rows.append(
                    {
                        "case": case.id,
                        "draw": draw,
                        "error": exc.message,
                        "magnitude_ms": magnitude,
                        "unit_stable": True,
                    }
                )
                continue
            partition = probe["partition"]
            if partition is None or base_partition is None:
                rows.append(
                    {
                        "case": case.id,
                        "draw": draw,
                        "error": "partition unresolved",
                        "magnitude_ms": magnitude,
                        "unit_stable": True,
                    }
                )
                continue
            moved = len(set(base_partition) ^ partition)
            rows.append(
                {
                    "barrier_flips": len(
                        set(shadow_case.barrier_units) ^ set(probe["barrier_units"])
                    ),
                    "case": case.id,
                    "cue_count": probe["cue_count"],
                    "draw": draw,
                    "error": None,
                    "fallback_intervals": probe["fallback_intervals"],
                    "magnitude_ms": magnitude,
                    "moved_count": moved,
                    "unit_stable": True,
                }
            )
    return {
        "coverage": {
            "executed_by_magnitude": {
                str(magnitude): sum(row["magnitude_ms"] == magnitude for row in rows)
                for magnitude in magnitudes
            },
            "planned_by_magnitude": {
                str(magnitude): PERTURB_JITTER_DRAWS for magnitude in magnitudes
            },
        },
        "draws_per_magnitude": PERTURB_JITTER_DRAWS,
        "rows": rows,
    }


def shot_cycle_probe() -> dict[str, Any]:
    """P2's neighbour-free 9f->10f terminal/cycle envelope crossing."""
    from voxweave.core import finalizer as fin
    from voxweave.core.authority import AuthorityLedger
    from voxweave.core.segdoc import DisplayProfile

    frame = 1.0 / 24.0
    profile = DisplayProfile(
        language="en",
        max_line_length=42,
        max_lines=2,
        clause_ms=400.0,
        vad_skip_ms=250.0,
        offline_ms=700.0,
        min_cue_s=0.0,
        max_cue_s=0.0,
        glue_gap_s=0.3,
        cps=0.0,
        lag_out_s=0.0,
        shot_snap_s=11 * frame,
    )

    def solve(frames: int, evaluation_id: str) -> Any:
        cue = {
            "text": "word",
            "start": frames * frame,
            "end": 5.0,
            "word_data": [{"text": "word", "start": None, "end": None}],
            "speech_start": None,
            "speech_end": None,
        }
        ledger = AuthorityLedger()
        capture = fin.capture_v1_reference([cue], ledger=ledger)
        stream = fin.phase1_from_v1_capture(
            capture,
            profile=profile,
            ledger=ledger,
            row_id="delivery_finalizer/v1",
            evaluation_id=evaluation_id,
        )
        return fin.finalize(
            stream,
            profile=profile,
            evidence=fin.FinalizeEvidence(shots=(0.0, 22 * frame)),
            policy=fin.FinalizePolicy(),
        )

    before, after = solve(9, "P2-9f"), solve(10, "P2-10f")
    toggled = {before.trace.terminal, after.trace.terminal} == {
        "fixed-point",
        "cycle-adoption",
    }
    effective = before.cues[0]["start"] != after.cues[0]["start"]
    return {
        "after_start": after.cues[0]["start"],
        "after_terminal": after.trace.terminal,
        "attempted": 1,
        "before_start": before.cues[0]["start"],
        "before_terminal": before.trace.terminal,
        "cycle_terminal_toggled": toggled,
        "effective": int(effective),
        "failures": (
            []
            if toggled and effective
            else ["P2 cycle-adjacent shot probe did not cross its frozen envelope"]
        ),
        "influence_cell": {
            "outside": [],
            "radius_units": 0,
            "unit_count": 1,
        },
    }


def run_perturbation(
    shadow_cases: Sequence[ShadowCase],
    *,
    modes: Sequence[str],
    magnitudes: Sequence[int],
    max_probes: int,
    near_cliff_only: bool = False,
) -> dict[str, Any]:
    """Drive both perturbation modes over the selected cases."""
    single: list[dict[str, Any]] = []
    jitter: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    completeness_unknown: list[dict[str, Any]] = []
    if not shadow_cases:
        completeness_unknown.append(
            {"error": "P1 selected no cases", "mode": "P1-completeness"}
        )
    if not modes:
        completeness_unknown.append(
            {"error": "P1 selected no modes", "mode": "P1-completeness"}
        )
    if not magnitudes:
        completeness_unknown.append(
            {"error": "P1 selected no magnitudes", "mode": "P1-completeness"}
        )
    for shadow_case in shadow_cases:
        if "single_gap" in modes:
            block = run_single_gap_probes(
                shadow_case,
                shadow_case.document,
                magnitudes=magnitudes,
                max_probes=max_probes,
                near_cliff_only=near_cliff_only,
            )
            block["case"] = shadow_case.case.id
            block["language"] = shadow_case.case.language
            single.append(block)
            for magnitude in magnitudes:
                key = str(magnitude)
                planned = int(block["coverage"]["planned_by_magnitude"][key])
                executed = int(block["coverage"]["executed_by_magnitude"][key])
                if planned <= 0 or executed <= 0:
                    completeness_unknown.append(
                        {
                            "case": shadow_case.case.id,
                            "error": (
                                f"P1 single_gap/{magnitude}ms has "
                                f"planned={planned}, executed={executed}"
                            ),
                            "magnitude_ms": magnitude,
                            "mode": "single_gap",
                        }
                    )
            failures.extend(p for p in block["probes"] if _probe_failed(p))
            unknown.extend(
                p for p in block["probes"] if not _probe_failed(p) and _probe_unknown(p)
            )
        if "global_jitter" in modes:
            block = run_global_jitter(shadow_case, magnitudes=magnitudes)
            block["case"] = shadow_case.case.id
            block["language"] = shadow_case.case.language
            jitter.append(block)
            for magnitude in magnitudes:
                key = str(magnitude)
                planned = int(block["coverage"]["planned_by_magnitude"][key])
                executed = int(block["coverage"]["executed_by_magnitude"][key])
                if planned <= 0 or executed <= 0:
                    completeness_unknown.append(
                        {
                            "case": shadow_case.case.id,
                            "error": (
                                f"P1 global_jitter/{magnitude}ms has "
                                f"planned={planned}, executed={executed}"
                            ),
                            "magnitude_ms": magnitude,
                            "mode": "global_jitter",
                        }
                    )
            unknown.extend(
                {
                    "case": shadow_case.case.id,
                    "error": row["error"],
                    "mode": "global_jitter",
                }
                for row in block["rows"]
                if row.get("error")
            )
    p2 = shot_cycle_probe()
    p3 = speaker_cliff_diagnostics()
    p1_attempted = sum(block["summary"]["probes"] for block in single) + sum(
        len(block["rows"]) for block in jitter
    )
    if p1_attempted == 0 and not completeness_unknown:
        completeness_unknown.append(
            {"error": "P1 executed zero probes", "mode": "P1-completeness"}
        )
    unknown.extend({"error": failure, "mode": "P2"} for failure in p2["failures"])
    unknown.extend({"error": failure, "mode": "P3"} for failure in p3["failures"])
    unknown.extend(completeness_unknown)
    p1_failures = sum(
        int(row.get("unit_stable") is False)
        for block in jitter
        for row in block["rows"]
    ) + sum(
        int(probe.get("unit_stable") is False)
        for block in single
        for probe in block["probes"]
    )
    return {
        "classes": {
            "P1-unit-stability": {
                "attempted": p1_attempted,
                "failures": p1_failures,
                "status": (
                    "invalid"
                    if completeness_unknown or p1_attempted == 0
                    else "pass"
                    if p1_failures == 0
                    else "unknown"
                ),
            },
            "P2-shot": p2,
            "P3-speaker-cliffs": p3,
        },
        "exhaustive": bool(shadow_cases)
        and (
            "single_gap" not in modes
            or bool(single)
            and all(block["coverage"]["exhaustive"] for block in single)
        ),
        "failures": failures,
        "global_jitter": jitter,
        "near_cliff_only": near_cliff_only,
        "unknown": unknown,
        "influence_radius_units": INFLUENCE_RADIUS_UNITS,
        "magnitudes_ms": list(magnitudes),
        "modes": list(modes),
        "note": (
            "single_gap runs two lanes per probe: natural (full recompute) and"
            " pinned (frozen robust-silence barriers, costs recomputed);"
            " global_jitter is aggregate-only because it has no probe unit."
            " Per-case `probes` is trimmed to every failure plus the worst movers"
            " -- `summary` counts what it was drawn from"
        ),
        "single_gap": single,
        "totals": {
            "errors": len(unknown),
            "failures": sum(block["summary"]["failures"] for block in single),
            "jitter_draws": sum(len(block["rows"]) for block in jitter),
            "max_influence_radius_units": max(
                (block["summary"]["max_influence_radius_units"] for block in single),
                default=0,
            ),
            "probes": sum(block["summary"]["probes"] for block in single),
            "unknown": len(unknown),
            "with_movement": sum(block["summary"]["with_movement"] for block in single),
        },
    }


# ----------------------------------------------------------- report and CLI


def shadow_exit_code(
    gate_results: Sequence[Mapping[str, Any]],
    c13_failures: Sequence[str],
    perturbation_failures: Sequence[Any],
    perturbation_unknown: Sequence[Any] = (),
    measurement_errors: Sequence[str] = (),
    unauthorized_stops: Sequence[Any] = (),
) -> int:
    """Fold the independent verdicts onto the shared 0/1/2 contract.

    An invalid measurement outranks a failed gate: a run that could not answer
    the question has no standing to call anything a regression (the same rule
    ``gate_exit_code`` already applies to a thin denominator). That is why an
    unevaluable perturbation probe and an unmeasured validator stage exit 2 and
    not 1 -- neither is evidence of a regression, and neither may pass.
    """
    code = gate_exit_code(gate_results)
    if (
        code == cc.EXIT_INVALID
        or measurement_errors
        or perturbation_unknown
        or unauthorized_stops
    ):
        return cc.EXIT_INVALID
    if code == cc.EXIT_GATE_FAILED or c13_failures or perturbation_failures:
        return cc.EXIT_GATE_FAILED
    return cc.EXIT_OK


def load_authorized_deferrals(
    path: str | Path = DEFAULT_AUTHORIZED_DEFERRALS,
) -> set[str]:
    """Read exact section-14 deferral ids from the tracked controller list."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise cc.CalibrationError(
            f"cannot read authorized deferrals: {source}"
        ) from exc
    if not lines or lines[0] != "# Source: p5-s14-addendum.md":
        raise cc.CalibrationError(
            f"{source}: missing exact addendum source header",
            ["expected '# Source: p5-s14-addendum.md' on line 1"],
        )
    ids = [
        line.strip() for line in lines[1:] if line.strip() and not line.startswith("#")
    ]
    if not ids or len(ids) != len(set(ids)):
        raise cc.CalibrationError(
            f"{source}: authorized deferral ids are empty or duplicated"
        )
    return set(ids)


def adjudicate_deferrals(
    stops: Sequence[Mapping[str, Any]],
    *,
    authorized_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Label known frozen-law deferrals and keep every unknown STOP invalid."""
    allowed = load_authorized_deferrals() if authorized_ids is None else authorized_ids
    authorized: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(stops):
        stop_id = raw.get("id")
        detail = raw.get("detail")
        if not isinstance(stop_id, str) or not stop_id or not isinstance(detail, str):
            unknown.append(
                {
                    "detail": f"malformed STOP at index {index}",
                    "id": f"malformed-stop-{index}",
                    "status": "unauthorized_stop",
                }
            )
            continue
        status = "authorized_deferred" if stop_id in allowed else "unauthorized_stop"
        row = {"detail": detail, "id": stop_id, "status": status}
        if stop_id in seen:
            row["detail"] = f"duplicate STOP: {detail}"
            row["status"] = "unauthorized_stop"
            unknown.append(row)
        elif status == "authorized_deferred":
            authorized.append(row)
        else:
            unknown.append(row)
        seen.add(stop_id)
    return authorized, unknown


def _lane_case_row(result: LaneResult) -> dict[str, Any]:
    measurement = result.measurement
    row: dict[str, Any] = {
        "cue_count": result.cue_count,
        "error": result.error,
        "projection": result.projection,
    }
    if measurement is None:
        return row
    row["metrics"] = {
        **{name: ratio.to_dict() for name, ratio in measurement.ratios.items()},
        "cps_p90": {
            "n": len(measurement.cps_samples),
            "value": cc.percentile(measurement.cps_samples, 90.0),
        },
    }
    row["unmapped_boundaries"] = measurement.diagnostics["unmapped_boundaries"]
    return row


def _by_language(shadow_cases: Sequence[ShadowCase], key: str) -> dict[str, int]:
    """Sum one integer/boolean coverage field per language, zeros included.

    Zeros are kept rather than omitted: a column that only appears once it is
    non-zero is a column nobody notices is missing.
    """
    out = {language: 0 for language in cc.CALIBRATION_LANGUAGES}
    for row in shadow_cases:
        language = cc.canonical_language_or(row.case.language, row.case.language)
        out[language] = out.get(language, 0) + int(
            row.artifact["coverage"].get(key) or 0
        )
    return dict(sorted(out.items()))


def build_shadow_report(
    corpus: Corpus,
    shadow_cases: Sequence[ShadowCase],
    *,
    lanes: Mapping[str, Mapping[str, Any]],
    gate_results: Sequence[Mapping[str, Any]],
    coverage_failures: Sequence[str],
    baseline: Mapping[str, Any] | None,
    baseline_path: str | None,
    partial: bool,
    ablation: Mapping[str, Any] | None,
    perturbation: Mapping[str, Any] | None,
    speaker_gates: Mapping[str, Any] | None = None,
    speech_truncation: Mapping[str, Any] | None = None,
    coarse_gates: Mapping[str, Any] | None = None,
    n14: Mapping[str, Any] | None = None,
    stop_items: Sequence[Mapping[str, Any]] = (),
    warnings: Sequence[str] = (),
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the shadow report -- its own file, sharing no block with quality."""
    definition = metric_definition_block()
    first = shadow_cases[0].artifact if shadow_cases else {}
    features: dict[str, list[dict[str, Any]]] = {}
    for row in shadow_cases:
        for group in cc.group_keys(row.case.language):
            features.setdefault(group, []).append(row.features)
    changes: list[dict[str, Any]] = []
    for row in shadow_cases:
        changes.extend(interval_changes(row.case, row.artifact))
    preview_blocks = [
        row.artifact.get("preview_fidelity") or {} for row in shadow_cases
    ]

    report: dict[str, Any] = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "metric_definition": definition,
        "metric_definition_digest": cc.canonical_digest(definition),
        "kind": SHADOW_REPORT_KIND,
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
        "engine_v2": first.get("engine_v2"),
        "policy_name": first.get("policy_name"),
        "policy_version": first.get("policy_version"),
        "policy_deltas": first.get("policy_deltas"),
        "shadow_env": SHADOW_ENV,
        "gated_lane": SHADOW_GATED_LANE,
        "gated_row": SHADOW_GATED_ROW,
        "influence_radius_units": INFLUENCE_RADIUS_UNITS,
        "gates": {metric: dict(SHADOW_GATES[metric]) for metric in METRICS},
        "gate_results": [dict(r) for r in gate_results],
        "preview_fidelity": {
            "checked_edges": sum(
                int(row.get("checked_edges") or 0) for row in preview_blocks
            ),
            "mismatch_count": sum(
                len(row.get("mismatches") or ()) for row in preview_blocks
            ),
            "scored_edges": sum(
                int(row.get("scored_edges") or 0) for row in preview_blocks
            ),
            "selected_mismatch_count": sum(
                len(selected.get("mismatches") or ())
                for block in preview_blocks
                for selected in (block.get("selected_rows") or {}).values()
            ),
            "uncheckable_edges": sum(
                int(row.get("uncheckable_edges") or 0) for row in preview_blocks
            ),
        },
        "speaker_gates": None if speaker_gates is None else dict(speaker_gates),
        "speech_truncation": (
            None if speech_truncation is None else dict(speech_truncation)
        ),
        "coarse_gates": None if coarse_gates is None else dict(coarse_gates),
        "n14": None if n14 is None else dict(n14),
        "stop_items": [dict(item) for item in stop_items],
        "baseline": (
            None
            if baseline is None
            else {
                "path": baseline_path,
                "corpus_digest": baseline["corpus_digest"],
                "metric_definition_digest": baseline["metric_definition_digest"],
                "generated_from_commit": baseline.get("generated_from_commit"),
            }
        ),
        "lanes": {lane: dict(block) for lane, block in sorted(lanes.items())},
        "violations": {
            "by_group": merge_violation_counts(shadow_cases),
            "note": (
                "keys are origin/stage/kind (+/waived); only unwaived v2 rows at"
                " an exit-driving validator stage drive the exit, and the v1 rows are"
                " here so an inherited class can be told from one v2 caused"
            ),
        },
        "coverage": {
            "failures": list(coverage_failures),
            # Two per-language columns that the C13 pass/fail line cannot carry.
            # ``coarse_granularity`` is an input limit, not an optimizer failure
            # (P5 splits below the source unit) even though it still counts in
            # ``fallback_intervals``; ``unprojected_v1`` is an invalid
            # measurement, because a case with no v1 reference has no comparison
            # to report. Both are zero on the tracked word-level corpus, which is
            # exactly why they have to be visible if they ever stop being.
            "coarse_granularity_by_language": _by_language(
                shadow_cases, "coarse_granularity_intervals"
            ),
            "unprojected_v1_by_language": _by_language(shadow_cases, "v1_unprojected"),
            "cases": [
                {
                    "case": row.case.id,
                    "projection_cross_check": row.artifact["lanes"][SHADOW_LANE_CORE][
                        "v2"
                    ].get("projection_cross_check"),
                    **{
                        key: row.artifact["coverage"][key]
                        for key in sorted(row.artifact["coverage"])
                    },
                }
                for row in shadow_cases
            ],
        },
        "features": {
            group: merge_feature_scans(scans)
            for group, scans in sorted(features.items())
        },
        "top_changed_intervals": _worst(changes),
        "ablation": None if ablation is None else dict(ablation),
        "perturbation": None if perturbation is None else dict(perturbation),
        "cases": [
            {
                "id": row.case.id,
                "language": row.case.language,
                "unit_count": len(row.case.units),
                "wall_time_s": round(row.wall_time_s, 4),
                "totals": row.artifact["totals"],
                "coverage": row.artifact["coverage"],
                "authorities": row.artifact.get("authorities"),
                "canonical_fallback_rechecks": row.artifact.get(
                    "canonical_fallback_rechecks"
                ),
                "diff_classification": row.artifact.get("diff_classification"),
                "margin_summary": row.artifact.get("margin_summary"),
                "preview_fidelity": row.artifact.get("preview_fidelity"),
                "refiner_comparison": row.artifact.get("refiner_comparison"),
                "speaker_projection": (row.artifact.get("speaker_evidence") or {}).get(
                    "projection"
                ),
                "v1_projection": row.artifact["v1_projection"],
                "validator": shadow_violation_counts(row.artifact),
                "agreement": {
                    lane: row.artifact["lanes"][lane].get("agreement")
                    for lane in SHADOW_LANES
                },
                "lanes": {
                    lane: {
                        engine: _lane_case_row(row.lanes[(lane, engine)])
                        for engine in SHADOW_LANE_ROWS[lane]
                        if (lane, engine) in row.lanes
                    }
                    for lane in SHADOW_LANES
                },
                "barrier_count": len(row.barrier_units),
                "production_degraded": row.artifact["production_degraded"],
                "shadow_degraded": row.artifact["shadow_degraded"],
            }
            for row in shadow_cases
        ],
        "warnings": list(warnings),
    }
    slowest = max(shadow_cases, key=lambda r: r.wall_time_s, default=None)
    report["timing"] = {
        "base_case_sum_s": round(sum(r.wall_time_s for r in shadow_cases), 4),
        "slowest_case": slowest.case.id if slowest else None,
        "slowest_wall_s": round(slowest.wall_time_s, 4) if slowest else None,
        **({} if timing is None else dict(timing)),
    }
    return report


def print_shadow_summary(report: Mapping[str, Any]) -> None:
    lanes = report["lanes"]
    lens = report["metric_definition"]["forbidden_end"]["ja_tail_lens"]
    print(f"corpus   : {report['corpus']['path']}")
    print(f"digest   : {report['corpus_digest'][:16]}...")
    print(
        f"metric   : {report['metric_definition_digest'][:16]}... ja-tail={lens['id']}"
    )
    print(f"engine   : {report['engine_v2']} policy={report['policy_version']}")
    for lane in SHADOW_LANES:
        block = lanes.get(lane) or {}
        print()
        print(f"  lane {lane}")
        print(
            f"  {'group':<6} {'eng':<3} {'cues':>6}  {'mid-phrase':<20}"
            f"  {'over-max-cue':<14} {'cps_p90':>8}  {'bad-end':<20}"
        )
        print("  " + "-" * 86)
        for name in (GROUP_ALL, *cc.CALIBRATION_LANGUAGES):
            for engine in SHADOW_LANE_ROWS[lane]:
                groups = block.get(engine) or {}
                row = groups.get(name)
                if row is None or not row["case_count"]:
                    continue
                over = row["over_7s_rate"]
                print(
                    f"  {name:<6} {engine:<3} {row['cue_count']:>6}"
                    f"  {_ratio_cell(row['len_break_mid_phrase_rate']):<20}"
                    f"  {'{}/{}'.format(over['bad'], over['eligible']):<14}"
                    f" {_fmt(row['cps_p90']['value'], 2):>8}"
                    f"  {_ratio_cell(row['forbidden_end_rate']):<20}"
                )
    results = report["gate_results"]
    if results:
        print()
        print(f"  non-inferiority gates ({report['gated_lane']}, v2 vs v1 baseline)")
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
                f" {result['mode']:<8} {_gate_value_cell(result)}"
                + (f"  {reasons}" if reasons else "")
            )
    coverage = report["coverage"]
    print()
    print(
        "  coverage    "
        + "  ".join(
            f"{language}: coarse={coverage['coarse_granularity_by_language'][language]}"
            f" unprojected_v1={coverage['unprojected_v1_by_language'][language]}"
            for language in sorted(coverage["coarse_granularity_by_language"])
        )
    )
    disagreed = [
        row["case"]
        for row in coverage["cases"]
        if not (row.get("projection_cross_check") or {}).get("agrees")
    ]
    print(
        f"  projection  cross-check agrees on"
        f" {len(coverage['cases']) - len(disagreed)}/{len(coverage['cases'])} cases"
        + (f" (disagreed: {', '.join(disagreed)})" if disagreed else "")
    )
    failures = coverage["failures"]
    if failures:
        print()
        print("  C13 coverage failures")
        for line in failures[:OFFENDER_LIMIT]:
            print(f"    {line}")
    perturbation = report.get("perturbation")
    if perturbation:
        totals = perturbation["totals"]
        print()
        print(
            f"  perturbation: {totals['probes']} probes,"
            f" {totals['jitter_draws']} jitter draws,"
            f" {totals['failures']} outside the influence cell,"
            f" {totals['unknown']} unevaluable"
        )
        classes = perturbation.get("classes") or {}
        if classes:
            p1 = classes["P1-unit-stability"]
            p2 = classes["P2-shot"]
            p3 = classes["P3-speaker-cliffs"]
            print(
                f"  P1/P2/P3    unit={p1['attempted']}/{p1['failures']}"
                f" shot={p2['effective']}/{p2['attempted']}"
                f" speaker={len(p3['probes'])}/{len(p3['failures'])}"
            )
    coarse = report.get("coarse_gates")
    if coarse:
        print()
        print(
            f"  coarse N4c: {len(coarse['cases'])} cases,"
            f" {len(coarse['failures'])} failures, {len(coarse['stops'])} STOPs"
        )
    n14 = report.get("n14")
    if n14:
        print(
            f"  N14          {n14['oracle_checked']} oracle pairs,"
            f" {len(n14['failures'])} failures, {len(n14['unknown'])} unknown"
        )
    for item in report.get("stop_items") or ():
        print(f"  {item['status']}: {item['id']} -- {item['detail']}")
    speech = report.get("speech_truncation")
    if speech:
        print(
            "  N6 speech   "
            + "  ".join(
                f"{language}: v2={row['v2']} legacy={row['legacy']}"
                for language, row in sorted(speech["languages"].items())
            )
        )
    for warning in report.get("warnings") or ():
        print(f"  warning: {warning}")


def cmd_shadow(args: argparse.Namespace) -> int:
    command_started = time.perf_counter()
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
    if not cases:
        raise cc.CalibrationError(f"{corpus.path} selected no cases")

    warnings: list[str] = []
    stop_items: list[dict[str, str]] = []
    baseline: dict[str, Any] | None = None
    baseline_path = args.baseline
    if (
        baseline_path is None
        and Path(args.corpus).resolve() == DEFAULT_CORPUS.resolve()
    ):
        baseline_path = str(DEFAULT_BASELINE)
    if baseline_path:
        baseline = load_baseline(baseline_path, corpus)
        drift = environment_drift(baseline)
        if drift:
            if args.allow_environment_drift:
                warnings.extend(f"environment drift ignored: {d}" for d in drift)
            else:
                raise cc.CalibrationError(
                    f"{baseline_path} was recorded in a different environment",
                    [
                        *drift,
                        "segmenter versions move where breaks land",
                        "re-record deliberately, or pass --allow-environment-drift",
                    ],
                )

    base_started = time.perf_counter()
    shadow_cases = [run_shadow_case(case) for case in cases]
    base_wall_s = time.perf_counter() - base_started

    coverage_failures: list[str] = []
    measurement_errors: list[str] = []
    tracked_n14 = [
        n14_artifact_evidence(row.artifact, case_id=row.case.id, corpus="tracked")
        for row in shadow_cases
    ]
    for evidence in tracked_n14:
        coverage_failures.extend(evidence["failures"])
    for row in shadow_cases:
        violations = shadow_violation_counts(row.artifact)
        measurement_errors.extend(
            shadow_measurement_errors(row.case, row.artifact, violations)
        )
        coverage_failures.extend(
            c13_case_failures(row.case, row.artifact["coverage"], violations)
        )
        if row.tripwire_error is not None:
            coverage_failures.append(f"{row.case.id}: {row.tripwire_error}")
        for violation in (row.artifact.get("authorities") or {}).get("violations", ()):
            coverage_failures.append(f"{row.case.id}: authority: {violation}")
        classification = row.artifact.get("diff_classification") or {}
        if classification.get("alignment_error"):
            coverage_failures.append(f"{row.case.id}: N11 unit-range alignment failed")
        if int(classification.get("unclassified_field_diff") or 0):
            coverage_failures.append(
                f"{row.case.id}: N11 has "
                f"{classification['unclassified_field_diff']} unclassified field diff(s)"
            )
        if int(classification.get("relation_failures") or 0):
            coverage_failures.append(
                f"{row.case.id}: N11 has "
                f"{classification['relation_failures']} allowed-relation failure(s)"
            )
        if classification.get("trigger_mismatches"):
            coverage_failures.append(
                f"{row.case.id}: N11 producer/independent triggers disagree: "
                + ", ".join(classification["trigger_mismatches"])
            )
        refiner = row.artifact.get("refiner_comparison") or {}
        if (
            refiner.get("status") == "tracked-identity"
            and refiner.get("byte_identical") is not True
        ):
            coverage_failures.append(
                f"{row.case.id}: tracked refiner-on/off optimizer artifacts differ"
            )
        finalizer_rows = row.artifact["lanes"][SHADOW_LANE_FINALIZER]["rows"]
        for row_id in ("v1", "v2"):
            finalizer = (finalizer_rows.get(row_id) or {}).get("finalizer") or {}
            fallback_rows = [
                entry
                for entry in finalizer.get("entries") or ()
                if entry.get("kind") == "canonical-text-fallback"
            ]
            rechecks = [
                check
                for check in row.artifact.get("canonical_fallback_rechecks") or ()
                if check.get("row") == row_id
            ]
            if (
                fallback_rows
                and all(
                    (entry.get("evidence") or {}).get("reason")
                    == "granularity-unreconciled"
                    for entry in fallback_rows
                )
                and len(rechecks) == len(fallback_rows)
                and all(
                    check.get("with_owned_footprint") == "word-data"
                    for check in rechecks
                )
            ):
                item = {
                    "detail": (
                        f"{len(fallback_rows)} canonical fallback(s); the frozen W1 "
                        "authority factory exposes no owned-footprint input, although "
                        "the same word_data reconciles when that normative footprint "
                        "is supplied"
                    ),
                    "id": f"N20/{row.case.id}/{row_id}",
                }
                stop_items.append(item)
            elif fallback_rows:
                coverage_failures.append(
                    f"{row.case.id}: {row_id} fired "
                    f"{len(fallback_rows)} canonical fallback(s)"
                )
        if int(row.artifact["coverage"].get("coarse_granularity_intervals") or 0):
            warnings.append(
                f"{row.case.id}: "
                f"{row.artifact['coverage']['coarse_granularity_intervals']}"
                " interval(s) fell back because the source units are coarser than"
                " a cue -- an input-granularity limit (P5 splits below the unit),"
                " not an optimizer failure"
            )
        if violations["not_conservation_evidence"]:
            warnings.append(
                f"{row.case.id}: {violations['not_conservation_evidence']} raw-stage"
                " conservation rows come from overlapping adopted_v1 ranges and are"
                " not conservation evidence"
            )
        for key, lane in sorted(row.lanes.items()):
            if lane.error is None:
                continue
            message = (
                f"{row.case.id}: lane {key[0]}/{key[1]} not measured: {lane.error}"
            )
            if key[0] == SHADOW_GATED_LANE:
                raise cc.CalibrationError(
                    f"{row.case.relpath}: the gated lane is not measurable", [message]
                )
            warnings.append(message)

    if measurement_errors:
        raise cc.CalibrationError(
            "the shadow run did not produce a valid measurement", measurement_errors
        )

    lanes = {
        lane: {
            engine: lane_groups(shadow_cases, lane, engine)
            for engine in SHADOW_LANE_ROWS[lane]
            if any((lane, engine) in row.lanes for row in shadow_cases)
        }
        for lane in SHADOW_LANES
    }
    schema_problems = [
        f"{lane}/{engine}/{error}"
        for lane, block in sorted(lanes.items())
        for engine, groups in sorted(block.items())
        for error in shadow_group_errors(groups)
    ]
    if schema_problems:
        raise cc.CalibrationError(
            "a shadow lane aggregate is not schema-valid", schema_problems
        )

    gated = lanes[SHADOW_GATED_LANE][SHADOW_GATED_ROW]
    n1_results = [] if partial else evaluate_gates(gated, SHADOW_GATES, baseline)
    for result in n1_results:
        result["family"] = "N1-finalizer-v2-vs-tracked-baseline"
    n3a_results = [] if partial else finalizer_vs_legacy_gates(shadow_cases)
    gate_results = [*n1_results, *n3a_results]
    tracked_registry = Path(corpus.path).resolve() == DEFAULT_CORPUS.resolve()
    tracked_full = not partial and tracked_registry
    speaker_gates = speaker_gate_block(shadow_cases)
    speech_truncation = None if partial else speech_truncation_gates(shadow_cases)
    if speech_truncation is not None:
        coverage_failures.extend(speech_truncation["failures"])
    if tracked_full:
        for gate in speaker_gates["gates"]:
            if gate["status"] == "fail":
                coverage_failures.append(
                    f"{gate['id']}: value={gate.get('value')} target={gate.get('target')}"
                )
            elif gate["status"] == "stopped":
                item = {
                    "detail": (
                        f"requested rate {gate['target']:.6f} exceeds the frozen "
                        f"lineage's expressible ceiling {gate['possible_rate']:.6f}"
                    ),
                    "id": str(gate["id"]),
                }
                stop_items.append(item)
    if partial:
        warnings.append("partial run (--case): non-inferiority gates skipped")

    coarse_started = time.perf_counter()
    coarse_gates = run_coarse_gates(corpus) if tracked_full else None
    coarse_wall_s = time.perf_counter() - coarse_started
    if coarse_gates is not None:
        coverage_failures.extend(coarse_gates["failures"])
        for coarse_stop in coarse_gates["stops"]:
            stop_items.append(dict(coarse_stop))

    n14_evidence = [
        *tracked_n14,
        *(coarse_gates.get("n14") or () if coarse_gates is not None else ()),
    ]
    n14_matrix = n14_matrix_unknown(n14_evidence) if tracked_full else []
    n14_unknown = [
        *(item for evidence in n14_evidence for item in evidence.get("unknown") or ()),
        *n14_matrix,
    ]
    n14_code = n14_exit_code(n14_evidence, require_matrix=tracked_full)
    n14_report = {
        "cases": n14_evidence,
        "failures": [
            item for evidence in n14_evidence for item in evidence.get("failures") or ()
        ],
        "matrix_unknown": n14_matrix,
        "oracle_checked": sum(
            int((evidence.get("oracle") or {}).get("checked") or 0)
            for evidence in n14_evidence
        ),
        "status": {
            cc.EXIT_OK: "pass",
            cc.EXIT_GATE_FAILED: "fail",
            cc.EXIT_INVALID: "invalid",
        }[n14_code],
        "unknown": n14_unknown,
    }

    ablation_started = time.perf_counter()
    ablation = None
    if args.ablation:
        ablation_cases, ablation_selection = ablation_case_selection(
            cases, tracked_registry=tracked_registry
        )
        selected_ids = {case.id for case in ablation_cases}
        ablation_reference = lane_groups(
            [row for row in shadow_cases if row.case.id in selected_ids],
            SHADOW_GATED_LANE,
            SHADOW_GATED_ROW,
        )
        ablation = run_ablation(ablation_cases, ablation_reference)
        ablation["selection"] = ablation_selection
    ablation_wall_s = time.perf_counter() - ablation_started
    ablation_unknown = list(ablation["unknown"]) if ablation is not None else []

    perturbation_started = time.perf_counter()
    perturbation = None
    if args.perturb:
        selected = shadow_cases
        if args.perturb_case:
            wanted = set(args.perturb_case)
            selected = [r for r in shadow_cases if r.case.id in wanted]
            missing = wanted - {r.case.id for r in selected}
            if missing:
                raise cc.CalibrationError(
                    "no such case in this run: " + ", ".join(sorted(missing))
                )
        perturbation = run_perturbation(
            selected,
            modes=args.perturb_mode or list(PERTURB_MODES),
            magnitudes=args.perturb_magnitude or list(PERTURB_MAGNITUDES_MS),
            max_probes=int(args.perturb_max_probes),
            near_cliff_only=bool(args.perturb_near_cliff_only),
        )
        if not perturbation["exhaustive"]:
            warnings.append(
                "--perturb-max-probes truncated the near-cliff set: this run is not"
                " exhaustive coverage and must not be reported as such"
            )
        if perturbation["near_cliff_only"]:
            warnings.append(
                "--perturb-near-cliff-only skipped the seeded 10% sample of"
                " non-near-cliff gaps: a bounded AD-2 slice, not full coverage"
            )
        warnings.extend(perturbation["classes"]["P3-speaker-cliffs"]["warnings"])
    perturbation_wall_s = time.perf_counter() - perturbation_started

    report_started = time.perf_counter()
    authorized_deferrals, unauthorized_stops = adjudicate_deferrals(stop_items)
    adjudicated_stops = [*authorized_deferrals, *unauthorized_stops]
    report = build_shadow_report(
        corpus,
        shadow_cases,
        lanes=lanes,
        gate_results=gate_results,
        coverage_failures=coverage_failures,
        baseline=baseline,
        baseline_path=baseline_path,
        partial=partial,
        ablation=ablation,
        perturbation=perturbation,
        speaker_gates=speaker_gates,
        speech_truncation=speech_truncation,
        coarse_gates=coarse_gates,
        n14=n14_report,
        stop_items=adjudicated_stops,
        warnings=warnings,
        timing={
            "ablation_wall_s": round(ablation_wall_s, 4),
            "base_wall_s": round(base_wall_s, 4),
            "coarse_wall_s": round(coarse_wall_s, 4),
            "perturbation_wall_s": round(perturbation_wall_s, 4),
        },
    )
    report_wall_s = time.perf_counter() - report_started
    print_shadow_summary(report)

    perturbation_failures = (
        list(perturbation["failures"]) if perturbation is not None else []
    )
    perturbation_unknown = (
        list(perturbation["unknown"]) if perturbation is not None else []
    )
    # The verdict is computed and reported either way; ``--check`` only decides
    # whether it reaches the process exit. A summary line reading "pass" beside a
    # non-zero failure count would be the harness lying to a log parser.
    verdict = shadow_exit_code(
        gate_results,
        coverage_failures,
        perturbation_failures,
        perturbation_unknown,
        [*ablation_unknown, *n14_unknown],
        unauthorized_stops,
    )
    code = verdict if args.check else cc.EXIT_OK
    failures = (
        sum(
            1 for r in gate_results if r["mode"] == "blocking" and r["status"] != "pass"
        )
        + len(coverage_failures)
        + len(perturbation_failures)
        + len(perturbation_unknown)
        + len(ablation_unknown)
        + len(n14_unknown)
        + len(unauthorized_stops)
    )
    warned = (
        sum(1 for r in gate_results if r["mode"] == "warning" and r["status"] != "pass")
        + len(warnings)
        + len(authorized_deferrals)
    )
    status = {
        cc.EXIT_OK: "pass",
        cc.EXIT_GATE_FAILED: "fail",
        cc.EXIT_INVALID: "invalid",
    }[verdict]
    total_wall_s = time.perf_counter() - command_started
    accounted = (
        base_wall_s
        + coarse_wall_s
        + ablation_wall_s
        + perturbation_wall_s
        + report_wall_s
    )
    report["timing"].update(
        {
            "other_wall_s": round(max(0.0, total_wall_s - accounted), 4),
            "report_wall_s": round(report_wall_s, 4),
            "total_wall_s": round(total_wall_s, 4),
        }
    )
    destination = "-"
    if args.json_out:
        destination = str(cc.write_json(args.json_out, report))
    print(
        f"QUALITY segmentation-shadow status={status} cases={len(shadow_cases)}"
        f" failures={failures} warnings={warned} report={destination}"
    )
    return code


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

    shadow = sub.add_parser(
        "shadow",
        help="P5: measure the complete shadow lane/row matrix (separate from `quality`)",
    )
    shadow.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    shadow.add_argument(
        "--baseline",
        default=None,
        help="tracked v1 baseline the non-inferiority gate compares against",
    )
    shadow.add_argument("--json-out", default=str(DEFAULT_SHADOW_REPORT))
    shadow.add_argument(
        "--case",
        action="append",
        default=None,
        help="restrict to one case id (repeatable); skips the gates",
    )
    shadow.add_argument(
        "--check",
        action="store_true",
        help="apply the exits: 1 on a C13/gate/perturbation failure, 2 on invalid",
    )
    shadow.add_argument(
        "--allow-environment-drift",
        action="store_true",
        help="downgrade a segmenter-version mismatch with the baseline to a warning",
    )
    shadow.add_argument(
        "--no-ablation",
        dest="ablation",
        action="store_false",
        default=True,
        help="skip the one-at-a-time weight ablation (14 extra corpus replays)",
    )
    shadow.add_argument(
        "--perturb",
        action="store_true",
        help="also run the AD-2 perturbation probes (expensive; see --perturb-case)",
    )
    shadow.add_argument(
        "--perturb-case",
        action="append",
        default=None,
        help="restrict perturbation to one case id (repeatable)",
    )
    shadow.add_argument(
        "--perturb-mode",
        action="append",
        choices=list(PERTURB_MODES),
        default=None,
        help=f"repeatable; default {', '.join(PERTURB_MODES)}",
    )
    shadow.add_argument(
        "--perturb-magnitude",
        action="append",
        type=int,
        default=None,
        help=f"repeatable, in ms; default {', '.join(map(str, PERTURB_MAGNITUDES_MS))}",
    )
    shadow.add_argument(
        "--perturb-near-cliff-only",
        action="store_true",
        help="probe only near-cliff gaps (skip the seeded 10%% sample): a bounded"
        " AD-2 slice cheap enough for the frozen Makefile target",
    )
    shadow.add_argument(
        "--perturb-max-probes",
        type=int,
        default=0,
        help="0 = exhaustive near-cliff coverage; any cap marks the run non-exhaustive",
    )
    shadow.set_defaults(func=cmd_shadow)

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
