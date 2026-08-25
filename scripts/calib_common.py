"""Shared primitives for the voxweave calibration harnesses.

Both rulers -- ``scripts/calib_alignment.py`` (acoustic / display accuracy) and
``scripts/calib_segmentation.py`` (subtitle segmentation quality) -- import this
module so that they cannot drift apart on the boring parts: schema validation,
one percentile definition, one digest definition, one language canonicalization,
micro aggregation and the three exit codes.

Deliberate non-goals: nothing here decides *what* a metric means.  Metric
definitions belong to the harness that owns them; this module must never grow
into a second pipeline.

Import weight is a hard constraint.  The module stays importable in a bare
environment whose only third-party package is ``jsonschema`` -- no torch, no
model code -- so schema and contract checks can run anywhere, including a CI job
that never installs the inference stack.  ``voxweave.lang`` is reused when it
happens to be importable (it is pure stdlib) and mirrored by a local fallback
table otherwise.

Exit codes (shared by every calibration CLI):

* ``0`` -- data valid and every enabled gate passed.
* ``1`` -- data valid but a quality gate failed.
* ``2`` -- manifest / schema / coverage / reference / tooling invalid; this run
  has no standing to judge quality at all.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

__all__ = [
    "CALIBRATION_LANGUAGES",
    "DEFAULT_ERROR_THRESHOLDS",
    "EXIT_GATE_FAILED",
    "EXIT_INVALID",
    "EXIT_OK",
    "REPO_ROOT",
    "SCHEMA_DIR",
    "CalibrationError",
    "MicroAggregator",
    "Ratio",
    "canonical_digest",
    "canonical_json",
    "canonical_language",
    "canonical_language_or",
    "die_gate",
    "die_invalid",
    "exit_code",
    "group_keys",
    "is_calibration_language",
    "languages_match",
    "load_schema",
    "load_validated_json",
    "merge_ratios",
    "metric_block",
    "percentile",
    "read_json",
    "read_json_or_exit2",
    "require_calibration_language",
    "run_cli",
    "schema_errors",
    "sha256_file",
    "threshold_key",
    "validate_or_exit2",
    "write_json",
]

# --------------------------------------------------------------------------- #
# Exit codes and errors
# --------------------------------------------------------------------------- #

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_INVALID = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "calibration" / "schemas"

#: The only languages the calibration corpora cover (see design doc 3.1 / 4.4).
CALIBRATION_LANGUAGES = ("en", "ja", "zh")


class CalibrationError(Exception):
    """Invalid input, schema, coverage or tooling -- always maps to exit code 2.

    Raised (not printed) by library helpers so that callers can add context.
    ``run_cli`` turns it into the stable exit code at the process boundary.
    """

    def __init__(self, message: str, details: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.details = list(details)

    def render(self) -> str:
        lines = [self.message, *(f"  - {d}" for d in self.details)]
        return "\n".join(lines)


def exit_code(*, valid: bool, gates_passed: bool) -> int:
    """Map the two independent outcomes onto the shared 0/1/2 contract."""
    if not valid:
        return EXIT_INVALID
    return EXIT_OK if gates_passed else EXIT_GATE_FAILED


def die_invalid(message: str, details: Sequence[str] = ()) -> NoReturn:
    """Report an invalid run on stderr and exit 2."""
    print(CalibrationError(message, details).render(), file=sys.stderr)
    raise SystemExit(EXIT_INVALID)


def die_gate(message: str, details: Sequence[str] = ()) -> NoReturn:
    """Report a failed quality gate on stderr and exit 1 (the data was valid)."""
    print(CalibrationError(message, details).render(), file=sys.stderr)
    raise SystemExit(EXIT_GATE_FAILED)


def run_cli(main: Callable[[], int]) -> NoReturn:
    """Run ``main`` and exit with the shared contract, mapping errors to 2.

    Keeping the mapping here means no harness can accidentally return 1 for a
    broken manifest, which would read as "quality regressed" in CI.
    """
    try:
        code = main()
    except CalibrationError as exc:
        print(exc.render(), file=sys.stderr)
        raise SystemExit(EXIT_INVALID) from None
    raise SystemExit(int(code))


# --------------------------------------------------------------------------- #
# Canonical JSON, digests and plain I/O
# --------------------------------------------------------------------------- #


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` deterministically: sorted keys, compact separators, UTF-8 text.

    Only this spelling is allowed to feed a digest, so a digest recorded on one
    machine keeps matching after unrelated formatting changes.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_digest(obj: Any) -> str:
    """sha256 of the canonical JSON form of ``obj`` (lowercase hex)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """sha256 of raw file bytes -- for media and model artifacts, not JSON documents."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    """Parse a JSON file, raising ``CalibrationError`` on any read/parse problem."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationError(f"cannot read {p}: {exc}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{p} is not valid JSON: {exc}") from None


def read_json_or_exit2(path: str | Path) -> Any:
    try:
        return read_json(path)
    except CalibrationError as exc:
        die_invalid(exc.message, exc.details)


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """Write a report atomically (same-directory temp file + replace), creating parents."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


# --------------------------------------------------------------------------- #
# JSON Schema
# --------------------------------------------------------------------------- #

_SCHEMA_CACHE: dict[Path, dict[str, Any]] = {}


def _jsonschema_module() -> Any:
    # Imported lazily so that percentile/digest/language helpers stay usable in an
    # environment without jsonschema, and a missing install reports as exit 2.
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise CalibrationError(
            "jsonschema is required for calibration validation "
            "(dev group: `uv sync --extra cuda --dev`)",
            [str(exc)],
        ) from None
    return jsonschema


def load_schema(name: str, *, schema_dir: Path | None = None) -> dict[str, Any]:
    """Load a tracked schema by short name (``segmentation-case``) or file name."""
    directory = schema_dir or SCHEMA_DIR
    filename = name if name.endswith(".json") else f"{name}.schema.json"
    path = (directory / filename).resolve()
    cached = _SCHEMA_CACHE.get(path)
    if cached is not None:
        return cached
    schema = read_json(path)
    if not isinstance(schema, dict):
        raise CalibrationError(f"{path} is not a JSON Schema object")
    _SCHEMA_CACHE[path] = schema
    return schema


def _resolve_schema(schema: str | Mapping[str, Any]) -> Mapping[str, Any]:
    return load_schema(schema) if isinstance(schema, str) else schema


def schema_errors(
    instance: Any,
    schema: str | Mapping[str, Any],
    *,
    limit: int = 20,
) -> list[str]:
    """Return human-readable validation errors, best-first, capped at ``limit``.

    All errors are reported at once: fixing a manifest one error per run is the
    kind of friction that makes people stop running the harness.
    """
    jsonschema = _jsonschema_module()
    validator = jsonschema.Draft202012Validator(_resolve_schema(schema))
    out: list[str] = []
    for err in sorted(
        validator.iter_errors(instance), key=jsonschema.exceptions.relevance
    ):
        where = "/".join(str(part) for part in err.absolute_path) or "<root>"
        out.append(f"{where}: {err.message}")
        if len(out) >= limit:
            out.append("... (further errors suppressed)")
            break
    return out


def validate_or_exit2(
    instance: Any,
    schema: str | Mapping[str, Any],
    *,
    label: str = "document",
) -> None:
    """Validate ``instance`` or terminate the process with exit code 2."""
    errors = schema_errors(instance, schema)
    if errors:
        die_invalid(f"{label} failed schema validation", errors)


def load_validated_json(
    path: str | Path,
    schema: str | Mapping[str, Any],
    *,
    label: str | None = None,
) -> Any:
    """Read a JSON file and validate it, raising ``CalibrationError`` on failure."""
    doc = read_json(path)
    errors = schema_errors(doc, schema)
    if errors:
        raise CalibrationError(
            f"{label or Path(path)} failed schema validation", errors
        )
    return doc


# --------------------------------------------------------------------------- #
# Language tags
# --------------------------------------------------------------------------- #

# ISO-639-2/3 and legacy tags seen in container metadata (ffprobe stream tags),
# MFA corpora and dictionary names. voxweave.lang only speaks ISO-639-1 plus the
# aligner's English names, so these are resolved before delegating to it.
_ISO3_TO_ISO1 = {
    "chi": "zh",
    "cmn": "zh",
    "zho": "zh",
    "eng": "en",
    "jpn": "ja",
    "kor": "ko",
    "yue": "yue",
    "deu": "de",
    "ger": "de",
    "fra": "fr",
    "fre": "fr",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "spa": "es",
}

# Minimal mirror of voxweave.lang's name table, used only when voxweave is not
# importable (bare schema-checking environment).
_NAME_TO_ISO1 = {
    "cantonese": "yue",
    "chinese": "zh",
    "mandarin": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}

_ISO1_CODES = frozenset(
    {"en", "ja", "zh", "yue", "ko", "de", "fr", "it", "pt", "ru", "es"}
)


def _primary_subtag(raw: str) -> str:
    """BCP-47-ish primary subtag: ``en-US`` / ``zh_Hans_CN`` / ``ja-JP`` -> ``en`` / ``zh`` / ``ja``."""
    return raw.strip().strip("\"'").lower().replace("_", "-").split("-", 1)[0]


def _voxweave_iso(value: str) -> str | None:
    """Delegate to voxweave.lang when the package is importable; None otherwise.

    The import is local and optional on purpose: voxweave.lang is pure stdlib and
    is the authority on the aligner's language names, but this module must keep
    working where voxweave itself is not installed.
    """
    try:
        from voxweave.lang import to_iso_or
    except ImportError:
        return None
    return to_iso_or(value, None)


def canonical_language_or(raw: str | None, default: str | None) -> str | None:
    """Canonicalize a language tag to an ISO-639-1 code, or return ``default``.

    Accepts what the surrounding tooling actually emits: BCP-47 tags with region
    or script subtags, ISO-639-2/3 codes from container metadata, and the
    aligner's English language names.
    """
    if raw is None:
        return default
    primary = _primary_subtag(str(raw))
    if not primary:
        return default
    if primary in _ISO3_TO_ISO1:
        return _ISO3_TO_ISO1[primary]
    if primary in _ISO1_CODES:
        return primary
    if primary in _NAME_TO_ISO1:
        return _NAME_TO_ISO1[primary]
    return _voxweave_iso(primary) or default


def canonical_language(raw: str | None) -> str:
    """Strict form of :func:`canonical_language_or`; unknown tags are invalid input."""
    iso = canonical_language_or(raw, None)
    if iso is None:
        raise CalibrationError(f"unrecognized language tag {raw!r}")
    return iso


def is_calibration_language(raw: str | None) -> bool:
    return canonical_language_or(raw, None) in CALIBRATION_LANGUAGES


def require_calibration_language(raw: str | None) -> str:
    """Canonicalize and assert the tag is one of the corpus languages."""
    iso = canonical_language(raw)
    if iso not in CALIBRATION_LANGUAGES:
        raise CalibrationError(
            f"language {raw!r} resolves to {iso!r}, outside the calibration set "
            f"{', '.join(CALIBRATION_LANGUAGES)}"
        )
    return iso


def languages_match(a: str | None, b: str | None) -> bool:
    """True only when both tags canonicalize to the same known language.

    Cross-language pairing is never allowed to be silently "close enough": an
    English release track is not a reference for a Japanese lane.
    """
    left = canonical_language_or(a, None)
    right = canonical_language_or(b, None)
    return left is not None and left == right


# --------------------------------------------------------------------------- #
# Statistics: one percentile definition, one metric block shape
# --------------------------------------------------------------------------- #


def percentile(
    values: Sequence[float], p: float, *, default: float | None = None
) -> float | None:
    """Type-7 (R-7 / numpy default) linear-interpolation percentile, ``p`` in [0, 100].

    Pinned by unit test: every calibration percentile -- alignment p90, cps_p90,
    legacy calib_segmentation output -- must come from this one definition, or
    baselines recorded by one harness stop meaning anything to another.
    """
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p must be in [0, 100], got {p!r}")
    data = sorted(float(v) for v in values)
    n = len(data)
    if n == 0:
        return default
    if n == 1:
        return data[0]
    idx = (n - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return data[-1]
    frac = idx - lo
    return data[lo] * (1 - frac) + data[hi] * frac


#: Absolute-error thresholds reported for every alignment metric block.
DEFAULT_ERROR_THRESHOLDS: tuple[float, ...] = (0.025, 0.05, 0.10, 0.25, 1.0)

# Report field names are part of the schema, so the spelling is a table, not a
# format string (0.10 -> "0_10" but 0.05 -> "0_05" is not derivable by rounding).
_THRESHOLD_KEYS = {
    0.025: "pct_le_0_025",
    0.05: "pct_le_0_05",
    0.10: "pct_le_0_10",
    0.25: "pct_le_0_25",
    1.0: "pct_le_1_0",
}

# Comparisons at a threshold are inclusive; the epsilon only absorbs binary
# representation noise from subtracting two timestamps.
_THRESHOLD_EPS = 1e-9


def threshold_key(threshold: float) -> str:
    """Report field name for an absolute-error threshold."""
    known = _THRESHOLD_KEYS.get(threshold)
    if known is not None:
        return known
    return "pct_le_" + format(threshold, "g").replace(".", "_").replace("-", "m")


def metric_block(
    errors: Iterable[float],
    *,
    thresholds: Sequence[float] = DEFAULT_ERROR_THRESHOLDS,
    ci95: tuple[float, float] | None = None,
    interpretive_lower_bound: float | None = None,
) -> dict[str, Any]:
    """Build one ``alignment-report`` metric block from raw absolute errors.

    The block always carries ``n`` next to the aggregates: an error figure whose
    sample count is not visible invites reading a two-sample lane as a result.
    Empty input yields ``n = 0`` with null aggregates instead of a fake zero.
    """
    data: list[float] = []
    for raw in errors:
        value = float(raw)
        if not math.isfinite(value):
            raise CalibrationError(f"non-finite error sample {raw!r}")
        if value < 0:
            raise CalibrationError(f"negative absolute error sample {raw!r}")
        data.append(value)
    data.sort()
    n = len(data)

    block: dict[str, Any] = {
        "n": n,
        "mae": None,
        "median": None,
        "p90": None,
    }
    for t in thresholds:
        block[threshold_key(t)] = None
    if n:
        block["mae"] = sum(data) / n
        block["median"] = percentile(data, 50.0)
        block["p90"] = percentile(data, 90.0)
        for t in thresholds:
            hits = sum(1 for v in data if v <= t + _THRESHOLD_EPS)
            block[threshold_key(t)] = hits / n
    if ci95 is not None:
        block["ci95_low"], block["ci95_high"] = float(ci95[0]), float(ci95[1])
    if interpretive_lower_bound is not None:
        block["interpretive_lower_bound"] = float(interpretive_lower_bound)
    return block


# --------------------------------------------------------------------------- #
# Micro aggregation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ratio:
    """A rate that keeps its numerator and denominator.

    Percentages are aggregated by summing ``bad`` and ``eligible`` across cases,
    never by averaging per-case rates -- a 60 s clip must not weigh the same as
    a 150 s clip. ``value`` is ``None`` (not 0.0) when nothing was eligible, so
    "no measurable boundary" cannot masquerade as a perfect score.
    """

    bad: int = 0
    eligible: int = 0

    def __post_init__(self) -> None:
        if self.bad < 0 or self.eligible < 0:
            raise CalibrationError(
                f"negative ratio counts: bad={self.bad} eligible={self.eligible}"
            )
        if self.bad > self.eligible:
            raise CalibrationError(
                f"ratio numerator exceeds denominator: {self.bad}/{self.eligible}"
            )

    @property
    def value(self) -> float | None:
        return self.bad / self.eligible if self.eligible else None

    def __add__(self, other: Ratio) -> Ratio:
        if not isinstance(other, Ratio):  # pragma: no cover - defensive
            return NotImplemented
        return Ratio(self.bad + other.bad, self.eligible + other.eligible)

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the ``segmentation-baseline`` ratio shape."""
        return {"bad": self.bad, "eligible": self.eligible, "value": self.value}


def merge_ratios(parts: Iterable[Ratio]) -> Ratio:
    """Micro-aggregate ratios: sum numerators and denominators."""
    total = Ratio()
    for part in parts:
        total = total + part
    return total


def group_keys(language: str) -> tuple[str, str]:
    """Reporting groups a sample belongs to: the ``all`` summary and its language."""
    return ("all", require_calibration_language(language))


@dataclass
class MicroAggregator:
    """Accumulates ratios and raw sample pools per ``(group, metric)``.

    Raw samples are pooled rather than pre-reduced because a percentile of
    per-case percentiles is not a percentile of anything.
    """

    _ratios: dict[tuple[str, str], Ratio] = field(default_factory=dict)
    _samples: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    def add_ratio(self, group: str, metric: str, bad: int, eligible: int) -> None:
        key = (group, metric)
        self._ratios[key] = self._ratios.get(key, Ratio()) + Ratio(bad, eligible)

    def add_samples(self, group: str, metric: str, values: Iterable[float]) -> None:
        self._samples.setdefault((group, metric), []).extend(float(v) for v in values)

    def ratio(self, group: str, metric: str) -> Ratio:
        return self._ratios.get((group, metric), Ratio())

    def samples(self, group: str, metric: str) -> list[float]:
        return list(self._samples.get((group, metric), ()))

    def groups(self) -> list[str]:
        seen = {g for g, _ in self._ratios} | {g for g, _ in self._samples}
        return sorted(seen)

    def metrics(self, group: str) -> list[str]:
        seen = {m for g, m in self._ratios if g == group} | {
            m for g, m in self._samples if g == group
        }
        return sorted(seen)
