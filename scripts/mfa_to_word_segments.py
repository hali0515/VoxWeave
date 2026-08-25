"""Convert Montreal Forced Aligner output into the read-only alignment-reference contract.

MFA is a *reference producer*, never a runtime dependency: this script reads
TextGrids that an offline `mfa align` run already wrote and emits one
``calibration/schemas/alignment-reference.schema.json`` document
(``kind = "mfa_words"``) per media file or window.

What it guarantees, and why each rule exists:

* **Nothing is silently dropped.**  Every interval of the word tier lands in
  exactly one bucket -- truth sample, excluded sample, or (with ``--range``)
  outside the window -- and ``included + excluded + outside_window == total``
  is asserted before anything is written.  ``spn``, dictionary OOV and
  unalignable words are written into ``segments`` with ``excluded: true`` and a
  reason, so a later report can state an excluded ratio instead of quietly
  measuring a smaller corpus.  Pure silence (``sil`` / ``sp`` / ``<eps>`` /
  empty label) cannot be a schema segment (``text`` has ``minLength: 1``) and is
  therefore accounted for in the conversion report only -- ``--report`` writes
  the full per-interval list, and the stdout summary always prints the counts.
* **Provenance is supplied, not guessed.**  A TextGrid embeds neither the MFA
  version nor the acoustic model or dictionary that produced it, so those come
  from flags (or ``--provenance``) and are mandatory.  Floating identifiers
  (``latest``, ``main``, ...) are refused: a truth set pinned to a moving model
  is not a truth set.  When a model/dictionary identifier happens to be a real
  file path its sha256 is computed automatically.
* **Nominal MFA uncertainty is metadata.**  ``--nominal-uncertainty-s``
  (default 0.02 s, the dataset-level word-onset mean absolute boundary error
  reported for English MFA 3.0) is stored in
  ``provenance.reference_uncertainty_s`` and *never* subtracted from an observed
  error anywhere.  A published dataset mean is not a per-boundary error bound.
* **Times stay raw.**  Only the shard offset (``global_offset_s``) and, with
  ``--range``, the window rebase are applied.  Text is stored verbatim;
  case folding, punctuation stripping and simplified/traditional conversion
  belong to the matcher at compare time, not to the truth file.

Usage::

    uv run python scripts/mfa_to_word_segments.py \\
        --mfa-output MFA_OUT \\
        --shard-map shard-map.json \\
        --language ja \\
        --reference-id ja-episode-01-mfa \\
        --mfa-version 3.0.6 \\
        --acoustic-model japanese_mfa \\
        --dictionary japanese_mfa \\
        --output truth/ja-episode-01.words.json

``--shard-map`` maps each TextGrid onto its offset in the transcription
timeline; either shape is accepted::

    {"shards": [{"id": "ep01-0007", "textgrid": "ep01-0007.TextGrid",
                 "global_offset_s": 615.0}]}
    {"ep01-0007": 615.0}

Exit codes follow the shared calibration contract: 0 = written, 2 = invalid
input / schema / tooling.  This converter has no quality gate of its own, so it
never exits 1.
"""

from __future__ import annotations

import argparse
import codecs
import datetime as dt
import importlib.util
import itertools
import math
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent


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

REFERENCE_SCHEMA = "alignment-reference"
SCHEMA_VERSION = 1
SOURCE_KIND = "mfa_words"

#: Times are stored at microsecond resolution -- far below any boundary MFA can
#: resolve, and enough to keep a rebased float from printing 17 digits.
TIME_DECIMALS = 6

#: ``provenance.created_by`` when neither a flag nor a provenance file names one.
DEFAULT_CREATED_BY = "scripts/mfa_to_word_segments.py"

#: Dataset-level word-onset mean absolute boundary error published for English
#: MFA 3.0 (TIMIT 19.93 ms / Buckeye 21.75 ms), rounded to 20 ms.  Stored as
#: metadata; see the module docstring for why it is never subtracted.
NOMINAL_UNCERTAINTY_S = 0.020

#: Word-tier labels that mean "no word here".  They are not truth and cannot be
#: schema segments (``text`` requires at least one character), so they are
#: accounted for in the conversion report instead.
SILENCE_LABELS = frozenset({"", "sil", "sp", "silence", "<eps>", "<sil>", "<p>"})

#: Labels MFA uses for speech it could not map to the dictionary.
SPN_LABELS = frozenset({"spn", "<spn>", "unk", "<unk>"})

#: Identifiers that move under the reader's feet; a pinned truth set cannot use them.
FLOATING_IDENTIFIERS = frozenset({"latest", "main", "master", "head", "dev", "nightly"})

#: Provenance keys the schema accepts.  ``source_digest`` is owned by this
#: script (it digests the actual TextGrids), so it may not be supplied by hand.
PROVENANCE_FIELDS = (
    "created_by",
    "created_at",
    "tool_version",
    "command",
    "acoustic_model",
    "acoustic_model_sha256",
    "dictionary",
    "dictionary_sha256",
    "g2p_model",
    "g2p_model_sha256",
    "reference_uncertainty_s",
    "license",
    "annotators",
)
REQUIRED_PROVENANCE = ("created_by", "tool_version", "acoustic_model", "dictionary")

_REFERENCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# TextGrid parsing (long and short Praat formats, no third-party dependency)
# --------------------------------------------------------------------------- #

# `item []:`, `item [1]:`, `intervals [7]:`, `points [2]:` -- structure, not data.
_STRUCT_RE = re.compile(
    r"^(item|intervals|points)\s*\[\s*\d*\s*\]\s*:?\s*$", re.IGNORECASE
)
_FILE_TYPE_RE = re.compile(r'^File type\s*=\s*"ooTextFile"', re.IGNORECASE)
_OBJECT_CLASS_RE = re.compile(r'^Object class\s*=\s*"TextGrid"', re.IGNORECASE)

INTERVAL_TIER = "IntervalTier"
POINT_TIER_CLASSES = frozenset({"TextTier", "PointTier"})


@dataclass(frozen=True)
class Interval:
    """One labelled interval of an ``IntervalTier``, in the TextGrid's own timeline."""

    xmin: float
    xmax: float
    text: str


@dataclass(frozen=True)
class Tier:
    kind: str
    name: str
    xmin: float
    xmax: float
    intervals: tuple[Interval, ...]


@dataclass(frozen=True)
class TextGrid:
    xmin: float
    xmax: float
    tiers: tuple[Tier, ...]

    def tier_names(self) -> list[str]:
        return [t.name for t in self.tiers]


def _decode_textgrid(raw: bytes, path: Path) -> str:
    """Decode Praat's three encodings in the wild: UTF-16 (BOM), UTF-8 (BOM) and UTF-8."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise cc.CalibrationError(
                f"{path}: file starts with a UTF-16 BOM but is not valid UTF-16",
                [str(exc)],
            ) from None
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise cc.CalibrationError(
            f"{path}: TextGrid is neither UTF-8 nor BOM-marked UTF-16", [str(exc)]
        ) from None


def _quote_closed(text: str) -> bool:
    """True when ``text`` is a complete Praat string literal (``""`` escapes a quote)."""
    if not text.startswith('"'):
        return True
    i = 1
    while i < len(text):
        if text[i] == '"':
            if i + 1 < len(text) and text[i + 1] == '"':
                i += 2
                continue
            return True
        i += 1
    return False


def _unquote(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


def _value_stream(text: str, path: Path) -> list[str]:
    """Reduce either TextGrid dialect to the one value sequence both encode.

    Long and short formats differ only in decoration: the long one prefixes
    every value with ``key = `` and interleaves ``item [k]:`` headers, the short
    one writes bare values.  Stripping the decoration leaves the identical
    ordered stream, so a single parser reads both instead of two parsers
    drifting apart.
    """
    lines = text.splitlines()
    header = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()][:2]
    if (
        len(header) < 2
        or not _FILE_TYPE_RE.match(header[0][1])
        or not _OBJECT_CLASS_RE.match(header[1][1])
    ):
        raise cc.CalibrationError(
            f"{path} is not a Praat TextGrid",
            ['expected `File type = "ooTextFile"` then `Object class = "TextGrid"`'],
        )

    out: list[str] = []
    i = header[1][0] + 1
    while i < len(lines):
        raw = lines[i]
        i += 1
        value = raw.strip()
        if not value:
            continue
        if value.lower().startswith("tiers?"):
            value = value[len("tiers?") :].strip()
        elif "=" in value and not value.startswith('"'):
            # `xmin = 0`, `text = "a = b"`, `intervals: size = 3`
            value = value.split("=", 1)[1].strip()
        elif _STRUCT_RE.match(value):
            continue
        if not value:
            continue
        if value.startswith('"') and not _quote_closed(value):
            # Praat allows a newline inside a quoted label; keep reading.
            buf = [value]
            while i < len(lines) and not _quote_closed("\n".join(buf)):
                buf.append(lines[i])
                i += 1
            value = "\n".join(buf)
            if not _quote_closed(value):
                raise cc.CalibrationError(
                    f"{path}: unterminated quoted label near line {i}"
                )
        out.append(value)
    return out


@dataclass
class _Values:
    """Cursor over the value stream, reporting position-aware exit-2 errors."""

    path: Path
    items: list[str]
    pos: int = 0

    def take(self, what: str) -> str:
        if self.pos >= len(self.items):
            raise cc.CalibrationError(
                f"{self.path}: TextGrid ends before {what} (truncated file?)"
            )
        value = self.items[self.pos]
        self.pos += 1
        return value

    def peek(self) -> str | None:
        return self.items[self.pos] if self.pos < len(self.items) else None

    def remaining(self) -> int:
        return len(self.items) - self.pos

    def number(self, what: str) -> float:
        raw = self.take(what)
        try:
            value = float(_unquote(raw))
        except ValueError:
            raise cc.CalibrationError(
                f"{self.path}: expected a number for {what}, got {raw!r}"
            ) from None
        if not math.isfinite(value):
            raise cc.CalibrationError(
                f"{self.path}: {what} is not finite ({raw!r})"
            ) from None
        return value

    def integer(self, what: str) -> int:
        raw = _unquote(self.take(what))
        try:
            number = float(raw)
        except ValueError:
            raise cc.CalibrationError(
                f"{self.path}: expected an integer for {what}, got {raw!r}"
            ) from None
        value = int(number)
        if value != number:
            raise cc.CalibrationError(
                f"{self.path}: expected an integer for {what}, got {raw!r}"
            )
        if value < 0:
            raise cc.CalibrationError(f"{self.path}: {what} is negative ({value})")
        return value

    def string(self, what: str) -> str:
        return _unquote(self.take(what))


def _parse_tier(values: _Values, index: int) -> Tier:
    kind = values.string(f"tier {index} class")
    name = values.string(f"tier {index} name")
    xmin = values.number(f"tier {index} xmin")
    xmax = values.number(f"tier {index} xmax")
    count = values.integer(f"tier {index} entry count")
    intervals: list[Interval] = []
    if kind == INTERVAL_TIER:
        for k in range(1, count + 1):
            start = values.number(f"tier {index} interval {k} xmin")
            end = values.number(f"tier {index} interval {k} xmax")
            label = values.string(f"tier {index} interval {k} text")
            intervals.append(Interval(start, end, label))
    elif kind in POINT_TIER_CLASSES:
        # Point tiers carry no word boundaries; consume them so the following
        # tiers stay aligned with the value stream.
        for k in range(1, count + 1):
            values.number(f"tier {index} point {k} time")
            values.string(f"tier {index} point {k} mark")
    else:
        raise cc.CalibrationError(
            f"{values.path}: unknown tier class {kind!r} in tier {index}",
            [f"expected {INTERVAL_TIER} or one of {sorted(POINT_TIER_CLASSES)}"],
        )
    return Tier(kind, name, xmin, xmax, tuple(intervals))


def parse_textgrid(path: str | Path) -> TextGrid:
    """Parse a Praat TextGrid (long or short format); any defect raises exit-2."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise cc.CalibrationError(f"cannot read {p}: {exc}") from None
    values = _Values(p, _value_stream(_decode_textgrid(raw, p), p))
    xmin = values.number("the grid xmin")
    xmax = values.number("the grid xmax")
    flag = values.peek()
    if flag is not None and flag.startswith("<"):
        values.pos += 1
        if flag.strip("<>").lower() != "exists":
            raise cc.CalibrationError(f"{p}: TextGrid declares no tiers ({flag})")
    size = values.integer("the tier count")
    tiers = [_parse_tier(values, i) for i in range(1, size + 1)]
    if values.remaining():
        raise cc.CalibrationError(
            f"{p}: {values.remaining()} unread value(s) after {size} tier(s)",
            ["a declared size does not match the body -- the file is malformed"],
        )
    if not tiers:
        raise cc.CalibrationError(f"{p}: TextGrid contains no tiers")
    return TextGrid(xmin, xmax, tuple(tiers))


def select_tiers(grid: TextGrid, wanted: str) -> list[Tier]:
    """Interval tiers named ``wanted``, including MFA's ``<speaker> - words`` form."""
    target = wanted.strip().casefold()
    out: list[Tier] = []
    for tier in grid.tiers:
        if tier.kind != INTERVAL_TIER:
            continue
        name = tier.name.strip().casefold()
        if name == target or name.endswith((f"- {target}", f"-{target}")):
            out.append(tier)
    return out


# --------------------------------------------------------------------------- #
# Window arithmetic (mirrors capture_scenario.Window; kept local so the two
# scripts stay independently loadable)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Window:
    """Clip window in reference time. ``end=None`` means "to the end of the file"."""

    start: float = 0.0
    end: float | None = None

    @property
    def bounded(self) -> bool:
        return self.end is not None

    def rebase(self, t: float) -> float:
        return round(max(0.0, float(t) - self.start), TIME_DECIMALS)

    def contains(self, start: float, end: float) -> bool:
        """True when ``[start, end]`` lies wholly inside the window.

        A word straddling an edge is dropped whole: half a word has no boundary
        worth measuring, and clipping one would invent a boundary MFA never saw.
        """
        if start < self.start - _EPS:
            return False
        return self.end is None or end <= self.end + _EPS


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
# Shards
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Shard:
    """One aligned TextGrid plus where it sits in the transcription timeline."""

    id: str
    path: Path
    offset_s: float


def discover_textgrids(root: Path) -> list[Path]:
    """Every ``*.TextGrid`` under ``root`` (or ``root`` itself when it is one file)."""
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise cc.CalibrationError(f"--mfa-output not found: {root}")
    found = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.casefold() == ".textgrid"
    )
    if not found:
        raise cc.CalibrationError(f"no *.TextGrid under {root}")
    return found


def _shard_entries(document: Any, path: Path) -> dict[str, tuple[str, float]]:
    """Normalize either shard-map shape to ``{match key: (shard id, offset)}``."""
    if not isinstance(document, Mapping):
        raise cc.CalibrationError(f"{path} is not a JSON object")
    raw_shards = document.get("shards")
    entries: list[Mapping[str, Any]] = []
    if raw_shards is None:
        for key, value in document.items():
            if key in ("schema_version", "media", "notes"):
                continue
            if isinstance(value, Mapping):
                entries.append({"id": key, **value})
            else:
                entries.append({"id": key, "global_offset_s": value})
    elif isinstance(raw_shards, list):
        for item in raw_shards:
            if not isinstance(item, Mapping):
                raise cc.CalibrationError(
                    f"{path}: every `shards` entry must be an object"
                )
            entries.append(item)
    else:
        raise cc.CalibrationError(f"{path}: `shards` must be an array")

    if not entries:
        raise cc.CalibrationError(f"{path} declares no shards")

    lookup: dict[str, tuple[str, float]] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        shard_id = entry.get("id") or entry.get("textgrid")
        if not isinstance(shard_id, str) or not shard_id.strip():
            raise cc.CalibrationError(f"{path}: a shard entry has no usable `id`")
        shard_id = shard_id.strip()
        if shard_id in seen_ids:
            raise cc.CalibrationError(f"{path}: duplicate shard id {shard_id!r}")
        seen_ids.add(shard_id)
        offset = entry.get("global_offset_s", entry.get("offset_s"))
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise cc.CalibrationError(
                f"{path}: shard {shard_id!r} has no numeric `global_offset_s`"
            )
        offset = float(offset)
        if not math.isfinite(offset) or offset < 0:
            raise cc.CalibrationError(
                f"{path}: shard {shard_id!r} offset must be finite and >= 0, got {offset!r}"
            )
        keys = {shard_id, Path(shard_id).name, Path(shard_id).stem}
        textgrid = entry.get("textgrid")
        if isinstance(textgrid, str) and textgrid.strip():
            name = textgrid.strip()
            keys |= {name, Path(name).name, Path(name).stem}
        for key in keys:
            previous = lookup.get(key)
            if previous is not None and previous[0] != shard_id:
                raise cc.CalibrationError(
                    f"{path}: {key!r} matches both {previous[0]!r} and {shard_id!r}"
                )
            lookup[key] = (shard_id, offset)
    return lookup


def resolve_shards(
    files: Sequence[Path], root: Path, shard_map_path: Path | None
) -> list[Shard]:
    """Pair every TextGrid with its offset; an unmatched file on either side is fatal.

    Defaulting a missing offset to 0 would stack every shard on top of the first
    one and produce a truth file that looks plausible and is entirely wrong, so
    an incomplete map fails instead.
    """
    if shard_map_path is None:
        if len(files) == 1:
            return [Shard(files[0].stem, files[0], 0.0)]
        raise cc.CalibrationError(
            f"{len(files)} TextGrids need --shard-map to place them on one timeline",
            ["a shard without a declared global_offset_s cannot be positioned"],
        )

    lookup = _shard_entries(cc.read_json(shard_map_path), Path(shard_map_path))
    shards: list[Shard] = []
    unmatched: list[str] = []
    used: set[str] = set()
    for path in files:
        candidates = [path.name, path.stem]
        try:
            candidates.insert(0, path.relative_to(root).as_posix())
        except ValueError:
            pass
        hit = next((lookup[key] for key in candidates if key in lookup), None)
        if hit is None:
            unmatched.append(str(path))
            continue
        shard_id, offset = hit
        used.add(shard_id)
        shards.append(Shard(shard_id, path, offset))
    leftovers = sorted({sid for sid, _ in lookup.values()} - used)
    if unmatched or leftovers:
        raise cc.CalibrationError(
            f"{shard_map_path} does not describe the MFA output exactly",
            [f"TextGrid with no shard entry: {p}" for p in unmatched]
            + [f"shard entry with no TextGrid: {s}" for s in leftovers],
        )
    shards.sort(key=lambda s: (s.offset_s, s.id))
    return shards


# --------------------------------------------------------------------------- #
# Interval classification
# --------------------------------------------------------------------------- #

#: Reason codes; the first three never become schema segments (see docstring).
REASON_SILENCE = "silence"
REASON_ZERO_DURATION = "zero_duration"
REASON_NON_FINITE = "non_finite"
REASON_SPN = "spn"
REASON_OOV_PHONES = "oov_spn_phones"
REASON_OOV_LIST = "oov_dictionary"

#: Excluded reasons that are recorded in the report only.
REPORT_ONLY_REASONS = frozenset(
    {REASON_SILENCE, REASON_ZERO_DURATION, REASON_NON_FINITE}
)


def load_oov_words(path: str | Path) -> frozenset[str]:
    """Read an ``mfa find_oovs`` list: one word per line, optional count column."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise cc.CalibrationError(f"cannot read {p}: {exc}") from None
    words: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        words.add(stripped.split("\t")[0].strip().casefold())
    if not words:
        raise cc.CalibrationError(f"{p} lists no OOV words")
    return frozenset(words)


def _lexical_phones(tiers: Sequence[Tier], start: float, end: float) -> list[str]:
    """Phone labels whose midpoint falls inside ``[start, end]``, silence removed."""
    out: list[str] = []
    for tier in tiers:
        for iv in tier.intervals:
            mid = (iv.xmin + iv.xmax) / 2.0
            if start - _EPS <= mid <= end + _EPS:
                label = iv.text.strip().casefold()
                if label and label not in SILENCE_LABELS:
                    out.append(label)
    return out


def classify_interval(
    interval: Interval,
    *,
    phone_tiers: Sequence[Tier] = (),
    oov_words: frozenset[str] = frozenset(),
) -> tuple[bool, str | None]:
    """Return ``(is_truth, exclude_reason)`` for one word-tier interval.

    Order matters: a degenerate span is unusable regardless of its label, and a
    silence label is silence even when the phone tier disagrees.
    """
    label = interval.text.strip()
    folded = label.casefold()
    if not (math.isfinite(interval.xmin) and math.isfinite(interval.xmax)):
        return False, REASON_NON_FINITE
    if folded in SILENCE_LABELS:
        return False, REASON_SILENCE
    if interval.xmax - interval.xmin <= 0:
        return False, REASON_ZERO_DURATION
    if folded in SPN_LABELS:
        return False, REASON_SPN
    if folded in oov_words:
        return False, REASON_OOV_LIST
    if phone_tiers:
        phones = _lexical_phones(phone_tiers, interval.xmin, interval.xmax)
        if phones and all(p in SPN_LABELS for p in phones):
            return False, REASON_OOV_PHONES
    return True, None


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def _normalize_sha(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    folded = str(value).strip().casefold()
    if not _SHA256_RE.match(folded):
        raise cc.CalibrationError(
            f"{label} must be 64 lowercase hex digits, got {value!r}"
        )
    return folded


def _reject_floating(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise cc.CalibrationError(f"{label} must not be empty")
    if text.casefold() in FLOATING_IDENTIFIERS:
        raise cc.CalibrationError(
            f"{label} must pin an exact version, not {value!r}",
            ["a truth set produced by a moving model cannot be reproduced"],
        )
    return text


def _artifact_sha(
    identifier: str | None, explicit: str | None, label: str
) -> str | None:
    """Explicit sha wins; otherwise hash the identifier when it names a real file."""
    sha = _normalize_sha(explicit, f"--{label}-sha256")
    if sha is not None or identifier is None:
        return sha
    candidate = Path(identifier).expanduser()
    if candidate.is_file():
        return cc.sha256_file(candidate)
    return None


def load_provenance_file(path: str | Path) -> dict[str, Any]:
    """Read a provenance side file, refusing keys the schema (or this tool) will not take."""
    document = cc.read_json(path)
    if not isinstance(document, Mapping):
        raise cc.CalibrationError(f"{path} is not a JSON object")
    unknown = sorted(set(document) - set(PROVENANCE_FIELDS))
    if unknown:
        details = [f"unknown key: {key}" for key in unknown]
        if "source_digest" in unknown:
            details.append(
                "source_digest is computed from the TextGrids and cannot be supplied"
            )
        raise cc.CalibrationError(f"{path} carries keys the schema rejects", details)
    return dict(document)


def build_provenance(
    args: argparse.Namespace, *, source_digest: str, argv: Sequence[str]
) -> dict[str, Any]:
    """Merge ``--provenance`` with the flags (flags win) and enforce what MFA cannot embed."""
    merged: dict[str, Any] = {}
    if args.provenance:
        merged.update(load_provenance_file(args.provenance))

    overrides = {
        "created_by": args.created_by,
        "created_at": args.created_at,
        "tool_version": args.mfa_version,
        "command": args.mfa_command,
        "acoustic_model": args.acoustic_model,
        "acoustic_model_sha256": args.acoustic_model_sha256,
        "dictionary": args.dictionary,
        "dictionary_sha256": args.dictionary_sha256,
        "g2p_model": args.g2p_model,
        "g2p_model_sha256": args.g2p_model_sha256,
        "license": args.license,
    }
    # Only values the caller actually typed override the side file: an argparse
    # default is not a decision, and letting one win would silently discard a
    # provenance file the operator wrote by hand.
    merged.update({k: v for k, v in overrides.items() if v is not None})
    if args.annotators is not None:
        merged["annotators"] = args.annotators
    if args.nominal_uncertainty_s is not None:
        merged["reference_uncertainty_s"] = args.nominal_uncertainty_s
    merged.setdefault("created_by", DEFAULT_CREATED_BY)

    missing = [
        key for key in REQUIRED_PROVENANCE if not str(merged.get(key) or "").strip()
    ]
    if missing:
        raise cc.CalibrationError(
            "MFA provenance is incomplete -- a TextGrid embeds none of it",
            [
                f"missing {key} (pass --{key.replace('_', '-')} or --provenance)"
                for key in missing
            ],
        )
    for key in ("tool_version", "acoustic_model", "dictionary", "g2p_model"):
        if merged.get(key) is not None:
            merged[key] = _reject_floating(merged[key], key)

    for key in ("acoustic_model", "dictionary", "g2p_model"):
        merged[f"{key}_sha256"] = _artifact_sha(
            merged.get(key), merged.get(f"{key}_sha256"), key.replace("_", "-")
        )

    uncertainty = float(merged.get("reference_uncertainty_s", NOMINAL_UNCERTAINTY_S))
    if not math.isfinite(uncertainty) or uncertainty < 0:
        raise cc.CalibrationError(
            f"reference_uncertainty_s must be finite and >= 0, got {uncertainty!r}"
        )
    merged["reference_uncertainty_s"] = uncertainty

    annotators = int(merged.get("annotators", 0))
    if annotators < 0:
        raise cc.CalibrationError(f"annotators must be >= 0, got {annotators}")
    merged["annotators"] = annotators

    merged.setdefault("created_at", _now_iso())
    merged.setdefault("command", shlex.join(list(argv)))
    merged["source_digest"] = source_digest
    return merged


def _now_iso() -> str:
    return (
        dt.datetime.now(dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Candidate:
    """One word-tier interval already placed on the shard-offset timeline."""

    shard: str
    tier: str
    index: int
    start: float
    end: float
    label: str
    truth: bool
    reason: str | None


def _round(value: float) -> float:
    return round(float(value), TIME_DECIMALS)


def _read_shard(
    shard: Shard, *, tier_name: str, phone_tier_name: str, oov_words: frozenset[str]
) -> tuple[list[_Candidate], int]:
    grid = parse_textgrid(shard.path)
    word_tiers = select_tiers(grid, tier_name)
    if not word_tiers:
        raise cc.CalibrationError(
            f"{shard.path} has no {tier_name!r} interval tier",
            [f"tiers present: {', '.join(grid.tier_names()) or '(none)'}"],
        )
    phone_tiers = select_tiers(grid, phone_tier_name) if phone_tier_name else []

    out: list[_Candidate] = []
    for tier in word_tiers:
        for index, interval in enumerate(tier.intervals, start=1):
            truth, reason = classify_interval(
                interval, phone_tiers=phone_tiers, oov_words=oov_words
            )
            out.append(
                _Candidate(
                    shard=shard.id,
                    tier=tier.name,
                    index=index,
                    start=interval.xmin + shard.offset_s,
                    end=interval.xmax + shard.offset_s,
                    label=interval.text.strip(),
                    truth=truth,
                    reason=reason,
                )
            )
    return out, len(word_tiers)


def _check_speaker_overlap(candidates: Sequence[_Candidate], shard: Shard) -> None:
    """Two word tiers in one shard must not talk over each other.

    The reference schema has no speaker field, so overlapping tiers cannot be
    represented; say so instead of writing a file the loader will reject.
    """
    ordered = sorted(
        (c for c in candidates if c.reason not in REPORT_ONLY_REASONS),
        key=lambda c: (c.start, c.end),
    )
    for previous, current in itertools.pairwise(ordered):
        if current.start < previous.end - _EPS and previous.tier != current.tier:
            raise cc.CalibrationError(
                f"{shard.path}: word tiers {previous.tier!r} and {current.tier!r} overlap",
                [
                    (
                        f"{previous.label!r} [{previous.start:.3f},{previous.end:.3f}]"
                        f" vs {current.label!r} [{current.start:.3f},{current.end:.3f}]"
                    ),
                    "select a single speaker tier with --tier",
                ],
            )


def _check_shard_spans(spans: Mapping[str, tuple[float, float]]) -> None:
    """Adjacent shards must not overlap once their offsets are applied."""
    ordered = sorted(spans.items(), key=lambda item: item[1])
    for (left_id, left), (right_id, right) in itertools.pairwise(ordered):
        if right[0] < left[1] - _EPS:
            raise cc.CalibrationError(
                f"shards {left_id!r} and {right_id!r} overlap after applying their offsets",
                [
                    f"{left_id}: [{left[0]:.3f}, {left[1]:.3f}]",
                    f"{right_id}: [{right[0]:.3f}, {right[1]:.3f}]",
                    "check global_offset_s -- shard padding must not contain aligned words",
                ],
            )


def _validate_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    offset_s: float,
    media_duration_s: float | None,
) -> None:
    """The loader's own checks, enforced where the file is produced."""
    previous: tuple[float, float, str] | None = None
    last_end: dict[str, float] = {}
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        seg_id = str(segment["id"])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise cc.CalibrationError(f"segment {seg_id} has a non-finite time")
        if end <= start:
            raise cc.CalibrationError(
                f"segment {seg_id} has end <= start ({start} .. {end})"
            )
        current = (start, end, seg_id)
        if previous is not None and current < previous:
            raise cc.CalibrationError(
                f"segments are not monotonic by (start, end, id) at {seg_id}"
            )
        previous = current
        utterance = str(segment.get("utterance_id") or "")
        if utterance:
            if start < last_end.get(utterance, -math.inf) - _EPS:
                raise cc.CalibrationError(
                    f"segments overlap inside utterance {utterance!r} at {seg_id}"
                )
            last_end[utterance] = max(last_end.get(utterance, -math.inf), end)
        if media_duration_s is not None and (
            start + offset_s < -_EPS or end + offset_s > media_duration_s + _EPS
        ):
            raise cc.CalibrationError(
                f"segment {seg_id} falls outside the media after offset_s={offset_s}",
                [
                    (
                        f"segment [{start}, {end}] + {offset_s} exceeds "
                        f"media_duration_s={media_duration_s}"
                    )
                ],
            )


def convert(
    *,
    reference_id: str,
    language: str,
    shards: Sequence[Shard],
    root: Path,
    window: Window | None = None,
    tier_name: str = "words",
    phone_tier_name: str = "phones",
    oov_words: frozenset[str] = frozenset(),
    offset_s: float = 0.0,
    media_duration_s: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the reference document and the conversion report from parsed shards.

    Returns ``(reference, report)``. The report is the accounting document: every
    source interval appears in exactly one bucket and the identity
    ``included + excluded + outside_window == total`` is asserted here, so no
    interval can go missing between the TextGrid and the truth file.
    """
    window = window or Window()
    if not _REFERENCE_ID_RE.match(reference_id):
        raise cc.CalibrationError(
            f"--reference-id {reference_id!r} is not a manifest-safe id",
            [
                "allowed: lowercase letters, digits, '.', '_', '-'; at least 2 characters"
            ],
        )
    iso = cc.require_calibration_language(language)
    if not math.isfinite(offset_s):
        raise cc.CalibrationError(f"--media-offset must be finite, got {offset_s!r}")

    candidates: list[_Candidate] = []
    shard_rows: list[dict[str, Any]] = []
    spans: dict[str, tuple[float, float]] = {}
    for shard in shards:
        rows, tier_count = _read_shard(
            shard,
            tier_name=tier_name,
            phone_tier_name=phone_tier_name,
            oov_words=oov_words,
        )
        _check_speaker_overlap(rows, shard)
        real = [r for r in rows if r.reason not in REPORT_ONLY_REASONS]
        if real:
            spans[shard.id] = (
                min(r.start for r in real),
                max(r.end for r in real),
            )
        try:
            name = shard.path.relative_to(root).as_posix()
        except ValueError:
            name = shard.path.name
        shard_rows.append(
            {
                "id": shard.id,
                "textgrid": name,
                "sha256": cc.sha256_file(shard.path),
                "global_offset_s": _round(shard.offset_s),
                "word_tiers": tier_count,
                "intervals": len(rows),
            }
        )
        candidates.extend(rows)
    _check_shard_spans(spans)

    total = len(candidates)
    outside = 0
    kept: list[_Candidate] = []
    for candidate in candidates:
        if not window.contains(candidate.start, candidate.end):
            outside += 1
            continue
        kept.append(candidate)

    kept.sort(key=lambda c: (c.start, c.end, c.label, c.shard, c.index))
    segments: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    by_reason: dict[str, int] = {}
    included = 0
    excluded = 0
    for candidate in kept:
        start, end = window.rebase(candidate.start), window.rebase(candidate.end)
        if candidate.truth:
            included += 1
            segments.append(
                {
                    "id": f"w{len(segments):06d}",
                    "text": candidate.label,
                    "start": start,
                    "end": end,
                    "utterance_id": candidate.shard,
                }
            )
            continue
        excluded += 1
        reason = candidate.reason or "unknown"
        by_reason[reason] = by_reason.get(reason, 0) + 1
        excluded_rows.append(
            {
                "shard": candidate.shard,
                "tier": candidate.tier,
                "interval": candidate.index,
                "start": start,
                "end": end,
                "label": candidate.label,
                "reason": reason,
            }
        )
        if reason in REPORT_ONLY_REASONS:
            # No schema segment exists for these: `text` needs at least one
            # character and `end > start` must hold. The report is their record.
            continue
        segments.append(
            {
                "id": f"w{len(segments):06d}",
                "text": candidate.label,
                "start": start,
                "end": end,
                "utterance_id": candidate.shard,
                "excluded": True,
                "exclude_reason": reason,
            }
        )

    if included + excluded + outside != total:
        raise cc.CalibrationError(  # pragma: no cover - accounting invariant
            f"interval accounting lost data: {included} + {excluded} + {outside} != {total}"
        )
    if not segments:
        raise cc.CalibrationError(
            f"no usable word interval for reference {reference_id!r}",
            [
                (
                    f"{total} source interval(s): {included} truth, "
                    f"{excluded} excluded, {outside} outside the window"
                )
            ],
        )
    if included == 0:
        raise cc.CalibrationError(
            f"reference {reference_id!r} would carry no truth sample at all",
            [f"every one of {excluded} usable interval(s) was excluded"],
        )

    duration = media_duration_s
    if duration is None and window.bounded:
        assert window.end is not None
        duration = _round(window.end - window.start)
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise cc.CalibrationError(
            f"--media-duration must be finite and >= 0, got {duration!r}"
        )
    _validate_segments(segments, offset_s=offset_s, media_duration_s=duration)

    reference: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": reference_id,
        "language": iso,
        "kind": SOURCE_KIND,
        "timebase": "seconds",
        "media_duration_s": duration,
        "offset_s": _round(offset_s),
        "provenance": dict(provenance or {}),
        "segments": segments,
    }

    # Two different questions, so two rates, both keeping numerator and
    # denominator (calib_common.Ratio): `excluded_ratio` covers every interval
    # and is dominated by inter-word silence, while `lexical_excluded_ratio` is
    # the one a report should quote -- the share of real words (spn / OOV /
    # unalignable) that this reference cannot vouch for.
    lexical_excluded = sum(
        count
        for reason, count in by_reason.items()
        if reason not in REPORT_ONLY_REASONS
    )
    report: dict[str, Any] = {
        "reference_id": reference_id,
        "language": iso,
        "kind": SOURCE_KIND,
        "tier": tier_name,
        "window": {
            "start_s": _round(window.start),
            "end_s": _round(window.end) if window.end is not None else None,
        },
        "shards": shard_rows,
        "intervals": {
            "total": total,
            "included": included,
            "excluded": excluded,
            "outside_window": outside,
        },
        "excluded_ratio": cc.Ratio(excluded, included + excluded).to_dict(),
        "lexical_excluded_ratio": cc.Ratio(
            lexical_excluded, included + lexical_excluded
        ).to_dict(),
        "excluded_by_reason": dict(sorted(by_reason.items())),
        "excluded_intervals": excluded_rows,
        "reference_uncertainty_s": (provenance or {}).get(
            "reference_uncertainty_s", NOMINAL_UNCERTAINTY_S
        ),
        "uncertainty_note": (
            "nominal MFA boundary uncertainty is metadata about the reference; "
            "it is never subtracted from an observed error"
        ),
    }
    return reference, report


def source_digest(shards: Sequence[Shard], root: Path) -> str:
    """Digest of what was converted: every TextGrid's bytes plus its declared offset."""
    payload: dict[str, Any] = {}
    for shard in shards:
        try:
            name = shard.path.relative_to(root).as_posix()
        except ValueError:
            name = shard.path.name
        payload[shard.id] = {
            "textgrid": name,
            "sha256": cc.sha256_file(shard.path),
            "global_offset_s": _round(shard.offset_s),
        }
    return cc.canonical_digest(payload)


def write_reference(
    path: Path, reference: Mapping[str, Any], *, force: bool = False
) -> Path:
    """Validate against the shared schema and write atomically; never clobber silently."""
    path = Path(path)
    if path.exists() and not force:
        raise cc.CalibrationError(
            f"{path} already exists; pass --force to overwrite it",
            ["a truth file is a baseline -- silent replacement hides drift"],
        )
    cc.validate_or_exit2(reference, REFERENCE_SCHEMA, label=str(path))
    return cc.write_json(path, reference)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert MFA TextGrid word tiers into an alignment reference.",
    )
    ap.add_argument(
        "--mfa-output",
        required=True,
        metavar="PATH",
        help="directory of aligned TextGrids (searched recursively) or a single .TextGrid",
    )
    ap.add_argument(
        "--shard-map",
        metavar="PATH",
        default=None,
        help="JSON placing every TextGrid on the transcription timeline; required"
        " for more than one shard",
    )
    ap.add_argument("--language", required=True, help="en, ja or zh (BCP-47 accepted)")
    ap.add_argument(
        "--reference-id",
        required=True,
        metavar="ID",
        help="reference id used by the alignment manifest (e.g. ja-episode-01-mfa)",
    )
    ap.add_argument(
        "--output", required=True, metavar="PATH", help="reference JSON to write"
    )
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing reference"
    )
    ap.add_argument(
        "--report",
        metavar="PATH",
        default=None,
        help="write the interval accounting report (counts, reasons, excluded list)",
    )

    shape = ap.add_argument_group("tiers and window")
    shape.add_argument(
        "--tier",
        default="words",
        help="word tier name; MFA's `<speaker> - words` form matches too (default: words)",
    )
    shape.add_argument(
        "--phone-tier",
        default="phones",
        help="phone tier used to spot OOV words realized as spn (default: phones;"
        " empty string disables the check)",
    )
    shape.add_argument(
        "--oov-list",
        metavar="PATH",
        default=None,
        help="`mfa find_oovs` output; listed words are excluded with a reason",
    )
    shape.add_argument(
        "--range",
        metavar="START:END",
        default=None,
        help="seconds; keep only words wholly inside the window and rebase by -START",
    )
    shape.add_argument(
        "--media-offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="written as offset_s: the media-vs-WAV zero point, applied at compare"
        " time (default: 0)",
    )
    shape.add_argument(
        "--media-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="media duration; every segment must stay inside it once offset_s is applied",
    )

    prov = ap.add_argument_group("provenance (a TextGrid embeds none of this)")
    prov.add_argument(
        "--provenance",
        metavar="PATH",
        default=None,
        help="JSON with any provenance field; the flags below win over it",
    )
    prov.add_argument(
        "--created-by",
        default=None,
        help=f"who produced this reference (default: {DEFAULT_CREATED_BY})",
    )
    prov.add_argument(
        "--created-at", default=None, help="ISO-8601 timestamp (default: now, UTC)"
    )
    prov.add_argument(
        "--mfa-version", default=None, help="exact MFA patch version, e.g. 3.0.6"
    )
    prov.add_argument(
        "--mfa-command",
        default=None,
        help="the `mfa align` command that produced the TextGrids",
    )
    prov.add_argument(
        "--acoustic-model", default=None, help="acoustic model name or path"
    )
    prov.add_argument("--acoustic-model-sha256", default=None)
    prov.add_argument(
        "--dictionary", default=None, help="pronunciation dictionary name or path"
    )
    prov.add_argument("--dictionary-sha256", default=None)
    prov.add_argument(
        "--g2p-model", default=None, help="G2P model name or path (OOV only)"
    )
    prov.add_argument("--g2p-model-sha256", default=None)
    prov.add_argument("--license", default=None, help="license of the underlying media")
    prov.add_argument(
        "--annotators",
        type=int,
        default=None,
        help="humans who checked the boundaries (default: 0 -- MFA output is unreviewed)",
    )
    prov.add_argument(
        "--nominal-uncertainty-s",
        type=float,
        default=None,
        metavar="SECONDS",
        help=f"nominal word-boundary uncertainty recorded as metadata (default:"
        f" {NOMINAL_UNCERTAINTY_S}); never subtracted from observed errors",
    )
    return ap


def _summarize(
    out: Path, reference: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    counts = report["intervals"]
    truth = sum(1 for s in reference["segments"] if not s.get("excluded"))
    print(f"[mfa] wrote {out}")
    print(
        f"      language={reference['language']} shards={len(report['shards'])} "
        f"segments={len(reference['segments'])} (truth {truth})"
    )
    print(
        f"      intervals: total={counts['total']} included={counts['included']} "
        f"excluded={counts['excluded']} outside_window={counts['outside_window']}"
    )
    if report["excluded_by_reason"]:
        breakdown = " ".join(
            f"{k}={v}" for k, v in report["excluded_by_reason"].items()
        )
        print(f"      excluded by reason: {breakdown}")
    lexical = report["lexical_excluded_ratio"]
    print(
        f"      lexical exclusions: {lexical['bad']}/{lexical['eligible']} words"
        " (spn / OOV / unalignable)"
    )
    print(
        f"      nominal reference uncertainty: {report['reference_uncertainty_s']}s "
        "(metadata only -- never subtracted from observed errors)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    root = Path(args.mfa_output)
    files = discover_textgrids(root)
    search_root = root if root.is_dir() else root.parent
    shards = resolve_shards(
        files, search_root, Path(args.shard_map) if args.shard_map else None
    )

    window = parse_range(args.range) if args.range else Window()
    oov_words = load_oov_words(args.oov_list) if args.oov_list else frozenset()
    provenance = build_provenance(
        args,
        source_digest=source_digest(shards, search_root),
        argv=[Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
    )

    reference, report = convert(
        reference_id=args.reference_id,
        language=args.language,
        shards=shards,
        root=search_root,
        window=window,
        tier_name=args.tier,
        phone_tier_name=args.phone_tier,
        oov_words=oov_words,
        offset_s=float(args.media_offset),
        media_duration_s=args.media_duration,
        provenance=provenance,
    )

    out = write_reference(Path(args.output), reference, force=args.force)
    if args.report:
        cc.write_json(Path(args.report), report)
    _summarize(out, reference, report)
    return cc.EXIT_OK


if __name__ == "__main__":
    cc.run_cli(main)
