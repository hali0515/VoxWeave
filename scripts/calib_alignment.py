"""The alignment ruler: how close are voxweave's timestamps to same-language truth.

This harness answers one question per *lane*, where a lane is a
``(source_kind, language)`` pair, and it never lets two lanes share a
percentile:

============================ ================================================
``mfa_words`` / ``manual_words``  are the acoustic word boundaries accurate
``commercial_cues`` / ``manual_cues``  how close is the shipped cue to a release track
============================ ================================================

An MFA word onset and a release cue onset measure different things; pooling
their samples into one ``p90`` produces a number that is not an estimate of
anything. Lanes are therefore kept apart end to end, and the per-item detail
inside each lane keeps ``reference_id`` so the stratification survives into the
report.

Three rules make the measurement worth trusting:

* **The matcher never reads a timestamp.** Pairing is driven by normalized text
  only (unique n-gram anchors, then a bounded monotonic DP). Timestamps are the
  thing under test; letting them influence pairing would make a badly aligned
  hypothesis match itself and score well.
* **No cross-language pairing.** A reference whose *detected* language differs
  from the lane is ``reference_language_mismatch`` -- exit 2 for that item, not
  a degraded mode. The Japanese lane refuses an English release track even when
  its cue times look plausible.
* **Coverage is reported next to error.** An error figure computed over the 12%
  of cues that happened to match is not a result. Both sides report matched and
  unmatched characters and segments, and coverage below the manifest floor
  invalidates the run instead of quietly shrinking the denominator.

Subtitle tracks are discovered with ``ffprobe`` (never a hardcoded ``0:s:0``),
selected by canonicalized ``tags.language`` plus a text-based language check,
and parsed by :func:`voxweave.subformats.load_subtitle_blocks` -- this file
contains no subtitle parser of its own.

Commands::

    uv run python scripts/calib_alignment.py inspect-tracks MEDIA --lang ja [--json]
    uv run python scripts/calib_alignment.py report --manifest M [--json-out P]
    uv run python scripts/calib_alignment.py check --manifest M --baseline B
    uv run python scripts/calib_alignment.py record-baseline --manifest M --report R --output O

Exit codes follow the shared calibration contract: ``0`` valid and gates
passed, ``1`` valid but a gate failed, ``2`` manifest / schema / reference /
coverage / tooling invalid.

``record-baseline`` is a deliberate human action: it refuses a report whose
manifest digest differs and it is never wired into CI or a default make target.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent


def _load_calib_common() -> Any:
    """Import ``scripts/calib_common.py`` by path -- ``scripts/`` is not a package.

    Loading by path (rather than mutating ``sys.path``) keeps this module
    importable from a test that loads it the same way, without two copies of
    ``calib_common`` disagreeing about which one is authoritative.
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
# Versions and tuning constants
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = 1
METRIC_DEFINITION_VERSION = 1
BASELINE_SCHEMA_VERSION = 1

#: Bumping this changes what "the same text" means, so every recorded baseline
#: becomes incomparable. It is stored in the report and checked by ``check``.
TEXT_NORM_VERSION = 1

WORD_KINDS = ("mfa_words", "manual_words")
CUE_KINDS = ("commercial_cues", "manual_cues")

#: Thresholds each lane reports (design 3.7). Cue lanes are a display ruler, so
#: 25 ms buckets would be noise; word lanes are an acoustic ruler, so a 1 s
#: bucket would be uninformative.
CUE_THRESHOLDS = (0.25, 1.0)
WORD_THRESHOLDS = (0.025, 0.05, 0.10, 0.25)

#: Below these counts a lane is a diagnostic, not a measurement.
MIN_WORD_SAMPLES = 30
MIN_CUE_GROUPS = 20

MANIFEST_DEFAULTS: dict[str, float] = {
    "min_hyp_coverage": 0.8,
    "min_ref_coverage": 0.8,
    "min_pair_similarity": 0.6,
    "bootstrap_samples": 1000,
}

#: DP costs. ``edit`` is a normalized Levenshtein distance in [0, 1], so one
#: skipped segment is priced as exactly one completely wrong group and merging
#: an extra segment into a neighbour group costs a fixed surcharge. Character
#: weight deliberately stays out of the cost: dropping long content must show up
#: in the character coverage gate, not be traded away inside the objective.
GROUP_PENALTY = 0.15
SKIP_COST = 1.0

#: Extra similarity a merged group must show per additional segment.
#:
#: Concatenation lets one good half carry a bad half: gluing a correct 1:1 pair
#: to an unrelated pair produced a 2:2 group at 0.66 similarity in testing, which
#: cleared a flat 0.6 bar and turned two honest skips into two fabricated
#: boundary samples. A merged group therefore has to explain itself better than a
#: 1:1 pair does. Real merges and splits sit near 1.0, so the ladder costs them
#: nothing; a genuine contraction ("don't" vs "do not", 0.73) still clears the
#: one-step bar.
MERGE_SIMILARITY_STEP = 0.08

#: ``(max hypothesis segments, max reference segments)`` per matched group.
#: Word lanes allow 1..12 hypothesis units per reference word (ja/zh char units
#: against one MFA word) and up to 2 reference words (contractions, tokenizer
#: disagreement). Cue lanes allow 1..4 on both sides (merge and split).
MAX_GROUP: dict[str, tuple[int, int]] = {"word": (12, 2), "cue": (4, 4)}

#: Anchor n-gram size: words for a spaced language, characters otherwise.
ANCHOR_N: dict[str, int] = {"en": 3}
ANCHOR_N_DEFAULT = 6

#: Bounded DP. Small windows run the full grid; large ones are restricted to a
#: diagonal band, and the terminal cell is retried unbanded if the band cut it
#: off, so a pathological window degrades in speed but never in correctness.
DP_FULL_CELL_LIMIT = 4096
DP_BAND_MIN = 64
DP_BAND_SLACK = 32
DP_MAX_WINDOW_CELLS = 4_000_000

#: Above this the exact Levenshtein DP is replaced by ``1 - ratio``; only very
#: long merged groups reach it, and only the tie-break resolution suffers.
LEVENSHTEIN_CELL_LIMIT = 250_000

#: Language detection needs enough lexical characters to mean anything.
MIN_DETECT_CHARS = 16
#: Japanese script-ratio floor from design 3.4 rule 6.
JA_SCRIPT_RATIO_MIN = 0.5
#: Two candidate tracks this close in text coverage are not distinguishable.
TRACK_COVERAGE_AMBIGUITY = 0.02

#: One-way gate tolerances used by ``check`` unless the baseline overrides them.
DEFAULT_TOLERANCES: dict[str, float] = {
    "absolute_s": 0.010,
    "relative": 0.05,
    "rate_absolute": 0.02,
    "coverage_absolute": 0.02,
}

#: Lower is better; gated upward only.
GATED_ERROR_FIELDS = ("mae", "median", "p90")
#: Higher is better; gated downward only.
GATED_RATE_FIELDS = (
    "pct_le_0_025",
    "pct_le_0_05",
    "pct_le_0_10",
    "pct_le_0_25",
    "pct_le_1_0",
)


def is_gated_metric(name: str) -> bool:
    """Only absolute-error blocks are gates; the rest are diagnostics.

    ``*_early_s`` / ``*_late_s`` split one sample pool by sign, so their ``n``
    moves whenever the bias direction shifts -- gating them would turn a bias
    that improved its way out of existence into a failure. ``lexical_span_*`` is
    explicitly not a word MAE (the group spans several reference words, so it has
    no interior boundary to be right about), and it disappears entirely when
    tokenization agreement improves.
    """
    return name.endswith("_abs_s") and not name.startswith("lexical_span_")


SUBTITLE_EXTS = (".vtt", ".srt", ".ass", ".ssa")


class ReferenceLanguageMismatch(cc.CalibrationError):
    """The reference is not in the lane's language, so it is not a reference.

    A distinct type rather than a message substring: the report's failure code is
    part of its contract, and matching on prose would silently reclassify the
    moment someone rewords the message.
    """


class TrackSelectionAmbiguous(cc.CalibrationError):
    """Two candidate tracks are indistinguishable; the manifest must pin one."""


class InsufficientCoverage(cc.CalibrationError):
    """Too little of the text paired up for the errors to describe the file."""


#: Exception type -> the ``failures[].code`` it becomes in the report.
FAILURE_CODES: tuple[tuple[type[Exception], str], ...] = (
    (ReferenceLanguageMismatch, "reference_language_mismatch"),
    (TrackSelectionAmbiguous, "track_selection_ambiguous"),
    (InsufficientCoverage, "insufficient_coverage"),
)

#: Image-based subtitles carry no text, so they can never be a text reference.
BITMAP_SUBTITLE_CODECS = frozenset(
    {
        "dvb_subtitle",
        "dvb_teletext",
        "dvbsub",
        "dvd_subtitle",
        "hdmv_pgs_subtitle",
        "hdmv_text_subtitle",
        "pgssub",
        "xsub",
    }
)

#: Titles that mark a partial track: it covers signs or songs, not the dialogue.
_PARTIAL_TRACK_TITLE_RE = re.compile(
    r"\b(sign|signs|song|songs|lyric|lyrics|karaoke|commentary|forced)\b", re.IGNORECASE
)


# --------------------------------------------------------------------------- #
# Text normalization (text_norm_version 1)
# --------------------------------------------------------------------------- #

_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_BRACKETED_RE = re.compile(r"[\[\(（【［〔][^\]\)）】］〕]*[\]\)）】］〕]")
# A speaker prefix is an upper-case-ish name followed by a colon at line start;
# the bound keeps it from eating an ordinary sentence that contains a colon.
_SPEAKER_PREFIX_RE = re.compile(r"^[^\w]{0,3}[A-Z][A-Z0-9 .'’-]{0,23}:\s*")
_LEADING_DASH_RE = re.compile(r"^\s*[-‐-―]\s*")
# Spelled as escapes on purpose: these characters are invisible in an editor, so
# a literal table would be unreviewable and easy to corrupt by accident.
_ZERO_WIDTH = dict.fromkeys(
    (
        0x00AD,  # soft hyphen
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        0x2060,  # word joiner
        0xFEFF,  # zero width no-break space / BOM
    )
)
_APOSTROPHES = "’‘‛`´"
_EN_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

# Script ranges, checked by codepoint rather than ``unicodedata.name`` because
# the latter costs a lookup per character on whole-episode text.
_HIRAGANA = ((0x3041, 0x309F),)
_KATAKANA = ((0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F))
_HAN = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)
_LATIN = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


def strip_markup(text: str) -> str:
    """Remove everything that is styling or annotation rather than speech.

    ASS override blocks, HTML tags, zero-width characters, music notes, bracketed
    sound effects, leading dialogue dashes and ``NAME:`` speaker prefixes all go.
    This runs only on the matcher's copy; the reference and hypothesis keep their
    original ``text`` so a later normalization change cannot rewrite the truth.
    """
    s = _ASS_OVERRIDE_RE.sub(" ", text)
    s = _HTML_TAG_RE.sub(" ", s)
    s = s.translate(_ZERO_WIDTH)
    s = s.replace("♪", " ").replace("♫", " ").replace("\\N", "\n")
    lines: list[str] = []
    for raw in s.replace("\r", "\n").split("\n"):
        line = _BRACKETED_RE.sub(" ", raw)
        line = _LEADING_DASH_RE.sub("", line)
        line = _SPEAKER_PREFIX_RE.sub("", line)
        lines.append(line)
    return " ".join(part for part in lines if part.strip())


def normalize_text(text: str, language: str) -> str:
    """Pure, versioned normalization used for every text comparison.

    English keeps word-internal apostrophes and joins tokens with a single
    space; CJK languages drop punctuation and whitespace entirely and keep Han,
    kana, Latin and digits. The result is the only string the matcher ever sees.
    """
    s = unicodedata.normalize("NFKC", strip_markup(text))
    for apostrophe in _APOSTROPHES:
        s = s.replace(apostrophe, "'")
    s = s.casefold()
    if language == "en":
        return " ".join(_EN_TOKEN_RE.findall(s))
    return "".join(ch for ch in s if ch.isalnum())


def joiner_for(language: str) -> str:
    """Separator used when concatenating several normalized segments."""
    return " " if language == "en" else ""


def lexical_len(norm: str) -> int:
    """Characters that carry content, i.e. everything but the token separator."""
    return sum(1 for ch in norm if not ch.isspace())


def script_counts(text: str) -> dict[str, int]:
    """Lexical codepoints per script family, for language detection."""
    counts = {"kana": 0, "han": 0, "latin": 0, "other": 0}
    for ch in unicodedata.normalize("NFKC", text):
        if not ch.isalnum():
            continue
        code = ord(ch)
        if _in_ranges(code, _HIRAGANA) or _in_ranges(code, _KATAKANA):
            counts["kana"] += 1
        elif _in_ranges(code, _HAN):
            counts["han"] += 1
        elif _in_ranges(code, _LATIN):
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def detect_text_language(text: str) -> str | None:
    """Script-ratio language detection over whole-track text; ``None`` if unsure.

    Deliberately coarse and whole-document: it exists to catch the one failure
    that silently poisons a lane -- an English release track tagged (or manifest
    -declared) as the Japanese reference. Per-cue detection would be noisy;
    across a whole track the kana / Han / Latin ratios are unambiguous.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if total < MIN_DETECT_CHARS:
        return None
    kana = counts["kana"] / total
    han = counts["han"] / total
    latin = counts["latin"] / total
    if kana >= 0.05 and (kana + han) >= JA_SCRIPT_RATIO_MIN:
        return "ja"
    if han >= JA_SCRIPT_RATIO_MIN and kana < 0.01:
        return "zh"
    if latin >= 0.85:
        return "en"
    return None


def japanese_script_ratio(text: str) -> float:
    """Fraction of lexical codepoints that are kana or Han (design 3.4 rule 6)."""
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return 0.0
    return (counts["kana"] + counts["han"]) / total


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Segment:
    """One comparable unit on either side: a cue, or a word/char unit."""

    id: str
    text: str
    start: float
    end: float
    norm: str
    utterance: str | None = None

    @property
    def chars(self) -> int:
        return lexical_len(self.norm)


def make_segments(
    rows: Sequence[Mapping[str, Any]],
    *,
    language: str,
    prefix: str,
) -> list[Segment]:
    """Build segments from ``{text|word, start, end}`` rows, normalizing text once."""
    out: list[Segment] = []
    for i, row in enumerate(rows):
        text = row.get("text")
        if text is None:
            text = row.get("word")
        if text is None:
            text = ""
        start = row.get("start")
        end = row.get("end")
        if start is None or end is None:
            continue
        try:
            s, e = float(start), float(end)
        except (TypeError, ValueError):
            raise cc.CalibrationError(
                f"{prefix}{i}: non-numeric start/end {start!r}/{end!r}"
            ) from None
        if not (math.isfinite(s) and math.isfinite(e)):
            raise cc.CalibrationError(f"{prefix}{i}: non-finite start/end")
        seg_id = str(row.get("id") or f"{prefix}{i:05d}")
        utt = row.get("utterance_id")
        out.append(
            Segment(
                id=seg_id,
                text=str(text),
                start=s,
                end=e,
                norm=normalize_text(str(text), language),
                utterance=str(utt) if utt else None,
            )
        )
    return out


def prepare_segments(segments: Sequence[Segment]) -> tuple[list[Segment], int]:
    """Split into matchable segments and the count that normalized to nothing.

    A cue holding only a music note or a bracketed sound effect has no text to
    match on. Dropping it silently would inflate coverage, so it is counted.
    """
    keep = [s for s in segments if s.norm.strip()]
    return keep, len(segments) - len(keep)


# --------------------------------------------------------------------------- #
# Matcher: unique n-gram anchors + bounded monotonic DP. No timestamps.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MatchGroup:
    """``hyp[hyp_start : hyp_start+hyp_count]`` paired with the reference slice."""

    hyp_start: int
    hyp_count: int
    ref_start: int
    ref_count: int
    similarity: float

    @property
    def shape(self) -> str:
        if self.hyp_count == 1 and self.ref_count == 1:
            return "1:1"
        if self.ref_count == 1:
            return "N:1"
        if self.hyp_count == 1:
            return "1:N"
        return "N:M"


@dataclass
class MatchResult:
    groups: list[MatchGroup]
    hyp: list[Segment]
    ref: list[Segment]
    hyp_empty: int = 0
    ref_empty: int = 0

    def matched_hyp_indices(self) -> set[int]:
        return {g.hyp_start + k for g in self.groups for k in range(g.hyp_count)}

    def matched_ref_indices(self) -> set[int]:
        return {g.ref_start + k for g in self.groups for k in range(g.ref_count)}


def _levenshtein(a: str, b: str) -> int:
    """Plain two-row edit distance; only ever called on one matched group."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
            )
        prev = cur
    return prev[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Edit distance in [0, 1]; long pairs fall back to ``1 - ratio``."""
    if not a and not b:
        return 0.0
    if len(a) * len(b) > LEVENSHTEIN_CELL_LIMIT:  # pragma: no cover - huge groups
        return 1.0 - SequenceMatcher(None, a, b, autojunk=False).ratio()
    return _levenshtein(a, b) / max(len(a), len(b))


class _Side:
    """Cached concatenation of normalized segment text (matcher input only)."""

    def __init__(self, segments: Sequence[Segment], joiner: str) -> None:
        self.segments = segments
        self.joiner = joiner
        self._cache: dict[tuple[int, int], str] = {}

    def concat(self, start: int, count: int) -> str:
        key = (start, count)
        hit = self._cache.get(key)
        if hit is None:
            hit = self.joiner.join(s.norm for s in self.segments[start : start + count])
            self._cache[key] = hit
        return hit


def _token_stream(
    segments: Sequence[Segment], language: str
) -> tuple[list[str], list[int]]:
    """Flatten segments into anchor tokens plus the segment index of each token."""
    tokens: list[str] = []
    owner: list[int] = []
    for idx, seg in enumerate(segments):
        parts = seg.norm.split(" ") if language == "en" else list(seg.norm)
        for part in parts:
            if not part:
                continue
            tokens.append(part)
            owner.append(idx)
    return tokens, owner


def unique_ngram_anchors(
    hyp: Sequence[Segment], ref: Sequence[Segment], language: str
) -> list[tuple[int, int]]:
    """Segment pairs joined by an n-gram that occurs exactly once on both sides.

    Anchors are pure synchronization points: they bound the DP windows so the
    quadratic cost stays local and a repeated line cannot make a greedy matcher
    jump to the wrong occurrence. Only text participates -- no timestamps.
    """
    n = ANCHOR_N.get(language, ANCHOR_N_DEFAULT)
    h_tokens, h_owner = _token_stream(hyp, language)
    r_tokens, r_owner = _token_stream(ref, language)
    if len(h_tokens) < n or len(r_tokens) < n:
        return []

    def index(tokens: list[str]) -> dict[tuple[str, ...], list[int]]:
        table: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(tokens) - n + 1):
            table.setdefault(tuple(tokens[i : i + n]), []).append(i)
        return table

    h_index = index(h_tokens)
    r_index = index(r_tokens)
    pairs: list[tuple[int, int]] = []
    for gram, h_positions in h_index.items():
        if len(h_positions) != 1:
            continue
        r_positions = r_index.get(gram)
        if r_positions is None or len(r_positions) != 1:
            continue
        pairs.append((h_owner[h_positions[0]], r_owner[r_positions[0]]))

    # One anchor per segment pair, then the longest strictly increasing chain:
    # anchors that cross each other cannot both be true under a monotonic edit.
    unique_pairs = sorted(set(pairs))
    seen_h: set[int] = set()
    seen_r: set[int] = set()
    duplicate_h = {h for h, _ in unique_pairs if (h in seen_h or seen_h.add(h))}
    duplicate_r = {r for _, r in unique_pairs if (r in seen_r or seen_r.add(r))}
    candidates = [
        (h, r) for h, r in unique_pairs if h not in duplicate_h and r not in duplicate_r
    ]
    return _longest_increasing_pairs(candidates)


def _longest_increasing_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest chain strictly increasing in both coordinates (O(n^2), n is small)."""
    if not pairs:
        return []
    ordered = sorted(pairs)
    best = [1] * len(ordered)
    back = [-1] * len(ordered)
    for i in range(len(ordered)):
        for j in range(i):
            strictly_increasing = (
                ordered[j][0] < ordered[i][0] and ordered[j][1] < ordered[i][1]
            )
            if strictly_increasing and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                back[i] = j
    end = max(range(len(ordered)), key=lambda i: (best[i], -i))
    chain: list[tuple[int, int]] = []
    while end != -1:
        chain.append(ordered[end])
        end = back[end]
    chain.reverse()
    return chain


def _windows(
    n_hyp: int, n_ref: int, anchors: Sequence[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    """``(h_lo, h_hi, r_lo, r_hi)`` slices between consecutive anchors.

    Each anchor starts the window that follows it, so the anchored pair itself is
    still matched by the DP (at 1:1, its cheapest option) rather than asserted.
    """
    out: list[tuple[int, int, int, int]] = []
    prev_h = prev_r = 0
    for h, r in anchors:
        if h > prev_h or r > prev_r:
            out.append((prev_h, h, prev_r, r))
            prev_h, prev_r = h, r
    out.append((prev_h, n_hyp, prev_r, n_ref))
    return [w for w in out if w[1] > w[0] or w[3] > w[2]]


@dataclass(frozen=True)
class _State:
    cost: float
    skips: int
    groups: int
    back: tuple[int, int] | None
    move: MatchGroup | None

    @property
    def key(self) -> tuple[float, int, int]:
        return (self.cost, self.skips, self.groups)


def _dp_window(
    h_side: _Side,
    r_side: _Side,
    *,
    h_lo: int,
    h_hi: int,
    r_lo: int,
    r_hi: int,
    max_h: int,
    max_r: int,
    min_similarity: float,
    banded: bool,
) -> list[MatchGroup] | None:
    """Monotonic DP over one window; ``None`` when the band cut off the terminal.

    Cells are finalized in increasing ``i + j`` order (every transition advances
    at least one index), and among equal ``i + j`` the smaller reference index is
    expanded first, so an exact cost tie resolves to the earlier reference -- the
    tie-break is fixed by iteration order, not by dict or sort stability.
    """
    n_h = h_hi - h_lo
    n_r = r_hi - r_lo
    if n_h * n_r > DP_MAX_WINDOW_CELLS:  # pragma: no cover - needs a huge corpus
        raise cc.CalibrationError(
            f"unanchored matcher window of {n_h}x{n_r} segments is too large; "
            "the two texts share almost no unique n-gram, which usually means "
            "the reference is not the same content"
        )
    band = max(DP_BAND_MIN, abs(n_h - n_r) + DP_BAND_SLACK)

    def in_band(i: int, j: int) -> bool:
        if not banded or n_h == 0 or n_r == 0:
            return True
        return abs(j - i * n_r / n_h) <= band

    start = _State(0.0, 0, 0, None, None)
    dp: dict[tuple[int, int], _State] = {(0, 0): start}

    def relax(cell: tuple[int, int], cand: _State) -> None:
        if not in_band(*cell):
            return
        cur = dp.get(cell)
        if cur is None or cand.key < cur.key:
            dp[cell] = cand

    # Finalization order: i + j ascending, then j ascending (earlier reference
    # index wins ties), then i.
    order = sorted(
        ((i + j, j, i) for i in range(n_h + 1) for j in range(n_r + 1)),
    )
    for _total, j, i in order:
        state = dp.get((i, j))
        if state is None:
            continue
        if i == n_h and j == n_r:
            continue
        for hn in range(1, min(max_h, n_h - i) + 1):
            hs = h_side.concat(h_lo + i, hn)
            for rn in range(1, min(max_r, n_r - j) + 1):
                rs = r_side.concat(r_lo + j, rn)
                sim = SequenceMatcher(None, hs, rs, autojunk=False).ratio()
                if sim < min_similarity + MERGE_SIMILARITY_STEP * (hn + rn - 2):
                    continue
                cost = (
                    state.cost
                    + normalized_levenshtein(hs, rs)
                    + GROUP_PENALTY * (hn + rn - 2)
                )
                relax(
                    (i + hn, j + rn),
                    _State(
                        cost,
                        state.skips,
                        state.groups + hn + rn,
                        (i, j),
                        MatchGroup(h_lo + i, hn, r_lo + j, rn, sim),
                    ),
                )
        if i < n_h:
            relax(
                (i + 1, j),
                _State(
                    state.cost + SKIP_COST, state.skips + 1, state.groups, (i, j), None
                ),
            )
        if j < n_r:
            relax(
                (i, j + 1),
                _State(
                    state.cost + SKIP_COST, state.skips + 1, state.groups, (i, j), None
                ),
            )

    terminal = dp.get((n_h, n_r))
    if terminal is None:
        return None
    groups: list[MatchGroup] = []
    cell: tuple[int, int] | None = (n_h, n_r)
    while cell is not None:
        state = dp[cell]
        if state.move is not None:
            groups.append(state.move)
        cell = state.back
    groups.reverse()
    return groups


def pair_monotonic(
    hyp_segments: Sequence[Segment],
    ref_segments: Sequence[Segment],
    *,
    language: str,
    level: str,
    min_pair_similarity: float,
) -> MatchResult:
    """Pair hypothesis against reference segments using text only.

    Timestamps are absent from anchor discovery, from the transition cost and
    from the tie-break. This is the checklist item that makes the whole ruler
    meaningful: a hypothesis whose timing collapsed must still pair by its words
    and then score badly, instead of matching itself because it is "close in
    time".
    """
    if level not in MAX_GROUP:
        raise cc.CalibrationError(f"unknown match level {level!r}")
    hyp, hyp_empty = prepare_segments(hyp_segments)
    ref, ref_empty = prepare_segments(ref_segments)
    if not hyp or not ref:
        return MatchResult([], hyp, ref, hyp_empty, ref_empty)

    joiner = joiner_for(language)
    h_side = _Side(hyp, joiner)
    r_side = _Side(ref, joiner)
    max_h, max_r = MAX_GROUP[level]
    anchors = unique_ngram_anchors(hyp, ref, language)

    groups: list[MatchGroup] = []
    for h_lo, h_hi, r_lo, r_hi in _windows(len(hyp), len(ref), anchors):
        cells = (h_hi - h_lo) * (r_hi - r_lo)
        banded = cells > DP_FULL_CELL_LIMIT
        window = _dp_window(
            h_side,
            r_side,
            h_lo=h_lo,
            h_hi=h_hi,
            r_lo=r_lo,
            r_hi=r_hi,
            max_h=max_h,
            max_r=max_r,
            min_similarity=min_pair_similarity,
            banded=banded,
        )
        if window is None:  # pragma: no cover - band never reached the terminal
            window = _dp_window(
                h_side,
                r_side,
                h_lo=h_lo,
                h_hi=h_hi,
                r_lo=r_lo,
                r_hi=r_hi,
                max_h=max_h,
                max_r=max_r,
                min_similarity=min_pair_similarity,
                banded=False,
            )
        if window is None:  # pragma: no cover - defensive
            raise cc.CalibrationError("matcher failed to reach the window terminal")
        groups.extend(window)

    _reject_crossing_or_duplicate(groups)
    return MatchResult(groups, hyp, ref, hyp_empty, ref_empty)


def _reject_crossing_or_duplicate(groups: Sequence[MatchGroup]) -> None:
    """Guard the monotonic invariant: groups never overlap and never cross."""
    h_cursor = r_cursor = 0
    for g in groups:
        if g.hyp_start < h_cursor or g.ref_start < r_cursor:
            raise cc.CalibrationError(
                "matcher produced crossing or duplicate pairs "
                f"(hyp {g.hyp_start}, ref {g.ref_start})"
            )
        h_cursor = g.hyp_start + g.hyp_count
        r_cursor = g.ref_start + g.ref_count


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def _fraction(matched: int, total: int) -> dict[str, Any]:
    """A coverage rate that keeps its numerator and denominator.

    ``value`` is ``None`` -- never ``0.0`` -- when nothing was eligible, so an
    empty lane cannot look like a total miss (or, inverted, a perfect score).
    """
    return {
        "matched": matched,
        "total": total,
        "value": (matched / total) if total else None,
    }


def coverage_of(result: MatchResult) -> dict[str, Any]:
    """Matched / unmatched characters and segments on both sides."""
    h_matched = result.matched_hyp_indices()
    r_matched = result.matched_ref_indices()
    h_chars = sum(s.chars for s in result.hyp)
    r_chars = sum(s.chars for s in result.ref)
    h_chars_matched = sum(result.hyp[i].chars for i in h_matched)
    r_chars_matched = sum(result.ref[i].chars for i in r_matched)
    shapes = {"1:1": 0, "1:N": 0, "N:1": 0, "N:M": 0}
    for g in result.groups:
        shapes[g.shape] += 1
    return {
        "hyp_chars": _fraction(h_chars_matched, h_chars),
        "ref_chars": _fraction(r_chars_matched, r_chars),
        "hyp_segments": _fraction(len(h_matched), len(result.hyp)),
        "ref_segments": _fraction(len(r_matched), len(result.ref)),
        "hyp_unmatched_segments": len(result.hyp) - len(h_matched),
        "ref_unmatched_segments": len(result.ref) - len(r_matched),
        "hyp_unmatched_chars": h_chars - h_chars_matched,
        "ref_unmatched_chars": r_chars - r_chars_matched,
        "empty_after_normalization": {
            "hypothesis": result.hyp_empty,
            "reference": result.ref_empty,
        },
        "groups": len(result.groups),
        "match_shapes": shapes,
    }


# --------------------------------------------------------------------------- #
# Errors and metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoundaryError:
    """One matched group reduced to its two boundary errors."""

    hyp_ids: tuple[str, ...]
    ref_ids: tuple[str, ...]
    ref_count: int
    similarity: float
    start_abs: float
    end_abs: float
    start_signed: float
    end_signed: float
    cluster: str
    shape: str = "1:1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize one pair for manual spot-checking (design 3.6)."""
        return {
            "hyp_ids": list(self.hyp_ids),
            "ref_ids": list(self.ref_ids),
            "match_shape": self.shape,
            "similarity": self.similarity,
            "start_abs_s": self.start_abs,
            "end_abs_s": self.end_abs,
            "start_signed_s": self.start_signed,
            "end_signed_s": self.end_signed,
        }


def boundary_errors(result: MatchResult, *, cluster_prefix: str) -> list[BoundaryError]:
    """Outer-boundary error of every matched group (this is where time enters).

    Timestamps are read only here, after pairing is already fixed, so they can
    influence the score but never the correspondence.
    """
    out: list[BoundaryError] = []
    for g in result.groups:
        h0 = result.hyp[g.hyp_start]
        h1 = result.hyp[g.hyp_start + g.hyp_count - 1]
        r0 = result.ref[g.ref_start]
        r1 = result.ref[g.ref_start + g.ref_count - 1]
        start_signed = h0.start - r0.start
        end_signed = h1.end - r1.end
        out.append(
            BoundaryError(
                hyp_ids=tuple(
                    result.hyp[g.hyp_start + k].id for k in range(g.hyp_count)
                ),
                ref_ids=tuple(
                    result.ref[g.ref_start + k].id for k in range(g.ref_count)
                ),
                ref_count=g.ref_count,
                similarity=g.similarity,
                start_abs=abs(start_signed),
                end_abs=abs(end_signed),
                start_signed=start_signed,
                end_signed=end_signed,
                cluster=f"{cluster_prefix}|{r0.utterance or r0.id}",
                shape=g.shape,
            )
        )
    return out


def cluster_bootstrap_ci(
    clusters: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int = 0,
) -> tuple[float, float] | None:
    """95% CI of the median, resampling *clusters* (utterances), not samples.

    Hundreds of word boundaries inside one sentence are not hundreds of
    independent observations; resampling the sentence is what keeps the interval
    honest. Deterministic: the RNG is seeded per call.
    """
    keys = [k for k, v in clusters.items() if v]
    if samples <= 0 or len(keys) < 2:
        return None
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(samples):
        pool: list[float] = []
        for _ in range(len(keys)):
            pool.extend(clusters[keys[rng.randrange(len(keys))]])
        value = cc.percentile(pool, 50.0)
        if value is not None:
            medians.append(value)
    if not medians:  # pragma: no cover - only when every cluster is empty
        return None
    low = cc.percentile(medians, 2.5)
    high = cc.percentile(medians, 97.5)
    if low is None or high is None:  # pragma: no cover - defensive
        return None
    return (low, high)


def _lower_bound(block: Mapping[str, Any], uncertainty: float | None) -> float | None:
    """Interpretive floor ``max(0, median - reference_uncertainty)``.

    Reported *next to* the observed value under a name that says what it is. The
    per-sample errors are never reduced: subtracting a nominal 20 ms from each
    MFA boundary would manufacture accuracy the measurement does not have.
    """
    if uncertainty is None or block.get("median") is None:
        return None
    return max(0.0, float(block["median"]) - float(uncertainty))


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReferenceSpec:
    id: str
    kind: str
    language: str
    path: Path | None
    media: Path | None
    stream_index: int | None
    expected_codec: str | None
    quality: str | None
    enabled: bool

    @property
    def level(self) -> str:
        return "word" if self.kind in WORD_KINDS else "cue"


@dataclass(frozen=True)
class ItemSpec:
    id: str
    language: str
    media: Path | None
    hypothesis_path: Path
    references: tuple[ReferenceSpec, ...]
    include_ranges: tuple[tuple[float, float], ...]
    exclude_ranges: tuple[tuple[float, float], ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    path: Path
    document: dict[str, Any]
    digest: str
    defaults: dict[str, float]
    items: tuple[ItemSpec, ...]


_ENV_PREFIX_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}[/\\]?")


def resolve_manifest_path(raw: str, *, base: Path, root_env: str) -> Path:
    """Resolve a manifest path, expanding a leading ``${VOXWEAVE_CALIB_ROOT}``.

    The placeholder is only honoured at the very start of the string, and the
    canonical result must stay inside that root -- a private corpus reference
    must not be able to reach out of it via ``..``.
    """
    text = str(raw)
    match = _ENV_PREFIX_RE.match(text)
    if match is not None:
        name = match.group(1)
        if name != root_env:
            raise cc.CalibrationError(
                f"path {raw!r} uses ${{{name}}} but the manifest declares "
                f"data_root_env={root_env!r}"
            )
        root_raw = os.environ.get(name)
        if not root_raw:
            raise cc.CalibrationError(
                f"path {raw!r} needs environment variable {name} to be set"
            )
        root = Path(root_raw).expanduser().resolve()
        resolved = (root / text[match.end() :]).resolve()
        if root not in resolved.parents and resolved != root:
            raise cc.CalibrationError(
                f"path {raw!r} resolves to {resolved}, outside {name}={root}"
            )
        return resolved
    if "${" in text:
        raise cc.CalibrationError(
            f"path {raw!r} may only use ${{{root_env}}} and only as a prefix"
        )
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def _normalize_ranges(
    raw: Sequence[Sequence[float]] | None, *, label: str
) -> tuple[tuple[float, float], ...]:
    """Validate ``[start, end]`` pairs and merge overlaps into a sorted union."""
    if not raw:
        return ()
    spans: list[tuple[float, float]] = []
    for pair in raw:
        start, end = float(pair[0]), float(pair[1])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise cc.CalibrationError(f"{label}: non-finite range {pair!r}")
        if end <= start:
            raise cc.CalibrationError(f"{label}: range end must exceed start: {pair!r}")
        spans.append((start, end))
    spans.sort()
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _within(t: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(lo <= t <= hi for lo, hi in spans)


def filter_by_ranges(
    segments: Sequence[Segment],
    includes: Sequence[tuple[float, float]],
    excludes: Sequence[tuple[float, float]],
) -> list[Segment]:
    """Select the evaluated window by segment midpoint.

    Range selection reads timestamps, and that is fine: it chooses *which region
    is measured*, before any pairing happens. It is never consulted by the
    matcher.
    """
    out: list[Segment] = []
    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        if includes and not _within(mid, includes):
            continue
        if excludes and _within(mid, excludes):
            continue
        out.append(seg)
    return out


def load_manifest(path: str | Path) -> Manifest:
    """Read, schema-validate and semantically check an alignment manifest.

    JSON Schema cannot express the cross-object rules, so the loader owns them:
    item language equals every reference language, ids are unique, ranges are
    well-formed, and paths resolve inside the declared data root.
    """
    manifest_path = Path(path).resolve()
    document = cc.read_json(manifest_path)
    errors = cc.schema_errors(document, "alignment-manifest")
    if errors:
        raise cc.CalibrationError(f"{manifest_path} failed schema validation", errors)
    base = manifest_path.parent
    root_env = str(document.get("data_root_env") or "VOXWEAVE_CALIB_ROOT")
    defaults = dict(MANIFEST_DEFAULTS)
    defaults.update(document.get("defaults") or {})

    items: list[ItemSpec] = []
    seen_items: set[str] = set()
    for raw_item in document["items"]:
        item_id = raw_item["id"]
        if item_id in seen_items:
            raise cc.CalibrationError(f"duplicate item id {item_id!r}")
        seen_items.add(item_id)
        language = cc.require_calibration_language(raw_item["language"])
        media = raw_item.get("media")
        references: list[ReferenceSpec] = []
        seen_refs: set[str] = set()
        for raw_ref in raw_item["references"]:
            ref_id = raw_ref["id"]
            if ref_id in seen_refs:
                raise cc.CalibrationError(
                    f"item {item_id!r}: duplicate reference id {ref_id!r}"
                )
            seen_refs.add(ref_id)
            ref_language = cc.require_calibration_language(raw_ref["language"])
            if ref_language != language:
                raise cc.CalibrationError(
                    f"item {item_id!r} is {language!r} but reference {ref_id!r} "
                    f"declares {ref_language!r}; cross-language pairing is never valid"
                )
            ref_media = raw_ref.get("media") or media
            references.append(
                ReferenceSpec(
                    id=ref_id,
                    kind=raw_ref["kind"],
                    language=ref_language,
                    path=(
                        resolve_manifest_path(
                            raw_ref["path"], base=base, root_env=root_env
                        )
                        if raw_ref.get("path")
                        else None
                    ),
                    media=(
                        resolve_manifest_path(ref_media, base=base, root_env=root_env)
                        if ref_media
                        else None
                    ),
                    stream_index=raw_ref.get("stream_index"),
                    expected_codec=raw_ref.get("expected_codec"),
                    quality=raw_ref.get("quality"),
                    enabled=bool(raw_ref.get("enabled", True)),
                )
            )
        items.append(
            ItemSpec(
                id=item_id,
                language=language,
                media=(
                    resolve_manifest_path(media, base=base, root_env=root_env)
                    if media
                    else None
                ),
                hypothesis_path=resolve_manifest_path(
                    raw_item["hypothesis"]["path"], base=base, root_env=root_env
                ),
                references=tuple(references),
                include_ranges=_normalize_ranges(
                    raw_item.get("include_ranges"),
                    label=f"item {item_id} include_ranges",
                ),
                exclude_ranges=_normalize_ranges(
                    raw_item.get("exclude_ranges"),
                    label=f"item {item_id} exclude_ranges",
                ),
                tags=tuple(raw_item.get("tags") or ()),
            )
        )

    return Manifest(
        path=manifest_path,
        document=document,
        digest=cc.canonical_digest(document),
        defaults=defaults,
        items=tuple(items),
    )


# --------------------------------------------------------------------------- #
# Hypothesis and reference documents
# --------------------------------------------------------------------------- #


def _subtitle_blocks(path: Path) -> list[dict[str, Any]]:
    """Parse any supported subtitle file through the production loader.

    ``voxweave.subformats`` already tolerates the encodings, the inverted
    timestamps and the ASS override syntax found in the wild. A second parser
    here would drift from the one the pipeline actually ships.
    """
    try:
        from voxweave.subformats import load_subtitle_blocks
    except ImportError as exc:  # pragma: no cover - environment problem
        raise cc.CalibrationError(
            "voxweave must be importable to parse subtitle files", [str(exc)]
        ) from None
    try:
        return load_subtitle_blocks(Path(path))
    except (OSError, RuntimeError, ValueError) as exc:
        raise cc.CalibrationError(f"cannot parse {path}: {exc}") from None


def load_hypothesis_segments(path: Path, *, language: str, level: str) -> list[Segment]:
    """Load the voxweave output under test: a subtitle file or a sibling JSON.

    Cue lanes read the rendered subtitle (or a ``cues`` array); word lanes read
    the sibling JSON's ``word_segments``, because a subtitle carries no word
    units at all.
    """
    p = Path(path)
    if not p.exists():
        raise cc.CalibrationError(f"hypothesis {p} does not exist")
    if p.suffix.lower() in SUBTITLE_EXTS:
        if level == "word":
            raise cc.CalibrationError(
                f"hypothesis {p.name} is a subtitle file, which has no word units; "
                "a word lane needs the sibling JSON with word_segments"
            )
        return make_segments(_subtitle_blocks(p), language=language, prefix="hyp")
    if p.suffix.lower() != ".json":
        raise cc.CalibrationError(
            f"hypothesis {p.name}: expected one of {', '.join(SUBTITLE_EXTS)} or .json"
        )
    doc = cc.read_json(p)
    if isinstance(doc, list):
        rows: Any = doc
    elif isinstance(doc, Mapping):
        key = "word_segments" if level == "word" else "cues"
        rows = doc.get(key)
        if rows is None and level == "word":
            rows = doc.get("words")
        if rows is None and level == "cue":
            rows = doc.get("blocks")
        if rows is None:
            raise cc.CalibrationError(
                f"hypothesis {p.name}: no {key!r} array for a {level} lane"
            )
    else:
        raise cc.CalibrationError(f"hypothesis {p.name}: unexpected JSON top level")
    if not isinstance(rows, list) or not rows:
        raise cc.CalibrationError(f"hypothesis {p.name}: empty segment list")
    return make_segments(rows, language=language, prefix="hyp")


def load_reference_document(path: Path) -> dict[str, Any]:
    """Read and validate a normalized reference against its tracked schema."""
    doc = cc.read_json(path)
    errors = cc.schema_errors(doc, "alignment-reference")
    if errors:
        raise cc.CalibrationError(f"{path} failed schema validation", errors)
    return doc


def reference_segments(
    doc: Mapping[str, Any], *, language: str
) -> tuple[list[Segment], int]:
    """Apply ``offset_s``, check the read-only invariants, drop excluded segments.

    Returns the usable segments plus the number that were explicitly excluded --
    MFA ``spn`` / OOV intervals are marked, never silently deleted, so the report
    can state what fraction of the truth was unusable.
    """
    offset = float(doc.get("offset_s") or 0.0)
    duration = doc.get("media_duration_s")
    rows: list[dict[str, Any]] = []
    excluded = 0
    previous: tuple[float, float, str] | None = None
    by_utterance: dict[str, float] = {}
    for raw in doc["segments"]:
        start = float(raw["start"]) + offset
        end = float(raw["end"]) + offset
        if not (math.isfinite(start) and math.isfinite(end)):
            raise cc.CalibrationError(f"segment {raw['id']!r}: non-finite time")
        if end <= start:
            raise cc.CalibrationError(
                f"segment {raw['id']!r}: end {end} must exceed start {start}"
            )
        if start < 0:
            raise cc.CalibrationError(
                f"segment {raw['id']!r}: offset_s pushes start to {start}"
            )
        if duration is not None and end > float(duration) + 1e-6:
            raise cc.CalibrationError(
                f"segment {raw['id']!r}: end {end} exceeds media_duration_s {duration}"
            )
        key = (start, end, str(raw["id"]))
        if previous is not None and key < previous:
            raise cc.CalibrationError(
                f"segment {raw['id']!r} breaks (start, end, id) monotonicity"
            )
        previous = key
        utt = raw.get("utterance_id")
        if utt:
            last_end = by_utterance.get(str(utt))
            if last_end is not None and start < last_end - 1e-6:
                raise cc.CalibrationError(
                    f"segment {raw['id']!r} overlaps another segment of "
                    f"utterance {utt!r}"
                )
            by_utterance[str(utt)] = end
        if raw.get("excluded"):
            excluded += 1
            continue
        rows.append(
            {
                "id": raw["id"],
                "text": raw["text"],
                "start": start,
                "end": end,
                "utterance_id": utt,
            }
        )
    return make_segments(rows, language=language, prefix="ref"), excluded


# --------------------------------------------------------------------------- #
# Subtitle track discovery (ffprobe behind a thin, stubbable seam)
# --------------------------------------------------------------------------- #

Runner = Callable[[Sequence[str]], str]


def run_command(args: Sequence[str], *, timeout: float = 300.0) -> str:
    """Run an external tool and return stdout; every failure maps to exit 2.

    Every ffprobe / ffmpeg invocation in this module funnels through here so a
    test can replace one seam instead of patching ``subprocess``.
    """
    try:
        proc = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        raise cc.CalibrationError(
            f"{args[0]} not found on PATH; install ffmpeg to inspect media tracks"
        ) from None
    except subprocess.TimeoutExpired:
        raise cc.CalibrationError(
            f"{args[0]} timed out after {timeout:g}s: {' '.join(args)}"
        ) from None
    if proc.returncode != 0:
        raise cc.CalibrationError(
            f"{args[0]} failed with exit {proc.returncode}",
            [line for line in (proc.stderr or "").splitlines()[-6:] if line.strip()],
        )
    return proc.stdout


@dataclass(frozen=True)
class SubtitleStream:
    """One ``ffprobe`` subtitle stream, with its tags already canonicalized."""

    index: int
    codec: str
    language: str | None
    raw_language: str | None
    title: str
    default: bool
    forced: bool
    hearing_impaired: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "codec": self.codec,
            "language": self.language,
            "raw_language": self.raw_language,
            "title": self.title,
            "default": self.default,
            "forced": self.forced,
            "hearing_impaired": self.hearing_impaired,
        }


def probe_subtitle_streams(
    media: Path, *, run: Runner = run_command
) -> list[SubtitleStream]:
    """List every subtitle stream with its real container index.

    The index returned is the absolute stream index, which is what ``-map 0:N``
    wants. Nothing here assumes ``0:s:0`` exists, let alone that it is the right
    language -- that assumption is exactly what this harness replaces.
    """
    raw = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            (
                "stream=index,codec_name"
                ":stream_tags=language,title,handler_name"
                ":stream_disposition=default,forced,hearing_impaired"
            ),
            "-of",
            "json",
            str(media),
        ]
    )
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise cc.CalibrationError(f"ffprobe returned invalid JSON: {exc}") from None
    streams: list[SubtitleStream] = []
    for entry in doc.get("streams") or []:
        tags = entry.get("tags") or {}
        disposition = entry.get("disposition") or {}
        raw_language = tags.get("language")
        streams.append(
            SubtitleStream(
                index=int(entry["index"]),
                codec=str(entry.get("codec_name") or ""),
                language=cc.canonical_language_or(raw_language, None),
                raw_language=raw_language,
                title=str(tags.get("title") or tags.get("handler_name") or ""),
                default=bool(disposition.get("default")),
                forced=bool(disposition.get("forced")),
                hearing_impaired=bool(disposition.get("hearing_impaired")),
            )
        )
    return streams


def extract_subtitle_track(
    media: Path, index: int, dest: Path, *, codec: str = "", run: Runner = run_command
) -> Path:
    """Demux one subtitle stream to a temporary file the shared parser can read."""
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(media), "-map"]
    args += [f"0:{index}"]
    args += ["-c:s", "copy" if codec in ("ass", "ssa", "webvtt") else "srt"]
    args += [str(dest)]
    run(args)
    return dest


def track_suffix(codec: str) -> str:
    """Container-native extension for a text subtitle codec."""
    if codec in ("ass", "ssa"):
        return ".ass"
    if codec == "webvtt":
        return ".vtt"
    return ".srt"


def _ngrams(norm: str, language: str, n: int) -> set[tuple[str, ...]]:
    tokens = norm.split(" ") if language == "en" else list(norm)
    tokens = [t for t in tokens if t]
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def text_coverage(hypothesis: str, candidate: str, language: str) -> float:
    """Cheap text-only overlap used to rank candidate tracks.

    Deliberately not the DP matcher: ranking runs over every candidate track and
    only needs to tell "same content" from "different content". Like the matcher,
    it never looks at a timestamp.
    """
    n = ANCHOR_N.get(language, ANCHOR_N_DEFAULT)
    hyp_grams = _ngrams(hypothesis, language, n)
    if not hyp_grams:
        return 0.0
    cand_grams = _ngrams(candidate, language, n)
    return len(hyp_grams & cand_grams) / len(hyp_grams)


@dataclass(frozen=True)
class TrackCandidate:
    stream: SubtitleStream
    rejected: str | None = None
    coverage: float | None = None
    detected_language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = self.stream.to_dict()
        out["rejected"] = self.rejected
        out["coverage"] = self.coverage
        out["detected_language"] = self.detected_language
        return out


def _static_rejection(
    stream: SubtitleStream, language: str, expected_codec: str | None
) -> str | None:
    """Reasons a stream is out before any text is read."""
    if stream.codec in BITMAP_SUBTITLE_CODECS:
        return "bitmap_subtitle_codec"
    if expected_codec and stream.codec != expected_codec:
        return f"codec_mismatch(expected {expected_codec})"
    if stream.language is None:
        # An untagged track is never guessed into the item language; name it
        # explicitly in the manifest if it really is the reference.
        return "untagged_language"
    if stream.language != language:
        return f"language_mismatch({stream.language})"
    if stream.forced:
        return "forced_disposition"
    if _PARTIAL_TRACK_TITLE_RE.search(stream.title):
        return "partial_track_title"
    return None


@dataclass(frozen=True)
class TrackSelection:
    stream: SubtitleStream
    blocks: list[dict[str, Any]]
    coverage: float | None
    detected_language: str | None
    candidates: tuple[TrackCandidate, ...]


def select_subtitle_track(
    media: Path,
    *,
    language: str,
    hypothesis_norm: str,
    explicit_index: int | None = None,
    expected_codec: str | None = None,
    probe: Callable[[Path], list[SubtitleStream]] = probe_subtitle_streams,
    extract: Callable[[Path, int, Path, str], Path] | None = None,
) -> TrackSelection:
    """Pick the same-language dialogue track, by tags first and text second.

    Order of business (design 3.4): an explicit manifest index wins but is still
    validated; language tags are canonicalized and never guessed; bitmap, forced
    and signs/songs tracks are dropped; the survivors are parsed and ranked by
    text coverage against the hypothesis; two candidates within 2 points of each
    other are an ambiguity the manifest has to resolve; and a Japanese lane
    additionally requires a kana/Han script ratio, so a mistagged English
    translation track is refused rather than paired.
    """
    if not Path(media).exists():
        raise cc.CalibrationError(f"media {media} does not exist")
    do_extract = extract or (
        lambda m, i, d, c: extract_subtitle_track(m, i, d, codec=c)
    )
    streams = probe(Path(media))
    if not streams:
        raise cc.CalibrationError(f"{Path(media).name}: no subtitle streams")

    if explicit_index is not None:
        chosen = [s for s in streams if s.index == explicit_index]
        if not chosen:
            raise cc.CalibrationError(
                f"{Path(media).name}: stream_index {explicit_index} is not a "
                f"subtitle stream (found {[s.index for s in streams]})"
            )
        stream = chosen[0]
        if expected_codec and stream.codec != expected_codec:
            raise cc.CalibrationError(
                f"stream {stream.index}: codec {stream.codec!r} != expected "
                f"{expected_codec!r}"
            )
        if stream.language is not None and stream.language != language:
            raise ReferenceLanguageMismatch(
                f"stream {stream.index}: tagged {stream.language!r}, lane is "
                f"{language!r}; cross-language pairing is never valid"
            )
        blocks = _read_track(media, stream, do_extract)
        detected = detect_text_language(" ".join(b.get("text", "") for b in blocks))
        _require_same_language(stream, blocks, detected, language)
        return TrackSelection(
            stream=stream,
            blocks=blocks,
            coverage=None,
            detected_language=detected,
            candidates=(TrackCandidate(stream, None, None, detected),),
        )

    candidates: list[TrackCandidate] = []
    scored: list[tuple[float, SubtitleStream, list[dict[str, Any]], str | None]] = []
    for stream in streams:
        reason = _static_rejection(stream, language, expected_codec)
        if reason is not None:
            candidates.append(TrackCandidate(stream, reason))
            continue
        blocks = _read_track(media, stream, do_extract)
        text = " ".join(str(b.get("text", "")) for b in blocks)
        detected = detect_text_language(text)
        if detected is not None and not cc.languages_match(detected, language):
            candidates.append(
                TrackCandidate(stream, f"detected_language({detected})", None, detected)
            )
            continue
        if language == "ja" and japanese_script_ratio(text) < JA_SCRIPT_RATIO_MIN:
            candidates.append(
                TrackCandidate(
                    stream, "japanese_script_ratio_below_floor", None, detected
                )
            )
            continue
        coverage = text_coverage(
            hypothesis_norm, normalize_text(text, language), language
        )
        candidates.append(TrackCandidate(stream, None, coverage, detected))
        scored.append((coverage, stream, blocks, detected))

    if not scored:
        raise cc.CalibrationError(
            f"{Path(media).name}: no same-language subtitle track survived selection",
            [f"stream {c.stream.index}: {c.rejected}" for c in candidates],
        )
    scored.sort(key=lambda row: (-row[0], row[1].index))
    if len(scored) > 1 and (scored[0][0] - scored[1][0]) < TRACK_COVERAGE_AMBIGUITY:
        raise TrackSelectionAmbiguous(
            f"{Path(media).name}: subtitle track selection is ambiguous "
            f"({scored[0][0]:.3f} vs {scored[1][0]:.3f}); pin stream_index in the manifest",
            [f"stream {s.index}: coverage {c:.3f}" for c, s, _, _ in scored[:3]],
        )
    coverage, stream, blocks, detected = scored[0]
    return TrackSelection(
        stream=stream,
        blocks=blocks,
        coverage=coverage,
        detected_language=detected,
        candidates=tuple(candidates),
    )


def _read_track(
    media: Path,
    stream: SubtitleStream,
    extract: Callable[[Path, int, Path, str], Path],
) -> list[dict[str, Any]]:
    """Demux one stream into a temp file and parse it with the shared loader."""
    with tempfile.TemporaryDirectory(prefix="voxweave-calib-") as tmp:
        dest = Path(tmp) / f"track{stream.index}{track_suffix(stream.codec)}"
        extract(Path(media), stream.index, dest, stream.codec)
        if not dest.exists():
            raise cc.CalibrationError(
                f"stream {stream.index}: extraction produced no file"
            )
        return _subtitle_blocks(dest)


def _require_same_language(
    stream: SubtitleStream,
    blocks: Sequence[Mapping[str, Any]],
    detected: str | None,
    language: str,
) -> None:
    text = " ".join(str(b.get("text", "")) for b in blocks)
    if detected is not None and not cc.languages_match(detected, language):
        raise ReferenceLanguageMismatch(
            f"stream {stream.index}: text detects as {detected!r} but the lane is "
            f"{language!r}"
        )
    if language == "ja" and japanese_script_ratio(text) < JA_SCRIPT_RATIO_MIN:
        raise ReferenceLanguageMismatch(
            f"stream {stream.index}: Japanese script ratio "
            f"{japanese_script_ratio(text):.2f} is below {JA_SCRIPT_RATIO_MIN}"
        )


# --------------------------------------------------------------------------- #
# Per-item evaluation
# --------------------------------------------------------------------------- #


@dataclass
class ItemOutcome:
    """What one (item, reference) pair contributed to its lane."""

    item_id: str
    reference_id: str
    source_kind: str
    language: str
    quality: str | None
    status: str
    coverage: dict[str, Any]
    errors: list[BoundaryError] = field(default_factory=list)
    excluded_reference_segments: int = 0
    reference_uncertainty_s: float | None = None
    notes: list[str] = field(default_factory=list)
    failure: dict[str, Any] | None = None
    track: dict[str, Any] | None = None

    @property
    def lane_key(self) -> tuple[str, str]:
        return (self.source_kind, self.language)


def _failure(
    code: str,
    message: str,
    *,
    item_id: str,
    reference_id: str | None = None,
    severity: str = "invalid",
    details: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "item_id": item_id,
        "reference_id": reference_id,
        "message": message,
        "details": list(details),
    }


def evaluate_reference(
    item: ItemSpec,
    ref: ReferenceSpec,
    *,
    defaults: Mapping[str, float],
    probe: Callable[[Path], list[SubtitleStream]] = probe_subtitle_streams,
    extract: Callable[[Path, int, Path, str], Path] | None = None,
) -> ItemOutcome:
    """Match one reference against the item hypothesis and reduce it to errors."""
    level = ref.level
    empty_coverage: dict[str, Any] = {}
    try:
        hyp_all = load_hypothesis_segments(
            item.hypothesis_path, language=item.language, level=level
        )
        hyp = filter_by_ranges(hyp_all, item.include_ranges, item.exclude_ranges)

        uncertainty: float | None = None
        track_info: dict[str, Any] | None = None
        if ref.path is not None:
            doc = load_reference_document(ref.path)
            if doc["language"] != ref.language:
                raise ReferenceLanguageMismatch(
                    f"reference {ref.id!r} declares {doc['language']!r} but the "
                    f"manifest lane is {ref.language!r}"
                )
            if doc["kind"] != ref.kind:
                raise cc.CalibrationError(
                    f"reference {ref.id!r}: file kind {doc['kind']!r} != manifest "
                    f"kind {ref.kind!r}"
                )
            segments, excluded = reference_segments(doc, language=ref.language)
            uncertainty = (doc.get("provenance") or {}).get("reference_uncertainty_s")
            detected = detect_text_language(" ".join(s.text for s in segments))
            if detected is not None and not cc.languages_match(detected, ref.language):
                raise ReferenceLanguageMismatch(
                    f"reference {ref.id!r}: text detects as {detected!r} but the lane "
                    f"is {ref.language!r}"
                )
        elif ref.media is not None:
            selection = select_subtitle_track(
                ref.media,
                language=ref.language,
                hypothesis_norm=joiner_for(item.language).join(s.norm for s in hyp),
                explicit_index=ref.stream_index,
                expected_codec=ref.expected_codec,
                probe=probe,
                extract=extract,
            )
            segments = make_segments(
                selection.blocks, language=ref.language, prefix="ref"
            )
            excluded = 0
            track_info = {
                "stream_index": selection.stream.index,
                "codec": selection.stream.codec,
                "language": selection.stream.language,
                "title": selection.stream.title,
                "text_coverage": selection.coverage,
                "detected_language": selection.detected_language,
                "candidates": [c.to_dict() for c in selection.candidates],
            }
        else:  # pragma: no cover - schema requires path or media
            raise cc.CalibrationError(
                f"reference {ref.id!r} has neither path nor media"
            )

        reference = filter_by_ranges(segments, item.include_ranges, item.exclude_ranges)
        result = pair_monotonic(
            hyp,
            reference,
            language=item.language,
            level=level,
            min_pair_similarity=float(defaults["min_pair_similarity"]),
        )
        coverage = coverage_of(result)
        coverage["excluded_reference_segments"] = excluded
        coverage["thresholds"] = {
            "min_hyp_coverage": float(defaults["min_hyp_coverage"]),
            "min_ref_coverage": float(defaults["min_ref_coverage"]),
            "min_pair_similarity": float(defaults["min_pair_similarity"]),
        }
        errors = boundary_errors(result, cluster_prefix=f"{item.id}|{ref.id}")
        signed_start = [e.start_signed for e in errors]
        signed_end = [e.end_signed for e in errors]
        coverage["signed_bias_s"] = {
            "start_median": cc.percentile(signed_start, 50.0),
            "end_median": cc.percentile(signed_end, 50.0),
        }

        hyp_cov = coverage["hyp_chars"]["value"]
        ref_cov = coverage["ref_chars"]["value"]
        if hyp_cov is None or ref_cov is None:
            return ItemOutcome(
                item_id=item.id,
                reference_id=ref.id,
                source_kind=ref.kind,
                language=ref.language,
                quality=ref.quality,
                status="insufficient_samples",
                coverage=coverage,
                excluded_reference_segments=excluded,
                reference_uncertainty_s=uncertainty,
                notes=["no lexical content left after normalization"],
                track=track_info,
            )
        if hyp_cov < float(defaults["min_hyp_coverage"]):
            raise InsufficientCoverage(
                f"hypothesis character coverage {hyp_cov:.3f} is below "
                f"{float(defaults['min_hyp_coverage']):.3f}; this run cannot judge "
                "alignment quality"
            )
        if ref_cov < float(defaults["min_ref_coverage"]):
            raise InsufficientCoverage(
                f"reference character coverage {ref_cov:.3f} is below "
                f"{float(defaults['min_ref_coverage']):.3f}; this run cannot judge "
                "alignment quality"
            )
    except cc.CalibrationError as exc:
        code = next(
            (name for kind, name in FAILURE_CODES if isinstance(exc, kind)),
            "invalid_measurement",
        )
        return ItemOutcome(
            item_id=item.id,
            reference_id=ref.id,
            source_kind=ref.kind,
            language=ref.language,
            quality=ref.quality,
            status="invalid",
            coverage=empty_coverage,
            failure=_failure(
                code,
                exc.message,
                item_id=item.id,
                reference_id=ref.id,
                details=exc.details,
            ),
        )

    notes: list[str] = []
    primary = sum(1 for e in errors if level != "word" or e.ref_count == 1)
    floor = MIN_WORD_SAMPLES if level == "word" else MIN_CUE_GROUPS
    if primary < floor:
        notes.append(f"only {primary} usable samples (item floor is {floor})")
    return ItemOutcome(
        item_id=item.id,
        reference_id=ref.id,
        source_kind=ref.kind,
        language=ref.language,
        quality=ref.quality,
        status="ok",
        coverage=coverage,
        errors=errors,
        excluded_reference_segments=excluded,
        reference_uncertainty_s=uncertainty,
        notes=notes,
        track=track_info,
    )


# --------------------------------------------------------------------------- #
# Lanes and report
# --------------------------------------------------------------------------- #


def _metric_pair(
    name: str,
    errors: Sequence[BoundaryError],
    *,
    thresholds: Sequence[float],
    uncertainty: float | None,
    clusters: Mapping[str, Sequence[float]] | None = None,
    bootstrap_samples: int = 0,
) -> dict[str, dict[str, Any]]:
    """Build ``<name>_start_abs_s`` / ``_end_abs_s`` / ``_pooled_abs_s`` blocks."""
    starts = [e.start_abs for e in errors]
    ends = [e.end_abs for e in errors]
    pooled = starts + ends
    ci = (
        cluster_bootstrap_ci(clusters, samples=bootstrap_samples)
        if clusters is not None
        else None
    )
    blocks: dict[str, dict[str, Any]] = {}
    for suffix, data, ci95 in (
        ("start_abs_s", starts, None),
        ("end_abs_s", ends, None),
        ("pooled_abs_s", pooled, ci),
    ):
        block = cc.metric_block(data, thresholds=thresholds, ci95=ci95)
        lb = _lower_bound(block, uncertainty)
        if lb is not None:
            block["interpretive_lower_bound"] = lb
        blocks[f"{name}{suffix}"] = block

    early_late = {
        f"{name}start_early_s": [-e.start_signed for e in errors if e.start_signed < 0],
        f"{name}start_late_s": [e.start_signed for e in errors if e.start_signed > 0],
        f"{name}end_early_s": [-e.end_signed for e in errors if e.end_signed < 0],
        f"{name}end_late_s": [e.end_signed for e in errors if e.end_signed > 0],
    }
    for key, data in early_late.items():
        blocks[key] = cc.metric_block(data, thresholds=thresholds)
    return blocks


def lane_metrics(
    level: str,
    errors: Sequence[BoundaryError],
    *,
    uncertainty: float | None,
    bootstrap_samples: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Metric blocks for one lane plus its primary sample count.

    Word lanes split by reference cardinality on purpose: a hypothesis unit
    matched against two MFA words has no interior boundary to compare, so it is
    reported as a lexical span, never as a word MAE.
    """
    if level == "cue":
        blocks = _metric_pair(
            "",
            errors,
            thresholds=CUE_THRESHOLDS,
            uncertainty=uncertainty,
            clusters=_clusters(errors),
            bootstrap_samples=bootstrap_samples,
        )
        return blocks, len(errors)

    single = [e for e in errors if e.ref_count == 1]
    spans = [e for e in errors if e.ref_count > 1]
    blocks = _metric_pair(
        "word_",
        single,
        thresholds=WORD_THRESHOLDS,
        uncertainty=uncertainty,
        clusters=_clusters(single),
        bootstrap_samples=bootstrap_samples,
    )
    blocks.update(
        _metric_pair(
            "lexical_span_",
            spans,
            thresholds=WORD_THRESHOLDS,
            uncertainty=None,
        )
    )
    return blocks, len(single)


def _clusters(errors: Sequence[BoundaryError]) -> dict[str, list[float]]:
    """Group boundary samples by utterance so the bootstrap resamples sentences."""
    out: dict[str, list[float]] = {}
    for e in errors:
        out.setdefault(e.cluster, []).extend((e.start_abs, e.end_abs))
    return out


def _merge_coverage(outcomes: Sequence[ItemOutcome]) -> dict[str, Any]:
    """Micro-aggregate lane coverage: sum numerators and denominators."""
    counters = {
        "hyp_chars": [0, 0],
        "ref_chars": [0, 0],
        "hyp_segments": [0, 0],
        "ref_segments": [0, 0],
    }
    unmatched = {
        "hyp_unmatched_segments": 0,
        "ref_unmatched_segments": 0,
        "hyp_unmatched_chars": 0,
        "ref_unmatched_chars": 0,
    }
    empty = {"hypothesis": 0, "reference": 0}
    shapes = {"1:1": 0, "1:N": 0, "N:1": 0, "N:M": 0}
    groups = 0
    excluded = 0
    for outcome in outcomes:
        cov = outcome.coverage
        if not cov:
            continue
        for key, slot in counters.items():
            slot[0] += int(cov[key]["matched"])
            slot[1] += int(cov[key]["total"])
        for key in unmatched:
            unmatched[key] += int(cov.get(key, 0))
        for side in empty:
            empty[side] += int(cov["empty_after_normalization"][side])
        for shape in shapes:
            shapes[shape] += int(cov["match_shapes"][shape])
        groups += int(cov.get("groups", 0))
        excluded += int(cov.get("excluded_reference_segments", 0))
    merged: dict[str, Any] = {
        key: _fraction(slot[0], slot[1]) for key, slot in counters.items()
    }
    merged.update(unmatched)
    merged["empty_after_normalization"] = empty
    merged["match_shapes"] = shapes
    merged["groups"] = groups
    merged["excluded_reference_segments"] = excluded
    return merged


def build_lanes(
    outcomes: Sequence[ItemOutcome],
    *,
    bootstrap_samples: int,
    pairs: str = "worst",
    pairs_limit: int = 25,
) -> list[dict[str, Any]]:
    """Group outcomes into ``(source_kind, language)`` lanes, in a fixed order."""
    by_lane: dict[tuple[str, str], list[ItemOutcome]] = {}
    for outcome in outcomes:
        by_lane.setdefault(outcome.lane_key, []).append(outcome)

    lanes: list[dict[str, Any]] = []
    for source_kind, language in sorted(by_lane):
        members = sorted(by_lane[(source_kind, language)], key=lambda o: o.item_id)
        level = "word" if source_kind in WORD_KINDS else "cue"
        usable = [m for m in members if m.status != "invalid"]
        errors = [e for m in usable for e in m.errors]
        uncertainties = [
            m.reference_uncertainty_s
            for m in usable
            if m.reference_uncertainty_s is not None
        ]
        uncertainty = max(uncertainties) if uncertainties else None
        metrics, primary = lane_metrics(
            level,
            errors,
            uncertainty=uncertainty,
            bootstrap_samples=bootstrap_samples,
        )
        coverage = _merge_coverage(usable)
        coverage["signed_bias_s"] = {
            "start_median": cc.percentile([e.start_signed for e in errors], 50.0),
            "end_median": cc.percentile([e.end_signed for e in errors], 50.0),
        }
        coverage["reference_uncertainty_s"] = uncertainty
        coverage["sample_floor"] = (
            MIN_WORD_SAMPLES if level == "word" else MIN_CUE_GROUPS
        )

        if any(m.status == "invalid" for m in members):
            status = "invalid"
        elif primary < coverage["sample_floor"]:
            status = "insufficient_samples"
        else:
            status = "pass"

        reference_ids = sorted({m.reference_id for m in members})
        lanes.append(
            {
                "source_kind": source_kind,
                "language": language,
                "reference_id": reference_ids[0] if len(reference_ids) == 1 else None,
                "status": status,
                "coverage": coverage,
                "metrics": metrics,
                "items": [
                    _item_detail(m, level, pairs=pairs, pairs_limit=pairs_limit)
                    for m in members
                ],
            }
        )
    return lanes


def _pair_details(
    errors: Sequence[BoundaryError], *, pairs: str, limit: int
) -> dict[str, Any]:
    """Per-group detail for manual spot-checking (design 3.6).

    ``worst`` (the default) keeps the largest boundary errors, which is what a
    human actually opens the report for; the selection is named, and
    ``pairs_total`` states how many groups exist, so a truncated list can never
    read as the whole population. ``all`` keeps every group, ``none`` keeps none.
    """
    if pairs == "none":
        return {"pairs_total": len(errors), "pairs_kept": "none"}
    if pairs == "all":
        return {
            "pairs_total": len(errors),
            "pairs_kept": "all",
            "pairs": [e.to_dict() for e in errors],
        }
    ranked = sorted(
        errors,
        key=lambda e: (-max(e.start_abs, e.end_abs), e.hyp_ids, e.ref_ids),
    )[:limit]
    return {
        "pairs_total": len(errors),
        "pairs_kept": f"worst_{limit}",
        "worst_pairs": [e.to_dict() for e in ranked],
    }


def _item_detail(
    outcome: ItemOutcome, level: str, *, pairs: str, pairs_limit: int
) -> dict[str, Any]:
    """Per-item stratification kept inside the lane (design 3.1)."""
    metrics: dict[str, Any] = {}
    if outcome.status != "invalid":
        metrics, _ = lane_metrics(
            level,
            outcome.errors,
            uncertainty=outcome.reference_uncertainty_s,
            bootstrap_samples=0,
        )
    detail = {
        "item_id": outcome.item_id,
        "reference_id": outcome.reference_id,
        "reference_quality": outcome.quality,
        "status": outcome.status,
        "coverage": outcome.coverage,
        "metrics": metrics,
        "notes": outcome.notes,
        "track": outcome.track,
    }
    detail.update(_pair_details(outcome.errors, pairs=pairs, limit=pairs_limit))
    return detail


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_commit() -> str | None:
    """Commit id from the environment only -- this harness never shells out to git."""
    for name in ("VOXWEAVE_GIT_COMMIT", "GITHUB_SHA", "CI_COMMIT_SHA"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def build_report(
    manifest: Manifest,
    outcomes: Sequence[ItemOutcome],
    *,
    failures: Sequence[Mapping[str, Any]] = (),
    pairs: str = "worst",
    pairs_limit: int = 25,
) -> dict[str, Any]:
    """Assemble the report document and validate it against its tracked schema."""
    bootstrap = int(manifest.defaults["bootstrap_samples"])
    lanes = build_lanes(
        outcomes,
        bootstrap_samples=bootstrap,
        pairs=pairs,
        pairs_limit=pairs_limit,
    )
    all_failures = [dict(f) for f in failures]
    all_failures.extend(dict(o.failure) for o in outcomes if o.failure is not None)
    invalid = any(f.get("severity") == "invalid" for f in all_failures)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "manifest_digest": manifest.digest,
        "generated_at": _now(),
        "git_commit": _git_commit(),
        "status": "invalid" if invalid else "pass",
        "lanes": lanes,
        "health": {
            "config": {
                "text_norm_version": TEXT_NORM_VERSION,
                # Name only: the digest is the manifest's identity, and an
                # absolute path would make two machines' reports differ.
                "manifest": manifest.path.name,
                "defaults": dict(manifest.defaults),
                "group_penalty": GROUP_PENALTY,
                "skip_cost": SKIP_COST,
                "merge_similarity_step": MERGE_SIMILARITY_STEP,
                "max_group": {k: list(v) for k, v in MAX_GROUP.items()},
            }
        },
        "failures": all_failures,
    }
    errors = cc.schema_errors(report, "alignment-report")
    if errors:  # pragma: no cover - a bug in this file, not in the input
        raise cc.CalibrationError("generated report failed schema validation", errors)
    return report


def evaluate(
    manifest: Manifest,
    *,
    source_filter: Sequence[str] = (),
    item_filter: Sequence[str] = (),
    probe: Callable[[Path], list[SubtitleStream]] = probe_subtitle_streams,
    extract: Callable[[Path, int, Path, str], Path] | None = None,
    pairs: str = "worst",
    pairs_limit: int = 25,
) -> dict[str, Any]:
    """Run every enabled (item, reference) pair and build the report."""
    outcomes: list[ItemOutcome] = []
    failures: list[dict[str, Any]] = []
    selected = 0
    for item in manifest.items:
        if item_filter and item.id not in item_filter:
            continue
        for ref in item.references:
            if not ref.enabled:
                continue
            if source_filter and ref.kind not in source_filter:
                continue
            selected += 1
            outcomes.append(
                evaluate_reference(
                    item,
                    ref,
                    defaults=manifest.defaults,
                    probe=probe,
                    extract=extract,
                )
            )
    if not selected:
        failures.append(
            _failure(
                "no_references_selected",
                "the manifest filters selected no enabled reference",
                item_id="<manifest>",
            )
        )
    return build_report(
        manifest,
        outcomes,
        failures=failures,
        pairs=pairs,
        pairs_limit=pairs_limit,
    )


# --------------------------------------------------------------------------- #
# Baseline and one-way gates
# --------------------------------------------------------------------------- #


def baseline_document(
    report: Mapping[str, Any], *, tolerances: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Reduce a report to the numbers ``check`` compares against.

    Only the gated absolute-error blocks and the two coverage rates are
    recorded: a baseline is the gate contract, not a second copy of the report.
    """
    tol = dict(DEFAULT_TOLERANCES)
    tol.update(tolerances or {})
    lanes = []
    for lane in report["lanes"]:
        metrics = {}
        for name, block in sorted(lane["metrics"].items()):
            if not block.get("n") or not is_gated_metric(name):
                continue
            metrics[name] = {
                key: block.get(key)
                for key in ("n", *GATED_ERROR_FIELDS, *GATED_RATE_FIELDS)
                if block.get(key) is not None
            }
        lanes.append(
            {
                "source_kind": lane["source_kind"],
                "language": lane["language"],
                "status": lane["status"],
                "coverage": {
                    "hyp_chars": lane["coverage"]["hyp_chars"]["value"],
                    "ref_chars": lane["coverage"]["ref_chars"]["value"],
                },
                "metrics": metrics,
            }
        )
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "metric_definition_version": report["metric_definition_version"],
        "text_norm_version": TEXT_NORM_VERSION,
        "manifest_digest": report["manifest_digest"],
        "recorded_at": _now(),
        "tolerances": tol,
        "lanes": lanes,
    }


def _allowed_ceiling(base: float, tol: Mapping[str, float]) -> float:
    return base + max(float(tol["absolute_s"]), abs(base) * float(tol["relative"]))


def _allowed_floor(base: float, tol: Mapping[str, float], absolute: float) -> float:
    return base - max(absolute, abs(base) * float(tol["relative"]))


def apply_gates(
    report: dict[str, Any], baseline: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Compare a report against a baseline with one-way gates.

    Every gate is one-sided in the direction that means "worse": error metrics
    may not rise, hit rates and coverage may not fall. An improvement can never
    fail. A structural mismatch (different manifest digest, a lane that
    disappeared, a metric whose samples vanished) is *invalid*, not a
    regression -- a run with different inputs has no standing to judge quality.
    """
    if int(baseline.get("schema_version", 0)) != BASELINE_SCHEMA_VERSION:
        raise cc.CalibrationError(
            f"baseline schema_version {baseline.get('schema_version')!r} != "
            f"{BASELINE_SCHEMA_VERSION}"
        )
    if baseline.get("metric_definition_version") != report["metric_definition_version"]:
        raise cc.CalibrationError(
            "baseline metric_definition_version "
            f"{baseline.get('metric_definition_version')!r} != report "
            f"{report['metric_definition_version']!r}; re-record deliberately"
        )
    if int(baseline.get("text_norm_version", -1)) != TEXT_NORM_VERSION:
        raise cc.CalibrationError(
            f"baseline text_norm_version {baseline.get('text_norm_version')!r} != "
            f"{TEXT_NORM_VERSION}; normalization changed, re-record deliberately"
        )
    if baseline.get("manifest_digest") != report["manifest_digest"]:
        raise cc.CalibrationError(
            "manifest digest differs from the baseline; the corpus changed, so this "
            "run cannot be compared against it"
        )

    tol = dict(DEFAULT_TOLERANCES)
    tol.update(baseline.get("tolerances") or {})
    current = {
        (lane["source_kind"], lane["language"]): lane for lane in report["lanes"]
    }
    failures: list[dict[str, Any]] = []

    for base_lane in baseline["lanes"]:
        key = (base_lane["source_kind"], base_lane["language"])
        lane = current.get(key)
        if lane is None:
            raise cc.CalibrationError(
                f"lane {key[0]}/{key[1]} is in the baseline but missing from the report"
            )
        label = f"{key[0]}/{key[1]}"
        base_cov = base_lane.get("coverage") or {}
        for side in ("hyp_chars", "ref_chars"):
            base_value = base_cov.get(side)
            now = lane["coverage"][side]["value"]
            if base_value is None:
                continue
            if now is None:
                raise cc.CalibrationError(
                    f"{label}: {side} coverage disappeared; the run is not comparable"
                )
            floor = _allowed_floor(
                float(base_value), tol, float(tol["coverage_absolute"])
            )
            if now < floor:
                failures.append(
                    _gate_failure(
                        label, f"coverage.{side}", now, base_value, floor, "min"
                    )
                )

        for name, base_block in sorted(base_lane["metrics"].items()):
            if not is_gated_metric(name):
                continue
            block = lane["metrics"].get(name)
            if block is None or not block.get("n"):
                raise cc.CalibrationError(
                    f"{label}: metric {name!r} has no samples in this report but the "
                    "baseline recorded some; the run is not comparable"
                )
            for metric_field in GATED_ERROR_FIELDS:
                base_value = base_block.get(metric_field)
                now = block.get(metric_field)
                if base_value is None or now is None:
                    continue
                ceiling = _allowed_ceiling(float(base_value), tol)
                if float(now) > ceiling:
                    failures.append(
                        _gate_failure(
                            label,
                            f"{name}.{metric_field}",
                            now,
                            base_value,
                            ceiling,
                            "max",
                        )
                    )
            for metric_field in GATED_RATE_FIELDS:
                base_value = base_block.get(metric_field)
                now = block.get(metric_field)
                if base_value is None or now is None:
                    continue
                floor = _allowed_floor(
                    float(base_value), tol, float(tol["rate_absolute"])
                )
                if float(now) < floor:
                    failures.append(
                        _gate_failure(
                            label,
                            f"{name}.{metric_field}",
                            now,
                            base_value,
                            floor,
                            "min",
                        )
                    )

    if failures:
        failed_lanes = {f["lane"] for f in failures}
        for lane in report["lanes"]:
            label = f"{lane['source_kind']}/{lane['language']}"
            if label in failed_lanes and lane["status"] != "invalid":
                lane["status"] = "fail"
        report["failures"].extend(failures)
        if report["status"] != "invalid":
            report["status"] = "fail"
    return failures


def _gate_failure(
    lane: str,
    metric: str,
    value: float,
    baseline_value: float,
    bound: float,
    direction: str,
) -> dict[str, Any]:
    return {
        "code": "gate_regression",
        "severity": "gate",
        "lane": lane,
        "metric": metric,
        "value": float(value),
        "baseline": float(baseline_value),
        "allowed": float(bound),
        "direction": direction,
        "message": (
            f"{lane}: {metric} = {value:.6g}, baseline {baseline_value:.6g}, "
            f"allowed {'<=' if direction == 'max' else '>='} {bound:.6g}"
        ),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DEFAULT_REPORT_PATH = REPO_ROOT / "build" / "calibration" / "alignment-report.json"


def _fmt(value: Any, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(float(value), spec)


def print_report(report: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    """Human summary: error next to coverage, one block per lane, never pooled."""
    print(f"status: {report['status']}", file=stream)
    for lane in report["lanes"]:
        cov = lane["coverage"]
        print(
            f"\n[{lane['source_kind']}/{lane['language']}] {lane['status']}"
            f"  items={len(lane['items'])}"
            f"  hyp_cov={_fmt(cov['hyp_chars']['value'])}"
            f" ({cov['hyp_chars']['matched']}/{cov['hyp_chars']['total']} chars)"
            f"  ref_cov={_fmt(cov['ref_chars']['value'])}"
            f" ({cov['ref_chars']['matched']}/{cov['ref_chars']['total']} chars)",
            file=stream,
        )
        shapes = ", ".join(f"{k}={v}" for k, v in cov["match_shapes"].items())
        print(
            f"  shapes: {shapes}"
            f"  unmatched: hyp={cov['hyp_unmatched_segments']}"
            f" ref={cov['ref_unmatched_segments']}",
            file=stream,
        )
        for name, block in sorted(lane["metrics"].items()):
            if not block.get("n"):
                continue
            rates = " ".join(
                f"{key}={_fmt(block[key])}"
                for key in GATED_RATE_FIELDS
                if block.get(key) is not None
            )
            print(
                f"  {name}: n={block['n']} mae={_fmt(block.get('mae'), '.4f')}"
                f" median={_fmt(block.get('median'), '.4f')}"
                f" p90={_fmt(block.get('p90'), '.4f')} {rates}".rstrip(),
                file=stream,
            )
    for failure in report["failures"]:
        print(
            f"! {failure['severity']}: {failure.get('code')}: {failure['message']}",
            file=stream,
        )


def _load_manifest_or_die(path: str) -> Manifest:
    try:
        return load_manifest(path)
    except cc.CalibrationError as exc:
        cc.die_invalid(exc.message, exc.details)


def cmd_inspect_tracks(args: argparse.Namespace) -> int:
    """List subtitle streams and why each one would or would not be selected."""
    language = cc.require_calibration_language(args.lang)
    media = Path(args.media)
    if not media.exists():
        cc.die_invalid(f"media {media} does not exist")
    streams = probe_subtitle_streams(media)
    hypothesis_norm = ""
    if args.hypothesis:
        hyp = load_hypothesis_segments(
            Path(args.hypothesis), language=language, level="cue"
        )
        hypothesis_norm = joiner_for(language).join(s.norm for s in hyp)

    rows: list[dict[str, Any]] = []
    for stream in streams:
        reason = _static_rejection(stream, language, None)
        candidate = TrackCandidate(stream, reason)
        if reason is None and hypothesis_norm:
            blocks = _read_track(
                media,
                stream,
                lambda m, i, d, c: extract_subtitle_track(m, i, d, codec=c),
            )
            text = " ".join(str(b.get("text", "")) for b in blocks)
            detected = detect_text_language(text)
            if detected is not None and not cc.languages_match(detected, language):
                candidate = TrackCandidate(
                    stream, f"detected_language({detected})", None, detected
                )
            else:
                candidate = TrackCandidate(
                    stream,
                    None,
                    text_coverage(
                        hypothesis_norm, normalize_text(text, language), language
                    ),
                    detected,
                )
        rows.append(candidate.to_dict())

    if args.json:
        print(
            json.dumps(
                {"media": str(media), "streams": rows}, ensure_ascii=False, indent=2
            )
        )
    else:
        for row in rows:
            verdict = row["rejected"] or (
                f"candidate (coverage {_fmt(row['coverage'])})"
                if row["coverage"] is not None
                else "candidate"
            )
            print(
                f"0:{row['index']}  {row['codec']:<18} "
                f"lang={row['language']!s:<5} "
                f"forced={int(row['forced'])} default={int(row['default'])} "
                f"title={row['title']!r} -> {verdict}"
            )
    return cc.EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    """Compute the report and write it; gates only run when a baseline is given."""
    manifest = _load_manifest_or_die(args.manifest)
    report = evaluate(
        manifest,
        source_filter=tuple(args.source or ()),
        item_filter=tuple(args.item or ()),
        pairs=args.pairs,
        pairs_limit=args.pairs_limit,
    )
    out = Path(args.json_out) if args.json_out else DEFAULT_REPORT_PATH
    cc.write_json(out, report)
    print_report(report)
    print(f"\nwrote {out}")
    if report["status"] == "invalid":
        cc.die_invalid(
            "the run is invalid; it has no standing to judge alignment quality",
            [f["message"] for f in report["failures"]],
        )
    return cc.EXIT_OK


def cmd_check(args: argparse.Namespace) -> int:
    """Compare a report against a recorded baseline with one-way gates."""
    manifest = _load_manifest_or_die(args.manifest)
    if args.report:
        report = cc.read_json_or_exit2(args.report)
        errors = cc.schema_errors(report, "alignment-report")
        if errors:
            cc.die_invalid(f"{args.report} failed schema validation", errors)
    else:
        report = evaluate(
            manifest,
            source_filter=tuple(args.source or ()),
            item_filter=tuple(args.item or ()),
            pairs=args.pairs,
            pairs_limit=args.pairs_limit,
        )
    if report["status"] == "invalid":
        if args.json_out:
            cc.write_json(args.json_out, report)
        print_report(report)
        cc.die_invalid("the run is invalid; it cannot judge quality")

    baseline = cc.read_json_or_exit2(args.baseline)
    failures = apply_gates(report, baseline)
    if args.json_out:
        cc.write_json(args.json_out, report)
    print_report(report)
    if failures:
        cc.die_gate(
            f"{len(failures)} alignment gate(s) regressed",
            [f["message"] for f in failures],
        )
    return cc.EXIT_OK


def cmd_record_baseline(args: argparse.Namespace) -> int:
    """Record a new baseline. Explicit, reviewed, and never run by CI."""
    manifest = _load_manifest_or_die(args.manifest)
    report = cc.read_json_or_exit2(args.report)
    errors = cc.schema_errors(report, "alignment-report")
    if errors:
        cc.die_invalid(f"{args.report} failed schema validation", errors)
    if report["status"] == "invalid":
        cc.die_invalid("refusing to record a baseline from an invalid report")
    if report["manifest_digest"] != manifest.digest:
        cc.die_invalid(
            "report manifest_digest does not match the manifest; re-run the report "
            "against this manifest before recording a baseline",
            [f"report:   {report['manifest_digest']}", f"manifest: {manifest.digest}"],
        )
    baseline = baseline_document(report)
    cc.write_json(args.output, baseline)
    print(f"recorded baseline from {args.report} -> {args.output}")
    for lane in baseline["lanes"]:
        gated = ", ".join(sorted(lane["metrics"]))
        print(f"  {lane['source_kind']}/{lane['language']} [{lane['status']}]: {gated}")
    return cc.EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calib_alignment.py",
        description=(
            "Alignment accuracy ruler. Lanes are kept separate per "
            "(source_kind, language) and pairing never reads a timestamp."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect-tracks",
        help="list subtitle streams and the selection verdict for each",
    )
    inspect.add_argument("media")
    inspect.add_argument("--lang", required=True)
    inspect.add_argument(
        "--hypothesis", help="hypothesis subtitle/JSON, enables text coverage ranking"
    )
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=cmd_inspect_tracks)

    def add_shared(target: argparse.ArgumentParser) -> None:
        target.add_argument("--manifest", required=True)
        target.add_argument("--json-out")
        target.add_argument(
            "--source", action="append", choices=[*WORD_KINDS, *CUE_KINDS]
        )
        target.add_argument("--item", action="append")
        target.add_argument(
            "--pairs",
            choices=("none", "worst", "all"),
            default="worst",
            help="per-group detail kept for manual spot-checking (default: worst)",
        )
        target.add_argument("--pairs-limit", type=int, default=25)

    report = sub.add_parser("report", help="compute the alignment report and write it")
    add_shared(report)
    report.set_defaults(func=cmd_report)

    check = sub.add_parser("check", help="gate a report against a recorded baseline")
    add_shared(check)
    check.add_argument("--baseline", required=True)
    check.add_argument(
        "--report", help="reuse an existing report instead of recomputing"
    )
    check.set_defaults(func=cmd_check)

    record = sub.add_parser(
        "record-baseline",
        help="record a new baseline from a report (human action, never CI)",
    )
    record.add_argument("--manifest", required=True)
    record.add_argument("--report", required=True)
    record.add_argument("--output", required=True)
    record.set_defaults(func=cmd_record_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    cc.run_cli(main)
