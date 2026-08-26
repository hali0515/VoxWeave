"""Hard-contract validator over a LOCKED partition and the cues it produced.

This module deliberately knows nothing about how the partition was found: it
consumes a cut list plus the finished cue stream and re-derives every hard
predicate from the source units alone. That independence is the point. A solver
that validates itself proves only that it agrees with itself, and P6 hands an
*aligner*-locked partition through this same seam, so the checker must not need
a lattice, a cost model or a solver to have an opinion.

Two contracts here carry weight beyond P4:

* **both speech anchors are checked.** Guarding only the out-time is the classic
  half-measure -- it cannot see the shot-snap class of damage, where a cue's
  display *start* is pushed past the first word that was actually spoken.
* **attribution is typed.** Every violation records where the partition came
  from (``origin``) and which post-transform stage produced the cue stream
  (``stage``). Only unwaived ``v2`` violations at the ``raw``/``core`` stages may
  fail a shadow run: v1's own damage and the legacy display overlay's damage are
  evidence for the next phase, not a reason to reject an otherwise legal v2
  partition.

Waivers are declared exemptions, never silent ones: each records the span and
cap it exempts, so a reader can re-derive the exemption instead of trusting its
label.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard

from .layout import (
    _fits_budget,
    _join,
    _line_budget_width,
    _vis_width,
    strip_punct_for_subtitles,
)
from .schema import Cue
from .segdoc import DisplayProfile, SourceUnit
from .smart_split import _display_chars

#: The single comparison tolerance of this module. Stated once so no call site
#: can drift: it is deliberately coarser than the lattice's own duration
#: epsilon, which means the validator can never contradict an edge the solver
#: already certified as legal.
EPS: float = 1e-6

Origin = Literal["v1", "v2"]
Stage = Literal["raw", "core", "legacy-overlay"]

ORIGINS: tuple[Origin, ...] = ("v1", "v2")
STAGES: tuple[Stage, ...] = ("raw", "core", "legacy-overlay")

#: Closed vocabulary of hard-contract failures, sorted so artifact bytes are
#: stable and a reader can diff two runs' violation sets directly.
VIOLATION_KINDS: tuple[str, ...] = (
    "duration-cap",
    "line-capacity",
    "non-finite-time",
    "non-monotone-time",
    "overlap",
    "reversed-cue",
    "speech-truncated-end",
    "speech-truncated-start",
    "text-conservation",
    "unit-conservation",
)

#: Closed vocabulary of declared exemptions. Only the held-word duration waiver
#: exists today; the tuple is the whitelist a new one has to join explicitly.
WAIVER_KINDS: tuple[str, ...] = ("held-chain-duration",)

#: Stages whose unwaived ``v2`` violations drive the shadow exit.
EXIT_DRIVING_STAGES: frozenset[str] = frozenset({"raw", "core"})


def normalize_text(text: str) -> str:
    """The display-equivalence normal form both sides of conservation are read in.

    Cue text is finalized *after* the units are frozen, and three transforms sit
    between a unit stream and the rendered string: punctuation stripping (whose
    ``.``/``,`` rule is context-sensitive on the *joined* stream), stutter
    hyphenation, and the wrap pass' inserted newlines. Reusing
    ``smart_split._display_chars`` rather than restating those rules is what
    keeps the checker from disagreeing with the renderer about which characters
    survive; newlines are folded to spaces first because a wrap is layout, not
    content.
    """
    return "".join(_display_chars([text.replace("\n", " ")]))


def owned_unit_ids(
    partition: Sequence[int], unit_count: int
) -> tuple[tuple[int, int], ...]:
    """Half-open source-unit range per cue, from the interior cut list."""
    bounds = (0, *partition, unit_count)
    return tuple((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))


@dataclass(frozen=True)
class Waiver:
    """A declared, evidence-backed exemption from one hard predicate.

    Lives in the validator rather than in the lattice so a later phase can import
    the waiver vocabulary without importing a solver. ``span`` and ``cap`` are
    recorded so the exemption can be re-derived from the artifact instead of
    being taken on trust.
    """

    kind: str
    cue_index: int
    unit_ids: tuple[int, ...]
    span: tuple[float | None, float | None]
    cap: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cap": self.cap,
            "cue_index": self.cue_index,
            "detail": self.detail,
            "kind": self.kind,
            "span": list(self.span),
            "unit_ids": list(self.unit_ids),
        }


@dataclass(frozen=True)
class Violation:
    """One hard-contract failure, with the provenance that decides its weight."""

    kind: str
    origin: Origin
    stage: Stage
    cue_index: int | None
    unit_ids: tuple[int, ...]
    detail: str
    waived_by: Waiver | None = None

    @property
    def waived(self) -> bool:
        return self.waived_by is not None

    @property
    def exit_driving(self) -> bool:
        """Only unwaived v2 damage at the raw/core stages may fail the run."""
        return (
            not self.waived
            and self.origin == "v2"
            and self.stage in EXIT_DRIVING_STAGES
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_index": self.cue_index,
            "detail": self.detail,
            "exit_driving": self.exit_driving,
            "kind": self.kind,
            "origin": self.origin,
            "stage": self.stage,
            "unit_ids": list(self.unit_ids),
            "waived": self.waived,
            "waived_by": None if self.waived_by is None else self.waived_by.to_dict(),
        }


@dataclass(frozen=True)
class PartitionCheckResult:
    """Every violation one (origin, stage) pass found, plus its waiver ledger."""

    origin: Origin
    stage: Stage
    violations: tuple[Violation, ...]
    waivers: tuple[Waiver, ...]
    cue_count: int
    unit_count: int

    @property
    def unwaived(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if not v.waived)

    @property
    def exit_driving(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.exit_driving)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_count": self.cue_count,
            "exit_driving": len(self.exit_driving),
            "origin": self.origin,
            "stage": self.stage,
            "unit_count": self.unit_count,
            "unwaived": len(self.unwaived),
            "violations": [v.to_dict() for v in self.violations],
            "waivers": [w.to_dict() for w in self.waivers],
        }


def _is_real(value: Any) -> TypeGuard[float]:
    """A finite, non-bool number -- the only shape a display time may take."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _wrappable(text: str, profile: DisplayProfile, lang: str) -> bool:
    """Could the wrap pass still fold this text into the line budget?

    This is a DECLARED widening of the pinned predicate. ``p4-api.md`` section 1
    item 8 states line capacity as ``len(lines) <= max_lines`` and
    ``_vis_width(line) <= budget`` on the final rendered text, with only empty
    text exempt. Taken literally that rule fails every legal cue at the ``raw``
    stage, because a raw cue's text is the unstripped, unwrapped join -- and
    AD4-1's all-invisible cue (raw text: two thousand ``!``) would fail it too,
    though the spec requires that cue to pass. So the predicate is split by
    whether layout has run:

    * a text carrying a newline is a committed wrap decision and is measured line
      by line exactly as delivered -- the literal rule;
    * a text with no newline has not been through the wrap pass, so the honest
      question is whether it *can* fit, asked of the same oracle that certified
      the candidate cue in the first place.

    Asking the delivered single line instead would report a violation for every
    two-line cue in the stream and, worse, would contradict an edge the solver
    already certified as legal. The projection is stripped first for the same
    reason: punctuation that renders to nothing must not be charged for width.
    """
    if "\n" in text:
        return False
    return _fits_budget(
        strip_punct_for_subtitles(text),
        profile.max_line_length,
        profile.max_lines,
        lang,
    )


def _whitelisted(waiver: Waiver | None) -> bool:
    """Only a declared exemption kind exempts anything.

    :data:`WAIVER_KINDS` is a whitelist a new exemption has to join explicitly,
    which it cannot be if an unrecognised label waives just as well as a known
    one. An off-whitelist waiver is still carried in the ledger -- the record
    stays complete -- it simply does not suppress the violation.
    """
    return waiver is not None and waiver.kind in WAIVER_KINDS


def _clamp_range(lo: Any, hi: Any, unit_count: int) -> tuple[int, int]:
    """A slice that is always well-formed, however malformed the partition was.

    A malformed partition is reported once as ``unit-conservation``; the per-cue
    checks still run so one bad cut point does not hide every other defect in
    the same stream.
    """
    try:
        low = int(lo)
        high = int(hi)
    except (TypeError, ValueError):
        return unit_count, unit_count
    low = min(max(low, 0), unit_count)
    high = min(max(high, low), unit_count)
    return low, high


def check_partition(
    partition: Sequence[int],
    cues: Sequence[Cue],
    *,
    units: Sequence[SourceUnit],
    profile: DisplayProfile,
    origin: Origin,
    stage: Stage,
    waivers: Mapping[int, Waiver] | None = None,
    origins: Mapping[int, Origin] | None = None,
    expect_no_overlap: bool = True,
) -> PartitionCheckResult:
    """Run every hard predicate over one locked partition and its cue stream.

    ``partition`` is the interior cut points in source-unit space, strictly
    increasing inside ``(0, len(units))``; cue ``k`` owns
    ``units[bounds[k]:bounds[k + 1]]`` for ``bounds = (0, *partition,
    len(units))``. ``waivers`` maps a cue index to the exemption that covers it;
    a waiver handed in for a cue that violates nothing is still reported (so the
    ledger is complete) but waives nothing. Only a waiver whose ``kind`` is in
    :data:`WAIVER_KINDS` exempts anything: the whitelist is the point of having
    one, and an unknown label is reported as an unwaived violation rather than
    silently granted.

    ``origins`` overrides ``origin`` per cue index. AD3-3 types attribution per
    *violation* -- "which engine produced the cue that violated" -- and a document
    stream can be mixed, because a typed fallback splices v1's own cues into an
    otherwise v2 partition. Attributing the whole stream to whichever engine
    dominated would either excuse real v2 damage (everything reads v1) or blame
    v2 for v1's (everything reads v2); both have been observed. ``origin`` stays
    the default for cues the mapping does not name and for the whole-partition
    row, which belongs to no single cue.

    The checks are independent by construction -- one cue can raise several --
    except that a cue whose own display times are not real numbers skips the
    comparisons those times would poison, since a NaN compares false against
    everything and would otherwise manufacture a violation on both sides.
    """
    supplied = dict(waivers or {})
    by_cue: dict[int, Origin] = dict(origins or {})
    unit_count = len(units)
    lang = profile.language
    violations: list[Violation] = []

    def report(
        kind: str,
        cue_index: int | None,
        unit_ids: tuple[int, ...],
        detail: str,
        waived_by: Waiver | None = None,
    ) -> None:
        violations.append(
            Violation(
                kind=kind,
                origin=origin if cue_index is None else by_cue.get(cue_index, origin),
                stage=stage,
                cue_index=cue_index,
                unit_ids=unit_ids,
                detail=detail,
                waived_by=waived_by,
            )
        )

    cut_list = list(partition)
    well_formed = (
        all(isinstance(c, int) and not isinstance(c, bool) for c in cut_list)
        and all(0 < int(c) < unit_count for c in cut_list)
        and all(int(a) < int(b) for a, b in zip(cut_list, cut_list[1:]))
        and len(cues) == len(cut_list) + 1
    )
    if not well_formed:
        report(
            "unit-conservation",
            None,
            (),
            f"partition {cut_list!r} does not tile {unit_count} units into "
            f"{len(cues)} cues",
        )

    bounds = owned_unit_ids(cut_list, unit_count)
    budget = _line_budget_width(profile.max_line_length, lang)
    cap = profile.max_cue_s

    for index, cue in enumerate(cues):
        raw_range = bounds[index] if index < len(bounds) else (unit_count, unit_count)
        lo, hi = _clamp_range(raw_range[0], raw_range[1], unit_count)
        unit_ids = tuple(range(lo, hi))

        expected = normalize_text(_join([units[i].surface for i in unit_ids], lang))
        actual = normalize_text(str(cue.get("text", "")))
        if expected != actual:
            report(
                "text-conservation",
                index,
                unit_ids,
                f"owned units normalize to {expected!r}, cue text to {actual!r}",
            )

        start = cue.get("start")
        end = cue.get("end")
        timed = _is_real(start) and _is_real(end)
        if not timed:
            report(
                "non-finite-time",
                index,
                unit_ids,
                f"start={start!r} end={end!r} is not a finite pair",
            )
        else:
            if end < start - EPS:
                report(
                    "reversed-cue", index, unit_ids, f"end {end} precedes start {start}"
                )

            if index > 0:
                prev = cues[index - 1]
                p_start = prev.get("start")
                p_end = prev.get("end")
                if _is_real(p_start) and _is_real(p_end):
                    # Exactly p4-api.md section 1 item 5: monotone *starts*. A
                    # start that sits between the previous cue's start and its
                    # end is an ``overlap`` and nothing else -- reporting it as
                    # both doubled every overlap in the ledger, which would make
                    # any future count-based v1/v2 comparison read 2x for this
                    # one class.
                    if start < p_start - EPS:
                        report(
                            "non-monotone-time",
                            index,
                            unit_ids,
                            f"start {start} precedes the previous cue's "
                            f"start {p_start}",
                        )
                    if expect_no_overlap and start < p_end - EPS:
                        report(
                            "overlap",
                            index,
                            unit_ids,
                            f"start {start} overlaps the previous cue's end {p_end}",
                        )

            speech_start = cue.get("speech_start")
            if _is_real(speech_start) and start > speech_start + EPS:
                report(
                    "speech-truncated-start",
                    index,
                    unit_ids,
                    f"display start {start} is later than speech start {speech_start}",
                )
            speech_end = cue.get("speech_end")
            if _is_real(speech_end) and end < speech_end - EPS:
                report(
                    "speech-truncated-end",
                    index,
                    unit_ids,
                    f"display end {end} is earlier than speech end {speech_end}",
                )

        text = str(cue.get("text", ""))
        lines = text.split("\n")
        widths = [_vis_width(line) for line in lines]
        if len(lines) > profile.max_lines or (
            any(w > budget for w in widths) and not _wrappable(text, profile, lang)
        ):
            report(
                "line-capacity",
                index,
                unit_ids,
                f"{len(lines)} lines of widths {widths} exceed "
                f"{profile.max_lines} x {budget}",
            )

        if cap > 0 and timed and end - start > cap + EPS:
            candidate = supplied.get(index)
            report(
                "duration-cap",
                index,
                unit_ids,
                f"span {end - start} exceeds the cap {cap}",
                waived_by=candidate if _whitelisted(candidate) else None,
            )

    violations.sort(key=lambda v: (-1 if v.cue_index is None else v.cue_index, v.kind))
    return PartitionCheckResult(
        origin=origin,
        stage=stage,
        violations=tuple(violations),
        waivers=tuple(supplied[k] for k in sorted(supplied)),
        cue_count=len(cues),
        unit_count=unit_count,
    )
