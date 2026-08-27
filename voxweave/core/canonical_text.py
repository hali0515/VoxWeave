"""The one authority for a cue's delivered text, lines and reading load.

Everything downstream of a partition asks the same three questions about a
candidate cue -- what ships, does it fit, how long does it take to read -- and
before P5 each caller answered them for itself off whatever string happened to
be nearest. That is how the two worlds drift: the cost model priced the raw
join, the renderer shipped the stripped-and-wrapped form, and the reading load
they disagreed about is a duration difference in the delivered subtitle
(registry FD-1).

Two contracts here carry the weight.

**The primary source is the immutable ``word_data``**, never the cue's own
``text``. A projection that reads what it previously wrote is not a function of
the seed, and every determinism claim in the finalizer rests on it being one.
The cue's text is a typed *fallback*, taken only when the reconstruction is
unusable, and each such taking is a ledgered
``canonical-text-fallback(reason)`` with a reason from a closed vocabulary --
"the list was truthy" is not a usability test.

**Legality is decided by direct inspection** of the delivered lines and cell
widths (:func:`canonical_legal`). :func:`voxweave.core.layout._fits_budget`
answers a different question -- "could *a* rewrap fold this" -- and at
degenerate profiles it admits text that ships over-wide (the kinsoku pull-back
``あっ`` overflow). It is not the predicate anywhere in the v2 lane.

The stutter pass is bounded rather than run to a fixpoint (FD-9): the shipped
:func:`voxweave.core.layout._merge_stutters` iterates ``while prev != text``,
which is a termination proof only for substitutions that shrink. Four scans are
charged, and failing to *observe* stability inside them is reported, not hidden.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .layout import (
    _join,
    _line_budget_width,
    _reading_chars,
    _STUTTER_RE,
    _vis_width,
    _wrap_units,
    strip_punct_for_subtitles,
    wrap_cue_text,
)
from .partition_check import normalize_text
from .schema import Unit
from .segdoc import DisplayProfile
from .smart_split import _surface_ranges, _unit_text

__all__ = [
    "CANONICAL_PASS_FACTOR",
    "CanonicalWork",
    "FALLBACK_REASONS",
    "FinalText",
    "STUTTER_MAX_SCANS",
    "band_scan_lower_bound_exceeded",
    "bounded_stutter",
    "canonical_legal",
    "canonical_text",
    "line_budget",
    "over_wide_token",
    "reconstruct_surface",
]

#: FD-9's bound on the stutter pass. Scans are charged, not attempted-and-
#: forgotten: exhausting them without observing a no-op is a reported fact.
STUTTER_MAX_SCANS: int = 4

#: The exact worst-case number of character passes one projection makes:
#: 1 strip + at most STUTTER_MAX_SCANS stutter scans + 1 wrap. The N14 work
#: bound is stated in terms of this factor, so it may not be rounded up for
#: comfort.
CANONICAL_PASS_FACTOR: int = 6

#: Closed vocabulary of the reasons a projection falls back to the cue's own
#: text. A reason outside this tuple means a new failure mode landed without a
#: registry row.
FALLBACK_REASONS: tuple[str, ...] = (
    "empty-reconstruction",
    "footprint-mismatch",
    "granularity-unreconciled",
)


@dataclass(frozen=True)
class FinalText:
    """One cue's canonical projection: what ships, how it lays out, its load.

    ``text`` is the ``"\\n"``-joined delivered string and ``lines``/
    ``cell_widths`` are its rendered decomposition -- recorded rather than
    re-derived so a legality check and the renderer cannot disagree about which
    lines exist. ``reading_chars`` is measured on the wrapped text and is
    newline-insensitive: a wrap is layout, not content, and must not move a
    CPS-driven duration.
    """

    text: str
    lines: tuple[str, ...]
    cell_widths: tuple[int, ...]
    reading_chars: int
    source: Literal["word-data", "fallback"]
    fallback_reason: str | None
    stutter_stable: bool
    stutter_scans: int


@dataclass
class CanonicalWork:
    """Per-interval work ledger plus the projection cache.

    ``canonical_chars`` counts RAW character visits, not projections: the N14
    bound is about the work a band scan does, and a per-call counter would hide
    a long span behind a small call count. The cache is keyed by
    ``(start_node, end_node)`` because that pair, not the text, is what a
    lattice enumerates repeatedly.
    """

    canonical_chars: int = 0
    cache: dict[tuple[int, int], FinalText] = field(default_factory=dict)

    def charge(self, chars: int) -> None:
        self.canonical_chars += chars

    def cached(self, key: tuple[int, int], build: Callable[[], FinalText]) -> FinalText:
        """Return the cached projection for ``key``, building it at most once.

        A hit charges nothing -- that is the whole point of the cache being in
        the same object as the counter.
        """
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        value = build()
        self.cache[key] = value
        return value


def reconstruct_surface(word_data: Sequence[Unit], lang: str) -> str:
    """The immutable primary source: the language-join of the stored surfaces.

    Read-only over ``word_data``; the projection never writes back into the
    stream it reads.
    """
    return _join([_unit_text(unit) for unit in word_data], lang)


def _stutter_sub(text: str) -> str:
    """ONE stutter substitution pass.

    Module level so the FD-9 fixture can inject a length-nonincreasing double
    that needs five or more scans; the real substitution reaches a fixpoint in
    far fewer.
    """
    return _STUTTER_RE.sub(r"\1-\3", text)


def bounded_stutter(text: str) -> tuple[str, bool, int]:
    """Merge stutters within :data:`STUTTER_MAX_SCANS` scans (FD-9).

    Returns ``(text, stable, scans)``. ``stable`` is True only when a scan was
    *observed* to change nothing; the terminating no-op scan is charged like any
    other, so ``scans`` is the work done rather than the substitutions made.
    Exhausting the bound delivers the four-times-substituted text and reports --
    it never loops on an unproven contraction.
    """
    for scan in range(1, STUTTER_MAX_SCANS + 1):
        new = _stutter_sub(text)
        if new == text:
            return text, True, scan
        text = new
    return text, False, STUTTER_MAX_SCANS


def line_budget(profile: DisplayProfile) -> int:
    """The profile's line budget in half-width cells (CJK presets double)."""
    return _line_budget_width(profile.max_line_length, profile.language)


def canonical_legal(final: FinalText, profile: DisplayProfile) -> bool:
    """PD-TEXT: is what ships legal, by direct inspection of what ships.

    Not ``_fits_budget``: that asks whether some rewrap of the string could fit,
    which is a different -- and at degenerate profiles a strictly weaker --
    question than whether the delivered lines do.
    """
    budget = line_budget(profile)
    return len(final.lines) <= profile.max_lines and all(
        width <= budget for width in final.cell_widths
    )


def band_scan_lower_bound_exceeded(joined: str, profile: DisplayProfile) -> bool:
    """The ONLY early break a band scan may take: a monotone TRUE lower bound.

    Wrapping may discard one normalized separator at each line break, so those
    cells cannot prove illegality. After conservatively excluding up to
    ``max_lines - 1`` such separators, visual cells above ``max_lines x budget``
    prove no layout can fit. The packer's own cell arithmetic is NOT a necessary
    condition (an ASCII run bridging a CJK line measures 41 cells yet lays out
    legally as ``(20, 20)``), so it is never consulted for admission.
    """
    stripped = strip_punct_for_subtitles(joined)
    removable_separators = min(stripped.count(" "), max(profile.max_lines - 1, 0))
    lower_bound = _vis_width(stripped) - removable_separators
    return lower_bound > profile.max_lines * line_budget(profile)


def over_wide_token(line: str, lang: str, budget: int) -> str | None:
    """The first indivisible atom of ``line`` wider than ``budget``, else None.

    ``None`` on an over-wide line is meaningful: it says the line is over budget
    because the wrap ran out of lines, not because any one token cannot be
    broken.
    """
    for atom, _gap in _wrap_units(line, lang):
        if _vis_width(atom) > budget:
            return atom
    return None


def _usability(
    source: str,
    word_data: Sequence[Unit],
    expected_footprint: str | None,
) -> str | None:
    """The typed reason ``source`` is unusable, or None when it is usable.

    Never a truthy-list test. With a footprint supplied, usability IS
    conservation of that expected owned character stream under
    ``normalize_text`` -- the same normal form the partition checker reads, so
    the projection and the checker cannot disagree about which characters
    survive. Without one, usability is internal consistency: the stored
    surfaces must reconcile against the stream they spell
    (``_surface_ranges``), the predicate ``_unit_ranges`` degrades from.
    """
    if not any(not char.isspace() for char in source):
        return "empty-reconstruction"
    if expected_footprint is not None:
        if normalize_text(source) != normalize_text(expected_footprint):
            return "footprint-mismatch"
        return None
    if not any(_unit_text(unit) for unit in word_data):
        return "granularity-unreconciled"
    if _surface_ranges([source], word_data) is None:
        return "granularity-unreconciled"
    return None


def canonical_text(
    word_data: Sequence[Unit],
    *,
    fallback_text: str,
    lang: str,
    profile: DisplayProfile,
    expected_footprint: str | None = None,
    work: CanonicalWork | None = None,
) -> FinalText:
    """Project one cue's ``word_data`` into the text that will ship.

    ``strip_punct_for_subtitles`` -> bounded stutter -> ``wrap_cue_text``, over
    the reconstruction when it is usable and over ``fallback_text`` otherwise.
    The wrap is handed the resolved half-width budget, so a CJK profile's native
    cell count is converted exactly once, here.
    """
    source = reconstruct_surface(word_data, lang)
    if work is not None:
        work.charge(len(source))

    reason = _usability(source, word_data, expected_footprint)
    raw = source if reason is None else fallback_text

    stripped = strip_punct_for_subtitles(raw)
    if work is not None:
        work.charge(len(raw))
    merged, stable, scans = bounded_stutter(stripped)
    if work is not None:
        work.charge(len(stripped) * scans)
    text = wrap_cue_text(
        merged, lang, profile.max_lines, max_line_length=line_budget(profile)
    )
    if work is not None:
        work.charge(len(merged))

    lines = tuple(text.split("\n"))
    return FinalText(
        text=text,
        lines=lines,
        cell_widths=tuple(_vis_width(line) for line in lines),
        reading_chars=_reading_chars(text),
        source="word-data" if reason is None else "fallback",
        fallback_reason=reason,
        stutter_stable=stable,
        stutter_scans=scans,
    )
