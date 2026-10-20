"""The line-packing measure: the policy-1 packer and the canonical measure.

``IncrementalPacker`` prices a cue by v1's packing rules; ``_canonical_pack_measure``
prices it by the delivered text. This is the one lattice cluster that needs the
canonical-text legality functions. ``boundary_lattice`` re-exports every name here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .boundary_atoms import LatticeAtom
from .canonical_text import CanonicalWork, canonical_legal, canonical_text
from .layout import (
    _PUNCT_TO_SPACE_RE,
    _join,
    _line_budget_width,
    _no_spaces,
    _vis_width,
    strip_punct_for_subtitles,
)
from .segdoc import DisplayProfile


# --------------------------------------------------------- C6 packing oracle


@dataclass(frozen=True)
class PackMeasure:
    """The batch layout answer for one atom prefix."""

    fits: bool
    lines: int
    line_widths: tuple[int, ...]
    text: str

    @property
    def balance(self) -> float:
        if self.lines != 2:
            return 0.0
        return float(abs(self.line_widths[0] - self.line_widths[1]))


class _PackState:
    """Everything about a prefix measurement that the flush can still change."""

    __slots__ = (
        "started",
        "space_pending",
        "token",
        "finished",
        "cur_width",
        "cur_open",
    )

    def __init__(self) -> None:
        self.started = False
        self.space_pending = False
        self.token = ""
        self.finished: list[int] = []
        self.cur_width = 0
        self.cur_open = False

    def clone(self) -> _PackState:
        other = _PackState()
        other.started = self.started
        other.space_pending = self.space_pending
        other.token = self.token
        other.finished = list(self.finished)
        other.cur_width = self.cur_width
        other.cur_open = self.cur_open
        return other


class IncrementalPacker:
    """Prefix-extension oracle over the normalized display stream.

    The input projection is ``strip_punct_for_subtitles(_join(texts, lang))``,
    and the naive way to get it per prefix is to redo all four passes every
    time. The fold is for spaced languages only, where it is differential-pinned
    to the batch packing oracle, and refuses a no-space language. No-space
    admission is governed by cached :func:`canonical_text` projections instead:
    kinsoku can move a closing glyph across a normalized gap, so no bounded
    streaming state is allowed to approximate that decision.

    Each pass is a left-to-right fold with bounded lookahead, so the whole chain
    streams:

    * punctuation stripping needs exactly **one character** of lookahead -- the
      rule that keeps the dot of ``3.75`` is ``[.,](?!\\d)``, and every match is a
      single character, so the production regex can be asked about one character
      at a time by matching it against ``char + next_char``;
    * whitespace collapsing and trimming need only a "space pending" flag;
    * tokenization is local, and greedy first-fit line packing never re-flows a
      line once a later token has been placed.

    What cannot be committed early is held: the last raw character (awaiting its
    lookahead) and a partial token. A measurement
    therefore flushes a **copy** of that bounded tail, which is what makes the
    reported answer the batch answer for the current prefix rather than for the
    prefix plus whatever comes next.

    One honest caveat: the *decision* (fits, line count, widths) is O(1)
    amortised per atom, but materializing ``PackMeasure.text`` is linear in the
    prefix. That is not a shortcoming of the fold -- the caller needs that exact
    string for the edge it is about to build, so the cost is inherent to the
    output rather than to the measurement.
    """

    def __init__(self, lang: str, max_line_length: int, max_lines: int) -> None:
        if _no_spaces(lang):
            raise ValueError(
                f"IncrementalPacker is the spaced-language fold; {lang!r} is "
                "admitted through canonical_text"
            )
        self._sep = " "
        self._sep_width = _vis_width(self._sep)
        self._budget = _line_budget_width(max_line_length, lang)
        self._max_lines = max_lines
        self._steps = 0
        self.reset()

    def reset(self) -> None:
        """Start a fresh prefix. The step counter deliberately survives."""
        self._first = True
        self._hold: str | None = None
        self._norm: list[str] = []
        self._state = _PackState()

    @property
    def steps(self) -> int:
        """Cumulative extension steps -- the work counter, not a prefix length."""
        return self._steps

    def extend(self, atom_text: str) -> PackMeasure:
        """Append one atom and return the batch measure for the whole prefix."""
        self._steps += 1
        chunk = atom_text if self._first else self._sep + atom_text
        self._first = False
        for char in chunk:
            if self._hold is not None:
                self._norm.append(
                    self._feed(self._state, self._resolve(self._hold, char))
                )
            self._hold = char
        return self._measure()

    # -- stage A: punctuation, resolved one character at a time ---------------

    @staticmethod
    def _resolve(char: str, nxt: str | None) -> str:
        """The character the strip pass keeps, given its single lookahead.

        Asked of the production regex directly: every alternative it can match is
        exactly one character wide, so a match anchored at position 0 of
        ``char + lookahead`` is precisely the substitution the batch pass would
        make at that offset.
        """
        return " " if _PUNCT_TO_SPACE_RE.match(char + (nxt or "")) else char

    # -- stages B and C: whitespace, tokenization ----------------------------

    def _feed(self, state: _PackState, char: str) -> str:
        """Push one stripped character through collapsing and tokenization."""
        if char.isspace():
            if state.started:
                state.space_pending = True
            return ""
        out = ""
        if state.space_pending:
            state.space_pending = False
            out += " "
            self._feed_token(state, " ")
        state.started = True
        self._feed_token(state, char)
        return out + char

    def _feed_token(self, state: _PackState, char: str) -> None:
        if char == " ":
            self._close_token(state)
        else:
            state.token += char

    def _close_token(self, state: _PackState) -> None:
        if state.token:
            self._commit_token(state, state.token)
        state.token = ""

    # -- stage D: greedy first-fit packing -----------------------------------

    def _commit_token(self, state: _PackState, token: str) -> None:
        width = _vis_width(token)
        extra = self._sep_width if state.cur_open else 0
        if state.cur_open and state.cur_width + width + extra > self._budget:
            state.finished.append(state.cur_width)
            state.cur_width = width
        else:
            state.cur_width += width + extra
        state.cur_open = True

    def _measure(self) -> PackMeasure:
        state = self._state.clone()
        tail = ""
        if self._hold is not None:
            tail = self._feed(state, self._resolve(self._hold, None))
        self._close_token(state)
        widths = list(state.finished)
        if state.cur_open:
            widths.append(state.cur_width)
        if not widths:
            # An empty projection is one zero-width line, matching the batch
            # call's behaviour on the empty string.
            widths = [0]
        return PackMeasure(
            fits=len(widths) <= self._max_lines
            and all(width <= self._budget for width in widths),
            lines=len(widths),
            line_widths=tuple(widths),
            text="".join(self._norm) + tail,
        )


def _canonical_pack_measure(
    atoms: Sequence[LatticeAtom],
    start: int,
    end: int,
    profile: DisplayProfile,
    work: CanonicalWork,
) -> PackMeasure:
    """Direct no-space admission facts for one cached lattice span.

    The cache key is the lattice's own ``(start, end)`` pair. Legality and the
    recorded line facts come only from ``FinalText``; the unwrapped display text
    remains the direct batch strip used by the cost model before finalization.
    """
    chunk = atoms[start:end]
    lang = profile.language
    raw = _join([atom.text for atom in chunk], lang)
    final = work.cached(
        (start, end),
        lambda: canonical_text(
            [
                {"text": atom.text, "start": atom.start, "end": atom.end}
                for atom in chunk
            ],
            fallback_text=raw,
            lang=lang,
            profile=profile,
            expected_footprint=raw,
            work=work,
        ),
    )
    return PackMeasure(
        fits=canonical_legal(final, profile),
        lines=len(final.lines),
        line_widths=final.cell_widths,
        text=strip_punct_for_subtitles(raw),
    )
