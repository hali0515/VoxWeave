"""Deterministic, text-preserving refinement below coarse source units.

The refiner is a shadow-only preprocessing step.  It never edits the production
``SegDocument`` or its persisted unit stream: callers receive newly minted,
positional :class:`~voxweave.core.segdoc.SourceUnit` records plus an origin map
back to the production parents.

Interior times are display conveniences, not acoustic evidence.  They are
distributed in proportion to :func:`layout._token_char_count` and every minted
unit is therefore marked ``subunit-<evidence>``.  The provenance-aware materializer
uses that marker to withhold an endpoint speech anchor unless the selected cue's
actual endpoint unit is still aligner-provenance.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .authority import digest_payload
from .boundary_lattice import CAP_EPS_S, preflight_profile, preflight_units
from .breakpoints import _load_jieba, _load_parser, phrase_atoms
from .langsets import LANGUAGES_WITHOUT_SPACES
from .layout import (
    _PUNCT_TO_SPACE_RE,
    _join,
    _line_budget_width,
    _token_char_count,
    _vis_width,
)
from .providers import note_degraded
from .segdoc import DisplayProfile, SegDocument, SourceUnit

__all__ = [
    "EVIDENCE_KINDS",
    "EVIDENCE_RANKING",
    "RefineResult",
    "RefinementAuthorityError",
    "RefinementConservationError",
    "assert_refinement_conserved",
    "empty_refine_result",
    "refine_document",
    "refine_units",
    "require_issued_refinement",
    "speech_span_units",
]


# The ranking is policy; the sorted companion is the byte-stable artifact key
# order.  Keep the spelling aligned with ``provenance='subunit-<evidence>'``.
EVIDENCE_RANKING: tuple[str, ...] = (
    "whitespace",
    "punct",
    "phrase",
    "per-char",
)
EVIDENCE_KINDS: tuple[str, ...] = tuple(sorted(EVIDENCE_RANKING))


class RefinementConservationError(ValueError):
    """A proposed refinement changed or ambiguously owned the source text."""


class RefinementAuthorityError(RefinementConservationError):
    """A refinement payload was not issued intact by this module's refiner."""


_REFINEMENT_ISSUER = object()


@dataclass(frozen=True)
class _RefinementSeal:
    issuer: object
    digest: str


@dataclass(frozen=True)
class RefineResult:
    """The refined unit stream and its audit metadata.

    ``minted`` counts derived children (not the net length increase).  Thus the
    invariant is ``len(units) == parents - refined_parent_count + minted``.
    ``origin`` is always available to W3's parent projection and remains explicit
    in the artifact even for an identity refinement.  That makes the parent
    ownership claim directly auditable instead of asking readers to infer a
    special ``null`` convention that the LAW does not define.
    """

    units: tuple[SourceUnit, ...]
    origin: tuple[int, ...]
    refined_parent_count: int
    minted: int
    evidence: Mapping[str, int]
    parent_units: tuple[SourceUnit, ...]
    parent_language: str
    degraded: tuple[str, ...] = ()
    _seal: _RefinementSeal | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        units = tuple(self.units)
        origin = tuple(self.origin)
        parent_units = tuple(self.parent_units)
        evidence = dict(self.evidence)
        degraded = tuple(self.degraded)
        if not isinstance(self.parent_language, str) or not self.parent_language:
            raise RefinementConservationError(
                "refinement parent language must be non-empty"
            )
        if (
            isinstance(self.refined_parent_count, bool)
            or not isinstance(self.refined_parent_count, int)
            or self.refined_parent_count < 0
            or isinstance(self.minted, bool)
            or not isinstance(self.minted, int)
            or self.minted < 0
        ):
            raise RefinementConservationError(
                "refinement counts must be non-negative integers"
            )
        if len(units) != len(origin):
            raise RefinementConservationError(
                "refined units and origin map have different cardinality"
            )
        if [unit.id for unit in units] != [f"u{index}" for index in range(len(units))]:
            raise RefinementConservationError("refined unit ids are not positional")
        if any(
            isinstance(parent, bool) or not isinstance(parent, int) or parent < 0
            for parent in origin
        ):
            raise RefinementConservationError(
                "origin map must contain non-negative parent indices"
            )
        if any(left > right for left, right in zip(origin, origin[1:])):
            raise RefinementConservationError("origin map is not monotone")

        actual_parent_count = 0 if not origin else origin[-1] + 1
        claimed_parent_count = len(units) - self.minted + self.refined_parent_count
        if (
            tuple(sorted(set(origin))) != tuple(range(actual_parent_count))
            or actual_parent_count != claimed_parent_count
        ):
            raise RefinementConservationError(
                "origin map and refinement counts claim different parent streams"
            )
        if len(parent_units) != actual_parent_count:
            raise RefinementConservationError(
                "refinement parent payload cardinality disagrees with origin"
            )
        group_sizes = Counter(origin)
        if self.refined_parent_count != sum(size > 1 for size in group_sizes.values()):
            raise RefinementConservationError(
                "refined parent accounting is inconsistent"
            )
        if self.minted != sum(size for size in group_sizes.values() if size > 1):
            raise RefinementConservationError("minted unit accounting is inconsistent")
        if set(evidence) != set(EVIDENCE_KINDS) or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in evidence.values()
        ):
            raise RefinementConservationError(
                "evidence accounting has an invalid vocabulary"
            )
        if self.minted != sum(evidence.values()):
            raise RefinementConservationError("evidence and minted counts disagree")
        observed_evidence: Counter[str] = Counter()
        for unit, parent in zip(units, origin):
            if group_sizes[parent] <= 1:
                if unit.provenance.startswith("subunit-"):
                    raise RefinementConservationError(
                        "derived provenance has no refined parent"
                    )
                continue
            prefix, separator, kind = unit.provenance.partition("subunit-")
            if prefix or separator != "subunit-" or kind not in EVIDENCE_KINDS:
                raise RefinementConservationError(
                    "refined unit provenance has an invalid evidence kind"
                )
            observed_evidence[kind] += 1
        if any(observed_evidence[kind] != evidence[kind] for kind in EVIDENCE_KINDS):
            raise RefinementConservationError(
                "evidence accounting disagrees with refined unit provenance"
            )
        if (
            any(not isinstance(reason, str) or not reason for reason in degraded)
            or tuple(sorted(set(degraded))) != degraded
        ):
            raise RefinementConservationError(
                "degradation reasons must be unique, non-empty, and sorted"
            )

        object.__setattr__(self, "units", units)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "parent_units", parent_units)
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        object.__setattr__(self, "degraded", degraded)

        # The payload is the immutable authority W3 later compares with its
        # parent-projected speaker evidence.  Validate that it really is the
        # source stream claimed by the origin map before accepting the result.
        assert_refinement_conserved(
            parent_units,
            units,
            origin,
            lang=self.parent_language,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the staged ``subunit_split`` block in stable key order."""
        return {
            "degraded": list(self.degraded),
            "evidence": {kind: int(self.evidence[kind]) for kind in EVIDENCE_KINDS},
            "minted": self.minted,
            "origin": list(self.origin),
            "refined_parent_count": self.refined_parent_count,
        }


def _unit_payload(unit: SourceUnit) -> dict[str, Any]:
    return {
        "confidence": unit.confidence,
        "end": unit.end,
        "id": unit.id,
        "provenance": unit.provenance,
        "start": unit.start,
        "surface": unit.surface,
    }


def _refinement_payload(result: RefineResult) -> dict[str, Any]:
    """Complete immutable parent/child relationship covered by the seal."""
    return {
        "degraded": list(result.degraded),
        "evidence": {kind: int(result.evidence[kind]) for kind in EVIDENCE_KINDS},
        "minted": result.minted,
        "origin": list(result.origin),
        "parent_language": result.parent_language,
        "parent_units": [_unit_payload(unit) for unit in result.parent_units],
        "refined_parent_count": result.refined_parent_count,
        "units": [_unit_payload(unit) for unit in result.units],
    }


def _issue_refinement(result: RefineResult) -> RefineResult:
    """Seal a result at the sole refiner-controlled issuance point."""
    object.__setattr__(
        result,
        "_seal",
        _RefinementSeal(
            issuer=_REFINEMENT_ISSUER,
            digest=digest_payload(_refinement_payload(result)),
        ),
    )
    return result


def require_issued_refinement(result: RefineResult) -> None:
    """Reject unissued or altered refinement metadata before optimization."""
    seal = result._seal
    if not isinstance(seal, _RefinementSeal) or seal.issuer is not _REFINEMENT_ISSUER:
        raise RefinementAuthorityError(
            "subunit_split lacks issued refinement authority"
        )
    if seal.digest != digest_payload(_refinement_payload(result)):
        raise RefinementAuthorityError(
            "subunit_split broke its issued refinement authority seal"
        )


@dataclass(frozen=True)
class _Piece:
    surface: str
    evidence: str


def empty_refine_result(
    units: Sequence[SourceUnit] = (), *, language: str = "und"
) -> RefineResult:
    """Identity metadata for a row on which no refiner result was supplied."""
    return _issue_refinement(
        RefineResult(
            units=tuple(units),
            origin=tuple(range(len(units))),
            refined_parent_count=0,
            minted=0,
            evidence={kind: 0 for kind in EVIDENCE_KINDS},
            parent_units=tuple(units),
            parent_language=language,
            degraded=(),
        )
    )


def _remint(
    source: SourceUnit,
    *,
    index: int,
) -> SourceUnit:
    return SourceUnit(
        id=f"u{index}",
        surface=source.surface,
        start=source.start,
        end=source.end,
        provenance=source.provenance,
        confidence=source.confidence,
    )


def _identity(units: Sequence[SourceUnit], *, lang: str) -> RefineResult:
    reminted = tuple(_remint(item, index=index) for index, item in enumerate(units))
    return _issue_refinement(
        RefineResult(
            units=reminted,
            origin=tuple(range(len(reminted))),
            refined_parent_count=0,
            minted=0,
            evidence={kind: 0 for kind in EVIDENCE_KINDS},
            parent_units=tuple(units),
            parent_language=lang,
        )
    )


def _timed_span(unit: SourceUnit) -> float | None:
    start, end = unit.start, unit.end
    if (
        start is None
        or end is None
        or isinstance(start, bool)
        or isinstance(end, bool)
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end < start
    ):
        return None
    return end - start


def _triggered(unit: SourceUnit, profile: DisplayProfile) -> bool:
    budget = _line_budget_width(profile.max_line_length, profile.language)
    width = _vis_width(unit.surface) > budget
    span = _timed_span(unit)
    duration = (
        profile.max_cue_s > 0
        and span is not None
        and span > profile.max_cue_s + CAP_EPS_S
    )
    return width or duration


def _whitespace_pieces(text: str, lang: str) -> list[str]:
    """Expose whitespace boundaries while preserving the language join exactly."""
    if not any(character.isspace() for character in text):
        return [text]
    matches = tuple(re.finditer(r"\s+", text))
    if lang in LANGUAGES_WITHOUT_SPACES:
        pieces: list[str] = []
        cursor = 0
        for match in matches:
            if not any(not char.isspace() for char in text[cursor : match.start()]):
                continue
            if not any(not char.isspace() for char in text[match.end() :]):
                continue
            end = match.end()
            pieces.append(text[cursor:end])
            cursor = end
        if cursor < len(text):
            pieces.append(text[cursor:])
    else:
        # ``_join`` inserts one ASCII space.  Consume exactly one literal space
        # at every useful internal run; leading/trailing whitespace and the rest
        # of a multi-character run stay on a surface byte-for-byte.  A run made
        # only of tabs/newlines cannot be represented as a join boundary and is
        # left inside a piece rather than normalized silently.
        pieces = []
        cursor = 0
        for match in matches:
            if not any(not char.isspace() for char in text[cursor : match.start()]):
                continue
            if not any(not char.isspace() for char in text[match.end() :]):
                continue
            split = next(
                (
                    index
                    for index in range(match.start(), match.end())
                    if text[index] == " "
                ),
                None,
            )
            if split is None:
                continue
            pieces.append(text[cursor:split])
            cursor = split + 1
        if cursor < len(text):
            pieces.append(text[cursor:])
    if len(pieces) < 2 or any(not piece for piece in pieces):
        return [text]
    return pieces if _join(pieces, lang) == text else [text]


def _punct_pieces(text: str, lang: str) -> list[str]:
    """Split after punctuation recognized by the shared subtitle strip pass."""
    if lang not in LANGUAGES_WITHOUT_SPACES:
        # A non-space cut in a spaced language would make ``_join`` insert a
        # character that was not in the parent.  Such a token stays indivisible.
        return [text]
    boundaries = sorted(
        {
            match.end()
            for match in _PUNCT_TO_SPACE_RE.finditer(text)
            if match.end() < len(text)
        }
    )
    if not boundaries:
        return [text]
    points = (0, *boundaries, len(text))
    pieces = [text[left:right] for left, right in zip(points, points[1:])]
    return pieces if len(pieces) > 1 and "".join(pieces) == text else [text]


def _phrase_pieces(text: str, lang: str) -> tuple[list[str], str | None]:
    """Return provider-derived phrases, or name the per-char degradation."""
    if lang not in LANGUAGES_WITHOUT_SPACES:
        return [text], None

    available = False
    if lang == "zh":
        available = _load_jieba() is not None or _load_parser(lang) is not None
    elif lang == "ja":
        available = _load_parser(lang) is not None
    if not available:
        return [text], "no-provider:per-char"

    pieces = phrase_atoms(text, lang)
    if len(pieces) < 2 or "".join(pieces) != text:
        return [text], None
    return pieces, None


def _per_char_pieces(text: str, lang: str) -> list[str]:
    if lang not in LANGUAGES_WITHOUT_SPACES:
        return [text]
    return list(text)


def _needs_split(
    text: str,
    *,
    weight: int,
    total_weight: int,
    parent_duration: float | None,
    profile: DisplayProfile,
) -> bool:
    if _vis_width(text) > _line_budget_width(profile.max_line_length, profile.language):
        return True
    if profile.max_cue_s > 0 and parent_duration is not None and total_weight > 0:
        duration = parent_duration * weight / total_weight
        return duration > profile.max_cue_s + CAP_EPS_S
    return False


def _split_recursive(
    text: str,
    *,
    start_rank: int,
    total_weight: int,
    parent_duration: float | None,
    lang: str,
    profile: DisplayProfile,
) -> tuple[list[_Piece], set[str]]:
    pending_degradation: str | None = None
    for rank in range(start_rank, len(EVIDENCE_RANKING)):
        evidence = EVIDENCE_RANKING[rank]
        if evidence == "whitespace":
            surfaces = _whitespace_pieces(text, lang)
        elif evidence == "punct":
            surfaces = _punct_pieces(text, lang)
        elif evidence == "phrase":
            surfaces, reason = _phrase_pieces(text, lang)
            pending_degradation = pending_degradation or reason
        else:
            surfaces = _per_char_pieces(text, lang)

        if len(surfaces) < 2:
            continue

        degraded: set[str] = set()
        out: list[_Piece] = []
        for surface in surfaces:
            weight = _token_char_count(surface)
            recurse = (
                lang in LANGUAGES_WITHOUT_SPACES
                and rank + 1 < len(EVIDENCE_RANKING)
                and _needs_split(
                    surface,
                    weight=weight,
                    total_weight=total_weight,
                    parent_duration=parent_duration,
                    profile=profile,
                )
            )
            if recurse:
                children, child_degraded = _split_recursive(
                    surface,
                    start_rank=rank + 1,
                    total_weight=total_weight,
                    parent_duration=parent_duration,
                    lang=lang,
                    profile=profile,
                )
                if len(children) > 1:
                    out.extend(children)
                    degraded |= child_degraded
                    continue
            out.append(_Piece(surface=surface, evidence=evidence))

        if evidence == "per-char":
            degraded.add(pending_degradation or "no-usable-boundary:per-char")
        return out, degraded
    return [_Piece(surface=text, evidence="per-char")], set()


def _split_parent(
    parent: SourceUnit, *, lang: str, profile: DisplayProfile
) -> tuple[list[_Piece], set[str]]:
    total_weight = _token_char_count(parent.surface)
    if total_weight <= 0:
        return [_Piece(parent.surface, "per-char")], set()
    return _split_recursive(
        parent.surface,
        start_rank=0,
        total_weight=total_weight,
        parent_duration=_timed_span(parent),
        lang=lang,
        profile=profile,
    )


def _piece_bounds(
    parent: SourceUnit, pieces: Sequence[_Piece]
) -> tuple[tuple[float | None, float | None], ...]:
    start, end = parent.start, parent.end
    if (
        start is None
        or end is None
        or isinstance(start, bool)
        or isinstance(end, bool)
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end < start
    ):
        return tuple((None, None) for _ in pieces)
    weights = [_token_char_count(piece.surface) for piece in pieces]
    total = sum(weights)
    if total <= 0:
        return tuple((None, None) for _ in pieces)
    duration = end - start
    cursor = 0
    out: list[tuple[float, float]] = []
    for index, weight in enumerate(weights):
        piece_start = start + duration * cursor / total
        cursor += weight
        piece_end = (
            end if index == len(weights) - 1 else start + duration * cursor / total
        )
        out.append((piece_start, piece_end))
    return tuple(out)


def assert_refinement_conserved(
    original: Sequence[SourceUnit],
    refined: Sequence[SourceUnit],
    origin: Sequence[int],
    *,
    lang: str,
) -> None:
    """Enforce N9's exact text/ownership conservation as a hard gate."""
    if len(refined) != len(origin):
        raise RefinementConservationError(
            "refined units and origin map have different cardinality"
        )
    if [item.id for item in refined] != [f"u{index}" for index in range(len(refined))]:
        raise RefinementConservationError("refined unit ids are not positional")
    if any(
        isinstance(parent, bool)
        or not isinstance(parent, int)
        or parent < 0
        or parent >= len(original)
        for parent in origin
    ):
        raise RefinementConservationError("origin map names a nonexistent parent")
    if any(left > right for left, right in zip(origin, origin[1:])):
        raise RefinementConservationError("origin map is not monotone")
    if original and tuple(sorted(set(origin))) != tuple(range(len(original))):
        raise RefinementConservationError("origin map does not cover every parent")

    for parent_index, parent in enumerate(original):
        surfaces = [
            item.surface
            for item, owner in zip(refined, origin)
            if owner == parent_index
        ]
        if _join(surfaces, lang) != parent.surface:
            raise RefinementConservationError(
                f"parent {parent_index} changed its character stream"
            )
    if _join([item.surface for item in refined], lang) != _join(
        [item.surface for item in original], lang
    ):
        raise RefinementConservationError(
            "refinement changed the document character stream"
        )


def refine_units(
    units: Sequence[SourceUnit], *, lang: str, profile: DisplayProfile
) -> RefineResult:
    """Refine locally triggered units and re-mint the whole shadow stream.

    Profile and source-span preflights precede provider work.  An invalid profile
    remains the optimizer's typed invalid measurement, while an invalid source
    unit remains intact so the lattice's existing ``span-preflight`` reason is
    not hidden by invented children.
    """
    source = tuple(units)
    if preflight_profile(profile):
        return _identity(source, lang=lang)
    invalid_units = {violation.unit_index for violation in preflight_units(source)}

    output: list[SourceUnit] = []
    origin: list[int] = []
    evidence = {kind: 0 for kind in EVIDENCE_KINDS}
    degraded: set[str] = set()
    degradation_events: list[str] = []
    refined_parents = 0
    minted = 0

    for parent_index, parent in enumerate(source):
        pieces: list[_Piece]
        local_degraded: set[str]
        if parent_index in invalid_units or not _triggered(parent, profile):
            pieces, local_degraded = [_Piece(parent.surface, parent.provenance)], set()
        else:
            pieces, local_degraded = _split_parent(parent, lang=lang, profile=profile)

        if len(pieces) < 2:
            output.append(_remint(parent, index=len(output)))
            origin.append(parent_index)
            continue

        refined_parents += 1
        minted += len(pieces)
        degraded |= local_degraded
        degradation_events.extend(sorted(local_degraded))
        for piece, (piece_start, piece_end) in zip(
            pieces, _piece_bounds(parent, pieces)
        ):
            evidence[piece.evidence] += 1
            output.append(
                SourceUnit(
                    id=f"u{len(output)}",
                    surface=piece.surface,
                    start=piece_start,
                    end=piece_end,
                    provenance=f"subunit-{piece.evidence}",
                    confidence=parent.confidence,
                )
            )
            origin.append(parent_index)

    result = _issue_refinement(
        RefineResult(
            units=tuple(output),
            origin=tuple(origin),
            refined_parent_count=refined_parents,
            minted=minted,
            evidence=evidence,
            parent_units=source,
            parent_language=lang,
            degraded=tuple(sorted(degraded)),
        )
    )
    assert_refinement_conserved(source, result.units, result.origin, lang=lang)
    expected_length = len(source) - refined_parents + minted
    if len(result.units) != expected_length or minted != sum(evidence.values()):
        raise RefinementConservationError("refinement accounting is inconsistent")
    for reason in degradation_events:
        note_degraded("subunit", reason)
    return result


def refine_document(document: SegDocument) -> tuple[SegDocument, RefineResult]:
    """Return a detached, positionally re-minted shadow document and metadata."""
    result = refine_units(
        document.units, lang=document.language, profile=document.profile
    )
    shadow = SegDocument(
        language=document.language,
        units=list(result.units),
        profile=document.profile,
        vad_speech=copy.deepcopy(document.vad_speech),
        shot_changes=copy.deepcopy(document.shot_changes),
        sing_spans=copy.deepcopy(document.sing_spans),
        speaker_turns=copy.deepcopy(document.speaker_turns),
        manifest=copy.deepcopy(document.manifest),
        text=document.text,
    )
    return shadow, result


def speech_span_units(
    units: Sequence[SourceUnit],
) -> tuple[float | None, float | None]:
    """Return exact endpoint anchors by boundary validity.

    Interior provenance is irrelevant.  The start exists only when the first
    owned unit is aligner-provenance with a finite start; the end mirrors the
    last owned unit.  A derived prefix/suffix therefore cannot be laundered into
    an apparently exact acoustic boundary by an interior aligned unit.
    """
    if not units:
        return None, None
    first, last = units[0], units[-1]
    start = (
        first.start
        if first.provenance == "aligner"
        and first.start is not None
        and not isinstance(first.start, bool)
        and math.isfinite(first.start)
        else None
    )
    end = (
        last.end
        if last.provenance == "aligner"
        and last.end is not None
        and not isinstance(last.end, bool)
        and math.isfinite(last.end)
        else None
    )
    return start, end
