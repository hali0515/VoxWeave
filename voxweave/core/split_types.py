"""Per-call configuration types of the segmentation engine.

``SplitThresholds`` carries the display and timing limits; ``SplitContext``
bundles the invariants one split call shares. Both are frozen dataclasses, so
the scoring, packing and repair modules can import them without importing
``smart_split``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .timing import GLUE_MAX_GAP_S


@dataclass(frozen=True)
class SplitThresholds:
    """Gap-aware segmentation knobs — one typed source for field names + defaults.

    Built from ``config.gap_thresholds()``'s mapping at the ``smart_split_segments`` boundary via
    :meth:`from_mapping`. ``smart_split_segments`` and ``split_long_cues_with_word_timings``
    require one; ``SplitThresholds()`` gives these defaults.
    """

    clause_ms: int = 400
    vad_skip_ms: int = 1000
    offline_ms: int = 700
    min_cue_s: float = 0.5
    max_cue_s: float = 7.0
    glue_gap_s: float = GLUE_MAX_GAP_S
    # Reading-speed linger (0 = off): a cue shorter than reading_chars/cps extends
    # into the following gap, capped at LINGER_CAP_S past speech end. lag_out_s is
    # a flat tail pad applied to every cue end (0 = off). config.gap_thresholds
    # supplies per-language values; the dataclass defaults keep both off so direct
    # constructions (tests) preserve exact timing.
    cps: float = 0.0
    lag_out_s: float = 0.0
    # Shot-change pairing window (0 = off): a cue boundary within this of a
    # detected cut gets the Netflix zone treatment (see _snap_to_shots). 11
    # frames @24fps covers the outermost adjustment zone. Only consulted when
    # the caller passes shot_changes.
    shot_snap_s: float = 11.0 / 24.0

    @classmethod
    def from_mapping(cls, d: dict) -> SplitThresholds:
        """Build from a (possibly partial) mapping, ignoring unknown keys and filling defaults."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SplitContext:
    """Per-call segmentation context: language, line budget, thresholds, VAD.

    Bundles the invariants of one ``split_long_cues_with_word_timings`` call so
    they travel into the packing loop (``_pack_atoms_into_chunks`` /
    ``_classify_atom_break``) as one value instead of five parallel parameters.
    """

    lang: str
    max_line_length: int
    max_lines: int
    th: SplitThresholds
    speech_spans: list[tuple[float, float]] | None = None
