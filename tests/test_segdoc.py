# tests/test_segdoc.py
"""P3 PIN 1 -- the immutable segmentation IR (``voxweave.core.segdoc``).

Three claims are pinned here:

* ``DisplayProfile.from_resolved`` is a *recorder*, not a resolver: it stores the
  mapping it is handed verbatim (``float`` coercion only), applies no defaults of
  its own, clamps nothing, and raises ``KeyError`` when a threshold is missing.
  That is what keeps the manifest honest -- the two disagreeing threshold
  definitions in the tree (``config.gap_thresholds`` vs the ``SplitThresholds``
  dataclass defaults) can never be silently substituted for what actually ran.
* ``SourceUnit`` ids are positional (``u0``..``uN``) and therefore identical
  across replays of the same stream.
* ``SegDocument`` carries the optional evidence arrays and the manifest verbatim
  -- it is a record, not a normalizer.
"""

import pytest

from voxweave.core.segdoc import (
    DisplayProfile,
    SegDocument,
    SourceUnit,
    build_seg_document,
)

# The nine keys ``segment_document`` resolves into ``thresholds_used``.
THRESHOLD_KEYS = (
    "clause_ms",
    "vad_skip_ms",
    "offline_ms",
    "min_cue_s",
    "max_cue_s",
    "glue_gap_s",
    "cps",
    "lag_out_s",
    "shot_snap_s",
)

# config.gap_thresholds("ja") at HEAD.
JA_THRESHOLDS = {
    "clause_ms": 560,
    "vad_skip_ms": 1000,
    "offline_ms": 980,
    "min_cue_s": 0.5,
    "max_cue_s": 7.0,
    "glue_gap_s": 0.3,
    "cps": 7.0,
    "lag_out_s": 0.25,
    "shot_snap_s": 0.458,
}

UNITS = [
    {"text": "Where", "start": 0.0, "end": 0.4},
    {"text": "did", "start": 0.5, "end": 0.8},
    {"text": "you", "start": 0.9, "end": 1.2},
    {"text": "go", "start": 1.4, "end": 2.0},
]


def _profile(thresholds=None, **kwargs) -> DisplayProfile:
    kwargs.setdefault("max_line_length", 18)
    kwargs.setdefault("max_lines", 1)
    return DisplayProfile.from_resolved(
        "ja", dict(JA_THRESHOLDS if thresholds is None else thresholds), **kwargs
    )


def _document(**kwargs) -> SegDocument:
    kwargs.setdefault("language", "en")
    kwargs.setdefault("units", UNITS)
    kwargs.setdefault("profile", _profile())
    kwargs.setdefault("manifest", {"engine": "legacy-v1"})
    return build_seg_document(**kwargs)


# --- DisplayProfile.from_resolved -------------------------------------------


def test_from_resolved_stores_every_threshold_verbatim():
    profile = _profile()
    for key in THRESHOLD_KEYS:
        assert getattr(profile, key) == float(JA_THRESHOLDS[key]), key
        assert isinstance(getattr(profile, key), float), key
    assert profile.language == "ja"


def test_from_resolved_records_the_layout_it_was_handed():
    profile = _profile(max_line_length=12, max_lines=2)
    assert profile.max_line_length == 12
    assert profile.max_lines == 2
    assert isinstance(profile.max_line_length, int)
    assert isinstance(profile.max_lines, int)


def test_from_resolved_applies_no_defaults_of_its_own():
    """The SplitThresholds dataclass defaults cps/lag_out to 0 while config
    supplies 7.0/0.25. A profile that "fixed up" a zero would record a run that
    never happened, so zeros must survive."""
    off = dict(JA_THRESHOLDS, cps=0.0, lag_out_s=0.0, glue_gap_s=0.0)
    profile = _profile(off)
    assert profile.cps == 0.0
    assert profile.lag_out_s == 0.0
    assert profile.glue_gap_s == 0.0


def test_from_resolved_does_not_clamp_or_renormalize():
    """min_cue_s is clamped to 5/6 by config and shot_snap_s has two
    near-but-unequal defaults (0.458 vs 11/24). The profile records whichever
    value actually reached the engine, unmodified."""
    odd = dict(JA_THRESHOLDS, min_cue_s=99.0, shot_snap_s=11.0 / 24.0, max_cue_s=0.0)
    profile = _profile(odd)
    assert profile.min_cue_s == 99.0
    assert profile.shot_snap_s == 11.0 / 24.0
    assert profile.shot_snap_s != 0.458
    assert profile.max_cue_s == 0.0


@pytest.mark.parametrize("missing", THRESHOLD_KEYS)
def test_from_resolved_raises_key_error_on_a_missing_threshold(missing):
    """The caller passes ``thresholds_used``, which always has all nine; a hole
    means the caller resolved something else and must not be papered over."""
    partial = {k: v for k, v in JA_THRESHOLDS.items() if k != missing}
    with pytest.raises(KeyError):
        DisplayProfile.from_resolved("ja", partial, max_line_length=18, max_lines=1)


def test_from_resolved_ignores_unknown_keys_in_the_mapping():
    """A richer mapping is recorded down to the nine declared fields, not rejected."""
    profile = _profile(dict(JA_THRESHOLDS, some_future_knob=1.0))
    assert profile.clause_ms == 560.0
    assert not hasattr(profile, "some_future_knob")


def test_display_profile_is_frozen():
    profile = _profile()
    with pytest.raises(Exception):
        profile.cps = 1.0  # type: ignore[misc]


# --- SourceUnit minting ------------------------------------------------------


def test_source_unit_ids_are_positional():
    doc = _document()
    assert [u.id for u in doc.units] == ["u0", "u1", "u2", "u3"]


def test_source_unit_minting_is_deterministic_across_replays():
    first = _document().units
    second = _document().units
    assert [(u.id, u.surface, u.start, u.end) for u in first] == [
        (u.id, u.surface, u.start, u.end) for u in second
    ]


def test_source_unit_surface_is_the_unit_text_view():
    """``_unit_text``: ``text`` wins, ``word`` is the fallback, absent is ``""``."""
    doc = _document(
        units=[
            {"text": "aligner", "word": "asr", "start": 0.0, "end": 1.0},
            {"word": "asr-only", "start": 1.0, "end": 2.0},
            {"text": "", "word": "empty-text", "start": 2.0, "end": 3.0},
            {"start": 3.0, "end": 4.0},
        ]
    )
    assert [u.surface for u in doc.units] == ["aligner", "asr-only", "empty-text", ""]


def test_source_unit_spans_carry_none_through():
    doc = _document(
        units=[
            {"text": "a", "start": 0.5, "end": 1.5},
            {"text": "b"},
            {"text": "c", "start": None, "end": None},
        ]
    )
    assert [(u.start, u.end) for u in doc.units] == [
        (0.5, 1.5),
        (None, None),
        (None, None),
    ]


def test_source_unit_is_frozen():
    unit = _document().units[0]
    assert isinstance(unit, SourceUnit)
    with pytest.raises(Exception):
        unit.surface = "MUTATED"  # type: ignore[misc]


def test_empty_unit_stream_mints_nothing():
    assert _document(units=[]).units == []


# --- SegDocument evidence ----------------------------------------------------


def test_seg_document_carries_evidence_verbatim():
    vad = [(0.0, 2.0), (2.4, 3.6)]
    shots = [2.3, 3.9]
    sing = [(0.0, 1.0)]
    turns = [(0.0, 2.2, "SPEAKER_00"), (2.3, 3.8, "SPEAKER_01")]
    doc = _document(
        vad_speech=vad, shot_changes=shots, sing_spans=sing, speaker_turns=turns
    )
    assert doc.vad_speech == vad
    assert doc.shot_changes == shots
    assert doc.sing_spans == sing
    assert doc.speaker_turns == turns


def test_seg_document_absent_evidence_stays_none():
    doc = _document()
    assert doc.vad_speech is None
    assert doc.shot_changes is None
    assert doc.sing_spans is None
    assert doc.speaker_turns is None


def test_seg_document_keeps_the_manifest_object_it_was_given():
    manifest = {"manifest_version": 1, "engine": "legacy-v1"}
    doc = _document(manifest=manifest)
    assert doc.manifest is manifest


def test_seg_document_records_the_profile_and_language():
    profile = _profile(max_line_length=42, max_lines=2)
    doc = _document(language="en", profile=profile)
    assert doc.profile is profile
    assert doc.language == "en"


def test_seg_document_records_the_joined_text_verbatim():
    """``text`` is the engine's input stream, stored as handed in -- the builder
    never re-joins the surfaces itself (that would duplicate the no-space rule)."""
    doc = _document(text="Where did you go")
    assert doc.text == "Where did you go"
    # a no-space join is recorded exactly as the caller made it
    assert _document(text="こんにちは世界").text == "こんにちは世界"


def test_seg_document_text_defaults_to_none():
    """Optional and additive: a builder call that predates it is still valid."""
    assert _document().text is None


# --- reserved P5 fields ------------------------------------------------------


def test_source_unit_reserved_fields_have_their_defaults():
    """``provenance``/``confidence`` are carried, not consumed: minting applies
    the declared defaults so P5 can start writing them without a second IR
    migration."""
    for unit in _document().units:
        assert unit.provenance == "aligner"
        assert unit.confidence is None


def test_source_unit_reserved_fields_are_settable_and_still_frozen():
    unit = SourceUnit(
        id="u0",
        surface="go",
        start=1.4,
        end=2.0,
        provenance="manual",
        confidence=0.5,
    )
    assert (unit.provenance, unit.confidence) == ("manual", 0.5)
    with pytest.raises(Exception):
        unit.confidence = 0.9  # type: ignore[misc]


def test_source_unit_positional_construction_is_unchanged():
    """The reserved fields are appended with defaults, so the four-argument form
    every existing caller uses still builds an equal unit."""
    assert SourceUnit("u0", "go", 1.4, 2.0) == SourceUnit(
        id="u0", surface="go", start=1.4, end=2.0, provenance="aligner", confidence=None
    )
