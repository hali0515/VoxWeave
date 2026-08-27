# tests/test_finalizer_preview.py
"""``FinalizerPreview`` is phase 1, and phase 1 is the whole promise (P5 §4).

The preview seam exists so the optimizer prices the cue it will actually get.
Two properties carry that:

* what the preview returns IS ``phase1_cue`` -- the same projection, the same
  absolute desire, float-exact, so gate N7's "the consumed facts equal the
  finalizer's phase-1 output" is true by construction rather than by agreement;
* phase 2 is excluded BY CONTRACT. The candidate carries ``next_start`` because
  the legacy mirror needs it, and this preview must ignore it: an edge is scored
  before its neighbour exists, so a promise about the neighbour clamp would be a
  promise this object cannot keep.

The declared FD-1 asymmetry against ``LegacyCleanupPreview`` is pinned here too:
the legacy mirror answers with the RAW text facts because that is what the pass
it mirrors consumes.
"""

import pytest

from voxweave.core.finalizer import FinalizerPreview, phase1_cue
from voxweave.core.segdoc import DisplayProfile
from voxweave.core.timing_preview import (
    CueCandidate,
    DisplayTimingPreview,
    LegacyCleanupPreview,
)

F = 1.0 / 24.0


def profile(**over):
    base = dict(
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
        shot_snap_s=11 * F,
    )
    base.update(over)
    return DisplayProfile(**base)


P_EN_CPS = profile(max_cue_s=7.0, cps=17.0)


def candidate(prof=P_EN_CPS, **over):
    base = dict(
        start=0.0,
        end=0.5,
        next_start=None,
        text="Hello, world!!",
        word_data=[
            {"text": "Hello,", "start": 0.0, "end": 0.25},
            {"text": "world!!", "start": 0.25, "end": 0.5},
        ],
        speech_start=0.0,
        speech_end=0.5,
        profile=prof,
    )
    base.update(over)
    return CueCandidate(**base)


def as_cue(item: CueCandidate) -> dict:
    return {
        "text": item.text,
        "start": item.start,
        "end": item.end,
        "word_data": list(item.word_data),
        "speech_start": item.speech_start,
        "speech_end": item.speech_end,
    }


def test_preview_is_phase_one_float_exactly():
    item = candidate()
    preview = FinalizerPreview().preview_cue(item)
    built = phase1_cue(as_cue(item), profile=item.profile, index=0)
    assert preview.display_start == built.start
    assert preview.display_end == built.end
    assert preview.final_text == built.text
    assert preview.line_count == len(built.lines)
    assert preview.reading_chars == built.reading_chars
    assert preview.available_s == built.end - built.start


def test_next_start_is_ignored_by_contract():
    """Phase 2 is excluded from the preview: the neighbour clamp is not promised."""
    alone = FinalizerPreview().preview_cue(candidate(next_start=None))
    crowded = FinalizerPreview().preview_cue(candidate(next_start=0.51))
    assert alone == crowded
    assert alone.display_end == 10 / 17.0  # CF-9's canonical desire, unclamped


def test_fd1_asymmetry_against_the_legacy_mirror():
    """Same candidate, two loads: canonical 10 versus the raw 13 (registry FD-1)."""
    item = candidate()
    canonical = FinalizerPreview().preview_cue(item)
    legacy = LegacyCleanupPreview().preview_cue(item)
    assert (canonical.reading_chars, legacy.reading_chars) == (10, 13)
    assert canonical.final_text == "Hello world"
    assert legacy.final_text == "Hello, world!!"
    assert canonical.display_end == 10 / 17.0
    assert legacy.display_end == 13 / 17.0


def test_refusals_carry_the_typed_reports():
    """CF-12: the over-wide line ships and the preview says so."""
    token = "supercalifragilisticexpialidociousandthensome_extra_tail"
    item = candidate(
        text=token,
        word_data=[{"text": token, "start": 0.0, "end": 1.0}],
        end=1.0,
        speech_end=1.0,
    )
    preview = FinalizerPreview().preview_cue(item)
    tag = next(t for t in preview.refusals if t.kind == "line-capacity")
    assert tag.evidence["token"] == token
    assert preview.waivers == ()  # the only waiver kind is minted by the cap slot


def test_an_untimed_candidate_previews_text_but_no_span():
    item = candidate(start=None, end=None, speech_start=None, speech_end=None)
    preview = FinalizerPreview().preview_cue(item)
    assert preview.display_start is None
    assert preview.display_end is None
    assert preview.available_s == 0.0
    assert preview.final_text == "Hello world"


def test_the_scalar_seam_mirrors_available_s():
    item = candidate()
    preview = FinalizerPreview(profile=item.profile)
    assert isinstance(preview, DisplayTimingPreview)
    span = preview.preview_display_span(
        0.0,
        0.5,
        None,
        text=item.text,
        word_data=item.word_data,
        min_cue_s=item.profile.min_cue_s,
        max_cue_s=item.profile.max_cue_s,
        cps=item.profile.cps,
        lag_out_s=item.profile.lag_out_s,
    )
    assert span == preview.preview_cue(item).available_s


def test_the_scalar_seam_refuses_without_a_profile():
    """It receives loose thresholds, not a profile, so it cannot invent a language."""
    with pytest.raises(ValueError):
        FinalizerPreview().preview_display_span(
            0.0,
            0.5,
            None,
            text="hi",
            word_data=[],
            min_cue_s=0.0,
            max_cue_s=0.0,
        )
