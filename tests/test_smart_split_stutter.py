"""Intra-cue stutter merging (_merge_stutters) + em-dash end-to-end splitting."""

from voxweave.core.layout import _merge_stutters
from voxweave.core.smart_split import smart_split_segments

THRESH = {"clause_ms": 400, "vad_skip_ms": 1000, "offline_ms": 700, "max_cue_s": 7.0}


# --------------------------------------------------------------------------- #
# _merge_stutters
# --------------------------------------------------------------------------- #
def test_stutter_single_letter():
    assert _merge_stutters("I I commissioned") == "I-I commissioned"


def test_stutter_long_word():
    assert _merge_stutters("a negative negative IQ") == "a negative-negative IQ"


def test_stutter_triple_chains():
    assert _merge_stutters("I I I") == "I-I-I"
    assert _merge_stutters("no no no way") == "no-no-no way"


def test_stutter_case_insensitive_preserves_casing():
    assert _merge_stutters("The the dog") == "The-the dog"


def test_stutter_leaves_distinct_words():
    assert _merge_stutters("I really wanted") == "I really wanted"
    assert _merge_stutters("well-known fact") == "well-known fact"


def test_stutter_ignores_cjk_and_digits():
    assert _merge_stutters("私 私 です") == "私 私 です"  # CJK left untouched
    assert _merge_stutters("section 2 2") == "section 2 2"  # digits left untouched


# --------------------------------------------------------------------------- #
# smart_split gating: stutter merging is opt-in with thresholds (gap-aware mode)
# --------------------------------------------------------------------------- #
def _seg(text, words):
    return {
        "text": text,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


def test_stutter_applied_in_production_path():
    words = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "I", "start": 0.3, "end": 0.5},
        {"word": "commissioned", "start": 0.6, "end": 1.4},
    ]
    cues = smart_split_segments(
        [_seg("I I commissioned", words)], lang="en", thresholds=THRESH
    )
    joined = " ".join(c["text"] for c in cues)
    assert "I-I" in joined


def test_stutter_not_applied_in_legacy_path():
    # no thresholds (legacy path) -> text left untouched; byte-compatibility preserved
    words = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "I", "start": 0.3, "end": 0.5},
    ]
    cues = smart_split_segments([_seg("I I", words)], lang="en")
    assert "I-I" not in " ".join(c["text"] for c in cues)


# --------------------------------------------------------------------------- #
# em-dash end-to-end: reinject-split token -> smart_split gap-split produces two cues
# --------------------------------------------------------------------------- #
def test_em_dash_splits_into_two_cues_with_gap():
    # "today—" ends at 2.0, "a debut..." starts at 5.0 -> 3s pause >= vad_skip -> must split into two cues
    words = [
        {"word": "tiny", "start": 0.6, "end": 1.0},
        {"word": "debut", "start": 1.0, "end": 1.6},
        {"word": "today—", "start": 1.6, "end": 2.0},
        {"word": "a", "start": 5.0, "end": 5.2},
        {"word": "debut", "start": 5.2, "end": 5.8},
    ]
    text = "tiny debut today— a debut"
    cues = smart_split_segments([_seg(text, words)], lang="en", thresholds=THRESH)
    assert len(cues) == 2
    # first cue ends before the pause, second cue starts after -> no cue spans the silence
    assert cues[0]["end"] <= 2.01
    assert cues[1]["start"] >= 4.99


def test_em_dash_no_gap_stays_one_cue():
    # same em-dash but no pause (back-to-back) -> no split (gap-split acts on real pauses, not on dashes)
    words = [
        {"word": "today—", "start": 1.6, "end": 2.0},
        {"word": "a", "start": 2.05, "end": 2.3},
    ]
    cues = smart_split_segments([_seg("today— a", words)], lang="en", thresholds=THRESH)
    assert len(cues) == 1
