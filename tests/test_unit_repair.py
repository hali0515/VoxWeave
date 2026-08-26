"""Stranded-tail unit repair: aligner artifacts fixed before cue formation.

The real specimen: MMS parked the word-final い of 弱い 6.42s past the word's
body, across VAD-confirmed dead air, alone on a later speech island. The repair
collapses such tails to the previous unit's end; every gate that guards it is
pinned here, because each one exists to protect a legitimate case.
"""

from voxweave.core.breakpoints import phrase_atoms
from voxweave.core.unit_repair import repair_stranded_tails


def _units(*specs):
    return [{"text": t, "start": s, "end": e} for t, s, e in specs]


def test_stranded_tail_collapses_to_previous_end():
    # Precondition: budoux keeps 弱い as one phrase, so い is not a phrase start.
    assert phrase_atoms("弱い", "ja") == ["弱い"]
    units = _units(("弱", 22.86, 23.08), ("い", 29.50, 30.20))
    out = repair_stranded_tails(units, "ja", [(22.0, 23.7), (29.5, 30.2)])
    assert out[1]["start"] == out[1]["end"] == 23.08
    # input is never mutated
    assert units[1]["start"] == 29.50


def test_phrase_start_after_pause_is_a_new_utterance_not_a_tail():
    assert phrase_atoms("弱い明日は", "ja")[0] == "弱い"
    units = _units(
        ("弱", 0.0, 0.2),
        ("い", 0.2, 0.3),
        ("明", 5.0, 5.1),
        ("日", 5.1, 5.2),
        ("は", 5.2, 5.3),
    )
    out = repair_stranded_tails(units, "ja", [(0.0, 0.4), (5.0, 5.4)])
    assert out[2]["start"] == 5.0  # untouched


def test_isolation_gate_spares_a_tail_at_a_real_speech_onset():
    # The stray's island also hosts the onset of a later phrase: pulling the
    # tail back would drag a genuine utterance start with it, so leave it.
    assert phrase_atoms("弱い明日は", "ja")[0] == "弱い"
    units = _units(
        ("弱", 0.0, 0.2),
        ("い", 5.0, 5.1),
        ("明", 5.2, 5.3),
        ("日", 5.3, 5.4),
        ("は", 5.4, 5.5),
    )
    out = repair_stranded_tails(units, "ja", [(0.0, 0.4), (4.9, 5.6)])
    assert out[1]["start"] == 5.0  # untouched


def test_small_gap_or_voiced_gap_is_not_stranded():
    units = _units(("弱", 0.0, 0.2), ("い", 1.5, 1.6))
    out = repair_stranded_tails(units, "ja", [(0.0, 2.0)])
    assert out[1]["start"] == 1.5  # gap under threshold
    units = _units(("弱", 0.0, 0.2), ("い", 3.0, 3.1))
    out = repair_stranded_tails(units, "ja", [(0.0, 3.2)])
    assert out[1]["start"] == 3.0  # wide gap but fully voiced: no silence proof


def test_no_vad_evidence_means_no_repair():
    units = _units(("弱", 0.0, 0.2), ("い", 9.0, 9.1))
    assert repair_stranded_tails(units, "ja", None)[1]["start"] == 9.0
    assert repair_stranded_tails(units, "ja", [])[1]["start"] == 9.0


def test_spaced_language_is_untouched():
    units = _units(("weak", 0.0, 0.2), ("ly", 9.0, 9.1))
    assert repair_stranded_tails(units, "en", [(0.0, 0.4), (9.0, 9.2)]) == units
