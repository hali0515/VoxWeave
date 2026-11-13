from __future__ import annotations

from voxweave.core.unit_repair import repair_stranded_tails


def test_stranded_tail_repair_reads_surfaces_like_the_engine():
    """A ``text`` that is present but empty falls back to ``word``, as it does
    in the engine; ``text: None`` must not count as the four characters 'None'."""
    base = [
        {"text": "今日", "start": 0.0, "end": 0.4},
        {"text": "は", "start": 0.4, "end": 0.6},
        {"text": "い", "start": 0.6, "end": 0.8},
        {"text": "い", "start": 4.0, "end": 4.2},
        {"text": "天気", "start": 8.0, "end": 8.4},
    ]
    speech = [(0.0, 0.8), (4.0, 4.2), (8.0, 8.4)]
    expected = repair_stranded_tails(base, "ja", speech)
    for blank in ("", None):
        variant = [
            {
                "text": blank,
                "word": unit["text"],
                "start": unit["start"],
                "end": unit["end"],
            }
            for unit in base
        ]
        repaired = repair_stranded_tails(variant, "ja", speech)
        assert [(u["start"], u["end"]) for u in repaired] == [
            (u["start"], u["end"]) for u in expected
        ]
