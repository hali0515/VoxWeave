# tests/test_diarize_format.py
# Speaker-formatting defects fixed after the GPU E2E audit (pure post-pass, no
# pyannote/GPU):
#   Fix 1 - dash cues must render <=2 lines (normalize pre-wrapped pieces before the
#           dual-budget test; re-wrap split pieces per language).
#   Fix 2 - split/dash pieces must go through timing polish (_cleanup_cues) so they
#           are not sub-flash cues; the no-thresholds call stays byte-compatible.
#   Fix 3 - speaker runs must not cut mid-word (absorb <0.2s label thrash, snap
#           surviving boundaries to jieba/BudouX phrase edges).
#   Fix 5 - a surviving boundary must not strand a bound word on the left cue when
#           the cue has no cleaner phrase edge to offer and no audible gap to
#           justify it (merge instead). ja only, scored with UniDic POS.
from voxweave.core.layout import _vis_width
from voxweave.diarize import (
    _merge_bad_tail_runs,
    _speaker_runs,
    apply_speaker_format,
    format_speaker_cues,
)


def _cue(text, start, end, words):
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": [{"start": s, "end": e} for s, e in words],
    }


def _atoms(items):
    return [{"text": t, "start": s, "end": e} for t, s, e in items]


def _lines(text):
    return text.split("\n")


def _max_lines(cues):
    return max(c["text"].count("\n") + 1 for c in cues)


# --- Fix 1: dash cues render <=2 lines ---------------------------------------

# Two speakers over 9 words; the cue text arrives pre-wrapped (contains "\n")
# because smart_split wrapped it before diarization ran.
EN_DASH_TURNS = [(0.0, 4.0, "SPEAKER_00"), (4.5, 6.0, "SPEAKER_01")]


def _en_dash_cue():
    return _cue(
        "And yours is mine\nand mine is yours Correct",
        0.5,
        5.5,
        [
            (0.5, 0.8),
            (0.9, 1.2),
            (1.3, 1.5),
            (1.6, 1.9),
            (2.0, 2.3),
            (2.4, 2.7),
            (2.8, 3.0),
            (3.1, 3.5),
            (5.0, 5.5),  # "Correct" -> SPEAKER_01
        ],
    )


def test_dash_cue_normalizes_prewrapped_text_to_two_lines():
    out = format_speaker_cues([_en_dash_cue()], EN_DASH_TURNS, "en")
    assert len(out) == 1
    text = out[0]["text"]
    # exactly one newline: two visible lines, not three
    assert text.count("\n") == 1
    lines = _lines(text)
    # both lines start with a bare hyphen (no space) and fit one 42-col line
    for ln in lines:
        assert ln.startswith("-") and not ln.startswith("- ")
        assert _vis_width(ln) <= 42


def test_no_output_cue_renders_more_than_two_lines():
    # Speaker A's line is 43 cols: once "-" is added it no longer fits one line, so
    # the dual event (which would be 3 lines) must NOT be emitted. The cue falls
    # through to a split, and the pre-wrapped "\n" piece re-wraps to <=2 clean lines.
    cue = _cue(
        "the quick brown fox jumps over\nthe lazy dog runs away",
        0.0,
        6.0,
        [
            (0.0, 0.3),
            (0.4, 0.7),
            (0.8, 1.1),
            (1.2, 1.5),
            (1.6, 1.9),
            (2.0, 2.3),
            (2.4, 2.7),
            (2.8, 3.1),
            (3.2, 3.5),  # "dog" -> last SPEAKER_00 word
            (4.6, 4.9),  # "runs" -> SPEAKER_01
            (5.0, 5.4),  # "away"
        ],
    )
    turns = [(0.0, 4.0, "SPEAKER_00"), (4.5, 6.5, "SPEAKER_01")]
    out = format_speaker_cues([cue], turns, "en")
    assert len(out) >= 2  # dual rejected -> split branch taken
    assert _max_lines(out) <= 2


def test_ja_split_piece_has_no_newline():
    # ja is single-line: a two-speaker ja cue splits (no dash pairing). The cue
    # text carries a stale "\n" (position 6, inside speaker B's run); each emitted
    # piece must be re-wrapped so it contains zero "\n".
    cue = _cue(
        "これはテスト\nですね",
        0.0,
        1.8,
        [
            (0.0, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1.0),
            (1.0, 1.2),
            (1.2, 1.4),
            (1.4, 1.6),
            (1.6, 1.8),
        ],
    )
    turns = [(0.0, 0.6, "SPEAKER_00"), (0.6, 1.8, "SPEAKER_01")]
    out = format_speaker_cues([cue], turns, "ja")
    assert len(out) >= 2  # it actually split
    for c in out:
        assert "\n" not in c["text"]


# --- Fix 2: split/dash pieces go through timing polish -----------------------

TH = {"min_cue_s": 0.5, "max_cue_s": 7.0, "cps": 0.0, "lag_out_s": 0.0}
ZH_TURNS = [(0.0, 0.4, "SPEAKER_00"), (0.4, 4.0, "SPEAKER_01")]


def _zh_split_cue():
    # "你好朋友" -> "你好" (SPEAKER_00) / "朋友" (SPEAKER_01)
    return _cue(
        "你好朋友",
        0.0,
        0.7,
        [(0.0, 0.2), (0.2, 0.4), (0.4, 0.55), (0.55, 0.7)],
    )


def test_short_split_piece_extended_into_following_gap():
    cue2 = _cue("在", 2.0, 2.1, [(2.0, 2.1)])
    out = apply_speaker_format([_zh_split_cue(), cue2], ZH_TURNS, "zh", thresholds=TH)
    peng = next(c for c in out if c["text"] == "朋友")
    zai = next(c for c in out if c["text"] == "在")
    assert peng["end"] - peng["start"] >= TH["min_cue_s"] - 1e-9
    assert peng["end"] <= zai["start"] + 1e-9


def test_distinct_speaker_abutting_pieces_not_merged():
    out = apply_speaker_format([_zh_split_cue()], ZH_TURNS, "zh", thresholds=TH)
    assert [c["text"] for c in out] == ["你好", "朋友"]


def test_no_thresholds_call_preserves_timing():
    cue2 = _cue("在", 2.0, 2.1, [(2.0, 2.1)])
    out = apply_speaker_format([_zh_split_cue(), cue2], ZH_TURNS, "zh")
    peng = next(c for c in out if c["text"] == "朋友")
    # no polish: the piece keeps its raw 0.4-0.7 span (no min-dur extension)
    assert peng["end"] == 0.7


# --- Fix 3: speaker runs never cut mid-word ----------------------------------


def test_label_thrash_inside_one_word_collapses_to_single_run():
    # A-B-A over the 3 chars of one jieba word "大碴子" (30-80ms turns).
    atoms = _atoms([("大", 0.0, 0.08), ("碴", 0.08, 0.16), ("子", 0.16, 0.24)])
    turns = [(0.0, 0.08, "A"), (0.08, 0.16, "B"), (0.16, 0.24, "A")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert len(runs) == 1


def test_label_flip_inside_jieba_word_snaps_to_word_edge():
    # "大碴子" is one jieba lexeme; a mid-word speaker flip must snap to the word
    # edge so the word is never split across cues.
    atoms = _atoms([("大", 0.0, 0.3), ("碴", 0.3, 0.65), ("子", 0.65, 0.9)])
    turns = [(0.0, 0.3, "A"), (0.3, 1.0, "B")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert len(runs) == 1
    assert "".join(a["text"] for _, ats in runs for a in ats) == "大碴子"


def test_genuine_second_speaker_run_still_splits():
    # Two distinct jieba words, each a >0.5s speaker run: must stay two runs.
    atoms = _atoms(
        [("你", 0.0, 0.3), ("好", 0.3, 0.6), ("朋", 0.6, 0.9), ("友", 0.9, 1.2)]
    )
    turns = [(0.0, 0.6, "A"), (0.6, 1.2, "B")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert [("".join(a["text"] for a in ats)) for _, ats in runs] == [
        "你好",
        "朋友",
    ]


# --- Fix 4: position-aware tiny-run absorption -------------------------------
# A blanket "absorb any run < MIN_RUN_S" also ate real short trailing/leading
# second-speaker utterances. Policy is now position-aware: A-B-A thrash is always
# absorbed, but an edge run (or an A-B-C middle run) survives when it is at least
# EDGE_RUN_MIN_S long.


def test_trailing_edge_second_speaker_run_kept():
    # Real case: zh cue '我想听听普通话 你好' -- speaker 02 speaks 3.3s, then a
    # 160ms '你好' trailing run by speaker 00. 0.16s < MIN_RUN_S but it is a real
    # edge utterance >= EDGE_RUN_MIN_S, so it must be KEPT and split off cleanly.
    atoms = _atoms(
        [
            ("我", 0.0, 0.5),
            ("想", 0.5, 1.0),
            ("听", 1.0, 1.5),
            ("听", 1.5, 2.0),
            ("普", 2.0, 2.5),
            ("通", 2.5, 3.0),
            ("话", 3.0, 3.3),
            ("你", 3.8, 3.88),
            ("好", 3.88, 3.96),  # 0.16s trailing edge run
        ]
    )
    turns = [(0.0, 3.5, "SPEAKER_02"), (3.74, 5.0, "SPEAKER_00")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert len(runs) == 2
    assert runs[0][0] == "SPEAKER_02"
    assert runs[1][0] == "SPEAKER_00"
    # the trailing piece is the complete word, not a fragment
    assert "".join(a["text"] for a in runs[1][1]) == "你好"


def test_leading_edge_sub_floor_fragment_absorbed():
    # An 80ms '来' fragment at the leading edge is pyannote noise (< EDGE_RUN_MIN_S):
    # absorb it so the cue stays one speaker.
    atoms = _atoms(
        [
            ("来", 0.0, 0.08),  # 0.08s leading edge fragment
            ("我", 0.2, 0.7),
            ("们", 0.7, 1.2),
            ("走", 1.2, 1.7),
            ("吧", 1.7, 2.1),
        ]
    )
    turns = [(0.0, 0.1, "SPEAKER_01"), (0.1, 2.5, "SPEAKER_00")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert len(runs) == 1
    assert runs[0][0] == "SPEAKER_00"


def test_same_speaker_sandwich_absorbed_regardless_of_edge_floor():
    # A-B-A with B=0.16s: even though 0.16 >= EDGE_RUN_MIN_S, a same-speaker
    # sandwich is thrash and must be absorbed (rule 1 overrides the edge floor).
    atoms = _atoms([("我", 0.0, 0.5), ("好", 0.5, 0.66), ("吗", 0.66, 1.2)])
    turns = [(0.0, 0.5, "A"), (0.5, 0.66, "B"), (0.66, 1.5, "A")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert len(runs) == 1
    assert runs[0][0] == "A"


def test_abc_middle_run_kept_above_edge_floor():
    # A-B-C middle run 0.16s between two *different* speakers: not a same-speaker
    # sandwich, and >= EDGE_RUN_MIN_S, so kept -> three runs. (en isolates the
    # policy from jieba phrase snapping.)
    atoms = _atoms([("hello", 0.0, 0.5), ("hi", 0.5, 0.66), ("bye", 0.66, 1.2)])
    turns = [(0.0, 0.5, "SPK_A"), (0.5, 0.66, "SPK_B"), (0.66, 1.5, "SPK_C")]
    runs = _speaker_runs(atoms, turns, "en")
    assert [lb for lb, _ in runs] == ["SPK_A", "SPK_B", "SPK_C"]
    assert "".join(a["text"] for a in runs[1][1]) == "hi"


def test_abc_middle_run_absorbed_below_edge_floor():
    # A-B-C middle run 0.08s (< EDGE_RUN_MIN_S): pyannote noise, absorbed.
    atoms = _atoms([("hello", 0.0, 0.5), ("hi", 0.5, 0.58), ("bye", 0.58, 1.2)])
    turns = [(0.0, 0.5, "SPK_A"), (0.5, 0.58, "SPK_B"), (0.58, 1.5, "SPK_C")]
    runs = _speaker_runs(atoms, turns, "en")
    assert len(runs) == 2
    assert "SPK_B" not in [lb for lb, _ in runs]


# --- Fix 5: a boundary must not strand a bound word (ja only) ----------------
# Snapping to a phrase edge keeps lexemes whole but says nothing about whether
# the edge is a decent *cue* end. When the only legal edge inside a cue is a
# forbidden line-end, the split used to happen anyway: ja-07 replay cut
# '正直この写真欲しい' into a standalone '正直この' (連体詞 この).
#
# The gate is AND-composed -- Level-2 bad tail AND no cleaner edge AND no audible
# gap -- so it never fires on duration, on a clean tail, or across a real beat.
# It is ja-only and scores with UniDic POS (ja_pos_end_penalties), the same signal
# smart_split uses: the Level-1 char table flags *every* trailing の/に, which
# merged away genuine second speakers (そうなの|それは, そんなに|ギリギリ), and no
# validated equivalent grading exists for zh.


def _per_char_atoms(text, step=0.3):
    return _atoms([(c, step * i, step * (i + 1)) for i, c in enumerate(text)])


def _run_texts(runs):
    return ["".join(a["text"] for a in ats) for _, ats in runs]


def _even_spans(n, step=0.3, start=0.0):
    return [(start + step * i, start + step * (i + 1)) for i in range(n)]


def _timed_cue(text, spans):
    """Cue whose word_data is one (start, end) entry per char of ``text``."""
    return _cue(text, spans[0][0], spans[-1][1], spans)


# atoms 0-3 land in the first turn, 4-8 in the second, so the raw boundary falls
# exactly on the single BudouX edge of 正直この|写真欲しい.
JA_TAIL_TURNS = [(0.0, 1.2, "SPK_A"), (1.2, 3.0, "SPK_B")]


def test_ja_bound_determiner_tail_merges_contiguous_runs():
    # (a) The ja-07 defect: この is 連体詞 (Level-2 penalty 2), it is BudouX's only
    # internal edge, and the two runs abut -- so the standalone '正直この' cue must
    # not be emitted.
    cue = _timed_cue("正直この写真欲しい", _even_spans(9))
    out = format_speaker_cues([cue], JA_TAIL_TURNS, "ja")
    assert [c["text"] for c in out] == ["正直この写真欲しい"]


def test_ja_merged_run_keeps_the_longer_speakers_label():
    runs = _speaker_runs(_per_char_atoms("正直この写真欲しい"), JA_TAIL_TURNS, "ja")
    assert len(runs) == 1
    # the merged run keeps the longer speaker (1.5s vs 1.2s)
    assert runs[0][0] == "SPK_B"


def test_ja_nominalising_no_tail_does_not_merge():
    # (b) そうなの|それは: the char table scores the trailing の 2, but UniDic reads
    # it as 準体助詞 (0), so SPK_B's line survives.
    cue = _timed_cue("そうなのそれは", _even_spans(7))
    turns = [(0.0, 1.2, "SPK_A"), (1.2, 3.0, "SPK_B")]
    out = format_speaker_cues([cue], turns, "ja")
    assert [c["text"] for c in out] == ["そうなの", "それは"]


# ja-01 corpus shape: 'そんなに' 0.5s, then 'ギリギリ' 0.4s after a 0.1s gap.
JA01_SPANS = [
    (0.0, 0.125),
    (0.125, 0.25),
    (0.25, 0.375),
    (0.375, 0.5),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
]


def test_ja_adverbial_copula_tail_does_not_merge():
    # (c) The ja-01 corpus false positive: そんなに ends in に, which the Level-1
    # char table scores 2, but UniDic reads 形状詞 + 助動詞 -- a complete adverbial,
    # not a dangling case particle. The 0.1s gap is below BAD_TAIL_MAX_GAP_S, so
    # the scorer alone has to keep the two speakers apart.
    cue = _timed_cue("そんなにギリギリ", JA01_SPANS)
    turns = [(0.0, 0.55, "S4"), (0.55, 1.2, "S2")]
    out = format_speaker_cues([cue], turns, "ja")
    assert [c["text"] for c in out] == ["そんなに", "ギリギリ"]


def test_ja_bound_tail_across_an_audible_gap_does_not_merge():
    # (d) The merged shape again, but the reply starts 0.25s later. At or above
    # BAD_TAIL_MAX_GAP_S the second run is a real turn, whatever the left tail
    # looks like -- swallowing it is the EDGE_RUN_MIN_S regression class.
    cue = _timed_cue("正直この写真欲しい", _even_spans(4) + _even_spans(5, start=1.45))
    turns = [(0.0, 1.3, "SPK_A"), (1.4, 3.2, "SPK_B")]
    out = format_speaker_cues([cue], turns, "ja")
    assert [c["text"] for c in out] == ["正直この", "写真欲しい"]


def test_ja_trailing_edge_reply_after_a_beat_survives_the_gate():
    # (e) The 2026-07-03 regression class in the gate's own language: a 160ms
    # trailing reply (< MIN_RUN_S but >= EDGE_RUN_MIN_S) behind a bad tail. Only
    # the gap guard keeps it, since この is the cue's one internal edge.
    cue = _timed_cue("正直この写真", _even_spans(4) + [(1.5, 1.58), (1.58, 1.66)])
    turns = [(0.0, 1.3, "SPEAKER_02"), (1.45, 3.0, "SPEAKER_00")]
    out = format_speaker_cues([cue], turns, "ja")
    assert [c["text"] for c in out] == ["正直この", "写真"]
    assert out[1]["start"] == 1.5


def test_ja_clean_tail_on_same_shape_still_splits():
    # Negative control: identical 4+5 shape and turns, but BudouX phrase 1 ends on
    # そう (penalty 0), so the boundary is left exactly where it was.
    cue = _timed_cue("正直そう写真欲しい", _even_spans(9))
    out = format_speaker_cues([cue], JA_TAIL_TURNS, "ja")
    assert [c["text"] for c in out] == ["正直そう", "写真欲しい"]


def test_ja_bad_tail_with_cleaner_alternative_edge_left_alone():
    # 私は|この|写真欲しい: the boundary tail この scores 2, but the edge after 私は
    # scores 0, so the AND gate stays shut -- relocating the boundary is a
    # separate concern.
    cue = _timed_cue("私はこの写真欲しい", _even_spans(9))
    out = format_speaker_cues([cue], JA_TAIL_TURNS, "ja")
    assert [c["text"] for c in out] == ["私はこの", "写真欲しい"]


def test_gate_stays_shut_without_a_level2_pos_source(monkeypatch):
    # No fugashi (or VOXWEAVE_JA_POS=0) -> no Level-2 signal. The char table would
    # score この 2 and merge, so the gate degrades to doing nothing instead of
    # firing on the Level-1 score it was rejected for.
    from voxweave.core import kinsoku

    monkeypatch.setattr(kinsoku, "_load_ja_tagger", lambda: None)
    cue = _timed_cue("正直この写真欲しい", _even_spans(9))
    out = format_speaker_cues([cue], JA_TAIL_TURNS, "ja")
    assert [c["text"] for c in out] == ["正直この", "写真欲しい"]


def _split_runs(text, cut, step=0.3):
    atoms = _per_char_atoms(text, step)
    return [("SPK_A", atoms[:cut]), ("SPK_B", atoms[cut:])]


def test_bad_tail_gate_is_ja_only():
    # zh 不|可能 ('no' / 'maybe') is a real two-speaker exchange, and jieba's only
    # internal edge is the boundary. The UniDic tagger reads 不 as 接頭辞 (2), so
    # without the language guard the gate would swallow the second speaker --
    # exactly the over-fire the char-table version was rejected for.
    runs = _split_runs("不可能", 1, step=0.5)
    for lang in ("zh", "yue"):
        assert _merge_bad_tail_runs(runs, lang) == runs
    # sanity: the same predicate does fire on the language it was validated for
    assert len(_merge_bad_tail_runs(_split_runs("正直この写真欲しい", 4), "ja")) == 1


def test_bad_tail_gate_needs_a_measurable_gap():
    # A run edge with no timestamp makes the inter-run silence unmeasurable; that
    # must not be read as "contiguous".
    runs = _split_runs("正直この写真欲しい", 4)
    runs[0][1][-1]["end"] = None
    assert _merge_bad_tail_runs(runs, "ja") == runs


def test_bad_tail_gate_ceiling_is_exclusive():
    # BAD_TAIL_MAX_GAP_S is documented as the gap a merge may NOT span, but
    # 1.4 - 1.2 evaluates to 0.19999999999999996, so a nominally 0.2s beat used to
    # slip under the ceiling and merge. _run_gap carries the epsilon that closes it.
    from voxweave.diarize import BAD_TAIL_MAX_GAP_S, _run_gap

    left = [{"text": "こ", "start": 1.0, "end": 1.2}]
    right = [{"text": "写", "start": 1.4, "end": 1.6}]
    assert 1.4 - 1.2 < BAD_TAIL_MAX_GAP_S  # the raw float artifact
    assert _run_gap(left, right) >= BAD_TAIL_MAX_GAP_S

    cue = _timed_cue("正直この写真欲しい", _even_spans(4) + _even_spans(5, start=1.4))
    turns = [(0.0, 1.3, "SPK_A"), (1.35, 3.2, "SPK_B")]
    out = format_speaker_cues([cue], turns, "ja")
    assert [c["text"] for c in out] == ["正直この", "写真欲しい"]


def test_bad_tail_gate_leaves_trailing_edge_run_pin_intact():
    # Byte-identical duplicate of test_trailing_edge_second_speaker_run_kept (the
    # 2026-07-03 user-reported regression): the gate must not reopen it. zh, so
    # the gate does not even run -- the ja counterpart is
    # test_ja_trailing_edge_reply_after_a_beat_survives_the_gate.
    atoms = _atoms(
        [
            ("我", 0.0, 0.5),
            ("想", 0.5, 1.0),
            ("听", 1.0, 1.5),
            ("听", 1.5, 2.0),
            ("普", 2.0, 2.5),
            ("通", 2.5, 3.0),
            ("话", 3.0, 3.3),
            ("你", 3.8, 3.88),
            ("好", 3.88, 3.96),
        ]
    )
    turns = [(0.0, 3.5, "SPEAKER_02"), (3.74, 5.0, "SPEAKER_00")]
    runs = _speaker_runs(atoms, turns, "zh")
    assert [lb for lb, _ in runs] == ["SPEAKER_02", "SPEAKER_00"]
    assert _run_texts(runs) == ["我想听听普通话", "你好"]
