import numpy as np

from voxweave import songdet
from voxweave.songdet import (
    IDX_MUSIC,
    IDX_SING,
    IDX_SPEECH,
    excise_spans_from_segments,
    expand_spans_to_voiced_blocks,
    filter_short_spans,
    group_segments_by_spans,
    merge_spans,
    sing_flags,
    song_flags,
    speech_flags,
)


def test_ensure_panns_labels_noop_when_present(monkeypatch, tmp_path):
    # CSV already at ~/panns_data -> no download attempted (urllib/hf would raise if called)
    monkeypatch.setattr(songdet.Path, "home", staticmethod(lambda: tmp_path))
    dst = tmp_path / "panns_data" / "class_labels_indices.csv"
    dst.parent.mkdir(parents=True)
    dst.write_text("index,mid,display_name\n")

    def _boom(*a, **k):
        raise AssertionError("must not download when csv already present")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    songdet._ensure_panns_labels()  # should return immediately, no exception


def test_ensure_panns_labels_falls_back_to_url(monkeypatch, tmp_path):
    # HF download fails -> urllib fetches the canonical csv and writes it to ~/panns_data
    monkeypatch.setattr(songdet.Path, "home", staticmethod(lambda: tmp_path))

    def _hf_fail(*a, **k):
        raise RuntimeError("repo has no csv")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hf_fail)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"index,mid,display_name\n0,/m/x,Speech\n"

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    songdet._ensure_panns_labels()
    dst = tmp_path / "panns_data" / "class_labels_indices.csv"
    assert dst.read_bytes().startswith(b"index,mid,display_name")


def _probs(rows: list[dict]) -> np.ndarray:
    """rows: per-window dicts {'speech':v,'sing':v,'music':v} -> (n,527) probability matrix."""
    p = np.zeros((len(rows), 527), dtype="float32")
    for i, r in enumerate(rows):
        p[i, IDX_SPEECH[0]] = r.get("speech", 0.0)
        p[i, IDX_SING[0]] = r.get("sing", 0.0)
        p[i, IDX_MUSIC[0]] = r.get("music", 0.0)
    return p


def test_song_flags_speech_not_flagged():
    p = _probs([{"speech": 0.8, "sing": 0.0, "music": 0.05}])
    assert song_flags(p).tolist() == [False]


def test_song_flags_singing_flagged():
    # sing exceeds speech and > 0.15
    p = _probs([{"speech": 0.05, "sing": 0.5, "music": 0.1}])
    assert song_flags(p).tolist() == [True]


def test_song_flags_music_low_speech_flagged():
    # music>0.30 and speech<0.25 -> counts as music even with low sing
    p = _probs([{"speech": 0.10, "sing": 0.05, "music": 0.6}])
    assert song_flags(p).tolist() == [True]


def test_song_flags_dialogue_over_bgm_not_flagged():
    # dialogue over BGM: speech is high -> not flagged (music on separated vocals is already low; this tests the criterion itself)
    p = _probs([{"speech": 0.7, "sing": 0.0, "music": 0.6}])
    assert song_flags(p).tolist() == [False]


def test_song_flags_silence_baseline_not_flagged():
    p = _probs([{"speech": 0.05, "sing": 0.0, "music": 0.13}])
    assert song_flags(p).tolist() == [False]


def test_merge_spans_contiguous():
    flags = np.array([True, True, True])
    spans = merge_spans(flags, [0.0, 1.0, 2.0])
    assert spans == [(0.0, 4.0)]  # win=2, last window start=2 -> end=4


def test_merge_spans_drops_short():
    flags = np.array([True, False, False])
    # single window [0,2] length 2 < min_span 3 -> dropped
    assert merge_spans(flags, [0.0, 1.0, 2.0]) == []


def test_merge_spans_splits_on_gap():
    # starts 0,1 (song) then 6,7 (song), gap 6-2=4 > gap_merge 2 -> two separate spans
    flags = np.array([True, True, False, False, False, False, True, True])
    starts = [float(i) for i in range(8)]
    spans = merge_spans(flags, starts)
    assert spans == [(0.0, 3.0), (6.0, 9.0)]


def test_excise_cuts_song_out_of_segments():
    spans = [(10.0, 20.0)]
    segs = [
        {"start": 0.0, "end": 5.0},  # outside: untouched
        {"start": 12.0, "end": 18.0},  # fully inside: removed entirely
        {"start": 8.0, "end": 12.0},  # straddles span start: dialogue half survives
        {"start": 19.0, "end": 25.0},  # straddles span end: dialogue half survives
    ]
    kept, cut = excise_spans_from_segments(segs, spans)
    assert cut == [(10.0, 20.0)]  # no silences -> raw span edges
    assert kept == [
        {"start": 0.0, "end": 5.0},
        {"start": 8.0, "end": 10.0},
        {"start": 20.0, "end": 25.0},
    ]


def test_excise_mixed_segment_keeps_flanking_dialogue():
    # The user-reported shape: one VAD segment = speech + brief pause + hummed bar + speech.
    # Whole-segment dropping would lose both speech runs; excision keeps them.
    segs = [{"start": 100.0, "end": 130.0}]
    spans = [(112.0, 118.0)]  # hummed bar inside the segment
    kept, cut = excise_spans_from_segments(segs, spans)
    assert kept == [
        {"start": 100.0, "end": 112.0},
        {"start": 118.0, "end": 130.0},
    ]
    assert cut == [(112.0, 118.0)]


def test_excise_snaps_cuts_into_silence():
    # Fine-VAD silences near the coarse PANNs edges: cuts land at silence midpoints,
    # so the song leaves with its flanking silence and no dialogue word is bisected.
    segs = [{"start": 100.0, "end": 130.0}]
    spans = [(112.0, 118.0)]
    silences = [(110.8, 111.4), (118.6, 119.0)]  # real pauses around the hum
    kept, cut = excise_spans_from_segments(segs, spans, silences=silences)
    assert cut == [(111.1, 118.8)]  # snapped to silence midpoints
    assert kept == [
        {"start": 100.0, "end": 111.1},
        {"start": 118.8, "end": 130.0},
    ]


def test_excise_snap_ignores_far_silence_and_keeps_in_silence_points():
    # cut already inside a silence stays put; silences beyond snap_sec are ignored
    segs = [{"start": 0.0, "end": 40.0}]
    spans = [(10.0, 20.0)]
    silences = [
        (9.8, 10.2),
        (26.0, 27.0),
    ]  # 10.0 sits in silence; 26.5 too far from 20.0
    kept, cut = excise_spans_from_segments(segs, spans, silences=silences, snap_sec=1.5)
    assert cut == [(10.0, 20.0)]


def test_excise_drops_sub_minimum_shards():
    segs = [{"start": 9.8, "end": 20.0}]  # only 0.2s precedes the span
    kept, _ = excise_spans_from_segments(segs, [(10.0, 20.0)], min_keep_sec=0.4)
    assert kept == []  # 0.2s shard dropped, nothing else remains


def test_excise_no_spans_keeps_all():
    segs = [{"start": 0.0, "end": 5.0}]
    kept, cut = excise_spans_from_segments(segs, [])
    assert kept == segs and cut == []


def test_expand_spans_grabs_rap_in_same_block():
    # OP: rap verse 36-64 (PANNs classifies as speech) + singing chorus 66-86; one continuous voiced block.
    # Only chorus detected (65-86) -> expanded to full block 36-86, pulling in the rap.
    segs = [
        {"start": 36.0, "end": 50.0},
        {"start": 52.0, "end": 64.0},
        {"start": 66.0, "end": 86.0},
        {
            "start": 99.0,
            "end": 120.0,
        },  # second singing segment, 13s true silence away -> separate block
    ]
    spans = [(65.0, 86.0), (99.0, 120.0)]
    out = expand_spans_to_voiced_blocks(segs, spans)
    assert out == [(36.0, 86.0), (99.0, 120.0)]


def test_expand_spans_does_not_eat_far_dialogue():
    # dialogue block (159-180) separated from song span by silence -> not absorbed
    segs = [
        {"start": 60.0, "end": 86.0},
        {"start": 159.0, "end": 170.0},
        {"start": 171.0, "end": 180.0},
    ]
    spans = [(65.0, 86.0)]
    out = expand_spans_to_voiced_blocks(segs, spans)
    assert out == [(60.0, 86.0)]


def test_expand_spans_empty_noop():
    segs = [{"start": 0.0, "end": 5.0}]
    assert expand_spans_to_voiced_blocks(segs, []) == []


# --------------------------------------------------------------------------- #
# sing_flags + expandable gate on expansion (fixes mid-stream BGM absorbing dialogue)
# --------------------------------------------------------------------------- #
def test_sing_flags_singing_true():
    p = _probs([{"speech": 0.05, "sing": 0.5, "music": 0.1}])
    assert sing_flags(p).tolist() == [True]


def test_sing_flags_pure_music_false():
    # pure-instrumental BGM: music dominant, sing~0 -> not classified as "contains singing" (though song_flags still flags it as music)
    p = _probs([{"speech": 0.07, "sing": 0.04, "music": 0.72}])
    assert sing_flags(p).tolist() == [False]
    assert song_flags(p).tolist() == [True]


def test_sing_flags_speech_false():
    p = _probs([{"speech": 0.8, "sing": 0.0, "music": 0.05}])
    assert sing_flags(p).tolist() == [False]


def test_expand_music_only_span_does_not_eat_following_speech():
    # Reproduces real bug: BGM (148-151, pure instrumental) ends and the host speaks immediately (153-156, same voiced block, gap<3s).
    # Pure-instrumental span is not in expandable -> does not absorb the whole block -> speech segments 153/155 are preserved.
    segs = [
        {"start": 148.5, "end": 149.5},  # BGM
        {"start": 150.4, "end": 150.9},  # BGM
        {"start": 153.0, "end": 153.8},  # speech
        {"start": 155.4, "end": 156.3},  # speech
    ]
    spans = [(148.0, 151.0)]  # pure-instrumental span detected
    out = expand_spans_to_voiced_blocks(segs, spans, expandable=[])
    assert out == [(148.0, 151.0)]  # no expansion
    kept, _ = excise_spans_from_segments(segs, out)
    assert {"start": 153.0, "end": 153.8} in kept  # speech preserved
    assert {"start": 155.4, "end": 156.3} in kept


def test_filter_short_spans_drops_brief_bgm():
    # 3s misclassified instrumental BGM dropped (transcribe as content, not skip); real OP/ED long spans kept (skipped)
    spans = [(148.0, 151.0), (10.0, 80.0)]
    assert filter_short_spans(spans, min_sec=8.0) == [(10.0, 80.0)]


def test_filter_short_spans_keeps_all_when_long():
    spans = [(10.0, 80.0), (100.0, 160.0)]
    assert filter_short_spans(spans, min_sec=8.0) == spans


def test_filter_short_spans_empty():
    assert filter_short_spans([], min_sec=8.0) == []


def test_expand_singing_span_still_grabs_rap_with_expandable():
    # singing span is in expandable -> still expands to grab rap in the same block (no regression)
    segs = [
        {"start": 36.0, "end": 50.0},  # rap (classified as speech)
        {"start": 52.0, "end": 64.0},  # rap
        {"start": 66.0, "end": 86.0},  # singing chorus (detected)
    ]
    spans = [(65.0, 86.0)]
    out = expand_spans_to_voiced_blocks(segs, spans, expandable=[(65.0, 86.0)])
    assert out == [(36.0, 86.0)]


# --------------------------------------------------------------------------- #
# clean-dialogue signature + expansion edge-trimming (fixes ED singing span overshooting and absorbing adjacent dialogue)
# --------------------------------------------------------------------------- #
def test_speech_flags_clean_dialogue_true():
    # clean dialogue on separated vocals: speech dominant, almost no singing/instrumental residue
    p = _probs([{"speech": 0.7, "sing": 0.0, "music": 0.05}])
    assert speech_flags(p).tolist() == [True]


def test_speech_flags_song_false():
    p = _probs([{"speech": 0.05, "sing": 0.3, "music": 0.4}])
    assert speech_flags(p).tolist() == [False]


def test_speech_flags_rap_with_residual_music_false():
    # carries rhythmic/instrumental residue (music>=0.2 or sing>=0.1) -> not clean dialogue
    # -> rap verse is NOT trimmed as dialogue (preserves pit-2 protection)
    assert speech_flags(
        _probs([{"speech": 0.7, "sing": 0.0, "music": 0.25}])
    ).tolist() == [False]
    assert speech_flags(
        _probs([{"speech": 0.7, "sing": 0.15, "music": 0.05}])
    ).tolist() == [False]


def test_speech_flags_quiet_pause_false():
    # speech score too low (pause / soft voice) -> not flagged as clean dialogue
    p = _probs([{"speech": 0.3, "sing": 0.0, "music": 0.05}])
    assert speech_flags(p).tolist() == [False]


def test_expand_trims_leading_clean_speech_dialogue():
    # Real ED bug (block A): dialogue 1226-1264 is flush against the ED opening 1266-1356, same voiced block (gap<3, ED is singing).
    # protect marks the dialogue -> expansion trims it from the left edge, keeping only the song core -> dialogue not absorbed.
    segs = [
        {"start": 1226.0, "end": 1264.0},  # dialogue (clean speech)
        {"start": 1266.0, "end": 1271.0},  # ED opening (singing)
        {"start": 1273.0, "end": 1356.0},  # ED body (singing)
    ]
    spans = [(1266.0, 1356.0)]
    out = expand_spans_to_voiced_blocks(
        segs, spans, expandable=spans, protect=[(1226.0, 1264.0)]
    )
    assert out == [(1266.0, 1356.0)]  # dialogue 1226-1264 preserved


def test_expand_trims_trailing_clean_speech_dialogue():
    # Block C: ED tail 1320-1356 + dialogue 1357-1410 in same block -> trailing dialogue trimmed
    segs = [
        {"start": 1320.0, "end": 1356.0},  # ED tail (singing)
        {"start": 1357.0, "end": 1410.0},  # dialogue
    ]
    spans = [(1266.0, 1356.0)]
    out = expand_spans_to_voiced_blocks(
        segs, spans, expandable=spans, protect=[(1357.0, 1410.0)]
    )
    assert out == [(1266.0, 1356.0)]  # trailing dialogue preserved


def test_expand_keeps_interior_rap_between_choruses():
    # Pit-2 no regression: rap verse between two choruses (interior to the block) -> NOT trimmed even if protect marks it; whole block absorbed
    segs = [
        {"start": 66.0, "end": 80.0},  # chorus1 (singing, detected)
        {"start": 82.0, "end": 96.0},  # rap verse (clean speech, interior)
        {"start": 98.0, "end": 112.0},  # chorus2 (singing, detected)
    ]
    spans = [(66.0, 80.0), (98.0, 112.0)]
    out = expand_spans_to_voiced_blocks(
        segs, spans, expandable=spans, protect=[(82.0, 96.0)]
    )
    assert out == [(66.0, 112.0)]  # interior verse stays within the song span


def test_expand_trims_dialogue_past_isolated_sting():
    # Isekai bug: an OP core (26-89) merged via one marginal VAD gap with a long dialogue block
    # that contains a brief embedded sting (138-141) far (49s) from the core. The sting must NOT
    # anchor the whole dialogue tail into the song: dialogue 92-145 is freed, sting kept as its
    # own (short) span. Old code stopped the trailing trim at the sting -> absorbed 26-145.
    segs = [
        {"start": 26.0, "end": 89.0},  # OP core (singing, expandable)
        {"start": 92.0, "end": 137.0},  # clean dialogue
        {"start": 138.0, "end": 145.0},  # dialogue tail clipping a 3s sting (138-141)
    ]
    spans = [(26.0, 89.0), (138.0, 141.0)]  # OP core + isolated sting
    out = expand_spans_to_voiced_blocks(
        segs, spans, expandable=[(26.0, 89.0)], protect=[(92.0, 145.0)]
    )
    assert out == [(26.0, 89.0), (138.0, 141.0)]  # dialogue freed; sting isolated


def test_expand_trims_dialogue_in_gap_between_core_and_clustered_sting():
    # Cores are the cluster's MEMBER spans, not its hull: dialogue in the 12s gap between the
    # OP end and a clustered (non-expandable) sting overlaps no actual song span and must be
    # trimmed, not absorbed. (A hull 100-178 would cover 161-169 and swallow its subtitles.)
    segs = [
        {"start": 100.0, "end": 160.0},  # OP core
        {"start": 161.0, "end": 169.0},  # clean dialogue in the gap
    ]
    spans = [
        (100.0, 160.0),
        (172.0, 178.0),
    ]  # OP + sting 12s past OP end (same cluster)
    out = expand_spans_to_voiced_blocks(
        segs, spans, expandable=[(100.0, 160.0)], protect=[(161.0, 169.0)]
    )
    assert out == [
        (100.0, 160.0),
        (172.0, 178.0),
    ]  # gap dialogue trimmed, spans untouched


def test_expand_keeps_short_sting_clustered_with_core():
    # Counterpart: a short sung span within SONG_CORE_MERGE_SEC of a long core (a rap-OP intro
    # ~12s before the chorus) is part of the song -> the block absorbs through the interlude.
    segs = [
        {"start": 60.0, "end": 63.0},  # short sung intro
        {"start": 64.0, "end": 120.0},  # rap interlude (clean) + chorus (expandable)
    ]
    spans = [
        (60.0, 63.0),
        (75.0, 120.0),
    ]  # intro 12s before chorus (< SONG_CORE_MERGE_SEC)
    out = expand_spans_to_voiced_blocks(
        segs, spans, expandable=[(75.0, 120.0)], protect=[(64.0, 75.0)]
    )
    assert out == [(60.0, 120.0)]  # intro clustered into the core; whole block absorbed


def test_expand_no_protect_is_legacy_whole_block():
    # protect=None (default) -> legacy whole-block absorption behavior, backward compatible
    segs = [
        {"start": 1226.0, "end": 1264.0},
        {"start": 1266.0, "end": 1271.0},
        {"start": 1273.0, "end": 1356.0},
    ]
    spans = [(1266.0, 1356.0)]
    out = expand_spans_to_voiced_blocks(segs, spans, expandable=spans)
    assert out == [(1226.0, 1356.0)]  # blindly absorbs the whole block


def test_group_segments_breaks_at_span():
    # song span 65-86 falls between segments 60-64 and 123-130 -> split into two groups
    spans = [(65.0, 86.0), (99.0, 120.0)]
    segs = [
        {"start": 55.0, "end": 60.0},
        {"start": 60.0, "end": 64.0},
        {"start": 123.0, "end": 130.0},
    ]
    groups = group_segments_by_spans(segs, spans)
    assert groups == [
        [{"start": 55.0, "end": 60.0}, {"start": 60.0, "end": 64.0}],
        [{"start": 123.0, "end": 130.0}],
    ]


def test_group_segments_no_span_between_stays_one():
    spans = [(200.0, 210.0)]
    segs = [{"start": 10.0, "end": 20.0}, {"start": 21.0, "end": 30.0}]
    assert group_segments_by_spans(segs, segs and spans) == [segs]


def test_group_segments_no_spans():
    segs = [{"start": 0.0, "end": 5.0}]
    assert group_segments_by_spans(segs, []) == [segs]
    assert group_segments_by_spans([], [(1.0, 2.0)]) == []
