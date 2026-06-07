from voxweave.chunking import pack_speech_segments, plan_dp_chunks


def test_packs_into_single_chunk_when_short():
    segs = [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 5.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert chunks == [{"start": 0.0, "end": 5.0, "offset": 0.0}]


def test_splits_at_silence_when_exceeding_max():
    # three segments, each 100s, 1s gap; max=240 -> seg1+seg2 one chunk, seg3 one chunk
    segs = [
        {"start": 0.0, "end": 100.0},
        {"start": 101.0, "end": 201.0},
        {"start": 202.0, "end": 302.0},
    ]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert len(chunks) == 2
    assert chunks[0] == {"start": 0.0, "end": 201.0, "offset": 0.0}
    assert chunks[1] == {"start": 202.0, "end": 302.0, "offset": 202.0}


def test_single_segment_longer_than_max_is_hard_split():
    # 500s continuous speech with no silence, max=240 -> hard cut (word cuts tolerated)
    segs = [{"start": 0.0, "end": 500.0}]
    chunks = pack_speech_segments(segs, max_sec=240.0)
    assert len(chunks) == 3
    assert chunks[0]["start"] == 0.0 and chunks[0]["end"] == 240.0
    assert chunks[1]["start"] == 240.0 and chunks[1]["end"] == 480.0
    assert chunks[2]["start"] == 480.0 and chunks[2]["end"] == 500.0
    assert [c["offset"] for c in chunks] == [0.0, 240.0, 480.0]


def test_empty_returns_empty():
    assert pack_speech_segments([], max_sec=240.0) == []


# --- plan_dp_chunks: silence-anchored DP chunking for over-budget alignment ---


def test_dp_within_budget_is_single_chunk():
    bounds = [(0.0, 2.0), (3.0, 5.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5)
    # one chunk over all cues; crop padded at file edges (left clamped to 0)
    assert chunks == [{"lo": 0, "hi": 2, "start": 0.0, "end": 5.5}]


def test_dp_empty_returns_empty():
    assert plan_dp_chunks([], max_sec=240.0) == []


def test_dp_splits_at_large_gap_when_over_budget():
    # three 100s cues, 2s gaps; budget 240 -> [cue0,cue1] + [cue2]
    bounds = [(0.0, 100.0), (102.0, 202.0), (204.0, 304.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, min_gap_sec=1.5, pad_sec=0.5)
    assert len(chunks) == 2
    # boundary at gap midpoint (202+204)/2 = 203; adjacent crops meet there
    assert chunks[0] == {"lo": 0, "hi": 2, "start": 0.0, "end": 203.0}
    assert chunks[1] == {"lo": 2, "hi": 3, "start": 203.0, "end": 304.5}


def test_dp_prefers_large_gap_over_in_budget_small_gap():
    # small gap (0.5s) after cue0, large gap (2s) after cue1; both within budget.
    # must cut at the large gap, not the earlier small one.
    bounds = [(0.0, 100.0), (100.5, 200.0), (202.0, 302.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, min_gap_sec=1.5, pad_sec=0.5)
    assert [(c["lo"], c["hi"]) for c in chunks] == [(0, 2), (2, 3)]


def test_dp_falls_back_to_cue_boundary_when_no_large_gap():
    # all gaps tiny (<min_gap) but total > budget: cut at latest cue boundary in budget.
    # cue boundaries never split words (smart_split invariant), so this stays word-safe.
    bounds = [(0.0, 100.0), (100.5, 200.5), (201.0, 301.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, min_gap_sec=1.5, pad_sec=0.5)
    assert [(c["lo"], c["hi"]) for c in chunks] == [(0, 2), (2, 3)]


def test_dp_single_oversized_cue_is_its_own_chunk():
    bounds = [(0.0, 300.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5)
    assert [(c["lo"], c["hi"]) for c in chunks] == [(0, 1)]


def test_dp_timestampless_insertion_cue_rides_along():
    # None-bound cue (insertion / empty) carries no anchor; it stays in its chunk.
    bounds = [(0.0, 2.0), None, (3.0, 5.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5)
    assert chunks == [{"lo": 0, "hi": 3, "start": 0.0, "end": 5.5}]


def test_dp_audio_end_caps_last_chunk():
    bounds = [(0.0, 2.0), (3.0, 5.0)]
    chunks = plan_dp_chunks(bounds, max_sec=240.0, pad_sec=0.5, audio_end=5.2)
    assert chunks[-1]["end"] == 5.2
