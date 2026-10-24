import itertools
import sys
import types

import numpy as np
import pytest

from voxweave import backend
from voxweave.qwen_decode import decode_slot_times, expected_classes, monotone_path

SEGMENT_MS = 80.0


def _best_score(scores, strict_pairs):
    slots, classes = scores.shape
    best = -np.inf
    for path in itertools.combinations_with_replacement(range(classes), slots):
        if strict_pairs and any(path[s] <= path[s - 1] for s in range(1, slots, 2)):
            continue
        best = max(best, sum(scores[s, c] for s, c in enumerate(path)))
    return best


@pytest.mark.parametrize("strict_pairs", [False, True])
def test_monotone_path_is_optimal_and_valid(strict_pairs):
    rng = np.random.default_rng(7)
    for _ in range(25):
        scores = rng.normal(size=(6, 7))
        path = monotone_path(scores, strict_pairs)
        assert np.all(np.diff(path) >= 0)
        if strict_pairs:
            assert all(path[s] > path[s - 1] for s in range(1, 6, 2))
        total = sum(scores[s, c] for s, c in enumerate(path))
        assert total == pytest.approx(_best_score(scores, strict_pairs))


def test_monotone_path_ignores_a_per_slot_offset():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(8, 9))
    shifted = scores + rng.normal(size=(8, 1)) * 5
    assert list(monotone_path(scores, True)) == list(monotone_path(shifted, True))


def test_expected_classes_leans_toward_the_heavier_neighbour():
    scores = np.full((1, 10), -30.0)
    scores[0, 4] = 5.0
    scores[0, 5] = 4.0
    got = expected_classes(scores, np.array([4]), 3)[0]
    assert 4.0 < got < 4.5


def test_expected_classes_clips_at_both_ends():
    scores = np.full((2, 6), -30.0)
    scores[0, 0] = 1.0
    scores[1, 5] = 1.0
    got = expected_classes(scores, np.array([0, 5]), 3)
    assert got == pytest.approx([0.0, 5.0], abs=1e-6)


def _peaks(classes_per_slot, width=40):
    scores = np.full((len(classes_per_slot), width), -20.0)
    for slot, cls in enumerate(classes_per_slot):
        scores[slot, cls] = 10.0
    return scores


def test_decode_places_peaks_on_the_grid():
    times = decode_slot_times(_peaks([2, 4, 5, 9]), SEGMENT_MS, 4.0)
    assert list(times) == pytest.approx([0.16, 0.32, 0.4, 0.72], abs=1e-3)


def test_decode_moves_boundaries_off_the_grid_by_the_probability_lean():
    scores = _peaks([2, 4])
    scores[0, 3] = 9.0
    times = decode_slot_times(scores, SEGMENT_MS, 4.0)
    assert 0.16 < times[0] < 0.24
    assert times[1] == pytest.approx(0.32, abs=1e-3)


def test_monotone_path_separates_a_start_and_end_that_peak_on_one_class():
    path = monotone_path(_peaks([6, 6, 6, 6]), strict_pairs=True)
    assert path[1] > path[0]
    assert path[3] > path[2]


def test_decode_gives_a_near_tie_end_its_own_time():
    # start and end peak on one class, the end's runner-up is the next class: argmax + LIS makes a
    # zero-duration word, the strict path puts the end on the runner-up and the soft step keeps it
    scores = _peaks([6, 6, 12, 12])
    scores[1, 7] = 9.5
    scores[3, 13] = 9.5
    times = decode_slot_times(scores, SEGMENT_MS, 4.0)
    assert np.all(np.diff(times) >= 0)
    assert times[1] > times[0]
    assert times[3] > times[2]


def test_decode_repairs_a_misplaced_slot_without_copying_a_neighbour():
    times = decode_slot_times(_peaks([2, 3, 20, 5, 6, 7]), SEGMENT_MS, 4.0)
    assert np.all(np.diff(times) >= 0)
    assert len(set(times.tolist())) == 6 or times[2] > times[1]


def test_decode_never_picks_a_class_past_the_audio():
    scores = _peaks([2, 4, 30, 35], width=60)
    times = decode_slot_times(scores, SEGMENT_MS, 2.0)
    assert times.max() <= (int(2.0 * 1000 / SEGMENT_MS) + 3) * SEGMENT_MS / 1000


def test_decode_survives_more_words_than_the_audio_has_classes():
    words = 12
    scores = _peaks([min(i, 4) for i in range(2 * words)], width=40)
    times = decode_slot_times(scores, SEGMENT_MS, 0.4)
    assert times.shape == (2 * words,)
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) >= 0)


@pytest.mark.parametrize("slots", [0, 3])
def test_decode_rejects_an_unpaired_slot_count(slots):
    with pytest.raises(ValueError, match="timestamp slots"):
        decode_slot_times(np.zeros((slots, 10)), SEGMENT_MS, 1.0)


def test_decode_output_has_millisecond_resolution():
    rng = np.random.default_rng(11)
    times = decode_slot_times(rng.normal(size=(10, 30)) * 3, SEGMENT_MS, 2.0)
    assert np.allclose(times, np.round(times, 3))


# --- backend._qwen_units: same prompt and forward pass as the wrapper, qwen_decode for the output ---


class _Batch(dict):
    def to(self, *_args):
        return self


def _fake_aligner(torch, slot_classes, words):
    class _Processor:
        timestamp_marker = "<timestamp>"

        def __call__(self, text, audio, return_tensors, padding):
            assert text == ["<prompt>"]
            ids = torch.tensor([[1, 2, 99, 99, 3, 99, 99]])
            return _Batch(input_ids=ids)

    class _Thinker:
        def __call__(self, **_inputs):
            logits = torch.full((1, 7, 40), -20.0)
            for row, cls in zip((2, 3, 5, 6), slot_classes):
                logits[0, row, cls] = 10.0
            return types.SimpleNamespace(logits=logits)

    class _AlignerProcessor:
        def encode_timestamp(self, text, language):
            self.seen = (text, language)
            return words, "<prompt>"

    return types.SimpleNamespace(
        processor=_Processor(),
        aligner_processor=_AlignerProcessor(),
        model=types.SimpleNamespace(
            device="cpu", dtype=torch.float32, thinker=_Thinker()
        ),
        timestamp_token_id=99,
        timestamp_segment_time=SEGMENT_MS,
    )


def _install_fake_qwen_asr(monkeypatch, seconds):
    sample_rate = 16000
    utils = types.ModuleType("qwen_asr.inference.utils")
    utils.SAMPLE_RATE = sample_rate
    utils.normalize_audios = lambda path: [
        np.zeros(int(seconds * sample_rate), dtype=np.float32)
    ]
    inference = types.ModuleType("qwen_asr.inference")
    inference.utils = utils
    package = types.ModuleType("qwen_asr")
    package.inference = inference
    monkeypatch.setitem(sys.modules, "qwen_asr", package)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference", inference)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference.utils", utils)


def test_qwen_units_decodes_the_timestamp_slots(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    _install_fake_qwen_asr(monkeypatch, seconds=4.0)
    aligner = _fake_aligner(torch, [2, 4, 4, 9], ["ni", "hao"])
    units = backend._qwen_units(aligner, tmp_path / "a.wav", "ni hao", "Chinese")
    assert aligner.aligner_processor.seen == ("ni hao", "Chinese")
    assert [u["text"] for u in units] == ["ni", "hao"]
    assert units[0]["start"] == pytest.approx(0.16, abs=1e-3)
    assert units[0]["end"] == pytest.approx(0.32, abs=1e-3)
    assert units[1]["start"] >= units[0]["end"]
    assert units[1]["end"] > units[1]["start"]


def test_qwen_units_rejects_text_with_no_alignable_token(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    _install_fake_qwen_asr(monkeypatch, seconds=1.0)
    aligner = _fake_aligner(torch, [2, 4, 4, 9], [])
    with pytest.raises(ValueError, match="no alignable tokens"):
        backend._qwen_units(aligner, tmp_path / "a.wav", "。。", "Chinese")
