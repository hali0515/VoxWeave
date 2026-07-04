import numpy as np

from voxweave import align_common


def _fake_pass(recorder):
    def pass_fn(wav, texts, offset_s):
        recorder["len"] = int(wav.shape[-1])
        recorder["offset"] = offset_s
        return [
            [{"text": "w", "start": 0.0, "end": 0.1}]
        ]  # one block, one unit at local t=0

    return pass_fn


def test_dp_chunked_pass_crops_to_envelope_when_opted_in():
    # With crop_to_envelope=True (transcribe path: bounds are fresh VAD chunk windows), the
    # single global pass is cropped to [first_bound-pad, last_bound+pad] so a leading/trailing
    # skipped song (present in wav, absent from text) can't host stretched tokens.
    sr = 16000
    wav = np.zeros(
        int(60.0 * sr), dtype="float32"
    )  # 60s -> 3000 frames, well under budget
    bounds = [(30.0, 50.0)]  # 0-30 (skipped song) and 50-60 are untranscribed
    rec: dict = {}
    out = align_common._dp_chunked_pass(
        wav, sr, ["hello"], bounds, _fake_pass(rec), "TEST", crop_to_envelope=True
    )

    pad = align_common.CTC_ENVELOPE_PAD_SEC
    lo, hi = 30.0 - pad, 50.0 + pad
    assert rec["len"] == int(hi * sr) - int(
        lo * sr
    )  # pass_fn saw only the cropped slice
    assert rec["offset"] == lo
    assert out[0][0]["start"] == lo  # local t=0 shifted back to absolute lo


def test_dp_chunked_pass_default_does_not_crop():
    # Routing-free invariant: WITHOUT the opt-in the full wav is passed even when bounds sit
    # far from the edges. The align subcommand's bounds are input-VTT timestamps — exactly
    # what may be wrong — so cropping to them would confine words to a possibly-shifted
    # window (the documented B-path failure). Default must stay full-pass at offset 0.
    sr = 16000
    wav = np.zeros(int(60.0 * sr), dtype="float32")
    rec: dict = {}
    out = align_common._dp_chunked_pass(
        wav, sr, ["hello"], [(30.0, 50.0)], _fake_pass(rec), "TEST"
    )

    assert rec["len"] == wav.shape[-1]
    assert rec["offset"] == 0.0
    assert out[0][0]["start"] == 0.0


def test_dp_chunked_pass_no_crop_when_envelope_is_full():
    # Bounds already span the whole wav -> pad clamps to [0, total] -> no crop, offset 0.
    sr = 16000
    wav = np.zeros(int(40.0 * sr), dtype="float32")
    rec: dict = {}
    out = align_common._dp_chunked_pass(
        wav, sr, ["x"], [(0.0, 40.0)], _fake_pass(rec), "TEST", crop_to_envelope=True
    )

    assert rec["len"] == wav.shape[-1]
    assert rec["offset"] == 0.0
    assert out[0][0]["start"] == 0.0


def test_dp_chunked_pass_no_bounds_runs_full_pass():
    # No usable bounds -> fall back to a single full-wav pass at offset 0 (no crop).
    sr = 16000
    wav = np.zeros(int(20.0 * sr), dtype="float32")
    rec: dict = {}
    align_common._dp_chunked_pass(
        wav, sr, ["x"], None, _fake_pass(rec), "TEST", crop_to_envelope=True
    )

    assert rec["len"] == wav.shape[-1]
    assert rec["offset"] == 0.0
