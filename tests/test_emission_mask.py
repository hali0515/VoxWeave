# tests/test_emission_mask.py
# VAD emission masking (stable-ts analogue): outside speech spans, non-blank
# log-probs get a soft penalty so the global CTC DP cannot park words inside
# music/silence. Blank stays untouched; spans are dilated against VAD jitter;
# no spans means no masking (never "everything is silence").
import pytest

torch = pytest.importorskip("torch")

from voxweave.align_common import (  # noqa: E402
    _VAD_MASK_DILATE_S,
    _VAD_MASK_PENALTY,
    _mask_emissions_outside_speech,
)

SR = 16000
SPF = 0.02  # seconds per frame at 320x downsample


def _logp(t=50, v=4):
    return torch.zeros(t, v)  # uniform; only relative values matter


def test_nonspeech_penalized_blank_untouched():
    out = _mask_emissions_outside_speech(_logp(), [(0.0, 0.4)], 50 * 320, SR, 0)
    assert torch.all(out[:20] == 0)  # speech zone untouched
    assert out[40, 1] == -_VAD_MASK_PENALTY  # silence: non-blank penalized
    assert out[40, 0] == 0  # blank column untouched


def test_no_spans_no_mask():
    lp = _logp()
    assert _mask_emissions_outside_speech(lp, [], 50 * 320, SR, 0) is lp
    assert _mask_emissions_outside_speech(lp, None, 50 * 320, SR, 0) is lp


def test_all_speech_short_circuit():
    lp = _logp()
    assert _mask_emissions_outside_speech(lp, [(0.0, 10.0)], 50 * 320, SR, 0) is lp


def test_dilation_protects_span_edges():
    out = _mask_emissions_outside_speech(_logp(), [(0.5, 0.6)], 50 * 320, SR, 0)
    first_kept = int((0.5 - _VAD_MASK_DILATE_S) / SPF)
    assert torch.all(out[first_kept] == 0)
    assert out[first_kept - 2, 1] < 0


def test_dp_prefers_speech_after_mask():
    # behavioral: token evidence is slightly STRONGER inside a silence region;
    # unmasked forced_align picks the silence placement, masked picks speech.
    import torchaudio.functional as AF

    t, blank = 30, 0
    logits = torch.full((t, 3), -8.0)
    logits[:, blank] = 0.0
    logits[5:8, 1] = 1.0  # in-speech candidate
    logits[20:23, 1] = 1.5  # in-silence candidate (stronger)
    lp = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([[1]], dtype=torch.int32)

    aligned, _ = AF.forced_align(lp.unsqueeze(0).contiguous(), targets, blank=blank)
    pos_unmasked = torch.nonzero(aligned[0] == 1).flatten()
    assert pos_unmasked.min() >= 20  # control: silence placement wins unmasked

    spans = [(0.0, 10 * SPF)]  # speech = first 10 frames
    masked = _mask_emissions_outside_speech(lp, spans, t * 320, SR, blank)
    aligned2, _ = AF.forced_align(
        masked.unsqueeze(0).contiguous(), targets, blank=blank
    )
    pos_masked = torch.nonzero(aligned2[0] == 1).flatten()
    assert pos_masked.max() < 10 + int(_VAD_MASK_DILATE_S / SPF) + 1
