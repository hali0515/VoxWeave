"""Vendored Mel-Band Roformer (frozen third-party model code).

Source: lucidrains/BS-RoFormer (MIT), taken from the audio-separator (uvr_lib_v5) copy.

Why it is frozen in-tree: the latest PyPI bs-roformer (1.1.0) has drifted in
architecture (hyper-connections / pope were added), so the community Mel-Band Roformer
checkpoints (e.g. the Kim vocals model) no longer load: load_state_dict reports 498
missing / 120 unexpected keys. This copy matches those checkpoints (0 missing /
0 unexpected in practice).

Local modifications (each marked ``# voxweave patch`` in the source):

- ``mel_band_roformer.py`` ``MelBandRoformer.forward``: the real/imag pair tensors are
  upcast to fp32 (``.float()``) before ``torch.view_as_complex``. Under
  ``torch.autocast`` (bf16) the mask estimators emit bfloat16, which
  ``view_as_complex`` rejects; the following ``.type(stft_repr.dtype)`` cast already
  produced complex64, so downstream is unchanged, and on fp32 inputs ``.float()``
  returns the same tensor (the default fp32 path is byte-identical).

The directory is excluded from ruff and pyright (see pyproject.toml).
"""
