"""Speaker-embedding networks behind the decoupled voiceprint embedders.

Imported lazily by :mod:`voxweave.voiceembed` (this module imports torch). Each
builder takes the object ``torch.load(weights_only=True)`` returned for a
hash-verified checkpoint and returns the network plus the embedding dimension
its head declares, so the caller can check it against the registry before any
vector is trusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

# anime-va: the published checkpoint is SpeechBrain's ECAPA-TDNN with every
# BatchNorm1d replaced by GroupNorm (32 groups), trained on waveforms scaled by
# 32768 (int16 range) before the Fbank front end. Both facts come from the model
# card / reference implementation and are reproduced here, not imported.
ANIME_VA_NORM_GROUPS = 32
ANIME_VA_WAVEFORM_SCALE = 32768.0
ANIME_VA_N_MELS = 80
ANIME_VA_CHANNELS = [1024, 1024, 1024, 1024, 3072]
ANIME_VA_KERNEL_SIZES = [5, 3, 3, 3, 1]
ANIME_VA_EMBEDDING_DIM = 192


class CheckpointLayoutError(ValueError):
    """A verified checkpoint does not have the layout its builder expects."""


class ReDimNet2Embedder(nn.Module):
    """ReDimNet2 wrapper: ``[batch, samples]`` 16 kHz audio -> ``[batch, dim]``."""

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        from voxweave.vendor.redimnet2 import ReDimNet2Wrap

        super().__init__()
        self.model = ReDimNet2Wrap(**dict(model_config))

    @property
    def declared_dim(self) -> int:
        return int(self.model.linear.out_features)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.model(wave).reshape(wave.shape[0], -1)


class AnimeVoiceActorEmbedder(nn.Module):
    """GroupNorm ECAPA-TDNN behind the SpeechBrain Fbank (anime voice-actor model).

    Attribute names (``backbone``, ``fbank``) mirror the checkpoint's key prefixes
    so the state dict loads strictly, including the unused delta kernel buffer.
    """

    def __init__(self) -> None:
        from voxweave.vendor.speechbrain_ecapa import ECAPA_TDNN, Fbank
        from voxweave.vendor.speechbrain_ecapa.nnet import BatchNorm1d

        super().__init__()
        self.backbone = ECAPA_TDNN(
            input_size=ANIME_VA_N_MELS,
            lin_neurons=ANIME_VA_EMBEDDING_DIM,
            channels=list(ANIME_VA_CHANNELS),
            kernel_sizes=list(ANIME_VA_KERNEL_SIZES),
        )
        for module in self.backbone.modules():
            if isinstance(module, BatchNorm1d):
                channels = int(module.norm.num_features)
                # Replace the wrapper's inner torch BatchNorm1d; the wrapper keeps
                # its (batch, channel, time) call convention unchanged.
                setattr(module, "norm", nn.GroupNorm(ANIME_VA_NORM_GROUPS, channels))
        self.fbank = Fbank(sample_rate=16_000, n_mels=ANIME_VA_N_MELS)

    @property
    def declared_dim(self) -> int:
        return int(self.backbone.fc.conv.out_channels)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave = wave.to(torch.float32)
        # Only a clipping waveform is peak-normalized; in-range audio keeps its
        # level, exactly as the model was trained.
        peak = wave.abs().max()
        if peak > 1.0:
            wave = wave / peak
        features = self.fbank(wave * ANIME_VA_WAVEFORM_SCALE)
        return self.backbone(features).reshape(wave.shape[0], -1)


def _require_mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointLayoutError(f"{what} is not a mapping")
    return value


def build_redimnet2(checkpoint: object) -> tuple[nn.Module, int]:
    """Build ReDimNet2 from a release asset: ``{"model_config", "state_dict"}``."""
    root = _require_mapping(checkpoint, "ReDimNet2 checkpoint")
    model_config = _require_mapping(
        root.get("model_config"), "ReDimNet2 checkpoint model_config"
    )
    state_dict = _require_mapping(
        root.get("state_dict"), "ReDimNet2 checkpoint state_dict"
    )
    network = ReDimNet2Embedder(model_config)
    network.model.load_state_dict(state_dict, strict=True)
    return network, network.declared_dim


def build_anime_va(checkpoint: object) -> tuple[nn.Module, int]:
    """Build the anime voice-actor ECAPA-TDNN from its flat state dict."""
    state_dict = _require_mapping(checkpoint, "anime-va checkpoint")
    network = AnimeVoiceActorEmbedder()
    network.load_state_dict(state_dict, strict=True)
    return network, network.declared_dim


__all__ = [
    "AnimeVoiceActorEmbedder",
    "CheckpointLayoutError",
    "ReDimNet2Embedder",
    "build_anime_va",
    "build_redimnet2",
]
