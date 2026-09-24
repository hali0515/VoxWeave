"""Vendored SpeechBrain ECAPA-TDNN + Fbank (frozen third-party code, Apache-2.0).

Source: https://github.com/speechbrain/speechbrain tag v1.0.3 (commit
31c1e329048c0380dc7f2acbe680c44a036b6286). The Apache License 2.0 text is in
this directory; each module header names the upstream files it was copied from
and the exact modifications.

Why it is vendored instead of depending on ``speechbrain``: the Japanese
voice-actor embedder (``litagin/anime_speaker_embedding_by_va_ecapa_tdnn_groupnorm``)
only needs the ECAPA-TDNN network and the Fbank front end, while the full
package pulls a large dependency tree into the core install. Only these
definitions are kept, verbatim; the GroupNorm swap the checkpoint needs is done
by ``voxweave.voiceembed``, not here.

The directory is excluded from ruff and pyright (see pyproject.toml).
"""

from .ecapa_tdnn import ECAPA_TDNN
from .features import Fbank

__all__ = ["ECAPA_TDNN", "Fbank"]
