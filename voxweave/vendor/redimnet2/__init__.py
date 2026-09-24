"""Vendored ReDimNet2 speaker-embedding model code (frozen third-party code).

Source: https://github.com/PalabraAI/redimnet2 at commit
c5bbe0b76e37df698c403f8844e41304ceab6307 (MIT; the upstream LICENSE text is in
this directory). ``layers/`` keeps its per-file notices: most files are MIT
(ID R&D, Inc.), ``layers/poolings.py`` is Apache-2.0 (wespeaker), with the
Apache License 2.0 text shipped here as ``LICENSE.Apache-2.0``.

Only the model definition is vendored. The upstream ``load_custom`` downloader is
not: ``voxweave.voiceembed`` fetches the release asset itself, verifies its
SHA-256, and loads the state dict with ``torch.load(weights_only=True)``.

Local modification: every absolute ``redimnet2.layers`` import is package-relative
so the code lives under ``voxweave.vendor``. Nothing else differs from upstream.

The directory is excluded from ruff and pyright (see pyproject.toml).
"""

from .redimnet2 import ReDimNet2Wrap

__all__ = ["ReDimNet2Wrap"]
