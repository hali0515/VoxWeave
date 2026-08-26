"""Language-provider identity and degradation ledger for segmentation.

Segmentation leans on five optional third-party providers -- pysbd (sentences),
BudouX and jieba ``cut`` (atoms), jieba ``posseg`` and fugashi/UniDic (POS) --
plus two "no provider" fallbacks (per-char CJK, ``str.split``). Every one of them
is lazy-imported and fails open with no diagnostic, so today a missing jieba
silently swaps zh from word segmentation to BudouX-zh to per-char, changing which
breaks are legal at all, and nothing records it.

This module adds the record and changes no behaviour:

* :func:`provider_snapshot` -- the static identity (provider + version, plus the
  ja dictionary flavor and the declared jieba decode mode) of the providers
  *this* language uses. It never loads one the language does not use: an English
  document must not pay for fugashi, and a Japanese one must not pay for jieba.
* :func:`note_degraded` / :func:`degradation_capture` -- which fallbacks actually
  fired during one segmentation, aggregated by ``(slot, reason)``, plus a
  once-per-process ``log.warning`` on the ``voxweave`` logger. Never
  ``warnings.warn``: ``ui.install_logging``'s filters swallow those.

Instrumentation built on this is observation only -- an instrumented call must
return exactly what it returned before. Nothing heavy is imported at module
import time; the CJK providers are reached through the same cached loaders
production uses, from inside the functions, which also keeps the instrumented
modules free to import this one at their top level.

SCOPE OF THE LEDGER: the capture lives in a :class:`~contextvars.ContextVar`, and
CPython starts a new :class:`threading.Thread` with a fresh empty context, so a
capture established on one thread is invisible to worker threads it spawns --
a degradation raised off-thread is dropped from the manifest even though its
once-per-process warning still reaches the log. Nothing in the segmentation path
is threaded today (``pipeline.segment_document`` runs the whole engine inline),
so no event is currently lost; anything that later moves provider work onto a
pool has to carry the context across itself (``contextvars.copy_context()``) or
the manifest will under-report.
"""

from __future__ import annotations

import contextvars
import functools
import importlib.metadata
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .langsets import LANGUAGES_WITHOUT_SPACES

log = logging.getLogger("voxweave")

#: Active capture, or ``None`` when nobody is recording. A ``ContextVar`` rather
#: than a module global so a capture cannot leak across concurrent pipelines.
_LEDGER: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "voxweave_degradation_ledger", default=None
)

#: ``(slot, reason)`` pairs already logged in this process -- the ledger is the
#: complete record, the log line only has to say it once.
_WARNED: set[tuple[str, str]] = set()

#: Set inside a capture that must not consume the once-per-process warning.
#: A measurement lane re-runs the same providers over the same document, so
#: without this it can WIN the latch and swallow the one line the shipping run
#: was entitled to -- the ledger stays correct either way, but an operator would
#: see the degradation attributed to nothing, or not at all.
_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "voxweave_degradation_quiet", default=False
)

#: jieba is tokenized twice over the same shared tokenizer: ``phrase_atoms`` uses
#: ``cut(HMM=True)`` while ``zh_pos_boundary_penalties`` uses ``cut(HMM=False)``.
#: The mode is a call-site constant, not something the library reports, so the
#: POS slot declares it.
_POSSEG_DECODE_MODE = "HMM=False"


def note_degraded(slot: str, reason: str) -> None:
    """Record that ``slot`` fell back to a lesser provider, once per event.

    A no-op beyond one ``ContextVar`` read when no capture is active, so call
    sites can sit on the hot path. Repeats of the same ``(slot, reason)`` inside
    one capture increment ``count`` instead of appending, which keeps per-cue
    events (tokenizer disagreements) from swamping the manifest.
    """
    ledger = _LEDGER.get()
    if ledger is not None:
        for entry in ledger:
            if entry["slot"] == slot and entry["reason"] == reason:
                entry["count"] += 1
                break
        else:
            ledger.append({"slot": slot, "reason": reason, "count": 1})
    key = (slot, reason)
    if key not in _WARNED and not _QUIET.get():
        _WARNED.add(key)
        log.warning("segmentation %s provider degraded: %s", slot, reason)


@contextmanager
def degradation_capture(*, quiet: bool = False) -> Iterator[list[dict[str, Any]]]:
    """Collect degradation events raised inside the block.

    Yields the live list of ``{"slot", "reason", "count"}`` entries and restores
    whatever capture was active before, so nesting is safe.

    ``quiet=True`` records into the ledger but takes no claim on the
    once-per-process warning: a nested measurement capture that re-runs the same
    providers would otherwise emit the single log line the outer, *shipping* run
    should have emitted, leaving production's degradation silent. It suppresses
    only the log, never the ledger entry.
    """
    ledger: list[dict[str, Any]] = []
    token = _LEDGER.set(ledger)
    quiet_token = _QUIET.set(quiet or _QUIET.get())
    try:
        yield ledger
    finally:
        _QUIET.reset(quiet_token)
        _LEDGER.reset(token)


def _version(distribution: str) -> str | None:
    """Installed version of ``distribution``, ``None`` when it is absent.

    Read from distribution metadata rather than by importing (mirrors
    ``scripts/calib_segmentation.dependency_versions``).
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=32)
def _pysbd_supports(iso: str) -> bool:
    """Whether ``_segment_sentences`` gets a pysbd segmenter for ``iso``.

    Mirrors ``smart_split._segment_sentences``: pysbd absent *and* pysbd having
    no model for the language (``Segmenter(language="yue")`` raises) both land on
    the regex fallback. Cached per language so a snapshot never re-pays the
    construction, and never re-raises.
    """
    try:
        import pysbd  # type: ignore
    except ImportError:
        return False
    try:
        pysbd.Segmenter(language=iso, clean=False)
    except Exception:  # noqa: BLE001 - any construction failure means regex
        return False
    return True


def _sentences_slot(iso: str) -> dict[str, Any]:
    if _pysbd_supports(iso):
        return {"provider": "pysbd", "version": _version("pysbd")}
    return {"provider": "regex", "version": None}


def _atoms_slot(iso: str) -> dict[str, Any]:
    """Mirrors the branch order of ``breakpoints.phrase_atoms``.

    Spaced languages split on whitespace -- the designed provider, not a
    degradation. zh prefers jieba (BudouX's zh model is too weak), then BudouX,
    then per-char; the other no-space languages have only BudouX, and yue has
    neither, so it always lands on per-char.
    """
    if iso not in LANGUAGES_WITHOUT_SPACES:
        return {"provider": "whitespace", "version": None}
    from voxweave.core import breakpoints

    if iso == "zh" and breakpoints._load_jieba() is not None:
        return {"provider": "jieba", "version": _version("jieba")}
    if breakpoints._load_parser(iso) is not None:
        return {"provider": "budoux", "version": _version("budoux")}
    return {"provider": "per_char", "version": None}


def _unidic_flavor(tagger: Any) -> str | None:
    """Which UniDic fugashi resolved: ``"<dist> <version> (<feature class>)"``.

    The dictionary, not fugashi, decides the tagset that ``_pos_penalty`` scores,
    and the installed flavor is only distinguishable at runtime -- the feature
    class name is ``UnidicFeatures26`` for unidic-lite.
    """
    try:
        feature_cls = type(next(iter(tagger("の"))).feature).__name__
    except Exception:  # noqa: BLE001 - identity reporting must never break a run
        return None
    for distribution in ("unidic-lite", "unidic"):
        version = _version(distribution)
        if version is not None:
            return f"{distribution} {version} ({feature_cls})"
    return f"unknown ({feature_cls})"


@functools.lru_cache(maxsize=1)
def _posseg_available() -> bool:
    """Whether ``zh_pos_boundary_penalties`` gets a POS tagger.

    Probes the exact callable the scorer uses -- ``quiet_import_jieba(posseg=True)``
    (kinsoku.py) -- not ``breakpoints._load_jieba()``, which is
    ``quiet_import_jieba()`` with ``posseg=False``. They are two different imports
    and a half-installed jieba can answer them differently, in which case
    reporting the atoms-side loader would claim a POS provider that demonstrably
    did not run while ``degraded`` says ``posseg-import-failed``. Cached because
    the snapshot is per document and the import is not free; only zh/yue
    documents ever reach it, so ja/en still load no jieba at all.
    """
    from voxweave.core import breakpoints

    return breakpoints.quiet_import_jieba(posseg=True) is not None


def _pos_slot(iso: str) -> dict[str, Any]:
    """ja is scored by fugashi/UniDic, zh and yue by Mandarin jieba posseg.

    ``VOXWEAVE_JA_POS=0`` and an absent fugashi are both latched inside
    ``kinsoku._load_ja_tagger``'s cache, so the state is read from that cached
    loader (populating it exactly as production would) instead of from a fresh
    env read that could disagree with what the run actually used.
    """
    if iso == "ja":
        from voxweave.core import kinsoku

        tagger = kinsoku._load_ja_tagger()
        if tagger is None:
            return {
                "provider": None,
                "version": None,
                "dict": None,
                "ja_pos_enabled": False,
            }
        return {
            "provider": "fugashi-unidic",
            "version": _version("fugashi"),
            "dict": _unidic_flavor(tagger),
            "ja_pos_enabled": True,
        }
    if iso in ("zh", "yue"):
        available = _posseg_available()
        return {
            "provider": "jieba-posseg" if available else None,
            "version": _version("jieba") if available else None,
            "decode_mode": _POSSEG_DECODE_MODE,
        }
    return {"provider": None, "version": None}


def provider_snapshot(iso: str) -> dict[str, Any]:
    """Static provider identity for a document in ``iso``.

    Three slots -- ``sentences``, ``atoms``, ``pos`` -- each carrying at least
    ``provider`` and ``version``. JSON-scalar only, deterministic for a fixed
    environment, and it touches no provider the language does not use.
    """
    return {
        "sentences": _sentences_slot(iso),
        "atoms": _atoms_slot(iso),
        "pos": _pos_slot(iso),
    }
