import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Opt-in gates of the real-model tests: they choose what runs, not how it runs,
# so they are the only VOXWEAVE_* variables a test run inherits.
_REAL_MODEL_GATES = ("VOXWEAVE_RUN_DIARIZE_C1_GPU_E2E", "VOXWEAVE_REAL_WEIGHT_TESTS")
_OPT_IN_TEST_ENV = frozenset(
    {
        *_REAL_MODEL_GATES,
        "VOXWEAVE_DIARIZE_C1_E2E_WAV",
        "VOXWEAVE_DIARIZE_C1_E2E_WAV_SHA256",
    }
)
# Where a developer keeps real weights (and the token for gated ones). Kept only
# when a real-model gate is set, so those tests find the weights already on disk
# instead of skipping or downloading them again under ~/.cache/voxweave.
_REAL_MODEL_WEIGHT_ENV = frozenset(
    {"VOXWEAVE_CACHE_ROOT", "VOXWEAVE_MODEL_DIR", "VOXWEAVE_HF_TOKEN"}
)


def _isolate_process_environment() -> None:
    """Drop inherited user knobs before any test module imports voxweave.

    Documented VOXWEAVE_* variables change defaults that tests assert, and some
    are read once at import time, so a developer's shell settings would leak
    into (and break) the suite. The LLM endpoint variables are dropped for the
    same reason. Unless a real-model gate is set, Hugging Face access is cut
    too: no inherited token, and offline mode, so a fake that misses a seam
    fails fast instead of reaching huggingface.co (or waiting out its retries
    when offline).
    """
    gated = any(os.environ.get(gate) == "1" for gate in _REAL_MODEL_GATES)
    keep = _OPT_IN_TEST_ENV | _REAL_MODEL_WEIGHT_ENV if gated else _OPT_IN_TEST_ENV
    for name in tuple(os.environ):
        if name.startswith("VOXWEAVE_") and name not in keep:
            del os.environ[name]
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        os.environ.pop(name, None)
    if gated:
        return
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"


_isolate_process_environment()


@pytest.fixture(autouse=True)
def _isolate_voxweave_cache(tmp_path: Path) -> Iterator[None]:
    """Point VOXWEAVE_CACHE_ROOT (media snapshots) into each test's temporary root.

    config.CACHE_ROOT / ASR_CACHE / ALIGN_CACHE / AUDIO_CACHE are import-time
    constants and are not affected; episode artifacts live beside the media."""
    name = "VOXWEAVE_CACHE_ROOT"
    previous = os.environ.get(name)
    os.environ[name] = str(tmp_path / ".voxweave-cache")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.fixture(autouse=True)
def _isolate_voiceprint_model_env() -> Iterator[None]:
    """Never inherit a developer's voiceprint embedder selection or checkpoints.

    Plain os.environ bookkeeping on purpose: requesting ``monkeypatch`` from an
    autouse conftest fixture would instantiate it before every module-level
    autouse fixture and change their teardown order.
    """
    names = (
        "VOXWEAVE_VOICEPRINT_MODEL",
        "VOXWEAVE_REDIMNET2_CKPT",
        "VOXWEAVE_ANIME_VA_CKPT",
    )
    previous = {name: os.environ.pop(name, None) for name in names}
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def _pin_pyannote_speaker_clustering() -> Iterator[None]:
    """Run diarization tests on pyannote's own clustering unless they opt out.

    Diarization tests drive fake pyannote pipelines over synthetic audio; the
    voiceprint clustering stage would load (or download) the real ReDimNet2
    checkpoint. Pinning the env layer keeps the suite independent of the
    built-in default and of a developer's environment; tests of the voiceprint
    stage or of knob precedence set or delete ``VOXWEAVE_DIARIZE_CLUSTERING``
    themselves. Plain os.environ bookkeeping for the fixture-ordering reason
    above.
    """
    name = "VOXWEAVE_DIARIZE_CLUSTERING"
    previous = os.environ.get(name)
    os.environ[name] = "pyannote"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.fixture(autouse=True)
def _offline_voiceprint_prefetch(request: pytest.FixtureRequest) -> Iterator[None]:
    """Keep process()/transcribe() from fetching or hashing real checkpoints.

    A voiceprint run prefetches its embedder checkpoints before any audio work,
    which would hit the network (or hash the developer's cache) in every test
    that enables ``voiceprints``. The prefetch becomes a no-op here; tests that
    exercise it opt back in with ``@pytest.mark.real_voiceprint_prefetch`` and
    fake the network themselves. Plain attribute bookkeeping for the same
    fixture-ordering reason as above.
    """
    if request.node.get_closest_marker("real_voiceprint_prefetch") is not None:
        yield
        return
    from voxweave import voiceembed

    original = voiceembed.prefetch_checkpoints
    voiceembed.prefetch_checkpoints = lambda *_args, **_kwargs: ()  # type: ignore[assignment]
    try:
        yield
    finally:
        voiceembed.prefetch_checkpoints = original


@pytest.fixture(autouse=True)
def _isolate_voice_library(tmp_path: Path) -> Iterator[None]:
    """Never read or write the developer's real voice library.

    The built-in library location follows ``XDG_DATA_HOME``, so pointing it
    at the test's temporary root keeps the default layer exercised while an
    inherited ``VOXWEAVE_VOICES_DIR`` (or tier-2 threshold) cannot leak in.
    Plain os.environ bookkeeping for the fixture-ordering reason above.
    """
    names = ("VOXWEAVE_VOICES_DIR", "VOXWEAVE_VOICES_GLOBAL_SUGGEST", "XDG_DATA_HOME")
    previous = {name: os.environ.pop(name, None) for name in names}
    os.environ["XDG_DATA_HOME"] = str(tmp_path / ".xdg-data")
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def _isolate_voxweave_config(tmp_path: Path) -> Iterator[None]:
    """Never let a test read the developer's real ~/.config/voxweave.conf.

    Points VOXWEAVE_CONFIG at a file that does not exist, so config._load() sees
    an empty config unless a test writes its own (the conf_at fixtures do).
    """
    name = "VOXWEAVE_CONFIG"
    previous = os.environ.get(name)
    os.environ[name] = str(tmp_path / "voxweave.conf")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
