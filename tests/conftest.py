import functools
import json
import os
import platform
import subprocess
import sys
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
        "VOXWEAVE_STRICT_ORACLE",
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
    if os.environ.get("TERM", "").lower() in {"dumb", "unknown"}:
        os.environ.pop("TERM", None)
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


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ORACLE_MANIFEST = _REPO_ROOT / "calibration" / "p6-oracle" / "manifest.json"
_ALIGN_SHADOW_MANIFEST = _REPO_ROOT / "calibration" / "align-shadow" / "manifest.json"
_SEG_BASELINE = _REPO_ROOT / "calibration" / "segmentation" / "baseline.json"

# Tests that execute the detached P6 oracle against its recorded host pin
# (exact interpreter micro, Linux x86_64, and full git history). Schema,
# layout, digest, and alignment-unit tests in the same modules still run on
# every platform and Python version.
_HOST_PINNED_ORACLE_TESTS: dict[str, frozenset[str]] = {
    "tests/test_p6_oracle_public_commands.py": frozenset(
        {
            "test_every_oracle_case_executes_its_recorded_public_command_in_a_clean_root",
            "test_public_command_artifact_matches_the_standalone_projector_authority",
            "test_public_command_environment_rejects_ambient_feature_flags",
            "test_align_public_cases_expose_production_owned_runtime_phase_traces",
            "test_mms_scenario_validator_rejects_incomplete_or_reordered_live_trace",
            "test_uncontained_ao15_validator_rejects_all_late_work",
            "test_live_ao_order_mutation_forces_gate_exit_one_without_changing_declarations",
        }
    ),
    "tests/test_p6_oracle_release_version.py": frozenset(
        {
            "test_versioned_expected_bytes_are_both_emitted_by_the_standalone_projector",
            "test_simulated_release_refresh_passes_all_three_oracle_gates",
        }
    ),
    "tests/test_p6_fix_oracle_rat4.py": frozenset(
        {
            "test_p6_oracle_runner_has_exact_exit_contract_and_read_only_compare",
            "test_p6_oracle_compare_writes_a_deterministic_closed_json_report",
            "test_p6_source_registry_and_dependency_gate_is_checked",
            "test_p6_oracle_pytest_lane_is_hermetic_under_hostile_locale",
            "test_canonical_p6_oracle_gate_target_pins_hostile_locale",
        }
    ),
}

# Tests that execute calibration/align-shadow/manifest.json (pins Python 3.13
# and linux-x86_64). Layout, schema, serializer, and ownership tests in the
# same file remain active across all platforms and Python versions.
_ALIGN_SHADOW_PINNED_TESTS: dict[str, frozenset[str]] = {
    "tests/test_p6_align_shadow_corpus.py": frozenset(
        {
            "test_align_shadow_runner_exact_zero_one_two_and_check_only_contract",
            "test_align_shadow_report_has_exact_real_pipeline_matrix",
            "test_align_shadow_report_pins_fresh_interpreter_import_truth",
            "test_valid_rich_artifacts_retain_complete_atomic_semantic_facts",
            "test_typed_invalid_observation_is_a_successful_corpus_outcome",
            "test_post_finalize_partition_cutoff_retains_only_the_semantic_group",
            "test_each_internal_rich_construction_failure_yields_one_minimal_callback",
            "test_minimal_unavailable_waives_callback_and_makes_harness_exit_two",
            "test_align_shadow_report_is_relocation_stable",
        }
    ),
}

# CLI shadow replay tests that compare against calibration/segmentation/baseline.json
# without --allow-environment-drift (pins Python minor 3.13).
_SEG_BASELINE_PINNED_TESTS: dict[str, frozenset[str]] = {
    "tests/test_p5_close.py": frozenset(
        {
            "test_shadow_cli_exits_two_for_post_construction_value_mutations",
            "test_n14_cli_marks_live_work_disagreement_invalid_with_organic_fd9",
            "test_n14_cli_rejects_live_all_zero_work_complete_matrix",
        }
    ),
}


@functools.lru_cache(maxsize=8)
def _load_oracle_pin(
    manifest_path: Path,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    data = json.loads(manifest_path.read_bytes())
    execution = data["execution"]
    commits = tuple(
        sorted((str(k), str(v)) for k, v in data["reference_commits"].items())
    )
    return str(execution["interpreter"]), str(execution["platform"]), commits


@functools.lru_cache(maxsize=8)
def _load_align_shadow_pin(manifest_path: Path) -> tuple[str, str]:
    data = json.loads(manifest_path.read_bytes())
    env = data["environment"]
    return str(env["python_major_minor"]), str(env["platform"])


@functools.lru_cache(maxsize=8)
def _load_seg_baseline_python_minor(baseline_path: Path) -> str:
    data = json.loads(baseline_path.read_bytes())
    raw_py = str((data.get("environment") or {}).get("python") or "")
    parts = raw_py.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else raw_py


@functools.lru_cache(maxsize=16)
def _git_commit_reachable(repo_root: Path, commit: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return proc.returncode == 0


def oracle_host_mismatch_reason(
    *,
    manifest_path: Path = _ORACLE_MANIFEST,
    repo_root: Path = _REPO_ROOT,
) -> str | None:
    """Return why the current host cannot run the recorded P6 oracle, or None."""
    if not manifest_path.is_file():
        return None
    recorded_interpreter, recorded_platform, commits = _load_oracle_pin(manifest_path)
    observed_interpreter = (
        f"{platform.python_implementation()} "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if observed_interpreter != recorded_interpreter:
        return (
            f"interpreter is {observed_interpreter!r}, "
            f"oracle pins {recorded_interpreter!r}"
        )
    observed_platform = f"{platform.system()} {platform.machine()}"
    if observed_platform != recorded_platform:
        return f"platform is {observed_platform!r}, oracle pins {recorded_platform!r}"
    for name, commit in commits:
        if not _git_commit_reachable(repo_root, commit):
            return (
                f"reference commit {name} ({commit[:7]}) is unreachable "
                "(shallow git clone)"
            )
    return None


def align_shadow_host_mismatch_reason(
    *,
    manifest_path: Path = _ALIGN_SHADOW_MANIFEST,
) -> str | None:
    """Return why the current host cannot run align-shadow corpus tests, or None."""
    if not manifest_path.is_file():
        return None
    recorded_py, recorded_platform = _load_align_shadow_pin(manifest_path)
    observed_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if observed_py != recorded_py:
        return f"Python is {observed_py!r}, align-shadow manifest pins {recorded_py!r}"
    observed_platform = f"{platform.system().lower()}-{platform.machine().lower()}"
    if observed_platform != recorded_platform:
        return (
            f"platform is {observed_platform!r}, "
            f"align-shadow manifest pins {recorded_platform!r}"
        )
    return None


def seg_baseline_host_mismatch_reason(
    *,
    baseline_path: Path = _SEG_BASELINE,
) -> str | None:
    """Return why the current host drifts from segmentation baseline Python minor."""
    if not baseline_path.is_file():
        return None
    recorded_py = _load_seg_baseline_python_minor(baseline_path)
    observed_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if recorded_py and observed_py != recorded_py:
        return f"Python is {observed_py!r}, segmentation baseline pins {recorded_py!r}"
    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    if os.environ.get("VOXWEAVE_STRICT_ORACLE") == "1":
        return
    try:
        rel = item.path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return
    base_name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]

    oracle_names = _HOST_PINNED_ORACLE_TESTS.get(rel)
    if oracle_names is not None and base_name in oracle_names:
        reason = oracle_host_mismatch_reason()
        if reason is not None:
            pytest.skip(
                f"P6 oracle host pin mismatch ({reason}); "
                "set VOXWEAVE_STRICT_ORACLE=1 to enforce"
            )

    align_names = _ALIGN_SHADOW_PINNED_TESTS.get(rel)
    if align_names is not None and base_name in align_names:
        reason = align_shadow_host_mismatch_reason()
        if reason is not None:
            pytest.skip(
                f"align-shadow calibration host pin mismatch ({reason}); "
                "set VOXWEAVE_STRICT_ORACLE=1 to enforce"
            )

    seg_names = _SEG_BASELINE_PINNED_TESTS.get(rel)
    if seg_names is not None and base_name in seg_names:
        reason = seg_baseline_host_mismatch_reason()
        if reason is not None:
            pytest.skip(
                f"segmentation baseline host pin mismatch ({reason}); "
                "set VOXWEAVE_STRICT_ORACLE=1 to enforce"
            )
