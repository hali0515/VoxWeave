"""Optimistic per-episode commits and private generation observations.

This module owns filesystem generations, staging, publication order, and the
post-primary SDH compare-and-swap. It deliberately has no model or renderer
dependency: callers hand it final bytes produced and independently checked
outside the episode lock.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voxweave.align_context import (
    IssuedAlignContext,
    IssuedSegmentationContext,
    consume_context_role,
    verify_context_binding,
)
from voxweave.align_failures import CanonicalFailure, SecondaryFailure
from voxweave.speakers import load_speaker_mapping_bytes
from voxweave.voicebase import media_fingerprint
from voxweave.voiceepisode import episode_lock


ProcessSourceMode = Literal["transcribed-media", "injected-words"]
TransactionCommand = Literal["process", "split", "align"]
MappingStat = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class FileGeneration:
    present: bool
    bytes_value: bytes | None

    def __post_init__(self) -> None:
        if type(self.present) is not bool or self.present != (
            self.bytes_value is not None
        ):
            raise ValueError("file generation presence and bytes disagree")

    @property
    def sha256(self) -> str | None:
        if self.bytes_value is None:
            return None
        return hashlib.sha256(self.bytes_value).hexdigest()


@dataclass(frozen=True)
class MappingObservation:
    read_operation: Literal["read-bytes"]
    read_exception_class: type[BaseException]
    read_errno: int | None
    lstat_value: MappingStat | None
    lstat_exception_class: type[BaseException] | None
    lstat_errno: int | None

    def __post_init__(self) -> None:
        if (self.lstat_value is None) == (self.lstat_exception_class is None):
            raise ValueError("mapping observation needs exactly one lstat outcome")
        if self.lstat_value is not None and self.lstat_errno is not None:
            raise ValueError("successful mapping lstat cannot carry errno")


@dataclass(frozen=True)
class SpeakerMappingGeneration:
    kind: Literal["absent", "readable-bytes", "tolerated-unreadable"]
    bytes_value: bytes | None
    loader_status: Literal["not-present", "valid", "tolerated-invalid", "unreadable"]
    private_observation: MappingObservation | None
    names: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.kind == "absent":
            valid = (
                self.bytes_value is None
                and self.loader_status == "not-present"
                and self.private_observation is None
                and not self.names
            )
        elif self.kind == "readable-bytes":
            valid = (
                self.bytes_value is not None
                and self.loader_status in ("valid", "tolerated-invalid")
                and self.private_observation is None
            )
        else:
            valid = (
                self.bytes_value is None
                and self.loader_status == "unreadable"
                and self.private_observation is not None
                and not self.names
            )
        if not valid or len(dict(self.names)) != len(self.names):
            raise ValueError("speaker mapping generation is incongruent")


@dataclass(frozen=True)
class ArtifactCleanup:
    path: Path
    detail_code: Literal[
        "voiceprints-unlink", "suggest-unlink", "html-unlink", "evidence-unlink"
    ]


@dataclass(frozen=True)
class MachineArtifactPublication:
    path: Path
    bytes_value: bytes


@dataclass(frozen=True)
class EvidencePublication:
    path: Path
    bytes_value: bytes


@dataclass(frozen=True)
class TransactionReceipt:
    landed: tuple[Path, ...]
    auxiliary_landed: tuple[Path, ...] = ()
    leftovers: tuple[Path, ...] = ()
    machine_landed: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _OwnedStage:
    target: Path
    path: Path


@dataclass(frozen=True)
class _SpeakerMappingBinding:
    context: IssuedSegmentationContext
    path: Path
    generation: SpeakerMappingGeneration


_SPEAKER_MAPPING_BINDINGS: dict[int, _SpeakerMappingBinding] = {}
_SPEAKER_MAPPING_BINDINGS_LOCK = threading.RLock()


class InputStaleError(RuntimeError):
    """A declared optimistic generation changed before commit."""

    def __init__(self, detail_code: str, message: str):
        super().__init__(message)
        self.failure = CanonicalFailure("input-stale", "recheck", detail_code)
        self.landed: tuple[Path, ...] = ()
        self.leftovers: tuple[Path, ...] = ()


class MediaStaleError(RuntimeError):
    """Selected media changed between compute and commit."""

    def __init__(self, detail_code: str, message: str):
        super().__init__(message)
        self.failure = CanonicalFailure("media-stale", "recheck", detail_code)
        self.landed: tuple[Path, ...] = ()
        self.leftovers: tuple[Path, ...] = ()


class ArtifactCleanupError(RuntimeError):
    """A required post-primary unlink failed after ordered publication."""

    def __init__(self, cleanup: ArtifactCleanup, cause: OSError):
        super().__init__(
            "primary JSON/VTT outputs landed but required artifact cleanup failed: "
            f"could not delete {cleanup.path}: {cause}"
        )
        self.failure = CanonicalFailure(
            "artifact-cleanup-failed", "cleanup", cleanup.detail_code
        )
        self.landed: tuple[Path, ...] = ()
        self.leftovers: tuple[Path, ...] = ()


class TransactionOperationError(RuntimeError):
    """A staged transaction operation failed at one closed P6 terminal."""

    def __init__(
        self,
        kind: Literal["stage-failed", "episode-lock-failed", "commit-failed"],
        phase: Literal["stage", "episode-lock", "commit"],
        detail_code: str,
        cause: BaseException,
    ):
        super().__init__(str(cause))
        self.failure = CanonicalFailure(kind, phase, detail_code)
        self.landed: tuple[Path, ...] = ()
        self.leftovers: tuple[Path, ...] = ()


class StageResidueError(RuntimeError):
    """Owned transaction stages remained after best-effort disposal."""

    def __init__(self, leftovers: Sequence[Path]):
        super().__init__("one or more owned transaction stages could not be removed")
        self.failure = CanonicalFailure(
            "snapshot-dispose-failed", "dispose", "stage-residue"
        )
        self.landed: tuple[Path, ...] = ()
        self.leftovers = tuple(leftovers)


def capture_file_generation(path: Path) -> FileGeneration:
    """Capture exact presence and bytes without normalizing their contents."""
    target = Path(path)
    try:
        value = target.read_bytes()
    except FileNotFoundError as exc:
        try:
            target.lstat()
        except FileNotFoundError:
            return FileGeneration(False, None)
        except OSError:
            raise exc
        raise
    return FileGeneration(True, value)


def same_file_generation(path: Path, expected: FileGeneration) -> bool:
    try:
        return capture_file_generation(path) == expected
    except OSError:
        return False


def _mapping_stat(
    path: Path,
) -> tuple[MappingStat | None, type[BaseException] | None, int | None]:
    try:
        observed = path.lstat()
    except BaseException as exc:
        return None, type(exc), getattr(exc, "errno", None)
    return (
        (
            observed.st_dev,
            observed.st_ino,
            stat.S_IFMT(observed.st_mode),
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        ),
        None,
        None,
    )


def _mapping_warning(path: Path, exc: BaseException) -> str:
    return f"{path.name}: ignoring unreadable speaker mapping: {exc}"


def capture_speaker_mapping(
    path: Path,
    *,
    known_ids: Sequence[str] | set[str],
    warn: Callable[[str], None],
) -> SpeakerMappingGeneration:
    """Capture P11's one tolerant mapping observation and ordered name view."""
    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        stat_value, stat_class, stat_errno = _mapping_stat(target)
        if isinstance(exc, FileNotFoundError) and stat_class is FileNotFoundError:
            return SpeakerMappingGeneration("absent", None, "not-present", None, ())
        warn(_mapping_warning(target, exc))
        return SpeakerMappingGeneration(
            "tolerated-unreadable",
            None,
            "unreadable",
            MappingObservation(
                "read-bytes",
                type(exc),
                getattr(exc, "errno", None),
                stat_value,
                stat_class,
                stat_errno,
            ),
            (),
        )
    try:
        names = load_speaker_mapping_bytes(raw, known_ids, source=target.name)
    except (RuntimeError, UnicodeError) as exc:
        warn(_mapping_warning(target, exc))
        return SpeakerMappingGeneration(
            "readable-bytes", raw, "tolerated-invalid", None, ()
        )
    return SpeakerMappingGeneration(
        "readable-bytes", raw, "valid", None, tuple(names.items())
    )


def observe_speaker_mapping_generation(path: Path) -> SpeakerMappingGeneration:
    """Capture S1 without parsing names or emitting the tolerant S0 warning."""
    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        stat_value, stat_class, stat_errno = _mapping_stat(target)
        if isinstance(exc, FileNotFoundError) and stat_class is FileNotFoundError:
            return SpeakerMappingGeneration("absent", None, "not-present", None, ())
        return SpeakerMappingGeneration(
            "tolerated-unreadable",
            None,
            "unreadable",
            MappingObservation(
                "read-bytes",
                type(exc),
                getattr(exc, "errno", None),
                stat_value,
                stat_class,
                stat_errno,
            ),
            (),
        )
    return SpeakerMappingGeneration("readable-bytes", raw, "valid", None, ())


def bind_split_speaker_mapping_generation(
    context: IssuedSegmentationContext,
    path: Path,
    generation: SpeakerMappingGeneration,
) -> None:
    """Bind split's private S0 observation to its issued commit authority."""
    if not isinstance(context, IssuedSegmentationContext):
        raise TypeError("speaker mapping generation requires segmentation context")
    if not isinstance(generation, SpeakerMappingGeneration):
        raise TypeError("speaker mapping generation is not typed")
    binding = _SpeakerMappingBinding(context, Path(path).resolve(), generation)
    with _SPEAKER_MAPPING_BINDINGS_LOCK:
        existing = _SPEAKER_MAPPING_BINDINGS.get(id(context))
        if existing is not None:
            raise ValueError("speaker mapping generation is already bound")
        _SPEAKER_MAPPING_BINDINGS[id(context)] = binding


def release_split_speaker_mapping_generation(
    context: IssuedSegmentationContext,
) -> None:
    """Retire the private S0 path/observation binding with its selection."""
    with _SPEAKER_MAPPING_BINDINGS_LOCK:
        binding = _SPEAKER_MAPPING_BINDINGS.get(id(context))
        if binding is not None and binding.context is context:
            del _SPEAKER_MAPPING_BINDINGS[id(context)]


def _require_bound_speaker_mapping(
    context: IssuedSegmentationContext | None,
    path: Path,
    generation: SpeakerMappingGeneration,
) -> None:
    if context is None:
        raise ValueError("split mapping CAS requires a segmentation context")
    with _SPEAKER_MAPPING_BINDINGS_LOCK:
        binding = _SPEAKER_MAPPING_BINDINGS.get(id(context))
    if (
        binding is None
        or binding.context is not context
        or binding.path != Path(path).resolve()
        or binding.generation != generation
    ):
        raise ValueError("split speaker mapping generation is not context-bound")


def same_speaker_mapping_generation(
    left: SpeakerMappingGeneration,
    right: SpeakerMappingGeneration,
) -> bool:
    """Compare the closed RAT-7 S0/S1 domain without exception prose or paths."""
    if left.kind != right.kind:
        return False
    if left.kind == "absent":
        return True
    if left.kind == "readable-bytes":
        return left.bytes_value == right.bytes_value
    return left.private_observation == right.private_observation


def _stage_bytes(target: Path, value: bytes) -> _OwnedStage:
    destination = Path(target)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=f".part{destination.suffix}",
    )
    stage = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    return _OwnedStage(destination, stage)


def _replace_stage(stage: _OwnedStage) -> None:
    """Publication seam kept small for ordered failure injection."""
    os.replace(stage.path, stage.target)


def _stage_primary(target: Path, value: bytes, detail_code: str) -> _OwnedStage:
    try:
        return _stage_bytes(target, value)
    except Exception as exc:
        raise TransactionOperationError(
            "stage-failed", "stage", detail_code, exc
        ) from exc


def _replace_primary(stage: _OwnedStage, detail_code: str) -> None:
    try:
        _replace_stage(stage)
    except Exception as exc:
        raise TransactionOperationError(
            "commit-failed", "commit", detail_code, exc
        ) from exc


@contextmanager
def _transaction_lock(path: Path) -> Iterator[None]:
    try:
        manager = episode_lock(Path(path))
        manager.__enter__()
    except Exception as exc:
        raise TransactionOperationError(
            "episode-lock-failed",
            "episode-lock",
            "episode-lock-acquire",
            exc,
        ) from exc
    try:
        yield
    except BaseException as exc:
        suppress = manager.__exit__(type(exc), exc, exc.__traceback__)
        if not suppress:
            raise
    else:
        manager.__exit__(None, None, None)


def _cleanup_after_primary(cleanup: ArtifactCleanup) -> None:
    try:
        cleanup.path.unlink(missing_ok=True)
    except OSError as exc:
        raise ArtifactCleanupError(cleanup, exc) from exc


def _discard_stages(stages: Iterable[_OwnedStage]) -> tuple[Path, ...]:
    owned = tuple(stages)
    for stage in owned:
        try:
            stage.path.unlink(missing_ok=True)
        except OSError:
            pass
    leftovers: list[Path] = []
    for stage in owned:
        try:
            stage.path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            leftovers.append(stage.path)
        else:
            leftovers.append(stage.path)
    return tuple(leftovers)


def _append_stage_residue_secondary(exc: BaseException) -> None:
    secondary = SecondaryFailure("snapshot-dispose-failed", "dispose", "stage-residue")
    failure = getattr(exc, "failure", None)
    if isinstance(failure, CanonicalFailure):
        try:
            setattr(
                exc,
                "failure",
                CanonicalFailure(
                    failure.kind,
                    failure.phase,
                    failure.detail_code,
                    failure.secondary + (secondary,),
                ),
            )
        except Exception:
            pass
    try:
        current = tuple(getattr(exc, "secondary_failures", ()))
        setattr(exc, "secondary_failures", current + (secondary,))
    except Exception:
        pass


def _annotate_failure(
    exc: BaseException,
    *,
    landed: Sequence[Path],
    leftovers: Sequence[Path],
    machine_landed: Sequence[Path] = (),
) -> None:
    for name, value in (
        ("landed", tuple(landed)),
        ("leftovers", tuple(leftovers)),
        ("machine_landed", tuple(machine_landed)),
    ):
        try:
            setattr(exc, name, value)
        except Exception:
            pass


def _stale(command: TransactionCommand, detail: str) -> InputStaleError:
    messages = {
        "process": "output changed during processing; re-run",
        "split": "input changed during replay; re-run",
        "align": "input changed during replay; re-run",
    }
    return InputStaleError(detail, messages[command])


def _check_primary_generations(
    *,
    command: TransactionCommand,
    json_path: Path,
    vtt_path: Path,
    expected_json: FileGeneration,
    expected_vtt: FileGeneration | None,
) -> None:
    if not same_file_generation(json_path, expected_json):
        detail = (
            "process-output-generation"
            if command == "process"
            else "sibling-generation"
        )
        raise _stale(command, detail)
    if command != "split":
        if expected_vtt is None or not same_file_generation(vtt_path, expected_vtt):
            detail = (
                "process-output-generation"
                if command == "process"
                else "vtt-generation"
            )
            raise _stale(command, detail)


def require_primary_generations(
    *,
    command: TransactionCommand,
    json_path: Path,
    vtt_path: Path,
    expected_json: FileGeneration,
    expected_vtt: FileGeneration | None,
) -> None:
    """Run the generation check inside a lock owned by an existing writer."""
    _check_primary_generations(
        command=command,
        json_path=Path(json_path),
        vtt_path=Path(vtt_path),
        expected_json=expected_json,
        expected_vtt=expected_vtt,
    )


def require_media_generation(path: Path, expected_fingerprint: str) -> None:
    try:
        observed = media_fingerprint(Path(path))
    except OSError as exc:
        raise MediaStaleError(
            "media-generation", "selected media changed during processing; re-run"
        ) from exc
    if observed != expected_fingerprint:
        raise MediaStaleError(
            "media-generation", "selected media changed during processing; re-run"
        )


def require_media_pair_decision(
    path: Path,
    *,
    voiceprint_media_fingerprint: str,
    expected_decision: bool,
) -> None:
    """Recheck the snapshot-backed P11 pair decision under the episode lock."""
    try:
        observed = media_fingerprint(Path(path))
    except OSError as exc:
        raise MediaStaleError(
            "pair-decision",
            "selected media pair decision changed during processing; re-run",
        ) from exc
    if (observed == voiceprint_media_fingerprint) is not expected_decision:
        raise MediaStaleError(
            "pair-decision",
            "selected media pair decision changed during processing; re-run",
        )


def _consume_commit_role(
    context: IssuedAlignContext | IssuedSegmentationContext | None,
    *,
    command: TransactionCommand,
    vtt_path: Path,
    json_path: Path,
    media_path: Path | None,
) -> None:
    if context is None:
        return
    if isinstance(context, IssuedAlignContext):
        if command != "align" or media_path is None:
            raise ValueError("align context requires an align media transaction")
        consumer = "align-transaction"
        binding_media = media_path
    elif isinstance(context, IssuedSegmentationContext):
        if command == "align":
            raise ValueError("segmentation context cannot authorize align")
        consumer = "segmentation-transaction"
        binding_media = None
    else:
        raise TypeError("unknown transaction context")
    verify_context_binding(
        context,
        target_path=vtt_path,
        sibling_path=json_path,
        media_path=binding_media,
    )
    consume_context_role(context, "commit", consumer=consumer)


def commit_primary_outputs(
    *,
    command: TransactionCommand,
    episode_path: Path,
    json_path: Path,
    vtt_path: Path,
    expected_json: FileGeneration,
    expected_vtt: FileGeneration | None,
    main_json_bytes: bytes,
    vtt_bytes: bytes,
    cleanup_paths: Sequence[ArtifactCleanup] = (),
    context: IssuedAlignContext | IssuedSegmentationContext | None = None,
    media_path: Path | None = None,
    expected_media_fingerprint: str | None = None,
    expected_voiceprint_media_fingerprint: str | None = None,
    expected_pair_decision: bool | None = None,
    machine_artifact: MachineArtifactPublication | None = None,
    evidence_artifact: EvidencePublication | None = None,
    speaker_mapping_path: Path | None = None,
    expected_speaker_mapping: SpeakerMappingGeneration | None = None,
) -> TransactionReceipt:
    """Stage final bytes, recheck, authorize, and publish in command order."""
    if command not in ("process", "split", "align"):
        raise ValueError("unknown transaction command")
    if evidence_artifact is not None and command != "align":
        raise ValueError("only align may publish durable evidence")
    if (speaker_mapping_path is None) != (expected_speaker_mapping is None):
        raise ValueError("split mapping CAS requires both path and S0 generation")
    if speaker_mapping_path is not None and command != "split":
        raise ValueError("only split may recheck a speaker mapping generation")
    if (expected_voiceprint_media_fingerprint is None) != (
        expected_pair_decision is None
    ):
        raise ValueError("pair decision recheck requires its fingerprint and decision")
    if expected_pair_decision is not None and type(expected_pair_decision) is not bool:
        raise TypeError("expected pair decision must be an exact bool")
    if expected_pair_decision is not None and (
        command != "align" or media_path is None or expected_media_fingerprint is None
    ):
        raise ValueError("pair decision recheck requires an align media generation")
    json_target, vtt_target = Path(json_path), Path(vtt_path)
    stages: list[_OwnedStage] = []
    landed: list[Path] = []
    machine_landed: list[Path] = []
    machine_stage: _OwnedStage | None = None
    evidence_stage: _OwnedStage | None = None
    try:
        stages.append(_stage_primary(json_target, main_json_bytes, "main-json-stage"))
        stages.append(_stage_primary(vtt_target, vtt_bytes, "vtt-stage"))
        if machine_artifact is not None:
            machine_stage = _stage_primary(
                machine_artifact.path,
                machine_artifact.bytes_value,
                "machine-artifact-stage",
            )
            stages.append(machine_stage)
        if evidence_artifact is not None:
            evidence_stage = _stage_primary(
                evidence_artifact.path,
                evidence_artifact.bytes_value,
                "evidence-stage",
            )
            stages.append(evidence_stage)
        with _transaction_lock(Path(episode_path)):
            _check_primary_generations(
                command=command,
                json_path=json_target,
                vtt_path=vtt_target,
                expected_json=expected_json,
                expected_vtt=expected_vtt,
            )
            if expected_media_fingerprint is not None:
                if media_path is None:
                    raise ValueError("media generation requires a media path")
                require_media_generation(media_path, expected_media_fingerprint)
            if expected_pair_decision is not None:
                assert media_path is not None
                assert expected_voiceprint_media_fingerprint is not None
                require_media_pair_decision(
                    media_path,
                    voiceprint_media_fingerprint=(
                        expected_voiceprint_media_fingerprint
                    ),
                    expected_decision=expected_pair_decision,
                )
            if speaker_mapping_path is not None:
                assert expected_speaker_mapping is not None
                segmentation_context = (
                    context if isinstance(context, IssuedSegmentationContext) else None
                )
                _require_bound_speaker_mapping(
                    segmentation_context,
                    speaker_mapping_path,
                    expected_speaker_mapping,
                )
                observed_mapping = observe_speaker_mapping_generation(
                    speaker_mapping_path
                )
                if not same_speaker_mapping_generation(
                    expected_speaker_mapping, observed_mapping
                ):
                    raise _stale(command, "speaker-mapping-generation")
            _consume_commit_role(
                context,
                command=command,
                vtt_path=vtt_target,
                json_path=json_target,
                media_path=media_path,
            )
            for stage, detail_code in zip(
                stages[:2], ("main-json-replace", "vtt-replace"), strict=True
            ):
                _replace_primary(stage, detail_code)
                landed.append(stage.target)
            for cleanup in (
                item for item in cleanup_paths if item.detail_code != "evidence-unlink"
            ):
                _cleanup_after_primary(cleanup)
            if machine_artifact is not None:
                assert machine_stage is not None
                _replace_primary(machine_stage, "machine-artifact-replace")
                machine_landed.append(machine_artifact.path)
            for cleanup in (
                item
                for item in cleanup_paths
                if item.detail_code == "evidence-unlink" and evidence_artifact is None
            ):
                _cleanup_after_primary(cleanup)
            if evidence_artifact is not None:
                assert evidence_stage is not None
                _replace_primary(evidence_stage, "evidence-replace")
                landed.append(evidence_artifact.path)
    except BaseException as exc:
        leftovers = _discard_stages(stages)
        if leftovers:
            _append_stage_residue_secondary(exc)
        _annotate_failure(
            exc,
            landed=landed,
            leftovers=leftovers,
            machine_landed=machine_landed,
        )
        raise
    leftovers = _discard_stages(stages)
    if leftovers:
        error = StageResidueError(leftovers)
        _annotate_failure(
            error,
            landed=landed,
            leftovers=leftovers,
            machine_landed=machine_landed,
        )
        raise error
    return TransactionReceipt(tuple(landed), machine_landed=tuple(machine_landed))


def commit_correction(
    *,
    vtt_path: Path,
    expected_vtt: FileGeneration,
    rendered_vtt_bytes: bytes,
    evidence_path: Path,
) -> TransactionReceipt:
    """Unconditionally rewrite C1 after C0 CAS and retire changed-byte evidence."""
    target = Path(vtt_path)
    stage = _stage_primary(target, rendered_vtt_bytes, "vtt-stage")
    landed: list[Path] = []
    try:
        with _transaction_lock(target):
            if not same_file_generation(target, expected_vtt):
                raise InputStaleError(
                    "correct-generation", "input changed during correction; re-run"
                )
            _replace_primary(stage, "vtt-replace")
            landed.append(target)
            if rendered_vtt_bytes != expected_vtt.bytes_value:
                _cleanup_after_primary(
                    ArtifactCleanup(Path(evidence_path), "evidence-unlink")
                )
    except BaseException as exc:
        leftovers = _discard_stages((stage,))
        if leftovers:
            _append_stage_residue_secondary(exc)
        _annotate_failure(exc, landed=landed, leftovers=leftovers)
        raise
    leftovers = _discard_stages((stage,))
    if leftovers:
        error = StageResidueError(leftovers)
        _annotate_failure(error, landed=landed, leftovers=leftovers)
        raise error
    return TransactionReceipt(tuple(landed))


def commit_auxiliary_sdh(
    *,
    episode_path: Path,
    sidecar_path: Path,
    sidecar_bytes: bytes,
    json_path: Path,
    expected_json: FileGeneration,
    vtt_path: Path,
    expected_vtt: FileGeneration,
    media_path: Path | None = None,
    expected_media_fingerprint: str | None = None,
) -> bool:
    """Replace only SDH when committed primaries and applicable media still match."""
    stage = _stage_bytes(Path(sidecar_path), sidecar_bytes)
    committed = False
    try:
        with _transaction_lock(Path(episode_path)):
            eligible = same_file_generation(
                json_path, expected_json
            ) and same_file_generation(vtt_path, expected_vtt)
            if eligible and expected_media_fingerprint is not None:
                if media_path is None:
                    eligible = False
                else:
                    try:
                        require_media_generation(media_path, expected_media_fingerprint)
                    except MediaStaleError:
                        eligible = False
            if eligible:
                _replace_stage(stage)
                committed = True
    except BaseException as exc:
        leftovers = _discard_stages((stage,))
        if leftovers:
            _append_stage_residue_secondary(exc)
        _annotate_failure(exc, landed=(), leftovers=leftovers)
        raise
    leftovers = _discard_stages((stage,))
    if leftovers:
        raise StageResidueError(leftovers)
    return committed


__all__ = [
    "ArtifactCleanup",
    "ArtifactCleanupError",
    "EvidencePublication",
    "FileGeneration",
    "InputStaleError",
    "MappingObservation",
    "MachineArtifactPublication",
    "MediaStaleError",
    "ProcessSourceMode",
    "SpeakerMappingGeneration",
    "StageResidueError",
    "TransactionReceipt",
    "TransactionOperationError",
    "capture_file_generation",
    "capture_speaker_mapping",
    "bind_split_speaker_mapping_generation",
    "commit_auxiliary_sdh",
    "commit_correction",
    "commit_primary_outputs",
    "release_split_speaker_mapping_generation",
    "require_media_generation",
    "require_primary_generations",
    "same_file_generation",
    "same_speaker_mapping_generation",
    "observe_speaker_mapping_generation",
]
