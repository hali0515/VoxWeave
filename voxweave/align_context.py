"""Context-bound, single-use P6 orchestration roles.

Stable content identity and live invocation authority are deliberately separate.
Stable digests contain only caller-supplied immutable content facts plus the
registry/route choice.  Private bindings additionally contain canonical live
paths and opaque IDs, and are validated through this module's issuance registry
rather than by object shape.
"""

from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from voxweave.align_failures import CanonicalFailure
from voxweave.align_snapshot import (
    FrozenArray,
    FrozenObject,
    FrozenString,
    frozen_json_digest,
    thaw_json,
)
from voxweave.engine_registry import (
    EngineFamily,
    canonical_registry_iso,
    engine_family_for,
)

RouteKind = Literal["ctc-full", "mms-full", "qwen-crop"]
AlignRole = Literal["acquisition", "adapter", "encoder", "evidence-bind", "commit"]
SegmentationRole = Literal["adapter", "encoder", "commit"]
ContextRole = AlignRole | SegmentationRole
RoleTerminal = Literal["consumed", "retired"]

ALIGN_ROLE_ORDER: tuple[AlignRole, ...] = (
    "acquisition",
    "adapter",
    "encoder",
    "evidence-bind",
    "commit",
)
SEGMENTATION_ROLE_ORDER: tuple[SegmentationRole, ...] = (
    "adapter",
    "encoder",
    "commit",
)

_ROLE_CONSUMER = {
    "acquisition": "FreshAlignmentIssuer",
    "adapter": {
        "align": "run_locked_align_adapter",
        "segmentation": "run_locked_segmentation_adapter",
    },
    "encoder": {
        "align": "encode_align_candidates",
        "segmentation": "encode_segmentation_candidates",
    },
    "evidence-bind": "bind_align_evidence",
    "commit": {
        "align": "align-transaction",
        "segmentation": "segmentation-transaction",
    },
}


class ContextAuthorityError(RuntimeError):
    """A context or role failed the closed authority checks."""

    def __init__(self, detail_code: str, message: str):
        super().__init__(message)
        self.detail_code = detail_code
        self.failure = CanonicalFailure(
            "context-authority-invalid", "context", detail_code
        )


@dataclass(frozen=True, init=False)
class IssuedAlignContext:
    context_content_digest: str
    context_binding_digest: str
    engine_family: EngineFamily
    effective_iso: str
    route_kind: RouteKind
    _issuance_nonce: str


@dataclass(frozen=True, init=False)
class IssuedSegmentationContext:
    context_content_digest: str
    context_binding_digest: str
    engine_family: EngineFamily
    effective_iso: str
    _issuance_nonce: str


IssuedContext = IssuedAlignContext | IssuedSegmentationContext


@dataclass(frozen=True)
class ContextRoleEvent:
    role: str
    terminal: RoleTerminal
    consumer: str
    ordinal: int


@dataclass
class _RoleRecord:
    terminal: RoleTerminal | None = None
    consumer: str | None = None


@dataclass
class _ContextRecord:
    context: IssuedContext
    kind: Literal["align", "segmentation"]
    private_context_id: str
    private_evaluation_id: str
    paths: tuple[str, str, str | None]
    stable_fields: FrozenObject
    authority_limit_profile: object | None
    public_seal: tuple[object, ...]
    role_order: tuple[str, ...]
    roles: dict[str, _RoleRecord]
    events: list[ContextRoleEvent]


_ISSUED: dict[int, _ContextRecord] = {}
_LOCK = threading.RLock()


def _public_seal(context: IssuedContext) -> tuple[object, ...]:
    common: tuple[object, ...] = (
        context.context_content_digest,
        context.context_binding_digest,
        context.engine_family,
        context.effective_iso,
    )
    if isinstance(context, IssuedAlignContext):
        return (*common, context.route_kind, context._issuance_nonce)
    return (*common, context._issuance_nonce)


def _canonical(path: Path) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(Path(path))))


def _binding_digest(
    *,
    private_context_id: str,
    private_evaluation_id: str,
    target_path: Path,
    sibling_path: Path,
    media_path: Path | None,
) -> str:
    return frozen_json_digest(
        FrozenArray(
            (
                FrozenString("p6-private-context-binding-v2"),
                FrozenString(private_context_id),
                FrozenString(private_evaluation_id),
                FrozenString(_canonical(target_path)),
                FrozenString(_canonical(sibling_path)),
                FrozenString(_canonical(media_path))
                if media_path is not None
                else FrozenString("<no-media>"),
            )
        )
    )


def _require_stable_fields(value: Any) -> FrozenObject:
    if not isinstance(value, FrozenObject):
        raise ContextAuthorityError(
            "context-binding", "stable context fields must be a FrozenObject"
        )
    return value


def _register(
    *,
    context: IssuedContext,
    kind: Literal["align", "segmentation"],
    target_path: Path,
    sibling_path: Path,
    media_path: Path | None,
    stable_fields: FrozenObject,
    authority_limit_profile: object | None,
    role_order: tuple[str, ...],
    private_context_id: str,
    private_evaluation_id: str,
) -> None:
    record = _ContextRecord(
        context=context,
        kind=kind,
        private_context_id=private_context_id,
        private_evaluation_id=private_evaluation_id,
        paths=(
            _canonical(target_path),
            _canonical(sibling_path),
            _canonical(media_path) if media_path is not None else None,
        ),
        stable_fields=stable_fields,
        authority_limit_profile=authority_limit_profile,
        public_seal=_public_seal(context),
        role_order=role_order,
        roles={role: _RoleRecord() for role in role_order},
        events=[],
    )
    with _LOCK:
        _ISSUED[id(context)] = record


def issue_align_context(
    *,
    stable_fields: FrozenObject,
    target_path: Path,
    sibling_path: Path,
    media_path: Path,
    effective_iso: str,
    route_kind: RouteKind,
) -> IssuedAlignContext:
    """Issue one align context with five independently terminal roles."""
    stable = _require_stable_fields(stable_fields)
    from voxweave.align_distribution import capture_authority_limit_profile

    authority_profile = capture_authority_limit_profile()
    iso = canonical_registry_iso(effective_iso)
    if iso is None:
        raise ContextAuthorityError(
            "context-binding", f"unsupported context language {effective_iso!r}"
        )
    if route_kind not in ("ctc-full", "mms-full", "qwen-crop"):
        raise ContextAuthorityError(
            "context-binding", f"unsupported align route {route_kind!r}"
        )
    family = engine_family_for(iso)
    content_digest = frozen_json_digest(
        FrozenArray(
            (
                FrozenString("align-context-v2"),
                stable,
                FrozenString(authority_profile.kind),
                FrozenString(authority_profile.profile_digest),
                FrozenString(iso),
                FrozenString(route_kind),
                FrozenString(family),
            )
        )
    )
    private_context_id = secrets.token_hex(24)
    private_evaluation_id = secrets.token_hex(24)
    nonce = secrets.token_hex(32)
    binding_digest = _binding_digest(
        private_context_id=private_context_id,
        private_evaluation_id=private_evaluation_id,
        target_path=target_path,
        sibling_path=sibling_path,
        media_path=media_path,
    )
    context = object.__new__(IssuedAlignContext)
    object.__setattr__(context, "context_content_digest", content_digest)
    object.__setattr__(context, "context_binding_digest", binding_digest)
    object.__setattr__(context, "engine_family", family)
    object.__setattr__(context, "effective_iso", iso)
    object.__setattr__(context, "route_kind", route_kind)
    object.__setattr__(context, "_issuance_nonce", nonce)
    _register(
        context=context,
        kind="align",
        target_path=target_path,
        sibling_path=sibling_path,
        media_path=media_path,
        stable_fields=stable,
        authority_limit_profile=authority_profile,
        role_order=ALIGN_ROLE_ORDER,
        private_context_id=private_context_id,
        private_evaluation_id=private_evaluation_id,
    )
    return context


def issue_segmentation_context(
    *,
    stable_fields: FrozenObject,
    target_path: Path,
    sibling_path: Path,
    effective_iso: str,
) -> IssuedSegmentationContext:
    """Issue one process/split context with adapter, encoder, and commit roles."""
    stable = _require_stable_fields(stable_fields)
    iso = canonical_registry_iso(effective_iso)
    if iso is None:
        raise ContextAuthorityError(
            "context-binding", f"unsupported context language {effective_iso!r}"
        )
    family = engine_family_for(iso)
    content_digest = frozen_json_digest(
        FrozenArray(
            (
                FrozenString("segmentation-context-v2"),
                stable,
                FrozenString(iso),
                FrozenString(family),
            )
        )
    )
    private_context_id = secrets.token_hex(24)
    private_evaluation_id = secrets.token_hex(24)
    nonce = secrets.token_hex(32)
    binding_digest = _binding_digest(
        private_context_id=private_context_id,
        private_evaluation_id=private_evaluation_id,
        target_path=target_path,
        sibling_path=sibling_path,
        media_path=None,
    )
    context = object.__new__(IssuedSegmentationContext)
    object.__setattr__(context, "context_content_digest", content_digest)
    object.__setattr__(context, "context_binding_digest", binding_digest)
    object.__setattr__(context, "engine_family", family)
    object.__setattr__(context, "effective_iso", iso)
    object.__setattr__(context, "_issuance_nonce", nonce)
    _register(
        context=context,
        kind="segmentation",
        target_path=target_path,
        sibling_path=sibling_path,
        media_path=None,
        stable_fields=stable,
        authority_limit_profile=None,
        role_order=SEGMENTATION_ROLE_ORDER,
        private_context_id=private_context_id,
        private_evaluation_id=private_evaluation_id,
    )
    return context


def _record_for(context: IssuedContext) -> _ContextRecord:
    with _LOCK:
        record = _ISSUED.get(id(context))
        if record is None or record.context is not context:
            raise ContextAuthorityError(
                "context-unissued", "context was not issued by this process"
            )
        nonce = getattr(context, "_issuance_nonce", None)
        if type(nonce) is not str or not nonce:
            raise ContextAuthorityError("context-unissued", "context seal is absent")
        if _public_seal(context) != record.public_seal:
            raise ContextAuthorityError(
                "context-binding", "issued context public fields changed"
            )
        return record


def _expected_consumer(record: _ContextRecord, role: str) -> str | None:
    expected = _ROLE_CONSUMER.get(role)
    if isinstance(expected, dict):
        return expected.get(record.kind)
    return expected


def consume_context_role(
    context: IssuedContext, role: str, *, consumer: str
) -> ContextRoleEvent:
    """Verify and consume one stage-specific role exactly once."""
    with _LOCK:
        record = _record_for(context)
        role_record = record.roles.get(role)
        expected = _expected_consumer(record, role)
        if role_record is None or expected is None or consumer != expected:
            raise ContextAuthorityError(
                "context-role", f"{consumer!r} cannot consume context role {role!r}"
            )
        if role_record.terminal is not None:
            raise ContextAuthorityError(
                "context-consumed", f"context role {role!r} is already terminal"
            )
        role_record.terminal = "consumed"
        role_record.consumer = consumer
        event = ContextRoleEvent(role, "consumed", consumer, len(record.events))
        record.events.append(event)
        return event


def retire_context_role(
    context: IssuedContext, role: str, *, consumer: str = "outer-orchestration"
) -> ContextRoleEvent:
    """Retire one still-live role; a terminal role cannot be retired again."""
    with _LOCK:
        record = _record_for(context)
        role_record = record.roles.get(role)
        if role_record is None:
            raise ContextAuthorityError(
                "context-role", f"unknown context role {role!r}"
            )
        if role_record.terminal is not None:
            raise ContextAuthorityError(
                "context-consumed", f"context role {role!r} is already terminal"
            )
        role_record.terminal = "retired"
        role_record.consumer = consumer
        event = ContextRoleEvent(role, "retired", consumer, len(record.events))
        record.events.append(event)
        return event


def retire_live_context_roles(
    context: IssuedContext, *, consumer: str = "outer-orchestration"
) -> tuple[ContextRoleEvent, ...]:
    """Retire every still-live role in issuance order, idempotently."""
    retired: list[ContextRoleEvent] = []
    with _LOCK:
        record = _record_for(context)
        for role in record.role_order:
            role_record = record.roles[role]
            if role_record.terminal is not None:
                continue
            role_record.terminal = "retired"
            role_record.consumer = consumer
            event = ContextRoleEvent(role, "retired", consumer, len(record.events))
            record.events.append(event)
            retired.append(event)
    return tuple(retired)


def role_events(context: IssuedContext) -> tuple[ContextRoleEvent, ...]:
    with _LOCK:
        return tuple(_record_for(context).events)


def role_vector(context: IssuedContext) -> tuple[Literal["C", "R", "L"], ...]:
    with _LOCK:
        record = _record_for(context)
        out: list[Literal["C", "R", "L"]] = []
        for role in record.role_order:
            terminal = record.roles[role].terminal
            if terminal == "consumed":
                out.append("C")
            elif terminal == "retired":
                out.append("R")
            else:
                out.append("L")
        return tuple(out)


def verify_context_roles_terminal(context: IssuedContext) -> None:
    """Reject a top-level exit while any issued role remains live."""
    with _LOCK:
        record = _record_for(context)
        if any(role.terminal is None for role in record.roles.values()):
            raise ContextAuthorityError(
                "context-unused-role",
                "issued context has an unused live role at top-level exit",
            )


def verify_context_expected_vtt_generation(
    context: IssuedAlignContext,
    *,
    observed_vtt_sha256: str | None,
) -> None:
    """Bind the private correct/apply handoff to the AO-01 VTT generation."""
    with _LOCK:
        record = _record_for(context)
        if record.kind != "align":
            raise ContextAuthorityError("context-binding", "context is not align")
        stable = thaw_json(record.stable_fields)
        expected = stable.get("expected_vtt_sha256")
        if expected is None:
            return
        if expected != observed_vtt_sha256:
            raise ContextAuthorityError(
                "expected-vtt-generation",
                "expected VTT generation does not match the snapped input",
            )


def verify_context_binding(
    context: IssuedContext,
    *,
    target_path: Path,
    sibling_path: Path,
    media_path: Path | None,
) -> None:
    """Verify that a genuine context still names this live invocation target."""
    with _LOCK:
        record = _record_for(context)
        observed = (
            _canonical(target_path),
            _canonical(sibling_path),
            _canonical(media_path) if media_path is not None else None,
        )
        if observed != record.paths:
            raise ContextAuthorityError(
                "context-binding", "context paths do not match the issued invocation"
            )


def verify_context_content(context: IssuedContext, stable_fields: FrozenObject) -> None:
    """Reject a stable-content splice even if public digests were copied."""
    with _LOCK:
        record = _record_for(context)
        if stable_fields != record.stable_fields:
            raise ContextAuthorityError(
                "context-binding", "stable context fields changed after issuance"
            )


def _align_context_authority_profile(context: IssuedAlignContext) -> object:
    """Return the AO-05 profile bound to a genuine align context."""
    with _LOCK:
        record = _record_for(context)
        if record.kind != "align" or record.authority_limit_profile is None:
            raise ContextAuthorityError(
                "context-binding", "align authority profile is unavailable"
            )
        return record.authority_limit_profile


def _align_context_stable_fields(context: IssuedAlignContext) -> FrozenObject:
    """Return the immutable stable input projection to trusted leaf stages."""
    with _LOCK:
        record = _record_for(context)
        if record.kind != "align":
            raise ContextAuthorityError("context-binding", "context is not align")
        return record.stable_fields


def _private_context_subject(context: IssuedContext) -> tuple[str, str]:
    """Internal audit subject; never serialize this pair."""
    with _LOCK:
        record = _record_for(context)
        return record.private_context_id, record.private_evaluation_id
