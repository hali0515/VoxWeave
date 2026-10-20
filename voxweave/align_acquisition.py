"""Strict fresh-alignment capture, physical transforms, and private terminals.

This module sits after the selected-legacy helper chain.  Backend result nodes
are observed opaquely before distribution, then captured all-or-none into the
closed P6 value domain.  Absolute authority is derived from the physical audio
origin; it never borrows the selected legacy route origin.
"""

from __future__ import annotations

import copy
import math
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from voxweave.align_context import (
    IssuedAlignContext,
    _align_context_authority_profile,
    _align_context_stable_fields,
    _private_context_subject,
    consume_context_role,
    role_vector,
)
from voxweave.align_distribution import (
    AuthorityBlock,
    AuthorityCallInput,
    AuthorityDistributionReceipt,
    AuthorityLimitProfile,
    AuthoritySkippedBlockInput,
    LegacyCallDistributionReceipt,
    LegacyDistributionResult,
    RouteClaim,
    RouteExpectation,
    StrictFailureLocator,
    _build_context_authority_distribution,
    legacy_distribute_before_shift,
    legacy_retain_qwen_before_shift,
)
from voxweave.align_failures import CanonicalFailure, SecondaryFailure
from voxweave.align_runtime import align_runtime_activity
from voxweave.align_snapshot import (
    FrozenArray,
    frozen_json_digest,
)
from voxweave.align_strict_capture import (
    Provenance as Provenance,
    StrictCapturedUnit as StrictCapturedUnit,
    StrictCaptureResult as StrictCaptureResult,
    FreshUnit as FreshUnit,
    AuthorityTransformResult as AuthorityTransformResult,
    QwenSampleGeometry as QwenSampleGeometry,
    PhysicalCallReceipt as PhysicalCallReceipt,
    SampleGeometryError as SampleGeometryError,
    _attach_canonical_failure as _attach_canonical_failure,
    _exact_index as _exact_index,
    _exact_optional_float as _exact_optional_float,
    _same_optional_float as _same_optional_float,
    _strict_node as _strict_node,
    _optional_json_float as _optional_json_float,
    _relative_unit_value as _relative_unit_value,
    capture_strict_units as capture_strict_units,
    _absolute_unit_value as _absolute_unit_value,
    qwen_sample_geometry as qwen_sample_geometry,
)
from voxweave.align_fresh_seals import (
    AcquisitionAdmissionEvent as AcquisitionAdmissionEvent,
    AcquisitionAdmissionLedger as AcquisitionAdmissionLedger,
    FreshSealBroken as FreshSealBroken,
    _distribution_seal_failure as _distribution_seal_failure,
    IssuedFreshAlignment as IssuedFreshAlignment,
    VerifiedFreshAlignment as VerifiedFreshAlignment,
    FreshAlignmentSession as FreshAlignmentSession,
    _ObservedPhysicalCall as _ObservedPhysicalCall,
    ReferencePhysicalCallFacts as ReferencePhysicalCallFacts,
    _AdapterPayload as _AdapterPayload,
    _FreshSeals as _FreshSeals,
    _FreshRecord as _FreshRecord,
    _VerifiedRecord as _VerifiedRecord,
    _SessionRecord as _SessionRecord,
    EvidenceCoreProducerInputs as EvidenceCoreProducerInputs,
    EvidenceCoreReferenceInputs as EvidenceCoreReferenceInputs,
    _stable_fact_digest as _stable_fact_digest,
    _capture_digest_facts as _capture_digest_facts,
    _stable_value as _stable_value,
    _stable_digest as _stable_digest,
    _physical_projection as _physical_projection,
    _context_seal_inputs as _context_seal_inputs,
    _context_seal_digest as _context_seal_digest,
    _capture_seal_inputs as _capture_seal_inputs,
    _raw_seal_digest as _raw_seal_digest,
    _relative_seal_digest as _relative_seal_digest,
    _legacy_slice_seal_inputs as _legacy_slice_seal_inputs,
    _legacy_slice_seal_digest as _legacy_slice_seal_digest,
    _authority_seal_inputs as _authority_seal_inputs,
    _authority_seal_digest as _authority_seal_digest,
    _distribution_seal_inputs as _distribution_seal_inputs,
    _distribution_seal_digest as _distribution_seal_digest,
    _phase1_seal_inputs as _phase1_seal_inputs,
    _phase1_seal_digest as _phase1_seal_digest,
    _SEALS as _SEALS,
    _seal_fresh_record as _seal_fresh_record,
    _same_objects as _same_objects,
    _SEAL_PHASES as _SEAL_PHASES,
    _raise_component_seal as _raise_component_seal,
    _verify_fresh_seals as _verify_fresh_seals,
)


def transform_strict_units(
    capture: StrictCaptureResult,
    *,
    physical_origin_seconds: float,
    identity: bool,
    retained_unit_count: int | None = None,
) -> AuthorityTransformResult:
    """Project complete relative capture through the sealed physical origin."""
    if not isinstance(capture, StrictCaptureResult):
        raise TypeError("capture must be a StrictCaptureResult")
    if type(identity) is not bool:
        raise TypeError("identity must be an exact bool")
    if type(physical_origin_seconds) is not float or not math.isfinite(
        physical_origin_seconds
    ):
        raise SampleGeometryError(
            "sample-geometry", "physical origin must be one finite exact float"
        )
    if retained_unit_count is None:
        retained_count = len(capture.observed_unit_ids)
    else:
        retained_count = _exact_index(retained_unit_count, name="retained_unit_count")
        if retained_count > len(capture.observed_unit_ids):
            raise ValueError("retained_unit_count exceeds the captured result")
    if capture.status != "valid" or capture.units is None:
        return AuthorityTransformResult(
            capture.call_index, "invalid", capture, None, None, capture.failure
        )

    transformed: list[FreshUnit] = []
    for index, unit in enumerate(capture.units):
        relative_start = unit.relative_start
        relative_end = unit.relative_end
        valid = (
            relative_start is not None
            and relative_end is not None
            and math.isfinite(relative_start)
            and math.isfinite(relative_end)
            and relative_start <= relative_end
        )
        if not valid:
            detail = (
                "authority-recompute" if index < retained_count else "surplus-transform"
            )
            return AuthorityTransformResult(
                capture.call_index,
                "invalid",
                capture,
                None,
                None,
                StrictFailureLocator("authority-transform", index, detail),
            )
        assert relative_start is not None
        assert relative_end is not None
        if identity:
            start = relative_start
            end = relative_end
        else:
            start = relative_start + physical_origin_seconds
            end = relative_end + physical_origin_seconds
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            detail = (
                "authority-recompute" if index < retained_count else "surplus-transform"
            )
            return AuthorityTransformResult(
                capture.call_index,
                "invalid",
                capture,
                None,
                None,
                StrictFailureLocator("authority-transform", index, detail),
            )
        transformed.append(
            FreshUnit(
                unit.unit_id,
                unit.call_index,
                unit.call_unit_index,
                unit.surface,
                relative_start,
                relative_end,
                physical_origin_seconds,
                start,
                end,
                unit.provenance,
                unit.original_relative_start,
                unit.original_relative_end,
                unit.raw,
            )
        )
    units = tuple(transformed)
    return AuthorityTransformResult(
        capture.call_index,
        "valid",
        capture,
        units,
        frozen_json_digest(FrozenArray(tuple(_absolute_unit_value(u) for u in units))),
        None,
    )


def raise_distribution_seal_mismatch(
    context: IssuedAlignContext,
    *,
    terminal_call_index: int,
    ledger: AcquisitionAdmissionLedger,
    dispose: Callable[[], None],
) -> None:
    """Append the sole private AO-13 terminal, dispose, and always raise."""
    call_index = _exact_index(terminal_call_index, name="terminal_call_index")
    if not isinstance(ledger, AcquisitionAdmissionLedger):
        raise TypeError("ledger must be an AcquisitionAdmissionLedger")
    if not callable(dispose):
        raise TypeError("dispose must be callable")
    context_key = id(context)
    with ledger._lock:
        if context_key in ledger._terminal_contexts:
            raise FreshSealBroken(_distribution_seal_failure())
        vector = role_vector(context)
        if not vector or vector[0] != "C":
            raise FreshSealBroken(
                CanonicalFailure(
                    "fresh-authority-invalid",
                    "authority-distribution",
                    "acquisition-unissued",
                )
            )
        subject = _private_context_subject(context)
        event = AcquisitionAdmissionEvent(
            "acquisition-failed",
            subject,
            (
                "AO-13",
                "fresh-seal-broken",
                "authority-distribution",
                "distribution-seal",
                call_index,
                (),
            ),
        )
        ledger.events.append(event)
        ledger._terminal_contexts.add(context_key)

    secondary: tuple[SecondaryFailure, ...] = ()
    try:
        dispose()
    except Exception:  # noqa: BLE001 - cleanup is a canonical secondary terminal
        secondary = (
            SecondaryFailure("snapshot-dispose-failed", "dispose", "stage-residue"),
        )
    raise FreshSealBroken(_distribution_seal_failure(secondary))


_FRESH: dict[int, _FreshRecord] = {}
_VERIFIED_FRESH: dict[int, _VerifiedRecord] = {}
_FRESH_SESSIONS: dict[int, _SessionRecord] = {}
_FRESH_LOCK = threading.RLock()
_ISSUER_TOKEN = object()


def _invalid_capture(
    call_index: int,
    raw_unit_ids: tuple[str, ...],
    *,
    index: int = 0,
) -> StrictCaptureResult:
    return StrictCaptureResult(
        call_index,
        "invalid",
        None,
        None,
        None,
        StrictFailureLocator("strict-capture", index, "strict-raw-node"),
        raw_unit_ids,
    )


class FreshAlignmentIssuer:
    """Private AO-05 through AO-13 owner of live physical-call observations."""

    def __init__(
        self,
        token: object,
        context: IssuedAlignContext,
        *,
        alignment_texts: Sequence[str],
        source_texts: Sequence[str] | None,
        source_indices: Sequence[int],
        language: str,
        prepared_audio_sample_count: int,
        sample_rate: int,
        backend_model_config_digest: str | None,
        route_input_digest: str | None,
        backend_model_config_facts: object | None,
        route_input_facts: object | None,
        ledger: AcquisitionAdmissionLedger | None,
    ) -> None:
        if token is not _ISSUER_TOKEN:
            raise TypeError("FreshAlignmentIssuer is private")
        if language != context.effective_iso:
            raise ValueError("fresh acquisition language is cross-context")
        if (
            type(prepared_audio_sample_count) is not int
            or prepared_audio_sample_count < 0
        ):
            raise ValueError("prepared audio sample count must be nonnegative")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample rate must be a positive exact integer")
        texts = tuple(alignment_texts)
        sealed_texts = texts if source_texts is None else tuple(source_texts)
        original_sources = tuple(source_indices)
        if (
            len(original_sources) != len(texts)
            or len(sealed_texts) != len(texts)
            or len(set(original_sources)) != len(original_sources)
            or any(type(index) is not int or index < 0 for index in original_sources)
            or any(type(text) is not str for text in sealed_texts)
        ):
            raise ValueError(
                "fresh acquisition source indices must be unique nonnegative integers"
            )
        consume_context_role(context, "acquisition", consumer="FreshAlignmentIssuer")
        self.context = context
        self.blocks = tuple(
            AuthorityBlock(source_index, str(text), sealed_text)
            for source_index, text, sealed_text in zip(
                original_sources, texts, sealed_texts, strict=True
            )
        )
        self.language = language
        self.prepared_audio_sample_count = prepared_audio_sample_count
        self.sample_rate = sample_rate
        (
            self.backend_model_config_facts,
            self.backend_model_config_digest,
        ) = _capture_digest_facts(
            backend_model_config_facts,
            backend_model_config_digest,
            label="backend model/config",
            default_facts={
                "kind": "default",
                "route": context.route_kind,
                "language": language,
            },
        )
        self.route_input_facts, self.route_input_digest = _capture_digest_facts(
            route_input_facts,
            route_input_digest,
            label="route input",
            default_facts={
                "kind": "default",
                "context_content_digest": context.context_content_digest,
                "route": context.route_kind,
            },
        )
        self.ledger = ledger or AcquisitionAdmissionLedger()
        self.observed: list[_ObservedPhysicalCall] = []
        self.raw_cursor = 0
        self.sealed = False

    def _observe_physical_call(
        self,
        post_units: Sequence[Any],
        original_units: Sequence[Any] | None,
        source_indices: Sequence[int],
        legacy_origin_seconds: float,
        *,
        audio_sample_start: int | None = None,
        audio_sample_end: int | None = None,
        sample_rate: int | None = None,
        sample_count: int | None = None,
        nominal_end_seconds: float | None = None,
    ) -> None:
        """AO-07 callback: retain identity/length only; recurse only at sealing."""
        if self.sealed:
            raise RuntimeError("fresh acquisition issuer is already sealed")
        try:
            raw_count = len(post_units)
        except Exception as exc:
            classified = TypeError("backend result must be one sized sequence")
            _attach_canonical_failure(
                classified,
                kind="fresh-backend-output-invalid",
                phase="backend-output",
                detail_code="backend-call-shape",
            )
            raise classified from exc
        if original_units is not None:
            try:
                len(original_units)
            except Exception as exc:
                classified = TypeError("original backend result must be sized")
                _attach_canonical_failure(
                    classified,
                    kind="fresh-backend-output-invalid",
                    phase="backend-output",
                    detail_code="backend-call-shape",
                )
                raise classified from exc
        source_positions = tuple(source_indices)
        if (
            not source_positions
            or any(
                type(position) is not int or not 0 <= position < len(self.blocks)
                for position in source_positions
            )
            or len(set(source_positions)) != len(source_positions)
        ):
            raise ValueError("physical call delivery positions are invalid")
        sources = tuple(
            self.blocks[position].source_index for position in source_positions
        )
        call_index = len(self.observed)
        raw_start = self.raw_cursor
        raw_ids = tuple(
            f"r{index}" for index in range(raw_start, raw_start + raw_count)
        )
        self.raw_cursor += raw_count

        rate = self.sample_rate if sample_rate is None else sample_rate
        count = (
            self.prepared_audio_sample_count if sample_count is None else sample_count
        )
        geometry_failure: StrictFailureLocator | None = None
        if type(legacy_origin_seconds) is not float or not math.isfinite(
            legacy_origin_seconds
        ):
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            legacy_origin = 0.0
        else:
            legacy_origin = legacy_origin_seconds
        if type(rate) is not int or rate <= 0 or type(count) is not int or count < 0:
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            rate = self.sample_rate
            count = self.prepared_audio_sample_count

        geometry_supplied = (
            audio_sample_start is not None
            and audio_sample_end is not None
            and sample_rate is not None
            and sample_count is not None
        )
        if geometry_supplied:
            start = audio_sample_start
            end = audio_sample_end
        elif self.context.route_kind != "qwen-crop" and legacy_origin == 0.0:
            start = 0
            end = count
        else:
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            start = 0
            end = 0
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= count
        ):
            geometry_failure = StrictFailureLocator(
                "sample-geometry", None, "sample-geometry"
            )
            start = 0
            end = 0
        physical_origin = start / rate
        if self.context.route_kind == "qwen-crop":
            legacy_kind: Literal["identity", "sample-origin", "nominal-route"] = (
                "nominal-route"
            )
            if nominal_end_seconds is None:
                geometry_failure = StrictFailureLocator(
                    "sample-geometry", None, "sample-geometry"
                )
            else:
                try:
                    expected = qwen_sample_geometry(
                        nominal_start=legacy_origin,
                        nominal_end=nominal_end_seconds,
                        sample_rate=rate,
                        sample_count=count,
                    )
                except SampleGeometryError:
                    geometry_failure = StrictFailureLocator(
                        "sample-geometry", None, "sample-geometry"
                    )
                else:
                    if expected.sample_start != start or expected.sample_end != end:
                        geometry_failure = StrictFailureLocator(
                            "sample-geometry", None, "physical-origin-mismatch"
                        )
        elif start == 0 and legacy_origin == 0.0:
            legacy_kind = "identity"
        else:
            legacy_kind = "sample-origin"
            if physical_origin.hex() != legacy_origin.hex():
                geometry_failure = StrictFailureLocator(
                    "sample-geometry", None, "physical-origin-mismatch"
                )

        # Every call shares the issuer's validated source-fact digests.
        model_digest = self.backend_model_config_digest
        input_digest = self.route_input_digest
        with align_runtime_activity("AO-07", "raw-id-and-call-observation"):
            self.observed.append(
                _ObservedPhysicalCall(
                    call_index,
                    sources,
                    post_units,
                    original_units,
                    raw_ids,
                    (raw_start, raw_start + raw_count),
                    start,
                    end,
                    rate,
                    count,
                    physical_origin,
                    legacy_origin,
                    legacy_kind,
                    physical_origin,
                    model_digest,
                    input_digest,
                    geometry_failure,
                )
            )

    def _invoke_backend_call(self, backend_call: Callable[[], Any]) -> Any:
        """Preserve a backend exception while attaching its closed classification."""
        if not callable(backend_call):
            raise TypeError("backend call must be callable")
        with align_runtime_activity("AO-07", "backend-call"):
            try:
                return backend_call()
            except Exception as exc:
                _attach_canonical_failure(
                    exc,
                    kind="fresh-backend-output-invalid",
                    phase="backend-call",
                    detail_code="backend-raised",
                )
                raise

    def _invoke_qwen_physical_call(
        self,
        backend_call: Callable[[], Sequence[Any]],
        source_position: int,
        legacy_origin_seconds: float,
        nominal_end_seconds: float,
        *,
        audio_sample_start: int | None,
        audio_sample_end: int | None,
        sample_rate: int | None,
        sample_count: int | None,
    ) -> Sequence[Any]:
        """Own one Qwen backend attempt and its inseparable AO-07 observation."""
        if self.context.route_kind != "qwen-crop":
            raise ValueError("Qwen invocation is unavailable for this route")
        if not callable(backend_call):
            raise TypeError("Qwen backend call must be callable")
        raw_units = self._invoke_backend_call(backend_call)
        self._observe_physical_call(
            raw_units,
            None,
            (source_position,),
            legacy_origin_seconds,
            audio_sample_start=audio_sample_start,
            audio_sample_end=audio_sample_end,
            sample_rate=sample_rate,
            sample_count=sample_count,
            nominal_end_seconds=nominal_end_seconds,
        )
        return raw_units

    def _dispose(self) -> None:
        self.observed.clear()


def begin_fresh_alignment(
    context: IssuedAlignContext,
    *,
    alignment_texts: Sequence[str],
    source_texts: Sequence[str] | None = None,
    source_indices: Sequence[int],
    language: str,
    prepared_audio_sample_count: int,
    sample_rate: int = 16_000,
    backend_model_config_digest: str | None = None,
    route_input_digest: str | None = None,
    backend_model_config_facts: object | None = None,
    route_input_facts: object | None = None,
    ledger: AcquisitionAdmissionLedger | None = None,
) -> FreshAlignmentSession:
    """Issue an opaque session; the private backend seam owns its observations."""
    issuer = FreshAlignmentIssuer(
        _ISSUER_TOKEN,
        context,
        alignment_texts=alignment_texts,
        source_texts=source_texts,
        source_indices=source_indices,
        language=language,
        prepared_audio_sample_count=prepared_audio_sample_count,
        sample_rate=sample_rate,
        backend_model_config_digest=backend_model_config_digest,
        route_input_digest=route_input_digest,
        backend_model_config_facts=backend_model_config_facts,
        route_input_facts=route_input_facts,
        ledger=ledger,
    )
    session = object.__new__(FreshAlignmentSession)
    binding = secrets.token_hex(32)
    object.__setattr__(
        session, "context_content_digest", context.context_content_digest
    )
    object.__setattr__(session, "_binding", binding)
    with _FRESH_LOCK:
        _FRESH_SESSIONS[id(session)] = _SessionRecord(session, issuer, binding)
    return session


def _fresh_session_record(session: FreshAlignmentSession) -> _SessionRecord:
    with _FRESH_LOCK:
        record = _FRESH_SESSIONS.get(id(session))
        if (
            record is None
            or record.session is not session
            or session._binding != record.binding
            or session.context_content_digest
            != record.issuer.context.context_content_digest
            or record.sealed
        ):
            raise ValueError(
                "fresh alignment session is unissued, changed, or consumed"
            )
        return record


def _fresh_alignment_call_observer(
    session: FreshAlignmentSession,
) -> Callable[..., None]:
    """Return the issuer-bound callback used only inside backend physical seams."""
    issuer = _fresh_session_record(session).issuer
    return issuer._observe_physical_call


def _fresh_alignment_qwen_invoker(
    session: FreshAlignmentSession,
) -> Callable[..., Sequence[Any]]:
    """Return the private issuer-owned Qwen physical invocation seam."""
    issuer = _fresh_session_record(session).issuer
    return issuer._invoke_qwen_physical_call


def _fresh_alignment_backend_invoker(
    session: FreshAlignmentSession,
) -> Callable[[Callable[[], Any]], Any]:
    """Return the issuer-owned wrapper for one configured backend attempt."""
    issuer = _fresh_session_record(session).issuer
    return issuer._invoke_backend_call


def _legacy_call_projection(
    issuer: FreshAlignmentIssuer,
    call: _ObservedPhysicalCall,
) -> LegacyDistributionResult:
    blocks_by_source = {block.source_index: block for block in issuer.blocks}
    texts = tuple(
        blocks_by_source[index].alignment_text for index in call.source_block_indices
    )
    try:
        if issuer.context.route_kind == "qwen-crop":
            legacy = legacy_retain_qwen_before_shift(
                call.post_units,
                origin=call.legacy_origin_seconds,
                identity=call.legacy_origin_kind == "identity",
                raw_unit_ids=call.raw_unit_ids,
                source_indices=call.source_block_indices,
            )
        else:
            legacy = legacy_distribute_before_shift(
                call.post_units,
                texts=texts,
                iso=issuer.language,
                origin=call.legacy_origin_seconds,
                identity=call.legacy_origin_kind == "identity",
                raw_unit_ids=call.raw_unit_ids,
                source_indices=call.source_block_indices,
            )
    except Exception as exc:
        missing_key = exc.args[0] if isinstance(exc, KeyError) and exc.args else None
        retained_field = missing_key if type(missing_key) is str else ""
        detail_code = {
            "text": "retained-unit-text",
            "start": "retained-unit-start",
            "end": "retained-unit-end",
        }.get(retained_field, "retained-unit-operand")
        _attach_canonical_failure(
            exc,
            kind="legacy-time-transform-failed",
            phase="legacy-time-transform",
            detail_code=detail_code,
        )
        raise
    return legacy


def _strict_call_capture(call: _ObservedPhysicalCall) -> StrictCaptureResult:
    with align_runtime_activity("AO-11", "strict-capture"):
        try:
            return capture_strict_units(
                call.post_units,
                original_units=call.original_units,
                call_index=call.call_index,
                raw_unit_ids=call.raw_unit_ids,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(getattr(exc, "failure", None), CanonicalFailure):
                raise
            return _invalid_capture(call.call_index, call.raw_unit_ids)


def _authority_call_transform(
    call: _ObservedPhysicalCall,
    capture: StrictCaptureResult,
    legacy: LegacyDistributionResult,
) -> AuthorityTransformResult:
    if capture.status != "valid":
        return AuthorityTransformResult(
            call.call_index,
            "invalid",
            capture,
            None,
            None,
            capture.failure,
        )
    retained_count = len(legacy.receipt.consumed_prefix_unit_ids)
    with align_runtime_activity("AO-12", "physical-origin-transform"):
        if call.geometry_failure is not None:
            return AuthorityTransformResult(
                call.call_index,
                "invalid",
                capture,
                None,
                None,
                call.geometry_failure,
            )
        return transform_strict_units(
            capture,
            physical_origin_seconds=call.authority_origin_seconds,
            identity=call.legacy_origin_kind == "identity",
            retained_unit_count=retained_count,
        )


def _assemble_call_capture(
    call: _ObservedPhysicalCall,
    capture: StrictCaptureResult,
    transform: AuthorityTransformResult,
    legacy: LegacyDistributionResult,
) -> tuple[
    StrictCaptureResult,
    AuthorityTransformResult,
    AuthorityCallInput,
    PhysicalCallReceipt,
    LegacyCallDistributionReceipt,
    tuple[tuple[Mapping[str, Any], ...], ...],
    tuple[tuple[Mapping[str, Any], ...], ...],
]:
    if capture.status != "valid":
        preflight_status: Literal["valid", "capture-invalid", "transform-invalid"] = (
            "capture-invalid"
        )
        surfaces = None
        failure = capture.failure
    elif transform.status != "valid":
        preflight_status = "transform-invalid"
        surfaces = tuple(unit.surface for unit in capture.units or ())
        failure = transform.failure
    else:
        preflight_status = "valid"
        surfaces = tuple(unit.surface for unit in capture.units or ())
        failure = None
    call_input = AuthorityCallInput(
        call.call_index,
        call.source_block_indices,
        call.raw_node_range,
        call.raw_unit_ids,
        surfaces,
        preflight_status,
        failure,
    )
    legacy_slice_digest = _stable_digest(legacy.receipt)
    try:
        legacy_absolute_digest = _stable_digest(legacy.block_units)
    except Exception as exc:
        # A retained unit outside the JSON domain (e.g. a non-float time) cannot be
        # sealed; every later consumer would reject a missing digest, so fail here
        # with the unit's own error instead.
        _attach_canonical_failure(
            exc,
            kind="legacy-time-transform-failed",
            phase="legacy-time-transform",
            detail_code="retained-unit-operand",
        )
        raise
    physical = PhysicalCallReceipt(
        call_index=call.call_index,
        source_block_indices=call.source_block_indices,
        audio_sample_start=call.audio_sample_start,
        audio_sample_end=call.audio_sample_end,
        sample_rate=call.sample_rate,
        physical_origin_seconds=call.physical_origin_seconds,
        legacy_origin_seconds=call.legacy_origin_seconds,
        legacy_origin_kind=call.legacy_origin_kind,
        authority_origin_seconds=call.authority_origin_seconds,
        backend_model_config_digest=call.backend_model_config_digest,
        route_input_digest=call.route_input_digest,
        strict_unit_status=capture.status,
        strict_failure=transform.failure,
        raw_units_digest=capture.raw_units_digest,
        normalized_relative_digest=capture.normalized_relative_digest,
        legacy_slice_digest=legacy_slice_digest,
        legacy_absolute_digest=legacy_absolute_digest,
        authority_transform_status=transform.status,
        authority_absolute_digest=transform.authority_absolute_digest,
        raw_unit_ids=call.raw_unit_ids,
    )
    return (
        capture,
        transform,
        call_input,
        physical,
        legacy.receipt,
        tuple(tuple(unit for unit in owner) for owner in legacy.block_units),
        tuple(tuple(unit for unit in owner) for owner in legacy.relative_block_units),
    )


def _route_inputs(
    issuer: FreshAlignmentIssuer,
    calls: tuple[AuthorityCallInput, ...],
) -> tuple[
    tuple[RouteExpectation, ...],
    tuple[AuthoritySkippedBlockInput, ...],
    tuple[RouteClaim, ...],
]:
    owners = {
        source_index: call.call_index
        for call in calls
        for source_index in call.source_block_indices
    }
    route: list[RouteExpectation] = []
    skipped: list[AuthoritySkippedBlockInput] = []
    claims: list[RouteClaim] = []
    for delivery_index, block in enumerate(issuer.blocks):
        owner_index = owners.get(block.source_index)
        if owner_index is not None:
            route.append(
                RouteExpectation(
                    delivery_index, block.source_index, "call", owner_index
                )
            )
            claims.append(
                RouteClaim("call", owner_index, delivery_index, block.source_index)
            )
            continue
        stripped = block.alignment_text.strip()
        skip_index = len(skipped)
        skipped.append(
            AuthoritySkippedBlockInput(
                delivery_index,
                block.source_index,
                "empty-alignment-text" if not stripped else "missing-crop",
                "empty"
                if not block.alignment_text
                else "whitespace"
                if not stripped
                else "nonempty",
            )
        )
        route.append(
            RouteExpectation(delivery_index, block.source_index, "skip", skip_index)
        )
        claims.append(
            RouteClaim("skip", skip_index, delivery_index, block.source_index)
        )
    return tuple(route), tuple(skipped), tuple(claims)


def _issued_public_snapshot(issued: IssuedFreshAlignment) -> tuple[object, ...]:
    return (
        issued.context_content_digest,
        issued.receipt_digest,
        issued.physical_calls,
        issued.distribution,
        issued.seed_status,
        issued.seed_reasons,
        issued._binding,
    )


def _reference_call_facts(call: _ObservedPhysicalCall) -> ReferencePhysicalCallFacts:
    return ReferencePhysicalCallFacts(
        call.call_index,
        call.source_block_indices,
        tuple(
            copy.deepcopy(call.post_units[index])
            for index in range(len(call.post_units))
        ),
        None
        if call.original_units is None
        else tuple(
            copy.deepcopy(call.original_units[index])
            for index in range(len(call.original_units))
        ),
        call.raw_unit_ids,
        call.raw_node_range,
        call.audio_sample_start,
        call.audio_sample_end,
        call.sample_rate,
        call.sample_count,
        call.physical_origin_seconds,
        call.legacy_origin_seconds,
        call.legacy_origin_kind,
        call.authority_origin_seconds,
        call.geometry_failure,
    )


def seal_fresh_alignment(session: FreshAlignmentSession) -> IssuedFreshAlignment:
    """Run AO-11 through AO-13 and atomically issue one sealed acquisition."""
    if not isinstance(session, FreshAlignmentSession):
        raise TypeError("fresh alignment session is invalid")
    session_record = _fresh_session_record(session)
    session_record.sealed = True
    issuer = session_record.issuer
    issuer.sealed = True
    observed = tuple(issuer.observed)
    legacy_projections = tuple(
        _legacy_call_projection(issuer, call) for call in observed
    )
    captures = tuple(_strict_call_capture(call) for call in observed)
    transforms = tuple(
        _authority_call_transform(call, capture, legacy)
        for call, capture, legacy in zip(
            observed, captures, legacy_projections, strict=True
        )
    )
    rows = tuple(
        _assemble_call_capture(call, capture, transform, legacy)
        for call, capture, transform, legacy in zip(
            observed,
            captures,
            transforms,
            legacy_projections,
            strict=True,
        )
    )
    reference_calls = tuple(_reference_call_facts(call) for call in observed)
    call_inputs = tuple(row[2] for row in rows)
    physical_calls = tuple(row[3] for row in rows)
    legacy_receipts = tuple(row[4] for row in rows)
    legacy_by_source: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for call, row in zip(observed, rows, strict=True):
        for source_index, owner in zip(call.source_block_indices, row[5], strict=True):
            legacy_by_source[source_index] = owner
    legacy_block_units = tuple(
        legacy_by_source.get(block.source_index, ()) for block in issuer.blocks
    )
    legacy_relative_by_source: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for call, row in zip(observed, rows, strict=True):
        for source_index, owner in zip(call.source_block_indices, row[6], strict=True):
            legacy_relative_by_source[source_index] = owner
    legacy_relative_block_units = tuple(
        legacy_relative_by_source.get(block.source_index, ()) for block in issuer.blocks
    )
    with align_runtime_activity("AO-13", "route-topology-preflight"):
        route, skipped, claims = _route_inputs(issuer, call_inputs)
        profile = _align_context_authority_profile(issuer.context)
        if not isinstance(profile, AuthorityLimitProfile):
            raise TypeError("context authority profile has the wrong type")
    with align_runtime_activity("AO-13", "canonical-authority-allocation"):
        distribution = _build_context_authority_distribution(
            blocks=issuer.blocks,
            delivery_route=route,
            calls=call_inputs,
            skipped_blocks=skipped,
            route_claims=claims,
            iso=issuer.language,
            _limits=profile,
        )
    if distribution.work.status == "seal-mismatch":
        terminal = distribution.work.terminal_call_index
        raise_distribution_seal_mismatch(
            issuer.context,
            terminal_call_index=0 if terminal is None else terminal,
            ledger=issuer.ledger,
            dispose=issuer._dispose,
        )
    from voxweave.core.align_seed import build_align_seed

    fresh_units = tuple(
        unit
        for transform in transforms
        for unit in (transform.units if transform.units is not None else ())
    )
    with align_runtime_activity("AO-13", "authority-seed"):
        seed = build_align_seed(
            blocks=issuer.blocks,
            units=fresh_units,
            distribution=distribution,
            iso=issuer.language,
        )
    receipt_digest = _stable_digest(
        {
            "context_content_digest": issuer.context.context_content_digest,
            "physical_calls": physical_calls,
            "legacy_distribution": legacy_receipts,
            "authority_distribution": distribution,
            "seed_status": seed.status,
            "seed_reasons": seed.reasons,
        }
    )
    issued = object.__new__(IssuedFreshAlignment)
    object.__setattr__(
        issued, "context_content_digest", issuer.context.context_content_digest
    )
    object.__setattr__(issued, "receipt_digest", receipt_digest)
    object.__setattr__(issued, "physical_calls", physical_calls)
    object.__setattr__(issued, "distribution", distribution)
    object.__setattr__(issued, "seed_status", seed.status)
    object.__setattr__(issued, "seed_reasons", seed.reasons)
    object.__setattr__(issued, "_binding", secrets.token_hex(32))
    record = _FreshRecord(
        issuer.context,
        issued,
        (),
        issuer.blocks,
        copy.deepcopy(captures),
        copy.deepcopy(transforms),
        copy.deepcopy(distribution),
        copy.deepcopy(seed),
        copy.deepcopy(physical_calls),
        copy.deepcopy(legacy_receipts),
        copy.deepcopy(legacy_block_units),
        copy.deepcopy(legacy_relative_block_units),
        reference_calls,
        copy.deepcopy(issuer.backend_model_config_facts),
        copy.deepcopy(issuer.route_input_facts),
    )
    record.public_snapshot = _issued_public_snapshot(issued)
    _seal_fresh_record(record)
    with _FRESH_LOCK:
        _FRESH[id(issued)] = record
    issuer._dispose()
    return issued


def _fresh_record(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> _FreshRecord:
    with _FRESH_LOCK:
        record = _FRESH.get(id(acquisition))
        if (
            record is None
            or record.context is not context
            or record.issued is not acquisition
        ):
            raise ValueError("fresh alignment is unissued, changed, or cross-context")
        _verify_fresh_seals(record)
        if (
            acquisition.context_content_digest != context.context_content_digest
            or _issued_public_snapshot(acquisition) != record.public_snapshot
        ):
            raise ValueError("fresh alignment is unissued, changed, or cross-context")
        return record


def _release_fresh_alignments(context: IssuedAlignContext) -> None:
    """Forget the sessions, sealed acquisitions and transfers issued for ``context``.

    Their records hold deep copies of every captured unit; see
    :func:`voxweave.align_orchestration.release_align_selection`.
    """
    with _FRESH_LOCK:
        for key in [key for key, row in _FRESH.items() if row.context is context]:
            del _FRESH[key]
        for key in [
            key for key, row in _VERIFIED_FRESH.items() if row.context is context
        ]:
            del _VERIFIED_FRESH[key]
        for key in [
            key for key, row in _FRESH_SESSIONS.items() if row.issuer.context is context
        ]:
            del _FRESH_SESSIONS[key]


def _bind_fresh_adapter_payload(
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
    *,
    legacy_delivery: object,
    projection_inputs: object,
    strict_input_status: object,
    v2_policy_status: object,
    profile_resolution: object,
    evidence_resolution: object,
) -> None:
    record = _fresh_record(context, acquisition)
    if record.adapter_payload is not None:
        raise ValueError("fresh alignment adapter payload is already bound")
    cues = getattr(legacy_delivery, "cues", None)
    top_units = getattr(legacy_delivery, "word_segments", None)
    if not isinstance(cues, tuple) or not isinstance(top_units, tuple):
        raise ValueError("legacy delivery lacks its immutable AO-10 unit projection")

    def scalar_fact(value: Any) -> tuple[str, Any]:
        return ("float", value.hex()) if type(value) is float else ("value", value)

    def expected_facts(
        units: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                scalar_fact(unit["text"]),
                scalar_fact(unit["start"]),
                scalar_fact(unit["end"]),
            )
            for unit in units
        )

    def delivered_facts(units: Sequence[object]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                scalar_fact(getattr(unit, "text", None)),
                scalar_fact(getattr(unit, "start", None)),
                scalar_fact(getattr(unit, "end", None)),
            )
            for unit in units
        )

    if tuple(getattr(cue, "source_index", None) for cue in cues) != tuple(
        block.source_index for block in record.blocks
    ):
        raise ValueError("legacy delivery source order changed after acquisition")
    for cue, expected_units in zip(cues, record.legacy_block_units, strict=True):
        if delivered_facts(getattr(cue, "word_data", ())) != expected_facts(
            expected_units
        ):
            raise ValueError(
                "legacy delivery block units do not match the sealed slice"
            )
    expected_top = tuple(
        fact for owner in record.legacy_block_units for fact in expected_facts(owner)
    )
    if delivered_facts(top_units) != expected_top:
        raise ValueError(
            "legacy delivery top-level units do not match the sealed slice"
        )
    record.adapter_payload = copy.deepcopy(
        _AdapterPayload(
            legacy_delivery,
            projection_inputs,
            strict_input_status,
            v2_policy_status,
            profile_resolution,
            evidence_resolution,
        )
    )


def _fresh_adapter_payload(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> _AdapterPayload:
    record = _fresh_record(context, acquisition)
    if record.adapter_payload is None:
        raise ValueError("fresh alignment lacks its sealed adapter payload")
    return copy.deepcopy(record.adapter_payload)


def _verify_fresh_alignment(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> VerifiedFreshAlignment | None:
    record = _fresh_record(context, acquisition)
    seed = record.seed
    if (
        acquisition.distribution.status != "valid"
        or getattr(seed, "status", None) != "valid"
        or any(transform.status != "valid" for transform in record.transforms)
    ):
        record.transfer_terminal = "retired"
        return None
    if record.verified is not None or record.transfer_terminal != "live":
        raise ValueError("fresh alignment transfer was already issued or retired")
    if record.adapter_payload is None:
        raise ValueError("fresh alignment transfer lacks adapter admission")
    profile = getattr(record.adapter_payload.profile_resolution, "profile", None)
    if profile is None:
        record.transfer_terminal = "retired"
        return None
    evidence = record.adapter_payload.evidence_resolution
    sing_spans = getattr(evidence, "sing_spans", None)
    if sing_spans is None:
        admitted_sing_spans: tuple[tuple[float, float], ...] = ()
    else:
        admitted_sing_spans = tuple(tuple(span) for span in sing_spans)
    verified = object.__new__(VerifiedFreshAlignment)
    object.__setattr__(
        verified, "context_content_digest", context.context_content_digest
    )
    object.__setattr__(verified, "receipt_digest", acquisition.receipt_digest)
    object.__setattr__(verified, "_binding", secrets.token_hex(32))
    record.verified = verified
    with _FRESH_LOCK:
        _VERIFIED_FRESH[id(verified)] = _VerifiedRecord(
            context,
            acquisition,
            verified,
            copy.deepcopy(seed),
            copy.deepcopy(profile),
            copy.deepcopy(admitted_sing_spans),
            verified._binding,
        )
    return verified


def _consume_verified_fresh_alignment(
    verified: VerifiedFreshAlignment,
    *,
    profile: object,
) -> tuple[object, str, tuple[tuple[float, float], ...]]:
    with _FRESH_LOCK:
        verified_record = _VERIFIED_FRESH.get(id(verified))
        if (
            verified_record is None
            or verified_record.verified is not verified
            or verified._binding != verified_record.binding
            or verified.context_content_digest
            != verified_record.context.context_content_digest
            or verified.receipt_digest != verified_record.acquisition.receipt_digest
            or profile != verified_record.profile_snapshot
        ):
            raise ValueError("verified fresh alignment is unissued or changed")
        record = _fresh_record(verified_record.context, verified_record.acquisition)
        if record.verified is not verified or record.transfer_terminal != "live":
            raise ValueError("fresh alignment transfer is not live")
        record.transfer_terminal = "consumed"
        return (
            copy.deepcopy(verified_record.seed_snapshot),
            verified.receipt_digest,
            copy.deepcopy(verified_record.sing_spans_snapshot),
        )


def _retire_fresh_transfer(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> None:
    record = _fresh_record(context, acquisition)
    if record.transfer_terminal == "live":
        record.transfer_terminal = "retired"


def _fresh_core_inputs(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> tuple[
    tuple[AuthorityBlock, ...],
    tuple[StrictCaptureResult, ...],
    tuple[AuthorityTransformResult, ...],
    AuthorityDistributionReceipt,
    str,
    tuple[str, ...],
    tuple[PhysicalCallReceipt, ...],
]:
    record = _fresh_record(context, acquisition)
    seed = record.seed
    return (
        copy.deepcopy(record.blocks),
        copy.deepcopy(record.captures),
        copy.deepcopy(record.transforms),
        copy.deepcopy(record.distribution),
        getattr(seed, "status"),
        tuple(getattr(seed, "reasons")),
        copy.deepcopy(record.physical_calls),
    )


def _fresh_producer_core_inputs(
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
    *,
    strict_input_status: object,
    v2_policy_status: object,
    profile_status: object,
    evidence_status: object,
) -> EvidenceCoreProducerInputs:
    """Return the producer's own thaw of all AO-16 fields."""
    record = _fresh_record(context, acquisition)
    seed = record.seed
    profile = _align_context_authority_profile(context)
    if not isinstance(profile, AuthorityLimitProfile):
        raise TypeError("context authority profile has the wrong type")
    return EvidenceCoreProducerInputs(
        context.context_content_digest,
        acquisition.receipt_digest,
        context.engine_family,
        context.effective_iso,
        context.route_kind,
        copy.deepcopy(_align_context_stable_fields(context)),
        copy.deepcopy(profile),
        copy.deepcopy(record.blocks),
        copy.deepcopy(record.captures),
        copy.deepcopy(record.transforms),
        copy.deepcopy(record.distribution),
        getattr(seed, "status"),
        tuple(getattr(seed, "reasons")),
        copy.deepcopy(record.physical_calls),
        copy.deepcopy(record.legacy_receipts),
        copy.deepcopy(record.legacy_block_units),
        copy.deepcopy(record.legacy_relative_block_units),
        copy.deepcopy(record.backend_model_config_facts),
        copy.deepcopy(record.route_input_facts),
        copy.deepcopy(strict_input_status),
        copy.deepcopy(v2_policy_status),
        copy.deepcopy(profile_status),
        copy.deepcopy(evidence_status),
    )


def _fresh_reference_core_inputs(
    context: IssuedAlignContext,
    acquisition: IssuedFreshAlignment,
) -> EvidenceCoreReferenceInputs:
    """Return separate issuer/context thaws for independent projector C."""
    record = _fresh_record(context, acquisition)
    payload = record.adapter_payload
    if payload is None:
        raise ValueError("fresh alignment lacks its sealed adapter payload")
    profile = _align_context_authority_profile(context)
    if not isinstance(profile, AuthorityLimitProfile):
        raise TypeError("context authority profile has the wrong type")
    seed = record.seed
    profile_status = getattr(payload.profile_resolution, "status")
    evidence_status = getattr(payload.evidence_resolution, "status")
    return EvidenceCoreReferenceInputs(
        context.context_content_digest,
        acquisition.receipt_digest,
        context.engine_family,
        context.effective_iso,
        context.route_kind,
        copy.deepcopy(_align_context_stable_fields(context)),
        copy.deepcopy(profile),
        copy.deepcopy(record.blocks),
        copy.deepcopy(record.captures),
        copy.deepcopy(record.transforms),
        copy.deepcopy(record.distribution),
        getattr(seed, "status"),
        tuple(getattr(seed, "reasons")),
        copy.deepcopy(record.physical_calls),
        copy.deepcopy(record.legacy_receipts),
        copy.deepcopy(record.legacy_block_units),
        copy.deepcopy(record.legacy_relative_block_units),
        copy.deepcopy(record.reference_calls),
        copy.deepcopy(record.backend_model_config_facts),
        copy.deepcopy(record.route_input_facts),
        copy.deepcopy(payload.strict_input_status),
        copy.deepcopy(payload.v2_policy_status),
        copy.deepcopy(profile_status),
        copy.deepcopy(evidence_status),
    )


def _fresh_seed(
    context: IssuedAlignContext, acquisition: IssuedFreshAlignment
) -> object:
    return copy.deepcopy(_fresh_record(context, acquisition).seed)


__all__ = [
    "AcquisitionAdmissionEvent",
    "AcquisitionAdmissionLedger",
    "AuthorityTransformResult",
    "FreshSealBroken",
    "FreshAlignmentSession",
    "FreshUnit",
    "IssuedFreshAlignment",
    "PhysicalCallReceipt",
    "QwenSampleGeometry",
    "SampleGeometryError",
    "StrictCaptureResult",
    "StrictCapturedUnit",
    "VerifiedFreshAlignment",
    "begin_fresh_alignment",
    "capture_strict_units",
    "qwen_sample_geometry",
    "raise_distribution_seal_mismatch",
    "seal_fresh_alignment",
    "transform_strict_units",
]
