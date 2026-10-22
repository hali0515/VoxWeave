"""Hand the shadow artifact of a finished align call to the caller's observer.

The artifact is built after the call has been committed and disposed of. It
never changes the production result: every failure here is logged and dropped.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from typing import Any

from voxweave.align_failures import CanonicalFailure

log = logging.getLogger("voxweave")


def notify_align_shadow_observer(
    observer: Callable[[object], object],
    *,
    selection: Any,
    input_summary: Mapping[str, Any],
    prepared_audio_sha256: str,
) -> None:
    """Build rich/minimal observation after disposal; never change production."""
    from voxweave.align_evidence import encode_align_evidence

    evidence_sha256 = hashlib.sha256(
        encode_align_evidence(selection.evidence)
    ).hexdigest()
    try:
        from voxweave import align_shadow

        artifact = align_shadow.build_rich_align_shadow_artifact(
            selection=selection,
            input_summary=input_summary,
            prepared_audio_sha256=prepared_audio_sha256,
        )
    except Exception as rich_error:
        log.warning("rich align shadow construction failed: %s", rich_error)
        try:
            from voxweave import align_shadow_minimal

            artifact = align_shadow_minimal.build_minimal_align_shadow_failure_artifact(
                context_content_digest=selection.context.context_content_digest,
                receipt_digest=selection.result.receipt_digest,
                engine_family=selection.verified.engine_family,
                vtt_sha256=selection.verified.vtt_sha256,
                json_sha256=selection.verified.main_json_sha256,
                evidence_sha256=evidence_sha256,
                prior_failure=(
                    selection.observation_failure or selection.result.v2_status.failure
                ),
            )
        except Exception as minimal_error:
            failure = CanonicalFailure(
                "shadow-artifact-unavailable",
                "minimal-artifact",
                "minimal-artifact-construction",
            )
            log.warning(
                "align shadow artifact unavailable: %s",
                minimal_error,
                extra={"failure": failure},
            )
            return
    try:
        observer(artifact)
    except Exception as exc:
        failure = CanonicalFailure(
            "observer-failed",
            "observer",
            "observer-callback",
        )
        log.warning(
            "align shadow observer failed: %s",
            exc,
            extra={"failure": failure},
        )
