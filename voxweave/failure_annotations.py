from __future__ import annotations

from voxweave.align_failures import CanonicalFailure, SecondaryFailure


def _attach_canonical_failure(
    exc: BaseException,
    *,
    kind: str,
    phase: str,
    detail_code: str,
) -> None:
    """Classify an unchanged public exception without replacing its boundary."""
    try:
        if not isinstance(getattr(exc, "failure", None), CanonicalFailure):
            setattr(exc, "failure", CanonicalFailure(kind, phase, detail_code))
    except Exception:
        pass


def _append_secondary_terminal(
    primary: BaseException,
    secondary_exception: BaseException,
    *,
    kind: str,
    phase: str,
    detail_code: str,
) -> None:
    """Append one later closed terminal without replacing the first exception."""
    secondary = SecondaryFailure(kind, phase, detail_code)
    failure = getattr(primary, "failure", None)
    if isinstance(failure, CanonicalFailure):
        try:
            setattr(
                primary,
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
        current = tuple(getattr(primary, "secondary_failures", ()))
        setattr(primary, "secondary_failures", current + (secondary,))
        exceptions = tuple(getattr(primary, "secondary_exceptions", ()))
        setattr(
            primary,
            "secondary_exceptions",
            exceptions + (secondary_exception,),
        )
    except Exception:
        pass


def _record_disposal_failure(
    production_failure: BaseException | None,
    disposal_failure: BaseException | None,
    exc: BaseException,
    *,
    detail_code: str,
) -> BaseException:
    primary = production_failure or disposal_failure
    if primary is None:
        _attach_canonical_failure(
            exc,
            kind="snapshot-dispose-failed",
            phase="dispose",
            detail_code=detail_code,
        )
        return exc
    _append_secondary_terminal(
        primary,
        exc,
        kind="snapshot-dispose-failed",
        phase="dispose",
        detail_code=detail_code,
    )
    return disposal_failure or exc
