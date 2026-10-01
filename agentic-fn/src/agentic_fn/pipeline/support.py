"""Shared helpers for step execution, error actions, and tracing records."""

from __future__ import annotations

import copy
from typing import Any

from agentic_fn.core.errors import FrameworkError, PermanentError
from agentic_fn.core.result import StepRecord
from agentic_fn.core.state import ERRORS_KEY, State
from agentic_fn.core.step import StepSpec
from agentic_fn.type.enums import ErrorAction, StepKind


def append_error(state: State, *, step: str, error: BaseException) -> State:
    """Attach a soft-failure entry under the reserved ``_errors`` key."""

    existing = state.get(ERRORS_KEY, [])
    if not isinstance(existing, list):
        existing = list(existing) if isinstance(existing, tuple) else []
    errors = copy.deepcopy(existing)
    errors.append(
        {
            "step": step,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )
    return state.update(**{ERRORS_KEY: errors})


def apply_error_action(
    *,
    action: ErrorAction,
    state: State,
    step: StepSpec,
    error: BaseException,
) -> State:
    """
    Apply a configured error action and return the next state.

    Raises
    ------
    BaseException
        Re-raises when action is FAIL, or FALLBACK without a fallback callable.
    """

    if action is ErrorAction.FAIL:
        if isinstance(error, FrameworkError):
            raise error
        raise PermanentError(str(error), cause=error) from error

    if action is ErrorAction.SKIP:
        return state

    if action is ErrorAction.CONTINUE:
        return append_error(state, step=step.name, error=error)

    if action is ErrorAction.FALLBACK:
        fallback = step.policy.fallback
        if fallback is None:
            raise PermanentError(
                f"step {step.name!r} configured FALLBACK without fallback callable",
                cause=error,
            ) from error
        result = fallback(state, error)
        if not isinstance(result, State):
            raise PermanentError(
                f"fallback for step {step.name!r} must return State, got {type(result)!r}"
            )
        return result

    raise PermanentError(f"unknown error action: {action!r}", cause=error)


def can_retry_effect(step: StepSpec, attempt_index: int) -> bool:
    """
    Return whether another attempt is allowed after ``attempt_index`` failed.

    Non-idempotent effects never auto-retry. Capability steps may retry.
    """

    _ = attempt_index
    if step.kind is not StepKind.EFFECT:
        return True
    return step.policy.idempotent


def make_step_record(
    step: StepSpec,
    *,
    attempt: int,
    duration_ms: float,
    success: bool,
    error: BaseException | None = None,
    state: State | None = None,
) -> StepRecord:
    """Build a trace record for a step attempt."""

    return StepRecord(
        name=step.name,
        kind=step.kind,
        attempt=attempt,
        duration_ms=duration_ms,
        success=success,
        error_type=type(error).__name__ if error is not None else None,
        state_digest=state.digest() if state is not None else None,
    )


def ensure_framework_error(exc: BaseException) -> FrameworkError:
    """Wrap unknown exceptions as ``PermanentError`` for a stable error surface."""

    if isinstance(exc, FrameworkError):
        return exc
    return PermanentError(str(exc), cause=exc)
