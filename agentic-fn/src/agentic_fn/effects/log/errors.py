"""Logging-specific errors and emit failure policies."""

from __future__ import annotations

from enum import Enum

from agentic_fn.core.errors import FrameworkError, PermanentError, TransientError


class LogEmitError(FrameworkError):
    """Raised when a log sink fails to emit a record."""


class LogConfigurationError(PermanentError):
    """Raised when logging components are misconfigured."""


class EmitFailurePolicy(str, Enum):
    """
    How ``make_log_effect`` / context logger react to sink failures.

    ``IGNORE`` is recommended for best-effort diagnostics logging.
    ``RAISE_TRANSIENT`` / ``RAISE_PERMANENT`` are for audit-critical sinks.
    """

    IGNORE = "ignore"
    RAISE_TRANSIENT = "raise_transient"
    RAISE_PERMANENT = "raise_permanent"


def raise_according_to_policy(policy: EmitFailurePolicy, message: str, cause: BaseException) -> None:
    """Raise the appropriate framework error for an emit failure policy."""

    if policy is EmitFailurePolicy.IGNORE:
        return
    if policy is EmitFailurePolicy.RAISE_TRANSIENT:
        raise TransientError(message, cause=cause) from cause
    raise PermanentError(message, cause=cause) from cause
