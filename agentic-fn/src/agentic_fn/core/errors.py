"""Framework exception hierarchy for production error handling."""

from __future__ import annotations


class FrameworkError(Exception):
    """Base class for all agentic-fn controlled failures."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        """
        Create a framework error.

        Parameters
        ----------
        message:
            Human-readable error description.
        cause:
            Optional underlying exception retained for debugging.
        """

        super().__init__(message)
        self.message = message
        self.cause = cause


class TransientError(FrameworkError):
    """Retryable failure such as network blips, rate limits, or soft timeouts."""


class PermanentError(FrameworkError):
    """Non-retryable failure such as validation errors or permission denials."""


class CancelledError(FrameworkError):
    """Raised when a run is cancelled cooperatively via context control."""


class BudgetExceeded(FrameworkError):
    """Raised when a configured budget (calls, tokens, cost, or time) is exhausted."""


class TimeoutError(FrameworkError):
    """Raised when a step or pipeline deadline is exceeded."""


class CircuitOpenError(TransientError):
    """Raised when a circuit breaker is open and fail-fast is required."""


class StateValidationError(PermanentError):
    """Raised when a step returns an invalid or non-conforming state."""


class MergeConflictError(PermanentError):
    """Raised when parallel branches write conflicting keys under ERROR mode."""


class ReplayMissError(PermanentError):
    """Raised in REPLAY mode when a recorded effect outcome cannot be found."""


class ConfigurationError(PermanentError):
    """Raised when pipeline construction or runtime configuration is invalid."""


class HarnessDeniedError(PermanentError):
    """Raised when a harness policy rejects execution of a step."""
