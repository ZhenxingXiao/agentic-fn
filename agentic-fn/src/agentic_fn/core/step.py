"""Step specifications and execution policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.core.retry import RetryPolicy
from agentic_fn.type.enums import ErrorAction, StepKind


@dataclass(frozen=True, slots=True)
class StepPolicy:
    """Per-step resilience and error-handling configuration."""

    timeout: float | None = None
    retry: RetryPolicy | None = None
    on_error: ErrorAction = ErrorAction.FAIL
    circuit_breaker: str | None = None
    idempotent: bool = False
    critical: bool = True
    fallback: Callable[..., Any] | None = None

    def resolved_timeout(self, default_timeout: float | None) -> float | None:
        """Return the effective timeout, preferring the step override."""

        return self.timeout if self.timeout is not None else default_timeout

    def resolved_retry(self, default_retry: RetryPolicy | None) -> RetryPolicy | None:
        """Return the effective retry policy, preferring the step override."""

        return self.retry if self.retry is not None else default_retry


@dataclass(frozen=True, slots=True)
class StepSpec:
    """
    A named pipeline step binding a function to policy and metadata.

    ``fn`` may be sync or async; the owning pipeline validates compatibility.
    """

    name: str
    kind: StepKind
    fn: Callable[..., Any]
    policy: StepPolicy = field(default_factory=StepPolicy)
    tags: tuple[str, ...] = ()
    is_async: bool = False
