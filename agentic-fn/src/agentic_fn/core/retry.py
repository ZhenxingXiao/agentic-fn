"""Retry and backoff policies for resilient step execution."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from agentic_fn.core.errors import TransientError


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """Exponential backoff with optional full jitter."""

    initial: float = 0.2
    factor: float = 2.0
    max_delay: float = 5.0
    jitter: bool = True

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """
        Compute sleep delay before the given zero-based retry attempt.

        Parameters
        ----------
        attempt:
            Zero-based retry index (0 means first retry after the initial try).
        rng:
            Optional RNG for deterministic jitter in tests/replay.
        """

        raw = min(self.max_delay, self.initial * (self.factor**attempt))
        if not self.jitter:
            return raw
        generator = rng or random.Random()
        return generator.uniform(0.0, raw)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Declarative retry policy attached to a step or pipeline defaults."""

    max_attempts: int = 3
    backoff: ExponentialBackoff = field(default_factory=ExponentialBackoff)
    retry_on: tuple[type[BaseException], ...] = (TransientError,)

    def __post_init__(self) -> None:
        """Validate policy configuration."""

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def should_retry(self, exc: BaseException, attempt_index: int) -> bool:
        """
        Return whether ``exc`` should be retried after ``attempt_index`` failed tries.

        ``attempt_index`` is zero-based for the attempt that just failed.
        """

        if attempt_index + 1 >= self.max_attempts:
            return False
        return isinstance(exc, self.retry_on)


def default_retry_policy() -> RetryPolicy:
    """Return the framework default retry policy."""

    return RetryPolicy()
