"""Route decision structures returned by the router."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_fn.models.router.config import Candidate, FailoverPolicy, RouteEntry


@dataclass(frozen=True, slots=True)
class RouteAttempt:
    """One concrete attempt the runtime should try."""

    provider: str
    physical_model: str
    breaker_key: str
    reason: str
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Ordered attempt plan produced by ``ModelRouter.resolve``."""

    logical_model: str
    strategy: str
    attempts: tuple[RouteAttempt, ...]
    entry: RouteEntry
    filtered: tuple[str, ...] = ()
    failover: FailoverPolicy | None = None

    @property
    def primary(self) -> RouteAttempt:
        """First attempt or raise."""

        if not self.attempts:
            raise LookupError("route decision has no attempts")
        return self.attempts[0]
