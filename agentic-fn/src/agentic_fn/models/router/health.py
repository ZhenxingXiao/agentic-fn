"""Provider health view used by latency-aware routing and filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic


@dataclass(slots=True)
class _Stats:
    """Rolling success/latency stats for one breaker key."""

    successes: int = 0
    failures: int = 0
    latency_ewma_ms: float | None = None
    open: bool = False
    updated_at: float = field(default_factory=monotonic)


@dataclass(slots=True)
class ProviderHealthView:
    """
    Thread-safe health registry consulted by the router.

    Circuit open state should be mirrored here by ``ModelRuntime`` after breaker
    transitions so latency_aware / filtering stay consistent.
    """

    alpha: float = 0.3  # EWMA factor for latency
    _stats: dict[str, _Stats] = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def set_open(self, breaker_key: str, open_: bool) -> None:
        """Mark a breaker key as open/closed."""

        with self._lock:
            stats = self._stats.setdefault(breaker_key, _Stats())
            stats.open = open_
            stats.updated_at = monotonic()

    def is_open(self, breaker_key: str) -> bool:
        """Return whether the key is marked open."""

        with self._lock:
            stats = self._stats.get(breaker_key)
            return bool(stats and stats.open)

    def record_success(self, breaker_key: str, latency_ms: float) -> None:
        """Record a successful call latency."""

        with self._lock:
            stats = self._stats.setdefault(breaker_key, _Stats())
            stats.successes += 1
            stats.open = False
            if stats.latency_ewma_ms is None:
                stats.latency_ewma_ms = latency_ms
            else:
                stats.latency_ewma_ms = (
                    self.alpha * latency_ms + (1.0 - self.alpha) * stats.latency_ewma_ms
                )
            stats.updated_at = monotonic()

    def record_failure(self, breaker_key: str) -> None:
        """Record a failed call."""

        with self._lock:
            stats = self._stats.setdefault(breaker_key, _Stats())
            stats.failures += 1
            stats.updated_at = monotonic()

    def latency_ms(self, breaker_key: str) -> float | None:
        """Return EWMA latency when known."""

        with self._lock:
            stats = self._stats.get(breaker_key)
            return None if stats is None else stats.latency_ewma_ms

    def error_rate(self, breaker_key: str) -> float:
        """Return failures / (successes+failures), or 0 when no data."""

        with self._lock:
            stats = self._stats.get(breaker_key)
            if stats is None:
                return 0.0
            total = stats.successes + stats.failures
            if total <= 0:
                return 0.0
            return stats.failures / total
