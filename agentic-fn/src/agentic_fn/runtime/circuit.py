"""Circuit breaker for protecting downstream effect dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from agentic_fn.core.errors import CircuitOpenError
from agentic_fn.type.enums import CircuitState
from agentic_fn.type.protocols import Clock


@dataclass(slots=True)
class SimpleCircuitBreaker:
    """
    Thread-safe circuit breaker with Closed → Open → HalfOpen transitions.

    Parameters
    ----------
    failure_threshold:
        Consecutive failures required to open the circuit.
    recovery_timeout:
        Seconds to wait before transitioning from Open to HalfOpen.
    clock:
        Clock used to measure recovery windows.
    """

    key: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    clock: Clock | None = None
    _state: CircuitState = CircuitState.CLOSED
    _consecutive_failures: int = 0
    _opened_at: float | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, applying recovery transitions."""

        with self._lock:
            self._maybe_half_open_unlocked()
            return self._state

    def before_call(self) -> None:
        """Raise ``CircuitOpenError`` when calls must fail fast."""

        with self._lock:
            self._maybe_half_open_unlocked()
            if self._state is CircuitState.OPEN:
                raise CircuitOpenError(f"circuit open: {self.key}")

    def on_success(self) -> None:
        """Record success and close the circuit."""

        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None

    def on_failure(self) -> None:
        """Record failure and potentially open the circuit."""

        with self._lock:
            self._consecutive_failures += 1
            if (
                self._state is CircuitState.HALF_OPEN
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                if self.clock is not None:
                    self._opened_at = self.clock.monotonic()
                else:
                    self._opened_at = 0.0

    def _maybe_half_open_unlocked(self) -> None:
        """Transition Open → HalfOpen when recovery timeout elapsed."""

        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return
        if self.clock is None:
            return
        if self.clock.monotonic() - self._opened_at >= self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN


@dataclass(slots=True)
class CircuitBreakerRegistry:
    """Named registry of circuit breakers shared by an effect runtime."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    clock: Clock | None = None
    _breakers: dict[str, SimpleCircuitBreaker] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def get(self, key: str) -> SimpleCircuitBreaker:
        """Return an existing breaker or create one for ``key``."""

        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is None:
                breaker = SimpleCircuitBreaker(
                    key=key,
                    failure_threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                    clock=self.clock,
                )
                self._breakers[key] = breaker
            else:
                breaker.clock = self.clock
            return breaker
