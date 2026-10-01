"""Clock implementations used by deadlines and metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemClock:
    """Clock backed by the process system timers."""

    def monotonic(self) -> float:
        """Return ``time.monotonic()`` in seconds."""

        return time.monotonic()

    def wall_time(self) -> float:
        """Return ``time.time()`` in seconds."""

        return time.time()


@dataclass(slots=True)
class FakeClock:
    """Deterministic clock for tests and replay scenarios."""

    mono: float = 0.0
    wall: float = 0.0

    def monotonic(self) -> float:
        """Return the current fake monotonic time."""

        return self.mono

    def wall_time(self) -> float:
        """Return the current fake wall time."""

        return self.wall

    def advance(self, seconds: float) -> None:
        """Advance both clocks by ``seconds``."""

        self.mono += seconds
        self.wall += seconds
