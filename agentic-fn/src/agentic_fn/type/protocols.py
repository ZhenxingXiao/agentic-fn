"""Structural protocols that define runtime contracts without binding implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentic_fn.core.state import State
    from agentic_fn.effects.log.record import LogRecord
    from agentic_fn.observe.snapshot import PipelineSnapshot
    from agentic_fn.type.enums import CircuitState


@runtime_checkable
class Clock(Protocol):
    """Monotonic and wall-clock time provider used for deadlines and metrics."""

    def monotonic(self) -> float:
        """Return a monotonic clock value in seconds."""

    def wall_time(self) -> float:
        """Return UNIX wall-clock time in seconds."""


@runtime_checkable
class Span(Protocol):
    """A tracing span representing a unit of work."""

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        """Attach a scalar attribute to the span."""

    def record_exception(self, exc: BaseException) -> None:
        """Record an exception on the span."""

    def end(self) -> None:
        """Finish the span."""


@runtime_checkable
class Tracer(Protocol):
    """Optional tracing backend (for example OpenTelemetry)."""

    def start_span(self, name: str, *, attributes: Mapping[str, str] | None = None) -> Span:
        """Create and start a new span."""


@runtime_checkable
class MergeStrategy(Protocol):
    """Merge parallel branch outputs back into a single state."""

    def merge(self, base: State, parts: Sequence[tuple[str, State]]) -> State:
        """
        Merge named branch results into ``base``.

        Parameters
        ----------
        base:
            State captured before fan-out.
        parts:
            Sequence of ``(branch_name, branch_state)`` pairs.
        """


@runtime_checkable
class Recorder(Protocol):
    """Records effect outputs for later replay or audit."""

    def record_effect(self, key: str, input_digest: str, output: State) -> None:
        """Persist an effect outcome keyed by effect identity and input digest."""


@runtime_checkable
class ReplayStore(Protocol):
    """Looks up previously recorded effect outcomes."""

    def lookup(self, key: str, input_digest: str) -> State | None:
        """Return a recorded state when present; otherwise ``None``."""


@runtime_checkable
class Harness(Protocol):
    """Optional production control plane (permissions, evaluation, sandboxing)."""

    def allow_step(self, step_name: str, kind: str) -> bool:
        """Return whether a step is permitted to execute."""

    def on_budget(self, remaining: Mapping[str, float | int | None]) -> None:
        """Receive budget updates for policy enforcement or metrics."""


@runtime_checkable
class SnapshotSource(Protocol):
    """Object that can produce a pipeline snapshot for ``query()``."""

    def query(self) -> PipelineSnapshot:
        """Return a thread/task-safe read-only snapshot."""


@runtime_checkable
class CircuitBreaker(Protocol):
    """Protects downstream effects from repeated failures."""

    @property
    def state(self) -> CircuitState:
        """Current breaker state."""

    def before_call(self) -> None:
        """Raise when the circuit is open and calls must fail fast."""

    def on_success(self) -> None:
        """Record a successful call."""

    def on_failure(self) -> None:
        """Record a failed call."""


@runtime_checkable
class LogSink(Protocol):
    """
    Pluggable destination for structured log records.

    Framework code depends only on this protocol so applications can bind
    stdlib logging, loguru, structlog, or proprietary sinks without coupling.
    """

    def emit(self, record: LogRecord) -> None:
        """Emit a single structured log record to the backing system."""


@runtime_checkable
class AsyncLogSink(Protocol):
    """Async counterpart of ``LogSink`` for non-blocking backends."""

    async def emit(self, record: LogRecord) -> None:
        """Asynchronously emit a single structured log record."""
