"""Run context: cancellation, deadlines, budget, effects, and observability."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agentic_fn.core.errors import BudgetExceeded, CancelledError, TimeoutError
from agentic_fn.core.state import State
from agentic_fn.effects.log.errors import EmitFailurePolicy
from agentic_fn.effects.log.sinks import NullLogSink
from agentic_fn.observe.hooks import HookBus
from agentic_fn.observe.snapshot import PipelineSnapshot
from agentic_fn.observe.tracer import NoOpTracer
from agentic_fn.runtime.budget import Budget, BudgetView, unlimited_budget
from agentic_fn.runtime.clock import SystemClock
from agentic_fn.runtime.effect import EffectRuntime
from agentic_fn.type.enums import RunStatus
from agentic_fn.type.protocols import Clock, Harness, LogSink, Tracer

if TYPE_CHECKING:
    from agentic_fn.effects.log.facade import ContextLogger
    from agentic_fn.models.runtime import ModelRuntime


@dataclass(slots=True)
class Context:
    """
    Control-plane object shared for the duration of a single pipeline run.

    Unlike ``State``, context is mutable and not passed as the business payload.
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    cancelled: bool = False
    deadline: float | None = None
    budget: Budget = field(default_factory=unlimited_budget)
    clock: Clock = field(default_factory=SystemClock)
    rng: random.Random = field(default_factory=random.Random)
    effects: EffectRuntime = field(default_factory=EffectRuntime)
    tracer: Tracer = field(default_factory=NoOpTracer)
    hooks: HookBus = field(default_factory=HookBus)
    harness: Harness | None = None
    log_sink: LogSink = field(default_factory=NullLogSink)
    log_failure_policy: EmitFailurePolicy = EmitFailurePolicy.IGNORE
    log_skip_on_replay: bool = True
    log_logger_name: str = "agentic_fn"
    models: ModelRuntime | None = None
    closed: bool = False
    _status: RunStatus = RunStatus.PENDING
    _current_path: tuple[str, ...] = ()
    _iteration: int | None = None
    _state: State | None = None
    _started_at: float = 0.0
    _last_error: str | None = None
    _lock: RLock = field(default_factory=RLock, repr=False)

    @property
    def log(self) -> ContextLogger:
        """
        Return a context-bound logger facade over ``log_sink``.

        Use inside Effect code for ergonomic structured logging without binding
        the framework to a concrete logging vendor.
        """

        from agentic_fn.effects.log.facade import ContextLogger

        return ContextLogger(
            sink=self.log_sink,
            ctx=self,
            logger_name=self.log_logger_name,
            failure_policy=self.log_failure_policy,
            skip_on_replay=self.log_skip_on_replay,
        )

    def check(self) -> None:
        """
        Raise when the run must stop due to cancel, deadline, or budget.

        Steps and control-flow nodes should call this at safe boundaries.
        """

        if self.closed:
            raise CancelledError("context is closed")
        if self.cancelled:
            raise CancelledError("pipeline run cancelled")
        if self.deadline is not None and self.clock.monotonic() >= self.deadline:
            raise TimeoutError("pipeline deadline exceeded")
        try:
            self.budget.ensure_available()
        except BudgetExceeded:
            raise

    def cancel(self) -> None:
        """Request cooperative cancellation of the active run."""

        with self._lock:
            self.cancelled = True
            self._status = RunStatus.CANCELLED

    def close(self) -> None:
        """Mark the context closed so late callbacks cannot mutate run state."""

        with self._lock:
            self.closed = True

    def set_status(self, status: RunStatus) -> None:
        """Update the run status."""

        with self._lock:
            self._status = status

    def set_path(self, path: tuple[str, ...]) -> None:
        """Update the current node path for snapshots and events."""

        with self._lock:
            self._current_path = path

    def set_iteration(self, iteration: int | None) -> None:
        """Update the current loop iteration counter."""

        with self._lock:
            self._iteration = iteration

    def set_state(self, state: State) -> None:
        """Publish the latest state for ``query()`` snapshots."""

        with self._lock:
            self._state = state

    def set_last_error(self, message: str | None) -> None:
        """Record the latest error message for snapshots."""

        with self._lock:
            self._last_error = message

    def mark_started(self) -> None:
        """Record the start timestamp using the configured clock."""

        with self._lock:
            self._started_at = self.clock.monotonic()
            self._status = RunStatus.RUNNING

    def remaining_timeout(self, step_timeout: float | None) -> float | None:
        """
        Compute the effective timeout for a step.

        Returns the stricter of step timeout and remaining pipeline deadline.
        """

        deadline_remaining: float | None = None
        if self.deadline is not None:
            deadline_remaining = max(0.0, self.deadline - self.clock.monotonic())
        if step_timeout is None:
            return deadline_remaining
        if deadline_remaining is None:
            return step_timeout
        return min(step_timeout, deadline_remaining)

    def query(self) -> PipelineSnapshot:
        """Return a thread-safe read-only snapshot of the current run."""

        with self._lock:
            digest = self._state.digest() if self._state is not None else ""
            return PipelineSnapshot(
                run_id=self.run_id,
                status=self._status,
                current_path=self._current_path,
                iteration=self._iteration,
                state_digest=digest,
                started_at=self._started_at,
                budget_remaining=self.budget.view(),
                last_error=self._last_error,
            )

    def budget_view(self) -> BudgetView:
        """Return the current budget view."""

        return self.budget.view()

    def spawn_child(self, *, label: str | None = None) -> Context:
        """
        Create a child context sharing budget/effects but with a new run id.

        Useful for nested evaluation harnesses; not required for normal steps.
        """

        _ = label
        return Context(
            run_id=str(uuid4()),
            cancelled=self.cancelled,
            deadline=self.deadline,
            budget=self.budget,
            clock=self.clock,
            rng=self.rng,
            effects=self.effects,
            tracer=self.tracer,
            hooks=self.hooks,
            harness=self.harness,
            log_sink=self.log_sink,
            log_failure_policy=self.log_failure_policy,
            log_skip_on_replay=self.log_skip_on_replay,
            log_logger_name=self.log_logger_name,
            models=self.models,
        )
