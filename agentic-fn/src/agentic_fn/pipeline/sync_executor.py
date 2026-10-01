"""Synchronous pipeline graph executor."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from agentic_fn.core.context import Context
from agentic_fn.core.errors import (
    CancelledError,
    ConfigurationError,
    FrameworkError,
    HarnessDeniedError,
    PermanentError,
    TimeoutError,
    TransientError,
)
from agentic_fn.core.result import PipelineResult, RunMetrics, StepRecord
from agentic_fn.core.state import State, validate_state
from agentic_fn.core.step import StepSpec
from agentic_fn.observe.events import Event
from agentic_fn.pipeline.defaults import PipelineDefaults
from agentic_fn.pipeline.nodes import Branch, Loop, MapNode, Node, Parallel, Seq
from agentic_fn.pipeline.support import (
    apply_error_action,
    can_retry_effect,
    ensure_framework_error,
    make_step_record,
)
from agentic_fn.runtime.circuit import CircuitBreakerRegistry
from agentic_fn.runtime.executor import BoundedExecutor
from agentic_fn.type.enums import ErrorAction, EventKind, RunStatus, StepKind


class SyncExecutor:
    """
    Execute a ``Node`` graph synchronously with production controls.

    Responsibilities include timeouts, retries, circuit breaking, events,
    cooperative cancellation, and control-flow primitives.
    """

    def __init__(self, defaults: PipelineDefaults) -> None:
        """Create an executor bound to pipeline defaults."""

        self.defaults = defaults
        self._records: list[StepRecord] = []
        self._retry_count = 0
        self._effect_count = 0
        self._circuits = CircuitBreakerRegistry()
        self._owned_executor: BoundedExecutor | None = None

    @property
    def records(self) -> tuple[StepRecord, ...]:
        """Return collected step trace records."""

        return tuple(self._records)

    def metrics(self, duration_ms: float) -> RunMetrics:
        """Build aggregate metrics for the finished run."""

        return RunMetrics(
            duration_ms=duration_ms,
            step_count=len(self._records),
            retry_count=self._retry_count,
            effect_count=self._effect_count,
        )

    def close(self) -> None:
        """Release any executor owned by this run."""

        if self._owned_executor is not None:
            self._owned_executor.shutdown(wait=False)
            self._owned_executor = None

    def execute(self, node: Node, state: State, ctx: Context, path: tuple[str, ...] = ()) -> State:
        """Dispatch execution based on node type."""

        ctx.check()
        if isinstance(node, StepSpec):
            return self._run_step(node, state, ctx, path)
        if isinstance(node, Seq):
            return self._run_seq(node, state, ctx, path)
        if isinstance(node, Parallel):
            return self._run_parallel(node, state, ctx, path)
        if isinstance(node, Branch):
            return self._run_branch(node, state, ctx, path)
        if isinstance(node, Loop):
            return self._run_loop(node, state, ctx, path)
        if isinstance(node, MapNode):
            return self._run_map(node, state, ctx, path)
        raise ConfigurationError(f"unsupported node type: {type(node)!r}")

    def _emit(self, ctx: Context, event: Event) -> None:
        """Emit an event through the context hook bus."""

        ctx.hooks.emit(event)

    def _run_seq(self, node: Seq, state: State, ctx: Context, path: tuple[str, ...]) -> State:
        """Execute children in order."""

        current = state
        node_path = (*path, node.name)
        ctx.set_path(node_path)
        for child in node.children:
            current = self.execute(child, current, ctx, node_path)
            ctx.set_state(current)
        return current

    def _run_branch(
        self,
        node: Branch,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Evaluate predicate and execute the selected branch."""

        node_path = (*path, node.name)
        ctx.set_path(node_path)
        taken = bool(node.predicate(state, ctx))
        self._emit(
            ctx,
            Event(
                kind=EventKind.BRANCH_TAKEN,
                run_id=ctx.run_id,
                step_name=node.name,
                path=node_path,
                attributes={"taken": "then" if taken else "else"},
                state_digest=state.digest(),
            ),
        )
        if taken:
            return self.execute(node.then_node, state, ctx, node_path)
        if node.else_node is None:
            return state
        return self.execute(node.else_node, state, ctx, node_path)

    def _run_loop(self, node: Loop, state: State, ctx: Context, path: tuple[str, ...]) -> State:
        """Execute loop body until predicate or max_iter is hit."""

        current = state
        node_path = (*path, node.name)
        for iteration in range(node.max_iter):
            ctx.check()
            ctx.set_iteration(iteration)
            ctx.set_path(node_path)
            if node.until(current, ctx):
                return current
            self._emit(
                ctx,
                Event(
                    kind=EventKind.LOOP_ITER,
                    run_id=ctx.run_id,
                    step_name=node.name,
                    path=node_path,
                    attributes={"iteration": iteration},
                    state_digest=current.digest(),
                ),
            )
            current = self.execute(node.body, current, ctx, node_path)
            ctx.set_state(current)

        if node.on_max_iter is ErrorAction.FAIL:
            raise PermanentError(
                f"loop {node.name!r} exceeded max_iter={node.max_iter}"
            )
        if node.on_max_iter is ErrorAction.CONTINUE:
            return current
        if node.on_max_iter is ErrorAction.SKIP:
            return current
        raise PermanentError(f"loop {node.name!r} hit max_iter with action {node.on_max_iter}")

    def _run_parallel(
        self,
        node: Parallel,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Fan-out branches then merge results."""

        node_path = (*path, node.name)
        ctx.set_path(node_path)
        enable_parallel = node.enable_sync_parallel or self.defaults.enable_sync_parallel
        if enable_parallel and len(node.branches) > 1:
            parts = self._parallel_map(
                [
                    (branch_name, lambda n=branch_node: self.execute(n, state, ctx, node_path))
                    for branch_name, branch_node in node.branches
                ]
            )
        else:
            parts = [
                (branch_name, self.execute(branch_node, state, ctx, node_path))
                for branch_name, branch_node in node.branches
            ]
        merged = node.merge.merge(state, parts)
        self._emit(
            ctx,
            Event(
                kind=EventKind.PARALLEL_JOIN,
                run_id=ctx.run_id,
                step_name=node.name,
                path=node_path,
                state_digest=merged.digest(),
            ),
        )
        return merged

    def _run_map(
        self,
        node: MapNode,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Map a body over a sequence field in state."""

        node_path = (*path, node.name)
        ctx.set_path(node_path)
        items = state.get(node.over)
        if not isinstance(items, list):
            raise PermanentError(
                f"map node {node.name!r} expected list at key {node.over!r}, got {type(items)!r}"
            )

        def run_one(item: Any) -> State:
            item_state = state.update(**{node.item_key: item})
            return self.execute(node.body, item_state, ctx, node_path)

        enable_parallel = node.enable_sync_parallel or self.defaults.enable_sync_parallel
        if enable_parallel and node.concurrency > 1 and len(items) > 1:
            results = self._parallel_values([lambda i=item: run_one(i) for item in items])
        else:
            results = [run_one(item) for item in items]

        mapped_payloads = [result.data for result in results]
        return state.update(**{node.results_key: mapped_payloads})

    def _parallel_map(
        self,
        jobs: list[tuple[str, Any]],
    ) -> list[tuple[str, State]]:
        """Run named jobs on a bounded executor and preserve names."""

        executor = self._get_executor()
        futures = [(name, executor.submit(fn)) for name, fn in jobs]
        return [(name, future.result()) for name, future in futures]

    def _parallel_values(self, jobs: list[Any]) -> list[State]:
        """Run zero-arg jobs on a bounded executor preserving order."""

        executor = self._get_executor()
        futures = [executor.submit(fn) for fn in jobs]
        return [future.result() for future in futures]

    def _get_executor(self) -> BoundedExecutor:
        """Return the configured or lazily owned bounded executor."""

        if self.defaults.executor is not None:
            return self.defaults.executor
        if self._owned_executor is None:
            self._owned_executor = BoundedExecutor(max_workers=self.defaults.max_parallel)
        return self._owned_executor

    def _run_step(
        self,
        step: StepSpec,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Execute a single step with retry/timeout/circuit/error policy."""

        if step.is_async:
            raise ConfigurationError(
                f"sync pipeline cannot execute async step {step.name!r}"
            )

        node_path = (*path, step.name)
        ctx.set_path(node_path)
        ctx.check()

        if ctx.harness is not None and not ctx.harness.allow_step(step.name, step.kind.value):
            raise HarnessDeniedError(f"harness denied step {step.name!r}")

        self._emit(
            ctx,
            Event(
                kind=EventKind.STEP_START,
                run_id=ctx.run_id,
                step_name=step.name,
                path=node_path,
                state_digest=state.digest(),
            ),
        )

        policy = step.policy
        retry_policy = policy.resolved_retry(self.defaults.retry)
        max_attempts = retry_policy.max_attempts if retry_policy is not None else 1
        timeout = ctx.remaining_timeout(policy.resolved_timeout(self.defaults.timeout))

        breaker = None
        if policy.circuit_breaker:
            self._circuits.clock = ctx.clock
            breaker = self._circuits.get(policy.circuit_breaker)
            breaker.before_call()

        last_error: BaseException | None = None
        for attempt in range(max_attempts):
            ctx.check()
            started = ctx.clock.monotonic()
            span = ctx.tracer.start_span(
                step.name,
                attributes={
                    "step.name": step.name,
                    "step.kind": step.kind.value,
                    "run.id": ctx.run_id,
                },
            )
            try:
                if step.kind is StepKind.EFFECT:
                    self._effect_count += 1
                    ctx.budget.consume(calls=1)
                    output = self._call_with_timeout(
                        lambda: ctx.effects.run(
                            step.fn,
                            state,
                            ctx,
                            effect_key=step.name,
                        ),
                        timeout=timeout,
                    )
                else:
                    output = self._call_with_timeout(
                        lambda: step.fn(state, ctx),
                        timeout=timeout,
                    )

                if self.defaults.validate_state:
                    output = validate_state(output)

                duration_ms = (ctx.clock.monotonic() - started) * 1000.0
                self._records.append(
                    make_step_record(
                        step,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        success=True,
                        state=output,
                    )
                )
                if breaker is not None:
                    breaker.on_success()
                span.end()
                self._emit(
                    ctx,
                    Event(
                        kind=EventKind.STEP_END,
                        run_id=ctx.run_id,
                        step_name=step.name,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        path=node_path,
                        state_digest=output.digest(),
                    ),
                )
                return output
            except BaseException as exc:  # noqa: BLE001 - classified below
                last_error = exc
                duration_ms = (ctx.clock.monotonic() - started) * 1000.0
                span.record_exception(exc)
                span.end()
                self._records.append(
                    make_step_record(
                        step,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        success=False,
                        error=exc,
                        state=state,
                    )
                )
                self._emit(
                    ctx,
                    Event(
                        kind=EventKind.STEP_ERROR,
                        run_id=ctx.run_id,
                        step_name=step.name,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        error_type=type(exc).__name__,
                        path=node_path,
                        state_digest=state.digest(),
                    ),
                )

                if isinstance(exc, (CancelledError,)):
                    raise

                should_retry = False
                if retry_policy is not None and can_retry_effect(step, attempt):
                    should_retry = retry_policy.should_retry(exc, attempt)

                if should_retry:
                    self._retry_count += 1
                    delay = retry_policy.backoff.delay_for(attempt, rng=ctx.rng)
                    self._emit(
                        ctx,
                        Event(
                            kind=EventKind.STEP_RETRY,
                            run_id=ctx.run_id,
                            step_name=step.name,
                            attempt=attempt,
                            path=node_path,
                            attributes={"delay": delay},
                        ),
                    )
                    time.sleep(delay)
                    continue

                if breaker is not None:
                    breaker.on_failure()
                return apply_error_action(
                    action=policy.on_error,
                    state=state,
                    step=step,
                    error=exc,
                )

        assert last_error is not None
        if breaker is not None:
            breaker.on_failure()
        return apply_error_action(
            action=policy.on_error,
            state=state,
            step=step,
            error=last_error,
        )

    def _call_with_timeout(self, fn: Any, *, timeout: float | None) -> State:
        """Invoke ``fn`` optionally bounded by a timeout using a worker thread."""

        if timeout is None:
            result = fn()
            return result
        if timeout <= 0:
            raise TimeoutError("timeout budget already exhausted")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"step timed out after {timeout}s") from exc
