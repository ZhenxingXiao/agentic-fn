"""Asynchronous pipeline graph executor."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agentic_fn.core.context import Context
from agentic_fn.core.errors import (
    CancelledError,
    ConfigurationError,
    HarnessDeniedError,
    PermanentError,
    TimeoutError,
)
from agentic_fn.core.result import RunMetrics, StepRecord
from agentic_fn.core.state import State, validate_state
from agentic_fn.core.step import StepSpec
from agentic_fn.observe.events import Event
from agentic_fn.pipeline.defaults import PipelineDefaults
from agentic_fn.pipeline.nodes import Branch, Loop, MapNode, Node, Parallel, Seq
from agentic_fn.pipeline.support import (
    apply_error_action,
    can_retry_effect,
    make_step_record,
)
from agentic_fn.runtime.circuit import CircuitBreakerRegistry
from agentic_fn.type.enums import ErrorAction, EventKind, StepKind


class AsyncExecutor:
    """Execute a ``Node`` graph with asyncio, including bounded concurrency."""

    def __init__(self, defaults: PipelineDefaults) -> None:
        """Create an async executor bound to pipeline defaults."""

        self.defaults = defaults
        self._records: list[StepRecord] = []
        self._retry_count = 0
        self._effect_count = 0
        self._circuits = CircuitBreakerRegistry()
        self._lock = asyncio.Lock()

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

    async def execute(
        self,
        node: Node,
        state: State,
        ctx: Context,
        path: tuple[str, ...] = (),
    ) -> State:
        """Dispatch async execution based on node type."""

        ctx.check()
        if isinstance(node, StepSpec):
            return await self._run_step(node, state, ctx, path)
        if isinstance(node, Seq):
            return await self._run_seq(node, state, ctx, path)
        if isinstance(node, Parallel):
            return await self._run_parallel(node, state, ctx, path)
        if isinstance(node, Branch):
            return await self._run_branch(node, state, ctx, path)
        if isinstance(node, Loop):
            return await self._run_loop(node, state, ctx, path)
        if isinstance(node, MapNode):
            return await self._run_map(node, state, ctx, path)
        raise ConfigurationError(f"unsupported node type: {type(node)!r}")

    async def _emit(self, ctx: Context, event: Event) -> None:
        """Emit an event through sync and async hooks."""

        await ctx.hooks.emit_async(event)

    async def _run_seq(
        self,
        node: Seq,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Execute children sequentially."""

        current = state
        node_path = (*path, node.name)
        ctx.set_path(node_path)
        for child in node.children:
            current = await self.execute(child, current, ctx, node_path)
            ctx.set_state(current)
        return current

    async def _run_branch(
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
        await self._emit(
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
            return await self.execute(node.then_node, state, ctx, node_path)
        if node.else_node is None:
            return state
        return await self.execute(node.else_node, state, ctx, node_path)

    async def _run_loop(
        self,
        node: Loop,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Execute loop body until predicate or max_iter is hit."""

        current = state
        node_path = (*path, node.name)
        for iteration in range(node.max_iter):
            ctx.check()
            ctx.set_iteration(iteration)
            ctx.set_path(node_path)
            if node.until(current, ctx):
                return current
            await self._emit(
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
            current = await self.execute(node.body, current, ctx, node_path)
            ctx.set_state(current)

        if node.on_max_iter is ErrorAction.FAIL:
            raise PermanentError(
                f"loop {node.name!r} exceeded max_iter={node.max_iter}"
            )
        if node.on_max_iter in {ErrorAction.CONTINUE, ErrorAction.SKIP}:
            return current
        raise PermanentError(f"loop {node.name!r} hit max_iter with action {node.on_max_iter}")

    async def _run_parallel(
        self,
        node: Parallel,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Fan-out branches concurrently then merge."""

        node_path = (*path, node.name)
        ctx.set_path(node_path)
        sem = asyncio.Semaphore(self.defaults.max_parallel)

        async def run_branch(branch_name: str, branch_node: Node) -> tuple[str, State]:
            async with sem:
                result = await self.execute(branch_node, state, ctx, node_path)
                return branch_name, result

        parts = list(
            await asyncio.gather(
                *[run_branch(name, branch) for name, branch in node.branches]
            )
        )
        merged = node.merge.merge(state, parts)
        await self._emit(
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

    async def _run_map(
        self,
        node: MapNode,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Map a body over a sequence field with bounded concurrency."""

        node_path = (*path, node.name)
        ctx.set_path(node_path)
        items = state.get(node.over)
        if not isinstance(items, list):
            raise PermanentError(
                f"map node {node.name!r} expected list at key {node.over!r}, got {type(items)!r}"
            )

        sem = asyncio.Semaphore(node.concurrency)

        async def run_one(item: Any) -> State:
            async with sem:
                item_state = state.update(**{node.item_key: item})
                return await self.execute(node.body, item_state, ctx, node_path)

        results = list(await asyncio.gather(*[run_one(item) for item in items]))
        mapped_payloads = [result.data for result in results]
        return state.update(**{node.results_key: mapped_payloads})

    async def _run_step(
        self,
        step: StepSpec,
        state: State,
        ctx: Context,
        path: tuple[str, ...],
    ) -> State:
        """Execute a single sync or async step under async orchestration."""

        node_path = (*path, step.name)
        ctx.set_path(node_path)
        ctx.check()

        if ctx.harness is not None and not ctx.harness.allow_step(step.name, step.kind.value):
            raise HarnessDeniedError(f"harness denied step {step.name!r}")

        await self._emit(
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
                output = await self._invoke_step(step, state, ctx, timeout=timeout)
                if self.defaults.validate_state:
                    output = validate_state(output)

                duration_ms = (ctx.clock.monotonic() - started) * 1000.0
                async with self._lock:
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
                await self._emit(
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
                async with self._lock:
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
                await self._emit(
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

                if isinstance(exc, CancelledError):
                    raise
                if isinstance(exc, asyncio.CancelledError):
                    raise CancelledError("async task cancelled", cause=exc) from exc

                should_retry = False
                if retry_policy is not None and can_retry_effect(step, attempt):
                    should_retry = retry_policy.should_retry(exc, attempt)

                if should_retry:
                    async with self._lock:
                        self._retry_count += 1
                    delay = retry_policy.backoff.delay_for(attempt, rng=ctx.rng)
                    await self._emit(
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
                    await asyncio.sleep(delay)
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

    async def _invoke_step(
        self,
        step: StepSpec,
        state: State,
        ctx: Context,
        *,
        timeout: float | None,
    ) -> State:
        """Invoke a step function, offloading sync effects to a worker thread."""

        async def call() -> State:
            if step.kind is StepKind.EFFECT:
                async with self._lock:
                    self._effect_count += 1
                ctx.budget.consume(calls=1)
                if step.is_async:
                    return await ctx.effects.run_async(
                        step.fn,
                        state,
                        ctx,
                        effect_key=step.name,
                        is_async=True,
                    )
                return await asyncio.to_thread(
                    ctx.effects.run,
                    step.fn,
                    state,
                    ctx,
                    effect_key=step.name,
                )

            if step.is_async:
                result = step.fn(state, ctx)
                if isinstance(result, Awaitable):
                    return await result
                raise PermanentError(f"async step {step.name!r} did not return awaitable")

            # Sync capability: run inline (CPU-light by convention).
            return step.fn(state, ctx)

        if timeout is None:
            return await call()
        if timeout <= 0:
            raise TimeoutError("timeout budget already exhausted")
        try:
            return await asyncio.wait_for(call(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"step timed out after {timeout}s") from exc
