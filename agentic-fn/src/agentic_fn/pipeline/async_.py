"""Asynchronous pipeline facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_fn.core.context import Context
from agentic_fn.core.errors import CancelledError, FrameworkError
from agentic_fn.core.result import PipelineResult
from agentic_fn.core.state import State
from agentic_fn.observe.events import Event
from agentic_fn.observe.snapshot import PipelineSnapshot
from agentic_fn.pipeline.async_executor import AsyncExecutor
from agentic_fn.pipeline.defaults import PipelineDefaults
from agentic_fn.pipeline.nodes import Node
from agentic_fn.pipeline.support import ensure_framework_error
from agentic_fn.type.enums import EventKind, RunStatus

if TYPE_CHECKING:
    from agentic_fn.pipeline.builder import AsyncPipelineBuilder


class AsyncPipeline:
    """
    Production-oriented asynchronous pipeline.

    Supports async steps natively and may offload synchronous effects through
    ``asyncio.to_thread`` under a concurrency bound.
    """

    def __init__(self, root: Node, *, defaults: PipelineDefaults | None = None) -> None:
        """
        Create an async pipeline.

        Parameters
        ----------
        root:
            Root node of the pipeline AST.
        defaults:
            Pipeline-wide defaults; a named default config is used when omitted.
        """

        self.root = root
        self.defaults = defaults or PipelineDefaults()
        self._active_ctx: Context | None = None

    @classmethod
    def builder(
        cls,
        name: str = "pipeline",
        **defaults_kwargs: object,
    ) -> AsyncPipelineBuilder:
        """Return a fluent builder for asynchronous pipelines."""

        from agentic_fn.pipeline.builder import AsyncPipelineBuilder

        defaults = PipelineDefaults(name=name, **defaults_kwargs)  # type: ignore[arg-type]
        return AsyncPipelineBuilder(defaults=defaults)

    def query(self) -> PipelineSnapshot:
        """
        Query the active run snapshot.

        Raises
        ------
        RuntimeError
            If no run is currently active on this pipeline instance.
        """

        if self._active_ctx is None:
            raise RuntimeError("no active async pipeline run")
        return self._active_ctx.query()

    async def run(self, state: State, *, ctx: Context | None = None) -> PipelineResult:
        """Execute the pipeline asynchronously and return a structured result."""

        context = ctx or Context()
        if state.meta.run_id:
            context.run_id = state.meta.run_id
        else:
            state = state.with_meta(state.meta.with_run_id(context.run_id))

        if self.defaults.on_event is not None:
            context.hooks.add(self.defaults.on_event)
        if self.defaults.on_event_async is not None:
            context.hooks.add_async(self.defaults.on_event_async)
        if self.defaults.run_deadline is not None:
            context.deadline = context.clock.monotonic() + self.defaults.run_deadline

        context.mark_started()
        context.set_state(state)
        self._active_ctx = context
        executor = AsyncExecutor(self.defaults)
        started = context.clock.monotonic()

        await context.hooks.emit_async(
            Event(
                kind=EventKind.PIPELINE_START,
                run_id=context.run_id,
                step_name=self.defaults.name,
                state_digest=state.digest(),
            )
        )

        result: PipelineResult
        try:
            output = await executor.execute(self.root, state, context)
            context.set_state(output)
            context.set_status(RunStatus.SUCCEEDED)
            duration_ms = (context.clock.monotonic() - started) * 1000.0
            result = PipelineResult(
                state=output,
                status=RunStatus.SUCCEEDED,
                trace=executor.records,
                error=None,
                metrics=executor.metrics(duration_ms),
            )
        except CancelledError as exc:
            result = self._failure_result(
                context,
                executor,
                started,
                state,
                status=RunStatus.CANCELLED,
                error=exc,
            )
            await context.hooks.emit_async(
                Event(
                    kind=EventKind.CANCELLED,
                    run_id=context.run_id,
                    step_name=self.defaults.name,
                    error_type=type(exc).__name__,
                )
            )
        except FrameworkError as exc:
            result = self._failure_result(
                context,
                executor,
                started,
                state,
                status=RunStatus.FAILED,
                error=exc,
            )
        except Exception as exc:  # noqa: BLE001 - normalize unknown failures
            wrapped = ensure_framework_error(exc)
            result = self._failure_result(
                context,
                executor,
                started,
                state,
                status=RunStatus.FAILED,
                error=wrapped,
            )
        finally:
            await context.hooks.emit_async(
                Event(
                    kind=EventKind.PIPELINE_END,
                    run_id=context.run_id,
                    step_name=self.defaults.name,
                    attributes={"status": context.query().status.value},
                )
            )
            context.close()
            self._active_ctx = None

        return result

    def _failure_result(
        self,
        context: Context,
        executor: AsyncExecutor,
        started: float,
        fallback_state: State,
        *,
        status: RunStatus,
        error: FrameworkError,
    ) -> PipelineResult:
        """Build a failed/cancelled result using the latest known state."""

        context.set_status(status)
        context.set_last_error(str(error))
        duration_ms = (context.clock.monotonic() - started) * 1000.0
        latest = getattr(context, "_state", None)
        state = latest if isinstance(latest, State) else fallback_state
        return PipelineResult(
            state=state,
            status=status,
            trace=executor.records,
            error=error,
            metrics=executor.metrics(duration_ms),
        )
