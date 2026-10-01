"""Synchronous pipeline facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_fn.core.context import Context
from agentic_fn.core.errors import CancelledError, FrameworkError
from agentic_fn.core.result import PipelineResult
from agentic_fn.core.state import State
from agentic_fn.observe.events import Event
from agentic_fn.observe.snapshot import PipelineSnapshot
from agentic_fn.pipeline.defaults import PipelineDefaults
from agentic_fn.pipeline.nodes import Node
from agentic_fn.pipeline.support import ensure_framework_error
from agentic_fn.pipeline.sync_executor import SyncExecutor
from agentic_fn.type.enums import EventKind, RunStatus

if TYPE_CHECKING:
    from agentic_fn.pipeline.builder import SyncPipelineBuilder


class SyncPipeline:
    """
    Production-oriented synchronous pipeline.

    Executes a node graph with timeouts, retries, budgets, events, and queryable
    snapshots. Async step functions are rejected at runtime.
    """

    def __init__(self, root: Node, *, defaults: PipelineDefaults | None = None) -> None:
        """
        Create a sync pipeline.

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
    ) -> SyncPipelineBuilder:
        """Return a fluent builder for synchronous pipelines."""

        from agentic_fn.pipeline.builder import SyncPipelineBuilder

        defaults = PipelineDefaults(name=name, **defaults_kwargs)  # type: ignore[arg-type]
        return SyncPipelineBuilder(defaults=defaults)

    def query(self) -> PipelineSnapshot:
        """
        Query the active run snapshot.

        Raises
        ------
        RuntimeError
            If no run is currently active on this pipeline instance.
        """

        if self._active_ctx is None:
            raise RuntimeError("no active sync pipeline run")
        return self._active_ctx.query()

    def run(self, state: State, *, ctx: Context | None = None) -> PipelineResult:
        """
        Execute the pipeline and return a structured result.

        Framework failures are captured into ``PipelineResult`` rather than
        escaping, enabling uniform caller handling in production services.
        """

        context = ctx or Context()
        if state.meta.run_id:
            context.run_id = state.meta.run_id
        else:
            state = state.with_meta(state.meta.with_run_id(context.run_id))

        if self.defaults.on_event is not None:
            context.hooks.add(self.defaults.on_event)
        if self.defaults.run_deadline is not None:
            context.deadline = context.clock.monotonic() + self.defaults.run_deadline

        context.mark_started()
        context.set_state(state)
        self._active_ctx = context
        executor = SyncExecutor(self.defaults)
        started = context.clock.monotonic()

        context.hooks.emit(
            Event(
                kind=EventKind.PIPELINE_START,
                run_id=context.run_id,
                step_name=self.defaults.name,
                state_digest=state.digest(),
            )
        )

        result: PipelineResult
        try:
            output = executor.execute(self.root, state, context)
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
            context.hooks.emit(
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
            context.hooks.emit(
                Event(
                    kind=EventKind.PIPELINE_END,
                    run_id=context.run_id,
                    step_name=self.defaults.name,
                    attributes={"status": context.query().status.value},
                )
            )
            executor.close()
            context.close()
            self._active_ctx = None

        return result

    def _failure_result(
        self,
        context: Context,
        executor: SyncExecutor,
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
        snap = context.query()
        latest = getattr(context, "_state", None)
        state = latest if isinstance(latest, State) else fallback_state
        _ = snap
        return PipelineResult(
            state=state,
            status=status,
            trace=executor.records,
            error=error,
            metrics=executor.metrics(duration_ms),
        )
