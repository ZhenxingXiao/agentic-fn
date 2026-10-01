"""Default configuration applied across a pipeline graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_fn.core.retry import RetryPolicy
from agentic_fn.runtime.executor import BoundedExecutor
from agentic_fn.type.aliases import AsyncOnEvent, OnEvent


@dataclass(frozen=True, slots=True)
class PipelineDefaults:
    """
    Pipeline-wide defaults for timeouts, retries, concurrency, and callbacks.

    ``enable_sync_parallel`` defaults to False so sync fan-out cannot silently
    saturate a thread pool in production.
    """

    timeout: float | None = 30.0
    retry: RetryPolicy | None = None
    max_parallel: int = 8
    enable_sync_parallel: bool = False
    executor: BoundedExecutor | None = None
    on_event: OnEvent | None = None
    on_event_async: AsyncOnEvent | None = None
    validate_state: bool = True
    fail_fast: bool = True
    run_deadline: float | None = None
    name: str = "pipeline"
