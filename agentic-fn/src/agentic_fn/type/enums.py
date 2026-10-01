"""Public enumerations used across the agentic-fn core."""

from __future__ import annotations

from enum import Enum


class StepKind(str, Enum):
    """Classification of a pipeline step function."""

    CAPABILITY = "capability"
    EFFECT = "effect"


class ErrorAction(str, Enum):
    """How a step failure should be handled by the pipeline engine."""

    FAIL = "fail"
    SKIP = "skip"
    FALLBACK = "fallback"
    CONTINUE = "continue"


class EventKind(str, Enum):
    """Observable lifecycle events emitted during pipeline execution."""

    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"
    STEP_START = "step.start"
    STEP_END = "step.end"
    STEP_RETRY = "step.retry"
    STEP_ERROR = "step.error"
    BRANCH_TAKEN = "branch.taken"
    LOOP_ITER = "loop.iter"
    PARALLEL_JOIN = "parallel.join"
    BUDGET = "budget"
    CANCELLED = "cancelled"


class ReplayMode(str, Enum):
    """Effect execution mode for live runs, recording, or deterministic replay."""

    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class RunStatus(str, Enum):
    """Lifecycle status of a pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConflictMode(str, Enum):
    """Strategy applied when parallel branches write the same state key."""

    ERROR = "error"
    OVERWRITE = "overwrite"
    PREFER_LEFT = "prefer_left"


class CircuitState(str, Enum):
    """Circuit breaker state machine values."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class LogLevel(str, Enum):
    """Ordered severity levels for structured framework logging."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Return a comparable numeric rank (higher means more severe)."""

        return _LOG_LEVEL_RANK[self]


_LOG_LEVEL_RANK: dict[LogLevel, int] = {
    LogLevel.TRACE: 10,
    LogLevel.DEBUG: 20,
    LogLevel.INFO: 30,
    LogLevel.WARNING: 40,
    LogLevel.ERROR: 50,
    LogLevel.CRITICAL: 60,
}
