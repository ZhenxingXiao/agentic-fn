"""Pipeline run results and step trace records."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_fn.core.errors import FrameworkError
from agentic_fn.core.state import State
from agentic_fn.type.enums import RunStatus, StepKind


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Trace record for a single step attempt outcome."""

    name: str
    kind: StepKind
    attempt: int
    duration_ms: float
    success: bool
    error_type: str | None = None
    state_digest: str | None = None


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Aggregate metrics collected for one pipeline run."""

    duration_ms: float = 0.0
    step_count: int = 0
    retry_count: int = 0
    effect_count: int = 0


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Final outcome returned by ``Pipeline.run`` / ``async run``."""

    state: State
    status: RunStatus
    trace: tuple[StepRecord, ...] = ()
    error: FrameworkError | None = None
    metrics: RunMetrics = field(default_factory=RunMetrics)

    @property
    def ok(self) -> bool:
        """Return True when the run completed successfully."""

        return self.status is RunStatus.SUCCEEDED
