"""Read-only pipeline snapshots for query APIs."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_fn.runtime.budget import BudgetView
from agentic_fn.type.enums import RunStatus


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Point-in-time view of an in-flight or finished pipeline run."""

    run_id: str
    status: RunStatus
    current_path: tuple[str, ...]
    iteration: int | None
    state_digest: str
    started_at: float
    budget_remaining: BudgetView
    last_error: str | None = None
