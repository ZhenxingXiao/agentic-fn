"""Observability events emitted during pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_fn.type.enums import EventKind


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable structured event for callbacks and audit logs."""

    kind: EventKind
    run_id: str
    step_name: str | None = None
    attempt: int | None = None
    duration_ms: float | None = None
    error_type: str | None = None
    state_digest: str | None = None
    path: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
