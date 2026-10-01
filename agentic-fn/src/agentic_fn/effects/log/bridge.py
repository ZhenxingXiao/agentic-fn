"""Bridge pipeline observability events into application log sinks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.effects.log.errors import EmitFailurePolicy, raise_according_to_policy
from agentic_fn.effects.log.record import LogRecord
from agentic_fn.observe.events import Event
from agentic_fn.type.enums import EventKind, LogLevel
from agentic_fn.type.protocols import LogSink

_DEFAULT_EVENT_LEVELS: dict[EventKind, LogLevel] = {
    EventKind.PIPELINE_START: LogLevel.INFO,
    EventKind.PIPELINE_END: LogLevel.INFO,
    EventKind.STEP_START: LogLevel.DEBUG,
    EventKind.STEP_END: LogLevel.DEBUG,
    EventKind.STEP_RETRY: LogLevel.WARNING,
    EventKind.STEP_ERROR: LogLevel.ERROR,
    EventKind.BRANCH_TAKEN: LogLevel.DEBUG,
    EventKind.LOOP_ITER: LogLevel.DEBUG,
    EventKind.PARALLEL_JOIN: LogLevel.DEBUG,
    EventKind.BUDGET: LogLevel.WARNING,
    EventKind.CANCELLED: LogLevel.WARNING,
}


@dataclass(slots=True)
class EventLogBridge:
    """
    Convert framework ``Event`` callbacks into ``LogRecord`` emissions.

    Attach via ``PipelineDefaults(on_event=bridge)`` or ``ctx.hooks.add(bridge)``.
    """

    sink: LogSink
    level_map: Mapping[EventKind, LogLevel] = field(
        default_factory=lambda: dict(_DEFAULT_EVENT_LEVELS)
    )
    min_level: LogLevel = LogLevel.DEBUG
    logger_name: str = "agentic_fn.events"
    failure_policy: EmitFailurePolicy = EmitFailurePolicy.IGNORE
    include_kinds: frozenset[EventKind] | None = None
    exclude_kinds: frozenset[EventKind] = field(default_factory=frozenset)
    message_fn: Callable[[Event], str] | None = None

    def __call__(self, event: Event) -> None:
        """HookBus-compatible synchronous callback."""

        if event.kind in self.exclude_kinds:
            return
        if self.include_kinds is not None and event.kind not in self.include_kinds:
            return

        level = self.level_map.get(event.kind, LogLevel.INFO)
        if level.rank < self.min_level.rank:
            return

        message = (
            self.message_fn(event)
            if self.message_fn is not None
            else self._default_message(event)
        )
        fields: dict[str, Any] = {
            "event_kind": event.kind.value,
            "attempt": event.attempt,
            "duration_ms": event.duration_ms,
            "error_type": event.error_type,
            **dict(event.attributes),
        }
        record = LogRecord(
            message=message,
            level=level,
            logger_name=self.logger_name,
            run_id=event.run_id,
            path=event.path,
            step_name=event.step_name,
            state_digest=event.state_digest,
            fields=fields,
            source="event_bridge",
            tags=("pipeline_event",),
        )
        try:
            self.sink.emit(record)
        except Exception as exc:  # noqa: BLE001 - policy decides propagation
            raise_according_to_policy(
                self.failure_policy,
                f"event log bridge emit failed: {exc}",
                exc,
            )

    def _default_message(self, event: Event) -> str:
        """Build a compact default message for an event."""

        step = event.step_name or "-"
        if event.error_type:
            return f"{event.kind.value} step={step} error={event.error_type}"
        if event.duration_ms is not None:
            return f"{event.kind.value} step={step} duration_ms={event.duration_ms:.2f}"
        return f"{event.kind.value} step={step}"
