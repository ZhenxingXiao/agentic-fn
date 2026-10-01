"""Context-bound logger facade for ergonomic effect-side logging."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.core.context import Context
from agentic_fn.core.state import State
from agentic_fn.effects.log.errors import EmitFailurePolicy, raise_according_to_policy
from agentic_fn.effects.log.record import LogRecord
from agentic_fn.type.enums import LogLevel, ReplayMode
from agentic_fn.type.protocols import LogSink


@dataclass(slots=True)
class ContextLogger:
    """
    Ergonomic logger bound to a ``Context`` and a pluggable ``LogSink``.

    Intended for use inside Effect functions and infrastructure code — not
    inside pure Capability functions.
    """

    sink: LogSink
    ctx: Context
    logger_name: str = "agentic_fn"
    bound_fields: dict[str, Any] = field(default_factory=dict)
    failure_policy: EmitFailurePolicy = EmitFailurePolicy.IGNORE
    skip_on_replay: bool = True

    def bind(self, **fields: Any) -> ContextLogger:
        """Return a child logger with additional bound fields."""

        merged = dict(self.bound_fields)
        merged.update(fields)
        return ContextLogger(
            sink=self.sink,
            ctx=self.ctx,
            logger_name=self.logger_name,
            bound_fields=merged,
            failure_policy=self.failure_policy,
            skip_on_replay=self.skip_on_replay,
        )

    def trace(self, message: str, **fields: Any) -> None:
        """Emit a TRACE record."""

        self.log(LogLevel.TRACE, message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        """Emit a DEBUG record."""

        self.log(LogLevel.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        """Emit an INFO record."""

        self.log(LogLevel.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        """Emit a WARNING record."""

        self.log(LogLevel.WARNING, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        """Emit an ERROR record."""

        self.log(LogLevel.ERROR, message, **fields)

    def critical(self, message: str, **fields: Any) -> None:
        """Emit a CRITICAL record."""

        self.log(LogLevel.CRITICAL, message, **fields)

    def exception(self, message: str, exc: BaseException, **fields: Any) -> None:
        """Emit an ERROR record enriched with exception details."""

        record = self._base_record(LogLevel.ERROR, message, fields).with_exception(exc)
        self._emit(record)

    def log(self, level: LogLevel, message: str, **fields: Any) -> None:
        """Emit a record at the given level."""

        self._emit(self._base_record(level, message, fields))

    def log_state(
        self,
        message: str,
        state: State,
        *,
        level: LogLevel = LogLevel.INFO,
        include_keys: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        """Emit a record that includes a state digest and optional key subset."""

        payload = dict(self.bound_fields)
        payload.update(fields)
        if include_keys is not None:
            payload.update(dict(include_keys))
        snap = self.ctx.query()
        record = LogRecord(
            message=message,
            level=level,
            logger_name=self.logger_name,
            fields=payload,
            state_digest=state.digest(),
            source="context_logger",
        ).with_context(
            run_id=self.ctx.run_id,
            path=snap.current_path,
            iteration=snap.iteration,
        )
        self._emit(record)

    def _base_record(
        self,
        level: LogLevel,
        message: str,
        fields: Mapping[str, Any],
    ) -> LogRecord:
        """Build a context-enriched record."""

        payload = dict(self.bound_fields)
        payload.update(fields)
        snap = self.ctx.query()
        return LogRecord(
            message=message,
            level=level,
            logger_name=self.logger_name,
            fields=payload,
            source="context_logger",
        ).with_context(
            run_id=self.ctx.run_id,
            path=snap.current_path,
            step_name=snap.current_path[-1] if snap.current_path else None,
            iteration=snap.iteration,
            state_digest=snap.state_digest or None,
        )

    def _emit(self, record: LogRecord) -> None:
        """Emit through the sink, honoring replay skip and failure policy."""

        if self.skip_on_replay and self.ctx.effects.mode is ReplayMode.REPLAY:
            return
        try:
            self.sink.emit(record)
        except Exception as exc:  # noqa: BLE001 - policy decides propagation
            raise_according_to_policy(
                self.failure_policy,
                f"context logger emit failed: {exc}",
                exc,
            )
