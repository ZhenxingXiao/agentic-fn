"""Structured log record model used by all log sinks and formatters."""

from __future__ import annotations

import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from agentic_fn.type.enums import LogLevel


@dataclass(frozen=True, slots=True)
class LogRecord:
    """
    Immutable, serializable unit of application/framework logging.

    This is pure data: constructing a ``LogRecord`` is a capability concern;
    emitting it through a ``LogSink`` is an effect concern.
    """

    message: str
    level: LogLevel = LogLevel.INFO
    logger_name: str = "agentic_fn"
    timestamp: float = field(default_factory=time.time)
    record_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str | None = None
    path: tuple[str, ...] = ()
    step_name: str | None = None
    iteration: int | None = None
    state_digest: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    exception_type: str | None = None
    exception_message: str | None = None
    exception_stack: str | None = None
    source: str | None = None

    def with_fields(self, **fields: Any) -> LogRecord:
        """Return a copy with merged structured fields."""

        merged = dict(self.fields)
        merged.update(fields)
        return replace(self, fields=merged)

    def with_context(
        self,
        *,
        run_id: str | None = None,
        path: tuple[str, ...] | None = None,
        step_name: str | None = None,
        iteration: int | None = None,
        state_digest: str | None = None,
    ) -> LogRecord:
        """Return a copy with pipeline context metadata filled in."""

        return replace(
            self,
            run_id=self.run_id if run_id is None else run_id,
            path=self.path if path is None else path,
            step_name=self.step_name if step_name is None else step_name,
            iteration=self.iteration if iteration is None else iteration,
            state_digest=self.state_digest if state_digest is None else state_digest,
        )

    def with_exception(self, exc: BaseException, *, include_stack: bool = True) -> LogRecord:
        """Return a copy enriched with exception details."""

        stack = None
        if include_stack:
            stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return replace(
            self,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            exception_stack=stack,
            level=self.level if self.level.rank >= LogLevel.ERROR.rank else LogLevel.ERROR,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly mapping."""

        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "logger_name": self.logger_name,
            "message": self.message,
            "run_id": self.run_id,
            "path": list(self.path),
            "step_name": self.step_name,
            "iteration": self.iteration,
            "state_digest": self.state_digest,
            "fields": dict(self.fields),
            "tags": list(self.tags),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "exception_stack": self.exception_stack,
            "source": self.source,
        }

    @classmethod
    def from_exception(
        cls,
        message: str,
        exc: BaseException,
        *,
        level: LogLevel = LogLevel.ERROR,
        logger_name: str = "agentic_fn",
        include_stack: bool = True,
        **fields: Any,
    ) -> LogRecord:
        """Build an error record directly from an exception."""

        return cls(
            message=message,
            level=level,
            logger_name=logger_name,
            fields=fields,
        ).with_exception(exc, include_stack=include_stack)
