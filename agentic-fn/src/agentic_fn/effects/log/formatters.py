"""Pure log formatters (capability side) that build ``LogRecord`` values."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentic_fn.core.context import Context
from agentic_fn.core.state import State
from agentic_fn.effects.log.record import LogRecord
from agentic_fn.type.enums import LogLevel

LogFormatter = Callable[[State, Context], LogRecord | None]


@dataclass(frozen=True, slots=True)
class StaticMessageFormatter:
    """Always emit a fixed message with optional static fields."""

    message: str
    level: LogLevel = LogLevel.INFO
    logger_name: str = "agentic_fn"
    fields: Mapping[str, Any] | None = None
    tags: tuple[str, ...] = ()
    source: str = "static"

    def __call__(self, state: State, ctx: Context) -> LogRecord:
        """Build a context-enriched static record."""

        snap = ctx.query()
        return LogRecord(
            message=self.message,
            level=self.level,
            logger_name=self.logger_name,
            fields=dict(self.fields or {}),
            tags=self.tags,
            source=self.source,
        ).with_context(
            run_id=ctx.run_id,
            path=snap.current_path,
            iteration=snap.iteration,
            state_digest=state.digest(),
        )


@dataclass(frozen=True, slots=True)
class StateKeysFormatter:
    """
    Emit selected state keys as structured fields.

    Missing keys are omitted unless ``require_keys`` is True.
    """

    message: str
    keys: Sequence[str]
    level: LogLevel = LogLevel.INFO
    logger_name: str = "agentic_fn"
    require_keys: bool = False
    include_digest: bool = True
    source: str = "state_keys"

    def __call__(self, state: State, ctx: Context) -> LogRecord | None:
        """Extract configured keys from state into a log record."""

        fields: dict[str, Any] = {}
        for key in self.keys:
            if key in state.data:
                fields[key] = state.data[key]
            elif self.require_keys:
                raise KeyError(f"required log state key missing: {key!r}")
        snap = ctx.query()
        return LogRecord(
            message=self.message,
            level=self.level,
            logger_name=self.logger_name,
            fields=fields,
            source=self.source,
            state_digest=state.digest() if self.include_digest else None,
        ).with_context(
            run_id=ctx.run_id,
            path=snap.current_path,
            iteration=snap.iteration,
        )


@dataclass(frozen=True, slots=True)
class TemplateFormatter:
    """
    Format ``message_template`` using ``str.format_map`` against state data.

    Extra static fields may be merged in. Returns ``None`` when formatting fails
    and ``strict`` is False.
    """

    message_template: str
    level: LogLevel = LogLevel.INFO
    logger_name: str = "agentic_fn"
    extra_fields: Mapping[str, Any] | None = None
    strict: bool = False
    source: str = "template"

    def __call__(self, state: State, ctx: Context) -> LogRecord | None:
        """Render a template message from state values."""

        values = dict(state.data)
        values.setdefault("run_id", ctx.run_id)
        try:
            message = self.message_template.format_map(_Default(values))
        except KeyError:
            if self.strict:
                raise
            return None
        snap = ctx.query()
        return LogRecord(
            message=message,
            level=self.level,
            logger_name=self.logger_name,
            fields=dict(self.extra_fields or {}),
            source=self.source,
            state_digest=state.digest(),
        ).with_context(
            run_id=ctx.run_id,
            path=snap.current_path,
            iteration=snap.iteration,
        )


@dataclass(frozen=True, slots=True)
class ConditionalFormatter:
    """Wrap another formatter and only emit when ``predicate`` is true."""

    formatter: LogFormatter
    predicate: Callable[[State, Context], bool]

    def __call__(self, state: State, ctx: Context) -> LogRecord | None:
        """Delegate to the inner formatter when the predicate matches."""

        if not self.predicate(state, ctx):
            return None
        return self.formatter(state, ctx)


@dataclass(frozen=True, slots=True)
class CompositeFormatter:
    """
    Run multiple formatters and return the first non-``None`` record.

    Useful for fallback messaging strategies.
    """

    formatters: tuple[LogFormatter, ...]

    def __call__(self, state: State, ctx: Context) -> LogRecord | None:
        """Return the first produced record."""

        for formatter in self.formatters:
            record = formatter(state, ctx)
            if record is not None:
                return record
        return None


class _Default(dict[str, Any]):
    """Mapping that returns ``{key}`` placeholders for missing template keys."""

    def __missing__(self, key: str) -> str:
        """Return an unresolved placeholder token."""

        return "{" + key + "}"


def bind_formatter_fields(
    formatter: LogFormatter,
    **fields: Any,
) -> LogFormatter:
    """Return a formatter that merges extra fields into every produced record."""

    def wrapped(state: State, ctx: Context) -> LogRecord | None:
        record = formatter(state, ctx)
        if record is None:
            return None
        return record.with_fields(**fields)

    return wrapped
