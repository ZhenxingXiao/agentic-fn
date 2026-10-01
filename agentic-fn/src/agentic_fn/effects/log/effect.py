"""Effect factories that bind log sinks to pipeline-compatible callables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from agentic_fn.core.context import Context
from agentic_fn.core.fn import effect
from agentic_fn.core.state import State
from agentic_fn.effects.log.errors import EmitFailurePolicy, raise_according_to_policy
from agentic_fn.effects.log.formatters import LogFormatter, StaticMessageFormatter
from agentic_fn.effects.log.record import LogRecord
from agentic_fn.type.aliases import AsyncEffectFn, EffectFn
from agentic_fn.type.enums import LogLevel, ReplayMode
from agentic_fn.type.protocols import AsyncLogSink, LogSink


@dataclass(frozen=True, slots=True)
class LogEffectOptions:
    """Tunables for bound log effects."""

    failure_policy: EmitFailurePolicy = EmitFailurePolicy.IGNORE
    skip_on_replay: bool = True
    write_summary_key: str | None = None
    summary_include_message: bool = True
    level_override: LogLevel | None = None
    extra_fields: Mapping[str, Any] | None = None
    idempotent: bool = True


def enrich_record_from_context(
    record: LogRecord,
    state: State,
    ctx: Context,
) -> LogRecord:
    """Attach run metadata from ``Context`` / ``State`` onto a record."""

    snap = ctx.query()
    return record.with_context(
        run_id=record.run_id or ctx.run_id,
        path=record.path or snap.current_path,
        step_name=record.step_name or (snap.current_path[-1] if snap.current_path else None),
        iteration=record.iteration if record.iteration is not None else snap.iteration,
        state_digest=record.state_digest or state.digest(),
    )


def emit_log(
    state: State,
    ctx: Context,
    *,
    sink: LogSink,
    formatter: LogFormatter,
    options: LogEffectOptions,
) -> State:
    """
    Capability/effect split core: format (pure-ish) then emit (IO).

    This function is intentionally explicit about ``sink`` / ``formatter`` so
    callers can bind them with ``partial`` or ``make_log_effect``.
    """

    if options.skip_on_replay and ctx.effects.mode is ReplayMode.REPLAY:
        return state

    record = formatter(state, ctx)
    if record is None:
        return state

    if options.level_override is not None:
        from dataclasses import replace

        record = replace(record, level=options.level_override)
    if options.extra_fields:
        record = record.with_fields(**dict(options.extra_fields))
    record = enrich_record_from_context(record, state, ctx)

    try:
        sink.emit(record)
    except Exception as exc:  # noqa: BLE001 - policy decides propagation
        raise_according_to_policy(
            options.failure_policy,
            f"log effect emit failed: {exc}",
            exc,
        )
        return state

    if options.write_summary_key is None:
        return state

    summary: dict[str, Any] = {
        "record_id": record.record_id,
        "level": record.level.value,
        "logger_name": record.logger_name,
    }
    if options.summary_include_message:
        summary["message"] = record.message
    return state.update(**{options.write_summary_key: summary})


async def emit_log_async(
    state: State,
    ctx: Context,
    *,
    sink: AsyncLogSink,
    formatter: LogFormatter,
    options: LogEffectOptions,
) -> State:
    """Async variant of ``emit_log`` for ``AsyncLogSink`` backends."""

    if options.skip_on_replay and ctx.effects.mode is ReplayMode.REPLAY:
        return state

    record = formatter(state, ctx)
    if record is None:
        return state

    if options.level_override is not None:
        from dataclasses import replace

        record = replace(record, level=options.level_override)
    if options.extra_fields:
        record = record.with_fields(**dict(options.extra_fields))
    record = enrich_record_from_context(record, state, ctx)

    try:
        await sink.emit(record)
    except Exception as exc:  # noqa: BLE001 - policy decides propagation
        raise_according_to_policy(
            options.failure_policy,
            f"async log effect emit failed: {exc}",
            exc,
        )
        return state

    if options.write_summary_key is None:
        return state

    summary: dict[str, Any] = {
        "record_id": record.record_id,
        "level": record.level.value,
        "logger_name": record.logger_name,
    }
    if options.summary_include_message:
        summary["message"] = record.message
    return state.update(**{options.write_summary_key: summary})


def make_log_effect(
    sink: LogSink,
    formatter: LogFormatter | None = None,
    *,
    message: str | None = None,
    level: LogLevel = LogLevel.INFO,
    logger_name: str = "agentic_fn",
    options: LogEffectOptions | None = None,
) -> EffectFn:
    """
    Bind a sink and formatter into a standard ``EffectFn``.

    Examples
    --------
    ``make_log_effect(StdlibLogSink("app"), message="step done")``

    ``make_log_effect(sink, StateKeysFormatter("dump", keys=("a", "b")))``
    """

    resolved_options = options or LogEffectOptions()
    resolved_formatter = formatter
    if resolved_formatter is None:
        if message is None:
            raise ValueError("make_log_effect requires formatter= or message=")
        resolved_formatter = StaticMessageFormatter(
            message=message,
            level=level,
            logger_name=logger_name,
        )

    bound = partial(
        emit_log,
        sink=sink,
        formatter=resolved_formatter,
        options=resolved_options,
    )
    return effect(idempotent=resolved_options.idempotent)(bound)


def make_async_log_effect(
    sink: AsyncLogSink,
    formatter: LogFormatter | None = None,
    *,
    message: str | None = None,
    level: LogLevel = LogLevel.INFO,
    logger_name: str = "agentic_fn",
    options: LogEffectOptions | None = None,
) -> AsyncEffectFn:
    """Bind an async sink and formatter into an ``AsyncEffectFn``."""

    resolved_options = options or LogEffectOptions()
    resolved_formatter = formatter
    if resolved_formatter is None:
        if message is None:
            raise ValueError("make_async_log_effect requires formatter= or message=")
        resolved_formatter = StaticMessageFormatter(
            message=message,
            level=level,
            logger_name=logger_name,
        )

    async def _fn(state: State, ctx: Context) -> State:
        return await emit_log_async(
            state,
            ctx,
            sink=sink,
            formatter=resolved_formatter,
            options=resolved_options,
        )

    return effect(idempotent=resolved_options.idempotent)(_fn)  # type: ignore[return-value]


def bind_sink(
    emit_fn: Callable[..., State],
    sink: LogSink,
    **kwargs: Any,
) -> EffectFn:
    """
    Low-level helper mirroring ``functools.partial`` for custom emit functions.

    Prefer ``make_log_effect`` for common cases.
    """

    return effect(idempotent=True)(partial(emit_fn, sink=sink, **kwargs))
