"""Concrete ``LogSink`` implementations for production and testing."""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.effects.log.errors import LogConfigurationError, LogEmitError
from agentic_fn.effects.log.record import LogRecord
from agentic_fn.effects.log.redact import RedactionPolicy, apply_redaction_to_record_fields
from agentic_fn.type.enums import LogLevel
from agentic_fn.type.protocols import AsyncLogSink, LogSink

_STDLIB_LEVEL: dict[LogLevel, int] = {
    LogLevel.TRACE: logging.DEBUG - 5,
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


@dataclass(slots=True)
class NullLogSink:
    """Discard all records. Safe default for frameworks and unit tests."""

    def emit(self, record: LogRecord) -> None:
        """Drop the record."""

        _ = record


@dataclass(slots=True)
class MemoryLogSink:
    """
    In-memory sink for tests, harnesses, and local replay inspection.

    Thread-safe within a process. Optionally bounds retained history.
    """

    max_records: int | None = 10_000
    records: MutableSequence[LogRecord] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def emit(self, record: LogRecord) -> None:
        """Append ``record`` to the in-memory buffer."""

        with self._lock:
            self.records.append(record)
            if self.max_records is not None and len(self.records) > self.max_records:
                overflow = len(self.records) - self.max_records
                del self.records[0:overflow]

    def clear(self) -> None:
        """Remove all retained records."""

        with self._lock:
            self.records.clear()

    def snapshot(self) -> list[LogRecord]:
        """Return a copy of retained records."""

        with self._lock:
            return list(self.records)


@dataclass(slots=True)
class CallableLogSink:
    """
    Adapt any callable into a ``LogSink``.

    Useful for loguru / structlog / proprietary SDKs without first-class adapters:

    ``CallableLogSink(lambda r: logger.bind(**r.fields).log(r.level.value, r.message))``
    """

    fn: Callable[[LogRecord], None]

    def emit(self, record: LogRecord) -> None:
        """Forward the record to the wrapped callable."""

        try:
            self.fn(record)
        except Exception as exc:  # noqa: BLE001 - normalize sink failures
            raise LogEmitError(f"callable log sink failed: {exc}", cause=exc) from exc


@dataclass(slots=True)
class StdlibLogSink:
    """
    Adapter over the Python standard-library ``logging`` package.

    Structured fields are attached via ``extra`` under ``agentic_fn`` to avoid
    colliding with reserved ``LogRecord`` attributes on stdlib records.
    """

    logger_name: str = "agentic_fn"
    include_exception_stack: bool = True
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the stdlib logger once."""

        self._logger = logging.getLogger(self.logger_name)

    def emit(self, record: LogRecord) -> None:
        """Emit through stdlib logging."""

        level = _STDLIB_LEVEL.get(record.level, logging.INFO)
        extra = {
            "agentic_fn": {
                "record_id": record.record_id,
                "run_id": record.run_id,
                "path": list(record.path),
                "step_name": record.step_name,
                "iteration": record.iteration,
                "state_digest": record.state_digest,
                "fields": dict(record.fields),
                "tags": list(record.tags),
                "source": record.source,
            }
        }
        try:
            if record.exception_type and self.include_exception_stack:
                self._logger.log(
                    level,
                    "%s | exc=%s: %s",
                    record.message,
                    record.exception_type,
                    record.exception_message,
                    extra=extra,
                )
                if record.exception_stack:
                    self._logger.log(level, record.exception_stack, extra=extra)
            else:
                self._logger.log(level, record.message, extra=extra)
        except Exception as exc:  # noqa: BLE001 - normalize sink failures
            raise LogEmitError(f"stdlib log sink failed: {exc}", cause=exc) from exc


@dataclass(slots=True)
class JsonStdlibLogSink:
    """Emit one JSON object per line through stdlib logging (structured ops style)."""

    logger_name: str = "agentic_fn"
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the stdlib logger once."""

        self._logger = logging.getLogger(self.logger_name)

    def emit(self, record: LogRecord) -> None:
        """Serialize the record as JSON and log it as the message body."""

        level = _STDLIB_LEVEL.get(record.level, logging.INFO)
        try:
            payload = json.dumps(record.to_dict(), ensure_ascii=False, default=str)
            self._logger.log(level, payload)
        except Exception as exc:  # noqa: BLE001 - normalize sink failures
            raise LogEmitError(f"json stdlib log sink failed: {exc}", cause=exc) from exc


@dataclass(slots=True)
class MultiplexLogSink:
    """
    Fan-out sink that emits to multiple downstream sinks.

    Downstream failures are isolated unless ``fail_fast`` is enabled.
    """

    sinks: Sequence[LogSink]
    fail_fast: bool = False
    on_error: Callable[[LogSink, LogRecord, BaseException], None] | None = None

    def emit(self, record: LogRecord) -> None:
        """Emit to every configured sink."""

        if not self.sinks:
            raise LogConfigurationError("MultiplexLogSink requires at least one sink")
        errors: list[BaseException] = []
        for sink in self.sinks:
            try:
                sink.emit(record)
            except Exception as exc:  # noqa: BLE001 - isolate fan-out failures
                errors.append(exc)
                if self.on_error is not None:
                    self.on_error(sink, record, exc)
                if self.fail_fast:
                    raise LogEmitError(
                        f"multiplex sink failed fast: {exc}",
                        cause=exc,
                    ) from exc
        if errors and self.fail_fast is False and len(errors) == len(self.sinks):
            raise LogEmitError(
                f"all multiplex sinks failed ({len(errors)})",
                cause=errors[-1],
            )


@dataclass(slots=True)
class FilteredLogSink:
    """Filter records by minimum level and/or a custom predicate before emit."""

    sink: LogSink
    min_level: LogLevel = LogLevel.INFO
    predicate: Callable[[LogRecord], bool] | None = None

    def emit(self, record: LogRecord) -> None:
        """Forward the record when it passes configured filters."""

        if record.level.rank < self.min_level.rank:
            return
        if self.predicate is not None and not self.predicate(record):
            return
        self.sink.emit(record)


@dataclass(slots=True)
class SamplingLogSink:
    """
    Probabilistically sample records at or below a given level.

    Records at ``always_emit_at_or_above`` always pass (defaults to ERROR).
    """

    sink: LogSink
    sample_rate: float = 1.0
    sample_below: LogLevel = LogLevel.INFO
    always_emit_at_or_above: LogLevel = LogLevel.ERROR
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        """Validate sampling configuration."""

        if not 0.0 <= self.sample_rate <= 1.0:
            raise LogConfigurationError("sample_rate must be in [0.0, 1.0]")

    def emit(self, record: LogRecord) -> None:
        """Emit according to sampling rules."""

        if record.level.rank >= self.always_emit_at_or_above.rank:
            self.sink.emit(record)
            return
        if record.level.rank <= self.sample_below.rank:
            if self.rng.random() > self.sample_rate:
                return
        self.sink.emit(record)


@dataclass(slots=True)
class RateLimitedLogSink:
    """
    Token-bucket style rate limiter for noisy logs.

    Allows ``rate`` emits per ``per_seconds`` window, with optional burst size.
    Dropped records are counted but not emitted.
    """

    sink: LogSink
    rate: float = 100.0
    per_seconds: float = 1.0
    burst: float | None = None
    _tokens: float = field(init=False, default=0.0)
    _updated_at: float = field(init=False, default_factory=time.monotonic)
    _dropped: int = field(init=False, default=0)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Initialize token bucket capacity."""

        if self.rate <= 0 or self.per_seconds <= 0:
            raise LogConfigurationError("rate and per_seconds must be positive")
        capacity = self.burst if self.burst is not None else self.rate
        if capacity <= 0:
            raise LogConfigurationError("burst must be positive")
        self._tokens = float(capacity)

    @property
    def dropped(self) -> int:
        """Number of records dropped by rate limiting."""

        with self._lock:
            return self._dropped

    def emit(self, record: LogRecord) -> None:
        """Emit when tokens are available; otherwise drop."""

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._updated_at
            self._updated_at = now
            capacity = self.burst if self.burst is not None else self.rate
            self._tokens = min(capacity, self._tokens + elapsed * (self.rate / self.per_seconds))
            if self._tokens < 1.0:
                self._dropped += 1
                return
            self._tokens -= 1.0
        self.sink.emit(record)


@dataclass(slots=True)
class RedactingLogSink:
    """Wrap another sink and redact sensitive fields before emit."""

    sink: LogSink
    policy: RedactionPolicy = field(default_factory=RedactionPolicy)

    def emit(self, record: LogRecord) -> None:
        """Emit a copy of ``record`` with redacted structured fields."""

        redacted_fields = apply_redaction_to_record_fields(record.fields, self.policy)
        # Reconstruct via to_dict/from fields to preserve frozen record semantics.
        safe = LogRecord(
            message=record.message,
            level=record.level,
            logger_name=record.logger_name,
            timestamp=record.timestamp,
            record_id=record.record_id,
            run_id=record.run_id,
            path=record.path,
            step_name=record.step_name,
            iteration=record.iteration,
            state_digest=record.state_digest,
            fields=redacted_fields,
            tags=record.tags,
            exception_type=record.exception_type,
            exception_message=record.exception_message,
            exception_stack=record.exception_stack,
            source=record.source,
        )
        self.sink.emit(safe)


@dataclass(slots=True)
class BufferedLogSink:
    """
    Buffer records and flush in batches.

    Call ``flush()`` explicitly, or rely on ``flush_interval`` / ``max_buffer``
    triggers during ``emit``.
    """

    sink: LogSink
    max_buffer: int = 100
    flush_interval: float | None = 1.0
    _buffer: deque[LogRecord] = field(default_factory=deque, repr=False)
    _last_flush: float = field(default_factory=time.monotonic, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def emit(self, record: LogRecord) -> None:
        """Buffer a record and flush when thresholds are met."""

        with self._lock:
            self._buffer.append(record)
            should_flush = len(self._buffer) >= self.max_buffer
            if self.flush_interval is not None:
                should_flush = should_flush or (
                    time.monotonic() - self._last_flush >= self.flush_interval
                )
            if should_flush:
                self._flush_unlocked()

    def flush(self) -> None:
        """Flush all buffered records to the downstream sink."""

        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Flush holding the internal lock."""

        while self._buffer:
            item = self._buffer.popleft()
            self.sink.emit(item)
        self._last_flush = time.monotonic()


@dataclass(slots=True)
class AsyncioLogSink:
    """
    Bridge an ``AsyncLogSink`` into sync ``LogSink.emit`` via a runner callback.

    Production services typically inject an event-loop aware runner; tests can
    pass ``asyncio.run`` for simple cases (not safe inside a running loop).
    """

    sink: AsyncLogSink
    runner: Callable[[Any], Any]

    def emit(self, record: LogRecord) -> None:
        """Run the async sink to completion using the configured runner."""

        try:
            self.runner(self.sink.emit(record))
        except Exception as exc:  # noqa: BLE001 - normalize sink failures
            raise LogEmitError(f"async log sink bridge failed: {exc}", cause=exc) from exc


def compose_sinks(*sinks: LogSink, fail_fast: bool = False) -> LogSink:
    """Compose multiple sinks into a multiplex sink."""

    return MultiplexLogSink(sinks=tuple(sinks), fail_fast=fail_fast)


def stdlib_sink(name: str = "agentic_fn", *, json_lines: bool = False) -> LogSink:
    """Factory for the most common stdlib-backed sinks."""

    if json_lines:
        return JsonStdlibLogSink(logger_name=name)
    return StdlibLogSink(logger_name=name)
