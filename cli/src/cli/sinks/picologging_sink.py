"""
Picologging adapter for agentic-fn ``LogSink``.

agentic-fn keeps logging vendor-agnostic: pipeline effects emit ``LogRecord``
values into a ``LogSink``. This module bridges those records to Microsoft's
``picologging`` (a high-performance, stdlib-compatible logger).

Why not reuse ``StdlibLogSink``?
- picologging is a separate package with its own ``getLogger`` tree
- picologging historically mishandles logging ``extra=``; we therefore fold
  structured fields into the message body instead of relying on ``extra``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import picologging
import picologging.handlers

from agentic_fn.effects.log.errors import LogEmitError
from agentic_fn.effects.log.record import LogRecord
from agentic_fn.type.enums import LogLevel

# Map framework levels onto picologging's stdlib-compatible integers.
# TRACE has no dedicated level in picologging; we drop it to DEBUG.
_PICO_LEVEL: dict[LogLevel, int] = {
    LogLevel.TRACE: picologging.DEBUG,
    LogLevel.DEBUG: picologging.DEBUG,
    LogLevel.INFO: picologging.INFO,
    LogLevel.WARNING: picologging.WARNING,
    LogLevel.ERROR: picologging.ERROR,
    LogLevel.CRITICAL: picologging.CRITICAL,
}


def configure_picologging(
    *,
    level: int = picologging.DEBUG,
    # Keep the format simple: picologging does not fully honor stdlib width
    # specs (e.g. ``%(levelname)-8s``) and ``%(asctime)s`` can mis-render on
    # some platforms. Prefer ``{...}`` style for predictable demo output.
    format: str = "{levelname} | {name} | {message}",
    style: str = "{",
    stream: Any | None = None,
    log_file: str | Path | None = None,
    max_bytes: int = 1_048_576,
    backup_count: int = 3,
) -> Path | None:
    """
    One-shot logging configuration for demos / scripts.

    Console output always goes to ``stream`` (stderr by default). When
    ``log_file`` is set, a ``RotatingFileHandler`` is also attached to the
    root logger: files roll at ``max_bytes`` and keep ``backup_count`` backups.

    ``force=True`` replaces any existing root handlers so repeated demo runs
    in the same process do not stack duplicate handlers.
    """

    picologging.basicConfig(
        level=level,
        format=format,
        style=style,
        stream=stream or sys.stderr,
        force=True,
    )
    if log_file is None:
        return None

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = picologging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(picologging.Formatter(format, style=style))
    picologging.getLogger().addHandler(file_handler)
    return path


@dataclass(slots=True)
class PicologgingLogSink:
    """
    ``LogSink`` that forwards agentic-fn records into picologging.

    Usage inside a pipeline::

        sink = PicologgingLogSink(logger_name="agentic_fn.demo")
        log_fx = make_log_effect(sink, message="step done")
        SyncPipeline.builder("x").effect("log", log_fx, idempotent=True).build()
    """

    logger_name: str = "agentic_fn"
    include_structured_fields: bool = True
    include_run_context: bool = True
    _logger: picologging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the picologging logger once per sink instance."""

        self._logger = picologging.getLogger(self.logger_name)

    def emit(self, record: LogRecord) -> None:
        """Translate and emit one framework ``LogRecord``."""

        level = _PICO_LEVEL.get(record.level, picologging.INFO)
        message = self._format_message(record)
        try:
            self._logger.log(level, message)
        except Exception as exc:  # noqa: BLE001 - normalize sink failures
            raise LogEmitError(f"picologging sink failed: {exc}", cause=exc) from exc

    def _format_message(self, record: LogRecord) -> str:
        """
        Build a single-line message.

        Structured metadata is appended as ``key=value`` pairs so demos remain
        readable without depending on picologging ``extra`` support.
        """

        parts: list[str] = [record.message]

        if self.include_run_context:
            if record.run_id:
                parts.append(f"run_id={record.run_id}")
            if record.step_name:
                parts.append(f"step={record.step_name}")
            if record.path:
                parts.append(f"path={'/'.join(record.path)}")
            if record.iteration is not None:
                parts.append(f"iter={record.iteration}")
            if record.state_digest:
                # Digest can be long; keep a short prefix for terminal demos.
                parts.append(f"digest={record.state_digest[:12]}")

        if self.include_structured_fields and record.fields:
            for key, value in record.fields.items():
                parts.append(f"{key}={value!r}")

        if record.exception_type:
            parts.append(f"exc={record.exception_type}:{record.exception_message}")

        if len(parts) == 1:
            return parts[0]
        return parts[0] + " | " + " ".join(parts[1:])
