"""Built-in side-effect building blocks (logging, and future IO adapters)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CallableLogSink",
    "ContextLogger",
    "EmitFailurePolicy",
    "EventLogBridge",
    "LogEffectOptions",
    "LogRecord",
    "MemoryLogSink",
    "NullLogSink",
    "StdlibLogSink",
    "make_log_effect",
    "stdlib_sink",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports to avoid import cycles with ``Context``."""

    if name in {
        "CallableLogSink",
        "MemoryLogSink",
        "NullLogSink",
        "StdlibLogSink",
        "stdlib_sink",
    }:
        from agentic_fn.effects import log as log_pkg

        return getattr(log_pkg, name)
    if name in {
        "ContextLogger",
        "EmitFailurePolicy",
        "EventLogBridge",
        "LogEffectOptions",
        "LogRecord",
        "make_log_effect",
    }:
        from agentic_fn.effects import log as log_pkg

        return getattr(log_pkg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
