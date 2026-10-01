"""Industrial-grade pluggable logging effects for agentic-fn."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AsyncioLogSink",
    "BufferedLogSink",
    "CallableLogSink",
    "CompositeFormatter",
    "ConditionalFormatter",
    "ContextLogger",
    "EmitFailurePolicy",
    "EventLogBridge",
    "FilteredLogSink",
    "JsonStdlibLogSink",
    "LogConfigurationError",
    "LogEffectOptions",
    "LogEmitError",
    "LogFormatter",
    "LogRecord",
    "MemoryLogSink",
    "MultiplexLogSink",
    "NullLogSink",
    "RateLimitedLogSink",
    "RedactingLogSink",
    "RedactionPolicy",
    "SamplingLogSink",
    "StateKeysFormatter",
    "StaticMessageFormatter",
    "StdlibLogSink",
    "TemplateFormatter",
    "apply_redaction_to_record_fields",
    "bind_formatter_fields",
    "bind_sink",
    "compose_sinks",
    "default_redaction_policy",
    "emit_log",
    "emit_log_async",
    "enrich_record_from_context",
    "make_async_log_effect",
    "make_log_effect",
    "stdlib_sink",
]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "LogRecord": (".record", "LogRecord"),
    "EmitFailurePolicy": (".errors", "EmitFailurePolicy"),
    "LogConfigurationError": (".errors", "LogConfigurationError"),
    "LogEmitError": (".errors", "LogEmitError"),
    "RedactionPolicy": (".redact", "RedactionPolicy"),
    "apply_redaction_to_record_fields": (".redact", "apply_redaction_to_record_fields"),
    "default_redaction_policy": (".redact", "default_redaction_policy"),
    "AsyncioLogSink": (".sinks", "AsyncioLogSink"),
    "BufferedLogSink": (".sinks", "BufferedLogSink"),
    "CallableLogSink": (".sinks", "CallableLogSink"),
    "FilteredLogSink": (".sinks", "FilteredLogSink"),
    "JsonStdlibLogSink": (".sinks", "JsonStdlibLogSink"),
    "MemoryLogSink": (".sinks", "MemoryLogSink"),
    "MultiplexLogSink": (".sinks", "MultiplexLogSink"),
    "NullLogSink": (".sinks", "NullLogSink"),
    "RateLimitedLogSink": (".sinks", "RateLimitedLogSink"),
    "RedactingLogSink": (".sinks", "RedactingLogSink"),
    "SamplingLogSink": (".sinks", "SamplingLogSink"),
    "StdlibLogSink": (".sinks", "StdlibLogSink"),
    "compose_sinks": (".sinks", "compose_sinks"),
    "stdlib_sink": (".sinks", "stdlib_sink"),
    "CompositeFormatter": (".formatters", "CompositeFormatter"),
    "ConditionalFormatter": (".formatters", "ConditionalFormatter"),
    "LogFormatter": (".formatters", "LogFormatter"),
    "StateKeysFormatter": (".formatters", "StateKeysFormatter"),
    "StaticMessageFormatter": (".formatters", "StaticMessageFormatter"),
    "TemplateFormatter": (".formatters", "TemplateFormatter"),
    "bind_formatter_fields": (".formatters", "bind_formatter_fields"),
    "ContextLogger": (".facade", "ContextLogger"),
    "LogEffectOptions": (".effect", "LogEffectOptions"),
    "bind_sink": (".effect", "bind_sink"),
    "emit_log": (".effect", "emit_log"),
    "emit_log_async": (".effect", "emit_log_async"),
    "enrich_record_from_context": (".effect", "enrich_record_from_context"),
    "make_async_log_effect": (".effect", "make_async_log_effect"),
    "make_log_effect": (".effect", "make_log_effect"),
    "EventLogBridge": (".bridge", "EventLogBridge"),
}


def __getattr__(name: str) -> Any:
    """Lazily import submodules to keep ``Context`` import-safe."""

    if name not in _LAZY_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_MAP[name]
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
