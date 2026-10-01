"""Industrial-grade tests for the pluggable logging subsystem."""

from __future__ import annotations

import unittest
from typing import Any

from agentic_fn import (
    Context,
    EffectRuntime,
    EmitFailurePolicy,
    Event,
    EventKind,
    EventLogBridge,
    LogEffectOptions,
    LogLevel,
    LogRecord,
    MemoryLogSink,
    ReplayMode,
    State,
    SyncPipeline,
    make_log_effect,
)
from agentic_fn.core.errors import PermanentError, TransientError
from agentic_fn.effects.log.formatters import (
    StateKeysFormatter,
    StaticMessageFormatter,
    TemplateFormatter,
)
from agentic_fn.effects.log.redact import RedactionPolicy, default_redaction_policy
from agentic_fn.effects.log.sinks import (
    CallableLogSink,
    FilteredLogSink,
    MultiplexLogSink,
    RateLimitedLogSink,
    RedactingLogSink,
    SamplingLogSink,
)


class LogSinkTests(unittest.TestCase):
    """Sink adapter and composition behavior."""

    def test_memory_and_callable_sink(self) -> None:
        """Callable sinks should forward records to user callables."""

        seen: list[str] = []
        sink = CallableLogSink(lambda r: seen.append(r.message))
        sink.emit(LogRecord(message="hello", level=LogLevel.INFO))
        self.assertEqual(seen, ["hello"])

    def test_filtered_sink(self) -> None:
        """Filtered sinks should drop records below the minimum level."""

        mem = MemoryLogSink()
        sink = FilteredLogSink(sink=mem, min_level=LogLevel.WARNING)
        sink.emit(LogRecord(message="dbg", level=LogLevel.DEBUG))
        sink.emit(LogRecord(message="warn", level=LogLevel.WARNING))
        self.assertEqual(len(mem.records), 1)
        self.assertEqual(mem.records[0].message, "warn")

    def test_multiplex_isolates_failures(self) -> None:
        """Multiplex should continue when fail_fast is disabled and one sink works."""

        ok = MemoryLogSink()

        def boom(_record: LogRecord) -> None:
            raise RuntimeError("nope")

        sink = MultiplexLogSink(sinks=(CallableLogSink(boom), ok), fail_fast=False)
        sink.emit(LogRecord(message="x"))
        self.assertEqual(len(ok.records), 1)

    def test_redacting_sink_masks_secrets(self) -> None:
        """Redacting sink should mask sensitive keys before forwarding."""

        mem = MemoryLogSink()
        sink = RedactingLogSink(sink=mem, policy=default_redaction_policy())
        sink.emit(
            LogRecord(
                message="auth",
                fields={"api_key": "secret-value", "user": "alice"},
            )
        )
        self.assertEqual(mem.records[0].fields["api_key"], "***REDACTED***")
        self.assertEqual(mem.records[0].fields["user"], "alice")

    def test_sampling_always_emits_errors(self) -> None:
        """Sampling must not drop error-level records."""

        mem = MemoryLogSink()
        sink = SamplingLogSink(sink=mem, sample_rate=0.0, sample_below=LogLevel.INFO)
        sink.emit(LogRecord(message="info", level=LogLevel.INFO))
        sink.emit(LogRecord(message="err", level=LogLevel.ERROR))
        self.assertEqual(len(mem.records), 1)
        self.assertEqual(mem.records[0].message, "err")

    def test_rate_limited_sink_drops_burst(self) -> None:
        """Rate limiter should drop once tokens are exhausted."""

        mem = MemoryLogSink()
        sink = RateLimitedLogSink(sink=mem, rate=1, per_seconds=60, burst=1)
        sink.emit(LogRecord(message="a"))
        sink.emit(LogRecord(message="b"))
        self.assertEqual(len(mem.records), 1)
        self.assertGreaterEqual(sink.dropped, 1)


class LogEffectTests(unittest.TestCase):
    """Pipeline log effect binding and policies."""

    def test_make_log_effect_in_pipeline(self) -> None:
        """Bound log effects should emit during pipeline execution."""

        mem = MemoryLogSink()
        log_fn = make_log_effect(
            mem,
            StateKeysFormatter("state dump", keys=("value",)),
            options=LogEffectOptions(write_summary_key="_last_log"),
        )
        pipe = SyncPipeline.builder("with_log").effect("log", log_fn, idempotent=True).build()
        result = pipe.run(State.empty().update(value=42))
        self.assertTrue(result.ok)
        self.assertEqual(len(mem.records), 1)
        self.assertEqual(mem.records[0].fields["value"], 42)
        self.assertIn("_last_log", result.state.data)

    def test_template_formatter(self) -> None:
        """Template formatter should render state fields into messages."""

        mem = MemoryLogSink()
        log_fn = make_log_effect(
            mem,
            TemplateFormatter("hello {name}"),
        )
        pipe = SyncPipeline.builder("tmpl").effect("log", log_fn, idempotent=True).build()
        result = pipe.run(State.empty().update(name="agentic"))
        self.assertTrue(result.ok)
        self.assertEqual(mem.records[0].message, "hello agentic")

    def test_skip_on_replay_via_effect_runtime(self) -> None:
        """Replay should short-circuit effect IO using recorded outcomes."""

        from agentic_fn import MemoryRecorder

        mem = MemoryLogSink()
        recorder = MemoryRecorder()
        log_fn = make_log_effect(mem, message="hello")
        pipe = SyncPipeline.builder("replay_log").effect("log", log_fn, idempotent=True).build()

        record_ctx = Context(
            effects=EffectRuntime(
                mode=ReplayMode.RECORD,
                recorder=recorder,
                replay=recorder,
            )
        )
        recorded = pipe.run(State.empty(), ctx=record_ctx)
        self.assertTrue(recorded.ok)
        self.assertEqual(len(mem.records), 1)

        mem.clear()
        replay_ctx = Context(
            effects=EffectRuntime(
                mode=ReplayMode.REPLAY,
                recorder=recorder,
                replay=recorder,
            )
        )
        replayed = pipe.run(State.empty(), ctx=replay_ctx)
        self.assertTrue(replayed.ok)
        self.assertEqual(len(mem.records), 0)

    def test_emit_log_skips_when_context_in_replay(self) -> None:
        """Direct emit path should no-op when context effect mode is REPLAY."""

        from agentic_fn.effects.log.effect import LogEffectOptions, emit_log
        from agentic_fn.effects.log.formatters import StaticMessageFormatter

        mem = MemoryLogSink()
        ctx = Context(effects=EffectRuntime(mode=ReplayMode.REPLAY))
        out = emit_log(
            State.empty(),
            ctx,
            sink=mem,
            formatter=StaticMessageFormatter("x"),
            options=LogEffectOptions(skip_on_replay=True),
        )
        self.assertEqual(out.data, {})
        self.assertEqual(len(mem.records), 0)

    def test_context_logger_skips_on_replay(self) -> None:
        """Context.log should skip sink IO during REPLAY by default."""

        mem = MemoryLogSink()
        ctx = Context(
            log_sink=mem,
            effects=EffectRuntime(mode=ReplayMode.REPLAY),
        )
        ctx.log.info("silent")
        self.assertEqual(len(mem.records), 0)

    def test_failure_policy_raise_transient(self) -> None:
        """Emit failures can escalate according to policy."""

        def boom(_record: LogRecord) -> None:
            raise RuntimeError("sink down")

        log_fn = make_log_effect(
            CallableLogSink(boom),
            message="x",
            options=LogEffectOptions(failure_policy=EmitFailurePolicy.RAISE_TRANSIENT),
        )
        pipe = SyncPipeline.builder("fail_log").effect("log", log_fn, idempotent=True).build()
        result = pipe.run(State.empty())
        self.assertFalse(result.ok)
        self.assertIsInstance(result.error, TransientError)

    def test_context_logger_facade(self) -> None:
        """Context.log should enrich records with run metadata."""

        mem = MemoryLogSink()
        ctx = Context(log_sink=mem)
        ctx.mark_started()
        ctx.set_path(("pipeline", "step_a"))
        ctx.log.bind(service="demo").info("hello", user_id=1)
        self.assertEqual(len(mem.records), 1)
        record = mem.records[0]
        self.assertEqual(record.message, "hello")
        self.assertEqual(record.fields["service"], "demo")
        self.assertEqual(record.fields["user_id"], 1)
        self.assertEqual(record.run_id, ctx.run_id)
        self.assertEqual(record.path, ("pipeline", "step_a"))

    def test_event_bridge(self) -> None:
        """Event bridges should map pipeline events into log records."""

        mem = MemoryLogSink()
        bridge = EventLogBridge(sink=mem, min_level=LogLevel.DEBUG)
        bridge(
            Event(
                kind=EventKind.STEP_ERROR,
                run_id="r1",
                step_name="action",
                error_type="TransientError",
                path=("react", "action"),
            )
        )
        self.assertEqual(len(mem.records), 1)
        self.assertEqual(mem.records[0].level, LogLevel.ERROR)
        self.assertEqual(mem.records[0].fields["event_kind"], "step.error")


class RedactionPolicyTests(unittest.TestCase):
    """Field redaction policy behavior."""

    def test_nested_redaction(self) -> None:
        """Nested mappings should redact sensitive keys recursively."""

        policy = RedactionPolicy()
        redacted = policy.redact_mapping(
            {
                "user": "a",
                "nested": {"token": "t", "ok": 1},
                "list": [{"password": "p"}, "x"],
            }
        )
        self.assertEqual(redacted["nested"]["token"], "***REDACTED***")
        self.assertEqual(redacted["nested"]["ok"], 1)
        self.assertEqual(redacted["list"][0]["password"], "***REDACTED***")


if __name__ == "__main__":
    unittest.main()
