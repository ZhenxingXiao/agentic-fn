"""
Demo: exercise agentic-fn log *effects* backed by picologging.

This is the primary place to see how side-effect logging plugs into a pipeline:

1. Configure picologging (vendor / transport layer)
2. Wrap it as a ``LogSink`` (framework boundary)
3. Bind the sink with ``make_log_effect`` / ``Context.log`` (pipeline / effect API)
4. Optionally multiplex into ``MemoryLogSink`` for post-run assertions
"""

from __future__ import annotations

from pathlib import Path

from agentic_fn import (
    Context,
    LogEffectOptions,
    LogLevel,
    MemoryLogSink,
    State,
    SyncPipeline,
    capability,
    effect,
    make_log_effect,
)
from agentic_fn.effects.log.formatters import StateKeysFormatter, TemplateFormatter
from agentic_fn.effects.log.sinks import MultiplexLogSink

from cli.printing import banner, kv, note, print_result
from cli.sinks import PicologgingLogSink, configure_picologging

# Project-local log file (cli/logs/). Rotated by configure_picologging.
_LOG_FILE = Path(__file__).resolve().parents[3] / "logs" / "log-effect.log"


def run() -> None:
    """
    Run a small pipeline that logs twice:

    - as an explicit pipeline Effect step (``make_log_effect``)
    - from inside a custom Effect via ``ctx.log`` (facade over the same sink)
    """

    banner("demo: log-effect - picologging-backed LogSink")
    note("Side-effect logs go through LogSink; picologging is only the backend")

    # ------------------------------------------------------------------
    # 1) Vendor setup (picologging)
    #    Console stays on stderr. File policy lives here, not on getLogger:
    #    RotatingFileHandler rolls at 1 MiB and keeps 3 backups.
    # ------------------------------------------------------------------
    log_file = configure_picologging(log_file=_LOG_FILE, max_bytes=1_048_576, backup_count=3)
    note(f"file log: {log_file}")

    # ------------------------------------------------------------------
    # 2) Framework boundary: LogSink adapters
    # ------------------------------------------------------------------
    # Memory sink: inspect records after the run (great for tests / harnesses).
    memory = MemoryLogSink()
    # Picologging sink: print to stderr through the high-performance logger.
    pico = PicologgingLogSink(logger_name="agentic_fn.cli.demo")
    # Fan-out so one emit hits both sinks. Failures are isolated by default.
    sink = MultiplexLogSink(sinks=(pico, memory), fail_fast=False)

    # ------------------------------------------------------------------
    # 3) Pure capability + side-effect steps
    # ------------------------------------------------------------------
    @capability
    def greet(state: State, ctx: Context) -> State:
        """Capability: pure-ish state transform (no logging here on purpose)."""

        _ = ctx
        name = str(state.get("name", "world"))
        return state.update(greeting=f"hello, {name}")

    # Pipeline-bound log effect: format selected state keys, then emit.
    log_state = make_log_effect(
        sink,
        StateKeysFormatter(
            message="state snapshot after greet",
            keys=("name", "greeting"),
            level=LogLevel.INFO,
            logger_name="agentic_fn.cli.demo",
        ),
        options=LogEffectOptions(
            # Persist a tiny summary back into State for later inspection.
            write_summary_key="_last_log",
            extra_fields={"demo": "log-effect", "phase": "after_greet"},
        ),
    )

    @effect(idempotent=True)
    def annotate(state: State, ctx: Context) -> State:
        """
        Effect: may perform IO.

        ``ctx.log`` is the ergonomic facade - still routed through the same
        ``LogSink``, so picologging (or any other backend) stays swappable.
        """

        ctx.log.bind(component="annotate").info(
            "annotating greeting inside an effect",
            greeting=state.get("greeting"),
        )
        return state.update(annotated=True)

    # A second bound effect using a template formatter.
    log_done = make_log_effect(
        sink,
        TemplateFormatter(
            "pipeline finished for {name}",
            level=LogLevel.INFO,
            logger_name="agentic_fn.cli.demo",
        ),
    )

    # ------------------------------------------------------------------
    # 4) Assemble and run
    # ------------------------------------------------------------------
    # Inject the sink on Context so ``ctx.log`` in annotate() uses picologging.
    ctx = Context(log_sink=sink, log_logger_name="agentic_fn.cli.demo")

    pipe = (
        SyncPipeline.builder("demo_log_effect")
        .capability("greet", greet)
        .effect("log_state", log_state, idempotent=True)
        .effect("annotate", annotate, idempotent=True)
        .effect("log_done", log_done, idempotent=True)
        .build()
    )

    result = pipe.run(State.empty().update(name="agentic-fn"), ctx=ctx)

    # ------------------------------------------------------------------
    # 5) Post-run inspection via MemoryLogSink
    # ------------------------------------------------------------------
    print("  -- pipeline result --")
    print_result(result)
    kv("memory_sink.records", len(memory.records))
    print("  memory_sink messages:")
    for idx, record in enumerate(memory.records, start=1):
        print(
            f"    {idx}. [{record.level.value}] {record.message!r} "
            f"fields={dict(record.fields)}"
        )
