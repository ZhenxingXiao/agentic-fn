"""
Demo registry for the agentic-fn CLI playground.

Each demo is a zero-argument callable that prints its own explanation and
result. Keep demos self-contained so they can be read as miniature tutorials.
"""

from __future__ import annotations

from collections.abc import Callable

from cli.demos import async_seq, log_effect, loop, parallel, replay, retry, seq, soft_error

# name -> (callable, one-line help shown in ``cli --help`` / ``cli list``)
DEMOS: dict[str, tuple[Callable[[], None], str]] = {
    "seq": (seq.run, "Sequential capabilities transforming State in order"),
    "loop": (loop.run, "Loop until a predicate becomes true (ReAct-style)"),
    "parallel": (parallel.run, "Fan-out branches and merge selected keys"),
    "retry": (retry.run, "Idempotent effect retries on TransientError"),
    "soft-error": (soft_error.run, "CONTINUE on error: soft-fail then keep going"),
    "replay": (replay.run, "Record an effect once, then replay without re-running"),
    "async-seq": (async_seq.run, "AsyncPipeline with awaitable capability steps"),
    "log-effect": (
        log_effect.run,
        "Log effects via picologging LogSink (make_log_effect + ctx.log)",
    ),
}


def names() -> list[str]:
    """Return demo names in a stable display order."""

    return list(DEMOS.keys())


def run(name: str) -> None:
    """Execute a single demo by name; raises KeyError if unknown."""

    fn, _help = DEMOS[name]
    fn()


def run_all() -> None:
    """Execute every registered demo in order."""

    for name in names():
        run(name)
