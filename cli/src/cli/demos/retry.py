"""Demo: idempotent effect retries on transient failures."""

from __future__ import annotations

from agentic_fn import (
    Context,
    RetryPolicy,
    State,
    SyncPipeline,
    TransientError,
    effect,
)

from cli.printing import banner, kv, note, print_result


def run() -> None:
    """
    Show why Effect steps need an ``idempotent`` flag for automatic retries.

    - Transient failures (network blips, rate limits) may be retried.
    - Non-idempotent effects must NOT be auto-retried (risk of double charge,
      duplicate writes, etc.). Mark them ``idempotent=False``.
    """

    banner("demo: retry - TransientError + idempotent effect")
    note("Effect fails twice, succeeds on attempt 3; RetryPolicy.max_attempts=5")

    calls = {"n": 0}

    @effect(idempotent=True)
    def flaky_llm(state: State, ctx: Context) -> State:
        """
        Fake an unreliable external call.

        Real LLM / HTTP adapters would live here and go through EffectRuntime
        so they can be recorded, timed out, and budgeted.
        """

        _ = ctx
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError(f"simulated flake (attempt {calls['n']})")
        return state.update(answer="hello from flaky effect", attempts=calls["n"])

    pipe = (
        SyncPipeline.builder(
            "demo_retry",
            # Pipeline-level default; steps can override per-step.
            retry=RetryPolicy(max_attempts=5),
        )
        .effect(
            "flaky_llm",
            flaky_llm,
            idempotent=True,
            retry=RetryPolicy(max_attempts=5),
        )
        .build()
    )

    result = pipe.run(State.empty().update(question="ping"))
    kv("live_call_count", calls["n"])
    print_result(result)
