"""Demo: soft-fail a step and continue the pipeline."""

from __future__ import annotations

from agentic_fn import (
    Context,
    ErrorAction,
    PermanentError,
    State,
    SyncPipeline,
    capability,
)

from cli.printing import banner, note, print_result


def run() -> None:
    """
    Use ``on_error=ErrorAction.CONTINUE`` for non-fatal steps.

    The failing step is recorded (including soft errors under ``_errors`` /
    the framework error bucket), then later steps still run. Use FAIL (default)
    when the step is required for correctness.
    """

    banner("demo: soft-error - ErrorAction.CONTINUE")
    note("``bad`` raises PermanentError but the pipeline continues to ``good``")

    @capability
    def bad(state: State, ctx: Context) -> State:
        _ = state, ctx
        raise PermanentError("optional enrichment failed")

    @capability
    def good(state: State, ctx: Context) -> State:
        _ = ctx
        return state.update(done=True)

    pipe = (
        SyncPipeline.builder("demo_soft_error")
        .capability("bad", bad, on_error=ErrorAction.CONTINUE)
        .capability("good", good)
        .build()
    )

    result = pipe.run(State.empty())
    print_result(result)
