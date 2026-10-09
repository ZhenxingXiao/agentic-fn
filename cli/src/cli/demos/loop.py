"""Demo: loop composition until a predicate is satisfied."""

from __future__ import annotations

from agentic_fn import Context, State, SyncPipeline, capability

from cli.printing import banner, note, print_result


def run() -> None:
    """
    Demonstrate a bounded Loop - the building block behind ReAct-style agents.

    The body runs while ``until(state, ctx)`` is false, up to ``max_iter``.
    Termination is driven by *state*, not by hidden mutable counters outside
    the pipeline.
    """

    banner("demo: loop - until predicate / max_iter")
    note("Bump ``round`` each iteration; stop when round >= 3 (final=True)")

    @capability
    def bump(state: State, ctx: Context) -> State:
        """Advance the loop counter and set the termination flag."""

        _ = ctx
        round_ = int(state.get("round", 0)) + 1
        return state.update(round=round_, final=round_ >= 3)

    pipe = (
        SyncPipeline.builder("demo_loop")
        .loop(
            "react_like",
            # Nested body builder: same fluency as the top-level builder.
            body=lambda b: b.capability("bump", bump),
            until=lambda s, _c: bool(s.get("final")),
            max_iter=10,  # hard ceiling - always protect agent loops
        )
        .build()
    )

    result = pipe.run(State.empty().update(round=0, final=False))
    print_result(result)
