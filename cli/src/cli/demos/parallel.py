"""Demo: parallel fan-out with keyed merge."""

from __future__ import annotations

from agentic_fn import Context, MergeByKeys, State, SyncPipeline, capability

from cli.printing import banner, note, print_result


def run() -> None:
    """
    Run independent branches, then merge only the keys each branch owns.

    ``MergeByKeys`` avoids accidental overwrites: branch ``left`` contributes
    ``a``, branch ``right`` contributes ``b``. Other merge strategies exist
    (``MergeAll``, ``ReduceMerge``) for different conflict policies.
    """

    banner("demo: parallel - fan-out + MergeByKeys")
    note("left writes a=1, right writes b=2; merge keeps both keys")

    @capability
    def left(state: State, ctx: Context) -> State:
        _ = ctx
        return state.update(a=1)

    @capability
    def right(state: State, ctx: Context) -> State:
        _ = ctx
        return state.update(b=2)

    pipe = (
        SyncPipeline.builder("demo_parallel")
        .parallel(
            "fan",
            branches={
                # Each branch is a mini sequential body.
                "left": lambda b: b.capability("left", left),
                "right": lambda b: b.capability("right", right),
            },
            merge=MergeByKeys({"left": ["a"], "right": ["b"]}),
        )
        .build()
    )

    result = pipe.run(State.empty())
    print_result(result)
