"""Demo: AsyncPipeline with awaitable capabilities."""

from __future__ import annotations

import asyncio

from agentic_fn import AsyncPipeline, Context, State, capability

from cli.printing import banner, note, print_result


def run() -> None:
    """
    Run a tiny async pipeline (IO-friendly orchestration).

    Use AsyncPipeline when steps await network / disk. Sync and async pipelines
    share the same composition vocabulary (seq / parallel / branch / loop) but
    must not be mixed carelessly inside one graph.
    """

    banner("demo: async-seq - AsyncPipeline")
    note("Awaitable capability adds 2 to value (1 -> 3)")

    @capability
    async def add_two(state: State, ctx: Context) -> State:
        """Simulate an async unit of work with a zero-duration yield."""

        _ = ctx
        await asyncio.sleep(0)
        return state.update(value=int(state.get("value", 0)) + 2)

    pipe = AsyncPipeline.builder("demo_async_seq").capability("add_two", add_two).build()

    # The CLI itself is sync; bridge with asyncio.run for this demo.
    result = asyncio.run(pipe.run(State.empty().update(value=1)))
    print_result(result)
