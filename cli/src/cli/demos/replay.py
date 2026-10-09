"""Demo: record an effect, then replay without re-executing it."""

from __future__ import annotations

from agentic_fn import (
    Context,
    EffectRuntime,
    MemoryRecorder,
    ReplayMode,
    State,
    SyncPipeline,
    effect,
)

from cli.printing import banner, note, print_result


def run() -> None:
    """
    Two-phase demo: RECORD live output, then REPLAY from the recorder.

    This is the foundation for offline debugging and golden-path evaluation -
    expensive side effects (LLM calls) are not re-invoked during replay.
    """

    banner("demo: replay - RECORD then REPLAY")
    note("Live effect returns answer='live'; replay must not call the effect again")

    recorder = MemoryRecorder()

    # ------------------------------------------------------------------
    # Phase 1: live run under RECORD mode
    # ------------------------------------------------------------------
    @effect(idempotent=True)
    def llm_live(state: State, ctx: Context) -> State:
        _ = ctx
        return state.update(answer="live", question=state.get("question"))

    live_ctx = Context(
        effects=EffectRuntime(
            mode=ReplayMode.RECORD,
            recorder=recorder,
            replay=recorder,
        )
    )
    live_pipe = (
        SyncPipeline.builder("demo_replay_live")
        .effect("llm", llm_live, idempotent=True)
        .build()
    )
    live_result = live_pipe.run(
        State.empty().update(question="hi"),
        ctx=live_ctx,
    )
    print("  -- live (RECORD) --")
    print_result(live_result)

    # ------------------------------------------------------------------
    # Phase 2: replay; a trap effect proves the body does not execute
    # ------------------------------------------------------------------
    @effect(idempotent=True)
    def llm_must_not_run(state: State, ctx: Context) -> State:
        _ = state, ctx
        raise AssertionError("effect body must not execute in REPLAY mode")

    replay_ctx = Context(
        effects=EffectRuntime(
            mode=ReplayMode.REPLAY,
            recorder=recorder,
            replay=recorder,
        )
    )
    replay_pipe = (
        SyncPipeline.builder("demo_replay_offline")
        .effect("llm", llm_must_not_run, idempotent=True)
        .build()
    )
    replay_result = replay_pipe.run(
        State.empty().update(question="hi"),
        ctx=replay_ctx,
    )
    print("  -- offline (REPLAY) --")
    print_result(replay_result)
