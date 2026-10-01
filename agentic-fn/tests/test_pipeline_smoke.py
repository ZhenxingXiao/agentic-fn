"""Smoke and behavioral tests for the pipeline core."""

from __future__ import annotations

import asyncio
import unittest

from agentic_fn import (
    AsyncPipeline,
    Context,
    EffectRuntime,
    ErrorAction,
    MemoryRecorder,
    MergeByKeys,
    PermanentError,
    ReplayMode,
    RetryPolicy,
    RunStatus,
    State,
    SyncPipeline,
    TransientError,
    capability,
    effect,
)


class SyncPipelineTests(unittest.TestCase):
    """Synchronous pipeline behavior."""

    def test_seq_capabilities(self) -> None:
        """Capability steps should transform state in order."""

        @capability
        def add_one(state: State, ctx: Context) -> State:
            _ = ctx
            return state.update(value=int(state.get("value", 0)) + 1)

        @capability
        def double(state: State, ctx: Context) -> State:
            _ = ctx
            return state.update(value=int(state.get("value", 0)) * 2)

        pipe = (
            SyncPipeline.builder("demo")
            .capability("add", add_one)
            .capability("double", double)
            .build()
        )
        result = pipe.run(State.empty().update(value=3))
        self.assertTrue(result.ok)
        self.assertEqual(result.state.get("value"), 8)

    def test_loop_until_final(self) -> None:
        """Loop should stop when until predicate becomes true."""

        @capability
        def bump(state: State, ctx: Context) -> State:
            _ = ctx
            round_ = int(state.get("round", 0)) + 1
            return state.update(round=round_, final=round_ >= 3)

        pipe = (
            SyncPipeline.builder("react")
            .loop(
                "react_loop",
                body=lambda b: b.capability("bump", bump),
                until=lambda s, _c: bool(s.get("final")),
                max_iter=10,
            )
            .build()
        )
        result = pipe.run(State.empty().update(round=0, final=False))
        self.assertTrue(result.ok)
        self.assertEqual(result.state.get("round"), 3)

    def test_effect_retry_requires_idempotent(self) -> None:
        """Non-idempotent effects must not auto-retry."""

        calls = {"n": 0}

        @effect(idempotent=False)
        def flaky(state: State, ctx: Context) -> State:
            _ = ctx
            calls["n"] += 1
            raise TransientError("boom")

        pipe = (
            SyncPipeline.builder(
                "fx",
                retry=RetryPolicy(max_attempts=3),
            )
            .effect("flaky", flaky, idempotent=False)
            .build()
        )
        result = pipe.run(State.empty())
        self.assertFalse(result.ok)
        self.assertEqual(calls["n"], 1)

    def test_effect_retry_when_idempotent(self) -> None:
        """Idempotent effects may retry transient failures."""

        calls = {"n": 0}

        @effect(idempotent=True)
        def flaky(state: State, ctx: Context) -> State:
            _ = ctx
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientError("boom")
            return state.update(ok=True)

        pipe = (
            SyncPipeline.builder(
                "fx",
                retry=RetryPolicy(max_attempts=5),
            )
            .effect("flaky", flaky, idempotent=True, retry=RetryPolicy(max_attempts=5))
            .build()
        )
        result = pipe.run(State.empty())
        self.assertTrue(result.ok)
        self.assertEqual(calls["n"], 3)
        self.assertTrue(result.state.get("ok"))

    def test_continue_error_action(self) -> None:
        """CONTINUE should record soft errors and keep going."""

        @capability
        def bad(state: State, ctx: Context) -> State:
            _ = state, ctx
            raise PermanentError("nope")

        @capability
        def good(state: State, ctx: Context) -> State:
            _ = ctx
            return state.update(done=True)

        pipe = (
            SyncPipeline.builder("soft")
            .capability("bad", bad, on_error=ErrorAction.CONTINUE)
            .capability("good", good)
            .build()
        )
        result = pipe.run(State.empty())
        self.assertTrue(result.ok)
        self.assertTrue(result.state.get("done"))
        self.assertTrue(result.state.get("_errors"))

    def test_parallel_merge_by_keys(self) -> None:
        """Parallel branches should merge selected keys."""

        @capability
        def left(state: State, ctx: Context) -> State:
            _ = ctx
            return state.update(a=1)

        @capability
        def right(state: State, ctx: Context) -> State:
            _ = ctx
            return state.update(b=2)

        pipe = (
            SyncPipeline.builder("par")
            .parallel(
                "fan",
                branches={
                    "left": lambda b: b.capability("left", left),
                    "right": lambda b: b.capability("right", right),
                },
                merge=MergeByKeys({"left": ["a"], "right": ["b"]}),
            )
            .build()
        )
        result = pipe.run(State.empty())
        self.assertTrue(result.ok)
        self.assertEqual(result.state.get("a"), 1)
        self.assertEqual(result.state.get("b"), 2)

    def test_replay_effect(self) -> None:
        """REPLAY mode should return recorded effect outputs."""

        recorder = MemoryRecorder()

        @effect(idempotent=True)
        def llm(state: State, ctx: Context) -> State:
            _ = ctx
            return state.update(answer="live")

        live_ctx = Context(
            effects=EffectRuntime(mode=ReplayMode.RECORD, recorder=recorder, replay=recorder)
        )
        live_pipe = SyncPipeline.builder("live").effect("llm", llm, idempotent=True).build()
        live_state = State.empty().update(q="hi")
        live_result = live_pipe.run(live_state, ctx=live_ctx)
        self.assertTrue(live_result.ok)

        @effect(idempotent=True)
        def llm_should_not_run(state: State, ctx: Context) -> State:
            _ = state, ctx
            raise AssertionError("effect should not execute in replay")

        replay_ctx = Context(
            effects=EffectRuntime(mode=ReplayMode.REPLAY, recorder=recorder, replay=recorder)
        )
        replay_pipe = (
            SyncPipeline.builder("replay")
            .effect("llm", llm_should_not_run, idempotent=True)
            .build()
        )
        replay_result = replay_pipe.run(State.empty().update(q="hi"), ctx=replay_ctx)
        self.assertTrue(replay_result.ok)
        self.assertEqual(replay_result.state.get("answer"), "live")


class AsyncPipelineTests(unittest.TestCase):
    """Asynchronous pipeline behavior."""

    def test_async_seq(self) -> None:
        """Async capabilities should await in order."""

        @capability
        async def add(state: State, ctx: Context) -> State:
            _ = ctx
            await asyncio.sleep(0)
            return state.update(value=int(state.get("value", 0)) + 2)

        pipe = AsyncPipeline.builder("async_demo").capability("add", add).build()
        result = asyncio.run(pipe.run(State.empty().update(value=1)))
        self.assertEqual(result.status, RunStatus.SUCCEEDED)
        self.assertEqual(result.state.get("value"), 3)


if __name__ == "__main__":
    unittest.main()
