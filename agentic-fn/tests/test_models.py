"""Production tests for model IO: router, runtime, effects, safety, multimodal."""

from __future__ import annotations

import unittest

from agentic_fn import (
    Context,
    FakeClient,
    Message,
    ModelRequest,
    State,
    SyncPipeline,
    build_model_runtime,
    demo_runtime,
    make_decide_effect,
    make_model_effect,
    put_messages,
)
from agentic_fn.models.message import MediaRef
from agentic_fn.content import json_ops
from agentic_fn.models.errors import (
    ModelExhaustedError,
    ModelRouteError,
    ModelSafetyError,
    ModelTransientError,
)
from agentic_fn.models.kinds import (
    CAP_CHAT,
    CAP_DECIDE,
    CAP_TOOLS,
    CAP_VISION,
    Modality,
    RouteStrategy,
)
from agentic_fn.models.request import DecideRequest, RoutingHints
from agentic_fn.models.router import (
    Candidate,
    DefaultModelRouter,
    FailoverPolicy,
    RouteEntry,
    RouteTable,
)
from agentic_fn.models.safety import DenySubstringFilter
from agentic_fn.runtime.budget import Budget


class MessageTests(unittest.TestCase):
    """Message / multimodal helpers."""

    def test_user_multimodal_requires_vision_cap(self) -> None:
        media = MediaRef(mime="image/png", uri="https://example.com/a.png")
        msg = Message.user_multimodal("describe", media)
        self.assertTrue(msg.has_vision())
        req = ModelRequest.chat("vision.ocr", msg, modality=Modality.MULTIMODAL)
        self.assertIn(CAP_VISION, req.required_capabilities())


class RouterTests(unittest.TestCase):
    """Route strategies and filtering."""

    def _table(self) -> RouteTable:
        return RouteTable(
            entries={
                "chat.default": RouteEntry(
                    modality=Modality.CHAT,
                    strategy=RouteStrategy.PRIORITY,
                    candidates=(
                        Candidate(
                            provider="a",
                            physical_model="m1",
                            priority=10,
                            capabilities=frozenset({CAP_CHAT, CAP_TOOLS}),
                        ),
                        Candidate(
                            provider="b",
                            physical_model="m2",
                            priority=20,
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                    ),
                    failover=FailoverPolicy(max_attempts=2),
                ),
                "chat.weighted": RouteEntry(
                    modality=Modality.CHAT,
                    strategy=RouteStrategy.WEIGHTED,
                    candidates=(
                        Candidate(
                            provider="a",
                            physical_model="m1",
                            weight=0.0,
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                        Candidate(
                            provider="b",
                            physical_model="m2",
                            weight=1.0,
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                    ),
                ),
                "chat.sticky": RouteEntry(
                    modality=Modality.CHAT,
                    strategy=RouteStrategy.STICKY,
                    candidates=(
                        Candidate(
                            provider="a",
                            physical_model="m1",
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                        Candidate(
                            provider="b",
                            physical_model="m2",
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                    ),
                ),
                "decide.policy": RouteEntry(
                    modality=Modality.DECIDE,
                    strategy=RouteStrategy.PRIORITY,
                    candidates=(
                        Candidate(
                            provider="laya",
                            physical_model="p1",
                            capabilities=frozenset({CAP_DECIDE}),
                        ),
                    ),
                ),
            }
        )

    def test_priority_and_capability_filter(self) -> None:
        router = DefaultModelRouter(table=self._table(), known_providers=("a", "b", "laya"))
        ctx = Context()
        decision = router.resolve(ModelRequest.chat("chat.default", Message.user("hi")), ctx)
        self.assertEqual(decision.attempts[0].provider, "a")
        self.assertEqual(decision.attempts[0].physical_model, "m1")

    def test_missing_vision_filters_candidate(self) -> None:
        router = DefaultModelRouter(table=self._table(), known_providers=("a", "b", "laya"))
        media = MediaRef(mime="image/png", uri="https://x/y.png")
        req = ModelRequest.chat(
            "chat.default",
            Message.user_multimodal("see", media),
            modality=Modality.MULTIMODAL,
        )
        with self.assertRaises(ModelRouteError):
            router.resolve(req, Context())

    def test_sticky_stable(self) -> None:
        router = DefaultModelRouter(table=self._table(), known_providers=("a", "b", "laya"))
        req = ModelRequest.chat(
            "chat.sticky",
            Message.user("hi"),
            routing=RoutingHints(sticky_key="user-42"),
        )
        d1 = router.resolve(req, Context())
        d2 = router.resolve(req, Context())
        self.assertEqual(d1.attempts[0].provider, d2.attempts[0].provider)

    def test_hot_reload(self) -> None:
        router = DefaultModelRouter(table=self._table(), known_providers=("a", "b", "laya"))
        new_table = RouteTable(
            entries={
                "chat.default": RouteEntry(
                    modality=Modality.CHAT,
                    candidates=(
                        Candidate(
                            provider="b",
                            physical_model="only",
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                    ),
                )
            }
        )
        router.reload(new_table)
        decision = router.resolve(ModelRequest.chat("chat.default", Message.user("x")), Context())
        self.assertEqual(decision.attempts[0].provider, "b")


class RuntimeTests(unittest.TestCase):
    """ModelRuntime failover, budget, safety, decide."""

    def test_complete_and_budget(self) -> None:
        runtime = demo_runtime(fake_reply="pong")
        ctx = Context(models=runtime, budget=Budget(max_calls=2, max_tokens=100))
        resp = runtime.complete(ModelRequest.chat("chat.default", Message.user("ping")), ctx)
        self.assertIn("pong", resp.message.text())
        self.assertEqual(ctx.budget.view().remaining_calls, 1)
        self.assertIsNotNone(resp.route.selected_provider)

    def test_failover_on_transient(self) -> None:
        failing = FakeClient(provider="bad")
        failing.reply_fn = lambda _r: (_ for _ in ()).throw(ModelTransientError("boom"))
        ok = FakeClient(provider="good", default_reply="recovered")
        table = RouteTable(
            entries={
                "chat.default": RouteEntry(
                    modality=Modality.CHAT,
                    failover=FailoverPolicy(max_attempts=2),
                    candidates=(
                        Candidate(
                            provider="bad",
                            physical_model="x",
                            priority=1,
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                        Candidate(
                            provider="good",
                            physical_model="y",
                            priority=2,
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                    ),
                )
            }
        )
        runtime = build_model_runtime(
            table=table,
            clients={"bad": failing, "good": ok},
        )
        resp = runtime.complete(
            ModelRequest.chat("chat.default", Message.user("hi")),
            Context(models=runtime),
        )
        self.assertEqual(resp.provider, "good")
        self.assertIn("recovered", resp.message.text())

    def test_exhausted_when_all_fail(self) -> None:
        bad = FakeClient(provider="bad")
        bad.reply_fn = lambda _r: (_ for _ in ()).throw(ModelTransientError("x"))
        table = RouteTable(
            entries={
                "chat.default": RouteEntry(
                    modality=Modality.CHAT,
                    candidates=(
                        Candidate(
                            provider="bad",
                            physical_model="x",
                            capabilities=frozenset({CAP_CHAT}),
                        ),
                    ),
                )
            }
        )
        runtime = build_model_runtime(table=table, clients={"bad": bad})
        with self.assertRaises(ModelExhaustedError):
            runtime.complete(
                ModelRequest.chat("chat.default", Message.user("hi")),
                Context(models=runtime),
            )

    def test_safety_filter(self) -> None:
        runtime = demo_runtime(fake_reply="contains SECRET_TOKEN here")
        runtime.output_filter = DenySubstringFilter(denied=("SECRET_TOKEN",))
        with self.assertRaises(ModelSafetyError):
            runtime.complete(
                ModelRequest.chat("chat.default", Message.user("x")),
                Context(models=runtime),
            )

    def test_decide(self) -> None:
        runtime = demo_runtime()
        resp = runtime.decide(
            DecideRequest(
                model="decide.policy",
                features=json_ops.from_value({"risk": 0.1}),
            ),
            Context(models=runtime),
        )
        self.assertEqual(resp.action, "allow")
        self.assertGreaterEqual(resp.confidence or 0, 0.0)

    def test_stream(self) -> None:
        runtime = demo_runtime()
        chunks = list(
            runtime.stream(
                ModelRequest.chat("chat.default", Message.user("hi")),
                Context(models=runtime),
            )
        )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(chunks[-1].message.text())


class EffectIntegrationTests(unittest.TestCase):
    """Pipeline effects using Context.models."""

    def test_make_model_effect(self) -> None:
        runtime = demo_runtime(fake_reply="effect-ok")
        pipe = (
            SyncPipeline.builder("model_fx")
            .effect("chat", make_model_effect(logical_model="chat.default"), idempotent=False)
            .build()
        )
        state = put_messages(State.empty(), "messages", [Message.user("hello")])
        result = pipe.run(state, ctx=Context(models=runtime))
        self.assertTrue(result.ok)
        self.assertIn("model_response", result.state.data)
        self.assertEqual(len(result.state.get("messages")), 2)

    def test_make_decide_effect(self) -> None:
        runtime = demo_runtime()
        pipe = (
            SyncPipeline.builder("decide_fx")
            .effect("decide", make_decide_effect(logical_model="decide.policy"), idempotent=True)
            .build()
        )
        state = State.empty().update(features={"score": 1})
        result = pipe.run(state, ctx=Context(models=runtime))
        self.assertTrue(result.ok)
        self.assertIn("decision", result.state.data)


if __name__ == "__main__":
    unittest.main()
