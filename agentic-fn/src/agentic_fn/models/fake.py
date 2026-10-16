"""In-memory FakeClient for tests, harnesses, and offline demos."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.core.context import Context
from agentic_fn.models.errors import ModelCapabilityError
from agentic_fn.models.kinds import (
    CAP_CHAT,
    CAP_DECIDE,
    CAP_JSON_SCHEMA,
    CAP_STREAM,
    CAP_TOOLS,
    CAP_VISION,
    FinishReason,
)
from agentic_fn.models.message import Message
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.response import DecideResponse, ModelResponse, RouteTrace, Usage
from agentic_fn.content import json_ops


@dataclass(slots=True)
class FakeClient:
    """
    Deterministic model client used by unit tests and REPLAY-adjacent harnesses.

    Configure canned replies via ``replies`` (request digest -> text) or
    ``reply_fn``. Decide calls return ``decide_fn`` / default action.
    """

    provider: str = "fake"
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {CAP_CHAT, CAP_TOOLS, CAP_VISION, CAP_DECIDE, CAP_JSON_SCHEMA, CAP_STREAM}
        )
    )
    default_reply: str = "ok"
    replies: Mapping[str, str] = field(default_factory=dict)
    reply_fn: Callable[[ModelRequest], str] | None = None
    decide_fn: Callable[[DecideRequest], DecideResponse] | None = None
    default_action: str = "allow"
    calls: list[ModelRequest | DecideRequest] = field(default_factory=list)
    stream_chunks: tuple[str, ...] = ("ok",)
    input_tokens: int = 10
    output_tokens: int = 5
    cost: float = 0.0

    def complete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Return a canned chat completion."""

        _ = ctx
        if CAP_CHAT not in self.capabilities and CAP_VISION not in self.capabilities:
            raise ModelCapabilityError(f"{self.provider} cannot complete")
        self.calls.append(request)
        text = self._resolve_text(request)
        physical = request.physical_model or "fake-model"
        return ModelResponse(
            logical_model=request.model,
            provider=self.provider,
            physical_model=physical,
            message=Message.assistant(text),
            finish_reason=FinishReason.STOP,
            usage=Usage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.input_tokens + self.output_tokens,
                cost=self.cost,
            ),
            route=RouteTrace(
                logical_model=request.model,
                strategy="fake",
                selected_provider=self.provider,
                selected_physical_model=physical,
            ),
            request_digest=request.digest(),
        )

    async def acomplete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Async complete."""

        return self.complete(request, ctx)

    def decide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Return a canned decide response."""

        _ = ctx
        if CAP_DECIDE not in self.capabilities:
            raise ModelCapabilityError(f"{self.provider} cannot decide")
        self.calls.append(request)
        if self.decide_fn is not None:
            return self.decide_fn(request)
        physical = request.physical_model or "fake-decide"
        return DecideResponse(
            logical_model=request.model,
            provider=self.provider,
            physical_model=physical,
            action=self.default_action,
            scores={"allow": 0.9, "deny": 0.1},
            confidence=0.9,
            explanation="fake decide",
            payload=json_ops.from_value({"action": self.default_action}),
            usage=Usage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.input_tokens + self.output_tokens,
                cost=self.cost,
            ),
            route=RouteTrace(
                logical_model=request.model,
                strategy="fake",
                selected_provider=self.provider,
                selected_physical_model=physical,
            ),
            request_digest=request.digest(),
        )

    async def adecide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Async decide."""

        return self.decide(request, ctx)

    def stream(self, request: ModelRequest, ctx: Context) -> Iterator[ModelResponse]:
        """Yield partial then final responses."""

        if CAP_STREAM not in self.capabilities:
            raise ModelCapabilityError(f"{self.provider} cannot stream")
        physical = request.physical_model or "fake-model"
        acc = ""
        for chunk in self.stream_chunks:
            acc += chunk
            yield ModelResponse(
                logical_model=request.model,
                provider=self.provider,
                physical_model=physical,
                message=Message.assistant(acc),
                finish_reason=FinishReason.UNKNOWN,
                usage=Usage.unknown_usage(),
                request_digest=request.digest(),
                labels={"partial": "true"},
            )
        final = self.complete(request, ctx)
        yield final

    def _resolve_text(self, request: ModelRequest) -> str:
        """Pick a reply string."""

        if self.reply_fn is not None:
            return self.reply_fn(request)
        digest = request.digest()
        if digest in self.replies:
            return self.replies[digest]
        # Fall back to last user text echo for demos.
        for message in reversed(request.messages):
            text = message.text()
            if text:
                return f"{self.default_reply}:{text}"
        return self.default_reply
