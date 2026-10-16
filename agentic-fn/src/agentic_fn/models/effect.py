"""Pipeline effect factories that bind ModelRuntime into steps."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agentic_fn.core.context import Context
from agentic_fn.core.fn import effect
from agentic_fn.core.state import State
from agentic_fn.models.codec import (
    get_messages,
    put_decide_response,
    put_messages,
    put_model_response,
)
from agentic_fn.models.message import Message
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.runtime import ModelRuntime
from agentic_fn.type.aliases import EffectFn
from agentic_fn.content.model import Content
from agentic_fn.content import json_ops


def make_model_effect(
    *,
    logical_model: str = "chat.default",
    input_key: str = "messages",
    output_key: str = "model_response",
    append_assistant: bool = True,
    idempotent: bool = False,
    request_builder: Callable[[State, Context], ModelRequest] | None = None,
) -> EffectFn:
    """
    Build an EffectFn that calls ``ctx.models.complete`` via the router.

    Expects ``Context.models`` to be a configured ``ModelRuntime``.
    """

    @effect(idempotent=idempotent, name=f"model:{logical_model}")
    def _fn(state: State, ctx: Context) -> State:
        runtime = _require_runtime(ctx)
        if request_builder is not None:
            request = request_builder(state, ctx)
        else:
            messages = get_messages(state, input_key)
            request = ModelRequest.chat(logical_model, *messages)
        response = runtime.complete(request, ctx)
        state = put_model_response(state, output_key, response)
        if append_assistant:
            messages = list(get_messages(state, input_key))
            messages.append(response.message)
            state = put_messages(state, input_key, messages)
        return state

    return _fn


def make_decide_effect(
    *,
    logical_model: str = "decide.policy",
    features_key: str = "features",
    output_key: str = "decision",
    idempotent: bool = False,
    request_builder: Callable[[State, Context], DecideRequest] | None = None,
) -> EffectFn:
    """Build an EffectFn that calls ``ctx.models.decide``."""

    @effect(idempotent=idempotent, name=f"decide:{logical_model}")
    def _fn(state: State, ctx: Context) -> State:
        runtime = _require_runtime(ctx)
        if request_builder is not None:
            request = request_builder(state, ctx)
        else:
            features = state.get(features_key)
            if isinstance(features, Content):
                content = features
            elif isinstance(features, dict) and features.get("_type"):
                content = Content.from_dict(features)
            else:
                content = json_ops.from_value(features if features is not None else {})
            request = DecideRequest(model=logical_model, features=content)
        response = runtime.decide(request, ctx)
        return put_decide_response(state, output_key, response)

    return _fn


def _require_runtime(ctx: Context) -> ModelRuntime:
    """Fetch ModelRuntime from context or raise."""

    runtime = getattr(ctx, "models", None)
    if runtime is None:
        raise RuntimeError(
            "Context.models is not configured; attach a ModelRuntime before running model effects"
        )
    if not isinstance(runtime, ModelRuntime):
        raise RuntimeError(f"Context.models must be ModelRuntime, got {type(runtime)!r}")
    return runtime
