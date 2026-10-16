"""State helpers for model requests / responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentic_fn.core.state import State
from agentic_fn.models.errors import ModelConfigurationError
from agentic_fn.models.kinds import DECIDE_TYPE_MARKER, MODEL_TYPE_MARKER
from agentic_fn.models.message import Message
from agentic_fn.models.response import DecideResponse, ModelResponse


def put_messages(state: State, key: str, messages: Sequence[Message]) -> State:
    """Store messages as serializable mappings."""

    return state.update(**{key: [m.to_dict() for m in messages]})


def get_messages(state: State, key: str) -> list[Message]:
    """Load messages previously stored via ``put_messages``."""

    raw = state.get(key)
    if raw is None:
        raise KeyError(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ModelConfigurationError(f"state[{key!r}] must be a list of messages")
    return [Message.from_dict(item) if isinstance(item, Mapping) else item for item in raw]


def put_model_response(state: State, key: str, response: ModelResponse) -> State:
    """Store a normalized model response."""

    return state.update(**{key: response.to_dict()})


def get_model_response(state: State, key: str) -> ModelResponse:
    """Load a model response mapping."""

    raw = state.get(key)
    if raw is None:
        raise KeyError(key)
    if isinstance(raw, ModelResponse):
        return raw
    if not isinstance(raw, Mapping) or raw.get("_type") not in {MODEL_TYPE_MARKER, None}:
        # Allow missing _type for forward compat if required fields exist.
        if not isinstance(raw, Mapping):
            raise ModelConfigurationError(f"state[{key!r}] is not a model response")
    return ModelResponse.from_dict(raw)


def put_decide_response(state: State, key: str, response: DecideResponse) -> State:
    """Store a decide response."""

    return state.update(**{key: response.to_dict()})


def get_decide_response(state: State, key: str) -> DecideResponse:
    """Load a decide response."""

    raw = state.get(key)
    if raw is None:
        raise KeyError(key)
    if isinstance(raw, DecideResponse):
        return raw
    if not isinstance(raw, Mapping):
        raise ModelConfigurationError(f"state[{key!r}] is not a decide response")
    if raw.get("_type") not in {DECIDE_TYPE_MARKER, None}:
        pass
    return DecideResponse.from_dict(raw)


def response_to_assistant_message(response: ModelResponse | Mapping[str, Any]) -> Message:
    """Extract the assistant message for conversation appending."""

    if isinstance(response, ModelResponse):
        return response.message
    return ModelResponse.from_dict(response).message
